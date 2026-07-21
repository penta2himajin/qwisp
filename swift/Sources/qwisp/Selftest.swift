import Foundation
import QwispCore

/// Fake backend for GPU-free completion-core tests: yields a canned token script,
/// honoring stopTokens (stops before emitting) + maxTokens exactly as the real backend must.
final class FakeBackend: LLMBackend {
    let script: [Int]
    init(modelDir: String, tier: SeedlessTier) throws { self.script = [] }   // unused in tests
    init(script: [Int]) { self.script = script }
    func generate(_ prompt: [Int], options: GenerateOptions) -> AsyncStream<Int> {
        let script = self.script
        return AsyncStream { cont in
            var n = 0
            for id in script {
                if options.stopTokens.contains(id) { break }
                if options.maxTokens >= 0 && n >= options.maxTokens { break }  // <0 = until EOS/context
                cont.yield(id); n += 1
            }
            cont.finish()
        }
    }
}

/// Completion-core self-test (GPU-free): real tokenizer for decode + FakeBackend.
func runCompletionSelftest(modelDir: String) async -> String {
    var passed = 0, total = 0
    var lines: [String] = []
    func check(_ name: String, _ ok: Bool) {
        total += 1
        lines.append("[comp-test] \(name): \(ok ? "PASS" : "FAIL")")
        if ok { passed += 1 }
    }
    let tok: QwispTokenizer
    do { tok = try await QwispTokenizer(modelDir: modelDir) }
    catch { return "[comp-test] load: FAIL(\(error))\nCOMPTEST 0/1" }
    let decode: ([Int]) -> String = { tok.decode($0) }

    // 1. basic: script decodes back to its text; finish=stop; token count matches.
    let helloIds = tok.encode("hello world")
    let r1 = await runGeneration(promptIds: [], maxTokens: 128, stopIds: [],
                                 decode: decode, backend: FakeBackend(script: helloIds)) { _ in }
    check("roundtrip_text", r1.text.contains("hello world") && r1.finishReason == "stop"
                            && r1.completionTokens == helloIds.count)

    // 2. maxTokens → finish=length, exactly maxTokens emitted.
    let longIds = tok.encode("one two three four five six seven eight")
    let r2 = await runGeneration(promptIds: [], maxTokens: 3, stopIds: [],
                                 decode: decode, backend: FakeBackend(script: longIds)) { _ in }
    check("maxtokens_length", r2.completionTokens == 3 && r2.finishReason == "length")

    // 3. EOS stop: stop id halts before emission; text excludes post-stop content.
    let stopId = 999_999   // synthetic id not in "hi"
    let hiIds = tok.encode("hi")
    let script3 = hiIds + [stopId] + tok.encode("SHOULDNOTAPPEAR")
    let r3 = await runGeneration(promptIds: [], maxTokens: 128, stopIds: [stopId],
                                 decode: decode, backend: FakeBackend(script: script3)) { _ in }
    check("eos_stop", r3.finishReason == "stop" && !r3.text.contains("SHOULDNOTAPPEAR")
                      && r3.completionTokens == hiIds.count)

    // 4. streaming deltas concatenate to the full text.
    var streamed = ""
    let r4 = await runGeneration(promptIds: [], maxTokens: 128, stopIds: [],
                                 decode: decode, backend: FakeBackend(script: helloIds)) { streamed += $0 }
    check("delta_concat", streamed == r4.text && !streamed.isEmpty)

    // 5-7. splitThink: reasoning/content separation (pure; Qwen3.6 <think> handling).
    let sp = splitThink("weighing options</think>\n\nThe answer is 42.")
    check("think_split_content", sp.content == "The answer is 42.")
    check("think_split_reasoning", sp.reasoning == "weighing options")
    let sp2 = splitThink("still thinking")
    check("think_no_close_all_reasoning", sp2.reasoning == "still thinking" && sp2.content == "")

    // 8+. tool-call parsing (pure; Qwen3.6 <tool_call> → OpenAI tool_calls).
    for (name, ok) in ToolParse.selfCheck() { check("tool_\(name)", ok) }

    // prefill progress line (issue #86; pure formatting).
    check("prefill_line", prefillLine(done: 4096, total: 14490, secs: 151.7) == "prefill 4096/14490 (28%) · 27 tok/s")
    check("prefill_line_norate", prefillLine(done: 64, total: 128, secs: 0) == "prefill 64/128 (50%)")

    // chat_template_kwargs (issue #77; pure decode + split routing).
    let kwReq = try? JSONDecoder().decode(ChatCompletionRequest.self, from: Data(
        #"{"messages":[{"role":"user","content":"hi"}],"chat_template_kwargs":{"enable_thinking":false}}"#.utf8))
    check("kwargs_thinking_disabled", kwReq?.thinkingDisabled == true)
    let plainReq = try? JSONDecoder().decode(ChatCompletionRequest.self, from: Data(
        #"{"messages":[{"role":"user","content":"hi"}]}"#.utf8))
    check("kwargs_omitted_thinking_on", plainReq?.thinkingDisabled == false)
    let so = splitOutput("The direct answer.", thinkingDisabled: true)
    check("split_nothink_all_content", so.content == "The direct answer." && so.reasoning.isEmpty)
    let so2 = splitOutput("pondering</think>\nAnswer.", thinkingDisabled: false)
    check("split_think_unchanged", so2.reasoning == "pondering" && so2.content == "Answer.")

    // calib warm-start artifact (issue #73; pure tmp-dir round trip, no GPU).
    for (name, ok) in CalibArtifact.selfCheck() { check("calib_\(name)", ok) }

    // prefix-cache disk persistence (issue #89; pure tmp-dir store checks, no GPU).
    for (name, ok) in PrefixPersist.selfCheck() { check("prefixpersist_\(name)", ok) }

    // prefix-cache RAM tier (issue #117; pure keyed-store checks, no GPU).
    for (name, ok) in PrefixRAMStore.selfCheck() { check("prefixram_\(name)", ok) }

    // speculation gate arithmetic (issue #119; pure, no GPU).
    for (name, ok) in Tell.specGateSelfCheck() { check("specgate_\(name)", ok) }

    // continuous-batching scheduler (issue #6; pure logic over a scripted fake engine).
    for (name, ok) in ContinuousScheduler.selfCheck() { check("batch_\(name)", ok) }

    return lines.joined(separator: "\n") + "\nCOMPTEST \(passed)/\(total)"
}

/// Tokenizer self-test (GPU-free integration check; needs the model's tokenizer files).
/// Mirrors the RAWTESTS pattern: prints "TOKTEST passed/total", nonzero-exit driven by the
/// shell wrapper on any FAIL.
func runTokenizerSelftest(modelDir: String) async -> String {
    var passed = 0, total = 0
    var lines: [String] = []
    func check(_ name: String, _ body: () throws -> Bool) {
        total += 1
        do {
            let ok = try body()
            lines.append("[tok-test] \(name): \(ok ? "PASS" : "FAIL")")
            if ok { passed += 1 }
        } catch {
            lines.append("[tok-test] \(name): FAIL(\(error))")
        }
    }

    let tok: QwispTokenizer
    do {
        tok = try await QwispTokenizer(modelDir: modelDir)
    } catch {
        return "[tok-test] load: FAIL(\(error))\nTOKTEST 0/1"
    }

    check("encode_decode_roundtrip") {
        let s = "Hello, world! def foo(x): return x + 1"
        let ids = tok.encode(s)
        let dec = tok.decode(ids)
        return !ids.isEmpty && dec.contains("Hello, world!") && dec.contains("def foo")
    }
    check("chat_template_nonempty") {
        let ids = try tok.render(messages: [["role": "user", "content": "Hi there"]])
        return !ids.isEmpty
    }
    check("chat_template_preserves_content") {
        let ids = try tok.render(messages: [["role": "user", "content": "MAGICWORD42"]])
        return tok.decode(ids).contains("MAGICWORD42")
    }
    // issue #77: enable_thinking=false pre-closes the think block in the generation prompt;
    // nil additionalContext must render byte-identically to the plain signature.
    check("template_enable_thinking_false") {
        let msgs: [[String: any Sendable]] = [["role": "user", "content": "Hi"]]
        let off = try tok.render(messages: msgs, additionalContext: ["enable_thinking": false])
        let raw = tok.tokenizer.decode(tokens: off, skipSpecialTokens: false)
        return raw.hasSuffix("<think>\n\n</think>\n\n")
    }
    check("template_default_thinking_on") {
        // Omitted kwargs keep the old behavior: generation prompt opens an unclosed think block.
        let msgs: [[String: any Sendable]] = [["role": "user", "content": "Hi"]]
        let ids = try tok.render(messages: msgs)
        return tok.tokenizer.decode(tokens: ids, skipSpecialTokens: false).hasSuffix("<think>\n")
    }

    return lines.joined(separator: "\n") + "\nTOKTEST \(passed)/\(total)"
}
