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

    // 4b. multibyte character SPLIT across tokens (the OpenCode mid-stream stall,
    // 2026-07-22): the half-char step must not emit U+FFFD — once a replacement char
    // is emitted and later rewritten, prefix-guarded delta streams either stall
    // (streamSSE) or duplicate (this path). Deltas must concat to the exact text.
    let frags: [[UInt8]] = [Array("a".utf8) + [0xF0, 0x9F], [0x9A, 0x80] + Array("x".utf8)]
    var mbStreamed = ""
    let r4b = await runGeneration(promptIds: [], maxTokens: 8, stopIds: [],
                                  decode: { ids in String(decoding: ids.flatMap { frags[$0] }, as: UTF8.self) },
                                  backend: FakeBackend(script: [0, 1])) { mbStreamed += $0 }
    check("delta_multibyte_split", mbStreamed == "a🚀x" && r4b.text == "a🚀x")
    check("delta_no_replacement_char", !mbStreamed.contains("\u{FFFD}"))

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

    // prefix arena cap default (the 64K cliff, 2026-07-22): resident follows the model
    // context so long OpenCode sessions never silently lose the cache; streaming keeps
    // the wired-pressure-safe 65536 (PR #70); tiny contexts clamp to the context.
    check("prefixmax_resident_follows_ctx",
          SeedlessBackend.prefixArenaMaxDefault(contextLen: 262_144, isStreaming: false) == 262_144)
    check("prefixmax_streaming_64k",
          SeedlessBackend.prefixArenaMaxDefault(contextLen: 262_144, isStreaming: true) == 65_536)
    check("prefixmax_small_ctx_clamp",
          SeedlessBackend.prefixArenaMaxDefault(contextLen: 32_768, isStreaming: true) == 32_768)

    // cached-arena generation budget (#135 follow-up): the arena must size to
    // prompt + bounded gen budget, never prompt + full context headroom (~80KB/token wired).
    check("genbudget_bounded_by_cap",
          SeedlessBackend.cachedGenBudget(promptLen: 19_600, ceiling: 242_544, arenaMax: 262_144, genCap: 16_384) == 16_384)
    check("genbudget_small_ceiling_wins",
          SeedlessBackend.cachedGenBudget(promptLen: 1_000, ceiling: 512, arenaMax: 262_144, genCap: 16_384) == 512)
    check("genbudget_arena_edge_clamp",
          SeedlessBackend.cachedGenBudget(promptLen: 64_000, ceiling: 100_000, arenaMax: 65_536, genCap: 16_384) == 1_536)

    // prefix-cache disk persistence (issue #89; pure tmp-dir store checks, no GPU).
    for (name, ok) in PrefixPersist.selfCheck() { check("prefixpersist_\(name)", ok) }

    // prefix-cache RAM tier (issue #117; pure keyed-store checks, no GPU).
    for (name, ok) in PrefixRAMStore.selfCheck() { check("prefixram_\(name)", ok) }

    // speculation gate arithmetic (issue #119; pure, no GPU).
    for (name, ok) in Tell.specGateSelfCheck() { check("specgate_\(name)", ok) }

    // continuous-batching scheduler (issue #6; pure logic over a scripted fake engine).
    for (name, ok) in ContinuousScheduler.selfCheck() { check("batch_\(name)", ok) }

    // lane admission plan (#121; pure boundary arithmetic, no GPU).
    for (name, ok) in LaneBatchSlots.admitSelfCheck() { check("laneadmit_\(name)", ok) }

    // token-budget admission scheduler (WS-B Stage A; pure logic over a scripted fake).
    for (name, ok) in ContinuousScheduler.tokenBudgetSelfCheck() { check("tokenbudget_\(name)", ok) }

    // incremental detokenizer (server O(n²) fix; pure fake byte-level tokenizer).
    for (name, ok) in StreamDetok.selfCheck() { check("detok_\(name)", ok) }

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
    // Incremental detokenizer (server O(n²) fix): pushing ids one at a time must equal the
    // full re-decode at EVERY step — this is the exact contract the streaming loop relied
    // on; stepwise equality on the REAL tokenizer makes the swap byte-identical by
    // induction. Japanese + emoji stress multi-byte characters split across BPE tokens.
    check("stream_detok_stepwise_equals_full") {
        let s = "日本語テキストと emoji 🚀🔧 の混在、多バイト分割: héllo — ✓ code `let x = 1`。"
        let ids = tok.encode(s)
        var d = StreamDetok(decode: { tok.decode($0) })
        for k in 0 ..< ids.count {
            if d.push(ids[k]) != tok.decode(Array(ids[0 ... k])) { return false }
        }
        return true
    }

    return lines.joined(separator: "\n") + "\nTOKTEST \(passed)/\(total)"
}
