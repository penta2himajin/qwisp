import Foundation
import QwispCore

// OpenAI-compatible /v1/chat/completions types + the transport-agnostic completion core
// (step 5c.2). The core is unit-tested against a fake LLMBackend (GPU-free); the HTTP/SSE
// adapter and the real SeedlessBackend wiring sit on top.

// ── Request ────────────────────────────────────────────────────────────────

/// OpenAI message `content`: the spec allows EITHER a plain string OR an array of typed
/// parts (`[{"type":"text","text":...}, ...]`). Clients like opencode send the array form
/// even for text-only turns (#82: `Type mismatch … expected 'String'`). Decode both,
/// flattening text parts to one string; non-text parts (images) are dropped — qwisp is
/// text-only, so a multimodal client at least gets its text through instead of a 4xx.
struct MessageContent: Codable {
    let text: String?
    init(text: String?) { self.text = text }
    init(from decoder: Decoder) throws {
        let c = try decoder.singleValueContainer()
        if c.decodeNil() { text = nil; return }
        if let s = try? c.decode(String.self) { text = s; return }
        struct Part: Codable { let type: String?; let text: String? }
        let parts = try c.decode([Part].self)   // let a genuinely malformed content still 4xx
        let joined = parts.compactMap { $0.type == "text" ? $0.text : nil }.joined()
        text = joined.isEmpty ? nil : joined
    }
    func encode(to encoder: Encoder) throws {
        var c = encoder.singleValueContainer()
        try c.encode(text)   // round-trips as a plain string
    }
}

struct ChatMessage: Codable {
    let role: String
    let content: MessageContent?         // string OR array of parts (#82); nil for tool-only turns
    var tool_calls: [ReqToolCall]? = nil // assistant → previous function calls (history)
    var tool_call_id: String? = nil      // role:"tool" → which call this result answers
    var name: String? = nil
}

struct ChatCompletionRequest: Codable {
    let model: String?
    let messages: [ChatMessage]
    let stream: Bool?
    let max_tokens: Int?
    // temperature / top_p / seed / penalties / logit_bias are HONORED via speculative sampling
    // (Option B). n > 1 is still ignored (single completion).
    let temperature: Double?
    let top_p: Double?
    let n: Int?
    let seed: Int?
    let frequency_penalty: Double?
    let presence_penalty: Double?
    let logit_bias: [String: Double]?   // OpenAI: token-id string → bias
    var tools: [Tool]? = nil            // function-calling tool specs
    var tool_choice: JSONValue? = nil   // accepted; the model decides (auto)
    // Extra Jinja variables for the chat template (issue #77), e.g.
    // {"enable_thinking": false} — the Qwen3.6 template then pre-closes the think block.
    var chat_template_kwargs: [String: JSONValue]? = nil
}

extension ChatCompletionRequest {
    /// chat_template_kwargs.enable_thinking == false (issue #77): the template emits an empty
    /// `<think>\n\n</think>` in the generation prompt, so the model's output IS the answer —
    /// splitThink must be bypassed (no `</think>` will ever appear in the output).
    var thinkingDisabled: Bool {
        if case .bool(false) = chat_template_kwargs?["enable_thinking"] { return true }
        return false
    }
}

/// splitThink, unless thinking is disabled for this request — then all output is content.
func splitOutput(_ s: String, thinkingDisabled: Bool) -> (reasoning: String, content: String) {
    thinkingDisabled ? ("", s) : splitThink(s)
}

// Qwen3.6 is a reasoning model: the chat template injects `<think>` into the generation prompt,
// so the model emits `[reasoning]</think>[answer]`. Split that so `content` is the clean answer
// and the thinking goes to `reasoning_content` (DeepSeek convention) instead of leaking into
// `content`. There is no plain-content phase before `</think>`, so an output that hasn't produced
// `</think>` yet is treated as all-reasoning (empty content).
func splitThink(_ s: String) -> (reasoning: String, content: String) {
    guard let r = s.range(of: "</think>") else { return (s, "") }
    let content = String(s[r.upperBound...]).drop(while: { $0 == "\n" })
    return (String(s[..<r.lowerBound]), String(content))
}

// ── Response (non-streaming) ─────────────────────────────────────────────────
struct ResponseMessage: Codable {
    let role: String
    var content: String? = nil          // nil when the turn is only tool_calls
    var reasoning_content: String? = nil
    var tool_calls: [ToolCall]? = nil
}
struct Choice: Codable { let index: Int; let message: ResponseMessage; let finish_reason: String }
struct Usage: Codable { let prompt_tokens: Int; let completion_tokens: Int; let total_tokens: Int }
struct ChatCompletionResponse: Codable {
    let id: String
    let object = "chat.completion"
    let created: Int
    let model: String
    let choices: [Choice]
    let usage: Usage
}

// ── Response (streaming chunk) ───────────────────────────────────────────────
struct Delta: Codable { var role: String? = nil; var content: String? = nil; var reasoning_content: String? = nil; var tool_calls: [ToolCallDelta]? = nil }
struct ChunkChoice: Codable { let index: Int; let delta: Delta; let finish_reason: String? }
struct ChatCompletionChunk: Codable {
    let id: String
    let object = "chat.completion.chunk"
    let created: Int
    let model: String
    let choices: [ChunkChoice]
}

/// One prefill progress line (issue #86): "prefill 4096/14490 (28%) · 27 tok/s".
/// Shared by the chat CLI (\r-overwritten) and the server log (one line / 10s).
func prefillLine(done: Int, total: Int, secs: Double) -> String {
    let pct = total > 0 ? done * 100 / total : 0
    let rate = secs > 0 ? String(format: " · %.0f tok/s", Double(done) / secs) : ""
    return "prefill \(done)/\(total) (\(pct)%)\(rate)"
}

// ── CLI (`qwisp chat`) — thin in-process wrapper over the same core ──────────
func runChat(prompt: String, tokenizer: QwispTokenizer, backend: any LLMBackend, maxTokens: Int,
             temperature: Double = 0, topP: Double = 1.0, seed: UInt64 = 0,
             frequencyPenalty: Double = 0, presencePenalty: Double = 0, logitBias: [Int: Double] = [:],
             noThinking: Bool = false) async {
    let promptIds: [Int]
    do { promptIds = try tokenizer.render(messages: [["role": "user", "content": prompt]],
                                          additionalContext: noThinking ? ["enable_thinking": false] : nil) }
    catch { fputs("chat: render error: \(error)\n", stderr); return }
    // Prefill progress → stderr (issue #86): a long prompt on a streaming tier prefills for
    // minutes, previously in silence. \r-overwritten; shown once a prefill run exceeds 1s.
    // The hook runs on the (single) decode thread; stats are read after drain() below.
    var pfRunStart: Date? = nil, pfShown = false, pfTok = 0, pfSecs = 0.0
    Tell.prefillProgress = { done, total in
        let now = Date()
        if done == 0 { pfRunStart = now; return }
        guard let t0 = pfRunStart else { return }
        let secs = now.timeIntervalSince(t0)
        if done == total { pfTok += total; pfSecs += secs; pfRunStart = nil }
        if secs >= 1 {
            pfShown = true
            FileHandle.standardError.write(Data("\r[qwisp] \(prefillLine(done: done, total: total, secs: secs))   ".utf8))
        }
        if done == total && pfShown { pfShown = false; FileHandle.standardError.write(Data("\n".utf8)) }
    }
    // Route the reasoning to stderr and the answer to stdout, so `qwisp chat … > file` captures
    // only the answer while the thinking still streams to the terminal.
    var full = "", sentR = "", sentC = ""
    var tFirst: Date? = nil
    let r = await runGeneration(promptIds: promptIds, maxTokens: maxTokens, stopIds: tokenizer.stopTokenIds,
                            decode: { tokenizer.decode($0) }, backend: backend,
                            temperature: temperature, topP: topP, seed: seed,
                            frequencyPenalty: frequencyPenalty, presencePenalty: presencePenalty,
                            logitBias: logitBias) { delta in
        if tFirst == nil { tFirst = Date() }
        full += delta
        let (r, c) = splitOutput(full, thinkingDisabled: noThinking)
        if r.count > sentR.count, r.hasPrefix(sentR) {
            FileHandle.standardError.write(Data(String(r.dropFirst(sentR.count)).utf8)); sentR = r
        }
        if c.count > sentC.count, c.hasPrefix(sentC) {
            fputs(String(c.dropFirst(sentC.count)), stdout); fflush(stdout); sentC = c
        }
    }
    fputs("\n", stdout)
    // Exit-teardown fix (#47 handoff): join the decode thread before main returns —
    // it outlives the EOS break above and its MLX evals race static teardown.
    (backend as? SeedlessBackend)?.drain()
    // Speed summary (issue #86 ask 3): prompt-processing speed alongside generation speed.
    let pfRate = pfSecs > 0 ? Double(pfTok) / pfSecs : 0
    let decSecs = tFirst.map { Date().timeIntervalSince($0) } ?? 0
    let decRate = decSecs > 0 ? Double(max(0, r.completionTokens - 1)) / decSecs : 0
    // Leading \n: the reasoning stream (stderr) ends mid-line.
    fputs(String(format: "\n[qwisp] prompt %d tok (%.0f tok/s) · gen %d tok (%.1f tok/s)\n",
                 promptIds.count, pfRate, r.completionTokens, decRate), stderr)
}

// ── Core ─────────────────────────────────────────────────────────────────────
struct CompletionResult {
    var text: String
    var completionTokens: Int
    var finishReason: String   // "stop" | "length"
}

/// Drive the token-id backend to completion, decoding incrementally. Calls `onDelta`
/// with each new text fragment (for SSE). Honors EOS/stop tokens (not emitted) and
/// maxTokens (finish_reason "length"). Transport-agnostic; GPU-free with a fake backend.
func runGeneration(promptIds: [Int], maxTokens: Int, stopIds: [Int],
                   decode: @escaping ([Int]) -> String, backend: any LLMBackend,
                   temperature: Double = 0, topP: Double = 1.0, seed: UInt64 = 0,
                   frequencyPenalty: Double = 0, presencePenalty: Double = 0,
                   logitBias: [Int: Double] = [:], promptContentLen: Int? = nil,
                   onDelta: (String) -> Void) async -> CompletionResult {
    let opts = GenerateOptions(maxTokens: maxTokens, stopTokens: stopIds,
                               temperature: temperature, topP: topP, seed: seed,
                               frequencyPenalty: frequencyPenalty, presencePenalty: presencePenalty,
                               logitBias: logitBias, promptContentLen: promptContentLen)
    var outIds: [Int] = []
    var emitted = ""
    var finish = "stop"
    // Incremental detokenize (StreamDetok, O(n²) fix): push() returns the same text as
    // decode(outIds) at every step (TOKTEST-locked contract), multi-token characters included.
    var detok = StreamDetok(decode: decode)
    for await id in backend.generate(promptIds, options: opts) {
        // Defensive: the backend must already honor stopTokens/maxTokens, but guard here too.
        if stopIds.contains(id) { finish = "stop"; break }
        outIds.append(id)
        let full = detok.push(id)
        let delta = full.hasPrefix(emitted) ? String(full[emitted.endIndex...]) : full
        emitted = full
        if !delta.isEmpty { onDelta(delta) }
        if maxTokens >= 0 && outIds.count >= maxTokens { finish = "length"; break }  // <0 = until EOS/context
    }
    return CompletionResult(text: emitted, completionTokens: outIds.count, finishReason: finish)
}
