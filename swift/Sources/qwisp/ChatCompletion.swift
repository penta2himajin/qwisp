import Foundation
import QwispCore

// OpenAI-compatible /v1/chat/completions types + the transport-agnostic completion core
// (step 5c.2). The core is unit-tested against a fake LLMBackend (GPU-free); the HTTP/SSE
// adapter and the real SeedlessBackend wiring sit on top.

// ── Request ────────────────────────────────────────────────────────────────
struct ChatMessage: Codable {
    let role: String
    let content: String
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
struct ResponseMessage: Codable { let role: String; let content: String; var reasoning_content: String? = nil }
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
struct Delta: Codable { var role: String? = nil; var content: String? = nil; var reasoning_content: String? = nil }
struct ChunkChoice: Codable { let index: Int; let delta: Delta; let finish_reason: String? }
struct ChatCompletionChunk: Codable {
    let id: String
    let object = "chat.completion.chunk"
    let created: Int
    let model: String
    let choices: [ChunkChoice]
}

// ── CLI (`qwisp chat`) — thin in-process wrapper over the same core ──────────
func runChat(prompt: String, tokenizer: QwispTokenizer, backend: any LLMBackend, maxTokens: Int,
             temperature: Double = 0, topP: Double = 1.0, seed: UInt64 = 0,
             frequencyPenalty: Double = 0, presencePenalty: Double = 0, logitBias: [Int: Double] = [:]) async {
    let promptIds: [Int]
    do { promptIds = try tokenizer.render(messages: [["role": "user", "content": prompt]]) }
    catch { fputs("chat: render error: \(error)\n", stderr); return }
    // Route the reasoning to stderr and the answer to stdout, so `qwisp chat … > file` captures
    // only the answer while the thinking still streams to the terminal.
    var full = "", sentR = "", sentC = ""
    _ = await runGeneration(promptIds: promptIds, maxTokens: maxTokens, stopIds: tokenizer.stopTokenIds,
                            decode: { tokenizer.decode($0) }, backend: backend,
                            temperature: temperature, topP: topP, seed: seed,
                            frequencyPenalty: frequencyPenalty, presencePenalty: presencePenalty,
                            logitBias: logitBias) { delta in
        full += delta
        let (r, c) = splitThink(full)
        if r.count > sentR.count, r.hasPrefix(sentR) {
            FileHandle.standardError.write(Data(String(r.dropFirst(sentR.count)).utf8)); sentR = r
        }
        if c.count > sentC.count, c.hasPrefix(sentC) {
            fputs(String(c.dropFirst(sentC.count)), stdout); fflush(stdout); sentC = c
        }
    }
    fputs("\n", stdout)
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
                   decode: ([Int]) -> String, backend: any LLMBackend,
                   temperature: Double = 0, topP: Double = 1.0, seed: UInt64 = 0,
                   frequencyPenalty: Double = 0, presencePenalty: Double = 0,
                   logitBias: [Int: Double] = [:],
                   onDelta: (String) -> Void) async -> CompletionResult {
    let opts = GenerateOptions(maxTokens: maxTokens, stopTokens: stopIds,
                               temperature: temperature, topP: topP, seed: seed,
                               frequencyPenalty: frequencyPenalty, presencePenalty: presencePenalty,
                               logitBias: logitBias)
    var outIds: [Int] = []
    var emitted = ""
    var finish = "stop"
    for await id in backend.generate(promptIds, options: opts) {
        // Defensive: the backend must already honor stopTokens/maxTokens, but guard here too.
        if stopIds.contains(id) { finish = "stop"; break }
        outIds.append(id)
        // Decode the whole sequence and emit the new suffix (handles multi-token characters).
        let full = decode(outIds)
        let delta = full.hasPrefix(emitted) ? String(full[emitted.endIndex...]) : full
        emitted = full
        if !delta.isEmpty { onDelta(delta) }
        if maxTokens >= 0 && outIds.count >= maxTokens { finish = "length"; break }  // <0 = until EOS/context
    }
    return CompletionResult(text: emitted, completionTokens: outIds.count, finishReason: finish)
}
