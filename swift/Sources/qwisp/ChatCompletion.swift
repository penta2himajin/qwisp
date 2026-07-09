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
    // Accepted but IGNORED — the engine is lossless greedy (see runGeneration / server header).
    let temperature: Double?
    let top_p: Double?
    let n: Int?
}

// ── Response (non-streaming) ─────────────────────────────────────────────────
struct ResponseMessage: Codable { let role: String; let content: String }
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
struct Delta: Codable { var role: String?; var content: String? }
struct ChunkChoice: Codable { let index: Int; let delta: Delta; let finish_reason: String? }
struct ChatCompletionChunk: Codable {
    let id: String
    let object = "chat.completion.chunk"
    let created: Int
    let model: String
    let choices: [ChunkChoice]
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
                   onDelta: (String) -> Void) async -> CompletionResult {
    let opts = GenerateOptions(maxTokens: maxTokens, stopTokens: stopIds)
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
        if outIds.count >= maxTokens { finish = "length"; break }
    }
    return CompletionResult(text: emitted, completionTokens: outIds.count, finishReason: finish)
}
