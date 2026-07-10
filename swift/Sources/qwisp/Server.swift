import Foundation
import Hummingbird
import NIOCore
import HTTPTypes
import QwispCore

// OpenAI-compatible HTTP server (step 5c) over the token-id backend.
// /v1/models + /v1/chat/completions (streaming SSE + non-streaming).

// ── /v1/models ───────────────────────────────────────────────────────────────
struct ModelObject: ResponseEncodable, Codable {
    let id: String
    let object = "model"
    let created = 0
    let owned_by = "qwisp"
}
struct ModelsResponse: ResponseEncodable, Codable {
    let object = "list"
    let data: [ModelObject]
}

// ── Async mutex: serialize generation (the engine holds one shared GPU/KV state) ──
// ponytail: minimal fair lock; cancellation-while-queued is not handled (rare for a
// local single-user server) — upgrade to a cancellation-aware lock if it ever bites.
actor AsyncLock {
    private var locked = false
    private var waiters: [CheckedContinuation<Void, Never>] = []
    func acquire() async {
        if !locked { locked = true; return }
        await withCheckedContinuation { waiters.append($0) }
    }
    func release() {
        if waiters.isEmpty { locked = false } else { waiters.removeFirst().resume() }
    }
}

// ── Engine: tokenizer + token-id backend, generation serialized ───────────────
// @unchecked Sendable is justified: every path that touches the backend's mutable
// engine/KV state runs under `lock`, so there is never concurrent access.
final class QwispEngine: @unchecked Sendable {
    let tokenizer: QwispTokenizer
    let backend: any LLMBackend
    let modelID: String
    private let lock = AsyncLock()

    init(tokenizer: QwispTokenizer, backend: any LLMBackend, modelID: String) {
        self.tokenizer = tokenizer
        self.backend = backend
        self.modelID = modelID
    }

    /// True when the client sent a param that is still unsupported (n > 1 only —
    /// temperature/top_p/seed are now honored via speculative sampling).
    func samplingRequested(_ req: ChatCompletionRequest) -> Bool {
        (req.n ?? 1) > 1
    }

    struct SamplingParams {
        var temperature: Double, topP: Double, seed: UInt64
        var frequencyPenalty: Double, presencePenalty: Double, logitBias: [Int: Double]
    }
    /// Resolve sampling from the request. temperature default 0 = greedy/lossless.
    func sampling(_ req: ChatCompletionRequest) -> SamplingParams {
        var bias: [Int: Double] = [:]
        for (k, v) in req.logit_bias ?? [:] { if let t = Int(k) { bias[t] = v } }
        return SamplingParams(temperature: req.temperature ?? 0, topP: req.top_p ?? 1.0,
                              seed: UInt64(bitPattern: Int64(req.seed ?? 0)),
                              frequencyPenalty: req.frequency_penalty ?? 0,
                              presencePenalty: req.presence_penalty ?? 0, logitBias: bias)
    }

    private func prompt(_ req: ChatCompletionRequest) throws -> (ids: [Int], maxTokens: Int, stop: [Int]) {
        let msgs = req.messages.map { ["role": $0.role, "content": $0.content] }
        let ids = try tokenizer.render(messages: msgs)
        return (ids, req.max_tokens ?? -1, tokenizer.stopTokenIds)   // omitted → until EOS/context
    }

    /// Non-streaming completion.
    func complete(_ req: ChatCompletionRequest) async throws -> ChatCompletionResponse {
        let p = try prompt(req)
        let s = sampling(req)
        await lock.acquire()
        let r = await runGeneration(promptIds: p.ids, maxTokens: p.maxTokens, stopIds: p.stop,
                                    decode: { tokenizer.decode($0) }, backend: backend,
                                    temperature: s.temperature, topP: s.topP, seed: s.seed,
                                    frequencyPenalty: s.frequencyPenalty, presencePenalty: s.presencePenalty,
                                    logitBias: s.logitBias) { _ in }
        await lock.release()
        let id = "chatcmpl-\(UUID().uuidString.prefix(24))"
        return ChatCompletionResponse(
            id: id, created: Int(Date().timeIntervalSince1970), model: modelID,
            choices: [Choice(index: 0, message: ResponseMessage(role: "assistant", content: r.text),
                             finish_reason: r.finishReason)],
            usage: Usage(prompt_tokens: p.ids.count, completion_tokens: r.completionTokens,
                         total_tokens: p.ids.count + r.completionTokens))
    }

    /// Streaming completion: writes OpenAI SSE chunks to the response writer.
    /// Mirrors runGeneration's loop (async writes preclude reusing its sync onDelta).
    func streamSSE(_ req: ChatCompletionRequest, writer: inout some ResponseBodyWriter) async throws {
        let p = try prompt(req)
        let id = "chatcmpl-\(UUID().uuidString.prefix(24))"
        let created = Int(Date().timeIntervalSince1970)
        let enc = JSONEncoder()
        func send(_ delta: Delta, _ finish: String?) async throws {
            let chunk = ChatCompletionChunk(id: id, created: created, model: modelID,
                                            choices: [ChunkChoice(index: 0, delta: delta, finish_reason: finish)])
            let json = String(data: (try? enc.encode(chunk)) ?? Data(), encoding: .utf8) ?? "{}"
            try await writer.write(ByteBuffer(string: "data: \(json)\n\n"))
        }

        await lock.acquire()
        defer { Task { await lock.release() } }

        try await send(Delta(role: "assistant", content: nil), nil)
        var outIds: [Int] = [], emitted = "", finish = "stop"
        let s = sampling(req)
        let opts = GenerateOptions(maxTokens: p.maxTokens, stopTokens: p.stop,
                                   temperature: s.temperature, topP: s.topP, seed: s.seed,
                                   frequencyPenalty: s.frequencyPenalty, presencePenalty: s.presencePenalty,
                                   logitBias: s.logitBias)
        for await tok in backend.generate(p.ids, options: opts) {
            if p.stop.contains(tok) { finish = "stop"; break }
            outIds.append(tok)
            let full = tokenizer.decode(outIds)
            let d = full.hasPrefix(emitted) ? String(full[emitted.endIndex...]) : full
            emitted = full
            if !d.isEmpty { try await send(Delta(role: nil, content: d), nil) }
            if p.maxTokens >= 0 && outIds.count >= p.maxTokens { finish = "length"; break }  // <0 = until EOS/context
        }
        try await send(Delta(role: nil, content: nil), finish)
        try await writer.write(ByteBuffer(string: "data: [DONE]\n\n"))
    }
}

// ── Routing ──────────────────────────────────────────────────────────────────
private let warnHeader = HTTPField.Name("x-qwisp-warning")!

func makeRouter(engine: QwispEngine, modelID: String) -> Router<BasicRequestContext> {
    let router = Router()

    router.get("/v1/models") { _, _ in
        ModelsResponse(data: [ModelObject(id: modelID)])
    }

    router.post("/v1/chat/completions") { request, context -> Response in
        let req = try await request.decode(as: ChatCompletionRequest.self, context: context)
        var headers = HTTPFields()
        if engine.samplingRequested(req) {
            headers[warnHeader] = "n > 1 ignored (single completion); temperature/top_p/seed are honored"
        }
        if req.stream == true {
            headers[.contentType] = "text/event-stream"
            headers[.cacheControl] = "no-cache"
            let body = ResponseBody { writer in
                try await engine.streamSSE(req, writer: &writer)
                try await writer.finish(nil)   // terminate the chunked body (not auto-called)
            }
            return Response(status: .ok, headers: headers, body: body)
        } else {
            let resp = try await engine.complete(req)
            headers[.contentType] = "application/json"
            let data = try JSONEncoder().encode(resp)
            return Response(status: .ok, headers: headers, body: ResponseBody(byteBuffer: ByteBuffer(bytes: data)))
        }
    }

    return router
}

func runServe(engine: QwispEngine, modelID: String, port: Int) async throws {
    let app = Application(
        router: makeRouter(engine: engine, modelID: modelID),
        configuration: .init(address: .hostname("127.0.0.1", port: port))
    )
    print("qwisp serve → http://127.0.0.1:\(port)  (model: \(modelID))")
    print("[qwisp serve] NOTE: temperature/top_p/seed honored via speculative sampling (Option B); n>1 ignored. Default temperature 0 = greedy/lossless.")
    try await app.runService()
}
