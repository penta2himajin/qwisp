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

// ── Prefill progress → service log (issue #86) ───────────────────────────────
// A long prompt (e.g. an agentic client's ~15K-token system prompt) prefills for minutes on a
// streaming tier with a silent log. Emit one progress line per ≥10s, and accumulate per-request
// prefill tok+secs for logPerf's prefill= field. Generation is serialized (AsyncLock) and the
// hook runs on the single decode thread, so plain statics suffice.
enum PrefillLog {
    nonisolated(unsafe) static var runStart: Date? = nil
    nonisolated(unsafe) static var lastPrint = Date.distantPast
    nonisolated(unsafe) static var tok = 0
    nonisolated(unsafe) static var secs = 0.0
    static func install() {
        Tell.prefillProgress = { done, total in
            let now = Date()
            if done == 0 { runStart = now; return }
            guard let t0 = runStart else { return }
            let runSecs = now.timeIntervalSince(t0)
            if done == total { tok += total; secs += runSecs; runStart = nil }
            if done < total && now.timeIntervalSince(lastPrint) >= 10 {
                lastPrint = now
                fputs("[qwisp] \(prefillLine(done: done, total: total, secs: runSecs))\n", stderr)
            }
        }
    }
    /// Per-request read + reset (called from logPerf, after generation completes).
    static func take() -> (tok: Int, secs: Double) {
        defer { tok = 0; secs = 0 }
        return (tok, secs)
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
    // false in continuous-batching mode (issue #6): concurrent requests are the point —
    // the scheduler batches them; serializing here would defeat it. Everything about the
    // shared-engine safety argument then lives inside ContinuousScheduler's single
    // decode thread instead of this lock.
    let serialize: Bool

    init(tokenizer: QwispTokenizer, backend: any LLMBackend, modelID: String, serialize: Bool = true) {
        self.tokenizer = tokenizer
        self.backend = backend
        self.modelID = modelID
        self.serialize = serialize
        PrefillLog.install()
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

    private func prompt(_ req: ChatCompletionRequest) throws -> (ids: [Int], maxTokens: Int, stop: [Int], contentLen: Int) {
        let msgs = req.messages.map { $0.renderDict }
        let tools = req.tools?.map { $0.spec }
        // chat_template_kwargs (issue #77) → extra Jinja variables, e.g. enable_thinking:false.
        let kwargs = req.chat_template_kwargs.map { $0.mapValues { $0.sendable } }
        let ids = try tokenizer.render(messages: msgs, tools: tools, additionalContext: kwargs)
        // Content boundary (prompt WITHOUT the generation-prompt suffix) → the prefix-cache reuse point.
        let contentLen = (try? tokenizer.render(messages: msgs, tools: tools, addGenerationPrompt: false,
                                                additionalContext: kwargs).count) ?? ids.count
        return (ids, req.max_tokens ?? -1, tokenizer.stopTokenIds, contentLen)   // omitted → until EOS/context
    }

    // One concise throughput line per request → the service log. prefix TTFT (prefill + first
    // token) is separated from the decode rate (tokens after the first / time after the first).
    private func logPerf(_ tag: String, prompt: Int, gen: Int, t0: Date, tFirst: Date?) {
        let now = Date()
        let ttft = tFirst.map { String(format: "%.0fms", $0.timeIntervalSince(t0) * 1000) } ?? "-"
        let dt = now.timeIntervalSince(tFirst ?? t0)
        let rate = dt > 0 ? Double(max(0, gen - 1)) / dt : 0
        // spec accept telemetry (greedy loop): tokens/step incl. the free u-token, draft accept %.
        let st = Tell.lastSpecStats
        let tokPerStep = st.steps > 0 ? Double(st.accepted + st.steps) / Double(st.steps) : 0
        let acc = st.drafted > 0 ? 100.0 * Double(st.accepted) / Double(st.drafted) : 0
        // prompt-processing speed (issue #86 ask 3), summed over the request's prefill runs.
        let pf = PrefillLog.take()
        let pfRate = pf.secs > 0 ? String(format: "%.0f", Double(pf.tok) / pf.secs) : "-"
        fputs(String(format: "[qwisp] %@ prompt=%d prefill=%@ tok/s gen=%d ttft=%@ decode=%.1f tok/s (%.2fs) spec[steps=%d tok/step=%.2f accept=%.0f%% d0=%d rej=%d alt=%d]\n",
                     tag, prompt, pfRate, gen, ttft, rate, now.timeIntervalSince(t0), st.steps, tokPerStep, acc, st.d0, st.rejects, st.altHits), stderr)
    }

    /// Non-streaming completion.
    func complete(_ req: ChatCompletionRequest) async throws -> ChatCompletionResponse {
        let p = try prompt(req)
        let s = sampling(req)
        if serialize { await lock.acquire() }
        let t0 = Date(); var tFirst: Date? = nil
        let r = await runGeneration(promptIds: p.ids, maxTokens: p.maxTokens, stopIds: p.stop,
                                    decode: { self.tokenizer.decode($0) }, backend: backend,
                                    temperature: s.temperature, topP: s.topP, seed: s.seed,
                                    frequencyPenalty: s.frequencyPenalty, presencePenalty: s.presencePenalty,
                                    logitBias: s.logitBias, promptContentLen: p.contentLen) { _ in if tFirst == nil { tFirst = Date() } }
        if serialize { await lock.release() }
        logPerf("complete", prompt: p.ids.count, gen: r.completionTokens, t0: t0, tFirst: tFirst)
        let id = "chatcmpl-\(UUID().uuidString.prefix(24))"
        let (reasoning, afterThink) = splitOutput(r.text, thinkingDisabled: req.thinkingDisabled)
        let (content, toolCalls) = ToolParse.parse(afterThink)
        let finish = toolCalls.isEmpty ? r.finishReason : "tool_calls"
        return ChatCompletionResponse(
            id: id, created: Int(Date().timeIntervalSince1970), model: modelID,
            choices: [Choice(index: 0, message: ResponseMessage(role: "assistant", content: content,
                                                                reasoning_content: reasoning.isEmpty ? nil : reasoning,
                                                                tool_calls: toolCalls.isEmpty ? nil : toolCalls),
                             finish_reason: finish)],
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

        if serialize { await lock.acquire() }
        defer { if serialize { Task { await self.lock.release() } } }

        try await send(Delta(role: "assistant"), nil)
        var outIds: [Int] = [], sentR = "", sentC = "", finish = "stop"
        let s = sampling(req)
        let opts = GenerateOptions(maxTokens: p.maxTokens, stopTokens: p.stop,
                                   temperature: s.temperature, topP: s.topP, seed: s.seed,
                                   frequencyPenalty: s.frequencyPenalty, presencePenalty: s.presencePenalty,
                                   logitBias: s.logitBias, promptContentLen: p.contentLen)
        let t0 = Date(); var tFirst: Date? = nil
        // Incremental detokenize (StreamDetok): full re-decode per token was O(n²) in the
        // generation length — the dominant share of the measured 14-25% per-token server tax.
        var detok = StreamDetok(decode: { self.tokenizer.decode($0) })
        for await tok in backend.generate(p.ids, options: opts) {
            if p.stop.contains(tok) { finish = "stop"; break }
            outIds.append(tok)
            if tFirst == nil { tFirst = Date() }
            detok.push(tok)
            // Deltas derive from the FINALIZED text only (append-only, no dangling
            // U+FFFD) — deriving them from the full view stalled the stream forever
            // once a split multibyte char emitted a replacement char that the next
            // token rewrote (hasPrefix(sent) never held again). See StreamDetok.
            let (r, afterThink) = splitOutput(detok.finalized, thinkingDisabled: req.thinkingDisabled)
            if r.count > sentR.count, r.hasPrefix(sentR) {
                try await send(Delta(reasoning_content: String(r.dropFirst(sentR.count))), nil); sentR = r
            }
            // Stream only the answer text before any <tool_call>; the call XML is buffered, not shown.
            let visible = afterThink.range(of: "<tool_call>").map { String(afterThink[..<$0.lowerBound]) } ?? afterThink
            if visible.count > sentC.count, visible.hasPrefix(sentC) {
                try await send(Delta(content: String(visible.dropFirst(sentC.count))), nil); sentC = visible
            }
            if p.maxTokens >= 0 && outIds.count >= p.maxTokens { finish = "length"; break }  // <0 = until EOS/context
        }
        // End-of-stream flush from the authoritative full decode: anything still held
        // back (pending window) or trailing bytes goes out here; finalized is always a
        // prefix of the full text, so the prefix guards hold by construction.
        let (rEnd, afterEnd) = splitOutput(tokenizer.decode(outIds), thinkingDisabled: req.thinkingDisabled)
        if rEnd.count > sentR.count, rEnd.hasPrefix(sentR) {
            try await send(Delta(reasoning_content: String(rEnd.dropFirst(sentR.count))), nil); sentR = rEnd
        }
        let visEnd = afterEnd.range(of: "<tool_call>").map { String(afterEnd[..<$0.lowerBound]) } ?? afterEnd
        if visEnd.count > sentC.count, visEnd.hasPrefix(sentC) {
            try await send(Delta(content: String(visEnd.dropFirst(sentC.count))), nil); sentC = visEnd
        }
        // Buffered tool calls: parse the completed output and emit them as tool_calls deltas.
        let (_, toolCalls) = ToolParse.parse(afterEnd)
        for (i, tc) in toolCalls.enumerated() {
            try await send(Delta(tool_calls: [ToolCallDelta(index: i, id: tc.id, type: "function", function: tc.function)]), nil)
        }
        if !toolCalls.isEmpty { finish = "tool_calls" }
        try await send(Delta(), finish)
        try await writer.write(ByteBuffer(string: "data: [DONE]\n\n"))
        logPerf("stream", prompt: p.ids.count, gen: outIds.count, t0: t0, tFirst: tFirst)
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
        if !engine.serialize, (req.temperature ?? 0) > 0 || (req.top_p ?? 1) < 1 {
            headers[warnHeader] = "continuous-batching mode (QWISP_BATCH) is greedy-only — sampling params ignored"
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
