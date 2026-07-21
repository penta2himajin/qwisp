import Foundation
import MLX
import MLXFast

// Continuous-batching serving path (issue #6 productization).
//
// Positioning (issue6-batching-validated, 2026-06-30): batching is a MULTI-USER lever
// only — single-user resident is already at the forward compute ceiling (~270 tok/s)
// via SuffixSpec, and batching×spec combining adds nothing (same ceiling). The value is
// aggregate throughput when several OpenAI requests are in flight: continuous slot-swap
// measured 1.78x vs static waves, per-stream correct greedy.
//
// Mode contract: resident tier ONLY (all experts resident; the MLX batched path — raw's
// edge is M=1 and vanishes at M≥8), GREEDY only, and NOT bit-exact with single-stream
// decode (batch-composition near-tie flips: reproducibility ≠ correctness). Hence the
// whole path is opt-in (QWISP_BATCH=<B>) and the default serial path is untouched.
//
// Structure: full-attn layers (25%) keep per-slot KVCache + per-slot RoPE position
// (AttentionLayer.callContinuous); GDN layers (75%) are position-independent → one
// batched cache with per-row reset. Prefill is standalone-then-inject (the validated
// design); prefill-overlap and SLA/timeout are the known production follow-ups.

// ── Scheduler seam ────────────────────────────────────────────────────────────
/// What the scheduler needs from a batched decode engine. Implemented by
/// ContinuousBatchEngine (GPU) and by a fake in the self-check (pure logic tests).
public protocol BatchSlots: AnyObject {
    var slotCount: Int { get }
    /// Prefill `prompt` into `slot` and return the first generated token (greedy at
    /// prompt end), or nil on engine error.
    func admit(prompt: [Int32], slot: Int) -> Int?
    /// One batched decode step. `last[b]` = the token to feed slot b (nil = slot idle).
    /// Returns the next token per slot (nil where idle).
    func step(last: [Int32?]) -> [Int?]
    /// Slot finished — drop its per-slot state so the next admit starts clean.
    func release(slot: Int)
}

// ── Scheduler (pure logic; GPU only through BatchSlots) ───────────────────────
/// Continuous scheduler: requests queue up, free slots admit immediately (no wave
/// barrier), every step decodes all active slots in one batch. One decode thread;
/// submissions from any thread/task. ponytail: no client-disconnect cancellation and
/// no SLA/timeout in v1 (requests run to stop/max) — the issue-#6 production TODO list.
public final class ContinuousScheduler {
    struct Request {
        let prompt: [Int32]
        let maxTokens: Int          // ≥0 (caller clamps "until EOS" to context headroom)
        let stop: Set<Int>
        let yield: (Int) -> Void
        let finish: () -> Void
    }
    private struct Active { var req: Request; var produced: Int; var last: Int32 }

    private let slots: BatchSlots
    private let cond = NSCondition()
    private var queue: [Request] = []
    private var running = false

    public init(slots: BatchSlots) { self.slots = slots }

    public func submit(prompt: [Int], maxTokens: Int, stopIds: [Int]) -> AsyncStream<Int> {
        AsyncStream { cont in
            let req = Request(prompt: prompt.map { Int32($0) }, maxTokens: Swift.max(0, maxTokens),
                              stop: Set(stopIds), yield: { cont.yield($0) }, finish: { cont.finish() })
            self.cond.lock()
            self.queue.append(req)
            let startThread = !self.running
            self.running = true
            self.cond.unlock()
            if startThread { Thread.detachNewThread { self.loop() } }
        }
    }

    private func loop() {
        let B = slots.slotCount
        var active: [Active?] = Array(repeating: nil, count: B)
        while true {
            // Admit pending requests into free slots (continuous refill — the 1.78x).
            cond.lock()
            if queue.isEmpty && active.allSatisfy({ $0 == nil }) {
                running = false          // idle → thread exits; next submit restarts it
                cond.unlock()
                return
            }
            var admits: [(slot: Int, req: Request)] = []
            for b in 0 ..< B where active[b] == nil && !queue.isEmpty {
                admits.append((b, queue.removeFirst()))
            }
            cond.unlock()
            for (b, req) in admits {
                guard req.maxTokens > 0, let t0 = slots.admit(prompt: req.prompt, slot: b),
                      !req.stop.contains(t0) else {
                    req.finish(); slots.release(slot: b)   // empty budget / engine error / instant EOS
                    continue
                }
                req.yield(t0)
                if req.maxTokens == 1 { req.finish(); slots.release(slot: b) }
                else { active[b] = Active(req: req, produced: 1, last: Int32(t0)) }
            }
            guard active.contains(where: { $0 != nil }) else { continue }
            // One batched step over all active slots.
            let out = slots.step(last: active.map { $0?.last })
            for b in 0 ..< B {
                guard var a = active[b], let tok = out[b] else { continue }
                if a.req.stop.contains(tok) {                       // EOS: not emitted
                    a.req.finish(); active[b] = nil; slots.release(slot: b)
                    continue
                }
                a.req.yield(tok)
                a.produced += 1
                a.last = Int32(tok)
                if a.produced >= a.req.maxTokens {                  // length cap
                    a.req.finish(); active[b] = nil; slots.release(slot: b)
                } else {
                    active[b] = a
                }
            }
        }
    }

    /// Pure self-check with a scripted fake engine (no GPU): completion, per-stream
    /// values, EOS/maxTokens honoring, and that concurrency never exceeds the slots.
    public static func selfCheck() -> [(String, Bool)] {
        final class Fake: BatchSlots, @unchecked Sendable {
            let slotCount = 2
            var seed: [Int32?] = [nil, nil]     // per-slot stream state: next = last*2+1 pattern? keep simple: next = last+seedStep
            var maxBusy = 0
            func busy() { maxBusy = Swift.max(maxBusy, seed.compactMap { $0 }.count) }
            func admit(prompt: [Int32], slot: Int) -> Int? {
                guard slot >= 0 else { return nil }
                seed[slot] = prompt.first ?? 0; busy()
                return Int((prompt.first ?? 0) + 1)                 // first token = p0+1
            }
            func step(last: [Int32?]) -> [Int?] {
                busy()
                return last.map { $0.map { Int($0) + 1 } }          // next token = last+1
            }
            func release(slot: Int) { if slot >= 0 { seed[slot] = nil } }
        }
        let fake = Fake()
        let sched = ContinuousScheduler(slots: fake)
        // 5 requests on 2 slots: request i has prompt [100*i], maxTokens 4, no stop →
        // expected stream: [100i+1, 100i+2, 100i+3, 100i+4].
        let sem = DispatchSemaphore(value: 0)
        final class Box: @unchecked Sendable { var streams: [[Int]] = Array(repeating: [], count: 6) }
        let box = Box()
        for i in 0 ..< 5 {
            let s = sched.submit(prompt: [100 * i], maxTokens: 4, stopIds: [])
            Task.detached { for await t in s { box.streams[i].append(t) }; sem.signal() }
        }
        // request 5: stop token cuts the stream after 2 tokens (503 is not emitted).
        let s5 = sched.submit(prompt: [500], maxTokens: 100, stopIds: [503])
        Task.detached { for await t in s5 { box.streams[5].append(t) }; sem.signal() }
        for _ in 0 ..< 6 { sem.wait() }
        var allLen = true
        for i in 0 ..< 5 {
            let base = 100 * i
            let expect: [Int] = [base + 1, base + 2, base + 3, base + 4]
            allLen = allLen && (box.streams[i] == expect)
        }
        return [
            ("streams_correct", allLen),
            ("stop_honored", box.streams[5] == [501, 502]),
            ("concurrency_capped", fake.maxBusy <= 2),
            ("slots_reused", fake.maxBusy == 2),
        ]
    }
}

// ── GPU engine ────────────────────────────────────────────────────────────────
/// Batched decode over the MLX model path: per-slot KV + position on full-attn layers
/// (callContinuous), one batched GDN cache with per-row inject/reset on linear layers.
public final class ContinuousBatchEngine: BatchSlots {
    let model: QwispModel
    public let slotCount: Int
    private var slotKV: [[KVCache]]        // [layer][slot]; only read on full-attn layers
    private var gdnCaches: [GDNCache?]     // [layer]; only non-nil on linear layers
    private var positions: [Int]           // per-slot current sequence position

    public init(model: QwispModel, slots: Int) {
        self.model = model
        self.slotCount = slots
        self.slotKV = model.layers.map { $0.isLinear ? [] : (0 ..< slots).map { _ in KVCache() } }
        self.gdnCaches = model.layers.map { $0.isLinear ? GDNCache() : nil }
        self.positions = Array(repeating: 0, count: slots)
    }

    /// Standalone prefill → inject (validated design): run the prompt through fresh
    /// single-stream caches, then swap the KV caches in per-slot and row-write the GDN
    /// states into the batched cache.
    public func admit(prompt: [Int32], slot: Int) -> Int? {
        guard slot >= 0, slot < slotCount, !prompt.isEmpty else { return nil }
        let fresh = model.makeCaches()
        let logits = model(MLXArray(prompt).reshaped([1, prompt.count]), caches: fresh)
        let tok = MLX.argMax(logits[0, prompt.count - 1], axis: -1)
        MLX.eval([tok] + fresh.flatMap { $0.stateArrays })
        for (i, layer) in model.layers.enumerated() {
            if layer.isLinear {
                let g = gdnCaches[i]!, f = fresh[i].gdn
                guard let fc = f.convState, let fr = f.recState else { return nil }
                // Lazily size the batched state [B, …] from the first observed shapes.
                if g.convState == nil {
                    g.convState = MLX.zeros([slotCount] + Array(fc.shape.dropFirst())).asType(fc.dtype)
                    g.recState = MLX.zeros([slotCount] + Array(fr.shape.dropFirst())).asType(fr.dtype)
                }
                g.convState![slot] = fc[0]      // row inject (research-validated subscript write)
                g.recState![slot] = fr[0]
            } else {
                slotKV[i][slot] = fresh[i].kv   // per-slot KV: just adopt the prefilled cache
            }
        }
        positions[slot] = prompt.count
        return tok.item(Int.self)
    }

    /// One batched decode step. Idle slots are fed token 0 — their rows compute garbage
    /// that is ignored and their state is replaced on the next admit (forward cost is
    /// row-sublinear; dynamic shrink measured +2% only and was rejected).
    public func step(last: [Int32?]) -> [Int?] {
        var x = model.embed(MLXArray(last.map { $0 ?? 0 }).reshaped([slotCount, 1]))   // [B,1,H]
        for (i, layer) in model.layers.enumerated() {
            x = layer.callContinuous(x, gdnCache: gdnCaches[i], slotKV: slotKV[i], positions: positions)
        }
        x = MLXFast.rmsNorm(x, weight: model.store.req("language_model.model.norm.weight"), eps: model.eps)
        let ids = MLX.argMax(model.headProj().apply(x)[0..., 0], axis: -1)             // [B]
        // Materialize the step + every cache it touched (lazy-graph growth guard).
        var state: [MLXArray] = [ids]
        for (i, layer) in model.layers.enumerated() {
            if layer.isLinear { state += [gdnCaches[i]!.convState, gdnCaches[i]!.recState].compactMap { $0 } }
            else { for kv in slotKV[i] { state += [kv.keys, kv.values].compactMap { $0 } } }
        }
        MLX.eval(state)
        let out = ids.asArray(Int32.self)
        for b in 0 ..< slotCount where last[b] != nil { positions[b] += 1 }
        return last.enumerated().map { $0.1 == nil ? nil : Int(out[$0.0]) }
    }

    public func release(slot: Int) {
        guard slot >= 0, slot < slotCount else { return }
        for (i, layer) in model.layers.enumerated() where !layer.isLinear { slotKV[i][slot] = KVCache() }
        positions[slot] = 0
        // GDN rows keep stale state until the next admit row-writes them — never read while idle.
    }
}

// ── Server backend ────────────────────────────────────────────────────────────
/// LLMBackend over the continuous scheduler: concurrent generate() calls batch together
/// instead of serializing. Resident tier only; greedy only (sampling params are ignored —
/// the server warns). The serial SeedlessBackend path is untouched; this backend is only
/// constructed under QWISP_BATCH=<B>.
public final class BatchBackend: LLMBackend, @unchecked Sendable {
    let scheduler: ContinuousScheduler
    let contextLen: Int
    public let slots: Int

    public convenience init(modelDir: String, tier: SeedlessTier) throws {
        try self.init(modelDir: modelDir, slots: Swift.max(2, Tell.envInt("QWISP_BATCH", 4)))
    }

    public init(modelDir: String, slots: Int) throws {
        guard DeviceCalibration.defaultC() >= 256 else {
            throw NSError(domain: "qwisp", code: 6, userInfo: [NSLocalizedDescriptionKey:
                "continuous batching (QWISP_BATCH) requires a resident-tier machine (≥32GB); this machine resolves to a streaming tier"])
        }
        let store = try WeightStore(modelDir: modelDir)
        store.residentAll()
        self.slots = slots
        self.scheduler = ContinuousScheduler(slots: ContinuousBatchEngine(model: QwispModel(store: store), slots: slots))
        self.contextLen = SeedlessBackend.readContextLen(modelDir)
    }

    // #121 workload guard: batch admits pay a FULL prefill per request (no prefix cache,
    // no SuffixSpec). On agentic-harness traffic — consecutive prompts sharing a large
    // system+tools prefix — this measured 2.6x SLOWER than the default serialize path
    // (and diverges: MLX f16 near-tie flips). Detect that signature and say so once.
    private var lastPromptHead: [Int] = []
    private var sharedPrefixSeen = 0
    private var warnedSharedPrefix = false

    public func generate(_ prompt: [Int], options: GenerateOptions) -> AsyncStream<Int> {
        if !warnedSharedPrefix {
            let head = Array(prompt.prefix(4096))
            var lcp = 0
            while lcp < head.count && lcp < lastPromptHead.count && head[lcp] == lastPromptHead[lcp] { lcp += 1 }
            lastPromptHead = head
            if lcp >= 1024 { sharedPrefixSeen += 1 }
            if sharedPrefixSeen >= 2 {
                warnedSharedPrefix = true
                FileHandle.standardError.write(Data(
                    "[qwisp] NOTE: requests share large prompt prefixes (agentic-harness signature). QWISP_BATCH re-prefills every request and loses ~2.6x to the default mode on this traffic (issue #121) — unset QWISP_BATCH unless you need multi-user cold-prompt throughput.\n".utf8))
            }
        }
        let headroom = Swift.max(0, contextLen - prompt.count)
        let ceiling = options.maxTokens < 0 ? headroom : Swift.min(options.maxTokens, headroom)
        return scheduler.submit(prompt: prompt, maxTokens: ceiling, stopIds: options.stopTokens)
    }
}
