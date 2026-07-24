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
/// Progress of a resumable prefill (WS-B Stage A, notes/21): one `admitStep` either
/// advances by ≥1 legal chunk and stays `.prefilling`, or reaches the prompt end and
/// returns the first decode token via `.done`.
public enum AdmitProgress {
    case prefilling(consumed: Int)   // paused at a legal chunk boundary; `consumed` tokens this call
    case done(firstToken: Int)       // whole prompt prefilled; greedy first token
    case failed                      // engine/arena error — drop the request
}

/// What the scheduler needs from a batched decode engine. Implemented by
/// ContinuousBatchEngine (GPU) and by a fake in the self-check (pure logic tests).
public protocol BatchSlots: AnyObject {
    var slotCount: Int { get }
    /// Prefill `prompt` into `slot` and return the first generated token (greedy at
    /// prompt end), or nil on engine error.
    func admit(prompt: [Int32], slot: Int) -> Int?
    /// Resumable prefill (WS-B Stage A): advance `slot`'s prefill by at most `tokenBudget`
    /// tokens, always by ≥1 legal chunk (forward progress), pausing only at a boundary
    /// today's atomic `admit()` would also stop at. The full `prompt` is passed every call;
    /// the slot tracks its own resume position. Default (non-resumable engines) runs the
    /// whole prompt in one call — see the extension below.
    func admitStep(prompt: [Int32], slot: Int, tokenBudget: Int) -> AdmitProgress
    /// One batched decode step. `last[b]` = the token to feed slot b (nil = slot idle).
    /// Returns the next token per slot (nil where idle).
    func step(last: [Int32?]) -> [Int?]
    /// Slot finished — drop its per-slot state so the next admit starts clean.
    func release(slot: Int)
}

public extension BatchSlots {
    /// Default for non-resumable engines (e.g. ContinuousBatchEngine): ignore the budget
    /// and prefill the whole prompt atomically. Keeps ContinuousBatchEngine/BatchBackend
    /// untouched — they inherit this and never override it. Resumable slots (LaneBatchSlots,
    /// Stage A) override to pause/resume at chunk boundaries.
    func admitStep(prompt: [Int32], slot: Int, tokenBudget: Int) -> AdmitProgress {
        guard let t = admit(prompt: prompt, slot: slot) else { return .failed }
        return .done(firstToken: t)
    }
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
    /// A reserved-but-not-yet-fully-admitted request (WS-B Stage A): owns `slot`, resumes
    /// prefill from `pos` each budgeted round via `admitStep`.
    private struct Pend { let req: Request; var pos: Int; let slot: Int }

    private let slots: BatchSlots
    private let tokenBudget: Int        // 0 = off (today's atomic drain-then-step); >0 = WS-B budgeted interleave
    private let cond = NSCondition()
    private var queue: [Request] = []
    private var running = false

    /// `tokenBudget` flows in ONLY through this param (never ProcessInfo/env inside the
    /// scheduler) so the self-check stays deterministic; LaneBackend resolves the env gate.
    public init(slots: BatchSlots, tokenBudget: Int = 0) {
        self.slots = slots
        self.tokenBudget = tokenBudget
    }

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
        if tokenBudget > 0 { loopBudgeted(); return }
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

    // ── WS-B Stage A: live budgeted loop (tokenBudget > 0, notes/21) ──────────────
    /// Same per-round policy as `budgetedStep` below, but pulls from the live `queue`
    /// (guarded by `cond`) instead of a static job list — new submissions can arrive
    /// mid-schedule. `loop()`'s `tokenBudget == 0` body above is untouched by this.
    private func loopBudgeted() {
        let B = slots.slotCount
        var localQueue: [Request] = []
        var pending: [Pend] = []
        var active: [Active?] = Array(repeating: nil, count: B)
        while true {
            cond.lock()
            if queue.isEmpty && localQueue.isEmpty && pending.isEmpty && active.allSatisfy({ $0 == nil }) {
                running = false
                cond.unlock()
                return
            }
            localQueue.append(contentsOf: queue)
            queue.removeAll()
            cond.unlock()
            budgetedStep(queue: &localQueue, pending: &pending, active: &active)
        }
    }

    /// One budgeted scheduling round (spec §1, notes/21): RUNNING slots' share is
    /// reserved first (`tokenBudget - runningCount`), then the remainder drains
    /// WAITING/mid-admission prefill FIFO via `admitStep` (≥1 legal chunk per call —
    /// forward-progress guarantee), then every active slot gets its one decode `step()`.
    /// Shared by the live `loopBudgeted()` and the pure `runToCompletion` self-check
    /// driver so both run the identical policy.
    private func budgetedStep(queue: inout [Request], pending: inout [Pend], active: inout [Active?]) {
        let B = slots.slotCount
        var owned = Set(pending.map { $0.slot })
        for b in 0 ..< B where active[b] == nil && !owned.contains(b) && !queue.isEmpty {
            let req = queue.removeFirst()
            guard req.maxTokens > 0 else { req.finish(); continue }   // 0-token request: instant no-op
            pending.append(Pend(req: req, pos: 0, slot: b))
            owned.insert(b)
        }
        let runningCount = active.compactMap { $0 }.count
        var pool = tokenBudget - runningCount
        var stillPending: [Pend] = []
        var toActivate: [(slot: Int, req: Request, first: Int)] = []
        for p in pending {
            guard pool > 0 else { stillPending.append(p); continue }
            switch slots.admitStep(prompt: p.req.prompt, slot: p.slot, tokenBudget: pool) {
            case .failed:
                p.req.finish(); slots.release(slot: p.slot)
            case .prefilling(let consumed):
                pool -= consumed
                stillPending.append(Pend(req: p.req, pos: p.pos + consumed, slot: p.slot))
            case .done(let first):
                pool -= (p.req.prompt.count - p.pos)
                toActivate.append((p.slot, p.req, first))
            }
        }
        pending = stillPending
        for (slot, req, first) in toActivate {
            guard !req.stop.contains(first) else { req.finish(); slots.release(slot: slot); continue }
            req.yield(first)
            if req.maxTokens == 1 { req.finish(); slots.release(slot: slot) }
            else { active[slot] = Active(req: req, produced: 1, last: Int32(first)) }
        }
        guard active.contains(where: { $0 != nil }) else { return }
        let out = slots.step(last: active.map { $0?.last })
        for b in 0 ..< B {
            guard var a = active[b], let tok = out[b] else { continue }
            if a.req.stop.contains(tok) { a.req.finish(); active[b] = nil; slots.release(slot: b); continue }
            a.req.yield(tok)
            a.produced += 1
            a.last = Int32(tok)
            if a.produced >= a.req.maxTokens { a.req.finish(); active[b] = nil; slots.release(slot: b) }
            else { active[b] = a }
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

    // ── WS-B Stage A: token-budget admission scheduling (issue #6 follow-up, notes/21) ──
    /// One scheduled job for the synchronous budgeted driver (self-check + loop() share it).
    struct SchedJob { let prompt: [Int32]; let maxTokens: Int; let stop: Set<Int> }

    /// Synchronous, single-threaded budgeted schedule over `jobs` (submission order),
    /// spending `self.tokenBudget` per iteration: RUNNING decode gets 1 token/slot first,
    /// then WAITING/partially-admitted prefill is drained FIFO via `admitStep` with the
    /// remainder (≥1 chunk forward progress per call). Returns each job's emitted-token
    /// stream in submission order. Pure and deterministic so `tokenBudgetSelfCheck()` can
    /// assert scheduling ORDER; the threaded `loop()` reuses this policy for its
    /// `tokenBudget > 0` branch. `tokenBudget == 0` reproduces today's atomic drain-then-step
    /// (byte-identical streams).
    func runToCompletion(_ jobs: [SchedJob]) -> [[Int]] {
        final class Sink: @unchecked Sendable { var out: [[Int]] = [] }
        let sink = Sink()
        sink.out = Array(repeating: [], count: jobs.count)
        var queue: [Request] = jobs.enumerated().map { i, j in
            Request(prompt: j.prompt, maxTokens: j.maxTokens, stop: j.stop,
                    yield: { sink.out[i].append($0) }, finish: {})
        }
        let B = slots.slotCount
        if tokenBudget <= 0 {
            // tokenBudget == 0: today's atomic admit-to-completion semantics (loop()'s
            // untouched body above), run synchronously so the self-check can compare it
            // byte-for-byte against the budgeted path on the same fake.
            var active: [Active?] = Array(repeating: nil, count: B)
            while true {
                if queue.isEmpty && active.allSatisfy({ $0 == nil }) { break }
                for b in 0 ..< B where active[b] == nil && !queue.isEmpty {
                    let req = queue.removeFirst()
                    guard req.maxTokens > 0, let t0 = slots.admit(prompt: req.prompt, slot: b),
                          !req.stop.contains(t0) else {
                        slots.release(slot: b); continue
                    }
                    req.yield(t0)
                    if req.maxTokens == 1 { slots.release(slot: b) }
                    else { active[b] = Active(req: req, produced: 1, last: Int32(t0)) }
                }
                guard active.contains(where: { $0 != nil }) else { continue }
                let out = slots.step(last: active.map { $0?.last })
                for b in 0 ..< B {
                    guard var a = active[b], let tok = out[b] else { continue }
                    if a.req.stop.contains(tok) { active[b] = nil; slots.release(slot: b); continue }
                    a.req.yield(tok); a.produced += 1; a.last = Int32(tok)
                    if a.produced >= a.req.maxTokens { active[b] = nil; slots.release(slot: b) }
                    else { active[b] = a }
                }
            }
            return sink.out
        }
        var pending: [Pend] = []
        var active: [Active?] = Array(repeating: nil, count: B)
        while !queue.isEmpty || !pending.isEmpty || active.contains(where: { $0 != nil }) {
            budgetedStep(queue: &queue, pending: &pending, active: &active)
        }
        return sink.out
    }

    /// Locked self-check (COMPTEST `tokenbudget_*`, no GPU, no model): the budgeted
    /// scheduler interleaves prefill with decode (no starvation), drains prefill FIFO,
    /// never reorders a stream's tokens vs the non-interleaved path, and only ever pauses
    /// prefill at a legal chunk boundary. Ordering-invariance only — GPU bit-exactness of
    /// the lane path is covered by the lane locked tests (RAWTESTS), not here.
    public static func tokenBudgetSelfCheck() -> [(String, Bool)] {
        // Resumable fake: prefill advances in whole 1024-token chunks (the final remainder
        // completes the prompt), ≥1 chunk per `admitStep` regardless of granted budget
        // (forward-progress guarantee), and every step()/admitStep() call is logged in order.
        // A 1024 fake chunk models the boundary predicate; captureAt sub-chunking is a
        // real-LaneBatchSlots concern the fake need not simulate.
        final class Fake: BatchSlots, @unchecked Sendable {
            enum Ev: Equatable { case step(slot: Int); case admit(slot: Int, budget: Int, consumed: Int) }
            let slotCount = 2
            let chunk = 1024
            var pos: [Int] = [0, 0]        // per-slot resume position
            var log: [Ev] = []
            func admit(prompt: [Int32], slot: Int) -> Int? {   // tokenBudget==0 atomic path
                guard slot >= 0, slot < slotCount, !prompt.isEmpty else { return nil }
                pos[slot] = prompt.count
                return Int((prompt.first ?? 0) + 1)
            }
            func admitStep(prompt: [Int32], slot: Int, tokenBudget: Int) -> AdmitProgress {
                guard slot >= 0, slot < slotCount, !prompt.isEmpty else { return .failed }
                let first = Int((prompt.first ?? 0) + 1)
                let remaining = prompt.count - pos[slot]
                if remaining <= 0 { return .done(firstToken: first) }
                let affordable = Swift.max(1, tokenBudget / chunk) * chunk   // ≥1 whole chunk
                let consumed = Swift.min(affordable, remaining)             // final remainder completes
                pos[slot] += consumed
                log.append(.admit(slot: slot, budget: tokenBudget, consumed: consumed))
                return pos[slot] >= prompt.count ? .done(firstToken: first) : .prefilling(consumed: consumed)
            }
            func step(last: [Int32?]) -> [Int?] {
                var out: [Int?] = Array(repeating: nil, count: slotCount)
                for s in 0 ..< slotCount where last[s] != nil {
                    log.append(.step(slot: s)); out[s] = Int(last[s]!) + 1
                }
                return out
            }
            func release(slot: Int) { if slot >= 0, slot < slotCount { pos[slot] = 0 } }
        }
        func lastAdmit(_ log: [Fake.Ev], slot: Int) -> Int? {
            var idx: Int? = nil
            for (i, e) in log.enumerated() { if case .admit(slot, _, _) = e { idx = i } }
            return idx
        }
        var result: [(String, Bool)] = []

        // 1. no_starvation: a short decode job (slot 0) keeps stepping while a large prompt
        //    (slot 1) is still prefilling — a decode step lands between every prefill chunk.
        let f1 = Fake()
        let out1 = ContinuousScheduler(slots: f1, tokenBudget: 2048).runToCompletion([
            SchedJob(prompt: Array(repeating: 7, count: 1024), maxTokens: 8, stop: []),   // slot 0
            SchedJob(prompt: Array(repeating: 9, count: 6144), maxTokens: 2, stop: []),   // slot 1
        ])
        let b1 = f1.log.enumerated().compactMap { (i, e) -> Int? in
            if case .admit(1, _, _) = e { return i }; return nil
        }
        var interleaved = b1.count >= 2 && out1.count == 2
                          && out1[0] == [8, 9, 10, 11, 12, 13, 14, 15]
        for k in 1 ..< Swift.max(1, b1.count) {
            interleaved = interleaved && f1.log[(b1[k - 1] + 1) ..< b1[k]].contains(.step(slot: 0))
        }
        result.append(("no_starvation", interleaved))

        // 2. fifo_fairness: two large prompts A then B, budget can't finish either in one
        //    iteration → A (submitted first) finishes before B.
        let f2 = Fake()
        let out2 = ContinuousScheduler(slots: f2, tokenBudget: 2048).runToCompletion([
            SchedJob(prompt: Array(repeating: 3, count: 4096), maxTokens: 1, stop: []),   // A → slot 0
            SchedJob(prompt: Array(repeating: 5, count: 4096), maxTokens: 1, stop: []),   // B → slot 1
        ])
        let aDone = lastAdmit(f2.log, slot: 0), bDone = lastAdmit(f2.log, slot: 1)
        let fifo = out2.count == 2 && out2[0] == [4] && out2[1] == [6]
                   && aDone != nil && bDone != nil && aDone! < bDone!
        result.append(("fifo_fairness", fifo))

        // 3. output_identical: same jobs through the SAME fake, tokenBudget 0 (atomic) vs a
        //    chunk multiple → per-stream tokens byte-identical (ordering-invariance only, not
        //    a GPU bit-exactness claim — see the lane locked tests for that).
        let f3 = Fake()
        let jobs3 = [
            SchedJob(prompt: Array(repeating: 7, count: 2000), maxTokens: 4, stop: []),
            SchedJob(prompt: Array(repeating: 21, count: 3000), maxTokens: 3, stop: []),
        ]
        let base = ContinuousScheduler(slots: f3, tokenBudget: 0).runToCompletion(jobs3)
        let inter = ContinuousScheduler(slots: f3, tokenBudget: 1024).runToCompletion(jobs3)
        result.append(("output_identical", base == inter && base == [[8, 9, 10, 11], [22, 23, 24]]))

        // 4. chunk_boundary_respected: every prefill pause lands on a whole chunk or the final
        //    remainder — never an arbitrary partial, never zero — even when the budget is
        //    smaller than one chunk (forward-progress guarantee).
        let f4 = Fake()
        let plen4 = 3 * 1024 + 300
        _ = ContinuousScheduler(slots: f4, tokenBudget: 512).runToCompletion([
            SchedJob(prompt: Array(repeating: 1, count: plen4), maxTokens: 1, stop: []),
        ])
        let consumes = f4.log.compactMap { e -> Int? in
            if case let .admit(0, _, c) = e { return c }; return nil
        }
        var cum = 0, boundaryOK = !consumes.isEmpty
        for c in consumes {
            cum += c
            boundaryOK = boundaryOK && c > 0 && (c % 1024 == 0 || cum == plen4)
        }
        boundaryOK = boundaryOK && cum == plen4
        result.append(("chunk_boundary_respected", boundaryOK))

        return result
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
