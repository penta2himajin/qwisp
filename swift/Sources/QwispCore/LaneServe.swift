import Foundation
import Metal
import MLX

// Lane-batched serving path (Stage 1 wiring, parallel sub-agent fan-out).
//
// BatchSlots over SeedlessLaneBatch: reuses the issue-#6 ContinuousScheduler
// (continuous slot refill, one decode thread) but swaps the MLX batched engine
// for per-slot raw-engine lanes. Unlike ContinuousBatchEngine (MLX, NOT
// bit-exact — batch near-tie flips), every lane here advances BIT-IDENTICALLY
// to solo raw greedy decode (locked tests 90/91), so fan-out responses match
// the serialize path byte-for-byte.
//
// Design (v1):
//   - admit = standalone chunked prefill on a fresh lane forward (the validated
//     standalone-then-inject shape; prefill-overlap is the known follow-up).
//   - step  = SeedlessLaneBatch.stepArgmaxBatch over the ACTIVE lanes only —
//     the batch is rebuilt when the active set changes (per-lane cost is ~linear
//     in B, so idle lanes must not ride along; rebuild cost is two tiny buffers
//     per lane).
//   - release = drop the lane forward (caches freed; next admit re-creates).
//   - Greedy only, resident tier only. Sequence budget per lane =
//     min(model context, QWISP_LANE_CTX, default 16384) — KV arena is allocated
//     per admit at this length.
public final class LaneBatchSlots: BatchSlots {
    let engine: SeedlessEngine
    let driver: SeedlessFusedVerify.SeedlessFusedForward
    public let slotCount: Int
    let maxSeqLen: Int
    private var lanes: [SeedlessFusedVerify.SeedlessFusedForward?]
    /// WS-B Stage A (notes/21): resumable per-slot prefill state, hoisted out of
    /// `admit()`'s local loop so a budgeted scheduler can pause/resume it across calls
    /// at existing legal chunk boundaries. Numerically identical to the old atomic loop —
    /// scheduling-only, zero kernel/precision changes.
    private struct Prefill {
        var fwd: SeedlessFusedVerify.SeedlessFusedForward
        let fnBuf: MTLBuffer
        let hybrid: Bool
        let chunkSize: Int
        let plan: LaneAdmitPlan
        var pos: Int
        var lastNormed: MLXArray?
    }
    private var prefills: [Prefill?]
    private var batch: SeedlessLaneBatch? = nil
    private var batchLanes: [ObjectIdentifier] = []   // active-set key for rebuild
    // #121 prefix-cache-aware admission: cross-request decode states (persistentStateData
    // blobs) so same-prefix fan-out and multi-turn extensions prefill only the delta.
    // Two tiers (one PrefixRAMStore each — mixing them is wrong: save()'s supersede
    // semantics would drop a shared-harness boundary entry every time a longer
    // full-prompt key lands, re-paying the shared prefill on alternate requests):
    //   sharedStore — recurrence-detected harness prefixes (fan-out sharing)
    //   convStore   — per-conversation last-boundary states (multi-turn extension)
    // Decode-thread only (ContinuousScheduler admits serially) → no lock.
    private var sharedStore: PrefixRAMStore
    private var convStore: PrefixRAMStore
    /// Gate observability (mirrors SeedlessBackend.prefixRAMHits): warm restores this process.
    public private(set) var restoreHits = 0

    public init?(store: WeightStore, slots: Int, maxSeqLen: Int) {
        self.engine = SeedlessEngine.build(store: store)
        // Driver = weights + M=slots scratch; its own caches are never advanced.
        guard let (drv, _) = engine.makeFused(maxM: Swift.max(8, slots), maxSeqLen: 8) else { return nil }
        self.driver = drv
        self.slotCount = slots
        self.maxSeqLen = maxSeqLen
        self.lanes = Array(repeating: nil, count: slots)
        self.prefills = Array(repeating: nil, count: slots)
        // One knob: QWISP_LANE_PREFIX_MB total budget (default 3072, resident tier) —
        // 1/3 recurrence tier, 2/3 conversation tier. 0 disables.
        let mb = Swift.max(0, Tell.envInt("QWISP_LANE_PREFIX_MB", 3072))
        self.sharedStore = PrefixRAMStore(budget: mb / 3 * 1_048_576)
        self.convStore = PrefixRAMStore(budget: mb * 2 / 3 * 1_048_576)
    }

    // ── #121 admission plan (pure logic; COMPTEST laneadmit_*) ────────────────
    /// Boundaries are EXACT positions (v2): capture sits at the exact recurrence LCP
    /// and the conversation state is saved at the prompt end. The v1 chunk-aligned
    /// grid was bit-safe by construction but measured 860-token re-prefills per admit
    /// (~2.4s each) whenever the shared prefix fell between grid points — the
    /// dominant fan-out cost after #121. Arbitrary-boundary restore + delta prefill
    /// re-chunks from the restore point, the SAME contract the shipped serialize
    /// RAM tier relies on (PREFIXE2E gate, #117 lossless 4/4); the lane gate is the
    /// replay identity diff (6/6 byte-identical required).
    public struct LaneAdmitPlan: Equatable {
        public var restoreLen: Int   // cached prefix to restore (0 = cold admit)
        public var captureAt: Int?   // exact recurrence LCP → sharedStore save (mid-prefill)
        public var saveAtEnd: Bool   // save the prompt-end state → convStore (post-prefill)
        public init(restoreLen: Int, captureAt: Int?, saveAtEnd: Bool) {
            self.restoreLen = restoreLen; self.captureAt = captureAt; self.saveAtEnd = saveAtEnd
        }
    }
    /// hitLen = longest whole-entry store hit over prompt.dropLast() (never the full
    /// prompt — the last position is always recomputed for the first-token argmax);
    /// lcp = longest PARTIAL key match (recurrence evidence: two different
    /// conversations sharing ≥minShared tokens define a harness prefix, same
    /// operational definition as PrefixPersist #112).
    public static func admitPlan(promptLen: Int, hitLen: Int, lcp: Int,
                                 minShared: Int) -> LaneAdmitPlan {
        let restoreLen = hitLen
        // Capture only when the recurrence boundary beats the restore point by more
        // than a blob copy is worth (~256 tokens of prefill), and leaves ≥1 token.
        var capture: Int? = Swift.min(lcp, promptLen - 1)
        if let c = capture, c < minShared || c < restoreLen + 256 { capture = nil }
        // Conversation-end state: skip only when the restore already covers the whole
        // prompt but its last token (nothing new past the stored entry).
        let saveAtEnd = promptLen >= minShared && restoreLen < promptLen - 1
        return LaneAdmitPlan(restoreLen: restoreLen, captureAt: capture, saveAtEnd: saveAtEnd)
    }

    /// Pure self-check (no GPU, no model): boundary arithmetic of the admission plan.
    public static func admitSelfCheck() -> [(String, Bool)] {
        func p(_ len: Int, _ hit: Int, _ lcp: Int) -> LaneAdmitPlan {
            admitPlan(promptLen: len, hitLen: hit, lcp: lcp, minShared: 1024)
        }
        return [
            // cold first request: no recurrence evidence, save the prompt-end state
            ("cold", p(9000, 0, 0) == LaneAdmitPlan(restoreLen: 0, captureAt: nil, saveAtEnd: true)),
            // second same-prefix request: capture at the EXACT shared boundary
            ("recurrence_capture", p(12000, 0, 9000) == LaneAdmitPlan(restoreLen: 0, captureAt: 9000, saveAtEnd: true)),
            // third request: warm restore at the shared boundary, no re-capture
            ("restore_no_recapture", p(12000, 9000, 9000) == LaneAdmitPlan(restoreLen: 9000, captureAt: nil, saveAtEnd: true)),
            // shared prefix below minShared: never capture
            ("min_shared", p(4000, 0, 900) == LaneAdmitPlan(restoreLen: 0, captureAt: nil, saveAtEnd: true)),
            // capture not worth a blob copy over the existing restore point
            ("capture_margin", p(12000, 8900, 9000) == LaneAdmitPlan(restoreLen: 8900, captureAt: nil, saveAtEnd: true)),
            // lcp reaching the prompt end clamps to promptLen-1 (≥1 token recomputed)
            ("capture_clamped", p(9000, 0, 9000) == LaneAdmitPlan(restoreLen: 0, captureAt: 8999, saveAtEnd: true)),
            // short prompt: not worth caching at all
            ("short_prompt", p(800, 0, 0) == LaneAdmitPlan(restoreLen: 0, captureAt: nil, saveAtEnd: false)),
            // restore covers all but the recomputed last token: nothing new to save
            ("restore_covers_save", p(9000, 8999, 8999) == LaneAdmitPlan(restoreLen: 8999, captureAt: nil, saveAtEnd: false)),
        ]
    }

    /// Fresh lane forward with the canonical hybrid wiring (same trios as TellRuntime's
    /// hybrid setup). maxM 1024 = the canonical steel-hybrid prefill chunk (the shipped
    /// serialize path prefills hybrid@1024; the canonical greedy stream is defined WITH
    /// it — raw chunked prefill produces the pre-hybrid stream and diverges ~100 tokens
    /// in on f16 near-ties). Costs ~200MB scratch per active lane, resident tier.
    private func makeLane(hybrid: Bool) -> (SeedlessFusedVerify.SeedlessFusedForward, MTLBuffer)? {
        guard let (fwd, fnBuf) = engine.makeFused(maxM: 1024, maxSeqLen: maxSeqLen) else { return nil }
        if hybrid {
            var hw: [Int: (qkv: (MLXArray, MLXArray, MLXArray), z: (MLXArray, MLXArray, MLXArray), out: (MLXArray, MLXArray, MLXArray))] = [:]
            var aw: [Int: (q: (MLXArray, MLXArray, MLXArray), k: (MLXArray, MLXArray, MLXArray), v: (MLXArray, MLXArray, MLXArray), o: (MLXArray, MLXArray, MLXArray))] = [:]
            for (i, spec) in engine.layers.enumerated() {
                if let g = spec.gdn {
                    hw[i] = (qkv: (g.qkvWq, g.qkvSc, g.qkvBi), z: (g.zWq, g.zSc, g.zBi), out: (g.outWq, g.outSc, g.outBi))
                } else if let a = spec.attn {
                    aw[i] = (q: (a.qWq, a.qSc, a.qBi), k: (a.kWq, a.kSc, a.kBi), v: (a.vWq, a.vSc, a.vBi), o: (a.oWq, a.oSc, a.oBi))
                }
            }
            fwd.hybridGdnW = hw
            fwd.hybridAttnW = aw
            if Tell.envInt("QWISP_HYBRID_MOE", 1) == 1 {
                var mw: [Int: (g: (MLXArray, MLXArray, MLXArray), u: (MLXArray, MLXArray, MLXArray), d: (MLXArray, MLXArray, MLXArray))] = [:]
                for (i, spec) in engine.layers.enumerated() {
                    let m = spec.moe
                    mw[i] = (g: (m.swGWq, m.swGSc, m.swGBi), u: (m.swUWq, m.swUSc, m.swUBi), d: (m.swDWq, m.swDSc, m.swDBi))
                }
                fwd.hybridMoEW = mw
            }
        }
        return (fwd, fnBuf)
    }

    /// Atomic prefill (thin wrapper, WS-B Stage A): drive `admitStep` to completion with
    /// an effectively-unbounded budget — one call, same result as the old atomic loop.
    public func admit(prompt: [Int32], slot: Int) -> Int? {
        while true {
            switch admitStep(prompt: prompt, slot: slot, tokenBudget: Int.max) {
            case .done(let firstToken, _): return firstToken
            case .failed: return nil
            case .prefilling: continue   // Int.max never actually pauses; defensive only
            }
        }
    }

    /// Resumable prefill (WS-B Stage A, notes/21): advances `slot`'s prefill by at most
    /// `tokenBudget` tokens, always by ≥1 legal chunk (forward-progress guarantee — see
    /// the `!first` guard below), pausing only at the same internal chunk / `captureAt`
    /// boundaries the old atomic loop would stop at. Setup (restore-plan lookup + lane
    /// creation) happens once per admission, on the first call for a slot.
    public func admitStep(prompt: [Int32], slot: Int, tokenBudget: Int) -> AdmitProgress {
        guard slot >= 0, slot < slotCount, !prompt.isEmpty, prompt.count < maxSeqLen else { return .failed }
        var st: Prefill
        if let existing = prefills[slot] {
            st = existing
        } else {
            let hybrid = ProcessInfo.processInfo.environment["QWISP_HYBRID_PREFILL"] != "0"
            let chunkSize = hybrid ? 1024 : 64
            guard var (fwd, fnBuf) = makeLane(hybrid: hybrid) else { return .failed }
            // #121: restore the longest cached prefix, prefill only the delta.
            // Default ON; QWISP_LANE_PREFIX=0 (or _MB=0) opts out. dropLast ⇒ a hit never
            // swallows the whole prompt (the last position is always recomputed for the
            // first-token argmax).
            var plan = LaneAdmitPlan(restoreLen: 0, captureAt: nil, saveAtEnd: false)
            if Tell.envInt("QWISP_LANE_PREFIX", 1) != 0,
               sharedStore.budget + convStore.budget > 0 {
                let matchable = Array(prompt.dropLast())
                let hs = sharedStore.bestMatch(content: matchable)
                let hc = convStore.bestMatch(content: matchable)
                let hit = [hs, hc].compactMap { $0 }.max { $0.tokens.count < $1.tokens.count }
                let lcp = Swift.max(sharedStore.maxCommonPrefix(with: prompt),
                                    convStore.maxCommonPrefix(with: prompt))
                plan = Self.admitPlan(promptLen: prompt.count,
                                      hitLen: hit?.tokens.count ?? 0, lcp: lcp,
                                      minShared: PrefixPersist.stableMinTokens)
                if plan.restoreLen > 0, let hit {
                    if fwd.restorePersistentState(hit.state) {
                        restoreHits += 1
                    } else {
                        // Shape/format mismatch half-writes the arena — this lane is unusable.
                        // Cannot happen with same-engine blobs; rebuild fresh and go cold.
                        guard let fresh = makeLane(hybrid: hybrid) else { return .failed }
                        (fwd, fnBuf) = fresh
                        plan.restoreLen = 0
                    }
                }
            }
            st = Prefill(fwd: fwd, fnBuf: fnBuf, hybrid: hybrid, chunkSize: chunkSize,
                        plan: plan, pos: plan.restoreLen, lastNormed: nil)
        }
        // Canonical prefill, mirroring Tell.prefill + the spec-loop entry EXACTLY (kernel
        // choice and tie-break included): prompt through chunked (hybrid) forward with
        // final norm. Re-chunks from st.pos and splits one chunk at captureAt (exact
        // boundaries — bit identity gated by the replay diff, see admitPlan). Always
        // completes ≥1 chunk before checking the budget (forward-progress guarantee) —
        // `first` gates the check, not the loop entry.
        var consumed = 0
        var first = true
        while st.pos < prompt.count {
            if !first, consumed >= tokenBudget { break }
            var end = Swift.min(st.pos + st.chunkSize, prompt.count)
            if let c = st.plan.captureAt, st.pos < c { end = Swift.min(end, c) }
            let x = engine.embed(tokens: Array(prompt[st.pos ..< end]))
            guard let normed = st.hybrid ? st.fwd.forwardRowsHybrid(x, M: end - st.pos, finalNormW: st.fnBuf)
                                          : st.fwd.forwardRows(x, M: end - st.pos, finalNormW: st.fnBuf)
            else { prefills[slot] = nil; return .failed }
            st.lastNormed = normed[end - st.pos - 1]
            consumed += end - st.pos
            st.pos = end
            first = false
            // Recurrence boundary state (blob = CPU memcpy of KV used slice + GDN state,
            // ~24KB/token) — captured mid-prefill at the exact shared-prefix end.
            if st.pos == st.plan.captureAt {
                sharedStore.save(tokens: Array(prompt[0 ..< st.pos]), state: st.fwd.persistentStateData())
            }
        }
        guard st.pos >= prompt.count else {
            prefills[slot] = st
            return .prefilling(consumed: consumed)
        }
        guard let ln = st.lastNormed?.reshaped([1, SeedlessEngine.H]),
              let lg = engine.logits(ln, M: 1) else { prefills[slot] = nil; return .failed }
        MLX.eval([lg])
        let firstToken = MLX.argMax(lg[0], axis: -1).item(Int.self)
        // Conversation-end state (multi-turn extension restores). The arena is exactly at
        // the prompt boundary here — the decode steps that follow only append past it.
        if st.plan.saveAtEnd {
            convStore.save(tokens: prompt, state: st.fwd.persistentStateData())
        }
        lanes[slot] = st.fwd
        prefills[slot] = nil
        return .done(firstToken: firstToken, consumed: consumed)
    }

    public func step(last: [Int32?]) -> [Int?] {
        let activeSlots = (0 ..< slotCount).filter { last[$0] != nil && lanes[$0] != nil }
        guard !activeSlots.isEmpty else { return Array(repeating: nil, count: slotCount) }
        let activeLanes = activeSlots.map { lanes[$0]! }
        let key = activeLanes.map { ObjectIdentifier($0) }
        if batch == nil || key != batchLanes {
            batch = SeedlessLaneBatch(driver: driver, lanes: activeLanes)
            batchLanes = key
        }
        guard let b = batch, let toks = b.stepArgmaxBatch(activeSlots.map { last[$0]! })
        else { return Array(repeating: nil, count: slotCount) }
        var out: [Int?] = Array(repeating: nil, count: slotCount)
        for (i, s) in activeSlots.enumerated() { out[s] = toks[i] }
        return out
    }

    public func release(slot: Int) {
        guard slot >= 0, slot < slotCount else { return }
        lanes[slot] = nil
        prefills[slot] = nil   // drop an aborted mid-admission's SeedlessFusedForward
        batch = nil; batchLanes = []   // active set changed
    }
}

// ── Server backend ────────────────────────────────────────────────────────────
/// LLMBackend over the continuous scheduler with lane-batched raw decode:
/// concurrent generate() calls batch together AND stay bit-exact with the
/// serialize path. Resident tier only; greedy only (server warns on sampling
/// params). Opt-in via QWISP_LANES=<B>; the default serial path is untouched.
public final class LaneBackend: LLMBackend, @unchecked Sendable {
    let scheduler: ContinuousScheduler
    let laneCtx: Int
    public let slots: Int

    public convenience init(modelDir: String, tier: SeedlessTier) throws {
        try self.init(modelDir: modelDir, slots: Swift.max(2, Tell.envInt("QWISP_LANES", 4)))
    }

    public init(modelDir: String, slots: Int) throws {
        guard DeviceCalibration.defaultC() >= 256 else {
            throw NSError(domain: "qwisp", code: 7, userInfo: [NSLocalizedDescriptionKey:
                "lane batching (QWISP_LANES) requires a resident-tier machine (≥32GB); this machine resolves to a streaming tier"])
        }
        let store = try WeightStore(modelDir: modelDir)
        store.residentAll()
        let ctx = Swift.min(SeedlessBackend.readContextLen(modelDir),
                            Swift.max(2048, Tell.envInt("QWISP_LANE_CTX", 16384)))
        guard let laneSlots = LaneBatchSlots(store: store, slots: slots, maxSeqLen: ctx) else {
            throw NSError(domain: "qwisp", code: 7, userInfo: [NSLocalizedDescriptionKey:
                "lane batching: engine build failed"])
        }
        self.slots = slots
        self.laneCtx = ctx
        // WS-B Stage A (notes/21): opt-in token-budget admission scheduler. Default OFF
        // (QWISP_TOKEN_BUDGET_SCHED=0) — today's atomic drain-then-step is unchanged
        // unless explicitly enabled. Gate/size env reads live here only, never inside the
        // scheduler or LaneBatchSlots, so their self-checks stay deterministic.
        let budgetSchedOn = Tell.envInt("QWISP_TOKEN_BUDGET_SCHED", 0) != 0
        let budget = budgetSchedOn ? Swift.max(1, Tell.envInt("QWISP_TOKEN_BUDGET", 2048)) : 0
        self.scheduler = ContinuousScheduler(slots: laneSlots, tokenBudget: budget)
    }

    public func generate(_ prompt: [Int], options: GenerateOptions) -> AsyncStream<Int> {
        let headroom = Swift.max(0, laneCtx - prompt.count - 1)
        let ceiling = options.maxTokens < 0 ? headroom : Swift.min(options.maxTokens, headroom)
        return scheduler.submit(prompt: prompt, maxTokens: ceiling, stopIds: options.stopTokens)
    }
}
