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
        // One knob: QWISP_LANE_PREFIX_MB total budget (default 3072, resident tier) —
        // 1/3 recurrence tier, 2/3 conversation tier. 0 disables.
        let mb = Swift.max(0, Tell.envInt("QWISP_LANE_PREFIX_MB", 3072))
        self.sharedStore = PrefixRAMStore(budget: mb / 3 * 1_048_576)
        self.convStore = PrefixRAMStore(budget: mb * 2 / 3 * 1_048_576)
    }

    // ── #121 admission plan (pure logic; COMPTEST laneadmit_*) ────────────────
    /// Every cache boundary (restore, capture, save) is an ABSOLUTE chunk-aligned
    /// position. The delta prefill then reproduces the cold path's chunk boundaries
    /// exactly, so restore+delta is bit-identical to cold BY CONSTRUCTION — no
    /// assumption about the hybrid kernels' chunk-composition invariance is needed
    /// (the serialize path's arbitrary-boundary restores lean on the PREFIXE2E gate;
    /// the lane path removes the assumption instead).
    public struct LaneAdmitPlan: Equatable {
        public var restoreLen: Int   // aligned cached prefix to restore (0 = cold admit)
        public var captureAt: Int?   // aligned recurrence boundary → sharedStore save
        public var saveAt: Int?      // aligned last boundary → convStore save
        public init(restoreLen: Int, captureAt: Int?, saveAt: Int?) {
            self.restoreLen = restoreLen; self.captureAt = captureAt; self.saveAt = saveAt
        }
    }
    /// hitLen = longest whole-entry store hit over prompt.dropLast() (aligned by
    /// construction — stores only ever receive aligned keys); lcp = longest PARTIAL
    /// key match (recurrence evidence: two different conversations sharing ≥minShared
    /// tokens define a harness prefix, same operational definition as PrefixPersist #112).
    public static func admitPlan(promptLen: Int, chunk: Int, hitLen: Int, lcp: Int,
                                 minShared: Int) -> LaneAdmitPlan {
        let restoreLen = hitLen
        var capture: Int? = (lcp / chunk) * chunk             // floor to the chunk grid
        if let c = capture, c < minShared || c <= restoreLen { capture = nil }
        var save: Int? = ((promptLen - 1) / chunk) * chunk    // ≥1 token always re-prefilled
        if let s = save, s <= restoreLen || s == capture { save = nil }
        return LaneAdmitPlan(restoreLen: restoreLen, captureAt: capture, saveAt: save)
    }

    /// Pure self-check (no GPU, no model): boundary arithmetic of the admission plan.
    public static func admitSelfCheck() -> [(String, Bool)] {
        func p(_ len: Int, _ hit: Int, _ lcp: Int) -> LaneAdmitPlan {
            admitPlan(promptLen: len, chunk: 1024, hitLen: hit, lcp: lcp, minShared: 1024)
        }
        return [
            // cold first request: no evidence, save the last aligned boundary
            ("cold", p(9000, 0, 0) == LaneAdmitPlan(restoreLen: 0, captureAt: nil, saveAt: 8192)),
            // second same-prefix request: partial match ⇒ capture the shared boundary too
            ("recurrence_capture", p(12000, 0, 9000) == LaneAdmitPlan(restoreLen: 0, captureAt: 8192, saveAt: 11264)),
            // third request: warm restore at the shared boundary, no re-capture
            ("restore_no_recapture", p(12000, 8192, 8192) == LaneAdmitPlan(restoreLen: 8192, captureAt: nil, saveAt: 11264)),
            // shared prefix below minShared: never capture
            ("min_shared", p(4000, 0, 900) == LaneAdmitPlan(restoreLen: 0, captureAt: nil, saveAt: 3072)),
            // capture and save on the same boundary: conv save skipped (one blob, one tier)
            ("save_capture_dedup", p(9000, 0, 8500) == LaneAdmitPlan(restoreLen: 0, captureAt: 8192, saveAt: nil)),
            // prompt shorter than one chunk: no boundary at all
            ("short_prompt", p(800, 0, 0) == LaneAdmitPlan(restoreLen: 0, captureAt: nil, saveAt: nil)),
            // restore already at the last boundary: nothing new to save
            ("restore_covers_save", p(9000, 8192, 8192) == LaneAdmitPlan(restoreLen: 8192, captureAt: nil, saveAt: nil)),
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

    public func admit(prompt: [Int32], slot: Int) -> Int? {
        guard slot >= 0, slot < slotCount, !prompt.isEmpty, prompt.count < maxSeqLen else { return nil }
        let hybrid = ProcessInfo.processInfo.environment["QWISP_HYBRID_PREFILL"] != "0"
        let chunkSize = hybrid ? 1024 : 64
        guard var (fwd, fnBuf) = makeLane(hybrid: hybrid) else { return nil }
        // #121: restore the longest cached aligned prefix, prefill only the delta.
        // Default ON; QWISP_LANE_PREFIX=0 (or _MB=0) opts out. dropLast ⇒ a hit never
        // swallows the whole prompt (the last position is always recomputed for the
        // first-token argmax).
        var plan = LaneAdmitPlan(restoreLen: 0, captureAt: nil, saveAt: nil)
        if Tell.envInt("QWISP_LANE_PREFIX", 1) != 0,
           sharedStore.budget + convStore.budget > 0 {
            let matchable = Array(prompt.dropLast())
            let hs = sharedStore.bestMatch(content: matchable)
            let hc = convStore.bestMatch(content: matchable)
            let hit = [hs, hc].compactMap { $0 }.max { $0.tokens.count < $1.tokens.count }
            let lcp = Swift.max(sharedStore.maxCommonPrefix(with: prompt),
                                convStore.maxCommonPrefix(with: prompt))
            plan = Self.admitPlan(promptLen: prompt.count, chunk: chunkSize,
                                  hitLen: hit?.tokens.count ?? 0, lcp: lcp,
                                  minShared: PrefixPersist.stableMinTokens)
            if plan.restoreLen > 0, let hit {
                if fwd.restorePersistentState(hit.state) {
                    restoreHits += 1
                } else {
                    // Shape/format mismatch half-writes the arena — this lane is unusable.
                    // Cannot happen with same-engine blobs; rebuild fresh and go cold.
                    guard let fresh = makeLane(hybrid: hybrid) else { return nil }
                    (fwd, fnBuf) = fresh
                    plan.restoreLen = 0
                }
            }
        }
        // Canonical prefill + first token, mirroring Tell.prefill + the spec-loop
        // entry EXACTLY (kernel choice and tie-break included): prompt through
        // chunked (hybrid) forward with final norm, then first token =
        // MLX.argMax(engine.logits(lastNormed)) — qmmTiled lm_head, MLX argmax.
        // restoreLen is chunk-aligned, so the loop's boundaries match a cold run's.
        var lastNormed: MLXArray? = nil
        var pos = plan.restoreLen
        while pos < prompt.count {
            let end = Swift.min(pos + chunkSize, prompt.count)
            let x = engine.embed(tokens: Array(prompt[pos ..< end]))
            guard let normed = hybrid ? fwd.forwardRowsHybrid(x, M: end - pos, finalNormW: fnBuf)
                                      : fwd.forwardRows(x, M: end - pos, finalNormW: fnBuf)
            else { return nil }
            lastNormed = normed[end - pos - 1]
            pos = end
            // Boundary states (blob = CPU memcpy of KV used slice + GDN state, ~24KB/token).
            if pos == plan.captureAt {
                sharedStore.save(tokens: Array(prompt[0 ..< pos]), state: fwd.persistentStateData())
            }
            if pos == plan.saveAt {
                convStore.save(tokens: Array(prompt[0 ..< pos]), state: fwd.persistentStateData())
            }
        }
        guard let ln = lastNormed?.reshaped([1, SeedlessEngine.H]),
              let lg = engine.logits(ln, M: 1) else { return nil }
        MLX.eval([lg])
        let first = MLX.argMax(lg[0], axis: -1).item(Int.self)
        lanes[slot] = fwd
        return first
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
        self.scheduler = ContinuousScheduler(slots: laneSlots)
    }

    public func generate(_ prompt: [Int], options: GenerateOptions) -> AsyncStream<Int> {
        let headroom = Swift.max(0, laneCtx - prompt.count - 1)
        let ceiling = options.maxTokens < 0 ? headroom : Swift.min(options.maxTokens, headroom)
        return scheduler.submit(prompt: prompt, maxTokens: ceiling, stopIds: options.stopTokens)
    }
}
