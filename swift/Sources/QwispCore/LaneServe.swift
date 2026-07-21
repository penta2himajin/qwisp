import Foundation
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

    public init?(store: WeightStore, slots: Int, maxSeqLen: Int) {
        self.engine = SeedlessEngine.build(store: store)
        // Driver = weights + M=slots scratch; its own caches are never advanced.
        guard let (drv, _) = engine.makeFused(maxM: Swift.max(8, slots), maxSeqLen: 8) else { return nil }
        self.driver = drv
        self.slotCount = slots
        self.maxSeqLen = maxSeqLen
        self.lanes = Array(repeating: nil, count: slots)
    }

    public func admit(prompt: [Int32], slot: Int) -> Int? {
        guard slot >= 0, slot < slotCount, !prompt.isEmpty, prompt.count < maxSeqLen,
              let (fwd, _) = engine.makeFused(maxM: 64, maxSeqLen: maxSeqLen) else { return nil }
        // Chunked prefill of all but the last prompt token; the last token goes
        // through stepArgmax so the first generated token comes out of the same
        // 1-CB primitive the decode loop uses (solo path — bit-exact anchor).
        var pos = 0
        while pos < prompt.count - 1 {
            let end = Swift.min(pos + 64, prompt.count - 1)
            guard fwd.forwardRows(engine.embed(tokens: Array(prompt[pos ..< end])), M: end - pos) != nil
            else { return nil }
            pos = end
        }
        guard let first = fwd.stepArgmax([prompt[prompt.count - 1]])?.first else { return nil }
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
