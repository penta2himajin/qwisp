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
//   - Greedy only, resident tier only. Sequence budget per lane is ctx-adaptive
//     (WS-B Stage B, notes/22): sized per admit from the request's own prompt +
//     gen budget (LaneBackend.sizePlan), capped by the model context /
//     QWISP_LANE_CTX override — KV arena is allocated per admit at that length.
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
        // Only ever read within the SAME admitStep call that sets it (the final chunk reaching
        // prompt.count, right before the first-token argmax) — a paused/resumed prefill never
        // consumes a stale value across the pause boundary.
        var lastNormed: MLXArray?
    }
    private var prefills: [Prefill?]
    /// WS-B Stage B (notes/22): aggregate lane KV-byte budget (the PR #135 collapse guard).
    /// `arenaBytes[slot]` is set once the slot's arena is actually allocated (makeLane
    /// succeeds — ceil-grant, unlike the self-check fake's floor-grant, see canAdmit below)
    /// and cleared on release.
    let kvBudgetBytes: Int
    private var arenaBytes: [Int]
    /// WS-B Stage B hardening: same-round TOCTOU guard. The scheduler's admission loop can
    /// call `canAdmit` for MULTIPLE free slots within a single `budgetedStep()` pass, before
    /// any of those candidates' `admitStep` has actually run — so `arenaBytes` alone is stale
    /// for the 2nd+ peek in that pass (each would see the same pre-round `committed` total,
    /// letting several individually-fitting big admits blow the aggregate budget together —
    /// exactly the concurrent-burst collapse this gate exists to stop). `reservedBytes` holds
    /// the running total of `canAdmit`-approved-but-not-yet-committed candidates; `admitStep`
    /// releases a candidate's hold (symmetric subtract, same formula) the moment it starts
    /// first-call setup for it, whether that setup succeeds or fails.
    private var reservedBytes: Int = 0
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

    public init?(store: WeightStore, slots: Int, maxSeqLen: Int, kvBudgetBytes: Int) {
        self.engine = SeedlessEngine.build(store: store)
        // Driver = weights + M=slots scratch; its own caches are never advanced.
        guard let (drv, _) = engine.makeFused(maxM: Swift.max(8, slots), maxSeqLen: 8) else { return nil }
        self.driver = drv
        self.slotCount = slots
        self.maxSeqLen = maxSeqLen
        self.kvBudgetBytes = kvBudgetBytes
        self.lanes = Array(repeating: nil, count: slots)
        self.prefills = Array(repeating: nil, count: slots)
        self.arenaBytes = Array(repeating: 0, count: slots)
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

    /// Pure self-check (no GPU, no model): WS-B Stage B ctx-adaptive sizing policy
    /// (`LaneBackend.sizePlan`). `oversized` must be computed from headroom ≤ 0 BEFORE
    /// `cachedGenBudget` (whose `max(1, …)` floor never signals "too big"); the genBudget
    /// reuses `SeedlessBackend.cachedGenBudget` verbatim (notes/22 fact 3).
    public static func lanesizeSelfCheck() -> [(String, Bool)] {
        // 1. unset maxTokens (until-EOS): a 35K prompt in a 262K ctx admits with the gen cap.
        let p1 = LaneBackend.sizePlan(promptLen: 35_000, maxTokens: -1, ctxMax: 262_144, genCap: 16_384)
        let c1 = !p1.oversized && p1.genBudget == 16_384 && p1.seqBudget == 35_000 + 1 + 16_384
        // 2. explicit maxTokens capped by the gen cap; expected genBudget via the real function.
        let ceiling2 = Swift.min(100_000, 32_768 - 1_000 - 1)
        let expect2 = SeedlessBackend.cachedGenBudget(promptLen: 1_000, ceiling: ceiling2, arenaMax: 32_768, genCap: 16_384)
        let p2 = LaneBackend.sizePlan(promptLen: 1_000, maxTokens: 100_000, ctxMax: 32_768, genCap: 16_384)
        let c2 = !p2.oversized && p2.genBudget == expect2
        // 3. prompt ≥ ctx: headroom ≤ 0 → oversized (must NOT be masked by the genBudget floor).
        let p3 = LaneBackend.sizePlan(promptLen: 40_000, maxTokens: -1, ctxMax: 32_768, genCap: 16_384)
        let c3 = p3.oversized == true
        // 4. legacy shape: seqBudget is exactly prompt + 1 + genBudget.
        let p4 = LaneBackend.sizePlan(promptLen: 100, maxTokens: 50, ctxMax: 8_192, genCap: 16_384)
        let c4 = !p4.oversized && p4.seqBudget == 100 + 1 + p4.genBudget
        // 5-7. Arena ALLOCATION size vs the eligibility cap (notes/22 addendum finding 3).
        // maxSeqLen is the model-context eligibility cap (262144); using it as the legacy
        // allocation size allocated 5.37GB per lane, ungated (canAdmit runs only on the
        // budgeted path) — measured as a flag-off B≥2 throughput collapse. The legacy
        // sentinel must keep the OLD fixed cap so QWISP_TOKEN_BUDGET_SCHED=0 stays
        // bit-for-bit legacy, arena included.
        let c5 = arenaSeqLen(seqBudget: 0, maxSeqLen: 262_144) == legacyArenaCap
        // A small eligibility cap still wins over the legacy cap (never allocate past it).
        let c6 = arenaSeqLen(seqBudget: 0, maxSeqLen: 8_192) == 8_192
        // Budgeted path is unchanged: per-request size, clamped to the eligibility cap.
        let c7 = arenaSeqLen(seqBudget: 51_385, maxSeqLen: 262_144) == 51_385
            && arenaSeqLen(seqBudget: 300_000, maxSeqLen: 262_144) == 262_144
        // 8-11. #151: eligibility must not promise more than the path will allocate.
        // Regression under test: flag-off admitted a 24K prompt (eligible against the
        // 262K ctx), then admitStep allocated legacyArenaCap and returned .failed, which
        // ContinuousBatch.loop() turns into an empty 200 with no NOTE. Capping eligibility
        // to the allocation makes sizePlan refuse it VISIBLY instead.
        let c8 = eligibilityCtx(budgetSchedOn: false, ctx: 262_144) == legacyArenaCap
        // Flag-ON is untouched — the budgeted path sizes each arena from its own seqBudget.
        let c9 = eligibilityCtx(budgetSchedOn: true, ctx: 262_144) == 262_144
        // The general invariant, stated once: on the flag-off path the arena actually
        // allocated for an eligible prompt always covers the whole eligibility window.
        let c10 = [8_192, 16_384, 32_768, 262_144].allSatisfy { ctx in
            let elig = eligibilityCtx(budgetSchedOn: false, ctx: ctx)
            return arenaSeqLen(seqBudget: 0, maxSeqLen: elig) == elig
        }
        // The reproduction itself: 24K prompt, model-context ctx. Refused when flag-off,
        // still served when flag-on.
        let off11 = LaneBackend.sizePlan(promptLen: 24_000, maxTokens: 3,
                                         ctxMax: eligibilityCtx(budgetSchedOn: false, ctx: 262_144),
                                         genCap: 16_384)
        let on11 = LaneBackend.sizePlan(promptLen: 24_000, maxTokens: 3,
                                        ctxMax: eligibilityCtx(budgetSchedOn: true, ctx: 262_144),
                                        genCap: 16_384)
        let c11 = off11.oversized == true && on11.oversized == false
        return [
            ("unset_maxtokens", c1),
            ("explicit_capped", c2),
            ("prompt_too_big", c3),
            ("legacy_shape", c4),
            ("legacy_arena_fixed_cap", c5),
            ("legacy_arena_respects_ctx", c6),
            ("budgeted_arena_per_request", c7),
            ("flagoff_eligibility_matches_arena", c8),
            ("flagon_eligibility_unchanged", c9),
            ("flagoff_arena_covers_eligibility", c10),
            ("flagoff_24k_refused_visibly", c11),
        ]
    }

    /// WS-B Stage B: the arena ALLOCATION size for one admission. `seqBudget > 0` is the
    /// budgeted path's per-request size, clamped to the eligibility cap. `seqBudget == 0`
    /// is the LEGACY SENTINEL and keeps the OLD FIXED CAP — `maxSeqLen` became the
    /// model-context eligibility cap (262144) in Stage B, so reusing it here allocated a
    /// 16x arena (335MB → 5.37GB per lane) on a path `canAdmit` never gates. See notes/22
    /// addendum finding 3 for the measurement.
    public static let legacyArenaCap = 16_384
    public static func arenaSeqLen(seqBudget: Int, maxSeqLen: Int) -> Int {
        seqBudget > 0 ? Swift.min(seqBudget, maxSeqLen)
                      : Swift.min(legacyArenaCap, maxSeqLen)
    }

    /// The ELIGIBILITY cap that must pair with `arenaSeqLen`'s ALLOCATION cap (#151).
    /// Stage B split the two and only the budgeted path re-coupled them (its `seqBudget`
    /// IS the allocation size). The flag-off path allocates at `legacyArenaCap` via the
    /// legacy sentinel, so admitting anything longer produced a `.failed` admitStep →
    /// `ContinuousBatch.loop()`'s `req.finish()` → an empty 200 with no NOTE, for every
    /// prompt over 16K. Capping eligibility to match routes those prompts to `sizePlan`'s
    /// EXISTING visible refusal instead. Keep this the only source of that pairing: the
    /// escape hatch's arena stays bit-for-bit legacy (d4f7e27), it just stops over-promising.
    public static func eligibilityCtx(budgetSchedOn: Bool, ctx: Int) -> Int {
        budgetSchedOn ? ctx : Swift.min(ctx, legacyArenaCap)
    }

    /// Fresh lane forward with the canonical hybrid wiring (same trios as TellRuntime's
    /// hybrid setup). maxM 1024 = the canonical steel-hybrid prefill chunk (the shipped
    /// serialize path prefills hybrid@1024; the canonical greedy stream is defined WITH
    /// it — raw chunked prefill produces the pre-hybrid stream and diverges ~100 tokens
    /// in on f16 near-ties). Costs ~200MB scratch per active lane, resident tier.
    private func makeLane(hybrid: Bool, seqLen: Int) -> (SeedlessFusedVerify.SeedlessFusedForward, MTLBuffer)? {
        guard let (fwd, fnBuf) = engine.makeFused(maxM: 1024, maxSeqLen: seqLen) else { return nil }
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

    /// WS-B Stage B (notes/22): wired KV-cache bytes per token per active lane. attn layers
    /// only (GDN layers carry conv/rec state, not a per-token growing cache) — numKV=2,
    /// headDim=256 are this model's fixed attention shape (same constants used throughout
    /// the engine, e.g. QwispModel/AttentionLayer). f16 K + f16 V = 2 * 2 bytes/element.
    func kvBytesPerToken() -> Int {
        let numKV = 2, headDim = 256
        return engine.layers.reduce(0) { sum, spec in
            spec.attn != nil ? sum + numKV * headDim * 2 * 2 : sum
        }
    }

    /// WS-B Stage B aggregate memory gate (the PR #135 collapse guard): can a request needing
    /// `seqBudget` arena tokens be admitted without blowing the lane KV byte budget? Ceil-grant
    /// against the bytes ALREADY committed by other slots' allocated arenas (see `arenaBytes`).
    public func canAdmit(promptLen: Int, seqBudget: Int) -> Bool {
        let committed = arenaBytes.reduce(0, +) + reservedBytes
        let candidate = Swift.min(seqBudget, maxSeqLen) * kvBytesPerToken()
        guard committed + candidate <= kvBudgetBytes else { return false }
        reservedBytes += candidate   // hold until this candidate's admitStep resolves it
        return true
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

    /// Legacy 3-arg entry point: size the arena at the legacy fixed cap (`seqBudget: 0` is
    /// the LEGACY SENTINEL, not oversized/invalid — WS-B Stage B, notes/22). See
    /// `arenaSeqLen` for why that cap is NOT `maxSeqLen` any more.
    public func admitStep(prompt: [Int32], slot: Int, tokenBudget: Int) -> AdmitProgress {
        admitStep(prompt: prompt, slot: slot, tokenBudget: tokenBudget, seqBudget: 0)
    }

    /// Resumable, ctx-adaptive prefill (WS-B Stage A notes/21 + Stage B notes/22): advances
    /// `slot`'s prefill by at most `tokenBudget` tokens, always by ≥1 legal chunk
    /// (forward-progress guarantee — see the `!first` guard below), pausing only at the same
    /// internal chunk / `captureAt` boundaries the old atomic loop would stop at. Setup
    /// (arena sizing + restore-plan lookup + lane creation) happens once per admission, on the
    /// first call for a slot; `seqBudget` (the caller's `sizePlan` result) sizes THIS lane's
    /// arena, clamped to `maxSeqLen` (now the model-context ELIGIBILITY cap, not the allocation
    /// size) — `0` falls back to `legacyArenaCap` (see `arenaSeqLen`).
    public func admitStep(prompt: [Int32], slot: Int, tokenBudget: Int, seqBudget: Int) -> AdmitProgress {
        guard slot >= 0, slot < slotCount, !prompt.isEmpty else { return .failed }
        var st: Prefill
        if let existing = prefills[slot] {
            st = existing
        } else {
            var seqLen = LaneBatchSlots.arenaSeqLen(seqBudget: seqBudget, maxSeqLen: maxSeqLen)
            // Release this candidate's canAdmit hold (same formula, symmetric) now that its
            // first-call setup is actually starting — regardless of whether setup goes on to
            // succeed or fail below, the reservation is resolved either way. seqBudget == 0
            // (legacy sentinel / the 3-arg atomic path) never went through canAdmit, so nothing
            // to release.
            if seqBudget > 0 {
                reservedBytes = Swift.max(0, reservedBytes - seqLen * kvBytesPerToken())
            }
            guard prompt.count < seqLen else { return .failed }
            // Clamp-to-fit: this single request's own arena alone exceeds the aggregate KV
            // budget — shrink it to what fits rather than refusing outright. Only fails if even
            // the clamped arena can't hold the prompt plus ≥1 generated token.
            let perTok = kvBytesPerToken()
            if perTok > 0, seqLen * perTok > kvBudgetBytes {
                seqLen = kvBudgetBytes / perTok
                guard seqLen >= prompt.count + 2 else {
                    FileHandle.standardError.write(Data(
                        "[qwisp] NOTE: lane admit dropped — prompt (\(prompt.count) tokens) doesn't fit the lane KV budget (\(kvBudgetBytes) bytes) even after clamping.\n".utf8))
                    return .failed
                }
            }
            let hybrid = ProcessInfo.processInfo.environment["QWISP_HYBRID_PREFILL"] != "0"
            let chunkSize = hybrid ? 1024 : 64
            guard var (fwd, fnBuf) = makeLane(hybrid: hybrid, seqLen: seqLen) else { return .failed }
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
                        guard let fresh = makeLane(hybrid: hybrid, seqLen: seqLen) else { return .failed }
                        (fwd, fnBuf) = fresh
                        plan.restoreLen = 0
                    }
                }
            }
            // Ceil-grant (WS-B Stage B, notes/22 B2.3): commit the arena's bytes as soon as it's
            // actually allocated, not when the admit finally completes — the self-check's
            // MemFake deliberately uses floor-grant (commits on .done) for a simpler test model;
            // do not "fix" one to match the other.
            arenaBytes[slot] = seqLen * perTok
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
        // One prefill chunk. Extracted so the whole body can optionally run inside an
        // autoreleasepool (#148 probe below) — returns false on engine error.
        func runChunk() -> Bool {
            var end = Swift.min(st.pos + st.chunkSize, prompt.count)
            if let c = st.plan.captureAt, st.pos < c { end = Swift.min(end, c) }
            let x = engine.embed(tokens: Array(prompt[st.pos ..< end]))
            guard let normed = st.hybrid ? st.fwd.forwardRowsHybrid(x, M: end - st.pos, finalNormW: st.fnBuf)
                                          : st.fwd.forwardRows(x, M: end - st.pos, finalNormW: st.fnBuf)
            else { return false }
            st.lastNormed = normed[end - st.pos - 1]
            consumed += end - st.pos
            st.pos = end
            first = false
            // Recurrence boundary state (blob = CPU memcpy of KV used slice + GDN state,
            // ~24KB/token) — captured mid-prefill at the exact shared-prefix end.
            if st.pos == st.plan.captureAt {
                sharedStore.save(tokens: Array(prompt[0 ..< st.pos]), state: st.fwd.persistentStateData())
            }
            return true
        }
        // ── #148 FIX: drain the prefill's autoreleased Metal wrappers every chunk ────────
        // The steel-hybrid prefill wraps every per-layer MLX temporary in a noCopy MTLBuffer
        // (`arrToBuf` → `asMTLBuffer`), ~2.29MB per prompt token by static count. Those
        // wrappers are autoreleased, and the decode thread's autorelease pool drains only at
        // THREAD EXIT — which for ContinuousScheduler is when the queue fully drains. So a
        // whole prefill's wrappers accumulate live: measured law
        //     retained Metal ≈ concurrency × promptLen × ~2.03MB/token
        // (fits both the vary and fixeduniq A/B to within ~100MB at every point). At agentic
        // prompt sizes × lanes that is tens of GB simultaneously, which is what killed the
        // trial server on a 14.9MB allocation. NOT a permanent leak — footprint sawtooths
        // back to a constant floor each round — but the in-round peak is fatal on its own.
        //
        // Draining per chunk bounds it to one chunk's wrappers. Measured (vary, 10 rounds x 2
        // concurrent): peak footprint 46,080 → 24,576MB, growth +20,480 → +1,024MB, and the
        // per-token term vanishes (nonMLXmetal +36,521MB → +310MB, i.e. only the constant
        // driver-wrapper base remains). Pure lifetime change, no math touched: verified
        // byte-identical generation with the flag off vs on (lanes=2, greedy) plus
        // RAWTESTS 97/97 / COMPTEST 93/93 / BENCHBATCHTEST.
        //
        // Default ON. QWISP_LANE_CHUNK_POOL=0 restores the pre-fix behavior for A/B only —
        // it reinstates the OOM, so never run a real server with it off.
        let chunkPool = Tell.envInt("QWISP_LANE_CHUNK_POOL", 1) != 0
        while st.pos < prompt.count {
            if !first, consumed >= tokenBudget { break }
            let ok = chunkPool ? autoreleasepool { runChunk() } : runChunk()
            if !ok { prefills[slot] = nil; return .failed }
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
        arenaBytes[slot] = 0   // WS-B Stage B: free this slot's KV-budget reservation
        batch = nil; batchLanes = []   // active set changed
        reportIdleMemory()
    }

    // ── #148 diagnostic: what actually retains lane memory? ───────────────────
    // footprint alone cannot tell "MLX pooled the freed buffers" from "something still
    // references them" — both read as growth. MLX.Memory splits them: activeMemory =
    // live MLXArrays, cacheMemory = MLX's free-buffer pool. Sampled at a FULLY IDLE
    // point (every lane released) so the only live MLX memory should be model residency.
    //   QWISP_LANE_MEMDBG=1 → observe only (natural active/cache split per idle point)
    //   QWISP_LANE_MEMDBG=2 → observe, clearCache(), observe again (the discriminating
    //                         experiment: does the pool actually give the memory back?)
    // Interpretation: active flat + cache ratcheting + clear drops it ⇒ allocator
    // retention (bound Memory.cacheLimit). active ratcheting ⇒ live references, and
    // cacheLimit is a no-op. Neither ⇒ raw MTLBuffer / CPU Data, MLX is not the holder.
    private var memDbgIdleCount = 0
    private func reportIdleMemory() {
        let mode = Tell.envInt("QWISP_LANE_MEMDBG", 0)
        guard mode > 0 else { return }
        guard lanes.allSatisfy({ $0 == nil }), prefills.allSatisfy({ $0 == nil }) else { return }
        memDbgIdleCount += 1
        func mb(_ b: Int) -> String { String(format: "%.0fMB", Double(b) / 1_048_576) }
        let before = MLX.Memory.snapshot()
        // metal = EVERY Metal allocation on this device, MLX's included. The lane path also
        // allocates raw MTLBuffers directly (KV arena via makeKVCacheBufs, the maxM=1024
        // attn/gdn/moe scratch, per-lane staging) which MLX neither tracks nor pools, so
        // `metal − (active+cache)` is the non-MLX Metal footprint. Needed because the
        // 2026-07-27 vary run showed footprint ratcheting to 47GB while MLX accounted for
        // only 21GB — a ~26GB hole that Memory.cacheLimit cannot reach.
        let metal = driver.device.currentAllocatedSize
        var line = "[lane-memdbg] idle#\(memDbgIdleCount) active=\(mb(before.activeMemory))"
            + " cache=\(mb(before.cacheMemory)) peak=\(mb(before.peakMemory))"
            + " metal=\(mb(metal)) nonMLXmetal=\(mb(metal - before.activeMemory - before.cacheMemory))"
        if mode >= 2 {
            MLX.Memory.clearCache()
            let after = MLX.Memory.snapshot()
            line += " | after clearCache: active=\(mb(after.activeMemory)) cache=\(mb(after.cacheMemory))"
        }
        FileHandle.standardError.write(Data((line + "\n").utf8))
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
    let genCap: Int
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
        // WS-B Stage B (notes/22): QWISP_LANE_CTX is an OVERRIDE-only knob now — unset means
        // "use the full model context" (ctx-adaptive sizing below decides the real per-request
        // arena), explicit means "never exceed this, even if the model context is larger".
        // Tell.envInt's default-fallback can't distinguish unset-vs-explicit, so this one read
        // goes straight through ProcessInfo. All three env reads for this backend live ONLY
        // here — never inside ContinuousScheduler or LaneBatchSlots — so their self-checks stay
        // deterministic.
        let modelCtx = SeedlessBackend.readContextLen(modelDir)
        let ctx: Int
        if let raw = ProcessInfo.processInfo.environment["QWISP_LANE_CTX"], let explicit = Int(raw) {
            ctx = Swift.min(modelCtx, Swift.max(2048, explicit))
        } else {
            ctx = modelCtx
        }
        let genCap = Swift.max(1024, Tell.envInt("QWISP_LANE_GEN_MAX", 16384))
        // KV budget: the PR #135 collapse guard (aggregate lane arenas can otherwise blow past
        // wired memory on large-context concurrent admits). 8GB default on ≥48GB boxes, 4GB
        // below — plenty of headroom either way once resident expert weights are accounted for.
        let kvBudgetBytes = Tell.envInt("QWISP_LANE_KV_MB",
            DeviceCalibration.physicalRAMGB() >= 48 ? 8192 : 4096) * 1_048_576
        guard let laneSlots = LaneBatchSlots(store: store, slots: slots, maxSeqLen: ctx,
                                             kvBudgetBytes: kvBudgetBytes) else {
            throw NSError(domain: "qwisp", code: 7, userInfo: [NSLocalizedDescriptionKey:
                "lane batching: engine build failed"])
        }
        self.slots = slots
        // #151: eligibility must match the arena the SELECTED path will actually allocate,
        // so the flag decision has to be read before laneCtx is fixed. `laneSlots` above
        // keeps the full `ctx` as its maxSeqLen — that is the allocation CLAMP, and the
        // legacy sentinel narrows it to legacyArenaCap on its own.
        let budgetSchedOn = Tell.envInt("QWISP_TOKEN_BUDGET_SCHED", 1) != 0
        self.laneCtx = LaneBatchSlots.eligibilityCtx(budgetSchedOn: budgetSchedOn, ctx: ctx)
        self.genCap = genCap
        // WS-B Stage A (notes/21): token-budget admission scheduler. Default ON as of the
        // GO-bar bench (notes/21 "Bench verification results", 2026-07-25): a 24K-token
        // concurrent admit's worst-case stall on another lane's decode stream drops from
        // an unbounded ~93.7s to a depth-bounded ~13.5s. QWISP_TOKEN_BUDGET_SCHED=0 opts
        // back out to the old atomic drain-then-step. Gate/size env reads live here only,
        // never inside the scheduler or LaneBatchSlots, so their self-checks stay
        // deterministic.
        let budget = budgetSchedOn ? Swift.max(1, Tell.envInt("QWISP_TOKEN_BUDGET", 2048)) : 0
        self.scheduler = ContinuousScheduler(slots: laneSlots, tokenBudget: budget)
        // Bound MLX's free-buffer pool. This is NOT the #148 leak fix (that is the per-chunk
        // autoreleasepool in admitStep) — it is the SECOND consumer, which only became
        // visible once the wrapper leak was gone and the crash configuration was measured
        // directly (lanes=4, prefix store ON, 6-15K prompts). There, post-fix:
        //   unbounded: cacheMemory 11,903 → 26,361MB, footprint peak 49,152MB (+26,624MB),
        //              ttft mean 49,546ms / total 991s / max 117,017ms
        //   2GB bound: cacheMemory pinned ~2,050MB, footprint peak 26,624MB (+3,072MB),
        //              ttft mean 44,688ms / total 894s / max 92,534ms
        // i.e. −22.5GB peak AND ~10% FASTER (max ttft −21%). The pool is not free at this
        // size: 26GB of pool on top of 19GB of resident weights puts a 64GB box into memory
        // pressure, which is what the ttft difference is. MLX's own docs recommend a lower
        // limit for long inference runs. Lane-scoped (LaneBackend is only built under
        // QWISP_LANES); the serialize path is untouched. 0 disables the bound. Setting the
        // limit does not purge what is already pooled, hence the clearCache().
        let mlxCacheMB = Swift.max(0, Tell.envInt("QWISP_LANE_MLX_CACHE_MB", 2048))
        if mlxCacheMB > 0 {
            MLX.Memory.cacheLimit = mlxCacheMB * 1_048_576
            MLX.Memory.clearCache()
        }
    }

    /// Ctx-adaptive sizing policy (WS-B Stage B, notes/22 Contract B1): from a request's
    /// prompt length and requested maxTokens, decide the per-admit gen budget and the arena
    /// size (seqBudget). `oversized` = the prompt leaves no room for even one generated token
    /// (headroom ≤ 0); the caller must check it FIRST and never submit — the `cachedGenBudget`
    /// floor of `max(1, …)` would otherwise silently succeed (notes/22 Subtlety 1).
    public static func sizePlan(promptLen: Int, maxTokens: Int, ctxMax: Int, genCap: Int)
        -> (genBudget: Int, seqBudget: Int, oversized: Bool) {
        let headroom = ctxMax - promptLen - 1
        guard headroom > 0 else { return (genBudget: 0, seqBudget: 0, oversized: true) }
        let ceiling = maxTokens < 0 ? headroom : Swift.min(maxTokens, headroom)
        let genBudget = SeedlessBackend.cachedGenBudget(promptLen: promptLen, ceiling: ceiling,
                                                         arenaMax: ctxMax, genCap: genCap)
        return (genBudget: genBudget, seqBudget: promptLen + 1 + genBudget, oversized: false)
    }

    public func generate(_ prompt: [Int], options: GenerateOptions) -> AsyncStream<Int> {
        let plan = LaneBackend.sizePlan(promptLen: prompt.count, maxTokens: options.maxTokens,
                                        ctxMax: laneCtx, genCap: genCap)
        guard !plan.oversized else {
            FileHandle.standardError.write(Data(
                "[qwisp] NOTE: prompt (\(prompt.count) tokens) leaves no room to generate in the \(laneCtx)-token lane context — request dropped.\n".utf8))
            return AsyncStream { $0.finish() }
        }
        return scheduler.submit(prompt: prompt, maxTokens: plan.genBudget, stopIds: options.stopTokens)
    }
}
