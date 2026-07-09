import Foundation
import Metal
import MLX
import MLXFast

/// U2b: raw-spec — SuffixSpec speculative loop on the SELF-CONSISTENT raw engine.
/// Resident tier (C=256 semantics): full expert tensors from store, no arena/guard/cert/VSEQ.
/// Batched verify == sequential by construction (proven by raw-smoke U2a) => strict lossless.
///
/// QWISP_RUN=raw-spec  QWISP_GEN=48  QWISP_DRAFT_K=96
/// QWISP_RAW_FUSED=1       — P3 fused backend(全層+final norm を 1 CB、cache GPU 常駐)
/// QWISP_RAW_C=<int>       — streaming tier(C < 256); implies fused if QWISP_RAW_FUSED not set
/// QWISP_RAW_BOLT=1        — bolt mode(QWISP_RAW_C 必須): calib → freeze → near-lossless 1-CB
/// QWISP_RAWSPEC_CHECK=1   — also run pure sequential raw greedy and report spec-vs-greedy k/N
/// QWISP_RAWSTREAM_CHECK=1 — (strict streaming only) compare stream output vs resident; needs ~20GB+
/// QWISP_DUMP_TOKENS=1     — print PROMPT_TOKENS / OUT_TOKENS lines (bench correctness axis)
extension Tell {

    /// notes/14 TODO-2: bolt の rolling re-calib R と async refresh chunk B を workload 名で
    /// 実証済み最適値へ切り替える opt-in preset(QWISP_BOLT_WORKLOAD)。純関数。
    /// 明示 env(QWISP_BOLT_RECALIB_R/QWISP_BOLT_REFRESH_B)が常にこの preset を上書きする。
    static func boltWorkloadPreset(_ w: String) -> (r: Int, b: Int) {
        switch w {
        case "code":    return (128, 64)
        case "agentic": return (256, 32)
        case "shortnl": return (128, 16)
        default:        return (128, 32)  // "", "longctx", unknown → current default
        }
    }

    /// notes/11 レバー②(measure-first): bolt の cold-selection(miss)を「どの gate weight /
    /// top-8 内 rank で起きるか」で特徴づける純関数。novice が効くのは低 gate の tail miss か、
    /// top miss 支配なら別処方かの go/no-go 数を出す。
    ///
    /// - Parameters は step×層ごとに層別に並べた観測列(caller が並べて渡す):
    ///   `topInds[t]` = 観測 t の top-8 expert index、`topWeights[t]` = 対応する正規化 gate weight
    ///   (norm_topk: top-8 内で和 1、降順とは限らない)、`resident[t]` = その層の常駐集合。
    /// - miss = `topInds[t][k]` が `resident[t]` に無い選択。その weight を収集。
    /// - p10/p50/p90 = miss weight の分位点(昇順 sort、index = floor(q*(n-1)))。
    /// - top1Share = miss のうち「その観測 t の top-8 内で weight 最大位(rank-1)」だった割合(0..1)。
    /// - meanMissMass = 観測 t あたりの miss weight 合計の平均(全観測数 = topInds.count で割る)。
    /// - missCount = 総 miss 数。miss ゼロなら数値は全て 0。
    ///
    /// GREEN: quantile/share/mass computation for bolt cold-selection analysis.
    static func missWeightStats(topInds: [[Int]], topWeights: [[Float]], resident: [Set<Int>])
        -> (p10: Float, p50: Float, p90: Float, top1Share: Float, meanMissMass: Float, missCount: Int) {
        let obsCount = topInds.count
        guard obsCount > 0 else { return (0, 0, 0, 0, 0, 0) }
        var missWeights: [Float] = []
        var top1Misses = 0
        var totalMissMass: Float = 0
        for t in 0 ..< obsCount {
            let inds = topInds[t]
            let weights = topWeights[t]
            let res = t < resident.count ? resident[t] : Set<Int>()
            let maxW = weights.max() ?? 0
            for k in 0 ..< inds.count {
                guard !res.contains(inds[k]) else { continue }
                let w = k < weights.count ? weights[k] : 0
                missWeights.append(w)
                totalMissMass += w
                if w == maxW { top1Misses += 1 }
            }
        }
        let missCount = missWeights.count
        guard missCount > 0 else { return (0, 0, 0, 0, 0, 0) }
        missWeights.sort()
        let n = missWeights.count
        let p10 = missWeights[Int(Float(0.1) * Float(n - 1))]
        let p50 = missWeights[Int(Float(0.5) * Float(n - 1))]
        let p90 = missWeights[Int(Float(0.9) * Float(n - 1))]
        return (p10: p10, p50: p50, p90: p90,
                top1Share: Float(top1Misses) / Float(missCount),
                meanMissMass: totalMissMass / Float(obsCount),
                missCount: missCount)
    }

    /// spec ループが必要とする engine 操作の抽象(composed / fused の 2 実装)。
    /// forward: tokens → 最終 norm 済み hidden [M, H]。
    /// stepArgmax: tokens → 行毎 greedy argmax(cache も前進)。1-CB 実装が無ければ
    /// forward+logits+MLX argMax で合成される(makeStepArgmax)。
    /// snapshot/rollback: 直後の forward/stepArgmax「1 回だけ」を取り消す(partial reject 用)。
    struct SpecBackend {
        let forward: ([Int32]) -> MLXArray?
        let stepArgmax: ([Int32]) -> [Int]?
        let snapshot: () -> Any
        let rollback: (Any) -> Void
        var chainedStepArgmax: ((Int32, Int) -> [Int]?)? = nil   // Phase II-a
    }

    /// forward+lm_head+MLX argMax による stepArgmax 合成(composed 用 fallback)。
    static func makeStepArgmax(engine: SeedlessEngine, forward: @escaping ([Int32]) -> MLXArray?) -> ([Int32]) -> [Int]? {
        return { tokens in
            guard let n = forward(tokens), let l = engine.logits(n, M: tokens.count) else { return nil }
            MLX.eval([l])
            return (0 ..< tokens.count).map { MLX.argMax(l[$0], axis: -1).item(Int.self) }
        }
    }

    /// composed backend(per-op CB、MLX cache 参照 snapshot)。
    static func composedBackend(engine: SeedlessEngine) -> SpecBackend {
        let caches = engine.freshCaches()
        let fwd: ([Int32]) -> MLXArray? = { tokens in
            let x = engine.embed(tokens: tokens)
            return engine.forwardRows(x, caches: caches, M: tokens.count)
        }
        return SpecBackend(
            forward: fwd,
            stepArgmax: makeStepArgmax(engine: engine, forward: fwd),
            snapshot: { caches.map { $0.copyState() } },
            rollback: { snap in
                guard let snaps = snap as? [SeedlessVerifyForward.LayerCaches] else { return }
                for (i, s) in snaps.enumerated() {
                    caches[i].kCache = s.kCache; caches[i].vCache = s.vCache
                    caches[i].convState = s.convState; caches[i].recState = s.recState
                }
            })
    }

    /// fused backend(1-CB step: embed→40層→final norm→lm_head→argmax、readback は token id のみ。
    /// rollback は KV len 巻き戻し+ping-pong swap)。
    static func fusedBackend(engine: SeedlessEngine, maxM: Int, maxSeqLen: Int) -> SpecBackend? {
        return fusedBackendWithFwd(engine: engine, maxM: maxM, maxSeqLen: maxSeqLen)?.0
    }

    /// fusedBackend + 内部 SeedlessFusedForward の handle も返す変種(MTP-D1 draft が
    /// normedBuffer を GPU-bind するために必要。streamingBackend と同形)。
    static func fusedBackendWithFwd(engine: SeedlessEngine, maxM: Int, maxSeqLen: Int)
        -> (SpecBackend, SeedlessFusedVerify.SeedlessFusedForward)? {
        guard let (fwd, fnBuf) = engine.makeFused(maxM: maxM, maxSeqLen: maxSeqLen) else { return nil }
        let forward: ([Int32]) -> MLXArray? = { tokens in
            let x = engine.embed(tokens: tokens)
            return fwd.forwardRows(x, M: tokens.count, finalNormW: fnBuf)
        }
        let step: ([Int32]) -> [Int]?
        if fwd.head != nil {
            step = { tokens in fwd.stepArgmax(tokens) }
        } else {
            step = makeStepArgmax(engine: engine, forward: forward)
        }
        var backend = SpecBackend(
            forward: forward,
            stepArgmax: step,
            snapshot: { fwd.snapshot() },
            rollback: { snap in
                guard let s = snap as? SeedlessFusedVerify.SeedlessFusedForward.Snapshot else { return }
                fwd.rollbackOneStep(s)
            })
        // Phase II-a: wire chained greedy decode when the 1-CB head path is active.
        // Only resident/bolt return non-nil (strict returns nil → per-step fallback).
        if fwd.head != nil {
            backend.chainedStepArgmax = { token, k in fwd.chainedStepArgmax(token, K: k) }
        }
        return (backend, fwd)
    }

    /// streaming fused backend(strict segmented per-layer CB)。
    /// existingProviders を渡すと arena を再利用し fresh forward のみ構築する(bolt phase 2 用)。
    static func streamingBackend(engine: SeedlessEngine, modelDir: String, maxM: Int, maxSeqLen: Int, C: Int,
                                  existingProviders: [ArenaExpertProvider]? = nil)
        -> (SpecBackend, SeedlessFusedVerify.SeedlessFusedForward, [ArenaExpertProvider])? {
        guard let (fwd, fnBuf, providers) = engine.makeFusedStreaming(
            modelDir: modelDir, maxM: maxM, maxSeqLen: maxSeqLen, C: C,
            existingProviders: existingProviders) else { return nil }
        let forward: ([Int32]) -> MLXArray? = { tokens in
            let x = engine.embed(tokens: tokens)
            return fwd.forwardRows(x, M: tokens.count, finalNormW: fnBuf)
        }
        let step: ([Int32]) -> [Int]?
        if fwd.head != nil {
            step = { tokens in fwd.stepArgmax(tokens) }
        } else {
            step = makeStepArgmax(engine: engine, forward: forward)
        }
        var backend = SpecBackend(
            forward: forward,
            stepArgmax: step,
            snapshot: { fwd.snapshot() },
            rollback: { snap in
                guard let s = snap as? SeedlessFusedVerify.SeedlessFusedForward.Snapshot else { return }
                fwd.rollbackOneStep(s)
            })
        // Phase II-a: wire chained greedy decode when the 1-CB head path is active.
        if fwd.head != nil {
            backend.chainedStepArgmax = { token, k in fwd.chainedStepArgmax(token, K: k) }
        }
        return (backend, fwd, providers)
    }

    // ── Phase II-b: bolt-tier chain wiring seam ─────────────────────────────
    //
    // The BOLT decode loop (runBoltMode, phase 6) is a self-contained spec loop that,
    // in its D==0 greedy span, calls `backend.stepArgmax([u])` once per token — it never
    // touches the chain path, so QWISP_CHAIN_K has no effect on bolt runs (they are pure
    // per-step greedy after freeze). This is the highest-benefit chain regime.
    //
    // `boltGreedyChainSpan` is the shared seam the bolt loop delegates its D==0 span to.
    // Contract (mirrors the standard-path chain block in `run`, lines ~358–368):
    //   • When chainK > 0 AND backend.chainedStepArgmax is non-nil (bolt tables active →
    //     1-CB head path → non-nil chain), it calls chainedStepArgmax(u, chainK) and packs
    //     the result as: emitted = [u] + chainResult[0 ..< min(chainResult.count-1, budget-1)],
    //     nextU = chainResult.last.  (`budget` caps how many tokens the span may emit,
    //     including u, so the caller stays within its remaining N.)
    //   • Because bolt chainedStepArgmax is bit-exact to `chainK` sequential stepArgmax
    //     calls, emitted + [nextU] is byte-identical to [u] + K sequential greedy tokens.
    //     Bolt is deterministic buddy-greedy → chain MUST NOT change OUT_TOKENS.
    //   • Returns nil when chainK <= 0 or the backend exposes no chain fn (caller then
    //     falls back to per-step) OR on backend error.
    //
    // Note: strict streaming intentionally exposes NO chain fn (SeedlessFusedForward.
    // chainedStepArgmax guards `streamMode == .resident || .bolt`), because strict needs
    // a CPU turn between steps to ensure/pread the next expert union — so this seam
    // returns nil there and the strict path stays per-step by construction.
    //
    // Phase II-b implementation: delegate the D==0 greedy span to chainedStepArgmax when
    // chainK>0 and the backend exposes the chain fn (bolt tables active → non-nil).
    // Packing: emitted = [u] + chainResult[0 ..< min(chainResult.count-1, budget-1)],
    //          nextU   = chainResult.last.
    // Returns nil when chainK<=0, no chain fn, or the chain call fails (caller falls back
    // to per-step greedy). Strict streaming returns nil here by construction because its
    // chainedStepArgmax is never wired (strict needs a CPU turn between steps to ensure/pread
    // the next expert union).
    static func boltGreedyChainSpan(backend: SpecBackend, u: Int, chainK: Int, budget: Int)
        -> (emitted: [Int], nextU: Int)? {
        guard chainK > 0, let chainFn = backend.chainedStepArgmax else { return nil }
        guard let chainResult = chainFn(Int32(u), chainK), !chainResult.isEmpty else { return nil }
        let tailEnd = Swift.min(chainResult.count - 1, budget - 1)
        let emitted = [u] + Array(chainResult[0 ..< tailEnd])
        let nextU = chainResult.last!
        return (emitted: emitted, nextU: nextU)
    }

    // ── MTP-D1 draft seam (①③ Step 4, notes/17) ─────────────────────────
    //
    // Fills the suffix-draftless (D==0) span with a 1-token draft from the raw
    // MTPHead, mirroring Tell.swift's D==0 MTP block. The draft is verified by
    // the normal batched-verify path, so any wrong draft is rejected → lossless
    // by construction. Contract:
    //   • head nil (QWISP_MTP_DRAFT unset / non-resident tier) → nil, caller keeps
    //     the pre-existing greedy path → flag-off byte-identity by construction.
    //   • rowOfU < 0 (no valid hidden row, e.g. after a chained span) → nil.
    //   • Otherwise: head.draftArgmax on normed row `rowOfU` (the post-final-norm
    //     hidden that produced u — Step 2 normedBuffer accessor, VOLATILE: valid
    //     only until the next forward). READ-ONLY on head KV (len unchanged,
    //     locked test 75); feedPairs (Step 5) is the sole writer.
    static func mtpDraftSpan(head: SeedlessFusedVerify.SeedlessMTPHead?, hPrevBuf: MTLBuffer?,
                             rowOfU: Int, u: Int) -> Int? {
        guard let head, let hPrevBuf, rowOfU >= 0 else { return nil }
        return head.draftArgmax(hPrevBuf: hPrevBuf, hPrevRow: rowOfU, tok: Int32(u))
    }

    // ── Prefill helper ────────────────────────────────────────────────────

    /// Chunked prefill: runs all prompt tokens through the backend, chunk=64.
    /// Returns normed hidden of the very last position [1, H], or nil on error.
    /// mtpHead/mtpFwd (①③ Step 5): ingest prompt pairs (h_i, id_{i+1}) per chunk —
    /// pLen-1 pairs total; the final position's pair (h_last, u) is fed by the spec
    /// loop's pre-verify feed once u is known (KV-read draft discipline, notes/17).
    static func prefill(promptIds: [Int32], backend: SpecBackend,
                        mtpHead: SeedlessFusedVerify.SeedlessMTPHead? = nil,
                        mtpFwd: SeedlessFusedVerify.SeedlessFusedForward? = nil) -> MLXArray? {
        let pLen = promptIds.count
        guard pLen > 0 else { return nil }
        let chunkSize = 64
        var lastNormed: MLXArray? = nil
        var pos = 0
        while pos < pLen {
            let end = Swift.min(pos + chunkSize, pLen)
            let chunk = Array(promptIds[pos ..< end])
            guard let normed = backend.forward(chunk) else { return nil }
            if let head = mtpHead, let fwd = mtpFwd {
                // Non-final chunk: k pairs (last row pairs with the next chunk's head
                // token). Final chunk: k-1 pairs (h_last has no committed successor yet).
                let nPairs = (end < pLen) ? chunk.count : chunk.count - 1
                if nPairs > 0 {
                    let toks = Array(promptIds[(pos + 1) ..< (pos + 1 + nPairs)])
                    _ = head.feedPairs(hBuf: fwd.normedBuffer, rowRange: 0 ..< nPairs, toks: toks)
                }
            }
            // Keep last row [H] — will be overwritten each chunk until the final one.
            lastNormed = normed[chunk.count - 1]    // [H]
            pos = end
        }
        return lastNormed.map { $0.reshaped([1, SeedlessEngine.H]) }   // [1, H]
    }

    // ── Spec loop helper ──────────────────────────────────────────────────

    /// SuffixSpec ループ本体(main run / self-check / stream-vs-resident check の 3 箇所で共用)。
    /// Returns out[0..<N] token ids.
    static func runSpecLoop(promptIds: [Int32], backend: SpecBackend, engine: SeedlessEngine,
                             N: Int, maxK: Int, useA3: Bool = false,
                             onToken: ((Int) -> Void)? = nil) -> [Int]? {
        guard let lastNormed = prefill(promptIds: promptIds, backend: backend) else { return nil }
        guard let lg0 = engine.logits(lastNormed, M: 1) else { return nil }
        MLX.eval([lg0])
        var u = MLX.argMax(lg0[0], axis: -1).item(Int.self)

        var hist = promptIds.map { Int($0) }
        var out: [Int] = []
        var pending: [Int] = []  // A3: pending prefix tokens
        let pendingCap = 24
        // Incremental streaming seam: onToken nil → zero behavior change (out/return unchanged).
        var streamed = 0
        func flush() { if let onToken { while streamed < out.count { onToken(out[streamed]); streamed += 1 } } }

        while out.count < N {
            flush()
            let drafts = Tell.suffixDraft(hist + [u], maxMatch: 32, draftK: maxK, minMatch: 4)
            let D      = drafts.count

            var snap = backend.snapshot()

            if D == 0 {
                if useA3 && !pending.isEmpty {
                    // A3 D==0: stepArgmax on [pending, u] (batched)—ONE forward to realize pending
                    let pk = pending.count
                    let stepTokens: [Int32] = pending.map { Int32($0) } + [Int32(u)]
                    guard let evals = backend.stepArgmax(stepTokens) else { return nil }
                    out.append(u); hist.append(u)
                    u = evals[pk]  // next token after pending+u (pk = position of u)
                    pending = []
                } else {
                    guard let evals = backend.stepArgmax([Int32(u)]) else { return nil }
                    out.append(u); hist.append(u)
                    u = evals[0]
                }
                continue
            }

            if useA3 {
                // A3: fuse pending + [u] + drafts into ONE verify batch (no flush-before-verify)
                let pk = pending.count
                let verifyTokens: [Int32] = pending.map { Int32($0) } + [Int32(u)] + drafts.map { Int32($0) }
                guard let evals = backend.stepArgmax(verifyTokens) else { return nil }

                // Decision row offset: evals[pk] = prediction after u (row pk in fused batch).
                // drafts[p] should equal evals[pk + p] (mirror of TellBolt's evals[p] after
                // slicing vlg[0, pk ..< pk+D+1]).  The spec document wrote pk+1+p but that is
                // an off-by-one: the u-row IS the comparison origin, not a skip.
                var p = 0
                while p < D && drafts[p] == evals[pk + p] { p += 1 }

                if p == D {
                    // A3 full accept: fused forward already advanced cache to B+pk+1+D
                    out.append(u); hist.append(u)
                    for d in drafts { out.append(d); hist.append(d) }
                    pending = []
                    u = evals[pk + D]
                } else {
                    // A3 partial reject: rollback to B (before pending was realized), re-add to pending
                    backend.rollback(snap)
                    out.append(u); hist.append(u)
                    for d in drafts.prefix(p) { out.append(d); hist.append(d) }
                    pending.append(u)
                    for d in drafts.prefix(p) { pending.append(d) }
                    u = evals[pk + p]

                    // Cap flush: only if pending exceeds 24 (safety to bound M)
                    if pending.count >= pendingCap {
                        let pendingTokens: [Int32] = pending.map { Int32($0) }
                        guard let _ = backend.forward(pendingTokens) else { return nil }
                        pending = []
                    }
                }
            } else {
                // Non-A3 path (unchanged)
                let verifyTokens: [Int32] = [Int32(u)] + drafts.map { Int32($0) }
                guard let evals = backend.stepArgmax(verifyTokens) else { return nil }

                var p = 0
                while p < D && drafts[p] == evals[p] { p += 1 }

                if p == D {
                    out.append(u); hist.append(u)
                    for d in drafts { out.append(d); hist.append(d) }
                    u = evals[D]
                } else {
                    backend.rollback(snap)
                    out.append(u); hist.append(u)
                    for d in drafts.prefix(p) { out.append(d); hist.append(d) }
                    let rebuildTokens: [Int32] = [Int32(u)] + drafts.prefix(p).map { Int32($0) }
                    guard let _ = backend.forward(rebuildTokens) else { return nil }
                    u = evals[p]
                }
            }
        }
        flush()
        return Array(out.prefix(N))
    }

    // ── Tier-gating resolution seams (pure/testable) ───────────────────────
    //
    // recon #16 fix: two default-path wiring changes, both lossless (output
    // byte-identical; only default selection differs). Factored as pure helpers
    // so the resolved defaults are unit-testable via the production seam without
    // a real model (SeedlessVerifyTests 52/53). run() MUST call these.
    //
    // STUB (RED): implementer wires the real logic here. Stubs return failing
    // sentinels and MUST NOT delegate to existing env-parsing code.

    /// Resolve the resident `useFused` default from the raw QWISP_RAW_FUSED value
    /// (nil = unset). Contract: unset → true (fused 1-CB ON, the fast default);
    /// "0" → false (composed, debug); any other int → its `!= 0` truth. Mirrors
    /// `Tell.envInt("QWISP_RAW_FUSED", 1) != 0`.
    static func resolveUseFused(env raw: String?) -> Bool {
        guard let raw else { return true }          // unset → fused ON (fast default)
        return (Int(raw) ?? 1) != 0
    }

    /// Resolve raw-spec C from the raw QWISP_RAW_C value (nil = unset) and the
    /// device-tier default. Contract: unset → `defaultC` (RAM tier via
    /// DeviceCalibration.defaultC()); explicit value overrides (incl. "0" = resident).
    static func resolveRawC(envC raw: String?, defaultC: Int) -> Int {
        guard let raw else { return defaultC }      // unset → RAM-tier default
        return Int(raw) ?? defaultC
    }

    // ── Main runner ───────────────────────────────────────────────────────

    /// forceBolt: the default QWISP_BOLT=1 entry routes here with bolt requested (shipping bolt =
    /// raw, not MLX boltCore). Bolt is streaming-only by construction (buddy calib→freeze→1-CB); at
    /// resident tier it degenerates to strict (io=0 already), so a resident forceBolt falls back to
    /// strict raw with a notice. Same effect as env QWISP_RAW_BOLT=1 but without mutating the env.
    public static func run(modelDir: String, refPath: String, forceBolt: Bool = false) throws -> String {
        // ── streaming tier detection ──────────────────────────────────────
        let env = ProcessInfo.processInfo.environment
        let rawC = resolveRawC(envC: env["QWISP_RAW_C"], defaultC: DeviceCalibration.defaultC())
        let isStreaming = rawC > 0 && rawC < 256
        let boltReq = forceBolt || Tell.envFlag("QWISP_RAW_BOLT")
        let isBolt = isStreaming && boltReq
        if boltReq && !isStreaming {
            print("[raw-spec] bolt requested but C=\(rawC) is resident tier (bolt is streaming-only; io=0 at resident) → strict raw")
        }
        let useFused = resolveUseFused(env: env["QWISP_RAW_FUSED"])
        let useA3 = Tell.envFlag("QWISP_RAW_A3")
        if isStreaming && !useFused {
            print("[raw-spec] QWISP_RAW_C=\(rawC) → streaming tier; fused OFF (QWISP_RAW_FUSED=0)")
        }

        print("[raw-spec] loading model from \(modelDir) ...")
        let store = try WeightStore(modelDir: modelDir)
        if isStreaming {
            store.residentNonExperts()
            print("[raw-spec] residentNonExperts complete (streaming tier C=\(rawC))")
        } else {
            store.residentAll()
            print("[raw-spec] residentAll complete")
        }

        print("[raw-spec] building SeedlessEngine ...")
        let engine = SeedlessEngine.build(store: store)
        print("[raw-spec] engine ready (moeI=\(engine.moeI))")

        // ── load ref ─────────────────────────────────────────────────────
        let r = try loadArrays(url: URL(fileURLWithPath: refPath))
        guard let promptArr = r["spec_prompt"], let gRefArr = r["spec_greedy"]
        else { return "[raw-spec] ERROR: no spec_prompt/spec_greedy in \(refPath)" }

        let promptIds: [Int32] = promptArr.asType(.int32).asArray(Int32.self)
        let gRefIds:   [Int]   = gRefArr.asType(.int32).asArray(Int32.self).map { Int($0) }
        let N    = Swift.min(Tell.envInt("QWISP_GEN", 48), gRefIds.count)
        // streaming tier の default maxK: C*3/8(MLX strict と同一の動作点)。resident は 96 のまま。
        let maxK = isStreaming
            ? Tell.envInt("QWISP_DRAFT_K", Swift.max(4, rawC * 3 / 8))
            : Tell.envInt("QWISP_DRAFT_K", 96)
        print("[raw-spec] promptLen=\(promptIds.count) N=\(N) maxK=\(maxK) fused=\(useFused) streaming=\(isStreaming) C=\(rawC) bolt=\(isBolt) a3=\(useA3)")

        // backend 構築(fused: maxM=verify 最大行数, maxSeqLen=prompt+生成+draft+margin)
        // A3: maxM must be at least pendingCap + maxK + 1 to fit fused batches
        let pendingCap = 24
        let maxM = Swift.max(pendingCap + maxK + 1, 64)
        let maxSeqLen = promptIds.count + N + maxK + 64

        // ── BOLT PATH ────────────────────────────────────────────────────
        if isBolt {
            return try runBoltMode(engine: engine, modelDir: modelDir, promptIds: promptIds,
                                   gRefIds: gRefIds, N: N, maxK: maxK, C: rawC,
                                   maxM: maxM, maxSeqLen: maxSeqLen, refPath: refPath)
        }

        // ── STANDARD PATH (resident or strict streaming) ──────────────────
        // reset per-run accounting
        LayerExpertCache.ensureNanos = 0
        LayerExpertCache.preadNanos  = 0
        LayerExpertCache.missTotal   = 0
        LayerExpertCache.chunkTotal  = 0

        // reuse-rerank: capture streaming fwd/providers for expert-aware draft (notes/10)
        var _reuseStreamFwd: SeedlessFusedVerify.SeedlessFusedForward? = nil
        var _reuseStreamProviders: [ArenaExpertProvider]? = nil

        let mkBackend: () -> SpecBackend? = {
            if isStreaming {
                return streamingBackend(engine: engine, modelDir: modelDir, maxM: maxM,
                                        maxSeqLen: maxSeqLen, C: rawC).map { $0.0 }
            } else if useFused {
                return fusedBackend(engine: engine, maxM: maxM, maxSeqLen: maxSeqLen)
            } else {
                return composedBackend(engine: engine)
            }
        }
        // Build main backend; for streaming, capture fwd/providers for optional rerank.
        // For resident fused, capture fwd for the MTP-D1 draft head (normedBuffer bind).
        var _residentFwd: SeedlessFusedVerify.SeedlessFusedForward? = nil
        let backend: SpecBackend
        if isStreaming {
            guard let (b, fwd, providers) = streamingBackend(engine: engine, modelDir: modelDir,
                                                              maxM: maxM, maxSeqLen: maxSeqLen, C: rawC)
            else { return "[raw-spec] ERROR: backend init nil (fused=\(useFused), streaming=\(isStreaming))" }
            _reuseStreamFwd = fwd
            _reuseStreamProviders = providers
            backend = b
        } else if useFused {
            guard let (b, fwd) = fusedBackendWithFwd(engine: engine, maxM: maxM, maxSeqLen: maxSeqLen)
            else { return "[raw-spec] ERROR: backend init nil (fused=\(useFused), streaming=\(isStreaming))" }
            _residentFwd = fwd
            backend = b
        } else {
            guard let b = mkBackend()
            else { return "[raw-spec] ERROR: backend init nil (fused=\(useFused), streaming=\(isStreaming))" }
            backend = b
        }

        // ── MTP-D1 raw draft head (①③ Step 4, notes/17) ───────────────────
        // Opt-in QWISP_MTP_DRAFT=1, resident fused tier only (C>=nE — mirrors
        // Tell.swift:105). Flag off / other tiers → mtpHead=nil → the loop below is
        // byte-identical to pre-change (seam contract, mtpDraftSpan).
        var mtpHead: SeedlessFusedVerify.SeedlessMTPHead? = nil
        if Tell.envFlag("QWISP_MTP_DRAFT"), let _ = _residentFwd, !useA3 {
            if let spec = try? SeedlessMTPValidate.loadSpec(modelDir: modelDir, store: store,
                                                       maxSeqLen: maxSeqLen),
               let h = SeedlessFusedVerify.SeedlessMTPHead(spec: spec) {
                mtpHead = h
                print("[raw-spec] MTP-D1 raw draft head active (maxSeqLen=\(maxSeqLen))")
            } else {
                print("[raw-spec] WARN: QWISP_MTP_DRAFT set but raw head init failed → drafts off")
            }
        } else if Tell.envFlag("QWISP_MTP_DRAFT") && useA3 {
            // ponytail: A3 pending-prefix + head pair-feed の整合(pending 実現タイミングの
            // pair 順序)は未配線 — A3 は opt-in 実験経路なので mtp とは相互排他にする。
            print("[raw-spec] NOTE: QWISP_MTP_DRAFT ignored (QWISP_RAW_A3 set — mutually exclusive)")
        }

        // reuse-rerank setup (flag-off = zero-cost nil path, notes/10 §1c-1e)
        let _useRerank = isStreaming && Tell.envFlag("QWISP_REUSE_RERANK")
        let _reuseAlpha = Double(Tell.envFloat("QWISP_REUSE_ALPHA", 0.0))
        var _reuseCtx = ReuseContext()
        var _reuseVerifyToks: [Int] = []   // updated before each stepArgmax; hook reads this
        if let fwd = _reuseStreamFwd, _useRerank {
            fwd.indsCaptureHook = { li, inds in
                let toks = _reuseVerifyToks
                guard !toks.isEmpty, !inds.isEmpty else { return }
                let ktop = inds.count / toks.count
                guard ktop > 0 else { return }
                _reuseCtx.observe(rowTokens: toks, layer: li, inds: inds, Ktop: ktop)
            }
        }

        // streaming stats 追跡は LayerExpertCache の static counters 経由(リセット済み)。
        // stream-vs-resident check は別途 resident backend を構築する。

        // ── PREFILL ───────────────────────────────────────────────────────
        // (mtpHead non-nil → prompt pairs ingested into head KV per chunk, Step 5)
        guard let lastNormed = prefill(promptIds: promptIds, backend: backend,
                                       mtpHead: mtpHead, mtpFwd: _residentFwd)
        else { return "[raw-spec] ERROR: prefill returned nil" }

        // First token: argmax of logits at the last prompt position.
        guard let lg0 = engine.logits(lastNormed, M: 1)
        else { return "[raw-spec] ERROR: prefill logits nil" }
        MLX.eval([lg0])
        var u = MLX.argMax(lg0[0], axis: -1).item(Int.self)

        // ── SPEC LOOP ─────────────────────────────────────────────────────
        // hist: all token ids so far (prompt + generated), used as suffix-draft context.
        var hist = promptIds.map { Int($0) }
        var out: [Int] = []
        var accTok = 0     // total accepted draft tokens (for accept/step)
        var steps  = 0
        var pending: [Int] = []  // A3: tokens committed but not yet realized in cache
        // pendingCap = 24 (already declared above for maxM computation)

        // MTP-D1: row (in the most recent forward's normed buffer) of the hidden that
        // produced the current u. Invariant: last row of the most recent forward
        // (A3-reject exception: verify rows survive rollback, row = pk+p).
        // -1 = invalid (draft seam disabled until the next tracked forward).
        // Prefill chunk=64 → last chunk's final row is (pLen-1) % 64.
        var rowOfU = (promptIds.count - 1) % 64

        // Phase II-a: QWISP_CHAIN_K=<k>opt-in GPU token-feedback chained greedy decode.
        // Only the D==0 non-A3/empty-pending greedy span uses the chain path; A3 and draft-
        // bearing steps keep the per-step path (snapshot/rollback + suffix-draft needs CPU tokens).
        let chainK = Tell.envInt("QWISP_CHAIN_K", SeedlessFusedVerify.SeedlessFusedForward.chainKDefault)

        let t0 = DispatchTime.now()

        while out.count < N {
            let _reuseArg: (ctx: ReuseContext, residentPerLayer: [Set<Int>], alpha: Double)? = _useRerank ? {
                let resPerLayer = _reuseStreamProviders?.map { Set($0.cache.slotOf.keys) } ?? []
                return (ctx: _reuseCtx, residentPerLayer: resPerLayer, alpha: _reuseAlpha)
            }() : nil
            var drafts = Tell.suffixDraft(hist + [u], maxMatch: 32, draftK: maxK, minMatch: 4,
                                          reuseCtx: _reuseArg)
            var D      = drafts.count

            // ★ MTP-D1 (Step 4): suffix-draftless かつ pending 空 → raw head の 1-token draft。
            //   下の verify path が必ず照合する(reject 経路は greedy と同一 token 列)ので lossless。
            //   flag off = mtpHead nil = mtpDraftSpan nil = このブロック不変(byte-identity)。
            var mtpDrafted = false
            if D == 0, mtpHead != nil, pending.isEmpty,
               let d = mtpDraftSpan(head: mtpHead, hPrevBuf: _residentFwd?.normedBuffer,
                                    rowOfU: rowOfU, u: u) {
                drafts = [d]; D = 1
                mtpDrafted = true
            }

            // ★ MTP-D1 (Step 5): u はこの step で必ず commit される → pair (h_prev, u) を
            //   forward が normed を上書きする前に feed(draft が同 pair を読んだ後 = validate
            //   と同一規約: draftArgmax read-only → feedPairs same pair)。rowOfU<0 は skip
            //   (KV desync するが draft は verify に photograph されるので lossless のまま)。
            //   Step 6 fold: draft が走った step は同 pair の k/v が既に pos=len に書かれて
            //   いる → commitLastDraft(len advance のみ、feed CB 不要)。
            if let head = mtpHead, let fwd = _residentFwd, rowOfU >= 0 {
                if mtpDrafted {
                    _ = head.commitLastDraft()
                } else {
                    _ = head.feedPairs(hBuf: fwd.normedBuffer,
                                       rowRange: rowOfU ..< (rowOfU + 1), toks: [Int32(u)])
                }
            }

            // Snapshot before the batched verify (backend-specific representation).
            var snap = backend.snapshot()

            if D == 0 {
                // No draft available
                // Phase II-a chain path: only the non-A3 / empty-pending greedy span.
                // chainedStepArgmax is bit-exact to K sequential stepArgmax([t]) calls, so OUT_TOKENS
                // stays byte-identical to per-step while collapsing K CPU round-trips to 1.
                // mtpHead active → chain disabled: chained steps would skip the per-commit
                // pair feed (head KV position desync). Flag off (nil) → condition unchanged.
                if mtpHead == nil, chainK > 0, let chainFn = backend.chainedStepArgmax,
                   (!useA3 || pending.isEmpty) {
                    if let chainResult = chainFn(Int32(u), chainK), !chainResult.isEmpty {
                        out.append(u); hist.append(u)
                        let addN = Swift.min(chainResult.count - 1, N - out.count)
                        for i in 0 ..< addN {
                            out.append(chainResult[i]); hist.append(chainResult[i])
                        }
                        u = chainResult[chainResult.count - 1]
                        // ponytail: normed-row state after a chained CB is unverified → disable
                        // the draft seam until the next tracked forward re-establishes it.
                        rowOfU = -1
                        steps += 1; continue
                    }
                    // chain returned nil (e.g. strict mode) → fall through to per-step
                }
                if useA3 && !pending.isEmpty {
                    // A3 D==0 path: stepArgmax on [pending, u] (batched)
                    let stepTokens: [Int32] = pending.map { Int32($0) } + [Int32(u)]
                    if _useRerank { _reuseVerifyToks = stepTokens.map { Int($0) } }
                    guard let evals = backend.stepArgmax(stepTokens)
                    else { return "[raw-spec] ERROR: A3 step(D=0) nil" }
                    out.append(u); hist.append(u)
                    rowOfU = pending.count    // u = evals[pk] ← row pk (last row)
                    u = evals[pending.count]  // argmax at position pk (where u is)
                    pending = []
                } else {
                    // Non-A3 or empty pending: simple single step
                    if _useRerank { _reuseVerifyToks = [Int(u)] }
                    guard let evals = backend.stepArgmax([Int32(u)])
                    else { return "[raw-spec] ERROR: step(D=0) nil" }
                    out.append(u); hist.append(u)
                    u = evals[0]
                    rowOfU = 0
                }
                steps += 1
                continue
            }

            // Batched verify construction
            if useA3 {
                // A3: fuse pending + [u] + drafts into ONE verify batch (no flush-before-verify)
                let pk = pending.count
                let verifyTokens: [Int32] = pending.map { Int32($0) } + [Int32(u)] + drafts.map { Int32($0) }
                if _useRerank { _reuseVerifyToks = verifyTokens.map { Int($0) } }
                guard let evals = backend.stepArgmax(verifyTokens)
                else { return "[raw-spec] ERROR: A3 verify step nil" }

                // Decision row offset: evals[pk] = prediction after u (row pk in fused batch).
                // drafts[p] should equal evals[pk + p] (mirror of TellBolt's evals[p] after
                // slicing vlg[0, pk ..< pk+D+1]).  The spec document wrote pk+1+p but that is
                // an off-by-one: the u-row IS the comparison origin, not a skip.
                var p = 0
                while p < D && drafts[p] == evals[pk + p] { p += 1 }

                if p == D {
                    // ── A3 full accept: fused forward already advanced cache ───
                    out.append(u); hist.append(u)
                    for d in drafts { out.append(d); hist.append(d) }
                    accTok += D
                    steps  += 1
                    pending = []
                    u = evals[pk + D]
                    rowOfU = pk + D           // last row of the verify forward
                } else {
                    // ── A3 partial reject: rollback to B, re-add to pending ────
                    backend.rollback(snap)
                    out.append(u); hist.append(u)
                    for d in drafts.prefix(p) { out.append(d); hist.append(d) }
                    accTok += p
                    steps  += 1
                    // Add u + accepted drafts to pending
                    pending.append(u)
                    for d in drafts.prefix(p) { pending.append(d) }
                    u = evals[pk + p]
                    rowOfU = pk + p           // rollback leaves normed intact → verify row pk+p

                    // Cap flush: only if pending exceeds 24 (safety to bound M)
                    if pending.count >= pendingCap {
                        let pendingTokens: [Int32] = pending.map { Int32($0) }
                        if _useRerank { _reuseVerifyToks = [] }
                        guard let _ = backend.forward(pendingTokens)
                        else { return "[raw-spec] ERROR: A3 pending flush nil" }
                        // flush realizes pending; its last row's hidden is u's predecessor
                        rowOfU = pendingTokens.count - 1
                        pending = []
                    }
                }
            } else {
                // ── Non-A3 path (unchanged) ──────────────────────────────
                let verifyTokens: [Int32] = [Int32(u)] + drafts.map { Int32($0) }
                if _useRerank { _reuseVerifyToks = verifyTokens.map { Int($0) } }
                guard let evals = backend.stepArgmax(verifyTokens)
                else { return "[raw-spec] ERROR: verify step nil" }

                var p = 0
                while p < D && drafts[p] == evals[p] { p += 1 }


                if p == D {
                    // ── full accept ───────────────────────────────────────
                    out.append(u); hist.append(u)
                    for d in drafts { out.append(d); hist.append(d) }
                    accTok += D
                    steps  += 1
                    u = evals[D]
                    rowOfU = D                // last row of the verify forward
                } else {
                    // ── partial reject ────────────────────────────────────
                    backend.rollback(snap)
                    out.append(u); hist.append(u)
                    for d in drafts.prefix(p) { out.append(d); hist.append(d) }
                    accTok += p
                    steps  += 1
                    let rebuildTokens: [Int32] = [Int32(u)] + drafts.prefix(p).map { Int32($0) }
                    if _useRerank { _reuseVerifyToks = [] }
                    guard let _ = backend.forward(rebuildTokens)
                    else { return "[raw-spec] ERROR: rebuild forwardRows nil" }
                    u = evals[p]
                    rowOfU = p                // last row of the rebuild forward
                }

                // ★ MTP-D1 (Step 5): accepted-draft pairs (h_i, tok_{i+1}) を feed。
                //   Tell.mtpFeedPlan(test 70, engine-agnostic)準拠: pk=0 → feedRows 0..<p、
                //   lastHRow=p(rowOfU の branch 値と一致)。accept 時は verify rows、reject 時は
                //   rebuild rows(同 token 列を同 state から再計算 = 同値)— どちらも現 normed。
                if let head = mtpHead, let fwd = _residentFwd,
                   let plan = Tell.mtpFeedPlan(pk: 0, p: p, path: p == D ? .fullAccept : .reject),
                   !plan.feedRows.isEmpty {
                    _ = head.feedPairs(hBuf: fwd.normedBuffer, rowRange: plan.feedRows,
                                       toks: drafts.prefix(p).map { Int32($0) })
                }
            }
        }

        // ── timing + quality ─────────────────────────────────────────────
        let secs  = Double(DispatchTime.now().uptimeNanoseconds - t0.uptimeNanoseconds) / 1e9
        let tokps = Double(N) / secs
        let outN  = Array(out.prefix(N))
        let match = zip(outN, gRefIds.prefix(N)).filter { $0 == $1 }.count

        if Tell.envFlag("QWISP_DUMP_TOKENS") {
            print("PROMPT_TOKENS:" + promptIds.map { String($0) }.joined(separator: ","))
            print("OUT_TOKENS:"    + outN.map      { String($0) }.joined(separator: ","))
        }

        // streaming per-run stats
        var statsLine = ""
        if isStreaming {
            let ensMs  = Double(LayerExpertCache.ensureNanos) / 1e6
            let preadMs = Double(LayerExpertCache.preadNanos) / 1e6
            statsLine = String(format: "\n[RawSpec] streaming stats: ensure=%.1fms pread=%.1fms misses=%d chunks=%d",
                               ensMs, preadMs, LayerExpertCache.missTotal, LayerExpertCache.chunkTotal)
            if _useRerank {
                statsLine += "\n[RawSpec] reuse diag: votes=\(Tell.reuseVotes) forks=\(Tell.reuseForks) flips=\(Tell.reuseFlips)"
            }
        }

        let tierTag = isStreaming ? "streaming C=\(rawC)" : "resident"
        let summary = String(format:
            "[RawSpec] raw engine(\(tierTag)): %.1f tok/s  accept/step=%.2f  品質(vs ref spec_greedy) %d/%d=%.0f%%",
            tokps,
            Double(accTok) / Double(Swift.max(steps, 1)),
            match, N, Double(match) / Double(N) * 100) + statsLine

        // ── STREAM-VS-RESIDENT CHECK ──────────────────────────────────────
        // QWISP_RAWSTREAM_CHECK=1 (strict streaming only): run resident path and diff outputs.
        if isStreaming && Tell.envFlag("QWISP_RAWSTREAM_CHECK") {
            let checkResult = runStreamVsResidentCheck(
                engine: engine, store: store, modelDir: modelDir,
                promptIds: promptIds, streamOut: outN, N: N,
                maxM: maxM, maxSeqLen: maxSeqLen, maxK: maxK)
            guard Tell.envFlag("QWISP_RAWSPEC_CHECK") else { return summary + checkResult }
            let selfCheck = runSelfCheck(engine: engine, promptIds: promptIds, N: N, maxK: maxK,
                                         mkBackend: mkBackend, outN: outN)
            return summary + checkResult + selfCheck
        }

        // ── SELF-CHECK: spec output vs pure sequential raw greedy ─────────
        // QWISP_RAWSPEC_CHECK=1: run M=1 greedy from scratch and verify N/N lossless.
        guard Tell.envFlag("QWISP_RAWSPEC_CHECK") else { return summary }
        let selfCheck = runSelfCheck(engine: engine, promptIds: promptIds, N: N, maxK: maxK,
                                      mkBackend: mkBackend, outN: outN)
        return summary + selfCheck
    }

    // ── BOLT MODE ────────────────────────────────────────────────────────

    /// Bolt run: calib → freeze tables → 1-CB bolt decode。
    /// Deviations from TellBolt: no B3 in-decode fetch, no A3 pending-prefix(raw spec loop 不変)。
    private static func runBoltMode(engine: SeedlessEngine, modelDir: String, promptIds: [Int32],
                                     gRefIds: [Int], N: Int, maxK: Int, C: Int,
                                     maxM: Int, maxSeqLen: Int, refPath: String) throws -> String {
        let calibN = Tell.envInt("QWISP_CALIB", 48)
        let nLayers = SeedlessEngine.numLayers
        let nE = 256

        // ── phase 1: build strict streaming backend #1 for calib ─────────
        print("[raw-spec bolt] phase 1: building strict streaming backend for calib ...")
        guard let (backend1, fwd1, providers) = streamingBackend(
            engine: engine, modelDir: modelDir, maxM: maxM, maxSeqLen: maxSeqLen, C: C)
        else { return "[raw-spec bolt] ERROR: streaming backend #1 init nil" }

        // ── phase 2: frequency + co-activation calib ──────────────────────
        // counts[layer][e] = how many times expert e was routed in calib steps.
        // coact[layer][a][b] = co-activation count of experts a,b in same token(symmetric).
        // During calib all steps are M=1 greedy.
        var counts = [[Int]](repeating: [Int](repeating: 0, count: nE), count: nLayers)
        var coact  = [[[Int]]](repeating: [[Int]](repeating: [Int](repeating: 0, count: nE), count: nE),
                               count: nLayers)

        fwd1.indsCaptureHook = { li, inds in
            // inds is [M*Ktop] where M=1, Ktop=8 for greedy calib steps → 8 expert ids
            let distinct = Array(Set(inds.map { Int($0) }))
            for e in distinct { counts[li][e] += 1 }
            let n = distinct.count
            for ai in 0 ..< n {
                for bi in (ai + 1) ..< n {
                    let a = distinct[ai], b = distinct[bi]
                    coact[li][a][b] += 1; coact[li][b][a] += 1
                }
            }
        }

        print("[raw-spec bolt] calib: prefill + \(calibN) greedy steps ...")
        guard let lastNormedCalib = prefill(promptIds: promptIds, backend: backend1)
        else { return "[raw-spec bolt] ERROR: calib prefill nil" }
        guard let lg0calib = engine.logits(lastNormedCalib, M: 1)
        else { return "[raw-spec bolt] ERROR: calib logits nil" }
        MLX.eval([lg0calib])
        var uCalib = MLX.argMax(lg0calib[0], axis: -1).item(Int.self)
        for _ in 0 ..< calibN {
            guard let evals = backend1.stepArgmax([Int32(uCalib)])
            else { return "[raw-spec bolt] ERROR: calib greedy step nil" }
            uCalib = evals[0]
        }
        fwd1.indsCaptureHook = nil
        print("[raw-spec bolt] calib done.")

        // ── phase 3: fresh fwd for timed run — reuse same providers(arena persists) ──
        print("[raw-spec bolt] phase 3: building fresh strict backend (reusing arena providers) ...")
        guard let (backend2, fwd2, _) = streamingBackend(
            engine: engine, modelDir: modelDir, maxM: maxM, maxSeqLen: maxSeqLen, C: C,
            existingProviders: providers)
        else { return "[raw-spec bolt] ERROR: streaming backend #2 init nil" }

        // ── phase 4: exact prefill on fresh backend ────────────────────────
        print("[raw-spec bolt] phase 4: exact prefill ...")
        guard let lastNormed = prefill(promptIds: promptIds, backend: backend2)
        else { return "[raw-spec bolt] ERROR: prefill nil" }
        guard let lg0 = engine.logits(lastNormed, M: 1)
        else { return "[raw-spec bolt] ERROR: prefill logits nil" }
        MLX.eval([lg0])
        let u0 = MLX.argMax(lg0[0], axis: -1).item(Int.self)

        // ── phase 5: per-layer hot-pin + buddy table ───────────────────────
        print("[raw-spec bolt] phase 5: per-layer top-C ensure + buildBuddyTable ...")
        var tables: [[Int32]] = []
        for (li, provider) in providers.enumerated() {
            // top-C experts by counts[li], tie-break by lower expert id
            let top = counts[li].enumerated()
                .sorted { $0.element != $1.element ? $0.element > $1.element : $0.offset < $1.offset }
                .prefix(C)
                .map { $0.offset }
            _ = provider.cache.ensure(Array(top))
            provider.cache.buildBuddyTable(coact: coact[li], numExperts: nE)
            tables.append(provider.cache.buddyTableCPU)
        }

        // freeze: switch to bolt mode (1-CB, no ensure during decode)
        fwd2.setBoltTables(tables)
        print("[raw-spec bolt] tables set, bolt mode active.")

        // ── Stage 1 案B: gate-score residency bias ────────────────────────
        // QWISP_ROUTE_BIAS_EPS > 0: 各層の常駐 expert(buddyExpertCPU[e]==e)に eps を加算して優先選択。
        // ★notes/13 採用(owner 2026-07-06): 既定 0.25(recalib とセットで default 昇格)。0 で opt-out
        //   =旧 bolt と byte-identical(bias dispatch 追加ゼロ)。
        let boltBiasEps = Float(Tell.envFloat("QWISP_ROUTE_BIAS_EPS", 0.25))
        // 案B experiment-3: horizon-decay。staleness 仮説(resident set は calib 48tok で freeze され
        // 生成が進むほど prior が陳腐化→bias が後半で有害)に基づき、ε(t) = ε0·max(0, 1 − t/H) で
        // 線形減衰(t=生成済み token 数)。H=0(既定)= decay 無し=Stage 1 挙動と完全一致。
        let boltBiasDecayH = Tell.envInt("QWISP_ROUTE_BIAS_DECAY_H", 0)
        if boltBiasEps > 0 {
            let biasMasks: [[Int32]] = providers.map { p in
                (0 ..< nE).map { Int32(p.cache.buddyExpertCPU[$0] == $0 ? 1 : 0) }
            }
            fwd2.setRouteBias(masks: biasMasks, eps: boltBiasEps)
            print(String(format: "[raw-spec bolt] route bias eps=%.3f active%@.", boltBiasEps,
                         boltBiasDecayH > 0 ? " (decay H=\(boltBiasDecayH))" : ""))
        }

        // ── QWISP_BOLT_DIAG=1: routing telemetry (notes/11 案B Stage 0, measurement-only) ──
        // 層別 side-buffer に route 直後(remap 前)の inds/gl をコピーし、greedy step 毎に CPU 読出。
        // cold-selection 率と near-tie margin(常駐が +ε で置換に必要な量)を集計する。
        // diag 中は chain を無効化(1 CB=1 step にして step 毎読出を可能に)。OUT_TOKENS は不変。
        let boltDiag = Tell.envFlag("QWISP_BOLT_DIAG")
        let Ktop = 8

        // ── notes/13 採用: free-run rolling re-calib(既定 ON, R=128。0 で opt-out=旧 bolt)──
        // R token 毎に「直近窓の観測 routing」から counts/coact を再集計し top-C ensure(pread)+
        // buddy table + bias mask を re-freeze。観測は diag_copy_route の slot/M 一般化 layout
        // (chain 全 step = slot 別 / verify M 行 = kE.x=M*Ktop)で全 coverage。
        // notes/14 TODO-2: QWISP_BOLT_WORKLOAD が preset (R,B) を選択。明示 env が常に上書き。
        let boltWL = Tell.envStr("QWISP_BOLT_WORKLOAD", "")
        let preset = Tell.boltWorkloadPreset(boltWL)
        let boltRecalibR = Tell.envInt("QWISP_BOLT_RECALIB_R", preset.r)
        // notes/14 async refresh wiring (QWISP_BOLT_REFRESH_ASYNC, default 1)
        // B=32: chunk IO(~38ms@1.5GB/s)と decode の均衡 + 深さ2 pipeline で hide できる粒度。
        // S=0(既定)= plan 毎に自動導出: 窓の 3/4 で全 chunk を消化し切る間隔に均す(決定的)。
        let boltRefreshAsync = boltRecalibR > 0 && Tell.envInt("QWISP_BOLT_REFRESH_ASYNC", 1) != 0
        let boltRefreshB     = Tell.envInt("QWISP_BOLT_REFRESH_B", preset.b)
        if !boltWL.isEmpty {
            print("[raw-spec bolt] workload=\(boltWL) preset R=\(preset.r) B=\(preset.b)")
        }
        let boltRefreshS     = Tell.envInt("QWISP_BOLT_REFRESH_S", 0)
        func refreshStride(_ nChunks: Int) -> Int {
            // 半窓 R/2 に均等割り(OAT 実測 2026-07-07): 全幅だと table 鮮度遅れで TF が旧 bolt を割る
            // (shortnl 71.3<72.5)、半窓で回復(73.2)しつつ slow-NAND 速度は −2% 程度。
            // 詰めすぎ(S=2)は swap block=GPU idle で速度・fidelity とも悪化。
            boltRefreshS > 0 ? boltRefreshS
                             : Swift.max(1, (boltRecalibR / 2) / Swift.max(1, nChunks))
        }

        // Async refresh staging: ping-pong の 2 arena(N=B each)。bg thread は bounded-buffer
        // (semFree/semReady)で全 chunk を先読み pipeline し、swap は通常 block しない(notes/14)。
        var stagingArenas: [ExpertArena] = []
        let bgRefreshQueue = DispatchQueue(label: "qwisp.bolt.async_refresh", qos: .userInitiated)
        if boltRefreshAsync {
            if let first = providers.first {
                for _ in 0 ..< 2 {
                    if let a = try? ExpertArena(device: first.cache.arena.device,
                                                source: first.cache.arena.source,
                                                N: boltRefreshB, refLayer: 0) { stagingArenas.append(a) }
                }
            }
            if stagingArenas.count < 2 {
                stagingArenas = []
                print("[raw-spec bolt] WARNING: staging arena failed, async refresh disabled")
            }
        }
        let stagingArena: ExpertArena? = stagingArenas.first   // nil 判定用(既存コードの guard 共用)

        // chain: diag 単独時は step 毎読出のため無効化。recalib は slot 観測で chain と共存。
        let boltChainK = boltDiag ? 0 : Tell.envInt("QWISP_CHAIN_K", SeedlessFusedVerify.SeedlessFusedForward.chainKDefault)

        // 観測 buffer(diag/recalib 共用, 一般 layout)。obsMaxM/obsSlots は recalib 有効時のみ拡張。
        let obsMaxM = boltRecalibR > 0 ? maxM : 1
        let obsSlots = boltRecalibR > 0 ? Swift.max(1, boltChainK) : 1
        var diagResidents: [Set<Int>] = [], diagBuddyExp: [[Int32]] = []
        var diagIBuf: MTLBuffer? = nil, diagGBuf: MTLBuffer? = nil
        if boltDiag || boltRecalibR > 0 {
            guard SeedlessMetalForward.compileDiagCopyRoute(),
                  let (device, _) = SeedlessMetalForward.ensure(),
                  let ib = device.makeBuffer(length: obsSlots * nLayers * obsMaxM * Ktop * 4,
                                             options: .storageModeShared),
                  let gb = device.makeBuffer(length: nLayers * nE * 2, options: .storageModeShared)
            else { return "[raw-spec bolt] ERROR: diag/recalib setup failed" }
            diagIBuf = ib; diagGBuf = gb
            fwd2.diagObsMaxM = obsMaxM
            fwd2.diagRouteBufs = (ib, gb)
        }
        if boltDiag {
            for p in providers {
                diagResidents.append(Set(p.cache.slotOf.keys))
                diagBuddyExp.append(p.cache.buddyExpertCPU)
            }
            print("[raw-spec bolt] diag mode: telemetry on, chain disabled")
        }
        // free-run recalib 窓・集計
        var frWinCounts: [[Int]] = [], frWinCoact: [[[Int]]] = []
        var frRefreshes = 0, frNextRefresh = boltRecalibR
        let frMissBase = LayerExpertCache.missTotal
        if boltRecalibR > 0 {
            frWinCounts = [[Int]](repeating: [Int](repeating: 0, count: nE), count: nLayers)
            frWinCoact = [[[Int]]](repeating: [[Int]](repeating: [Int](repeating: 0, count: nE), count: nE),
                                   count: nLayers)
            print("[raw-spec bolt] recalib active: R=\(boltRecalibR) eps=\(boltBiasEps) chainK=\(boltChainK)")
        }
        /// 観測読出: slot(chain 位置)と M(行数)を指定して全層分を窓に累積。
        func frAccumulate(slot: Int, M: Int) {
            guard boltRecalibR > 0, let ib = diagIBuf else { return }
            for li in 0 ..< nLayers {
                let off = ((slot * nLayers + li) * obsMaxM) * Ktop * 4
                let inds = Array(UnsafeBufferPointer(
                    start: ib.contents().advanced(by: off).assumingMemoryBound(to: Int32.self),
                    count: M * Ktop))
                _ = SeedlessFusedVerify.SeedlessFusedForward.recalibAccumulate(
                    inds: inds, M: M, Ktop: Ktop, nE: nE,
                    counts: &frWinCounts[li], coact: &frWinCoact[li])
            }
        }
        /// refresh: 窓 top-C ensure(pread)→ buddy table → slot table → bias mask → 窓リセット。
        func frRefresh() {
            guard boltRecalibR > 0 else { return }
            var newTables: [[Int32]] = []
            for (li, provider) in providers.enumerated() {
                let top = frWinCounts[li].enumerated()
                    .sorted { $0.element != $1.element ? $0.element > $1.element : $0.offset < $1.offset }
                    .prefix(C)
                    .map { $0.offset }
                _ = provider.cache.ensure(Array(top))
                provider.cache.buildBuddyTable(coact: frWinCoact[li], numExperts: nE)
                newTables.append(provider.cache.buddyTableCPU)
            }
            fwd2.setBoltTables(newTables)
            if boltBiasEps > 0 {
                let masks: [[Int32]] = providers.map { p in
                    (0 ..< nE).map { Int32(p.cache.buddyExpertCPU[$0] == $0 ? 1 : 0) }
                }
                fwd2.setRouteBias(masks: masks, eps: boltBiasEps)
            }
            for li in 0 ..< nLayers {
                for e in 0 ..< nE { frWinCounts[li][e] = 0; frWinCoact[li][e] = [Int](repeating: 0, count: nE) }
            }
            frRefreshes += 1
        }
        // Async plan state (notes/14)
        struct BoltCrossJob { let li: Int; let expert: Int; let victimSlot: Int }
        var asyncBoundary  = 0
        var asyncChunks    = [[BoltCrossJob]]()
        var asyncNextSwap  = 0
        var asyncStride    = 1   // plan 毎に refreshStride で再計算
        var asyncSemFree   = DispatchSemaphore(value: 2)   // plan 毎に再生成(消化後は初期値に戻る)
        var asyncSemReady  = DispatchSemaphore(value: 0)
        var asyncCoactSnap = [[[Int]]]()   // coact snapshot at plan creation
        /// Kick background pipeline: 全 chunk を bounded-buffer(2 staging)で先読み pread。
        /// Background は stagingArenas のみに書く — providers には触れない。
        func startBgPlan() {
            guard boltRefreshAsync, stagingArenas.count == 2, !asyncChunks.isEmpty else { return }
            let chunks = asyncChunks, stages = stagingArenas
            let semFree = asyncSemFree, semReady = asyncSemReady
            bgRefreshQueue.async {
                for (j, chunk) in chunks.enumerated() {
                    semFree.wait()                       // swap 済みで空いた staging を待つ
                    let stage = stages[j % 2]
                    var byLayer = [Int: [(e: Int, slot: Int)]]()
                    for (k, cj) in chunk.enumerated() {
                        byLayer[cj.li, default: []].append((e: cj.expert, slot: k))
                    }
                    for (layer, jobs) in byLayer { stage.loadMany(layer, jobs) }
                    semReady.signal()
                }
            }
        }
        /// Atomic CPU-turn swap: apply asyncChunks[j].
        /// Must be called only at loop head (GPU idle). Waits for IO if not done.
        func swapBgChunk(_ j: Int) {
            guard boltRefreshAsync, stagingArenas.count == 2, j < asyncChunks.count else { return }
            asyncSemReady.wait()   // block until pread done (pipeline 先読み済みなら即)
            let chunk = asyncChunks[j]
            // memcpy staging slot k → main arena victimSlot (same sliceBytes — all layers share shape)
            let stage = stagingArenas[j % 2]
            for (k, cj) in chunk.enumerated() {
                let arena = providers[cj.li].cache.arena
                for key in stage.slots.keys {
                    guard let srcS = stage.slots[key], let dstS = arena.slots[key] else { continue }
                    memcpy(dstS.ptr + cj.victimSlot * dstS.sliceBytes,
                           srcS.ptr + k             * srcS.sliceBytes,
                           srcS.sliceBytes)
                }
                // Direct bookkeeping (no ensure() — doctrine: same-call invariant)
                let cache = providers[cj.li].cache
                let cur = cache.expertAt[cj.victimSlot]
                if cur >= 0 && cur != cj.expert { cache.slotOf.removeValue(forKey: cur) }
                cache.slotOf[cj.expert]        = cj.victimSlot
                cache.expertAt[cj.victimSlot]  = cj.expert
                cache.clock                   += 1
                cache.tick[cj.victimSlot]      = cache.clock
            }
            LayerExpertCache.missTotal += chunk.count   // pread io 計上(staging 経路は ensure 迂回のため手動)
            // Rebuild buddy + freeze — chunk が触った層のみ(層独立なので full rebuild と内容同一)。
            // 全層 rebuild + 40 mask 再確保は swap 毎数 ms×多 swap で IO-bound 上限を食っていた。
            for li in Set(chunk.map { $0.li }).sorted() {
                let snap = li < asyncCoactSnap.count ? asyncCoactSnap[li]
                           : [[Int]](repeating: [Int](repeating: 0, count: nE), count: nE)
                let cache = providers[li].cache
                cache.buildBuddyTable(coact: snap, numExperts: nE)
                fwd2.setBoltTable(li, cache.buddyTableCPU)
                if boltBiasEps > 0 {
                    fwd2.updateRouteBiasMask(li, (0..<nE).map { Int32(cache.buddyExpertCPU[$0] == $0 ? 1 : 0) })
                }
            }
            asyncSemFree.signal()   // staging を bg pipeline に返却(次々 chunk の先読み解禁)
        }
        // group(li): early 0-12 / mid 13-26 / late 27-39
        struct DiagAcc { var cold = 0; var routed = 0; var stepsWithCold = 0; var margins: [Float] = [] }
        var diagAcc = [DiagAcc(), DiagAcc(), DiagAcc()]
        var diagSteps = 0
        // missWeightStats accumulators (per-layer observations, boltDiag only)
        var diagMissInds: [[Int]] = []
        var diagMissWeights: [[Float]] = []
        var diagMissResidents: [Set<Int>] = []
        func diagReadStep() {
            guard let ib = diagIBuf, let gb = diagGBuf else { return }
            diagSteps += 1
            var groupHadCold = [false, false, false]
            for li in 0 ..< nLayers {
                // 一般 layout(slot 0, row 0): stride は obsMaxM(recalib 共存時 >1)。
                let indsArr = Array(UnsafeBufferPointer(
                    start: ib.contents().advanced(by: li * obsMaxM * Ktop * 4).assumingMemoryBound(to: Int32.self),
                    count: Ktop))
                let glArr = Array(UnsafeBufferPointer(
                    start: gb.contents().advanced(by: li * nE * 2).assumingMemoryBound(to: Float16.self),
                    count: nE))
                let (cold, mar) = SeedlessFusedVerify.SeedlessFusedForward.computeRouteDiag(
                    inds: indsArr, gl: glArr, resident: diagResidents[li],
                    buddyExpert: diagBuddyExp[li], Ktop: Ktop)
                let g = li <= 12 ? 0 : (li <= 26 ? 1 : 2)
                diagAcc[g].cold += cold.count
                diagAcc[g].routed += Ktop
                diagAcc[g].margins.append(contentsOf: mar)
                if !cold.isEmpty { groupHadCold[g] = true }
                // accumulate norm_topk weights for missWeightStats
                if boltDiag {
                    // ponytail: reproduce route_top8_rows exactly — stable softmax over E=256
                    // logits, then renorm the selected Ktop probs to sum 1 (=combine's norm_topk).
                    let logitsF = glArr.map { Float($0) }
                    let maxL = logitsF.max() ?? 0
                    let exps = logitsF.map { Foundation.exp($0 - maxL) }
                    let Z = exps.reduce(0, +)
                    let gates = Z > 0 ? exps.map { $0 / Z } : exps.map { _ in 1.0 / Float(nE) }
                    let topGates = indsArr.map { gates[Int($0)] }
                    let ss = topGates.reduce(0, +)
                    let normW = ss > 0 ? topGates.map { $0 / ss } : topGates.map { _ in 1.0 / Float(Ktop) }
                    diagMissInds.append(indsArr.map { Int($0) })
                    diagMissWeights.append(normW)
                    diagMissResidents.append(diagResidents[li])
                }
            }
            for g in 0 ..< 3 where groupHadCold[g] { diagAcc[g].stepsWithCold += 1 }
        }

        // ── phase 6: bolt spec decode loop ────────────────────────────────
        var hist = promptIds.map { Int($0) }
        var out: [Int] = []
        var accTok = 0, steps = 0
        var u = u0

        // Phase II-b: chain wiring for the D==0 greedy span.
        // chainedStepArgmax is bit-exact to K sequential stepArgmax calls (bolt = deterministic
        // buddy-greedy), so OUT_TOKENS is byte-identical with chain on or off.
        // (boltChainK は observation buffer 設計のため diag/recalib block の前で定義済み)

        let t0 = DispatchTime.now()

        while out.count < N {
            // notes/13 recalib: R 境界で re-freeze(全経路共通=loop 先頭で判定)。
            if boltRecalibR > 0 && out.count >= frNextRefresh {
                if boltRefreshAsync && stagingArena != nil {
                    // Flush any leftover chunks from previous plan (block-wait; bg pipeline 先読み済みなら軽い)
                    while asyncNextSwap < asyncChunks.count {
                        swapBgChunk(asyncNextSwap)
                        asyncNextSwap += 1
                    }
                    // Build new plan: snapshot coact, compute per-layer diffs, flatten + chunk
                    asyncCoactSnap = frWinCoact
                    var allJobs = [BoltCrossJob]()
                    for (li, provider) in providers.enumerated() {
                        if let plan = BoltAsyncRefresh.makePlan(
                            counts: frWinCounts[li], coact: frWinCoact[li],
                            slotOf: provider.cache.slotOf, expertAt: provider.cache.expertAt,
                            tick: provider.cache.tick, pinnedSlots: provider.cache.pinnedSlots,
                            C: C, nE: nE, B: boltRefreshB) {
                            for job in plan.jobs {
                                allJobs.append(BoltCrossJob(li: li, expert: job.expert, victimSlot: job.victimSlot))
                            }
                        }
                    }
                    asyncBoundary = out.count
                    asyncChunks = stride(from: 0, to: allJobs.count, by: boltRefreshB).map {
                        Array(allJobs[$0..<Swift.min($0 + boltRefreshB, allJobs.count)])
                    }
                    asyncNextSwap = 0
                    asyncStride = refreshStride(asyncChunks.count)
                    // Reset observation window
                    for li in 0..<nLayers {
                        for e in 0..<nE { frWinCounts[li][e] = 0; frWinCoact[li][e] = [Int](repeating: 0, count: nE) }
                    }
                    frRefreshes += 1
                    // bg pipeline 起動(semaphore は plan 毎に fresh — 前 plan は flush で初期値に戻り済み)
                    asyncSemFree = DispatchSemaphore(value: 2)
                    asyncSemReady = DispatchSemaphore(value: 0)
                    startBgPlan()
                } else {
                    frRefresh()   // QWISP_BOLT_REFRESH_ASYNC=0: sync path, byte-identical
                }
                frNextRefresh += boltRecalibR
            }
            // Async: swap scheduled chunks at fixed token positions (notes/14 §機構-3)。
            // chain/verify span で out.count が跳ぶため、この head で due の chunk は全部消化
            // (while)。IO は bg pipeline が先読み済みなので通常 block しない。
            while boltRefreshAsync, stagingArena != nil, asyncNextSwap < asyncChunks.count,
                  out.count >= asyncBoundary + (asyncNextSwap + 1) * asyncStride {
                swapBgChunk(asyncNextSwap)
                asyncNextSwap += 1
            }
            // experiment-3 horizon-decay: 生成位置で ε を線形減衰(H=0 で無効=常時 ε0)。
            if boltBiasEps > 0 && boltBiasDecayH > 0 {
                fwd2.setRouteBiasEps(boltBiasEps * Swift.max(0, 1 - Float(out.count) / Float(boltBiasDecayH)))
            }
            let drafts = Tell.suffixDraft(hist + [u], maxMatch: 32, draftK: maxK, minMatch: 4)
            let D      = drafts.count
            let snap   = backend2.snapshot()

            if D == 0 {
                // Phase II-b: delegate to boltGreedyChainSpan when chainK>0 and head is wired.
                if let (emitted, nextU) = boltGreedyChainSpan(
                    backend: backend2, u: u, chainK: boltChainK,
                    budget: N - out.count) {
                    // recalib 観測: chain の K step は slot 別に side-buffer 常駐(全 step coverage)。
                    if boltRecalibR > 0 { for k in 0 ..< boltChainK { frAccumulate(slot: k, M: 1) } }
                    for t in emitted { out.append(t); hist.append(t) }
                    u = nextU; steps += 1; continue
                }
                // chain unavailable or disabled → per-step fallback.
                guard let evals = backend2.stepArgmax([Int32(u)])
                else { return "[raw-spec bolt] ERROR: step(D=0) nil" }
                if boltDiag { diagReadStep() }   // M==1 greedy step: side-buffers fresh
                if boltRecalibR > 0 { frAccumulate(slot: 0, M: 1) }
                out.append(u); hist.append(u)
                u = evals[0]; steps += 1; continue
            }

            let verifyTokens: [Int32] = [Int32(u)] + drafts.map { Int32($0) }
            guard let evals = backend2.stepArgmax(verifyTokens)
            else { return "[raw-spec bolt] ERROR: verify step nil" }
            // recalib 観測: verify M=D+1 行(rebuild forward の重複観測は読まない)。
            if boltRecalibR > 0 { frAccumulate(slot: 0, M: D + 1) }

            var p = 0
            while p < D && drafts[p] == evals[p] { p += 1 }

            if p == D {
                out.append(u); hist.append(u)
                for d in drafts { out.append(d); hist.append(d) }
                accTok += D; steps += 1; u = evals[D]
            } else {
                backend2.rollback(snap)
                out.append(u); hist.append(u)
                for d in drafts.prefix(p) { out.append(d); hist.append(d) }
                accTok += p; steps += 1
                let rebuildTokens: [Int32] = [Int32(u)] + drafts.prefix(p).map { Int32($0) }
                guard let _ = backend2.forward(rebuildTokens)
                else { return "[raw-spec bolt] ERROR: rebuild nil" }
                u = evals[p]
            }
        }

        let secs  = Double(DispatchTime.now().uptimeNanoseconds - t0.uptimeNanoseconds) / 1e9
        let tokps = Double(N) / secs
        // 未 swap chunk の drain(計時外)。semaphore 収支を初期値に戻す — 残したまま return すると
        // bg 完了済みの場合 semFree<init で _dispatch_semaphore_dispose が trap(SIGTRAP)。
        // 副作用(table 更新)は無害: decode 済み、TF は開始時に re-canonicalize、self-check は recalib 時 skip。
        while asyncNextSwap < asyncChunks.count { swapBgChunk(asyncNextSwap); asyncNextSwap += 1 }
        let outN  = Array(out.prefix(N))
        let match = zip(outN, gRefIds.prefix(N)).filter { $0 == $1 }.count

        if Tell.envFlag("QWISP_DUMP_TOKENS") {
            print("PROMPT_TOKENS:" + promptIds.map { String($0) }.joined(separator: ","))
            print("OUT_TOKENS:"    + outN.map      { String($0) }.joined(separator: ","))
        }

        let summary = String(format:
            "[RawSpec] bolt(L3 near-lossless) C=\(C): %.1f tok/s  accept/step=%.2f  品質(vs ref) %d/%d=%.0f%%",
            tokps, Double(accTok) / Double(Swift.max(steps, 1)), match, N, Double(match) / Double(N) * 100)
        print("[RawSpec] NOTE: bolt is L3 near-lossless (buddy remap, not strict). Quality vs ref is informational.")

        // [BoltRecalib] (notes/13): free-run refresh 回数と pread io コスト。
        if boltRecalibR > 0 {
            print("[BoltRecalib] free-run refreshes=\(frRefreshes) preadMisses=\(LayerExpertCache.missTotal - frMissBase) R=\(boltRecalibR) eps=\(String(format: "%.2f", boltBiasEps))")
        }

        // [BoltDiag] telemetry report (notes/11 案B Stage 0): greedy-step のみ集計(M==1)。
        if boltDiag {
            let groupNames = ["early(0-12)", "mid(13-26)", "late(27-39)"]
            print("[BoltDiag] greedy-steps=\(diagSteps) of \(steps) total steps, Ktop=\(Ktop), C=\(C)")
            for g in 0 ..< 3 {
                let a = diagAcc[g]
                let coldRate = a.routed > 0 ? Double(a.cold) / Double(a.routed) * 100 : 0
                let stepPct = diagSteps > 0 ? Double(a.stepsWithCold) / Double(diagSteps) * 100 : 0
                let fin = a.margins.filter { $0.isFinite }.sorted()
                let infN = a.margins.count - fin.count
                func pct(_ p: Double) -> Float {
                    fin.isEmpty ? Float.nan : fin[Swift.min(fin.count - 1, Int(Double(fin.count) * p))]
                }
                let flips = [0.5, 1.0, 2.0, 4.0].map { eps in
                    a.margins.isEmpty ? 0.0
                        : Double(a.margins.filter { $0 < Float(eps) }.count) / Double(a.margins.count) * 100
                }
                print(String(format: "[BoltDiag] group=%@ coldRate=%d/%d=%.1f%% stepsWithCold=%.0f%% " +
                             "marginP10/50/90=%.2f/%.2f/%.2f inf=%d flip@eps{0.5:%.0f%% 1:%.0f%% 2:%.0f%% 4:%.0f%%}",
                             groupNames[g], a.cold, a.routed, coldRate, stepPct,
                             pct(0.10), pct(0.50), pct(0.90), infN,
                             flips[0], flips[1], flips[2], flips[3]))
            }
            let mw = Tell.missWeightStats(topInds: diagMissInds, topWeights: diagMissWeights, resident: diagMissResidents)
            print(String(format: "[BoltDiag] missWeight p10/50/90=%.3f/%.3f/%.3f top1Share=%.1f%% missMass/token=%.3f misses=%d",
                         mw.p10, mw.p50, mw.p90, mw.top1Share * 100, mw.meanMissMass, mw.missCount))
        }

        // ── SELF-CHECK (bolt): spec vs bolt greedy (same frozen tables = self-consistent) ──
        // ── TEACHER-FORCED FIDELITY (bolt vs strict canonical): QWISP_RAW_TF=1 ──
        // chaos-free 品質軸。canonical ref(spec_greedy)を 1 個ずつ teacher-force し、bolt の
        // argmax が次の canonical トークンと一致する率を数える(MLX bolt TF ~88-97% に相当)。
        // free-run の 3-4% は greedy-chaos で無意味([[bench-harness]])→ これが bolt 品質の正指標。
        if Tell.envFlag("QWISP_RAW_TF") {
            print("[raw-spec bolt] teacher-forced fidelity: feeding canonical ref, counting argmax match ...")
            if let (backendTF, fwdTF, _) = streamingBackend(
                engine: engine, modelDir: modelDir, maxM: maxM, maxSeqLen: maxSeqLen, C: C,
                existingProviders: providers) {
                // ★TF 開始状態の正準化(notes/13): free-run recalib が cache slot を動かした後は
                // phase-5 の `tables` が stale(slot が別 expert を保持)。calib の counts/coact から
                // 「現在の cache 状態と整合する」table を再構築して calib-fresh 状態に戻す。
                // (recalib off なら ensure は全 hit・buildBuddyTable は同一結果=従来と等価。)
                var tfTables: [[Int32]] = []
                for (li, provider) in providers.enumerated() {
                    let top = counts[li].enumerated()
                        .sorted { $0.element != $1.element ? $0.element > $1.element : $0.offset < $1.offset }
                        .prefix(C)
                        .map { $0.offset }
                    _ = provider.cache.ensure(Array(top))
                    provider.cache.buildBuddyTable(coact: coact[li], numExperts: nE)
                    tfTables.append(provider.cache.buddyTableCPU)
                }
                fwdTF.setBoltTables(tfTables)
                if boltBiasEps > 0 {
                    let biasMasks: [[Int32]] = providers.map { p in
                        (0 ..< nE).map { Int32(p.cache.buddyExpertCPU[$0] == $0 ? 1 : 0) }
                    }
                    fwdTF.setRouteBias(masks: biasMasks, eps: boltBiasEps)
                }
                // ── experiment-1 rolling re-calib (staleness 根本治療, TF loop 限定の実験配線) ──
                // QWISP_BOLT_RECALIB_R=<R>: R token 毎に「直近窓の routing 観測(side-buffer)」から
                // counts/coact を再集計し、top-C ensure(pread=io>0!)+ buddy table + bias mask を
                // re-freeze する。窓はリセット(直近分布)。free-run 側は未配線(実験は TF が測定系)。
                let recalibR = boltRecalibR   // notes/13: default 128(free-run と同一 knob)
                var winCounts = [[Int]](repeating: [Int](repeating: 0, count: nE), count: nLayers)
                var winCoact  = [[[Int]]](repeating: [[Int]](repeating: [Int](repeating: 0, count: nE), count: nE),
                                          count: recalibR > 0 ? nLayers : 0)
                var tfIBuf: MTLBuffer? = nil
                var tfRefreshes = 0
                let missBase = LayerExpertCache.missTotal
                if recalibR > 0 {
                    if SeedlessMetalForward.compileDiagCopyRoute(),
                       let (device, _) = SeedlessMetalForward.ensure(),
                       let ib = device.makeBuffer(length: nLayers * Ktop * 4, options: .storageModeShared),
                       let gb = device.makeBuffer(length: nLayers * nE * 2, options: .storageModeShared) {
                        tfIBuf = ib
                        fwdTF.diagRouteBufs = (ib, gb)
                        print("[raw-spec bolt] TF recalib mode: R=\(recalibR)")
                    }
                }
                func tfRecalibRead() {
                    guard let ib = tfIBuf else { return }
                    for li in 0 ..< nLayers {
                        let inds = UnsafeBufferPointer(
                            start: ib.contents().advanced(by: li * Ktop * 4).assumingMemoryBound(to: Int32.self),
                            count: Ktop)
                        let distinct = Array(Set(inds.map { Int($0) }))
                        for e in distinct { winCounts[li][e] += 1 }
                        for ai in 0 ..< distinct.count {
                            for bi in (ai + 1) ..< distinct.count {
                                let a = distinct[ai], b = distinct[bi]
                                winCoact[li][a][b] += 1; winCoact[li][b][a] += 1
                            }
                        }
                    }
                }
                func tfRecalibRefresh() {
                    guard tfIBuf != nil else { return }
                    var newTables: [[Int32]] = []
                    for (li, provider) in providers.enumerated() {
                        let top = winCounts[li].enumerated()
                            .sorted { $0.element != $1.element ? $0.element > $1.element : $0.offset < $1.offset }
                            .prefix(C)
                            .map { $0.offset }
                        _ = provider.cache.ensure(Array(top))   // pread misses = recalib の io コスト
                        provider.cache.buildBuddyTable(coact: winCoact[li], numExperts: nE)
                        newTables.append(provider.cache.buddyTableCPU)
                    }
                    fwdTF.setBoltTables(newTables)
                    if boltBiasEps > 0 {
                        let masks: [[Int32]] = providers.map { p in
                            (0 ..< nE).map { Int32(p.cache.buddyExpertCPU[$0] == $0 ? 1 : 0) }
                        }
                        fwdTF.setRouteBias(masks: masks, eps: boltBiasEps)
                    }
                    for li in 0 ..< nLayers {
                        for e in 0 ..< nE { winCounts[li][e] = 0; winCoact[li][e] = [Int](repeating: 0, count: nE) }
                    }
                    tfRefreshes += 1
                }
                // ── notes/14 §6: async schedule emulation ──────────────────────────
                // free-run async と同じ plan/chunk 分割・同じ token 位置 swap を「同期で」再現し、
                // schedule 分割の fidelity 影響を async 機構なしで測る。TF は同期(GPU idle at CPU
                // turn)なので staging 不要 = victim slot へ直接 pread しても free-run swap と同結果。
                let tfAsyncEmu = boltRefreshAsync && stagingArena != nil
                var tfAsyncChunks: [[(li: Int, job: BoltAsyncRefresh.Job)]] = []
                var tfAsyncNextSwap = 0, tfAsyncBoundary = 0, tfAsyncStride = 1
                var tfAsyncCoactSnap: [[[Int]]] = []
                func tfApplyChunk(_ j: Int) {
                    guard j < tfAsyncChunks.count else { return }
                    for (li, job) in tfAsyncChunks[j] {
                        let cache = providers[li].cache
                        cache.arena.loadMany(cache.layer, [(e: job.expert, slot: job.victimSlot)])
                        let cur = cache.expertAt[job.victimSlot]
                        if cur >= 0 && cur != job.expert { cache.slotOf.removeValue(forKey: cur) }
                        cache.slotOf[job.expert]       = job.victimSlot
                        cache.expertAt[job.victimSlot] = job.expert
                        cache.clock                   += 1
                        cache.tick[job.victimSlot]     = cache.clock
                    }
                    LayerExpertCache.missTotal += tfAsyncChunks[j].count   // pread io 計上(ensure 迂回のため手動)
                    // free-run swap と同じ affected-layers-only 更新(内容は full rebuild と同一)。
                    for li in Set(tfAsyncChunks[j].map { $0.li }).sorted() {
                        let snap = li < tfAsyncCoactSnap.count ? tfAsyncCoactSnap[li]
                                   : [[Int]](repeating: [Int](repeating: 0, count: nE), count: nE)
                        let cache = providers[li].cache
                        cache.buildBuddyTable(coact: snap, numExperts: nE)
                        fwdTF.setBoltTable(li, cache.buddyTableCPU)
                        if boltBiasEps > 0 {
                            fwdTF.updateRouteBiasMask(li, (0 ..< nE).map { Int32(cache.buddyExpertCPU[$0] == $0 ? 1 : 0) })
                        }
                    }
                }
                func tfAsyncPlan(boundary: Int) {
                    // free-run 同様: 前 plan の残 chunk を境界で全消化してから次 plan。
                    while tfAsyncNextSwap < tfAsyncChunks.count { tfApplyChunk(tfAsyncNextSwap); tfAsyncNextSwap += 1 }
                    tfAsyncCoactSnap = winCoact
                    var allJobs: [(li: Int, job: BoltAsyncRefresh.Job)] = []
                    for (li, provider) in providers.enumerated() {
                        if let plan = BoltAsyncRefresh.makePlan(
                            counts: winCounts[li], coact: winCoact[li],
                            slotOf: provider.cache.slotOf, expertAt: provider.cache.expertAt,
                            tick: provider.cache.tick, pinnedSlots: provider.cache.pinnedSlots,
                            C: C, nE: nE, B: boltRefreshB) {
                            for job in plan.jobs { allJobs.append((li, job)) }
                        }
                    }
                    tfAsyncBoundary = boundary
                    tfAsyncChunks = stride(from: 0, to: allJobs.count, by: boltRefreshB).map {
                        Array(allJobs[$0 ..< Swift.min($0 + boltRefreshB, allJobs.count)])
                    }
                    tfAsyncNextSwap = 0
                    tfAsyncStride = refreshStride(tfAsyncChunks.count)
                    for li in 0 ..< nLayers {
                        for e in 0 ..< nE { winCounts[li][e] = 0; winCoact[li][e] = [Int](repeating: 0, count: nE) }
                    }
                    tfRefreshes += 1
                }
                if let lastNormedTF = prefill(promptIds: promptIds, backend: backendTF),
                   let lg0TF = engine.logits(lastNormedTF, M: 1) {
                    MLX.eval([lg0TF])
                    var predTF = MLX.argMax(lg0TF[0], axis: -1).item(Int.self)   // position 0 の予測
                    var tfMatch = 0, tfTotal = 0
                    let nTF = Swift.min(N, gRefIds.count)
                    for i in 0 ..< (nTF - 1) {
                        if predTF == gRefIds[i] { tfMatch += 1 }   // 位置 i の予測 vs canonical[i]
                        tfTotal += 1
                        // experiment-3 horizon-decay: free-run と同じ ε(位置) を TF にも適用。
                        // recalib 併用時は refresh からの相対位置で decay(窓毎に ε が蘇生)。
                        if boltBiasEps > 0 && boltBiasDecayH > 0 {
                            let tPos = recalibR > 0 ? (i % recalibR) : i
                            fwdTF.setRouteBiasEps(boltBiasEps * Swift.max(0, 1 - Float(tPos) / Float(boltBiasDecayH)))
                        }
                        // teacher-force: canonical トークン gRefIds[i] を入力して次位置の予測を得る
                        guard let ev = backendTF.stepArgmax([Int32(gRefIds[i])]) else { break }
                        predTF = ev[0]
                        // experiment-1: routing 観測の累積 + R 境界で re-freeze
                        // notes/14: async 時は emulation(plan + token 位置 chunk swap)、sync 時は従来 bulk refresh。
                        if recalibR > 0 {
                            tfRecalibRead()
                            if (i + 1) % recalibR == 0 {
                                if tfAsyncEmu { tfAsyncPlan(boundary: i + 1) } else { tfRecalibRefresh() }
                            }
                            // free-run と同じ固定 token スケジュール(head 毎に due 分を全消化)。
                            while tfAsyncEmu, tfAsyncNextSwap < tfAsyncChunks.count,
                                  (i + 1) >= tfAsyncBoundary + (tfAsyncNextSwap + 1) * tfAsyncStride {
                                tfApplyChunk(tfAsyncNextSwap); tfAsyncNextSwap += 1
                            }
                        }
                    }
                    if predTF == gRefIds[nTF - 1] { tfMatch += 1 }; tfTotal += 1
                    let tfPct = Double(tfMatch) / Double(Swift.max(tfTotal, 1)) * 100
                    print(String(format: "[RawSpec] bolt TF fidelity vs strict-canonical: %d/%d=%.1f%% (chaos-free)",
                                 tfMatch, tfTotal, tfPct))
                    if recalibR > 0 {
                        print("[RawSpec] TF recalib: refreshes=\(tfRefreshes) preadMisses=\(LayerExpertCache.missTotal - missBase)")
                    }
                }
            }
        }

        // QWISP_RAWSPEC_CHECK=1: rebuild backend #3 reusing providers+tables, run greedy.
        guard Tell.envFlag("QWISP_RAWSPEC_CHECK") else { return summary }

        // notes/13: recalib 有効時は engine が適応的(観測依存の refresh)になり、spec と greedy で
        // 観測窓が異なる=table 軌跡が異なるため「spec≡greedy(同一凍結 table)」の前提が設計上不成立。
        // 決定性ゲート(同 env → byte-identical)が代替。self-check は R=0(凍結 bolt)専用。
        if boltRecalibR > 0 {
            return summary + "\n[RawSpec] bolt self-check SKIPPED (recalib R=\(boltRecalibR) active — adaptive tables; determinism gate applies instead)"
        }

        print("[raw-spec bolt] self-check: bolt greedy M=1 (same frozen tables) ...")
        guard let (backend3, fwd3, _) = streamingBackend(
            engine: engine, modelDir: modelDir, maxM: maxM, maxSeqLen: maxSeqLen, C: C,
            existingProviders: providers)
        else { return summary + "\n[RawSpec] bolt self-check ERROR: backend #3 nil" }
        fwd3.setBoltTables(tables)   // same frozen tables

        guard let lastNormed3 = prefill(promptIds: promptIds, backend: backend3)
        else { return summary + "\n[RawSpec] bolt self-check ERROR: prefill nil" }
        guard let lg003 = engine.logits(lastNormed3, M: 1)
        else { return summary + "\n[RawSpec] bolt self-check ERROR: logits nil" }
        MLX.eval([lg003])
        var uG = MLX.argMax(lg003[0], axis: -1).item(Int.self)
        var greedyOut: [Int] = []
        while greedyOut.count < N {
            guard let evals = backend3.stepArgmax([Int32(uG)])
            else { return summary + "\n[RawSpec] bolt self-check ERROR: greedy step nil" }
            greedyOut.append(uG); uG = evals[0]
        }
        let boltMatch = zip(outN, greedyOut.prefix(N)).filter { $0 == $1 }.count
        let boltTag   = boltMatch == N ? " SELF-CONSISTENT" : " MISMATCH"
        let checkLine = String(format: "\n[RawSpec] bolt self-check spec-vs-bolt-greedy: %d/%d%@  (L3: not lossless vs strict ref)",
                               boltMatch, N, boltTag)
        return summary + checkLine
    }

    // ── Self-check helper ──────────────────────────────────────────────────

    private static func runSelfCheck(engine: SeedlessEngine, promptIds: [Int32], N: Int, maxK: Int,
                                      mkBackend: () -> SpecBackend?, outN: [Int]) -> String {
        print("[raw-spec] self-check: running raw greedy (M=1) for \(N) tokens ...")
        guard let backend2 = mkBackend()
        else { return "\n[RawSpec] self-check ERROR: backend nil" }
        guard let lastNormed2 = prefill(promptIds: promptIds, backend: backend2)
        else { return "\n[RawSpec] self-check ERROR: prefill nil" }

        guard let lg00 = engine.logits(lastNormed2, M: 1)
        else { return "\n[RawSpec] self-check ERROR: logits nil" }
        MLX.eval([lg00])
        var uG = MLX.argMax(lg00[0], axis: -1).item(Int.self)

        var greedyOut: [Int] = []
        while greedyOut.count < N {
            guard let evals = backend2.stepArgmax([Int32(uG)])
            else { return "\n[RawSpec] self-check ERROR: greedy step nil" }
            greedyOut.append(uG)
            uG = evals[0]
        }

        let specMatch = zip(outN, greedyOut.prefix(N)).filter { $0 == $1 }.count
        let checkTag  = specMatch == N ? " LOSSLESS" : " MISMATCH (expected N/N)"
        return String(format: "\n[RawSpec] self-check spec-vs-greedy: %d/%d%@",
                      specMatch, N, checkTag)
    }

    // ── Stream-vs-resident check ───────────────────────────────────────────

    private static func runStreamVsResidentCheck(engine: SeedlessEngine, store: WeightStore,
                                                  modelDir: String, promptIds: [Int32],
                                                  streamOut: [Int], N: Int,
                                                  maxM: Int, maxSeqLen: Int, maxK: Int) -> String {
        print("[raw-spec] stream-vs-resident: loading resident weights ...")
        store.residentAll()
        print("[raw-spec] stream-vs-resident: building resident fused backend ...")
        guard let residentBackend = fusedBackend(engine: engine, maxM: maxM, maxSeqLen: maxSeqLen)
        else { return "\n[RawSpec] stream-vs-resident ERROR: resident backend nil" }

        print("[raw-spec] stream-vs-resident: running resident prefill + spec loop ...")
        let useA3 = Tell.envFlag("QWISP_RAW_A3")
        guard let resOut = runSpecLoop(promptIds: promptIds, backend: residentBackend,
                                        engine: engine, N: N, maxK: maxK, useA3: useA3)
        else { return "\n[RawSpec] stream-vs-resident ERROR: resident spec loop nil" }

        let identical = zip(streamOut, resOut.prefix(N)).filter { $0 == $1 }.count
        let tag = identical == N ? "IDENTICAL" : "MISMATCH at index \(zip(streamOut, resOut).enumerated().first(where: { $0.element.0 != $0.element.1 })?.offset ?? -1)"
        return String(format: "\n[RawSpec] stream-vs-resident: %d/%d %@", identical, N, tag)
    }
}
