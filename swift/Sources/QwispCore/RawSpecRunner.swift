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
public enum RawSpecRunner {

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
    static func makeStepArgmax(engine: RawEngine, forward: @escaping ([Int32]) -> MLXArray?) -> ([Int32]) -> [Int]? {
        return { tokens in
            guard let n = forward(tokens), let l = engine.logits(n, M: tokens.count) else { return nil }
            MLX.eval([l])
            return (0 ..< tokens.count).map { MLX.argMax(l[$0], axis: -1).item(Int.self) }
        }
    }

    /// composed backend(per-op CB、MLX cache 参照 snapshot)。
    static func composedBackend(engine: RawEngine) -> SpecBackend {
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
                guard let snaps = snap as? [RawVerifyForward.LayerCaches] else { return }
                for (i, s) in snaps.enumerated() {
                    caches[i].kCache = s.kCache; caches[i].vCache = s.vCache
                    caches[i].convState = s.convState; caches[i].recState = s.recState
                }
            })
    }

    /// fused backend(1-CB step: embed→40層→final norm→lm_head→argmax、readback は token id のみ。
    /// rollback は KV len 巻き戻し+ping-pong swap)。
    static func fusedBackend(engine: RawEngine, maxM: Int, maxSeqLen: Int) -> SpecBackend? {
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
                guard let s = snap as? RawFusedVerify.RawFusedForward.Snapshot else { return }
                fwd.rollbackOneStep(s)
            })
        // Phase II-a: wire chained greedy decode when the 1-CB head path is active.
        // Only resident/bolt return non-nil (strict returns nil → per-step fallback).
        if fwd.head != nil {
            backend.chainedStepArgmax = { token, k in fwd.chainedStepArgmax(token, K: k) }
        }
        return backend
    }

    /// streaming fused backend(strict segmented per-layer CB)。
    /// existingProviders を渡すと arena を再利用し fresh forward のみ構築する(bolt phase 2 用)。
    static func streamingBackend(engine: RawEngine, modelDir: String, maxM: Int, maxSeqLen: Int, C: Int,
                                  existingProviders: [ArenaExpertProvider]? = nil)
        -> (SpecBackend, RawFusedVerify.RawFusedForward, [ArenaExpertProvider])? {
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
                guard let s = snap as? RawFusedVerify.RawFusedForward.Snapshot else { return }
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
    // Note: strict streaming intentionally exposes NO chain fn (RawFusedForward.
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

    // ── Prefill helper ────────────────────────────────────────────────────

    /// Chunked prefill: runs all prompt tokens through the backend, chunk=64.
    /// Returns normed hidden of the very last position [1, H], or nil on error.
    static func prefill(promptIds: [Int32], backend: SpecBackend) -> MLXArray? {
        let pLen = promptIds.count
        guard pLen > 0 else { return nil }
        let chunkSize = 64
        var lastNormed: MLXArray? = nil
        var pos = 0
        while pos < pLen {
            let end = Swift.min(pos + chunkSize, pLen)
            let chunk = Array(promptIds[pos ..< end])
            guard let normed = backend.forward(chunk) else { return nil }
            // Keep last row [H] — will be overwritten each chunk until the final one.
            lastNormed = normed[chunk.count - 1]    // [H]
            pos = end
        }
        return lastNormed.map { $0.reshaped([1, RawEngine.H]) }   // [1, H]
    }

    // ── Spec loop helper ──────────────────────────────────────────────────

    /// SuffixSpec ループ本体(main run / self-check / stream-vs-resident check の 3 箇所で共用)。
    /// Returns out[0..<N] token ids.
    static func runSpecLoop(promptIds: [Int32], backend: SpecBackend, engine: RawEngine,
                             N: Int, maxK: Int, useA3: Bool = false) -> [Int]? {
        guard let lastNormed = prefill(promptIds: promptIds, backend: backend) else { return nil }
        guard let lg0 = engine.logits(lastNormed, M: 1) else { return nil }
        MLX.eval([lg0])
        var u = MLX.argMax(lg0[0], axis: -1).item(Int.self)

        var hist = promptIds.map { Int($0) }
        var out: [Int] = []
        var pending: [Int] = []  // A3: pending prefix tokens
        let pendingCap = 24

        while out.count < N {
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
        return Array(out.prefix(N))
    }

    // ── Tier-gating resolution seams (pure/testable) ───────────────────────
    //
    // recon #16 fix: two default-path wiring changes, both lossless (output
    // byte-identical; only default selection differs). Factored as pure helpers
    // so the resolved defaults are unit-testable via the production seam without
    // a real model (RawVerifyTests 52/53). run() MUST call these.
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

    public static func run(modelDir: String, refPath: String) throws -> String {
        // ── streaming tier detection ──────────────────────────────────────
        let env = ProcessInfo.processInfo.environment
        let rawC = resolveRawC(envC: env["QWISP_RAW_C"], defaultC: DeviceCalibration.defaultC())
        let isStreaming = rawC > 0 && rawC < 256
        let isBolt = isStreaming && Tell.envFlag("QWISP_RAW_BOLT")
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

        print("[raw-spec] building RawEngine ...")
        let engine = RawEngine.build(store: store)
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
        var _reuseStreamFwd: RawFusedVerify.RawFusedForward? = nil
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
        let backend: SpecBackend
        if isStreaming {
            guard let (b, fwd, providers) = streamingBackend(engine: engine, modelDir: modelDir,
                                                              maxM: maxM, maxSeqLen: maxSeqLen, C: rawC)
            else { return "[raw-spec] ERROR: backend init nil (fused=\(useFused), streaming=\(isStreaming))" }
            _reuseStreamFwd = fwd
            _reuseStreamProviders = providers
            backend = b
        } else {
            guard let b = mkBackend()
            else { return "[raw-spec] ERROR: backend init nil (fused=\(useFused), streaming=\(isStreaming))" }
            backend = b
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
        guard let lastNormed = prefill(promptIds: promptIds, backend: backend)
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

        // Phase II-a: QWISP_CHAIN_K=<k>opt-in GPU token-feedback chained greedy decode.
        // Only the D==0 non-A3/empty-pending greedy span uses the chain path; A3 and draft-
        // bearing steps keep the per-step path (snapshot/rollback + suffix-draft needs CPU tokens).
        let chainK = Tell.envInt("QWISP_CHAIN_K", RawFusedVerify.RawFusedForward.chainKDefault)

        let t0 = DispatchTime.now()

        while out.count < N {
            let _reuseArg: (ctx: ReuseContext, residentPerLayer: [Set<Int>], alpha: Double)? = _useRerank ? {
                let resPerLayer = _reuseStreamProviders?.map { Set($0.cache.slotOf.keys) } ?? []
                return (ctx: _reuseCtx, residentPerLayer: resPerLayer, alpha: _reuseAlpha)
            }() : nil
            let drafts = Tell.suffixDraft(hist + [u], maxMatch: 32, draftK: maxK, minMatch: 4,
                                          reuseCtx: _reuseArg)
            let D      = drafts.count

            // Snapshot before the batched verify (backend-specific representation).
            var snap = backend.snapshot()

            if D == 0 {
                // No draft available
                // Phase II-a chain path: only the non-A3 / empty-pending greedy span.
                // chainedStepArgmax is bit-exact to K sequential stepArgmax([t]) calls, so OUT_TOKENS
                // stays byte-identical to per-step while collapsing K CPU round-trips to 1.
                if chainK > 0, let chainFn = backend.chainedStepArgmax,
                   (!useA3 || pending.isEmpty) {
                    if let chainResult = chainFn(Int32(u), chainK), !chainResult.isEmpty {
                        out.append(u); hist.append(u)
                        let addN = Swift.min(chainResult.count - 1, N - out.count)
                        for i in 0 ..< addN {
                            out.append(chainResult[i]); hist.append(chainResult[i])
                        }
                        u = chainResult[chainResult.count - 1]
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
                    u = evals[pending.count]  // argmax at position pk (where u is)
                    pending = []
                } else {
                    // Non-A3 or empty pending: simple single step
                    if _useRerank { _reuseVerifyToks = [Int(u)] }
                    guard let evals = backend.stepArgmax([Int32(u)])
                    else { return "[raw-spec] ERROR: step(D=0) nil" }
                    out.append(u); hist.append(u)
                    u = evals[0]
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

                    // Cap flush: only if pending exceeds 24 (safety to bound M)
                    if pending.count >= pendingCap {
                        let pendingTokens: [Int32] = pending.map { Int32($0) }
                        if _useRerank { _reuseVerifyToks = [] }
                        guard let _ = backend.forward(pendingTokens)
                        else { return "[raw-spec] ERROR: A3 pending flush nil" }
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
    private static func runBoltMode(engine: RawEngine, modelDir: String, promptIds: [Int32],
                                     gRefIds: [Int], N: Int, maxK: Int, C: Int,
                                     maxM: Int, maxSeqLen: Int, refPath: String) throws -> String {
        let calibN = Tell.envInt("QWISP_CALIB", 48)
        let nLayers = RawEngine.numLayers
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

        // ── QWISP_BOLT_DIAG=1: routing telemetry (notes/11 案B Stage 0, measurement-only) ──
        // 層別 side-buffer に route 直後(remap 前)の inds/gl をコピーし、greedy step 毎に CPU 読出。
        // cold-selection 率と near-tie margin(常駐が +ε で置換に必要な量)を集計する。
        // diag 中は chain を無効化(1 CB=1 step にして step 毎読出を可能に)。OUT_TOKENS は不変。
        let boltDiag = Tell.envFlag("QWISP_BOLT_DIAG")
        let Ktop = 8
        var diagResidents: [Set<Int>] = [], diagBuddyExp: [[Int32]] = []
        var diagIBuf: MTLBuffer? = nil, diagGBuf: MTLBuffer? = nil
        if boltDiag {
            guard RawMetalForward.compileDiagCopyRoute(),
                  let (device, _) = RawMetalForward.ensure(),
                  let ib = device.makeBuffer(length: nLayers * Ktop * 4, options: .storageModeShared),
                  let gb = device.makeBuffer(length: nLayers * nE * 2, options: .storageModeShared)
            else { return "[raw-spec bolt] ERROR: diag setup failed" }
            diagIBuf = ib; diagGBuf = gb
            fwd2.diagRouteBufs = (ib, gb)
            for p in providers {
                diagResidents.append(Set(p.cache.slotOf.keys))
                diagBuddyExp.append(p.cache.buddyExpertCPU)
            }
            print("[raw-spec bolt] diag mode: telemetry on, chain disabled")
        }
        // group(li): early 0-12 / mid 13-26 / late 27-39
        struct DiagAcc { var cold = 0; var routed = 0; var stepsWithCold = 0; var margins: [Float] = [] }
        var diagAcc = [DiagAcc(), DiagAcc(), DiagAcc()]
        var diagSteps = 0
        func diagReadStep() {
            guard let ib = diagIBuf, let gb = diagGBuf else { return }
            diagSteps += 1
            var groupHadCold = [false, false, false]
            for li in 0 ..< nLayers {
                let indsArr = Array(UnsafeBufferPointer(
                    start: ib.contents().advanced(by: li * Ktop * 4).assumingMemoryBound(to: Int32.self),
                    count: Ktop))
                let glArr = Array(UnsafeBufferPointer(
                    start: gb.contents().advanced(by: li * nE * 2).assumingMemoryBound(to: Float16.self),
                    count: nE))
                let (cold, mar) = RawFusedVerify.RawFusedForward.computeRouteDiag(
                    inds: indsArr, gl: glArr, resident: diagResidents[li],
                    buddyExpert: diagBuddyExp[li], Ktop: Ktop)
                let g = li <= 12 ? 0 : (li <= 26 ? 1 : 2)
                diagAcc[g].cold += cold.count
                diagAcc[g].routed += Ktop
                diagAcc[g].margins.append(contentsOf: mar)
                if !cold.isEmpty { groupHadCold[g] = true }
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
        let boltChainK = boltDiag ? 0 : Tell.envInt("QWISP_CHAIN_K", RawFusedVerify.RawFusedForward.chainKDefault)

        let t0 = DispatchTime.now()

        while out.count < N {
            let drafts = Tell.suffixDraft(hist + [u], maxMatch: 32, draftK: maxK, minMatch: 4)
            let D      = drafts.count
            let snap   = backend2.snapshot()

            if D == 0 {
                // Phase II-b: delegate to boltGreedyChainSpan when chainK>0 and head is wired.
                if let (emitted, nextU) = boltGreedyChainSpan(
                    backend: backend2, u: u, chainK: boltChainK,
                    budget: N - out.count) {
                    for t in emitted { out.append(t); hist.append(t) }
                    u = nextU; steps += 1; continue
                }
                // chain unavailable or disabled → per-step fallback.
                guard let evals = backend2.stepArgmax([Int32(u)])
                else { return "[raw-spec bolt] ERROR: step(D=0) nil" }
                if boltDiag { diagReadStep() }   // M==1 greedy step: side-buffers fresh
                out.append(u); hist.append(u)
                u = evals[0]; steps += 1; continue
            }

            let verifyTokens: [Int32] = [Int32(u)] + drafts.map { Int32($0) }
            guard let evals = backend2.stepArgmax(verifyTokens)
            else { return "[raw-spec bolt] ERROR: verify step nil" }

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
                fwdTF.setBoltTables(tables)   // same frozen bolt tables
                if let lastNormedTF = prefill(promptIds: promptIds, backend: backendTF),
                   let lg0TF = engine.logits(lastNormedTF, M: 1) {
                    MLX.eval([lg0TF])
                    var predTF = MLX.argMax(lg0TF[0], axis: -1).item(Int.self)   // position 0 の予測
                    var tfMatch = 0, tfTotal = 0
                    let nTF = Swift.min(N, gRefIds.count)
                    for i in 0 ..< (nTF - 1) {
                        if predTF == gRefIds[i] { tfMatch += 1 }   // 位置 i の予測 vs canonical[i]
                        tfTotal += 1
                        // teacher-force: canonical トークン gRefIds[i] を入力して次位置の予測を得る
                        guard let ev = backendTF.stepArgmax([Int32(gRefIds[i])]) else { break }
                        predTF = ev[0]
                    }
                    if predTF == gRefIds[nTF - 1] { tfMatch += 1 }; tfTotal += 1
                    let tfPct = Double(tfMatch) / Double(Swift.max(tfTotal, 1)) * 100
                    print(String(format: "[RawSpec] bolt TF fidelity vs strict-canonical: %d/%d=%.1f%% (chaos-free)",
                                 tfMatch, tfTotal, tfPct))
                }
            }
        }

        // QWISP_RAWSPEC_CHECK=1: rebuild backend #3 reusing providers+tables, run greedy.
        guard Tell.envFlag("QWISP_RAWSPEC_CHECK") else { return summary }

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

    private static func runSelfCheck(engine: RawEngine, promptIds: [Int32], N: Int, maxK: Int,
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

    private static func runStreamVsResidentCheck(engine: RawEngine, store: WeightStore,
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
