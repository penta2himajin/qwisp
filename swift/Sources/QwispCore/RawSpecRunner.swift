import Foundation
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
        return SpecBackend(
            forward: forward,
            stepArgmax: step,
            snapshot: { fwd.snapshot() },
            rollback: { snap in
                guard let s = snap as? RawFusedVerify.RawFusedForward.Snapshot else { return }
                fwd.rollbackOneStep(s)
            })
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
        let backend = SpecBackend(
            forward: forward,
            stepArgmax: step,
            snapshot: { fwd.snapshot() },
            rollback: { snap in
                guard let s = snap as? RawFusedVerify.RawFusedForward.Snapshot else { return }
                fwd.rollbackOneStep(s)
            })
        return (backend, fwd, providers)
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
                             N: Int, maxK: Int) -> [Int]? {
        guard let lastNormed = prefill(promptIds: promptIds, backend: backend) else { return nil }
        guard let lg0 = engine.logits(lastNormed, M: 1) else { return nil }
        MLX.eval([lg0])
        var u = MLX.argMax(lg0[0], axis: -1).item(Int.self)

        var hist = promptIds.map { Int($0) }
        var out: [Int] = []

        while out.count < N {
            let drafts = Tell.suffixDraft(hist + [u], maxMatch: 32, draftK: maxK, minMatch: 4)
            let D      = drafts.count

            let snap = backend.snapshot()

            if D == 0 {
                guard let evals = backend.stepArgmax([Int32(u)]) else { return nil }
                out.append(u); hist.append(u)
                u = evals[0]
                continue
            }

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
        return Array(out.prefix(N))
    }

    // ── Main runner ───────────────────────────────────────────────────────

    public static func run(modelDir: String, refPath: String) throws -> String {
        // ── streaming tier detection ──────────────────────────────────────
        let rawC = Tell.envInt("QWISP_RAW_C", 0)
        let isStreaming = rawC > 0 && rawC < 256
        let isBolt = isStreaming && Tell.envFlag("QWISP_RAW_BOLT")
        var useFused = Tell.envFlag("QWISP_RAW_FUSED")
        if isStreaming && !useFused {
            print("[raw-spec] QWISP_RAW_C=\(rawC) → streaming tier; enabling fused implicitly (set QWISP_RAW_FUSED=1 to suppress this note)")
            useFused = true
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
        print("[raw-spec] promptLen=\(promptIds.count) N=\(N) maxK=\(maxK) fused=\(useFused) streaming=\(isStreaming) C=\(rawC) bolt=\(isBolt)")

        // backend 構築(fused: maxM=verify 最大行数, maxSeqLen=prompt+生成+draft+margin)
        let maxM = Swift.max(maxK + 1, 64)
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
        guard let backend = mkBackend()
        else { return "[raw-spec] ERROR: backend init nil (fused=\(useFused), streaming=\(isStreaming))" }

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

        let t0 = DispatchTime.now()

        while out.count < N {
            let drafts = Tell.suffixDraft(hist + [u], maxMatch: 32, draftK: maxK, minMatch: 4)
            let D      = drafts.count

            // Snapshot before the batched verify (backend-specific representation).
            let snap = backend.snapshot()

            if D == 0 {
                // No draft available: single M=1 step(1-CB: forward+lm_head+argmax)。
                guard let evals = backend.stepArgmax([Int32(u)])
                else { return "[raw-spec] ERROR: step(D=0) nil" }
                out.append(u); hist.append(u)
                u = evals[0]
                steps += 1
                continue
            }

            // Batched verify: [u] + drafts[0..<D], total M = D+1 rows.
            // By raw-engine construction (order-stable, per-row kernels) batched == sequential.
            let verifyTokens: [Int32] = [Int32(u)] + drafts.map { Int32($0) }
            guard let evals = backend.stepArgmax(verifyTokens)
            else { return "[raw-spec] ERROR: verify step nil" }

            // Accept prefix p: longest i s.t. drafts[i] == evals[i] for all i < p.
            var p = 0
            while p < D && drafts[p] == evals[p] { p += 1 }

            if p == D {
                // ── full accept ───────────────────────────────────────────
                // Committed: u + all D drafts (D+1 tokens).  Bonus: evals[D].
                // Caches advanced by D+1 positions — exactly the committed tokens. ✓
                out.append(u); hist.append(u)
                for d in drafts { out.append(d); hist.append(d) }
                accTok += D
                steps  += 1
                u = evals[D]
            } else {
                // ── partial reject ────────────────────────────────────────
                // 1. Rollback caches to the pre-verify snapshot(直前 1 forward の取り消し)。
                backend.rollback(snap)
                // 2. Commit u + accepted drafts[0..<p].
                out.append(u); hist.append(u)
                for d in drafts.prefix(p) { out.append(d); hist.append(d) }
                accTok += p
                steps  += 1
                // 3. Rebuild forward with the p+1 committed tokens to advance caches.
                //    We do NOT call logits here — evals[p] from the batched verify is
                //    bit-identical (same order-stable engine).
                let rebuildTokens: [Int32] = [Int32(u)] + drafts.prefix(p).map { Int32($0) }
                guard let _ = backend.forward(rebuildTokens)
                else { return "[raw-spec] ERROR: rebuild forwardRows nil" }
                u = evals[p]
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
            statsLine = String(format: "\n[RawSpec] streaming stats: ensure=%.1fms pread=%.1fms misses=%d",
                               ensMs, preadMs, LayerExpertCache.missTotal)
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

        // ── phase 6: bolt spec decode loop ────────────────────────────────
        var hist = promptIds.map { Int($0) }
        var out: [Int] = []
        var accTok = 0, steps = 0
        var u = u0

        let t0 = DispatchTime.now()

        while out.count < N {
            let drafts = Tell.suffixDraft(hist + [u], maxMatch: 32, draftK: maxK, minMatch: 4)
            let D      = drafts.count
            let snap   = backend2.snapshot()

            if D == 0 {
                guard let evals = backend2.stepArgmax([Int32(u)])
                else { return "[raw-spec bolt] ERROR: step(D=0) nil" }
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

        // ── SELF-CHECK (bolt): spec vs bolt greedy (same frozen tables = self-consistent) ──
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
        guard let resOut = runSpecLoop(promptIds: promptIds, backend: residentBackend,
                                        engine: engine, N: N, maxK: maxK)
        else { return "\n[RawSpec] stream-vs-resident ERROR: resident spec loop nil" }

        let identical = zip(streamOut, resOut.prefix(N)).filter { $0 == $1 }.count
        let tag = identical == N ? "IDENTICAL" : "MISMATCH at index \(zip(streamOut, resOut).enumerated().first(where: { $0.element.0 != $0.element.1 })?.offset ?? -1)"
        return String(format: "\n[RawSpec] stream-vs-resident: %d/%d %@", identical, N, tag)
    }
}
