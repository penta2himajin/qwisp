import Foundation
import MLX
import MLXFast

/// U2b: raw-spec — SuffixSpec speculative loop on the SELF-CONSISTENT raw engine.
/// Resident tier (C=256 semantics): full expert tensors from store, no arena/guard/cert/VSEQ.
/// Batched verify == sequential by construction (proven by raw-smoke U2a) => strict lossless.
///
/// QWISP_RUN=raw-spec  QWISP_GEN=48  QWISP_DRAFT_K=96
/// QWISP_RAW_FUSED=1       — P3 fused backend(全層+final norm を 1 CB、cache GPU 常駐)
/// QWISP_RAWSPEC_CHECK=1   — also run pure sequential raw greedy and report spec-vs-greedy k/N
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

    // ── Main runner ───────────────────────────────────────────────────────

    public static func run(modelDir: String, refPath: String) throws -> String {
        print("[raw-spec] loading model from \(modelDir) ...")
        let store = try WeightStore(modelDir: modelDir)
        store.residentAll()
        print("[raw-spec] residentAll complete")

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
        let maxK = Tell.envInt("QWISP_DRAFT_K", 96)          // resident default; alpha*p cap from suffixDraft
        let useFused = Tell.envFlag("QWISP_RAW_FUSED")
        print("[raw-spec] promptLen=\(promptIds.count) N=\(N) maxK=\(maxK) fused=\(useFused)")

        // backend 構築(fused: maxM=verify 最大行数, maxSeqLen=prompt+生成+draft+margin)
        let maxM = Swift.max(maxK + 1, 64)
        let maxSeqLen = promptIds.count + N + maxK + 64
        let mkBackend: () -> SpecBackend? = {
            useFused ? fusedBackend(engine: engine, maxM: maxM, maxSeqLen: maxSeqLen)
                     : composedBackend(engine: engine)
        }
        guard let backend = mkBackend()
        else { return "[raw-spec] ERROR: backend init nil (fused=\(useFused))" }

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

        let summary = String(format:
            "[RawSpec] resident raw engine: %.1f tok/s  accept/step=%.2f  品質(vs ref spec_greedy) %d/%d=%.0f%%",
            tokps,
            Double(accTok) / Double(Swift.max(steps, 1)),
            match, N, Double(match) / Double(N) * 100)

        // ── SELF-CHECK: spec output vs pure sequential raw greedy ─────────
        // QWISP_RAWSPEC_CHECK=1: run M=1 greedy from scratch and verify N/N lossless.
        guard Tell.envFlag("QWISP_RAWSPEC_CHECK") else { return summary }

        print("[raw-spec] self-check: running raw greedy (M=1) for \(N) tokens ...")
        guard let backend2 = mkBackend()
        else { return summary + "\n[RawSpec] self-check ERROR: backend nil" }
        guard let lastNormed2 = prefill(promptIds: promptIds, backend: backend2)
        else { return summary + "\n[RawSpec] self-check ERROR: prefill nil" }

        guard let lg00 = engine.logits(lastNormed2, M: 1)
        else { return summary + "\n[RawSpec] self-check ERROR: logits nil" }
        MLX.eval([lg00])
        var uG = MLX.argMax(lg00[0], axis: -1).item(Int.self)

        var greedyOut: [Int] = []
        while greedyOut.count < N {
            guard let evals = backend2.stepArgmax([Int32(uG)])
            else { return summary + "\n[RawSpec] self-check ERROR: greedy step nil" }
            greedyOut.append(uG)
            uG = evals[0]
        }

        let specMatch = zip(outN, greedyOut.prefix(N)).filter { $0 == $1 }.count
        let checkTag  = specMatch == N ? " LOSSLESS" : " MISMATCH (expected N/N)"
        let checkLine = String(format: "\n[RawSpec] self-check spec-vs-greedy: %d/%d%@",
                               specMatch, N, checkTag)
        return summary + checkLine
    }
}
