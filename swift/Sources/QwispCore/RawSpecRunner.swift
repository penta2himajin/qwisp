import Foundation
import MLX
import MLXFast

/// U2b: raw-spec — SuffixSpec speculative loop on the SELF-CONSISTENT raw engine.
/// Resident tier (C=256 semantics): full expert tensors from store, no arena/guard/cert/VSEQ.
/// Batched verify == sequential by construction (proven by raw-smoke U2a) => strict lossless.
///
/// QWISP_RUN=raw-spec  QWISP_GEN=48  QWISP_DRAFT_K=96
/// QWISP_RAWSPEC_CHECK=1   — also run pure sequential raw greedy and report spec-vs-greedy k/N
/// QWISP_DUMP_TOKENS=1     — print PROMPT_TOKENS / OUT_TOKENS lines (bench correctness axis)
public enum RawSpecRunner {

    // ── Prefill helper ────────────────────────────────────────────────────

    /// Chunked prefill: runs all prompt tokens through the engine, chunk=64.
    /// Returns normed hidden of the very last position [1, H], or nil on error.
    static func prefill(promptIds: [Int32], engine: RawEngine,
                        caches: [RawVerifyForward.LayerCaches]) -> MLXArray? {
        let pLen = promptIds.count
        guard pLen > 0 else { return nil }
        let chunkSize = 64
        var lastNormed: MLXArray? = nil
        var pos = 0
        while pos < pLen {
            let end = Swift.min(pos + chunkSize, pLen)
            let chunk = Array(promptIds[pos ..< end])
            let M     = chunk.count
            let x     = engine.embed(tokens: chunk)          // [M, H]
            guard let normed = engine.forwardRows(x, caches: caches, M: M)
            else { return nil }
            // Keep last row [H] — will be overwritten each chunk until the final one.
            lastNormed = normed[M - 1]    // [H]
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
        print("[raw-spec] promptLen=\(promptIds.count) N=\(N) maxK=\(maxK)")

        // ── PREFILL ───────────────────────────────────────────────────────
        let caches = engine.freshCaches()
        guard let lastNormed = prefill(promptIds: promptIds, engine: engine, caches: caches)
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

            // Snapshot all caches before the batched verify.
            // LayerCaches is a final class; copyState() copies the four MLXArray references.
            // Because MLXArray values are immutable (new ops create new arrays), the snapshot
            // retains the pre-verify tensors even after verifyForwardRows stores new ones.
            let snaps = caches.map { $0.copyState() }

            if D == 0 {
                // No draft available: single M=1 forward.
                let xS = engine.embed(tokens: [Int32(u)])
                guard let nS = engine.forwardRows(xS, caches: caches, M: 1)
                else { return "[raw-spec] ERROR: forward(D=0) nil" }
                guard let lS = engine.logits(nS, M: 1)
                else { return "[raw-spec] ERROR: logits(D=0) nil" }
                MLX.eval([lS])
                out.append(u); hist.append(u)
                u = MLX.argMax(lS[0], axis: -1).item(Int.self)
                steps += 1
                continue
            }

            // Batched verify: [u] + drafts[0..<D], total M = D+1 rows.
            // By raw-engine construction (order-stable, per-row kernels) batched == sequential.
            let verifyTokens: [Int32] = [Int32(u)] + drafts.map { Int32($0) }
            let xV = engine.embed(tokens: verifyTokens)                      // [D+1, H]
            guard let nV = engine.forwardRows(xV, caches: caches, M: D + 1)
            else { return "[raw-spec] ERROR: verify forwardRows nil" }
            guard let lV = engine.logits(nV, M: D + 1)
            else { return "[raw-spec] ERROR: verify logits nil" }
            MLX.eval([lV])

            // Per-row argmax on CPU.
            let evals: [Int] = (0 ..< (D + 1)).map { i in
                MLX.argMax(lV[i], axis: -1).item(Int.self)
            }

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
                // 1. Rollback caches to pre-verify snapshot.
                for (i, snap) in snaps.enumerated() {
                    caches[i].kCache    = snap.kCache
                    caches[i].vCache    = snap.vCache
                    caches[i].convState = snap.convState
                    caches[i].recState  = snap.recState
                }
                // 2. Commit u + accepted drafts[0..<p].
                out.append(u); hist.append(u)
                for d in drafts.prefix(p) { out.append(d); hist.append(d) }
                accTok += p
                steps  += 1
                // 3. Rebuild forward with the p+1 committed tokens to advance caches.
                //    We do NOT call logits here — evals[p] from the batched verify is
                //    bit-identical (same order-stable engine).
                let rebuildTokens: [Int32] = [Int32(u)] + drafts.prefix(p).map { Int32($0) }
                let xR = engine.embed(tokens: rebuildTokens)
                guard let _ = engine.forwardRows(xR, caches: caches, M: p + 1)
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
        let caches2 = engine.freshCaches()
        guard let lastNormed2 = prefill(promptIds: promptIds, engine: engine, caches: caches2)
        else { return summary + "\n[RawSpec] self-check ERROR: prefill nil" }

        guard let lg00 = engine.logits(lastNormed2, M: 1)
        else { return summary + "\n[RawSpec] self-check ERROR: logits nil" }
        MLX.eval([lg00])
        var uG = MLX.argMax(lg00[0], axis: -1).item(Int.self)

        var greedyOut: [Int] = []
        while greedyOut.count < N {
            let xG = engine.embed(tokens: [Int32(uG)])
            guard let nG = engine.forwardRows(xG, caches: caches2, M: 1)
            else { return summary + "\n[RawSpec] self-check ERROR: greedy forward nil" }
            guard let lG = engine.logits(nG, M: 1)
            else { return summary + "\n[RawSpec] self-check ERROR: greedy logits nil" }
            MLX.eval([lG])
            greedyOut.append(uG)
            uG = MLX.argMax(lG[0], axis: -1).item(Int.self)
        }

        let specMatch = zip(outN, greedyOut.prefix(N)).filter { $0 == $1 }.count
        let checkTag  = specMatch == N ? " LOSSLESS" : " MISMATCH (expected N/N)"
        let checkLine = String(format: "\n[RawSpec] self-check spec-vs-greedy: %d/%d%@",
                               specMatch, N, checkTag)
        return summary + checkLine
    }
}
