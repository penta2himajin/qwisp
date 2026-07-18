import Foundation
import MLX

// #90 Step 0 — seqMT-free verify feasibility measurements (probe only, no engine change).
//
// Q1 (decomposition + M-scaling): where does the M-row verify wall go, and how does
//    r_M = (M-row forward wall) / (M=1 forward wall) scale at M ∈ {2, 4, 9, 17}?
//    r_M ~ O(M) ⇒ verify sequentializes ⇒ seqMT-free is a hard prerequisite for every
//    M>1 consumer (MTP-D1 r_M2, DFlash blocks at M=9/17). r_M ≪ M ⇒ already amortized.
//    Wall vs GPU-busy (profLastGPUMs) splits kernel time from CPU dispatch gap.
//
// Q2 (approach-(B) replay budget): replay-normalized state re-runs ONLY the
//    state-updating path (GDN recurrence) sequentially for committed rows. Its unit
//    cost is bounded by the GDN-recurrence share of an M=1 forward, measured by the
//    profSkipGDNRecur differential (skip-vs-full — the prefill-breakdown idiom).
//    (B) is viable iff committedRows × recurShare < the seqMT tax it removes.
extension Tell {
    public static func seqMTScalingProbe(modelDir: String) -> String {
        guard let store = try? WeightStore(modelDir: modelDir) else { return "[seqmt-m] load fail\nSEQMTM done" }
        store.residentAll()
        let engine = SeedlessEngine.build(store: store)
        let prefillLen = Tell.envInt("QWISP_SEQMT_PREFILL", 512)
        let totalAppend = Tell.envInt("QWISP_SEQMT_TOKENS", 340)   // per-M appended budget (KV growth parity)
        let prompt = (0 ..< prefillLen).map { Int32((($0 &* 7 &+ 13) % 5000) + 100) }
        var lines = ["[seqmt-m] resident C=256, prefill=\(prefillLen), ~\(totalAppend) appended tokens per M"]

        // One measurement pass: fresh backend, prefill, then N M-row forwards.
        // Returns (mean wall ms, mean GPU-busy ms) per forward.
        func run(M: Int, skipRecur: Bool = false, skipMixer: Bool = false, skipMoE: Bool = false) -> (wall: Double, gpu: Double)? {
            guard let b = Tell.fusedBackend(engine: engine, maxM: Swift.max(32, M + 8),
                                            maxSeqLen: prefillLen + totalAppend + 64) else { return nil }
            guard Tell.prefill(promptIds: prompt, backend: b) != nil else { return nil }
            let n = Swift.max(8, totalAppend / M)
            var tok: Int32 = 1000
            func step() -> Bool {
                let ids = (0 ..< M).map { _ -> Int32 in tok += 1; return 1000 + (tok % 4000) }
                return b.forward(ids) != nil
            }
            for _ in 0 ..< 3 { _ = step() }                       // warmup (pipeline compile etc.)
            SeedlessMetalForward.profSkipGDNRecur = skipRecur
            SeedlessMetalForward.profSkipMixer = skipMixer
            SeedlessMetalForward.profSkipMoEExperts = skipMoE
            defer {
                SeedlessMetalForward.profSkipGDNRecur = false
                SeedlessMetalForward.profSkipMixer = false
                SeedlessMetalForward.profSkipMoEExperts = false
            }
            var wall = 0.0, gpu = 0.0
            for _ in 0 ..< n {
                let t0 = Date()
                guard step() else { return nil }
                wall += Date().timeIntervalSince(t0) * 1000
                gpu += SeedlessFusedVerify.SeedlessFusedForward.profLastGPUMs
            }
            return (wall / Double(n), gpu / Double(n))
        }

        // Q1: M-scaling + decomposition.
        guard let base = run(M: 1) else { return "[seqmt-m] M=1 run fail\nSEQMTM done" }
        lines.append(String(format: "  M=%2d  wall %7.2fms  gpu %7.2fms  cpu-gap %6.2fms   r_M = 1.00  (baseline)",
                            1, base.wall, base.gpu, base.wall - base.gpu))
        var rM: [Int: Double] = [1: 1.0]
        for m in [2, 4, 9, 17] {
            guard let r = run(M: m) else { lines.append("  M=\(m) FAIL"); continue }
            rM[m] = r.wall / base.wall
            lines.append(String(format: "  M=%2d  wall %7.2fms  gpu %7.2fms  cpu-gap %6.2fms   r_M = %.2f  (r_M/M = %.2f)",
                                m, r.wall, r.gpu, r.wall - r.gpu, r.wall / base.wall, r.wall / base.wall / Double(m)))
        }

        // Q2: GDN-recurrence share via skip differential (M=1 and M=17).
        lines.append("  — approach-(B) replay budget —")
        for m in [1, 17] {
            guard let full = run(M: m), let skip = run(M: m, skipRecur: true) else { continue }
            let share = (full.wall - skip.wall) / base.wall
            lines.append(String(format: "  M=%2d  GDN-recur cost = %6.2fms (%.2f of an M=1 forward%@)",
                                m, full.wall - skip.wall, share,
                                m == 1 ? " — (B)'s per-committed-row replay unit" : ""))
        }
        if let r2 = rM[2] {
            lines.append(String(format: "  seqMT tax at M=2 (r_M2 − 1) = %.2f  — (B) viable iff committedRows × replayUnit < this", r2 - 1))
        }

        // Component attribution of the M-row marginal cost: differential timing at M=1 vs 2 vs 9
        // for mixer (GDN body + attention) and MoE experts. If the marginal cost lands in the
        // generic components (not the state path), it is honest compute, not a removable tax.
        lines.append("  — marginal-cost attribution (skip differentials, ms) —")
        for m in [1, 2, 9] {
            guard let full = run(M: m), let noMix = run(M: m, skipMixer: true),
                  let noMoE = run(M: m, skipMoE: true) else { continue }
            lines.append(String(format: "  M=%2d  mixer %6.2f  moe %6.2f  rest %6.2f   (wall %6.2f)",
                                m, full.wall - noMix.wall, full.wall - noMoE.wall,
                                noMix.wall + noMoE.wall - full.wall, full.wall))
        }
        lines.append("SEQMTM done")
        return lines.joined(separator: "\n")
    }
}
