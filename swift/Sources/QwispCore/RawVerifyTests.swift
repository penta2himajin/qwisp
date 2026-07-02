import Foundation
import MLX
import MLXFast
import MLXRandom

/// D1 TDD RED phase — M-row (batched, order-stable) kernel scaffolding.
///
/// All 7 kernel-level tests FAIL initially because the D1 stub APIs in
/// RawMetalForward return nil.  Two baseline tests PASS to confirm the
/// harness itself is correct (existing M=1 kernels are deterministic).
///
/// Run:  QWISP_RUN=raw-tests ./qwisp-poc stream
///   or: qwisp/test_raw.sh
public enum RawVerifyTests {

    // ── Bit-exact comparison helper ───────────────────────────────────────

    /// Compare two MLXArrays element-wise at Float32 precision.
    /// Returns (true, "ok") on exact match; (false, detail) on first nonzero diff.
    static func bitEqual(_ a: MLXArray, _ b: MLXArray) -> (Bool, String) {
        let af = a.reshaped([-1]).asType(.float32)
        let bf = b.reshaped([-1]).asType(.float32)
        MLX.eval([af, bf])
        let na = af.size, nb = bf.size
        guard na == nb else {
            return (false, "size \(a.shape)(=\(na)) vs \(b.shape)(=\(nb))")
        }
        let aArr = af.asArray(Float.self)
        let bArr = bf.asArray(Float.self)
        var maxDiff: Float = 0
        var firstIdx = -1
        for i in 0..<na {
            let d = abs(aArr[i] - bArr[i])
            if d > maxDiff {
                maxDiff = d
                if firstIdx < 0 { firstIdx = i }
            }
        }
        if maxDiff == 0 { return (true, "ok") }
        return (false,
                "max|Δ|=\(maxDiff) first@idx=\(firstIdx) got=\(aArr[firstIdx]) ref=\(bArr[firstIdx])")
    }

    // ── Main suite ────────────────────────────────────────────────────────

    public static func runAll() -> String {
        // Fixed seed → deterministic RNG across all tests
        MLXRandom.seed(UInt64(42))
        var lines: [String] = []
        var passed = 0
        let total = 9

        // Nested runner: records result and increments counter
        func run(_ name: String, body: () -> (Bool, String)) {
            let (ok, detail) = body()
            lines.append(ok
                ? "[raw-test] \(name): PASS"
                : "[raw-test] \(name): FAIL(\(detail))")
            if ok { passed += 1 }
        }

        // ── PASS baselines ────────────────────────────────────────────────
        // Verify that the harness itself is correct: existing M=1 kernels are
        // deterministic (same inputs → bit-identical outputs on two calls).

        run("qmm4_m1_selfconsistent") {
            let K = 2048, N = 2048
            let x  = MLXRandom.normal([1, K]).asType(.float16)
            let wf = MLXRandom.normal([N, K]).asType(.float16)
            let (wq, sc, biOpt) = MLX.quantized(wf, groupSize: 64, bits: 4, mode: .affine)
            guard let bi = biOpt else { return (false, "biases nil") }
            MLX.eval([x, wq, sc, bi])
            guard let a = RawMetalForward.qmm(x, wq, scales: sc, biases: bi, M: 1, K: K, N: N),
                  let b = RawMetalForward.qmm(x, wq, scales: sc, biases: bi, M: 1, K: K, N: N)
            else { return (false, "qmm returned nil") }
            MLX.eval([a, b])
            return bitEqual(a, b)
        }

        run("gqmm4_m1_selfconsistent") {
            let E = 64, K = 2048, N = 512, Ktop = 4
            let wf = MLXRandom.normal([E, N, K]).asType(.float16)
            let (wq, sc, biOpt) = MLX.quantized(wf, groupSize: 64, bits: 4, mode: .affine)
            guard let bi = biOpt else { return (false, "biases nil") }
            let x = MLXRandom.normal([1, K]).asType(.float16)
            let indsArr: [Int32] = [3, 17, 40, 62]
            let inds = MLXArray(indsArr, [Ktop])
            MLX.eval([x, wq, sc, bi, inds])
            guard let a = RawMetalForward.gatherQmm(x, wq, scales: sc, biases: bi,
                                                     inds: inds, Ktop: Ktop, K: K, N: N),
                  let b = RawMetalForward.gatherQmm(x, wq, scales: sc, biases: bi,
                                                     inds: inds, Ktop: Ktop, K: K, N: N)
            else { return (false, "gatherQmm returned nil") }
            MLX.eval([a, b])
            return bitEqual(a, b)
        }

        // ── RED tests: all FAIL because D1 stubs return nil ──────────────

        // Test 1: qmm4_rows_bitexact
        // Reference: per-row loop of existing M=1 qmm, results concatenated → [M, N].
        // Also exercises N=8192 (lm_head stand-in) to catch wide-N regressions.
        run("qmm4_rows_bitexact") {
            let K = 2048
            for N in [2048, 8192] {
                let wf = MLXRandom.normal([N, K]).asType(.float16)
                let (wq, sc, biOpt) = MLX.quantized(wf, groupSize: 64, bits: 4, mode: .affine)
                guard let bi = biOpt else { return (false, "biases nil N=\(N)") }
                MLX.eval([wq, sc, bi])
                for M in [1, 2, 9, 17, 25] {
                    let x = MLXRandom.normal([M, K]).asType(.float16); x.eval()
                    // Reference: loop M=1 qmm
                    var refParts: [MLXArray] = []
                    for m in 0..<M {
                        let xm = x[m ..< m+1]   // [1, K]
                        guard let r = RawMetalForward.qmm(xm, wq, scales: sc, biases: bi,
                                                           M: 1, K: K, N: N)
                        else { return (false, "ref qmm nil N=\(N) M=\(M) m=\(m)") }
                        r.eval(); refParts.append(r)
                    }
                    let ref = MLX.concatenated(refParts, axis: 0); ref.eval()   // [M, N]
                    // Stub
                    guard let got = RawMetalForward.qmmRows(x, wq, scales: sc, biases: bi,
                                                             M: M, K: K, N: N)
                    else { return (false, "not implemented (N=\(N) M=\(M))") }
                    got.eval()
                    let (ok, d) = bitEqual(got, ref)
                    if !ok { return (false, "N=\(N) M=\(M): \(d)") }
                }
            }
            return (true, "ok")
        }

        // Test 2: gqmm4_rows_bitexact
        // Reference: per-row gatherQmm with per-row inds[Ktop], concat → [M*Ktop, N].
        // inds[M*Ktop] row-major mirrors verify's MoE shape (each row routes to Ktop experts).
        run("gqmm4_rows_bitexact") {
            let E = 64, K = 2048, N = 512, Ktop = 4
            let wf = MLXRandom.normal([E, N, K]).asType(.float16)
            let (wq, sc, biOpt) = MLX.quantized(wf, groupSize: 64, bits: 4, mode: .affine)
            guard let bi = biOpt else { return (false, "biases nil") }
            MLX.eval([wq, sc, bi])
            // Fixed pool of expert indices — cycles so all M variants are covered
            let pool: [Int32] = [3, 17, 40, 62,  0, 10, 25, 50, 33,  7,
                                 55, 20, 41, 15,  2, 60,  8, 30, 11, 45,
                                 22, 38, 61,  5, 18, 44, 29,  1, 36, 52,
                                  6, 19,  9, 27, 48, 14, 37,  4, 23, 53,
                                 12, 16, 31, 39, 47, 57, 13, 21, 32, 49]
            for M in [1, 2, 9, 17, 25] {
                let x = MLXRandom.normal([M, K]).asType(.float16)
                let indsFlat = (0..<M*Ktop).map { pool[$0 % pool.count] }
                let inds = MLXArray(indsFlat, [M * Ktop])
                MLX.eval([x, inds])
                // Reference: per-row loop of existing gatherQmm
                var refParts: [MLXArray] = []
                for m in 0..<M {
                    let xm = x[m ..< m+1]   // [1, K]
                    let rowInds = MLXArray(Array(indsFlat[m*Ktop ..< (m+1)*Ktop]), [Ktop])
                    MLX.eval([xm, rowInds])
                    guard let r = RawMetalForward.gatherQmm(xm, wq, scales: sc, biases: bi,
                                                             inds: rowInds, Ktop: Ktop, K: K, N: N)
                    else { return (false, "ref gatherQmm nil M=\(M) m=\(m)") }
                    r.eval(); refParts.append(r)   // [Ktop, N]
                }
                let ref = MLX.concatenated(refParts, axis: 0); ref.eval()   // [M*Ktop, N]
                // Stub: inds[M*Ktop] row-major, x[M,K]
                guard let got = RawMetalForward.gatherQmmRows(x, wq, scales: sc, biases: bi,
                                                               inds: inds,
                                                               M: M, Ktop: Ktop, K: K, N: N)
                else { return (false, "not implemented (M=\(M))") }
                got.eval()
                let (ok, d) = bitEqual(got, ref)
                if !ok { return (false, "M=\(M): \(d)") }
            }
            return (true, "ok")
        }

        // Test 3: sdpa_rows_bitexact
        // Reference: per-row sdpaDecode with causal prefix lengths L0, L0+1, ..., L0+M-1.
        // Row m attends to the first L0+m keys (strictly causal ordering for batched verify).
        // Note: sdpaDecode requires D==256 (kernel constant); using D=256 here.
        run("sdpa_rows_bitexact") {
            let H = 16, KV = 2, D = 256, L0 = 64
            let scale = Float(pow(Double(D), -0.5))
            for M in [1, 2, 9, 17, 25] {
                let totalSeq = L0 + M - 1   // max sequence length needed
                let q     = MLXRandom.normal([M * H, D]).asType(.float16)
                let kFull = MLXRandom.normal([KV, totalSeq, D]).asType(.float16)
                let vFull = MLXRandom.normal([KV, totalSeq, D]).asType(.float16)
                MLX.eval([q, kFull, vFull])
                // Reference: M calls to sdpaDecode, row m uses S=L0+m
                var refParts: [MLXArray] = []
                for m in 0..<M {
                    let S  = L0 + m
                    let qm = q[m*H ..< (m+1)*H]     // [H, D]
                    // Slice first S tokens from the KV cache: [:, 0:S, :]
                    let km = (S == totalSeq) ? kFull : kFull[0..., 0..<S]  // [KV, S, D]
                    let vm = (S == totalSeq) ? vFull : vFull[0..., 0..<S]
                    MLX.eval([qm, km, vm])
                    guard let r = RawMetalForward.sdpaDecode(qm, km, vm,
                                                              H: H, KV: KV, D: D, S: S, scale: scale)
                    else { return (false, "ref sdpaDecode nil M=\(M) m=\(m) S=\(S)") }
                    r.eval(); refParts.append(r)   // [H, D]
                }
                let ref = MLX.concatenated(refParts, axis: 0); ref.eval()   // [M*H, D]
                // Stub: q[M*H,D], kFull[KV,totalSeq,D], vFull[KV,totalSeq,D]
                guard let got = RawMetalForward.sdpaRows(q, kFull, vFull,
                                                          H: H, KV: KV, D: D,
                                                          baseLen: L0, M: M, scale: scale)
                else { return (false, "not implemented (M=\(M))") }
                got.eval()
                let (ok, d) = bitEqual(got, ref)
                if !ok { return (false, "M=\(M): \(d)") }
            }
            return (true, "ok")
        }

        // Test 4: conv1d_rows_bitexact
        // Reference: per-window conv1dSilu with a K-frame sliding window, output [1,1,C] → [1,C].
        // Stub input: windows[M,K,C]; stub output: [M,C].
        run("conv1d_rows_bitexact") {
            let K = 4, C = 8192
            let w = MLXRandom.normal([C, K]).asType(.float16); w.eval()
            for M in [1, 2, 9, 17, 25] {
                // Sliding-window buffer: buf[M+K-1, C]; window m = buf[m:m+K, :]
                let buf = MLXRandom.normal([M + K - 1, C]).asType(.float16); buf.eval()
                // Reference: M sequential conv1dSilu calls
                var refParts: [MLXArray] = []
                for m in 0..<M {
                    let window = buf[m ..< m+K]   // [K, C]
                    window.eval()
                    guard let r = RawMetalForward.conv1dSilu(window, w, K: K, C: C)
                    else { return (false, "ref conv1dSilu nil M=\(M) m=\(m)") }
                    r.eval(); refParts.append(r.reshaped([1, C]))   // normalise [1,1,C]→[1,C]
                }
                let ref = MLX.concatenated(refParts, axis: 0); ref.eval()   // [M, C]
                // Build windows[M, K, C] by stacking individual windows
                let windowsArr = (0..<M).map { buf[$0 ..< $0+K].reshaped([1, K, C]) }
                let windows = MLX.concatenated(windowsArr, axis: 0); windows.eval()   // [M, K, C]
                // Stub
                guard let got = RawMetalForward.conv1dSiluRows(windows, w, M: M, K: K, C: C)
                else { return (false, "not implemented (M=\(M))") }
                got.eval()
                let (ok, d) = bitEqual(got, ref)
                if !ok { return (false, "M=\(M): \(d)") }
            }
            return (true, "ok")
        }

        // Test 5: gdn_step_rows_bitexact
        // Reference: M sequential T=1 recurrent calls with explicit state threading.
        // Checks BOTH per-position outputs (y) and the final state for bit-equality.
        // Shapes: B=1, Hk=16, Dk=128, Hv=32, Dv=128 (matching GDN layer config).
        run("gdn_step_rows_bitexact") {
            let B = 1, Hk = 16, Dk = 128, Hv = 32, Dv = 128
            let Mmax = 25
            // Draw at Mmax once; tests use prefix slices → stable sub-arrays
            let q    = MLXRandom.normal([B, Mmax, Hk, Dk]).asType(.float16)
            let k    = MLXRandom.normal([B, Mmax, Hk, Dk]).asType(.float16)
            let v    = MLXRandom.normal([B, Mmax, Hv, Dv]).asType(.float16)
            let aRaw = MLXRandom.normal([B, Mmax, Hv]).asType(.float16)   // 'a' for computeG
            let bRaw = MLXRandom.normal([B, Mmax, Hv]).asType(.float16)   // 'b' for sigmoid
            let aLog = MLXRandom.normal([Hv]).asType(.float32)
            let dtB  = MLXRandom.normal([Hv]).asType(.float32)
            let initState = MLXRandom.normal([B, Hv, Dv, Dk]).asType(.float32)
            MLX.eval([q, k, v, aRaw, bRaw, aLog, dtB, initState])
            // Pre-compute g and beta for all Mmax positions (reuse across M variants)
            let betaAll = MLX.sigmoid(bRaw)                              // [B,Mmax,Hv] f16
            let gAll    = GatedDelta.computeG(aLog, aRaw, dtB)           // [B,Mmax,Hv] f32
            MLX.eval([betaAll, gAll])

            for M in [1, 2, 9, 17, 25] {
                // Slice to M positions
                let qM    = q[0..., 0..<M]
                let kM    = k[0..., 0..<M]
                let vM    = v[0..., 0..<M]
                let gM    = gAll[0..., 0..<M]
                let betaM = betaAll[0..., 0..<M]
                MLX.eval([qM, kM, vM, gM, betaM])
                // Reference: M sequential T=1 recurrent calls threading state
                var state = initState
                var refOutputs: [MLXArray] = []
                var refOK = true; var refErr = ""
                for m in 0..<M {
                    let qm    = qM[0..., m ..< m+1]      // [B,1,Hk,Dk]
                    let km    = kM[0..., m ..< m+1]
                    let vm    = vM[0..., m ..< m+1]       // [B,1,Hv,Dv]
                    let gm    = gM[0..., m ..< m+1]       // [B,1,Hv] f32
                    let betam = betaM[0..., m ..< m+1]    // [B,1,Hv] f16
                    MLX.eval([qm, km, vm, gm, betam])
                    guard let (ym, ns) = RawMetalForward.recurrent(
                        qm, km, vm, g: gm, beta: betam, state: state,
                        B: B, T: 1, Hk: Hk, Dk: Dk, Hv: Hv, Dv: Dv)
                    else { refOK = false; refErr = "ref recurrent nil M=\(M) m=\(m)"; break }
                    ym.eval(); ns.eval()
                    refOutputs.append(ym)   // [B,1,Hv,Dv]
                    state = ns
                }
                if !refOK { return (false, refErr) }
                let refY     = MLX.concatenated(refOutputs, axis: 1); refY.eval()   // [B,M,Hv,Dv]
                let refState = state; refState.eval()
                // Stub: takes all M positions at once, initial state → (y[B,M,Hv,Dv], finalState)
                guard let (gotY, gotState) = RawMetalForward.gatedDeltaStepRows(
                    qM, kM, vM, g: gM, beta: betaM, state: initState,
                    M: M, B: B, Hk: Hk, Dk: Dk, Hv: Hv, Dv: Dv)
                else { return (false, "not implemented (M=\(M))") }
                gotY.eval(); gotState.eval()
                let (okY, dY) = bitEqual(gotY, refY)
                if !okY { return (false, "M=\(M) y: \(dY)") }
                let (okS, dS) = bitEqual(gotState, refState)
                if !okS { return (false, "M=\(M) state: \(dS)") }
            }
            return (true, "ok")
        }

        // Test 6: rope_rows_bitexact
        // Reference: per-position rope with offset = startOffset+m applied to numHeads rows,
        //            results concatenated → [M*numHeads, HD].
        run("rope_rows_bitexact") {
            let HD = 128, rd = 64, numHeads = 16
            let base: Float = 1e7, startOffset = 37
            for M in [1, 2, 9, 17, 25] {
                let x = MLXRandom.normal([M * numHeads, HD]).asType(.float16); x.eval()
                // Reference: M rope calls, position m uses offset startOffset+m
                var refParts: [MLXArray] = []
                for m in 0..<M {
                    let xm = x[m*numHeads ..< (m+1)*numHeads]   // [numHeads, HD]
                    xm.eval()
                    guard let r = RawMetalForward.rope(xm, headDim: HD, ropeDim: rd,
                                                        base: base, offset: startOffset + m)
                    else { return (false, "ref rope nil M=\(M) m=\(m)") }
                    r.eval(); refParts.append(r)   // [numHeads, HD]
                }
                let ref = MLX.concatenated(refParts, axis: 0); ref.eval()   // [M*numHeads, HD]
                // Stub: x[M*numHeads, HD], groups of numHeads share position startOffset+m
                guard let got = RawMetalForward.ropeRows(x, headDim: HD, ropeDim: rd,
                                                          base: base,
                                                          startOffset: startOffset,
                                                          M: M, numHeads: numHeads)
                else { return (false, "not implemented (M=\(M))") }
                got.eval()
                let (ok, d) = bitEqual(got, ref)
                if !ok { return (false, "M=\(M): \(d)") }
            }
            return (true, "ok")
        }

        // Test 7: rmsnorm_rows_bitexact
        // Reference: per-row rmsNorm (M=1 calls), concatenated → [M, D].
        // Exercises existing rmsNorm row-semantics explicitly via the loop.
        run("rmsnorm_rows_bitexact") {
            let D = 2048
            let wt = MLXRandom.normal([D]).asType(.float16); wt.eval()
            for M in [1, 2, 9, 17, 25] {
                let x = MLXRandom.normal([M, D]).asType(.float16); x.eval()
                // Reference: M individual rmsNorm calls, each on x[m:m+1]
                var refParts: [MLXArray] = []
                for m in 0..<M {
                    let xm = x[m ..< m+1]   // [1, D]
                    xm.eval()
                    guard let r = RawMetalForward.rmsNorm(xm, wt, eps: 1e-6, D: D)
                    else { return (false, "ref rmsNorm nil M=\(M) m=\(m)") }
                    r.eval(); refParts.append(r)   // [1, D]
                }
                let ref = MLX.concatenated(refParts, axis: 0); ref.eval()   // [M, D]
                // Stub: x[M, D] → y[M, D]
                guard let got = RawMetalForward.rmsNormRows(x, wt, M: M, eps: 1e-6, D: D)
                else { return (false, "not implemented (M=\(M))") }
                got.eval()
                let (ok, d) = bitEqual(got.reshaped(ref.shape), ref)
                if !ok { return (false, "M=\(M): \(d)") }
            }
            return (true, "ok")
        }

        // ── Summary ───────────────────────────────────────────────────────
        return lines.joined(separator: "\n") + "\nRAWTESTS \(passed)/\(total)"
    }
}
