import Foundation
import MLX
import MLXFast
import MLXRandom
import Metal

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
        let total = 46

        // Nested runner: records result and increments counter
        func run(_ name: String, body: () -> (Bool, String)) {
            let (ok, detail) = body()
            lines.append(ok
                ? "[raw-test] \(name): PASS"
                : "[raw-test] \(name): FAIL(\(detail))")
            if ok { passed += 1 }
        }

        // Production Metal scale_mul kernel driver (copy-based, no alias with MLX memory).
        // References for test 40 steps ⑬⑭ MUST use this instead of MLX f32 arithmetic:
        // the Metal kernel computes x[i] = (half)s * x[i] (half·half mul), which can
        // differ from the f32-multiply path by 1 f16 ULP on ~1/2048 elements when `s`
        // is not exactly representable in f16 (e.g. invScale = 1/sqrt(headKDim)).
        func scaleMulKernel(_ input: MLXArray, s: Float, total: Int) -> MLXArray? {
            guard let (device, queue) = RawMetalForward.ensure(),
                  RawMetalForward.ensureAuxPipelines() else { return nil }
            let f16 = input.reshaped([-1]).asType(.float16); f16.eval()
            let arr = f16.asArray(Float16.self)
            guard let buf = arr.withUnsafeBytes({ ptr in
                device.makeBuffer(bytes: ptr.baseAddress!, length: total * 2, options: .storageModeShared)
            }) else { return nil }
            let cb = queue.makeCommandBuffer()!
            let enc = cb.makeComputeCommandEncoder()!
            RawFusedVerify.encodeScaleMul(enc, x: buf, s: s, total: total)
            enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
            let ptr = buf.contents().bindMemory(to: Float16.self, capacity: total)
            return MLXArray(Array(UnsafeBufferPointer(start: ptr, count: total)), input.shape)
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

        // Test 10: qmm_tiled_m_invariant
        // Property: for a fixed weight and input row, qmmTiled's per-row output is
        // bit-identical regardless of how many other rows are batched in the same call (M-independence).
        // Kernel proof: wdq[] is dequanted once before the M-loop; per-row k-accumulation
        // uses stride=(lid→K, step tgs=256) then a 256-slot binary reduction tree —
        // order depends only on K and tgs, not on M.
        // Also checks: full M-row output == concat of M individual qmmTiled(M=1) calls.
        // NOTE: we do NOT compare tiled against qmv/MLX — they legitimately differ
        // (different reduction order); M-invariance of the tiled kernel itself is what we test.
        run("qmm_tiled_m_invariant") {
            let K = 2048, N = 2048, Mmax = 25
            let wf = MLXRandom.normal([N, K]).asType(.float16)
            let (wq, sc, biOpt) = MLX.quantized(wf, groupSize: 64, bits: 4, mode: .affine)
            guard let bi = biOpt else { return (false, "biases nil") }
            let xAll = MLXRandom.normal([Mmax, K]).asType(.float16)
            MLX.eval([wq, sc, bi, xAll])

            let Ms = [1, 2, 9, 17, 25]

            // Step 1: collect row-0 output for each M; all must be bit-identical.
            var row0Ref: MLXArray? = nil
            for M in Ms {
                let xM = xAll[0..<M]   // [M, K]
                guard let out = RawMetalForward.qmmTiled(xM, wq, scales: sc, biases: bi, M: M, K: K, N: N)
                else { return (false, "qmmTiled returned nil M=\(M)") }
                out.eval()
                let row0 = out[0..<1]   // [1, N]
                row0.eval()
                if let ref0 = row0Ref {
                    let (ok, d) = bitEqual(row0, ref0)
                    if !ok { return (false, "M-independence FAIL at M=\(M): \(d)") }
                } else {
                    row0Ref = row0
                }
            }

            // Step 2: for each M, full output must equal concat of M individual M=1 calls.
            for M in Ms {
                let xM = xAll[0..<M]
                guard let outM = RawMetalForward.qmmTiled(xM, wq, scales: sc, biases: bi, M: M, K: K, N: N)
                else { return (false, "qmmTiled(M=\(M)) nil in concat-check") }
                outM.eval()
                var refParts: [MLXArray] = []
                for m in 0..<M {
                    let x1 = xAll[m..<m+1]   // [1, K]
                    guard let r = RawMetalForward.qmmTiled(x1, wq, scales: sc, biases: bi, M: 1, K: K, N: N)
                    else { return (false, "qmmTiled(M=1) nil M=\(M) m=\(m)") }
                    r.eval(); refParts.append(r)
                }
                let ref = MLX.concatenated(refParts, axis: 0); ref.eval()
                let (ok, d) = bitEqual(outM, ref)
                if !ok { return (false, "row-loop FAIL M=\(M): \(d)") }
            }
            return (true, "ok")
        }

        // Test 11: qmm_tiled_selfconsistent
        // Two identical qmmTiled(M=9) calls must produce bit-identical outputs
        // (determinism under fixed safe-math compilation and fixed kernel reduction order).
        run("qmm_tiled_selfconsistent") {
            let K = 2048, N = 2048, M = 9
            let wf = MLXRandom.normal([N, K]).asType(.float16)
            let (wq, sc, biOpt) = MLX.quantized(wf, groupSize: 64, bits: 4, mode: .affine)
            guard let bi = biOpt else { return (false, "biases nil") }
            let x = MLXRandom.normal([M, K]).asType(.float16)
            MLX.eval([wq, sc, bi, x])
            guard let a = RawMetalForward.qmmTiled(x, wq, scales: sc, biases: bi, M: M, K: K, N: N),
                  let b = RawMetalForward.qmmTiled(x, wq, scales: sc, biases: bi, M: M, K: K, N: N)
            else { return (false, "qmmTiled returned nil") }
            MLX.eval([a, b])
            return bitEqual(a, b)
        }


        // Test 12 (U1a): attention 層 × M 行 — rows ≡ M=1 ループ(出力 + KV cache 終状態とも bit 一致)
        run("attn_layer_rows_bitexact") {
            let H = 2048, numHeads = 16, numKV = 2, headDim = 256, baseLen = 16
            func quant(_ n: Int, _ k: Int) -> (MLXArray, MLXArray, MLXArray) {
                let wf = MLXRandom.normal([n, k]).asType(.float16)
                let (q, s, b) = MLX.quantized(wf, groupSize: 64, bits: 4, mode: .affine)
                return (q, s, b!)
            }
            let (qW, qS, qB) = quant(numHeads * 2 * headDim, H)
            let (kW, kS, kB) = quant(numKV * headDim, H)
            let (vW, vS, vB) = quant(numKV * headDim, H)
            let (oW, oS, oB) = quant(H, numHeads * headDim)
            let qN = MLXRandom.normal([headDim]).asType(.float16)
            let kN = MLXRandom.normal([headDim]).asType(.float16)
            let w = RawVerifyForward.AttnLayerW(qWq: qW, qSc: qS, qBi: qB, kWq: kW, kSc: kS, kBi: kB,
                                                vWq: vW, vSc: vS, vBi: vB, oWq: oW, oSc: oS, oBi: oB,
                                                qNorm: qN, kNorm: kN)
            let kC0 = MLXRandom.normal([numKV, baseLen, headDim]).asType(.float16)
            let vC0 = MLXRandom.normal([numKV, baseLen, headDim]).asType(.float16)
            MLX.eval([kC0, vC0])
            for M in [1, 2, 9, 17] {
                let x = MLXRandom.normal([M, H]).asType(.float16); x.eval()
                var k1 = kC0, v1 = vC0
                guard let got = RawVerifyForward.attnLayerRows(x, w, kCache: &k1, vCache: &v1, M: M)
                else { return (false, "rows nil M=\(M)") }
                got.eval()
                var k2 = kC0, v2 = vC0
                var refParts: [MLXArray] = []
                for m in 0..<M {
                    guard let r = RawVerifyForward.attnLayerRows(x[m ..< m+1], w, kCache: &k2, vCache: &v2, M: 1)
                    else { return (false, "ref nil M=\(M) m=\(m)") }
                    r.eval(); refParts.append(r)
                }
                let ref = MLX.concatenated(refParts, axis: 0); ref.eval()
                let (ok1, d1) = bitEqual(got, ref)
                if !ok1 { return (false, "out M=\(M): \(d1)") }
                let (ok2, d2) = bitEqual(k1, k2)
                if !ok2 { return (false, "kCache M=\(M): \(d2)") }
                let (ok3, d3) = bitEqual(v1, v2)
                if !ok3 { return (false, "vCache M=\(M): \(d3)") }
            }
            return (true, "ok")
        }


        // Test 13 (U1b): GDN 層 × M 行 — rows ≡ M=1 ループ(出力 + conv/rec state とも bit 一致)
        run("gdn_layer_rows_bitexact") {
            let H = 2048, Hk = 16, Dk = 128, Hv = 32, Dv = 128, cK = 4
            let convDim = Hk * Dk * 2 + Hv * Dv
            func quant(_ n: Int, _ k: Int) -> (MLXArray, MLXArray, MLXArray) {
                let wf = MLXRandom.normal([n, k]).asType(.float16)
                let (q, s, b) = MLX.quantized(wf, groupSize: 64, bits: 4, mode: .affine)
                return (q, s, b!)
            }
            let (qkvW, qkvS, qkvB) = quant(convDim, H)
            let (zW, zS, zB) = quant(Hv * Dv, H)
            let (bW, bS, bB) = quant(Hv, H)
            let (aW, aS, aB) = quant(Hv, H)
            let (oW, oS, oB) = quant(H, Hv * Dv)
            let convW = MLXRandom.normal([convDim, cK]).asType(.float16)
            let normW = MLXRandom.normal([Dv]).asType(.float16)
            let aLog = MLXRandom.normal([Hv]).asType(.float32)
            let dtB = MLXRandom.normal([Hv]).asType(.float32)
            let w = RawVerifyForward.GDNLayerW(qkvWq: qkvW, qkvSc: qkvS, qkvBi: qkvB,
                                               zWq: zW, zSc: zS, zBi: zB, bWq: bW, bSc: bS, bBi: bB,
                                               aWq: aW, aSc: aS, aBi: aB, outWq: oW, outSc: oS, outBi: oB,
                                               conv1dW: convW, normWeight: normW, aLog: aLog, dtBias: dtB)
            let cs0 = MLXRandom.normal([cK - 1, convDim]).asType(.float16)
            let rs0 = MLXRandom.normal([1, Hv, Dv, Dk]).asType(.float32)
            MLX.eval([cs0, rs0, convW, normW, aLog, dtB])
            for M in [1, 2, 9, 17] {
                let x = MLXRandom.normal([M, H]).asType(.float16); x.eval()
                var c1 = cs0, r1 = rs0
                guard let got = RawVerifyForward.gdnLayerRows(x, w, convState: &c1, recState: &r1, M: M)
                else { return (false, "rows nil M=\(M)") }
                got.eval()
                var c2 = cs0, r2 = rs0
                var refParts: [MLXArray] = []
                for m in 0..<M {
                    guard let r = RawVerifyForward.gdnLayerRows(x[m ..< m+1], w, convState: &c2, recState: &r2, M: 1)
                    else { return (false, "ref nil M=\(M) m=\(m)") }
                    r.eval(); refParts.append(r)
                }
                let ref = MLX.concatenated(refParts, axis: 0); ref.eval()
                let (ok1, d1) = bitEqual(got, ref)
                if !ok1 { return (false, "out M=\(M): \(d1)") }
                let (ok2, d2) = bitEqual(c1, c2)
                if !ok2 { return (false, "convState M=\(M): \(d2)") }
                let (ok3, d3) = bitEqual(r1, r2)
                if !ok3 { return (false, "recState M=\(M): \(d3)") }
            }
            return (true, "ok")
        }


        // Test 14 (U1c): MoE block × M 行 — rows ≡ M=1 ループ(routing の M 形状安定性も含めて検証)
        run("moe_block_rows_bitexact") {
            let H = 2048, E = 16, I = 512, Ktop = 8
            func quant(_ n: Int, _ k: Int, bits: Int = 4) -> (MLXArray, MLXArray, MLXArray) {
                let wf = MLXRandom.normal([n, k]).asType(.float16)
                let (q, s, b) = MLX.quantized(wf, groupSize: 64, bits: bits, mode: .affine)
                return (q, s, b!)
            }
            func quantE(_ e: Int, _ n: Int, _ k: Int) -> (MLXArray, MLXArray, MLXArray) {
                let wf = MLXRandom.normal([e, n, k]).asType(.float16)
                let (q, s, b) = MLX.quantized(wf, groupSize: 64, bits: 4, mode: .affine)
                return (q, s, b!)
            }
            let (gW, gS, gB) = quant(E, H, bits: 8)
            let (sgW, sgS, sgB) = quant(8, H, bits: 8)
            let (swG0, swG1, swG2) = quantE(E, I, H)
            let (swU0, swU1, swU2) = quantE(E, I, H)
            let (swD0, swD1, swD2) = quantE(E, H, I)
            let (shG0, shG1, shG2) = quant(I, H)
            let (shU0, shU1, shU2) = quant(I, H)
            let (shD0, shD1, shD2) = quant(H, I)
            let w = RawVerifyForward.MoEBlockW(gateWq: gW, gateSc: gS, gateBi: gB,
                swGWq: swG0, swGSc: swG1, swGBi: swG2, swUWq: swU0, swUSc: swU1, swUBi: swU2,
                swDWq: swD0, swDSc: swD1, swDBi: swD2, shGWq: shG0, shGSc: shG1, shGBi: shG2,
                shUWq: shU0, shUSc: shU1, shUBi: shU2, shDWq: shD0, shDSc: shD1, shDBi: shD2,
                sharedGateWq: sgW, sharedGateSc: sgS, sharedGateBi: sgB)
            for M in [1, 2, 9, 17] {
                let x = MLXRandom.normal([M, H]).asType(.float16); x.eval()
                guard let got = RawVerifyForward.moeBlockRows(x, w, M: M, E: E, I: I, Ktop: Ktop)
                else { return (false, "rows nil M=\(M)") }
                got.eval()
                var refParts: [MLXArray] = []
                for m in 0..<M {
                    guard let r = RawVerifyForward.moeBlockRows(x[m ..< m+1], w, M: 1, E: E, I: I, Ktop: Ktop)
                    else { return (false, "ref nil M=\(M) m=\(m)") }
                    r.eval(); refParts.append(r)
                }
                let ref = MLX.concatenated(refParts, axis: 0); ref.eval()
                let (ok1, d1) = bitEqual(got, ref)
                if !ok1 { return (false, "M=\(M): \(d1)") }
            }
            return (true, "ok")
        }


        // Test 15 (U1d): 2 層(GDN+attn)合成 verifyForwardRows — rows ≡ M=1 ループ(hidden+全cache bit一致)
        run("verify_forward_rows_bitexact") {
            let H = 2048, E = 16, I = 512
            func q4(_ n: Int, _ k: Int) -> (MLXArray, MLXArray, MLXArray) {
                let wf = MLXRandom.normal([n, k]).asType(.float16)
                let (q, s, b) = MLX.quantized(wf, groupSize: 64, bits: 4, mode: .affine); return (q, s, b!)
            }
            func q8(_ n: Int, _ k: Int) -> (MLXArray, MLXArray, MLXArray) {
                let wf = MLXRandom.normal([n, k]).asType(.float16)
                let (q, s, b) = MLX.quantized(wf, groupSize: 64, bits: 8, mode: .affine); return (q, s, b!)
            }
            func q4e(_ e: Int, _ n: Int, _ k: Int) -> (MLXArray, MLXArray, MLXArray) {
                let wf = MLXRandom.normal([e, n, k]).asType(.float16)
                let (q, s, b) = MLX.quantized(wf, groupSize: 64, bits: 4, mode: .affine); return (q, s, b!)
            }
            func mkMoE() -> RawVerifyForward.MoEBlockW {
                let (gW, gS, gB) = q8(E, H); let (sgW, sgS, sgB) = q8(8, H)
                let (a0, a1, a2) = q4e(E, I, H); let (b0, b1, b2) = q4e(E, I, H); let (c0, c1, c2) = q4e(E, H, I)
                let (d0, d1, d2) = q4(I, H); let (e0, e1, e2) = q4(I, H); let (f0, f1, f2) = q4(H, I)
                return RawVerifyForward.MoEBlockW(gateWq: gW, gateSc: gS, gateBi: gB,
                    swGWq: a0, swGSc: a1, swGBi: a2, swUWq: b0, swUSc: b1, swUBi: b2,
                    swDWq: c0, swDSc: c1, swDBi: c2, shGWq: d0, shGSc: d1, shGBi: d2,
                    shUWq: e0, shUSc: e1, shUBi: e2, shDWq: f0, shDSc: f1, shDBi: f2,
                    sharedGateWq: sgW, sharedGateSc: sgS, sharedGateBi: sgB)
            }
            // layer 0: GDN
            let Hk = 16, Dk = 128, Hv = 32, Dv = 128, cK = 4
            let convDim = Hk * Dk * 2 + Hv * Dv
            let (qkvW, qkvS, qkvB) = q4(convDim, H); let (zW, zS, zB) = q4(Hv * Dv, H)
            let (bW, bS, bB) = q4(Hv, H); let (aW, aS, aB) = q4(Hv, H); let (oW, oS, oB) = q4(H, Hv * Dv)
            let gdnW = RawVerifyForward.GDNLayerW(qkvWq: qkvW, qkvSc: qkvS, qkvBi: qkvB,
                zWq: zW, zSc: zS, zBi: zB, bWq: bW, bSc: bS, bBi: bB, aWq: aW, aSc: aS, aBi: aB,
                outWq: oW, outSc: oS, outBi: oB,
                conv1dW: MLXRandom.normal([convDim, cK]).asType(.float16),
                normWeight: MLXRandom.normal([Dv]).asType(.float16),
                aLog: MLXRandom.normal([Hv]).asType(.float32), dtBias: MLXRandom.normal([Hv]).asType(.float32))
            // layer 1: attn
            let nH = 16, nKV = 2, hD = 256
            let (aqW, aqS, aqB) = q4(nH * 2 * hD, H); let (akW, akS, akB) = q4(nKV * hD, H)
            let (avW, avS, avB) = q4(nKV * hD, H); let (aoW, aoS, aoB) = q4(H, nH * hD)
            let attnW = RawVerifyForward.AttnLayerW(qWq: aqW, qSc: aqS, qBi: aqB, kWq: akW, kSc: akS, kBi: akB,
                vWq: avW, vSc: avS, vBi: avB, oWq: aoW, oSc: aoS, oBi: aoB,
                qNorm: MLXRandom.normal([hD]).asType(.float16), kNorm: MLXRandom.normal([hD]).asType(.float16))
            let layers = [
                RawVerifyForward.LayerSpec(isLinear: true,
                    inputLN: MLXRandom.normal([H]).asType(.float16), postLN: MLXRandom.normal([H]).asType(.float16),
                    gdn: gdnW, attn: nil, moe: mkMoE(), moeE: E, moeI: I),
                RawVerifyForward.LayerSpec(isLinear: false,
                    inputLN: MLXRandom.normal([H]).asType(.float16), postLN: MLXRandom.normal([H]).asType(.float16),
                    gdn: nil, attn: attnW, moe: mkMoE(), moeE: E, moeI: I),
            ]
            let cs0 = MLXRandom.normal([cK - 1, convDim]).asType(.float16)
            let rs0 = MLXRandom.normal([1, Hv, Dv, Dk]).asType(.float32)
            let kC0 = MLXRandom.normal([nKV, 16, hD]).asType(.float16)
            let vC0 = MLXRandom.normal([nKV, 16, hD]).asType(.float16)
            MLX.eval([cs0, rs0, kC0, vC0])
            func freshCaches() -> [RawVerifyForward.LayerCaches] {
                [RawVerifyForward.LayerCaches(convState: cs0, recState: rs0),
                 RawVerifyForward.LayerCaches(kCache: kC0, vCache: vC0)]
            }
            for M in [1, 2, 9, 17] {
                let x = MLXRandom.normal([M, H]).asType(.float16); x.eval()
                let c1 = freshCaches()
                guard let got = RawVerifyForward.verifyForwardRows(x, layers: layers, caches: c1, M: M)
                else { return (false, "rows nil M=\(M)") }
                let c2 = freshCaches()
                var refParts: [MLXArray] = []
                for m in 0..<M {
                    guard let r = RawVerifyForward.verifyForwardRows(x[m ..< m+1], layers: layers, caches: c2, M: 1)
                    else { return (false, "ref nil M=\(M) m=\(m)") }
                    refParts.append(r)
                }
                let ref = MLX.concatenated(refParts, axis: 0); ref.eval()
                let (ok1, d1) = bitEqual(got, ref)
                if !ok1 { return (false, "hidden M=\(M): \(d1)") }
                let pairs: [(MLXArray?, MLXArray?, String)] = [
                    (c1[0].convState, c2[0].convState, "convState"), (c1[0].recState, c2[0].recState, "recState"),
                    (c1[1].kCache, c2[1].kCache, "kCache"), (c1[1].vCache, c2[1].vCache, "vCache")]
                for (a, b, nm) in pairs {
                    guard let aa = a, let bb = b else { return (false, "\(nm) nil M=\(M)") }
                    let (ok, d) = bitEqual(aa, bb)
                    if !ok { return (false, "\(nm) M=\(M): \(d)") }
                }
            }
            return (true, "ok")
        }


        // Test 16 (P2a): 単一CB + 常駐中間 buffer での 2-qmm チェーン ≡ per-op qmmRows 2回(順序保存の礎石)
        run("fused_chain_qmm_bitexact") {
            let K = 2048, N1 = 512, N2 = 2048
            func q4(_ n: Int, _ k: Int) -> (MLXArray, MLXArray, MLXArray) {
                let wf = MLXRandom.normal([n, k]).asType(.float16)
                let (q, s, b) = MLX.quantized(wf, groupSize: 64, bits: 4, mode: .affine); return (q, s, b!)
            }
            let w1 = q4(N1, K), w2 = q4(N2, N1)
            for M in [1, 2, 9, 17, 25] {
                let x = MLXRandom.normal([M, K]).asType(.float16); x.eval()
                // reference: per-op qmmRows 2回(個別CB + readback)
                guard let mid = RawMetalForward.qmmRows(x, w1.0, scales: w1.1, biases: w1.2, M: M, K: K, N: N1),
                      let ref = RawMetalForward.qmmRows(mid, w2.0, scales: w2.1, biases: w2.2, M: M, K: N1, N: N2)
                else { return (false, "ref nil M=\(M)") }
                ref.eval()
                // fused: 単一CB + 常駐 midBuf
                guard let got = RawFusedVerify.fusedTwoQmm(x, w1: w1, N1: N1, w2: w2, N2: N2, M: M, K: K)
                else { return (false, "fused nil M=\(M)") }
                got.eval()
                let (ok, d) = bitEqual(got, ref)
                if !ok { return (false, "M=\(M): \(d)") }
            }
            return (true, "ok")
        }


        // Test 17 (P2b): routeTop8Rows — 各行独立 top-8 が M不変(rows ≡ M=1ループ, inds+scores bit一致)
        run("route_top8_rows_invariant") {
            let N = 256, K = 8
            for M in [1, 2, 9, 17, 25] {
                let logits = MLXRandom.normal([M, N]).asType(.float16); logits.eval()
                guard let (gi, gs) = RawFusedVerify.routeTop8Rows(logits, M: M, N: N, K: K)
                else { return (false, "rows nil M=\(M)") }
                gi.eval(); gs.eval()
                var iParts: [MLXArray] = [], sParts: [MLXArray] = []
                for m in 0..<M {
                    guard let (ri, rs) = RawFusedVerify.routeTop8Rows(logits[m ..< m+1], M: 1, N: N, K: K)
                    else { return (false, "ref nil M=\(M) m=\(m)") }
                    ri.eval(); rs.eval(); iParts.append(ri); sParts.append(rs)
                }
                let refI = MLX.concatenated(iParts, axis: 0); refI.eval()
                let refS = MLX.concatenated(sParts, axis: 0); refS.eval()
                let (oki, di) = bitEqual(gi.asType(.float32), refI.asType(.float32))
                if !oki { return (false, "inds M=\(M): \(di)") }
                let (oks, ds) = bitEqual(gs, refS)
                if !oks { return (false, "scores M=\(M): \(ds)") }
            }
            return (true, "ok")
        }


        // Test 18 (P2c): metalRoute MoE block(argPartition→routeTop8Rows)が M不変 = self-consistent
        run("moe_metalroute_rows_invariant") {
            let H = 2048, E = 16, I = 512, Ktop = 8
            func q4(_ n: Int, _ k: Int) -> (MLXArray, MLXArray, MLXArray) {
                let wf = MLXRandom.normal([n, k]).asType(.float16)
                let (q, s, b) = MLX.quantized(wf, groupSize: 64, bits: 4, mode: .affine); return (q, s, b!)
            }
            func q8(_ n: Int, _ k: Int) -> (MLXArray, MLXArray, MLXArray) {
                let wf = MLXRandom.normal([n, k]).asType(.float16)
                let (q, s, b) = MLX.quantized(wf, groupSize: 64, bits: 8, mode: .affine); return (q, s, b!)
            }
            func q4e(_ e: Int, _ n: Int, _ k: Int) -> (MLXArray, MLXArray, MLXArray) {
                let wf = MLXRandom.normal([e, n, k]).asType(.float16)
                let (q, s, b) = MLX.quantized(wf, groupSize: 64, bits: 4, mode: .affine); return (q, s, b!)
            }
            let (gW, gS, gB) = q8(E, H); let (sgW, sgS, sgB) = q8(8, H)
            let (a0, a1, a2) = q4e(E, I, H); let (b0, b1, b2) = q4e(E, I, H); let (c0, c1, c2) = q4e(E, H, I)
            let (d0, d1, d2) = q4(I, H); let (e0, e1, e2) = q4(I, H); let (f0, f1, f2) = q4(H, I)
            let w = RawVerifyForward.MoEBlockW(gateWq: gW, gateSc: gS, gateBi: gB,
                swGWq: a0, swGSc: a1, swGBi: a2, swUWq: b0, swUSc: b1, swUBi: b2,
                swDWq: c0, swDSc: c1, swDBi: c2, shGWq: d0, shGSc: d1, shGBi: d2,
                shUWq: e0, shUSc: e1, shUBi: e2, shDWq: f0, shDSc: f1, shDBi: f2,
                sharedGateWq: sgW, sharedGateSc: sgS, sharedGateBi: sgB)
            for M in [1, 2, 9, 17] {
                let x = MLXRandom.normal([M, H]).asType(.float16); x.eval()
                guard let got = RawVerifyForward.moeBlockRows(x, w, M: M, E: E, I: I, Ktop: Ktop, metalRoute: true)
                else { return (false, "rows nil M=\(M)") }
                got.eval()
                var refParts: [MLXArray] = []
                for m in 0..<M {
                    guard let r = RawVerifyForward.moeBlockRows(x[m ..< m+1], w, M: 1, E: E, I: I, Ktop: Ktop, metalRoute: true)
                    else { return (false, "ref nil M=\(M) m=\(m)") }
                    r.eval(); refParts.append(r)
                }
                let ref = MLX.concatenated(refParts, axis: 0); ref.eval()
                let (ok, d) = bitEqual(got, ref)
                if !ok { return (false, "M=\(M): \(d)") }
            }
            return (true, "ok")
        }


        // Test 19 (P3): gather の単一CB常駐チェーン(gate→down 形)≡ per-op gatherQmmRows 2回
        run("fused_chain_gather_bitexact") {
            let E = 64, K = 2048, I = 512, K2 = 2048, Ktop = 8
            func q4e(_ e: Int, _ n: Int, _ k: Int) -> (MLXArray, MLXArray, MLXArray) {
                let wf = MLXRandom.normal([e, n, k]).asType(.float16)
                let (q, s, b) = MLX.quantized(wf, groupSize: 64, bits: 4, mode: .affine); return (q, s, b!)
            }
            let w1 = q4e(E, I, K), w2 = q4e(E, K2, I)
            let pool: [Int32] = (0..<200).map { Int32(($0 * 13) % E) }
            for M in [1, 2, 9, 17] {
                let x = MLXRandom.normal([M, K]).asType(.float16)
                let inds = MLXArray((0..<M*Ktop).map { pool[$0 % pool.count] }, [M * Ktop])
                MLX.eval([x, inds])
                // reference: gatherQmmRows 2回(mid = gate lhsPer=false, out = down lhsPer=true)
                guard let mid = RawMetalForward.gatherQmmRows(x, w1.0, scales: w1.1, biases: w1.2,
                                                             inds: inds, M: M, Ktop: Ktop, K: K, N: I),
                      let ref = RawMetalForward.gatherQmmRows(mid, w2.0, scales: w2.1, biases: w2.2,
                                                             inds: inds, M: M, Ktop: Ktop, K: I, N: K2, lhsPerExpert: true)
                else { return (false, "ref nil M=\(M)") }
                ref.eval()
                guard let got = RawFusedVerify.fusedGatherChain(x, inds: inds, w1: w1, I: I, w2: w2, K2: K2,
                                                                M: M, Ktop: Ktop, K: K)
                else { return (false, "fused nil M=\(M)") }
                got.eval()
                let (ok, d) = bitEqual(got, ref)
                if !ok { return (false, "M=\(M): \(d)") }
            }
            return (true, "ok")
        }

        // Test 20 (P3-A): fused MoE block(単一 encoder + 常駐中間, 全段 Metal)≡ composed moeBlockRows(metalRoute)
        run("fused_moe_block_bitexact") {
            let H = 2048, E = 16, I = 512, Ktop = 8
            func q4(_ n: Int, _ k: Int) -> (MLXArray, MLXArray, MLXArray) {
                let wf = MLXRandom.normal([n, k]).asType(.float16)
                let (q, s, b) = MLX.quantized(wf, groupSize: 64, bits: 4, mode: .affine); return (q, s, b!)
            }
            func q8(_ n: Int, _ k: Int) -> (MLXArray, MLXArray, MLXArray) {
                let wf = MLXRandom.normal([n, k]).asType(.float16)
                let (q, s, b) = MLX.quantized(wf, groupSize: 64, bits: 8, mode: .affine); return (q, s, b!)
            }
            func q4e(_ e: Int, _ n: Int, _ k: Int) -> (MLXArray, MLXArray, MLXArray) {
                let wf = MLXRandom.normal([e, n, k]).asType(.float16)
                let (q, s, b) = MLX.quantized(wf, groupSize: 64, bits: 4, mode: .affine); return (q, s, b!)
            }
            let (gW, gS, gB) = q8(E, H); let (sgW, sgS, sgB) = q8(8, H)
            let (a0, a1, a2) = q4e(E, I, H); let (b0, b1, b2) = q4e(E, I, H); let (c0, c1, c2) = q4e(E, H, I)
            let (d0, d1, d2) = q4(I, H); let (e0, e1, e2) = q4(I, H); let (f0, f1, f2) = q4(H, I)
            let w = RawVerifyForward.MoEBlockW(gateWq: gW, gateSc: gS, gateBi: gB,
                swGWq: a0, swGSc: a1, swGBi: a2, swUWq: b0, swUSc: b1, swUBi: b2,
                swDWq: c0, swDSc: c1, swDBi: c2, shGWq: d0, shGSc: d1, shGBi: d2,
                shUWq: e0, shUSc: e1, shUBi: e2, shDWq: f0, shDSc: f1, shDBi: f2,
                sharedGateWq: sgW, sharedGateSc: sgS, sharedGateBi: sgB)
            for M in [1, 2, 9, 17] {
                let x = MLXRandom.normal([M, H]).asType(.float16); x.eval()
                guard let ref = RawVerifyForward.moeBlockRows(x, w, M: M, E: E, I: I, Ktop: Ktop, metalRoute: true)
                else { return (false, "composed nil M=\(M)") }
                ref.eval()
                guard let got = RawFusedVerify.fusedMoEBlockRows(x, w, M: M, E: E, I: I, Ktop: Ktop)
                else { return (false, "fused nil M=\(M)") }
                got.eval()
                let (ok, d) = bitEqual(got, ref)
                if !ok {
                    // 段階バイセクト: composed per-op 中間 vs fused dump で最初の乖離段を特定
                    guard let dump = RawFusedVerify.fusedMoEBlockRowsDump(x, w, M: M, E: E, I: I, Ktop: Ktop),
                          let cgl = RawMetalForward.qmm8(x, w.gateWq, scales: w.gateSc, biases: w.gateBi, M: M, K: H, N: E),
                          let (ci, cs) = RawFusedVerify.routeTop8Rows(cgl, M: M, N: E, K: Ktop)
                    else { return (false, "M=\(M): \(d) (dump nil)") }
                    let cif = ci.reshaped([M * Ktop]).asType(.int32); cif.eval()
                    guard let cg = RawMetalForward.gatherQmmRows(x, w.swGWq, scales: w.swGSc, biases: w.swGBi,
                                                                 inds: cif, M: M, Ktop: Ktop, K: H, N: I),
                          let cu = RawMetalForward.gatherQmmRows(x, w.swUWq, scales: w.swUSc, biases: w.swUBi,
                                                                 inds: cif, M: M, Ktop: Ktop, K: H, N: I),
                          let ch = RawMetalForward.swigluRaw(cg, cu),
                          let cd = RawMetalForward.gatherQmmRows(ch, w.swDWq, scales: w.swDSc, biases: w.swDBi,
                                                                 inds: cif, M: M, Ktop: Ktop, K: I, N: H, lhsPerExpert: true),
                          let csg = RawMetalForward.qmmRows(x, w.shGWq, scales: w.shGSc, biases: w.shGBi, M: M, K: H, N: I),
                          let csu = RawMetalForward.qmmRows(x, w.shUWq, scales: w.shUSc, biases: w.shUBi, M: M, K: H, N: I),
                          let cshAct = RawMetalForward.swigluRaw(csg, csu),
                          let csharedY = RawMetalForward.qmmRows(cshAct, w.shDWq, scales: w.shDSc, biases: w.shDBi, M: M, K: I, N: H),
                          let csgl = RawMetalForward.qmm8(x, w.sharedGateWq, scales: w.sharedGateSc, biases: w.sharedGateBi, M: M, K: H, N: 8)
                    else { return (false, "M=\(M): \(d) (composed stage nil)") }
                    guard let cy = RawFusedVerify.combineRowsRaw(cd, cs, M: M, Ktop: Ktop, N: H)
                    else { return (false, "M=\(M): \(d) (combine nil)") }
                    cy.eval()
                    let stages: [(String, MLXArray)] = [
                        ("gl", cgl), ("inds", cif.asType(.int32)), ("scores", cs), ("g", cg), ("u", cu),
                        ("h", ch), ("d", cd), ("y", cy), ("sg", csg), ("su", csu),
                        ("shAct", cshAct), ("sharedY", csharedY), ("sgl", csgl)]
                    for (nm, cref) in stages {
                        guard let f = dump[nm] else { continue }
                        let (sok, sd) = bitEqual(f.asType(.float32), cref.asType(.float32))
                        if !sok { return (false, "M=\(M) stage=\(nm): \(sd)") }
                    }
                    return (false, "M=\(M) out-only: \(d)")
                }
            }
            return (true, "ok")
        }

        // Test 21 (P3-B): fused attn 層(単一 encoder + 常駐 KV cache)≡ composed attnLayerRows(出力+cache)
        run("fused_attn_layer_bitexact") {
            let H = 2048, nH = 16, nKV = 2, hD = 256
            func q4(_ n: Int, _ k: Int) -> (MLXArray, MLXArray, MLXArray) {
                let wf = MLXRandom.normal([n, k]).asType(.float16)
                let (q, s, b) = MLX.quantized(wf, groupSize: 64, bits: 4, mode: .affine); return (q, s, b!)
            }
            let (aqW, aqS, aqB) = q4(nH * 2 * hD, H); let (akW, akS, akB) = q4(nKV * hD, H)
            let (avW, avS, avB) = q4(nKV * hD, H); let (aoW, aoS, aoB) = q4(H, nH * hD)
            let w = RawVerifyForward.AttnLayerW(qWq: aqW, qSc: aqS, qBi: aqB, kWq: akW, kSc: akS, kBi: akB,
                vWq: avW, vSc: avS, vBi: avB, oWq: aoW, oSc: aoS, oBi: aoB,
                qNorm: MLXRandom.normal([hD]).asType(.float16), kNorm: MLXRandom.normal([hD]).asType(.float16))
            let kC0 = MLXRandom.normal([nKV, 16, hD]).asType(.float16)
            let vC0 = MLXRandom.normal([nKV, 16, hD]).asType(.float16)
            MLX.eval([kC0, vC0])
            for M in [1, 2, 9, 17] {
                let x = MLXRandom.normal([M, H]).asType(.float16); x.eval()
                var kc = kC0, vc = vC0
                guard let ref = RawVerifyForward.attnLayerRows(x, w, kCache: &kc, vCache: &vc, M: M)
                else { return (false, "composed nil M=\(M)") }
                ref.eval()
                guard let (got, gk, gv) = RawFusedVerify.fusedAttnLayerRows(x, w, kInit: kC0, vInit: vC0,
                                                                            maxLen: 64, M: M)
                else { return (false, "fused nil M=\(M)") }
                got.eval(); gk.eval(); gv.eval()
                let (ok1, d1) = bitEqual(got, ref)
                if !ok1 { return (false, "out M=\(M): \(d1)") }
                let (ok2, d2) = bitEqual(gk, kc)
                if !ok2 { return (false, "kCache M=\(M): \(d2)") }
                let (ok3, d3) = bitEqual(gv, vc)
                if !ok3 { return (false, "vCache M=\(M): \(d3)") }
            }
            return (true, "ok")
        }

        // Test 22 (P3-C): fused GDN 層(単一 encoder + 常駐 conv hist/rec state)≡ composed gdnLayerRows
        run("fused_gdn_layer_bitexact") {
            let H = 2048, Hk = 16, Dk = 128, Hv = 32, Dv = 128, cK = 4
            let convDim = Hk * Dk * 2 + Hv * Dv
            func q4(_ n: Int, _ k: Int) -> (MLXArray, MLXArray, MLXArray) {
                let wf = MLXRandom.normal([n, k]).asType(.float16)
                let (q, s, b) = MLX.quantized(wf, groupSize: 64, bits: 4, mode: .affine); return (q, s, b!)
            }
            let (qkvW, qkvS, qkvB) = q4(convDim, H); let (zW, zS, zB) = q4(Hv * Dv, H)
            let (bW, bS, bB) = q4(Hv, H); let (aW, aS, aB) = q4(Hv, H); let (oW, oS, oB) = q4(H, Hv * Dv)
            let w = RawVerifyForward.GDNLayerW(qkvWq: qkvW, qkvSc: qkvS, qkvBi: qkvB,
                zWq: zW, zSc: zS, zBi: zB, bWq: bW, bSc: bS, bBi: bB, aWq: aW, aSc: aS, aBi: aB,
                outWq: oW, outSc: oS, outBi: oB,
                conv1dW: MLXRandom.normal([convDim, cK]).asType(.float16),
                normWeight: MLXRandom.normal([Dv]).asType(.float16),
                aLog: MLXRandom.normal([Hv]).asType(.float32), dtBias: MLXRandom.normal([Hv]).asType(.float32))
            let cs0 = MLXRandom.normal([cK - 1, convDim]).asType(.float16)
            let rs0 = MLXRandom.normal([1, Hv, Dv, Dk]).asType(.float32)
            MLX.eval([cs0, rs0])
            for M in [1, 2, 9, 17] {
                let x = MLXRandom.normal([M, H]).asType(.float16); x.eval()
                var cs = cs0, rs = rs0
                guard let ref = RawVerifyForward.gdnLayerRows(x, w, convState: &cs, recState: &rs, M: M)
                else { return (false, "composed nil M=\(M)") }
                ref.eval()
                guard let (got, gcs, grs) = RawFusedVerify.fusedGdnLayerRows(x, w, convInit: cs0, recInit: rs0, M: M)
                else { return (false, "fused nil M=\(M)") }
                got.eval(); gcs.eval(); grs.eval()
                let (ok1, d1) = bitEqual(got, ref)
                if !ok1 { return (false, "out M=\(M): \(d1)") }
                let (ok2, d2) = bitEqual(gcs, cs)
                if !ok2 { return (false, "convState M=\(M): \(d2)") }
                let (ok3, d3) = bitEqual(grs, rs)
                if !ok3 { return (false, "recState M=\(M): \(d3)") }
            }
            return (true, "ok")
        }

        // Test 23 (P3-D): 全層 1-CB fused forward ≡ composed verifyForwardRows(metalRoute)
        // 2層(GDN+attn)合成、M 掃引 + 2-step チェーン(cache 常駐更新の連続性)を検証。
        run("fused_forward_rows_bitexact") {
            let H = 2048, E = 16, I = 512
            func q4(_ n: Int, _ k: Int) -> (MLXArray, MLXArray, MLXArray) {
                let wf = MLXRandom.normal([n, k]).asType(.float16)
                let (q, s, b) = MLX.quantized(wf, groupSize: 64, bits: 4, mode: .affine); return (q, s, b!)
            }
            func q8(_ n: Int, _ k: Int) -> (MLXArray, MLXArray, MLXArray) {
                let wf = MLXRandom.normal([n, k]).asType(.float16)
                let (q, s, b) = MLX.quantized(wf, groupSize: 64, bits: 8, mode: .affine); return (q, s, b!)
            }
            func q4e(_ e: Int, _ n: Int, _ k: Int) -> (MLXArray, MLXArray, MLXArray) {
                let wf = MLXRandom.normal([e, n, k]).asType(.float16)
                let (q, s, b) = MLX.quantized(wf, groupSize: 64, bits: 4, mode: .affine); return (q, s, b!)
            }
            func mkMoE() -> RawVerifyForward.MoEBlockW {
                let (gW, gS, gB) = q8(E, H); let (sgW, sgS, sgB) = q8(8, H)
                let (a0, a1, a2) = q4e(E, I, H); let (b0, b1, b2) = q4e(E, I, H); let (c0, c1, c2) = q4e(E, H, I)
                let (d0, d1, d2) = q4(I, H); let (e0, e1, e2) = q4(I, H); let (f0, f1, f2) = q4(H, I)
                return RawVerifyForward.MoEBlockW(gateWq: gW, gateSc: gS, gateBi: gB,
                    swGWq: a0, swGSc: a1, swGBi: a2, swUWq: b0, swUSc: b1, swUBi: b2,
                    swDWq: c0, swDSc: c1, swDBi: c2, shGWq: d0, shGSc: d1, shGBi: d2,
                    shUWq: e0, shUSc: e1, shUBi: e2, shDWq: f0, shDSc: f1, shDBi: f2,
                    sharedGateWq: sgW, sharedGateSc: sgS, sharedGateBi: sgB)
            }
            let Hk = 16, Dk = 128, Hv = 32, Dv = 128, cK = 4
            let convDim = Hk * Dk * 2 + Hv * Dv
            let (qkvW, qkvS, qkvB) = q4(convDim, H); let (zW, zS, zB) = q4(Hv * Dv, H)
            let (bW, bS, bB) = q4(Hv, H); let (aW, aS, aB) = q4(Hv, H); let (oW, oS, oB) = q4(H, Hv * Dv)
            let gdnW = RawVerifyForward.GDNLayerW(qkvWq: qkvW, qkvSc: qkvS, qkvBi: qkvB,
                zWq: zW, zSc: zS, zBi: zB, bWq: bW, bSc: bS, bBi: bB, aWq: aW, aSc: aS, aBi: aB,
                outWq: oW, outSc: oS, outBi: oB,
                conv1dW: MLXRandom.normal([convDim, cK]).asType(.float16),
                normWeight: MLXRandom.normal([Dv]).asType(.float16),
                aLog: MLXRandom.normal([Hv]).asType(.float32), dtBias: MLXRandom.normal([Hv]).asType(.float32))
            let nH = 16, nKV = 2, hD = 256
            let (aqW, aqS, aqB) = q4(nH * 2 * hD, H); let (akW, akS, akB) = q4(nKV * hD, H)
            let (avW, avS, avB) = q4(nKV * hD, H); let (aoW, aoS, aoB) = q4(H, nH * hD)
            let attnW = RawVerifyForward.AttnLayerW(qWq: aqW, qSc: aqS, qBi: aqB, kWq: akW, kSc: akS, kBi: akB,
                vWq: avW, vSc: avS, vBi: avB, oWq: aoW, oSc: aoS, oBi: aoB,
                qNorm: MLXRandom.normal([hD]).asType(.float16), kNorm: MLXRandom.normal([hD]).asType(.float16))
            let layers = [
                RawVerifyForward.LayerSpec(isLinear: true,
                    inputLN: MLXRandom.normal([H]).asType(.float16), postLN: MLXRandom.normal([H]).asType(.float16),
                    gdn: gdnW, attn: nil, moe: mkMoE(), moeE: E, moeI: I),
                RawVerifyForward.LayerSpec(isLinear: false,
                    inputLN: MLXRandom.normal([H]).asType(.float16), postLN: MLXRandom.normal([H]).asType(.float16),
                    gdn: nil, attn: attnW, moe: mkMoE(), moeE: E, moeI: I),
            ]
            let cs0 = MLXRandom.normal([cK - 1, convDim]).asType(.float16)
            let rs0 = MLXRandom.normal([1, Hv, Dv, Dk]).asType(.float32)
            let kC0 = MLXRandom.normal([nKV, 16, hD]).asType(.float16)
            let vC0 = MLXRandom.normal([nKV, 16, hD]).asType(.float16)
            MLX.eval([cs0, rs0, kC0, vC0])
            func freshCaches() -> [RawVerifyForward.LayerCaches] {
                [RawVerifyForward.LayerCaches(convState: cs0, recState: rs0),
                 RawVerifyForward.LayerCaches(kCache: kC0, vCache: vC0)]
            }
            // ステップ列: M 掃引 + 最後に 2-step チェーン(9→3)
            let stepPlans: [[Int]] = [[1], [2], [9], [17], [9, 3]]
            for plan in stepPlans {
                let comp = freshCaches()
                guard let fused = RawFusedVerify.RawFusedForward(layers: layers, caches: freshCaches(),
                                                                 maxM: 17, H: H, maxSeqLen: 64)
                else { return (false, "fused init nil plan=\(plan)") }
                for (si, M) in plan.enumerated() {
                    let x = MLXRandom.normal([M, H]).asType(.float16); x.eval()
                    guard let ref = RawVerifyForward.verifyForwardRows(x, layers: layers, caches: comp, M: M, metalRoute: true)
                    else { return (false, "composed nil plan=\(plan) step=\(si)") }
                    ref.eval()
                    guard let got = fused.forwardRows(x, M: M)
                    else { return (false, "fused nil plan=\(plan) step=\(si)") }
                    got.eval()
                    let (ok, d) = bitEqual(got, ref)
                    if !ok { return (false, "h plan=\(plan) step=\(si): \(d)") }
                }
                // cache 突き合わせ(gdn: conv/rec, attn: k/v)
                let (fc, fr) = fused.readLayerCache(0)
                let (fk, fv) = fused.readLayerCache(1)
                let pairs: [(MLXArray?, MLXArray?, String)] = [
                    (fc, comp[0].convState, "convState"), (fr, comp[0].recState, "recState"),
                    (fk, comp[1].kCache, "kCache"), (fv, comp[1].vCache, "vCache")]
                for (a, b, nm) in pairs {
                    guard let aa = a, let bb = b else { return (false, "\(nm) nil plan=\(plan)") }
                    let (ok, d) = bitEqual(aa, bb)
                    if !ok { return (false, "\(nm) plan=\(plan): \(d)") }
                }
            }
            return (true, "ok")
        }

        // Test 24 (P3-E): head kernel 単体 — embed_rows_q4 ≡ MLX dequant-take / argmax_rows ≡ MLX argMax
        run("head_step_kernels_bitexact") {
            let V = 1024, H = 2048
            let wf = MLXRandom.normal([V, H]).asType(.float16)
            let (wq, sc, biOpt) = MLX.quantized(wf, groupSize: 64, bits: 4, mode: .affine)
            guard let bi = biOpt else { return (false, "biases nil") }
            MLX.eval([wq, sc, bi])
            for M in [1, 2, 9] {
                // embed: raw kernel vs MLX(dequantize→take)
                let toks: [Int32] = (0 ..< M).map { Int32(($0 * 37 + 5) % V) }
                let ids = MLXArray(toks)
                let refE = ModelHead.embed(ids: ids, weight: wq, scales: sc, biases: bi, bits: 4)
                    .reshaped([M, H]).asType(.float16)
                refE.eval()
                guard let gotE = RawFusedVerify.embedRowsRaw(toks, w: wq, scales: sc, biases: bi, H: H)
                else { return (false, "embed nil M=\(M)") }
                gotE.eval()
                let (okE, dE) = bitEqual(gotE, refE)
                if !okE { return (false, "embed M=\(M): \(dE)") }
                // argmax: raw kernel vs MLX argMax(重複値で tie-break=先頭一致 も検証)
                var lg = MLXRandom.normal([M, V]).asType(.float16)
                lg[0..., 7] = lg[0..., 3]                       // 意図的 tie
                lg.eval()
                let refA: [Int] = (0 ..< M).map { MLX.argMax(lg[$0], axis: -1).item(Int.self) }
                guard let gotA = RawFusedVerify.argmaxRowsRaw(lg, M: M, V: V)
                else { return (false, "argmax nil M=\(M)") }
                if gotA != refA { return (false, "argmax M=\(M): got=\(gotA) ref=\(refA)") }
            }
            return (true, "ok")
        }

        // ── Streaming tests (25-28) ───────────────────────────────────────

        // Common model geometry for streaming tests (same as test 23).
        let stH = 2048, stE = 16, stI = 512
        func stQ4(_ n: Int, _ k: Int) -> (MLXArray, MLXArray, MLXArray) {
            let wf = MLXRandom.normal([n, k]).asType(.float16)
            let (q, s, b) = MLX.quantized(wf, groupSize: 64, bits: 4, mode: .affine); return (q, s, b!)
        }
        func stQ8(_ n: Int, _ k: Int) -> (MLXArray, MLXArray, MLXArray) {
            let wf = MLXRandom.normal([n, k]).asType(.float16)
            let (q, s, b) = MLX.quantized(wf, groupSize: 64, bits: 8, mode: .affine); return (q, s, b!)
        }
        func stQ4e(_ e: Int, _ n: Int, _ k: Int) -> (MLXArray, MLXArray, MLXArray) {
            let wf = MLXRandom.normal([e, n, k]).asType(.float16)
            let (q, s, b) = MLX.quantized(wf, groupSize: 64, bits: 4, mode: .affine); return (q, s, b!)
        }
        // Returns (MoEBlockW, gW, gSf16, gBf16, uW, uSf16, uBf16, dW, dSf16, dBf16)
        // so callers can share expert arrays between MoEBlockW and TestExpertProvider.
        typealias MoEWithArrays = (w: RawVerifyForward.MoEBlockW,
                                   gW: MLXArray, gSf16: MLXArray, gBf16: MLXArray,
                                   uW: MLXArray, uSf16: MLXArray, uBf16: MLXArray,
                                   dW: MLXArray, dSf16: MLXArray, dBf16: MLXArray)
        func stMkMoE() -> MoEWithArrays {
            let (gW, gS, gB) = stQ8(stE, stH); let (sgW, sgS, sgB) = stQ8(8, stH)
            let (a0, a1, a2) = stQ4e(stE, stI, stH)
            let (b0, b1, b2) = stQ4e(stE, stI, stH)
            let (c0, c1, c2) = stQ4e(stE, stH, stI)
            let (d0, d1, d2) = stQ4(stI, stH); let (e0, e1, e2) = stQ4(stI, stH); let (f0, f1, f2) = stQ4(stH, stI)
            let moeW = RawVerifyForward.MoEBlockW(gateWq: gW, gateSc: gS, gateBi: gB,
                swGWq: a0, swGSc: a1, swGBi: a2, swUWq: b0, swUSc: b1, swUBi: b2,
                swDWq: c0, swDSc: c1, swDBi: c2, shGWq: d0, shGSc: d1, shGBi: d2,
                shUWq: e0, shUSc: e1, shUBi: e2, shDWq: f0, shDSc: f1, shDBi: f2,
                sharedGateWq: sgW, sharedGateSc: sgS, sharedGateBi: sgB)
            return (w: moeW,
                    gW: a0, gSf16: a1.asType(.float16), gBf16: a2.asType(.float16),
                    uW: b0, uSf16: b1.asType(.float16), uBf16: b2.asType(.float16),
                    dW: c0, dSf16: c1.asType(.float16), dBf16: c2.asType(.float16))
        }
        let stHk = 16, stDk = 128, stHv = 32, stDv = 128, stCK = 4
        let stConvDim = stHk * stDk * 2 + stHv * stDv
        func stMkGdnW() -> RawVerifyForward.GDNLayerW {
            let (qkvW, qkvS, qkvB) = stQ4(stConvDim, stH); let (zW, zS, zB) = stQ4(stHv * stDv, stH)
            let (bW, bS, bB) = stQ4(stHv, stH); let (aW, aS, aB) = stQ4(stHv, stH); let (oW, oS, oB) = stQ4(stH, stHv * stDv)
            return RawVerifyForward.GDNLayerW(qkvWq: qkvW, qkvSc: qkvS, qkvBi: qkvB,
                zWq: zW, zSc: zS, zBi: zB, bWq: bW, bSc: bS, bBi: bB, aWq: aW, aSc: aS, aBi: aB,
                outWq: oW, outSc: oS, outBi: oB,
                conv1dW: MLXRandom.normal([stConvDim, stCK]).asType(.float16),
                normWeight: MLXRandom.normal([stDv]).asType(.float16),
                aLog: MLXRandom.normal([stHv]).asType(.float32), dtBias: MLXRandom.normal([stHv]).asType(.float32))
        }
        let stNh = 16, stNkv = 2, stHd = 256
        func stMkAttnW() -> RawVerifyForward.AttnLayerW {
            let (aqW, aqS, aqB) = stQ4(stNh * 2 * stHd, stH); let (akW, akS, akB) = stQ4(stNkv * stHd, stH)
            let (avW, avS, avB) = stQ4(stNkv * stHd, stH); let (aoW, aoS, aoB) = stQ4(stH, stNh * stHd)
            return RawVerifyForward.AttnLayerW(qWq: aqW, qSc: aqS, qBi: aqB, kWq: akW, kSc: akS, kBi: akB,
                vWq: avW, vSc: avS, vBi: avB, oWq: aoW, oSc: aoS, oBi: aoB,
                qNorm: MLXRandom.normal([stHd]).asType(.float16), kNorm: MLXRandom.normal([stHd]).asType(.float16))
        }

        // Test 25 (D1-A): strict streaming ≡ resident — C=8<E=16, multi-plan, chunk assertion.
        run("stream_fused_strict_bitexact") {
            guard let (device, _) = RawMetalForward.ensure() else { return (false, "no device") }
            let C = 8
            let moe0 = stMkMoE(), moe1 = stMkMoE()
            let gdnW = stMkGdnW(), attnW = stMkAttnW()
            let stCs0 = MLXRandom.normal([stCK - 1, stConvDim]).asType(.float16)
            let stRs0 = MLXRandom.normal([1, stHv, stDv, stDk]).asType(.float32)
            let stKc0 = MLXRandom.normal([stNkv, 16, stHd]).asType(.float16)
            let stVc0 = MLXRandom.normal([stNkv, 16, stHd]).asType(.float16)
            MLX.eval([stCs0, stRs0, stKc0, stVc0])
            func freshC() -> [RawVerifyForward.LayerCaches] {
                [RawVerifyForward.LayerCaches(convState: stCs0, recState: stRs0),
                 RawVerifyForward.LayerCaches(kCache: stKc0, vCache: stVc0)]
            }
            let layerSpecs = [
                RawVerifyForward.LayerSpec(isLinear: true,
                    inputLN: MLXRandom.normal([stH]).asType(.float16), postLN: MLXRandom.normal([stH]).asType(.float16),
                    gdn: gdnW, attn: nil, moe: moe0.w, moeE: stE, moeI: stI),
                RawVerifyForward.LayerSpec(isLinear: false,
                    inputLN: MLXRandom.normal([stH]).asType(.float16), postLN: MLXRandom.normal([stH]).asType(.float16),
                    gdn: nil, attn: attnW, moe: moe1.w, moeE: stE, moeI: stI),
            ]
            guard let tp0 = TestExpertProvider(E: stE, I: stI, H: stH, C: C,
                    gW: moe0.gW, gSf16: moe0.gSf16, gBf16: moe0.gBf16,
                    uW: moe0.uW, uSf16: moe0.uSf16, uBf16: moe0.uBf16,
                    dW: moe0.dW, dSf16: moe0.dSf16, dBf16: moe0.dBf16, device: device),
                  let tp1 = TestExpertProvider(E: stE, I: stI, H: stH, C: C,
                    gW: moe1.gW, gSf16: moe1.gSf16, gBf16: moe1.gBf16,
                    uW: moe1.uW, uSf16: moe1.uSf16, uBf16: moe1.uBf16,
                    dW: moe1.dW, dSf16: moe1.dSf16, dBf16: moe1.dBf16, device: device)
            else { return (false, "TestExpertProvider nil") }
            var sawMultiChunk = false
            let plans: [[Int]] = [[1], [2], [9], [9, 3]]
            for plan in plans {
                guard let res = RawFusedVerify.RawFusedForward(
                        layers: layerSpecs, caches: freshC(), maxM: 17, H: stH, maxSeqLen: 64),
                      let str = RawFusedVerify.RawFusedForward(
                        layers: layerSpecs, caches: freshC(), maxM: 17, H: stH, maxSeqLen: 64,
                        providers: [tp0, tp1])
                else { return (false, "init nil plan=\(plan)") }
                for (si, M) in plan.enumerated() {
                    let x = MLXRandom.normal([M, stH]).asType(.float16); x.eval()
                    guard let ref = res.forwardRows(x, M: M),
                          let got = str.forwardRows(x, M: M)
                    else { return (false, "forwardRows nil plan=\(plan) step=\(si)") }
                    ref.eval(); got.eval()
                    let (ok, d) = bitEqual(got, ref)
                    if !ok { return (false, "plan=\(plan) step=\(si): \(d)") }
                    if str.lastStepChunks >= 2 { sawMultiChunk = true }
                }
            }
            if !sawMultiChunk { return (false, "no multi-chunk observed across all plans (C=\(C),E=\(stE))") }
            return (true, "ok")
        }

        // Test 26 (D1-B): strict streaming M=17 — 複数 chunk 確実に発生、bit-exact。
        run("stream_fused_strict_m17_chunking") {
            guard let (device, _) = RawMetalForward.ensure() else { return (false, "no device") }
            let C = 8
            let moe0 = stMkMoE(), moe1 = stMkMoE()
            let gdnW = stMkGdnW(), attnW = stMkAttnW()
            let stCs0 = MLXRandom.normal([stCK - 1, stConvDim]).asType(.float16)
            let stRs0 = MLXRandom.normal([1, stHv, stDv, stDk]).asType(.float32)
            let stKc0 = MLXRandom.normal([stNkv, 16, stHd]).asType(.float16)
            let stVc0 = MLXRandom.normal([stNkv, 16, stHd]).asType(.float16)
            MLX.eval([stCs0, stRs0, stKc0, stVc0])
            let layerSpecs = [
                RawVerifyForward.LayerSpec(isLinear: true,
                    inputLN: MLXRandom.normal([stH]).asType(.float16), postLN: MLXRandom.normal([stH]).asType(.float16),
                    gdn: gdnW, attn: nil, moe: moe0.w, moeE: stE, moeI: stI),
                RawVerifyForward.LayerSpec(isLinear: false,
                    inputLN: MLXRandom.normal([stH]).asType(.float16), postLN: MLXRandom.normal([stH]).asType(.float16),
                    gdn: nil, attn: attnW, moe: moe1.w, moeE: stE, moeI: stI),
            ]
            guard let tp0 = TestExpertProvider(E: stE, I: stI, H: stH, C: C,
                    gW: moe0.gW, gSf16: moe0.gSf16, gBf16: moe0.gBf16,
                    uW: moe0.uW, uSf16: moe0.uSf16, uBf16: moe0.uBf16,
                    dW: moe0.dW, dSf16: moe0.dSf16, dBf16: moe0.dBf16, device: device),
                  let tp1 = TestExpertProvider(E: stE, I: stI, H: stH, C: C,
                    gW: moe1.gW, gSf16: moe1.gSf16, gBf16: moe1.gBf16,
                    uW: moe1.uW, uSf16: moe1.uSf16, uBf16: moe1.uBf16,
                    dW: moe1.dW, dSf16: moe1.dSf16, dBf16: moe1.dBf16, device: device)
            else { return (false, "TestExpertProvider nil") }
            let freshC: [RawVerifyForward.LayerCaches] = [
                RawVerifyForward.LayerCaches(convState: stCs0, recState: stRs0),
                RawVerifyForward.LayerCaches(kCache: stKc0, vCache: stVc0)]
            guard let res = RawFusedVerify.RawFusedForward(
                    layers: layerSpecs, caches: freshC, maxM: 17, H: stH, maxSeqLen: 64),
                  let str = RawFusedVerify.RawFusedForward(
                    layers: layerSpecs, caches: freshC, maxM: 17, H: stH, maxSeqLen: 64,
                    providers: [tp0, tp1])
            else { return (false, "init nil") }
            let M = 17
            let x = MLXRandom.normal([M, stH]).asType(.float16); x.eval()
            guard let ref = res.forwardRows(x, M: M),
                  let got = str.forwardRows(x, M: M)
            else { return (false, "forwardRows nil") }
            ref.eval(); got.eval()
            let (ok, d) = bitEqual(got, ref)
            if !ok { return (false, "M=17: \(d)") }
            if str.lastStepChunks < 2 { return (false, "expected >=2 chunks M=17 C=\(C), got \(str.lastStepChunks)") }
            return (true, "ok (chunks=\(str.lastStepChunks))")
        }

        // Test 27 (D1-C): eviction chain — 3-step [3,3,3] に渡る LRU 退去・再ロードが bit-exact を保持する。
        run("stream_fused_eviction_chain") {
            guard let (device, _) = RawMetalForward.ensure() else { return (false, "no device") }
            let C = 8  // C < E=16 → eviction forced
            let moe0 = stMkMoE(), moe1 = stMkMoE()
            let gdnW = stMkGdnW(), attnW = stMkAttnW()
            let stCs0 = MLXRandom.normal([stCK - 1, stConvDim]).asType(.float16)
            let stRs0 = MLXRandom.normal([1, stHv, stDv, stDk]).asType(.float32)
            let stKc0 = MLXRandom.normal([stNkv, 16, stHd]).asType(.float16)
            let stVc0 = MLXRandom.normal([stNkv, 16, stHd]).asType(.float16)
            MLX.eval([stCs0, stRs0, stKc0, stVc0])
            func freshC() -> [RawVerifyForward.LayerCaches] {
                [RawVerifyForward.LayerCaches(convState: stCs0, recState: stRs0),
                 RawVerifyForward.LayerCaches(kCache: stKc0, vCache: stVc0)]
            }
            let layerSpecs = [
                RawVerifyForward.LayerSpec(isLinear: true,
                    inputLN: MLXRandom.normal([stH]).asType(.float16), postLN: MLXRandom.normal([stH]).asType(.float16),
                    gdn: gdnW, attn: nil, moe: moe0.w, moeE: stE, moeI: stI),
                RawVerifyForward.LayerSpec(isLinear: false,
                    inputLN: MLXRandom.normal([stH]).asType(.float16), postLN: MLXRandom.normal([stH]).asType(.float16),
                    gdn: nil, attn: attnW, moe: moe1.w, moeE: stE, moeI: stI),
            ]
            guard let tp0 = TestExpertProvider(E: stE, I: stI, H: stH, C: C,
                    gW: moe0.gW, gSf16: moe0.gSf16, gBf16: moe0.gBf16,
                    uW: moe0.uW, uSf16: moe0.uSf16, uBf16: moe0.uBf16,
                    dW: moe0.dW, dSf16: moe0.dSf16, dBf16: moe0.dBf16, device: device),
                  let tp1 = TestExpertProvider(E: stE, I: stI, H: stH, C: C,
                    gW: moe1.gW, gSf16: moe1.gSf16, gBf16: moe1.gBf16,
                    uW: moe1.uW, uSf16: moe1.uSf16, uBf16: moe1.uBf16,
                    dW: moe1.dW, dSf16: moe1.dSf16, dBf16: moe1.dBf16, device: device)
            else { return (false, "TestExpertProvider nil") }
            guard let res = RawFusedVerify.RawFusedForward(
                    layers: layerSpecs, caches: freshC(), maxM: 9, H: stH, maxSeqLen: 64),
                  let str = RawFusedVerify.RawFusedForward(
                    layers: layerSpecs, caches: freshC(), maxM: 9, H: stH, maxSeqLen: 64,
                    providers: [tp0, tp1])
            else { return (false, "init nil") }
            for si in 0..<3 {
                let M = 3
                let x = MLXRandom.normal([M, stH]).asType(.float16); x.eval()
                guard let ref = res.forwardRows(x, M: M),
                      let got = str.forwardRows(x, M: M)
                else { return (false, "forwardRows nil step=\(si)") }
                ref.eval(); got.eval()
                let (ok, d) = bitEqual(got, ref)
                if !ok { return (false, "step=\(si): \(d)") }
            }
            return (true, "ok")
        }

        // Test 28 (D1-D): bolt streaming ≡ resident — C=E=16 全エキスパート事前ロード+frozen table。
        run("stream_fused_bolt_exact_table") {
            guard let (device, _) = RawMetalForward.ensure() else { return (false, "no device") }
            let C = stE  // C = E: all experts fit without eviction
            let moe0 = stMkMoE(), moe1 = stMkMoE()
            let gdnW = stMkGdnW(), attnW = stMkAttnW()
            let stCs0 = MLXRandom.normal([stCK - 1, stConvDim]).asType(.float16)
            let stRs0 = MLXRandom.normal([1, stHv, stDv, stDk]).asType(.float32)
            let stKc0 = MLXRandom.normal([stNkv, 16, stHd]).asType(.float16)
            let stVc0 = MLXRandom.normal([stNkv, 16, stHd]).asType(.float16)
            MLX.eval([stCs0, stRs0, stKc0, stVc0])
            func freshC() -> [RawVerifyForward.LayerCaches] {
                [RawVerifyForward.LayerCaches(convState: stCs0, recState: stRs0),
                 RawVerifyForward.LayerCaches(kCache: stKc0, vCache: stVc0)]
            }
            let layerSpecs = [
                RawVerifyForward.LayerSpec(isLinear: true,
                    inputLN: MLXRandom.normal([stH]).asType(.float16), postLN: MLXRandom.normal([stH]).asType(.float16),
                    gdn: gdnW, attn: nil, moe: moe0.w, moeE: stE, moeI: stI),
                RawVerifyForward.LayerSpec(isLinear: false,
                    inputLN: MLXRandom.normal([stH]).asType(.float16), postLN: MLXRandom.normal([stH]).asType(.float16),
                    gdn: nil, attn: attnW, moe: moe1.w, moeE: stE, moeI: stI),
            ]
            // Create providers, warm all experts, build frozen slot tables.
            guard let tp0 = TestExpertProvider(E: stE, I: stI, H: stH, C: C,
                    gW: moe0.gW, gSf16: moe0.gSf16, gBf16: moe0.gBf16,
                    uW: moe0.uW, uSf16: moe0.uSf16, uBf16: moe0.uBf16,
                    dW: moe0.dW, dSf16: moe0.dSf16, dBf16: moe0.dBf16, device: device),
                  let tp1 = TestExpertProvider(E: stE, I: stI, H: stH, C: C,
                    gW: moe1.gW, gSf16: moe1.gSf16, gBf16: moe1.gBf16,
                    uW: moe1.uW, uSf16: moe1.uSf16, uBf16: moe1.uBf16,
                    dW: moe1.dW, dSf16: moe1.dSf16, dBf16: moe1.dBf16, device: device)
            else { return (false, "TestExpertProvider nil") }
            let sm0 = tp0.ensure(Array(0..<stE))  // warm all E experts into arena
            let sm1 = tp1.ensure(Array(0..<stE))
            var tbl0 = [Int32](repeating: 0, count: stE), tbl1 = [Int32](repeating: 0, count: stE)
            for (e, s) in sm0 { tbl0[e] = Int32(s) }
            for (e, s) in sm1 { tbl1[e] = Int32(s) }
            // Create resident (no providers) and bolt (with providers) forwards.
            guard let res = RawFusedVerify.RawFusedForward(
                    layers: layerSpecs, caches: freshC(), maxM: 17, H: stH, maxSeqLen: 64),
                  let blt = RawFusedVerify.RawFusedForward(
                    layers: layerSpecs, caches: freshC(), maxM: 17, H: stH, maxSeqLen: 64,
                    providers: [tp0, tp1])
            else { return (false, "init nil") }
            blt.setBoltTables([tbl0, tbl1])
            for M in [1, 2, 9, 17] {
                let x = MLXRandom.normal([M, stH]).asType(.float16); x.eval()
                guard let ref = res.forwardRows(x, M: M),
                      let got = blt.forwardRows(x, M: M)
                else { return (false, "forwardRows nil M=\(M)") }
                ref.eval(); got.eval()
                let (ok, d) = bitEqual(got, ref)
                if !ok { return (false, "M=\(M): \(d)") }
            }
            return (true, "ok")
        }

        // ── A3 acceptance gate tests (29-31) ────────────────────────────────────────
        //
        // WRITE-LOCKED: Haiku (implementer) MUST NOT modify these tests.
        // They encode the G1 acceptance gate from notes/04-a3-pending-prefix-spec.md §6.
        //
        // All three use the same 2-layer mini-model geometry as tests 25–28 (stH/stE/stI/
        // stMk* helpers defined above). The model has random weights — no real model load.
        //
        // Kernel status: the raw fused kernel is per-row order-stable (proven by test 23,
        // fused_forward_rows_bitexact). Therefore tests 29 and 30 are GREEN by premise —
        // they guard against future regressions and document the exact A3 index contract
        // for the implementer. Test 31 validates the flush path specifically.
        //
        // RED gate before A3 is implemented: test_a3.sh (G2) — the shell script checks
        // (a) QWISP_RAW_A3=1 is acknowledged by the binary (binary must log "A3" when active),
        // (b) OUT_TOKENS byte-identical between A3=0 and A3=1, and
        // (c) A3 self-check reports "128/128 LOSSLESS".
        // Since the binary currently ignores QWISP_RAW_A3, check (a) fails → RED.

        // Test 29 (T-A3-fuse): §4 invariant at kernel level.
        // forwardRows(pending+[u]+drafts) decision rows evals[pk..] are BIT-IDENTICAL to
        // forwardRows(pending) then forwardRows([u]+drafts), for pk ∈ {0,3,7,17} × D ∈ {1,4,8}.
        //
        // This is spec §4's "position-wise causal" invariant: row i output depends only
        // on rows 0..i, so a batched call is order-stable with respect to sequential calls.
        //
        // GREEN by premise: per-row order-stability is proven by test 23 (fused_forward_rows_bitexact).
        // This test specialises that property for the A3 pending-prefix shapes and serves as
        // a regression guard. It does NOT require A3 implementation in RawSpecRunner.swift.
        run("a3_fuse_invariant") {
            let H = stH
            let moe0a = stMkMoE(), moe1a = stMkMoE()
            let gdnWa = stMkGdnW(), attnWa = stMkAttnW()
            let csA = MLXRandom.normal([stCK - 1, stConvDim]).asType(.float16)
            let rsA = MLXRandom.normal([1, stHv, stDv, stDk]).asType(.float32)
            let kcA = MLXRandom.normal([stNkv, 8, stHd]).asType(.float16)   // 8-position initial KV
            let vcA = MLXRandom.normal([stNkv, 8, stHd]).asType(.float16)
            MLX.eval([csA, rsA, kcA, vcA])
            let iLN0a = MLXRandom.normal([H]).asType(.float16)
            let pLN0a = MLXRandom.normal([H]).asType(.float16)
            let iLN1a = MLXRandom.normal([H]).asType(.float16)
            let pLN1a = MLXRandom.normal([H]).asType(.float16)
            MLX.eval([iLN0a, pLN0a, iLN1a, pLN1a])
            let layersA = [
                RawVerifyForward.LayerSpec(isLinear: true,
                    inputLN: iLN0a, postLN: pLN0a,
                    gdn: gdnWa, attn: nil, moe: moe0a.w, moeE: stE, moeI: stI),
                RawVerifyForward.LayerSpec(isLinear: false,
                    inputLN: iLN1a, postLN: pLN1a,
                    gdn: nil, attn: attnWa, moe: moe1a.w, moeE: stE, moeI: stI),
            ]
            func freshCachesA() -> [RawVerifyForward.LayerCaches] {
                [RawVerifyForward.LayerCaches(convState: csA, recState: rsA),
                 RawVerifyForward.LayerCaches(kCache: kcA, vCache: vcA)]
            }
            // spec §6 G1 matrix: pk ∈ {0,3,7,17} × D ∈ {1,4,8}
            for pk in [0, 3, 7, 17] {
                for D in [1, 4, 8] {
                    let M = pk + 1 + D   // total rows in the batched call
                    // maxSeqLen covers initial 8 + M + margin
                    guard let fused = RawFusedVerify.RawFusedForward(
                        layers: layersA, caches: freshCachesA(),
                        maxM: M + 2, H: H, maxSeqLen: 8 + M + 8)
                    else { return (false, "init nil pk=\(pk) D=\(D)") }
                    let allX = MLXRandom.normal([M, H]).asType(.float16); allX.eval()
                    // ── BATCHED PATH: forwardRows(all M rows) ──
                    let snapA = fused.snapshot()
                    guard let batchOut = fused.forwardRows(allX, M: M)
                    else { return (false, "batch fwd nil pk=\(pk) D=\(D)") }
                    batchOut.eval()
                    // Decision rows: indices [pk .. M-1], shape [1+D, H]
                    let decisionBatch = batchOut[pk ..< M]; decisionBatch.eval()
                    // ── SEQUENTIAL PATH: rollback → forward(pending if pk>0) → forward([u]+drafts) ──
                    fused.rollbackOneStep(snapA)
                    if pk > 0 {
                        guard let _ = fused.forwardRows(allX[0 ..< pk], M: pk)
                        else { return (false, "pending fwd nil pk=\(pk) D=\(D)") }
                    }
                    guard let seqOut = fused.forwardRows(allX[pk ..< M], M: 1 + D)
                    else { return (false, "seq fwd nil pk=\(pk) D=\(D)") }
                    seqOut.eval()
                    // ── ASSERT: decision rows bit-identical ──
                    let (ok29, d29) = bitEqual(decisionBatch, seqOut)
                    if !ok29 { return (false, "pk=\(pk) D=\(D): \(d29)") }
                }
            }
            return (true, "ok")
        }

        // Test 30 (T-A3-reject-rollback): 1-step rollback regularity at kernel level.
        // Simulates a partial reject at position p, then verifies the next iteration
        // produces bit-identical hidden states whether:
        //   non-A3 path: rollback → rebuild forward([u]+drafts.prefix(p)) [cache→B+pk] →
        //                forward([u']+next_drafts) from state B+pk  → H_nonA3
        //   A3 path proxy: rollback → skip rebuild [cache stays at B] →
        //                  forwardRows(pending+[u']+next_drafts) from state B at rows [pk..] → H_A3
        //   Assert H_nonA3 == H_A3  (by §4 invariant)
        //
        // Fixed parameters: D=4 (step-1 drafts), p=2 (reject position), pk=p+1=3 (pending count),
        // D2=3 (step-2 drafts). This concretises the spec §3 reject→pending scenario.
        //
        // GREEN by premise: this is the §4 invariant applied to a 2-step sequence with a reject
        // in between. Additional value: verifies rollbackOneStep correctly undoes a multi-row
        // (M=D+1=5) batched forward, which is the per-reject rollback pattern in the A3 loop.
        run("a3_reject_rollback_equiv") {
            let H = stH
            let D   = 4       // drafts in step 1
            let p   = 2       // reject position: first p drafts accepted
            let pk  = p + 1   // pending count: [u]+drafts.prefix(p) = 3 tokens
            let D2  = 3       // drafts in step 2 (next verify after reject)
            let M1  = D + 1         // step-1 batched rows = 5
            let M2  = pk + D2 + 1   // step-2 A3 batched rows = pending+[u']+next_drafts = 7
            let moe0b = stMkMoE(), moe1b = stMkMoE()
            let gdnWb = stMkGdnW(), attnWb = stMkAttnW()
            let csB = MLXRandom.normal([stCK - 1, stConvDim]).asType(.float16)
            let rsB = MLXRandom.normal([1, stHv, stDv, stDk]).asType(.float32)
            let kcB = MLXRandom.normal([stNkv, 4, stHd]).asType(.float16)
            let vcB = MLXRandom.normal([stNkv, 4, stHd]).asType(.float16)
            MLX.eval([csB, rsB, kcB, vcB])
            let iLN0b = MLXRandom.normal([H]).asType(.float16)
            let pLN0b = MLXRandom.normal([H]).asType(.float16)
            let iLN1b = MLXRandom.normal([H]).asType(.float16)
            let pLN1b = MLXRandom.normal([H]).asType(.float16)
            MLX.eval([iLN0b, pLN0b, iLN1b, pLN1b])
            let layersB = [
                RawVerifyForward.LayerSpec(isLinear: true,
                    inputLN: iLN0b, postLN: pLN0b,
                    gdn: gdnWb, attn: nil, moe: moe0b.w, moeE: stE, moeI: stI),
                RawVerifyForward.LayerSpec(isLinear: false,
                    inputLN: iLN1b, postLN: pLN1b,
                    gdn: nil, attn: attnWb, moe: moe1b.w, moeE: stE, moeI: stI),
            ]
            func freshCachesB() -> [RawVerifyForward.LayerCaches] {
                [RawVerifyForward.LayerCaches(convState: csB, recState: rsB),
                 RawVerifyForward.LayerCaches(kCache: kcB, vCache: vcB)]
            }
            let maxSeqB = 4 + M1 + M2 + 8    // initial(4) + max scenario(M1 or M2) + margin
            let maxMb   = Swift.max(M1, M2) + 2
            // Two independent fused forwards starting from identical state B
            guard let fusedNA = RawFusedVerify.RawFusedForward(
                    layers: layersB, caches: freshCachesB(), maxM: maxMb, H: H, maxSeqLen: maxSeqB),
                  let fusedA3 = RawFusedVerify.RawFusedForward(
                    layers: layersB, caches: freshCachesB(), maxM: maxMb, H: H, maxSeqLen: maxSeqB)
            else { return (false, "init nil") }
            // Random hidden inputs shared between paths
            let step1X  = MLXRandom.normal([M1, H]).asType(.float16); step1X.eval()
            let step2X  = MLXRandom.normal([D2 + 1, H]).asType(.float16); step2X.eval()
            let pendingX = step1X[0 ..< pk]   // [u]+drafts.prefix(p), shape [pk=3, H]
            // A3 step-2 batch: pending + [u'] + next_drafts = [pk+D2+1, H]
            let step2A3X = MLX.concatenated([pendingX, step2X], axis: 0); step2A3X.eval()
            // ── NON-A3 PATH ──────────────────────────────────────────────────────
            // Step 1: batched verify [u]+drafts → reject at p → rollback to B → rebuild forward(pending)
            let snapNA = fusedNA.snapshot()
            guard let _ = fusedNA.forwardRows(step1X, M: M1)
            else { return (false, "non-A3 step1 nil") }
            fusedNA.rollbackOneStep(snapNA)              // cache back to B
            guard let _ = fusedNA.forwardRows(pendingX, M: pk)  // rebuild → B+pk
            else { return (false, "non-A3 rebuild nil") }
            // Step 2: from rebuilt state B+pk, forward([u']+next_drafts)
            guard let hNA = fusedNA.forwardRows(step2X, M: D2 + 1)
            else { return (false, "non-A3 step2 nil") }
            hNA.eval()
            // ── A3 PATH PROXY (kernel-level simulation) ──────────────────────────
            // Step 1: same verify → reject at p → rollback to B → NO rebuild (cache stays at B)
            let snapA3 = fusedA3.snapshot()
            guard let _ = fusedA3.forwardRows(step1X, M: M1)
            else { return (false, "A3 step1 nil") }
            fusedA3.rollbackOneStep(snapA3)              // cache back to B; no rebuild
            // Step 2 (A3): fuse pending+[u']+next_drafts in one batched call from B
            guard let hA3full = fusedA3.forwardRows(step2A3X, M: M2)
            else { return (false, "A3 step2 fused nil") }
            hA3full.eval()
            let hA3 = hA3full[pk ..< M2]; hA3.eval()   // decision rows [pk..M2-1], shape [D2+1, H]
            // ── ASSERT: non-A3 step-2 ≡ A3 step-2 decision rows (bit-identical) ──
            let (ok30, d30) = bitEqual(hNA, hA3)
            if !ok30 { return (false, "non-A3 vs A3: \(d30)") }
            return (true, "ok")
        }

        // Test 31 (T-A3-flush): pending cap(24) flush advances the committed boundary correctly.
        // Asserts that flushing 24 pending tokens in one batched forward, then forwarding a
        // verify batch ([u]+drafts D=3), produces decision rows bit-identical to a reference
        // path of 28 sequential single-row forwards.
        //
        // Validates the flush mechanism (§3: "pending.count が cap を超えたら forward(pending)"):
        //   flush path: forwardRows(pending×24) → forwardRows([u]+drafts×4) → verify rows
        //   ref path:   forwardRows([xi])×28 sequentially → same verify rows
        //   Assert: verify rows bit-identical.
        //
        // GREEN by premise: kernel order-stability (test 23). This test catches A3 flush bugs
        // where the pending buffer is mis-serialised, the boundary advances by the wrong count,
        // or the subsequent verify call uses a stale cache state.
        run("a3_flush_boundary") {
            let H          = stH
            let pendingCap = 24     // A3 pending cap from spec §3 (same as MLX version)
            let flushN     = pendingCap    // flush exactly at cap
            let D          = 3             // drafts after flush
            let M_total    = flushN + 1 + D  // 28 positions total in reference path
            let moe0c = stMkMoE(), moe1c = stMkMoE()
            let gdnWc = stMkGdnW(), attnWc = stMkAttnW()
            let csC = MLXRandom.normal([stCK - 1, stConvDim]).asType(.float16)
            let rsC = MLXRandom.normal([1, stHv, stDv, stDk]).asType(.float32)
            let kcC = MLXRandom.normal([stNkv, 4, stHd]).asType(.float16)  // 4-position initial KV
            let vcC = MLXRandom.normal([stNkv, 4, stHd]).asType(.float16)
            MLX.eval([csC, rsC, kcC, vcC])
            let iLN0c = MLXRandom.normal([H]).asType(.float16)
            let pLN0c = MLXRandom.normal([H]).asType(.float16)
            let iLN1c = MLXRandom.normal([H]).asType(.float16)
            let pLN1c = MLXRandom.normal([H]).asType(.float16)
            MLX.eval([iLN0c, pLN0c, iLN1c, pLN1c])
            let layersC = [
                RawVerifyForward.LayerSpec(isLinear: true,
                    inputLN: iLN0c, postLN: pLN0c,
                    gdn: gdnWc, attn: nil, moe: moe0c.w, moeE: stE, moeI: stI),
                RawVerifyForward.LayerSpec(isLinear: false,
                    inputLN: iLN1c, postLN: pLN1c,
                    gdn: nil, attn: attnWc, moe: moe1c.w, moeE: stE, moeI: stI),
            ]
            func freshCachesC() -> [RawVerifyForward.LayerCaches] {
                [RawVerifyForward.LayerCaches(convState: csC, recState: rsC),
                 RawVerifyForward.LayerCaches(kCache: kcC, vCache: vcC)]
            }
            // maxSeqLen: initial(4) + M_total(28) + margin = 40; use 64 for safety
            // maxM: must cover flushN=24 (largest single call in flush path)
            guard let refFused = RawFusedVerify.RawFusedForward(
                    layers: layersC, caches: freshCachesC(),
                    maxM: flushN + 2, H: H, maxSeqLen: 64),
                  let flushFused = RawFusedVerify.RawFusedForward(
                    layers: layersC, caches: freshCachesC(),
                    maxM: flushN + 2, H: H, maxSeqLen: 64)
            else { return (false, "init nil") }
            // Random inputs for all M_total positions (shared between paths)
            let allXc     = MLXRandom.normal([M_total, H]).asType(.float16); allXc.eval()
            let pendingXc = allXc[0 ..< flushN]        // 24 rows (flush batch)
            let verifyXc  = allXc[flushN ..< M_total]  // 4 rows ([u]+drafts)
            // ── REFERENCE PATH: M_total sequential single-row forwards ────────────
            var refParts: [MLXArray] = []
            for i in 0 ..< M_total {
                let xi = allXc[i ..< i + 1]    // [1, H]
                guard let out = refFused.forwardRows(xi, M: 1)
                else { return (false, "ref fwd nil i=\(i)") }
                out.eval()
                if i >= flushN { refParts.append(out) }   // collect verify rows only
            }
            let refOutC = MLX.concatenated(refParts, axis: 0); refOutC.eval()   // [1+D, H]
            // ── FLUSH PATH: batch forward(pending=24) then batch forward([u]+drafts=4) ──
            guard let _ = flushFused.forwardRows(pendingXc, M: flushN)   // flush pending
            else { return (false, "flush fwd nil") }
            guard let flushOutC = flushFused.forwardRows(verifyXc, M: 1 + D)  // verify
            else { return (false, "verify fwd nil") }
            flushOutC.eval()
            // ── ASSERT: flush path verify rows ≡ reference path verify rows ──
            let (ok31, d31) = bitEqual(flushOutC, refOutC)
            if !ok31 { return (false, "flush boundary: \(d31)") }
            return (true, "ok")
        }

        // ── G1 gate: fuse_gu tests (32-33) — notes/06-fusion-poc-spec.md §4 ─────────────
        //
        // WRITE-LOCKED: Haiku (implementer) MUST NOT modify these tests.
        // Reference is computed via existing production kernels only (gatherQmmRows × 2 +
        // swigluRaw) — never via MLX or CPU reimplementation, so bit-identity with the
        // current 3-kernel chain is the sole correctness contract.
        //
        // Note: integration-level flag QWISP_FUSE_GU (encodeMoEGatherRowsRange branch) is
        // gated separately by G2 real-weight identity; these unit tests do NOT cover it.

        // Test 32 (G1-bitexact): gatherQmmSwigluRows output ≡ gatherQmmRows×2+swigluRaw.
        // Sweeps M∈{1,8,17} × Ktop∈{1,8} × N∈{512,1536}, K=2048, E=16, gs=64.
        run("fuse_gu_bitexact") {
            let E = 16, K = 2048
            func q4e(_ e: Int, _ n: Int, _ k: Int) -> (MLXArray, MLXArray, MLXArray) {
                let wf = MLXRandom.normal([e, n, k]).asType(.float16)
                let (q, s, b) = MLX.quantized(wf, groupSize: 64, bits: 4, mode: .affine)
                return (q, s, b!)
            }
            // Deterministic expert index pool (fixed seed carried from suite)
            let pool: [Int32] = (0..<200).map { Int32(($0 * 13 + 3) % E) }
            for N in [512, 1536] {
                let (wGq, wGS, wGB) = q4e(E, N, K)
                let (wUq, wUS, wUB) = q4e(E, N, K)
                MLX.eval([wGq, wGS, wGB, wUq, wUS, wUB])
                for M in [1, 8, 17] {
                    for Ktop in [1, 8] {
                        let x = MLXRandom.normal([M, K]).asType(.float16)
                        let indsFlat = (0..<M*Ktop).map { pool[$0 % pool.count] }
                        let inds = MLXArray(indsFlat, [M * Ktop])
                        MLX.eval([x, inds])
                        // Reference: existing 3-kernel chain (production kernels only — no MLX/CPU reimpl)
                        guard let g = RawMetalForward.gatherQmmRows(x, wGq, scales: wGS, biases: wGB,
                                                                     inds: inds, M: M, Ktop: Ktop, K: K, N: N),
                              let u = RawMetalForward.gatherQmmRows(x, wUq, scales: wUS, biases: wUB,
                                                                     inds: inds, M: M, Ktop: Ktop, K: K, N: N)
                        else { return (false, "ref gather nil M=\(M) Ktop=\(Ktop) N=\(N)") }
                        g.eval(); u.eval()
                        guard let hRef = RawMetalForward.swigluRaw(g, u)
                        else { return (false, "ref swiglu nil M=\(M) Ktop=\(Ktop) N=\(N)") }
                        hRef.eval()
                        // Stub under test
                        guard let hGot = RawFusedVerify.gatherQmmSwigluRows(
                            x: x, inds: inds,
                            wG: wGq, sG: wGS, bG: wGB,
                            wU: wUq, sU: wUS, bU: wUB,
                            M: M, Ktop: Ktop, K: K, N: N)
                        else { return (false, "not implemented (M=\(M) Ktop=\(Ktop) N=\(N))") }
                        hGot.eval()
                        let (ok, d) = bitEqual(hGot, hRef)
                        if !ok { return (false, "M=\(M) Ktop=\(Ktop) N=\(N): \(d)") }
                    }
                }
            }
            return (true, "ok")
        }

        // Test 33 (G1-m_invariance): fused kernel batched (M=8,Ktop=8) ≡ M=1 per-row loop.
        // Verifies M-independence of the fused kernel (same idiom as existing gather tests).
        run("fuse_gu_m_invariance") {
            let E = 16, K = 2048, N = 512, M = 8, Ktop = 8
            let wGq: MLXArray, wGS: MLXArray, wGB: MLXArray
            let wUq: MLXArray, wUS: MLXArray, wUB: MLXArray
            do {
                let wgf = MLXRandom.normal([E, N, K]).asType(.float16)
                let (q, s, b) = MLX.quantized(wgf, groupSize: 64, bits: 4, mode: .affine)
                (wGq, wGS, wGB) = (q, s, b!)
            }
            do {
                let wuf = MLXRandom.normal([E, N, K]).asType(.float16)
                let (q, s, b) = MLX.quantized(wuf, groupSize: 64, bits: 4, mode: .affine)
                (wUq, wUS, wUB) = (q, s, b!)
            }
            let pool: [Int32] = (0..<200).map { Int32(($0 * 7 + 5) % E) }
            let x = MLXRandom.normal([M, K]).asType(.float16)
            let indsFlat = (0..<M*Ktop).map { pool[$0 % pool.count] }
            let inds = MLXArray(indsFlat, [M * Ktop])
            MLX.eval([wGq, wGS, wGB, wUq, wUS, wUB, x, inds])
            // Batched call M=8
            guard let hBatch = RawFusedVerify.gatherQmmSwigluRows(
                x: x, inds: inds,
                wG: wGq, sG: wGS, bG: wGB,
                wU: wUq, sU: wUS, bU: wUB,
                M: M, Ktop: Ktop, K: K, N: N)
            else { return (false, "not implemented (M=\(M))") }
            hBatch.eval()
            // Per-row M=1 loop of the same fused kernel
            var refParts: [MLXArray] = []
            for m in 0..<M {
                let xm = x[m ..< m+1]
                let rowInds = MLXArray(Array(indsFlat[m*Ktop ..< (m+1)*Ktop]), [Ktop])
                xm.eval(); rowInds.eval()
                guard let hm = RawFusedVerify.gatherQmmSwigluRows(
                    x: xm, inds: rowInds,
                    wG: wGq, sG: wGS, bG: wGB,
                    wU: wUq, sU: wUS, bU: wUB,
                    M: 1, Ktop: Ktop, K: K, N: N)
                else { return (false, "not implemented (M=1 m=\(m))") }
                hm.eval(); refParts.append(hm)
            }
            let hLoop = MLX.concatenated(refParts, axis: 0); hLoop.eval()
            return bitEqual(hBatch, hLoop)
        }

        // ── Wave 1 GDN fusion tests (34-37) ──────────────────────────────
        //
        // WRITE-LOCKED: implementer MUST NOT modify these tests.
        // They encode the G1 acceptance gate from notes/07-gdn-fusion-spec.md §3 Wave 1.
        //
        // All four tests use random quantised weights with the canonical GDN geometry
        // (H=2048, Hk=16, Dk=128, Hv=32, Dv=128, convKernel=4) matching test 13.
        // References use ONLY existing production kernels (qmmRows, conv1dSiluRows,
        // rmsNormRows, gateRaw) — never CPU/MLX computation reimplementations.
        // Tests are RED (FAIL with "not implemented") until Wave 1 is implemented.

        // Test 34 (F1): gdnInProjConcat — single qmmRows on N-axis-concatenated in-proj
        //   weights ≡ four separate qmmRows calls, outputs bit-identical at all offsets.
        //   Validates: (a) concat build is correct, (b) offsets slice right, (c) M∈{1,8}.
        run("fuse_gdn_inproj_concat") {
            let H = 2048, Hk = 16, Dk = 128, Hv = 32, Dv = 128
            let convDim  = Hk * Dk * 2 + Hv * Dv    // 8192
            let valueDim = Hv * Dv                    // 4096
            let numVH    = Hv                         // 32
            func quant(_ n: Int, _ k: Int) -> (MLXArray, MLXArray, MLXArray) {
                let wf = MLXRandom.normal([n, k]).asType(.float16)
                let (q, s, b) = MLX.quantized(wf, groupSize: 64, bits: 4, mode: .affine)
                return (q, s, b!)
            }
            let (qkvW, qkvS, qkvB) = quant(convDim,  H)
            let (zW,   zS,   zB)   = quant(valueDim, H)
            let (bW,   bS,   bB)   = quant(numVH,    H)
            let (aW,   aS,   aB)   = quant(numVH,    H)
            MLX.eval([qkvW, qkvS, qkvB, zW, zS, zB, bW, bS, bB, aW, aS, aB])
            // Fused: build concatenated in-proj triple (stub under test)
            guard let (catW, catS, catB) = RawFusedVerify.gdnInProjConcat(
                qkvW: qkvW, qkvS: qkvS, qkvB: qkvB,
                zW:   zW,   zS:   zS,   zB:   zB,
                bW:   bW,   bS:   bS,   bB:   bB,
                aW:   aW,   aS:   aS,   aB:   aB)
            else { return (false, "not implemented") }
            MLX.eval([catW, catS, catB])
            let totalN = convDim + valueDim + numVH + numVH   // 12352
            for M in [1, 8] {
                let x = MLXRandom.normal([M, H]).asType(.float16); x.eval()
                // Reference: 4 separate qmmRows (existing production kernel)
                guard let refQkv = RawMetalForward.qmmRows(x, qkvW, scales: qkvS, biases: qkvB,
                                                            M: M, K: H, N: convDim),
                      let refZ   = RawMetalForward.qmmRows(x, zW,   scales: zS,   biases: zB,
                                                            M: M, K: H, N: valueDim),
                      let refB   = RawMetalForward.qmmRows(x, bW,   scales: bS,   biases: bB,
                                                            M: M, K: H, N: numVH),
                      let refA   = RawMetalForward.qmmRows(x, aW,   scales: aS,   biases: aB,
                                                            M: M, K: H, N: numVH)
                else { return (false, "ref qmmRows nil M=\(M)") }
                MLX.eval([refQkv, refZ, refB, refA])
                // Fused: single qmmRows on concatenated weight, sliced at offsets
                guard let fused = RawMetalForward.qmmRows(x, catW, scales: catS, biases: catB,
                                                           M: M, K: H, N: totalN)
                else { return (false, "fused qmmRows nil M=\(M)") }
                fused.eval()
                var off = 0
                let gotQkv = fused[0..., off ..< off + convDim];  off += convDim
                let gotZ   = fused[0..., off ..< off + valueDim]; off += valueDim
                let gotB   = fused[0..., off ..< off + numVH];    off += numVH
                let gotA   = fused[0..., off ..< off + numVH]
                MLX.eval([gotQkv, gotZ, gotB, gotA])
                for (nm, got, ref) in [("qkv", gotQkv, refQkv), ("z", gotZ, refZ),
                                       ("b", gotB, refB), ("a", gotA, refA)] {
                    let (ok, d) = bitEqual(got, ref)
                    if !ok { return (false, "M=\(M) \(nm): \(d)") }
                }
            }
            return (true, "ok")
        }

        // Test 35 (F3): gdnConvShiftFused — fused conv1d_silu_hist + shift_conv
        //   ≡ conv1dSiluRows (production kernel) + tail-slice (pure data movement).
        //   Validates both convOut and histOut are bit-exact, M∈{1,8}.
        run("fuse_gdn_conv_shift") {
            let Hk = 16, Dk = 128, Hv = 32, Dv = 128, K = 4
            let C = Hk * Dk * 2 + Hv * Dv    // convDim = 8192
            let convW = MLXRandom.normal([C, K]).asType(.float16); convW.eval()
            for M in [1, 8] {
                let histIn = MLXRandom.normal([K - 1, C]).asType(.float16)   // [K-1, C]
                let qkv    = MLXRandom.normal([M, C]).asType(.float16)        // [M, C]
                MLX.eval([histIn, qkv])
                // Reference convOut: conv1dSiluRows with explicit windows (existing production kernel).
                // conv1d_silu_hist_rows and conv1dSiluRows compute identical silu(conv1d(.)) —
                // test 22 (fused_gdn_layer_bitexact) proves them bit-identical for this data layout.
                let convInput = MLX.concatenated([histIn, qkv], axis: 0); convInput.eval()
                let windowParts = (0 ..< M).map { convInput[$0 ..< $0 + K].reshaped([1, K, C]) }
                let windows = MLX.concatenated(windowParts, axis: 0); windows.eval()
                guard let convOutRef = RawMetalForward.conv1dSiluRows(windows, convW, M: M, K: K, C: C)
                else { return (false, "ref conv1dSiluRows nil M=\(M)") }
                // Reference histOut: tail K-1 frames of (histIn‖qkv) — pure data movement,
                // same bytes shift_conv_rows copies; not a computation reimplementation.
                let histOutRef = convInput[M ..< M + K - 1].asType(.float16)
                MLX.eval([convOutRef, histOutRef])
                // Fused: stub under test
                guard let (convOutGot, histOutGot) = RawFusedVerify.gdnConvShiftFused(
                    histIn: histIn, qkv: qkv, w: convW, M: M, K: K, C: C)
                else { return (false, "not implemented (M=\(M))") }
                MLX.eval([convOutGot, histOutGot])
                let (ok1, d1) = bitEqual(convOutGot, convOutRef)
                if !ok1 { return (false, "convOut M=\(M): \(d1)") }
                let (ok2, d2) = bitEqual(histOutGot, histOutRef)
                if !ok2 { return (false, "histOut M=\(M): \(d2)") }
            }
            return (true, "ok")
        }

        // Test 36 (F4): gdnNormGateFused — fused per-head rmsnorm + gate
        //   ≡ rmsNormRows (production kernel) then gateRaw (production kernel).
        //   Sweeps M∈{1,8} × promoteF32∈{false,true}.
        run("fuse_gdn_norm_gate") {
            let Hv = 32, Dv = 128
            let valueDim = Hv * Dv   // 4096
            for M in [1, 8] {
                let coreOut = MLXRandom.normal([M * Hv, Dv]).asType(.float16)   // [M*Hv, Dv]
                let z       = MLXRandom.normal([M, valueDim]).asType(.float16)   // [M, valueDim]
                MLX.eval([coreOut, z])
                for promote in [false, true] {
                    let normW = MLXRandom.normal([Dv]).asType(promote ? .float32 : .float16)
                    normW.eval()
                    // Reference: rmsNormRows (existing production kernel)
                    guard let normedRef = RawMetalForward.rmsNormRows(coreOut, normW,
                                                                        M: M * Hv, eps: 1e-6, D: Dv,
                                                                        promoteF32: promote)
                    else { return (false, "ref rmsNormRows nil M=\(M) promote=\(promote)") }
                    normedRef.eval()
                    // then gateRaw (existing production kernel wrapping encodeGate)
                    guard let outVRef = RawFusedVerify.gateRaw(z, normedRef,
                                                                promote: promote, total: M * valueDim)
                    else { return (false, "ref gateRaw nil M=\(M) promote=\(promote)") }
                    outVRef.eval()
                    // Fused: stub under test
                    guard let outVGot = RawFusedVerify.gdnNormGateFused(
                        coreOut: coreOut, z: z, normWeight: normW,
                        M: M, Hv: Hv, Dv: Dv, eps: 1e-6, promoteF32: promote)
                    else { return (false, "not implemented (M=\(M) promote=\(promote))") }
                    outVGot.eval()
                    let (ok, d) = bitEqual(outVGot, outVRef.reshaped([M, valueDim]))
                    if !ok { return (false, "M=\(M) promote=\(promote): \(d)") }
                }
            }
            return (true, "ok")
        }

        // Test 37 (fuseGU M-branch): structural predicate gate for the fuseGU M-branch.
        // Asserts that RawFusedForward.fuseGUActive(M:) is implemented and returns:
        //   true  iff QWISP_FUSE_GU=1 AND M==1   (fused gather+swiglu, M=1 only)
        //   false iff QWISP_FUSE_GU=1 AND M>1    (register-pressure fallback)
        //   false iff QWISP_FUSE_GU=0             (flag disabled)
        // Stub returns nil → RED. Behavioral correctness (actual tokens) is gated by G2.
        run("fuse_gu_m_branch") {
            guard let active1 = RawFusedVerify.RawFusedForward.fuseGUActive(M: 1)
            else { return (false, "not implemented") }
            guard let active8 = RawFusedVerify.RawFusedForward.fuseGUActive(M: 8)
            else { return (false, "not implemented M=8") }
            let fuseOn = RawFusedVerify.RawFusedForward.fuseGU
            if fuseOn {
                if !active1 { return (false, "fuseGU=1: expected fuseGUActive(1)=true, got false") }
                if  active8 { return (false, "fuseGU=1: expected fuseGUActive(8)=false, got true") }
            } else {
                if  active1 { return (false, "fuseGU=0: expected fuseGUActive(1)=false, got true") }
                if  active8 { return (false, "fuseGU=0: expected fuseGUActive(8)=false, got true") }
            }
            return (true, "ok")
        }

        // ── Wave 1 GDN fusion re-design tests (38-39) — §6 F1/F4 re-design ──
        //
        // WRITE-LOCKED: implementer MUST NOT modify these tests.
        // They gate the redesigned F1 (demux-type single dispatch) and F4 (true fused
        // single-dispatch kernel) from the §6 adversarial review verdict.
        // F1 old design: concat+slice×4=5 dispatch self-defeat. New: 1 dispatch demux.
        // F4 old design: 2-kernel wrapper = 0 dispatch reduction. New: 1 true kernel.
        // Tests are RED (FAIL with "not implemented") until the production kernels are implemented.

        // Test 38 (F1-demux): gdnInProjDemux — single qmm4 dispatch over concatenated in-proj
        //   weights writes DIRECTLY into 4 separate output buffers (no downstream concat+slice).
        //   Reference: 4 separate RawMetalForward.qmmRows (existing production kernel).
        //   Dims: qkv=1024, z=512, b=64, a=64 (all multiples of 8, threadgroup column alignment),
        //   K=512. Build triples via the existing gdnInProjConcat helper. M∈{1,8}.
        run("fuse_gdn_inproj_demux") {
            let K = 512
            let dims = (qkv: 1024, z: 512, b: 64, a: 64)   // all multiples of 8 (threadgroup alignment)
            func quant(_ n: Int, _ k: Int) -> (MLXArray, MLXArray, MLXArray) {
                let wf = MLXRandom.normal([n, k]).asType(.float16)
                let (q, s, b) = MLX.quantized(wf, groupSize: 64, bits: 4, mode: .affine)
                return (q, s, b!)
            }
            let (qkvW, qkvS, qkvB) = quant(dims.qkv, K)
            let (zW,   zS,   zB)   = quant(dims.z,   K)
            let (bW,   bS,   bB)   = quant(dims.b,   K)
            let (aW,   aS,   aB)   = quant(dims.a,   K)
            MLX.eval([qkvW, qkvS, qkvB, zW, zS, zB, bW, bS, bB, aW, aS, aB])
            // Build concatenated weight triple via existing gdnInProjConcat helper
            guard let (catW, catS, catB) = RawFusedVerify.gdnInProjConcat(
                qkvW: qkvW, qkvS: qkvS, qkvB: qkvB,
                zW:   zW,   zS:   zS,   zB:   zB,
                bW:   bW,   bS:   bS,   bB:   bB,
                aW:   aW,   aS:   aS,   aB:   aB)
            else { return (false, "gdnInProjConcat nil") }
            MLX.eval([catW, catS, catB])
            for M in [1, 8] {
                let x = MLXRandom.normal([M, K]).asType(.float16); x.eval()
                // Reference: 4 separate qmmRows (existing production kernel)
                guard let refQkv = RawMetalForward.qmmRows(x, qkvW, scales: qkvS, biases: qkvB,
                                                            M: M, K: K, N: dims.qkv),
                      let refZ   = RawMetalForward.qmmRows(x, zW,   scales: zS,   biases: zB,
                                                            M: M, K: K, N: dims.z),
                      let refB   = RawMetalForward.qmmRows(x, bW,   scales: bS,   biases: bB,
                                                            M: M, K: K, N: dims.b),
                      let refA   = RawMetalForward.qmmRows(x, aW,   scales: aS,   biases: aB,
                                                            M: M, K: K, N: dims.a)
                else { return (false, "ref qmmRows nil M=\(M)") }
                MLX.eval([refQkv, refZ, refB, refA])
                // Stub under test: single dispatch demuxing into 4 output buffers
                guard let (gotQkv, gotZ, gotB, gotA) = RawFusedVerify.gdnInProjDemux(
                    x: x, catW: catW, catS: catS, catB: catB,
                    M: M, K: K, dims: dims)
                else { return (false, "not implemented (M=\(M))") }
                MLX.eval([gotQkv, gotZ, gotB, gotA])
                for (nm, got, ref) in [("qkv", gotQkv, refQkv), ("z", gotZ, refZ),
                                       ("b", gotB, refB), ("a", gotA, refA)] {
                    let (ok, d) = bitEqual(got, ref)
                    if !ok { return (false, "M=\(M) \(nm): \(d)") }
                }
            }
            return (true, "ok")
        }

        // Test 39 (F4-true): gdnNormGateRows — TRUE single-dispatch fused kernel (per-(m,head)
        //   threadgroup: rmsnorm reduction identical to existing rmsnorm kernel, then silu(z)⊙
        //   in registers). This is DISTINCT from the existing fuse_gdn_norm_gate test which gates
        //   the wrapper gdnNormGateFused (2 separate dispatches chained). This test gates the
        //   single-dispatch production kernel that actually reduces dispatch count.
        //   Reference: rmsNormRows (existing production kernel) + gateRaw chain.
        //   Bit-equal for M∈{1,8} × promoteF32∈{false,true}.
        run("fuse_gdn_norm_gate_true") {
            let Hv = 32, Dv = 128
            let valueDim = Hv * Dv   // 4096
            for M in [1, 8] {
                let coreOut = MLXRandom.normal([M * Hv, Dv]).asType(.float16)   // [M*Hv, Dv]
                let z       = MLXRandom.normal([M, valueDim]).asType(.float16)   // [M, valueDim]
                MLX.eval([coreOut, z])
                for promote in [false, true] {
                    let normW = MLXRandom.normal([Dv]).asType(promote ? .float32 : .float16)
                    normW.eval()
                    // Reference: rmsNormRows (existing production kernel) then gateRaw
                    guard let normedRef = RawMetalForward.rmsNormRows(coreOut, normW,
                                                                       M: M * Hv, eps: 1e-6, D: Dv,
                                                                       promoteF32: promote)
                    else { return (false, "ref rmsNormRows nil M=\(M) promote=\(promote)") }
                    normedRef.eval()
                    guard let outVRef = RawFusedVerify.gateRaw(z, normedRef,
                                                                promote: promote, total: M * valueDim)
                    else { return (false, "ref gateRaw nil M=\(M) promote=\(promote)") }
                    outVRef.eval()
                    // Stub under test: single-dispatch production kernel (distinct from wrapper)
                    guard let outVGot = RawFusedVerify.gdnNormGateRows(
                        coreOut: coreOut, z: z, normWeight: normW,
                        M: M, Hv: Hv, Dv: Dv, eps: 1e-6, promoteF32: promote)
                    else { return (false, "not implemented (M=\(M) promote=\(promote))") }
                    outVGot.eval()
                    let (ok, d) = bitEqual(outVGot, outVRef.reshaped([M, valueDim]))
                    if !ok { return (false, "M=\(M) promote=\(promote): \(d)") }
                }
            }
            return (true, "ok")
        }

        // ── Wave 2 GDN fusion tests (40-41) ──────────────────────────────
        //
        // WRITE-LOCKED: implementer MUST NOT modify these tests.
        // They encode the G1 acceptance gate from notes/07-gdn-fusion-spec.md §3 Wave 2.
        //
        // Canonical GDN geometry: Hk=16, Dk=128, Hv=32, Dv=128.
        // References use ONLY existing production kernel wrappers (rmsNormRows,
        // computeGBetaRowsRaw) and pure MLX data-movement for slices — no CPU
        // reimplementations.  Tests are RED until Wave 2 implementation is complete.

        // ── Test 40 (F2): gdnPrepFused ───────────────────────────────────
        // Fused ⑧slice q ⑨slice k ⑩slice v ⑪rmsnorm qn ⑫rmsnorm kn
        //       ⑬scale_mul q ⑭scale_mul k ⑮compute_g_beta
        // ≡ 8-kernel reference chain.  5 outputs (qn/kn/v/g/beta) bit-exact.
        run("fuse_gdn_prep") {
            let numKHeads = 16, headKDim = 128, numVHeads = 32
            let keyDim   = numKHeads * headKDim    // 2048
            let valueDim = numVHeads * headKDim    // 4096  (headVDim == headKDim == 128)
            let convDim  = keyDim * 2 + valueDim   // 8192
            let invScale = Float(pow(Double(headKDim), -0.5))
            let eps: Float = 1e-6
            let onesQ = MLXArray.ones([headKDim]).asType(.float16); onesQ.eval()
            for M in [1, 8] {
                let convOut = MLXRandom.normal([M, convDim]).asType(.float16)
                let aP      = MLXRandom.normal([M, numVHeads]).asType(.float16)
                let bP      = MLXRandom.normal([M, numVHeads]).asType(.float16)
                let aLog    = MLXRandom.normal([numVHeads]).asType(.float32)
                let dtBias  = MLXRandom.normal([numVHeads]).asType(.float32)
                MLX.eval([convOut, aP, bP, aLog, dtBias])
                // ── Reference: 8-kernel chain ──
                // ⑧ slice q  ⑨ slice k  ⑩ slice v  (pure data movement)
                let q1 = convOut[0..., 0 ..< keyDim]
                    .asType(.float16).reshaped([M * numKHeads, headKDim])
                let k1 = convOut[0..., keyDim ..< 2 * keyDim]
                    .asType(.float16).reshaped([M * numKHeads, headKDim])
                let v1 = convOut[0..., 2 * keyDim ..< 2 * keyDim + valueDim].asType(.float16)
                MLX.eval([q1, k1, v1])
                // ⑪ rmsnorm qn (ones weight, per-head over headKDim)
                guard let qnNorm = RawMetalForward.rmsNormRows(
                        q1, onesQ, M: M * numKHeads, eps: eps, D: headKDim)
                else { return (false, "ref rmsNorm qn nil M=\(M)") }
                // ⑫ rmsnorm kn (ones weight, per-head over headKDim)
                guard let knNorm = RawMetalForward.rmsNormRows(
                        k1, onesQ, M: M * numKHeads, eps: eps, D: headKDim)
                else { return (false, "ref rmsNorm kn nil M=\(M)") }
                qnNorm.eval(); knNorm.eval()
                // ⑬⑭ scale_mul q/k — PRODUCTION Metal scale_mul kernel (encodeScaleMul).
                // NOT MLX f32-multiply: Metal x[i]=(half)s·x[i] (half·half mul) can differ
                // from f32-multiply by 1 f16 ULP on ~1/2048 elements when s is not exactly
                // representable in f16 (invScale = 1/sqrt(headKDim) is not exact in f16).
                let qkDim = M * numKHeads * headKDim
                guard let qnRef = scaleMulKernel(qnNorm, s: invScale * invScale, total: qkDim),
                      let knRef = scaleMulKernel(knNorm, s: invScale, total: qkDim)
                else { return (false, "scale_mul kernel failed M=\(M)") }
                qnRef.eval(); knRef.eval()
                // ⑮ compute_g_beta (per-op production wrapper)
                guard let (gRef, betaRef) = RawFusedVerify.computeGBetaRowsRaw(
                        aP, bP, aLog, dtBias, M: M, Hv: numVHeads)
                else { return (false, "ref computeGBeta nil M=\(M)") }
                gRef.eval(); betaRef.eval()
                // ── Fused stub ──
                guard let (qnGot, knGot, vGot, gGot, betaGot) = RawFusedVerify.gdnPrepFused(
                    convOut: convOut, aP: aP, bP: bP, aLog: aLog, dtBias: dtBias,
                    M: M, keyDim: keyDim, valueDim: valueDim,
                    numKHeads: numKHeads, headKDim: headKDim, numVHeads: numVHeads,
                    invScale: invScale, eps: eps)
                else { return (false, "not implemented (M=\(M))") }
                MLX.eval([qnGot, knGot, vGot, gGot, betaGot])
                // ── Bit-exact check: all 5 outputs ──
                let checks: [(String, MLXArray, MLXArray)] = [
                    ("qn",   qnGot,   qnRef),
                    ("kn",   knGot,   knRef),
                    ("v",    vGot,    v1),
                    ("g",    gGot,    gRef),
                    ("beta", betaGot, betaRef)]
                for (nm, got, ref) in checks {
                    let (ok, d) = bitEqual(got, ref)
                    if !ok { return (false, "M=\(M) \(nm): \(d)") }
                }
            }
            return (true, "ok")
        }

        // ── Test 41 (F5): gdnResidPostNormFused ──────────────────────────
        // Fused ⑳resid_add (hBuf += mixerOut) ㉑rmsnorm post (hBuf → postNorm)
        // ≡ resid_add then rmsNormRows reference.  Both outputs (h, postNorm) bit-exact.
        run("fuse_gdn_resid_postnorm") {
            let H = 2048, eps: Float = 1e-6
            for M in [1, 8] {
                let hBuf     = MLXRandom.normal([M, H]).asType(.float16)
                let mixerOut = MLXRandom.normal([M, H]).asType(.float16)
                let postW    = MLXRandom.normal([H]).asType(.float16)
                MLX.eval([hBuf, mixerOut, postW])
                // ── Reference: resid_add then rmsnorm ──
                // resid_add: (half)((float)h[i] + (float)r[i]) — matches Metal kernel semantics
                let hRef = (hBuf.asType(.float32) + mixerOut.asType(.float32)).asType(.float16)
                hRef.eval()
                guard let postNormRef = RawMetalForward.rmsNormRows(hRef, postW, M: M, eps: eps, D: H)
                else { return (false, "ref rmsNormRows nil M=\(M)") }
                postNormRef.eval()
                // ── Fused stub ──
                guard let (hGot, postNormGot) = RawFusedVerify.gdnResidPostNormFused(
                    hBuf: hBuf, mixerOut: mixerOut, postW: postW, M: M, H: H, eps: eps)
                else { return (false, "not implemented (M=\(M))") }
                MLX.eval([hGot, postNormGot])
                // ── Bit-exact check: both outputs ──
                let (ok1, d1) = bitEqual(hGot, hRef)
                if !ok1 { return (false, "h M=\(M): \(d1)") }
                let (ok2, d2) = bitEqual(postNormGot, postNormRef)
                if !ok2 { return (false, "postNorm M=\(M): \(d2)") }
            }
            return (true, "ok")
        }

        // ── Wave 3: attn + shared-expert fusion tests (42-46) ────────────
        //
        // WRITE-LOCKED: implementer (Haiku) MUST NOT modify these tests.
        // They encode the G1 acceptance gate from notes/08-wave3-attn-shexp-spec.md §3-§4.
        //
        // References use ONLY existing production kernel wrappers (qmmRows, rmsNormRows,
        // ropeRows, qmm8, finalCombineRowsRaw, swigluRaw) and pure data-movement MLX
        // ops (slice/reshape/transpose/concatenate — no arithmetic reimplementation).
        // All 5 tests are RED (FAIL "not implemented") until the stubs are implemented.
        //
        // A4 (sdpa+sigmoid_mul epilogue) is a stretch atom gated only by G2 (no unit stub).

        // Test 42 (A1): attn_qkv_demux_bitexact
        // attnQkvDemux ≡ 3 × qmmRows (q/k/v proj), bit-exact for all 3 outputs.
        // Demux N boundaries (all multiples of 8): q=8192, k=512, v=512.
        // Sweeps M∈{1,8}.
        run("attn_qkv_demux_bitexact") {
            let H = 2048, numHeads = 16, numKV = 2, headDim = 256
            let qd2 = 2 * headDim   // 512
            let Nq = numHeads * qd2, Nk = numKV * headDim, Nv = numKV * headDim
            func q4(_ n: Int, _ k: Int) -> (MLXArray, MLXArray, MLXArray) {
                let wf = MLXRandom.normal([n, k]).asType(.float16)
                let (q, s, b) = MLX.quantized(wf, groupSize: 64, bits: 4, mode: .affine)
                return (q, s, b!)
            }
            let (qW, qS, qB) = q4(Nq, H)
            let (kW, kS, kB) = q4(Nk, H)
            let (vW, vS, vB) = q4(Nv, H)
            MLX.eval([qW, qS, qB, kW, kS, kB, vW, vS, vB])
            for M in [1, 8] {
                let x = MLXRandom.normal([M, H]).asType(.float16); x.eval()
                // Reference: 3 separate qmmRows calls (existing production kernel)
                guard let qRef = RawMetalForward.qmmRows(x, qW, scales: qS, biases: qB, M: M, K: H, N: Nq),
                      let kRef = RawMetalForward.qmmRows(x, kW, scales: kS, biases: kB, M: M, K: H, N: Nk),
                      let vRef = RawMetalForward.qmmRows(x, vW, scales: vS, biases: vB, M: M, K: H, N: Nv)
                else { return (false, "ref qmmRows nil M=\(M)") }
                MLX.eval([qRef, kRef, vRef])
                // Stub under test
                guard let (qGot, kGot, vGot) = RawFusedVerify.attnQkvDemux(x,
                        qW: qW, qS: qS, qB: qB,
                        kW: kW, kS: kS, kB: kB,
                        vW: vW, vS: vS, vB: vB,
                        M: M, H: H, numHeads: numHeads, numKV: numKV, headDim: headDim)
                else { return (false, "not implemented (M=\(M))") }
                MLX.eval([qGot, kGot, vGot])
                for (nm, got, ref) in [("qOut", qGot, qRef), ("kOut", kGot, kRef), ("vOut", vGot, vRef)] {
                    let (ok, d) = bitEqual(got, ref)
                    if !ok { return (false, "M=\(M) \(nm): \(d)") }
                }
            }
            return (true, "ok")
        }

        // Test 43 (A2): attn_q_prep_fused_bitexact
        // attnQPrepFused ≡ extract(lower-headDim slice) + rmsNormRows + ropeRows, bit-exact.
        // extract is pure data movement (MLX slice); rmsNorm/rope use production kernels.
        // Uses startOffset=37 (non-zero) to exercise non-trivial rope angle computation.
        // Sweeps M∈{1,8}.
        run("attn_q_prep_fused_bitexact") {
            let numHeads = 16, headDim = 256, ropeDim = 64
            let qd2 = 2 * headDim   // 512
            let ropeBase: Float = 1e7, eps: Float = 1e-6
            let startOffset = 37   // non-zero: exercises rope angle
            let qNorm = MLXRandom.normal([headDim]).asType(.float16); qNorm.eval()
            for M in [1, 8] {
                // qOut[M*numHeads, qd2]: gate in upper headDim, query in lower headDim
                let qOut = MLXRandom.normal([M * numHeads, qd2]).asType(.float16); qOut.eval()
                // Reference chain:
                // ④ extract q: pure data movement — lower headDim slice of each q head
                let qX = qOut[0..., 0..<headDim].asType(.float16)   // [M*numHeads, headDim]
                qX.eval()
                // ⑤ rmsnorm q (per-head, weight qNorm[headDim])
                guard let qN = RawMetalForward.rmsNormRows(qX, qNorm,
                                                            M: M * numHeads, eps: eps, D: headDim)
                else { return (false, "ref rmsNormRows nil M=\(M)") }
                qN.eval()
                // ⑦ rope q (numHeads lanes, position = startOffset + m)
                guard let qRot = RawMetalForward.ropeRows(qN, headDim: headDim, ropeDim: ropeDim,
                                                           base: ropeBase, startOffset: startOffset,
                                                           M: M, numHeads: numHeads)
                else { return (false, "ref ropeRows nil M=\(M)") }
                qRot.eval()
                // Stub under test
                guard let qRotGot = RawFusedVerify.attnQPrepFused(qOut, qNorm: qNorm,
                        startOffset: startOffset, M: M,
                        numHeads: numHeads, headDim: headDim,
                        ropeDim: ropeDim, ropeBase: ropeBase, eps: eps)
                else { return (false, "not implemented (M=\(M))") }
                qRotGot.eval()
                let (ok, d) = bitEqual(qRotGot, qRot)
                if !ok { return (false, "M=\(M) qRot: \(d)") }
            }
            return (true, "ok")
        }

        // Test 44 (A3): attn_k_prep_fused_bitexact
        // attnKPrepFused ≡ rmsNormRows + ropeRows + write_kv cache scatter, both outputs bit-exact.
        // Verifies BOTH the rotated kRot output AND the kCache scatter contents after write.
        // Cache scatter: src[M*KV, D] → cache[KV, baseLen+M, D]; pure data movement
        //   (kRot[m*KV+h, :] → cache[h, baseLen+m, :]).
        // Uses startOffset=12 (non-zero). Sweeps M∈{1,8}.
        run("attn_k_prep_fused_bitexact") {
            let numKV = 2, headDim = 256, ropeDim = 64
            let ropeBase: Float = 1e7, eps: Float = 1e-6
            let baseLen = 12   // non-zero: exercises pos offset in rope + cache scatter
            let kNorm = MLXRandom.normal([headDim]).asType(.float16)
            let kCacheInit = MLXRandom.normal([numKV, baseLen, headDim]).asType(.float16)
            MLX.eval([kNorm, kCacheInit])
            for M in [1, 8] {
                let kOut = MLXRandom.normal([M * numKV, headDim]).asType(.float16); kOut.eval()
                // Reference chain:
                // ⑥ rmsnorm k (per-kv-head, weight kNorm[headDim])
                guard let kN = RawMetalForward.rmsNormRows(kOut, kNorm,
                                                            M: M * numKV, eps: eps, D: headDim)
                else { return (false, "ref rmsNormRows nil M=\(M)") }
                kN.eval()
                // ⑧ rope k (numHeads=numKV for k-path, position = baseLen + m)
                guard let kRot = RawMetalForward.ropeRows(kN, headDim: headDim, ropeDim: ropeDim,
                                                           base: ropeBase, startOffset: baseLen,
                                                           M: M, numHeads: numKV)
                else { return (false, "ref ropeRows nil M=\(M)") }
                kRot.eval()
                // ⑨ write_kv cache scatter: pure data movement — no arithmetic.
                // write_kv_rows: src[M*KV, D] → cache[KV, maxLen, D] at [pos..pos+M).
                // Memory: src[(m*KV+h)*D+dd] → cache[h*maxLen*D + (pos+m)*D + dd].
                // Equivalent reshape+transpose: kRot[M*KV,D] → [M,KV,D] → [KV,M,D].
                let kRotFC = kRot.reshaped([M, numKV, headDim])
                                  .transposed(1, 0, 2)   // [KV, M, headDim]
                kRotFC.eval()
                let kCacheRef = MLX.concatenated([kCacheInit, kRotFC], axis: 1)   // [KV, baseLen+M, D]
                kCacheRef.eval()
                // Stub under test
                let maxLen = baseLen + M + 4
                guard let (kRotGot, kCacheGot) = RawFusedVerify.attnKPrepFused(kOut, kNorm: kNorm,
                        kCacheInit: kCacheInit, startOffset: baseLen, maxLen: maxLen, M: M,
                        numKV: numKV, headDim: headDim,
                        ropeDim: ropeDim, ropeBase: ropeBase, eps: eps)
                else { return (false, "not implemented (M=\(M))") }
                kRotGot.eval(); kCacheGot.eval()
                let (ok1, d1) = bitEqual(kRotGot, kRot)
                if !ok1 { return (false, "M=\(M) kRot: \(d1)") }
                let (ok2, d2) = bitEqual(kCacheGot, kCacheRef)
                if !ok2 { return (false, "M=\(M) kCache: \(d2)") }
            }
            return (true, "ok")
        }

        // Test 45 (S1): shared_gu_swiglu_fused_bitexact
        // sharedGUSwigluFused ≡ qmmRows(shG) + qmmRows(shU) + swigluRaw, bit-exact.
        // Plain-qmm variant (no gather): x[M,H] → shAct[M,I].
        // M==1 only: fuseSHEXPActive predicate gate (register-pressure, same as fuseGU doctrine).
        run("shared_gu_swiglu_fused_bitexact") {
            let H = 2048, I = 512
            func q4(_ n: Int, _ k: Int) -> (MLXArray, MLXArray, MLXArray) {
                let wf = MLXRandom.normal([n, k]).asType(.float16)
                let (q, s, b) = MLX.quantized(wf, groupSize: 64, bits: 4, mode: .affine)
                return (q, s, b!)
            }
            let (shGW, shGS, shGB) = q4(I, H)
            let (shUW, shUS, shUB) = q4(I, H)
            MLX.eval([shGW, shGS, shGB, shUW, shUS, shUB])
            let M = 1   // M==1 branch only (S1 fuseSHEXPActive gate)
            let x = MLXRandom.normal([M, H]).asType(.float16); x.eval()
            // Reference: 3 existing production kernels (qmmRows×2 + swigluRaw)
            guard let sg = RawMetalForward.qmmRows(x, shGW, scales: shGS, biases: shGB,
                                                    M: M, K: H, N: I),
                  let su = RawMetalForward.qmmRows(x, shUW, scales: shUS, biases: shUB,
                                                    M: M, K: H, N: I)
            else { return (false, "ref qmmRows nil") }
            sg.eval(); su.eval()
            guard let shActRef = RawMetalForward.swigluRaw(sg, su)
            else { return (false, "ref swigluRaw nil") }
            shActRef.eval()
            // Stub under test
            guard let shActGot = RawFusedVerify.sharedGUSwigluFused(x,
                    shGW: shGW, shGS: shGS, shGB: shGB,
                    shUW: shUW, shUS: shUS, shUB: shUB,
                    M: M, H: H, I: I)
            else { return (false, "not implemented (M=\(M))") }
            shActGot.eval()
            return bitEqual(shActGot, shActRef)
        }

        // Test 46 (S2): shared_gate_combine_fused_bitexact
        // sharedGateCombineFused ≡ qmm8(x→sgl N=8) + finalCombineRowsRaw(y,sharedY,sgl),
        // bit-exact for both steps.
        // qmm8: 8-bit weights, group_size=64, N=8, K=H=2048 (K%512==0 ✓, N%8==0 ✓).
        // final_combine: out[i] = y[i] + stable_sigmoid_f16(sgl[m*8]) * sharedY[i].
        // Sweeps M∈{1,8}.
        run("shared_gate_combine_fused_bitexact") {
            let H = 2048
            func q8(_ n: Int, _ k: Int) -> (MLXArray, MLXArray, MLXArray) {
                let wf = MLXRandom.normal([n, k]).asType(.float16)
                let (q, s, b) = MLX.quantized(wf, groupSize: 64, bits: 8, mode: .affine)
                return (q, s, b!)
            }
            let (sgW, sgS, sgB) = q8(8, H)
            MLX.eval([sgW, sgS, sgB])
            for M in [1, 8] {
                let x       = MLXRandom.normal([M, H]).asType(.float16)
                let y       = MLXRandom.normal([M, H]).asType(.float16)
                let sharedY = MLXRandom.normal([M, H]).asType(.float16)
                MLX.eval([x, y, sharedY])
                // Reference: qmm8 then finalCombineRowsRaw (both production kernels)
                guard let sgl = RawMetalForward.qmm8(x, sgW, scales: sgS, biases: sgB,
                                                      M: M, K: H, N: 8)
                else { return (false, "ref qmm8 nil M=\(M)") }
                sgl.eval()
                guard let outRef = RawFusedVerify.finalCombineRowsRaw(y, sharedY, sgl, M: M, N: H)
                else { return (false, "ref finalCombineRowsRaw nil M=\(M)") }
                outRef.eval()
                // Stub under test
                guard let outGot = RawFusedVerify.sharedGateCombineFused(x, y: y, sharedY: sharedY,
                        sgW: sgW, sgS: sgS, sgB: sgB, M: M, H: H)
                else { return (false, "not implemented (M=\(M))") }
                outGot.eval()
                let (ok, d) = bitEqual(outGot, outRef)
                if !ok { return (false, "M=\(M): \(d)") }
            }
            return (true, "ok")
        }

        // ── Summary ───────────────────────────────────────────────────────
        return lines.joined(separator: "\n") + "\nRAWTESTS \(passed)/\(total)"
    }

    /// D1 perf probe: per-row(qmv-style)M-row kernel の重み再読コストの実測。
    /// 問い=「M 行が同じ weight を読む時、SLC/L2 がどれだけ吸収するか」。
    /// t(M)/t(1)≈M なら再読が丸コスト(→tiled ピボット検討)、≪M なら吸収(→このまま統合へ)。
    /// GPU 時間は 1 command buffer に reps 回 encode して gpuEnd-gpuStart/reps で計測(dispatch 償却)。
    public static func runPerfProbe() -> String {
        guard let (device, queue) = RawMetalForward.ensure() else { return "no device" }
        MLXRandom.seed(UInt64(7))
        var lines: [String] = []
        func gpuMs(_ reps: Int, _ encode: (MTLComputeCommandEncoder) -> Void) -> Double {
            let cb = queue.makeCommandBuffer()!
            let enc = cb.makeComputeCommandEncoder()!
            for _ in 0 ..< reps { encode(enc) }
            enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
            return (cb.gpuEndTime - cb.gpuStartTime) * 1000.0 / Double(reps)
        }

        // ── A: dense qmm4 (qmv per-row) ──
        for N in [2048, 8192] {
            let K = 2048
            let wf = MLXRandom.normal([N, K]).asType(.float16)
            let (wq, sc, biOpt) = MLX.quantized(wf, groupSize: 64, bits: 4, mode: .affine)
            guard let bi = biOpt else { return "bi nil" }
            let xAll = MLXRandom.normal([25, K]).asType(.float16)
            MLX.eval([wq, sc, bi, xAll])
            // warm compile
            _ = RawMetalForward.qmm(xAll[0 ..< 1], wq, scales: sc, biases: bi, M: 1, K: K, N: N)
            guard let bx = xAll.asType(.float16).asMTLBuffer(device: device, noCopy: false),
                  let bwq = wq.asMTLBuffer(device: device, noCopy: false),
                  let bsc = sc.asType(.float16).asMTLBuffer(device: device, noCopy: false),
                  let bbi = bi.asType(.float16).asMTLBuffer(device: device, noCopy: false) else { return "buf nil" }
            let outBuf = device.makeBuffer(length: 25 * N * 2, options: .storageModeShared)!
            var t1 = 0.0
            var row = "[raw-perf] qmm4 N=\(N) K=\(K):"
            for M in [1, 5, 9, 13, 17, 25] {
                let ms = gpuMs(50) { enc in
                    enc.setComputePipelineState(RawMetalForward._qmmPipeline!)
                    enc.setBuffer(bwq, offset: 0, index: 0); enc.setBuffer(bsc, offset: 0, index: 1)
                    enc.setBuffer(bbi, offset: 0, index: 2); enc.setBuffer(bx, offset: 0, index: 3)
                    enc.setBuffer(outBuf, offset: 0, index: 4)
                    var kk = Int32(K), nn = Int32(N)
                    enc.setBytes(&kk, length: 4, index: 5); enc.setBytes(&nn, length: 4, index: 6)
                    RawMetalForward.bindStop(enc, 16)
                    enc.dispatchThreadgroups(MTLSize(width: M, height: N / 8, depth: 1),
                                             threadsPerThreadgroup: MTLSize(width: 32, height: 2, depth: 1))
                }
                if M == 1 { t1 = ms }
                row += String(format: " M%d=%.3fms(x%.1f)", M, ms, ms / t1)
            }
            lines.append(row)
            // MLX 参照(壁時計, warm): batched quantizedMatmul
            var mlxRow = "[raw-perf] MLX qmm N=\(N):"
            for M in [1, 9, 17, 25] {
                let xm = xAll[0 ..< M]
                let warm = MLX.quantizedMatmul(xm, wq, scales: sc, biases: bi, transpose: true, groupSize: 64, bits: 4, mode: .affine); warm.eval()
                let t0 = DispatchTime.now().uptimeNanoseconds
                for _ in 0 ..< 30 {
                    let y = MLX.quantizedMatmul(xm, wq, scales: sc, biases: bi, transpose: true, groupSize: 64, bits: 4, mode: .affine)
                    y.eval()
                }
                let ms = Double(DispatchTime.now().uptimeNanoseconds - t0) / 30e6
                mlxRow += String(format: " M%d=%.3fms", M, ms)
            }
            lines.append(mlxRow + "  (wall, graph込み)")
        }

        // ── B: gather gqmm4_rows(MoE 形状, per-(row,expert)再読の worst 側)──
        do {
            let E = 64, K = 2048, N = 512, Ktop = 8
            let wf = MLXRandom.normal([E, N, K]).asType(.float16)
            let (wq, sc, biOpt) = MLX.quantized(wf, groupSize: 64, bits: 4, mode: .affine)
            guard let bi = biOpt else { return "bi nil" }
            let xAll = MLXRandom.normal([25, K]).asType(.float16)
            let pool: [Int32] = (0 ..< 200).map { Int32(($0 * 13) % E) }   // 行ごとに異なる expert 集合(重なりは現実的に発生)
            let indsAll = MLXArray((0 ..< 25 * Ktop).map { pool[$0 % pool.count] }, [25 * Ktop])
            MLX.eval([wq, sc, bi, xAll, indsAll])
            _ = RawMetalForward.gatherQmmRows(xAll[0 ..< 1], wq, scales: sc, biases: bi,
                                              inds: indsAll[0 ..< Ktop], M: 1, Ktop: Ktop, K: K, N: N)   // warm compile
            guard let bx = xAll.asType(.float16).asMTLBuffer(device: device, noCopy: false),
                  let bwq = wq.asMTLBuffer(device: device, noCopy: false),
                  let bsc = sc.asType(.float16).asMTLBuffer(device: device, noCopy: false),
                  let bbi = bi.asType(.float16).asMTLBuffer(device: device, noCopy: false),
                  let bin = indsAll.asType(.int32).asMTLBuffer(device: device, noCopy: false) else { return "buf nil" }
            let outBuf = device.makeBuffer(length: 25 * Ktop * N * 2, options: .storageModeShared)!
            var t1 = 0.0
            var row = "[raw-perf] gqmm4_rows E=\(E) N=\(N) Ktop=\(Ktop):"
            for M in [1, 5, 9, 13, 17, 25] {
                let ms = gpuMs(50) { enc in
                    enc.setComputePipelineState(RawMetalForward._gqmmRowsPipeline!)
                    enc.setBuffer(bwq, offset: 0, index: 0); enc.setBuffer(bsc, offset: 0, index: 1)
                    enc.setBuffer(bbi, offset: 0, index: 2); enc.setBuffer(bx, offset: 0, index: 3)
                    enc.setBuffer(bin, offset: 0, index: 4); enc.setBuffer(outBuf, offset: 0, index: 5)
                    var kk = Int32(K), nn = Int32(N), kt = Int32(Ktop)
                    enc.setBytes(&kk, length: 4, index: 6); enc.setBytes(&nn, length: 4, index: 7)
                    enc.setBytes(&kt, length: 4, index: 8)
                    RawMetalForward.bindStop(enc, 9)
                    enc.dispatchThreadgroups(MTLSize(width: 1, height: N / 8, depth: M * Ktop),
                                             threadsPerThreadgroup: MTLSize(width: 32, height: 2, depth: 1))
                }
                if M == 1 { t1 = ms }
                row += String(format: " M%d=%.3fms(x%.1f)", M, ms, ms / t1)
            }
            lines.append(row)
        }
        lines.append("[raw-perf] 判定基準: x(M)≪M なら SLC 吸収=per-row 方式続行 / x(M)≈M なら tiled ピボット検討")
        return lines.joined(separator: "\n")
    }
}
