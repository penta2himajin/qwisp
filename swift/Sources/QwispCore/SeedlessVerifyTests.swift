import Foundation
import MLX
import MLXFast
import MLXRandom
import Metal

/// D1 TDD RED phase — M-row (batched, order-stable) kernel scaffolding.
///
/// All 7 kernel-level tests FAIL initially because the D1 stub APIs in
/// SeedlessMetalForward return nil.  Two baseline tests PASS to confirm the
/// harness itself is correct (existing M=1 kernels are deterministic).
///
/// Run:  QWISP_RUN=raw-tests ./qwisp-poc stream
///   or: qwisp/test_raw.sh
public enum SeedlessVerifyTests {

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
        let total = 88

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
            guard let (device, queue) = SeedlessMetalForward.ensure(),
                  SeedlessMetalForward.ensureAuxPipelines() else { return nil }
            let f16 = input.reshaped([-1]).asType(.float16); f16.eval()
            let arr = f16.asArray(Float16.self)
            guard let buf = arr.withUnsafeBytes({ ptr in
                device.makeBuffer(bytes: ptr.baseAddress!, length: total * 2, options: .storageModeShared)
            }) else { return nil }
            let cb = queue.makeCommandBuffer()!
            let enc = cb.makeComputeCommandEncoder()!
            SeedlessFusedVerify.encodeScaleMul(enc, x: buf, s: s, total: total)
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
            guard let a = SeedlessMetalForward.qmm(x, wq, scales: sc, biases: bi, M: 1, K: K, N: N),
                  let b = SeedlessMetalForward.qmm(x, wq, scales: sc, biases: bi, M: 1, K: K, N: N)
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
            guard let a = SeedlessMetalForward.gatherQmm(x, wq, scales: sc, biases: bi,
                                                     inds: inds, Ktop: Ktop, K: K, N: N),
                  let b = SeedlessMetalForward.gatherQmm(x, wq, scales: sc, biases: bi,
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
                        guard let r = SeedlessMetalForward.qmm(xm, wq, scales: sc, biases: bi,
                                                           M: 1, K: K, N: N)
                        else { return (false, "ref qmm nil N=\(N) M=\(M) m=\(m)") }
                        r.eval(); refParts.append(r)
                    }
                    let ref = MLX.concatenated(refParts, axis: 0); ref.eval()   // [M, N]
                    // Stub
                    guard let got = SeedlessMetalForward.qmmRows(x, wq, scales: sc, biases: bi,
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
                    guard let r = SeedlessMetalForward.gatherQmm(xm, wq, scales: sc, biases: bi,
                                                             inds: rowInds, Ktop: Ktop, K: K, N: N)
                    else { return (false, "ref gatherQmm nil M=\(M) m=\(m)") }
                    r.eval(); refParts.append(r)   // [Ktop, N]
                }
                let ref = MLX.concatenated(refParts, axis: 0); ref.eval()   // [M*Ktop, N]
                // Stub: inds[M*Ktop] row-major, x[M,K]
                guard let got = SeedlessMetalForward.gatherQmmRows(x, wq, scales: sc, biases: bi,
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
                    guard let r = SeedlessMetalForward.sdpaDecode(qm, km, vm,
                                                              H: H, KV: KV, D: D, S: S, scale: scale)
                    else { return (false, "ref sdpaDecode nil M=\(M) m=\(m) S=\(S)") }
                    r.eval(); refParts.append(r)   // [H, D]
                }
                let ref = MLX.concatenated(refParts, axis: 0); ref.eval()   // [M*H, D]
                // Stub: q[M*H,D], kFull[KV,totalSeq,D], vFull[KV,totalSeq,D]
                guard let got = SeedlessMetalForward.sdpaRows(q, kFull, vFull,
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
                    guard let r = SeedlessMetalForward.conv1dSilu(window, w, K: K, C: C)
                    else { return (false, "ref conv1dSilu nil M=\(M) m=\(m)") }
                    r.eval(); refParts.append(r.reshaped([1, C]))   // normalise [1,1,C]→[1,C]
                }
                let ref = MLX.concatenated(refParts, axis: 0); ref.eval()   // [M, C]
                // Build windows[M, K, C] by stacking individual windows
                let windowsArr = (0..<M).map { buf[$0 ..< $0+K].reshaped([1, K, C]) }
                let windows = MLX.concatenated(windowsArr, axis: 0); windows.eval()   // [M, K, C]
                // Stub
                guard let got = SeedlessMetalForward.conv1dSiluRows(windows, w, M: M, K: K, C: C)
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
                    guard let (ym, ns) = SeedlessMetalForward.recurrent(
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
                guard let (gotY, gotState) = SeedlessMetalForward.gatedDeltaStepRows(
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
                    guard let r = SeedlessMetalForward.rope(xm, headDim: HD, ropeDim: rd,
                                                        base: base, offset: startOffset + m)
                    else { return (false, "ref rope nil M=\(M) m=\(m)") }
                    r.eval(); refParts.append(r)   // [numHeads, HD]
                }
                let ref = MLX.concatenated(refParts, axis: 0); ref.eval()   // [M*numHeads, HD]
                // Stub: x[M*numHeads, HD], groups of numHeads share position startOffset+m
                guard let got = SeedlessMetalForward.ropeRows(x, headDim: HD, ropeDim: rd,
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
                    guard let r = SeedlessMetalForward.rmsNorm(xm, wt, eps: 1e-6, D: D)
                    else { return (false, "ref rmsNorm nil M=\(M) m=\(m)") }
                    r.eval(); refParts.append(r)   // [1, D]
                }
                let ref = MLX.concatenated(refParts, axis: 0); ref.eval()   // [M, D]
                // Stub: x[M, D] → y[M, D]
                guard let got = SeedlessMetalForward.rmsNormRows(x, wt, M: M, eps: 1e-6, D: D)
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
                guard let out = SeedlessMetalForward.qmmTiled(xM, wq, scales: sc, biases: bi, M: M, K: K, N: N)
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
                guard let outM = SeedlessMetalForward.qmmTiled(xM, wq, scales: sc, biases: bi, M: M, K: K, N: N)
                else { return (false, "qmmTiled(M=\(M)) nil in concat-check") }
                outM.eval()
                var refParts: [MLXArray] = []
                for m in 0..<M {
                    let x1 = xAll[m..<m+1]   // [1, K]
                    guard let r = SeedlessMetalForward.qmmTiled(x1, wq, scales: sc, biases: bi, M: 1, K: K, N: N)
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
            guard let a = SeedlessMetalForward.qmmTiled(x, wq, scales: sc, biases: bi, M: M, K: K, N: N),
                  let b = SeedlessMetalForward.qmmTiled(x, wq, scales: sc, biases: bi, M: M, K: K, N: N)
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
            let w = SeedlessVerifyForward.AttnLayerW(qWq: qW, qSc: qS, qBi: qB, kWq: kW, kSc: kS, kBi: kB,
                                                vWq: vW, vSc: vS, vBi: vB, oWq: oW, oSc: oS, oBi: oB,
                                                qNorm: qN, kNorm: kN)
            let kC0 = MLXRandom.normal([numKV, baseLen, headDim]).asType(.float16)
            let vC0 = MLXRandom.normal([numKV, baseLen, headDim]).asType(.float16)
            MLX.eval([kC0, vC0])
            for M in [1, 2, 9, 17] {
                let x = MLXRandom.normal([M, H]).asType(.float16); x.eval()
                var k1 = kC0, v1 = vC0
                guard let got = SeedlessVerifyForward.attnLayerRows(x, w, kCache: &k1, vCache: &v1, M: M)
                else { return (false, "rows nil M=\(M)") }
                got.eval()
                var k2 = kC0, v2 = vC0
                var refParts: [MLXArray] = []
                for m in 0..<M {
                    guard let r = SeedlessVerifyForward.attnLayerRows(x[m ..< m+1], w, kCache: &k2, vCache: &v2, M: 1)
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
            let w = SeedlessVerifyForward.GDNLayerW(qkvWq: qkvW, qkvSc: qkvS, qkvBi: qkvB,
                                               zWq: zW, zSc: zS, zBi: zB, bWq: bW, bSc: bS, bBi: bB,
                                               aWq: aW, aSc: aS, aBi: aB, outWq: oW, outSc: oS, outBi: oB,
                                               conv1dW: convW, normWeight: normW, aLog: aLog, dtBias: dtB)
            let cs0 = MLXRandom.normal([cK - 1, convDim]).asType(.float16)
            let rs0 = MLXRandom.normal([1, Hv, Dv, Dk]).asType(.float32)
            MLX.eval([cs0, rs0, convW, normW, aLog, dtB])
            for M in [1, 2, 9, 17] {
                let x = MLXRandom.normal([M, H]).asType(.float16); x.eval()
                var c1 = cs0, r1 = rs0
                guard let got = SeedlessVerifyForward.gdnLayerRows(x, w, convState: &c1, recState: &r1, M: M)
                else { return (false, "rows nil M=\(M)") }
                got.eval()
                var c2 = cs0, r2 = rs0
                var refParts: [MLXArray] = []
                for m in 0..<M {
                    guard let r = SeedlessVerifyForward.gdnLayerRows(x[m ..< m+1], w, convState: &c2, recState: &r2, M: 1)
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
            let w = SeedlessVerifyForward.MoEBlockW(gateWq: gW, gateSc: gS, gateBi: gB,
                swGWq: swG0, swGSc: swG1, swGBi: swG2, swUWq: swU0, swUSc: swU1, swUBi: swU2,
                swDWq: swD0, swDSc: swD1, swDBi: swD2, shGWq: shG0, shGSc: shG1, shGBi: shG2,
                shUWq: shU0, shUSc: shU1, shUBi: shU2, shDWq: shD0, shDSc: shD1, shDBi: shD2,
                sharedGateWq: sgW, sharedGateSc: sgS, sharedGateBi: sgB)
            for M in [1, 2, 9, 17] {
                let x = MLXRandom.normal([M, H]).asType(.float16); x.eval()
                guard let got = SeedlessVerifyForward.moeBlockRows(x, w, M: M, E: E, I: I, Ktop: Ktop)
                else { return (false, "rows nil M=\(M)") }
                got.eval()
                var refParts: [MLXArray] = []
                for m in 0..<M {
                    guard let r = SeedlessVerifyForward.moeBlockRows(x[m ..< m+1], w, M: 1, E: E, I: I, Ktop: Ktop)
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
            func mkMoE() -> SeedlessVerifyForward.MoEBlockW {
                let (gW, gS, gB) = q8(E, H); let (sgW, sgS, sgB) = q8(8, H)
                let (a0, a1, a2) = q4e(E, I, H); let (b0, b1, b2) = q4e(E, I, H); let (c0, c1, c2) = q4e(E, H, I)
                let (d0, d1, d2) = q4(I, H); let (e0, e1, e2) = q4(I, H); let (f0, f1, f2) = q4(H, I)
                return SeedlessVerifyForward.MoEBlockW(gateWq: gW, gateSc: gS, gateBi: gB,
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
            let gdnW = SeedlessVerifyForward.GDNLayerW(qkvWq: qkvW, qkvSc: qkvS, qkvBi: qkvB,
                zWq: zW, zSc: zS, zBi: zB, bWq: bW, bSc: bS, bBi: bB, aWq: aW, aSc: aS, aBi: aB,
                outWq: oW, outSc: oS, outBi: oB,
                conv1dW: MLXRandom.normal([convDim, cK]).asType(.float16),
                normWeight: MLXRandom.normal([Dv]).asType(.float16),
                aLog: MLXRandom.normal([Hv]).asType(.float32), dtBias: MLXRandom.normal([Hv]).asType(.float32))
            // layer 1: attn
            let nH = 16, nKV = 2, hD = 256
            let (aqW, aqS, aqB) = q4(nH * 2 * hD, H); let (akW, akS, akB) = q4(nKV * hD, H)
            let (avW, avS, avB) = q4(nKV * hD, H); let (aoW, aoS, aoB) = q4(H, nH * hD)
            let attnW = SeedlessVerifyForward.AttnLayerW(qWq: aqW, qSc: aqS, qBi: aqB, kWq: akW, kSc: akS, kBi: akB,
                vWq: avW, vSc: avS, vBi: avB, oWq: aoW, oSc: aoS, oBi: aoB,
                qNorm: MLXRandom.normal([hD]).asType(.float16), kNorm: MLXRandom.normal([hD]).asType(.float16))
            let layers = [
                SeedlessVerifyForward.LayerSpec(isLinear: true,
                    inputLN: MLXRandom.normal([H]).asType(.float16), postLN: MLXRandom.normal([H]).asType(.float16),
                    gdn: gdnW, attn: nil, moe: mkMoE(), moeE: E, moeI: I),
                SeedlessVerifyForward.LayerSpec(isLinear: false,
                    inputLN: MLXRandom.normal([H]).asType(.float16), postLN: MLXRandom.normal([H]).asType(.float16),
                    gdn: nil, attn: attnW, moe: mkMoE(), moeE: E, moeI: I),
            ]
            let cs0 = MLXRandom.normal([cK - 1, convDim]).asType(.float16)
            let rs0 = MLXRandom.normal([1, Hv, Dv, Dk]).asType(.float32)
            let kC0 = MLXRandom.normal([nKV, 16, hD]).asType(.float16)
            let vC0 = MLXRandom.normal([nKV, 16, hD]).asType(.float16)
            MLX.eval([cs0, rs0, kC0, vC0])
            func freshCaches() -> [SeedlessVerifyForward.LayerCaches] {
                [SeedlessVerifyForward.LayerCaches(convState: cs0, recState: rs0),
                 SeedlessVerifyForward.LayerCaches(kCache: kC0, vCache: vC0)]
            }
            for M in [1, 2, 9, 17] {
                let x = MLXRandom.normal([M, H]).asType(.float16); x.eval()
                let c1 = freshCaches()
                guard let got = SeedlessVerifyForward.verifyForwardRows(x, layers: layers, caches: c1, M: M)
                else { return (false, "rows nil M=\(M)") }
                let c2 = freshCaches()
                var refParts: [MLXArray] = []
                for m in 0..<M {
                    guard let r = SeedlessVerifyForward.verifyForwardRows(x[m ..< m+1], layers: layers, caches: c2, M: 1)
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
                guard let mid = SeedlessMetalForward.qmmRows(x, w1.0, scales: w1.1, biases: w1.2, M: M, K: K, N: N1),
                      let ref = SeedlessMetalForward.qmmRows(mid, w2.0, scales: w2.1, biases: w2.2, M: M, K: N1, N: N2)
                else { return (false, "ref nil M=\(M)") }
                ref.eval()
                // fused: 単一CB + 常駐 midBuf
                guard let got = SeedlessFusedVerify.fusedTwoQmm(x, w1: w1, N1: N1, w2: w2, N2: N2, M: M, K: K)
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
                guard let (gi, gs) = SeedlessFusedVerify.routeTop8Rows(logits, M: M, N: N, K: K)
                else { return (false, "rows nil M=\(M)") }
                gi.eval(); gs.eval()
                var iParts: [MLXArray] = [], sParts: [MLXArray] = []
                for m in 0..<M {
                    guard let (ri, rs) = SeedlessFusedVerify.routeTop8Rows(logits[m ..< m+1], M: 1, N: N, K: K)
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


        // Test 63 (G-A.1 — 案B Stage 1): route_bias eps=0 恒等 + all-resident 等シフト恒等。
        // Reference = 本番 routeTop8Rows(CPU 再実装 oracle なし)。
        //  Part A: eps=0 なら mask 不問で bias kernel ≡ routeTop8Rows(inds+scores bit 一致)。
        //  Part B(adversarial): all-resident mask + eps>0 は全 logit 等シフト=選択不変
        //          → inds bit 一致・scores bit 一致(bias が score に漏れない & 順序不変)。
        run("route_bias_eps0_identity") {
            let N = 256, K = 8
            for M in [1, 2] {
                // well-separated logits so a uniform +eps never collides two selected values
                let logits = (MLXRandom.normal([M, N]).asType(.float16) * MLXArray(Float16(4)))
                    .asType(.float16); logits.eval()
                guard let (refI, refS) = SeedlessFusedVerify.routeTop8Rows(logits, M: M, N: N, K: K)
                else { return (false, "ref routeTop8Rows nil M=\(M)") }
                refI.eval(); refS.eval()

                // Part A: eps=0, several masks (all-cold, pattern) → must equal unbiased exactly.
                let masks: [[Int32]] = [
                    [Int32](repeating: 0, count: N),
                    (0..<N).map { Int32($0 % 3 == 0 ? 1 : 0) },
                ]
                for mask in masks {
                    guard let (gi, gs) = SeedlessFusedVerify.routeTop8RowsBias(
                        logits, residentMask: mask, eps: 0, M: M, N: N, K: K)
                    else { return (false, "not implemented (eps0 M=\(M))") }
                    gi.eval(); gs.eval()
                    let (oki, di) = bitEqual(gi.asType(.float32), refI.asType(.float32))
                    if !oki { return (false, "eps0 inds M=\(M): \(di)") }
                    let (oks, ds) = bitEqual(gs, refS)
                    if !oks { return (false, "eps0 scores M=\(M): \(ds)") }
                }

                // Part B: all-resident + eps>0 → uniform shift → selection & scores unchanged.
                let allRes = [Int32](repeating: 1, count: N)
                guard let (bi, bs) = SeedlessFusedVerify.routeTop8RowsBias(
                    logits, residentMask: allRes, eps: 0.5, M: M, N: N, K: K)
                else { return (false, "not implemented (allres M=\(M))") }
                bi.eval(); bs.eval()
                let (oki, di) = bitEqual(bi.asType(.float32), refI.asType(.float32))
                if !oki { return (false, "allres inds M=\(M): \(di)") }
                let (oks, ds) = bitEqual(bs, refS)
                if !oks { return (false, "allres scores M=\(M): \(ds)") }
            }
            return (true, "ok")
        }


        // Test 64 (G-A.2 — 案B Stage 1): near-tie flip。手作り合成で resident R が cold C を
        // margin だけ下回る(top-8 集合は eps に依らず固定=R,C,と 6 個の中位 expert)。
        //  margin < eps → 選択が flip(R が rank0 に、集合は不変)。scores は「選ばれた expert 自身の
        //     unbiased gate 値」= bias が score に漏れない(集合不変ゆえ renorm 分母も不変で per-id bit 一致)。
        //  margin > eps → 選択不変(inds+scores bit 一致)。
        // Reference = 本番 routeTop8Rows(unbiased)。CPU 再実装 oracle なし。
        run("route_bias_neartie_flip") {
            let N = 256, K = 8
            let R = 100, C = 50                 // resident / cold near-tie pair
            let others = [10, 20, 30, 40, 60, 70]   // 6 mid experts, guaranteed in top-8
            // logits: C=5.2 (highest, cold), R=5.0 (resident), others=3.0, rest=0.
            // f16(5.2)≈5.19921875 → effective margin ≈ 0.199.
            var lg = [Float](repeating: 0, count: N)
            lg[C] = 5.2; lg[R] = 5.0
            for e in others { lg[e] = 3.0 }
            let logits = MLXArray(lg, [1, N]).asType(.float16); logits.eval()
            var mask = [Int32](repeating: 0, count: N); mask[R] = 1   // only R resident

            guard let (uI, uS) = SeedlessFusedVerify.routeTop8Rows(logits, M: 1, N: N, K: K)
            else { return (false, "ref routeTop8Rows nil") }
            uI.eval(); uS.eval()
            let ui = uI.asType(.int32).asArray(Int32.self)
            let us = uS.asType(.float32).asArray(Float.self)
            // sanity on the reference: unbiased top-1 is the cold expert C
            if Int(ui[0]) != C { return (false, "ref top-1 expected C=\(C), got \(ui[0])") }

            // flip: eps=0.5 > margin(≈0.199) → R promoted to rank 0, set unchanged.
            guard let (fI, fS) = SeedlessFusedVerify.routeTop8RowsBias(
                logits, residentMask: mask, eps: 0.5, M: 1, N: N, K: K)
            else { return (false, "not implemented (flip)") }
            fI.eval(); fS.eval()
            let fi = fI.asType(.int32).asArray(Int32.self)
            let fs = fS.asType(.float32).asArray(Float.self)
            if Int(fi[0]) != R { return (false, "flip: rank0 expected R=\(R), got \(fi[0])") }
            if Set(fi) != Set(ui) { return (false, "flip: top-8 set changed \(fi) vs \(ui)") }
            // score non-contamination: each expert keeps its unbiased gate value (match by id).
            for (bIdx, e) in fi.enumerated() {
                guard let uIdx = ui.firstIndex(of: e) else { return (false, "flip: id \(e) not in ref") }
                if fs[bIdx] != us[uIdx] {
                    return (false, "flip: score leak for e=\(e): biased \(fs[bIdx]) vs unbiased \(us[uIdx])")
                }
            }

            // no-flip: eps=0.1 < margin(≈0.199) → selection identical to unbiased (inds+scores bit-exact).
            guard let (nI, nS) = SeedlessFusedVerify.routeTop8RowsBias(
                logits, residentMask: mask, eps: 0.1, M: 1, N: N, K: K)
            else { return (false, "not implemented (noflip)") }
            nI.eval(); nS.eval()
            let (oki, di) = bitEqual(nI.asType(.float32), uI.asType(.float32))
            if !oki { return (false, "noflip inds: \(di)") }
            let (oks, ds) = bitEqual(nS, uS)
            if !oks { return (false, "noflip scores: \(ds)") }
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
            let w = SeedlessVerifyForward.MoEBlockW(gateWq: gW, gateSc: gS, gateBi: gB,
                swGWq: a0, swGSc: a1, swGBi: a2, swUWq: b0, swUSc: b1, swUBi: b2,
                swDWq: c0, swDSc: c1, swDBi: c2, shGWq: d0, shGSc: d1, shGBi: d2,
                shUWq: e0, shUSc: e1, shUBi: e2, shDWq: f0, shDSc: f1, shDBi: f2,
                sharedGateWq: sgW, sharedGateSc: sgS, sharedGateBi: sgB)
            for M in [1, 2, 9, 17] {
                let x = MLXRandom.normal([M, H]).asType(.float16); x.eval()
                guard let got = SeedlessVerifyForward.moeBlockRows(x, w, M: M, E: E, I: I, Ktop: Ktop, metalRoute: true)
                else { return (false, "rows nil M=\(M)") }
                got.eval()
                var refParts: [MLXArray] = []
                for m in 0..<M {
                    guard let r = SeedlessVerifyForward.moeBlockRows(x[m ..< m+1], w, M: 1, E: E, I: I, Ktop: Ktop, metalRoute: true)
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
                guard let mid = SeedlessMetalForward.gatherQmmRows(x, w1.0, scales: w1.1, biases: w1.2,
                                                             inds: inds, M: M, Ktop: Ktop, K: K, N: I),
                      let ref = SeedlessMetalForward.gatherQmmRows(mid, w2.0, scales: w2.1, biases: w2.2,
                                                             inds: inds, M: M, Ktop: Ktop, K: I, N: K2, lhsPerExpert: true)
                else { return (false, "ref nil M=\(M)") }
                ref.eval()
                guard let got = SeedlessFusedVerify.fusedGatherChain(x, inds: inds, w1: w1, I: I, w2: w2, K2: K2,
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
            let w = SeedlessVerifyForward.MoEBlockW(gateWq: gW, gateSc: gS, gateBi: gB,
                swGWq: a0, swGSc: a1, swGBi: a2, swUWq: b0, swUSc: b1, swUBi: b2,
                swDWq: c0, swDSc: c1, swDBi: c2, shGWq: d0, shGSc: d1, shGBi: d2,
                shUWq: e0, shUSc: e1, shUBi: e2, shDWq: f0, shDSc: f1, shDBi: f2,
                sharedGateWq: sgW, sharedGateSc: sgS, sharedGateBi: sgB)
            for M in [1, 2, 9, 17] {
                let x = MLXRandom.normal([M, H]).asType(.float16); x.eval()
                guard let ref = SeedlessVerifyForward.moeBlockRows(x, w, M: M, E: E, I: I, Ktop: Ktop, metalRoute: true)
                else { return (false, "composed nil M=\(M)") }
                ref.eval()
                guard let got = SeedlessFusedVerify.fusedMoEBlockRows(x, w, M: M, E: E, I: I, Ktop: Ktop)
                else { return (false, "fused nil M=\(M)") }
                got.eval()
                let (ok, d) = bitEqual(got, ref)
                if !ok {
                    // 段階バイセクト: composed per-op 中間 vs fused dump で最初の乖離段を特定
                    guard let dump = SeedlessFusedVerify.fusedMoEBlockRowsDump(x, w, M: M, E: E, I: I, Ktop: Ktop),
                          let cgl = SeedlessMetalForward.qmm8(x, w.gateWq, scales: w.gateSc, biases: w.gateBi, M: M, K: H, N: E),
                          let (ci, cs) = SeedlessFusedVerify.routeTop8Rows(cgl, M: M, N: E, K: Ktop)
                    else { return (false, "M=\(M): \(d) (dump nil)") }
                    let cif = ci.reshaped([M * Ktop]).asType(.int32); cif.eval()
                    guard let cg = SeedlessMetalForward.gatherQmmRows(x, w.swGWq, scales: w.swGSc, biases: w.swGBi,
                                                                 inds: cif, M: M, Ktop: Ktop, K: H, N: I),
                          let cu = SeedlessMetalForward.gatherQmmRows(x, w.swUWq, scales: w.swUSc, biases: w.swUBi,
                                                                 inds: cif, M: M, Ktop: Ktop, K: H, N: I),
                          let ch = SeedlessMetalForward.swigluRaw(cg, cu),
                          let cd = SeedlessMetalForward.gatherQmmRows(ch, w.swDWq, scales: w.swDSc, biases: w.swDBi,
                                                                 inds: cif, M: M, Ktop: Ktop, K: I, N: H, lhsPerExpert: true),
                          let csg = SeedlessMetalForward.qmmRows(x, w.shGWq, scales: w.shGSc, biases: w.shGBi, M: M, K: H, N: I),
                          let csu = SeedlessMetalForward.qmmRows(x, w.shUWq, scales: w.shUSc, biases: w.shUBi, M: M, K: H, N: I),
                          let cshAct = SeedlessMetalForward.swigluRaw(csg, csu),
                          let csharedY = SeedlessMetalForward.qmmRows(cshAct, w.shDWq, scales: w.shDSc, biases: w.shDBi, M: M, K: I, N: H),
                          let csgl = SeedlessMetalForward.qmm8(x, w.sharedGateWq, scales: w.sharedGateSc, biases: w.sharedGateBi, M: M, K: H, N: 8)
                    else { return (false, "M=\(M): \(d) (composed stage nil)") }
                    guard let cy = SeedlessFusedVerify.combineRowsRaw(cd, cs, M: M, Ktop: Ktop, N: H)
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
            let w = SeedlessVerifyForward.AttnLayerW(qWq: aqW, qSc: aqS, qBi: aqB, kWq: akW, kSc: akS, kBi: akB,
                vWq: avW, vSc: avS, vBi: avB, oWq: aoW, oSc: aoS, oBi: aoB,
                qNorm: MLXRandom.normal([hD]).asType(.float16), kNorm: MLXRandom.normal([hD]).asType(.float16))
            let kC0 = MLXRandom.normal([nKV, 16, hD]).asType(.float16)
            let vC0 = MLXRandom.normal([nKV, 16, hD]).asType(.float16)
            MLX.eval([kC0, vC0])
            for M in [1, 2, 9, 17] {
                let x = MLXRandom.normal([M, H]).asType(.float16); x.eval()
                var kc = kC0, vc = vC0
                guard let ref = SeedlessVerifyForward.attnLayerRows(x, w, kCache: &kc, vCache: &vc, M: M)
                else { return (false, "composed nil M=\(M)") }
                ref.eval()
                guard let (got, gk, gv) = SeedlessFusedVerify.fusedAttnLayerRows(x, w, kInit: kC0, vInit: vC0,
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
            let w = SeedlessVerifyForward.GDNLayerW(qkvWq: qkvW, qkvSc: qkvS, qkvBi: qkvB,
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
                guard let ref = SeedlessVerifyForward.gdnLayerRows(x, w, convState: &cs, recState: &rs, M: M)
                else { return (false, "composed nil M=\(M)") }
                ref.eval()
                guard let (got, gcs, grs) = SeedlessFusedVerify.fusedGdnLayerRows(x, w, convInit: cs0, recInit: rs0, M: M)
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
            func mkMoE() -> SeedlessVerifyForward.MoEBlockW {
                let (gW, gS, gB) = q8(E, H); let (sgW, sgS, sgB) = q8(8, H)
                let (a0, a1, a2) = q4e(E, I, H); let (b0, b1, b2) = q4e(E, I, H); let (c0, c1, c2) = q4e(E, H, I)
                let (d0, d1, d2) = q4(I, H); let (e0, e1, e2) = q4(I, H); let (f0, f1, f2) = q4(H, I)
                return SeedlessVerifyForward.MoEBlockW(gateWq: gW, gateSc: gS, gateBi: gB,
                    swGWq: a0, swGSc: a1, swGBi: a2, swUWq: b0, swUSc: b1, swUBi: b2,
                    swDWq: c0, swDSc: c1, swDBi: c2, shGWq: d0, shGSc: d1, shGBi: d2,
                    shUWq: e0, shUSc: e1, shUBi: e2, shDWq: f0, shDSc: f1, shDBi: f2,
                    sharedGateWq: sgW, sharedGateSc: sgS, sharedGateBi: sgB)
            }
            let Hk = 16, Dk = 128, Hv = 32, Dv = 128, cK = 4
            let convDim = Hk * Dk * 2 + Hv * Dv
            let (qkvW, qkvS, qkvB) = q4(convDim, H); let (zW, zS, zB) = q4(Hv * Dv, H)
            let (bW, bS, bB) = q4(Hv, H); let (aW, aS, aB) = q4(Hv, H); let (oW, oS, oB) = q4(H, Hv * Dv)
            let gdnW = SeedlessVerifyForward.GDNLayerW(qkvWq: qkvW, qkvSc: qkvS, qkvBi: qkvB,
                zWq: zW, zSc: zS, zBi: zB, bWq: bW, bSc: bS, bBi: bB, aWq: aW, aSc: aS, aBi: aB,
                outWq: oW, outSc: oS, outBi: oB,
                conv1dW: MLXRandom.normal([convDim, cK]).asType(.float16),
                normWeight: MLXRandom.normal([Dv]).asType(.float16),
                aLog: MLXRandom.normal([Hv]).asType(.float32), dtBias: MLXRandom.normal([Hv]).asType(.float32))
            let nH = 16, nKV = 2, hD = 256
            let (aqW, aqS, aqB) = q4(nH * 2 * hD, H); let (akW, akS, akB) = q4(nKV * hD, H)
            let (avW, avS, avB) = q4(nKV * hD, H); let (aoW, aoS, aoB) = q4(H, nH * hD)
            let attnW = SeedlessVerifyForward.AttnLayerW(qWq: aqW, qSc: aqS, qBi: aqB, kWq: akW, kSc: akS, kBi: akB,
                vWq: avW, vSc: avS, vBi: avB, oWq: aoW, oSc: aoS, oBi: aoB,
                qNorm: MLXRandom.normal([hD]).asType(.float16), kNorm: MLXRandom.normal([hD]).asType(.float16))
            let layers = [
                SeedlessVerifyForward.LayerSpec(isLinear: true,
                    inputLN: MLXRandom.normal([H]).asType(.float16), postLN: MLXRandom.normal([H]).asType(.float16),
                    gdn: gdnW, attn: nil, moe: mkMoE(), moeE: E, moeI: I),
                SeedlessVerifyForward.LayerSpec(isLinear: false,
                    inputLN: MLXRandom.normal([H]).asType(.float16), postLN: MLXRandom.normal([H]).asType(.float16),
                    gdn: nil, attn: attnW, moe: mkMoE(), moeE: E, moeI: I),
            ]
            let cs0 = MLXRandom.normal([cK - 1, convDim]).asType(.float16)
            let rs0 = MLXRandom.normal([1, Hv, Dv, Dk]).asType(.float32)
            let kC0 = MLXRandom.normal([nKV, 16, hD]).asType(.float16)
            let vC0 = MLXRandom.normal([nKV, 16, hD]).asType(.float16)
            MLX.eval([cs0, rs0, kC0, vC0])
            func freshCaches() -> [SeedlessVerifyForward.LayerCaches] {
                [SeedlessVerifyForward.LayerCaches(convState: cs0, recState: rs0),
                 SeedlessVerifyForward.LayerCaches(kCache: kC0, vCache: vC0)]
            }
            // ステップ列: M 掃引 + 最後に 2-step チェーン(9→3)
            let stepPlans: [[Int]] = [[1], [2], [9], [17], [9, 3]]
            for plan in stepPlans {
                let comp = freshCaches()
                guard let fused = SeedlessFusedVerify.SeedlessFusedForward(layers: layers, caches: freshCaches(),
                                                                 maxM: 17, H: H, maxSeqLen: 64)
                else { return (false, "fused init nil plan=\(plan)") }
                for (si, M) in plan.enumerated() {
                    let x = MLXRandom.normal([M, H]).asType(.float16); x.eval()
                    guard let ref = SeedlessVerifyForward.verifyForwardRows(x, layers: layers, caches: comp, M: M, metalRoute: true)
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
                guard let gotE = SeedlessFusedVerify.embedRowsRaw(toks, w: wq, scales: sc, biases: bi, H: H)
                else { return (false, "embed nil M=\(M)") }
                gotE.eval()
                let (okE, dE) = bitEqual(gotE, refE)
                if !okE { return (false, "embed M=\(M): \(dE)") }
                // argmax: raw kernel vs MLX argMax(重複値で tie-break=先頭一致 も検証)
                var lg = MLXRandom.normal([M, V]).asType(.float16)
                lg[0..., 7] = lg[0..., 3]                       // 意図的 tie
                lg.eval()
                let refA: [Int] = (0 ..< M).map { MLX.argMax(lg[$0], axis: -1).item(Int.self) }
                guard let gotA = SeedlessFusedVerify.argmaxRowsRaw(lg, M: M, V: V)
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
        typealias MoEWithArrays = (w: SeedlessVerifyForward.MoEBlockW,
                                   gW: MLXArray, gSf16: MLXArray, gBf16: MLXArray,
                                   uW: MLXArray, uSf16: MLXArray, uBf16: MLXArray,
                                   dW: MLXArray, dSf16: MLXArray, dBf16: MLXArray)
        func stMkMoE() -> MoEWithArrays {
            let (gW, gS, gB) = stQ8(stE, stH); let (sgW, sgS, sgB) = stQ8(8, stH)
            let (a0, a1, a2) = stQ4e(stE, stI, stH)
            let (b0, b1, b2) = stQ4e(stE, stI, stH)
            let (c0, c1, c2) = stQ4e(stE, stH, stI)
            let (d0, d1, d2) = stQ4(stI, stH); let (e0, e1, e2) = stQ4(stI, stH); let (f0, f1, f2) = stQ4(stH, stI)
            let moeW = SeedlessVerifyForward.MoEBlockW(gateWq: gW, gateSc: gS, gateBi: gB,
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
        func stMkGdnW() -> SeedlessVerifyForward.GDNLayerW {
            let (qkvW, qkvS, qkvB) = stQ4(stConvDim, stH); let (zW, zS, zB) = stQ4(stHv * stDv, stH)
            let (bW, bS, bB) = stQ4(stHv, stH); let (aW, aS, aB) = stQ4(stHv, stH); let (oW, oS, oB) = stQ4(stH, stHv * stDv)
            return SeedlessVerifyForward.GDNLayerW(qkvWq: qkvW, qkvSc: qkvS, qkvBi: qkvB,
                zWq: zW, zSc: zS, zBi: zB, bWq: bW, bSc: bS, bBi: bB, aWq: aW, aSc: aS, aBi: aB,
                outWq: oW, outSc: oS, outBi: oB,
                conv1dW: MLXRandom.normal([stConvDim, stCK]).asType(.float16),
                normWeight: MLXRandom.normal([stDv]).asType(.float16),
                aLog: MLXRandom.normal([stHv]).asType(.float32), dtBias: MLXRandom.normal([stHv]).asType(.float32))
        }
        let stNh = 16, stNkv = 2, stHd = 256
        func stMkAttnW() -> SeedlessVerifyForward.AttnLayerW {
            let (aqW, aqS, aqB) = stQ4(stNh * 2 * stHd, stH); let (akW, akS, akB) = stQ4(stNkv * stHd, stH)
            let (avW, avS, avB) = stQ4(stNkv * stHd, stH); let (aoW, aoS, aoB) = stQ4(stH, stNh * stHd)
            return SeedlessVerifyForward.AttnLayerW(qWq: aqW, qSc: aqS, qBi: aqB, kWq: akW, kSc: akS, kBi: akB,
                vWq: avW, vSc: avS, vBi: avB, oWq: aoW, oSc: aoS, oBi: aoB,
                qNorm: MLXRandom.normal([stHd]).asType(.float16), kNorm: MLXRandom.normal([stHd]).asType(.float16))
        }

        // Test 25 (D1-A): strict streaming ≡ resident — C=8<E=16, multi-plan, chunk assertion.
        run("stream_fused_strict_bitexact") {
            guard let (device, _) = SeedlessMetalForward.ensure() else { return (false, "no device") }
            let C = 8
            let moe0 = stMkMoE(), moe1 = stMkMoE()
            let gdnW = stMkGdnW(), attnW = stMkAttnW()
            let stCs0 = MLXRandom.normal([stCK - 1, stConvDim]).asType(.float16)
            let stRs0 = MLXRandom.normal([1, stHv, stDv, stDk]).asType(.float32)
            let stKc0 = MLXRandom.normal([stNkv, 16, stHd]).asType(.float16)
            let stVc0 = MLXRandom.normal([stNkv, 16, stHd]).asType(.float16)
            MLX.eval([stCs0, stRs0, stKc0, stVc0])
            func freshC() -> [SeedlessVerifyForward.LayerCaches] {
                [SeedlessVerifyForward.LayerCaches(convState: stCs0, recState: stRs0),
                 SeedlessVerifyForward.LayerCaches(kCache: stKc0, vCache: stVc0)]
            }
            let layerSpecs = [
                SeedlessVerifyForward.LayerSpec(isLinear: true,
                    inputLN: MLXRandom.normal([stH]).asType(.float16), postLN: MLXRandom.normal([stH]).asType(.float16),
                    gdn: gdnW, attn: nil, moe: moe0.w, moeE: stE, moeI: stI),
                SeedlessVerifyForward.LayerSpec(isLinear: false,
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
                guard let res = SeedlessFusedVerify.SeedlessFusedForward(
                        layers: layerSpecs, caches: freshC(), maxM: 17, H: stH, maxSeqLen: 64),
                      let str = SeedlessFusedVerify.SeedlessFusedForward(
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
            guard let (device, _) = SeedlessMetalForward.ensure() else { return (false, "no device") }
            let C = 8
            let moe0 = stMkMoE(), moe1 = stMkMoE()
            let gdnW = stMkGdnW(), attnW = stMkAttnW()
            let stCs0 = MLXRandom.normal([stCK - 1, stConvDim]).asType(.float16)
            let stRs0 = MLXRandom.normal([1, stHv, stDv, stDk]).asType(.float32)
            let stKc0 = MLXRandom.normal([stNkv, 16, stHd]).asType(.float16)
            let stVc0 = MLXRandom.normal([stNkv, 16, stHd]).asType(.float16)
            MLX.eval([stCs0, stRs0, stKc0, stVc0])
            let layerSpecs = [
                SeedlessVerifyForward.LayerSpec(isLinear: true,
                    inputLN: MLXRandom.normal([stH]).asType(.float16), postLN: MLXRandom.normal([stH]).asType(.float16),
                    gdn: gdnW, attn: nil, moe: moe0.w, moeE: stE, moeI: stI),
                SeedlessVerifyForward.LayerSpec(isLinear: false,
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
            let freshC: [SeedlessVerifyForward.LayerCaches] = [
                SeedlessVerifyForward.LayerCaches(convState: stCs0, recState: stRs0),
                SeedlessVerifyForward.LayerCaches(kCache: stKc0, vCache: stVc0)]
            guard let res = SeedlessFusedVerify.SeedlessFusedForward(
                    layers: layerSpecs, caches: freshC, maxM: 17, H: stH, maxSeqLen: 64),
                  let str = SeedlessFusedVerify.SeedlessFusedForward(
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
            guard let (device, _) = SeedlessMetalForward.ensure() else { return (false, "no device") }
            let C = 8  // C < E=16 → eviction forced
            let moe0 = stMkMoE(), moe1 = stMkMoE()
            let gdnW = stMkGdnW(), attnW = stMkAttnW()
            let stCs0 = MLXRandom.normal([stCK - 1, stConvDim]).asType(.float16)
            let stRs0 = MLXRandom.normal([1, stHv, stDv, stDk]).asType(.float32)
            let stKc0 = MLXRandom.normal([stNkv, 16, stHd]).asType(.float16)
            let stVc0 = MLXRandom.normal([stNkv, 16, stHd]).asType(.float16)
            MLX.eval([stCs0, stRs0, stKc0, stVc0])
            func freshC() -> [SeedlessVerifyForward.LayerCaches] {
                [SeedlessVerifyForward.LayerCaches(convState: stCs0, recState: stRs0),
                 SeedlessVerifyForward.LayerCaches(kCache: stKc0, vCache: stVc0)]
            }
            let layerSpecs = [
                SeedlessVerifyForward.LayerSpec(isLinear: true,
                    inputLN: MLXRandom.normal([stH]).asType(.float16), postLN: MLXRandom.normal([stH]).asType(.float16),
                    gdn: gdnW, attn: nil, moe: moe0.w, moeE: stE, moeI: stI),
                SeedlessVerifyForward.LayerSpec(isLinear: false,
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
            guard let res = SeedlessFusedVerify.SeedlessFusedForward(
                    layers: layerSpecs, caches: freshC(), maxM: 9, H: stH, maxSeqLen: 64),
                  let str = SeedlessFusedVerify.SeedlessFusedForward(
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
            guard let (device, _) = SeedlessMetalForward.ensure() else { return (false, "no device") }
            let C = stE  // C = E: all experts fit without eviction
            let moe0 = stMkMoE(), moe1 = stMkMoE()
            let gdnW = stMkGdnW(), attnW = stMkAttnW()
            let stCs0 = MLXRandom.normal([stCK - 1, stConvDim]).asType(.float16)
            let stRs0 = MLXRandom.normal([1, stHv, stDv, stDk]).asType(.float32)
            let stKc0 = MLXRandom.normal([stNkv, 16, stHd]).asType(.float16)
            let stVc0 = MLXRandom.normal([stNkv, 16, stHd]).asType(.float16)
            MLX.eval([stCs0, stRs0, stKc0, stVc0])
            func freshC() -> [SeedlessVerifyForward.LayerCaches] {
                [SeedlessVerifyForward.LayerCaches(convState: stCs0, recState: stRs0),
                 SeedlessVerifyForward.LayerCaches(kCache: stKc0, vCache: stVc0)]
            }
            let layerSpecs = [
                SeedlessVerifyForward.LayerSpec(isLinear: true,
                    inputLN: MLXRandom.normal([stH]).asType(.float16), postLN: MLXRandom.normal([stH]).asType(.float16),
                    gdn: gdnW, attn: nil, moe: moe0.w, moeE: stE, moeI: stI),
                SeedlessVerifyForward.LayerSpec(isLinear: false,
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
            guard let res = SeedlessFusedVerify.SeedlessFusedForward(
                    layers: layerSpecs, caches: freshC(), maxM: 17, H: stH, maxSeqLen: 64),
                  let blt = SeedlessFusedVerify.SeedlessFusedForward(
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
        // a regression guard. It does NOT require A3 implementation in Tell.swift.
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
                SeedlessVerifyForward.LayerSpec(isLinear: true,
                    inputLN: iLN0a, postLN: pLN0a,
                    gdn: gdnWa, attn: nil, moe: moe0a.w, moeE: stE, moeI: stI),
                SeedlessVerifyForward.LayerSpec(isLinear: false,
                    inputLN: iLN1a, postLN: pLN1a,
                    gdn: nil, attn: attnWa, moe: moe1a.w, moeE: stE, moeI: stI),
            ]
            func freshCachesA() -> [SeedlessVerifyForward.LayerCaches] {
                [SeedlessVerifyForward.LayerCaches(convState: csA, recState: rsA),
                 SeedlessVerifyForward.LayerCaches(kCache: kcA, vCache: vcA)]
            }
            // spec §6 G1 matrix: pk ∈ {0,3,7,17} × D ∈ {1,4,8}
            for pk in [0, 3, 7, 17] {
                for D in [1, 4, 8] {
                    let M = pk + 1 + D   // total rows in the batched call
                    // maxSeqLen covers initial 8 + M + margin
                    guard let fused = SeedlessFusedVerify.SeedlessFusedForward(
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
                SeedlessVerifyForward.LayerSpec(isLinear: true,
                    inputLN: iLN0b, postLN: pLN0b,
                    gdn: gdnWb, attn: nil, moe: moe0b.w, moeE: stE, moeI: stI),
                SeedlessVerifyForward.LayerSpec(isLinear: false,
                    inputLN: iLN1b, postLN: pLN1b,
                    gdn: nil, attn: attnWb, moe: moe1b.w, moeE: stE, moeI: stI),
            ]
            func freshCachesB() -> [SeedlessVerifyForward.LayerCaches] {
                [SeedlessVerifyForward.LayerCaches(convState: csB, recState: rsB),
                 SeedlessVerifyForward.LayerCaches(kCache: kcB, vCache: vcB)]
            }
            let maxSeqB = 4 + M1 + M2 + 8    // initial(4) + max scenario(M1 or M2) + margin
            let maxMb   = Swift.max(M1, M2) + 2
            // Two independent fused forwards starting from identical state B
            guard let fusedNA = SeedlessFusedVerify.SeedlessFusedForward(
                    layers: layersB, caches: freshCachesB(), maxM: maxMb, H: H, maxSeqLen: maxSeqB),
                  let fusedA3 = SeedlessFusedVerify.SeedlessFusedForward(
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
                SeedlessVerifyForward.LayerSpec(isLinear: true,
                    inputLN: iLN0c, postLN: pLN0c,
                    gdn: gdnWc, attn: nil, moe: moe0c.w, moeE: stE, moeI: stI),
                SeedlessVerifyForward.LayerSpec(isLinear: false,
                    inputLN: iLN1c, postLN: pLN1c,
                    gdn: nil, attn: attnWc, moe: moe1c.w, moeE: stE, moeI: stI),
            ]
            func freshCachesC() -> [SeedlessVerifyForward.LayerCaches] {
                [SeedlessVerifyForward.LayerCaches(convState: csC, recState: rsC),
                 SeedlessVerifyForward.LayerCaches(kCache: kcC, vCache: vcC)]
            }
            // maxSeqLen: initial(4) + M_total(28) + margin = 40; use 64 for safety
            // maxM: must cover flushN=24 (largest single call in flush path)
            guard let refFused = SeedlessFusedVerify.SeedlessFusedForward(
                    layers: layersC, caches: freshCachesC(),
                    maxM: flushN + 2, H: H, maxSeqLen: 64),
                  let flushFused = SeedlessFusedVerify.SeedlessFusedForward(
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
                        guard let g = SeedlessMetalForward.gatherQmmRows(x, wGq, scales: wGS, biases: wGB,
                                                                     inds: inds, M: M, Ktop: Ktop, K: K, N: N),
                              let u = SeedlessMetalForward.gatherQmmRows(x, wUq, scales: wUS, biases: wUB,
                                                                     inds: inds, M: M, Ktop: Ktop, K: K, N: N)
                        else { return (false, "ref gather nil M=\(M) Ktop=\(Ktop) N=\(N)") }
                        g.eval(); u.eval()
                        guard let hRef = SeedlessMetalForward.swigluRaw(g, u)
                        else { return (false, "ref swiglu nil M=\(M) Ktop=\(Ktop) N=\(N)") }
                        hRef.eval()
                        // Stub under test
                        guard let hGot = SeedlessFusedVerify.gatherQmmSwigluRows(
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
            guard let hBatch = SeedlessFusedVerify.gatherQmmSwigluRows(
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
                guard let hm = SeedlessFusedVerify.gatherQmmSwigluRows(
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
            guard let (catW, catS, catB) = SeedlessFusedVerify.gdnInProjConcat(
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
                guard let refQkv = SeedlessMetalForward.qmmRows(x, qkvW, scales: qkvS, biases: qkvB,
                                                            M: M, K: H, N: convDim),
                      let refZ   = SeedlessMetalForward.qmmRows(x, zW,   scales: zS,   biases: zB,
                                                            M: M, K: H, N: valueDim),
                      let refB   = SeedlessMetalForward.qmmRows(x, bW,   scales: bS,   biases: bB,
                                                            M: M, K: H, N: numVH),
                      let refA   = SeedlessMetalForward.qmmRows(x, aW,   scales: aS,   biases: aB,
                                                            M: M, K: H, N: numVH)
                else { return (false, "ref qmmRows nil M=\(M)") }
                MLX.eval([refQkv, refZ, refB, refA])
                // Fused: single qmmRows on concatenated weight, sliced at offsets
                guard let fused = SeedlessMetalForward.qmmRows(x, catW, scales: catS, biases: catB,
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
                guard let convOutRef = SeedlessMetalForward.conv1dSiluRows(windows, convW, M: M, K: K, C: C)
                else { return (false, "ref conv1dSiluRows nil M=\(M)") }
                // Reference histOut: tail K-1 frames of (histIn‖qkv) — pure data movement,
                // same bytes shift_conv_rows copies; not a computation reimplementation.
                let histOutRef = convInput[M ..< M + K - 1].asType(.float16)
                MLX.eval([convOutRef, histOutRef])
                // Fused: stub under test
                guard let (convOutGot, histOutGot) = SeedlessFusedVerify.gdnConvShiftFused(
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
                    guard let normedRef = SeedlessMetalForward.rmsNormRows(coreOut, normW,
                                                                        M: M * Hv, eps: 1e-6, D: Dv,
                                                                        promoteF32: promote)
                    else { return (false, "ref rmsNormRows nil M=\(M) promote=\(promote)") }
                    normedRef.eval()
                    // then gateRaw (existing production kernel wrapping encodeGate)
                    guard let outVRef = SeedlessFusedVerify.gateRaw(z, normedRef,
                                                                promote: promote, total: M * valueDim)
                    else { return (false, "ref gateRaw nil M=\(M) promote=\(promote)") }
                    outVRef.eval()
                    // Fused: stub under test
                    guard let outVGot = SeedlessFusedVerify.gdnNormGateFused(
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
        // Asserts that SeedlessFusedForward.fuseGUActive(M:) is implemented and returns:
        //   true  iff QWISP_FUSE_GU=1 AND M==1   (fused gather+swiglu, M=1 only)
        //   false iff QWISP_FUSE_GU=1 AND M>1    (register-pressure fallback)
        //   false iff QWISP_FUSE_GU=0             (flag disabled)
        // Stub returns nil → RED. Behavioral correctness (actual tokens) is gated by G2.
        run("fuse_gu_m_branch") {
            guard let active1 = SeedlessFusedVerify.SeedlessFusedForward.fuseGUActive(M: 1)
            else { return (false, "not implemented") }
            guard let active8 = SeedlessFusedVerify.SeedlessFusedForward.fuseGUActive(M: 8)
            else { return (false, "not implemented M=8") }
            let fuseOn = SeedlessFusedVerify.SeedlessFusedForward.fuseGU
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
        //   Reference: 4 separate SeedlessMetalForward.qmmRows (existing production kernel).
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
            guard let (catW, catS, catB) = SeedlessFusedVerify.gdnInProjConcat(
                qkvW: qkvW, qkvS: qkvS, qkvB: qkvB,
                zW:   zW,   zS:   zS,   zB:   zB,
                bW:   bW,   bS:   bS,   bB:   bB,
                aW:   aW,   aS:   aS,   aB:   aB)
            else { return (false, "gdnInProjConcat nil") }
            MLX.eval([catW, catS, catB])
            for M in [1, 8] {
                let x = MLXRandom.normal([M, K]).asType(.float16); x.eval()
                // Reference: 4 separate qmmRows (existing production kernel)
                guard let refQkv = SeedlessMetalForward.qmmRows(x, qkvW, scales: qkvS, biases: qkvB,
                                                            M: M, K: K, N: dims.qkv),
                      let refZ   = SeedlessMetalForward.qmmRows(x, zW,   scales: zS,   biases: zB,
                                                            M: M, K: K, N: dims.z),
                      let refB   = SeedlessMetalForward.qmmRows(x, bW,   scales: bS,   biases: bB,
                                                            M: M, K: K, N: dims.b),
                      let refA   = SeedlessMetalForward.qmmRows(x, aW,   scales: aS,   biases: aB,
                                                            M: M, K: K, N: dims.a)
                else { return (false, "ref qmmRows nil M=\(M)") }
                MLX.eval([refQkv, refZ, refB, refA])
                // Stub under test: single dispatch demuxing into 4 output buffers
                guard let (gotQkv, gotZ, gotB, gotA) = SeedlessFusedVerify.gdnInProjDemux(
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
                    guard let normedRef = SeedlessMetalForward.rmsNormRows(coreOut, normW,
                                                                       M: M * Hv, eps: 1e-6, D: Dv,
                                                                       promoteF32: promote)
                    else { return (false, "ref rmsNormRows nil M=\(M) promote=\(promote)") }
                    normedRef.eval()
                    guard let outVRef = SeedlessFusedVerify.gateRaw(z, normedRef,
                                                                promote: promote, total: M * valueDim)
                    else { return (false, "ref gateRaw nil M=\(M) promote=\(promote)") }
                    outVRef.eval()
                    // Stub under test: single-dispatch production kernel (distinct from wrapper)
                    guard let outVGot = SeedlessFusedVerify.gdnNormGateRows(
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
                guard let qnNorm = SeedlessMetalForward.rmsNormRows(
                        q1, onesQ, M: M * numKHeads, eps: eps, D: headKDim)
                else { return (false, "ref rmsNorm qn nil M=\(M)") }
                // ⑫ rmsnorm kn (ones weight, per-head over headKDim)
                guard let knNorm = SeedlessMetalForward.rmsNormRows(
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
                guard let (gRef, betaRef) = SeedlessFusedVerify.computeGBetaRowsRaw(
                        aP, bP, aLog, dtBias, M: M, Hv: numVHeads)
                else { return (false, "ref computeGBeta nil M=\(M)") }
                gRef.eval(); betaRef.eval()
                // ── Fused stub ──
                guard let (qnGot, knGot, vGot, gGot, betaGot) = SeedlessFusedVerify.gdnPrepFused(
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
                guard let postNormRef = SeedlessMetalForward.rmsNormRows(hRef, postW, M: M, eps: eps, D: H)
                else { return (false, "ref rmsNormRows nil M=\(M)") }
                postNormRef.eval()
                // ── Fused stub ──
                guard let (hGot, postNormGot) = SeedlessFusedVerify.gdnResidPostNormFused(
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
                guard let qRef = SeedlessMetalForward.qmmRows(x, qW, scales: qS, biases: qB, M: M, K: H, N: Nq),
                      let kRef = SeedlessMetalForward.qmmRows(x, kW, scales: kS, biases: kB, M: M, K: H, N: Nk),
                      let vRef = SeedlessMetalForward.qmmRows(x, vW, scales: vS, biases: vB, M: M, K: H, N: Nv)
                else { return (false, "ref qmmRows nil M=\(M)") }
                MLX.eval([qRef, kRef, vRef])
                // Stub under test
                guard let (qGot, kGot, vGot) = SeedlessFusedVerify.attnQkvDemux(x,
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
                guard let qN = SeedlessMetalForward.rmsNormRows(qX, qNorm,
                                                            M: M * numHeads, eps: eps, D: headDim)
                else { return (false, "ref rmsNormRows nil M=\(M)") }
                qN.eval()
                // ⑦ rope q (numHeads lanes, position = startOffset + m)
                guard let qRot = SeedlessMetalForward.ropeRows(qN, headDim: headDim, ropeDim: ropeDim,
                                                           base: ropeBase, startOffset: startOffset,
                                                           M: M, numHeads: numHeads)
                else { return (false, "ref ropeRows nil M=\(M)") }
                qRot.eval()
                // Stub under test
                guard let qRotGot = SeedlessFusedVerify.attnQPrepFused(qOut, qNorm: qNorm,
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
                guard let kN = SeedlessMetalForward.rmsNormRows(kOut, kNorm,
                                                            M: M * numKV, eps: eps, D: headDim)
                else { return (false, "ref rmsNormRows nil M=\(M)") }
                kN.eval()
                // ⑧ rope k (numHeads=numKV for k-path, position = baseLen + m)
                guard let kRot = SeedlessMetalForward.ropeRows(kN, headDim: headDim, ropeDim: ropeDim,
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
                guard let (kRotGot, kCacheGot) = SeedlessFusedVerify.attnKPrepFused(kOut, kNorm: kNorm,
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
            guard let sg = SeedlessMetalForward.qmmRows(x, shGW, scales: shGS, biases: shGB,
                                                    M: M, K: H, N: I),
                  let su = SeedlessMetalForward.qmmRows(x, shUW, scales: shUS, biases: shUB,
                                                    M: M, K: H, N: I)
            else { return (false, "ref qmmRows nil") }
            sg.eval(); su.eval()
            guard let shActRef = SeedlessMetalForward.swigluRaw(sg, su)
            else { return (false, "ref swigluRaw nil") }
            shActRef.eval()
            // Stub under test
            guard let shActGot = SeedlessFusedVerify.sharedGUSwigluFused(x,
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
                guard let sgl = SeedlessMetalForward.qmm8(x, sgW, scales: sgS, biases: sgB,
                                                      M: M, K: H, N: 8)
                else { return (false, "ref qmm8 nil M=\(M)") }
                sgl.eval()
                guard let outRef = SeedlessFusedVerify.finalCombineRowsRaw(y, sharedY, sgl, M: M, N: H)
                else { return (false, "ref finalCombineRowsRaw nil M=\(M)") }
                outRef.eval()
                // Stub under test
                guard let outGot = SeedlessFusedVerify.sharedGateCombineFused(x, y: y, sharedY: sharedY,
                        sgW: sgW, sgS: sgS, sgB: sgB, M: M, H: H)
                else { return (false, "not implemented (M=\(M))") }
                outGot.eval()
                let (ok, d) = bitEqual(outGot, outRef)
                if !ok { return (false, "M=\(M): \(d)") }
            }
            return (true, "ok")
        }

        // ── Phase II-a G1 gate: K-step chained greedy decode tests (47-48) ──────────
        //
        // WRITE-LOCKED: implementer MUST NOT modify these tests.
        // They encode the G1 acceptance gate from notes/09-phase2-token-feedback-spec.md §4.
        //
        // Model geometry: same 2-layer mini-model as tests 25–28 (stH/stE/stI,
        // resident mode, no providers). Small vocab V=256 keeps embed/lm_head cheap.
        // Stub returns nil → both tests RED. GREEN-by-delegation is explicitly forbidden:
        // the stub must not call stepArgmax or forwardRows (nil-return contract enforces this).

        // Test 47 (II-a G1-bitexact): chained_greedy_bitexact
        // chainedStepArgmax(K) K-token list + post-chain cache state (KV len/contents,
        // GDN conv+rec state) bit-equal to K sequential stepArgmax([t]) calls, K ∈ {2, 3, 8}.
        // Both engines start from identical random initial cache state and shared head weights.
        // Adversarial cases: K=2 (minimal chain), K=3 (odd, non-power-of-2), K=8 (8×GDN/attn steps).
        run("chained_greedy_bitexact") {
            let V = 256   // small vocab — embed/lm_head stay cheap in test
            // Shared layer weights (same objects passed to both chain and ref engines)
            let moeC0 = stMkMoE(), moeC1 = stMkMoE()
            let gdnWC = stMkGdnW(), attnWC = stMkAttnW()
            let iLNC0 = MLXRandom.normal([stH]).asType(.float16)
            let pLNC0 = MLXRandom.normal([stH]).asType(.float16)
            let iLNC1 = MLXRandom.normal([stH]).asType(.float16)
            let pLNC1 = MLXRandom.normal([stH]).asType(.float16)
            MLX.eval([iLNC0, pLNC0, iLNC1, pLNC1])
            let layerSpecsC = [
                SeedlessVerifyForward.LayerSpec(isLinear: true,
                    inputLN: iLNC0, postLN: pLNC0,
                    gdn: gdnWC, attn: nil, moe: moeC0.w, moeE: stE, moeI: stI),
                SeedlessVerifyForward.LayerSpec(isLinear: false,
                    inputLN: iLNC1, postLN: pLNC1,
                    gdn: nil, attn: attnWC, moe: moeC1.w, moeE: stE, moeI: stI),
            ]
            // Initial cache state — same values for all K iterations; each engine gets its own copy
            let csC = MLXRandom.normal([stCK - 1, stConvDim]).asType(.float16)
            let rsC = MLXRandom.normal([1, stHv, stDv, stDk]).asType(.float32)
            let kcC = MLXRandom.normal([stNkv, 8, stHd]).asType(.float16)   // 8 initial KV positions
            let vcC = MLXRandom.normal([stNkv, 8, stHd]).asType(.float16)
            MLX.eval([csC, rsC, kcC, vcC])
            func freshCachesC() -> [SeedlessVerifyForward.LayerCaches] {
                [SeedlessVerifyForward.LayerCaches(convState: csC, recState: rsC),
                 SeedlessVerifyForward.LayerCaches(kCache: kcC, vCache: vcC)]
            }
            // Shared head weights (same objects for both engines — embed = lm_head = 4-bit q)
            let ewf = MLXRandom.normal([V, stH]).asType(.float16)
            let (ewq, esc, ebiOpt) = MLX.quantized(ewf, groupSize: 64, bits: 4, mode: .affine)
            guard let ebi = ebiOpt else { return (false, "embed biases nil") }
            let lwf = MLXRandom.normal([V, stH]).asType(.float16)
            let (lwq, lsc, lbiOpt) = MLX.quantized(lwf, groupSize: 64, bits: 4, mode: .affine)
            guard let lbi = lbiOpt else { return (false, "lm biases nil") }
            let fnWC = MLXRandom.normal([stH]).asType(.float16)
            MLX.eval([ewq, esc, ebi, lwq, lsc, lbi, fnWC])
            // maxSeqLen: initial 8 + up to K=8 steps + margin = 32
            let maxSeqC = 32
            // Fixed seed token within [0, V)
            let firstToken = Int32(7)
            for K in [2, 3, 8] {
                // Chain engine — will call chainedStepArgmax (stub → RED)
                guard let eng1 = SeedlessFusedVerify.SeedlessFusedForward(
                        layers: layerSpecsC, caches: freshCachesC(),
                        maxM: 4, H: stH, maxSeqLen: maxSeqC)
                else { return (false, "init eng1 nil K=\(K)") }
                guard eng1.attachHead(embedW: ewq, embedS: esc, embedB: ebi,
                                      lmW: lwq, lmS: lsc, lmB: lbi,
                                      fnW: fnWC, vocab: V)
                else { return (false, "attachHead eng1 nil K=\(K)") }
                // Reference engine — will call stepArgmax K times
                guard let eng2 = SeedlessFusedVerify.SeedlessFusedForward(
                        layers: layerSpecsC, caches: freshCachesC(),
                        maxM: 4, H: stH, maxSeqLen: maxSeqC)
                else { return (false, "init eng2 nil K=\(K)") }
                guard eng2.attachHead(embedW: ewq, embedS: esc, embedB: ebi,
                                      lmW: lwq, lmS: lsc, lmB: lbi,
                                      fnW: fnWC, vocab: V)
                else { return (false, "attachHead eng2 nil K=\(K)") }
                // STUB returns nil → test goes RED here (K=2 on first iteration)
                guard let gotTokens = eng1.chainedStepArgmax(firstToken, K: K)
                else { return (false, "not implemented (K=\(K))") }
                if gotTokens.count != K {
                    return (false, "token count K=\(K): got=\(gotTokens.count) want=\(K)")
                }
                // Reference: K sequential single-token stepArgmax calls
                var refTokens: [Int] = []
                var cur = firstToken
                for step in 0..<K {
                    guard let t = eng2.stepArgmax([cur])
                    else { return (false, "ref stepArgmax nil K=\(K) step=\(step)") }
                    refTokens.append(t[0])
                    cur = Int32(t[0])
                }
                // Token list must be bit-equal
                if gotTokens != refTokens {
                    return (false, "tokens K=\(K): got=\(gotTokens) ref=\(refTokens)")
                }
                // KV len must match for every layer
                let snap1 = eng1.snapshot(), snap2 = eng2.snapshot()
                for li in 0..<layerSpecsC.count {
                    if snap1.kvLens[li] != snap2.kvLens[li] {
                        return (false, "kvLen K=\(K) layer=\(li): got=\(snap1.kvLens[li]) ref=\(snap2.kvLens[li])")
                    }
                }
                // KV cache contents: layer 1 (attn). readLayerCache returns [KV, len, D] (already sliced)
                let (k1c, v1c) = eng1.readLayerCache(1)
                let (k2c, v2c) = eng2.readLayerCache(1)
                for (nm, ca, cb) in [("kCache", k1c, k2c), ("vCache", v1c, v2c)] {
                    guard let aa = ca, let bb = cb else { return (false, "KV cache nil K=\(K) \(nm)") }
                    aa.eval(); bb.eval()
                    let (ok, d) = bitEqual(aa, bb)
                    if !ok { return (false, "K=\(K) \(nm): \(d)") }
                }
                // GDN state: layer 0. readLayerCache returns (convState [K-1,C], recState [1,Hv,Dv,Dk])
                let (gconv1, grec1) = eng1.readLayerCache(0)
                let (gconv2, grec2) = eng2.readLayerCache(0)
                for (nm, ca, cb) in [("convState", gconv1, gconv2), ("recState", grec1, grec2)] {
                    guard let aa = ca, let bb = cb else { return (false, "GDN state nil K=\(K) \(nm)") }
                    aa.eval(); bb.eval()
                    let (ok, d) = bitEqual(aa, bb)
                    if !ok { return (false, "K=\(K) \(nm): \(d)") }
                }
            }
            return (true, "ok")
        }

        // Test 48 (II-a G1-boundary): chained_boundary
        // chain→per-step→chain interleave stays bit-exact:
        //   interleaved: chainedStepArgmax(K1=3) → stepArgmax×K2=2 → chainedStepArgmax(K3=4)
        //   reference:   stepArgmax×(K1+K2+K3=9) sequential
        // Token list and final cache state (KV + GDN) both bit-equal.
        // Stub nil → RED at the very first chain call (phase 1).
        run("chained_boundary") {
            let V = 256
            let K1 = 3, K2 = 2, K3 = 4   // chain-step-chain counts; total = K1+K2+K3 = 9
            let moeB0 = stMkMoE(), moeB1 = stMkMoE()
            let gdnWB = stMkGdnW(), attnWB = stMkAttnW()
            let iLNB0 = MLXRandom.normal([stH]).asType(.float16)
            let pLNB0 = MLXRandom.normal([stH]).asType(.float16)
            let iLNB1 = MLXRandom.normal([stH]).asType(.float16)
            let pLNB1 = MLXRandom.normal([stH]).asType(.float16)
            MLX.eval([iLNB0, pLNB0, iLNB1, pLNB1])
            let layerSpecsB = [
                SeedlessVerifyForward.LayerSpec(isLinear: true,
                    inputLN: iLNB0, postLN: pLNB0,
                    gdn: gdnWB, attn: nil, moe: moeB0.w, moeE: stE, moeI: stI),
                SeedlessVerifyForward.LayerSpec(isLinear: false,
                    inputLN: iLNB1, postLN: pLNB1,
                    gdn: nil, attn: attnWB, moe: moeB1.w, moeE: stE, moeI: stI),
            ]
            let csB = MLXRandom.normal([stCK - 1, stConvDim]).asType(.float16)
            let rsB = MLXRandom.normal([1, stHv, stDv, stDk]).asType(.float32)
            let kcB = MLXRandom.normal([stNkv, 8, stHd]).asType(.float16)
            let vcB = MLXRandom.normal([stNkv, 8, stHd]).asType(.float16)
            MLX.eval([csB, rsB, kcB, vcB])
            func freshCachesB() -> [SeedlessVerifyForward.LayerCaches] {
                [SeedlessVerifyForward.LayerCaches(convState: csB, recState: rsB),
                 SeedlessVerifyForward.LayerCaches(kCache: kcB, vCache: vcB)]
            }
            let ewfB = MLXRandom.normal([V, stH]).asType(.float16)
            let (ewqB, escB, ebiOptB) = MLX.quantized(ewfB, groupSize: 64, bits: 4, mode: .affine)
            guard let ebiB = ebiOptB else { return (false, "embed biases nil") }
            let lwfB = MLXRandom.normal([V, stH]).asType(.float16)
            let (lwqB, lscB, lbiOptB) = MLX.quantized(lwfB, groupSize: 64, bits: 4, mode: .affine)
            guard let lbiB = lbiOptB else { return (false, "lm biases nil") }
            let fnWB = MLXRandom.normal([stH]).asType(.float16)
            MLX.eval([ewqB, escB, ebiB, lwqB, lscB, lbiB, fnWB])
            // maxSeqLen: initial 8 + total K1+K2+K3=9 steps + margin = 32
            let maxSeqB = 32
            let seedToken = Int32(13)
            // Interleaved engine (chain → step × K2 → chain)
            guard let engC = SeedlessFusedVerify.SeedlessFusedForward(
                    layers: layerSpecsB, caches: freshCachesB(),
                    maxM: 4, H: stH, maxSeqLen: maxSeqB)
            else { return (false, "init engC nil") }
            guard engC.attachHead(embedW: ewqB, embedS: escB, embedB: ebiB,
                                   lmW: lwqB, lmS: lscB, lmB: lbiB,
                                   fnW: fnWB, vocab: V)
            else { return (false, "attachHead engC nil") }
            // Reference engine (all sequential stepArgmax)
            guard let engR = SeedlessFusedVerify.SeedlessFusedForward(
                    layers: layerSpecsB, caches: freshCachesB(),
                    maxM: 4, H: stH, maxSeqLen: maxSeqB)
            else { return (false, "init engR nil") }
            guard engR.attachHead(embedW: ewqB, embedS: escB, embedB: ebiB,
                                   lmW: lwqB, lmS: lscB, lmB: lbiB,
                                   fnW: fnWB, vocab: V)
            else { return (false, "attachHead engR nil") }
            // Phase 1: chain K1 steps (STUB → nil → RED immediately)
            guard let phase1Toks = engC.chainedStepArgmax(seedToken, K: K1)
            else { return (false, "not implemented (chain phase1)") }
            var refCur = seedToken
            var refAll: [Int] = []
            for _ in 0..<K1 {
                guard let t = engR.stepArgmax([refCur]) else { return (false, "ref stepArgmax K1") }
                refAll.append(t[0]); refCur = Int32(t[0])
            }
            if phase1Toks != Array(refAll[0..<K1]) {
                return (false, "phase1 tokens: got=\(phase1Toks) ref=\(Array(refAll[0..<K1]))")
            }
            // Phase 2: K2 per-step calls on engC (simulating spec verification / greedy resumption)
            var chainCur = Int32(phase1Toks.last!)
            for _ in 0..<K2 {
                guard let ct = engC.stepArgmax([chainCur]) else { return (false, "engC stepArgmax K2") }
                guard let rt = engR.stepArgmax([refCur])   else { return (false, "engR stepArgmax K2") }
                if ct[0] != rt[0] { return (false, "phase2 token mismatch: got=\(ct[0]) ref=\(rt[0])") }
                refAll.append(rt[0]); chainCur = Int32(ct[0]); refCur = Int32(rt[0])
            }
            // Phase 3: chain K3 steps (will also be nil if stub is still nil — but RED is at phase1)
            guard let phase3Toks = engC.chainedStepArgmax(chainCur, K: K3)
            else { return (false, "not implemented (chain phase3)") }
            for _ in 0..<K3 {
                guard let t = engR.stepArgmax([refCur]) else { return (false, "ref stepArgmax K3") }
                refAll.append(t[0]); refCur = Int32(t[0])
            }
            let refPhase3 = Array(refAll[K1+K2 ..< K1+K2+K3])
            if phase3Toks != refPhase3 {
                return (false, "phase3 tokens: got=\(phase3Toks) ref=\(refPhase3)")
            }
            // Final cache state comparison (KV + GDN)
            let snapC = engC.snapshot(), snapR = engR.snapshot()
            for li in 0..<layerSpecsB.count {
                if snapC.kvLens[li] != snapR.kvLens[li] {
                    return (false, "final kvLen layer=\(li): got=\(snapC.kvLens[li]) ref=\(snapR.kvLens[li])")
                }
            }
            let (fkC, fvC) = engC.readLayerCache(1)
            let (fkR, fvR) = engR.readLayerCache(1)
            for (nm, ca, cb) in [("kCache", fkC, fkR), ("vCache", fvC, fvR)] {
                guard let aa = ca, let bb = cb else { return (false, "final KV nil \(nm)") }
                aa.eval(); bb.eval()
                let (ok, d) = bitEqual(aa, bb)
                if !ok { return (false, "final \(nm): \(d)") }
            }
            let (fcC, frC) = engC.readLayerCache(0)
            let (fcR, frR) = engR.readLayerCache(0)
            for (nm, ca, cb) in [("convState", fcC, fcR), ("recState", frC, frR)] {
                guard let aa = ca, let bb = cb else { return (false, "final GDN nil \(nm)") }
                aa.eval(); bb.eval()
                let (ok, d) = bitEqual(aa, bb)
                if !ok { return (false, "final \(nm): \(d)") }
            }
            return (true, "ok")
        }

        // ── Default-flip tests (49-50): promote proven-best config to default ──
        //
        // WRITE-LOCKED (locked7): implementer MUST NOT modify these tests.
        // Goal: with NO fuse/chain env vars set — which is exactly the raw-tests
        // process environment — the raw engine's resolved defaults must be the
        // all-flags-on + chain=8 ("proven-best") configuration. After the flip the
        // env vars become opt-OUT (QWISP_FUSE_X=0 / QWISP_CHAIN_K=0 disable), and the
        // flag-off paths stay reachable via explicit =0 for bisection.
        //
        // These read the SAME production statics/constant that SeedlessFusedForward and
        // Tell consume at runtime — no reimplemented oracle. They are RED on
        // the pre-flip tree (fuse statics are ["QWISP_FUSE_X"] == "1" = false when
        // unset; chainKDefault == 0) and GREEN once the driver flips the defaults
        // (fuse statics → != "0"; chainKDefault → 8).

        // Test 49: fuse flags default ON (opt-out). Reads the production env-resolved
        // statics directly (the same ones encodeGdnLayerRows / encodeAttnLayerRows /
        // fuseGUActive / fuseSHEXPActive branch on). RED now because each resolves to
        // false when its QWISP_FUSE_* var is unset; GREEN after the flip to != "0".
        run("default_fuse_flags_on") {
            let gu    = SeedlessFusedVerify.SeedlessFusedForward.fuseGU
            let gdn   = SeedlessFusedVerify.SeedlessFusedForward.fuseGDN
            let attn  = SeedlessFusedVerify.SeedlessFusedForward.fuseATTN
            let shexp = SeedlessFusedVerify.SeedlessFusedForward.fuseSHEXP
            if !gu    { return (false, "fuseGU default expected ON, got OFF") }
            if !gdn   { return (false, "fuseGDN default expected ON, got OFF") }
            if !attn  { return (false, "fuseATTN default expected ON, got OFF") }
            if !shexp { return (false, "fuseSHEXP default expected ON, got OFF") }
            return (true, "ok")
        }

        // Test 50: chain default K == 8 (opt-out). Tell resolves the chain
        // length as Tell.envInt("QWISP_CHAIN_K", SeedlessFusedForward.chainKDefault); with
        // QWISP_CHAIN_K unset the resolved value equals this seam constant, so pinning
        // the constant to 8 pins the production default. Reads the wired production
        // seam (Tell references chainKDefault) — not a reimplemented parse.
        // Also asserts the opt-out contract via the SAME Tell.envInt production path
        // Tell uses: unset → chainKDefault, explicit "0" → 0 (disabled).
        // RED now because chainKDefault == 0; GREEN after the flip to 8.
        run("default_chain_k_eight") {
            let d = SeedlessFusedVerify.SeedlessFusedForward.chainKDefault
            if d != 8 { return (false, "chainKDefault expected 8, got \(d)") }
            // Unset env resolves to the seam default (raw-tests sets no QWISP_CHAIN_K).
            let resolvedUnset = Tell.envInt("QWISP_CHAIN_K", SeedlessFusedVerify.SeedlessFusedForward.chainKDefault)
            if resolvedUnset != 8 {
                return (false, "unset QWISP_CHAIN_K expected 8, got \(resolvedUnset)")
            }
            return (true, "ok")
        }

        // ── Phase II-b G1 gate: bolt-tier chain wiring (test 51) ─────────────
        //
        // WRITE-LOCKED (locked8): implementer MUST NOT modify this test.
        // Encodes the goal: the BOLT decode loop must be able to advance its D==0 greedy
        // span through the GPU token-feedback chain (chainedStepArgmax) when chainK>0,
        // and doing so MUST be bit-identical to per-step greedy (bolt = deterministic
        // buddy-greedy → chain must never change tokens).
        //
        // Seam under test: Tell.boltGreedyChainSpan (STUB → nil → RED). It is the
        // shared span decoder that runBoltMode delegates its D==0 branch to. The backend's
        // chainedStepArgmax / stepArgmax here are the PRODUCTION SeedlessFusedForward methods on a
        // real bolt engine (providers + setBoltTables + attachHead) — no reimplemented oracle.
        //
        // References:
        //   • bolt engine construction = stream_fused_bolt_exact_table idiom (test with
        //     TestExpertProvider warm-all + frozen slot tables + setBoltTables).
        //   • head + chainedStepArgmax = chained_greedy_bitexact idiom (test 47).
        //   • reference token sequence = K sequential PRODUCTION stepArgmax on a second,
        //     identically-seeded bolt engine.
        //
        // Assertions:
        //   (a) boltGreedyChainSpan returns non-nil (stub nil → RED here).
        //   (b) it USED the chain (per-step stepArgmax NOT called for the chained tail).
        //   (c) packing is correct: emitted[0]==u, emitted.count==chainK.
        //   (d) determinism: emitted + [nextU] == [u] + K sequential bolt greedy tokens
        //       (chain-on OUT_TOKENS byte-identical to chain-off).
        run("bolt_chain_span_wired") {
            guard let (device, _) = SeedlessMetalForward.ensure() else { return (false, "no device") }
            let V = 256
            let C = stE                    // C = E: all experts fit, deterministic slot table
            let chainK = 4
            let firstToken = Int32(7)

            // Shared layer weights (same objects → chain and ref engines are identical models).
            let moe0 = stMkMoE(), moe1 = stMkMoE()
            let gdnW = stMkGdnW(), attnW = stMkAttnW()
            let iLN0 = MLXRandom.normal([stH]).asType(.float16)
            let pLN0 = MLXRandom.normal([stH]).asType(.float16)
            let iLN1 = MLXRandom.normal([stH]).asType(.float16)
            let pLN1 = MLXRandom.normal([stH]).asType(.float16)
            MLX.eval([iLN0, pLN0, iLN1, pLN1])
            let layerSpecs = [
                SeedlessVerifyForward.LayerSpec(isLinear: true,
                    inputLN: iLN0, postLN: pLN0,
                    gdn: gdnW, attn: nil, moe: moe0.w, moeE: stE, moeI: stI),
                SeedlessVerifyForward.LayerSpec(isLinear: false,
                    inputLN: iLN1, postLN: pLN1,
                    gdn: nil, attn: attnW, moe: moe1.w, moeE: stE, moeI: stI),
            ]
            // Initial cache state (8 KV positions) — each engine gets its own copy.
            let cs0 = MLXRandom.normal([stCK - 1, stConvDim]).asType(.float16)
            let rs0 = MLXRandom.normal([1, stHv, stDv, stDk]).asType(.float32)
            let kc0 = MLXRandom.normal([stNkv, 8, stHd]).asType(.float16)
            let vc0 = MLXRandom.normal([stNkv, 8, stHd]).asType(.float16)
            MLX.eval([cs0, rs0, kc0, vc0])
            func freshC() -> [SeedlessVerifyForward.LayerCaches] {
                [SeedlessVerifyForward.LayerCaches(convState: cs0, recState: rs0),
                 SeedlessVerifyForward.LayerCaches(kCache: kc0, vCache: vc0)]
            }
            // Shared head weights (embed = lm_head = 4-bit q), same objects for both engines.
            let ewf = MLXRandom.normal([V, stH]).asType(.float16)
            let (ewq, esc, ebiOpt) = MLX.quantized(ewf, groupSize: 64, bits: 4, mode: .affine)
            guard let ebi = ebiOpt else { return (false, "embed biases nil") }
            let lwf = MLXRandom.normal([V, stH]).asType(.float16)
            let (lwq, lsc, lbiOpt) = MLX.quantized(lwf, groupSize: 64, bits: 4, mode: .affine)
            guard let lbi = lbiOpt else { return (false, "lm biases nil") }
            let fnW = MLXRandom.normal([stH]).asType(.float16)
            MLX.eval([ewq, esc, ebi, lwq, lsc, lbi, fnW])
            let maxSeq = 32

            // Build a bolt engine: providers (warm all E) + frozen slot tables + head.
            func mkBoltEngine() -> SeedlessFusedVerify.SeedlessFusedForward? {
                guard let tp0 = TestExpertProvider(E: stE, I: stI, H: stH, C: C,
                        gW: moe0.gW, gSf16: moe0.gSf16, gBf16: moe0.gBf16,
                        uW: moe0.uW, uSf16: moe0.uSf16, uBf16: moe0.uBf16,
                        dW: moe0.dW, dSf16: moe0.dSf16, dBf16: moe0.dBf16, device: device),
                      let tp1 = TestExpertProvider(E: stE, I: stI, H: stH, C: C,
                        gW: moe1.gW, gSf16: moe1.gSf16, gBf16: moe1.gBf16,
                        uW: moe1.uW, uSf16: moe1.uSf16, uBf16: moe1.uBf16,
                        dW: moe1.dW, dSf16: moe1.dSf16, dBf16: moe1.dBf16, device: device)
                else { return nil }
                let sm0 = tp0.ensure(Array(0..<stE)), sm1 = tp1.ensure(Array(0..<stE))
                var tbl0 = [Int32](repeating: 0, count: stE), tbl1 = [Int32](repeating: 0, count: stE)
                for (e, s) in sm0 { tbl0[e] = Int32(s) }
                for (e, s) in sm1 { tbl1[e] = Int32(s) }
                guard let eng = SeedlessFusedVerify.SeedlessFusedForward(
                        layers: layerSpecs, caches: freshC(), maxM: 4, H: stH,
                        maxSeqLen: maxSeq, providers: [tp0, tp1])
                else { return nil }
                eng.setBoltTables([tbl0, tbl1])   // → streamMode == .bolt
                guard eng.attachHead(embedW: ewq, embedS: esc, embedB: ebi,
                                     lmW: lwq, lmS: lsc, lmB: lbi,
                                     fnW: fnW, vocab: V)
                else { return nil }
                return eng
            }
            guard let bltEng = mkBoltEngine() else { return (false, "bolt chain engine nil") }
            guard let refEng = mkBoltEngine() else { return (false, "bolt ref engine nil") }

            // Reference: K sequential PRODUCTION stepArgmax on the ref bolt engine.
            var refSeq: [Int] = []
            var cur = firstToken
            for _ in 0 ..< chainK {
                guard let t = refEng.stepArgmax([cur]) else { return (false, "ref stepArgmax nil") }
                refSeq.append(t[0]); cur = Int32(t[0])
            }

            // Instrument a SpecBackend wrapping the chain bolt engine. stepArgmax counts
            // per-step calls so we can prove the chained tail did NOT go per-step.
            var perStepCalls = 0
            var backend = Tell.SpecBackend(
                forward: { _ in nil },     // not used by the greedy chain span
                stepArgmax: { toks in perStepCalls += 1; return bltEng.stepArgmax(toks) },
                snapshot: { bltEng.snapshot() },
                rollback: { _ in })
            backend.chainedStepArgmax = { token, k in bltEng.chainedStepArgmax(token, K: k) }

            // Under test: bolt-runner greedy span with budget = chainK (emit u + K-1 tail).
            guard let (emitted, nextU) = Tell.boltGreedyChainSpan(
                backend: backend, u: Int(firstToken), chainK: chainK, budget: chainK)
            else { return (false, "not implemented (boltGreedyChainSpan nil)") }

            // (b) chain used — no per-step fallback for the chained tail.
            if perStepCalls != 0 {
                return (false, "chain not used: perStepCalls=\(perStepCalls) (want 0)")
            }
            // (c) packing.
            if emitted.first != Int(firstToken) {
                return (false, "emitted[0]=\(String(describing: emitted.first)) want u=\(firstToken)")
            }
            if emitted.count != chainK {
                return (false, "emitted.count=\(emitted.count) want chainK=\(chainK)")
            }
            // (d) determinism: chain-on tokens == chain-off sequential tokens.
            let chainAll = emitted + [nextU]
            let refAll = [Int(firstToken)] + refSeq
            if chainAll != refAll {
                return (false, "chain tokens \(chainAll) != sequential \(refAll)")
            }
            return (true, "ok")
        }

        // ── II-C G1: MoE combine_rows → S2 fold, M==1-GATED (test 52) ──────
        //
        // WRITE-LOCKED: implementer MUST NOT modify this test.
        // Encodes the M==1 gate for fuseMOE2 (same register-pressure doctrine as
        // fuseGU / fuseSHEXP-S1, which already gate on M==1): the combine→S2 fold
        // saves a dispatch at M==1 (decode / code / agentic regimes) but REGRESSES
        // the M>1 verify batches (inlining combine into S2's grid loses the
        // parallelism combine_rows had at M>1). GOAL: fold applies ONLY at M==1;
        // verify batches (M>1) keep the separate combine_rows + non-fold S2 path.
        //
        // Reference = fold-OFF path (combine_rows separate + S2 reads sc.y) via the
        // production encodeMoEBlockRows (fuseMOE2Enabled=false). Candidate = fold-ON
        // flag (fuseMOE2Enabled=true). The test sweeps M∈{1,8} and asserts the
        // M-DEPENDENT dispatch contract below.
        //
        // Dispatch contract (encodeMoEGatherRowsRange fires combine exactly once per
        // fusedMoEBlockRows call — single range r0:0..r1:M):
        //   M==1 fold-ON  → combine dispatch SKIPPED  → _combineRowsDispatchCount == 0
        //   M==8 fold-ON  → combine dispatch RUNS      → _combineRowsDispatchCount == 1
        //                   (fold INACTIVE at M>1; separate path preserved)
        // AND: moeOut byte-identical to the fold-OFF reference in BOTH M cases
        //   (M==1 via the fold kernel; M==8 via the unchanged separate combine path).
        //
        // RED gate: the CURRENT tree folds at ALL M (encodeMoEGatherRowsRange skips
        // combine whenever fuseMOE2Enabled && fuseSHEXP, with no M==1 guard; S2
        // dispatches the fold kernel at every M). So at M==8 fold-ON the current code
        // yields _combineRowsDispatchCount == 0, but this test requires 1 → FAIL.
        // GREEN when the implementer adds M==1 to BOTH the combine-skip guard
        // (encodeMoEGatherRowsRange) and the S2 fold-kernel selection (encodeMoESharedRows).
        //
        // Adversarial cases:
        //   M=1: exercises fuseSHEXP M=1 path (S1 fused swiglu) + S2 fold; combine fold
        //        must not corrupt the sc.d/sc.scores→moeOut pipeline (bit-exact).
        //   M=8: exercises the M>1 verify batch — fold MUST NOT fire; separate
        //        combine_rows + non-fold S2 must run and stay bit-exact.
        //
        // Seam: fuseMOE2Enabled (flag, default ON from env) + _combineRowsDispatchCount
        //       (incremented in encodeCombineRows; M-dependent expected value).
        run("moe_combine_fold_bitexact") {
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
            let (gW, gS, gB)   = q8(E, H); let (sgW, sgS, sgB) = q8(8, H)
            let (a0, a1, a2)   = q4e(E, I, H); let (b0, b1, b2) = q4e(E, I, H); let (c0, c1, c2) = q4e(E, H, I)
            let (d0, d1, d2)   = q4(I, H); let (e0, e1, e2) = q4(I, H); let (f0, f1, f2) = q4(H, I)
            let w = SeedlessVerifyForward.MoEBlockW(gateWq: gW, gateSc: gS, gateBi: gB,
                swGWq: a0, swGSc: a1, swGBi: a2, swUWq: b0, swUSc: b1, swUBi: b2,
                swDWq: c0, swDSc: c1, swDBi: c2, shGWq: d0, shGSc: d1, shGBi: d2,
                shUWq: e0, shUSc: e1, shUBi: e2, shDWq: f0, shDSc: f1, shDBi: f2,
                sharedGateWq: sgW, sharedGateSc: sgS, sharedGateBi: sgB)

            // Snapshot of flags we toggle, restored unconditionally at the end.
            let savedMOE2  = SeedlessFusedVerify.SeedlessFusedForward.fuseMOE2Enabled
            let savedSHEXP = SeedlessFusedVerify.SeedlessFusedForward.fuseSHEXP

            // Ensure S2 path is active for both arms (fold applies only when fuseSHEXP is ON).
            SeedlessFusedVerify.SeedlessFusedForward.fuseSHEXP = true

            for M in [1, 8] {
                // M==1 folds (dispatch skipped); M>1 keeps the separate combine dispatch.
                let expectedCount = (M == 1) ? 0 : 1

                let x = MLXRandom.normal([M, H]).asType(.float16); x.eval()

                // ── Reference (fold-OFF): combine_rows separate dispatch + S2 reads sc.y ──
                SeedlessFusedVerify.SeedlessFusedForward.fuseMOE2Enabled = false
                guard let ref = SeedlessFusedVerify.fusedMoEBlockRows(x, w, M: M, E: E, I: I, Ktop: Ktop)
                else {
                    SeedlessFusedVerify.SeedlessFusedForward.fuseMOE2Enabled = savedMOE2
                    SeedlessFusedVerify.SeedlessFusedForward.fuseSHEXP       = savedSHEXP
                    return (false, "fold-OFF ref nil M=\(M)")
                }
                ref.eval()

                // ── Candidate (fold-ON flag): M==1 folds, M>1 keeps the separate path ──
                SeedlessFusedVerify.SeedlessFusedForward.fuseMOE2Enabled = true
                SeedlessFusedVerify._combineRowsDispatchCount = 0   // reset before fold-ON call
                guard let got = SeedlessFusedVerify.fusedMoEBlockRows(x, w, M: M, E: E, I: I, Ktop: Ktop)
                else {
                    SeedlessFusedVerify.SeedlessFusedForward.fuseMOE2Enabled = savedMOE2
                    SeedlessFusedVerify.SeedlessFusedForward.fuseSHEXP       = savedSHEXP
                    return (false, "fold-ON nil M=\(M)")
                }
                got.eval()
                let dispatchCount = SeedlessFusedVerify._combineRowsDispatchCount

                // (a) moeOut byte-identical to fold-OFF reference (both M cases).
                //     M==1: through the fold kernel. M==8: through the unchanged path.
                let (okOut, dOut) = bitEqual(got, ref)
                if !okOut {
                    SeedlessFusedVerify.SeedlessFusedForward.fuseMOE2Enabled = savedMOE2
                    SeedlessFusedVerify.SeedlessFusedForward.fuseSHEXP       = savedSHEXP
                    return (false, "M=\(M) out mismatch: \(dOut)")
                }

                // (b) M-dependent combine_rows dispatch count.
                //     RED on current tree: M==8 folds (count 0) but contract wants 1.
                //     GREEN when fold is M==1-gated (skip at M==1, keep at M>1).
                if dispatchCount != expectedCount {
                    SeedlessFusedVerify.SeedlessFusedForward.fuseMOE2Enabled = savedMOE2
                    SeedlessFusedVerify.SeedlessFusedForward.fuseSHEXP       = savedSHEXP
                    return (false,
                            "M=\(M) combine_rows dispatch count=\(dispatchCount) want \(expectedCount) " +
                            "(fold must be M==1-gated: skip at M==1, keep separate combine at M>1)")
                }
            }

            SeedlessFusedVerify.SeedlessFusedForward.fuseMOE2Enabled = savedMOE2
            SeedlessFusedVerify.SeedlessFusedForward.fuseSHEXP       = savedSHEXP
            return (true, "ok")
        }

        // ── recon #16 tier-gating product-correctness (tests 52-53) ─────────
        //
        // WRITE-LOCKED: implementer MUST NOT modify tests 52-53.
        //
        // Goal: two lossless default-path wiring fixes in Tell —
        //   (1) resident useFused defaults ON (fused 1-CB ~92 tok/s), not the
        //       composedBackend footgun (~1.3 tok/s); QWISP_RAW_FUSED=0 restores
        //       composed for debug.
        //   (2) raw-spec C, when QWISP_RAW_C is unset, consults the RAM tier via
        //       DeviceCalibration.defaultC() (8→64/16→128/24→192/32+→256); an
        //       explicit QWISP_RAW_C still overrides.
        // Both are output-invariant (fused ≡ composed greedy byte-identical; C only
        // changes streaming footprint, not tokens). Encoded via pure production
        // seams Tell.resolveUseFused / resolveRawC (STUB → RED now).

        // Test 52: resident useFused default resolves ON when QWISP_RAW_FUSED unset.
        // Seam = Tell.resolveUseFused (STUB returns false → RED). Also pins
        // the opt-out contract ("0" → composed) and the pass-through ("1" → fused),
        // and binds to the live production env value (raw-tests sets no QWISP_RAW_FUSED
        // → unset → must be true).
        run("raw_resident_fused_default_on") {
            // unset → fused ON (the fix; STUB false → RED here)
            if Tell.resolveUseFused(env: nil) != true {
                return (false, "unset QWISP_RAW_FUSED must resolve useFused=true (fused default ON)")
            }
            // explicit "0" → composed (debug opt-out preserved)
            if Tell.resolveUseFused(env: "0") != false {
                return (false, "QWISP_RAW_FUSED=0 must resolve useFused=false (composed)")
            }
            // explicit "1" → fused
            if Tell.resolveUseFused(env: "1") != true {
                return (false, "QWISP_RAW_FUSED=1 must resolve useFused=true (fused)")
            }
            // live production env (raw-tests sets no QWISP_RAW_FUSED → unset → true)
            let live = Tell.resolveUseFused(
                env: ProcessInfo.processInfo.environment["QWISP_RAW_FUSED"])
            if live != true {
                return (false, "live unset QWISP_RAW_FUSED must resolve useFused=true, got \(live)")
            }
            return (true, "ok")
        }

        // Test 53: raw-spec C consults DeviceCalibration.defaultC() when QWISP_RAW_C
        // unset, and an explicit value overrides. Seam = Tell.resolveRawC
        // (STUB returns -1 → RED). References the PRODUCTION tier fn (no reimplemented
        // oracle): the unset result must equal DeviceCalibration.defaultC().
        run("raw_c_tier_default_wired") {
            let tierC = DeviceCalibration.defaultC()
            // unset → tiered default (the fix; STUB -1 → RED here)
            if Tell.resolveRawC(envC: nil, defaultC: tierC) != tierC {
                return (false, "unset QWISP_RAW_C must consult DeviceCalibration.defaultC()=\(tierC)")
            }
            // arbitrary tier default is honored when unset (not hard-coded to resident)
            if Tell.resolveRawC(envC: nil, defaultC: 128) != 128 {
                return (false, "unset QWISP_RAW_C must return the passed defaultC (128)")
            }
            // explicit streaming C overrides the tier default
            if Tell.resolveRawC(envC: "64", defaultC: 256) != 64 {
                return (false, "explicit QWISP_RAW_C=64 must override defaultC")
            }
            // explicit "0" (resident) overrides too (distinct from unset)
            if Tell.resolveRawC(envC: "0", defaultC: 128) != 0 {
                return (false, "explicit QWISP_RAW_C=0 must override to resident (0), not defaultC")
            }
            // live production env (raw-tests sets no QWISP_RAW_C → unset → tiered)
            let live = Tell.resolveRawC(
                envC: ProcessInfo.processInfo.environment["QWISP_RAW_C"], defaultC: tierC)
            if live != tierC {
                return (false, "live unset QWISP_RAW_C must resolve to defaultC=\(tierC), got \(live)")
            }
            return (true, "ok")
        }

        // ── gqmm3 correctness gate (tests 55-56) — notes/11 §Locked test ─────────────
        //
        // WRITE-LOCKED: implementer MUST NOT modify these tests.
        // They encode the G1 acceptance gate from notes/11-gqmm3-3bit-tier-spec.md §'Locked test'.
        //
        // Oracle = MLX.quantizedMatmul(bits:3, mode:.affine, groupSize:64) per selected expert row
        // — NOT a self-loop. Both tests gate the two stub APIs:
        //   gqmm3      (M=1 single-row, analogous to gatherQmm)
        //   gqmm3Rows  (M∈{1,2,9,17,25} multi-row, analogous to gatherQmmRows)
        //
        // TWO mandatory dtype cases (spec §'Locked test' item 4):
        //   Test 55 (f16):  wf/scales/biases = float16  (standard path)
        //   Test 56 (bf16): wf = bfloat16 → scales/biases come out bfloat16
        //                   (real UD-model path — forces templated scale dtype in kernel; the
        //                    #1 correctness lever per spec §Ground truth)
        //
        // Geometry: E=64, K=2048, N=512, Ktop=4 (matches real expert shapes).
        // Expert-index pool: max index = 62 < E=64 ✓ (same pool as gqmm4_rows_bitexact).
        // Both tests RED because gqmm3 / gqmm3Rows return nil (STUB).

        // Helper: fixed expert-index pool (all < E=64, same as gqmm4_rows_bitexact).
        let gqmm3Pool: [Int32] = [3, 17, 40, 62,  0, 10, 25, 50, 33,  7,
                                   55, 20, 41, 15,  2, 60,  8, 30, 11, 45,
                                   22, 38, 61,  5, 18, 44, 29,  1, 36, 52,
                                    6, 19,  9, 27, 48, 14, 37,  4, 23, 53,
                                   12, 16, 31, 39, 47, 57, 13, 21, 32, 49]

        // Test 55: gqmm3 + gqmm3Rows bit-exact against MLX oracle — float16 scales.
        // Oracle: MLX.quantizedMatmul(bits:3) per expert (one call per (row,expert) slot).
        // Goes RED at first gqmm3 nil check.
        run("gqmm3_rows_bitexact_f16") {
            let E = 64, K = 2048, N = 512, Ktop = 4
            // f16 weights → f16 scales/biases from MLX.quantized
            let wf = MLXRandom.normal([E, N, K]).asType(.float16)
            let (wq, sc, biOpt) = MLX.quantized(wf, groupSize: 64, bits: 3, mode: .affine)
            guard let bi = biOpt else { return (false, "biases nil") }
            MLX.eval([wq, sc, bi])

            // ── gqmm3 (M=1) vs MLX oracle ──
            let x1 = MLXRandom.normal([1, K]).asType(.float16); x1.eval()
            let inds1Arr: [Int32] = [3, 17, 40, 62]   // Ktop=4, all < E
            let inds1 = MLXArray(inds1Arr, [Ktop]); inds1.eval()
            // Oracle: one MLX.quantizedMatmul call per expert slot → concat [Ktop, N]
            var oracle1Parts: [MLXArray] = []
            for ki in 0..<Ktop {
                let e = Int(inds1Arr[ki])
                let r = MLX.quantizedMatmul(x1, wq[e], scales: sc[e], biases: bi[e],
                                            transpose: true, groupSize: 64, bits: 3, mode: .affine)
                r.eval(); oracle1Parts.append(r)
            }
            let oracle1 = MLX.concatenated(oracle1Parts, axis: 0); oracle1.eval()   // [Ktop, N]
            // Stub — goes RED here (gqmm3 returns nil)
            guard let got1 = SeedlessMetalForward.gqmm3(x1, wq, scales: sc, biases: bi,
                                                     inds: inds1, Ktop: Ktop, K: K, N: N)
            else { return (false, "gqmm3 not implemented (M=1 f16)") }
            got1.eval()
            let (ok1, d1) = bitEqual(got1, oracle1)
            if !ok1 { return (false, "gqmm3 M=1 f16: \(d1)") }

            // ── gqmm3Rows (M∈{1,2,9,17,25}) vs MLX oracle ──
            for M in [1, 2, 9, 17, 25] {
                let x = MLXRandom.normal([M, K]).asType(.float16)
                let indsFlat = (0..<M*Ktop).map { gqmm3Pool[$0 % gqmm3Pool.count] }
                let inds = MLXArray(indsFlat, [M * Ktop])
                MLX.eval([x, inds])
                // Oracle: per-(row,expert) MLX quantizedMatmul → concat [M*Ktop, N]
                var refParts: [MLXArray] = []
                for m in 0..<M {
                    let xm = x[m ..< m+1]   // [1, K]
                    xm.eval()
                    for ki in 0..<Ktop {
                        let e = Int(indsFlat[m * Ktop + ki])
                        let r = MLX.quantizedMatmul(xm, wq[e], scales: sc[e], biases: bi[e],
                                                    transpose: true, groupSize: 64, bits: 3, mode: .affine)
                        r.eval(); refParts.append(r)   // [1, N]
                    }
                }
                let ref = MLX.concatenated(refParts, axis: 0); ref.eval()   // [M*Ktop, N]
                guard let got = SeedlessMetalForward.gqmm3Rows(x, wq, scales: sc, biases: bi,
                                                           inds: inds, M: M, Ktop: Ktop, K: K, N: N)
                else { return (false, "gqmm3Rows not implemented (M=\(M) f16)") }
                got.eval()
                let (ok, d) = bitEqual(got, ref)
                if !ok { return (false, "gqmm3Rows M=\(M) f16: \(d)") }
            }
            return (true, "ok")
        }

        // Test 56: gqmm3 + gqmm3Rows bit-exact against MLX oracle — bfloat16 scales.
        // wf=bfloat16 → MLX.quantized returns bfloat16 scales/biases (real UD-model path).
        // This case forces the kernel to template scale dtype (NOT downcast bf16→f16).
        // Structure is identical to test 55 — differs ONLY in the dtype of wf.
        // Goes RED at first gqmm3 nil check.
        run("gqmm3_rows_bitexact_bf16") {
            let E = 64, K = 2048, N = 512, Ktop = 4
            // bf16 weights → bf16 scales/biases from MLX.quantized (real UD-model dtype path)
            let wf = MLXRandom.normal([E, N, K]).asType(.bfloat16)
            let (wq, sc, biOpt) = MLX.quantized(wf, groupSize: 64, bits: 3, mode: .affine)
            guard let bi = biOpt else { return (false, "biases nil") }
            MLX.eval([wq, sc, bi])

            // ── gqmm3 (M=1) vs MLX oracle ──
            let x1 = MLXRandom.normal([1, K]).asType(.float16); x1.eval()
            let inds1Arr: [Int32] = [3, 17, 40, 62]   // Ktop=4, all < E
            let inds1 = MLXArray(inds1Arr, [Ktop]); inds1.eval()
            // Oracle: MLX.quantizedMatmul handles bf16 scales natively — pins kernel to MLX
            var oracle1Parts: [MLXArray] = []
            for ki in 0..<Ktop {
                let e = Int(inds1Arr[ki])
                let r = MLX.quantizedMatmul(x1, wq[e], scales: sc[e], biases: bi[e],
                                            transpose: true, groupSize: 64, bits: 3, mode: .affine)
                r.eval(); oracle1Parts.append(r)
            }
            let oracle1 = MLX.concatenated(oracle1Parts, axis: 0); oracle1.eval()   // [Ktop, N]
            // Stub — goes RED here (gqmm3 returns nil)
            guard let got1 = SeedlessMetalForward.gqmm3(x1, wq, scales: sc, biases: bi,
                                                     inds: inds1, Ktop: Ktop, K: K, N: N)
            else { return (false, "gqmm3 not implemented (M=1 bf16)") }
            got1.eval()
            let (ok1, d1) = bitEqual(got1, oracle1)
            if !ok1 { return (false, "gqmm3 M=1 bf16: \(d1)") }

            // ── gqmm3Rows (M∈{1,2,9,17,25}) vs MLX oracle ──
            for M in [1, 2, 9, 17, 25] {
                let x = MLXRandom.normal([M, K]).asType(.float16)
                let indsFlat = (0..<M*Ktop).map { gqmm3Pool[$0 % gqmm3Pool.count] }
                let inds = MLXArray(indsFlat, [M * Ktop])
                MLX.eval([x, inds])
                // Oracle: per-(row,expert) MLX quantizedMatmul — bf16 scales natively handled
                var refParts: [MLXArray] = []
                for m in 0..<M {
                    let xm = x[m ..< m+1]   // [1, K]
                    xm.eval()
                    for ki in 0..<Ktop {
                        let e = Int(indsFlat[m * Ktop + ki])
                        let r = MLX.quantizedMatmul(xm, wq[e], scales: sc[e], biases: bi[e],
                                                    transpose: true, groupSize: 64, bits: 3, mode: .affine)
                        r.eval(); refParts.append(r)   // [1, N]
                    }
                }
                let ref = MLX.concatenated(refParts, axis: 0); ref.eval()   // [M*Ktop, N]
                guard let got = SeedlessMetalForward.gqmm3Rows(x, wq, scales: sc, biases: bi,
                                                           inds: inds, M: M, Ktop: Ktop, K: K, N: N)
                else { return (false, "gqmm3Rows not implemented (M=\(M) bf16)") }
                got.eval()
                let (ok, d) = bitEqual(got, ref)
                if !ok { return (false, "gqmm3Rows M=\(M) bf16: \(d)") }
            }
            return (true, "ok")
        }

        // Test 56b (W1, notes/18-mixed-precision-residency-spec.md): gqmm2 + gqmm2Rows
        // bit-exact against MLX bits=2 oracle — float16 scales ONLY.
        // The real 2-bit residency artifact is F16-scaled; the bf16 UD-model path does not
        // exist for 2-bit, so (deliberate deviation from the gqmm3 pair) there is no bf16 case.
        // Oracle: MLX.quantizedMatmul(bits:2) per (row,expert) slot — NOT a self-loop.
        // Goes RED at first gqmm2 nil check (STUB — W1 RED).
        run("gqmm2_rows_bitexact_f16") {
            let E = 64, K = 2048, N = 512, Ktop = 4
            // Same 50-entry expert-index pool as gqmm3 (all < E=64).
            let gqmm2Pool: [Int32] = [3, 17, 40, 62,  0, 10, 25, 50, 33,  7,
                                       55, 20, 41, 15,  2, 60,  8, 30, 11, 45,
                                       22, 38, 61,  5, 18, 44, 29,  1, 36, 52,
                                        6, 19,  9, 27, 48, 14, 37,  4, 23, 53,
                                       12, 16, 31, 39, 47, 57, 13, 21, 32, 49]
            // f16 weights → f16 scales/biases from MLX.quantized (bits=2 affine)
            let wf = MLXRandom.normal([E, N, K]).asType(.float16)
            let (wq, sc, biOpt) = MLX.quantized(wf, groupSize: 64, bits: 2, mode: .affine)
            guard let bi = biOpt else { return (false, "biases nil") }
            MLX.eval([wq, sc, bi])

            // ── gqmm2 (M=1) vs MLX oracle ──
            let x1 = MLXRandom.normal([1, K]).asType(.float16); x1.eval()
            let inds1Arr: [Int32] = [3, 17, 40, 62]   // Ktop=4, all < E
            let inds1 = MLXArray(inds1Arr, [Ktop]); inds1.eval()
            var oracle1Parts: [MLXArray] = []
            for ki in 0..<Ktop {
                let e = Int(inds1Arr[ki])
                let r = MLX.quantizedMatmul(x1, wq[e], scales: sc[e], biases: bi[e],
                                            transpose: true, groupSize: 64, bits: 2, mode: .affine)
                r.eval(); oracle1Parts.append(r)
            }
            let oracle1 = MLX.concatenated(oracle1Parts, axis: 0); oracle1.eval()   // [Ktop, N]
            // Stub — goes RED here (gqmm2 returns nil)
            guard let got1 = SeedlessMetalForward.gqmm2(x1, wq, scales: sc, biases: bi,
                                                     inds: inds1, Ktop: Ktop, K: K, N: N)
            else { return (false, "gqmm2 not implemented (M=1 f16)") }
            got1.eval()
            let (ok1, d1) = bitEqual(got1, oracle1)
            if !ok1 { return (false, "gqmm2 M=1 f16: \(d1)") }

            // ── gqmm2Rows (M∈{1,2,9,17,25}) vs MLX oracle ──
            for M in [1, 2, 9, 17, 25] {
                let x = MLXRandom.normal([M, K]).asType(.float16)
                let indsFlat = (0..<M*Ktop).map { gqmm2Pool[$0 % gqmm2Pool.count] }
                let inds = MLXArray(indsFlat, [M * Ktop])
                MLX.eval([x, inds])
                var refParts: [MLXArray] = []
                for m in 0..<M {
                    let xm = x[m ..< m+1]   // [1, K]
                    xm.eval()
                    for ki in 0..<Ktop {
                        let e = Int(indsFlat[m * Ktop + ki])
                        let r = MLX.quantizedMatmul(xm, wq[e], scales: sc[e], biases: bi[e],
                                                    transpose: true, groupSize: 64, bits: 2, mode: .affine)
                        r.eval(); refParts.append(r)   // [1, N]
                    }
                }
                let ref = MLX.concatenated(refParts, axis: 0); ref.eval()   // [M*Ktop, N]
                guard let got = SeedlessMetalForward.gqmm2Rows(x, wq, scales: sc, biases: bi,
                                                           inds: inds, M: M, Ktop: Ktop, K: K, N: N)
                else { return (false, "gqmm2Rows not implemented (M=\(M) f16)") }
                got.eval()
                let (ok, d) = bitEqual(got, ref)
                if !ok { return (false, "gqmm2Rows M=\(M) f16: \(d)") }
            }
            return (true, "ok")
        }

        // ── G-A: expert-reuse rerank unit tests (57-60) ──────────────────────
        //
        // WRITE-LOCKED: implementer MUST NOT modify these tests.
        // They encode the §3 G-A acceptance gate from notes/10-expert-reuse-rerank-spec.md.
        //
        // All 4 tests are model-free (no weights loaded). They gate the stub APIs:
        //   ReuseContext.observe  (row→token expert attribution, mutating)
        //   ReuseContext.reuseScore  (resident-overlap score)
        //   Tell.suffixDraft(…, reuseCtx:)  (rerank-aware draft selection)
        //
        // Stubs (implementation pending):
        //   ReuseContext.observe = no-op
        //   ReuseContext.reuseScore = 0.0 (neutral)
        //   suffixDraft(reuseCtx: nonNil) = [] (not implemented)
        // All 4 tests are RED on the stub tree and GREEN after correct implementation.
        // Existing 56 tests are additive-only and unchanged.

        // Test 57 (G-A-1): suffixDraft α=0 恒等 (identity).
        // reuseCtx:nil and reuseCtx:(…, alpha:0) must return byte-identical draft arrays.
        // Verifies α=0 is a strict generalisation of nil — reranking is exactly disabled.
        // Seq [5,7,11,5,7,13,5,7]: pattern [5,7] matches at pos 0(→11) and pos 3(→13);
        // tie-break picks 13 (most recent), nil draft = [13,5]. Stub non-nil → [] → RED.
        run("rerank_alpha_zero_identity") {
            let seq = [5, 7, 11, 5, 7, 13, 5, 7]
            let maxMatch = 3, draftK = 2, minMatch = 2
            // nil path — reference result (must be non-empty for the test to be meaningful)
            let draftNil = Tell.suffixDraft(seq, maxMatch: maxMatch, draftK: draftK,
                                            minMatch: minMatch, reuseCtx: nil)
            if draftNil.isEmpty {
                return (false, "nil draft unexpectedly empty on canonical seq \(seq)")
            }
            // alpha=0 path — must be byte-identical to nil
            var ctx = ReuseContext()
            ctx.observe(rowTokens: [11, 13], layer: 0, inds: [0, 1, 2, 3, 4, 5, 6, 7], Ktop: 4)
            let residentAny: [Set<Int>] = [Set([0, 1, 2, 3])]
            let draftAlpha0 = Tell.suffixDraft(seq, maxMatch: maxMatch, draftK: draftK,
                                               minMatch: minMatch,
                                               reuseCtx: (ctx: ctx,
                                                          residentPerLayer: residentAny,
                                                          alpha: 0.0))
            if draftNil != draftAlpha0 {
                return (false, "α=0 not byte-identical to nil: nil=\(draftNil) α0=\(draftAlpha0)")
            }
            return (true, "ok")
        }

        // Test 58 (G-A-2): reuseScore 単調性 (monotonicity and exact value).
        // reuseScore(t, residentPerLayer) = Σ_li |tokenExperts[t][li] ∩ residentPerLayer[li]|.
        // Token 42: layer 0 experts {0,1,2,3}, layer 1 experts {8,9,10,11}.
        //   full resident: score = Ktop*nLayers = 8; partial: 4; none: 0.
        // Stub reuseScore=0.0 → 0==0 → monotonicity fails → RED.
        run("reuse_score_monotonic") {
            let Ktop = 4, nLayers = 2
            var ctx = ReuseContext()
            ctx.observe(rowTokens: [42], layer: 0, inds: [0, 1, 2, 3], Ktop: Ktop)
            ctx.observe(rowTokens: [42], layer: 1, inds: [8, 9, 10, 11], Ktop: Ktop)
            // Full overlap: resident exactly equals observed experts
            let resFull: [Set<Int>] = [Set([0, 1, 2, 3]), Set([8, 9, 10, 11])]
            let scoreFull = ctx.reuseScore(token: 42, residentPerLayer: resFull)
            // Partial overlap: half experts match per layer (2+2=4)
            let resPartial: [Set<Int>] = [Set([0, 1, 20, 21]), Set([8, 9, 30, 31])]
            let scorePartial = ctx.reuseScore(token: 42, residentPerLayer: resPartial)
            // No overlap: entirely disjoint resident sets
            let resNone: [Set<Int>] = [Set([50, 51, 52, 53]), Set([60, 61, 62, 63])]
            let scoreNone = ctx.reuseScore(token: 42, residentPerLayer: resNone)
            // Monotonicity: full > partial > none
            if scoreFull <= scorePartial {
                return (false,
                        "monotonicity fail: full(\(scoreFull)) not > partial(\(scorePartial))")
            }
            if scorePartial <= scoreNone {
                return (false,
                        "monotonicity fail: partial(\(scorePartial)) not > none(\(scoreNone))")
            }
            // Exact-value correctness
            let expected = Double(Ktop * nLayers)   // 8
            if scoreFull != expected {
                return (false,
                        "full-overlap score=\(scoreFull) expected \(expected) " +
                        "(Ktop=\(Ktop) × nLayers=\(nLayers))")
            }
            if scoreNone != 0.0 {
                return (false, "no-overlap score=\(scoreNone) expected 0.0")
            }
            return (true, "ok")
        }

        // Test 59 (G-A-3): tie-break — α=0 defers to existing most-recent tie-break;
        //   α>0 promotes the high-reuseScore token.
        // Seq [10,20,100,10,20,200,10,20]: pattern [10,20] matches at pos 0(→100) and
        //   pos 3(→200). counts: 100=1, 200=1 (exact tie). Most-recent=pos 3 → nil picks 200.
        //   reuseScore(100)=4 (full resident), reuseScore(200)=0 → α>0 picks 100.
        // Stub non-nil → [] → all assertions fail → RED.
        run("rerank_tie_break") {
            let seq = [10, 20, 100, 10, 20, 200, 10, 20]
            let Ktop = 4
            // Verify nil gives [200] (most-recent tie-break: pos 3 is newer than pos 0)
            let nilDraft = Tell.suffixDraft(seq, maxMatch: 3, draftK: 1, minMatch: 2,
                                            reuseCtx: nil)
            if nilDraft != [200] {
                return (false,
                        "nil draft expected [200] (most-recent tie-break), got \(nilDraft)")
            }
            // ReuseContext: token 100 → full resident overlap; token 200 → none
            var ctx = ReuseContext()
            ctx.observe(rowTokens: [100], layer: 0, inds: [0, 1, 2, 3], Ktop: Ktop)
            ctx.observe(rowTokens: [200], layer: 0, inds: [10, 11, 12, 13], Ktop: Ktop)
            let resident: [Set<Int>] = [Set([0, 1, 2, 3])]   // overlaps 100 fully, 200 not at all
            // α=0: must be byte-identical to nil → [200]
            let alpha0Draft = Tell.suffixDraft(seq, maxMatch: 3, draftK: 1, minMatch: 2,
                                               reuseCtx: (ctx: ctx,
                                                          residentPerLayer: resident,
                                                          alpha: 0.0))
            if alpha0Draft != [200] {
                return (false,
                        "α=0 expected [200] (identical to nil tie-break), got \(alpha0Draft)")
            }
            // α>0: high-reuse token 100 must win
            let alpha1Draft = Tell.suffixDraft(seq, maxMatch: 3, draftK: 1, minMatch: 2,
                                               reuseCtx: (ctx: ctx,
                                                          residentPerLayer: resident,
                                                          alpha: 1.0))
            if alpha1Draft.isEmpty || alpha1Draft[0] != 100 {
                return (false,
                        "α=1 expected [100] (high-reuse wins tie), got \(alpha1Draft)")
            }
            return (true, "ok")
        }

        // Test 60 (G-A-4): observe 帰属 (attribution).
        // After observe(rowTokens:[t0,t1,t2], layer:li, inds:[M*Ktop]), reuseScore must
        // attribute inds[m*Ktop ..< (m+1)*Ktop] to rowTokens[m] per layer — not to other tokens.
        // Verified via reuseScore: each token with its own observed experts → Ktop*nLayers;
        //   each token with another token's experts → 0 (no cross-attribution).
        // Stub reuseScore=0 → attribution checks fail → RED.
        run("observe_attribution") {
            let Ktop = 4, nLayers = 2
            // Token 10: layer 0 experts {1,2,3,4}, layer 1 experts {21,22,23,24}
            // Token 20: layer 0 experts {5,6,7,8}, layer 1 experts {25,26,27,28}
            // Token 30: layer 0 experts {9,10,11,12}, layer 1 experts {29,30,31,32}
            var ctx = ReuseContext()
            ctx.observe(rowTokens: [10, 20, 30], layer: 0,
                        inds: [1,2,3,4, 5,6,7,8, 9,10,11,12], Ktop: Ktop)
            ctx.observe(rowTokens: [10, 20, 30], layer: 1,
                        inds: [21,22,23,24, 25,26,27,28, 29,30,31,32], Ktop: Ktop)
            let expectedOwn = Double(Ktop * nLayers)   // 8: full overlap on both layers
            // Each token with its own observed experts → expectedOwn
            let res10own: [Set<Int>] = [Set([1,2,3,4]), Set([21,22,23,24])]
            let score10own = ctx.reuseScore(token: 10, residentPerLayer: res10own)
            if score10own != expectedOwn {
                return (false,
                        "token 10 own-expert score=\(score10own) expected \(expectedOwn)")
            }
            let res20own: [Set<Int>] = [Set([5,6,7,8]), Set([25,26,27,28])]
            let score20own = ctx.reuseScore(token: 20, residentPerLayer: res20own)
            if score20own != expectedOwn {
                return (false,
                        "token 20 own-expert score=\(score20own) expected \(expectedOwn)")
            }
            let res30own: [Set<Int>] = [Set([9,10,11,12]), Set([29,30,31,32])]
            let score30own = ctx.reuseScore(token: 30, residentPerLayer: res30own)
            if score30own != expectedOwn {
                return (false,
                        "token 30 own-expert score=\(score30own) expected \(expectedOwn)")
            }
            // Cross-attribution: token 10 vs token 20's experts → 0 (no overlap)
            let res10cross: [Set<Int>] = [Set([5,6,7,8]), Set([25,26,27,28])]
            let score10cross = ctx.reuseScore(token: 10, residentPerLayer: res10cross)
            if score10cross != 0.0 {
                return (false,
                        "token 10 cross-expert score=\(score10cross) expected 0.0 " +
                        "(wrong attribution: token 10 should not match token 20 experts)")
            }
            return (true, "ok")
        }

        // Test 61 (G-A.1): diag_copy_route copy correctness (notes/11 案B Stage 0)
        // The bolt routing-telemetry kernel copies per-layer routing state into a layer-
        // indexed side-buffer: inds[Ktop] int32 → offset li*Ktop, gl[E] half → offset li*E.
        // Reference = identity copy (the kernel's own definition), NOT a reimplemented oracle:
        // the returned li-slice must bit-match the source inds/gl. Exercising li∈{0,2,1}
        // (numLayers=3) proves the layer-offset math, not just the trivial li=0 case.
        run("diag_copy_route_bitexact") {
            let E = 8, Ktop = 4, numLayers = 3
            let inds: [Int32] = [4, 1, 6, 2]
            // f16-exact gate logits so the half round-trip is lossless.
            let gl: [Float16] = [1.0, 5.0, 3.0, 2.0, 9.0, 0.5, 7.0, 4.0]
            for li in [0, 2, 1] {
                guard let (gotI, gotG) = SeedlessFusedVerify.SeedlessFusedForward.diagCopyRouteSelfTest(
                    inds: inds, gl: gl, Ktop: Ktop, E: E, numLayers: numLayers, li: li)
                else { return (false, "not implemented (li=\(li))") }
                if gotI != inds { return (false, "li=\(li) inds \(gotI) != \(inds)") }
                if gotG.count != E { return (false, "li=\(li) gl count \(gotG.count) != \(E)") }
                for e in 0..<E where gotG[e] != gl[e] {
                    return (false, "li=\(li) gl[\(e)]=\(gotG[e]) != \(gl[e])")
                }
            }
            return (true, "ok")
        }

        // Test 62 (G-A.2): computeRouteDiag synthetic hand-check (notes/11 案B Stage 0)
        // Pure CPU aggregation. resident = pinned top-C; cold iff buddyExpert[e] != e;
        // cold e's margin = gl[e] − max{gl[r] : r ∈ resident, r ∉ routed}; empty ⇒ +inf.
        run("compute_route_diag_synthetic") {
            let Ktop = 4
            // buddyExpert: hot(self) for 0-3, cold(remapped) for 4-7.
            let buddyExpert: [Int32] = [0, 1, 2, 3, 0, 1, 2, 3]
            let gl: [Float16] = [1.0, 5.0, 3.0, 2.0, 9.0, 0.5, 7.0, 4.0]

            // Case A: routed=[4,1,6,2]; resident={0,1,2,3}; resident∉routed={0,3}.
            //   cold selected = {4,6}. max{gl[0],gl[3]} = max(1,2) = 2.
            //   margin[4]=9-2=7, margin[6]=7-2=5.
            let (coldA, marA) = SeedlessFusedVerify.SeedlessFusedForward.computeRouteDiag(
                inds: [4, 1, 6, 2], gl: gl, resident: Set([0, 1, 2, 3]),
                buddyExpert: buddyExpert, Ktop: Ktop)
            if Set(coldA) != Set([4, 6]) { return (false, "A coldSelected=\(coldA) want {4,6}") }
            if coldA.count != marA.count { return (false, "A parallel len \(coldA.count)!=\(marA.count)") }
            // map expert→margin for order-independent check
            var mA: [Int: Float] = [:]
            for (i, e) in coldA.enumerated() { mA[e] = marA[i] }
            if mA[4] != 7.0 { return (false, "A margin[4]=\(String(describing: mA[4])) want 7.0") }
            if mA[6] != 5.0 { return (false, "A margin[6]=\(String(describing: mA[6])) want 5.0") }

            // Case B (+inf): resident={0,1}, routed=[0,1,5], resident∉routed={} → margin[5]=+inf.
            let (coldB, marB) = SeedlessFusedVerify.SeedlessFusedForward.computeRouteDiag(
                inds: [0, 1, 5], gl: gl, resident: Set([0, 1]),
                buddyExpert: buddyExpert, Ktop: 3)
            if coldB != [5] { return (false, "B coldSelected=\(coldB) want [5]") }
            if marB.count != 1 || !marB[0].isInfinite || marB[0] < 0 {
                return (false, "B margins=\(marB) want [+inf]")
            }

            // Case C (no cold): all routed experts hot → empty.
            let (coldC, marC) = SeedlessFusedVerify.SeedlessFusedForward.computeRouteDiag(
                inds: [0, 1, 2, 3], gl: gl, resident: Set([0, 1, 2, 3]),
                buddyExpert: buddyExpert, Ktop: Ktop)
            if !coldC.isEmpty || !marC.isEmpty { return (false, "C cold=\(coldC) mar=\(marC) want empty") }
            return (true, "ok")
        }

        // Test 65 (G-A.1 — notes/13): diag_copy_route の (slot, li, M) 一般化 offset の bit-exact 検証。
        // 新 self-test hook(既存 diagCopyRouteSelfTest は不変・別名 hook を追加)。side-buffer 全体を
        // 返させ、① inds[M*Ktop] が期待 element offset ((slot*nLayers+li)*diagObsMaxM)*Ktop に identity
        // copy されている ② それ以外は 0(offset/copy長ともに exact — over-copy も wrong-offset も検出)
        // を確認。defaults(slot=0, diagObsMaxM=1, M=1) は element offset li*Ktop = 旧 byte offset
        // li*Ktop*4 に退化 → test 61 互換。Reference = kernel の identity copy 定義(CPU 再実装 oracle 無)。
        run("diag_copy_slot_m_layout") {
            let Ktop = 4, E = 8, nLayers = 3, chainKMax = 8
            let diagObsMaxM = 4
            // up to M=4 rows worth of source inds (M*Ktop = 16), row-major.
            let src16: [Int32] = [4, 1, 6, 2,  5, 0, 7, 3,  2, 6, 1, 4,  0, 3, 5, 7]
            let bufLen = chainKMax * nLayers * diagObsMaxM * Ktop
            for slot in [0, 3] {
                for li in [0, 2] {
                    for M in [1, 4] {
                        let src = Array(src16.prefix(M * Ktop))
                        guard let buf = SeedlessFusedVerify.SeedlessFusedForward.diagCopySlotMLayoutSelfTest(
                            inds: src, slot: slot, li: li, M: M,
                            diagObsMaxM: diagObsMaxM, nLayers: nLayers,
                            chainKMax: chainKMax, Ktop: Ktop, E: E)
                        else { return (false, "not implemented (slot=\(slot) li=\(li) M=\(M))") }
                        if buf.count != bufLen {
                            return (false, "slot=\(slot) li=\(li) M=\(M): buf.count \(buf.count) != \(bufLen)")
                        }
                        let want = (slot * nLayers + li) * diagObsMaxM * Ktop   // element offset
                        // identity copy present at the expected offset
                        for j in 0 ..< (M * Ktop) where buf[want + j] != src[j] {
                            return (false, "slot=\(slot) li=\(li) M=\(M): buf[\(want + j)]=\(buf[want + j]) != \(src[j])")
                        }
                        // nothing written outside [want, want+M*Ktop) — pins offset & length exactly
                        for j in 0 ..< bufLen where (j < want || j >= want + M * Ktop) && buf[j] != 0 {
                            return (false, "slot=\(slot) li=\(li) M=\(M): leak buf[\(j)]=\(buf[j]) (want region [\(want),\(want + M * Ktop)))")
                        }
                    }
                }
            }
            // defaults(slot=0, diagObsMaxM=1, M=1) → element offset li*Ktop == old byte offset li*Ktop*4
            for li in [0, 2, 1] {
                let src = Array(src16.prefix(Ktop))
                guard let buf = SeedlessFusedVerify.SeedlessFusedForward.diagCopySlotMLayoutSelfTest(
                    inds: src, slot: 0, li: li, M: 1,
                    diagObsMaxM: 1, nLayers: nLayers, chainKMax: chainKMax, Ktop: Ktop, E: E)
                else { return (false, "not implemented defaults li=\(li)") }
                let want = li * Ktop   // == old layout li*Ktop*4 bytes
                for j in 0 ..< Ktop where buf[want + j] != src[j] {
                    return (false, "defaults li=\(li): buf[\(want + j)]=\(buf[want + j]) != \(src[j]) (old layout li*Ktop*4)")
                }
            }
            return (true, "ok")
        }

        // Test 66 (G-A.2 — notes/13): free-run recalib 観測累積の純関数を手計算で照合。
        // 検証点: ①行独立性(M>1: 各行の distinct を独立集計、cross-row pair は作らない)
        //         ②行内重複の dedup(同一 expert が 1 行に2回 → counts は1)
        //         ③行跨ぎは加算(同一 expert が別行に → counts は2)
        //         ④additive(2回目の呼び出しが既存 counts/coact に積み上がる)。
        // Reference = 手計算(合成小ケース、CPU 再実装 oracle でなく数式定義)。
        run("recalib_obs_accumulate") {
            let Ktop = 4, nE = 8
            var counts = [Int](repeating: 0, count: nE)
            var coact = [[Int]](repeating: [Int](repeating: 0, count: nE), count: nE)

            // Call 1: M=2 rows.
            //   row0 [1,2,3,2] → distinct {1,2,3} (2 dedup within row)
            //   row1 [3,4,5,3] → distinct {3,4,5} (3 dedup within row; 3 also in row0 → counts twice)
            guard SeedlessFusedVerify.SeedlessFusedForward.recalibAccumulate(
                inds: [1, 2, 3, 2,  3, 4, 5, 3], M: 2, Ktop: Ktop, nE: nE,
                counts: &counts, coact: &coact)
            else { return (false, "not implemented (call1)") }
            let want1: [Int] = [0, 1, 1, 2, 1, 1, 0, 0]
            if counts != want1 { return (false, "call1 counts=\(counts) want \(want1)") }
            // coact pairs: row0 (1,2)(1,3)(2,3), row1 (3,4)(3,5)(4,5); symmetric.
            func chk(_ a: Int, _ b: Int, _ v: Int, _ tag: String) -> (Bool, String)? {
                if coact[a][b] != v || coact[b][a] != v {
                    return (false, "\(tag) coact[\(a)][\(b)]=\(coact[a][b]) coact[\(b)][\(a)]=\(coact[b][a]) want \(v)")
                }
                return nil
            }
            for (a, b) in [(1, 2), (1, 3), (2, 3), (3, 4), (3, 5), (4, 5)] {
                if let f = chk(a, b, 1, "call1") { return f }
            }
            // row independence: no cross-row pair (1 from row0, 4 from row1)
            if let f = chk(1, 4, 0, "call1 cross-row") { return f }
            if let f = chk(2, 5, 0, "call1 cross-row") { return f }

            // Call 2: M=1 row [1,1,6,7] → distinct {1,6,7} (1 dedup within row) — must ADD.
            guard SeedlessFusedVerify.SeedlessFusedForward.recalibAccumulate(
                inds: [1, 1, 6, 7], M: 1, Ktop: Ktop, nE: nE,
                counts: &counts, coact: &coact)
            else { return (false, "not implemented (call2)") }
            let want2: [Int] = [0, 2, 1, 2, 1, 1, 1, 1]   // counts[1] 1→2, counts[6],[7] 0→1
            if counts != want2 { return (false, "call2 counts=\(counts) want \(want2)") }
            for (a, b) in [(1, 6), (1, 7), (6, 7)] {
                if let f = chk(a, b, 1, "call2") { return f }
            }
            // prior pairs preserved & not double-counted by the additive second call
            if let f = chk(1, 2, 1, "call2 preserve") { return f }
            if let f = chk(4, 5, 1, "call2 preserve") { return f }
            return (true, "ok")
        }

        // Test 67 (G-A.1 — notes/14): refresh plan is deterministic and correct.
        // Checks: diff = newTop ∖ resident, victim ∉ (newTop slots ∪ pinnedSlots),
        // chunk splitting covers all jobs with each chunk ≤ B, same input → same plan.
        // Adversarial (Scenario B): newTop-resident expert occupies a slot that would
        // be the LRU victim — plan must exclude it, unlike vanilla ensure().
        run("refresh_plan_deterministic") {
            // ── Scenario A ────────────────────────────────────────────────
            // nE=8, C=4, B=2
            // Resident: {0→slot0(tick=10), 1→slot1(tick=5), 2→slot2(tick=8), 3→slot3(tick=3)}
            // Pinned: slot 0 (holds expert 0)
            // Counts: expert 4=10, 5=8, 6=7, rest=0
            // Expected newTop: {0,4,5,6}
            //   (4th-best count=0: tie among {1,2,3,7} → lower id = expert 0 wins tie-break)
            // diff = {4,5,6}  (experts 4,5,6 not resident)
            // Victims LRU (exclude pinned slot0 + exclude newTop-resident slot0 for exp0):
            //   slot3(tick=3)→expert4, slot1(tick=5)→expert5, slot2(tick=8)→expert6
            let nE = 8, C = 4, B = 2
            let counts = [0, 0, 0, 0, 10, 8, 7, 0]
            var coact = [[Int]](repeating: [Int](repeating: 0, count: nE), count: nE)
            coact[4][5] = 3; coact[5][4] = 3; coact[5][6] = 2; coact[6][5] = 2
            let slotOf: [Int: Int] = [0: 0, 1: 1, 2: 2, 3: 3]
            let expertAt: [Int] = [0, 1, 2, 3]
            let tick: [Int] = [10, 5, 8, 3]
            let pinned: Set<Int> = [0]

            guard let plan = BoltAsyncRefresh.makePlan(
                counts: counts, coact: coact,
                slotOf: slotOf, expertAt: expertAt, tick: tick,
                pinnedSlots: pinned, C: C, nE: nE, B: B)
            else { return (false, "not implemented") }

            // newTop must be exactly C experts
            if plan.newTop.count != C {
                return (false, "newTop.count \(plan.newTop.count) != C=\(C)")
            }
            let newTopSet = Set(plan.newTop)
            if newTopSet != Set([0, 4, 5, 6]) {
                return (false, "newTop \(plan.newTop.sorted()) != [0,4,5,6]")
            }
            // diff = newTop ∖ resident
            let diffSet = Set(plan.diff)
            let resident = Set(slotOf.keys)
            if diffSet != Set([4, 5, 6]) {
                return (false, "diff \(plan.diff.sorted()) != [4,5,6]")
            }
            if !diffSet.isDisjoint(with: resident) {
                return (false, "diff ∩ resident ≠ ∅")
            }
            // victim slots ∉ pinnedSlots and ∉ slots of newTop-resident experts
            let newTopResSlots = Set(plan.newTop.compactMap { slotOf[$0] })
            for job in plan.jobs {
                if pinned.contains(job.victimSlot) {
                    return (false, "victim slot \(job.victimSlot) is pinned")
                }
                if newTopResSlots.contains(job.victimSlot) {
                    return (false, "victim slot \(job.victimSlot) holds a newTop resident")
                }
            }
            // jobs must cover exactly the diff set
            if Set(plan.jobs.map { $0.expert }) != diffSet {
                return (false, "jobs experts \(plan.jobs.map{$0.expert}.sorted()) != diff \(plan.diff.sorted())")
            }
            // chunk splitting: chunks.flatMap == jobs, each chunk ≤ B, none empty
            let flatChunkJobs = plan.chunks.flatMap { $0 }
            if flatChunkJobs != plan.jobs {
                return (false, "chunks do not reproduce jobs exactly: \(flatChunkJobs.count) vs \(plan.jobs.count)")
            }
            for (ci, chunk) in plan.chunks.enumerated() {
                if chunk.isEmpty  { return (false, "chunk \(ci) is empty") }
                if chunk.count > B { return (false, "chunk \(ci) size \(chunk.count) > B=\(B)") }
            }

            // Determinism: same input → same plan
            guard let plan2 = BoltAsyncRefresh.makePlan(
                counts: counts, coact: coact,
                slotOf: slotOf, expertAt: expertAt, tick: tick,
                pinnedSlots: pinned, C: C, nE: nE, B: B)
            else { return (false, "second call returned nil") }
            if plan.newTop != plan2.newTop { return (false, "newTop non-deterministic") }
            if plan.diff   != plan2.diff   { return (false, "diff non-deterministic") }
            if plan.jobs   != plan2.jobs   { return (false, "jobs non-deterministic") }

            // ── Scenario B (adversarial): newTop-resident in LRU-oldest slot ─────
            // nE=6, C=3, B=2
            // Resident: {0→slot0(tick=1), 1→slot1(tick=2), 2→slot2(tick=10)}, pinned={}
            // Counts: [0,0,5,8,4,0] → newTop={3,2,4} (top-3)
            // Expert 2 is resident in slot2(tick=10, highest tick = NOT LRU).
            // diff = {3,4}
            // Vanilla ensure would pick slot0(tick=1) and slot1(tick=2) as victims.
            // Plan must ALSO exclude slot2 (holds newTop expert 2) from victims —
            // here it is naturally excluded because tick=10 is not LRU, confirming
            // the algorithm doesn't accidentally evict a newTop resident even when
            // it's LRU (the adversarial check below uses a fresh scenario where
            // the newTop resident has low tick to make the exclusion load-bearing).
            let nE2 = 6, C2 = 3, B2 = 2
            let counts2 = [0, 0, 5, 8, 4, 0]
            let coact2 = [[Int]](repeating: [Int](repeating: 0, count: nE2), count: nE2)
            let slotOf2: [Int: Int] = [0: 0, 1: 1, 2: 2]
            let expertAt2: [Int]   = [0, 1, 2]
            let tick2: [Int]       = [1, 2, 10]   // expert 2 (in newTop) is highest tick

            guard let planB = BoltAsyncRefresh.makePlan(
                counts: counts2, coact: coact2,
                slotOf: slotOf2, expertAt: expertAt2, tick: tick2,
                pinnedSlots: Set(), C: C2, nE: nE2, B: B2)
            else { return (false, "scenario B: not implemented") }
            if Set(planB.newTop) != Set([2, 3, 4]) {
                return (false, "B newTop \(planB.newTop.sorted()) != [2,3,4]")
            }
            if Set(planB.diff) != Set([3, 4]) {
                return (false, "B diff \(planB.diff.sorted()) != [3,4]")
            }
            // expert 2 in newTop → slot 2 must NOT be a victim
            let newTopResSlots2 = Set(planB.newTop.compactMap { slotOf2[$0] })  // = {2}
            for job in planB.jobs {
                if newTopResSlots2.contains(job.victimSlot) {
                    return (false, "B: victim slot \(job.victimSlot) holds newTop-resident exp2")
                }
            }
            // Victims must be slot 0 and slot 1 (LRU of {slot0,slot1}, slot2 excluded)
            if Set(planB.jobs.map { $0.victimSlot }) != Set([0, 1]) {
                return (false, "B victim slots \(planB.jobs.map{$0.victimSlot}.sorted()) != [0,1]")
            }
            // Adversarial variant: same scenario but with expert 2 having LOW tick (tick2b).
            // Now slot2 would be LRU if not excluded — the exclusion becomes load-bearing.
            let tick2b: [Int] = [10, 8, 1]   // expert 2 in slot2 has tick=1 (oldest)
            guard let planB2 = BoltAsyncRefresh.makePlan(
                counts: counts2, coact: coact2,
                slotOf: slotOf2, expertAt: expertAt2, tick: tick2b,
                pinnedSlots: Set(), C: C2, nE: nE2, B: B2)
            else { return (false, "scenario B adversarial: not implemented") }
            // newTop = {2,3,4} still (counts unchanged)
            if Set(planB2.newTop) != Set([2, 3, 4]) {
                return (false, "B-adv newTop \(planB2.newTop.sorted()) != [2,3,4]")
            }
            // slot2 (holding newTop expert 2) MUST NOT be a victim even though tick=1 is LRU
            for job in planB2.jobs {
                if job.victimSlot == 2 {
                    return (false, "B-adv: slot2 (newTop expert 2, tick=1) was chosen as victim — exclusion rule violated")
                }
            }
            // Victims must come from {slot0(tick=10), slot1(tick=8)} only
            if Set(planB2.jobs.map { $0.victimSlot }) != Set([0, 1]) {
                return (false, "B-adv victim slots \(planB2.jobs.map{$0.victimSlot}.sorted()) != [0,1]")
            }
            return (true, "ok")
        }


        // Test 68 (G-A.2 — notes/14): chunk swap is atomic and slot-consistent.
        // Per chunk: (a) slotOf is injective after each chunk (no double-mapping),
        //            (b) expertAt[slotOf[e]] == e for all resident e.
        // After all chunks + rebuildBuddyCPU: slotOf/expertAt/buddyTableCPU match the
        // sync ensure path (same LRU algorithm, newTop-resident exclusion consistent).
        // Adversarial: single-chunk (B ≥ diff) performs all swaps atomically.
        run("chunk_swap_atomic") {
            // ── Shared setup ─────────────────────────────────────────────
            // nE=8, C=4; initial resident {0→s0(t=10),1→s1(t=5),2→s2(t=8),3→s3(t=3)}
            // counts → newTop={0,4,5,6}, diff={4,5,6}
            // experts in diff have simple coact pairs: (4,5) and (5,6)
            let nE = 8, C = 4
            var coact = [[Int]](repeating: [Int](repeating: 0, count: nE), count: nE)
            coact[4][5] = 3; coact[5][4] = 3; coact[5][6] = 2; coact[6][5] = 2
            coact[4][6] = 1; coact[6][4] = 1
            let initSlotOf: [Int: Int] = [0: 0, 1: 1, 2: 2, 3: 3]
            let initExpertAt: [Int]   = [0, 1, 2, 3]
            let initTick: [Int]       = [10, 5, 8, 3]
            let pinned: Set<Int> = []

            // ── Scenario A: B=2 (multi-chunk) ────────────────────────────
            let B = 2
            let counts = [0, 0, 0, 0, 10, 8, 7, 0]
            guard let plan = BoltAsyncRefresh.makePlan(
                counts: counts, coact: coact,
                slotOf: initSlotOf, expertAt: initExpertAt, tick: initTick,
                pinnedSlots: pinned, C: C, nE: nE, B: B)
            else { return (false, "makePlan not implemented") }

            // Path A: apply chunks one at a time; check slot-consistency after each.
            var stateA = BoltAsyncRefresh.CacheState(
                slotOf: initSlotOf, expertAt: initExpertAt,
                tick: initTick, clock: 100,
                buddyTableCPU: [Int32](repeating: 0, count: nE),
                buddyExpertCPU: [Int32](repeating: -1, count: nE))
            for (ci, chunk) in plan.chunks.enumerated() {
                guard stateA.applyChunkCPU(jobs: chunk)
                else { return (false, "applyChunkCPU failed at chunk=\(ci)") }
                // Slot-consistency invariant after each partial application:
                // (1) slotOf is injective
                let slots = Array(stateA.slotOf.values)
                if Set(slots).count != slots.count {
                    return (false, "double-mapping after chunk=\(ci): \(stateA.slotOf)")
                }
                // (2) forward-reverse maps agree
                for (e, s) in stateA.slotOf {
                    if stateA.expertAt[s] != e {
                        return (false, "expertAt inconsistent after chunk=\(ci): expertAt[\(s)]=\(stateA.expertAt[s]) != e=\(e)")
                    }
                }
            }
            guard stateA.rebuildBuddyCPU(coact: coact, nE: nE)
            else { return (false, "rebuildBuddyCPU not implemented") }

            // Path B (sync reference): apply all diff experts at once via LRU ensure.
            // expert 0 has tick=10 (highest), so vanilla LRU won't evict it even without
            // the newTop-exclusion rule — both paths must agree on the final slot assignment.
            var stateB = BoltAsyncRefresh.CacheState(
                slotOf: initSlotOf, expertAt: initExpertAt,
                tick: initTick, clock: 100,
                buddyTableCPU: [Int32](repeating: 0, count: nE),
                buddyExpertCPU: [Int32](repeating: -1, count: nE))
            guard stateB.syncEnsure(experts: plan.diff, pinnedSlots: pinned)
            else { return (false, "syncEnsure not implemented") }
            guard stateB.rebuildBuddyCPU(coact: coact, nE: nE)
            else { return (false, "rebuildBuddyCPU (B) not implemented") }

            // Compare final states
            if stateA.slotOf != stateB.slotOf {
                let aS = stateA.slotOf.sorted { $0.key < $1.key }
                let bS = stateB.slotOf.sorted { $0.key < $1.key }
                return (false, "slotOf mismatch A=\(aS) B=\(bS)")
            }
            if stateA.expertAt != stateB.expertAt {
                return (false, "expertAt mismatch A=\(stateA.expertAt) B=\(stateB.expertAt)")
            }
            if stateA.buddyTableCPU != stateB.buddyTableCPU {
                return (false, "buddyTableCPU mismatch A=\(stateA.buddyTableCPU) B=\(stateB.buddyTableCPU)")
            }
            // final resident set = newTop
            if Set(stateA.slotOf.keys) != Set(plan.newTop) {
                return (false, "final resident \(stateA.slotOf.keys.sorted()) != newTop \(plan.newTop.sorted())")
            }

            // ── Scenario B (adversarial): B=8 ≥ diff (single chunk) ──────
            // All swaps happen atomically in one chunk — no partial-state phase.
            // Verify that a single-chunk plan is structurally valid and produces
            // the same final state as the multi-chunk plan (same diff, coact, tick).
            let Bsingle = 8
            guard let planSingle = BoltAsyncRefresh.makePlan(
                counts: counts, coact: coact,
                slotOf: initSlotOf, expertAt: initExpertAt, tick: initTick,
                pinnedSlots: pinned, C: C, nE: nE, B: Bsingle)
            else { return (false, "scenario B: makePlan not implemented") }
            if planSingle.chunks.count != 1 {
                return (false, "B: expected 1 chunk for B=\(Bsingle) diff.count=\(planSingle.diff.count), got \(planSingle.chunks.count)")
            }
            var stateC = BoltAsyncRefresh.CacheState(
                slotOf: initSlotOf, expertAt: initExpertAt,
                tick: initTick, clock: 100,
                buddyTableCPU: [Int32](repeating: 0, count: nE),
                buddyExpertCPU: [Int32](repeating: -1, count: nE))
            guard stateC.applyChunkCPU(jobs: planSingle.chunks[0])
            else { return (false, "B: applyChunkCPU failed") }
            guard stateC.rebuildBuddyCPU(coact: coact, nE: nE)
            else { return (false, "B: rebuildBuddyCPU not implemented") }
            // Final resident set must still equal newTop
            if Set(stateC.slotOf.keys) != Set(planSingle.newTop) {
                return (false, "B final resident \(stateC.slotOf.keys.sorted()) != newTop \(planSingle.newTop.sorted())")
            }
            // buddyTableCPU must match the multi-chunk path (same coact, same final slotOf)
            if stateC.slotOf != stateA.slotOf {
                let cS = stateC.slotOf.sorted { $0.key < $1.key }
                let aS = stateA.slotOf.sorted { $0.key < $1.key }
                return (false, "B slotOf mismatch C=\(cS) A=\(aS)")
            }
            if stateC.buddyTableCPU != stateA.buddyTableCPU {
                return (false, "B buddyTableCPU mismatch C=\(stateC.buddyTableCPU) A=\(stateA.buddyTableCPU)")
            }
            return (true, "ok")
        }


        // Test 69 (notes/14 TODO-2): QWISP_BOLT_WORKLOAD per-workload preset (R/B).
        // Pure function Tell.boltWorkloadPreset maps a workload name to the
        // proven-optimal (recalib R, refresh B). Known names get tuned values; any
        // other string ("", "longctx", unknown) falls back to the current default
        // (R=128, B=32) so QWISP_BOLT_WORKLOAD unset is byte-identical to old bolt.
        // Process-env-independent: calls the pure function directly.
        run("bolt_workload_preset") {
            let cases: [(String, Int, Int)] = [
                ("code",    128, 64),
                ("agentic", 256, 32),
                ("shortnl", 128, 16),
                ("longctx", 128, 32),   // unknown-but-real → default
                ("zzz",     128, 32),   // arbitrary unknown → default
            ]
            for (w, wantR, wantB) in cases {
                let got = Tell.boltWorkloadPreset(w)
                if got.r != wantR || got.b != wantB {
                    return (false, "workload=\(w): got (R=\(got.r), B=\(got.b)) want (R=\(wantR), B=\(wantB))")
                }
            }
            // Empty string must also yield the default (unset-env path).
            let empty = Tell.boltWorkloadPreset("")
            if empty.r != 128 || empty.b != 32 {
                return (false, "empty workload: got (R=\(empty.r), B=\(empty.b)) want (R=128, B=32)")
            }
            return (true, "ok")
        }

        // Test 70 (notes/15 G-A): mtp_feed_plan — pure-function row-map contract.
        // Calls Tell.mtpFeedPlan(pk:p:path:) directly (no process-env dependency).
        // Verifies the row-map convention from notes/15 §head-sync:
        //   rows = [pending pk][u][drafts] in H2 from verify.
        //   fullAccept/reject: feedRows = 0..<(pk+p), lastHRow = pk+p.
        //   replay:  feedRows = 0..<0, lastHRow = -1   (replay feeds sequentially; pk/p ignored)
        //   single:  feedRows = 0..<pk, lastHRow = pk  (feed pending hiddens; lastH = u hidden)
        // All cases are RED until the stub returns a non-nil value.
        run("mtp_feed_plan") {
            struct Case {
                let pk: Int, p: Int, path: FeedPath
                let wantFeed: Range<Int>, wantLastH: Int
            }
            let cases: [Case] = [
                // fullAccept × 4
                Case(pk: 0, p: 0, path: .fullAccept, wantFeed: 0..<0,  wantLastH: 0),
                Case(pk: 0, p: 3, path: .fullAccept, wantFeed: 0..<3,  wantLastH: 3),
                Case(pk: 2, p: 0, path: .fullAccept, wantFeed: 0..<2,  wantLastH: 2),
                Case(pk: 2, p: 3, path: .fullAccept, wantFeed: 0..<5,  wantLastH: 5),
                // reject × 4 — identical contract to fullAccept (both flush committed prefix)
                Case(pk: 0, p: 0, path: .reject,     wantFeed: 0..<0,  wantLastH: 0),
                Case(pk: 0, p: 3, path: .reject,     wantFeed: 0..<3,  wantLastH: 3),
                Case(pk: 2, p: 0, path: .reject,     wantFeed: 0..<2,  wantLastH: 2),
                Case(pk: 2, p: 3, path: .reject,     wantFeed: 0..<5,  wantLastH: 5),
                // replay — adversarial: pk/p ignored, always empty+(-1)
                Case(pk: 0, p: 0, path: .replay,     wantFeed: 0..<0,  wantLastH: -1),
                Case(pk: 2, p: 3, path: .replay,     wantFeed: 0..<0,  wantLastH: -1),
                // single — pending hiddens only; lastH = u row
                Case(pk: 0, p: 0, path: .single,     wantFeed: 0..<0,  wantLastH: 0),
                Case(pk: 2, p: 0, path: .single,     wantFeed: 0..<2,  wantLastH: 2),
            ]
            for c in cases {
                guard let got = Tell.mtpFeedPlan(pk: c.pk, p: c.p, path: c.path)
                else { return (false, "not implemented (pk=\(c.pk) p=\(c.p) path=\(c.path))") }
                if got.feedRows != c.wantFeed {
                    return (false, "pk=\(c.pk) p=\(c.p) path=\(c.path): feedRows=\(got.feedRows) want \(c.wantFeed)")
                }
                if got.lastHRow != c.wantLastH {
                    return (false, "pk=\(c.pk) p=\(c.p) path=\(c.path): lastHRow=\(got.lastHRow) want \(c.wantLastH)")
                }
            }
            return (true, "ok")
        }

        // Test 71 (notes/11 レバー② measure-first): missWeightStats — pure function.
        // bolt の cold-selection(miss)の gate-weight 分布 + top-8 内 rank 集計。
        // Calls Tell.missWeightStats directly (no process-env / GPU dependency).
        // Hand-computed cases: 分位点 index=floor(q*(n-1)), top1Share=fraction 0..1,
        // meanMissMass = Σ(miss weight) / topInds.count(全観測数), missCount=総 miss 数。
        // RED until the stub returns real values (stub returns -1 sentinel).
        run("missWeightStats") {
            func approx(_ a: Float, _ b: Float) -> Bool { abs(a - b) < 1e-5 }

            struct Case {
                let name: String
                let inds: [[Int]]
                let w: [[Float]]
                let res: [Set<Int>]
                let p10: Float, p50: Float, p90: Float
                let top1: Float, mass: Float, misses: Int
            }
            let cases: [Case] = [
                // (A) no miss — every routed expert resident → all zeros.
                Case(name: "no-miss",
                     inds: [[0, 1, 2, 3]],
                     w:    [[0.4, 0.3, 0.2, 0.1]],
                     res:  [Set([0, 1, 2, 3])],
                     p10: 0, p50: 0, p90: 0, top1: 0, mass: 0, misses: 0),
                // (B) all miss — resident empty. weights sorted asc [0.1,0.2,0.3,0.4], n=4.
                //   p10 idx=floor(0.1*3)=0→0.1  p50 idx=floor(0.5*3)=1→0.2  p90 idx=floor(0.9*3)=2→0.3
                //   rank-1 (max=0.4 @ idx0) is a miss → top1Share=1/4=0.25
                //   mass = (0.4+0.3+0.2+0.1)/1obs = 1.0 ; misses=4
                Case(name: "all-miss",
                     inds: [[0, 1, 2, 3]],
                     w:    [[0.4, 0.3, 0.2, 0.1]],
                     res:  [Set<Int>()],
                     p10: 0.1, p50: 0.2, p90: 0.3, top1: 0.25, mass: 1.0, misses: 4),
                // (C) tail-only miss — rank1,2 resident; low-gate tail misses. n=4 miss weights
                //   sorted asc [0.05,0.07,0.08,0.10].  p10 idx0→0.05  p50 idx1→0.07  p90 idx2→0.08
                //   rank-1 (max=0.5 @ idx0) resident → NOT a miss → top1Share=0
                //   mass = (0.1+0.08+0.07+0.05)/1obs = 0.30 ; misses=4
                Case(name: "tail-only",
                     inds: [[0, 1, 2, 3, 4, 5]],
                     w:    [[0.5, 0.2, 0.1, 0.08, 0.07, 0.05]],
                     res:  [Set([0, 1])],
                     p10: 0.05, p50: 0.07, p90: 0.08, top1: 0.0, mass: 0.30, misses: 4),
                // (D) top1 miss mixed across 2 observations — divides mass by obs count(2).
                //   obs0: miss idx0 w=0.6 = rank-1 (top1).  obs1: miss idx3 w=0.05 = tail.
                //   miss weights sorted asc [0.05,0.6], n=2. p10/p50/p90 idx=floor(q*1):
                //     0.1→0 0.5→0 0.9→0  → all 0.05
                //   top1Share = 1/2 = 0.5 ; mass = (0.6+0.05)/2obs = 0.325 ; misses=2
                Case(name: "top1-mixed",
                     inds: [[10, 11, 12, 13], [20, 21, 22, 23]],
                     w:    [[0.6, 0.2, 0.15, 0.05], [0.4, 0.35, 0.2, 0.05]],
                     res:  [Set([11, 12, 13]), Set([20, 21, 22])],
                     p10: 0.05, p50: 0.05, p90: 0.05, top1: 0.5, mass: 0.325, misses: 2),
            ]
            for c in cases {
                let g = Tell.missWeightStats(topInds: c.inds, topWeights: c.w, resident: c.res)
                if g.missCount != c.misses {
                    return (false, "\(c.name): missCount=\(g.missCount) want \(c.misses)")
                }
                if !approx(g.p10, c.p10) || !approx(g.p50, c.p50) || !approx(g.p90, c.p90) {
                    return (false, "\(c.name): p10/50/90=\(g.p10)/\(g.p50)/\(g.p90) want \(c.p10)/\(c.p50)/\(c.p90)")
                }
                if !approx(g.top1Share, c.top1) {
                    return (false, "\(c.name): top1Share=\(g.top1Share) want \(c.top1)")
                }
                if !approx(g.meanMissMass, c.mass) {
                    return (false, "\(c.name): meanMissMass=\(g.meanMissMass) want \(c.mass)")
                }
            }
            return (true, "ok")
        }

        // Test 72 (MTP-D1 raw §Step 2): post-final-norm hidden rows accessor.
        //   hiddenRows(M:) must return the exact [M,H] f16 that forwardRows(finalNormW:)
        //   returned (both read the same `normed` buffer). normedBuffer exposes it for
        //   GPU-direct bind (§Step 3). (ii) stepArgmax parity skipped — head build is heavy
        //   in synthetic and its input path (embed) differs from forwardRows(x); forwardRows
        //   side is the verifiable goal per task spec.
        run("fused_hidden_rows") {
            guard let (device, _) = SeedlessMetalForward.ensure() else { return (false, "no device") }
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
            func mkMoE() -> SeedlessVerifyForward.MoEBlockW {
                let (gW, gS, gB) = q8(E, H); let (sgW, sgS, sgB) = q8(8, H)
                let (a0, a1, a2) = q4e(E, I, H); let (b0, b1, b2) = q4e(E, I, H); let (c0, c1, c2) = q4e(E, H, I)
                let (d0, d1, d2) = q4(I, H); let (e0, e1, e2) = q4(I, H); let (f0, f1, f2) = q4(H, I)
                return SeedlessVerifyForward.MoEBlockW(gateWq: gW, gateSc: gS, gateBi: gB,
                    swGWq: a0, swGSc: a1, swGBi: a2, swUWq: b0, swUSc: b1, swUBi: b2,
                    swDWq: c0, swDSc: c1, swDBi: c2, shGWq: d0, shGSc: d1, shGBi: d2,
                    shUWq: e0, shUSc: e1, shUBi: e2, shDWq: f0, shDSc: f1, shDBi: f2,
                    sharedGateWq: sgW, sharedGateSc: sgS, sharedGateBi: sgB)
            }
            let Hk = 16, Dk = 128, Hv = 32, Dv = 128, cK = 4
            let convDim = Hk * Dk * 2 + Hv * Dv
            let (qkvW, qkvS, qkvB) = q4(convDim, H); let (zW, zS, zB) = q4(Hv * Dv, H)
            let (bW, bS, bB) = q4(Hv, H); let (aW, aS, aB) = q4(Hv, H); let (oW, oS, oB) = q4(H, Hv * Dv)
            let gdnW = SeedlessVerifyForward.GDNLayerW(qkvWq: qkvW, qkvSc: qkvS, qkvBi: qkvB,
                zWq: zW, zSc: zS, zBi: zB, bWq: bW, bSc: bS, bBi: bB, aWq: aW, aSc: aS, aBi: aB,
                outWq: oW, outSc: oS, outBi: oB,
                conv1dW: MLXRandom.normal([convDim, cK]).asType(.float16),
                normWeight: MLXRandom.normal([Dv]).asType(.float16),
                aLog: MLXRandom.normal([Hv]).asType(.float32), dtBias: MLXRandom.normal([Hv]).asType(.float32))
            let nH = 16, nKV = 2, hD = 256
            let (aqW, aqS, aqB) = q4(nH * 2 * hD, H); let (akW, akS, akB) = q4(nKV * hD, H)
            let (avW, avS, avB) = q4(nKV * hD, H); let (aoW, aoS, aoB) = q4(H, nH * hD)
            let attnW = SeedlessVerifyForward.AttnLayerW(qWq: aqW, qSc: aqS, qBi: aqB, kWq: akW, kSc: akS, kBi: akB,
                vWq: avW, vSc: avS, vBi: avB, oWq: aoW, oSc: aoS, oBi: aoB,
                qNorm: MLXRandom.normal([hD]).asType(.float16), kNorm: MLXRandom.normal([hD]).asType(.float16))
            let layers = [
                SeedlessVerifyForward.LayerSpec(isLinear: true,
                    inputLN: MLXRandom.normal([H]).asType(.float16), postLN: MLXRandom.normal([H]).asType(.float16),
                    gdn: gdnW, attn: nil, moe: mkMoE(), moeE: E, moeI: I),
                SeedlessVerifyForward.LayerSpec(isLinear: false,
                    inputLN: MLXRandom.normal([H]).asType(.float16), postLN: MLXRandom.normal([H]).asType(.float16),
                    gdn: nil, attn: attnW, moe: mkMoE(), moeE: E, moeI: I),
            ]
            let cs0 = MLXRandom.normal([cK - 1, convDim]).asType(.float16)
            let rs0 = MLXRandom.normal([1, Hv, Dv, Dk]).asType(.float32)
            let kC0 = MLXRandom.normal([nKV, 16, hD]).asType(.float16)
            let vC0 = MLXRandom.normal([nKV, 16, hD]).asType(.float16)
            MLX.eval([cs0, rs0, kC0, vC0])
            func freshCaches() -> [SeedlessVerifyForward.LayerCaches] {
                [SeedlessVerifyForward.LayerCaches(convState: cs0, recState: rs0),
                 SeedlessVerifyForward.LayerCaches(kCache: kC0, vCache: vC0)]
            }
            let maxM = 8
            guard let fused = SeedlessFusedVerify.SeedlessFusedForward(layers: layers, caches: freshCaches(),
                                                             maxM: maxM, H: H, maxSeqLen: 64)
            else { return (false, "fused init nil") }
            // final RMSNorm weight → MTLBuffer (same helper attachHead uses for fnW).
            let fnA = MLXRandom.normal([H]).asType(.float16); fnA.eval()
            guard let fn = SeedlessMetalForward.mtlBuf(fnA, device) else { return (false, "fn buf nil") }

            let M = 3
            let x = MLXRandom.normal([M, H]).asType(.float16); x.eval()
            guard let ref = fused.forwardRows(x, M: M, finalNormW: fn)
            else { return (false, "forwardRows nil") }
            ref.eval()
            // (i) hiddenRows(M:) ≡ forwardRows(finalNormW:) return (both read `normed`).
            guard let got = fused.hiddenRows(M: M) else { return (false, "hiddenRows nil M=\(M)") }
            got.eval()
            let (ok, d) = bitEqual(got, ref)
            if !ok { return (false, "hiddenRows mismatch M=\(M): \(d)") }
            // maxM+1 → nil (out-of-range guard).
            if fused.hiddenRows(M: maxM + 1) != nil { return (false, "hiddenRows M>maxM not nil") }
            // normedBuffer exposes the real [maxM,H] f16 buffer for §Step 3 GPU bind.
            if fused.normedBuffer.length < maxM * H * 2 {
                return (false, "normedBuffer.length=\(fused.normedBuffer.length) want >= \(maxM * H * 2)")
            }
            return (true, "ok")
        }

        // ── MTP-D1 raw port §Step 3 locked tests (73-75) ─────────────────

        // Test 73 (T-fmm): fmm_rows encoder vs MLX matmul(x, Wᵀ) in f16.
        // Exercises M∈{1,3} and non-aligned K=96,N=40.
        // Ideal: bit-exact. f16 accumulation order differs → rel ≤ 1e-3 acceptable.
        // M-invariance MANDATORY: row m of M=3 call must be bit-identical to the
        // corresponding M=1 call (order-stable rule).
        run("mtp_fmm_rows_vs_mlx") {
            guard let (device, queue) = SeedlessMetalForward.ensure() else {
                return (false, "no device")
            }
            // Non-aligned dimensions per spec (K=96, N=40).
            let K = 96, N = 40
            // W[N,K] f16
            let wF = MLXRandom.normal([N, K]).asType(.float16); wF.eval()
            guard let wBuf = SeedlessMetalForward.mtlBuf(wF, device) else {
                return (false, "w buf nil")
            }
            // Helper: run encodeFmmRows in a fresh CB for M rows of x, return [M,N] f16.
            func runFmm(_ x: MLXArray, M: Int) -> MLXArray? {
                x.eval()
                guard let xBuf = SeedlessMetalForward.mtlBuf(x.asType(.float16), device),
                      let outBuf = device.makeBuffer(length: M * N * 2,
                                                     options: .storageModeShared) else { return nil }
                let cb = queue.makeCommandBuffer()!
                let enc = cb.makeComputeCommandEncoder()!
                SeedlessFusedVerify.encodeFmmRows(enc, w: wBuf, x: xBuf, out: outBuf,
                                             M: M, K: K, N: N)
                enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
                // If the stub did nothing, output will be zeros — distinct from any real result.
                // A nil stub guard is in the caller: we return the buffer as-is so the
                // rel-error check catches the all-zeros vs non-zero reference mismatch.
                let ptr = outBuf.contents().bindMemory(to: Float16.self, capacity: M * N)
                return MLXArray(Array(UnsafeBufferPointer(start: ptr, count: M * N)), [M, N])
            }
            // MLX reference: x[M,K] @ W[N,K].T = x @ Wᵀ
            func mlxRef(_ x: MLXArray) -> MLXArray {
                MLX.matmul(x.asType(.float16), wF.transposed())   // [M,N] f16
            }
            func relErr(_ got: MLXArray, _ ref: MLXArray) -> Float {
                let g = got.asType(.float32).reshaped([-1]).asArray(Float.self)
                let r = ref.asType(.float32).reshaped([-1]).asArray(Float.self)
                var maxR: Float = 0
                for i in 0..<g.count {
                    let denom = max(abs(r[i]), Float(1e-6))
                    maxR = max(maxR, abs(g[i] - r[i]) / denom)
                }
                return maxR
            }
            // Check: stub returns without dispatching → output will be all-zeros,
            // so relErr will be large and the test FAILs (RED).
            for M in [1, 3] {
                let x = MLXRandom.normal([M, K]).asType(.float16); x.eval()
                guard let got = runFmm(x, M: M) else {
                    return (false, "fmm buf nil M=\(M)")
                }
                let ref = mlxRef(x); ref.eval()
                let rel = relErr(got, ref)
                if rel > 1e-3 { return (false, "rel=\(rel) > 1e-3 M=\(M) (STUB not implemented)") }
            }
            // M-invariance: row 0 of M=3 must be bit-identical to M=1 on the same row-0 input.
            let x3 = MLXRandom.normal([3, K]).asType(.float16); x3.eval()
            let x1 = x3[0 ..< 1]; x1.eval()
            guard let got3 = runFmm(x3, M: 3),
                  let got1 = runFmm(x1, M: 1) else {
                return (false, "fmm buf nil m-invariance")
            }
            let row0of3 = got3[0 ..< 1]; row0of3.eval()
            let (okInv, dInv) = bitEqual(row0of3, got1)
            if !okInv { return (false, "M-invariance FAIL: \(dInv)") }
            return (true, "ok")
        }

        // Test 74 (T-head): SeedlessMTPHead end-to-end draft vs production MLX class composition.
        // Real-shape synthetic: H=2048, V=256, E=16, Ktop=8, I=512, nH=16, nKV=2, hD=256, rD=64.
        // WeightsSpec injection (seed-trick impossible: weights arrive as MLXArray, not regenerated
        // inside the impl).  Reference = AttentionLayer(.plain) + MoEBlock(expertGroupSize:64) +
        // ModelHead.embed + Proj.quantized composed exactly as MTPHead.callWithHidden:73-90.
        // Assert (a) len=0 single draft matches MLX argmax.
        // Assert (b) after feedPairs 2 pairs, raw draft == MLX reference with KVCache history.
        // Assert (c) re-draft after (b) returns same token and len is unchanged (READ-ONLY proof).
        // RED: init?(spec:) returns nil → "STUB not implemented" FAIL on all three asserts.
        run("mtp_head_vs_mlx_ref") {
            guard let (device, _) = SeedlessMetalForward.ensure() else {
                return (false, "no device")
            }

            // ── Geometry ─────────────────────────────────────────────────
            let H = 2048, V = 256, E = 16, Ktop = 8, I = 512
            let nH = 16, nKV = 2, hD = 256, rD = 64
            let gs = 64   // expertGroupSize for synthetic (spec §T-74)
            let eps: Float = 1e-6
            let ropeBase: Float = 1e7

            // ── Weight helpers ────────────────────────────────────────────
            func f16(_ shape: [Int]) -> MLXArray { (MLXRandom.normal(shape) * 0.05).asType(.float16) }
            func q4(_ n: Int, _ k: Int) -> (MLXArray, MLXArray, MLXArray) {
                let wf = (MLXRandom.normal([n, k]) * 0.05).asType(.float16)
                let (q, s, bOpt) = MLX.quantized(wf, groupSize: gs, bits: 4, mode: .affine)
                return (q, s, bOpt!)
            }
            func q4e(_ e: Int, _ n: Int, _ k: Int) -> (MLXArray, MLXArray, MLXArray) {
                let wf = (MLXRandom.normal([e, n, k]) * 0.05).asType(.float16)
                let (q, s, bOpt) = MLX.quantized(wf, groupSize: gs, bits: 4, mode: .affine)
                return (q, s, bOpt!)
            }
            // Plain F16 proj weights  [out, in] row-major
            let fcW  = f16([H, 2 * H])          // fc [H, 2H]
            let qW   = f16([nH * 2 * hD, H])    // q+gate qd2 format
            let kW   = f16([nKV * hD, H])
            let vW   = f16([nKV * hD, H])
            let oW   = f16([H, nH * hD])
            let gateW = f16([E, H])              // router gate
            let shGW  = f16([I, H]); let shUW = f16([I, H]); let shDW = f16([H, I])
            let sharedGW = f16([1, H])           // shared_expert_gate [1,H]
            // Norms F16 (already RECOVERED — no +1 shift needed in tests)
            let preEmb = f16([H]); let preHid = f16([H])
            let inputLN = f16([H]); let postLN = f16([H])
            let qNorm = f16([hD]); let kNorm = f16([hD]); let finalNorm = f16([H])
            // 4-bit quantized triples
            let (embedWq, embedSc, embedBi) = q4(V, H)            // embed [V,H]
            let (swGWq, swGSc, swGBi) = q4e(E, I, H)
            let (swUWq, swUSc, swUBi) = q4e(E, I, H)
            let (swDWq, swDSc, swDBi) = q4e(E, H, I)
            let (lmWq, lmSc, lmBi) = q4(V, H)                     // lm_head [V,H]

            MLX.eval([fcW, qW, kW, vW, oW, gateW, shGW, shUW, shDW, sharedGW,
                      preEmb, preHid, inputLN, postLN, qNorm, kNorm, finalNorm,
                      embedWq, embedSc, embedBi, swGWq, swGSc, swGBi,
                      swUWq, swUSc, swUBi, swDWq, swDSc, swDBi, lmWq, lmSc, lmBi])

            // ── WeightsSpec ───────────────────────────────────────────────
            let spec = SeedlessFusedVerify.SeedlessMTPHead.WeightsSpec(
                H: H, V: V, E: E, I: I, Ktop: Ktop,
                numHeads: nH, numKV: nKV, headDim: hD, ropeDim: rD,
                ropeBase: ropeBase, eps: eps, maxSeqLen: 128,
                expertGroupSize: gs,
                fc: fcW,
                qW: qW, kW: kW, vW: vW, oW: oW,
                routerGate: gateW,
                shGate: shGW, shUp: shUW, shDown: shDW, sharedGate: sharedGW,
                preEmb: preEmb, preHid: preHid, inputLN: inputLN, postLN: postLN,
                qNorm: qNorm, kNorm: kNorm, finalNorm: finalNorm,
                embedWq: embedWq, embedSc: embedSc, embedBi: embedBi,
                swGWq: swGWq, swGSc: swGSc, swGBi: swGBi,
                swUWq: swUWq, swUSc: swUSc, swUBi: swUBi,
                swDWq: swDWq, swDSc: swDSc, swDBi: swDBi,
                lmWq: lmWq, lmSc: lmSc, lmBi: lmBi)

            // RED: stub returns nil
            guard let head = SeedlessFusedVerify.SeedlessMTPHead(spec: spec) else {
                return (false, "init? nil (STUB not implemented)")
            }

            // ── MLX reference: production class composition (MTPHead:73-90) ──
            // Mirrors callWithHidden exactly: embed → preNorm → fc → attn(KVCache) → MoE → finalNorm → lm_head.
            let attnMLX = AttentionLayer(
                numHeads: nH, numKVHeads: nKV, headDim: hD, ropeDim: rD,
                ropeBase: ropeBase, eps: eps,
                qProj: .plain(qW),
                kProj: .plain(kW),
                vProj: .plain(vW),
                oProj: .plain(oW),
                qNorm: qNorm,
                kNorm: kNorm)
            let moeMLX = MoEBlock(
                topK: Ktop, numExperts: E, normTopk: true, expertBits: 4,
                expertGroupSize: gs,
                gate: .plain(gateW),
                swGateW: swGWq, swGateS: swGSc, swGateB: swGBi,
                swUpW: swUWq, swUpS: swUSc, swUpB: swUBi,
                swDownW: swDWq, swDownS: swDSc, swDownB: swDBi,
                shGate: .plain(shGW), shUp: .plain(shUW), shDown: .plain(shDW),
                sharedGate: .plain(sharedGW))
            // lm_head proj
            let lmHead = Proj.quantized(lmWq, lmSc, lmBi, 4)

            // Forward function: hPrev [1,1,H] f16, tok Int32 → argmax Int
            // KVCache is caller-supplied (nil for no history).
            func mlxDraft(_ hPrev: MLXArray, tok: Int32, cache: KVCache?) -> Int {
                let tokArr = MLXArray([tok], [1, 1])
                let emb = ModelHead.embed(ids: tokArr, weight: embedWq, scales: embedSc,
                                          biases: embedBi, bits: 4)          // [1,1,H]
                let e = MLXFast.rmsNorm(emb, weight: preEmb, eps: eps)
                let hh = MLXFast.rmsNorm(hPrev, weight: preHid, eps: eps)
                let cat = MLX.concatenated([e, hh], axis: -1)                // [1,1,2H]
                var x = MLX.matmul(cat, fcW.transposed())                    // [1,1,H]
                let r = attnMLX(MLXFast.rmsNorm(x, weight: inputLN, eps: eps), cache: cache)
                x = x + r
                let B = x.dim(0), L = x.dim(1), Hd = x.dim(2)
                let post = MLXFast.rmsNorm(x, weight: postLN, eps: eps)
                x = x + moeMLX(post.reshaped([B * L, Hd])).reshaped([B, L, Hd])
                let normed = MLXFast.rmsNorm(x, weight: finalNorm, eps: eps)
                let logits = lmHead.apply(normed)                             // [1,1,V]
                logits.eval()
                return MLX.argMax(logits[0, 0], axis: -1).item(Int.self)
            }

            // ── (a) len=0 draft: raw argmax == MLX argmax ────────────────
            let tok0: Int32 = 42
            let hPrev0 = MLXRandom.normal([1, 1, H]).asType(.float16); hPrev0.eval()
            guard let hBuf0 = SeedlessMetalForward.mtlBuf(hPrev0.reshaped([1, H]).asType(.float16), device)
            else { return (false, "hBuf0 nil") }

            guard let rawDraft0 = head.draftArgmax(hPrevBuf: hBuf0, hPrevRow: 0, tok: tok0)
            else { return (false, "draftArgmax nil len=0 (STUB not implemented)") }

            let refDraft0 = mlxDraft(hPrev0, tok: tok0, cache: nil)
            if rawDraft0 != refDraft0 {
                return (false, "(a) len=0 draft mismatch: raw=\(rawDraft0) ref=\(refDraft0)")
            }
            if head.len != 0 {
                return (false, "(a) len must be 0 after draftArgmax (READ-ONLY violated): len=\(head.len)")
            }

            // ── (b) feedPairs 2 pairs → draft with KV history ────────────
            // Build hBuf for 2 feed rows [2, H] f16.
            let hFeed = MLXRandom.normal([2, H]).asType(.float16); hFeed.eval()
            let toks2: [Int32] = [7, 19]
            guard let hBufFeed = SeedlessMetalForward.mtlBuf(hFeed, device)
            else { return (false, "hBufFeed nil") }

            // Feed 2 pairs into raw head.
            let feedOK = head.feedPairs(hBuf: hBufFeed, rowRange: 0..<2, toks: toks2)
            if !feedOK { return (false, "(b) feedPairs returned false (STUB not implemented)") }
            if head.len != 2 { return (false, "(b) len must be 2 after feedPairs: got \(head.len)") }

            // MLX reference: build KVCache by feeding same 2 pairs sequentially.
            // Each pair: embed(tok) + hPrev row → attn with KVCache writes → KV grows.
            // This mirrors feedPairs semantics: lm_head skipped, KV committed.
            let refCache = KVCache()
            for i in 0..<2 {
                let hRow = hFeed[i ..< i+1].reshaped([1, 1, H])
                _ = mlxDraft(hRow, tok: toks2[i], cache: refCache)
            }

            // Now raw draft at len=2 must match MLX draft with same KV history.
            let tok1: Int32 = 99
            let hPrev1 = MLXRandom.normal([1, 1, H]).asType(.float16); hPrev1.eval()
            guard let hBuf1 = SeedlessMetalForward.mtlBuf(hPrev1.reshaped([1, H]).asType(.float16), device)
            else { return (false, "hBuf1 nil") }

            guard let rawDraft1 = head.draftArgmax(hPrevBuf: hBuf1, hPrevRow: 0, tok: tok1)
            else { return (false, "(b) draftArgmax nil after feedPairs (STUB not implemented)") }

            let refDraft1 = mlxDraft(hPrev1, tok: tok1, cache: refCache)
            if rawDraft1 != refDraft1 {
                return (false, "(b) post-feed draft mismatch: raw=\(rawDraft1) ref=\(refDraft1)")
            }
            let lenAfterDraft1 = head.len
            if lenAfterDraft1 != 2 {
                return (false, "(b) len changed after draftArgmax: want 2 got \(lenAfterDraft1)")
            }

            // ── (c) re-draft: same token, len unchanged ───────────────────
            guard let rawDraft1b = head.draftArgmax(hPrevBuf: hBuf1, hPrevRow: 0, tok: tok1)
            else { return (false, "(c) re-draft nil (STUB not implemented)") }
            if rawDraft1b != rawDraft1 {
                return (false, "(c) re-draft mismatch: first=\(rawDraft1) second=\(rawDraft1b)")
            }
            if head.len != 2 {
                return (false, "(c) len changed after re-draft: want 2 got \(head.len)")
            }

            return (true, "ok")
        }

        // Test 75 (T-kv): batch feedPairs vs sequential feedPairs produce identical drafts.
        // Same synthetic geometry as T-head. No MLX reference needed (raw vs raw).
        // head A: feedPairs(hBuf, rowRange: 0..<3, toks: 3) — batch ingest.
        // head B: feedPairs 3 × (rowRange: i..<i+1, toks: [toks[i]]) — sequential ingest.
        // Assert: draftArgmax(same hPrev, tok) agrees for A and B, and both report len==3.
        // Exercises rope position writing: each pair maps to a distinct position, so any
        // position-index bug (e.g., all rows landing at position 0) would produce different
        // attention and diverge. RED: init? nil → "STUB not implemented" FAIL.
        run("mtp_kv_batch_vs_sequential") {
            guard let (device, _) = SeedlessMetalForward.ensure() else {
                return (false, "no device")
            }

            let H = 2048, V = 256, E = 16, Ktop = 8, I = 512
            let nH = 16, nKV = 2, hD = 256, rD = 64
            let gs = 64
            let eps: Float = 1e-6
            let ropeBase: Float = 1e7

            // Shared weight construction (same idiom as T-head).
            func f16(_ shape: [Int]) -> MLXArray { (MLXRandom.normal(shape) * 0.05).asType(.float16) }
            func q4(_ n: Int, _ k: Int) -> (MLXArray, MLXArray, MLXArray) {
                let wf = (MLXRandom.normal([n, k]) * 0.05).asType(.float16)
                let (q, s, bOpt) = MLX.quantized(wf, groupSize: gs, bits: 4, mode: .affine)
                return (q, s, bOpt!)
            }
            func q4e(_ e: Int, _ n: Int, _ k: Int) -> (MLXArray, MLXArray, MLXArray) {
                let wf = (MLXRandom.normal([e, n, k]) * 0.05).asType(.float16)
                let (q, s, bOpt) = MLX.quantized(wf, groupSize: gs, bits: 4, mode: .affine)
                return (q, s, bOpt!)
            }
            let fcW  = f16([H, 2 * H])
            let qW   = f16([nH * 2 * hD, H])
            let kW   = f16([nKV * hD, H])
            let vW   = f16([nKV * hD, H])
            let oW   = f16([H, nH * hD])
            let gateW = f16([E, H])
            let shGW  = f16([I, H]); let shUW = f16([I, H]); let shDW = f16([H, I])
            let sharedGW = f16([1, H])
            let preEmb = f16([H]); let preHid = f16([H])
            let inputLN = f16([H]); let postLN = f16([H])
            let qNorm = f16([hD]); let kNorm = f16([hD]); let finalNorm = f16([H])
            let (embedWq, embedSc, embedBi) = q4(V, H)
            let (swGWq, swGSc, swGBi) = q4e(E, I, H)
            let (swUWq, swUSc, swUBi) = q4e(E, I, H)
            let (swDWq, swDSc, swDBi) = q4e(E, H, I)
            let (lmWq, lmSc, lmBi) = q4(V, H)

            MLX.eval([fcW, qW, kW, vW, oW, gateW, shGW, shUW, shDW, sharedGW,
                      preEmb, preHid, inputLN, postLN, qNorm, kNorm, finalNorm,
                      embedWq, embedSc, embedBi, swGWq, swGSc, swGBi,
                      swUWq, swUSc, swUBi, swDWq, swDSc, swDBi, lmWq, lmSc, lmBi])

            func makeSpec() -> SeedlessFusedVerify.SeedlessMTPHead.WeightsSpec {
                SeedlessFusedVerify.SeedlessMTPHead.WeightsSpec(
                    H: H, V: V, E: E, I: I, Ktop: Ktop,
                    numHeads: nH, numKV: nKV, headDim: hD, ropeDim: rD,
                    ropeBase: ropeBase, eps: eps, maxSeqLen: 128,
                    expertGroupSize: gs,
                    fc: fcW,
                    qW: qW, kW: kW, vW: vW, oW: oW,
                    routerGate: gateW,
                    shGate: shGW, shUp: shUW, shDown: shDW, sharedGate: sharedGW,
                    preEmb: preEmb, preHid: preHid, inputLN: inputLN, postLN: postLN,
                    qNorm: qNorm, kNorm: kNorm, finalNorm: finalNorm,
                    embedWq: embedWq, embedSc: embedSc, embedBi: embedBi,
                    swGWq: swGWq, swGSc: swGSc, swGBi: swGBi,
                    swUWq: swUWq, swUSc: swUSc, swUBi: swUBi,
                    swDWq: swDWq, swDSc: swDSc, swDBi: swDBi,
                    lmWq: lmWq, lmSc: lmSc, lmBi: lmBi)
            }

            // Two independent heads with identical weights.
            guard let headA = SeedlessFusedVerify.SeedlessMTPHead(spec: makeSpec()),
                  let headB = SeedlessFusedVerify.SeedlessMTPHead(spec: makeSpec()) else {
                return (false, "init? nil (STUB not implemented)")
            }

            // Feed data: 3 rows.
            let nPairs = 3
            let hFeed = MLXRandom.normal([nPairs, H]).asType(.float16); hFeed.eval()
            let toks: [Int32] = [3, 11, 27]
            guard let hBufFeed = SeedlessMetalForward.mtlBuf(hFeed, device)
            else { return (false, "hBufFeed nil") }

            // head A: batch ingest all 3 at once.
            let okA = headA.feedPairs(hBuf: hBufFeed, rowRange: 0..<nPairs, toks: toks)
            if !okA { return (false, "headA feedPairs batch failed (STUB not implemented)") }
            if headA.len != nPairs { return (false, "headA len=\(headA.len) want \(nPairs)") }

            // head B: sequential ingest 1 row at a time.
            for i in 0..<nPairs {
                let ok = headB.feedPairs(hBuf: hBufFeed, rowRange: i..<(i+1), toks: [toks[i]])
                if !ok { return (false, "headB feedPairs seq i=\(i) failed (STUB not implemented)") }
            }
            if headB.len != nPairs { return (false, "headB len=\(headB.len) want \(nPairs)") }

            // Draft from both heads must agree.
            let tokQ: Int32 = 55
            let hQ = MLXRandom.normal([1, H]).asType(.float16); hQ.eval()
            guard let hBufQ = SeedlessMetalForward.mtlBuf(hQ, device)
            else { return (false, "hBufQ nil") }

            guard let draftA = headA.draftArgmax(hPrevBuf: hBufQ, hPrevRow: 0, tok: tokQ)
            else { return (false, "headA draftArgmax nil (STUB not implemented)") }
            guard let draftB = headB.draftArgmax(hPrevBuf: hBufQ, hPrevRow: 0, tok: tokQ)
            else { return (false, "headB draftArgmax nil (STUB not implemented)") }

            if draftA != draftB {
                return (false, "batch vs sequential draft mismatch: A=\(draftA) B=\(draftB)")
            }

            // len must still be nPairs (draftArgmax is READ-ONLY).
            if headA.len != nPairs || headB.len != nPairs {
                return (false, "len changed after draftArgmax: A=\(headA.len) B=\(headB.len)")
            }

            return (true, "ok")
        }

        // Test 76 (T-seam, ①③ Step 4): Tell.mtpDraftSpan — the D==0 draft seam.
        // (a) head nil (QWISP_MTP_DRAFT unset) → nil: the greedy path is untouched =
        //     flag-off byte-identity by construction.
        // (b) rowOfU < 0 (invalid hidden row, e.g. after a chained span) → nil.
        // (c) active head → non-nil token in [0,V), and head KV len UNCHANGED through
        //     the seam (draft is READ-ONLY; feedPairs is the sole writer).
        // (d) determinism: two identical spans return the same token (read-only ⇒
        //     same state ⇒ same argmax — a wrong-but-stable draft is lossless because
        //     verify rejects it; a nondeterministic seam would break rowOfU reasoning).
        run("mtp_draft_span_seam") {
            guard let (device, _) = SeedlessMetalForward.ensure() else {
                return (false, "no device")
            }
            // Synthetic head, same real-shape geometry as tests 74/75 (H=2048, gs=64).
            let H = 2048, V = 256, E = 16, Ktop = 8, I = 512
            let nH = 16, nKV = 2, hD = 256, rD = 64
            let gs = 64
            func f16(_ shape: [Int]) -> MLXArray { (MLXRandom.normal(shape) * 0.05).asType(.float16) }
            func q4(_ n: Int, _ k: Int) -> (MLXArray, MLXArray, MLXArray) {
                let wf = (MLXRandom.normal([n, k]) * 0.05).asType(.float16)
                let (q, s, bOpt) = MLX.quantized(wf, groupSize: gs, bits: 4, mode: .affine)
                return (q, s, bOpt!)
            }
            func q4e(_ e: Int, _ n: Int, _ k: Int) -> (MLXArray, MLXArray, MLXArray) {
                let wf = (MLXRandom.normal([e, n, k]) * 0.05).asType(.float16)
                let (q, s, bOpt) = MLX.quantized(wf, groupSize: gs, bits: 4, mode: .affine)
                return (q, s, bOpt!)
            }
            let (embedWq, embedSc, embedBi) = q4(V, H)
            let (swGWq, swGSc, swGBi) = q4e(E, I, H)
            let (swUWq, swUSc, swUBi) = q4e(E, I, H)
            let (swDWq, swDSc, swDBi) = q4e(E, H, I)
            let (lmWq, lmSc, lmBi) = q4(V, H)
            let spec = SeedlessFusedVerify.SeedlessMTPHead.WeightsSpec(
                H: H, V: V, E: E, I: I, Ktop: Ktop,
                numHeads: nH, numKV: nKV, headDim: hD, ropeDim: rD,
                ropeBase: 1e7, eps: 1e-6, maxSeqLen: 128,
                expertGroupSize: gs,
                fc: f16([H, 2 * H]),
                qW: f16([nH * 2 * hD, H]), kW: f16([nKV * hD, H]),
                vW: f16([nKV * hD, H]), oW: f16([H, nH * hD]),
                routerGate: f16([E, H]),
                shGate: f16([I, H]), shUp: f16([I, H]), shDown: f16([H, I]),
                sharedGate: f16([1, H]),
                preEmb: f16([H]), preHid: f16([H]), inputLN: f16([H]), postLN: f16([H]),
                qNorm: f16([hD]), kNorm: f16([hD]), finalNorm: f16([H]),
                embedWq: embedWq, embedSc: embedSc, embedBi: embedBi,
                swGWq: swGWq, swGSc: swGSc, swGBi: swGBi,
                swUWq: swUWq, swUSc: swUSc, swUBi: swUBi,
                swDWq: swDWq, swDSc: swDSc, swDBi: swDBi,
                lmWq: lmWq, lmSc: lmSc, lmBi: lmBi)
            guard let head = SeedlessFusedVerify.SeedlessMTPHead(spec: spec) else {
                return (false, "head init nil")
            }
            // Fake normed buffer: 4 rows, hidden of u at row 2.
            let hRows = (MLXRandom.normal([4, H]) * 0.05).asType(.float16); hRows.eval()
            guard let hBuf = SeedlessMetalForward.mtlBuf(hRows, device) else {
                return (false, "hBuf nil")
            }
            let u = 7

            // (a) flag-off: head nil → nil
            if Tell.mtpDraftSpan(head: nil, hPrevBuf: hBuf, rowOfU: 2, u: u) != nil {
                return (false, "(a) head nil must yield nil")
            }
            // (b) invalid row → nil
            if Tell.mtpDraftSpan(head: head, hPrevBuf: hBuf, rowOfU: -1, u: u) != nil {
                return (false, "(b) rowOfU=-1 must yield nil")
            }
            // (c) active: draft token in range, len unchanged
            guard let d1 = Tell.mtpDraftSpan(head: head, hPrevBuf: hBuf, rowOfU: 2, u: u)
            else { return (false, "(c) active span returned nil") }
            if d1 < 0 || d1 >= V { return (false, "(c) draft out of range: \(d1)") }
            if head.len != 0 { return (false, "(c) len changed through seam: \(head.len)") }
            // (d) determinism (read-only ⇒ identical repeat)
            guard let d2 = Tell.mtpDraftSpan(head: head, hPrevBuf: hBuf, rowOfU: 2, u: u)
            else { return (false, "(d) repeat span returned nil") }
            if d1 != d2 { return (false, "(d) nondeterministic: \(d1) vs \(d2)") }

            return (true, "ok")
        }

        // Test 77 (T-lifetime, ①③ Step 5 real-bug regression): SeedlessMTPHead must retain the
        // backing MLXArrays of its noCopy weight buffers. THE BUG: init converted weights via
        // temporary MLXArrays (asType) and bound their Metal buffers noCopy; once the caller's
        // WeightsSpec went out of scope, the MLX allocator recycled the weight memory and the
        // head's weights turned to garbage mid-run (found in run(): accept 0.00, NaN logits —
        // masked in validate/tests because their spec stayed alive in a local). Protocol:
        // draft with spec alive → drop the spec scope → churn the MLX allocator hard →
        // same (h, tok) draft must be identical.
        run("mtp_head_weight_lifetime") {
            guard let (device, _) = SeedlessMetalForward.ensure() else {
                return (false, "no device")
            }
            let H = 2048, V = 256, E = 16, Ktop = 8, I = 512
            let nH = 16, nKV = 2, hD = 256, rD = 64
            let gs = 64
            let hQ = (MLXRandom.normal([1, H]) * 0.05).asType(.float16); hQ.eval()
            guard let hBuf = SeedlessMetalForward.mtlBuf(hQ, device) else { return (false, "hBuf nil") }
            let tok: Int32 = 42

            var head: SeedlessFusedVerify.SeedlessMTPHead? = nil
            var d0 = -1
            do {  // spec lives ONLY in this scope
                func f16(_ shape: [Int]) -> MLXArray { (MLXRandom.normal(shape) * 0.05).asType(.float16) }
                func q4(_ n: Int, _ k: Int) -> (MLXArray, MLXArray, MLXArray) {
                    let wf = (MLXRandom.normal([n, k]) * 0.05).asType(.float16)
                    let (q, s, bOpt) = MLX.quantized(wf, groupSize: gs, bits: 4, mode: .affine)
                    return (q, s, bOpt!)
                }
                func q4e(_ e: Int, _ n: Int, _ k: Int) -> (MLXArray, MLXArray, MLXArray) {
                    let wf = (MLXRandom.normal([e, n, k]) * 0.05).asType(.float16)
                    let (q, s, bOpt) = MLX.quantized(wf, groupSize: gs, bits: 4, mode: .affine)
                    return (q, s, bOpt!)
                }
                let (embedWq, embedSc, embedBi) = q4(V, H)
                let (swGWq, swGSc, swGBi) = q4e(E, I, H)
                let (swUWq, swUSc, swUBi) = q4e(E, I, H)
                let (swDWq, swDSc, swDBi) = q4e(E, H, I)
                let (lmWq, lmSc, lmBi) = q4(V, H)
                let spec = SeedlessFusedVerify.SeedlessMTPHead.WeightsSpec(
                    H: H, V: V, E: E, I: I, Ktop: Ktop,
                    numHeads: nH, numKV: nKV, headDim: hD, ropeDim: rD,
                    ropeBase: 1e7, eps: 1e-6, maxSeqLen: 128,
                    expertGroupSize: gs,
                    fc: f16([H, 2 * H]),
                    qW: f16([nH * 2 * hD, H]), kW: f16([nKV * hD, H]),
                    vW: f16([nKV * hD, H]), oW: f16([H, nH * hD]),
                    routerGate: f16([E, H]),
                    shGate: f16([I, H]), shUp: f16([I, H]), shDown: f16([H, I]),
                    sharedGate: f16([1, H]),
                    preEmb: f16([H]), preHid: f16([H]), inputLN: f16([H]), postLN: f16([H]),
                    qNorm: f16([hD]), kNorm: f16([hD]), finalNorm: f16([H]),
                    embedWq: embedWq, embedSc: embedSc, embedBi: embedBi,
                    swGWq: swGWq, swGSc: swGSc, swGBi: swGBi,
                    swUWq: swUWq, swUSc: swUSc, swUBi: swUBi,
                    swDWq: swDWq, swDSc: swDSc, swDBi: swDBi,
                    lmWq: lmWq, lmSc: lmSc, lmBi: lmBi)
                guard let h = SeedlessFusedVerify.SeedlessMTPHead(spec: spec) else {
                    return (false, "head init nil")
                }
                head = h
                guard let d = h.draftArgmax(hPrevBuf: hBuf, hPrevRow: 0, tok: tok) else {
                    return (false, "draft (spec alive) nil")
                }
                d0 = d
            }
            // spec + weight temporaries released → churn the MLX allocator so any
            // non-retained noCopy backing gets recycled and overwritten.
            for _ in 0 ..< 12 {
                let junk = MLXRandom.normal([2048, 2048]).asType(.float16)
                junk.eval()
            }
            guard let d1 = head!.draftArgmax(hPrevBuf: hBuf, hPrevRow: 0, tok: tok) else {
                return (false, "draft (after churn) nil")
            }
            if d0 != d1 {
                return (false, "weights corrupted after spec release: before=\(d0) after=\(d1) (noCopy lifetime)")
            }
            return (true, "ok")
        }

        // Test 78 (T-fold, ①③ Step 6): commitLastDraft ≡ feedPairs for the drafted pair.
        // draftArgmax already writes the pair's k/v at pos=len (same encodeForward feedPairs
        // runs, deterministic kernels) — so draft+commitLastDraft must leave the head in a
        // state indistinguishable from an explicit feedPairs of the same pair.
        // Protocol: two heads, identical weights. A: draftArgmax(h0,t0) → commitLastDraft.
        // B: feedPairs(h0,[t0]). Then both draft (h1,t1): tokens must match, len must match.
        run("mtp_commit_last_draft_fold") {
            guard let (device, _) = SeedlessMetalForward.ensure() else {
                return (false, "no device")
            }
            let H = 2048, V = 256, E = 16, Ktop = 8, I = 512
            let nH = 16, nKV = 2, hD = 256, rD = 64
            let gs = 64
            func f16(_ shape: [Int]) -> MLXArray { (MLXRandom.normal(shape) * 0.05).asType(.float16) }
            func q4(_ n: Int, _ k: Int) -> (MLXArray, MLXArray, MLXArray) {
                let wf = (MLXRandom.normal([n, k]) * 0.05).asType(.float16)
                let (q, s, bOpt) = MLX.quantized(wf, groupSize: gs, bits: 4, mode: .affine)
                return (q, s, bOpt!)
            }
            func q4e(_ e: Int, _ n: Int, _ k: Int) -> (MLXArray, MLXArray, MLXArray) {
                let wf = (MLXRandom.normal([e, n, k]) * 0.05).asType(.float16)
                let (q, s, bOpt) = MLX.quantized(wf, groupSize: gs, bits: 4, mode: .affine)
                return (q, s, bOpt!)
            }
            let (embedWq, embedSc, embedBi) = q4(V, H)
            let (swGWq, swGSc, swGBi) = q4e(E, I, H)
            let (swUWq, swUSc, swUBi) = q4e(E, I, H)
            let (swDWq, swDSc, swDBi) = q4e(E, H, I)
            let (lmWq, lmSc, lmBi) = q4(V, H)
            let spec = SeedlessFusedVerify.SeedlessMTPHead.WeightsSpec(
                H: H, V: V, E: E, I: I, Ktop: Ktop,
                numHeads: nH, numKV: nKV, headDim: hD, ropeDim: rD,
                ropeBase: 1e7, eps: 1e-6, maxSeqLen: 128,
                expertGroupSize: gs,
                fc: f16([H, 2 * H]),
                qW: f16([nH * 2 * hD, H]), kW: f16([nKV * hD, H]),
                vW: f16([nKV * hD, H]), oW: f16([H, nH * hD]),
                routerGate: f16([E, H]),
                shGate: f16([I, H]), shUp: f16([I, H]), shDown: f16([H, I]),
                sharedGate: f16([1, H]),
                preEmb: f16([H]), preHid: f16([H]), inputLN: f16([H]), postLN: f16([H]),
                qNorm: f16([hD]), kNorm: f16([hD]), finalNorm: f16([H]),
                embedWq: embedWq, embedSc: embedSc, embedBi: embedBi,
                swGWq: swGWq, swGSc: swGSc, swGBi: swGBi,
                swUWq: swUWq, swUSc: swUSc, swUBi: swUBi,
                swDWq: swDWq, swDSc: swDSc, swDBi: swDBi,
                lmWq: lmWq, lmSc: lmSc, lmBi: lmBi)
            guard let headA = SeedlessFusedVerify.SeedlessMTPHead(spec: spec),
                  let headB = SeedlessFusedVerify.SeedlessMTPHead(spec: spec) else {
                return (false, "head init nil")
            }
            let h0 = (MLXRandom.normal([1, H]) * 0.05).asType(.float16); h0.eval()
            let h1 = (MLXRandom.normal([1, H]) * 0.05).asType(.float16); h1.eval()
            guard let b0 = SeedlessMetalForward.mtlBuf(h0, device),
                  let b1 = SeedlessMetalForward.mtlBuf(h1, device) else { return (false, "hBuf nil") }
            let t0: Int32 = 9, t1: Int32 = 77

            // A: draft then fold-commit
            guard let _ = headA.draftArgmax(hPrevBuf: b0, hPrevRow: 0, tok: t0)
            else { return (false, "A draft nil") }
            guard headA.commitLastDraft() else { return (false, "A commitLastDraft failed") }
            // B: explicit feed
            guard headB.feedPairs(hBuf: b0, rowRange: 0 ..< 1, toks: [t0])
            else { return (false, "B feedPairs failed") }

            if headA.len != 1 || headB.len != 1 {
                return (false, "len mismatch: A=\(headA.len) B=\(headB.len) want 1")
            }
            guard let dA = headA.draftArgmax(hPrevBuf: b1, hPrevRow: 0, tok: t1),
                  let dB = headB.draftArgmax(hPrevBuf: b1, hPrevRow: 0, tok: t1)
            else { return (false, "post-commit draft nil") }
            if dA != dB {
                return (false, "fold != feed: A=\(dA) B=\(dB)")
            }
            return (true, "ok")
        }

        // Test 79: seedless_config — facade tier sizing (productization step 2).
        // Pure/GPU-free seam; mirrors Tell.run()'s tier arithmetic:
        //   maxK        = streaming ? max(4, C*3/8) : 96
        //   maxM        = max(pendingCap(24) + maxK + 1, 64)
        //   maxSeqLen   = promptLen + maxTokens + maxK + 64
        run("seedless_config") {
            let s = SeedlessBackend.config(tier: .streaming(c: 64), promptLen: 10, maxTokens: 48)
            let sWant = SeedlessBackend.Config(isStreaming: true, c: 64, maxK: 24, maxM: 64, maxSeqLen: 146)
            if s != sWant { return (false, "streaming c=64 got \(s) want \(sWant)") }
            let r = SeedlessBackend.config(tier: .resident, promptLen: 10, maxTokens: 48)
            let rWant = SeedlessBackend.Config(isStreaming: false, c: 256, maxK: 96, maxM: 121, maxSeqLen: 218)
            if r != rWant { return (false, "resident got \(r) want \(rWant)") }
            return (true, "ok")
        }

        // ── W1b: mixed-precision (4-bit core + 2-bit tail) gather kernels ────
        //
        // WRITE-LOCKED: implementer MUST NOT modify these two tests.
        // They encode the notes/18 W1 "Mixed dispatch design" acceptance gate.
        //
        // Geometry E=64 = K4=16 four-bit core + M2=48 two-bit tail, K=2048, N=512,
        // Ktop=4, M∈{1,2,9,17,25}. A single mixed dispatch serves routed rows whose
        // slot may be either class: slot s<16 → 4-bit path (w4[s]); s>=16 → 2-bit path
        // (w2[s-16]). scales/biases are ONE global-slot-indexed buffer (concat of the
        // per-class quant outputs), so scales[s]==s4[s] for s<16 and ==s2[s-16] else.
        //
        // Reference uses ONLY already-proven production kernels: the 4-bit branch is
        // exactly gatherQmmRows(M:1,Ktop:1) on the 4-bit buffers, the 2-bit branch is
        // gqmm2Rows(M:1,Ktop:1) on the 2-bit buffers, composed per (row,slot) pair and
        // concatenated → [M*Ktop, N].  Both tests are RED ("not implemented") on the
        // stub tree and GREEN only when the mixed kernel copies both branch bodies
        // verbatim (add order + safe-math preserved).

        // Fixed 48-entry index pool spanning BOTH classes (values <16 and >=16 present),
        // cycled so every M variant covers a mix of 4-bit and 2-bit slots.
        let mixPool: [Int32] = [ 3, 20, 40, 62,  0, 17, 25, 50,
                                33,  7, 55, 12, 41, 15,  2, 60,
                                 8, 30, 11, 45, 22, 38, 61,  5,
                                18, 44, 29,  1, 36, 52,  6, 19,
                                 9, 27, 48, 14, 37,  4, 23, 53,
                                10, 16, 31, 39, 47, 57, 13, 21]

        // Test 80 (W1b-1): gqmmMixRows ≡ per-(row,slot)-class gatherQmmRows/gqmm2Rows.
        run("gqmm_mix_rows_bitexact") {
            let E = 64, K4 = 16, K = 2048, N = 512, Ktop = 4
            // f16 weight fixture; split into 4-bit core (0..<16) and 2-bit tail (16..<64).
            let wf = MLXRandom.normal([E, N, K]).asType(.float16)
            let (w4, s4, b4o) = MLX.quantized(wf[0 ..< K4],  groupSize: 64, bits: 4, mode: .affine)
            let (w2, s2, b2o) = MLX.quantized(wf[K4 ..< E],  groupSize: 64, bits: 2, mode: .affine)
            guard let b4 = b4o, let b2 = b2o else { return (false, "biases nil") }
            // Global-slot-indexed scales/biases: concat of the two per-class outputs.
            let scales = MLX.concatenated([s4, s2], axis: 0).asType(.float16)
            let biases = MLX.concatenated([b4, b2], axis: 0).asType(.float16)
            MLX.eval([w4, s4, b4, w2, s2, b2, scales, biases])

            // Reference for one (mk-indexed) inds vector against ONE gqmmMixRows call.
            func check(_ x: MLXArray, _ indsFlat: [Int32], _ M: Int, lhsPer: Bool) -> (Bool, String) {
                let inds = MLXArray(indsFlat, [M * Ktop]); inds.eval()
                var refParts: [MLXArray] = []
                for mk in 0 ..< M * Ktop {
                    let row = lhsPer ? mk : mk / Ktop
                    let xRow = x[row ..< row + 1]; xRow.eval()   // [1, K]
                    let s = Int(indsFlat[mk])
                    let r: MLXArray?
                    if s < K4 {
                        let si = MLXArray([Int32(s)], [1]); si.eval()
                        r = SeedlessMetalForward.gatherQmmRows(xRow, w4, scales: s4, biases: b4,
                                                               inds: si, M: 1, Ktop: 1, K: K, N: N)
                    } else {
                        let si = MLXArray([Int32(s - K4)], [1]); si.eval()
                        r = SeedlessMetalForward.gqmm2Rows(xRow, w2, scales: s2, biases: b2,
                                                           inds: si, M: 1, Ktop: 1, K: K, N: N)
                    }
                    guard let rr = r else { return (false, "ref nil M=\(M) mk=\(mk) s=\(s)") }
                    rr.eval(); refParts.append(rr)   // [1, N]
                }
                let ref = MLX.concatenated(refParts, axis: 0); ref.eval()   // [M*Ktop, N]
                guard let got = SeedlessMetalForward.gqmmMixRows(x, w4: w4, w2: w2,
                                                                 scales: scales, biases: biases,
                                                                 inds: inds, K4: K4,
                                                                 M: M, Ktop: Ktop, K: K, N: N,
                                                                 lhsPerExpert: lhsPer)
                else { return (false, "not implemented (M=\(M) lhsPer=\(lhsPer))") }
                got.eval()
                return bitEqual(got, ref)
            }

            for M in [1, 2, 9, 17, 25] {
                let x = MLXRandom.normal([M, K]).asType(.float16); x.eval()
                let indsFlat = (0 ..< M * Ktop).map { mixPool[$0 % mixPool.count] }
                let (ok, d) = check(x, indsFlat, M, lhsPer: false)
                if !ok { return (false, "M=\(M): \(d)") }
            }
            // lhsPerExpert:true — per-row lhs, x[M*Ktop, K].
            do {
                let M = 9
                let x = MLXRandom.normal([M * Ktop, K]).asType(.float16); x.eval()
                let indsFlat = (0 ..< M * Ktop).map { mixPool[$0 % mixPool.count] }
                let (ok, d) = check(x, indsFlat, M, lhsPer: true)
                if !ok { return (false, "lhsPer M=\(M): \(d)") }
            }
            return (true, "ok")
        }

        // Test 81 (W1b-2): gqmmMixSwigluRows ≡ gqmmMixRows(gate) ⊗ gqmmMixRows(up) via swigluRaw.
        // Reference composes the (already test-80-validated) mixed gather twice through the
        // production swigluRaw — bit-exact only if the fused epilogue is copied verbatim from
        // gqmm4_swiglu_rows (swigluRaw implements the identical half/sigmoid sequence).
        run("gqmm_mix_swiglu_rows_bitexact") {
            let E = 64, K4 = 16, K = 2048, N = 512, Ktop = 4
            func mixWeights(_ seed: MLXArray) -> (MLXArray, MLXArray, MLXArray, MLXArray, MLXArray) {
                let (q4, s4, b4o) = MLX.quantized(seed[0 ..< K4], groupSize: 64, bits: 4, mode: .affine)
                let (q2, s2, b2o) = MLX.quantized(seed[K4 ..< E], groupSize: 64, bits: 2, mode: .affine)
                let sc = MLX.concatenated([s4, s2], axis: 0).asType(.float16)
                let bi = MLX.concatenated([b4o!, b2o!], axis: 0).asType(.float16)
                MLX.eval([q4, q2, sc, bi])
                return (q4, q2, sc, bi, seed)   // seed returned only to keep it alive
            }
            // Independent gate and up fixtures, each split/quantized the same way.
            let gf = MLXRandom.normal([E, N, K]).asType(.float16)
            let uf = MLXRandom.normal([E, N, K]).asType(.float16)
            let (gw4, gw2, gsc, gbi, _) = mixWeights(gf)
            let (uw4, uw2, usc, ubi, _) = mixWeights(uf)

            for M in [1, 2, 9, 17, 25] {
                let x = MLXRandom.normal([M, K]).asType(.float16)
                let indsFlat = (0 ..< M * Ktop).map { mixPool[$0 % mixPool.count] }
                let inds = MLXArray(indsFlat, [M * Ktop])
                MLX.eval([x, inds])
                // Reference: mixed gather (gate) and mixed gather (up), then swiglu.
                guard let g = SeedlessMetalForward.gqmmMixRows(x, w4: gw4, w2: gw2,
                                                               scales: gsc, biases: gbi, inds: inds,
                                                               K4: K4, M: M, Ktop: Ktop, K: K, N: N),
                      let u = SeedlessMetalForward.gqmmMixRows(x, w4: uw4, w2: uw2,
                                                               scales: usc, biases: ubi, inds: inds,
                                                               K4: K4, M: M, Ktop: Ktop, K: K, N: N)
                else { return (false, "ref mix gather nil (M=\(M))") }
                g.eval(); u.eval()
                guard let hRef = SeedlessMetalForward.swigluRaw(g, u)
                else { return (false, "ref swiglu nil (M=\(M))") }
                hRef.eval()
                // Fused kernel under test.
                guard let hGot = SeedlessMetalForward.gqmmMixSwigluRows(x,
                                    gw4: gw4, gw2: gw2, gsc: gsc, gbi: gbi,
                                    uw4: uw4, uw2: uw2, usc: usc, ubi: ubi, inds: inds,
                                    K4: K4, M: M, Ktop: Ktop, K: K, N: N)
                else { return (false, "not implemented (M=\(M))") }
                hGot.eval()
                let (ok, d) = bitEqual(hGot, hRef)
                if !ok { return (false, "M=\(M): \(d)") }
            }
            return (true, "ok")
        }

        // ── W2 (notes/18): MixedResidency — 4-bit core + 2-bit tail residency ─────
        // Test 82 (W2-1): pure-CPU mixed-slot cache bookkeeping (chunk_swap_atomic style).
        // K4=2 static-pinned core (global slots 0..<2), M2=3 LRU tail (global slots 2..<5).
        // Hand-calc: core experts return their pinned slot and never touch the tail LRU; tail
        // fills empties in call order then evicts least-recently-used (excluding slots touched
        // THIS call); a mid-call tail HIT protects its slot from a later same-call eviction; a
        // call with >M2 distinct tail experts returns nil (overflow) leaving state untouched so
        // the caller can retry. Global tail slot = K4 + local index. Counters track TAIL only.
        run("mixed_cache_bookkeeping") {
            var state = MixedCacheState(K4: 2, M2: 3)
            // pinCore sorts ASC → core slots {1:0, 5:1}.
            guard state.pinCore([5, 1]) else { return (false, "pinCore returned false") }
            if state.coreOf != [1: 0, 5: 1] { return (false, "coreOf \(state.coreOf) != [1:0,5:1]") }
            // wrong-count / duplicate guards (fresh states)
            var g2 = MixedCacheState(K4: 2, M2: 3)
            if g2.pinCore([3, 3]) { return (false, "pinCore accepted duplicates") }
            if g2.pinCore([3]) { return (false, "pinCore accepted count=1") }
            if g2.pinCore([3, 4, 5]) { return (false, "pinCore accepted count=3") }

            func chk(_ tag: String, _ got: (slots: [Int: Int], missJobs: [(e: Int, slot: Int)])?,
                     _ eslots: [Int: Int], _ ejobs: [[Int]], _ eh: Int, _ em: Int) -> String? {
                guard let g = got else { return "\(tag): ensure nil" }
                if g.slots != eslots { return "\(tag): slots \(g.slots) != \(eslots)" }
                let jobs = g.missJobs.map { [$0.e, $0.slot] }
                if jobs != ejobs { return "\(tag): missJobs \(jobs) != \(ejobs)" }
                if state.hits != eh { return "\(tag): hits \(state.hits) != \(eh)" }
                if state.misses != em { return "\(tag): misses \(state.misses) != \(em)" }
                return nil
            }

            // Call 1: 2 core hits + 2 tail misses fill empties local0,local1 (global 2,3).
            if let e = chk("c1", state.ensure([1, 5, 3, 6]),
                           [1: 0, 5: 1, 3: 2, 6: 3], [[3, 2], [6, 3]], 0, 2) { return (false, e) }
            if state.tailExpertAt != [3, 6, -1] { return (false, "c1 tailExpertAt \(state.tailExpertAt)") }

            // Call 2: core hit(1) + tail hit(3) + tail miss(7) fills last empty local2(global4).
            if let e = chk("c2", state.ensure([1, 3, 7]),
                           [1: 0, 3: 2, 7: 4], [[7, 4]], 1, 3) { return (false, e) }
            if state.tailExpertAt != [3, 6, 7] { return (false, "c2 tailExpertAt \(state.tailExpertAt)") }

            // Call 3: tail full → two misses evict LRU. e6(local1,oldest) evicted first; then
            // e3(local0) is oldest among non-touched (local1 was just touched this call).
            if let e = chk("c3", state.ensure([8, 2]),
                           [8: 3, 2: 2], [[8, 3], [2, 2]], 1, 5) { return (false, e) }
            if state.tailExpertAt != [2, 8, 7] { return (false, "c3 tailExpertAt \(state.tailExpertAt)") }

            // Call 4: e7(local2) is the OLDEST tail but is HIT first this call → re-ticked, so
            // the two following misses evict e8 then e2, NEVER the freshly-hit e7 (same-call rule).
            if let e = chk("c4", state.ensure([7, 10, 11]),
                           [7: 4, 10: 3, 11: 2], [[10, 3], [11, 2]], 2, 7) { return (false, e) }
            if state.tailExpertAt != [11, 10, 7] {
                return (false, "c4 tailExpertAt \(state.tailExpertAt) (same-call hit must protect e7@local2)")
            }

            // Call 5: 4 distinct tail experts > M2=3 → overflow → nil, state UNCHANGED.
            let h0 = state.hits, m0 = state.misses, ta0 = state.tailExpertAt
            if state.ensure([20, 21, 22, 23]) != nil { return (false, "overflow did not return nil") }
            if state.hits != h0 || state.misses != m0 || state.tailExpertAt != ta0 {
                return (false, "overflow mutated state: h=\(state.hits) m=\(state.misses) tail=\(state.tailExpertAt)")
            }

            // Call 6: retry with exactly M2=3 distinct tail experts → succeeds, evicts all three.
            if let e = chk("c6", state.ensure([20, 21, 22]),
                           [20: 4, 21: 3, 22: 2], [[20, 4], [21, 3], [22, 2]], 2, 10) { return (false, e) }
            if state.tailExpertAt != [22, 21, 20] { return (false, "c6 tailExpertAt \(state.tailExpertAt)") }

            // Call 7: core hit(5) + 3 tail hits (all resident) → no misses; a core expert does NOT
            // count toward the M2 overflow bound (4 experts, only 3 are tail).
            if let e = chk("c7", state.ensure([5, 20, 21, 22]),
                           [5: 1, 20: 4, 21: 3, 22: 2], [], 5, 10) { return (false, e) }
            if state.tailExpertAt != [22, 21, 20] { return (false, "c7 tailExpertAt \(state.tailExpertAt)") }
            return (true, "ok")
        }

        // Test 83 (W2-2): buddyTable over the GLOBAL slot space (core + tail residents combined).
        // State: K4=2 core {1:0, 5:1}; ensure([2,7]) → tail residents {2:2, 7:3} (global slots).
        // Combined hot set (residents sorted ASC) = [1,2,5,7] at global slots [0,2,1,3].
        // Cold experts (rotation (i+e)%n, strictly-greater tie-break, n=4):
        //   e0: coact{5:3}       → prefers CORE resident 5 → core slot 1, buddy 5
        //   e3: coact{7:4}       → prefers TAIL resident 7 → global tail slot 3, buddy 7
        //   e6: coact{1:2,5:2} tie → e6 rotation scans [5,7,1,2] → 5 wins (seen first) → slot 1, buddy 5
        //   e4: no coact          → fallbackSlot(1), buddy -1
        run("mixed_buddy_global_slots") {
            var state = MixedCacheState(K4: 2, M2: 3)
            guard state.pinCore([5, 1]) else { return (false, "pinCore failed") }
            guard state.ensure([2, 7]) != nil else { return (false, "ensure([2,7]) nil") }
            if state.tailSlotOf != [2: 2, 7: 3] { return (false, "tailSlotOf \(state.tailSlotOf) != [2:2,7:3]") }
            var coact = [[Int]](repeating: [Int](repeating: 0, count: 8), count: 8)
            coact[0][5] = 3
            coact[3][7] = 4
            coact[6][1] = 2; coact[6][5] = 2
            let (table, buddy) = state.buddyTable(coact: coact, nE: 8, fallbackSlot: 1)
            let eTable: [Int32] = [1, 0, 2, 3, 1, 1, 1, 3]
            let eBuddy: [Int32] = [5, 1, 2, 7, -1, 5, 5, 7]
            if table != eTable { return (false, "table \(table) != \(eTable)") }
            if buddy != eBuddy { return (false, "buddy \(buddy) != \(eBuddy)") }
            return (true, "ok")
        }

        // Test 84 (W2-3): MixedExpertArena pread layout over synthetic safetensors (NO model,
        // NO GPU compute). Two temp-dir checkpoints — 4-bit: weight [4,8,16] U32; 2-bit: weight
        // [4,8,8] U32; scales/biases [4,8,2] F16 both — filled with a distinct constant byte per
        // (dir, proj, part, expert). loadCore pulls 4-bit expert 3 → core slot 0; loadTailMany
        // pulls 2-bit experts 1,2 → global slots 2,3 (local 0,1). Verify each arena slot holds the
        // exact source expert's bytes, weight buffers split per class (w4[K4]/w2[M2]) and
        // scales/biases uniform[K4+M2] indexed by GLOBAL slot; plus sliceBytes relations.
        run("mixed_arena_pread_layout") {
            guard let (device, _) = SeedlessMetalForward.ensure() else { return (false, "no device") }
            let fm = FileManager.default
            let base = fm.temporaryDirectory.appendingPathComponent("qwisp_w2_\(UUID().uuidString)")
            let dir4 = base.appendingPathComponent("m4"), dir2 = base.appendingPathComponent("m2")
            try? fm.createDirectory(at: dir4, withIntermediateDirectories: true)
            try? fm.createDirectory(at: dir2, withIntermediateDirectories: true)
            defer { try? fm.removeItem(at: base) }

            let E = 4
            let projs = ExpertSource.projs, parts = ExpertSource.parts   // [gate,up,down] × [weight,scales,biases]
            // distinct constant byte per (dir, projIdx, partIdx, expert)
            func fillByte(_ dir: Int, _ p: Int, _ q: Int, _ e: Int) -> UInt8 {
                UInt8((dir &* 101 &+ (p &* 3 &+ q) &* 37 &+ e &* 7) & 0xFF)
            }
            // minimal safetensors writer: 8B LE header-len + JSON header + data (offsets relative
            // to data start) + model.safetensors.index.json weight_map.
            func writeCkpt(_ dir: URL, _ dirTag: Int, wCols: Int) throws {
                var meta: [String: Any] = [:]
                var blob = Data()
                for (p, proj) in projs.enumerated() {
                    for (q, part) in parts.enumerated() {
                        let isW = (part == "weight")
                        let shape = isW ? [E, 8, wCols] : [E, 8, 2]
                        let itemSize = isW ? 4 : 2                       // U32 vs F16
                        let per = shape.dropFirst().reduce(1, *) * itemSize
                        let begin = blob.count
                        for e in 0 ..< E {
                            blob.append(contentsOf: [UInt8](repeating: fillByte(dirTag, p, q, e), count: per))
                        }
                        meta["\(ExpertSource.prefix).0.mlp.switch_mlp.\(proj).\(part)"] =
                            ["dtype": isW ? "U32" : "F16", "shape": shape, "data_offsets": [begin, blob.count]]
                    }
                }
                let hdr = try JSONSerialization.data(withJSONObject: meta)
                var out = Data(); var hlen = UInt64(hdr.count).littleEndian
                withUnsafeBytes(of: &hlen) { out.append(contentsOf: $0) }
                out.append(hdr); out.append(blob)
                try out.write(to: dir.appendingPathComponent("model.safetensors"))
                var wm: [String: String] = [:]
                for proj in projs {
                    for part in parts { wm["\(ExpertSource.prefix).0.mlp.switch_mlp.\(proj).\(part)"] = "model.safetensors" }
                }
                let idx = try JSONSerialization.data(withJSONObject: ["weight_map": wm])
                try idx.write(to: dir.appendingPathComponent("model.safetensors.index.json"))
            }

            do {
                try writeCkpt(dir4, 0, wCols: 16)   // 4-bit: weight [4,8,16] U32
                try writeCkpt(dir2, 1, wCols: 8)    // 2-bit: weight [4,8,8]  U32
                let src4 = try ExpertSource(modelDir: dir4.path)
                let src2 = try ExpertSource(modelDir: dir2.path)
                let arena = try MixedExpertArena(device: device, source4: src4, source2: src2,
                                                 K4: 2, M2: 2, refLayer: 0)
                arena.loadCore(0, 3, slot: 0)                               // 4-bit expert 3 → core slot 0
                arena.loadTailMany(0, [(e: 1, slot: 2), (e: 2, slot: 3)])   // 2-bit experts 1,2 → global slots 2,3

                guard let bufs = arena.gatherBuffers12(device: device), bufs.count == 12 else {
                    return (false, "gatherBuffers12 nil / count != 12")
                }
                // source-side sliceBytes parity (the arena enforces this as a precondition)
                if try src4.sliceBytes(0, "gate_proj", "scales") != src2.sliceBytes(0, "gate_proj", "scales") {
                    return (false, "src4/src2 scales sliceBytes differ")
                }

                func slotAllEqual(_ buf: MTLBuffer, _ off: Int, _ n: Int, _ want: UInt8) -> Bool {
                    let p = buf.contents().advanced(by: off).bindMemory(to: UInt8.self, capacity: n)
                    for i in 0 ..< n where p[i] != want { return false }
                    return true
                }

                // per proj: buffers [w4, w2, scales, biases] at indices p*4+{0,1,2,3}
                for p in 0 ..< 3 {
                    let proj = projs[p]
                    let w4 = bufs[p * 4 + 0], w2 = bufs[p * 4 + 1], sc = bufs[p * 4 + 2], bi = bufs[p * 4 + 3]
                    let sbW4 = arena.sliceBytes(proj, "w4"), sbW2 = arena.sliceBytes(proj, "w2")
                    let sbS = arena.sliceBytes(proj, "scales"), sbB = arena.sliceBytes(proj, "biases")
                    if sbW4 != 512 || sbW2 != 256 { return (false, "proj\(p) weight sliceBytes w4=\(sbW4) w2=\(sbW2)") }
                    if sbW2 != sbW4 / 2 { return (false, "proj\(p) w2 != w4/2") }
                    if sbS != 32 || sbB != 32 || sbS != sbB { return (false, "proj\(p) s/b sliceBytes \(sbS)/\(sbB)") }
                    // w4 core slot 0 ← src4 expert 3 (partIdx 0)
                    if !slotAllEqual(w4, 0, sbW4, fillByte(0, p, 0, 3)) { return (false, "w4 slot0 proj\(p)") }
                    // w2 local 0/1 ← src2 experts 1/2
                    if !slotAllEqual(w2, 0 * sbW2, sbW2, fillByte(1, p, 0, 1)) { return (false, "w2 local0 proj\(p)") }
                    if !slotAllEqual(w2, 1 * sbW2, sbW2, fillByte(1, p, 0, 2)) { return (false, "w2 local1 proj\(p)") }
                    // uniform scales (partIdx 1): global slot0 ← src4 e3; slots 2,3 ← src2 e1,e2
                    if !slotAllEqual(sc, 0 * sbS, sbS, fillByte(0, p, 1, 3)) { return (false, "scales slot0 proj\(p)") }
                    if !slotAllEqual(sc, 2 * sbS, sbS, fillByte(1, p, 1, 1)) { return (false, "scales slot2 proj\(p)") }
                    if !slotAllEqual(sc, 3 * sbS, sbS, fillByte(1, p, 1, 2)) { return (false, "scales slot3 proj\(p)") }
                    // uniform biases (partIdx 2)
                    if !slotAllEqual(bi, 0 * sbB, sbB, fillByte(0, p, 2, 3)) { return (false, "biases slot0 proj\(p)") }
                    if !slotAllEqual(bi, 2 * sbB, sbB, fillByte(1, p, 2, 1)) { return (false, "biases slot2 proj\(p)") }
                    if !slotAllEqual(bi, 3 * sbB, sbB, fillByte(1, p, 2, 2)) { return (false, "biases slot3 proj\(p)") }
                }
                return (true, "ok")
            } catch {
                return (false, "arena setup threw: \(error)")
            }
        }

        // Test 85b (Phase A(a)): MixedExpertArena with K4=0 (all-2-bit tail, cov-115 design
        // point). Regression for the SIGTRAP found in the W4c battery: MLXArray.zeros([0]+shape)
        // → asMTLBuffer force-unwraps a nil data ptr inside mlx-swift. Init must succeed (dummy
        // 1-row w4, never addressed) and tail preads must land at GLOBAL slot = local index.
        run("mixed_arena_k4_zero") {
            guard let (device, _) = SeedlessMetalForward.ensure() else { return (false, "no device") }
            let fm = FileManager.default
            let base = fm.temporaryDirectory.appendingPathComponent("qwisp_k40_\(UUID().uuidString)")
            let dir4 = base.appendingPathComponent("m4"), dir2 = base.appendingPathComponent("m2")
            try? fm.createDirectory(at: dir4, withIntermediateDirectories: true)
            try? fm.createDirectory(at: dir2, withIntermediateDirectories: true)
            defer { try? fm.removeItem(at: base) }

            let E = 4
            let projs = ExpertSource.projs, parts = ExpertSource.parts
            func fillByte(_ dir: Int, _ p: Int, _ q: Int, _ e: Int) -> UInt8 {
                UInt8((dir &* 101 &+ (p &* 3 &+ q) &* 37 &+ e &* 7) & 0xFF)
            }
            func writeCkpt(_ dir: URL, _ dirTag: Int, wCols: Int) throws {
                var meta: [String: Any] = [:]
                var blob = Data()
                for (p, proj) in projs.enumerated() {
                    for (q, part) in parts.enumerated() {
                        let isW = (part == "weight")
                        let shape = isW ? [E, 8, wCols] : [E, 8, 2]
                        let per = shape.dropFirst().reduce(1, *) * (isW ? 4 : 2)
                        let begin = blob.count
                        for e in 0 ..< E {
                            blob.append(contentsOf: [UInt8](repeating: fillByte(dirTag, p, q, e), count: per))
                        }
                        meta["\(ExpertSource.prefix).0.mlp.switch_mlp.\(proj).\(part)"] =
                            ["dtype": isW ? "U32" : "F16", "shape": shape, "data_offsets": [begin, blob.count]]
                    }
                }
                let hdr = try JSONSerialization.data(withJSONObject: meta)
                var out = Data(); var hlen = UInt64(hdr.count).littleEndian
                withUnsafeBytes(of: &hlen) { out.append(contentsOf: $0) }
                out.append(hdr); out.append(blob)
                try out.write(to: dir.appendingPathComponent("model.safetensors"))
                var wm: [String: String] = [:]
                for proj in projs {
                    for part in parts { wm["\(ExpertSource.prefix).0.mlp.switch_mlp.\(proj).\(part)"] = "model.safetensors" }
                }
                try JSONSerialization.data(withJSONObject: ["weight_map": wm])
                    .write(to: dir.appendingPathComponent("model.safetensors.index.json"))
            }

            do {
                try writeCkpt(dir4, 0, wCols: 16)
                try writeCkpt(dir2, 1, wCols: 8)
                let src4 = try ExpertSource(modelDir: dir4.path)
                let src2 = try ExpertSource(modelDir: dir2.path)
                let arena = try MixedExpertArena(device: device, source4: src4, source2: src2,
                                                 K4: 0, M2: 3, refLayer: 0)   // ← K4=0: crashed pre-fix
                arena.loadTailMany(0, [(e: 1, slot: 0), (e: 3, slot: 2)])     // global slot = local idx
                guard let bufs = arena.gatherBuffers12(device: device), bufs.count == 12 else {
                    return (false, "gatherBuffers12 nil / count != 12")
                }
                func slotAllEqual(_ buf: MTLBuffer, _ off: Int, _ n: Int, _ want: UInt8) -> Bool {
                    let p = buf.contents().advanced(by: off).bindMemory(to: UInt8.self, capacity: n)
                    for i in 0 ..< n where p[i] != want { return false }
                    return true
                }
                for p in 0 ..< 3 {
                    let proj = projs[p]
                    let w2 = bufs[p * 4 + 1], sc = bufs[p * 4 + 2], bi = bufs[p * 4 + 3]
                    let sbW2 = arena.sliceBytes(proj, "w2")
                    let sbS = arena.sliceBytes(proj, "scales"), sbB = arena.sliceBytes(proj, "biases")
                    if !slotAllEqual(w2, 0 * sbW2, sbW2, fillByte(1, p, 0, 1)) { return (false, "w2 local0 proj\(p)") }
                    if !slotAllEqual(w2, 2 * sbW2, sbW2, fillByte(1, p, 0, 3)) { return (false, "w2 local2 proj\(p)") }
                    if !slotAllEqual(sc, 0 * sbS, sbS, fillByte(1, p, 1, 1)) { return (false, "scales slot0 proj\(p)") }
                    if !slotAllEqual(sc, 2 * sbS, sbS, fillByte(1, p, 1, 3)) { return (false, "scales slot2 proj\(p)") }
                    if !slotAllEqual(bi, 0 * sbB, sbB, fillByte(1, p, 2, 1)) { return (false, "biases slot0 proj\(p)") }
                    if !slotAllEqual(bi, 2 * sbB, sbB, fillByte(1, p, 2, 3)) { return (false, "biases slot2 proj\(p)") }
                }
                return (true, "ok")
            } catch {
                return (false, "K4=0 arena threw: \(error)")
            }
        }

        // ── W3a (notes/18): mixed-precision encode path + (K4,M2) byte-budget rule ─────
        // Test 86 (W3a-1): the NEW encode-only statics encodeGqmmMixRows/encodeGqmmMixSwigluRows
        // reproduce the shipped one-shot gqmmMixRows/gqmmMixSwigluRows bit-for-bit when driven in a
        // self-contained CB (device buffers built here, same mtlBuf conversions the wrappers use).
        // This locks the encode-only bind order/indices to the production kernels. RED: the stubs
        // encode nothing (ensureMixPipelines()==false / zero out) so bitEqual fails.
        run("mix_encode_parity") {
            let E = 64, K4 = 16, K = 2048, N = 512, Ktop = 4
            guard let (device, queue) = SeedlessMetalForward.ensure() else { return (false, "no device") }

            // Split fixture: 4-bit core (0..<K4) + 2-bit tail (K4..<E); global-slot-indexed sc/bi.
            func mixWeights(_ seed: MLXArray) -> (MLXArray, MLXArray, MLXArray, MLXArray) {
                let (q4, s4, b4o) = MLX.quantized(seed[0 ..< K4], groupSize: 64, bits: 4, mode: .affine)
                let (q2, s2, b2o) = MLX.quantized(seed[K4 ..< E], groupSize: 64, bits: 2, mode: .affine)
                let sc = MLX.concatenated([s4, s2], axis: 0).asType(.float16)
                let bi = MLX.concatenated([b4o!, b2o!], axis: 0).asType(.float16)
                MLX.eval([q4, q2, sc, bi])
                return (q4, q2, sc, bi)
            }
            // Readback a shared-storage F16 out buffer as [rows, N].
            func readback(_ buf: MTLBuffer, _ rows: Int) -> MLXArray {
                let ptr = buf.contents().bindMemory(to: Float16.self, capacity: rows * N)
                return MLXArray(Array(UnsafeBufferPointer(start: ptr, count: rows * N)), [rows, N])
            }

            // ── gqmmMixRows encode parity (M in {1,9}, incl. one lhsPer:true) ──
            let wf = MLXRandom.normal([E, N, K]).asType(.float16)
            let (w4, w2, scales, biases) = mixWeights(wf)
            func rowsCase(_ M: Int, lhsPer: Bool) -> (Bool, String) {
                let rows = lhsPer ? M * Ktop : M
                let x = MLXRandom.normal([rows, K]).asType(.float16); x.eval()
                let indsFlat = (0 ..< M * Ktop).map { mixPool[$0 % mixPool.count] }
                let inds = MLXArray(indsFlat, [M * Ktop]); inds.eval()
                // Reference: shipped one-shot kernel (also compiles the mix pipeline lazily).
                guard let ref = SeedlessMetalForward.gqmmMixRows(x, w4: w4, w2: w2,
                                    scales: scales, biases: biases, inds: inds, K4: K4,
                                    M: M, Ktop: Ktop, K: K, N: N, lhsPerExpert: lhsPer)
                else { return (false, "ref nil M=\(M) lhsPer=\(lhsPer)") }
                ref.eval()
                // Encode-only path: pipelines must be ready without a dispatch of their own.
                guard SeedlessMetalForward.ensureMixPipelines()
                else { return (false, "ensureMixPipelines false (M=\(M) lhsPer=\(lhsPer))") }
                guard let bx = SeedlessMetalForward.mtlBuf(x.asType(.float16), device),
                      let bw4 = SeedlessMetalForward.mtlBuf(w4, device),
                      let bw2 = SeedlessMetalForward.mtlBuf(w2, device),
                      let bsc = SeedlessMetalForward.mtlBuf(scales.asType(.float16), device),
                      let bbi = SeedlessMetalForward.mtlBuf(biases.asType(.float16), device),
                      let bin = SeedlessMetalForward.mtlBuf(inds.asType(.int32), device)
                else { return (false, "rows buf nil (M=\(M))") }
                let outBuf = device.makeBuffer(length: M * Ktop * N * 2, options: .storageModeShared)!
                let cb = queue.makeCommandBuffer()!; let enc = cb.makeComputeCommandEncoder()!
                SeedlessFusedVerify.encodeGqmmMixRows(enc, w4: bw4, w2: bw2, sc: bsc, bi: bbi,
                                    x: bx, inds: bin, out: outBuf,
                                    M: M, Ktop: Ktop, K: K, N: N, lhsPer: lhsPer, k4: K4)
                enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
                let got = readback(outBuf, M * Ktop); got.eval()
                return bitEqual(got, ref)
            }
            for (M, lp) in [(1, false), (9, false), (9, true)] {
                let (ok, d) = rowsCase(M, lhsPer: lp)
                if !ok { return (false, "rows M=\(M) lhsPer=\(lp): \(d)") }
            }

            // ── gqmmMixSwigluRows encode parity (M in {1,9}) ──
            let gf = MLXRandom.normal([E, N, K]).asType(.float16)
            let uf = MLXRandom.normal([E, N, K]).asType(.float16)
            let (gw4, gw2, gsc, gbi) = mixWeights(gf)
            let (uw4, uw2, usc, ubi) = mixWeights(uf)
            func swigluCase(_ M: Int) -> (Bool, String) {
                let x = MLXRandom.normal([M, K]).asType(.float16); x.eval()
                let indsFlat = (0 ..< M * Ktop).map { mixPool[$0 % mixPool.count] }
                let inds = MLXArray(indsFlat, [M * Ktop]); inds.eval()
                guard let ref = SeedlessMetalForward.gqmmMixSwigluRows(x,
                                    gw4: gw4, gw2: gw2, gsc: gsc, gbi: gbi,
                                    uw4: uw4, uw2: uw2, usc: usc, ubi: ubi, inds: inds,
                                    K4: K4, M: M, Ktop: Ktop, K: K, N: N)
                else { return (false, "ref swiglu nil M=\(M)") }
                ref.eval()
                guard SeedlessMetalForward.ensureMixPipelines()
                else { return (false, "ensureMixPipelines false swiglu (M=\(M))") }
                guard let bx = SeedlessMetalForward.mtlBuf(x.asType(.float16), device),
                      let bgw4 = SeedlessMetalForward.mtlBuf(gw4, device),
                      let bgw2 = SeedlessMetalForward.mtlBuf(gw2, device),
                      let bgsc = SeedlessMetalForward.mtlBuf(gsc.asType(.float16), device),
                      let bgbi = SeedlessMetalForward.mtlBuf(gbi.asType(.float16), device),
                      let buw4 = SeedlessMetalForward.mtlBuf(uw4, device),
                      let buw2 = SeedlessMetalForward.mtlBuf(uw2, device),
                      let busc = SeedlessMetalForward.mtlBuf(usc.asType(.float16), device),
                      let bubi = SeedlessMetalForward.mtlBuf(ubi.asType(.float16), device),
                      let bin = SeedlessMetalForward.mtlBuf(inds.asType(.int32), device)
                else { return (false, "swiglu buf nil (M=\(M))") }
                let hBuf = device.makeBuffer(length: M * Ktop * N * 2, options: .storageModeShared)!
                let cb = queue.makeCommandBuffer()!; let enc = cb.makeComputeCommandEncoder()!
                SeedlessFusedVerify.encodeGqmmMixSwigluRows(enc,
                                    gw4: bgw4, gw2: bgw2, gsc: bgsc, gbi: bgbi,
                                    uw4: buw4, uw2: buw2, usc: busc, ubi: bubi,
                                    x: bx, inds: bin, h: hBuf,
                                    M: M, Ktop: Ktop, K: K, N: N, k4: K4)
                enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
                let got = readback(hBuf, M * Ktop); got.eval()
                return bitEqual(got, ref)
            }
            for M in [1, 9] {
                let (ok, d) = swigluCase(M)
                if !ok { return (false, "swiglu M=\(M): \(d)") }
            }
            return (true, "ok")
        }

        // Test 87 (W3a-2): pure-CPU (K4,M2) byte-budget rule. slot4=1728 KiB, slot2=960 KiB,
        // budget = 64·slot4. Hand-calc: m2 = floor((budget − k4·slot4)/slot2), k4 clamped to
        // [0, budget/slot4], m2 never negative. RED: stub returns (k4, -1).
        run("mixed_km_budget") {
            let slot4 = 1728 * 1024, slot2 = 960 * 1024, budget = 64 * slot4
            func km(_ k4: Int) -> (Int, Int) {
                let r = DeviceCalibration.mixedKM(budgetBytes: budget, slot4Bytes: slot4, slot2Bytes: slot2, k4: k4)
                return (r.k4, r.m2)
            }
            // Design points (notes/18 W1 Part-A): K4 8/20/32/0 → M2 100/79/57/115.
            if km(8)  != (8, 100)  { return (false, "k4=8 → \(km(8)) != (8,100)") }
            if km(20) != (20, 79)  { return (false, "k4=20 → \(km(20)) != (20,79)") }
            if km(32) != (32, 57)  { return (false, "k4=32 → \(km(32)) != (32,57)") }
            if km(0)  != (0, 115)  { return (false, "k4=0 → \(km(0)) != (0,115)") }
            // Adversarial: budget < one slot4, k4=1 must clamp k4 to 0 and never go negative m2.
            let small = DeviceCalibration.mixedKM(budgetBytes: slot4 / 2, slot4Bytes: slot4, slot2Bytes: slot2, k4: 1)
            if small.k4 != 0 { return (false, "tiny budget k4=1 did not clamp to 0: \(small)") }
            if small.m2 < 0  { return (false, "m2 negative: \(small)") }
            // k4 far beyond budget clamps to budget/slot4=64, leaving m2=0.
            let over = DeviceCalibration.mixedKM(budgetBytes: budget, slot4Bytes: slot4, slot2Bytes: slot2, k4: 999)
            if over.k4 != 64 || over.m2 != 0 { return (false, "k4=999 → \(over) != (64,0)") }
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
        guard let (device, queue) = SeedlessMetalForward.ensure() else { return "no device" }
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
            _ = SeedlessMetalForward.qmm(xAll[0 ..< 1], wq, scales: sc, biases: bi, M: 1, K: K, N: N)
            guard let bx = xAll.asType(.float16).asMTLBuffer(device: device, noCopy: false),
                  let bwq = wq.asMTLBuffer(device: device, noCopy: false),
                  let bsc = sc.asType(.float16).asMTLBuffer(device: device, noCopy: false),
                  let bbi = bi.asType(.float16).asMTLBuffer(device: device, noCopy: false) else { return "buf nil" }
            let outBuf = device.makeBuffer(length: 25 * N * 2, options: .storageModeShared)!
            var t1 = 0.0
            var row = "[raw-perf] qmm4 N=\(N) K=\(K):"
            for M in [1, 5, 9, 13, 17, 25] {
                let ms = gpuMs(50) { enc in
                    enc.setComputePipelineState(SeedlessMetalForward._qmmPipeline!)
                    enc.setBuffer(bwq, offset: 0, index: 0); enc.setBuffer(bsc, offset: 0, index: 1)
                    enc.setBuffer(bbi, offset: 0, index: 2); enc.setBuffer(bx, offset: 0, index: 3)
                    enc.setBuffer(outBuf, offset: 0, index: 4)
                    var kk = Int32(K), nn = Int32(N)
                    enc.setBytes(&kk, length: 4, index: 5); enc.setBytes(&nn, length: 4, index: 6)
                    SeedlessMetalForward.bindStop(enc, 16)
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
            _ = SeedlessMetalForward.gatherQmmRows(xAll[0 ..< 1], wq, scales: sc, biases: bi,
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
                    enc.setComputePipelineState(SeedlessMetalForward._gqmmRowsPipeline!)
                    enc.setBuffer(bwq, offset: 0, index: 0); enc.setBuffer(bsc, offset: 0, index: 1)
                    enc.setBuffer(bbi, offset: 0, index: 2); enc.setBuffer(bx, offset: 0, index: 3)
                    enc.setBuffer(bin, offset: 0, index: 4); enc.setBuffer(outBuf, offset: 0, index: 5)
                    var kk = Int32(K), nn = Int32(N), kt = Int32(Ktop)
                    enc.setBytes(&kk, length: 4, index: 6); enc.setBytes(&nn, length: 4, index: 7)
                    enc.setBytes(&kt, length: 4, index: 8)
                    SeedlessMetalForward.bindStop(enc, 9)
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
