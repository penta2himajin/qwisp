import Foundation
import Metal
import MLX

// Lane-batched greedy decode — Stage 1 of the parallel-agent speedup.
//
// B concurrent sequences ("lanes"), each with its OWN caches (KV arena + GDN
// state) held by a per-lane SeedlessFusedForward, advance ONE token each through
// a SINGLE fused pass so the dense/MoE weight reads amortize across lanes (the
// M>1 prize) while the sequence-coupled kernels (attention SDPA/KV, GDN
// conv/recurrence) run per-lane at literal M=1 against that lane's caches.
//
// Fast path (fusion flags ON, the default): every row-independent stage runs at
// M=B on the DRIVER scratch — in-proj demux, GDN prep, norm/gate, attn sigmoid
// gate, out/o-proj — and only the sequence-coupled kernels are dispatched per
// lane, bound at a BYTE OFFSET into the driver scratch so lane b's core reads
// and writes driver row b directly (no staging copies at all). Per lane per
// layer that leaves 2 dispatches (GDN: conv shift + recurrence) or 4 (attn:
// q-prep + k-prep + v-append + SDPA); everything else is one M=B dispatch per
// layer regardless of B. Big projection weights are read ONCE per layer.
//
// Bit-exactness by construction (locked test `lane_batch_bitexact`):
//   - per-lane kernels: same pipeline, same constants, same cache as solo M=1 —
//     the offset binding only relocates the row storage ⇒ identical bits.
//   - M=B stages: per-row grids whose row math is M-invariant — the engine's
//     core self-consistency guarantee (M-row kernel suite, RAWTESTS) that
//     already underwrites strict verify ≡ M=1 decode.
//
// Fallback path (any fusion flag OFF / pipelines absent): the Stage-1 staged
// form — projections at M=B, mixer CORE per lane via the hybridDense seam of
// encodeGdnLayerRows/encodeAttnLayerRows, rows staged via lane_row_copy.
//
// Additive: reuses the frozen encode statics + pipelines + Layer weight structs;
// no frozen file is modified. Resident tier only (streaming's chunked expert IO
// is a different scheduling problem — out of Stage 1 scope).
public final class SeedlessLaneBatch {
    let driver: SeedlessFusedVerify.SeedlessFusedForward     // weights + M=B scratch
    let lanes: [SeedlessFusedVerify.SeedlessFusedForward]    // per-lane caches + scratch
    public let B: Int
    // Per-lane staging for the FALLBACK path only (one [1, H] pair PER lane: a
    // shared pair would chain every lane through one buffer and Metal's
    // buffer-granular hazard tracking would serialize the lane cores).
    let laneXs: [MTLBuffer]
    let laneOuts: [MTLBuffer]

    nonisolated(unsafe) static var copyPipeline: MTLComputePipelineState? = nil
    static func compileCopy(_ device: MTLDevice) -> Bool {
        if copyPipeline != nil { return true }
        let src = """
        #include <metal_stdlib>
        using namespace metal;
        kernel void lane_row_copy(device const half* src [[buffer(0)]],
                                  device half* dst [[buffer(1)]],
                                  constant uint& srcOff [[buffer(2)]],
                                  constant uint& dstOff [[buffer(3)]],
                                  uint i [[thread_position_in_grid]]) {
            dst[dstOff + i] = src[srcOff + i];
        }
        """
        guard let lib = try? device.makeLibrary(source: src, options: nil),
              let fn = lib.makeFunction(name: "lane_row_copy"),
              let ps = try? device.makeComputePipelineState(function: fn) else { return false }
        copyPipeline = ps
        return true
    }

    // ── qmm4_rows_b: B-row qmv (Stage 1b projection lever — measured NO-GO) ───
    // Same grid/simdgroup structure and PER-ROW arithmetic order as the shipped
    // qmm4 (qmv) kernel — the ONLY change is an inner b-loop so the device weight
    // reads are shared across up to 4 x-rows instead of re-read per row.
    // Per-(row,b) accumulation = identical fma sequence to qmm4 with that x row
    // ⇒ bit-exact by construction (byte-compared PASS in lane-kernel-bench).
    // VERDICT (2026-07-21, lane-kernel-bench @[N=8192,K=2048]): 4-6x SLOWER than
    // qmm4_rows at every B (B=1 107µs vs 26; B=3 402 vs 65) — the x_thread[4][16]
    // register file collapses occupancy and costs far more than the ~0.7x/row
    // weight re-read it saves. NOT wired into any forward path; kept only as the
    // measured evidence + harness for the Stage-1b projection-lever NO-GO. Do not
    // re-propose weight-read amortization for qmv decode shapes without a design
    // that adds rows WITHOUT growing per-thread register state.
    nonisolated(unsafe) static var qmmBPipeline: MTLComputePipelineState? = nil
    static func compileQmmB(_ device: MTLDevice) -> Bool {
        if qmmBPipeline != nil { return true }
        let src = """
        #include <metal_stdlib>
        using namespace metal;
        #define SIMD_SIZE 32
        inline float ld16(const device half* x, thread float* xt) {
            float sum = 0.0f;
            for (int i = 0; i < 16; i += 4) {
                sum += x[i] + x[i+1] + x[i+2] + x[i+3];
                xt[i]   = x[i];
                xt[i+1] = x[i+1] / 16.0f;
                xt[i+2] = x[i+2] / 256.0f;
                xt[i+3] = x[i+3] / 4096.0f;
            }
            return sum;
        }
        inline float qd4(const device uint8_t* w, const thread float* xt, float scale, float bias, float sum) {
            float accum = 0.0f;
            const device uint16_t* ws = (const device uint16_t*)w;
            for (int i = 0; i < 4; i++) {
                accum += (xt[4*i]   * (float)(ws[i] & 0x000f) +
                          xt[4*i+1] * (float)(ws[i] & 0x00f0) +
                          xt[4*i+2] * (float)(ws[i] & 0x0f00) +
                          xt[4*i+3] * (float)(ws[i] & 0xf000));
            }
            return scale * accum + sum * bias;
        }
        kernel void qmm4_rows_b(device const uint32_t* w      [[buffer(0)]],
                                device const half*     scales [[buffer(1)]],
                                device const half*     biases [[buffer(2)]],
                                device const half*     x      [[buffer(3)]],
                                device half*           y      [[buffer(4)]],
                                constant int&          in_vec_size  [[buffer(5)]],   // K
                                constant int&          out_vec_size [[buffer(6)]],   // N
                                constant int&          nrows        [[buffer(7)]],   // B ≤ 4
                                uint3 tid      [[threadgroup_position_in_grid]],
                                uint  simd_gid [[simdgroup_index_in_threadgroup]],
                                uint  simd_lid [[thread_index_in_simdgroup]]) {
            constexpr int packs_per_thread = 2;
            constexpr int num_simdgroups = 2;
            constexpr int results_per_simdgroup = 4;
            constexpr int pack_factor = 8;
            constexpr int bytes_per_pack = 4;
            constexpr int values_per_thread = 16;
            constexpr int block_size = 512;
            constexpr int scale_step_per_thread = 4;
            const device uint8_t* ws = (const device uint8_t*)w;
            typedef float U;
            thread U x_thread[4][16];
            thread U sums[4];
            thread U result[4][4] = {{0}};        // [out-row][b]
            const int B = nrows;
            const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
            const int in_vec_size_g = in_vec_size / 64;
            const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) + simd_gid * results_per_simdgroup;
            ws     += out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
            scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
            biases += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
            y += out_row;
            int xoff = simd_lid * values_per_thread;
            for (int k = 0; k < in_vec_size; k += block_size) {
                for (int b = 0; b < B; b++)
                    sums[b] = ld16(x + b * in_vec_size + xoff, x_thread[b]);
                for (int row = 0; row < results_per_simdgroup; row++) {
                    auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
                    const device half* sl = scales + row * in_vec_size_g;
                    const device half* bl = biases + row * in_vec_size_g;
                    U s = sl[0]; U bi = bl[0];
                    for (int b = 0; b < B; b++)
                        result[row][b] += qd4(wl, x_thread[b], s, bi, sums[b]);
                }
                ws += block_size * bytes_per_pack / pack_factor;
                scales += block_size / 64;
                biases += block_size / 64;
                xoff += block_size;
            }
            for (int row = 0; row < results_per_simdgroup; row++) {
                for (int b = 0; b < B; b++) {
                    result[row][b] = simd_sum(result[row][b]);
                    if (simd_lid == 0) y[b * out_vec_size + row] = (half)result[row][b];
                }
            }
        }
        """
        // Math opts MUST mirror the shipped qmm4 compile (QWISP_QMM_MATH, default
        // safe) — bit-exactness of qd4 depends on identical FMA-contraction setting.
        let opts = MTLCompileOptions()
        let mathSel = ProcessInfo.processInfo.environment["QWISP_QMM_MATH"] ?? "safe"
        if #available(macOS 15.0, *) {
            switch mathSel {
            case "fast": opts.mathMode = .fast
            case "relaxed": opts.mathMode = .relaxed
            default: opts.mathMode = .safe
            }
        } else {
            opts.fastMathEnabled = (mathSel == "fast")
        }
        guard let lib = try? device.makeLibrary(source: src, options: opts),
              let fn = lib.makeFunction(name: "qmm4_rows_b"),
              let ps = try? device.makeComputePipelineState(function: fn) else { return false }
        qmmBPipeline = ps
        return true
    }

    /// Encode qmm4_rows_b: x[B,K] · W[N,K] → y[B,N], B ≤ 4 per dispatch (callers
    /// split larger B into row groups). x/y offsets are BYTE offsets (row groups).
    static func encodeQmmRowsB(_ enc: MTLComputeCommandEncoder,
                               w: MTLBuffer, scales: MTLBuffer, biases: MTLBuffer,
                               x: MTLBuffer, xOff: Int, out: MTLBuffer, outOff: Int,
                               B: Int, K: Int, N: Int) {
        let p = SeedlessLaneBatch.qmmBPipeline!
        enc.setComputePipelineState(p)
        enc.setBuffer(w, offset: 0, index: 0)
        enc.setBuffer(scales, offset: 0, index: 1)
        enc.setBuffer(biases, offset: 0, index: 2)
        enc.setBuffer(x, offset: xOff, index: 3)
        enc.setBuffer(out, offset: outOff, index: 4)
        var kk = Int32(K), nn = Int32(N), bb = Int32(B)
        enc.setBytes(&kk, length: 4, index: 5)
        enc.setBytes(&nn, length: 4, index: 6)
        enc.setBytes(&bb, length: 4, index: 7)
        enc.dispatchThreadgroups(MTLSize(width: 1, height: N / 8, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 32, height: 2, depth: 1))
    }

    static func encodeRowCopyStatic(_ enc: MTLComputeCommandEncoder,
                                    src: MTLBuffer, srcOff: Int, dst: MTLBuffer, dstOff: Int, count: Int) {
        let ps = SeedlessLaneBatch.copyPipeline!
        enc.setComputePipelineState(ps)
        enc.setBuffer(src, offset: 0, index: 0)
        enc.setBuffer(dst, offset: 0, index: 1)
        var so = UInt32(srcOff), do_ = UInt32(dstOff)
        enc.setBytes(&so, length: 4, index: 2)
        enc.setBytes(&do_, length: 4, index: 3)
        enc.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: min(ps.maxTotalThreadsPerThreadgroup, 256), height: 1, depth: 1))
    }

    private func encodeRowCopy(_ enc: MTLComputeCommandEncoder,
                               src: MTLBuffer, srcOff: Int, dst: MTLBuffer, dstOff: Int, count: Int) {
        SeedlessLaneBatch.encodeRowCopyStatic(enc, src: src, srcOff: srcOff, dst: dst, dstOff: dstOff, count: count)
    }

    /// driver: weights + scratch sized maxM ≥ lanes.count. lanes: one forward per
    /// sequence (its caches are the lane state; its scratch serves the fallback
    /// M=1 mixer). All must be resident-mode with the same layer stack.
    public init?(driver: SeedlessFusedVerify.SeedlessFusedForward,
                 lanes: [SeedlessFusedVerify.SeedlessFusedForward]) {
        guard !lanes.isEmpty, driver.maxM >= lanes.count,
              lanes.allSatisfy({ $0.layers.count == driver.layers.count }),
              SeedlessLaneBatch.compileCopy(driver.device) else { return nil }
        self.driver = driver
        self.lanes = lanes
        self.B = lanes.count
        var lxs: [MTLBuffer] = [], los: [MTLBuffer] = []
        for _ in lanes {
            guard let lx = driver.device.makeBuffer(length: driver.H * 2, options: .storageModeShared),
                  let lo = driver.device.makeBuffer(length: driver.H * 2, options: .storageModeShared) else { return nil }
            lxs.append(lx); los.append(lo)
        }
        self.laneXs = lxs
        self.laneOuts = los
    }

    // ── Offset-bound wrappers over the frozen per-lane pipelines ──────────────
    // Same pipeline, same constants, same grid as the SeedlessFusedVerify encode
    // statics; the ONLY difference is a byte offset on the row buffers so lane
    // b's sequence-coupled kernel reads/writes driver scratch row b in place.

    private func laneConvShift(_ enc: MTLComputeCommandEncoder, hist: MTLBuffer,
                               qkv: MTLBuffer, qkvOff: Int, w: MTLBuffer,
                               convOut: MTLBuffer, convOff: Int, histOut: MTLBuffer,
                               K: Int, C: Int) {
        let p = SeedlessFusedVerify._convShiftFusedRowsPipeline!
        enc.setComputePipelineState(p)
        enc.setBuffer(hist, offset: 0, index: 0); enc.setBuffer(qkv, offset: qkvOff, index: 1)
        enc.setBuffer(w, offset: 0, index: 2); enc.setBuffer(convOut, offset: convOff, index: 3)
        enc.setBuffer(histOut, offset: 0, index: 4)
        var kk = UInt32(K), cc = UInt32(C), mm = UInt32(1)
        enc.setBytes(&kk, length: 4, index: 5); enc.setBytes(&cc, length: 4, index: 6); enc.setBytes(&mm, length: 4, index: 7)
        enc.dispatchThreads(MTLSize(width: C, height: 2, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: min(p.maxTotalThreadsPerThreadgroup, 256), height: 1, depth: 1))
    }

    private func laneDeltaStep(_ enc: MTLComputeCommandEncoder,
                               q: MTLBuffer, qOff: Int, k: MTLBuffer, kOff: Int, v: MTLBuffer, vOff: Int,
                               g: MTLBuffer, gOff: Int, beta: MTLBuffer, betaOff: Int,
                               stateIn: MTLBuffer, stateOut: MTLBuffer, y: MTLBuffer, yOff: Int,
                               Hv: Int, Dv: Int) {
        let p = SeedlessMetalForward._recurPipeline!
        enc.setComputePipelineState(p)
        enc.setBuffer(q, offset: qOff, index: 0); enc.setBuffer(k, offset: kOff, index: 1)
        enc.setBuffer(v, offset: vOff, index: 2)
        enc.setBuffer(g, offset: gOff, index: 3); enc.setBuffer(beta, offset: betaOff, index: 4)
        enc.setBuffer(stateIn, offset: 0, index: 5)
        var tt = Int32(1); enc.setBytes(&tt, length: 4, index: 6)
        enc.setBuffer(y, offset: yOff, index: 7); enc.setBuffer(stateOut, offset: 0, index: 8)
        SeedlessMetalForward.bindStop(enc, 16)
        enc.dispatchThreads(MTLSize(width: 32, height: Dv, depth: Hv),
                            threadsPerThreadgroup: MTLSize(width: 32, height: 4, depth: 1))
    }

    private func laneQPrep(_ enc: MTLComputeCommandEncoder, qOut: MTLBuffer, qOutOff: Int,
                           qNorm: MTLBuffer, qRot: MTLBuffer, qRotOff: Int,
                           qd2: Int, headDim: Int, ropeDim: Int, base: Float,
                           startOffset: Int, numHeads: Int, eps: Float) {
        let p = SeedlessFusedVerify._attnQPrepRowsPipeline!
        enc.setComputePipelineState(p)
        enc.setBuffer(qOut, offset: qOutOff, index: 0)
        enc.setBuffer(qNorm, offset: 0, index: 1)
        enc.setBuffer(qRot, offset: qRotOff, index: 2)
        var qd2v = UInt32(qd2), hd = UInt32(headDim), rd = UInt32(ropeDim)
        var bs = base, so = UInt32(startOffset), nh = UInt32(numHeads), ee = eps
        enc.setBytes(&qd2v, length: 4, index: 3); enc.setBytes(&hd, length: 4, index: 4)
        enc.setBytes(&rd, length: 4, index: 5); enc.setBytes(&bs, length: 4, index: 6)
        enc.setBytes(&so, length: 4, index: 7); enc.setBytes(&nh, length: 4, index: 8)
        enc.setBytes(&ee, length: 4, index: 9)
        SeedlessMetalForward.bindStop(enc, 16)
        let tgSize = (((headDim + 3) / 4 + 31) / 32) * 32
        enc.dispatchThreadgroups(MTLSize(width: numHeads, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
    }

    private func laneKPrep(_ enc: MTLComputeCommandEncoder, kOut: MTLBuffer, kOutOff: Int,
                           kNorm: MTLBuffer, kRot: MTLBuffer, kRotOff: Int, kCache: MTLBuffer,
                           headDim: Int, ropeDim: Int, base: Float,
                           startOffset: Int, numKV: Int, maxLen: Int, eps: Float) {
        let p = SeedlessFusedVerify._attnKPrepRowsPipeline!
        enc.setComputePipelineState(p)
        enc.setBuffer(kOut, offset: kOutOff, index: 0)
        enc.setBuffer(kNorm, offset: 0, index: 1)
        enc.setBuffer(kRot, offset: kRotOff, index: 2)
        enc.setBuffer(kCache, offset: 0, index: 3)
        var hd = UInt32(headDim), rd = UInt32(ropeDim), bs = base
        var so = UInt32(startOffset), nkv = UInt32(numKV), ml = UInt32(maxLen), ee = eps
        enc.setBytes(&hd, length: 4, index: 4); enc.setBytes(&rd, length: 4, index: 5)
        enc.setBytes(&bs, length: 4, index: 6); enc.setBytes(&so, length: 4, index: 7)
        enc.setBytes(&nkv, length: 4, index: 8); enc.setBytes(&ml, length: 4, index: 9)
        enc.setBytes(&ee, length: 4, index: 10)
        SeedlessMetalForward.bindStop(enc, 16)
        let tgSize = (((headDim + 3) / 4 + 31) / 32) * 32
        enc.dispatchThreadgroups(MTLSize(width: numKV, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
    }

    private func laneWriteKV(_ enc: MTLComputeCommandEncoder, src: MTLBuffer, srcOff: Int,
                             cache: MTLBuffer, KV: Int, D: Int, maxLen: Int, pos: Int) {
        let p = SeedlessFusedVerify._writeKVRowsPipeline!
        enc.setComputePipelineState(p)
        enc.setBuffer(src, offset: srcOff, index: 0); enc.setBuffer(cache, offset: 0, index: 1)
        var kv = UInt32(KV), dd = UInt32(D), ml = UInt32(maxLen), pp = UInt32(pos), t = UInt32(KV * D)
        enc.setBytes(&kv, length: 4, index: 2); enc.setBytes(&dd, length: 4, index: 3)
        enc.setBytes(&ml, length: 4, index: 4); enc.setBytes(&pp, length: 4, index: 5); enc.setBytes(&t, length: 4, index: 6)
        enc.dispatchThreads(MTLSize(width: KV * D, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: min(p.maxTotalThreadsPerThreadgroup, 256), height: 1, depth: 1))
    }

    private func laneSdpa(_ enc: MTLComputeCommandEncoder, q: MTLBuffer, qOff: Int,
                          k: MTLBuffer, v: MTLBuffer, out: MTLBuffer, outOff: Int,
                          H: Int, KV: Int, D: Int, baseLenPlus1: Int, scale: Float, maxLen: Int) {
        let p = SeedlessMetalForward._sdpaRowsPipeline!
        enc.setComputePipelineState(p)
        enc.setBuffer(q, offset: qOff, index: 0); enc.setBuffer(k, offset: 0, index: 1)
        enc.setBuffer(v, offset: 0, index: 2); enc.setBuffer(out, offset: outOff, index: 3)
        var gqa = Int32(H / KV), bn = Int32(baseLenPlus1)
        var khs = Int32(maxLen * D), kss = Int32(D), vhs = Int32(maxLen * D), vss = Int32(D), sc = scale
        enc.setBytes(&gqa, length: 4, index: 4); enc.setBytes(&bn, length: 4, index: 5)
        enc.setBytes(&khs, length: 4, index: 6); enc.setBytes(&kss, length: 4, index: 7)
        enc.setBytes(&vhs, length: 4, index: 8); enc.setBytes(&vss, length: 4, index: 9)
        enc.setBytes(&sc, length: 4, index: 10)
        enc.dispatchThreadgroups(MTLSize(width: H, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 1024, height: 1, depth: 1))
    }

    // ── Per-layer encoders ────────────────────────────────────────────────────

    /// Fast GDN layer: demux (M=B) → conv shift per lane (offset-bound) → prep
    /// (M=B) → recurrence per lane (offset-bound) → norm/gate (M=B) → out_proj (M=B).
    private func encodeGdnLayerFast(_ enc: MTLComputeCommandEncoder, li: Int, gw: SeedlessFusedVerify.GdnLayerBufs,
                                    keyDim: Int, valueDim: Int, convDim: Int) {
        let H = driver.H, Hv = driver.numVHeads
        SeedlessFusedVerify.encodeQmmInProjDemuxRows(
            enc, w: gw.catInProjW!, scales: gw.catInProjS!, biases: gw.catInProjB!, x: driver.normed,
            outQkv: driver.gdnSc.qkv, outZ: driver.gdnSc.z,
            outB: driver.gdnSc.bP, outA: driver.gdnSc.aP,
            M: B, K: H, dims: (qkv: convDim, z: valueDim, b: Hv, a: Hv))
        for b in 0 ..< B {
            let gc = lanes[b].layers[li].gdnCache!
            laneConvShift(enc, hist: gc.convHist, qkv: driver.gdnSc.qkv, qkvOff: b * convDim * 2,
                          w: gw.conv1dW, convOut: driver.gdnSc.convOut, convOff: b * convDim * 2,
                          histOut: gc.convHistOut, K: driver.convKernel, C: convDim)
        }
        let invScale = Float(pow(Double(driver.headKDim), -0.5))
        SeedlessFusedVerify.encodeGdnPrepRows(enc, convOut: driver.gdnSc.convOut,
                                              aP: driver.gdnSc.aP, bP: driver.gdnSc.bP,
                                              aLog: gw.aLog, dtBias: gw.dtBias,
                                              qn: driver.gdnSc.qn, kn: driver.gdnSc.kn, v: driver.gdnSc.v1,
                                              g: driver.gdnSc.g, beta: driver.gdnSc.beta,
                                              M: B, numKH: driver.numKHeads, headKD: driver.headKDim, numVH: Hv,
                                              keyDim: keyDim, valDim: valueDim, eps: driver.eps,
                                              qScale: invScale * invScale, kScale: invScale)
        for b in 0 ..< B {
            let gc = lanes[b].layers[li].gdnCache!
            laneDeltaStep(enc, q: driver.gdnSc.qn, qOff: b * keyDim * 2,
                          k: driver.gdnSc.kn, kOff: b * keyDim * 2,
                          v: driver.gdnSc.v1, vOff: b * valueDim * 2,
                          g: driver.gdnSc.g, gOff: b * Hv * 4,
                          beta: driver.gdnSc.beta, betaOff: b * Hv * 4,
                          stateIn: gc.state, stateOut: gc.stateOut,
                          y: driver.gdnSc.coreOut, yOff: b * valueDim * 2,
                          Hv: Hv, Dv: driver.headVDim)
            gc.swapState()
        }
        SeedlessFusedVerify.encodeGdnNormGateRows(enc, coreOut: driver.gdnSc.coreOut, z: driver.gdnSc.z,
                                                  normWeight: gw.normWeight, outV: driver.gdnSc.outV,
                                                  M: B, Hv: Hv, Dv: driver.headVDim,
                                                  eps: driver.eps, promoteF32: gw.promoteRMS)
        SeedlessFusedVerify.encodeQmmRows(enc, w: gw.outW, scales: gw.outS, biases: gw.outB,
                                          x: driver.gdnSc.outV, out: driver.mixerOut,
                                          M: B, K: valueDim, N: H)
    }

    /// Fast attn layer: qkv demux (M=B) → q-prep/k-prep/v-append/SDPA per lane
    /// (offset-bound, RoPE at the lane's own position) → sigmoid gate (M=B) → o_proj (M=B).
    private func encodeAttnLayerFast(_ enc: MTLComputeCommandEncoder, li: Int, aw: SeedlessFusedVerify.AttnLayerBufs) {
        let H = driver.H, nH = driver.numHeads, nKV = driver.numKV, hD = driver.headDim
        let qd2 = 2 * hD
        let scale = Float(pow(Double(hD), -0.5))
        SeedlessFusedVerify.encodeQmmInProjDemuxRows(
            enc, w: aw.catQkvW!, scales: aw.catQkvS!, biases: aw.catQkvB!, x: driver.normed,
            outQkv: driver.attnSc.qOut, outZ: driver.attnSc.kOut,
            outB: driver.attnSc.vOut, outA: aw.catQkvDummy!,
            M: B, K: H, dims: (qkv: nH * qd2, z: nKV * hD, b: nKV * hD, a: 0))
        for b in 0 ..< B {
            let kv = lanes[b].layers[li].kvCache!
            let baseLen = kv.len
            laneQPrep(enc, qOut: driver.attnSc.qOut, qOutOff: b * nH * qd2 * 2,
                      qNorm: aw.qNorm, qRot: driver.attnSc.qRot, qRotOff: b * nH * hD * 2,
                      qd2: qd2, headDim: hD, ropeDim: driver.ropeDim, base: driver.ropeBase,
                      startOffset: baseLen, numHeads: nH, eps: driver.eps)
            laneKPrep(enc, kOut: driver.attnSc.kOut, kOutOff: b * nKV * hD * 2,
                      kNorm: aw.kNorm, kRot: driver.attnSc.kRot, kRotOff: b * nKV * hD * 2,
                      kCache: kv.kCache, headDim: hD, ropeDim: driver.ropeDim, base: driver.ropeBase,
                      startOffset: baseLen, numKV: nKV, maxLen: kv.maxLen, eps: driver.eps)
            laneWriteKV(enc, src: driver.attnSc.vOut, srcOff: b * nKV * hD * 2,
                        cache: kv.vCache, KV: nKV, D: hD, maxLen: kv.maxLen, pos: baseLen)
            laneSdpa(enc, q: driver.attnSc.qRot, qOff: b * nH * hD * 2,
                     k: kv.kCache, v: kv.vCache, out: driver.attnSc.attnOut, outOff: b * nH * hD * 2,
                     H: nH, KV: nKV, D: hD, baseLenPlus1: baseLen + 1, scale: scale, maxLen: kv.maxLen)
            kv.len += 1
        }
        SeedlessFusedVerify.encodeSigmoidMul(enc, attnOut: driver.attnSc.attnOut, qOut: driver.attnSc.qOut,
                                             gated: driver.attnSc.gated,
                                             headDim: hD, qd2: qd2, total: B * nH * hD)
        SeedlessFusedVerify.encodeQmmRows(enc, w: aw.oW, scales: aw.oS, biases: aw.oB,
                                          x: driver.attnSc.gated, out: driver.mixerOut,
                                          M: B, K: nH * hD, N: H)
    }

    /// Fallback GDN layer (fusion flags off): in-proj at M=B where possible, mixer
    /// core per lane via the hybridDense seam, rows staged through lane scratch.
    private func encodeGdnLayerStaged(_ enc: MTLComputeCommandEncoder, li: Int, gw: SeedlessFusedVerify.GdnLayerBufs,
                                      keyDim: Int, valueDim: Int, convDim: Int) {
        let H = driver.H
        SeedlessFusedVerify.encodeQmmRows(enc, w: gw.qkvW, scales: gw.qkvS, biases: gw.qkvB,
                                          x: driver.normed, out: driver.gdnSc.qkv, M: B, K: H, N: convDim)
        SeedlessFusedVerify.encodeQmmRows(enc, w: gw.zW, scales: gw.zS, biases: gw.zB,
                                          x: driver.normed, out: driver.gdnSc.z, M: B, K: H, N: valueDim)
        for b in 0 ..< B {
            let lane = lanes[b]
            let gc = lane.layers[li].gdnCache!
            encodeRowCopy(enc, src: driver.normed, srcOff: b * H, dst: laneXs[b], dstOff: 0, count: H)
            encodeRowCopy(enc, src: driver.gdnSc.qkv, srcOff: b * convDim,
                          dst: lane.gdnSc.qkv, dstOff: 0, count: convDim)
            encodeRowCopy(enc, src: driver.gdnSc.z, srcOff: b * valueDim,
                          dst: lane.gdnSc.z, dstOff: 0, count: valueDim)
            SeedlessFusedVerify.encodeGdnLayerRows(enc, x: laneXs[b], out: laneOuts[b], w: gw,
                                                   sc: lane.gdnSc, cache: gc, M: 1, H: H,
                                                   numKHeads: driver.numKHeads, numVHeads: driver.numVHeads,
                                                   headKDim: driver.headKDim, headVDim: driver.headVDim,
                                                   convKernel: driver.convKernel, eps: driver.eps,
                                                   hybridDense: true)
            gc.swapState()
            encodeRowCopy(enc, src: lane.gdnSc.outV, srcOff: 0,
                          dst: driver.gdnSc.outV, dstOff: b * valueDim, count: valueDim)
        }
        SeedlessFusedVerify.encodeQmmRows(enc, w: gw.outW, scales: gw.outS, biases: gw.outB,
                                          x: driver.gdnSc.outV, out: driver.mixerOut,
                                          M: B, K: valueDim, N: H)
    }

    /// Fallback attn layer (fusion flags off): qkv at M=B, core per lane via the
    /// hybridDense seam, rows staged through lane scratch.
    private func encodeAttnLayerStaged(_ enc: MTLComputeCommandEncoder, li: Int, aw: SeedlessFusedVerify.AttnLayerBufs) {
        let H = driver.H, nH = driver.numHeads, nKV = driver.numKV, hD = driver.headDim
        let qd2 = 2 * hD
        SeedlessFusedVerify.encodeQmmRows(enc, w: aw.qW, scales: aw.qS, biases: aw.qB,
                                          x: driver.normed, out: driver.attnSc.qOut, M: B, K: H, N: nH * qd2)
        SeedlessFusedVerify.encodeQmmRows(enc, w: aw.kW, scales: aw.kS, biases: aw.kB,
                                          x: driver.normed, out: driver.attnSc.kOut, M: B, K: H, N: nKV * hD)
        SeedlessFusedVerify.encodeQmmRows(enc, w: aw.vW, scales: aw.vS, biases: aw.vB,
                                          x: driver.normed, out: driver.attnSc.vOut, M: B, K: H, N: nKV * hD)
        for b in 0 ..< B {
            let lane = lanes[b]
            let kv = lane.layers[li].kvCache!
            encodeRowCopy(enc, src: driver.attnSc.qOut, srcOff: b * nH * qd2,
                          dst: lane.attnSc.qOut, dstOff: 0, count: nH * qd2)
            encodeRowCopy(enc, src: driver.attnSc.kOut, srcOff: b * nKV * hD,
                          dst: lane.attnSc.kOut, dstOff: 0, count: nKV * hD)
            encodeRowCopy(enc, src: driver.attnSc.vOut, srcOff: b * nKV * hD,
                          dst: lane.attnSc.vOut, dstOff: 0, count: nKV * hD)
            // x unused by the hybridDense attn path (① skipped) — per-lane placeholder.
            SeedlessFusedVerify.encodeAttnLayerRows(enc, x: laneXs[b], out: laneOuts[b], w: aw,
                                                    sc: lane.attnSc, kv: kv, M: 1, H: H,
                                                    numHeads: nH, numKV: nKV,
                                                    headDim: hD, ropeDim: driver.ropeDim,
                                                    ropeBase: driver.ropeBase, eps: driver.eps,
                                                    hybridDense: true)
            kv.len += 1
            encodeRowCopy(enc, src: lane.attnSc.gated, srcOff: 0,
                          dst: driver.attnSc.gated, dstOff: b * nH * hD, count: nH * hD)
        }
        SeedlessFusedVerify.encodeQmmRows(enc, w: aw.oW, scales: aw.oS, biases: aw.oB,
                                          x: driver.attnSc.gated, out: driver.mixerOut,
                                          M: B, K: nH * hD, N: H)
    }

    /// Encode all layers for one batched step (rows already in driver.hBuf [B, H]).
    /// Shared by forwardRowsBatch and stepArgmaxBatch.
    private func encodeAllLayers(_ enc: MTLComputeCommandEncoder) {
        let H = driver.H
        let keyDim = driver.headKDim * driver.numKHeads
        let valueDim = driver.headVDim * driver.numVHeads
        let convDim = keyDim * 2 + valueDim
        let gdnFast = SeedlessFusedVerify.SeedlessFusedForward.fuseGDN
            && SeedlessFusedVerify._gdnPrepRowsPipeline != nil
            && SeedlessFusedVerify._convShiftFusedRowsPipeline != nil
        let attnFast = SeedlessFusedVerify.SeedlessFusedForward.fuseATTN
            && SeedlessFusedVerify.SeedlessFusedForward.fuseA1Enabled
            && SeedlessFusedVerify.ensureWave3Pipelines()

        for li in driver.layers.indices {
            let L = driver.layers[li]
            SeedlessFusedVerify.encodeRmsNormRows(enc, x: driver.hBuf, w: L.inputLN,
                                                  out: driver.normed, rows: B, D: H, eps: driver.eps)
            if L.isLinear, let gw = L.gdn {
                if gdnFast, gw.catInProjW != nil,
                   gw.totalInProjN == convDim + valueDim + 2 * driver.numVHeads,
                   gw.promoteRMS ? SeedlessFusedVerify._gdnNormGateRowsF32Pipeline != nil
                                 : SeedlessFusedVerify._gdnNormGateRowsPipeline != nil {
                    encodeGdnLayerFast(enc, li: li, gw: gw, keyDim: keyDim, valueDim: valueDim, convDim: convDim)
                } else {
                    encodeGdnLayerStaged(enc, li: li, gw: gw, keyDim: keyDim, valueDim: valueDim, convDim: convDim)
                }
            } else if let aw = L.attn {
                if attnFast, aw.catQkvW != nil, aw.catQkvDummy != nil {
                    encodeAttnLayerFast(enc, li: li, aw: aw)
                } else {
                    encodeAttnLayerStaged(enc, li: li, aw: aw)
                }
            }
            // resid + postNorm at M=B — mirrors encodePreMoE's tail exactly.
            if SeedlessFusedVerify.SeedlessFusedForward.fuseGDN,
               SeedlessFusedVerify._gdnResidPostNormRowsPipeline != nil {
                SeedlessFusedVerify.encodeGdnResidPostNormRows(enc, h: driver.hBuf, r: driver.mixerOut,
                                                               w: L.postLN, postNorm: driver.postNorm,
                                                               M: B, H: H, eps: driver.eps)
            } else {
                SeedlessFusedVerify.encodeResidAdd(enc, h: driver.hBuf, r: driver.mixerOut, total: B * H)
                SeedlessFusedVerify.encodeRmsNormRows(enc, x: driver.hBuf, w: L.postLN,
                                                      out: driver.postNorm, rows: B, D: H, eps: driver.eps)
            }
            SeedlessFusedVerify.encodeMoEBlockRows(enc, x: driver.postNorm, out: driver.moeOut,
                                                   w: L.moe, sc: driver.moeSc,
                                                   M: B, E: L.E, I: L.I, Ktop: L.Ktop, H: H)
            SeedlessFusedVerify.encodeResidAdd(enc, h: driver.hBuf, r: driver.moeOut, total: B * H)
        }
    }

    /// One batched step: row b of `x` [B, H] is lane b's next-token hidden input.
    /// Returns the residual-stream output rows [B, H] (pre-final-norm, mirroring
    /// forwardRows(finalNormW: nil)). Each lane's caches advance by exactly one
    /// position — bit-identical to that lane running forwardRows(M: 1) alone.
    public func forwardRowsBatch(_ x: MLXArray) -> MLXArray? {
        let H = driver.H
        guard x.dim(0) == B, x.dim(1) == H else { return nil }
        let xf = x.asType(.float16).reshaped([-1]); xf.eval()
        let arr = xf.asArray(Float16.self)
        driver.hBuf.contents().bindMemory(to: Float16.self, capacity: driver.maxM * H)
            .update(from: arr, count: B * H)

        let cb = driver.queue.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        encodeAllLayers(enc)
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        SeedlessFusedVerify.SeedlessFusedForward.profLastGPUMs = (cb.gpuEndTime - cb.gpuStartTime) * 1000.0

        let ptr = driver.hBuf.contents().bindMemory(to: Float16.self, capacity: driver.maxM * H)
        return MLXArray(Array(UnsafeBufferPointer(start: ptr, count: B * H)), [B, H])
    }

    /// One batched GREEDY step: token ids [B] → next token ids [B]. A single CB
    /// (embed → layers → final norm → lm_head → argmax), int32-only readback —
    /// the batched mirror of the driver's solo stepArgmax. The head stages run at
    /// M=B (row-independent ⇒ M-invariant); uses the DRIVER's attached head.
    /// Bit-identical per lane to that lane running solo stepArgmax (locked test
    /// lane_batch_argmax_bitexact).
    public func stepArgmaxBatch(_ tokens: [Int32]) -> [Int]? {
        guard let hd = driver.head, tokens.count == B,
              SeedlessFusedVerify._embedRowsPipeline != nil,
              SeedlessFusedVerify._argmaxRowsPipeline != nil else { return nil }
        let H = driver.H
        hd.tokensIn.contents().bindMemory(to: Int32.self, capacity: driver.maxM)
            .update(from: tokens, count: B)

        let cb = driver.queue.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        // embed: tokens → hBuf [B, H] (mirrors stepArgmax's encodeEmbed at M=B)
        let ep = SeedlessFusedVerify._embedRowsPipeline!
        enc.setComputePipelineState(ep)
        enc.setBuffer(hd.embedW, offset: 0, index: 0); enc.setBuffer(hd.embedS, offset: 0, index: 1)
        enc.setBuffer(hd.embedB, offset: 0, index: 2); enc.setBuffer(hd.tokensIn, offset: 0, index: 3)
        enc.setBuffer(driver.hBuf, offset: 0, index: 4)
        var hh = UInt32(H), tt = UInt32(B * H)
        enc.setBytes(&hh, length: 4, index: 5); enc.setBytes(&tt, length: 4, index: 6)
        enc.dispatchThreads(MTLSize(width: B * H, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: min(ep.maxTotalThreadsPerThreadgroup, 256), height: 1, depth: 1))
        encodeAllLayers(enc)
        // final norm → lm_head → argmax at M=B (mirrors stepArgmax's encodeFinalOps)
        SeedlessFusedVerify.encodeRmsNormRows(enc, x: driver.hBuf, w: hd.fnW,
                                              out: driver.normed, rows: B, D: H, eps: driver.eps)
        if SeedlessFusedVerify.SeedlessFusedForward.lmHeadQmv {
            SeedlessFusedVerify.encodeQmmRows(enc, w: hd.lmW, scales: hd.lmS, biases: hd.lmB,
                                              x: driver.normed, out: hd.logits, M: B, K: H, N: hd.vocab)
        } else {
            let qp = SeedlessMetalForward._qmm4TiledPipeline!
            enc.setComputePipelineState(qp)
            enc.setBuffer(hd.lmW, offset: 0, index: 0); enc.setBuffer(hd.lmS, offset: 0, index: 1)
            enc.setBuffer(hd.lmB, offset: 0, index: 2); enc.setBuffer(driver.normed, offset: 0, index: 3)
            enc.setBuffer(hd.logits, offset: 0, index: 4)
            var kk = Int32(H), nn = Int32(hd.vocab), mm = Int32(B)
            enc.setBytes(&kk, length: 4, index: 5); enc.setBytes(&nn, length: 4, index: 6); enc.setBytes(&mm, length: 4, index: 7)
            enc.dispatchThreadgroups(MTLSize(width: hd.vocab, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }
        let ap = SeedlessFusedVerify._argmaxRowsPipeline!
        enc.setComputePipelineState(ap)
        enc.setBuffer(hd.logits, offset: 0, index: 0); enc.setBuffer(hd.tokensOut, offset: 0, index: 1)
        var vv = UInt32(hd.vocab); enc.setBytes(&vv, length: 4, index: 2)
        enc.dispatchThreadgroups(MTLSize(width: B, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        SeedlessFusedVerify.SeedlessFusedForward.profLastGPUMs = (cb.gpuEndTime - cb.gpuStartTime) * 1000.0

        let ptr = hd.tokensOut.contents().bindMemory(to: Int32.self, capacity: driver.maxM)
        return (0 ..< B).map { Int(ptr[$0]) }
    }
}
