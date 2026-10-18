import Foundation
import Metal

// Coalesced F16 GEMV for the MTP draft head (QWISP_MTP_FAST_FMM, default on).
//
// fmm_rows computes one output per thread with a sequential, row-strided walk over K, so the
// head's dense projections (fc 16 MB, q 33 MB, o 17 MB, ... ≈ 78 MB of F16 per draft) ran far
// below read bandwidth: draft 4.6 ms GPU, pair feed ~2 ms GPU on an M3 Pro. Here a simdgroup owns
// 4 output rows and its 32 lanes stream K in contiguous 8-half chunks (float accumulate,
// simd_sum). The head only produces drafts — the bit-exact verify decides every emitted token —
// so changing its accumulation order cannot change the output stream. Shapes outside
// N % 8 == 0, K % 256 == 0 keep fmm_rows.
extension SeedlessFusedVerify {
    nonisolated(unsafe) static var _fmv16Pipeline: MTLComputePipelineState?
    static let mtpFastFmm = ProcessInfo.processInfo.environment["QWISP_MTP_FAST_FMM"] != "0"

    static func ensureFmv16Pipeline() -> Bool {
        if _fmv16Pipeline != nil { return true }
        guard let (device, _) = SeedlessMetalForward.ensure() else { return false }
        let src = """
        #include <metal_stdlib>
        using namespace metal;
        // out[m, n] = sum_k x[m, k] * W[n, k]; grid (M, N/8), threads (32, 2): 4 rows per simdgroup.
        kernel void fmv16_rows(device const half* W [[buffer(0)]], device const half* x [[buffer(1)]],
                               device half* out [[buffer(2)]], constant int& K [[buffer(3)]],
                               constant int& N [[buffer(4)]],
                               uint3 tid [[threadgroup_position_in_grid]],
                               uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
            const int row0 = (int)tid.y * 8 + (int)sg * 4;
            const device half* xm = x + (size_t)tid.x * K;
            float acc[4] = {0, 0, 0, 0};
            for (int k = (int)lane * 8; k < K; k += 256) {
                half4 xa = *(const device half4*)(xm + k), xb = *(const device half4*)(xm + k + 4);
                for (int r = 0; r < 4; r++) {
                    const device half* wr = W + (size_t)(row0 + r) * K + k;
                    half4 wa = *(const device half4*)wr, wb = *(const device half4*)(wr + 4);
                    acc[r] += dot(float4(xa), float4(wa)) + dot(float4(xb), float4(wb));
                }
            }
            for (int r = 0; r < 4; r++) {
                float s = simd_sum(acc[r]);
                if (lane == 0) out[(size_t)tid.x * N + row0 + r] = half(s);
            }
        }
        """
        do {
            let lib = try device.makeLibrary(source: src, options: SeedlessMetalForward.mlxMatchCompileOpts())
            _fmv16Pipeline = try device.makeComputePipelineState(function: lib.makeFunction(name: "fmv16_rows")!)
            return true
        } catch { print("[fmv16] compile: \(error)"); return false }
    }

    /// Fast path for encodeFmmRows; returns false when the shape or flag does not allow it.
    static func encodeFmv16Rows(_ enc: MTLComputeCommandEncoder, w: MTLBuffer, x: MTLBuffer, out: MTLBuffer,
                                M: Int, K: Int, N: Int) -> Bool {
        guard mtpFastFmm, N % 8 == 0, K % 256 == 0, ensureFmv16Pipeline(), let p = _fmv16Pipeline else { return false }
        enc.setComputePipelineState(p)
        enc.setBuffer(w, offset: 0, index: 0); enc.setBuffer(x, offset: 0, index: 1); enc.setBuffer(out, offset: 0, index: 2)
        var kk = Int32(K), nn = Int32(N)
        enc.setBytes(&kk, length: 4, index: 3); enc.setBytes(&nn, length: 4, index: 4)
        enc.dispatchThreadgroups(MTLSize(width: M, height: N / 8, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 32, height: 2, depth: 1))
        return true
    }
}

// Row-batched helpers for the MTP head's pair feed (feedPairs over M rows in one pass).
extension SeedlessFusedVerify {
    nonisolated(unsafe) static var _mtpConcatRowsPipeline: MTLComputePipelineState?
    nonisolated(unsafe) static var _mtpFinalCombineRowsPipeline: MTLComputePipelineState?

    static func ensureMTPRowsPipelines() -> Bool {
        if _mtpConcatRowsPipeline != nil && _mtpFinalCombineRowsPipeline != nil { return true }
        guard let (device, _) = SeedlessMetalForward.ensure() else { return false }
        let src = """
        #include <metal_stdlib>
        using namespace metal;
        // out[m, 0:H] = a[m], out[m, H:2H] = b[m]
        kernel void mtp_concat_rows(device const half* a [[buffer(0)]], device const half* b [[buffer(1)]],
                                    device half* out [[buffer(2)]], constant uint& H [[buffer(3)]],
                                    uint2 pos [[thread_position_in_grid]]) {
            uint i = pos.x, m = pos.y;
            if (i >= 2 * H) return;
            out[m * 2 * H + i] = i < H ? a[m * H + i] : b[m * H + i - H];
        }
        // out[m, n] = y[m, n] + sigmoid(sgl[m]) * sharedY[m, n]   (sgl stride 1)
        kernel void mtp_final_combine_rows(device const half* y [[buffer(0)]], device const half* sy [[buffer(1)]],
                                           device const half* sgl [[buffer(2)]], device half* out [[buffer(3)]],
                                           constant uint& N [[buffer(4)]], uint2 pos [[thread_position_in_grid]]) {
            uint n = pos.x, m = pos.y;
            if (n >= N) return;
            float g = 1.0f / (1.0f + exp(-float(sgl[m])));
            out[m * N + n] = half(float(y[m * N + n]) + g * float(sy[m * N + n]));
        }
        """
        do {
            let lib = try device.makeLibrary(source: src, options: SeedlessMetalForward.mlxMatchCompileOpts())
            _mtpConcatRowsPipeline = try device.makeComputePipelineState(function: lib.makeFunction(name: "mtp_concat_rows")!)
            _mtpFinalCombineRowsPipeline = try device.makeComputePipelineState(function: lib.makeFunction(name: "mtp_final_combine_rows")!)
            return true
        } catch { print("[mtp-rows] compile: \(error)"); return false }
    }

    static func encodeMTPConcatRows(_ enc: MTLComputeCommandEncoder, a: MTLBuffer, b: MTLBuffer, out: MTLBuffer, H: Int, M: Int) {
        let p = _mtpConcatRowsPipeline!
        enc.setComputePipelineState(p)
        enc.setBuffer(a, offset: 0, index: 0); enc.setBuffer(b, offset: 0, index: 1); enc.setBuffer(out, offset: 0, index: 2)
        var hh = UInt32(H); enc.setBytes(&hh, length: 4, index: 3)
        enc.dispatchThreads(MTLSize(width: 2 * H, height: M, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: Swift.min(p.maxTotalThreadsPerThreadgroup, 256), height: 1, depth: 1))
    }

    static func encodeMTPFinalCombineRows(_ enc: MTLComputeCommandEncoder, y: MTLBuffer, sharedY: MTLBuffer, sgl: MTLBuffer,
                                          out: MTLBuffer, N: Int, M: Int) {
        let p = _mtpFinalCombineRowsPipeline!
        enc.setComputePipelineState(p)
        enc.setBuffer(y, offset: 0, index: 0); enc.setBuffer(sharedY, offset: 0, index: 1)
        enc.setBuffer(sgl, offset: 0, index: 2); enc.setBuffer(out, offset: 0, index: 3)
        var nn = UInt32(N); enc.setBytes(&nn, length: 4, index: 4)
        enc.dispatchThreads(MTLSize(width: N, height: M, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: Swift.min(p.maxTotalThreadsPerThreadgroup, 256), height: 1, depth: 1))
    }
}

// Draft-vocabulary lm_head for the MTP head (QWISP_MTP_DRAFT_VOCAB, default 65536).
// The head's lm_head reads all 248K rows (286 MB, ~2.2 ms on an M3 Pro) for one argmax per draft.
// Drafting only needs the argmax over likely tokens, so the head scores a subset of rows (ids below
// the base plus every token the server has seen in prompts and outputs) and maps the winner back
// to its id. A token outside the subset can never be drafted — that lowers acceptance, never output.
extension SeedlessFusedVerify {
    nonisolated(unsafe) static var _qmm4GatherRowsPipeline: MTLComputePipelineState?
    nonisolated(unsafe) static var _argmaxMapPipeline: MTLComputePipelineState?

    static func ensureDraftVocabPipelines() -> Bool {
        if _qmm4GatherRowsPipeline != nil && _argmaxMapPipeline != nil { return true }
        guard let (device, _) = SeedlessMetalForward.ensure() else { return false }
        let src = """
        #include <metal_stdlib>
        using namespace metal;
        inline float ld16(const device half* x, thread float* xt) {
            float sum = 0.0f;
            for (int i = 0; i < 16; i += 4) { sum += x[i]+x[i+1]+x[i+2]+x[i+3];
                xt[i]=x[i]; xt[i+1]=x[i+1]/16.0f; xt[i+2]=x[i+2]/256.0f; xt[i+3]=x[i+3]/4096.0f; }
            return sum;
        }
        inline float qd4(const device uint8_t* w, const thread float* xt, float scale, float bias, float sum) {
            float accum = 0.0f; const device uint16_t* ws = (const device uint16_t*)w;
            for (int i = 0; i < 4; i++) {
                accum += (xt[4*i]*(float)(ws[i]&0x000f) + xt[4*i+1]*(float)(ws[i]&0x00f0) +
                          xt[4*i+2]*(float)(ws[i]&0x0f00) + xt[4*i+3]*(float)(ws[i]&0xf000));
            }
            return scale * accum + sum * bias;
        }
        // y[j] = W[ids[j]] · x  (4-bit gs64 rows; grid (1, n/8), threads (32, 2): 4 subset rows per simdgroup)
        kernel void qmm4_gather_rows(device const uint32_t* w [[buffer(0)]], device const half* scales [[buffer(1)]],
                                     device const half* biases [[buffer(2)]], device const half* x [[buffer(3)]],
                                     device half* y [[buffer(4)]], device const uint* ids [[buffer(5)]],
                                     constant int& K [[buffer(6)]],
                                     uint3 tid [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                                     uint lane [[thread_index_in_simdgroup]]) {
            const int Kw = K / 2, Kg = K / 64;   // bytes / groups per row
            const int j0 = (int)tid.y * 8 + (int)sg * 4;
            thread float xt[16]; float acc[4] = {0, 0, 0, 0};
            for (int k = (int)lane * 16; k < K; k += 512) {
                float sum = ld16(x + k, xt);
                for (int r = 0; r < 4; r++) {
                    const uint row = ids[j0 + r];
                    const device uint8_t* wr = (const device uint8_t*)w + (size_t)row * Kw + k / 2;
                    const int g = (int)row * Kg + k / 64;
                    acc[r] += qd4(wr, xt, (float)scales[g], (float)biases[g], sum);
                }
            }
            for (int r = 0; r < 4; r++) { float s = simd_sum(acc[r]); if (lane == 0) y[j0 + r] = (half)s; }
        }
        // out = ids[argmax_j logits[j]]  (one threadgroup of 1024)
        kernel void argmax_map(device const half* logits [[buffer(0)]], device const uint* ids [[buffer(1)]],
                               device int* out [[buffer(2)]], constant uint& n [[buffer(3)]],
                               uint t [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]]) {
            float best = -INFINITY; uint bi = 0;
            for (uint j = t; j < n; j += 1024) { float v = (float)logits[j]; if (v > best) { best = v; bi = j; } }
            for (uint off = 16; off > 0; off >>= 1) {
                float ov = simd_shuffle_down(best, off); uint oi = simd_shuffle_down(bi, off);
                if (ov > best || (ov == best && oi < bi)) { best = ov; bi = oi; }
            }
            threadgroup float tb[32]; threadgroup uint ti[32];
            if (lane == 0) { tb[sg] = best; ti[sg] = bi; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sg == 0) {
                best = tb[lane]; bi = ti[lane];
                for (uint off = 16; off > 0; off >>= 1) {
                    float ov = simd_shuffle_down(best, off); uint oi = simd_shuffle_down(bi, off);
                    if (ov > best || (ov == best && oi < bi)) { best = ov; bi = oi; }
                }
                if (lane == 0) out[0] = (int)ids[bi];
            }
        }
        """
        do {
            let lib = try device.makeLibrary(source: src, options: SeedlessMetalForward.mlxMatchCompileOpts())
            _qmm4GatherRowsPipeline = try device.makeComputePipelineState(function: lib.makeFunction(name: "qmm4_gather_rows")!)
            _argmaxMapPipeline = try device.makeComputePipelineState(function: lib.makeFunction(name: "argmax_map")!)
            return true
        } catch { print("[draft-vocab] compile: \(error)"); return false }
    }

    /// logits over a row subset (n % 8 == 0) then the argmax's id into out[outOffset].
    static func encodeDraftVocabArgmax(_ enc: MTLComputeCommandEncoder, w: MTLBuffer, scales: MTLBuffer, biases: MTLBuffer,
                                       x: MTLBuffer, ids: MTLBuffer, n: Int, K: Int, logits: MTLBuffer,
                                       out: MTLBuffer, outOffset: Int) {
        let gp = _qmm4GatherRowsPipeline!
        enc.setComputePipelineState(gp)
        enc.setBuffer(w, offset: 0, index: 0); enc.setBuffer(scales, offset: 0, index: 1); enc.setBuffer(biases, offset: 0, index: 2)
        enc.setBuffer(x, offset: 0, index: 3); enc.setBuffer(logits, offset: 0, index: 4); enc.setBuffer(ids, offset: 0, index: 5)
        var kk = Int32(K); enc.setBytes(&kk, length: 4, index: 6)
        enc.dispatchThreadgroups(MTLSize(width: 1, height: n / 8, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 2, depth: 1))
        let ap = _argmaxMapPipeline!
        enc.setComputePipelineState(ap)
        enc.setBuffer(logits, offset: 0, index: 0); enc.setBuffer(ids, offset: 0, index: 1); enc.setBuffer(out, offset: outOffset, index: 2)
        var nn = UInt32(n); enc.setBytes(&nn, length: 4, index: 3)
        enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1024, height: 1, depth: 1))
    }
}
