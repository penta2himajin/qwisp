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
