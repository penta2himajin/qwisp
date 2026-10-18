import Foundation
import Metal

// Shared expert folded into the routed MoE dispatches (M = 1 decode, resident weights).
//
// Measured on M3 Pro (moe-roofline, QWISP_MOEB_K9=1): the shared expert's two GEMVs run at
// 60-80 % of copy bandwidth on their own (1.18 MB + 0.59 MB are too small to fill the GPU),
// while adding a ninth slice to the routed g/u and down dispatches costs +9-11 µs and
// +0-5 µs against 13-14 µs and 7-8 µs standalone. Folding saves ~6-10 µs per layer.
//
// Bit-exactness: slice Ktop of each dispatch is the shared expert, computed with the same
// per-row arithmetic as the standalone path (shared S1 already runs gqmm4_swiglu_rows with
// Ktop = 1, inds = [0]; shared down runs qmm4, whose row loop is identical to gqmm4_rows).
// Slices 0..<Ktop are the routed experts, unchanged. The shared slice writes h[Ktop] and
// d[Ktop], so MoEScratch.h and .d carry one extra slice.

extension SeedlessMetalForward {
    nonisolated(unsafe) static var _gqmmSwigluRowsShPipeline: MTLComputePipelineState?
    nonisolated(unsafe) static var _gqmmRowsShPipeline: MTLComputePipelineState?

    static func compileSharedInRouted() -> Bool {
        if _gqmmSwigluRowsShPipeline != nil && _gqmmRowsShPipeline != nil { return true }
        guard let (device, _) = ensure() else { return false }
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
        // gqmm4_swiglu_rows with slice z == nrouted taken from the shared expert's weights.
        kernel void gqmm4_swiglu_rows_sh(device const uint32_t* gw [[buffer(0)]], device const half* gsc [[buffer(1)]], device const half* gbi [[buffer(2)]],
                                         device const uint32_t* uw [[buffer(3)]], device const half* usc [[buffer(4)]], device const half* ubi [[buffer(5)]],
                                         device const half* x [[buffer(6)]], device const int* inds [[buffer(7)]], device half* h [[buffer(8)]],
                                         constant int& in_vec_size [[buffer(9)]], constant int& out_vec_size [[buffer(10)]],
                                         constant int& ktop [[buffer(11)]],
                                         device const int* stopFlag [[buffer(12)]],
                                         device const uint32_t* sgw [[buffer(13)]], device const half* sgsc [[buffer(14)]], device const half* sgbi [[buffer(15)]],
                                         device const uint32_t* suw [[buffer(16)]], device const half* susc [[buffer(17)]], device const half* subi [[buffer(18)]],
                                         constant int& nrouted [[buffer(19)]],
                                         uint3 tid [[threadgroup_position_in_grid]], uint simd_gid [[simdgroup_index_in_threadgroup]], uint simd_lid [[thread_index_in_simdgroup]]) {
            if (stopFlag[0] != 0) return;
            constexpr int packs_per_thread = 2, num_simdgroups = 2, results_per_simdgroup = 4;
            constexpr int pack_factor = 8, bytes_per_pack = 4, values_per_thread = 16;
            constexpr int block_size = 512, scale_step_per_thread = 4;
            typedef float U;
            thread U x_thread[16]; thread U gres[4] = {0}, ures[4] = {0};
            const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
            const int in_vec_size_g = in_vec_size / 64;
            uint mk = tid.z;
            const device uint8_t* gws; const device uint8_t* uws;
            if (mk == (uint)nrouted) {
                gws = (const device uint8_t*)sgw; uws = (const device uint8_t*)suw;
                gsc = sgsc; gbi = sgbi; usc = susc; ubi = subi;
            } else {
                uint e = (uint)inds[mk];
                size_t eOffW = (size_t)e * out_vec_size * in_vec_size_w, eOffG = (size_t)e * out_vec_size * in_vec_size_g;
                gws = (const device uint8_t*)gw + eOffW; uws = (const device uint8_t*)uw + eOffW;
                gsc += eOffG; gbi += eOffG; usc += eOffG; ubi += eOffG;
            }
            const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) + simd_gid * results_per_simdgroup;
            gws += out_row*in_vec_size_w + simd_lid*packs_per_thread*bytes_per_pack;
            uws += out_row*in_vec_size_w + simd_lid*packs_per_thread*bytes_per_pack;
            gsc += out_row*in_vec_size_g + simd_lid/scale_step_per_thread; gbi += out_row*in_vec_size_g + simd_lid/scale_step_per_thread;
            usc += out_row*in_vec_size_g + simd_lid/scale_step_per_thread; ubi += out_row*in_vec_size_g + simd_lid/scale_step_per_thread;
            x += simd_lid * values_per_thread;   // M == 1: every slice reads row 0
            h += (size_t)mk * out_vec_size + out_row;
            for (int k = 0; k < in_vec_size; k += block_size) {
                U sum = ld16(x, x_thread);
                for (int row = 0; row < results_per_simdgroup; row++) {
                    gres[row] += qd4((const device uint8_t*)(gws + row*in_vec_size_w), x_thread, gsc[row*in_vec_size_g], gbi[row*in_vec_size_g], sum);
                    ures[row] += qd4((const device uint8_t*)(uws + row*in_vec_size_w), x_thread, usc[row*in_vec_size_g], ubi[row*in_vec_size_g], sum);
                }
                gws += block_size*bytes_per_pack/pack_factor; uws += block_size*bytes_per_pack/pack_factor;
                gsc += block_size/64; gbi += block_size/64; usc += block_size/64; ubi += block_size/64; x += block_size;
            }
            for (int row = 0; row < results_per_simdgroup; row++) {
                half gv = (half)simd_sum(gres[row]); half uv = (half)simd_sum(ures[row]);
                if (simd_lid == 0) { half y = (half)1/((half)1+exp(metal::abs(gv))); half s = (gv<(half)0)?y:((half)1-y); h[row] = (gv*s)*uv; }
            }
        }
        // gqmm4_rows (lhsPer) with slice z == nrouted taken from the shared expert's down weights.
        kernel void gqmm4_rows_sh(device const uint32_t* w      [[buffer(0)]],
                                  device const half*     scales [[buffer(1)]],
                                  device const half*     biases [[buffer(2)]],
                                  device const half*     x      [[buffer(3)]],
                                  device const int*      inds   [[buffer(4)]],
                                  device half*           y      [[buffer(5)]],
                                  constant int& in_vec_size  [[buffer(6)]],
                                  constant int& out_vec_size [[buffer(7)]],
                                  device const int* stopFlag [[buffer(9)]],
                                  device const uint32_t* sw  [[buffer(11)]],
                                  device const half*     ss  [[buffer(12)]],
                                  device const half*     sb  [[buffer(13)]],
                                  constant int& nrouted      [[buffer(14)]],
                                  uint3 tid      [[threadgroup_position_in_grid]],
                                  uint  simd_gid [[simdgroup_index_in_threadgroup]],
                                  uint  simd_lid [[thread_index_in_simdgroup]]) {
            if (stopFlag[0] != 0) return;
            constexpr int packs_per_thread = 2, num_simdgroups = 2, results_per_simdgroup = 4;
            constexpr int pack_factor = 8, bytes_per_pack = 4, values_per_thread = 16;
            constexpr int block_size = 512, scale_step_per_thread = 4;
            typedef float U;
            thread U x_thread[16];
            thread U result[4] = {0};
            const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
            const int in_vec_size_g = in_vec_size / 64;
            uint mk = tid.z;
            const device uint8_t* ws;
            if (mk == (uint)nrouted) {
                ws = (const device uint8_t*)sw; scales = ss; biases = sb;
            } else {
                uint e = (uint)inds[mk];
                ws      = (const device uint8_t*)w + (size_t)e * out_vec_size * in_vec_size_w;
                scales += (size_t)e * out_vec_size * in_vec_size_g;
                biases += (size_t)e * out_vec_size * in_vec_size_g;
            }
            const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) + simd_gid * results_per_simdgroup;
            ws     += out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
            scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
            biases += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
            x += (size_t)mk * in_vec_size + simd_lid * values_per_thread;
            y += (size_t)mk * out_vec_size + out_row;
            for (int k = 0; k < in_vec_size; k += block_size) {
                U sum = ld16(x, x_thread);
                for (int row = 0; row < results_per_simdgroup; row++) {
                    auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
                    const device half* sl = scales + row * in_vec_size_g;
                    const device half* bl = biases + row * in_vec_size_g;
                    U s = sl[0]; U b = bl[0];
                    result[row] += qd4(wl, x_thread, s, b, sum);
                }
                ws += block_size * bytes_per_pack / pack_factor;
                scales += block_size / 64; biases += block_size / 64; x += block_size;
            }
            for (int row = 0; row < results_per_simdgroup; row++) {
                result[row] = simd_sum(result[row]);
                if (simd_lid == 0) y[row] = (half)result[row];
            }
        }
        """
        do {
            let lib = try device.makeLibrary(source: src, options: mlxMatchCompileOpts())
            _gqmmSwigluRowsShPipeline = try device.makeComputePipelineState(function: lib.makeFunction(name: "gqmm4_swiglu_rows_sh")!)
            _gqmmRowsShPipeline = try device.makeComputePipelineState(function: lib.makeFunction(name: "gqmm4_rows_sh")!)
            return true
        } catch { print("[shared-in-routed] compile: \(error)"); return false }
    }
}

extension SeedlessFusedVerify {
    /// Routed experts + shared expert for one decode row: g/u+swiglu and down as two dispatches of
    /// Ktop + 1 slices each, then the S2 gate/combine reading the shared output from d[Ktop].
    /// Caller guarantees M == 1, resident weights (no slot table, no mixed residency), fuseSHEXP.
    static func encodeMoEGatherSharedM1(_ enc: MTLComputeCommandEncoder, x: MTLBuffer, out: MTLBuffer,
                                        w: MoEBlockBufs, sc: MoEScratch, Ktop: Int, I: Int, H: Int) {
        let n = Int32(Ktop)
        enc.setComputePipelineState(SeedlessMetalForward._gqmmSwigluRowsShPipeline!)
        enc.setBuffer(w.swGW, offset: 0, index: 0); enc.setBuffer(w.swGS, offset: 0, index: 1); enc.setBuffer(w.swGB, offset: 0, index: 2)
        enc.setBuffer(w.swUW, offset: 0, index: 3); enc.setBuffer(w.swUS, offset: 0, index: 4); enc.setBuffer(w.swUB, offset: 0, index: 5)
        enc.setBuffer(x, offset: 0, index: 6); enc.setBuffer(sc.inds, offset: 0, index: 7); enc.setBuffer(sc.h, offset: 0, index: 8)
        var kk = Int32(H), nn = Int32(I), kt = n, nr = n
        enc.setBytes(&kk, length: 4, index: 9); enc.setBytes(&nn, length: 4, index: 10); enc.setBytes(&kt, length: 4, index: 11)
        SeedlessMetalForward.bindStop(enc, 12)
        enc.setBuffer(w.shGW, offset: 0, index: 13); enc.setBuffer(w.shGS, offset: 0, index: 14); enc.setBuffer(w.shGB, offset: 0, index: 15)
        enc.setBuffer(w.shUW, offset: 0, index: 16); enc.setBuffer(w.shUS, offset: 0, index: 17); enc.setBuffer(w.shUB, offset: 0, index: 18)
        enc.setBytes(&nr, length: 4, index: 19)
        enc.dispatchThreadgroups(MTLSize(width: 1, height: I / 8, depth: Ktop + 1),
                                 threadsPerThreadgroup: MTLSize(width: 32, height: 2, depth: 1))

        enc.setComputePipelineState(SeedlessMetalForward._gqmmRowsShPipeline!)
        enc.setBuffer(w.swDW, offset: 0, index: 0); enc.setBuffer(w.swDS, offset: 0, index: 1); enc.setBuffer(w.swDB, offset: 0, index: 2)
        enc.setBuffer(sc.h, offset: 0, index: 3); enc.setBuffer(sc.inds, offset: 0, index: 4); enc.setBuffer(sc.d, offset: 0, index: 5)
        var dk = Int32(I), dn = Int32(H)
        enc.setBytes(&dk, length: 4, index: 6); enc.setBytes(&dn, length: 4, index: 7)
        SeedlessMetalForward.bindStop(enc, 9)
        enc.setBuffer(w.shDW, offset: 0, index: 11); enc.setBuffer(w.shDS, offset: 0, index: 12); enc.setBuffer(w.shDB, offset: 0, index: 13)
        enc.setBytes(&nr, length: 4, index: 14)
        enc.dispatchThreadgroups(MTLSize(width: 1, height: H / 8, depth: Ktop + 1),
                                 threadsPerThreadgroup: MTLSize(width: 32, height: 2, depth: 1))

        let shOff = Ktop * H * 2
        if SeedlessFusedForward.fuseMOE2Enabled {
            encodeSharedGateCombineRowsFold(enc, sgW: w.sgW, sgS: w.sgS, sgB: w.sgB,
                                            x: x, d: sc.d, scores: sc.scores,
                                            sharedY: sc.d, out: out,
                                            K: H, H: H, M: 1, Ktop: Ktop, sharedYOffset: shOff)
        } else {
            encodeCombineRows(enc, d: sc.d, scores: sc.scores, y: sc.y, Ktop: Ktop, N: H, M: 1,
                              dByteOffset: 0, scoresOffset: 0, yByteOffset: 0)
            encodeSharedGateCombineRows(enc, sgW: w.sgW, sgS: w.sgS, sgB: w.sgB,
                                        x: x, y: sc.y, sharedY: sc.d, out: out,
                                        K: H, H: H, M: 1, sharedYOffset: shOff)
        }
    }
}
