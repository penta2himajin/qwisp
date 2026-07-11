import Foundation
import Metal
import MLX

// Grouped-MoE gather kernel PoC (prefill kernel-speedup recon, branch claude/prefill-kernel).
//
// Hypothesis (from prefill-stage-profile): prefill MoE (50.8% of prefill GPU time) is flat in
// chunk size because gqmm4_rows assigns one threadgroup per (row, expert) PAIR — every row
// assigned to an expert re-reads that expert's full weight slab from DRAM (union >> SLC at
// prefill M, so the hardware cannot amortize; 484e401's own analysis says explicit grouping
// only wins when working set > L2 — decode B=8 didn't qualify, prefill does).
//
// Prototype: gqmm4_grouped_rows — one threadgroup per (EXPERT GROUP, n-tile). The tg loops the
// R rows routed to that expert; the 4-output-row × K weight slice it walks (~4KB) stays in L1
// across the R iterations, so DRAM weight traffic drops ~R×. Bit-exact BY CONSTRUCTION: the
// per-(row, out_row) instruction sequence (ld16 / qd4 K-walk / simd_sum) is byte-identical to
// gqmm4_rows — only the tg→work mapping changes, and rows are independent outputs.
//
// This is measurement-only scaffolding: synthetic weights/routing, self-contained pipelines.
// Production wiring happens only if this bench wins (then via devloop, RAWTESTS-gated).
public enum GroupedMoEPoC {
    nonisolated(unsafe) static var pipeRef: MTLComputePipelineState?
    nonisolated(unsafe) static var pipeGrp: MTLComputePipelineState?
    nonisolated(unsafe) static var pipeGrp2: MTLComputePipelineState?
    nonisolated(unsafe) static var pipeNoDeq: MTLComputePipelineState?

    static let kernelSrc = """
    #include <metal_stdlib>
    using namespace metal;
    inline float ld16(const device half* x, thread float* xt) {
        float sum = 0.0f;
        for (int i = 0; i < 16; i += 4) {
            sum += x[i] + x[i+1] + x[i+2] + x[i+3];
            xt[i] = x[i]; xt[i+1] = x[i+1]/16.0f; xt[i+2] = x[i+2]/256.0f; xt[i+3] = x[i+3]/4096.0f;
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

    // Reference: verbatim gqmm4_rows structure (one tg per (row,expert) pair).
    kernel void gq_ref(device const uint32_t* w      [[buffer(0)]],
                       device const half*     scales [[buffer(1)]],
                       device const half*     biases [[buffer(2)]],
                       device const half*     x      [[buffer(3)]],
                       device const int*      inds   [[buffer(4)]],
                       device half*           y      [[buffer(5)]],
                       constant int& in_vec_size  [[buffer(6)]],
                       constant int& out_vec_size [[buffer(7)]],
                       constant int& ktop         [[buffer(8)]],
                       constant uint& lhsPer      [[buffer(9)]],
                       uint3 tid      [[threadgroup_position_in_grid]],
                       uint  simd_gid [[simdgroup_index_in_threadgroup]],
                       uint  simd_lid [[thread_index_in_simdgroup]]) {
        constexpr int packs_per_thread = 2, results_per_simdgroup = 4;
        constexpr int bytes_per_pack = 4, values_per_thread = 16;
        constexpr int block_size = 512, scale_step_per_thread = 4;
        const device uint8_t* ws = (const device uint8_t*)w;
        typedef float U;
        thread U x_thread[16];
        thread U result[4] = {0};
        const int in_vec_size_w = in_vec_size / 2;
        const int in_vec_size_g = in_vec_size / 64;
        uint mk = tid.z;
        uint e = (uint)inds[mk];
        ws     += (size_t)e * out_vec_size * in_vec_size_w;
        scales += (size_t)e * out_vec_size * in_vec_size_g;
        biases += (size_t)e * out_vec_size * in_vec_size_g;
        const int out_row = tid.y * 8 + simd_gid * results_per_simdgroup;
        ws     += out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
        scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
        biases += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
        x += (size_t)(lhsPer ? mk : mk / (uint)ktop) * in_vec_size + simd_lid * values_per_thread;
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
            ws += block_size / 2;
            scales += block_size / 64; biases += block_size / 64; x += block_size;
        }
        for (int row = 0; row < results_per_simdgroup; row++) {
            result[row] = simd_sum(result[row]);
            if (simd_lid == 0) y[row] = (half)result[row];
        }
    }

    // Grouped: one tg per (expert group, n-tile); loops the R rows of the group. Per-(row,out_row)
    // arithmetic sequence identical to gq_ref; the tg's 4-row weight slice stays hot in L1 across r.
    kernel void gq_grouped(device const uint32_t* w      [[buffer(0)]],
                           device const half*     scales [[buffer(1)]],
                           device const half*     biases [[buffer(2)]],
                           device const half*     x      [[buffer(3)]],
                           device const int*      gExpert [[buffer(4)]],   // [G]
                           device const int*      gRowOff [[buffer(5)]],   // [G+1] CSR offsets
                           device const int*      gRowIdx [[buffer(6)]],   // [M*Ktop] mk indices
                           device half*           y      [[buffer(7)]],
                           constant int& in_vec_size  [[buffer(8)]],
                           constant int& out_vec_size [[buffer(9)]],
                           constant int& ktop         [[buffer(10)]],
                           constant uint& lhsPer      [[buffer(11)]],
                           uint3 tid      [[threadgroup_position_in_grid]],
                           uint  simd_gid [[simdgroup_index_in_threadgroup]],
                           uint  simd_lid [[thread_index_in_simdgroup]]) {
        constexpr int packs_per_thread = 2, results_per_simdgroup = 4;
        constexpr int bytes_per_pack = 4, values_per_thread = 16;
        constexpr int block_size = 512, scale_step_per_thread = 4;
        const device uint8_t* ws0 = (const device uint8_t*)w;
        typedef float U;
        const int in_vec_size_w = in_vec_size / 2;
        const int in_vec_size_g = in_vec_size / 64;
        uint g = tid.z;
        uint e = (uint)gExpert[g];
        ws0    += (size_t)e * out_vec_size * in_vec_size_w;
        scales += (size_t)e * out_vec_size * in_vec_size_g;
        biases += (size_t)e * out_vec_size * in_vec_size_g;
        const int out_row = tid.y * 8 + simd_gid * results_per_simdgroup;
        ws0    += out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
        scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
        biases += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
        const int i0 = gRowOff[g], i1 = gRowOff[g + 1];
        for (int i = i0; i < i1; ++i) {
            uint mk = (uint)gRowIdx[i];
            const device half* xr = x + (size_t)(lhsPer ? mk : mk / (uint)ktop) * in_vec_size + simd_lid * values_per_thread;
            device half* yr = y + (size_t)mk * out_vec_size + out_row;
            thread U x_thread[16];
            thread U result[4] = {0};
            auto wsr = ws0;
            auto sr = scales; auto br = biases;
            for (int k = 0; k < in_vec_size; k += block_size) {
                U sum = ld16(xr, x_thread);
                for (int row = 0; row < results_per_simdgroup; row++) {
                    auto wl = (const device uint8_t*)(wsr + row * in_vec_size_w);
                    const device half* sl = sr + row * in_vec_size_g;
                    const device half* bl = br + row * in_vec_size_g;
                    U s = sl[0]; U b = bl[0];
                    result[row] += qd4(wl, x_thread, s, b, sum);
                }
                wsr += block_size / 2;
                sr += block_size / 64; br += block_size / 64; xr += block_size;
            }
            for (int row = 0; row < results_per_simdgroup; row++) {
                result[row] = simd_sum(result[row]);
                if (simd_lid == 0) yr[row] = (half)result[row];
            }
        }
    }
    // v2: threadgroup-shared UNPACKED weights — the (float)(ws & mask) dequant converts are done
    // ONCE per (group, K-block) and shared across the group's rows (dequant is ~half the inner-loop
    // ALU and the per-pair kernel is ALU-bound, per v1). Bit-exact: the shared values are the exact
    // floats qd4 computes, and each (row, out_row) accumulates them in the identical i/block order
    // with the identical simd_sum. Groups are capped at R_MAX=8 rows (register accumulators).
    kernel void gq_grouped2(device const uint32_t* w      [[buffer(0)]],
                            device const half*     scales [[buffer(1)]],
                            device const half*     biases [[buffer(2)]],
                            device const half*     x      [[buffer(3)]],
                            device const int*      gExpert [[buffer(4)]],
                            device const int*      gRowOff [[buffer(5)]],
                            device const int*      gRowIdx [[buffer(6)]],
                            device half*           y      [[buffer(7)]],
                            constant int& in_vec_size  [[buffer(8)]],
                            constant int& out_vec_size [[buffer(9)]],
                            constant int& ktop         [[buffer(10)]],
                            constant uint& lhsPer      [[buffer(11)]],
                            uint3 tid      [[threadgroup_position_in_grid]],
                            uint  simd_gid [[simdgroup_index_in_threadgroup]],
                            uint  simd_lid [[thread_index_in_simdgroup]]) {
        constexpr int packs_per_thread = 2, results_per_simdgroup = 4;
        constexpr int bytes_per_pack = 4, values_per_thread = 16;
        constexpr int block_size = 512, scale_step_per_thread = 4;
        constexpr int R_MAX = 8;
        const device uint8_t* ws0 = (const device uint8_t*)w;
        typedef float U;
        threadgroup float wsh[8][512];                       // [simd_gid*4+row][block value]
        const int in_vec_size_w = in_vec_size / 2;
        const int in_vec_size_g = in_vec_size / 64;
        uint g = tid.z;
        uint e = (uint)gExpert[g];
        ws0    += (size_t)e * out_vec_size * in_vec_size_w;
        scales += (size_t)e * out_vec_size * in_vec_size_g;
        biases += (size_t)e * out_vec_size * in_vec_size_g;
        const int out_row = tid.y * 8 + simd_gid * results_per_simdgroup;
        ws0    += out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
        scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
        biases += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
        const int i0 = gRowOff[g];
        const int R = min(gRowOff[g + 1] - i0, R_MAX);
        thread U result[R_MAX][4] = {{0}};
        thread U x_thread[16];
        auto wsr = ws0; auto sr = scales; auto br = biases;
        for (int k = 0; k < in_vec_size; k += block_size) {
            // unpack this K-block's 4 weight rows (per simdgroup) into shared memory — exactly the
            // masked floats qd4 uses; each lane converts its own 16-value slice per row.
            for (int row = 0; row < results_per_simdgroup; row++) {
                const device uint16_t* wl = (const device uint16_t*)(wsr + row * in_vec_size_w);
                threadgroup float* dst = wsh[simd_gid * results_per_simdgroup + row] + simd_lid * values_per_thread;
                for (int i = 0; i < 4; i++) {
                    uint16_t v = wl[i];
                    dst[4*i]   = (float)(v & 0x000f);
                    dst[4*i+1] = (float)(v & 0x00f0);
                    dst[4*i+2] = (float)(v & 0x0f00);
                    dst[4*i+3] = (float)(v & 0xf000);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (int r = 0; r < R; ++r) {
                uint mk = (uint)gRowIdx[i0 + r];
                const device half* xr = x + (size_t)(lhsPer ? mk : mk / (uint)ktop) * in_vec_size + k + simd_lid * values_per_thread;
                U sum = ld16(xr, x_thread);
                for (int row = 0; row < results_per_simdgroup; row++) {
                    const threadgroup float* wf = wsh[simd_gid * results_per_simdgroup + row] + simd_lid * values_per_thread;
                    const device half* sl = sr + row * in_vec_size_g;
                    const device half* bl = br + row * in_vec_size_g;
                    U s = sl[0]; U b = bl[0];
                    U accum = 0.0f;
                    for (int i = 0; i < 4; i++) {
                        accum += (x_thread[4*i]   * wf[4*i] +
                                  x_thread[4*i+1] * wf[4*i+1] +
                                  x_thread[4*i+2] * wf[4*i+2] +
                                  x_thread[4*i+3] * wf[4*i+3]);
                    }
                    result[r][row] += s * accum + sum * b;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            wsr += block_size / 2;
            sr += block_size / 64; br += block_size / 64;
        }
        for (int r = 0; r < R; ++r) {
            uint mk = (uint)gRowIdx[i0 + r];
            device half* yr = y + (size_t)mk * out_vec_size + out_row;
            for (int row = 0; row < results_per_simdgroup; row++) {
                U v = simd_sum(result[r][row]);
                if (simd_lid == 0) yr[row] = (half)v;
            }
        }
    }

    // Timing-only: gq_ref with the dequant ALU (mask+convert) REMOVED — multiplies the raw packed
    // uint16 reinterpreted as float (garbage output). Isolates the dequant tax: if this is not much
    // faster than gq_ref, the kernel is FMA/load-bound and NO dequant-sharing scheme can win.
    kernel void gq_nodeq(device const uint32_t* w      [[buffer(0)]],
                         device const half*     scales [[buffer(1)]],
                         device const half*     biases [[buffer(2)]],
                         device const half*     x      [[buffer(3)]],
                         device const int*      inds   [[buffer(4)]],
                         device half*           y      [[buffer(5)]],
                         constant int& in_vec_size  [[buffer(6)]],
                         constant int& out_vec_size [[buffer(7)]],
                         constant int& ktop         [[buffer(8)]],
                         constant uint& lhsPer      [[buffer(9)]],
                         uint3 tid      [[threadgroup_position_in_grid]],
                         uint  simd_gid [[simdgroup_index_in_threadgroup]],
                         uint  simd_lid [[thread_index_in_simdgroup]]) {
        constexpr int packs_per_thread = 2, results_per_simdgroup = 4;
        constexpr int bytes_per_pack = 4, values_per_thread = 16;
        constexpr int block_size = 512, scale_step_per_thread = 4;
        const device uint8_t* ws = (const device uint8_t*)w;
        typedef float U;
        thread U x_thread[16];
        thread U result[4] = {0};
        const int in_vec_size_w = in_vec_size / 2;
        const int in_vec_size_g = in_vec_size / 64;
        uint mk = tid.z;
        uint e = (uint)inds[mk];
        ws     += (size_t)e * out_vec_size * in_vec_size_w;
        scales += (size_t)e * out_vec_size * in_vec_size_g;
        biases += (size_t)e * out_vec_size * in_vec_size_g;
        const int out_row = tid.y * 8 + simd_gid * results_per_simdgroup;
        ws     += out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
        scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
        biases += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
        x += (size_t)(lhsPer ? mk : mk / (uint)ktop) * in_vec_size + simd_lid * values_per_thread;
        y += (size_t)mk * out_vec_size + out_row;
        for (int k = 0; k < in_vec_size; k += block_size) {
            U sum = ld16(x, x_thread);
            for (int row = 0; row < results_per_simdgroup; row++) {
                const device uint16_t* wl = (const device uint16_t*)(ws + row * in_vec_size_w);
                const device half* sl = scales + row * in_vec_size_g;
                const device half* bl = biases + row * in_vec_size_g;
                U s = sl[0]; U b = bl[0];
                U accum = 0.0f;
                for (int i = 0; i < 4; i++) {
                    // same loads, same FMA count — mask+convert dropped (as_type reinterpret)
                    accum += (x_thread[4*i]   * as_type<half>(wl[i]) +
                              x_thread[4*i+1] * as_type<half>(wl[i]) +
                              x_thread[4*i+2] * as_type<half>(wl[i]) +
                              x_thread[4*i+3] * as_type<half>(wl[i]));
                }
                result[row] += s * accum + sum * b;
            }
            ws += block_size / 2;
            scales += block_size / 64; biases += block_size / 64; x += block_size;
        }
        for (int row = 0; row < results_per_simdgroup; row++) {
            result[row] = simd_sum(result[row]);
            if (simd_lid == 0) y[row] = (half)result[row];
        }
    }
    """

    static func compile(_ device: MTLDevice) -> Bool {
        if pipeRef != nil { return true }
        do {
            let lib = try device.makeLibrary(source: kernelSrc, options: nil)
            pipeRef = try device.makeComputePipelineState(function: lib.makeFunction(name: "gq_ref")!)
            pipeGrp = try device.makeComputePipelineState(function: lib.makeFunction(name: "gq_grouped")!)
            pipeGrp2 = try device.makeComputePipelineState(function: lib.makeFunction(name: "gq_grouped2")!)
            pipeNoDeq = try device.makeComputePipelineState(function: lib.makeFunction(name: "gq_nodeq")!)
            return true
        } catch { print("[grouped-moe] compile: \(error)"); return false }
    }

    // Deterministic LCG (no seeded RNG dependency; reproducible bench).
    struct LCG { var s: UInt64; mutating func next() -> UInt64 { s = s &* 6364136223846793005 &+ 1442695040888963407; return s >> 33 } }

    public static func bench() -> String {
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue(),
              compile(device) else { return "[grouped-moe] setup fail\nGROUPEDMOE done" }
        let E = 256, Ktop = 8
        var out = ["[grouped-moe] gq_ref (per-pair, production structure) vs gq_grouped (per-expert, L1 reuse)",
                   "  E=\(E) Ktop=\(Ktop), synthetic weights, uniform-random routing (worst case for grouping)"]

        // shapes: g/u proj (K=2048→N=512, x per token) and d proj (K=512→N=2048, x per pair)
        for (label, K, N, lhsPer) in [("g/u 2048→512", 2048, 512, false), ("d   512→2048", 512, 2048, true)] {
            var rng = LCG(s: 42)
            // weights: [E, N, K/8] uint32 + scales/biases [E, N, K/64] f16 (small values, exactness-irrelevant)
            let wCount = E * N * K / 8
            let wBuf = device.makeBuffer(length: wCount * 4, options: .storageModeShared)!
            let wp = wBuf.contents().bindMemory(to: UInt32.self, capacity: wCount)
            for i in 0..<wCount { wp[i] = UInt32(truncatingIfNeeded: rng.next()) }
            let sCount = E * N * K / 64
            let sBuf = device.makeBuffer(length: sCount * 2, options: .storageModeShared)!
            let bBuf = device.makeBuffer(length: sCount * 2, options: .storageModeShared)!
            let sp = sBuf.contents().bindMemory(to: Float16.self, capacity: sCount)
            let bp = bBuf.contents().bindMemory(to: Float16.self, capacity: sCount)
            for i in 0..<sCount { sp[i] = Float16(Double(rng.next() % 1000) / 50000.0 + 0.001); bp[i] = Float16(Double(rng.next() % 1000) / 100000.0) }

            out.append("  ── \(label) (lhsPer=\(lhsPer ? 1 : 0)) ──")
            for M in [64, 256, 1024] {
                // routing: per token, 8 distinct experts of E
                var inds = [Int32](); inds.reserveCapacity(M * Ktop)
                for _ in 0..<M {
                    var seen = Set<Int32>()
                    while seen.count < Ktop { seen.insert(Int32(rng.next() % UInt64(E))) }
                    inds.append(contentsOf: seen.sorted())
                }
                // x rows: per token (g/u) or per pair (d)
                let xRows = lhsPer ? M * Ktop : M
                let xBuf = device.makeBuffer(length: xRows * K * 2, options: .storageModeShared)!
                let xp = xBuf.contents().bindMemory(to: Float16.self, capacity: xRows * K)
                for i in 0..<(xRows * K) { xp[i] = Float16(Double(rng.next() % 2000) / 1000.0 - 1.0) }
                let indsBuf = device.makeBuffer(bytes: inds, length: inds.count * 4, options: .storageModeShared)!
                // CSR groups: rows per expert
                var byExpert = [[Int32]](repeating: [], count: E)
                for (mk, e) in inds.enumerated() { byExpert[Int(e)].append(Int32(mk)) }
                var gExpert = [Int32](), gRowOff: [Int32] = [0], gRowIdx = [Int32]()
                for e in 0..<E where !byExpert[e].isEmpty {
                    gExpert.append(Int32(e)); gRowIdx.append(contentsOf: byExpert[e]); gRowOff.append(Int32(gRowIdx.count))
                }
                let G = gExpert.count
                let geBuf = device.makeBuffer(bytes: gExpert, length: G * 4, options: .storageModeShared)!
                let goBuf = device.makeBuffer(bytes: gRowOff, length: (G + 1) * 4, options: .storageModeShared)!
                let giBuf = device.makeBuffer(bytes: gRowIdx, length: gRowIdx.count * 4, options: .storageModeShared)!
                // v2 CSR: same row order, groups split at R_MAX=8 (register accumulators bound)
                var gExpert2 = [Int32](), gRowOff2: [Int32] = [0]
                for gi in 0..<G {
                    var lo = Int(gRowOff[gi])
                    let hi = Int(gRowOff[gi + 1])
                    while lo < hi {
                        let take = Swift.min(8, hi - lo)
                        gExpert2.append(gExpert[gi]); gRowOff2.append(Int32(lo + take)); lo += take
                    }
                }
                let G2 = gExpert2.count
                let geBuf2 = device.makeBuffer(bytes: gExpert2, length: G2 * 4, options: .storageModeShared)!
                let goBuf2 = device.makeBuffer(bytes: gRowOff2, length: (G2 + 1) * 4, options: .storageModeShared)!
                let yRef = device.makeBuffer(length: M * Ktop * N * 2, options: .storageModeShared)!
                let yGrp = device.makeBuffer(length: M * Ktop * N * 2, options: .storageModeShared)!
                let yGrp2 = device.makeBuffer(length: M * Ktop * N * 2, options: .storageModeShared)!

                var kk = Int32(K), nn = Int32(N), kt = Int32(Ktop), lp: UInt32 = lhsPer ? 1 : 0
                func encRef(_ enc: MTLComputeCommandEncoder) {
                    enc.setComputePipelineState(pipeRef!)
                    enc.setBuffer(wBuf, offset: 0, index: 0); enc.setBuffer(sBuf, offset: 0, index: 1); enc.setBuffer(bBuf, offset: 0, index: 2)
                    enc.setBuffer(xBuf, offset: 0, index: 3); enc.setBuffer(indsBuf, offset: 0, index: 4); enc.setBuffer(yRef, offset: 0, index: 5)
                    enc.setBytes(&kk, length: 4, index: 6); enc.setBytes(&nn, length: 4, index: 7); enc.setBytes(&kt, length: 4, index: 8)
                    enc.setBytes(&lp, length: 4, index: 9)
                    enc.dispatchThreadgroups(MTLSize(width: 1, height: N / 8, depth: M * Ktop), threadsPerThreadgroup: MTLSize(width: 32, height: 2, depth: 1))
                }
                func encGrp(_ enc: MTLComputeCommandEncoder) {
                    enc.setComputePipelineState(pipeGrp!)
                    enc.setBuffer(wBuf, offset: 0, index: 0); enc.setBuffer(sBuf, offset: 0, index: 1); enc.setBuffer(bBuf, offset: 0, index: 2)
                    enc.setBuffer(xBuf, offset: 0, index: 3); enc.setBuffer(geBuf, offset: 0, index: 4); enc.setBuffer(goBuf, offset: 0, index: 5)
                    enc.setBuffer(giBuf, offset: 0, index: 6); enc.setBuffer(yGrp, offset: 0, index: 7)
                    enc.setBytes(&kk, length: 4, index: 8); enc.setBytes(&nn, length: 4, index: 9); enc.setBytes(&kt, length: 4, index: 10)
                    enc.setBytes(&lp, length: 4, index: 11)
                    enc.dispatchThreadgroups(MTLSize(width: 1, height: N / 8, depth: G), threadsPerThreadgroup: MTLSize(width: 32, height: 2, depth: 1))
                }
                func encGrp2(_ enc: MTLComputeCommandEncoder) {
                    enc.setComputePipelineState(pipeGrp2!)
                    enc.setBuffer(wBuf, offset: 0, index: 0); enc.setBuffer(sBuf, offset: 0, index: 1); enc.setBuffer(bBuf, offset: 0, index: 2)
                    enc.setBuffer(xBuf, offset: 0, index: 3); enc.setBuffer(geBuf2, offset: 0, index: 4); enc.setBuffer(goBuf2, offset: 0, index: 5)
                    enc.setBuffer(giBuf, offset: 0, index: 6); enc.setBuffer(yGrp2, offset: 0, index: 7)
                    enc.setBytes(&kk, length: 4, index: 8); enc.setBytes(&nn, length: 4, index: 9); enc.setBytes(&kt, length: 4, index: 10)
                    enc.setBytes(&lp, length: 4, index: 11)
                    enc.dispatchThreadgroups(MTLSize(width: 1, height: N / 8, depth: G2), threadsPerThreadgroup: MTLSize(width: 32, height: 2, depth: 1))
                }
                func encNoDeq(_ enc: MTLComputeCommandEncoder) {
                    enc.setComputePipelineState(pipeNoDeq!)
                    enc.setBuffer(wBuf, offset: 0, index: 0); enc.setBuffer(sBuf, offset: 0, index: 1); enc.setBuffer(bBuf, offset: 0, index: 2)
                    enc.setBuffer(xBuf, offset: 0, index: 3); enc.setBuffer(indsBuf, offset: 0, index: 4); enc.setBuffer(yGrp2, offset: 0, index: 5)
                    enc.setBytes(&kk, length: 4, index: 6); enc.setBytes(&nn, length: 4, index: 7); enc.setBytes(&kt, length: 4, index: 8)
                    enc.setBytes(&lp, length: 4, index: 9)
                    enc.dispatchThreadgroups(MTLSize(width: 1, height: N / 8, depth: M * Ktop), threadsPerThreadgroup: MTLSize(width: 32, height: 2, depth: 1))
                }
                func gpuTime(_ iters: Int, _ enc1: (MTLComputeCommandEncoder) -> Void) -> Double {
                    let cb = queue.makeCommandBuffer()!
                    let enc = cb.makeComputeCommandEncoder()!
                    for _ in 0..<iters { enc1(enc) }
                    enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
                    return (cb.gpuEndTime - cb.gpuStartTime) * 1000.0 / Double(iters)
                }
                _ = gpuTime(2, encRef); _ = gpuTime(2, encGrp); _ = gpuTime(2, encGrp2)   // warmup
                let tR = gpuTime(10, encRef)
                let tG = gpuTime(10, encGrp)
                let tG2 = gpuTime(10, encGrp2)
                _ = gpuTime(2, encNoDeq)
                let tN = gpuTime(10, encNoDeq)
                // bit-exact: single clean dispatch each, byte compare
                _ = gpuTime(1, encRef); _ = gpuTime(1, encGrp); _ = gpuTime(1, encGrp2)
                let same = memcmp(yRef.contents(), yGrp.contents(), M * Ktop * N * 2) == 0
                let same2 = memcmp(yRef.contents(), yGrp2.contents(), M * Ktop * N * 2) == 0
                let rowsPerE = Double(M * Ktop) / Double(G)
                out.append(String(format: "  M=%4d (%4.1f r/e) ref %7.3f | grp %6.3f %4.2fx%@ | grp2 %6.3f %4.2fx%@ | noDeq %6.3f %4.2fx(timing-only)",
                                  M, rowsPerE, tR, tG, tR / tG, same ? "✅" : "❌",
                                  tG2, tR / tG2, same2 ? "✅" : "❌", tN, tR / tN))
            }
        }
        out.append("GROUPEDMOE done")
        return out.joined(separator: "\n")
    }

    // Probe B (matrix-unit-canonical route): DENSE qmm — production scalar qmv structure vs the
    // production qmm4_tiled structure (TG-shared dequant, M-invariant per locked test 10) at
    // PREFILL M on real dense shapes. Decides whether tiled-for-prefill buys the dense share.
    nonisolated(unsafe) static var pipeDenseRef: MTLComputePipelineState?
    nonisolated(unsafe) static var pipeDenseTiled: MTLComputePipelineState?
    static let denseSrc = """
    #include <metal_stdlib>
    using namespace metal;
    inline float ld16(const device half* x, thread float* xt) {
        float sum = 0.0f;
        for (int i = 0; i < 16; i += 4) {
            sum += x[i] + x[i+1] + x[i+2] + x[i+3];
            xt[i] = x[i]; xt[i+1] = x[i+1]/16.0f; xt[i+2] = x[i+2]/256.0f; xt[i+3] = x[i+3]/4096.0f;
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
    // production qmm4 structure batched over M rows: grid (1, N/8, M), tg per (row, n-tile)
    kernel void dq_ref(device const uint32_t* w [[buffer(0)]], device const half* scales [[buffer(1)]],
                       device const half* biases [[buffer(2)]], device const half* x [[buffer(3)]],
                       device half* y [[buffer(4)]],
                       constant int& in_vec_size [[buffer(5)]], constant int& out_vec_size [[buffer(6)]],
                       uint3 tid [[threadgroup_position_in_grid]],
                       uint simd_gid [[simdgroup_index_in_threadgroup]],
                       uint simd_lid [[thread_index_in_simdgroup]]) {
        constexpr int packs_per_thread = 2, results_per_simdgroup = 4;
        constexpr int bytes_per_pack = 4, values_per_thread = 16;
        constexpr int block_size = 512, scale_step_per_thread = 4;
        const device uint8_t* ws = (const device uint8_t*)w;
        typedef float U;
        thread U x_thread[16];
        thread U result[4] = {0};
        const int in_vec_size_w = in_vec_size / 2;
        const int in_vec_size_g = in_vec_size / 64;
        uint m = tid.z;
        const int out_row = tid.y * 8 + simd_gid * results_per_simdgroup;
        ws     += out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
        scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
        biases += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
        x += (size_t)m * in_vec_size + simd_lid * values_per_thread;
        y += (size_t)m * out_vec_size + out_row;
        for (int k = 0; k < in_vec_size; k += block_size) {
            U sum = ld16(x, x_thread);
            for (int row = 0; row < results_per_simdgroup; row++) {
                auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
                const device half* sl = scales + row * in_vec_size_g;
                const device half* bl = biases + row * in_vec_size_g;
                U s = sl[0]; U b = bl[0];
                result[row] += qd4(wl, x_thread, s, b, sum);
            }
            ws += block_size / 2;
            scales += block_size / 64; biases += block_size / 64; x += block_size;
        }
        for (int row = 0; row < results_per_simdgroup; row++) {
            result[row] = simd_sum(result[row]);
            if (simd_lid == 0) y[row] = (half)result[row];
        }
    }
    // production qmm4_tiled structure (verbatim): tg per output column, TG-shared dequant, M-loop
    kernel void dq_tiled(device const uint32_t* w [[buffer(0)]], device const half* scales [[buffer(1)]],
                         device const half* biases [[buffer(2)]], device const half* x [[buffer(3)]],
                         device half* y [[buffer(4)]], constant int& K [[buffer(5)]], constant int& N [[buffer(6)]],
                         constant int& M [[buffer(7)]],
                         uint n [[threadgroup_position_in_grid]], uint lid [[thread_position_in_threadgroup]],
                         uint tgs [[threads_per_threadgroup]]) {
        threadgroup float wdq[2048];
        threadgroup float red[256];
        int Kg = K / 64;
        for (int k = (int)lid; k < K; k += (int)tgs) {
            uint pack = w[n * (K/8) + k/8];
            uint nib = (pack >> (4*(k%8))) & 0xf;
            int g = k/64;
            wdq[k] = (float)scales[n*Kg+g] * (float)nib + (float)biases[n*Kg+g];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int m = 0; m < M; m++) {
            float acc = 0.0f;
            for (int k = (int)lid; k < K; k += (int)tgs) acc += (float)x[m*K+k] * wdq[k];
            red[lid] = acc; threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint s = tgs/2; s > 0; s >>= 1) { if (lid < s) red[lid] += red[lid+s]; threadgroup_barrier(mem_flags::mem_threadgroup); }
            if (lid == 0) y[m*N + n] = (half)red[0];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    """

    public static func denseBench() -> String {
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else { return "[dense-bench] no device\nDENSEBENCH done" }
        if pipeDenseRef == nil {
            do {
                let lib = try device.makeLibrary(source: denseSrc, options: nil)
                pipeDenseRef = try device.makeComputePipelineState(function: lib.makeFunction(name: "dq_ref")!)
                pipeDenseTiled = try device.makeComputePipelineState(function: lib.makeFunction(name: "dq_tiled")!)
            } catch { return "[dense-bench] compile: \(error)\nDENSEBENCH done" }
        }
        var out = ["[dense-bench] scalar qmv structure vs qmm4_tiled structure — dense prefill shapes (K<=2048)"]
        var rng = LCG(s: 7)
        // real dense shapes: GDN in_proj (2048->12352), attn qkv+gate (2048->~5120), MoE shared g/u (2048->512), d (512->2048)
        for (label, K, N) in [("GDN inproj 2048→12352", 2048, 12352), ("attn qkv  2048→5120", 2048, 5120), ("shared g/u 2048→512", 2048, 512), ("d/out      512→2048", 512, 2048)] {
            let wCount = N * K / 8
            let wBuf = device.makeBuffer(length: wCount * 4, options: .storageModeShared)!
            let wp = wBuf.contents().bindMemory(to: UInt32.self, capacity: wCount)
            for i in 0..<wCount { wp[i] = UInt32(truncatingIfNeeded: rng.next()) }
            let sCount = N * K / 64
            let sBuf = device.makeBuffer(length: sCount * 2, options: .storageModeShared)!
            let bBuf = device.makeBuffer(length: sCount * 2, options: .storageModeShared)!
            let sp = sBuf.contents().bindMemory(to: Float16.self, capacity: sCount)
            let bp = bBuf.contents().bindMemory(to: Float16.self, capacity: sCount)
            for i in 0..<sCount { sp[i] = Float16(Double(rng.next() % 1000) / 50000.0 + 0.001); bp[i] = Float16(Double(rng.next() % 1000) / 100000.0) }
            var line = "  \(label): "
            for M in [64, 256, 1024] {
                let xBuf = device.makeBuffer(length: M * K * 2, options: .storageModeShared)!
                let xp = xBuf.contents().bindMemory(to: Float16.self, capacity: M * K)
                for i in 0..<(M * K) { xp[i] = Float16(Double(rng.next() % 2000) / 1000.0 - 1.0) }
                let yBuf = device.makeBuffer(length: M * N * 2, options: .storageModeShared)!
                var kk = Int32(K), nn = Int32(N), mm = Int32(M)
                func encR(_ enc: MTLComputeCommandEncoder) {
                    enc.setComputePipelineState(pipeDenseRef!)
                    enc.setBuffer(wBuf, offset: 0, index: 0); enc.setBuffer(sBuf, offset: 0, index: 1); enc.setBuffer(bBuf, offset: 0, index: 2)
                    enc.setBuffer(xBuf, offset: 0, index: 3); enc.setBuffer(yBuf, offset: 0, index: 4)
                    enc.setBytes(&kk, length: 4, index: 5); enc.setBytes(&nn, length: 4, index: 6)
                    enc.dispatchThreadgroups(MTLSize(width: 1, height: N / 8, depth: M), threadsPerThreadgroup: MTLSize(width: 32, height: 2, depth: 1))
                }
                func encT(_ enc: MTLComputeCommandEncoder) {
                    enc.setComputePipelineState(pipeDenseTiled!)
                    enc.setBuffer(wBuf, offset: 0, index: 0); enc.setBuffer(sBuf, offset: 0, index: 1); enc.setBuffer(bBuf, offset: 0, index: 2)
                    enc.setBuffer(xBuf, offset: 0, index: 3); enc.setBuffer(yBuf, offset: 0, index: 4)
                    enc.setBytes(&kk, length: 4, index: 5); enc.setBytes(&nn, length: 4, index: 6); enc.setBytes(&mm, length: 4, index: 7)
                    enc.dispatchThreadgroups(MTLSize(width: N, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                }
                func t(_ e: (MTLComputeCommandEncoder) -> Void) -> Double {
                    let cb = queue.makeCommandBuffer()!; let enc = cb.makeComputeCommandEncoder()!
                    for _ in 0..<8 { e(enc) }
                    enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
                    return (cb.gpuEndTime - cb.gpuStartTime) * 1000.0 / 8.0
                }
                _ = t(encR); _ = t(encT)
                let tR = t(encR), tT = t(encT)
                line += String(format: "M%d %.2f/%.2fms %.2fx  ", M, tR, tT, tR / tT)
            }
            out.append(line)
        }
        out.append("DENSEBENCH done")
        return out.joined(separator: "\n")
    }

    // steel-route-bench: (1) MLX steel dense speed at prefill shapes (vs recorded scalar qmv
    // baselines from dense-tiled-bench), (2) steel GATHER via CSR tiling — rows sorted by expert
    // into fixed R=16 tiles (pad w/ zeros) so the inner matmul is M=16 => gather_qmm_t (matrix
    // units) instead of production-shape gather_qmv (inner M=1), (3) tile-composition invariance
    // (a row's output must not depend on which rows/pads share its tile), (4) MLX eval launch
    // overhead (the raw-CB<->MLX sync cost per hybrid boundary).
    public static func steelRouteBench() -> String {
        var out = ["[steel-route] MLX steel dense + CSR-tiled steel gather"]
        var rng = LCG(s: 99)
        func randX(_ rows: Int, _ K: Int) -> MLXArray {
            let a = (0..<(rows*K)).map { _ in Float16(Double(rng.next() % 2000) / 1000.0 - 1.0) }
            return MLXArray(a, [rows, K])
        }
        func timeEval(_ iters: Int, _ f: () -> MLXArray) -> Double {
            var best = Double.infinity
            for _ in 0..<2 {
                let t0 = Date()
                for _ in 0..<iters { let y = f(); y.eval() }
                best = Swift.min(best, Date().timeIntervalSince(t0) * 1000.0 / Double(iters))
            }
            return best
        }
        // ── (1) dense steel vs recorded scalar baselines ──
        let scalarMs: [String: [Int: Double]] = [
            "inproj 2048→12352": [64: 1.88, 256: 7.51, 1024: 30.03],
            "qkv    2048→5120":  [64: 0.77, 256: 3.15, 1024: 12.39],
            "shared 2048→512":   [64: 0.08, 256: 0.31, 1024: 1.24],
            "d/out   512→2048":  [64: 0.11, 256: 0.42, 1024: 1.67]]
        for (label, K, N) in [("inproj 2048→12352", 2048, 12352), ("qkv    2048→5120", 2048, 5120),
                              ("shared 2048→512", 2048, 512), ("d/out   512→2048", 512, 2048)] {
            let wf = MLXRandom.normal([N, K]).asType(.float16)
            let (wq, sc, biOpt) = MLX.quantized(wf, groupSize: 64, bits: 4, mode: .affine)
            guard let bi = biOpt else { return "[steel-route] biases nil" }
            MLX.eval([wq, sc, bi])
            var line = "  \(label): "
            for M in [64, 256, 1024] {
                let x = randX(M, K); x.eval()
                _ = timeEval(3) { MLX.quantizedMatmul(x, wq, scales: sc, biases: bi, transpose: true, groupSize: 64, bits: 4) }
                let t = timeEval(10) { MLX.quantizedMatmul(x, wq, scales: sc, biases: bi, transpose: true, groupSize: 64, bits: 4) }
                let base = scalarMs[label]?[M] ?? 0
                line += String(format: "M%d %.2fms(%.1fx)  ", M, t, base / t)
            }
            out.append(line)
        }
        // ── (2)(3) steel gather via CSR R=16 tiles ──
        let E = 256, Ktop = 8, K = 2048, N = 512, R = 16
        let wf = MLXRandom.normal([E, N, K]).asType(.float16)
        let (wq, sc, biOpt) = MLX.quantized(wf, groupSize: 64, bits: 4, mode: .affine)
        guard let bi = biOpt else { return "[steel-route] gw nil" }
        MLX.eval([wq, sc, bi])
        out.append("  ── gather K=2048→N=512, E=256, top-8, R=16 tiles ──")
        for T in [64, 256, 1024] {
            var inds = [Int32]()
            for _ in 0..<T { var s = Set<Int32>(); while s.count < Ktop { s.insert(Int32(rng.next() % UInt64(E))) }; inds.append(contentsOf: s.sorted()) }
            let x = randX(T, K); x.eval()
            let indsArr = MLXArray(inds, [T, Ktop])
            // production shape: x[T,1,1,K], rhsIndices [T,Ktop] → inner M=1 gather_qmv
            let xe = x.expandedDimensions(axes: [-2, -3])
            _ = timeEval(3) { MLX.gatherQuantizedMatmul(xe, wq, scales: sc, biases: bi, rhsIndices: indsArr, transpose: true, groupSize: 64, bits: 4, mode: .affine, sortedIndices: false) }
            let tProd = timeEval(10) { MLX.gatherQuantizedMatmul(xe, wq, scales: sc, biases: bi, rhsIndices: indsArr, transpose: true, groupSize: 64, bits: 4, mode: .affine, sortedIndices: false) }
            // CSR tiles: rows (pair index mk uses x row mk/Ktop) grouped by expert, fixed R=16, zero-pad
            var byE = [[Int32]](repeating: [], count: E)
            for (mk, e) in inds.enumerated() { byE[Int(e)].append(Int32(mk)) }
            var tileExpert = [Int32](), tileRows = [[Int32]]()   // each tile: expert + up-to-16 mk
            for e in 0..<E {
                var lo = 0
                while lo < byE[e].count { tileExpert.append(Int32(e)); tileRows.append(Array(byE[e][lo ..< Swift.min(lo+R, byE[e].count)])); lo += R }
            }
            let Gt = tileExpert.count
            let xh = x.asArray(Float16.self)
            var xg = [Float16](repeating: 0, count: Gt * R * K)
            for (ti, rows) in tileRows.enumerated() {
                for (ri, mk) in rows.enumerated() {
                    let src = Int(mk) / Ktop * K
                    for k in 0..<K { xg[(ti * R + ri) * K + k] = xh[src + k] }
                }
            }
            let xgArr = MLXArray(xg, [Gt, R, K]); xgArr.eval()
            let teArr = MLXArray(tileExpert, [Gt])
            _ = timeEval(3) { MLX.gatherQuantizedMatmul(xgArr, wq, scales: sc, biases: bi, rhsIndices: teArr, transpose: true, groupSize: 64, bits: 4, mode: .affine, sortedIndices: true) }
            let tSteel = timeEval(10) { MLX.gatherQuantizedMatmul(xgArr, wq, scales: sc, biases: bi, rhsIndices: teArr, transpose: true, groupSize: 64, bits: 4, mode: .affine, sortedIndices: true) }
            // VALUE correctness: steel-CSR re-scattered vs production-shape gather (different kernels
            // → not bit-equal; rel err ~1e-3 = same semantics, large = wrong rhsIndices broadcast)
            var relV = -1.0
            do {
                let yP = MLX.gatherQuantizedMatmul(xe, wq, scales: sc, biases: bi, rhsIndices: indsArr, transpose: true, groupSize: 64, bits: 4, mode: .affine, sortedIndices: false)
                let yS = MLX.gatherQuantizedMatmul(xgArr, wq, scales: sc, biases: bi, rhsIndices: teArr, transpose: true, groupSize: 64, bits: 4, mode: .affine, sortedIndices: true)
                MLX.eval([yP, yS])
                let ap = yP.reshaped([-1]).asArray(Float16.self)          // [T,Ktop,1,N] flat = mk-major
                let as_ = yS.reshaped([-1]).asArray(Float16.self)         // [Gt,R,N] flat = tile-major
                var pos = [Int: Int]()
                for (ti, rows) in tileRows.enumerated() { for (ri, mk) in rows.enumerated() { pos[Int(mk)] = ti * R + ri } }
                var nv = 0.0, dv = 0.0
                for mk in 0..<(T * Ktop) {
                    let o1 = mk * N, o2 = pos[mk]! * N
                    for j in 0..<N { nv += abs(Double(ap[o1+j]) - Double(as_[o2+j])); dv += abs(Double(ap[o1+j])) }
                }
                relV = nv / Swift.max(dv, 1e-9)
            }
            // FULL-CHAIN value check: g→u→swiglu→d in steel-CSR tile space vs production per-pair shapes.
            // Isolates the d-shape (K=512→N=2048) steel gather semantics, never value-verified before.
            var relChain = -1.0
            do {
                let wf2 = MLXRandom.normal([E, K, N]).asType(.float16)     // "down": [E, H=K? no: E, out=K(2048), in=N(512)]
                let (dq, ds, dbO) = MLX.quantized(wf2, groupSize: 64, bits: 4, mode: .affine)
                let db = dbO!
                // production: per-pair
                let gP = MLX.gatherQuantizedMatmul(xe, wq, scales: sc, biases: bi, rhsIndices: indsArr, transpose: true, groupSize: 64, bits: 4, mode: .affine, sortedIndices: false)
                let gPs = gP * Float16(0.01)   // scale down: synthetic weights overflow f16 in silu(g)*g
                let hP = (gPs * MLX.sigmoid(gPs)) * gPs                     // [T,Ktop,1,N]
                let hPr = hP.reshaped([T * Ktop, 1, 1, N])
                let indsFlat = indsArr.reshaped([T * Ktop, 1])
                let dP = MLX.gatherQuantizedMatmul(hPr, dq, scales: ds, biases: db, rhsIndices: indsFlat, transpose: true, groupSize: 64, bits: 4, mode: .affine, sortedIndices: false)
                // steel-CSR: tile space
                let gS = MLX.gatherQuantizedMatmul(xgArr, wq, scales: sc, biases: bi, rhsIndices: teArr, transpose: true, groupSize: 64, bits: 4, mode: .affine, sortedIndices: true)
                let gSs = gS * Float16(0.01)
                let hS = (gSs * MLX.sigmoid(gSs)) * gSs                     // matches hP transform
                let dS = MLX.gatherQuantizedMatmul(hS, dq, scales: ds, biases: db, rhsIndices: teArr, transpose: true, groupSize: 64, bits: 4, mode: .affine, sortedIndices: true)
                MLX.eval([dP, dS])
                let a = dP.reshaped([-1]).asArray(Float16.self)             // mk-major [T*Ktop, K]
                let b = dS.reshaped([-1]).asArray(Float16.self)             // tile-major [Gt,R,K]
                var pos = [Int: Int]()
                for (ti, rows) in tileRows.enumerated() { for (ri, mk) in rows.enumerated() { pos[Int(mk)] = ti * R + ri } }
                var nv = 0.0, dv = 0.0
                let NN = K   // down output dim = K (2048) in this synthetic setup
                for mk in 0..<(T * Ktop) {
                    let o1 = mk * NN, o2 = pos[mk]! * NN
                    for j in 0..<NN { nv += abs(Double(a[o1+j]) - Double(b[o2+j])); dv += abs(Double(a[o1+j])) }
                }
                relChain = nv / Swift.max(dv, 1e-9)
            }
            let padFactor = Double(Gt * R) / Double(T * Ktop)
            out.append(String(format: "  T=%4d  prod(qmv) %7.3fms | steel-CSR %7.3fms  %5.2fx  (tiles=%d pad=%.2fx)  valErr %.2e %@  chainErr %.2e %@",
                              T, tProd, tSteel, tProd / tSteel, Gt, padFactor, relV, relV < 5e-3 ? "✅" : "SEMANTICS❌", relChain, relChain < 5e-3 ? "✅" : "CHAIN❌"))
            // (3) tile-composition invariance at this T: recompute with a DIFFERENT composition
            // (shuffle rows into tiles of the same expert differently: reverse order within expert)
            if T == 256 {
                var tileRows2 = [[Int32]](), tileExpert2 = [Int32]()
                for e in 0..<E {
                    let rev = byE[e].reversed().map { $0 }
                    var lo = 0
                    while lo < rev.count { tileExpert2.append(Int32(e)); tileRows2.append(Array(rev[lo ..< Swift.min(lo+R, rev.count)])); lo += R }
                }
                var xg2 = [Float16](repeating: 0, count: tileExpert2.count * R * K)
                for (ti, rows) in tileRows2.enumerated() {
                    for (ri, mk) in rows.enumerated() {
                        let src = Int(mk) / Ktop * K
                        for k in 0..<K { xg2[(ti * R + ri) * K + k] = xh[src + k] }
                    }
                }
                let xg2Arr = MLXArray(xg2, [tileExpert2.count, R, K])
                let y1 = MLX.gatherQuantizedMatmul(xgArr, wq, scales: sc, biases: bi, rhsIndices: teArr, transpose: true, groupSize: 64, bits: 4, mode: .affine, sortedIndices: true)
                let y2 = MLX.gatherQuantizedMatmul(xg2Arr, wq, scales: sc, biases: bi, rhsIndices: MLXArray(tileExpert2, [tileExpert2.count]), transpose: true, groupSize: 64, bits: 4, mode: .affine, sortedIndices: true)
                MLX.eval([y1, y2])
                // compare per-mk outputs across the two compositions
                let a1 = y1.asArray(Float16.self), a2 = y2.asArray(Float16.self)
                var ok = true
                var pos1 = [Int32: Int](), pos2 = [Int32: Int]()
                for (ti, rows) in tileRows.enumerated() { for (ri, mk) in rows.enumerated() { pos1[mk] = ti * R + ri } }
                for (ti, rows) in tileRows2.enumerated() { for (ri, mk) in rows.enumerated() { pos2[mk] = ti * R + ri } }
                outer: for mk in pos1.keys {
                    let o1 = pos1[mk]! * N, o2 = pos2[mk]! * N
                    for j in 0..<N where a1[o1 + j] != a2[o2 + j] { ok = false; break outer }
                }
                out.append("    tile-composition invariance @T=256: \(ok ? "IDENTICAL ✅" : "DIFFER ❌")")
            }
        }
        // ── (4) MLX eval launch overhead (per hybrid boundary) ──
        let wSmall = MLXRandom.normal([512, 512]).asType(.float16)
        let (wq2, sc2, bi2o) = MLX.quantized(wSmall, groupSize: 64, bits: 4, mode: .affine)
        let xs = randX(16, 512)
        MLX.eval([wq2, sc2, bi2o!, xs])
        let tiny = timeEval(50) { MLX.quantizedMatmul(xs, wq2, scales: sc2, biases: bi2o!, transpose: true, groupSize: 64, bits: 4) }
        out.append(String(format: "  eval launch overhead (tiny qmm M=16): %.3fms/boundary → ~%.0fms/chunk at 80 boundaries", tiny, tiny * 80))
        out.append("STEELROUTE done")
        return out.joined(separator: "\n")
    }
}
