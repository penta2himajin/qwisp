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
}
