import Foundation
import Metal
import MLX

// GPU speculative-sampling kernel (Option B "本速度化"). The greedy path reads back only the
// argmax index; sampling instead needs a categorical draw + the accept test per verify row.
// Doing that on the GPU (Gumbel-max — argmax of logit/T + Gumbel noise, no softmax normalization,
// no sort) keeps the readback tiny (a few ints per row) and lifts the CPU-readback maxK cap.
//
// Per row m the kernel returns: full categorical sample, residual sample (excluding the draft
// token), and the speculative-sampling accept flag (coin < p(draft)). The CPU loop then walks the
// accept flags to pick the accepted prefix + next token — exactly like the CPU sample loop, but
// with the per-vocab work on the GPU. At temperature 0 (invT ≤ 0) it degenerates to argmax with
// the same first-max tie-break as `argmax_rows`, so the T=0 stream stays byte-identical to greedy.
//
// v1 scope: temperature only (no penalties/logit_bias/top_p in the kernel — those stay on the
// optimized CPU path). `useAdj`/`logitAdj` are wired for a future penalties-on-GPU follow-up.
public enum SamplerGPU {
    nonisolated(unsafe) static var _pipeline: MTLComputePipelineState?

    static let kernelSrc = """
    #include <metal_stdlib>
    using namespace metal;

    inline uint qhash(uint a, uint b, uint c) {
        uint h = a * 0x9E3779B1u + 0x165667B1u;
        h ^= b + 0x9E3779B9u + (h << 6) + (h >> 2);
        h ^= c + 0x9E3779B9u + (h << 6) + (h >> 2);
        h ^= h >> 15; h *= 0x2C1B3C6Du; h ^= h >> 12; h *= 0x297A2D39u; h ^= h >> 15;
        return h;
    }
    inline float u01(uint h) { return fmax((float)(h >> 8) * (1.0f / 16777216.0f), 1e-7f); }

    // One threadgroup per verify row m; threads stride over the vocab V.
    kernel void spec_sample_rows(
        device const half*  logits    [[buffer(0)]],   // [M, V]
        device const float* logitAdj  [[buffer(1)]],   // [V] penalties+bias (read iff useAdj)
        device const int*   draftToks [[buffer(2)]],   // [M] draft per row, -1 = none/bonus
        device int*         sampFull  [[buffer(3)]],   // [M] categorical sample
        device int*         sampResid [[buffer(4)]],   // [M] categorical excluding draft
        device int*         acceptOut [[buffer(5)]],   // [M] 1 if draft accepted
        constant float&     invT      [[buffer(6)]],   // 1/T; <=0 → greedy (argmax)
        constant uint&      V         [[buffer(7)]],
        constant uint&      seedLo    [[buffer(8)]],
        constant uint&      seedHi    [[buffer(9)]],
        constant uint&      basePos   [[buffer(10)]],
        constant uint&      useAdj    [[buffer(11)]],
        uint m   [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]],
        uint tgs [[threads_per_threadgroup]])
    {
        threadgroup float rMax[256];
        threadgroup float rFS[256]; threadgroup int rFI[256];
        threadgroup float rRS[256]; threadgroup int rRI[256];
        threadgroup float rZ[256];

        device const half* row = logits + (size_t)m * V;
        int d = draftToks[m];
        bool greedy = invT <= 0.0f;
        uint posSalt = seedHi ^ (basePos + m);

        float locMax = -INFINITY;
        float locFS = -INFINITY; int locFI = 0x7fffffff;   // Gumbel argmax (full)
        float locRS = -INFINITY; int locRI = 0x7fffffff;   // Gumbel argmax (excl draft)
        for (uint v = tid; v < V; v += tgs) {
            float L = (float)row[v] + (useAdj ? logitAdj[v] : 0.0f);
            if (L > locMax) locMax = L;
            float score;
            if (greedy) { score = L; }
            else { score = L * invT + (-log(-log(u01(qhash(seedLo, posSalt, v))))); }
            if (score > locFS || (score == locFS && (int)v < locFI)) { locFS = score; locFI = (int)v; }
            if ((int)v != d && (score > locRS || (score == locRS && (int)v < locRI))) { locRS = score; locRI = (int)v; }
        }
        rMax[tid] = locMax; rFS[tid] = locFS; rFI[tid] = locFI; rRS[tid] = locRS; rRI[tid] = locRI;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = tgs / 2; s > 0; s >>= 1) {
            if (tid < s) {
                if (rMax[tid+s] > rMax[tid]) rMax[tid] = rMax[tid+s];
                if (rFS[tid+s] > rFS[tid] || (rFS[tid+s]==rFS[tid] && rFI[tid+s]<rFI[tid])) { rFS[tid]=rFS[tid+s]; rFI[tid]=rFI[tid+s]; }
                if (rRS[tid+s] > rRS[tid] || (rRS[tid+s]==rRS[tid] && rRI[tid+s]<rRI[tid])) { rRS[tid]=rRS[tid+s]; rRI[tid]=rRI[tid+s]; }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        float gmax = rMax[0];
        int fullIdx = rFI[0], residIdx = rRI[0];

        // pass 2: softmax normalizer Z (only needed for the accept probability at T>0).
        float locZ = 0.0f;
        if (!greedy) {
            for (uint v = tid; v < V; v += tgs) {
                float L = (float)row[v] + (useAdj ? logitAdj[v] : 0.0f);
                locZ += exp((L - gmax) * invT);
            }
        }
        rZ[tid] = locZ;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = tgs / 2; s > 0; s >>= 1) { if (tid < s) rZ[tid] += rZ[tid+s]; threadgroup_barrier(mem_flags::mem_threadgroup); }

        if (tid == 0) {
            sampFull[m]  = fullIdx;
            sampResid[m] = (d < 0) ? fullIdx : residIdx;
            float pDraft;
            if (d < 0) pDraft = 0.0f;
            else if (greedy) pDraft = (d == fullIdx) ? 1.0f : 0.0f;
            else { float Ld = (float)row[d] + (useAdj ? logitAdj[d] : 0.0f); pDraft = exp((Ld - gmax) * invT) / rZ[0]; }
            float coin = u01(qhash(seedLo, posSalt, 0x9999AAAAu));
            acceptOut[m] = (coin < pDraft) ? 1 : 0;
        }
    }
    """

    @discardableResult
    public static func ensurePipeline(_ device: MTLDevice) -> Bool {
        if _pipeline != nil { return true }
        do {
            let lib = try device.makeLibrary(source: kernelSrc, options: nil)
            _pipeline = try device.makeComputePipelineState(function: lib.makeFunction(name: "spec_sample_rows")!)
            return true
        } catch { return false }
    }

    /// Standalone dispatch on caller-supplied logits (for the distribution test; not the decode path).
    /// Returns (full, resid, accept) per row.
    public static func sampleRows(device: MTLDevice, queue: MTLCommandQueue,
                                  logits: [[Float]], drafts: [Int], invT: Float,
                                  seed: UInt64, basePos: Int) -> (full: [Int], resid: [Int], accept: [Bool])? {
        guard ensurePipeline(device), let pipe = _pipeline, let first = logits.first else { return nil }
        let M = logits.count, V = first.count
        var flat = [UInt16](repeating: 0, count: M * V)   // f16
        for m in 0..<M { for v in 0..<V { flat[m*V + v] = f32to16(logits[m][v]) } }
        guard let lgBuf = device.makeBuffer(length: M*V*2, options: .storageModeShared),
              let adjBuf = device.makeBuffer(length: max(1,V)*4, options: .storageModeShared),
              let dBuf = device.makeBuffer(length: M*4, options: .storageModeShared),
              let fBuf = device.makeBuffer(length: M*4, options: .storageModeShared),
              let rBuf = device.makeBuffer(length: M*4, options: .storageModeShared),
              let aBuf = device.makeBuffer(length: M*4, options: .storageModeShared) else { return nil }
        flat.withUnsafeBytes { lgBuf.contents().copyMemory(from: $0.baseAddress!, byteCount: M*V*2) }
        let dp = dBuf.contents().bindMemory(to: Int32.self, capacity: M)
        for m in 0..<M { dp[m] = Int32(m < drafts.count ? drafts[m] : -1) }

        let cb = queue.makeCommandBuffer()!, enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pipe)
        enc.setBuffer(lgBuf, offset: 0, index: 0); enc.setBuffer(adjBuf, offset: 0, index: 1)
        enc.setBuffer(dBuf, offset: 0, index: 2); enc.setBuffer(fBuf, offset: 0, index: 3)
        enc.setBuffer(rBuf, offset: 0, index: 4); enc.setBuffer(aBuf, offset: 0, index: 5)
        var it = invT, vv = UInt32(V), sl = UInt32(truncatingIfNeeded: seed), sh = UInt32(truncatingIfNeeded: seed >> 32)
        var bp = UInt32(basePos), ua = UInt32(0)
        enc.setBytes(&it, length: 4, index: 6); enc.setBytes(&vv, length: 4, index: 7)
        enc.setBytes(&sl, length: 4, index: 8); enc.setBytes(&sh, length: 4, index: 9)
        enc.setBytes(&bp, length: 4, index: 10); enc.setBytes(&ua, length: 4, index: 11)
        enc.dispatchThreadgroups(MTLSize(width: M, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()

        let fp = fBuf.contents().bindMemory(to: Int32.self, capacity: M)
        let rp = rBuf.contents().bindMemory(to: Int32.self, capacity: M)
        let ap = aBuf.contents().bindMemory(to: Int32.self, capacity: M)
        return ((0..<M).map { Int(fp[$0]) }, (0..<M).map { Int(rp[$0]) }, (0..<M).map { ap[$0] != 0 })
    }

    // minimal f32→f16 bit conversion (round-to-nearest-even not required for test logits).
    static func f32to16(_ f: Float) -> UInt16 {
        let bits = f.bitPattern
        let sign = UInt16((bits >> 16) & 0x8000)
        let exp = Int((bits >> 23) & 0xFF) - 127 + 15
        let mant = bits & 0x7FFFFF
        if exp <= 0 { return sign }
        if exp >= 0x1F { return sign | 0x7C00 }
        return sign | UInt16(exp << 10) | UInt16(mant >> 13)
    }

    /// Distribution test: sample a fixed synthetic distribution many times on the GPU and compare
    /// the empirical histogram to the analytic softmax (and to the CPU sampler). PASS on small TV.
    public static func distributionSelfCheck() -> (passed: Int, total: Int, log: [String]) {
        var passed = 0, total = 0; var log: [String] = []
        func ok(_ n: String, _ c: Bool, _ extra: String = "") {
            total += 1; if c { passed += 1 }; log.append("[gpusample-test] \(n): \(c ? "PASS" : "FAIL")\(extra.isEmpty ? "" : " (\(extra))")")
        }
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
            return (0, 1, ["[gpusample-test] no_metal_device: FAIL"])
        }
        let logits: [Float] = [3.0, 2.0, 1.0, 0.5, 0.0, -1.0, -2.0, -4.0]
        let T: Float = 1.0
        let analytic = Sampler.probs(logits: logits, temperature: Double(T), topP: 1.0)
        let N = 40000
        var histGPU = [Int](repeating: 0, count: logits.count)
        for i in 0..<N {
            if let r = sampleRows(device: device, queue: queue, logits: [logits], drafts: [-1],
                                  invT: 1.0 / T, seed: UInt64(i) &* 0x9E3779B97F4A7C15, basePos: i) {
                histGPU[r.full[0]] += 1
            }
        }
        var tvGPU = 0.0
        for i in 0..<logits.count { tvGPU += abs(Double(histGPU[i]) / Double(N) - Double(analytic[i])) }
        tvGPU *= 0.5
        ok("gpu_matches_softmax", tvGPU < 0.02, String(format: "TV=%.4f", tvGPU))

        // T=0 (greedy): GPU full-sample must be the argmax regardless of seed.
        let amax = analytic.firstIndex(of: analytic.max()!)!
        var greedyOK = true
        for i in 0..<50 {
            if let r = sampleRows(device: device, queue: queue, logits: [logits], drafts: [-1],
                                  invT: -1.0, seed: UInt64(i), basePos: i) { if r.full[0] != amax { greedyOK = false } }
        }
        ok("gpu_temp0_is_argmax", greedyOK)

        // accept flag calibration: for draft == a mid token, accept rate ≈ p(draft) over seeds.
        let draft = 2
        var acc = 0
        for i in 0..<N {
            if let r = sampleRows(device: device, queue: queue, logits: [logits], drafts: [draft],
                                  invT: 1.0 / T, seed: UInt64(i) &* 0xD1B54A32D192ED03, basePos: i) {
                if r.accept[0] { acc += 1 }
            }
        }
        let accRate = Double(acc) / Double(N)
        ok("accept_rate_matches_p", abs(accRate - Double(analytic[draft])) < 0.02,
           String(format: "acc=%.3f p=%.3f", accRate, analytic[draft]))

        return (passed, total, log)
    }
}
