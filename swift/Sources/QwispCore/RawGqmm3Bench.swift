import Foundation
import Metal
import MLX
import MLXRandom

/// gqmm3-bench: kernel-level throughput of the 3-bit gather-qmv (gqmm3) vs the 4-bit (gqmm4),
/// GPU-timestamped single-CB (per-call CB/alloc overhead excluded, like gatherBench).
///
/// gqmm3 is not yet wired e2e (Stage 2-4 pending), so this is the ONLY meaningful speed number:
/// the pure per-gather kernel time at the real MoE expert shapes.
///
/// Two comparisons matter:
///   (A) gqmm3-f16 vs gqmm4-f16  — isolates the bit-width effect (3-bit reads 3/8 vs 4/8 bytes/val).
///   (B) gqmm3-f32 vs gqmm4-f16  — the REAL deployment path: the UD model's scales are BF16, so
///       MLX promotes gqmm3 to f32 (2x activation/out bytes + f32 ALU), while gqmm4 (4-bit shared
///       expert) stays f16. This shows whether the 3-bit weight saving survives the f32 tax.
public enum RawGqmm3Bench {
    public static func run() -> String {
        guard let dev = RawMetalForward.ensure() else { return "[gqmm3-bench] no device" }
        let (device, queue) = dev
        let E = 64, Ktop = 8, reps = 400
        var out = "[gqmm3-bench] GPU-timestamped, M=1 Ktop=\(Ktop) E=\(E) reps=\(reps) (per-gather µs, 純kernel)\n"
        out += "  weight bytes/gather: 3bit = Ktop·N·K·3/8 ; 4bit = Ktop·N·K/2 (3bit=0.75×)\n"

        // shapes: gate/up (K=2048,N=512) and down (K=512,N=2048) — the two real expert projections.
        for (tag, K, N) in [("gate/up K=2048 N=512", 2048, 512), ("down K=512 N=2048", 512, 2048)] {
            out += "  ── \(tag) ──\n"
            // compile both pipelines by calling the public wrappers once (f16 and f32 for gqmm3).
            let wq3 = MLXRandom.randInt(0 ..< 255, [E, N, K * 3 / 32]).asType(.uint32)
            let wq4 = MLXRandom.randInt(0 ..< 255, [E, N, K / 8]).asType(.uint32)
            let scF16 = (MLXRandom.normal([E, N, K / 64]) * 0.02).asType(.float16)
            let biF16 = (MLXRandom.normal([E, N, K / 64]) * 0.02).asType(.float16)
            let scBF16 = scF16.asType(.bfloat16)
            let biBF16 = biF16.asType(.bfloat16)
            let xF16 = (MLXRandom.normal([1, K]) * 0.1).asType(.float16)
            let inds = MLXArray((0 ..< Ktop).map { Int32($0) })
            MLX.eval([wq3, wq4, scF16, biF16, scBF16, biBF16, xF16, inds])

            // warm/compile: gqmm3 f16 (scales f16), gqmm3 f32 (scales bf16→promote f32), gqmm4 f16
            _ = RawMetalForward.gqmm3Rows(xF16, wq3, scales: scF16, biases: biF16, inds: inds, M: 1, Ktop: Ktop, K: K, N: N)
            _ = RawMetalForward.gqmm3Rows(xF16, wq3, scales: scBF16, biases: biBF16, inds: inds, M: 1, Ktop: Ktop, K: K, N: N)
            _ = RawMetalForward.gatherQmmRows(xF16, wq4, scales: scF16, biases: biF16, inds: inds, M: 1, Ktop: Ktop, K: K, N: N)

            func timeKernel(_ pipe: MTLComputePipelineState?, wq: MLXArray, sc: MLXArray, bi: MLXArray, dt: DType, elem: Int) -> Double? {
                guard let pipe = pipe else { return nil }
                guard let bwq = wq.asMTLBuffer(device: device, noCopy: true),
                      let bsc = sc.asType(dt).asMTLBuffer(device: device, noCopy: true),
                      let bbi = bi.asType(dt).asMTLBuffer(device: device, noCopy: true),
                      let bx = xF16.asType(dt).asMTLBuffer(device: device, noCopy: true) else { return nil }
                var indsA = (0 ..< Ktop).map { Int32($0) }
                let bin = device.makeBuffer(bytes: &indsA, length: Ktop * 4, options: .storageModeShared)!
                let outBuf = device.makeBuffer(length: Ktop * N * elem, options: .storageModeShared)!
                let zero: [Int32] = [0]; let bstop = device.makeBuffer(bytes: zero, length: 4, options: .storageModeShared)!
                var kk = Int32(K), nn = Int32(N), kt = Int32(Ktop), lp = UInt32(0)
                func runCB(_ count: Int) -> Double {
                    let cb = queue.makeCommandBuffer()!; let enc = cb.makeComputeCommandEncoder()!
                    for _ in 0 ..< count {
                        enc.setComputePipelineState(pipe)
                        enc.setBuffer(bwq, offset: 0, index: 0); enc.setBuffer(bsc, offset: 0, index: 1)
                        enc.setBuffer(bbi, offset: 0, index: 2); enc.setBuffer(bx, offset: 0, index: 3)
                        enc.setBuffer(bin, offset: 0, index: 4); enc.setBuffer(outBuf, offset: 0, index: 5)
                        enc.setBytes(&kk, length: 4, index: 6); enc.setBytes(&nn, length: 4, index: 7); enc.setBytes(&kt, length: 4, index: 8)
                        enc.setBuffer(bstop, offset: 0, index: 9)
                        enc.setBytes(&lp, length: 4, index: 10)
                        enc.dispatchThreadgroups(MTLSize(width: 1, height: N / 8, depth: Ktop),
                                                 threadsPerThreadgroup: MTLSize(width: 32, height: 2, depth: 1))
                    }
                    enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
                    return (cb.gpuEndTime - cb.gpuStartTime) * 1e6
                }
                _ = runCB(20)
                return runCB(reps) / Double(reps)
            }

            let bytes3 = Double(Ktop) * (Double(N * K) * 3.0 / 8.0 + Double(N) * Double(K / 64) * 2.0 * 2.0)
            let bytes4 = Double(Ktop) * (Double(N * K) / 2.0 + Double(N) * Double(K / 64) * 2.0 * 2.0)
            let g3f16 = timeKernel(RawMetalForward._gqmm3RowsPipeline, wq: wq3, sc: scF16, bi: biF16, dt: .float16, elem: 2)
            let g3f32 = timeKernel(RawMetalForward._gqmm3RowsF32Pipeline, wq: wq3, sc: scBF16, bi: biBF16, dt: .float32, elem: 4)
            let g4f16 = timeKernel(RawMetalForward._gqmmRowsPipeline, wq: wq4, sc: scF16, bi: biF16, dt: .float16, elem: 2)
            func line(_ name: String, _ us: Double?, _ bytes: Double) -> String {
                guard let us = us else { return "    \(name): (nil pipeline)\n" }
                let gbs = bytes / (us * 1e-6) / 1e9
                return String(format: "    %-16s %.1f µs/gather  %.0f GB/s\n", (name as NSString).utf8String!, us, gbs)
            }
            out += line("gqmm3 f16", g3f16, bytes3)
            out += line("gqmm3 f32 (REAL)", g3f32, bytes3)
            out += line("gqmm4 f16", g4f16, bytes4)
            if let a = g3f16, let b = g4f16 { out += String(format: "    → (A) 3bit/4bit f16 = %.2f×\n", a / b) }
            if let a = g3f32, let b = g4f16 { out += String(format: "    → (B) 3bit-f32(real)/4bit-f16 = %.2f×  (>1=slower)\n", a / b) }
        }
        out += "  註: gqmm3 は未配線(Stage2-4)ゆえ e2e model tok/s は未測。これは純 kernel gather コスト。"
        return out
    }
}
