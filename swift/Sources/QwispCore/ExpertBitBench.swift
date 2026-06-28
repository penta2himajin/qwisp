import Foundation
import MLX
import MLXRandom

/// pillar B feasibility gate: MoE expert の gatherQuantizedMatmul が bit 幅でどう速くなるか。
/// 研究の警告「naive 3-bit は 4-bit より遅いことがある」を最初に潰す。3-bit が batch=1(T=1)で
/// 4-bit より速ければ B は成立、遅ければ B は kernel 律速で死路。実重み不要(速度は shape+bit のみ依存)。
public enum ExpertBitBench {
    /// gate/up [512,2048] + down [2048,512]、256 experts/topK=8、gs=64。T=1,2,8 で bits 2/3/4/8 計時。
    public static func run() -> String {
        MLXRandom.seed(0)
        let H = 2048, I = 512, E = 256, K = 8, gs = 64
        // ランダム f16 重み → 各 bit で量子化（gate/up: [E,I,H], down: [E,H,I]）
        let gW = (MLXRandom.normal([E, I, H]) * 0.02).asType(.float16)
        let dW = (MLXRandom.normal([E, H, I]) * 0.02).asType(.float16)

        func quant(_ w: MLXArray, _ b: Int) -> (MLXArray, MLXArray, MLXArray) {
            let (wq, s, bi) = MLX.quantized(w, groupSize: gs, bits: b); return (wq, s, bi!)
        }

        func bench(_ bits: Int, _ T: Int) -> Double? {
            // 量子化（失敗＝非対応 bit なら nil）
            let gq: (MLXArray, MLXArray, MLXArray), uq: (MLXArray, MLXArray, MLXArray), dq: (MLXArray, MLXArray, MLXArray)
            gq = quant(gW, bits); uq = quant(gW, bits); dq = quant(dW, bits)
            let xg = (MLXRandom.normal([T, 1, 1, H]) * 0.5).asType(.float16)
            let inds = MLXArray((0 ..< T * K).map { Int32(($0 * 37) % E) }, [T, K]).asType(.uint32)
            // gate/up: [T,1,1,H] gather→ [T,K,1,I]; down: [T,K,1,I] gather→ [T,K,1,H]
            func moe() -> MLXArray {
                let g = gatherQuantizedMatmul(xg, gq.0, scales: gq.1, biases: gq.2, rhsIndices: inds,
                                              transpose: true, groupSize: gs, bits: bits, mode: .affine, sortedIndices: false)
                let u = gatherQuantizedMatmul(xg, uq.0, scales: uq.1, biases: uq.2, rhsIndices: inds,
                                              transpose: true, groupSize: gs, bits: bits, mode: .affine, sortedIndices: false)
                let h = (g * u).reshaped([T, K, 1, I])    // 簡略 act（速度のみ目的）
                let d = gatherQuantizedMatmul(h, dq.0, scales: dq.1, biases: dq.2, rhsIndices: inds,
                                              transpose: true, groupSize: gs, bits: bits, mode: .affine, sortedIndices: false)
                return d
            }
            let w = moe(); w.eval()                       // warmup
            let reps = 100
            let t = DispatchTime.now().uptimeNanoseconds
            for _ in 0 ..< reps { let y = moe(); y.eval() }
            return Double(DispatchTime.now().uptimeNanoseconds - t) / Double(reps) / 1e6
        }

        var lines: [String] = []
        for T in [1, 2, 4, 8, 16, 24] {
            var row = "T=\(T): "
            var ref4 = 0.0
            for bits in [4, 3, 2, 8] {
                if let ms = bench(bits, T) {
                    if bits == 4 { ref4 = ms }
                    let sp = (bits != 4 && ref4 > 0) ? String(format: " (%.2fx vs4)", ref4 / ms) : ""
                    row += String(format: "%dbit=%.3fms%@  ", bits, ms, sp)
                } else { row += "\(bits)bit=N/A  " }
            }
            lines.append(row)
        }
        return "[ExpertBitBench] MoE gather(gate+up+down, E=256/K=8, H=2048/I=512, gs=64):\n  "
            + lines.joined(separator: "\n  ")
            + "\n  → 3bit が 4bit より速ければ pillar B(MoE 3bit 量子化)成立、遅ければ kernel 律速で死路"
    }
}
