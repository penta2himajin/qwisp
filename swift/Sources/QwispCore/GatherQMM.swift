import Foundation
import MLX

/// M1: mlx-swift の gatherQuantizedMatmul が Python mlx と**ビット一致**するか検証.
/// Python(`qwisp.swift_ref`)が保存した x/inds/w/scales/biases と expected を読み、
/// Swift 側で再計算して max|Δ| を出す。これで Swift エンジンの量子化 matmul 基盤を確立する。
public enum GatherQMMValidation {
    public static func run(refPath: String) throws -> String {
        let url = URL(fileURLWithPath: refPath)
        let ref = try loadArrays(url: url)
        guard let x = ref["x"], let inds = ref["inds"], let w = ref["w"],
              let scales = ref["scales"], let biases = ref["biases"],
              let expected = ref["expected"]
        else { return "ERROR: ref に必要な配列が無い (keys=\(ref.keys.sorted()))" }

        // Python: xe = expand_dims(x,(-2,-3)) → [T,1,1,IN]; gather_qmm(...,bits=2,gs=64,transpose=True)
        let xe = x.expandedDimensions(axes: [-2, -3])
        let rhs = inds.asType(.uint32)
        var y = gatherQuantizedMatmul(
            xe, w, scales: scales, biases: biases,
            rhsIndices: rhs, transpose: true,
            groupSize: 64, bits: 2, mode: .affine, sortedIndices: false)
        // [T,K,1,OUT] → [T,K,OUT]
        let T = x.dim(0), K = inds.dim(1), OUT = w.dim(1)
        y = y.reshaped([T, K, OUT])
        y.eval()

        let diff = MLX.max(MLX.abs(y - expected)).item(Float.self)
        let denom = MLX.max(MLX.abs(expected)).item(Float.self)
        let rel = diff / (denom + 1e-9)
        let ok = rel < 1e-3
        return String(
            format: "[M1] gatherQuantizedMatmul vs Python: out=[%d,%d,%d] max|Δ|=%.3e rel=%.3e  %@",
            T, K, OUT, diff, rel, ok ? "OK ✅ (bit一致)" : "MISMATCH ❌")
    }
}
