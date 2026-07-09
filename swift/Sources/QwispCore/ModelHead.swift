import Foundation
import MLX
import MLXFast

/// embed_tokens(QuantizedEmbedding) と lm_head(QuantizedLinear) + final norm（M2b-3 入口/出口）.
public enum ModelHead {
    /// QuantizedEmbedding: 行を gather → dequantize。w/s/b は [vocab, H/8] 等、ids は整数。
    public static func embed(ids: MLXArray, weight: MLXArray, scales: MLXArray, biases: MLXArray,
                             bits: Int = 4) -> MLXArray {
        let flat = ids.reshaped([-1])
        let w = MLX.take(weight, flat, axis: 0)
        let s = MLX.take(scales, flat, axis: 0)
        let b = MLX.take(biases, flat, axis: 0)
        let deq = MLX.dequantized(w, scales: s, biases: b, groupSize: 64, bits: bits, mode: .affine)
        let H = deq.dim(-1)
        return deq.reshaped(ids.shape + [H])
    }
}
