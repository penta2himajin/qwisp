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

public enum ModelHeadValidation {
    public static func run(refPath: String) throws -> String {
        let r = try loadArrays(url: URL(fileURLWithPath: refPath))
        guard let ids = r["ids"], let expEmbed = r["embed_out"], let h = r["h"],
              let expLogits = r["logits"], let normW = r["norm_weight"] else {
            return "ERROR: head ref 不足"
        }
        // embed
        let embed = ModelHead.embed(ids: ids, weight: r["embed.weight"]!,
                                    scales: r["embed.scales"]!, biases: r["embed.biases"]!, bits: 4)
        embed.eval()
        let dE = MLX.max(MLX.abs(embed - expEmbed)).item(Float.self)
            / (MLX.max(MLX.abs(expEmbed)).item(Float.self) + 1e-9)

        // final norm + lm_head
        let normed = MLXFast.rmsNorm(h, weight: normW, eps: 1e-6)
        let head = Proj.quantized(r["lm_head.weight"]!, r["lm_head.scales"]!, r["lm_head.biases"]!, 4)
        let logits = head.apply(normed)
        logits.eval()
        let dL = MLX.max(MLX.abs(logits - expLogits)).item(Float.self)
            / (MLX.max(MLX.abs(expLogits)).item(Float.self) + 1e-9)

        // argmax 一致（最終トークン）
        let T = ids.dim(-1)
        let amSwift = MLX.argMax(logits[0, T - 1], axis: -1).item(Int.self)
        let amRef = MLX.argMax(expLogits[0, T - 1], axis: -1).item(Int.self)

        let ok = dE < 2e-3 && dL < 2e-3 && amSwift == amRef
        return String(format: "[M2b-3] embed+norm+lm_head: embed_rel=%.2e logits_rel=%.2e argmax(%d==%d)  %@",
                      dE, dL, amSwift, amRef, ok ? "OK ✅ 実重み一致" : "MISMATCH ❌")
    }
}
