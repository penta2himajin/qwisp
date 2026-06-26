import Foundation
import MLX
import MLXFast

/// Qwen3NextAttention.__call__ の Swift 移植（M2b-2 full-attention 層）.
/// GQA(16 q-heads / 2 kv-heads, head_dim=256) + q/k RMSNorm + partial RoPE(64dim) +
/// gated output o_proj(out * sigmoid(gate))。cache=None / mask=causal の単一チャンク。
public struct AttentionLayer {
    let numHeads: Int        // 16
    let numKVHeads: Int      // 2
    let headDim: Int         // 256
    let ropeDim: Int         // 64 (= head_dim * partial_rotary_factor)
    let ropeBase: Float      // 1e7
    let eps: Float

    let qProj: MLXArray       // [numHeads*headDim*2, H]
    let kProj: MLXArray       // [numKVHeads*headDim, H]
    let vProj: MLXArray
    let oProj: MLXArray       // [H, numHeads*headDim]
    let qNorm: MLXArray       // [headDim]
    let kNorm: MLXArray       // [headDim]

    var scale: Float { Float(pow(Double(headDim), -0.5)) }

    public init(numHeads: Int, numKVHeads: Int, headDim: Int, ropeDim: Int, ropeBase: Float,
                eps: Float, qProj: MLXArray, kProj: MLXArray, vProj: MLXArray, oProj: MLXArray,
                qNorm: MLXArray, kNorm: MLXArray) {
        self.numHeads = numHeads; self.numKVHeads = numKVHeads; self.headDim = headDim
        self.ropeDim = ropeDim; self.ropeBase = ropeBase; self.eps = eps
        self.qProj = qProj; self.kProj = kProj; self.vProj = vProj; self.oProj = oProj
        self.qNorm = qNorm; self.kNorm = kNorm
    }

    static func linear(_ x: MLXArray, _ w: MLXArray) -> MLXArray { MLX.matmul(x, w.transposed()) }

    func rope(_ x: MLXArray) -> MLXArray {
        MLXFast.RoPE(x, dimensions: ropeDim, traditional: false, base: ropeBase,
                     scale: 1.0, offset: 0)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let B = x.dim(0), L = x.dim(1)

        let qOut = AttentionLayer.linear(x, qProj).reshaped([B, L, numHeads, 2 * headDim])
        var queries = qOut[0..., 0..., 0..., 0 ..< headDim]            // [B,L,H,headDim]
        let gate = qOut[0..., 0..., 0..., headDim...].reshaped([B, L, -1])  // [B,L,H*headDim]

        var keys = AttentionLayer.linear(x, kProj).reshaped([B, L, numKVHeads, headDim])
        var values = AttentionLayer.linear(x, vProj).reshaped([B, L, numKVHeads, headDim])

        // q/k RMSNorm（最終軸 headDim, weight 有り）→ transpose to [B,heads,L,headDim]
        queries = MLXFast.rmsNorm(queries, weight: qNorm, eps: eps).transposed(0, 2, 1, 3)
        keys = MLXFast.rmsNorm(keys, weight: kNorm, eps: eps).transposed(0, 2, 1, 3)
        values = values.transposed(0, 2, 1, 3)

        queries = rope(queries)
        keys = rope(keys)

        var output = MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values, scale: scale, mask: .causal)
        output = output.transposed(0, 2, 1, 3).reshaped([B, L, -1])   // [B,L,H*headDim]

        return AttentionLayer.linear(output * MLX.sigmoid(gate), oProj)
    }
}

public enum AttentionLayerValidation {
    public static func run(refPath: String) throws -> String {
        let r = try loadArrays(url: URL(fileURLWithPath: refPath))
        guard let x = r["x"], let qp = r["q_proj"], let kp = r["k_proj"], let vp = r["v_proj"],
              let op = r["o_proj"], let qn = r["q_norm"], let kn = r["k_norm"],
              let expOut = r["out"] else {
            return "ERROR: attn ref 不足"
        }
        let attn = AttentionLayer(
            numHeads: 16, numKVHeads: 2, headDim: 256, ropeDim: 64, ropeBase: 1e7, eps: 1e-6,
            qProj: qp, kProj: kp, vProj: vp, oProj: op, qNorm: qn, kNorm: kn)
        let out = attn(x)
        out.eval()
        let d = MLX.max(MLX.abs(out - expOut)).item(Float.self)
            / (MLX.max(MLX.abs(expOut)).item(Float.self) + 1e-9)
        let ok = d < 1e-3
        return String(format: "[M2b-2] full-attention 層: out_rel=%.2e  %@",
                      d, ok ? "OK ✅ bit一致" : "MISMATCH ❌")
    }
}
