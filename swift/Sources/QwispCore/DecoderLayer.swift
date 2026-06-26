import Foundation
import MLX
import MLXFast

/// qwen3_5.DecoderLayer の Swift 移植（M2b-3 結線）.
/// input_layernorm → (linear_attn | self_attn) → residual → post_attention_layernorm
/// → mlp(MoE) → residual。cache=None の prefill/単一チャンク。
public struct DecoderLayer {
    let isLinear: Bool
    let eps: Float
    let inputLayernorm: MLXArray
    let postAttentionLayernorm: MLXArray
    let gdn: GatedDeltaNetLayer?   // isLinear のとき
    let attn: AttentionLayer?      // それ以外
    let mlp: MoEBlock

    public init(isLinear: Bool, eps: Float, inputLayernorm: MLXArray,
                postAttentionLayernorm: MLXArray, gdn: GatedDeltaNetLayer?,
                attn: AttentionLayer?, mlp: MoEBlock) {
        self.isLinear = isLinear; self.eps = eps
        self.inputLayernorm = inputLayernorm
        self.postAttentionLayernorm = postAttentionLayernorm
        self.gdn = gdn; self.attn = attn; self.mlp = mlp
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let normed = MLXFast.rmsNorm(x, weight: inputLayernorm, eps: eps)
        let r = isLinear ? gdn!(normed) : attn!(normed)
        let h = x + r
        // mlp は [T,H] を取るので [B,S,H]→[B*S,H] に畳んで戻す
        let postNorm = MLXFast.rmsNorm(h, weight: postAttentionLayernorm, eps: eps)
        let B = h.dim(0), S = h.dim(1), H = h.dim(2)
        let flat = postNorm.reshaped([B * S, H])
        let mlpOut = mlp(flat).reshaped([B, S, H])
        return h + mlpOut
    }
}

public enum DecoderLayerValidation {
    static func mlpFrom(_ r: [String: MLXArray]) -> MoEBlock {
        func q8(_ n: String) -> Proj {
            .quantized(r["\(n).weight"]!, r["\(n).scales"]!, r["\(n).biases"]!, 8)
        }
        func q4(_ n: String) -> Proj {
            .quantized(r["\(n).weight"]!, r["\(n).scales"]!, r["\(n).biases"]!, 4)
        }
        return MoEBlock(
            topK: 8, numExperts: 256, normTopk: true, expertBits: 4,
            gate: q8("gate"),
            swGateW: r["switch_mlp.gate_proj.weight"]!, swGateS: r["switch_mlp.gate_proj.scales"]!,
            swGateB: r["switch_mlp.gate_proj.biases"]!,
            swUpW: r["switch_mlp.up_proj.weight"]!, swUpS: r["switch_mlp.up_proj.scales"]!,
            swUpB: r["switch_mlp.up_proj.biases"]!,
            swDownW: r["switch_mlp.down_proj.weight"]!, swDownS: r["switch_mlp.down_proj.scales"]!,
            swDownB: r["switch_mlp.down_proj.biases"]!,
            shGate: q4("shared_expert.gate_proj"), shUp: q4("shared_expert.up_proj"),
            shDown: q4("shared_expert.down_proj"), sharedGate: q8("shared_expert_gate"))
    }

    public static func run(refPath: String, label: String) throws -> String {
        let r = try loadArrays(url: URL(fileURLWithPath: refPath))
        func q4(_ n: String) -> Proj {
            .quantized(r["\(n).weight"]!, r["\(n).scales"]!, r["\(n).biases"]!, 4)
        }
        guard let x = r["x"], let expY = r["y"], let iln = r["input_layernorm_weight"],
              let pln = r["post_attention_layernorm_weight"] else {
            return "ERROR: \(label) ref 不足"
        }
        let isLinear = r["conv1d"] != nil
        var gdn: GatedDeltaNetLayer? = nil
        var attn: AttentionLayer? = nil
        if isLinear {
            gdn = GatedDeltaNetLayer(
                numKHeads: 16, numVHeads: 32, headKDim: 128, headVDim: 128, convKernel: 4, eps: 1e-6,
                inProjQKV: q4("in_proj_qkv"), inProjZ: q4("in_proj_z"), inProjB: q4("in_proj_b"),
                inProjA: q4("in_proj_a"), outProj: q4("out_proj"),
                conv1dW: r["conv1d"]!, normWeight: r["la_norm_weight"]!,
                aLog: r["A_log"]!, dtBias: r["dt_bias"]!)
        } else {
            attn = AttentionLayer(
                numHeads: 16, numKVHeads: 2, headDim: 256, ropeDim: 64, ropeBase: 1e7, eps: 1e-6,
                qProj: q4("q_proj"), kProj: q4("k_proj"), vProj: q4("v_proj"), oProj: q4("o_proj"),
                qNorm: r["q_norm_weight"]!, kNorm: r["k_norm_weight"]!)
        }
        let layer = DecoderLayer(
            isLinear: isLinear, eps: 1e-6, inputLayernorm: iln, postAttentionLayernorm: pln,
            gdn: gdn, attn: attn, mlp: mlpFrom(r))
        let y = layer(x)
        y.eval()
        let d = MLX.max(MLX.abs(y - expY)).item(Float.self)
            / (MLX.max(MLX.abs(expY)).item(Float.self) + 1e-9)
        let ok = d < 2e-3
        return String(format: "[M2b-3] %@(%@): y_rel=%.2e  %@",
                      label, isLinear ? "linear" : "full-attn", d,
                      ok ? "OK ✅ 実重み一致" : "MISMATCH ❌")
    }
}
