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

    public func callAsFunction(_ x: MLXArray, cache: LayerCache? = nil) -> MLXArray {
        let normed = MLXFast.rmsNorm(x, weight: inputLayernorm, eps: eps)
        let r = isLinear ? gdn!(normed, cache: cache?.gdn) : attn!(normed, cache: cache?.kv)
        let h = x + r
        // mlp は [T,H] を取るので [B,S,H]→[B*S,H] に畳んで戻す
        let postNorm = MLXFast.rmsNorm(h, weight: postAttentionLayernorm, eps: eps)
        let B = h.dim(0), S = h.dim(1), H = h.dim(2)
        let flat = postNorm.reshaped([B * S, H])
        let mlpOut = mlp(flat).reshaped([B, S, H])
        return h + mlpOut
    }

    /// ★ continuous batching: attn 層は per-slot KV+position(callContinuous), GDN 層は batched gdnCache。
    /// MoE/norm/residual は batched(amortize)。x[B,1,H]。
    public func callContinuous(_ x: MLXArray, gdnCache: GDNCache?, slotKV: [KVCache], positions: [Int]) -> MLXArray {
        let normed = MLXFast.rmsNorm(x, weight: inputLayernorm, eps: eps)
        let r = isLinear ? gdn!(normed, cache: gdnCache) : attn!.callContinuous(normed, slotKV: slotKV, positions: positions)
        let h = x + r
        let postNorm = MLXFast.rmsNorm(h, weight: postAttentionLayernorm, eps: eps)
        let B = h.dim(0), S = h.dim(1), H = h.dim(2)
        let mlpOut = mlp(postNorm.reshaped([B * S, H])).reshaped([B, S, H])
        return h + mlpOut
    }
}

