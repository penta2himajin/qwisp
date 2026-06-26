import Foundation
import MLX
import MLXFast

/// MTP head（EAGLE 流 1 段ドラフト）の Swift 移植（M2c）.
/// fc(concat(norm(emb), norm(h_prev))) → full-attn 層 → MoE → mtp.norm → main lm_head 流用。
/// experts のみ 4bit gs=32、attn/gate/shared/fc/norm は F16。norm 重みは +1 シフト復元。
public final class MTPHead {
    let store: WeightStore               // main の embed/lm_head 共有
    let eps: Float
    // F16 plain
    let fc: MLXArray                      // [2H, H] → cat[2H] を H に
    let preEmb: MLXArray, preHid: MLXArray, finalNorm: MLXArray
    let inputLayernorm: MLXArray, postAttentionLayernorm: MLXArray
    let attn: AttentionLayer
    let moe: MoEBlock

    public init(modelDir: String, store: WeightStore, eps: Float = 1e-6) throws {
        self.store = store; self.eps = eps
        let url = URL(fileURLWithPath: modelDir).appendingPathComponent("mtp.safetensors")
        let w = try loadArrays(url: url)
        func g(_ k: String) -> MLXArray { w["mtp.\(k)"]! }
        // norm は -1 シフト格納 → +1 で復元（in/post_ln, q/k_norm のみ。pre_*/final は据置=canonical）
        func gn(_ k: String) -> MLXArray { w["mtp.\(k)"]! + MLXArray(Float(1)).asType(w["mtp.\(k)"]!.dtype) }

        fc = g("fc.weight")
        preEmb = g("pre_fc_norm_embedding.weight")
        preHid = g("pre_fc_norm_hidden.weight")
        finalNorm = g("norm.weight")
        inputLayernorm = gn("layers.0.input_layernorm.weight")
        postAttentionLayernorm = gn("layers.0.post_attention_layernorm.weight")

        attn = AttentionLayer(
            numHeads: 16, numKVHeads: 2, headDim: 256, ropeDim: 64, ropeBase: 1e7, eps: eps,
            qProj: .plain(g("layers.0.self_attn.q_proj.weight")),
            kProj: .plain(g("layers.0.self_attn.k_proj.weight")),
            vProj: .plain(g("layers.0.self_attn.v_proj.weight")),
            oProj: .plain(g("layers.0.self_attn.o_proj.weight")),
            qNorm: gn("layers.0.self_attn.q_norm.weight"),
            kNorm: gn("layers.0.self_attn.k_norm.weight"))

        // experts を [256, ...] に stack（mtp は per-expert 格納, 4bit gs=32）
        func stackE(_ proj: String, _ part: String) -> MLXArray {
            MLX.stacked((0 ..< 256).map { w["mtp.layers.0.mlp.experts.\($0).\(proj).\(part)"]! }, axis: 0)
        }
        moe = MoEBlock(
            topK: 8, numExperts: 256, normTopk: true, expertBits: 4, expertGroupSize: 32,
            gate: .plain(g("layers.0.mlp.gate.weight")),
            swGateW: stackE("gate_proj", "weight"), swGateS: stackE("gate_proj", "scales"),
            swGateB: stackE("gate_proj", "biases"),
            swUpW: stackE("up_proj", "weight"), swUpS: stackE("up_proj", "scales"),
            swUpB: stackE("up_proj", "biases"),
            swDownW: stackE("down_proj", "weight"), swDownS: stackE("down_proj", "scales"),
            swDownB: stackE("down_proj", "biases"),
            shGate: .plain(g("layers.0.mlp.shared_expert.gate_proj.weight")),
            shUp: .plain(g("layers.0.mlp.shared_expert.up_proj.weight")),
            shDown: .plain(g("layers.0.mlp.shared_expert.down_proj.weight")),
            sharedGate: .plain(g("layers.0.mlp.shared_expert_gate.weight")))
    }

    func embedTok(_ tok: MLXArray) -> MLXArray {
        ModelHead.embed(ids: tok, weight: store.req("language_model.model.embed_tokens.weight"),
                        scales: store.req("language_model.model.embed_tokens.scales"),
                        biases: store.req("language_model.model.embed_tokens.biases"), bits: 4)
    }

    /// hPrev:[1,L,H] main post-norm hidden, tok:[1,L] 条件トークン → 次々トークン logits[1,L,V]
    public func callAsFunction(_ hPrev: MLXArray, _ tok: MLXArray) -> MLXArray {
        let emb = embedTok(tok)
        let e = MLXFast.rmsNorm(emb, weight: preEmb, eps: eps)
        let hh = MLXFast.rmsNorm(hPrev, weight: preHid, eps: eps)
        let cat = MLX.concatenated([e, hh], axis: -1)            // [1,L,2H]  (emb_hid 順)
        var x = MLX.matmul(cat, fc.transposed())                 // [1,L,H]
        let r = attn(MLXFast.rmsNorm(x, weight: inputLayernorm, eps: eps))  // L>1→causal
        x = x + r
        let B = x.dim(0), L = x.dim(1), H = x.dim(2)
        let post = MLXFast.rmsNorm(x, weight: postAttentionLayernorm, eps: eps)
        x = x + moe(post.reshaped([B * L, H])).reshaped([B, L, H])
        let normed = MLXFast.rmsNorm(x, weight: finalNorm, eps: eps)
        let head = Proj.quantized(store.req("language_model.lm_head.weight"),
                                  store.req("language_model.lm_head.scales"),
                                  store.req("language_model.lm_head.biases"), 4)
        return head.apply(normed)
    }
}

public enum MTPHeadValidation {
    public static func run(modelDir: String, refPath: String) throws -> String {
        let r = try loadArrays(url: URL(fileURLWithPath: refPath))
        guard let hidden = r["hidden"], let tok = r["tok"], let expDraft = r["draft"],
              let loHi = r["lo_hi"], let target = r["target"] else {
            return "ERROR: mtp ref 不足"
        }
        let store = try WeightStore(modelDir: modelDir)
        store.residentNonExperts()
        let head = try MTPHead(modelDir: modelDir, store: store)

        let L = hidden.dim(0)
        let hPrev = hidden.reshaped([1, L, hidden.dim(1)])
        let tokIn = tok.reshaped([1, L])
        let logits = head(hPrev, tokIn)
        let draft = MLX.argMax(logits[0], axis: -1).asType(.int32)   // [L]
        draft.eval()

        // 全位置で Python の draft argmax と一致するか
        let lh = loHi.asArray(Int32.self); let lo = Int(lh[0]); let hi = Int(lh[1])
        let dSwift = draft.asArray(Int32.self)
        let dRef = expDraft.asArray(Int32.self)
        var argmaxMatch = 0
        for i in 0 ..< L where dSwift[i] == dRef[i] { argmaxMatch += 1 }
        // eval 窓の acceptance（Swift draft vs target）
        let tgt = target.asArray(Int32.self)
        var acc = 0
        for (j, i) in (lo ..< hi).enumerated() where j < tgt.count && dSwift[i] == tgt[j] { acc += 1 }
        let accRate = Double(acc) / Double(hi - lo)
        let ok = argmaxMatch == L
        return String(format: "[M2c] MTP head: draft argmax %d/%d %@  acceptance=%.3f (Python 0.952)",
                      argmaxMatch, L, ok ? "OK ✅ Python一致" : "MISMATCH ❌", accRate)
    }
}
