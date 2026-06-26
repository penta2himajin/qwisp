import Foundation
import MLX
import MLXFast

/// 4 shard safetensors を mmap ロードし name→MLXArray を保持（M2b-3 weight ローダ後者版）.
/// この checkpoint は conv1d 既 sanitized([.,K,1])・mtp 別ファイル・名前は language_model. 前置済
/// なので sanitize 変換は不要（名前引きのみ）。
public final class WeightStore {
    public private(set) var arrays: [String: MLXArray] = [:]

    public init(modelDir: String) throws {
        let dir = URL(fileURLWithPath: modelDir)
        let idxURL = dir.appendingPathComponent("model.safetensors.index.json")
        let data = try Data(contentsOf: idxURL)
        let top = try JSONSerialization.jsonObject(with: data) as? [String: Any] ?? [:]
        let wm = (top["weight_map"] as? [String: String]) ?? [:]
        let shards = Set(wm.values)
        for shard in shards.sorted() {
            let m = try loadArrays(url: dir.appendingPathComponent(shard))
            for (k, v) in m { arrays[k] = v }
        }
    }

    public func get(_ name: String) -> MLXArray? { arrays[name] }
    public func req(_ name: String) -> MLXArray { arrays[name]! }

    /// expert(switch_mlp) 以外を eval して常駐させる（experts は mmap のまま on-demand）。
    public func residentNonExperts() {
        let nonExpert = arrays.filter { !$0.key.contains(".switch_mlp.") }.map { $0.value }
        MLX.eval(nonExpert)
    }
}

/// qwen3_5_moe full forward（cache=None prefill）。embed→DecoderLayer×40→norm→lm_head。
public final class QwispModel {
    let store: WeightStore
    let numLayers: Int
    let fullAttnInterval: Int
    let eps: Float
    var layers: [DecoderLayer] = []

    public init(store: WeightStore, numLayers: Int = 40, fullAttnInterval: Int = 4,
                eps: Float = 1e-6) {
        self.store = store; self.numLayers = numLayers
        self.fullAttnInterval = fullAttnInterval; self.eps = eps
        for i in 0 ..< numLayers { layers.append(buildLayer(i)) }
    }

    func isLinear(_ i: Int) -> Bool { (i + 1) % fullAttnInterval != 0 }

    func q(_ name: String, _ bits: Int) -> Proj {
        .quantized(store.req("\(name).weight"), store.req("\(name).scales"),
                   store.req("\(name).biases"), bits)
    }

    func buildMoE(_ p: String) -> MoEBlock {
        MoEBlock(
            topK: 8, numExperts: 256, normTopk: true, expertBits: 4,
            gate: q("\(p).gate", 8),
            swGateW: store.req("\(p).switch_mlp.gate_proj.weight"),
            swGateS: store.req("\(p).switch_mlp.gate_proj.scales"),
            swGateB: store.req("\(p).switch_mlp.gate_proj.biases"),
            swUpW: store.req("\(p).switch_mlp.up_proj.weight"),
            swUpS: store.req("\(p).switch_mlp.up_proj.scales"),
            swUpB: store.req("\(p).switch_mlp.up_proj.biases"),
            swDownW: store.req("\(p).switch_mlp.down_proj.weight"),
            swDownS: store.req("\(p).switch_mlp.down_proj.scales"),
            swDownB: store.req("\(p).switch_mlp.down_proj.biases"),
            shGate: q("\(p).shared_expert.gate_proj", 4),
            shUp: q("\(p).shared_expert.up_proj", 4),
            shDown: q("\(p).shared_expert.down_proj", 4),
            sharedGate: q("\(p).shared_expert_gate", 8))
    }

    func buildLayer(_ i: Int) -> DecoderLayer {
        let p = "language_model.model.layers.\(i)"
        let lin = isLinear(i)
        var gdn: GatedDeltaNetLayer? = nil
        var attn: AttentionLayer? = nil
        if lin {
            let la = "\(p).linear_attn"
            gdn = GatedDeltaNetLayer(
                numKHeads: 16, numVHeads: 32, headKDim: 128, headVDim: 128, convKernel: 4, eps: eps,
                inProjQKV: q("\(la).in_proj_qkv", 4), inProjZ: q("\(la).in_proj_z", 4),
                inProjB: q("\(la).in_proj_b", 4), inProjA: q("\(la).in_proj_a", 4),
                outProj: q("\(la).out_proj", 4),
                conv1dW: store.req("\(la).conv1d.weight"), normWeight: store.req("\(la).norm.weight"),
                aLog: store.req("\(la).A_log"), dtBias: store.req("\(la).dt_bias"))
        } else {
            let sa = "\(p).self_attn"
            attn = AttentionLayer(
                numHeads: 16, numKVHeads: 2, headDim: 256, ropeDim: 64, ropeBase: 1e7, eps: eps,
                qProj: q("\(sa).q_proj", 4), kProj: q("\(sa).k_proj", 4),
                vProj: q("\(sa).v_proj", 4), oProj: q("\(sa).o_proj", 4),
                qNorm: store.req("\(sa).q_norm.weight"), kNorm: store.req("\(sa).k_norm.weight"))
        }
        return DecoderLayer(
            isLinear: lin, eps: eps,
            inputLayernorm: store.req("\(p).input_layernorm.weight"),
            postAttentionLayernorm: store.req("\(p).post_attention_layernorm.weight"),
            gdn: gdn, attn: attn, mlp: buildMoE("\(p).mlp"))
    }

    func embed(_ ids: MLXArray) -> MLXArray {
        ModelHead.embed(ids: ids, weight: store.req("language_model.model.embed_tokens.weight"),
                        scales: store.req("language_model.model.embed_tokens.scales"),
                        biases: store.req("language_model.model.embed_tokens.biases"), bits: 4)
    }

    func headProj() -> Proj {
        .quantized(store.req("language_model.lm_head.weight"),
                   store.req("language_model.lm_head.scales"),
                   store.req("language_model.lm_head.biases"), 4)
    }

    /// ids: [1, T] → logits [1, T, vocab]（cache=None prefill）。f32=true で activations を float32 に。
    public func callAsFunction(_ ids: MLXArray, f32: Bool = false) -> MLXArray {
        var h = embed(ids)
        if f32 { h = h.asType(.float32) }
        for layer in layers { h = layer(h) }
        h = MLXFast.rmsNorm(h, weight: store.req("language_model.model.norm.weight"), eps: eps)
        return headProj().apply(h)
    }

    /// 中間 hidden を捕捉する forward（diagnostics）。captureLayers の各層後の h を返す。
    public func forwardCapturing(_ ids: MLXArray, _ captureLayers: Set<Int>)
        -> (logits: MLXArray, embed: MLXArray, captured: [Int: MLXArray], normed: MLXArray) {
        var h = embed(ids)
        let h0 = h
        var captured: [Int: MLXArray] = [:]
        for (i, layer) in layers.enumerated() {
            h = layer(h)
            if captureLayers.contains(i) { captured[i] = h }
        }
        let normed = MLXFast.rmsNorm(h, weight: store.req("language_model.model.norm.weight"), eps: eps)
        return (headProj().apply(normed), h0, captured, normed)
    }
}

public enum FullModelValidation {
    public static func run(modelDir: String, refPath: String) throws -> String {
        let r = try loadArrays(url: URL(fileURLWithPath: refPath))
        guard let ids = r["ids"], let expLogits = r["logits"] else {
            return "ERROR: full-model ref 不足"
        }
        let t0 = DispatchTime.now()
        let store = try WeightStore(modelDir: modelDir)
        store.residentNonExperts()
        let model = QwispModel(store: store)
        let tLoad = Double(DispatchTime.now().uptimeNanoseconds - t0.uptimeNanoseconds) / 1e9

        func rel(_ a: MLXArray, _ b: MLXArray) -> Float {
            let af = a.asType(.float32)
            return MLX.max(MLX.abs(af - b)).item(Float.self)
                / (MLX.max(MLX.abs(b)).item(Float.self) + 1e-9)
        }

        // 中間 hidden の層別 rel を出して発散点を局在化
        var diag = ""
        let caps: Set<Int> = [0, 1, 3, 19, 39]
        let (logits, h0, captured, normed) = model.forwardCapturing(ids, caps)
        logits.eval()
        if let he = r["h_embed"] { diag += String(format: " embed=%.1e", rel(h0, he)) }
        for i in caps.sorted() {
            if let hr = r["h_after_\(i)"], let hc = captured[i] {
                diag += String(format: " L%d=%.1e", i, rel(hc, hr))
            }
        }
        if let hn = r["h_normed"] { diag += String(format: " norm=%.1e", rel(normed, hn)) }

        let d = rel(logits, expLogits)
        let T = ids.dim(-1)
        func matchCount(_ lg: MLXArray, _ exp: MLXArray) -> Int {
            var m = 0
            for t in 0 ..< T where MLX.argMax(lg[0, t], axis: -1).item(Int.self)
                == MLX.argMax(exp[0, t], axis: -1).item(Int.self) { m += 1 }
            return m
        }
        let match = matchCount(logits, expLogits)

        // float32 クロスチェック（バグ排除: Python f32 と一致すれば配線は正しく f16 差は精度）
        var f32Line = ""
        if let expL32 = r["logits_f32"] {
            let l32 = model(ids, f32: true)
            l32.eval()
            f32Line = String(format: "\n   f32クロスチェック: logits_rel=%.2e argmax %d/%d",
                             rel(l32, expL32), matchCount(l32, expL32), T)
        }

        let ok = match == T
        return String(format: "[M2b-3] FULL forward(40層): logits_rel=%.2e argmax %d/%d (f16)  %@  (load %.1fs)\n   層別rel:%@%@",
                      d, match, T, ok ? "OK ✅" : "≈ 精度差", tLoad, diag, f32Line)
    }
}
