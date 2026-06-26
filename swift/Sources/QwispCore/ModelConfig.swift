import Foundation

/// Qwen3.6-35B-A3B (qwen3_5_moe_text) の config（M2b-0）。config.json の text_config から読む。
public struct QwispConfig: Codable, Sendable {
    public var hiddenSize: Int
    public var numHiddenLayers: Int
    public var numAttentionHeads: Int
    public var numKeyValueHeads: Int
    public var headDim: Int
    public var rmsNormEps: Float
    public var vocabSize: Int
    public var ropeTheta: Float
    // MoE
    public var numExperts: Int
    public var numExpertsPerTok: Int
    public var moeIntermediateSize: Int
    public var sharedExpertIntermediateSize: Int
    public var normTopkProb: Bool
    // GatedDeltaNet (linear attention)
    public var linearNumValueHeads: Int
    public var linearNumKeyHeads: Int
    public var linearKeyHeadDim: Int
    public var linearValueHeadDim: Int
    public var linearConvKernelDim: Int
    public var fullAttentionInterval: Int
    public var layerTypes: [String]

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case rmsNormEps = "rms_norm_eps"
        case vocabSize = "vocab_size"
        case ropeTheta = "rope_theta"
        case numExperts = "num_experts"
        case numExpertsPerTok = "num_experts_per_tok"
        case moeIntermediateSize = "moe_intermediate_size"
        case sharedExpertIntermediateSize = "shared_expert_intermediate_size"
        case normTopkProb = "norm_topk_prob"
        case linearNumValueHeads = "linear_num_value_heads"
        case linearNumKeyHeads = "linear_num_key_heads"
        case linearKeyHeadDim = "linear_key_head_dim"
        case linearValueHeadDim = "linear_value_head_dim"
        case linearConvKernelDim = "linear_conv_kernel_dim"
        case fullAttentionInterval = "full_attention_interval"
        case layerTypes = "layer_types"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        numHiddenLayers = try c.decode(Int.self, forKey: .numHiddenLayers)
        numAttentionHeads = try c.decode(Int.self, forKey: .numAttentionHeads)
        numKeyValueHeads = try c.decode(Int.self, forKey: .numKeyValueHeads)
        headDim = try c.decode(Int.self, forKey: .headDim)
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        vocabSize = try c.decode(Int.self, forKey: .vocabSize)
        ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10_000_000
        numExperts = try c.decode(Int.self, forKey: .numExperts)
        numExpertsPerTok = try c.decode(Int.self, forKey: .numExpertsPerTok)
        moeIntermediateSize = try c.decode(Int.self, forKey: .moeIntermediateSize)
        sharedExpertIntermediateSize = try c.decode(Int.self, forKey: .sharedExpertIntermediateSize)
        normTopkProb = try c.decodeIfPresent(Bool.self, forKey: .normTopkProb) ?? true
        linearNumValueHeads = try c.decode(Int.self, forKey: .linearNumValueHeads)
        linearNumKeyHeads = try c.decode(Int.self, forKey: .linearNumKeyHeads)
        linearKeyHeadDim = try c.decode(Int.self, forKey: .linearKeyHeadDim)
        linearValueHeadDim = try c.decode(Int.self, forKey: .linearValueHeadDim)
        linearConvKernelDim = try c.decode(Int.self, forKey: .linearConvKernelDim)
        fullAttentionInterval = try c.decodeIfPresent(Int.self, forKey: .fullAttentionInterval) ?? 4
        layerTypes = try c.decodeIfPresent([String].self, forKey: .layerTypes) ?? []
    }

    public func isLinearLayer(_ i: Int) -> Bool {
        if i < layerTypes.count { return layerTypes[i] == "linear_attention" }
        return (i + 1) % fullAttentionInterval != 0   // フォールバック: 4 周期で full
    }

    /// model dir の config.json（text_config）から読む。
    public static func load(modelDir: String) throws -> QwispConfig {
        let url = URL(fileURLWithPath: modelDir).appendingPathComponent("config.json")
        let data = try Data(contentsOf: url)
        let top = try JSONSerialization.jsonObject(with: data) as? [String: Any] ?? [:]
        let textConfig = (top["text_config"] as? [String: Any]) ?? top
        let tcData = try JSONSerialization.data(withJSONObject: textConfig)
        return try JSONDecoder().decode(QwispConfig.self, from: tcData)
    }
}
