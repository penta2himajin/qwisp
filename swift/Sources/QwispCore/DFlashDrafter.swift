import Foundation
import MLX
import MLXFast
import MLXRandom

// #98 DFlash phase 2 — MLX drafter module.
//
// A faithful Swift/MLX port of the official dflash drafter (reference:
// scratchpad/model_mlx.py). This file is the ONLY new implementation surface for
// the workstream — the frozen forward path and existing model-layer files are
// reused by reference only, never modified.

enum DFlashDrafterError: Error { case stub, invalidConfig }

// MARK: - 1. Config

public struct DFlashDrafterConfig {
    public var hiddenSize: Int
    public var numLayers: Int
    public var numHeads: Int
    public var numKVHeads: Int
    public var headDim: Int
    public var mlpDim: Int
    public var vocabSize: Int
    public var rmsEps: Float
    public var ropeTheta: Float
    /// isSliding per layer (true = sliding_attention, false = full_attention).
    public var layerTypes: [Bool]
    public var slidingWindow: Int
    public var blockSize: Int
    public var targetLayerIds: [Int]
    public var maskTokenId: Int
    /// = targetLayerIds.count * TARGET hidden size (fc maps ctxFeatureDim -> hiddenSize).
    /// NOTE: different from the drafter's own hiddenSize.
    public var ctxFeatureDim: Int

    public init(hiddenSize: Int, numLayers: Int, numHeads: Int, numKVHeads: Int,
                headDim: Int, mlpDim: Int, vocabSize: Int, rmsEps: Float, ropeTheta: Float,
                layerTypes: [Bool], slidingWindow: Int, blockSize: Int,
                targetLayerIds: [Int], maskTokenId: Int, ctxFeatureDim: Int) {
        self.hiddenSize = hiddenSize
        self.numLayers = numLayers
        self.numHeads = numHeads
        self.numKVHeads = numKVHeads
        self.headDim = headDim
        self.mlpDim = mlpDim
        self.vocabSize = vocabSize
        self.rmsEps = rmsEps
        self.ropeTheta = ropeTheta
        self.layerTypes = layerTypes
        self.slidingWindow = slidingWindow
        self.blockSize = blockSize
        self.targetLayerIds = targetLayerIds
        self.maskTokenId = maskTokenId
        self.ctxFeatureDim = ctxFeatureDim
    }

    /// Parse the HF config.json (nested rope_parameters.rope_theta, dflash_config.{block_size,
    /// mask_token_id, target_layer_ids}, layer_types -> isSliding).
    /// ponytail: ctxFeatureDim here is best-effort (targetLayerIds.count * hiddenSize, matching
    /// the upstream Python literal formula) — the real-checkpoint fc.weight shape is the actual
    /// source of truth and is read directly from the safetensors in `load(dir:)`; that later
    /// phase (real-checkpoint bit-parity) is explicitly out of scope for this round.
    public static func load(configJSON: URL) throws -> DFlashDrafterConfig {
        let data = try Data(contentsOf: configJSON)
        guard let top = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw DFlashDrafterError.invalidConfig
        }
        func num(_ v: Any?) -> Double? { (v as? NSNumber)?.doubleValue }
        guard let hiddenSize = num(top["hidden_size"]).map(Int.init),
              let numLayers = num(top["num_hidden_layers"]).map(Int.init),
              let numHeads = num(top["num_attention_heads"]).map(Int.init),
              let numKVHeads = num(top["num_key_value_heads"]).map(Int.init),
              let headDim = num(top["head_dim"]).map(Int.init),
              let mlpDim = num(top["intermediate_size"]).map(Int.init),
              let vocabSize = num(top["vocab_size"]).map(Int.init)
        else { throw DFlashDrafterError.invalidConfig }
        let rmsEps = Float(num(top["rms_norm_eps"]) ?? 1e-6)
        let ropeParams = top["rope_parameters"] as? [String: Any]
        let ropeTheta = Float(num(ropeParams?["rope_theta"]) ?? num(top["rope_theta"]) ?? 1e7)
        let layerTypeStrs = (top["layer_types"] as? [String])
            ?? Array(repeating: "full_attention", count: numLayers)
        let layerTypes = layerTypeStrs.map { $0 == "sliding_attention" }
        let slidingWindow = num(top["sliding_window"]).map(Int.init) ?? 0
        let dflash = (top["dflash_config"] as? [String: Any]) ?? [:]
        let blockSize = num(dflash["block_size"]).map(Int.init) ?? 0
        let maskTokenId = num(dflash["mask_token_id"]).map(Int.init) ?? 0
        let targetLayerIds = (dflash["target_layer_ids"] as? [NSNumber])?.map { $0.intValue } ?? []
        let ctxFeatureDim = targetLayerIds.count * hiddenSize
        return DFlashDrafterConfig(
            hiddenSize: hiddenSize, numLayers: numLayers, numHeads: numHeads, numKVHeads: numKVHeads,
            headDim: headDim, mlpDim: mlpDim, vocabSize: vocabSize, rmsEps: rmsEps, ropeTheta: ropeTheta,
            layerTypes: layerTypes, slidingWindow: slidingWindow, blockSize: blockSize,
            targetLayerIds: targetLayerIds, maskTokenId: maskTokenId, ctxFeatureDim: ctxFeatureDim)
    }
}

// MARK: - 2. Drafter-local KV cache (do NOT modify Cache.swift)

/// keys/values [B, nKV, S, headDim] + absolute-position `offset`
/// (model_mlx RotatingKVCache semantics). Keys are stored ALREADY roped at
/// their absolute positions, so a sliding front-drop preserves correctness.
public final class DFlashKVCache {
    public var keys: MLXArray?
    public var values: MLXArray?
    public var offset: Int = 0
    /// nil for a full_attention layer; sliding-window size for a sliding layer.
    public let slidingWindow: Int?

    public init(slidingWindow: Int?) {
        self.slidingWindow = slidingWindow
    }

    /// Append on axis 2, offset += new S, return the full cached (keys, values).
    /// Sliding eviction (front-drop, offset untouched) fires after the append.
    public func update(_ k: MLXArray, _ v: MLXArray) -> (MLXArray, MLXArray) {
        if let kk = keys, let vv = values {
            keys = MLX.concatenated([kk, k], axis: 2)
            values = MLX.concatenated([vv, v], axis: 2)
        } else {
            keys = k
            values = v
        }
        offset += k.dim(2)
        if let sw = slidingWindow, let kk = keys, let vv = values {
            let keep = sw - 1
            let cur = kk.dim(2)
            if cur > keep {
                keys = kk[0..., 0..., (cur - keep)..., 0...]
                values = vv[0..., 0..., (cur - keep)..., 0...]
            }
        }
        return (keys!, values!)
    }

    /// Drop the last n cached rows and roll offset back by n (reject/commit rollback).
    public func trimBack(_ n: Int) {
        guard n > 0, let kk = keys, let vv = values else { return }
        let cur = kk.dim(2)
        if n >= cur {
            keys = nil
            values = nil
        } else {
            keys = kk[0..., 0..., 0 ..< (cur - n), 0...]
            values = vv[0..., 0..., 0 ..< (cur - n), 0...]
        }
        offset -= n
    }

    /// Pre-projection crop bookkeeping: bump the absolute offset without touching stored rows
    /// (model_mlx DFlashAttention.__call__ `cache.offset += skip`).
    public func bumpOffset(_ n: Int) {
        offset += n
    }
}

// MARK: - internal per-layer attention / decoder (model_mlx DFlashAttention / DFlashDecoderLayer)

private struct DFlashAttentionLayer {
    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let ropeTheta: Float
    let eps: Float
    let isSliding: Bool
    let slidingWindow: Int   // only meaningful when isSliding
    let qProj: Proj
    let kProj: Proj
    let vProj: Proj
    let oProj: Proj
    let qNorm: MLXArray
    let kNorm: MLXArray

    var scale: Float { Float(pow(Double(headDim), -0.5)) }

    private func rope(_ x: MLXArray, _ offset: Int) -> MLXArray {
        // Full head-dim RoPE (no partial rotary factor, unlike AttentionLayer/target model).
        MLXFast.RoPE(x, dimensions: headDim, traditional: false, base: ropeTheta, scale: 1.0, offset: offset)
    }

    /// Boolean windowed-causal mask: query row i (absolute idx offset+i) attends key j iff
    /// j <= offset+i AND j > offset+i - slidingWindow. `totalKeys` = key count after concat.
    private func windowedCausalMask(L: Int, offset: Int, totalKeys: Int) -> MLXArray {
        let qIdx = MLXArray((0 ..< L).map { Int32(offset + $0) }).reshaped([L, 1])
        let kIdx = MLXArray((0 ..< totalKeys).map { Int32($0) }).reshaped([1, totalKeys])
        let notTooOld = kIdx .> (qIdx - MLXArray(Int32(slidingWindow)))
        let notFuture = kIdx .<= qIdx
        return MLX.logicalAnd(notFuture, notTooOld).reshaped([1, 1, L, totalKeys])
    }

    /// x: [B,L,hiddenSize] (evolving normed hidden). xCtx: [B,S,hiddenSize] (hCtx, computed once).
    func callAsFunction(_ x: MLXArray, xCtx: MLXArray, cache: DFlashKVCache) -> MLXArray {
        let B = x.dim(0), L = x.dim(1)
        var xCtxLocal = xCtx
        var S = xCtxLocal.dim(1)
        if isSliding {
            let keepCtx = slidingWindow - 1
            if S > keepCtx {
                let skip = S - keepCtx
                xCtxLocal = xCtxLocal[0..., skip..., 0...]
                S = xCtxLocal.dim(1)
                cache.bumpOffset(skip)
            }
        }

        let queries0 = qProj.apply(x)
        let ctxKeys0 = kProj.apply(xCtxLocal)
        let ctxValues0 = vProj.apply(xCtxLocal)
        let propKeys0 = kProj.apply(x)
        let propValues0 = vProj.apply(x)

        var queries = MLXFast.rmsNorm(queries0.reshaped([B, L, numHeads, headDim]), weight: qNorm, eps: eps)
            .transposed(0, 2, 1, 3)
        var ctxKeys = MLXFast.rmsNorm(ctxKeys0.reshaped([B, S, numKVHeads, headDim]), weight: kNorm, eps: eps)
            .transposed(0, 2, 1, 3)
        var ctxValues = ctxValues0.reshaped([B, S, numKVHeads, headDim]).transposed(0, 2, 1, 3)
        var propKeys = MLXFast.rmsNorm(propKeys0.reshaped([B, L, numKVHeads, headDim]), weight: kNorm, eps: eps)
            .transposed(0, 2, 1, 3)
        var propValues = propValues0.reshaped([B, L, numKVHeads, headDim]).transposed(0, 2, 1, 3)

        queries = rope(queries, cache.offset + S)
        ctxKeys = rope(ctxKeys, cache.offset)
        propKeys = rope(propKeys, cache.offset + S)

        let (cachedKeys, cachedValues) = cache.update(ctxKeys, ctxValues)
        let ctxLen = cachedKeys.dim(2)
        let keys = MLX.concatenated([cachedKeys, propKeys], axis: 2)
        let values = MLX.concatenated([cachedValues, propValues], axis: 2)

        var mask: MLXFast.ScaledDotProductAttentionMaskMode = .none   // full-attention: bidirectional
        if isSliding {
            if ctxLen + L <= slidingWindow {
                mask = .causal
            } else {
                mask = .array(windowedCausalMask(L: L, offset: ctxLen, totalKeys: ctxLen + L))
            }
        }

        let output = MLXFast.scaledDotProductAttention(queries: queries, keys: keys, values: values,
                                                        scale: scale, mask: mask)
        return oProj.apply(output.transposed(0, 2, 1, 3).reshaped([B, L, -1]))
    }
}

private struct DFlashDecoderLayer {
    let attn: DFlashAttentionLayer
    let gateProj: Proj
    let upProj: Proj
    let downProj: Proj
    let inputLN: MLXArray
    let postLN: MLXArray
    let eps: Float

    func callAsFunction(_ x: MLXArray, hCtx: MLXArray, cache: DFlashKVCache) -> MLXArray {
        let h = x + attn(MLXFast.rmsNorm(x, weight: inputLN, eps: eps), xCtx: hCtx, cache: cache)
        let post = MLXFast.rmsNorm(h, weight: postLN, eps: eps)
        let g = gateProj.apply(post)
        let u = upProj.apply(post)
        let mlpOut = downProj.apply((g * MLX.sigmoid(g)) * u)   // swiglu (idiom: MoELayer.swift line 42)
        return h + mlpOut
    }
}

// MARK: - 3. Drafter

public final class DFlashDrafter {
    public let config: DFlashDrafterConfig
    private let fc: Proj
    private let hiddenNorm: MLXArray
    private let finalNorm: MLXArray
    private let layers: [DFlashDecoderLayer]

    /// Map the HF checkpoint names (see spec) into dense f16 weights. Returns nil on any
    /// missing key.
    public init?(config: DFlashDrafterConfig, weights: [String: MLXArray]) {
        func g(_ key: String) -> MLXArray? { weights[key]?.asType(.float16) }
        guard let fcW = g("fc.weight"), let hn = g("hidden_norm.weight"), let fn = g("norm.weight")
        else { return nil }

        var builtLayers: [DFlashDecoderLayer] = []
        builtLayers.reserveCapacity(config.numLayers)
        for i in 0 ..< config.numLayers {
            let p = "layers.\(i)."
            guard let qW = g(p + "self_attn.q_proj.weight"),
                  let kW = g(p + "self_attn.k_proj.weight"),
                  let vW = g(p + "self_attn.v_proj.weight"),
                  let oW = g(p + "self_attn.o_proj.weight"),
                  let qN = g(p + "self_attn.q_norm.weight"),
                  let kN = g(p + "self_attn.k_norm.weight"),
                  let gateW = g(p + "mlp.gate_proj.weight"),
                  let upW = g(p + "mlp.up_proj.weight"),
                  let downW = g(p + "mlp.down_proj.weight"),
                  let inLN = g(p + "input_layernorm.weight"),
                  let postLN = g(p + "post_attention_layernorm.weight")
            else { return nil }

            let isSliding = i < config.layerTypes.count ? config.layerTypes[i] : false
            let attn = DFlashAttentionLayer(
                numHeads: config.numHeads, numKVHeads: config.numKVHeads, headDim: config.headDim,
                ropeTheta: config.ropeTheta, eps: config.rmsEps, isSliding: isSliding,
                slidingWindow: config.slidingWindow,
                qProj: .plain(qW), kProj: .plain(kW), vProj: .plain(vW), oProj: .plain(oW),
                qNorm: qN, kNorm: kN)
            builtLayers.append(DFlashDecoderLayer(
                attn: attn, gateProj: .plain(gateW), upProj: .plain(upW), downProj: .plain(downW),
                inputLN: inLN, postLN: postLN, eps: config.rmsEps))
        }

        self.config = config
        self.fc = .plain(fcW)
        self.hiddenNorm = hn
        self.finalNorm = fn
        self.layers = builtLayers
    }

    /// config.json + all *.safetensors in dir (idiom: MTPHead.swift). Strips a leading
    /// `model.` prefix from checkpoint names when present.
    public static func load(dir: URL) -> DFlashDrafter? {
        guard let cfg = try? DFlashDrafterConfig.load(configJSON: dir.appendingPathComponent("config.json")),
              let entries = try? FileManager.default.contentsOfDirectory(
                  at: dir, includingPropertiesForKeys: nil)
        else { return nil }
        var weights: [String: MLXArray] = [:]
        for url in entries where url.pathExtension == "safetensors" {
            guard let m = try? loadArrays(url: url) else { return nil }
            for (k, v) in m {
                let stripped = k.hasPrefix("model.") ? String(k.dropFirst("model.".count)) : k
                weights[stripped] = v
            }
        }
        return DFlashDrafter(config: cfg, weights: weights)
    }

    public func makeCaches() -> [DFlashKVCache] {
        config.layerTypes.map { DFlashKVCache(slidingWindow: $0 ? config.slidingWindow : nil) }
    }

    /// noise [1, L, hiddenSize] = already-embedded anchor+mask rows (caller embeds).
    /// ctx  [1, S, ctxFeatureDim] = concatenated target taps.
    /// Returns FINAL-NORMED hidden [1, L, hiddenSize] (caller applies the target lm_head).
    public func forward(noise: MLXArray, ctx: MLXArray, caches: [DFlashKVCache]) -> MLXArray {
        let hCtx = MLXFast.rmsNorm(fc.apply(ctx), weight: hiddenNorm, eps: config.rmsEps)   // computed ONCE
        var h = noise
        for (layer, cache) in zip(layers, caches) {
            h = layer(h, hCtx: hCtx, cache: cache)
        }
        return MLXFast.rmsNorm(h, weight: finalNorm, eps: config.rmsEps)
    }

    /// Port of _trim_recent_cache: n = cache.offset - committed; if n > 0 trimBack(n) per cache.
    public func trimTo(committed: Int, caches: [DFlashKVCache]) {
        for cache in caches {
            let n = cache.offset - committed
            if n > 0 { cache.trimBack(n) }
        }
    }
}
