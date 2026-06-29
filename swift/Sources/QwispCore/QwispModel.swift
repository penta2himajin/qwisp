import Foundation
import Metal
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

    /// 全 tensor を eval（experts 含む常駐）。resident regime のベンチ用。
    public func residentAll() { MLX.eval(Array(arrays.values)) }
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

    public func makeCaches() -> [LayerCache] { (0 ..< numLayers).map { _ in LayerCache() } }

    /// cached forward で (post-norm hidden, logits) を返す（MTP 投機用。hidden=lm.model() 相当）。
    public func forwardHidden(_ ids: MLXArray, caches: [LayerCache]) -> (hidden: MLXArray, logits: MLXArray) {
        var h = embed(ids)
        for (i, layer) in layers.enumerated() { h = layer(h, cache: caches[i]) }
        let hidden = MLXFast.rmsNorm(h, weight: store.req("language_model.model.norm.weight"), eps: eps)
        return (hidden, headProj().apply(hidden))
    }
    public var isLinearFlags: [Bool] { layers.map { $0.isLinear } }

    /// cache を使う forward（prefill: S>1, decode: S=1）。caches は in-place 更新される。
    public func callAsFunction(_ ids: MLXArray, caches: [LayerCache], f32: Bool = false) -> MLXArray {
        var h = embed(ids)
        if f32 { h = h.asType(.float32) }
        for (i, layer) in layers.enumerated() { h = layer(h, cache: caches[i]) }
        h = MLXFast.rmsNorm(h, weight: store.req("language_model.model.norm.weight"), eps: eps)
        return headProj().apply(h)
    }

    // ── raw-Metal full forward（task#4）: 全40層を raw decoder layer で回す。decode T=1, cold cache。──
    /// raw decoder layer 1 層（input_norm→mixer raw→res→post_norm→MoE raw→res）。h[1,H]→[1,H]。
    func rawDecoderLayer(_ h: MLXArray, _ i: Int) -> MLXArray? {
        let p = "language_model.model.layers.\(i)", H = h.dim(-1)
        guard let normed = RawMetalForward.rmsNorm(h, store.req("\(p).input_layernorm.weight"), eps: eps, D: H) else { return nil }
        let r: MLXArray
        if isLinear(i) {
            let la = "\(p).linear_attn"
            let rw = RawMetalForward.GDNRawWeights(
                qkvWq: store.req("\(la).in_proj_qkv.weight"), qkvSc: store.req("\(la).in_proj_qkv.scales"), qkvBi: store.req("\(la).in_proj_qkv.biases"),
                zWq: store.req("\(la).in_proj_z.weight"), zSc: store.req("\(la).in_proj_z.scales"), zBi: store.req("\(la).in_proj_z.biases"),
                bWq: store.req("\(la).in_proj_b.weight"), bSc: store.req("\(la).in_proj_b.scales"), bBi: store.req("\(la).in_proj_b.biases"),
                aWq: store.req("\(la).in_proj_a.weight"), aSc: store.req("\(la).in_proj_a.scales"), aBi: store.req("\(la).in_proj_a.biases"),
                outWq: store.req("\(la).out_proj.weight"), outSc: store.req("\(la).out_proj.scales"), outBi: store.req("\(la).out_proj.biases"),
                conv1dW: store.req("\(la).conv1d.weight").reshaped([8192, 4]).asType(.float32), normWeight: store.req("\(la).norm.weight"),
                aLog: store.req("\(la).A_log"), dtBias: store.req("\(la).dt_bias"))
            guard let ro = RawMetalForward.gdnLayerRaw(normed.reshaped([1, 1, H]), rw) else { return nil }
            r = ro
        } else {
            let sa = "\(p).self_attn"
            let aw = RawMetalForward.AttnRawWeights(
                qWq: store.req("\(sa).q_proj.weight"), qSc: store.req("\(sa).q_proj.scales"), qBi: store.req("\(sa).q_proj.biases"),
                kWq: store.req("\(sa).k_proj.weight"), kSc: store.req("\(sa).k_proj.scales"), kBi: store.req("\(sa).k_proj.biases"),
                vWq: store.req("\(sa).v_proj.weight"), vSc: store.req("\(sa).v_proj.scales"), vBi: store.req("\(sa).v_proj.biases"),
                oWq: store.req("\(sa).o_proj.weight"), oSc: store.req("\(sa).o_proj.scales"), oBi: store.req("\(sa).o_proj.biases"),
                qNorm: store.req("\(sa).q_norm.weight"), kNorm: store.req("\(sa).k_norm.weight"))
            let pf = (store.req("\(sa).q_norm.weight").dtype == .float32)
            guard let ro = RawMetalForward.attnLayerRaw(normed.reshaped([1, 1, H]), aw, promoteF32: pf) else { return nil }
            r = ro.asType(h.dtype)
        }
        let h2 = h + r
        guard let postNorm = RawMetalForward.rmsNorm(h2, store.req("\(p).post_attention_layernorm.weight"), eps: eps, D: H) else { return nil }
        let mp = "\(p).mlp"
        func tup(_ n: String) -> (MLXArray, MLXArray, MLXArray) { (store.req("\(n).weight"), store.req("\(n).scales"), store.req("\(n).biases")) }
        guard let mlpOut = RawMetalForward.moeRawForward(postNorm, gate: q("\(mp).gate", 8), sharedGate: q("\(mp).shared_expert_gate", 8),
            swG: tup("\(mp).switch_mlp.gate_proj"), swU: tup("\(mp).switch_mlp.up_proj"), swD: tup("\(mp).switch_mlp.down_proj"),
            shG: tup("\(mp).shared_expert.gate_proj"), shU: tup("\(mp).shared_expert.up_proj"), shD: tup("\(mp).shared_expert.down_proj"))
        else { return nil }
        return h2 + mlpOut
    }

    // ── SE full forward（task#4 速度）: GDN SE + MoE expert SE（resident buffer）。──
    nonisolated(unsafe) var gdnBufCache: [Int: RawMetalForward.GDNBuffers] = [:]
    nonisolated(unsafe) var moeBufCache: [Int: RawMetalForward.MoEBuffers] = [:]
    nonisolated(unsafe) var attnBufCache: [Int: RawMetalForward.AttnBuffers] = [:]

    /// SE decoder layer 1 層（GDN: SE / attn: round-trip raw / MoE: routing=MLX + expert SE + combine=MLX）。
    func seDecoderLayer(_ h: MLXArray, _ i: Int) -> MLXArray? {
        let p = "language_model.model.layers.\(i)", H = h.dim(-1)
        guard let normed = RawMetalForward.rmsNorm(h, store.req("\(p).input_layernorm.weight"), eps: eps, D: H) else { return nil }
        let r: MLXArray
        if isLinear(i) {
            let la = "\(p).linear_attn"
            if gdnBufCache[i] == nil {
                let rw = RawMetalForward.GDNRawWeights(
                    qkvWq: store.req("\(la).in_proj_qkv.weight"), qkvSc: store.req("\(la).in_proj_qkv.scales"), qkvBi: store.req("\(la).in_proj_qkv.biases"),
                    zWq: store.req("\(la).in_proj_z.weight"), zSc: store.req("\(la).in_proj_z.scales"), zBi: store.req("\(la).in_proj_z.biases"),
                    bWq: store.req("\(la).in_proj_b.weight"), bSc: store.req("\(la).in_proj_b.scales"), bBi: store.req("\(la).in_proj_b.biases"),
                    aWq: store.req("\(la).in_proj_a.weight"), aSc: store.req("\(la).in_proj_a.scales"), aBi: store.req("\(la).in_proj_a.biases"),
                    outWq: store.req("\(la).out_proj.weight"), outSc: store.req("\(la).out_proj.scales"), outBi: store.req("\(la).out_proj.biases"),
                    conv1dW: store.req("\(la).conv1d.weight").reshaped([8192, 4]).asType(.float32), normWeight: store.req("\(la).norm.weight"),
                    aLog: store.req("\(la).A_log"), dtBias: store.req("\(la).dt_bias"))
                gdnBufCache[i] = RawMetalForward.prepareGDNBuffers(rw, H: H)
            }
            guard let buf = gdnBufCache[i], let ro = RawMetalForward.gdnLayerSingleEncoder(normed.reshaped([1, 1, H]), buf, eps: eps) else { return nil }
            r = ro
        } else {
            let sa = "\(p).self_attn"
            if attnBufCache[i] == nil {
                let aw = RawMetalForward.AttnRawWeights(
                    qWq: store.req("\(sa).q_proj.weight"), qSc: store.req("\(sa).q_proj.scales"), qBi: store.req("\(sa).q_proj.biases"),
                    kWq: store.req("\(sa).k_proj.weight"), kSc: store.req("\(sa).k_proj.scales"), kBi: store.req("\(sa).k_proj.biases"),
                    vWq: store.req("\(sa).v_proj.weight"), vSc: store.req("\(sa).v_proj.scales"), vBi: store.req("\(sa).v_proj.biases"),
                    oWq: store.req("\(sa).o_proj.weight"), oSc: store.req("\(sa).o_proj.scales"), oBi: store.req("\(sa).o_proj.biases"),
                    qNorm: store.req("\(sa).q_norm.weight"), kNorm: store.req("\(sa).k_norm.weight"))
                attnBufCache[i] = RawMetalForward.prepareAttnBuffers(aw, H: H)
            }
            guard let abuf = attnBufCache[i], let ro = RawMetalForward.attnLayerSingleEncoder(normed.reshaped([1, 1, H]), abuf) else { return nil }
            r = ro.asType(h.dtype)
        }
        let h2 = h + r
        guard let postNorm = RawMetalForward.rmsNorm(h2, store.req("\(p).post_attention_layernorm.weight"), eps: eps, D: H) else { return nil }
        // MoE: routing(MLX) + expert SE + combine(MLX)
        let mp = "\(p).mlp"
        let gates = MLX.softmax(q("\(mp).gate", 8).apply(postNorm), axis: -1, precise: true)
        let order = MLX.argPartition(gates, kth: 256 - 8, axis: -1)
        let inds = order[0..., (256 - 8)...].asType(.int32)
        var scores = MLX.takeAlong(gates, inds, axis: -1)
        scores = scores / scores.sum(axis: -1, keepDims: true)
        if moeBufCache[i] == nil {
            func tup(_ n: String) -> (MLXArray, MLXArray, MLXArray) { (store.req("\(n).weight"), store.req("\(n).scales"), store.req("\(n).biases")) }
            moeBufCache[i] = RawMetalForward.prepareMoEBuffers(
                swG: tup("\(mp).switch_mlp.gate_proj"), swU: tup("\(mp).switch_mlp.up_proj"), swD: tup("\(mp).switch_mlp.down_proj"),
                shG: tup("\(mp).shared_expert.gate_proj"), shU: tup("\(mp).shared_expert.up_proj"), shD: tup("\(mp).shared_expert.down_proj"),
                Hin: H, I: store.req("\(mp).switch_mlp.gate_proj.weight").dim(-2), topK: 8)
        }
        guard let moeBuf = moeBufCache[i],
              let (d, sharedY) = RawMetalForward.moeExpertSingleEncoder(postNorm, inds.reshaped([8]), moeBuf) else { return nil }
        let y = (d * scores.reshaped([8, 1])).sum(axis: 0).reshaped([1, H])
        let gateScale = MLX.sigmoid(q("\(mp).shared_expert_gate", 8).apply(postNorm))
        return h2 + (y + gateScale * sharedY)
    }

    // ── 層全体融合 full forward（task#4）: residual stream を resident GPU buffer に保持 ──
    nonisolated(unsafe) var normWeightCache: [Int: RawMetalForward.NormWeightBuffers] = [:]
    nonisolated(unsafe) var hBuf: MTLBuffer?          // residual stream（H f16, 全40層常駐）
    nonisolated(unsafe) var postNormBuf: MTLBuffer?   // post_norm 出力（routing 用 readback）
    nonisolated(unsafe) var combinedBuf: MTLBuffer?   // MoE residual scratch

    /// GDN/attn mixer buffer を lazy 確保し、norm 重みも cache。戻り値: GDN なら true。
    func ensureFusedBuffers(_ i: Int, H: Int) -> Bool {
        let p = "language_model.model.layers.\(i)"
        if normWeightCache[i] == nil {
            normWeightCache[i] = RawMetalForward.prepareNormWeights(
                input: store.req("\(p).input_layernorm.weight"), post: store.req("\(p).post_attention_layernorm.weight"))
        }
        if moeBufCache[i] == nil {
            let mp = "\(p).mlp"
            func tup(_ n: String) -> (MLXArray, MLXArray, MLXArray) { (store.req("\(n).weight"), store.req("\(n).scales"), store.req("\(n).biases")) }
            moeBufCache[i] = RawMetalForward.prepareMoEBuffers(
                swG: tup("\(mp).switch_mlp.gate_proj"), swU: tup("\(mp).switch_mlp.up_proj"), swD: tup("\(mp).switch_mlp.down_proj"),
                shG: tup("\(mp).shared_expert.gate_proj"), shU: tup("\(mp).shared_expert.up_proj"), shD: tup("\(mp).shared_expert.down_proj"),
                Hin: H, I: store.req("\(mp).switch_mlp.gate_proj.weight").dim(-2), topK: 8)
        }
        if isLinear(i) {
            if gdnBufCache[i] == nil {
                let la = "\(p).linear_attn"
                let rw = RawMetalForward.GDNRawWeights(
                    qkvWq: store.req("\(la).in_proj_qkv.weight"), qkvSc: store.req("\(la).in_proj_qkv.scales"), qkvBi: store.req("\(la).in_proj_qkv.biases"),
                    zWq: store.req("\(la).in_proj_z.weight"), zSc: store.req("\(la).in_proj_z.scales"), zBi: store.req("\(la).in_proj_z.biases"),
                    bWq: store.req("\(la).in_proj_b.weight"), bSc: store.req("\(la).in_proj_b.scales"), bBi: store.req("\(la).in_proj_b.biases"),
                    aWq: store.req("\(la).in_proj_a.weight"), aSc: store.req("\(la).in_proj_a.scales"), aBi: store.req("\(la).in_proj_a.biases"),
                    outWq: store.req("\(la).out_proj.weight"), outSc: store.req("\(la).out_proj.scales"), outBi: store.req("\(la).out_proj.biases"),
                    conv1dW: store.req("\(la).conv1d.weight").reshaped([8192, 4]).asType(.float32), normWeight: store.req("\(la).norm.weight"),
                    aLog: store.req("\(la).A_log"), dtBias: store.req("\(la).dt_bias"))
                gdnBufCache[i] = RawMetalForward.prepareGDNBuffers(rw, H: H)
            }
            return true
        } else {
            if attnBufCache[i] == nil {
                let sa = "\(p).self_attn"
                let aw = RawMetalForward.AttnRawWeights(
                    qWq: store.req("\(sa).q_proj.weight"), qSc: store.req("\(sa).q_proj.scales"), qBi: store.req("\(sa).q_proj.biases"),
                    kWq: store.req("\(sa).k_proj.weight"), kSc: store.req("\(sa).k_proj.scales"), kBi: store.req("\(sa).k_proj.biases"),
                    vWq: store.req("\(sa).v_proj.weight"), vSc: store.req("\(sa).v_proj.scales"), vBi: store.req("\(sa).v_proj.biases"),
                    oWq: store.req("\(sa).o_proj.weight"), oSc: store.req("\(sa).o_proj.scales"), oBi: store.req("\(sa).o_proj.biases"),
                    qNorm: store.req("\(sa).q_norm.weight"), kNorm: store.req("\(sa).k_norm.weight"))
                attnBufCache[i] = RawMetalForward.prepareAttnBuffers(aw, H: H, maxLen: gpuMaxLen)
            }
            return false
        }
    }

    /// 層融合 1 層: mixer-half（input_norm+mixer+residual+post_norm を 1 encoder, hBuf 直更新）→
    /// routing(MLX) → expert SE → combine(MLX) → MoE residual(kernel)。hBuf は GPU 常駐のまま。
    func fusedDecoderLayer(_ i: Int, H: Int, pendingResid: MTLBuffer?) -> Bool {
        let isGDN = ensureFusedBuffers(i, H: H)
        guard let nw = normWeightCache[i], let hb = hBuf, let pnb = postNormBuf, let cb = combinedBuf else { return false }
        guard let postNorm = RawMetalForward.fusedMixerHalf(
            hBuf: hb, nw: nw, postNormBuf: pnb,
            gdn: isGDN ? gdnBufCache[i] : nil, attn: isGDN ? nil : attnBufCache[i], H: H, eps: eps,
            pendingResid: pendingResid) else { return false }
        // MoE: routing(MLX or Metal) + expert SE + combine(MLX) + residual(kernel)
        let p = "language_model.model.layers.\(i)", mp = "\(p).mlp"
        let inds: MLXArray, scores: MLXArray
        if RawMetalForward.metalRoute {
            // ★ Metal routing: gate qmm8(bit-exact) + route_top8。combine/shared は MLX のまま（誤差源分離）。
            guard let (ri, rs) = RawMetalForward.metalRouteGate(
                postNorm, gateW: store.req("\(mp).gate.weight"), gateS: store.req("\(mp).gate.scales"),
                gateB: store.req("\(mp).gate.biases"), H: H) else { return false }
            inds = ri.reshaped([1, 8]); scores = rs.asType(.float16).reshaped([1, 8])
        } else {
            let gates = MLX.softmax(q("\(mp).gate", 8).apply(postNorm), axis: -1, precise: true)
            let order = MLX.argPartition(gates, kth: 256 - 8, axis: -1)
            inds = order[0..., (256 - 8)...].asType(.int32)
            var sc = MLX.takeAlong(gates, inds, axis: -1)
            sc = sc / sc.sum(axis: -1, keepDims: true)
            scores = sc
        }
        if moeBufCache[i] == nil {
            func tup(_ n: String) -> (MLXArray, MLXArray, MLXArray) { (store.req("\(n).weight"), store.req("\(n).scales"), store.req("\(n).biases")) }
            moeBufCache[i] = RawMetalForward.prepareMoEBuffers(
                swG: tup("\(mp).switch_mlp.gate_proj"), swU: tup("\(mp).switch_mlp.up_proj"), swD: tup("\(mp).switch_mlp.down_proj"),
                shG: tup("\(mp).shared_expert.gate_proj"), shU: tup("\(mp).shared_expert.up_proj"), shD: tup("\(mp).shared_expert.down_proj"),
                Hin: H, I: store.req("\(mp).switch_mlp.gate_proj.weight").dim(-2), topK: 8)
        }
        guard let moeBuf = moeBufCache[i],
              let (d, sharedY) = RawMetalForward.moeExpertSingleEncoder(postNorm, inds.reshaped([8]), moeBuf) else { return false }
        let y = (d * scores.reshaped([8, 1])).sum(axis: 0).reshaped([1, H])
        let gateScale = MLX.sigmoid(q("\(mp).shared_expert_gate", 8).apply(postNorm))
        let combined = y + gateScale * sharedY
        combined.eval()
        RawMetalForward.writeBuffer(cb, combined, H)   // combinedBuf に保存→次層 mixer 先頭で hBuf に畳む
        return true
    }

    /// ★ 層融合 full forward: residual stream を hBuf に常駐し全40層で MLXArray 往復を排除。decode T=1。
    public func fusedRawForward(_ ids: MLXArray) -> MLXArray? {
        let e = embed(ids); let H = e.dim(-1)
        if hBuf == nil { hBuf = RawMetalForward.makeResidentBuffer(H * 2) }
        if postNormBuf == nil { postNormBuf = RawMetalForward.makeResidentBuffer(H * 2) }
        if combinedBuf == nil { combinedBuf = RawMetalForward.makeResidentBuffer(H * 2) }
        guard let hb = hBuf, let cb = combinedBuf else { return nil }
        RawMetalForward.writeBuffer(hb, e, H)                      // embed → hBuf
        // 各層 MoE residual は次層 mixer encoder 先頭に畳む（pendingResid）。初層は無し。
        for i in 0 ..< numLayers {
            if !fusedDecoderLayer(i, H: H, pendingResid: i == 0 ? nil : cb) { return nil }
        }
        RawMetalForward.fusedMoEResidual(hBuf: hb, combinedBuf: cb, H: H)  // 最終層 MoE residual を flush
        let h = RawMetalForward.readBuffer(hb, H)                  // hBuf → MLXArray（最後だけ readback）
        guard let fn = RawMetalForward.rmsNorm(h, store.req("language_model.model.norm.weight"), eps: eps, D: H) else { return nil }
        return headProj().apply(fn.reshaped([1, 1, H]))
    }

    // ── all-GPU 多層 1-CB forward（task#8: routing/combine も Metal＝MLX 非依存, 層間 sync 排除）──
    nonisolated(unsafe) var gate8Cache: [Int: RawMetalForward.Gate8Buffers] = [:]
    nonisolated(unsafe) var sharedGate8Cache: [Int: RawMetalForward.Gate8Buffers] = [:]
    nonisolated(unsafe) var gpuScratch: RawMetalForward.GPUScratch?
    nonisolated(unsafe) var gpuWarmed = false
    nonisolated(unsafe) var gpuLayers: [RawMetalForward.GPULayer]?
    nonisolated(unsafe) var gpuMaxLen = 2048      // decode KV cache 最大長（prompt+生成）
    nonisolated(unsafe) var finalNormBuf: MTLBuffer?   // final norm weight（GPU CB 同梱用）

    func ensureFinalNorm() -> MTLBuffer? {
        if finalNormBuf == nil {
            finalNormBuf = RawMetalForward.f16Buffer(store.req("language_model.model.norm.weight"))
        }
        return finalNormBuf
    }

    /// 全層 GPU buffer を ensure し [GPULayer] を構築（初回のみ, cache）。
    func buildGPULayers(_ ids: MLXArray, _ H: Int) -> [RawMetalForward.GPULayer]? {
        if !gpuWarmed { _ = rawForward(ids)?.eval(); _ = RawMetalForward.compileGqmmSwiglu(); gpuWarmed = true }   // 標準 + 融合 kernel compile
        if let cached = gpuLayers { return cached }
        var layers: [RawMetalForward.GPULayer] = []
        for i in 0 ..< numLayers {
            let isGDN = ensureFusedBuffers(i, H: H)
            let mp = "language_model.model.layers.\(i).mlp"
            if gate8Cache[i] == nil {
                gate8Cache[i] = RawMetalForward.prepareGate8(store.req("\(mp).gate.weight"), store.req("\(mp).gate.scales"), store.req("\(mp).gate.biases"))
            }
            if sharedGate8Cache[i] == nil {
                sharedGate8Cache[i] = RawMetalForward.prepareGate8(store.req("\(mp).shared_expert_gate.weight"), store.req("\(mp).shared_expert_gate.scales"), store.req("\(mp).shared_expert_gate.biases"))
            }
            guard let nw = normWeightCache[i], let moe = moeBufCache[i],
                  let g8 = gate8Cache[i], let sg8 = sharedGate8Cache[i] else { return nil }
            layers.append(RawMetalForward.GPULayer(
                nw: nw, gdn: isGDN ? gdnBufCache[i] : nil, attn: isGDN ? nil : attnBufCache[i],
                moe: moe, gate: g8, sharedGate: sg8))
        }
        gpuLayers = layers
        return layers
    }

    /// GDN conv cache / recurrent state を 0 リセット（新シーケンス開始時）。KV cache は write-before-read で不要。
    func resetGPUState() {
        for (_, b) in gdnBufCache {
            memset(b.convInput.contents(), 0, b.convKernel * b.convDim * 2)
            memset(b.stateBuf.contents(), 0, b.Hv * b.Dv * b.Dk * 4)
        }
    }

    /// ★ all-GPU forward: embed/final norm/lm_head 以外を全 Metal、全40層を単一 command buffer で実行。
    /// routing(qmm8+route_top8) は near-tie で lossless 検証済(route-decode-lossless)。cold state T=1（benchmark）。
    public func fusedRawForwardGPU(_ ids: MLXArray) -> MLXArray? {
        let e = embed(ids); let H = e.dim(-1)
        guard let layers = buildGPULayers(ids, H) else { return nil }
        if gpuScratch == nil { gpuScratch = RawMetalForward.makeGPUScratch(H: H, E: 256, K: 8) }
        if hBuf == nil { hBuf = RawMetalForward.makeResidentBuffer(H * 2) }
        guard let hb = hBuf, let sc = gpuScratch else { return nil }
        RawMetalForward.writeBuffer(hb, e, H)
        RawMetalForward.fusedForwardGPU(hBuf: hb, layers: layers, scratch: sc, H: H, E: 256, K: 8, eps: eps,
                                        finalNormW: ensureFinalNorm())
        let fn = RawMetalForward.readBuffer(sc.normed, H)   // final norm 同梱済（CB 内）
        return headProj().apply(fn.reshaped([1, 1, H]))
    }

    /// ★ all-GPU **decode 1 step**: token を pos に投入（GDN state feedback + KV cache）。logits[1,1,vocab]。
    /// 事前に buildGPULayers + resetGPUState（シーケンス先頭）が必要。
    public func fusedDecodeStepGPU(_ tokenId: Int32, pos: Int, H: Int) -> MLXArray? {
        guard let layers = gpuLayers else { return nil }
        if gpuScratch == nil { gpuScratch = RawMetalForward.makeGPUScratch(H: H, E: 256, K: 8) }
        if hBuf == nil { hBuf = RawMetalForward.makeResidentBuffer(H * 2) }
        guard let hb = hBuf, let sc = gpuScratch else { return nil }
        let e = embed(MLXArray([tokenId], [1, 1]))
        RawMetalForward.writeBuffer(hb, e, H)
        RawMetalForward.fusedForwardGPU(hBuf: hb, layers: layers, scratch: sc, H: H, E: 256, K: 8, eps: eps,
                                        decode: true, pos: pos, finalNormW: ensureFinalNorm())
        let fn = RawMetalForward.readBuffer(sc.normed, H)   // final norm 同梱済（CB 内）
        return headProj().apply(fn.reshaped([1, 1, H]))
    }

    /// SE full forward（GDN/MoE は single-encoder, attn は round-trip）。decode T=1。
    public func seRawForward(_ ids: MLXArray) -> MLXArray? {
        let e = embed(ids); let H = e.dim(-1)
        var h = e.reshaped([1, H])
        for i in 0 ..< numLayers { guard let h2 = seDecoderLayer(h, i) else { return nil }; h = h2 }
        guard let fn = RawMetalForward.rmsNorm(h, store.req("language_model.model.norm.weight"), eps: eps, D: H) else { return nil }
        return headProj().apply(fn.reshaped([1, 1, H]))
    }

    /// raw full forward（embed=MLX, 40 層=raw, final norm=raw, lm_head=MLX）。ids[1,1]→logits[1,1,vocab]。
    public func rawForward(_ ids: MLXArray) -> MLXArray? {
        let e = embed(ids); let H = e.dim(-1)
        var h = e.reshaped([1, H])                                          // T=1
        for i in 0 ..< numLayers { guard let h2 = rawDecoderLayer(h, i) else { return nil }; h = h2 }
        guard let fn = RawMetalForward.rmsNorm(h, store.req("language_model.model.norm.weight"), eps: eps, D: H) else { return nil }
        return headProj().apply(fn.reshaped([1, 1, H]))
    }

    /// 検証: raw full forward(40層 raw) vs MLX full forward の logits（decode T=1）。
    public static func runRawFullForward(modelDir: String) throws -> String {
        let store = try WeightStore(modelDir: modelDir)
        let model = QwispModel(store: store)
        let prevF32 = GatedDeltaNetLayer.f32Conv; GatedDeltaNetLayer.f32Conv = true   // raw conv は f32 累積＝MLX を f32Conv に合わせる
        defer { GatedDeltaNetLayer.f32Conv = prevF32 }
        let ids = MLXArray([Int32(100)], [1, 1])
        let ref = model(ids); ref.eval()                                   // MLX forward(f16, f32Conv)
        guard let got = model.rawForward(ids) else { return "[raw-full] rawForward 失敗" }
        got.eval()
        let rf = ref.reshaped([ref.size]), gf = got.reshaped([got.size])
        let d = MLX.max(MLX.abs(gf.asType(.float32) - rf.asType(.float32))).item(Float.self)
        let rel = d / (MLX.max(MLX.abs(rf.asType(.float32))).item(Float.self) + 1e-9)
        let amR = MLX.argMax(rf).item(Int.self), amG = MLX.argMax(gf).item(Int.self)
        var out = String(format: "[raw-full-forward] raw 40層 full forward vs MLX (decode T=1)\n"
            + "  logits rel=%.3e  argmax raw=%d ref=%d %@  %@",
            rel, amG, amR, amG == amR ? "一致✅" : "不一致❌",
            rel == 0 ? "TRUE bit-exact ✅✅" : (rel < 1e-3 ? "△ near" : "❌ f16累積"))
        // 時間計測: round-trip raw forward vs MLX full forward
        func now() -> Double { Double(DispatchTime.now().uptimeNanoseconds) / 1e6 }
        let reps = 20
        for _ in 0..<3 { _ = model.rawForward(ids)?.eval() }
        var t0 = now(); for _ in 0..<reps { _ = model.rawForward(ids)?.eval() }; let rawMs = (now()-t0)/Double(reps)
        for _ in 0..<3 { model(ids).eval() }
        t0 = now(); for _ in 0..<reps { model(ids).eval() }; let mlxMs = (now()-t0)/Double(reps)
        out += String(format: "\n  時間/forward: round-trip raw=%.1fms(%.1f tok/s) | MLX=%.1fms(%.1f tok/s) → %.2fx",
                      rawMs, 1000/rawMs, mlxMs, 1000/mlxMs, mlxMs/Swift.max(0.01, rawMs))
        // SE vs round-trip 層別診断（同一 h で seDecoderLayer vs rawDecoderLayer）
        if ProcessInfo.processInfo.environment["QWISP_SE_DIAG"] == "1" {
            var hh = model.embed(ids); let H = hh.dim(-1)
            for i in 0 ..< model.numLayers {
                if let se = model.seDecoderLayer(hh.reshaped([1, H]), i), let rt = model.rawDecoderLayer(hh.reshaped([1, H]), i) {
                    se.eval(); rt.eval()
                    let lr = MLX.max(MLX.abs(se.reshaped([H]).asType(.float32) - rt.reshaped([H]).asType(.float32))).item(Float.self)
                       / (MLX.max(MLX.abs(rt.asType(.float32))).item(Float.self) + 1e-9)
                    if lr > 1e-6 { out += String(format: "\n   SE-vs-RT layer %d (%@): rel=%.3e", i, model.isLinear(i) ? "GDN" : "attn", lr) }
                    hh = rt.reshaped([1, H])
                }
            }
        }
        // ★ SE full forward（GDN/MoE single-encoder, attn round-trip）: bit-exact + tok/s
        if let se = model.seRawForward(ids) {
            se.eval()
            let sf = se.reshaped([se.size])
            let sd = MLX.max(MLX.abs(sf.asType(.float32) - rf.asType(.float32))).item(Float.self)
            let srel = sd / (MLX.max(MLX.abs(rf.asType(.float32))).item(Float.self) + 1e-9)
            let amS = MLX.argMax(sf).item(Int.self)
            for _ in 0..<3 { _ = model.seRawForward(ids)?.eval() }
            t0 = now(); for _ in 0..<reps { _ = model.seRawForward(ids)?.eval() }; let seMs = (now()-t0)/Double(reps)
            out += String(format: "\n  ── SE full forward（GDN/attn/MoE=single-encoder）──\n   logits rel=%.3e argmax %d(ref %d)%@  時間=%.1fms(%.1f tok/s) → vs MLX %.2fx",
                          srel, amS, amR, amS == amR ? "✅" : "❌", seMs, 1000/seMs, mlxMs/Swift.max(0.01, seMs))
        }
        // ★★ 層融合 full forward（residual stream を hBuf 常駐, input_norm+mixer+residual+post_norm を 1 encoder）
        // routing=MLX（安全）と routing=Metal（task#8 検証）の両方を計測し rel/argmax を比較。
        func measureFused(_ label: String) {
            guard let fu = model.fusedRawForward(ids) else { return }
            fu.eval()
            let ff = fu.reshaped([fu.size])
            let fd = MLX.max(MLX.abs(ff.asType(.float32) - rf.asType(.float32))).item(Float.self)
            let frel = fd / (MLX.max(MLX.abs(rf.asType(.float32))).item(Float.self) + 1e-9)
            let amF = MLX.argMax(ff).item(Int.self)
            for _ in 0..<3 { _ = model.fusedRawForward(ids)?.eval() }
            t0 = now(); for _ in 0..<reps { _ = model.fusedRawForward(ids)?.eval() }; let fuMs = (now()-t0)/Double(reps)
            out += String(format: "\n  ── 層融合 full forward（%@）──\n   logits rel=%.3e argmax %d(ref %d)%@  時間=%.1fms(%.1f tok/s) → vs MLX %.2fx",
                          label, frel, amF, amR, amF == amR ? "✅" : "❌", fuMs, 1000/fuMs, mlxMs/Swift.max(0.01, fuMs))
        }
        RawMetalForward.metalRoute = false; measureFused("routing=MLX, norm/residual=kernel")
        // per-layer routing 逸脱（全40層, K入力）: Metal routing が生む combined の rel 分布。
        if ProcessInfo.processInfo.environment["QWISP_ROUTE_DIAG"] == "1" {
            let H = model.embed(ids).dim(-1)
            var worstSet = true; var maxScoreDiff: Float = 0; var maxCombRel: Float = 0; var nbad = 0
            for i in 0 ..< model.numLayers {
                let mp = "language_model.model.layers.\(i).mlp"
                for _ in 0 ..< 8 {
                    let pn = MLXRandom.normal([1, H]).asType(.float16); pn.eval()
                    // MLX routing
                    let gates = MLX.softmax(model.q("\(mp).gate", 8).apply(pn), axis: -1, precise: true)
                    let order = MLX.argPartition(gates, kth: 256 - 8, axis: -1)
                    let indsM = order[0..., (256 - 8)...].asType(.int32)
                    var scM = MLX.takeAlong(gates, indsM, axis: -1); scM = scM / scM.sum(axis: -1, keepDims: true)
                    // Metal routing
                    guard let (indsR, scR) = RawMetalForward.metalRouteGate(pn, gateW: model.store.req("\(mp).gate.weight"),
                        gateS: model.store.req("\(mp).gate.scales"), gateB: model.store.req("\(mp).gate.biases"), H: H) else { continue }
                    let imA = indsM.reshaped([8]).asArray(Int32.self), irA = indsR.asArray(Int32.self)
                    var rmap: [Int32: Float] = [:]; let smA = scM.reshaped([8]).asType(.float32).asArray(Float.self)
                    for k in 0..<8 { rmap[imA[k]] = smA[k] }
                    let srA = scR.asType(.float32).asArray(Float.self)
                    for k in 0..<8 { if let rs = rmap[irA[k]] { maxScoreDiff = max(maxScoreDiff, abs(rs - srA[k])) } else { worstSet = false; nbad += 1 } }
                }
            }
            out += String(format: "\n   [route-diag 全40層×8入力] expert集合%@(不一致%d) score最大差=%.3e", worstSet ? "全一致✅" : "❌", nbad, maxScoreDiff)
            _ = maxCombRel
        }
        RawMetalForward.metalRoute = true; measureFused("routing=Metal(qmm8+top8), task#8")
        RawMetalForward.metalRoute = false
        // ★★ all-GPU 多層 1-CB forward（routing/combine 全 Metal, 層間 sync 排除）
        if let gpu = model.fusedRawForwardGPU(ids) {
            gpu.eval()
            let gv = gpu.reshaped([gpu.size])
            let gd = MLX.max(MLX.abs(gv.asType(.float32) - rf.asType(.float32))).item(Float.self)
            let grel = gd / (MLX.max(MLX.abs(rf.asType(.float32))).item(Float.self) + 1e-9)
            let amG2 = MLX.argMax(gv).item(Int.self)
            for _ in 0..<3 { _ = model.fusedRawForwardGPU(ids)?.eval() }
            t0 = now(); for _ in 0..<reps { _ = model.fusedRawForwardGPU(ids)?.eval() }; let gMs = (now()-t0)/Double(reps)
            out += String(format: "\n  ── ★all-GPU 多層 1-CB forward（routing/combine 全 Metal, 層間 sync 排除）──\n   logits rel=%.3e argmax %d(ref %d)%@  時間=%.1fms(%.1f tok/s) → vs MLX %.2fx",
                          grel, amG2, amR, amG2 == amR ? "✅" : "❌", gMs, 1000/gMs, mlxMs/Swift.max(0.01, gMs))
        }
        out += "\n   ※注: 上の hidden rel(0.13)は lossless 指標でない（1 境界 expert flip で hidden が膨張するが argmax は別物）。"
            + "\n     プロジェクト基準(near-tie rank≤2 除く argmax 一致)での判定は raw-route-lossless を参照"
            + "\n     →routing 単独の不一致は near-tie 支配(GDN drift と同クラス)。多層融合は未閉鎖。"
        // 層別診断: 同一 h(MLX 経路)を raw layer i と MLX layer i に入れ in-context per-layer rel を見る。
        if ProcessInfo.processInfo.environment["QWISP_FULL_DIAG"] == "1" {
            var hM = model.embed(ids); let H = hM.dim(-1)
            var worst = 0; var worstRel: Float = 0
            for i in 0 ..< model.numLayers {
                let mlxOut = model.layers[i](hM); mlxOut.eval()
                if let rawOut = model.rawDecoderLayer(hM.reshaped([1, H]), i) {
                    rawOut.eval()
                    let lr = MLX.max(MLX.abs(rawOut.reshaped([H]).asType(.float32) - mlxOut.reshaped([H]).asType(.float32))).item(Float.self)
                       / (MLX.max(MLX.abs(mlxOut.asType(.float32))).item(Float.self) + 1e-9)
                    if lr > worstRel { worstRel = lr; worst = i }
                    if lr > 1e-5 { out += String(format: "\n   layer %d (%@): in-context rel=%.3e", i, model.isLinear(i) ? "GDN" : "attn", lr) }
                }
                hM = mlxOut                                                 // MLX 経路を進める(共通入力)
            }
            out += String(format: "\n   worst layer=%d rel=%.3e", worst, worstRel)
        }
        return out
    }

    /// ★ task#8 再検証: routing の lossless 性を **プロジェクト基準（near-tie rank≤2 を除く argmax 一致）** で判定。
    /// 多数の入力 token を sweep し、fused(MLX-routing) と fused(Metal-routing) の予測 argmax を MLX 参照と比較、
    /// 不一致を refRank で分類。Metal routing が MLX-routing(=GDN drift のみ)を超える **真の乖離(rank≫2)** を
    /// 出すかが争点。出さなければ「routing も near-tie 同クラス＝プロジェクト基準で lossless」。
    /// env QWISP_RUN=raw-route-lossless / QWISP_GEN(sweep 数, 既定 96)。
    public static func runRouteLossless(modelDir: String) throws -> String {
        let store = try WeightStore(modelDir: modelDir)
        let model = QwispModel(store: store)
        let prevF32 = GatedDeltaNetLayer.f32Conv; GatedDeltaNetLayer.f32Conv = true
        defer { GatedDeltaNetLayer.f32Conv = prevF32 }
        let N = ProcessInfo.processInfo.environment["QWISP_GEN"].flatMap { Int($0) } ?? 96
        // sweep 用 input token（語彙から散らす）。各々 cold-state T=1 forward = 全40層の routing を実行。
        let vocab = store.req("language_model.model.embed_tokens.weight").dim(0)
        func classify(_ label: String, metal: Bool) -> String {
            RawMetalForward.metalRoute = metal
            var match = 0, nearTie = 0, trueDiv = 0
            var worstGap: Float = 0; var examples: [String] = []
            for t in 0 ..< N {
                let tid = Int32((t * 9973 + 17) % vocab)            // 決定的に散らす
                let ids = MLXArray([tid], [1, 1])
                let ref = model(ids); ref.eval()
                let rv = ref.reshaped([ref.size])
                guard let got = model.fusedRawForward(ids) else { continue }
                got.eval()
                let amR = MLX.argMax(rv).item(Int.self)
                let amG = MLX.argMax(got.reshaped([got.size])).item(Int.self)
                if amG == amR { match += 1; continue }
                // 不一致: amG(予測)の参照 logit 内 rank と top1 との gap
                let gLogit = rv[amG].item(Float.self)
                let top1 = MLX.max(rv).item(Float.self)
                let rank = MLX.sum(rv .> gLogit).item(Int.self) + 1     // 1-indexed
                let gap = top1 - gLogit
                if rank <= 2 { nearTie += 1 } else { trueDiv += 1 }
                worstGap = Swift.max(worstGap, rank > 2 ? gap : 0)
                if examples.count < 6 { examples.append(String(format: "t%d:rank%d(gap%.3f)", t, rank, gap)) }
            }
            return String(format: "  [%@] argmax一致 %d/%d=%.1f%%  不一致=%d (near-tie rank≤2: %d, 真の乖離 rank>2: %d, 最大gap %.3f)\n    例: %@",
                          label, match, N, Double(match)/Double(N)*100, N - match, nearTie, trueDiv, worstGap,
                          examples.isEmpty ? "(none)" : examples.joined(separator: " | "))
        }
        // warm pipelines（rawForward が standalone kernel を compile）。
        _ = model.rawForward(MLXArray([Int32(1)], [1, 1]))?.eval()
        var out = "[raw-route-lossless] routing lossless 再検証（プロジェクト基準=near-tie 除く argmax 一致, sweep \(N)）\n"
        out += "  ※両 routing とも fused mixer の GDN 1-ULP drift(2.97e-3) を共有。GDN drift 自体も token を flip しうる。\n"
        out += classify("routing=MLX  (GDN drift のみ)", metal: false) + "\n"
        out += classify("routing=Metal(qmm8+top8)   ", metal: true) + "\n"
        // all-GPU path（combine/shared_gate8 kernel 込み）の argmax を MLX 参照と rank 分類。
        do {
            var match = 0, near = 0, tru = 0; var ex: [String] = []
            for t in 0 ..< N {
                let tid = Int32((t * 9973 + 17) % vocab); let ids = MLXArray([tid], [1, 1])
                let ref = model(ids); ref.eval(); let rv = ref.reshaped([ref.size])
                guard let got = model.fusedRawForwardGPU(ids) else { continue }; got.eval()
                let amR = MLX.argMax(rv).item(Int.self), amG = MLX.argMax(got.reshaped([got.size])).item(Int.self)
                if amG == amR { match += 1; continue }
                let rank = MLX.sum(rv .> rv[amG]).item(Int.self)
                if rank <= 2 { near += 1 } else { tru += 1 }
                if ex.count < 6 { ex.append("t\(t):rank\(rank)") }
            }
            out += String(format: "  [all-GPU 1-CB] argmax一致 %d/%d=%.1f%% 不一致=%d (near-tie %d, 真の乖離 %d) %@\n",
                          match, N, Double(match)/Double(N)*100, N - match, near, tru, ex.isEmpty ? "" : ex.joined(separator: " "))
        }
        // ★ routing 単独の影響を分離: fused(MLX) vs fused(Metal) を直接比較（GDN drift は両者共通で相殺）。
        var agree = 0, rtNear = 0, rtTrue = 0; var rtEx: [String] = []
        for t in 0 ..< N {
            let tid = Int32((t * 9973 + 17) % vocab); let ids = MLXArray([tid], [1, 1])
            RawMetalForward.metalRoute = false; guard let fM = model.fusedRawForward(ids) else { continue }; fM.eval()
            RawMetalForward.metalRoute = true;  guard let fX = model.fusedRawForward(ids) else { continue }; fX.eval()
            RawMetalForward.metalRoute = false
            let fMv = fM.reshaped([fM.size])
            let amM = MLX.argMax(fMv).item(Int.self)
            let amX = MLX.argMax(fX.reshaped([fX.size])).item(Int.self)
            if amM == amX { agree += 1; continue }
            // fused(MLX) を基準に fused(Metal) の予測 rank（routing だけが原因の flip）
            let xLogit = fMv[amX].item(Float.self); let top1 = MLX.max(fMv).item(Float.self)
            let rank = MLX.sum(fMv .> xLogit).item(Int.self) + 1
            if rank <= 2 { rtNear += 1 } else { rtTrue += 1 }
            if rtEx.count < 6 { rtEx.append(String(format: "t%d:rank%d(gap%.3f)", t, rank, top1 - xLogit)) }
        }
        out += String(format: "  [routing 単独分離: fused(MLX) vs fused(Metal)] 一致 %d/%d=%.1f%%  不一致=%d (near-tie: %d, 真の乖離 rank>2: %d)\n    例: %@",
                      agree, N, Double(agree)/Double(N)*100, N - agree, rtNear, rtTrue, rtEx.isEmpty ? "(完全一致)" : rtEx.joined(separator: " | "))
        RawMetalForward.metalRoute = false
        out += "\n  → 判定: routing 単独の不一致が全て near-tie(rank≤2) なら、Metal routing は GDN drift と同クラス＝プロジェクト基準で lossless。真の乖離が出れば routing 固有の問題。"
        return out
    }

    /// ★ task#8 本筋検証: **teacher-forced 多トークン decode で Metal routing の lossless 性を rank 判定**。
    /// 参照=MLX-routing greedy（エンジン自身）, テスト=Metal-routing teacher-forced。measureMLXFidelity と同方法。
    /// 不一致を refRank で分類（≤2=near-tie=許容, >2=真の乖離）。これが本プロジェクト基準の lossless 判定。
    /// env QWISP_RUN=route-decode-lossless / QWISP_GEN(decode 数, 既定 64), refPath=spec_prompt 入り(qwisp_long_ref 等)。
    public static func runRouteDecodeLossless(modelDir: String, refPath: String) throws -> String {
        let store = try WeightStore(modelDir: modelDir)
        let model = QwispModel(store: store)
        guard let r = try? loadArrays(url: URL(fileURLWithPath: refPath)), let pa = r["spec_prompt"] else {
            return "[route-decode-lossless] skip: spec_prompt 無し \(refPath)（QWISP_MTP_REF に qwisp_long_ref.safetensors 等を指定）"
        }
        let promptIds = pa.asType(.int32).reshaped([1, pa.dim(0)]); let T = promptIds.dim(1)
        // 参照 = mlx_lm greedy（spec_greedy）。両 routing を**同一外部参照**に teacher-force し fidelity を公平比較。
        guard let gRefArr = r["spec_greedy"] else { return "[route-decode-lossless] spec_greedy 無し" }
        let gR = gRefArr.asArray(Int32.self).map { Int($0) }
        let N = Swift.min(ProcessInfo.processInfo.environment["QWISP_GEN"].flatMap { Int($0) } ?? 96, gR.count)
        defer { RawMetalForward.metalRoute = false }
        // 各 routing mode を mlx_lm 参照に teacher-force（measureMLXFidelity と同方法）。
        func fidelity(_ metal: Bool) -> (match: Int, near: Int, tru: Int, ex: [String]) {
            RawMetalForward.metalRoute = metal
            let c = model.makeCaches()
            var match = 0, near = 0, tru = 0; var ex: [String] = []
            func record(_ i: Int, _ v: MLXArray) {
                let pred = MLX.argMax(v, axis: -1).item(Int.self)
                if pred == gR[i] { match += 1; return }
                let refLogit = v[gR[i]].item(Float.self), top1 = MLX.max(v).item(Float.self)
                let rank = MLX.sum(v .> refLogit).item(Int.self)        // 0-indexed: 参照 token が test 内で何位
                if rank <= 2 { near += 1 } else { tru += 1 }
                if ex.count < 8 { ex.append(String(format: "p%d:pred=%d ref=%d(rank%d,gap%.2f)", i, pred, gR[i], rank, top1 - refLogit)) }
            }
            var lg = model(promptIds, caches: c)
            var v = lg[0, T - 1]; record(0, v); MLX.eval([v] + c.flatMap { $0.stateArrays })
            for i in 0 ..< (N - 1) {
                lg = model(MLXArray([Int32(gR[i])], [1, 1]), caches: c)
                v = lg[0, 0]; record(i + 1, v); MLX.eval([v] + c.flatMap { $0.stateArrays })
            }
            return (match, near, tru, ex)
        }
        let mlx = fidelity(false)   // MLX routing（既存エンジンの基準 fidelity）
        let met = fidelity(true)    // Metal routing
        RawMetalForward.metalRoute = false
        func line(_ lbl: String, _ f: (match: Int, near: Int, tru: Int, ex: [String])) -> String {
            String(format: "  [%@] %d/%d=%.1f%%  不一致%d（near-tie %d, 真の乖離 %d）%@", lbl, f.match, N,
                   Double(f.match)/Double(N)*100, N - f.match, f.near, f.tru,
                   f.ex.isEmpty ? "" : "\n      " + f.ex.joined(separator: " | "))
        }
        return """
            [route-decode-lossless] teacher-forced vs mlx_lm(spec_greedy), prompt T=\(T), decode \(N)
              プロジェクト基準（measureMLXFidelity 同方法, near-tie refRank≤2 許容）。両 routing を同一参照に比較:
            \(line("routing=MLX  ", mlx))
            \(line("routing=Metal", met))
              → Metal の『真の乖離』が MLX と同数なら、routing 差は fidelity を悪化させず＝プロジェクト基準で lossless 同等（多層融合 GO）。
            """
    }

    /// ★ task#9: all-GPU decode path（GDN state feedback + KV cache）の lossless 検証 + tok/s。
    /// teacher-forced で mlx_lm(spec_greedy)参照に rank 比較。env QWISP_RUN=decode-gpu / QWISP_GEN。
    public static func runDecodeGPULossless(modelDir: String, refPath: String) throws -> String {
        let store = try WeightStore(modelDir: modelDir)
        let model = QwispModel(store: store)
        guard let r = try? loadArrays(url: URL(fileURLWithPath: refPath)),
              let pa = r["spec_prompt"], let gRefArr = r["spec_greedy"] else {
            return "[decode-gpu] skip: spec_prompt/spec_greedy 無し \(refPath)"
        }
        let prompt = pa.asType(.int32).asArray(Int32.self)
        let gR = gRefArr.asArray(Int32.self).map { Int($0) }
        let N = Swift.min(ProcessInfo.processInfo.environment["QWISP_GEN"].flatMap { Int($0) } ?? 64, gR.count)
        let H = model.embed(MLXArray([Int32(0)], [1, 1])).dim(-1)
        guard model.buildGPULayers(MLXArray([Int32(1)], [1, 1]), H) != nil else { return "[decode-gpu] build 失敗" }
        // 1. teacher-forced fidelity（vs mlx_lm）
        model.resetGPUState()
        var pos = 0; var v: MLXArray? = nil
        for t in prompt { v = model.fusedDecodeStepGPU(t, pos: pos, H: H); pos += 1 }
        var match = 0, near = 0, tru = 0; var ex: [String] = []
        func record(_ i: Int, _ lg: MLXArray) {
            let vv = lg.reshaped([lg.size]); let pred = MLX.argMax(vv).item(Int.self)
            if pred == gR[i] { match += 1; return }
            let rank = MLX.sum(vv .> vv[gR[i]]).item(Int.self)
            if rank <= 2 { near += 1 } else { tru += 1 }
            if ex.count < 8 { ex.append("p\(i):pred=\(pred) ref=\(gR[i])(rank\(rank))") }
        }
        if let v0 = v { record(0, v0) }
        for i in 0 ..< (N - 1) {
            guard let lg = model.fusedDecodeStepGPU(Int32(gR[i]), pos: pos, H: H) else { break }
            lg.eval(); pos += 1; record(i + 1, lg)
        }
        // 2. tok/s（free-running greedy 32 step, prompt 既 prefill 済の state を作り直して計測）
        model.resetGPUState(); pos = 0; var last: MLXArray? = nil
        for t in prompt { last = model.fusedDecodeStepGPU(t, pos: pos, H: H); pos += 1 }
        last?.eval()
        var cur = Int32(MLX.argMax(last!.reshaped([last!.size])).item(Int.self))
        let t0 = DispatchTime.now()
        let steps = 32
        for _ in 0 ..< steps {
            guard let lg = model.fusedDecodeStepGPU(cur, pos: pos, H: H) else { break }
            cur = Int32(MLX.argMax(lg.reshaped([lg.size])).item(Int.self)); pos += 1
        }
        let secs = Double(DispatchTime.now().uptimeNanoseconds - t0.uptimeNanoseconds) / 1e9
        let tokps = Double(steps) / secs
        // ── decode step glue 分解（embed / forward CB / lm_head+argmax）──
        func nowMs() -> Double { Double(DispatchTime.now().uptimeNanoseconds) / 1e6 }
        var tEmbed = 0.0, tFwd = 0.0, tHead = 0.0, gpuExec = 0.0; let pr = 16
        for _ in 0 ..< pr {
            var s = nowMs(); let e = model.embed(MLXArray([cur], [1, 1])); e.eval(); tEmbed += nowMs()-s
            s = nowMs(); _ = model.fusedDecodeStepGPU(cur, pos: pos, H: H); tFwd += nowMs()-s; gpuExec += RawMetalForward.lastGPUExecMs
            let fn = RawMetalForward.readBuffer(model.gpuScratch!.normed, H)
            s = nowMs(); let lg = model.headProj().apply(fn.reshaped([1,1,H])); let am = MLX.argMax(lg.reshaped([lg.size])).item(Int.self); tHead += nowMs()-s; _ = am
            pos += 1
        }
        return String(format: """
            [decode-gpu] all-GPU decode（GDN state feedback + KV cache）prompt T=\(prompt.count), N=\(N)
              lossless（teacher-forced vs mlx_lm, near-tie rank≤2 許容）: argmax %d/%d=%.1f%% 不一致%d（near-tie %d, 真の乖離 %d）
              %@
              tok/s（free-running greedy 32 step）: %.1f tok/s (%.1f ms/tok)
              glue 分解(/tok): embed=%.2fms  forwardStep=%.2fms(GPU-exec %.2fms)  lm_head+argmax=%.2fms
            """, match, N, Double(match)/Double(N)*100, N - match, near, tru,
            ex.isEmpty ? "(完全一致)" : "first: " + ex.joined(separator: " | "), tokps, secs/Double(steps)*1000,
            tEmbed/Double(pr), tFwd/Double(pr), gpuExec/Double(pr), tHead/Double(pr))
    }

    /// ★ task#9 owner 指摘①: per-kernel GPU-exec 帰属 profile（②single-thread 並列化で足りるか③大 kernel 移植が要るか）。
    /// + baseline: 素朴 MLX decode tok/s。env QWISP_RUN=profile-gpu。
    public static func runProfileGPU(modelDir: String) throws -> String {
        let store = try WeightStore(modelDir: modelDir)
        let model = QwispModel(store: store)
        let prevF32 = GatedDeltaNetLayer.f32Conv; GatedDeltaNetLayer.f32Conv = true
        defer { GatedDeltaNetLayer.f32Conv = prevF32 }
        let ids = MLXArray([Int32(100)], [1, 1])
        func now() -> Double { Double(DispatchTime.now().uptimeNanoseconds) / 1e6 }
        // GPU forward を測る（wall + GPU-exec）。skip フラグで category 帰属。
        func measure(_ label: String, _ setup: () -> Void) -> (wall: Double, gpu: Double) {
            setup()
            for _ in 0..<3 { _ = model.fusedRawForwardGPU(ids)?.eval() }
            let reps = 30; var gpuAcc = 0.0; let t0 = now()
            for _ in 0..<reps { _ = model.fusedRawForwardGPU(ids)?.eval(); gpuAcc += RawMetalForward.lastGPUExecMs }
            return ((now()-t0)/Double(reps), gpuAcc/Double(reps))
        }
        _ = model.fusedRawForwardGPU(ids)?.eval()   // warm + build layers
        // ★ A/B: MoE gate+up gather+swiglu 融合の効果（GPU-exec, min over reps で thermal 耐性）
        func measureMin(_ setup: () -> Void) -> Double {
            setup(); for _ in 0..<3 { _ = model.fusedRawForwardGPU(ids)?.eval() }
            var mn = 1e9; for _ in 0..<40 { _ = model.fusedRawForwardGPU(ids)?.eval(); mn = Swift.min(mn, RawMetalForward.lastGPUExecMs) }
            return mn
        }
        let gFuse = measureMin { RawMetalForward.moeFuseGateUp = true }
        let gNoFuse = measureMin { RawMetalForward.moeFuseGateUp = false }
        RawMetalForward.moeFuseGateUp = true
        let full = measure("full") { RawMetalForward.profSkipSingleThread = false; RawMetalForward.profSkipMoEExperts = false; RawMetalForward.profSkipMixer = false }
        let noST = measure("skip-ST") { RawMetalForward.profSkipSingleThread = true }
        RawMetalForward.profSkipSingleThread = false
        let noMoE = measure("skip-MoE") { RawMetalForward.profSkipMoEExperts = true }
        RawMetalForward.profSkipMoEExperts = false
        let noMix = measure("skip-mixer") { RawMetalForward.profSkipMixer = true }
        RawMetalForward.profSkipMixer = false
        // mixer 内訳（GDN: matmul(in_proj×4+out_proj) vs recurrent）
        let noMM = measure("skip-GDN-matmul") { RawMetalForward.profSkipGDNMatmul = true }
        RawMetalForward.profSkipGDNMatmul = false
        let noRec = measure("skip-GDN-recur") { RawMetalForward.profSkipGDNRecur = true }
        RawMetalForward.profSkipGDNRecur = false
        // baseline: 素朴 MLX decode（caches, 32 step greedy）
        let T = 8
        let pid = MLXArray((0..<T).map { Int32($0 + 10) }, [1, T])
        let caches = model.makeCaches()
        var lg = model(pid, caches: caches)
        var nxt = MLX.argMax(lg[0, T-1], axis: -1).reshaped([1,1]); MLX.eval([nxt] + caches.flatMap { $0.stateArrays })
        let st = now(); let N = 32
        for _ in 0..<N { lg = model(nxt, caches: caches); nxt = MLX.argMax(lg[0,0], axis:-1).reshaped([1,1]); MLX.eval([nxt] + caches.flatMap { $0.stateArrays }) }
        let mlxMs = (now()-st)/Double(N)
        return String(format: """
            [profile-gpu] GPU-exec 帰属（owner 指摘①, cold T=1 forward, M1 系）
              ★MoE gate+up 融合 A/B（GPU-exec min over 40）: 融合=%.2fms / 非融合=%.2fms → 差 %.2fms
              full:        wall %.1fms / GPU-exec %.2fms（CPU-encode bubble = wall-gpu = %.2fms）
              single-thread(route_top8+shared_gate8) 寄与 ≈ %.2fms（full-skipST GPU-exec 差）
              MoE-experts(gather/swiglu/shared/combine/final) 寄与 ≈ %.2fms
              mixer(GDN/attn body) 寄与 ≈ %.2fms
                └ GDN matmul(in_proj×4+out_proj) 寄与 ≈ %.2fms / GDN recurrent 寄与 ≈ %.2fms
              skip-ST GPU=%.2f / skip-MoE GPU=%.2f / skip-mixer GPU=%.2f
              ── baseline ── 素朴 MLX decode(caches, %d step): %.2fms/tok (%.1f tok/s)
              → 判定: single-thread 寄与が大なら②並列化で足りる。mixer/MoE 大 kernel 寄与が支配なら③ mx.fast 移植深掘り。
            """, gFuse, gNoFuse, gNoFuse-gFuse, full.wall, full.gpu, full.wall-full.gpu,
            full.gpu-noST.gpu, full.gpu-noMoE.gpu, full.gpu-noMix.gpu,
            full.gpu-noMM.gpu, full.gpu-noRec.gpu,
            noST.gpu, noMoE.gpu, noMix.gpu, N, mlxMs, 1000/mlxMs)
    }

    /// ★ issue#6 第一歩: 既存 MLX batched forward の throughput スケール実測（C=256 resident, cold [B,1]）。
    /// dense amortization vs MoE expert-union の綱引きを B=1..32 で。新 kernel 無し。env QWISP_RUN=batch-scale。
    public static func runBatchScale(modelDir: String) throws -> String {
        let store = try WeightStore(modelDir: modelDir)
        store.residentAll()   // C=256 全常駐（IO ノイズ排除、batching の sweet spot 前提）
        let model = QwispModel(store: store)
        func now() -> Double { Double(DispatchTime.now().uptimeNanoseconds) / 1e6 }
        var out = "[batch-scale] MLX batched forward throughput（C=256 resident, cold [B,1]）\n"
        out += "  B    ms/forward   tok/s(=B/ms)   vs B=1 throughput\n"
        var base1 = 0.0
        for B in [1, 2, 4, 8, 16, 32] {
            let ids = MLXArray((0 ..< B).map { Int32(($0 * 131 + 7) % 100000) }, [B, 1])
            for _ in 0..<3 { model(ids).eval() }
            let reps = 15; let t0 = now()
            for _ in 0..<reps { model(ids).eval() }
            let ms = (now() - t0) / Double(reps)
            let tokps = Double(B) / ms * 1000
            if B == 1 { base1 = tokps }
            out += String(format: "  %-4d %8.1f %12.1f %14.2fx\n", B, ms, tokps, tokps / base1)
        }
        out += "  → throughput が B で伸びれば batching 有効（dense amortize 勝ち）。頭打ちなら MoE expert-union 律速。\n"
        // ★ per-stream correctness: batched 行 i の argmax vs 同 token の standalone [1,1] forward。
        //   不一致を rank 分類（near-tie=batch-variance で実用許容, rank≫=真の bug）。
        let Bc = 16
        let toks = (0 ..< Bc).map { Int32(($0 * 4099 + 31) % 100000) }
        let batched = model(MLXArray(toks, [Bc, 1])); batched.eval()   // [Bc, 1, vocab]
        var matchC = 0, nearC = 0, truC = 0; var exc: [String] = []
        for i in 0 ..< Bc {
            let solo = model(MLXArray([toks[i]], [1, 1])); solo.eval()
            let sv = solo.reshaped([solo.size]); let amS = MLX.argMax(sv).item(Int.self)
            let bvRow = batched[i].reshaped([batched.dim(-1)])
            let amB = MLX.argMax(bvRow).item(Int.self)
            if amB == amS { matchC += 1; continue }
            let rank = MLX.sum(sv .> sv[amB]).item(Int.self)   // standalone 内で batched-pred が何位
            if rank <= 2 { nearC += 1 } else { truC += 1 }
            if exc.count < 6 { exc.append("row\(i):rank\(rank)") }
        }
        out += String(format: "  [per-stream correct] batched vs standalone(B=%d): 一致 %d/%d, 不一致=%d（near-tie %d, 真の乖離 %d）%@\n",
                      Bc, matchC, Bc, Bc - matchC, nearC, truC, exc.isEmpty ? "" : exc.joined(separator: " "))
        out += "  → 不一致が全 near-tie なら各 stream は correct greedy（batch 構成で near-tie flip＝issue#6 既知, 実用許容）。\n"
        // ★ batched decode throughput（per-stream KV+GDN state cache, prefill [B,T]→N step [B,1]）＝実 throughput。
        out += "  ── batched decode throughput（caches, prefill T=8 → 16 step）──\n"
        out += "  B    ms/step   tok/s(=B/ms)\n"
        for B in [1, 4, 8, 16] {
            let pid = MLXArray((0 ..< B*8).map { Int32(($0 * 97 + 3) % 100000) }, [B, 8])
            let caches = model.makeCaches()
            var lg = model(pid, caches: caches)
            var nxt = MLX.argMax(lg[0..., 7], axis: -1).reshaped([B, 1])
            MLX.eval([nxt] + caches.flatMap { $0.stateArrays })
            for _ in 0..<3 { lg = model(nxt, caches: caches); nxt = MLX.argMax(lg[0..., 0], axis: -1).reshaped([B,1]); MLX.eval([nxt] + caches.flatMap { $0.stateArrays }) }
            let steps = 16; let t0 = now()
            for _ in 0..<steps { lg = model(nxt, caches: caches); nxt = MLX.argMax(lg[0..., 0], axis: -1).reshaped([B,1]); MLX.eval([nxt] + caches.flatMap { $0.stateArrays }) }
            let ms = (now()-t0)/Double(steps)
            out += String(format: "  %-4d %8.1f %12.1f\n", B, ms, Double(B)/ms*1000)
        }
        return out
    }

    /// ★ issue#6 #1: continuous(ragged) vs static batching の throughput 利得を**実測 per-B コスト駆動で定量化**。
    /// 可変 gen 長の workload で、static(wave で最長 gen を待つ＝finished slot idle)と continuous(slot を即埋め)を
    /// 実 decode ms(B) で simulate。利得が大なら本実装(per-stream RoPE/mask)の価値確定。env QWISP_RUN=continuous-sim。
    public static func runContinuousSim(modelDir: String) throws -> String {
        let store = try WeightStore(modelDir: modelDir)
        store.residentAll()
        let model = QwispModel(store: store)
        func now() -> Double { Double(DispatchTime.now().uptimeNanoseconds) / 1e6 }
        // 1. ms(B) 実測（batched decode step）
        var msB: [Int: Double] = [:]
        let Bs = [1, 2, 4, 8, 16]
        for B in Bs {
            let pid = MLXArray((0 ..< B*4).map { Int32(($0 * 97 + 3) % 100000) }, [B, 4])
            let caches = model.makeCaches()
            var lg = model(pid, caches: caches)
            var nxt = MLX.argMax(lg[0..., 3], axis: -1).reshaped([B, 1]); MLX.eval([nxt] + caches.flatMap { $0.stateArrays })
            for _ in 0..<3 { lg = model(nxt, caches: caches); nxt = MLX.argMax(lg[0..., 0], axis: -1).reshaped([B,1]); MLX.eval([nxt] + caches.flatMap { $0.stateArrays }) }
            let steps = 12; let t0 = now()
            for _ in 0..<steps { lg = model(nxt, caches: caches); nxt = MLX.argMax(lg[0..., 0], axis: -1).reshaped([B,1]); MLX.eval([nxt] + caches.flatMap { $0.stateArrays }) }
            msB[B] = (now()-t0)/Double(steps)
        }
        // 2. workload: N requests、gen 長は可変（決定論的 LCG, 16..160 の歪み分布）。
        let N = 256
        var gens: [Int] = []; var seed: UInt64 = 12345
        for _ in 0..<N { seed = seed &* 6364136223846793005 &+ 1; let r = Double((seed >> 33) & 0xFFFF) / 65535.0
            gens.append(16 + Int(pow(r, 2.2) * 144)) }   // 歪み（多くは短く、少数が長い）
        let totalTok = gens.reduce(0, +)
        // 3. simulate: static-wave vs continuous（同一 B で比較）。ms(B) を per-step コストに使用。
        func simStatic(_ B: Int) -> Double {   // wave ごとに最長 gen を全 slot が待つ
            var t = 0.0; var i = 0
            while i < N { let wave = Array(gens[i ..< Swift.min(i+B, N)]); t += Double(wave.max()!) * msB[B]!; i += B }
            return t
        }
        func simContinuous(_ B: Int) -> Double {  // B slot を常に埋める＝総 token-step / B（端数は減衰）
            // 到着順に slot へ。各 step で active slot 数だけ進む。finished は即 queue から補充。
            var t = 0.0; var queue = gens; var slots: [Int] = []   // slot は残り step 数
            var qi = 0
            while !slots.isEmpty || qi < queue.count {
                while slots.count < B && qi < queue.count { slots.append(queue[qi]); qi += 1 }
                let active = slots.count
                t += msB[active] ?? msB[B]!            // active 数に応じた step コスト（埋まってれば ms(B)）
                slots = slots.map { $0 - 1 }.filter { $0 > 0 }
            }
            return t
        }
        var out = "[continuous-sim] continuous vs static batching 利得（実測 ms(B) 駆動, N=\(N) req, total \(totalTok) tok）\n"
        out += "  実測 ms/step: " + Bs.map { "B\($0)=\(String(format: "%.0f", msB[$0]!))" }.joined(separator: " ") + "\n"
        out += "  B    static(s)  continuous(s)  利得   static tok/s  continuous tok/s\n"
        for B in [4, 8, 16] {
            let st = simStatic(B)/1000, co = simContinuous(B)/1000
            out += String(format: "  %-4d %9.1f %13.1f %6.2fx %12.0f %16.0f\n",
                          B, st, co, st/co, Double(totalTok)/st, Double(totalTok)/co)
        }
        out += "  → 利得が大なら continuous 本実装(per-stream RoPE/mask + slot 即補充)の価値確定。static は短 req が長 req を待ち idle。"
        return out
    }

    /// ★ continuous batching forward: GDN 層は batched gdnCaches[i]、attn 層は per-slot attnKV[i][B]+positions。
    public func forwardContinuous(_ tokens: MLXArray, positions: [Int],
                                  gdnCaches: [LayerCache], attnKV: [[KVCache]]) -> MLXArray {
        var h = embed(tokens)   // [B,1,H]
        for (i, layer) in layers.enumerated() {
            h = layer.callContinuous(h, gdnCache: isLinear(i) ? gdnCaches[i].gdn : nil,
                                     slotKV: isLinear(i) ? [] : attnKV[i], positions: positions)
        }
        h = MLXFast.rmsNorm(h, weight: store.req("language_model.model.norm.weight"), eps: eps)
        return headProj().apply(h)
    }

    /// ★ issue#6 #1: continuous batching の end-to-end 検証 + throughput。B 本の異 position stream を 1 batch で
    /// decode し、各 stream の予測を standalone と比較（per-stream correct）+ tok/s。env QWISP_RUN=continuous-batch。
    public static func runContinuousBatch(modelDir: String) throws -> String {
        let store = try WeightStore(modelDir: modelDir); store.residentAll()
        let model = QwispModel(store: store)
        let L = model.numLayers
        let B = 8
        // 各 stream に異なる長さの prompt（position を散らす）。standalone で prefill → caches を取得。
        let plens = (0 ..< B).map { 4 + $0 * 3 }   // 4,7,10,...,25
        var soloCaches: [[LayerCache]] = []; var firstTok: [Int32] = []
        for b in 0 ..< B {
            let pid = MLXArray((0 ..< plens[b]).map { Int32(($0 * 53 + b * 17 + 1) % 100000) }, [1, plens[b]])
            let c = model.makeCaches()
            let lg = model(pid, caches: c); lg.eval()
            firstTok.append(Int32(MLX.argMax(lg[0, plens[b]-1], axis: -1).item(Int.self)))
            MLX.eval(c.flatMap { $0.stateArrays })
            soloCaches.append(c)
        }
        // continuous state: GDN は per-stream state を [B,...] に stack、attn は per-slot KVCache 配列。
        var gdnCaches: [LayerCache] = (0 ..< L).map { _ in LayerCache() }
        var attnKV: [[KVCache]] = (0 ..< L).map { _ in [] }
        for i in 0 ..< L {
            if model.isLinear(i) {
                let recs = soloCaches.map { $0[i].gdn.recState! }
                let convs = soloCaches.map { $0[i].gdn.convState }
                gdnCaches[i].gdn.recState = MLX.concatenated(recs, axis: 0)
                if convs.allSatisfy({ $0 != nil }) { gdnCaches[i].gdn.convState = MLX.concatenated(convs.map { $0! }, axis: 0) }
            } else {
                attnKV[i] = soloCaches.map { $0[i].kv }   // per-slot 独立 KVCache をそのまま流用
            }
        }
        var positions = plens
        // 1 step continuous decode + per-stream correctness（standalone と比較）
        let tokB = MLXArray(firstTok, [B, 1])
        let lgC = model.forwardContinuous(tokB, positions: positions, gdnCaches: gdnCaches, attnKV: attnKV); lgC.eval()
        var match = 0, near = 0, tru = 0
        for b in 0 ..< B {
            let soloLg = model(MLXArray([firstTok[b]], [1,1]), caches: soloCaches[b]); soloLg.eval()
            let sv = soloLg.reshaped([soloLg.size]); let amS = MLX.argMax(sv).item(Int.self)
            let amC = MLX.argMax(lgC[b].reshaped([lgC.dim(-1)])).item(Int.self)
            if amC == amS { match += 1 } else { let rank = MLX.sum(sv .> sv[amC]).item(Int.self); if rank <= 2 { near += 1 } else { tru += 1 } }
        }
        // throughput: 続けて 16 step continuous decode（positions/state を進める）
        var cur = MLX.argMax(lgC[0..., 0], axis: -1).reshaped([B, 1])
        for b in 0 ..< B { positions[b] += 1 }
        MLX.eval([cur] + gdnCaches.flatMap { $0.stateArrays } + attnKV.flatMap { $0 }.compactMap { $0.keys })
        func now() -> Double { Double(DispatchTime.now().uptimeNanoseconds) / 1e6 }
        for _ in 0..<3 { let lg = model.forwardContinuous(cur, positions: positions, gdnCaches: gdnCaches, attnKV: attnKV); cur = MLX.argMax(lg[0..., 0], axis: -1).reshaped([B,1]); for b in 0..<B { positions[b] += 1 }; MLX.eval([cur] + gdnCaches.flatMap { $0.stateArrays }) }
        let steps = 16; let t0 = now()
        for _ in 0..<steps { let lg = model.forwardContinuous(cur, positions: positions, gdnCaches: gdnCaches, attnKV: attnKV); cur = MLX.argMax(lg[0..., 0], axis: -1).reshaped([B,1]); for b in 0..<B { positions[b] += 1 }; MLX.eval([cur] + gdnCaches.flatMap { $0.stateArrays }) }
        let ms = (now()-t0)/Double(steps)
        return String(format: """
            [continuous-batch] B=%d 異 position stream を 1 batch で continuous decode（per-slot KV + per-stream RoPE, GDN batched）
              per-stream correctness（step1 vs standalone）: 一致 %d/%d, 不一致=%d（near-tie %d, 真の乖離 %d）
              throughput: %.1f ms/step, %.1f tok/s aggregate
              → 一致/near-tie のみなら continuous batching が end-to-end で correct。
            """, B, match, B, B-match, near, tru, ms, Double(B)/ms*1000)
    }

    /// ★ issue#6 #1: continuous batching scheduler（slot 即補充）の real throughput vs static。
    /// 可変 gen 長 workload を (a)static-wave (b)continuous で実走し total 時間を比較。env QWISP_RUN=continuous-run。
    public static func runContinuousRun(modelDir: String) throws -> String {
        let store = try WeightStore(modelDir: modelDir); store.residentAll()
        let model = QwispModel(store: store); let L = model.numLayers
        let B = 8, N = 48
        var seed: UInt64 = 999
        func nextGen() -> Int { seed = seed &* 6364136223846793005 &+ 1; let r = Double((seed>>33)&0xFFFF)/65535.0; return 8 + Int(pow(r,2.0)*56) }
        let reqGens = (0..<N).map { _ in nextGen() }   // 各 req の生成トークン数
        func prompt(_ id: Int) -> MLXArray { MLXArray((0..<6).map { Int32(($0*53+id*17+1)%100000) }, [1, 6]) }
        func now() -> Double { Double(DispatchTime.now().uptimeNanoseconds)/1e6 }

        // ── (b) continuous: B slot を queue から即補充。GDN は batched(行 reset)、attn は per-slot KV 差替。
        func prefillSlot(_ id: Int) -> ([LayerCache], Int32, Int) {
            let c = model.makeCaches(); let p = prompt(id); let lg = model(p, caches: c); lg.eval()
            MLX.eval(c.flatMap { $0.stateArrays })
            return (c, Int32(MLX.argMax(lg[0, p.dim(1)-1], axis: -1).item(Int.self)), p.dim(1))
        }
        var gdnCaches: [LayerCache] = (0..<L).map { _ in LayerCache() }
        var attnKV: [[KVCache]] = (0..<L).map { _ in (0..<B).map { _ in KVCache() } }
        var slotReq = [Int](repeating: -1, count: B), slotRemain = [Int](repeating: 0, count: B)
        var positions = [Int](repeating: 0, count: B), cur = [Int32](repeating: 0, count: B)
        var nextReq = 0, done = 0
        // 初期 B slot を埋める（standalone prefill → batched に stack/inject）
        var pf: [[LayerCache]] = []
        for b in 0..<B { let (c, t, pl) = prefillSlot(nextReq); pf.append(c); slotReq[b]=nextReq; slotRemain[b]=reqGens[nextReq]; positions[b]=pl; cur[b]=t; nextReq += 1 }
        for i in 0..<L {
            if model.isLinear(i) {
                gdnCaches[i].gdn.recState = MLX.concatenated(pf.map { $0[i].gdn.recState! }, axis: 0)
                let convs = pf.map { $0[i].gdn.convState }
                if convs.allSatisfy({ $0 != nil }) { gdnCaches[i].gdn.convState = MLX.concatenated(convs.map { $0! }, axis: 0) }
            } else { for b in 0..<B { attnKV[i][b] = pf[b][i].kv } }
        }
        let tC0 = now(); var prefillMs = 0.0, decodeSteps = 0; var activeAcc = 0
        while done < N {
            let ts = now()
            let lg = model.forwardContinuous(MLXArray(cur, [B,1]), positions: positions, gdnCaches: gdnCaches, attnKV: attnKV)
            let nx = MLX.argMax(lg[0..., 0], axis: -1); nx.eval()
            MLX.eval(gdnCaches.flatMap { $0.stateArrays })
            decodeSteps += 1; _ = ts
            activeAcc += (0..<B).filter { slotReq[$0] >= 0 }.count   // この step の active(非idle) slot 数
            let nxA = nx.asArray(Int32.self)
            for b in 0..<B { cur[b]=nxA[b]; positions[b] += 1; slotRemain[b] -= 1 }
            // 完了 slot を補充
            for b in 0..<B where slotRemain[b] <= 0 && slotReq[b] >= 0 {
                done += 1; slotReq[b] = -1
                if nextReq < N {
                    let tp = now()
                    let (c, t, pl) = prefillSlot(nextReq)
                    for i in 0..<L {
                        if model.isLinear(i) {
                            gdnCaches[i].gdn.recState![b] = c[i].gdn.recState!.squeezed(axis: 0)
                            if let cv = c[i].gdn.convState, gdnCaches[i].gdn.convState != nil { gdnCaches[i].gdn.convState![b] = cv.squeezed(axis: 0) }
                        } else { attnKV[i][b] = c[i].kv }
                    }
                    prefillMs += now() - tp
                    slotReq[b]=nextReq; slotRemain[b]=reqGens[nextReq]; positions[b]=pl; cur[b]=t; nextReq += 1
                } else { slotRemain[b] = Int.max }   // queue 空＝この slot は idle（active 数減）
            }
        }
        let contMs = now() - tC0
        let avgActive = Double(activeAcc) / Double(max(1, decodeSteps))
        // ── (a) static-wave: B 本ずつ、wave 内最長 gen まで全 slot 回す（finished idle）。
        let tS0 = now(); var i = 0
        while i < N {
            let wave = Array(reqGens[i ..< Swift.min(i+B, N)]); let bb = wave.count; let maxg = wave.max()!
            var pf2: [[LayerCache]] = []; var c2 = [Int32](repeating:0,count:bb); var pos2=[Int](repeating:0,count:bb)
            for b in 0..<bb { let (c,t,pl) = prefillSlot(i+b); pf2.append(c); c2[b]=t; pos2[b]=pl }
            var g2: [LayerCache] = (0..<L).map { _ in LayerCache() }; var a2: [[KVCache]] = (0..<L).map { _ in (0..<bb).map { _ in KVCache() } }
            for li in 0..<L { if model.isLinear(li) { g2[li].gdn.recState = MLX.concatenated(pf2.map { $0[li].gdn.recState! }, axis: 0); let cv = pf2.map { $0[li].gdn.convState }; if cv.allSatisfy({ $0 != nil }) { g2[li].gdn.convState = MLX.concatenated(cv.map { $0! }, axis: 0) } } else { for b in 0..<bb { a2[li][b] = pf2[b][li].kv } } }
            for _ in 0..<maxg { let lg = model.forwardContinuous(MLXArray(c2,[bb,1]), positions: pos2, gdnCaches: g2, attnKV: a2); let nx = MLX.argMax(lg[0...,0],axis: -1); nx.eval(); MLX.eval(g2.flatMap { $0.stateArrays }); let na = nx.asArray(Int32.self); for b in 0..<bb { c2[b]=na[b]; pos2[b] += 1 } }
            i += B
        }
        let statMs = now() - tS0
        let totTok = reqGens.reduce(0,+)
        return String(format: """
            [continuous-run] B=%d, N=%d req(可変 gen), total %d tok ── 実走 continuous vs static
              continuous: %.1f s, %.1f tok/s   static-wave: %.1f s, %.1f tok/s   → 利得 %.2fx
              ★内訳: continuous %d decode-step, prefill %.1f s(%.0f%%), 平均 active slot %.1f/%d
                （定常 decode rate ≈ %.1f tok/s。差は prefill(同期 %d 回)+末尾 idle slot(平均 active<%d)）
            """, B, N, totTok, contMs/1000, Double(totTok)/contMs*1000, statMs/1000, Double(totTok)/statMs*1000, statMs/contMs,
            decodeSteps, prefillMs/1000, prefillMs/contMs*100, avgActive, B,
            Double(totTok)/((contMs-prefillMs)/1000), N, B)
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

public enum DecodeValidation {
    public static func run(modelDir: String, refPath: String) throws -> String {
        let r = try loadArrays(url: URL(fileURLWithPath: refPath))
        guard let ids = r["ids"] else { return "ERROR: decode ref に ids 無し" }
        let store = try WeightStore(modelDir: modelDir)
        store.residentAll()   // resident regime: experts も常駐させて mmap オーバーヘッド排除
        let model = QwispModel(store: store)
        let T = ids.dim(-1)

        // (1) cache 正しさ: no-cache full の最終位置 logits と prefill+1decode の logits を f32 比較
        let full = model(ids, f32: true)
        let lastFull = full[0, T - 1]
        let caches = model.makeCaches()
        _ = model(ids[0..., 0 ..< (T - 1)], caches: caches, f32: true)   // prefill
        let dec = model(ids[0..., (T - 1)...], caches: caches, f32: true) // 1 token decode
        let lastDec = dec[0, 0]
        lastFull.eval(); lastDec.eval()
        let cacheRel = MLX.max(MLX.abs(lastFull.asType(.float32) - lastDec.asType(.float32))).item(Float.self)
            / (MLX.max(MLX.abs(lastFull.asType(.float32))).item(Float.self) + 1e-9)
        let amFull = MLX.argMax(lastFull, axis: -1).item(Int.self)
        let amDec = MLX.argMax(lastDec, axis: -1).item(Int.self)

        // (2) tok/s: f16 で prefill→32 step greedy decode を計測
        let gCaches = model.makeCaches()
        var logits = model(ids, caches: gCaches)
        var next = MLX.argMax(logits[0, T - 1], axis: -1).reshaped([1, 1])
        MLX.eval([next] + gCaches.flatMap { $0.stateArrays })
        let N = 32
        var toks: [Int] = []
        let t0 = DispatchTime.now()
        for _ in 0 ..< N {
            logits = model(next, caches: gCaches)
            next = MLX.argMax(logits[0, 0], axis: -1).reshaped([1, 1])
            // next と cache 状態を毎 step eval（lazy グラフが step 毎に増殖するのを防ぐ）
            MLX.eval([next] + gCaches.flatMap { $0.stateArrays })
            toks.append(next.item(Int.self))
        }
        let secs = Double(DispatchTime.now().uptimeNanoseconds - t0.uptimeNanoseconds) / 1e9
        let tokPerSec = Double(N) / secs

        // 内訳プロファイル: lm_head 単体 vs embed+40層+norm（cache 無しの単発で粗く）
        func timeIt(_ reps: Int, _ f: () -> MLXArray) -> Double {
            for _ in 0 ..< 3 { f().eval() }
            let s = DispatchTime.now()
            for _ in 0 ..< reps { f().eval() }
            return Double(DispatchTime.now().uptimeNanoseconds - s.uptimeNanoseconds) / 1e6 / Double(reps)
        }
        let one = ids[0..., 0 ..< 1]
        let pcache = model.makeCaches()
        _ = model(ids[0..., 0 ..< (T - 1)], caches: pcache)  // 状態を進めておく
        MLX.eval(pcache.flatMap { $0.stateArrays })
        let hid = MLXArray.zeros([1, 1, 2048], dtype: .float16)
        let msHead = timeIt(30) { model.headProj().apply(hid) }
        let msStep = timeIt(30) {
            let c = model.makeCaches()
            return model(one, caches: c)
        }

        let cacheOK = cacheRel < 1e-4 && amFull == amDec
        return String(format: """
            [M2b-3] decode cache 正しさ(f32): last_logits_rel=%.2e argmax(%d==%d) %@
               tok/s 粗計測(f16, prefill T=%d→%d step decode): %.1f tok/s (%.1f ms/tok)  最初の生成=%@
               内訳: lm_head=%.1f ms  embed+40層+norm+head(1step)=%.1f ms  → head が %.0f%%
            """,
            cacheRel, amFull, amDec, cacheOK ? "OK ✅" : "MISMATCH ❌",
            T, N, tokPerSec, secs / Double(N) * 1000, "\(toks.prefix(6))",
            msHead, msStep, msHead / msStep * 100)
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
