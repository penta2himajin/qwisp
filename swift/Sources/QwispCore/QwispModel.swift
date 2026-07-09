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
