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
                attnBufCache[i] = RawMetalForward.prepareAttnBuffers(aw, H: H)
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
        // MoE: routing(MLX) + expert SE + combine(MLX) + residual(kernel)
        let p = "language_model.model.layers.\(i)", mp = "\(p).mlp"
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
        if let fu = model.fusedRawForward(ids) {
            fu.eval()
            let ff = fu.reshaped([fu.size])
            let fd = MLX.max(MLX.abs(ff.asType(.float32) - rf.asType(.float32))).item(Float.self)
            let frel = fd / (MLX.max(MLX.abs(rf.asType(.float32))).item(Float.self) + 1e-9)
            let amF = MLX.argMax(ff).item(Int.self)
            for _ in 0..<3 { _ = model.fusedRawForward(ids)?.eval() }
            t0 = now(); for _ in 0..<reps { _ = model.fusedRawForward(ids)?.eval() }; let fuMs = (now()-t0)/Double(reps)
            out += String(format: "\n  ── 層融合 full forward（hBuf 常駐, norm/residual も kernel）──\n   logits rel=%.3e argmax %d(ref %d)%@  時間=%.1fms(%.1f tok/s) → vs MLX %.2fx",
                          frel, amF, amR, amF == amR ? "✅" : "❌", fuMs, 1000/fuMs, mlxMs/Swift.max(0.01, fuMs))
        }
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
