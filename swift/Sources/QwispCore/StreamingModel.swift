import Foundation
import MLX
import MLXFast
import Metal

/// streaming 版 DecoderLayer: norm/attn/gdn は resident、MoE のみ持続 arena streaming。
public final class StreamingDecoderLayer {
    let isLinear: Bool
    let eps: Float
    let inputLayernorm: MLXArray
    let postAttentionLayernorm: MLXArray
    let gdn: GatedDeltaNetLayer?
    let attn: AttentionLayer?
    let mlp: StreamingMoEBlock

    public init(isLinear: Bool, eps: Float, inputLayernorm: MLXArray, postAttentionLayernorm: MLXArray,
                gdn: GatedDeltaNetLayer?, attn: AttentionLayer?, mlp: StreamingMoEBlock) {
        self.isLinear = isLinear; self.eps = eps
        self.inputLayernorm = inputLayernorm; self.postAttentionLayernorm = postAttentionLayernorm
        self.gdn = gdn; self.attn = attn; self.mlp = mlp
    }

    public func callAsFunction(_ x: MLXArray, cache: LayerCache?) throws -> MLXArray {
        let normed = MLXFast.rmsNorm(x, weight: inputLayernorm, eps: eps)
        let r = isLinear ? gdn!(normed, cache: cache?.gdn) : attn!(normed, cache: cache?.kv)
        let h = x + r
        let postNorm = MLXFast.rmsNorm(h, weight: postAttentionLayernorm, eps: eps)
        let B = h.dim(0), S = h.dim(1), H = h.dim(2)
        let mlpOut = try mlp(postNorm.reshaped([B * S, H])).reshaped([B, S, H])
        return h + mlpOut
    }
}

/// streaming full model: embed/norm/lm_head/attn/gdn resident, switch_mlp は arena streaming。
public final class StreamingQwispModel {
    let store: WeightStore
    let arena: ExpertArena
    let numLayers: Int
    let eps: Float
    var layers: [StreamingDecoderLayer] = []

    public var expertCaches: [LayerExpertCache] = []

    public init(store: WeightStore, arena: ExpertArena, numLayers: Int = 40,
                fullAttnInterval: Int = 4, eps: Float = 1e-6,
                device: MTLDevice? = nil, source: ExpertSource? = nil, cacheC: Int? = nil) throws {
        self.store = store; self.arena = arena; self.numLayers = numLayers; self.eps = eps
        let base = QwispModel(store: store, numLayers: numLayers, fullAttnInterval: fullAttnInterval,
                              eps: eps)
        for i in 0 ..< numLayers {
            let p = "language_model.model.layers.\(i)"
            func q8(_ n: String) -> Proj {
                .quantized(store.req("\(n).weight"), store.req("\(n).scales"), store.req("\(n).biases"), 8)
            }
            func q4(_ n: String) -> Proj {
                .quantized(store.req("\(n).weight"), store.req("\(n).scales"), store.req("\(n).biases"), 4)
            }
            // MoE 層のみ cache を作る（cacheC 指定時）
            var layerCache: LayerExpertCache? = nil
            let bl = base.layers[i]
            if let C = cacheC, let dev = device, let src = source, bl.gdn != nil || bl.attn != nil {
                layerCache = try LayerExpertCache(device: dev, source: src, layer: i, C: C)
                expertCaches.append(layerCache!)
            }
            let mlp = StreamingMoEBlock(
                topK: 8, numExperts: 256, normTopk: true, expertBits: 4, layer: i,
                gate: q8("\(p).mlp.gate"), shGate: q4("\(p).mlp.shared_expert.gate_proj"),
                shUp: q4("\(p).mlp.shared_expert.up_proj"), shDown: q4("\(p).mlp.shared_expert.down_proj"),
                sharedGate: q8("\(p).mlp.shared_expert_gate"), arena: arena, cache: layerCache)
            layers.append(StreamingDecoderLayer(
                isLinear: bl.isLinear, eps: eps,
                inputLayernorm: store.req("\(p).input_layernorm.weight"),
                postAttentionLayernorm: store.req("\(p).post_attention_layernorm.weight"),
                gdn: bl.gdn, attn: bl.attn, mlp: mlp))
        }
    }

    func embed(_ ids: MLXArray) -> MLXArray {
        ModelHead.embed(ids: ids, weight: store.req("language_model.model.embed_tokens.weight"),
                        scales: store.req("language_model.model.embed_tokens.scales"),
                        biases: store.req("language_model.model.embed_tokens.biases"), bits: 4)
    }

    public func makeCaches() -> [LayerCache] { (0 ..< numLayers).map { _ in LayerCache() } }

    public func callAsFunction(_ ids: MLXArray, caches: [LayerCache]) throws -> MLXArray {
        var h = embed(ids)
        for (i, layer) in layers.enumerated() { h = try layer(h, cache: caches[i]) }
        h = MLXFast.rmsNorm(h, weight: store.req("language_model.model.norm.weight"), eps: eps)
        let head = Proj.quantized(store.req("language_model.lm_head.weight"),
                                  store.req("language_model.lm_head.scales"),
                                  store.req("language_model.lm_head.biases"), 4)
        return head.apply(h)
    }
}

public enum StreamingDecode {
    static func rssGB() -> Double {
        var u = rusage()
        getrusage(RUSAGE_SELF, &u)
        return Double(u.ru_maxrss) / 1e9   // macOS: bytes
    }

    public static func run(modelDir: String, refPath: String) throws -> String {
        guard let device = MTLCreateSystemDefaultDevice() else { return "ERROR: no Metal device" }
        let r = try loadArrays(url: URL(fileURLWithPath: refPath))
        guard let ids = r["ids"] else { return "ERROR: ref に ids 無し" }

        // cache slot 数 C を env QWISP_CACHE_C で指定（0/未指定=cache無）。
        let C = Int(ProcessInfo.processInfo.environment["QWISP_CACHE_C"] ?? "64") ?? 64
        let store = try WeightStore(modelDir: modelDir)
        store.residentNonExperts()       // expert は eval しない（mmap のまま、streaming で pread）
        let source = try ExpertSource(modelDir: modelDir)
        try source.warm()                // 並列 pread 前に header/fd を先読み（dict 競合回避）
        let arena = try ExpertArena(device: device, source: source, N: 64)
        let model = try StreamingQwispModel(store: store, arena: arena,
                                            device: device, source: source, cacheC: C > 0 ? C : nil)
        let rssLoad = rssGB()

        let T = ids.dim(-1)
        let caches = model.makeCaches()
        var logits = try model(ids, caches: caches)
        var next = MLX.argMax(logits[0, T - 1], axis: -1).reshaped([1, 1])
        MLX.eval([next] + caches.flatMap { $0.stateArrays })

        let N = 32
        var toks: [Int] = []
        let t0 = DispatchTime.now()
        for _ in 0 ..< N {
            logits = try model(next, caches: caches)
            next = MLX.argMax(logits[0, 0], axis: -1).reshaped([1, 1])
            MLX.eval([next] + caches.flatMap { $0.stateArrays })
            toks.append(next.item(Int.self))
        }
        let secs = Double(DispatchTime.now().uptimeNanoseconds - t0.uptimeNanoseconds) / 1e9
        let rssPeak = rssGB()
        let hits = model.expertCaches.reduce(0) { $0 + $1.hits }
        let misses = model.expertCaches.reduce(0) { $0 + $1.misses }
        let hitRate = hits + misses > 0 ? Double(hits) / Double(hits + misses) * 100 : 0

        return String(format: """
            [S3] streaming decode(8GB狙い, LRU cache C=%d/層, experts 非常駐):
               %.1f tok/s (%.1f ms/tok)  RSS: load=%.1fGB peak=%.1fGB
               cache hit=%.0f%% (hit=%d miss=%d)  生成=%@
            """,
            C, Double(N) / secs, secs / Double(N) * 1000, rssLoad, rssPeak,
            hitRate, hits, misses, "\(toks.prefix(6))")
    }
}
