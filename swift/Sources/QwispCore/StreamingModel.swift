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
        if StreamingMoEBlock.captureLayerInput { mlp.cache?.preAttnInput = x }   // 予測器 calib
        if StreamingMoEBlock.profileLayers { return try profiledForward(x, cache: cache) }
        let normed = MLXFast.rmsNorm(x, weight: inputLayernorm, eps: eps)
        let r = isLinear ? gdn!(normed, cache: cache?.gdn) : attn!(normed, cache: cache?.kv)
        let h = x + r
        let postNorm = MLXFast.rmsNorm(h, weight: postAttentionLayernorm, eps: eps)
        let B = h.dim(0), S = h.dim(1), H = h.dim(2)
        let mlpOut = try mlp(postNorm.reshaped([B * S, H])).reshaped([B, S, H])
        return h + mlpOut
    }

    /// barrier 計測版: norm / (gdn|attn) / MoE を eval 区切りで個別計時。
    func profiledForward(_ x: MLXArray, cache: LayerCache?) throws -> MLXArray {
        func now() -> UInt64 { DispatchTime.now().uptimeNanoseconds }
        MLX.eval(x)
        var t = now()
        let normed = MLXFast.rmsNorm(x, weight: inputLayernorm, eps: eps)
        MLX.eval(normed); StreamingMoEBlock.tNorm += now() - t; t = now()
        let r = isLinear ? gdn!(normed, cache: cache?.gdn) : attn!(normed, cache: cache?.kv)
        let stateAfter = isLinear ? [cache?.gdn.convState, cache?.gdn.recState].compactMap { $0 }
                                  : [cache?.kv.keys, cache?.kv.values].compactMap { $0 }
        MLX.eval([r] + stateAfter)
        if isLinear { StreamingMoEBlock.tGDN += now() - t } else { StreamingMoEBlock.tAttn += now() - t }
        let h = x + r
        t = now()
        let postNorm = MLXFast.rmsNorm(h, weight: postAttentionLayernorm, eps: eps)
        MLX.eval(postNorm); StreamingMoEBlock.tNorm += now() - t
        let B = h.dim(0), S = h.dim(1), H = h.dim(2)
        let mlpOut = try mlp(postNorm.reshaped([B * S, H])).reshaped([B, S, H])
        MLX.eval(mlpOut)   // MoE 内訳は StreamingMoEBlock 側で計時
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

    /// Tell 用: 層範囲 [lo, hi) だけ処理して hidden を返す（chunk overlap 用）。
    public func runChunk(_ h: MLXArray, _ lo: Int, _ hi: Int, caches: [LayerCache]) throws -> MLXArray {
        var x = h
        for i in lo ..< hi { x = try layers[i](x, cache: caches[i]) }
        return x
    }
    public func embedPub(_ ids: MLXArray) -> MLXArray { embed(ids) }
    public func finalNorm(_ h: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(h, weight: store.req("language_model.model.norm.weight"), eps: eps)
    }
    public func logitsFromNorm(_ n: MLXArray) -> MLXArray { headProj().apply(n) }
    public func finalLogits(_ h: MLXArray) -> MLXArray { logitsFromNorm(finalNorm(h)) }
    public var layerCount: Int { numLayers }
    /// Tell M1: 層 i の expert を任意 hidden h から予測（cross-layer）。
    public func predictLayerInds(_ i: Int, _ h: MLXArray) -> MLXArray { layers[i].mlp.predictInds(h) }
    /// 幅可変版: 層 i の top-k 予測（exact-pipeline prefetch 幅振り用）。
    public func predictLayerIndsK(_ i: Int, _ h: MLXArray, _ k: Int) -> MLXArray { layers[i].mlp.predictIndsK(h, k) }

    func headProj() -> Proj {
        .quantized(store.req("language_model.lm_head.weight"),
                   store.req("language_model.lm_head.scales"),
                   store.req("language_model.lm_head.biases"), 4)
    }

    public func callAsFunction(_ ids: MLXArray, caches: [LayerCache]) throws -> MLXArray {
        try forwardHidden(ids, caches: caches).logits
    }

    /// **[8GB exact-pipeline] per-layer 予測 prefetch + miss escalation の exact forward（L=1 専用）**
    /// 各層 i: intra1(前層 gate入力)で top-w 予測→ensure(resident 化)→no-sync gather→1int miss drain。
    /// true top-8 ⊂ predicted-resident(miss=0)なら no-sync exact、miss>0 ならその層だけ snapshot 巻戻し
    /// sync 再計算(exact)。出力は exact 経路と bit 一致。escLayers=sync 再計算した層数。
    /// ※直列版(予測+ensure は同期)。overlap 無しゆえ予測コストを含む＝速度は async 化前提の go/no-go 計測用。
    public func forwardHiddenPipeline(_ ids: MLXArray, caches: [LayerCache], predictW: Int, isLin: [Bool])
        throws -> (hidden: MLXArray, logits: MLXArray, escLayers: Int) {
        var h = embed(ids)
        var esc = 0
        let trim = ids.dim(1)
        StreamingMoEBlock.captureGateInput = true
        for i in 0 ..< numLayers {
            if i > 0, let src = expertCaches[i - 1].lastGateInput {        // intra1 予測 → prefetch
                let pred = layers[i].mlp.predictIndsK(src, predictW)
                var seen = Set<Int>(); var u: [Int] = []
                for e in pred.asArray(Int32.self) { let v = Int(e); if seen.insert(v).inserted { u.append(v) } }
                _ = expertCaches[i].ensure(u)
            }
            let snap = caches[i].snapshot()
            StreamingMoEBlock.hotMissAccum = nil
            StreamingMoEBlock.probeNoSync = (i > 0)                        // 層0は予測元無し→sync
            StreamingMoEBlock.countHotMiss = (i > 0)
            var hn = try layers[i](h, cache: caches[i])
            if i > 0 {
                let missArr = StreamingMoEBlock.hotMissAccum ?? MLXArray(Int32(0))
                MLX.eval([hn, missArr] + caches[i].stateArrays)            // per-layer 1int drain（+ 出力確定）
                if missArr.item(Int32.self) != 0 {                        // cold routed → その層だけ sync 再計算
                    esc += 1
                    caches[i].restore(snap, isLinear: isLin[i], trim: trim)
                    StreamingMoEBlock.probeNoSync = false; StreamingMoEBlock.countHotMiss = false
                    hn = try layers[i](h, cache: caches[i])
                }
            }
            h = hn
        }
        StreamingMoEBlock.captureGateInput = false
        StreamingMoEBlock.probeNoSync = false; StreamingMoEBlock.countHotMiss = false
        let hidden = MLXFast.rmsNorm(h, weight: store.req("language_model.model.norm.weight"), eps: eps)
        return (hidden, headProj().apply(hidden), esc)
    }

    /// cached forward で (post-norm hidden, logits)（MTP 投機用）。
    public func forwardHidden(_ ids: MLXArray, caches: [LayerCache]) throws -> (hidden: MLXArray, logits: MLXArray) {
        var h = embed(ids)
        for (i, layer) in layers.enumerated() { h = try layer(h, cache: caches[i]) }
        let hidden = MLXFast.rmsNorm(h, weight: store.req("language_model.model.norm.weight"), eps: eps)
        return (hidden, headProj().apply(hidden))
    }

    /// layer-skip draft 用: skip に含まれる層は identity(計算ごと省略)。draft を計算ごと安くする。
    /// 注意: skip 層は cache を更新しないので draft state は近似（throwaway 前提）。
    public func forwardHiddenSkip(_ ids: MLXArray, caches: [LayerCache], skip: Set<Int>)
        throws -> (hidden: MLXArray, logits: MLXArray) {
        var h = embed(ids)
        for (i, layer) in layers.enumerated() where !skip.contains(i) { h = try layer(h, cache: caches[i]) }
        let hidden = MLXFast.rmsNorm(h, weight: store.req("language_model.model.norm.weight"), eps: eps)
        return (hidden, headProj().apply(hidden))
    }

    /// チャンク分割 prefill。1 forward の |U| が cache slot C を超えると in-place arena が
    /// 破綻するため、chunk≤C/topK で分割（chunk 毎に eval して arena 上書き前に materialize）。
    /// ※ union-overflow guard は spec verify loop 専用で prefill には効かないため、この上限は hard cap。
    /// chunk 省略時は **C 非依存の定数 8**（`chunk:` 引数はテスト用に残す）:
    ///   ★prefill の batched kernel は chunk 形状に数値依存（order-stable でない）ため、chunk を
    ///   C 依存にすると「同 prompt でも RAM tier ごとに出力が変わる」= L1 の cross-C 一貫性を毀損する
    ///   （2026-07-02 実測: C 依存 chunk で code free-run が正準 ref から 28-73% に分岐）。
    ///   8 は最小 tier C=64 の capacity 上限 C/topK=8 と一致し全 tier 安全・全 tier 同一計算。
    ///   prefill schedule は「正準計算」の一部: これを変えたら refs 再生成が必須。
    /// lm_head は最終 chunk の最終位置のみ計算（全 call site が last-position argmax のみ消費、
    /// 非最終 chunk の full-vocab 射影 [chunk,248320]≈254MB 重み読みを省く。先頭 token も decode と
    /// 同じ M=1 lm_head kernel になり一様）。
    /// hidden は全 chunk の post-norm を concat して返す（MTP head が prompt 全 hidden を消費）。
    public func prefillChunked(_ ids: MLXArray, caches: [LayerCache], chunk: Int? = nil)
        throws -> (hidden: MLXArray, logits: MLXArray) {
        let topK = layers.first?.mlp.topK ?? 8
        let cap = expertCaches.first?.C ?? arena.N          // per-layer arena capacity
        let ch = Swift.min(chunk ?? 8, Swift.max(1, cap / topK))  // capacity hard cap
        let P = ids.dim(1)
        var hiddens: [MLXArray] = []
        var lastLogits = MLXArray.zeros([1, 1, 1])
        var pos = 0
        while pos < P {
            let end = Swift.min(pos + ch, P)
            var h = embed(ids[0..., pos ..< end])
            for (i, layer) in layers.enumerated() { h = try layer(h, cache: caches[i]) }
            let hidden = finalNorm(h)
            if end == P {                                   // 最終 chunk のみ lm_head（最終位置のみ）
                let lg = logitsFromNorm(hidden[0..., (hidden.dim(1) - 1)...])
                MLX.eval([hidden, lg] + caches.flatMap { $0.stateArrays })  // arena 上書き前に確定
                lastLogits = lg
            } else {                                        // headless: 層 + final norm のみ
                MLX.eval([hidden] + caches.flatMap { $0.stateArrays })      // arena 上書き前に確定
            }
            hiddens.append(hidden)
            pos = end
        }
        return (MLX.concatenated(hiddens, axis: 1), lastLogits)
    }
    public var isLinearFlags: [Bool] { layers.map { $0.isLinear } }
}

public enum StreamingDecode {
    static func rssGB() -> Double {
        var u = rusage()
        getrusage(RUSAGE_SELF, &u)
        return Double(u.ru_maxrss) / 1e9   // macOS: bytes
    }

    /// hybrid fast decode: ほとんどの token を GPU-remap no-sync(高速・近似)で回し、
    /// 毎 refreshEvery token だけ sync mode(正確・cache 更新)。quality-GREEN 前提の速度モード。
    /// tok/s と greedy 一致率(品質)の両方を計測。
    public static func runHybridFast(modelDir: String, refPath: String) throws -> String {
        guard let device = MTLCreateSystemDefaultDevice() else { return "ERROR: no Metal device" }
        let r = try loadArrays(url: URL(fileURLWithPath: refPath))
        guard let promptArr = r["spec_prompt"], let gRef = r["spec_greedy"] else {
            return "[fast] skip: spec_prompt 無し"
        }
        let C = Int(ProcessInfo.processInfo.environment["QWISP_CACHE_C"] ?? "64") ?? 64
        let RE = Int(ProcessInfo.processInfo.environment["QWISP_REFRESH"] ?? "4") ?? 4
        let store = try WeightStore(modelDir: modelDir)
        store.residentNonExperts()
        let source = try ExpertSource(modelDir: modelDir); try source.warm()
        let arena = try ExpertArena(device: device, source: source, N: 64)
        let model = try StreamingQwispModel(store: store, arena: arena, device: device,
                                            source: source, cacheC: C)
        let gR = gRef.asArray(Int32.self).map { Int($0) }
        let ids = promptArr.asType(.int32).reshaped([1, promptArr.dim(0)])
        let N = Swift.min(Int(ProcessInfo.processInfo.environment["QWISP_GEN"] ?? "48") ?? 48, gR.count)
        let caches = model.makeCaches()

        // prefill(sync, chunked) + cache warmup
        StreamingMoEBlock.probeNoSync = false
        var (_, lg) = try model.prefillChunked(ids, caches: caches)
        var next = MLX.argMax(lg[0, lg.dim(1) - 1], axis: -1).reshaped([1, 1])
        MLX.eval([next] + caches.flatMap { $0.stateArrays })

        var out: [Int] = []
        let t0 = DispatchTime.now()
        for i in 0 ..< N {
            out.append(next.item(Int.self))
            StreamingMoEBlock.probeNoSync = (i % RE != 0)   // refreshEvery 毎に sync(正確+cache更新)
            (_, lg) = try model.forwardHidden(next, caches: caches)
            next = MLX.argMax(lg[0, 0], axis: -1).reshaped([1, 1])
            MLX.eval([next] + caches.flatMap { $0.stateArrays })
        }
        StreamingMoEBlock.probeNoSync = false
        let secs = Double(DispatchTime.now().uptimeNanoseconds - t0.uptimeNanoseconds) / 1e9
        let match = zip(out, gR).filter { $0 == $1 }.count
        return String(format: """
            [fast] hybrid no-sync decode(C=%d, sync 1/%d): %.1f tok/s  品質(greedy一致) %d/%d=%.0f%%
            """,
            C, RE, Double(N) / secs, match, N, Double(match) / Double(N) * 100)
    }

    /// margin 適応 fast decode: fast logits の top1-top2 gap が小さい(near-tie=harmful 候補)token
    /// だけ sync 訂正。benign miss は margin 大で素通し→選択的に安定化。
    public static func runMarginFast(modelDir: String, refPath: String) throws -> String {
        guard let device = MTLCreateSystemDefaultDevice() else { return "ERROR: no Metal device" }
        let r = try loadArrays(url: URL(fileURLWithPath: refPath))
        guard let promptArr = r["spec_prompt"], let gRef = r["spec_greedy"] else { return "[margin] skip" }
        let C = Int(ProcessInfo.processInfo.environment["QWISP_CACHE_C"] ?? "64") ?? 64
        let TH = Float(ProcessInfo.processInfo.environment["QWISP_MARGIN"] ?? "6") ?? 6
        let store = try WeightStore(modelDir: modelDir)
        store.residentNonExperts()
        let source = try ExpertSource(modelDir: modelDir); try source.warm()
        let arena = try ExpertArena(device: device, source: source, N: 64)
        let model = try StreamingQwispModel(store: store, arena: arena, device: device,
                                            source: source, cacheC: C)
        let ids = promptArr.asType(.int32).reshaped([1, promptArr.dim(0)])
        let gR = gRef.asArray(Int32.self).map { Int($0) }
        let isLin = model.isLinearFlags
        let N = Swift.min(Int(ProcessInfo.processInfo.environment["QWISP_GEN"] ?? "48") ?? 48, gR.count)
        let caches = model.makeCaches()

        StreamingMoEBlock.probeNoSync = false
        var (_, lg) = try model.prefillChunked(ids, caches: caches)
        var cur = MLX.argMax(lg[0, lg.dim(1) - 1], axis: -1).reshaped([1, 1])
        MLX.eval([cur] + caches.flatMap { $0.stateArrays })

        func margin(_ l: MLXArray) -> (MLXArray, MLXArray) {       // (next, marginScalar)
            let m1 = MLX.max(l, axis: -1, keepDims: true)
            let masked = MLX.where(l .>= m1, MLXArray(Float(-1e9)), l)
            let m2 = MLX.max(masked, axis: -1, keepDims: true)
            return (MLX.argMax(l, axis: -1).reshaped([1, 1]), (m1 - m2).reshaped([1]))
        }

        var out: [Int] = []; var syncTokens = 0
        let t0 = DispatchTime.now()
        for _ in 0 ..< N {
            let prev = cur
            let snaps = caches.map { $0.snapshot() }
            StreamingMoEBlock.probeNoSync = true
            (_, lg) = try model.forwardHidden(prev, caches: caches)
            var (next, mg) = margin(lg[0])
            MLX.eval([next, mg] + caches.flatMap { $0.stateArrays })
            if mg.item(Float.self) < TH {                         // near-tie → sync 訂正
                syncTokens += 1
                for (i, c) in caches.enumerated() { c.restore(snaps[i], isLinear: isLin[i], trim: 1) }
                StreamingMoEBlock.probeNoSync = false
                (_, lg) = try model.forwardHidden(prev, caches: caches)
                next = MLX.argMax(lg[0, 0], axis: -1).reshaped([1, 1])
                MLX.eval([next] + caches.flatMap { $0.stateArrays })
            }
            out.append(prev.item(Int.self)); cur = next
        }
        StreamingMoEBlock.probeNoSync = false
        let secs = Double(DispatchTime.now().uptimeNanoseconds - t0.uptimeNanoseconds) / 1e9
        let match = zip(out, gR).filter { $0 == $1 }.count
        return String(format: """
            [margin] margin適応 fast(C=%d, TH=%.1f): %.1f tok/s  品質(greedy一致) %d/%d=%.0f%%  sync %d/%d token
            """,
            C, TH, Double(N) / secs, match, N, Double(match) / Double(N) * 100, syncTokens, N)
    }

    /// 適応 fast(lossless): fast(GPU-remap,sync無) forward → batched eval で miss 検出 →
    /// miss が出た token だけ sync 訂正。常に 100% 品質。cache coverage が十分なら redo 稀=高速。
    /// 研究(MoE-SpeQ/SP-MoE)の「予測 prefetch で sync 除去」を C を上げて lossless 実証する。
    public static func runAdaptiveFast(modelDir: String, refPath: String) throws -> String {
        guard let device = MTLCreateSystemDefaultDevice() else { return "ERROR: no Metal device" }
        let r = try loadArrays(url: URL(fileURLWithPath: refPath))
        guard let promptArr = r["spec_prompt"], let gRef = r["spec_greedy"] else { return "[adapt] skip" }
        let C = Int(ProcessInfo.processInfo.environment["QWISP_CACHE_C"] ?? "64") ?? 64
        let store = try WeightStore(modelDir: modelDir)
        store.residentNonExperts()
        let source = try ExpertSource(modelDir: modelDir); try source.warm()
        let arena = try ExpertArena(device: device, source: source, N: 64)
        let model = try StreamingQwispModel(store: store, arena: arena, device: device,
                                            source: source, cacheC: C)
        let ids = promptArr.asType(.int32).reshaped([1, promptArr.dim(0)])
        let gR = gRef.asArray(Int32.self).map { Int($0) }
        let isLin = model.isLinearFlags
        let N = Swift.min(Int(ProcessInfo.processInfo.environment["QWISP_GEN"] ?? "48") ?? 48, gR.count)
        let caches = model.makeCaches()

        StreamingMoEBlock.probeNoSync = false
        var (_, lg) = try model.prefillChunked(ids, caches: caches)
        var cur = MLX.argMax(lg[0, lg.dim(1) - 1], axis: -1).reshaped([1, 1])
        MLX.eval([cur] + caches.flatMap { $0.stateArrays })

        var out: [Int] = []; var syncTokens = 0
        let t0 = DispatchTime.now()
        for _ in 0 ..< N {
            let prev = cur
            let snaps = caches.map { $0.snapshot() }
            StreamingMoEBlock.probeNoSync = true
            (_, lg) = try model.forwardHidden(prev, caches: caches)
            var next = MLX.argMax(lg[0, 0], axis: -1).reshaped([1, 1])
            MLX.eval([next] + caches.flatMap { $0.stateArrays } + model.expertCaches.compactMap { $0.lastInds })
            let miss = model.expertCaches.reduce(0) { $0 + $1.missCount() }
            if miss > 0 {
                syncTokens += 1
                for (i, c) in caches.enumerated() { c.restore(snaps[i], isLinear: isLin[i], trim: 1) }
                StreamingMoEBlock.probeNoSync = false
                (_, lg) = try model.forwardHidden(prev, caches: caches)
                next = MLX.argMax(lg[0, 0], axis: -1).reshaped([1, 1])
                MLX.eval([next] + caches.flatMap { $0.stateArrays })
            }
            out.append(prev.item(Int.self)); cur = next
        }
        StreamingMoEBlock.probeNoSync = false
        let secs = Double(DispatchTime.now().uptimeNanoseconds - t0.uptimeNanoseconds) / 1e9
        let match = zip(out, gR).filter { $0 == $1 }.count
        return String(format: """
            [adapt] lossless adaptive fast(C=%d): %.1f tok/s  品質 %d/%d=%.0f%%  redo発火 %d/%d=%.0f%%
            """,
            C, Double(N) / secs, match, N, Double(match) / Double(N) * 100,
            syncTokens, N, Double(syncTokens) / Double(N) * 100)
    }

    /// async overlap 実現可能性テスト: fast forward を asyncEval→GPU 実行中に CPU で expert pread→eval。
    /// overlap すれば total ≈ max(GPU, pread)、しなければ GPU+pread。one-pass prefetch の土台確認。
    public static func runAsyncProbe(modelDir: String, refPath: String) throws -> String {
        guard let device = MTLCreateSystemDefaultDevice() else { return "ERROR: no Metal device" }
        let r = try loadArrays(url: URL(fileURLWithPath: refPath))
        guard let promptArr = r["spec_prompt"] else { return "[async] skip" }
        let store = try WeightStore(modelDir: modelDir)
        store.residentNonExperts()
        let source = try ExpertSource(modelDir: modelDir); try source.warm()
        let arena = try ExpertArena(device: device, source: source, N: 64)
        let model = try StreamingQwispModel(store: store, arena: arena, device: device,
                                            source: source, cacheC: 64)
        let ids = promptArr.asType(.int32).reshaped([1, promptArr.dim(0)])
        let caches = model.makeCaches()
        StreamingMoEBlock.probeNoSync = false
        _ = try model.prefillChunked(ids, caches: caches)
        var tok = MLX.argMax(MLX.zeros([1, 248320]), axis: -1).reshaped([1, 1])  // dummy
        tok = ids[0..., 0 ..< 1]
        MLX.eval([tok] + caches.flatMap { $0.stateArrays })

        // scratch buffer に ~120 expert(9テンソル) を pread する作業（per-token prefetch 相当）
        let sb = try source.sliceBytes(0, "gate_proj", "weight")
        let scratch = UnsafeMutableRawPointer.allocate(byteCount: sb * 16, alignment: 16)
        defer { scratch.deallocate() }
        func preadWork() {
            for e in 0 ..< 120 {
                try? source.preadInto(scratch.advanced(by: (e % 16) * sb), 0, "gate_proj", "weight", e % 256)
            }
        }
        func fwd() -> MLXArray {
            StreamingMoEBlock.probeNoSync = true
            let c = model.makeCaches()
            let (_, lg) = try! model.forwardHidden(tok, caches: c)
            return MLX.argMax(lg[0, 0], axis: -1)
        }
        func t(_ f: () -> Void) -> Double {
            for _ in 0 ..< 3 { f() }
            let s = DispatchTime.now(); for _ in 0 ..< 20 { f() }
            return Double(DispatchTime.now().uptimeNanoseconds - s.uptimeNanoseconds) / 1e6 / 20
        }
        let tGpu = t { let o = fwd(); o.eval() }
        let tPread = t { preadWork() }
        let tSeq = t { let o = fwd(); o.eval(); preadWork() }      // 逐次
        let tAsync = t { let o = fwd(); MLX.asyncEval([o]); preadWork(); o.eval() }  // 同一スレッド
        // 別スレッドで pread を forward と並行（真の async prefetch パターン）
        let tBg = t {
            let sem = DispatchSemaphore(value: 0)
            DispatchQueue.global(qos: .userInitiated).async { preadWork(); sem.signal() }
            let o = fwd(); o.eval()
            sem.wait()
        }
        StreamingMoEBlock.probeNoSync = false
        let best = Swift.min(tAsync, tBg)
        return String(format: """
            [async] overlap 可能性: GPU forward=%.1fms  pread(120exp)=%.1fms  逐次=%.1fms
               同一スレ async=%.1fms  別スレ bg=%.1fms  理論下限 max=%.1fms
               → overlap %@ (best %.1fms / 逐次 %.1fms = %.0f%% 隠蔽, 完全なら %.0f%%)
            """,
            tGpu, tPread, tSeq, tAsync, tBg, Swift.max(tGpu, tPread),
            best < tSeq * 0.9 ? "成立✅" : "弱い⚠️", best, tSeq,
            (tSeq - best) / tPread * 100, (tSeq - Swift.max(tGpu, tPread)) / tPread * 100)
    }

    /// 2-pass cross-layer 予測 fast(Fate 流): pass-1 で各層 routing を予測→prefetch→pass-2 GPU-remap。
    /// 予測カバー率が高ければ誤差小で安定するはず（fast mode の 31%誤り→~3%へ）。品質+速度を測る。
    public static func runCrossLayerFast(modelDir: String, refPath: String) throws -> String {
        guard let device = MTLCreateSystemDefaultDevice() else { return "ERROR: no Metal device" }
        let r = try loadArrays(url: URL(fileURLWithPath: refPath))
        guard let promptArr = r["spec_prompt"], let gRef = r["spec_greedy"] else { return "[xlayer] skip" }
        let C = Int(ProcessInfo.processInfo.environment["QWISP_CACHE_C"] ?? "64") ?? 64
        let store = try WeightStore(modelDir: modelDir)
        store.residentNonExperts()
        let source = try ExpertSource(modelDir: modelDir); try source.warm()
        let arena = try ExpertArena(device: device, source: source, N: 64)
        let model = try StreamingQwispModel(store: store, arena: arena, device: device,
                                            source: source, cacheC: C)
        let ids = promptArr.asType(.int32).reshaped([1, promptArr.dim(0)])
        let gR = gRef.asArray(Int32.self).map { Int($0) }
        let isLin = model.isLinearFlags
        let N = Swift.min(Int(ProcessInfo.processInfo.environment["QWISP_GEN"] ?? "48") ?? 48, gR.count)
        let caches = model.makeCaches()

        StreamingMoEBlock.probeNoSync = false
        var (_, lg) = try model.prefillChunked(ids, caches: caches)
        var cur = MLX.argMax(lg[0, lg.dim(1) - 1], axis: -1).reshaped([1, 1])
        MLX.eval([cur] + caches.flatMap { $0.stateArrays })

        var out: [Int] = []
        let t0 = DispatchTime.now()
        for _ in 0 ..< N {
            let prev = cur
            let snaps = caches.map { $0.snapshot() }
            // pass-1: 軽量予測（routed gather 省略, shared expert のみ）→ 各層 inds 捕捉
            let lightPredict = ProcessInfo.processInfo.environment["QWISP_LIGHT"] == "1"  // 既定=full(品質)
            StreamingMoEBlock.predictOnly = lightPredict
            StreamingMoEBlock.probeNoSync = !lightPredict
            _ = try model.forwardHidden(prev, caches: caches)
            MLX.eval(caches.flatMap { $0.stateArrays } + model.expertCaches.compactMap { $0.lastInds })
            StreamingMoEBlock.predictOnly = false
            // prefetch 予測 experts
            for c in model.expertCaches { c.prefetchLastInds() }
            // pass-1 の KV/GDN 変化を rollback
            for (i, c) in caches.enumerated() { c.restore(snaps[i], isLinear: isLin[i], trim: 1) }
            // pass-2: 本番（warmed cache で GPU-remap）
            StreamingMoEBlock.probeNoSync = true
            (_, lg) = try model.forwardHidden(prev, caches: caches)
            let next = MLX.argMax(lg[0, 0], axis: -1).reshaped([1, 1])
            MLX.eval([next] + caches.flatMap { $0.stateArrays })
            out.append(prev.item(Int.self)); cur = next
        }
        StreamingMoEBlock.probeNoSync = false
        let secs = Double(DispatchTime.now().uptimeNanoseconds - t0.uptimeNanoseconds) / 1e9
        let match = zip(out, gR).filter { $0 == $1 }.count
        return String(format: """
            [xlayer] 2-pass cross-layer fast(C=%d): %.1f tok/s  品質(greedy一致) %d/%d=%.0f%%
            """,
            C, Double(N) / secs, match, N, Double(match) / Double(N) * 100)
    }

    /// MTP D1 投機 on streaming（verify が ~1.85 トークン/forward を生成→per-layer sync を償却）。
    public static func runSpeculative(modelDir: String, refPath: String) throws -> String {
        guard let device = MTLCreateSystemDefaultDevice() else { return "ERROR: no Metal device" }
        let r = try loadArrays(url: URL(fileURLWithPath: refPath))
        guard let promptArr = r["spec_prompt"], let gRef = r["spec_greedy"], let spRef = r["spec_spec"] else {
            return "[M2c×stream] skip: spec_prompt 無し"
        }
        let C = Int(ProcessInfo.processInfo.environment["QWISP_CACHE_C"] ?? "64") ?? 64
        let store = try WeightStore(modelDir: modelDir)
        store.residentNonExperts()
        let source = try ExpertSource(modelDir: modelDir)
        try source.warm()
        let arena = try ExpertArena(device: device, source: source, N: 64)
        let model = try StreamingQwispModel(store: store, arena: arena, device: device,
                                            source: source, cacheC: C)
        let head = try MTPHead(modelDir: modelDir, store: store)   // MTP head は resident(~400MB)
        let rssLoad = rssGB()

        let ids = promptArr.asType(.int32).reshaped([1, promptArr.dim(0)])
        let N = Swift.min(Int(ProcessInfo.processInfo.environment["QWISP_GEN"] ?? "48") ?? 48, spRef.dim(0))
        let gR = gRef.asArray(Int32.self).map { Int($0) }
        // 先に streaming greedy が Python greedy と一致するか（forward の正しさを切り分け）
        let sgCaches = model.makeCaches()
        var (_, glg) = try model.prefillChunked(ids, caches: sgCaches)
        var gnext = MLX.argMax(glg[0, glg.dim(1) - 1], axis: -1).reshaped([1, 1])
        MLX.eval([gnext] + sgCaches.flatMap { $0.stateArrays })
        var sg: [Int] = []
        for _ in 0 ..< N {
            sg.append(gnext.item(Int.self))
            (_, glg) = try model.forwardHidden(gnext, caches: sgCaches)
            gnext = MLX.argMax(glg[0, 0], axis: -1).reshaped([1, 1])
            MLX.eval([gnext] + sgCaches.flatMap { $0.stateArrays })
        }
        let sgMatch = zip(sg, gR).filter { $0 == $1 }.count
        let (sp, steps, acc, secs) = try specLoop(model, head, ids, N, diagGreedy: gR)
        let spR = spRef.asArray(Int32.self).map { Int($0) }
        let vsPython = zip(sp, spR).filter { $0 == $1 }.count
        let vsGreedy = zip(sp, gR).filter { $0 == $1 }.count
        let rssPeak = rssGB()
        let ok = vsPython == N
        return String(format: """
            [M2c×stream] MTP 投機 on 8GB streaming(C=%d): streaming greedy=Python %d/%d  spec vs Python %d/%d vs greedy %d/%d %@
               accept=%.3f  %.1f tok/s  RSS load=%.1fGB peak=%.1fGB
            """,
            C, sgMatch, N, vsPython, N, vsGreedy, N, ok ? "OK ✅" : "❌",
            Double(acc) / Double(steps), Double(N) / secs, rssLoad, rssPeak)
    }

    /// streaming model 用の D1 投機ループ（Speculative.speculative の streaming 版）。
    /// refreshEvery>0 で verify を hybrid no-sync fast（refreshEvery 毎に sync で cache 更新）。
    static func specLoop(_ model: StreamingQwispModel, _ head: MTPHead, _ promptIds: MLXArray,
                         _ maxTokens: Int, refreshEvery: Int = 0, diagGreedy: [Int]? = nil)
        throws -> (toks: [Int], steps: Int, acc: Int, secs: Double) {
        let mainCaches = model.makeCaches()
        let mtpKV = KVCache()
        let isLin = model.isLinearFlags
        let P = promptIds.dim(-1)
        let diag = ProcessInfo.processInfo.environment["QWISP_SPEC_DIAG"] == "1" && diagGreedy != nil
        var diagDone = false
        let forceReject = ProcessInfo.processInfo.environment["QWISP_FORCE_REJECT"] == "1"
        let acceptResync = ProcessInfo.processInfo.environment["QWISP_ACCEPT_RESYNC"] == "1"
        if ProcessInfo.processInfo.environment["QWISP_F32_ATTN"] == "1" { AttentionLayer.f32SDPA = true }
        // verify の attention を逐次化＝true greedy-lossless（既定 ON）。QWISP_BATCHED_VERIFY=1 で旧 batched。
        let seqAttn = ProcessInfo.processInfo.environment["QWISP_BATCHED_VERIFY"] != "1"
        StreamingMoEBlock.probeNoSync = false
        let (H, lg) = try model.prefillChunked(promptIds, caches: mainCaches)
        var uArr = MLX.argMax(lg[0..., (lg.dim(1) - 1)...], axis: -1)
        var lastH = H[0..., (P - 1)...]
        _ = head(H[0..., 0 ..< (P - 1)], promptIds[0..., 1...], cache: mtpKV)
        MLX.eval([uArr, lastH] + [mtpKV.keys, mtpKV.values].compactMap { $0 } + mainCaches.flatMap { $0.stateArrays })

        // prefill/head 後にのみ有効化: verify([u,v])の attention を逐次化（prefill は batched のまま）
        if seqAttn { AttentionLayer.seqMultiToken = true }
        var out: [Int] = []; var steps = 0, acc = 0
        let t0 = DispatchTime.now()
        while out.count < maxTokens {
            // fast モード: refreshEvery 毎に sync(cache 更新)、それ以外は GPU-remap no-sync
            StreamingMoEBlock.probeNoSync = refreshEvery > 0 && (steps % refreshEvery != 0)
            steps += 1
            let dl = head(lastH, uArr, cache: mtpKV)
            let dArr = MLX.argMax(dl[0..., 0...], axis: -1)
            let ud = MLX.concatenated([uArr, dArr], axis: 1)
            let snaps = mainCaches.map { $0.snapshot() }
            let (H2, lg2) = try model.forwardHidden(ud, caches: mainCaches)
            let vw = MLX.argMax(lg2[0, 0 ..< 2], axis: -1)
            let vals = MLX.concatenated([dArr[0], vw]).asArray(Int32.self)
            var d = Int(vals[0]); let v = Int(vals[1])
            if forceReject { d = -1 }                          // 常に reject → commit 状態は [u] 再forward 由来のみ
            // 診断: 検証の位置-u argmax(=greedy 次トークンであるべき)を greedy ref と照合し、
            // 初回乖離点で top-2 margin を出す（near-tie=GDN batched 数値, clean gap=logic bug）。
            if diag && !diagDone, let gr = diagGreedy, out.count + 1 < gr.count {
                let vwu = Int(vals[1])                          // = vw[0] = verify argmax @ position u
                let expected = gr[out.count + 1]                // greedy token AFTER u (u itself = gr[out.count])
                if vwu != expected {
                    let row = lg2[0, 0]                          // [V] logits @ position u
                    let top1 = row.max().item(Float.self)
                    let i1 = MLX.argMax(row, axis: -1).item(Int.self)
                    let masked = MLX.where(MLX.arange(row.dim(0)) .== Int32(i1), MLXArray(-1e30 as Float), row)
                    let top2 = masked.max().item(Float.self)
                    let grLogit = row[expected].item(Float.self)
                    print(String(format: "[SPEC-DIAG] first diverge @out=%d step=%d accept=%@  vw_u=%d expected=%d  "
                        + "top1(id=%d)=%.4f top2=%.4f margin=%.4f  greedy_tok_logit=%.4f gap=%.4f",
                        out.count, steps, (v == d) ? "Y" : "N", vwu, expected, i1, top1, top2, top1 - top2,
                        grLogit, top1 - grLogit))
                    diagDone = true
                }
            }
            out.append(uArr.item(Int.self))
            if v == d {
                acc += 1; out.append(d)
                _ = head(H2[0..., 0 ..< 1], dArr, cache: mtpKV)
                if acceptResync {
                    // 検証: commit 状態を batched[u,v] でなく逐次(u→v 各単一 token)で作り直す。
                    for (i, c) in mainCaches.enumerated() { c.restore(snaps[i], isLinear: isLin[i], trim: 2) }
                    _ = try model.forwardHidden(uArr, caches: mainCaches)              // u 単独
                    let (Hv, lgv) = try model.forwardHidden(dArr, caches: mainCaches)  // v 単独
                    uArr = MLX.argMax(lgv[0, 0], axis: -1).reshaped([1, 1]); lastH = Hv[0..., 0 ..< 1]
                } else {
                    uArr = vw[1 ..< 2].reshaped([1, 1]); lastH = H2[0..., 1 ..< 2]
                }
            } else {                                       // reject → [u] のみ再投入(look-ahead 重複回避)
                for (i, c) in mainCaches.enumerated() { c.restore(snaps[i], isLinear: isLin[i], trim: 2) }
                let (H1, _) = try model.forwardHidden(uArr, caches: mainCaches)
                uArr = vw[0 ..< 1].reshaped([1, 1]); lastH = H1[0..., 0 ..< 1]
            }
            MLX.eval([uArr, lastH] + [mtpKV.keys, mtpKV.values].compactMap { $0 } + mainCaches.flatMap { $0.stateArrays })
        }
        uArr.eval()
        StreamingMoEBlock.probeNoSync = false
        AttentionLayer.seqMultiToken = false   // global static を後続(M0/M2/M4)に漏らさない
        return (Array(out.prefix(maxTokens)), steps, acc,
                Double(DispatchTime.now().uptimeNanoseconds - t0.uptimeNanoseconds) / 1e9)
    }

    /// MTP × hybrid fast: 投機(verify ~1.85トークン) × no-sync fast。最速モード。
    public static func runSpeculativeFast(modelDir: String, refPath: String) throws -> String {
        guard let device = MTLCreateSystemDefaultDevice() else { return "ERROR: no Metal device" }
        let r = try loadArrays(url: URL(fileURLWithPath: refPath))
        guard let promptArr = r["spec_prompt"], let gRef = r["spec_greedy"] else {
            return "[M2c×fast] skip"
        }
        let C = Int(ProcessInfo.processInfo.environment["QWISP_CACHE_C"] ?? "64") ?? 64
        let RE = Int(ProcessInfo.processInfo.environment["QWISP_REFRESH"] ?? "8") ?? 8
        let store = try WeightStore(modelDir: modelDir)
        store.residentNonExperts()
        let source = try ExpertSource(modelDir: modelDir); try source.warm()
        let arena = try ExpertArena(device: device, source: source, N: 64)
        let model = try StreamingQwispModel(store: store, arena: arena, device: device,
                                            source: source, cacheC: C)
        let head = try MTPHead(modelDir: modelDir, store: store)
        let ids = promptArr.asType(.int32).reshaped([1, promptArr.dim(0)])
        let gR = gRef.asArray(Int32.self).map { Int($0) }
        let N = 48
        let (sp, steps, acc, secs) = try specLoop(model, head, ids, N, refreshEvery: RE)
        let match = zip(sp, gR).filter { $0 == $1 }.count
        return String(format: """
            [M2c×fast] MTP投機 × hybrid no-sync(C=%d, sync 1/%d): %.1f tok/s  accept=%.3f  品質(greedy一致) %d/%d=%.0f%%  RSS=%.1fGB
            """,
            C, RE, Double(N) / secs, Double(acc) / Double(steps), match, N,
            Double(match) / Double(N) * 100, rssGB())
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

        let caches = model.makeCaches()
        var (_, logits) = try model.prefillChunked(ids, caches: caches)   // |U|>C 破綻回避
        var next = MLX.argMax(logits[0, logits.dim(1) - 1], axis: -1).reshaped([1, 1])
        MLX.eval([next] + caches.flatMap { $0.stateArrays })

        let N = 32
        var toks: [Int] = []
        LayerExpertCache.ensureNanos = 0; StreamingMoEBlock.syncNanos = 0
        let t0 = DispatchTime.now()
        for _ in 0 ..< N {
            logits = try model(next, caches: caches)
            next = MLX.argMax(logits[0, 0], axis: -1).reshaped([1, 1])
            MLX.eval([next] + caches.flatMap { $0.stateArrays })
            toks.append(next.item(Int.self))
        }
        let secs = Double(DispatchTime.now().uptimeNanoseconds - t0.uptimeNanoseconds) / 1e9
        let ensureMs = Double(LayerExpertCache.ensureNanos) / 1e6 / Double(N)
        let syncMs = Double(StreamingMoEBlock.syncNanos) / 1e6 / Double(N)

        // 天井計測: GPU remap, per-layer sync 無し（warmup 後の cache を凍結利用、出力は近似）
        StreamingMoEBlock.probeNoSync = true
        var pn = next
        for _ in 0 ..< 4 { let l = try model(pn, caches: caches); pn = MLX.argMax(l[0, 0], axis: -1).reshaped([1, 1]); pn.eval() }
        let tp = DispatchTime.now()
        for _ in 0 ..< N {
            let l = try model(pn, caches: caches)
            pn = MLX.argMax(l[0, 0], axis: -1).reshaped([1, 1])
            MLX.eval([pn] + caches.flatMap { $0.stateArrays })
        }
        let probeSecs = Double(DispatchTime.now().uptimeNanoseconds - tp.uptimeNanoseconds) / 1e9
        StreamingMoEBlock.probeNoSync = false
        let probeTps = Double(N) / probeSecs
        let rssPeak = rssGB()
        let hits = model.expertCaches.reduce(0) { $0 + $1.hits }
        let misses = model.expertCaches.reduce(0) { $0 + $1.misses }
        let hitRate = hits + misses > 0 ? Double(hits) / Double(hits + misses) * 100 : 0

        return String(format: """
            [S3] streaming decode(8GB狙い, LRU cache C=%d/層, experts 非常駐):
               %.1f tok/s (%.1f ms/tok)  RSS: load=%.1fGB peak=%.1fGB
               cache hit=%.0f%% (hit=%d miss=%d)  内訳/tok: sync=%.1fms ensure(load+IO)=%.1fms
               天井(GPU remap, sync無): %.1f tok/s ← sync 除去の上限
               生成=%@
            """,
            C, Double(N) / secs, secs / Double(N) * 1000, rssLoad, rssPeak,
            hitRate, hits, misses, syncMs, ensureMs, probeTps, "\(toks.prefix(6))")
    }
}
