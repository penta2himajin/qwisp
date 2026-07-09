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

    /// ★ issue#7 (c): 背景 prefetch overlap + whole-token no-sync + escalate-from-first-miss。
    /// 前トークンの per-layer inds(priorInds)を hint に背景スレッドが全層を先行 ensure（per-layer
    /// semaphore=CPU 同期、GPU barrier 無し）。main は whole-token を no-sync lazy 実行（per-layer drain
    /// 排除）→ 単一 eval + per-layer miss。miss 層から sync 再計算で exact 維持。lastInds に当 token の
    /// 実 routing を残す（次 token prefetch 用）。priorInds=nil の token は全層 sync(cold-start 初手)。
    public func forwardHiddenPrefetchWhole(_ ids: MLXArray, caches: [LayerCache], priorInds: [[Int]]?, isLin: [Bool])
        throws -> (hidden: MLXArray, logits: MLXArray, esc: Int) {
        let L = numLayers
        let trim = ids.dim(1)
        // 背景 prefetch（前トークン inds を全層先行 ensure）
        let sem = (0 ..< L).map { _ in DispatchSemaphore(value: 0) }
        if let pi = priorInds {
            let q = DispatchQueue(label: "qwisp.prefetch")
            q.async { [expertCaches] in
                for i in 0 ..< L { _ = expertCaches[i].ensure(pi[i]); sem[i].signal() }
            }
        } else {
            for i in 0 ..< L { sem[i].signal() }   // prior 無し＝prefetch せず（全層 miss→escalate=実質 sync）
        }
        let snaps = caches.map { $0.snapshot() }
        StreamingMoEBlock.captureInds = true      // 当 token の実 routing を lastInds に残す
        var h = embed(ids)
        var hInputs: [MLXArray] = []; var perLayerMiss: [MLXArray] = []
        let nosync = priorInds != nil
        for i in 0 ..< L {
            sem[i].wait()                         // 背景が層 i を ensure 済（CPU-CPU 同期, GPU 非 barrier）
            hInputs.append(h)
            StreamingMoEBlock.hotMissAccum = nil
            StreamingMoEBlock.probeNoSync = nosync
            StreamingMoEBlock.countHotMiss = nosync
            h = try layers[i](h, cache: caches[i])
            perLayerMiss.append(StreamingMoEBlock.hotMissAccum ?? MLXArray(Int32(0)))
        }
        StreamingMoEBlock.probeNoSync = false; StreamingMoEBlock.countHotMiss = false
        let hid0 = MLXFast.rmsNorm(h, weight: store.req("language_model.model.norm.weight"), eps: eps)
        let lg0 = headProj().apply(hid0)
        // 単一 eval（per-layer miss + 全 cache state + lastInds + logits）
        MLX.eval([lg0] + perLayerMiss + caches.flatMap { $0.stateArrays } + expertCaches.compactMap { $0.lastInds })
        // 最初の miss 層を特定
        var firstMiss = -1
        if nosync { for i in 0 ..< L where perLayerMiss[i].item(Int32.self) != 0 { firstMiss = i; break } }
        if firstMiss < 0 {
            StreamingMoEBlock.captureInds = false
            return (hid0, lg0, 0)                  // 全層 exact（no-sync が全常駐）
        }
        // escalate: miss 層以降を sync 再計算（cache 復元）。層 0..firstMiss-1 は resident＝exact, 保持。
        for i in firstMiss ..< L { caches[i].restore(snaps[i], isLinear: isLin[i], trim: trim) }
        var hh = hInputs[firstMiss]
        for i in firstMiss ..< L { hh = try layers[i](hh, cache: caches[i]) }   // sync(ensure で miss ロード)
        StreamingMoEBlock.captureInds = false
        let hid = MLXFast.rmsNorm(hh, weight: store.req("language_model.model.norm.weight"), eps: eps)
        let lg = headProj().apply(hid)
        MLX.eval([lg] + caches.flatMap { $0.stateArrays } + expertCaches.compactMap { $0.lastInds })
        return (hid, lg, L - firstMiss)
    }

    /// ★ issue#7 layer-batch: chunk(K 層)単位の cross-layer 予測 prefetch + no-sync + chunk内 escalate。
    /// 各 chunk: 入力 h を materialize→K 層を一括予測(gate_{j+d}(h))→prefetch→chunk no-sync lazy→単一 eval
    /// →chunk 内 first-miss から sync 再計算。drain=2/chunk(40/K に削減)、escalate は chunk 内に bounded。
    /// 短距離予測ゆえ高 cover(marginK=32,d≤3 で 54-83%)。bit-exact。
    public func forwardHiddenChunked(_ ids: MLXArray, caches: [LayerCache], K: Int, marginK: Int, isLin: [Bool])
        throws -> (hidden: MLXArray, logits: MLXArray, esc: Int) {
        let L = numLayers, trim = ids.dim(1)
        func distinct(_ a: MLXArray) -> [Int] { var s = Set<Int>(); var u: [Int] = []; for e in a.asArray(Int32.self) { let v = Int(e); if s.insert(v).inserted { u.append(v) } }; return u }
        var h = embed(ids); var esc = 0
        StreamingMoEBlock.captureInds = true
        var j = 0
        while j < L {
            let end = Swift.min(j + K, L)
            h.eval()                                            // chunk 入力 materialize（drain 1/chunk）
            let h2 = h.reshaped([1, h.dim(-1)])
            // K 層を一括予測（gate_{layer}(h)）→ eval 1 回 → prefetch
            var preds: [MLXArray] = []
            for layer in j ..< end { preds.append(predictLayerIndsK(layer, h2, marginK)) }
            MLX.eval(preds)
            for (idx, layer) in (j ..< end).enumerated() { _ = expertCaches[layer].ensure(distinct(preds[idx])) }
            // chunk を no-sync lazy 実行 + per-layer miss
            let snaps = (j ..< end).map { caches[$0].snapshot() }
            var hIn: [MLXArray] = []; var miss: [MLXArray] = []
            for layer in j ..< end {
                hIn.append(h)
                StreamingMoEBlock.hotMissAccum = nil
                StreamingMoEBlock.probeNoSync = true; StreamingMoEBlock.countHotMiss = true
                h = try layers[layer](h, cache: caches[layer])
                miss.append(StreamingMoEBlock.hotMissAccum ?? MLXArray(Int32(0)))
            }
            StreamingMoEBlock.probeNoSync = false; StreamingMoEBlock.countHotMiss = false
            MLX.eval([h] + miss + (j ..< end).flatMap { caches[$0].stateArrays })   // chunk 末尾 eval（drain 1/chunk）
            // chunk 内 first-miss から sync 再計算
            var fm = -1
            for (idx, layer) in (j ..< end).enumerated() where miss[idx].item(Int32.self) != 0 { fm = layer; break }
            if fm >= 0 {
                for layer in fm ..< end { caches[layer].restore(snaps[layer - j], isLinear: isLin[layer], trim: trim) }
                var hh = hIn[fm - j]
                for layer in fm ..< end { hh = try layers[layer](hh, cache: caches[layer]) }   // sync(ensure)
                MLX.eval([hh] + (fm ..< end).flatMap { caches[$0].stateArrays })
                h = hh; esc += end - fm
            }
            j = end
        }
        StreamingMoEBlock.captureInds = false
        let hid = MLXFast.rmsNorm(h, weight: store.req("language_model.model.norm.weight"), eps: eps)
        let lg = headProj().apply(hid); MLX.eval([lg])
        return (hid, lg, esc)
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
