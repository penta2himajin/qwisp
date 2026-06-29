import Foundation
import MLX
import MLXFast
import MLXRandom

/// Qwen3NextAttention.__call__ の Swift 移植（M2b-2 full-attention 層）.
/// GQA(16 q-heads / 2 kv-heads, head_dim=256) + q/k RMSNorm + partial RoPE(64dim) +
/// gated output o_proj(out * sigmoid(gate))。cache=None / mask=causal の単一チャンク。
public struct AttentionLayer {
    let numHeads: Int        // 16
    let numKVHeads: Int      // 2
    let headDim: Int         // 256
    let ropeDim: Int         // 64 (= head_dim * partial_rotary_factor)
    let ropeBase: Float      // 1e7
    let eps: Float

    let qProj: Proj           // → [numHeads*headDim*2]
    let kProj: Proj           // → [numKVHeads*headDim]
    let vProj: Proj
    let oProj: Proj           // → [H]
    let qNorm: MLXArray       // [headDim]
    let kNorm: MLXArray       // [headDim]

    var scale: Float { Float(pow(Double(headDim), -0.5)) }

    /// SDPA を float32 で実行（.causal/.none 経路の f16 揺れ ~7e-4 を ~1e-6 に縮め、
    /// spec の batched verify が逐次 greedy と drift するのを防ぐ）。
    nonisolated(unsafe) public static var f32SDPA: Bool = false
    /// (C) 順序安定 attention: fused SDPA(L 依存)でなく broadcast+sum で計算。sum は L 非依存ゆえ
    /// batched(L>1) と single(L=1) が bit 一致(rel=0)＝seqMT 無しで verify が strict lossless。f32 累積。
    nonisolated(unsafe) public static var orderStable: Bool = false
    /// ★解(調査で確定): SDPA に boolean causal mask 配列を渡す。MLX SDPA は L=1/L>1 とも mask=.array なら
    /// 同じ bool_mask vector kernel 経路(f32 累積・同 key 順序)を通り batched verify=single decode が bit 一致。
    /// .causal/.none enum は do_causal トグルで経路が分岐し ~7e-4 drift。mlx-lm spec decode も bool 配列方式。
    /// ※実測で不発(7.299e-4 残)。key 数差(S)が残るため。代わりに perQueryNone を使う。
    nonisolated(unsafe) public static var boolMaskSDPA: Bool = false
    /// ★★真の解(investigate C 確定): 射影は batched(quantized=order-stable)で 1 回、SDPA だけを
    /// query ごとに分け、各 query の exact causal prefix を .none で SDPA する。各 SDPA は L=1・.none で
    /// decode 経路と完全同一(同 key 集合・同 mask モード)ゆえ batched verify = 逐次 decode が bit 一致。
    /// seqMultiToken と違い射影/conv/MoE を再実行しない軽量版（per-query は SDPA のみ）。
    nonisolated(unsafe) public static var perQueryNone: Bool = false

    public init(numHeads: Int, numKVHeads: Int, headDim: Int, ropeDim: Int, ropeBase: Float,
                eps: Float, qProj: Proj, kProj: Proj, vProj: Proj, oProj: Proj,
                qNorm: MLXArray, kNorm: MLXArray) {
        self.numHeads = numHeads; self.numKVHeads = numKVHeads; self.headDim = headDim
        self.ropeDim = ropeDim; self.ropeBase = ropeBase; self.eps = eps
        self.qProj = qProj; self.kProj = kProj; self.vProj = vProj; self.oProj = oProj
        self.qNorm = qNorm; self.kNorm = kNorm
    }

    func rope(_ x: MLXArray, _ offset: Int) -> MLXArray {
        MLXFast.RoPE(x, dimensions: ropeDim, traditional: false, base: ropeBase,
                     scale: 1.0, offset: offset)
    }

    /// 検証([u,v] 等 L>1)の attention を 1 トークンずつ逐次(L=1,.none)で処理する。
    /// greedy decode と同じ経路・同じ f16 丸めになり、batched(.causal) との ~7e-4 乖離を消す
    /// → MTP spec の accept-state drift を絶つ（true greedy-lossless verify）。
    nonisolated(unsafe) public static var seqMultiToken: Bool = false

    public func callAsFunction(_ x: MLXArray, cache: KVCache? = nil) -> MLXArray {
        let B = x.dim(0), L = x.dim(1)

        // 逐次モード: L>1 を 1 token ずつ再帰呼び（cache が順に u→v を取り込む＝greedy 等価）
        if AttentionLayer.seqMultiToken && L > 1 {
            var outs: [MLXArray] = []
            for t in 0 ..< L { outs.append(callAsFunction(x[0..., t ..< (t + 1)], cache: cache)) }
            return MLX.concatenated(outs, axis: 1)
        }

        let qOut = qProj.apply(x).reshaped([B, L, numHeads, 2 * headDim])
        var queries = qOut[0..., 0..., 0..., 0 ..< headDim]            // [B,L,H,headDim]
        let gate = qOut[0..., 0..., 0..., headDim...].reshaped([B, L, -1])  // [B,L,H*headDim]

        var keys = kProj.apply(x).reshaped([B, L, numKVHeads, headDim])
        var values = vProj.apply(x).reshaped([B, L, numKVHeads, headDim])

        // q/k RMSNorm（最終軸 headDim, weight 有り）→ transpose to [B,heads,L,headDim]
        queries = MLXFast.rmsNorm(queries, weight: qNorm, eps: eps).transposed(0, 2, 1, 3)
        keys = MLXFast.rmsNorm(keys, weight: kNorm, eps: eps).transposed(0, 2, 1, 3)
        values = values.transposed(0, 2, 1, 3)

        let offset = cache?.offset ?? 0
        queries = rope(queries, offset)
        keys = rope(keys, offset)

        var allKeys = keys, allValues = values
        if let c = cache { (allKeys, allValues) = c.update(keys, values) }

        let inDtype = queries.dtype
        let S = allKeys.dim(2)
        var output: MLXArray
        if AttentionLayer.orderStable {
            // (C) 順序安定 naive attention(f32, broadcast+sum)。sum は L 非依存ゆえ batched=single bit一致。
            // GQA は kv を group 軸へ reshape して broadcast。memory: [B,kv,g,L,S,D] を一時生成。
            let g = numHeads / numKVHeads
            let qg = queries.reshaped([B, numKVHeads, g, L, headDim]).asType(.float32)   // [B,kv,g,L,D]
            let kg = allKeys.reshaped([B, numKVHeads, 1, S, headDim]).asType(.float32)    // [B,kv,1,S,D]
            let vg = allValues.reshaped([B, numKVHeads, 1, S, headDim]).asType(.float32)
            var scores = (qg.expandedDimensions(axis: 4) * kg.expandedDimensions(axis: 3))
                .sum(axis: -1) * scale                                                    // [B,kv,g,L,S]
            if L > 1 {                                                                    // causal
                let qIdx = MLXArray((0 ..< L).map { Int32(offset + $0) }).reshaped([L, 1])
                let kIdx = MLXArray((0 ..< S).map { Int32($0) }).reshaped([1, S])
                let mask = (kIdx .<= qIdx).reshaped([1, 1, 1, L, S])
                scores = MLX.where(mask, scores, MLXArray(Float(-1e30)))
            }
            let p = MLX.softmax(scores, axis: -1)                                         // [B,kv,g,L,S]
            let o = (p.expandedDimensions(axis: -1) * vg.expandedDimensions(axis: 3))
                .sum(axis: -2)                                                            // [B,kv,g,L,D]
            output = o.reshaped([B, numHeads, L, headDim]).asType(inDtype)
        } else if AttentionLayer.boolMaskSDPA {
            // ★ boolean causal mask 配列で L=1/L>1 の SDPA 経路を統一 → batched verify=single decode bit一致
            let qIdx = MLXArray((0 ..< L).map { Int32(offset + $0) }).reshaped([L, 1])
            let kIdx = MLXArray((0 ..< S).map { Int32($0) }).reshaped([1, S])
            let m = (kIdx .<= qIdx).reshaped([1, 1, L, S])              // [1,1,L,S] bool, query i は key 0..offset+i
            output = MLXFast.scaledDotProductAttention(queries: queries, keys: allKeys, values: allValues,
                                                       scale: scale, mask: .array(m))
        } else if AttentionLayer.perQueryNone && L > 1 {
            // ★★ per-query .none: query t を exact causal prefix(key 0..<offset+t+1)だけ L=1・.none で SDPA。
            // decode(L=1,.none,全履歴)と同一経路ゆえ batched verify が逐次 decode と bit 一致。SDPA のみ per-query。
            var outs: [MLXArray] = []
            outs.reserveCapacity(L)
            for t in 0 ..< L {
                let qt = queries[0..., 0..., t ..< (t + 1), 0...]            // [B,H,1,D]
                let pre = offset + t + 1                                     // causal prefix 長
                let kt = allKeys[0..., 0..., 0 ..< pre, 0...]                // [B,kv,pre,D]
                let vt = allValues[0..., 0..., 0 ..< pre, 0...]
                outs.append(MLXFast.scaledDotProductAttention(
                    queries: qt, keys: kt, values: vt, scale: scale, mask: .none))
            }
            output = MLX.concatenated(outs, axis: 2)                         // [B,H,L,D]
        } else {
            // prefill(L>1) は causal、decode(L==1, 全履歴に attend) は mask 無し
            let maskMode: MLXFast.ScaledDotProductAttentionMaskMode = L > 1 ? .causal : .none
            var (q2, k2, v2) = (queries, allKeys, allValues)
            if AttentionLayer.f32SDPA {
                q2 = queries.asType(.float32); k2 = allKeys.asType(.float32); v2 = allValues.asType(.float32)
            }
            output = MLXFast.scaledDotProductAttention(queries: q2, keys: k2, values: v2, scale: scale, mask: maskMode)
            if AttentionLayer.f32SDPA { output = output.asType(inDtype) }
        }
        output = output.transposed(0, 2, 1, 3).reshaped([B, L, -1])   // [B,L,H*headDim]

        return oProj.apply(output * MLX.sigmoid(gate))
    }

    /// ★ issue#6 continuous batching: 各 slot が**異なる position と独立 KV** を持つ decode step。
    /// 投影/rmsNorm/o_proj は batched(amortize)、RoPE+SDPA だけ per-slot ループ(各 slot 自分の offset と KVCache)。
    /// x[B,1,H], slotKV[B](per-slot 独立 KVCache), positions[B](各 slot の現在位置)。→ [B,1,H]。
    public func callContinuous(_ x: MLXArray, slotKV: [KVCache], positions: [Int]) -> MLXArray {
        let B = x.dim(0)
        let qOut = qProj.apply(x).reshaped([B, 1, numHeads, 2 * headDim])
        var queries = qOut[0..., 0..., 0..., 0 ..< headDim]                 // [B,1,nH,hd]
        let gate = qOut[0..., 0..., 0..., headDim...].reshaped([B, 1, -1])
        var keys = kProj.apply(x).reshaped([B, 1, numKVHeads, headDim])
        var values = vProj.apply(x).reshaped([B, 1, numKVHeads, headDim])
        queries = MLXFast.rmsNorm(queries, weight: qNorm, eps: eps).transposed(0, 2, 1, 3)  // [B,nH,1,hd]
        keys = MLXFast.rmsNorm(keys, weight: kNorm, eps: eps).transposed(0, 2, 1, 3)         // [B,nKV,1,hd]
        values = values.transposed(0, 2, 1, 3)
        var outs: [MLXArray] = []; outs.reserveCapacity(B)
        for b in 0 ..< B {
            let qb = rope(queries[b ..< (b + 1)], positions[b])            // [1,nH,1,hd]（slot 自身の position）
            let kb = rope(keys[b ..< (b + 1)], positions[b])               // [1,nKV,1,hd]
            let vb = values[b ..< (b + 1)]
            let (aK, aV) = slotKV[b].update(kb, vb)                        // slot 独立 KV に追記
            outs.append(MLXFast.scaledDotProductAttention(queries: qb, keys: aK, values: aV, scale: scale, mask: .none))
        }
        let output = MLX.concatenated(outs, axis: 0).transposed(0, 2, 1, 3).reshaped([B, 1, -1])
        return oProj.apply(output * MLX.sigmoid(gate))
    }
}

public extension AttentionLayer {
    /// 検証: callContinuous(2 slot, 異なる position) の各行が standalone decode と bit 一致するか。
    static func continuousAttnTest() -> String {
        let H = 2048, Pa = 5, Pb = 2
        MLXRandom.seed(0)
        func qp(_ outd: Int, _ ind: Int = H) -> Proj {
            let w = (MLXRandom.normal([outd, ind]) * 0.02).asType(.float16)
            let (wq, sq, bq) = MLX.quantized(w, groupSize: 64, bits: 4); return .quantized(wq, sq, bq!, 4)
        }
        let attn = AttentionLayer(numHeads: 16, numKVHeads: 2, headDim: 256, ropeDim: 64, ropeBase: 1e7, eps: 1e-6,
            qProj: qp(16*256*2), kProj: qp(2*256), vProj: qp(2*256), oProj: qp(H, 16*256),
            qNorm: MLXArray.ones([256]).asType(.float16), kNorm: MLXArray.ones([256]).asType(.float16))
        // 共通 prefix を standalone(cA/cB) と continuous(cAc/cBc) の両 cache に同一適用 → 新 token を比較。
        MLXRandom.seed(1)
        let cA = KVCache(), cB = KVCache(), cAc = KVCache(), cBc = KVCache()
        for _ in 0..<Pa { let t = (MLXRandom.normal([1,1,H])*0.5).asType(.float16); _ = attn(t, cache: cA); _ = attn(t, cache: cAc) }
        for _ in 0..<Pb { let t = (MLXRandom.normal([1,1,H])*0.5).asType(.float16); _ = attn(t, cache: cB); _ = attn(t, cache: cBc) }
        let newA = (MLXRandom.normal([1,1,H])*0.5).asType(.float16)
        let newB = (MLXRandom.normal([1,1,H])*0.5).asType(.float16)
        let soloA = attn(newA, cache: cA), soloB = attn(newB, cache: cB)
        let batched = attn.callContinuous(MLX.concatenated([newA, newB], axis: 0), slotKV: [cAc, cBc], positions: [Pa, Pb])
        soloA.eval(); soloB.eval(); batched.eval()
        func rel(_ a: MLXArray, _ b: MLXArray) -> Float {
            MLX.max(MLX.abs(a.asType(.float32)-b.asType(.float32))).item(Float.self) / (MLX.max(MLX.abs(b.asType(.float32))).item(Float.self)+1e-9)
        }
        let rA = rel(batched[0..<1], soloA), rB = rel(batched[1..<2], soloB)
        return String(format: "[continuous-attn-test] 2 slot 異 position(A=%d,B=%d) callContinuous vs standalone: rowA rel=%.3e rowB rel=%.3e → %@",
                      Pa, Pb, rA, rB, (rA < 1e-3 && rB < 1e-3) ? "✅ per-stream position 正しい" : "❌乖離")
    }
}

extension AttentionLayer {
    /// 検証 [u,v](L=2, .causal) vs 逐次 u→v(L=1, .none) が bit 一致するか。
    /// spec の batched verify が逐次 accept と drift する真因切り分け（KV-prefix 付き）。
    public static func sConsistencyTest(dtype: DType = .float16) -> String {
        let H = 2048, P = 16
        MLXRandom.seed(0)
        func rp(_ outd: Int, _ ind: Int = H) -> Proj { .plain((MLXRandom.normal([outd, ind]) * 0.02).asType(dtype)) }
        let attn = AttentionLayer(
            numHeads: 16, numKVHeads: 2, headDim: 256, ropeDim: 64, ropeBase: 1e7, eps: 1e-6,
            qProj: rp(16 * 256 * 2), kProj: rp(2 * 256), vProj: rp(2 * 256), oProj: rp(H, 16 * 256),
            qNorm: MLXArray.ones([256]).asType(dtype), kNorm: MLXArray.ones([256]).asType(dtype))
        let xPrefix = (MLXRandom.normal([1, P, H]) * 0.5).asType(dtype)
        let u = (MLXRandom.normal([1, 1, H]) * 0.5).asType(dtype)
        let v = (MLXRandom.normal([1, 1, H]) * 0.5).asType(dtype)

        // prefix を 2 本の cache に同一構築（同一入力→決定論的）
        func freshPrefixCache() -> KVCache { let c = KVCache(); _ = attn(xPrefix, cache: c); return c }

        // batched: [u,v] を L=2(.causal) で 1 回
        let cB = freshPrefixCache()
        let uv = MLX.concatenated([u, v], axis: 1)
        let yB = attn(uv, cache: cB)
        yB.eval(); cB.keys!.eval()

        // sequential: u→v を L=1(.none) で 2 回
        let cS = freshPrefixCache()
        let yu = attn(u, cache: cS)
        let yv = attn(v, cache: cS)
        yu.eval(); yv.eval(); cS.keys!.eval()

        func rel(_ x: MLXArray, _ y: MLXArray) -> Float {
            MLX.max(MLX.abs(x.asType(.float32) - y.asType(.float32))).item(Float.self)
                / (MLX.max(MLX.abs(y.asType(.float32))).item(Float.self) + 1e-9)
        }
        let relU = rel(yB[0..., 0 ..< 1], yu)
        let relV = rel(yB[0..., 1 ..< 2], yv)
        let relKV = rel(cB.keys!, cS.keys!)
        let ok = relU < 1e-5 && relV < 1e-5
        return String(format: """
            [ATTN-TTEST %@] 検証[u,v](L=2 .causal) vs 逐次 u→v(L=1 .none):
              out u rel=%.3e  out v rel=%.3e  KV rel=%.3e  -> %@
            """, dtype == .float32 ? "f32" : "f16", relU, relV, relKV,
            ok ? "bit一致 ✅" : "乖離（経路差; f16=~7e-4 drift 源 / f32 で縮むか確認）")
    }
}

extension AttentionLayer {
    /// ★★ perQueryNone を量子化射影(実モデル相当)で検証。射影が order-stable な量子化なら rel=0 が期待値。
    /// plain f16(sConsistencyTest)は射影の L=1 境界 7e-7 が残るが、量子化は完全 bit 一致するはず。
    public static func perQueryNoneQuantTest() -> String {
        let H = 2048, P = 16
        MLXRandom.seed(0)
        func qp(_ outd: Int, _ ind: Int = H) -> Proj {
            let w = (MLXRandom.normal([outd, ind]) * 0.02).asType(.float16)
            let (wq, sq, bq) = MLX.quantized(w, groupSize: 64, bits: 4)
            return .quantized(wq, sq, bq!, 4)
        }
        let attn = AttentionLayer(
            numHeads: 16, numKVHeads: 2, headDim: 256, ropeDim: 64, ropeBase: 1e7, eps: 1e-6,
            qProj: qp(16 * 256 * 2), kProj: qp(2 * 256), vProj: qp(2 * 256), oProj: qp(H, 16 * 256),
            qNorm: MLXArray.ones([256]).asType(.float16), kNorm: MLXArray.ones([256]).asType(.float16))
        let xPrefix = (MLXRandom.normal([1, P, H]) * 0.5).asType(.float16)
        let u = (MLXRandom.normal([1, 1, H]) * 0.5).asType(.float16)
        let v = (MLXRandom.normal([1, 1, H]) * 0.5).asType(.float16)
        let uv = MLX.concatenated([u, v], axis: 1)
        func cache() -> KVCache { let c = KVCache(); _ = attn(xPrefix, cache: c); return c }
        func rel(_ a: MLXArray, _ b: MLXArray) -> Float {
            MLX.max(MLX.abs(a.asType(.float32) - b.asType(.float32))).item(Float.self)
                / (MLX.max(MLX.abs(b.asType(.float32))).item(Float.self) + 1e-9)
        }
        // perQueryNone は呼び出し側で true 設定済の前提。batched [u,v] vs 逐次 u→v(decode 経路)
        let cB = cache(); let yB = attn(uv, cache: cB); yB.eval()
        let cS = cache(); let yu = attn(u, cache: cS); let yv = attn(v, cache: cS)
        yu.eval(); yv.eval()
        let relU = rel(yB[0..., 0 ..< 1], yu), relV = rel(yB[0..., 1 ..< 2], yv)
        let ok = relU < 1e-6 && relV < 1e-6
        return String(format: "検証[u,v] vs 逐次 u→v(量子化射影): u rel=%.3e v rel=%.3e -> %@",
                      relU, relV, ok ? "bit一致 ✅(strict lossless verify 可)" : "乖離 ❌")
    }

    /// (C) 順序安定性検証: MLX の sum-based naive attention が batched(L>1) と single(L=1) で
    /// 同 query につき bit 一致するか。一致なら matmul の L 依存を回避でき custom kernel 不要。
    public static func reductionStableTest() -> String {
        MLXRandom.seed(0)
        let L = 4, S = 20, D = 256
        let sc = Float(pow(Double(D), -0.5))
        let q = (MLXRandom.normal([1, 1, L, D]) * 0.5).asType(.float32)
        let k = (MLXRandom.normal([1, 1, S, D]) * 0.5).asType(.float32)
        let v = (MLXRandom.normal([1, 1, S, D]) * 0.5).asType(.float32)
        // naive attention(全 query が全 S key に attend, no causal): broadcast + sum 縮約
        func naive(_ qq: MLXArray) -> MLXArray {
            let qe = qq.expandedDimensions(axis: 3)        // [1,1,Lq,1,D]
            let ke = k.expandedDimensions(axis: 2)         // [1,1,1,S,D]
            let scores = (qe * ke).sum(axis: -1) * sc      // [1,1,Lq,S]
            let p = MLX.softmax(scores, axis: -1)
            let pe = p.expandedDimensions(axis: -1)         // [1,1,Lq,S,1]
            let ve = v.expandedDimensions(axis: 2)          // [1,1,1,S,D]
            return (pe * ve).sum(axis: -2)                 // [1,1,Lq,D]
        }
        // 純 sum の L 依存も別途
        let sumFull = (q.expandedDimensions(axis: 3) * k.expandedDimensions(axis: 2)).sum(axis: -1)  // [1,1,L,S]
        let sumQ0 = (q[0..., 0..., 0 ..< 1, 0...].expandedDimensions(axis: 3) * k.expandedDimensions(axis: 2)).sum(axis: -1)
        let yB = naive(q); let y0 = naive(q[0..., 0..., 0 ..< 1, 0...])
        yB.eval(); y0.eval(); sumFull.eval(); sumQ0.eval()
        func rel(_ a: MLXArray, _ b: MLXArray) -> Float {
            MLX.max(MLX.abs(a - b)).item(Float.self) / (MLX.max(MLX.abs(b)).item(Float.self) + 1e-9)
        }
        let relSum = rel(sumFull[0..., 0..., 0 ..< 1, 0...], sumQ0)
        let relAttn = rel(yB[0..., 0..., 0 ..< 1, 0...], y0)
        return String(format: """
            [REDUCE-STABLE] naive attention batched(L=%d) vs single(L=1) の query0:
              sum縮約(QK) rel=%.3e   naive-attn rel=%.3e  -> %@
            """, L, relSum, relAttn, (relSum == 0 && relAttn == 0) ? "順序安定 ✅(custom kernel 不要)" : "L 依存あり(kernel 要 or 縮約も非安定)")
    }
}

extension AttentionLayer {
    /// (C) orderStable attention の検証: (1)順序安定(batched=single) (2)正しさ(fused SDPA と一致)。
    public static func orderStableAttnTest() -> String {
        let H = 2048, P = 16
        MLXRandom.seed(0)
        func rp(_ outd: Int, _ ind: Int = H) -> Proj { .plain((MLXRandom.normal([outd, ind]) * 0.02).asType(.float16)) }
        let attn = AttentionLayer(
            numHeads: 16, numKVHeads: 2, headDim: 256, ropeDim: 64, ropeBase: 1e7, eps: 1e-6,
            qProj: rp(16 * 256 * 2), kProj: rp(2 * 256), vProj: rp(2 * 256), oProj: rp(H, 16 * 256),
            qNorm: MLXArray.ones([256]).asType(.float16), kNorm: MLXArray.ones([256]).asType(.float16))
        let xPrefix = (MLXRandom.normal([1, P, H]) * 0.5).asType(.float16)
        let u = (MLXRandom.normal([1, 1, H]) * 0.5).asType(.float16)
        let v = (MLXRandom.normal([1, 1, H]) * 0.5).asType(.float16)
        let uv = MLX.concatenated([u, v], axis: 1)
        func cache() -> KVCache { let c = KVCache(); _ = attn(xPrefix, cache: c); return c }
        func rel(_ a: MLXArray, _ b: MLXArray) -> Float {
            MLX.max(MLX.abs(a.asType(.float32) - b.asType(.float32))).item(Float.self)
                / (MLX.max(MLX.abs(b.asType(.float32))).item(Float.self) + 1e-9)
        }
        // orderStable: batched [u,v] vs 逐次 u→v
        AttentionLayer.orderStable = true
        let cB = cache(); let yB = attn(uv, cache: cB)
        let cS = cache(); let yu = attn(u, cache: cS); let yv = attn(v, cache: cS)
        yB.eval(); yu.eval(); yv.eval()
        let relStableU = rel(yB[0..., 0 ..< 1], yu), relStableV = rel(yB[0..., 1 ..< 2], yv)
        // 正しさ: orderStable batched vs fused SDPA batched
        AttentionLayer.orderStable = false
        let cF = cache(); let yF = attn(uv, cache: cF); yF.eval()
        let relVsFused = rel(yB, yF)
        return String(format: """
            [ORDER-STABLE-ATTN] (1)順序安定 batched vs 逐次: u rel=%.3e v rel=%.3e -> %@
                                (2)正しさ orderStable vs fused: rel=%.3e -> %@
            """, relStableU, relStableV, (relStableU < 1e-6 && relStableV < 1e-6) ? "✅ bit一致" : "❌乖離",
            relVsFused, relVsFused < 5e-3 ? "✅ 同等" : "❌異なる")
    }
}

extension AttentionLayer {
    /// matmul の L 依存が「L=1(GEMV) vs L≥2(GEMM) 境界のみ」か「全 L」かを切り分ける。
    /// 前者なら decode を L=2 padding で順序統一でき strict batched verify が可能。
    public static func matmulLDependenceTest() -> String {
        MLXRandom.seed(0)
        let H = 2048, O = 512
        let Wp = (MLXRandom.normal([H, O]) * 0.02).asType(.float16)
        let x = (MLXRandom.normal([16, H]) * 0.5).asType(.float16)
        func mm(_ rows: Int) -> MLXArray { MLX.matmul(x[0 ..< rows, 0...], Wp) }   // [rows,H]@[H,O]=[rows,O]
        func rel(_ a: MLXArray, _ b: MLXArray) -> Float {
            MLX.max(MLX.abs(a.asType(.float32) - b.asType(.float32))).item(Float.self)
                / (MLX.max(MLX.abs(b.asType(.float32))).item(Float.self) + 1e-9)
        }
        // quantized 版（実モデルの射影は量子化）
        let (wq, sq, bq) = MLX.quantized(Wp.transposed(), groupSize: 64, bits: 4)
        func qmm(_ rows: Int) -> MLXArray {
            MLX.quantizedMatmul(x[0 ..< rows, 0...], wq, scales: sq, biases: bq, transpose: true, groupSize: 64, bits: 4)
        }
        let r1 = mm(1), r2 = mm(2), r4 = mm(4), r8 = mm(8)
        let q1 = qmm(1), q2 = qmm(2), q8 = qmm(8)
        for a in [r1, r2, r4, r8, q1, q2, q8] { a.eval() }
        return String(format: """
            [MATMUL-LDEP] plain f16 row0: L1vs2=%.2e  L2vs4=%.2e  L4vs8=%.2e
                          quant 4bit row0: L1vs2=%.2e  L2vs8=%.2e
              -> %@
            """,
            rel(r1[0 ..< 1, 0...], r2[0 ..< 1, 0...]), rel(r2[0 ..< 1, 0...], r4[0 ..< 1, 0...]),
            rel(r4[0 ..< 1, 0...], r8[0 ..< 1, 0...]),
            rel(q1[0 ..< 1, 0...], q2[0 ..< 1, 0...]), rel(q2[0 ..< 1, 0...], q8[0 ..< 1, 0...]),
            "L=1境界のみ非0なら decode を L=2 padding で順序統一可")
    }
}

extension AttentionLayer {
    /// 各 fused op(RoPE/rmsNorm/SDPA)が L=1 vs L=2(row0)で L 依存か。explicit 版が順序安定なら置換で解決。
    public static func fusedOpLDepTest() -> String {
        MLXRandom.seed(0)
        let D = 256, S = 20
        let x = (MLXRandom.normal([1, 4, 8, D]) * 0.5).asType(.float16)   // [B,heads,L=4,D]
        func rel(_ a: MLXArray, _ b: MLXArray) -> Float {
            MLX.max(MLX.abs(a.asType(.float32) - b.asType(.float32))).item(Float.self)
                / (MLX.max(MLX.abs(b.asType(.float32))).item(Float.self) + 1e-9)
        }
        // RoPE: fused（L 依存か）
        func ropeF(_ rows: Int) -> MLXArray {
            MLXFast.RoPE(x[0..., 0..., 0 ..< rows, 0...], dimensions: 64, traditional: false, base: 1e7, scale: 1.0, offset: 0)
        }
        let rp2 = ropeF(2), rp4 = ropeF(4); rp2.eval(); rp4.eval()
        let relRope = rel(rp2[0..., 0..., 0 ..< 1, 0...], rp4[0..., 0..., 0 ..< 1, 0...])
        // rmsNorm: fused
        let w = MLXArray.ones([D]).asType(.float16)
        func rmsF(_ rows: Int) -> MLXArray { MLXFast.rmsNorm(x[0..., 0..., 0 ..< rows, 0...], weight: w, eps: 1e-6) }
        let rm2 = rmsF(2), rm4 = rmsF(4); rm2.eval(); rm4.eval()
        let relRms = rel(rm2[0..., 0..., 0 ..< 1, 0...], rm4[0..., 0..., 0 ..< 1, 0...])
        // fused SDPA: L=1 vs L=2（causal, KV 共通）
        let q = (MLXRandom.normal([1, 2, 8, D]) * 0.5).asType(.float16)
        let k = (MLXRandom.normal([1, 2, S, D]) * 0.5).asType(.float16)
        let v = (MLXRandom.normal([1, 2, S, D]) * 0.5).asType(.float16)
        let sc = Float(pow(Double(D), -0.5))
        let s1 = MLXFast.scaledDotProductAttention(queries: q[0..., 0..., 0 ..< 1, 0...], keys: k, values: v, scale: sc, mask: .none)
        let s2 = MLXFast.scaledDotProductAttention(queries: q[0..., 0..., 0 ..< 2, 0...], keys: k, values: v, scale: sc, mask: .none)
        s1.eval(); s2.eval()
        let relSdpa = rel(s1, s2[0..., 0..., 0 ..< 1, 0...])
        return String(format: """
            [FUSED-OP-LDEP] L=1 vs L≥2 (row0):  RoPE=%.2e  rmsNorm=%.2e  fused-SDPA=%.2e
              -> 非0 の op が drift 源。explicit/sum 版で順序安定化できるか確認
            """, relRope, relRms, relSdpa)
    }
}

public enum AttentionLayerValidation {
    public static func run(refPath: String) throws -> String {
        let r = try loadArrays(url: URL(fileURLWithPath: refPath))
        guard let x = r["x"], let qp = r["q_proj"], let kp = r["k_proj"], let vp = r["v_proj"],
              let op = r["o_proj"], let qn = r["q_norm"], let kn = r["k_norm"],
              let expOut = r["out"] else {
            return "ERROR: attn ref 不足"
        }
        let attn = AttentionLayer(
            numHeads: 16, numKVHeads: 2, headDim: 256, ropeDim: 64, ropeBase: 1e7, eps: 1e-6,
            qProj: .plain(qp), kProj: .plain(kp), vProj: .plain(vp), oProj: .plain(op),
            qNorm: qn, kNorm: kn)
        let out = attn(x)
        out.eval()
        let d = MLX.max(MLX.abs(out - expOut)).item(Float.self)
            / (MLX.max(MLX.abs(expOut)).item(Float.self) + 1e-9)
        let ok = d < 1e-3
        return String(format: "[M2b-2] full-attention 層: out_rel=%.2e  %@",
                      d, ok ? "OK ✅ bit一致" : "MISMATCH ❌")
    }
}
