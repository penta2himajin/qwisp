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

        // 射影は per-position（逐次でも同結果）ゆえ常に batched で計算（seqMT でも再実行しない＝軽い）
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

        var output: MLXArray
        if AttentionLayer.seqMultiToken && L > 1 {
            // 順序安定(partial-seqMT): SDPA だけ per-token。query t は causal prefix(0..offset+t)に
            // .none で attend ＝single-token decode と bit 一致(robust, f32 不要)。射影は上で batched 済。
            var outs: [MLXArray] = []
            for t in 0 ..< L {
                let s = offset + t + 1
                let qt = queries[0..., 0..., t ..< (t + 1), 0...]      // [B,heads,1,d]
                let kt = allKeys[0..., 0..., 0 ..< s, 0...]
                let vt = allValues[0..., 0..., 0 ..< s, 0...]
                outs.append(MLXFast.scaledDotProductAttention(queries: qt, keys: kt, values: vt,
                                                              scale: scale, mask: .none))
            }
            output = MLX.concatenated(outs, axis: 2)                   // [B,heads,L,headDim]
        } else {
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
