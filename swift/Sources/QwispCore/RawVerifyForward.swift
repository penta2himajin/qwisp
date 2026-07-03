import Foundation
import MLX
import Metal
import MLXNN

/// D1 U1: M-row verify 合成層。RawMetalForward の M-row kernel 群(全て rows≡M=1ループ bit一致検証済み)を
/// 実 decoder 層の op 列どおりに合成する。参照実装は attnLayerRaw / gdnLayerRaw / runMoeBlockTest の
/// M=1 chain(いずれも MLX と bit-exact 照合済み)。
/// 設計規律: 全 matmul/attention/conv/recurrence は tested Rows wrapper のみ。MLX glue は
/// elementwise/位置毎/データ移動(reshape/transpose/concat/slice)に限定 — cross-row reduction 禁止。
public enum RawVerifyForward {

    /// attention 層重み(qwen3.5: fused q+gate proj, qk-norm, sigmoid-gated output)。
    public struct AttnLayerW {
        let qWq: MLXArray, qSc: MLXArray, qBi: MLXArray     // [numHeads*2*headDim, H] 4bit
        let kWq: MLXArray, kSc: MLXArray, kBi: MLXArray     // [numKV*headDim, H]
        let vWq: MLXArray, vSc: MLXArray, vBi: MLXArray
        let oWq: MLXArray, oSc: MLXArray, oBi: MLXArray     // [H, numHeads*headDim]
        let qNorm: MLXArray, kNorm: MLXArray                // [headDim]
        public init(qWq: MLXArray, qSc: MLXArray, qBi: MLXArray,
                    kWq: MLXArray, kSc: MLXArray, kBi: MLXArray,
                    vWq: MLXArray, vSc: MLXArray, vBi: MLXArray,
                    oWq: MLXArray, oSc: MLXArray, oBi: MLXArray,
                    qNorm: MLXArray, kNorm: MLXArray) {
            self.qWq = qWq; self.qSc = qSc; self.qBi = qBi
            self.kWq = kWq; self.kSc = kSc; self.kBi = kBi
            self.vWq = vWq; self.vSc = vSc; self.vBi = vBi
            self.oWq = oWq; self.oSc = oSc; self.oBi = oBi
            self.qNorm = qNorm; self.kNorm = kNorm
        }
    }

    /// attention 層 × M 行(f16 経路)。attnLayerRaw(M=1, promoteF32=false)の op 列を Rows wrapper で
    /// M 行化したもの。row m は kCache/vCache の先頭 baseLen+m+1 key(自身含む)を見る(causal)。
    /// kCache/vCache は [numKV, L, headDim](post-RoPE k / raw v)で、呼び出し後 L+M に成長して返る。
    /// 戻り値は o_proj 出力 [M, H](residual は呼び出し側)。
    public static func attnLayerRows(_ x: MLXArray, _ w: AttnLayerW,
                                     kCache: inout MLXArray, vCache: inout MLXArray,
                                     M: Int,
                                     numHeads: Int = 16, numKV: Int = 2, headDim: Int = 256,
                                     ropeDim: Int = 64, ropeBase: Float = 1e7, eps: Float = 1e-6) -> MLXArray? {
        let H = x.dim(-1)
        let baseLen = kCache.dim(1)
        let scale = Float(pow(Double(headDim), -0.5))
        let qd2 = 2 * headDim
        // q(+gate)/k/v projection — per-row order-stable qmmRows
        guard let qOut = RawMetalForward.qmmRows(x, w.qWq, scales: w.qSc, biases: w.qBi, M: M, K: H, N: numHeads * qd2),
              let kOut = RawMetalForward.qmmRows(x, w.kWq, scales: w.kSc, biases: w.kBi, M: M, K: H, N: numKV * headDim),
              let vOut = RawMetalForward.qmmRows(x, w.vWq, scales: w.vSc, biases: w.vBi, M: M, K: H, N: numKV * headDim)
        else { return nil }
        let qOutR = qOut.reshaped([M * numHeads, qd2])
        let queries = qOutR[0..., 0 ..< headDim]                                   // [M*numHeads, headDim]
        // qk-norm(f16 weight, per-row = per-head)
        guard let qN = RawMetalForward.rmsNormRows(queries, w.qNorm.asType(.float16), M: M * numHeads, eps: eps, D: headDim),
              let kN = RawMetalForward.rmsNormRows(kOut.reshaped([M * numKV, headDim]), w.kNorm.asType(.float16),
                                                   M: M * numKV, eps: eps, D: headDim)
        else { return nil }
        // RoPE: 行 m の位置 = baseLen + m(q は numHeads 群、k は numKV 群)
        guard let qRot = RawMetalForward.ropeRows(qN, headDim: headDim, ropeDim: ropeDim, base: ropeBase,
                                                  startOffset: baseLen, M: M, numHeads: numHeads),
              let kRot = RawMetalForward.ropeRows(kN, headDim: headDim, ropeDim: ropeDim, base: ropeBase,
                                                  startOffset: baseLen, M: M, numHeads: numKV)
        else { return nil }
        // cache append: [M*numKV, headDim] → [M, numKV, headDim] → [numKV, M, headDim] → concat(L 軸)
        let kNew = kRot.reshaped([M, numKV, headDim]).transposed(1, 0, 2)
        let vNew = vOut.reshaped([M, numKV, headDim]).transposed(1, 0, 2)
        // baseLen==0 guard: MLX.concatenated with an empty 0-dim array may error on some versions.
        if baseLen == 0 {
            kCache = kNew; kCache.eval()
            vCache = vNew; vCache.eval()
        } else {
            kCache = MLX.concatenated([kCache, kNew], axis: 1); kCache.eval()
            vCache = MLX.concatenated([vCache, vNew], axis: 1); vCache.eval()
        }
        // SDPA: 行 m は先頭 baseLen+m+1 key(自身含む)を見る → sdpaRows の baseN = baseLen+1
        guard let attnOut = RawMetalForward.sdpaRows(qRot, kCache, vCache,
                                                     H: numHeads, KV: numKV, D: headDim,
                                                     baseLen: baseLen + 1, M: M, scale: scale)
        else { return nil }
        // sigmoid-gated output — raw kernel(sigmoid_mul, gate は qOut から strided 読み)= fused と同一数値系
        guard let gated0 = RawFusedVerify.sigmoidMulRaw(attnOut, qOut, headDim: headDim, qd2: qd2,
                                                        total: M * numHeads * headDim) else { return nil }
        let gated = gated0.reshaped([M, numHeads * headDim])
        return RawMetalForward.qmmRows(gated, w.oWq, scales: w.oSc, biases: w.oBi, M: M, K: numHeads * headDim, N: H)
    }

    /// GDN 層重み(qwen3.5: in_proj 4分割, grouped conv, gated-delta recurrence, RMSNormGated)。
    public struct GDNLayerW {
        let qkvWq: MLXArray, qkvSc: MLXArray, qkvBi: MLXArray   // [convDim, H]
        let zWq: MLXArray, zSc: MLXArray, zBi: MLXArray         // [valueDim, H]
        let bWq: MLXArray, bSc: MLXArray, bBi: MLXArray         // [numVHeads, H]
        let aWq: MLXArray, aSc: MLXArray, aBi: MLXArray         // [numVHeads, H]
        let outWq: MLXArray, outSc: MLXArray, outBi: MLXArray   // [H, valueDim]
        let conv1dW: MLXArray                                    // [convDim, K]
        let normWeight: MLXArray                                 // [headVDim] (f16 or f32)
        let aLog: MLXArray, dtBias: MLXArray                     // [numVHeads] f32
        public init(qkvWq: MLXArray, qkvSc: MLXArray, qkvBi: MLXArray,
                    zWq: MLXArray, zSc: MLXArray, zBi: MLXArray,
                    bWq: MLXArray, bSc: MLXArray, bBi: MLXArray,
                    aWq: MLXArray, aSc: MLXArray, aBi: MLXArray,
                    outWq: MLXArray, outSc: MLXArray, outBi: MLXArray,
                    conv1dW: MLXArray, normWeight: MLXArray, aLog: MLXArray, dtBias: MLXArray) {
            self.qkvWq = qkvWq; self.qkvSc = qkvSc; self.qkvBi = qkvBi
            self.zWq = zWq; self.zSc = zSc; self.zBi = zBi
            self.bWq = bWq; self.bSc = bSc; self.bBi = bBi
            self.aWq = aWq; self.aSc = aSc; self.aBi = aBi
            self.outWq = outWq; self.outSc = outSc; self.outBi = outBi
            self.conv1dW = conv1dW; self.normWeight = normWeight
            self.aLog = aLog; self.dtBias = dtBias
        }
    }

    /// GDN 層 × M 行。gdnLayerRaw(M=1)の op 列を Rows wrapper で M 行化。
    /// convState [K-1, convDim] f16 / recState [1, Hv, Dv, Dk] f32 を inout で更新(逐次 threading と bit 一致)。
    public static func gdnLayerRows(_ x: MLXArray, _ w: GDNLayerW,
                                    convState: inout MLXArray, recState: inout MLXArray,
                                    M: Int,
                                    numKHeads: Int = 16, numVHeads: Int = 32,
                                    headKDim: Int = 128, headVDim: Int = 128,
                                    convKernel: Int = 4, eps: Float = 1e-6) -> MLXArray? {
        let H = x.dim(-1)
        let keyDim = headKDim * numKHeads
        let valueDim = headVDim * numVHeads
        let convDim = keyDim * 2 + valueDim
        // ① in_proj ×4(per-row order-stable)
        guard let qkv = RawMetalForward.qmmRows(x, w.qkvWq, scales: w.qkvSc, biases: w.qkvBi, M: M, K: H, N: convDim),
              let z   = RawMetalForward.qmmRows(x, w.zWq,   scales: w.zSc,   biases: w.zBi,   M: M, K: H, N: valueDim),
              let bP  = RawMetalForward.qmmRows(x, w.bWq,   scales: w.bSc,   biases: w.bBi,   M: M, K: H, N: numVHeads),
              let aP  = RawMetalForward.qmmRows(x, w.aWq,   scales: w.aSc,   biases: w.aBi,   M: M, K: H, N: numVHeads)
        else { return nil }
        // ② conv 窓構築(データ移動のみ): convInput = [convState; qkv] → 行 m の窓 = convInput[m .. m+K-1]
        let convInput = MLX.concatenated([convState, qkv.asType(.float16)], axis: 0)   // [K-1+M, convDim]
        var windowParts: [MLXArray] = []
        for m in 0 ..< M { windowParts.append(convInput[m ..< m + convKernel]) }       // 各 [K, convDim]
        let windows = MLX.stacked(windowParts, axis: 0)                                 // [M, K, convDim]
        MLX.eval([windows])
        guard let convOut = RawMetalForward.conv1dSiluRows(windows, w.conv1dW, M: M, K: convKernel, C: convDim)
        else { return nil }                                                             // [M, convDim]
        convState = convInput[M ..< M + convKernel - 1]; convState.eval()               // 窓の残り = 新 conv state
        // ③ split → q,k,v(行毎)
        let q1 = convOut[0..., 0 ..< keyDim].reshaped([M * numKHeads, headKDim])
        let k1 = convOut[0..., keyDim ..< 2 * keyDim].reshaped([M * numKHeads, headKDim])
        let v1 = convOut[0..., (2 * keyDim)...].reshaped([1, M, numVHeads, headVDim])
        // ④ qk-norm(no-weight)+ scalar scale(elementwise)
        let invScale = Float(pow(Double(headKDim), -0.5))
        guard let qn0 = RawMetalForward.rmsNormRows(q1, nil, M: M * numKHeads, eps: eps, D: headKDim),
              let kn0 = RawMetalForward.rmsNormRows(k1, nil, M: M * numKHeads, eps: eps, D: headKDim) else { return nil }
        let qN = ((invScale * invScale) * qn0).reshaped([1, M, numKHeads, headKDim])
        let kN = (invScale * kn0).reshaped([1, M, numKHeads, headKDim])
        // ⑤ recurrence: in-kernel T=M 逐次(chained T=1 と bit 一致は kernel テスト済み)。
        // g/β は raw kernel(compute_g_beta_rows)— fused と数値系共有
        guard let (g, beta) = RawFusedVerify.computeGBetaRowsRaw(aP, bP, w.aLog, w.dtBias, M: M, Hv: numVHeads)
        else { return nil }
        guard let (coreOut, stOut) = RawMetalForward.gatedDeltaStepRows(qN, kN, v1, g: g, beta: beta, state: recState,
                                                                        M: M, B: 1, Hk: numKHeads, Dk: headKDim,
                                                                        Hv: numVHeads, Dv: headVDim) else { return nil }
        recState = stOut; recState.eval()
        // ⑥ RMSNormGated(per-row)
        let promoteRMS = (w.normWeight.dtype == .float32)
        guard let normed = RawMetalForward.rmsNormRows(coreOut.reshaped([M * numVHeads, headVDim]), w.normWeight,
                                                       M: M * numVHeads, eps: eps, D: headVDim, promoteF32: promoteRMS)
        else { return nil }
        // silu(z)·normed は raw kernel(gate/gate16)— fused と数値系共有
        guard let gated = RawFusedVerify.gateRaw(z, normed, promote: promoteRMS, total: M * valueDim)
        else { return nil }
        let outV = gated.reshaped([M, valueDim])
        // ⑦ out_proj
        return RawMetalForward.qmmRows(outV, w.outWq, scales: w.outSc, biases: w.outBi, M: M, K: valueDim, N: H)
    }

    /// MoE block 重み(routed experts + shared expert + shared gate)。
    public struct MoEBlockW {
        let gateWq: MLXArray, gateSc: MLXArray, gateBi: MLXArray       // [E, H] 8bit
        let swGWq: MLXArray, swGSc: MLXArray, swGBi: MLXArray          // [E, I, H] 4bit
        let swUWq: MLXArray, swUSc: MLXArray, swUBi: MLXArray
        let swDWq: MLXArray, swDSc: MLXArray, swDBi: MLXArray          // [E, H, I]
        let shGWq: MLXArray, shGSc: MLXArray, shGBi: MLXArray          // [I, H] 4bit
        let shUWq: MLXArray, shUSc: MLXArray, shUBi: MLXArray
        let shDWq: MLXArray, shDSc: MLXArray, shDBi: MLXArray          // [H, I]
        let sharedGateWq: MLXArray, sharedGateSc: MLXArray, sharedGateBi: MLXArray  // [Ngate(>=8 pad), H] 8bit — 列0のみ使用
        public init(gateWq: MLXArray, gateSc: MLXArray, gateBi: MLXArray,
                    swGWq: MLXArray, swGSc: MLXArray, swGBi: MLXArray,
                    swUWq: MLXArray, swUSc: MLXArray, swUBi: MLXArray,
                    swDWq: MLXArray, swDSc: MLXArray, swDBi: MLXArray,
                    shGWq: MLXArray, shGSc: MLXArray, shGBi: MLXArray,
                    shUWq: MLXArray, shUSc: MLXArray, shUBi: MLXArray,
                    shDWq: MLXArray, shDSc: MLXArray, shDBi: MLXArray,
                    sharedGateWq: MLXArray, sharedGateSc: MLXArray, sharedGateBi: MLXArray) {
            self.gateWq = gateWq; self.gateSc = gateSc; self.gateBi = gateBi
            self.swGWq = swGWq; self.swGSc = swGSc; self.swGBi = swGBi
            self.swUWq = swUWq; self.swUSc = swUSc; self.swUBi = swUBi
            self.swDWq = swDWq; self.swDSc = swDSc; self.swDBi = swDBi
            self.shGWq = shGWq; self.shGSc = shGSc; self.shGBi = shGBi
            self.shUWq = shUWq; self.shUSc = shUSc; self.shUBi = shUBi
            self.shDWq = shDWq; self.shDSc = shDSc; self.shDBi = shDBi
            self.sharedGateWq = sharedGateWq; self.sharedGateSc = sharedGateSc; self.sharedGateBi = sharedGateBi
        }
    }

    /// MoE block × M 行。routing は production と同一の MLX glue(softmax precise → argPartition top-K →
    /// renorm — 全て per-row op。M 形状安定性は rows≡loop テストが検証し、破れたら raw route_top8 へピボット)。
    /// expert 計算は gatherQmmRows(gate/up: 行共有 lhs, down: per-(row,ki) lhs)+ 明示順序 combine。
    /// metalRoute=false: MLX routing(softmax/argPartition, oracle 用)。
    /// metalRoute=true : routeTop8Rows(per-row Metal, sync 島なし = 融合の本命)。両者とも各行独立=M不変。
    public static func moeBlockRows(_ x: MLXArray, _ w: MoEBlockW,
                                    M: Int, E: Int, I: Int, Ktop: Int = 8, metalRoute: Bool = false) -> MLXArray? {
        let H = x.dim(-1)
        // routing(per-row)
        guard let gl = RawMetalForward.qmm8(x, w.gateWq, scales: w.gateSc, biases: w.gateBi, M: M, K: H, N: E)
        else { return nil }
        let inds: MLXArray, scores: MLXArray
        if metalRoute {
            guard let (mi, ms) = RawFusedVerify.routeTop8Rows(gl, M: M, N: E, K: Ktop) else { return nil }
            inds = mi; scores = ms                                              // [M,Ktop] 選択+renorm 済
        } else {
            let gates = MLX.softmax(gl, axis: -1, precise: true)
            let order = MLX.argPartition(gates, kth: E - Ktop, axis: -1)
            inds = order[0..., (E - Ktop)...]                                   // [M, Ktop]
            let sc = MLX.takeAlong(gates, inds, axis: -1)
            scores = sc / sc.sum(axis: -1, keepDims: true)                      // normTopk
        }
        let indsFlat = inds.reshaped([M * Ktop]).asType(.int32); indsFlat.eval()
        // routed experts: gate/up(行共有 x)→ swiglu → down(per-mk lhs)
        guard let g = RawMetalForward.gatherQmmRows(x, w.swGWq, scales: w.swGSc, biases: w.swGBi,
                                                    inds: indsFlat, M: M, Ktop: Ktop, K: H, N: I),
              let u = RawMetalForward.gatherQmmRows(x, w.swUWq, scales: w.swUSc, biases: w.swUBi,
                                                    inds: indsFlat, M: M, Ktop: Ktop, K: H, N: I)
        else { return nil }
        // swiglu は raw Metal kernel(stable sigmoid, f16)— fused 経路と同一数値系(engine 内自己整合が規範)
        guard let h = RawMetalForward.swigluRaw(g, u) else { return nil }       // [M*Ktop, I] elementwise
        guard let d = RawMetalForward.gatherQmmRows(h, w.swDWq, scales: w.swDSc, biases: w.swDBi,
                                                    inds: indsFlat, M: M, Ktop: Ktop, K: I, N: H,
                                                    lhsPerExpert: true)
        else { return nil }
        // combine: raw kernel(k 昇順 f16 逐次和)— fused の combine_rows と丸め列を共有(M 非依存)
        guard let y = RawFusedVerify.combineRowsRaw(d, scores, M: M, Ktop: Ktop, N: H) else { return nil }
        // shared expert + sigmoid(sharedGate) — sharedGate は pad 8 列の列 0 を使用
        guard let sg = RawMetalForward.qmmRows(x, w.shGWq, scales: w.shGSc, biases: w.shGBi, M: M, K: H, N: I),
              let su = RawMetalForward.qmmRows(x, w.shUWq, scales: w.shUSc, biases: w.shUBi, M: M, K: H, N: I)
        else { return nil }
        guard let shAct = RawMetalForward.swigluRaw(sg, su) else { return nil } // raw swiglu(fused と同一)
        guard let sharedY = RawMetalForward.qmmRows(shAct, w.shDWq, scales: w.shDSc, biases: w.shDBi, M: M, K: I, N: H),
              let sgl = RawMetalForward.qmm8(x, w.sharedGateWq, scales: w.sharedGateSc, biases: w.sharedGateBi,
                                             M: M, K: H, N: 8)
        else { return nil }
        // final: y + sigmoid(sgl[:,0])·sharedY — raw kernel(fused の final_combine_rows と同一)
        return RawFusedVerify.finalCombineRowsRaw(y, sharedY, sgl, M: M, N: H)
    }

    /// decoder 層 1 枚の仕様(mixer 種別 + 重み + MoE)。
    public struct LayerSpec {
        let isLinear: Bool
        let inputLN: MLXArray, postLN: MLXArray      // [H] f16
        let gdn: GDNLayerW?
        let attn: AttnLayerW?
        let moe: MoEBlockW
        let moeE: Int, moeI: Int, moeKtop: Int
        public init(isLinear: Bool, inputLN: MLXArray, postLN: MLXArray,
                    gdn: GDNLayerW?, attn: AttnLayerW?, moe: MoEBlockW,
                    moeE: Int, moeI: Int, moeKtop: Int = 8) {
            self.isLinear = isLinear; self.inputLN = inputLN; self.postLN = postLN
            self.gdn = gdn; self.attn = attn; self.moe = moe
            self.moeE = moeE; self.moeI = moeI; self.moeKtop = moeKtop
        }
    }

    /// 層別 cache(GDN: conv+rec / attn: kv)。
    public final class LayerCaches {
        public var kCache: MLXArray?, vCache: MLXArray?
        public var convState: MLXArray?, recState: MLXArray?
        public init(kCache: MLXArray? = nil, vCache: MLXArray? = nil,
                    convState: MLXArray? = nil, recState: MLXArray? = nil) {
            self.kCache = kCache; self.vCache = vCache
            self.convState = convState; self.recState = recState
        }
        public func copyState() -> LayerCaches {
            LayerCaches(kCache: kCache, vCache: vCache, convState: convState, recState: recState)
        }
    }

    /// U1d: decoder 層列 × M 行の verify forward 合成。
    /// StreamingDecoderLayer と同一 op 列: inputLN → mixer → resid → postLN → MoE → resid。
    /// 戻り値 [M, H](final norm/lm_head は呼び出し側)。caches は逐次 threading と bit 一致で更新。
    public static func verifyForwardRows(_ x: MLXArray, layers: [LayerSpec], caches: [LayerCaches],
                                         M: Int, eps: Float = 1e-6, metalRoute: Bool = false) -> MLXArray? {
        let H = x.dim(-1)
        var h = x
        for (i, L) in layers.enumerated() {
            guard let normed = RawMetalForward.rmsNormRows(h, L.inputLN, M: M, eps: eps, D: H) else { return nil }
            let r: MLXArray?
            if L.isLinear, let gw = L.gdn {
                var cs = caches[i].convState!, rs = caches[i].recState!
                r = gdnLayerRows(normed, gw, convState: &cs, recState: &rs, M: M, eps: eps)
                caches[i].convState = cs; caches[i].recState = rs
            } else if let aw = L.attn {
                var kc = caches[i].kCache!, vc = caches[i].vCache!
                r = attnLayerRows(normed, aw, kCache: &kc, vCache: &vc, M: M, eps: eps)
                caches[i].kCache = kc; caches[i].vCache = vc
            } else { return nil }
            guard let rr = r else { return nil }
            h = h + rr                                                            // residual(elementwise)
            guard let postNorm = RawMetalForward.rmsNormRows(h, L.postLN, M: M, eps: eps, D: H) else { return nil }
            guard let moeOut = moeBlockRows(postNorm, L.moe, M: M, E: L.moeE, I: L.moeI, Ktop: L.moeKtop, metalRoute: metalRoute)
            else { return nil }
            h = h + moeOut
        }
        h.eval()
        return h
    }
}
