import Foundation
import MLX
import Metal

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
        let gate = qOutR[0..., headDim...].reshaped([M, numHeads * headDim])       // [M, numHeads*headDim]
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
        kCache = MLX.concatenated([kCache, kNew], axis: 1); kCache.eval()
        vCache = MLX.concatenated([vCache, vNew], axis: 1); vCache.eval()
        // SDPA: 行 m は先頭 baseLen+m+1 key(自身含む)を見る → sdpaRows の baseN = baseLen+1
        guard let attnOut = RawMetalForward.sdpaRows(qRot, kCache, vCache,
                                                     H: numHeads, KV: numKV, D: headDim,
                                                     baseLen: baseLen + 1, M: M, scale: scale)
        else { return nil }
        // sigmoid-gated output(f16 elementwise, per-row)→ o_proj
        let outR = attnOut.reshaped([M, numHeads * headDim]).asType(.float16)
        let gated = outR * MLX.sigmoid(gate)
        return RawMetalForward.qmmRows(gated, w.oWq, scales: w.oSc, biases: w.oBi, M: M, K: numHeads * headDim, N: H)
    }
}
