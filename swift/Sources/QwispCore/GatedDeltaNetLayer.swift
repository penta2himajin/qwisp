import Foundation
import MLX
import MLXFast
import MLXNN

/// Qwen3NextGatedDeltaNet.__call__ 全体の Swift 移植（M2b-1 層 wrapping）.
/// 核(GatedDelta.update)を包む in_proj + fix_ordering + grouped causal conv1d(+silu)
/// + q/k rms_norm スケール + RMSNormGated(z) + out_proj。cache=None/mask=None の単一チャンク。
public struct GatedDeltaNetLayer {
    // config 由来の形状
    let numKHeads: Int       // linear_num_key_heads = 16
    let numVHeads: Int       // linear_num_value_heads = 32
    let headKDim: Int        // 128
    let headVDim: Int        // 128
    let convKernel: Int      // 4
    let eps: Float

    var keyDim: Int { headKDim * numKHeads }       // 2048
    var valueDim: Int { headVDim * numVHeads }     // 4096
    var convDim: Int { keyDim * 2 + valueDim }     // 8192

    // 重み（Linear は [out, in]、conv1d は [convDim, K, 1]、norm は [headVDim]）
    let inProjQKVZ: MLXArray
    let inProjBA: MLXArray
    let conv1dW: MLXArray
    let normWeight: MLXArray
    let outProjW: MLXArray
    let aLog: MLXArray
    let dtBias: MLXArray

    public init(numKHeads: Int, numVHeads: Int, headKDim: Int, headVDim: Int,
                convKernel: Int, eps: Float,
                inProjQKVZ: MLXArray, inProjBA: MLXArray, conv1dW: MLXArray,
                normWeight: MLXArray, outProjW: MLXArray, aLog: MLXArray, dtBias: MLXArray) {
        self.numKHeads = numKHeads; self.numVHeads = numVHeads
        self.headKDim = headKDim; self.headVDim = headVDim
        self.convKernel = convKernel; self.eps = eps
        self.inProjQKVZ = inProjQKVZ; self.inProjBA = inProjBA
        self.conv1dW = conv1dW; self.normWeight = normWeight; self.outProjW = outProjW
        self.aLog = aLog; self.dtBias = dtBias
    }

    /// weight=None 相当の rms_norm（最終軸正規化、スケール無し）= ones を渡して fused kernel と一致.
    static func rmsNormNoWeight(_ x: MLXArray, eps: Float) -> MLXArray {
        let w = MLXArray.ones([x.dim(-1)], dtype: x.dtype)
        return MLXFast.rmsNorm(x, weight: w, eps: eps)
    }

    static func linear(_ x: MLXArray, _ w: MLXArray) -> MLXArray {
        // nn.Linear(bias=false): x @ w.T
        MLX.matmul(x, w.transposed())
    }

    /// fix_query_key_value_ordering 相当。返り値 q,k:[B,S,Hk,Dk] v,z:[B,S,Hv,Dv] b,a:[B,S,Hv]
    func fixOrdering(_ mixedQKVZ: MLXArray, _ mixedBA: MLXArray)
        -> (MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, MLXArray) {
        let B = mixedQKVZ.dim(0), S = mixedQKVZ.dim(1)
        let nk = numKHeads, dn = headKDim, nv = numVHeads, dv = headVDim
        let qkvz = mixedQKVZ.reshaped([B, S, nk, -1])     // [B,S,16, 768]
        let ba = mixedBA.reshaped([B, S, nk, -1])          // [B,S,16, 4]
        // split at [dn, 2dn, 2dn + nv/nk*dv] = [128, 256, 512] on last axis (size 768)
        let s1 = dn, s2 = 2 * dn, s3 = 2 * dn + (nv / nk) * dv
        let q = qkvz[0..., 0..., 0..., 0 ..< s1]
        let k = qkvz[0..., 0..., 0..., s1 ..< s2]
        let v = qkvz[0..., 0..., 0..., s2 ..< s3].reshaped([B, S, -1, dv])  // [B,S,32,128]
        let z = qkvz[0..., 0..., 0..., s3...].reshaped([B, S, -1, dv])      // [B,S,32,128]
        // ba split at nv/nk = 2 → b:[..0:2] a:[..2:4]
        let bb = ba[0..., 0..., 0..., 0 ..< (nv / nk)].reshaped([B, S, nv]) // [B,S,32]
        let aa = ba[0..., 0..., 0..., (nv / nk)...].reshaped([B, S, nv])    // [B,S,32]
        return (q, k, v, z, bb, aa)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let B = x.dim(0), S = x.dim(1)
        let (q0, k0, v0, z, b, a) = fixOrdering(
            GatedDeltaNetLayer.linear(x, inProjQKVZ),
            GatedDeltaNetLayer.linear(x, inProjBA))

        // mixed_qkv = concat(q,k,v に reshape) → [B,S,convDim]
        let mixedQKV = MLX.concatenated([
            q0.reshaped([B, S, -1]), k0.reshaped([B, S, -1]), v0.reshaped([B, S, -1]),
        ], axis: -1)
        // conv_state = zeros[B, K-1, convDim] を前置（因果 padding）
        let convState = MLXArray.zeros([B, convKernel - 1, convDim], dtype: x.dtype)
        let convInput = MLX.concatenated([convState, mixedQKV], axis: 1)  // [B, S+K-1, convDim]
        // depthwise causal conv1d(groups=convDim, padding=0) → 長さ S
        let convOut = silu(MLX.conv1d(convInput, conv1dW, stride: 1, padding: 0,
                                          dilation: 1, groups: convDim))   // [B,S,convDim]

        // split conv_out → q,k,v
        let q1 = convOut[0..., 0..., 0 ..< keyDim].reshaped([B, S, numKHeads, headKDim])
        let k1 = convOut[0..., 0..., keyDim ..< (2 * keyDim)].reshaped([B, S, numKHeads, headKDim])
        let v1 = convOut[0..., 0..., (2 * keyDim)...].reshaped([B, S, numVHeads, headVDim])

        let invScale = Float(pow(Double(headKDim), -0.5))
        let qN = (invScale * invScale) * GatedDeltaNetLayer.rmsNormNoWeight(q1, eps: 1e-6)
        let kN = invScale * GatedDeltaNetLayer.rmsNormNoWeight(k1, eps: 1e-6)

        let (coreOut, _) = GatedDelta.update(qN, kN, v1, a, b, aLog, dtBias)  // [B,S,Hv,Dv]

        // RMSNormGated(out, z): silu(z) * rms_norm(out, normWeight)
        let normed = MLXFast.rmsNorm(coreOut, weight: normWeight, eps: eps)
        let gated = silu(z.asType(.float32)) * normed.asType(.float32)
        let outV = gated.asType(coreOut.dtype).reshaped([B, S, -1])  // [B,S,valueDim]
        return GatedDeltaNetLayer.linear(outV, outProjW)
    }
}

public enum GatedDeltaNetLayerValidation {
    public static func run(refPath: String) throws -> String {
        let r = try loadArrays(url: URL(fileURLWithPath: refPath))
        guard let x = r["x"], let iq = r["in_proj_qkvz"], let ib = r["in_proj_ba"],
              let cw = r["conv1d"], let nw = r["norm_weight"], let ow = r["out_proj"],
              let aLog = r["A_log"], let dtBias = r["dt_bias"], let expOut = r["out"] else {
            return "ERROR: gdn-layer ref 不足"
        }
        let layer = GatedDeltaNetLayer(
            numKHeads: 16, numVHeads: 32, headKDim: 128, headVDim: 128,
            convKernel: 4, eps: 1e-6,
            inProjQKVZ: iq, inProjBA: ib, conv1dW: cw, normWeight: nw, outProjW: ow,
            aLog: aLog, dtBias: dtBias)
        let out = layer(x)
        out.eval()
        let d = MLX.max(MLX.abs(out - expOut)).item(Float.self)
            / (MLX.max(MLX.abs(expOut)).item(Float.self) + 1e-9)
        let ok = d < 1e-3
        return String(format: "[M2b-1] GatedDeltaNet層 wrapping: out_rel=%.2e  %@",
                      d, ok ? "OK ✅ bit一致" : "MISMATCH ❌")
    }
}
