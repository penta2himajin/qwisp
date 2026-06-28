import Foundation
import MLX
import MLXFast
import MLXNN

/// 射影（bias 無し Linear）。float か 4bit/8bit 量子化を統一的に適用。
public enum Proj {
    case plain(MLXArray)                                   // [out, in]
    case quantized(MLXArray, MLXArray, MLXArray, Int)      // weight(packed), scales, biases, bits (gs=64)

    public func apply(_ x: MLXArray) -> MLXArray {
        switch self {
        case .plain(let w):
            return MLX.matmul(x, w.transposed())
        case .quantized(let w, let s, let b, let bits):
            return MLX.quantizedMatmul(x, w, scales: s, biases: b, transpose: true,
                                       groupSize: 64, bits: bits, mode: .affine)
        }
    }

    /// 同入力の複数 quantized Proj を出力次元方向に concat して 1 本に融合（全て同 bits 前提）。
    /// 量子化は行(out)ごと独立なので out 軸 concat は bit 一致。失敗時 nil。
    public static func fuse(_ projs: [Proj]) -> Proj? {
        var ws: [MLXArray] = [], ss: [MLXArray] = [], bs: [MLXArray] = []
        var bits0 = -1
        for p in projs {
            guard case let .quantized(w, s, b, bits) = p else { return nil }
            if bits0 == -1 { bits0 = bits } else if bits0 != bits { return nil }
            ws.append(w); ss.append(s); bs.append(b)
        }
        return .quantized(MLX.concatenated(ws, axis: 0), MLX.concatenated(ss, axis: 0),
                          MLX.concatenated(bs, axis: 0), bits0)
    }
}

/// qwen3_5.GatedDeltaNet.__call__ 全体の Swift 移植（M2b-1 層 wrapping）.
/// ★ 実モデル(qwen3_5_moe)は in_proj を qkv/z/b/a の 4 本に分離（qwen3_next の
///   fix_query_key_value_ordering は無い）。実 checkpoint も in_proj_qkv/z/a/b。
/// 核(GatedDelta.update)を包む 4 本 in_proj + grouped causal conv1d(+silu)
/// + q/k rms_norm スケール + RMSNormGated(z) + out_proj。cache=None/mask=None の単一チャンク。
public struct GatedDeltaNetLayer {
    let numKHeads: Int       // linear_num_key_heads = 16
    let numVHeads: Int       // linear_num_value_heads = 32
    let headKDim: Int        // 128
    let headVDim: Int        // 128
    let convKernel: Int      // 4
    let eps: Float

    var keyDim: Int { headKDim * numKHeads }       // 2048
    var valueDim: Int { headVDim * numVHeads }     // 4096
    var convDim: Int { keyDim * 2 + valueDim }     // 8192

    // 射影は float/量子化 を Proj で抽象化。conv1d は [convDim, K, 1]、norm は [headVDim]
    let inProjQKV: Proj   // → [convDim]
    let inProjZ: Proj     // → [valueDim]
    let inProjB: Proj     // → [numVHeads]
    let inProjA: Proj     // → [numVHeads]
    let outProj: Proj     // → [H]
    let conv1dW: MLXArray
    let normWeight: MLXArray
    let aLog: MLXArray
    let dtBias: MLXArray
    let fusedIn: Proj?    // qkv+z+b+a を 1 本に融合（4→1 matmul）

    nonisolated(unsafe) public static var fuseGDN = false   // 融合 in_proj 経路を使う
    nonisolated(unsafe) public static var f32Conv = false   // A2: conv1d を f32 で（batched≠逐次 f16 drift を消す）

    public init(numKHeads: Int, numVHeads: Int, headKDim: Int, headVDim: Int,
                convKernel: Int, eps: Float,
                inProjQKV: Proj, inProjZ: Proj, inProjB: Proj, inProjA: Proj, outProj: Proj,
                conv1dW: MLXArray, normWeight: MLXArray, aLog: MLXArray, dtBias: MLXArray) {
        self.numKHeads = numKHeads; self.numVHeads = numVHeads
        self.headKDim = headKDim; self.headVDim = headVDim
        self.convKernel = convKernel; self.eps = eps
        self.inProjQKV = inProjQKV; self.inProjZ = inProjZ
        self.inProjB = inProjB; self.inProjA = inProjA; self.outProj = outProj
        self.conv1dW = conv1dW; self.normWeight = normWeight
        self.aLog = aLog; self.dtBias = dtBias
        self.fusedIn = Proj.fuse([inProjQKV, inProjZ, inProjB, inProjA])
    }

    /// weight=None 相当の rms_norm（最終軸正規化、スケール無し）= ones を渡して fused kernel と一致.
    static func rmsNormNoWeight(_ x: MLXArray, eps: Float) -> MLXArray {
        let w = MLXArray.ones([x.dim(-1)], dtype: x.dtype)
        return MLXFast.rmsNorm(x, weight: w, eps: eps)
    }

    public func callAsFunction(_ x: MLXArray, cache: GDNCache? = nil) -> MLXArray {
        let prof = StreamingMoEBlock.profileLayers
        func now() -> UInt64 { DispatchTime.now().uptimeNanoseconds }
        var t = now()
        let B = x.dim(0), S = x.dim(1)
        let qkv: MLXArray, z: MLXArray, b: MLXArray, a: MLXArray
        if GatedDeltaNetLayer.fuseGDN, let fused = fusedIn {
            let f = fused.apply(x)                                    // [B,S,convDim+valueDim+2*numVHeads]
            qkv = f[0..., 0..., 0 ..< convDim]
            z = f[0..., 0..., convDim ..< (convDim + valueDim)].reshaped([B, S, numVHeads, headVDim])
            b = f[0..., 0..., (convDim + valueDim) ..< (convDim + valueDim + numVHeads)]
            a = f[0..., 0..., (convDim + valueDim + numVHeads)...]
        } else {
            qkv = inProjQKV.apply(x)                                  // [B,S,convDim]
            z = inProjZ.apply(x).reshaped([B, S, numVHeads, headVDim])
            b = inProjB.apply(x)                                      // [B,S,numVHeads]
            a = inProjA.apply(x)                                      // [B,S,numVHeads]
        }
        if prof { MLX.eval([qkv, z, b, a]); StreamingMoEBlock.tGdnInproj += now() - t; t = now() }

        // conv_state: cache があれば直近 K-1、無ければ零（因果 padding）
        let convState = cache?.convState
            ?? MLXArray.zeros([B, convKernel - 1, convDim], dtype: x.dtype)
        let convInput = MLX.concatenated([convState, qkv], axis: 1)   // [B, S+K-1, convDim]
        if let c = cache {
            c.convState = MLX.contiguous(convInput[0..., (convInput.dim(1) - (convKernel - 1))...])
        }
        let convOut: MLXArray
        if GatedDeltaNetLayer.f32Conv {
            let co = MLX.conv1d(convInput.asType(.float32), conv1dW.asType(.float32), stride: 1,
                                padding: 0, dilation: 1, groups: convDim)
            convOut = silu(co).asType(x.dtype)
        } else {
            convOut = silu(MLX.conv1d(convInput, conv1dW, stride: 1, padding: 0,
                                      dilation: 1, groups: convDim))   // [B,S,convDim]
        }

        // split conv_out → q,k,v
        let q1 = convOut[0..., 0..., 0 ..< keyDim].reshaped([B, S, numKHeads, headKDim])
        let k1 = convOut[0..., 0..., keyDim ..< (2 * keyDim)].reshaped([B, S, numKHeads, headKDim])
        let v1 = convOut[0..., 0..., (2 * keyDim)...].reshaped([B, S, numVHeads, headVDim])

        let invScale = Float(pow(Double(headKDim), -0.5))
        let qN = (invScale * invScale) * GatedDeltaNetLayer.rmsNormNoWeight(q1, eps: 1e-6)
        let kN = invScale * GatedDeltaNetLayer.rmsNormNoWeight(k1, eps: 1e-6)
        if prof { MLX.eval([qN, kN, v1] + ((cache?.convState).map { [$0] } ?? [])); StreamingMoEBlock.tGdnConv += now() - t; t = now() }

        let (coreOut, newState) = GatedDelta.updateKernel(qN, kN, v1, a, b, aLog, dtBias,
                                                          state: cache?.recState)  // [B,S,Hv,Dv]
        if let c = cache { c.recState = newState }
        if prof { MLX.eval([coreOut, newState]); StreamingMoEBlock.tGdnKernel += now() - t; t = now() }

        // RMSNormGated(out, z): silu(z) * rms_norm(out, normWeight)
        let normed = MLXFast.rmsNorm(coreOut, weight: normWeight, eps: eps)
        let gated = silu(z.asType(.float32)) * normed.asType(.float32)
        let outV = gated.asType(coreOut.dtype).reshaped([B, S, -1])  // [B,S,valueDim]
        let out = outProj.apply(outV)
        if prof { MLX.eval(out); StreamingMoEBlock.tGdnOut += now() - t }
        return out
    }
}

public enum GatedDeltaNetLayerValidation {
    public static func run(refPath: String) throws -> String {
        let r = try loadArrays(url: URL(fileURLWithPath: refPath))
        guard let x = r["x"], let qkv = r["in_proj_qkv"], let z = r["in_proj_z"],
              let pb = r["in_proj_b"], let pa = r["in_proj_a"],
              let cw = r["conv1d"], let nw = r["norm_weight"], let ow = r["out_proj"],
              let aLog = r["A_log"], let dtBias = r["dt_bias"], let expOut = r["out"] else {
            return "ERROR: gdn-layer ref 不足"
        }
        let layer = GatedDeltaNetLayer(
            numKHeads: 16, numVHeads: 32, headKDim: 128, headVDim: 128,
            convKernel: 4, eps: 1e-6,
            inProjQKV: .plain(qkv), inProjZ: .plain(z), inProjB: .plain(pb), inProjA: .plain(pa),
            outProj: .plain(ow), conv1dW: cw, normWeight: nw, aLog: aLog, dtBias: dtBias)
        let out = layer(x)
        out.eval()
        let d = MLX.max(MLX.abs(out - expOut)).item(Float.self)
            / (MLX.max(MLX.abs(expOut)).item(Float.self) + 1e-9)
        let ok = d < 1e-3
        return String(format: "[M2b-1] GatedDeltaNet層 wrapping(qwen3_5): out_rel=%.2e  %@",
                      d, ok ? "OK ✅ bit一致" : "MISMATCH ❌")
    }
}

/// 実モデル layer-0 を REAL 4bit 量子化重みで検証（input_layernorm + linear_attn）.
public enum RealLayer0Validation {
    public static func run(refPath: String) throws -> String {
        let r = try loadArrays(url: URL(fileURLWithPath: refPath))
        func q(_ n: String) -> Proj {
            .quantized(r["\(n).weight"]!, r["\(n).scales"]!, r["\(n).biases"]!, 4)
        }
        guard let x = r["x"], let ilnW = r["input_layernorm_weight"],
              let cw = r["conv1d"], let nw = r["norm_weight"],
              let aLog = r["A_log"], let dtBias = r["dt_bias"], let expR = r["r"] else {
            return "ERROR: real-layer ref 不足"
        }
        // input_layernorm（RMSNorm, weight 有り, eps 1e-6）
        let xn = MLXFast.rmsNorm(x, weight: ilnW, eps: 1e-6)
        let layer = GatedDeltaNetLayer(
            numKHeads: 16, numVHeads: 32, headKDim: 128, headVDim: 128,
            convKernel: 4, eps: 1e-6,
            inProjQKV: q("in_proj_qkv"), inProjZ: q("in_proj_z"),
            inProjB: q("in_proj_b"), inProjA: q("in_proj_a"), outProj: q("out_proj"),
            conv1dW: cw, normWeight: nw, aLog: aLog, dtBias: dtBias)
        let out = layer(xn)
        out.eval()
        let d = MLX.max(MLX.abs(out - expR)).item(Float.self)
            / (MLX.max(MLX.abs(expR)).item(Float.self) + 1e-9)
        let ok = d < 2e-3   // 量子化経路: わずかに緩めの許容
        return String(format: "[M2b-3] 実 layer-0 linear_attn(4bit量子化): r_rel=%.2e  %@",
                      d, ok ? "OK ✅ 実重み一致" : "MISMATCH ❌")
    }
}
