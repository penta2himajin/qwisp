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
    nonisolated(unsafe) public static var fuseRMSGated = false  // issue#5: out-stage RMSNormGated を custom kernel 1 dispatch に融合

    /// issue#5 融合カーネル: RMSNormGated。out-stage の rms_norm(coreOut,w)→silu(z)*normed を 1 dispatch に。
    /// 行(=B·S·Hv)ごとに D=headVDim 要素を f32 で処理。numerics は元 MLX 経路を踏襲:
    ///   normed = (coreOut/rms)*w を f16 丸め(元 rmsNorm は coreOut.dtype 出力)→ f32 で silu(z) と積。
    /// eps/D はこのモデル固定(1e-6 / 128)ゆえカーネル内定数化。出力 dtype は template T。
    nonisolated(unsafe) static var _rmsGatedKernel: MLXFast.MLXFastKernel?
    static func rmsNormGatedFused(_ coreOut: MLXArray, _ z: MLXArray, _ weight: MLXArray,
                                  D: Int, eps: Float, outType: DType) -> MLXArray {
        let rows = coreOut.size / D
        let co = coreOut.reshaped([rows, D]), zz = z.reshaped([rows, D])
        if _rmsGatedKernel == nil {
            // 行(=B·S·Hv)ごとに 1 threadgroup・D スレッド。各スレッドが 1 要素を担当し、
            // sum(c^2) を threadgroup 内ツリー reduction → rms → gate。GPU 並列で 4 op を 1 dispatch 化。
            _rmsGatedKernel = MLXFast.metalKernel(
                name: "rms_norm_gated",
                inputNames: ["co", "zz", "w"],
                outputNames: ["out"],
                source: """
                    const uint D = co_shape[1];
                    uint d = thread_position_in_threadgroup.x;
                    uint row = thread_position_in_grid.y;
                    threadgroup float sh[1024];
                    float c = (float)co[row*D + d];
                    sh[d] = c * c;
                    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
                    for (uint s = D >> 1; s > 0; s >>= 1) {
                        if (d < s) sh[d] += sh[d + s];
                        threadgroup_barrier(metal::mem_flags::mem_threadgroup);
                    }
                    float rms = metal::rsqrt(sh[0] / (float)D + 1e-6f);
                    T normed_t = (T)(c * rms * (float)w[d]);     // 元: rms_norm 出力は coreOut.dtype 丸め
                    float zv = (float)zz[row*D + d];
                    float sgate = zv / (1.0f + metal::exp(-zv));
                    out[row*D + d] = (T)(sgate * (float)normed_t);
                """)
        }
        let r = _rmsGatedKernel!(
            [co, zz, weight],
            template: [("T", outType)],
            grid: (D, rows, 1), threadGroup: (D, 1, 1),
            outputShapes: [[rows, D]], outputDTypes: [outType])
        return r[0]
    }

    /// issue#5 融合カーネル: rmsNormNoWeight(x)*scale を 1 dispatch に（conv-stage の qN/kN 用）。
    /// 元: scale * rms_norm(x, ones)。normed を x.dtype 丸め → scale 積（元の scalar mul 経路）。
    nonisolated(unsafe) static var _rmsScaledKernel: MLXFast.MLXFastKernel?
    static func rmsNormScaledFused(_ x: MLXArray, scale: Float, D: Int, outType: DType) -> MLXArray {
        let rows = x.size / D
        let xr = x.reshaped([rows, D])
        if _rmsScaledKernel == nil {
            _rmsScaledKernel = MLXFast.metalKernel(
                name: "rms_norm_scaled",
                inputNames: ["x", "sc"],
                outputNames: ["out"],
                source: """
                    const uint D = x_shape[1];
                    uint d = thread_position_in_threadgroup.x;
                    uint row = thread_position_in_grid.y;
                    threadgroup float sh[1024];
                    float c = (float)x[row*D + d];
                    sh[d] = c * c;
                    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
                    for (uint s = D >> 1; s > 0; s >>= 1) {
                        if (d < s) sh[d] += sh[d + s];
                        threadgroup_barrier(metal::mem_flags::mem_threadgroup);
                    }
                    float rms = metal::rsqrt(sh[0] / (float)D + 1e-6f);
                    T normed_t = (T)(c * rms);              // 元: rms_norm(ones) 出力は x.dtype 丸め
                    out[row*D + d] = (T)((float)normed_t * sc[0]);
                """)
        }
        let r = _rmsScaledKernel!(
            [xr, MLXArray([scale], [1])],
            template: [("T", outType)],
            grid: (D, rows, 1), threadGroup: (D, 1, 1),
            outputShapes: [[rows, D]], outputDTypes: [outType])
        return r[0]
    }

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
        let qN: MLXArray, kN: MLXArray
        if GatedDeltaNetLayer.fuseRMSGated {                          // issue#5: qk-norm を融合カーネルに
            qN = GatedDeltaNetLayer.rmsNormScaledFused(q1, scale: invScale * invScale, D: headKDim, outType: q1.dtype)
                .reshaped([B, S, numKHeads, headKDim])
            kN = GatedDeltaNetLayer.rmsNormScaledFused(k1, scale: invScale, D: headKDim, outType: k1.dtype)
                .reshaped([B, S, numKHeads, headKDim])
        } else {
            qN = (invScale * invScale) * GatedDeltaNetLayer.rmsNormNoWeight(q1, eps: 1e-6)
            kN = invScale * GatedDeltaNetLayer.rmsNormNoWeight(k1, eps: 1e-6)
        }
        if prof { MLX.eval([qN, kN, v1] + ((cache?.convState).map { [$0] } ?? [])); StreamingMoEBlock.tGdnConv += now() - t; t = now() }

        let (coreOut, newState) = GatedDelta.updateKernel(qN, kN, v1, a, b, aLog, dtBias,
                                                          state: cache?.recState)  // [B,S,Hv,Dv]
        if let c = cache { c.recState = newState }
        if prof { MLX.eval([coreOut, newState]); StreamingMoEBlock.tGdnKernel += now() - t; t = now() }

        // RMSNormGated(out, z): silu(z) * rms_norm(out, normWeight)
        let outV: MLXArray
        if GatedDeltaNetLayer.fuseRMSGated {                          // issue#5: 1 dispatch 融合カーネル
            let g = GatedDeltaNetLayer.rmsNormGatedFused(coreOut, z, normWeight,
                                                         D: headVDim, eps: eps, outType: coreOut.dtype)
            outV = g.reshaped([B, S, -1])
        } else {
            let normed = MLXFast.rmsNorm(coreOut, weight: normWeight, eps: eps)
            let gated = silu(z.asType(.float32)) * normed.asType(.float32)
            outV = gated.asType(coreOut.dtype).reshaped([B, S, -1])  // [B,S,valueDim]
        }
        let out = outProj.apply(outV)
        if prof { MLX.eval(out); StreamingMoEBlock.tGdnOut += now() - t }
        return out
    }
}
