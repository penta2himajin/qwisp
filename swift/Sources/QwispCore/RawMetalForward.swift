import Foundation
import Metal
import MLX
import MLXFast
import MLXRandom

/// issue#5 raw-Metal forward 本実装の足場。MLX を迂回し forward を自前 Metal kernel + 単一 encoder で
/// 組むための基盤。第一歩 = quantized matmul(4-bit affine, gs=64)を MLX の quantizedMatmul と bit-exact
/// 照合（最難関の format + MLX weight buffer 共有 を検証）。
public enum RawMetalForward {
    nonisolated(unsafe) static var _qmmPipeline: MTLComputePipelineState?
    nonisolated(unsafe) static var _device: MTLDevice?
    nonisolated(unsafe) static var _queue: MTLCommandQueue?

    static func ensure() -> (MTLDevice, MTLCommandQueue)? {
        if let d = _device, let q = _queue { return (d, q) }
        guard let d = MTLCreateSystemDefaultDevice(), let q = d.makeCommandQueue() else { return nil }
        _device = d; _queue = q
        return (d, q)
    }

    /// 4-bit affine quantized matmul（decode gemv 一般: x[M,K] · Wq[N,K] → out[M,N], transpose=true）。
    /// dequant: w[n,k] = scales[n, k/gs]·nibble + biases[n, k/gs]、nibble=低位から 8 個/uint32。
    /// MLX weight buffer(wq/scales/biases)を asMTLBuffer(noCopy)で共有して読む。
    static func qmm(_ x: MLXArray, _ wq: MLXArray, scales: MLXArray, biases: MLXArray,
                    M: Int, K: Int, N: Int, bits: Int = 4, gs: Int = 64) -> MLXArray? {
        guard let (device, queue) = ensure() else { return nil }
        if _qmmPipeline == nil {
            let src = """
            #include <metal_stdlib>
            using namespace metal;
            kernel void qmm4(device const half*  x      [[buffer(0)]],
                             device const uint*  wq     [[buffer(1)]],
                             device const half*  scales [[buffer(2)]],
                             device const half*  biases [[buffer(3)]],
                             device half*        out    [[buffer(4)]],
                             constant uint&       K      [[buffer(5)]],
                             constant uint&       N      [[buffer(6)]],
                             constant uint&       GS     [[buffer(7)]],
                             uint gid [[thread_position_in_grid]]) {
                uint m = gid / N, n = gid % N;
                uint kp = K / 8;          // uint32 / row（4bit×8）
                uint kg = K / GS;         // group / row
                float acc = 0.0f;
                for (uint k = 0; k < K; ++k) {
                    uint packed = wq[n*kp + (k >> 3)];
                    uint nib = (packed >> (4u * (k & 7u))) & 0xFu;
                    uint g = k / GS;
                    float w = (float)scales[n*kg + g] * (float)nib + (float)biases[n*kg + g];
                    acc += (float)x[m*K + k] * w;
                }
                out[m*N + n] = (half)acc;
            }
            """
            do {
                let lib = try device.makeLibrary(source: src, options: nil)
                _qmmPipeline = try device.makeComputePipelineState(function: lib.makeFunction(name: "qmm4")!)
            } catch { print("[raw-qmm] compile error: \(error)"); return nil }
        }
        // MLX weight を MTLBuffer 共有（noCopy）。x も同様。out は新規。
        guard let bx = x.asType(.float16).asMTLBuffer(device: device, noCopy: false),
              let bwq = wq.asMTLBuffer(device: device, noCopy: false),
              let bsc = scales.asType(.float16).asMTLBuffer(device: device, noCopy: false),
              let bbi = biases.asType(.float16).asMTLBuffer(device: device, noCopy: false)
        else { return nil }
        let outBuf = device.makeBuffer(length: M * N * 2, options: .storageModeShared)!
        let cb = queue.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(_qmmPipeline!)
        enc.setBuffer(bx, offset: 0, index: 0)
        enc.setBuffer(bwq, offset: 0, index: 1)
        enc.setBuffer(bsc, offset: 0, index: 2)
        enc.setBuffer(bbi, offset: 0, index: 3)
        enc.setBuffer(outBuf, offset: 0, index: 4)
        var kk = UInt32(K), nn = UInt32(N), g = UInt32(gs)
        enc.setBytes(&kk, length: 4, index: 5)
        enc.setBytes(&nn, length: 4, index: 6)
        enc.setBytes(&g, length: 4, index: 7)
        let total = M * N
        let tgw = min(_qmmPipeline!.maxTotalThreadsPerThreadgroup, 256)
        enc.dispatchThreads(MTLSize(width: total, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: tgw, height: 1, depth: 1))
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        // MTLBuffer → MLXArray（f16, [M,N]）
        let ptr = outBuf.contents().bindMemory(to: Float16.self, capacity: total)
        let arr = Array(UnsafeBufferPointer(start: ptr, count: total))
        return MLXArray(arr, [M, N])
    }

    nonisolated(unsafe) static var _rmsPipeline: MTLComputePipelineState?
    nonisolated(unsafe) static var _softmaxPipeline: MTLComputePipelineState?

    /// raw-Metal rmsNorm: out[r,d] = x[r,d]·rsqrt(mean_d(x^2)+eps)·w[d]。行ごと TG スレッドで stride reduction
    /// (D>1024 でも可)。MLXFast.rmsNorm と一致。weight=nil で no-weight(ones 相当)。
    static func rmsNorm(_ x: MLXArray, _ weight: MLXArray?, eps: Float, D: Int) -> MLXArray? {
        guard let (device, queue) = ensure() else { return nil }
        if _rmsPipeline == nil {
            let src = """
            #include <metal_stdlib>
            using namespace metal;
            kernel void rmsnorm(device const half* x [[buffer(0)]],
                                device const half* w [[buffer(1)]],
                                device half* out     [[buffer(2)]],
                                constant uint& D     [[buffer(3)]],
                                constant float& eps  [[buffer(4)]],
                                constant uint& hasW  [[buffer(5)]],
                                uint t  [[thread_position_in_threadgroup]],
                                uint TG [[threads_per_threadgroup]],
                                uint row [[threadgroup_position_in_grid]]) {
                threadgroup float sh[1024];
                float local = 0.0f;
                for (uint d = t; d < D; d += TG) { float c = (float)x[row*D+d]; local += c*c; }
                sh[t] = local;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint s = TG>>1; s>0; s>>=1) { if (t<s) sh[t]+=sh[t+s]; threadgroup_barrier(mem_flags::mem_threadgroup); }
                float r = rsqrt(sh[0]/(float)D + eps);
                for (uint d = t; d < D; d += TG) {
                    float wv = hasW ? (float)w[d] : 1.0f;
                    out[row*D+d] = (half)((float)x[row*D+d] * r * wv);
                }
            }
            """
            do { let lib = try device.makeLibrary(source: src, options: nil)
                 _rmsPipeline = try device.makeComputePipelineState(function: lib.makeFunction(name: "rmsnorm")!)
            } catch { print("[raw-rms] compile: \(error)"); return nil }
        }
        let rows = x.size / D
        guard let bx = x.asType(.float16).asMTLBuffer(device: device, noCopy: false) else { return nil }
        let wArr = weight?.asType(.float16) ?? MLXArray.ones([1], dtype: .float16)
        guard let bw = wArr.asMTLBuffer(device: device, noCopy: false) else { return nil }
        let outBuf = device.makeBuffer(length: rows * D * 2, options: .storageModeShared)!
        let cb = queue.makeCommandBuffer()!; let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(_rmsPipeline!)
        enc.setBuffer(bx, offset: 0, index: 0); enc.setBuffer(bw, offset: 0, index: 1); enc.setBuffer(outBuf, offset: 0, index: 2)
        var dd = UInt32(D), ee = eps, hw = UInt32(weight == nil ? 0 : 1)
        enc.setBytes(&dd, length: 4, index: 3); enc.setBytes(&ee, length: 4, index: 4); enc.setBytes(&hw, length: 4, index: 5)
        let TG = min(D, 1024)
        enc.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: TG, height: 1, depth: 1))
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        let ptr = outBuf.contents().bindMemory(to: Float16.self, capacity: rows * D)
        return MLXArray(Array(UnsafeBufferPointer(start: ptr, count: rows * D)), [rows, D])
    }

    /// raw-Metal softmax(precise, f32): 行ごと max→exp→sum→div。MLX.softmax(precise:true) と一致。
    static func softmax(_ x: MLXArray, D: Int) -> MLXArray? {
        guard let (device, queue) = ensure() else { return nil }
        if _softmaxPipeline == nil {
            let src = """
            #include <metal_stdlib>
            using namespace metal;
            kernel void smax(device const half* x [[buffer(0)]], device half* out [[buffer(1)]],
                             constant uint& D [[buffer(2)]],
                             uint t [[thread_position_in_threadgroup]], uint TG [[threads_per_threadgroup]],
                             uint row [[threadgroup_position_in_grid]]) {
                threadgroup float sh[1024];
                float m = -INFINITY;
                for (uint d=t; d<D; d+=TG) m = max(m, (float)x[row*D+d]);
                sh[t]=m; threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint s=TG>>1; s>0; s>>=1){ if(t<s) sh[t]=max(sh[t],sh[t+s]); threadgroup_barrier(mem_flags::mem_threadgroup);}
                float mx=sh[0]; threadgroup_barrier(mem_flags::mem_threadgroup);
                float se=0.0f; for (uint d=t; d<D; d+=TG) se += exp((float)x[row*D+d]-mx);
                sh[t]=se; threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint s=TG>>1; s>0; s>>=1){ if(t<s) sh[t]+=sh[t+s]; threadgroup_barrier(mem_flags::mem_threadgroup);}
                float sum=sh[0];
                for (uint d=t; d<D; d+=TG) out[row*D+d] = (half)(exp((float)x[row*D+d]-mx)/sum);
            }
            """
            do { let lib = try device.makeLibrary(source: src, options: nil)
                 _softmaxPipeline = try device.makeComputePipelineState(function: lib.makeFunction(name: "smax")!)
            } catch { print("[raw-smax] compile: \(error)"); return nil }
        }
        let rows = x.size / D
        guard let bx = x.asType(.float16).asMTLBuffer(device: device, noCopy: false) else { return nil }
        let outBuf = device.makeBuffer(length: rows * D * 2, options: .storageModeShared)!
        let cb = queue.makeCommandBuffer()!; let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(_softmaxPipeline!)
        enc.setBuffer(bx, offset: 0, index: 0); enc.setBuffer(outBuf, offset: 0, index: 1)
        var dd = UInt32(D); enc.setBytes(&dd, length: 4, index: 2)
        let TG = min(D, 1024)
        enc.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: TG, height: 1, depth: 1))
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        let ptr = outBuf.contents().bindMemory(to: Float16.self, capacity: rows * D)
        return MLXArray(Array(UnsafeBufferPointer(start: ptr, count: rows * D)), [rows, D])
    }

    nonisolated(unsafe) static var _ropePipeline: MTLComputePipelineState?
    nonisolated(unsafe) static var _conv1dPipeline: MTLComputePipelineState?

    /// raw-Metal grouped causal conv1d + silu（GDN, decode S=1）。input[K,C](K=conv窓), w[C,K] → out[C]。
    /// out[c]=silu(Σ_k input[k,c]·w[c,k])、f32 累積（f32Conv 経路一致）。MLX conv1d(groups=C)+silu と照合。
    static func conv1dSilu(_ input: MLXArray, _ w: MLXArray, K: Int, C: Int) -> MLXArray? {
        guard let (device, queue) = ensure() else { return nil }
        if _conv1dPipeline == nil {
            let src = """
            #include <metal_stdlib>
            using namespace metal;
            kernel void conv1d_silu(device const half* x [[buffer(0)]],   // [K, C]
                                    device const half* w [[buffer(1)]],   // [C, K]
                                    device half* out     [[buffer(2)]],   // [C]
                                    constant uint& K [[buffer(3)]], constant uint& C [[buffer(4)]],
                                    uint c [[thread_position_in_grid]]) {
                if (c >= C) return;
                float acc = 0.0f;
                for (uint k = 0; k < K; ++k) acc += (float)x[k*C + c] * (float)w[c*K + k];
                out[c] = (half)(acc / (1.0f + exp(-acc)));     // silu
            }
            """
            do { let lib = try device.makeLibrary(source: src, options: nil)
                 _conv1dPipeline = try device.makeComputePipelineState(function: lib.makeFunction(name: "conv1d_silu")!)
            } catch { print("[raw-conv1d] compile: \(error)"); return nil }
        }
        guard let bx = input.asType(.float16).asMTLBuffer(device: device, noCopy: false),
              let bw = w.asType(.float16).asMTLBuffer(device: device, noCopy: false) else { return nil }
        let outBuf = device.makeBuffer(length: C * 2, options: .storageModeShared)!
        let cb = queue.makeCommandBuffer()!; let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(_conv1dPipeline!)
        enc.setBuffer(bx, offset: 0, index: 0); enc.setBuffer(bw, offset: 0, index: 1); enc.setBuffer(outBuf, offset: 0, index: 2)
        var kk = UInt32(K), cc = UInt32(C); enc.setBytes(&kk, length: 4, index: 3); enc.setBytes(&cc, length: 4, index: 4)
        let tgw = min(_conv1dPipeline!.maxTotalThreadsPerThreadgroup, 256)
        enc.dispatchThreads(MTLSize(width: C, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: tgw, height: 1, depth: 1))
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        let ptr = outBuf.contents().bindMemory(to: Float16.self, capacity: C)
        return MLXArray(Array(UnsafeBufferPointer(start: ptr, count: C)), [1, 1, C])
    }

    /// raw-Metal RoPE(非 traditional/NeoX, partial rotary)。x[rows, HD]、各行 position=offset(decode S=1)。
    /// rotary 部 rd dim を半分ペア(i, i+rd/2)で回転、rd..HD は passthrough。MLXFast.RoPE(traditional:false) 一致。
    static func rope(_ x: MLXArray, headDim HD: Int, ropeDim rd: Int, base: Float, offset: Int) -> MLXArray? {
        guard let (device, queue) = ensure() else { return nil }
        if _ropePipeline == nil {
            let src = """
            #include <metal_stdlib>
            using namespace metal;
            kernel void rope(device const half* x [[buffer(0)]], device half* out [[buffer(1)]],
                             constant uint& HD [[buffer(2)]], constant uint& RD [[buffer(3)]],
                             constant float& base [[buffer(4)]], constant float& pos [[buffer(5)]],
                             uint gid [[thread_position_in_grid]]) {
                uint row = gid / HD, d = gid % HD;
                if (d >= RD) { out[gid] = x[gid]; return; }
                uint hd2 = RD >> 1;
                uint i = d < hd2 ? d : d - hd2;
                float freq = exp(-2.0f * (float)i / (float)RD * log(base));
                float ang = pos * freq;
                float c = cos(ang), s = sin(ang);
                float x0 = (float)x[row*HD + i], x1 = (float)x[row*HD + i + hd2];
                out[gid] = (half_t)(d < hd2 ? (x0*c - x1*s) : (x0*s + x1*c));
            }
            """
            do { let lib = try device.makeLibrary(source: src.replacingOccurrences(of: "half_t", with: "half"), options: nil)
                 _ropePipeline = try device.makeComputePipelineState(function: lib.makeFunction(name: "rope")!)
            } catch { print("[raw-rope] compile: \(error)"); return nil }
        }
        let rows = x.size / HD
        guard let bx = x.asType(.float16).asMTLBuffer(device: device, noCopy: false) else { return nil }
        let outBuf = device.makeBuffer(length: rows * HD * 2, options: .storageModeShared)!
        let cb = queue.makeCommandBuffer()!; let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(_ropePipeline!)
        enc.setBuffer(bx, offset: 0, index: 0); enc.setBuffer(outBuf, offset: 0, index: 1)
        var h = UInt32(HD), r = UInt32(rd), b = base, p = Float(offset)
        enc.setBytes(&h, length: 4, index: 2); enc.setBytes(&r, length: 4, index: 3)
        enc.setBytes(&b, length: 4, index: 4); enc.setBytes(&p, length: 4, index: 5)
        let total = rows * HD, tgw = min(_ropePipeline!.maxTotalThreadsPerThreadgroup, 256)
        enc.dispatchThreads(MTLSize(width: total, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: tgw, height: 1, depth: 1))
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        let ptr = outBuf.contents().bindMemory(to: Float16.self, capacity: total)
        return MLXArray(Array(UnsafeBufferPointer(start: ptr, count: total)), [rows, HD])
    }

    /// 検証: rmsNorm / softmax を MLX と bit-exact 照合。
    /// - env: QWISP_RUN=raw-ops-test / QWISP_QMM_K(D, 既定2048) / QWISP_QMM_M(rows, 既定4)
    public static func runOpsTest() -> String {
        let D = envInt("QWISP_QMM_K", 2048), rows = envInt("QWISP_QMM_M", 4)
        var out = "[raw-ops-test rows=\(rows) D=\(D)] raw-Metal vs MLX bit-exact\n"
        // rmsNorm(weight 有)
        let x = MLXRandom.normal([rows, D]).asType(.float16)
        let w = MLXRandom.normal([D]).asType(.float16)
        let refR = MLXFast.rmsNorm(x, weight: w, eps: 1e-6); refR.eval()
        if let g = rmsNorm(x, w, eps: 1e-6, D: D) {
            g.eval()
            let rel = relErr(g, refR)
            out += String(format: "  rmsNorm(w):  rel=%.3e  %@\n", rel, rel < 2e-3 ? "OK ✅" : "MISMATCH ❌")
        } else { out += "  rmsNorm: kernel 失敗\n" }
        // softmax(precise)
        let refS = MLX.softmax(x, axis: -1, precise: true); refS.eval()
        if let g = softmax(x, D: D) {
            g.eval()
            let rel = relErr(g, refS)
            out += String(format: "  softmax:     rel=%.3e  %@\n", rel, rel < 2e-3 ? "OK ✅" : "MISMATCH ❌")
        } else { out += "  softmax: kernel 失敗\n" }
        // RoPE(non-traditional, partial rotary 64, base 1e7, offset 37) — attention config
        let HD = 256, rd = 64, base: Float = 1e7, offset = 37
        let xr = MLXRandom.normal([1, 16, 1, HD]).asType(.float16)   // [B=1, H=16, S=1, D] 実 attention 形状
        let refRo = MLXFast.RoPE(xr, dimensions: rd, traditional: false, base: base, scale: 1.0, offset: offset); refRo.eval()
        if let g = rope(xr.reshaped([16, HD]), headDim: HD, ropeDim: rd, base: base, offset: offset) {
            g.eval()
            let rel = relErr(g.reshaped([16, 1, HD]), refRo)
            out += String(format: "  RoPE:        rel=%.3e  %@", rel, rel < 2e-3 ? "OK ✅" : "MISMATCH ❌")
            if rel >= 2e-3 {
                let rfl = refRo.reshaped([16, HD]).asArray(Float.self)
                let gfl = g.asArray(Float.self)
                var mi = 0; var md: Float = 0
                for k in 0 ..< rfl.count { let dd = abs(rfl[k] - gfl[k]); if dd > md { md = dd; mi = k } }
                out += String(format: "\n    max diff @ row=%d dim=%d: ref=%.4f got=%.4f", mi / HD, mi % HD, rfl[mi], gfl[mi])
            }
        } else { out += "  RoPE: kernel 失敗" }
        // conv1d + silu (GDN grouped causal, K=4, decode S=1)
        let Cc = 512, Kk = 4
        let ci = MLXRandom.normal([1, Kk, Cc]).asType(.float16)
        let cw = MLXRandom.normal([Cc, Kk, 1]).asType(.float16)
        let conv = MLX.conv1d(ci.asType(.float32), cw.asType(.float32), stride: 1, padding: 0, dilation: 1, groups: Cc)
        let refC = (conv * MLX.sigmoid(conv)).asType(.float16); refC.eval()   // silu
        if let g = conv1dSilu(ci, cw.reshaped([Cc, Kk]), K: Kk, C: Cc) {
            g.eval()
            let rel = relErr(g.reshaped([1, 1, Cc]), refC)
            out += String(format: "\n  conv1d+silu: rel=%.3e  %@", rel, rel < 2e-3 ? "OK ✅" : "MISMATCH ❌")
        } else { out += "\n  conv1d: kernel 失敗" }
        return out
    }

    static func relErr(_ a: MLXArray, _ b: MLXArray) -> Float {
        let d = MLX.max(MLX.abs(a.asType(.float32) - b.asType(.float32))).item(Float.self)
        return d / (MLX.max(MLX.abs(b.asType(.float32))).item(Float.self) + 1e-9)
    }

    /// 検証: ランダム W を MLX.quantized で量子化 → MLX.quantizedMatmul vs raw qmm を bit-exact 照合。
    /// - env: QWISP_RUN=raw-qmm-test / QWISP_QMM_K(既定2048) / QWISP_QMM_N(既定2048) / QWISP_QMM_M(既定1)
    public static func runQmmTest() -> String {
        let K = envInt("QWISP_QMM_K", 2048), N = envInt("QWISP_QMM_N", 2048), M = envInt("QWISP_QMM_M", 1)
        let gs = 64, bits = 4
        let x = MLXRandom.normal([M, K]).asType(.float16)
        let w = MLXRandom.normal([N, K]).asType(.float16)
        let (wq, scales, biasesOpt) = MLX.quantized(w, groupSize: gs, bits: bits, mode: .affine)
        guard let biases = biasesOpt else { return "[raw-qmm] ERROR: affine biases nil" }
        MLX.eval([x, wq, scales, biases])
        let ref = MLX.quantizedMatmul(x, wq, scales: scales, biases: biases, transpose: true,
                                      groupSize: gs, bits: bits, mode: .affine)
        MLX.eval([ref])
        guard let got = qmm(x, wq, scales: scales, biases: biases, M: M, K: K, N: N, bits: bits, gs: gs) else {
            return "[raw-qmm] ERROR: kernel 実行失敗"
        }
        MLX.eval([got])
        let d = MLX.max(MLX.abs(got.asType(.float32) - ref.asType(.float32))).item(Float.self)
        let scale = MLX.max(MLX.abs(ref.asType(.float32))).item(Float.self) + 1e-9
        let rel = d / scale
        let ok = rel < 2e-3
        return String(format: "[raw-qmm-test M=%d K=%d N=%d, 4bit affine gs=64] raw-Metal vs MLX quantizedMatmul\n"
            + "  max|Δ|=%.3e  rel=%.3e  %@", M, K, N, d, rel,
            ok ? "OK ✅ bit-exact(MLX weight buffer 共有 + format 一致)" : "MISMATCH ❌(format 要修正)")
    }

    static func envInt(_ k: String, _ d: Int) -> Int {
        guard let v = ProcessInfo.processInfo.environment[k], let i = Int(v) else { return d }
        return i
    }
}
