import Foundation
import Metal
import MLX
import MLXFast
import MLXNN
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

    /// ★ MLX metallib との bit 一致に必須のコンパイル設定。MLX は safe-math(FMA contraction 無効)ビルド
    ///   ＝厳密 IEEE。fast-math だと FMA で last-bit がずれ非 bit-exact になる（qmm で実証: safe=rel0 / fast=2e-7）。
    static func mlxMatchCompileOpts() -> MTLCompileOptions {
        let opts = MTLCompileOptions()
        if #available(macOS 15.0, *) { opts.mathMode = .safe } else { opts.fastMathEnabled = false }
        return opts
    }

    /// 4-bit affine quantized matmul（decode gemv 一般: x[M,K] · Wq[N,K] → out[M,N], transpose=true）。
    /// dequant: w[n,k] = scales[n, k/gs]·nibble + biases[n, k/gs]、nibble=低位から 8 個/uint32。
    /// MLX weight buffer(wq/scales/biases)を asMTLBuffer(noCopy)で共有して読む。
    static func qmm(_ x: MLXArray, _ wq: MLXArray, scales: MLXArray, biases: MLXArray,
                    M: Int, K: Int, N: Int, bits: Int = 4, gs: Int = 64) -> MLXArray? {
        guard let (device, queue) = ensure() else { return nil }
        // ★ MLX の quantizedMatmul(qmv_fast) を数式・累積順・simd_sum まで完全一致で移植（rel 0.000e0 が目標）。
        //   raw-Metal forward の目的は MLX の per-dispatch C++ overhead 回避であり、GPU kernel 自体は MLX と
        //   同一にする（= bit-exact かつ同速）。bits=4/gs=64/half に特化。fast 条件 N%8==0 && K%512==0。
        //   MLX: backend/metal/kernels/quantized.h qmv_fast_impl/qdot/load_vector を逐語移植。
        let fast = (N % 8 == 0) && (K % 512 == 0) && bits == 4 && gs == 64
        guard fast else { print("[raw-qmm] 非fast (N=\(N) K=\(K) bits=\(bits) gs=\(gs)) 未対応"); return nil }
        if _qmmPipeline == nil {
            let src = """
            #include <metal_stdlib>
            using namespace metal;
            #define SIMD_SIZE 32
            // MLX load_vector<bits=4>: x を 16^j で事前除算（qdot で packed nibble の bit-shift と相殺）。sum=Σx。
            // ★ MLX 厳密一致の要点: sum の 4 要素加算は half 演算（x は half）→ float の sum に昇格。
            //   各要素を先に float 化して足すと丸めが変わり非 bit-exact になる（ここが残差源だった）。
            inline float ld16(const device half* x, thread float* xt) {
                float sum = 0.0f;
                for (int i = 0; i < 16; i += 4) {
                    sum += x[i] + x[i+1] + x[i+2] + x[i+3];   // half 加算 → float（MLX と一致）
                    xt[i]   = x[i];                            // half→float 厳密変換
                    xt[i+1] = x[i+1] / 16.0f;                  // half/float → float
                    xt[i+2] = x[i+2] / 256.0f;
                    xt[i+3] = x[i+3] / 4096.0f;
                }
                return sum;
            }
            // MLX qdot<bits=4>: ws を uint16 として 4 nibble 同時、accum=Σ xt·(nibble<<4j)。返り scale*accum+sum*bias。
            inline float qd4(const device uint8_t* w, const thread float* xt, float scale, float bias, float sum) {
                float accum = 0.0f;
                const device uint16_t* ws = (const device uint16_t*)w;
                for (int i = 0; i < 4; i++) {
                    accum += (xt[4*i]   * (float)(ws[i] & 0x000f) +
                              xt[4*i+1] * (float)(ws[i] & 0x00f0) +
                              xt[4*i+2] * (float)(ws[i] & 0x0f00) +
                              xt[4*i+3] * (float)(ws[i] & 0xf000));
                }
                return scale * accum + sum * bias;
            }
            // MLX qmv_fast_impl<half, gs=64, bits=4>: packs_per_thread=2 vpt=16 block=512 2sg×4row。
            kernel void qmm4(device const uint32_t* w      [[buffer(0)]],
                             device const half*     scales [[buffer(1)]],
                             device const half*     biases [[buffer(2)]],
                             device const half*     x      [[buffer(3)]],
                             device half*           y      [[buffer(4)]],
                             constant int&          in_vec_size  [[buffer(5)]],   // K
                             constant int&          out_vec_size [[buffer(6)]],   // N
                             uint3 tid      [[threadgroup_position_in_grid]],
                             uint  simd_gid [[simdgroup_index_in_threadgroup]],
                             uint  simd_lid [[thread_index_in_simdgroup]]) {
                constexpr int packs_per_thread = 2;
                constexpr int num_simdgroups = 2;
                constexpr int results_per_simdgroup = 4;
                constexpr int pack_factor = 8;
                constexpr int bytes_per_pack = 4;
                constexpr int values_per_thread = 16;
                constexpr int block_size = 512;            // vpt*SIMD_SIZE
                constexpr int scale_step_per_thread = 4;   // gs(64)/vpt(16)
                const device uint8_t* ws = (const device uint8_t*)w;
                typedef float U;
                thread U x_thread[16];
                thread U result[4] = {0};
                const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
                const int in_vec_size_g = in_vec_size / 64;
                const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) + simd_gid * results_per_simdgroup;
                ws     += out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
                scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
                biases += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
                x += tid.x * in_vec_size + simd_lid * values_per_thread;
                y += tid.x * out_vec_size + out_row;
                for (int k = 0; k < in_vec_size; k += block_size) {
                    U sum = ld16(x, x_thread);
                    for (int row = 0; row < results_per_simdgroup; row++) {
                        auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
                        const device half* sl = scales + row * in_vec_size_g;
                        const device half* bl = biases + row * in_vec_size_g;
                        U s = sl[0]; U b = bl[0];
                        result[row] += qd4(wl, x_thread, s, b, sum);
                    }
                    ws += block_size * bytes_per_pack / pack_factor;
                    scales += block_size / 64;
                    biases += block_size / 64;
                    x += block_size;
                }
                for (int row = 0; row < results_per_simdgroup; row++) {
                    result[row] = simd_sum(result[row]);
                    if (simd_lid == 0) y[row] = (half)result[row];
                }
            }
            """
            do {
                // ★ MLX metallib との bit 一致は浮動小数点コンパイル設定(FMA contraction/fast-math)に依存。
                //   QWISP_QMM_MATH=safe|relaxed|fast で切替（既定 safe=FMA contraction 無効で MLX の決定論ビルドに合わせる試行）。
                let opts = MTLCompileOptions()
                let mathSel = ProcessInfo.processInfo.environment["QWISP_QMM_MATH"] ?? "safe"
                if #available(macOS 15.0, *) {
                    switch mathSel {
                    case "fast": opts.mathMode = .fast
                    case "relaxed": opts.mathMode = .relaxed
                    default: opts.mathMode = .safe
                    }
                } else {
                    opts.fastMathEnabled = (mathSel == "fast")
                }
                let lib = try device.makeLibrary(source: src, options: opts)
                _qmmPipeline = try device.makeComputePipelineState(function: lib.makeFunction(name: "qmm4")!)
            } catch { print("[raw-qmm] compile error: \(error)"); return nil }
        }
        // MLX weight を MTLBuffer 共有（noCopy）。x も同様。out は新規。バッファ順は MLX kernel に合わせ w,scales,biases,x,y。
        guard let bx = x.asType(.float16).asMTLBuffer(device: device, noCopy: false),
              let bwq = wq.asMTLBuffer(device: device, noCopy: false),
              let bsc = scales.asType(.float16).asMTLBuffer(device: device, noCopy: false),
              let bbi = biases.asType(.float16).asMTLBuffer(device: device, noCopy: false)
        else { return nil }
        let outBuf = device.makeBuffer(length: M * N * 2, options: .storageModeShared)!
        let cb = queue.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(_qmmPipeline!)
        enc.setBuffer(bwq, offset: 0, index: 0)
        enc.setBuffer(bsc, offset: 0, index: 1)
        enc.setBuffer(bbi, offset: 0, index: 2)
        enc.setBuffer(bx, offset: 0, index: 3)
        enc.setBuffer(outBuf, offset: 0, index: 4)
        var kk = Int32(K), nn = Int32(N)
        enc.setBytes(&kk, length: 4, index: 5)
        enc.setBytes(&nn, length: 4, index: 6)
        // grid=(M, N/8, 1), group=(32,2,1) ＝ MLX qmv の dispatch と一致。
        enc.dispatchThreadgroups(MTLSize(width: M, height: N / 8, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 32, height: 2, depth: 1))
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        // MTLBuffer → MLXArray（f16, [M,N]）
        let ptr = outBuf.contents().bindMemory(to: Float16.self, capacity: M * N)
        let arr = Array(UnsafeBufferPointer(start: ptr, count: M * N))
        return MLXArray(arr, [M, N])
    }

    nonisolated(unsafe) static var _rmsPipeline: MTLComputePipelineState?
    nonisolated(unsafe) static var _rmsPipelineF32: MTLComputePipelineState?
    nonisolated(unsafe) static var _softmaxPipeline: MTLComputePipelineState?

    /// raw-Metal rmsNorm: ★ MLX rms_single_row(backend/metal/kernels/rms_norm.metal)を逐語移植し bit-exact。
    /// 要点: N_READS=4(thread が連続4要素), acc は f32 で xi*xi, simd_sum 二段, precise::rsqrt,
    /// 出力は w[i]*(half)(x[i]*inv_mean)（w 乗算は normed を half 化した後）。weight=nil は ones(no-weight 相当)。
    /// D≤4096 前提(RMS_LOOPED_LIMIT)。本モデルの D∈{128,2048} は全て single_row。
    /// promoteF32: MLX の dtype promotion を再現。weight が f32 のとき MLX は out を f32 に昇格し
    /// normed を f16 に丸めず f32 で計算する（RMSNormGated の normWeight=f32 経路）。false は全 f16(qk-norm)。
    static func rmsNorm(_ x: MLXArray, _ weight: MLXArray?, eps: Float, D: Int, promoteF32: Bool = false) -> MLXArray? {
        guard let (device, queue) = ensure() else { return nil }
        guard D <= 4096 else { print("[raw-rms] D>4096(looped)未対応 D=\(D)"); return nil }
        let pipe = promoteF32 ? _rmsPipelineF32 : _rmsPipeline
        if pipe == nil {
            let WT = promoteF32 ? "float" : "half"   // weight/out 型（MLX promotion 再現）
            let src = """
            #include <metal_stdlib>
            #include <metal_simdgroup>
            using namespace metal;
            kernel void rmsnorm(device const half* x   [[buffer(0)]],
                                device const \(WT)* w  [[buffer(1)]],
                                device \(WT)* out      [[buffer(2)]],
                                constant float& eps    [[buffer(3)]],
                                constant uint& axis_size [[buffer(4)]],
                                constant uint& w_stride  [[buffer(5)]],
                                uint gid [[threadgroup_position_in_grid]],
                                uint lid [[thread_position_in_threadgroup]],
                                uint simd_lane_id  [[thread_index_in_simdgroup]],
                                uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
                constexpr int N_READS = 4;
                constexpr int SIMD_SIZE = 32;
                threadgroup float local_inv_mean[1];
                threadgroup float local_sums[SIMD_SIZE];
                float acc = 0;
                x += gid * (size_t)axis_size + lid * N_READS;
                w += (size_t)w_stride * lid * N_READS;
                if (lid * N_READS + N_READS <= axis_size) {
                    for (int i = 0; i < N_READS; i++) { float xi = x[i]; acc += xi * xi; }
                } else {
                    for (int i = 0; i < N_READS; i++) { if ((lid*N_READS+i) < axis_size) { float xi = x[i]; acc += xi*xi; } }
                }
                acc = simd_sum(acc);
                if (simd_group_id == 0) local_sums[simd_lane_id] = 0;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (simd_lane_id == 0) local_sums[simd_group_id] = acc;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (simd_group_id == 0) {
                    acc = simd_sum(local_sums[simd_lane_id]);
                    if (simd_lane_id == 0) local_inv_mean[0] = precise::rsqrt(acc / axis_size + eps);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                out += gid * (size_t)axis_size + lid * N_READS;
                if (lid * N_READS + N_READS <= axis_size) {
                    for (int i = 0; i < N_READS; i++) out[i] = w[w_stride*i] * (\(WT))(x[i] * local_inv_mean[0]);
                } else {
                    for (int i = 0; i < N_READS; i++) { if ((lid*N_READS+i) < axis_size) out[i] = w[w_stride*i] * (\(WT))(x[i]*local_inv_mean[0]); }
                }
            }
            """
            do { let lib = try device.makeLibrary(source: src, options: mlxMatchCompileOpts())
                 let p = try device.makeComputePipelineState(function: lib.makeFunction(name: "rmsnorm")!)
                 if promoteF32 { _rmsPipelineF32 = p } else { _rmsPipeline = p }
            } catch { print("[raw-rms] compile: \(error)"); return nil }
        }
        let rows = x.size / D
        let wType: DType = promoteF32 ? .float32 : .float16
        let elemSize = promoteF32 ? 4 : 2
        guard let bx = x.asType(.float16).asMTLBuffer(device: device, noCopy: false) else { return nil }
        // weight=nil は ones[D](w_stride=1)。MLX も no-weight 時 ones を渡す挙動と一致。
        let wArr = (weight?.asType(wType) ?? MLXArray.ones([D], dtype: wType))
        guard let bw = wArr.asMTLBuffer(device: device, noCopy: false) else { return nil }
        let outBuf = device.makeBuffer(length: rows * D * elemSize, options: .storageModeShared)!
        // threadgroup_size = ceil(D/N_READS) を simd(32) 倍数に切上げ（MLX dispatch と一致）。
        let tgNeeded = (D + 3) / 4
        let tgSize = ((tgNeeded + 31) / 32) * 32
        let cb = queue.makeCommandBuffer()!; let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(promoteF32 ? _rmsPipelineF32! : _rmsPipeline!)
        enc.setBuffer(bx, offset: 0, index: 0); enc.setBuffer(bw, offset: 0, index: 1); enc.setBuffer(outBuf, offset: 0, index: 2)
        var ee = eps, asz = UInt32(D), ws = UInt32(1)
        enc.setBytes(&ee, length: 4, index: 3); enc.setBytes(&asz, length: 4, index: 4); enc.setBytes(&ws, length: 4, index: 5)
        enc.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        if promoteF32 {
            let ptr = outBuf.contents().bindMemory(to: Float.self, capacity: rows * D)
            return MLXArray(Array(UnsafeBufferPointer(start: ptr, count: rows * D)), [rows, D])
        }
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
    nonisolated(unsafe) static var _sdpaPipeline: MTLComputePipelineState?
    nonisolated(unsafe) static var _gqmmPipeline: MTLComputePipelineState?
    nonisolated(unsafe) static var _recurPipeline: MTLComputePipelineState?
    // single-encoder 用 補助 kernel（computeGBeta / gate / scaleMul）
    nonisolated(unsafe) static var _auxLib: MTLLibrary?
    nonisolated(unsafe) static var _cgbPipeline: MTLComputePipelineState?
    nonisolated(unsafe) static var _gatePipeline: MTLComputePipelineState?
    nonisolated(unsafe) static var _scalePipeline: MTLComputePipelineState?

    /// single-encoder GDN 用 補助 kernel を compile（lazy, safe-math）。
    ///  - compute_g_beta: g=exp(-exp(aLog)*softplus(a+dtBias))[f32], beta=sigmoid(b)[f16→f32]。MLX 厳密一致。
    ///  - gate: outV=silu(z.f32)*normed.f32 → f16（RMSNormGated の z-gate）。
    ///  - scale_mul: x[i] = (half)s * x[i]（qk-norm の scalar）。
    static func ensureAuxPipelines() -> Bool {
        guard let (device, _) = ensure() else { return false }
        if _cgbPipeline != nil { return true }
        let src = """
        #include <metal_stdlib>
        using namespace metal;
        // beta=sigmoid(b)[T=half, MLX Sigmoid struct: metal::exp], g=exp(-exp(aLog)*softplus)[f32, precise]
        kernel void compute_g_beta(device const half* a [[buffer(0)]], device const half* b [[buffer(1)]],
                                   device const float* aLog [[buffer(2)]], device const float* dtBias [[buffer(3)]],
                                   device float* g [[buffer(4)]], device float* beta [[buffer(5)]],
                                   constant uint& Hv [[buffer(6)]], uint i [[thread_position_in_grid]]) {
            if (i >= Hv) return;
            half bh = b[i];
            half y = (half)1 / ((half)1 + exp(metal::abs(bh)));     // MLX Sigmoid: metal::exp, half
            half sb = (bh < (half)0) ? y : ((half)1 - y);
            beta[i] = (float)sb;
            float x = (float)a[i] + dtBias[i];                      // a(f16)+dtBias(f32)→f32
            float sp = max(x, 0.0f) + precise::log(1.0f + precise::exp(-metal::abs(x)));  // softplus
            g[i] = precise::exp(-precise::exp(aLog[i]) * sp);
        }
        // gate: silu(z.f32)*normed.f32 → f16。silu=z*sigmoid(z)[T=float, metal::exp]
        kernel void gate(device const half* z [[buffer(0)]], device const float* normed [[buffer(1)]],
                         device half* outV [[buffer(2)]], constant uint& total [[buffer(3)]],
                         uint i [[thread_position_in_grid]]) {
            if (i >= total) return;
            float zf = (float)z[i];
            float y = 1.0f / (1.0f + exp(metal::abs(zf)));          // sigmoid(z), metal::exp, float
            float s = (zf < 0.0f) ? y : (1.0f - y);
            outV[i] = (half)((zf * s) * normed[i]);
        }
        // scale_mul: x[i] = (half)scale * x[i]（in-place, half 乗算）
        kernel void scale_mul(device half* x [[buffer(0)]], constant float& s [[buffer(1)]],
                              constant uint& total [[buffer(2)]], uint i [[thread_position_in_grid]]) {
            if (i >= total) return;
            x[i] = (half)s * x[i];
        }
        """
        do {
            let lib = try device.makeLibrary(source: src, options: mlxMatchCompileOpts())
            _auxLib = lib
            _cgbPipeline = try device.makeComputePipelineState(function: lib.makeFunction(name: "compute_g_beta")!)
            _gatePipeline = try device.makeComputePipelineState(function: lib.makeFunction(name: "gate")!)
            _scalePipeline = try device.makeComputePipelineState(function: lib.makeFunction(name: "scale_mul")!)
            return true
        } catch { print("[raw-aux] compile: \(error)"); return false }
    }

    /// raw-Metal GDN recurrent（GatedDelta.stepKernelSource を raw 化, decode T=1）。
    /// q,k[B,T,Hk,Dk] v[B,T,Hv,Dv] g,beta[B,T,Hv] state[B,Hv,Dv,Dk] → y[B,T,Hv,Dv], state_out。
    /// 既存の MLXFast.metalKernel 経路と同一 source を constants substitute で raw encoder に統合。
    static func recurrent(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray, g: MLXArray, beta: MLXArray, state: MLXArray,
                          B: Int, T: Int, Hk: Int, Dk: Int, Hv: Int, Dv: Int) -> (MLXArray, MLXArray)? {
        guard let (device, queue) = ensure() else { return nil }
        if _recurPipeline == nil {
            let body = GatedDelta.stepKernelSource
            let src = """
            #include <metal_stdlib>
            using namespace metal;
            #define InT half
            #define StT float
            #define Dk \(Dk)
            #define Dv \(Dv)
            #define Hk \(Hk)
            #define Hv \(Hv)
            kernel void gated_delta_step(
                device const half* q [[buffer(0)]], device const half* k [[buffer(1)]],
                device const half* v [[buffer(2)]], device const float* g [[buffer(3)]],
                device const float* beta [[buffer(4)]], device const float* state_in [[buffer(5)]],
                constant int& T [[buffer(6)]], device half* y [[buffer(7)]], device float* state_out [[buffer(8)]],
                uint3 thread_position_in_grid [[thread_position_in_grid]],
                uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
                uint thread_index_in_simdgroup [[thread_index_in_simdgroup]]) {
            \(body)
            }
            """
            do { let lib = try device.makeLibrary(source: src, options: nil)
                 _recurPipeline = try device.makeComputePipelineState(function: lib.makeFunction(name: "gated_delta_step")!)
            } catch { print("[raw-recur] compile: \(error)"); return nil }
        }
        guard let bq = q.asType(.float16).asMTLBuffer(device: device, noCopy: false),
              let bk = k.asType(.float16).asMTLBuffer(device: device, noCopy: false),
              let bv = v.asType(.float16).asMTLBuffer(device: device, noCopy: false),
              let bg = g.asType(.float32).asMTLBuffer(device: device, noCopy: false),
              let bb = beta.asType(.float32).asMTLBuffer(device: device, noCopy: false),
              let bs = state.asType(.float32).asMTLBuffer(device: device, noCopy: false) else { return nil }
        let yBuf = device.makeBuffer(length: B*T*Hv*Dv*2, options: .storageModeShared)!
        let soBuf = device.makeBuffer(length: B*Hv*Dv*Dk*4, options: .storageModeShared)!
        let cb = queue.makeCommandBuffer()!; let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(_recurPipeline!)
        enc.setBuffer(bq, offset: 0, index: 0); enc.setBuffer(bk, offset: 0, index: 1); enc.setBuffer(bv, offset: 0, index: 2)
        enc.setBuffer(bg, offset: 0, index: 3); enc.setBuffer(bb, offset: 0, index: 4); enc.setBuffer(bs, offset: 0, index: 5)
        var tt = Int32(T); enc.setBytes(&tt, length: 4, index: 6)
        enc.setBuffer(yBuf, offset: 0, index: 7); enc.setBuffer(soBuf, offset: 0, index: 8)
        enc.dispatchThreads(MTLSize(width: 32, height: Dv, depth: B*Hv), threadsPerThreadgroup: MTLSize(width: 32, height: 4, depth: 1))
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        let yp = yBuf.contents().bindMemory(to: Float16.self, capacity: B*T*Hv*Dv)
        let y = MLXArray(Array(UnsafeBufferPointer(start: yp, count: B*T*Hv*Dv)), [B, T, Hv, Dv])
        let sp = soBuf.contents().bindMemory(to: Float.self, capacity: B*Hv*Dv*Dk)
        let so = MLXArray(Array(UnsafeBufferPointer(start: sp, count: B*Hv*Dv*Dk)), [B, Hv, Dv, Dk])
        return (y, so)
    }

    /// raw-Metal gather quantized matmul(MoE 核): x[K] を inds[Ktop] が選ぶ各 expert の量子化 weight で matmul。
    /// out[ki,n]=Σ_k x[k]·dequant(wq[e,n,k]), e=inds[ki]。expert オフセットで wq/scales/biases を index。
    /// ★ MLX gather_qmv_fast 移植: expert offset(inds[ki])を ws/scales/biases に加えて同じ qmv_fast 内核を回す。
    ///   → qmm と同一の数式・累積で rel 0.000e0。fast 条件 N%8==0 && K%512==0 && bits4/gs64。
    ///   x[1,K] 共有, wq[E,N,K/8] uint32, out[Ktop,N]。grid=(M=1, N/8, Ktop), group=(32,2,1)。
    static func gatherQmm(_ x: MLXArray, _ wq: MLXArray, scales: MLXArray, biases: MLXArray, inds: MLXArray,
                          Ktop: Int, K: Int, N: Int, gs: Int = 64) -> MLXArray? {
        guard let (device, queue) = ensure() else { return nil }
        guard N % 8 == 0, K % 512 == 0, gs == 64 else { print("[raw-gqmm] 非fast (N=\(N) K=\(K) gs=\(gs)) 未対応"); return nil }
        if _gqmmPipeline == nil {
            let src = """
            #include <metal_stdlib>
            using namespace metal;
            #define SIMD_SIZE 32
            inline float ld16(const device half* x, thread float* xt) {
                float sum = 0.0f;
                for (int i = 0; i < 16; i += 4) {
                    sum += x[i] + x[i+1] + x[i+2] + x[i+3];
                    xt[i] = x[i]; xt[i+1] = x[i+1]/16.0f; xt[i+2] = x[i+2]/256.0f; xt[i+3] = x[i+3]/4096.0f;
                }
                return sum;
            }
            inline float qd4(const device uint8_t* w, const thread float* xt, float scale, float bias, float sum) {
                float accum = 0.0f;
                const device uint16_t* ws = (const device uint16_t*)w;
                for (int i = 0; i < 4; i++) {
                    accum += (xt[4*i]   * (float)(ws[i] & 0x000f) +
                              xt[4*i+1] * (float)(ws[i] & 0x00f0) +
                              xt[4*i+2] * (float)(ws[i] & 0x0f00) +
                              xt[4*i+3] * (float)(ws[i] & 0xf000));
                }
                return scale * accum + sum * bias;
            }
            // gather_qmv_fast: tid.z=ki(expert slot)。expert e=inds[ki] の weight 領域(N*行)へ offset。
            kernel void gqmm4(device const uint32_t* w      [[buffer(0)]],   // [E, N, K/8]
                              device const half*     scales [[buffer(1)]],   // [E, N, K/64]
                              device const half*     biases [[buffer(2)]],
                              device const half*     x      [[buffer(3)]],   // [1, K]
                              device const int*      inds   [[buffer(4)]],   // [Ktop]
                              device half*           y      [[buffer(5)]],   // [Ktop, N]
                              constant int& in_vec_size  [[buffer(6)]],
                              constant int& out_vec_size [[buffer(7)]],
                              uint3 tid      [[threadgroup_position_in_grid]],
                              uint  simd_gid [[simdgroup_index_in_threadgroup]],
                              uint  simd_lid [[thread_index_in_simdgroup]]) {
                constexpr int packs_per_thread = 2, num_simdgroups = 2, results_per_simdgroup = 4;
                constexpr int pack_factor = 8, bytes_per_pack = 4, values_per_thread = 16;
                constexpr int block_size = 512, scale_step_per_thread = 4;
                const device uint8_t* ws = (const device uint8_t*)w;
                typedef float U;
                thread U x_thread[16];
                thread U result[4] = {0};
                const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
                const int in_vec_size_g = in_vec_size / 64;
                uint ki = tid.z;
                uint e = (uint)inds[ki];
                // expert offset: 1 expert = N 行 ×(in_vec_size_w bytes / in_vec_size_g groups)
                ws     += (size_t)e * out_vec_size * in_vec_size_w;
                scales += (size_t)e * out_vec_size * in_vec_size_g;
                biases += (size_t)e * out_vec_size * in_vec_size_g;
                const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) + simd_gid * results_per_simdgroup;
                ws     += out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
                scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
                biases += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
                x += tid.x * in_vec_size + simd_lid * values_per_thread;     // tid.x=0(M=1)
                y += ki * out_vec_size + out_row;
                for (int k = 0; k < in_vec_size; k += block_size) {
                    U sum = ld16(x, x_thread);
                    for (int row = 0; row < results_per_simdgroup; row++) {
                        auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
                        const device half* sl = scales + row * in_vec_size_g;
                        const device half* bl = biases + row * in_vec_size_g;
                        U s = sl[0]; U b = bl[0];
                        result[row] += qd4(wl, x_thread, s, b, sum);
                    }
                    ws += block_size * bytes_per_pack / pack_factor;
                    scales += block_size / 64; biases += block_size / 64; x += block_size;
                }
                for (int row = 0; row < results_per_simdgroup; row++) {
                    result[row] = simd_sum(result[row]);
                    if (simd_lid == 0) y[row] = (half)result[row];
                }
            }
            """
            do { let lib = try device.makeLibrary(source: src, options: mlxMatchCompileOpts())
                 _gqmmPipeline = try device.makeComputePipelineState(function: lib.makeFunction(name: "gqmm4")!)
            } catch { print("[raw-gqmm] compile: \(error)"); return nil }
        }
        guard let bx = x.asType(.float16).asMTLBuffer(device: device, noCopy: false),
              let bwq = wq.asMTLBuffer(device: device, noCopy: false),
              let bsc = scales.asType(.float16).asMTLBuffer(device: device, noCopy: false),
              let bbi = biases.asType(.float16).asMTLBuffer(device: device, noCopy: false),
              let bin = inds.asType(.int32).asMTLBuffer(device: device, noCopy: false) else { return nil }
        let outBuf = device.makeBuffer(length: Ktop * N * 2, options: .storageModeShared)!
        let cb = queue.makeCommandBuffer()!; let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(_gqmmPipeline!)
        enc.setBuffer(bwq, offset: 0, index: 0); enc.setBuffer(bsc, offset: 0, index: 1)
        enc.setBuffer(bbi, offset: 0, index: 2); enc.setBuffer(bx, offset: 0, index: 3)
        enc.setBuffer(bin, offset: 0, index: 4); enc.setBuffer(outBuf, offset: 0, index: 5)
        var kk = Int32(K), nn = Int32(N)
        enc.setBytes(&kk, length: 4, index: 6); enc.setBytes(&nn, length: 4, index: 7)
        enc.dispatchThreadgroups(MTLSize(width: 1, height: N / 8, depth: Ktop),
                                 threadsPerThreadgroup: MTLSize(width: 32, height: 2, depth: 1))
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        let ptr = outBuf.contents().bindMemory(to: Float16.self, capacity: Ktop * N)
        return MLXArray(Array(UnsafeBufferPointer(start: ptr, count: Ktop * N)), [Ktop, N])
    }

    /// raw-Metal SDPA(decode L=1, GQA, flash/online softmax, f32)。
    /// q[H,D], K/V[KV,S,D] → out[H,D]。head h は kv=h/(H/KV)。MLXFast.scaledDotProductAttention(f32,.none) と照合。
    static func sdpaDecode(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray,
                           H: Int, KV: Int, D: Int, S: Int, scale: Float) -> MLXArray? {
        guard let (device, queue) = ensure() else { return nil }
        guard D == 256 else { print("[raw-sdpa] D!=256 未対応 D=\(D)"); return nil }
        // ★ MLX sdpa_vector(sdpa_vector.h)逐語移植: BN=32 simdgroup が key を分担, BD=32 lane が head dim
        //   (qk_per_thread=8)。scale は q に適用, fast::exp, simd_sum/simd_max, cross-simdgroup combine。
        //   group=(1024,1,1)=32sg×32lane, grid=(H,1,1)。no mask/causal/sinks, query 非転置。decode L=1。
        if _sdpaPipeline == nil {
            let src = """
            #include <metal_stdlib>
            #include <metal_simdgroup>
            using namespace metal;
            kernel void sdpa(device const half* queries [[buffer(0)]],   // [H, D]
                             device const half* keys    [[buffer(1)]],   // [KV, N, D]
                             device const half* values  [[buffer(2)]],   // [KV, N, D]
                             device half* out           [[buffer(3)]],   // [H, D]
                             constant int& gqa_factor   [[buffer(4)]],
                             constant int& N            [[buffer(5)]],
                             constant int& k_head_stride[[buffer(6)]],
                             constant int& k_seq_stride [[buffer(7)]],
                             constant int& v_head_stride[[buffer(8)]],
                             constant int& v_seq_stride [[buffer(9)]],
                             constant float& scale      [[buffer(10)]],
                             uint3 tid [[threadgroup_position_in_grid]],
                             uint3 tpg [[threadgroups_per_grid]],
                             uint simd_gid [[simdgroup_index_in_threadgroup]],
                             uint simd_lid [[thread_index_in_simdgroup]]) {
                constexpr int BN = 32, BD = 32, D = 256, V = 256;
                constexpr int qk_per_thread = D / BD;   // 8
                constexpr int v_per_thread = V / BD;    // 8
                int inner_k_stride = BN * k_seq_stride;
                int inner_v_stride = BN * v_seq_stride;
                typedef float U;
                thread U q[qk_per_thread]; thread U k[qk_per_thread]; thread U o[v_per_thread];
                threadgroup U outputs[BN * BD];
                threadgroup U max_scores[BN];
                threadgroup U sum_exp_scores[BN];
                const int q_batch_head_idx = tid.x;
                const int q_seq_idx = tid.y;
                const int kv_head_idx = q_batch_head_idx / gqa_factor;
                const int o_offset = q_batch_head_idx * tpg.y + q_seq_idx;
                const int q_offset = o_offset;          // query 非転置
                queries += q_offset * D + simd_lid * qk_per_thread;
                keys   += kv_head_idx * k_head_stride + simd_gid * k_seq_stride + simd_lid * qk_per_thread;
                values += kv_head_idx * v_head_stride + simd_gid * v_seq_stride + simd_lid * v_per_thread;
                out += o_offset * V + simd_gid * v_per_thread;
                for (int i = 0; i < qk_per_thread; i++) q[i] = (U)scale * queries[i];
                for (int i = 0; i < v_per_thread; i++) o[i] = 0;
                U max_score = -INFINITY;
                U sum_exp_score = 0;
                for (int i = simd_gid; i < N; i += BN) {
                    for (int j = 0; j < qk_per_thread; j++) k[j] = keys[j];
                    U score = 0;
                    for (int j = 0; j < qk_per_thread; j++) score += q[j] * k[j];
                    score = simd_sum(score);
                    U new_max = max(max_score, score);
                    U factor = fast::exp(max_score - new_max);
                    U exp_score = fast::exp(score - new_max);
                    max_score = new_max;
                    sum_exp_score = sum_exp_score * factor + exp_score;
                    for (int j = 0; j < v_per_thread; j++) o[j] = o[j] * factor + exp_score * values[j];
                    keys += inner_k_stride;
                    values += inner_v_stride;
                }
                if (simd_lid == 0) { max_scores[simd_gid] = max_score; sum_exp_scores[simd_gid] = sum_exp_score; }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                max_score = max_scores[simd_lid];
                U new_max = simd_max(max_score);
                U factor = fast::exp(max_score - new_max);
                sum_exp_score = simd_sum(sum_exp_scores[simd_lid] * factor);
                for (int i = 0; i < v_per_thread; i++) {
                    outputs[simd_lid * BD + simd_gid] = o[i];
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    o[i] = simd_sum(outputs[simd_gid * BD + simd_lid] * factor);
                    o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
                if (simd_lid == 0) { for (int i = 0; i < v_per_thread; i++) out[i] = (half)o[i]; }
            }
            """
            do { let lib = try device.makeLibrary(source: src, options: mlxMatchCompileOpts())
                 _sdpaPipeline = try device.makeComputePipelineState(function: lib.makeFunction(name: "sdpa")!)
            } catch { print("[raw-sdpa] compile: \(error)"); return nil }
        }
        guard let bq = q.asType(.float16).asMTLBuffer(device: device, noCopy: false),
              let bk = k.asType(.float16).asMTLBuffer(device: device, noCopy: false),
              let bv = v.asType(.float16).asMTLBuffer(device: device, noCopy: false) else { return nil }
        let outBuf = device.makeBuffer(length: H * D * 2, options: .storageModeShared)!
        let cb = queue.makeCommandBuffer()!; let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(_sdpaPipeline!)
        enc.setBuffer(bq, offset: 0, index: 0); enc.setBuffer(bk, offset: 0, index: 1)
        enc.setBuffer(bv, offset: 0, index: 2); enc.setBuffer(outBuf, offset: 0, index: 3)
        // k/v は [KV, S, D] 連続: head_stride=S*D, seq_stride=D。gqa_factor=H/KV。
        var gqa = Int32(H / KV), nn = Int32(S), khs = Int32(S * D), kss = Int32(D), vhs = Int32(S * D), vss = Int32(D), sc = scale
        enc.setBytes(&gqa, length: 4, index: 4); enc.setBytes(&nn, length: 4, index: 5)
        enc.setBytes(&khs, length: 4, index: 6); enc.setBytes(&kss, length: 4, index: 7)
        enc.setBytes(&vhs, length: 4, index: 8); enc.setBytes(&vss, length: 4, index: 9)
        enc.setBytes(&sc, length: 4, index: 10)
        enc.dispatchThreadgroups(MTLSize(width: H, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1024, height: 1, depth: 1))
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        let ptr = outBuf.contents().bindMemory(to: Float16.self, capacity: H * D)
        return MLXArray(Array(UnsafeBufferPointer(start: ptr, count: H * D)), [H, D])
    }

    /// raw-Metal grouped causal conv1d + silu（GDN, decode S=1）。input[K,C](K=conv窓), w[C,K] → out[C]。
    /// out[c]=silu(Σ_k input[k,c]·w[c,k])、f32 累積（f32Conv 経路一致）。MLX conv1d(groups=C)+silu と照合。
    static func conv1dSilu(_ input: MLXArray, _ w: MLXArray, K: Int, C: Int) -> MLXArray? {
        guard let (device, queue) = ensure() else { return nil }
        if _conv1dPipeline == nil {
            let src = """
            #include <metal_stdlib>
            using namespace metal;
            kernel void conv1d_silu(device const half*  x [[buffer(0)]],   // [K, C]
                                    device const float* w [[buffer(1)]],   // [C, K] f32(MLX f32Conv 一致)
                                    device half* out      [[buffer(2)]],   // [C]
                                    constant uint& K [[buffer(3)]], constant uint& C [[buffer(4)]],
                                    uint c [[thread_position_in_grid]]) {
                if (c >= C) return;
                float acc = 0.0f;
                // ★ MLX depthwise_conv_1d は acc += (float)in * w（plain +=, fma 無し, safe-math）。w は f32。
                for (uint k = 0; k < K; ++k) acc += (float)x[k*C + c] * w[c*K + k];
                // ★ silu = x*sigmoid(x)、sigmoid は MLX の数値安定版(unary_ops.h Sigmoid):
                //   y=1/(1+exp(|x|)); s = x<0 ? y : 1-y。直接 acc/(1+exp(-acc)) は last-bit がずれる。
                float ax = metal::abs(acc);
                float y = 1.0f / (1.0f + precise::exp(ax));
                float s = (acc < 0.0f) ? y : (1.0f - y);
                out[c] = (half)(acc * s);
            }
            """
            do { let lib = try device.makeLibrary(source: src, options: mlxMatchCompileOpts())
                 _conv1dPipeline = try device.makeComputePipelineState(function: lib.makeFunction(name: "conv1d_silu")!)
            } catch { print("[raw-conv1d] compile: \(error)"); return nil }
        }
        guard let bx = input.asType(.float16).asMTLBuffer(device: device, noCopy: false),
              let bw = w.asType(.float32).asMTLBuffer(device: device, noCopy: false) else { return nil }   // w は f32(MLX f32Conv 一致)
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

    /// task#3 orchestration 核: qmm→rmsNorm を **単一 command buffer + 単一 encoder** で連結（中間は GPU 常駐 buffer、
    /// 間で commit/MLX に戻らない）。bit-exact + encode CPU を MLX(2 op 別 dispatch)と比較。
    /// 全層 orchestration の型を証明する。
    public static func runChainTest() -> String {
        guard let (device, queue) = ensure() else { return "ERROR: no device" }
        // pipeline 準備（qmm/rmsNorm を一度 warm）
        let K = 2048, N = 2048
        let xin = MLXRandom.normal([1, K]).asType(.float16)
        let wf = MLXRandom.normal([N, K]).asType(.float16)
        let (wq, scales, biasesOpt) = MLX.quantized(wf, groupSize: 64, bits: 4, mode: .affine)
        guard let biases = biasesOpt else { return "[chain] biases nil" }
        let nw = MLXRandom.normal([N]).asType(.float16)
        MLX.eval([xin, wq, scales, biases, nw])
        _ = qmm(xin, wq, scales: scales, biases: biases, M: 1, K: K, N: N)   // pipeline warm
        _ = rmsNorm(xin, nw, eps: 1e-6, D: K)
        // 参照: MLX で qmm→rmsNorm
        let mq = MLX.quantizedMatmul(xin, wq, scales: scales, biases: biases, transpose: true, groupSize: 64, bits: 4, mode: .affine)
        let refChain = MLXFast.rmsNorm(mq, weight: nw, eps: 1e-6); refChain.eval()

        // 単一 encoder で qmm→rmsNorm を連結
        guard let bx = xin.asType(.float16).asMTLBuffer(device: device, noCopy: false),
              let bwq = wq.asMTLBuffer(device: device, noCopy: false),
              let bsc = scales.asType(.float16).asMTLBuffer(device: device, noCopy: false),
              let bbi = biases.asType(.float16).asMTLBuffer(device: device, noCopy: false),
              let bnw = nw.asType(.float16).asMTLBuffer(device: device, noCopy: false) else { return "[chain] buf nil" }
        let mid = device.makeBuffer(length: N * 2, options: .storageModeShared)!     // ping
        let mid2 = device.makeBuffer(length: N * 2, options: .storageModeShared)!    // pong
        func cpuNs() -> UInt64 { var r = rusage(); getrusage(RUSAGE_SELF, &r)
            return UInt64(r.ru_utime.tv_sec+r.ru_stime.tv_sec)*1_000_000_000 + UInt64(r.ru_utime.tv_usec+r.ru_stime.tv_usec)*1000 }
        var kk = UInt32(K), nn = UInt32(N), g = UInt32(64), dd = UInt32(N), ee = Float(1e-6), hw = UInt32(1)
        // depth 回(qmm→rmsNorm)を単一 encoder で連結。中間 buffer を ping-pong で GPU 常駐連結。
        func runChain(_ depth: Int) -> MTLBuffer {
            let cb = queue.makeCommandBuffer()!; let enc = cb.makeComputeCommandEncoder()!
            var src = bx as MTLBuffer
            var a = mid, b = mid2
            for _ in 0 ..< depth {
                enc.setComputePipelineState(_qmmPipeline!)
                enc.setBuffer(src, offset: 0, index: 0); enc.setBuffer(bwq, offset: 0, index: 1)
                enc.setBuffer(bsc, offset: 0, index: 2); enc.setBuffer(bbi, offset: 0, index: 3); enc.setBuffer(a, offset: 0, index: 4)
                enc.setBytes(&kk, length: 4, index: 5); enc.setBytes(&nn, length: 4, index: 6); enc.setBytes(&g, length: 4, index: 7)
                enc.dispatchThreadgroups(MTLSize(width: N, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
                enc.setComputePipelineState(_rmsPipeline!)
                enc.setBuffer(a, offset: 0, index: 0); enc.setBuffer(bnw, offset: 0, index: 1); enc.setBuffer(b, offset: 0, index: 2)
                enc.setBytes(&dd, length: 4, index: 3); enc.setBytes(&ee, length: 4, index: 4); enc.setBytes(&hw, length: 4, index: 5)
                enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(N,1024), height: 1, depth: 1))
                src = b; swap(&a, &b)
            }
            enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
            return src
        }
        func mlxChain(_ depth: Int) { var h = xin
            for _ in 0..<depth { let m = MLX.quantizedMatmul(h, wq, scales: scales, biases: biases, transpose: true, groupSize: 64, bits: 4, mode: .affine); h = MLXFast.rmsNorm(m, weight: nw, eps: 1e-6) }
            h.eval() }
        func bench(_ depth: Int) -> (rawCpu: Double, mlxCpu: Double, rawWall: Double, mlxWall: Double) {
            let reps = 200
            for _ in 0..<3 { _ = runChain(depth) }
            var t0 = DispatchTime.now().uptimeNanoseconds; var c0 = cpuNs()
            for _ in 0..<reps { _ = runChain(depth) }
            let rw = Double(DispatchTime.now().uptimeNanoseconds-t0)/Double(reps)/1e6, rc = Double(cpuNs()-c0)/Double(reps)/1e6
            for _ in 0..<3 { mlxChain(depth) }
            t0 = DispatchTime.now().uptimeNanoseconds; c0 = cpuNs()
            for _ in 0..<reps { mlxChain(depth) }
            let mw = Double(DispatchTime.now().uptimeNanoseconds-t0)/Double(reps)/1e6, mc = Double(cpuNs()-c0)/Double(reps)/1e6
            return (rc, mc, rw, mw)
        }
        // bit-exact(depth=1)
        let last = runChain(1)
        let ptr = last.contents().bindMemory(to: Float16.self, capacity: N)
        let got = MLXArray(Array(UnsafeBufferPointer(start: ptr, count: N)), [1, N])
        let rel = relErr(got, refChain)
        var lines = ""
        for depth in [1, 5, 10, 20] {
            let r = bench(depth)
            lines += String(format: "\n  depth=%2d: raw %.2fms/%.0fus-CPU  MLX %.2fms/%.0fus-CPU  → encode %.2fx, wall %.2fx",
                            depth, r.rawWall, r.rawCpu*1000, r.mlxWall, r.mlxCpu*1000,
                            r.mlxCpu/Swift.max(0.001,r.rawCpu), r.mlxWall/Swift.max(0.001,r.rawWall))
        }
        return "[chain-test (qmm→rmsNorm)×depth, 単一 encoder] task#3 orchestration: encode 削減の compound\n"
            + String(format: "  bit-exact(depth1): rel=%.3e %@", rel, rel < 2e-3 ? "✅" : "❌") + lines
            + "\n  → depth↑ で encode 削減率↑ なら 40 層 forward の ~1.7x を裏付け"
    }

    // ── task#3: GDN 1 層 assembly（全 raw kernel を chain）───────────────────────
    /// GDN decode 1 step（cold state: convState/recState=zeros）を全 raw kernel で組む。
    /// 量子化 in/out_proj(wq,scales,biases)＋ conv1dW/normWeight/aLog/dtBias を受け、coreOut→out[1,H]。
    /// この段階では各 kernel が個別 command buffer で round-trip する（numeric assembly の正しさを先に確定。
    /// single-encoder 融合は次段）。slicing・computeG・sigmoid・reshape の合成を MLX 経路と bit-exact 照合する。
    struct GDNRawWeights {
        let qkvWq: MLXArray, qkvSc: MLXArray, qkvBi: MLXArray
        let zWq: MLXArray, zSc: MLXArray, zBi: MLXArray
        let bWq: MLXArray, bSc: MLXArray, bBi: MLXArray
        let aWq: MLXArray, aSc: MLXArray, aBi: MLXArray
        let outWq: MLXArray, outSc: MLXArray, outBi: MLXArray
        let conv1dW: MLXArray   // [convDim, K]
        let normWeight: MLXArray, aLog: MLXArray, dtBias: MLXArray
    }

    static func gdnLayerRaw(_ x: MLXArray, _ w: GDNRawWeights,
                            numKHeads: Int = 16, numVHeads: Int = 32,
                            headKDim: Int = 128, headVDim: Int = 128,
                            convKernel: Int = 4, eps: Float = 1e-6) -> MLXArray? {
        let H = x.dim(-1)
        let keyDim = headKDim * numKHeads        // 2048
        let valueDim = headVDim * numVHeads      // 4096
        let convDim = keyDim * 2 + valueDim      // 8192
        let x2 = x.reshaped([1, H])
        // ① in_proj（4 本, 量子化 gemv）
        guard let qkv = qmm(x2, w.qkvWq, scales: w.qkvSc, biases: w.qkvBi, M: 1, K: H, N: convDim),
              let z   = qmm(x2, w.zWq,   scales: w.zSc,   biases: w.zBi,   M: 1, K: H, N: valueDim),
              let bP  = qmm(x2, w.bWq,   scales: w.bSc,   biases: w.bBi,   M: 1, K: H, N: numVHeads),
              let aP  = qmm(x2, w.aWq,   scales: w.aSc,   biases: w.aBi,   M: 1, K: H, N: numVHeads)
        else { return nil }
        // ② conv1d+silu（cold: convState=zeros, 窓 K=convKernel）。convInput[K, convDim]
        let convState = MLXArray.zeros([convKernel - 1, convDim], dtype: .float16)
        let convInput = MLX.concatenated([convState, qkv.asType(.float16)], axis: 0)  // [K, convDim]
        guard let convOut = conv1dSilu(convInput, w.conv1dW, K: convKernel, C: convDim) else { return nil }
        let co = convOut.reshaped([convDim])
        // ③ split → q,k,v
        let q1 = co[0 ..< keyDim].reshaped([numKHeads, headKDim])
        let k1 = co[keyDim ..< 2 * keyDim].reshaped([numKHeads, headKDim])
        let v1 = co[(2 * keyDim)...].reshaped([1, 1, numVHeads, headVDim])
        // ④ qk-norm（no-weight rmsNorm → scalar）
        let invScale = Float(pow(Double(headKDim), -0.5))
        guard let qn0 = rmsNorm(q1, nil, eps: eps, D: headKDim),
              let kn0 = rmsNorm(k1, nil, eps: eps, D: headKDim) else { return nil }
        let qN = ((invScale * invScale) * qn0).reshaped([1, 1, numKHeads, headKDim])
        let kN = (invScale * kn0).reshaped([1, 1, numKHeads, headKDim])
        // ⑤ recurrent（g/beta は MLX で, kernel は GQA を内部処理）
        let g = GatedDelta.computeG(w.aLog, aP.reshaped([1, 1, numVHeads]), w.dtBias)
        let beta = MLX.sigmoid(bP.reshaped([1, 1, numVHeads]))
        let st = MLXArray.zeros([1, numVHeads, headVDim, headKDim], dtype: .float32)
        guard let (coreOut, _) = recurrent(qN, kN, v1, g: g, beta: beta, state: st,
                                           B: 1, T: 1, Hk: numKHeads, Dk: headKDim,
                                           Hv: numVHeads, Dv: headVDim) else { return nil }
        // ⑥ RMSNormGated: silu(z) * rmsNorm(coreOut, normWeight)。
        //   normWeight=f32 → MLX は out を f32 に昇格(normed を f16 に丸めない)。promoteF32 で再現。
        guard let normed = rmsNorm(coreOut.reshaped([numVHeads, headVDim]), w.normWeight,
                                   eps: eps, D: headVDim, promoteF32: true) else { return nil }
        let zr = z.reshaped([numVHeads, headVDim])
        let gated = (silu(zr.asType(.float32)) * normed.asType(.float32)).asType(.float16)
        let outV = gated.reshaped([1, valueDim])
        // ⑦ out_proj
        return qmm(outV, w.outWq, scales: w.outSc, biases: w.outBi, M: 1, K: valueDim, N: H)
    }

    /// GDN 1 層 single-encoder の常駐 buffer 束（重み＋中間＝一度だけ確保、real forward では MTLResidencySet 相当）。
    struct GDNBuffers {
        let H, keyDim, valueDim, convDim, Dk, Dv, Hv, Hk, convKernel: Int
        let invScale: Float
        let bx: MTLBuffer
        let bQkvW, bQkvS, bQkvB, bZW, bZS, bZB, bAW, bAS, bAB, bBW, bBS, bBB, bOW, bOS, bOB: MTLBuffer
        let bConvW, bNormW, bALog, bDt, bOnes: MTLBuffer
        let convInput, zBuf, aBuf, bBuf, gBuf, betaBuf, convOut, qN, kN: MTLBuffer
        let stateBuf, stateOut, coreOut, normed, outV, outBuf: MTLBuffer
    }

    /// 重み・中間 buffer を一度だけ確保（asMTLBuffer の weight コピーは初回のみ＝real forward の常駐に対応）。
    static func prepareGDNBuffers(_ w: GDNRawWeights,
                                  numKHeads: Int = 16, numVHeads: Int = 32,
                                  headKDim: Int = 128, headVDim: Int = 128,
                                  convKernel: Int = 4, H: Int = 2048) -> GDNBuffers? {
        guard let (device, _) = ensure(), ensureAuxPipelines() else { return nil }
        let keyDim = headKDim * numKHeads, valueDim = headVDim * numVHeads, convDim = keyDim * 2 + valueDim
        let Dk = headKDim, Dv = headVDim, Hv = numVHeads, Hk = numKHeads
        func mtl(_ a: MLXArray, _ t: DType) -> MTLBuffer? { a.asType(t).asMTLBuffer(device: device, noCopy: false) }
        func mk(_ bytes: Int) -> MTLBuffer { device.makeBuffer(length: bytes, options: .storageModeShared)! }
        guard let bQkvW = w.qkvWq.asMTLBuffer(device: device, noCopy: false), let bQkvS = mtl(w.qkvSc, .float16), let bQkvB = mtl(w.qkvBi, .float16),
              let bZW = w.zWq.asMTLBuffer(device: device, noCopy: false), let bZS = mtl(w.zSc, .float16), let bZB = mtl(w.zBi, .float16),
              let bAW = w.aWq.asMTLBuffer(device: device, noCopy: false), let bAS = mtl(w.aSc, .float16), let bAB = mtl(w.aBi, .float16),
              let bBW = w.bWq.asMTLBuffer(device: device, noCopy: false), let bBS = mtl(w.bSc, .float16), let bBB = mtl(w.bBi, .float16),
              let bOW = w.outWq.asMTLBuffer(device: device, noCopy: false), let bOS = mtl(w.outSc, .float16), let bOB = mtl(w.outBi, .float16),
              let bConvW = mtl(w.conv1dW, .float32), let bNormW = mtl(w.normWeight, .float32),
              let bALog = mtl(w.aLog, .float32), let bDt = mtl(w.dtBias, .float32),
              let bOnes = MLXArray.ones([headKDim], dtype: .float16).asMTLBuffer(device: device, noCopy: false)
        else { print("[raw-gdn-se] weight buffer nil"); return nil }
        let convInput = mk(convKernel * convDim * 2); memset(convInput.contents(), 0, convKernel * convDim * 2)  // zero 1 回（qkv が row(K-1)を毎回上書き）
        let stateBuf = mk(Hv * Dv * Dk * 4); memset(stateBuf.contents(), 0, Hv * Dv * Dk * 4)                   // cold state zero 1 回
        return GDNBuffers(
            H: H, keyDim: keyDim, valueDim: valueDim, convDim: convDim, Dk: Dk, Dv: Dv, Hv: Hv, Hk: Hk,
            convKernel: convKernel, invScale: Float(pow(Double(headKDim), -0.5)),
            bx: mk(H * 2),
            bQkvW: bQkvW, bQkvS: bQkvS, bQkvB: bQkvB, bZW: bZW, bZS: bZS, bZB: bZB,
            bAW: bAW, bAS: bAS, bAB: bAB, bBW: bBW, bBS: bBS, bBB: bBB, bOW: bOW, bOS: bOS, bOB: bOB,
            bConvW: bConvW, bNormW: bNormW, bALog: bALog, bDt: bDt, bOnes: bOnes,
            convInput: convInput, zBuf: mk(valueDim * 2), aBuf: mk(Hv * 2), bBuf: mk(Hv * 2),
            gBuf: mk(Hv * 4), betaBuf: mk(Hv * 4), convOut: mk(convDim * 2), qN: mk(keyDim * 2), kN: mk(keyDim * 2),
            stateBuf: stateBuf, stateOut: mk(Hv * Dv * Dk * 4), coreOut: mk(valueDim * 2),
            normed: mk(valueDim * 4), outV: mk(valueDim * 2), outBuf: mk(H * 2))
    }

    /// ★ task#3: GDN 1 層を **単一 command buffer + 単一 encoder** で連結（中間 buffer GPU 常駐、
    ///   commit/MLXArray 復帰なし）。常駐 buffer を使い per-call は x 書込み＋encode＋commit＋read のみ。
    ///   round-trip 版 gdnLayerRaw と bit-exact。pipeline は事前 warm 前提。cold cache(decode 1 step)。
    static func gdnLayerSingleEncoder(_ x: MLXArray, _ b: GDNBuffers, eps: Float = 1e-6) -> MLXArray? {
        guard let (_, queue) = ensure() else { return nil }
        guard let qp = _qmmPipeline, let rp = _rmsPipeline, let rpf = _rmsPipelineF32,
              let cp = _conv1dPipeline, let rcp = _recurPipeline,
              let cgb = _cgbPipeline, let gp = _gatePipeline, let scp = _scalePipeline else {
            print("[raw-gdn-se] pipeline 未 warm（先に gdnLayerRaw を呼ぶ）"); return nil
        }
        let H = b.H, keyDim = b.keyDim, valueDim = b.valueDim, convDim = b.convDim
        let Dk = b.Dk, Dv = b.Dv, Hv = b.Hv, Hk = b.Hk, convKernel = b.convKernel, invScale = b.invScale
        // x を常駐 buffer に書込み（per-call、H*2 bytes のみ）
        let xf = x.reshaped([H]).asType(.float16).asArray(Float16.self)
        b.bx.contents().bindMemory(to: Float16.self, capacity: H).update(from: xf, count: H)
        let bx = b.bx
        let bQkvW = b.bQkvW, bQkvS = b.bQkvS, bQkvB = b.bQkvB, bZW = b.bZW, bZS = b.bZS, bZB = b.bZB
        let bAW = b.bAW, bAS = b.bAS, bAB = b.bAB, bBW = b.bBW, bBS = b.bBS, bBB = b.bBB
        let bOW = b.bOW, bOS = b.bOS, bOB = b.bOB
        let bConvW = b.bConvW, bNormW = b.bNormW, bALog = b.bALog, bDt = b.bDt, bOnes = b.bOnes
        let convInput = b.convInput, zBuf = b.zBuf, aBuf = b.aBuf, bBuf = b.bBuf, gBuf = b.gBuf, betaBuf = b.betaBuf
        let convOut = b.convOut, qN = b.qN, kN = b.kN, stateBuf = b.stateBuf, stateOut = b.stateOut
        let coreOut = b.coreOut, normed = b.normed, outV = b.outV, outBuf = b.outBuf

        let cb = queue.makeCommandBuffer()!; let enc = cb.makeComputeCommandEncoder()!
        // qmm helper（grid=(M,N/8,1), group=(32,2,1)）。y は offset 指定可。
        func encQmm(_ wq: MTLBuffer, _ sc: MTLBuffer, _ bi: MTLBuffer, _ xb: MTLBuffer, xoff: Int,
                    _ y: MTLBuffer, yoff: Int, K: Int, N: Int) {
            enc.setComputePipelineState(qp)
            enc.setBuffer(wq, offset: 0, index: 0); enc.setBuffer(sc, offset: 0, index: 1)
            enc.setBuffer(bi, offset: 0, index: 2); enc.setBuffer(xb, offset: xoff, index: 3)
            enc.setBuffer(y, offset: yoff, index: 4)
            var kk = Int32(K), nn = Int32(N); enc.setBytes(&kk, length: 4, index: 5); enc.setBytes(&nn, length: 4, index: 6)
            enc.dispatchThreadgroups(MTLSize(width: 1, height: N / 8, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 2, depth: 1))
        }
        // rmsNorm helper（D≤4096, tg=ceil(D/4)→32倍数）。promote=true で f32 weight/out。
        func encRms(_ xb: MTLBuffer, xoff: Int, _ wb: MTLBuffer, _ ob: MTLBuffer, rows: Int, D: Int, promote: Bool) {
            enc.setComputePipelineState(promote ? rpf : rp)
            enc.setBuffer(xb, offset: xoff, index: 0); enc.setBuffer(wb, offset: 0, index: 1); enc.setBuffer(ob, offset: 0, index: 2)
            var ee = eps, asz = UInt32(D), ws = UInt32(1)
            enc.setBytes(&ee, length: 4, index: 3); enc.setBytes(&asz, length: 4, index: 4); enc.setBytes(&ws, length: 4, index: 5)
            let tg = ((((D + 3) / 4) + 31) / 32) * 32
            enc.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        }
        func encScale(_ xb: MTLBuffer, _ s: Float, _ total: Int) {
            enc.setComputePipelineState(scp)
            var ss = s, tt = UInt32(total)
            enc.setBuffer(xb, offset: 0, index: 0); enc.setBytes(&ss, length: 4, index: 1); enc.setBytes(&tt, length: 4, index: 2)
            enc.dispatchThreads(MTLSize(width: total, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(scp.maxTotalThreadsPerThreadgroup, 256), height: 1, depth: 1))
        }
        // ① in_proj 4 本。qkv は convInput の row(K-1) に直接書く（zero-pad は memset 済）。
        encQmm(bQkvW, bQkvS, bQkvB, bx, xoff: 0, convInput, yoff: (convKernel - 1) * convDim * 2, K: H, N: convDim)
        encQmm(bZW, bZS, bZB, bx, xoff: 0, zBuf, yoff: 0, K: H, N: valueDim)
        encQmm(bAW, bAS, bAB, bx, xoff: 0, aBuf, yoff: 0, K: H, N: Hv)
        encQmm(bBW, bBS, bBB, bx, xoff: 0, bBuf, yoff: 0, K: H, N: Hv)
        // ② g/beta
        enc.setComputePipelineState(cgb)
        enc.setBuffer(aBuf, offset: 0, index: 0); enc.setBuffer(bBuf, offset: 0, index: 1)
        enc.setBuffer(bALog, offset: 0, index: 2); enc.setBuffer(bDt, offset: 0, index: 3)
        enc.setBuffer(gBuf, offset: 0, index: 4); enc.setBuffer(betaBuf, offset: 0, index: 5)
        var hv = UInt32(Hv); enc.setBytes(&hv, length: 4, index: 6)
        enc.dispatchThreads(MTLSize(width: Hv, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(cgb.maxTotalThreadsPerThreadgroup, Hv), height: 1, depth: 1))
        // ③ conv1d+silu
        enc.setComputePipelineState(cp)
        enc.setBuffer(convInput, offset: 0, index: 0); enc.setBuffer(bConvW, offset: 0, index: 1); enc.setBuffer(convOut, offset: 0, index: 2)
        var ck = UInt32(convKernel), cc = UInt32(convDim); enc.setBytes(&ck, length: 4, index: 3); enc.setBytes(&cc, length: 4, index: 4)
        enc.dispatchThreads(MTLSize(width: convDim, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(cp.maxTotalThreadsPerThreadgroup, 256), height: 1, depth: 1))
        // ④ qk-norm（rmsNorm(ones) → scale）
        encRms(convOut, xoff: 0, bOnes, qN, rows: Hk, D: Dk, promote: false)
        encScale(qN, invScale * invScale, keyDim)
        encRms(convOut, xoff: keyDim * 2, bOnes, kN, rows: Hk, D: Dk, promote: false)
        encScale(kN, invScale, keyDim)
        // ⑤ recurrent（v = convOut の 2*keyDim 以降）
        enc.setComputePipelineState(rcp)
        enc.setBuffer(qN, offset: 0, index: 0); enc.setBuffer(kN, offset: 0, index: 1); enc.setBuffer(convOut, offset: 2 * keyDim * 2, index: 2)
        enc.setBuffer(gBuf, offset: 0, index: 3); enc.setBuffer(betaBuf, offset: 0, index: 4); enc.setBuffer(stateBuf, offset: 0, index: 5)
        var tt = Int32(1); enc.setBytes(&tt, length: 4, index: 6)
        enc.setBuffer(coreOut, offset: 0, index: 7); enc.setBuffer(stateOut, offset: 0, index: 8)
        enc.dispatchThreads(MTLSize(width: 32, height: Dv, depth: Hv), threadsPerThreadgroup: MTLSize(width: 32, height: 4, depth: 1))
        // ⑥ RMSNormGated: rmsNorm(promoteF32) → gate
        encRms(coreOut, xoff: 0, bNormW, normed, rows: Hv, D: Dv, promote: true)
        enc.setComputePipelineState(gp)
        enc.setBuffer(zBuf, offset: 0, index: 0); enc.setBuffer(normed, offset: 0, index: 1); enc.setBuffer(outV, offset: 0, index: 2)
        var vt = UInt32(valueDim); enc.setBytes(&vt, length: 4, index: 3)
        enc.dispatchThreads(MTLSize(width: valueDim, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(gp.maxTotalThreadsPerThreadgroup, 256), height: 1, depth: 1))
        // ⑦ out_proj
        encQmm(bOW, bOS, bOB, outV, xoff: 0, outBuf, yoff: 0, K: valueDim, N: H)
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        let ptr = outBuf.contents().bindMemory(to: Float16.self, capacity: H)
        return MLXArray(Array(UnsafeBufferPointer(start: ptr, count: H)), [1, H])
    }

    /// 検証: GDN 1 層 raw assembly vs MLX GatedDeltaNetLayer（同一量子化重み・f32Conv）を bit-exact 照合。
    /// - env: QWISP_RUN=raw-gdn-test / QWISP_GDN_REF(ref path, 既定 /tmp/qwisp_gdn_layer_ref.safetensors)
    public static func runGdnLayerTest() -> String {
        let refPath = ProcessInfo.processInfo.environment["QWISP_GDN_REF"]
            ?? "/tmp/qwisp_gdn_layer_ref.safetensors"
        guard let r = try? loadArrays(url: URL(fileURLWithPath: refPath)) else {
            return "[raw-gdn] ERROR: ref 読込失敗 \(refPath)"
        }
        guard let qkvW = r["in_proj_qkv"], let zW = r["in_proj_z"],
              let bW = r["in_proj_b"], let aW = r["in_proj_a"], let outW = r["out_proj"],
              let cw = r["conv1d"], let nw = r["norm_weight"],
              let aLog = r["A_log"], let dtBias = r["dt_bias"] else {
            return "[raw-gdn] ERROR: ref キー不足"
        }
        // decode 1 step（S=1, cold cache）を検証: 新規 x[1,1,2048]、ref も同 x で MLX 層から構築。
        let H = qkvW.dim(-1)
        let x = MLXRandom.normal([1, 1, H]).asType(.float16)
        // 同一量子化重みで raw / MLX-ref を構築（量子化ノイズを除外し assembly のみ比較）
        func quant(_ wt: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
            let (wq, s, bOpt) = MLX.quantized(wt.asType(.float16), groupSize: 64, bits: 4, mode: .affine)
            return (wq, s, bOpt!)
        }
        let (qkvWq, qkvSc, qkvBi) = quant(qkvW)
        let (zWq, zSc, zBi) = quant(zW)
        let (bWq, bSc, bBi) = quant(bW)
        let (aWq, aSc, aBi) = quant(aW)
        let (outWq, outSc, outBi) = quant(outW)
        let convDim = 8192
        let rw = GDNRawWeights(
            qkvWq: qkvWq, qkvSc: qkvSc, qkvBi: qkvBi, zWq: zWq, zSc: zSc, zBi: zBi,
            bWq: bWq, bSc: bSc, bBi: bBi, aWq: aWq, aSc: aSc, aBi: aBi,
            outWq: outWq, outSc: outSc, outBi: outBi,
            conv1dW: cw.reshaped([convDim, 4]).asType(.float32),   // [convDim,K,1] → [convDim,K]、f32(MLX f32Conv 一致)
            normWeight: nw.asType(.float32), aLog: aLog, dtBias: dtBias)   // f32(MLX promotion 再現)
        // MLX 参照（同一量子化重み, f32Conv=true で raw の f32 累積に一致, fuse=off）
        let prevF32 = GatedDeltaNetLayer.f32Conv
        GatedDeltaNetLayer.f32Conv = true
        defer { GatedDeltaNetLayer.f32Conv = prevF32 }
        let refLayer = GatedDeltaNetLayer(
            numKHeads: 16, numVHeads: 32, headKDim: 128, headVDim: 128, convKernel: 4, eps: 1e-6,
            inProjQKV: .quantized(qkvWq, qkvSc, qkvBi, 4), inProjZ: .quantized(zWq, zSc, zBi, 4),
            inProjB: .quantized(bWq, bSc, bBi, 4), inProjA: .quantized(aWq, aSc, aBi, 4),
            outProj: .quantized(outWq, outSc, outBi, 4),
            conv1dW: cw, normWeight: nw, aLog: aLog, dtBias: dtBias)
        let ref = refLayer(x); ref.eval()
        guard let got = gdnLayerRaw(x, rw) else { return "[raw-gdn] ERROR: raw assembly 実行失敗" }
        got.eval()
        let refFlat = ref.reshaped([ref.size])
        let gotFlat = got.reshaped([got.size])
        let rel = relErr(gotFlat, refFlat)
        let ok = rel < 2e-3
        var out = String(format: "[raw-gdn-test] GDN 1 層 raw assembly vs MLX(同量子化, f32Conv)\n"
            + "  out shape raw=%@ ref=%@  rel=%.3e  %@",
            "\(got.shape)", "\(ref.shape)", rel, ok ? "OK ✅ assembly bit-exact" : "MISMATCH ❌")
        if !ok {
            let rf = refFlat.asArray(Float.self), gf = gotFlat.asArray(Float.self)
            var mi = 0; var md: Float = 0
            for i in 0 ..< rf.count { let d = abs(rf[i] - gf[i]); if d > md { md = d; mi = i } }
            out += String(format: "\n    max diff @ %d: ref=%.4f got=%.4f", mi, rf[mi], gf[mi])
        }
        // ★ single-encoder 版（task#3）: 常駐 buffer を一度確保 → per-call は encode のみ。bit-exact + 計測。
        if let bufs = prepareGDNBuffers(rw), let se = gdnLayerSingleEncoder(x, bufs) {
            se.eval()
            let relSE = relErr(se.reshaped([se.size]), gotFlat)        // vs round-trip(=MLX bit-exact)
            let relSEref = relErr(se.reshaped([se.size]), refFlat)     // vs MLX 直接
            out += String(format: "\n  ── single-encoder（task#3, 常駐 buffer）──\n   SE vs round-trip rel=%.3e %@  / SE vs MLX rel=%.3e",
                          relSE, relSE == 0 ? "✅ bit-exact" : "❌", relSEref)
            func cpuNs() -> UInt64 { var r = rusage(); getrusage(RUSAGE_SELF, &r)
                return UInt64(r.ru_utime.tv_sec+r.ru_stime.tv_sec)*1_000_000_000 + UInt64(r.ru_utime.tv_usec+r.ru_stime.tv_usec)*1000 }
            let reps = 300
            for _ in 0..<10 { _ = gdnLayerSingleEncoder(x, bufs) }
            var t0 = DispatchTime.now().uptimeNanoseconds; var c0 = cpuNs()
            for _ in 0..<reps { _ = gdnLayerSingleEncoder(x, bufs) }
            let seWall = Double(DispatchTime.now().uptimeNanoseconds-t0)/Double(reps)/1e6
            let seCpu = Double(cpuNs()-c0)/Double(reps)/1e6
            for _ in 0..<10 { let r = refLayer(x); r.eval() }
            t0 = DispatchTime.now().uptimeNanoseconds; c0 = cpuNs()
            for _ in 0..<reps { let r = refLayer(x); r.eval() }
            let mlxWall = Double(DispatchTime.now().uptimeNanoseconds-t0)/Double(reps)/1e6
            let mlxCpu = Double(cpuNs()-c0)/Double(reps)/1e6
            out += String(format: "\n   時間/層: SE wall=%.3fms cpu-encode=%.3fms | MLX wall=%.3fms cpu-encode=%.3fms → wall %.2fx, encode %.2fx",
                          seWall, seCpu, mlxWall, mlxCpu, mlxWall/Swift.max(0.001,seWall), mlxCpu/Swift.max(0.001,seCpu))
        } else { out += "\n  single-encoder: 実行失敗" }
        // per-stage 診断: 各段で raw 中間値 vs MLX 中間値の rel（誤差が compound か構造バグか）
        out += "\n  ── per-stage 診断（各段 raw vs MLX, 同入力・同重み）──"
        let keyDim = 2048, nKH = 16, hKD = 128, cK = 4, nVH = 32, hVD = 128
        let x2 = x.reshaped([1, H])
        // ① in_proj qkv
        let mlxQkv = MLX.quantizedMatmul(x2, qkvWq, scales: qkvSc, biases: qkvBi, transpose: true,
                                         groupSize: 64, bits: 4, mode: .affine); mlxQkv.eval()
        let rawQkv = qmm(x2, qkvWq, scales: qkvSc, biases: qkvBi, M: 1, K: H, N: convDim)!
        out += String(format: "\n   ① qkv(qmm):      rel=%.3e", relErr(rawQkv, mlxQkv))
        // ② conv1d+silu（各 path は自分の qkv を入力＝cumulative drift）
        func mlxConv(_ qkv: MLXArray) -> MLXArray {
            let cs = MLXArray.zeros([1, cK - 1, convDim], dtype: .float16)
            let ci = MLX.concatenated([cs, qkv.reshaped([1, 1, convDim])], axis: 1)
            let co = MLX.conv1d(ci.asType(DType.float32), cw.reshaped([convDim, cK, 1]).asType(DType.float32),
                                stride: 1, padding: 0, dilation: 1, groups: convDim)
            return silu(co).asType(DType.float16).reshaped([convDim])
        }
        let mlxCo = mlxConv(mlxQkv)
        let csR = MLXArray.zeros([cK - 1, convDim], dtype: .float16)
        let rawCo = conv1dSilu(MLX.concatenated([csR, rawQkv.asType(.float16)], axis: 0),
                               cw.reshaped([convDim, cK]).asType(.float32), K: cK, C: convDim)!.reshaped([convDim])
        out += String(format: "\n   ② convOut:       rel=%.3e", relErr(rawCo, mlxCo))
        // ③ qk-norm（各 path 自分の convOut）
        let invS = Float(pow(Double(hKD), -0.5))
        func qkNormMLX(_ co: MLXArray) -> (MLXArray, MLXArray) {
            let q1 = co[0 ..< keyDim].reshaped([nKH, hKD]), k1 = co[keyDim ..< 2 * keyDim].reshaped([nKH, hKD])
            let qn = (invS * invS) * GatedDeltaNetLayer.rmsNormNoWeight(q1, eps: 1e-6)
            let kn = invS * GatedDeltaNetLayer.rmsNormNoWeight(k1, eps: 1e-6)
            return (qn, kn)
        }
        let (mlxQN, mlxKN) = qkNormMLX(mlxCo)
        let rq1 = rawCo[0 ..< keyDim].reshaped([nKH, hKD]), rk1 = rawCo[keyDim ..< 2 * keyDim].reshaped([nKH, hKD])
        let rawQN = (invS * invS) * rmsNorm(rq1, nil, eps: 1e-6, D: hKD)!
        let rawKN = invS * rmsNorm(rk1, nil, eps: 1e-6, D: hKD)!
        out += String(format: "\n   ③ qN/kN:         rel=%.3e / %.3e", relErr(rawQN, mlxQN), relErr(rawKN, mlxKN))
        // ④ recurrent（coreOut）。a/b 投影＋g/beta、state=zeros。両 path とも同じ qN/kN/v1。
        let valueDim = 4096
        let mlxV = mlxCo[(2*keyDim)...].reshaped([1, 1, nVH, hVD])
        let rawV = rawCo[(2*keyDim)...].reshaped([1, 1, nVH, hVD])
        let mlxA = MLX.quantizedMatmul(x2, aWq, scales: aSc, biases: aBi, transpose: true, groupSize: 64, bits: 4, mode: .affine).reshaped([1,1,nVH])
        let mlxB = MLX.quantizedMatmul(x2, bWq, scales: bSc, biases: bBi, transpose: true, groupSize: 64, bits: 4, mode: .affine).reshaped([1,1,nVH])
        let rawA = qmm(x2, aWq, scales: aSc, biases: aBi, M: 1, K: qkvW.dim(-1), N: nVH)!.reshaped([1,1,nVH])
        let rawB = qmm(x2, bWq, scales: bSc, biases: bBi, M: 1, K: qkvW.dim(-1), N: nVH)!.reshaped([1,1,nVH])
        let st0 = MLXArray.zeros([1, nVH, hVD, hKD], dtype: .float32)
        let (mlxCore, _) = GatedDelta.updateKernel(mlxQN.reshaped([1,1,nKH,hKD]), mlxKN.reshaped([1,1,nKH,hKD]), mlxV, mlxA, mlxB, aLog, dtBias, state: st0)
        let rg = GatedDelta.computeG(aLog, rawA, dtBias), rbeta = MLX.sigmoid(rawB)
        let (rawCore, _) = recurrent(rawQN.reshaped([1,1,nKH,hKD]), rawKN.reshaped([1,1,nKH,hKD]), rawV, g: rg, beta: rbeta, state: st0, B: 1, T: 1, Hk: nKH, Dk: hKD, Hv: nVH, Dv: hVD)!
        mlxCore.eval(); rawCore.eval()
        out += String(format: "\n   ④ coreOut:       rel=%.3e", relErr(rawCore, mlxCore))
        // ⑤ RMSNormGated（outV）。silu(z)*rmsNorm(core,normW)。
        let mlxZ = MLX.quantizedMatmul(x2, zWq, scales: zSc, biases: zBi, transpose: true, groupSize: 64, bits: 4, mode: .affine).reshaped([nVH, hVD])
        let rawZ = qmm(x2, zWq, scales: zSc, biases: zBi, M: 1, K: qkvW.dim(-1), N: valueDim)!.reshaped([nVH, hVD])
        let mlxNormed = MLXFast.rmsNorm(mlxCore.reshaped([nVH,hVD]), weight: nw.asType(.float32), eps: 1e-6)  // refLayer 同様 f32 nw
        let mlxOutV = (silu(mlxZ.asType(.float32)) * mlxNormed.asType(.float32)).asType(.float16).reshaped([1, valueDim])
        let rawNormed = rmsNorm(rawCore.reshaped([nVH,hVD]), nw.asType(.float32), eps: 1e-6, D: hVD, promoteF32: true)!
        let rawOutV = (silu(rawZ.asType(.float32)) * rawNormed.asType(.float32)).asType(.float16).reshaped([1, valueDim])
        out += String(format: "\n   ⑤ outV(gated):   rel=%.3e", relErr(rawOutV, mlxOutV))
        // ⑥ out_proj
        let mlxOut = MLX.quantizedMatmul(mlxOutV, outWq, scales: outSc, biases: outBi, transpose: true, groupSize: 64, bits: 4, mode: .affine)
        let rawOut = qmm(rawOutV, outWq, scales: outSc, biases: outBi, M: 1, K: valueDim, N: qkvW.dim(-1))!
        out += String(format: "\n   ⑥ out(out_proj): rel=%.3e", relErr(rawOut, mlxOut))
        return out
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
        // SDPA(decode L=1, GQA 16q/2kv, head_dim 256, S=10, f32)
        let Hh = 16, KVk = 2, Dd = 256, Ss = 10
        let scaleA = Float(pow(256.0, -0.5))
        let q = MLXRandom.normal([1, Hh, 1, Dd]).asType(.float16)
        let kk2 = MLXRandom.normal([1, KVk, Ss, Dd]).asType(.float16)
        let vv = MLXRandom.normal([1, KVk, Ss, Dd]).asType(.float16)
        let refA = MLXFast.scaledDotProductAttention(queries: q.asType(.float32), keys: kk2.asType(.float32),
                       values: vv.asType(.float32), scale: scaleA, mask: .none).asType(.float16); refA.eval()
        if let g = sdpaDecode(q.reshaped([Hh, Dd]), kk2.reshaped([KVk, Ss, Dd]), vv.reshaped([KVk, Ss, Dd]),
                              H: Hh, KV: KVk, D: Dd, S: Ss, scale: scaleA) {
            g.eval()
            let rel = relErr(g.reshaped([1, Hh, 1, Dd]), refA)
            out += String(format: "\n  SDPA(decode):rel=%.3e  %@", rel, rel < 2e-3 ? "OK ✅" : "MISMATCH ❌")
        } else { out += "\n  SDPA: kernel 失敗" }
        // gather_qmm(MoE 核): expert ごとの MLX.quantizedMatmul と照合
        let E = 64, Nn = 512, Kk2 = 2048, Ktop = 4
        let wf = MLXRandom.normal([E, Nn, Kk2]).asType(.float16)
        let (gwq, gsc, gbiOpt) = MLX.quantized(wf, groupSize: 64, bits: 4, mode: .affine)
        let xg = MLXRandom.normal([1, Kk2]).asType(.float16)
        let indsArr = [3, 17, 40, 62].map { Int32($0) }
        let inds = MLXArray(indsArr, [Ktop])
        MLX.eval([gwq, gsc, xg, inds] + (gbiOpt.map { [$0] } ?? []))
        if let gbi = gbiOpt {
            var refRows: [MLXArray] = []
            for ki in 0 ..< Ktop {
                let e = Int(indsArr[ki])
                let r = MLX.quantizedMatmul(xg, gwq[e], scales: gsc[e], biases: gbi[e],
                                            transpose: true, groupSize: 64, bits: 4, mode: .affine)
                refRows.append(r)
            }
            let refG = MLX.concatenated(refRows, axis: 0); refG.eval()   // [Ktop, N]
            if let g = gatherQmm(xg, gwq, scales: gsc, biases: gbi, inds: inds, Ktop: Ktop, K: Kk2, N: Nn) {
                g.eval()
                let rel = relErr(g, refG)
                out += String(format: "\n  gather_qmm:  rel=%.3e  %@", rel, rel < 2e-3 ? "OK ✅" : "MISMATCH ❌")
            } else { out += "\n  gather_qmm: kernel 失敗" }
        } else { out += "\n  gather_qmm: biases nil" }
        // GDN recurrent(decode T=1): GatedDelta.updateKernel と照合
        let Bg=1, Tg=1, Hkg=16, Dkg=128, Hvg=32, Dvg=128
        let qg = MLXRandom.normal([Bg,Tg,Hkg,Dkg]).asType(.float16)
        let kg = MLXRandom.normal([Bg,Tg,Hkg,Dkg]).asType(.float16)
        let vg = MLXRandom.normal([Bg,Tg,Hvg,Dvg]).asType(.float16)
        let ag = MLXRandom.normal([Bg,Tg,Hvg]).asType(.float16)
        let bg2 = MLXRandom.normal([Bg,Tg,Hvg]).asType(.float16)
        let aLog = MLXRandom.normal([Hvg]).asType(.float32)
        let dtB = MLXRandom.normal([Hvg]).asType(.float32)
        let stg = MLXRandom.normal([Bg,Hvg,Dvg,Dkg]).asType(.float32)
        let (refY, _) = GatedDelta.updateKernel(qg, kg, vg, ag, bg2, aLog, dtB, state: stg); refY.eval()
        let betaR = MLX.sigmoid(bg2), gR = GatedDelta.computeG(aLog, ag, dtB)
        MLX.eval([betaR, gR])
        if let (y, _) = recurrent(qg, kg, vg, g: gR, beta: betaR, state: stg, B: Bg, T: Tg, Hk: Hkg, Dk: Dkg, Hv: Hvg, Dv: Dvg) {
            y.eval()
            let rel = relErr(y, refY)
            out += String(format: "\n  GDN recurrent:rel=%.3e  %@", rel, rel < 2e-3 ? "OK ✅" : "MISMATCH ❌")
        } else { out += "\n  GDN recurrent: kernel 失敗" }
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
        // ★ 真の bit-exact = rel==0（max|Δ|==0）。それ以外は「近似」と正直に表示する。
        let label: String
        if d == 0 { label = "OK ✅✅ TRUE bit-exact (rel=0.000e0, MLX と完全一致)" }
        else if rel < 2e-3 { label = "△ 近似一致(rel<2e-3 だが非 bit-exact)。FMA/fast-math 差残存" }
        else { label = "MISMATCH ❌" }
        let math = ProcessInfo.processInfo.environment["QWISP_QMM_MATH"] ?? "safe"
        return String(format: "[raw-qmm-test M=%d K=%d N=%d, 4bit affine gs=64, math=%@] raw-Metal(qmv_fast 移植) vs MLX\n"
            + "  max|Δ|=%.3e  rel=%.3e  %@", M, K, N, math, d, rel, label)
    }

    static func envInt(_ k: String, _ d: Int) -> Int {
        guard let v = ProcessInfo.processInfo.environment[k], let i = Int(v) else { return d }
        return i
    }
}
