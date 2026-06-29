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
    nonisolated(unsafe) static var metalRoute = false    // task#8 検証: MoE routing を Metal(qmm8+route_top8)で行う
    // ── profile（task#9 owner 指摘①: per-kernel GPU-exec 帰属）──
    nonisolated(unsafe) static var lastGPUExecMs = 0.0    // 直近 fusedForwardGPU の GPU-exec(gpuEnd-gpuStart)
    nonisolated(unsafe) static var profSkipSingleThread = false  // route_top8/shared_gate8 を skip(timing 用)
    nonisolated(unsafe) static var profSkipMoEExperts = false    // gather g/u/d/swiglu/shared を skip(timing 用)
    nonisolated(unsafe) static var profSkipMixer = false         // mixer body を skip(timing 用)
    nonisolated(unsafe) static var profSkipGDNMatmul = false      // GDN in_proj×4 + out_proj qmm を skip
    nonisolated(unsafe) static var profSkipGDNRecur = false       // GDN recurrent を skip
    nonisolated(unsafe) static var _qmmPipeline: MTLComputePipelineState?
    nonisolated(unsafe) static var _qmmF32Pipeline: MTLComputePipelineState?
    nonisolated(unsafe) static var _qmm8Pipeline: MTLComputePipelineState?     // 8bit qmv_fast（router gate）
    nonisolated(unsafe) static var _ropeF32Pipeline: MTLComputePipelineState?
    nonisolated(unsafe) static var _sdpaF32Pipeline: MTLComputePipelineState?
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
                    M: Int, K: Int, N: Int, bits: Int = 4, gs: Int = 64, xF32: Bool = false) -> MLXArray? {
        guard let (device, queue) = ensure() else { return nil }
        // ★ MLX の quantizedMatmul(qmv_fast) を数式・累積順・simd_sum まで完全一致で移植（rel 0.000e0 が目標）。
        //   bits=4/gs=64。fast 条件 N%8==0 && K%512==0。xF32=true で x/scales/biases/out を f32（attention f32 cascade,
        //   o_proj の入力が f32 になる経路。MLX も x f32 時は scales を f32 にして qmv_fast<float> を回す）。
        let fast = (N % 8 == 0) && (K % 512 == 0) && bits == 4 && gs == 64
        guard fast else { print("[raw-qmm] 非fast (N=\(N) K=\(K) bits=\(bits) gs=\(gs)) 未対応"); return nil }
        let XT = xF32 ? "float" : "half"
        let needCompile = xF32 ? (_qmmF32Pipeline == nil) : (_qmmPipeline == nil)
        if needCompile {
            let src = """
            #include <metal_stdlib>
            using namespace metal;
            #define SIMD_SIZE 32
            // MLX load_vector<bits=4>: x を 16^j で事前除算。sum の 4 要素加算は T 演算（XT, MLX と一致）→ float 昇格。
            inline float ld16(const device \(XT)* x, thread float* xt) {
                float sum = 0.0f;
                for (int i = 0; i < 16; i += 4) {
                    sum += x[i] + x[i+1] + x[i+2] + x[i+3];
                    xt[i]   = x[i];
                    xt[i+1] = x[i+1] / 16.0f;
                    xt[i+2] = x[i+2] / 256.0f;
                    xt[i+3] = x[i+3] / 4096.0f;
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
            kernel void qmm4(device const uint32_t* w      [[buffer(0)]],
                             device const \(XT)*    scales [[buffer(1)]],
                             device const \(XT)*    biases [[buffer(2)]],
                             device const \(XT)*    x      [[buffer(3)]],
                             device \(XT)*          y      [[buffer(4)]],
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
                        const device \(XT)* sl = scales + row * in_vec_size_g;
                        const device \(XT)* bl = biases + row * in_vec_size_g;
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
                    if (simd_lid == 0) y[row] = (\(XT))result[row];
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
                let p = try device.makeComputePipelineState(function: lib.makeFunction(name: "qmm4")!)
                if xF32 { _qmmF32Pipeline = p } else { _qmmPipeline = p }
            } catch { print("[raw-qmm] compile error: \(error)"); return nil }
        }
        let dt: DType = xF32 ? .float32 : .float16
        let elem = xF32 ? 4 : 2
        guard let bx = x.asType(dt).asMTLBuffer(device: device, noCopy: false),
              let bwq = wq.asMTLBuffer(device: device, noCopy: false),
              let bsc = scales.asType(dt).asMTLBuffer(device: device, noCopy: false),
              let bbi = biases.asType(dt).asMTLBuffer(device: device, noCopy: false)
        else { return nil }
        let outBuf = device.makeBuffer(length: M * N * elem, options: .storageModeShared)!
        let cb = queue.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(xF32 ? _qmmF32Pipeline! : _qmmPipeline!)
        enc.setBuffer(bwq, offset: 0, index: 0)
        enc.setBuffer(bsc, offset: 0, index: 1)
        enc.setBuffer(bbi, offset: 0, index: 2)
        enc.setBuffer(bx, offset: 0, index: 3)
        enc.setBuffer(outBuf, offset: 0, index: 4)
        var kk = Int32(K), nn = Int32(N)
        enc.setBytes(&kk, length: 4, index: 5)
        enc.setBytes(&nn, length: 4, index: 6)
        enc.dispatchThreadgroups(MTLSize(width: M, height: N / 8, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 32, height: 2, depth: 1))
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        if xF32 {
            let ptr = outBuf.contents().bindMemory(to: Float.self, capacity: M * N)
            return MLXArray(Array(UnsafeBufferPointer(start: ptr, count: M * N)), [M, N])
        }
        let ptr = outBuf.contents().bindMemory(to: Float16.self, capacity: M * N)
        return MLXArray(Array(UnsafeBufferPointer(start: ptr, count: M * N)), [M, N])
    }

    /// 8bit qmv_fast 移植（router gate: postNorm[H]→gate_logits[N], f16）。MLX qmv_fast<half,gs=64,bits=8>。
    /// bits=8 は load_vector/qdot が単純（x_thread=x そのまま、accum=Σ x·w_byte）。fast 条件 N%8==0 && K%512==0。
    /// 単一 encoder 融合用に encodeQmm8 も提供。
    static func compileQmm8() -> Bool {
        guard let (device, _) = ensure() else { return false }
        if _qmm8Pipeline != nil { return true }
        let src = """
        #include <metal_stdlib>
        using namespace metal;
        #define SIMD_SIZE 32
        // MLX load_vector<bits=8>: x_thread=x そのまま, sum=Σ x（昇格無し, 4bit の 16^j 除算は無し）。
        inline float ld8(const device half* x, thread float* xt) {
            float sum = 0.0f;
            for (int i = 0; i < 8; i++) { float v = x[i]; sum += v; xt[i] = v; }
            return sum;
        }
        // MLX qdot<bits=8>: accum = Σ_i xt[i]*w_byte[i]（i=0..7 順次）, return scale*accum + sum*bias。
        inline float qd8(const device uint8_t* w, const thread float* xt, float scale, float bias, float sum) {
            float accum = 0.0f;
            for (int i = 0; i < 8; i++) accum += xt[i] * (float)w[i];
            return scale * accum + sum * bias;
        }
        kernel void qmm8(device const uint32_t* w      [[buffer(0)]],
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
            constexpr int pack_factor = 4;             // 32/8
            constexpr int bytes_per_pack = 4;
            constexpr int values_per_thread = 8;       // pack_factor*packs_per_thread
            constexpr int block_size = 256;            // vpt*SIMD_SIZE
            constexpr int scale_step_per_thread = 8;   // gs(64)/vpt(8)
            const device uint8_t* ws = (const device uint8_t*)w;
            typedef float U;
            thread U x_thread[8];
            thread U result[4] = {0};
            const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;   // K bytes
            const int in_vec_size_g = in_vec_size / 64;
            const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) + simd_gid * results_per_simdgroup;
            ws     += out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
            scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
            biases += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
            x += tid.x * in_vec_size + simd_lid * values_per_thread;
            y += tid.x * out_vec_size + out_row;
            for (int k = 0; k < in_vec_size; k += block_size) {
                U sum = ld8(x, x_thread);
                for (int row = 0; row < results_per_simdgroup; row++) {
                    auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
                    const device half* sl = scales + row * in_vec_size_g;
                    const device half* bl = biases + row * in_vec_size_g;
                    U s = sl[0]; U b = bl[0];
                    result[row] += qd8(wl, x_thread, s, b, sum);
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
            let lib = try device.makeLibrary(source: src, options: mlxMatchCompileOpts())
            _qmm8Pipeline = try device.makeComputePipelineState(function: lib.makeFunction(name: "qmm8")!)
            return true
        } catch { print("[raw-qmm8] compile: \(error)"); return false }
    }

    static func qmm8(_ x: MLXArray, _ wq: MLXArray, scales: MLXArray, biases: MLXArray, M: Int, K: Int, N: Int) -> MLXArray? {
        guard let (device, queue) = ensure(), compileQmm8() else { return nil }
        guard N % 8 == 0, K % 512 == 0 else { print("[raw-qmm8] 非fast N=\(N) K=\(K)"); return nil }
        guard let bx = x.asType(.float16).asMTLBuffer(device: device, noCopy: false),
              let bwq = wq.asMTLBuffer(device: device, noCopy: false),
              let bsc = scales.asType(.float16).asMTLBuffer(device: device, noCopy: false),
              let bbi = biases.asType(.float16).asMTLBuffer(device: device, noCopy: false) else { return nil }
        let outBuf = device.makeBuffer(length: M * N * 2, options: .storageModeShared)!
        let cb = queue.makeCommandBuffer()!; let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(_qmm8Pipeline!)
        enc.setBuffer(bwq, offset: 0, index: 0); enc.setBuffer(bsc, offset: 0, index: 1)
        enc.setBuffer(bbi, offset: 0, index: 2); enc.setBuffer(bx, offset: 0, index: 3); enc.setBuffer(outBuf, offset: 0, index: 4)
        var kk = Int32(K), nn = Int32(N); enc.setBytes(&kk, length: 4, index: 5); enc.setBytes(&nn, length: 4, index: 6)
        enc.dispatchThreadgroups(MTLSize(width: M, height: N / 8, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 2, depth: 1))
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        let ptr = outBuf.contents().bindMemory(to: Float16.self, capacity: M * N)
        return MLXArray(Array(UnsafeBufferPointer(start: ptr, count: M * N)), [M, N])
    }

    nonisolated(unsafe) static var _routePipeline: MTLComputePipelineState?
    /// MoE routing top-8 選択 kernel（gate_logits[N]→inds[K] int32, scores[K] f16）。
    /// MLX softmax(precise)→argPartition(top-K)→takeAlong→normalize を再現。renorm で softmax Z は相殺するが
    /// MLX は f16 gates を経由するので f16 丸めも再現（gates=f16(exp/Z), scores=f16 gates[top], ssum=f16 順次）。
    /// 単一 thread（decode M=1, N=256 は軽量）。値ベース top-K は argPartition と同一 expert 集合を選ぶ。
    static func compileRoute() -> Bool {
        guard let (device, _) = ensure() else { return false }
        if _routePipeline != nil { return true }
        let src = """
        #include <metal_stdlib>
        using namespace metal;
        // ★並列化(1 threadgroup, N lanes): max/Z を木 reduction、top-K は K 回の argmax(値+index)reduction。
        //   選択は logits(Z 非依存・単調)＝single-thread 版と同じ集合(near-tie lossless 維持)。tie は lower-index。
        kernel void route_top8(device const half* logits [[buffer(0)]],
                               device int* inds [[buffer(1)]], device half* scores [[buffer(2)]],
                               constant uint& N [[buffer(3)]], constant uint& K [[buffer(4)]],
                               uint tid [[thread_position_in_threadgroup]], uint tgs [[threads_per_threadgroup]]) {
            threadgroup float red[256]; threadgroup int redi[256];
            threadgroup float gates[256]; threadgroup float work[256];
            threadgroup float bcast[1];
            float lg = (tid < N) ? (float)logits[tid] : -INFINITY;
            // max
            red[tid] = lg; threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint s = tgs/2; s > 0; s >>= 1) { if (tid < s) red[tid] = max(red[tid], red[tid+s]); threadgroup_barrier(mem_flags::mem_threadgroup); }
            if (tid == 0) bcast[0] = red[0]; threadgroup_barrier(mem_flags::mem_threadgroup);
            float m = bcast[0];
            // Z = Σ exp(lg-m)
            float e = (tid < N) ? precise::exp(lg - m) : 0.0f;
            red[tid] = e; threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint s = tgs/2; s > 0; s >>= 1) { if (tid < s) red[tid] += red[tid+s]; threadgroup_barrier(mem_flags::mem_threadgroup); }
            if (tid == 0) bcast[0] = red[0]; threadgroup_barrier(mem_flags::mem_threadgroup);
            float Z = bcast[0];
            if (tid < N) { gates[tid] = (float)(half)(e / Z); work[tid] = lg; }
            else { work[tid] = -INFINITY; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            // top-K: K 回の argmax(値+index)reduction
            for (uint k = 0; k < K; k++) {
                red[tid] = work[tid]; redi[tid] = (int)tid; threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint s = tgs/2; s > 0; s >>= 1) {
                    if (tid < s) { if (red[tid+s] > red[tid]) { red[tid] = red[tid+s]; redi[tid] = redi[tid+s]; } }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
                if (tid == 0) { int bi = redi[0]; inds[k] = bi; scores[k] = (half)gates[bi]; work[bi] = -INFINITY; }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            // normalize(thread0, K 小)
            if (tid == 0) { half ss = (half)0; for (uint k = 0; k < K; k++) ss += scores[k]; for (uint k = 0; k < K; k++) scores[k] = scores[k] / ss; }
        }
        """
        do { let lib = try device.makeLibrary(source: src, options: mlxMatchCompileOpts())
             _routePipeline = try device.makeComputePipelineState(function: lib.makeFunction(name: "route_top8")!)
             return true
        } catch { print("[raw-route] compile: \(error)"); return false }
    }

    /// standalone routing（検証用）: gate_logits[N] → (inds[K] int32, scores[K] f16)。
    static func routeTop8(_ logits: MLXArray, N: Int, K: Int) -> (MLXArray, MLXArray)? {
        guard let (device, queue) = ensure(), compileRoute() else { return nil }
        guard let bl = logits.asType(.float16).asMTLBuffer(device: device, noCopy: false) else { return nil }
        let bInds = device.makeBuffer(length: K * 4, options: .storageModeShared)!
        let bScores = device.makeBuffer(length: K * 2, options: .storageModeShared)!
        let cb = queue.makeCommandBuffer()!; let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(_routePipeline!)
        enc.setBuffer(bl, offset: 0, index: 0); enc.setBuffer(bInds, offset: 0, index: 1); enc.setBuffer(bScores, offset: 0, index: 2)
        var nn = UInt32(N), kk = UInt32(K); enc.setBytes(&nn, length: 4, index: 3); enc.setBytes(&kk, length: 4, index: 4)
        enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        let ip = bInds.contents().bindMemory(to: Int32.self, capacity: K)
        let sp = bScores.contents().bindMemory(to: Float16.self, capacity: K)
        return (MLXArray(Array(UnsafeBufferPointer(start: ip, count: K))),
                MLXArray(Array(UnsafeBufferPointer(start: sp, count: K))))
    }

    /// Metal routing（gate qmm8 + route_top8）: postNorm[H] → (inds[K] int32, scores[K] f16)。
    /// gate logits は qmm8（MLX と bit-exact 検証済）、選択/正規化は route_top8。combine/shared は呼び元（現状 MLX）。
    static func metalRouteGate(_ postNorm: MLXArray, gateW: MLXArray, gateS: MLXArray, gateB: MLXArray,
                               H: Int, N: Int = 256, K: Int = 8) -> (MLXArray, MLXArray)? {
        guard let logits = qmm8(postNorm.reshaped([1, H]), gateW, scales: gateS, biases: gateB, M: 1, K: H, N: N) else { return nil }
        return routeTop8(logits.reshaped([N]), N: N, K: K)
    }

    /// 検証: route_top8 vs MLX routing（softmax precise→argPartition→takeAlong→normalize）。順序非依存で集合一致。
    public static func runRouteTest() -> String {
        let N = 256, K = 8
        let logits = MLXRandom.normal([1, N]).asType(.float16); logits.eval()
        let gates = MLX.softmax(logits, axis: -1, precise: true)
        let order = MLX.argPartition(gates, kth: N - K, axis: -1)
        let indsM = order[0..., (N - K)...].asType(.int32)
        var scoresM = MLX.takeAlong(gates, indsM, axis: -1)
        scoresM = scoresM / scoresM.sum(axis: -1, keepDims: true)
        let indsMA = indsM.reshaped([K]).asArray(Int32.self)
        let scMA = scoresM.reshaped([K]).asArray(Float.self)
        var refMap: [Int32: Float] = [:]; for i in 0..<K { refMap[indsMA[i]] = scMA[i] }
        guard let (indsR, scoresR) = routeTop8(logits.reshaped([N]), N: N, K: K) else { return "[raw-route] 実行失敗" }
        let indsRA = indsR.asArray(Int32.self), scRA = scoresR.asType(.float32).asArray(Float.self)
        var setOK = true; var maxScoreDiff: Float = 0
        for i in 0..<K {
            guard let rs = refMap[indsRA[i]] else { setOK = false; continue }
            maxScoreDiff = max(maxScoreDiff, abs(rs - scRA[i]))
        }
        return String(format: "[raw-route-test] top-%d 選択 vs MLX argPartition: expert集合%@ score最大差=%.3e %@",
                      K, setOK ? "一致✅" : "不一致❌", maxScoreDiff,
                      setOK && maxScoreDiff < 2e-3 ? "✅ argmax-lossless 同等" : "⚠️")
    }

    /// 検証: 8bit qmv (router gate) vs MLX quantizedMatmul(bits=8)。env QWISP_RUN=raw-router-test。
    public static func runRouterTest() -> String {
        let H = 2048, N = 256
        let x = MLXRandom.normal([1, H]).asType(.float16)
        let wf = MLXRandom.normal([N, H]).asType(.float16)
        let (wq, sc, biOpt) = MLX.quantized(wf, groupSize: 64, bits: 8, mode: .affine)
        guard let bi = biOpt else { return "[raw-router] biases nil" }
        MLX.eval([x, wq, sc, bi])
        let ref = MLX.quantizedMatmul(x, wq, scales: sc, biases: bi, transpose: true, groupSize: 64, bits: 8, mode: .affine); ref.eval()
        guard let got = qmm8(x, wq, scales: sc, biases: bi, M: 1, K: H, N: N) else { return "[raw-router] qmm8 失敗" }
        got.eval()
        let rel = relErr(got.reshaped([N]), ref.reshaped([N]))
        return String(format: "[raw-router-test] 8bit qmv (gate %d←%d) vs MLX: rel=%.3e %@",
                      N, H, rel, rel == 0 ? "✅ TRUE bit-exact" : (rel < 2e-3 ? "△ near" : "❌"))
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
    nonisolated(unsafe) static var _swigluPipeline: MTLComputePipelineState?
    nonisolated(unsafe) static var _combinePipeline: MTLComputePipelineState?
    nonisolated(unsafe) static var _gate16Pipeline: MTLComputePipelineState?
    nonisolated(unsafe) static var _extractQPipeline: MTLComputePipelineState?     // attn SE: strided query 抽出
    nonisolated(unsafe) static var _sigmoidMulPipeline: MTLComputePipelineState?   // attn SE: gated=attnOut*sigmoid(gate)
    nonisolated(unsafe) static var _residAddPipeline: MTLComputePipelineState?     // 層融合: h += r（residual stream f16）
    nonisolated(unsafe) static var _sharedGate8Pipeline: MTLComputePipelineState?  // all-GPU MoE: shared gate scalar
    nonisolated(unsafe) static var _finalCombinePipeline: MTLComputePipelineState? // all-GPU MoE: y+gateScale*sharedY
    nonisolated(unsafe) static var _shiftConvPipeline: MTLComputePipelineState?    // GDN decode: conv cache shift
    nonisolated(unsafe) static var _writeKVPipeline: MTLComputePipelineState?      // attn decode: KV cache 散布

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
        // gate: silu(z.f32)*normed.f32 → f16。silu=z*sigmoid(z)[T=float, metal::exp]。normed が f32(promote)版
        kernel void gate(device const half* z [[buffer(0)]], device const float* normed [[buffer(1)]],
                         device half* outV [[buffer(2)]], constant uint& total [[buffer(3)]],
                         uint i [[thread_position_in_grid]]) {
            if (i >= total) return;
            float zf = (float)z[i];
            float y = 1.0f / (1.0f + exp(metal::abs(zf)));          // sigmoid(z), metal::exp, float
            float s = (zf < 0.0f) ? y : (1.0f - y);
            outV[i] = (half)((zf * s) * normed[i]);
        }
        // gate16: normed が f16(non-promote, la_norm F16)版。f16 normed を widen して gate。
        kernel void gate16(device const half* z [[buffer(0)]], device const half* normed [[buffer(1)]],
                           device half* outV [[buffer(2)]], constant uint& total [[buffer(3)]],
                           uint i [[thread_position_in_grid]]) {
            if (i >= total) return;
            float zf = (float)z[i];
            float y = 1.0f / (1.0f + exp(metal::abs(zf)));
            float s = (zf < 0.0f) ? y : (1.0f - y);
            outV[i] = (half)((zf * s) * (float)normed[i]);
        }
        // scale_mul: x[i] = (half)scale * x[i]（in-place, half 乗算）
        kernel void scale_mul(device half* x [[buffer(0)]], constant float& s [[buffer(1)]],
                              constant uint& total [[buffer(2)]], uint i [[thread_position_in_grid]]) {
            if (i >= total) return;
            x[i] = (half)s * x[i];
        }
        // swiglu: h = (g*sigmoid(g))*u（MoE/shared expert, 全 f16）。sigmoid は MLX stable(half, metal::exp)
        kernel void swiglu(device const half* g [[buffer(0)]], device const half* u [[buffer(1)]],
                           device half* h [[buffer(2)]], constant uint& total [[buffer(3)]],
                           uint i [[thread_position_in_grid]]) {
            if (i >= total) return;
            half gv = g[i];
            half y = (half)1 / ((half)1 + exp(metal::abs(gv)));
            half s = (gv < (half)0) ? y : ((half)1 - y);
            h[i] = (gv * s) * u[i];
        }
        // combine: y[n] = Σ_k (d[k,n]*scores[k])。★MLX は f16 sum(remap_reduce_types: float は {in,in}=f16 累積)。
        //   scores も f16(precise softmax でも出力 f16)。積も累積も f16、sequential k で MLX reduce と一致。
        kernel void combine(device const half* d [[buffer(0)]], device const half* scores [[buffer(1)]],
                            device half* y [[buffer(2)]], constant uint& K [[buffer(3)]],
                            constant uint& N [[buffer(4)]], uint n [[thread_position_in_grid]]) {
            if (n >= N) return;
            half acc = (half)0;
            for (uint k = 0; k < K; ++k) acc += d[k*N + n] * scores[k];
            y[n] = acc;
        }
        // extract_q（attn SE）: qOut[Hh, qd2] の query 部 [:,0:headDim] を contiguous q[Hh, headDim] に複製。
        //   rms の row-stride 問題(qOut の row stride=qd2≠headDim)を回避。純コピー(演算無し, bit-exact)。
        kernel void extract_q(device const half* qOut [[buffer(0)]], device half* q [[buffer(1)]],
                              constant uint& headDim [[buffer(2)]], constant uint& qd2 [[buffer(3)]],
                              constant uint& total [[buffer(4)]], uint i [[thread_position_in_grid]]) {
            if (i >= total) return;
            uint h = i / headDim, d = i % headDim;
            q[i] = qOut[h * qd2 + d];
        }
        // sigmoid_mul（attn SE）: gated[i]=attnOut[i]*sigmoid(gate[i])。gate は qOut[:,headDim:qd2] から strided 読み。
        //   MLX gated=output_f16*sigmoid(gate_f16) を f16 で再現（sigmoid は MLX stable, half, metal::exp）。
        kernel void sigmoid_mul(device const half* attnOut [[buffer(0)]], device const half* qOut [[buffer(1)]],
                                device half* gated [[buffer(2)]], constant uint& headDim [[buffer(3)]],
                                constant uint& qd2 [[buffer(4)]], constant uint& total [[buffer(5)]],
                                uint i [[thread_position_in_grid]]) {
            if (i >= total) return;
            uint h = i / headDim, d = i % headDim;
            half gv = qOut[h * qd2 + headDim + d];
            half y = (half)1 / ((half)1 + exp(metal::abs(gv)));
            half s = (gv < (half)0) ? y : ((half)1 - y);
            gated[i] = attnOut[i] * s;
        }
        // resid_add（層融合）: h[i] = h[i] + r[i]（residual stream f16, in-place into h）。MLX h+r を f16 で再現。
        kernel void resid_add(device half* h [[buffer(0)]], device const half* r [[buffer(1)]],
                              constant uint& total [[buffer(2)]], uint i [[thread_position_in_grid]]) {
            if (i >= total) return;
            h[i] = h[i] + r[i];
        }
        // shared_gate8（all-GPU MoE）: shared_expert_gate(8bit, N=1, K, gs=64)の dot → sigmoid → scalar[0]。
        //   ★並列化(threadgroup reduction): 各 thread が group を stride 分担し partial sum → 木 reduction。
        //   near-tie 許容ゆえ reduction 順序差は無害(shared gate は元々 f32 近似)。1 threadgroup。
        kernel void shared_gate8(device const uint8_t* w [[buffer(0)]], device const half* scales [[buffer(1)]],
                                 device const half* biases [[buffer(2)]], device const half* x [[buffer(3)]],
                                 device half* out [[buffer(4)]], constant uint& K [[buffer(5)]],
                                 uint tid [[thread_position_in_threadgroup]], uint tgs [[threads_per_threadgroup]]) {
            threadgroup float part[256];
            uint G = K / 64;
            float acc = 0.0f;
            for (uint g = tid; g < G; g += tgs) {
                float sq = 0.0f, sx = 0.0f;
                for (uint j = 0; j < 64; j++) { uint k = g*64 + j; float xv = (float)x[k]; sq += xv * (float)w[k]; sx += xv; }
                acc += (float)scales[g] * sq + (float)biases[g] * sx;
            }
            part[tid] = acc;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint s = tgs / 2; s > 0; s >>= 1) { if (tid < s) part[tid] += part[tid + s]; threadgroup_barrier(mem_flags::mem_threadgroup); }
            if (tid == 0) { float a = part[0]; float y = 1.0f / (1.0f + precise::exp(metal::abs(a))); out[0] = (half)(a < 0.0f ? y : (1.0f - y)); }
        }
        // final_combine（all-GPU MoE）: combined[i] = y[i] + gateScale[0]*sharedY[i]（f16）。
        kernel void final_combine(device const half* y [[buffer(0)]], device const half* sharedY [[buffer(1)]],
                                  device const half* gateScale [[buffer(2)]], device half* combined [[buffer(3)]],
                                  constant uint& H [[buffer(4)]], uint i [[thread_position_in_grid]]) {
            if (i >= H) return;
            combined[i] = y[i] + gateScale[0] * sharedY[i];
        }
        // shift_conv（GDN decode）: conv cache を 1 行上シフト（最古を捨て新トークン分を空ける）。
        //   thread=列 c（race-free: 各 thread 自分の列の K 行を昇順 read/write）。その後 qkv が row(K-1)を書く。
        kernel void shift_conv(device half* conv [[buffer(0)]], constant uint& K [[buffer(1)]],
                               constant uint& C [[buffer(2)]], uint c [[thread_position_in_grid]]) {
            if (c >= C) return;
            for (uint j = 0; j + 1 < K; j++) conv[j*C + c] = conv[(j+1)*C + c];
        }
        // write_kv（attention decode）: src[KV, D] を cache[KV, maxLen, D] の seq 位置 pos に散布。
        kernel void write_kv(device const half* src [[buffer(0)]], device half* cache [[buffer(1)]],
                             constant uint& KV [[buffer(2)]], constant uint& D [[buffer(3)]],
                             constant uint& maxLen [[buffer(4)]], constant uint& pos [[buffer(5)]],
                             uint i [[thread_position_in_grid]]) {
            if (i >= KV * D) return;
            uint h = i / D, d = i % D;
            cache[h * maxLen * D + pos * D + d] = src[h * D + d];
        }
        """
        do {
            let lib = try device.makeLibrary(source: src, options: mlxMatchCompileOpts())
            _auxLib = lib
            _cgbPipeline = try device.makeComputePipelineState(function: lib.makeFunction(name: "compute_g_beta")!)
            _gatePipeline = try device.makeComputePipelineState(function: lib.makeFunction(name: "gate")!)
            _scalePipeline = try device.makeComputePipelineState(function: lib.makeFunction(name: "scale_mul")!)
            _swigluPipeline = try device.makeComputePipelineState(function: lib.makeFunction(name: "swiglu")!)
            _combinePipeline = try device.makeComputePipelineState(function: lib.makeFunction(name: "combine")!)
            _gate16Pipeline = try device.makeComputePipelineState(function: lib.makeFunction(name: "gate16")!)
            _extractQPipeline = try device.makeComputePipelineState(function: lib.makeFunction(name: "extract_q")!)
            _sigmoidMulPipeline = try device.makeComputePipelineState(function: lib.makeFunction(name: "sigmoid_mul")!)
            _residAddPipeline = try device.makeComputePipelineState(function: lib.makeFunction(name: "resid_add")!)
            _sharedGate8Pipeline = try device.makeComputePipelineState(function: lib.makeFunction(name: "shared_gate8")!)
            _finalCombinePipeline = try device.makeComputePipelineState(function: lib.makeFunction(name: "final_combine")!)
            _shiftConvPipeline = try device.makeComputePipelineState(function: lib.makeFunction(name: "shift_conv")!)
            _writeKVPipeline = try device.makeComputePipelineState(function: lib.makeFunction(name: "write_kv")!)
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
                          Ktop: Int, K: Int, N: Int, gs: Int = 64, lhsPerExpert: Bool = false) -> MLXArray? {
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
                              constant uint& lhsPer      [[buffer(8)]],   // 1=x を ki 行で index(down用), 0=共有(gate/up)
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
                x += (lhsPer ? (size_t)ki : 0) * in_vec_size + simd_lid * values_per_thread;  // down は ki 行、gate/up は共有
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
        var kk = Int32(K), nn = Int32(N), lhs = UInt32(lhsPerExpert ? 1 : 0)
        enc.setBytes(&kk, length: 4, index: 6); enc.setBytes(&nn, length: 4, index: 7); enc.setBytes(&lhs, length: 4, index: 8)
        enc.dispatchThreadgroups(MTLSize(width: 1, height: N / 8, depth: Ktop),
                                 threadsPerThreadgroup: MTLSize(width: 32, height: 2, depth: 1))
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        let ptr = outBuf.contents().bindMemory(to: Float16.self, capacity: Ktop * N)
        return MLXArray(Array(UnsafeBufferPointer(start: ptr, count: Ktop * N)), [Ktop, N])
    }

    /// raw-Metal SDPA(decode L=1, GQA, flash/online softmax, f32)。
    /// q[H,D], K/V[KV,S,D] → out[H,D]。head h は kv=h/(H/KV)。MLXFast.scaledDotProductAttention(f32,.none) と照合。
    static func sdpaDecode(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray,
                           H: Int, KV: Int, D: Int, S: Int, scale: Float, inF32: Bool = false) -> MLXArray? {
        guard let (device, queue) = ensure() else { return nil }
        guard D == 256 else { print("[raw-sdpa] D!=256 未対応 D=\(D)"); return nil }
        // ★ MLX sdpa_vector(sdpa_vector.h)逐語移植: BN=32 simdgroup が key を分担, BD=32 lane が head dim
        //   (qk_per_thread=8)。scale は q に適用, fast::exp, simd_sum/simd_max, cross-simdgroup combine。
        //   group=(1024,1,1)=32sg×32lane, grid=(H,1,1)。no mask/causal/sinks, query 非転置。decode L=1。
        //   inF32=true で q/k/v/out を f32（attention f32 cascade: qk-norm 後 f32, values も f32 昇格）。
        let XT = inF32 ? "float" : "half"
        if (inF32 ? _sdpaF32Pipeline : _sdpaPipeline) == nil {
            let src = """
            #include <metal_stdlib>
            #include <metal_simdgroup>
            using namespace metal;
            kernel void sdpa(device const \(XT)* queries [[buffer(0)]],   // [H, D]
                             device const \(XT)* keys    [[buffer(1)]],   // [KV, N, D]
                             device const \(XT)* values  [[buffer(2)]],   // [KV, N, D]
                             device \(XT)* out           [[buffer(3)]],   // [H, D]
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
                if (simd_lid == 0) { for (int i = 0; i < v_per_thread; i++) out[i] = (\(XT))o[i]; }
            }
            """
            do { let lib = try device.makeLibrary(source: src, options: mlxMatchCompileOpts())
                 let p = try device.makeComputePipelineState(function: lib.makeFunction(name: "sdpa")!)
                 if inF32 { _sdpaF32Pipeline = p } else { _sdpaPipeline = p }
            } catch { print("[raw-sdpa] compile: \(error)"); return nil }
        }
        let dt: DType = inF32 ? .float32 : .float16, elem = inF32 ? 4 : 2
        guard let bq = q.asType(dt).asMTLBuffer(device: device, noCopy: false),
              let bk = k.asType(dt).asMTLBuffer(device: device, noCopy: false),
              let bv = v.asType(dt).asMTLBuffer(device: device, noCopy: false) else { return nil }
        let outBuf = device.makeBuffer(length: H * D * elem, options: .storageModeShared)!
        let cb = queue.makeCommandBuffer()!; let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(inF32 ? _sdpaF32Pipeline! : _sdpaPipeline!)
        enc.setBuffer(bq, offset: 0, index: 0); enc.setBuffer(bk, offset: 0, index: 1)
        enc.setBuffer(bv, offset: 0, index: 2); enc.setBuffer(outBuf, offset: 0, index: 3)
        var gqa = Int32(H / KV), nn = Int32(S), khs = Int32(S * D), kss = Int32(D), vhs = Int32(S * D), vss = Int32(D), sc = scale
        enc.setBytes(&gqa, length: 4, index: 4); enc.setBytes(&nn, length: 4, index: 5)
        enc.setBytes(&khs, length: 4, index: 6); enc.setBytes(&kss, length: 4, index: 7)
        enc.setBytes(&vhs, length: 4, index: 8); enc.setBytes(&vss, length: 4, index: 9)
        enc.setBytes(&sc, length: 4, index: 10)
        enc.dispatchThreadgroups(MTLSize(width: H, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1024, height: 1, depth: 1))
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        if inF32 {
            let ptr = outBuf.contents().bindMemory(to: Float.self, capacity: H * D)
            return MLXArray(Array(UnsafeBufferPointer(start: ptr, count: H * D)), [H, D])
        }
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
    static func rope(_ x: MLXArray, headDim HD: Int, ropeDim rd: Int, base: Float, offset: Int, xF32: Bool = false) -> MLXArray? {
        guard let (device, queue) = ensure() else { return nil }
        let XT = xF32 ? "float" : "half"
        if (xF32 ? _ropeF32Pipeline : _ropePipeline) == nil {
            let src = """
            #include <metal_stdlib>
            using namespace metal;
            kernel void rope(device const \(XT)* x [[buffer(0)]], device \(XT)* out [[buffer(1)]],
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
                out[gid] = (\(XT))(d < hd2 ? (x0*c - x1*s) : (x0*s + x1*c));
            }
            """
            // ★ RoPE は MLX が fast-math transcendental(sin/cos/exp)＝既定 options(nil)で f16 rel 0(safe だと 2.9e-5)。
            do { let lib = try device.makeLibrary(source: src, options: nil)
                 let p = try device.makeComputePipelineState(function: lib.makeFunction(name: "rope")!)
                 if xF32 { _ropeF32Pipeline = p } else { _ropePipeline = p }
            } catch { print("[raw-rope] compile: \(error)"); return nil }
        }
        let rows = x.size / HD
        let dt: DType = xF32 ? .float32 : .float16, elem = xF32 ? 4 : 2
        guard let bx = x.asType(dt).asMTLBuffer(device: device, noCopy: false) else { return nil }
        let outBuf = device.makeBuffer(length: rows * HD * elem, options: .storageModeShared)!
        let cb = queue.makeCommandBuffer()!; let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(xF32 ? _ropeF32Pipeline! : _ropePipeline!)
        enc.setBuffer(bx, offset: 0, index: 0); enc.setBuffer(outBuf, offset: 0, index: 1)
        var h = UInt32(HD), r = UInt32(rd), b = base, p = Float(offset)
        enc.setBytes(&h, length: 4, index: 2); enc.setBytes(&r, length: 4, index: 3)
        enc.setBytes(&b, length: 4, index: 4); enc.setBytes(&p, length: 4, index: 5)
        let total = rows * HD, tgw = min((xF32 ? _ropeF32Pipeline! : _ropePipeline!).maxTotalThreadsPerThreadgroup, 256)
        enc.dispatchThreads(MTLSize(width: total, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: tgw, height: 1, depth: 1))
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        if xF32 {
            let ptr = outBuf.contents().bindMemory(to: Float.self, capacity: total)
            return MLXArray(Array(UnsafeBufferPointer(start: ptr, count: total)), [rows, HD])
        }
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
        //   normWeight が f32 のときのみ MLX は out を f32 に昇格(promoteF32)。f16 なら f16(dec0 の la_norm は F16)。
        let promoteRMS = (w.normWeight.dtype == .float32)
        guard let normed = rmsNorm(coreOut.reshaped([numVHeads, headVDim]), w.normWeight,
                                   eps: eps, D: headVDim, promoteF32: promoteRMS) else { return nil }
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
        let normF32: Bool                        // la_norm が F32(promote) か F16 か
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
              let bConvW = mtl(w.conv1dW, .float32),
              let bNormW = mtl(w.normWeight, w.normWeight.dtype == .float32 ? .float32 : .float16),   // la_norm dtype 維持
              let bALog = mtl(w.aLog, .float32), let bDt = mtl(w.dtBias, .float32),
              let bOnes = MLXArray.ones([headKDim], dtype: .float16).asMTLBuffer(device: device, noCopy: false)
        else { print("[raw-gdn-se] weight buffer nil"); return nil }
        let normF32 = (w.normWeight.dtype == .float32)
        let convInput = mk(convKernel * convDim * 2); memset(convInput.contents(), 0, convKernel * convDim * 2)  // zero 1 回（qkv が row(K-1)を毎回上書き）
        let stateBuf = mk(Hv * Dv * Dk * 4); memset(stateBuf.contents(), 0, Hv * Dv * Dk * 4)                   // cold state zero 1 回
        return GDNBuffers(
            H: H, keyDim: keyDim, valueDim: valueDim, convDim: convDim, Dk: Dk, Dv: Dv, Hv: Hv, Hk: Hk,
            convKernel: convKernel, invScale: Float(pow(Double(headKDim), -0.5)), normF32: normF32,
            bx: mk(H * 2),
            bQkvW: bQkvW, bQkvS: bQkvS, bQkvB: bQkvB, bZW: bZW, bZS: bZS, bZB: bZB,
            bAW: bAW, bAS: bAS, bAB: bAB, bBW: bBW, bBS: bBS, bBB: bBB, bOW: bOW, bOS: bOS, bOB: bOB,
            bConvW: bConvW, bNormW: bNormW, bALog: bALog, bDt: bDt, bOnes: bOnes,
            convInput: convInput, zBuf: mk(valueDim * 2), aBuf: mk(Hv * 2), bBuf: mk(Hv * 2),
            gBuf: mk(Hv * 4), betaBuf: mk(Hv * 4), convOut: mk(convDim * 2), qN: mk(keyDim * 2), kN: mk(keyDim * 2),
            stateBuf: stateBuf, stateOut: mk(Hv * Dv * Dk * 4), coreOut: mk(valueDim * 2),
            normed: mk(valueDim * (normF32 ? 4 : 2)), outV: mk(valueDim * 2), outBuf: mk(H * 2))
    }

    /// ★ task#3: GDN 1 層を **単一 command buffer + 単一 encoder** で連結（中間 buffer GPU 常駐、
    ///   commit/MLXArray 復帰なし）。常駐 buffer を使い per-call は x 書込み＋encode＋commit＋read のみ。
    ///   round-trip 版 gdnLayerRaw と bit-exact。pipeline は事前 warm 前提。cold cache(decode 1 step)。
    static func gdnLayerSingleEncoder(_ x: MLXArray, _ b: GDNBuffers, eps: Float = 1e-6) -> MLXArray? {
        guard let (_, queue) = ensure() else { return nil }
        guard checkGDNPipelines(b) else { return nil }
        let H = b.H
        // x を常駐 buffer に書込み（per-call、H*2 bytes のみ）
        let xf = x.reshaped([H]).asType(.float16).asArray(Float16.self)
        b.bx.contents().bindMemory(to: Float16.self, capacity: H).update(from: xf, count: H)
        let cb = queue.makeCommandBuffer()!; let enc = cb.makeComputeCommandEncoder()!
        encodeGDNBody(enc, b, eps: eps)                  // b.bx → b.outBuf
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        let ptr = b.outBuf.contents().bindMemory(to: Float16.self, capacity: H)
        return MLXArray(Array(UnsafeBufferPointer(start: ptr, count: H)), [1, H])
    }

    static func checkGDNPipelines(_ b: GDNBuffers) -> Bool {
        guard _qmmPipeline != nil, _rmsPipeline != nil, _conv1dPipeline != nil, _recurPipeline != nil,
              _cgbPipeline != nil, _gatePipeline != nil, _scalePipeline != nil else {
            print("[raw-gdn-se] pipeline 未 warm（先に gdnLayerRaw を呼ぶ）"); return false
        }
        if b.normF32 && _rmsPipelineF32 == nil { print("[raw-gdn-se] rmsF32 未 warm"); return false }
        return true
    }

    /// GDN 1 層の kernel chain を既存 encoder に encode（b.bx → b.outBuf）。input_norm/residual/post_norm は含まない。
    /// gdnLayerSingleEncoder（単体）と fusedDecoderLayer（層融合）で共有する単一ソース。
    static func encodeGDNBody(_ enc: MTLComputeCommandEncoder, _ b: GDNBuffers, eps: Float, decode: Bool = false) {
        let qp = _qmmPipeline!, rp = _rmsPipeline!, cp = _conv1dPipeline!, rcp = _recurPipeline!
        let cgb = _cgbPipeline!, gp = _gatePipeline!, scp = _scalePipeline!
        let H = b.H, keyDim = b.keyDim, valueDim = b.valueDim, convDim = b.convDim
        let Dk = b.Dk, Dv = b.Dv, Hv = b.Hv, Hk = b.Hk, convKernel = b.convKernel, invScale = b.invScale
        let bx = b.bx
        let bConvW = b.bConvW, bNormW = b.bNormW, bALog = b.bALog, bDt = b.bDt, bOnes = b.bOnes
        let convInput = b.convInput, zBuf = b.zBuf, aBuf = b.aBuf, bBuf = b.bBuf, gBuf = b.gBuf, betaBuf = b.betaBuf
        let convOut = b.convOut, qN = b.qN, kN = b.kN, stateBuf = b.stateBuf, stateOut = b.stateOut
        let coreOut = b.coreOut, normed = b.normed, outV = b.outV, outBuf = b.outBuf
        func encQmm(_ wq: MTLBuffer, _ sc: MTLBuffer, _ bi: MTLBuffer, _ xb: MTLBuffer, xoff: Int,
                    _ y: MTLBuffer, yoff: Int, K: Int, N: Int) {
            enc.setComputePipelineState(qp)
            enc.setBuffer(wq, offset: 0, index: 0); enc.setBuffer(sc, offset: 0, index: 1)
            enc.setBuffer(bi, offset: 0, index: 2); enc.setBuffer(xb, offset: xoff, index: 3)
            enc.setBuffer(y, offset: yoff, index: 4)
            var kk = Int32(K), nn = Int32(N); enc.setBytes(&kk, length: 4, index: 5); enc.setBytes(&nn, length: 4, index: 6)
            enc.dispatchThreadgroups(MTLSize(width: 1, height: N / 8, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 2, depth: 1))
        }
        func encRms(_ xb: MTLBuffer, xoff: Int, _ wb: MTLBuffer, _ ob: MTLBuffer, rows: Int, D: Int, promote: Bool) {
            enc.setComputePipelineState(promote ? _rmsPipelineF32! : rp)
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
        // ⓪ decode: conv cache を 1 行上シフト（最古を捨てる）。cold(prefill 先頭)は memset 済ゆえ不要。
        if decode {
            enc.setComputePipelineState(_shiftConvPipeline!)
            enc.setBuffer(convInput, offset: 0, index: 0)
            var kk = UInt32(convKernel), cc = UInt32(convDim); enc.setBytes(&kk, length: 4, index: 1); enc.setBytes(&cc, length: 4, index: 2)
            enc.dispatchThreads(MTLSize(width: convDim, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(_shiftConvPipeline!.maxTotalThreadsPerThreadgroup, 256), height: 1, depth: 1))
        }
        // ① in_proj 4 本。qkv は convInput の row(K-1) に直接書く（zero-pad は memset 済）。
        if !profSkipGDNMatmul {
            encQmm(b.bQkvW, b.bQkvS, b.bQkvB, bx, xoff: 0, convInput, yoff: (convKernel - 1) * convDim * 2, K: H, N: convDim)
            encQmm(b.bZW, b.bZS, b.bZB, bx, xoff: 0, zBuf, yoff: 0, K: H, N: valueDim)
            encQmm(b.bAW, b.bAS, b.bAB, bx, xoff: 0, aBuf, yoff: 0, K: H, N: Hv)
            encQmm(b.bBW, b.bBS, b.bBB, bx, xoff: 0, bBuf, yoff: 0, K: H, N: Hv)
        }
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
        if !profSkipGDNRecur {
        enc.setComputePipelineState(rcp)
        enc.setBuffer(qN, offset: 0, index: 0); enc.setBuffer(kN, offset: 0, index: 1); enc.setBuffer(convOut, offset: 2 * keyDim * 2, index: 2)
        enc.setBuffer(gBuf, offset: 0, index: 3); enc.setBuffer(betaBuf, offset: 0, index: 4); enc.setBuffer(stateBuf, offset: 0, index: 5)
        var tt = Int32(1); enc.setBytes(&tt, length: 4, index: 6)
        // decode は state を in-place 更新（stateBuf を out にも＝次トークンへ feedback。recurrent は register に
        // state を読んでから書くので in-place 安全）。cold は別 stateOut（毎回独立）。
        enc.setBuffer(coreOut, offset: 0, index: 7); enc.setBuffer(decode ? stateBuf : stateOut, offset: 0, index: 8)
        enc.dispatchThreads(MTLSize(width: 32, height: Dv, depth: Hv), threadsPerThreadgroup: MTLSize(width: 32, height: 4, depth: 1))
        }
        // ⑥ RMSNormGated: rmsNorm(promote=normF32) → gate(normF32 で f32/f16 版)
        encRms(coreOut, xoff: 0, bNormW, normed, rows: Hv, D: Dv, promote: b.normF32)
        enc.setComputePipelineState(b.normF32 ? gp : _gate16Pipeline!)
        enc.setBuffer(zBuf, offset: 0, index: 0); enc.setBuffer(normed, offset: 0, index: 1); enc.setBuffer(outV, offset: 0, index: 2)
        var vt = UInt32(valueDim); enc.setBytes(&vt, length: 4, index: 3)
        enc.dispatchThreads(MTLSize(width: valueDim, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(gp.maxTotalThreadsPerThreadgroup, 256), height: 1, depth: 1))
        // ⑦ out_proj
        if !profSkipGDNMatmul { encQmm(b.bOW, b.bOS, b.bOB, outV, xoff: 0, outBuf, yoff: 0, K: valueDim, N: H) }
    }

    // ══════════ 層全体融合（task#4）: residual stream を resident GPU buffer に保持 ══════════
    // 各層の input_norm + mixer + residual + post_norm を **単一 command buffer** で連結し、
    // MLXArray 往復（standalone rmsNorm CB ×2/層・mixer の x-copy/readback・residual の MLX add）を排除。
    // MoE routing(argPartition) のみ MLX 必須＝層に 1 sync 不可避（postNorm readback → routing → expert SE）。

    /// 層融合用の norm 重み常駐 buffer（per layer: input/post layernorm の f16 重み）。
    struct NormWeightBuffers { let inputNormW, postNormW: MTLBuffer }
    static func prepareNormWeights(input: MLXArray, post: MLXArray) -> NormWeightBuffers? {
        guard let (device, _) = ensure() else { return nil }
        guard let iw = input.asType(.float16).asMTLBuffer(device: device, noCopy: false),
              let pw = post.asType(.float16).asMTLBuffer(device: device, noCopy: false) else { return nil }
        return NormWeightBuffers(inputNormW: iw, postNormW: pw)
    }

    /// 共有 residual stream buffer（H f16）を確保。
    static func makeResidentBuffer(_ bytes: Int) -> MTLBuffer? {
        guard let (device, _) = ensure() else { return nil }
        return device.makeBuffer(length: bytes, options: .storageModeShared)
    }
    /// MLXArray を f16 MTLBuffer 化（重み常駐用）。
    static func f16Buffer(_ a: MLXArray) -> MTLBuffer? {
        guard let (device, _) = ensure() else { return nil }
        return a.asType(.float16).asMTLBuffer(device: device, noCopy: false)
    }
    /// MLXArray[*,H] → hBuf（f16）書込み。
    static func writeBuffer(_ buf: MTLBuffer, _ a: MLXArray, _ H: Int) {
        let f = a.reshaped([H]).asType(.float16).asArray(Float16.self)
        buf.contents().bindMemory(to: Float16.self, capacity: H).update(from: f, count: H)
    }
    /// hBuf（f16）→ MLXArray[1,H]。
    static func readBuffer(_ buf: MTLBuffer, _ H: Int) -> MLXArray {
        let ptr = buf.contents().bindMemory(to: Float16.self, capacity: H)
        return MLXArray(Array(UnsafeBufferPointer(start: ptr, count: H)), [1, H])
    }

    /// encoder に rmsNorm を encode（src→out, weight=f16 non-promote）。層融合の input/post norm 用。
    private static func encodeRms(_ enc: MTLComputeCommandEncoder, src: MTLBuffer, w: MTLBuffer, out: MTLBuffer,
                                  D: Int, eps: Float) {
        enc.setComputePipelineState(_rmsPipeline!)
        enc.setBuffer(src, offset: 0, index: 0); enc.setBuffer(w, offset: 0, index: 1); enc.setBuffer(out, offset: 0, index: 2)
        var ee = eps, asz = UInt32(D), ws = UInt32(1)
        enc.setBytes(&ee, length: 4, index: 3); enc.setBytes(&asz, length: 4, index: 4); enc.setBytes(&ws, length: 4, index: 5)
        let tg = ((((D + 3) / 4) + 31) / 32) * 32
        enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
    }
    /// encoder に resid_add を encode（h[i] += r[i], in-place）。
    private static func encodeResidAdd(_ enc: MTLComputeCommandEncoder, h: MTLBuffer, r: MTLBuffer, total: Int) {
        enc.setComputePipelineState(_residAddPipeline!)
        enc.setBuffer(h, offset: 0, index: 0); enc.setBuffer(r, offset: 0, index: 1)
        var tt = UInt32(total); enc.setBytes(&tt, length: 4, index: 2)
        enc.dispatchThreads(MTLSize(width: total, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(_residAddPipeline!.maxTotalThreadsPerThreadgroup, 256), height: 1, depth: 1))
    }

    /// ★ 層融合 mixer-half: input_norm + mixer(GDN/attn) + residual + post_norm を **単一 encoder** で連結。
    /// hBuf(residual stream, in/out, f16) を直読/直更新。postNorm を返す（MoE routing 用, MLXArray）。
    /// gdn か attn のどちらか一方を渡す。mixer の入力 buffer(b.bx) に input_norm 結果を書く。
    static func fusedMixerHalf(hBuf: MTLBuffer, nw: NormWeightBuffers, postNormBuf: MTLBuffer,
                               gdn: GDNBuffers?, attn: AttnBuffers?, H: Int, eps: Float,
                               pendingResid: MTLBuffer? = nil) -> MLXArray? {
        guard let (_, queue) = ensure(), _rmsPipeline != nil, _residAddPipeline != nil else { return nil }
        let mixerBx: MTLBuffer, mixerOut: MTLBuffer
        if let g = gdn { guard checkGDNPipelines(g) else { return nil }; mixerBx = g.bx; mixerOut = g.outBuf }
        else if let a = attn { guard checkAttnPipelines() else { return nil }; mixerBx = a.bx; mixerOut = a.outBuf }
        else { return nil }
        let cb = queue.makeCommandBuffer()!; let enc = cb.makeComputeCommandEncoder()!
        // ⓪ 前層 MoE residual を畳む（hBuf += prevCombined）。別 CB を消し本 encoder 先頭に統合。
        if let pr = pendingResid { encodeResidAdd(enc, h: hBuf, r: pr, total: H) }
        // ① input_norm: hBuf → mixer.bx（f16, non-promote）
        encodeRms(enc, src: hBuf, w: nw.inputNormW, out: mixerBx, D: H, eps: eps)
        // ② mixer: mixer.bx → mixer.outBuf
        if let g = gdn { encodeGDNBody(enc, g, eps: eps) } else { encodeAttnBody(enc, attn!) }
        // ③ residual: hBuf += mixer.outBuf
        encodeResidAdd(enc, h: hBuf, r: mixerOut, total: H)
        // ④ post_norm: hBuf → postNormBuf（f16）
        encodeRms(enc, src: hBuf, w: nw.postNormW, out: postNormBuf, D: H, eps: eps)
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        return readBuffer(postNormBuf, H)
    }

    /// 層融合 MoE residual flush: hBuf += combinedBuf（最終層のみ。中間層は次層 mixer 先頭で畳む）。
    static func fusedMoEResidual(hBuf: MTLBuffer, combinedBuf: MTLBuffer, H: Int) {
        guard let (_, queue) = ensure(), _residAddPipeline != nil else { return }
        let cb = queue.makeCommandBuffer()!; let enc = cb.makeComputeCommandEncoder()!
        encodeResidAdd(enc, h: hBuf, r: combinedBuf, total: H)
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
    }

    // ══════════ all-GPU 層融合（task#8, 多層 1-CB）: routing/combine も Metal＝forward 全体 MLX 非依存 ══════════
    // routing は near-tie で lossless 検証済(route-decode-lossless)。全 kernel を **単一 serial encoder** に連結し、
    // 複数層を 1 command buffer に詰めて per-layer waitUntilCompleted を排除→GPU が層間 pipeline。

    /// gate router(8bit) 重み常駐 buffer。
    struct Gate8Buffers { let wq, sc, bi: MTLBuffer; let N: Int }
    static func prepareGate8(_ w: MLXArray, _ s: MLXArray, _ b: MLXArray) -> Gate8Buffers? {
        guard let (device, _) = ensure(), compileQmm8(), compileRoute() else { return nil }
        guard let wq = w.asMTLBuffer(device: device, noCopy: false),
              let sc = s.asType(.float16).asMTLBuffer(device: device, noCopy: false),
              let bi = b.asType(.float16).asMTLBuffer(device: device, noCopy: false) else { return nil }
        return Gate8Buffers(wq: wq, sc: sc, bi: bi, N: w.dim(0))
    }

    /// all-GPU 層融合の共有中間 buffer（全層で再利用＝serial 実行ゆえ安全）。normed=final norm 出力(同一 CB 内)。
    struct GPUScratch { let postNorm, gateLogits, scores, y, gateScale, combined, normed: MTLBuffer }
    static func makeGPUScratch(H: Int, E: Int, K: Int) -> GPUScratch? {
        guard let (device, _) = ensure() else { return nil }
        func mk(_ n: Int) -> MTLBuffer { device.makeBuffer(length: n, options: .storageModeShared)! }
        return GPUScratch(postNorm: mk(H*2), gateLogits: mk(E*2), scores: mk(K*2), y: mk(H*2), gateScale: mk(2), combined: mk(H*2), normed: mk(H*2))
    }

    /// mixer-half を encoder に encode（CB/commit 無し, postNorm は postNormBuf に残る）。多層 1-CB 用。
    static func encodeMixerHalf(_ enc: MTLComputeCommandEncoder, hBuf: MTLBuffer, nw: NormWeightBuffers,
                                postNormBuf: MTLBuffer, gdn: GDNBuffers?, attn: AttnBuffers?, H: Int, eps: Float,
                                pendingResid: MTLBuffer?, decode: Bool = false, pos: Int = 0) {
        if let pr = pendingResid { encodeResidAdd(enc, h: hBuf, r: pr, total: H) }
        let mixerBx: MTLBuffer, mixerOut: MTLBuffer
        if let g = gdn { mixerBx = g.bx; mixerOut = g.outBuf } else { mixerBx = attn!.bx; mixerOut = attn!.outBuf }
        encodeRms(enc, src: hBuf, w: nw.inputNormW, out: mixerBx, D: H, eps: eps)
        if !profSkipMixer {
            if let g = gdn { encodeGDNBody(enc, g, eps: eps, decode: decode) } else { encodeAttnBody(enc, attn!, decode: decode, pos: pos) }
        }
        encodeResidAdd(enc, h: hBuf, r: mixerOut, total: H)
        encodeRms(enc, src: hBuf, w: nw.postNormW, out: postNormBuf, D: H, eps: eps)
    }

    /// MoE 全体（gate→route→experts→combine→shared→final）を encoder に encode（全 GPU, MLX 非依存）。
    /// 出力 = sc.combined（= y + gateScale*sharedY）。inds は moe.binds に直接書く（gather が読む）。
    static func encodeMoEGPU(_ enc: MTLComputeCommandEncoder, postNorm: MTLBuffer, gate: Gate8Buffers,
                             moe: MoEBuffers, sc: GPUScratch, sharedGateW: Gate8Buffers, H: Int, E: Int, K: Int) {
        let qp = _qmmPipeline!, gqp = _gqmmPipeline!, swp = _swigluPipeline!
        let Hin = moe.Hin, I = moe.I
        // ① gate qmm8: postNorm → gateLogits[E]
        enc.setComputePipelineState(_qmm8Pipeline!)
        enc.setBuffer(gate.wq, offset: 0, index: 0); enc.setBuffer(gate.sc, offset: 0, index: 1); enc.setBuffer(gate.bi, offset: 0, index: 2)
        enc.setBuffer(postNorm, offset: 0, index: 3); enc.setBuffer(sc.gateLogits, offset: 0, index: 4)
        var kk = Int32(H), nn = Int32(E); enc.setBytes(&kk, length: 4, index: 5); enc.setBytes(&nn, length: 4, index: 6)
        enc.dispatchThreadgroups(MTLSize(width: 1, height: E/8, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 2, depth: 1))
        // ② route_top8: gateLogits → moe.binds(inds), scores（single-thread; profile で skip 可）
        if !profSkipSingleThread {
            enc.setComputePipelineState(_routePipeline!)
            enc.setBuffer(sc.gateLogits, offset: 0, index: 0); enc.setBuffer(moe.binds, offset: 0, index: 1); enc.setBuffer(sc.scores, offset: 0, index: 2)
            var en = UInt32(E), kn = UInt32(K); enc.setBytes(&en, length: 4, index: 3); enc.setBytes(&kn, length: 4, index: 4)
            enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }
        if profSkipMoEExperts { return }   // gather/swiglu/shared/combine/final を skip(timing)
        // ③ experts: g/u gather(postNorm) → swiglu → d gather(per-expert)
        func encGather(_ wq: MTLBuffer, _ s: MTLBuffer, _ bi: MTLBuffer, _ xb: MTLBuffer, _ y: MTLBuffer, K kk: Int, N nn: Int, lhs: Bool) {
            enc.setComputePipelineState(gqp)
            enc.setBuffer(wq, offset: 0, index: 0); enc.setBuffer(s, offset: 0, index: 1); enc.setBuffer(bi, offset: 0, index: 2)
            enc.setBuffer(xb, offset: 0, index: 3); enc.setBuffer(moe.binds, offset: 0, index: 4); enc.setBuffer(y, offset: 0, index: 5)
            var k = Int32(kk), n = Int32(nn), l = UInt32(lhs ? 1 : 0)
            enc.setBytes(&k, length: 4, index: 6); enc.setBytes(&n, length: 4, index: 7); enc.setBytes(&l, length: 4, index: 8)
            enc.dispatchThreadgroups(MTLSize(width: 1, height: nn/8, depth: K), threadsPerThreadgroup: MTLSize(width: 32, height: 2, depth: 1))
        }
        func encQmm(_ wq: MTLBuffer, _ s: MTLBuffer, _ bi: MTLBuffer, _ xb: MTLBuffer, _ y: MTLBuffer, K kk: Int, N nn: Int) {
            enc.setComputePipelineState(qp)
            enc.setBuffer(wq, offset: 0, index: 0); enc.setBuffer(s, offset: 0, index: 1); enc.setBuffer(bi, offset: 0, index: 2)
            enc.setBuffer(xb, offset: 0, index: 3); enc.setBuffer(y, offset: 0, index: 4)
            var k = Int32(kk), n = Int32(nn); enc.setBytes(&k, length: 4, index: 5); enc.setBytes(&n, length: 4, index: 6)
            enc.dispatchThreadgroups(MTLSize(width: 1, height: nn/8, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 2, depth: 1))
        }
        func encSwiglu(_ g: MTLBuffer, _ u: MTLBuffer, _ h: MTLBuffer, _ total: Int) {
            enc.setComputePipelineState(swp)
            enc.setBuffer(g, offset: 0, index: 0); enc.setBuffer(u, offset: 0, index: 1); enc.setBuffer(h, offset: 0, index: 2)
            var t = UInt32(total); enc.setBytes(&t, length: 4, index: 3)
            enc.dispatchThreads(MTLSize(width: total, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(swp.maxTotalThreadsPerThreadgroup, 256), height: 1, depth: 1))
        }
        encGather(moe.swGW, moe.swGS, moe.swGB, postNorm, moe.g, K: Hin, N: I, lhs: false)
        encGather(moe.swUW, moe.swUS, moe.swUB, postNorm, moe.u, K: Hin, N: I, lhs: false)
        encSwiglu(moe.g, moe.u, moe.h, K * I)
        encGather(moe.swDW, moe.swDS, moe.swDB, moe.h, moe.d, K: I, N: Hin, lhs: true)
        // shared expert: sg/su qmm(postNorm) → swiglu → sharedY qmm
        encQmm(moe.shGW, moe.shGS, moe.shGB, postNorm, moe.sg, K: Hin, N: I)
        encQmm(moe.shUW, moe.shUS, moe.shUB, postNorm, moe.su, K: Hin, N: I)
        encSwiglu(moe.sg, moe.su, moe.shAct, I)
        encQmm(moe.shDW, moe.shDS, moe.shDB, moe.shAct, moe.sharedY, K: I, N: Hin)
        // ④ combine: d[K,H]·scores → y[H]
        enc.setComputePipelineState(_combinePipeline!)
        enc.setBuffer(moe.d, offset: 0, index: 0); enc.setBuffer(sc.scores, offset: 0, index: 1); enc.setBuffer(sc.y, offset: 0, index: 2)
        var ck = UInt32(K), cn = UInt32(Hin); enc.setBytes(&ck, length: 4, index: 3); enc.setBytes(&cn, length: 4, index: 4)
        enc.dispatchThreads(MTLSize(width: Hin, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(_combinePipeline!.maxTotalThreadsPerThreadgroup, 256), height: 1, depth: 1))
        // ⑤ shared gate scalar(sigmoid) → final_combine: combined = y + gateScale*sharedY
        if !profSkipSingleThread {
            enc.setComputePipelineState(_sharedGate8Pipeline!)
            enc.setBuffer(sharedGateW.wq, offset: 0, index: 0); enc.setBuffer(sharedGateW.sc, offset: 0, index: 1); enc.setBuffer(sharedGateW.bi, offset: 0, index: 2)
            enc.setBuffer(postNorm, offset: 0, index: 3); enc.setBuffer(sc.gateScale, offset: 0, index: 4)
            var sk = UInt32(H); enc.setBytes(&sk, length: 4, index: 5)
            enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }
        enc.setComputePipelineState(_finalCombinePipeline!)
        enc.setBuffer(sc.y, offset: 0, index: 0); enc.setBuffer(moe.sharedY, offset: 0, index: 1); enc.setBuffer(sc.gateScale, offset: 0, index: 2); enc.setBuffer(sc.combined, offset: 0, index: 3)
        var fh = UInt32(Hin); enc.setBytes(&fh, length: 4, index: 4)
        enc.dispatchThreads(MTLSize(width: Hin, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(_finalCombinePipeline!.maxTotalThreadsPerThreadgroup, 256), height: 1, depth: 1))
    }

    /// all-GPU 1 層分の常駐 buffer 束（QwispModel の cache から構築）。
    struct GPULayer {
        let nw: NormWeightBuffers; let gdn: GDNBuffers?; let attn: AttnBuffers?
        let moe: MoEBuffers; let gate: Gate8Buffers; let sharedGate: Gate8Buffers
    }

    /// ★ 多層 1-CB forward: 全層の mixer+MoE を **単一 serial encoder + 単一 command buffer** に詰め、
    ///   per-layer waitUntilCompleted を排除（GPU が層間 pipeline）。embed→hBuf 書込み済前提、hBuf に結果。
    ///   各層 MoE residual は次層 mixer 先頭で hBuf に畳む（pendingResid=scratch.combined）。最終層は末尾 flush。
    nonisolated(unsafe) static var fusedNumCB = 1   // 層を G 個の CB に分割。実測 G>1 は inter-CB gap で逆効果＝1 が最速

    static func fusedForwardGPU(hBuf: MTLBuffer, layers: [GPULayer], scratch: GPUScratch,
                                H: Int, E: Int, K: Int, eps: Float, decode: Bool = false, pos: Int = 0,
                                finalNormW: MTLBuffer? = nil) {
        guard let (_, queue) = ensure() else { return }
        // ★ 多 command buffer: 各 CB を commit(待たず)→GPU が CB_g を実行中に CPU が CB_{g+1} を encode＝
        //   CPU-encode bubble を GPU-exec の裏に隠す。同一 queue の CB は commit 順に serial 実行＋メモリ整合。
        let G = max(1, min(fusedNumCB, layers.count))
        let per = (layers.count + G - 1) / G
        var cbs: [MTLCommandBuffer] = []
        for g in 0 ..< G {
            let lo = g * per, hi = min((g + 1) * per, layers.count)
            if lo >= hi { break }
            let cb = queue.makeCommandBuffer()!; let enc = cb.makeComputeCommandEncoder()!
            for i in lo ..< hi {
                let L = layers[i]
                encodeMixerHalf(enc, hBuf: hBuf, nw: L.nw, postNormBuf: scratch.postNorm,
                                gdn: L.gdn, attn: L.attn, H: H, eps: eps, pendingResid: i == 0 ? nil : scratch.combined,
                                decode: decode, pos: pos)
                encodeMoEGPU(enc, postNorm: scratch.postNorm, gate: L.gate, moe: L.moe, sc: scratch,
                             sharedGateW: L.sharedGate, H: H, E: E, K: K)
            }
            if hi == layers.count {   // 最終 CB に最終 residual + final norm
                encodeResidAdd(enc, h: hBuf, r: scratch.combined, total: H)
                if let fnw = finalNormW { encodeRms(enc, src: hBuf, w: fnw, out: scratch.normed, D: H, eps: eps) }
            }
            enc.endEncoding(); cb.commit()    // ★待たずに commit（次 CB の encode と GPU-exec を overlap）
            cbs.append(cb)
        }
        cbs.last?.waitUntilCompleted()        // 最後だけ wait
        if let f = cbs.first, let l = cbs.last { lastGPUExecMs = (l.gpuEndTime - f.gpuStartTime) * 1000.0 }
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

    // ── attention 層 round-trip assembly（decode S=1, cold cache）─────────────────
    struct AttnRawWeights {
        let qWq: MLXArray, qSc: MLXArray, qBi: MLXArray   // q_proj → [H, 2*headDim] per head
        let kWq: MLXArray, kSc: MLXArray, kBi: MLXArray
        let vWq: MLXArray, vSc: MLXArray, vBi: MLXArray
        let oWq: MLXArray, oSc: MLXArray, oBi: MLXArray
        let qNorm: MLXArray, kNorm: MLXArray
    }
    /// attention decode 1 step を raw kernel で組む（cold cache: S=1, offset=0）。
    /// q/k/v proj → q/k-norm(weight) → RoPE → SDPA(.none) → gated(out*sigmoid(gate)) → o_proj。
    /// promoteF32: q_norm/k_norm が f32 のとき qk-norm が f32 昇格→RoPE/SDPA も f32（cascade）。
    static func attnLayerRaw(_ x: MLXArray, _ w: AttnRawWeights, promoteF32: Bool,
                             numHeads: Int = 16, numKV: Int = 2, headDim: Int = 256,
                             ropeDim: Int = 64, ropeBase: Float = 1e7, eps: Float = 1e-6) -> MLXArray? {
        let H = x.dim(-1)
        let x2 = x.reshaped([1, H])
        let scale = Float(pow(Double(headDim), -0.5))
        let qd2 = 2 * headDim
        guard let qOut = qmm(x2, w.qWq, scales: w.qSc, biases: w.qBi, M: 1, K: H, N: numHeads * qd2),
              let keys = qmm(x2, w.kWq, scales: w.kSc, biases: w.kBi, M: 1, K: H, N: numKV * headDim),
              let values = qmm(x2, w.vWq, scales: w.vSc, biases: w.vBi, M: 1, K: H, N: numKV * headDim)
        else { return nil }
        let qOutR = qOut.reshaped([numHeads, qd2])
        let queries = qOutR[0..., 0 ..< headDim]                  // [H, headDim]
        let gate = qOutR[0..., headDim...].reshaped([1, numHeads * headDim])  // [1, H*headDim]
        // qk-norm（weight 有り, promoteF32 で f32 昇格）
        let wT: DType = promoteF32 ? .float32 : .float16
        guard let qN = rmsNorm(queries, w.qNorm.asType(wT), eps: eps, D: headDim, promoteF32: promoteF32),
              let kN = rmsNorm(keys.reshaped([numKV, headDim]), w.kNorm.asType(wT), eps: eps, D: headDim, promoteF32: promoteF32)
        else { return nil }
        // RoPE（offset 0）→ SDPA（S=1）。promoteF32 なら全 raw f32 variant（cascade）。
        let qRot: MLXArray, kRot: MLXArray, attnOut: MLXArray
        if promoteF32 {
            guard let qr = rope(qN, headDim: headDim, ropeDim: ropeDim, base: ropeBase, offset: 0, xF32: true),
                  let kr = rope(kN, headDim: headDim, ropeDim: ropeDim, base: ropeBase, offset: 0, xF32: true),
                  let ao = sdpaDecode(qr, kr.reshaped([numKV, 1, headDim]), values.reshaped([numKV, 1, headDim]),
                                      H: numHeads, KV: numKV, D: headDim, S: 1, scale: scale, inF32: true) else { return nil }
            qRot = qr; kRot = kr; attnOut = ao
        } else {
            guard let qr = rope(qN, headDim: headDim, ropeDim: ropeDim, base: ropeBase, offset: 0),
                  let kr = rope(kN, headDim: headDim, ropeDim: ropeDim, base: ropeBase, offset: 0),
                  let ao = sdpaDecode(qr, kr.reshaped([numKV, 1, headDim]), values.reshaped([numKV, 1, headDim]),
                                      H: numHeads, KV: numKV, D: headDim, S: 1, scale: scale) else { return nil }
            qRot = qr; kRot = kr; attnOut = ao
        }
        // gated = output * sigmoid(gate)。promoteF32 では output f32・sigmoid(gate) は f16(MLX 一致)→ gated f32、o_proj は f32 qmm。
        if promoteF32 {
            let outR = attnOut.reshaped([1, numHeads * headDim])                       // f32（SDPA f32 出力）
            let gated = outR.asType(.float32) * MLX.sigmoid(gate)                       // f32 * sigmoid(gate_f16) → f32
            return qmm(gated, w.oWq, scales: w.oSc, biases: w.oBi, M: 1, K: numHeads * headDim, N: H, xF32: true)
        } else {
            // f16 経路: MLX は output_f16 * sigmoid(gate_f16) を f16 で計算（f32 にしない）。
            let outR = attnOut.reshaped([1, numHeads * headDim]).asType(.float16)
            let gated = outR * MLX.sigmoid(gate)
            return qmm(gated, w.oWq, scales: w.oSc, biases: w.oBi, M: 1, K: numHeads * headDim, N: H)
        }
    }

    /// attention 計算の常駐 buffer（重み＋中間＝一度だけ確保）。decode S=1, cold cache, f16 経路。
    struct AttnBuffers {
        let H, numHeads, numKV, headDim, qd2, ropeDim, maxLen: Int
        let ropeBase, scale, eps: Float
        let bx: MTLBuffer
        let qW, qS, qB, kW, kS, kB, vW, vS, vB, oW, oS, oB, qNormW, kNormW: MTLBuffer
        let qOut, keys, values, queries, qN, kN, qRot, kRot, attnOut, gated, outBuf: MTLBuffer
        let kCache, vCache: MTLBuffer   // decode 用 KV cache [numKV, maxLen, headDim]（post-RoPE k / raw v）
    }

    /// attn 重み・中間 buffer を一度だけ確保（GDN SE と同型）。real model は q/k_norm F16 → f16 経路のみ。
    /// maxLen>0 で decode 用 KV cache を確保（prefill+生成の最大長）。
    static func prepareAttnBuffers(_ w: AttnRawWeights,
                                   numHeads: Int = 16, numKV: Int = 2, headDim: Int = 256,
                                   ropeDim: Int = 64, ropeBase: Float = 1e7, eps: Float = 1e-6,
                                   H: Int = 2048, maxLen: Int = 0) -> AttnBuffers? {
        guard let (device, _) = ensure(), ensureAuxPipelines() else { return nil }
        let qd2 = 2 * headDim
        func mtl(_ a: MLXArray, _ t: DType) -> MTLBuffer? { a.asType(t).asMTLBuffer(device: device, noCopy: false) }
        func mk(_ bytes: Int) -> MTLBuffer { device.makeBuffer(length: bytes, options: .storageModeShared)! }
        guard let qW = w.qWq.asMTLBuffer(device: device, noCopy: false), let qS = mtl(w.qSc, .float16), let qB = mtl(w.qBi, .float16),
              let kW = w.kWq.asMTLBuffer(device: device, noCopy: false), let kS = mtl(w.kSc, .float16), let kB = mtl(w.kBi, .float16),
              let vW = w.vWq.asMTLBuffer(device: device, noCopy: false), let vS = mtl(w.vSc, .float16), let vB = mtl(w.vBi, .float16),
              let oW = w.oWq.asMTLBuffer(device: device, noCopy: false), let oS = mtl(w.oSc, .float16), let oB = mtl(w.oBi, .float16),
              let qNormW = mtl(w.qNorm, .float16), let kNormW = mtl(w.kNorm, .float16)
        else { print("[raw-attn-se] weight buffer nil"); return nil }
        let qDim = numHeads * qd2, kvDim = numKV * headDim, vDim = numHeads * headDim
        let cacheLen = max(1, maxLen)
        return AttnBuffers(
            H: H, numHeads: numHeads, numKV: numKV, headDim: headDim, qd2: qd2, ropeDim: ropeDim, maxLen: cacheLen,
            ropeBase: ropeBase, scale: Float(pow(Double(headDim), -0.5)), eps: eps,
            bx: mk(H * 2),
            qW: qW, qS: qS, qB: qB, kW: kW, kS: kS, kB: kB, vW: vW, vS: vS, vB: vB, oW: oW, oS: oS, oB: oB,
            qNormW: qNormW, kNormW: kNormW,
            qOut: mk(qDim * 2), keys: mk(kvDim * 2), values: mk(kvDim * 2), queries: mk(vDim * 2),
            qN: mk(vDim * 2), kN: mk(kvDim * 2), qRot: mk(vDim * 2), kRot: mk(kvDim * 2),
            attnOut: mk(vDim * 2), gated: mk(vDim * 2), outBuf: mk(H * 2),
            kCache: mk(numKV * cacheLen * headDim * 2), vCache: mk(numKV * cacheLen * headDim * 2))
    }

    /// ★ attention 1 層を **単一 command buffer + 単一 encoder** で連結（GDN SE 同型）。decode S=1, cold cache, f16。
    ///   q/k/v proj→extract_q→qk-norm→RoPE→SDPA→sigmoid_mul→o_proj。round-trip attnLayerRaw(pf=false) と bit-exact。
    ///   pipeline(qmm/rms/rope/sdpa)は事前 warm 前提（rawForward の attnLayerRaw が warm する）。
    static func checkAttnPipelines() -> Bool {
        guard _qmmPipeline != nil, _rmsPipeline != nil, _ropePipeline != nil, _sdpaPipeline != nil,
              _extractQPipeline != nil, _sigmoidMulPipeline != nil else {
            print("[raw-attn-se] pipeline 未 warm（先に attnLayerRaw を呼ぶ）"); return false
        }
        return true
    }

    static func attnLayerSingleEncoder(_ x: MLXArray, _ b: AttnBuffers) -> MLXArray? {
        guard let (_, queue) = ensure(), checkAttnPipelines() else { return nil }
        let H = b.H
        // x を常駐 buffer に書込み（per-call, H*2 bytes）
        let xf = x.reshaped([H]).asType(.float16).asArray(Float16.self)
        b.bx.contents().bindMemory(to: Float16.self, capacity: H).update(from: xf, count: H)
        let cb = queue.makeCommandBuffer()!; let enc = cb.makeComputeCommandEncoder()!
        encodeAttnBody(enc, b)                            // b.bx → b.outBuf
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        let ptr = b.outBuf.contents().bindMemory(to: Float16.self, capacity: H)
        return MLXArray(Array(UnsafeBufferPointer(start: ptr, count: H)), [1, H])
    }

    /// attention 1 層の kernel chain を既存 encoder に encode（b.bx → b.outBuf）。input_norm/residual/post_norm 非含。
    /// decode=true: RoPE offset=pos, post-RoPE k と raw v を KV cache(seq pos)に書き SDPA を N=pos+1 で。
    static func encodeAttnBody(_ enc: MTLComputeCommandEncoder, _ b: AttnBuffers, decode: Bool = false, pos: Int = 0) {
        let qp = _qmmPipeline!, rp = _rmsPipeline!, rope = _ropePipeline!, sdpa = _sdpaPipeline!
        let exq = _extractQPipeline!, sgm = _sigmoidMulPipeline!
        let H = b.H, numHeads = b.numHeads, numKV = b.numKV, headDim = b.headDim, qd2 = b.qd2
        let qDim = numHeads * qd2, kvDim = numKV * headDim, vDim = numHeads * headDim
        func encQmm(_ wq: MTLBuffer, _ sc: MTLBuffer, _ bi: MTLBuffer, _ xb: MTLBuffer, _ y: MTLBuffer, K: Int, N: Int) {
            enc.setComputePipelineState(qp)
            enc.setBuffer(wq, offset: 0, index: 0); enc.setBuffer(sc, offset: 0, index: 1)
            enc.setBuffer(bi, offset: 0, index: 2); enc.setBuffer(xb, offset: 0, index: 3); enc.setBuffer(y, offset: 0, index: 4)
            var kk = Int32(K), nn = Int32(N); enc.setBytes(&kk, length: 4, index: 5); enc.setBytes(&nn, length: 4, index: 6)
            enc.dispatchThreadgroups(MTLSize(width: 1, height: N / 8, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 2, depth: 1))
        }
        func encRms(_ xb: MTLBuffer, _ wb: MTLBuffer, _ ob: MTLBuffer, rows: Int, D: Int) {
            enc.setComputePipelineState(rp)
            enc.setBuffer(xb, offset: 0, index: 0); enc.setBuffer(wb, offset: 0, index: 1); enc.setBuffer(ob, offset: 0, index: 2)
            var ee = b.eps, asz = UInt32(D), ws = UInt32(1)
            enc.setBytes(&ee, length: 4, index: 3); enc.setBytes(&asz, length: 4, index: 4); enc.setBytes(&ws, length: 4, index: 5)
            let tg = ((((D + 3) / 4) + 31) / 32) * 32
            enc.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        }
        func encRope(_ xb: MTLBuffer, _ ob: MTLBuffer, rows: Int) {
            enc.setComputePipelineState(rope)
            enc.setBuffer(xb, offset: 0, index: 0); enc.setBuffer(ob, offset: 0, index: 1)
            var hd = UInt32(headDim), rd = UInt32(b.ropeDim), base = b.ropeBase, p = Float(pos)   // decode: offset=pos
            enc.setBytes(&hd, length: 4, index: 2); enc.setBytes(&rd, length: 4, index: 3)
            enc.setBytes(&base, length: 4, index: 4); enc.setBytes(&p, length: 4, index: 5)
            let total = rows * headDim, tgw = min(rope.maxTotalThreadsPerThreadgroup, 256)
            enc.dispatchThreads(MTLSize(width: total, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: tgw, height: 1, depth: 1))
        }
        func encWriteKV(_ src: MTLBuffer, _ cache: MTLBuffer) {
            enc.setComputePipelineState(_writeKVPipeline!)
            enc.setBuffer(src, offset: 0, index: 0); enc.setBuffer(cache, offset: 0, index: 1)
            var kv = UInt32(numKV), d = UInt32(headDim), ml = UInt32(b.maxLen), p = UInt32(pos)
            enc.setBytes(&kv, length: 4, index: 2); enc.setBytes(&d, length: 4, index: 3)
            enc.setBytes(&ml, length: 4, index: 4); enc.setBytes(&p, length: 4, index: 5)
            enc.dispatchThreads(MTLSize(width: numKV * headDim, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(_writeKVPipeline!.maxTotalThreadsPerThreadgroup, 256), height: 1, depth: 1))
        }
        func encElem(_ p: MTLComputePipelineState, _ total: Int, _ bind: () -> Void) {
            enc.setComputePipelineState(p); bind()
            enc.dispatchThreads(MTLSize(width: total, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(p.maxTotalThreadsPerThreadgroup, 256), height: 1, depth: 1))
        }
        // ① q/k/v proj
        encQmm(b.qW, b.qS, b.qB, b.bx, b.qOut, K: H, N: qDim)
        encQmm(b.kW, b.kS, b.kB, b.bx, b.keys, K: H, N: kvDim)
        encQmm(b.vW, b.vS, b.vB, b.bx, b.values, K: H, N: kvDim)
        // ② query 抽出（qOut[:,0:headDim] → queries contiguous）→ qk-norm
        encElem(exq, vDim) {
            enc.setBuffer(b.qOut, offset: 0, index: 0); enc.setBuffer(b.queries, offset: 0, index: 1)
            var hd = UInt32(headDim), q2 = UInt32(qd2), tot = UInt32(vDim)
            enc.setBytes(&hd, length: 4, index: 2); enc.setBytes(&q2, length: 4, index: 3); enc.setBytes(&tot, length: 4, index: 4)
        }
        encRms(b.queries, b.qNormW, b.qN, rows: numHeads, D: headDim)
        encRms(b.keys, b.kNormW, b.kN, rows: numKV, D: headDim)
        // ③ RoPE（decode: offset=pos）。decode は post-RoPE k と raw v を KV cache(seq pos)へ。
        encRope(b.qN, b.qRot, rows: numHeads)
        encRope(b.kN, b.kRot, rows: numKV)
        if decode { encWriteKV(b.kRot, b.kCache); encWriteKV(b.values, b.vCache) }
        // ④ SDPA。cold: N=1（kRot/values）。decode: N=pos+1（kCache/vCache, head_stride=maxLen*D）。
        let keysBuf = decode ? b.kCache : b.kRot, valsBuf = decode ? b.vCache : b.values
        let seqStrideDim = decode ? b.maxLen * headDim : headDim
        enc.setComputePipelineState(sdpa)
        enc.setBuffer(b.qRot, offset: 0, index: 0); enc.setBuffer(keysBuf, offset: 0, index: 1)
        enc.setBuffer(valsBuf, offset: 0, index: 2); enc.setBuffer(b.attnOut, offset: 0, index: 3)
        var gqa = Int32(numHeads / numKV), nn = Int32(decode ? pos + 1 : 1), khs = Int32(seqStrideDim), kss = Int32(headDim)
        var vhs = Int32(seqStrideDim), vss = Int32(headDim), sc = b.scale
        enc.setBytes(&gqa, length: 4, index: 4); enc.setBytes(&nn, length: 4, index: 5)
        enc.setBytes(&khs, length: 4, index: 6); enc.setBytes(&kss, length: 4, index: 7)
        enc.setBytes(&vhs, length: 4, index: 8); enc.setBytes(&vss, length: 4, index: 9); enc.setBytes(&sc, length: 4, index: 10)
        enc.dispatchThreadgroups(MTLSize(width: numHeads, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1024, height: 1, depth: 1))
        // ⑤ gated = attnOut * sigmoid(gate)（gate は qOut から strided）→ o_proj
        encElem(sgm, vDim) {
            enc.setBuffer(b.attnOut, offset: 0, index: 0); enc.setBuffer(b.qOut, offset: 0, index: 1); enc.setBuffer(b.gated, offset: 0, index: 2)
            var hd = UInt32(headDim), q2 = UInt32(qd2), tot = UInt32(vDim)
            enc.setBytes(&hd, length: 4, index: 3); enc.setBytes(&q2, length: 4, index: 4); enc.setBytes(&tot, length: 4, index: 5)
        }
        encQmm(b.oW, b.oS, b.oB, b.gated, b.outBuf, K: vDim, N: H)
    }

    /// 検証: attention 1 層 raw assembly vs MLX AttentionLayer（同量子化, decode S=1）。
    /// - env: QWISP_RUN=raw-attn-test / QWISP_ATTN_REF(既定 /tmp/qwisp_attn_ref.safetensors)
    public static func runAttnLayerTest() -> String {
        let refPath = ProcessInfo.processInfo.environment["QWISP_ATTN_REF"] ?? "/tmp/qwisp_attn_ref.safetensors"
        guard let r = try? loadArrays(url: URL(fileURLWithPath: refPath)),
              let qp = r["q_proj"], let kp = r["k_proj"], let vp = r["v_proj"], let op = r["o_proj"],
              let qn = r["q_norm"], let kn = r["k_norm"] else { return "[raw-attn] ref キー不足" }
        let H = qp.dim(-1)
        let x = MLXRandom.normal([1, 1, H]).asType(.float16)
        func quant(_ wt: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
            let (q, s, b) = MLX.quantized(wt.asType(.float16), groupSize: 64, bits: 4, mode: .affine); return (q, s, b!)
        }
        let (qWq, qSc, qBi) = quant(qp), (kWq, kSc, kBi) = quant(kp)
        let (vWq, vSc, vBi) = quant(vp), (oWq, oSc, oBi) = quant(op)
        let rw = AttnRawWeights(qWq: qWq, qSc: qSc, qBi: qBi, kWq: kWq, kSc: kSc, kBi: kBi,
                                vWq: vWq, vSc: vSc, vBi: vBi, oWq: oWq, oSc: oSc, oBi: oBi, qNorm: qn, kNorm: kn)
        // MLX 参照（同量子化, decode S=1, cold cache）
        let refLayer = AttentionLayer(
            numHeads: 16, numKVHeads: 2, headDim: 256, ropeDim: 64, ropeBase: 1e7, eps: 1e-6,
            qProj: .quantized(qWq, qSc, qBi, 4), kProj: .quantized(kWq, kSc, kBi, 4),
            vProj: .quantized(vWq, vSc, vBi, 4), oProj: .quantized(oWq, oSc, oBi, 4),
            qNorm: qn, kNorm: kn)
        let ref = refLayer(x, cache: KVCache()); ref.eval()
        var out = "[raw-attn-test] attention 1 層 raw assembly vs MLX（同量子化, decode S=1）"
        for pf in [false, true] {
            guard let got = attnLayerRaw(x, rw, promoteF32: pf) else { out += "\n  promoteF32=\(pf): 実行失敗"; continue }
            got.eval()
            let rel = relErr(got.reshaped([got.size]), ref.reshaped([ref.size]))
            out += String(format: "\n  promoteF32=%@: rel=%.3e %@", pf ? "true " : "false",
                          rel, rel == 0 ? "✅ TRUE bit-exact" : (rel < 2e-3 ? "△ 近似" : "❌"))
        }
        // ★ attn SE（single-encoder, f16）vs round-trip raw（pf=false）: bit-exact 不変条件 + 速度。
        //   real model は q/k_norm F16 → SE は f16 経路で round-trip(pf=false) と byte 一致すべき。
        let rtF16 = attnLayerRaw(x, rw, promoteF32: false)!; rtF16.eval()
        if let buf = prepareAttnBuffers(rw), let se = attnLayerSingleEncoder(x, buf) {
            se.eval()
            let seRel = relErr(se.reshaped([se.size]), rtF16.reshaped([rtF16.size]))
            func now() -> Double { Double(DispatchTime.now().uptimeNanoseconds) / 1e6 }
            let reps = 300
            for _ in 0..<5 { _ = attnLayerSingleEncoder(x, buf)?.eval() }
            var t0 = now(); for _ in 0..<reps { _ = attnLayerSingleEncoder(x, buf)?.eval() }; let seMs = (now()-t0)/Double(reps)
            for _ in 0..<5 { _ = attnLayerRaw(x, rw, promoteF32: false)?.eval() }
            t0 = now(); for _ in 0..<reps { _ = attnLayerRaw(x, rw, promoteF32: false)?.eval() }; let rtMs = (now()-t0)/Double(reps)
            out += String(format: "\n  ── attn SE（single-encoder, f16）vs round-trip ──\n   SE-vs-RT rel=%.3e %@  SE=%.3fms RT=%.3fms → %.2fx",
                          seRel, seRel == 0 ? "✅ bit-exact" : (seRel < 2e-3 ? "△ near" : "❌"), seMs, rtMs, rtMs/Swift.max(0.001, seMs))
        } else { out += "\n  attn SE: prepare/実行 失敗" }
        return out
    }

    /// MoE expert 計算の常駐 buffer（重み＋中間＝一度だけ確保）。routing(inds)/combine は外部(MLX)。
    struct MoEBuffers {
        let Hin, I, topK: Int
        let bx, binds: MTLBuffer
        let swGW, swGS, swGB, swUW, swUS, swUB, swDW, swDS, swDB: MTLBuffer
        let shGW, shGS, shGB, shUW, shUS, shUB, shDW, shDS, shDB: MTLBuffer
        let g, u, h, d, sg, su, shAct, sharedY: MTLBuffer
    }
    static func prepareMoEBuffers(swG: (MLXArray, MLXArray, MLXArray), swU: (MLXArray, MLXArray, MLXArray), swD: (MLXArray, MLXArray, MLXArray),
                                  shG: (MLXArray, MLXArray, MLXArray), shU: (MLXArray, MLXArray, MLXArray), shD: (MLXArray, MLXArray, MLXArray),
                                  Hin: Int, I: Int, topK: Int) -> MoEBuffers? {
        guard let (device, _) = ensure(), ensureAuxPipelines() else { return nil }
        func wb(_ a: MLXArray) -> MTLBuffer? { a.asMTLBuffer(device: device, noCopy: false) }
        func fb(_ a: MLXArray) -> MTLBuffer? { a.asType(.float16).asMTLBuffer(device: device, noCopy: false) }
        func mk(_ n: Int) -> MTLBuffer { device.makeBuffer(length: n, options: .storageModeShared)! }
        guard let swGW = wb(swG.0), let swGS = fb(swG.1), let swGB = fb(swG.2),
              let swUW = wb(swU.0), let swUS = fb(swU.1), let swUB = fb(swU.2),
              let swDW = wb(swD.0), let swDS = fb(swD.1), let swDB = fb(swD.2),
              let shGW = wb(shG.0), let shGS = fb(shG.1), let shGB = fb(shG.2),
              let shUW = wb(shU.0), let shUS = fb(shU.1), let shUB = fb(shU.2),
              let shDW = wb(shD.0), let shDS = fb(shD.1), let shDB = fb(shD.2) else { return nil }
        return MoEBuffers(Hin: Hin, I: I, topK: topK, bx: mk(Hin * 2), binds: mk(topK * 4),
            swGW: swGW, swGS: swGS, swGB: swGB, swUW: swUW, swUS: swUS, swUB: swUB, swDW: swDW, swDS: swDS, swDB: swDB,
            shGW: shGW, shGS: shGS, shGB: shGB, shUW: shUW, shUS: shUS, shUB: shUB, shDW: shDW, shDS: shDS, shDB: shDB,
            g: mk(topK * I * 2), u: mk(topK * I * 2), h: mk(topK * I * 2), d: mk(topK * Hin * 2),
            sg: mk(I * 2), su: mk(I * 2), shAct: mk(I * 2), sharedY: mk(Hin * 2))
    }

    /// ★ MoE expert+shared を **単一 encoder** で連結（gather g/u→swiglu→gather d(per-expert) / 並行 shared）。
    ///   routing(inds, scores) は外部(MLX)、combine も外部。d[topK,Hin] と sharedY[Hin] を返す。
    static func moeExpertSingleEncoder(_ x: MLXArray, _ inds: MLXArray, _ b: MoEBuffers) -> (MLXArray, MLXArray)? {
        guard let (_, queue) = ensure() else { return nil }
        guard let gqp = _gqmmPipeline, let qp = _qmmPipeline, let swp = _swigluPipeline else { return nil }
        let Hin = b.Hin, I = b.I, K = b.topK
        // x, inds を常駐 buffer に書込み
        let xf = x.reshaped([Hin]).asType(.float16).asArray(Float16.self)
        b.bx.contents().bindMemory(to: Float16.self, capacity: Hin).update(from: xf, count: Hin)
        let indsI = inds.reshaped([K]).asType(.int32).asArray(Int32.self)
        b.binds.contents().bindMemory(to: Int32.self, capacity: K).update(from: indsI, count: K)
        let cb = queue.makeCommandBuffer()!; let enc = cb.makeComputeCommandEncoder()!
        func encGather(_ wq: MTLBuffer, _ sc: MTLBuffer, _ bi: MTLBuffer, _ xb: MTLBuffer, _ y: MTLBuffer, K kk: Int, N nn: Int, lhs: Bool) {
            enc.setComputePipelineState(gqp)
            enc.setBuffer(wq, offset: 0, index: 0); enc.setBuffer(sc, offset: 0, index: 1); enc.setBuffer(bi, offset: 0, index: 2)
            enc.setBuffer(xb, offset: 0, index: 3); enc.setBuffer(b.binds, offset: 0, index: 4); enc.setBuffer(y, offset: 0, index: 5)
            var k = Int32(kk), n = Int32(nn), l = UInt32(lhs ? 1 : 0)
            enc.setBytes(&k, length: 4, index: 6); enc.setBytes(&n, length: 4, index: 7); enc.setBytes(&l, length: 4, index: 8)
            enc.dispatchThreadgroups(MTLSize(width: 1, height: nn / 8, depth: K), threadsPerThreadgroup: MTLSize(width: 32, height: 2, depth: 1))
        }
        func encQmm(_ wq: MTLBuffer, _ sc: MTLBuffer, _ bi: MTLBuffer, _ xb: MTLBuffer, _ y: MTLBuffer, K kk: Int, N nn: Int) {
            enc.setComputePipelineState(qp)
            enc.setBuffer(wq, offset: 0, index: 0); enc.setBuffer(sc, offset: 0, index: 1); enc.setBuffer(bi, offset: 0, index: 2)
            enc.setBuffer(xb, offset: 0, index: 3); enc.setBuffer(y, offset: 0, index: 4)
            var k = Int32(kk), n = Int32(nn); enc.setBytes(&k, length: 4, index: 5); enc.setBytes(&n, length: 4, index: 6)
            enc.dispatchThreadgroups(MTLSize(width: 1, height: nn / 8, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 2, depth: 1))
        }
        func encSwiglu(_ gb: MTLBuffer, _ ub: MTLBuffer, _ hb: MTLBuffer, total: Int) {
            enc.setComputePipelineState(swp)
            enc.setBuffer(gb, offset: 0, index: 0); enc.setBuffer(ub, offset: 0, index: 1); enc.setBuffer(hb, offset: 0, index: 2)
            var t = UInt32(total); enc.setBytes(&t, length: 4, index: 3)
            enc.dispatchThreads(MTLSize(width: total, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(swp.maxTotalThreadsPerThreadgroup, 256), height: 1, depth: 1))
        }
        // routed experts: g/u gather → swiglu → d gather(per-expert)
        encGather(b.swGW, b.swGS, b.swGB, b.bx, b.g, K: Hin, N: I, lhs: false)
        encGather(b.swUW, b.swUS, b.swUB, b.bx, b.u, K: Hin, N: I, lhs: false)
        encSwiglu(b.g, b.u, b.h, total: K * I)
        encGather(b.swDW, b.swDS, b.swDB, b.h, b.d, K: I, N: Hin, lhs: true)
        // shared expert: sg/su qmm → swiglu → sharedY qmm
        encQmm(b.shGW, b.shGS, b.shGB, b.bx, b.sg, K: Hin, N: I)
        encQmm(b.shUW, b.shUS, b.shUB, b.bx, b.su, K: Hin, N: I)
        encSwiglu(b.sg, b.su, b.shAct, total: I)
        encQmm(b.shDW, b.shDS, b.shDB, b.shAct, b.sharedY, K: I, N: Hin)
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        let dp = b.d.contents().bindMemory(to: Float16.self, capacity: K * Hin)
        let d = MLXArray(Array(UnsafeBufferPointer(start: dp, count: K * Hin)), [K, Hin])
        let sp = b.sharedY.contents().bindMemory(to: Float16.self, capacity: Hin)
        let sharedY = MLXArray(Array(UnsafeBufferPointer(start: sp, count: Hin)), [1, Hin])
        return (d, sharedY)
    }

    /// raw swiglu: h=(g*sigmoid(g))*u（全 f16）。round-trip(個別 cmd buffer)。
    static func swigluRaw(_ g: MLXArray, _ u: MLXArray) -> MLXArray? {
        guard let (device, queue) = ensure(), ensureAuxPipelines(), let p = _swigluPipeline else { return nil }
        let total = g.size
        guard let bg = g.asType(.float16).asMTLBuffer(device: device, noCopy: false),
              let bu = u.asType(.float16).asMTLBuffer(device: device, noCopy: false) else { return nil }
        let outBuf = device.makeBuffer(length: total * 2, options: .storageModeShared)!
        let cb = queue.makeCommandBuffer()!; let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(p)
        enc.setBuffer(bg, offset: 0, index: 0); enc.setBuffer(bu, offset: 0, index: 1); enc.setBuffer(outBuf, offset: 0, index: 2)
        var t = UInt32(total); enc.setBytes(&t, length: 4, index: 3)
        enc.dispatchThreads(MTLSize(width: total, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(p.maxTotalThreadsPerThreadgroup, 256), height: 1, depth: 1))
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        let ptr = outBuf.contents().bindMemory(to: Float16.self, capacity: total)
        return MLXArray(Array(UnsafeBufferPointer(start: ptr, count: total)), g.shape)
    }
    /// raw combine: y[n]=Σ_k (d[k,n]*scores[k])（全 f16, MLX の f16 sum と一致）。
    static func combineRaw(_ d: MLXArray, _ scores: MLXArray, K: Int, N: Int) -> MLXArray? {
        guard let (device, queue) = ensure(), ensureAuxPipelines(), let p = _combinePipeline else { return nil }
        guard let bd = d.asType(.float16).asMTLBuffer(device: device, noCopy: false),
              let bs = scores.asType(.float16).asMTLBuffer(device: device, noCopy: false) else { return nil }
        let outBuf = device.makeBuffer(length: N * 2, options: .storageModeShared)!
        let cb = queue.makeCommandBuffer()!; let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(p)
        enc.setBuffer(bd, offset: 0, index: 0); enc.setBuffer(bs, offset: 0, index: 1); enc.setBuffer(outBuf, offset: 0, index: 2)
        var kk = UInt32(K), nn = UInt32(N); enc.setBytes(&kk, length: 4, index: 3); enc.setBytes(&nn, length: 4, index: 4)
        enc.dispatchThreads(MTLSize(width: N, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(p.maxTotalThreadsPerThreadgroup, 256), height: 1, depth: 1))
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        let ptr = outBuf.contents().bindMemory(to: Float16.self, capacity: N)
        return MLXArray(Array(UnsafeBufferPointer(start: ptr, count: N)), [N])
    }

    /// 検証: MoE block の raw gather 経路（routing/combine/shared は MLX glue, 4bit gather g/u/d は raw）vs MLX MoEBlock。
    /// down gather は per-expert 入力(h[ki])。bit-exact なら raw gather が full MoE に組み込み可能。
    /// - env: QWISP_RUN=raw-moe-test / QWISP_DEC0_REF(既定 /tmp/qwisp_dec0_ref.safetensors)
    public static func runMoeBlockTest() -> String {
        let refPath = ProcessInfo.processInfo.environment["QWISP_DEC0_REF"] ?? "/tmp/qwisp_dec0_ref.safetensors"
        guard let r = try? loadArrays(url: URL(fileURLWithPath: refPath)) else { return "[raw-moe] ref 読込失敗 \(refPath)" }
        func q(_ n: String, _ bits: Int) -> Proj? {
            guard let w = r["\(n).weight"], let s = r["\(n).scales"], let b = r["\(n).biases"] else { return nil }
            return .quantized(w, s, b, bits)
        }
        guard let gate = q("gate", 8), let shGate = q("shared_expert.gate_proj", 4),
              let shUp = q("shared_expert.up_proj", 4), let shDown = q("shared_expert.down_proj", 4),
              let sharedGate = q("shared_expert_gate", 8),
              let swGW = r["switch_mlp.gate_proj.weight"], let swGS = r["switch_mlp.gate_proj.scales"], let swGB = r["switch_mlp.gate_proj.biases"],
              let swUW = r["switch_mlp.up_proj.weight"], let swUS = r["switch_mlp.up_proj.scales"], let swUB = r["switch_mlp.up_proj.biases"],
              let swDW = r["switch_mlp.down_proj.weight"], let swDS = r["switch_mlp.down_proj.scales"], let swDB = r["switch_mlp.down_proj.biases"]
        else { return "[raw-moe] ref キー不足（gate/switch_mlp/shared_expert）" }
        // shared expert の raw 量子化タプル（4bit, my qmm 用）
        func tup(_ n: String) -> (MLXArray, MLXArray, MLXArray) { (r["\(n).weight"]!, r["\(n).scales"]!, r["\(n).biases"]!) }
        let sgW = tup("shared_expert.gate_proj"), suW = tup("shared_expert.up_proj"), sdW = tup("shared_expert.down_proj")
        let Hin = swGS.dim(-1) * 64                                          // K = groups*gs = 2048
        let topK = 8, E = 256, I = swGW.dim(-2)                              // I=intermediate=512
        let blk = MoEBlock(topK: topK, numExperts: E, normTopk: true, expertBits: 4, gate: gate,
                           swGateW: swGW, swGateS: swGS, swGateB: swGB, swUpW: swUW, swUpS: swUS, swUpB: swUB,
                           swDownW: swDW, swDownS: swDS, swDownB: swDB, shGate: shGate, shUp: shUp, shDown: shDown, sharedGate: sharedGate)
        let x = MLXRandom.normal([1, Hin]).asType(.float16)                 // [T=1, H]
        let ref = blk(x); ref.eval()
        // routing（MLX, MoEBlock と同一）
        let gates = MLX.softmax(gate.apply(x), axis: -1, precise: true)     // [1, E]
        let order = MLX.argPartition(gates, kth: E - topK, axis: -1)
        let inds = order[0..., (E - topK)...].asType(.int32)                 // [1,8]
        var scores = MLX.takeAlong(gates, inds, axis: -1)
        scores = scores / scores.sum(axis: -1, keepDims: true)               // normTopk
        let indsFlat = inds.reshaped([topK])
        let x2 = x.reshaped([1, Hin])
        // raw 4bit gather: g/u（共有 x）, d（per-expert h）
        guard let g = gatherQmm(x2, swGW, scales: swGS, biases: swGB, inds: indsFlat, Ktop: topK, K: Hin, N: I),
              let u = gatherQmm(x2, swUW, scales: swUS, biases: swUB, inds: indsFlat, Ktop: topK, K: Hin, N: I)
        else { return "[raw-moe] gather g/u 失敗" }
        guard let h = swigluRaw(g, u) else { return "[raw-moe] swiglu 失敗" }  // [8, I] raw swiglu
        guard let d = gatherQmm(h, swDW, scales: swDS, biases: swDB, inds: indsFlat, Ktop: topK, K: I, N: Hin, lhsPerExpert: true)
        else { return "[raw-moe] gather d 失敗" }                           // [8, H] per-expert
        let useRawCombine = ProcessInfo.processInfo.environment["QWISP_RAW_COMBINE"] == "1"
        let y: MLXArray
        if useRawCombine { guard let yc = combineRaw(d, scores.reshaped([topK]), K: topK, N: Hin) else { return "[raw-moe] combine 失敗" }; y = yc }
        else { y = (d * scores.reshaped([topK, 1])).sum(axis: 0) }            // MLX combine（切り分け用）
        // shared expert（shGate/shUp/shDown は raw 4bit qmm + raw swiglu, sharedGate 8bit は MLX）
        guard let sg = qmm(x, sgW.0, scales: sgW.1, biases: sgW.2, M: 1, K: Hin, N: I),
              let su = qmm(x, suW.0, scales: suW.1, biases: suW.2, M: 1, K: Hin, N: I),
              let shAct = swigluRaw(sg, su),
              let sharedY = qmm(shAct, sdW.0, scales: sdW.1, biases: sdW.2, M: 1, K: I, N: Hin)
        else { return "[raw-moe] shared expert 失敗" }
        let gateScale = MLX.sigmoid(sharedGate.apply(x))                     // 8bit gate は MLX
        let gotRaw = (y.reshaped([1, Hin]) + gateScale * sharedY)
        gotRaw.eval()
        let rel = relErr(gotRaw.reshaped([gotRaw.size]), ref.reshaped([ref.size]))
        // raw combine の単体 rel も併記（MLX reduce_col の sum 順序差で near, 重い計算は全 raw で bit-exact）
        var note = ""
        if !useRawCombine, let yr = combineRaw(d, scores.reshaped([topK]), K: topK, N: Hin) {
            let yMlx = (d * scores.reshaped([topK, 1])).sum(axis: 0)
            note = String(format: "\n  （raw combine 単体 rel=%.3e: MLX reduce_col の f16 sum 順序差。combine のみ MLX 推奨）",
                          relErr(yr, yMlx))
        }
        // ★ MoE expert single-encoder（task#7）: 常駐 buffer + 単一 encoder。bit-exact + 計測。
        if let mb = prepareMoEBuffers(swG: (swGW, swGS, swGB), swU: (swUW, swUS, swUB), swD: (swDW, swDS, swDB),
                                      shG: sgW, shU: suW, shD: sdW, Hin: Hin, I: I, topK: topK),
           let (dSE, sharedYSE) = moeExpertSingleEncoder(x, indsFlat, mb) {
            // combine + final add は MLX（routing/combine は外部 glue）
            let ySE = (dSE * scores.reshaped([topK, 1])).sum(axis: 0)
            let gotSE = (ySE.reshaped([1, Hin]) + gateScale * sharedYSE)
            gotSE.eval()
            let relSE = relErr(gotSE.reshaped([gotSE.size]), ref.reshaped([ref.size]))
            func cpuNs() -> UInt64 { var r = rusage(); getrusage(RUSAGE_SELF, &r)
                return UInt64(r.ru_utime.tv_sec+r.ru_stime.tv_sec)*1_000_000_000 + UInt64(r.ru_utime.tv_usec+r.ru_stime.tv_usec)*1000 }
            let reps = 200
            for _ in 0..<10 { _ = moeExpertSingleEncoder(x, indsFlat, mb) }
            var t0 = DispatchTime.now().uptimeNanoseconds; var c0 = cpuNs()
            for _ in 0..<reps { _ = moeExpertSingleEncoder(x, indsFlat, mb) }
            let seWall = Double(DispatchTime.now().uptimeNanoseconds-t0)/Double(reps)/1e6
            let seCpu = Double(cpuNs()-c0)/Double(reps)/1e6
            for _ in 0..<10 { let y = blk(x); y.eval() }
            t0 = DispatchTime.now().uptimeNanoseconds; c0 = cpuNs()
            for _ in 0..<reps { let y = blk(x); y.eval() }
            let mlxWall = Double(DispatchTime.now().uptimeNanoseconds-t0)/Double(reps)/1e6
            let mlxCpu = Double(cpuNs()-c0)/Double(reps)/1e6
            note += String(format: "\n  ── MoE expert single-encoder（task#7, routing/combine=MLX）──\n   SE 全体 rel=%.3e %@\n   時間: SE expert-encoder wall=%.3fms cpu=%.3fms | MLX MoE全体 wall=%.3fms cpu=%.3fms",
                           relSE, relSE == 0 ? "✅ bit-exact" : "△", seWall, seCpu, mlxWall, mlxCpu)
        }
        return String(format: "[raw-moe-test] MoE block raw（gather g/u/d・swiglu・shared expert=raw, routing/combine=MLX）vs MLX MoEBlock\n"
            + "  E=%d topK=%d Hin=%d I=%d。重い計算(gather 3種+swiglu+shared)は全 raw\n"
            + "  MoE block 全体 rel=%.3e %@%@",
            E, topK, Hin, I, rel, rel == 0 ? "✅ TRUE bit-exact" : (rel < 2e-3 ? "△ 近似(raw combine)" : "❌"), note)
    }

    /// MoE block を raw kernel で forward（routing/combine は MLX glue, gather/swiglu/shared は raw, decode T=1）。
    /// runMoeBlockTest で bit-exact 検証済の経路を関数化（decoder layer 結線用）。x[1,H] → [1,H]。
    static func moeRawForward(_ x: MLXArray, gate: Proj, sharedGate: Proj,
                              swG: (MLXArray, MLXArray, MLXArray), swU: (MLXArray, MLXArray, MLXArray), swD: (MLXArray, MLXArray, MLXArray),
                              shG: (MLXArray, MLXArray, MLXArray), shU: (MLXArray, MLXArray, MLXArray), shD: (MLXArray, MLXArray, MLXArray),
                              topK: Int = 8, E: Int = 256) -> MLXArray? {
        let Hin = x.dim(-1), I = swG.0.dim(-2)
        let gates = MLX.softmax(gate.apply(x), axis: -1, precise: true)
        let order = MLX.argPartition(gates, kth: E - topK, axis: -1)
        let inds = order[0..., (E - topK)...].asType(.int32)
        var scores = MLX.takeAlong(gates, inds, axis: -1)
        scores = scores / scores.sum(axis: -1, keepDims: true)
        let indsFlat = inds.reshaped([topK])
        guard let g = gatherQmm(x, swG.0, scales: swG.1, biases: swG.2, inds: indsFlat, Ktop: topK, K: Hin, N: I),
              let u = gatherQmm(x, swU.0, scales: swU.1, biases: swU.2, inds: indsFlat, Ktop: topK, K: Hin, N: I),
              let h = swigluRaw(g, u),
              let d = gatherQmm(h, swD.0, scales: swD.1, biases: swD.2, inds: indsFlat, Ktop: topK, K: I, N: Hin, lhsPerExpert: true),
              let sg = qmm(x, shG.0, scales: shG.1, biases: shG.2, M: 1, K: Hin, N: I),
              let su = qmm(x, shU.0, scales: shU.1, biases: shU.2, M: 1, K: Hin, N: I),
              let shAct = swigluRaw(sg, su),
              let sharedY = qmm(shAct, shD.0, scales: shD.1, biases: shD.2, M: 1, K: I, N: Hin)
        else { return nil }
        let y = (d * scores.reshaped([topK, 1])).sum(axis: 0)               // combine(MLX)
        let gateScale = MLX.sigmoid(sharedGate.apply(x))
        return y.reshaped([1, Hin]) + gateScale * sharedY
    }

    /// 検証: GDN decoder layer 1 層 end-to-end を raw 組成 vs MLX DecoderLayer（decode S=1）。
    /// input_norm(raw rmsNorm)→GDN raw→residual→post_norm→MoE raw→residual。40 層 tile の単位。
    /// - env: QWISP_RUN=raw-declayer-test / QWISP_DEC0_REF
    public static func runDecoderLayerTest() -> String {
        // QWISP_DEC_REF で ref 切替（既定 dec0=GDN, dec3=attn も可）。conv1d 有=GDN / 無=attn を自動判定。
        let refPath = ProcessInfo.processInfo.environment["QWISP_DEC_REF"]
            ?? ProcessInfo.processInfo.environment["QWISP_DEC0_REF"] ?? "/tmp/qwisp_dec0_ref.safetensors"
        guard let r = try? loadArrays(url: URL(fileURLWithPath: refPath)) else { return "[raw-declayer] ref 読込失敗 \(refPath)" }
        func q4(_ n: String) -> Proj { .quantized(r["\(n).weight"]!, r["\(n).scales"]!, r["\(n).biases"]!, 4) }
        func q8(_ n: String) -> Proj { .quantized(r["\(n).weight"]!, r["\(n).scales"]!, r["\(n).biases"]!, 8) }
        func tup(_ n: String) -> (MLXArray, MLXArray, MLXArray) { (r["\(n).weight"]!, r["\(n).scales"]!, r["\(n).biases"]!) }
        guard let iln = r["input_layernorm_weight"], let pln = r["post_attention_layernorm_weight"] else { return "[raw-declayer] layernorm 無" }
        let H = iln.dim(-1)
        let isLinear = r["conv1d"] != nil
        // MLX 参照 DecoderLayer
        var gdn: GatedDeltaNetLayer? = nil; var attn: AttentionLayer? = nil
        if isLinear {
            gdn = GatedDeltaNetLayer(numKHeads: 16, numVHeads: 32, headKDim: 128, headVDim: 128, convKernel: 4, eps: 1e-6,
                inProjQKV: q4("in_proj_qkv"), inProjZ: q4("in_proj_z"), inProjB: q4("in_proj_b"), inProjA: q4("in_proj_a"),
                outProj: q4("out_proj"), conv1dW: r["conv1d"]!, normWeight: r["la_norm_weight"]!, aLog: r["A_log"]!, dtBias: r["dt_bias"]!)
        } else {
            attn = AttentionLayer(numHeads: 16, numKVHeads: 2, headDim: 256, ropeDim: 64, ropeBase: 1e7, eps: 1e-6,
                qProj: q4("q_proj"), kProj: q4("k_proj"), vProj: q4("v_proj"), oProj: q4("o_proj"),
                qNorm: r["q_norm_weight"]!, kNorm: r["k_norm_weight"]!)
        }
        let layer = DecoderLayer(isLinear: isLinear, eps: 1e-6, inputLayernorm: iln, postAttentionLayernorm: pln,
                                 gdn: gdn, attn: attn, mlp: DecoderLayerValidation.mlpFrom(r))
        let prevF32 = GatedDeltaNetLayer.f32Conv; GatedDeltaNetLayer.f32Conv = true
        defer { GatedDeltaNetLayer.f32Conv = prevF32 }
        let x = MLXRandom.normal([1, 1, H]).asType(.float16)
        let ref = layer(x); ref.eval()
        // raw mixer
        guard let normed = rmsNorm(x.reshaped([1, H]), iln, eps: 1e-6, D: H) else { return "[raw-declayer] input_norm 失敗" }
        let rOut: MLXArray
        if isLinear {
            let rw = GDNRawWeights(
                qkvWq: r["in_proj_qkv.weight"]!, qkvSc: r["in_proj_qkv.scales"]!, qkvBi: r["in_proj_qkv.biases"]!,
                zWq: r["in_proj_z.weight"]!, zSc: r["in_proj_z.scales"]!, zBi: r["in_proj_z.biases"]!,
                bWq: r["in_proj_b.weight"]!, bSc: r["in_proj_b.scales"]!, bBi: r["in_proj_b.biases"]!,
                aWq: r["in_proj_a.weight"]!, aSc: r["in_proj_a.scales"]!, aBi: r["in_proj_a.biases"]!,
                outWq: r["out_proj.weight"]!, outSc: r["out_proj.scales"]!, outBi: r["out_proj.biases"]!,
                conv1dW: r["conv1d"]!.reshaped([8192, 4]).asType(.float32), normWeight: r["la_norm_weight"]!,
                aLog: r["A_log"]!, dtBias: r["dt_bias"]!)
            guard let ro = gdnLayerRaw(normed.reshaped([1, 1, H]), rw) else { return "[raw-declayer] GDN 失敗" }
            rOut = ro
        } else {
            let aw = AttnRawWeights(
                qWq: r["q_proj.weight"]!, qSc: r["q_proj.scales"]!, qBi: r["q_proj.biases"]!,
                kWq: r["k_proj.weight"]!, kSc: r["k_proj.scales"]!, kBi: r["k_proj.biases"]!,
                vWq: r["v_proj.weight"]!, vSc: r["v_proj.scales"]!, vBi: r["v_proj.biases"]!,
                oWq: r["o_proj.weight"]!, oSc: r["o_proj.scales"]!, oBi: r["o_proj.biases"]!,
                qNorm: r["q_norm_weight"]!, kNorm: r["k_norm_weight"]!)
            let pf = (r["q_norm_weight"]!.dtype == .float32)              // dec3 は F16→f16 経路
            guard let ro = attnLayerRaw(normed.reshaped([1, 1, H]), aw, promoteF32: pf) else { return "[raw-declayer] attn 失敗" }
            rOut = ro.asType(x.dtype)                                      // attn 出力を residual dtype(f16)へ
        }
        let h = x.reshaped([1, H]) + rOut                                   // residual
        guard let postNorm = rmsNorm(h, pln, eps: 1e-6, D: H) else { return "[raw-declayer] post_norm 失敗" }
        guard let mlpOut = moeRawForward(postNorm, gate: q8("gate"), sharedGate: q8("shared_expert_gate"),
            swG: tup("switch_mlp.gate_proj"), swU: tup("switch_mlp.up_proj"), swD: tup("switch_mlp.down_proj"),
            shG: tup("shared_expert.gate_proj"), shU: tup("shared_expert.up_proj"), shD: tup("shared_expert.down_proj"))
        else { return "[raw-declayer] MoE 失敗" }
        let got = h + mlpOut
        got.eval()
        let rel = relErr(got.reshaped([got.size]), ref.reshaped([ref.size]))
        return String(format: "[raw-declayer-test] %@ decoder layer 1層 raw(input_norm→mixer→res→post_norm→MoE→res) vs MLX\n"
            + "  H=%d  rel=%.3e %@", isLinear ? "GDN" : "attn", H, rel, rel == 0 ? "✅ TRUE bit-exact（40層 tile の単位）" : (rel < 2e-3 ? "△ 近似" : "❌"))
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
