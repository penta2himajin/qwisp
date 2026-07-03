import Foundation
import MLX
import MLXRandom
import Metal

/// D1 P2(path A): order-stable rows kernel を単一 command buffer + GPU 常駐中間 buffer で連結する
/// 融合経路。per-op CB commit/wait/readback(composed 経路の律速)を除去し、fused アーキテクチャへ
/// 収束させる。**演算(per-thread reduction)は rows kernel と同一のまま** — CB を束ねるだけなので
/// order-stable が構造的に保たれ、既存 test_raw.sh(15/15)がそのままゲートする。
public enum RawFusedVerify {

    /// qmm4(f16)を「既存 encoder に encode するだけ」の形で提供。cb/commit/readback 無し。
    /// out/x/w/s/b は全て MTLBuffer(常駐)。_qmmPipeline(qmm と共有)を使う。
    static func encodeQmmRows(_ enc: MTLComputeCommandEncoder,
                              w: MTLBuffer, scales: MTLBuffer, biases: MTLBuffer,
                              x: MTLBuffer, out: MTLBuffer, M: Int, K: Int, N: Int) {
        enc.setComputePipelineState(RawMetalForward._qmmPipeline!)
        enc.setBuffer(w, offset: 0, index: 0)
        enc.setBuffer(scales, offset: 0, index: 1)
        enc.setBuffer(biases, offset: 0, index: 2)
        enc.setBuffer(x, offset: 0, index: 3)
        enc.setBuffer(out, offset: 0, index: 4)
        var kk = Int32(K), nn = Int32(N)
        enc.setBytes(&kk, length: 4, index: 5)
        enc.setBytes(&nn, length: 4, index: 6)
        RawMetalForward.bindStop(enc, 16)
        enc.dispatchThreadgroups(MTLSize(width: M, height: N / 8, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 32, height: 2, depth: 1))
    }

    /// gqmm4_rows を encode-only で提供(cb/readback 無し)。inds/x/w/s/b/out 全て MTLBuffer 常駐。
    /// _gqmmRowsPipeline(gatherQmmRows と共有)。lhsPer: down_proj(x を mk 行で index)は true。
    static func encodeGatherQmmRows(_ enc: MTLComputeCommandEncoder,
                                    w: MTLBuffer, scales: MTLBuffer, biases: MTLBuffer,
                                    x: MTLBuffer, inds: MTLBuffer, out: MTLBuffer,
                                    M: Int, Ktop: Int, K: Int, N: Int, lhsPer: Bool) {
        enc.setComputePipelineState(RawMetalForward._gqmmRowsPipeline!)
        enc.setBuffer(w, offset: 0, index: 0); enc.setBuffer(scales, offset: 0, index: 1)
        enc.setBuffer(biases, offset: 0, index: 2); enc.setBuffer(x, offset: 0, index: 3)
        enc.setBuffer(inds, offset: 0, index: 4); enc.setBuffer(out, offset: 0, index: 5)
        var kk = Int32(K), nn = Int32(N), kt = Int32(Ktop)
        enc.setBytes(&kk, length: 4, index: 6); enc.setBytes(&nn, length: 4, index: 7); enc.setBytes(&kt, length: 4, index: 8)
        RawMetalForward.bindStop(enc, 9)
        var lp = UInt32(lhsPer ? 1 : 0); enc.setBytes(&lp, length: 4, index: 10)
        enc.dispatchThreadgroups(MTLSize(width: 1, height: N / 8, depth: M * Ktop),
                                 threadsPerThreadgroup: MTLSize(width: 32, height: 2, depth: 1))
    }

    /// P3 テスト支援: gate(lhsPer=false, x[M,K]共有 → g[M*Ktop,I])→ down(lhsPer=true, g → out[M*Ktop,K2])を
    /// 単一 CB + 常駐中間で実行。gatherQmmRows 2 回と bit 一致すれば gather の CB 融合が順序保存であることの証明。
    public static func fusedGatherChain(_ x: MLXArray, inds: MLXArray,
                                        w1: (MLXArray, MLXArray, MLXArray), I: Int,
                                        w2: (MLXArray, MLXArray, MLXArray), K2: Int,
                                        M: Int, Ktop: Int, K: Int) -> MLXArray? {
        guard let (device, queue) = RawMetalForward.ensure() else { return nil }
        _ = RawMetalForward.gatherQmmRows(x[0 ..< 1], w1.0, scales: w1.1, biases: w1.2,
                                          inds: inds[0 ..< Ktop], M: 1, Ktop: Ktop, K: K, N: I)   // warm compile
        guard let bx = RawMetalForward.mtlBuf(x.asType(.float16), device),
              let bin = RawMetalForward.mtlBuf(inds.asType(.int32), device),
              let bw1 = RawMetalForward.mtlBuf(w1.0, device),
              let bs1 = RawMetalForward.mtlBuf(w1.1.asType(.float16), device),
              let bb1 = RawMetalForward.mtlBuf(w1.2.asType(.float16), device),
              let bw2 = RawMetalForward.mtlBuf(w2.0, device),
              let bs2 = RawMetalForward.mtlBuf(w2.1.asType(.float16), device),
              let bb2 = RawMetalForward.mtlBuf(w2.2.asType(.float16), device) else { return nil }
        let midBuf = device.makeBuffer(length: M * Ktop * I * 2, options: .storageModeShared)!
        let outBuf = device.makeBuffer(length: M * Ktop * K2 * 2, options: .storageModeShared)!
        let cb = queue.makeCommandBuffer()!; let enc = cb.makeComputeCommandEncoder()!
        encodeGatherQmmRows(enc, w: bw1, scales: bs1, biases: bb1, x: bx, inds: bin, out: midBuf,
                            M: M, Ktop: Ktop, K: K, N: I, lhsPer: false)
        encodeGatherQmmRows(enc, w: bw2, scales: bs2, biases: bb2, x: midBuf, inds: bin, out: outBuf,
                            M: M, Ktop: Ktop, K: I, N: K2, lhsPer: true)
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        let ptr = outBuf.contents().bindMemory(to: Float16.self, capacity: M * Ktop * K2)
        return MLXArray(Array(UnsafeBufferPointer(start: ptr, count: M * Ktop * K2)), [M * Ktop, K2])
    }

    nonisolated(unsafe) static var _routeRowsPipeline: MTLComputePipelineState?
    /// M-row route_top8: 各 threadgroup が 1 token の top-8 を独立に選ぶ(route_top8 と同一の
    /// per-token reduction — precise::exp softmax + 決定的 K 回 argmax)。grid.x=M でトークン offset
    /// するだけ → M 不変。MLX argPartition の sync 島を Metal に置換(routing の中間は argPartition と
    /// 非一致だが、engine 自己整合(batched≡sequential)は保たれ、出力トークンが最重要=owner 方針)。
    /// logits[M,N] f16 → inds[M,K] int32, scores[M,K] f16(row 毎 renorm 済)。
    public static func routeTop8Rows(_ logits: MLXArray, M: Int, N: Int = 256, K: Int = 8) -> (MLXArray, MLXArray)? {
        guard let (device, queue) = RawMetalForward.ensure() else { return nil }
        if _routeRowsPipeline == nil {
            let src = """
            #include <metal_stdlib>
            using namespace metal;
            kernel void route_top8_rows(device const half* logits [[buffer(0)]],
                                        device int* inds [[buffer(1)]], device half* scores [[buffer(2)]],
                                        constant uint& N [[buffer(3)]], constant uint& K [[buffer(4)]],
                                        uint tgid [[threadgroup_position_in_grid]],
                                        uint tid [[thread_position_in_threadgroup]], uint tgs [[threads_per_threadgroup]]) {
                const device half* lgrow = logits + tgid * N;   // 行 tgid.x の logits
                device int* indrow = inds + tgid * K;
                device half* scrow = scores + tgid * K;
                threadgroup float red[256]; threadgroup int redi[256];
                threadgroup float gates[256]; threadgroup float work[256];
                threadgroup float bcast[1];
                float lg = (tid < N) ? (float)lgrow[tid] : -INFINITY;
                red[tid] = lg; threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint s = tgs/2; s > 0; s >>= 1) { if (tid < s) red[tid] = max(red[tid], red[tid+s]); threadgroup_barrier(mem_flags::mem_threadgroup); }
                if (tid == 0) bcast[0] = red[0]; threadgroup_barrier(mem_flags::mem_threadgroup);
                float m = bcast[0];
                float e = (tid < N) ? precise::exp(lg - m) : 0.0f;
                red[tid] = e; threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint s = tgs/2; s > 0; s >>= 1) { if (tid < s) red[tid] += red[tid+s]; threadgroup_barrier(mem_flags::mem_threadgroup); }
                if (tid == 0) bcast[0] = red[0]; threadgroup_barrier(mem_flags::mem_threadgroup);
                float Z = bcast[0];
                if (tid < N) { gates[tid] = (float)(half)(e / Z); work[tid] = lg; }
                else { work[tid] = -INFINITY; }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint k = 0; k < K; k++) {
                    red[tid] = work[tid]; redi[tid] = (int)tid; threadgroup_barrier(mem_flags::mem_threadgroup);
                    for (uint s = tgs/2; s > 0; s >>= 1) {
                        if (tid < s) { if (red[tid+s] > red[tid]) { red[tid] = red[tid+s]; redi[tid] = redi[tid+s]; } }
                        threadgroup_barrier(mem_flags::mem_threadgroup);
                    }
                    if (tid == 0) { int bi = redi[0]; indrow[k] = bi; scrow[k] = (half)gates[bi]; work[bi] = -INFINITY; }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
                if (tid == 0) { half ss = (half)0; for (uint k = 0; k < K; k++) ss += scrow[k]; for (uint k = 0; k < K; k++) scrow[k] = scrow[k] / ss; }
            }
            """
            do { let lib = try device.makeLibrary(source: src, options: RawMetalForward.mlxMatchCompileOpts())
                 _routeRowsPipeline = try device.makeComputePipelineState(function: lib.makeFunction(name: "route_top8_rows")!)
            } catch { print("[raw-route-rows] compile: \(error)"); return nil }
        }
        guard let bl = RawMetalForward.mtlBuf(logits.asType(.float16), device) else { return nil }
        let bInds = device.makeBuffer(length: M * K * 4, options: .storageModeShared)!
        let bScores = device.makeBuffer(length: M * K * 2, options: .storageModeShared)!
        let cb = queue.makeCommandBuffer()!; let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(_routeRowsPipeline!)
        enc.setBuffer(bl, offset: 0, index: 0); enc.setBuffer(bInds, offset: 0, index: 1); enc.setBuffer(bScores, offset: 0, index: 2)
        var nn = UInt32(N), kk = UInt32(K); enc.setBytes(&nn, length: 4, index: 3); enc.setBytes(&kk, length: 4, index: 4)
        enc.dispatchThreadgroups(MTLSize(width: M, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        let ip = bInds.contents().bindMemory(to: Int32.self, capacity: M * K)
        let sp = bScores.contents().bindMemory(to: Float16.self, capacity: M * K)
        return (MLXArray(Array(UnsafeBufferPointer(start: ip, count: M * K)), [M, K]),
                MLXArray(Array(UnsafeBufferPointer(start: sp, count: M * K)), [M, K]))
    }

    /// _qmmPipeline が未コンパイルなら小さな qmm 呼びで確実にコンパイルさせる(big qmm 関数を触らない)。
    static func ensureQmmPipeline() {
        if RawMetalForward._qmmPipeline != nil { return }
        let x = MLXRandom.normal([1, 512]).asType(.float16)
        let wf = MLXRandom.normal([8, 512]).asType(.float16)
        let (wq, s, b) = MLX.quantized(wf, groupSize: 64, bits: 4, mode: .affine)
        MLX.eval([x, wq, s, b!])
        _ = RawMetalForward.qmm(x, wq, scales: s, biases: b!, M: 1, K: 512, N: 8)
    }

    /// P2a テスト支援: x → (w1) → mid → (w2) → out を **単一 CB + 常駐中間 midBuf** で実行し out を返す。
    /// per-op 版(qmmRows 2 回)と bit 一致すれば「CB 融合 + 中間常駐」が順序保存であることの証明。
    public static func fusedTwoQmm(_ x: MLXArray, w1: (MLXArray, MLXArray, MLXArray), N1: Int,
                                   w2: (MLXArray, MLXArray, MLXArray), N2: Int, M: Int, K: Int) -> MLXArray? {
        guard let (device, queue) = RawMetalForward.ensure() else { return nil }
        ensureQmmPipeline()
        guard let bx = RawMetalForward.mtlBuf(x.asType(.float16), device),
              let bw1 = RawMetalForward.mtlBuf(w1.0, device),
              let bs1 = RawMetalForward.mtlBuf(w1.1.asType(.float16), device),
              let bb1 = RawMetalForward.mtlBuf(w1.2.asType(.float16), device),
              let bw2 = RawMetalForward.mtlBuf(w2.0, device),
              let bs2 = RawMetalForward.mtlBuf(w2.1.asType(.float16), device),
              let bb2 = RawMetalForward.mtlBuf(w2.2.asType(.float16), device) else { return nil }
        let midBuf = device.makeBuffer(length: M * N1 * 2, options: .storageModeShared)!
        let outBuf = device.makeBuffer(length: M * N2 * 2, options: .storageModeShared)!
        let cb = queue.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        encodeQmmRows(enc, w: bw1, scales: bs1, biases: bb1, x: bx, out: midBuf, M: M, K: K, N: N1)
        // 中間は同一 encoder 内で連続 dispatch。同一 encoder の逐次 dispatch はプログラム順に実行され
        // buffer 依存(midBuf 書込→読込)は自動でメモリ整合される(同一 queue/encoder のシリアル実行)。
        encodeQmmRows(enc, w: bw2, scales: bs2, biases: bb2, x: midBuf, out: outBuf, M: M, K: N1, N: N2)
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        let ptr = outBuf.contents().bindMemory(to: Float16.self, capacity: M * N2)
        return MLXArray(Array(UnsafeBufferPointer(start: ptr, count: M * N2)), [M, N2])
    }

    // ── Stage A(P3 続き): fused MoE block — moeBlockRows(metalRoute) 全段を単一 encoder 化 ──

    nonisolated(unsafe) static var _combineRowsPipeline: MTLComputePipelineState?
    nonisolated(unsafe) static var _finalCombineRowsPipeline: MTLComputePipelineState?
    nonisolated(unsafe) static var _writeKVRowsPipeline: MTLComputePipelineState?
    nonisolated(unsafe) static var _convHistRowsPipeline: MTLComputePipelineState?
    nonisolated(unsafe) static var _shiftConvRowsPipeline: MTLComputePipelineState?
    nonisolated(unsafe) static var _sliceRowsPipeline: MTLComputePipelineState?
    nonisolated(unsafe) static var _computeGBetaRowsPipeline: MTLComputePipelineState?

    /// M-row elementwise 補助 kernel(combine/final)。composed の MLX glue と同一の演算列を
    /// per-element/per-token 独立で再現(f16 逐次和・stable sigmoid)→ M 非依存。
    static func ensureRowsAuxPipelines() -> Bool {
        guard let (device, _) = RawMetalForward.ensure() else { return false }
        if _combineRowsPipeline != nil && _finalCombineRowsPipeline != nil && _writeKVRowsPipeline != nil
            && _convHistRowsPipeline != nil && _shiftConvRowsPipeline != nil
            && _sliceRowsPipeline != nil && _computeGBetaRowsPipeline != nil { return true }
        let src = """
        #include <metal_stdlib>
        using namespace metal;
        // combine_rows: y[m,n] = Σ_k d[(m*K+k)*N+n]·scores[m*K+k]（f16 の k 昇順逐次和 =
        // composed moeBlockRows の明示順序 ki ループと同一の演算列。f16 で 0+x==x ゆえ acc=0 開始も同値）。
        kernel void combine_rows(device const half* d [[buffer(0)]], device const half* scores [[buffer(1)]],
                                 device half* y [[buffer(2)]], constant uint& K [[buffer(3)]],
                                 constant uint& N [[buffer(4)]], constant uint& total [[buffer(5)]],
                                 uint i [[thread_position_in_grid]]) {
            if (i >= total) return;
            uint m = i / N, n = i % N;
            half acc = (half)0;
            for (uint k = 0; k < K; ++k) acc += d[(m*K + k)*N + n] * scores[m*K + k];
            y[i] = acc;
        }
        // final_combine_rows: out[m,n] = y[m,n] + sigmoid(sgl[m*8])·sharedY[m,n]。
        // sigmoid は MLX stable(half, metal::exp)= composed の MLX.sigmoid(sgl[:,0:1]) と同一。
        kernel void final_combine_rows(device const half* y [[buffer(0)]], device const half* sharedY [[buffer(1)]],
                                       device const half* sgl [[buffer(2)]], device half* outp [[buffer(3)]],
                                       constant uint& N [[buffer(4)]], constant uint& total [[buffer(5)]],
                                       uint i [[thread_position_in_grid]]) {
            if (i >= total) return;
            uint m = i / N;
            half gv = sgl[m * 8];
            half yv = (half)1 / ((half)1 + exp(metal::abs(gv)));
            half s = (gv < (half)0) ? yv : ((half)1 - yv);
            outp[i] = y[i] + s * sharedY[i];
        }
        // write_kv_rows: src[M*KV, D](行 m の kv head h)を cache[KV, maxLen, D] の seq 位置 pos+m に散布。
        // 純コピー(演算無し)= composed の transpose+concat と bit 同値。
        kernel void write_kv_rows(device const half* src [[buffer(0)]], device half* cache [[buffer(1)]],
                                  constant uint& KV [[buffer(2)]], constant uint& D [[buffer(3)]],
                                  constant uint& maxLen [[buffer(4)]], constant uint& pos [[buffer(5)]],
                                  constant uint& total [[buffer(6)]],
                                  uint i [[thread_position_in_grid]]) {
            if (i >= total) return;
            uint m = i / (KV*D), rem = i % (KV*D), h = rem / D, dd = rem % D;
            cache[h*maxLen*D + (pos+m)*D + dd] = src[(m*KV + h)*D + dd];
        }
        // conv1d_silu_hist_rows: conv 窓を hist[K-1,C]+qkv[M,C] の直読みで構成(composed の
        // concat+stack と同値の値列)。演算は conv1d_silu_rows と完全同一(f32 acc, precise silu)。
        kernel void conv1d_silu_hist_rows(device const half* hist [[buffer(0)]],
                                          device const half* qkv  [[buffer(1)]],
                                          device const float* w   [[buffer(2)]],
                                          device half* outp       [[buffer(3)]],
                                          constant uint& K [[buffer(4)]], constant uint& C [[buffer(5)]],
                                          uint2 pos [[thread_position_in_grid]]) {
            uint c = pos.x, m = pos.y;
            if (c >= C) return;
            float acc = 0.0f;
            for (uint k = 0; k < K; ++k) {
                uint idx = m + k;
                float xv = (idx < K - 1) ? (float)hist[idx*C + c] : (float)qkv[(idx - (K-1))*C + c];
                acc += xv * w[c*K + k];
            }
            float ax = metal::abs(acc);
            float y = 1.0f / (1.0f + precise::exp(ax));
            float s = (acc < 0.0f) ? y : (1.0f - y);
            outp[m*C + c] = (half)(acc * s);
        }
        // shift_conv_rows: hist ← concat(hist,qkv)[M .. M+K-2](composed の convState 更新と同値)。
        // thread=列 c、j 昇順で src=M+j>j ゆえ in-place race-free。
        kernel void shift_conv_rows(device half* hist [[buffer(0)]], device const half* qkv [[buffer(1)]],
                                    constant uint& K [[buffer(2)]], constant uint& C [[buffer(3)]],
                                    constant uint& M [[buffer(4)]],
                                    uint c [[thread_position_in_grid]]) {
            if (c >= C) return;
            for (uint j = 0; j + 1 < K; ++j) {
                uint src = M + j;
                hist[j*C + c] = (src < K - 1) ? hist[src*C + c] : qkv[(src - (K-1))*C + c];
            }
        }
        // slice_rows: 行毎 strided 抽出(純コピー)。out[m*W+j] = in[m*stride + off + j]。
        kernel void slice_rows(device const half* inp [[buffer(0)]], device half* outp [[buffer(1)]],
                               constant uint& off [[buffer(2)]], constant uint& W [[buffer(3)]],
                               constant uint& strideLen [[buffer(4)]], constant uint& total [[buffer(5)]],
                               uint i [[thread_position_in_grid]]) {
            if (i >= total) return;
            uint m = i / W, j = i % W;
            outp[i] = inp[m*strideLen + off + j];
        }
        // compute_g_beta_rows: 既存 compute_g_beta の M-row 拡張(i を M*Hv に、aLog/dtBias は i%Hv)。
        kernel void compute_g_beta_rows(device const half* a [[buffer(0)]], device const half* b [[buffer(1)]],
                                        device const float* aLog [[buffer(2)]], device const float* dtBias [[buffer(3)]],
                                        device float* g [[buffer(4)]], device float* beta [[buffer(5)]],
                                        constant uint& Hv [[buffer(6)]], constant uint& total [[buffer(7)]],
                                        uint i [[thread_position_in_grid]]) {
            if (i >= total) return;
            uint hv = i % Hv;
            half bh = b[i];
            half y = (half)1 / ((half)1 + exp(metal::abs(bh)));
            half sb = (bh < (half)0) ? y : ((half)1 - y);
            beta[i] = (float)sb;
            float x = (float)a[i] + dtBias[hv];
            float sp = max(x, 0.0f) + precise::log(1.0f + precise::exp(-metal::abs(x)));
            g[i] = precise::exp(-precise::exp(aLog[hv]) * sp);
        }
        """
        do {
            let lib = try device.makeLibrary(source: src, options: RawMetalForward.mlxMatchCompileOpts())
            _combineRowsPipeline = try device.makeComputePipelineState(function: lib.makeFunction(name: "combine_rows")!)
            _finalCombineRowsPipeline = try device.makeComputePipelineState(function: lib.makeFunction(name: "final_combine_rows")!)
            _writeKVRowsPipeline = try device.makeComputePipelineState(function: lib.makeFunction(name: "write_kv_rows")!)
            _convHistRowsPipeline = try device.makeComputePipelineState(function: lib.makeFunction(name: "conv1d_silu_hist_rows")!)
            _shiftConvRowsPipeline = try device.makeComputePipelineState(function: lib.makeFunction(name: "shift_conv_rows")!)
            _sliceRowsPipeline = try device.makeComputePipelineState(function: lib.makeFunction(name: "slice_rows")!)
            _computeGBetaRowsPipeline = try device.makeComputePipelineState(function: lib.makeFunction(name: "compute_g_beta_rows")!)
            return true
        } catch { print("[raw-fused-aux] compile: \(error)"); return false }
    }

    /// qmm8(router gate / shared gate logits)を encode-only で提供。qmm8 と同一 pipeline/dispatch。
    static func encodeQmm8Rows(_ enc: MTLComputeCommandEncoder,
                               w: MTLBuffer, scales: MTLBuffer, biases: MTLBuffer,
                               x: MTLBuffer, out: MTLBuffer, M: Int, K: Int, N: Int) {
        enc.setComputePipelineState(RawMetalForward._qmm8Pipeline!)
        enc.setBuffer(w, offset: 0, index: 0); enc.setBuffer(scales, offset: 0, index: 1)
        enc.setBuffer(biases, offset: 0, index: 2); enc.setBuffer(x, offset: 0, index: 3)
        enc.setBuffer(out, offset: 0, index: 4)
        var kk = Int32(K), nn = Int32(N)
        enc.setBytes(&kk, length: 4, index: 5); enc.setBytes(&nn, length: 4, index: 6)
        RawMetalForward.bindStop(enc, 16)
        enc.dispatchThreadgroups(MTLSize(width: M, height: N / 8, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 32, height: 2, depth: 1))
    }

    /// route_top8_rows を encode-only で提供。routeTop8Rows と同一 pipeline/dispatch。
    static func encodeRouteTop8Rows(_ enc: MTLComputeCommandEncoder,
                                    logits: MTLBuffer, inds: MTLBuffer, scores: MTLBuffer,
                                    M: Int, N: Int, K: Int) {
        enc.setComputePipelineState(_routeRowsPipeline!)
        enc.setBuffer(logits, offset: 0, index: 0); enc.setBuffer(inds, offset: 0, index: 1)
        enc.setBuffer(scores, offset: 0, index: 2)
        var nn = UInt32(N), kk = UInt32(K)
        enc.setBytes(&nn, length: 4, index: 3); enc.setBytes(&kk, length: 4, index: 4)
        enc.dispatchThreadgroups(MTLSize(width: M, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    /// swiglu(既存 aux kernel, per-element 独立=M 不変)を encode-only で提供。
    static func encodeSwiglu(_ enc: MTLComputeCommandEncoder, g: MTLBuffer, u: MTLBuffer, h: MTLBuffer, total: Int) {
        let p = RawMetalForward._swigluPipeline!
        enc.setComputePipelineState(p)
        enc.setBuffer(g, offset: 0, index: 0); enc.setBuffer(u, offset: 0, index: 1); enc.setBuffer(h, offset: 0, index: 2)
        var t = UInt32(total); enc.setBytes(&t, length: 4, index: 3)
        enc.dispatchThreads(MTLSize(width: total, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: min(p.maxTotalThreadsPerThreadgroup, 256), height: 1, depth: 1))
    }

    static func encodeCombineRows(_ enc: MTLComputeCommandEncoder, d: MTLBuffer, scores: MTLBuffer, y: MTLBuffer,
                                  Ktop: Int, N: Int, M: Int) {
        let p = _combineRowsPipeline!
        enc.setComputePipelineState(p)
        enc.setBuffer(d, offset: 0, index: 0); enc.setBuffer(scores, offset: 0, index: 1); enc.setBuffer(y, offset: 0, index: 2)
        var kk = UInt32(Ktop), nn = UInt32(N), t = UInt32(M * N)
        enc.setBytes(&kk, length: 4, index: 3); enc.setBytes(&nn, length: 4, index: 4); enc.setBytes(&t, length: 4, index: 5)
        enc.dispatchThreads(MTLSize(width: M * N, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: min(p.maxTotalThreadsPerThreadgroup, 256), height: 1, depth: 1))
    }

    static func encodeFinalCombineRows(_ enc: MTLComputeCommandEncoder, y: MTLBuffer, sharedY: MTLBuffer,
                                       sgl: MTLBuffer, out: MTLBuffer, N: Int, M: Int) {
        let p = _finalCombineRowsPipeline!
        enc.setComputePipelineState(p)
        enc.setBuffer(y, offset: 0, index: 0); enc.setBuffer(sharedY, offset: 0, index: 1)
        enc.setBuffer(sgl, offset: 0, index: 2); enc.setBuffer(out, offset: 0, index: 3)
        var nn = UInt32(N), t = UInt32(M * N)
        enc.setBytes(&nn, length: 4, index: 4); enc.setBytes(&t, length: 4, index: 5)
        enc.dispatchThreads(MTLSize(width: M * N, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: min(p.maxTotalThreadsPerThreadgroup, 256), height: 1, depth: 1))
    }

    /// per-op wrapper: combine_rows(y[m,n]=Σ_k d·scores)を単発 CB で実行。composed が使うことで
    /// fused と combine の丸め列(FMA 有無含む)を共有する。
    public static func combineRowsRaw(_ d: MLXArray, _ scores: MLXArray, M: Int, Ktop: Int, N: Int) -> MLXArray? {
        guard let (device, queue) = RawMetalForward.ensure(), ensureRowsAuxPipelines() else { return nil }
        guard let bd = RawMetalForward.mtlBuf(d.asType(.float16), device),
              let bs = RawMetalForward.mtlBuf(scores.asType(.float16), device),
              let outBuf = device.makeBuffer(length: M * N * 2, options: .storageModeShared) else { return nil }
        let cb = queue.makeCommandBuffer()!; let enc = cb.makeComputeCommandEncoder()!
        encodeCombineRows(enc, d: bd, scores: bs, y: outBuf, Ktop: Ktop, N: N, M: M)
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        let ptr = outBuf.contents().bindMemory(to: Float16.self, capacity: M * N)
        return MLXArray(Array(UnsafeBufferPointer(start: ptr, count: M * N)), [M, N])
    }

    /// per-op wrapper: final_combine_rows(out = y + sigmoid(sgl[:,0])·sharedY)を単発 CB で実行。
    /// composed moeBlockRows がこれを使うことで fused と elementwise 数値系を共有する。
    public static func finalCombineRowsRaw(_ y: MLXArray, _ sharedY: MLXArray, _ sgl: MLXArray,
                                           M: Int, N: Int) -> MLXArray? {
        guard let (device, queue) = RawMetalForward.ensure(), ensureRowsAuxPipelines() else { return nil }
        guard let by = RawMetalForward.mtlBuf(y.asType(.float16), device),
              let bs = RawMetalForward.mtlBuf(sharedY.asType(.float16), device),
              let bg = RawMetalForward.mtlBuf(sgl.asType(.float16), device),
              let outBuf = device.makeBuffer(length: M * N * 2, options: .storageModeShared) else { return nil }
        let cb = queue.makeCommandBuffer()!; let enc = cb.makeComputeCommandEncoder()!
        encodeFinalCombineRows(enc, y: by, sharedY: bs, sgl: bg, out: outBuf, N: N, M: M)
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        let ptr = outBuf.contents().bindMemory(to: Float16.self, capacity: M * N)
        return MLXArray(Array(UnsafeBufferPointer(start: ptr, count: M * N)), [M, N])
    }

    /// MoE block 用の重み MTLBuffer 束(composed wrapper と同一の型変換で常駐化)。
    public struct MoEBlockBufs {
        let gW: MTLBuffer, gS: MTLBuffer, gB: MTLBuffer
        let swGW: MTLBuffer, swGS: MTLBuffer, swGB: MTLBuffer
        let swUW: MTLBuffer, swUS: MTLBuffer, swUB: MTLBuffer
        let swDW: MTLBuffer, swDS: MTLBuffer, swDB: MTLBuffer
        let shGW: MTLBuffer, shGS: MTLBuffer, shGB: MTLBuffer
        let shUW: MTLBuffer, shUS: MTLBuffer, shUB: MTLBuffer
        let shDW: MTLBuffer, shDS: MTLBuffer, shDB: MTLBuffer
        let sgW: MTLBuffer, sgS: MTLBuffer, sgB: MTLBuffer
    }

    static func prepareMoEBlockBufs(_ w: RawVerifyForward.MoEBlockW, _ device: MTLDevice) -> MoEBlockBufs? {
        func trio(_ q: MLXArray, _ s: MLXArray, _ b: MLXArray) -> (MTLBuffer, MTLBuffer, MTLBuffer)? {
            guard let bq = RawMetalForward.mtlBuf(q, device),
                  let bs = RawMetalForward.mtlBuf(s.asType(.float16), device),
                  let bb = RawMetalForward.mtlBuf(b.asType(.float16), device) else { return nil }
            return (bq, bs, bb)
        }
        guard let g = trio(w.gateWq, w.gateSc, w.gateBi),
              let swG = trio(w.swGWq, w.swGSc, w.swGBi), let swU = trio(w.swUWq, w.swUSc, w.swUBi),
              let swD = trio(w.swDWq, w.swDSc, w.swDBi),
              let shG = trio(w.shGWq, w.shGSc, w.shGBi), let shU = trio(w.shUWq, w.shUSc, w.shUBi),
              let shD = trio(w.shDWq, w.shDSc, w.shDBi),
              let sg = trio(w.sharedGateWq, w.sharedGateSc, w.sharedGateBi) else { return nil }
        return MoEBlockBufs(gW: g.0, gS: g.1, gB: g.2,
                            swGW: swG.0, swGS: swG.1, swGB: swG.2,
                            swUW: swU.0, swUS: swU.1, swUB: swU.2,
                            swDW: swD.0, swDS: swD.1, swDB: swD.2,
                            shGW: shG.0, shGS: shG.1, shGB: shG.2,
                            shUW: shU.0, shUS: shU.1, shUB: shU.2,
                            shDW: shD.0, shDS: shD.1, shDB: shD.2,
                            sgW: sg.0, sgS: sg.1, sgB: sg.2)
    }

    /// MoE block × M 行の全段(routing→routed experts→combine→shared→final)を **既存 encoder に
    /// encode するだけ**の形で提供。入力 x[M,H] と出力 out[M,H] は常駐 MTLBuffer。
    /// 演算列は moeBlockRows(metalRoute: true) と 1:1(同一 pipeline・同一 dispatch 形状)。
    /// scratch は呼び出し側が確保(encodeMoEBlockScratch)。
    public struct MoEScratch {
        let gl: MTLBuffer, inds: MTLBuffer, scores: MTLBuffer
        let g: MTLBuffer, u: MTLBuffer, h: MTLBuffer, d: MTLBuffer, y: MTLBuffer
        let sg: MTLBuffer, su: MTLBuffer, shAct: MTLBuffer, sharedY: MTLBuffer, sgl: MTLBuffer
    }

    static func makeMoEScratch(_ device: MTLDevice, M: Int, E: Int, I: Int, Ktop: Int, H: Int) -> MoEScratch? {
        func buf(_ n: Int) -> MTLBuffer? { device.makeBuffer(length: n, options: .storageModeShared) }
        guard let gl = buf(M * E * 2), let inds = buf(M * Ktop * 4), let scores = buf(M * Ktop * 2),
              let g = buf(M * Ktop * I * 2), let u = buf(M * Ktop * I * 2), let h = buf(M * Ktop * I * 2),
              let d = buf(M * Ktop * H * 2), let y = buf(M * H * 2),
              let sg = buf(M * I * 2), let su = buf(M * I * 2), let shAct = buf(M * I * 2),
              let sharedY = buf(M * H * 2), let sgl = buf(M * 8 * 2) else { return nil }
        return MoEScratch(gl: gl, inds: inds, scores: scores, g: g, u: u, h: h, d: d, y: y,
                          sg: sg, su: su, shAct: shAct, sharedY: sharedY, sgl: sgl)
    }

    static func encodeMoEBlockRows(_ enc: MTLComputeCommandEncoder, x: MTLBuffer, out: MTLBuffer,
                                   w: MoEBlockBufs, sc: MoEScratch,
                                   M: Int, E: Int, I: Int, Ktop: Int, H: Int) {
        // ① routing: gate qmm8 → route_top8_rows(inds+renorm scores)
        encodeQmm8Rows(enc, w: w.gW, scales: w.gS, biases: w.gB, x: x, out: sc.gl, M: M, K: H, N: E)
        encodeRouteTop8Rows(enc, logits: sc.gl, inds: sc.inds, scores: sc.scores, M: M, N: E, K: Ktop)
        // ② routed experts: gather g/u(行共有 lhs)→ swiglu → gather d(per-mk lhs)→ combine
        encodeGatherQmmRows(enc, w: w.swGW, scales: w.swGS, biases: w.swGB, x: x, inds: sc.inds, out: sc.g,
                            M: M, Ktop: Ktop, K: H, N: I, lhsPer: false)
        encodeGatherQmmRows(enc, w: w.swUW, scales: w.swUS, biases: w.swUB, x: x, inds: sc.inds, out: sc.u,
                            M: M, Ktop: Ktop, K: H, N: I, lhsPer: false)
        encodeSwiglu(enc, g: sc.g, u: sc.u, h: sc.h, total: M * Ktop * I)
        encodeGatherQmmRows(enc, w: w.swDW, scales: w.swDS, biases: w.swDB, x: sc.h, inds: sc.inds, out: sc.d,
                            M: M, Ktop: Ktop, K: I, N: H, lhsPer: true)
        encodeCombineRows(enc, d: sc.d, scores: sc.scores, y: sc.y, Ktop: Ktop, N: H, M: M)
        // ③ shared expert: sg/su qmm → swiglu → down qmm
        encodeQmmRows(enc, w: w.shGW, scales: w.shGS, biases: w.shGB, x: x, out: sc.sg, M: M, K: H, N: I)
        encodeQmmRows(enc, w: w.shUW, scales: w.shUS, biases: w.shUB, x: x, out: sc.su, M: M, K: H, N: I)
        encodeSwiglu(enc, g: sc.sg, u: sc.su, h: sc.shAct, total: M * I)
        encodeQmmRows(enc, w: w.shDW, scales: w.shDS, biases: w.shDB, x: sc.shAct, out: sc.sharedY, M: M, K: I, N: H)
        // ④ shared gate logits(qmm8 N=8, 列0のみ使用)→ final combine
        encodeQmm8Rows(enc, w: w.sgW, scales: w.sgS, biases: w.sgB, x: x, out: sc.sgl, M: M, K: H, N: 8)
        encodeFinalCombineRows(enc, y: sc.y, sharedY: sc.sharedY, sgl: sc.sgl, out: out, N: H, M: M)
    }

    /// 全 pipeline を warm(compile)。fused 経路の前提(encode 時に force-unwrap するため)。
    static func ensureMoEPipelines(E: Int = 256, Ktop: Int = 8) -> Bool {
        ensureQmmPipeline()
        guard RawMetalForward.compileQmm8(), RawMetalForward.ensureAuxPipelines(), ensureRowsAuxPipelines()
        else { return false }
        if _routeRowsPipeline == nil {
            let dummy = MLXArray.zeros([1, E]).asType(.float16); dummy.eval()
            _ = routeTop8Rows(dummy, M: 1, N: E, K: Ktop)
        }
        if RawMetalForward._gqmmRowsPipeline == nil {
            let x = MLXRandom.normal([1, 512]).asType(.float16)
            let wf = MLXRandom.normal([2, 8, 512]).asType(.float16)
            let (wq, s, b) = MLX.quantized(wf, groupSize: 64, bits: 4, mode: .affine)
            let inds = MLXArray([Int32(0)], [1])
            MLX.eval([x, wq, s, b!, inds])
            _ = RawMetalForward.gatherQmmRows(x, wq, scales: s, biases: b!, inds: inds, M: 1, Ktop: 1, K: 512, N: 8)
        }
        return _routeRowsPipeline != nil && RawMetalForward._gqmmRowsPipeline != nil
    }

    /// debug: fused MoE block の全中間 buffer を読み出して返す(段階別バイセクト用)。
    public static func fusedMoEBlockRowsDump(_ x: MLXArray, _ w: RawVerifyForward.MoEBlockW,
                                             M: Int, E: Int, I: Int, Ktop: Int = 8) -> [String: MLXArray]? {
        guard let (device, queue) = RawMetalForward.ensure() else { return nil }
        let H = x.dim(-1)
        guard ensureMoEPipelines(E: E, Ktop: Ktop) else { return nil }
        guard let bufs = prepareMoEBlockBufs(w, device),
              let sc = makeMoEScratch(device, M: M, E: E, I: I, Ktop: Ktop, H: H),
              let bx = RawMetalForward.mtlBuf(x.asType(.float16), device),
              let outBuf = device.makeBuffer(length: M * H * 2, options: .storageModeShared) else { return nil }
        let cb = queue.makeCommandBuffer()!; let enc = cb.makeComputeCommandEncoder()!
        encodeMoEBlockRows(enc, x: bx, out: outBuf, w: bufs, sc: sc, M: M, E: E, I: I, Ktop: Ktop, H: H)
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        func rd(_ b: MTLBuffer, _ n: Int, _ shape: [Int]) -> MLXArray {
            let p = b.contents().bindMemory(to: Float16.self, capacity: n)
            return MLXArray(Array(UnsafeBufferPointer(start: p, count: n)), shape)
        }
        func rdI(_ b: MTLBuffer, _ n: Int, _ shape: [Int]) -> MLXArray {
            let p = b.contents().bindMemory(to: Int32.self, capacity: n)
            return MLXArray(Array(UnsafeBufferPointer(start: p, count: n)), shape)
        }
        return ["gl": rd(sc.gl, M * E, [M, E]),
                "inds": rdI(sc.inds, M * Ktop, [M * Ktop]),
                "scores": rd(sc.scores, M * Ktop, [M, Ktop]),
                "g": rd(sc.g, M * Ktop * I, [M * Ktop, I]),
                "u": rd(sc.u, M * Ktop * I, [M * Ktop, I]),
                "h": rd(sc.h, M * Ktop * I, [M * Ktop, I]),
                "d": rd(sc.d, M * Ktop * H, [M * Ktop, H]),
                "y": rd(sc.y, M * H, [M, H]),
                "sg": rd(sc.sg, M * I, [M, I]),
                "su": rd(sc.su, M * I, [M, I]),
                "shAct": rd(sc.shAct, M * I, [M, I]),
                "sharedY": rd(sc.sharedY, M * H, [M, H]),
                "sgl": rd(sc.sgl, M * 8, [M, 8]),
                "out": rd(outBuf, M * H, [M, H])]
    }

    /// テスト支援: fused MoE block 単発実行(単一 CB + 常駐中間)。moeBlockRows(metalRoute) と bit 一致すべき。
    public static func fusedMoEBlockRows(_ x: MLXArray, _ w: RawVerifyForward.MoEBlockW,
                                         M: Int, E: Int, I: Int, Ktop: Int = 8) -> MLXArray? {
        guard let (device, queue) = RawMetalForward.ensure() else { return nil }
        let H = x.dim(-1)
        guard ensureMoEPipelines(E: E, Ktop: Ktop) else { return nil }
        guard let bufs = prepareMoEBlockBufs(w, device),
              let sc = makeMoEScratch(device, M: M, E: E, I: I, Ktop: Ktop, H: H),
              let bx = RawMetalForward.mtlBuf(x.asType(.float16), device),
              let outBuf = device.makeBuffer(length: M * H * 2, options: .storageModeShared) else { return nil }
        let cb = queue.makeCommandBuffer()!; let enc = cb.makeComputeCommandEncoder()!
        encodeMoEBlockRows(enc, x: bx, out: outBuf, w: bufs, sc: sc, M: M, E: E, I: I, Ktop: Ktop, H: H)
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        let ptr = outBuf.contents().bindMemory(to: Float16.self, capacity: M * H)
        return MLXArray(Array(UnsafeBufferPointer(start: ptr, count: M * H)), [M, H])
    }

    // ── Stage B(P3 続き): fused attention 層 — attnLayerRows 全段を単一 encoder 化 ──

    /// rmsnorm(既存 _rmsPipeline, MLX 逐語移植)を encode-only で提供。rows=threadgroup 数。
    /// weight は非 nil buffer(no-weight は ones を渡す)。promoteF32 は _rmsPipelineF32(out f32)。
    static func encodeRmsNormRows(_ enc: MTLComputeCommandEncoder, x: MTLBuffer, w: MTLBuffer, out: MTLBuffer,
                                  rows: Int, D: Int, eps: Float, promoteF32: Bool = false) {
        let p = promoteF32 ? RawMetalForward._rmsPipelineF32! : RawMetalForward._rmsPipeline!
        enc.setComputePipelineState(p)
        enc.setBuffer(x, offset: 0, index: 0); enc.setBuffer(w, offset: 0, index: 1); enc.setBuffer(out, offset: 0, index: 2)
        var ee = eps, asz = UInt32(D), ws = UInt32(1)
        enc.setBytes(&ee, length: 4, index: 3); enc.setBytes(&asz, length: 4, index: 4); enc.setBytes(&ws, length: 4, index: 5)
        RawMetalForward.bindStop(enc, 16)
        let tgNeeded = (D + 3) / 4
        let tgSize = ((tgNeeded + 31) / 32) * 32
        enc.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
    }

    /// rope_rows(既存 _ropeRowsPipeline, fast-math コンパイル)を encode-only で提供。
    static func encodeRopeRows(_ enc: MTLComputeCommandEncoder, x: MTLBuffer, out: MTLBuffer,
                               headDim: Int, ropeDim: Int, base: Float, startOffset: Int,
                               M: Int, numHeads: Int) {
        let p = RawMetalForward._ropeRowsPipeline!
        enc.setComputePipelineState(p)
        enc.setBuffer(x, offset: 0, index: 0); enc.setBuffer(out, offset: 0, index: 1)
        var h = UInt32(headDim), r = UInt32(ropeDim), b = base, so = UInt32(startOffset), nh = UInt32(numHeads)
        enc.setBytes(&h, length: 4, index: 2); enc.setBytes(&r, length: 4, index: 3)
        enc.setBytes(&b, length: 4, index: 4); enc.setBytes(&so, length: 4, index: 5)
        enc.setBytes(&nh, length: 4, index: 6)
        let total = M * numHeads * headDim
        enc.dispatchThreads(MTLSize(width: total, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: min(p.maxTotalThreadsPerThreadgroup, 256), height: 1, depth: 1))
    }

    /// sdpa_rows(既存 _sdpaRowsPipeline)を encode-only で提供。k/v は cache buffer [KV, maxLen, D]
    /// (stride を maxLen 基準で渡す — 論理位置の値列は composed と同一なので bit 一致)。
    static func encodeSdpaRows(_ enc: MTLComputeCommandEncoder, q: MTLBuffer, k: MTLBuffer, v: MTLBuffer, out: MTLBuffer,
                               H: Int, KV: Int, D: Int, baseLenPlus1: Int, M: Int, scale: Float, maxLen: Int) {
        let p = RawMetalForward._sdpaRowsPipeline!
        enc.setComputePipelineState(p)
        enc.setBuffer(q, offset: 0, index: 0); enc.setBuffer(k, offset: 0, index: 1)
        enc.setBuffer(v, offset: 0, index: 2); enc.setBuffer(out, offset: 0, index: 3)
        var gqa = Int32(H / KV), bn = Int32(baseLenPlus1)
        var khs = Int32(maxLen * D), kss = Int32(D), vhs = Int32(maxLen * D), vss = Int32(D), sc = scale
        enc.setBytes(&gqa, length: 4, index: 4); enc.setBytes(&bn, length: 4, index: 5)
        enc.setBytes(&khs, length: 4, index: 6); enc.setBytes(&kss, length: 4, index: 7)
        enc.setBytes(&vhs, length: 4, index: 8); enc.setBytes(&vss, length: 4, index: 9)
        enc.setBytes(&sc, length: 4, index: 10)
        enc.dispatchThreadgroups(MTLSize(width: H, height: M, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 1024, height: 1, depth: 1))
    }

    /// extract_q(既存 aux kernel, 純コピー)を encode-only で提供。qOut[R,qd2] の先頭 headDim 列 → q[R,headDim]。
    static func encodeExtractQ(_ enc: MTLComputeCommandEncoder, qOut: MTLBuffer, q: MTLBuffer,
                               headDim: Int, qd2: Int, total: Int) {
        let p = RawMetalForward._extractQPipeline!
        enc.setComputePipelineState(p)
        enc.setBuffer(qOut, offset: 0, index: 0); enc.setBuffer(q, offset: 0, index: 1)
        var hd = UInt32(headDim), q2 = UInt32(qd2), t = UInt32(total)
        enc.setBytes(&hd, length: 4, index: 2); enc.setBytes(&q2, length: 4, index: 3); enc.setBytes(&t, length: 4, index: 4)
        enc.dispatchThreads(MTLSize(width: total, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: min(p.maxTotalThreadsPerThreadgroup, 256), height: 1, depth: 1))
    }

    /// sigmoid_mul(既存 aux kernel): gated[i]=attnOut[i]·sigmoid(qOut[h·qd2+headDim+d])。
    static func encodeSigmoidMul(_ enc: MTLComputeCommandEncoder, attnOut: MTLBuffer, qOut: MTLBuffer, gated: MTLBuffer,
                                 headDim: Int, qd2: Int, total: Int) {
        let p = RawMetalForward._sigmoidMulPipeline!
        enc.setComputePipelineState(p)
        enc.setBuffer(attnOut, offset: 0, index: 0); enc.setBuffer(qOut, offset: 0, index: 1); enc.setBuffer(gated, offset: 0, index: 2)
        var hd = UInt32(headDim), q2 = UInt32(qd2), t = UInt32(total)
        enc.setBytes(&hd, length: 4, index: 3); enc.setBytes(&q2, length: 4, index: 4); enc.setBytes(&t, length: 4, index: 5)
        enc.dispatchThreads(MTLSize(width: total, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: min(p.maxTotalThreadsPerThreadgroup, 256), height: 1, depth: 1))
    }

    static func encodeWriteKVRows(_ enc: MTLComputeCommandEncoder, src: MTLBuffer, cache: MTLBuffer,
                                  KV: Int, D: Int, maxLen: Int, pos: Int, M: Int) {
        let p = _writeKVRowsPipeline!
        enc.setComputePipelineState(p)
        enc.setBuffer(src, offset: 0, index: 0); enc.setBuffer(cache, offset: 0, index: 1)
        var kv = UInt32(KV), dd = UInt32(D), ml = UInt32(maxLen), pp = UInt32(pos), t = UInt32(M * KV * D)
        enc.setBytes(&kv, length: 4, index: 2); enc.setBytes(&dd, length: 4, index: 3)
        enc.setBytes(&ml, length: 4, index: 4); enc.setBytes(&pp, length: 4, index: 5); enc.setBytes(&t, length: 4, index: 6)
        enc.dispatchThreads(MTLSize(width: M * KV * D, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: min(p.maxTotalThreadsPerThreadgroup, 256), height: 1, depth: 1))
    }

    /// per-op wrapper: sigmoid_mul を単発 CB で実行(composed attnLayerRows が使い fused と数値系共有)。
    /// attnOut[R, headDim] × qOut[R, qd2](gate 部 strided 読み)→ gated[R, headDim]。
    public static func sigmoidMulRaw(_ attnOut: MLXArray, _ qOut: MLXArray,
                                     headDim: Int, qd2: Int, total: Int) -> MLXArray? {
        guard let (device, queue) = RawMetalForward.ensure(), RawMetalForward.ensureAuxPipelines() else { return nil }
        guard let ba = RawMetalForward.mtlBuf(attnOut.asType(.float16), device),
              let bq = RawMetalForward.mtlBuf(qOut.asType(.float16), device),
              let outBuf = device.makeBuffer(length: total * 2, options: .storageModeShared) else { return nil }
        let cb = queue.makeCommandBuffer()!; let enc = cb.makeComputeCommandEncoder()!
        encodeSigmoidMul(enc, attnOut: ba, qOut: bq, gated: outBuf, headDim: headDim, qd2: qd2, total: total)
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        let ptr = outBuf.contents().bindMemory(to: Float16.self, capacity: total)
        return MLXArray(Array(UnsafeBufferPointer(start: ptr, count: total)), [total / headDim, headDim])
    }

    /// attention 層用の重み MTLBuffer 束。
    public struct AttnLayerBufs {
        let qW: MTLBuffer, qS: MTLBuffer, qB: MTLBuffer
        let kW: MTLBuffer, kS: MTLBuffer, kB: MTLBuffer
        let vW: MTLBuffer, vS: MTLBuffer, vB: MTLBuffer
        let oW: MTLBuffer, oS: MTLBuffer, oB: MTLBuffer
        let qNorm: MTLBuffer, kNorm: MTLBuffer
    }

    static func prepareAttnLayerBufs(_ w: RawVerifyForward.AttnLayerW, _ device: MTLDevice) -> AttnLayerBufs? {
        func trio(_ q: MLXArray, _ s: MLXArray, _ b: MLXArray) -> (MTLBuffer, MTLBuffer, MTLBuffer)? {
            guard let bq = RawMetalForward.mtlBuf(q, device),
                  let bs = RawMetalForward.mtlBuf(s.asType(.float16), device),
                  let bb = RawMetalForward.mtlBuf(b.asType(.float16), device) else { return nil }
            return (bq, bs, bb)
        }
        guard let q = trio(w.qWq, w.qSc, w.qBi), let k = trio(w.kWq, w.kSc, w.kBi),
              let v = trio(w.vWq, w.vSc, w.vBi), let o = trio(w.oWq, w.oSc, w.oBi),
              let qn = RawMetalForward.mtlBuf(w.qNorm.asType(.float16), device),
              let kn = RawMetalForward.mtlBuf(w.kNorm.asType(.float16), device) else { return nil }
        return AttnLayerBufs(qW: q.0, qS: q.1, qB: q.2, kW: k.0, kS: k.1, kB: k.2,
                             vW: v.0, vS: v.1, vB: v.2, oW: o.0, oS: o.1, oB: o.2,
                             qNorm: qn, kNorm: kn)
    }

    /// attention 層の常駐 scratch(fused 中間)。
    public struct AttnScratch {
        let qOut: MTLBuffer, kOut: MTLBuffer, vOut: MTLBuffer
        let qX: MTLBuffer, qN: MTLBuffer, kN: MTLBuffer
        let qRot: MTLBuffer, kRot: MTLBuffer
        let attnOut: MTLBuffer, gated: MTLBuffer
    }

    static func makeAttnScratch(_ device: MTLDevice, M: Int, numHeads: Int, numKV: Int, headDim: Int) -> AttnScratch? {
        func buf(_ n: Int) -> MTLBuffer? { device.makeBuffer(length: n, options: .storageModeShared) }
        let qd2 = 2 * headDim
        guard let qOut = buf(M * numHeads * qd2 * 2), let kOut = buf(M * numKV * headDim * 2),
              let vOut = buf(M * numKV * headDim * 2),
              let qX = buf(M * numHeads * headDim * 2), let qN = buf(M * numHeads * headDim * 2),
              let kN = buf(M * numKV * headDim * 2),
              let qRot = buf(M * numHeads * headDim * 2), let kRot = buf(M * numKV * headDim * 2),
              let attnOut = buf(M * numHeads * headDim * 2), let gated = buf(M * numHeads * headDim * 2)
        else { return nil }
        return AttnScratch(qOut: qOut, kOut: kOut, vOut: vOut, qX: qX, qN: qN, kN: kN,
                           qRot: qRot, kRot: kRot, attnOut: attnOut, gated: gated)
    }

    /// 層別 KV cache 常駐 buffer([KV, maxLen, D] f16)+ 現在長。
    public final class KVCacheBufs {
        let kCache: MTLBuffer, vCache: MTLBuffer
        let maxLen: Int, KV: Int, D: Int
        var len: Int
        init(kCache: MTLBuffer, vCache: MTLBuffer, maxLen: Int, KV: Int, D: Int, len: Int) {
            self.kCache = kCache; self.vCache = vCache; self.maxLen = maxLen; self.KV = KV; self.D = D; self.len = len
        }
    }

    /// KV cache buffer を確保し、初期 cache([KV, len0, D] MLX)を先頭に preload。
    static func makeKVCacheBufs(_ device: MTLDevice, kInit: MLXArray?, vInit: MLXArray?,
                                maxLen: Int, KV: Int, D: Int) -> KVCacheBufs? {
        guard let kB = device.makeBuffer(length: KV * maxLen * D * 2, options: .storageModeShared),
              let vB = device.makeBuffer(length: KV * maxLen * D * 2, options: .storageModeShared) else { return nil }
        var len0 = 0
        if let k0 = kInit, let v0 = vInit {
            len0 = k0.dim(1)
            let kf = k0.asType(.float16).reshaped([-1]); kf.eval()
            let vf = v0.asType(.float16).reshaped([-1]); vf.eval()
            let kArr = kf.asArray(Float16.self), vArr = vf.asArray(Float16.self)
            let kp = kB.contents().bindMemory(to: Float16.self, capacity: KV * maxLen * D)
            let vp = vB.contents().bindMemory(to: Float16.self, capacity: KV * maxLen * D)
            for h in 0 ..< KV {
                for t in 0 ..< len0 {
                    for dd in 0 ..< D {
                        kp[h * maxLen * D + t * D + dd] = kArr[(h * len0 + t) * D + dd]
                        vp[h * maxLen * D + t * D + dd] = vArr[(h * len0 + t) * D + dd]
                    }
                }
            }
        }
        return KVCacheBufs(kCache: kB, vCache: vB, maxLen: maxLen, KV: KV, D: D, len: len0)
    }

    /// cache buffer の先頭 len 位置を [KV, len, D] MLXArray として読み出す(テスト比較用)。
    static func readKVCache(_ kv: KVCacheBufs) -> (MLXArray, MLXArray) {
        let KV = kv.KV, D = kv.D, maxLen = kv.maxLen, len = kv.len
        let kp = kv.kCache.contents().bindMemory(to: Float16.self, capacity: KV * maxLen * D)
        let vp = kv.vCache.contents().bindMemory(to: Float16.self, capacity: KV * maxLen * D)
        var kArr = [Float16](repeating: 0, count: KV * len * D)
        var vArr = [Float16](repeating: 0, count: KV * len * D)
        for h in 0 ..< KV {
            for t in 0 ..< len {
                for dd in 0 ..< D {
                    kArr[(h * len + t) * D + dd] = kp[h * maxLen * D + t * D + dd]
                    vArr[(h * len + t) * D + dd] = vp[h * maxLen * D + t * D + dd]
                }
            }
        }
        return (MLXArray(kArr, [KV, len, D]), MLXArray(vArr, [KV, len, D]))
    }

    /// attention 層 × M 行の全段を既存 encoder に encode。演算列は attnLayerRows と 1:1。
    /// kv.len は encode 時点の baseLen として読み、呼び出し側が encode 後に kv.len += M する。
    static func encodeAttnLayerRows(_ enc: MTLComputeCommandEncoder, x: MTLBuffer, out: MTLBuffer,
                                    w: AttnLayerBufs, sc: AttnScratch, kv: KVCacheBufs,
                                    M: Int, H: Int,
                                    numHeads: Int = 16, numKV: Int = 2, headDim: Int = 256,
                                    ropeDim: Int = 64, ropeBase: Float = 1e7, eps: Float = 1e-6) {
        let baseLen = kv.len
        let qd2 = 2 * headDim
        let scale = Float(pow(Double(headDim), -0.5))
        // ① q(+gate)/k/v projection
        encodeQmmRows(enc, w: w.qW, scales: w.qS, biases: w.qB, x: x, out: sc.qOut, M: M, K: H, N: numHeads * qd2)
        encodeQmmRows(enc, w: w.kW, scales: w.kS, biases: w.kB, x: x, out: sc.kOut, M: M, K: H, N: numKV * headDim)
        encodeQmmRows(enc, w: w.vW, scales: w.vS, biases: w.vB, x: x, out: sc.vOut, M: M, K: H, N: numKV * headDim)
        // ② queries 抽出(純コピー)→ qk-norm
        encodeExtractQ(enc, qOut: sc.qOut, q: sc.qX, headDim: headDim, qd2: qd2, total: M * numHeads * headDim)
        encodeRmsNormRows(enc, x: sc.qX, w: w.qNorm, out: sc.qN, rows: M * numHeads, D: headDim, eps: eps)
        encodeRmsNormRows(enc, x: sc.kOut, w: w.kNorm, out: sc.kN, rows: M * numKV, D: headDim, eps: eps)
        // ③ RoPE(行 m の位置 = baseLen + m)
        encodeRopeRows(enc, x: sc.qN, out: sc.qRot, headDim: headDim, ropeDim: ropeDim, base: ropeBase,
                       startOffset: baseLen, M: M, numHeads: numHeads)
        encodeRopeRows(enc, x: sc.kN, out: sc.kRot, headDim: headDim, ropeDim: ropeDim, base: ropeBase,
                       startOffset: baseLen, M: M, numHeads: numKV)
        // ④ cache 散布(post-RoPE k / raw v)
        encodeWriteKVRows(enc, src: sc.kRot, cache: kv.kCache, KV: numKV, D: headDim, maxLen: kv.maxLen, pos: baseLen, M: M)
        encodeWriteKVRows(enc, src: sc.vOut, cache: kv.vCache, KV: numKV, D: headDim, maxLen: kv.maxLen, pos: baseLen, M: M)
        // ⑤ SDPA(行 m は先頭 baseLen+m+1 key)
        encodeSdpaRows(enc, q: sc.qRot, k: kv.kCache, v: kv.vCache, out: sc.attnOut,
                       H: numHeads, KV: numKV, D: headDim, baseLenPlus1: baseLen + 1, M: M, scale: scale, maxLen: kv.maxLen)
        // ⑥ sigmoid gate → o_proj
        encodeSigmoidMul(enc, attnOut: sc.attnOut, qOut: sc.qOut, gated: sc.gated,
                         headDim: headDim, qd2: qd2, total: M * numHeads * headDim)
        encodeQmmRows(enc, w: w.oW, scales: w.oS, biases: w.oB, x: sc.gated, out: out, M: M, K: numHeads * headDim, N: H)
    }

    /// Stage B の pipeline warm。
    static func ensureAttnPipelines() -> Bool {
        ensureQmmPipeline()
        guard RawMetalForward.ensureAuxPipelines(), ensureRowsAuxPipelines() else { return false }
        if RawMetalForward._rmsPipeline == nil {
            let x = MLXRandom.normal([1, 128]).asType(.float16); x.eval()
            _ = RawMetalForward.rmsNorm(x, nil, eps: 1e-6, D: 128)
        }
        if RawMetalForward._ropeRowsPipeline == nil {
            let x = MLXRandom.normal([1, 256]).asType(.float16); x.eval()
            _ = RawMetalForward.ropeRows(x, headDim: 256, ropeDim: 64, base: 1e7, startOffset: 0, M: 1, numHeads: 1)
        }
        if RawMetalForward._sdpaRowsPipeline == nil {
            let q = MLXRandom.normal([1, 256]).asType(.float16)
            let k = MLXRandom.normal([1, 2, 256]).asType(.float16)
            let v = MLXRandom.normal([1, 2, 256]).asType(.float16)
            MLX.eval([q, k, v])
            _ = RawMetalForward.sdpaRows(q, k, v, H: 1, KV: 1, D: 256, baseLen: 2, M: 1, scale: 1.0)
        }
        return RawMetalForward._rmsPipeline != nil && RawMetalForward._ropeRowsPipeline != nil
            && RawMetalForward._sdpaRowsPipeline != nil
    }

    /// テスト支援: fused attn 層 単発実行(単一 CB)。attnLayerRows と出力+cache が bit 一致すべき。
    public static func fusedAttnLayerRows(_ x: MLXArray, _ w: RawVerifyForward.AttnLayerW,
                                          kInit: MLXArray, vInit: MLXArray, maxLen: Int, M: Int,
                                          numHeads: Int = 16, numKV: Int = 2, headDim: Int = 256,
                                          ropeDim: Int = 64, ropeBase: Float = 1e7, eps: Float = 1e-6)
        -> (out: MLXArray, kCache: MLXArray, vCache: MLXArray)? {
        guard let (device, queue) = RawMetalForward.ensure(), ensureAttnPipelines() else { return nil }
        let H = x.dim(-1)
        guard let bufs = prepareAttnLayerBufs(w, device),
              let sc = makeAttnScratch(device, M: M, numHeads: numHeads, numKV: numKV, headDim: headDim),
              let kv = makeKVCacheBufs(device, kInit: kInit, vInit: vInit, maxLen: maxLen, KV: numKV, D: headDim),
              let bx = RawMetalForward.mtlBuf(x.asType(.float16), device),
              let outBuf = device.makeBuffer(length: M * H * 2, options: .storageModeShared) else { return nil }
        let cb = queue.makeCommandBuffer()!; let enc = cb.makeComputeCommandEncoder()!
        encodeAttnLayerRows(enc, x: bx, out: outBuf, w: bufs, sc: sc, kv: kv, M: M, H: H,
                            numHeads: numHeads, numKV: numKV, headDim: headDim,
                            ropeDim: ropeDim, ropeBase: ropeBase, eps: eps)
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        kv.len += M
        let ptr = outBuf.contents().bindMemory(to: Float16.self, capacity: M * H)
        let out = MLXArray(Array(UnsafeBufferPointer(start: ptr, count: M * H)), [M, H])
        let (kc, vc) = readKVCache(kv)
        return (out, kc, vc)
    }

    // ── Stage C(P3 続き): fused GDN 層 — gdnLayerRows 全段を単一 encoder 化 ──

    static func encodeConvHistRows(_ enc: MTLComputeCommandEncoder, hist: MTLBuffer, qkv: MTLBuffer,
                                   w: MTLBuffer, out: MTLBuffer, K: Int, C: Int, M: Int) {
        let p = _convHistRowsPipeline!
        enc.setComputePipelineState(p)
        enc.setBuffer(hist, offset: 0, index: 0); enc.setBuffer(qkv, offset: 0, index: 1)
        enc.setBuffer(w, offset: 0, index: 2); enc.setBuffer(out, offset: 0, index: 3)
        var kk = UInt32(K), cc = UInt32(C)
        enc.setBytes(&kk, length: 4, index: 4); enc.setBytes(&cc, length: 4, index: 5)
        enc.dispatchThreads(MTLSize(width: C, height: M, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: min(p.maxTotalThreadsPerThreadgroup, 256), height: 1, depth: 1))
    }

    static func encodeShiftConvRows(_ enc: MTLComputeCommandEncoder, hist: MTLBuffer, qkv: MTLBuffer,
                                    K: Int, C: Int, M: Int) {
        let p = _shiftConvRowsPipeline!
        enc.setComputePipelineState(p)
        enc.setBuffer(hist, offset: 0, index: 0); enc.setBuffer(qkv, offset: 0, index: 1)
        var kk = UInt32(K), cc = UInt32(C), mm = UInt32(M)
        enc.setBytes(&kk, length: 4, index: 2); enc.setBytes(&cc, length: 4, index: 3); enc.setBytes(&mm, length: 4, index: 4)
        enc.dispatchThreads(MTLSize(width: C, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: min(p.maxTotalThreadsPerThreadgroup, 256), height: 1, depth: 1))
    }

    static func encodeSliceRows(_ enc: MTLComputeCommandEncoder, input: MTLBuffer, out: MTLBuffer,
                                off: Int, W: Int, stride: Int, M: Int) {
        let p = _sliceRowsPipeline!
        enc.setComputePipelineState(p)
        enc.setBuffer(input, offset: 0, index: 0); enc.setBuffer(out, offset: 0, index: 1)
        var o = UInt32(off), ww = UInt32(W), st = UInt32(stride), t = UInt32(M * W)
        enc.setBytes(&o, length: 4, index: 2); enc.setBytes(&ww, length: 4, index: 3)
        enc.setBytes(&st, length: 4, index: 4); enc.setBytes(&t, length: 4, index: 5)
        enc.dispatchThreads(MTLSize(width: M * W, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: min(p.maxTotalThreadsPerThreadgroup, 256), height: 1, depth: 1))
    }

    static func encodeComputeGBetaRows(_ enc: MTLComputeCommandEncoder, a: MTLBuffer, b: MTLBuffer,
                                       aLog: MTLBuffer, dtBias: MTLBuffer, g: MTLBuffer, beta: MTLBuffer,
                                       Hv: Int, M: Int) {
        let p = _computeGBetaRowsPipeline!
        enc.setComputePipelineState(p)
        enc.setBuffer(a, offset: 0, index: 0); enc.setBuffer(b, offset: 0, index: 1)
        enc.setBuffer(aLog, offset: 0, index: 2); enc.setBuffer(dtBias, offset: 0, index: 3)
        enc.setBuffer(g, offset: 0, index: 4); enc.setBuffer(beta, offset: 0, index: 5)
        var hv = UInt32(Hv), t = UInt32(M * Hv)
        enc.setBytes(&hv, length: 4, index: 6); enc.setBytes(&t, length: 4, index: 7)
        enc.dispatchThreads(MTLSize(width: M * Hv, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: min(p.maxTotalThreadsPerThreadgroup, 256), height: 1, depth: 1))
    }

    /// scale_mul(既存 aux kernel, in-place x[i]=(half)s·x[i])を encode-only で提供。
    static func encodeScaleMul(_ enc: MTLComputeCommandEncoder, x: MTLBuffer, s: Float, total: Int) {
        let p = RawMetalForward._scalePipeline!
        enc.setComputePipelineState(p)
        enc.setBuffer(x, offset: 0, index: 0)
        var ss = s, t = UInt32(total)
        enc.setBytes(&ss, length: 4, index: 1); enc.setBytes(&t, length: 4, index: 2)
        enc.dispatchThreads(MTLSize(width: total, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: min(p.maxTotalThreadsPerThreadgroup, 256), height: 1, depth: 1))
    }

    /// gate/gate16(既存 aux kernel): outV=(half)(silu(z)·normed)。promote=normed f32。
    static func encodeGate(_ enc: MTLComputeCommandEncoder, z: MTLBuffer, normed: MTLBuffer, outV: MTLBuffer,
                           total: Int, promote: Bool) {
        let p = promote ? RawMetalForward._gatePipeline! : RawMetalForward._gate16Pipeline!
        enc.setComputePipelineState(p)
        enc.setBuffer(z, offset: 0, index: 0); enc.setBuffer(normed, offset: 0, index: 1); enc.setBuffer(outV, offset: 0, index: 2)
        var t = UInt32(total); enc.setBytes(&t, length: 4, index: 3)
        enc.dispatchThreads(MTLSize(width: total, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: min(p.maxTotalThreadsPerThreadgroup, 256), height: 1, depth: 1))
    }

    /// gated_delta_step(既存 _recurPipeline, T=M 逐次)を encode-only で提供。
    static func encodeGatedDeltaStepRows(_ enc: MTLComputeCommandEncoder,
                                         q: MTLBuffer, k: MTLBuffer, v: MTLBuffer,
                                         g: MTLBuffer, beta: MTLBuffer,
                                         stateIn: MTLBuffer, stateOut: MTLBuffer, y: MTLBuffer,
                                         T: Int, B: Int, Hv: Int, Dv: Int) {
        let p = RawMetalForward._recurPipeline!
        enc.setComputePipelineState(p)
        enc.setBuffer(q, offset: 0, index: 0); enc.setBuffer(k, offset: 0, index: 1); enc.setBuffer(v, offset: 0, index: 2)
        enc.setBuffer(g, offset: 0, index: 3); enc.setBuffer(beta, offset: 0, index: 4); enc.setBuffer(stateIn, offset: 0, index: 5)
        var tt = Int32(T); enc.setBytes(&tt, length: 4, index: 6)
        enc.setBuffer(y, offset: 0, index: 7); enc.setBuffer(stateOut, offset: 0, index: 8)
        RawMetalForward.bindStop(enc, 16)
        enc.dispatchThreads(MTLSize(width: 32, height: Dv, depth: B * Hv),
                            threadsPerThreadgroup: MTLSize(width: 32, height: 4, depth: 1))
    }

    /// per-op wrapper: compute_g_beta_rows(composed gdnLayerRows が使い fused と g/β 数値系を共有)。
    /// aP/bP[M,Hv] f16, aLog/dtBias[Hv] f32 → (g, beta) 各 [1,M,Hv] f32。
    public static func computeGBetaRowsRaw(_ aP: MLXArray, _ bP: MLXArray, _ aLog: MLXArray, _ dtBias: MLXArray,
                                           M: Int, Hv: Int) -> (MLXArray, MLXArray)? {
        guard let (device, queue) = RawMetalForward.ensure(), ensureRowsAuxPipelines() else { return nil }
        guard let ba = RawMetalForward.mtlBuf(aP.asType(.float16), device),
              let bb = RawMetalForward.mtlBuf(bP.asType(.float16), device),
              let bal = RawMetalForward.mtlBuf(aLog.asType(.float32), device),
              let bdt = RawMetalForward.mtlBuf(dtBias.asType(.float32), device),
              let gBuf = device.makeBuffer(length: M * Hv * 4, options: .storageModeShared),
              let betaBuf = device.makeBuffer(length: M * Hv * 4, options: .storageModeShared) else { return nil }
        let cb = queue.makeCommandBuffer()!; let enc = cb.makeComputeCommandEncoder()!
        encodeComputeGBetaRows(enc, a: ba, b: bb, aLog: bal, dtBias: bdt, g: gBuf, beta: betaBuf, Hv: Hv, M: M)
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        let gp = gBuf.contents().bindMemory(to: Float.self, capacity: M * Hv)
        let bp = betaBuf.contents().bindMemory(to: Float.self, capacity: M * Hv)
        return (MLXArray(Array(UnsafeBufferPointer(start: gp, count: M * Hv)), [1, M, Hv]),
                MLXArray(Array(UnsafeBufferPointer(start: bp, count: M * Hv)), [1, M, Hv]))
    }

    /// per-op wrapper: gate/gate16(composed gdnLayerRows が使い fused と silu-gate 数値系を共有)。
    /// z[M,valueDim] f16 × normed(f32 promote / f16)→ outV [total] f16。
    public static func gateRaw(_ z: MLXArray, _ normed: MLXArray, promote: Bool, total: Int) -> MLXArray? {
        guard let (device, queue) = RawMetalForward.ensure(), RawMetalForward.ensureAuxPipelines() else { return nil }
        guard let bz = RawMetalForward.mtlBuf(z.asType(.float16), device),
              let bn = RawMetalForward.mtlBuf(normed.asType(promote ? .float32 : .float16), device),
              let outBuf = device.makeBuffer(length: total * 2, options: .storageModeShared) else { return nil }
        let cb = queue.makeCommandBuffer()!; let enc = cb.makeComputeCommandEncoder()!
        encodeGate(enc, z: bz, normed: bn, outV: outBuf, total: total, promote: promote)
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        let ptr = outBuf.contents().bindMemory(to: Float16.self, capacity: total)
        return MLXArray(Array(UnsafeBufferPointer(start: ptr, count: total)), [total])
    }

    /// GDN 層用の重み MTLBuffer 束。
    public struct GdnLayerBufs {
        let qkvW: MTLBuffer, qkvS: MTLBuffer, qkvB: MTLBuffer
        let zW: MTLBuffer, zS: MTLBuffer, zB: MTLBuffer
        let bW: MTLBuffer, bS: MTLBuffer, bB: MTLBuffer
        let aW: MTLBuffer, aS: MTLBuffer, aB: MTLBuffer
        let outW: MTLBuffer, outS: MTLBuffer, outB: MTLBuffer
        let conv1dW: MTLBuffer          // f32 [C, K]
        let normWeight: MTLBuffer       // f16 or f32(promote に対応)
        let promoteRMS: Bool
        let aLog: MTLBuffer, dtBias: MTLBuffer   // f32 [Hv]
        let onesDk: MTLBuffer           // f16 ones [Dk](qk-norm no-weight 用)
    }

    static func prepareGdnLayerBufs(_ w: RawVerifyForward.GDNLayerW, Dk: Int, _ device: MTLDevice) -> GdnLayerBufs? {
        func trio(_ q: MLXArray, _ s: MLXArray, _ b: MLXArray) -> (MTLBuffer, MTLBuffer, MTLBuffer)? {
            guard let bq = RawMetalForward.mtlBuf(q, device),
                  let bs = RawMetalForward.mtlBuf(s.asType(.float16), device),
                  let bb = RawMetalForward.mtlBuf(b.asType(.float16), device) else { return nil }
            return (bq, bs, bb)
        }
        let promote = (w.normWeight.dtype == .float32)
        let ones = MLXArray.ones([Dk]).asType(.float16); ones.eval()
        guard let qkv = trio(w.qkvWq, w.qkvSc, w.qkvBi), let z = trio(w.zWq, w.zSc, w.zBi),
              let b = trio(w.bWq, w.bSc, w.bBi), let a = trio(w.aWq, w.aSc, w.aBi),
              let o = trio(w.outWq, w.outSc, w.outBi),
              let cw = RawMetalForward.mtlBuf(w.conv1dW.asType(.float32), device),
              let nw = RawMetalForward.mtlBuf(w.normWeight.asType(promote ? .float32 : .float16), device),
              let al = RawMetalForward.mtlBuf(w.aLog.asType(.float32), device),
              let dt = RawMetalForward.mtlBuf(w.dtBias.asType(.float32), device),
              let od = RawMetalForward.mtlBuf(ones, device) else { return nil }
        return GdnLayerBufs(qkvW: qkv.0, qkvS: qkv.1, qkvB: qkv.2, zW: z.0, zS: z.1, zB: z.2,
                            bW: b.0, bS: b.1, bB: b.2, aW: a.0, aS: a.1, aB: a.2,
                            outW: o.0, outS: o.1, outB: o.2,
                            conv1dW: cw, normWeight: nw, promoteRMS: promote, aLog: al, dtBias: dt, onesDk: od)
    }

    /// GDN 層の常駐 cache(conv hist [K-1,C] f16 + rec state [1,Hv,Dv,Dk] f32 ping-pong)。
    public final class GdnCacheBufs {
        let convHist: MTLBuffer
        var state: MTLBuffer       // 現在 state(encode 入力)
        var stateOut: MTLBuffer    // encode 出力(encode 後に swap)
        let K: Int, C: Int, Hv: Int, Dv: Int, Dk: Int
        init(convHist: MTLBuffer, state: MTLBuffer, stateOut: MTLBuffer, K: Int, C: Int, Hv: Int, Dv: Int, Dk: Int) {
            self.convHist = convHist; self.state = state; self.stateOut = stateOut
            self.K = K; self.C = C; self.Hv = Hv; self.Dv = Dv; self.Dk = Dk
        }
        func swapState() { let t = state; state = stateOut; stateOut = t }
    }

    static func makeGdnCacheBufs(_ device: MTLDevice, convInit: MLXArray?, recInit: MLXArray?,
                                 K: Int, C: Int, Hv: Int, Dv: Int, Dk: Int) -> GdnCacheBufs? {
        guard let hist = device.makeBuffer(length: (K - 1) * C * 2, options: .storageModeShared),
              let st = device.makeBuffer(length: Hv * Dv * Dk * 4, options: .storageModeShared),
              let stOut = device.makeBuffer(length: Hv * Dv * Dk * 4, options: .storageModeShared) else { return nil }
        if let c0 = convInit {
            let cf = c0.asType(.float16).reshaped([-1]); cf.eval()
            let arr = cf.asArray(Float16.self)
            hist.contents().bindMemory(to: Float16.self, capacity: (K - 1) * C)
                .update(from: arr, count: min(arr.count, (K - 1) * C))
        }
        if let r0 = recInit {
            let rf = r0.asType(.float32).reshaped([-1]); rf.eval()
            let arr = rf.asArray(Float.self)
            st.contents().bindMemory(to: Float.self, capacity: Hv * Dv * Dk)
                .update(from: arr, count: min(arr.count, Hv * Dv * Dk))
        }
        return GdnCacheBufs(convHist: hist, state: st, stateOut: stOut, K: K, C: C, Hv: Hv, Dv: Dv, Dk: Dk)
    }

    static func readGdnCache(_ c: GdnCacheBufs) -> (MLXArray, MLXArray) {
        let hp = c.convHist.contents().bindMemory(to: Float16.self, capacity: (c.K - 1) * c.C)
        let sp = c.state.contents().bindMemory(to: Float.self, capacity: c.Hv * c.Dv * c.Dk)
        return (MLXArray(Array(UnsafeBufferPointer(start: hp, count: (c.K - 1) * c.C)), [c.K - 1, c.C]),
                MLXArray(Array(UnsafeBufferPointer(start: sp, count: c.Hv * c.Dv * c.Dk)), [1, c.Hv, c.Dv, c.Dk]))
    }

    /// GDN 層の常駐 scratch。
    public struct GdnScratch {
        let qkv: MTLBuffer, z: MTLBuffer, bP: MTLBuffer, aP: MTLBuffer
        let convOut: MTLBuffer
        let q1: MTLBuffer, k1: MTLBuffer, v1: MTLBuffer
        let qn: MTLBuffer, kn: MTLBuffer
        let g: MTLBuffer, beta: MTLBuffer
        let coreOut: MTLBuffer, normed: MTLBuffer, outV: MTLBuffer
    }

    static func makeGdnScratch(_ device: MTLDevice, M: Int, C: Int, keyDim: Int, valueDim: Int,
                               Hv: Int, promote: Bool) -> GdnScratch? {
        func buf(_ n: Int) -> MTLBuffer? { device.makeBuffer(length: n, options: .storageModeShared) }
        guard let qkv = buf(M * C * 2), let z = buf(M * valueDim * 2),
              let bP = buf(M * Hv * 2), let aP = buf(M * Hv * 2),
              let convOut = buf(M * C * 2),
              let q1 = buf(M * keyDim * 2), let k1 = buf(M * keyDim * 2), let v1 = buf(M * valueDim * 2),
              let qn = buf(M * keyDim * 2), let kn = buf(M * keyDim * 2),
              let g = buf(M * Hv * 4), let beta = buf(M * Hv * 4),
              let coreOut = buf(M * valueDim * 2),
              let normed = buf(M * valueDim * (promote ? 4 : 2)),
              let outV = buf(M * valueDim * 2) else { return nil }
        return GdnScratch(qkv: qkv, z: z, bP: bP, aP: aP, convOut: convOut, q1: q1, k1: k1, v1: v1,
                          qn: qn, kn: kn, g: g, beta: beta, coreOut: coreOut, normed: normed, outV: outV)
    }

    /// GDN 層 × M 行の全段を既存 encoder に encode。演算列は gdnLayerRows と 1:1。
    /// encode 後に cache.swapState() を呼ぶこと(state ping-pong)。
    static func encodeGdnLayerRows(_ enc: MTLComputeCommandEncoder, x: MTLBuffer, out: MTLBuffer,
                                   w: GdnLayerBufs, sc: GdnScratch, cache: GdnCacheBufs,
                                   M: Int, H: Int,
                                   numKHeads: Int = 16, numVHeads: Int = 32,
                                   headKDim: Int = 128, headVDim: Int = 128,
                                   convKernel: Int = 4, eps: Float = 1e-6) {
        let keyDim = headKDim * numKHeads
        let valueDim = headVDim * numVHeads
        let convDim = keyDim * 2 + valueDim
        // ① in_proj ×4
        encodeQmmRows(enc, w: w.qkvW, scales: w.qkvS, biases: w.qkvB, x: x, out: sc.qkv, M: M, K: H, N: convDim)
        encodeQmmRows(enc, w: w.zW, scales: w.zS, biases: w.zB, x: x, out: sc.z, M: M, K: H, N: valueDim)
        encodeQmmRows(enc, w: w.bW, scales: w.bS, biases: w.bB, x: x, out: sc.bP, M: M, K: H, N: numVHeads)
        encodeQmmRows(enc, w: w.aW, scales: w.aS, biases: w.aB, x: x, out: sc.aP, M: M, K: H, N: numVHeads)
        // ② conv(hist 直読み)→ hist shift 更新(conv の後=旧 hist を読む)
        encodeConvHistRows(enc, hist: cache.convHist, qkv: sc.qkv, w: w.conv1dW, out: sc.convOut,
                           K: convKernel, C: convDim, M: M)
        encodeShiftConvRows(enc, hist: cache.convHist, qkv: sc.qkv, K: convKernel, C: convDim, M: M)
        // ③ split q/k/v(純コピー)
        encodeSliceRows(enc, input: sc.convOut, out: sc.q1, off: 0, W: keyDim, stride: convDim, M: M)
        encodeSliceRows(enc, input: sc.convOut, out: sc.k1, off: keyDim, W: keyDim, stride: convDim, M: M)
        encodeSliceRows(enc, input: sc.convOut, out: sc.v1, off: 2 * keyDim, W: valueDim, stride: convDim, M: M)
        // ④ qk-norm(no-weight)+ scalar scale
        let invScale = Float(pow(Double(headKDim), -0.5))
        encodeRmsNormRows(enc, x: sc.q1, w: w.onesDk, out: sc.qn, rows: M * numKHeads, D: headKDim, eps: eps)
        encodeRmsNormRows(enc, x: sc.k1, w: w.onesDk, out: sc.kn, rows: M * numKHeads, D: headKDim, eps: eps)
        encodeScaleMul(enc, x: sc.qn, s: invScale * invScale, total: M * keyDim)
        encodeScaleMul(enc, x: sc.kn, s: invScale, total: M * keyDim)
        // ⑤ g/β → recurrence(in-kernel T=M 逐次)
        encodeComputeGBetaRows(enc, a: sc.aP, b: sc.bP, aLog: w.aLog, dtBias: w.dtBias,
                               g: sc.g, beta: sc.beta, Hv: numVHeads, M: M)
        encodeGatedDeltaStepRows(enc, q: sc.qn, k: sc.kn, v: sc.v1, g: sc.g, beta: sc.beta,
                                 stateIn: cache.state, stateOut: cache.stateOut, y: sc.coreOut,
                                 T: M, B: 1, Hv: numVHeads, Dv: headVDim)
        // ⑥ RMSNormGated → silu(z)·normed
        encodeRmsNormRows(enc, x: sc.coreOut, w: w.normWeight, out: sc.normed,
                          rows: M * numVHeads, D: headVDim, eps: eps, promoteF32: w.promoteRMS)
        encodeGate(enc, z: sc.z, normed: sc.normed, outV: sc.outV, total: M * valueDim, promote: w.promoteRMS)
        // ⑦ out_proj
        encodeQmmRows(enc, w: w.outW, scales: w.outS, biases: w.outB, x: sc.outV, out: out, M: M, K: valueDim, N: H)
    }

    /// Stage C の pipeline warm(recurrent は #define 次元固定なので実次元で warm)。
    static func ensureGdnPipelines(Hk: Int = 16, Dk: Int = 128, Hv: Int = 32, Dv: Int = 128) -> Bool {
        ensureQmmPipeline()
        guard RawMetalForward.ensureAuxPipelines(), ensureRowsAuxPipelines() else { return false }
        if RawMetalForward._rmsPipeline == nil {
            let x = MLXRandom.normal([1, 128]).asType(.float16); x.eval()
            _ = RawMetalForward.rmsNorm(x, nil, eps: 1e-6, D: 128)
        }
        if RawMetalForward._recurPipeline == nil {
            let q = MLXArray.zeros([1, 1, Hk, Dk]).asType(.float16)
            let v = MLXArray.zeros([1, 1, Hv, Dv]).asType(.float16)
            let g = MLXArray.zeros([1, 1, Hv]).asType(.float32)
            let st = MLXArray.zeros([1, Hv, Dv, Dk]).asType(.float32)
            MLX.eval([q, v, g, st])
            _ = RawMetalForward.recurrent(q, q, v, g: g, beta: g, state: st, B: 1, T: 1, Hk: Hk, Dk: Dk, Hv: Hv, Dv: Dv)
        }
        return RawMetalForward._rmsPipeline != nil && RawMetalForward._recurPipeline != nil
    }

    /// テスト支援: fused GDN 層 単発実行(単一 CB)。gdnLayerRows と出力+conv/rec state が bit 一致すべき。
    public static func fusedGdnLayerRows(_ x: MLXArray, _ w: RawVerifyForward.GDNLayerW,
                                         convInit: MLXArray, recInit: MLXArray, M: Int,
                                         numKHeads: Int = 16, numVHeads: Int = 32,
                                         headKDim: Int = 128, headVDim: Int = 128,
                                         convKernel: Int = 4, eps: Float = 1e-6)
        -> (out: MLXArray, convState: MLXArray, recState: MLXArray)? {
        guard let (device, queue) = RawMetalForward.ensure(),
              ensureGdnPipelines(Hk: numKHeads, Dk: headKDim, Hv: numVHeads, Dv: headVDim) else { return nil }
        let H = x.dim(-1)
        let keyDim = headKDim * numKHeads
        let valueDim = headVDim * numVHeads
        let convDim = keyDim * 2 + valueDim
        guard let bufs = prepareGdnLayerBufs(w, Dk: headKDim, device),
              let sc = makeGdnScratch(device, M: M, C: convDim, keyDim: keyDim, valueDim: valueDim,
                                      Hv: numVHeads, promote: bufs.promoteRMS),
              let cache = makeGdnCacheBufs(device, convInit: convInit, recInit: recInit,
                                           K: convKernel, C: convDim, Hv: numVHeads, Dv: headVDim, Dk: headKDim),
              let bx = RawMetalForward.mtlBuf(x.asType(.float16), device),
              let outBuf = device.makeBuffer(length: M * H * 2, options: .storageModeShared) else { return nil }
        let cb = queue.makeCommandBuffer()!; let enc = cb.makeComputeCommandEncoder()!
        encodeGdnLayerRows(enc, x: bx, out: outBuf, w: bufs, sc: sc, cache: cache, M: M, H: H,
                           numKHeads: numKHeads, numVHeads: numVHeads,
                           headKDim: headKDim, headVDim: headVDim, convKernel: convKernel, eps: eps)
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        cache.swapState()
        let ptr = outBuf.contents().bindMemory(to: Float16.self, capacity: M * H)
        let out = MLXArray(Array(UnsafeBufferPointer(start: ptr, count: M * H)), [M, H])
        let (cs, rs) = readGdnCache(cache)
        return (out, cs, rs)
    }

    // ── Stage D(P3 続き): 層全体 encode + 全層 1-CB forward ──

    /// resid_add(既存 aux kernel, in-place h[i]+=r[i])を encode-only で提供。
    static func encodeResidAdd(_ enc: MTLComputeCommandEncoder, h: MTLBuffer, r: MTLBuffer, total: Int) {
        let p = RawMetalForward._residAddPipeline!
        enc.setComputePipelineState(p)
        enc.setBuffer(h, offset: 0, index: 0); enc.setBuffer(r, offset: 0, index: 1)
        var t = UInt32(total); enc.setBytes(&t, length: 4, index: 2)
        enc.dispatchThreads(MTLSize(width: total, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: min(p.maxTotalThreadsPerThreadgroup, 256), height: 1, depth: 1))
    }

    /// verifyForwardRows と同一 op 列の M-row fused forward。全層を単一 CB に encode し、
    /// residual stream h は GPU 常駐 buffer。cache(KV/conv/rec)も常駐で複数 step チェーン可。
    public final class RawFusedForward {
        struct Layer {
            let isLinear: Bool
            let inputLN: MTLBuffer, postLN: MTLBuffer
            let gdn: GdnLayerBufs?, attn: AttnLayerBufs?
            let moe: MoEBlockBufs
            let gdnCache: GdnCacheBufs?, kvCache: KVCacheBufs?
            let E: Int, I: Int, Ktop: Int
        }
        var layers: [Layer] = []
        let maxM: Int, H: Int
        // attn/gdn 次元(実モデル定数)
        let numHeads: Int, numKV: Int, headDim: Int, ropeDim: Int, ropeBase: Float
        let numKHeads: Int, numVHeads: Int, headKDim: Int, headVDim: Int, convKernel: Int
        let eps: Float
        // 共有 scratch(層間で再利用、maxM でサイズ確保)
        let hBuf: MTLBuffer, normed: MTLBuffer, mixerOut: MTLBuffer, postNorm: MTLBuffer, moeOut: MTLBuffer
        let attnSc: AttnScratch, gdnSc: GdnScratch, moeSc: MoEScratch
        let device: MTLDevice, queue: MTLCommandQueue

        public init?(layers specs: [RawVerifyForward.LayerSpec], caches: [RawVerifyForward.LayerCaches],
                     maxM: Int, H: Int, maxSeqLen: Int,
                     numHeads: Int = 16, numKV: Int = 2, headDim: Int = 256,
                     ropeDim: Int = 64, ropeBase: Float = 1e7,
                     numKHeads: Int = 16, numVHeads: Int = 32, headKDim: Int = 128, headVDim: Int = 128,
                     convKernel: Int = 4, eps: Float = 1e-6) {
            guard let (device, queue) = RawMetalForward.ensure() else { return nil }
            self.device = device; self.queue = queue
            self.maxM = maxM; self.H = H
            self.numHeads = numHeads; self.numKV = numKV; self.headDim = headDim
            self.ropeDim = ropeDim; self.ropeBase = ropeBase
            self.numKHeads = numKHeads; self.numVHeads = numVHeads
            self.headKDim = headKDim; self.headVDim = headVDim; self.convKernel = convKernel
            self.eps = eps
            let keyDim = headKDim * numKHeads
            let valueDim = headVDim * numVHeads
            let convDim = keyDim * 2 + valueDim
            // pipeline warm(全種)
            let maxE = specs.map { $0.moeE }.max() ?? 256
            let maxKtop = specs.map { $0.moeKtop }.max() ?? 8
            guard RawFusedVerify.ensureMoEPipelines(E: maxE, Ktop: maxKtop),
                  RawFusedVerify.ensureAttnPipelines(),
                  RawFusedVerify.ensureGdnPipelines(Hk: numKHeads, Dk: headKDim, Hv: numVHeads, Dv: headVDim)
            else { return nil }
            // 共有 scratch
            let maxI = specs.map { $0.moeI }.max() ?? 512
            let anyPromote = specs.contains { ($0.gdn?.normWeight.dtype ?? .float16) == .float32 }
            guard let hB = device.makeBuffer(length: maxM * H * 2, options: .storageModeShared),
                  let nB = device.makeBuffer(length: maxM * H * 2, options: .storageModeShared),
                  let mB = device.makeBuffer(length: maxM * H * 2, options: .storageModeShared),
                  let pB = device.makeBuffer(length: maxM * H * 2, options: .storageModeShared),
                  let oB = device.makeBuffer(length: maxM * H * 2, options: .storageModeShared),
                  let aSc = RawFusedVerify.makeAttnScratch(device, M: maxM, numHeads: numHeads, numKV: numKV, headDim: headDim),
                  let gSc = RawFusedVerify.makeGdnScratch(device, M: maxM, C: convDim, keyDim: keyDim,
                                                          valueDim: valueDim, Hv: numVHeads, promote: anyPromote),
                  let mSc = RawFusedVerify.makeMoEScratch(device, M: maxM, E: maxE, I: maxI, Ktop: maxKtop, H: H)
            else { return nil }
            hBuf = hB; normed = nB; mixerOut = mB; postNorm = pB; moeOut = oB
            attnSc = aSc; gdnSc = gSc; moeSc = mSc
            // 層別 weight/cache buffer 化
            for (i, s) in specs.enumerated() {
                guard let ln = RawMetalForward.mtlBuf(s.inputLN.asType(.float16), device),
                      let pn = RawMetalForward.mtlBuf(s.postLN.asType(.float16), device),
                      let moe = RawFusedVerify.prepareMoEBlockBufs(s.moe, device) else { return nil }
                var gdnB: GdnLayerBufs? = nil, attnB: AttnLayerBufs? = nil
                var gdnC: GdnCacheBufs? = nil, kvC: KVCacheBufs? = nil
                if s.isLinear, let gw = s.gdn {
                    guard let gb = RawFusedVerify.prepareGdnLayerBufs(gw, Dk: headKDim, device),
                          let gc = RawFusedVerify.makeGdnCacheBufs(device, convInit: caches[i].convState,
                                                                   recInit: caches[i].recState,
                                                                   K: convKernel, C: convDim,
                                                                   Hv: numVHeads, Dv: headVDim, Dk: headKDim)
                    else { return nil }
                    gdnB = gb; gdnC = gc
                } else if let aw = s.attn {
                    guard let ab = RawFusedVerify.prepareAttnLayerBufs(aw, device),
                          let kc = RawFusedVerify.makeKVCacheBufs(device, kInit: caches[i].kCache,
                                                                  vInit: caches[i].vCache,
                                                                  maxLen: maxSeqLen, KV: numKV, D: headDim)
                    else { return nil }
                    attnB = ab; kvC = kc
                } else { return nil }
                layers.append(Layer(isLinear: s.isLinear, inputLN: ln, postLN: pn,
                                    gdn: gdnB, attn: attnB, moe: moe, gdnCache: gdnC, kvCache: kvC,
                                    E: s.moeE, I: s.moeI, Ktop: s.moeKtop))
            }
        }

        /// 1 層を encoder に encode(norm→mixer→resid→postNorm→MoE→resid)。
        func encodeLayer(_ enc: MTLComputeCommandEncoder, _ L: Layer, M: Int) {
            RawFusedVerify.encodeRmsNormRows(enc, x: hBuf, w: L.inputLN, out: normed, rows: M, D: H, eps: eps)
            if L.isLinear, let gw = L.gdn, let gc = L.gdnCache {
                RawFusedVerify.encodeGdnLayerRows(enc, x: normed, out: mixerOut, w: gw, sc: gdnSc, cache: gc,
                                                  M: M, H: H, numKHeads: numKHeads, numVHeads: numVHeads,
                                                  headKDim: headKDim, headVDim: headVDim,
                                                  convKernel: convKernel, eps: eps)
                gc.swapState()
            } else if let aw = L.attn, let kv = L.kvCache {
                RawFusedVerify.encodeAttnLayerRows(enc, x: normed, out: mixerOut, w: aw, sc: attnSc, kv: kv,
                                                   M: M, H: H, numHeads: numHeads, numKV: numKV, headDim: headDim,
                                                   ropeDim: ropeDim, ropeBase: ropeBase, eps: eps)
                kv.len += M
            }
            RawFusedVerify.encodeResidAdd(enc, h: hBuf, r: mixerOut, total: M * H)
            RawFusedVerify.encodeRmsNormRows(enc, x: hBuf, w: L.postLN, out: postNorm, rows: M, D: H, eps: eps)
            RawFusedVerify.encodeMoEBlockRows(enc, x: postNorm, out: moeOut, w: L.moe, sc: moeSc,
                                              M: M, E: L.E, I: L.I, Ktop: L.Ktop, H: H)
            RawFusedVerify.encodeResidAdd(enc, h: hBuf, r: moeOut, total: M * H)
        }

        /// 全層 forward(単一 CB)。x[M,H] → h[M,H]。cache は常駐更新(次 call にチェーン)。
        public func forwardRows(_ x: MLXArray, M: Int) -> MLXArray? {
            guard M <= maxM else { return nil }
            let xf = x.asType(.float16).reshaped([-1]); xf.eval()
            let arr = xf.asArray(Float16.self)
            hBuf.contents().bindMemory(to: Float16.self, capacity: maxM * H).update(from: arr, count: M * H)
            let cb = queue.makeCommandBuffer()!
            let enc = cb.makeComputeCommandEncoder()!
            for L in layers { encodeLayer(enc, L, M: M) }
            enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
            let ptr = hBuf.contents().bindMemory(to: Float16.self, capacity: maxM * H)
            return MLXArray(Array(UnsafeBufferPointer(start: ptr, count: M * H)), [M, H])
        }

        /// テスト比較用: 層 i の cache を MLX で読む(gdn: (conv, rec) / attn: (k, v))。
        public func readLayerCache(_ i: Int) -> (MLXArray?, MLXArray?) {
            let L = layers[i]
            if let gc = L.gdnCache { let (c, r) = RawFusedVerify.readGdnCache(gc); return (c, r) }
            if let kv = L.kvCache { let (k, v) = RawFusedVerify.readKVCache(kv); return (k, v) }
            return (nil, nil)
        }
    }
}
