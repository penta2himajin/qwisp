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
    /// xByteOffset/indsOffset/outByteOffset: 行範囲 gather(streaming chunk)でバイトオフセット指定可。
    static func encodeGatherQmmRows(_ enc: MTLComputeCommandEncoder,
                                    w: MTLBuffer, scales: MTLBuffer, biases: MTLBuffer,
                                    x: MTLBuffer, inds: MTLBuffer, out: MTLBuffer,
                                    M: Int, Ktop: Int, K: Int, N: Int, lhsPer: Bool,
                                    xByteOffset: Int = 0, indsOffset: Int = 0, outByteOffset: Int = 0) {
        enc.setComputePipelineState(RawMetalForward._gqmmRowsPipeline!)
        enc.setBuffer(w, offset: 0, index: 0); enc.setBuffer(scales, offset: 0, index: 1)
        enc.setBuffer(biases, offset: 0, index: 2)
        enc.setBuffer(x, offset: xByteOffset, index: 3)
        enc.setBuffer(inds, offset: indsOffset, index: 4)
        enc.setBuffer(out, offset: outByteOffset, index: 5)
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
    nonisolated(unsafe) static var _embedRowsPipeline: MTLComputePipelineState?
    nonisolated(unsafe) static var _argmaxRowsPipeline: MTLComputePipelineState?
    nonisolated(unsafe) static var _hnormRowsPipeline: MTLComputePipelineState?
    nonisolated(unsafe) static var _argmaxCertRowsPipeline: MTLComputePipelineState?
    nonisolated(unsafe) static var _argmaxRowsFlaggedPipeline: MTLComputePipelineState?
    // qmm4_rows_flagged = qmm4(qmv) の忠実コピー + per-row certFlag early-out。既存 _qmmPipeline とは
    // 別 pipeline(QWISP_QMM_MATH 一致の compile opts で別関数として compile)。uncert 行の logits4 は
    // _qmmPipeline と bit-identical(同一 source・同一 compile opts・同一 dispatch 形状)。
    nonisolated(unsafe) static var _qmm4RowsFlaggedPipeline: MTLComputePipelineState?

    /// M-row elementwise 補助 kernel(combine/final)。composed の MLX glue と同一の演算列を
    /// per-element/per-token 独立で再現(f16 逐次和・stable sigmoid)→ M 非依存。
    static func ensureRowsAuxPipelines() -> Bool {
        guard let (device, _) = RawMetalForward.ensure() else { return false }
        if _combineRowsPipeline != nil && _finalCombineRowsPipeline != nil && _writeKVRowsPipeline != nil
            && _convHistRowsPipeline != nil && _shiftConvRowsPipeline != nil
            && _sliceRowsPipeline != nil && _computeGBetaRowsPipeline != nil
            && _embedRowsPipeline != nil && _argmaxRowsPipeline != nil
            && _hnormRowsPipeline != nil && _argmaxCertRowsPipeline != nil
            && _argmaxRowsFlaggedPipeline != nil { return true }
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
        // shift_conv_rows: histOut ← concat(histIn,qkv)[M .. M+K-2](composed の convState 更新と同値)。
        // ping-pong(histIn 不変)= 1-step rollback を swap 戻しで実現(spec の partial reject 用)。
        kernel void shift_conv_rows(device half* histOut [[buffer(0)]], device const half* histIn [[buffer(1)]],
                                    device const half* qkv [[buffer(2)]],
                                    constant uint& K [[buffer(3)]], constant uint& C [[buffer(4)]],
                                    constant uint& M [[buffer(5)]],
                                    uint c [[thread_position_in_grid]]) {
            if (c >= C) return;
            for (uint j = 0; j + 1 < K; ++j) {
                uint src = M + j;
                histOut[j*C + c] = (src < K - 1) ? histIn[src*C + c] : qkv[(src - (K-1))*C + c];
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
        // embed_rows_q4: token id → 4bit affine dequant 行(half 積和, per-element 独立=M 不変)。
        kernel void embed_rows_q4(device const uint32_t* w [[buffer(0)]],   // [V, H/8]
                                  device const half* scales [[buffer(1)]],  // [V, H/64]
                                  device const half* biases [[buffer(2)]],
                                  device const int* tokens [[buffer(3)]],   // [M]
                                  device half* x [[buffer(4)]],             // [M, H]
                                  constant uint& H [[buffer(5)]], constant uint& total [[buffer(6)]],
                                  uint i [[thread_position_in_grid]]) {
            if (i >= total) return;
            uint m = i / H, h = i % H;
            uint row = (uint)tokens[m];
            uint pack = w[row * (H/8) + h/8];
            half nib = (half)((pack >> (4*(h%8))) & 0xf);
            uint g = h / 64;
            x[i] = scales[row*(H/64)+g] * nib + biases[row*(H/64)+g];
        }
        // argmax_rows: 行毎 argmax(先頭一致 tie-break = MLX argMax と同一)。1 threadgroup/行。
        kernel void argmax_rows(device const half* logits [[buffer(0)]],   // [M, V]
                                device int* outIdx [[buffer(1)]],          // [M]
                                constant uint& V [[buffer(2)]],
                                uint m [[threadgroup_position_in_grid]],
                                uint tid [[thread_position_in_threadgroup]],
                                uint tgs [[threads_per_threadgroup]]) {
            threadgroup float red[256]; threadgroup int redi[256];
            device const half* row = logits + (size_t)m * V;
            float best = -INFINITY; int bi = 0x7fffffff;
            for (uint v = tid; v < V; v += tgs) {
                float lv = (float)row[v];
                if (lv > best) { best = lv; bi = (int)v; }   // thread 内は v 昇順 → 先頭一致
            }
            red[tid] = best; redi[tid] = bi;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint s = tgs/2; s > 0; s >>= 1) {
                if (tid < s) {
                    if (red[tid+s] > red[tid] || (red[tid+s] == red[tid] && redi[tid+s] < redi[tid])) {
                        red[tid] = red[tid+s]; redi[tid] = redi[tid+s];
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (tid == 0) outIdx[m] = redi[0];
        }
        // hnorm_rows: 行毎 L2 norm。x[M,K] f16 -> out[M] f32(1 threadgroup/行, 256 threads, f32 tree reduction)。
        kernel void hnorm_rows(device const half* x  [[buffer(0)]],   // [M, K]
                               device float*       out [[buffer(1)]],  // [M]
                               constant uint& K [[buffer(2)]],
                               uint m   [[threadgroup_position_in_grid]],
                               uint tid [[thread_position_in_threadgroup]],
                               uint tgs [[threads_per_threadgroup]]) {
            threadgroup float red[256];
            device const half* row = x + (size_t)m * K;
            float acc = 0.0f;
            for (uint k = tid; k < K; k += tgs) { float xv = (float)row[k]; acc += xv * xv; }
            red[tid] = acc; threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint s = tgs/2; s > 0; s >>= 1) {
                if (tid < s) red[tid] += red[tid + s];
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (tid == 0) out[m] = precise::sqrt(red[0]);
        }
        // argmax_cert_rows: 行 m ごとに(1 threadgroup/行)1 pass で (a)logits2 の top-1(=v*) と
        // (b)logits2+rowE·hnorm+ε の top-2(value+idx, first-index tie-break)を求め、
        // challenger = (bTop1.idx != v*) ? bTop1.val : bTop2.val とし、
        // a[v*] - (rowE[v*]·hnorm + ε) > challenger なら tokensOut[m]=v*, certFlag[m]=1, atomic++certCount
        // (§3.2 cert 条件、strict inequality)。さもなくば certFlag[m]=0(tokensOut は後段 argmax_rows_flagged が書く)。
        // 上記 upd_top2 は first-index tie-break を保存する top-2 更新(value+idx)。
        inline void upd_top2(thread float& m1, thread int& i1, thread float& m2, thread int& i2,
                             float val, int idx) {
            if (val > m1 || (val == m1 && idx < i1)) { m2 = m1; i2 = i1; m1 = val; i1 = idx; }
            else if (val > m2 || (val == m2 && idx < i2)) { m2 = val; i2 = idx; }
        }
        kernel void argmax_cert_rows(device const half*   logits2 [[buffer(0)]],   // [M, V] f16(qmm2 出力)
                                     device const half*   rowE    [[buffer(1)]],    // [V]  f16
                                     device const float*  hnorm   [[buffer(2)]],    // [M]  f32
                                     device int*          tokensOut [[buffer(3)]],  // [M]  int32
                                     device int*          certFlag  [[buffer(4)]],  // [M]  int32 (0/1)
                                     device atomic_int*   certCount [[buffer(5)]],   // [1]  atomic
                                     constant uint&  V   [[buffer(6)]],
                                     constant float& EPS [[buffer(7)]],
                                     uint m   [[threadgroup_position_in_grid]],
                                     uint tid [[thread_position_in_threadgroup]],
                                     uint tgs [[threads_per_threadgroup]]) {
            threadgroup float redA[256];  threadgroup int rediA[256];   // top-1 of a (logits2)
            threadgroup float redB1[256]; threadgroup int redB1i[256];  // top-1 of b
            threadgroup float redB2[256]; threadgroup int redB2i[256];  // top-2 of b
            device const half* lrow = logits2 + (size_t)m * V;
            float hn = hnorm[m];
            float aBest = -INFINITY; int aIdx = 0;
            float b1 = -INFINITY, b2 = -INFINITY; int bi1 = 0, bi2 = 0;
            for (uint v = tid; v < V; v += tgs) {
                float lv = (float)lrow[v];
                if (lv > aBest) { aBest = lv; aIdx = (int)v; }   // thread 内は v 昇順 → 先頭一致(argmax_rows と同一)
                float bv = lv + (float)rowE[v] * hn + EPS;
                upd_top2(b1, bi1, b2, bi2, bv, (int)v);
            }
            redA[tid] = aBest; rediA[tid] = aIdx;
            redB1[tid] = b1; redB1i[tid] = bi1; redB2[tid] = b2; redB2i[tid] = bi2;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint s = tgs/2; s > 0; s >>= 1) {
                if (tid < s) {
                    if (redA[tid+s] > redA[tid] || (redA[tid+s] == redA[tid] && rediA[tid+s] < rediA[tid])) {
                        redA[tid] = redA[tid+s]; rediA[tid] = rediA[tid+s];
                    }
                    float m1 = redB1[tid], m2 = redB2[tid]; int i1 = redB1i[tid], i2 = redB2i[tid];
                    upd_top2(m1, i1, m2, i2, redB1[tid+s], redB1i[tid+s]);
                    upd_top2(m1, i1, m2, i2, redB2[tid+s], redB2i[tid+s]);
                    redB1[tid] = m1; redB1i[tid] = i1; redB2[tid] = m2; redB2i[tid] = i2;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (tid == 0) {
                int vstar = rediA[0];
                float aStar = redA[0];
                float chall = (redB1i[0] != vstar) ? redB1[0] : redB2[0];
                float lower = aStar - ((float)rowE[vstar] * hn + EPS);
                if (lower > chall) {
                    tokensOut[m] = vstar;
                    certFlag[m] = 1;
                    atomic_fetch_add_explicit(certCount, 1, memory_order_relaxed);
                } else {
                    certFlag[m] = 0;
                }
            }
        }
        // argmax_rows_flagged: 既存 argmax_rows と同一 reduction だが threadgroup 冒頭で
        // certFlag[m]==1 なら即 return(cert 行は tokensOut[m] を上書きしない=argmax_cert_rows が設定済み)。
        // uncert 行は hd.logits(=logits4, qmm4_rows_flagged 出力)の argmax を書く=既存 4-bit 経路そのもの。
        kernel void argmax_rows_flagged(device const half* logits [[buffer(0)]],   // [M, V]
                                        device int* outIdx [[buffer(1)]],          // [M]
                                        device const int* certFlag [[buffer(2)]],  // [M]
                                        constant uint& V [[buffer(3)]],
                                        uint m [[threadgroup_position_in_grid]],
                                        uint tid [[thread_position_in_threadgroup]],
                                        uint tgs [[threads_per_threadgroup]]) {
            if (certFlag[m] != 0) return;
            threadgroup float red[256]; threadgroup int redi[256];
            device const half* row = logits + (size_t)m * V;
            float best = -INFINITY; int bi = 0x7fffffff;
            for (uint v = tid; v < V; v += tgs) {
                float lv = (float)row[v];
                if (lv > best) { best = lv; bi = (int)v; }   // thread 内は v 昇順 → 先頭一致
            }
            red[tid] = best; redi[tid] = bi;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint s = tgs/2; s > 0; s >>= 1) {
                if (tid < s) {
                    if (red[tid+s] > red[tid] || (red[tid+s] == red[tid] && redi[tid+s] < redi[tid])) {
                        red[tid] = red[tid+s]; redi[tid] = redi[tid+s];
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (tid == 0) outIdx[m] = redi[0];
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
            _embedRowsPipeline = try device.makeComputePipelineState(function: lib.makeFunction(name: "embed_rows_q4")!)
            _argmaxRowsPipeline = try device.makeComputePipelineState(function: lib.makeFunction(name: "argmax_rows")!)
            _hnormRowsPipeline = try device.makeComputePipelineState(function: lib.makeFunction(name: "hnorm_rows")!)
            _argmaxCertRowsPipeline = try device.makeComputePipelineState(function: lib.makeFunction(name: "argmax_cert_rows")!)
            _argmaxRowsFlaggedPipeline = try device.makeComputePipelineState(function: lib.makeFunction(name: "argmax_rows_flagged")!)
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
    /// byteOffset: streaming chunk で g/u/h の共通バイトオフセット(3 バッファとも同一)。
    static func encodeSwiglu(_ enc: MTLComputeCommandEncoder, g: MTLBuffer, u: MTLBuffer, h: MTLBuffer,
                              total: Int, byteOffset: Int = 0) {
        let p = RawMetalForward._swigluPipeline!
        enc.setComputePipelineState(p)
        enc.setBuffer(g, offset: byteOffset, index: 0)
        enc.setBuffer(u, offset: byteOffset, index: 1)
        enc.setBuffer(h, offset: byteOffset, index: 2)
        var t = UInt32(total); enc.setBytes(&t, length: 4, index: 3)
        enc.dispatchThreads(MTLSize(width: total, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: min(p.maxTotalThreadsPerThreadgroup, 256), height: 1, depth: 1))
    }

    /// dByteOffset/scoresOffset/yByteOffset: streaming chunk で行範囲バイトオフセット指定可。
    static func encodeCombineRows(_ enc: MTLComputeCommandEncoder, d: MTLBuffer, scores: MTLBuffer, y: MTLBuffer,
                                  Ktop: Int, N: Int, M: Int,
                                  dByteOffset: Int = 0, scoresOffset: Int = 0, yByteOffset: Int = 0) {
        let p = _combineRowsPipeline!
        enc.setComputePipelineState(p)
        enc.setBuffer(d, offset: dByteOffset, index: 0)
        enc.setBuffer(scores, offset: scoresOffset, index: 1)
        enc.setBuffer(y, offset: yByteOffset, index: 2)
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

    /// per-op wrapper(テスト用): embed_rows_q4 単発実行。tokens[M] → x[M,H] f16。
    public static func embedRowsRaw(_ tokens: [Int32], w: MLXArray, scales: MLXArray, biases: MLXArray,
                                    H: Int) -> MLXArray? {
        guard let (device, queue) = RawMetalForward.ensure(), ensureRowsAuxPipelines() else { return nil }
        let M = tokens.count
        let sc = scales.asType(.float16), bi = biases.asType(.float16)
        guard let bw = RawMetalForward.mtlBuf(w, device),
              let bs = RawMetalForward.mtlBuf(sc, device),
              let bb = RawMetalForward.mtlBuf(bi, device),
              let bt = device.makeBuffer(length: M * 4, options: .storageModeShared),
              let outBuf = device.makeBuffer(length: M * H * 2, options: .storageModeShared) else { return nil }
        bt.contents().bindMemory(to: Int32.self, capacity: M).update(from: tokens, count: M)
        let p = _embedRowsPipeline!
        let cb = queue.makeCommandBuffer()!; let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(p)
        enc.setBuffer(bw, offset: 0, index: 0); enc.setBuffer(bs, offset: 0, index: 1)
        enc.setBuffer(bb, offset: 0, index: 2); enc.setBuffer(bt, offset: 0, index: 3)
        enc.setBuffer(outBuf, offset: 0, index: 4)
        var hh = UInt32(H), tt = UInt32(M * H)
        enc.setBytes(&hh, length: 4, index: 5); enc.setBytes(&tt, length: 4, index: 6)
        enc.dispatchThreads(MTLSize(width: M * H, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: min(p.maxTotalThreadsPerThreadgroup, 256), height: 1, depth: 1))
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        let ptr = outBuf.contents().bindMemory(to: Float16.self, capacity: M * H)
        return MLXArray(Array(UnsafeBufferPointer(start: ptr, count: M * H)), [M, H])
    }

    /// per-op wrapper(テスト用): argmax_rows 単発実行。logits[M,V] f16 → [M] int(先頭一致 tie)。
    public static func argmaxRowsRaw(_ logits: MLXArray, M: Int, V: Int) -> [Int]? {
        guard let (device, queue) = RawMetalForward.ensure(), ensureRowsAuxPipelines() else { return nil }
        guard let bl = RawMetalForward.mtlBuf(logits.asType(.float16), device),
              let outBuf = device.makeBuffer(length: M * 4, options: .storageModeShared) else { return nil }
        let p = _argmaxRowsPipeline!
        let cb = queue.makeCommandBuffer()!; let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(p)
        enc.setBuffer(bl, offset: 0, index: 0); enc.setBuffer(outBuf, offset: 0, index: 1)
        var vv = UInt32(V); enc.setBytes(&vv, length: 4, index: 2)
        enc.dispatchThreadgroups(MTLSize(width: M, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        let ptr = outBuf.contents().bindMemory(to: Int32.self, capacity: M)
        return (0 ..< M).map { Int(ptr[$0]) }
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
        // ★ zero-copy(bytesNoCopy)buffer の裏 MLXArray を保持。asType 変換が生む一時 array の
        //   buffer が解放→MLX allocator 再利用で clobber されるのを防ぐ(mlx-swift asMTLBuffer の寿命規約)。
        let retained: [MLXArray]
    }

    /// expertOverride 非 nil 時は sw*(routed expert)フィールドをそこから取り、
    /// w.swGWq 等は一切 materialize しない(streaming 時は mmap-lazy 重みをメモリに展開しないため)。
    /// gate/shared/sharedGate は常に w から変換。
    static func prepareMoEBlockBufs(_ w: RawVerifyForward.MoEBlockW, _ device: MTLDevice,
                                     expertOverride: [MTLBuffer]? = nil) -> MoEBlockBufs? {
        var keep: [MLXArray] = []
        func trio(_ q: MLXArray, _ s: MLXArray, _ b: MLXArray) -> (MTLBuffer, MTLBuffer, MTLBuffer)? {
            let sc = s.asType(.float16), bc = b.asType(.float16)
            keep.append(contentsOf: [q, sc, bc])
            guard let bq = RawMetalForward.mtlBuf(q, device),
                  let bs = RawMetalForward.mtlBuf(sc, device),
                  let bb = RawMetalForward.mtlBuf(bc, device) else { return nil }
            return (bq, bs, bb)
        }
        guard let g = trio(w.gateWq, w.gateSc, w.gateBi) else { return nil }

        // sw*(routed expert): expertOverride あれば arena buffer 直接使用。
        let swGW: MTLBuffer, swGS: MTLBuffer, swGB: MTLBuffer
        let swUW: MTLBuffer, swUS: MTLBuffer, swUB: MTLBuffer
        let swDW: MTLBuffer, swDS: MTLBuffer, swDB: MTLBuffer
        if let ov = expertOverride, ov.count >= 9 {
            (swGW, swGS, swGB) = (ov[0], ov[1], ov[2])
            (swUW, swUS, swUB) = (ov[3], ov[4], ov[5])
            (swDW, swDS, swDB) = (ov[6], ov[7], ov[8])
        } else {
            guard let swG = trio(w.swGWq, w.swGSc, w.swGBi),
                  let swU = trio(w.swUWq, w.swUSc, w.swUBi),
                  let swD = trio(w.swDWq, w.swDSc, w.swDBi) else { return nil }
            (swGW, swGS, swGB) = (swG.0, swG.1, swG.2)
            (swUW, swUS, swUB) = (swU.0, swU.1, swU.2)
            (swDW, swDS, swDB) = (swD.0, swD.1, swD.2)
        }

        guard let shG = trio(w.shGWq, w.shGSc, w.shGBi),
              let shU = trio(w.shUWq, w.shUSc, w.shUBi),
              let shD = trio(w.shDWq, w.shDSc, w.shDBi),
              let sg = trio(w.sharedGateWq, w.sharedGateSc, w.sharedGateBi) else { return nil }
        return MoEBlockBufs(gW: g.0, gS: g.1, gB: g.2,
                            swGW: swGW, swGS: swGS, swGB: swGB,
                            swUW: swUW, swUS: swUS, swUB: swUB,
                            swDW: swDW, swDS: swDS, swDB: swDB,
                            shGW: shG.0, shGS: shG.1, shGB: shG.2,
                            shUW: shU.0, shUS: shU.1, shUB: shU.2,
                            shDW: shD.0, shDS: shD.1, shDB: shD.2,
                            sgW: sg.0, sgS: sg.1, sgB: sg.2,
                            retained: keep)
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

    // ── MoE block の 3 フェーズ分割 ──────────────────────────────────────────────────────────────
    // strict streaming では ①(route) で CB を切り、inds 読み出し後に ②(gather) を chunk 毎に
    // encode し、最後に ③④(shared) をまとめて encode する。resident/bolt は 3 つを連続 encode。

    /// ① routing: gate qmm8 → route_top8_rows(inds+renorm scores)。
    static func encodeMoERouteRows(_ enc: MTLComputeCommandEncoder, x: MTLBuffer,
                                    w: MoEBlockBufs, sc: MoEScratch,
                                    M: Int, E: Int, H: Int, Ktop: Int) {
        encodeQmm8Rows(enc, w: w.gW, scales: w.gS, biases: w.gB, x: x, out: sc.gl, M: M, K: H, N: E)
        encodeRouteTop8Rows(enc, logits: sc.gl, inds: sc.inds, scores: sc.scores, M: M, N: E, K: Ktop)
    }

    /// ② routed experts gather フェーズ (行 [r0, r1) のみ)。
    /// slotTable 非 nil の場合は gather 前に inds[r0*Ktop ..< r1*Ktop] を GPU remap。
    /// w.sw* は arena buffer(streaming 時)または resident buffer(bolt/resident 時)。
    static func encodeMoEGatherRowsRange(_ enc: MTLComputeCommandEncoder,
                                          x: MTLBuffer, w: MoEBlockBufs, sc: MoEScratch,
                                          r0: Int, r1: Int, Ktop: Int, I: Int, H: Int,
                                          slotTable: MTLBuffer?) {
        let Mc = r1 - r0
        let xOff    = r0 * H * 2
        let indsOff = r0 * Ktop * 4
        let guOff   = r0 * Ktop * I * 2
        let dOff    = r0 * Ktop * H * 2
        let scOff   = r0 * Ktop * 2
        let yOff    = r0 * H * 2

        if let st = slotTable {
            RawMetalForward.encodeSlotRemapRows(enc, inds: sc.inds, indsByteOffset: indsOff,
                                                table: st, count: Mc * Ktop)
        }
        // gather g/u(行共有 lhs)
        encodeGatherQmmRows(enc, w: w.swGW, scales: w.swGS, biases: w.swGB,
                            x: x, inds: sc.inds, out: sc.g,
                            M: Mc, Ktop: Ktop, K: H, N: I, lhsPer: false,
                            xByteOffset: xOff, indsOffset: indsOff, outByteOffset: guOff)
        encodeGatherQmmRows(enc, w: w.swUW, scales: w.swUS, biases: w.swUB,
                            x: x, inds: sc.inds, out: sc.u,
                            M: Mc, Ktop: Ktop, K: H, N: I, lhsPer: false,
                            xByteOffset: xOff, indsOffset: indsOff, outByteOffset: guOff)
        // swiglu
        encodeSwiglu(enc, g: sc.g, u: sc.u, h: sc.h, total: Mc * Ktop * I, byteOffset: guOff)
        // gather d(per-mk lhs, x=sc.h)
        encodeGatherQmmRows(enc, w: w.swDW, scales: w.swDS, biases: w.swDB,
                            x: sc.h, inds: sc.inds, out: sc.d,
                            M: Mc, Ktop: Ktop, K: I, N: H, lhsPer: true,
                            xByteOffset: guOff, indsOffset: indsOff, outByteOffset: dOff)
        // combine → sc.y[r0*H .. r1*H]
        encodeCombineRows(enc, d: sc.d, scores: sc.scores, y: sc.y,
                          Ktop: Ktop, N: H, M: Mc,
                          dByteOffset: dOff, scoresOffset: scOff, yByteOffset: yOff)
    }

    /// ③④ shared expert + final combine → out[M,H]。
    static func encodeMoESharedRows(_ enc: MTLComputeCommandEncoder, x: MTLBuffer, out: MTLBuffer,
                                     w: MoEBlockBufs, sc: MoEScratch, M: Int, I: Int, H: Int) {
        encodeQmmRows(enc, w: w.shGW, scales: w.shGS, biases: w.shGB, x: x, out: sc.sg, M: M, K: H, N: I)
        encodeQmmRows(enc, w: w.shUW, scales: w.shUS, biases: w.shUB, x: x, out: sc.su, M: M, K: H, N: I)
        encodeSwiglu(enc, g: sc.sg, u: sc.su, h: sc.shAct, total: M * I)
        encodeQmmRows(enc, w: w.shDW, scales: w.shDS, biases: w.shDB, x: sc.shAct, out: sc.sharedY, M: M, K: I, N: H)
        encodeQmm8Rows(enc, w: w.sgW, scales: w.sgS, biases: w.sgB, x: x, out: sc.sgl, M: M, K: H, N: 8)
        encodeFinalCombineRows(enc, y: sc.y, sharedY: sc.sharedY, sgl: sc.sgl, out: out, N: H, M: M)
    }

    /// MoE block 全段(3 フェーズ連続 encode)。resident 経路(slotTable:nil)と bolt 経路(slotTable:非nil)を兼ねる。
    /// resident 時: slotTable=nil → remap なし、sw* は常駐 buffer。
    /// bolt 時: slotTable=frozen table → GPU remap、sw* は arena buffer(expertOverride 済み)。
    static func encodeMoEBlockRows(_ enc: MTLComputeCommandEncoder, x: MTLBuffer, out: MTLBuffer,
                                   w: MoEBlockBufs, sc: MoEScratch,
                                   M: Int, E: Int, I: Int, Ktop: Int, H: Int,
                                   slotTable: MTLBuffer? = nil) {
        encodeMoERouteRows(enc, x: x, w: w, sc: sc, M: M, E: E, H: H, Ktop: Ktop)
        encodeMoEGatherRowsRange(enc, x: x, w: w, sc: sc, r0: 0, r1: M, Ktop: Ktop, I: I, H: H, slotTable: slotTable)
        encodeMoESharedRows(enc, x: x, out: out, w: w, sc: sc, M: M, I: I, H: H)
    }

    /// 全 pipeline を warm(compile)。fused 経路の前提(encode 時に force-unwrap するため)。
    static func ensureMoEPipelines(E: Int = 256, Ktop: Int = 8) -> Bool {
        ensureQmmPipeline()
        guard RawMetalForward.compileQmm8(), RawMetalForward.ensureAuxPipelines(), ensureRowsAuxPipelines()
        else { return false }
        // slot_remap_rows(streaming chunk remap 用 grid kernel)も同時に warm
        _ = RawMetalForward.compileSlotRemapRows()
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
        let retained: [MLXArray]   // zero-copy buffer の裏 array 保持(寿命規約)
    }

    static func prepareAttnLayerBufs(_ w: RawVerifyForward.AttnLayerW, _ device: MTLDevice) -> AttnLayerBufs? {
        var keep: [MLXArray] = []
        func trio(_ q: MLXArray, _ s: MLXArray, _ b: MLXArray) -> (MTLBuffer, MTLBuffer, MTLBuffer)? {
            let sc = s.asType(.float16), bc = b.asType(.float16)
            keep.append(contentsOf: [q, sc, bc])
            guard let bq = RawMetalForward.mtlBuf(q, device),
                  let bs = RawMetalForward.mtlBuf(sc, device),
                  let bb = RawMetalForward.mtlBuf(bc, device) else { return nil }
            return (bq, bs, bb)
        }
        let qnA = w.qNorm.asType(.float16), knA = w.kNorm.asType(.float16)
        keep.append(contentsOf: [qnA, knA])
        guard let q = trio(w.qWq, w.qSc, w.qBi), let k = trio(w.kWq, w.kSc, w.kBi),
              let v = trio(w.vWq, w.vSc, w.vBi), let o = trio(w.oWq, w.oSc, w.oBi),
              let qn = RawMetalForward.mtlBuf(qnA, device),
              let kn = RawMetalForward.mtlBuf(knA, device) else { return nil }
        return AttnLayerBufs(qW: q.0, qS: q.1, qB: q.2, kW: k.0, kS: k.1, kB: k.2,
                             vW: v.0, vS: v.1, vB: v.2, oW: o.0, oS: o.1, oB: o.2,
                             qNorm: qn, kNorm: kn, retained: keep)
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

    static func encodeShiftConvRows(_ enc: MTLComputeCommandEncoder, histOut: MTLBuffer, histIn: MTLBuffer,
                                    qkv: MTLBuffer, K: Int, C: Int, M: Int) {
        let p = _shiftConvRowsPipeline!
        enc.setComputePipelineState(p)
        enc.setBuffer(histOut, offset: 0, index: 0); enc.setBuffer(histIn, offset: 0, index: 1)
        enc.setBuffer(qkv, offset: 0, index: 2)
        var kk = UInt32(K), cc = UInt32(C), mm = UInt32(M)
        enc.setBytes(&kk, length: 4, index: 3); enc.setBytes(&cc, length: 4, index: 4); enc.setBytes(&mm, length: 4, index: 5)
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
        let retained: [MLXArray]        // zero-copy buffer の裏 array 保持(寿命規約)
    }

    static func prepareGdnLayerBufs(_ w: RawVerifyForward.GDNLayerW, Dk: Int, _ device: MTLDevice) -> GdnLayerBufs? {
        var keep: [MLXArray] = []
        func trio(_ q: MLXArray, _ s: MLXArray, _ b: MLXArray) -> (MTLBuffer, MTLBuffer, MTLBuffer)? {
            let sc = s.asType(.float16), bc = b.asType(.float16)
            keep.append(contentsOf: [q, sc, bc])
            guard let bq = RawMetalForward.mtlBuf(q, device),
                  let bs = RawMetalForward.mtlBuf(sc, device),
                  let bb = RawMetalForward.mtlBuf(bc, device) else { return nil }
            return (bq, bs, bb)
        }
        let promote = (w.normWeight.dtype == .float32)
        let ones = MLXArray.ones([Dk]).asType(.float16); ones.eval()
        let cwA = w.conv1dW.asType(.float32)
        let nwA = w.normWeight.asType(promote ? .float32 : .float16)
        let alA = w.aLog.asType(.float32), dtA = w.dtBias.asType(.float32)
        keep.append(contentsOf: [ones, cwA, nwA, alA, dtA])
        guard let qkv = trio(w.qkvWq, w.qkvSc, w.qkvBi), let z = trio(w.zWq, w.zSc, w.zBi),
              let b = trio(w.bWq, w.bSc, w.bBi), let a = trio(w.aWq, w.aSc, w.aBi),
              let o = trio(w.outWq, w.outSc, w.outBi),
              let cw = RawMetalForward.mtlBuf(cwA, device),
              let nw = RawMetalForward.mtlBuf(nwA, device),
              let al = RawMetalForward.mtlBuf(alA, device),
              let dt = RawMetalForward.mtlBuf(dtA, device),
              let od = RawMetalForward.mtlBuf(ones, device) else { return nil }
        return GdnLayerBufs(qkvW: qkv.0, qkvS: qkv.1, qkvB: qkv.2, zW: z.0, zS: z.1, zB: z.2,
                            bW: b.0, bS: b.1, bB: b.2, aW: a.0, aS: a.1, aB: a.2,
                            outW: o.0, outS: o.1, outB: o.2,
                            conv1dW: cw, normWeight: nw, promoteRMS: promote, aLog: al, dtBias: dt, onesDk: od,
                            retained: keep)
    }

    /// GDN 層の常駐 cache(conv hist [K-1,C] f16 と rec state [1,Hv,Dv,Dk] f32、両方 ping-pong)。
    /// ping-pong = encode は In を読み Out に書く。encode 後 swap()。直前 1 step の rollback は
    /// もう一度 swap()(裏面に pre-step 値が無傷で残る — spec の partial reject 用)。
    public final class GdnCacheBufs {
        var convHist: MTLBuffer     // 現在(encode 入力)
        var convHistOut: MTLBuffer  // encode 出力
        var state: MTLBuffer        // 現在 state(encode 入力)
        var stateOut: MTLBuffer     // encode 出力
        let K: Int, C: Int, Hv: Int, Dv: Int, Dk: Int
        init(convHist: MTLBuffer, convHistOut: MTLBuffer, state: MTLBuffer, stateOut: MTLBuffer,
             K: Int, C: Int, Hv: Int, Dv: Int, Dk: Int) {
            self.convHist = convHist; self.convHistOut = convHistOut
            self.state = state; self.stateOut = stateOut
            self.K = K; self.C = C; self.Hv = Hv; self.Dv = Dv; self.Dk = Dk
        }
        func swapState() {
            let t = state; state = stateOut; stateOut = t
            let c = convHist; convHist = convHistOut; convHistOut = c
        }
    }

    static func makeGdnCacheBufs(_ device: MTLDevice, convInit: MLXArray?, recInit: MLXArray?,
                                 K: Int, C: Int, Hv: Int, Dv: Int, Dk: Int) -> GdnCacheBufs? {
        guard let hist = device.makeBuffer(length: (K - 1) * C * 2, options: .storageModeShared),
              let histOut = device.makeBuffer(length: (K - 1) * C * 2, options: .storageModeShared),
              let st = device.makeBuffer(length: Hv * Dv * Dk * 4, options: .storageModeShared),
              let stOut = device.makeBuffer(length: Hv * Dv * Dk * 4, options: .storageModeShared) else { return nil }
        if let c0 = convInit, c0.size > 0 {
            let cf = c0.asType(.float16).reshaped([-1]); cf.eval()
            let arr = cf.asArray(Float16.self)
            hist.contents().bindMemory(to: Float16.self, capacity: (K - 1) * C)
                .update(from: arr, count: min(arr.count, (K - 1) * C))
        }
        if let r0 = recInit, r0.size > 0 {
            let rf = r0.asType(.float32).reshaped([-1]); rf.eval()
            let arr = rf.asArray(Float.self)
            st.contents().bindMemory(to: Float.self, capacity: Hv * Dv * Dk)
                .update(from: arr, count: min(arr.count, Hv * Dv * Dk))
        }
        return GdnCacheBufs(convHist: hist, convHistOut: histOut, state: st, stateOut: stOut,
                            K: K, C: C, Hv: Hv, Dv: Dv, Dk: Dk)
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
        // ② conv(hist 直読み)→ hist shift 更新(ping-pong: In を読み Out へ)
        encodeConvHistRows(enc, hist: cache.convHist, qkv: sc.qkv, w: w.conv1dW, out: sc.convOut,
                           K: convKernel, C: convDim, M: M)
        encodeShiftConvRows(enc, histOut: cache.convHistOut, histIn: cache.convHist, qkv: sc.qkv,
                            K: convKernel, C: convDim, M: M)
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
        /// zero-copy buffer の裏 MLXArray 保持(asType 変換の一時 array を allocator 再利用から守る)
        public var retainedArrays: [MLXArray] = []
        /// profile: 直近 step の全 CB GPU-exec 合計 ms。
        nonisolated(unsafe) public static var profLastGPUMs = 0.0
        /// lm_head を qmm4(qmv, 高 occupancy)にする(QWISP_LMHEAD_QMV=1)。M=1 decode 高速化。M 不変維持。
        nonisolated(unsafe) public static var lmHeadQmv =
            ProcessInfo.processInfo.environment["QWISP_LMHEAD_QMV"] != "0"   // 既定 ON(全M で速い無条件改善)
        /// lm_head を margin-certified 2-bit 先読み + 4-bit fallback に置換する(QWISP_LMHEAD2=1)。既定 off。
        /// notes/05 §3。cert 行は 2-bit qmv → argmax(4-bit と bit 一致を bound+ε で保証)、
        /// uncert 行のみ既存 4-bit qmv kernel で fallback → 全行で 4-bit 経路と bit-exact。
        nonisolated(unsafe) public static var lmHead2 =
            ProcessInfo.processInfo.environment["QWISP_LMHEAD2"] == "1"
        /// lmhead2 cert 累計(引擎 build 以後): certified は GPU atomic counter、total は CPU カウンタ。
        public var certTotalRows: Int = 0
        public var totalRows: Int = 0

        // ── streaming mode ─────────────────────────────────────────────────────────────
        public enum RawStreamMode { case resident, strict, bolt }
        public private(set) var streamMode: RawStreamMode = .resident
        /// per-layer provider(streaming 時非 nil)。layers と等長。
        var providers: [RawFusedExpertProvider]? = nil
        /// per-layer slot table buffer([E] int32, storageModeShared)。strict/bolt で ensure 結果を書き込む。
        var slotTables: [MTLBuffer] = []
        /// 直近 step の最大 chunk 数/層(strict モードのテスト検証用)。
        public private(set) var lastStepChunks: Int = 0
        /// bolt calib: strict モードで層毎の routing(expert id)を観測する(readback は既に行われる=無料)。
        /// (layerIndex, inds[M*Ktop]) — nil の間は呼び出しコスト無し。
        public var indsCaptureHook: ((Int, [Int32]) -> Void)? = nil

        public init?(layers specs: [RawVerifyForward.LayerSpec], caches: [RawVerifyForward.LayerCaches],
                     maxM: Int, H: Int, maxSeqLen: Int,
                     numHeads: Int = 16, numKV: Int = 2, headDim: Int = 256,
                     ropeDim: Int = 64, ropeBase: Float = 1e7,
                     numKHeads: Int = 16, numVHeads: Int = 32, headKDim: Int = 128, headVDim: Int = 128,
                     convKernel: Int = 4, eps: Float = 1e-6,
                     providers initProviders: [RawFusedExpertProvider]? = nil) {
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
                let lnA = s.inputLN.asType(.float16), pnA = s.postLN.asType(.float16)
                retainedArrays.append(contentsOf: [lnA, pnA])
                // streaming: sw*(routed expert)は provider の arena buffer を使う
                let ov = initProviders?[i].gatherBuffers(device: device)
                guard let ln = RawMetalForward.mtlBuf(lnA, device),
                      let pn = RawMetalForward.mtlBuf(pnA, device),
                      let moe = RawFusedVerify.prepareMoEBlockBufs(s.moe, device, expertOverride: ov) else { return nil }
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
            // streaming mode setup
            if let p = initProviders {
                guard p.count == specs.count else { return nil }
                providers = p
                streamMode = .strict
                for (i, s) in specs.enumerated() {
                    guard let st = device.makeBuffer(length: s.moeE * 4, options: .storageModeShared) else { return nil }
                    slotTables.append(st)
                    _ = i  // suppress unused warning
                }
            }
        }

        /// 1 層を encoder に encode(norm→mixer→resid→postNorm→MoE→resid)。resident/bolt 経路のみ。
        func encodeLayer(_ enc: MTLComputeCommandEncoder, _ L: Layer, M: Int) {
            encodePreMoE(enc, L, M: M)
            RawFusedVerify.encodeMoEBlockRows(enc, x: postNorm, out: moeOut, w: L.moe, sc: moeSc,
                                              M: M, E: L.E, I: L.I, Ktop: L.Ktop, H: H)
            RawFusedVerify.encodeResidAdd(enc, h: hBuf, r: moeOut, total: M * H)
        }

        /// bolt 層: resident と同一だが MoE gather 前に全 inds を GPU remap する。
        func encodeLayerBolt(_ enc: MTLComputeCommandEncoder, _ L: Layer, M: Int, li: Int) {
            encodePreMoE(enc, L, M: M)
            RawFusedVerify.encodeMoEBlockRows(enc, x: postNorm, out: moeOut, w: L.moe, sc: moeSc,
                                              M: M, E: L.E, I: L.I, Ktop: L.Ktop, H: H,
                                              slotTable: slotTables[li])
            RawFusedVerify.encodeResidAdd(enc, h: hBuf, r: moeOut, total: M * H)
        }

        /// MoE 前半 encode: norm → mixer(+cache bookkeeping) → resid → postNorm。
        func encodePreMoE(_ enc: MTLComputeCommandEncoder, _ L: Layer, M: Int) {
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
        }

        // ── streaming helpers ─────────────────────────────────────────────────────────

        struct MoEChunk { let r0, r1: Int; let experts: [Int] }

        /// inds[M*Ktop] をスキャンし、各 chunk の distinct expert union が C 以下になるように貪欲分割。
        private func partitionChunks(_ inds: [Int32], M: Int, Ktop: Int, C: Int) -> [MoEChunk] {
            var chunks: [MoEChunk] = []
            var r0 = 0
            var cur = Set<Int>()
            for m in 0..<M {
                var row = Set<Int>()
                for k in 0..<Ktop { row.insert(Int(inds[m * Ktop + k])) }
                let next = cur.union(row)
                if next.count > C && r0 < m {
                    chunks.append(MoEChunk(r0: r0, r1: m, experts: Array(cur).sorted()))
                    r0 = m; cur = row
                } else {
                    cur = next
                }
            }
            if r0 < M { chunks.append(MoEChunk(r0: r0, r1: M, experts: Array(cur).sorted())) }
            return chunks
        }

        // ── setBoltTables / setStrictStreaming ────────────────────────────────────────

        /// per-layer frozen slot table を設定し bolt モードに切り替え。tables[li][e] = slot id。
        public func setBoltTables(_ tables: [[Int32]]) {
            for (li, tbl) in tables.enumerated() {
                guard li < slotTables.count else { break }
                let p = slotTables[li].contents().bindMemory(to: Int32.self, capacity: tbl.count)
                for (e, s) in tbl.enumerated() { p[e] = s }
            }
            streamMode = .bolt
        }

        /// bolt → strict に戻す(slot table は保持)。
        public func setStrictStreaming() { streamMode = .strict }

        /// 全層 forward。x[M,H] → h[M,H]。cache は常駐更新(次 call にチェーン)。
        /// finalNormW を渡すと最終 rmsNorm も同梱し normed [M,H] を返す。
        public func forwardRows(_ x: MLXArray, M: Int, finalNormW: MTLBuffer? = nil) -> MLXArray? {
            guard M <= maxM else { return nil }
            // hBuf upload(resident/bolt/strict 共通)
            let xf = x.asType(.float16).reshaped([-1]); xf.eval()
            let arr = xf.asArray(Float16.self)
            hBuf.contents().bindMemory(to: Float16.self, capacity: maxM * H).update(from: arr, count: M * H)

            switch streamMode {
            case .resident:
                let cb = queue.makeCommandBuffer()!
                let enc = cb.makeComputeCommandEncoder()!
                for L in layers { encodeLayer(enc, L, M: M) }
                if let fw = finalNormW {
                    RawFusedVerify.encodeRmsNormRows(enc, x: hBuf, w: fw, out: normed, rows: M, D: H, eps: eps)
                }
                enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
                RawFusedForward.profLastGPUMs = (cb.gpuEndTime - cb.gpuStartTime) * 1000.0

            case .bolt:
                let cb = queue.makeCommandBuffer()!
                let enc = cb.makeComputeCommandEncoder()!
                for (li, L) in layers.enumerated() { encodeLayerBolt(enc, L, M: M, li: li) }
                if let fw = finalNormW {
                    RawFusedVerify.encodeRmsNormRows(enc, x: hBuf, w: fw, out: normed, rows: M, D: H, eps: eps)
                }
                enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
                RawFusedForward.profLastGPUMs = (cb.gpuEndTime - cb.gpuStartTime) * 1000.0

            case .strict:
                runStrictLayers(M: M, firstCBExtra: nil, finalCBExtra: { enc in
                    if let fw = finalNormW {
                        RawFusedVerify.encodeRmsNormRows(enc, x: self.hBuf, w: fw, out: self.normed,
                                                         rows: M, D: self.H, eps: self.eps)
                    }
                })
            }

            let src = finalNormW != nil ? normed : hBuf
            let ptr = src.contents().bindMemory(to: Float16.self, capacity: maxM * H)
            return MLXArray(Array(UnsafeBufferPointer(start: ptr, count: M * H)), [M, H])
        }

        /// strict モードのコア: per-layer CB 分割 + CPU ensure + chunk gather ループ。
        /// firstCBExtra: 最初の CB の先頭に encode する追加 op(stepArgmax の embed 等)。
        /// finalCBExtra: 最後の CB に追加する op(finalNorm / lm_head / argmax 等)。
        private func runStrictLayers(M: Int,
                                     firstCBExtra: ((MTLComputeCommandEncoder) -> Void)?,
                                     finalCBExtra: (MTLComputeCommandEncoder) -> Void) {
            var gpuMs = 0.0
            var curCB: MTLCommandBuffer? = nil
            var curEnc: MTLComputeCommandEncoder? = nil

            func openCB() {
                curCB = queue.makeCommandBuffer()!
                curEnc = curCB!.makeComputeCommandEncoder()!
            }
            func flushCB() {
                curEnc!.endEncoding()
                curCB!.commit()
                curCB!.waitUntilCompleted()
                gpuMs += (curCB!.gpuEndTime - curCB!.gpuStartTime) * 1000.0
                curEnc = nil; curCB = nil
            }

            // 最初の CB: (embed など)+ layer[0] pre-MoE + route
            openCB()
            firstCBExtra?(curEnc!)
            encodePreMoE(curEnc!, layers[0], M: M)
            RawFusedVerify.encodeMoERouteRows(curEnc!, x: postNorm, w: layers[0].moe, sc: moeSc,
                                              M: M, E: layers[0].E, H: H, Ktop: layers[0].Ktop)
            flushCB()   // route 読み出しのため待機

            var maxChunks = 0

            for (li, L) in layers.enumerated() {
                let provider = providers![li]
                let indsCount = M * L.Ktop
                let indsRaw = moeSc.inds.contents().bindMemory(to: Int32.self, capacity: indsCount)
                let inds = Array(UnsafeBufferPointer(start: indsRaw, count: indsCount))
                indsCaptureHook?(li, inds)
                let chunks = partitionChunks(inds, M: M, Ktop: L.Ktop, C: provider.C)
                maxChunks = max(maxChunks, chunks.count)

                let stPtr = slotTables[li].contents().bindMemory(to: Int32.self, capacity: L.E)

                for (ci, chunk) in chunks.enumerated() {
                    let slotMap = provider.ensure(chunk.experts)
                    for (e, slot) in slotMap { stPtr[e] = Int32(slot) }

                    if curEnc == nil { openCB() }
                    RawFusedVerify.encodeMoEGatherRowsRange(curEnc!, x: postNorm, w: L.moe, sc: moeSc,
                                                            r0: chunk.r0, r1: chunk.r1,
                                                            Ktop: L.Ktop, I: L.I, H: H,
                                                            slotTable: slotTables[li])
                    if ci < chunks.count - 1 { flushCB() }   // 次 ensure 前に GPU 完了を保証
                }

                // 最後 chunk の CB にそのまま shared + resid を連結
                RawFusedVerify.encodeMoESharedRows(curEnc!, x: postNorm, out: moeOut, w: L.moe, sc: moeSc,
                                                    M: M, I: L.I, H: H)
                RawFusedVerify.encodeResidAdd(curEnc!, h: hBuf, r: moeOut, total: M * H)

                if li + 1 < layers.count {
                    // 次層の pre-MoE + route を同 CB に連結してから flush(route 読み出し)
                    encodePreMoE(curEnc!, layers[li + 1], M: M)
                    RawFusedVerify.encodeMoERouteRows(curEnc!, x: postNorm, w: layers[li + 1].moe, sc: moeSc,
                                                      M: M, E: layers[li + 1].E, H: H, Ktop: layers[li + 1].Ktop)
                    flushCB()
                }
                // else: 最終層 → enc を開けたまま finalCBExtra に渡す
            }

            // 最後の CB に final ops を追加して flush
            finalCBExtra(curEnc!)
            flushCB()

            lastStepChunks = maxChunks
            RawFusedForward.profLastGPUMs = gpuMs
        }

        // ── head 同梱(1-CB step): embed → 40層 → final norm → lm_head(qmmTiled) → argmax ──
        struct HeadBufs {
            let embedW: MTLBuffer, embedS: MTLBuffer, embedB: MTLBuffer
            let lmW: MTLBuffer, lmS: MTLBuffer, lmB: MTLBuffer
            let fnW: MTLBuffer
            let vocab: Int
            let tokensIn: MTLBuffer      // [maxM] int32
            let logits: MTLBuffer        // [maxM, vocab] f16
            let tokensOut: MTLBuffer     // [maxM] int32
            let retained: [MLXArray]
            // ── lmhead2 (Round B): nil 当 feature off。cert 経路の追加常駐 buffer 群。──
            let lmW2: MTLBuffer?, lmS2: MTLBuffer?, lmB2: MTLBuffer?   // 2-bit copy [V,K/16]/[V,K/64]
            let rowE: MTLBuffer?      // [V] f16  per-row error norms
            let hnorm: MTLBuffer?     // [maxM] f32
            let certFlag: MTLBuffer?  // [maxM] int32 (0/1)
            let certCount: MTLBuffer? // [1] int32 atomic(GPU 側で Step 毎加算 → readback で累計)
        }
        var head: HeadBufs? = nil

        /// head(embed/lm_head/final norm)を常駐 buffer 化して 1-CB step を有効化する。
        public func attachHead(embedW: MLXArray, embedS: MLXArray, embedB: MLXArray,
                               lmW: MLXArray, lmS: MLXArray, lmB: MLXArray,
                               fnW: MLXArray, vocab: Int,
                               lm2: (MLXArray, MLXArray, MLXArray, MLXArray)? = nil) -> Bool {
            if RawMetalForward._qmm4TiledPipeline == nil {                 // pipeline warm(compile)
                let x = MLXRandom.normal([1, 512]).asType(.float16)
                let wf = MLXRandom.normal([8, 512]).asType(.float16)
                let (wq, s, b) = MLX.quantized(wf, groupSize: 64, bits: 4, mode: .affine)
                MLX.eval([x, wq, s, b!])
                _ = RawMetalForward.qmmTiled(x, wq, scales: s, biases: b!, M: 1, K: 512, N: 8)
            }
            guard RawMetalForward._qmm4TiledPipeline != nil else { return false }
            var keep: [MLXArray] = []
            let esA = embedS.asType(.float16), ebA = embedB.asType(.float16)
            let lsA = lmS.asType(.float16), lbA = lmB.asType(.float16)
            let fnA = fnW.asType(.float16)
            keep.append(contentsOf: [embedW, esA, ebA, lmW, lsA, lbA, fnA])
            guard let ew = RawMetalForward.mtlBuf(embedW, device),
                  let es = RawMetalForward.mtlBuf(esA, device),
                  let eb = RawMetalForward.mtlBuf(ebA, device),
                  let lw = RawMetalForward.mtlBuf(lmW, device),
                  let ls = RawMetalForward.mtlBuf(lsA, device),
                  let lb = RawMetalForward.mtlBuf(lbA, device),
                  let fn = RawMetalForward.mtlBuf(fnA, device),
                  let ti = device.makeBuffer(length: maxM * 4, options: .storageModeShared),
                  let lg = device.makeBuffer(length: maxM * vocab * 2, options: .storageModeShared),
                  let to = device.makeBuffer(length: maxM * 4, options: .storageModeShared) else { return false }
            head = HeadBufs(embedW: ew, embedS: es, embedB: eb, lmW: lw, lmS: ls, lmB: lb,
                            fnW: fn, vocab: vocab, tokensIn: ti, logits: lg, tokensOut: to, retained: keep,
                            lmW2: nil, lmS2: nil, lmB2: nil, rowE: nil, hnorm: nil, certFlag: nil, certCount: nil)
            return true
        }

        /// lmhead2 用の常駐 buffer 群を構築して head の 2-bit フィールドを埋める(QWISP_LMHEAD2=1 時)。
        /// lm2 = buildLMHead2 の戻り値(w2,s2,b2,rowE)。noCopy 寿命規約: 裏 MLXArray を retain。
        public func attachHeadLMHead2(_ lm2: (MLXArray, MLXArray, MLXArray, MLXArray)) -> Bool {
            guard var hd = head, let (device, _) = RawMetalForward.ensure() else { return false }
            guard ensureRowsAuxPipelines(), ensureQmm2RowsPipeline(), ensureQmm4RowsFlaggedPipeline() else { return false }
            let (w2, s2, b2, rowE) = lm2
            let s2A = s2.asType(.float16), b2A = b2.asType(.float16), rowEA = rowE.asType(.float16)
            guard let bw2 = RawMetalForward.mtlBuf(w2, device),
                  let bs2 = RawMetalForward.mtlBuf(s2A, device),
                  let bb2 = RawMetalForward.mtlBuf(b2A, device),
                  let bre = RawMetalForward.mtlBuf(rowEA, device),
                  let hn = device.makeBuffer(length: maxM * 4, options: .storageModeShared),
                  let cf = device.makeBuffer(length: maxM * 4, options: .storageModeShared),
                  let cc = device.makeBuffer(length: 4, options: .storageModeShared) else { return false }
            retainedArrays.append(contentsOf: [w2, s2A, b2A, rowEA])
            head = HeadBufs(embedW: hd.embedW, embedS: hd.embedS, embedB: hd.embedB,
                            lmW: hd.lmW, lmS: hd.lmS, lmB: hd.lmB, fnW: hd.fnW, vocab: hd.vocab,
                            tokensIn: hd.tokensIn, logits: hd.logits, tokensOut: hd.tokensOut,
                            retained: hd.retained,
                            lmW2: bw2, lmS2: bs2, lmB2: bb2, rowE: bre,
                            hnorm: hn, certFlag: cf, certCount: cc)
            return true
        }

        /// lmhead2 telemetry: engine build 以後の累計 (certified, total)。flag off なら (0,0)。
        public func certRate() -> (certified: Int, total: Int) { (certTotalRows, totalRows) }

        /// 1-CB decode/verify step: token ids → 行毎 greedy argmax token ids。
        /// CB 1 本(resident/bolt)または multi-CB(strict)。readback は int32 [M] のみ(MLX op ゼロ)。
        public func stepArgmax(_ tokens: [Int32]) -> [Int]? {
            guard let hd = head, tokens.count <= maxM else { return nil }
            let M = tokens.count
            hd.tokensIn.contents().bindMemory(to: Int32.self, capacity: maxM).update(from: tokens, count: M)

            // embed: tokens → hBuf [M, H]
            func encodeEmbed(_ enc: MTLComputeCommandEncoder) {
                let ep = RawFusedVerify._embedRowsPipeline!
                enc.setComputePipelineState(ep)
                enc.setBuffer(hd.embedW, offset: 0, index: 0); enc.setBuffer(hd.embedS, offset: 0, index: 1)
                enc.setBuffer(hd.embedB, offset: 0, index: 2); enc.setBuffer(hd.tokensIn, offset: 0, index: 3)
                enc.setBuffer(hBuf, offset: 0, index: 4)
                var hh = UInt32(H), tt = UInt32(M * H)
                enc.setBytes(&hh, length: 4, index: 5); enc.setBytes(&tt, length: 4, index: 6)
                enc.dispatchThreads(MTLSize(width: M * H, height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: min(ep.maxTotalThreadsPerThreadgroup, 256), height: 1, depth: 1))
            }

            // final norm → lm_head(qmm4_tiled or qmv) → argmax
            // lm_head: 既定=qmm4_tiled(重み行を threadgroup に 1 回 dequant し M 行で共有=verify M>1 で有利)。
            // QWISP_LMHEAD_QMV=1: qmm4(per-row qmv, threadgroup dequant 無し=高 occupancy)= M=1 decode で有利。
            // 両者とも M 不変(test 済)ゆえ flag を run 中固定なら decode≡verify(self-consistent)。速度トレードオフのみ。
            func encodeFinalOps(_ enc: MTLComputeCommandEncoder) {
                RawFusedVerify.encodeRmsNormRows(enc, x: hBuf, w: hd.fnW, out: normed, rows: M, D: H, eps: eps)
                // ── lmhead2 (Round B): hnorm → qmm2 → argmax_cert → qmm4_flagged → argmax_flagged ──
                //   同一 encoder/CB、CPU sync 無し。flag off ならこの block 全体を skip(既存経路 byte 不変)。
                if RawFusedForward.lmHead2, let w2 = hd.lmW2, let s2 = hd.lmS2, let b2 = hd.lmB2,
                   let re = hd.rowE, let hn = hd.hnorm, let cf = hd.certFlag, let cc = hd.certCount {
                    cc.contents().bindMemory(to: Int32.self, capacity: 1).pointee = 0   // atomic counter reset
                    RawFusedVerify.encodeHnormRows(enc, x: normed, out: hn, M: M, K: H)
                    RawFusedVerify.encodeQmm2Rows(enc, w: w2, scales: s2, biases: b2, x: normed,
                                                 out: hd.logits, M: M, K: H, V: hd.vocab)
                    RawFusedVerify.encodeArgmaxCertRows(enc, logits2: hd.logits, rowE: re, hnorm: hn,
                                                       tokensOut: hd.tokensOut, certFlag: cf, certCount: cc,
                                                       M: M, V: hd.vocab)
                    RawFusedVerify.encodeQmm4RowsFlagged(enc, w: hd.lmW, scales: hd.lmS, biases: hd.lmB,
                                                        x: normed, out: hd.logits, certFlag: cf,
                                                        M: M, K: H, N: hd.vocab)
                    RawFusedVerify.encodeArgmaxRowsFlagged(enc, logits: hd.logits, outIdx: hd.tokensOut,
                                                           certFlag: cf, M: M, V: hd.vocab)
                    return
                }
                if RawFusedForward.lmHeadQmv {
                    RawFusedVerify.encodeQmmRows(enc, w: hd.lmW, scales: hd.lmS, biases: hd.lmB,
                                                 x: normed, out: hd.logits, M: M, K: H, N: hd.vocab)
                } else {
                    let qp = RawMetalForward._qmm4TiledPipeline!
                    enc.setComputePipelineState(qp)
                    enc.setBuffer(hd.lmW, offset: 0, index: 0); enc.setBuffer(hd.lmS, offset: 0, index: 1)
                    enc.setBuffer(hd.lmB, offset: 0, index: 2); enc.setBuffer(normed, offset: 0, index: 3)
                    enc.setBuffer(hd.logits, offset: 0, index: 4)
                    var kk = Int32(H), nn = Int32(hd.vocab), mm = Int32(M)
                    enc.setBytes(&kk, length: 4, index: 5); enc.setBytes(&nn, length: 4, index: 6); enc.setBytes(&mm, length: 4, index: 7)
                    enc.dispatchThreadgroups(MTLSize(width: hd.vocab, height: 1, depth: 1),
                                             threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                }
                let ap = RawFusedVerify._argmaxRowsPipeline!
                enc.setComputePipelineState(ap)
                enc.setBuffer(hd.logits, offset: 0, index: 0); enc.setBuffer(hd.tokensOut, offset: 0, index: 1)
                var vv = UInt32(hd.vocab); enc.setBytes(&vv, length: 4, index: 2)
                enc.dispatchThreadgroups(MTLSize(width: M, height: 1, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            }

            switch streamMode {
            case .resident:
                let cb = queue.makeCommandBuffer()!
                let enc = cb.makeComputeCommandEncoder()!
                encodeEmbed(enc)
                for L in layers { encodeLayer(enc, L, M: M) }
                encodeFinalOps(enc)
                enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
                RawFusedForward.profLastGPUMs = (cb.gpuEndTime - cb.gpuStartTime) * 1000.0

            case .bolt:
                let cb = queue.makeCommandBuffer()!
                let enc = cb.makeComputeCommandEncoder()!
                encodeEmbed(enc)
                for (li, L) in layers.enumerated() { encodeLayerBolt(enc, L, M: M, li: li) }
                encodeFinalOps(enc)
                enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
                RawFusedForward.profLastGPUMs = (cb.gpuEndTime - cb.gpuStartTime) * 1000.0

            case .strict:
                runStrictLayers(M: M, firstCBExtra: encodeEmbed, finalCBExtra: encodeFinalOps)
            }

            let ptr = hd.tokensOut.contents().bindMemory(to: Int32.self, capacity: maxM)
            // lmhead2 telemetry: certCount(今 step の cert 行数)を累計。flag off なら hd.certCount == nil。
            if let cc = hd.certCount, RawFusedForward.lmHead2 {
                certTotalRows += Int(cc.contents().bindMemory(to: Int32.self, capacity: 1).pointee)
                totalRows += M
            }
            return (0 ..< M).map { Int(ptr[$0]) }
        }

        /// spec の partial reject 用: 直前 forwardRows「1 回だけ」の cache 前進を取り消すための snapshot。
        /// KV は len の巻き戻し、GDN state/conv hist は ping-pong の swap 戻し(裏面に pre-step 値)。
        public struct Snapshot { let kvLens: [Int] }
        public func snapshot() -> Snapshot { Snapshot(kvLens: layers.map { $0.kvCache?.len ?? 0 }) }
        /// snapshot 以降に forwardRows をちょうど 1 回だけ呼んだ状態から巻き戻す(2 回以上は不可)。
        public func rollbackOneStep(_ s: Snapshot) {
            for (i, L) in layers.enumerated() {
                if let kv = L.kvCache { kv.len = s.kvLens[i] }
                if let gc = L.gdnCache { gc.swapState() }
            }
        }

        /// テスト比較用: 層 i の cache を MLX で読む(gdn: (conv, rec) / attn: (k, v))。
        public func readLayerCache(_ i: Int) -> (MLXArray?, MLXArray?) {
            let L = layers[i]
            if let gc = L.gdnCache { let (c, r) = RawFusedVerify.readGdnCache(gc); return (c, r) }
            if let kv = L.kvCache { let (k, v) = RawFusedVerify.readKVCache(kv); return (k, v) }
            return (nil, nil)
        }
    }

    // ── LMHEAD2 (Round A): margin-certified 2-bit lm_head — notes/05-lmhead-margin-cert-spec.md §3 ──
    //
    // These two functions are the CONTRACT API for the margin-certified 2-bit lm_head feature.
    // ε = 0.05 (fixed constant, spec §3.2) lives here only; it must NOT appear in any test
    // assertion or tolerance.

    nonisolated(unsafe) static var _qmm2RowsPipeline: MTLComputePipelineState?

    /// qmm2_rows pipeline を compule(冪等)。qmm2RowsFlat(単発)と fused chain(encodeQmm2Rows)で共有。
    static func ensureQmm2RowsPipeline() -> Bool {
        guard let (device, _) = RawMetalForward.ensure() else { return false }
        if _qmm2RowsPipeline != nil { return true }
        let src = """
        #include <metal_stdlib>
        using namespace metal;
        // qmm2_rows: 2-bit affine dequant-then-dot, one threadgroup per output column n(=v).
        // value j within a uint32 occupies bits 2*j .. 2*j+1 (little-endian, MLX convention —
        // matches the 4-bit kernel's (pack >> (4*(k%8))) & 0xf idiom extended to 2-bit).
        kernel void qmm2_rows(device const uint32_t* w   [[buffer(0)]],   // [V, K/16]
                              device const half*     scales [[buffer(1)]], // [V, K/64]
                              device const half*     biases [[buffer(2)]],
                              device const half*     x      [[buffer(3)]], // [M, K]
                              device half*           y      [[buffer(4)]], // [M, V]
                              constant int& K [[buffer(5)]], constant int& N [[buffer(6)]],
                              constant int& M [[buffer(7)]],
                              uint  n   [[threadgroup_position_in_grid]],
                              uint  lid [[thread_position_in_threadgroup]],
                              uint  tgs [[threads_per_threadgroup]]) {
            threadgroup float wdq[2048];   // dequant-済 weight 行 n (K ≤ 2048)
            threadgroup float red[256];
            const int Kg = K / 64;
            for (int k = (int)lid; k < K; k += (int)tgs) {
                uint pack = w[n * (K/16) + k/16];
                uint q2 = (pack >> (2u * (uint)(k % 16))) & 0x3u;
                int g = k / 64;
                wdq[k] = (float)scales[n*Kg + g] * (float)q2 + (float)biases[n*Kg + g];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (int m = 0; m < M; m++) {
                float acc = 0.0f;
                for (int k = (int)lid; k < K; k += (int)tgs) acc += (float)x[m*K + k] * wdq[k];
                red[lid] = acc; threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint s = tgs/2; s > 0; s >>= 1) {
                    if (lid < s) red[lid] += red[lid + s];
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
                if (lid == 0) y[m*N + n] = (half)red[0];
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
        """
        do { let lib = try device.makeLibrary(source: src, options: RawMetalForward.mlxMatchCompileOpts())
             _qmm2RowsPipeline = try device.makeComputePipelineState(function: lib.makeFunction(name: "qmm2_rows")!)
             return true
        } catch { print("[raw-qmm2-rows] compile: \(error)"); return false }
    }

    /// §3.2 step 2: qmm2_rows — 2-bit affine qmv (gs=64). x[M,K] · Wq2[V,K] → y[M,V] (f16).
    /// Layout (analogous to qmm4_tiled, derived from MLX packing convention used in qmm4_rows):
    ///   - w2   : uint32 [V, K/16]   (2-bit packs 16 values per uint32, little-endian bit order)
    ///   - s2/b2:  f16   [V, K/64]   (one (scale,bias) per group of 64)
    /// Each threadgroup owns one output column v ∈ [0,V): cooperative-dequant weight row v into
    /// threadgroup memory once, then dot against each of the M x-rows (f32 accumulate, f16 store).
    /// Per-row accumulation order depends only on K and threadgroup size (M-independent).
    /// Grid=(V,1,1), threadgroup=256. K must be ≤ 2048 (threadgroup array bound). Returns [M*V] f16
    /// row-major (y[m*V + v]).
    static func qmm2RowsFlat(_ x: MLXArray, _ wq2: MLXArray, scales: MLXArray, biases: MLXArray,
                             M: Int, K: Int, V: Int) -> [Float16]? {
        guard let (device, queue) = RawMetalForward.ensure() else { return nil }
        guard K <= 2048, K % 64 == 0, M >= 1, V >= 1 else { return nil }
        guard ensureQmm2RowsPipeline() else { return nil }
        guard let bx = RawMetalForward.mtlBuf(x.asType(.float16), device),
              let bwq = RawMetalForward.mtlBuf(wq2, device),
              let bsc = RawMetalForward.mtlBuf(scales.asType(.float16), device),
              let bbi = RawMetalForward.mtlBuf(biases.asType(.float16), device)
        else { return nil }
        let outBuf = device.makeBuffer(length: M * V * 2, options: .storageModeShared)!
        let cb = queue.makeCommandBuffer()!; let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(_qmm2RowsPipeline!)
        enc.setBuffer(bwq, offset: 0, index: 0)
        enc.setBuffer(bsc, offset: 0, index: 1)
        enc.setBuffer(bbi, offset: 0, index: 2)
        enc.setBuffer(bx, offset: 0, index: 3)
        enc.setBuffer(outBuf, offset: 0, index: 4)
        var kk = Int32(K), nn = Int32(V), mm = Int32(M)
        enc.setBytes(&kk, length: 4, index: 5); enc.setBytes(&nn, length: 4, index: 6); enc.setBytes(&mm, length: 4, index: 7)
        enc.dispatchThreadgroups(MTLSize(width: V, height: 1, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        let ptr = outBuf.contents().bindMemory(to: Float16.self, capacity: M * V)
        return Array(UnsafeBufferPointer(start: ptr, count: M * V))
    }

    // ── Round B encode helpers: 上記 kernel 群を「既存 encoder に encode するだけ」で提供
    //    (cb/readback 無し)。fused 1-CB chain(hnorm→qmm2→argmax_cert→qmm4_flagged→argmax_flagged)用。

    /// §3.2 step 1 encode: x[M,K] f16 → hnorm[M] f32。1 threadgroup/行, 256 threads。
    static func encodeHnormRows(_ enc: MTLComputeCommandEncoder, x: MTLBuffer, out: MTLBuffer,
                                M: Int, K: Int) {
        enc.setComputePipelineState(_hnormRowsPipeline!)
        enc.setBuffer(x, offset: 0, index: 0); enc.setBuffer(out, offset: 0, index: 1)
        var kk = UInt32(K); enc.setBytes(&kk, length: 4, index: 2)
        enc.dispatchThreadgroups(MTLSize(width: M, height: 1, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    /// §3.2 step 2 encode: 2-bit affine qmv。x[M,K]·Wq2[V,K]→out[M,V] f16(_qmm2RowsPipeline 使用)。
    /// encodeQmmQmv と同 idiom: buffer 0-4 + K/N/M。grid=(V,1,1), threadgroup=256。
    static func encodeQmm2Rows(_ enc: MTLComputeCommandEncoder,
                               w: MTLBuffer, scales: MTLBuffer, biases: MTLBuffer,
                               x: MTLBuffer, out: MTLBuffer, M: Int, K: Int, V: Int) {
        enc.setComputePipelineState(_qmm2RowsPipeline!)
        enc.setBuffer(w, offset: 0, index: 0); enc.setBuffer(scales, offset: 0, index: 1)
        enc.setBuffer(biases, offset: 0, index: 2); enc.setBuffer(x, offset: 0, index: 3)
        enc.setBuffer(out, offset: 0, index: 4)
        var kk = Int32(K), nn = Int32(V), mm = Int32(M)
        enc.setBytes(&kk, length: 4, index: 5); enc.setBytes(&nn, length: 4, index: 6); enc.setBytes(&mm, length: 4, index: 7)
        enc.dispatchThreadgroups(MTLSize(width: V, height: 1, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    /// §3.2 step 3 encode: argmax_cert_rows。logits2[M,V]·rowE[V]·hnorm[M]→tokensOut[M],certFlag[M],certCount。
    static func encodeArgmaxCertRows(_ enc: MTLComputeCommandEncoder,
                                     logits2: MTLBuffer, rowE: MTLBuffer, hnorm: MTLBuffer,
                                     tokensOut: MTLBuffer, certFlag: MTLBuffer, certCount: MTLBuffer,
                                     M: Int, V: Int) {
        enc.setComputePipelineState(_argmaxCertRowsPipeline!)
        enc.setBuffer(logits2, offset: 0, index: 0); enc.setBuffer(rowE, offset: 0, index: 1)
        enc.setBuffer(hnorm, offset: 0, index: 2); enc.setBuffer(tokensOut, offset: 0, index: 3)
        enc.setBuffer(certFlag, offset: 0, index: 4); enc.setBuffer(certCount, offset: 0, index: 5)
        var vv = UInt32(V); var eps = Float(0.05)
        enc.setBytes(&vv, length: 4, index: 6); enc.setBytes(&eps, length: 4, index: 7)
        enc.dispatchThreadgroups(MTLSize(width: M, height: 1, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    /// §3.2 step 5 encode: argmax_rows_flagged。certFlag[m]==0 の行のみ hd.logits(=logits4)の argmax を書く。
    static func encodeArgmaxRowsFlagged(_ enc: MTLComputeCommandEncoder,
                                        logits: MTLBuffer, outIdx: MTLBuffer, certFlag: MTLBuffer,
                                        M: Int, V: Int) {
        enc.setComputePipelineState(_argmaxRowsFlaggedPipeline!)
        enc.setBuffer(logits, offset: 0, index: 0); enc.setBuffer(outIdx, offset: 0, index: 1)
        enc.setBuffer(certFlag, offset: 0, index: 2)
        var vv = UInt32(V); enc.setBytes(&vv, length: 4, index: 3)
        enc.dispatchThreadgroups(MTLSize(width: M, height: 1, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    /// qmm4_rows_flagged pipeline を compile(既存 qmm4 qmv kernel source の忠実コピー + certFlag early-out。
    /// 不等号・演算順序・定数・SIMD 構成は qmm4 と同一 → uncert 行の logits4 は _qmmPipeline と bit-identical。
    /// compile opts も QWISP_QMM_MATH を尊重して qmm4 と一致させる)。
    static func ensureQmm4RowsFlaggedPipeline() -> Bool {
        guard let (device, _) = RawMetalForward.ensure() else { return false }
        if _qmm4RowsFlaggedPipeline != nil { return true }
        let XT = "half"
        let src = """
        #include <metal_stdlib>
        using namespace metal;
        #define SIMD_SIZE 32
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
        kernel void qmm4_rows_flagged(device const uint32_t* w      [[buffer(0)]],
                                      device const \(XT)*    scales [[buffer(1)]],
                                      device const \(XT)*    biases [[buffer(2)]],
                                      device const \(XT)*    x      [[buffer(3)]],
                                      device \(XT)*          y      [[buffer(4)]],
                                      constant int&          in_vec_size  [[buffer(5)]],
                                      constant int&          out_vec_size [[buffer(6)]],
                                      device const int*      certFlag    [[buffer(7)]],
                                      device const int*      stopFlag [[buffer(16)]],
                                      uint3 tid      [[threadgroup_position_in_grid]],
                                      uint  simd_gid [[simdgroup_index_in_threadgroup]],
                                      uint  simd_lid [[thread_index_in_simdgroup]]) {
            if (stopFlag[0] != 0) return;
            if (certFlag[tid.x] != 0) return;               // ★ cert 行は weight を読まず即 return
            constexpr int packs_per_thread = 2;
            constexpr int num_simdgroups = 2;
            constexpr int results_per_simdgroup = 4;
            constexpr int pack_factor = 8;
            constexpr int bytes_per_pack = 4;
            constexpr int values_per_thread = 16;
            constexpr int block_size = 512;
            constexpr int scale_step_per_thread = 4;
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
        do { let lib = try device.makeLibrary(source: src, options: opts)
             _qmm4RowsFlaggedPipeline = try device.makeComputePipelineState(function: lib.makeFunction(name: "qmm4_rows_flagged")!)
             return true
        } catch { print("[raw-qmm4-rows-flagged] compile: \(error)"); return false }
    }

    /// §3.2 step 4 encode: qmm4_rows_flagged(既存 qmm4 qmv と同一計算 + certFlag early-out)。
    /// uncert 行のみ logits4 を hd.logits へ上書き。cert 行は weight を読まない。
    static func encodeQmm4RowsFlagged(_ enc: MTLComputeCommandEncoder,
                                      w: MTLBuffer, scales: MTLBuffer, biases: MTLBuffer,
                                      x: MTLBuffer, out: MTLBuffer, certFlag: MTLBuffer,
                                      M: Int, K: Int, N: Int) {
        enc.setComputePipelineState(_qmm4RowsFlaggedPipeline!)
        enc.setBuffer(w, offset: 0, index: 0); enc.setBuffer(scales, offset: 0, index: 1)
        enc.setBuffer(biases, offset: 0, index: 2); enc.setBuffer(x, offset: 0, index: 3)
        enc.setBuffer(out, offset: 0, index: 4)
        var kk = Int32(K), nn = Int32(N)
        enc.setBytes(&kk, length: 4, index: 5); enc.setBytes(&nn, length: 4, index: 6)
        enc.setBuffer(certFlag, offset: 0, index: 7)
        RawMetalForward.bindStop(enc, 16)
        enc.dispatchThreadgroups(MTLSize(width: M, height: N / 8, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 32, height: 2, depth: 1))
    }

    /// Test entry: full margin-cert chain on provided quantized weights.
    /// x      [M, K] f16  — final-normed hidden rows for M decode steps.
    /// w4/s4/b4            — 4-bit affine lm_head (groupSize=64) already resident.
    /// w2/s2/b2            — 2-bit affine copy built by buildLMHead2 (groupSize=64).
    /// rowE   [V] f16      — per-row error norms: rowE[v] = ‖dq2(v)−dq4(v)‖₂.
    /// Returns (tokens:[M], cert:[M]) or nil on setup failure.
    //
    // Round B: production kernel chain(spec §3.2 — composed==fused doctrine)。1 CB で
    //   hnorm_rows → qmm2_rows → argmax_cert_rows → qmm4_rows_flagged → argmax_rows_flagged
    //   を実行し tokensOut+certFlag を読み出す。EPS=0.05 は argmax_cert_rows kernel 内のみ。
    //   (Round A の CPU composition は削除: 単体 test が本 chain を直接 gate する)
    public static func lmhead2CertStep(x: MLXArray,
                                       w4: MLXArray, s4: MLXArray, b4: MLXArray,
                                       w2: MLXArray, s2: MLXArray, b2: MLXArray,
                                       rowE: MLXArray,
                                       M: Int, K: Int, V: Int) -> (tokens: [Int], cert: [Bool])? {
        guard let (device, queue) = RawMetalForward.ensure() else { return nil }
        // Round B: production kernel chain を 1 CB で駆動(spec §3.2 — composed==fused doctrine)。
        //   hnorm_rows → qmm2_rows → argmax_cert_rows → qmm4_rows_flagged → argmax_rows_flagged。
        //   readback は tokensOut(全部行) + certFlag(各行 0/1)。EPS=0.05 は argmax_cert_rows kernel 内のみ。
        guard ensureRowsAuxPipelines(), ensureQmm2RowsPipeline(), ensureQmm4RowsFlaggedPipeline() else { return nil }
        let xf16 = x.asType(.float16)
        let s2A = s2.asType(.float16), b2A = b2.asType(.float16), rowEA = rowE.asType(.float16)
        let s4A = s4.asType(.float16), b4A = b4.asType(.float16)
        guard let bx = RawMetalForward.mtlBuf(xf16, device),
              let bw2 = RawMetalForward.mtlBuf(w2, device),
              let bs2 = RawMetalForward.mtlBuf(s2A, device),
              let bb2 = RawMetalForward.mtlBuf(b2A, device),
              let bre = RawMetalForward.mtlBuf(rowEA, device),
              let bw4 = RawMetalForward.mtlBuf(w4, device),
              let bs4 = RawMetalForward.mtlBuf(s4A, device),
              let bb4 = RawMetalForward.mtlBuf(b4A, device),
              let lg = device.makeBuffer(length: M * V * 2, options: .storageModeShared),
              let hn = device.makeBuffer(length: M * 4, options: .storageModeShared),
              let cf = device.makeBuffer(length: M * 4, options: .storageModeShared),
              let cc = device.makeBuffer(length: 4, options: .storageModeShared),
              let to = device.makeBuffer(length: M * 4, options: .storageModeShared) else { return nil }
        cc.contents().bindMemory(to: Int32.self, capacity: 1).pointee = 0   // atomic counter reset
        let cb = queue.makeCommandBuffer()!; let enc = cb.makeComputeCommandEncoder()!
        encodeHnormRows(enc, x: bx, out: hn, M: M, K: K)
        encodeQmm2Rows(enc, w: bw2, scales: bs2, biases: bb2, x: bx, out: lg, M: M, K: K, V: V)
        encodeArgmaxCertRows(enc, logits2: lg, rowE: bre, hnorm: hn,
                             tokensOut: to, certFlag: cf, certCount: cc, M: M, V: V)
        encodeQmm4RowsFlagged(enc, w: bw4, scales: bs4, biases: bb4, x: bx, out: lg, certFlag: cf,
                             M: M, K: K, N: V)
        encodeArgmaxRowsFlagged(enc, logits: lg, outIdx: to, certFlag: cf, M: M, V: V)
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        let tp = to.contents().bindMemory(to: Int32.self, capacity: M)
        let cp = cf.contents().bindMemory(to: Int32.self, capacity: M)
        let tokens = (0 ..< M).map { Int(tp[$0]) }
        let cert = (0 ..< M).map { cp[$0] != 0 }
        return (tokens, cert)
    }

    /// Build 2-bit copy of a 4-bit lm_head and compute per-row error norms.
    /// w4/s4/b4 — input 4-bit affine lm_head (groupSize=64, shape [V, K/8] packed).
    /// Returns (w2, s2, b2, rowE) where rowE[v] = ‖dq2(v)−dq4(v)‖₂ stored as f16.
    //
    // §3.1: 2-bit is the quantization of the 4-bit *dequant result* (not the original weights), so
    // the error vector dq2−dq4 is exactly what the bound (§3.2) uses. rowE computed in f32 then
    // downcast to f16 per spec.
    public static func buildLMHead2(w4: MLXArray, s4: MLXArray, b4: MLXArray,
                                    V: Int, K: Int)
        -> (w2: MLXArray, s2: MLXArray, b2: MLXArray, rowE: MLXArray)? {
        // W4 = dequant(w4, s4, b4) (f32 for stable L2), kept only transiently.
        let dq4f = MLX.dequantized(w4, scales: s4, biases: b4, groupSize: 64, bits: 4, mode: .affine)
            .asType(.float32)
        MLX.eval(dq4f)
        // The 4-bit dequant result is fed to MLX.quantized as f16 (same convention as the rest of
        // the repo: MLX.quantized is always called with an f16 source here).
        let dq4f16 = dq4f.asType(.float16)
        MLX.eval(dq4f16)
        // (lmW2, lmS2, lmB2) = quantize the 4-bit dequant result to 2-bit affine, gs=64.
        let (w2, s2, b2opt) = MLX.quantized(dq4f16, groupSize: 64, bits: 2, mode: .affine)
        guard let b2 = b2opt else { return nil }
        // Re-dequant the 2-bit copy to compute the per-row error norm.
        let dq2f = MLX.dequantized(w2, scales: s2, biases: b2, groupSize: 64, bits: 2, mode: .affine)
            .asType(.float32)
        MLX.eval(dq2f)
        // rowE[v] = ‖dq2(v) − dq4(v)‖₂  (L2 over K), f32 → f16.
        let diff = dq2f - dq4f
        let rowE = MLX.sqrt(MLX.sum(diff * diff, axis: -1)).asType(.float16)
        MLX.eval([w2, s2, b2, rowE])
        return (w2, s2, b2, rowE)
    }
}
