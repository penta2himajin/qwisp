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

    /// gqmm4_swiglu_rows を encode-only で提供。grid/threads は gather-g+gather-u 2 dispatch と同一 shape。
    /// x を 1 回読み(gate/up 共有)、g/u を register 内で swiglu → h 直接書き。3 dispatch+中間 g/u を 1 dispatch に。
    static func encodeGatherQmmSwigluRows(_ enc: MTLComputeCommandEncoder,
                                          wG: MTLBuffer, sG: MTLBuffer, bG: MTLBuffer,
                                          wU: MTLBuffer, sU: MTLBuffer, bU: MTLBuffer,
                                          x: MTLBuffer, inds: MTLBuffer, out: MTLBuffer,
                                          M: Int, Ktop: Int, K: Int, N: Int,
                                          xByteOffset: Int = 0, indsOffset: Int = 0, outByteOffset: Int = 0) {
        enc.setComputePipelineState(RawMetalForward._gqmmSwigluRowsPipeline!)
        enc.setBuffer(wG, offset: 0, index: 0); enc.setBuffer(sG, offset: 0, index: 1); enc.setBuffer(bG, offset: 0, index: 2)
        enc.setBuffer(wU, offset: 0, index: 3); enc.setBuffer(sU, offset: 0, index: 4); enc.setBuffer(bU, offset: 0, index: 5)
        enc.setBuffer(x,  offset: xByteOffset, index: 6)
        enc.setBuffer(inds, offset: indsOffset, index: 7)
        enc.setBuffer(out, offset: outByteOffset, index: 8)
        var kk = Int32(K), nn = Int32(N), kt = Int32(Ktop)
        enc.setBytes(&kk, length: 4, index: 9); enc.setBytes(&nn, length: 4, index: 10); enc.setBytes(&kt, length: 4, index: 11)
        RawMetalForward.bindStop(enc, 12)
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
    /// Testable seam: incremented each time encodeCombineRows dispatches a combine_rows kernel.
    /// When the MOE2 fold is active, encodeMoEGatherRowsRange skips the separate combine dispatch
    /// and the count stays at 0 for that block call.
    nonisolated(unsafe) public static var _combineRowsDispatchCount: Int = 0
    nonisolated(unsafe) static var _finalCombineRowsPipeline: MTLComputePipelineState?
    nonisolated(unsafe) static var _writeKVRowsPipeline: MTLComputePipelineState?
    nonisolated(unsafe) static var _convHistRowsPipeline: MTLComputePipelineState?
    nonisolated(unsafe) static var _shiftConvRowsPipeline: MTLComputePipelineState?
    nonisolated(unsafe) static var _sliceRowsPipeline: MTLComputePipelineState?
    nonisolated(unsafe) static var _computeGBetaRowsPipeline: MTLComputePipelineState?
    nonisolated(unsafe) static var _embedRowsPipeline: MTLComputePipelineState?
    nonisolated(unsafe) static var _argmaxRowsPipeline: MTLComputePipelineState?
    nonisolated(unsafe) static var _convShiftFusedRowsPipeline: MTLComputePipelineState?
    nonisolated(unsafe) static var _normGateFusedPipeline: MTLComputePipelineState?        // f16 weight (non-promote)
    nonisolated(unsafe) static var _normGateFusedF32Pipeline: MTLComputePipelineState?    // f32 weight (promote)
    // ── Wave 1 GDN fusion re-design (notes/07 §6) — F1 demux + F4 true fused norm+gate ──
    nonisolated(unsafe) static var _qmmInProjDemuxRowsPipeline: MTLComputePipelineState?   // F1: 1 qmm4 over concat weights → 4 out buffers
    nonisolated(unsafe) static var _gdnNormGateRowsPipeline: MTLComputePipelineState?      // F4: f16 weight (non-promote)
    nonisolated(unsafe) static var _gdnNormGateRowsF32Pipeline: MTLComputePipelineState?   // F4: f32 weight (promote)
    // ── Wave 2 GDN fusion (notes/07 §3 Wave 2) — F2 gdn_prep_rows + F5 gdn_resid_postnorm_rows ──
    nonisolated(unsafe) static var _gdnPrepRowsPipeline: MTLComputePipelineState?           // F2: 8→1 dispatch (slice+rmsnorm+scale+gbeta)
    nonisolated(unsafe) static var _gdnResidPostNormRowsPipeline: MTLComputePipelineState?  // F5: 2→1 dispatch (resid_add+postNorm)
    // ── Wave 3 attn+shexp fusion (notes/08 §3) — A2/A3/S2 single-dispatch kernels ──
    nonisolated(unsafe) static var _sharedGateCombineRowsPipeline: MTLComputePipelineState?  // S2: qmm8(8 dots)+final_combine → 1 dispatch (safe-math)
    nonisolated(unsafe) static var _attnQPrepRowsPipeline: MTLComputePipelineState?          // A2: extract+rmsnorm+rope → 1 dispatch (fast-math, matches rope_rows)
    nonisolated(unsafe) static var _attnKPrepRowsPipeline: MTLComputePipelineState?          // A3: rmsnorm+rope+write_kv → 1 dispatch (fast-math, matches rope_rows)
    nonisolated(unsafe) static var _sharedGateCombineRowsFoldPipeline: MTLComputePipelineState?  // S2 fold (MOE2): qmm8 + inlined combine_rows → 1 dispatch (safe-math)
    // ── Wave 3 attn+shexp: helpers ──
    /// S1 production wiring: dummy inds=[0] for gqmm4_swiglu_rows plain-qmm (no gather) path.
    nonisolated(unsafe) static var _zeroOneInds: MTLBuffer?
    static func zeroOneIndsBuf() -> MTLBuffer? {
        if _zeroOneInds == nil, let (device, _) = RawMetalForward.ensure() {
            let buf = device.makeBuffer(length: 4, options: .storageModeShared)!
            buf.contents().bindMemory(to: Int32.self, capacity: 1)[0] = 0
            _zeroOneInds = buf
        }
        return _zeroOneInds
    }

    /// M-row elementwise 補助 kernel(combine/final)。composed の MLX glue と同一の演算列を
    /// per-element/per-token 独立で再現(f16 逐次和・stable sigmoid)→ M 非依存。
    static func ensureRowsAuxPipelines() -> Bool {
        guard let (device, _) = RawMetalForward.ensure() else { return false }
        if _combineRowsPipeline != nil && _finalCombineRowsPipeline != nil && _writeKVRowsPipeline != nil
            && _convHistRowsPipeline != nil && _shiftConvRowsPipeline != nil
            && _sliceRowsPipeline != nil && _computeGBetaRowsPipeline != nil
            && _embedRowsPipeline != nil && _argmaxRowsPipeline != nil
            && _convShiftFusedRowsPipeline != nil
            && _normGateFusedPipeline != nil && _normGateFusedF32Pipeline != nil
            && _qmmInProjDemuxRowsPipeline != nil
            && _gdnNormGateRowsPipeline != nil && _gdnNormGateRowsF32Pipeline != nil
            && _gdnPrepRowsPipeline != nil && _gdnResidPostNormRowsPipeline != nil { return true }
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
            // ── Wave 1 GDN fusion kernels (notes/07 §3) — separate library(f32 weight variant) ──
            try ensureGdnFusionPipelines(device)
            return true
        } catch { print("[raw-fused-aux] compile: \(error)"); return false }
    }

    /// ── Wave 1 GDN fusion pipelines (notes/07 §3) — separate compile(bilingual weight dtype) ──
    /// conv_shift_fused_rows: 1 dispatch producing BOTH convOut[M,C] (= conv1d_silu_hist_rows
    ///   arithmetic replicated verbatim) AND histOut[K-1,C] (= shift_conv_rows arithmetic).
    ///   Single 2D grid C×(M+1) packed: rows [0,M) compute convOut, row M sweeps histOut.
    /// norm_gate_fused / norm_gate_fused_f32: per-head rmsnorm(reduction tree identical to the
    ///   existing rmsnorm kernel incl. N_READS=4 + simd_sum two-stage + precise::rsqrt, fully
    ///   templated on WT) then silu(z)·normed applied in registers.
    static func ensureGdnFusionPipelines(_ device: MTLDevice) throws {
        if _convShiftFusedRowsPipeline != nil
            && _normGateFusedPipeline != nil && _normGateFusedF32Pipeline != nil
            && _qmmInProjDemuxRowsPipeline != nil
            && _gdnNormGateRowsPipeline != nil && _gdnNormGateRowsF32Pipeline != nil
            && _gdnPrepRowsPipeline != nil && _gdnResidPostNormRowsPipeline != nil { return }
        let srcStatic = """
        #include <metal_stdlib>
        #include <metal_simdgroup>
        using namespace metal;
        // conv1d_silu_hist + shift_conv in one packed 2D dispatch.
        // Arithmetic is byte-for-byte the existing two kernels (same order/precision).
        kernel void conv_shift_fused_rows(device const half* hist  [[buffer(0)]],
                                          device const half* qkv   [[buffer(1)]],
                                          device const float* w    [[buffer(2)]],
                                          device half* convOut     [[buffer(3)]],
                                          device half* histOut     [[buffer(4)]],
                                          constant uint& K [[buffer(5)]], constant uint& C [[buffer(6)]],
                                          constant uint& M [[buffer(7)]],
                                          uint2 pos [[thread_position_in_grid]]) {
            uint c = pos.x;
            if (c >= C) return;
            if (pos.y < M) {
                uint m = pos.y;
                float acc = 0.0f;
                for (uint k = 0; k < K; ++k) {
                    uint idx = m + k;
                    float xv = (idx < K - 1) ? (float)hist[idx*C + c] : (float)qkv[(idx - (K-1))*C + c];
                    acc += xv * w[c*K + k];
                }
                float ax = metal::abs(acc);
                float y = 1.0f / (1.0f + precise::exp(ax));
                float s = (acc < 0.0f) ? y : (1.0f - y);
                convOut[m*C + c] = (half)(acc * s);
            } else if (pos.y == M) {
                // shift_conv_rows: histOut[j*C+c] = (hist‖qkv)[M+j][c], j∈[0,K-1)
                for (uint j = 0; j + 1 < K; ++j) {
                    uint src = M + j;
                    histOut[j*C + c] = (src < K - 1) ? hist[src*C + c] : qkv[(src - (K-1))*C + c];
                }
            }
        }
        // ── F1 re-design (§6): qmm4_inproj_demux_rows — ONE qmm4 over concatenated weights
        //   [totalN, K] 4-bit; each threadgroup (8 output cols, multiples-of-8 boundaries) selects
        //   which of FOUR output buffers (qkv/z/bP/aP) to write at the right local offset.
        //   Dot arithmetic / accumulation order is byte-identical to qmm4_rows → bit-exact.
        inline float ld16_demux(const device half* x, thread float* xt) {
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
        inline float qd4_demux(const device uint8_t* w, const thread float* xt, float scale, float bias, float sum) {
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
        kernel void qmm4_inproj_demux_rows(device const uint32_t* w   [[buffer(0)]],
                                           device const half*  scales [[buffer(1)]],
                                           device const half*  biases [[buffer(2)]],
                                           device const half*  x      [[buffer(3)]],
                                           device half*  yQkv         [[buffer(4)]],
                                           device half*  yZ           [[buffer(5)]],
                                           device half*  yB           [[buffer(6)]],
                                           device half*  yA           [[buffer(7)]],
                                           constant int& in_vec_size  [[buffer(8)]],   // K
                                           constant int& out_vec_size [[buffer(9)]],   // totalN
                                           constant int& qkvN         [[buffer(10)]],
                                           constant int& zN           [[buffer(11)]],
                                           constant int& bN           [[buffer(12)]],
                                           device const int* stopFlag [[buffer(16)]],
                                           uint3 tid      [[threadgroup_position_in_grid]],
                                           uint  simd_gid [[simdgroup_index_in_threadgroup]],
                                           uint  simd_lid [[thread_index_in_simdgroup]]) {
            if (stopFlag[0] != 0) return;
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
            // Demux: 8-col threadgroup block never straddles a boundary (all dims %8==0).
            // ws/scales/biases use the absolute out_row (concat layout); y is remapped to local.
            ws     += out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
            scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
            biases += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
            device half* y; int bufN; int localRow;
            int zEnd = qkvN + zN, bEnd = zEnd + bN;
            if (out_row < qkvN)      { y = yQkv; bufN = qkvN;  localRow = out_row; }
            else if (out_row < zEnd) { y = yZ;   bufN = zN;    localRow = out_row - qkvN; }
            else if (out_row < bEnd) { y = yB;   bufN = bN;    localRow = out_row - zEnd; }
            else                     { y = yA;   bufN = out_vec_size - bEnd; localRow = out_row - bEnd; }
            x += tid.x * in_vec_size + simd_lid * values_per_thread;
            y += tid.x * bufN + localRow;
            for (int k = 0; k < in_vec_size; k += block_size) {
                U sum = ld16_demux(x, x_thread);
                for (int row = 0; row < results_per_simdgroup; row++) {
                    auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
                    const device half* sl = scales + row * in_vec_size_g;
                    const device half* bl = biases + row * in_vec_size_g;
                    U s = sl[0]; U b = bl[0];
                    result[row] += qd4_demux(wl, x_thread, s, b, sum);
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
        // ── F4 re-design (§6): gdn_norm_gate_rows — TRUE single-dispatch kernel.
        //   1 threadgroup per (m, head): per-head rmsnorm over Dv with reduction tree
        //   byte-identical to the existing rmsnorm kernel (N_READS=4, simd_sum two-stage,
        //   precise::rsqrt, same eps handling, same threadgroup size), then silu(z)⊙ applied
        //   in registers. f16 (non-promote) + promoteF32 (f32 weight) variants.
        //   Reproduces rmsnorm→gate[/gate16] chain bit-exactly: normed = w·(WT)(x·inv_mean)
        //   then outV = (half)(silu(z) · (float)normed).
        """
        // Generate two concrete kernel variants (half / float weight) from one body template
        // via Swift string interpolation — avoids C-preprocessor macro backslash pitfalls in
        // Swift multiline string literals.
        func normGateKernel(_ NAME: String, _ WT: String) -> String {
            return """
            kernel void \(NAME)(device const half* x   [[buffer(0)]],
                                device const half* z   [[buffer(1)]],
                                device const \(WT)*  w    [[buffer(2)]],
                                device half* outV      [[buffer(3)]],
                                constant float&  eps        [[buffer(4)]],
                                constant uint&   axis_size  [[buffer(5)]],
                                constant uint&   w_stride   [[buffer(6)]],
                                device const int* stopFlag [[buffer(16)]],
                                uint gid [[threadgroup_position_in_grid]],
                                uint lid [[thread_position_in_threadgroup]],
                                uint simd_lane_id  [[thread_index_in_simdgroup]],
                                uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
                if (stopFlag[0] != 0) return;
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
                z += gid * (size_t)axis_size + lid * N_READS;
                outV += gid * (size_t)axis_size + lid * N_READS;
                float inv = local_inv_mean[0];
                if (lid * N_READS + N_READS <= axis_size) {
                    for (int i = 0; i < N_READS; i++) {
                        float nrm = (float)(w[w_stride*i] * (\(WT))((float)x[i] * inv));
                        float zf = (float)z[i];
                        float y = 1.0f / (1.0f + exp(metal::abs(zf)));
                        float s = (zf < 0.0f) ? y : (1.0f - y);
                        outV[i] = (half)((zf * s) * nrm);
                    }
                } else {
                    for (int i = 0; i < N_READS; i++) {
                        if ((lid*N_READS+i) < axis_size) {
                            float nrm = (float)(w[w_stride*i] * (\(WT))((float)x[i] * inv));
                            float zf = (float)z[i];
                            float y = 1.0f / (1.0f + exp(metal::abs(zf)));
                            float s = (zf < 0.0f) ? y : (1.0f - y);
                            outV[i] = (half)((zf * s) * nrm);
                        }
                    }
                }
            }
            """
        }
        // ── Wave 2 GDN fusion kernel sources (notes/07 §3 Wave 2) ──
        let wave2Src = """
        // F2: gdn_prep_rows — fused ⑧slice q ⑨slice k ⑩slice v ⑪rmsnorm qn(ones)
        // ⑫rmsnorm kn(ones) ⑬scale_mul q(qScale=invScale²) ⑭scale_mul k(kScale=invScale)
        // ⑮compute_g_beta in ONE dispatch.
        // Grid: M*(numKH*2+numVH) threadgroups × tgSize threads (tgSize=ceil(headKD/4,32)).
        // Seg Q [0, M*numKH):           rmsnorm(ones)+scale_mul(qScale) → qn
        // Seg K [M*numKH, 2*M*numKH):   rmsnorm(ones)+scale_mul(kScale) → kn
        // Seg G [2*M*numKH, ...+M*numVH): v copy (all threads) + compute_g_beta (thread 0)
        // Reduction tree = byte-identical to existing rmsnorm kernel (N_READS=4, SIMD_SIZE=32).
        // scale_mul semantics: x[i]=(half)s*x[i] where (half)s rounds s to f16 FIRST.
        kernel void gdn_prep_rows(
            device const half*  convOut  [[buffer(0)]],
            device const half*  aP       [[buffer(1)]],
            device const half*  bP       [[buffer(2)]],
            device const float* aLog     [[buffer(3)]],
            device const float* dtBias   [[buffer(4)]],
            device half*        qn       [[buffer(5)]],
            device half*        kn       [[buffer(6)]],
            device half*        v        [[buffer(7)]],
            device float*       g        [[buffer(8)]],
            device float*       beta     [[buffer(9)]],
            constant uint&      M_       [[buffer(10)]],
            constant uint&      numKH    [[buffer(11)]],
            constant uint&      headKD   [[buffer(12)]],
            constant uint&      numVH    [[buffer(13)]],
            constant uint&      keyDim   [[buffer(14)]],
            constant uint&      valDim   [[buffer(15)]],
            device const int*   stopFlag [[buffer(16)]],
            constant float&     eps      [[buffer(17)]],
            constant float&     qScale   [[buffer(18)]],
            constant float&     kScale   [[buffer(19)]],
            uint gid [[threadgroup_position_in_grid]],
            uint lid [[thread_position_in_threadgroup]],
            uint simd_lane_id  [[thread_index_in_simdgroup]],
            uint simd_group_id [[simdgroup_index_in_threadgroup]])
        {
            if (stopFlag[0] != 0) return;
            constexpr int N_READS = 4;
            constexpr int SIMD_SIZE = 32;
            const uint convDim = keyDim * 2 + valDim;
            const uint qkEnd = M_ * numKH;
            const uint kEnd  = 2 * M_ * numKH;
            threadgroup float local_inv_mean[1];
            threadgroup float local_sums[SIMD_SIZE];
            if (gid < kEnd) {
                // Segments Q and K: rmsnorm(ones-weight) + scale_mul
                const bool isK = (gid >= qkEnd);
                const uint segGid = isK ? (gid - qkEnd) : gid;
                const uint m   = segGid / numKH;
                const uint hd  = segGid % numKH;
                const uint srcOff = m * convDim + (isK ? keyDim : 0u) + hd * headKD;
                const device half* xp = convOut + srcOff + lid * N_READS;
                float acc = 0;
                if (lid * N_READS + N_READS <= headKD) {
                    for (int i = 0; i < N_READS; i++) { float xi = (float)xp[i]; acc += xi * xi; }
                } else {
                    for (int i = 0; i < N_READS; i++) { if (lid*N_READS+i < headKD) { float xi=(float)xp[i]; acc+=xi*xi; } }
                }
                acc = simd_sum(acc);
                if (simd_group_id == 0) local_sums[simd_lane_id] = 0;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (simd_lane_id == 0) local_sums[simd_group_id] = acc;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (simd_group_id == 0) {
                    acc = simd_sum(local_sums[simd_lane_id]);
                    if (simd_lane_id == 0) local_inv_mean[0] = precise::rsqrt(acc / headKD + eps);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                // Apply: out[i] = (half)scale * (half)(x[i]*inv_mean)
                // (half)scale rounds s to f16 first — matches scale_mul kernel x[i]=(half)s*x[i].
                const half hs = isK ? (half)kScale : (half)qScale;
                float inv = local_inv_mean[0];
                device half* outp = isK ? (kn + segGid * headKD) : (qn + segGid * headKD);
                if (lid * N_READS + N_READS <= headKD) {
                    for (int i = 0; i < N_READS; i++)
                        outp[lid * N_READS + i] = hs * (half)((float)xp[i] * inv);
                } else {
                    for (int i = 0; i < N_READS; i++)
                        if (lid * N_READS + i < headKD)
                            outp[lid * N_READS + i] = hs * (half)((float)xp[i] * inv);
                }
            } else {
                // Segment G/Beta + V
                const uint segGid = gid - kEnd;
                const uint m  = segGid / numVH;
                const uint hv = segGid % numVH;
                // V copy: headKD elements per (m,hv)
                const uint vsrc = m * convDim + 2 * keyDim + hv * headKD;
                const uint vdst = m * valDim  + hv * headKD;
                if (lid * N_READS + N_READS <= headKD) {
                    for (int i = 0; i < N_READS; i++)
                        v[vdst + lid * N_READS + i] = convOut[vsrc + lid * N_READS + i];
                } else {
                    for (int i = 0; i < N_READS; i++)
                        if (lid * N_READS + i < headKD)
                            v[vdst + lid * N_READS + i] = convOut[vsrc + lid * N_READS + i];
                }
                // compute_g_beta: thread 0 only (byte-identical to compute_g_beta_rows kernel)
                if (lid == 0) {
                    const uint idx = m * numVH + hv;
                    half bh = bP[idx];
                    half y = (half)1 / ((half)1 + exp(metal::abs(bh)));
                    half sb = (bh < (half)0) ? y : ((half)1 - y);
                    beta[idx] = (float)sb;
                    float x = (float)aP[idx] + dtBias[hv];
                    float sp = max(x, 0.0f) + precise::log(1.0f + precise::exp(-metal::abs(x)));
                    g[idx] = precise::exp(-precise::exp(aLog[hv]) * sp);
                }
            }
        }
        // F5: gdn_resid_postnorm_rows — fused ⑳resid_add ㉑rmsnorm(post) in ONE dispatch.
        // resid_add: h[i]=(half)((float)h[i]+(float)r[i])  (matches existing resid_add kernel).
        // rmsnorm uses the updated h (half-precision) with the production reduction tree.
        // Grid: M threadgroups × tgSize threads (tgSize=ceil(H/4,32)).
        kernel void gdn_resid_postnorm_rows(
            device half*        h        [[buffer(0)]],
            device const half*  r        [[buffer(1)]],
            device const half*  w        [[buffer(2)]],
            device half*        postNorm [[buffer(3)]],
            constant float&     eps      [[buffer(4)]],
            constant uint&      H_       [[buffer(5)]],
            device const int*   stopFlag [[buffer(16)]],
            uint gid [[threadgroup_position_in_grid]],
            uint lid [[thread_position_in_threadgroup]],
            uint simd_lane_id  [[thread_index_in_simdgroup]],
            uint simd_group_id [[simdgroup_index_in_threadgroup]])
        {
            if (stopFlag[0] != 0) return;
            constexpr int N_READS = 4;
            constexpr int SIMD_SIZE = 32;
            h        += gid * (size_t)H_;
            r        += gid * (size_t)H_;
            postNorm += gid * (size_t)H_;
            threadgroup float local_inv_mean[1];
            threadgroup float local_sums[SIMD_SIZE];
            // Phase 1: resid_add in-place and accumulate h_new^2 for rmsnorm
            float acc = 0;
            const uint base = lid * N_READS;
            if (base + N_READS <= H_) {
                for (int i = 0; i < N_READS; i++) {
                    half hn = (half)((float)h[base+i] + (float)r[base+i]);
                    h[base+i] = hn;
                    acc += (float)hn * (float)hn;
                }
            } else {
                for (int i = 0; i < N_READS; i++) {
                    if (base+i < H_) {
                        half hn = (half)((float)h[base+i] + (float)r[base+i]);
                        h[base+i] = hn;
                        acc += (float)hn * (float)hn;
                    }
                }
            }
            // Reduction (identical to existing rmsnorm kernel)
            acc = simd_sum(acc);
            if (simd_group_id == 0) local_sums[simd_lane_id] = 0;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (simd_lane_id == 0) local_sums[simd_group_id] = acc;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (simd_group_id == 0) {
                acc = simd_sum(local_sums[simd_lane_id]);
                if (simd_lane_id == 0) local_inv_mean[0] = precise::rsqrt(acc / H_ + eps);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            // Phase 2: rmsnorm → postNorm (reads h which now has resid_add result)
            float inv = local_inv_mean[0];
            if (base + N_READS <= H_) {
                for (int i = 0; i < N_READS; i++)
                    postNorm[base+i] = w[base+i] * (half)((float)h[base+i] * inv);
            } else {
                for (int i = 0; i < N_READS; i++)
                    if (base+i < H_)
                        postNorm[base+i] = w[base+i] * (half)((float)h[base+i] * inv);
            }
        }
        """
        let src = srcStatic + "\n" + normGateKernel("gdn_norm_gate_rows", "half") + "\n" + normGateKernel("gdn_norm_gate_rows_f32", "float") + "\n" + wave2Src
        // ── Wave 1 conv+shift fused kernel only. norm+gate fusion is realised at the encode level
        // (chaining the existing _rmsPipeline[_F32] and _gate[_16] in one encoder = bit-exact by
        // construction and reduces the wave-1 dispatch count by 1 with no kernel duplication).
        let lib = try device.makeLibrary(source: src, options: RawMetalForward.mlxMatchCompileOpts())
        _convShiftFusedRowsPipeline = try device.makeComputePipelineState(function: lib.makeFunction(name: "conv_shift_fused_rows")!)
        _qmmInProjDemuxRowsPipeline = try device.makeComputePipelineState(function: lib.makeFunction(name: "qmm4_inproj_demux_rows")!)
        _gdnNormGateRowsPipeline = try device.makeComputePipelineState(function: lib.makeFunction(name: "gdn_norm_gate_rows")!)
        _gdnNormGateRowsF32Pipeline = try device.makeComputePipelineState(function: lib.makeFunction(name: "gdn_norm_gate_rows_f32")!)
        _gdnPrepRowsPipeline = try device.makeComputePipelineState(function: lib.makeFunction(name: "gdn_prep_rows")!)
        _gdnResidPostNormRowsPipeline = try device.makeComputePipelineState(function: lib.makeFunction(name: "gdn_resid_postnorm_rows")!)
        // norm+gate: reuse existing per-op kernels chained in one encoder (bit-exact by construction).
        _normGateFusedPipeline = RawMetalForward._rmsPipeline
        _normGateFusedF32Pipeline = RawMetalForward._rmsPipelineF32
    }

    /// ── Wave 3 attn+shexp fusion pipelines (notes/08 §3) ──
    /// S2 (shared_gate_combine_rows): compiled with mlxMatchCompileOpts (safe-math) — both
    ///   qmm8 and final_combine_rows use safe-math, so the fused kernel matches bit-exactly.
    /// A2 (attn_q_prep_rows) / A3 (attn_k_prep_rows): compiled with options:nil (fast-math)
    ///   to match rope_rows transcendentals. rmsnorm reduction uses precise::rsqrt
    ///   (option-independent). FMA contraction of `xi*xi` is harmless: float16 squared is
    ///   exact in float32 (≤22 mantissa bits < 24), so fma(xi,xi,acc) == (float)(xi*xi)+acc.
    static func ensureWave3Pipelines() -> Bool {
        guard let (device, _) = RawMetalForward.ensure() else { return false }
        if _sharedGateCombineRowsPipeline != nil && _sharedGateCombineRowsFoldPipeline != nil && _attnQPrepRowsPipeline != nil && _attnKPrepRowsPipeline != nil { return true }
        // S2 kernel: safe-math (matches qmm8 + final_combine_rows)
        let s2Src = """
        #include <metal_stdlib>
        #include <metal_simdgroup>
        using namespace metal;
        #define SIMD_SIZE 32
        inline float ld8(const device half* x, thread float* xt) {
            float sum = 0.0f;
            for (int i = 0; i < 8; i++) { float v = x[i]; sum += v; xt[i] = v; }
            return sum;
        }
        inline float qd8(const device uint8_t* w, const thread float* xt, float scale, float bias, float sum) {
            float accum = 0.0f;
            for (int i = 0; i < 8; i++) accum += xt[i] * (float)w[i];
            return scale * accum + sum * bias;
        }
        // S2: fuses qmm8(x→sgl N=8) + final_combine(y,sharedY,sgl→out) into ONE dispatch.
        // Phase 1: qmm8 accumulation byte-identical to qmm8 kernel (same thread layout, same
        //   block_size=256, same simd_sum). Phase 2: final_combine byte-identical to
        //   final_combine_rows (half sigmoid, half multiply-add).
        kernel void shared_gate_combine_rows(
            device const uint32_t* sgW     [[buffer(0)]],
            device const half*     sgS     [[buffer(1)]],
            device const half*     sgB     [[buffer(2)]],
            device const half*     x       [[buffer(3)]],
            device const half*     y       [[buffer(4)]],
            device const half*     sharedY [[buffer(5)]],
            device half*           out     [[buffer(6)]],
            constant int&          K_      [[buffer(7)]],
            constant int&          H_      [[buffer(8)]],
            device const int*      stopFlag [[buffer(9)]],
            uint3 tid      [[threadgroup_position_in_grid]],
            uint  simd_gid [[simdgroup_index_in_threadgroup]],
            uint  simd_lid [[thread_index_in_simdgroup]])
        {
            if (stopFlag[0] != 0) return;
            // ★perf 再設計(bit 不変): combine が読むのは sgl[row 0] の 1 dot のみ。
            // grid = (M, ceil(H/256)) × 256 threads。各 tg は row-0 dot を既存 qmm8 と同一の
            // 演算順で冗長計算(2048 MAC=無視可)→ barrier → 256 thread 並列で H を combine。
            // 旧版は (M,1)×64 threads で H=2048 を 64 thread 直列 → −384µs 退行の真因だった。
            constexpr int packs_per_thread = 2;
            constexpr int pack_factor = 4;
            constexpr int bytes_per_pack = 4;
            constexpr int values_per_thread = 8;
            constexpr int block_size = 256;
            constexpr int scale_step_per_thread = 8;
            const device uint8_t* ws = (const device uint8_t*)sgW;
            threadgroup half sgl_shared[1];
            typedef float U;
            thread U x_thread[8];
            thread U result0 = 0;
            const int in_vec_size_g = K_ / 64;
            if (simd_gid == 0) {
                // row 0 の dot(既存 qmm8 の lane 割当・block 走査・qd8 累積と同一順)
                const device uint8_t* wl = ws + simd_lid * packs_per_thread * bytes_per_pack;
                const device half* sl = sgS + simd_lid / scale_step_per_thread;
                const device half* bl = sgB + simd_lid / scale_step_per_thread;
                const device half* xr = x + tid.x * K_ + simd_lid * values_per_thread;
                for (int k = 0; k < K_; k += block_size) {
                    U sum = ld8(xr, x_thread);
                    U s = sl[0]; U b = bl[0];
                    result0 += qd8(wl, x_thread, s, b, sum);
                    wl += block_size * bytes_per_pack / pack_factor;
                    sl += block_size / 64;
                    bl += block_size / 64;
                    xr += block_size;
                }
                result0 = simd_sum(result0);
                if (simd_lid == 0) sgl_shared[0] = (half)result0;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            half gv = sgl_shared[0];
            half yv = (half)1 / ((half)1 + exp(metal::abs(gv)));
            half s = (gv < (half)0) ? yv : ((half)1 - yv);
            uint tid_in_tg = simd_gid * 32 + simd_lid;
            uint base = tid.x * (uint)H_;
            uint n = tid.y * 256 + tid_in_tg;
            if (n < (uint)H_)
                out[base + n] = y[base + n] + s * sharedY[base + n];
        }
        kernel void shared_gate_combine_rows_fold(
            device const uint32_t* sgW     [[buffer(0)]],
            device const half*     sgS     [[buffer(1)]],
            device const half*     sgB     [[buffer(2)]],
            device const half*     x       [[buffer(3)]],
            device const half*     d       [[buffer(4)]],
            device const half*     scores  [[buffer(5)]],
            device const half*     sharedY [[buffer(6)]],
            device half*           out     [[buffer(7)]],
            constant int&          K_      [[buffer(8)]],
            constant int&          H_      [[buffer(9)]],
            constant int&          Ktop_   [[buffer(10)]],
            device const int*      stopFlag [[buffer(16)]],
            uint3 tid      [[threadgroup_position_in_grid]],
            uint  simd_gid [[simdgroup_index_in_threadgroup]],
            uint  simd_lid [[thread_index_in_simdgroup]])
        {
            if (stopFlag[0] != 0) return;
            constexpr int packs_per_thread = 2;
            constexpr int pack_factor = 4;
            constexpr int bytes_per_pack = 4;
            constexpr int values_per_thread = 8;
            constexpr int block_size = 256;
            constexpr int scale_step_per_thread = 8;
            const device uint8_t* ws = (const device uint8_t*)sgW;
            threadgroup half sgl_shared[1];
            typedef float U;
            thread U x_thread[8];
            thread U result0 = 0;
            if (simd_gid == 0) {
                const device uint8_t* wl = ws + simd_lid * packs_per_thread * bytes_per_pack;
                const device half* sl = sgS + simd_lid / scale_step_per_thread;
                const device half* bl = sgB + simd_lid / scale_step_per_thread;
                const device half* xr = x + tid.x * K_ + simd_lid * values_per_thread;
                for (int k = 0; k < K_; k += block_size) {
                    U sum = ld8(xr, x_thread);
                    U s = sl[0]; U b = bl[0];
                    result0 += qd8(wl, x_thread, s, b, sum);
                    wl += block_size * bytes_per_pack / pack_factor;
                    sl += block_size / 64;
                    bl += block_size / 64;
                    xr += block_size;
                }
                result0 = simd_sum(result0);
                if (simd_lid == 0) sgl_shared[0] = (half)result0;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            half gv = sgl_shared[0];
            half yv = (half)1 / ((half)1 + exp(metal::abs(gv)));
            half s = (gv < (half)0) ? yv : ((half)1 - yv);
            uint tid_in_tg = simd_gid * 32 + simd_lid;
            uint base_row = tid.x;
            uint n = tid.y * 256 + tid_in_tg;
            if (n < (uint)H_) {
                half acc = (half)0;
                for (int k = 0; k < Ktop_; k++) {
                    acc += d[(base_row * Ktop_ + k) * H_ + n] * scores[base_row * Ktop_ + k];
                }
                out[base_row * H_ + n] = acc + s * sharedY[base_row * H_ + n];
            }
        }
        """
        // A2/A3 kernels: fast-math (options:nil) to match rope_rows transcendentals.
        //   rmsnorm uses precise::rsqrt (option-independent → matches rmsnorm kernel).
        let a23Src = """
        #include <metal_stdlib>
        #include <metal_simdgroup>
        using namespace metal;
        // A2: fuses extract_q(lower headDim slice) + rmsnorm(q,qNorm) + rope(q) into ONE dispatch.
        // 1 threadgroup per (m, head). rmsnorm reduction tree byte-identical to rmsnorm kernel
        // (N_READS=4, simd_sum two-stage, precise::rsqrt). rope angle byte-identical to rope_rows.
        // Shared memory bridges rmsnorm output → rope pairing (d ↔ d+hd2 cross-thread access).
        kernel void attn_q_prep_rows(
            device const half* qOut     [[buffer(0)]],
            device const half* qNorm    [[buffer(1)]],
            device half*       qRot     [[buffer(2)]],
            constant uint&     qd2_     [[buffer(3)]],
            constant uint&     headDim_ [[buffer(4)]],
            constant uint&     ropeDim_ [[buffer(5)]],
            constant float&    base     [[buffer(6)]],
            constant uint&     startOff [[buffer(7)]],
            constant uint&     numHeads_ [[buffer(8)]],
            constant float&    eps      [[buffer(9)]],
            device const int*  stopFlag [[buffer(16)]],
            uint gid [[threadgroup_position_in_grid]],
            uint lid [[thread_position_in_threadgroup]],
            uint simd_lane_id  [[thread_index_in_simdgroup]],
            uint simd_group_id [[simdgroup_index_in_threadgroup]])
        {
            if (stopFlag[0] != 0) return;
            constexpr int N_READS = 4;
            constexpr int SIMD_SIZE = 32;
            threadgroup half normed[256];
            threadgroup float local_inv_mean[1];
            threadgroup float local_sums[SIMD_SIZE];
            uint m = gid / numHeads_;
            device const half* xp = qOut + (size_t)gid * qd2_;
            float acc = 0;
            if (lid * N_READS + N_READS <= headDim_) {
                for (int i = 0; i < N_READS; i++) { float xi = (float)xp[lid*N_READS+i]; acc += xi * xi; }
            } else {
                for (int i = 0; i < N_READS; i++) { if (lid*N_READS+i < headDim_) { float xi = (float)xp[lid*N_READS+i]; acc += xi*xi; } }
            }
            acc = simd_sum(acc);
            if (simd_group_id == 0) local_sums[simd_lane_id] = 0;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (simd_lane_id == 0) local_sums[simd_group_id] = acc;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (simd_group_id == 0) {
                acc = simd_sum(local_sums[simd_lane_id]);
                if (simd_lane_id == 0) local_inv_mean[0] = precise::rsqrt(acc / (float)headDim_ + eps);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float inv = local_inv_mean[0];
            if (lid * N_READS + N_READS <= headDim_) {
                for (int i = 0; i < N_READS; i++)
                    normed[lid*N_READS+i] = qNorm[lid*N_READS+i] * (half)((float)xp[lid*N_READS+i] * inv);
            } else {
                for (int i = 0; i < N_READS; i++)
                    if (lid*N_READS+i < headDim_)
                        normed[lid*N_READS+i] = qNorm[lid*N_READS+i] * (half)((float)xp[lid*N_READS+i] * inv);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            device half* outp = qRot + (size_t)gid * headDim_;
            uint hd2 = ropeDim_ >> 1;
            float pos = (float)(startOff + m);
            if (lid * N_READS + N_READS <= headDim_) {
                for (int i = 0; i < N_READS; i++) {
                    uint d = lid*N_READS + i;
                    if (d >= ropeDim_) { outp[d] = normed[d]; }
                    else {
                        uint ii = (d < hd2) ? d : (d - hd2);
                        float freq = exp(-2.0f * (float)ii / (float)ropeDim_ * log(base));
                        float ang = pos * freq;
                        float c = cos(ang), s = sin(ang);
                        float x0 = (float)normed[ii], x1 = (float)normed[ii + hd2];
                        outp[d] = (half)(d < hd2 ? (x0*c - x1*s) : (x0*s + x1*c));
                    }
                }
            } else {
                for (int i = 0; i < N_READS; i++) {
                    uint d = lid*N_READS + i;
                    if (d < headDim_) {
                        if (d >= ropeDim_) { outp[d] = normed[d]; }
                        else {
                            uint ii = (d < hd2) ? d : (d - hd2);
                            float freq = exp(-2.0f * (float)ii / (float)ropeDim_ * log(base));
                            float ang = pos * freq;
                            float c = cos(ang), s = sin(ang);
                            float x0 = (float)normed[ii], x1 = (float)normed[ii + hd2];
                            outp[d] = (half)(d < hd2 ? (x0*c - x1*s) : (x0*s + x1*c));
                        }
                    }
                }
            }
        }
        // A3: fuses rmsnorm(k,kNorm) + rope(k) + write_kv scatter into ONE dispatch.
        // 1 threadgroup per (m, kv-head). Writes BOTH kRot[gid,headDim] and
        // kCache[h, startOff+m, headDim] (scatter byte-identical to write_kv_rows).
        kernel void attn_k_prep_rows(
            device const half* kOut     [[buffer(0)]],
            device const half* kNorm    [[buffer(1)]],
            device half*       kRot     [[buffer(2)]],
            device half*       kCache   [[buffer(3)]],
            constant uint&     headDim_ [[buffer(4)]],
            constant uint&     ropeDim_ [[buffer(5)]],
            constant float&    base     [[buffer(6)]],
            constant uint&     startOff [[buffer(7)]],
            constant uint&     numKV_   [[buffer(8)]],
            constant uint&     maxLen_  [[buffer(9)]],
            constant float&    eps      [[buffer(10)]],
            device const int*  stopFlag [[buffer(16)]],
            uint gid [[threadgroup_position_in_grid]],
            uint lid [[thread_position_in_threadgroup]],
            uint simd_lane_id  [[thread_index_in_simdgroup]],
            uint simd_group_id [[simdgroup_index_in_threadgroup]])
        {
            if (stopFlag[0] != 0) return;
            constexpr int N_READS = 4;
            constexpr int SIMD_SIZE = 32;
            threadgroup half normed[256];
            threadgroup float local_inv_mean[1];
            threadgroup float local_sums[SIMD_SIZE];
            uint m = gid / numKV_;
            uint h = gid % numKV_;
            device const half* xp = kOut + (size_t)gid * headDim_;
            float acc = 0;
            if (lid * N_READS + N_READS <= headDim_) {
                for (int i = 0; i < N_READS; i++) { float xi = (float)xp[lid*N_READS+i]; acc += xi * xi; }
            } else {
                for (int i = 0; i < N_READS; i++) { if (lid*N_READS+i < headDim_) { float xi = (float)xp[lid*N_READS+i]; acc += xi*xi; } }
            }
            acc = simd_sum(acc);
            if (simd_group_id == 0) local_sums[simd_lane_id] = 0;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (simd_lane_id == 0) local_sums[simd_group_id] = acc;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (simd_group_id == 0) {
                acc = simd_sum(local_sums[simd_lane_id]);
                if (simd_lane_id == 0) local_inv_mean[0] = precise::rsqrt(acc / (float)headDim_ + eps);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float inv = local_inv_mean[0];
            if (lid * N_READS + N_READS <= headDim_) {
                for (int i = 0; i < N_READS; i++)
                    normed[lid*N_READS+i] = kNorm[lid*N_READS+i] * (half)((float)xp[lid*N_READS+i] * inv);
            } else {
                for (int i = 0; i < N_READS; i++)
                    if (lid*N_READS+i < headDim_)
                        normed[lid*N_READS+i] = kNorm[lid*N_READS+i] * (half)((float)xp[lid*N_READS+i] * inv);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            device half* outp = kRot + (size_t)gid * headDim_;
            device half* cachep = kCache + (size_t)h * maxLen_ * headDim_ + (size_t)(startOff + m) * headDim_;
            uint hd2 = ropeDim_ >> 1;
            float pos = (float)(startOff + m);
            if (lid * N_READS + N_READS <= headDim_) {
                for (int i = 0; i < N_READS; i++) {
                    uint d = lid*N_READS + i;
                    half val;
                    if (d >= ropeDim_) { val = normed[d]; }
                    else {
                        uint ii = (d < hd2) ? d : (d - hd2);
                        float freq = exp(-2.0f * (float)ii / (float)ropeDim_ * log(base));
                        float ang = pos * freq;
                        float c = cos(ang), s = sin(ang);
                        float x0 = (float)normed[ii], x1 = (float)normed[ii + hd2];
                        val = (half)(d < hd2 ? (x0*c - x1*s) : (x0*s + x1*c));
                    }
                    outp[d] = val; cachep[d] = val;
                }
            } else {
                for (int i = 0; i < N_READS; i++) {
                    uint d = lid*N_READS + i;
                    if (d < headDim_) {
                        half val;
                        if (d >= ropeDim_) { val = normed[d]; }
                        else {
                            uint ii = (d < hd2) ? d : (d - hd2);
                            float freq = exp(-2.0f * (float)ii / (float)ropeDim_ * log(base));
                            float ang = pos * freq;
                            float c = cos(ang), s = sin(ang);
                            float x0 = (float)normed[ii], x1 = (float)normed[ii + hd2];
                            val = (half)(d < hd2 ? (x0*c - x1*s) : (x0*s + x1*c));
                        }
                        outp[d] = val; cachep[d] = val;
                    }
                }
            }
        }
        """
        do {
            let libSafe = try device.makeLibrary(source: s2Src, options: RawMetalForward.mlxMatchCompileOpts())
            _sharedGateCombineRowsPipeline = try device.makeComputePipelineState(function: libSafe.makeFunction(name: "shared_gate_combine_rows")!)
            _sharedGateCombineRowsFoldPipeline = try device.makeComputePipelineState(function: libSafe.makeFunction(name: "shared_gate_combine_rows_fold")!)
            let libFast = try device.makeLibrary(source: a23Src, options: nil)
            _attnQPrepRowsPipeline = try device.makeComputePipelineState(function: libFast.makeFunction(name: "attn_q_prep_rows")!)
            _attnKPrepRowsPipeline = try device.makeComputePipelineState(function: libFast.makeFunction(name: "attn_k_prep_rows")!)
            return true
        } catch { print("[wave3] compile: \(error)"); return false }
    }

    /// F2 (Wave 2): encode gdn_prep_rows — ONE dispatch replacing ⑧slice q/k/v ⑪⑫rmsnorm ⑬⑭scale_mul ⑮compute_g_beta.
    /// convOut[M, convDim] → qn[M*numKH, headKD], kn[M*numKH, headKD], v[M, valDim], g[M*numVH] f32, beta[M*numVH] f32.
    /// Reduction tree byte-identical to existing rmsnorm kernel. scale_mul: (half)scale * (half)(x*inv_mean).
    static func encodeGdnPrepRows(_ enc: MTLComputeCommandEncoder,
                                  convOut: MTLBuffer, aP: MTLBuffer, bP: MTLBuffer,
                                  aLog: MTLBuffer, dtBias: MTLBuffer,
                                  qn: MTLBuffer, kn: MTLBuffer, v: MTLBuffer,
                                  g: MTLBuffer, beta: MTLBuffer,
                                  M: Int, numKH: Int, headKD: Int, numVH: Int,
                                  keyDim: Int, valDim: Int, eps: Float, qScale: Float, kScale: Float) {
        let p = _gdnPrepRowsPipeline!
        enc.setComputePipelineState(p)
        enc.setBuffer(convOut, offset: 0, index: 0); enc.setBuffer(aP,   offset: 0, index: 1)
        enc.setBuffer(bP,      offset: 0, index: 2); enc.setBuffer(aLog, offset: 0, index: 3)
        enc.setBuffer(dtBias,  offset: 0, index: 4); enc.setBuffer(qn,   offset: 0, index: 5)
        enc.setBuffer(kn,      offset: 0, index: 6); enc.setBuffer(v,    offset: 0, index: 7)
        enc.setBuffer(g,       offset: 0, index: 8); enc.setBuffer(beta, offset: 0, index: 9)
        var mm = UInt32(M), nkh = UInt32(numKH), hkd = UInt32(headKD)
        var nvh = UInt32(numVH), kd = UInt32(keyDim), vd = UInt32(valDim)
        enc.setBytes(&mm,  length: 4, index: 10); enc.setBytes(&nkh, length: 4, index: 11)
        enc.setBytes(&hkd, length: 4, index: 12); enc.setBytes(&nvh, length: 4, index: 13)
        enc.setBytes(&kd,  length: 4, index: 14); enc.setBytes(&vd,  length: 4, index: 15)
        RawMetalForward.bindStop(enc, 16)
        var ee = eps, qs = qScale, ks = kScale
        enc.setBytes(&ee, length: 4, index: 17); enc.setBytes(&qs, length: 4, index: 18)
        enc.setBytes(&ks, length: 4, index: 19)
        let tgNeeded = (headKD + 3) / 4
        let tgSize = ((tgNeeded + 31) / 32) * 32
        let numTG = M * (numKH * 2 + numVH)
        enc.dispatchThreadgroups(MTLSize(width: numTG, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
    }

    /// F5 (Wave 2): encode gdn_resid_postnorm_rows — ONE dispatch replacing ⑳resid_add ㉑rmsnorm(post).
    /// h[M,H] is updated in-place (resid_add), postNorm[M,H] receives the rmsnorm result.
    static func encodeGdnResidPostNormRows(_ enc: MTLComputeCommandEncoder,
                                           h: MTLBuffer, r: MTLBuffer, w: MTLBuffer,
                                           postNorm: MTLBuffer, M: Int, H: Int, eps: Float) {
        let p = _gdnResidPostNormRowsPipeline!
        enc.setComputePipelineState(p)
        enc.setBuffer(h,        offset: 0, index: 0); enc.setBuffer(r,        offset: 0, index: 1)
        enc.setBuffer(w,        offset: 0, index: 2); enc.setBuffer(postNorm, offset: 0, index: 3)
        var ee = eps, hh = UInt32(H)
        enc.setBytes(&ee, length: 4, index: 4); enc.setBytes(&hh, length: 4, index: 5)
        RawMetalForward.bindStop(enc, 16)
        let tgNeeded = (H + 3) / 4
        let tgSize = ((tgNeeded + 31) / 32) * 32
        enc.dispatchThreadgroups(MTLSize(width: M, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
    }

    /// encodeGdnFusionConvShift: drive conv_shift_fused_rows in a single command buffer record.
    static func encodeGdnFusionConvShift(_ enc: MTLComputeCommandEncoder, hist: MTLBuffer, qkv: MTLBuffer,
                                         w: MTLBuffer, convOut: MTLBuffer, histOut: MTLBuffer,
                                         M: Int, K: Int, C: Int) {
        let p = _convShiftFusedRowsPipeline!
        enc.setComputePipelineState(p)
        enc.setBuffer(hist, offset: 0, index: 0); enc.setBuffer(qkv, offset: 0, index: 1)
        enc.setBuffer(w, offset: 0, index: 2); enc.setBuffer(convOut, offset: 0, index: 3)
        enc.setBuffer(histOut, offset: 0, index: 4)
        var kk = UInt32(K), cc = UInt32(C), mm = UInt32(M)
        enc.setBytes(&kk, length: 4, index: 5); enc.setBytes(&cc, length: 4, index: 6); enc.setBytes(&mm, length: 4, index: 7)
        enc.dispatchThreads(MTLSize(width: C, height: M + 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: min(p.maxTotalThreadsPerThreadgroup, 256), height: 1, depth: 1))
    }

    /// F1 re-design (§6): encode qmm4_inproj_demux_rows — ONE qmm4 dispatch over the concatenated
    /// in-proj weights [totalN, K] 4-bit, demuxing the 8-col threadgroup output blocks into FOUR
    /// separate output buffers (qkv/z/bP/aP) at the right local offset. Bit-exact with 4 qmm4_rows.
    /// Downstream kernels read the separate buffers unchanged. Dispatch grid = (M, totalN/8, 1).
    static func encodeQmmInProjDemuxRows(_ enc: MTLComputeCommandEncoder,
                                         w: MTLBuffer, scales: MTLBuffer, biases: MTLBuffer,
                                         x: MTLBuffer,
                                         outQkv: MTLBuffer, outZ: MTLBuffer, outB: MTLBuffer, outA: MTLBuffer,
                                         M: Int, K: Int,
                                         dims: (qkv: Int, z: Int, b: Int, a: Int)) {
        let totalN = dims.qkv + dims.z + dims.b + dims.a
        enc.setComputePipelineState(_qmmInProjDemuxRowsPipeline!)
        enc.setBuffer(w, offset: 0, index: 0); enc.setBuffer(scales, offset: 0, index: 1)
        enc.setBuffer(biases, offset: 0, index: 2); enc.setBuffer(x, offset: 0, index: 3)
        enc.setBuffer(outQkv, offset: 0, index: 4); enc.setBuffer(outZ, offset: 0, index: 5)
        enc.setBuffer(outB, offset: 0, index: 6); enc.setBuffer(outA, offset: 0, index: 7)
        var kk = Int32(K), nn = Int32(totalN)
        var qkvN = Int32(dims.qkv), zN = Int32(dims.z), bN = Int32(dims.b)
        enc.setBytes(&kk, length: 4, index: 8); enc.setBytes(&nn, length: 4, index: 9)
        enc.setBytes(&qkvN, length: 4, index: 10); enc.setBytes(&zN, length: 4, index: 11); enc.setBytes(&bN, length: 4, index: 12)
        RawMetalForward.bindStop(enc, 16)
        enc.dispatchThreadgroups(MTLSize(width: M, height: totalN / 8, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 32, height: 2, depth: 1))
    }

    /// F4 re-design (§6): encode gdn_norm_gate_rows — ONE dispatch, 1 threadgroup per (m, head).
    /// per-head rmsnorm over Dv (reduction tree identical to existing rmsnorm kernel) then
    /// silu(z)⊙normed applied in registers, writing outV. Bit-exact with rmsNormRows + gate chain.
    /// grid = M*Hv threadgroups; threadgroup size matches rmsnorm ((Dv/4 ceil 32)).
    static func encodeGdnNormGateRows(_ enc: MTLComputeCommandEncoder,
                                      coreOut: MTLBuffer, z: MTLBuffer, normWeight: MTLBuffer,
                                      outV: MTLBuffer, M: Int, Hv: Int, Dv: Int,
                                      eps: Float, promoteF32: Bool) {
        let p = promoteF32 ? _gdnNormGateRowsF32Pipeline! : _gdnNormGateRowsPipeline!
        enc.setComputePipelineState(p)
        enc.setBuffer(coreOut, offset: 0, index: 0); enc.setBuffer(z, offset: 0, index: 1)
        enc.setBuffer(normWeight, offset: 0, index: 2); enc.setBuffer(outV, offset: 0, index: 3)
        var ee = eps, asz = UInt32(Dv), ws = UInt32(1)
        enc.setBytes(&ee, length: 4, index: 4); enc.setBytes(&asz, length: 4, index: 5); enc.setBytes(&ws, length: 4, index: 6)
        RawMetalForward.bindStop(enc, 16)
        let tgNeeded = (Dv + 3) / 4
        let tgSize = ((tgNeeded + 31) / 32) * 32
        enc.dispatchThreadgroups(MTLSize(width: M * Hv, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
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
        _combineRowsDispatchCount += 1   // testable seam: MOE2 fold skips this call (count stays 0)
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
        // gather g/u + swiglu: flag-on で 1-dispatch 融合(但し M=1 のみ=register 圧迫退行)、
        // flag-off or M>1 で既存 3-kernel 連鎖(byte 不変)。fuseGUActive(M) == (fuseGU && M==1)。
        if RawFusedForward.fuseGUActive(M: Mc) ?? false {
            encodeGatherQmmSwigluRows(enc, wG: w.swGW, sG: w.swGS, bG: w.swGB,
                                      wU: w.swUW, sU: w.swUS, bU: w.swUB,
                                      x: x, inds: sc.inds, out: sc.h,
                                      M: Mc, Ktop: Ktop, K: H, N: I,
                                      xByteOffset: xOff, indsOffset: indsOff, outByteOffset: guOff)
        } else {
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
        }
        // gather d(per-mk lhs, x=sc.h)
        encodeGatherQmmRows(enc, w: w.swDW, scales: w.swDS, biases: w.swDB,
                            x: sc.h, inds: sc.inds, out: sc.d,
                            M: Mc, Ktop: Ktop, K: I, N: H, lhsPer: true,
                            xByteOffset: guOff, indsOffset: indsOff, outByteOffset: dOff)
        // combine → sc.y[r0*H .. r1*H]
        // MOE2 fold: fuseMOE2Enabled && fuseSHEXP && Mc==1 → S2 inlines combine; skip separate dispatch.
        // M>1 (verify batches) keeps the separate combine_rows dispatch (fold inactive at M>1).
        if !(RawFusedForward.fuseMOE2Enabled && RawFusedForward.fuseSHEXP && Mc == 1) {
            encodeCombineRows(enc, d: sc.d, scores: sc.scores, y: sc.y,
                              Ktop: Ktop, N: H, M: Mc,
                              dByteOffset: dOff, scoresOffset: scOff, yByteOffset: yOff)
        }
    }

    /// ③④ shared expert + final combine → out[M,H]。
    /// S1(fuseSHEXP, M==1): shG+shU+swiglu 3 dispatch → 1 gqmm4_swiglu_rows dispatch(−2/MoEブロック)。
    static func encodeMoESharedRows(_ enc: MTLComputeCommandEncoder, x: MTLBuffer, out: MTLBuffer,
                                     w: MoEBlockBufs, sc: MoEScratch, M: Int, I: Int, H: Int, Ktop: Int) {
        // S1: M==1 branch only (register-pressure gate, same doctrine as fuseGU).
        // 原子別 kill-switch(bisect/構成用): QWISP_FUSE_S1=0 で S1 のみ無効。
        if RawFusedForward.fuseSHEXPActive(M: M) ?? false,
           RawFusedForward.fuseS1Enabled,
           let zeroInds = zeroOneIndsBuf(),
           RawMetalForward._gqmmSwigluRowsPipeline != nil {
            // gqmm4_swiglu_rows with Ktop=1, inds=[0]: bit-exact with qmmRows×2+swiglu (no gather)
            encodeGatherQmmSwigluRows(enc, wG: w.shGW, sG: w.shGS, bG: w.shGB,
                                      wU: w.shUW, sU: w.shUS, bU: w.shUB,
                                      x: x, inds: zeroInds, out: sc.shAct,
                                      M: 1, Ktop: 1, K: H, N: I)
        } else {
            encodeQmmRows(enc, w: w.shGW, scales: w.shGS, biases: w.shGB, x: x, out: sc.sg, M: M, K: H, N: I)
            encodeQmmRows(enc, w: w.shUW, scales: w.shUS, biases: w.shUB, x: x, out: sc.su, M: M, K: H, N: I)
            encodeSwiglu(enc, g: sc.sg, u: sc.su, h: sc.shAct, total: M * I)
        }
        encodeQmmRows(enc, w: w.shDW, scales: w.shDS, biases: w.shDB, x: sc.shAct, out: sc.sharedY, M: M, K: I, N: H)
        // S2(fuseSHEXP): qmm8(N=8)+final_combine → ONE dispatch (safe-math, bit-exact, −1/MoEブロック)。
        // MOE2 fold-ON: S2 inlines combine_rows (encodeCombineRows was skipped in encodeMoEGatherRowsRange).
        // Fold applies only at M==1 (same M==1 gate as fuseGU/fuseSHEXP-S1): M>1 uses non-fold S2 path.
        if RawFusedForward.fuseSHEXP && ensureWave3Pipelines() {
            if RawFusedForward.fuseMOE2Enabled && M == 1 {
                encodeSharedGateCombineRowsFold(enc, sgW: w.sgW, sgS: w.sgS, sgB: w.sgB,
                                                x: x, d: sc.d, scores: sc.scores,
                                                sharedY: sc.sharedY, out: out,
                                                K: H, H: H, M: M, Ktop: Ktop)
            } else {
                encodeSharedGateCombineRows(enc, sgW: w.sgW, sgS: w.sgS, sgB: w.sgB,
                                            x: x, y: sc.y, sharedY: sc.sharedY, out: out,
                                            K: H, H: H, M: M)
            }
        } else {
            encodeQmm8Rows(enc, w: w.sgW, scales: w.sgS, biases: w.sgB, x: x, out: sc.sgl, M: M, K: H, N: 8)
            encodeFinalCombineRows(enc, y: sc.y, sharedY: sc.sharedY, sgl: sc.sgl, out: out, N: H, M: M)
        }
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
        encodeMoESharedRows(enc, x: x, out: out, w: w, sc: sc, M: M, I: I, H: H, Ktop: Ktop)
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
        if RawMetalForward._gqmmSwigluRowsPipeline == nil {
            _ = RawMetalForward.compileGqmmSwigluRows()
        }
        return _routeRowsPipeline != nil && RawMetalForward._gqmmRowsPipeline != nil && RawMetalForward._gqmmSwigluRowsPipeline != nil
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

    /// A2 (Wave 3): encode attn_q_prep_rows — ONE dispatch replacing ④extract_q + ⑤rmsnorm_q + ⑦rope_q.
    /// Input qOut[M*numHeads, qd2]; output qRot[M*numHeads, headDim].
    /// 1 threadgroup per (m, head); rmsnorm reduction tree byte-identical to rmsnorm kernel.
    static func encodeAttnQPrepRows(_ enc: MTLComputeCommandEncoder,
                                    qOut: MTLBuffer, qNorm: MTLBuffer, qRot: MTLBuffer,
                                    qd2: Int, headDim: Int, ropeDim: Int, base: Float,
                                    startOffset: Int, numHeads: Int, M: Int, eps: Float) {
        let p = _attnQPrepRowsPipeline!
        enc.setComputePipelineState(p)
        enc.setBuffer(qOut,  offset: 0, index: 0)
        enc.setBuffer(qNorm, offset: 0, index: 1)
        enc.setBuffer(qRot,  offset: 0, index: 2)
        var qd2v = UInt32(qd2), hd = UInt32(headDim), rd = UInt32(ropeDim)
        var bs = base, so = UInt32(startOffset), nh = UInt32(numHeads), ee = eps
        enc.setBytes(&qd2v, length: 4, index: 3); enc.setBytes(&hd, length: 4, index: 4)
        enc.setBytes(&rd, length: 4, index: 5); enc.setBytes(&bs, length: 4, index: 6)
        enc.setBytes(&so, length: 4, index: 7); enc.setBytes(&nh, length: 4, index: 8)
        enc.setBytes(&ee, length: 4, index: 9)
        RawMetalForward.bindStop(enc, 16)
        let tgNeeded = (headDim + 3) / 4
        let tgSize = ((tgNeeded + 31) / 32) * 32
        enc.dispatchThreadgroups(MTLSize(width: M * numHeads, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
    }

    /// A3 (Wave 3): encode attn_k_prep_rows — ONE dispatch replacing ⑥rmsnorm_k + ⑧rope_k + ⑨write_kv_k.
    /// Input kOut[M*numKV, headDim]; writes kRot[M*numKV, headDim] and kCache[h, startOff+m, headDim].
    /// 1 threadgroup per (m, kv-head); rmsnorm + rope + scatter all in registers.
    static func encodeAttnKPrepRows(_ enc: MTLComputeCommandEncoder,
                                    kOut: MTLBuffer, kNorm: MTLBuffer, kRot: MTLBuffer, kCache: MTLBuffer,
                                    headDim: Int, ropeDim: Int, base: Float,
                                    startOffset: Int, numKV: Int, maxLen: Int, M: Int, eps: Float) {
        let p = _attnKPrepRowsPipeline!
        enc.setComputePipelineState(p)
        enc.setBuffer(kOut,   offset: 0, index: 0)
        enc.setBuffer(kNorm,  offset: 0, index: 1)
        enc.setBuffer(kRot,   offset: 0, index: 2)
        enc.setBuffer(kCache, offset: 0, index: 3)
        var hd = UInt32(headDim), rd = UInt32(ropeDim), bs = base
        var so = UInt32(startOffset), nkv = UInt32(numKV), ml = UInt32(maxLen), ee = eps
        enc.setBytes(&hd, length: 4, index: 4); enc.setBytes(&rd, length: 4, index: 5)
        enc.setBytes(&bs, length: 4, index: 6); enc.setBytes(&so, length: 4, index: 7)
        enc.setBytes(&nkv, length: 4, index: 8); enc.setBytes(&ml, length: 4, index: 9)
        enc.setBytes(&ee, length: 4, index: 10)
        RawMetalForward.bindStop(enc, 16)
        let tgNeeded = (headDim + 3) / 4
        let tgSize = ((tgNeeded + 31) / 32) * 32
        enc.dispatchThreadgroups(MTLSize(width: M * numKV, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
    }

    /// S2 (Wave 3): encode shared_gate_combine_rows — ONE dispatch replacing qmm8(N=8)+final_combine.
    /// Compiled with safe-math (mlxMatchCompileOpts) to match both source kernels bit-exactly.
    static func encodeSharedGateCombineRows(_ enc: MTLComputeCommandEncoder,
                                             sgW: MTLBuffer, sgS: MTLBuffer, sgB: MTLBuffer,
                                             x: MTLBuffer, y: MTLBuffer, sharedY: MTLBuffer,
                                             out: MTLBuffer, K: Int, H: Int, M: Int) {
        let p = _sharedGateCombineRowsPipeline!
        enc.setComputePipelineState(p)
        enc.setBuffer(sgW,    offset: 0, index: 0); enc.setBuffer(sgS,     offset: 0, index: 1)
        enc.setBuffer(sgB,    offset: 0, index: 2); enc.setBuffer(x,       offset: 0, index: 3)
        enc.setBuffer(y,      offset: 0, index: 4); enc.setBuffer(sharedY, offset: 0, index: 5)
        enc.setBuffer(out,    offset: 0, index: 6)
        var kk = Int32(K), hh = Int32(H)
        enc.setBytes(&kk, length: 4, index: 7); enc.setBytes(&hh, length: 4, index: 8)
        RawMetalForward.bindStop(enc, 9)
        // ★grid 再設計: (M, ceil(H/256)) × 256 threads(8 simdgroups)。tg ごとに row-0 dot を
        // 冗長計算し、combine を H 全並列で行う(旧 (M,1)×64 は combine 直列化で −384µs)。
        enc.dispatchThreadgroups(MTLSize(width: M, height: (H + 255) / 256, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 32, height: 8, depth: 1))
    }

    /// S2 fold (MOE2): encode shared_gate_combine_rows_fold — ONE dispatch replacing
    /// qmm8(N=8) + combine_rows + final_combine. Reads d/scores directly (combine inlined).
    /// fold-ON (fuseMOE2Enabled && fuseSHEXP) path; bit-exact with the 3-dispatch chain.
    static func encodeSharedGateCombineRowsFold(_ enc: MTLComputeCommandEncoder,
                                                 sgW: MTLBuffer, sgS: MTLBuffer, sgB: MTLBuffer,
                                                 x: MTLBuffer, d: MTLBuffer, scores: MTLBuffer,
                                                 sharedY: MTLBuffer, out: MTLBuffer,
                                                 K: Int, H: Int, M: Int, Ktop: Int) {
        let p = _sharedGateCombineRowsFoldPipeline!
        enc.setComputePipelineState(p)
        enc.setBuffer(sgW,     offset: 0, index: 0); enc.setBuffer(sgS,     offset: 0, index: 1)
        enc.setBuffer(sgB,     offset: 0, index: 2); enc.setBuffer(x,       offset: 0, index: 3)
        enc.setBuffer(d,       offset: 0, index: 4); enc.setBuffer(scores,  offset: 0, index: 5)
        enc.setBuffer(sharedY, offset: 0, index: 6); enc.setBuffer(out,     offset: 0, index: 7)
        var kk = Int32(K), hh = Int32(H), kt = Int32(Ktop)
        enc.setBytes(&kk, length: 4, index: 8); enc.setBytes(&hh, length: 4, index: 9)
        enc.setBytes(&kt, length: 4, index: 10)
        RawMetalForward.bindStop(enc, 16)
        enc.dispatchThreadgroups(MTLSize(width: M, height: (H + 255) / 256, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 32, height: 8, depth: 1))
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
        // Wave 3 A1: q/k/v 結合 in-proj(fuseATTN || QWISP_PROF_AB 時に build)
        let catQkvW: MTLBuffer?, catQkvS: MTLBuffer?, catQkvB: MTLBuffer?
        let catQkvDummy: MTLBuffer?   // aN=0 placeholder(4th demux output, never written)
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
        // Wave 3 A1: build concat q/k/v in-proj weights when fuseATTN or PROF_AB is active.
        let profAB = ProcessInfo.processInfo.environment["QWISP_PROF_AB"] == "1"
        var catQkvW: MTLBuffer? = nil, catQkvS: MTLBuffer? = nil, catQkvB: MTLBuffer? = nil
        var catQkvDummy: MTLBuffer? = nil
        if RawFusedForward.fuseATTN || profAB {
            let catWArr = MLX.concatenated([w.qWq, w.kWq, w.vWq], axis: 0)
            let catSArr = MLX.concatenated([w.qSc.asType(.float16), w.kSc.asType(.float16), w.vSc.asType(.float16)], axis: 0)
            let catBArr = MLX.concatenated([w.qBi.asType(.float16), w.kBi.asType(.float16), w.vBi.asType(.float16)], axis: 0)
            MLX.eval([catWArr, catSArr, catBArr])
            keep.append(contentsOf: [catWArr, catSArr, catBArr])
            catQkvW = RawMetalForward.mtlBuf(catWArr, device)
            catQkvS = RawMetalForward.mtlBuf(catSArr, device)
            catQkvB = RawMetalForward.mtlBuf(catBArr, device)
            catQkvDummy = device.makeBuffer(length: 4, options: .storageModeShared)
        }
        return AttnLayerBufs(qW: q.0, qS: q.1, qB: q.2, kW: k.0, kS: k.1, kB: k.2,
                             vW: v.0, vS: v.1, vB: v.2, oW: o.0, oS: o.1, oB: o.2,
                             qNorm: qn, kNorm: kn, retained: keep,
                             catQkvW: catQkvW, catQkvS: catQkvS, catQkvB: catQkvB,
                             catQkvDummy: catQkvDummy)
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
        // A1(fuseATTN): 3 separate qmm4 dispatches → 1 demux dispatch(−2 per attn layer).
        // Cat weights built at prepare time; bit-exact by qmm4_inproj_demux_rows construction.
        // 原子別 kill-switch(bisect/構成用): QWISP_FUSE_A1=0 で A1 のみ無効。
        if RawFusedForward.fuseATTN,
           RawFusedForward.fuseA1Enabled,
           let cw = w.catQkvW, let cs = w.catQkvS, let cb = w.catQkvB, let dummy = w.catQkvDummy {
            let Nq = numHeads * qd2, Nk = numKV * headDim, Nv = numKV * headDim
            encodeQmmInProjDemuxRows(enc, w: cw, scales: cs, biases: cb, x: x,
                                     outQkv: sc.qOut, outZ: sc.kOut, outB: sc.vOut, outA: dummy,
                                     M: M, K: H, dims: (qkv: Nq, z: Nk, b: Nv, a: 0))
        } else {
            encodeQmmRows(enc, w: w.qW, scales: w.qS, biases: w.qB, x: x, out: sc.qOut, M: M, K: H, N: numHeads * qd2)
            encodeQmmRows(enc, w: w.kW, scales: w.kS, biases: w.kB, x: x, out: sc.kOut, M: M, K: H, N: numKV * headDim)
            encodeQmmRows(enc, w: w.vW, scales: w.vS, biases: w.vB, x: x, out: sc.vOut, M: M, K: H, N: numKV * headDim)
        }
        // ②③④ qk-norm + RoPE + cache scatter / A2+A3(fuseATTN): ONE dispatch each(−4 dispatches per attn layer)
        if RawFusedForward.fuseATTN && ensureWave3Pipelines() {
            // A2: extract_q + rmsnorm_q + rope_q → qRot (ONE dispatch, bit-exact)
            encodeAttnQPrepRows(enc, qOut: sc.qOut, qNorm: w.qNorm, qRot: sc.qRot,
                                qd2: qd2, headDim: headDim, ropeDim: ropeDim, base: ropeBase,
                                startOffset: baseLen, numHeads: numHeads, M: M, eps: eps)
            // A3: rmsnorm_k + rope_k + write_kv_k → kRot + kCache (ONE dispatch, bit-exact)
            encodeAttnKPrepRows(enc, kOut: sc.kOut, kNorm: w.kNorm, kRot: sc.kRot, kCache: kv.kCache,
                                headDim: headDim, ropeDim: ropeDim, base: ropeBase,
                                startOffset: baseLen, numKV: numKV, maxLen: kv.maxLen, M: M, eps: eps)
        } else {
            // ② queries 抽出(純コピー)→ qk-norm
            encodeExtractQ(enc, qOut: sc.qOut, q: sc.qX, headDim: headDim, qd2: qd2, total: M * numHeads * headDim)
            encodeRmsNormRows(enc, x: sc.qX, w: w.qNorm, out: sc.qN, rows: M * numHeads, D: headDim, eps: eps)
            encodeRmsNormRows(enc, x: sc.kOut, w: w.kNorm, out: sc.kN, rows: M * numKV, D: headDim, eps: eps)
            // ③ RoPE(行 m の位置 = baseLen + m)
            encodeRopeRows(enc, x: sc.qN, out: sc.qRot, headDim: headDim, ropeDim: ropeDim, base: ropeBase,
                           startOffset: baseLen, M: M, numHeads: numHeads)
            encodeRopeRows(enc, x: sc.kN, out: sc.kRot, headDim: headDim, ropeDim: ropeDim, base: ropeBase,
                           startOffset: baseLen, M: M, numHeads: numKV)
            // ④ cache 散布(post-RoPE k)
            encodeWriteKVRows(enc, src: sc.kRot, cache: kv.kCache, KV: numKV, D: headDim, maxLen: kv.maxLen, pos: baseLen, M: M)
        }
        // v-cache 散布(raw v, always unfused)
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

    // ── Wave 3: attn + shared-expert fusion atom stubs (notes/08) ──────────────
    //
    // ── Wave 3: attn + shared-expert fusion atom implementations (notes/08) ───────
    //
    // A4 (sdpa+sigmoid_mul epilogue) is a stretch atom gated only by G2; it has
    // no unit stub here.

    /// A1: QKV demux — fuses the 3 in-proj dispatches (q/k/v qmm4) into 1 dispatch.
    /// Demux N-axis boundaries (all multiples of 8):
    ///   q: numHeads×qd2 = 16×512 = 8192 (8192/8 = 1024 ✓)
    ///   k: numKV×headDim = 2×256 = 512   (512/8  = 64  ✓)
    ///   v: numKV×headDim = 512            (same         ✓)
    /// Returns (qOut[M,Nq], kOut[M,Nk], vOut[M,Nv]) — bit-identical to 3 × qmmRows.
    public static func attnQkvDemux(_ x: MLXArray,
                                     qW: MLXArray, qS: MLXArray, qB: MLXArray,
                                     kW: MLXArray, kS: MLXArray, kB: MLXArray,
                                     vW: MLXArray, vS: MLXArray, vB: MLXArray,
                                     M: Int, H: Int,
                                     numHeads: Int = 16, numKV: Int = 2, headDim: Int = 256)
        -> (qOut: MLXArray, kOut: MLXArray, vOut: MLXArray)? {
        guard let (device, queue) = RawMetalForward.ensure(), ensureRowsAuxPipelines() else { return nil }
        let qd2 = 2 * headDim
        let Nq = numHeads * qd2, Nk = numKV * headDim, Nv = numKV * headDim
        // Concatenate q/k/v weights for the demux kernel (same logic as prepareAttnLayerBufs)
        let catW = MLX.concatenated([qW, kW, vW], axis: 0)
        let catS = MLX.concatenated([qS.asType(.float16), kS.asType(.float16), vS.asType(.float16)], axis: 0)
        let catB = MLX.concatenated([qB.asType(.float16), kB.asType(.float16), vB.asType(.float16)], axis: 0)
        MLX.eval([catW, catS, catB])
        guard let bx    = RawMetalForward.mtlBuf(x.asType(.float16), device),
              let bw    = RawMetalForward.mtlBuf(catW, device),
              let bs    = RawMetalForward.mtlBuf(catS, device),
              let bb    = RawMetalForward.mtlBuf(catB, device),
              let qOutBuf = device.makeBuffer(length: M * Nq * 2, options: .storageModeShared),
              let kOutBuf = device.makeBuffer(length: M * Nk * 2, options: .storageModeShared),
              let vOutBuf = device.makeBuffer(length: M * Nv * 2, options: .storageModeShared),
              let dummy   = device.makeBuffer(length: 4, options: .storageModeShared) else { return nil }
        let cb = queue.makeCommandBuffer()!; let enc = cb.makeComputeCommandEncoder()!
        encodeQmmInProjDemuxRows(enc, w: bw, scales: bs, biases: bb, x: bx,
                                 outQkv: qOutBuf, outZ: kOutBuf, outB: vOutBuf, outA: dummy,
                                 M: M, K: H, dims: (qkv: Nq, z: Nk, b: Nv, a: 0))
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        let qp = qOutBuf.contents().bindMemory(to: Float16.self, capacity: M * Nq)
        let kp = kOutBuf.contents().bindMemory(to: Float16.self, capacity: M * Nk)
        let vp = vOutBuf.contents().bindMemory(to: Float16.self, capacity: M * Nv)
        return (MLXArray(Array(UnsafeBufferPointer(start: qp, count: M * Nq)), [M, Nq]),
                MLXArray(Array(UnsafeBufferPointer(start: kp, count: M * Nk)), [M, Nk]),
                MLXArray(Array(UnsafeBufferPointer(start: vp, count: M * Nv)), [M, Nv]))
    }

    /// A2: Q-prep — fuses extract_q(④) + rmsNorm_q(⑤) + rope_q(⑦) into 1 dispatch.
    /// Input qOut[M×numHeads, 2×headDim]; output qRot[M×numHeads, headDim].
    /// Extract: pure copy of lower headDim slice per head.
    /// RMSNorm: per-head over headDim, weight=qNorm.
    /// RoPE: startOffset carries the baseLen position; numHeads lanes share same M.
    public static func attnQPrepFused(_ qOut: MLXArray, qNorm: MLXArray,
                                       startOffset: Int, M: Int,
                                       numHeads: Int = 16, headDim: Int = 256,
                                       ropeDim: Int = 64, ropeBase: Float = 1e7,
                                       eps: Float = 1e-6)
        -> MLXArray? {
        guard let (device, queue) = RawMetalForward.ensure(), ensureWave3Pipelines() else { return nil }
        let qd2 = 2 * headDim
        let rows = M * numHeads
        guard let bqOut  = RawMetalForward.mtlBuf(qOut.asType(.float16), device),
              let bqNorm = RawMetalForward.mtlBuf(qNorm.asType(.float16), device),
              let qRotBuf = device.makeBuffer(length: rows * headDim * 2, options: .storageModeShared)
        else { return nil }
        let cb = queue.makeCommandBuffer()!; let enc = cb.makeComputeCommandEncoder()!
        encodeAttnQPrepRows(enc, qOut: bqOut, qNorm: bqNorm, qRot: qRotBuf,
                            qd2: qd2, headDim: headDim, ropeDim: ropeDim, base: ropeBase,
                            startOffset: startOffset, numHeads: numHeads, M: M, eps: eps)
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        let ptr = qRotBuf.contents().bindMemory(to: Float16.self, capacity: rows * headDim)
        return MLXArray(Array(UnsafeBufferPointer(start: ptr, count: rows * headDim)), [rows, headDim])
    }

    /// A3: K-prep — fuses rmsNorm_k(⑥) + rope_k(⑧) + write_kv_k(⑨) into 1 dispatch.
    /// Input kOut[M×numKV, headDim].
    /// Returns (kRot[M×numKV, headDim], kCache[KV, baseLen+M, headDim]).
    /// kCache is the filled portion of the [KV, maxLen, D] buffer after the scatter.
    /// The write_kv scatter is: src[m×KV+h, :] → cache[h, pos+m, :] (pure copy).
    ///
    /// A4 (sdpa+sigmoid_mul epilogue) gets NO unit stub — stretch atom gated only by G2.
    public static func attnKPrepFused(_ kOut: MLXArray, kNorm: MLXArray,
                                       kCacheInit: MLXArray,
                                       startOffset: Int, maxLen: Int, M: Int,
                                       numKV: Int = 2, headDim: Int = 256,
                                       ropeDim: Int = 64, ropeBase: Float = 1e7,
                                       eps: Float = 1e-6)
        -> (kRot: MLXArray, kCache: MLXArray)? {
        guard let (device, queue) = RawMetalForward.ensure(), ensureWave3Pipelines() else { return nil }
        let rows = M * numKV
        let baseLen = startOffset  // positions [0..baseLen) are pre-filled; kernel writes [baseLen..baseLen+M)
        guard let bkOut   = RawMetalForward.mtlBuf(kOut.asType(.float16), device),
              let bkNorm  = RawMetalForward.mtlBuf(kNorm.asType(.float16), device),
              let kRotBuf   = device.makeBuffer(length: rows * headDim * 2, options: .storageModeShared),
              let kCacheBuf = device.makeBuffer(length: numKV * maxLen * headDim * 2, options: .storageModeShared)
        else { return nil }
        // Preload kCacheInit [numKV, baseLen, headDim] into kCacheBuf (stride = maxLen)
        let kInitF = kCacheInit.asType(.float16).reshaped([-1]); kInitF.eval()
        let kInitArr = kInitF.asArray(Float16.self)
        let kcp = kCacheBuf.contents().bindMemory(to: Float16.self, capacity: numKV * maxLen * headDim)
        for h in 0..<numKV {
            for t in 0..<baseLen {
                for d in 0..<headDim {
                    kcp[h * maxLen * headDim + t * headDim + d] = kInitArr[(h * baseLen + t) * headDim + d]
                }
            }
        }
        let cb = queue.makeCommandBuffer()!; let enc = cb.makeComputeCommandEncoder()!
        encodeAttnKPrepRows(enc, kOut: bkOut, kNorm: bkNorm, kRot: kRotBuf, kCache: kCacheBuf,
                            headDim: headDim, ropeDim: ropeDim, base: ropeBase,
                            startOffset: startOffset, numKV: numKV, maxLen: maxLen, M: M, eps: eps)
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        let rp = kRotBuf.contents().bindMemory(to: Float16.self, capacity: rows * headDim)
        let kRot = MLXArray(Array(UnsafeBufferPointer(start: rp, count: rows * headDim)), [rows, headDim])
        // Read back kCache [numKV, baseLen+M, headDim] (the filled portion)
        let totalLen = baseLen + M
        var kCacheArr = [Float16](repeating: 0, count: numKV * totalLen * headDim)
        for h in 0..<numKV {
            for t in 0..<totalLen {
                for d in 0..<headDim {
                    kCacheArr[(h * totalLen + t) * headDim + d] = kcp[h * maxLen * headDim + t * headDim + d]
                }
            }
        }
        return (kRot, MLXArray(kCacheArr, [numKV, totalLen, headDim]))
    }

    /// S1: shG+shU+swiglu — fuses the 3 shared-expert gate/up/swiglu dispatches into 1.
    /// Plain-qmm variant (no gather): x[M,H] → shAct[M,I].
    /// M==1 branch only (register-pressure gate, same doctrine as fuseGU):
    ///   fuseSHEXPActive(M: m) = (fuseSHEXP && m == 1)
    public static func sharedGUSwigluFused(_ x: MLXArray,
                                            shGW: MLXArray, shGS: MLXArray, shGB: MLXArray,
                                            shUW: MLXArray, shUS: MLXArray, shUB: MLXArray,
                                            M: Int, H: Int, I: Int)
        -> MLXArray? {
        guard M == 1 else { return nil }  // S1 M==1 gate (register-pressure, fuseSHEXPActive doctrine)
        // gqmm4_swiglu_rows with Ktop=1, inds=[0]: bit-exact with qmmRows×2+swigluRaw (no gather)
        let inds = MLXArray([Int32(0)], [1]); inds.eval()
        return gatherQmmSwigluRows(x: x, inds: inds,
                                   wG: shGW, sG: shGS, bG: shGB,
                                   wU: shUW, sU: shUS, bU: shUB,
                                   M: M, Ktop: 1, K: H, N: I)
    }

    /// S2: sgl+final_combine — fuses qmm8(x→sgl N=8) + final_combine(y,sharedY,sgl→out).
    /// 8-bit dequant for sgl: group_size=64, K%512==0 required (K=H=2048 ✓, N=8 ✓).
    /// final_combine: out[i] = y[i] + sigmoid_h16(sgl[m*8]) * sharedY[i]  (stable f16 sigmoid).
    public static func sharedGateCombineFused(_ x: MLXArray, y: MLXArray, sharedY: MLXArray,
                                               sgW: MLXArray, sgS: MLXArray, sgB: MLXArray,
                                               M: Int, H: Int)
        -> MLXArray? {
        guard let (device, queue) = RawMetalForward.ensure(),
              RawMetalForward.compileQmm8(),
              ensureWave3Pipelines() else { return nil }
        guard H % 512 == 0 else { return nil }
        let sgSF = sgS.asType(.float16), sgBF = sgB.asType(.float16)
        guard let bsgW = RawMetalForward.mtlBuf(sgW, device),
              let bsgS = RawMetalForward.mtlBuf(sgSF, device),
              let bsgB = RawMetalForward.mtlBuf(sgBF, device),
              let bx   = RawMetalForward.mtlBuf(x.asType(.float16), device),
              let by   = RawMetalForward.mtlBuf(y.asType(.float16), device),
              let bsY  = RawMetalForward.mtlBuf(sharedY.asType(.float16), device),
              let outBuf = device.makeBuffer(length: M * H * 2, options: .storageModeShared)
        else { return nil }
        let cb = queue.makeCommandBuffer()!; let enc = cb.makeComputeCommandEncoder()!
        encodeSharedGateCombineRows(enc, sgW: bsgW, sgS: bsgS, sgB: bsgB,
                                    x: bx, y: by, sharedY: bsY, out: outBuf,
                                    K: H, H: H, M: M)
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        let ptr = outBuf.contents().bindMemory(to: Float16.self, capacity: M * H)
        return MLXArray(Array(UnsafeBufferPointer(start: ptr, count: M * H)), [M, H])
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
        // ── Wave 1 GDN fusion(notes/07 §3)事前連結 in-proj 重み。QWISP_FUSE_GDN=1 時のみ使用。
        //   prepareGdnLayerBufs で各 4-bit trio(qkv/z/b/a)を N 軸で連結し 1 本の qmm4_rows で撃つ。
        //   gs=64 行独立性で bit-exact by construction。flag-off 時は nil(既存 4 dispatch 経路不変)。
        let catInProjW: MTLBuffer?, catInProjS: MTLBuffer?, catInProjB: MTLBuffer?
        let totalInProjN: Int        // convDim+valueDim+numVH+numVH(連結 N 軸幅)
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
        // ── Wave 1 GDN fusion: 連結 in-proj 4-bit trio(qkv‖z‖b‖a)を prepare 時に 1 回 build。
        //   gs=64 行独立性で連結は bit-exact by construction。flag-on 時のみ使用(nil なら flag-off 経路)。
        //   gdnInProjConcat は MLXArray 連結→retained に保持し noCopy buffer の寿命を守る。
        //   ★§6 追加修正: ~190MB 無条件確保は 8GB tier に有害 → fuseGDN || QWISP_PROF_AB 時のみ build。
        let profAB = ProcessInfo.processInfo.environment["QWISP_PROF_AB"] == "1"
        var catWBuf: MTLBuffer? = nil, catSBuf: MTLBuffer? = nil, catBBuf: MTLBuffer? = nil
        var totalInProjN = 0
        if RawFusedForward.fuseGDN || profAB {
            if let cat = gdnInProjConcat(qkvW: w.qkvWq, qkvS: w.qkvSc, qkvB: w.qkvBi,
                                           zW: w.zWq,   zS: w.zSc,   zB: w.zBi,
                                           bW: w.bWq,   bS: w.bSc,   bB: w.bBi,
                                           aW: w.aWq,   aS: w.aSc,   aB: w.aBi) {
                keep.append(contentsOf: [cat.w, cat.s, cat.b])
                totalInProjN = cat.w.shape[0]
                catWBuf = RawMetalForward.mtlBuf(cat.w, device)
                catSBuf = RawMetalForward.mtlBuf(cat.s.asType(.float16), device)
                catBBuf = RawMetalForward.mtlBuf(cat.b.asType(.float16), device)
            }
        }

        return GdnLayerBufs(qkvW: qkv.0, qkvS: qkv.1, qkvB: qkv.2, zW: z.0, zS: z.1, zB: z.2,
                            bW: b.0, bS: b.1, bB: b.2, aW: a.0, aS: a.1, aB: a.2,
                            outW: o.0, outS: o.1, outB: o.2,
                            conv1dW: cw, normWeight: nw, promoteRMS: promote, aLog: al, dtBias: dt, onesDk: od,
                            catInProjW: catWBuf, catInProjS: catSBuf, catInProjB: catBBuf,
                            totalInProjN: totalInProjN,
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
        // F1 re-design (notes/07 §6): fuseGDN 時は concat 重み [totalN, H] に対し
        // qmm4_inproj_demux_rows 1 dispatch で qkv/z/bP/aP の 4 buffer に書き分ける(−3/層)。
        // dot 演算は既存 qmm4_rows と同一順 = bit-exact by construction。catInProjW は
        // prepareGdnLayerBufs が fuseGDN||QWISP_PROF_AB 時のみ build(nil なら flag-off 4 dispatch へ)。
        if RawFusedForward.fuseGDN, let cw = w.catInProjW, let cs = w.catInProjS, let cb = w.catInProjB,
           w.totalInProjN == convDim + valueDim + numVHeads + numVHeads {
            encodeQmmInProjDemuxRows(enc, w: cw, scales: cs, biases: cb, x: x,
                                     outQkv: sc.qkv, outZ: sc.z, outB: sc.bP, outA: sc.aP,
                                     M: M, K: H,
                                     dims: (qkv: convDim, z: valueDim, b: numVHeads, a: numVHeads))
        } else {
            encodeQmmRows(enc, w: w.qkvW, scales: w.qkvS, biases: w.qkvB, x: x, out: sc.qkv, M: M, K: H, N: convDim)
            encodeQmmRows(enc, w: w.zW, scales: w.zS, biases: w.zB, x: x, out: sc.z, M: M, K: H, N: valueDim)
            encodeQmmRows(enc, w: w.bW, scales: w.bS, biases: w.bB, x: x, out: sc.bP, M: M, K: H, N: numVHeads)
            encodeQmmRows(enc, w: w.aW, scales: w.aS, biases: w.aB, x: x, out: sc.aP, M: M, K: H, N: numVHeads)
        }
        // ② conv(hist 直読み)→ hist shift 更新(ping-pong: In を読み Out へ)
        // F3(notes/07 §3 Wave 1): fuseGDN 時に 2→1 dispatch 融合。conv_shift_fused_rows は
        // conv1d_silu_hist_rows + shift_conv_rows と bit-exact by construction(RAWTESTS test35)。
        // ensureGdnPipelines が init 時に _convShiftFusedRowsPipeline を保証する。
        if RawFusedForward.fuseGDN {
            encodeGdnFusionConvShift(enc, hist: cache.convHist, qkv: sc.qkv, w: w.conv1dW,
                                     convOut: sc.convOut, histOut: cache.convHistOut,
                                     M: M, K: convKernel, C: convDim)
        } else {
            encodeConvHistRows(enc, hist: cache.convHist, qkv: sc.qkv, w: w.conv1dW, out: sc.convOut,
                               K: convKernel, C: convDim, M: M)
            encodeShiftConvRows(enc, histOut: cache.convHistOut, histIn: cache.convHist, qkv: sc.qkv,
                                K: convKernel, C: convDim, M: M)
        }
        // ③-⑤ F2 (notes/07 §3 Wave 2): fuseGDN 時は gdn_prep_rows 1 dispatch で
        // split q/k/v + rmsnorm qn/kn + scale_mul q/k + compute_g_beta を融合(−7/層)。
        // scale_mul semantics: (half)s * x[i] — 既存 encodeScaleMul kernel と bit-exact。
        let invScale = Float(pow(Double(headKDim), -0.5))
        if RawFusedForward.fuseGDN, _gdnPrepRowsPipeline != nil {
            encodeGdnPrepRows(enc, convOut: sc.convOut, aP: sc.aP, bP: sc.bP,
                              aLog: w.aLog, dtBias: w.dtBias,
                              qn: sc.qn, kn: sc.kn, v: sc.v1, g: sc.g, beta: sc.beta,
                              M: M, numKH: numKHeads, headKD: headKDim, numVH: numVHeads,
                              keyDim: keyDim, valDim: valueDim, eps: eps,
                              qScale: invScale * invScale, kScale: invScale)
        } else {
            // ③ split q/k/v(純コピー)
            encodeSliceRows(enc, input: sc.convOut, out: sc.q1, off: 0, W: keyDim, stride: convDim, M: M)
            encodeSliceRows(enc, input: sc.convOut, out: sc.k1, off: keyDim, W: keyDim, stride: convDim, M: M)
            encodeSliceRows(enc, input: sc.convOut, out: sc.v1, off: 2 * keyDim, W: valueDim, stride: convDim, M: M)
            // ④ qk-norm(no-weight)+ scalar scale
            encodeRmsNormRows(enc, x: sc.q1, w: w.onesDk, out: sc.qn, rows: M * numKHeads, D: headKDim, eps: eps)
            encodeRmsNormRows(enc, x: sc.k1, w: w.onesDk, out: sc.kn, rows: M * numKHeads, D: headKDim, eps: eps)
            encodeScaleMul(enc, x: sc.qn, s: invScale * invScale, total: M * keyDim)
            encodeScaleMul(enc, x: sc.kn, s: invScale, total: M * keyDim)
            // ⑤ g/β → recurrence(in-kernel T=M 逐次)
            encodeComputeGBetaRows(enc, a: sc.aP, b: sc.bP, aLog: w.aLog, dtBias: w.dtBias,
                                   g: sc.g, beta: sc.beta, Hv: numVHeads, M: M)
        }
        encodeGatedDeltaStepRows(enc, q: sc.qn, k: sc.kn, v: sc.v1, g: sc.g, beta: sc.beta,
                                 stateIn: cache.state, stateOut: cache.stateOut, y: sc.coreOut,
                                 T: M, B: 1, Hv: numVHeads, Dv: headVDim)
        // ⑥ RMSNormGated → silu(z)·normed
        // F4 re-design (notes/07 §6): fuseGDN 時は gdn_norm_gate_rows 1 dispatch で
        // per-head rmsnorm(coreOut) + silu(z)⊙normed を融合(−1/層)。reduction tree は既存
        // rmsnorm kernel と同一(N_READS=4 + simd_sum 二段 + precise::rsqrt)= bit-exact。
        if RawFusedForward.fuseGDN, _gdnNormGateRowsPipeline != nil {
            encodeGdnNormGateRows(enc, coreOut: sc.coreOut, z: sc.z, normWeight: w.normWeight,
                                  outV: sc.outV, M: M, Hv: numVHeads, Dv: headVDim,
                                  eps: eps, promoteF32: w.promoteRMS)
        } else {
            encodeRmsNormRows(enc, x: sc.coreOut, w: w.normWeight, out: sc.normed,
                              rows: M * numVHeads, D: headVDim, eps: eps, promoteF32: w.promoteRMS)
            encodeGate(enc, z: sc.z, normed: sc.normed, outV: sc.outV, total: M * valueDim, promote: w.promoteRMS)
        }
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
        /// gate+up gather+swiglu 融合(1 dispatch)へ分岐。既定 ON。QWISP_FUSE_GU=0 で opt-out。
        /// flag-on でも演算順同一なので bit-exact(G2 gate)。flag-off で byte 不変(G3 gate)。
        nonisolated(unsafe) public static var fuseGU =
            ProcessInfo.processInfo.environment["QWISP_FUSE_GU"] != "0"

        /// GDN 層融合(notes/07 §3 Wave 1)へ分岐。既定 ON。QWISP_FUSE_GDN=0 で opt-out。
        /// flag-on でも各融合原子は既存 kernel 連鎖と bit-exact(G1 gate)なので OUT byte 不変(G2)。
        /// flag-off で encodeGdnLayerRows は既存経路のままで byte 不変(G3 gate)。
        nonisolated(unsafe) public static var fuseGDN =
            ProcessInfo.processInfo.environment["QWISP_FUSE_GDN"] != "0"

        /// attn 層融合(notes/08 §3)へ分岐。既定 ON。QWISP_FUSE_ATTN=0 で opt-out。
        /// flag-on でも各融合原子は既存 kernel 連鎖と bit-exact(G1 gate)なので OUT byte 不変(G2)。
        /// flag-off で encodeAttnLayerRows は既存経路のままで byte 不変(G3 gate)。
        /// 原子別 kill-switch のキャッシュ(per-encode の ProcessInfo.environment 読みは 18.8µs/回で
        /// chain 時 K×層 倍化し fusion 利得を食う — recon 2026-07-04 実測。必ず static で一度だけ読む)
        nonisolated(unsafe) public static let fuseS1Enabled =
            ProcessInfo.processInfo.environment["QWISP_FUSE_S1"] != "0"
        nonisolated(unsafe) public static let fuseA1Enabled =
            ProcessInfo.processInfo.environment["QWISP_FUSE_A1"] != "0"
        nonisolated(unsafe) public static var fuseATTN =
            ProcessInfo.processInfo.environment["QWISP_FUSE_ATTN"] != "0"

        /// MoE shared expert 融合(notes/08 §3)へ分岐。既定 ON。QWISP_FUSE_SHEXP=0 で opt-out。
        /// flag-on でも各融合原子は既存 kernel 連鎖と bit-exact(G1 gate)なので OUT byte 不変(G2)。
        /// flag-off で encodeMoESharedRows は既存経路のままで byte 不変(G3 gate)。
        nonisolated(unsafe) public static var fuseSHEXP =
            ProcessInfo.processInfo.environment["QWISP_FUSE_SHEXP"] != "0"

        /// MoE combine_rows→S2 fold (notes/10 §3). 既定 ON。QWISP_FUSE_MOE2=0 で opt-out。
        /// fold-ON(MOE2=1 かつ fuseSHEXP かつ M==1): S2 が combine をインライン実行し encodeCombineRows dispatch を skip。
        /// fold-OFF: 現行経路 combine_rows 別 dispatch + S2 が sc.y 読み。byte 不変。
        /// ★既定 OFF(opt-in): bit-exact だが利得 sub-noise(paired A/B M=1 +68µs=+0.5%、wall 計測不能)。
        /// stage-1 recon の +3-4% は fuseSHEXP proxy の過大評価(proxy は compute も畳んでいた)。QWISP_FUSE_MOE2=1 で有効。
        nonisolated(unsafe) public static var fuseMOE2Enabled =
            ProcessInfo.processInfo.environment["QWISP_FUSE_MOE2"] == "1"

        /// Phase II-a chain default length (QWISP_CHAIN_K unset). Single production seam
        /// for the chained GPU token-feedback decode default. RawSpecRunner resolves the
        /// chain length as Tell.envInt("QWISP_CHAIN_K", RawFusedForward.chainKDefault),
        /// so this constant is the sole source of the unset default. QWISP_CHAIN_K=0
        /// still disables (envInt returns 0).
        public static let chainKDefault = 8

        /// M-branch predicate for the fuseGU path.
        /// Contract: fuseGUActive(M) == (fuseGU && M == 1)
        ///   — the fused gather+swiglu kernel is activated for M=1 only (register pressure guard).
        ///   verify batches (M>1) use the unfused 3-kernel path (measured regression at M=8).
        public static func fuseGUActive(M: Int) -> Bool? {
            return fuseGU && M == 1
        }

        /// M-branch predicate for the fuseSHEXP path.
        /// Contract: fuseSHEXPActive(M) == (fuseSHEXP && M == 1)
        ///   — the fused shared-expert gate+up+swiglu kernel is activated for M=1 only
        ///   (register pressure guard, same doctrine as fuseGU).
        public static func fuseSHEXPActive(M: Int) -> Bool? {
            return fuseSHEXP && M == 1
        }

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
            // F5 (notes/07 §3 Wave 2): fuseGDN 時は gdn_resid_postnorm_rows 1 dispatch で
            // resid_add ⑳ + rmsnorm post ㉑ を融合(−1/層)。全層型(GDN/Attn)に適用。
            if RawFusedForward.fuseGDN, RawFusedVerify._gdnResidPostNormRowsPipeline != nil {
                RawFusedVerify.encodeGdnResidPostNormRows(enc, h: hBuf, r: mixerOut,
                                                          w: L.postLN, postNorm: postNorm,
                                                          M: M, H: H, eps: eps)
            } else {
                RawFusedVerify.encodeResidAdd(enc, h: hBuf, r: mixerOut, total: M * H)
                RawFusedVerify.encodeRmsNormRows(enc, x: hBuf, w: L.postLN, out: postNorm, rows: M, D: H, eps: eps)
            }
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
                LayerExpertCache.chunkTotal += chunks.count

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
                                                    M: M, I: L.I, H: H, Ktop: L.Ktop)
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
        }
        var head: HeadBufs? = nil

        /// head(embed/lm_head/final norm)を常駐 buffer 化して 1-CB step を有効化する。
        public func attachHead(embedW: MLXArray, embedS: MLXArray, embedB: MLXArray,
                               lmW: MLXArray, lmS: MLXArray, lmB: MLXArray,
                               fnW: MLXArray, vocab: Int) -> Bool {
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
                            fnW: fn, vocab: vocab, tokensIn: ti, logits: lg, tokensOut: to, retained: keep)
            return true
        }

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

        /// Phase II-a G1 gate: K-step chained greedy decode in a single GPU command buffer.
        /// Encodes K greedy steps using indirect embed (step k+1 reads GPU-side tokensOut from
        /// step k, no CPU round-trip). firstToken is the seed input for step 0.
        /// Returns K token ids [t_0, ..., t_{K-1}] and advances KV/GDN cache state identically
        /// to K sequential stepArgmax([t_i]) calls.
        ///
        /// STUB — implementation pending.
        /// NOTE: Delegation to forwardRows or stepArgmax in this stub is FORBIDDEN per §4-G1.
        public func chainedStepArgmax(_ firstToken: Int32, K: Int) -> [Int]? {
            guard let hd = head, K > 0 else { return nil }
            guard streamMode == .resident || streamMode == .bolt else { return nil }

            // Write seed token for step 0 (CPU → shared buffer; GPU reads it inside the CB).
            hd.tokensIn.contents().bindMemory(to: Int32.self, capacity: maxM)[0] = firstToken

            // Per-call chain output buffer (avoids the maxM size limit on hd.tokensOut —
            // the test uses maxM=4 with K=8, so hd.tokensOut cannot hold K int32s).
            guard let chainBuf = device.makeBuffer(
                length: K * MemoryLayout<Int32>.stride, options: .storageModeShared
            ) else { return nil }

            let cb = queue.makeCommandBuffer()!
            let enc = cb.makeComputeCommandEncoder()!

            let ep = RawFusedVerify._embedRowsPipeline!
            let ap = RawFusedVerify._argmaxRowsPipeline!

            for k in 0 ..< K {
                // ── Embed 1 token ──────────────────────────────────────────────────
                // Step 0: reads firstToken from hd.tokensIn (CPU-written before CB encode).
                // Step k>0: indirect embed — reads GPU-written chainBuf[k-1] (no CPU round-trip).
                enc.setComputePipelineState(ep)
                enc.setBuffer(hd.embedW, offset: 0, index: 0)
                enc.setBuffer(hd.embedS, offset: 0, index: 1)
                enc.setBuffer(hd.embedB, offset: 0, index: 2)
                if k == 0 {
                    enc.setBuffer(hd.tokensIn, offset: 0, index: 3)
                } else {
                    enc.setBuffer(chainBuf, offset: (k - 1) * MemoryLayout<Int32>.stride, index: 3)
                }
                enc.setBuffer(hBuf, offset: 0, index: 4)
                var hh = UInt32(H), tt = UInt32(H)   // M=1 → total = H
                enc.setBytes(&hh, length: 4, index: 5)
                enc.setBytes(&tt, length: 4, index: 6)
                let tgs = min(ep.maxTotalThreadsPerThreadgroup, 256)
                enc.dispatchThreads(MTLSize(width: H, height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: tgs, height: 1, depth: 1))

                // ── All layers M=1 ─────────────────────────────────────────────────
                // encodeLayer/encodeLayerBolt internally call gc.swapState() and kv.len += 1
                // at encode time (CPU pointer swap / counter bump), giving correct ping-pong
                // and KV position for each step. The encode captures MTLBuffer objects (not
                // variable bindings), so sequential encodes within this single CB are independent
                // and memory-coherent on GPU (Metal guarantees program-order dispatch + barrier).
                if streamMode == .bolt {
                    for (li, L) in layers.enumerated() { encodeLayerBolt(enc, L, M: 1, li: li) }
                } else {
                    for L in layers { encodeLayer(enc, L, M: 1) }
                }

                // ── Final ops: fnorm + lm_head + argmax → chainBuf[k] ─────────────
                RawFusedVerify.encodeRmsNormRows(enc, x: hBuf, w: hd.fnW, out: normed,
                                                 rows: 1, D: H, eps: eps)
                if RawFusedForward.lmHeadQmv {
                    RawFusedVerify.encodeQmmRows(enc, w: hd.lmW, scales: hd.lmS, biases: hd.lmB,
                                                 x: normed, out: hd.logits, M: 1, K: H, N: hd.vocab)
                } else {
                    let qp = RawMetalForward._qmm4TiledPipeline!
                    enc.setComputePipelineState(qp)
                    enc.setBuffer(hd.lmW, offset: 0, index: 0); enc.setBuffer(hd.lmS, offset: 0, index: 1)
                    enc.setBuffer(hd.lmB, offset: 0, index: 2); enc.setBuffer(normed, offset: 0, index: 3)
                    enc.setBuffer(hd.logits, offset: 0, index: 4)
                    var kk = Int32(H), nn = Int32(hd.vocab), mm = Int32(1)
                    enc.setBytes(&kk, length: 4, index: 5); enc.setBytes(&nn, length: 4, index: 6)
                    enc.setBytes(&mm, length: 4, index: 7)
                    enc.dispatchThreadgroups(MTLSize(width: hd.vocab, height: 1, depth: 1),
                                             threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                }
                enc.setComputePipelineState(ap)
                enc.setBuffer(hd.logits, offset: 0, index: 0)
                enc.setBuffer(chainBuf, offset: k * MemoryLayout<Int32>.stride, index: 1)
                var vv = UInt32(hd.vocab); enc.setBytes(&vv, length: 4, index: 2)
                enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            }

            enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
            RawFusedForward.profLastGPUMs = (cb.gpuEndTime - cb.gpuStartTime) * 1000.0

            let ptr = chainBuf.contents().bindMemory(to: Int32.self, capacity: K)
            return (0 ..< K).map { Int(ptr[$0]) }
        }
    }

    // ── G1 gate: gqmm4_swiglu_rows stub (notes/06-fusion-poc-spec.md §4) ──────────────
    //
    // Implementer (GLM-5.2): add the encode-level dispatch below, matching encodeGatherQmmRows
    // style exactly (same buffer indices, same lhsPer=false for g+u shared-x semantics).
    // Signature contract (copy verbatim, then fill body):
    //
    //   static func encodeGatherQmmSwigluRows(_ enc: MTLComputeCommandEncoder,
    //                                         wG: MTLBuffer, sG: MTLBuffer, bG: MTLBuffer,
    //                                         wU: MTLBuffer, sU: MTLBuffer, bU: MTLBuffer,
    //                                         x: MTLBuffer, inds: MTLBuffer, out: MTLBuffer,
    //                                         M: Int, Ktop: Int, K: Int, N: Int,
    //                                         xByteOffset: Int = 0, indsOffset: Int = 0, outByteOffset: Int = 0)
    //
    // Kernel: gqmm4_swiglu_rows — grid (1, N/8, M·Ktop), threads (32,2,1). Each threadgroup
    // computes 8 output cols of one mk row for gate and up (shared x load via ld16), applies
    // swiglu in-register, writes h[mk*N + out_col] directly (no g/u intermediate buffers).
    // Operand order must reproduce the existing 3-kernel chain (gqmm4_rows×2 + swigluRaw) bit-exactly:
    //   gv = (half)simd_sum(g_acc),  uv = (half)simd_sum(u_acc)   [cast to half before swiglu]
    //   y = (gv*sigmoid(gv))*uv                                    [stable sigmoid as in existing swiglu]
    // (This matches the existing gqmm4_swiglu kernel at RawMetalForward.swift ~1298-1337.)
    //
    // Note: integration-level flag QWISP_FUSE_GU (encodeMoEGatherRowsRange branch) is gated
    // separately by G2 real-weight identity and does NOT depend on these unit tests.

    // ── Wave 1 GDN fusion stubs (notes/07 §3) — FUSE_GDN STUB — implementation pending ──

    /// F1: N-axis concatenation of the 4 GDN in-proj 4-bit weight triples.
    /// qkv: [convDim, H/2] 4-bit quantised — projects hidden → conv input (qkv for delta-net)
    /// z:   [valueDim, H/2] 4-bit — projects hidden → z (gate for output)
    /// b:   [numVHeads, H/2] 4-bit — projects hidden → β sigmoid input
    /// a:   [numVHeads, H/2] 4-bit — projects hidden → α log input
    /// Returns (w, s, b) where each is the N-axis (axis-0) concatenation of the four respective
    /// components: w.shape = [convDim+valueDim+numVHeads+numVHeads, H/2], and similarly for s, b.
    /// gs=64 row-independence guarantees that each 64-row scale/bias group stays self-contained
    /// across the concat boundary, making a single qmmRows call bit-exact by construction.
    public static func gdnInProjConcat(
        qkvW: MLXArray, qkvS: MLXArray, qkvB: MLXArray,
        zW: MLXArray,   zS: MLXArray,   zB: MLXArray,
        bW: MLXArray,   bS: MLXArray,   bB: MLXArray,
        aW: MLXArray,   aS: MLXArray,   aB: MLXArray
    ) -> (w: MLXArray, s: MLXArray, b: MLXArray)? {
        // N 軸(axis 0)連結: qkv ‖ z ‖ b ‖ a。gs=64 の行独立性で各 scale/bias group は
        // 連結境界を跨がない → 1 本の qmm4_rows で 4 回個別 qmm と bit 一致(by construction)。
        let catW = MLX.concatenated([qkvW, zW, bW, aW], axis: 0)
        let catS = MLX.concatenated([qkvS, zS, bS, aS], axis: 0)
        let catB = MLX.concatenated([qkvB, zB, bB, aB], axis: 0)
        MLX.eval([catW, catS, catB])
        return (catW, catS, catB)
    }

    /// F3: Fused ⑥conv1d_silu_hist_rows + ⑦shift_conv_rows in one command buffer.
    /// histIn: [K-1, C] f16 — current conv history (read-only input).
    /// qkv:    [M, C]   f16 — current M-row in-proj output.
    /// w:      [C, K]   f16 — conv1d kernel weights (promoted to f32 inside kernel).
    /// Returns (convOut: [M, C] f16, histOut: [K-1, C] f16) where:
    ///   convOut = per-row silu(sum_k w[c,k] · (hist‖qkv)[m+k, c]) — same as conv1d_silu_hist_rows
    ///   histOut = (hist‖qkv)[M .. M+K-2, :] — same as shift_conv_rows (pure data movement)
    /// M∈{1,8} must be bit-exact with the two-kernel reference.
    public static func gdnConvShiftFused(
        histIn: MLXArray, qkv: MLXArray, w: MLXArray,
        M: Int, K: Int, C: Int
    ) -> (convOut: MLXArray, histOut: MLXArray)? {
        guard let (device, queue) = RawMetalForward.ensure(), ensureRowsAuxPipelines() else { return nil }
        // fused pipeline を compile（既存 rows aux の後に呼ばれる; rms pipeline は gateRaw 側で要る）
        if _convShiftFusedRowsPipeline == nil {
            do { try ensureGdnFusionPipelines(device) } catch { print("[gdn-fusion] compile: \(error)"); return nil }
        }
        guard _convShiftFusedRowsPipeline != nil else { return nil }
        guard let bHist = RawMetalForward.mtlBuf(histIn.asType(.float16), device),
              let bqkv = RawMetalForward.mtlBuf(qkv.asType(.float16), device),
              let bw = RawMetalForward.mtlBuf(w.asType(.float32), device),
              let bConv = device.makeBuffer(length: M * C * 2, options: .storageModeShared),
              let bHistOut = device.makeBuffer(length: (K - 1) * C * 2, options: .storageModeShared) else { return nil }
        let cb = queue.makeCommandBuffer()!; let enc = cb.makeComputeCommandEncoder()!
        encodeGdnFusionConvShift(enc, hist: bHist, qkv: bqkv, w: bw, convOut: bConv, histOut: bHistOut, M: M, K: K, C: C)
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        let pc = bConv.contents().bindMemory(to: Float16.self, capacity: M * C)
        let ph = bHistOut.contents().bindMemory(to: Float16.self, capacity: (K - 1) * C)
        return (MLXArray(Array(UnsafeBufferPointer(start: pc, count: M * C)), [M, C]),
                MLXArray(Array(UnsafeBufferPointer(start: ph, count: (K - 1) * C)), [K - 1, C]))
    }

    /// F4: Fused ⑰per-head rmsnorm(coreOut) + ⑱gate(silu(z)⊙normed) in one command buffer.
    /// coreOut:    [M*Hv, Dv] f16 — recurrent core output (M*numVHeads rows, each headVDim wide).
    /// z:          [M, Hv*Dv] f16 — z in-proj output used as the silu gate.
    /// normWeight: [Dv] f16 or f32 — per-head rmsnorm weight; dtype determines promoteF32.
    /// promoteF32: when true, rmsnorm intermediate is f32 (matching _rmsPipelineF32 path); must
    ///             equal (normWeight.dtype == .float32) to stay bit-exact with gateRaw.
    /// Returns outV: [M, Hv*Dv] f16 — bit-identical to rmsNormRows then gateRaw chain.
    /// F4: per-head rmsnorm(coreOut) + silu(z)· 1 CB 連係化 = bit-exact by construction、dispatch 数を 1 削減。
    /// rmsnorm reduction tree は既存 _rmsPipeline[_F32] そのまま、gate は既存 _gate[_gate16] そのまま。
    /// promoteF32: normWeight.dtype==.float32 経路。outV: [M, Hv*Dv] f16。
    public static func gdnNormGateFused(
        coreOut: MLXArray, z: MLXArray, normWeight: MLXArray,
        M: Int, Hv: Int, Dv: Int,
        eps: Float = 1e-6, promoteF32: Bool = false
    ) -> MLXArray? {
        let valueDim = Hv * Dv
        guard let normed = RawMetalForward.rmsNormRows(coreOut, normWeight, M: M * Hv, eps: eps, D: Dv, promoteF32: promoteF32)
        else { return nil }
        guard let outV = RawFusedVerify.gateRaw(z, normed, promote: promoteF32, total: M * valueDim)
        else { return nil }
        return outV.reshaped([M, valueDim])
    }

    // ── Wave 1 GDN fusion re-design stubs (notes/07 §6) — FUSE_GDN2 STUB — implementation pending ──

    /// F1 re-design (demux type, §6 Wave1 review): ONE qmm4 dispatch over the concatenated
    /// weights [totalN, K] 4-bit; the kernel demuxes output columns into FOUR separate output
    /// buffers (qkv/z/bP/aP) according to dim boundaries. Downstream kernels read the separate
    /// buffers unchanged — no dispatch after this one.
    /// dims: (qkv, z, b, a) each must be multiples of 8 (threadgroup column alignment).
    /// dot arithmetic is identical to existing qmm4_rows → bit-exact by construction.
    /// Returns nil until implemented (stub gates the test RED).
    public static func gdnInProjDemux(
        x: MLXArray,
        catW: MLXArray, catS: MLXArray, catB: MLXArray,
        M: Int, K: Int,
        dims: (qkv: Int, z: Int, b: Int, a: Int)
    ) -> (qkv: MLXArray, z: MLXArray, bP: MLXArray, aP: MLXArray)? {
        guard let (device, queue) = RawMetalForward.ensure(),
              ensureRowsAuxPipelines() else { return nil }
        // warm the demux pipeline (compiled inside ensureGdnFusionPipelines via ensureRowsAuxPipelines)
        guard _qmmInProjDemuxRowsPipeline != nil else { return nil }
        guard dims.qkv % 8 == 0, dims.z % 8 == 0, dims.b % 8 == 0, dims.a % 8 == 0,
              K % 512 == 0 else { print("[gdn-demux] 非fast / 非8整列"); return nil }
        guard let bx = RawMetalForward.mtlBuf(x.asType(.float16), device),
              let bw = RawMetalForward.mtlBuf(catW, device),
              let bs = RawMetalForward.mtlBuf(catS.asType(.float16), device),
              let bb = RawMetalForward.mtlBuf(catB.asType(.float16), device),
              let bQkv = device.makeBuffer(length: M * dims.qkv * 2, options: .storageModeShared),
              let bZ   = device.makeBuffer(length: M * dims.z   * 2, options: .storageModeShared),
              let bB   = device.makeBuffer(length: M * dims.b   * 2, options: .storageModeShared),
              let bA   = device.makeBuffer(length: M * dims.a   * 2, options: .storageModeShared) else { return nil }
        let cb = queue.makeCommandBuffer()!; let enc = cb.makeComputeCommandEncoder()!
        encodeQmmInProjDemuxRows(enc, w: bw, scales: bs, biases: bb, x: bx,
                                 outQkv: bQkv, outZ: bZ, outB: bB, outA: bA,
                                 M: M, K: K, dims: dims)
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        let pq = bQkv.contents().bindMemory(to: Float16.self, capacity: M * dims.qkv)
        let pz = bZ.contents().bindMemory(to: Float16.self, capacity: M * dims.z)
        let pb = bB.contents().bindMemory(to: Float16.self, capacity: M * dims.b)
        let pa = bA.contents().bindMemory(to: Float16.self, capacity: M * dims.a)
        return (MLXArray(Array(UnsafeBufferPointer(start: pq, count: M * dims.qkv)), [M, dims.qkv]),
                MLXArray(Array(UnsafeBufferPointer(start: pz, count: M * dims.z)), [M, dims.z]),
                MLXArray(Array(UnsafeBufferPointer(start: pb, count: M * dims.b)), [M, dims.b]),
                MLXArray(Array(UnsafeBufferPointer(start: pa, count: M * dims.a)), [M, dims.a]))
    }

    /// F4 re-design (true fused single-dispatch kernel, §6 Wave1 review):
    /// ONE kernel, 1 threadgroup per (m, head): rmsnorm reduction (identical to existing
    /// rmsnorm kernel incl. N_READS=4 + simd_sum two-stage + precise::rsqrt) then
    /// silu(z)⊙ applied in registers. f16 + promoteF32 variants.
    /// This is distinct from gdnNormGateFused which merely chains two existing dispatches.
    /// Returns outV: [M, Hv*Dv] f16.
    /// Returns nil until implemented (stub gates the test RED).
    public static func gdnNormGateRows(
        coreOut: MLXArray, z: MLXArray, normWeight: MLXArray,
        M: Int, Hv: Int, Dv: Int,
        eps: Float, promoteF32: Bool
    ) -> MLXArray? {
        let valueDim = Hv * Dv
        guard let (device, queue) = RawMetalForward.ensure(),
              ensureRowsAuxPipelines() else { return nil }
        guard _gdnNormGateRowsPipeline != nil else { return nil }
        let wType: DType = promoteF32 ? .float32 : .float16
        guard let bx = RawMetalForward.mtlBuf(coreOut.asType(.float16), device),
              let bz = RawMetalForward.mtlBuf(z.asType(.float16), device),
              let bw = RawMetalForward.mtlBuf(normWeight.asType(wType), device),
              let bOut = device.makeBuffer(length: M * valueDim * 2, options: .storageModeShared) else { return nil }
        let cb = queue.makeCommandBuffer()!; let enc = cb.makeComputeCommandEncoder()!
        encodeGdnNormGateRows(enc, coreOut: bx, z: bz, normWeight: bw, outV: bOut,
                              M: M, Hv: Hv, Dv: Dv, eps: eps, promoteF32: promoteF32)
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        let ptr = bOut.contents().bindMemory(to: Float16.self, capacity: M * valueDim)
        return MLXArray(Array(UnsafeBufferPointer(start: ptr, count: M * valueDim)), [M, valueDim])
    }

    // ── Wave 2 GDN fusion (notes/07 §3 Wave 2) — real single-kernel implementations ──

    /// F2: gdn_prep fused — ⑧slice q ⑨slice k ⑩slice v (from convOut) ⑪rmsnorm qn (ones weight,
    /// per-head over headKDim) ⑫rmsnorm kn (ones) ⑬scale_mul q (invScale²) ⑭scale_mul k (invScale)
    /// ⑮compute_g_beta (from aP,bP with aLog,dtBias) — 8 separate kernels → 1 fused dispatch.
    /// Uses gdn_prep_rows: a TRUE single Metal kernel dispatch (not a chain).
    /// scale_mul semantics: (half)s * (half)(x*inv_mean) — s rounded to f16 FIRST, matching
    /// the production scale_mul kernel x[i]=(half)s*x[i] and the test oracle's scaleMulKernel.
    public static func gdnPrepFused(
        convOut: MLXArray, aP: MLXArray, bP: MLXArray, aLog: MLXArray, dtBias: MLXArray,
        M: Int, keyDim: Int, valueDim: Int, numKHeads: Int, headKDim: Int,
        numVHeads: Int, invScale: Float, eps: Float = 1e-6
    ) -> (qn: MLXArray, kn: MLXArray, v: MLXArray, g: MLXArray, beta: MLXArray)? {
        guard let (device, queue) = RawMetalForward.ensure(),
              ensureGdnPipelines(), _gdnPrepRowsPipeline != nil else { return nil }
        guard let bConv = RawMetalForward.mtlBuf(convOut.asType(.float16), device),
              let bA    = RawMetalForward.mtlBuf(aP.asType(.float16), device),
              let bB    = RawMetalForward.mtlBuf(bP.asType(.float16), device),
              let bALog = RawMetalForward.mtlBuf(aLog.asType(.float32), device),
              let bDt   = RawMetalForward.mtlBuf(dtBias.asType(.float32), device) else { return nil }
        guard let bQn   = device.makeBuffer(length: M * keyDim * 2, options: .storageModeShared),
              let bKn   = device.makeBuffer(length: M * keyDim * 2, options: .storageModeShared),
              let bV    = device.makeBuffer(length: M * valueDim * 2, options: .storageModeShared),
              let bG    = device.makeBuffer(length: M * numVHeads * 4, options: .storageModeShared),
              let bBeta = device.makeBuffer(length: M * numVHeads * 4, options: .storageModeShared) else { return nil }
        let cb = queue.makeCommandBuffer()!; let enc = cb.makeComputeCommandEncoder()!
        // ONE dispatch — gdn_prep_rows fuses all 8 ops
        encodeGdnPrepRows(enc, convOut: bConv, aP: bA, bP: bB, aLog: bALog, dtBias: bDt,
                          qn: bQn, kn: bKn, v: bV, g: bG, beta: bBeta,
                          M: M, numKH: numKHeads, headKD: headKDim, numVH: numVHeads,
                          keyDim: keyDim, valDim: valueDim, eps: eps,
                          qScale: invScale * invScale, kScale: invScale)
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        let pqn   = bQn.contents().bindMemory(to: Float16.self, capacity: M * keyDim)
        let pkn   = bKn.contents().bindMemory(to: Float16.self, capacity: M * keyDim)
        let pv    = bV.contents().bindMemory(to: Float16.self, capacity: M * valueDim)
        let pg    = bG.contents().bindMemory(to: Float.self, capacity: M * numVHeads)
        let pbeta = bBeta.contents().bindMemory(to: Float.self, capacity: M * numVHeads)
        return (
            MLXArray(Array(UnsafeBufferPointer(start: pqn,   count: M * keyDim)),    [M * numKHeads, headKDim]),
            MLXArray(Array(UnsafeBufferPointer(start: pkn,   count: M * keyDim)),    [M * numKHeads, headKDim]),
            MLXArray(Array(UnsafeBufferPointer(start: pv,    count: M * valueDim)),  [M, valueDim]),
            MLXArray(Array(UnsafeBufferPointer(start: pg,    count: M * numVHeads)), [1, M, numVHeads]),
            MLXArray(Array(UnsafeBufferPointer(start: pbeta, count: M * numVHeads)), [1, M, numVHeads])
        )
    }

    /// F5: gdn_resid_post_norm fused — ⑳resid_add (hBuf += mixerOut) ㉑rmsnorm post
    /// (hBuf → postNorm) — 2 separate kernels → 1 fused dispatch.
    /// Uses gdn_resid_postnorm_rows: a TRUE single Metal kernel dispatch (not a chain).
    public static func gdnResidPostNormFused(
        hBuf: MLXArray, mixerOut: MLXArray, postW: MLXArray,
        M: Int, H: Int, eps: Float
    ) -> (h: MLXArray, postNorm: MLXArray)? {
        guard let (device, queue) = RawMetalForward.ensure(),
              ensureGdnPipelines(), _gdnResidPostNormRowsPipeline != nil else { return nil }
        // bH is a CPU-side copy of hBuf (mtlBuf copies the data); resid_add modifies it in-place.
        guard let bH    = RawMetalForward.mtlBuf(hBuf.asType(.float16), device),
              let bR    = RawMetalForward.mtlBuf(mixerOut.asType(.float16), device),
              let bW    = RawMetalForward.mtlBuf(postW.asType(.float16), device),
              let bPost = device.makeBuffer(length: M * H * 2, options: .storageModeShared) else { return nil }
        let cb = queue.makeCommandBuffer()!; let enc = cb.makeComputeCommandEncoder()!
        // ONE dispatch — gdn_resid_postnorm_rows fuses resid_add + rmsnorm
        encodeGdnResidPostNormRows(enc, h: bH, r: bR, w: bW, postNorm: bPost, M: M, H: H, eps: eps)
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        let ph = bH.contents().bindMemory(to: Float16.self, capacity: M * H)
        let pn = bPost.contents().bindMemory(to: Float16.self, capacity: M * H)
        return (
            MLXArray(Array(UnsafeBufferPointer(start: ph, count: M * H)), [M, H]),
            MLXArray(Array(UnsafeBufferPointer(start: pn, count: M * H)), [M, H])
        )
    }

    /// Test-entry wrapper: drives gqmm4_swiglu_rows in a self-contained command buffer.
    /// x[M,K] f16, inds[M*Ktop] int32, wG/wU [E,N,K/2] 4-bit, sG/sU/bG/bU [E,N,K/64] f16.
    /// Returns h[M*Ktop, N] f16 — bit-identical to gatherQmmRows(g)+gatherQmmRows(u)+swigluRaw.
    public static func gatherQmmSwigluRows(x: MLXArray, inds: MLXArray,
                                           wG: MLXArray, sG: MLXArray, bG: MLXArray,
                                           wU: MLXArray, sU: MLXArray, bU: MLXArray,
                                           M: Int, Ktop: Int, K: Int, N: Int) -> MLXArray? {
        guard let (device, queue) = RawMetalForward.ensure() else { return nil }
        guard N % 8 == 0, K % 512 == 0 else { print("[raw-gqmm-swiglu-rows] 非fast (N=\(N) K=\(K)) 未対応"); return nil }
        _ = RawMetalForward.compileGqmmSwigluRows()
        guard let bx  = RawMetalForward.mtlBuf(x.asType(.float16), device),
              let bin = RawMetalForward.mtlBuf(inds.asType(.int32), device),
              let bwG = RawMetalForward.mtlBuf(wG, device),
              let bsG = RawMetalForward.mtlBuf(sG.asType(.float16), device),
              let bbG = RawMetalForward.mtlBuf(bG.asType(.float16), device),
              let bwU = RawMetalForward.mtlBuf(wU, device),
              let bsU = RawMetalForward.mtlBuf(sU.asType(.float16), device),
              let bbU = RawMetalForward.mtlBuf(bU.asType(.float16), device) else { return nil }
        let outBuf = device.makeBuffer(length: M * Ktop * N * 2, options: .storageModeShared)!
        let cb = queue.makeCommandBuffer()!; let enc = cb.makeComputeCommandEncoder()!
        encodeGatherQmmSwigluRows(enc, wG: bwG, sG: bsG, bG: bbG,
                                  wU: bwU, sU: bsU, bU: bbU,
                                  x: bx, inds: bin, out: outBuf,
                                  M: M, Ktop: Ktop, K: K, N: N)
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        let ptr = outBuf.contents().bindMemory(to: Float16.self, capacity: M * Ktop * N)
        return MLXArray(Array(UnsafeBufferPointer(start: ptr, count: M * Ktop * N)), [M * Ktop, N])
    }
}
