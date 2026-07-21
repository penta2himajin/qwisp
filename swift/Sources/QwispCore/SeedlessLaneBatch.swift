import Foundation
import Metal
import MLX

// Lane-batched greedy decode — Stage 1 of the parallel-agent speedup.
//
// B concurrent sequences ("lanes"), each with its OWN caches (KV arena + GDN
// state) held by a per-lane SeedlessFusedForward, advance ONE token each through
// a SINGLE fused pass so the dense/MoE weight reads amortize across lanes (the
// M>1 prize: ~3x/token at M=16, spec-width probe) while the sequence-coupled
// mixers (attention / GDN recurrence) run per-lane at literal M=1 with that
// lane's caches — the exact solo kernels on the exact solo state.
//
// Bit-exactness by construction (locked test `lane_batch_bitexact`):
//   - mixer rows: literal M=1 solo path per lane (same kernel, same shapes,
//     same cache) ⇒ identical bits.
//   - norm/MoE/resid rows at M=B: per-row grids whose row math is M-invariant —
//     the engine's core self-consistency guarantee (M-row kernel suite,
//     RAWTESTS) that already underwrites strict verify ≡ M=1 decode.
//
// Per layer: norm (M=B) → mixer per lane (M=1, staged via lane_row_copy) →
// resid+postNorm (M=B) → MoE (M=B, per-row routing) → residAdd (M=B).
//
// Additive: reuses the frozen encode statics + Layer weight structs from the
// driver; no frozen file is modified. Resident tier only (streaming's chunked
// expert IO is a different scheduling problem — out of Stage 1 scope).
public final class SeedlessLaneBatch {
    let driver: SeedlessFusedVerify.SeedlessFusedForward     // weights + M=B scratch
    let lanes: [SeedlessFusedVerify.SeedlessFusedForward]    // per-lane caches + scratch
    public let B: Int
    let laneX: MTLBuffer                                     // [1, H] mixer input staging
    let laneOut: MTLBuffer                                   // [1, H] mixer output staging

    nonisolated(unsafe) static var copyPipeline: MTLComputePipelineState? = nil
    static func compileCopy(_ device: MTLDevice) -> Bool {
        if copyPipeline != nil { return true }
        let src = """
        #include <metal_stdlib>
        using namespace metal;
        kernel void lane_row_copy(device const half* src [[buffer(0)]],
                                  device half* dst [[buffer(1)]],
                                  constant uint& srcOff [[buffer(2)]],
                                  constant uint& dstOff [[buffer(3)]],
                                  uint i [[thread_position_in_grid]]) {
            dst[dstOff + i] = src[srcOff + i];
        }
        """
        guard let lib = try? device.makeLibrary(source: src, options: nil),
              let fn = lib.makeFunction(name: "lane_row_copy"),
              let ps = try? device.makeComputePipelineState(function: fn) else { return false }
        copyPipeline = ps
        return true
    }

    private func encodeRowCopy(_ enc: MTLComputeCommandEncoder,
                               src: MTLBuffer, srcOff: Int, dst: MTLBuffer, dstOff: Int, count: Int) {
        let ps = SeedlessLaneBatch.copyPipeline!
        enc.setComputePipelineState(ps)
        enc.setBuffer(src, offset: 0, index: 0)
        enc.setBuffer(dst, offset: 0, index: 1)
        var so = UInt32(srcOff), do_ = UInt32(dstOff)
        enc.setBytes(&so, length: 4, index: 2)
        enc.setBytes(&do_, length: 4, index: 3)
        enc.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: min(ps.maxTotalThreadsPerThreadgroup, 256), height: 1, depth: 1))
    }

    /// driver: weights + scratch sized maxM ≥ lanes.count. lanes: one forward per
    /// sequence (its caches are the lane state; its scratch serves the M=1 mixer).
    /// All must be resident-mode with the same layer stack.
    public init?(driver: SeedlessFusedVerify.SeedlessFusedForward,
                 lanes: [SeedlessFusedVerify.SeedlessFusedForward]) {
        guard !lanes.isEmpty, driver.maxM >= lanes.count,
              lanes.allSatisfy({ $0.layers.count == driver.layers.count }),
              SeedlessLaneBatch.compileCopy(driver.device) else { return nil }
        self.driver = driver
        self.lanes = lanes
        self.B = lanes.count
        guard let lx = driver.device.makeBuffer(length: driver.H * 2, options: .storageModeShared),
              let lo = driver.device.makeBuffer(length: driver.H * 2, options: .storageModeShared) else { return nil }
        self.laneX = lx
        self.laneOut = lo
    }

    /// One batched step: row b of `x` [B, H] is lane b's next-token hidden input.
    /// Returns the residual-stream output rows [B, H] (pre-final-norm, mirroring
    /// forwardRows(finalNormW: nil)). Each lane's caches advance by exactly one
    /// position — bit-identical to that lane running forwardRows(M: 1) alone.
    public func forwardRowsBatch(_ x: MLXArray) -> MLXArray? {
        let H = driver.H
        guard x.dim(0) == B, x.dim(1) == H else { return nil }
        let xf = x.asType(.float16).reshaped([-1]); xf.eval()
        let arr = xf.asArray(Float16.self)
        driver.hBuf.contents().bindMemory(to: Float16.self, capacity: driver.maxM * H)
            .update(from: arr, count: B * H)

        let cb = driver.queue.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        for li in driver.layers.indices {
            let L = driver.layers[li]
            SeedlessFusedVerify.encodeRmsNormRows(enc, x: driver.hBuf, w: L.inputLN,
                                                  out: driver.normed, rows: B, D: H, eps: driver.eps)
            for b in 0 ..< B {
                let lane = lanes[b]
                let LL = lane.layers[li]
                encodeRowCopy(enc, src: driver.normed, srcOff: b * H, dst: laneX, dstOff: 0, count: H)
                if L.isLinear, let gw = L.gdn, let gc = LL.gdnCache {
                    SeedlessFusedVerify.encodeGdnLayerRows(enc, x: laneX, out: laneOut, w: gw,
                                                           sc: lane.gdnSc, cache: gc, M: 1, H: H,
                                                           numKHeads: driver.numKHeads, numVHeads: driver.numVHeads,
                                                           headKDim: driver.headKDim, headVDim: driver.headVDim,
                                                           convKernel: driver.convKernel, eps: driver.eps)
                    gc.swapState()
                } else if let aw = L.attn, let kv = LL.kvCache {
                    SeedlessFusedVerify.encodeAttnLayerRows(enc, x: laneX, out: laneOut, w: aw,
                                                            sc: lane.attnSc, kv: kv, M: 1, H: H,
                                                            numHeads: driver.numHeads, numKV: driver.numKV,
                                                            headDim: driver.headDim, ropeDim: driver.ropeDim,
                                                            ropeBase: driver.ropeBase, eps: driver.eps)
                    kv.len += 1
                }
                encodeRowCopy(enc, src: laneOut, srcOff: 0, dst: driver.mixerOut, dstOff: b * H, count: H)
            }
            // resid + postNorm at M=B — mirrors encodePreMoE's tail exactly.
            if SeedlessFusedVerify.SeedlessFusedForward.fuseGDN,
               SeedlessFusedVerify._gdnResidPostNormRowsPipeline != nil {
                SeedlessFusedVerify.encodeGdnResidPostNormRows(enc, h: driver.hBuf, r: driver.mixerOut,
                                                               w: L.postLN, postNorm: driver.postNorm,
                                                               M: B, H: H, eps: driver.eps)
            } else {
                SeedlessFusedVerify.encodeResidAdd(enc, h: driver.hBuf, r: driver.mixerOut, total: B * H)
                SeedlessFusedVerify.encodeRmsNormRows(enc, x: driver.hBuf, w: L.postLN,
                                                      out: driver.postNorm, rows: B, D: H, eps: driver.eps)
            }
            SeedlessFusedVerify.encodeMoEBlockRows(enc, x: driver.postNorm, out: driver.moeOut,
                                                   w: L.moe, sc: driver.moeSc,
                                                   M: B, E: L.E, I: L.I, Ktop: L.Ktop, H: H)
            SeedlessFusedVerify.encodeResidAdd(enc, h: driver.hBuf, r: driver.moeOut, total: B * H)
        }
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        SeedlessFusedVerify.SeedlessFusedForward.profLastGPUMs = (cb.gpuEndTime - cb.gpuStartTime) * 1000.0

        let ptr = driver.hBuf.contents().bindMemory(to: Float16.self, capacity: driver.maxM * H)
        return MLXArray(Array(UnsafeBufferPointer(start: ptr, count: B * H)), [B, H])
    }
}
