import Foundation
import Metal
import MLX

extension Tell {
    /// MoE decode roofline (M=1): per-stage GPU µs, bytes read, achieved GB/s and FLOP/byte for the
    /// shipped resident MoE block, against a measured copy-bandwidth ceiling. Each stage is encoded
    /// for all 40 layers in order (distinct weights, so the SLC cannot serve repeats), REPS times in
    /// ONE command buffer; time = (gpuEnd - gpuStart) / (REPS * 40). No timers inside the encodes.
    /// Env: QWISP_MOEB_REPS (default 20).
    public static func moeDecodeRoofline(modelDir: String) -> String {
        guard let store = try? WeightStore(modelDir: modelDir) else { return "[moe-roofline] load fail" }
        store.residentAll()
        let engine = SeedlessEngine.build(store: store)
        guard let (fwd, _) = engine.makeFused(maxM: 8, maxSeqLen: 64),
              let (device, _) = SeedlessMetalForward.ensure() else { return "[moe-roofline] makeFused fail" }
        let queue = fwd.queue
        let reps = Swift.max(1, envInt("QWISP_MOEB_REPS", 20))
        let H = fwd.H
        let L0 = fwd.layers[0]
        let E = L0.E, I = L0.I, K = L0.Ktop
        // random activation in postNorm
        let xp = fwd.postNorm.contents().bindMemory(to: Float16.self, capacity: H)
        for i in 0 ..< H { xp[i] = Float16(Float.random(in: -0.5 ... 0.5)) }

        func gpuMs(_ body: (MTLComputeCommandEncoder) -> Void) -> Double {
            let cb = queue.makeCommandBuffer()!; let enc = cb.makeComputeCommandEncoder()!
            body(enc); enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
            return (cb.gpuEndTime - cb.gpuStartTime) * 1000
        }
        // copy-bandwidth ceiling: blit 1 GB (read + write)
        let n = 1 << 30
        let a = device.makeBuffer(length: n, options: .storageModePrivate)!, b = device.makeBuffer(length: n, options: .storageModePrivate)!
        var bwBest = 0.0
        for _ in 0 ..< 4 {
            let cb = queue.makeCommandBuffer()!; let bl = cb.makeBlitCommandEncoder()!
            bl.copy(from: a, sourceOffset: 0, to: b, destinationOffset: 0, size: n); bl.endEncoding()
            cb.commit(); cb.waitUntilCompleted()
            bwBest = Swift.max(bwBest, 2.0 * Double(n) / (cb.gpuEndTime - cb.gpuStartTime) / 1e9)
        }

        // bytes / flops per layer (from the actual buffers)
        func len(_ bs: [MTLBuffer]) -> Double { Double(bs.reduce(0) { $0 + $1.length }) }
        let w = L0.moe
        let routeB = len([w.gW, w.gS, w.gB])
        let guB = len([w.swGW, w.swGS, w.swGB, w.swUW, w.swUS, w.swUB]) / Double(E) * Double(K)
        let downB = len([w.swDW, w.swDS, w.swDB]) / Double(E) * Double(K)
        let shB = len([w.shGW, w.shGS, w.shGB, w.shUW, w.shUS, w.shUB, w.shDW, w.shDS, w.shDB, w.sgW, w.sgS, w.sgB])
        let routeF = 2.0 * Double(H * E), guF = 2.0 * 2 * Double(I * H * K), downF = 2.0 * Double(I * H * K)
        let shF = 2.0 * 3 * Double(I * H) + 2.0 * Double(H)

        // fill inds once (route on layer 0)
        _ = gpuMs { enc in
            SeedlessFusedVerify.encodeMoERouteRows(enc, x: fwd.postNorm, w: w, sc: fwd.moeSc, M: 1, E: E, H: H, Ktop: K)
        }
        let sc = fwd.moeSc
        func stage(_ name: String, bytes: Double, flops: Double,
                   _ encode: (MTLComputeCommandEncoder, SeedlessFusedVerify.SeedlessFusedForward.Layer) -> Void) -> String {
            _ = gpuMs { enc in for L in fwd.layers { encode(enc, L) } }   // warm
            let ms = gpuMs { enc in for _ in 0 ..< reps { for L in fwd.layers { encode(enc, L) } } }
            let calls = Double(reps * fwd.layers.count)
            let us = ms * 1000 / calls
            let gbs = bytes / (us * 1e-6) / 1e9
            return String(format: "  %-18@ %7.1f µs/layer  %6.2f MB  %6.1f GB/s (%3.0f%% of copy)  %5.2f FLOP/B  x40 = %5.2f ms",
                          name as NSString, us, bytes / 1e6, gbs, 100 * gbs / bwBest, flops / bytes, us * 40 / 1000)
        }
        var lines = [String(format: "[moe-roofline] M=1 E=%d K=%d I=%d H=%d reps=%d  copy ceiling %.1f GB/s", E, K, I, H, reps, bwBest)]
        lines.append(stage("route", bytes: routeB, flops: routeF) { enc, L in
            SeedlessFusedVerify.encodeMoERouteRows(enc, x: fwd.postNorm, w: L.moe, sc: sc, M: 1, E: E, H: H, Ktop: K)
        })
        lines.append(stage("gate+up+swiglu", bytes: guB, flops: guF) { enc, L in
            SeedlessFusedVerify.encodeGatherQmmSwigluRows(enc, wG: L.moe.swGW, sG: L.moe.swGS, bG: L.moe.swGB,
                                                          wU: L.moe.swUW, sU: L.moe.swUS, bU: L.moe.swUB,
                                                          x: fwd.postNorm, inds: sc.inds, out: sc.h,
                                                          M: 1, Ktop: K, K: H, N: I,
                                                          xByteOffset: 0, indsOffset: 0, outByteOffset: 0)
        })
        lines.append(stage("down", bytes: downB, flops: downF) { enc, L in
            SeedlessFusedVerify.encodeGatherQmmRows(enc, w: L.moe.swDW, scales: L.moe.swDS, biases: L.moe.swDB,
                                                    x: sc.h, inds: sc.inds, out: sc.d,
                                                    M: 1, Ktop: K, K: I, N: H, lhsPer: true,
                                                    xByteOffset: 0, indsOffset: 0, outByteOffset: 0)
        })
        lines.append(stage("routed (g+u+d+c)", bytes: guB + downB, flops: guF + downF) { enc, L in
            SeedlessFusedVerify.encodeMoEGatherRowsRange(enc, x: fwd.postNorm, w: L.moe, sc: sc, r0: 0, r1: 1,
                                                         Ktop: K, I: I, H: H, slotTable: nil)
        })
        lines.append(stage("shared", bytes: shB, flops: shF) { enc, L in
            SeedlessFusedVerify.encodeMoESharedRows(enc, x: fwd.postNorm, out: fwd.moeOut, w: L.moe, sc: sc,
                                                    M: 1, I: I, H: H, Ktop: K)
        })
        lines.append(stage("block total", bytes: routeB + guB + downB + shB, flops: routeF + guF + downF + shF) { enc, L in
            SeedlessFusedVerify.encodeMoEBlockRows(enc, x: fwd.postNorm, out: fwd.moeOut, w: L.moe, sc: sc,
                                                   M: 1, E: E, I: I, Ktop: K, H: H)
        })
        // ── sub-kernels of route / shared (latency-bound suspects) ──
        let qmm8B = routeB, sgB = len([w.sgW, w.sgS, w.sgB])
        let s1B = len([w.shGW, w.shGS, w.shGB, w.shUW, w.shUS, w.shUB]), sdB = len([w.shDW, w.shDS, w.shDB])
        lines.append(stage("  route.qmm8", bytes: qmm8B, flops: routeF) { enc, L in
            SeedlessFusedVerify.encodeQmm8Rows(enc, w: L.moe.gW, scales: L.moe.gS, biases: L.moe.gB,
                                               x: fwd.postNorm, out: sc.gl, M: 1, K: H, N: E)
        })
        lines.append(stage("  route.top8", bytes: Double(E * 2), flops: 0) { enc, _ in
            SeedlessFusedVerify.encodeRouteTop8Rows(enc, logits: sc.gl, inds: sc.inds, scores: sc.scores, M: 1, N: E, K: K)
        })
        if let zi = SeedlessFusedVerify.zeroOneIndsBuf() {
            lines.append(stage("  shared.S1", bytes: s1B, flops: 2.0 * 2 * Double(I * H)) { enc, L in
                SeedlessFusedVerify.encodeGatherQmmSwigluRows(enc, wG: L.moe.shGW, sG: L.moe.shGS, bG: L.moe.shGB,
                                                              wU: L.moe.shUW, sU: L.moe.shUS, bU: L.moe.shUB,
                                                              x: fwd.postNorm, inds: zi, out: sc.shAct,
                                                              M: 1, Ktop: 1, K: H, N: I,
                                                              xByteOffset: 0, indsOffset: 0, outByteOffset: 0)
            })
        }
        lines.append(stage("  shared.down", bytes: sdB, flops: 2.0 * Double(I * H)) { enc, L in
            SeedlessFusedVerify.encodeQmmRows(enc, w: L.moe.shDW, scales: L.moe.shDS, biases: L.moe.shDB,
                                              x: sc.shAct, out: sc.sharedY, M: 1, K: I, N: H)
        })
        lines.append(stage("  shared.S2", bytes: sgB, flops: 2.0 * Double(H)) { enc, L in
            SeedlessFusedVerify.encodeSharedGateCombineRows(enc, sgW: L.moe.sgW, sgS: L.moe.sgS, sgB: L.moe.sgB,
                                                            x: fwd.postNorm, y: sc.y, sharedY: sc.sharedY, out: fwd.moeOut,
                                                            K: H, H: H, M: 1)
        })
        lines.append(stage("  combine", bytes: Double(K * H * 2), flops: 0) { enc, _ in
            SeedlessFusedVerify.encodeCombineRows(enc, d: sc.d, scores: sc.scores, y: sc.y, Ktop: K, N: H, M: 1,
                                                  dByteOffset: 0, scoresOffset: 0, yByteOffset: 0)
        })
        // ── horizontal-fusion ceiling: independent pairs on a CONCURRENT encoder (no barrier inside
        // a pair, barrier between pairs) vs the same pair serial. Upper bound for fusing them. ──
        func gpuMsConc(_ body: (MTLComputeCommandEncoder) -> Void) -> Double {
            let cb = queue.makeCommandBuffer()!; let enc = cb.makeComputeCommandEncoder(dispatchType: .concurrent)!
            body(enc); enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
            return (cb.gpuEndTime - cb.gpuStartTime) * 1000
        }
        func pair(_ name: String, _ a: (MTLComputeCommandEncoder, SeedlessFusedVerify.SeedlessFusedForward.Layer) -> Void,
                  _ b: (MTLComputeCommandEncoder, SeedlessFusedVerify.SeedlessFusedForward.Layer) -> Void) -> String {
            let calls = Double(reps * fwd.layers.count)
            _ = gpuMs { enc in for L in fwd.layers { a(enc, L); b(enc, L) } }
            let ser = gpuMs { enc in for _ in 0 ..< reps { for L in fwd.layers { a(enc, L); b(enc, L) } } } * 1000 / calls
            _ = gpuMsConc { enc in for L in fwd.layers { a(enc, L); b(enc, L); enc.memoryBarrier(scope: .buffers) } }
            let con = gpuMsConc { enc in for _ in 0 ..< reps { for L in fwd.layers { a(enc, L); b(enc, L); enc.memoryBarrier(scope: .buffers) } } } * 1000 / calls
            return String(format: "  pair %-28@ serial %6.1f µs  concurrent %6.1f µs  (save %5.1f µs/layer, %4.2f ms/token)",
                          name as NSString, ser, con, ser - con, (ser - con) * 40 / 1000)
        }
        if let zi = SeedlessFusedVerify.zeroOneIndsBuf() {
            lines.append(pair("qmm8 || shared.S1", { enc, L in
                SeedlessFusedVerify.encodeQmm8Rows(enc, w: L.moe.gW, scales: L.moe.gS, biases: L.moe.gB,
                                                   x: fwd.postNorm, out: sc.gl, M: 1, K: H, N: E)
            }, { enc, L in
                SeedlessFusedVerify.encodeGatherQmmSwigluRows(enc, wG: L.moe.shGW, sG: L.moe.shGS, bG: L.moe.shGB,
                                                              wU: L.moe.shUW, sU: L.moe.shUS, bU: L.moe.shUB,
                                                              x: fwd.postNorm, inds: zi, out: sc.shAct,
                                                              M: 1, Ktop: 1, K: H, N: I,
                                                              xByteOffset: 0, indsOffset: 0, outByteOffset: 0)
            }))
        }
        lines.append(pair("top8 || shared.down", { enc, _ in
            SeedlessFusedVerify.encodeRouteTop8Rows(enc, logits: sc.gl, inds: sc.inds, scores: sc.scores, M: 1, N: E, K: K)
        }, { enc, L in
            SeedlessFusedVerify.encodeQmmRows(enc, w: L.moe.shDW, scales: L.moe.shDS, biases: L.moe.shDB,
                                              x: sc.shAct, out: sc.sharedY, M: 1, K: I, N: H)
        }))
        // ── rest of the decode step: GDN mixers (30), attention mixers (10, shallow ctx), lm_head ──
        func bufBytes(_ v: Any) -> (all: Double, packed: Double) {
            var all = 0.0, packed = 0.0
            for c in Mirror(reflecting: v).children {
                if let b = c.value as? MTLBuffer {
                    all += Double(b.length)
                    if let l = c.label, l.hasSuffix("W") || l.hasSuffix("Wq") { packed += Double(b.length) }
                }
            }
            return (all, packed)
        }
        let gdnLayers = fwd.layers.filter { $0.isLinear }, attnLayers = fwd.layers.filter { !$0.isLinear }
        if let g0 = gdnLayers.first?.gdn, let gc = gdnLayers.first?.gdnCache {
            let wb = bufBytes(g0), stateB = Double(gc.Hv * gc.Dv * gc.Dk * 4 * 2 + (gc.K - 1) * gc.C * 2 * 2)
            let us = { () -> Double in
                _ = gpuMs { enc in for L in gdnLayers { fwd.encodePreMoE(enc, L, M: 1) } }
                return gpuMs { enc in for _ in 0 ..< reps { for L in gdnLayers { fwd.encodePreMoE(enc, L, M: 1) } } }
                    * 1000 / Double(reps * gdnLayers.count)
            }()
            // bytes from model dims (the buffer set also holds fused concat copies of the same weights)
            let q4 = 0.5625   // 4-bit + f16 scale/bias per 64
            let gdnParams = Double((2 * gc.Hv * gc.Dk / 2 + gc.Hv * gc.Dv + gc.Hv * gc.Dv + 2 * gc.Hv) * H + H * gc.Hv * gc.Dv)
            _ = wb
            let bytes = gdnParams * q4 + stateB, flops = 2.0 * gdnParams
            lines.append(String(format: "  %-18@ %7.1f µs/layer  %6.2f MB  %6.1f GB/s (%3.0f%% of copy)  %5.2f FLOP/B  x%d = %5.2f ms   (weights %.2f MB + state r/w %.2f MB)",
                                "gdn mixer" as NSString, us, bytes / 1e6, bytes / (us * 1e-6) / 1e9, 100 * bytes / (us * 1e-6) / 1e9 / bwBest,
                                flops / bytes, gdnLayers.count, us * Double(gdnLayers.count) / 1000, gdnParams * q4 / 1e6, stateB / 1e6))
        }
        // GDN sub-kernels (fuseGDN path), over the 30 GDN layers
        if SeedlessFusedVerify.SeedlessFusedForward.fuseGDN {
            let gs = fwd.gdnSc
            let keyDim = fwd.headKDim * fwd.numKHeads, valueDim = fwd.headVDim * fwd.numVHeads, convDim = 2 * keyDim + valueDim
            let invScale = Float(pow(Double(fwd.headKDim), -0.5))
            func gstage(_ name: String, bytes: Double, flops: Double,
                        _ body: (MTLComputeCommandEncoder, SeedlessFusedVerify.SeedlessFusedForward.Layer) -> Void) -> String {
                _ = gpuMs { enc in for L in gdnLayers { body(enc, L) } }
                let us = gpuMs { enc in for _ in 0 ..< reps { for L in gdnLayers { body(enc, L) } } } * 1000 / Double(reps * gdnLayers.count)
                return String(format: "    %-16@ %7.1f µs/layer  %6.2f MB  %6.1f GB/s  %5.2f FLOP/B  x%d = %5.2f ms",
                              name as NSString, us, bytes / 1e6, bytes / (us * 1e-6) / 1e9, bytes > 0 ? flops / bytes : 0,
                              gdnLayers.count, us * Double(gdnLayers.count) / 1000)
            }
            let inN = Double(convDim + valueDim + 2 * fwd.numVHeads)
            lines.append(gstage("gdn.inproj", bytes: inN * Double(H) * 0.5625, flops: 2 * inN * Double(H)) { enc, L in
                let w = L.gdn!
                SeedlessFusedVerify.encodeQmmInProjDemuxRows(enc, w: w.catInProjW!, scales: w.catInProjS!, biases: w.catInProjB!, x: fwd.normed,
                                                             outQkv: gs.qkv, outZ: gs.z, outB: gs.bP, outA: gs.aP, M: 1, K: H,
                                                             dims: (qkv: convDim, z: valueDim, b: fwd.numVHeads, a: fwd.numVHeads))
            })
            lines.append(gstage("gdn.convshift", bytes: Double(convDim * (fwd.convKernel * 2 + 2) * 2), flops: 0) { enc, L in
                SeedlessFusedVerify.encodeGdnFusionConvShift(enc, hist: L.gdnCache!.convHist, qkv: gs.qkv, w: L.gdn!.conv1dW,
                                                             convOut: gs.convOut, histOut: L.gdnCache!.convHistOut,
                                                             M: 1, K: fwd.convKernel, C: convDim)
            })
            lines.append(gstage("gdn.prep", bytes: Double(convDim * 2 * 2), flops: 0) { enc, L in
                let w = L.gdn!
                SeedlessFusedVerify.encodeGdnPrepRows(enc, convOut: gs.convOut, aP: gs.aP, bP: gs.bP, aLog: w.aLog, dtBias: w.dtBias,
                                                      qn: gs.qn, kn: gs.kn, v: gs.v1, g: gs.g, beta: gs.beta,
                                                      M: 1, numKH: fwd.numKHeads, headKD: fwd.headKDim, numVH: fwd.numVHeads,
                                                      keyDim: keyDim, valDim: valueDim, eps: fwd.eps,
                                                      qScale: invScale * invScale, kScale: invScale)
            })
            let stB = Double(fwd.numVHeads * fwd.headVDim * fwd.headKDim * 4 * 2)
            lines.append(gstage("gdn.deltastep", bytes: stB, flops: 6.0 * Double(fwd.numVHeads * fwd.headVDim * fwd.headKDim)) { enc, L in
                let c = L.gdnCache!
                SeedlessFusedVerify.encodeGatedDeltaStepRows(enc, q: gs.qn, k: gs.kn, v: gs.v1, g: gs.g, beta: gs.beta,
                                                             stateIn: c.state, stateOut: c.stateOut, y: gs.coreOut,
                                                             T: 1, B: 1, Hv: fwd.numVHeads, Dv: fwd.headVDim)
            })
            lines.append(gstage("gdn.normgate", bytes: Double(valueDim * 2 * 3), flops: 0) { enc, L in
                let w = L.gdn!
                SeedlessFusedVerify.encodeGdnNormGateRows(enc, coreOut: gs.coreOut, z: gs.z, normWeight: w.normWeight,
                                                          outV: gs.outV, M: 1, Hv: fwd.numVHeads, Dv: fwd.headVDim,
                                                          eps: fwd.eps, promoteF32: w.promoteRMS)
            })
            lines.append(gstage("gdn.outproj", bytes: Double(valueDim * H) * 0.5625, flops: 2.0 * Double(valueDim * H)) { enc, L in
                let w = L.gdn!
                SeedlessFusedVerify.encodeQmmRows(enc, w: w.outW, scales: w.outS, biases: w.outB, x: gs.outV, out: fwd.mixerOut,
                                                  M: 1, K: valueDim, N: H)
            })
        }
        if let a0 = attnLayers.first?.attn {
            let wb = bufBytes(a0)
            let us = { () -> Double in
                _ = gpuMs { enc in for L in attnLayers { fwd.encodePreMoE(enc, L, M: 1) } }
                return gpuMs { enc in for _ in 0 ..< reps { for L in attnLayers { fwd.encodePreMoE(enc, L, M: 1) } } }
                    * 1000 / Double(reps * attnLayers.count)
            }()
            _ = wb
            let nh = fwd.numHeads, nkv = fwd.numKV, hd = fwd.headDim
            let attnParams = Double((nh * hd * 2 + 2 * nkv * hd) * H + H * nh * hd)
            let ab = attnParams * 0.5625, flops = 2.0 * attnParams
            lines.append(String(format: "  %-18@ %7.1f µs/layer  %6.2f MB  %6.1f GB/s (%3.0f%% of copy)  %5.2f FLOP/B  x%d = %5.2f ms   (weights only; ctx <= %d)",
                                "attn mixer" as NSString, us, ab / 1e6, ab / (us * 1e-6) / 1e9, 100 * ab / (us * 1e-6) / 1e9 / bwBest,
                                flops / ab, attnLayers.count, us * Double(attnLayers.count) / 1000, 2 * reps + 2))
        }
        if let hd = fwd.head {
            let bytes = Double(hd.lmW.length + hd.lmS.length + hd.lmB.length)
            _ = gpuMs { enc in SeedlessFusedVerify.encodeQmmRows(enc, w: hd.lmW, scales: hd.lmS, biases: hd.lmB, x: fwd.normed, out: hd.logits, M: 1, K: H, N: hd.vocab) }
            let us = gpuMs { enc in for _ in 0 ..< reps { SeedlessFusedVerify.encodeQmmRows(enc, w: hd.lmW, scales: hd.lmS, biases: hd.lmB, x: fwd.normed, out: hd.logits, M: 1, K: H, N: hd.vocab) } } * 1000 / Double(reps)
            lines.append(String(format: "  %-18@ %7.1f µs/token  %6.2f MB  %6.1f GB/s (%3.0f%% of copy)  %5.2f FLOP/B",
                                "lm_head" as NSString, us, bytes / 1e6, bytes / (us * 1e-6) / 1e9, 100 * bytes / (us * 1e-6) / 1e9 / bwBest,
                                2.0 * Double(hd.lmW.length) * 2 / bytes))
        }
        lines.append("MOEROOFLINE done")
        return lines.joined(separator: "\n")
    }
}
