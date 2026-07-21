import Foundation
import Metal
import MLX

// Feasibility PoC for cross-request prefix caching (issue: agentic TTFT dominated by re-prefilling
// the whole ~8K context every request). Proves — using ONLY shipped primitives (prefill / forward /
// snapshot / rollback / stepArgmax), no forward-path change — that the full cross-request protocol
//
//     prefill(A) → snapshot S → prefill(tail we discard) → rollback(S) → prefill(B) → decode
//
// yields a BYTE-IDENTICAL token stream to the baseline prefill(A+B) → decode, while paying only the
// prefill of the suffix B instead of the whole A+B. A=stable content prefix (reused across requests),
// tail = the generation prompt + generated tokens that the next request rewinds past, B = the new
// content appended in the next request.
extension Tell {
    // Lane-batch bench (Stage 1, parallel agents): real-model speed of the
    // bit-exact lane-batched greedy step. B lanes prefill DIFFERENT prompts
    // (diverse routing = worst-case MoE union), then decode batched; reports
    // ms/step and per-stream + aggregate tok/s vs the M=1 solo baseline.
    // Env: QWISP_LANE_B (default "1,2,3,4,8" sweep), QWISP_LANE_CTX (default 1024).
    public static func laneBatchBench(modelDir: String) -> String {
        guard let store = try? WeightStore(modelDir: modelDir) else { return "[lane-bench] load fail\nLANEBENCH done" }
        store.residentAll()
        let engine = SeedlessEngine.build(store: store)
        let ctx = Tell.envInt("QWISP_LANE_CTX", 1024)
        let bs = (ProcessInfo.processInfo.environment["QWISP_LANE_B"] ?? "1,2,3,4,8")
            .split(separator: ",").compactMap { Int($0) }
        func tok(_ n: Int, _ salt: Int) -> [Int32] { (0..<n).map { Int32((($0 &* 7 &+ salt) % 5000) + 100) } }
        var lines = ["[lane-bench] ctx=\(ctx)/lane resident — bit-exact lane-batched greedy step",
                     "     B   ms/step   per-stream   aggregate"]
        for B in bs {
            // Fresh lanes each B: own arena + prefill of a DIFFERENT prompt.
            var lanes: [SeedlessFusedVerify.SeedlessFusedForward] = []
            for b in 0 ..< B {
                guard let (fwd, _) = engine.makeFused(maxM: 64, maxSeqLen: ctx + 128) else {
                    return lines.joined(separator: "\n") + "\n[lane-bench] lane makeFused nil\nLANEBENCH done"
                }
                let prompt = tok(ctx, 13 + b * 17)
                var pos = 0
                while pos < ctx {
                    let end = Swift.min(pos + 64, ctx)
                    _ = fwd.forwardRows(engine.embed(tokens: Array(prompt[pos ..< end])), M: end - pos)
                    pos = end
                }
                lanes.append(fwd)
            }
            guard let (drv, _) = engine.makeFused(maxM: Swift.max(8, B), maxSeqLen: 128),
                  let batch = SeedlessLaneBatch(driver: drv, lanes: lanes) else {
                return lines.joined(separator: "\n") + "\n[lane-bench] driver nil\nLANEBENCH done"
            }
            var ms = 0.0, gpuMs = 0.0
            let reps = 30
            for r in 0 ..< reps {
                let toks = (0 ..< B).map { Int32(200 + r * 7 + $0 * 31) }
                let t0 = Date()
                _ = batch.forwardRowsBatch(engine.embed(tokens: toks))
                if r >= 5 {   // drop warmup
                    ms += Date().timeIntervalSince(t0) * 1000
                    gpuMs += SeedlessFusedVerify.SeedlessFusedForward.profLastGPUMs
                }
            }
            let step = ms / Double(reps - 5)
            lines.append(String(format: "  %4d  %8.2f  %8.1f tok/s  %8.1f tok/s   (gpu %.2f ms)",
                                B, step, 1000.0 / step, Double(B) * 1000.0 / step, gpuMs / Double(reps - 5)))
            // Full greedy step (embed→layers→head→argmax, 1 CB, int-only readback)
            // — the serve-path primitive; token feedback chains the trajectory.
            var ams = 0.0, agpu = 0.0
            var toks = (0 ..< B).map { Int32(300 + $0 * 13) }
            var argmaxOK = true
            for r in 0 ..< reps {
                let t0 = Date()
                guard let next = batch.stepArgmaxBatch(toks) else { argmaxOK = false; break }
                if r >= 5 {
                    ams += Date().timeIntervalSince(t0) * 1000
                    agpu += SeedlessFusedVerify.SeedlessFusedForward.profLastGPUMs
                }
                toks = next.map { Int32($0) }
            }
            if argmaxOK {
                let astep = ams / Double(reps - 5)
                lines.append(String(format: "        argmax %6.2f  %8.1f tok/s  %8.1f tok/s   (gpu %.2f ms)",
                                    astep, 1000.0 / astep, Double(B) * 1000.0 / astep, agpu / Double(reps - 5)))
            } else {
                lines.append("        argmax: stepArgmaxBatch nil (no head?)")
            }
        }
        return lines.joined(separator: "\n") + "\nLANEBENCH done"
    }

    // Lane-kernel micro-bench (Stage 1b go/no-go discriminator; GPU, no model):
    // per-dispatch cost of each PER-LANE sequence-coupled kernel at real dims,
    // hazard-chained like the real encoder (a shared output buffer WAW-serializes
    // consecutive dispatches, mirroring the encoder-wide barrier behaviour), with
    // state buffers cycling over a working set larger than the SLC so recurrence /
    // KV traffic hits DRAM. Splits the per-lane step cost into dispatch tax
    // (merge-able by B-lane kernels) vs state bandwidth (irreducible per lane).
    public static func laneKernelBench() -> String {
        guard let (device, queue) = SeedlessMetalForward.ensure(),
              SeedlessFusedVerify.ensureRowsAuxPipelines(),
              SeedlessFusedVerify.ensureGdnPipelines(),
              SeedlessFusedVerify.ensureAttnPipelines(),
              SeedlessFusedVerify.ensureWave3Pipelines(),
              SeedlessLaneBatch.compileCopy(device)
        else { return "[lane-kbench] pipeline ensure fail\nLANEKBENCH done" }
        let Hk = 16, Dk = 128, Hv = 32, Dv = 128, cK = 4
        let keyDim = Hk * Dk, valueDim = Hv * Dv, convDim = 2 * keyDim + valueDim
        let nH = 16, nKV = 2, hD = 256, qd2 = 2 * hD, ropeDim = 64
        let ctx = Tell.envInt("QWISP_LANE_CTX", 1024)
        let eps: Float = 1e-6, ropeBase: Float = 1e7
        func buf(_ n: Int) -> MTLBuffer { device.makeBuffer(length: n, options: .storageModeShared)! }
        let N = 300                      // dispatches per measurement (~10 steps of 30)
        let nSets = 16                   // state working set: 16×8.4MB rec = 134MB ≫ SLC

        // shared (read or chained-write) buffers
        let qB = buf(keyDim * 2), kB = buf(keyDim * 2), vB = buf(valueDim * 2)
        let gB = buf(Hv * 4), betaB = buf(Hv * 4), yB = buf(valueDim * 2)
        let qkvB = buf(convDim * 2), convOutB = buf(convDim * 2), convW = buf(convDim * cK * 4)
        let qOutB = buf(nH * qd2 * 2), qNormB = buf(hD * 2), qRotB = buf(nH * hD * 2)
        let kOutB = buf(nKV * hD * 2), kNormB = buf(hD * 2), kRotB = buf(nKV * hD * 2)
        let vSrcB = buf(nKV * hD * 2), attnOutB = buf(nH * hD * 2)
        // cycling per-"lane-layer" state sets
        let recIn = (0 ..< nSets).map { _ in buf(Hv * Dv * Dk * 4) }
        let recOut = (0 ..< nSets).map { _ in buf(Hv * Dv * Dk * 4) }
        let hists = (0 ..< nSets).map { _ in buf(cK * convDim * 2) }
        let histOuts = (0 ..< nSets).map { _ in buf(cK * convDim * 2) }
        let kCaches = (0 ..< nSets).map { _ in buf(nKV * ctx * hD * 2) }
        let vCaches = (0 ..< nSets).map { _ in buf(nKV * ctx * hD * 2) }
        let copyA = buf(256 * 2), copyBf = buf(256 * 2)

        func timeCB(_ body: (MTLComputeCommandEncoder, Int) -> Void) -> Double {
            let cb = queue.makeCommandBuffer()!
            let enc = cb.makeComputeCommandEncoder()!
            for i in 0 ..< N { body(enc, i) }
            enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
            return (cb.gpuEndTime - cb.gpuStartTime) * 1000.0
        }
        // measure twice (clock ramp), keep the second; report µs/dispatch
        func us(_ body: @escaping (MTLComputeCommandEncoder, Int) -> Void) -> Double {
            _ = timeCB(body)
            return timeCB(body) * 1000.0 / Double(N)
        }

        let copyUs = us { enc, _ in
            // dispatch+barrier floor: tiny chained copy (WAW on copyBf)
            SeedlessLaneBatch.encodeRowCopyStatic(enc, src: copyA, srcOff: 0, dst: copyBf, dstOff: 0, count: 256)
        }
        let convUs = us { enc, i in
            SeedlessFusedVerify.encodeGdnFusionConvShift(enc, hist: hists[i % nSets], qkv: qkvB,
                                                         w: convW, convOut: convOutB, histOut: histOuts[i % nSets],
                                                         M: 1, K: cK, C: convDim)
        }
        let recUs = us { enc, i in
            SeedlessFusedVerify.encodeGatedDeltaStepRows(enc, q: qB, k: kB, v: vB, g: gB, beta: betaB,
                                                         stateIn: recIn[i % nSets], stateOut: recOut[i % nSets],
                                                         y: yB, T: 1, B: 1, Hv: Hv, Dv: Dv)
        }
        let recHotUs = us { enc, _ in
            // same kernel, ONE state pair reused → SLC-hot: the DRAM-vs-latency split
            SeedlessFusedVerify.encodeGatedDeltaStepRows(enc, q: qB, k: kB, v: vB, g: gB, beta: betaB,
                                                         stateIn: recIn[0], stateOut: recOut[0],
                                                         y: yB, T: 1, B: 1, Hv: Hv, Dv: Dv)
        }
        let qPrepUs = us { enc, _ in
            SeedlessFusedVerify.encodeAttnQPrepRows(enc, qOut: qOutB, qNorm: qNormB, qRot: qRotB,
                                                    qd2: qd2, headDim: hD, ropeDim: ropeDim, base: ropeBase,
                                                    startOffset: ctx - 1, numHeads: nH, M: 1, eps: eps)
        }
        let kPrepUs = us { enc, i in
            SeedlessFusedVerify.encodeAttnKPrepRows(enc, kOut: kOutB, kNorm: kNormB, kRot: kRotB,
                                                    kCache: kCaches[i % nSets], headDim: hD, ropeDim: ropeDim,
                                                    base: ropeBase, startOffset: ctx - 1, numKV: nKV,
                                                    maxLen: ctx, M: 1, eps: eps)
        }
        let wkvUs = us { enc, i in
            SeedlessFusedVerify.encodeWriteKVRows(enc, src: vSrcB, cache: vCaches[i % nSets],
                                                  KV: nKV, D: hD, maxLen: ctx, pos: ctx - 1, M: 1)
        }
        let sdpaUs = us { enc, i in
            SeedlessFusedVerify.encodeSdpaRows(enc, q: qRotB, k: kCaches[i % nSets], v: vCaches[i % nSets],
                                               out: attnOutB, H: nH, KV: nKV, D: hD,
                                               baseLenPlus1: ctx, M: 1,
                                               scale: Float(pow(Double(hD), -0.5)), maxLen: ctx)
        }

        let gdnLane = 30.0 * (convUs + recUs)
        let attnLane = 10.0 * (qPrepUs + kPrepUs + wkvUs + sdpaUs)
        let perLaneMs = (gdnLane + attnLane) / 1000.0
        let dispatches = 30.0 * 2 + 10.0 * 4
        let taxMs = dispatches * copyUs / 1000.0
        let recDramMs = 30.0 * Swift.max(0, recUs - recHotUs) / 1000.0
        let lines = ["[lane-kbench] per-dispatch µs at real dims, hazard-chained, ctx=\(ctx), N=\(N), sets=\(nSets)",
                     String(format: "  copy(floor) %6.1f | conv %6.1f | recur %6.1f (hot %6.1f) | qprep %6.1f | kprep %6.1f | wkv %6.1f | sdpa@%d %6.1f",
                            copyUs, convUs, recUs, recHotUs, qPrepUs, kPrepUs, wkvUs, ctx, sdpaUs),
                     String(format: "  per-lane/step: GDN 30×(conv+recur)=%.2fms + attn 10×(4)=%.2fms = %.2fms",
                            gdnLane / 1000, attnLane / 1000, perLaneMs),
                     String(format: "  split: dispatch-tax(%.0f × copy-floor)=%.2fms | recur DRAM excess=%.2fms | rest=%.2fms",
                            dispatches, taxMs, recDramMs, perLaneMs - taxMs - recDramMs),
                     String(format: "  Stage-1b ceiling @B=3 (tax/3, bandwidth kept): %.2fms/lane vs now %.2fms",
                            perLaneMs - taxMs * 2.0 / 3.0, perLaneMs)]

        // ── projection-kernel M-scaling probe: does qmm4_rows (qmv, per-row weight
        // reads) vs qmm4_tiled (threadgroup-shared dequant) flip at small M for
        // LAYER-shaped matrices? This is the residual B-scaling lever: the M=B
        // projection dispatches dominate the per-lane increment if weights re-read per row.
        guard SeedlessMetalForward._qmm4TiledPipeline != nil || {
            let x = MLXRandom.normal([1, 512]).asType(.float16)
            let wf = MLXRandom.normal([8, 512]).asType(.float16)
            let (wq, s, b) = MLX.quantized(wf, groupSize: 64, bits: 4, mode: .affine)
            MLX.eval([x, wq, s, b!])
            return SeedlessMetalForward.qmmTiled(x, wq, scales: s, biases: b!, M: 1, K: 512, N: 8) != nil
        }() else { return lines.joined(separator: "\n") + "\nLANEKBENCH done" }
        let pK = 2048, pN = 8192                      // GDN in-proj shape [8192, 2048]
        let pw = buf(pN * pK / 2), ps = buf(pN * (pK / 64) * 2), pb = buf(pN * (pK / 64) * 2)
        var projLines: [String] = ["  proj [N=\(pN),K=\(pK)] µs/dispatch: M → rows | tiled"]
        for M in [1, 2, 3, 4, 8] {
            let px = buf(M * pK * 2), po = buf(M * pN * 2)
            let rowsUs = us { enc, _ in
                SeedlessFusedVerify.encodeQmmRows(enc, w: pw, scales: ps, biases: pb, x: px, out: po, M: M, K: pK, N: pN)
            }
            let tiledUs = us { enc, _ in
                let qp = SeedlessMetalForward._qmm4TiledPipeline!
                enc.setComputePipelineState(qp)
                enc.setBuffer(pw, offset: 0, index: 0); enc.setBuffer(ps, offset: 0, index: 1)
                enc.setBuffer(pb, offset: 0, index: 2); enc.setBuffer(px, offset: 0, index: 3)
                enc.setBuffer(po, offset: 0, index: 4)
                var kk = Int32(pK), nn = Int32(pN), mm = Int32(M)
                enc.setBytes(&kk, length: 4, index: 5); enc.setBytes(&nn, length: 4, index: 6); enc.setBytes(&mm, length: 4, index: 7)
                enc.dispatchThreadgroups(MTLSize(width: pN, height: 1, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            }
            projLines.append(String(format: "    M=%d  %7.1f | %7.1f", M, rowsUs, tiledUs))
        }

        // ── qmm4_rows_b (B-row qmv): correctness (byte vs qmm4_rows, random data)
        // + speed. If B=3 ≪ 3× the M=1 rows cost, the projection lever is real.
        var bLines: [String] = []
        if SeedlessLaneBatch.compileQmmB(device) {
            let xa = MLXRandom.normal([4, pK]).asType(.float16)
            let wf = MLXRandom.normal([pN, pK]).asType(.float16)
            let (wq, s0, b0) = MLX.quantized(wf, groupSize: 64, bits: 4, mode: .affine)
            let sc = s0.asType(.float16), bi = b0!.asType(.float16)
            MLX.eval([xa, wq, sc, bi])
            if let bx = SeedlessMetalForward.mtlBuf(xa, device),
               let bw = SeedlessMetalForward.mtlBuf(wq, device),
               let bs = SeedlessMetalForward.mtlBuf(sc, device),
               let bbq = SeedlessMetalForward.mtlBuf(bi, device) {
                let oA = buf(4 * pN * 2), oB = buf(4 * pN * 2)
                let ccb = queue.makeCommandBuffer()!
                let cenc = ccb.makeComputeCommandEncoder()!
                SeedlessFusedVerify.encodeQmmRows(cenc, w: bw, scales: bs, biases: bbq, x: bx, out: oA, M: 4, K: pK, N: pN)
                SeedlessLaneBatch.encodeQmmRowsB(cenc, w: bw, scales: bs, biases: bbq, x: bx, xOff: 0, out: oB, outOff: 0, B: 4, K: pK, N: pN)
                cenc.endEncoding(); ccb.commit(); ccb.waitUntilCompleted()
                let same = memcmp(oA.contents(), oB.contents(), 4 * pN * 2) == 0
                withExtendedLifetime([xa, wq, sc, bi]) {}
                bLines.append("  qmm4_rows_b vs qmm4_rows byte-compare (B=4, random): \(same ? "PASS" : "FAIL")")
            }
            for M in [1, 2, 3, 4] {
                let px = buf(M * pK * 2), po = buf(M * pN * 2)
                let bUs = us { enc, _ in
                    SeedlessLaneBatch.encodeQmmRowsB(enc, w: pw, scales: ps, biases: pb,
                                                     x: px, xOff: 0, out: po, outOff: 0, B: M, K: pK, N: pN)
                }
                bLines.append(String(format: "    rows_b B=%d  %7.1f", M, bUs))
            }
        } else {
            bLines.append("  qmm4_rows_b: compile FAIL")
        }
        return (lines + projLines + bLines).joined(separator: "\n") + "\nLANEKBENCH done"
    }

    // Long-context decay probe (#117 follow-up): the production log shows prefill chunk rate
    // decaying 348→66 tok/s over 0→40K and decode collapsing 45→7 tok/s at 48K. This isolates
    // WHY, per stage, using forwardRowsProfiled (exact per-encoder GPU ms bucketed GDN/attn/MoE).
    // One long prefill builds the context; at checkpoints it also runs M=1 forwards (= decode
    // step cost at that context). If attn ms scales ~linearly with position while GDN/MoE stay
    // flat, the driver is full-attention O(N) reads (fundamental; only the 10 full-attn layers
    // grow) — not a fixable constant. Env: QWISP_DECAY_MAX (default 49152).
    public static func longContextDecayProbe(modelDir: String) -> String {
        guard let store = try? WeightStore(modelDir: modelDir) else { return "[decay] load fail\nDECAY done" }
        store.residentAll()
        let engine = SeedlessEngine.build(store: store)
        let maxCtx = Tell.envInt("QWISP_DECAY_MAX", 49152)
        let chunk = 1024
        guard let (fwd, _) = engine.makeFused(maxM: chunk + 8, maxSeqLen: maxCtx + 128) else {
            return "[decay] makeFused nil\nDECAY done"
        }
        let checkpoints = [4096, 8192, 16384, 32768, 49152].filter { $0 <= maxCtx }
        var lines = ["[decay] maxCtx=\(maxCtx) chunk=\(chunk) resident/fused — per-stage GPU ms (exact)",
                     "  ── prefill: per-chunk stage ms by context position ──",
                     "     posStart   attn_ms   gdn_ms   moe_ms   chunk_ms   tok/s"]
        func tok(_ n: Int, _ salt: Int) -> [Int32] { (0..<n).map { Int32((($0 &* 7 &+ salt) % 5000) + 100) } }
        let prompt = tok(maxCtx, 13)
        var pos = 0
        let sampleAt = Set([0, 1024, 2048, 4096, 8192, 16384, 24576, 32768, 40960, 47104])
        var decodeRows: [String] = []
        while pos < maxCtx {
            let end = Swift.min(pos + chunk, maxCtx)
            let x = engine.embed(tokens: Array(prompt[pos ..< end]))
            guard let t = fwd.forwardRowsProfiled(x, M: end - pos) else { break }
            if sampleAt.contains(pos) {
                let cm = t.gdn + t.attn + t.moe
                lines.append(String(format: "     %8d  %8.2f %8.2f %8.2f  %8.2f  %6.0f",
                                    pos, t.attn, t.gdn, t.moe, cm, Double(end - pos) / (cm / 1000)))
            }
            pos = end
            // Decode-step cost (M=1) at this context depth — average 5, skip warmup.
            if checkpoints.contains(pos) {
                var a = 0.0, g = 0.0, m = 0.0; let reps = 6
                for r in 0..<reps {
                    let x1 = engine.embed(tokens: [prompt[pos % maxCtx]])
                    guard let t1 = fwd.forwardRowsProfiled(x1, M: 1) else { break }
                    if r > 0 { a += t1.attn; g += t1.gdn; m += t1.moe }   // drop rep 0 (warmup)
                }
                let n = Double(reps - 1); let step = (a + g + m) / n
                decodeRows.append(String(format: "     %8d  %8.3f %8.3f %8.3f  %8.3f  %6.1f",
                                         pos, a/n, g/n, m/n, step, 1000.0 / step))
            }
        }
        lines.append("  ── decode (M=1) step stage ms by context length ──")
        lines.append("     context    attn_ms   gdn_ms   moe_ms   step_ms   tok/s")
        lines.append(contentsOf: decodeRows)
        return lines.joined(separator: "\n") + "\nDECAY done"
    }

    // Spec-verify width probe (#117 follow-up): the decay probe showed the RAW M=1 forward at
    // 49K is ~36 tok/s, yet the production log's spec-decode collapses to 6-7 tok/s there. The
    // suspect is the verify forward: full-attention is O(M·N), so a K-token draft costs ~K× the
    // attention of a single row at that context. This prefills to QWISP_SPEC_CTX (default 32768)
    // then sweeps M (verify width) measuring per-stage forward ms. If attn ms scales ~linearly
    // with M while GDN/MoE stay flat, long drafts are counterproductive at long context and the
    // fix is a context-aware draft-length cap (spec loop, additive, lossless). Measure-first.
    public static func specWidthProbe(modelDir: String) -> String {
        guard let store = try? WeightStore(modelDir: modelDir) else { return "[spec-width] load fail\nSPECWIDTH done" }
        store.residentAll()
        let engine = SeedlessEngine.build(store: store)
        let ctx = Tell.envInt("QWISP_SPEC_CTX", 32768)
        let maxM = 32
        guard let (fwd, _) = engine.makeFused(maxM: maxM, maxSeqLen: ctx + maxM + 8) else {
            return "[spec-width] makeFused nil\nSPECWIDTH done"
        }
        func tok(_ n: Int, _ salt: Int) -> [Int32] { (0..<n).map { Int32((($0 &* 7 &+ salt) % 5000) + 100) } }
        // Prefill to `ctx` in chunks of 1024 (build the KV context).
        let pre = tok(ctx, 13); var pos = 0
        while pos < ctx {
            let end = Swift.min(pos + 1024, ctx)
            _ = fwd.forwardRowsProfiled(engine.embed(tokens: Array(pre[pos ..< end])), M: end - pos)
            pos = end
        }
        var lines = ["[spec-width] ctx=\(ctx) resident/fused — forward stage ms by verify width M",
                     "     M    attn_ms   gdn_ms   moe_ms   step_ms  attn/M   (draft cost model)"]
        // Sweep M; each M runs a few reps (dropping warmup). forwardRowsProfiled appends to KV,
        // but we only need the timing — the small KV growth (a few hundred rows total) is noise
        // against a 32K context.
        for M in [1, 2, 4, 8, 16] {
            var a = 0.0, g = 0.0, m = 0.0; let reps = 5
            for r in 0..<reps {
                let x = engine.embed(tokens: tok(M, 100 + r))
                guard let t = fwd.forwardRowsProfiled(x, M: M) else { break }
                if r > 0 { a += t.attn; g += t.gdn; m += t.moe }
            }
            let n = Double(reps - 1)
            lines.append(String(format: "  %4d  %8.3f %8.3f %8.3f %8.3f %8.3f",
                                M, a/n, g/n, m/n, (a+g+m)/n, (a/n)/Double(M)))
        }
        return lines.joined(separator: "\n") + "\nSPECWIDTH done"
    }

    // Prefill throughput probe: measures tok/s prefilling a fixed prompt at several chunk sizes.
    // If larger chunks are much faster → per-forward overhead dominates (raise the chunk).
    // If flat → per-token compute is the limit (kernel-level work). Measure-before-implement.
    public static func prefillThroughputProbe(modelDir: String) -> String {
        guard let store = try? WeightStore(modelDir: modelDir) else { return "[prefill-probe] load fail" }
        store.residentAll()
        let engine = SeedlessEngine.build(store: store)
        let promptLen = 4096
        let prompt = (0..<promptLen).map { Int32((($0 &* 7 &+ 13) % 5000) + 100) }
        var lines = ["[prefill-probe] promptLen=\(promptLen), resident"]
        for chunk in [64, 128, 256, 512] {
            guard let b = Tell.fusedBackend(engine: engine, maxM: chunk + 8, maxSeqLen: promptLen + 128) else {
                lines.append("  chunk=\(chunk): backend nil"); continue
            }
            let t0 = Date()
            var pos = 0
            while pos < promptLen {
                let end = Swift.min(pos + chunk, promptLen)
                _ = b.forward(Array(prompt[pos ..< end]))
                pos = end
            }
            let dt = Date().timeIntervalSince(t0)
            lines.append(String(format: "  chunk=%4d: %.3fs  %.0f tok/s", chunk, dt, Double(promptLen) / dt))
        }
        return lines.joined(separator: "\n") + "\nPREFILLPROBE done"
    }

    // Prefill component breakdown: differential timing via profSkip flags on the raw forward.
    // Runs a full prefill, then repeats with each component skipped; cost(X) ≈ t_full - t_skipX.
    // Timing-only (skipped output is garbage) — attributes the cold-prefill wall to GDN-recur /
    // GDN-matmul / attention / MoE-experts / routing / floor, so we know which lever (if any) pays.
    // Env: QWISP_PREFILL_LEN (default 8192), QWISP_PREFILL_CHUNK (default 64).
    public static func prefillBreakdownProbe(modelDir: String) -> String {
        guard let store = try? WeightStore(modelDir: modelDir) else { return "[prefill-bd] load fail\nPREFILLBD done" }
        store.residentAll()
        let engine = SeedlessEngine.build(store: store)
        let promptLen = Tell.envInt("QWISP_PREFILL_LEN", 8192)
        let chunk = Tell.envInt("QWISP_PREFILL_CHUNK", 64)
        let prompt = (0..<promptLen).map { Int32((($0 &* 7 &+ 13) % 5000) + 100) }

        // reset all skip flags to a known-off baseline
        func allOff() {
            SeedlessMetalForward.profSkipGDNMatmul = false
            SeedlessMetalForward.profSkipGDNRecur = false
            SeedlessMetalForward.profSkipMixer = false
            SeedlessMetalForward.profSkipMoEExperts = false
            SeedlessMetalForward.profSkipMoERouted = false
            SeedlessMetalForward.profSkipMoEShared = false
            SeedlessMetalForward.profSkipSingleThread = false
        }
        // one full prefill of `prompt` at `chunk`, returns wall seconds. Fresh backend each run so
        // KV starts empty (identical work every config).
        func runPrefill() -> Double {
            guard let b = Tell.fusedBackend(engine: engine, maxM: chunk + 8, maxSeqLen: promptLen + 128) else { return -1 }
            let t0 = Date()
            var pos = 0
            while pos < promptLen {
                let end = Swift.min(pos + chunk, promptLen)
                _ = b.forward(Array(prompt[pos ..< end]))
                pos = end
            }
            return Date().timeIntervalSince(t0)
        }

        // v2: split WALL vs GPU-busy (profLastGPUMs per CB) per config. The decisive question:
        // is prefill time GPU execution (kernel-level, capped ~22% by the skip test) or CPU-side
        // gap (MLX embed eval + hBuf upload + normed readback + commit/wait per chunk) — the
        // latter is a scheduling prize, lossless by construction (no math change).
        func runAt(_ c: Int) -> (wall: Double, gpu: Double) {
            guard let bk = Tell.fusedBackend(engine: engine, maxM: c + 8, maxSeqLen: promptLen + 128) else { return (-1, 0) }
            var gpu = 0.0
            let t0 = Date(); var pos = 0
            while pos < promptLen {
                let e = Swift.min(pos + c, promptLen)
                _ = bk.forward(Array(prompt[pos ..< e]))
                gpu += SeedlessFusedVerify.SeedlessFusedForward.profLastGPUMs / 1000.0
                pos = e
            }
            return (Date().timeIntervalSince(t0), gpu)
        }
        func floorSet() {   // skip ALL heavy GPU compute → what remains is the irreducible floor
            SeedlessMetalForward.profSkipMixer = true          // GDN body + attention body
            SeedlessMetalForward.profSkipMoEExperts = true     // routed+shared+combine gather
            SeedlessMetalForward.profSkipSingleThread = true   // route_top8/shared_gate8
        }
        allOff(); _ = runAt(chunk)                             // warmup: compile all pipelines
        _ = runPrefill                                          // (kept for API symmetry)

        var lines = ["[prefill-bd] promptLen=\(promptLen) resident  — wall vs GPU-busy per config"]
        func row(_ label: String, _ c: Int, floored: Bool) -> Double {
            allOff(); if floored { floorSet() }
            let a = runAt(c), b = runAt(c)
            allOff()
            let r = a.wall < b.wall ? a : b
            lines.append(String(format: "  %@ chunk=%4d   wall %6.1fs   gpu %6.1fs   cpu-gap %6.1fs (%4.1f%%)   %.0f tok/s",
                                label, c, r.wall, r.gpu, r.wall - r.gpu, 100 * (r.wall - r.gpu) / r.wall,
                                Double(promptLen) / r.wall))
            return r.wall
        }
        _ = row("FULL ", chunk, floored: false)
        _ = row("FULL ", 256, floored: false)
        _ = row("floor", chunk, floored: true)
        _ = row("floor", 256, floored: true)
        _ = row("floor", 1024, floored: true)
        lines.append("PREFILLBD done")
        return lines.joined(separator: "\n")
    }

    // End-to-end lossless gate for the Design-B + multi-slot prefix cache: drives SeedlessBackend.generate
    // through a scripted request sequence with the cache ON (warm: reuse/extend/cross-branch/reset) and
    // compares each token stream to the same request generated with the cache OFF (segmented cold path).
    // Byte-identical everywhere ⇒ the multi-slot restore + stride re-prefill + arena growth are lossless.
    public static func prefixCacheE2E(modelDir: String) -> String {
        // #76: QWISP_PREFIX_E2E_C=<c> runs this same gate on a strict-streaming backend
        // (unset/0 = resident, the original gate). Streaming cached mode requires the
        // lossless (strict) posture — bolt keeps the segmented path by design.
        let ec = Tell.envInt("QWISP_PREFIX_E2E_C", 0)
        let tier: SeedlessTier = ec > 0 ? .streaming(c: ec) : .auto
        guard let backend = try? SeedlessBackend(modelDir: modelDir, tier: tier) else { return "[prefix-e2e] load fail\nPREFIXE2E FAIL" }
        if ec > 0 { backend.losslessForced = true }
        // stride small so a shared 256-token prefix produces reusable sub-content boundaries.
        setenv("QWISP_PREFIX_SNAP_STRIDE", "64", 1)

        func toks(_ n: Int, _ salt: Int) -> [Int] { (0..<n).map { (($0 &* 7 &+ salt) % 5000) + 100 } }
        let A = toks(256, 13)                       // shared "system+tools" prefix
        // (content, contentLen) per request; full prompt = content + 8-token gen-prompt suffix.
        func req(_ content: [Int]) -> (prompt: [Int], cl: Int) { (content + toks(8, 777), content.count) }
        let c1 = A + toks(32, 1)                     // R1: A + userX
        let c2 = c1 + toks(40, 2)                    // R2: extends R1 (intra-conversation)
        let c3 = A + toks(32, 3)                     // R3: A + userZ (cross-conversation: shares A)
        let c4 = toks(200, 999)                      // R4: shares nothing → cold reset (restore empty)
        let reqs = [req(c1), req(c2), req(c3), req(c4)]
        let maxTok = 24

        final class Box: @unchecked Sendable { var v: [Int] = [] }
        func gen(_ p: [Int], _ cl: Int) -> [Int] {
            let box = Box()
            let sem = DispatchSemaphore(value: 0)
            let stream = backend.generate(p, options: GenerateOptions(maxTokens: maxTok, promptContentLen: cl))
            Task { for await t in stream { box.v.append(t) }; sem.signal() }
            sem.wait()
            return box.v
        }

        // Reference: cache OFF (segmented cold path) — each request independent.
        backend.prefixCacheForced = false
        let ref = reqs.map { gen($0.prompt, $0.cl) }
        // Cached: cache ON, cold start, requests in sequence so the cache warms/reuses across them.
        backend.prefixCacheForced = true
        backend.resetPrefixCache()
        let got = reqs.map { gen($0.prompt, $0.cl) }

        var lines = ["[prefix-e2e] 4 requests (reuse/extend/cross-branch/reset), maxTok=\(maxTok), tier=\(ec > 0 ? "streaming C=\(ec)" : "resident")"]
        var pass = true
        let labels = ["R1 cold      ", "R2 extend    ", "R3 cross-conv", "R4 reset     "]
        for i in 0..<reqs.count {
            let ok = ref[i] == got[i]
            pass = pass && ok
            lines.append("  \(labels[i])  ref=\(ref[i].count)tok  cached=\(got[i].count)tok  \(ok ? "IDENTICAL ✅" : "DIVERGE ❌")")
            if !ok { lines.append("    ref   : \(ref[i].prefix(12))"); lines.append("    cached: \(got[i].prefix(12))") }
        }
        lines.append("PREFIXE2E \(pass ? "PASS" : "FAIL")")
        return lines.joined(separator: "\n")
    }

    // #89 gate: PREFIXE2E with a simulated process restart. R1 warms the cache and persists
    // the content boundary to disk; resetPrefixCache() is the fresh-process equivalent (arena,
    // providers, in-memory slots all dropped — only the DISK store survives); R2 (extends R1)
    // must then restore from disk and produce a stream byte-identical to the cache-off path.
    // restoreHits asserts the disk restore actually fired (byte-identity alone can't tell
    // "restored" from "silently re-prefilled cold"). QWISP_PREFIX_E2E_C as in prefixCacheE2E.
    public static func prefixPersistE2E(modelDir: String) -> String {
        let ec = Tell.envInt("QWISP_PREFIX_E2E_C", 0)
        let tier: SeedlessTier = ec > 0 ? .streaming(c: ec) : .auto
        guard let backend = try? SeedlessBackend(modelDir: modelDir, tier: tier) else { return "[prefix-persist-e2e] load fail\nPREFIXPERSISTE2E FAIL" }
        if ec > 0 { backend.losslessForced = true }
        let store = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("qwisp-prefix-persist-e2e-\(ProcessInfo.processInfo.processIdentifier)")
        setenv("QWISP_PREFIX_PERSIST_DIR", store.path, 1)
        defer { unsetenv("QWISP_PREFIX_PERSIST_DIR"); try? FileManager.default.removeItem(at: store) }

        func toks(_ n: Int, _ salt: Int) -> [Int] { (0..<n).map { (($0 &* 7 &+ salt) % 5000) + 100 } }
        let c1 = toks(256, 13) + toks(32, 1)
        let c2 = c1 + toks(40, 2)                    // continues the conversation across the "restart"
        func req(_ content: [Int]) -> (prompt: [Int], cl: Int) { (content + toks(8, 777), content.count) }
        let (p1, cl1) = req(c1), (p2, cl2) = req(c2)
        let maxTok = 24

        final class Box: @unchecked Sendable { var v: [Int] = [] }
        func gen(_ p: [Int], _ cl: Int) -> [Int] {
            let box = Box()
            let sem = DispatchSemaphore(value: 0)
            let stream = backend.generate(p, options: GenerateOptions(maxTokens: maxTok, promptContentLen: cl))
            Task { for await t in stream { box.v.append(t) }; sem.signal() }
            sem.wait()
            return box.v
        }

        // Reference: cache OFF (segmented cold path).
        backend.prefixCacheForced = false
        let ref2 = gen(p2, cl2)
        // Persisted run: R1 (cache on + persist writes disk async) → wait for the file →
        // reset (= restart) → R2 restores from disk.
        setenv("QWISP_PREFIX_PERSIST", "1", 1)
        defer { unsetenv("QWISP_PREFIX_PERSIST") }
        backend.prefixCacheForced = true
        backend.resetPrefixCache()
        _ = gen(p1, cl1)
        var waited = 0.0
        while waited < 15 {                          // async save → poll for the artifact
            if let n = try? FileManager.default.contentsOfDirectory(atPath: store.path).count, n > 0 { break }
            Thread.sleep(forTimeInterval: 0.2); waited += 0.2
        }
        backend.resetPrefixCache()                   // "restart": only the disk store survives
        let hits0 = PrefixPersist.restoreHits
        let got2 = gen(p2, cl2)
        let restored = PrefixPersist.restoreHits > hits0
        let identical = got2 == ref2
        var lines = ["[prefix-persist-e2e] tier=\(ec > 0 ? "streaming C=\(ec)" : "resident"), maxTok=\(maxTok)"]
        lines.append("  restart-continue  ref=\(ref2.count)tok  restored=\(got2.count)tok  \(identical ? "IDENTICAL ✅" : "DIVERGE ❌")  diskRestore=\(restored ? "HIT" : "MISS ❌")")
        if !identical { lines.append("    ref: \(ref2.prefix(12))"); lines.append("    got: \(got2.prefix(12))") }
        lines.append("PREFIXPERSISTE2E \(identical && restored ? "PASS" : "FAIL")")
        return lines.joined(separator: "\n")
    }

    // #112 gate: stable-prefix persist. Conversation A warms the arena; conversation B
    // (same ≥1024-token "harness prefix", different tail) diverges → the recurrence
    // trigger persists the shared restore point ONCE to disk. resetPrefixCache() (=
    // process restart: arena, slots, and the RAM tier all die) then conversation C with
    // the same prefix must warm-start from the DISK entry (restoreHits asserts the
    // restore fired) and stream byte-identical to the cache-off path. Also asserts the
    // store holds exactly ONE file (recurrence writes once — no per-turn wear).
    public static func prefixStableE2E(modelDir: String) -> String {
        let ec = Tell.envInt("QWISP_PREFIX_E2E_C", 0)
        let tier: SeedlessTier = ec > 0 ? .streaming(c: ec) : .auto
        guard let backend = try? SeedlessBackend(modelDir: modelDir, tier: tier) else { return "[prefix-stable-e2e] load fail\nPREFIXSTABLEE2E FAIL" }
        if ec > 0 { backend.losslessForced = true }
        let store = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("qwisp-prefix-stable-e2e-\(ProcessInfo.processInfo.processIdentifier)")
        setenv("QWISP_PREFIX_PERSIST_DIR", store.path, 1)
        setenv("QWISP_PREFIX_SNAP_STRIDE", "512", 1)   // slots inside the 1200-token prefix
        defer { unsetenv("QWISP_PREFIX_PERSIST_DIR"); try? FileManager.default.removeItem(at: store) }

        func toks(_ n: Int, _ salt: Int) -> [Int] { (0..<n).map { (($0 &* 7 &+ salt) % 5000) + 100 } }
        let shared = toks(1200, 13)                    // ≥ stableMinTokens at a 1024 slot boundary
        let cA = shared + toks(48, 1)
        let cB = shared + toks(48, 2)                  // recurrence: same prefix, new conversation
        let cC = shared + toks(48, 3)                  // post-"restart" conversation
        func req(_ content: [Int]) -> (prompt: [Int], cl: Int) { (content + toks(8, 777), content.count) }
        let maxTok = 24

        final class Box: @unchecked Sendable { var v: [Int] = [] }
        func gen(_ p: [Int], _ cl: Int) -> [Int] {
            let box = Box()
            let sem = DispatchSemaphore(value: 0)
            let stream = backend.generate(p, options: GenerateOptions(maxTokens: maxTok, promptContentLen: cl))
            Task { for await t in stream { box.v.append(t) }; sem.signal() }
            sem.wait()
            return box.v
        }
        func binCount() -> Int {
            (try? FileManager.default.contentsOfDirectory(atPath: store.path).filter { $0.hasSuffix(".bin") }.count) ?? 0
        }

        backend.prefixCacheForced = false
        let refs = [req(cA), req(cB), req(cC)].map { gen($0.prompt, $0.cl) }
        backend.prefixCacheForced = true
        backend.resetPrefixCache()
        let gotA = gen(req(cA).prompt, req(cA).cl)                 // cold, warms the arena path
        let filesAfterA = binCount()
        let gotB = gen(req(cB).prompt, req(cB).cl)                 // divergence → stable persist fires
        var waited = 0.0                                            // async write → poll
        while waited < 15, binCount() == 0 { Thread.sleep(forTimeInterval: 0.2); waited += 0.2 }
        let filesAfterB = binCount()
        backend.resetPrefixCache()                                  // "restart": only the disk survives
        let hits0 = PrefixPersist.restoreHits
        let gotC = gen(req(cC).prompt, req(cC).cl)
        let diskRestored = PrefixPersist.restoreHits > hits0
        let filesAfterC = binCount()

        var lines = ["[prefix-stable-e2e] shared=1200tok, stride=512, tier=\(ec > 0 ? "streaming C=\(ec)" : "resident")"]
        var pass = true
        func check(_ name: String, _ ok: Bool, _ detail: String) {
            pass = pass && ok
            lines.append("  \(name)  \(detail)  \(ok ? "✅" : "❌")")
        }
        check("R1 A cold      ", refs[0] == gotA && filesAfterA == 0, "identical=\(refs[0] == gotA) files=\(filesAfterA) (want 0 — one conversation is no evidence)")
        check("R2 B recurrence", refs[1] == gotB && filesAfterB == 1, "identical=\(refs[1] == gotB) files=\(filesAfterB) (want 1 — persisted once)")
        check("R3 C restart   ", refs[2] == gotC && diskRestored, "identical=\(refs[2] == gotC) diskRestore=\(diskRestored ? "HIT" : "MISS")")
        check("write-once     ", filesAfterC == 1, "files=\(filesAfterC) (want 1 — no per-turn growth)")
        lines.append("PREFIXSTABLEE2E \(pass ? "PASS" : "FAIL")")
        return lines.joined(separator: "\n")
    }

    // #117 gate: PREFIXE2E for the RAM tier. Interleaved conversations — A cold, B cold
    // (unrelated → replaces A's arena path), then A extended: the third request must
    // restore A's whole-conversation state from the RAM store (prefixRAMHits asserts the
    // restore actually fired) and stream byte-identical to the cache-off path. A fourth
    // request extends the third: the in-path slot now ties the RAM entry, so the gain
    // guard (>512 tok) must keep the cheap in-path restore (hits stays 1).
    public static func prefixRAME2E(modelDir: String) -> String {
        let ec = Tell.envInt("QWISP_PREFIX_E2E_C", 0)
        let tier: SeedlessTier = ec > 0 ? .streaming(c: ec) : .auto
        guard let backend = try? SeedlessBackend(modelDir: modelDir, tier: tier) else { return "[prefix-ram-e2e] load fail\nPREFIXRAME2E FAIL" }
        if ec > 0 { backend.losslessForced = true }
        setenv("QWISP_PREFIX_RAM_MB", "512", 1)      // force the tier ON (streaming defaults OFF)
        defer { unsetenv("QWISP_PREFIX_RAM_MB") }

        func toks(_ n: Int, _ salt: Int) -> [Int] { (0..<n).map { (($0 &* 7 &+ salt) % 5000) + 100 } }
        let cA = toks(700, 1)                        // conv A (700 > the 512-tok gain guard)
        let cB = toks(700, 2)                        // conv B, no shared prefix
        let cA2 = cA + toks(40, 3)                   // A continues after the switch
        let cA3 = cA2 + toks(40, 4)                  // A continues in-path (RAM must NOT fire)
        func req(_ content: [Int]) -> (prompt: [Int], cl: Int) { (content + toks(8, 777), content.count) }
        let reqs = [req(cA), req(cB), req(cA2), req(cA3)]
        let maxTok = 24

        final class Box: @unchecked Sendable { var v: [Int] = [] }
        func gen(_ p: [Int], _ cl: Int) -> [Int] {
            let box = Box()
            let sem = DispatchSemaphore(value: 0)
            let stream = backend.generate(p, options: GenerateOptions(maxTokens: maxTok, promptContentLen: cl))
            Task { for await t in stream { box.v.append(t) }; sem.signal() }
            sem.wait()
            return box.v
        }

        backend.prefixCacheForced = false
        let ref = reqs.map { gen($0.prompt, $0.cl) }
        backend.prefixCacheForced = true
        backend.resetPrefixCache()
        var hitsAfter: [Int] = []
        let got = reqs.map { r in let g = gen(r.prompt, r.cl); hitsAfter.append(backend.prefixRAMHits); return g }

        var lines = ["[prefix-ram-e2e] A cold / B cold / A-extend (RAM restore) / A-extend (in-path), maxTok=\(maxTok), tier=\(ec > 0 ? "streaming C=\(ec)" : "resident")"]
        var pass = true
        let labels = ["R1 A cold    ", "R2 B switch  ", "R3 A ram-hit ", "R4 A in-path "]
        let wantHits = [0, 0, 1, 1]
        for i in 0..<reqs.count {
            let ok = ref[i] == got[i] && hitsAfter[i] == wantHits[i]
            pass = pass && ok
            lines.append("  \(labels[i])  ref=\(ref[i].count)tok  cached=\(got[i].count)tok  ramHits=\(hitsAfter[i]) (want \(wantHits[i]))  \(ok ? "IDENTICAL ✅" : "DIVERGE ❌")")
            if ref[i] != got[i] { lines.append("    ref   : \(ref[i].prefix(12))"); lines.append("    cached: \(got[i].prefix(12))") }
        }
        lines.append("PREFIXRAME2E \(pass ? "PASS" : "FAIL")")
        return lines.joined(separator: "\n")
    }

    // #76 gate (bolt side): bolt-mode decode with the in-RAM prompt-prefix blob
    // (QWISP_BOLT_PREFIX) vs without must be byte-identical. Two INDEPENDENT backends
    // (own BoltServe, own arena) are fed the same request sequence so per-process calib
    // and rolling recalib evolve identically; only the prefix reuse differs. The restored
    // KV/GDN bytes equal a fresh strict prefill's by construction; freeze() re-ensures
    // top-C of the same basis either way — divergence would indicate a slot-state leak.
    public static func prefixBoltE2E(modelDir: String) -> String {
        let c = Tell.envInt("QWISP_PREFIX_E2E_C", 64)
        guard let bRef = try? SeedlessBackend(modelDir: modelDir, tier: .streaming(c: c)),
              let bCached = try? SeedlessBackend(modelDir: modelDir, tier: .streaming(c: c))
        else { return "[prefix-bolt-e2e] load fail\nPREFIXBOLTE2E FAIL" }
        func toks(_ n: Int, _ salt: Int) -> [Int] { (0..<n).map { (($0 &* 7 &+ salt) % 5000) + 100 } }
        let A = toks(300, 13), genSuffix = toks(8, 777)
        // 3-request extend chain (agentic shape): each request extends the previous content.
        let reqs = [A + genSuffix,
                    A + toks(40, 2) + genSuffix,
                    A + toks(40, 2) + toks(40, 5) + genSuffix]
        final class Box: @unchecked Sendable { var v: [Int] = [] }
        func gen(_ b: SeedlessBackend, _ p: [Int], prefix: Bool) -> [Int] {
            setenv("QWISP_BOLT_PREFIX", prefix ? "1" : "0", 1)
            let box = Box(); let sem = DispatchSemaphore(value: 0)
            let s = b.generate(p, options: GenerateOptions(maxTokens: 24, promptContentLen: p.count - 8))
            Task.detached { for await t in s { box.v.append(t) }; sem.signal() }
            sem.wait()
            return box.v
        }
        var lines = ["[prefix-bolt-e2e] streaming C=\(c), bolt default mode, 3-request extend chain"]
        var pass = true
        for (i, p) in reqs.enumerated() {
            let r = gen(bRef, p, prefix: false)
            let g = gen(bCached, p, prefix: true)
            let ok = r == g && !r.isEmpty
            pass = pass && ok
            lines.append("  R\(i + 1) prompt=\(p.count)  ref=\(r.count)tok cached=\(g.count)tok  \(ok ? "IDENTICAL ✅" : "DIVERGE ❌")")
            if !ok { lines.append("    ref:    \(r.prefix(12))"); lines.append("    cached: \(g.prefix(12))") }
        }
        unsetenv("QWISP_BOLT_PREFIX")
        lines.append("PREFIXBOLTE2E \(pass ? "PASS" : "FAIL")")
        return lines.joined(separator: "\n")
    }

    // Speed probe for the multi-slot cache: TTFT (time-to-first-token) for a cold request vs a
    // cross-conversation request that shares a large prefix vs an intra-conversation extend.
    // Shows the multi-slot win = skipping re-prefill of the shared system+tools prefix on a NEW convo.
    // Env: QWISP_PREFIX_SHARED (shared prefix length, default 4096).
    public static func prefixCacheSpeedProbe(modelDir: String) -> String {
        guard let backend = try? SeedlessBackend(modelDir: modelDir) else { return "[prefix-speed] load fail\nPREFIXSPEED done" }
        setenv("QWISP_PREFIX_SNAP_STRIDE", "2048", 1)
        let P = Tell.envInt("QWISP_PREFIX_SHARED", 4096)
        func toks(_ n: Int, _ salt: Int) -> [Int] { (0..<n).map { (($0 &* 7 &+ salt) % 5000) + 100 } }
        let shared = toks(P, 13)
        func req(_ content: [Int]) -> (p: [Int], cl: Int) { (content + toks(8, 777), content.count) }
        let r1 = req(shared + toks(64, 1))                       // cold: cache empty
        let r2 = req(shared + toks(64, 2))                       // cross-conversation: shares `shared`
        let r3 = req(shared + toks(64, 2) + toks(80, 3))         // intra-conversation: extends r2

        final class Box: @unchecked Sendable { var v: [Int] = []; var ttft = 0.0 }
        func ttft(_ p: [Int], _ cl: Int) -> Double {
            let box = Box(); let sem = DispatchSemaphore(value: 0); let t0 = Date()
            let stream = backend.generate(p, options: GenerateOptions(maxTokens: 4, promptContentLen: cl))
            Task { for await t in stream { if box.v.isEmpty { box.ttft = Date().timeIntervalSince(t0) }; box.v.append(t) }; sem.signal() }
            sem.wait(); return box.ttft
        }

        backend.prefixCacheForced = true
        backend.resetPrefixCache()
        let cold  = ttft(r1.p, r1.cl)     // cache empty → full prefill of `shared`+tail
        let cross = ttft(r2.p, r2.cl)     // multi-slot restores a boundary inside `shared`
        let intra = ttft(r3.p, r3.cl)     // extends r2 → restores the top boundary
        func fmt(_ s: Double) -> String { String(format: "%.2fs", s) }
        return ["[prefix-speed] shared prefix=\(P) tok, resident/fused, greedy",
                "  R1 cold (cache empty)      TTFT \(fmt(cold))",
                String(format: "  R2 cross-conversation      TTFT %@   (%.1fx vs cold)", fmt(cross), cold / max(cross, 1e-3)),
                String(format: "  R3 intra-conversation      TTFT %@   (%.1fx vs cold)", fmt(intra), cold / max(intra, 1e-3)),
                "PREFIXSPEED done"].joined(separator: "\n")
    }

    // Prefill stage profile (kernel-speedup recon): ① raw path — per-stage GPU time via
    // forwardRowsProfiled (CB-split, exact math) bucketed into GDN/attn/MoE, swept over chunk size.
    // ② MLX path — same prompt chunk-prefilled through QwispModel (MLX gather_qmm uses matrix-unit
    // kernels at M>=14), as free evidence of what a GEMM-structured MoE kernel buys at prefill M.
    // Env: QWISP_PREFILL_LEN (default 2048).
    public static func prefillStageProfile(modelDir: String) -> String {
        guard let store = try? WeightStore(modelDir: modelDir) else { return "[stage-prof] load fail\nSTAGEPROF done" }
        store.residentAll()
        let engine = SeedlessEngine.build(store: store)
        let promptLen = Tell.envInt("QWISP_PREFILL_LEN", 2048)
        let promptI32 = (0..<promptLen).map { Int32((($0 &* 7 &+ 13) % 5000) + 100) }
        var lines = ["[stage-prof] promptLen=\(promptLen) resident"]

        // ① raw per-stage GPU time, chunk sweep
        lines.append("  ── raw path: per-stage GPU ms (CB-split, exact) ──")
        for chunk in [64, 256, 1024] {
            guard let (fwd, _) = engine.makeFused(maxM: chunk + 8, maxSeqLen: promptLen + 128) else {
                lines.append("  chunk=\(chunk): makeFused nil"); continue
            }
            var g = 0.0, a = 0.0, m = 0.0
            let t0 = Date(); var pos = 0
            while pos < promptLen {
                let end = Swift.min(pos + chunk, promptLen)
                let x = engine.embed(tokens: Array(promptI32[pos ..< end]))
                guard let t = fwd.forwardRowsProfiled(x, M: end - pos) else { break }
                g += t.gdn; a += t.attn; m += t.moe
                pos = end
            }
            let wall = Date().timeIntervalSince(t0)
            let tot = g + a + m
            lines.append(String(format: "  chunk=%4d  GDN %6.2fs (%4.1f%%)  attn %6.2fs (%4.1f%%)  MoE %6.2fs (%4.1f%%)  | stageSum %5.1fs  wall %5.1fs  %.0f tok/s",
                                chunk, g/1000, 100*g/tot, a/1000, 100*a/tot, m/1000, 100*m/tot,
                                tot/1000, wall, Double(promptLen)/wall))
        }

        // ② MLX path (matrix-unit gather_qmm at M>=14): same prompt, chunked cached prefill
        lines.append("  ── MLX path (QwispModel, gather_qmm matrix-unit) ──")
        let model = QwispModel(store: store)
        let ids = promptI32.map { Int($0) }
        for chunk in [64, 256] {
            let caches = model.makeCaches()
            let t0 = Date(); var pos = 0
            while pos < promptLen {
                let end = Swift.min(pos + chunk, promptLen)
                let (hidden, _) = model.forwardHidden(MLXArray(ids[pos ..< end].map { Int32($0) }).reshaped([1, end - pos]),
                                                      caches: caches)
                MLX.eval(hidden)
                pos = end
            }
            let wall = Date().timeIntervalSince(t0)
            lines.append(String(format: "  chunk=%4d  wall %5.1fs  %.0f tok/s", chunk, wall, Double(promptLen)/wall))
        }
        // MLX GDN stage attribution (GatedDeltaNetLayer prof counters; eval barriers inflate wall,
        // stage sums still attribute where MLX spends its GDN time vs our sequential T-loop kernel).
        StreamingMoEBlock.profileLayers = true
        StreamingMoEBlock.tGdnInproj = 0; StreamingMoEBlock.tGdnConv = 0
        StreamingMoEBlock.tGdnKernel = 0; StreamingMoEBlock.tGdnOut = 0
        do {
            let caches = model.makeCaches()
            let t0 = Date(); var pos = 0
            while pos < promptLen {
                let end = Swift.min(pos + 64, promptLen)
                let (hidden, _) = model.forwardHidden(MLXArray(ids[pos ..< end].map { Int32($0) }).reshaped([1, end - pos]),
                                                      caches: caches)
                MLX.eval(hidden)
                pos = end
            }
            let wall = Date().timeIntervalSince(t0)
            func s(_ ns: UInt64) -> Double { Double(ns) / 1e9 }
            lines.append(String(format: "  MLX GDN stages (c64, barriered wall %.1fs): inproj %.2fs  conv %.2fs  recur-kernel %.2fs  out %.2fs  | GDN total %.2fs",
                                wall, s(StreamingMoEBlock.tGdnInproj), s(StreamingMoEBlock.tGdnConv),
                                s(StreamingMoEBlock.tGdnKernel), s(StreamingMoEBlock.tGdnOut),
                                s(StreamingMoEBlock.tGdnInproj + StreamingMoEBlock.tGdnConv + StreamingMoEBlock.tGdnKernel + StreamingMoEBlock.tGdnOut)))
        }
        StreamingMoEBlock.profileLayers = false
        lines.append("STAGEPROF done")
        return lines.joined(separator: "\n")
    }

    // Verification ① for the MLX-dense-prefill route: per-element M-invariance of MLX
    // quantizedMatmul. Row r's output must be bit-identical no matter how many other rows share
    // the call (and under zero-padding), or split-point-invariant prefill via MLX dense is dead.
    // MLX dispatches qmv (M<14, scalar) vs steel qmm (M>=14, matrix units) — the padding variant
    // tests whether pinning ALL prefill matmuls to the steel class (pad M up to >=14) is coherent.
    public static func mlxQmmInvariance(modelDir: String) -> String {
        var lines = ["[mlx-qmm-minv] MLX quantizedMatmul per-element M-invariance (value-equality per row)"]
        for (label, K, N) in [("2048→512", 2048, 512), ("2048→12352", 2048, 12352)] {
            let wf = MLXRandom.normal([N, K]).asType(.float16)
            let (wq, sc, biOpt) = MLX.quantized(wf, groupSize: 64, bits: 4, mode: .affine)
            guard let bi = biOpt else { return "[mlx-qmm-minv] biases nil" }
            let xAll = MLXRandom.normal([1024, K]).asType(.float16)
            MLX.eval([wq, sc, bi, xAll])
            func qmm(_ x: MLXArray) -> MLXArray {
                let y = MLX.quantizedMatmul(x, wq, scales: sc, biases: bi, transpose: true, groupSize: 64, bits: 4)
                y.eval(); return y
            }
            let full = qmm(xAll)                                   // M=1024 reference
            func rowsEqual(_ a: MLXArray, _ b: MLXArray) -> Bool {
                let d = MLX.all(a .== b)
                d.eval(); return d.item(Bool.self)
            }
            lines.append("  ── \(label) ──")
            for M in [1, 8, 13, 14, 16, 64, 256] {
                let out = qmm(xAll[0..<M])
                let ok = rowsEqual(out, full[0..<M])
                lines.append("    M=\(M)\tvs M=1024 rows[0..<\(M)]: \(ok ? "IDENTICAL ✅" : "DIFFER ❌")")
            }
            // zero-pad variant: 3 real rows + 13 zero rows = M=16 call; real rows vs full
            let pad = MLX.concatenated([xAll[0..<3], MLXArray.zeros([13, K]).asType(.float16)], axis: 0)
            let outPad = qmm(pad)
            let okPad = rowsEqual(outPad[0..<3], full[0..<3])
            lines.append("    M=3+13pad(=16)\tvs M=1024 rows[0..<3]: \(okPad ? "IDENTICAL ✅" : "DIFFER ❌")")
        }
        lines.append("MLXQMMMINV done")
        return lines.joined(separator: "\n")
    }

    // Hybrid-prefill estimate: measures the steel-replaceable sub-stage costs (GDN matmuls, GDN
    // recurrence, MoE routed gather, MoE shared) by warm-buffer skip differentials (KV/GDN state
    // reset via fullRestore between runs; buffers stay numerically sane after a warm pass, avoiding
    // the cold-garbage artifact that invalidated v1), then folds in the MEASURED steel ratios
    // (steel-route-bench @1a41c38) + sync/copy overheads to print the projected hybrid speedup.
    // Additivity of the differentials is reported as a sanity check.
    public static func hybridEstimate(modelDir: String) -> String {
        guard let store = try? WeightStore(modelDir: modelDir) else { return "[hybrid-est] load fail\nHYBRIDEST done" }
        store.residentAll()
        let engine = SeedlessEngine.build(store: store)
        let promptLen = Tell.envInt("QWISP_PREFILL_LEN", 2048)
        let prompt = (0..<promptLen).map { Int32((($0 &* 7 &+ 13) % 5000) + 100) }
        func allOff() {
            SeedlessMetalForward.profSkipGDNMatmul = false
            SeedlessMetalForward.profSkipGDNRecur = false
            SeedlessMetalForward.profSkipMoERouted = false
            SeedlessMetalForward.profSkipMoEShared = false
        }
        var out = ["[hybrid-est] promptLen=\(promptLen) resident — sub-stage differentials (warm) + steel fold-in"]
        // measured steel speedups (steel-route-bench): [chunk: (gdnMatmul, moeRouted, moeShared)]
        // gdnMatmul = blend of inproj (3.8x@256, 4.3x@1024) and outproj (~1.4x@256, ~3.3x@1024) → ~3.0/4.0
        let steel: [Int: (gdnMat: Double, moeR: Double, moeS: Double)] = [
            256:  (3.0, 1.40, 1.0),
            1024: (4.0, 2.24, 2.3)]
        for chunk in [256, 1024] {
            guard let b = Tell.fusedBackend(engine: engine, maxM: chunk + 8, maxSeqLen: promptLen + 128) else { continue }
            guard let empty = b.fullSnapshot?() else { continue }
            func prefill() -> Double {
                b.fullRestore?(empty)
                let t0 = Date(); var pos = 0
                while pos < promptLen {
                    let e = Swift.min(pos + chunk, promptLen)
                    _ = b.forward(Array(prompt[pos ..< e])); pos = e
                }
                return Date().timeIntervalSince(t0)
            }
            allOff(); _ = prefill()                                  // warm: sane buffers + pipelines
            func timed(_ set: () -> Void) -> Double {
                allOff(); set()
                let a = prefill(), c = prefill()
                allOff(); return Swift.min(a, c)
            }
            let full  = timed {}
            let dGMat = full - timed { SeedlessMetalForward.profSkipGDNMatmul = true }
            let dGRec = full - timed { SeedlessMetalForward.profSkipGDNRecur = true }
            let dMoER = full - timed { SeedlessMetalForward.profSkipMoERouted = true }
            let dMoES = full - timed { SeedlessMetalForward.profSkipMoEShared = true }
            let parts = dGMat + dGRec + dMoER + dMoES
            let r = steel[chunk]!
            let boundaries = Double(promptLen / chunk) * (30.0 * 2 + 40.0 * 2)   // per chunk: GDN 2/layer + MoE 2/layer
            let sync = boundaries * 0.263e-3
            let copies = Double(promptLen / chunk) * 0.015 * Double(chunk) / 1024.0   // ~15ms/chunk@1024 measured-class memcpy
            let hybrid = full - dGMat * (1 - 1 / r.gdnMat) - dMoER * (1 - 1 / r.moeR) - dMoES * (1 - 1 / r.moeS) + sync + copies
            out.append(String(format: "  chunk=%4d  full %5.2fs (%.0f tok/s)", chunk, full, Double(promptLen) / full))
            out.append(String(format: "    sub-stage: GDNmat %5.2fs  GDNrec %5.2fs  MoErouted %5.2fs  MoEshared %5.2fs  (sum %.2fs = %.0f%% of full — additivity check)",
                              dGMat, dGRec, dMoER, dMoES, parts, 100 * parts / full))
            out.append(String(format: "    HYBRID est: %5.2fs (%.0f tok/s)  → %.2fx   [steel %0.1f/%0.2f/%0.1fx, sync %.2fs, copies %.2fs]",
                              hybrid, Double(promptLen) / hybrid, full / hybrid, r.gdnMat, r.moeR, r.moeS, sync, copies))
        }
        out.append("HYBRIDEST done")
        return out.joined(separator: "\n")
    }

    // Stage-1 gate for the steel-prefill hybrid: (a) model-math sanity — hybrid final normed must be
    // CLOSE to raw (different rounding, tiny rel err; catches mis-wired buffers/slices), (b)
    // determinism — two hybrid runs byte-identical, (c) split-invariance — chunk 512 vs 1024 byte-
    // identical (steel padding pins every M into one kernel class), (d) speed — hybrid vs raw prefill.
    public static func hybridPrefillBench(modelDir: String) -> String {
        guard let store = try? WeightStore(modelDir: modelDir) else { return "[hybrid-bench] load fail\nHYBRIDBENCH FAIL" }
        store.residentAll()
        let engine = SeedlessEngine.build(store: store)
        let promptLen = Tell.envInt("QWISP_PREFILL_LEN", 2048)
        let prompt = (0..<promptLen).map { Int32((($0 &* 7 &+ 13) % 5000) + 100) }

        func mkBackend(_ hybrid: Bool) -> Tell.SpecBackend? {
            setenv("QWISP_HYBRID_PREFILL", hybrid ? "1" : "0", 1)
            return Tell.fusedBackend(engine: engine, maxM: 1032, maxSeqLen: promptLen + 128)
        }
        // prefill via Tell.prefill (uses hybrid closure + chunk automatically); returns final normed row.
        func run(_ b: Tell.SpecBackend, chunk: Int? = nil) -> [Float16]? {
            var bb = b
            if let c = chunk { bb.hybridChunk = c }
            guard let n = Tell.prefill(promptIds: prompt, backend: bb) else { return nil }
            n.eval()
            return n.reshaped([-1]).asArray(Float16.self)
        }
        var out = ["[hybrid-bench] promptLen=\(promptLen) resident, maxM=1032"]

        guard let rawB = mkBackend(false) else { return "[hybrid-bench] raw backend nil\nHYBRIDBENCH FAIL" }
        let tR0 = Date(); guard let rawOut = run(rawB) else { return "[hybrid-bench] raw run nil\nHYBRIDBENCH FAIL" }
        let tRaw = Date().timeIntervalSince(tR0)

        guard let hyB = mkBackend(true), hyB.forwardHybrid != nil else { return "[hybrid-bench] hybrid backend nil\nHYBRIDBENCH FAIL" }
        let tH0 = Date(); guard let hy1 = run(hyB) else { return "[hybrid-bench] hybrid run nil\nHYBRIDBENCH FAIL" }
        let tHy = Date().timeIntervalSince(tH0)

        // (a) model-math sanity: rel err of final normed row
        var num = 0.0, den = 0.0
        for i in 0..<rawOut.count { num += abs(Double(hy1[i]) - Double(rawOut[i])); den += abs(Double(rawOut[i])) }
        let rel = num / Swift.max(den, 1e-9)
        // (b) determinism: fresh hybrid backend, same run
        guard let hyB2 = mkBackend(true), let hy2 = run(hyB2) else { return "[hybrid-bench] hy2 nil\nHYBRIDBENCH FAIL" }
        let det = hy1 == hy2
        // (c) split-invariance: chunk 512 on a fresh hybrid backend
        guard let hyB3 = mkBackend(true), let hy3 = run(hyB3, chunk: 512) else { return "[hybrid-bench] hy3 nil\nHYBRIDBENCH FAIL" }
        let splitInv = hy1 == hy3

        out.append(String(format: "  raw    prefill %5.2fs (%.0f tok/s)  [chunk 64]", tRaw, Double(promptLen) / tRaw))
        out.append(String(format: "  hybrid prefill %5.2fs (%.0f tok/s)  [chunk 1024]  → %.2fx", tHy, Double(promptLen) / tHy, tRaw / tHy))
        // (a2) single-op wiring sanity: steel qkv on layer-0 weights vs f32-x reference (~1e-3 = correct)
        var relOp = -1.0
        if let g0 = engine.layers.first(where: { $0.gdn != nil })?.gdn {
            let xt = MLXRandom.normal([64, 2048]).asType(.float16)
            let ys = MLX.quantizedMatmul(xt, g0.qkvWq, scales: g0.qkvSc, biases: g0.qkvBi, transpose: true, groupSize: 64, bits: 4)
            let yr = MLX.quantizedMatmul(xt.asType(.float32), g0.qkvWq, scales: g0.qkvSc.asType(.float32), biases: g0.qkvBi.asType(.float32), transpose: true, groupSize: 64, bits: 4)
            MLX.eval([ys, yr])
            let a = ys.asType(.float32).reshaped([-1]).asArray(Float.self)
            let b = yr.reshaped([-1]).asArray(Float.self)
            var n2 = 0.0, d2 = 0.0
            for i in 0..<a.count { n2 += Double(abs(a[i] - b[i])); d2 += Double(abs(b[i])) }
            relOp = n2 / Swift.max(d2, 1e-9)
        }
        // (a3) fidelity-family baseline: full-MLX (QwispModel) prefill final hidden vs raw
        let model = QwispModel(store: store)
        let caches = model.makeCaches()
        var mlxLast: [Float16] = []
        var pos = 0
        while pos < promptLen {
            let end = Swift.min(pos + 64, promptLen)
            let ids = prompt[pos ..< end].map { Int32($0) }
            let (hidden, _) = model.forwardHidden(MLXArray(ids).reshaped([1, end - pos]), caches: caches)
            hidden.eval()
            if end == promptLen { mlxLast = hidden[0, end - pos - 1].asType(.float16).reshaped([-1]).asArray(Float16.self) }
            pos = end
        }
        var nM = 0.0, dM = 0.0
        for i in 0..<Swift.min(mlxLast.count, rawOut.count) { nM += abs(Double(mlxLast[i]) - Double(rawOut[i])); dM += abs(Double(rawOut[i])) }
        let relMLX = nM / Swift.max(dM, 1e-9)
        out.append(String(format: "  (a) rel err vs raw      : %.2e   [single-op steel vs f32-ref: %.2e; full-MLX vs raw baseline: %.2e]", rel, relOp, relMLX))
        let sane = relOp < 5e-3 && rel < Swift.max(2 * relMLX, 1e-2)
        out.append("      wiring \(relOp < 5e-3 ? "OK ✅" : "MISWIRED ❌"), fidelity \(rel < Swift.max(2 * relMLX, 1e-2) ? "IN-FAMILY ✅ (≤2x full-MLX baseline)" : "OUT ❌")")
        out.append("  (b) determinism         : \(det ? "IDENTICAL ✅" : "DIVERGE ❌")")
        out.append("  (c) split-inv (512 vs 1024): \(splitInv ? "IDENTICAL ✅" : "DIVERGE ❌")")
        let pass = sane && det && splitInv
        out.append("HYBRIDBENCH \(pass ? "PASS" : "FAIL")")
        setenv("QWISP_HYBRID_PREFILL", "0", 1)
        return out.joined(separator: "\n")
    }

    public static func prefixCachePoC(modelDir: String) -> String {
        guard let store = try? WeightStore(modelDir: modelDir) else { return "[prefix-poc] load fail\nPREFIXPOC FAIL" }
        store.residentAll()
        let engine = SeedlessEngine.build(store: store)

        let aLen = 1024, tailLen = 128, bLen = 256, decodeN = 32
        func toks(_ n: Int, _ salt: Int) -> [Int32] { (0..<n).map { Int32((($0 &* 7 &+ salt) % 5000) + 100) } }
        let A = toks(aLen, 13), tail = toks(tailLen, 999), B = toks(bLen, 5)
        let full = A + B
        let maxSeqLen = A.count + tailLen + B.count + decodeN + 128

        func greedy(_ backend: Tell.SpecBackend, firstNormed: MLXArray) -> [Int] {
            guard let l0 = engine.logits(firstNormed, M: 1) else { return [] }
            MLX.eval([l0])
            var tok = MLX.argMax(l0[0], axis: -1).item(Int.self)
            var out = [tok]
            for _ in 1..<decodeN {
                guard let nx = backend.stepArgmax([Int32(tok)])?.first else { break }
                out.append(nx); tok = nx
            }
            return out
        }

        // Run the reuse protocol against a backend and compare to its own baseline. `full` builds a
        // fresh backend each call, so the two paths share no mutable state.
        func trial(_ label: String, _ mk: () -> Tell.SpecBackend?, timed: Bool) -> String {
            guard let bBase = mk(), let bReuse = mk() else { return "  \(label): backend nil" }
            let t0 = Date()
            guard let nBase = Tell.prefill(promptIds: full, backend: bBase) else { return "  \(label): prefill base nil" }
            let tFull = Date().timeIntervalSince(t0)
            let seqBase = greedy(bBase, firstNormed: nBase)

            _ = Tell.prefill(promptIds: A, backend: bReuse)   // cache the stable content prefix
            // Full-state snapshot for arbitrary rewind; falls back to snapshot (composed's is already full).
            let snap = (bReuse.fullSnapshot ?? bReuse.snapshot)()          // content-boundary snapshot
            _ = Tell.prefill(promptIds: tail, backend: bReuse) // gen prompt + generated (request-specific)
            (bReuse.fullRestore ?? bReuse.rollback)(snap)      // next request rewinds past them
            let t1 = Date()
            guard let nReuse = Tell.prefill(promptIds: B, backend: bReuse) else { return "  \(label): prefill reuse nil" }
            let tSuffix = Date().timeIntervalSince(t1)
            let seqReuse = greedy(bReuse, firstNormed: nReuse)

            let ok = seqBase == seqReuse
            let sp = tSuffix > 0 ? String(format: " full=%.2fs suffix=%.2fs speedup=%.1fx", tFull, tSuffix, tFull / tSuffix) : ""
            let detail = ok ? "" : "  base=\(Array(seqBase.prefix(6))) reuse=\(Array(seqReuse.prefix(6)))"
            return "  \(label): byte-identical=\(ok ? "YES" : "NO")\(timed ? sp : "")\(detail)"
        }

        let composed = trial("composed(full copyState)", { Tell.composedBackend(engine: engine) }, timed: false)
        let fused = trial("fused(full snapshot)", { Tell.fusedBackend(engine: engine, maxM: 96, maxSeqLen: maxSeqLen) }, timed: true)
        let pass = composed.contains("byte-identical=YES") && fused.contains("byte-identical=YES")
        return """
        [prefix-poc] A=\(aLen) tail=\(tailLen) B=\(bLen) decodeN=\(decodeN)
        \(composed)
        \(fused)
        PREFIXPOC \(pass ? "PASS" : "FAIL")   (both paths must be byte-identical to full prefill)
        """
    }
}
