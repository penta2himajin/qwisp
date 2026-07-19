import Foundation
import MLX
import Metal

// #98 DFlash phase A1 — raw-Metal drafter forward (spec: scratchpad/dflash-phaseA1-spec.md).
//
// ONE command buffer per block, following the RawMTPHead idiom (SeedlessFusedVerify.swift
// ~5079 SeedlessMTPHead): persistent f16 MTLBuffer weights (noCopy — the cast MLXArrays are
// retained for the drafter's lifetime), persistent per-layer KV MTLBuffers, everything else
// reused from the frozen SeedlessFusedVerify/SeedlessMetalForward statics (fmm_rows,
// rmsNormRows, ropeRows, embed_rows_q4, qmm4/qmmRows, argmax_rows, swiglu, resid_add,
// write_kv_rows). The ONLY new Metal source is the drafter's own SDPA kernel (below) — the
// drafter has no gate path and no partial-rotary rope, so the existing gated-attention prep
// kernels (attn_q_prep_rows/attn_k_prep_rows) don't fit and the existing D=256-hardcoded
// sdpa_rows kernel doesn't fit a runtime headDim — a small headDim-parametric kernel is
// compiled here with its own makeLibrary (diag-kernel precedent).
//
// v1 scope (pinned): NO sliding eviction. ctxCap = slidingWindow - 1 for ALL layers; forward
// returns nil once ctxLen+ctxCount would exceed it (caller falls back to the MLX drafter).

public final class DFlashRawDrafter {
    public let config: DFlashDrafterConfig

    // ── Geometry (cached from config for terse call sites) ─────────────────
    private let H: Int
    private let numHeads: Int, numKV: Int, headDim: Int
    private let mlpDim: Int
    private let ctxFeatureDim: Int
    private let ropeTheta: Float, eps: Float
    private let blockSize: Int
    private let ctxCap: Int                 // v1 cap: slidingWindow - 1, ALL layers

    private let device: MTLDevice
    private let queue: MTLCommandQueue

    // ── Per-layer persistent weights + KV cache ─────────────────────────────
    private struct Layer {
        let qW: MTLBuffer, kW: MTLBuffer, vW: MTLBuffer, oW: MTLBuffer
        let qNorm: MTLBuffer, kNorm: MTLBuffer
        let inputLN: MTLBuffer, postLN: MTLBuffer
        let gateW: MTLBuffer, upW: MTLBuffer, downW: MTLBuffer
        let kCache: MTLBuffer, vCache: MTLBuffer   // [numKV, ctxCap, headDim] f16, persistent
        let isSliding: Bool
    }
    private let layers: [Layer]

    // ── Global drafter weights ──────────────────────────────────────────────
    private let fcW: MTLBuffer            // [H, ctxFeatureDim]
    private let hiddenNorm: MTLBuffer     // [H]
    private let finalNorm: MTLBuffer      // [H]  (drafter's OWN final norm — not the target's)

    // ── Target head (attached post-init; embed for noise, lm_head for logits) ──
    private var embedWBuf: MTLBuffer?, embedSBuf: MTLBuffer?, embedBBuf: MTLBuffer?
    private var lmWBuf: MTLBuffer?, lmSBuf: MTLBuffer?, lmBBuf: MTLBuffer?
    private var vocab: Int = 0
    private var logitsBuf: MTLBuffer?

    // noCopy 寿命規約 (notes/03): mtlBuf(noCopy) の backing MLXArray を drafter と同寿命で保持。
    private var retained: [MLXArray] = []

    // ── Shared committed context length (v1: every layer advances together) ──
    private var ctxLen: Int = 0
    private var hasRun = false

    // ── Scratch buffers, reused every forward() call ────────────────────────
    private let tokBuf: MTLBuffer                  // [blockSize] int32
    private let hBuf: MTLBuffer                    // [blockSize, H] evolving residual stream
    private let anLNBuf: MTLBuffer                 // [blockSize, H] rmsNorm(inputLN) scratch
    private let qRawBuf, qNormedBuf, qRotBuf: MTLBuffer          // [blockSize*numHeads, headDim]
    private let pkRawBuf, pkNormedBuf, pkRotBuf: MTLBuffer       // [blockSize*numKV, headDim]
    private let pvBuf: MTLBuffer                                // [blockSize*numKV, headDim]
    private let attnOutBuf: MTLBuffer              // [blockSize*numHeads, headDim]
    private let attnResBuf: MTLBuffer              // [blockSize, H]
    private let mnBuf: MTLBuffer                   // [blockSize, H] post-attn norm
    private let gateBuf, upBuf, actBuf: MTLBuffer   // [blockSize, mlpDim]
    private let downBuf: MTLBuffer                 // [blockSize, H]
    private let normedBuf: MTLBuffer               // [blockSize, H] final hidden (lastFinalHidden)
    private let tokenOutBuf: MTLBuffer             // [blockSize] int32 argmax

    private let ctxInputBuf: MTLBuffer             // [ctxCap, ctxFeatureDim]
    private let fcOutBuf: MTLBuffer                // [ctxCap, H]
    private let hCtxBuf: MTLBuffer                 // [ctxCap, H]
    private let ckRawBuf, ckNormedBuf, ckRotBuf: MTLBuffer       // [ctxCap*numKV, headDim]
    private let cvBuf: MTLBuffer                                // [ctxCap*numKV, headDim]

    // ── init? ────────────────────────────────────────────────────────────────
    public init?(config: DFlashDrafterConfig, weights: [String: MLXArray]) {
        guard config.slidingWindow > 1, config.blockSize > 1, config.numHeads > 0,
              config.numKVHeads > 0, config.headDim > 0
        else { return nil }
        guard let (dev, q) = SeedlessMetalForward.ensure() else { return nil }
        guard SeedlessFusedVerify.ensureFmmPipeline(),
              SeedlessFusedVerify.ensureRowsAuxPipelines(),
              SeedlessMetalForward.ensureAuxPipelines(),
              DFlashRawKernels.ensureSdpaPipeline(dev)
        else { return nil }
        SeedlessFusedVerify.ensureQmmPipeline()
        guard SeedlessMetalForward._qmmPipeline != nil else { return nil }
        // qmm4_tiled warm (attachHead idiom — the lm_head encode force-unwraps the pipeline;
        // locked tests build this class standalone, so don't rely on suite ordering).
        if SeedlessMetalForward._qmm4TiledPipeline == nil {
            let x = MLXRandom.normal([1, 512]).asType(.float16)
            let wf = MLXRandom.normal([8, 512]).asType(.float16)
            let (wq, s, b) = MLX.quantized(wf, groupSize: 64, bits: 4, mode: .affine)
            MLX.eval([x, wq, s, b!])
            _ = SeedlessMetalForward.qmmTiled(x, wq, scales: s, biases: b!, M: 1, K: 512, N: 8)
        }
        guard SeedlessMetalForward._qmm4TiledPipeline != nil else { return nil }
        // Force-compile the lazily-built rms(F16/F32) and rope pipelines (encode-only helpers
        // force-unwrap them) — RawMTPHead init idiom, needed on a fresh process too.
        let warmW = MLXArray.ones([8]).asType(.float16)
        guard SeedlessMetalForward.rmsNormRows(warmW, warmW, M: 1, eps: 1e-6, D: 8) != nil,
              SeedlessMetalForward.rmsNormRows(warmW, warmW, M: 1, eps: 1e-6, D: 8, promoteF32: true) != nil,
              SeedlessMetalForward.ropeRows(MLXArray.zeros([1, 8]).asType(.float16),
                                            headDim: 8, ropeDim: 8, base: 10000, startOffset: 0,
                                            M: 1, numHeads: 1) != nil
        else { return nil }

        device = dev; queue = q
        self.config = config
        H = config.hiddenSize
        numHeads = config.numHeads; numKV = config.numKVHeads; headDim = config.headDim
        mlpDim = config.mlpDim; ctxFeatureDim = config.ctxFeatureDim
        ropeTheta = config.ropeTheta; eps = config.rmsEps
        blockSize = config.blockSize
        ctxCap = config.slidingWindow - 1

        func g(_ key: String) -> MLXArray? { weights[key]?.asType(.float16) }
        var localRetained: [MLXArray] = []
        func f16b(_ a: MLXArray) -> MTLBuffer? {
            let c = a.asType(.float16); c.eval(); localRetained.append(c)
            return SeedlessMetalForward.mtlBuf(c, dev)
        }

        guard let fcArr = g("fc.weight"), let hnArr = g("hidden_norm.weight"), let fnArr = g("norm.weight"),
              let fcB = f16b(fcArr), let hnB = f16b(hnArr), let fnB = f16b(fnArr)
        else { return nil }
        fcW = fcB; hiddenNorm = hnB; finalNorm = fnB

        let kvBytes = numKV * ctxCap * headDim * 2
        var builtLayers: [Layer] = []
        builtLayers.reserveCapacity(config.numLayers)
        for i in 0 ..< config.numLayers {
            let p = "layers.\(i)."
            guard let qWArr = g(p + "self_attn.q_proj.weight"),
                  let kWArr = g(p + "self_attn.k_proj.weight"),
                  let vWArr = g(p + "self_attn.v_proj.weight"),
                  let oWArr = g(p + "self_attn.o_proj.weight"),
                  let qNArr = g(p + "self_attn.q_norm.weight"),
                  let kNArr = g(p + "self_attn.k_norm.weight"),
                  let gateArr = g(p + "mlp.gate_proj.weight"),
                  let upArr = g(p + "mlp.up_proj.weight"),
                  let downArr = g(p + "mlp.down_proj.weight"),
                  let inLNArr = g(p + "input_layernorm.weight"),
                  let postLNArr = g(p + "post_attention_layernorm.weight"),
                  let qWB = f16b(qWArr), let kWB = f16b(kWArr), let vWB = f16b(vWArr), let oWB = f16b(oWArr),
                  let qNB = f16b(qNArr), let kNB = f16b(kNArr),
                  let gateB = f16b(gateArr), let upB = f16b(upArr), let downB = f16b(downArr),
                  let inLNB = f16b(inLNArr), let postLNB = f16b(postLNArr),
                  let kCache = dev.makeBuffer(length: kvBytes, options: .storageModeShared),
                  let vCache = dev.makeBuffer(length: kvBytes, options: .storageModeShared)
            else { return nil }
            let isSliding = i < config.layerTypes.count ? config.layerTypes[i] : false
            builtLayers.append(Layer(qW: qWB, kW: kWB, vW: vWB, oW: oWB, qNorm: qNB, kNorm: kNB,
                                     inputLN: inLNB, postLN: postLNB, gateW: gateB, upW: upB, downW: downB,
                                     kCache: kCache, vCache: vCache, isSliding: isSliding))
        }
        layers = builtLayers
        retained = localRetained

        // ── scratch buffers ──
        func hb(_ n: Int) -> MTLBuffer? { dev.makeBuffer(length: n * 2, options: .storageModeShared) }
        func ib(_ n: Int) -> MTLBuffer? { dev.makeBuffer(length: n * 4, options: .storageModeShared) }
        let bs = blockSize, cc = ctxCap
        guard let _tok = ib(bs),
              let _h = hb(bs * H), let _anLN = hb(bs * H),
              let _qRaw = hb(bs * numHeads * headDim), let _qN = hb(bs * numHeads * headDim), let _qR = hb(bs * numHeads * headDim),
              let _pkRaw = hb(bs * numKV * headDim), let _pkN = hb(bs * numKV * headDim), let _pkR = hb(bs * numKV * headDim),
              let _pv = hb(bs * numKV * headDim),
              let _attnOut = hb(bs * numHeads * headDim), let _attnRes = hb(bs * H),
              let _mn = hb(bs * H),
              let _gate = hb(bs * mlpDim), let _up = hb(bs * mlpDim), let _act = hb(bs * mlpDim),
              let _down = hb(bs * H), let _normed = hb(bs * H),
              let _tokOut = ib(bs),
              let _ctxIn = hb(cc * ctxFeatureDim), let _fcOut = hb(cc * H), let _hCtx = hb(cc * H),
              let _ckRaw = hb(cc * numKV * headDim), let _ckN = hb(cc * numKV * headDim), let _ckR = hb(cc * numKV * headDim),
              let _cv = hb(cc * numKV * headDim)
        else { return nil }
        tokBuf = _tok; hBuf = _h; anLNBuf = _anLN
        qRawBuf = _qRaw; qNormedBuf = _qN; qRotBuf = _qR
        pkRawBuf = _pkRaw; pkNormedBuf = _pkN; pkRotBuf = _pkR
        pvBuf = _pv
        attnOutBuf = _attnOut; attnResBuf = _attnRes
        mnBuf = _mn
        gateBuf = _gate; upBuf = _up; actBuf = _act
        downBuf = _down; normedBuf = _normed
        tokenOutBuf = _tokOut
        ctxInputBuf = _ctxIn; fcOutBuf = _fcOut; hCtxBuf = _hCtx
        ckRawBuf = _ckRaw; ckNormedBuf = _ckN; ckRotBuf = _ckR
        cvBuf = _cv
    }

    // ── Target head attachment ──────────────────────────────────────────────
    public func attachTargetHead(embedW: MLXArray, embedS: MLXArray, embedB: MLXArray,
                                 lmW: MLXArray, lmS: MLXArray, lmB: MLXArray, vocab: Int) -> Bool {
        func rawb(_ a: MLXArray) -> MTLBuffer? { a.eval(); retained.append(a); return SeedlessMetalForward.mtlBuf(a, device) }
        func f16b(_ a: MLXArray) -> MTLBuffer? {
            let c = a.asType(.float16); c.eval(); retained.append(c)
            return SeedlessMetalForward.mtlBuf(c, device)
        }
        guard let ew = rawb(embedW), let es = f16b(embedS), let eb = f16b(embedB),
              let lw = rawb(lmW), let ls = f16b(lmS), let lb = f16b(lmB),
              let logits = device.makeBuffer(length: blockSize * vocab * 2, options: .storageModeShared)
        else { return false }
        embedWBuf = ew; embedSBuf = es; embedBBuf = eb
        lmWBuf = lw; lmSBuf = ls; lmBBuf = lb
        self.vocab = vocab
        logitsBuf = logits
        return true
    }

    // ── Private encode helpers for the two kernels not covered by an existing
    //    "encode-only" static (embed noise into hBuf directly; final argmax). ──

    private func encodeEmbedNoise(_ enc: MTLComputeCommandEncoder,
                                  ew: MTLBuffer, es: MTLBuffer, eb: MTLBuffer) {
        let p = SeedlessFusedVerify._embedRowsPipeline!
        enc.setComputePipelineState(p)
        enc.setBuffer(ew, offset: 0, index: 0); enc.setBuffer(es, offset: 0, index: 1)
        enc.setBuffer(eb, offset: 0, index: 2); enc.setBuffer(tokBuf, offset: 0, index: 3)
        enc.setBuffer(hBuf, offset: 0, index: 4)
        var hh = UInt32(H), tt = UInt32(blockSize * H)
        enc.setBytes(&hh, length: 4, index: 5); enc.setBytes(&tt, length: 4, index: 6)
        enc.dispatchThreads(MTLSize(width: blockSize * H, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: min(p.maxTotalThreadsPerThreadgroup, 256), height: 1, depth: 1))
    }

    private func encodeArgmax(_ enc: MTLComputeCommandEncoder, logits: MTLBuffer, V: Int) {
        let p = SeedlessFusedVerify._argmaxRowsPipeline!
        enc.setComputePipelineState(p)
        enc.setBuffer(logits, offset: 0, index: 0); enc.setBuffer(tokenOutBuf, offset: 0, index: 1)
        var vv = UInt32(V); enc.setBytes(&vv, length: 4, index: 2)
        enc.dispatchThreadgroups(MTLSize(width: blockSize, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    // ── forward ──────────────────────────────────────────────────────────────
    public func forward(u: Int32, ctxRows: [Float16], ctxCount: Int) -> [Int]? {
        guard let ew = embedWBuf, let es = embedSBuf, let eb = embedBBuf,
              let lw = lmWBuf, let ls = lmSBuf, let lb = lmBBuf, let logits = logitsBuf
        else { return nil }   // attachTargetHead not called
        guard ctxLen + ctxCount <= ctxCap else { return nil }   // v1 cap contract

        // 1. noise tokens: [u] ++ [maskId × (blockSize-1)]
        var toks = [Int32](repeating: Int32(config.maskTokenId), count: blockSize)
        toks[0] = u
        let tokPtr = tokBuf.contents().bindMemory(to: Int32.self, capacity: blockSize)
        for i in 0 ..< blockSize { tokPtr[i] = toks[i] }

        // 2. ctx rows upload (persistent scratch buffer, refreshed each call)
        if ctxCount > 0 {
            ctxRows.withUnsafeBytes { raw in
                memcpy(ctxInputBuf.contents(), raw.baseAddress!, ctxCount * ctxFeatureDim * 2)
            }
        }

        let cachedLen = ctxLen + ctxCount   // KV length valid after this call's ctx append
        let scale = Float(pow(Double(headDim), -0.5))

        let trace = Tell.envFlag("QWISP_DFLASH_TRACE")
        var cb = queue.makeCommandBuffer()!
        var enc = cb.makeComputeCommandEncoder()!
        var stamps: [(String, Double)] = []
        // trace-only stage split (flag off = single CB, identical to production).
        func split(_ label: String) {
            guard trace else { return }
            enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
            stamps.append((label, (cb.gpuEndTime - cb.gpuStartTime) * 1000.0))
            cb = queue.makeCommandBuffer()!
            enc = cb.makeComputeCommandEncoder()!
        }

        // noise embed → hBuf (residual stream init)
        encodeEmbedNoise(enc, ew: ew, es: es, eb: eb)

        // ctx feature path (computed ONCE, shared by every layer's k/v proj)
        if ctxCount > 0 {
            DFlashRawKernels.encodeFmmTiled(enc, w: fcW, x: ctxInputBuf, out: fcOutBuf,
                                              M: ctxCount, K: ctxFeatureDim, N: H)
            SeedlessFusedVerify.encodeRmsNormRows(enc, x: fcOutBuf, w: hiddenNorm, out: hCtxBuf,
                                                  rows: ctxCount, D: H, eps: eps)
        }

        split("embed+ctx")

        for layer in layers {
            // attn input norm (block rows only — ctx k/v project straight from hCtx, no inputLN)
            SeedlessFusedVerify.encodeRmsNormRows(enc, x: hBuf, w: layer.inputLN, out: anLNBuf,
                                                  rows: blockSize, D: H, eps: eps)
            // q / propK / propV projections
            DFlashRawKernels.encodeFmmTiled(enc, w: layer.qW, x: anLNBuf, out: qRawBuf,
                                              M: blockSize, K: H, N: numHeads * headDim)
            DFlashRawKernels.encodeFmmTiled(enc, w: layer.kW, x: anLNBuf, out: pkRawBuf,
                                              M: blockSize, K: H, N: numKV * headDim)
            DFlashRawKernels.encodeFmmTiled(enc, w: layer.vW, x: anLNBuf, out: pvBuf,
                                              M: blockSize, K: H, N: numKV * headDim)
            // q/k rmsNorm (last-dim headDim, treating each (row,head) as an independent chunk)
            SeedlessFusedVerify.encodeRmsNormRows(enc, x: qRawBuf, w: layer.qNorm, out: qNormedBuf,
                                                  rows: blockSize * numHeads, D: headDim, eps: eps)
            SeedlessFusedVerify.encodeRmsNormRows(enc, x: pkRawBuf, w: layer.kNorm, out: pkNormedBuf,
                                                  rows: blockSize * numKV, D: headDim, eps: eps)
            // FULL-headDim RoPE: q and propK at position cachedLen + row
            SeedlessFusedVerify.encodeRopeRows(enc, x: qNormedBuf, out: qRotBuf,
                                              headDim: headDim, ropeDim: headDim, base: ropeTheta,
                                              startOffset: cachedLen, M: blockSize, numHeads: numHeads)
            SeedlessFusedVerify.encodeRopeRows(enc, x: pkNormedBuf, out: pkRotBuf,
                                              headDim: headDim, ropeDim: headDim, base: ropeTheta,
                                              startOffset: cachedLen, M: blockSize, numHeads: numKV)

            if ctxCount > 0 {
                // ctx k/v from hCtx (this layer's own k/v proj — NOT re-normed by inputLN)
                DFlashRawKernels.encodeFmmTiled(enc, w: layer.kW, x: hCtxBuf, out: ckRawBuf,
                                                  M: ctxCount, K: H, N: numKV * headDim)
                DFlashRawKernels.encodeFmmTiled(enc, w: layer.vW, x: hCtxBuf, out: cvBuf,
                                                  M: ctxCount, K: H, N: numKV * headDim)
                SeedlessFusedVerify.encodeRmsNormRows(enc, x: ckRawBuf, w: layer.kNorm, out: ckNormedBuf,
                                                      rows: ctxCount * numKV, D: headDim, eps: eps)
                // ctxK at position ctxLen(before this call) + row
                SeedlessFusedVerify.encodeRopeRows(enc, x: ckNormedBuf, out: ckRotBuf,
                                                  headDim: headDim, ropeDim: headDim, base: ropeTheta,
                                                  startOffset: ctxLen, M: ctxCount, numHeads: numKV)
                // commit ctx k/v into this layer's persistent KV cache at [ctxLen, ctxLen+ctxCount)
                SeedlessFusedVerify.encodeWriteKVRows(enc, src: ckRotBuf, cache: layer.kCache,
                                                      KV: numKV, D: headDim, maxLen: ctxCap, pos: ctxLen, M: ctxCount)
                SeedlessFusedVerify.encodeWriteKVRows(enc, src: cvBuf, cache: layer.vCache,
                                                      KV: numKV, D: headDim, maxLen: ctxCap, pos: ctxLen, M: ctxCount)
            }

            // SDPA: keys = cached[0..<cachedLen] ‖ propK[0..<blockSize]
            //   sliding (causalOffset=true):  row r sees cachedLen + (r+1) keys
            //   full    (causalOffset=false): every row sees cachedLen + blockSize keys (no mask)
            DFlashRawKernels.encode(enc, q: qRotBuf, kCache: layer.kCache, vCache: layer.vCache,
                                   propK: pkRotBuf, propV: pvBuf, out: attnOutBuf,
                                   numHeads: numHeads, numKV: numKV, D: headDim,
                                   cachedLen: cachedLen, ctxCap: ctxCap, M: blockSize,
                                   scale: scale, causalOffset: layer.isSliding)

            // o_proj — NO gate (unlike the target's attention)
            DFlashRawKernels.encodeFmmTiled(enc, w: layer.oW, x: attnOutBuf, out: attnResBuf,
                                              M: blockSize, K: numHeads * headDim, N: H)
            SeedlessFusedVerify.encodeResidAdd(enc, h: hBuf, r: attnResBuf, total: blockSize * H)

            // MLP (swiglu)
            SeedlessFusedVerify.encodeRmsNormRows(enc, x: hBuf, w: layer.postLN, out: mnBuf,
                                                  rows: blockSize, D: H, eps: eps)
            DFlashRawKernels.encodeFmmTiled(enc, w: layer.gateW, x: mnBuf, out: gateBuf,
                                              M: blockSize, K: H, N: mlpDim)
            DFlashRawKernels.encodeFmmTiled(enc, w: layer.upW, x: mnBuf, out: upBuf,
                                              M: blockSize, K: H, N: mlpDim)
            SeedlessFusedVerify.encodeSwiglu(enc, g: gateBuf, u: upBuf, h: actBuf, total: blockSize * mlpDim)
            DFlashRawKernels.encodeFmmTiled(enc, w: layer.downW, x: actBuf, out: downBuf,
                                              M: blockSize, K: mlpDim, N: H)
            SeedlessFusedVerify.encodeResidAdd(enc, h: hBuf, r: downBuf, total: blockSize * H)
        }

        split("layers")

        // final rmsNorm (drafter's OWN norm — not the target's final norm)
        SeedlessFusedVerify.encodeRmsNormRows(enc, x: hBuf, w: finalNorm, out: normedBuf,
                                              rows: blockSize, D: H, eps: eps)
        // target lm_head over ALL rows (row 0 discarded on readback — simpler than an x-offset
        // qmm variant, at the cost of one extra row of compute). qmm4_tiled (threadgroup
        // dequant shared across the M rows — same kernel stepArgmax uses for verify M>1):
        // the per-row qmv (encodeQmmRows) re-reads+dequants the whole 4-bit lm_head PER ROW,
        // ~8x the weight traffic at blockSize=8.
        do {
            let qp = SeedlessMetalForward._qmm4TiledPipeline!
            enc.setComputePipelineState(qp)
            enc.setBuffer(lw, offset: 0, index: 0); enc.setBuffer(ls, offset: 0, index: 1)
            enc.setBuffer(lb, offset: 0, index: 2); enc.setBuffer(normedBuf, offset: 0, index: 3)
            enc.setBuffer(logits, offset: 0, index: 4)
            var kk = Int32(H), nn = Int32(vocab), mm = Int32(blockSize)
            enc.setBytes(&kk, length: 4, index: 5); enc.setBytes(&nn, length: 4, index: 6)
            enc.setBytes(&mm, length: 4, index: 7)
            enc.dispatchThreadgroups(MTLSize(width: vocab, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }
        split("lmhead")
        encodeArgmax(enc, logits: logits, V: vocab)

        enc.endEncoding()
        let t0 = DispatchTime.now()
        cb.commit(); cb.waitUntilCompleted()
        if trace {
            let wallMs = Double(DispatchTime.now().uptimeNanoseconds - t0.uptimeNanoseconds) / 1e6
            stamps.append(("head", (cb.gpuEndTime - cb.gpuStartTime) * 1000.0))
            let parts = stamps.map { String(format: "%@=%.2fms", $0.0, $0.1) }.joined(separator: " ")
            FileHandle.standardError.write(Data(String(
                format: "[dflash-raw-time] %@ (tailWall=%.2fms) ctx=%d cached=%d\n",
                parts, wallMs, ctxCount, cachedLen).utf8))
        }

        ctxLen += ctxCount
        hasRun = true

        let ptr = tokenOutBuf.contents().bindMemory(to: Int32.self, capacity: blockSize)
        return (1 ..< blockSize).map { Int(ptr[$0]) }
    }

    // ── Rollback / reset (v1: no KV erase — tail rows are simply overwritten on next append) ──
    public func trimTo(committed: Int) {
        ctxLen = min(ctxLen, committed)
    }

    public func reset() {
        ctxLen = 0
    }

    // ── Test-only accessor ──────────────────────────────────────────────────
    public func lastFinalHidden() -> [Float16]? {
        guard hasRun else { return nil }
        let ptr = normedBuf.contents().bindMemory(to: Float16.self, capacity: blockSize * H)
        return Array(UnsafeBufferPointer(start: ptr, count: blockSize * H))
    }

    // ── Steady-state micro-bench (QWISP_RUN=dflash-raw-bench): back-to-back forwards on the
    //    real checkpoint + a synthetic 4-bit head, isolating drafter cost from decode-loop
    //    interleaving (GPU clock-ramp diagnosis: cold/idle-gap CBs downclock ~10x). ──
    public static func bench() -> String {
        let home = FileManager.default.homeDirectoryForCurrentUser.path
        let dir = ProcessInfo.processInfo.environment["QWISP_DFLASH_DIR"]
            ?? "\(home)/.mtplx/models/z-lab--Qwen3.6-35B-A3B-DFlash"
        guard let (cfg, weights) = Tell.loadDFlashConfigAndWeights(dir: URL(fileURLWithPath: dir)) else {
            return "[dflash-raw-bench] weights load failed (\(dir))"
        }
        var cfg8 = cfg
        cfg8.blockSize = 8
        guard let raw = DFlashRawDrafter(config: cfg8, weights: weights) else {
            return "[dflash-raw-bench] init failed"
        }
        // synthetic 4-bit head (test-91 idiom) — vocab shrunk: head cost measured separately below
        let V = 8192
        let ew = MLXRandom.normal([V, cfg.hiddenSize]).asType(.float16)
        let (ewq, es, ebO) = MLX.quantized(ew, groupSize: 64, bits: 4, mode: .affine)
        let lw = MLXRandom.normal([V, cfg.hiddenSize]).asType(.float16)
        let (lwq, ls, lbO) = MLX.quantized(lw, groupSize: 64, bits: 4, mode: .affine)
        guard let eBias = ebO, let lBias = lbO else { return "[dflash-raw-bench] quantize failed" }
        MLX.eval([ewq, es, eBias, lwq, ls, lBias])
        guard raw.attachTargetHead(embedW: ewq, embedS: es, embedB: eBias,
                                   lmW: lwq, lmS: ls, lmB: lBias, vocab: V) else {
            return "[dflash-raw-bench] attach failed"
        }
        let ctxRow = [Float16](repeating: Float16(0.01), count: 2 * cfg.ctxFeatureDim)
        // GPU clock-pinning ballast: a heavy dense matmul immediately before each drafter
        // forward. If the drafter collapses to a few ms with the ballast, the 20-50ms
        // "cost" is the frequency governor idling on an all-low-occupancy workload — the
        // production fix is then fusing the drafter into the verify CB, not more kernels.
        let bm = 2048
        let bA = MLXRandom.normal([bm, bm]).asType(.float16)
        let bB = MLXRandom.normal([bm, bm]).asType(.float16)
        MLX.eval([bA, bB])
        var lines: [String] = ["[dflash-raw-bench] 30 back-to-back forwards (real drafter weights, synthetic head V=\(V))"]
        for ballast in [false, true] {
            var times: [Double] = []
            for i in 0 ..< 30 {
                if raw.ctxLen + 2 > raw.ctxCap { raw.reset() }
                if ballast {
                    let c = MLX.matmul(bA, bB)
                    c.eval()
                }
                let t0 = DispatchTime.now()
                guard raw.forward(u: 7, ctxRows: ctxRow, ctxCount: 2) != nil else {
                    return "[dflash-raw-bench] forward nil at iter \(i)"
                }
                let ms = Double(DispatchTime.now().uptimeNanoseconds - t0.uptimeNanoseconds) / 1e6
                times.append(ms)
            }
            let head = times.prefix(3).map { String(format: "%.2f", $0) }.joined(separator: ", ")
            let tail = times.suffix(20)
            let mean = tail.reduce(0, +) / Double(tail.count)
            let mn = tail.min() ?? 0
            lines.append(String(format: "  ballast=%@  first3: [%@] ms   steady(last20): mean=%.2fms min=%.2fms",
                                ballast ? "ON " : "off", head, mean, mn))
        }
        return lines.joined(separator: "\n")
    }
}

// ── New Metal source: the drafter's own SDPA kernel ─────────────────────────
// Not reusable from the frozen files: the existing `sdpa_rows` kernel hardcodes headDim=256
// at compile time (drafter headDim is 16 in tests / 128 in production), and the drafter has
// no gate path so the gated attn_q_prep_rows/attn_k_prep_rows kernels don't apply either.
//
// Occupancy fix (round 2, adversarial review g5): v1 dispatched one thread per (head,row) —
// only numHeads*M threads total (256 in the real config), each a fully serial loop over up to
// ~4095 cached keys. That starves the GPU (far below the thread count needed to fill it) and
// measured 2.3x SLOWER than the MLX drafter it's meant to beat. Fix: one SIMD-group (32 lanes)
// per (head,row) query instead of one thread — lanes split the key loop by stride-32 (same
// idiom as GroupedMoEPoC/SeedlessFusedVerify's simd_sum row-reduction kernels already in this
// codebase), each lane keeping its own partial online-softmax (max, sum, acc[D]), then a single
// simd_max/simd_sum merge combines the 32 partials. Same total FLOPs, 32x more concurrent
// threadgroups (numHeads*M threadgroups of 32 vs previously ceil(numHeads/32)*M groups), ~32x
// less serial depth per lane. Key→lane assignment depends only on absolute key position, so the
// two-call-vs-one-call byte-exact cache-consistency contract (locked test 97) is unaffected.
private enum DFlashRawKernels {
    nonisolated(unsafe) static var sdpaPipeline: MTLComputePipelineState?
    nonisolated(unsafe) static var fmmTiledPipeline: MTLComputePipelineState?

    static func ensureSdpaPipeline(_ device: MTLDevice) -> Bool {
        if sdpaPipeline != nil { return true }
        let src = """
        #include <metal_stdlib>
        using namespace metal;
        // dflash_fmm_tiled: out[M,N] = x[M,K] @ W[N,K]^T, all f16. One 256-thread threadgroup
        // per output column n; the K-strided W-row read is coalesced and stays hot in cache
        // across the M-loop, so W traffic is ~once per block instead of once per (row, col)
        // thread. Replaces fmm_rows for the drafter: fmm_rows (one thread per (n,m), 16-wide
        // groups, serial K) re-reads every W row M times at poor effective bandwidth — fine
        // for the 1-layer M<=2 MTP head it was built for, catastrophic for 6 layers x M=8
        // (measured 165-170ms GPU per block, ~10x the MLX drafter it was meant to beat).
        kernel void dflash_fmm_tiled(
            device const half* W   [[buffer(0)]],   // [N, K]
            device const half* x   [[buffer(1)]],   // [M, K]
            device half*       out [[buffer(2)]],   // [M, N]
            constant int&      K   [[buffer(3)]],
            constant int&      N   [[buffer(4)]],
            constant int&      M   [[buffer(5)]],
            uint n   [[threadgroup_position_in_grid]],
            uint tid [[thread_index_in_threadgroup]],
            uint tgs [[threads_per_threadgroup]])
        {
            // dq_tiled structure (GroupedMoEPoC denseBench, the measured-good dense shape):
            // stage the W row into threadgroup memory in K-chunks (read from device ONCE),
            // then each m-pass dots against TG memory — W re-reads are free, x passes are
            // coalesced (per-m contiguous) and L2-hot (x is 32KB total). Two earlier
            // variants measured bad: m-outer device re-reads (L1 thrash across TGs, ~20ms)
            // and m-inner strided x (8 cache lines per k per thread, ~48ms).
            constexpr int MAXM = 16;
            constexpr int CHUNK = 2048;
            threadgroup half ws[CHUNK];
            threadgroup float red[256];
            const device half* wrow = W + (size_t)n * (size_t)K;
            float acc[MAXM];
            for (int m = 0; m < M; m++) acc[m] = 0.0f;
            for (int k0 = 0; k0 < K; k0 += CHUNK) {
                int kc = min(CHUNK, K - k0);
                for (int k = (int)tid; k < kc; k += (int)tgs) ws[k] = wrow[k0 + k];
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (int m = 0; m < M; m++) {
                    const device half* xrow = x + (size_t)m * (size_t)K + k0;
                    float p = 0.0f;
                    for (int k = (int)tid; k < kc; k += (int)tgs)
                        p += (float)xrow[k] * (float)ws[k];
                    acc[m] += p;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            for (int m = 0; m < M; m++) {
                red[tid] = acc[m]; threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint s = tgs / 2; s > 0; s >>= 1) {
                    if (tid < s) red[tid] += red[tid + s];
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
                if (tid == 0) out[m * N + (int)n] = (half)red[0];
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
        // dflash_sdpa_rows: keys/values = cache[0..<cachedLen] ‖ prop[0..<propKeys(m)].
        // causalOffset!=0: propKeys(m) = m+1 (causal-with-offset). causalOffset==0: propKeys(m) = M
        // (no mask, bidirectional within the block). Cached keys are ALWAYS fully visible.
        // One threadgroup == one simdgroup (32 lanes) == one (head, query-row) pair; lanes
        // stride-partition the key loop, then merge via simd_max/simd_sum (online-softmax merge).
        kernel void dflash_sdpa_rows(
            device const half* q        [[buffer(0)]],   // [M*numHeads, D]
            device const half* kCache   [[buffer(1)]],   // [numKV, ctxCap, D]
            device const half* vCache   [[buffer(2)]],   // [numKV, ctxCap, D]
            device const half* propK    [[buffer(3)]],   // [M*numKV, D]
            device const half* propV    [[buffer(4)]],   // [M*numKV, D]
            device half* out            [[buffer(5)]],   // [M*numHeads, D]
            constant uint& numHeads     [[buffer(6)]],
            constant uint& numKV        [[buffer(7)]],
            constant uint& D            [[buffer(8)]],
            constant uint& cachedLen    [[buffer(9)]],
            constant uint& ctxCap      [[buffer(10)]],
            constant uint& M            [[buffer(11)]],
            constant float& scale       [[buffer(12)]],
            constant uint& causalOffset [[buffer(13)]],
            uint2 tgid    [[threadgroup_position_in_grid]],
            uint  lane    [[thread_index_in_simdgroup]])
        {
            // Key-serial / D-parallel dataflow: keys are walked SERIALLY (deterministic order —
            // locked test 97's two-call-vs-one-shot byte-exactness needs a key order that
            // depends only on absolute position), the 32 lanes cooperate on each key: the
            // q·k dot is a stride-32 partial + simd_sum broadcast, and the online-softmax
            // accumulator lives DISTRIBUTED across lanes (D/32 floats each — registers, no
            // spill). The previous lane-per-key variant kept a full acc[256] per lane, which
            // spilled and ran slower than the MLX drafter it was meant to replace.
            uint h = tgid.x, m = tgid.y;
            uint gqa = numHeads / numKV;
            uint kvh = h / gqa;
            uint propKeys = (causalOffset != 0) ? (m + 1) : M;
            uint totalKeys = cachedLen + propKeys;
            constexpr uint MAXDL = 8;              // supports D up to 8*32 = 256
            uint perLane = (D + 31) / 32;
            const device half* qrow = q + (m * numHeads + h) * D;
            float qv[MAXDL];
            float acc[MAXDL];
            for (uint t = 0; t < perLane; t++) {
                uint d = lane + t * 32;
                qv[t] = (d < D) ? (float)qrow[d] : 0.0f;
                acc[t] = 0.0f;
            }
            float runMax = -INFINITY, runSum = 0.0f;
            for (uint j = 0; j < totalKeys; j++) {
                const device half* krow;
                const device half* vrow;
                if (j < cachedLen) {
                    krow = kCache + (kvh * ctxCap + j) * D;
                    vrow = vCache + (kvh * ctxCap + j) * D;
                } else {
                    uint pj = j - cachedLen;
                    krow = propK + (pj * numKV + kvh) * D;
                    vrow = propV + (pj * numKV + kvh) * D;
                }
                float part = 0.0f;
                for (uint t = 0; t < perLane; t++) {
                    uint d = lane + t * 32;
                    if (d < D) part += qv[t] * (float)krow[d];
                }
                float score = simd_sum(part) * scale;   // broadcast: every lane holds the score
                float newMax = max(runMax, score);
                float factor = fast::exp(runMax - newMax);
                float e = fast::exp(score - newMax);
                runMax = newMax;
                runSum = runSum * factor + e;
                for (uint t = 0; t < perLane; t++) {
                    uint d = lane + t * 32;
                    if (d < D) acc[t] = acc[t] * factor + e * (float)vrow[d];
                }
            }
            device half* orow = out + (m * numHeads + h) * D;
            float inv = runSum > 0.0f ? (1.0f / runSum) : 1.0f;
            for (uint t = 0; t < perLane; t++) {
                uint d = lane + t * 32;
                if (d < D) orow[d] = (half)(acc[t] * inv);
            }
        }
        """
        do {
            let lib = try device.makeLibrary(source: src, options: SeedlessMetalForward.mlxMatchCompileOpts())
            sdpaPipeline = try device.makeComputePipelineState(function: lib.makeFunction(name: "dflash_sdpa_rows")!)
            fmmTiledPipeline = try device.makeComputePipelineState(function: lib.makeFunction(name: "dflash_fmm_tiled")!)
        } catch { print("[dflash-raw-sdpa] compile: \(error)"); return false }
        return sdpaPipeline != nil && fmmTiledPipeline != nil
    }

    /// Tiled f16 matmul: out[M,N] = x[M,K] @ W[N,K]^T. Drop-in for the drafter's projection/
    /// MLP/fc sites (bandwidth-correct replacement for fmm_rows at M=blockSize x 6 layers).
    static func encodeFmmTiled(_ enc: MTLComputeCommandEncoder,
                               w: MTLBuffer, x: MTLBuffer, out: MTLBuffer,
                               M: Int, K: Int, N: Int) {
        let p = fmmTiledPipeline!
        enc.setComputePipelineState(p)
        enc.setBuffer(w, offset: 0, index: 0)
        enc.setBuffer(x, offset: 0, index: 1)
        enc.setBuffer(out, offset: 0, index: 2)
        var kk = Int32(K), nn = Int32(N), mm = Int32(M)
        enc.setBytes(&kk, length: 4, index: 3)
        enc.setBytes(&nn, length: 4, index: 4)
        enc.setBytes(&mm, length: 4, index: 5)
        enc.dispatchThreadgroups(MTLSize(width: N, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    static func encode(_ enc: MTLComputeCommandEncoder,
                       q: MTLBuffer, kCache: MTLBuffer, vCache: MTLBuffer,
                       propK: MTLBuffer, propV: MTLBuffer, out: MTLBuffer,
                       numHeads: Int, numKV: Int, D: Int,
                       cachedLen: Int, ctxCap: Int, M: Int,
                       scale: Float, causalOffset: Bool) {
        let p = sdpaPipeline!
        enc.setComputePipelineState(p)
        enc.setBuffer(q, offset: 0, index: 0)
        enc.setBuffer(kCache, offset: 0, index: 1)
        enc.setBuffer(vCache, offset: 0, index: 2)
        enc.setBuffer(propK, offset: 0, index: 3)
        enc.setBuffer(propV, offset: 0, index: 4)
        enc.setBuffer(out, offset: 0, index: 5)
        var nh = UInt32(numHeads), nkv = UInt32(numKV), dd = UInt32(D)
        var cl = UInt32(cachedLen), cc = UInt32(ctxCap), mm = UInt32(M)
        var sc = scale, co = UInt32(causalOffset ? 1 : 0)
        enc.setBytes(&nh, length: 4, index: 6); enc.setBytes(&nkv, length: 4, index: 7)
        enc.setBytes(&dd, length: 4, index: 8); enc.setBytes(&cl, length: 4, index: 9)
        enc.setBytes(&cc, length: 4, index: 10); enc.setBytes(&mm, length: 4, index: 11)
        enc.setBytes(&sc, length: 4, index: 12); enc.setBytes(&co, length: 4, index: 13)
        // One threadgroup (== one simdgroup, 32 lanes) per (head, row) query — numHeads*M
        // threadgroups total, vs the old dispatchThreads' ceil(numHeads/32)*M groups of 32
        // serial-loop threads. See kernel comment above for the occupancy rationale.
        enc.dispatchThreadgroups(MTLSize(width: numHeads, height: M, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    }
}
