import Foundation
import MLX
import Metal
import MetalPerformanceShaders

// #98 DFlash phase A1 — raw-Metal drafter forward (spec: scratchpad/dflash-phaseA1-spec.md).
//
// ONE command buffer per block: persistent f16 MTLBuffer weights (COPIED into Metal-owned
// buffers at init — no noCopy lifetime obligation), persistent per-layer KV MTLBuffers.
// GEMMs (q/k/v/o, MLP, fc, lm_head) are MPSMatrixMultiplication — the hand-rolled
// fmm/qmm4 kernels measured 75-100ms/block at drafter shapes (M=8, 6 thin layers; ~2-3ms
// roofline), overhead-dominated across 3 kernel designs (#98 A1 record). Small ops reuse
// the frozen SeedlessFusedVerify/SeedlessMetalForward statics (rmsNormRows, ropeRows,
// embed_rows_q4, argmax_rows, swiglu, resid_add, write_kv_rows). The ONLY new Metal source
// is the drafter's own SDPA kernel (below) — the drafter has no gate path and no
// partial-rotary rope, so the existing gated-attention prep kernels don't fit and the
// existing D=256-hardcoded sdpa_rows kernel doesn't fit a runtime headDim.
//
// Fused draft+verify (#98 A1 round 3): prepare() → encode(cb:) → finish() lets the verify
// CB carry the drafter as a prologue (MPS opens its own internal encoders, so the seam is
// CB-level); forward() composes the three standalone.
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
    // lm_head is DEQUANTIZED to f16 (V×H×2 ≈ 0.6GB for the real head): the drafter is
    // resident-tier-only + opt-in, and the 4-bit qmm4_tiled path measured 33ms at drafter
    // shapes (overhead-dominated) vs ~3ms roofline for an MPS f16 GEMM. Embed stays 4-bit
    // (per-row gather is cheap).
    private var embedWBuf: MTLBuffer?, embedSBuf: MTLBuffer?, embedBBuf: MTLBuffer?
    private var lmF16Buf: MTLBuffer?
    private var vocab: Int = 0
    private var logitsBuf: MTLBuffer?

    // ── MPS GEMM plumbing (#98 A1 round 3): the hand-rolled fmm/qmm kernels measured
    // 75-100ms/block at drafter shapes (M=8, 6 thin layers) vs ~2-3ms roofline — the pinned
    // fallback lever is MPSMatrixMultiplication, which encodes into the SAME command buffer
    // (its own internal encoders), so the fused draft+verify CB structure is preserved.
    private var mpsMulCache: [String: MPSMatrixMultiplication] = [:]
    private var _cb: MTLCommandBuffer?
    private var _enc: MTLComputeCommandEncoder?
    private func enc() -> MTLComputeCommandEncoder {
        if _enc == nil { _enc = _cb!.makeComputeCommandEncoder()! }
        return _enc!
    }
    private func closeEnc() { _enc?.endEncoding(); _enc = nil }
    private func mpsMat(_ buf: MTLBuffer, _ rows: Int, _ cols: Int, off: Int = 0) -> MPSMatrix {
        MPSMatrix(buffer: buf, offset: off, descriptor: MPSMatrixDescriptor(
            rows: rows, columns: cols, rowBytes: cols * 2, dataType: .float16))
    }
    /// out[M,N] = x[M,K] @ W[N,K]^T, all f16 (f32 accumulate inside MPS).
    private func gemm(w: MTLBuffer, x: MTLBuffer, out: MTLBuffer, M: Int, K: Int, N: Int,
                      xOff: Int = 0, outOff: Int = 0) {
        closeEnc()
        let key = "\(M)/\(K)/\(N)"
        let mul: MPSMatrixMultiplication
        if let m = mpsMulCache[key] { mul = m } else {
            mul = MPSMatrixMultiplication(device: device, transposeLeft: false, transposeRight: true,
                                          resultRows: M, resultColumns: N, interiorColumns: K,
                                          alpha: 1.0, beta: 0.0)
            mpsMulCache[key] = mul
        }
        mul.encode(commandBuffer: _cb!, leftMatrix: mpsMat(x, M, K, off: xOff),
                   rightMatrix: mpsMat(w, N, K), resultMatrix: mpsMat(out, M, N, off: outOff))
    }
    /// ctx-path GEMM: one M=1 GEMM PER ROW. ctxCount varies call-to-call and batched MPS
    /// output is not M-invariant at the byte level — but the drafter KV cache must be
    /// byte-stable across call splits (two-call vs one-shot, locked test 97). Per-row M=1
    /// makes every row an identical subproblem regardless of batching.
    private func gemmPerRow(w: MTLBuffer, x: MTLBuffer, out: MTLBuffer, rows: Int, K: Int, N: Int) {
        for r in 0 ..< rows {
            gemm(w: w, x: x, out: out, M: 1, K: K, N: N, xOff: r * K * 2, outOff: r * N * 2)
        }
    }

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
        // COPY into a Metal-owned buffer (not bytesNoCopy aliasing MLX memory): the drafter
        // weights are touched only ~every 50ms for ~2ms — noCopy MLX-backed buffers measured
        // ~17GB/s effective GPU read at these shapes (identical timings across 2 unrelated
        // GEMM implementations = shared-resource bound), Metal-owned pages read at roofline.
        // Copy also drops the noCopy lifetime obligation for these.
        func f16b(_ a: MLXArray) -> MTLBuffer? {
            let c = a.asType(.float16); c.eval()
            guard let src = SeedlessMetalForward.mtlBuf(c, dev),
                  let dst = dev.makeBuffer(length: src.length, options: .storageModeShared)
            else { return nil }
            memcpy(dst.contents(), src.contents(), src.length)
            return dst
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
        // COPY into Metal-owned buffers (same rationale as the init's f16b — the noCopy
        // MLX-backed alias reads ~17GB/s at drafter access patterns).
        func copyb(_ a: MLXArray) -> MTLBuffer? {
            a.eval()
            guard let src = SeedlessMetalForward.mtlBuf(a, device),
                  let dst = device.makeBuffer(length: src.length, options: .storageModeShared)
            else { return nil }
            memcpy(dst.contents(), src.contents(), src.length)
            return dst
        }
        func f16b(_ a: MLXArray) -> MTLBuffer? { copyb(a.asType(.float16)) }
        // lm_head → dequantized f16 [V, H] for the MPS GEMM (see field comment).
        let lmF16 = MLX.dequantized(lmW, scales: lmS, biases: lmB,
                                    groupSize: 64, bits: 4, mode: .affine).asType(.float16)
        guard let ew = copyb(embedW), let es = f16b(embedS), let eb = f16b(embedB),
              let lf = f16b(lmF16),
              let logits = device.makeBuffer(length: blockSize * vocab * 2, options: .storageModeShared)
        else { return false }
        embedWBuf = ew; embedSBuf = es; embedBBuf = eb
        lmF16Buf = lf
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

    // ── forward, split into prepare / encode / finish ───────────────────────
    // #98 A1 fused draft+verify: the verify CB encodes the drafter as a prologue
    // (prepare → encode(callerEnc) → [caller commits] → finish). forward() composes the
    // three on a private CB for the standalone path (tests / bench / dispatch fallback).

    private var pendingCount = -1   // prepared ctxCount; -1 = no prepare() outstanding

    /// CPU-side stage: attach + ctx-cap guards, token & ctx scratch upload. No GPU work,
    /// no state advance — abandoning after prepare() (fused-step failure) is side-effect-free.
    public func prepare(u: Int32, ctxRows: [Float16], ctxCount: Int) -> Bool {
        guard embedWBuf != nil, logitsBuf != nil else { return false }   // attachTargetHead not called
        guard ctxLen + ctxCount <= ctxCap else { return false }          // v1 cap contract

        // 1. noise tokens: [u] ++ [maskId × (blockSize-1)]
        let tokPtr = tokBuf.contents().bindMemory(to: Int32.self, capacity: blockSize)
        tokPtr[0] = u
        for i in 1 ..< blockSize { tokPtr[i] = Int32(config.maskTokenId) }

        // 2. ctx rows upload (persistent scratch buffer, refreshed each call)
        if ctxCount > 0 {
            ctxRows.withUnsafeBytes { raw in
                memcpy(ctxInputBuf.contents(), raw.baseAddress!, ctxCount * ctxFeatureDim * 2)
            }
        }
        pendingCount = ctxCount
        return true
    }

    /// GPU-side stage: encodes the whole drafter block (embed → ctx fc → layers → lm_head →
    /// argmax → tokenOutBuf) into the caller's command buffer. GEMMs go through MPS (which
    /// opens its own internal encoders — hence the CB-level seam, not an encoder-level one);
    /// the small ops (norm/rope/SDPA/swiglu/residAdd/writeKV/argmax) use the existing raw
    /// pipelines on interleaved compute encoders. Requires a successful prepare().
    public func encode(cb: MTLCommandBuffer) {
        guard pendingCount >= 0 else { return }
        let ctxCount = pendingCount
        let ew = embedWBuf!, es = embedSBuf!, eb = embedBBuf!
        let lf = lmF16Buf!, logits = logitsBuf!

        let cachedLen = ctxLen + ctxCount   // KV length valid after this call's ctx append
        let scale = Float(pow(Double(headDim), -0.5))
        _cb = cb
        defer { closeEnc(); _cb = nil }

        // noise embed → hBuf (residual stream init)
        encodeEmbedNoise(enc(), ew: ew, es: es, eb: eb)

        // ctx feature path (computed ONCE, shared by every layer's k/v proj)
        if ctxCount > 0 {
            gemmPerRow(w: fcW, x: ctxInputBuf, out: fcOutBuf, rows: ctxCount, K: ctxFeatureDim, N: H)
            SeedlessFusedVerify.encodeRmsNormRows(enc(), x: fcOutBuf, w: hiddenNorm, out: hCtxBuf,
                                                  rows: ctxCount, D: H, eps: eps)
        }

        for layer in layers {
            // attn input norm (block rows only — ctx k/v project straight from hCtx, no inputLN)
            SeedlessFusedVerify.encodeRmsNormRows(enc(), x: hBuf, w: layer.inputLN, out: anLNBuf,
                                                  rows: blockSize, D: H, eps: eps)
            // q / propK / propV projections (+ ctx k/v from hCtx — grouped into ONE MPS run)
            gemm(w: layer.qW, x: anLNBuf, out: qRawBuf, M: blockSize, K: H, N: numHeads * headDim)
            gemm(w: layer.kW, x: anLNBuf, out: pkRawBuf, M: blockSize, K: H, N: numKV * headDim)
            gemm(w: layer.vW, x: anLNBuf, out: pvBuf, M: blockSize, K: H, N: numKV * headDim)
            if ctxCount > 0 {
                // ctx k/v (this layer's own k/v proj — NOT re-normed by inputLN)
                gemmPerRow(w: layer.kW, x: hCtxBuf, out: ckRawBuf, rows: ctxCount, K: H, N: numKV * headDim)
                gemmPerRow(w: layer.vW, x: hCtxBuf, out: cvBuf, rows: ctxCount, K: H, N: numKV * headDim)
            }
            // q/k rmsNorm (last-dim headDim, treating each (row,head) as an independent chunk)
            SeedlessFusedVerify.encodeRmsNormRows(enc(), x: qRawBuf, w: layer.qNorm, out: qNormedBuf,
                                                  rows: blockSize * numHeads, D: headDim, eps: eps)
            SeedlessFusedVerify.encodeRmsNormRows(enc(), x: pkRawBuf, w: layer.kNorm, out: pkNormedBuf,
                                                  rows: blockSize * numKV, D: headDim, eps: eps)
            // FULL-headDim RoPE: q and propK at position cachedLen + row
            SeedlessFusedVerify.encodeRopeRows(enc(), x: qNormedBuf, out: qRotBuf,
                                              headDim: headDim, ropeDim: headDim, base: ropeTheta,
                                              startOffset: cachedLen, M: blockSize, numHeads: numHeads)
            SeedlessFusedVerify.encodeRopeRows(enc(), x: pkNormedBuf, out: pkRotBuf,
                                              headDim: headDim, ropeDim: headDim, base: ropeTheta,
                                              startOffset: cachedLen, M: blockSize, numHeads: numKV)

            if ctxCount > 0 {
                SeedlessFusedVerify.encodeRmsNormRows(enc(), x: ckRawBuf, w: layer.kNorm, out: ckNormedBuf,
                                                      rows: ctxCount * numKV, D: headDim, eps: eps)
                // ctxK at position ctxLen(before this call) + row
                SeedlessFusedVerify.encodeRopeRows(enc(), x: ckNormedBuf, out: ckRotBuf,
                                                  headDim: headDim, ropeDim: headDim, base: ropeTheta,
                                                  startOffset: ctxLen, M: ctxCount, numHeads: numKV)
                // commit ctx k/v into this layer's persistent KV cache at [ctxLen, ctxLen+ctxCount)
                SeedlessFusedVerify.encodeWriteKVRows(enc(), src: ckRotBuf, cache: layer.kCache,
                                                      KV: numKV, D: headDim, maxLen: ctxCap, pos: ctxLen, M: ctxCount)
                SeedlessFusedVerify.encodeWriteKVRows(enc(), src: cvBuf, cache: layer.vCache,
                                                      KV: numKV, D: headDim, maxLen: ctxCap, pos: ctxLen, M: ctxCount)
            }

            // SDPA: keys = cached[0..<cachedLen] ‖ propK[0..<blockSize]
            //   sliding (causalOffset=true):  row r sees cachedLen + (r+1) keys
            //   full    (causalOffset=false): every row sees cachedLen + blockSize keys (no mask)
            DFlashRawKernels.encode(enc(), q: qRotBuf, kCache: layer.kCache, vCache: layer.vCache,
                                   propK: pkRotBuf, propV: pvBuf, out: attnOutBuf,
                                   numHeads: numHeads, numKV: numKV, D: headDim,
                                   cachedLen: cachedLen, ctxCap: ctxCap, M: blockSize,
                                   scale: scale, causalOffset: layer.isSliding)

            // o_proj — NO gate (unlike the target's attention)
            gemm(w: layer.oW, x: attnOutBuf, out: attnResBuf, M: blockSize, K: numHeads * headDim, N: H)
            SeedlessFusedVerify.encodeResidAdd(enc(), h: hBuf, r: attnResBuf, total: blockSize * H)

            // MLP (swiglu)
            SeedlessFusedVerify.encodeRmsNormRows(enc(), x: hBuf, w: layer.postLN, out: mnBuf,
                                                  rows: blockSize, D: H, eps: eps)
            gemm(w: layer.gateW, x: mnBuf, out: gateBuf, M: blockSize, K: H, N: mlpDim)
            gemm(w: layer.upW, x: mnBuf, out: upBuf, M: blockSize, K: H, N: mlpDim)
            SeedlessFusedVerify.encodeSwiglu(enc(), g: gateBuf, u: upBuf, h: actBuf, total: blockSize * mlpDim)
            gemm(w: layer.downW, x: actBuf, out: downBuf, M: blockSize, K: mlpDim, N: H)
            SeedlessFusedVerify.encodeResidAdd(enc(), h: hBuf, r: downBuf, total: blockSize * H)
        }

        // final rmsNorm (drafter's OWN norm — not the target's final norm)
        SeedlessFusedVerify.encodeRmsNormRows(enc(), x: hBuf, w: finalNorm, out: normedBuf,
                                              rows: blockSize, D: H, eps: eps)
        // target lm_head over ALL rows (row 0 discarded on readback — simpler than an x-offset
        // variant, at the cost of one extra row of compute). f16 MPS GEMM on the dequantized
        // head (see lmF16Buf comment).
        gemm(w: lf, x: normedBuf, out: logits, M: blockSize, K: H, N: vocab)
        encodeArgmax(enc(), logits: logits, V: vocab)
    }

    /// Post-completion stage: advances ctxLen and reads the draft token ids (rows 1..<block).
    /// Call ONLY after the CB containing encode() has completed.
    public func finish() -> [Int] {
        ctxLen += max(0, pendingCount)
        pendingCount = -1
        hasRun = true
        let ptr = tokenOutBuf.contents().bindMemory(to: Int32.self, capacity: blockSize)
        return (1 ..< blockSize).map { Int(ptr[$0]) }
    }

    /// Fused-CB seam: the drafter's argmax output ([blockSize] int32; row 0 = u's own argmax,
    /// discarded) — blit source for the verify tokensIn.
    public var tokenOutBuffer: MTLBuffer { tokenOutBuf }

    /// Standalone forward (tests / bench / MLX-fallback dispatch path): private 1-CB compose
    /// of prepare → encode → finish.
    public func forward(u: Int32, ctxRows: [Float16], ctxCount: Int) -> [Int]? {
        guard prepare(u: u, ctxRows: ctxRows, ctxCount: ctxCount) else { return nil }
        let cb = queue.makeCommandBuffer()!
        encode(cb: cb)
        cb.commit(); cb.waitUntilCompleted()
        if Tell.envFlag("QWISP_DFLASH_TRACE") {
            FileHandle.standardError.write(Data(String(
                format: "[dflash-raw-time] gpu=%.2fms ctx=%d cached=%d\n",
                (cb.gpuEndTime - cb.gpuStartTime) * 1000.0, ctxCount, ctxLen + ctxCount).utf8))
        }
        return finish()
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

    static func ensureSdpaPipeline(_ device: MTLDevice) -> Bool {
        if sdpaPipeline != nil { return true }
        let src = """
        #include <metal_stdlib>
        using namespace metal;
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
        } catch { print("[dflash-raw-sdpa] compile: \(error)"); return false }
        return sdpaPipeline != nil
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
