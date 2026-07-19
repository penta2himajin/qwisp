import Foundation
import MLX
import Metal

/// #98 DFlash phase 3: dispatch wiring.
///
/// Drives the DFlash block drafter (DFlashDrafter) on the draftless (D==0) decode span,
/// gated by `QWISP_DFLASH` (default OFF ⇒ this object is never constructed and every path
/// stays byte-identical). Resident tier only (v1 tier scoping).
///
/// Divergence from the Python `model_mlx` reference: v1 feeds ONLY committed rows as ctx
/// (pendingCtx is appended post-accept, in the tap harvest), so the drafter cache never
/// contains rejected rows and `trimTo` is not called by the dispatch. (model_mlx trims
/// because it feeds hidden before knowing the accept; we don't.)
///
public final class DFlashDispatch {
    let drafter: DFlashDrafter
    // #98 phase A1: raw-first draft path. nil ⇒ MLX-only (load/attach failure, or the
    // caller never built one). QWISP_DFLASH_MLX=1 forces the MLX path even when raw is
    // available (A/B measurement seam). Raw returns nil past its v1 ctx cap (4095 committed
    // rows) or once attachTargetHead was never called — draft() falls back to MLX either way.
    var raw: DFlashRawDrafter?
    var caches: [DFlashKVCache]
    let blockSize: Int                          // QWISP_DFLASH_BLOCK, default 8
    let maskId: Int32                           // drafter.config.maskTokenId
    let nTap: Int, H: Int, ctxDim: Int          // nTap = |targetLayerIds|; ctxDim = ctxFeatureDim; H = ctxDim/nTap (tap hidden)
    // Injected hooks (testability: locked tests pass synthetic closures):
    let embedFn: ([Int32]) -> MLXArray?         // engine.embed wrapped → [M,H] f16
    let logitsArgmaxFn: (MLXArray) -> [Int]?    // drafter hidden [L,H] → per-row argmax ids
    // ctx accumulation (pending until next draft call):
    private(set) var pendingCtx: [Float16] = [] // pendingRows * ctxDim, row-major
    private(set) var pendingRows: Int = 0
    private(set) var ctxRowsFed: Int = 0        // total ctx rows in drafter cache

    public init(drafter: DFlashDrafter, raw: DFlashRawDrafter? = nil, blockSize: Int,
                embedFn: @escaping ([Int32]) -> MLXArray?,
                logitsArgmaxFn: @escaping (MLXArray) -> [Int]?) {
        self.drafter = drafter
        self.raw = raw
        self.caches = drafter.makeCaches()
        self.blockSize = blockSize
        self.maskId = Int32(drafter.config.maskTokenId)
        self.nTap = drafter.config.targetLayerIds.count
        self.ctxDim = drafter.config.ctxFeatureDim
        self.H = self.nTap > 0 ? self.ctxDim / self.nTap : 0
        self.embedFn = embedFn
        self.logitsArgmaxFn = logitsArgmaxFn
    }

    /// Append committed rows [0,rows) of the per-forward tap buffer to pendingCtx.
    /// For row r, appends the concatenation over tap slots t=0..<nTap of
    /// `tapBuf[t*maxM*H + r*H ..< +H]` (tap-slot order = targetLayerIds order).
    public func appendCtx(tapBuf: MTLBuffer, maxM: Int, rows: Int) {
        guard rows > 0, nTap > 0, H > 0 else { return }
        let ptr = tapBuf.contents().assumingMemoryBound(to: Float16.self)
        pendingCtx.reserveCapacity(pendingCtx.count + rows * ctxDim)
        for r in 0 ..< rows {
            for t in 0 ..< nTap {
                let base = t * maxM * H + r * H
                for h in 0 ..< H { pendingCtx.append(ptr[base + h]) }
            }
        }
        pendingRows += rows
    }

    /// Returns blockSize-1 draft token ids, or nil (caller falls back to the old path).
    /// v1 semantics (see file-header note): only called with pendingRows >= 1 (there is
    /// always >= 1 new committed row since the last draft — the anchor commit). Cold start
    /// (pendingRows == 0, nothing harvested yet) and "no new ctx since the last draft" both
    /// collapse to the same `pendingRows == 0` check.
    public func draft(u: Int) -> [Int]? {
        guard pendingRows > 0 else { return nil }
        // Raw-first (#98 phase A1): try the one-CB raw forward before the MLX path. nil
        // (attach missing, or v1 ctx-cap exceeded) falls through to MLX unchanged — pendingCtx
        // is untouched until whichever path actually consumes it.
        if let raw, !Tell.envFlag("QWISP_DFLASH_MLX"),
           let ids = raw.forward(u: Int32(u), ctxRows: pendingCtx, ctxCount: pendingRows) {
            ctxRowsFed += pendingRows
            pendingCtx = []
            pendingRows = 0
            return ids
        }
        let tokens: [Int32] = [Int32(u)] + Array(repeating: maskId, count: blockSize - 1)
        guard let embedded = embedFn(tokens) else { return nil }
        let noise = embedded.reshaped([1, blockSize, -1])
        let ctx = MLXArray(pendingCtx, [1, pendingRows, ctxDim])
        let hidden = drafter.forward(noise: noise, ctx: ctx, caches: caches)
        ctxRowsFed += pendingRows
        pendingCtx = []
        pendingRows = 0
        guard let ids = logitsArgmaxFn(hidden[0, 1...]), ids.count == blockSize - 1 else { return nil }
        // Materialize drafter cache state so the lazy graph doesn't accumulate across
        // blocks (LayerCache.stateArrays idiom) — the arrays were already computed as part
        // of the ids readback, so this is a cheap pin, not a second compute.
        let state = caches.flatMap { c in [c.keys, c.values].compactMap { $0 } }
        if !state.isEmpty { MLX.eval(state) }
        return ids
    }

    /// #98 A1 fused draft+verify: encodes the raw drafter as a prologue of the verify CB on
    /// `fwd` (MUST be the same forward the spec loop verifies on — the fused step advances its
    /// cache exactly like stepArgmax([u]+drafts)). ONE CB / ONE readback returns both the
    /// draft ids and the verify argmax rows; the caller computes the accept prefix on CPU.
    /// nil → nothing ran GPU-side (prepare is scratch-only); caller falls back to
    /// draft()+stepArgmax unchanged.
    public func draftFused(u: Int, fwd: SeedlessFusedVerify.SeedlessFusedForward)
        -> (drafts: [Int], evals: [Int])? {
        guard pendingRows > 0, let raw, !Tell.envFlag("QWISP_DFLASH_MLX"),
              !Tell.envFlag("QWISP_DFLASH_NOFUSE"),   // A/B: force the split raw path
              raw.prepare(u: Int32(u), ctxRows: pendingCtx, ctxCount: pendingRows)
        else { return nil }
        var toks = [Int32](repeating: 0, count: blockSize)  // rows 1.. GPU-overwritten by the blit
        toks[0] = Int32(u)
        let t0 = DispatchTime.now()
        guard let evals = fwd.stepArgmax(toks, draftPrologue: { [raw] cb in raw.encode(cb: cb) },
                                         draftTokensBuf: raw.tokenOutBuffer)
        else { return nil }
        if Tell.envFlag("QWISP_DFLASH_TRACE") {
            let wallMs = Double(DispatchTime.now().uptimeNanoseconds - t0.uptimeNanoseconds) / 1e6
            FileHandle.standardError.write(Data(String(
                format: "[dflash-fused-time] gpu=%.2fms wall=%.2fms ctx+=%d\n",
                SeedlessFusedVerify.SeedlessFusedForward.profLastGPUMs, wallMs, pendingRows).utf8))
        }
        let drafts = raw.finish()
        ctxRowsFed += pendingRows
        pendingCtx = []
        pendingRows = 0
        return (drafts, evals)
    }

    /// Fresh caches, clears pending/ctxRowsFed (new request/segment).
    public func reset() {
        caches = drafter.makeCaches()
        raw?.reset()
        pendingCtx = []
        pendingRows = 0
        ctxRowsFed = 0
    }
}
