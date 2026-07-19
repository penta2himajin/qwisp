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

    public init(drafter: DFlashDrafter, blockSize: Int,
                embedFn: @escaping ([Int32]) -> MLXArray?,
                logitsArgmaxFn: @escaping (MLXArray) -> [Int]?) {
        self.drafter = drafter
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
        let tokens: [Int32] = [Int32(u)] + Array(repeating: maskId, count: blockSize - 1)
        guard let embedded = embedFn(tokens) else { return nil }
        let noise = embedded.reshaped([1, blockSize, -1])
        let ctx = MLXArray(pendingCtx, [1, pendingRows, ctxDim])
        let hidden = drafter.forward(noise: noise, ctx: ctx, caches: caches)
        ctxRowsFed += pendingRows
        pendingCtx = []
        pendingRows = 0
        guard let ids = logitsArgmaxFn(hidden[0, 1...]), ids.count == blockSize - 1 else { return nil }
        return ids
    }

    /// Fresh caches, clears pending/ctxRowsFed (new request/segment).
    public func reset() {
        caches = drafter.makeCaches()
        pendingCtx = []
        pendingRows = 0
        ctxRowsFed = 0
    }
}
