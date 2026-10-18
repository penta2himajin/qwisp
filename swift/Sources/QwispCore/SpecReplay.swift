import Foundation
import Metal

// Partial-reject rollback by replay (QWISP_SPEC_REPLAY=1).
//
// After an M-row verify accepts only the first `rows` rows, the old path rolls the cache back one
// step (KV length + GDN ping-pong swap) and re-runs a full forward over those rows to rebuild the
// GDN recurrent state and conv history. Only those two depend on the rejected rows: attention KV
// for rows < `rows` was written by the verify itself, and the next token is already known.
//
// Replay instead records, during the verify, each GDN layer's recurrence inputs (q, k, v, g, beta)
// and pre-conv qkv rows ("tape"), then re-applies the first `rows` of them to the pre-verify state
// with the same gated-delta kernel (it walks rows sequentially, so the state after row r does not
// depend on later rows) and shifts the conv history with shift_conv_rows (pure copies, the same
// bytes conv_shift_fused_rows writes). The result is byte-identical to rollback + rebuild at a
// cost of one recurrence step per GDN layer per row instead of a whole forward.
//
// Validity: the tape belongs to the most recent forward. Any forward that is not a tapped verify
// (M == 1, chains, prefill chunks above the tape capacity) clears it, and replay refuses.

extension SeedlessFusedVerify {
    final class GdnTape {
        let qn: MTLBuffer, kn: MTLBuffer, v: MTLBuffer, g: MTLBuffer, beta: MTLBuffer, qkv: MTLBuffer
        init?(_ device: MTLDevice, rows: Int, keyDim: Int, valueDim: Int, Hv: Int, convDim: Int) {
            func b(_ n: Int) -> MTLBuffer? { device.makeBuffer(length: n, options: .storageModeShared) }
            guard let qn = b(rows * keyDim * 2), let kn = b(rows * keyDim * 2), let v = b(rows * valueDim * 2),
                  let g = b(rows * Hv * 4), let beta = b(rows * Hv * 4), let qkv = b(rows * convDim * 2)
            else { return nil }
            self.qn = qn; self.kn = kn; self.v = v; self.g = g; self.beta = beta; self.qkv = qkv
        }
    }
}

extension SeedlessFusedVerify.SeedlessFusedForward {
    /// Opt-in: tape GDN inputs on verify forwards and replay accepted rows on partial reject.
    static let specReplay = ProcessInfo.processInfo.environment["QWISP_SPEC_REPLAY"] == "1"
    /// Largest verify width that is taped (wider forwards fall back to rollback + rebuild).
    static let specReplayCap = Swift.max(2, Tell.envInt("QWISP_SPEC_REPLAY_CAP", 16))
    static let replayTrace = ProcessInfo.processInfo.environment["QWISP_SPEC_REPLAY_TRACE"] == "1"

    var gdnDims: (keyDim: Int, valueDim: Int, convDim: Int) {
        let k = headKDim * numKHeads, v = headVDim * numVHeads
        return (k, v, 2 * k + v)
    }

    /// Called from encodePreMoE right after a GDN layer's encode (before its ping-pong swap).
    func tapeGdnLayer(_ enc: MTLComputeCommandEncoder, cache: SeedlessFusedVerify.GdnCacheBufs, M: Int) {
        guard Self.specReplay, M > 1, M <= Self.specReplayCap, SeedlessMetalForward.compileVecCopy()
        else { tapeRows = 0; return }
        let key = ObjectIdentifier(cache)
        let d = gdnDims
        if gdnTapes[key] == nil {
            gdnTapes[key] = SeedlessFusedVerify.GdnTape(device, rows: Self.specReplayCap, keyDim: d.keyDim,
                                                     valueDim: d.valueDim, Hv: numVHeads, convDim: d.convDim)
        }
        guard let t = gdnTapes[key] else { tapeRows = 0; return }
        SeedlessMetalForward.encodeVecCopy(enc, dst: t.qn, src: gdnSc.qn, n: M * d.keyDim)
        SeedlessMetalForward.encodeVecCopy(enc, dst: t.kn, src: gdnSc.kn, n: M * d.keyDim)
        SeedlessMetalForward.encodeVecCopy(enc, dst: t.v, src: gdnSc.v1, n: M * d.valueDim)
        SeedlessMetalForward.encodeVecCopy(enc, dst: t.g, src: gdnSc.g, n: M * numVHeads * 2)        // f32 as 2 halves
        SeedlessMetalForward.encodeVecCopy(enc, dst: t.beta, src: gdnSc.beta, n: M * numVHeads * 2)
        SeedlessMetalForward.encodeVecCopy(enc, dst: t.qkv, src: gdnSc.qkv, n: M * d.convDim)
        tapeRows = M
    }

    /// Commit only the first `rows` rows of the last (taped) verify forward, starting from the
    /// cache state captured by `s`. Returns false when no valid tape covers `rows` — the caller
    /// then falls back to rollbackOneStep + rebuild forward.
    public func replayAccepted(_ s: Snapshot, rows: Int) -> Bool {
        if Self.replayTrace {
            FileHandle.standardError.write(Data("[spec-replay] rows=\(rows) tapeRows=\(tapeRows)\n".utf8))
        }
        guard Self.specReplay, rows >= 1, rows <= tapeRows, SeedlessFusedVerify.ensureRowsAuxPipelines() else { return false }
        cancelPendingChain()
        let d = gdnDims
        for (i, L) in layers.enumerated() {
            if let gc = L.gdnCache, gdnTapes[ObjectIdentifier(gc)] == nil { return false }
            // The last forward must be exactly the taped verify on top of `s`.
            if let kv = L.kvCache, kv.len != s.kvLens[i] + tapeRows { return false }
        }
        guard let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else { return false }
        for (i, L) in layers.enumerated() {
            if let kv = L.kvCache { kv.len = s.kvLens[i] + rows }
            if let gc = L.gdnCache, let t = gdnTapes[ObjectIdentifier(gc)] {
                gc.swapState()   // back to the pre-verify state / history
                SeedlessFusedVerify.encodeGatedDeltaStepRows(enc, q: t.qn, k: t.kn, v: t.v, g: t.g, beta: t.beta,
                                                         stateIn: gc.state, stateOut: gc.stateOut, y: gdnSc.coreOut,
                                                         T: rows, B: 1, Hv: numVHeads, Dv: headVDim)
                SeedlessFusedVerify.encodeShiftConvRows(enc, histOut: gc.convHistOut, histIn: gc.convHist, qkv: t.qkv,
                                                    K: convKernel, C: d.convDim, M: rows)
                gc.swapState()
            }
        }
        enc.endEncoding()
        cb.commit()   // queue order serializes it before the next forward; nothing reads it on the CPU
        lastReplayCB = cb
        tapeRows = 0
        return true
    }

    /// Wait for the last replay CB (only needed before reading cache buffers on the CPU).
    public func syncReplay() { lastReplayCB?.waitUntilCompleted(); lastReplayCB = nil }

    /// Allocate the tapes and compile the copy kernel up front (first verify would pay it otherwise).
    public func prepareSpecReplay() {
        guard Self.specReplay, SeedlessMetalForward.compileVecCopy() else { return }
        _ = SeedlessFusedVerify.ensureRowsAuxPipelines()
        let d = gdnDims
        for L in layers {
            guard let gc = L.gdnCache, gdnTapes[ObjectIdentifier(gc)] == nil else { continue }
            gdnTapes[ObjectIdentifier(gc)] = SeedlessFusedVerify.GdnTape(device, rows: Self.specReplayCap, keyDim: d.keyDim,
                                                                      valueDim: d.valueDim, Hv: numVHeads, convDim: d.convDim)
        }
    }
}
