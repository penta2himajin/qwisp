import Foundation
import Metal
import MLX

// Server-side bolt runtime (near-lossless L3, streaming tiers only).
//
// Productization decision (2026-07-09): <32GB (streaming, C<256) defaults to bolt;
// ≥32GB (resident) stays strict; --lossless forces strict on every tier.
//
// This is the SERVER counterpart of the bench-only Tell.runBoltMode (phases 1-6):
// the same primitives (streamingBackend / indsCaptureHook calib / buildBuddyTable /
// setBoltTables / setRouteBias / recalibAccumulate) recomposed for a persistent,
// multi-request process. runBoltMode itself is untouched — it stays the measurement
// harness; byte-level parity of the bench path is preserved by construction.
//
// ORDER INVARIANT (notes/13 "TF 開始状態の正準化"): buddy tables map experts to arena
// SLOTS, so any ensure() after freezing (e.g. a strict prefill pulling prompt experts)
// silently invalidates the tables → garbage activations. Therefore every segment runs
// its strict prefill FIRST and freezes (ensure top-C + rebuild tables + setBoltTables)
// AFTER, mirroring runBoltMode's phase 4 → phase 5 order.
//
// v1 scope (ponytail): greedy only (the server falls back to strict streaming for
// sampling requests), sync rolling recalib (notes/13, R=128); async refresh
// (notes/14 staging pipeline, slow-NAND +41-48%) is a follow-up.
//
// Lifetime: one instance per SeedlessBackend. The expert arena (providers) and the
// freeze basis (last calib/recalib routing window) persist across requests; each
// request builds a fresh forward sized to the request (KV/GDN state is per-request),
// prefills, then freezes. Calibration (strict prefill + calibN greedy steps with
// routing capture) runs once, on the first request; rolling recalib adapts from there.
// The server serializes generation (AsyncLock), so no internal locking.
final class BoltServe {
    private let engine: SeedlessEngine
    private let modelDir: String
    private let C: Int
    private let maxK: Int
    private let maxM: Int

    private var providers: [ArenaExpertProvider]? = nil
    private var calibrated = false
    // Freeze basis: the routing window (counts/coact) the current residency was built
    // from — calib initially, then the latest recalib window.
    private var baseCounts: [[Int]] = []
    private var baseCoact: [[[Int]]] = []
    // Rolling recalib window (persists across requests — R boundaries span requests).
    private var winCounts: [[Int]] = []
    private var winCoact: [[[Int]]] = []
    private var tokensSinceRefresh = 0

    private let calibN = Tell.envInt("QWISP_CALIB", 48)
    private let biasEps = Tell.envFloat("QWISP_ROUTE_BIAS_EPS", 0.25)
    private let recalibR = Tell.envInt("QWISP_BOLT_RECALIB_R", 128)
    private let chainK = Tell.envInt("QWISP_CHAIN_K", SeedlessFusedVerify.SeedlessFusedForward.chainKDefault)
    private static let nE = 256
    private static let Ktop = 8

    init(engine: SeedlessEngine, modelDir: String, C: Int, maxK: Int, maxM: Int) {
        self.engine = engine
        self.modelDir = modelDir
        self.C = C
        self.maxK = maxK
        self.maxM = maxM
    }

    /// Freeze current residency to `fwd`: ensure top-C of the basis window, rebuild
    /// buddy tables against the CURRENT slot state, upload tables + residency bias.
    /// Must run AFTER any ensure-moving operation (prefill, refresh) — see header.
    private func freeze(_ fwd: SeedlessFusedVerify.SeedlessFusedForward) {
        guard let providers else { return }
        for (li, provider) in providers.enumerated() {
            let top = baseCounts[li].enumerated()
                .sorted { $0.element != $1.element ? $0.element > $1.element : $0.offset < $1.offset }
                .prefix(C).map { $0.offset }
            _ = provider.cache.ensure(Array(top))
            provider.cache.buildBuddyTable(coact: baseCoact[li], numExperts: Self.nE)
        }
        fwd.setBoltTables(providers.map { $0.cache.buddyTableCPU })
        if biasEps > 0 {
            let masks: [[Int32]] = providers.map { p in
                (0 ..< Self.nE).map { Int32(p.cache.buddyExpertCPU[$0] == $0 ? 1 : 0) }
            }
            fwd.setRouteBias(masks: masks, eps: biasEps)
        }
    }

    /// One generation segment (mirror of Tell.runSpecLoop's contract: returns out[0..<N],
    /// streams via onToken, stops early on isCancelled). nil on backend error.
    func runSegment(promptIds: [Int32], N: Int, maxSeqLen: Int,
                    isCancelled: (() -> Bool)? = nil,
                    onToken: ((Int) -> Void)? = nil) -> [Int]? {
        let nLayers = SeedlessEngine.numLayers
        let nE = Self.nE, Ktop = Self.Ktop

        // ── First request: strict calib pass (routing frequency + co-activation) ──
        if !calibrated {
            print("[qwisp] bolt tier active (near-lossless, streaming default): C=\(C) — pass --lossless for bit-exact strict")
            guard let (backend1, fwd1, provs) = Tell.streamingBackend(
                engine: engine, modelDir: modelDir, maxM: maxM, maxSeqLen: maxSeqLen, C: C)
            else { return nil }
            var counts = [[Int]](repeating: [Int](repeating: 0, count: nE), count: nLayers)
            var coact = [[[Int]]](repeating: [[Int]](repeating: [Int](repeating: 0, count: nE), count: nE),
                                  count: nLayers)
            fwd1.indsCaptureHook = { li, inds in
                let distinct = Array(Set(inds.map { Int($0) }))
                for e in distinct { counts[li][e] += 1 }
                let n = distinct.count
                for ai in 0 ..< n {
                    for bi in (ai + 1) ..< n {
                        let a = distinct[ai], b = distinct[bi]
                        coact[li][a][b] += 1; coact[li][b][a] += 1
                    }
                }
            }
            guard let lastNormed = Tell.prefill(promptIds: promptIds, backend: backend1),
                  let lg0 = engine.logits(lastNormed, M: 1) else { return nil }
            MLX.eval([lg0])
            var u = MLX.argMax(lg0[0], axis: -1).item(Int.self)
            for _ in 0 ..< calibN {
                guard let evals = backend1.stepArgmax([Int32(u)]) else { return nil }
                u = evals[0]
            }
            fwd1.indsCaptureHook = nil
            providers = provs
            baseCounts = counts
            baseCoact = coact
            winCounts = [[Int]](repeating: [Int](repeating: 0, count: nE), count: nLayers)
            winCoact = [[[Int]]](repeating: [[Int]](repeating: [Int](repeating: 0, count: nE), count: nE),
                                 count: nLayers)
            calibrated = true
        }
        guard let providers else { return nil }

        // ── Fresh forward for this segment (arena persists via providers) ──
        guard let (backend, fwd, _) = Tell.streamingBackend(
            engine: engine, modelDir: modelDir, maxM: maxM, maxSeqLen: maxSeqLen, C: C,
            existingProviders: providers)
        else { return nil }
        // Exact (strict) prefill FIRST — its ensures may move arena slots — then freeze.
        guard let lastNormed = Tell.prefill(promptIds: promptIds, backend: backend),
              let lg0 = engine.logits(lastNormed, M: 1) else { return nil }
        MLX.eval([lg0])
        var u = MLX.argMax(lg0[0], axis: -1).item(Int.self)
        freeze(fwd)

        // Recalib observation buffers (side-buffer copy of route inds, notes/13 layout).
        let obsMaxM = recalibR > 0 ? maxM : 1
        let obsSlots = recalibR > 0 ? Swift.max(1, chainK) : 1
        var obsBuf: MTLBuffer? = nil
        if recalibR > 0 {
            guard SeedlessMetalForward.compileDiagCopyRoute(),
                  let (device, _) = SeedlessMetalForward.ensure(),
                  let ib = device.makeBuffer(length: obsSlots * nLayers * obsMaxM * Ktop * 4,
                                             options: .storageModeShared),
                  let gb = device.makeBuffer(length: nLayers * nE * 2, options: .storageModeShared)
            else { return nil }
            obsBuf = ib
            fwd.diagObsMaxM = obsMaxM
            fwd.diagRouteBufs = (ib, gb)
        }
        func accumulate(slot: Int, M: Int) {
            guard recalibR > 0, let ib = obsBuf else { return }
            for li in 0 ..< nLayers {
                let off = ((slot * nLayers + li) * obsMaxM) * Ktop * 4
                let inds = Array(UnsafeBufferPointer(
                    start: ib.contents().advanced(by: off).assumingMemoryBound(to: Int32.self),
                    count: M * Ktop))
                _ = SeedlessFusedVerify.SeedlessFusedForward.recalibAccumulate(
                    inds: inds, M: M, Ktop: Ktop, nE: nE,
                    counts: &winCounts[li], coact: &winCoact[li])
            }
        }
        /// Recalib refresh: the observation window becomes the new freeze basis.
        func refresh() {
            swap(&baseCounts, &winCounts)
            swap(&baseCoact, &winCoact)
            freeze(fwd)
            for li in 0 ..< nLayers {
                for e in 0 ..< nE { winCounts[li][e] = 0; winCoact[li][e] = [Int](repeating: 0, count: nE) }
            }
            tokensSinceRefresh = 0
        }

        // ── Bolt spec decode loop (mirror of runBoltMode phase 6, server contract) ──
        var hist = promptIds.map { Int($0) }
        var out: [Int] = []
        var stSteps = 0, stDrafted = 0, stAccepted = 0, stD0 = 0
        var streamed = 0
        func flush() { if let onToken { while streamed < out.count { onToken(out[streamed]); streamed += 1 } } }

        while out.count < N && !(isCancelled?() ?? false) {
            flush()
            if recalibR > 0 && tokensSinceRefresh >= recalibR { refresh() }
            let before = out.count
            let drafts = Tell.suffixDraft(hist + [u], maxMatch: 32, draftK: maxK, minMatch: Tell.suffixMinMatch)
            let D = drafts.count
            stSteps += 1; stDrafted += D; if D == 0 { stD0 += 1 }
            let snap = backend.snapshot()

            if D == 0 {
                if let (emitted, nextU) = Tell.boltGreedyChainSpan(
                    backend: backend, u: u, chainK: chainK, budget: N - out.count) {
                    if recalibR > 0 { for k in 0 ..< chainK { accumulate(slot: k, M: 1) } }
                    for t in emitted { out.append(t); hist.append(t) }
                    u = nextU
                } else {
                    guard let evals = backend.stepArgmax([Int32(u)]) else { return nil }
                    if recalibR > 0 { accumulate(slot: 0, M: 1) }
                    out.append(u); hist.append(u)
                    u = evals[0]
                }
                tokensSinceRefresh += out.count - before
                continue
            }

            let verifyTokens: [Int32] = [Int32(u)] + drafts.map { Int32($0) }
            guard let evals = backend.stepArgmax(verifyTokens) else { return nil }
            if recalibR > 0 { accumulate(slot: 0, M: D + 1) }

            var p = 0
            while p < D && drafts[p] == evals[p] { p += 1 }
            stAccepted += p

            if p == D {
                out.append(u); hist.append(u)
                for d in drafts { out.append(d); hist.append(d) }
                u = evals[D]
            } else {
                backend.rollback(snap)
                out.append(u); hist.append(u)
                for d in drafts.prefix(p) { out.append(d); hist.append(d) }
                let rebuildTokens: [Int32] = [Int32(u)] + drafts.prefix(p).map { Int32($0) }
                guard let _ = backend.forward(rebuildTokens) else { return nil }
                u = evals[p]
            }
            tokensSinceRefresh += out.count - before
        }
        flush()
        Tell.lastSpecStats = (stSteps, stDrafted, stAccepted, stD0, 0, 0)
        return Array(out.prefix(N))
    }
}
