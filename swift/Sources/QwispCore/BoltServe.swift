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
// Scope: greedy only (the server falls back to strict streaming for sampling
// requests). Rolling recalib (notes/13, R=128) refreshes residency; the async
// staged-swap pipeline (notes/14) is default ON (QWISP_BOLT_REFRESH_ASYNC=0 →
// sync). Async is free-run-safe only since the FALLBACK-SLOT fix: free-run A/B
// on nl text (600 tok, C=64, deterministic) initially showed async decode
// locking into a repetition attractor while sync stayed coherent. The evidence
// chain (swap bytes memcmp-MATCH on all 9 planes × all jobs; S=1 immediate
// swaps; atomic single-turn application; convergence freeze ⇒ endpoint ≡ sync)
// eliminated every timing/data suspect — the one semantic difference left was
// buildBuddyTable's slot-0 fallback: cold experts with no in-window
// co-activation (~100+/256 on a sparse 128-token window) remap to WHATEVER
// EXPERT OCCUPIES SLOT 0, and ensure() vs staged-swap victim assignment park
// different experts there. That fallback-target lottery compounded per refresh
// into the attractor. Fixed by pinning the fallback to the basis window's
// top-count resident (buildBuddyTable(fallbackSlot:), additive, default 0 =
// bench byte-identical). The notes/14 validation measured teacher-forced
// fidelity on a re-canonicalized state, which structurally hides both effects.
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
    private let refreshB = Tell.envInt("QWISP_BOLT_REFRESH_B", 32)
    // Default ON since the fallback-slot fix (see header); QWISP_BOLT_REFRESH_ASYNC=0 opts out to sync.
    private var refreshAsync: Bool { recalibR > 0 && Tell.envInt("QWISP_BOLT_REFRESH_ASYNC", 1) != 0 && stagingArenas.count == 2 }
    // Async refresh staging: 2 ping-pong arenas (N=refreshB each), background pread pipeline
    // (bounded buffer via semaphores). Created once, after the first calib (needs providers).
    private var stagingArenas: [ExpertArena] = []
    private let bgQueue = DispatchQueue(label: "qwisp.boltserve.async_refresh", qos: .userInitiated)
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
    /// Slot of the basis window's top-count expert — the deterministic remap target for
    /// coactivation-less cold experts. The historical slot-0 fallback made that target the
    /// slot-0 occupant LOTTERY: ensure() and staged swaps assign slots differently, so sync
    /// and async runs silently remapped the fallback mass (~100+/256 experts on a sparse
    /// 128-token window) to DIFFERENT experts — measured as compounding free-run divergence.
    private func fallbackSlot(_ li: Int) -> Int {
        guard let providers else { return 0 }
        let top1 = baseCounts[li].enumerated()
            .sorted { $0.element != $1.element ? $0.element > $1.element : $0.offset < $1.offset }
            .first?.offset
        guard let t = top1, let s = providers[li].cache.slotOf[t] else { return 0 }
        return s
    }

    private func freeze(_ fwd: SeedlessFusedVerify.SeedlessFusedForward) {
        guard let providers else { return }
        for (li, provider) in providers.enumerated() {
            let top = baseCounts[li].enumerated()
                .sorted { $0.element != $1.element ? $0.element > $1.element : $0.offset < $1.offset }
                .prefix(C).map { $0.offset }
            _ = provider.cache.ensure(Array(top))
            provider.cache.buildBuddyTable(coact: baseCoact[li], numExperts: Self.nE, fallbackSlot: fallbackSlot(li))
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
            // stderr like the server's perf log — keeps `qwisp benchtest > report.md` clean.
            FileHandle.standardError.write(Data("[qwisp] bolt tier active (near-lossless, streaming default): C=\(C) — pass --lossless for bit-exact strict\n".utf8))
            FileHandle.standardError.write(Data("[qwisp] calibrating expert routing (one-time per process, runs at strict speed — can take a few minutes on this tier) …\n".utf8))
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
            // Minimum calib corpus: a trivial first request (e.g. "hi", ~18 rows + 48 steps)
            // freezes residency off nearly no routing evidence, and the next REAL request
            // then decodes into the greedy repetition attractor (measured via benchtest:
            // "Say hello." warmup → every later prompt LOOPY; representative warmup → ok).
            // Extend the calib greedy run so prefill+steps cover ≥ calibMinRows observation
            // rows — real prompts (≥~100 tokens) pay nothing, tiny ones pay a few seconds
            // of strict-speed decode ONCE per process.
            let calibMinRows = Tell.envInt("QWISP_CALIB_MIN_ROWS", 128)
            let steps = Swift.max(calibN, calibMinRows - promptIds.count)
            for _ in 0 ..< steps {
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
            // Staging arenas for the async refresh pipeline (best-effort: on failure the
            // refreshAsync computed property stays false → sync refresh path).
            if Tell.envInt("QWISP_BOLT_REFRESH_ASYNC", 1) != 0, let first = provs.first {
                for _ in 0 ..< 2 {
                    if let a = try? ExpertArena(device: first.cache.arena.device,
                                                source: first.cache.arena.source,
                                                N: refreshB, refLayer: 0) { stagingArenas.append(a) }
                }
                if stagingArenas.count < 2 { stagingArenas = [] }
            }
            calibrated = true
            FileHandle.standardError.write(Data("[qwisp] calibration done — decoding at bolt speed from here\n".utf8))
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
        var gateBuf: MTLBuffer? = nil   // per-expert gate values (gl); cold-gate-mass = substitution-error proxy (#47)
        if recalibR > 0 {
            guard SeedlessMetalForward.compileDiagCopyRoute(),
                  let (device, _) = SeedlessMetalForward.ensure(),
                  let ib = device.makeBuffer(length: obsSlots * nLayers * obsMaxM * Ktop * 4,
                                             options: .storageModeShared),
                  let gb = device.makeBuffer(length: nLayers * nE * 2, options: .storageModeShared)
            else { return nil }
            obsBuf = ib
            gateBuf = gb
            fwd.diagObsMaxM = obsMaxM
            fwd.diagRouteBufs = (ib, gb)
        }
        // Routing-divergence trace (#47 Part A, QWISP_MISS_TRACE=path): per accumulate call,
        // how many of the EXACT top-8 routed experts are cold (not resident → buddy-remapped)
        // vs total routed. bolt's routing is exact; the miss set IS the residency divergence
        // from the "true" (all-resident/strict) expert set. Tests whether the loop onset is
        // preceded by a small, swappable divergence or a broad one (capacity wall).
        let missTracePath = ProcessInfo.processInfo.environment["QWISP_MISS_TRACE"]
        var missTrace: [(tok: Int, miss: Int, routed: Int, M: Int, coldGate: Float, totGate: Float, margin: Float, coldW: Float, entropy: Float, top8: String)] = []
        // Cold-expert histogram in the pre-cliff ramp window (#47 Part A, QWISP_MISS_HIST=path
        // + QWISP_CLIFF_TOK=N): is the ramp cold set a small recurring set (pinnable) or diffuse
        // (capacity wall)? Counts cold routings per (layer, expert) for tok ∈ [cliff-70, cliff+10].
        let missHistPath = ProcessInfo.processInfo.environment["QWISP_MISS_HIST"]
        let cliffTok = Tell.envInt("QWISP_CLIFF_TOK", -1)
        var coldHist = [[Int: Int]](repeating: [:], count: nLayers)   // [layer][expert] = cold count in ramp
        // Hazard-burst forced refresh (#47 Part A probe 13, opt-in QWISP_HAZARD_REFRESH=1):
        // when the margin×miss both-bad burst (probe 11/12 conjunction) fires, force a sync
        // recalib refresh — the deliberate form of the refresh rescue that saved clean QS in
        // the probe-12 identical-prefix counterexample (loop/clean differed ONLY in refresh
        // timing). FPs are harmless: a refresh on a clean burst is what default cadence does
        // anyway. Needs the QWISP_MARGIN_TRACE path (Tell.lastMargin) and sync refresh
        // (run with QWISP_BOLT_REFRESH_ASYNC=0) — experiment-grade, default off.
        let hazardOn = Tell.envInt("QWISP_HAZARD_REFRESH", 0) != 0
        let hazardCooldown = Tell.envInt("QWISP_HAZARD_COOLDOWN", 32)
        var hazardWin: [Bool] = []
        var hazardCool = 0
        var hazardFire = false
        var hazardFires: [Int] = []
        func accumulate(slot: Int, M: Int, tok: Int = 0) {
            guard recalibR > 0, let ib = obsBuf else { return }
            var traceMiss = 0, traceRouted = 0
            var coldGate: Float = 0, totGate: Float = 0, coldW: Float = 0
            // gl gate values are captured only for M==1/slot0 (Stage-0 diag glOn condition).
            let gatePtr = (missTracePath != nil && M == 1 && slot == 0)
                ? gateBuf?.contents().assumingMemoryBound(to: Float16.self) : nil
            let inRamp = missHistPath != nil && cliffTok > 0 && tok >= cliffTok - 70 && tok <= cliffTok + 10
            for li in 0 ..< nLayers {
                let off = ((slot * nLayers + li) * obsMaxM) * Ktop * 4
                let inds = Array(UnsafeBufferPointer(
                    start: ib.contents().advanced(by: off).assumingMemoryBound(to: Int32.self),
                    count: M * Ktop))
                if missTracePath != nil || hazardOn {
                    let s = providers[li].cache.slotOf
                    var lg: [(cold: Bool, g: Float)] = []   // this layer's routed raw gate logits
                    for e in inds where e >= 0 {
                        traceRouted += 1
                        let g = gatePtr.map { Float($0[li * nE + Int(e)]) } ?? 0
                        totGate += g
                        let cold = s[Int(e)] == nil
                        if cold { traceMiss += 1; coldGate += g }   // substitution-error proxy
                        if gatePtr != nil { lg.append((cold, g)) }
                    }
                    // Softmax-normalized cold gate share, summed over layers (#47 hazard ratio
                    // ρ = coldW/margin, qwisp-lean LoopTrigger.lean): per-layer softmax over the
                    // routed raw logits — comparable across steps, unlike the raw coldGate sums.
                    if let mx = lg.map(\.g).max() {
                        let exps = lg.map { expf($0.g - mx) }
                        let z = exps.reduce(0, +)
                        if z > 0 {
                            for (i, p) in lg.enumerated() where p.cold { coldW += exps[i] / z }
                        }
                    }
                }
                if inRamp {
                    let s = providers[li].cache.slotOf
                    for e in inds where e >= 0 && s[Int(e)] == nil { coldHist[li][Int(e), default: 0] += 1 }
                }
                _ = SeedlessFusedVerify.SeedlessFusedForward.recalibAccumulate(
                    inds: inds, M: M, Ktop: Ktop, nE: nE,
                    counts: &winCounts[li], coact: &winCoact[li])
            }
            if missTracePath != nil { missTrace.append((tok, traceMiss, traceRouted, M, coldGate, totGate, Tell.lastMargin, coldW, Tell.lastEntropy, Tell.lastTop8)) }
            // Hazard window update + burst check (M=1 decode rows only; margin<3 ∧ miss>0.20,
            // burst ≥4/10 = probe 11 canon). Cooldown lets the rebased basis take effect.
            if hazardOn && M == 1 {
                let mr = traceRouted > 0 ? Float(traceMiss) / Float(traceRouted) : 0
                let bad = Tell.lastMargin < 3 && mr > 0.20
                if hazardCool > 0 { hazardCool -= 1 }
                hazardWin.append(bad)
                if hazardWin.count > 10 { hazardWin.removeFirst() }
                if hazardCool == 0, hazardWin.count == 10,
                   hazardWin.lazy.filter({ $0 }).count >= 4 {
                    hazardFire = true
                    hazardCool = hazardCooldown
                    hazardWin.removeAll()
                    hazardFires.append(tok)
                }
            }
        }
        /// Recalib refresh (sync): the observation window becomes the new freeze basis.
        func refresh() {
            swap(&baseCounts, &winCounts)
            swap(&baseCoact, &winCoact)
            if Tell.envFlag("QWISP_BOLT_DEBUG") {
                let obs = baseCounts.reduce(0) { $0 + $1.reduce(0, +) }
                let top8 = baseCounts[0].enumerated().sorted { $0.element > $1.element }.prefix(8).map { "\($0.offset):\($0.element)" }
                FileHandle.standardError.write(Data("[boltserve] sync refresh: windowObs=\(obs) L0top8=\(top8.joined(separator: ","))\n".utf8))
            }
            freeze(fwd)
            for li in 0 ..< nLayers {
                for e in 0 ..< nE { winCounts[li][e] = 0; winCoact[li][e] = [Int](repeating: 0, count: nE) }
            }
            tokensSinceRefresh = 0
        }

        // Decode-loop state (declared before the async machinery below captures it).
        var hist = promptIds.map { Int($0) }
        var out: [Int] = []
        var stSteps = 0, stDrafted = 0, stAccepted = 0, stD0 = 0
        var streamed = 0
        func flush() { if let onToken { while streamed < out.count { onToken(out[streamed]); streamed += 1 } } }

        // ── Async refresh plan state (notes/14; mirror of runBoltMode's machinery) ──
        struct CrossJob { let li: Int; let expert: Int; let victimSlot: Int }
        var asyncBoundary = 0
        var asyncChunks = [[CrossJob]]()
        var asyncNextSwap = 0
        var asyncStride = 1
        var semFree = DispatchSemaphore(value: 2)
        var semReady = DispatchSemaphore(value: 0)
        /// Background pipeline: pread every chunk into the 2 staging arenas (bounded buffer).
        /// Writes ONLY to stagingArenas — providers are touched by swapChunk on this thread.
        func startBgPlan() {
            guard refreshAsync, !asyncChunks.isEmpty else { return }
            let chunks = asyncChunks, stages = stagingArenas
            let sf = semFree, sr = semReady
            bgQueue.async {
                for (j, chunk) in chunks.enumerated() {
                    sf.wait()
                    let stage = stages[j % 2]
                    var byLayer = [Int: [(e: Int, slot: Int)]]()
                    for (k, cj) in chunk.enumerated() {
                        byLayer[cj.li, default: []].append((e: cj.expert, slot: k))
                    }
                    for (layer, jobs) in byLayer { stage.loadMany(layer, jobs) }
                    sr.signal()
                }
            }
        }
        /// Atomic CPU-turn swap of chunk j: staging → arena victim slots + bookkeeping +
        /// per-touched-layer buddy/table/bias rebuild (baseCoact = the plan's window snapshot).
        func swapChunk(_ j: Int) {
            guard refreshAsync, j < asyncChunks.count else { return }
            semReady.wait()
            let chunk = asyncChunks[j]
            let stage = stagingArenas[j % 2]
            for (k, cj) in chunk.enumerated() {
                let cache = providers[cj.li].cache
                let arena = cache.arena
                for key in stage.slots.keys {
                    guard let srcS = stage.slots[key], let dstS = arena.slots[key] else { continue }
                    memcpy(dstS.ptr + cj.victimSlot * dstS.sliceBytes,
                           srcS.ptr + k             * srcS.sliceBytes,
                           srcS.sliceBytes)
                }
                let cur = cache.expertAt[cj.victimSlot]
                if cur >= 0 && cur != cj.expert { cache.slotOf.removeValue(forKey: cur) }
                cache.slotOf[cj.expert]       = cj.victimSlot
                cache.expertAt[cj.victimSlot] = cj.expert
                cache.clock                  += 1
                cache.tick[cj.victimSlot]     = cache.clock
                // ensure() parity: slotOf changed → the STRICT-path derived GPU arrays
                // (slotTableGPU / hotMask) must rebuild, or the next segment's strict
                // prefill remaps routed experts to pre-swap slots = silent garbage.
                // (runBoltMode never runs strict again after its swaps, so the bench
                // template omits this; the server does — every segment prefills strict.)
                cache.slotTableDirty = true
                cache.slotVersion   += 1
            }
            LayerExpertCache.missTotal += chunk.count
            if Tell.envFlag("QWISP_BOLT_DEBUG") {
                // Full bytecheck: every job × every plane, arena victim vs fresh pread.
                var bad: [String: Int] = [:], checked = 0
                for cj in chunk {
                    let arena = providers[cj.li].cache.arena
                    for proj in ExpertSource.projs {
                        for part in ExpertSource.parts {
                            guard let s = arena.slots["\(proj).\(part)"] else { continue }
                            let tmp = UnsafeMutableRawPointer.allocate(byteCount: s.sliceBytes, alignment: 16)
                            defer { tmp.deallocate() }
                            try? arena.source.preadInto(tmp, cj.li, proj, part, cj.expert)
                            checked += 1
                            if memcmp(tmp, s.ptr + cj.victimSlot * s.sliceBytes, s.sliceBytes) != 0 {
                                bad["\(proj).\(part)", default: 0] += 1
                            }
                        }
                    }
                }
                FileHandle.standardError.write(Data("[boltserve] swap chunk \(j): \(chunk.count) jobs, planes checked=\(checked) mismatch=\(bad.isEmpty ? "NONE" : String(describing: bad))\n".utf8))
            }
            for li in Set(chunk.map { $0.li }).sorted() {
                let cache = providers[li].cache
                cache.buildBuddyTable(coact: baseCoact[li], numExperts: nE, fallbackSlot: fallbackSlot(li))
                fwd.setBoltTable(li, cache.buddyTableCPU)
                if biasEps > 0 {
                    fwd.updateRouteBiasMask(li, (0 ..< nE).map { Int32(cache.buddyExpertCPU[$0] == $0 ? 1 : 0) })
                }
            }
            // Convergence freeze: after the last chunk, rebuild ALL layers against the
            // plan window — endpoint identical to a sync refresh (cheap; ensure all-hit).
            if j == asyncChunks.count - 1 { freeze(fwd) }
            semFree.signal()
        }
        /// Drain all pending chunks (blocking). MUST run before replacing the plan/semaphores
        /// and before every return — an undrained semaphore pair traps in dispose (SIGTRAP),
        /// and undrained victim bookkeeping would go stale across the next segment's prefill.
        func drainAsync() {
            while asyncNextSwap < asyncChunks.count { swapChunk(asyncNextSwap); asyncNextSwap += 1 }
        }
        defer { drainAsync() }
        /// Recalib refresh (async): window → freeze basis, plan diffs, kick the bg pipeline.
        /// Swaps land at fixed token positions (asyncStride) at the decode loop head.
        func refreshAsyncPlan() {
            drainAsync()                       // leftover chunks from the previous plan
            swap(&baseCounts, &winCounts)
            swap(&baseCoact, &winCoact)
            var perLayer = [[CrossJob]]()
            for (li, provider) in providers.enumerated() {
                var jobs = [CrossJob]()
                if let plan = BoltAsyncRefresh.makePlan(
                    counts: baseCounts[li], coact: baseCoact[li],
                    slotOf: provider.cache.slotOf, expertAt: provider.cache.expertAt,
                    tick: provider.cache.tick, pinnedSlots: provider.cache.pinnedSlots,
                    C: C, nE: nE, B: refreshB) {
                    for job in plan.jobs {
                        jobs.append(CrossJob(li: li, expert: job.expert, victimSlot: job.victimSlot))
                    }
                }
                perLayer.append(jobs)
            }
            // Round-robin interleave across layers: each chunk advances EVERY layer by
            // ~1 expert, so intermediate states are uniform-vintage snapshots (a series
            // of small sync-like refreshes). Layer-contiguous chunking staggered layer
            // vintages across the transition window — the measured free-run attractor
            // seed (see header). Uniformity is what makes async free-run-safe.
            var allJobs = [CrossJob]()
            let maxLen = perLayer.map(\.count).max() ?? 0
            for r in 0 ..< maxLen {
                for lj in perLayer where r < lj.count { allJobs.append(lj[r]) }
            }
            asyncBoundary = out.count
            asyncChunks = stride(from: 0, to: allJobs.count, by: refreshB).map {
                Array(allJobs[$0 ..< Swift.min($0 + refreshB, allJobs.count)])
            }
            if Tell.envFlag("QWISP_BOLT_DEBUG") {
                let obs = baseCounts.reduce(0) { $0 + $1.reduce(0, +) }
                let top8 = baseCounts[0].enumerated().sorted { $0.element > $1.element }.prefix(8).map { "\($0.offset):\($0.element)" }
                FileHandle.standardError.write(Data("[boltserve] async plan: jobs=\(allJobs.count) chunks=\(asyncChunks.count) boundary=\(asyncBoundary) windowObs=\(obs) L0top8=\(top8.joined(separator: ","))\n".utf8))
            }
            asyncNextSwap = 0
            // Spread swaps over the half-window (OAT 2026-07-07: full-width lags table
            // freshness, tighter blocks on GPU-idle swaps). QWISP_BOLT_REFRESH_S overrides
            // (bench parity knob).
            let sOverride = Tell.envInt("QWISP_BOLT_REFRESH_S", 0)
            asyncStride = sOverride > 0 ? sOverride
                        : Swift.max(1, (recalibR / 2) / Swift.max(1, asyncChunks.count))
            for li in 0 ..< nLayers {
                for e in 0 ..< nE { winCounts[li][e] = 0; winCoact[li][e] = [Int](repeating: 0, count: nE) }
            }
            tokensSinceRefresh = 0
            semFree = DispatchSemaphore(value: 2)
            semReady = DispatchSemaphore(value: 0)
            startBgPlan()
            // Atomic-apply diag (QWISP_BOLT_ATOMIC=1): drain every chunk right here —
            // staged IO, but application collapses to ONE CPU turn like a sync refresh.
            // Discriminates "spread-out application is the poison" from "swap mechanics".
            if Tell.envFlag("QWISP_BOLT_ATOMIC") { drainAsync() }
        }

        // ── Bolt spec decode loop (mirror of runBoltMode phase 6, server contract) ──
        while out.count < N && !(isCancelled?() ?? false) {
            flush()
            // Hazard-burst forced refresh (probe 13): sync-only — run with
            // QWISP_BOLT_REFRESH_ASYNC=0 so it cannot race the async staging pipeline.
            if hazardFire {
                hazardFire = false
                if !refreshAsync {
                    FileHandle.standardError.write(Data("[qwisp] hazard-refresh @ tok \(out.count)\n".utf8))
                    refresh()
                }
            }
            if recalibR > 0 && tokensSinceRefresh >= recalibR {
                if refreshAsync { refreshAsyncPlan() } else { refresh() }
            }
            // Async: swap due chunks at fixed token positions (chain/verify spans jump
            // out.count, so consume ALL due chunks here — bg pipeline has them preread).
            while refreshAsync, asyncNextSwap < asyncChunks.count,
                  out.count >= asyncBoundary + (asyncNextSwap + 1) * asyncStride {
                swapChunk(asyncNextSwap); asyncNextSwap += 1
            }
            let before = out.count
            let drafts = Tell.suffixDraft(hist + [u], maxMatch: 32, draftK: maxK, minMatch: Tell.suffixMinMatch)
            let D = drafts.count
            stSteps += 1; stDrafted += D; if D == 0 { stD0 += 1 }
            let snap = backend.snapshot()

            if D == 0 {
                if let (emitted, nextU) = Tell.boltGreedyChainSpan(
                    backend: backend, u: u, chainK: chainK, budget: N - out.count) {
                    if recalibR > 0 { for k in 0 ..< chainK { accumulate(slot: k, M: 1, tok: out.count + k) } }
                    for t in emitted { out.append(t); hist.append(t) }
                    u = nextU
                } else {
                    guard let evals = backend.stepArgmax([Int32(u)]) else { return nil }
                    if recalibR > 0 { accumulate(slot: 0, M: 1, tok: out.count) }
                    out.append(u); hist.append(u)
                    u = evals[0]
                }
                tokensSinceRefresh += out.count - before
                continue
            }

            let verifyTokens: [Int32] = [Int32(u)] + drafts.map { Int32($0) }
            guard let evals = backend.stepArgmax(verifyTokens) else { return nil }
            if recalibR > 0 { accumulate(slot: 0, M: D + 1, tok: out.count) }

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
        // Emitted token ids (#47 probe 14): the TF-replay input (QWISP_TF_REPLAY reads this
        // to teacher-force the bolt stream through strict and locate realized flips).
        if let p = ProcessInfo.processInfo.environment["QWISP_TOK_DUMP"] {
            try? out.map(String.init).joined(separator: "\n").write(toFile: p, atomically: true, encoding: .utf8)
        }
        if hazardOn {
            FileHandle.standardError.write(Data("[qwisp] hazard-refresh fires=\(hazardFires.count) at toks \(hazardFires)\n".utf8))
        }
        if let p = missTracePath, !missTrace.isEmpty {
            let body = missTrace.map { "\($0.tok)\t\($0.miss)\t\($0.routed)\t\($0.M)\t\($0.coldGate)\t\($0.totGate)\t\($0.margin)\t\($0.coldW)\t\($0.entropy)\t\($0.top8)" }.joined(separator: "\n")
            try? ("tok\tmiss\trouted\tM\tcoldGate\ttotGate\tmargin\tcoldW\tentropy\ttop8\n" + body).write(toFile: p, atomically: true, encoding: .utf8)
        }
        if let p = ProcessInfo.processInfo.environment["QWISP_MARGIN_TRACE"], !Tell.marginTrace.isEmpty {
            try? ("margin\n" + Tell.marginTrace.map { String($0) }.joined(separator: "\n")).write(toFile: p, atomically: true, encoding: .utf8)
        }
        // Ramp cold-expert histogram: layer, expert, cold-count, and the expert's frequency RANK
        // in the frozen basis (baseCounts) — so we can tell if the ramp cold set is the same
        // "next-after-C" tail that C=80 already added (rank ~65-80) or a distinct deep-tail set.
        if let p = missHistPath, coldHist.contains(where: { !$0.isEmpty }) {
            var lines = ["layer\texpert\tcold_count\tfreq_rank"]
            for li in 0 ..< nLayers {
                let ranked = baseCounts[li].enumerated()
                    .sorted { $0.element != $1.element ? $0.element > $1.element : $0.offset < $1.offset }
                    .map { $0.offset }
                var rankOf = [Int: Int](); for (r, e) in ranked.enumerated() { rankOf[e] = r }
                for (e, c) in coldHist[li].sorted(by: { $0.value > $1.value }) {
                    lines.append("\(li)\t\(e)\t\(c)\t\(rankOf[e] ?? -1)")
                }
            }
            try? lines.joined(separator: "\n").write(toFile: p, atomically: true, encoding: .utf8)
        }
        Tell.lastSpecStats = (stSteps, stDrafted, stAccepted, stD0, 0, 0)
        return Array(out.prefix(N))
    }
}
