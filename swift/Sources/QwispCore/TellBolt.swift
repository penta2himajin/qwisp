// Stage-1 bolt-mode: buddy+spec near-lossless (L3, opt-in). Reuses Tell.suffixDraft + the
// SuffixSpec accept-prefix loop, but verify (and reject re-run) run buddy no-sync
// (probeNoSync=true, skipMode=3) so the committed output = buddy-model greedy at spec speed.
// No exact escalation, no union-overflow guard (buddy is lossy-by-design). nl auto-falls-back:
// suffixDraft()==[] on high-entropy tokens => single buddy no-sync forward/token.
//
// Value only on the slow-NAND tier: buddy makes io->0, so under SSD throttle (Neo) bolt is not
// IO-starved like strict. On fast SSD / resident, strict SuffixSpec dominates — bolt is Neo-only.
// Measure with QWISP_SSD_THROTTLE_GBS=1.5 to see the win. Strict-L1 stays the default (runSuffixSpec).
import Foundation
import MLX
import Metal

extension Tell {
    /// **bolt-mode L3 (opt-in, near-lossless)**: buddy no-sync draft/verify SuffixSpec.
    /// output = buddy-greedy (deterministic); headline quality = token-match vs strict-4bit
    /// **f32-full** greedy (the canonical L1 reference), computed in-run when QWISP_SWIFT_REF=1.
    /// env: QWISP_BOLT=1 or QWISP_RUN=bolt / QWISP_CACHE_C / QWISP_DRAFT_K(debug-only override) /
    ///      QWISP_CALIB / QWISP_SWIFT_REF / QWISP_GEN / QWISP_FUSE_GDN(stage2)
    public static func runBolt(modelDir: String, refPath: String) throws -> String {
        guard let device = MTLCreateSystemDefaultDevice() else { return "ERROR: no Metal device" }
        let r = try loadArrays(url: URL(fileURLWithPath: refPath))
        guard let promptArr = r["spec_prompt"], let gRef = r["spec_greedy"] else { return "[Bolt] skip" }
        let C = Tell.envInt("QWISP_CACHE_C", DeviceCalibration.defaultC())
        let store = try WeightStore(modelDir: modelDir); store.residentNonExperts()
        let source = try ExpertSource(modelDir: modelDir); try source.warm()
        let arena = try ExpertArena(device: device, source: source, N: 64)
        let model = try StreamingQwispModel(store: store, arena: arena, device: device, source: source, cacheC: C)
        let ids = promptArr.asType(.int32).reshaped([1, promptArr.dim(0)])
        let gR = gRef.asArray(Int32.self).map { Int($0) }
        return try boltCore(model: model, ids: ids, gR: gR, C: C).summary
    }

    /// ★ T1: bolt 本体 core。prebuilt model を受け取り、runBolt と TellBench(batch bench)の両方から呼ぶ。
    /// in-process 連続実行を想定し、依存する global static は entry で全て明示 set、exit(defer)で reset。
    /// timed decode(phase 3c)直前に ExpertSource.throttleActive を立てる（T2 defer gate）。
    static func boltCore(model: StreamingQwispModel, ids: MLXArray, gR: [Int], C: Int)
        throws -> (summary: String, tokps: Double) {
        let calibN = Tell.envInt("QWISP_CALIB", 48)
        // bolt decode は ensure を呼ばない（cold は buddyTable で常駐 slot へ remap）ため strict の
        // C·3/8 は容量制約としては不要だが、**default 値としては最良動作点**（実測 2026-07-02:
        // C=64 で 48 に上げると mean −13%。realistic mix は accept が短く長 draft は verify の無駄 row）。
        // hard clamp は撤去済み＝QWISP_DRAFT_K で自由に超過可（高 accept ワークロード向け）。
        let maxK = Tell.envInt("QWISP_DRAFT_K", Swift.max(4, C * 3 / 8))   // debug-only override
        let minMatch = 4    // OAT-tuned(9b157d9): 最適近傍で鈍感
        let maxMatch = 32   // OAT-tuned(9b157d9): 最適近傍で鈍感
        // bolt uses f32-full verify (same as strict SuffixSpec) for accept/reject STABILITY:
        // f16 batched-verify diverges from the sequential reject re-run (spec-gdn-incompat),
        // which corrupts cache state on partial-reject/fallback and cascades to garbage. The
        // speedup comes from buddy (io=0) + spec, NOT from dropping f32-full (only 2 divergent
        // ops: attention SDPA + GDN conv1d; cheap). Buddy still provides the slow-NAND win.
        // ★ entry で依存 static を全て明示 set（in-process 連続実行では前 run の leak が実バグになる）。
        GatedDeltaNetLayer.f32Conv = true; AttentionLayer.f32SDPA = true
        // B2: fusion は out軸 concat の bit-exact 変換で code/mix +8-14% 実測済み → default ON（QWISP_FUSE_GDN=0 で opt-out）
        GatedDeltaNetLayer.fuseGDN = Tell.envStr("QWISP_FUSE_GDN", "1") != "0"
        AttentionLayer.seqMultiToken = false; AttentionLayer.perQueryNone = false
        StreamingMoEBlock.probeNoSync = false; StreamingMoEBlock.skipMode = 0
        StreamingMoEBlock.captureInds = false
        StreamingMoEBlock.countHotMiss = false; StreamingMoEBlock.hotMissAccum = nil
        LayerExpertCache.overflowCheck = false; LayerExpertCache.overflowMaxUnion = 0
        LayerExpertCache.overflowSafeRows = Int.max
        defer {
            StreamingMoEBlock.probeNoSync = false; StreamingMoEBlock.skipMode = 0
            StreamingMoEBlock.captureInds = false; GatedDeltaNetLayer.fuseGDN = false
            GatedDeltaNetLayer.f32Conv = false; AttentionLayer.f32SDPA = false
            AttentionLayer.seqMultiToken = false; AttentionLayer.perQueryNone = false
            StreamingMoEBlock.countHotMiss = false; StreamingMoEBlock.hotMissAccum = nil
            LayerExpertCache.overflowCheck = false
        }
        let isLin = model.isLinearFlags
        let N = Swift.min(Tell.envInt("QWISP_GEN", 48), gR.count)
        let nE = 256, nMoE = model.expertCaches.count

        // phase 1: calib — frequency + co-activation (for buddy table). Exact routing (probeNoSync=false).
        var counts = [[Int]](repeating: [Int](repeating: 0, count: nE), count: nMoE)
        var coact = [[[Int]]](repeating: [[Int]](repeating: [Int](repeating: 0, count: nE), count: nE), count: nMoE)
        let cc = model.makeCaches()
        StreamingMoEBlock.probeNoSync = false; StreamingMoEBlock.skipMode = 0; StreamingMoEBlock.captureInds = true
        var (_, clg) = try model.prefillChunked(ids, caches: cc)
        var ccur = MLX.argMax(clg[0, clg.dim(1) - 1], axis: -1).reshaped([1, 1])
        MLX.eval([ccur] + cc.flatMap { $0.stateArrays })
        for _ in 0 ..< calibN {
            (_, clg) = try model.forwardHidden(ccur, caches: cc)
            MLX.eval([clg] + cc.flatMap { $0.stateArrays } + model.expertCaches.compactMap { $0.lastInds })
            for (mi, ec) in model.expertCaches.enumerated() {
                if let li = ec.lastInds {
                    let es = li.asArray(Int32.self).map { Int($0) }
                    for e in es { counts[mi][e] += 1 }
                    for a in 0 ..< es.count { for b in (a + 1) ..< es.count {
                        coact[mi][es[a]][es[b]] += 1; coact[mi][es[b]][es[a]] += 1 } }
                }
            }
            ccur = MLX.argMax(clg[0, 0], axis: -1).reshaped([1, 1]); MLX.eval([ccur])
        }
        StreamingMoEBlock.captureInds = false

        // Canonical strict-4bit greedy reference (f32-full, exact routing) for losslessness classification.
        // ★正準構成の固定(2026-07-02): canonical = f32-full + **fuseGDN OFF** + chunk-8 prefill +
        //   M=1 sequential decode + sync routing。fusion は kernel 形状依存の微小 drift があり
        //   (fuse-ON ref vs fuse-OFF strict が near-tie p33 で乖離した実測)、bolt の B2 default-ON を
        //   ここに漏らすと ref が汚染される。gSwift ループ中のみ明示 OFF にし、終了後 bolt 設定へ復帰。
        var gSwift: [Int] = []
        if Tell.envFlag("QWISP_SWIFT_REF") {
            let boltFuse = GatedDeltaNetLayer.fuseGDN
            GatedDeltaNetLayer.fuseGDN = false
            defer { GatedDeltaNetLayer.fuseGDN = boltFuse }
            GatedDeltaNetLayer.f32Conv = true; AttentionLayer.f32SDPA = true   // <- canonical L1 reference
            StreamingMoEBlock.probeNoSync = false; StreamingMoEBlock.skipMode = 0
            let cref = model.makeCaches()
            var (_, rlg) = try model.prefillChunked(ids, caches: cref)
            var rcur = MLX.argMax(rlg[0, rlg.dim(1) - 1], axis: -1).reshaped([1, 1])
            MLX.eval([rcur] + cref.flatMap { $0.stateArrays })
            for _ in 0 ..< N {
                gSwift.append(rcur.item(Int.self))
                (_, rlg) = try model.forwardHidden(rcur, caches: cref)
                rcur = MLX.argMax(rlg[0, 0], axis: -1).reshaped([1, 1])
                MLX.eval([rcur] + cref.flatMap { $0.stateArrays })
            }
            // leave f32-full ON — bolt's own verify uses it too (stability, see above).
        }

        // phase 3a: EXACT prefill — the prompt context must be exact; buddy prefill corrupts every
        // downstream token (so even 100% decode escalation cannot reach strict). One-time, off timer.
        let mc = model.makeCaches()
        StreamingMoEBlock.probeNoSync = false; StreamingMoEBlock.skipMode = 0
        let (_, lg) = try model.prefillChunked(ids, caches: mc)
        var uArr = MLX.argMax(lg[0, lg.dim(1) - 1], axis: -1).reshaped([1, 1])
        MLX.eval([uArr] + mc.flatMap { $0.stateArrays })

        // phase 3b: top-C hot-pin + buddy table AFTER prefill (so arena = hot set, buddyTable valid).
        for (mi, ec) in model.expertCaches.enumerated() {
            _ = ec.ensure(Array(counts[mi].enumerated()
                .sorted { $0.element != $1.element ? $0.element > $1.element : $0.offset < $1.offset }
                .prefix(C).map { $0.offset }))
            ec.buildBuddyTable(coact: coact[mi], numExperts: nE)
        }

        // phase 3c: DETERMINISTIC buddy decode. probeNoSync=false keeps the per-layer CPU sync
        // barrier (deterministic, no cross-layer race); skipMode=3 remaps cold->buddy slot in the
        // synced path with NO pread => io=0. This replaces the racy no-sync buddy path.
        StreamingMoEBlock.probeNoSync = false
        StreamingMoEBlock.skipMode = 3            // cold -> buddy slot (deterministic synced remap)
        var hist = ids.asArray(Int32.self).map { Int($0) }

        var out: [Int] = []; var steps = 0, accTok = 0, draftSteps = 0
        ExpertSource.throttleActive = true   // T2: deferred throttle はここ（phase-3c timed decode 開始）から有効
        let t0 = DispatchTime.now()
        while out.count < N {
            steps += 1
            let u = uArr.item(Int.self)
            let drafts = suffixDraft(hist + [u], maxMatch: maxMatch, draftK: maxK, minMatch: minMatch)
            let D = drafts.count
            if D == 0 {                              // nl / novel: single buddy no-sync greedy (fallback)
                let (_, glg) = try model.forwardHidden(uArr, caches: mc)
                out.append(u); hist.append(u)
                uArr = MLX.argMax(glg[0, 0], axis: -1).reshaped([1, 1])
                MLX.eval([uArr] + mc.flatMap { $0.stateArrays })
                continue
            }
            draftSteps += 1
            let snaps = mc.map { $0.snapshot() }
            let seq = MLX.concatenated([uArr, MLXArray(drafts.map { Int32($0) }, [1, D])], axis: 1)  // [1,D+1]
            let (_, vlg) = try model.forwardHidden(seq, caches: mc)   // buddy no-sync batched verify
            let evals = MLX.argMax(vlg[0, 0 ..< (D + 1)], axis: -1).asArray(Int32.self).map { Int($0) }
            var p = 0
            while p < D && drafts[p] == evals[p] { p += 1 }
            out.append(u); hist.append(u)
            for i in 0 ..< p { out.append(drafts[i]); hist.append(drafts[i]) }
            accTok += p
            if p == D {                              // full accept: state already advanced D+1
                uArr = MLXArray([Int32(evals[D])], [1, 1])
                MLX.eval([uArr] + mc.flatMap { $0.stateArrays })
            } else {                                 // partial: restore + re-run buddy prefix to sync state
                for (i, c) in mc.enumerated() { c.restore(snaps[i], isLinear: isLin[i], trim: D + 1) }
                let acc = [u] + Array(drafts.prefix(p))
                _ = try model.forwardHidden(MLXArray(acc.map { Int32($0) }, [1, acc.count]), caches: mc)
                uArr = MLXArray([Int32(evals[p])], [1, 1])
                MLX.eval([uArr] + mc.flatMap { $0.stateArrays })
            }
        }
        let secs = Double(DispatchTime.now().uptimeNanoseconds - t0.uptimeNanoseconds) / 1e9
        let outN = Array(out.prefix(N))
        let matchPy = zip(outN, gR).filter { $0 == $1 }.count
        // Headline near-lossless metric = vs strict-4bit f32-full greedy (Swift-greedy).
        let headline: String
        if gSwift.isEmpty {
            headline = String(format: "品質(vs ref=Swift正準greedy) %d/%d=%.0f%% [QWISP_SWIFT_REF=1 で in-run 再計算]",
                              matchPy, N, Double(matchPy) / Double(N) * 100)
        } else {
            let m = zip(outN, gSwift).filter { $0 == $1 }.count
            headline = String(format: "★near-lossless(vs strict-4bit greedy) %d/%d=%.1f%%  (vs ref %d/%d)",
                              m, N, Double(m) / Double(N) * 100, matchPy, N)
        }
        if Tell.envFlag("QWISP_DUMP_TOKENS") {
            print("PROMPT_TOKENS:" + ids.asArray(Int32.self).map { String($0) }.joined(separator: ","))
            print("BOLT_TOKENS:" + outN.map { String($0) }.joined(separator: ","))
            if !gSwift.isEmpty { print("STRICT_TOKENS:" + gSwift.map { String($0) }.joined(separator: ",")) }
        }
        let tokps = Double(N) / secs
        let summary = String(format: """
            [Bolt L3] buddy no-sync draft(maxK=%d)+buddy verify(C=%d, skipMode=3): %.1f tok/s  \
            accept/step=%.2f  spec-steps=%d/%d  %@
            """, maxK, C, tokps, Double(accTok) / Double(Swift.max(1, draftSteps)), draftSteps, steps, headline)
        return (summary, tokps)
    }
}
