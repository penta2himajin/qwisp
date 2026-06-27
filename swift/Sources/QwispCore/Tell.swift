import Foundation
import MLX
import Metal

/// Tell runtime（William Tell = 的=expert を先読みして射抜く）.
/// mlx の batched eval を回避し、chunk 単位で asyncEval しながら次 chunk の expert を
/// background prefetch で先読み → prefetch I/O を GPU 計算に隠す。cross-layer 予測 prefetch の
/// efficient 化（Fate one-pass 相当）を mlx 上で実現する独自スケジューラ。
public enum Tell {
    // env 読み出しヘルパ（ProcessInfo の冗長な記述を集約）。Tell.envXxx で全 runner から利用。
    static func envInt(_ k: String, _ d: Int) -> Int { Int(ProcessInfo.processInfo.environment[k] ?? "") ?? d }
    static func envFloat(_ k: String, _ d: Float) -> Float { Float(ProcessInfo.processInfo.environment[k] ?? "") ?? d }
    static func envStr(_ k: String, _ d: String) -> String { ProcessInfo.processInfo.environment[k] ?? d }
    static func envFlag(_ k: String) -> Bool { ProcessInfo.processInfo.environment[k] == "1" }
    /// **buddy-draft speculative decode（8GB strict lossless ベースライン）**
    /// - 機構: buddy no-sync で K トークン draft → exact batched verify(seqMultiToken)で照合、draft==exact の prefix を採用、外れは exact 訂正
    /// - lossless: **strict**（long horizon でも vs Swift-exact 100%。唯一の真 lossless）
    /// - 速度: 8GB C=64 ~27 / 16GB C=128 ~34-36 tok/s（accept/step ~3.7-4.0）
    /// - 研究: Speculative Decoding (Leviathan 2023, Chen 2023); draft=BuddyMoE (2511.10054)
    /// - env: QWISP_SPECK / QWISP_DRAFT_K / QWISP_SKIPMODE=3(buddy) / QWISP_CACHE_C
    /// - 旧名: SpecK / runHotColdSpecK（git ed60472, 382ccf7）。詳細 notes/00-strict-vs-near-lossless.md
    public static func runSpecVerify(modelDir: String, refPath: String) throws -> String {
        guard let device = MTLCreateSystemDefaultDevice() else { return "ERROR: no Metal device" }
        let r = try loadArrays(url: URL(fileURLWithPath: refPath))
        guard let promptArr = r["spec_prompt"], let gRef = r["spec_greedy"] else { return "[HotColdSpecK] skip" }
        let C = Tell.envInt("QWISP_CACHE_C", 64)
        let calibN = Tell.envInt("QWISP_CALIB", 32)
        let K = Tell.envInt("QWISP_DRAFT_K", 4)
        let skipStride = Tell.envInt("QWISP_SKIP_STRIDE", 0)
        let buddy = ProcessInfo.processInfo.environment["QWISP_SKIPMODE"] == "3"   // draft を buddy 代替で高精度化
        let store = try WeightStore(modelDir: modelDir)
        store.residentNonExperts()
        let source = try ExpertSource(modelDir: modelDir); try source.warm()
        let arena = try ExpertArena(device: device, source: source, N: 64)
        let model = try StreamingQwispModel(store: store, arena: arena, device: device,
                                            source: source, cacheC: C)
        let ids = promptArr.asType(.int32).reshaped([1, promptArr.dim(0)])
        let gR = gRef.asArray(Int32.self).map { Int($0) }
        let isLin = model.isLinearFlags
        let N = Swift.min(Tell.envInt("QWISP_GEN", 64), gR.count)
        let nE = 256, nMoE = model.expertCaches.count
        let caches = model.makeCaches()
        var counts = [[Int]](repeating: [Int](repeating: 0, count: nE), count: nMoE)

        var coact: [[[Int]]] = buddy
            ? [[[Int]]](repeating: [[Int]](repeating: [Int](repeating: 0, count: nE), count: nE), count: nMoE) : []
        StreamingMoEBlock.probeNoSync = false; StreamingMoEBlock.captureInds = true
        var (_, lg) = try model.prefillChunked(ids, caches: caches)
        var cur = MLX.argMax(lg[0, lg.dim(1) - 1], axis: -1).reshaped([1, 1])
        MLX.eval([cur] + caches.flatMap { $0.stateArrays })
        for _ in 0 ..< calibN {
            (_, lg) = try model.forwardHidden(cur, caches: caches)
            MLX.eval([lg] + caches.flatMap { $0.stateArrays } + model.expertCaches.compactMap { $0.lastInds })
            for (mi, ec) in model.expertCaches.enumerated() {
                if let li = ec.lastInds {
                    let es = li.asArray(Int32.self).map { Int($0) }
                    for e in es { counts[mi][e] += 1 }
                    if buddy { for a in 0 ..< es.count { for b in (a + 1) ..< es.count {
                        coact[mi][es[a]][es[b]] += 1; coact[mi][es[b]][es[a]] += 1 } } }
                }
            }
            cur = MLX.argMax(lg[0, 0], axis: -1).reshaped([1, 1]); MLX.eval([cur])
        }
        StreamingMoEBlock.captureInds = false
        for (mi, ec) in model.expertCaches.enumerated() {
            _ = ec.ensure(Array(counts[mi].enumerated()
                .sorted { $0.element != $1.element ? $0.element > $1.element : $0.offset < $1.offset }
                .prefix(C).map { $0.offset }))
            if buddy { ec.buildBuddyTable(coact: coact[mi], numExperts: nE) }
        }
        defer { StreamingMoEBlock.skipMode = 0 }

        // Swift-exact-greedy 参照（lossless 検証用。SpecK は構成上 strict lossless なので 100% のはず。
        // long128tok は vs-Python が f16 で無意味なので vs Swift-greedy で測る）。QWISP_SWIFT_REF=1。
        var gSwift: [Int] = []
        if Tell.envFlag("QWISP_SWIFT_REF") {
            let cref = model.makeCaches()
            StreamingMoEBlock.probeNoSync = false; StreamingMoEBlock.skipMode = 0
            var (_, rlg) = try model.prefillChunked(ids, caches: cref)
            var rcur = MLX.argMax(rlg[0, rlg.dim(1) - 1], axis: -1).reshaped([1, 1])
            MLX.eval([rcur] + cref.flatMap { $0.stateArrays })
            for _ in 0 ..< N {
                gSwift.append(rcur.item(Int.self))
                (_, rlg) = try model.forwardHidden(rcur, caches: cref)
                rcur = MLX.argMax(rlg[0, 0], axis: -1).reshaped([1, 1])
                MLX.eval([rcur] + cref.flatMap { $0.stateArrays })
            }
        }

        let mc = model.makeCaches()
        StreamingMoEBlock.probeNoSync = false
        (_, lg) = try model.prefillChunked(ids, caches: mc)
        var uArr = MLX.argMax(lg[0..., (lg.dim(1) - 1)...], axis: -1)
        MLX.eval([uArr] + mc.flatMap { $0.stateArrays })
        // layer-skip draft 集合（skipStride≥2: i%stride==stride-1 を間引く。層0/末尾は残す）
        let L = model.layerCount
        var skip = Set<Int>()
        if skipStride >= 2 { for i in 1 ..< (L - 1) where i % skipStride == (skipStride - 1) { skip.insert(i) } }

        let prof = Tell.envFlag("QWISP_SPECK_PROF")
        var tDraft: UInt64 = 0, tVerify: UInt64 = 0, tCommit: UInt64 = 0
        func now() -> UInt64 { DispatchTime.now().uptimeNanoseconds }
        var out: [Int] = []; var steps = 0, accTok = 0
        let t0 = DispatchTime.now()
        while out.count < N {
            steps += 1
            // draft K（no-sync + layer-skip, state は捨てる）
            var ts = now()
            let snaps0 = mc.map { $0.snapshot() }
            StreamingMoEBlock.probeNoSync = true
            StreamingMoEBlock.skipMode = buddy ? 3 : 0   // draft のみ buddy 代替（verify は exact）
            var drafts: [Int] = []; var dcur = uArr
            for _ in 0 ..< K {
                let (_, dl) = skip.isEmpty
                    ? try model.forwardHidden(dcur, caches: mc)
                    : try model.forwardHiddenSkip(dcur, caches: mc, skip: skip)
                dcur = MLX.argMax(dl[0..., (dl.dim(1) - 1)...], axis: -1)
                drafts.append(dcur.item(Int.self))
            }
            // skip した層は未実行＝未前進なので trim しない（非skip層のみ巻き戻し）
            for (i, c) in mc.enumerated() where !skip.contains(i) { c.restore(snaps0[i], isLinear: isLin[i], trim: K) }
            if prof { tDraft += now() - ts; ts = now() }
            // verify batched [u, d1..dK]。QWISP_VERIFY_NOSYNC=1 で no-sync 化(draft が cache に載せた
            // expert を流用し per-layer sync を消す=near-lossless で高速)。QWISP_VERIFY_SEQ=0 で seqMT 無効。
            let vseq = ProcessInfo.processInfo.environment["QWISP_VERIFY_SEQ"] != "0"
            let vNoSync = Tell.envFlag("QWISP_VERIFY_NOSYNC")
            StreamingMoEBlock.probeNoSync = vNoSync; StreamingMoEBlock.skipMode = 0   // verify は exact gather
            AttentionLayer.seqMultiToken = vseq
            let snaps1 = mc.map { $0.snapshot() }
            let draftArr = MLXArray(drafts.map { Int32($0) }, [1, K])
            let seq = MLX.concatenated([uArr, draftArr], axis: 1)            // [1, K+1]
            let (_, vlg) = try model.forwardHidden(seq, caches: mc)
            let evals = MLX.argMax(vlg[0, 0 ..< (K + 1)], axis: -1).asArray(Int32.self).map { Int($0) }
            if prof { tVerify += now() - ts; ts = now() }
            // accept: drafts[i]==evals[i] が続く長さ p
            var p = 0
            while p < K && drafts[p] == evals[p] { p += 1 }
            out.append(uArr.item(Int.self))
            for i in 0 ..< p { out.append(drafts[i]) }                       // 受理 draft
            accTok += p
            if p == K {
                uArr = MLXArray([Int32(evals[K])], [1, 1])                   // 全受理→次=最終位置 exact
                AttentionLayer.seqMultiToken = false
                MLX.eval([uArr] + mc.flatMap { $0.stateArrays })             // 全 K+1 commit 済
            } else {
                // reject: pre-verify に戻し accepted prefix [u, d1..dp] を exact 再走で commit
                for (i, c) in mc.enumerated() { c.restore(snaps1[i], isLinear: isLin[i], trim: K + 1) }
                let acceptedTok = [uArr.item(Int.self)] + Array(drafts.prefix(p))
                let accSeq = MLXArray(acceptedTok.map { Int32($0) }, [1, acceptedTok.count])
                _ = try model.forwardHidden(accSeq, caches: mc)
                AttentionLayer.seqMultiToken = false
                uArr = MLXArray([Int32(evals[p])], [1, 1])                   // 訂正トークン
                MLX.eval([uArr] + mc.flatMap { $0.stateArrays })
            }
            if prof { tCommit += now() - ts }
        }
        StreamingMoEBlock.probeNoSync = false; AttentionLayer.seqMultiToken = false
        let secs = Double(DispatchTime.now().uptimeNanoseconds - t0.uptimeNanoseconds) / 1e9
        let match = zip(out.prefix(N), gR).filter { $0 == $1 }.count
        let outN = Array(out.prefix(N))
        let swiftTag = gSwift.isEmpty ? ""
            : String(format: "  [vs Swift-greedy %d/%d=%.0f%%]",
                     zip(outN, gSwift).filter { $0 == $1 }.count, N,
                     Double(zip(outN, gSwift).filter { $0 == $1 }.count) / Double(N) * 100)
        if prof {
            let s = Double(steps)
            FileHandle.standardError.write(String(format:
                "[SPECK-PROF/step] draft(K×no-sync)=%.1f verify(seqMT exact)=%.1f commit/reject=%.1f (ms)  steps=%d\n",
                Double(tDraft)/s/1e6, Double(tVerify)/s/1e6, Double(tCommit)/s/1e6, steps).data(using: .utf8)!)
        }
        return String(format: """
            [HotColdSpecK] %@draft K=%d skip=%d/%d + batched verify: %.1f tok/s  accept/step=%.2f  品質(vs Python) %d/%d=%.0f%%%@
            """, buddy ? "buddy-" : "no-sync ", K, skip.count, L, Double(N) / secs, Double(accTok) / Double(steps), match, N, Double(match) / Double(N) * 100, swiftTag)
    }

    /// output-similarity buddy（研究推奨指標, Qwen は weight 相関無しなので output 必須）:
    /// 全 256 expert を calib 活性 acts[A,H] に通し、出力 fingerprint の cosine で各 cold expert の
    /// 最類似 hot expert を選ぶ。co-activation(共起=補完的) でなく functional equivalence(置換可) を測る。
    static func outputSimBuddyTable(device: MTLDevice, source: ExpertSource, layer: Int,
                                    acts: MLXArray, slotOf: [Int: Int], expertBits: Int,
                                    numExperts: Int) throws -> MLXArray {
        let tmp = try ExpertArena(device: device, source: source, N: numExperts, refLayer: layer)
        try tmp.load(layer, Array(0 ..< numExperts))
        let A = acts.dim(0), H = acts.dim(1)
        let xe = acts.expandedDimensions(axes: [-2, -3])            // [A,1,1,H]
        var rv = [Int32](); rv.reserveCapacity(A * numExperts)
        for _ in 0 ..< A { for e in 0 ..< numExperts { rv.append(Int32(e)) } }
        let remap = MLXArray(rv, [A, numExperts]).asType(.uint32)
        func gq(_ x: MLXArray, _ proj: String) -> MLXArray {
            gatherQuantizedMatmul(x, tmp.arr(proj, "weight"), scales: tmp.arr(proj, "scales"),
                                  biases: tmp.arr(proj, "biases"), rhsIndices: remap, transpose: true,
                                  groupSize: 64, bits: expertBits, mode: .affine, sortedIndices: false)
        }
        let g = gq(xe, "gate_proj")                                 // [A,E,1,I]
        let u = gq(xe, "up_proj")
        let h = (g * MLX.sigmoid(g)) * u
        let d = gq(h, "down_proj").squeezed(axis: -2)               // [A,E,H]
        let fp = d.transposed(1, 0, 2).reshaped([numExperts, A * H])  // [E, A*H]
        let norm = MLX.sqrt((fp * fp).sum(axis: -1, keepDims: true)) + 1e-6
        let fpn = fp / norm
        let sim = MLX.matmul(fpn, fpn.transposed()); sim.eval()      // [E, E] cosine
        let simArr = sim.asArray(Float.self)
        let hot = Array(slotOf.keys)
        var bmap = [Int32](repeating: 0, count: numExperts)
        for e in 0 ..< numExperts {
            if let s = slotOf[e] { bmap[e] = Int32(s); continue }
            var bestH = -1; var bestSim: Float = -2
            for hexp in hot { let sv = simArr[e * numExperts + hexp]; if sv > bestSim { bestSim = sv; bestH = hexp } }
            bmap[e] = bestH >= 0 ? Int32(slotOf[bestH]!) : 0
        }
        let arr = MLXArray(bmap, [numExperts]); arr.eval()
        return arr
    }

    /// **pure no-sync buddy substitution（最速 near-lossless）**
    /// - 機構: hot-pin した top-C を no-sync gather、cold は co-activation 最類似 hot に table 差替（verify 無し・per-token コスト0）
    /// - lossless: **near**（C=64 で vs Swift-exact ~98%。C>64 では発散、verify 無しゆえ保証なし）
    /// - 速度: 8GB C=64 ~56-58 tok/s（RSS 6.9GB）
    /// - 研究: BuddyMoE (2511.10054), expert-skip (2402.14800)
    /// - env: QWISP_FAST / QWISP_SKIPMODE=3 / QWISP_CACHE_C / QWISP_BUDDY_OUTSIM(neg)
    /// - 旧名: Fast / runHotColdFast（git 0b923e0, f668e62）
    public static func runBuddyNoSync(modelDir: String, refPath: String) throws -> String {
        guard let device = MTLCreateSystemDefaultDevice() else { return "ERROR: no Metal device" }
        let r = try loadArrays(url: URL(fileURLWithPath: refPath))
        guard let promptArr = r["spec_prompt"], let gRef = r["spec_greedy"] else { return "[HotColdFast] skip" }
        let C = Tell.envInt("QWISP_CACHE_C", 64)
        let calibN = Tell.envInt("QWISP_CALIB", 48)
        let store = try WeightStore(modelDir: modelDir)
        store.residentNonExperts()
        let source = try ExpertSource(modelDir: modelDir); try source.warm()
        let arena = try ExpertArena(device: device, source: source, N: 64)
        let model = try StreamingQwispModel(store: store, arena: arena, device: device,
                                            source: source, cacheC: C)
        let ids = promptArr.asType(.int32).reshaped([1, promptArr.dim(0)])
        let gR = gRef.asArray(Int32.self).map { Int($0) }
        let N = Swift.min(Tell.envInt("QWISP_GEN", 64), gR.count)
        let nE = 256, nMoE = model.expertCaches.count
        let caches = model.makeCaches()

        let skipMode = Tell.envInt("QWISP_SKIPMODE", 0)
        let outsim = Tell.envFlag("QWISP_BUDDY_OUTSIM")  // output 類似度 buddy
        let aMax = Tell.envInt("QWISP_OUTSIM_A", 8)  // 出力 fingerprint 用活性数
        // --- phase 1: calibration（exact decode で頻度集計 + buddy 用 co-activation/活性）---
        var counts = [[Int]](repeating: [Int](repeating: 0, count: nE), count: nMoE)
        // co-activation[layer][e][e'] = 同 token で共 routed した回数（mode3 && !outsim のみ）
        var coact: [[[Int]]] = (skipMode == 3 && !outsim)
            ? [[[Int]]](repeating: [[Int]](repeating: [Int](repeating: 0, count: nE), count: nE), count: nMoE) : []
        var acts: [[MLXArray]] = (skipMode == 3 && outsim) ? [[MLXArray]](repeating: [], count: nMoE) : []  // 出力類似度用
        StreamingMoEBlock.probeNoSync = false; StreamingMoEBlock.captureInds = true
        if skipMode == 3 && outsim { StreamingMoEBlock.captureGateInput = true }
        var (_, lg) = try model.prefillChunked(ids, caches: caches)
        var cur = MLX.argMax(lg[0, lg.dim(1) - 1], axis: -1).reshaped([1, 1])
        MLX.eval([cur] + caches.flatMap { $0.stateArrays })
        for _ in 0 ..< calibN {
            (_, lg) = try model.forwardHidden(cur, caches: caches)
            MLX.eval([lg] + caches.flatMap { $0.stateArrays } + model.expertCaches.compactMap { $0.lastInds }
                     + (outsim ? model.expertCaches.compactMap { $0.lastGateInput } : []))
            for (mi, ec) in model.expertCaches.enumerated() {
                if let li = ec.lastInds {
                    let es = li.asArray(Int32.self).map { Int($0) }
                    for e in es { counts[mi][e] += 1 }
                    if skipMode == 3 && !outsim {
                        for a in 0 ..< es.count { for b in (a + 1) ..< es.count {
                            coact[mi][es[a]][es[b]] += 1; coact[mi][es[b]][es[a]] += 1
                        } }
                    }
                }
                if skipMode == 3 && outsim, acts[mi].count < aMax, let gi = ec.lastGateInput {
                    acts[mi].append(gi[(gi.dim(0) - 1)...])   // [1,H] 最終位置
                }
            }
            cur = MLX.argMax(lg[0, 0], axis: -1).reshaped([1, 1])
            MLX.eval([cur] + caches.flatMap { $0.stateArrays })
        }
        StreamingMoEBlock.captureInds = false; StreamingMoEBlock.captureGateInput = false

        // Swift-exact-greedy 参照（f16-near-lossless 評価用。Python-ref は長 horizon で f16 乖離する
        // ので、同一エンジンの exact greedy を真値に no-sync/buddy の品質を測る）。QWISP_SWIFT_REF=1。
        var gSwift: [Int] = []
        if Tell.envFlag("QWISP_SWIFT_REF") {
            let cref = model.makeCaches()
            StreamingMoEBlock.probeNoSync = false
            var (_, rlg) = try model.prefillChunked(ids, caches: cref)
            var rcur = MLX.argMax(rlg[0, rlg.dim(1) - 1], axis: -1).reshaped([1, 1])
            MLX.eval([rcur] + cref.flatMap { $0.stateArrays })
            for _ in 0 ..< N {
                gSwift.append(rcur.item(Int.self))
                (_, rlg) = try model.forwardHidden(rcur, caches: cref)
                rcur = MLX.argMax(rlg[0, 0], axis: -1).reshaped([1, 1])
                MLX.eval([rcur] + cref.flatMap { $0.stateArrays })
            }
        }

        // --- phase 2: top-C hot を各層 pin（ensure で常駐ロード）+ buddy table 構築 ---
        for (mi, ec) in model.expertCaches.enumerated() {
            // tie は index 昇順で決定的に（非安定ソートの run 間ブレ＝hot set 変動を排除）
            let hot = Array(counts[mi].enumerated()
                .sorted { $0.element != $1.element ? $0.element > $1.element : $0.offset < $1.offset }
                .prefix(C).map { $0.offset })
            _ = ec.ensure(hot)
            if skipMode == 3 {
                if outsim {
                    let A = MLX.concatenated(acts[mi], axis: 0)   // [A,H]
                    ec.buddyTable = try outputSimBuddyTable(device: device, source: source, layer: ec.layer,
                                                            acts: A, slotOf: ec.slotMap, expertBits: 4, numExperts: nE)
                } else {
                    ec.buildBuddyTable(coact: coact[mi], numExperts: nE)
                }
            }
        }

        // --- phase 3: 実プロンプトから pure no-sync decode（ensure 無し＝hot 固定）---
        let caches2 = model.makeCaches()
        StreamingMoEBlock.probeNoSync = false
        (_, lg) = try model.prefillChunked(ids, caches: caches2)
        cur = MLX.argMax(lg[0, lg.dim(1) - 1], axis: -1).reshaped([1, 1])
        MLX.eval([cur] + caches2.flatMap { $0.stateArrays })
        StreamingMoEBlock.probeNoSync = true   // 以降 no-sync（hot 固定 slotTable で gather）
        StreamingMoEBlock.skipMode = skipMode       // 1=cold寄与0, 2=0+renorm, 3=buddy 代替
        // coverage 計測: 全層・全 token の routed-but-not-cached(miss) 数を GPU 累積。
        // C=128 で 0 なら no-sync gather が exact 経路と bit 一致＝構成上 strict lossless。
        let countMiss = ProcessInfo.processInfo.environment["QWISP_COUNT_MISS"] != "0"
        StreamingMoEBlock.countHotMiss = countMiss
        StreamingMoEBlock.hotMissAccum = nil
        var out: [Int] = []
        let t0 = DispatchTime.now()
        for _ in 0 ..< N {
            out.append(cur.item(Int.self))
            (_, lg) = try model.forwardHidden(cur, caches: caches2)
            cur = MLX.argMax(lg[0, 0], axis: -1).reshaped([1, 1])
            MLX.eval([cur] + caches2.flatMap { $0.stateArrays })
        }
        var missTotal = -1
        if countMiss, let acc = StreamingMoEBlock.hotMissAccum { acc.eval(); missTotal = acc.item(Int.self) }
        StreamingMoEBlock.probeNoSync = false; StreamingMoEBlock.skipMode = 0
        StreamingMoEBlock.countHotMiss = false; StreamingMoEBlock.hotMissAccum = nil
        let rss = StreamingDecode.rssGB()
        let secs = Double(DispatchTime.now().uptimeNanoseconds - t0.uptimeNanoseconds) / 1e9
        let match = zip(out, gR).filter { $0 == $1 }.count
        let skipTag = skipMode == 1 ? "+skip" : (skipMode == 2 ? "+skip-renorm" : (skipMode == 3 ? "+buddy" : ""))
        let swiftTag = gSwift.isEmpty ? ""
            : String(format: "  [vs Swift-greedy %d/%d=%.0f%%]",
                     zip(out, gSwift).filter { $0 == $1 }.count, N,
                     Double(zip(out, gSwift).filter { $0 == $1 }.count) / Double(N) * 100)
        let missTag = missTotal < 0 ? "" : String(format: "  miss=%d (%.2f/tok)", missTotal, Double(missTotal) / Double(N))
        return String(format: """
            [HotColdFast] hot-pin top-%d + pure no-sync%@ (calib=%d): %.1f tok/s  品質(vs Python) %d/%d=%.0f%%%@%@  RSS=%.1fGB
            """, C, skipTag, calibN, Double(N) / secs, match, N, Double(match) / Double(N) * 100, swiftTag, missTag, rss)
    }
}
