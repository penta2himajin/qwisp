import Foundation
import MLX
import Metal

/// Tell の探索/計測バリアント置き場。
/// 本流は Tell.swift の 2 系統（runSpecVerify=8GB strict lossless ベースライン /
/// runBuddyNoSync=最速 near-lossless buddy no-sync）。ここはそれ以外の M 系・hybrid・
/// 各種 calib・診断(measureMLXFidelity 等)を保管し、QWISP_* ゲート経由で呼び出す。
/// 詳細・実測は notes/00-strict-vs-near-lossless.md を参照。
extension Tell {
    /// **[計測] top-B hot expert の routing coverage**
    /// - 機構: 実 decode の per-layer expert 使用頻度を集計、top-B が routing の何%を覆うか
    /// - 結果例: top-64 が code 86% / nl 94%（B128=100%）
    /// - env: QWISP_HOTCOLD_CALIB
    /// - 旧名: HotColdCalib / runHotColdCalib（git a54785a）
    public static func measureCoverage(modelDir: String, refPath: String) throws -> String {
        guard let device = MTLCreateSystemDefaultDevice() else { return "ERROR: no Metal device" }
        let r = try loadArrays(url: URL(fileURLWithPath: refPath))
        guard let promptArr = r["spec_prompt"], let gRef = r["spec_greedy"] else { return "[Coverage] skip" }
        let C = Tell.envInt("QWISP_CACHE_C", 64)
        let store = try WeightStore(modelDir: modelDir)
        store.residentNonExperts()
        let source = try ExpertSource(modelDir: modelDir); try source.warm()
        let arena = try ExpertArena(device: device, source: source, N: 64)
        let model = try StreamingQwispModel(store: store, arena: arena, device: device,
                                            source: source, cacheC: C)
        let ids = promptArr.asType(.int32).reshaped([1, promptArr.dim(0)])
        let gR = gRef.asArray(Int32.self).map { Int($0) }
        let N = Swift.min(Tell.envInt("QWISP_GEN", 64), gR.count)
        let nE = 256
        let caches = model.makeCaches()
        let nMoE = model.expertCaches.count
        var counts = [[Int]](repeating: [Int](repeating: 0, count: nE), count: nMoE)  // [moeLayer][expert]
        var total = 0

        StreamingMoEBlock.probeNoSync = false; StreamingMoEBlock.captureInds = true
        var (_, lg) = try model.prefillChunked(ids, caches: caches)
        var cur = MLX.argMax(lg[0, lg.dim(1) - 1], axis: -1).reshaped([1, 1])
        MLX.eval([cur] + caches.flatMap { $0.stateArrays })
        for _ in 0 ..< N {
            (_, lg) = try model.forwardHidden(cur, caches: caches)
            MLX.eval([lg] + caches.flatMap { $0.stateArrays }
                     + model.expertCaches.compactMap { $0.lastInds })
            for (mi, ec) in model.expertCaches.enumerated() {
                guard let li = ec.lastInds else { continue }
                for e in li.asArray(Int32.self) { counts[mi][Int(e)] += 1; if mi == 0 { } }
            }
            total += 1
            cur = MLX.argMax(lg[0, 0], axis: -1).reshaped([1, 1])
            MLX.eval([cur] + caches.flatMap { $0.stateArrays })
        }
        StreamingMoEBlock.probeNoSync = false; StreamingMoEBlock.captureInds = false

        // 各層 top-B coverage = top-B expert の使用回数 / 全使用回数。層平均を出す。
        func coverage(_ B: Int) -> Double {
            var sum = 0.0
            for mi in 0 ..< nMoE {
                let sorted = counts[mi].sorted(by: >)
                let tot = sorted.reduce(0, +)
                let top = sorted.prefix(B).reduce(0, +)
                sum += tot > 0 ? Double(top) / Double(tot) : 0
            }
            return sum / Double(nMoE) * 100
        }
        // distinct expert 数の層平均（活性集合サイズ）
        var distinctAvg = 0.0
        for mi in 0 ..< nMoE { distinctAvg += Double(counts[mi].filter { $0 > 0 }.count) }
        distinctAvg /= Double(nMoE)

        return String(format: """
            [Coverage-CALIB] %d tok, %d MoE層, top-k=8/256。per-layer 活性 expert 平均=%.0f/256
              top-B hot coverage(routing の何%%): B16=%.0f%% B32=%.0f%% B48=%.0f%% B64=%.0f%% B96=%.0f%% B128=%.0f%%
            """,
            total, nMoE, distinctAvg,
            coverage(16), coverage(32), coverage(48), coverage(64), coverage(96), coverage(128))
    }

    /// **SS-MoE 流 no-sync draft + exact verify（初期投機・死路）**
    /// - 機構: hot-pin no-sync draft を exact verify で照合（SpecK=runSpecVerify の前身）
    /// - lossless: strict だが ❌**速度出ず**（accept 0.94-0.97 でも sync 律速で ~20-24 < M0）
    /// - 研究: SS-MoE (no-sync draft + exact verify)
    /// - env: QWISP_HOTCOLD_SPEC
    /// - 旧名: HotColdSpec / runHotColdSpec（git e398e10）
    public static func runSSMoEDraftVerify(modelDir: String, refPath: String) throws -> String {
        guard let device = MTLCreateSystemDefaultDevice() else { return "ERROR: no Metal device" }
        let r = try loadArrays(url: URL(fileURLWithPath: refPath))
        guard let promptArr = r["spec_prompt"], let gRef = r["spec_greedy"] else { return "[SSMoEDraftVerify] skip" }
        let C = Tell.envInt("QWISP_CACHE_C", 64)
        let calibN = Tell.envInt("QWISP_CALIB", 32)
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

        // calib → hot pin（draft の no-sync を高精度化）
        StreamingMoEBlock.probeNoSync = false; StreamingMoEBlock.captureInds = true
        var (_, lg) = try model.prefillChunked(ids, caches: caches)
        var cur = MLX.argMax(lg[0, lg.dim(1) - 1], axis: -1).reshaped([1, 1])
        MLX.eval([cur] + caches.flatMap { $0.stateArrays })
        for _ in 0 ..< calibN {
            (_, lg) = try model.forwardHidden(cur, caches: caches)
            MLX.eval([lg] + caches.flatMap { $0.stateArrays } + model.expertCaches.compactMap { $0.lastInds })
            for (mi, ec) in model.expertCaches.enumerated() {
                if let li = ec.lastInds { for e in li.asArray(Int32.self) { counts[mi][Int(e)] += 1 } }
            }
            cur = MLX.argMax(lg[0, 0], axis: -1).reshaped([1, 1]); MLX.eval([cur])
        }
        StreamingMoEBlock.captureInds = false
        for (mi, ec) in model.expertCaches.enumerated() {
            _ = ec.ensure(Array(counts[mi].enumerated().sorted { $0.element > $1.element }.prefix(C).map { $0.offset }))
        }

        // fresh prefill → D1 self-spec（no-sync draft + exact verify）
        let mc = model.makeCaches()
        StreamingMoEBlock.probeNoSync = false
        (_, lg) = try model.prefillChunked(ids, caches: mc)
        var uArr = MLX.argMax(lg[0..., (lg.dim(1) - 1)...], axis: -1)
        MLX.eval([uArr] + mc.flatMap { $0.stateArrays })
        var out: [Int] = []; var steps = 0, acc = 0
        let t0 = DispatchTime.now()
        while out.count < N {
            steps += 1
            // draft: no-sync forward of u → d（state は捨てる: snapshot→forward→restore）
            let snaps = mc.map { $0.snapshot() }
            StreamingMoEBlock.probeNoSync = true
            let (_, dlg) = try model.forwardHidden(uArr, caches: mc)
            let dArr = MLX.argMax(dlg[0..., (dlg.dim(1) - 1)...], axis: -1)
            MLX.eval([dArr])
            for (i, c) in mc.enumerated() { c.restore(snaps[i], isLinear: isLin[i], trim: 1) }
            // verify: exact [u, d]（seqMultiToken で lossless）
            StreamingMoEBlock.probeNoSync = false; AttentionLayer.seqMultiToken = true
            let snaps2 = mc.map { $0.snapshot() }
            let ud = MLX.concatenated([uArr, dArr], axis: 1)
            let (_, lg2) = try model.forwardHidden(ud, caches: mc)
            let vw = MLX.argMax(lg2[0, 0 ..< 2], axis: -1)
            AttentionLayer.seqMultiToken = false
            let vals = MLX.concatenated([dArr[0], vw]).asArray(Int32.self)
            let d = Int(vals[0]), v = Int(vals[1])
            out.append(uArr.item(Int.self))
            if v == d {                                   // accept 2
                acc += 1; out.append(d)
                uArr = vw[1 ..< 2].reshaped([1, 1])
            } else {                                      // reject → [u] のみ commit
                for (i, c) in mc.enumerated() { c.restore(snaps2[i], isLinear: isLin[i], trim: 2) }
                _ = try model.forwardHidden(uArr, caches: mc)
                uArr = vw[0 ..< 1].reshaped([1, 1])
            }
            MLX.eval([uArr] + mc.flatMap { $0.stateArrays })
        }
        StreamingMoEBlock.probeNoSync = false; AttentionLayer.seqMultiToken = false
        let secs = Double(DispatchTime.now().uptimeNanoseconds - t0.uptimeNanoseconds) / 1e9
        let match = zip(out.prefix(N), gR).filter { $0 == $1 }.count
        return String(format: """
            [SSMoEDraftVerify] no-sync draft + exact verify(C=%d): %.1f tok/s  accept=%.3f  品質 %d/%d=%.0f%%
            """, C, Double(N) / secs, Double(acc) / Double(steps), match, N, Double(match) / Double(N) * 100)
    }

    /// **[計測] pre-attention expert 予測器の recall**
    /// - 機構: 学習不要(閉形式 ridge)の pre-attention 予測器を構築し routing recall を測る
    /// - 結果: GDN hybrid ゆえ ~82-84% 止まり（標準 attention 論文値 94.69% に未達）
    /// - env: QWISP_PRED_CALIB
    /// - 旧名: PredictorCalib / runPredictorCalib（git 43f3607）
    public static func measurePredictorRecall(modelDir: String, refPath: String) throws -> String {
        guard let device = MTLCreateSystemDefaultDevice() else { return "ERROR: no Metal device" }
        let r = try loadArrays(url: URL(fileURLWithPath: refPath))
        guard let promptArr = r["spec_prompt"], let gRef = r["spec_greedy"] else { return "[PredictorRecall] skip" }
        let store = try WeightStore(modelDir: modelDir)
        store.residentNonExperts()
        let source = try ExpertSource(modelDir: modelDir); try source.warm()
        let arena = try ExpertArena(device: device, source: source, N: 64)
        let model = try StreamingQwispModel(store: store, arena: arena, device: device, source: source, cacheC: 64)
        let ids = promptArr.asType(.int32).reshaped([1, promptArr.dim(0)])
        _ = gRef
        let N = Tell.envInt("QWISP_GEN", 512)   // data 量（gR 非依存）
        let nMoE = model.expertCaches.count
        let H = 2048
        let caches = model.makeCaches()

        StreamingMoEBlock.probeNoSync = false
        StreamingMoEBlock.captureInds = true; StreamingMoEBlock.captureLayerInput = true
        var (_, lg) = try model.prefillChunked(ids, caches: caches)
        var cur = MLX.argMax(lg[0, lg.dim(1) - 1], axis: -1).reshaped([1, 1])
        MLX.eval([cur] + caches.flatMap { $0.stateArrays })
        var Xacc = [[MLXArray]](repeating: [], count: nMoE)   // per layer: list of [1,H]
        var Yacc = [[Int32]](repeating: [], count: nMoE)      // per layer: flat top-8 ids
        for _ in 0 ..< N {
            (_, lg) = try model.forwardHidden(cur, caches: caches)
            MLX.eval([lg] + caches.flatMap { $0.stateArrays }
                     + model.expertCaches.compactMap { $0.lastInds }
                     + model.expertCaches.compactMap { $0.preAttnInput })
            for (mi, ec) in model.expertCaches.enumerated() {
                if let x = ec.preAttnInput, let li = ec.lastInds {
                    Xacc[mi].append(x.reshaped([1, H]).asType(.float32))
                    Yacc[mi].append(contentsOf: li.asArray(Int32.self))   // [8]
                }
            }
            cur = MLX.argMax(lg[0, 0], axis: -1).reshaped([1, 1]); MLX.eval([cur])
        }
        StreamingMoEBlock.captureInds = false; StreamingMoEBlock.captureLayerInput = false

        var dict: [String: MLXArray] = [:]
        for l in 0 ..< nMoE {
            dict["X_\(l)"] = MLX.concatenated(Xacc[l], axis: 0)           // [N, H]
            dict["Y_\(l)"] = MLXArray(Yacc[l], [Yacc[l].count / 8, 8])    // [N, 8]
        }
        dict["meta"] = MLXArray([Int32(N), Int32(nMoE), Int32(H)], [3])
        let outPath = Tell.envStr("QWISP_PRED_OUT", "/tmp/qwisp_predictor_data.safetensors")
        try MLX.save(arrays: dict, url: URL(fileURLWithPath: outPath))
        return "[PredictorRecall] dumped X/Y for \(nMoE)層 × \(N) tok (H=\(H)) → \(outPath)"
    }

    /// **[計測] mmap 全 expert resident gather**
    /// - 機構: 全 expert を mmap 常駐し sync 無し gather（8GB に locality 無しの確認）
    /// - 結果: ~45 tok/s だが RSS ~24GB 要（8GB 不可、arena+sync が 8GB の正解）
    /// - env: QWISP_MMAP_GATHER
    /// - 旧名: MmapGather / runMmapGather（git fed8578）
    public static func measureMmapGather(modelDir: String, refPath: String) throws -> String {
        let r = try loadArrays(url: URL(fileURLWithPath: refPath))
        guard let promptArr = r["spec_prompt"], let gRef = r["spec_greedy"] else { return "[MmapGather] skip" }
        let store = try WeightStore(modelDir: modelDir)
        store.residentNonExperts()                          // experts は mmap のまま（paged）
        let rssLoad = StreamingDecode.rssGB()
        let model = QwispModel(store: store)
        let ids = promptArr.asType(.int32).reshaped([1, promptArr.dim(0)])
        let gR = gRef.asArray(Int32.self).map { Int($0) }
        let N = Swift.min(Tell.envInt("QWISP_GEN", 64), gR.count)
        let caches = model.makeCaches()

        // prefill（resident gather なので arena overflow 無し, chunk 不要）
        var lg = model.callAsFunction(ids, caches: caches)
        var cur = MLX.argMax(lg[0, lg.dim(1) - 1], axis: -1).reshaped([1, 1])
        MLX.eval([cur] + caches.flatMap { $0.stateArrays })
        var out: [Int] = []
        // warmup 1 token（初回 page-in を計時から除外）
        lg = model.callAsFunction(cur, caches: caches)
        var nxt = MLX.argMax(lg[0, 0], axis: -1).reshaped([1, 1]); MLX.eval([nxt] + caches.flatMap { $0.stateArrays })
        out.append(cur.item(Int.self)); cur = nxt

        let t0 = DispatchTime.now()
        for _ in 1 ..< N {
            out.append(cur.item(Int.self))
            lg = model.callAsFunction(cur, caches: caches)
            cur = MLX.argMax(lg[0, 0], axis: -1).reshaped([1, 1])
            MLX.eval([cur] + caches.flatMap { $0.stateArrays })
        }
        let secs = Double(DispatchTime.now().uptimeNanoseconds - t0.uptimeNanoseconds) / 1e9
        let rssPeak = StreamingDecode.rssGB()
        let match = zip(out, gR).filter { $0 == $1 }.count
        return String(format: """
            [MmapGather] mmap全expert + resident gather(sync無): %.1f tok/s  品質 %d/%d=%.0f%%  RSS load=%.1f peak=%.1fGB
            """, Double(N - 1) / secs, match, N, Double(match) / Double(N) * 100, rssLoad, rssPeak)
    }

    /// **[計測] layer-skip 可能性の calib**
    /// - 機構: 各層を skip しても出力が保たれるか（draft 軽量化の余地）を測る
    /// - 結果例: skippable code 3/40 ・ nl 37/40（prompt 依存）
    /// - env: QWISP_SWIFT_CALIB
    /// - 旧名: SwiftCalib / runSwiftCalib（git ba3c138）
    public static func measureSkippability(modelDir: String, refPath: String) throws -> String {
        guard let device = MTLCreateSystemDefaultDevice() else { return "ERROR: no Metal device" }
        let r = try loadArrays(url: URL(fileURLWithPath: refPath))
        guard let promptArr = r["spec_prompt"], let gRef = r["spec_greedy"] else { return "[Skippability] skip" }
        let store = try WeightStore(modelDir: modelDir)
        store.residentNonExperts()
        let source = try ExpertSource(modelDir: modelDir); try source.warm()
        let arena = try ExpertArena(device: device, source: source, N: 64)
        let model = try StreamingQwispModel(store: store, arena: arena, device: device, source: source, cacheC: 64)
        let ids = promptArr.asType(.int32).reshaped([1, promptArr.dim(0)])
        let gR = gRef.asArray(Int32.self).map { Int($0) }
        let isLin = model.isLinearFlags
        let L = model.layerCount
        let N = Swift.min(Tell.envInt("QWISP_GEN", 16), gR.count)
        let mc = model.makeCaches()

        StreamingMoEBlock.probeNoSync = false
        var (_, lg) = try model.prefillChunked(ids, caches: mc)
        var cur = MLX.argMax(lg[0, lg.dim(1) - 1], axis: -1).reshaped([1, 1])
        MLX.eval([cur] + mc.flatMap { $0.stateArrays })
        var matchL = [Int](repeating: 0, count: L)
        for _ in 0 ..< N {
            // 各層単独 skip の argmax（committed state から, snapshot→skip-L forward→restore）
            var skipArg = [Int](repeating: -1, count: L)
            for layer in 0 ..< L {
                let snaps = mc.map { $0.snapshot() }
                let (_, sl) = try model.forwardHiddenSkip(cur, caches: mc, skip: [layer])
                skipArg[layer] = MLX.argMax(sl[0, 0], axis: -1).item(Int.self)
                for (i, c) in mc.enumerated() where i != layer { c.restore(snaps[i], isLinear: isLin[i], trim: 1) }
            }
            // full forward（commit）
            let (_, fl) = try model.forwardHidden(cur, caches: mc)
            let full = MLX.argMax(fl[0, 0], axis: -1).item(Int.self)
            for layer in 0 ..< L where skipArg[layer] == full { matchL[layer] += 1 }
            cur = MLXArray([Int32(full)], [1, 1]); MLX.eval([cur] + mc.flatMap { $0.stateArrays })
        }
        StreamingMoEBlock.probeNoSync = false

        // matchness 降順で「skip しても出力を保つ」層を列挙
        let ranked = (0 ..< L).map { (l: $0, m: Double(matchL[$0]) / Double(N), lin: isLin[$0]) }
            .sorted { $0.m > $1.m }
        let skippable = ranked.filter { $0.m >= 0.95 }
        let ginfo = skippable.map { "\($0.l)\($0.lin ? "G" : "A")" }.joined(separator: ",")
        let nG = skippable.filter { $0.lin }.count, nA = skippable.filter { !$0.lin }.count
        return String(format: """
            [Skippability] %d step, 各層単独 skip の matchness。skip可(matchness≥0.95)=%d/%d 層 (GDN %d/attn %d)
              skip可層(番号+G/A): %@
              最 skip 可 top8: %@
            """, N, skippable.count, L, nG, nA, ginfo,
            ranked.prefix(8).map { String(format: "%d%@=%.2f", $0.l, $0.lin ? "G" : "A", $0.m) }.joined(separator: " "))
    }

    /// **prompt 毎 probe で no-sync/exact 自動切替**
    /// - 機構: 短い probe で no-sync の安全性を判定—全一致なら no-sync(高速)、drift 兆候なら exact 経路(lossless)へ
    /// - lossless: probe 依存（probe が誤れば非保証）
    /// - env: QWISP_HOTCOLD_AUTO
    /// - 旧名: Auto / runHotColdAuto（git a54785a）
    public static func runProbeAuto(modelDir: String, refPath: String) throws -> String {
        guard let device = MTLCreateSystemDefaultDevice() else { return "ERROR: no Metal device" }
        let r = try loadArrays(url: URL(fileURLWithPath: refPath))
        guard let promptArr = r["spec_prompt"], let gRef = r["spec_greedy"] else { return "[ProbeAuto] skip" }
        let C = Tell.envInt("QWISP_CACHE_C", 64)
        let calibN = Tell.envInt("QWISP_CALIB", 32)
        let probeK = Tell.envInt("QWISP_PROBE", 8)
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

        // calib + hot pin
        StreamingMoEBlock.probeNoSync = false; StreamingMoEBlock.captureInds = true
        var (_, lg) = try model.prefillChunked(ids, caches: caches)
        var cur = MLX.argMax(lg[0, lg.dim(1) - 1], axis: -1).reshaped([1, 1])
        MLX.eval([cur] + caches.flatMap { $0.stateArrays })
        for _ in 0 ..< calibN {
            (_, lg) = try model.forwardHidden(cur, caches: caches)
            MLX.eval([lg] + caches.flatMap { $0.stateArrays } + model.expertCaches.compactMap { $0.lastInds })
            for (mi, ec) in model.expertCaches.enumerated() {
                if let li = ec.lastInds { for e in li.asArray(Int32.self) { counts[mi][Int(e)] += 1 } }
            }
            cur = MLX.argMax(lg[0, 0], axis: -1).reshaped([1, 1]); MLX.eval([cur])
        }
        StreamingMoEBlock.captureInds = false
        for (mi, ec) in model.expertCaches.enumerated() {
            _ = ec.ensure(Array(counts[mi].enumerated().sorted { $0.element > $1.element }.prefix(C).map { $0.offset }))
        }

        // fresh prefill → probe: 各 step で no-sync(side) vs exact(真値) を比較
        let caches2 = model.makeCaches()
        StreamingMoEBlock.probeNoSync = false
        (_, lg) = try model.prefillChunked(ids, caches: caches2)
        cur = MLX.argMax(lg[0, lg.dim(1) - 1], axis: -1).reshaped([1, 1])
        MLX.eval([cur] + caches2.flatMap { $0.stateArrays })
        var out: [Int] = []
        var probeMiss = 0
        let t0 = DispatchTime.now()
        for _ in 0 ..< probeK {
            out.append(cur.item(Int.self))
            // side: no-sync 予測（snapshot→no-sync forward→restore）
            let snaps = caches2.map { $0.snapshot() }
            StreamingMoEBlock.probeNoSync = true
            let (_, lgn) = try model.forwardHidden(cur, caches: caches2)
            let nosyncTok = MLX.argMax(lgn[0, 0], axis: -1).item(Int.self)
            for (i, c) in caches2.enumerated() { c.restore(snaps[i], isLinear: isLin[i], trim: 1) }
            // 真値: exact forward（advance）
            StreamingMoEBlock.probeNoSync = false
            (_, lg) = try model.forwardHidden(cur, caches: caches2)
            let exactTok = MLX.argMax(lg[0, 0], axis: -1).item(Int.self)
            if nosyncTok != exactTok { probeMiss += 1 }
            cur = MLX.argMax(lg[0, 0], axis: -1).reshaped([1, 1]); MLX.eval([cur] + caches2.flatMap { $0.stateArrays })
        }
        let easy = probeMiss == 0
        // 残りを選択 mode で decode
        StreamingMoEBlock.probeNoSync = easy
        for _ in probeK ..< N {
            out.append(cur.item(Int.self))
            (_, lg) = try model.forwardHidden(cur, caches: caches2)
            cur = MLX.argMax(lg[0, 0], axis: -1).reshaped([1, 1])
            MLX.eval([cur] + caches2.flatMap { $0.stateArrays })
        }
        StreamingMoEBlock.probeNoSync = false
        let secs = Double(DispatchTime.now().uptimeNanoseconds - t0.uptimeNanoseconds) / 1e9
        let match = zip(out, gR).filter { $0 == $1 }.count
        return String(format: """
            [ProbeAuto] hot top-%d, probe=%d miss=%d → mode=%@: %.1f tok/s  品質 %d/%d=%.0f%%
            """, C, probeK, probeMiss, easy ? "no-sync(47)" : "exact(lossless)",
            Double(N) / secs, match, N, Double(match) / Double(N) * 100)
    }

    /// **静的 per-layer adaptive sync（死路）**
    /// - 機構: 層ごとに sync 要否を静的に決め hard 層だけ正確化
    /// - lossless: ❌neg（per-token + 予測が必要、静的では不足）
    /// - env: QWISP_HOTCOLD_ADAPT
    /// - 旧名: Adaptive / runHotColdAdaptive（git 864be4e）
    public static func runAdaptiveSync(modelDir: String, refPath: String) throws -> String {
        guard let device = MTLCreateSystemDefaultDevice() else { return "ERROR: no Metal device" }
        let r = try loadArrays(url: URL(fileURLWithPath: refPath))
        guard let promptArr = r["spec_prompt"], let gRef = r["spec_greedy"] else { return "[AdaptiveSync] skip" }
        let C = Tell.envInt("QWISP_CACHE_C", 64)
        let calibN = Tell.envInt("QWISP_CALIB", 48)
        let theta = Double(Tell.envStr("QWISP_THETA", "0.995")) ?? 0.995
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
        var counts = [[Int]](repeating: [Int](repeating: 0, count: nE), count: nMoE)
        // hard 判定用: 各層 per-token の「routed が hot 外」を最悪ケースで検出するため、calib で
        // routed expert を記録し、最終 hot set に対する per-token 最悪 coverage を層別に出す。
        var perTokInds = [[[Int]]](repeating: [], count: nMoE)   // [layer][token][experts]

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
                    perTokInds[mi].append(es)
                }
            }
            cur = MLX.argMax(lg[0, 0], axis: -1).reshaped([1, 1]); MLX.eval([cur])
        }
        StreamingMoEBlock.captureInds = false

        // 各層 hot set(top-C) と per-token 最悪 coverage → θ 未満なら hard(sync)
        var syncSet = Set<Int>()
        for mi in 0 ..< nMoE {
            let hot = Set(counts[mi].enumerated().sorted { $0.element > $1.element }.prefix(C).map { $0.offset })
            for (li, ec) in model.expertCaches.enumerated() where li == mi { _ = ec.ensure(Array(hot)) }
            var worst = 1.0
            for es in perTokInds[mi] {
                let cov = Double(es.filter { hot.contains($0) }.count) / Double(es.count)
                worst = Swift.min(worst, cov)
            }
            if worst < theta { syncSet.insert(mi) }
        }
        // model.expertCaches の index は MoE 層の通し番号だが、layer フィールドは実層 index。
        // syncLayers は StreamingMoEBlock.layer（実層）で判定するので実層 index に変換。
        var syncReal = Set<Int>()
        for (mi, ec) in model.expertCaches.enumerated() where syncSet.contains(mi) { syncReal.insert(ec.layer) }

        // decode: fresh prefill → token 0、易層 no-sync / hard 層 exact
        let caches2 = model.makeCaches()
        StreamingMoEBlock.probeNoSync = false
        (_, lg) = try model.prefillChunked(ids, caches: caches2)
        cur = MLX.argMax(lg[0, lg.dim(1) - 1], axis: -1).reshaped([1, 1])
        MLX.eval([cur] + caches2.flatMap { $0.stateArrays })
        StreamingMoEBlock.probeNoSync = true
        StreamingMoEBlock.syncLayers = syncReal
        var out: [Int] = []
        let t0 = DispatchTime.now()
        for _ in 0 ..< N {
            out.append(cur.item(Int.self))
            (_, lg) = try model.forwardHidden(cur, caches: caches2)
            cur = MLX.argMax(lg[0, 0], axis: -1).reshaped([1, 1])
            MLX.eval([cur] + caches2.flatMap { $0.stateArrays })
        }
        StreamingMoEBlock.probeNoSync = false; StreamingMoEBlock.syncLayers = nil
        let secs = Double(DispatchTime.now().uptimeNanoseconds - t0.uptimeNanoseconds) / 1e9
        let match = zip(out, gR).filter { $0 == $1 }.count
        return String(format: """
            [AdaptiveSync] hot top-%d + 適応sync(θ=%.3f): sync層=%d/%d  %.1f tok/s  品質 %d/%d=%.0f%%
            """, C, theta, syncReal.count, nMoE, Double(N) / secs, match, N, Double(match) / Double(N) * 100)
    }

    /// **hot-set の online 更新**
    /// - 機構: decode 中に hot expert 集合を逐次更新し coverage を上げる
    /// - lossless: near（online coverage code 76.5%/nl 89.7%）
    /// - env: QWISP_HOTCOLD_ONLINE
    /// - 旧名: Online / runHotColdOnline（git a54785a）
    public static func runOnlineHotSet(modelDir: String, refPath: String) throws -> String {
        guard let device = MTLCreateSystemDefaultDevice() else { return "ERROR: no Metal device" }
        let r = try loadArrays(url: URL(fileURLWithPath: refPath))
        guard let promptArr = r["spec_prompt"], let gRef = r["spec_greedy"] else { return "[OnlineHotSet] skip" }
        let C = Tell.envInt("QWISP_CACHE_C", 64)
        let calibN = Tell.envInt("QWISP_CALIB", 16)
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
        var counts = [[Int]](repeating: [Int](repeating: 0, count: nE), count: nMoE)

        func topC(_ c: [Int]) -> [Int] {
            Array(c.enumerated().sorted { $0.element > $1.element }.prefix(C).map { $0.offset })
        }

        // warm-start calib（exact, 短く）
        StreamingMoEBlock.probeNoSync = false; StreamingMoEBlock.captureInds = true
        var (_, lg) = try model.prefillChunked(ids, caches: caches)
        var cur = MLX.argMax(lg[0, lg.dim(1) - 1], axis: -1).reshaped([1, 1])
        MLX.eval([cur] + caches.flatMap { $0.stateArrays })
        for _ in 0 ..< calibN {
            (_, lg) = try model.forwardHidden(cur, caches: caches)
            MLX.eval([lg] + caches.flatMap { $0.stateArrays } + model.expertCaches.compactMap { $0.lastInds })
            for (mi, ec) in model.expertCaches.enumerated() {
                if let li = ec.lastInds { for e in li.asArray(Int32.self) { counts[mi][Int(e)] += 1 } }
            }
            cur = MLX.argMax(lg[0, 0], axis: -1).reshaped([1, 1]); MLX.eval([cur])
        }

        // 本番: fresh prefill から token 0 を decode（calib の counts を warm start として継続更新）。
        let caches2 = model.makeCaches()
        StreamingMoEBlock.probeNoSync = false
        (_, lg) = try model.prefillChunked(ids, caches: caches2)
        cur = MLX.argMax(lg[0, lg.dim(1) - 1], axis: -1).reshaped([1, 1])
        MLX.eval([cur] + caches2.flatMap { $0.stateArrays })
        var out: [Int] = []
        let t0 = DispatchTime.now()
        for _ in 0 ..< N {
            for (mi, ec) in model.expertCaches.enumerated() { _ = ec.ensure(topC(counts[mi])) }
            out.append(cur.item(Int.self))
            StreamingMoEBlock.probeNoSync = true
            (_, lg) = try model.forwardHidden(cur, caches: caches2)
            MLX.eval([lg] + caches2.flatMap { $0.stateArrays } + model.expertCaches.compactMap { $0.lastInds })
            for (mi, ec) in model.expertCaches.enumerated() {
                if let li = ec.lastInds { for e in li.asArray(Int32.self) { counts[mi][Int(e)] += 1 } }
            }
            StreamingMoEBlock.probeNoSync = false
            cur = MLX.argMax(lg[0, 0], axis: -1).reshaped([1, 1]); MLX.eval([cur])
        }
        StreamingMoEBlock.captureInds = false
        let secs = Double(DispatchTime.now().uptimeNanoseconds - t0.uptimeNanoseconds) / 1e9
        let match = zip(out, gR).filter { $0 == $1 }.count
        return String(format: """
            [OnlineHotSet] online-adaptive hot top-%d + no-sync: %.1f tok/s  品質(greedy一致) %d/%d=%.0f%%
            """, C, Double(N) / secs, match, N, Double(match) / Double(N) * 100)
    }

    /// **[計測] exact 比の per-token miss / worst-layer**
    /// - 機構: static/online hot set の exact 比 coverage と worst-layer miss を診断
    /// - 結果例: code static 63.8/online 76.5%、worst layer code 0%/nl 12%
    /// - env: QWISP_HOTCOLD_DIAG
    /// - 旧名: HotColdDiag / runHotColdDiag（git 864be4e）
    public static func measureMissCoverage(modelDir: String, refPath: String) throws -> String {
        guard let device = MTLCreateSystemDefaultDevice() else { return "ERROR: no Metal device" }
        let r = try loadArrays(url: URL(fileURLWithPath: refPath))
        guard let promptArr = r["spec_prompt"], let gRef = r["spec_greedy"] else { return "[MissCoverage] skip" }
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

        func topSet(_ c: [Int], _ k: Int) -> Set<Int> {
            Set(c.enumerated().sorted { $0.element > $1.element }.prefix(k).map { $0.offset })
        }

        StreamingMoEBlock.probeNoSync = false; StreamingMoEBlock.captureInds = true
        var (_, lg) = try model.prefillChunked(ids, caches: caches)
        var cur = MLX.argMax(lg[0, lg.dim(1) - 1], axis: -1).reshaped([1, 1])
        MLX.eval([cur] + caches.flatMap { $0.stateArrays })

        // running counts（calib + online 共用）。calib 期間で warm start。
        var counts = [[Int]](repeating: [Int](repeating: 0, count: nE), count: nMoE)
        for _ in 0 ..< calibN {
            (_, lg) = try model.forwardHidden(cur, caches: caches)
            MLX.eval([lg] + caches.flatMap { $0.stateArrays } + model.expertCaches.compactMap { $0.lastInds })
            for (mi, ec) in model.expertCaches.enumerated() {
                if let li = ec.lastInds { for e in li.asArray(Int32.self) { counts[mi][Int(e)] += 1 } }
            }
            cur = MLX.argMax(lg[0, 0], axis: -1).reshaped([1, 1]); MLX.eval([cur])
        }
        // 静的 calib hot set を固定
        let staticHot = (0 ..< nMoE).map { topSet(counts[$0], C) }

        // eval 期間: 実 routing vs 静的 / オンライン coverage
        var hitStatic = 0, hitOnline = 0, totalRoute = 0
        // per-token の最悪層 coverage（drift 起点の指標）
        var worstTokMin = 100.0
        for _ in 0 ..< N {
            (_, lg) = try model.forwardHidden(cur, caches: caches)
            MLX.eval([lg] + caches.flatMap { $0.stateArrays } + model.expertCaches.compactMap { $0.lastInds })
            var tokWorst = 100.0
            for (mi, ec) in model.expertCaches.enumerated() {
                guard let li = ec.lastInds else { continue }
                let es = li.asArray(Int32.self).map { Int($0) }
                let onlineHot = topSet(counts[mi], C)   // token 前までの running top-C
                var hS = 0, hO = 0
                for e in es {
                    if staticHot[mi].contains(e) { hS += 1 }
                    if onlineHot.contains(e) { hO += 1 }
                }
                hitStatic += hS; hitOnline += hO; totalRoute += es.count
                tokWorst = Swift.min(tokWorst, Double(hO) / Double(es.count) * 100)
                for e in es { counts[mi][e] += 1 }   // online 更新（causal: 計測後）
            }
            worstTokMin = Swift.min(worstTokMin, tokWorst)
            cur = MLX.argMax(lg[0, 0], axis: -1).reshaped([1, 1]); MLX.eval([cur])
        }
        StreamingMoEBlock.captureInds = false
        return String(format: """
            [MissCoverage] C=%d calib=%d eval=%d。実 routing に対する hot-set coverage:
              静的 calib hot = %.1f%%   オンライン適応 hot = %.1f%%   (per-token 最悪層 online=%.0f%%)
            """, C, calibN, N,
            Double(hitStatic) / Double(totalRoute) * 100,
            Double(hitOnline) / Double(totalRoute) * 100, worstTokMin)
    }

    /// **no-sync draft + per-token ゲート escalate**
    /// - 機構: buddy no-sync draft を per-token 判定—membership(miss==0 で採用=strict)/margin(top1-top2 で採用=near)—外れは exact 1-forward へ escalate
    /// - lossless: membership=**strict**(但し escalate 多発で遅い) / margin=**near**(long 発散)
    /// - 速度: membership 8GB ~15-17 / margin ~36-49 tok/s
    /// - 研究: SS-MoE; AdapMoE; Draft&Verify (Zhang 2023)
    /// - env: QWISP_HYBRID / QWISP_MARGIN(0=membership) / QWISP_PIN / QWISP_PARTIAL / QWISP_PREFETCH
    /// - 旧名: Hybrid / runHotColdHybrid（git a3f02c4, fdbbd11, ddc14e2）
    public static func runNoSyncGateEscalate(modelDir: String, refPath: String) throws -> String {
        guard let device = MTLCreateSystemDefaultDevice() else { return "ERROR: no Metal device" }
        let r = try loadArrays(url: URL(fileURLWithPath: refPath))
        guard let promptArr = r["spec_prompt"], let gRef = r["spec_greedy"] else { return "[NoSyncGateEscalate] skip" }
        let C = Tell.envInt("QWISP_CACHE_C", 64)
        // pinned hot 数。escalate 時の cold ロード用に LRU 枠を最低 8 残す（pin=C だと ensure 不能）。
        let nPin = Swift.min(Tell.envInt("QWISP_PIN", 48),
                             Swift.max(C - 8, 1))
        let calibN = Tell.envInt("QWISP_CALIB", 48)
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
        let buddy = ProcessInfo.processInfo.environment["QWISP_SKIPMODE"] == "3"   // no-sync draft を buddy 代替
        let caches = model.makeCaches()

        // phase 1: calib（exact decode で頻度集計 + buddy 用 co-activation）
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
            cur = MLX.argMax(lg[0, 0], axis: -1).reshaped([1, 1])
            MLX.eval([cur] + caches.flatMap { $0.stateArrays })
        }
        StreamingMoEBlock.captureInds = false

        // Swift-exact-greedy 参照（strict-vs-near 判定用。membership は構成上これと 100% 一致するはず、
        // margin は hard ref で <100% なら near である決定的証拠）。QWISP_SWIFT_REF=1。
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

        // phase 2: top-nPin hot を各層 pin（残り C-nPin は escalate 時の cold LRU 枠）+ buddy table
        for (mi, ec) in model.expertCaches.enumerated() {
            let hot = Array(counts[mi].enumerated().sorted { $0.element > $1.element }.prefix(nPin).map { $0.offset })
            ec.pin(hot)
            if buddy { ec.buildBuddyTable(coact: coact[mi], numExperts: nE) }
        }
        // buddy: no-sync draft 中のみ skipMode=3（escalate は probeNoSync=false で無影響）。
        StreamingMoEBlock.skipMode = buddy ? 3 : 0
        defer { StreamingMoEBlock.skipMode = 0 }

        // phase 3: hybrid decode（fresh prefill）
        let caches2 = model.makeCaches()
        StreamingMoEBlock.probeNoSync = false
        (_, lg) = try model.prefillChunked(ids, caches: caches2)
        cur = MLX.argMax(lg[0, lg.dim(1) - 1], axis: -1).reshaped([1, 1])
        MLX.eval([cur] + caches2.flatMap { $0.stateArrays })
        // accept gate: marginThresh>0 なら logit-margin gate（near-lossless, 確信高い no-sync を採用）、
        // ==0 なら membership gate（all-hot のみ採用＝厳密 lossless だが strict）。
        let marginThresh = Tell.envFloat("QWISP_MARGIN", 0)
        let useMargin = marginThresh > 0
        // partial-resume: escalate 時、no-sync draft の最初の miss 層 k を特定し、層 0..k-1 の計算を
        // 流用して層 k から exact tail だけ再走（厳密 lossless, escalate コストを k に比例して削減）。
        let partial = Tell.envFlag("QWISP_PARTIAL")
        // prefetch-verify(Q3b expert 予測常駐): 各 token no-sync draft で全層 inds 取得 → draft inds を
        // 一括 prefetch 常駐 → no-sync verify（resident）→ residual hotMiss を計数。draft 予測の
        // 取りこぼし(後段層で draft の corrupted-hidden 由来 inds がズレる分)を実測。
        let prefetchVerify = Tell.envFlag("QWISP_PREFETCH")
        let L = model.layerCount
        var out: [Int] = []
        var escalations = 0
        var firstMissSum = 0
        if prefetchVerify {
            StreamingMoEBlock.probeNoSync = true; StreamingMoEBlock.captureInds = true
            var residualSum = 0
            let t0 = DispatchTime.now()
            for _ in 0 ..< N {
                out.append(cur.item(Int.self))
                let snaps = caches2.map { $0.snapshot() }
                // pass-1: no-sync draft → 全層 draft inds（GPU, per層 sync 無し）
                StreamingMoEBlock.countHotMiss = false
                _ = try model.forwardHidden(cur, caches: caches2)
                MLX.eval(caches2.flatMap { $0.stateArrays } + model.expertCaches.compactMap { $0.lastInds })
                // prefetch: 各層 draft distinct inds を常駐化（pinned 保持で LRU 枠へ pread）
                for ec in model.expertCaches { ec.prefetchLastInds() }
                // restore（draft の cache 前進を巻き戻す）→ pass-2 no-sync verify（resident）
                for (i, c) in caches2.enumerated() { c.restore(snaps[i], isLinear: isLin[i], trim: 1) }
                StreamingMoEBlock.countHotMiss = true; StreamingMoEBlock.hotMissAccum = nil
                let (_, lg2) = try model.forwardHidden(cur, caches: caches2)
                let vTok = MLX.argMax(lg2[0, 0], axis: -1).reshaped([1, 1])
                let missArr = StreamingMoEBlock.hotMissAccum ?? MLXArray(Int32(0))
                MLX.eval([vTok, missArr] + caches2.flatMap { $0.stateArrays })
                let rm = missArr.item(Int.self)
                residualSum += rm; if rm > 0 { escalations += 1 }
                cur = vTok
            }
            StreamingMoEBlock.probeNoSync = false; StreamingMoEBlock.captureInds = false
            StreamingMoEBlock.countHotMiss = false; StreamingMoEBlock.hotMissAccum = nil
            let secs = Double(DispatchTime.now().uptimeNanoseconds - t0.uptimeNanoseconds) / 1e9
            let match = zip(out, gR).filter { $0 == $1 }.count
            return String(format: """
                [NoSyncGateEscalate] pin=%d/C=%d gate=prefetch-verify: %.1f tok/s  品質(greedy一致) %d/%d=%.0f%%  residual-miss平均=%.1f/tok(>0:%d/%d tok)
                """, nPin, C, Double(N) / secs, match, N, Double(match) / Double(N) * 100,
                Double(residualSum) / Double(N), escalations, N)
        }
        if partial {
            StreamingMoEBlock.probeNoSync = true; StreamingMoEBlock.countHotMiss = false
            StreamingMoEBlock.captureInds = true
            let t0 = DispatchTime.now()
            for _ in 0 ..< N {
                out.append(cur.item(Int.self))
                let snaps = caches2.map { $0.snapshot() }
                // pass-1: no-sync draft を層ごとに走らせ、各層 input hidden を保存
                var hsave: [MLXArray] = []; hsave.reserveCapacity(L)
                var h = model.embedPub(cur)
                for i in 0 ..< L { hsave.append(h); h = try model.runChunk(h, i, i + 1, caches: caches2) }
                let draftTok = MLX.argMax(model.finalLogits(h)[0, 0], axis: -1).reshaped([1, 1])
                // 一括 eval（draftTok + 全 hsave + 全 lastInds + states）= 1 sync で materialize
                MLX.eval([draftTok] + hsave + caches2.flatMap { $0.stateArrays }
                         + model.expertCaches.compactMap { $0.lastInds })
                // 最初の miss 層 k（MoE 層のうち lastInds が cache 外を含む最小 layer index）
                var firstMiss = L
                for ec in model.expertCaches where !ec.indsHot() { firstMiss = Swift.min(firstMiss, ec.layer) }
                if firstMiss == L {
                    cur = draftTok                                               // all-hot＝draft 採用
                } else {
                    escalations += 1; firstMissSum += firstMiss
                    // cache[firstMiss..] を token 先頭へ部分 restore（[0..firstMiss-1] は exact なので保持）
                    for i in firstMiss ..< L { caches2[i].restore(snaps[i], isLinear: isLin[i], trim: 1) }
                    StreamingMoEBlock.probeNoSync = false                        // exact tail
                    let h2 = try model.runChunk(hsave[firstMiss], firstMiss, L, caches: caches2)
                    cur = MLX.argMax(model.finalLogits(h2)[0, 0], axis: -1).reshaped([1, 1])
                    MLX.eval([cur] + caches2.flatMap { $0.stateArrays })
                    StreamingMoEBlock.probeNoSync = true
                }
            }
            StreamingMoEBlock.probeNoSync = false; StreamingMoEBlock.captureInds = false
            let secs = Double(DispatchTime.now().uptimeNanoseconds - t0.uptimeNanoseconds) / 1e9
            let match = zip(out, gR).filter { $0 == $1 }.count
            let avgK = escalations > 0 ? Double(firstMissSum) / Double(escalations) : 0
            return String(format: """
                [NoSyncGateEscalate] pin=%d/C=%d gate=partial-resume: %.1f tok/s  品質(greedy一致) %d/%d=%.0f%%  escalate=%d/%d(%.0f%%) 平均first-miss層=%.1f/%d
                """, nPin, C, Double(N) / secs, match, N, Double(match) / Double(N) * 100,
                escalations, N, Double(escalations) / Double(N) * 100, avgK, L)
        }
        StreamingMoEBlock.probeNoSync = true; StreamingMoEBlock.countHotMiss = !useMargin
        let t0 = DispatchTime.now()
        for _ in 0 ..< N {
            out.append(cur.item(Int.self))
            let snaps = caches2.map { $0.snapshot() }
            StreamingMoEBlock.hotMissAccum = nil
            let (_, lgn) = try model.forwardHidden(cur, caches: caches2)      // no-sync draft
            let v = lgn[0, 0]
            let nosyncTok = MLX.argMax(v, axis: -1).reshaped([1, 1])
            let accept: Bool
            if useMargin {
                // no-sync 出力の top1-top2 margin。大＝近似が argmax を反転しない確信が高い→採用。
                let sv = MLX.sorted(v, axis: -1); let n = sv.dim(0)
                let marginArr = sv[n - 1] - sv[n - 2]
                MLX.eval([nosyncTok, marginArr] + caches2.flatMap { $0.stateArrays })
                accept = marginArr.item(Float.self) >= marginThresh
            } else {
                let missArr = StreamingMoEBlock.hotMissAccum ?? MLXArray(Int32(0))
                MLX.eval([nosyncTok, missArr] + caches2.flatMap { $0.stateArrays })
                accept = missArr.item(Int.self) == 0                         // all-hot＝exact と bit 一致
            }
            if accept {
                cur = nosyncTok
            } else {
                escalations += 1
                for (i, c) in caches2.enumerated() { c.restore(snaps[i], isLinear: isLin[i], trim: 1) }
                StreamingMoEBlock.probeNoSync = false; StreamingMoEBlock.countHotMiss = false
                (_, lg) = try model.forwardHidden(cur, caches: caches2)      // exact 1-forward（ensure→LRU枠）
                cur = MLX.argMax(lg[0, 0], axis: -1).reshaped([1, 1])
                MLX.eval([cur] + caches2.flatMap { $0.stateArrays })
                StreamingMoEBlock.probeNoSync = true; StreamingMoEBlock.countHotMiss = !useMargin
            }
        }
        StreamingMoEBlock.probeNoSync = false; StreamingMoEBlock.countHotMiss = false
        StreamingMoEBlock.hotMissAccum = nil
        let secs = Double(DispatchTime.now().uptimeNanoseconds - t0.uptimeNanoseconds) / 1e9
        let match = zip(out, gR).filter { $0 == $1 }.count
        let swiftTag = gSwift.isEmpty ? ""
            : String(format: "  [vs Swift-greedy %d/%d=%.0f%%]",
                     zip(out, gSwift).filter { $0 == $1 }.count, N,
                     Double(zip(out, gSwift).filter { $0 == $1 }.count) / Double(N) * 100)
        return String(format: """
            [NoSyncGateEscalate] pin=%d/C=%d gate=%@: %.1f tok/s  品質(vs Python) %d/%d=%.0f%%%@  escalate=%d/%d(%.0f%%)
            """, nPin, C, useMargin ? String(format: "margin≥%.1f", marginThresh) : "membership",
            Double(N) / secs, match, N, Double(match) / Double(N) * 100, swiftTag,
            escalations, N, Double(escalations) / Double(N) * 100)
    }

    /// 各層の gate 入力の「最後の位置」だけ取る（chunk-0 予測の thrash 回避）。
    static func lastGate(_ model: StreamingQwispModel) -> [MLXArray] {
        // gate 入力は 2D [S, H]（MoE は [B*S, H] を受ける）。seq 軸=axis0 の最終行のみ取る。
        model.expertCaches.map { let g = $0.lastGateInput!; return g[(g.dim(0) - 1)...] }
    }

    static func distinctInts(_ a: MLXArray) -> [Int] {
        var seen = Set<Int>(); var u: [Int] = []
        for e in a.asArray(Int32.self) { let i = Int(e); if seen.insert(i).inserted { u.append(i) } }
        return u
    }

    /// Tell の chunked cross-layer prefetch forward（M2 の中核を再利用可能化）。
    /// prevGate(前 forward の各層 gate 入力)で chunk-0 を bootstrap、以降は前 chunk 最終 gate 入力で予測。
    /// 返り値: (post-norm hidden, logits)。captureGateInput により各層 gate 入力が cache に残る。
    /// exact=true で全 chunk を sync gather（spec 検証を lossless 化）。
    /// exact=false は cross-layer 予測（高速だが複数 token 検証では look-ahead 位置 v の
    /// GDN 再帰状態が drift し spec から乖離 → lossless でない。単一 token のみ安全）。
    static func tellForward(_ model: StreamingQwispModel, _ ids: MLXArray, _ caches: [LayerCache],
                            _ prevGate: [MLXArray]?, _ CH: Int, exact: Bool = false) throws -> (MLXArray, MLXArray) {
        StreamingMoEBlock.captureGateInput = true
        let L = model.layerCount
        var h = model.embedPub(ids)
        var pos = 0
        while pos < L {
            let end = Swift.min(pos + CH, L)
            if exact {
                StreamingMoEBlock.probeNoSync = false
                h = try model.runChunk(h, pos, end, caches: caches)
                pos = end
                continue
            }
            StreamingMoEBlock.probeNoSync = true
            let preds: [MLXArray]
            if pos > 0 {
                let src = model.expertCaches[pos - 1].lastGateInput!
                preds = (pos ..< end).map { model.predictLayerInds($0, src) }
            } else if let pg = prevGate {
                preds = (pos ..< end).map { model.predictLayerInds($0, pg[$0]) }
            } else {
                preds = (pos ..< end).map { model.predictLayerInds($0, h) }
            }
            MLX.eval(preds)
            for (k, i) in (pos ..< end).enumerated() { _ = model.expertCaches[i].ensure(distinctInts(preds[k])) }
            h = try model.runChunk(h, pos, end, caches: caches)
            pos = end
        }
        let normed = model.finalNorm(h)            // post-norm hidden（MTP の h_prev 用）
        return (normed, model.logitsFromNorm(normed))
    }

    /// **[計測] エンジンの MLX 忠実度（teacher-forced）**
    /// - 機構: reference gR を強制入力し mlx-swift exact の per-token argmax を mlx_lm と比較（自己回帰カオスを除去）
    /// - 結果: hard 100% / long 98.4%（不一致は f16 near-tie のみ）＝エンジンは MLX 準拠
    /// - env: QWISP_TFORCE
    /// - 旧名: TeacherForced / runTeacherForced（git 2516e30）。詳細 notes/00
    public static func measureMLXFidelity(modelDir: String, refPath: String) throws -> String {
        guard let device = MTLCreateSystemDefaultDevice() else { return "ERROR: no Metal device" }
        let r = try loadArrays(url: URL(fileURLWithPath: refPath))
        guard let promptArr = r["spec_prompt"], let gRef = r["spec_greedy"] else { return "[MLXFidelity] skip" }
        let C = Tell.envInt("QWISP_CACHE_C", 64)
        let store = try WeightStore(modelDir: modelDir)
        store.residentNonExperts()
        let source = try ExpertSource(modelDir: modelDir); try source.warm()
        let arena = try ExpertArena(device: device, source: source, N: 64)
        let model = try StreamingQwispModel(store: store, arena: arena, device: device, source: source, cacheC: C)
        let ids = promptArr.asType(.int32).reshaped([1, promptArr.dim(0)])
        let gR = gRef.asArray(Int32.self).map { Int($0) }
        let N = Swift.min(Tell.envInt("QWISP_GEN", 128), gR.count)
        let caches = model.makeCaches()
        StreamingMoEBlock.probeNoSync = false   // exact gather（demand-load）

        // prefill → 位置0の予測（gR[0] のはず）
        var (_, lg) = try model.prefillChunked(ids, caches: caches)
        var v = lg[0, lg.dim(1) - 1]
        var preds: [Int] = []
        var mism: [(pos: Int, pred: Int, ref: Int, refRank: Int, gap: Float)] = []
        func record(_ i: Int, _ v: MLXArray) {
            let pred = MLX.argMax(v, axis: -1).item(Int.self)
            preds.append(pred)
            if pred != gR[i] {
                // gR[i] の logit 順位（降順で何番目か）と top1 との logit 差を診断
                let sv = MLX.sorted(v, axis: -1); let n = sv.dim(0)
                let top1 = sv[n - 1].item(Float.self)
                let refLogit = v[gR[i]].item(Float.self)
                let rank = (MLX.sum(v .> refLogit).item(Int.self))   // 自分より大きい logit の数 = 0-indexed rank
                mism.append((i, pred, gR[i], rank, top1 - refLogit))
            }
        }
        record(0, v)
        MLX.eval([v] + caches.flatMap { $0.stateArrays })
        for i in 0 ..< (N - 1) {
            let inp = MLXArray([Int32(gR[i])], [1, 1])   // teacher-forced: reference token を強制入力
            (_, lg) = try model.forwardHidden(inp, caches: caches)
            v = lg[0, 0]
            record(i + 1, v)
            MLX.eval([v] + caches.flatMap { $0.stateArrays })
        }
        let match = zip(preds, gR).filter { $0 == $1 }.count
        // mismatch のうち「gR が rank-1（僅差で2位）」= f16 near-tie 反転、「rank≫1」= 真の乖離
        let nearTie = mism.filter { $0.refRank <= 2 }.count
        var diag = mism.prefix(8).map { String(format: "p%d:pred=%d ref=%d(rank%d,gap%.3f)", $0.pos, $0.pred, $0.ref, $0.refRank, $0.gap) }.joined(separator: " | ")
        if diag.isEmpty { diag = "(no mismatch)" }
        return String(format: """
            [MLXFidelity] mlx-swift exact vs mlx_lm(gR) per-token: %d/%d=%.1f%%  mismatch=%d (near-tie rank≤2: %d)
              first mismatches: %@
            """, match, N, Double(match) / Double(N) * 100, mism.count, nearTie, diag)
    }

    /// **2-pass 自己予測 prefetch decode**
    /// - 機構: pass1 no-sync で routing を自己予測→該当 expert を prefetch→pass2 を chunk overlap 実行（IO を計算裏に隠す）
    /// - lossless: **near**（pass1 が corrupted hidden 由来で recall<100%、long で発散 vs Swift-exact ~11%）
    /// - 速度: 8GB C=64 ~28-30 tok/s
    /// - 研究: Fate one-pass; Pre-gated MoE (Hwang 2024); MoE-SpeQ; SP-MoE
    /// - env: QWISP_RUN_M0 / QWISP_M0_SELK・QWISP_M0_TAU(選択的margin) / QWISP_M0_TOPK(neg)
    /// - 旧名: M0 / runM0（git ac73c33, 1b770e4）
    public static func runPredictPrefetch(modelDir: String, refPath: String) throws -> String {
        guard let device = MTLCreateSystemDefaultDevice() else { return "ERROR: no Metal device" }
        let r = try loadArrays(url: URL(fileURLWithPath: refPath))
        guard let promptArr = r["spec_prompt"], let gRef = r["spec_greedy"] else { return "[Tell] skip" }
        let C = Tell.envInt("QWISP_CACHE_C", 64)
        let CH = Tell.envInt("QWISP_CHUNK", 10)
        // prefetch margin: pass-1 で top-K を捕捉（pass-2 の miss を減らし strict-lossless 化）
        StreamingMoEBlock.captureK = Tell.envInt("QWISP_M0_TOPK", 0)
        defer { StreamingMoEBlock.captureK = 0 }
        // 選択的マージン: 不確実層(top-K mass < τ)だけ top-marginK を prefetch（一律 top-K の cache 圧迫を回避）
        let selK = Tell.envInt("QWISP_M0_SELK", 0)
        let selTau = Tell.envFloat("QWISP_M0_TAU", 0.6)
        StreamingMoEBlock.marginK = selK
        defer { StreamingMoEBlock.marginK = 0 }
        var widenTotal = 0, widenTokens = 0   // 診断: 拡張した層数の累積
        let store = try WeightStore(modelDir: modelDir)
        store.residentNonExperts()
        let source = try ExpertSource(modelDir: modelDir); try source.warm()
        let arena = try ExpertArena(device: device, source: source, N: 64)
        let model = try StreamingQwispModel(store: store, arena: arena, device: device,
                                            source: source, cacheC: C)
        let ids = promptArr.asType(.int32).reshaped([1, promptArr.dim(0)])
        let gR = gRef.asArray(Int32.self).map { Int($0) }
        let isLin = model.isLinearFlags
        let L = model.layerCount
        let N = Swift.min(Tell.envInt("QWISP_GEN", 48), gR.count)
        let caches = model.makeCaches()

        StreamingMoEBlock.probeNoSync = false
        var (_, lg) = try model.prefillChunked(ids, caches: caches)
        var cur = MLX.argMax(lg[0, lg.dim(1) - 1], axis: -1).reshaped([1, 1])
        MLX.eval([cur] + caches.flatMap { $0.stateArrays })

        // Swift-exact-greedy 参照（M0 は構成上 strict lossless なので 100% のはず）。QWISP_SWIFT_REF=1。
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

        // 各層の予測 distinct experts を抽出
        func distinct(_ a: MLXArray) -> [Int] {
            var seen = Set<Int>(); var u: [Int] = []
            for e in a.asArray(Int32.self) { let i = Int(e); if seen.insert(i).inserted { u.append(i) } }
            return u
        }
        func prefetch(_ lo: Int, _ hi: Int, _ pred: [[Int]]) {
            for i in lo ..< hi { _ = model.expertCaches[i].ensure(pred[i]) }
        }

        let prof = Tell.envFlag("QWISP_M0_PROF")
        var tP1: UInt64 = 0, tPred: UInt64 = 0, tBuild: UInt64 = 0, tWait: UInt64 = 0, tFinal: UInt64 = 0
        func now() -> UInt64 { DispatchTime.now().uptimeNanoseconds }

        var out: [Int] = []
        let t0 = DispatchTime.now()
        for _ in 0 ..< N {
            let prev = cur
            let snaps = caches.map { $0.snapshot() }
            // pass-1: 予測（full GPU-remap）
            var ts = now()
            StreamingMoEBlock.probeNoSync = true
            _ = try model.forwardHidden(prev, caches: caches)
            MLX.eval(caches.flatMap { $0.stateArrays } + model.expertCaches.compactMap { $0.lastInds }
                     + model.expertCaches.compactMap { $0.lastMarginInds }
                     + model.expertCaches.compactMap { $0.lastConf })
            if prof { tP1 += now() - ts; ts = now() }
            let pred: [[Int]]
            if selK > 8 {
                // 選択的マージン: 確信度 < τ の層は marginK 候補、それ以外は top-8 を prefetch
                widenTokens += 1
                pred = model.expertCaches.map { ec -> [Int] in
                    let conf = ec.lastConf.map { $0.min().item(Float.self) } ?? 1.0
                    if conf < selTau, let m = ec.lastMarginInds { widenTotal += 1; return distinct(m) }
                    return distinct(ec.lastInds ?? MLXArray([Int32]()))
                }
            } else {
                pred = model.expertCaches.map { distinct($0.lastInds ?? MLXArray([Int32]())) }
            }
            for (i, c) in caches.enumerated() { c.restore(snaps[i], isLinear: isLin[i], trim: 1) }
            if prof { tPred += now() - ts }

            // pass-2: chunked + overlapped prefetch（chunk N の eval 中に chunk N+1 を先読み）
            StreamingMoEBlock.probeNoSync = true
            var h = model.embedPub(prev)
            prefetch(0, Swift.min(CH, L), pred)              // chunk 0 を同期 prefetch
            var pos = 0
            while pos < L {
                let end = Swift.min(pos + CH, L)
                let nLo = end, nHi = Swift.min(end + CH, L)
                let sem = DispatchSemaphore(value: 0)
                if nLo < L { DispatchQueue.global(qos: .userInitiated).async { prefetch(nLo, nHi, pred); sem.signal() } }
                ts = now()
                h = try model.runChunk(h, pos, end, caches: caches)   // この chunk の graph build
                MLX.asyncEval([h] + caches[pos ..< end].flatMap { $0.stateArrays })  // 非同期実行
                if prof { tBuild += now() - ts; ts = now() }
                if nLo < L { sem.wait() }                            // 次 chunk prefetch 完了を待つ
                if prof { tWait += now() - ts }
                pos = end
            }
            ts = now()
            let logits = model.finalLogits(h)
            let next = MLX.argMax(logits[0, 0], axis: -1).reshaped([1, 1])
            MLX.eval([next] + caches.flatMap { $0.stateArrays })
            if prof { tFinal += now() - ts }
            out.append(prev.item(Int.self)); cur = next
        }
        StreamingMoEBlock.probeNoSync = false
        let secs = Double(DispatchTime.now().uptimeNanoseconds - t0.uptimeNanoseconds) / 1e9
        let match = zip(out, gR).filter { $0 == $1 }.count
        if prof {
            let n = Double(N)
            FileHandle.standardError.write(String(format:
                "[PredictPrefetch-PROF/tok] pass1=%.1f pred-readback=%.1f pass2-build=%.1f prefetch-wait=%.1f final-drain=%.1f (ms)\n",
                Double(tP1)/n/1e6, Double(tPred)/n/1e6, Double(tBuild)/n/1e6,
                Double(tWait)/n/1e6, Double(tFinal)/n/1e6).data(using: .utf8)!)
        }
        let selDiag = selK > 8
            ? String(format: "  [selK=%d τ=%.2f 拡張層=%.1f/%d層/tok]",
                     selK, selTau, Double(widenTotal) / Double(Swift.max(widenTokens, 1)), L)
            : ""
        let swiftTag = gSwift.isEmpty ? ""
            : String(format: "  [vs Swift-greedy %d/%d=%.0f%%]",
                     zip(out, gSwift).filter { $0 == $1 }.count, N,
                     Double(zip(out, gSwift).filter { $0 == $1 }.count) / Double(N) * 100)
        return String(format: """
            [PredictPrefetch] chunk overlap 2-pass(C=%d, chunk=%d): %.1f tok/s  品質(vs Python) %d/%d=%.0f%%%@%@
            """,
            C, CH, Double(N) / secs, match, N, Double(match) / Double(N) * 100, swiftTag, selDiag)
    }

    /// **routing 不動点 multipass decode（死路）**
    /// - 機構: cache と routing が一致するまで forward を反復（pass 間不整合を消し strict 狙い）
    /// - lossless: strict だが ❌**~7 tok/s**（avg 5.4 pass）。M0 の 1-pass 最適性を補強
    /// - env: QWISP_M6
    /// - 旧名: M6 / runM0Multi
    public static func runPredictFixpoint(modelDir: String, refPath: String) throws -> String {
        guard let device = MTLCreateSystemDefaultDevice() else { return "ERROR: no Metal device" }
        let r = try loadArrays(url: URL(fileURLWithPath: refPath))
        guard let promptArr = r["spec_prompt"], let gRef = r["spec_greedy"] else { return "[PredictFixpoint] skip" }
        let C = Tell.envInt("QWISP_CACHE_C", 64)
        let maxP = Tell.envInt("QWISP_MAXPASS", 4)
        let store = try WeightStore(modelDir: modelDir)
        store.residentNonExperts()
        let source = try ExpertSource(modelDir: modelDir); try source.warm()
        let arena = try ExpertArena(device: device, source: source, N: 64)
        let model = try StreamingQwispModel(store: store, arena: arena, device: device, source: source, cacheC: C)
        let ids = promptArr.asType(.int32).reshaped([1, promptArr.dim(0)])
        let gR = gRef.asArray(Int32.self).map { Int($0) }
        let isLin = model.isLinearFlags
        let L = model.layerCount
        let N = Swift.min(Tell.envInt("QWISP_GEN", 64), gR.count)
        let caches = model.makeCaches()
        func distinct(_ a: MLXArray) -> Set<Int> { Set(a.asArray(Int32.self).map { Int($0) }) }

        StreamingMoEBlock.probeNoSync = false
        var (_, lg) = try model.prefillChunked(ids, caches: caches)
        var cur = MLX.argMax(lg[0, lg.dim(1) - 1], axis: -1).reshaped([1, 1])
        MLX.eval([cur] + caches.flatMap { $0.stateArrays })
        var out: [Int] = []; var passTotal = 0
        let t0 = DispatchTime.now()
        for _ in 0 ..< N {
            let prev = cur
            let snaps = caches.map { $0.snapshot() }
            var prevR: [Set<Int>]? = nil
            var lastLg = lg
            StreamingMoEBlock.probeNoSync = true
            for pass in 0 ..< maxP {
                passTotal += 1
                if pass > 0 { for (i, c) in caches.enumerated() { c.restore(snaps[i], isLinear: isLin[i], trim: 1) } }
                let (_, lgt) = try model.forwardHidden(prev, caches: caches)
                MLX.eval([lgt] + caches.flatMap { $0.stateArrays } + model.expertCaches.compactMap { $0.lastInds })
                lastLg = lgt
                let R = model.expertCaches.map { distinct($0.lastInds ?? MLXArray([Int32]())) }
                for (i, ec) in model.expertCaches.enumerated() { _ = ec.ensure(Array(R[i])) }   // prefetch
                if let pr = prevR, zip(pr, R).allSatisfy({ $0 == $1 }) { break }   // 収束
                prevR = R
            }
            StreamingMoEBlock.probeNoSync = false
            let next = MLX.argMax(lastLg[0, 0], axis: -1).reshaped([1, 1])
            MLX.eval([next] + caches.flatMap { $0.stateArrays })
            out.append(prev.item(Int.self)); cur = next
        }
        StreamingMoEBlock.probeNoSync = false
        let secs = Double(DispatchTime.now().uptimeNanoseconds - t0.uptimeNanoseconds) / 1e9
        let match = zip(out, gR).filter { $0 == $1 }.count
        return String(format: """
            [PredictFixpoint] cache 不動点反復(C=%d, maxP=%d): %.1f tok/s  品質 %d/%d=%.0f%%  平均pass=%.2f
            """, C, maxP, Double(N) / secs, match, N, Double(match) / Double(N) * 100, Double(passTotal) / Double(N))
    }

    /// **軽量 cross-layer 1-pass 予測（死路）**
    /// - 機構: full hidden を使わない軽量予測で 1-pass を狙うが、予測 hidden がズレて失敗
    /// - lossless: ❌neg（cross-layer は full hidden 必須＝M0 が勝ち）
    /// - 旧名: M1 / runM1（git e12365a, d51937b）
    public static func runCrossLayerCheap(modelDir: String, refPath: String) throws -> String {
        guard let device = MTLCreateSystemDefaultDevice() else { return "ERROR: no Metal device" }
        let r = try loadArrays(url: URL(fileURLWithPath: refPath))
        guard let promptArr = r["spec_prompt"], let gRef = r["spec_greedy"] else { return "[Tell] skip" }
        let C = Tell.envInt("QWISP_CACHE_C", 64)
        let CH = Tell.envInt("QWISP_CHUNK", 4)
        let store = try WeightStore(modelDir: modelDir)
        store.residentNonExperts()
        let source = try ExpertSource(modelDir: modelDir); try source.warm()
        let arena = try ExpertArena(device: device, source: source, N: 64)
        let model = try StreamingQwispModel(store: store, arena: arena, device: device,
                                            source: source, cacheC: C)
        let ids = promptArr.asType(.int32).reshaped([1, promptArr.dim(0)])
        let gR = gRef.asArray(Int32.self).map { Int($0) }
        let L = model.layerCount
        let N = Swift.min(Tell.envInt("QWISP_GEN", 48), gR.count)
        let caches = model.makeCaches()

        StreamingMoEBlock.probeNoSync = false
        var (_, lg) = try model.prefillChunked(ids, caches: caches)
        var cur = MLX.argMax(lg[0, lg.dim(1) - 1], axis: -1).reshaped([1, 1])
        MLX.eval([cur] + caches.flatMap { $0.stateArrays })

        func distinct(_ a: MLXArray) -> [Int] {
            var seen = Set<Int>(); var u: [Int] = []
            for e in a.asArray(Int32.self) { let i = Int(e); if seen.insert(i).inserted { u.append(i) } }
            return u
        }

        var out: [Int] = []
        let t0 = DispatchTime.now()
        for _ in 0 ..< N {
            let prev = cur
            StreamingMoEBlock.probeNoSync = true
            var h = model.embedPub(prev)
            var pos = 0
            while pos < L {
                let end = Swift.min(pos + CH, L)
                // chunk 入力 h から chunk 内各層の expert を予測（layer pos は真の入力, 以降 cross-layer）
                let preds = (pos ..< end).map { model.predictLayerInds($0, h) }
                MLX.eval(preds)                                  // この chunk の予測 inds を確定（1 drain）
                for (k, i) in (pos ..< end).enumerated() { _ = model.expertCaches[i].ensure(distinct(preds[k])) }
                h = try model.runChunk(h, pos, end, caches: caches)   // GPU-remap で実行
                pos = end
            }
            let logits = model.finalLogits(h)
            let next = MLX.argMax(logits[0, 0], axis: -1).reshaped([1, 1])
            MLX.eval([next] + caches.flatMap { $0.stateArrays })
            out.append(prev.item(Int.self)); cur = next
        }
        StreamingMoEBlock.probeNoSync = false
        let secs = Double(DispatchTime.now().uptimeNanoseconds - t0.uptimeNanoseconds) / 1e9
        let match = zip(out, gR).filter { $0 == $1 }.count
        return String(format: """
            [CrossLayerCheap] one-pass cross-layer(C=%d, chunk=%d): %.1f tok/s  品質(greedy一致) %d/%d=%.0f%%
            """,
            C, CH, Double(N) / secs, match, N, Double(match) / Double(N) * 100)
    }

    /// **cross-layer 予測 one-pass decode（Fate 相当）**
    /// - 機構: 前 token / 前 chunk の gate 入力から各層 expert を予測し prefetch、1-pass で gather
    /// - lossless: **near**（temporal 予測ゆえ recall<100%、long で発散 ~12%。GDN が予測深度を ~2層に制限）
    /// - 速度: 8GB C=64 ~26-28 tok/s
    /// - 研究: Fate one-pass cross-layer prefetch
    /// - env: QWISP_RUN_M2 / QWISP_CHUNK / QWISP_MULTI(M3 multi-source)
    /// - 旧名: M2 / runM2（git c390fb2）
    public static func runCrossLayerPredict(modelDir: String, refPath: String) throws -> String {
        guard let device = MTLCreateSystemDefaultDevice() else { return "ERROR: no Metal device" }
        let r = try loadArrays(url: URL(fileURLWithPath: refPath))
        guard let promptArr = r["spec_prompt"], let gRef = r["spec_greedy"] else { return "[Tell] skip" }
        let C = Tell.envInt("QWISP_CACHE_C", 64)
        let CH = Tell.envInt("QWISP_CHUNK", 4)
        GatedDeltaNetLayer.fuseGDN = Tell.envFlag("QWISP_FUSE_GDN")
        let store = try WeightStore(modelDir: modelDir)
        store.residentNonExperts()
        let source = try ExpertSource(modelDir: modelDir); try source.warm()
        let arena = try ExpertArena(device: device, source: source, N: 64)
        let model = try StreamingQwispModel(store: store, arena: arena, device: device,
                                            source: source, cacheC: C)
        let ids = promptArr.asType(.int32).reshaped([1, promptArr.dim(0)])
        let gR = gRef.asArray(Int32.self).map { Int($0) }
        let L = model.layerCount
        let N = Swift.min(Tell.envInt("QWISP_GEN", 48), gR.count)
        let caches = model.makeCaches()

        StreamingMoEBlock.probeNoSync = false
        var (_, lg) = try model.prefillChunked(ids, caches: caches)
        var cur = MLX.argMax(lg[0, lg.dim(1) - 1], axis: -1).reshaped([1, 1])
        MLX.eval([cur] + caches.flatMap { $0.stateArrays })

        // Swift-exact-greedy 参照（M2 は temporal 予測ゆえ真に lossless か未確認＝要検証）。QWISP_SWIFT_REF=1。
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
            StreamingMoEBlock.probeNoSync = false
        }

        func distinct(_ a: MLXArray) -> [Int] {
            var seen = Set<Int>(); var u: [Int] = []
            for e in a.asArray(Int32.self) { let i = Int(e); if seen.insert(i).inserted { u.append(i) } }
            return u
        }
        // 前 token の各層 gate 入力（chunk-0 の bootstrap 用, temporal）
        var prevGate: [MLXArray]? = nil

        let prof = Tell.envFlag("QWISP_M2_PROF")
        let prof2 = Tell.envFlag("QWISP_M2_PROF2")
        var tEval: UInt64 = 0, tEnsure: UInt64 = 0, tFinal: UInt64 = 0, pSteps = 0
        var tEmbed: UInt64 = 0, tDistinct: UInt64 = 0, tRunChunk: UInt64 = 0, tLastGate: UInt64 = 0
        if prof2 { StreamingMoEBlock.profileLayers = true }
        LayerExpertCache.preadNanos = 0; LayerExpertCache.missTotal = 0; LayerExpertCache.ensureNanos = 0
        func now() -> UInt64 { DispatchTime.now().uptimeNanoseconds }
        var out: [Int] = []
        let t0 = DispatchTime.now()
        for ti in 0 ..< N {
            let prev = cur
            StreamingMoEBlock.captureGateInput = true
            if ti == 0 {
                // bootstrap: 最初の token は full sync(demand-load) で正確な gate 入力を得る
                StreamingMoEBlock.probeNoSync = false
                let (_, lgt) = try model.forwardHidden(prev, caches: caches)
                let next = MLX.argMax(lgt[0, 0], axis: -1).reshaped([1, 1])
                MLX.eval([next] + caches.flatMap { $0.stateArrays } + model.expertCaches.compactMap { $0.lastGateInput })
                prevGate = lastGate(model)
                out.append(prev.item(Int.self)); cur = next
                continue
            }
            StreamingMoEBlock.probeNoSync = true
            // M3 multi-source: token 開始時に全層 temporal 予測(prevGate[i] 同層)を一括 prefetch（stall 無し）
            let multiSrc = Tell.envFlag("QWISP_MULTI")
            if multiSrc, let pg = prevGate {
                let tpreds = (0 ..< L).map { model.predictLayerInds($0, pg[$0]) }
                MLX.eval(tpreds)
                for i in 0 ..< L { _ = model.expertCaches[i].ensure(distinct(tpreds[i])) }
            }
            var tt = now()
            var h = model.embedPub(prev)
            if prof2 { MLX.eval(h); tEmbed += now() - tt }
            var pos = 0
            while pos < L {
                let end = Swift.min(pos + CH, L)
                // 予測元: chunk 0 は前 token 同層 gate 入力(temporal)、以降は前 chunk 最終 gate 入力
                let src = pos > 0 ? model.expertCaches[pos - 1].lastGateInput! : prevGate![pos]
                tt = now()
                let preds = (pos ..< end).map { i in model.predictLayerInds(i, pos > 0 ? src : prevGate![i]) }
                MLX.eval(preds)                                       // ← 予測(gate matmul)のみ計時(前runChunkは下のbarrierで確定済)
                if prof || prof2 { tEval += now() - tt; tt = now() }
                let dists = (pos ..< end).map { distinct(preds[$0 - pos]) }   // CPU readback(inds→list)
                if prof2 { tDistinct += now() - tt; tt = now() }
                for (k, i) in (pos ..< end).enumerated() { _ = model.expertCaches[i].ensure(dists[k]) }
                if prof || prof2 { tEnsure += now() - tt; tt = now() }
                h = try model.runChunk(h, pos, end, caches: caches)
                if prof2 {                                           // runChunk(実forward計算)を barrier で確定→計時
                    MLX.eval([h] + caches[pos ..< end].flatMap { $0.stateArrays }
                             + (pos ..< end).compactMap { model.expertCaches[$0].lastGateInput })
                    tRunChunk += now() - tt
                }
                pos = end
            }
            tt = now()
            let logits = model.finalLogits(h)
            let next = MLX.argMax(logits[0, 0], axis: -1).reshaped([1, 1])
            MLX.eval([next] + caches.flatMap { $0.stateArrays } + model.expertCaches.compactMap { $0.lastGateInput })
            if prof || prof2 { tFinal += now() - tt; tt = now(); pSteps += 1 }
            prevGate = lastGate(model)
            if prof2 { tLastGate += now() - tt }
            out.append(prev.item(Int.self)); cur = next
        }
        StreamingMoEBlock.probeNoSync = false; StreamingMoEBlock.captureGateInput = false
        let secs = Double(DispatchTime.now().uptimeNanoseconds - t0.uptimeNanoseconds) / 1e9
        let match = zip(out, gR).filter { $0 == $1 }.count
        if prof, pSteps > 0 {
            let s = Double(pSteps)
            FileHandle.standardError.write(String(format:
                "[CrossLayerPredict-PROF/tok] eval(preds=stall)=%.1f ensure(IO)=%.1f final-drain=%.1f (ms) chunks/tok=%d\n",
                Double(tEval)/s/1e6, Double(tEnsure)/s/1e6, Double(tFinal)/s/1e6, (L + CH - 1) / CH).data(using: .utf8)!)
            FileHandle.standardError.write(String(format:
                "[CrossLayerPredict-PROF ensure内訳/tok] pread(IO)=%.2fms misses=%.1f/tok | ensure合計=%.2fms → CPU(slot/distinct)=%.2fms\n",
                Double(LayerExpertCache.preadNanos)/s/1e6, Double(LayerExpertCache.missTotal)/s,
                Double(LayerExpertCache.ensureNanos)/s/1e6,
                Double(LayerExpertCache.ensureNanos - LayerExpertCache.preadNanos)/s/1e6).data(using: .utf8)!)
        }
        if prof2, pSteps > 0 {
            StreamingMoEBlock.profileLayers = false
            let s = Double(pSteps); func m(_ x: UInt64) -> Double { Double(x)/s/1e6 }
            let tot = m(tEmbed)+m(tEval)+m(tDistinct)+m(tEnsure)+m(tRunChunk)+m(tFinal)+m(tLastGate)
            FileHandle.standardError.write(String(format:
                "[CrossLayerPredict-PROF2/tok barrier] embed=%.2f predict(gate)=%.2f distinct(readback)=%.2f ensure(IO)=%.2f "
                + "runChunk(attn/gdn/moe)=%.2f final(norm/lmhead)=%.2f lastGate=%.2f | 合計=%.1fms\n",
                m(tEmbed), m(tEval), m(tDistinct), m(tEnsure), m(tRunChunk), m(tFinal), m(tLastGate), tot).data(using: .utf8)!)
            FileHandle.standardError.write(String(format:
                "[CrossLayerPredict-PROF2 runChunk内訳/tok] GDN(30層)=%.2f attn(10層)=%.2f MoE-gather(40層)=%.2f "
                + "MoE-shared(40層)=%.2f norm=%.2f (ms)\n",
                m(StreamingMoEBlock.tGDN), m(StreamingMoEBlock.tAttn), m(StreamingMoEBlock.tMoEgather),
                m(StreamingMoEBlock.tMoEshared), m(StreamingMoEBlock.tNorm)).data(using: .utf8)!)
            FileHandle.standardError.write(String(format:
                "[CrossLayerPredict-PROF2 GDN内訳/tok] in_proj(4本)=%.2f conv1d+norm=%.2f recurrent-kernel=%.2f out_proj=%.2f (ms)\n",
                m(StreamingMoEBlock.tGdnInproj), m(StreamingMoEBlock.tGdnConv),
                m(StreamingMoEBlock.tGdnKernel), m(StreamingMoEBlock.tGdnOut)).data(using: .utf8)!)
        }
        let swiftTag = gSwift.isEmpty ? ""
            : String(format: "  [vs Swift-greedy %d/%d=%.0f%%]",
                     zip(out, gSwift).filter { $0 == $1 }.count, N,
                     Double(zip(out, gSwift).filter { $0 == $1 }.count) / Double(N) * 100)
        return String(format: """
            [CrossLayerPredict] Fate one-pass(C=%d, chunk=%d): %.1f tok/s  品質(vs Python) %d/%d=%.0f%%%@
            """,
            C, CH, Double(N) / secs, match, N, Double(match) / Double(N) * 100, swiftTag)
    }

    /// **pipeline decode（死路）**
    /// - 機構: layer pipeline 化で sync を隠す試み
    /// - lossless: strict だが ❌ ~27.4＝M2 超も M0 に勝てず
    /// - 旧名: M5 / runM5（git 0715f5b）
    public static func runPipelineDecode(modelDir: String, refPath: String) throws -> String {
        guard let device = MTLCreateSystemDefaultDevice() else { return "ERROR: no Metal device" }
        let r = try loadArrays(url: URL(fileURLWithPath: refPath))
        guard let promptArr = r["spec_prompt"], let gRef = r["spec_greedy"] else { return "[PipelineDecode] skip" }
        let C = Tell.envInt("QWISP_CACHE_C", 64)
        let CH = Tell.envInt("QWISP_CHUNK", 2)
        let store = try WeightStore(modelDir: modelDir)
        store.residentNonExperts()
        let source = try ExpertSource(modelDir: modelDir); try source.warm()
        let arena = try ExpertArena(device: device, source: source, N: 64)
        let model = try StreamingQwispModel(store: store, arena: arena, device: device,
                                            source: source, cacheC: C)
        let ids = promptArr.asType(.int32).reshaped([1, promptArr.dim(0)])
        let gR = gRef.asArray(Int32.self).map { Int($0) }
        let L = model.layerCount
        let N = Swift.min(Tell.envInt("QWISP_GEN", 48), gR.count)
        let caches = model.makeCaches()

        StreamingMoEBlock.probeNoSync = false
        var (_, lg) = try model.prefillChunked(ids, caches: caches)
        var cur = MLX.argMax(lg[0, lg.dim(1) - 1], axis: -1).reshaped([1, 1])
        MLX.eval([cur] + caches.flatMap { $0.stateArrays })
        var prevGate: [MLXArray]? = nil

        func predEnsure(_ lo: Int, _ hi: Int, _ src: [MLXArray]) {
            // src: 各層 lo..hi の予測元 gate 入力。preds を eval して distinct→ensure(gather)。
            let preds = (lo ..< hi).map { i in model.predictLayerInds(i, src[i - lo]) }
            MLX.eval(preds)
            for (k, i) in (lo ..< hi).enumerated() { _ = model.expertCaches[i].ensure(distinctInts(preds[k])) }
        }

        var out: [Int] = []
        let t0 = DispatchTime.now()
        for ti in 0 ..< N {
            let prev = cur
            StreamingMoEBlock.captureGateInput = true
            if ti == 0 {
                StreamingMoEBlock.probeNoSync = false
                let (_, lgt) = try model.forwardHidden(prev, caches: caches)
                let next = MLX.argMax(lgt[0, 0], axis: -1).reshaped([1, 1])
                MLX.eval([next] + caches.flatMap { $0.stateArrays } + model.expertCaches.compactMap { $0.lastGateInput })
                prevGate = lastGate(model)
                out.append(prev.item(Int.self)); cur = next
                continue
            }
            StreamingMoEBlock.probeNoSync = true
            var h = model.embedPub(prev)
            // chunk 0 は prevGate(temporal, dependency-free)で先に gather
            predEnsure(0, Swift.min(CH, L), Array(prevGate![0 ..< Swift.min(CH, L)]))
            var pos = 0
            while pos < L {
                let end = Swift.min(pos + CH, L)
                h = try model.runChunk(h, pos, end, caches: caches)
                // chunk の expert matmul を非同期で GPU へ。gate 入力(end-1)も併せて materialize 予約。
                var evals: [MLXArray] = [h] + caches[pos ..< end].flatMap { $0.stateArrays }
                if end < L, let lgi = model.expertCaches[end - 1].lastGateInput { evals.append(lgi) }
                MLX.asyncEval(evals)
                // 次 chunk の予測+gather を重ねる（expert matmul(current) と overlap 狙い）。
                if end < L {
                    let nEnd = Swift.min(end + CH, L)
                    let src = model.expertCaches[end - 1].lastGateInput!   // gate 入力で同期
                    predEnsure(end, nEnd, Array(repeating: src, count: nEnd - end))
                }
                pos = end
            }
            let logits = model.finalLogits(h)
            let next = MLX.argMax(logits[0, 0], axis: -1).reshaped([1, 1])
            MLX.eval([next] + caches.flatMap { $0.stateArrays } + model.expertCaches.compactMap { $0.lastGateInput })
            prevGate = lastGate(model)
            out.append(prev.item(Int.self)); cur = next
        }
        StreamingMoEBlock.probeNoSync = false; StreamingMoEBlock.captureGateInput = false
        let secs = Double(DispatchTime.now().uptimeNanoseconds - t0.uptimeNanoseconds) / 1e9
        let match = zip(out, gR).filter { $0 == $1 }.count
        return String(format: """
            [PipelineDecode] M2+pipeline(C=%d, chunk=%d): %.1f tok/s  品質(greedy一致) %d/%d=%.0f%%
            """,
            C, CH, Double(N) / secs, match, N, Double(match) / Double(N) * 100)
    }

    /// **depth-1 MTP head 投機 + clean exact verify（issue #1 準拠）**
    /// - 機構: depth-D MTP head で D token draft（EAGLE chain: head の hidden を次 h_prev へ連鎖）→
    ///   batched f32-full exact verify([u,d0..d_{D-1}])で最長受理 prefix を commit / 外れは exact 訂正。hot-pin top-C。
    /// - 動機: novel text でも head が model 自身の hidden から予測ゆえ accept が suffix lookup より高い
    ///   (nl 0.69 vs suffix 0.23)＝high-entropy(自然文)で SuffixSpec が伸びない領域の lever。
    /// - lossless: **strict**（verify が exact ゆえ構成的。vs Swift-exact で確認）
    /// - 研究: Speculative Decoding (Leviathan/Chen 2023) + Multi-Token Prediction (DeepSeek-V3; Gloeckle 2024) + EAGLE
    /// - env: QWISP_RUN=mtp-spec-verify / QWISP_MTP_DEPTH(draft 段数,既定4) / QWISP_CACHE_C / QWISP_CALIB /
    ///   QWISP_SWIFT_REF=1 / QWISP_SPECK_PROF / QWISP_F32_ATTN・QWISP_F32_CONV(既定1=f32-full verify)
    /// - 旧名: M4 / M2c。旧 depth-1+seqMT を depth-D+f32-full batched verify へ一般化(2026-06-28)
    public static func runMTPSpecVerify(modelDir: String, refPath: String) throws -> String {
        guard let device = MTLCreateSystemDefaultDevice() else { return "ERROR: no Metal device" }
        let r = try loadArrays(url: URL(fileURLWithPath: refPath))
        guard let promptArr = r["spec_prompt"], let gRef = r["spec_greedy"] else { return "[MTPSpecVerify] skip" }
        let C = Tell.envInt("QWISP_CACHE_C", 64)
        let calibN = Tell.envInt("QWISP_CALIB", 48)
        let store = try WeightStore(modelDir: modelDir)
        store.residentNonExperts()
        let source = try ExpertSource(modelDir: modelDir); try source.warm()
        let arena = try ExpertArena(device: device, source: source, N: 64)
        let model = try StreamingQwispModel(store: store, arena: arena, device: device,
                                            source: source, cacheC: C)
        let head = try MTPHead(modelDir: modelDir, store: store)
        let ids = promptArr.asType(.int32).reshaped([1, promptArr.dim(0)])
        let gR = gRef.asArray(Int32.self).map { Int($0) }
        let isLin = model.isLinearFlags
        let N = Swift.min(Tell.envInt("QWISP_GEN", 48), gR.count)
        let nE = 256, nMoE = model.expertCaches.count

        // phase 1: calib（hot-pin 用の頻度集計。verify は exact ゆえ buddy 不要）
        var counts = [[Int]](repeating: [Int](repeating: 0, count: nE), count: nMoE)
        let cc = model.makeCaches()
        StreamingMoEBlock.probeNoSync = false; StreamingMoEBlock.captureInds = true
        var (_, clg) = try model.prefillChunked(ids, caches: cc)
        var ccur = MLX.argMax(clg[0, clg.dim(1) - 1], axis: -1).reshaped([1, 1])
        MLX.eval([ccur] + cc.flatMap { $0.stateArrays })
        for _ in 0 ..< calibN {
            (_, clg) = try model.forwardHidden(ccur, caches: cc)
            MLX.eval([clg] + cc.flatMap { $0.stateArrays } + model.expertCaches.compactMap { $0.lastInds })
            for (mi, ec) in model.expertCaches.enumerated() {
                if let li = ec.lastInds { for e in li.asArray(Int32.self) { counts[mi][Int(e)] += 1 } }
            }
            ccur = MLX.argMax(clg[0, 0], axis: -1).reshaped([1, 1]); MLX.eval([ccur])
        }
        StreamingMoEBlock.captureInds = false

        // Swift-exact-greedy 参照（lossless 検証）
        var gSwift: [Int] = []
        if Tell.envFlag("QWISP_SWIFT_REF") {
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
        }

        // phase 2: top-C hot-pin（verify の expert を常駐化, demand-load を減らす）
        for (mi, ec) in model.expertCaches.enumerated() {
            _ = ec.ensure(Array(counts[mi].enumerated()
                .sorted { $0.element != $1.element ? $0.element > $1.element : $0.offset < $1.offset }
                .prefix(C).map { $0.offset }))
        }

        // phase 3: depth-D MTP draft（EAGLE chain: head の hidden x を次 draft の h_prev へ連鎖）
        //   + batched f32-full exact verify（SuffixSpec と同根, divergent op=attention SDPA+GDN conv1d のみ）。
        //   novel text でも head が model 自身の hidden から予測ゆえ accept が suffix より高い(nl 0.69 vs 0.23)。
        let depth = Swift.max(1, Tell.envInt("QWISP_MTP_DEPTH", 4))
        GatedDeltaNetLayer.f32Conv = Tell.envStr("QWISP_F32_CONV", "1") != "0"
        AttentionLayer.f32SDPA = Tell.envStr("QWISP_F32_ATTN", "1") != "0"
        defer { GatedDeltaNetLayer.f32Conv = false; AttentionLayer.f32SDPA = false }
        let mc = model.makeCaches()
        let mtpKV = KVCache()
        let P = ids.dim(-1)
        StreamingMoEBlock.probeNoSync = false
        let (Hf, lgf) = try model.prefillChunked(ids, caches: mc)
        var uArr = MLX.argMax(lgf[0..., (lgf.dim(1) - 1)...], axis: -1)
        var lastH = Hf[0..., (P - 1)...]
        _ = head(Hf[0..., 0 ..< (P - 1)], ids[0..., 1...], cache: mtpKV)
        MLX.eval([uArr, lastH] + [mtpKV.keys, mtpKV.values].compactMap { $0 } + mc.flatMap { $0.stateArrays })
        let prof = Tell.envFlag("QWISP_SPECK_PROF")
        var tDraft: UInt64 = 0, tVerify: UInt64 = 0
        func now() -> UInt64 { DispatchTime.now().uptimeNanoseconds }
        var out: [Int] = []; var steps = 0, acc = 0
        let t0 = DispatchTime.now()
        while out.count < N {
            steps += 1
            var ts = now()
            // depth-D draft: head を chain（dx を次 h_prev へ）。各 call が mtpKV に 1 entry 追加。
            var drafts: [MLXArray] = []
            var hPrev = lastH, tokIn = uArr
            for _ in 0 ..< depth {
                let (dl, dx) = head.callWithHidden(hPrev, tokIn, cache: mtpKV)
                let dArr = MLX.argMax(dl[0..., 0...], axis: -1)            // [1,1]
                drafts.append(dArr); hPrev = dx; tokIn = dArr
            }
            let draftToks = MLX.concatenated(drafts, axis: 1)             // [1,depth]
            let seq = MLX.concatenated([uArr, draftToks], axis: 1)        // [1,depth+1]
            if prof { MLX.eval([seq] + [mtpKV.keys, mtpKV.values].compactMap { $0 }); tDraft += now() - ts; ts = now() }
            let snaps = mc.map { $0.snapshot() }
            let (H, lg) = try model.forwardHidden(seq, caches: mc)        // ★batched f32-full exact verify
            let evals = MLX.argMax(lg[0, 0 ..< (depth + 1)], axis: -1).asArray(Int32.self).map { Int($0) }
            let dArrI = draftToks.asArray(Int32.self).map { Int($0) }
            var p = 0
            while p < depth && dArrI[p] == evals[p] { p += 1 }            // 最長受理 prefix
            out.append(uArr.item(Int.self))
            for i in 0 ..< p { out.append(dArrI[i]) }
            acc += p
            let pCorr = MLXArray([Int32(evals[p])], [1, 1])              // 次トークン(correction or next)
            if p < depth {                                               // reject: restore → committed 再forward
                for (i, c) in mc.enumerated() { c.restore(snaps[i], isLinear: isLin[i], trim: depth + 1) }
                let commit = p > 0 ? MLX.concatenated([uArr, draftToks[0..., 0 ..< p]], axis: 1) : uArr
                _ = try model.forwardHidden(commit, caches: mc)
            }
            // mtpKV を committed 状態へ: draft entries(depth 本)を trim → 真 hidden で committed(p+1 本)を catch-up
            mtpKV.trim(depth)
            let hPrevSeq = p > 0 ? MLX.concatenated([lastH, H[0..., 0 ..< p]], axis: 1) : lastH
            let commitKV = p > 0 ? MLX.concatenated([uArr, draftToks[0..., 0 ..< p]], axis: 1) : uArr
            _ = head(hPrevSeq, commitKV, cache: mtpKV)
            uArr = pCorr; lastH = H[0..., p ..< (p + 1)]
            MLX.eval([uArr, lastH] + [mtpKV.keys, mtpKV.values].compactMap { $0 } + mc.flatMap { $0.stateArrays })
            if prof { tVerify += now() - ts }
        }
        StreamingMoEBlock.probeNoSync = false
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
                "[MTPSpec-PROF/step] draft(MTP head×%d)=%.1f verify(f32-full batched)=%.1f (ms)  steps=%d\n",
                depth, Double(tDraft)/s/1e6, Double(tVerify)/s/1e6, steps).data(using: .utf8)!)
        }
        return String(format: """
            [MTPSpecVerify] depth-%d MTP draft + batched f32-full verify(C=%d): %.1f tok/s  accept/step=%.3f  品質(vs Python) %d/%d=%.0f%%%@
            """, depth, C, Double(N) / secs, Double(acc) / Double(steps), match, N, Double(match) / Double(N) * 100, swiftTag)
    }

    /// forward コストの L 依存ベンチ（streaming-bound vs compute/overhead-bound の切り分け）。
    /// hot-pin top-C で IO を排除し、teacher-forced で L=1,2,4,8,16,24 の forwardHidden を計時。
    /// cost(L)≈const(overhead 律速) なら multi-token 安く speculation 有利、cost(L)≈L·cost(1)(compute 律速)
    /// なら per-token compute 削減(GDN kernel 等)が要。restore で同一 prefill 状態から反復。
    /// - env: QWISP_RUN=forward-cost / QWISP_CACHE_C / QWISP_FC_REPS(反復,既定20)
    public static func runForwardCost(modelDir: String, refPath: String) throws -> String {
        guard let device = MTLCreateSystemDefaultDevice() else { return "ERROR: no Metal device" }
        let r = try loadArrays(url: URL(fileURLWithPath: refPath))
        guard let promptArr = r["spec_prompt"] else { return "[ForwardCost] skip" }
        let C = Tell.envInt("QWISP_CACHE_C", 64)
        let reps = Tell.envInt("QWISP_FC_REPS", 20)
        GatedDeltaNetLayer.f32Conv = true; AttentionLayer.f32SDPA = true
        defer { GatedDeltaNetLayer.f32Conv = false; AttentionLayer.f32SDPA = false }
        let store = try WeightStore(modelDir: modelDir); store.residentNonExperts()
        let source = try ExpertSource(modelDir: modelDir); try source.warm()
        let arena = try ExpertArena(device: device, source: source, N: 64)
        let model = try StreamingQwispModel(store: store, arena: arena, device: device, source: source, cacheC: C)
        let ids = promptArr.asType(.int32).reshaped([1, promptArr.dim(0)])
        let isLin = model.isLinearFlags
        let nE = 256, nMoE = model.expertCaches.count
        // calib + hot-pin top-C で IO 排除（pure compute を測る）
        var counts = [[Int]](repeating: [Int](repeating: 0, count: nE), count: nMoE)
        let cc = model.makeCaches()
        StreamingMoEBlock.probeNoSync = false; StreamingMoEBlock.captureInds = true
        var (_, clg) = try model.prefillChunked(ids, caches: cc)
        var ccur = MLX.argMax(clg[0, clg.dim(1) - 1], axis: -1).reshaped([1, 1])
        MLX.eval([ccur] + cc.flatMap { $0.stateArrays })
        for _ in 0 ..< 48 {
            (_, clg) = try model.forwardHidden(ccur, caches: cc)
            MLX.eval([clg] + cc.flatMap { $0.stateArrays } + model.expertCaches.compactMap { $0.lastInds })
            for (mi, ec) in model.expertCaches.enumerated() {
                if let li = ec.lastInds { for e in li.asArray(Int32.self) { counts[mi][Int(e)] += 1 } }
            }
            ccur = MLX.argMax(clg[0, 0], axis: -1).reshaped([1, 1]); MLX.eval([ccur])
        }
        StreamingMoEBlock.captureInds = false
        for (mi, ec) in model.expertCaches.enumerated() {
            _ = ec.ensure(Array(counts[mi].enumerated()
                .sorted { $0.element != $1.element ? $0.element > $1.element : $0.offset < $1.offset }
                .prefix(C).map { $0.offset }))
        }
        func now() -> UInt64 { DispatchTime.now().uptimeNanoseconds }
        var lines: [String] = []
        for L in [1, 2, 4, 8, 16, 24] {
            let bc = model.makeCaches()
            _ = try model.prefillChunked(ids, caches: bc)
            MLX.eval(bc.flatMap { $0.stateArrays })
            let snaps = bc.map { $0.snapshot() }
            let seq = MLXArray(Array(repeating: Int32(100), count: L), [1, L])   // teacher-forced ダミー
            // warmup（experts ensure 込み）
            let (hw, _) = try model.forwardHidden(seq, caches: bc); MLX.eval([hw] + bc.flatMap { $0.stateArrays })
            for (i, c) in bc.enumerated() { c.restore(snaps[i], isLinear: isLin[i], trim: L) }
            var tAcc: UInt64 = 0
            for _ in 0 ..< reps {
                let t = now()
                let (h, _) = try model.forwardHidden(seq, caches: bc)
                MLX.eval([h] + bc.flatMap { $0.stateArrays })
                tAcc += now() - t
                for (i, c) in bc.enumerated() { c.restore(snaps[i], isLinear: isLin[i], trim: L) }   // 非計時
            }
            let ms = Double(tAcc) / Double(reps) / 1e6
            lines.append(String(format: "L=%2d: %6.2f ms  (%.2f ms/token)", L, ms, ms / Double(L)))
        }
        return "[ForwardCost C=\(C), f32-full, hot-pin top-C]\n  " + lines.joined(separator: "\n  ")
    }

    /// **SuffixDecoding draft + clean exact verify（issue #2 軸B, 訓練不要）**
    /// - 機構: prompt+生成履歴の suffix を lookup し、過去に同 suffix の後に続いた token 列を K 個まで
    ///   無料 draft → batched f32-full exact verify で照合 → 一致 prefix を commit。draft cost ~0。
    /// - lossless: **strict**（batched f32-full verify が逐次 decode と bit-exact）。token/step は反復性に依存。
    /// - 速度: code/agentic の反復で高 accept→高 token/step。free-form(high-entropy)では accept 低下。
    ///   実測 8GB C=64: mix 88 tok/s @maxK24 / 16GB C=128: mix 133 tok/s @maxK24-48（vs Swift-greedy 100%）。
    /// - 研究: SuffixDecoding (arXiv:2411.04975), Prompt-Lookup Decoding
    /// - env: QWISP_RUN=suffix-spec / QWISP_CACHE_C / QWISP_DRAFT_K(最大draft長,既定16・C×3/8でクランプ) /
    ///   QWISP_SUFFIX_MIN(最小一致) / QWISP_SUFFIX_MATCH(最大一致) / QWISP_SWIFT_REF / QWISP_SPECK_PROF /
    ///   QWISP_F32_ATTN・QWISP_F32_CONV(既定1=f32-full, 0 で f16 batched) / QWISP_VERIFY_SEQ・QWISP_VERIFY_PQN(診断用逐次化)
    public static func runSuffixSpec(modelDir: String, refPath: String) throws -> String {
        guard let device = MTLCreateSystemDefaultDevice() else { return "ERROR: no Metal device" }
        let r = try loadArrays(url: URL(fileURLWithPath: refPath))
        guard let promptArr = r["spec_prompt"], let gRef = r["spec_greedy"] else { return "[SuffixSpec] skip" }
        let C = Tell.envInt("QWISP_CACHE_C", 64)
        let calibN = Tell.envInt("QWISP_CALIB", 48)
        // ★ batched f32-full verify が既定(investigate C + batched 再評価で確定):
        //   verify forward の divergent op は attention SDPA(.causal/.none 経路差 ~7e-4)と GDN conv1d のみ。
        //   両者を f32 化すると残り全 op(quantized matmul / GDN updateKernel / RoPE / rmsNorm / softmax)は
        //   order-stable(rel=0)ゆえ verify forward 全体が逐次 decode と bit 一致(micro-test attn=1.08e-6 確認)。
        //   → 逐次化(seqMT/perQueryNone)不要の単一 batched forward が provably lossless かつ最速。
        //   f16 batched は ~7e-4 drift だが SuffixSpec は reject 自己訂正ゆえ実用 lossless(誤受理は near-tie のみ・保証なし)。
        //   旧 maxK=4 上限は f16 運頼みを避ける保護だったが f32-full は bit-exact ゆえ撤廃。
        // ★ 但し別の上限が残る: D+1 トークンの batched verify で 1 層が同時に要するユニーク expert 数が
        //   per-layer cache 容量 C を超えると evict しきれず wrong-slot=silent garbage(クラッシュせず誤受理)。
        //   実測安全境界 C=64→maxK24 / C=128→maxK48 = maxK ≤ C×3/8。これで C 比例クランプ(精度でなく容量制約)。
        let maxKReq = Tell.envInt("QWISP_DRAFT_K", 16)
        let maxKSafe = Swift.max(4, C * 3 / 8)
        let maxK = Swift.min(maxKReq, maxKSafe)
        if maxK < maxKReq {
            print("[SuffixSpec] maxK \(maxKReq)→\(maxK) にクランプ(C=\(C) の arena 容量制約 C×3/8, |U|>C 回避)")
        }
        let minMatch = Tell.envInt("QWISP_SUFFIX_MIN", 2)
        let maxMatch = Tell.envInt("QWISP_SUFFIX_MATCH", 32)
        // 既定 f32-full(QWISP_F32_ATTN/CONV=0 で各々無効化可)。f16 batched を試すなら両方 0。
        GatedDeltaNetLayer.f32Conv = Tell.envStr("QWISP_F32_CONV", "1") != "0"
        AttentionLayer.f32SDPA = Tell.envStr("QWISP_F32_ATTN", "1") != "0"
        defer { GatedDeltaNetLayer.f32Conv = false; AttentionLayer.f32SDPA = false; AttentionLayer.perQueryNone = false }
        let store = try WeightStore(modelDir: modelDir)
        store.residentNonExperts()
        let source = try ExpertSource(modelDir: modelDir); try source.warm()
        let arena = try ExpertArena(device: device, source: source, N: 64)
        let model = try StreamingQwispModel(store: store, arena: arena, device: device, source: source, cacheC: C)
        let ids = promptArr.asType(.int32).reshaped([1, promptArr.dim(0)])
        let gR = gRef.asArray(Int32.self).map { Int($0) }
        let isLin = model.isLinearFlags
        let N = Swift.min(Tell.envInt("QWISP_GEN", 48), gR.count)
        let nE = 256, nMoE = model.expertCaches.count

        // phase 1: calib（hot-pin 用頻度）
        var counts = [[Int]](repeating: [Int](repeating: 0, count: nE), count: nMoE)
        let cc = model.makeCaches()
        StreamingMoEBlock.probeNoSync = false; StreamingMoEBlock.captureInds = true
        var (_, clg) = try model.prefillChunked(ids, caches: cc)
        var ccur = MLX.argMax(clg[0, clg.dim(1) - 1], axis: -1).reshaped([1, 1])
        MLX.eval([ccur] + cc.flatMap { $0.stateArrays })
        for _ in 0 ..< calibN {
            (_, clg) = try model.forwardHidden(ccur, caches: cc)
            MLX.eval([clg] + cc.flatMap { $0.stateArrays } + model.expertCaches.compactMap { $0.lastInds })
            for (mi, ec) in model.expertCaches.enumerated() {
                if let li = ec.lastInds { for e in li.asArray(Int32.self) { counts[mi][Int(e)] += 1 } }
            }
            ccur = MLX.argMax(clg[0, 0], axis: -1).reshaped([1, 1]); MLX.eval([ccur])
        }
        StreamingMoEBlock.captureInds = false

        // Swift-exact-greedy 参照
        var gSwift: [Int] = []
        if Tell.envFlag("QWISP_SWIFT_REF") {
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
        }

        // phase 2: top-C hot-pin
        for (mi, ec) in model.expertCaches.enumerated() {
            _ = ec.ensure(Array(counts[mi].enumerated()
                .sorted { $0.element != $1.element ? $0.element > $1.element : $0.offset < $1.offset }
                .prefix(C).map { $0.offset }))
        }

        // phase 3: suffix-lookup draft + clean exact verify
        var hist = ids.asArray(Int32.self).map { Int($0) }     // 履歴（prompt + commit token）
        let mc = model.makeCaches()
        StreamingMoEBlock.probeNoSync = false
        var (_, lg) = try model.prefillChunked(ids, caches: mc)
        var uArr = MLX.argMax(lg[0, lg.dim(1) - 1], axis: -1).reshaped([1, 1])
        MLX.eval([uArr] + mc.flatMap { $0.stateArrays })
        // verify 逐次化は既定 OFF(f32-full batched が代替)。診断用に明示時のみ有効化:
        //   QWISP_VERIFY_SEQ=1 → seqMT(層丸ごと per-token), QWISP_VERIFY_PQN=1 → per-query .none(SDPA のみ)。
        let vseq = Tell.envFlag("QWISP_VERIFY_SEQ")
        let vpqn = Tell.envFlag("QWISP_VERIFY_PQN")
        func setVerifyMode(_ on: Bool) {
            if vpqn { AttentionLayer.perQueryNone = on; AttentionLayer.seqMultiToken = false }
            else { AttentionLayer.seqMultiToken = on && vseq }
        }
        let prof = Tell.envFlag("QWISP_SPECK_PROF")
        var tDraft: UInt64 = 0, tVerify: UInt64 = 0
        func now() -> UInt64 { DispatchTime.now().uptimeNanoseconds }
        var out: [Int] = []; var steps = 0, accTok = 0, draftTot = 0
        let t0 = DispatchTime.now()
        while out.count < N {
            steps += 1
            let u = uArr.item(Int.self)
            var ts = now()
            let drafts = suffixDraft(hist + [u], maxMatch: maxMatch, draftK: maxK, minMatch: minMatch)
            let D = drafts.count
            draftTot += D
            if prof { tDraft += now() - ts; ts = now() }
            if D == 0 {                                          // 一致なし → 通常 greedy 1 step
                let (_, glg) = try model.forwardHidden(uArr, caches: mc)
                out.append(u); hist.append(u)
                uArr = MLX.argMax(glg[0, 0], axis: -1).reshaped([1, 1])
                MLX.eval([uArr] + mc.flatMap { $0.stateArrays })
                if prof { tVerify += now() - ts }
                continue
            }
            setVerifyMode(true)
            let snaps = mc.map { $0.snapshot() }
            let seq = MLX.concatenated([uArr, MLXArray(drafts.map { Int32($0) }, [1, D])], axis: 1)  // [1, D+1]
            let (_, vlg) = try model.forwardHidden(seq, caches: mc)
            let evals = MLX.argMax(vlg[0, 0 ..< (D + 1)], axis: -1).asArray(Int32.self).map { Int($0) }
            var p = 0
            while p < D && drafts[p] == evals[p] { p += 1 }
            out.append(u); hist.append(u)
            for i in 0 ..< p { out.append(drafts[i]); hist.append(drafts[i]) }
            accTok += p
            if p == D {
                uArr = MLXArray([Int32(evals[D])], [1, 1])
                setVerifyMode(false)
                MLX.eval([uArr] + mc.flatMap { $0.stateArrays })
            } else {
                for (i, c) in mc.enumerated() { c.restore(snaps[i], isLinear: isLin[i], trim: D + 1) }
                let acc = [u] + Array(drafts.prefix(p))
                _ = try model.forwardHidden(MLXArray(acc.map { Int32($0) }, [1, acc.count]), caches: mc)
                setVerifyMode(false)
                uArr = MLXArray([Int32(evals[p])], [1, 1])
                MLX.eval([uArr] + mc.flatMap { $0.stateArrays })
            }
            if prof { tVerify += now() - ts }
        }
        AttentionLayer.seqMultiToken = false; AttentionLayer.perQueryNone = false
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
                "[SuffixSpec-PROF/step] draft(lookup)=%.2f verify=%.1f (ms)  draft長平均=%.1f  steps=%d\n",
                Double(tDraft)/s/1e6, Double(tVerify)/s/1e6, Double(draftTot)/s, steps).data(using: .utf8)!)
        }
        return String(format: """
            [SuffixSpec] suffix draft(maxK=%d) + clean exact verify(C=%d): %.1f tok/s  accept/step=%.2f  品質(vs Python) %d/%d=%.0f%%%@
            """, maxK, C, Double(N) / secs, Double(accTok) / Double(steps), match, N, Double(match) / Double(N) * 100, swiftTag)
    }

    /// suffix lookup draft: seq 末尾の m token(minMatch..maxMatch の最長)が seq 内の earlier 位置に
    /// 出現した「直後の token 列」を draftK 個まで返す（最近・最長一致優先）。訓練不要・cost ~0。
    static func suffixDraft(_ seq: [Int], maxMatch: Int, draftK: Int, minMatch: Int) -> [Int] {
        let n = seq.count
        if n < minMatch + 1 { return [] }
        var m = Swift.min(maxMatch, n - 1)
        while m >= minMatch {
            let patStart = n - m
            var i = patStart - 1
            while i >= 0 {
                var ok = true
                for j in 0 ..< m where seq[i + j] != seq[patStart + j] { ok = false; break }
                if ok {
                    let s = i + m, e = Swift.min(i + m + draftK, n)
                    if s < e { return Array(seq[s ..< e]) }
                }
                i -= 1
            }
            m -= 1
        }
        return []
    }
}
