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
        // ★ C は device 別 auto-config(SuffixSpec と同じ calibration layer): 8→64/16→128/24→192/32→256。
        //   QWISP_CACHE_C で上書き可。QWISP_DEVICE_RAM=<GB> で他 tier を模擬。
        let C = Tell.envInt("QWISP_CACHE_C", DeviceCalibration.defaultC())
        if ProcessInfo.processInfo.environment["QWISP_CACHE_C"] == nil {
            print("[calibration] " + DeviceCalibration.recommend().summary)
        }
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
        let gate = Tell.envFloat("QWISP_MTP_GATE", 0)   // >0: head top1prob<gate で draft 打ち切り(AdaEDL 式)
        GatedDeltaNetLayer.f32Conv = Tell.envStr("QWISP_F32_CONV", "1") != "0"
        AttentionLayer.f32SDPA = Tell.envStr("QWISP_F32_ATTN", "1") != "0"
        defer { GatedDeltaNetLayer.f32Conv = false; AttentionLayer.f32SDPA = false }
        let mc = model.makeCaches()
        let mtpKV = KVCache()
        let P = ids.dim(-1)
        // ★ issue#9 突破口: C≥256(全常駐)では verify を no-sync 化(ミス無し→no-sync が exact)。
        //   per-layer drain を排除＝SuffixSpec の no-sync-pure と同速 path。batched 発散は f32-2op で吸収=軽い lossless verify。
        //   QWISP_VERIFY_NOSYNC=1 で強制、=0 で無効。既定 C≥256 で on。
        let noSyncVerify = (ProcessInfo.processInfo.environment["QWISP_VERIFY_NOSYNC"].map { $0 == "1" }) ?? (C >= 256)
        StreamingMoEBlock.probeNoSync = noSyncVerify
        let (Hf, lgf) = try model.prefillChunked(ids, caches: mc)
        var uArr = MLX.argMax(lgf[0..., (lgf.dim(1) - 1)...], axis: -1)
        var lastH = Hf[0..., (P - 1)...]
        _ = head(Hf[0..., 0 ..< (P - 1)], ids[0..., 1...], cache: mtpKV)
        MLX.eval([uArr, lastH] + [mtpKV.keys, mtpKV.values].compactMap { $0 } + mc.flatMap { $0.stateArrays })
        let prof = Tell.envFlag("QWISP_SPECK_PROF")
        var tDraft: UInt64 = 0, tVerify: UInt64 = 0, tVerifyFwd: UInt64 = 0
        var nReject = 0
        func now() -> UInt64 { DispatchTime.now().uptimeNanoseconds }
        var out: [Int] = []; var steps = 0, acc = 0
        // ★ profile: IO(pread)/sync(per-layer drain) を全 C で切り分け
        LayerExpertCache.preadNanos = 0; LayerExpertCache.ensureNanos = 0; LayerExpertCache.missTotal = 0
        StreamingMoEBlock.syncNanos = 0
        // ★ C<256 sync drain 削減: verify を no-sync whole-token + escalate(lossless)で回す。
        //   既定 C<256 で on(per-layer drain 半減・lossless 実証)、C>=256 は noSyncVerify が担うので off。QWISP_VERIFY_PREFETCH で上書き。
        let verifyPrefetch = (ProcessInfo.processInfo.environment["QWISP_VERIFY_PREFETCH"].map { $0 == "1" }) ?? (C < 256)
        var priorInds: [[Int]]? = nil
        var escSum = 0
        func distinctL(_ a: MLXArray?) -> [Int] {
            guard let a = a else { return [] }
            var seen = Set<Int>(); var u: [Int] = []
            for e in a.asArray(Int32.self) { let i = Int(e); if seen.insert(i).inserted { u.append(i) } }
            return u
        }
        let t0 = DispatchTime.now()
        while out.count < N {
            steps += 1
            var ts = now()
            // depth-D draft: head を chain（dx を次 h_prev へ）。各 call が mtpKV に 1 entry 追加。
            // gate>0: head top1prob<gate で打ち切り（hard token で無駄 draft node を払わない）。
            var drafts: [MLXArray] = []
            var hPrev = lastH, tokIn = uArr
            for _ in 0 ..< depth {
                let (dl, dx) = head.callWithHidden(hPrev, tokIn, cache: mtpKV)
                if gate > 0 {
                    let pc = MLX.max(MLX.softmax(dl[0, 0].asType(.float32), axis: -1)).item(Float.self)
                    if pc < gate { mtpKV.trim(1); break }                 // 低信頼 → この draft を捨て打ち切り
                }
                let dArr = MLX.argMax(dl[0..., 0...], axis: -1)            // [1,1]
                drafts.append(dArr); hPrev = dx; tokIn = dArr
            }
            let D = drafts.count
            if D == 0 {                                                   // draft 無し → 通常 greedy 1 step
                let (H1, lg1) = try model.forwardHidden(uArr, caches: mc)
                let nxt = MLX.argMax(lg1[0, 0], axis: -1).item(Int.self)
                out.append(uArr.item(Int.self))
                _ = head(lastH, uArr, cache: mtpKV)                       // mtpKV catch-up（uArr entry）
                uArr = MLXArray([Int32(nxt)], [1, 1]); lastH = H1[0..., 0 ..< 1]
                MLX.eval([uArr, lastH] + [mtpKV.keys, mtpKV.values].compactMap { $0 } + mc.flatMap { $0.stateArrays })
                if prof { tVerify += now() - ts }
                continue
            }
            let draftToks = MLX.concatenated(drafts, axis: 1)             // [1,D]
            let seq = MLX.concatenated([uArr, draftToks], axis: 1)        // [1,D+1]
            if prof { MLX.eval([seq] + [mtpKV.keys, mtpKV.values].compactMap { $0 }); tDraft += now() - ts; ts = now() }
            let snaps = mc.map { $0.snapshot() }
            // ★ C<256 sync drain 削減: verify を whole-token no-sync + escalate-from-first-miss(lossless)で回す。
            //   priorInds(前 step 実 routing)を prefetch hint に→早層 hit→escalate は late first-miss から=drain 2eval。
            //   QWISP_VERIFY_PREFETCH=1 で有効(既定 off=従来 exact forwardHidden)。
            let H: MLXArray, lg: MLXArray
            if verifyPrefetch {
                let (h2, l2, e2) = try model.forwardHiddenPrefetchWhole(seq, caches: mc, priorInds: priorInds, isLin: isLin)
                H = h2; lg = l2; escSum += e2
                priorInds = (0 ..< model.layerCount).map { distinctL(model.expertCaches[$0].lastInds) }   // 次 step hint
            } else {
                let (h2, l2) = try model.forwardHidden(seq, caches: mc)    // ★batched f32-full exact verify
                H = h2; lg = l2
            }
            let evals = MLX.argMax(lg[0, 0 ..< (D + 1)], axis: -1).asArray(Int32.self).map { Int($0) }  // ← 主 forward を materialize
            if prof { tVerifyFwd += now() - ts }                          // 主 batched verify forward(残=reject 再forward+mtpKV)
            let dArrI = draftToks.asArray(Int32.self).map { Int($0) }
            var p = 0
            while p < D && dArrI[p] == evals[p] { p += 1 }                // 最長受理 prefix
            out.append(uArr.item(Int.self))
            for i in 0 ..< p { out.append(dArrI[i]) }
            acc += p
            let pCorr = MLXArray([Int32(evals[p])], [1, 1])              // 次トークン(correction or next)
            if p < D {                                                   // reject: restore → committed 再forward
                nReject += 1
                for (i, c) in mc.enumerated() { c.restore(snaps[i], isLinear: isLin[i], trim: D + 1) }
                let commit = p > 0 ? MLX.concatenated([uArr, draftToks[0..., 0 ..< p]], axis: 1) : uArr
                _ = try model.forwardHidden(commit, caches: mc)
            }
            // mtpKV を committed 状態へ: draft entries(D 本)を trim → 真 hidden で committed(p+1 本)を catch-up
            mtpKV.trim(D)
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
            let s = Double(steps), tok = Double(N)
            let preadMs = Double(LayerExpertCache.preadNanos) / 1e6
            let ensureMs = Double(LayerExpertCache.ensureNanos) / 1e6
            let syncMs = Double(StreamingMoEBlock.syncNanos) / 1e6
            FileHandle.standardError.write(String(format:
                "[MTPSpec-PROF] C=%d depth=%d no-sync=%@ vprefetch=%@ steps=%d accept/step=%.2f reject=%d miss=%d escLayers=%d\n" +
                "  /step(ms): draft=%.1f verify=%.1f [主fwd=%.1f reject再fwd+他=%.1f]\n" +
                "  /token(ms): total=%.1f | pread(IO)=%.1f ensure=%.1f sync(drain)=%.1f\n",
                C, depth, (noSyncVerify ? "Y" : "N"), (verifyPrefetch ? "Y" : "N"), steps, Double(acc)/s, nReject, LayerExpertCache.missTotal, escSum,
                Double(tDraft)/s/1e6, Double(tVerify)/s/1e6, Double(tVerifyFwd)/s/1e6, Double(tVerify - tVerifyFwd)/s/1e6,
                secs/tok*1000, preadMs/tok, ensureMs/tok, syncMs/tok).data(using: .utf8)!)
        }
        return String(format: """
            [MTPSpecVerify] depth-%d MTP draft + batched f32-full verify(C=%d): %.1f tok/s  accept/step=%.3f  品質(vs Python) %d/%d=%.0f%%%@
            """, depth, C, Double(N) / secs, Double(acc) / Double(steps), match, N, Double(match) / Double(N) * 100, swiftTag)
    }

    /// pre-flight: MTP head draft 品質計測（narrow tree draft 着手判断, EAGLE-2 前提検証）。
    /// true greedy 列を teacher-force し各 position で head の top-K を取り:
    ///  (a) top-1..K coverage(=true next が top-k に入る率=width-k tree の accept 天井)
    ///  (b) confidence(top-1 prob)と accept@1 の相関(EAGLE-2 tree-value 機構=confidence で expand の前提)。
    /// top-K が top-1 を大きく超えれば narrow tree が効く / confidence が accept と単調なら tree-value 可。
    /// - env: QWISP_RUN=mtp-draft-calib / QWISP_CACHE_C / QWISP_CALIB / QWISP_GEN
    public static func runMTPDraftCalib(modelDir: String, refPath: String) throws -> String {
        guard let device = MTLCreateSystemDefaultDevice() else { return "ERROR: no Metal device" }
        let r = try loadArrays(url: URL(fileURLWithPath: refPath))
        guard let promptArr = r["spec_prompt"], let gRef = r["spec_greedy"] else { return "[MTPDraftCalib] skip" }
        let C = Tell.envInt("QWISP_CACHE_C", 64)
        let calibN = Tell.envInt("QWISP_CALIB", 48)
        let store = try WeightStore(modelDir: modelDir); store.residentNonExperts()
        let source = try ExpertSource(modelDir: modelDir); try source.warm()
        let arena = try ExpertArena(device: device, source: source, N: 64)
        let model = try StreamingQwispModel(store: store, arena: arena, device: device, source: source, cacheC: C)
        let head = try MTPHead(modelDir: modelDir, store: store)
        let ids = promptArr.asType(.int32).reshaped([1, promptArr.dim(0)])
        let gR = gRef.asArray(Int32.self).map { Int($0) }
        let N = Swift.min(Tell.envInt("QWISP_GEN", 64), gR.count)
        let nE = 256, nMoE = model.expertCaches.count
        let K = 4

        // calib + hot-pin
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
        for (mi, ec) in model.expertCaches.enumerated() {
            _ = ec.ensure(Array(counts[mi].enumerated()
                .sorted { $0.element != $1.element ? $0.element > $1.element : $0.offset < $1.offset }
                .prefix(C).map { $0.offset }))
        }

        // teacher-forced greedy + head top-K 計測
        let mc = model.makeCaches()
        let mtpKV = KVCache()
        let P = ids.dim(-1)
        let (Hf, lgf) = try model.prefillChunked(ids, caches: mc)
        var uArr = MLX.argMax(lgf[0..., (lgf.dim(1) - 1)...], axis: -1)
        var lastH = Hf[0..., (P - 1)...]
        _ = head(Hf[0..., 0 ..< (P - 1)], ids[0..., 1...], cache: mtpKV)
        MLX.eval([uArr, lastH] + [mtpKV.keys, mtpKV.values].compactMap { $0 } + mc.flatMap { $0.stateArrays })

        var covK = [Int](repeating: 0, count: K)
        let edges: [Float] = [0.0, 0.5, 0.7, 0.85, 0.95, 1.01]
        var binTot = [Int](repeating: 0, count: edges.count - 1)
        var binAcc = [Int](repeating: 0, count: edges.count - 1)
        var steps = 0
        for _ in 0 ..< N {
            let dl = head(lastH, uArr, cache: mtpKV)                       // [1,1,V]
            let logitsArr = dl[0, 0].asType(.float32).asArray(Float.self)  // [V] CPU
            // top-K（K 小ゆえ K パス選択）
            var topk: [Int] = []; var used = Set<Int>()
            for _ in 0 ..< K {
                var best = -1; var bestV = -Float.greatestFiniteMagnitude
                for i in 0 ..< logitsArr.count where !used.contains(i) {
                    if logitsArr[i] > bestV { bestV = logitsArr[i]; best = i }
                }
                topk.append(best); used.insert(best)
            }
            let mx = logitsArr.max()!
            var z: Float = 0; for v in logitsArr { z += exp(v - mx) }
            let top1Prob = exp(logitsArr[topk[0]] - mx) / z
            // true next token
            let (H2, lg2) = try model.forwardHidden(uArr, caches: mc)
            let trueNext = MLX.argMax(lg2[0, 0], axis: -1).item(Int.self)
            for j in 0 ..< K where topk[0 ... j].contains(trueNext) { covK[j] += 1 }
            let acc1 = (topk[0] == trueNext) ? 1 : 0
            for b in 0 ..< (edges.count - 1) where top1Prob >= edges[b] && top1Prob < edges[b + 1] {
                binTot[b] += 1; binAcc[b] += acc1; break
            }
            steps += 1
            uArr = MLXArray([Int32(trueNext)], [1, 1]); lastH = H2[0..., 0 ..< 1]
            MLX.eval([uArr, lastH] + [mtpKV.keys, mtpKV.values].compactMap { $0 } + mc.flatMap { $0.stateArrays })
        }
        let s = Double(steps)
        var cov = "top-K coverage(=width-k tree accept 天井): "
        for j in 0 ..< K { cov += String(format: "≤top%d=%.1f%%  ", j + 1, Double(covK[j]) / s * 100) }
        var corr = "confidence(top1prob)→accept@1: "
        for b in 0 ..< (edges.count - 1) where binTot[b] > 0 {
            corr += String(format: "[%.2f-%.2f]:%.0f%%(n%d)  ", edges[b], edges[b + 1] > 1 ? 1.0 : edges[b + 1],
                           Double(binAcc[b]) / Double(binTot[b]) * 100, binTot[b])
        }
        return "[MTPDraftCalib C=\(C), N=\(steps)]\n  \(cov)\n  \(corr)"
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
        GatedDeltaNetLayer.fuseGDN = Tell.envFlag("QWISP_FUSE_GDN")   // A3: GDN in_proj 4→1 融合の効果
        defer { GatedDeltaNetLayer.f32Conv = false; AttentionLayer.f32SDPA = false; GatedDeltaNetLayer.fuseGDN = false }
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
        let L0 = model.isLinearFlags.count                                   // 全層数
        // (A) L 依存（per-forward 固定費 + marginal compute）
        var lines: [String] = []
        for L in [1, 2, 4, 8, 16, 24] {
            let bc = model.makeCaches()
            _ = try model.prefillChunked(ids, caches: bc)
            MLX.eval(bc.flatMap { $0.stateArrays })
            let snaps = bc.map { $0.snapshot() }
            let seq = MLXArray(Array(repeating: Int32(100), count: L), [1, L])   // teacher-forced ダミー
            let (hw, _) = try model.forwardHidden(seq, caches: bc); MLX.eval([hw] + bc.flatMap { $0.stateArrays })
            for (i, c) in bc.enumerated() { c.restore(snaps[i], isLinear: isLin[i], trim: L) }
            LayerExpertCache.missTotal = 0
            var tAcc: UInt64 = 0
            for _ in 0 ..< reps {
                let t = now()
                let (h, _) = try model.forwardHidden(seq, caches: bc)
                MLX.eval([h] + bc.flatMap { $0.stateArrays })
                tAcc += now() - t
                for (i, c) in bc.enumerated() { c.restore(snaps[i], isLinear: isLin[i], trim: L) }   // 非計時
            }
            let ms = Double(tAcc) / Double(reps) / 1e6
            lines.append(String(format: "L=%2d: %6.2f ms  (%.2f ms/token)  misses/forward=%.1f",
                                L, ms, ms / Double(L), Double(LayerExpertCache.missTotal) / Double(reps)))
        }
        // (B) 有効層数依存（L=1）: forwardHiddenSkip で末尾層を skip → per-layer 固定費を抽出
        //   cost(active) ≈ const(embed+norm+head+launch床) + slope·active なら launch-chain 律速。
        var lines2: [String] = []
        for skipN in [0, 10, 20, 30] {
            let active = L0 - skipN
            let skip = Set((L0 - skipN) ..< L0)                              // 末尾 skipN 層を identity
            let bc = model.makeCaches()
            _ = try model.prefillChunked(ids, caches: bc)
            MLX.eval(bc.flatMap { $0.stateArrays })
            let snaps = bc.map { $0.snapshot() }
            let seq = MLXArray([Int32(100)], [1, 1])
            let (hw, _) = try model.forwardHiddenSkip(seq, caches: bc, skip: skip); MLX.eval([hw])
            for (i, c) in bc.enumerated() { c.restore(snaps[i], isLinear: isLin[i], trim: 1) }
            var tAcc: UInt64 = 0
            for _ in 0 ..< reps {
                let t = now()
                let (h, _) = try model.forwardHiddenSkip(seq, caches: bc, skip: skip)
                MLX.eval([h] + bc.flatMap { $0.stateArrays })
                tAcc += now() - t
                for (i, c) in bc.enumerated() { c.restore(snaps[i], isLinear: isLin[i], trim: 1) }
            }
            let ms = Double(tAcc) / Double(reps) / 1e6
            lines2.append(String(format: "active=%2d 層: %6.2f ms  (%.3f ms/層)", active, ms, ms / Double(Swift.max(1, active))))
        }
        return "[ForwardCost C=\(C), f32-full, hot-pin top-C]\n  (A) L依存:\n  "
            + lines.joined(separator: "\n  ")
            + "\n  (B) 有効層数依存(L=1, 末尾skip):\n  " + lines2.joined(separator: "\n  ")
    }

    /// **[計測] issue#3 §5: forward 50ms 床は dispatch-latency 律速か GPU-exec 床か（二値判定）**
    /// 機構 = 独立 forward パイプライン法。K 個の **データ依存の無い** L=1 forward を 1 回の eval に
    /// まとめて投入し、per-forward wall が K で下がるかを測る。単 forward が dispatch 律速（40 層の
    /// 逐次カーネル投入の間 GPU がアイドル）なら、独立 forward を束ねるとその idle 隙間が埋まり
    /// per-forward wall が **下がる**＝CPU submit が GPU を待たせている証拠。GPU が既に飽和（exec 床）
    /// なら K を増やしても **flat**＝50ms は本物の GPU-exec 床で graph-capture も無駄。GPU counter 不要・
    /// mlx 再ビルド不要でクリーンな二値数値が出る（RNN-T「GPU 80% idle」測定の型）。
    /// build-only(lazy 構築) vs eval(submit+exec) も分離して CPU-record 寄与も見る。
    /// hot-pin top-C で IO 排除。QWISP_GPUTRACE=path を渡すと K=1 区間を .gputrace capture（要 mlx
    /// MLX_METAL_DEBUG ビルド + MTL_CAPTURE_ENABLED=1。無ければ Instruments の Metal System Trace で代替可）。
    /// - env: QWISP_RUN=forward-gpu-busy / QWISP_CACHE_C / QWISP_FC_REPS(既定20) /
    ///        QWISP_GPUBUSY_K(既定"1,2,4,8") / QWISP_GPUTRACE(任意, capture 出力パス)
    public static func runForwardGpuBusy(modelDir: String, refPath: String) throws -> String {
        guard let device = MTLCreateSystemDefaultDevice() else { return "ERROR: no Metal device" }
        let r = try loadArrays(url: URL(fileURLWithPath: refPath))
        guard let promptArr = r["spec_prompt"] else { return "[GpuBusy] skip" }
        let C = Tell.envInt("QWISP_CACHE_C", 64)
        let reps = Tell.envInt("QWISP_FC_REPS", 20)
        let Ks = (ProcessInfo.processInfo.environment["QWISP_GPUBUSY_K"] ?? "1,2,4,8")
            .split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
        let tracePath = ProcessInfo.processInfo.environment["QWISP_GPUTRACE"]
        GatedDeltaNetLayer.f32Conv = true; AttentionLayer.f32SDPA = true
        GatedDeltaNetLayer.fuseRMSGated = Tell.envFlag("QWISP_FUSE_RMS")   // issue#5: RMSNormGated 融合
        defer { GatedDeltaNetLayer.f32Conv = false; AttentionLayer.f32SDPA = false; GatedDeltaNetLayer.fuseRMSGated = false }
        let store = try WeightStore(modelDir: modelDir); store.residentNonExperts()
        let source = try ExpertSource(modelDir: modelDir); try source.warm()
        let arena = try ExpertArena(device: device, source: source, N: 64)
        let model = try StreamingQwispModel(store: store, arena: arena, device: device, source: source, cacheC: C)
        let ids = promptArr.asType(.int32).reshaped([1, promptArr.dim(0)])
        let isLin = model.isLinearFlags
        let nE = 256, nMoE = model.expertCaches.count
        // calib + hot-pin top-C で IO 排除（pure compute を測る）— runForwardCost と同型
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
        // プロセス全体の累積 CPU 時間(user+sys, 全スレッド合算)。wall と比べて cores-busy 比を出す。
        // cores-busy ≈ 1+ なら CPU が wall を占有(=encode/sync 律速=dispatch 系)、≈0 なら CPU は GPU 待ちで block(=GPU-exec 床)。
        func cpuNs() -> UInt64 {
            var ru = rusage()
            getrusage(RUSAGE_SELF, &ru)
            let u = UInt64(ru.ru_utime.tv_sec) * 1_000_000_000 + UInt64(ru.ru_utime.tv_usec) * 1000
            let s = UInt64(ru.ru_stime.tv_sec) * 1_000_000_000 + UInt64(ru.ru_stime.tv_usec) * 1000
            return u + s
        }
        let seq = MLXArray([Int32(100)], [1, 1])                                  // L=1 ダミー decode
        // 各 K 用に独立した cache 群（データ依存なし）を prefill しておく
        let maxK = Ks.max() ?? 1
        var bcs: [[LayerCache]] = []
        var snapss: [[LayerCache.Snapshot]] = []
        for _ in 0 ..< maxK {
            let bc = model.makeCaches()
            _ = try model.prefillChunked(ids, caches: bc)
            MLX.eval(bc.flatMap { $0.stateArrays })
            bcs.append(bc); snapss.append(bc.map { $0.snapshot() })
        }
        // issue#5 trace 専用: no-sync forward を reps 回だけ tight loop（build-only/B-section を飛ばし
        //   Metal trace を短く清浄に保つ）。QWISP_GPUBUSY_TRACEONLY=1 + C=256(全常駐 exact)で使う。
        if Tell.envFlag("QWISP_GPUBUSY_TRACEONLY") {
            StreamingMoEBlock.probeNoSync = true
            var tAcc: UInt64 = 0; let cpu0 = cpuNs()
            for _ in 0 ..< reps {
                let t = now()
                let (h, _) = try model.forwardHidden(seq, caches: bcs[0])
                MLX.eval([h] + bcs[0].flatMap { $0.stateArrays })
                tAcc += now() - t
                for (i, c) in bcs[0].enumerated() { c.restore(snapss[0][i], isLinear: isLin[i], trim: 1) }
            }
            StreamingMoEBlock.probeNoSync = false
            let ms = Double(tAcc) / Double(reps) / 1e6
            let cores = Double(cpuNs() - cpu0) / Double(tAcc)
            return String(format: "[GpuBusy TRACEONLY no-sync C=\(C)] %.2f ms/forward, CPU-busy=%.2f cores (reps=\(reps))\n"
                + "  → Metal trace 記録対象。GPU-busy%% は tools/gpu_busy_from_trace.py で算出", ms, cores)
        }
        // issue#5 de-risk: 有効層数スイープ(forwardHiddenSkip)で no-sync forward の wall/CPU-busy が
        //   dispatch 数(≈層数)に比例するか測る。比例＆CPU-busy≈1 一定なら「dispatch を減らす=CPU-encode を
        //   減らす」が成立し mega-kernel fusion の ROI 天井が見積れる。QWISP_GPUBUSY_SCALING=1 + C=256。
        if Tell.envFlag("QWISP_GPUBUSY_SCALING") {
            StreamingMoEBlock.probeNoSync = true
            let L0 = model.isLinearFlags.count
            var rows: [String] = []
            var pts: [(a: Int, ms: Double)] = []
            for active in [8, 16, 24, 32, L0] {
                let skip = Set((active) ..< L0)
                // warmup
                let (hw, _) = try model.forwardHiddenSkip(seq, caches: bcs[0], skip: skip); MLX.eval([hw])
                for (i, c) in bcs[0].enumerated() { c.restore(snapss[0][i], isLinear: isLin[i], trim: 1) }
                var tAcc: UInt64 = 0; let cpu0 = cpuNs()
                for _ in 0 ..< reps {
                    let t = now()
                    let (h, _) = try model.forwardHiddenSkip(seq, caches: bcs[0], skip: skip)
                    MLX.eval([h] + bcs[0].flatMap { $0.stateArrays })
                    tAcc += now() - t
                    for (i, c) in bcs[0].enumerated() { c.restore(snapss[0][i], isLinear: isLin[i], trim: 1) }
                }
                let ms = Double(tAcc) / Double(reps) / 1e6
                let cores = Double(cpuNs() - cpu0) / Double(tAcc)
                pts.append((active, ms))
                rows.append(String(format: "  active=%2d 層: %6.2f ms  (%.3f ms/層, CPU-busy=%.2f cores)", active, ms, ms / Double(active), cores))
            }
            StreamingMoEBlock.probeNoSync = false
            // 線形 fit ms = a + b·active（a=層非依存 embed+norm+head, b=per-layer encode+exec）
            let n = Double(pts.count)
            let sx = pts.reduce(0.0) { $0 + Double($1.a) }, sy = pts.reduce(0.0) { $0 + $1.ms }
            let sxx = pts.reduce(0.0) { $0 + Double($1.a * $1.a) }, sxy = pts.reduce(0.0) { $0 + Double($1.a) * $1.ms }
            let b = (n * sxy - sx * sy) / (n * sxx - sx * sx), a = (sy - b * sx) / n
            return "[GpuBusy SCALING no-sync C=\(C)] issue#5 mega-fusion ROI de-risk\n"
                + rows.joined(separator: "\n")
                + String(format: "\n  線形fit: ms ≈ %.2f(層非依存) + %.3f·活性層数。per-layer = %.3f ms。", a, b, b)
                + "\n  → CPU-busy≈1.0 一定かつ ms が層数に比例なら、dispatch 削減=encode 削減が成立"
                + "（mega-fusion で per-layer op 数を 1/k に → b の encode 寄与が ~1/k に縮む見込み）"
        }
        // issue#5: サブレイヤ別内訳（融合ターゲット特定）。profileLayers の barrier 計時ゆえ絶対値は
        //   inflate するが相対 share は op-cost/encode 比率の目安。QWISP_GPUBUSY_SUBPROF=1 + C=256。
        if Tell.envFlag("QWISP_GPUBUSY_SUBPROF") {
            StreamingMoEBlock.probeNoSync = true
            StreamingMoEBlock.profileLayers = true
            let S = StreamingMoEBlock.self
            S.tNorm = 0; S.tGDN = 0; S.tAttn = 0; S.tMoEgather = 0; S.tMoEshared = 0
            S.tGdnInproj = 0; S.tGdnConv = 0; S.tGdnKernel = 0; S.tGdnOut = 0
            for _ in 0 ..< reps {
                let (h, _) = try model.forwardHidden(seq, caches: bcs[0]); MLX.eval([h])
                for (i, c) in bcs[0].enumerated() { c.restore(snapss[0][i], isLinear: isLin[i], trim: 1) }
            }
            StreamingMoEBlock.profileLayers = false; StreamingMoEBlock.probeNoSync = false
            let parts: [(String, UInt64)] = [
                ("GDN(linear層)", S.tGDN), ("attention(full層)", S.tAttn),
                ("MoE-gather", S.tMoEgather), ("MoE-shared", S.tMoEshared), ("norm×2", S.tNorm)]
            let tot = Double(parts.reduce(0) { $0 + $1.1 })
            let plines = parts.map { String(format: "  %-16@ %5.1f%%  (%.1f ms/forward)", $0.0,
                                            Double($0.1) / Swift.max(1, tot) * 100, Double($0.1) / Double(reps) / 1e6) }
            let gdnTot = Double(S.tGdnInproj + S.tGdnConv + S.tGdnKernel + S.tGdnOut)
            let gline = String(format: "  └GDN内訳: in_proj %.0f%% / conv %.0f%% / recurrent-kernel %.0f%% / out %.0f%%",
                               Double(S.tGdnInproj) / Swift.max(1, gdnTot) * 100, Double(S.tGdnConv) / Swift.max(1, gdnTot) * 100,
                               Double(S.tGdnKernel) / Swift.max(1, gdnTot) * 100, Double(S.tGdnOut) / Swift.max(1, gdnTot) * 100)
            return "[GpuBusy SUBPROF no-sync C=\(C), barrier計時=相対share用] issue#5 融合ターゲット\n"
                + plines.joined(separator: "\n") + "\n" + gline
                + "\n  → share 最大のサブレイヤが mega-fusion の第一候補（op 数多＝encode 寄与大）"
        }
        // build-only(lazy 構築のみ, eval せず) の CPU-record 寄与を 1 forward で測る
        var buildAcc: UInt64 = 0
        for _ in 0 ..< reps {
            let t = now()
            let (h, _) = try model.forwardHidden(seq, caches: bcs[0])     // lazy: ops 記録のみ
            _ = h
            buildAcc += now() - t
            for (i, c) in bcs[0].enumerated() { c.restore(snapss[0][i], isLinear: isLin[i], trim: 1) }
        }
        let buildMs = Double(buildAcc) / Double(reps) / 1e6
        // (A) パイプライン法: K 独立 forward を 1 eval、per-forward wall を測る
        //   issue#5: QWISP_GPUBUSY_NOSYNC=1 で K-loop を no-sync 化（要 C=256 全常駐で exact）。
        //   この loop を Metal System Trace で記録すると no-sync forward の GPU-busy% が取れる（capture 回収天井の go/no-go）。
        let traceNoSync = Tell.envFlag("QWISP_GPUBUSY_NOSYNC")
        if traceNoSync { StreamingMoEBlock.probeNoSync = true }
        defer { StreamingMoEBlock.probeNoSync = false }
        var lines: [String] = []
        var perK: [(K: Int, ms: Double, cores: Double)] = []
        for K in Ks {
            // warmup
            do {
                var hs: [MLXArray] = []
                for k in 0 ..< K { let (h, _) = try model.forwardHidden(seq, caches: bcs[k]); hs.append(h) }
                MLX.eval(hs + (0 ..< K).flatMap { bcs[$0].flatMap { $0.stateArrays } })
                for k in 0 ..< K { for (i, c) in bcs[k].enumerated() { c.restore(snapss[k][i], isLinear: isLin[i], trim: 1) } }
            }
            LayerExpertCache.missTotal = 0
            var tAcc: UInt64 = 0
            let doTrace = tracePath != nil && K == 1
            if doTrace, let tp = tracePath { MLX.GPU.startCapture(url: URL(fileURLWithPath: tp)) }
            let cpu0 = cpuNs()
            for _ in 0 ..< reps {
                let t = now()
                var hs: [MLXArray] = []
                for k in 0 ..< K { let (h, _) = try model.forwardHidden(seq, caches: bcs[k]); hs.append(h) }
                MLX.eval(hs + (0 ..< K).flatMap { bcs[$0].flatMap { $0.stateArrays } })   // K 本まとめて submit
                tAcc += now() - t
                for k in 0 ..< K { for (i, c) in bcs[k].enumerated() { c.restore(snapss[k][i], isLinear: isLin[i], trim: 1) } }   // 非計時(CPU はここも含むが小)
            }
            let cpuDelta = cpuNs() - cpu0
            if doTrace, let tp = tracePath { MLX.GPU.stopCapture(url: URL(fileURLWithPath: tp)) }
            let perFwd = Double(tAcc) / Double(reps) / Double(K) / 1e6
            let cores = Double(cpuDelta) / Double(tAcc)                       // CPU時間 / 計時 wall = 平均ビジー core 数
            perK.append((K, perFwd, cores))
            lines.append(String(format: "K=%d: %6.2f ms/forward  (batch wall %6.2f ms, CPU-busy=%.2f cores, misses/fwd=%.1f)%@",
                                K, perFwd, perFwd * Double(K), cores,
                                Double(LayerExpertCache.missTotal) / Double(reps) / Double(K),
                                doTrace ? "  [gputrace→\(tracePath!)]" : ""))
        }
        // (B) round-trip 除去の天井: sync(probeNoSync=false, 毎層 inds.asArray+ensure) vs
        //     no-sync(GPU slot-table remap, 毎層 CPU materialize 無し)を同一 resident 状態で A/B。
        //     hot-pin top-C に routed ⊂ なので no-sync gather は exact 経路と bit 一致のはず（lossless 確認込み）。
        StreamingMoEBlock.captureInds = false; StreamingMoEBlock.captureK = 0
        StreamingMoEBlock.marginK = 0; StreamingMoEBlock.skipMode = 0
        StreamingMoEBlock.countHotMiss = false; StreamingMoEBlock.syncLayers = nil
        func measureMode(_ noSync: Bool) throws -> (ms: Double, cores: Double, h: MLXArray) {
            StreamingMoEBlock.probeNoSync = noSync
            // warmup + 代表 hidden を1本確保（bit 比較用）
            let (hw, _) = try model.forwardHidden(seq, caches: bcs[0]); MLX.eval([hw])
            for (i, c) in bcs[0].enumerated() { c.restore(snapss[0][i], isLinear: isLin[i], trim: 1) }
            var tAcc: UInt64 = 0
            let cpu0 = cpuNs()
            for _ in 0 ..< reps {
                let t = now()
                let (h, _) = try model.forwardHidden(seq, caches: bcs[0])
                MLX.eval([h] + bcs[0].flatMap { $0.stateArrays })
                tAcc += now() - t
                for (i, c) in bcs[0].enumerated() { c.restore(snapss[0][i], isLinear: isLin[i], trim: 1) }
            }
            let cores = Double(cpuNs() - cpu0) / Double(tAcc)
            return (Double(tAcc) / Double(reps) / 1e6, cores, hw)
        }
        let mSync = try measureMode(false)
        let mNo = try measureMode(true)
        StreamingMoEBlock.probeNoSync = false
        let maxAbs = MLX.abs(mSync.h - mNo.h).max().item(Float.self)         // lossless 確認(0=bit一致)
        let speedup = mNo.ms > 0 ? mSync.ms / mNo.ms : 1.0
        let abLines = String(format:
            "  sync   (毎層 materialize+ensure): %6.2f ms/forward  CPU-busy=%.2f cores\n  "
            + "no-sync(GPU slot-table remap)  : %6.2f ms/forward  CPU-busy=%.2f cores\n  "
            + "→ round-trip 除去 speedup=%.2fx, max|Δhidden|=%.2e (%@)",
            mSync.ms, mSync.cores, mNo.ms, mNo.cores, speedup, maxAbs,
            maxAbs < 1e-3 ? "resident で bit一致=lossless" : "差あり(routed が hot-pin 外=要 resident 化)")
        // 二値判定(主=CPU-busy, 副=pipeline ratio):
        //   CPU-busy ≳1.0 → wall を CPU が占有=encode/routing-sync 律速(dispatch 系) → graph-capture/sync削減に価値
        //   CPU-busy ≲0.3 → CPU は GPU 待ちで block=GPU-exec 床 → 50ms は本物の compute 床、graph capture も無駄
        let base = perK.first?.ms ?? 0
        let best = perK.map { $0.ms }.min() ?? base
        let ratio = base > 0 ? best / base : 1.0
        let cores = perK.first?.cores ?? 0
        let verdict: String
        if cores >= 0.8 {
            verdict = String(format: "DISPATCH/SYNC 律速 (CPU-busy=%.2f cores≒wall占有, pipeline %.2fx). "
                + "wall は GPU exec でなく CPU の encode/per-layer routing 同期律速。"
                + "→ MoE routing を含むため native graph-capture は困難だが、sync 削減(routing 予測でround-trip除去)に価値。"
                + "「fundamental physics」でなく tooling/sync 床=issue#3 §4 の framing 修正が正しい", cores, ratio)
        } else if cores <= 0.3 {
            verdict = String(format: "GPU-EXEC 床 (CPU-busy=%.2f cores=GPU待ちで block, pipeline %.2fx flat). "
                + "→ 50ms は本物の GPU compute 床、graph capture も無駄。penta の「床」が正しい", cores, ratio)
        } else {
            verdict = String(format: "混在 (CPU-busy=%.2f cores, pipeline %.2fx). CPU sync と GPU exec が拮抗。"
                + "gputrace で GPU 占有率を裏取りして確定", cores, ratio)
        }
        return "[GpuBusy C=\(C), f32-full, hot-pin top-C, reps=\(reps)] issue#3 §5 dispatch vs exec\n"
            + String(format: "  build-only(eval せず forwardHidden のみ; lazy なら~0 のはず): %.2f ms/forward\n", buildMs)
            + "  (A) 独立 forward パイプライン (per-forward wall + CPU-busy cores):\n  " + lines.joined(separator: "\n  ")
            + "\n  (B) sync vs no-sync round-trip 除去 A/B (同一 resident):\n" + abLines
            + "\n  判定: " + verdict
            + (tracePath == nil
                ? "\n  ※ gputrace 裏取り: QWISP_GPUTRACE=/path/x.gputrace を渡す(要 mlx MLX_METAL_DEBUG ビルド)、"
                  + "または Instruments の Metal System Trace で本バイナリの GPU 占有率を直接読む"
                : "")
    }

    /// **[高速化] issue#3 lever-1: round-trip 除去で resident greedy を ~2x lossless 化**
    /// forward-gpu-busy の (B) A/B が「全 expert resident なら no-sync gather は exact 経路と bit 一致で 2x」
    /// と forward 単位で示した。本 runner はそれを **エンドツーエンドの実 greedy decode** で確認する:
    /// 同一プロンプトから N トークン greedy を sync / no-sync で各 1 回回し、(1)生成トークン完全一致(lossless)、
    /// (2)tok/s を比較。C=256(全 expert 常駐)なら gpuSlotTable が全 expert を持ち alias 皆無＝無条件 exact。
    /// C<256 では routed が cold に当たると slot-0 alias で発散(その divergent token 数も報告)。
    /// - env: QWISP_RUN=nosync-resident / QWISP_CACHE_C(既定256) / QWISP_GEN(既定64) / QWISP_CALIB(既定48)
    public static func runNoSyncResident(modelDir: String, refPath: String) throws -> String {
        guard let device = MTLCreateSystemDefaultDevice() else { return "ERROR: no Metal device" }
        let r = try loadArrays(url: URL(fileURLWithPath: refPath))
        guard let promptArr = r["spec_prompt"] else { return "[NoSyncResident] skip" }
        let C = Tell.envInt("QWISP_CACHE_C", 256)
        let N = Tell.envInt("QWISP_GEN", 64)
        let calibN = Tell.envInt("QWISP_CALIB", 48)
        GatedDeltaNetLayer.f32Conv = true; AttentionLayer.f32SDPA = true
        defer { GatedDeltaNetLayer.f32Conv = false; AttentionLayer.f32SDPA = false; StreamingMoEBlock.probeNoSync = false }
        let store = try WeightStore(modelDir: modelDir); store.residentNonExperts()
        let source = try ExpertSource(modelDir: modelDir); try source.warm()
        let arena = try ExpertArena(device: device, source: source, N: 64)
        let model = try StreamingQwispModel(store: store, arena: arena, device: device, source: source, cacheC: C)
        let ids = promptArr.asType(.int32).reshaped([1, promptArr.dim(0)])
        let isLin = model.isLinearFlags
        let nE = 256, nMoE = model.expertCaches.count
        // calib + hot-pin top-C（C=256 なら全 expert を resident 化）
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
        var residentExperts = 0
        for (mi, ec) in model.expertCaches.enumerated() {
            let pin = Array(counts[mi].enumerated()
                .sorted { $0.element != $1.element ? $0.element > $1.element : $0.offset < $1.offset }
                .prefix(C).map { $0.offset })
            _ = ec.ensure(pin); residentExperts += pin.count
        }
        let avgResident = residentExperts / Swift.max(1, nMoE)
        // greedy 1 run（probeNoSync 指定）。tokens と tok/s を返す。fresh cache（expert 常駐は model 側で持続）。
        func greedy(_ noSync: Bool) throws -> (toks: [Int], tps: Double) {
            let mc = model.makeCaches()
            var (_, lg) = try model.prefillChunked(ids, caches: mc)
            var u = MLX.argMax(lg[0, lg.dim(1) - 1], axis: -1).reshaped([1, 1])
            MLX.eval([u] + mc.flatMap { $0.stateArrays })
            // warmup 1（kernel/グラフ確定を計時外に）。snapshot を warmup 前に取り、cache を post-prefill へ巻戻す。
            StreamingMoEBlock.probeNoSync = noSync
            let snaps0 = mc.map { $0.snapshot() }
            let (_, wlg) = try model.forwardHidden(u, caches: mc); MLX.eval([wlg])
            for (i, c) in mc.enumerated() { c.restore(snaps0[i], isLinear: isLin[i], trim: 1) }
            var toks: [Int] = []
            let t0 = DispatchTime.now().uptimeNanoseconds
            for _ in 0 ..< N {
                (_, lg) = try model.forwardHidden(u, caches: mc)
                u = MLX.argMax(lg[0, 0], axis: -1).reshaped([1, 1])
                MLX.eval([u] + mc.flatMap { $0.stateArrays })
                toks.append(u.item(Int.self))
            }
            StreamingMoEBlock.probeNoSync = false
            let secs = Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e9
            return (toks, Double(N) / secs)
        }
        let sync = try greedy(false)      // 参照=exact
        let nosync = try greedy(true)
        let match = zip(sync.toks, nosync.toks).filter { $0 == $1 }.count
        // 先頭発散位置（lossless なら N、途中で alias 混入なら < N）
        var firstDiv = N
        for i in 0 ..< N where sync.toks[i] != nosync.toks[i] { firstDiv = i; break }
        let speedup = sync.tps > 0 ? nosync.tps / sync.tps : 0
        let losslessTag = match == N ? "✅ lossless(完全一致)"
            : "⚠️ \(N - match)/\(N) 発散(先頭 token#\(firstDiv))=C<256 で routed が cold alias。要 full-resident or 予測"
        return "[NoSyncResident C=\(C), avg resident=\(avgResident)/256 experts/層, N=\(N), f32-full]"
            + " issue#3 lever-1 round-trip 除去\n"
            + String(format: "  sync   (exact, 毎層 materialize+ensure): %.1f tok/s\n", sync.tps)
            + String(format: "  no-sync(GPU slot-table remap)          : %.1f tok/s\n", nosync.tps)
            + String(format: "  → speedup=%.2fx, token一致=%d/%d  %@", speedup, match, N, losslessTag)
    }

    /// **[高速化] issue#3 lever-1 製品化: 16GB(C=128) を no-sync + miss-escalation で真 lossless 化**
    /// C<256 では no-sync gather が cold routed を slot-0 alias=garbage 化するが、`countHotMiss` で
    /// その token の cold route 数を層横断 GPU 累積し、**hotMiss=0 の token は no-sync 結果が bit-exact
    /// ゆえ採用、hotMiss>0 の token だけ cache を巻戻して sync 再計算(exact, cold expert を ensure)**。
    /// 検出は cold route の厳密 superset ゆえ出力は pure-sync と完全一致(真 lossless)。miss-check は token
    /// 毎 1 scalar drain のみ(40 層 materialize より遥かに安い)。期待: 16GB を ~45-57 tok/s exact。
    /// - env: QWISP_RUN=nosync-escalate / QWISP_CACHE_C(既定128) / QWISP_GEN(既定64) / QWISP_CALIB(既定48)
    public static func runNoSyncEscalate(modelDir: String, refPath: String) throws -> String {
        guard let device = MTLCreateSystemDefaultDevice() else { return "ERROR: no Metal device" }
        let r = try loadArrays(url: URL(fileURLWithPath: refPath))
        guard let promptArr = r["spec_prompt"] else { return "[NoSyncEscalate] skip" }
        let C = Tell.envInt("QWISP_CACHE_C", 128)
        let N = Tell.envInt("QWISP_GEN", 64)
        let calibN = Tell.envInt("QWISP_CALIB", 48)
        GatedDeltaNetLayer.f32Conv = true; AttentionLayer.f32SDPA = true
        defer {
            GatedDeltaNetLayer.f32Conv = false; AttentionLayer.f32SDPA = false
            StreamingMoEBlock.probeNoSync = false; StreamingMoEBlock.countHotMiss = false
            StreamingMoEBlock.hotMissAccum = nil
        }
        let store = try WeightStore(modelDir: modelDir); store.residentNonExperts()
        let source = try ExpertSource(modelDir: modelDir); try source.warm()
        let arena = try ExpertArena(device: device, source: source, N: 64)
        let model = try StreamingQwispModel(store: store, arena: arena, device: device, source: source, cacheC: C)
        let ids = promptArr.asType(.int32).reshaped([1, promptArr.dim(0)])
        let isLin = model.isLinearFlags
        let nE = 256, nMoE = model.expertCaches.count
        // calib + hot-pin top-C
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
        for (mi, ec) in model.expertCaches.enumerated() {
            _ = ec.ensure(Array(counts[mi].enumerated()
                .sorted { $0.element != $1.element ? $0.element > $1.element : $0.offset < $1.offset }
                .prefix(C).map { $0.offset }))
        }
        // 参照: pure-sync greedy
        func greedySync() throws -> (toks: [Int], tps: Double) {
            StreamingMoEBlock.probeNoSync = false; StreamingMoEBlock.countHotMiss = false
            let mc = model.makeCaches()
            var (_, lg) = try model.prefillChunked(ids, caches: mc)
            var u = MLX.argMax(lg[0, lg.dim(1) - 1], axis: -1).reshaped([1, 1])
            MLX.eval([u] + mc.flatMap { $0.stateArrays })
            let s0 = mc.map { $0.snapshot() }
            (_, lg) = try model.forwardHidden(u, caches: mc); MLX.eval([lg])     // warmup
            for (i, c) in mc.enumerated() { c.restore(s0[i], isLinear: isLin[i], trim: 1) }
            var toks: [Int] = []; let t0 = DispatchTime.now().uptimeNanoseconds
            for _ in 0 ..< N {
                (_, lg) = try model.forwardHidden(u, caches: mc)
                u = MLX.argMax(lg[0, 0], axis: -1).reshaped([1, 1])
                MLX.eval([u] + mc.flatMap { $0.stateArrays }); toks.append(u.item(Int.self))
            }
            return (toks, Double(N) / (Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e9))
        }
        // no-sync + escalation greedy
        func greedyEscalate() throws -> (toks: [Int], tps: Double, escal: Int) {
            let mc = model.makeCaches()
            var (_, lg) = try model.prefillChunked(ids, caches: mc)
            var u = MLX.argMax(lg[0, lg.dim(1) - 1], axis: -1).reshaped([1, 1])
            MLX.eval([u] + mc.flatMap { $0.stateArrays })
            // warmup（両 mode のグラフを確定、cache 巻戻し）
            let s0 = mc.map { $0.snapshot() }
            StreamingMoEBlock.probeNoSync = true; StreamingMoEBlock.countHotMiss = true; StreamingMoEBlock.hotMissAccum = nil
            let (_, w1) = try model.forwardHidden(u, caches: mc); MLX.eval([w1])
            for (i, c) in mc.enumerated() { c.restore(s0[i], isLinear: isLin[i], trim: 1) }
            StreamingMoEBlock.probeNoSync = false; StreamingMoEBlock.countHotMiss = false
            let (_, w2) = try model.forwardHidden(u, caches: mc); MLX.eval([w2])
            for (i, c) in mc.enumerated() { c.restore(s0[i], isLinear: isLin[i], trim: 1) }
            var toks: [Int] = []; var escal = 0
            let t0 = DispatchTime.now().uptimeNanoseconds
            for _ in 0 ..< N {
                let snaps = mc.map { $0.snapshot() }
                StreamingMoEBlock.hotMissAccum = nil
                StreamingMoEBlock.probeNoSync = true; StreamingMoEBlock.countHotMiss = true
                let (_, lns) = try model.forwardHidden(u, caches: mc)
                let missArr = StreamingMoEBlock.hotMissAccum ?? MLXArray(Int32(0))
                MLX.eval([lns, missArr] + mc.flatMap { $0.stateArrays })           // token 毎 1 sync(scalar+logits)
                if missArr.item(Int32.self) == 0 {
                    u = MLX.argMax(lns[0, 0], axis: -1).reshaped([1, 1]); MLX.eval([u])   // 採用(cache exact)
                } else {
                    escal += 1
                    for (i, c) in mc.enumerated() { c.restore(snaps[i], isLinear: isLin[i], trim: 1) }
                    StreamingMoEBlock.probeNoSync = false; StreamingMoEBlock.countHotMiss = false
                    let (_, lex) = try model.forwardHidden(u, caches: mc)            // sync 再計算(exact, cold ensure)
                    u = MLX.argMax(lex[0, 0], axis: -1).reshaped([1, 1])
                    MLX.eval([u] + mc.flatMap { $0.stateArrays })
                }
                toks.append(u.item(Int.self))
            }
            StreamingMoEBlock.probeNoSync = false; StreamingMoEBlock.countHotMiss = false
            return (toks, Double(N) / (Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e9), escal)
        }
        let ref = try greedySync()
        let esc = try greedyEscalate()
        let match = zip(ref.toks, esc.toks).filter { $0 == $1 }.count
        let speedup = ref.tps > 0 ? esc.tps / ref.tps : 0
        let escRate = Double(esc.escal) / Double(N) * 100
        let tag = match == N ? "✅ 真 lossless(sync と完全一致)"
            : "❌ \(N - match)/\(N) 不一致=escalation 検出漏れ(バグ)"
        return "[NoSyncEscalate C=\(C), N=\(N), f32-full] issue#3 lever-1 16GB 安全化\n"
            + String(format: "  pure-sync(exact)         : %.1f tok/s\n", ref.tps)
            + String(format: "  no-sync+escalate         : %.1f tok/s  (escalation %d/%d=%.0f%%)\n",
                     esc.tps, esc.escal, N, escRate)
            + String(format: "  → speedup=%.2fx, token一致=%d/%d  %@", speedup, match, N, tag)
    }

    /// **[計測] 8GB exact-pipeline go/no-go: cross-layer 予測の per-layer/per-token hit 率**
    /// 「層跨ぎ予測 prefetch で routing round-trip を除去（bit-exact）」が成立するかの決め手を測る。
    /// 機構: 各 token を exact greedy で回し、各層の **真の top-8** を、前 token 同層 gate 入力(temporal, M2 signal)
    /// から予測した **top-w 集合**が覆うか判定。報告:
    ///   - per-expert recall(w=8): 真 top-8 のうち予測 top-8 に入る割合（memory の 82-84% と比較）。
    ///   - **per-layer all-8-hit 率**: その層の真 top-8 が全部 予測 top-w に入る確率（= その層が no-sync で exact に
    ///     回せる＝round-trip 不要な割合）。
    ///   - **per-token all-layers-hit 率**: 全層が hit する token の割合（= その token は全層 no-sync＝round-trip 完全ゼロ）。
    /// 高い w で per-token all-hit が高く、w<=C(=64) に収まれば exact-pipeline が成立（残り層だけ sync fallback）。
    /// - env: QWISP_RUN=cross-layer-hitrate / QWISP_CACHE_C(既定64) / QWISP_GEN(既定64)
    public static func runCrossLayerHitrate(modelDir: String, refPath: String) throws -> String {
        guard let device = MTLCreateSystemDefaultDevice() else { return "ERROR: no Metal device" }
        let r = try loadArrays(url: URL(fileURLWithPath: refPath))
        guard let promptArr = r["spec_prompt"] else { return "[CrossLayerHitrate] skip" }
        let C = Tell.envInt("QWISP_CACHE_C", 64)
        let N = Tell.envInt("QWISP_GEN", 64)
        let widths = [8, 16, 32, 48]
        GatedDeltaNetLayer.f32Conv = true; AttentionLayer.f32SDPA = true
        defer {
            GatedDeltaNetLayer.f32Conv = false; AttentionLayer.f32SDPA = false
            StreamingMoEBlock.captureGateInput = false; StreamingMoEBlock.captureInds = false
        }
        let store = try WeightStore(modelDir: modelDir); store.residentNonExperts()
        let source = try ExpertSource(modelDir: modelDir); try source.warm()
        let arena = try ExpertArena(device: device, source: source, N: 64)
        let model = try StreamingQwispModel(store: store, arena: arena, device: device, source: source, cacheC: C)
        let ids = promptArr.asType(.int32).reshaped([1, promptArr.dim(0)])
        let L = model.layerCount
        let caches = model.makeCaches()
        StreamingMoEBlock.probeNoSync = false
        StreamingMoEBlock.captureGateInput = true; StreamingMoEBlock.captureInds = true
        var (_, lg) = try model.prefillChunked(ids, caches: caches)
        var cur = MLX.argMax(lg[0, lg.dim(1) - 1], axis: -1).reshaped([1, 1])
        MLX.eval([cur] + caches.flatMap { $0.stateArrays }
            + model.expertCaches.compactMap { $0.lastGateInput })
        func capGate() -> [MLXArray] { (0 ..< L).map { model.expertCaches[$0].lastGateInput! } }
        var prevGate = capGate()
        // 集計
        var layerHit = [Int: Int](); var tokAllHit = [Int: Int]()
        for w in widths { layerHit[w] = 0; tokAllHit[w] = 0 }
        // 信号選択: temporal(前 token 同層 gate入力, full-token lead) / intra1(同 token 前層 gate入力, ~1層 lead)
        let sig = Tell.envStr("QWISP_SIGNAL", "temporal")
        var recallSum = 0.0; var lt = 0; var toks = 0
        for _ in 1 ..< N {
            // exact forward（真の routing + 各層 gate 入力を捕捉）
            let (_, lgt) = try model.forwardHidden(cur, caches: caches)
            MLX.eval([lgt] + caches.flatMap { $0.stateArrays }
                + model.expertCaches.compactMap { $0.lastInds }
                + model.expertCaches.compactMap { $0.lastGateInput })
            let true8 = (0 ..< L).map { Set(model.expertCaches[$0].lastInds!.asArray(Int32.self).map { Int($0) }) }
            let thisGate = capGate()
            // 予測元 signal（層毎）: forward 後に確定した入力で各幅予測（accuracy 計測）
            let srcGate: [MLXArray] = (0 ..< L).map { i in
                switch sig {
                case "intra1": return i > 0 ? thisGate[i - 1] : prevGate[i]   // 同 token 前層
                default:       return prevGate[i]                              // temporal
                }
            }
            var predByW = [Int: [Set<Int>]]()
            for w in widths {
                let preds = (0 ..< L).map { model.predictLayerIndsK($0, srcGate[$0], w) }
                MLX.eval(preds)
                predByW[w] = preds.map { Set($0.asArray(Int32.self).map { Int($0) }) }
            }
            // per-expert recall（w=8）
            for i in 0 ..< L {
                let inter = true8[i].intersection(predByW[8]![i]).count
                recallSum += Double(inter) / Double(Swift.max(1, true8[i].count))
            }
            // per-layer all-8-hit & per-token all-layers-hit（各幅）
            for w in widths {
                var allHit = true
                for i in 0 ..< L {
                    if true8[i].isSubset(of: predByW[w]![i]) { layerHit[w]! += 1 } else { allHit = false }
                }
                if allHit { tokAllHit[w]! += 1 }
            }
            lt += L; toks += 1
            prevGate = thisGate
            cur = MLX.argMax(lgt[0, 0], axis: -1).reshaped([1, 1]); MLX.eval([cur])
        }
        let recall = recallSum / Double(Swift.max(1, lt)) * 100
        var lines: [String] = []
        for w in widths {
            lines.append(String(format: "  w=%2d: per-layer all-8-hit=%5.1f%%  per-token all-layers-hit=%5.1f%%  (prefetch %d≤C=%d %@)",
                                w, Double(layerHit[w]!) / Double(Swift.max(1, lt)) * 100,
                                Double(tokAllHit[w]!) / Double(Swift.max(1, toks)) * 100,
                                w, C, w <= C ? "✓fit" : "✗"))
        }
        return "[CrossLayerHitrate C=\(C), N=\(N), signal=\(sig)] 8GB exact-pipeline go/no-go\n"
            + String(format: "  per-expert recall(w=8)=%.1f%% (memory 82-84% と比較), L=%d 層\n", recall, L)
            + lines.joined(separator: "\n")
            + "\n  判定: per-token all-layers-hit が高い w で exact-pipeline 有望(全層 no-sync=round-trip 0)。"
            + "低ければ per-layer fallback 頻発で sync 並みに縮退。"
    }

    /// **[高速化試作] 8GB exact-pipeline: per-layer 予測 prefetch + miss escalation の bit-exact 検証**
    /// `forwardHiddenPipeline` を実 greedy decode で回し、sync greedy と (1)token 完全一致(bit-exact)、
    /// (2)tok/s、(3)実 per-layer escalation 率 を比較。直列版ゆえ速度は async 化前の go/no-go 指標
    /// (escalation 率が低く bit-exact なら async overlap で round-trip 除去の勝算)。
    /// - env: QWISP_RUN=pipeline-exact / QWISP_CACHE_C(既定64) / QWISP_GEN(既定64) / QWISP_PREDICT_W(既定48) / QWISP_CALIB(48)
    public static func runPipelineExact(modelDir: String, refPath: String) throws -> String {
        guard let device = MTLCreateSystemDefaultDevice() else { return "ERROR: no Metal device" }
        let r = try loadArrays(url: URL(fileURLWithPath: refPath))
        guard let promptArr = r["spec_prompt"] else { return "[PipelineExact] skip" }
        let C = Tell.envInt("QWISP_CACHE_C", 64)
        let N = Tell.envInt("QWISP_GEN", 64)
        let predictW = Tell.envInt("QWISP_PREDICT_W", 48)
        let calibN = Tell.envInt("QWISP_CALIB", 48)
        GatedDeltaNetLayer.f32Conv = true; AttentionLayer.f32SDPA = true
        defer {
            GatedDeltaNetLayer.f32Conv = false; AttentionLayer.f32SDPA = false
            StreamingMoEBlock.probeNoSync = false; StreamingMoEBlock.countHotMiss = false
            StreamingMoEBlock.captureGateInput = false; StreamingMoEBlock.captureInds = false
        }
        let store = try WeightStore(modelDir: modelDir); store.residentNonExperts()
        let source = try ExpertSource(modelDir: modelDir); try source.warm()
        let arena = try ExpertArena(device: device, source: source, N: 64)
        let model = try StreamingQwispModel(store: store, arena: arena, device: device, source: source, cacheC: C)
        let ids = promptArr.asType(.int32).reshaped([1, promptArr.dim(0)])
        let isLin = model.isLinearFlags
        let nE = 256, nMoE = model.expertCaches.count
        // calib + hot-pin top-C（warm start）
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
        for (mi, ec) in model.expertCaches.enumerated() {
            _ = ec.ensure(Array(counts[mi].enumerated()
                .sorted { $0.element != $1.element ? $0.element > $1.element : $0.offset < $1.offset }
                .prefix(C).map { $0.offset }))
        }
        // 参照: sync greedy（exact, 先に実行して true token 列を確定）
        func greedySync() throws -> (toks: [Int], tps: Double) {
            StreamingMoEBlock.probeNoSync = false; StreamingMoEBlock.countHotMiss = false
            let mc = model.makeCaches()
            var (_, lg) = try model.prefillChunked(ids, caches: mc)
            var u = MLX.argMax(lg[0, lg.dim(1) - 1], axis: -1).reshaped([1, 1])
            MLX.eval([u] + mc.flatMap { $0.stateArrays })
            let s0 = mc.map { $0.snapshot() }
            (_, lg) = try model.forwardHidden(u, caches: mc); MLX.eval([lg])
            for (i, c) in mc.enumerated() { c.restore(s0[i], isLinear: isLin[i], trim: 1) }
            var toks: [Int] = []; let t0 = DispatchTime.now().uptimeNanoseconds
            for _ in 0 ..< N {
                (_, lg) = try model.forwardHidden(u, caches: mc)
                u = MLX.argMax(lg[0, 0], axis: -1).reshaped([1, 1])
                MLX.eval([u] + mc.flatMap { $0.stateArrays }); toks.append(u.item(Int.self))
            }
            return (toks, Double(N) / (Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e9))
        }
        // pipeline greedy（per-layer 予測 prefetch + miss escalation）
        func greedyPipeline() throws -> (toks: [Int], tps: Double, escPerTok: Double) {
            let mc = model.makeCaches()
            var (_, lg) = try model.prefillChunked(ids, caches: mc)
            var u = MLX.argMax(lg[0, lg.dim(1) - 1], axis: -1).reshaped([1, 1])
            MLX.eval([u] + mc.flatMap { $0.stateArrays })
            let s0 = mc.map { $0.snapshot() }
            let (_, wlg, _) = try model.forwardHiddenPipeline(u, caches: mc, predictW: predictW, isLin: isLin)
            MLX.eval([wlg])
            for (i, c) in mc.enumerated() { c.restore(s0[i], isLinear: isLin[i], trim: 1) }
            var toks: [Int] = []; var escTot = 0
            let t0 = DispatchTime.now().uptimeNanoseconds
            for _ in 0 ..< N {
                let (_, plg, esc) = try model.forwardHiddenPipeline(u, caches: mc, predictW: predictW, isLin: isLin)
                u = MLX.argMax(plg[0, 0], axis: -1).reshaped([1, 1])
                MLX.eval([u] + mc.flatMap { $0.stateArrays }); toks.append(u.item(Int.self)); escTot += esc
            }
            return (toks, Double(N) / (Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e9), Double(escTot) / Double(N))
        }
        let sync = try greedySync()
        let pipe = try greedyPipeline()
        let match = zip(sync.toks, pipe.toks).filter { $0 == $1 }.count
        var firstDiv = N; for i in 0 ..< N where sync.toks[i] != pipe.toks[i] { firstDiv = i; break }
        let speedup = sync.tps > 0 ? pipe.tps / sync.tps : 0
        let L = model.layerCount
        let tag = match == N ? "✅ bit-exact(sync と完全一致)"
            : "❌ \(N - match)/\(N) 不一致(先頭#\(firstDiv))=pipeline バグ(escalation 検出漏れ)"
        return "[PipelineExact C=\(C), N=\(N), predictW=\(predictW), intra1 signal, 直列(overlap無)] 8GB exact-pipeline\n"
            + String(format: "  sync     (exact)        : %.1f tok/s\n", sync.tps)
            + String(format: "  pipeline (予測prefetch) : %.1f tok/s  (escalation %.1f/%d 層=%.0f%%/token)\n",
                     pipe.tps, pipe.escPerTok, L, pipe.escPerTok / Double(L) * 100)
            + String(format: "  → speedup=%.2fx, token一致=%d/%d  %@\n", speedup, match, N, tag)
            + "  ※直列版ゆえ予測+ensure コスト込み。escalation 率が低く bit-exact なら async overlap で勝算"
    }

    /// cost-model 検証: 同一プロセスで (A)forward-cost→a,b fit (B)maxK-sweep の SuffixSpec 実測 を行い、
    /// 予測 tok/s=(1+p)/(a+b(D+1)) が実測と一致するか確認。dev 機は IO≈0 ゆえ forward+accept 項を検証。
    /// 一致なら cost-model で C/maxK/mode を予測選択して良い。ズレれば feedback/起動時実 sweep が要ると判る。
    /// - env: QWISP_RUN=cost-model-validate / QWISP_CACHE_C(既定128) / QWISP_GEN
    public static func runCostModelValidate(modelDir: String, refPath: String) throws -> String {
        guard let device = MTLCreateSystemDefaultDevice() else { return "ERROR: no Metal device" }
        let r = try loadArrays(url: URL(fileURLWithPath: refPath))
        guard let promptArr = r["spec_prompt"], let gRef = r["spec_greedy"] else { return "[CostModelValidate] skip" }
        let C = Tell.envInt("QWISP_CACHE_C", 128)
        GatedDeltaNetLayer.f32Conv = true; AttentionLayer.f32SDPA = true
        defer { GatedDeltaNetLayer.f32Conv = false; AttentionLayer.f32SDPA = false }
        let store = try WeightStore(modelDir: modelDir); store.residentNonExperts()
        let source = try ExpertSource(modelDir: modelDir); try source.warm()
        let arena = try ExpertArena(device: device, source: source, N: 64)
        let model = try StreamingQwispModel(store: store, arena: arena, device: device, source: source, cacheC: C)
        let ids = promptArr.asType(.int32).reshaped([1, promptArr.dim(0)])
        let gR = gRef.asArray(Int32.self).map { Int($0) }
        let isLin = model.isLinearFlags
        let N = Swift.min(Tell.envInt("QWISP_GEN", 48), gR.count)
        let nE = 256, nMoE = model.expertCaches.count
        func now() -> UInt64 { DispatchTime.now().uptimeNanoseconds }

        // calib + hot-pin top-C
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

        // (A) forward-cost L-sweep → a,b
        var fcPts: [(L: Int, ms: Double)] = []
        for L in [1, 2, 4, 8, 16, 24] {
            let bc = model.makeCaches(); _ = try model.prefillChunked(ids, caches: bc)
            MLX.eval(bc.flatMap { $0.stateArrays })
            let snaps = bc.map { $0.snapshot() }
            let seq = MLXArray(Array(repeating: Int32(100), count: L), [1, L])
            let (hw, _) = try model.forwardHidden(seq, caches: bc); MLX.eval([hw] + bc.flatMap { $0.stateArrays })
            for (i, c) in bc.enumerated() { c.restore(snaps[i], isLinear: isLin[i], trim: L) }
            var tAcc: UInt64 = 0
            for _ in 0 ..< 20 {
                let t = now(); let (h, _) = try model.forwardHidden(seq, caches: bc)
                MLX.eval([h] + bc.flatMap { $0.stateArrays }); tAcc += now() - t
                for (i, c) in bc.enumerated() { c.restore(snaps[i], isLinear: isLin[i], trim: L) }
            }
            fcPts.append((L, Double(tAcc) / 20 / 1e6))
        }
        let cm = CostModel.fit(fcPts)

        // (B) maxK-sweep SuffixSpec 実測（mix ref 前提=高 accept）
        func suffixRun(_ maxK: Int) throws -> (tokPerSec: Double, accept: Double, draftLen: Double) {
            var hist = ids.asArray(Int32.self).map { Int($0) }
            let mc = model.makeCaches()
            var (_, lg) = try model.prefillChunked(ids, caches: mc)
            var uArr = MLX.argMax(lg[0, lg.dim(1) - 1], axis: -1).reshaped([1, 1])
            MLX.eval([uArr] + mc.flatMap { $0.stateArrays })
            var out: [Int] = []; var steps = 0, accTok = 0, draftTot = 0
            let t0 = now()
            while out.count < N {
                steps += 1
                let u = uArr.item(Int.self)
                let drafts = Tell.suffixDraft(hist + [u], maxMatch: 32, draftK: maxK, minMatch: 2)
                let D = drafts.count; draftTot += D
                if D == 0 {
                    let (_, glg) = try model.forwardHidden(uArr, caches: mc)
                    out.append(u); hist.append(u)
                    uArr = MLX.argMax(glg[0, 0], axis: -1).reshaped([1, 1]); MLX.eval([uArr] + mc.flatMap { $0.stateArrays })
                    continue
                }
                let snaps = mc.map { $0.snapshot() }
                let seq = MLX.concatenated([uArr, MLXArray(drafts.map { Int32($0) }, [1, D])], axis: 1)
                let (_, vlg) = try model.forwardHidden(seq, caches: mc)
                let evals = MLX.argMax(vlg[0, 0 ..< (D + 1)], axis: -1).asArray(Int32.self).map { Int($0) }
                var p = 0; while p < D && drafts[p] == evals[p] { p += 1 }
                out.append(u); hist.append(u)
                for i in 0 ..< p { out.append(drafts[i]); hist.append(drafts[i]) }
                accTok += p
                if p == D {
                    uArr = MLXArray([Int32(evals[D])], [1, 1]); MLX.eval([uArr] + mc.flatMap { $0.stateArrays })
                } else {
                    for (i, c) in mc.enumerated() { c.restore(snaps[i], isLinear: isLin[i], trim: D + 1) }
                    let acc = [u] + Array(drafts.prefix(p))
                    _ = try model.forwardHidden(MLXArray(acc.map { Int32($0) }, [1, acc.count]), caches: mc)
                    uArr = MLXArray([Int32(evals[p])], [1, 1]); MLX.eval([uArr] + mc.flatMap { $0.stateArrays })
                }
            }
            let secs = Double(now() - t0) / 1e9
            return (Double(out.prefix(N).count) / secs, Double(accTok) / Double(steps), Double(draftTot) / Double(steps))
        }

        var rows: [String] = []
        var cVals: [Double] = []
        for maxK in [4, 8, 16, 24, 32] {
            // 3 回測って中央値（サーマルノイズ低減）
            var runs: [(Double, Double, Double)] = []
            for _ in 0 ..< 3 { runs.append(try suffixRun(maxK)) }
            let med = runs.sorted { $0.0 < $1.0 }[1]
            let (tps, accept, draftLen) = med
            let predNoC = cm.tokPerSec(draftLen: Int(draftLen.rounded()), acceptedPerStep: accept)
            let stepMs = (1.0 + accept) / tps * 1000.0                  // 実 step 時間
            let cImplied = stepMs - cm.forwardMs(Int(draftLen.rounded()) + 1)  // forward 超過分=per-step overhead
            cVals.append(cImplied)
            rows.append(String(format: "  maxK=%2d: 実測中央値 %.1f / 予測(c無) %.1f tok/s (誤差%+.0f%%)  accept=%.1f  step=%.0fms 内 forward=%.0f → 含意 c=%.0fms",
                               maxK, tps, predNoC, (tps - predNoC) / predNoC * 100, accept,
                               stepMs, cm.forwardMs(Int(draftLen.rounded()) + 1), cImplied))
        }
        // c を中央値で固定して再予測（c 項込みの当てはまり）
        let cMed = cVals.sorted()[cVals.count / 2]
        var rows2: [String] = []
        for (i, maxK) in [4, 8, 16, 24, 32].enumerated() {
            let parts = rows[i].split(separator: " ")
            _ = parts
            // 実測中央値は rows から再計算せず、c 込み予測のみ表示
            let dl = maxK   // mix 全受理ゆえ draftLen≈maxK
            let predC = (1.0 + Double(maxK)) / (cm.forwardMs(dl + 1) + cMed) * 1000.0
            rows2.append(String(format: "  maxK=%2d: 予測(c=%.0fms込) %.1f tok/s", maxK, cMed, predC))
        }
        let fcStr = fcPts.map { String(format: "L%d=%.1f", $0.L, $0.ms) }.joined(separator: " ")
        return String(format: """
            [CostModelValidate C=%d, 各maxK 3回中央値] forward-cost: %@
              fit: a=%.1fms b=%.2fms/tok  (forward_ms(L)=a+b·L)
            (1) 素の予測=(1+accept)/(a+b·(draftLen+1)) と含意 per-step overhead c:
            %@
            (2) c=中央値%.0fms を固定した c 項込み予測:
            %@
              → c が ~一定なら cost-model+c で予測選択 OK。c がバラつく/実測が非単調なら起動時実sweep か feedback が要
            """, C, fcStr, cm.a, cm.b, rows.joined(separator: "\n"), cMed, rows2.joined(separator: "\n"))
    }

    /// 起動時 calibration: device の RAM tier(mode/C/maxK) + cost-model(a,b,c)実測 を束ね DeviceConfig を完成。
    /// 製品の起動時に 1 回実行し、以降の予測/判定に使う。QWISP_DEVICE_RAM で他 device 模擬可。
    /// - env: QWISP_RUN=device-calibrate
    public static func runDeviceCalibrate(modelDir: String, refPath: String) throws -> String {
        guard let device = MTLCreateSystemDefaultDevice() else { return "ERROR: no Metal device" }
        let r = try loadArrays(url: URL(fileURLWithPath: refPath))
        guard let promptArr = r["spec_prompt"] else { return "[DeviceCalibrate] skip" }
        var cfg = DeviceCalibration.recommend()
        let C = cfg.C
        GatedDeltaNetLayer.f32Conv = true; AttentionLayer.f32SDPA = true
        defer { GatedDeltaNetLayer.f32Conv = false; AttentionLayer.f32SDPA = false }
        let store = try WeightStore(modelDir: modelDir); store.residentNonExperts()
        let source = try ExpertSource(modelDir: modelDir); try source.warm()
        let arena = try ExpertArena(device: device, source: source, N: 64)
        let model = try StreamingQwispModel(store: store, arena: arena, device: device, source: source, cacheC: C)
        let ids = promptArr.asType(.int32).reshaped([1, promptArr.dim(0)])
        let isLin = model.isLinearFlags
        let nE = 256, nMoE = model.expertCaches.count
        // calib + hot-pin top-C
        var counts = [[Int]](repeating: [Int](repeating: 0, count: nE), count: nMoE)
        let cc = model.makeCaches()
        StreamingMoEBlock.probeNoSync = false; StreamingMoEBlock.captureInds = true
        var (_, clg) = try model.prefillChunked(ids, caches: cc)
        var ccur = MLX.argMax(clg[0, clg.dim(1) - 1], axis: -1).reshaped([1, 1])
        MLX.eval([ccur] + cc.flatMap { $0.stateArrays })
        for _ in 0 ..< 32 {
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
        // cost-model 実測
        let cm = try DeviceCalibration.measureCostModel(model: model, ids: ids, isLin: isLin)
        cfg.costModel = cm
        // 予測: 高 accept(反復, p=maxK) と greedy 床(p=0)
        let predHigh = cm.tokPerSec(draftLen: cfg.maxK, acceptedPerStep: Double(cfg.maxK))
        let predFloor = cm.tokPerSec(draftLen: 0, acceptedPerStep: 0)
        return String(format: """
            [DeviceCalibrate] %@
              cost-model 実測: a=%.1fms b=%.2fms/tok c=%.1fms (step=forward_ms(D+1)+c+io)
              予測 tok/s: 反復(p=maxK=%d)→%.0f / greedy床(p=0)→%.0f
              ※IO 項は 8GB streaming 実機の cold SSD BW で後埋め(現 dev 機は resident で io=0)
            """, cfg.summary, cm.a, cm.b, cm.c, cfg.maxK, predHigh, predFloor)
    }

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
        guard let promptArr = r["spec_prompt"], let gRef = r["spec_greedy"] else { return "[SpecVerify] skip" }
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
                "[SpecVerify-PROF/step] draft(K×no-sync)=%.1f verify(seqMT exact)=%.1f commit/reject=%.1f (ms)  steps=%d\n",
                Double(tDraft)/s/1e6, Double(tVerify)/s/1e6, Double(tCommit)/s/1e6, steps).data(using: .utf8)!)
        }
        return String(format: """
            [SpecVerify] %@draft K=%d skip=%d/%d + batched verify: %.1f tok/s  accept/step=%.2f  品質(vs Python) %d/%d=%.0f%%%@
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
        guard let promptArr = r["spec_prompt"], let gRef = r["spec_greedy"] else { return "[BuddyNoSync] skip" }
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
            [BuddyNoSync] hot-pin top-%d + pure no-sync%@ (calib=%d): %.1f tok/s  品質(vs Python) %d/%d=%.0f%%%@%@  RSS=%.1fGB
            """, C, skipTag, calibN, Double(N) / secs, match, N, Double(match) / Double(N) * 100, swiftTag, missTag, rss)
    }
}
