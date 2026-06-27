import Foundation
import MLX
import Metal

/// Tell runtime（William Tell = 的=expert を先読みして射抜く）.
/// mlx の batched eval を回避し、chunk 単位で asyncEval しながら次 chunk の expert を
/// background prefetch で先読み → prefetch I/O を GPU 計算に隠す。cross-layer 予測 prefetch の
/// efficient 化（Fate one-pass 相当）を mlx 上で実現する独自スケジューラ。
public enum Tell {
    /// hot/cold 設計の Stage-1 計測: 実 decode の per-layer expert 使用頻度を集計し、
    /// 「top-B hot expert が routing の何 % をカバーするか」を出す（hot/cold の有望度判定）。
    public static func runHotColdCalib(modelDir: String, refPath: String) throws -> String {
        guard let device = MTLCreateSystemDefaultDevice() else { return "ERROR: no Metal device" }
        let r = try loadArrays(url: URL(fileURLWithPath: refPath))
        guard let promptArr = r["spec_prompt"], let gRef = r["spec_greedy"] else { return "[HotCold] skip" }
        let C = Int(ProcessInfo.processInfo.environment["QWISP_CACHE_C"] ?? "64") ?? 64
        let store = try WeightStore(modelDir: modelDir)
        store.residentNonExperts()
        let source = try ExpertSource(modelDir: modelDir); try source.warm()
        let arena = try ExpertArena(device: device, source: source, N: 64)
        let model = try StreamingQwispModel(store: store, arena: arena, device: device,
                                            source: source, cacheC: C)
        let ids = promptArr.asType(.int32).reshaped([1, promptArr.dim(0)])
        let gR = gRef.asArray(Int32.self).map { Int($0) }
        let N = Swift.min(Int(ProcessInfo.processInfo.environment["QWISP_GEN"] ?? "64") ?? 64, gR.count)
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
            [HotCold-CALIB] %d tok, %d MoE層, top-k=8/256。per-layer 活性 expert 平均=%.0f/256
              top-B hot coverage(routing の何%%): B16=%.0f%% B32=%.0f%% B48=%.0f%% B64=%.0f%% B96=%.0f%% B128=%.0f%%
            """,
            total, nMoE, distinctAvg,
            coverage(16), coverage(32), coverage(48), coverage(64), coverage(96), coverage(128))
    }

    /// SS-MoE D1: no-sync(hot subset-expert) forward を draft に、exact forward で verify(lossless)。
    /// MTP head の代わりに no-sync 自己 draft。受理率で SS-MoE(subset-expert draft)の有望度を測る。
    public static func runHotColdSpec(modelDir: String, refPath: String) throws -> String {
        guard let device = MTLCreateSystemDefaultDevice() else { return "ERROR: no Metal device" }
        let r = try loadArrays(url: URL(fileURLWithPath: refPath))
        guard let promptArr = r["spec_prompt"], let gRef = r["spec_greedy"] else { return "[HotColdSpec] skip" }
        let C = Int(ProcessInfo.processInfo.environment["QWISP_CACHE_C"] ?? "64") ?? 64
        let calibN = Int(ProcessInfo.processInfo.environment["QWISP_CALIB"] ?? "32") ?? 32
        let store = try WeightStore(modelDir: modelDir)
        store.residentNonExperts()
        let source = try ExpertSource(modelDir: modelDir); try source.warm()
        let arena = try ExpertArena(device: device, source: source, N: 64)
        let model = try StreamingQwispModel(store: store, arena: arena, device: device,
                                            source: source, cacheC: C)
        let ids = promptArr.asType(.int32).reshaped([1, promptArr.dim(0)])
        let gR = gRef.asArray(Int32.self).map { Int($0) }
        let isLin = model.isLinearFlags
        let N = Swift.min(Int(ProcessInfo.processInfo.environment["QWISP_GEN"] ?? "64") ?? 64, gR.count)
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
            [HotColdSpec] no-sync draft + exact verify(C=%d): %.1f tok/s  accept=%.3f  品質 %d/%d=%.0f%%
            """, C, Double(N) / secs, Double(acc) / Double(steps), match, N, Double(match) / Double(N) * 100)
    }

    /// 予測器 calib: exact decode で (層の pre-attention 入力 X, 真 top-8 routing Y) を層別に収集し
    /// safetensors に dump。Python で ridge/非線形予測器を fit し coverage を測る（訓練抜き予測器の検証）。
    public static func runPredictorCalib(modelDir: String, refPath: String) throws -> String {
        guard let device = MTLCreateSystemDefaultDevice() else { return "ERROR: no Metal device" }
        let r = try loadArrays(url: URL(fileURLWithPath: refPath))
        guard let promptArr = r["spec_prompt"], let gRef = r["spec_greedy"] else { return "[PredCalib] skip" }
        let store = try WeightStore(modelDir: modelDir)
        store.residentNonExperts()
        let source = try ExpertSource(modelDir: modelDir); try source.warm()
        let arena = try ExpertArena(device: device, source: source, N: 64)
        let model = try StreamingQwispModel(store: store, arena: arena, device: device, source: source, cacheC: 64)
        let ids = promptArr.asType(.int32).reshaped([1, promptArr.dim(0)])
        _ = gRef
        let N = Int(ProcessInfo.processInfo.environment["QWISP_GEN"] ?? "512") ?? 512   // data 量（gR 非依存）
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
        let outPath = ProcessInfo.processInfo.environment["QWISP_PRED_OUT"] ?? "/tmp/qwisp_predictor_data.safetensors"
        try MLX.save(arrays: dict, url: URL(fileURLWithPath: outPath))
        return "[PredCalib] dumped X/Y for \(nMoE)層 × \(N) tok (H=\(H)) → \(outPath)"
    }

    /// (A) mmap-gather: 全 expert を mmap のまま resident MoE で GPU-side gather（arena/ensure/per-layer
    /// sync 撤廃, exact）。OS demand paging で working set だけ常駐。sync 撤廃で routing tax が消え
    /// 8GB 内で速いか（thrash しないか）を検証。tok/s・RSS・品質を測る。
    public static func runMmapGather(modelDir: String, refPath: String) throws -> String {
        let r = try loadArrays(url: URL(fileURLWithPath: refPath))
        guard let promptArr = r["spec_prompt"], let gRef = r["spec_greedy"] else { return "[MmapGather] skip" }
        let store = try WeightStore(modelDir: modelDir)
        store.residentNonExperts()                          // experts は mmap のまま（paged）
        let rssLoad = StreamingDecode.rssGB()
        let model = QwispModel(store: store)
        let ids = promptArr.asType(.int32).reshaped([1, promptArr.dim(0)])
        let gR = gRef.asArray(Int32.self).map { Int($0) }
        let N = Swift.min(Int(ProcessInfo.processInfo.environment["QWISP_GEN"] ?? "64") ?? 64, gR.count)
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

    /// SWIFT 流 ①: 各層を単独 skip したときの matchness(skip-L argmax == full argmax 率)を計測。
    /// どの層が「抜いても出力を保つ=skip 可能」か、GDN/attn どちらが skip 可能かを判明させる。
    public static func runSwiftCalib(modelDir: String, refPath: String) throws -> String {
        guard let device = MTLCreateSystemDefaultDevice() else { return "ERROR: no Metal device" }
        let r = try loadArrays(url: URL(fileURLWithPath: refPath))
        guard let promptArr = r["spec_prompt"], let gRef = r["spec_greedy"] else { return "[SwiftCalib] skip" }
        let store = try WeightStore(modelDir: modelDir)
        store.residentNonExperts()
        let source = try ExpertSource(modelDir: modelDir); try source.warm()
        let arena = try ExpertArena(device: device, source: source, N: 64)
        let model = try StreamingQwispModel(store: store, arena: arena, device: device, source: source, cacheC: 64)
        let ids = promptArr.asType(.int32).reshaped([1, promptArr.dim(0)])
        let gR = gRef.asArray(Int32.self).map { Int($0) }
        let isLin = model.isLinearFlags
        let L = model.layerCount
        let N = Swift.min(Int(ProcessInfo.processInfo.environment["QWISP_GEN"] ?? "16") ?? 16, gR.count)
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
            [SwiftCalib] %d step, 各層単独 skip の matchness。skip可(matchness≥0.95)=%d/%d 層 (GDN %d/attn %d)
              skip可層(番号+G/A): %@
              最 skip 可 top8: %@
            """, N, skippable.count, L, nG, nA, ginfo,
            ranked.prefix(8).map { String(format: "%d%@=%.2f", $0.l, $0.lin ? "G" : "A", $0.m) }.joined(separator: " "))
    }

    /// SS-MoE DK: no-sync hot-pin draft を K トークン先まで → 1 回の batched exact verify(lossless)。
    /// 高 accept(0.94+)を multi-token で償却し no-sync 天井に安全到達。reject 時のみ accepted prefix を
    /// exact 再走（GDN partial-commit 回避）。QWISP_DRAFT_K で K。
    public static func runHotColdSpecK(modelDir: String, refPath: String) throws -> String {
        guard let device = MTLCreateSystemDefaultDevice() else { return "ERROR: no Metal device" }
        let r = try loadArrays(url: URL(fileURLWithPath: refPath))
        guard let promptArr = r["spec_prompt"], let gRef = r["spec_greedy"] else { return "[HotColdSpecK] skip" }
        let C = Int(ProcessInfo.processInfo.environment["QWISP_CACHE_C"] ?? "64") ?? 64
        let calibN = Int(ProcessInfo.processInfo.environment["QWISP_CALIB"] ?? "32") ?? 32
        let K = Int(ProcessInfo.processInfo.environment["QWISP_DRAFT_K"] ?? "4") ?? 4
        let skipStride = Int(ProcessInfo.processInfo.environment["QWISP_SKIP_STRIDE"] ?? "0") ?? 0
        let store = try WeightStore(modelDir: modelDir)
        store.residentNonExperts()
        let source = try ExpertSource(modelDir: modelDir); try source.warm()
        let arena = try ExpertArena(device: device, source: source, N: 64)
        let model = try StreamingQwispModel(store: store, arena: arena, device: device,
                                            source: source, cacheC: C)
        let ids = promptArr.asType(.int32).reshaped([1, promptArr.dim(0)])
        let gR = gRef.asArray(Int32.self).map { Int($0) }
        let isLin = model.isLinearFlags
        let N = Swift.min(Int(ProcessInfo.processInfo.environment["QWISP_GEN"] ?? "64") ?? 64, gR.count)
        let nE = 256, nMoE = model.expertCaches.count
        let caches = model.makeCaches()
        var counts = [[Int]](repeating: [Int](repeating: 0, count: nE), count: nMoE)

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

        let mc = model.makeCaches()
        StreamingMoEBlock.probeNoSync = false
        (_, lg) = try model.prefillChunked(ids, caches: mc)
        var uArr = MLX.argMax(lg[0..., (lg.dim(1) - 1)...], axis: -1)
        MLX.eval([uArr] + mc.flatMap { $0.stateArrays })
        // layer-skip draft 集合（skipStride≥2: i%stride==stride-1 を間引く。層0/末尾は残す）
        let L = model.layerCount
        var skip = Set<Int>()
        if skipStride >= 2 { for i in 1 ..< (L - 1) where i % skipStride == (skipStride - 1) { skip.insert(i) } }

        let prof = ProcessInfo.processInfo.environment["QWISP_SPECK_PROF"] == "1"
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
            let vNoSync = ProcessInfo.processInfo.environment["QWISP_VERIFY_NOSYNC"] == "1"
            StreamingMoEBlock.probeNoSync = vNoSync; AttentionLayer.seqMultiToken = vseq
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
        if prof {
            let s = Double(steps)
            FileHandle.standardError.write(String(format:
                "[SPECK-PROF/step] draft(K×no-sync)=%.1f verify(seqMT exact)=%.1f commit/reject=%.1f (ms)  steps=%d\n",
                Double(tDraft)/s/1e6, Double(tVerify)/s/1e6, Double(tCommit)/s/1e6, steps).data(using: .utf8)!)
        }
        return String(format: """
            [HotColdSpecK] no-sync draft K=%d skip=%d/%d + batched verify: %.1f tok/s  accept/step=%.2f  品質 %d/%d=%.0f%%
            """, K, skip.count, L, Double(N) / secs, Double(accTok) / Double(steps), match, N, Double(match) / Double(N) * 100)
    }

    /// hot/cold ④ per-prompt auto: hot 常駐 + 短い probe で no-sync 安全性を判定。
    /// probe(exact を真値に、no-sync を snapshot/restore で side 比較)が全一致なら no-sync(47, 速)、
    /// 不一致(drift 兆候)なら exact M2 経路(lossless)へ。プロンプト難度に自動適応。
    public static func runHotColdAuto(modelDir: String, refPath: String) throws -> String {
        guard let device = MTLCreateSystemDefaultDevice() else { return "ERROR: no Metal device" }
        let r = try loadArrays(url: URL(fileURLWithPath: refPath))
        guard let promptArr = r["spec_prompt"], let gRef = r["spec_greedy"] else { return "[HotColdAuto] skip" }
        let C = Int(ProcessInfo.processInfo.environment["QWISP_CACHE_C"] ?? "64") ?? 64
        let calibN = Int(ProcessInfo.processInfo.environment["QWISP_CALIB"] ?? "32") ?? 32
        let probeK = Int(ProcessInfo.processInfo.environment["QWISP_PROBE"] ?? "8") ?? 8
        let store = try WeightStore(modelDir: modelDir)
        store.residentNonExperts()
        let source = try ExpertSource(modelDir: modelDir); try source.warm()
        let arena = try ExpertArena(device: device, source: source, N: 64)
        let model = try StreamingQwispModel(store: store, arena: arena, device: device,
                                            source: source, cacheC: C)
        let ids = promptArr.asType(.int32).reshaped([1, promptArr.dim(0)])
        let gR = gRef.asArray(Int32.self).map { Int($0) }
        let isLin = model.isLinearFlags
        let N = Swift.min(Int(ProcessInfo.processInfo.environment["QWISP_GEN"] ?? "64") ?? 64, gR.count)
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
            [HotColdAuto] hot top-%d, probe=%d miss=%d → mode=%@: %.1f tok/s  品質 %d/%d=%.0f%%
            """, C, probeK, probeMiss, easy ? "no-sync(47)" : "exact(lossless)",
            Double(N) / secs, match, N, Double(match) / Double(N) * 100)
    }

    /// hot/cold ③ 適応 sync: hot 常駐 + per-layer 判定。calib で各層 hot coverage を測り、
    /// coverage≥θ の易層は no-sync（hot で賄う）、θ 未満の hard 層だけ exact sync（cold をロード=正確）。
    public static func runHotColdAdaptive(modelDir: String, refPath: String) throws -> String {
        guard let device = MTLCreateSystemDefaultDevice() else { return "ERROR: no Metal device" }
        let r = try loadArrays(url: URL(fileURLWithPath: refPath))
        guard let promptArr = r["spec_prompt"], let gRef = r["spec_greedy"] else { return "[HotColdAdaptive] skip" }
        let C = Int(ProcessInfo.processInfo.environment["QWISP_CACHE_C"] ?? "64") ?? 64
        let calibN = Int(ProcessInfo.processInfo.environment["QWISP_CALIB"] ?? "48") ?? 48
        let theta = Double(ProcessInfo.processInfo.environment["QWISP_THETA"] ?? "0.995") ?? 0.995
        let store = try WeightStore(modelDir: modelDir)
        store.residentNonExperts()
        let source = try ExpertSource(modelDir: modelDir); try source.warm()
        let arena = try ExpertArena(device: device, source: source, N: 64)
        let model = try StreamingQwispModel(store: store, arena: arena, device: device,
                                            source: source, cacheC: C)
        let ids = promptArr.asType(.int32).reshaped([1, promptArr.dim(0)])
        let gR = gRef.asArray(Int32.self).map { Int($0) }
        let N = Swift.min(Int(ProcessInfo.processInfo.environment["QWISP_GEN"] ?? "64") ?? 64, gR.count)
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
            [HotColdAdaptive] hot top-%d + 適応sync(θ=%.3f): sync層=%d/%d  %.1f tok/s  品質 %d/%d=%.0f%%
            """, C, theta, syncReal.count, nMoE, Double(N) / secs, match, N, Double(match) / Double(N) * 100)
    }

    /// hot/cold Step-2: オンライン適応 hot set + no-sync。毎 token 走行頻度の top-C を ensure し
    /// （安定なら大半 hit で IO 小）、no-sync forward。静的 calib の distribution shift を吸収できるか。
    public static func runHotColdOnline(modelDir: String, refPath: String) throws -> String {
        guard let device = MTLCreateSystemDefaultDevice() else { return "ERROR: no Metal device" }
        let r = try loadArrays(url: URL(fileURLWithPath: refPath))
        guard let promptArr = r["spec_prompt"], let gRef = r["spec_greedy"] else { return "[HotColdOnline] skip" }
        let C = Int(ProcessInfo.processInfo.environment["QWISP_CACHE_C"] ?? "64") ?? 64
        let calibN = Int(ProcessInfo.processInfo.environment["QWISP_CALIB"] ?? "16") ?? 16
        let store = try WeightStore(modelDir: modelDir)
        store.residentNonExperts()
        let source = try ExpertSource(modelDir: modelDir); try source.warm()
        let arena = try ExpertArena(device: device, source: source, N: 64)
        let model = try StreamingQwispModel(store: store, arena: arena, device: device,
                                            source: source, cacheC: C)
        let ids = promptArr.asType(.int32).reshaped([1, promptArr.dim(0)])
        let gR = gRef.asArray(Int32.self).map { Int($0) }
        let N = Swift.min(Int(ProcessInfo.processInfo.environment["QWISP_GEN"] ?? "64") ?? 64, gR.count)
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
            [HotColdOnline] online-adaptive hot top-%d + no-sync: %.1f tok/s  品質(greedy一致) %d/%d=%.0f%%
            """, C, Double(N) / secs, match, N, Double(match) / Double(N) * 100)
    }

    /// hot/cold Stage-1 診断: exact decode の実 routing に対し、(a)静的calib hot set と
    /// (b)オンライン適応 hot set の coverage を比較。code 失敗が distribution shift か予測不能かを切分け。
    public static func runHotColdDiag(modelDir: String, refPath: String) throws -> String {
        guard let device = MTLCreateSystemDefaultDevice() else { return "ERROR: no Metal device" }
        let r = try loadArrays(url: URL(fileURLWithPath: refPath))
        guard let promptArr = r["spec_prompt"], let gRef = r["spec_greedy"] else { return "[HotColdDiag] skip" }
        let C = Int(ProcessInfo.processInfo.environment["QWISP_CACHE_C"] ?? "64") ?? 64
        let calibN = Int(ProcessInfo.processInfo.environment["QWISP_CALIB"] ?? "48") ?? 48
        let store = try WeightStore(modelDir: modelDir)
        store.residentNonExperts()
        let source = try ExpertSource(modelDir: modelDir); try source.warm()
        let arena = try ExpertArena(device: device, source: source, N: 64)
        let model = try StreamingQwispModel(store: store, arena: arena, device: device,
                                            source: source, cacheC: C)
        let ids = promptArr.asType(.int32).reshaped([1, promptArr.dim(0)])
        let gR = gRef.asArray(Int32.self).map { Int($0) }
        let N = Swift.min(Int(ProcessInfo.processInfo.environment["QWISP_GEN"] ?? "64") ?? 64, gR.count)
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
            [HotColdDiag] C=%d calib=%d eval=%d。実 routing に対する hot-set coverage:
              静的 calib hot = %.1f%%   オンライン適応 hot = %.1f%%   (per-token 最悪層 online=%.0f%%)
            """, C, calibN, N,
            Double(hitStatic) / Double(totalRoute) * 100,
            Double(hitOnline) / Double(totalRoute) * 100, worstTokMin)
    }

    /// hot/cold 試行: 頻度上位 hot expert を C slot に pin(常駐)→ pure no-sync decode。
    /// hot resident で miss が減り no-sync drift が抑えられるか（速度はほぼ no-sync 天井）を検証。
    public static func runHotColdFast(modelDir: String, refPath: String) throws -> String {
        guard let device = MTLCreateSystemDefaultDevice() else { return "ERROR: no Metal device" }
        let r = try loadArrays(url: URL(fileURLWithPath: refPath))
        guard let promptArr = r["spec_prompt"], let gRef = r["spec_greedy"] else { return "[HotColdFast] skip" }
        let C = Int(ProcessInfo.processInfo.environment["QWISP_CACHE_C"] ?? "64") ?? 64
        let calibN = Int(ProcessInfo.processInfo.environment["QWISP_CALIB"] ?? "48") ?? 48
        let store = try WeightStore(modelDir: modelDir)
        store.residentNonExperts()
        let source = try ExpertSource(modelDir: modelDir); try source.warm()
        let arena = try ExpertArena(device: device, source: source, N: 64)
        let model = try StreamingQwispModel(store: store, arena: arena, device: device,
                                            source: source, cacheC: C)
        let ids = promptArr.asType(.int32).reshaped([1, promptArr.dim(0)])
        let gR = gRef.asArray(Int32.self).map { Int($0) }
        let N = Swift.min(Int(ProcessInfo.processInfo.environment["QWISP_GEN"] ?? "64") ?? 64, gR.count)
        let nE = 256, nMoE = model.expertCaches.count
        let caches = model.makeCaches()

        // --- phase 1: calibration（exact decode で頻度集計）---
        var counts = [[Int]](repeating: [Int](repeating: 0, count: nE), count: nMoE)
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
            cur = MLX.argMax(lg[0, 0], axis: -1).reshaped([1, 1])
            MLX.eval([cur] + caches.flatMap { $0.stateArrays })
        }
        StreamingMoEBlock.captureInds = false

        // --- phase 2: top-C hot を各層 pin（ensure で常駐ロード）---
        for (mi, ec) in model.expertCaches.enumerated() {
            let hot = Array(counts[mi].enumerated().sorted { $0.element > $1.element }.prefix(C).map { $0.offset })
            _ = ec.ensure(hot)
        }

        // --- phase 3: 実プロンプトから pure no-sync decode（ensure 無し＝hot 固定）---
        let caches2 = model.makeCaches()
        StreamingMoEBlock.probeNoSync = false
        (_, lg) = try model.prefillChunked(ids, caches: caches2)
        cur = MLX.argMax(lg[0, lg.dim(1) - 1], axis: -1).reshaped([1, 1])
        MLX.eval([cur] + caches2.flatMap { $0.stateArrays })
        StreamingMoEBlock.probeNoSync = true   // 以降 no-sync（hot 固定 slotTable で gather）
        var out: [Int] = []
        let t0 = DispatchTime.now()
        for _ in 0 ..< N {
            out.append(cur.item(Int.self))
            (_, lg) = try model.forwardHidden(cur, caches: caches2)
            cur = MLX.argMax(lg[0, 0], axis: -1).reshaped([1, 1])
            MLX.eval([cur] + caches2.flatMap { $0.stateArrays })
        }
        StreamingMoEBlock.probeNoSync = false
        let secs = Double(DispatchTime.now().uptimeNanoseconds - t0.uptimeNanoseconds) / 1e9
        let match = zip(out, gR).filter { $0 == $1 }.count
        return String(format: """
            [HotColdFast] hot-pin top-%d + pure no-sync (calib=%d): %.1f tok/s  品質(greedy一致) %d/%d=%.0f%%
            """, C, calibN, Double(N) / secs, match, N, Double(match) / Double(N) * 100)
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

    /// M0: 2-pass(pass-1 予測 → pass-2 chunked + overlapped prefetch)。near-lossless で速度を稼ぐ。
    public static func runM0(modelDir: String, refPath: String) throws -> String {
        guard let device = MTLCreateSystemDefaultDevice() else { return "ERROR: no Metal device" }
        let r = try loadArrays(url: URL(fileURLWithPath: refPath))
        guard let promptArr = r["spec_prompt"], let gRef = r["spec_greedy"] else { return "[Tell] skip" }
        let C = Int(ProcessInfo.processInfo.environment["QWISP_CACHE_C"] ?? "64") ?? 64
        let CH = Int(ProcessInfo.processInfo.environment["QWISP_CHUNK"] ?? "10") ?? 10
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
        let N = Swift.min(Int(ProcessInfo.processInfo.environment["QWISP_GEN"] ?? "48") ?? 48, gR.count)
        let caches = model.makeCaches()

        StreamingMoEBlock.probeNoSync = false
        var (_, lg) = try model.prefillChunked(ids, caches: caches)
        var cur = MLX.argMax(lg[0, lg.dim(1) - 1], axis: -1).reshaped([1, 1])
        MLX.eval([cur] + caches.flatMap { $0.stateArrays })

        // 各層の予測 distinct experts を抽出
        func distinct(_ a: MLXArray) -> [Int] {
            var seen = Set<Int>(); var u: [Int] = []
            for e in a.asArray(Int32.self) { let i = Int(e); if seen.insert(i).inserted { u.append(i) } }
            return u
        }
        func prefetch(_ lo: Int, _ hi: Int, _ pred: [[Int]]) {
            for i in lo ..< hi { _ = model.expertCaches[i].ensure(pred[i]) }
        }

        let prof = ProcessInfo.processInfo.environment["QWISP_M0_PROF"] == "1"
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
            MLX.eval(caches.flatMap { $0.stateArrays } + model.expertCaches.compactMap { $0.lastInds })
            if prof { tP1 += now() - ts; ts = now() }
            let pred = model.expertCaches.map { distinct($0.lastInds ?? MLXArray([Int32]())) }
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
                "[M0-PROF/tok] pass1=%.1f pred-readback=%.1f pass2-build=%.1f prefetch-wait=%.1f final-drain=%.1f (ms)\n",
                Double(tP1)/n/1e6, Double(tPred)/n/1e6, Double(tBuild)/n/1e6,
                Double(tWait)/n/1e6, Double(tFinal)/n/1e6).data(using: .utf8)!)
        }
        return String(format: """
            [Tell M0] chunk overlap 2-pass(C=%d, chunk=%d): %.1f tok/s  品質(greedy一致) %d/%d=%.0f%%
            """,
            C, CH, Double(N) / secs, match, N, Double(match) / Double(N) * 100)
    }

    /// M1: one-pass。pass-1 を除去し、chunk の入力 hidden から chunk 内各層の expert を cross-layer
    /// 予測(gate を hidden に適用)→prefetch→GPU-remap で chunk を実行。pass-1 の 17ms が消える。
    public static func runM1(modelDir: String, refPath: String) throws -> String {
        guard let device = MTLCreateSystemDefaultDevice() else { return "ERROR: no Metal device" }
        let r = try loadArrays(url: URL(fileURLWithPath: refPath))
        guard let promptArr = r["spec_prompt"], let gRef = r["spec_greedy"] else { return "[Tell] skip" }
        let C = Int(ProcessInfo.processInfo.environment["QWISP_CACHE_C"] ?? "64") ?? 64
        let CH = Int(ProcessInfo.processInfo.environment["QWISP_CHUNK"] ?? "4") ?? 4
        let store = try WeightStore(modelDir: modelDir)
        store.residentNonExperts()
        let source = try ExpertSource(modelDir: modelDir); try source.warm()
        let arena = try ExpertArena(device: device, source: source, N: 64)
        let model = try StreamingQwispModel(store: store, arena: arena, device: device,
                                            source: source, cacheC: C)
        let ids = promptArr.asType(.int32).reshaped([1, promptArr.dim(0)])
        let gR = gRef.asArray(Int32.self).map { Int($0) }
        let L = model.layerCount
        let N = Swift.min(Int(ProcessInfo.processInfo.environment["QWISP_GEN"] ?? "48") ?? 48, gR.count)
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
            [Tell M1] one-pass cross-layer(C=%d, chunk=%d): %.1f tok/s  品質(greedy一致) %d/%d=%.0f%%
            """,
            C, CH, Double(N) / secs, match, N, Double(match) / Double(N) * 100)
    }

    /// M2: Fate 流 one-pass。各層の真の gate 入力(=MoE 入力 x)を capture し、chunk N の最終 gate
    /// 入力で chunk N+1 を予測(隣接層 cosine>83% で高精度)→prefetch→GPU-remap。pass-1 不要=高速。
    public static func runM2(modelDir: String, refPath: String) throws -> String {
        guard let device = MTLCreateSystemDefaultDevice() else { return "ERROR: no Metal device" }
        let r = try loadArrays(url: URL(fileURLWithPath: refPath))
        guard let promptArr = r["spec_prompt"], let gRef = r["spec_greedy"] else { return "[Tell] skip" }
        let C = Int(ProcessInfo.processInfo.environment["QWISP_CACHE_C"] ?? "64") ?? 64
        let CH = Int(ProcessInfo.processInfo.environment["QWISP_CHUNK"] ?? "4") ?? 4
        GatedDeltaNetLayer.fuseGDN = ProcessInfo.processInfo.environment["QWISP_FUSE_GDN"] == "1"
        let store = try WeightStore(modelDir: modelDir)
        store.residentNonExperts()
        let source = try ExpertSource(modelDir: modelDir); try source.warm()
        let arena = try ExpertArena(device: device, source: source, N: 64)
        let model = try StreamingQwispModel(store: store, arena: arena, device: device,
                                            source: source, cacheC: C)
        let ids = promptArr.asType(.int32).reshaped([1, promptArr.dim(0)])
        let gR = gRef.asArray(Int32.self).map { Int($0) }
        let L = model.layerCount
        let N = Swift.min(Int(ProcessInfo.processInfo.environment["QWISP_GEN"] ?? "48") ?? 48, gR.count)
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
        // 前 token の各層 gate 入力（chunk-0 の bootstrap 用, temporal）
        var prevGate: [MLXArray]? = nil

        let prof = ProcessInfo.processInfo.environment["QWISP_M2_PROF"] == "1"
        let prof2 = ProcessInfo.processInfo.environment["QWISP_M2_PROF2"] == "1"
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
            let multiSrc = ProcessInfo.processInfo.environment["QWISP_MULTI"] == "1"
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
                "[M2-PROF/tok] eval(preds=stall)=%.1f ensure(IO)=%.1f final-drain=%.1f (ms) chunks/tok=%d\n",
                Double(tEval)/s/1e6, Double(tEnsure)/s/1e6, Double(tFinal)/s/1e6, (L + CH - 1) / CH).data(using: .utf8)!)
            FileHandle.standardError.write(String(format:
                "[M2-PROF ensure内訳/tok] pread(IO)=%.2fms misses=%.1f/tok | ensure合計=%.2fms → CPU(slot/distinct)=%.2fms\n",
                Double(LayerExpertCache.preadNanos)/s/1e6, Double(LayerExpertCache.missTotal)/s,
                Double(LayerExpertCache.ensureNanos)/s/1e6,
                Double(LayerExpertCache.ensureNanos - LayerExpertCache.preadNanos)/s/1e6).data(using: .utf8)!)
        }
        if prof2, pSteps > 0 {
            StreamingMoEBlock.profileLayers = false
            let s = Double(pSteps); func m(_ x: UInt64) -> Double { Double(x)/s/1e6 }
            let tot = m(tEmbed)+m(tEval)+m(tDistinct)+m(tEnsure)+m(tRunChunk)+m(tFinal)+m(tLastGate)
            FileHandle.standardError.write(String(format:
                "[M2-PROF2/tok barrier] embed=%.2f predict(gate)=%.2f distinct(readback)=%.2f ensure(IO)=%.2f "
                + "runChunk(attn/gdn/moe)=%.2f final(norm/lmhead)=%.2f lastGate=%.2f | 合計=%.1fms\n",
                m(tEmbed), m(tEval), m(tDistinct), m(tEnsure), m(tRunChunk), m(tFinal), m(tLastGate), tot).data(using: .utf8)!)
            FileHandle.standardError.write(String(format:
                "[M2-PROF2 runChunk内訳/tok] GDN(30層)=%.2f attn(10層)=%.2f MoE-gather(40層)=%.2f "
                + "MoE-shared(40層)=%.2f norm=%.2f (ms)\n",
                m(StreamingMoEBlock.tGDN), m(StreamingMoEBlock.tAttn), m(StreamingMoEBlock.tMoEgather),
                m(StreamingMoEBlock.tMoEshared), m(StreamingMoEBlock.tNorm)).data(using: .utf8)!)
            FileHandle.standardError.write(String(format:
                "[M2-PROF2 GDN内訳/tok] in_proj(4本)=%.2f conv1d+norm=%.2f recurrent-kernel=%.2f out_proj=%.2f (ms)\n",
                m(StreamingMoEBlock.tGdnInproj), m(StreamingMoEBlock.tGdnConv),
                m(StreamingMoEBlock.tGdnKernel), m(StreamingMoEBlock.tGdnOut)).data(using: .utf8)!)
        }
        return String(format: """
            [Tell M2] Fate one-pass(C=%d, chunk=%d): %.1f tok/s  品質(greedy一致) %d/%d=%.0f%%
            """,
            C, CH, Double(N) / secs, match, N, Double(match) / Double(N) * 100)
    }

    /// M5 = M2 one-pass + depth-1 software pipeline。
    /// 仮説: chunk N の expert matmul を asyncEval で GPU に流しつつ、chunk N+1 の予測+gather を
    /// CPU/IO で重ねれば、M2 の per-chunk eval stall(~23ms)を build(~15ms)へ近づけられるか検証。
    public static func runM5(modelDir: String, refPath: String) throws -> String {
        guard let device = MTLCreateSystemDefaultDevice() else { return "ERROR: no Metal device" }
        let r = try loadArrays(url: URL(fileURLWithPath: refPath))
        guard let promptArr = r["spec_prompt"], let gRef = r["spec_greedy"] else { return "[Tell M5] skip" }
        let C = Int(ProcessInfo.processInfo.environment["QWISP_CACHE_C"] ?? "64") ?? 64
        let CH = Int(ProcessInfo.processInfo.environment["QWISP_CHUNK"] ?? "2") ?? 2
        let store = try WeightStore(modelDir: modelDir)
        store.residentNonExperts()
        let source = try ExpertSource(modelDir: modelDir); try source.warm()
        let arena = try ExpertArena(device: device, source: source, N: 64)
        let model = try StreamingQwispModel(store: store, arena: arena, device: device,
                                            source: source, cacheC: C)
        let ids = promptArr.asType(.int32).reshaped([1, promptArr.dim(0)])
        let gR = gRef.asArray(Int32.self).map { Int($0) }
        let L = model.layerCount
        let N = Swift.min(Int(ProcessInfo.processInfo.environment["QWISP_GEN"] ?? "48") ?? 48, gR.count)
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
            [Tell M5] M2+pipeline(C=%d, chunk=%d): %.1f tok/s  品質(greedy一致) %d/%d=%.0f%%
            """,
            C, CH, Double(N) / secs, match, N, Double(match) / Double(N) * 100)
    }

    /// M4 = MTP D1 投機 × Tell M2 verify。verify(2トークン)に cross-layer prefetch を適用し、
    /// chunk stall を ~1.85トークンに償却。near-lossless 維持で M2(28) を超える狙い。
    public static func runM4(modelDir: String, refPath: String) throws -> String {
        guard let device = MTLCreateSystemDefaultDevice() else { return "ERROR: no Metal device" }
        let r = try loadArrays(url: URL(fileURLWithPath: refPath))
        guard let promptArr = r["spec_prompt"], let gRef = r["spec_greedy"] else { return "[Tell M4] skip" }
        let C = Int(ProcessInfo.processInfo.environment["QWISP_CACHE_C"] ?? "64") ?? 64
        let CH = Int(ProcessInfo.processInfo.environment["QWISP_CHUNK"] ?? "2") ?? 2
        // 既定は exact verify（spec を lossless に再現）。QWISP_M4_PRED=1 で予測 verify（高速だが lossy）。
        let exactVerify = ProcessInfo.processInfo.environment["QWISP_M4_PRED"] != "1"
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
        let N = Swift.min(Int(ProcessInfo.processInfo.environment["QWISP_GEN"] ?? "48") ?? 48, gR.count)
        let mainCaches = model.makeCaches()
        let mtpKV = KVCache()
        let P = ids.dim(-1)

        // prefill（sync, gate 入力 capture）→ prevGate
        // 注意: prompt 全体を 1 forward すると |U|>C で arena 破綻するため chunk 分割必須（specLoop と同じ）。
        StreamingMoEBlock.probeNoSync = false; StreamingMoEBlock.captureGateInput = true
        let (Hf, lgf) = try model.prefillChunked(ids, caches: mainCaches)
        var uArr = MLX.argMax(lgf[0..., (lgf.dim(1) - 1)...], axis: -1)
        var lastH = Hf[0..., (P - 1)...]
        _ = head(Hf[0..., 0 ..< (P - 1)], ids[0..., 1...], cache: mtpKV)
        var prevGate: [MLXArray]? = lastGate(model)
        func evalAll() {
            MLX.eval([uArr, lastH] + [mtpKV.keys, mtpKV.values].compactMap { $0 }
                     + mainCaches.flatMap { $0.stateArrays } + model.expertCaches.compactMap { $0.lastGateInput })
        }
        evalAll()
        prevGate = lastGate(model)

        // verify([u,v]) の attention を逐次化＝true greedy-lossless（既定 ON, prefill 後に有効化）。
        if ProcessInfo.processInfo.environment["QWISP_BATCHED_VERIFY"] != "1" { AttentionLayer.seqMultiToken = true }
        var out: [Int] = []; var steps = 0, acc = 0
        let t0 = DispatchTime.now()
        while out.count < N {
            steps += 1
            let dl = head(lastH, uArr, cache: mtpKV)
            let dArr = MLX.argMax(dl[0..., 0...], axis: -1)
            let ud = MLX.concatenated([uArr, dArr], axis: 1)
            let snaps = mainCaches.map { $0.snapshot() }
            let (H2, lg2) = try tellForward(model, ud, mainCaches, prevGate, CH, exact: exactVerify)   // ★Tell verify
            let pg2 = lastGate(model)
            let vw = MLX.argMax(lg2[0, 0 ..< 2], axis: -1)
            let vals = MLX.concatenated([dArr[0], vw]).asArray(Int32.self)
            let d = Int(vals[0]), v = Int(vals[1])
            out.append(uArr.item(Int.self))
            if v == d {
                acc += 1; out.append(d)
                _ = head(H2[0..., 0 ..< 1], dArr, cache: mtpKV)
                uArr = vw[1 ..< 2].reshaped([1, 1]); lastH = H2[0..., 1 ..< 2]
                prevGate = pg2
            } else {
                for (i, c) in mainCaches.enumerated() { c.restore(snaps[i], isLinear: isLin[i], trim: 2) }
                let (H1, _) = try tellForward(model, uArr, mainCaches, prevGate, CH, exact: exactVerify)   // [u] 再投入
                uArr = vw[0 ..< 1].reshaped([1, 1]); lastH = H1[0..., 0 ..< 1]
                prevGate = lastGate(model)
            }
            evalAll()
        }
        StreamingMoEBlock.probeNoSync = false; StreamingMoEBlock.captureGateInput = false
        AttentionLayer.seqMultiToken = false   // global static を後続に漏らさない
        let secs = Double(DispatchTime.now().uptimeNanoseconds - t0.uptimeNanoseconds) / 1e9
        let match = zip(out.prefix(N), gR).filter { $0 == $1 }.count
        return String(format: """
            [Tell M4] MTP × Tell verify(%@, C=%d, chunk=%d): %.1f tok/s  accept=%.3f  品質(greedy一致) %d/%d=%.0f%%
            """,
            exactVerify ? "exact" : "pred", C, CH, Double(N) / secs, Double(acc) / Double(steps), match, N, Double(match) / Double(N) * 100)
    }
}
