import Foundation
import MLX
import Metal

/// Tell runtime（William Tell = 的=expert を先読みして射抜く）.
/// mlx の batched eval を回避し、chunk 単位で asyncEval しながら次 chunk の expert を
/// background prefetch で先読み → prefetch I/O を GPU 計算に隠す。cross-layer 予測 prefetch の
/// efficient 化（Fate one-pass 相当）を mlx 上で実現する独自スケジューラ。
public enum Tell {
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
            var h = model.embedPub(prev)
            var pos = 0
            while pos < L {
                let end = Swift.min(pos + CH, L)
                // 予測元: chunk 0 は前 token 同層 gate 入力(temporal)、以降は前 chunk 最終 gate 入力
                let src = pos > 0 ? model.expertCaches[pos - 1].lastGateInput! : prevGate![pos]
                let preds = (pos ..< end).map { i in model.predictLayerInds(i, pos > 0 ? src : prevGate![i]) }
                MLX.eval(preds)
                for (k, i) in (pos ..< end).enumerated() { _ = model.expertCaches[i].ensure(distinct(preds[k])) }
                h = try model.runChunk(h, pos, end, caches: caches)
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
