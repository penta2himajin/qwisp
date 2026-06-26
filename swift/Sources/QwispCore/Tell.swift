import Foundation
import MLX
import Metal

/// Tell runtime（William Tell = 的=expert を先読みして射抜く）.
/// mlx の batched eval を回避し、chunk 単位で asyncEval しながら次 chunk の expert を
/// background prefetch で先読み → prefetch I/O を GPU 計算に隠す。cross-layer 予測 prefetch の
/// efficient 化（Fate one-pass 相当）を mlx 上で実現する独自スケジューラ。
public enum Tell {
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

        var out: [Int] = []
        let t0 = DispatchTime.now()
        for _ in 0 ..< N {
            let prev = cur
            let snaps = caches.map { $0.snapshot() }
            // pass-1: 予測（full GPU-remap）
            StreamingMoEBlock.probeNoSync = true
            _ = try model.forwardHidden(prev, caches: caches)
            MLX.eval(caches.flatMap { $0.stateArrays } + model.expertCaches.compactMap { $0.lastInds })
            let pred = model.expertCaches.map { distinct($0.lastInds ?? MLXArray([Int32]())) }
            for (i, c) in caches.enumerated() { c.restore(snaps[i], isLinear: isLin[i], trim: 1) }

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
                h = try model.runChunk(h, pos, end, caches: caches)   // この chunk の graph build
                MLX.asyncEval([h] + caches[pos ..< end].flatMap { $0.stateArrays })  // 非同期実行
                if nLo < L { sem.wait() }                            // 次 chunk prefetch 完了を待つ
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
                prevGate = model.expertCaches.map { $0.lastGateInput! }
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
            prevGate = model.expertCaches.map { $0.lastGateInput! }
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
}
