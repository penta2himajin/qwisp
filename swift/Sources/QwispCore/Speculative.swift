import Foundation
import MLX
import MLXFast

/// MTP D1 投機デコード（draft+verify）。出力は greedy と完全一致するはず（lossless）。
public enum SpeculativeDecode {
    /// greedy 参照（cached AR デコード）。prompt 後 maxTokens 生成。
    static func greedy(_ model: QwispModel, _ promptIds: MLXArray, _ maxTokens: Int) -> [Int] {
        let caches = model.makeCaches()
        var (_, lg) = model.forwardHidden(promptIds, caches: caches)
        var next = MLX.argMax(lg[0, promptIds.dim(-1) - 1], axis: -1).reshaped([1, 1])
        MLX.eval([next] + caches.flatMap { $0.stateArrays })
        var out: [Int] = []
        for _ in 0 ..< maxTokens {
            out.append(next.item(Int.self))
            (_, lg) = model.forwardHidden(next, caches: caches)
            next = MLX.argMax(lg[0, 0], axis: -1).reshaped([1, 1])
            MLX.eval([next] + caches.flatMap { $0.stateArrays })
        }
        return out
    }

    /// D1 投機。返り値: (tokens, steps, acceptedDrafts, decodeSecs)
    static func speculative(_ model: QwispModel, _ head: MTPHead, _ promptIds: MLXArray,
                            _ maxTokens: Int) -> (toks: [Int], steps: Int, acc: Int, secs: Double) {
        let mainCaches = model.makeCaches()
        let mtpKV = KVCache()
        let isLin = model.isLinearFlags
        let P = promptIds.dim(-1)

        // prefill main + MTP
        var (H, lg) = model.forwardHidden(promptIds, caches: mainCaches)
        var uArr = MLX.argMax(lg[0..., (P - 1)...], axis: -1)          // [1,1]
        var lastH = H[0..., (P - 1)...]                                 // [1,1,H]
        _ = head(H[0..., 0 ..< (P - 1)], promptIds[0..., 1...], cache: mtpKV)
        MLX.eval([uArr, lastH] + [mtpKV.keys, mtpKV.values].compactMap { $0 } + mainCaches.flatMap { $0.stateArrays })

        var out: [Int] = []
        var steps = 0, acc = 0
        let t0 = DispatchTime.now()
        while out.count < maxTokens {
            steps += 1
            // draft
            let dl = head(lastH, uArr, cache: mtpKV)
            let dArr = MLX.argMax(dl[0..., 0...], axis: -1)             // [1,1]
            let ud = MLX.concatenated([uArr, dArr], axis: 1)           // [1,2]
            // snapshot main caches（reject 巻き戻し用）
            let snaps = mainCaches.map { $0.snapshot() }
            // verify
            let (H2, lg2) = model.forwardHidden(ud, caches: mainCaches)
            let vw = MLX.argMax(lg2[0, 0 ..< 2], axis: -1)             // [2] : v=pos0, w=pos1
            let triple = MLX.concatenated([dArr[0], vw])              // [d, v, w]
            let vals = triple.asArray(Int32.self)
            let d = Int(vals[0]), v = Int(vals[1]), w = Int(vals[2])
            let u = uArr.item(Int.self)

            out.append(u)
            if v == d {                                                // accept → 2 tokens
                acc += 1
                out.append(d)
                _ = head(H2[0..., 0 ..< 1], dArr, cache: mtpKV)        // catch-up
                uArr = vw[1 ..< 2].reshaped([1, 1]); lastH = H2[0..., 1 ..< 2]
            } else {                                                   // reject → 巻戻し、[u] のみ再投入
                // ★Python の [u,v] 再投入は look-ahead v が cache に重複し長文で破綻。
                //   commit は u のみ→cache を u で終端（u の hidden は reject された d に依存しない）。
                for (i, c) in mainCaches.enumerated() { c.restore(snaps[i], isLinear: isLin[i], trim: 2) }
                let (H1, _) = model.forwardHidden(uArr, caches: mainCaches)   // [u] 1トークン
                uArr = vw[0 ..< 1].reshaped([1, 1]); lastH = H1[0..., 0 ..< 1]
            }
            MLX.eval([uArr, lastH] + [mtpKV.keys, mtpKV.values].compactMap { $0 } + mainCaches.flatMap { $0.stateArrays })
        }
        uArr.eval()
        let secs = Double(DispatchTime.now().uptimeNanoseconds - t0.uptimeNanoseconds) / 1e9
        return (Array(out.prefix(maxTokens)), steps, acc, secs)
    }

    public static func run(modelDir: String, refPath: String) throws -> String {
        let r = try loadArrays(url: URL(fileURLWithPath: refPath))
        guard let promptArr = r["spec_prompt"], let gRef = r["spec_greedy"],
              let spRef = r["spec_spec"] else {
            return "[M2c spec] skip: mtp ref に spec_prompt が無い（PY -m qwisp.mtp_ref 再実行）"
        }
        let ids = promptArr.asType(.int32).reshaped([1, promptArr.dim(0)])
        let store = try WeightStore(modelDir: modelDir)
        store.residentAll()                       // resident regime で投機の素の速度を測る
        let model = QwispModel(store: store)
        let head = try MTPHead(modelDir: modelDir, store: store)

        let N = Swift.min(Int(ProcessInfo.processInfo.environment["QWISP_GEN"] ?? "48") ?? 48,
                          gRef.dim(0))
        let g = greedy(model, ids, N)
        let (sp, steps, acc, secs) = speculative(model, head, ids, N)
        let gR = gRef.asArray(Int32.self).map { Int($0) }
        let spR = spRef.asArray(Int32.self).map { Int($0) }
        let vsGreedy = zip(g, sp).filter { $0 == $1 }.count          // lossless
        let vsPython = zip(sp, spR).filter { $0 == $1 }.count        // port 正しさ
        let greedyMatch = zip(g, gR).filter { $0 == $1 }.count       // greedy も Python 一致

        // greedy の素の tok/s
        let tg0 = DispatchTime.now()
        _ = greedy(model, ids, N)
        let gSecs = Double(DispatchTime.now().uptimeNanoseconds - tg0.uptimeNanoseconds) / 1e9

        let ok = vsPython == N && vsGreedy == N
        return String(format: """
            [M2c] MTP 投機デコード(実プロンプト): vs greedy %d/%d(lossless) vs Python spec %d/%d  greedy=Python %d/%d  %@
               accept=%.3f (steps=%d, Python 0.846)  speculative %.1f tok/s vs greedy %.1f tok/s = %.2fx
            """,
            vsGreedy, N, vsPython, N, greedyMatch, N, ok ? "OK ✅" : "❌",
            Double(acc) / Double(steps), steps,
            Double(N) / secs, Double(N) / gSecs, gSecs / secs)
    }
}
