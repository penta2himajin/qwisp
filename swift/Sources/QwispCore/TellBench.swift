// ★ T1: single-process batch bench runner。従来 bench.sh は cell（method×regime）毎に fresh process を
// 起動し、1 cell = 1-2 model load（各 ~10-15s、slow-NAND throttle 下ではさらに高価）×8 cell だった。
// ここでは WeightStore/ExpertSource/ExpertArena/StreamingQwispModel を 1 回だけ構築し、
// methods × regimes を in-process でループする。忠実性の担保:
//   - 各 cell 冒頭で resetCellState()（全 mode static + per-layer LayerExpertCache instance reset）
//     = fresh process の cold-start と意味的に等価（OS page cache の warm さは従来 multi-process
//     bench でも同一なので fidelity 上の差ではない。self-imposed preadInto throttle は常に適用）。
//   - 各 core（suffixSpecCore/boltCore/mlxFidelityCore）は entry で依存 static を明示 set、
//     fresh caches（makeCaches）を内部で作る。
// 出力（machine-parseable, bench_batch.sh が後処理）:
//   BENCHCELL|method=<m>|regime=<r>            … cell 区切り
//   （core の summary + QWISP_DUMP_TOKENS=1 時の PROMPT_TOKENS/OUT_TOKENS/BOLT_TOKENS dump）
//   BENCH|method=<m>|regime=<r>|tokps=<v>      … speed 軸
//   BENCHFID|method=bolt|regime=<r>|fid=X/Y=Z% … bolt の teacher-forced fidelity 軸
// env: QWISP_BENCH_REFS_DIR（必須）, QWISP_BENCH_REGIMES（既定 "code agentic longctx shortnl"）,
//      QWISP_BENCH_METHODS（既定 "suffix-spec bolt"）+ 通常の QWISP_CACHE_C/QWISP_GEN/
//      QWISP_SSD_THROTTLE_GBS/QWISP_THROTTLE_DEFER/QWISP_DUMP_TOKENS 等。
// 起動: QWISP_RUN=bench-batch qwisp-poc stream
import Foundation
import MLX
import Metal

extension Tell {
    /// 1 cell を fresh-process 相当の cold 状態に戻す（static AND instance state の全 reset）。
    /// - per-layer LayerExpertCache: slotOf/expertAt/tick/clock/hits/misses/lastInds ほか capture 群/
    ///   slotTableGPU(+version bump)/pinnedSlots/hotMask/buddyTable → instance reset()
    /// - LayerExpertCache statics: ensureNanos/preadNanos/missTotal/overflowCheck/overflowMaxUnion
    /// - StreamingMoEBlock statics: syncNanos/probeNoSync/predictOnly/captureGateInput/captureInds/
    ///   syncLayers/captureLayerInput/captureK/marginK/countHotMiss/skipMode/hotMissAccum/
    ///   profileLayers/各 timer
    /// - GatedDeltaNetLayer.fuseGDN/f32Conv, AttentionLayer.f32SDPA/orderStable/boolMaskSDPA/
    ///   perQueryNone/seqMultiToken
    /// - ExpertSource: acct カウンタ + throttle virtualClock（resetAcct）、throttleActive を
    ///   process-fresh 既定（!throttleDefer）へ
    /// KV/GDN cache は各 core が makeCaches() で毎回新規に作るためここでは対象外。
    static func resetCellState(model: StreamingQwispModel) {
        for ec in model.expertCaches { ec.reset() }
        LayerExpertCache.resetGlobals()
        StreamingMoEBlock.resetGlobals()
        GatedDeltaNetLayer.fuseGDN = false
        GatedDeltaNetLayer.f32Conv = false
        AttentionLayer.f32SDPA = false
        AttentionLayer.orderStable = false
        AttentionLayer.boolMaskSDPA = false
        AttentionLayer.perQueryNone = false
        AttentionLayer.seqMultiToken = false
        ExpertSource.resetAcct()
        ExpertSource.throttleActive = !ExpertSource.throttleDefer   // T2: cell 毎に gate を初期状態へ
    }

    /// ★ T1: single-process batch bench（QWISP_RUN=bench-batch）。1 model load で
    /// methods × regimes の全 cell（speed + bolt TF fidelity）を回す。
    public static func runBenchBatch(modelDir: String) throws -> String {
        guard let device = MTLCreateSystemDefaultDevice() else { return "ERROR: no Metal device" }
        let refsDir = Tell.envStr("QWISP_BENCH_REFS_DIR", "")
        guard !refsDir.isEmpty else { return "ERROR: QWISP_BENCH_REFS_DIR unset" }
        let regimes = Tell.envStr("QWISP_BENCH_REGIMES", "code agentic longctx shortnl")
            .split(separator: " ").map(String.init)
        let methods = Tell.envStr("QWISP_BENCH_METHODS", "suffix-spec bolt")
            .split(separator: " ").map(String.init)
        let C = Tell.envInt("QWISP_CACHE_C", DeviceCalibration.defaultC())
        if ProcessInfo.processInfo.environment["QWISP_CACHE_C"] == nil {
            print("[calibration] " + DeviceCalibration.recommend().summary)
        }
        // ★ 単一構築（従来 bench.sh の per-cell 8-12 model load を 1 回に）
        let store = try WeightStore(modelDir: modelDir)
        store.residentNonExperts()
        let source = try ExpertSource(modelDir: modelDir); try source.warm()
        let arena = try ExpertArena(device: device, source: source, N: 64)
        let model = try StreamingQwispModel(store: store, arena: arena, device: device, source: source, cacheC: C)

        var cells = 0
        for m in methods {
            for r in regimes {
                let refPath = refsDir + "/" + r + ".safetensors"
                guard FileManager.default.fileExists(atPath: refPath) else {
                    print("BENCHSKIP|method=\(m)|regime=\(r)|reason=missing-ref \(refPath)")
                    continue
                }
                let ra = try loadArrays(url: URL(fileURLWithPath: refPath))
                guard let promptArr = ra["spec_prompt"], let gRef = ra["spec_greedy"] else {
                    print("BENCHSKIP|method=\(m)|regime=\(r)|reason=ref-missing-keys")
                    continue
                }
                let ids = promptArr.asType(.int32).reshaped([1, promptArr.dim(0)])
                let gR = gRef.asArray(Int32.self).map { Int($0) }
                print("BENCHCELL|method=\(m)|regime=\(r)")
                // --- speed cell（free-run; QWISP_DUMP_TOKENS=1 なら core が token dump も print）---
                resetCellState(model: model)
                let res: (summary: String, tokps: Double)
                switch m {
                case "suffix-spec":
                    res = try Tell.suffixSpecCore(model: model, ids: ids, gR: gR, C: C, modelDir: modelDir)
                case "bolt":
                    res = try Tell.boltCore(model: model, ids: ids, gR: gR, C: C)
                default:
                    print("BENCHSKIP|method=\(m)|regime=\(r)|reason=unknown-method")
                    continue
                }
                print(res.summary)
                print(String(format: "BENCH|method=%@|regime=%@|tokps=%.1f", m, r, res.tokps))
                cells += 1
                // --- bolt のみ: teacher-forced fidelity を in-process で続けて計測
                //     （strict の fidelity 軸は T0 の token compare が shell 側で担う）。
                //     bench.sh 同様 throttle 非依存の軸なので unthrottled で実行。---
                if m == "bolt" {
                    resetCellState(model: model)
                    let savedThr = ExpertSource.throttleGBs
                    ExpertSource.throttleGBs = 0
                    ExpertSource.throttleActive = true
                    let f: (summary: String, match: Int, n: Int)
                    do {
                        f = try Tell.mlxFidelityCore(model: model, ids: ids, gR: gR, C: C, buddy: true)
                    } catch {
                        ExpertSource.throttleGBs = savedThr
                        throw error
                    }
                    ExpertSource.throttleGBs = savedThr
                    print(f.summary)
                    print(String(format: "BENCHFID|method=%@|regime=%@|fid=%d/%d=%.1f%%",
                                 m, r, f.match, f.n, Double(f.match) / Double(f.n) * 100))
                }
            }
        }
        return "[bench-batch] done: \(cells) cells (methods=\(methods.joined(separator: ","))"
            + " regimes=\(regimes.joined(separator: ",")) C=\(C)) — 1 model load total"
    }
}
