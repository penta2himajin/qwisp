import Foundation
import MLX
import Metal

/// Tell runtime（William Tell = 的=expert を先読みして射抜く）.
/// **標準手法 = SuffixSpec**（SuffixDecoding draft + batched f32-full exact verify, lossless）をここに置く。
/// 既定実行(QWISP_RUN 無指定)はこれ。全領域で Pareto 最適ゆえ task 別 dispatch は不要。
/// 旧ベースライン(SpecK/Fast)・各種探索バリアントは TellExperiments.swift。env ヘルパ(envXxx)は共用。
/// 正典: notes/01-speedup-investigation.md。
public enum Tell {
    // env 読み出しヘルパ（ProcessInfo の冗長な記述を集約）。Tell.envXxx で全 runner から利用。
    static func envInt(_ k: String, _ d: Int) -> Int { Int(ProcessInfo.processInfo.environment[k] ?? "") ?? d }
    static func envFloat(_ k: String, _ d: Float) -> Float { Float(ProcessInfo.processInfo.environment[k] ?? "") ?? d }
    static func envStr(_ k: String, _ d: String) -> String { ProcessInfo.processInfo.environment[k] ?? d }
    static func envFlag(_ k: String) -> Bool { ProcessInfo.processInfo.environment[k] == "1" }
    /// **SuffixDecoding draft + clean exact verify（issue #2 軸B, 訓練不要）**
    /// - 機構: prompt+生成履歴の suffix を lookup し、過去に同 suffix の後に続いた token 列を K 個まで
    ///   無料 draft → batched f32-full exact verify で照合 → 一致 prefix を commit。draft cost ~0。
    /// - lossless: **strict**（batched f32-full verify が逐次 decode と bit-exact）。token/step は反復性に依存。
    /// - 速度: code/agentic の反復で高 accept→高 token/step。free-form(high-entropy)では accept 低下。
    ///   実測 8GB C=64: mix 88 tok/s @maxK24 / 16GB C=128: mix 133 tok/s @maxK24-48（vs Swift-greedy 100%）。
    /// - 研究: SuffixDecoding (arXiv:2411.04975), Prompt-Lookup Decoding
    /// - env: QWISP_RUN=suffix-spec / QWISP_CACHE_C / QWISP_DRAFT_K(最大draft長,既定=C×3/8=最速安全上限) /
    ///   QWISP_SUFFIX_MIN(最小一致) / QWISP_SUFFIX_MATCH(最大一致) / QWISP_SWIFT_REF / QWISP_SPECK_PROF /
    ///   QWISP_F32_ATTN・QWISP_F32_CONV(既定1=f32-full, 0 で f16 batched) / QWISP_VERIFY_SEQ・QWISP_VERIFY_PQN(診断用逐次化)
    public static func runSuffixSpec(modelDir: String, refPath: String) throws -> String {
        guard let device = MTLCreateSystemDefaultDevice() else { return "ERROR: no Metal device" }
        let r = try loadArrays(url: URL(fileURLWithPath: refPath))
        guard let promptArr = r["spec_prompt"], let gRef = r["spec_greedy"] else { return "[SuffixSpec] skip" }
        // ★ C 既定は device 別自動選択(calibration layer): RAM tier 8→64/16→128/24→192/32+→256。
        //   QWISP_CACHE_C で明示上書き可。選択構成をログ出力。
        let C = Tell.envInt("QWISP_CACHE_C", DeviceCalibration.defaultC())
        if ProcessInfo.processInfo.environment["QWISP_CACHE_C"] == nil {
            print("[calibration] " + DeviceCalibration.recommend().summary)
        }
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
        // ★既定は C 安全上限(=最速。長 draft は反復で accept↑、novel では suffix が短く返すので中立)。
        let maxKSafe = Swift.max(4, C * 3 / 8)
        let maxKReq = Tell.envInt("QWISP_DRAFT_K", maxKSafe)
        let maxK = Swift.min(maxKReq, maxKSafe)
        if maxK < maxKReq {
            print("[SuffixSpec] maxK \(maxKReq)→\(maxK) にクランプ(C=\(C) の arena 容量制約 C×3/8, |U|>C 回避)")
        }
        let minMatch = Tell.envInt("QWISP_SUFFIX_MIN", 2)
        let maxMatch = Tell.envInt("QWISP_SUFFIX_MATCH", 32)
        // 既定 f32-full(QWISP_F32_ATTN/CONV=0 で各々無効化可)。f16 batched を試すなら両方 0。
        GatedDeltaNetLayer.f32Conv = Tell.envStr("QWISP_F32_CONV", "1") != "0"
        AttentionLayer.f32SDPA = Tell.envStr("QWISP_F32_ATTN", "1") != "0"
        defer {
            GatedDeltaNetLayer.f32Conv = false; AttentionLayer.f32SDPA = false; AttentionLayer.perQueryNone = false
            StreamingMoEBlock.probeNoSync = false; StreamingMoEBlock.countHotMiss = false; StreamingMoEBlock.hotMissAccum = nil
        }
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
        if Tell.envFlag("QWISP_PIN_DBG") {   // ★ no-sync C 依存バグ診断: hot-pin 充填 vs working-set
            for mi in [0, 1, 20] where mi < model.expertCaches.count {
                let ec = model.expertCaches[mi]
                let ws = counts[mi].enumerated().filter { $0.element > 0 }.map { $0.offset }   // calib working set
                let cached = ws.filter { ec.slotMap[$0] != nil }.count
                let slotMax = ec.slotMap.values.max() ?? -1
                let msg = "DBG-PIN layer\(mi) C=\(C): slotMap充填=\(ec.slotMap.count) working-set=\(ws.count) うち cached=\(cached) slot最大=\(slotMax)\n"
                FileHandle.standardError.write(msg.data(using: .utf8)!)
            }
        }

        // phase 3: suffix-lookup draft + clean exact verify
        // ★ lever-1(issue#3): per-layer routing round-trip 除去で ~1.7-1.8x lossless（dispatch/sync 律速）。
        //   C>=nE(256): 全 expert resident=no-sync gather(GPU slot-table remap)が無条件 bit-exact → pure no-sync。
        //   C < nE: 既定 sync(cold-start の greedy working set は frequency-pin に収まらず escalation が
        //     net 損＝warming 浪費。escalation の greedy 高速化は working set 事前常駐時のみ＝実質 C=256)。
        //   研究用に QWISP_NOSYNC_MIN=<C> を明示すると [その値, nE) で no-sync+escalation(率監視 fallback)を有効化。
        //   QWISP_NOSYNC=1 強制 pure / =0 無効。出力は常に exact(pure は residency 保証, escalation は sync 再計算)。
        let nosyncEnv = Tell.envStr("QWISP_NOSYNC", "auto")
        let escalMin = Tell.envInt("QWISP_NOSYNC_MIN", nE)   // 既定 nE=band 空=production は pure/sync のみ
        let pureNoSync = nosyncEnv == "1" || (nosyncEnv != "0" && C >= nE)
        var escalateActive = nosyncEnv != "0" && !pureNoSync && C >= escalMin
        // ★ no-sync+escalate を verify(batched M=D+1)にも適用する実験 flag。mix は draft が similar→union 小→
        //   C 内に収まれば no-sync exact(per-layer sync 税を消し ~resident 天井へ)、超過時のみ escalate 再計算で
        //   lossless。streaming mix の 130→276 gap を狙う。QWISP_VERIFY_ESCALATE=1 + QWISP_NOSYNC_MIN=<=C で有効。
        let verifyEscalate = Tell.envFlag("QWISP_VERIFY_ESCALATE")
        // ★ ガード: pure no-sync は C>=nE(全 resident)でのみ無条件 lossless。C<nE では uncached expert が
        //   slot-0 に alias する **近似**(2 agent 検証: verify が近似→true model に lossy, near-tie 非決定)。
        //   QWISP_NOSYNC=1 を C<nE で許すのは研究/計測用途のみ→「lossless」と誤認しないよう明示警告。
        if pureNoSync && C >= nE {
            print("[SuffixSpec] no-sync pure ON (C=\(C)>=\(nE) 全 resident, 無条件 lossless ~1.7x)")
        } else if pureNoSync {
            print("[SuffixSpec] ⚠️ no-sync pure ON (C=\(C)<\(nE)) = **APPROXIMATE / near-lossless**"
                + "(uncached expert→slot-0 alias, lossless 非保証・研究/計測用。lossless 化は QWISP_VERIFY_ESCALATE)")
        } else if escalateActive { print("[SuffixSpec] no-sync+escalation ON (C=\(C) in [\(escalMin),\(nE)), exact, 率監視 fallback)") }
        var hist = ids.asArray(Int32.self).map { Int($0) }     // 履歴（prompt + commit token）
        let mc = model.makeCaches()
        StreamingMoEBlock.probeNoSync = pureNoSync
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
        // adaptive escalation: escalateActive 時、input を no-sync で forward→cold routed があれば cache を
        //   巻戻して sync 再計算(exact)。escalation 率が高ければ escalateActive を落とし以降 sync(footprint 壁で無回帰)。
        //   非 escalation 時は global probeNoSync(pure か false)のまま素通し。返す logits は常に exact。
        //   escalate=false(verify 多トークン)では escalation 無効: working set が大きく必ず miss→2x 損。
        //   pure no-sync(C>=256)は global probeNoSync で verify も安全に no-sync 化される。
        //   fallback 判定は移動窓 + grace: escalate は cache を warm する(sync 再計算が cold expert を ensure)ので
        //   cold start は高 escalation でも収束しうる。grace 後の recent rate が break-even(no-sync 0.5x +
        //   escalate 2x ゆえ rate>~0.5 で sync より損)を超えたら footprint 壁と判断し sync へ自動 fallback。
        var escSeen = 0
        var escRecent: [Bool] = []
        let escWindow = Tell.envInt("QWISP_ESC_WINDOW", 16)
        let escGrace = Tell.envInt("QWISP_ESC_GRACE", 24)
        let escMaxRate = 0.45
        func decodeForward(_ input: MLXArray, rows: Int, escalate: Bool) throws -> MLXArray {
            if !escalateActive || !escalate {
                let (_, l) = try model.forwardHidden(input, caches: mc)
                return l
            }
            let snaps = mc.map { $0.snapshot() }
            StreamingMoEBlock.hotMissAccum = nil
            StreamingMoEBlock.probeNoSync = true; StreamingMoEBlock.countHotMiss = true
            let (_, l) = try model.forwardHidden(input, caches: mc)
            let missArr = StreamingMoEBlock.hotMissAccum ?? MLXArray(Int32(0))
            MLX.eval([l, missArr] + mc.flatMap { $0.stateArrays })
            StreamingMoEBlock.countHotMiss = false; StreamingMoEBlock.probeNoSync = false
            escSeen += 1
            var result = l
            let escalated = missArr.item(Int32.self) != 0
            if escalated {                                           // cold routed あり → sync 再計算(exact, cold ensure)
                for (i, c) in mc.enumerated() { c.restore(snaps[i], isLinear: isLin[i], trim: rows) }
                let (_, l2) = try model.forwardHidden(input, caches: mc)
                result = l2
            }
            escRecent.append(escalated); if escRecent.count > escWindow { escRecent.removeFirst() }
            if escSeen >= escGrace && escRecent.count == escWindow {  // warming 後に移動窓で判定
                let rate = Double(escRecent.filter { $0 }.count) / Double(escWindow)
                if rate > escMaxRate {
                    escalateActive = false
                    print(String(format: "[SuffixSpec] recent escalation率 %.0f%% > %.0f%% → sync へ fallback(footprint 壁, 無回帰)",
                                 rate * 100, escMaxRate * 100))
                }
            }
            return result
        }
        // ★ accept-gated 適応 maxK: 直近の受理長で draft 窓を調整。低 accept(nl)→maxK 縮小(≈greedy)で
        //   resident の f32 長 verify 過剰コスト回避(nl maxK96=55 → maxK≈1=63 回復)、高 accept(mix)→full maxK。
        //   混在ストリーム(散文↔コード)も accept が自動追従。既定 on@resident(C>=nE、streaming は nl penalty 無で中立)。
        let adaptiveK = (ProcessInfo.processInfo.environment["QWISP_ADAPTIVE_K"].map { $0 == "1" }) ?? (C >= nE)
        let adaptWin = Tell.envInt("QWISP_ADAPT_WINDOW", 8)
        let adaptGrace = Tell.envInt("QWISP_ADAPT_GRACE", 2)
        var accWindow: [Int] = []                                // 直近 step の受理長 p(D==0 は 0)
        // ★ overflow で縮む動的安全上限。union-overflow guard 発火の度に半減し、その内容が
        //   「per-layer union≤C に収まる draft 長」へ収束(毎 step の double-forward 浪費を回避)。adaptive on/off 問わず有効。
        var safeMaxK = maxK
        func effMaxK() -> Int {
            let cap = Swift.min(maxK, safeMaxK)
            if !adaptiveK || accWindow.isEmpty { return cap }
            let mean = Double(accWindow.reduce(0, +)) / Double(accWindow.count)
            return Swift.max(1, Swift.min(cap, Int(mean.rounded()) + adaptGrace))
        }
        func recordAccept(_ p: Int) { accWindow.append(p); if accWindow.count > adaptWin { accWindow.removeFirst() } }
        if adaptiveK { print("[SuffixSpec] accept-gated 適応 maxK ON(window=\(adaptWin) grace=\(adaptGrace), maxK≤\(maxK))") }
        let missDbg = Tell.envFlag("QWISP_VERIFY_MISS_DBG")
        let ofDbg = Tell.envFlag("QWISP_OVERFLOW_DBG")
        let ofMargin = Swift.max(10, Swift.min(99, Tell.envInt("QWISP_OVERFLOW_MARGIN", 80)))  // safe-union 目標 %（既定80）
        var missAccumDbg = 0, missStepsDbg = 0, missRoutedDbg = 0
        var out: [Int] = []; var steps = 0, accTok = 0, draftTot = 0, overflowCount = 0
        let t0 = DispatchTime.now()
        while out.count < N {
            steps += 1
            let u = uArr.item(Int.self)
            var ts = now()
            let drafts = suffixDraft(hist + [u], maxMatch: maxMatch, draftK: effMaxK(), minMatch: minMatch)
            let D = drafts.count
            draftTot += D
            if prof { tDraft += now() - ts; ts = now() }
            if D == 0 {                                          // 一致なし → 通常 greedy 1 step
                let glg = try decodeForward(uArr, rows: 1, escalate: true)
                out.append(u); hist.append(u)
                uArr = MLX.argMax(glg[0, 0], axis: -1).reshaped([1, 1])
                MLX.eval([uArr] + mc.flatMap { $0.stateArrays })
                recordAccept(0)                                  // draft 不発も「speculation 非生産」として窓へ
                if prof { tVerify += now() - ts }
                continue
            }
            setVerifyMode(true)
            let snaps = mc.map { $0.snapshot() }
            // ★ union-overflow guard(C<nE)+ safe-prefix re-verify: batched verify の per-layer expert union が
            //   C を超えると sync ensure が evict しきれず wrong-slot=silent garbage(hotMiss 非検出)で誤受理する。
            //   実測: C=64 code は maxK=C×3/8 でも 3/4 rep 非lossless(clamp は worst-case 過楽観, union~2×C)。
            //   検出は ensure に渡る CPU 側 [Int] の distinct 数(GPU sync 不要=安価。lastInds materialize は 19x 遅)。
            //   overflow なら draft を「union≤C に収まる最長 prefix」へ縮小して RE-VERIFY(single-token でなく
            //   複数 accept を回収)。比例縮小で 1-2 回で収束、prefix<1 でのみ safe single-token。strict-lossless 保証。
            let unionGuard = C < nE
            var curDrafts = drafts
            var vlg: MLXArray = uArr                              // placeholder（loop で必ず上書き）
            var singleFallback = false
            while true {
                let dd = curDrafts.count
                let seq = MLX.concatenated([uArr, MLXArray(curDrafts.map { Int32($0) }, [1, dd])], axis: 1)  // [1, dd+1]
                if missDbg { StreamingMoEBlock.countHotMiss = true; StreamingMoEBlock.hotMissAccum = nil }
                if unionGuard { LayerExpertCache.overflowCheck = true; LayerExpertCache.overflowMaxUnion = 0 }
                vlg = try decodeForward(seq, rows: dd + 1, escalate: verifyEscalate)
                if missDbg {
                    let ma = StreamingMoEBlock.hotMissAccum ?? MLXArray(Int32(0)); MLX.eval([vlg, ma])
                    missAccumDbg += Int(ma.item(Int32.self)); missStepsDbg += 1
                    missRoutedDbg += (dd + 1) * nMoE * 8
                    StreamingMoEBlock.countHotMiss = false
                }
                if !unionGuard { break }
                LayerExpertCache.overflowCheck = false
                let maxU = LayerExpertCache.overflowMaxUnion
                if ofDbg { FileHandle.standardError.write("OFDBG C=\(C) D=\(dd) maxU=\(maxU) safeMaxK=\(safeMaxK)\n".data(using: .utf8)!) }
                if maxU > C {                                    // overflow → 収まる最長 prefix へ縮小し re-verify
                    let target = Swift.max(1, dd * (C * ofMargin / 100) / maxU)   // union≈線形 in D の保守推定
                    safeMaxK = Swift.min(safeMaxK, target)
                    overflowCount += 1
                    for (i, c) in mc.enumerated() { c.restore(snaps[i], isLinear: isLin[i], trim: dd + 1) }
                    let newLen = Swift.min(dd - 1, target)
                    if newLen < 1 { singleFallback = true; break }
                    curDrafts = Array(curDrafts.prefix(newLen))
                    continue
                } else if maxU > 0 && maxU * 100 < C * ofMargin && safeMaxK < maxK {
                    safeMaxK = Swift.min(maxK, safeMaxK + Swift.max(2, safeMaxK / 6))  // 余裕十分→成長
                }
                break                                            // union≤C=fit → accept へ
            }
            if singleFallback {                                  // 安全 prefix が 1 未満 → single-token(union≤top8≤C)
                let glg = try decodeForward(uArr, rows: 1, escalate: true)
                out.append(u); hist.append(u)
                uArr = MLX.argMax(glg[0, 0], axis: -1).reshaped([1, 1])
                setVerifyMode(false)
                MLX.eval([uArr] + mc.flatMap { $0.stateArrays })   // recordAccept は呼ばない(容量不足≠非生産)
                if prof { tVerify += now() - ts }
                continue
            }
            let D2 = curDrafts.count
            let evals = MLX.argMax(vlg[0, 0 ..< (D2 + 1)], axis: -1).asArray(Int32.self).map { Int($0) }
            var p = 0
            while p < D2 && curDrafts[p] == evals[p] { p += 1 }
            out.append(u); hist.append(u)
            for i in 0 ..< p { out.append(curDrafts[i]); hist.append(curDrafts[i]) }
            accTok += p
            recordAccept(p)                                      // ★ 適応 maxK 用に受理長を記録
            if p == D2 {
                uArr = MLXArray([Int32(evals[D2])], [1, 1])
                setVerifyMode(false)
                MLX.eval([uArr] + mc.flatMap { $0.stateArrays })
            } else {
                for (i, c) in mc.enumerated() { c.restore(snaps[i], isLinear: isLin[i], trim: D2 + 1) }
                let acc = [u] + Array(curDrafts.prefix(p))
                _ = try model.forwardHidden(MLXArray(acc.map { Int32($0) }, [1, acc.count]), caches: mc)
                setVerifyMode(false)
                uArr = MLXArray([Int32(evals[p])], [1, 1])
                MLX.eval([uArr] + mc.flatMap { $0.stateArrays })
            }
            if prof { tVerify += now() - ts }
        }
        AttentionLayer.seqMultiToken = false; AttentionLayer.perQueryNone = false
        LayerExpertCache.overflowCheck = false
        if missDbg && missStepsDbg > 0 {
            FileHandle.standardError.write("DBG-MISS C=\(C): verify \(missStepsDbg) 回, hotMiss 累積=\(missAccumDbg) / routed \(missRoutedDbg) = \(String(format: "%.2f%%", Double(missAccumDbg)/Double(max(1,missRoutedDbg))*100)) (avg \(missAccumDbg/max(1,missStepsDbg))/verify)\n".data(using: .utf8)!)
        }
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
        let ovTag = overflowCount > 0 ? "  [union-overflow guard: \(overflowCount) step 安全 fallback]" : ""
        return String(format: """
            [SuffixSpec] suffix draft(maxK=%d) + clean exact verify(C=%d): %.1f tok/s  accept/step=%.2f  品質(vs Python) %d/%d=%.0f%%%@%@
            """, maxK, C, Double(N) / secs, Double(accTok) / Double(steps), match, N, Double(match) / Double(N) * 100, swiftTag, ovTag)
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
