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
    /// - env: QWISP_RUN=suffix-spec / QWISP_CACHE_C / QWISP_DRAFT_K(debug 専用 override, 既定=C×3/8 容量式;
    ///   per-step 長は α·p + guard で機械的) / QWISP_SWIFT_REF / QWISP_SPECK_PROF /
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
        let store = try WeightStore(modelDir: modelDir)
        store.residentNonExperts()
        let source = try ExpertSource(modelDir: modelDir); try source.warm()
        let arena = try ExpertArena(device: device, source: source, N: 64)
        let model = try StreamingQwispModel(store: store, arena: arena, device: device, source: source, cacheC: C)
        let ids = promptArr.asType(.int32).reshaped([1, promptArr.dim(0)])
        let gR = gRef.asArray(Int32.self).map { Int($0) }
        return try suffixSpecCore(model: model, ids: ids, gR: gR, C: C).summary
    }

    /// ★ T1: SuffixSpec 本体 core。prebuilt model を受け取り、runSuffixSpec と TellBench(batch bench)
    /// の両方から呼ぶ。in-process 連続実行を想定し、依存する global static は process-fresh 既定を
    /// 仮定せず entry で全て明示 set し、exit(defer)で reset する。timed decode 直前に
    /// ExpertSource.throttleActive を立てる（T2: QWISP_THROTTLE_DEFER=1 時のみ意味を持つ）。
    static func suffixSpecCore(model: StreamingQwispModel, ids: MLXArray, gR: [Int], C: Int)
        throws -> (summary: String, tokps: Double) {
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
        //   ★但し C×3/8 は真の安全境界でない: diverse routing で union は ~2×C まで膨張=lossless-by-luck。
        //   実効上限は下記 union-overflow guard が実 routing を観測して動的決定(strict-lossless)。
        // ★既定は C 上限(反復で長 draft が accept↑、novel では suffix が短く返す)。overflow は guard が回収。
        let maxKSafe = Swift.max(4, C * 3 / 8)
        let maxKReq = Tell.envInt("QWISP_DRAFT_K", maxKSafe)   // debug-only override（既定=C×3/8 容量式）
        let maxK = Swift.min(maxKReq, maxKSafe)
        if maxK < maxKReq {
            print("[SuffixSpec] maxK \(maxKReq)→\(maxK) にクランプ(C=\(C) の arena 容量制約 C×3/8, |U|>C 回避)")
        }
        let minMatch = 4    // OAT-tuned(9b157d9): 2→4 で偶然一致の無駄 draft 回避、最適近傍で鈍感
        let maxMatch = 32   // OAT-tuned(9b157d9): 最適近傍で鈍感
        // ★ union-overflow guard: maxK=C×3/8 は真の安全境界でない。diverse routing で per-layer expert
        //   union は ~2×C まで膨張し、C<nE では sync ensure が evict しきれず silent garbage=lossless-by-luck。
        //   実 routing の union を観測し overflow prefix を re-verify して strict-lossless 化。詳細 notes/00。
        //   制御則は exact safe-prefix(overflowSafeRows) + hysteresis（guard loop 内コメント参照）。
        let ofDbg = Tell.envFlag("QWISP_OVERFLOW_DBG")
        // ★ entry で依存 static を全て明示 set（in-process 連続実行では前 run の leak が実バグになる）。
        // 既定 f32-full(QWISP_F32_ATTN/CONV=0 で各々無効化可)。f16 batched を試すなら両方 0。
        GatedDeltaNetLayer.f32Conv = Tell.envStr("QWISP_F32_CONV", "1") != "0"
        AttentionLayer.f32SDPA = Tell.envStr("QWISP_F32_ATTN", "1") != "0"
        // B2 試験 knob: GDN in_proj 融合(out軸 concat=bit-exact)。opt-in で A/B 計測(bolt は default ON)。
        GatedDeltaNetLayer.fuseGDN = Tell.envFlag("QWISP_FUSE_GDN")
        AttentionLayer.seqMultiToken = false; AttentionLayer.perQueryNone = false
        StreamingMoEBlock.probeNoSync = false; StreamingMoEBlock.skipMode = 0
        StreamingMoEBlock.captureInds = false
        StreamingMoEBlock.countHotMiss = false; StreamingMoEBlock.hotMissAccum = nil
        LayerExpertCache.overflowCheck = false; LayerExpertCache.overflowMaxUnion = 0
        LayerExpertCache.overflowSafeRows = Int.max
        defer {
            GatedDeltaNetLayer.f32Conv = false; AttentionLayer.f32SDPA = false; AttentionLayer.perQueryNone = false
            GatedDeltaNetLayer.fuseGDN = false
            AttentionLayer.seqMultiToken = false
            StreamingMoEBlock.probeNoSync = false; StreamingMoEBlock.skipMode = 0
            StreamingMoEBlock.captureInds = false
            StreamingMoEBlock.countHotMiss = false; StreamingMoEBlock.hotMissAccum = nil
            LayerExpertCache.overflowCheck = false
        }
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
        if pureNoSync { print("[SuffixSpec] no-sync pure ON (C=\(C)>=\(nE) 全 resident, 無条件 lossless ~1.7x)") }
        else if escalateActive { print("[SuffixSpec] no-sync+escalation ON (C=\(C) in [\(escalMin),\(nE)), exact, 率監視 fallback)") }
        var hist = ids.asArray(Int32.self).map { Int($0) }     // 履歴（prompt + commit token）
        let mc = model.makeCaches()
        StreamingMoEBlock.probeNoSync = pureNoSync
        var (_, lg) = try model.prefillChunked(ids, caches: mc)
        var uArr = MLX.argMax(lg[0, lg.dim(1) - 1], axis: -1).reshaped([1, 1])
        MLX.eval([uArr] + mc.flatMap { $0.stateArrays })
        // ★2026-07-02: VERIFY_SEQ を default ON に（strict-lossless 修正）。batched f32 verify は
        //   order-stable でない（M=D+1 と M=1 で kernel 累積順が異なり微小 drift → near-tie の誤 accept。
        //   agentic で C 非依存に 80/128=62% へ分岐する反例を確認。PQN=SDPA のみ逐次化では直らず、
        //   seqMT=層丸ごと per-token で 128/128 回復）。コスト実測 agentic −5%/longctx −0%。
        //   QWISP_VERIFY_SEQ=0 で旧 batched（near-tie 誤 accept の保証なし）に戻せる。
        let vseq = Tell.envStr("QWISP_VERIFY_SEQ", "1") != "0"
        let vpqn = Tell.envFlag("QWISP_VERIFY_PQN")
        func setVerifyMode(_ on: Bool) {
            if vpqn { AttentionLayer.perQueryNone = on; AttentionLayer.seqMultiToken = false }
            else { AttentionLayer.seqMultiToken = on && vseq }
        }
        let prof = Tell.envFlag("QWISP_SPECK_PROF")
        // QWISP_ACCEPT_TRACE=1: accept-length histogram + fork analysis (k=2 draft prize).
        // Flag off = zero behavior change (guards only). Dumped as [AcceptTrace] at loop end.
        let acceptTrace = Tell.envFlag("QWISP_ACCEPT_TRACE")
        var atHist: [Int: Int] = [:]
        var atDraftless = 0, atSingleFB = 0, atMismatch = 0, atAltExist = 0, atAltHit = 0, atCertStop = 0
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
        var out: [Int] = []; var steps = 0, accTok = 0, draftTot = 0, overflowCount = 0
        // margin-certified accept の telemetry: certStop event 数 / replay M=1 forward 総数 /
        // commit token の内訳（margin-certified vs replay 産）
        var certFallbacks = 0, certReplayFwd = 0, certCommitted = 0, replayCommitted = 0
        var safeMaxK = maxK   // overflow で縮む動的安全上限（union-overflow guard）
        let unionGuard = C < nE
        // ★ A3 "pending prefix": normal reject の rebuild forward（logits 不使用の cache 再構築 =
        //   reject 1 回につき丸ごと 1 forward の無駄）を撤廃し、accepted prefix を「次の」verify に融合する。
        //   pending = out/hist へ commit 済みだが cache に未実体化の token 列。次 step の verify 入力は
        //   seq = pending + [u] + drafts（rows = pending.count + 1 + D）となり、decision row は
        //   offset pending.count から始まる（先頭 pending 行の logits は commit 済み token の予測ゆえ無視）。
        //   CORRECTNESS INVARIANT: KV/GDN cache は position-wise causal ゆえ、pre-pending 状態から
        //   [pending, u, drafts] を 1 回の batched forward で流すのは、[pending] → [u, drafts] と
        //   2 回の逐次 batched forward を流すのと数学的に同一の状態/logits を与える。但し fusion は
        //   batch shape を変えるため kernel 累積順の微小 drift は起こり得る（cert が存在する理由その
        //   もの）— decision row は全て margin-certified され、低 margin は逐次 replay に fallback
        //   するため strict-lossless は保たれる。
        var pending: [Int] = []
        // 連続 reject で pending は非有界に成長し得る → cap 超過で flush し verify サイズ/guard 圧を有界化
        let pendingCap = 24
        var a3Fused = 0, flushes = 0
        /// pending の cache 実体化のみを行う（logits 不使用 = 旧 rebuild-forward 相当のコストを edge case
        /// でだけ払う）。C<nE では union guard 下で safe-prefix chunk に分割 forward（1 row の union は
        /// ≤top8≤C ゆえ必ず前進 = 停止保証）。caller は verify mode（seqMT）を設定済みであること。
        func flushPending() throws {
            if pending.isEmpty { return }
            flushes += 1
            while !pending.isEmpty {
                var chunk = pending
                while true {
                    if unionGuard {
                        LayerExpertCache.overflowCheck = true
                        LayerExpertCache.overflowMaxUnion = 0
                        LayerExpertCache.overflowSafeRows = Int.max
                    }
                    let fsnaps = unionGuard ? mc.map { $0.snapshot() } : []
                    _ = try decodeForward(MLXArray(chunk.map { Int32($0) }, [1, chunk.count]),
                                          rows: chunk.count, escalate: false)
                    if !unionGuard { break }
                    LayerExpertCache.overflowCheck = false
                    if LayerExpertCache.overflowMaxUnion > C && chunk.count > 1 {
                        overflowCount += 1
                        for (i, c) in mc.enumerated() { c.restore(fsnaps[i], isLinear: isLin[i], trim: chunk.count) }
                        chunk = Array(chunk.prefix(Swift.max(1, Swift.min(chunk.count - 1,
                                                                          LayerExpertCache.overflowSafeRows))))
                        continue
                    }
                    break
                }
                MLX.eval(mc.flatMap { $0.stateArrays })
                pending.removeFirst(chunk.count)
            }
        }
        /// D==0 / singleFallback 共通: u を 1 token 進める。pending 非空なら fold（= pending+[u] を
        /// 1 回の batched forward で実体化しつつ最終 row logits から次 token を得る = flush+forward より
        /// 常に 1 forward 少ない）。fold の最終 row argmax は batched logits 由来の commit ゆえ既存の
        /// cert パターン通り margin 認定し、≤τ は pre-pending snapshot からの M=1 逐次 replay で確定。
        func advanceSingle(_ u: Int) throws {
            if pending.isEmpty {                                  // 従来どおりの single-token greedy
                setVerifyMode(false)
                let glg = try decodeForward(uArr, rows: 1, escalate: true)
                out.append(u); hist.append(u)
                uArr = MLX.argMax(glg[0, 0], axis: -1).reshaped([1, 1])
                MLX.eval([uArr] + mc.flatMap { $0.stateArrays })
                return
            }
            let pk = pending.count
            let snaps = mc.map { $0.snapshot() }                  // pre-pending 状態（前 step から不変）
            setVerifyMode(true)
            if unionGuard {
                LayerExpertCache.overflowCheck = true
                LayerExpertCache.overflowMaxUnion = 0
                LayerExpertCache.overflowSafeRows = Int.max
            }
            let flg = try decodeForward(MLXArray((pending + [u]).map { Int32($0) }, [1, pk + 1]),
                                        rows: pk + 1, escalate: false)
            if unionGuard {
                LayerExpertCache.overflowCheck = false
                if LayerExpertCache.overflowMaxUnion > C {
                    // pending+u すら容量 row に fit しない（稀）→ restore + chunked flush + 素の M=1 forward
                    overflowCount += 1
                    for (i, c) in mc.enumerated() { c.restore(snaps[i], isLinear: isLin[i], trim: pk + 1) }
                    try flushPending()
                    setVerifyMode(false)
                    let glg = try decodeForward(uArr, rows: 1, escalate: true)
                    out.append(u); hist.append(u)
                    uArr = MLX.argMax(glg[0, 0], axis: -1).reshaped([1, 1])
                    MLX.eval([uArr] + mc.flatMap { $0.stateArrays })
                    return
                }
            }
            let lastRow = flg[0, pk ..< (pk + 1)]                 // [1, V]: u 行 = 次 token の decision row
            let evArr = MLX.argMax(lastRow, axis: -1)
            let fm1 = MLX.max(lastRow, axis: -1, keepDims: true)
            let fm2 = MLX.max(MLX.where(lastRow .>= fm1, MLXArray(-Float.infinity), lastRow),
                              axis: -1, keepDims: true)
            let fmg = (fm1 - fm2).reshaped([1])
            MLX.eval([evArr, fmg])
            out.append(u); hist.append(u)
            if fmg.asArray(Float.self)[0] > certTau {             // certified → fold 成立（pending 実体化済み）
                certCommitted += 1
                uArr = MLXArray([evArr.asArray(Int32.self)[0]], [1, 1])
                pending = []
                setVerifyMode(false)
                MLX.eval([uArr] + mc.flatMap { $0.stateArrays })
            } else {
                // 未認定 → pre-pending snapshot から pending+[u] を M=1 逐次 replay（無条件 exact）
                certFallbacks += 1
                for (i, c) in mc.enumerated() { c.restore(snaps[i], isLinear: isLin[i], trim: pk + 1) }
                setVerifyMode(false)
                var rlg = flg
                for t in pending + [u] {
                    (_, rlg) = try model.forwardHidden(MLXArray([Int32(t)], [1, 1]), caches: mc)
                    MLX.eval([rlg] + mc.flatMap { $0.stateArrays })
                    certReplayFwd += 1
                }
                uArr = MLX.argMax(rlg[0, 0], axis: -1).reshaped([1, 1])
                MLX.eval([uArr])
                replayCommitted += 1
                pending = []
            }
        }
        ExpertSource.throttleActive = true   // T2: deferred throttle はここ（timed decode 開始）から有効
        let t0 = DispatchTime.now()
        while out.count < N {
            steps += 1
            let u = uArr.item(Int.self)
            var ts = now()
            let drafts = suffixDraft(hist + [u], maxMatch: maxMatch, draftK: Swift.min(maxK, safeMaxK), minMatch: minMatch,
                                     traceAlts: acceptTrace)
            let D = drafts.count
            draftTot += D
            if prof { tDraft += now() - ts; ts = now() }
            if D == 0 {                                          // 一致なし → 通常 greedy 1 step（pending は fold）
                if acceptTrace { atDraftless += 1 }
                try advanceSingle(u)
                if prof { tVerify += now() - ts }
                continue
            }
            setVerifyMode(true)
            // pending 非空時、この snapshot は前 step から不変の pre-pending 状態のスナップ（restore で
            // pre-pending へ戻り、新 pending = pending + [u] + accepted の連結になる）。
            let snaps = mc.map { $0.snapshot() }
            // ★ union-overflow guard(C<nE)+ EXACT safe-prefix + hysteresis: batched verify の per-layer
            //   expert union が C を超えると sync ensure が evict しきれず wrong-slot=silent garbage
            //   (hotMiss 非検出)で誤受理する。検出は ensure に渡る CPU 側 [Int] の distinct 数(GPU sync
            //   不要=安価)。sync 経路の dedup pass が同時に「distinct≤C を満たす最大 complete-row prefix」を厳密計測
            //   (overflowSafeRows=全層 MIN, row 0=u を含む)するため、overflow 時は drafts=safeRows-1 に
            //   縮小すれば構成上 fit → re-verify は overflow event あたり厳密 1 回。safeMaxK に学習。
            //   成長は hysteresis: union≤0.9·C で成長、(0.9C, C] は現状維持(dead band 無しの小 band)。
            //   prefix<1 でのみ safe single-token。strict-lossless 保証。詳細 notes/00。
            let pk = pending.count
            var curDrafts = drafts
            var vlg: MLXArray = uArr                              // placeholder（loop で必ず上書き）
            var singleFallback = false
            while true {
                let dd = curDrafts.count
                let rows = pk + 1 + dd
                // ★A3: verify 入力の先頭に pending を融合（実体化を兼ねる）。decision row offset = pk。
                let seq = MLXArray((pending + [u] + curDrafts).map { Int32($0) }, [1, rows])
                if unionGuard {
                    LayerExpertCache.overflowCheck = true
                    LayerExpertCache.overflowMaxUnion = 0
                    LayerExpertCache.overflowSafeRows = Int.max
                }
                vlg = try decodeForward(seq, rows: rows, escalate: false)
                if !unionGuard { break }
                LayerExpertCache.overflowCheck = false
                let maxU = LayerExpertCache.overflowMaxUnion
                if ofDbg { FileHandle.standardError.write("OFDBG C=\(C) D=\(dd) pk=\(pk) maxU=\(maxU) safeRows=\(LayerExpertCache.overflowSafeRows) safeMaxK=\(safeMaxK)\n".data(using: .utf8)!) }
                if maxU > C {                                    // overflow → exact safe-prefix へ縮小し re-verify(1 回で fit)
                    overflowCount += 1
                    for (i, c) in mc.enumerated() { c.restore(snaps[i], isLinear: isLin[i], trim: rows) }
                    // safeRows は pending 行と u row を含む row 数 → drafts へ割ける行 = safeRows - 1 - pk。
                    // safe prefix は pending+u prefix を決して侵食しない: 侵食が必要（newLen<1）なら draft は
                    // 一切 fit しない → singleFallback へ（advanceSingle が pending の fold/flush を処理）。
                    let newLen = Swift.min(dd - 1, LayerExpertCache.overflowSafeRows - 1 - pk)
                    safeMaxK = Swift.max(1, newLen)
                    if newLen < 1 { singleFallback = true; break }
                    curDrafts = Array(curDrafts.prefix(newLen))
                    continue
                } else if maxU > 0 && maxU * 10 <= C * 9 && safeMaxK < maxK {
                    safeMaxK = Swift.min(maxK, safeMaxK + Swift.max(2, safeMaxK / 6))  // union≤0.9C=余裕十分→成長
                }
                break                                            // union≤C=fit → accept へ
            }
            if singleFallback {                                  // 安全 prefix が 1 未満 → single-token（pending は fold）
                if acceptTrace { atSingleFB += 1 }
                try advanceSingle(u)
                if prof { tVerify += now() - ts }
                continue
            }
            let D2 = curDrafts.count
            // ★ margin-certified accept: batched verify logits は逐次 M=1 計算と order-stable でない
            //   (MLX kernel の累積順が batch shape 依存、f32 でも微小 drift → near-tie で token flip)。
            //   各 row の top1−top2 logit margin が certTau を超える token のみ batched 結果から commit、
            //   低 margin の境界 token は M=1 逐次 replay で exact に確定する（strict-lossless）。
            // ★A3: decision rows は offset pk から。row pk-1 以前（pending 行）の logits は commit 済み
            //   token の予測ゆえ無視。row pk（u 行）が d0 を予測する最初の decision row。
            let vRows = vlg[0, pk ..< (pk + D2 + 1)]              // [D2+1, V]
            let evalsArr = MLX.argMax(vRows, axis: -1)
            let vm1 = MLX.max(vRows, axis: -1, keepDims: true)
            let vm2 = MLX.max(MLX.where(vRows .>= vm1, MLXArray(-Float.infinity), vRows), axis: -1, keepDims: true)
            let marginsArr = (vm1 - vm2).reshaped([D2 + 1])
            MLX.eval([evalsArr, marginsArr])                      // evals と同一 sync point で margins も評価
            let evals = evalsArr.asArray(Int32.self).map { Int($0) }
            let margins = marginsArr.asArray(Float.self)
            var p = 0
            while p < D2 && curDrafts[p] == evals[p] && margins[p] > certTau { p += 1 }
            // commit する境界 token は常に row p 由来（draft 低 margin 停止・mismatch reject・bonus row
            //   の全ケース共通）→ margins[p] ≤ τ なら未認定 = sequential replay で確定。
            let certStop = margins[p] <= certTau
            if acceptTrace {
                atHist[p, default: 0] += 1
                if p < D2 {
                    if curDrafts[p] != evals[p] {                 // true draft mismatch (fork)
                        atMismatch += 1
                        let alt = p < Tell.lastDraftAlts.count ? Tell.lastDraftAlts[p] : -1
                        if alt >= 0 {
                            atAltExist += 1
                            if alt == evals[p] { atAltHit += 1 }  // 2nd choice would have caught it
                        }
                    } else {
                        atCertStop += 1                           // stopped by low margin only
                    }
                }
            }
            out.append(u); hist.append(u)
            for i in 0 ..< p { out.append(curDrafts[i]); hist.append(curDrafts[i]) }
            accTok += p
            certCommitted += p                                    // 受理 draft は margin-certified
            if certStop {
                // 未認定 → SEQUENTIAL REPLAY: snapshot 復元後 pending+[u]+accepted を 1 token ずつ M=1
                //   forward（greedy branch と同じ通常 single-token mode）。状態も次 token も逐次 exact =
                //   無条件 certified。replay は pending も丸ごと実体化する → pending=[]。
                certFallbacks += 1
                for (i, c) in mc.enumerated() { c.restore(snaps[i], isLinear: isLin[i], trim: pk + 1 + D2) }
                setVerifyMode(false)                              // replay は verify mode 外（greedy branch と同様）
                var rlg = vlg                                     // placeholder（replay ≥1 token で必ず上書き）
                for t in pending + [u] + Array(curDrafts.prefix(p)) {
                    (_, rlg) = try model.forwardHidden(MLXArray([Int32(t)], [1, 1]), caches: mc)
                    MLX.eval([rlg] + mc.flatMap { $0.stateArrays })
                    certReplayFwd += 1
                }
                uArr = MLX.argMax(rlg[0, 0], axis: -1).reshaped([1, 1])
                MLX.eval([uArr])
                replayCommitted += 1
                pending = []
            } else if p == D2 {
                certCommitted += 1                                // bonus token evals[D2] も certified
                uArr = MLXArray([Int32(evals[D2])], [1, 1])
                pending = []                                      // fused forward が pending ごと実体化済み
                setVerifyMode(false)
                MLX.eval([uArr] + mc.flatMap { $0.stateArrays })
            } else {
                certCommitted += 1                                // reject 境界 token evals[p] は certified
                for (i, c) in mc.enumerated() { c.restore(snaps[i], isLinear: isLin[i], trim: pk + 1 + D2) }
                // ★A3 fused reject: 旧 rebuild forward（logits 不使用の cache 再構築）を SKIP し、
                //   [u]+accepted を pending に積んで次 verify に融合（連続 reject では連結で成長）。
                pending += [u] + Array(curDrafts.prefix(p))
                a3Fused += 1
                if pending.count > pendingCap { try flushPending() }   // 非有界成長の cap（verify mode はまだ true）
                setVerifyMode(false)
                uArr = MLXArray([Int32(evals[p])], [1, 1])
                MLX.eval([uArr] + mc.flatMap { $0.stateArrays })
            }
            if prof { tVerify += now() - ts }
        }
        AttentionLayer.seqMultiToken = false; AttentionLayer.perQueryNone = false
        LayerExpertCache.overflowCheck = false
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
        if acceptTrace {
            let histStr = atHist.keys.sorted().map { "\($0):\(atHist[$0]!)" }.joined(separator: ",")
            print("[AcceptTrace] steps=\(steps) draftless=\(atDraftless) singleFB=\(atSingleFB) acceptHist=\(histStr) "
                  + "mismatch=\(atMismatch) alt2exist=\(atAltExist) alt2hit=\(atAltHit) certStopOnly=\(atCertStop)")
        }
        let ovTag = overflowCount > 0 ? "  [union-overflow guard: \(overflowCount) step re-verify/fallback]" : ""
        let certTag = String(format: "  cert-fallback=%d(%d fwd) commit(cert=%d/replay=%d)",
                             certFallbacks, certReplayFwd, certCommitted, replayCommitted)
        // A3 telemetry: fused-reject 数 = 節約した rebuild forward 数 / flush = edge-case で払った実体化
        let a3Tag = String(format: "  a3-fused=%d flush=%d", a3Fused, flushes)
        if Tell.envFlag("QWISP_DUMP_TOKENS") {   // bench correctness axis: prompt + method output
            print("PROMPT_TOKENS:" + ids.asArray(Int32.self).map { String($0) }.joined(separator: ","))
            print("OUT_TOKENS:" + outN.map { String($0) }.joined(separator: ","))
        }
        let tokps = Double(N) / secs
        let summary = String(format: """
            [SuffixSpec] suffix draft(maxK=%d) + clean exact verify(C=%d): %.1f tok/s  accept/step=%.2f  品質(vs ref=Swift正準greedy) %d/%d=%.0f%%%@%@%@%@
            """, maxK, C, tokps, Double(accTok) / Double(steps), match, N, Double(match) / Double(N) * 100, swiftTag, ovTag, certTag, a3Tag)
        return (summary, tokps)
    }

    /// α·p adaptive draft length（SuffixDecoding arXiv:2411.04975 の MAX_SPEC=α·p）:
    /// 弱い一致(m=4)は draft≤16、強い一致(m=32)は caller の容量 cap まで。
    static let suffixAlpha = 4

    /// margin-certified accept の閾値 τ: batched verify logits は逐次 M=1 と order-stable でなく
    /// (MLX kernel の累積順が batch shape 依存)、near-tie で commit token が flip し得る。
    /// 経験的に flip した near-tie の logit gap は ≲~0.06 → τ=0.1 は余裕込みでカバー。
    /// top1−top2 margin ≤ τ の境界 token は M=1 逐次 replay で確定（機械的 δ-calibration は将来 task）。
    static let certTau: Float = 0.1

    /// suffix lookup draft（SuffixDecoding-style, 訓練不要・cost ~0）:
    /// 1) seq 末尾の m token(minMatch..maxMatch の最長一致)が seq 内の earlier 位置に出現する
    ///    「全ての」出現位置を収集（旧: 最近 1 箇所のみ）。
    /// 2) 頻度重み付き greedy 継続: token を 1 個ずつ、alive な出現位置（ここまでの draft と継続が
    ///    一致している位置）が提案する次 token の多数決で伸長（同数 tie は最近位置の token=決定的）。
    ///    不一致の位置は脱落。alive が尽きるか長さ cap で停止。
    /// 3) 長さ cap = min(draftK, suffixAlpha·m)（draftK=caller の容量 cap: min(maxK, safeMaxK) 等）。
    /// コスト: alive-set loop は O(出現数 × draft長)。最長 m での出現数は通常少なく、hist が大きい
    /// 場合は既存の走査コストが支配的（既知・許容。longctx index は別 task）。
    ///
    ///
    /// reuseCtx 引数 (notes/10 §1c): nil で既存挙動と byte-identical。
    /// 非 nil かつ alpha=0 でも既存挙動と byte-identical（strict generalisation、G-A-1 で pin）。
    /// 非 nil かつ alpha>0 で weight(t) = counts[t] × (1 + alpha × reuseScore(t)) で rerank。
    // diag counters for reuse-rerank go/no-go (accumulated only when reuseCtx != nil)
    nonisolated(unsafe) static var reuseVotes = 0   // total vote iterations
    nonisolated(unsafe) static var reuseForks = 0   // votes with >1 distinct candidate
    nonisolated(unsafe) static var reuseFlips = 0   // votes where rerank picked ≠ count-majority

    // QWISP_ACCEPT_TRACE diag: per-position runner-up token of the last draft (-1 = no 2nd
    // candidate at that vote). Filled only when suffixDraft(traceAlts: true); measures the
    // k=2-parallel-draft prize ("would the 2nd choice have caught the mismatch?").
    nonisolated(unsafe) static var lastDraftAlts: [Int] = []

    static func suffixDraft(_ seq: [Int], maxMatch: Int, draftK: Int, minMatch: Int,
                            reuseCtx: (ctx: ReuseContext, residentPerLayer: [Set<Int>], alpha: Double)? = nil,
                            traceAlts: Bool = false) -> [Int] {
        let n = seq.count
        if traceAlts { lastDraftAlts = [] }
        if n < minMatch + 1 { return [] }
        var m = Swift.min(maxMatch, n - 1)
        while m >= minMatch {
            let patStart = n - m
            var occ: [Int] = []          // 一致開始位置（最近→過去の順に収集）
            var i = patStart - 1
            while i >= 0 {
                var ok = true
                for j in 0 ..< m where seq[i + j] != seq[patStart + j] { ok = false; break }
                if ok { occ.append(i) }
                i -= 1
            }
            if !occ.isEmpty {
                let cap = Swift.min(draftK, suffixAlpha * m)   // α·p length cap
                var draft: [Int] = []
                var alive = occ                                // draft と継続一致中の位置（最近順）
                while draft.count < cap && !alive.isEmpty {
                    var counts: [Int: Int] = [:]
                    var next: [Int] = []                       // alive[k] の提案 token（-1=尽きた）
                    for pos in alive {
                        let idx = pos + m + draft.count
                        if idx < n { let t = seq[idx]; next.append(t); counts[t, default: 0] += 1 }
                        else { next.append(-1) }
                    }
                    // Weight-based voting. Iterate alive in most-recent-first order.
                    // Strict > comparison: first token to reach the max weight wins (most-recent tie-break).
                    // When alpha=0 or reuseCtx==nil, weight = Double(counts[t]) → identical to old path.
                    var best = -1
                    var bestWeight = -1.0   // counts >= 1, so any valid token beats this
                    var countBest = -1, countBestCnt = 0   // diag: what pure count-majority would pick
                    var second = -1, secondWeight = -1.0   // traceAlts diag: runner-up token
                    for k in 0 ..< alive.count {
                        let t = next[k]
                        guard t >= 0, let c = counts[t] else { continue }
                        let w: Double
                        if let rc = reuseCtx {
                            w = Double(c) * (1.0 + rc.alpha * rc.ctx.reuseScore(token: t, residentPerLayer: rc.residentPerLayer))
                        } else {
                            w = Double(c)
                        }
                        if w > bestWeight {
                            if best >= 0 && best != t { second = best; secondWeight = bestWeight }
                            best = t; bestWeight = w
                        } else if t != best && w > secondWeight {
                            second = t; secondWeight = w
                        }
                        if c > countBestCnt { countBest = t; countBestCnt = c }
                    }
                    // diag counters (reuseCtx runs only): fork = >1 distinct candidate, flip = rerank changed pick
                    if reuseCtx != nil, best >= 0 {
                        reuseVotes += 1
                        if counts.count > 1 { reuseForks += 1 }
                        if best != countBest { reuseFlips += 1 }
                    }
                    if best < 0 { break }                      // 全 alive が末尾到達
                    if traceAlts { lastDraftAlts.append(second) }
                    draft.append(best)
                    var kept: [Int] = []
                    for k in 0 ..< alive.count where next[k] == best { kept.append(alive[k]) }
                    alive = kept
                }
                return draft
            }
            m -= 1
        }
        return []
    }
}

// ── ReuseContext: expert-reuse draft rerank context (notes/10 §2) ─────────────
// Accumulates per-token per-layer expert usage from streaming verify rows and
// provides reuseScore for suffixDraft candidate reranking.
// observe: row m of rowTokens maps to inds[m*Ktop ..< (m+1)*Ktop] at the given layer.
// reuseScore: returns Σ_li |tokenExperts[t][li] ∩ residentPerLayer[li]|
// Flag-off (QWISP_REUSE_RERANK unset) and alpha=0 are byte-identical to nil (no rerank).
public struct ReuseContext {
    // token -> layer -> Set of expert indices (accumulated across observe calls)
    private var tokenExperts: [Int: [Int: Set<Int>]] = [:]

    public init() {}

    /// Accumulate per-row expert routing. Row m of rowTokens routes to
    /// inds[m*Ktop ..< (m+1)*Ktop] at the given layer.
    public mutating func observe(rowTokens: [Int], layer: Int, inds: [Int32], Ktop: Int) {
        for (m, token) in rowTokens.enumerated() {
            let start = m * Ktop
            guard start + Ktop <= inds.count else { continue }
            var expertSet = tokenExperts[token]?[layer] ?? Set<Int>()
            for k in 0 ..< Ktop {
                expertSet.insert(Int(inds[start + k]))
            }
            if tokenExperts[token] == nil { tokenExperts[token] = [:] }
            tokenExperts[token]![layer] = expertSet
        }
    }

    /// Resident-overlap score: Σ_li |tokenExperts[t][li] ∩ residentPerLayer[li]|
    /// Unknown tokens return 0.0 (neutral — no bias toward or away from resident experts).
    public func reuseScore(token: Int, residentPerLayer: [Set<Int>]) -> Double {
        guard let layerMap = tokenExperts[token] else { return 0.0 }
        var score = 0.0
        for (li, residentSet) in residentPerLayer.enumerated() {
            if let observed = layerMap[li] {
                score += Double(observed.intersection(residentSet).count)
            }
        }
        return score
    }
}
