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
        defer { GatedDeltaNetLayer.f32Conv = false; AttentionLayer.f32SDPA = false; AttentionLayer.perQueryNone = false; StreamingMoEBlock.probeNoSync = false }
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
        // ★ lever-1(issue#3): 全 expert resident(C>=nE)なら no-sync 経路(GPU slot-table remap)が
        //   exact 経路と bit 一致(alias 不可能)＝per-layer routing round-trip 除去で ~1.7x lossless。
        //   QWISP_NOSYNC=1 強制 / =0 無効 / 既定 auto(C>=256 で自動)。C<nE は cold routed が slot-0
        //   alias で garbage ゆえ無効(escalation 路線は別 runner)。
        let nosyncEnv = Tell.envStr("QWISP_NOSYNC", "auto")
        let residentNoSync = nosyncEnv == "1" || (nosyncEnv != "0" && C >= nE)
        if residentNoSync {
            print("[SuffixSpec] no-sync exact 経路 ON (C=\(C)>=\(nE) 全 expert resident, round-trip 除去 ~1.7x lossless)")
        }
        var hist = ids.asArray(Int32.self).map { Int($0) }     // 履歴（prompt + commit token）
        let mc = model.makeCaches()
        StreamingMoEBlock.probeNoSync = residentNoSync
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
