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

    /// α·p adaptive draft length（SuffixDecoding arXiv:2411.04975 の MAX_SPEC=α·p）:
    /// 弱い一致(m=4)は draft≤16、強い一致(m=32)は caller の容量 cap まで。
    static let suffixAlpha = 4

    /// margin-certified accept の閾値 τ: batched verify logits は逐次 M=1 と order-stable でなく
    /// (MLX kernel の累積順が batch shape 依存)、near-tie で commit token が flip し得る。
    /// 経験的に flip した near-tie の logit gap は ≲~0.06 → τ=0.1 は余裕込みでカバー。
    /// top1−top2 margin ≤ τ の境界 token は M=1 逐次 replay で確定（機械的 δ-calibration は将来 task）。
    static let certTau: Float = 0.1

    /// SuffixSpec minimum match length (QWISP_SUFFIX_MINMATCH, default 4 = historical
    /// constant). Lower → more drafts on real traffic (lower d0) at the cost of wasted
    /// verify rows on rejects; lossless either way (verify gates every draft). Read once
    /// per process (env-in-loop is 18.8µs, fusion doctrine).
    static let suffixMinMatch = Swift.max(1, Tell.envInt("QWISP_SUFFIX_MINMATCH", 4))

    // ── #119: mechanical speculation gate ────────────────────────────────────────
    // A draft attempt runs the f32 strict verify, whose cost is ~linear in context
    // (measured 288ms/attempt @14K → 878ms @48K), while a chain step costs 17→29ms.
    // Break-even is ~17 accepted tokens/attempt @14K, ~30 @48K; low-accept agentic
    // regimes deliver 4-5 — every attempt is a net loss and decode collapses (issue
    // #119: 6-7 tok/s at 48K; gate off/on measured 18.8 → 38 tok/s). The gate is
    // mechanical (rolling measured accept, no prediction): suspend drafting when the
    // rolling mean accepted-per-attempt is below a ctx-scaled threshold, re-probe
    // after specGateReprobe emitted tokens. The window is NOT cleared on suspension,
    // so a still-bad regime re-suspends after a single probe attempt. Greedy chain is
    // the reference path — gating changes speed only, the token stream is unchanged.
    // QWISP_SPEC_GATE=0 opts out (always draft, pre-#119 behavior).
    static let specGateWindow = 8
    static var specGateEnabled: Bool { envInt("QWISP_SPEC_GATE", 1) != 0 }
    static var specGateReprobe: Int { Swift.max(8, envInt("QWISP_SPEC_REPROBE", 256)) }
    /// Conservative floor of the measured break-even curve (errs toward keeping spec on).
    static func specGateThreshold(histLen: Int) -> Int { 4 + histLen / 3000 }
    /// Evidence needed before suspending shrinks as context grows: an attempt's waste is
    /// ctx-linear (~900ms @48K) while a false suspension costs only one re-probe span, so
    /// at long ctx a single sub-threshold attempt suffices; short ctx demands a full window.
    static func specGateWindowNeeded(histLen: Int) -> Int { Swift.max(1, specGateWindow - histLen / 6000) }
    static func specGateShouldSuspend(window: [Int], histLen: Int) -> Bool {
        window.count >= specGateWindowNeeded(histLen: histLen)
            && window.reduce(0, +) < specGateThreshold(histLen: histLen) * window.count
    }

    // ── SCORE predictive gate (ported from qwisp-dense / SuffixDecoding §3) ───────
    // Reactive #119 alone loses on short agentic / longctx: the attempt window fills late
    // (or never), while wide drafts still pay linear verify. SCORE is the tree's expected
    // accept length, available at draft time with zero extra I/O.
    //   SCORE > r·D  → verify; else discard draft and chain (speed only; lossless either way).
    // QWISP_SPEC_GATE_SCORE=0 falls back to #119-only. Ornith 4-regime A/B (2026-08-23):
    // r=478‰ (dense) wins code/longctx/shortnl while SCORE still zeros verify on agentic;
    // r=550/650 over-suppresses Spec wins on longctx without lifting the agentic chain floor.
    static let specGateUseScore = envInt("QWISP_SPEC_GATE_SCORE", 1) != 0
    /// 1 draft row cost / AR step cost, permille. Default 478 (dense / Ornith A/B).
    static let specGateRowCostPermille = Swift.max(1, envInt("QWISP_SPEC_GATE_ROWCOST", 478))
    /// Laplace add-k on sibling counts — without it alive=1 → C=1 and SCORE saturates (dense §10).
    static let specGateSmoothK = Swift.max(0, envInt("QWISP_SPEC_SMOOTH_K", 1))
    /// SpecDec++ marginal truncation (separate r from the gate — do not unify).
    static let specTruncEnabled = envInt("QWISP_SPEC_TRUNC", 1) != 0
    static let specTruncRowCostPermille = Swift.max(1, envInt("QWISP_SPEC_TRUNC_ROWCOST", 790))
    /// Stop drafting when the winning vote has fewer than this many agreeing occurrences
    /// (tinycodr minSupport). Default 1 = historical; 2 cuts template-hole waste.
    static let suffixMinSupport = Swift.max(1, envInt("QWISP_SUFFIX_MINSUPPORT", 1))
    /// Accept-gated adaptive maxK (qwisp-sk). ON by default: clamp draftK to recent accept+grace.
    static let adaptiveKEnabled = envInt("QWISP_ADAPTIVE_K", 1) != 0
    static let adaptiveKWindow = Swift.max(1, envInt("QWISP_ADAPT_WINDOW", 8))
    static let adaptiveKGrace = Swift.max(0, envInt("QWISP_ADAPT_GRACE", 2))
    /// Expected accept tokens of the most recent suffixDraft (SCORE).
    nonisolated(unsafe) static var lastDraftScore: Double = 0

    static func specGateShouldDraft(score: Double, drafted: Int) -> Bool {
        guard drafted > 0 else { return true }
        return score * 1000.0 > Double(drafted * specGateRowCostPermille)
    }
    static func specGateShouldExtend(dRunning: Double) -> Bool {
        dRunning * 1000.0 > Double(specTruncRowCostPermille)
    }
    /// Soft maxK from recent accepted lengths (sk ADAPTIVE_K).
    static func adaptiveDraftK(cap: Int, recentAccepts: [Int]) -> Int {
        guard adaptiveKEnabled, !recentAccepts.isEmpty, cap > 1 else { return cap }
        let mean = recentAccepts.reduce(0, +) / recentAccepts.count
        return Swift.max(1, Swift.min(cap, mean + adaptiveKGrace))
    }

    /// Pure self-check (no GPU) for the gate arithmetic.
    public static func specGateSelfCheck() -> [(String, Bool)] {
        let bad = Array(repeating: 4, count: specGateWindow)      // 4 acc/attempt
        let good = Array(repeating: 20, count: specGateWindow)
        return [
            ("threshold_short", specGateThreshold(histLen: 2000) == 4),
            ("threshold_14k", specGateThreshold(histLen: 14000) == 8),
            ("threshold_48k", specGateThreshold(histLen: 48000) == 20),
            ("window_short_full", specGateWindowNeeded(histLen: 0) == 8),
            ("window_14k", specGateWindowNeeded(histLen: 14000) == 6),
            ("window_48k_single", specGateWindowNeeded(histLen: 48000) == 1),
            ("suspend_one_bad_long", specGateShouldSuspend(window: [4], histLen: 48000)),
            ("suspend_bad_long", specGateShouldSuspend(window: bad, histLen: 48000)),
            ("keep_good_long", !specGateShouldSuspend(window: good, histLen: 48000)),
            ("keep_one_good_long", !specGateShouldSuspend(window: [24], histLen: 48000)),
            ("keep_bad_short", !specGateShouldSuspend(window: bad, histLen: 0)),   // 4 ≥ threshold 4
            ("no_suspend_before_window_short", !specGateShouldSuspend(window: [0], histLen: 0)),
            ("rolling_recovers", !specGateShouldSuspend(window: Array(bad.dropFirst(4)) + [40, 40, 40, 40], histLen: 48000)),
            // SCORE gate (r=0.478)
            ("score_d4_above", specGateShouldDraft(score: 2.0, drafted: 4)),       // 2.0 > 1.912
            ("score_d4_below", !specGateShouldDraft(score: 1.8, drafted: 4)),
            ("score_reject_wide_low", !specGateShouldDraft(score: 11.0, drafted: 52)),
            ("score_reject_agentic", !specGateShouldDraft(score: 1.0, drafted: 28)),
            ("score_perfect_passes", specGateShouldDraft(score: 32.0, drafted: 32)),
            ("trunc_above", specGateShouldExtend(dRunning: 0.80)),
            ("trunc_below", !specGateShouldExtend(dRunning: 0.78)),
            ("adapt_k_clamps", adaptiveDraftK(cap: 96, recentAccepts: [2, 3, 2]) == 2 + adaptiveKGrace
                || !adaptiveKEnabled),
        ]
    }

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
        lastDraftScore = 0
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
                var dRunning = 1.0                             // D(N): product of C along the path
                var score = 0.0                                // SCORE = Σ D(N) = E[accepted]
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
                    if countBestCnt < suffixMinSupport { break }  // tinycodr confidence stop
                    // SuffixDecoding D(N)/SCORE accumulated while extending (paper §3).
                    // C(N) = COUNT(best) / (Σ siblings + Laplace k); D = D_parent · C; SCORE += D.
                    let sibling = counts.values.reduce(0, +)
                    if sibling > 0, let c = counts[best] {
                        // Trunc uses unsmoothed C: Laplace k=1 makes alive=1 → C=0.5, which is
                        // always < trunc-r (790‰) and empties every singleton draft (Ornith code
                        // A/B: SMOOTH_K=0 restored verify; k=1+trunc → verify=0). SCORE still
                        // uses Laplace so alive=1 does not saturate the gate (dense §10).
                        let dRaw = dRunning * Double(c) / Double(sibling)
                        if specTruncEnabled && !specGateShouldExtend(dRunning: dRaw) { break }
                        let d = dRunning * Double(c) / Double(sibling + specGateSmoothK)
                        dRunning = d
                        score += dRunning
                    }
                    if traceAlts { lastDraftAlts.append(second) }
                    draft.append(best)
                    var kept: [Int] = []
                    for k in 0 ..< alive.count where next[k] == best { kept.append(alive[k]) }
                    alive = kept
                }
                lastDraftScore = score
                return draft
            }
            m -= 1
        }
        return []
    }

    // Row-map pure function for MTP-D1 hybrid draft head-sync (notes/15 G-A).
    // rows = [pending pk][u][drafts] in H2 from verify.
    //   fullAccept/reject: feedRows = 0..<(pk+p), lastHRow = pk+p (flush committed prefix).
    //   replay:  feedRows = 0..<0,    lastHRow = -1   (caller feeds sequentially; pk/p ignored).
    //   single:  feedRows = 0..<pk,   lastHRow = pk   (feed pending hiddens; lastH = u hidden).
    public static func mtpFeedPlan(pk: Int, p: Int, path: FeedPath)
        -> (feedRows: Range<Int>, lastHRow: Int)? {
        switch path {
        case .fullAccept, .reject:
            return (feedRows: 0..<(pk + p), lastHRow: pk + p)
        case .replay:
            return (feedRows: 0..<0, lastHRow: -1)
        case .single:
            return (feedRows: 0..<pk, lastHRow: pk)
        }
    }
}

/// FeedPath discriminates the four head-sync wiring paths in MTP-D1 hybrid draft (notes/15).
/// - fullAccept: verify accepted p drafts (p may be 0 for a clean reject-all step)
/// - reject:     verify rejected all drafts (same contract as fullAccept; kept distinct for caller clarity)
/// - replay:     certStop replay — head is fed sequentially by the caller; feedRows always empty
/// - single:     advanceSingle — pending non-empty, no suffix draft; feeds pending hiddens only
public enum FeedPath { case fullAccept, reject, replay, single }

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
