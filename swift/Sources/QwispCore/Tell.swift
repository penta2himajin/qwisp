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
