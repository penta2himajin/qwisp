import Foundation

// Speculative-sampling support (Option B prototype).
//
// The greedy path verifies drafts by argmax-equality (lossless at T=0). To sample at T>0
// while keeping the target distribution EXACT, we use speculative sampling (Leviathan/Chen
// 2023) with a *deterministic* suffix draft (draft dist q = δ_d, so q(d)=1):
//   • accept the drafted token d with probability p(d)               [min(1, p(d)/q(d)) = p(d)]
//   • on reject, resample from the residual  norm(max(0, p − q))  =  p with d zeroed, renormalized
//   • if the whole draft is accepted, sample a bonus token from p at the next position
// This reduces EXACTLY to greedy at T=0 (p is one-hot at argmax → accept iff d==argmax), so the
// T=0 stream is byte-identical to the current engine regardless of seed — the small correctness gate.
//
// `p` here is the tempered + top_p-truncated distribution: "sample with temperature/top_p" means
// the target distribution IS that shaped one, and speculative sampling reproduces it exactly.

/// Deterministic, seedable PRNG (SplitMix64) — reproducible sampling for a given `seed`.
public struct SplitMix64: RandomNumberGenerator {
    private var state: UInt64
    public init(seed: UInt64) { self.state = seed }
    public mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
    /// Uniform Double in [0, 1).
    public mutating func unit() -> Double { Double(next() >> 11) * (1.0 / 9_007_199_254_740_992.0) }
}

public enum Sampler {
    /// Turn raw logits into the tempered + top_p-truncated probability vector.
    /// temperature == 0 → one-hot at argmax (greedy limit); tokens outside the top_p nucleus → 0.
    public static func probs(logits: [Float], temperature: Double, topP: Double) -> [Float] {
        let n = logits.count
        var p = [Float](repeating: 0, count: n)
        if temperature <= 0 {                       // greedy limit: one-hot at argmax
            var best = 0; var bv = logits[0]
            for i in 1..<n where logits[i] > bv { bv = logits[i]; best = i }
            p[best] = 1
            return p
        }
        // softmax(logits / T) in a numerically stable way.
        let invT = Float(1.0 / temperature)
        var mx = logits[0]
        for v in logits where v > mx { mx = v }
        var sum: Float = 0
        for i in 0..<n { let e = expf((logits[i] - mx) * invT); p[i] = e; sum += e }
        let inv = 1 / sum
        for i in 0..<n { p[i] *= inv }
        if topP < 1.0 { nucleusTruncate(&p, topP: topP) }
        return p
    }

    /// Zero everything outside the smallest set whose cumulative prob ≥ topP, then renormalize.
    static func nucleusTruncate(_ p: inout [Float], topP: Double) {
        let order = (0..<p.count).sorted { p[$0] > p[$1] }
        var cum: Double = 0
        var keep = Set<Int>()
        for i in order {
            keep.insert(i)
            cum += Double(p[i])
            if cum >= topP { break }               // include the token that crosses the threshold
        }
        var sum: Float = 0
        for i in 0..<p.count { if !keep.contains(i) { p[i] = 0 } else { sum += p[i] } }
        if sum > 0 { let inv = 1 / sum; for i in 0..<p.count { p[i] *= inv } }
    }

    /// Categorical draw from a (already normalized) probability vector.
    public static func categorical(_ p: [Float], rng: inout SplitMix64) -> Int {
        let r = Float(rng.unit())
        var acc: Float = 0
        for i in 0..<p.count { acc += p[i]; if r < acc { return i } }
        // Fallback for FP drift: last nonzero.
        for i in stride(from: p.count - 1, through: 0, by: -1) where p[i] > 0 { return i }
        return 0
    }

    /// Speculative-sampling accept for one draft token `d` against target `p`.
    /// Returns (accepted, resampled): if accepted, `resampled` is nil; else it is the residual draw.
    public static func acceptOrResample(p: [Float], draft d: Int, rng: inout SplitMix64) -> (Bool, Int?) {
        let coin = Float(rng.unit())
        if coin < p[d] { return (true, nil) }       // accept with prob p(d)
        var resid = p
        resid[d] = 0                                 // residual = max(0, p − δ_d), renormalized
        var sum: Float = 0
        for v in resid { sum += v }
        if sum > 0 { let inv = 1 / sum; for i in 0..<resid.count { resid[i] *= inv } }
        return (false, categorical(resid, rng: &rng))
    }

    /// Apply OpenAI penalties + logit_bias to raw logits before softmax.
    /// `counts` maps token id → occurrences so far (only present tokens). frequency_penalty scales
    /// with count; presence_penalty is applied once per present token; logit_bias is additive.
    public static func adjustLogits(_ logits: inout [Float], counts: [Int: Int],
                                    frequencyPenalty: Double = 0, presencePenalty: Double = 0,
                                    logitBias: [Int: Double] = [:]) {
        if frequencyPenalty != 0 || presencePenalty != 0 {
            let fp = Float(frequencyPenalty), pp = Float(presencePenalty)
            for (tok, c) in counts where tok >= 0 && tok < logits.count {
                logits[tok] -= fp * Float(c) + pp
            }
        }
        for (tok, b) in logitBias where tok >= 0 && tok < logits.count {
            logits[tok] += Float(b)
        }
    }

    /// GPU-free self-check of the sampling math (run via `qwisp sampletest`).
    public static func selfCheck() -> (passed: Int, total: Int, log: [String]) {
        var passed = 0, total = 0; var log: [String] = []
        func ok(_ name: String, _ cond: Bool) {
            total += 1; if cond { passed += 1 }; log.append("[sample-test] \(name): \(cond ? "PASS" : "FAIL")")
        }
        // 1. T=0 → one-hot at argmax.
        let lg: [Float] = [1.0, 3.0, 2.0, 0.5]
        let p0 = probs(logits: lg, temperature: 0, topP: 1)
        ok("temp0_onehot_argmax", p0 == [0, 1, 0, 0])

        // 2. T=0 accept: draft==argmax always accepts; draft≠argmax always rejects → resample=argmax.
        var r = SplitMix64(seed: 42)
        let (a1, _) = acceptOrResample(p: p0, draft: 1, rng: &r)
        let (a2, res2) = acceptOrResample(p: p0, draft: 0, rng: &r)
        ok("temp0_accept_argmax", a1 == true)
        ok("temp0_reject_nonargmax_resample_argmax", a2 == false && res2 == 1)

        // 3. softmax normalizes; higher logit → higher prob.
        let p1 = probs(logits: lg, temperature: 1.0, topP: 1)
        let s = p1.reduce(0, +)
        ok("softmax_normalized", abs(s - 1) < 1e-4 && p1[1] > p1[2] && p1[2] > p1[0])

        // 4. top_p truncation keeps only the nucleus (here the top-1 crosses 0.5 for this dist? check top-2).
        let pT = probs(logits: [10, 9, 0, -5], temperature: 1.0, topP: 0.5)
        ok("topp_truncates_tail", pT[2] == 0 && pT[3] == 0 && abs(pT.reduce(0,+) - 1) < 1e-4)

        // 5. determinism: same seed → same categorical draws.
        var ra = SplitMix64(seed: 7), rb = SplitMix64(seed: 7)
        let da = (0..<20).map { _ in categorical(p1, rng: &ra) }
        let db = (0..<20).map { _ in categorical(p1, rng: &rb) }
        ok("seed_reproducible", da == db)

        // 6. mean of many draws approximates p (statistical sanity).
        var rc = SplitMix64(seed: 123)
        var counts = [Int](repeating: 0, count: p1.count)
        let N = 20000
        for _ in 0..<N { counts[categorical(p1, rng: &rc)] += 1 }
        let empiricalMode = counts.firstIndex(of: counts.max()!)!
        let trueMode = p1.firstIndex(of: p1.max()!)!
        ok("empirical_mode_matches", empiricalMode == trueMode)

        // 7. logit_bias: -inf-ish bias bans a token; large positive forces it.
        var lb = [Float]([1, 2, 3, 0])
        adjustLogits(&lb, counts: [:], frequencyPenalty: 0, presencePenalty: 0, logitBias: [2: -100, 0: 100])
        ok("logit_bias_applied", lb[2] == -97 && lb[0] == 101)
        let pBias = probs(logits: lb, temperature: 1.0, topP: 1)
        ok("logit_bias_shifts_argmax", pBias.firstIndex(of: pBias.max()!) == 0)   // token 0 now dominates

        // 8. frequency/presence penalty lowers a repeated token's logit by count*fp + pp.
        var lp = [Float]([5, 5, 5, 5])
        adjustLogits(&lp, counts: [1: 3], frequencyPenalty: 0.5, presencePenalty: 1.0)
        ok("freq_presence_penalty", abs(lp[1] - (5 - 0.5*3 - 1.0)) < 1e-5 && lp[0] == 5)

        return (passed, total, log)
    }
}
