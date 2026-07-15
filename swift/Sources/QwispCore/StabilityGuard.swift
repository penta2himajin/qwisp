import Foundation

/// Free-run degeneration detector (#47 Part A). Greedy repetition loops collapse the
/// distinct n-gram ratio of the generated token stream toward loopLen/window; healthy
/// text stays near 1.0. Pure function — shared by benchtest (reporting) and BoltServe
/// (the opt-in stability guard that escalates a degenerating bolt segment to strict).
///
/// Measured basis for the intervention (2026-07-15, C=64 sweep): once a loop is
/// established the rolling-recalib window is self-poisoned by the loop's own routing
/// (~11 refreshes fired inside a loop without escaping), so the only reliable recovery
/// is leaving bolt for the strict path — not another recalibration.
public enum StabilityGuard {
    /// Distinct n-gram ratio over the tail `window` of `ids`. 1.0 when too short to judge.
    public static func ratio(_ ids: [Int], n: Int = 8, window: Int = 256) -> Double {
        let tail = Array(ids.suffix(window))
        guard tail.count >= n * 2 else { return 1.0 }
        var seen = Set<[Int]>()
        var total = 0
        for i in 0 ... (tail.count - n) {
            seen.insert(Array(tail[i ..< i + n])); total += 1
        }
        return Double(seen.count) / Double(total)
    }

    /// Trip threshold: benchtest's LOOPY line (< 0.5) — measured hard loops sit at
    /// 0.00-0.28, healthy code prompts at 0.80+, so the band is wide on both sides.
    public static let tripThreshold = 0.5
    /// Cadence: re-check every 64 generated tokens (a full ratio over a 256 window is
    /// microseconds of CPU; cadence only bounds detection lag).
    public static let checkEvery = 64

    /// Self-check (pure): unique stream ≈ 1.0, tight loop ≪ 0.5, short stream benign.
    public static func selfCheck() -> Bool {
        let unique = ratio(Array(0 ..< 300))
        let loop = ratio((0 ..< 20).flatMap { _ in Array(0 ..< 16) })
        let short = ratio([1, 2, 3])
        return unique > 0.9 && loop < 0.5 && short == 1.0
    }
}
