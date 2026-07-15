import Foundation

/// Free-run repetition guard for the bolt tier (#47 Part A).
///
/// C=64 frozen residency below the workload's per-layer footprint drives long bolt
/// free-runs into a deterministic repetition attractor that neither rolling recalib nor
/// forward-only strict escalation can escape (both measured — the established loop poisons
/// the KV context). The recovery that works is REWIND: drop the degenerate tail back to
/// its onset (a clean, ratio-1.0 prefix) and re-decode from there on the strict path.
///
/// This class is the mechanism: a rollback buffer + an early loop detector, both pure and
/// self-checked. It owns the generated-token list and mediates what has been streamed:
///   - push(t): append a generated token; return the (possibly empty) slice to stream now,
///     holding at most `rollbackDepth` tokens unsent so a detected loop can still be rewound
///     before its onset was streamed. The first `bypassPrefix` tokens stream immediately
///     (loops never start that early — measured min onset 87 tok — so short answers pay no
///     latency; a long answer pays one mid-stream buffer-fill pause).
///   - trip: set the step a sustained loop was confirmed; onset = first looped token.
///   - rewind(): truncate the generated list to onset (only tokens still unsent are dropped;
///     any already-streamed loop tokens can't be un-sent) and return the surviving prefix.
///
/// Detector (measured on real bolt loops, periods 1/8/11/16, 0 false positives on 2300
/// clean strict tokens): the tail is exactly period-p for some p ≤ maxPeriod over a span of
/// at least max(minReps·p, minSpan) tokens. minReps=3 + minSpan=24 rejects legitimate short
/// repeats ("aa", "10 10", "ha ha ha") while catching sustained degeneration ~2·period after
/// onset. rollbackDepth 64 keeps the onset comfortably inside the buffer at trip time (max
/// detect span 48) and covers periods up to 21.
public final class LoopGuard {
    public static let defaultRollbackDepth = Tell.envInt("QWISP_LOOP_ROLLBACK", 64)
    public static let defaultBypassPrefix = Tell.envInt("QWISP_LOOP_BYPASS", 64)

    // Detector tuning (see class doc; validated in the #47 sweep).
    static let maxPeriod = 64
    static let minReps = 3
    static let minSpan = 24

    let rollbackDepth: Int          // max tokens held unsent (Int.max = non-streaming, hold all)
    let bypassPrefix: Int           // first N tokens stream immediately (short-answer fast path)
    private(set) var gen: [Int] = []       // all generated tokens (authoritative)
    private(set) var sent = 0              // count of `gen` already streamed to the client
    public private(set) var trip: (onset: Int, period: Int, span: Int)? = nil

    public init(rollbackDepth: Int = LoopGuard.defaultRollbackDepth,
                bypassPrefix: Int = LoopGuard.defaultBypassPrefix) {
        self.rollbackDepth = rollbackDepth
        self.bypassPrefix = bypassPrefix
    }

    /// Append a generated token; return the tokens to stream now (in order). Holds up to
    /// `rollbackDepth` tokens unsent (0 while within `bypassPrefix`). Runs the detector; on
    /// a first confirmed loop, sets `trip` (the caller should then stop and rewind).
    @discardableResult
    public func push(_ t: Int) -> ArraySlice<Int> {
        gen.append(t)
        if trip == nil, let d = Self.detect(gen) { trip = d }
        // Target sent count: everything while within the bypass prefix, else hold `rollbackDepth`.
        let target = gen.count <= bypassPrefix ? gen.count
                   : Swift.max(sent, gen.count - rollbackDepth)
        guard target > sent else { return gen[sent ..< sent] }
        let out = gen[sent ..< target]
        sent = target
        return out
    }

    /// Rewind the generated list to the loop onset (or to the last streamed token if the
    /// onset was already streamed — those can't be recalled). Returns the surviving prefix
    /// (all generated tokens after rewind); the caller re-prefills from it on strict.
    /// Clears `trip` so the guard can re-arm for another pass (return-to-bolt policy).
    @discardableResult
    public func rewind() -> [Int] {
        guard let d = trip else { return gen }
        let cut = Swift.max(sent, d.onset)      // never drop already-streamed tokens
        gen.removeLast(gen.count - cut)
        trip = nil
        return gen
    }

    /// Flush anything still buffered (call at end of generation). Returns the tail slice.
    public func flush() -> ArraySlice<Int> {
        let out = gen[sent ..< gen.count]
        sent = gen.count
        return out
    }

    /// Tokens streamed so far (for the caller's produced-count accounting).
    public var sentCount: Int { sent }

    // ── Detector (pure) ────────────────────────────────────────────────────────
    /// Return (onset, period, span) if the tail of `toks` is a sustained period-p loop,
    /// else nil. onset = first token of the looped run.
    static func detect(_ toks: [Int]) -> (onset: Int, period: Int, span: Int)? {
        let n = toks.count
        let pMax = Swift.min(maxPeriod, n / 2)
        guard pMax >= 1 else { return nil }          // too short to hold any 2-period repeat
        for p in 1 ... pMax {
            let need = Swift.max(minReps * p, minSpan)
            if n < need { continue }
            // last block is toks[n-p ..< n]; count consecutive equal blocks backwards.
            var reps = 1, j = n - p
            while j - p >= 0 && Array(toks[(j - p) ..< j]) == Array(toks[(n - p) ..< n]) {
                reps += 1; j -= p
            }
            let span = reps * p
            if reps >= minReps && span >= minSpan {
                return (onset: n - span, period: p, span: span)
            }
        }
        return nil
    }

    /// Pure self-check (no GPU): detector fires on sustained loops, ignores short repeats
    /// and clean streams; buffer holds ≤ depth and rewind drops only unsent tokens.
    public static func selfCheck() -> Bool {
        // period-1 sustained (30×) → onset at start of the run
        let loop1 = Array(0 ..< 10) + Array(repeating: 7, count: 30)
        guard let d1 = detect(loop1), d1.period == 1, d1.onset == 10 else { return false }
        // period-8 sustained
        let unit = Array(100 ..< 108)
        let loop8 = Array(0 ..< 20) + Array(repeating: unit, count: 5).flatMap { $0 }
        guard let d8 = detect(loop8), d8.period == 8, d8.onset == 20 else { return false }
        // legit short repeats must NOT fire
        if detect([1, 2, 3, 5, 5, 9, 10, 10, 11]) != nil { return false }        // isolated doubles
        if detect(Array(0 ..< 200)) != nil { return false }                       // clean stream
        if detect([1, 2, 1, 2, 1, 2]) != nil { return false }                     // period-2 ×3 = span 6 < minSpan
        // buffer: hold ≤ depth, bypass streams the head, rewind drops only unsent
        let g = LoopGuard(rollbackDepth: 4, bypassPrefix: 2)
        var streamed: [Int] = []
        for t in 0 ..< 10 { streamed += Array(g.push(t)) }
        guard g.gen.count - g.sentCount <= 4 else { return false }                // ≤ depth unsent
        guard streamed == Array(0 ..< 6) else { return false }                    // 10 gen − 4 held
        return true
    }
}
