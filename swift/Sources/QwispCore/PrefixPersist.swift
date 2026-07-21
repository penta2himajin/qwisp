import Foundation

// Disk persistence for the cross-request prefix cache (issue #89, follow-up to #76).
//
// The in-process cache dies with the process: `qwisp chat` one-shots, service restarts on
// update, and reboots all re-pay full prefill (streaming prefill is the expensive kind —
// per-chunk expert IO). This store writes the content-boundary decode state (attention KV
// used slice + GDN state, via SeedlessFusedForward.persistentStateData) keyed by
// model + exact token prefix, and restores the longest stored prefix of a new request's
// content in a fresh process.
//
// Doctrine (same as #73): any key mismatch invalidates; restore is verified structurally
// (shape-checked per layer) and the strict path's losslessness is gated by
// PREFIXE2E-with-restart before any default flip.
//
// Opt-in: QWISP_PREFIX_PERSIST=1 (default OFF until fidelity + footprint data).
// Caps: QWISP_PREFIX_PERSIST_SLOTS files (LRU by mtime, default 4),
//       QWISP_PREFIX_PERSIST_MAX_MB per artifact (default 512 — skip larger saves).
public enum PrefixPersist {

    static let magic: UInt32 = 0x3150_5751          // "QWP1" LE
    static let format: UInt32 = 1

    public static var enabled: Bool { Tell.envInt("QWISP_PREFIX_PERSIST", 0) != 0 }
    // #112 stable-prefix tier (default ON; QWISP_PREFIX_STABLE=0 opts out): a harness's
    // system+tools block is operationally defined by RECURRENCE — the shared prefix of two
    // DIFFERENT conversations. That trigger writes the SAME key once per (harness × prompt
    // version), so the per-turn SSD wear that keeps #89 opt-in cannot occur here.
    public static var stableEnabled: Bool { Tell.envInt("QWISP_PREFIX_STABLE", 1) != 0 }
    /// Minimum shared-prefix restore point (tokens) that counts as recurrence evidence.
    public static var stableMinTokens: Int { Swift.max(64, Tell.envInt("QWISP_PREFIX_STABLE_MIN", 1024)) }
    /// Disk restore lookups run when either tier is on.
    public static var lookupEnabled: Bool { enabled || stableEnabled }
    static var maxSlots: Int { Swift.max(1, Tell.envInt("QWISP_PREFIX_PERSIST_SLOTS", 20)) }
    static var maxBytes: Int { Swift.max(1, Tell.envInt("QWISP_PREFIX_PERSIST_MAX_MB", 512)) * 1_048_576 }
    /// #112: total store byte budget (LRU-evicted), sized for 6-20 harness entries.
    static var storeBudget: Int { Swift.max(1, Tell.envInt("QWISP_PREFIX_STORE_MB", 2048)) * 1_048_576 }

    static var defaultDir: URL {
        if let p = ProcessInfo.processInfo.environment["QWISP_PREFIX_PERSIST_DIR"] {
            return URL(fileURLWithPath: p)     // test/gate isolation hook
        }
        return FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".cache/qwisp/prefix")
    }

    /// Diagnostics: number of successful cross-process (disk) restores this process.
    /// The restart gate asserts this increments — byte-identity alone can't distinguish
    /// "restored" from "silently re-prefilled cold".
    nonisolated(unsafe) public static var restoreHits = 0

    static func fnv1a(_ bytes: some Sequence<UInt8>) -> UInt64 {
        var h: UInt64 = 0xcbf2_9ce4_8422_2325
        for b in bytes { h = (h ^ UInt64(b)) &* 0x0000_0100_0000_01b3 }
        return h
    }

    static func url(dir: URL, modelDir: String, tokens: [Int32]) -> URL {
        var id = Array(modelDir.utf8)
        for t in tokens { withUnsafeBytes(of: t.littleEndian) { id.append(contentsOf: $0) } }
        return dir.appendingPathComponent(String(format: "prefix-%016llx.bin", fnv1a(id)))
    }

    // File layout: magic ∥ format ∥ modelLen u32 ∥ model utf8 ∥ tokenCount u32 ∥
    //              tokens Int32 LE ∥ state blob (SeedlessFusedForward layout, opaque here).
    static func encode(modelDir: String, tokens: [Int32], state: Data) -> Data {
        var d = Data()
        func u32(_ v: UInt32) { withUnsafeBytes(of: v.littleEndian) { d.append(contentsOf: $0) } }
        let m = Data(modelDir.utf8)
        u32(magic); u32(format); u32(UInt32(m.count))
        d.append(m)
        u32(UInt32(tokens.count))
        tokens.withUnsafeBufferPointer { p in p.baseAddress.map { d.append(Data(bytes: $0, count: tokens.count * 4)) } }
        d.append(state)
        return d
    }

    /// Parse header + tokens (cheap; `wantState` false skips materializing the blob).
    static func decode(_ d: Data, wantState: Bool) -> (model: String, tokens: [Int32], state: Data)? {
        var off = 0
        func u32() -> UInt32? {
            guard d.count >= off + 4 else { return nil }
            let v = d.subdata(in: off ..< off + 4).withUnsafeBytes { $0.loadUnaligned(as: UInt32.self) }
            off += 4
            return UInt32(littleEndian: v)
        }
        guard u32() == magic, u32() == format,
              let ml = u32().map(Int.init), d.count >= off + ml,
              let model = String(data: d.subdata(in: off ..< off + ml), encoding: .utf8) else { return nil }
        off += ml
        guard let tc = u32().map(Int.init), d.count >= off + tc * 4 else { return nil }
        let tokens: [Int32] = d.subdata(in: off ..< off + tc * 4).withUnsafeBytes { Array($0.bindMemory(to: Int32.self)) }
        off += tc * 4
        return (model, tokens, wantState ? d.subdata(in: off ..< d.count) : Data())
    }

    /// Cheap existence probe for `tokens` (no blob IO) — gates the expensive state copy.
    public static func has(modelDir: String, tokens: [Int32], dir: URL? = nil) -> Bool {
        FileManager.default.fileExists(atPath: url(dir: dir ?? defaultDir, modelDir: modelDir, tokens: tokens).path)
    }

    /// Persist one content-boundary state. Skips oversized blobs; LRU-evicts beyond maxSlots.
    public static func save(modelDir: String, tokens: [Int32], state: Data, dir: URL? = nil) {
        guard state.count <= maxBytes, !tokens.isEmpty else { return }
        let d = dir ?? defaultDir
        let fm = FileManager.default
        try? fm.createDirectory(at: d, withIntermediateDirectories: true)
        let u = url(dir: d, modelDir: modelDir, tokens: tokens)
        // Same key = same content: refresh mtime for LRU, skip the (large) rewrite.
        if fm.fileExists(atPath: u.path) {
            try? fm.setAttributes([.modificationDate: Date()], ofItemAtPath: u.path)
            return
        }
        try? encode(modelDir: modelDir, tokens: tokens, state: state).write(to: u, options: .atomic)
        evict(dir: d)
    }

    /// The stored entry whose token sequence is the LONGEST prefix of `content` (this model).
    public static func bestMatch(modelDir: String, content: [Int32], dir: URL? = nil) -> (tokens: [Int32], state: Data)? {
        let d = dir ?? defaultDir
        let fm = FileManager.default
        guard let files = try? fm.contentsOfDirectory(at: d, includingPropertiesForKeys: nil) else { return nil }
        var best: (url: URL, tokens: [Int32])? = nil
        for f in files where f.pathExtension == "bin" {
            guard let data = try? Data(contentsOf: f),
                  let (model, tokens, _) = decode(data, wantState: false),
                  model == modelDir, tokens.count <= content.count,
                  tokens.count > (best?.tokens.count ?? 0),
                  Array(content[0 ..< tokens.count]) == tokens else { continue }
            best = (f, tokens)
        }
        guard let best, let data = try? Data(contentsOf: best.url),
              let (_, tokens, state) = decode(data, wantState: true) else { return nil }
        try? fm.setAttributes([.modificationDate: Date()], ofItemAtPath: best.url.path)   // LRU touch
        return (tokens, state)
    }

    static func evict(dir: URL, budget: Int? = nil) {
        let fm = FileManager.default
        guard let files = try? fm.contentsOfDirectory(at: dir, includingPropertiesForKeys: [.contentModificationDateKey, .fileSizeKey]) else { return }
        var dated = files.filter { $0.pathExtension == "bin" }.map { f -> (url: URL, mtime: Date, size: Int) in
            let rv = try? f.resourceValues(forKeys: [.contentModificationDateKey, .fileSizeKey])
            return (f, rv?.contentModificationDate ?? .distantPast, rv?.fileSize ?? 0)
        }.sorted { $0.mtime < $1.mtime }                       // oldest first
        var total = dated.reduce(0) { $0 + $1.size }
        let cap = budget ?? storeBudget
        while let oldest = dated.first, dated.count > maxSlots || total > cap {
            try? fm.removeItem(at: oldest.url)
            total -= oldest.size
            dated.removeFirst()
        }
    }

    /// Pure self-check (no GPU, tmp dir): round trip, longest-prefix match, model isolation, LRU.
    public static func selfCheck() -> [(String, Bool)] {
        let tmp = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("qwisp-prefix-selfcheck-\(ProcessInfo.processInfo.processIdentifier)")
        defer { try? FileManager.default.removeItem(at: tmp) }
        let state = Data((0 ..< 64).map { UInt8($0) })
        let a: [Int32] = [1, 2, 3, 4], ab: [Int32] = [1, 2, 3, 4, 5, 6]
        save(modelDir: "/m", tokens: a, state: state, dir: tmp)
        save(modelDir: "/m", tokens: ab, state: state + state, dir: tmp)
        save(modelDir: "/other", tokens: ab + [7], state: state, dir: tmp)
        let hitLong = bestMatch(modelDir: "/m", content: ab + [9, 9], dir: tmp)
        let hitShort = bestMatch(modelDir: "/m", content: [1, 2, 3, 4, 99], dir: tmp)
        let missDiverge = bestMatch(modelDir: "/m", content: [8, 8, 8], dir: tmp)
        var checks: [(String, Bool)] = [
            ("longest_prefix", hitLong?.tokens == ab && hitLong?.state == state + state),
            ("shorter_prefix", hitShort?.tokens == a && hitShort?.state == state),
            ("model_isolation", bestMatch(modelDir: "/nope", content: ab, dir: tmp) == nil),
            ("miss_diverge", missDiverge == nil),
        ]
        // #112: existence probe (no blob IO).
        checks.append(("has_existing", has(modelDir: "/m", tokens: a, dir: tmp)))
        checks.append(("has_missing", !has(modelDir: "/m", tokens: [77, 78], dir: tmp)))
        // LRU: with maxSlots files already present, adding more evicts the oldest.
        for i in 0 ..< maxSlots + 2 {
            save(modelDir: "/m", tokens: [100, Int32(i)], state: state, dir: tmp)
        }
        let n = (try? FileManager.default.contentsOfDirectory(at: tmp, includingPropertiesForKeys: nil).filter { $0.pathExtension == "bin" }.count) ?? -1
        checks.append(("lru_cap", n == maxSlots))
        // #112: byte-budget eviction — squeeze the budget to one file's size ⇒ only the
        // most recently used survives.
        func binCount() -> Int {
            (try? FileManager.default.contentsOfDirectory(at: tmp, includingPropertiesForKeys: nil).filter { $0.pathExtension == "bin" }.count) ?? -1
        }
        let one = (try? FileManager.default.contentsOfDirectory(at: tmp, includingPropertiesForKeys: [.fileSizeKey])
            .filter { $0.pathExtension == "bin" }
            .compactMap { try? $0.resourceValues(forKeys: [.fileSizeKey]).fileSize }.max()) ?? 0
        evict(dir: tmp, budget: one)
        checks.append(("byte_budget_evict", binCount() == 1 && one > 0))
        return checks
    }
}

// RAM tier for the cross-request prefix cache (issue #117): interleaved conversations
// (OpenCode title-gen / tabs / sub-sessions) thrash the single-path in-arena cache — every
// switch re-prefills everything past the shared harness prefix (measured reuse 4–52%,
// TTFT up to 7 min on a 49K prompt). This store keeps whole-conversation decode states
// (SeedlessFusedForward.persistentStateData blobs — path-independent, unlike FullSnapshot's
// KV-length-only restore points) in RAM, keyed by token content, LRU-evicted by byte
// budget. No disk writes — the SSD-wear objection that keeps #89 opt-in does not apply.
public struct PrefixRAMStore {
    public var budget: Int                                    // bytes; 0 = disabled
    // MRU-first; an entry ≈ 24KB/token (attention KV used slice + GDN state).
    var entries: [(tokens: [Int32], state: Data)] = []
    public init(budget: Int = 0) { self.budget = budget }
    var bytes: Int { entries.reduce(0) { $0 + $1.state.count } }

    /// Store one content-boundary state. Entries that are a prefix of `tokens` (earlier
    /// turns of the same conversation, or the same key) are superseded and dropped.
    public mutating func save(tokens: [Int32], state: Data) {
        guard budget > 0, !tokens.isEmpty, state.count <= budget else { return }
        entries.removeAll { $0.tokens.count <= tokens.count
            && Array(tokens[0 ..< $0.tokens.count]) == $0.tokens }
        entries.insert((tokens, state), at: 0)
        while bytes > budget { entries.removeLast() }
    }

    /// The stored entry whose token sequence is the LONGEST prefix of `content`; touches LRU.
    public mutating func bestMatch(content: [Int32]) -> (tokens: [Int32], state: Data)? {
        var best = -1
        for i in entries.indices
        where entries[i].tokens.count <= content.count
            && (best < 0 || entries[i].tokens.count > entries[best].tokens.count)
            && Array(content[0 ..< entries[i].tokens.count]) == entries[i].tokens {
            best = i
        }
        guard best >= 0 else { return nil }
        let e = entries.remove(at: best)
        entries.insert(e, at: 0)
        return e
    }

    public mutating func removeAll() { entries.removeAll() }

    /// Pure self-check (no GPU): longest match, supersede, LRU byte eviction, budget gates.
    public static func selfCheck() -> [(String, Bool)] {
        func blob(_ n: Int, _ b: UInt8) -> Data { Data(repeating: b, count: n) }
        var s = PrefixRAMStore(budget: 1000)
        let a: [Int32] = [1, 2, 3, 4], a2: [Int32] = [1, 2, 3, 4, 5, 6], b: [Int32] = [9, 8, 7]
        s.save(tokens: a, state: blob(100, 1))
        s.save(tokens: b, state: blob(100, 2))
        s.save(tokens: a2, state: blob(100, 3))                   // supersedes a
        var checks: [(String, Bool)] = [
            ("supersede_prefix", s.entries.count == 2),
            ("longest_prefix", s.bestMatch(content: a2 + [7])?.state == blob(100, 3)),
            ("miss_after_supersede", s.bestMatch(content: [1, 2, 3, 4, 99]) == nil),
            ("unrelated_kept", s.bestMatch(content: b + [0])?.state == blob(100, 2)),
            ("miss_diverge", s.bestMatch(content: [42]) == nil),
        ]
        var l = PrefixRAMStore(budget: 250)
        l.save(tokens: [1], state: blob(100, 1))
        l.save(tokens: [2], state: blob(100, 2))
        _ = l.bestMatch(content: [1])                             // touch [1] → MRU
        l.save(tokens: [3], state: blob(100, 3))                  // 300 > 250 → evict LRU = [2]
        checks.append(("lru_evict_touched_kept", l.bestMatch(content: [2]) == nil && l.bestMatch(content: [1]) != nil))
        var d = PrefixRAMStore(budget: 0)
        d.save(tokens: [1], state: blob(10, 1))
        checks.append(("disabled_budget0", d.bestMatch(content: [1]) == nil))
        var o = PrefixRAMStore(budget: 100)
        o.save(tokens: [1], state: blob(200, 1))                  // entry larger than the whole budget
        checks.append(("oversize_skip", o.bestMatch(content: [1]) == nil))
        return checks
    }
}
