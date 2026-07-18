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
    static var maxSlots: Int { Swift.max(1, Tell.envInt("QWISP_PREFIX_PERSIST_SLOTS", 4)) }
    static var maxBytes: Int { Swift.max(1, Tell.envInt("QWISP_PREFIX_PERSIST_MAX_MB", 512)) * 1_048_576 }

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

    static func evict(dir: URL) {
        let fm = FileManager.default
        guard let files = try? fm.contentsOfDirectory(at: dir, includingPropertiesForKeys: [.contentModificationDateKey]) else { return }
        let bins = files.filter { $0.pathExtension == "bin" }
        guard bins.count > maxSlots else { return }
        let dated = bins.map { f -> (URL, Date) in
            ((f, (try? f.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast))
        }.sorted { $0.1 < $1.1 }
        for (f, _) in dated.prefix(bins.count - maxSlots) { try? fm.removeItem(at: f) }
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
        // LRU: with maxSlots files already present, adding more evicts the oldest.
        for i in 0 ..< maxSlots + 2 {
            save(modelDir: "/m", tokens: [100, Int32(i)], state: state, dir: tmp)
        }
        let n = (try? FileManager.default.contentsOfDirectory(at: tmp, includingPropertiesForKeys: nil).filter { $0.pathExtension == "bin" }.count) ?? -1
        checks.append(("lru_cap", n == maxSlots))
        return checks
    }
}
