import Foundation

// Bolt calibration warm-start artifact (issue #73).
//
// Bolt calibrates per process (strict prefill + ≥QWISP_CALIB_MIN_ROWS greedy rows with
// routing capture) — fine for the long-lived server, but `qwisp chat` one-shots pay
// minutes of strict-speed decode every invocation. This artifact persists the freeze
// basis (baseCounts + baseCoact) so a new process warm-starts and freezes immediately.
//
// Doctrine (calib-poisoning, fixed v0.3.2): a stale basis is the same failure mode as a
// trivial-warmup calib — the artifact SEEDS the basis, rolling recalib (R=128) adapts it
// from live traffic; on graceful shutdown the latest basis is written back
// (last-known-good). Any key mismatch (model, mixed tail, C, dims, format) invalidates.
//
// Opt-in: QWISP_CALIB_CACHE=1. Default OFF until the warm-vs-cold A/B (TF fidelity +
// LOOPY rate, free-run gate per the attractor doctrine) lands — see #73.
public enum CalibArtifact {

    static let magic: UInt32 = 0x3143_5751          // "QWC1" little-endian
    /// Bump on any layout/semantics change — old files then simply miss.
    static let format: UInt32 = 1

    public static var enabled: Bool { Tell.envInt("QWISP_CALIB_CACHE", 0) != 0 }

    /// FNV-1a 64 — stable across processes (Swift's Hasher is per-process seeded).
    static func fnv1a(_ s: String) -> UInt64 {
        var h: UInt64 = 0xcbf2_9ce4_8422_2325
        for b in s.utf8 { h = (h ^ UInt64(b)) &* 0x0000_0100_0000_01b3 }
        return h
    }

    static func url(dir: URL, modelDir: String, mixedTailDir: String?, C: Int) -> URL {
        let key = fnv1a("\(modelDir)|\(mixedTailDir ?? "")")
        return dir.appendingPathComponent(String(format: "calib-%016llx-C%d.bin", key, C))
    }

    static var defaultDir: URL {
        FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".cache/qwisp")
    }

    struct Header: Codable, Equatable {
        let model: String       // modelDir path — full-string check, not just the hash
        let mixed: String       // mixed tail dir ("" = generic bolt)
        let c: Int
        let nLayers: Int
        let nE: Int
        let savedAt: Double     // unix time, informational
    }

    /// Serialize: magic ∥ format ∥ headerLen ∥ header-JSON ∥ counts Int32 ∥ coact Int32.
    static func encode(header: Header, counts: [[Int]], coact: [[[Int]]]) -> Data? {
        guard let hj = try? JSONEncoder().encode(header) else { return nil }
        var d = Data()
        for v in [magic, format, UInt32(hj.count)] { withUnsafeBytes(of: v.littleEndian) { d.append(contentsOf: $0) } }
        d.append(hj)
        var flat: [Int32] = []
        flat.reserveCapacity(header.nLayers * header.nE * (header.nE + 1))
        for l in counts { for v in l { flat.append(Int32(clamping: v)) } }
        for l in coact { for r in l { for v in r { flat.append(Int32(clamping: v)) } } }
        flat.withUnsafeBufferPointer { d.append(Data(buffer: $0)) }
        return d
    }

    static func decode(_ d: Data, expect: Header) -> (counts: [[Int]], coact: [[[Int]]])? {
        var off = 0
        func u32() -> UInt32? {
            guard d.count >= off + 4 else { return nil }
            let v = d.subdata(in: off ..< off + 4).withUnsafeBytes { $0.loadUnaligned(as: UInt32.self) }
            off += 4
            return UInt32(littleEndian: v)
        }
        guard u32() == magic, u32() == format, let hl = u32().map(Int.init),
              d.count >= off + hl,
              let h = try? JSONDecoder().decode(Header.self, from: d.subdata(in: off ..< off + hl))
        else { return nil }
        off += hl
        // Key check: everything but savedAt must match the running configuration.
        guard h.model == expect.model, h.mixed == expect.mixed, h.c == expect.c,
              h.nLayers == expect.nLayers, h.nE == expect.nE else { return nil }
        let nCounts = h.nLayers * h.nE, nCoact = h.nLayers * h.nE * h.nE
        guard d.count == off + (nCounts + nCoact) * 4 else { return nil }
        let flat: [Int32] = d.subdata(in: off ..< d.count).withUnsafeBytes { Array($0.bindMemory(to: Int32.self)) }
        var counts = [[Int]](repeating: [], count: h.nLayers)
        var coact = [[[Int]]](repeating: [], count: h.nLayers)
        for li in 0 ..< h.nLayers {
            counts[li] = flat[li * h.nE ..< (li + 1) * h.nE].map(Int.init)
        }
        for li in 0 ..< h.nLayers {
            var rows = [[Int]](repeating: [], count: h.nE)
            for e in 0 ..< h.nE {
                let base = nCounts + (li * h.nE + e) * h.nE
                rows[e] = flat[base ..< base + h.nE].map(Int.init)
            }
            coact[li] = rows
        }
        return (counts, coact)
    }

    /// Load a matching artifact; nil on any miss/mismatch (caller falls back to cold calib).
    public static func load(modelDir: String, mixedTailDir: String?, C: Int,
                            nLayers: Int, nE: Int,
                            dir: URL? = nil) -> (counts: [[Int]], coact: [[[Int]]])? {
        let d = dir ?? defaultDir
        let expect = Header(model: modelDir, mixed: mixedTailDir ?? "", c: C,
                            nLayers: nLayers, nE: nE, savedAt: 0)
        guard let data = try? Data(contentsOf: url(dir: d, modelDir: modelDir, mixedTailDir: mixedTailDir, C: C)) else { return nil }
        return decode(data, expect: expect)
    }

    /// Write the current basis (atomic; best-effort — a failed save only costs the warm-start).
    public static func save(modelDir: String, mixedTailDir: String?, C: Int,
                            counts: [[Int]], coact: [[[Int]]],
                            dir: URL? = nil) {
        let d = dir ?? defaultDir
        guard let nE = counts.first?.count, !counts.isEmpty else { return }
        let header = Header(model: modelDir, mixed: mixedTailDir ?? "", c: C,
                            nLayers: counts.count, nE: nE,
                            savedAt: Date().timeIntervalSince1970)
        guard let data = encode(header: header, counts: counts, coact: coact) else { return }
        try? FileManager.default.createDirectory(at: d, withIntermediateDirectories: true)
        try? data.write(to: url(dir: d, modelDir: modelDir, mixedTailDir: mixedTailDir, C: C), options: .atomic)
    }

    /// Pure self-check (no GPU, tmp-dir round trip): save→load identity, key mismatches miss.
    public static func selfCheck() -> [(String, Bool)] {
        let tmp = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("qwisp-calib-selfcheck-\(ProcessInfo.processInfo.processIdentifier)")
        defer { try? FileManager.default.removeItem(at: tmp) }
        let L = 3, E = 4
        let counts = (0 ..< L).map { li in (0 ..< E).map { li * 10 + $0 } }
        let coact = (0 ..< L).map { li in (0 ..< E).map { a in (0 ..< E).map { li + a + $0 } } }
        save(modelDir: "/m", mixedTailDir: "/t", C: 64, counts: counts, coact: coact, dir: tmp)
        let hit = load(modelDir: "/m", mixedTailDir: "/t", C: 64, nLayers: L, nE: E, dir: tmp)
        return [
            ("roundtrip", hit?.counts == counts && hit?.coact == coact),
            ("miss_model", load(modelDir: "/other", mixedTailDir: "/t", C: 64, nLayers: L, nE: E, dir: tmp) == nil),
            ("miss_c", load(modelDir: "/m", mixedTailDir: "/t", C: 128, nLayers: L, nE: E, dir: tmp) == nil),
            ("miss_mixed", load(modelDir: "/m", mixedTailDir: nil, C: 64, nLayers: L, nE: E, dir: tmp) == nil),
            ("miss_dims", load(modelDir: "/m", mixedTailDir: "/t", C: 64, nLayers: L, nE: E + 1, dir: tmp) == nil),
        ]
    }
}
