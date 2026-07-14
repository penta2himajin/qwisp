import Foundation

// Version-update notice (issue #53). A single unauthenticated GET to GitHub's
// releases/latest — no payload, so not telemetry, but documented and opt-out anyway
// (QWISP_UPDATE_CHECK=0). benchtest never calls this: it stays network-silent.
//
// All output goes to stderr — release.sh's 3b guard string-compares `qwisp version`
// stdout against the tag, and stdout must stay a bare version string.
enum UpdateCheck {
    static let releasesURL = "https://api.github.com/repos/penta2himajin/qwisp/releases/latest"
    static var stampPath: String {
        FileManager.default.homeDirectoryForCurrentUser.path + "/.config/qwisp/last-update-check"
    }

    static var enabled: Bool { ProcessInfo.processInfo.environment["QWISP_UPDATE_CHECK"] != "0" }

    /// Latest release tag ("v0.3.4" → "0.3.4"). nil on ANY failure — the check must never
    /// surface an error of its own.
    static func latestVersion(timeout: TimeInterval = 2) async -> String? {
        guard let url = URL(string: releasesURL) else { return nil }
        var req = URLRequest(url: url, timeoutInterval: timeout)
        req.setValue("qwisp/\(Config.version)", forHTTPHeaderField: "User-Agent")
        guard let (data, _) = try? await URLSession.shared.data(for: req),
              let obj = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any],
              let tag = obj["tag_name"] as? String else { return nil }
        return tag.hasPrefix("v") ? String(tag.dropFirst()) : tag
    }

    /// Numeric-aware dotted compare: "0.3.10" > "0.3.4".
    static func isNewer(_ a: String, than b: String) -> Bool {
        let pa = a.split(separator: ".").map { Int($0) ?? 0 }
        let pb = b.split(separator: ".").map { Int($0) ?? 0 }
        for i in 0 ..< Swift.max(pa.count, pb.count) {
            let x = i < pa.count ? pa[i] : 0, y = i < pb.count ? pb[i] : 0
            if x != y { return x > y }
        }
        return false
    }

    /// `qwisp version`: check every time (the user is explicitly asking about versions).
    static func reportForVersionCommand() async {
        guard enabled, let latest = await latestVersion() else { return }
        let line = isNewer(latest, than: Config.version)
            ? "latest: v\(latest) — upgrade with: brew upgrade qwisp"
            : "up to date"
        FileHandle.standardError.write(Data((line + "\n").utf8))
    }

    /// chat/serve startup: at most once per 24h (mtime stamp), fully in the background,
    /// one stderr line only when a newer release exists.
    static func noticeInBackground() {
        guard enabled else { return }
        if let attrs = try? FileManager.default.attributesOfItem(atPath: stampPath),
           let m = attrs[.modificationDate] as? Date, Date().timeIntervalSince(m) < 86_400 { return }
        try? FileManager.default.createDirectory(
            atPath: (stampPath as NSString).deletingLastPathComponent, withIntermediateDirectories: true)
        FileManager.default.createFile(atPath: stampPath, contents: nil)   // stamp first: failures also wait 24h
        Task.detached(priority: .background) {
            guard let latest = await latestVersion(), isNewer(latest, than: Config.version) else { return }
            FileHandle.standardError.write(Data(
                "[qwisp] v\(latest) is available (running \(Config.version)) — brew upgrade qwisp   (opt out: QWISP_UPDATE_CHECK=0)\n".utf8))
        }
    }

    /// GPU-free self-check for the version compare (the only logic here worth breaking).
    static func selfCheck() -> (Int, Int, [String]) {
        var pass = 0
        var log: [String] = []
        let cases: [(String, String, Bool)] = [
            ("0.3.4", "0.3.3", true), ("0.3.3", "0.3.3", false), ("0.3.10", "0.3.4", true),
            ("1.0.0", "0.9.9", true), ("0.3.3", "0.3.4", false), ("0.4", "0.3.9", true),
        ]
        for (a, b, want) in cases {
            let got = isNewer(a, than: b)
            if got == want { pass += 1 } else { log.append("FAIL isNewer(\(a), \(b)) → \(got), want \(want)") }
        }
        return (pass, cases.count, log)
    }
}
