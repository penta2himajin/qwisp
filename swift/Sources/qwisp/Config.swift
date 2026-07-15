import Foundation
import QwispCore

// Resident-service config. `brew services` runs qwisp as a LaunchAgent, which does NOT
// inherit the shell environment — so the model path can't come only from QWISP_MODEL.
// Resolution order (most explicit wins):
//   1. QWISP_MODEL / QWISP_PORT env      (dev + scripts override)
//   2. ~/.config/qwisp/config.json       {"model": "...", "port": 8080}   (the resident source)
//   3. built-in default                  (drop the model at ~/.mtplx/… and it just works)
//
// Defaults live in code (below), NOT materialized into the user's file — the file stays a sparse
// override, so a new qwisp version's improved defaults aren't frozen out by a stale on-disk copy.
// `qwisp config` prints the effective values; `qwisp config --defaults` emits the full default set
// (generated here, never stale) for anyone who wants to pin one explicitly.
struct QwispConfig: Codable {
    var model: String?
    var port: Int?
    var lossless: Bool?
}

enum Config {
    // ── defaults (the SSoT) ──────────────────────────────────────────────
    /// Release version. release.sh auto-syncs this to the tag (commits the bump if needed)
    /// and still verifies the built binary's `qwisp version` output matches the tag.
    static let version = "0.3.4"
    static let defaultModel = FileManager.default.homeDirectoryForCurrentUser.path
        + "/.mtplx/models/Youssofal--Qwen3.6-35B-A3B-MTPLX-Optimized-Speed-FP16"
    static let defaultPort = 8080

    static var defaultPath: String {
        FileManager.default.homeDirectoryForCurrentUser.path + "/.config/qwisp/config.json"
    }

    // Missing file or malformed JSON → empty config (no crash — a fresh install has no config).
    static func load(path: String) -> QwispConfig {
        guard let data = FileManager.default.contents(atPath: path),
              let cfg = try? JSONDecoder().decode(QwispConfig.self, from: data)
        else { return QwispConfig(model: nil, port: nil, lossless: nil) }
        return cfg
    }

    static func resolveModel(env: [String: String], config: QwispConfig, default def: String) -> String {
        env["QWISP_MODEL"] ?? config.model ?? def
    }

    static func resolvePort(env: [String: String], config: QwispConfig, default def: Int) -> Int {
        if let e = env["QWISP_PORT"], let v = Int(e) { return v }
        return config.port ?? def
    }

    /// --lossless: force strict (bit-exact) on every tier. Streaming tiers (<32GB)
    /// otherwise default to bolt (near-lossless). env QWISP_LOSSLESS=1 > config > false.
    static func resolveLossless(env: [String: String], config: QwispConfig) -> Bool {
        if let e = env["QWISP_LOSSLESS"] { return e == "1" }
        return config.lossless ?? false
    }
    static func sourceOfLossless(env: [String: String], config: QwispConfig) -> String {
        env["QWISP_LOSSLESS"] != nil ? "env" : (config.lossless != nil ? "config" : "default")
    }

    // Where each effective value came from — for `qwisp config` transparency.
    static func sourceOfModel(env: [String: String], config: QwispConfig) -> String {
        env["QWISP_MODEL"] != nil ? "env" : (config.model != nil ? "config" : "default")
    }
    static func sourceOfPort(env: [String: String], config: QwispConfig) -> String {
        if let e = env["QWISP_PORT"], Int(e) != nil { return "env" }
        return config.port != nil ? "config" : "default"
    }

    // Write ONLY the model key, preserving any other keys already in the file (sparse override).
    static func writeModel(_ path: String, to cfgPath: String = defaultPath) throws {
        var cfg = load(path: cfgPath)
        cfg.model = path
        let dir = (cfgPath as NSString).deletingLastPathComponent
        try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        let enc = JSONEncoder(); enc.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        try enc.encode(cfg).write(to: URL(fileURLWithPath: cfgPath))
    }

    // Full default set as JSON — generated from the defaults above, so it can never be stale
    // relative to this binary. `qwisp config --defaults`.
    static func defaultsJSON() -> String {
        let enc = JSONEncoder(); enc.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        let full = QwispConfig(model: defaultModel, port: defaultPort, lossless: false)
        return (try? enc.encode(full)).flatMap { String(data: $0, encoding: .utf8) } ?? "{}"
    }

    // Human-readable effective config with per-key provenance. `qwisp config`.
    static func effectiveReport(env: [String: String], config: QwispConfig, path: String) -> String {
        let m = resolveModel(env: env, config: config, default: defaultModel)
        let p = resolvePort(env: env, config: config, default: defaultPort)
        let l = resolveLossless(env: env, config: config)
        return """
        config file: \(path)\(FileManager.default.fileExists(atPath: path) ? "" : "  (not present)")
          model    = \(m)  [\(sourceOfModel(env: env, config: config))]
          port     = \(p)  [\(sourceOfPort(env: env, config: config))]
          lossless = \(l)  [\(sourceOfLossless(env: env, config: config))]
        """
    }

    // GPU-free, model-free self-check — the CONFIGTEST gate.
    static func selfCheck() -> (Int, Int, [String]) {
        var passed = 0, total = 0, log: [String] = []
        func check(_ name: String, _ ok: Bool) {
            total += 1; if ok { passed += 1 } else { log.append("FAIL: \(name)") }
        }
        let cfg = QwispConfig(model: "/cfg/model", port: 9, lossless: true)
        let empty = QwispConfig(model: nil, port: nil, lossless: nil)

        check("env wins model",   resolveModel(env: ["QWISP_MODEL": "/env/m"], config: cfg, default: "/def") == "/env/m")
        check("config wins model", resolveModel(env: [:], config: cfg, default: "/def") == "/cfg/model")
        check("default model",     resolveModel(env: [:], config: empty, default: "/def") == "/def")
        check("env wins port",     resolvePort(env: ["QWISP_PORT": "1"], config: cfg, default: 8080) == 1)
        check("config wins port",  resolvePort(env: [:], config: cfg, default: 8080) == 9)
        check("default port",      resolvePort(env: [:], config: empty, default: 8080) == 8080)
        check("bad env port → config", resolvePort(env: ["QWISP_PORT": "nope"], config: cfg, default: 8080) == 9)

        // source attribution
        check("source model env",     sourceOfModel(env: ["QWISP_MODEL": "/m"], config: cfg) == "env")
        check("source model config",  sourceOfModel(env: [:], config: cfg) == "config")
        check("source model default", sourceOfModel(env: [:], config: empty) == "default")
        check("source port default",  sourceOfPort(env: [:], config: empty) == "default")
        check("source port env",      sourceOfPort(env: ["QWISP_PORT": "1"], config: empty) == "env")

        // lossless resolution (productization: <32GB defaults bolt; this forces strict)
        check("env wins lossless",     resolveLossless(env: ["QWISP_LOSSLESS": "1"], config: empty) == true)
        check("env 0 beats config",    resolveLossless(env: ["QWISP_LOSSLESS": "0"], config: cfg) == false)
        check("config lossless",       resolveLossless(env: [:], config: cfg) == true)
        check("default lossless off",  resolveLossless(env: [:], config: empty) == false)
        check("source lossless env",   sourceOfLossless(env: ["QWISP_LOSSLESS": "1"], config: empty) == "env")
        check("source lossless config", sourceOfLossless(env: [:], config: cfg) == "config")

        // strict-streaming C budget fit (issue #69): 16GB Mac (budget 10.9) → 64,
        // 18GB (12.3) → 96, forced-approx on a big machine (48+) → tier C unchanged.
        check("fitC 16GB → 64",   DeviceCalibration.fitC(tierC: 128, budgetGB: 10.9) == 64)
        check("fitC 18GB → 96",   DeviceCalibration.fitC(tierC: 128, budgetGB: 12.3) == 96)
        check("fitC big → tierC", DeviceCalibration.fitC(tierC: 128, budgetGB: 48) == 128)
        check("fitC floor is 64", DeviceCalibration.fitC(tierC: 128, budgetGB: 1) == 64)
        check("fitC 8GB tier stays", DeviceCalibration.fitC(tierC: 64, budgetGB: 5.4) == 64)

        // defaultsJSON carries the keys
        let dj = defaultsJSON()
        check("defaults json has model+port+lossless",
              dj.contains("\"model\"") && dj.contains("\"port\"") && dj.contains("8080") && dj.contains("\"lossless\""))

        // load(): missing file and malformed JSON both yield empty config, no throw.
        let tmp = NSTemporaryDirectory() + "qwisp-configtest-\(getpid())"
        check("missing file → empty", load(path: tmp + "/nope.json").model == nil)
        try? "{ not json".write(toFile: tmp + ".json", atomically: true, encoding: .utf8)
        check("malformed → empty", load(path: tmp + ".json").model == nil)
        try? #"{"model":"/x","port":7}"#.write(toFile: tmp + ".json", atomically: true, encoding: .utf8)
        let good = load(path: tmp + ".json")
        check("valid json parsed", good.model == "/x" && good.port == 7)

        // writeModel: sets model, preserves the existing port, round-trips.
        let wpath = tmp + "-write.json"
        try? #"{"port":1234}"#.write(toFile: wpath, atomically: true, encoding: .utf8)
        try? writeModel("/new/model", to: wpath)
        let after = load(path: wpath)
        check("writeModel sets model", after.model == "/new/model")
        check("writeModel keeps port", after.port == 1234)
        try? FileManager.default.removeItem(atPath: tmp + ".json")
        try? FileManager.default.removeItem(atPath: wpath)

        return (passed, total, log)
    }
}
