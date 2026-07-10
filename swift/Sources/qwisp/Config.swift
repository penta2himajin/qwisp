import Foundation

// Resident-service config. `brew services` runs qwisp as a LaunchAgent, which does NOT
// inherit the shell environment — so the model path can't come only from QWISP_MODEL.
// Resolution order (most explicit wins):
//   1. QWISP_MODEL / QWISP_PORT env      (dev + scripts override)
//   2. ~/.config/qwisp/config.json       {"model": "...", "port": 8080}   (the resident source)
//   3. built-in default                  (drop the model at ~/.mtplx/… and it just works)
struct QwispConfig: Decodable {
    var model: String?
    var port: Int?
}

enum Config {
    static var defaultPath: String {
        FileManager.default.homeDirectoryForCurrentUser.path + "/.config/qwisp/config.json"
    }

    // Missing file or malformed JSON → empty config (no crash — a fresh install has no config).
    static func load(path: String) -> QwispConfig {
        guard let data = FileManager.default.contents(atPath: path),
              let cfg = try? JSONDecoder().decode(QwispConfig.self, from: data)
        else { return QwispConfig(model: nil, port: nil) }
        return cfg
    }

    static func resolveModel(env: [String: String], config: QwispConfig, default def: String) -> String {
        env["QWISP_MODEL"] ?? config.model ?? def
    }

    static func resolvePort(env: [String: String], config: QwispConfig, default def: Int) -> Int {
        if let e = env["QWISP_PORT"], let v = Int(e) { return v }
        return config.port ?? def
    }

    // GPU-free, model-free self-check — the CONFIGTEST gate.
    static func selfCheck() -> (Int, Int, [String]) {
        var passed = 0, total = 0, log: [String] = []
        func check(_ name: String, _ ok: Bool) {
            total += 1; if ok { passed += 1 } else { log.append("FAIL: \(name)") }
        }
        let cfg = QwispConfig(model: "/cfg/model", port: 9)
        let empty = QwispConfig(model: nil, port: nil)

        check("env wins model",   resolveModel(env: ["QWISP_MODEL": "/env/m"], config: cfg, default: "/def") == "/env/m")
        check("config wins model", resolveModel(env: [:], config: cfg, default: "/def") == "/cfg/model")
        check("default model",     resolveModel(env: [:], config: empty, default: "/def") == "/def")
        check("env wins port",     resolvePort(env: ["QWISP_PORT": "1"], config: cfg, default: 8080) == 1)
        check("config wins port",  resolvePort(env: [:], config: cfg, default: 8080) == 9)
        check("default port",      resolvePort(env: [:], config: empty, default: 8080) == 8080)
        check("bad env port → config", resolvePort(env: ["QWISP_PORT": "nope"], config: cfg, default: 8080) == 9)

        // load(): missing file and malformed JSON both yield empty config, no throw.
        let tmp = NSTemporaryDirectory() + "qwisp-configtest-\(getpid())"
        check("missing file → empty", load(path: tmp + "/nope.json").model == nil)
        try? "{ not json".write(toFile: tmp + ".json", atomically: true, encoding: .utf8)
        check("malformed → empty", load(path: tmp + ".json").model == nil)
        try? #"{"model":"/x","port":7}"#.write(toFile: tmp + ".json", atomically: true, encoding: .utf8)
        let good = load(path: tmp + ".json")
        check("valid json parsed", good.model == "/x" && good.port == 7)
        try? FileManager.default.removeItem(atPath: tmp + ".json")

        return (passed, total, log)
    }
}
