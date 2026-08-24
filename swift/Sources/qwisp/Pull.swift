import Foundation
import Hub

// Model acquisition. `qwisp pull` downloads a checkpoint via swift-transformers' HubApi
// (already linked — no python / hf CLI needed) and writes its path into the config file.
// chat/serve reuse `ensureModel` to nudge the user when no model is present.
enum ModelStore {
    static let defaultRepo = "Youssofal/Qwen3.6-35B-A3B-MTPLX-Optimized-Speed-FP16"

    // A directory is a usable model if it holds a config.json (the checkpoint manifest).
    static func isModel(_ path: String) -> Bool {
        FileManager.default.fileExists(atPath: path + "/config.json")
    }

    /// The engine is specialised to the MTPLX *quant layout* (4-bit affine gs=64 dens/experts;
    /// router + shared_expert_gate only at 8-bit). Anything else must be rejected HERE with a
    /// real message — engine preconditions die as a bare trace trap (issue #51: an oQ4 requant
    /// of the same base SIGTRAPed). Acceptance:
    ///   1. `mtplx_policy` stamp (canonical Youssofal MTPLX), OR
    ///   2. the same quant recipe detected from `quantization` (e.g. Ornith-1.5 / mlx-community
    ///      plain 4-bit). Mixed-precision recipes (Nail/OptiQ with 8-bit attn) stay rejected.
    static func requireSupported(_ modelDir: String) {
        let url = URL(fileURLWithPath: modelDir).appendingPathComponent("config.json")
        guard let data = try? Data(contentsOf: url),
              let top = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] else {
            FileHandle.standardError.write(Data("cannot read \(url.path) — not a model directory?\n".utf8))
            exit(1)
        }
        if isSupportedCheckpoint(top) { return }
        FileHandle.standardError.write(Data("""
        Unsupported checkpoint: \(modelDir)
        qwisp needs the MTPLX quant layout of Qwen3.5/3.6-35B-A3B (4-bit affine gs=64; only
        mlp.gate + shared_expert_gate at 8-bit). This directory is a different architecture or
        a different quant recipe, and the engine's kernels are shaped for that layout exactly.
            qwisp pull    # download the supported checkpoint (~20 GB) + write config

        """.utf8))
        exit(1)
    }

    /// Pure gate: true iff `config.json` is the MTPLX stamp or an equivalent quant recipe.
    static func isSupportedCheckpoint(_ top: [String: Any]) -> Bool {
        if top["mtplx_policy"] != nil { return true }
        return matchesMTPLXQuantRecipe(top["quantization"] as? [String: Any])
    }

    /// MTPLX / mlx-community-4bit / Ornith-MLX-4bit recipe: default 4/64/affine, and every
    /// per-tensor override is bits=8 on `….mlp.gate` or `….mlp.shared_expert_gate` only.
    /// At least one router-gate override is required so uniform-4bit (gates packed as 4)
    /// cannot pass and then SIGTRAP in `qmm8`.
    static func matchesMTPLXQuantRecipe(_ q: [String: Any]?) -> Bool {
        guard let q else { return false }
        guard intVal(q["bits"]) == 4, intVal(q["group_size"]) == 64 else { return false }
        if let mode = q["mode"] as? String, mode != "affine" { return false }
        var sawRouterGate = false
        for (k, v) in q {
            if k == "bits" || k == "group_size" || k == "mode" { continue }
            guard let ov = v as? [String: Any], intVal(ov["bits"]) == 8 else { return false }
            let isRouter = k.hasSuffix(".mlp.gate")
            let isSharedGate = k.hasSuffix(".mlp.shared_expert_gate")
            guard isRouter || isSharedGate else { return false }
            if isRouter { sawRouterGate = true }
        }
        return sawRouterGate
    }

    /// JSONSerialization boxes numbers as NSNumber; accept Int bridging either way.
    private static func intVal(_ v: Any?) -> Int? {
        if let i = v as? Int { return i }
        if let n = v as? NSNumber { return n.intValue }
        return nil
    }

    /// GPU-free gate self-check (COMPTEST / Selftest).
    static func selfCheck() -> [(String, Bool)] {
        func cfg(_ bits: Int, overrides: [String: [String: Any]] = [:], policy: Bool = false) -> [String: Any] {
            var q: [String: Any] = ["bits": bits, "group_size": 64, "mode": "affine"]
            for (k, v) in overrides { q[k] = v }
            var top: [String: Any] = ["quantization": q]
            if policy { top["mtplx_policy"] = ["name": "test"] }
            return top
        }
        let gate8: [String: Any] = ["bits": 8, "group_size": 64]
        let ornith = cfg(4, overrides: [
            "language_model.model.layers.0.mlp.gate": gate8,
            "language_model.model.layers.0.mlp.shared_expert_gate": gate8,
        ])
        let nailish = cfg(4, overrides: [
            "language_model.model.layers.0.mlp.gate": gate8,
            "language_model.model.layers.0.linear_attn.in_proj_qkv": gate8,
        ])
        return [
            ("mtplx_stamp", isSupportedCheckpoint(cfg(4, policy: true))),
            ("ornith_recipe", isSupportedCheckpoint(ornith)),
            ("reject_uniform_4bit", !isSupportedCheckpoint(cfg(4))),
            ("reject_nail_attn8", !isSupportedCheckpoint(nailish)),
            ("reject_8bit_default", !isSupportedCheckpoint(cfg(8))),
            ("reject_no_quant", !isSupportedCheckpoint([:])),
        ]
    }

    // Download `repo` and point the config file at it. Returns the local model path.
    // HF_ENDPOINT switches the Hub host (mirrors — e.g. https://hf-mirror.com — for regions
    // where huggingface.co is slow or blocked); HF_TOKEN is picked up by HubApi itself.
    static func pull(repo: String = defaultRepo) async throws -> String {
        let sizeNote = repo == defaultRepo ? " (~20 GB — this takes a while)" : ""
        let endpoint = ProcessInfo.processInfo.environment["HF_ENDPOINT"]
        if let endpoint {
            FileHandle.standardError.write(Data("Using Hub endpoint \(endpoint) (HF_ENDPOINT)\n".utf8))
        }
        FileHandle.standardError.write(Data("Downloading \(repo)\(sizeNote)…\n".utf8))
        let hub = HubApi(endpoint: endpoint)
        var lastPct = -1
        let url: URL
        do {
            url = try await hub.snapshot(from: repo, matching: []) { progress in
                let pct = Int(progress.fractionCompleted * 100)
                if pct != lastPct {
                    lastPct = pct
                    FileHandle.standardError.write(Data("\r  \(pct)%   ".utf8))
                }
            }
        } catch {
            FileHandle.standardError.write(Data("""

            download failed: \(error.localizedDescription)
            If huggingface.co is slow or blocked in your region:
              • try a mirror:   HF_ENDPOINT=https://hf-mirror.com qwisp pull
              • or download it any other way (e.g. `hf download \(repo)`) and point qwisp at
                the directory via QWISP_MODEL or "model" in ~/.config/qwisp/config.json

            """.utf8))
            throw error
        }
        FileHandle.standardError.write(Data("\r  100%\n".utf8))
        let path = url.path
        try Config.writeModel(path)
        FileHandle.standardError.write(Data("Model ready: \(path)\n→ wrote \(Config.defaultPath)\n".utf8))
        return path
    }

    // Resolve the model, and if it's absent decide what to do based on interactivity.
    //  - interactive TTY: offer to pull now (y/N); on yes, download and return the new path.
    //  - non-interactive (pipe) or a daemon: return nil (caller prints the hint and exits).
    // `allowPrompt` is false for `serve` — a LaunchAgent has no TTY, must never block.
    static func ensureModel(_ path: String, allowPrompt: Bool) async -> String? {
        if isModel(path) { return path }
        guard allowPrompt, isatty(STDIN_FILENO) != 0 else { return nil }
        FileHandle.standardError.write(Data(
            "No model found at \(path).\nDownload \(defaultRepo) (~20 GB) now? [y/N] ".utf8))
        let answer = readLine(strippingNewline: true)?.lowercased() ?? ""
        guard answer == "y" || answer == "yes" else { return nil }
        return try? await pull()
    }

    static let missingModelHint = """
    No model found. Get one with:
        qwisp pull                 # default checkpoint (~20 GB) → writes ~/.config/qwisp/config.json
        qwisp pull <hf-repo-id>    # a specific checkpoint
    Or point qwisp at an existing directory via QWISP_MODEL or ~/.config/qwisp/config.json.
    """
}
