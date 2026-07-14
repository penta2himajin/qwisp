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

    /// The engine is specialised to the MTPLX checkpoint layout; anything else must be
    /// rejected HERE with a real message — engine preconditions die as a bare trace trap
    /// (issue #51: an oQ4 requant of the same base model SIGTRAPed benchtest). The MTPLX
    /// pipeline stamps `mtplx_policy` into config.json; that key is the format signature
    /// (the oQ4 repo lacks it while matching everything else).
    static func requireSupported(_ modelDir: String) {
        let url = URL(fileURLWithPath: modelDir).appendingPathComponent("config.json")
        guard let data = try? Data(contentsOf: url),
              let top = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] else {
            FileHandle.standardError.write(Data("cannot read \(url.path) — not a model directory?\n".utf8))
            exit(1)
        }
        guard top["mtplx_policy"] == nil else { return }
        FileHandle.standardError.write(Data("""
        Unsupported checkpoint: \(modelDir)
        qwisp is single-model-specialised: it supports the MTPLX build of Qwen3.6-35B-A3B only
        (\(defaultRepo)). This directory is a different model or a different quant layout of the
        same base model, and the engine's kernels are shaped for the MTPLX layout exactly.
            qwisp pull    # download the supported checkpoint (~20 GB) + write config

        """.utf8))
        exit(1)
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
