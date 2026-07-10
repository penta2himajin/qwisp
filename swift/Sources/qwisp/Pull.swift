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

    // Download `repo` and point the config file at it. Returns the local model path.
    static func pull(repo: String = defaultRepo) async throws -> String {
        let sizeNote = repo == defaultRepo ? " (~20 GB — this takes a while)" : ""
        FileHandle.standardError.write(Data("Downloading \(repo)\(sizeNote)…\n".utf8))
        let hub = HubApi()
        var lastPct = -1
        let url = try await hub.snapshot(from: repo, matching: []) { progress in
            let pct = Int(progress.fractionCompleted * 100)
            if pct != lastPct {
                lastPct = pct
                FileHandle.standardError.write(Data("\r  \(pct)%   ".utf8))
            }
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
