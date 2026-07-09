import Foundation
import Hummingbird
import Tokenizers
import Hub

// qwisp — product CLI + OpenAI-compatible server (productization step 5).
// Scaffold: proves the dependency graph (Hummingbird + swift-transformers) resolves
// and links. `serve` / `chat` are wired in follow-up sub-steps.

let defaultModel = "\(FileManager.default.homeDirectoryForCurrentUser.path)/.mtplx/models/Youssofal--Qwen3.6-35B-A3B-MTPLX-Optimized-Speed-FP16"
let model = ProcessInfo.processInfo.environment["QWISP_MODEL"] ?? defaultModel

let args = Array(CommandLine.arguments.dropFirst())
switch args.first {
case "serve":
    let port = Int(ProcessInfo.processInfo.environment["QWISP_PORT"] ?? "8080") ?? 8080
    let modelID = URL(fileURLWithPath: model).lastPathComponent
    try await runServe(modelID: modelID, port: port)
case "chat":
    print("qwisp chat — not yet implemented (step 6)")
case "selftest":
    print(await runTokenizerSelftest(modelDir: model))
default:
    print("usage: qwisp [serve|chat|selftest]")
}
