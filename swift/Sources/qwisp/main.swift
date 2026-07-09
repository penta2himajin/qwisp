import Foundation
import Hummingbird
import Tokenizers
import Hub
import QwispCore

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
    let tok = try await QwispTokenizer(modelDir: model)
    let backend: any LLMBackend
    if ProcessInfo.processInfo.environment["QWISP_FAKE"] == "1" {
        print("[qwisp serve] FAKE backend (no engine load) — wire-format testing only")
        backend = FakeBackend(script: tok.encode("Hello! This is qwisp with a fake backend for wire-format testing."))
    } else {
        print("[qwisp serve] loading Seedless engine (loads the model) …")
        backend = try SeedlessBackend(modelDir: model)
    }
    let engine = QwispEngine(tokenizer: tok, backend: backend, modelID: modelID)
    try await runServe(engine: engine, modelID: modelID, port: port)
case "chat":
    let promptText = args.dropFirst().joined(separator: " ")
    let prompt = promptText.isEmpty ? (readLine(strippingNewline: true) ?? "") : promptText
    if prompt.isEmpty {
        print("usage: qwisp chat <prompt>   (or pipe text via stdin)")
    } else {
        let tok = try await QwispTokenizer(modelDir: model)
        let backend: any LLMBackend
        if ProcessInfo.processInfo.environment["QWISP_FAKE"] == "1" {
            backend = FakeBackend(script: tok.encode("(fake backend) hello from qwisp chat."))
        } else {
            backend = try SeedlessBackend(modelDir: model)
        }
        let maxTokens = Int(ProcessInfo.processInfo.environment["QWISP_GEN"] ?? "512") ?? 512
        await runChat(prompt: prompt, tokenizer: tok, backend: backend, maxTokens: maxTokens)
    }
case "selftest":
    print(await runTokenizerSelftest(modelDir: model))
case "comptest":
    print(await runCompletionSelftest(modelDir: model))
default:
    print("usage: qwisp [serve|chat|selftest]")
}
