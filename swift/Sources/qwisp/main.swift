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
    // `--max-tokens N` | `--max-tokens=N` (matches mlx-lm); the rest of the args are the prompt.
    // Default -1 = generate until EOS / context (mlx-lm / llama.cpp semantics); N caps it.
    var rest = Array(args.dropFirst())
    var maxTokens = -1
    if let i = rest.firstIndex(where: { $0 == "--max-tokens" || $0.hasPrefix("--max-tokens=") }) {
        let flag = rest[i]
        if let eq = flag.firstIndex(of: "=") {
            maxTokens = Int(flag[flag.index(after: eq)...]) ?? maxTokens
            rest.remove(at: i)
        } else if i + 1 < rest.count, let v = Int(rest[i + 1]) {
            maxTokens = v
            rest.removeSubrange(i...(i + 1))
        } else {
            rest.remove(at: i)   // dangling flag → ignore, fall back to default
        }
    }
    let promptText = rest.joined(separator: " ")
    let prompt = promptText.isEmpty ? (readLine(strippingNewline: true) ?? "") : promptText
    if prompt.isEmpty {
        print("usage: qwisp chat [--max-tokens N] <prompt>   (or pipe text via stdin)")
    } else {
        let tok = try await QwispTokenizer(modelDir: model)
        let backend: any LLMBackend
        if ProcessInfo.processInfo.environment["QWISP_FAKE"] == "1" {
            backend = FakeBackend(script: tok.encode("(fake backend) hello from qwisp chat."))
        } else {
            backend = try SeedlessBackend(modelDir: model)
        }
        // Option B sampling knobs (the server uses the OpenAI API params instead).
        // maxTokens comes from the --max-tokens flag above (default -1 = until EOS/context).
        let env = ProcessInfo.processInfo.environment
        let temp = Double(env["QWISP_TEMP"] ?? "0") ?? 0
        let topP = Double(env["QWISP_TOPP"] ?? "1") ?? 1
        let seed = UInt64(env["QWISP_SEED"] ?? "0") ?? 0
        let freqPen = Double(env["QWISP_FREQPEN"] ?? "0") ?? 0
        let presPen = Double(env["QWISP_PRESPEN"] ?? "0") ?? 0
        var bias: [Int: Double] = [:]   // QWISP_LOGIT_BIAS="tok:bias,tok:bias"
        for pair in (env["QWISP_LOGIT_BIAS"] ?? "").split(separator: ",") {
            let kv = pair.split(separator: ":"); if kv.count == 2, let t = Int(kv[0]), let b = Double(kv[1]) { bias[t] = b }
        }
        await runChat(prompt: prompt, tokenizer: tok, backend: backend, maxTokens: maxTokens,
                      temperature: temp, topP: topP, seed: seed,
                      frequencyPenalty: freqPen, presencePenalty: presPen, logitBias: bias)
    }
case "selftest":
    print(await runTokenizerSelftest(modelDir: model))
case "comptest":
    print(await runCompletionSelftest(modelDir: model))
case "sampletest":
    let (passed, total, log) = Sampler.selfCheck()   // GPU-free sampling-math check
    print(log.joined(separator: "\n") + "\nSAMPLETEST \(passed)/\(total)")
    if passed != total { exit(1) }
case "gpusampletest":
    let (passed, total, log) = SamplerGPU.distributionSelfCheck()   // GPU kernel vs analytic softmax (no model)
    print(log.joined(separator: "\n") + "\nGPUSAMPLETEST \(passed)/\(total)")
    if passed != total { exit(1) }
default:
    print("usage: qwisp [serve|chat|selftest|comptest|sampletest]")
}
