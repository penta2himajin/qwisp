import Foundation
import Hummingbird
import Tokenizers
import Hub
import QwispCore

// qwisp — product CLI + OpenAI-compatible server (productization step 5).
// Scaffold: proves the dependency graph (Hummingbird + swift-transformers) resolves
// and links. `serve` / `chat` are wired in follow-up sub-steps.

let qwispConfig = Config.load(path: Config.defaultPath)
let model = Config.resolveModel(env: ProcessInfo.processInfo.environment, config: qwispConfig, default: Config.defaultModel)

let helpText = """
qwisp \(Config.version) — single-model local inference for Qwen3.6-35B-A3B on Apple Silicon

usage: qwisp <command> [options]

  serve                  OpenAI-compatible server on :8080 (QWISP_PORT / config)
  chat [opts] <prompt>   one-shot chat; reads stdin if no prompt is given
      --max-tokens N     cap generation (default: until EOS / context)
      --lossless         force strict bit-exact mode (streaming tiers default to bolt)
      --no-thinking      skip the <think> phase (enable_thinking=false; answers start immediately)
  pull [hf-repo-id]      download a checkpoint (default: Qwen3.6-35B-A3B MTPLX) + write config
  config [--defaults]    show effective settings / the full default set
  benchtest              community benchmark → markdown report + one-click submit URL
  simulate <N>gb         emulate a smaller-RAM Mac on this one (GPU+RAM ballast; ^C to
                         release; run benchtest in another terminal). Expert knobs:
                         --gpu-gb X / --ram-gb Y
  version                print the version

environment:
  QWISP_MODEL   model directory      QWISP_PORT      server port
  QWISP_LOSSLESS=1                   force strict on every tier
  chat sampling (default greedy): QWISP_TEMP QWISP_TOPP QWISP_SEED
                                  QWISP_FREQPEN QWISP_PRESPEN QWISP_LOGIT_BIAS="tok:bias,…"
  note: temperature > 0 on a streaming tier (<32GB) decodes via strict — bolt is greedy-only
  QWISP_UPDATE_CHECK=0               disable the update notice (a single GET to GitHub
                                     releases/latest, ≤once/day on chat/serve; no payload)
  QWISP_BATCH=<B>                    serve only: continuous batching with B slots (multi-user
                                     aggregate throughput; resident ≥32GB, greedy-only, not
                                     bit-exact with single-stream — opt-in)
  QWISP_LANES=<B>                    serve only: lane batching with B raw-engine lanes
                                     (parallel sub-agent fan-out; resident ≥32GB, greedy-only,
                                     BIT-EXACT with single-stream — opt-in). Per-lane context
                                     capped by QWISP_LANE_CTX (default 16384)

first run: `qwisp pull` downloads ~20GB. The first chat/serve request loads the model, and
on <32GB machines runs a one-time bolt calibration (a few minutes; progress goes to stderr).
"""

let args = Array(CommandLine.arguments.dropFirst())
if args.isEmpty || args.contains("--help") || args.contains("-h") || args.first == "help" {
    print(helpText)
    exit(0)
}
switch args.first {
case "serve":
    let port = Config.resolvePort(env: ProcessInfo.processInfo.environment, config: qwispConfig, default: Config.defaultPort)
    // A daemon has no TTY — never prompt. Missing model → fail fast with the hint.
    if ProcessInfo.processInfo.environment["QWISP_FAKE"] != "1" {
        if !ModelStore.isModel(model) {
            FileHandle.standardError.write(Data((ModelStore.missingModelHint + "\n").utf8)); exit(1)
        }
        ModelStore.requireSupported(model)
    }
    UpdateCheck.noticeInBackground()
    let modelID = URL(fileURLWithPath: model).lastPathComponent
    let tok = try await QwispTokenizer(modelDir: model)
    let backend: any LLMBackend
    var batchMode = false
    if ProcessInfo.processInfo.environment["QWISP_FAKE"] == "1" {
        print("[qwisp serve] FAKE backend (no engine load) — wire-format testing only")
        backend = FakeBackend(script: tok.encode("Hello! This is qwisp with a fake backend for wire-format testing."))
    } else if let ls = Int(ProcessInfo.processInfo.environment["QWISP_LANES"] ?? ""), ls >= 2 {
        // Lane batching (parallel sub-agent fan-out, opt-in): raw-engine lanes on the
        // continuous scheduler — bit-exact with the default serialize path, resident
        // tier only, greedy only. Per-lane context capped by QWISP_LANE_CTX (16384).
        print("[qwisp serve] lane batching: B=\(ls) lanes (raw engine, greedy, bit-exact; requests batch instead of queueing)")
        backend = try LaneBackend(modelDir: model, slots: ls)
        batchMode = true
    } else if let bs = Int(ProcessInfo.processInfo.environment["QWISP_BATCH"] ?? ""), bs >= 2 {
        // Continuous batching (issue #6, opt-in): multi-user aggregate-throughput mode.
        // Resident tier only, greedy only, and NOT bit-exact with single-stream decode
        // (batch near-tie flips) — hence never the default.
        print("[qwisp serve] continuous batching: B=\(bs) slots (resident MLX path, greedy; requests batch instead of queueing)")
        print("[qwisp serve] NOTE: batching targets multi-user cold-prompt throughput; for agentic harnesses (OpenCode etc.) the DEFAULT mode is faster (measured 2.6x) and bit-exact — see issue #121")
        backend = try BatchBackend(modelDir: model, slots: bs)
        batchMode = true
    } else {
        print("[qwisp serve] loading Seedless engine (loads the model) …")
        let sb = try SeedlessBackend(modelDir: model)
        // --lossless > env QWISP_LOSSLESS > config "lossless" > false. Streaming tiers
        // (<32GB) otherwise default to bolt (near-lossless).
        sb.losslessForced = args.contains("--lossless")
            || Config.resolveLossless(env: ProcessInfo.processInfo.environment, config: qwispConfig)
        backend = sb
    }
    let engine = QwispEngine(tokenizer: tok, backend: backend, modelID: modelID, serialize: !batchMode)
    try await runServe(engine: engine, modelID: modelID, port: port)
case "chat":
    // `--max-tokens N` | `--max-tokens=N` (matches mlx-lm); the rest of the args are the prompt.
    // Default -1 = generate until EOS / context (mlx-lm / llama.cpp semantics); N caps it.
    var rest = Array(args.dropFirst())
    var maxTokens = -1
    var chatLossless = Config.resolveLossless(env: ProcessInfo.processInfo.environment, config: qwispConfig)
    if let i = rest.firstIndex(of: "--lossless") { chatLossless = true; rest.remove(at: i) }
    var noThinking = false   // issue #77: enable_thinking=false via the chat template
    if let i = rest.firstIndex(of: "--no-thinking") { noThinking = true; rest.remove(at: i) }
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
        print("usage: qwisp chat [--max-tokens N] [--lossless] <prompt>   (or pipe text via stdin)")
    } else {
        let env = ProcessInfo.processInfo.environment
        // Ensure a model is present; interactively offer to pull one if not (TTY only).
        let effModel: String
        if env["QWISP_FAKE"] == "1" {
            effModel = model
        } else if let m = await ModelStore.ensureModel(model, allowPrompt: true) {
            ModelStore.requireSupported(m)
            effModel = m
        } else {
            FileHandle.standardError.write(Data((ModelStore.missingModelHint + "\n").utf8)); exit(1)
        }
        UpdateCheck.noticeInBackground()
        let tok = try await QwispTokenizer(modelDir: effModel)
        let backend: any LLMBackend
        if env["QWISP_FAKE"] == "1" {
            backend = FakeBackend(script: tok.encode("(fake backend) hello from qwisp chat."))
        } else {
            // The load is ~20GB and used to be silent (issue #45) — say what the wait is.
            FileHandle.standardError.write(Data("[qwisp] loading model from \(URL(fileURLWithPath: effModel).lastPathComponent) …\n".utf8))
            let sb = try SeedlessBackend(modelDir: effModel)
            sb.losslessForced = chatLossless
            backend = sb
        }
        // Option B sampling knobs (the server uses the OpenAI API params instead).
        // maxTokens comes from the --max-tokens flag above (default -1 = until EOS/context).
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
                      frequencyPenalty: freqPen, presencePenalty: presPen, logitBias: bias,
                      noThinking: noThinking)
    }
case "benchtest":
    // Community benchmark (call-for-testers): env + tiered speed/stability, markdown to stdout.
    if !ModelStore.isModel(model) {
        FileHandle.standardError.write(Data((ModelStore.missingModelHint + "\n").utf8)); exit(1)
    }
    ModelStore.requireSupported(model)
    print(await runBenchtest(modelDir: model))
case "version", "--version", "-v":
    print(Config.version)   // stdout stays the bare version — release.sh 3b compares it
    await UpdateCheck.reportForVersionCommand()
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
case "simulate":
    // qwisp simulate <N>gb [--gpu-gb X] [--ram-gb Y] — small-RAM Mac emulation (issue #71)
    exit(Simulate.run(args: Array(args.dropFirst())))
case "updatetest":
    let (passed, total, log) = UpdateCheck.selfCheck()   // network-free version-compare check
    print(log.joined(separator: "\n") + (log.isEmpty ? "" : "\n") + "UPDATETEST \(passed)/\(total)")
    if passed != total { exit(1) }
case "configtest":
    let (passed, total, log) = Config.selfCheck()   // model/port resolution + config load (no GPU, no model)
    print(log.joined(separator: "\n") + "\nCONFIGTEST \(passed)/\(total)")
    if passed != total { exit(1) }
case "pull":
    // qwisp pull [hf-repo-id] — download a checkpoint (default: Qwen3.6-35B-A3B MTPLX) and
    // point ~/.config/qwisp/config.json at it.
    let repo = args.dropFirst().first ?? ModelStore.defaultRepo
    // The error itself (with mirror/workaround hints) is printed inside pull() — a raw
    // uncaught-error crash dump on top of it helps nobody.
    do { _ = try await ModelStore.pull(repo: repo) } catch { exit(1) }
case "config":
    // qwisp config           — effective config with per-key provenance
    // qwisp config --defaults — full default set as JSON (for pinning explicitly)
    if args.dropFirst().first == "--defaults" {
        print(Config.defaultsJSON())
    } else {
        print(Config.effectiveReport(env: ProcessInfo.processInfo.environment, config: qwispConfig, path: Config.defaultPath))
    }
default:
    print(helpText)
}
