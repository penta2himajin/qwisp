# Qwisp

Fast, optionally-lossless local inference for **Qwen3.6-35B-A3B (MoE)** on Apple Silicon,
served over an OpenAI-compatible HTTP API.

Qwisp is a single-model-specialised engine. Its decode core, **Seedless**, runs raw Metal
command buffers *outside* the MLX op-graph (persistent buffers, int32 readback) to beat MLX's
own decode path, while still using MLX as the numeric substrate (tensors, quantization, weight
loading). The runtime that drives it is **Tell** (speculative decode, scheduling). MoE experts
can stream from flash on RAM-constrained machines, so a 35B model reaches smaller Macs.

- **Runtime = Tell**, **engine = Seedless** (raw Metal; MLX is the substrate).
- **Lossless option**: the strict path reproduces the quantised greedy token stream bit-for-bit.
- **Drop-in**: OpenAI `/v1/chat/completions` (SSE) + `/v1/models`, plus a `qwisp chat` CLI.

## Requirements

- **Apple Silicon Mac**, macOS 14+.
- **Xcode with the Metal Toolchain** (the engine ships hand-written Metal kernels).
- **The model** on disk — a Qwen3.6-35B-A3B MTPLX checkpoint (~20 GB). Point `QWISP_MODEL` at
  its directory (must contain `config.json`, the `*.safetensors` shards, and `tokenizer.json` +
  `chat_template.jinja`).
- RAM sets the tier automatically: `<32 GB` streams experts from flash; `≥32 GB` keeps everything
  resident (fastest decode). 16 GB is the practical floor for interactive use.

## Quickstart

```bash
# Build (Release; Metal Toolchain required; ~minutes on first build)
cd swift && xcodebuild build -scheme qwisp -configuration Release \
  -destination 'platform=macOS' -derivedDataPath ./.xcode-build-rel -skipPackagePluginValidation
BIN=swift/.xcode-build-rel/Build/Products/Release/qwisp

export QWISP_MODEL=/path/to/Qwen3.6-35B-A3B-MTPLX-…    # the model directory

# OpenAI-compatible server (QWISP_PORT, default 8080)
"$BIN" serve

# CLI (in-process, streams to stdout)
"$BIN" chat "Explain MoE routing in two sentences."
```

Talk to the server with any OpenAI client:

```bash
curl -N http://127.0.0.1:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"qwisp","messages":[{"role":"user","content":"hi"}],"stream":true}'
```

## API

| Endpoint | Notes |
|---|---|
| `GET /v1/models` | Lists the loaded model (id = model folder name). |
| `POST /v1/chat/completions` | `stream:true` → SSE (`chat.completion.chunk`); otherwise a `chat.completion` JSON with `usage`. |

**Sampling is ignored — the engine is lossless greedy.** `temperature` / `top_p` / `n` are
accepted but have no effect (output is deterministic). When any are supplied, the response carries
an `x-qwisp-warning: sampling params ignored (greedy/lossless engine)` header, and `serve` logs the
same at startup. `tools`, `logprobs`, and `n > 1` are not supported.

## Architecture

```
OpenAI HTTP  (Hummingbird)          swift/Sources/qwisp/        ← server + CLI + tokenizer
   │  messages ⇄ token ids  (swift-transformers Tokenizer + Qwen chat_template)
   ▼
LLMBackend  ── SeedlessBackend      swift/Sources/QwispCore/    ← the shipped engine
   │  [Int] prompt → AsyncStream<Int>
   ▼
Tell (runtime)  ──▶  Seedless (engine, raw Metal)   ── MLX (tensors / quant / mmap loader)
```

The tokenizer and chat template are a **server-layer** concern; the backend operates purely on
token ids. Design rationale for the engine lives in `notes/`, which the source comments reference
by number (e.g. `notes/10`).

## Development

```bash
# Correctness / regression gates:
scripts/test_raw.sh          # RAWTESTS 79/79  — engine unit tests (GPU, no model needed)
scripts/test_bench_batch.sh  # BENCHBATCHTEST  — bench-harness fixture (no GPU)
scripts/test_tokenizer.sh    # TOKTEST 3/3     — tokenizer round-trip + chat template (needs model)
scripts/test_completion.sh   # COMPTEST 4/4    — completion core, fake backend (needs model tokenizer)
```

`scripts/` also holds the Python **reference oracle** (bit-exact comparison + benchmark
prompts/refs). It is used only to validate the engine, never on the serving path, and needs an
MLX-capable Python environment plus the model — see `scripts/README.md`.

The dev conventions (TDD, commit style, the lossless doctrine, the frozen shipped path) are in
[`AGENTS.md`](AGENTS.md).

## Layout

```
swift/            # the product — Swift package
  Sources/QwispCore/   Tell runtime + Seedless engine (+ locked engine tests)
  Sources/qwisp/       OpenAI server + `qwisp chat` CLI + tokenizer
  Sources/qwisp-poc/   bench/gate binary (RAWTESTS / bench harness)
scripts/            # gate scripts + Python reference oracle
notes/            # engine design rationale (referenced from source comments)
docs/             # process docs (handoff protocol, i18n policy)
```

## License

Apache-2.0. See [LICENSE](LICENSE).
