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

- **Apple Silicon Mac**, macOS 14+. The Homebrew binary is self-contained; **Xcode 26 / Swift 6.3**
  is needed only to build from source.
- **A model** (~20 GB) — a Qwen3.6-35B-A3B MTPLX checkpoint. `qwisp pull` downloads the default
  ([`Youssofal/Qwen3.6-35B-A3B-MTPLX-Optimized-Speed-FP16`](https://huggingface.co/Youssofal/Qwen3.6-35B-A3B-MTPLX-Optimized-Speed-FP16))
  and writes the config for you (see Quickstart). You can also point qwisp at an existing checkpoint
  via `QWISP_MODEL` or `~/.config/qwisp/config.json`.
- RAM sets the tier automatically: `<32 GB` streams experts from flash; `≥32 GB` keeps everything
  resident (fastest decode). 16 GB is the practical floor for interactive use.

## Quickstart

```bash
brew install penta2himajin/qwisp/qwisp     # Apple Silicon, macOS 14+

qwisp pull                                 # download the default model (~20 GB) + write config
                                           #   …or: qwisp pull <hf-repo-id>

qwisp chat "Explain MoE routing in two sentences."
qwisp chat --max-tokens 256 "…"            # cap length (default: until EOS / context)
QWISP_TEMP=0.7 qwisp chat "…"              # sampling knobs: QWISP_TEMP / QWISP_TOPP / QWISP_SEED
                                           #   note: temp>0 on <32GB decodes via strict (bolt is greedy-only)

qwisp serve                                # OpenAI-compatible server on :8080 (QWISP_PORT to change)
brew services start qwisp                  #   …or run it as a resident background service

qwisp config                               # show effective settings + where each value came from
```

First run: the first chat/serve request loads the model (~20 GB), and on <32 GB machines runs a
one-time bolt calibration — a few minutes at strict speed, with progress on stderr. Later
requests in the same process decode at full bolt speed.

Build from source instead (needs Xcode 26 / Swift 6.3):

```bash
cd swift && xcodebuild build -scheme qwisp -configuration Release \
  -destination 'platform=macOS' -derivedDataPath ./.xcode-build-rel -skipPackagePluginValidation
# binary at swift/.xcode-build-rel/Build/Products/Release/qwisp
```

Talk to the server with any OpenAI client:

```bash
curl -N http://127.0.0.1:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"qwisp","messages":[{"role":"user","content":"hi"}],"stream":true}'
```

## Performance

Greedy decode rates from `qwisp benchtest` (256–600 token generations, TTFT excluded).
**Provenance is explicit** — most numbers come from one dev machine; real community hardware
is what we're collecting now.

| tier | mode | decode | provenance |
|---|---|---|---|
| resident (≥32 GB) | strict (lossless) | 85–91 tok/s | dev machine — M1 Max (32c GPU) / 64 GB ([#36](https://github.com/penta2himajin/qwisp/issues/36)) |
| streaming 8 GB (C=64) | bolt (default) | 73–94 tok/s | dev machine, RAM-forced (`QWISP_DEVICE_RAM=8`) — not real 8 GB hardware |
| streaming 8 GB (C=64) | strict (`--lossless`) | 23–26 tok/s | same |
| slow-NAND MacBook (~1.5 GB/s reads) | bolt | ~71 tok/s | SSD-throttle approximation (`QWISP_SSD_THROTTLE_GBS=1.5`) |
| streaming 16 GB (C=128) | bolt (default) | 46–58 tok/s | **community** — M1 Pro (16c GPU) / 16 GB ([#41](https://github.com/penta2himajin/qwisp/issues/41)) |
| streaming 16 GB (C=128) | strict (`--lossless`) | 2.3 tok/s | same — real 16 GB memory pressure is far harsher than the throttle approximation predicted |
| your Mac | | | **[post a row → #38](https://github.com/penta2himajin/qwisp/issues/38)** |

Modes: **strict** reproduces the quantised greedy token stream bit-for-bit (default on resident;
`--lossless` forces it anywhere). **bolt** is the near-lossless streaming default — same
architecture, cached expert routing, much faster under flash streaming.

### Benchmark your Mac

```bash
qwisp benchtest        # ~2 min after the model is pulled; prints a markdown report
```

Deterministic, no accounts, no telemetry — the report is printed to your terminal and the last
line is a one-click URL that opens a pre-filled GitHub issue (you see exactly what you post).
8/16 GB Macs and 256 GB-SSD MacBooks are the rows we need most — including unstable
(`LOOPY`) results. Details: [call for testers](https://github.com/penta2himajin/qwisp/issues/38).

## API

| Endpoint | Notes |
|---|---|
| `GET /v1/models` | Lists the loaded model (id = model folder name). |
| `POST /v1/chat/completions` | `stream:true` → SSE (`chat.completion.chunk`); otherwise a `chat.completion` JSON with `usage`. Omit `max_tokens` (or send a negative value) to generate until EOS / context; the KV arena grows on demand from an 8K baseline. |

**Sampling is honored.** `temperature` (default `0` = greedy/lossless), `top_p`, `seed`,
`frequency_penalty`, `presence_penalty`, and `logit_bias` all take effect — via speculative
sampling on the GPU, at near-greedy speed. `temperature: 0` (the default) stays deterministic and
bit-exact to the strict greedy path. `n > 1` is ignored (single completion) and sets an
`x-qwisp-warning` header; `logprobs` is not supported.

**Function calling is supported.** Pass `tools` (OpenAI function specs); the model's calls come
back as `tool_calls` with `finish_reason: "tool_calls"`, and `role: "tool"` results feed back for
the next turn — so agentic clients work. Parameter values are coerced to JSON scalars best-effort;
streaming emits `tool_calls` once the call is complete (not token-incremental).

**Reasoning is separated.** Qwen3.6 thinks before answering; qwisp splits that out so `content`
is the clean answer and the thinking goes to `reasoning_content` (`delta.reasoning_content` when
streaming) — it never leaks into `content`. Give reasoning models a generous `max_tokens`: a small
cap can be spent entirely on thinking, leaving `content` empty.

**Throughput logging.** Each request logs one line to the server log —
`[qwisp] stream prompt=… gen=… ttft=…ms decode=… tok/s` — separating time-to-first-token (prefill)
from the decode rate. `tail -f "$(brew --prefix)/var/log/qwisp.log"` while a client drives qwisp to
measure real, in-harness performance.

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

`oracle/` holds the Python **reference oracle** (bit-exact comparison + benchmark prompts/refs).
It is used only to validate the engine, never on the serving path, and needs an MLX-capable Python
environment plus the model — see `oracle/README.md`.

The dev conventions (TDD, commit style, the lossless doctrine, the frozen shipped path) are in
[`AGENTS.md`](AGENTS.md).

## Layout

```
swift/            # the product — Swift package
  Sources/QwispCore/   Tell runtime + Seedless engine (+ locked engine tests)
  Sources/qwisp/       OpenAI server + `qwisp chat` CLI + tokenizer
  Sources/qwisp-poc/   bench/gate binary (RAWTESTS / bench harness)
scripts/          # shell gate + benchmark scripts
oracle/           # Python reference oracle (bit-compare; needs MLX python + model)
notes/            # engine design rationale (referenced from source comments)
docs/             # process docs (handoff protocol, i18n policy)
```

## Acknowledgements

Qwisp stands on:

- **[Qwen](https://github.com/QwenLM/Qwen)** (Alibaba) — the Qwen3.6-35B-A3B model this engine specialises in.
- **[Youssofal](https://huggingface.co/Youssofal)** — the MTPLX checkpoint qwisp loads by default.
- **[MLX](https://github.com/ml-explore/mlx)** / **[mlx-swift](https://github.com/ml-explore/mlx-swift)** (Apple) — the numeric substrate (tensors, quantization, mmap loader).
- **[Hummingbird](https://github.com/hummingbird-project/hummingbird)** — the async HTTP server.
- **[swift-transformers](https://github.com/huggingface/swift-transformers)** (Hugging Face) — tokenizer + Qwen chat template.
- User community - with data from community is helping me to improve this product a lot.

## License

Apache-2.0. See [LICENSE](LICENSE).
