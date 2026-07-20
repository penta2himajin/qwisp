# OpenCode parallel-agent reproduction

Faithfully reproduce the request pattern OpenCode emits when it fans out **parallel sub-agents**
(`task(background=true)` → N independent streaming completions), then measure qwisp's two paths
against it:

- **serialize (default)** — bit-identity to the single-stream strict reference, no batching speedup.
- **batch (`QWISP_BATCH`)** — throughput, but the MLX-greedy path (see below) may diverge from strict.

## 1. Capture ground truth (needs a running qwisp server)

```bash
# terminal A: capturing proxy in front of the server
CAPTURE=./oc-capture.jsonl node tools/opencode-repro/capture_proxy.mjs   # :8081 → :8080
```

Point OpenCode's qwisp provider `baseURL` at `http://127.0.0.1:8081/v1` (temporarily), then run a
task that fans out parallel sub-agents. Every request lands in `oc-capture.jsonl` with the full
body, arrival timing, and in-flight concurrency.

## 2. Replay + compare (needs the server; GPU)

```bash
R=tools/opencode-repro/replay_concurrent.mjs
# reference: one request at a time against the default (serialize) server
node $R run oc-capture.jsonl 127.0.0.1:8080 serial     ref-serial.json     --greedy
# reproduce the fan-out load against the default server (serialize path)
node $R run oc-capture.jsonl 127.0.0.1:8080 concurrent cc-serialize.json   --greedy
# ...and against a server started with QWISP_BATCH=<N> (batch path)
node $R run oc-capture.jsonl 127.0.0.1:8080 concurrent cc-batch.json       --greedy

node $R diff ref-serial.json cc-serialize.json   # expect: all streams identical, speedup ~1x
node $R diff ref-serial.json cc-batch.json        # measures batch divergence % + throughput speedup
```

`diff` reports how many streams are byte-identical to the serial reference and the wall-clock
speedup. The batch run's divergence count is the empirical answer to "does continuous batching
lose bit-exactness, and how often" on real OpenCode traffic (not just in theory).

Notes: the wire carries text deltas, not token ids, so identity is compared on concatenated text
(a strong proxy for a token flip). `--greedy` forces temperature 0 so the comparison is
deterministic even if OpenCode sampled. No dependencies — Node built-in `http` only.
