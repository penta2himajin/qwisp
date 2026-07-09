# `scripts/` — gates + reference oracle

Test harness and the Python **reference oracle** for validating the engine. None of this is on
the serving path — the product is `swift/`.

## Gate scripts (shell)

Run from the repo root after building the `qwisp`/`qwisp-poc` schemes:

| Script | Checks | Needs |
|---|---|---|
| `test_raw.sh` | `RAWTESTS 79/79` — engine unit tests (kernels, spec verify) | GPU; **no model** |
| `test_bench_batch.sh` | `BENCHBATCHTEST` — bench-harness routing fixture | neither (stub) |
| `test_tokenizer.sh` | `TOKTEST 3/3` — text↔ids round-trip + Qwen chat_template | model tokenizer files |
| `test_completion.sh` | `COMPTEST 4/4` — completion core vs a fake backend | model tokenizer files |

`bench.sh` / `bench_matrix.sh` / `bench_batch.sh` drive speed/fidelity benchmarks across regimes.

## Python reference oracle

`bench_refs.py`, `mtp_decode.py`, `bench_prompts.py`, `bench_correctness.py`, `bench_tokcmp.py` —
generate the canonical measurement refs (raw-greedy token streams) and bit-compare Swift output
against MLX. Refs land in the gitignored `refs/`.

**Environment**: needs an MLX-capable Python (numpy, safetensors, `mlx.core`, `mlx_lm`) — Homebrew
python3 will not do — plus the model. Regenerate refs with, e.g.:

```bash
PYTHONPATH=<repo> <mlx-python> -m scripts.bench_refs --ingest-swift <regime> /tmp/<regime>.toks
```

Refs are the canonical strict-fidelity baseline; regenerate them only from Swift raw-greedy output,
never from MLX or a bootstrap.
