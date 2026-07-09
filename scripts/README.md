# `scripts/` — gate + benchmark scripts

Shell test/benchmark harness for the engine. Not on the serving path — the product is `swift/`.
The Python reference oracle these call lives in `oracle/`.

## Gate scripts

Run from the repo root after building the `qwisp` / `qwisp-poc` schemes:

| Script | Checks | Needs |
|---|---|---|
| `test_raw.sh` | `RAWTESTS 79/79` — engine unit tests (kernels, spec verify) | GPU; **no model** |
| `test_bench_batch.sh` | `BENCHBATCHTEST` — bench-harness routing fixture | neither (stub) |
| `test_tokenizer.sh` | `TOKTEST 3/3` — text↔ids round-trip + Qwen chat_template | model tokenizer files |
| `test_completion.sh` | `COMPTEST 4/4` — completion core vs a fake backend | model tokenizer files |

`bench.sh` / `bench_matrix.sh` / `bench_batch.sh` drive speed/fidelity benchmarks across regimes;
they call `oracle/` for ref generation and bit-compare.
