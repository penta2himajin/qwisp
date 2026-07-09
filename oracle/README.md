# `oracle/` — Python reference oracle

Bit-exact reference oracle for validating the Swift engine. Never on the serving path — the
product is `swift/`; the shell gates that call this live in `scripts/`.

`bench_refs.py`, `mtp_decode.py`, `bench_prompts.py`, `bench_correctness.py`, `bench_tokcmp.py` —
generate the canonical measurement refs (raw-greedy token streams) and bit-compare Swift output
against MLX. Refs land in the gitignored `refs/`.

**Environment**: needs an MLX-capable Python (numpy, safetensors, `mlx.core`, `mlx_lm`) — Homebrew
python3 will not do — plus the model. Regenerate refs with, e.g.:

```bash
PYTHONPATH=<repo> <mlx-python> -m oracle.bench_refs --ingest-swift <regime> /tmp/<regime>.toks
```

Refs are the canonical strict-fidelity baseline; regenerate them only from Swift raw-greedy output,
never from MLX or a bootstrap.
