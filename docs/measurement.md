# Measurement guide — which probe answers which question

qwisp is measurement-driven: every speed/lossless claim traces to a runner. This page maps
**symptom → the probe that answers it**. The full runner catalog with one-line descriptions:
`QWISP_RUN=list qwisp-poc stream` (no GPU needed for the listing itself).

All runners: `QWISP_RUN=<name> qwisp-poc stream`, model via `QWISP_MODEL` (defaults to the
Qwen3.6-35B MTPLX path). **The GPU is exclusive — never run two GPU processes (check
`ps aux | grep -E "qwisp serve|qwisp-poc"` and stop the brew service first).**

## Correctness (green = safe to commit)

| gate | command |
|---|---|
| engine kernels lossless (no model) | `scripts/test_raw.sh` → RAWTESTS N/N |
| completion core + pure logic (no GPU) | `scripts/test_completion.sh` → COMPTEST |
| bench harness fixture (no GPU) | `scripts/test_bench_batch.sh` |
| tokenizer round-trip (no GPU) | `scripts/test_tokenizer.sh` |
| prefix cache lossless (all tiers) | `QWISP_RUN=prefix-cache-e2e` (+`QWISP_PREFIX_E2E_C=<c>` streaming), `prefix-ram-e2e`, `prefix-persist-e2e`, `prefix-stable-e2e`, `prefix-bolt-e2e` |

A red strict-fidelity cell on longctx/shortnl after a fresh checkout is usually missing
`refs/` (gitignored) — regenerate from Swift raw greedy, never from MLX.

## "Decode got slow" / long-context regressions

1. **Real request, real path first**: `QWISP_SPEC_PROFILE=1` on `qwisp serve`, replay the
   offending request → one stderr line per generation:
   `[spec-profile] ... total = draft + chain + verify + rebuild + other`.
   - verify+rebuild dominates → speculation economics (see #119 / the spec gate; check
     accept in the serve log's `spec[...]` field). draft dominates → suffixDraft scan.
     chain ≈ total → the forward itself; go to 2.
2. **Forward physics by context**: `QWISP_RUN=long-context-decay` — per-stage
   (GDN/attn/MoE) GPU ms for prefill chunks by position AND M=1 decode steps by context.
   attn linear in ctx with flat GDN/MoE = fundamental full-attention O(N), not a bug.
3. **Is a wider verify worth it?** `QWISP_RUN=spec-width` (`QWISP_SPEC_CTX=<n>`) — stage
   ms by draft width M. `seqmt-m` for the r_M scaling view.
4. **Shipping-config tok/s + fidelity cells**: `scripts/bench_batch.sh` (strict/bolt ×
   4 regimes; needs refs).

Known verdicts before re-measuring: forward is near-optimal (M=1 @49K ≈ 36 tok/s
resident); the 48K collapse was speculation waste, fixed by the accept-driven gate
(`QWISP_SPEC_GATE=0` restores old behavior for A/B). Synthetic periodic prompts give
SuffixSpec unrealistically high accept — reproduce with natural text + repetitive tails.

## "TTFT / prefill is slow"

1. **Which kind of slow?** The serve log prints `prefill=<tok/s> ttft=<ms>` per request
   and `prefill a/b (x%) · r tok/s` progress lines. Low reuse on a warm server →
   prefix-cache miss (conversation switch? check `prompt=` sizes interleaving);
   cold-prefill rate itself low → kernel/position physics.
2. `QWISP_RUN=prefix-cache-speed` — TTFT cold vs cross-conversation vs intra.
3. `QWISP_RUN=prefill-probe` (chunk-size sweep: overhead- vs compute-bound),
   `prefill-breakdown` (wall vs GPU = dispatch/sync tax),
   `prefill-stage-profile` (per-stage + MLX matrix-unit reference),
   `hybrid-estimate` / `hybrid-prefill-bench` (steel-hybrid lever).
4. Position decay (348→66 tok/s over 0→40K) is fundamental full-attn O(N) — the
   mitigation is cache reuse (#117/#118), not kernel work.

## Server / harness behavior (OpenCode etc.)

- Wire-level ground truth + concurrent replay + identity/throughput diff:
  `tools/opencode-repro/` (capture proxy → replay serial/concurrent → diff). This is the
  acceptance gate for admission/batching work (#121).
- Per-request serve log fields: `prompt= prefill= gen= ttft= decode= spec[steps tok/step
  accept d0 rej]`.

## Kernel micro-benchmarks (no model)

`grouped-moe-bench`, `dense-tiled-bench`, `steel-route-bench`, `gqmm2-bench`,
`mlx-qmm-minv` (bit-stability of MLX kernel switches by M).

## Adding a new probe

One entry in the `runners` registry (`swift/Sources/qwisp-poc/main.swift`) with a one-line
desc — it appears in `QWISP_RUN=list` automatically. Put the implementation next to its
subject (e.g. prefix probes in PrefixCachePoC.swift). Add a row here only if it answers a
recurring symptom. Measure-first doctrine: a probe that answered a question is kept — it
is the regression tool for the next time the number moves.
