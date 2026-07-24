# 20 — Probe spec: matrix-unit (MMA) attention-prefill kernel `sdpa_prefill_mma`

Phase 1 of WS-A (issue #137). ADDITIVE, flag-off probe — refs are NOT touched in
this phase. GO bar and the re-canonicalization decision live in the issue.

## Why (measured, notes/19 §8)

`long-context-decay`: the attention term owns the prefill decay — 0.42s→17.72s per
1024-chunk from pos 0→47K (76% of chunk time at depth) while steel-hybrid GDN/MoE
stay near-flat. The current kernel (`sdpa_rows`, SeedlessMetalForward.swift ~L4893)
is a GEMV-style decode kernel stretched to M rows: one threadgroup per (head, row),
per-position `simd_sum` dot products, ZERO K/V reuse across the M query rows.

Physical plausibility (doctrine gate): at pos 47K, chunk 1024, H=16, D=256,
10 full-attn layers: QK^T+PV ≈ 7.9 TFLOP/chunk → ≥7x headroom at 25% MMA
utilization (~2.3s vs 17.7s). Bandwidth floor with TQ=16 row tiles: ~60GB KV
re-reads/chunk ≈ 0.3s at >200GB/s. The 2x GO bar is conservative.

## Contract

New kernel `sdpa_prefill_mma` (SeedlessMetalForward, additive) + encode-only
variant (pattern: `encodeSdpaRows`, SeedlessFusedVerify.swift ~L1978), wired ONLY
into the prefill-scoped forward (`forwardRowsHybrid` — the same seam
QWISP_HYBRID_PREFILL uses; TellRuntime ~L96 "Decode/verify NEVER use this").

Semantics identical to `sdpa_rows`: q[M*H, D], k/v shared cache buffers
[KV, totalSeq, D] (strides passed in), out[M*H, D] half; per-row causal prefix
N = baseN + m; gqa_factor = H/KV = 8; D = 256 fixed; fp32 accumulation.

Flag: `QWISP_ATTN_MMA_PREFILL` — **default OFF this phase**. OFF ⇒ every code path
byte-identical to today (RAWTESTS 92/92 must stay green with the flag unset).

## Kernel design constraints (load-bearing — from notes/19 §1)

1. **Fixed tile geometry, always.** TQ×TK compile-time constants (start TQ=16,
   TK=32; tune freely but ONE configuration ships). No shape-adaptive dispatch —
   that is the batch-invariance trap every CUDA engine fell into.
2. **Chunk-composition invariance** (the property `sdpa_rows` has by construction
   and PREFIXE2E depends on): row m's output bits must depend only on (q_m, KV
   prefix of length N_m) — never on which chunk/tile it sits in or on M. Partial
   tiles are handled by PADDING (pad q rows point at content-irrelevant data,
   outputs discarded — steelMoEGather precedent, SeedlessFusedVerify ~L3935).
3. **Fixed accumulation order.** Sequential KV-tile loop (ascending positions),
   fixed k-fragment order inside each 8x8 MMA chain, fp32 accumulators
   (simdgroup_float8x8 / BlockMMA-style). Online softmax per ROW in fp32, update
   order = KV-tile order. No atomics; any cross-simdgroup combine has a fixed
   order independent of timing.
4. Causal mask: position j participates for row m iff j < baseN + m. Masked
   positions must have ZERO influence on the row's bits (not just ~0 weight —
   exclude them from max/sum/PV entirely).
5. Bit-exactness to `sdpa_rows` is NOT required (this becomes a new canonical on
   GO). Required instead: determinism + the invariances above + numeric sanity.

## Locked tests (RAWTESTS 92 → 96; WRITE-LOCKED total bumps to 96)

93. `mma_prefill_determinism`: same inputs twice → bit-identical out (random
    q/k/v, M=33, baseN=277 — deliberately non-multiples of tiles).
94. `mma_prefill_composition_invariant`: rows computed as one M=33 call vs split
    calls (M=16 at baseN, M=17 at baseN+16) → per-row bit-identical.
95. `mma_prefill_causal_and_pad`: (a) mutate k/v at positions ≥ N_m → row m bits
    unchanged; (b) fill pad-row q slots with NaN/garbage → real rows unchanged.
96. `mma_prefill_reference_sanity`: vs float64 CPU reference on random inputs
    (M=8, N≤512): max |Δ| ≤ 2e-2 on half outputs AND argmax over a random
    [D→vocab-proxy] projection agrees ≥ 99% of rows (guards sign/off-by-one bugs
    that tolerance alone misses).

Tests call the kernel DIRECTLY (unit level, like existing sdpa_rows tests); no
model needed. Flag-off integration invariance is covered by the existing 92.

## Bench verification (driver runs after GREEN, GPU-exclusive)

`QWISP_RUN=long-context-decay` with `QWISP_ATTN_MMA_PREFILL=1` vs unset:
attn_ms at pos 47104 (baseline 17,718ms). **GO ≥ 2x** on the attention term;
record the full table in this file's addendum.

## Prohibitions (this phase)

- Do NOT modify `sdpa_rows`, the verify path, decode path, or any existing kernel.
- Do NOT regenerate refs/ or flip any default.
- Do NOT weaken/skip existing WRITE-LOCKED tests (total goes 92→96 by ADDITION).

---

## VERDICT (2026-07-25): NO-GO on M1-class hardware — phase 2 does not proceed

Correctness fully landed (RAWTESTS 96/96: determinism / chunk-composition
invariance / causal+pad zero-influence / f64 reference sanity). Performance
failed the bar, twice, on the only valid instrument (same-session paired 24K
decay A/B, scripts/bench_decay_ab.sh):

| pos | run1 OFF/ON | run2 OFF/ON | speedup |
|---|---|---|---|
| 4096 | 1885 / 7632 | 1305 / 6206 | 0.21-0.25x |
| 8192 | 2781 / 13778 | 2587 / 10708 | 0.20-0.24x |
| 16384 | 6676 / 14528 | 4421 / 15583 | 0.28-0.46x |

Mechanism (control-bounded, kernel doc header has details): the kernel is
BIMODAL — a fast mode (~0.011 ms/pos, the source of the misleading isolated
502ms probe) appears only on repeated identical dispatches; the encode path
always runs the deterministic slow mode (~0.13 ms/pos/layer). Pure MMA in the
identical grid sustains 10.1 TFLOPS; adding TG memory, barriers, and device
streams keeps 9-10; only the full kernel's dependency-chained per-lane
fragment gathers collapse it to ~½ core-equivalent. No structural variant
(register blocking / tile sizes / load vectorization / residency) recovered
it. sdpa_rows' 1024-wide coalesced GEMV sustains ~1 TFLOPS deterministically
— that shape wins this op on M1-class.

Measurement doctrine bought by this phase:
1. Isolated kernel probes on repeated identical dispatches are INVALID for
   this class of kernel (bimodality; SLC flattery). Only same-session paired
   in-situ A/B counts.
2. Cross-day decay comparisons are void (~45% drift on untouched stages).
3. Apple-GPU MMA in latency-bound contexts is fragment-gather-limited;
   residency/blocking cannot rescue it. Matches Rigel's finding (notes/19 §2)
   that Metal4 MPP is only 1.05-1.21x over simdgroup_matrix.

Door left open: M5-class NAX (16x16 tensor ops, different feed path) may
change the verdict — the kernel + locked tests are preserved flag-off as the
re-entry point. BaseRT's claimed simdgroup_matrix prefill win (notes/19 §5)
is on M4 Pro with different shapes; unverified, not evidence against this
measurement.

Consequence: the prefill-decay attack shifts to WS-B (scheduler: hide the
cliff) + the already-shipped cache tiers (never pay twice) + harness-side
ingestion reduction. The attention term itself stays on sdpa_rows.
