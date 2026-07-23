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
