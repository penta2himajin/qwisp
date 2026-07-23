# 19 — Prefill acceleration & determinism: external engine survey (2026-07)

Method: investigate workflow (5 leads, Sonnet; Opus plan), 111 adversarial verify
verdicts, 34 findings refuted — almost all refutations were inflated/misattributed
NUMBERS, not wrong mechanisms. Everything below survived verification or is marked
otherwise. Trigger: OpenCode trial showed first-sight prefill (35K new tok/turn at
70-150 tok/s deep-context) is the dominant UX pain; owner asked whether strict-L1
can adopt matrix-unit prefill and what the industry does.

## 1. The determinism landscape (the crux)

Industry default = trade determinism for speed:

- llama.cpp: maintainer explicitly REJECTS batch-size-invariant bit-exactness as a
  goal; deterministic-mode PR (fixed-tile matmul, fp32 accum, no split-K,
  batch-invariant RMSNorm, sequential MoE) is an UNMERGED draft.
  https://github.com/ggml-org/llama.cpp/pull/16016
- vLLM: "does not guarantee reproducibility by default, for the sake of
  performance"; VLLM_BATCH_INVARIANT=1 is opt-in and scoped to same-HW+same-version.
  https://docs.vllm.ai/en/latest/usage/reproducibility.html
- ExLlamaV2 (turboderp): philosophical rejection — quantized models have no "more
  correct" output; any canonical accumulation order is an arbitrary artifact.
  https://github.com/turboderp-org/exllamav2/issues/232
  (Inverted, this is exactly qwisp's differentiation: L1 makes that "arbitrary
  canonical" a product guarantee. Survey found NO engine certifying cross-
  implementation bit agreement; nobody occupies L1.)
- cuBLAS itself only guarantees bit-repro on same-arch + same-SM-count, never across
  streams/atomics modes — the ceiling all CUDA engines inherit.

Direct precedents that matrix-unit + order-stability CAN coexist:

- Thinking Machines "Defeating Nondeterminism": forward pass has NO atomics; the
  nondeterminism is SHAPE-DEPENDENT KERNEL DISPATCH. Fix = keep tensor cores, pin
  one kernel config for all shapes. Cost ≈ 20% GEMM, e2e 26s→42s in their vLLM demo.
  https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/
  → qwisp NOTE: that cost does not apply to us — prefill chunks are fixed
  (1024/2048), single device, exclusive GPU. The "variable shape" they had to kill
  never existed here.
- M1 AMX bit-exact GEMM (arXiv:2606.25426): hand-written matrix-unit kernel,
  bit-identical to Accelerate fp32 (max-abs-diff 0), llama.cpp prefill 291→420 tok/s
  (1.44x). CPU-side, but proves matrix-unit ≠ nondeterministic.
- In-repo: steelMoEGather (SeedlessFusedVerify.swift ~3935) already ships MLX steel
  gather_qmm_t (simdgroup_matrix) with fixed R=16 tiles, "Deterministic +
  tile-composition-invariant", default ON via QWISP_HYBRID_PREFILL; the canonical
  stream was re-canonicalized once already ("pre-hybrid canonical stream" comment,
  TellRuntime.swift). The door is half-walked.
- MLX has NO official determinism statement (only a third-party blog showing
  batch-invariance failure on the matrix-unit path; unverified).

## 2. Matrix-unit mechanics on Apple (verified against source)

- MLX dispatch: M=1 → simd_sum GEMV; M>1 → steel BlockMMA simdgroup_matrix<T,8,8>,
  fp32 accumulators (mma.h AccumType=float, static_assert), fixed K-loop order via
  simdgroup_multiply_accumulate. Quantized qmm = dequant tile to threadgroup mem →
  same BlockMMA. NAX path (MetalPerformancePrimitives 16x16 tensor ops) is gated to
  M5-class chips and TF32-gated for fp32 inputs.
  https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/quantized.cpp
- Apple M5 blog: prefill/TTFT up to ~4x vs M4 via Neural Accelerators; decode only
  +19-27% — the matrix-unit dividend is concentrated in prefill.
  https://machinelearning.apple.com/research/exploring-llms-mlx-m5
- llama.cpp Metal: mul_mv vs mul_mm split at ne11>8 (ne00>=64, Apple7+); MoE
  mul_mm_id engages matrix unit only at >=32 rows/expert; legacy simdgroup_float8x8
  vs Metal4 MPP matmul2d behind GGML_METAL_HAS_TENSOR.
  ggml/src/ggml-metal/ggml-metal-ops.cpp (~L2068-2420)
- Rigel (arXiv:2606.12765): MPP matmul2d only 1.05-1.21x over raw simdgroup_matrix
  on M4 Max; FP8 emulated (0.94x of FP16). Metal4 tensor API is not a step change.
- Quant-GEMM industry practice: prefill-scale M abandons fused quant kernels —
  ExLlamaV2 MAX_Q_GEMM_ROWS=32 → dequant+cuBLAS; EXL3 same (reconstruct path);
  Machete converges to FP16 parity at batch>=128. "Prefill-specialized quant kernel"
  effectively does not exist; the pattern is dequant-once → plain matrix-unit GEMM.
  This matches MLX qmm design; nothing unclaimed to harvest here.

## 3. Scheduling (maps onto the adaptive-matrix discussion)

- vLLM V1: ONE per-step token_budget; RUNNING (decode + in-flight prefill chunks)
  spend first, WAITING prefill gets the remainder. No mode switch exists anywhere in
  industry — "serial" is the idle-load degenerate case of continuous scheduling.
  vllm/v1/core/sched/scheduler.py (token_budget loop)
- Tuning: small budget (~2048) favors ITL/interactivity; large (>8192) favors TTFT.
  https://docs.vllm.ai/en/stable/configuration/optimization/
  → OpenCode is interactivity-bound: start at 2048 = our snapshot stride; boundary
  alignment is free.
- SGLang: chunk size by GPU-memory tier (2048 <20GB … 16384 >=160GB) — same
  philosophy as our device-tier calibration; add a column to it.
- Origin: SARATHI / Sarathi-Serve (OSDI'24) "piggyback decodes with chunked
  prefills". https://arxiv.org/abs/2308.16369
- TensorRT-LLM chunked context: memory motive O(L^2)→O(L·K) — strongest on the
  streaming (small-RAM) row.
- Prefix caching: vLLM APC block-hash / SGLang RadixAttention (prefix-match-aware
  scheduling) / LMCache (CPU/disk/S3 tiers, survives restarts) — our PrefixPersist +
  RAM tier is LMCache-shaped. NO engine documents a bit-identity guarantee for
  cache-restore-vs-recompute; PREFIXE2E is unique.

## 4. Exactness classification of prefill accelerations

| Technique | L1-compatible | Notes |
|---|---|---|
| Chunked prefill + token-budget scheduling | YES (numerics untouched) | Stage A |
| Prefix cache persistence/tiering | YES (gated by PREFIXE2E) | shipped (#89/#112/#117) |
| Matrix-unit prefill GEMM, fixed tiles | YES with refs re-canonicalization | half-shipped (hybrid) |
| Context-parallel ring attention | exact, but multi-GPU | N/A single Mac |
| Speculative prefill (SpecPrefill etc.) | NO — drops prompt tokens | bolt row only, if ever |
| Dynamic sparse prefill (MInference, DSA) | NO — approximation | bolt row dial; indexer becomes new bottleneck at very long ctx |
| ANE offload | impractical for LLM prefill | Apple's own LLM team chose GPU (bandwidth); whisper.cpp encoder 3x is the exception, not the rule |

## 5. Competitor flag

BaseRT (arXiv:2607.00501, Base Compute): from-scratch raw-Metal runtime,
simdgroup_matrix tiled-GEMM prefill + chunked prefill + online-softmax attention;
claims up to 1.78x prefill over MLX on Qwen3-30B-A3B 4-bit @ M4 Pro. Same arena as
qwisp (raw Metal × Qwen MoE × Apple Silicon), no determinism claims — L1 remains
unoccupied. Read before the next positioning update.

## 6. Refuted-citation warnings (do not re-quote)

- FA3 real numbers: 740 TFLOPS FP16 / 75% util / ~1.2 PFLOPS FP8 (NOT 840/85%/1.3);
  the "30K RAG 8s→2s" example appears in neither cited source.
- vLLM APC "32% at prefix-ratio 0.1→0.9": fabricated citation.
- LMCache "3-10x": unsupported by any of its cited pages.
- "indexer consumes 81% of prefill at 200K": misattributed between vLLM/lmsys posts.
- Superlinear-attention 3.3-8.6x @65K: not in the cited paper.
- yage.ai MLX-vs-llama.cpp prompt-processing numbers: page contains no such bench.

## 7. Implications / next probes

1. `prefill-stage-profile` + `long-context-decay`: decompose today's 341→65 tok/s
   decay (0→50K) into GDN/attn/MoE shares, and measure the REMAINING matrix-unit gap
   post-hybrid (the 1.38x figure predates the steel hybrid). If attention dominates
   the decay, matrix-unit attention prefill flattens the decay slope, worth more
   than the flat 1.38x suggests.
2. Matrix-unit canonicalization has no theoretical determinism obstacle (fixed-shape
   discipline + fp32 accum; precedents above). Process = the hybrid-prefill playbook:
   fixed-tile steel swap → regenerate refs from raw greedy → RAWTESTS/PREFIXE2E/
   fidelity matrix.
3. Stage A (token-budget interleaved admission on the lane scheduler) is the
   industry-standard shape; survey adds parameter defaults (budget 2048, tier column
   for chunk size, prefix-aware lane assignment via maxCommonPrefix).
