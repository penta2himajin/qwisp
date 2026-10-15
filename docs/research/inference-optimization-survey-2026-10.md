# Inference optimizations relevant to Qwisp (2026-10)

Status: research survey; no implementation decision  
Date: 2026-10-15  
Qwisp baseline: `05e745b` (branch `claude/m3pro-prefill-profile`; the non-strict fast prefill tier
landed on the same branch, see [Measured follow-ups](#measured-follow-ups-2026-10-15-m3-pro))

## 日本語サマリ

- **M3 Pro の strict prefill が遅い最有力仮説（コード読解、未測定）**: qwisp が vendored している MLX 0.31.1 では、
  `gather_qmm` の matrix-unit 経路と GEMV 経路を分ける閾値が chip ごとに違う。M3 Pro（`applegpu_g15s`）では
  MoE の形状で閾値が 18 行になる。strict の `steelMoEGather` が渡す R=16 行 tile はこれを下回るので、
  matrix unit ではなく `gather_qmv`（GEMV）で計算されているはずである。M1 系の 's'/'c' 版（閾値 14）なら
  matrix-unit 経路に乗る。最初の一手は 1 回の profile でこれを確かめること（P0）。仮説が正しければ、
  strict canonical stream が chip に依存していることにもなる。
- **MoE grouped GEMM**: 2026 年の MLX（#4572、0.32.3）と llama.cpp（#28301）は、どちらも「1 tile = 1 expert、
  少ない行数の tile」という形に収束した。qwisp の R=16 設計と同じ方向である。比較対象の mlx-lm 約 600 tok/s は、
  qwisp より新しい MLX で測った値の可能性があり、version を揃えて測り直す必要がある。
- **GatedDeltaNet**: strict と両立する高速化は mlx-lm #1559 の packed sequential kernel だけだった
  （Qwen3.6 形状で 1 層あたり 1.8–2.0x、M5 Max、reduction tree を source で固定して bit 一致を保証）。
  chunked（WY）形式は llama.cpp #29353 で top-1 一致率 98.8% と bit-exact ではなく、使えるのは非 strict の FastPrefill だけ。
- **Attention（head_dim 256）**: MLX 自身も pre-M5 では fused SDPA を使わず unfused にフォールバックしている
  （例外は qL==kL≥2048 のみ）。notes/20 の NO-GO と整合する。2x の改善が出ているのは M5 の NAX
  （llama.cpp #29570）だけである。
- **Speculative decoding**: 同系統の 35B-A3B で MTP は M4 Pro 上 1.04x にとどまった（mlx-lm #990）。MoE では
  verify の行数が増えると expert の和集合が広がり、verify コストが上がるためである。次の一手としては、
  SCORE gate の定数 r を実測した verify コスト曲線に置き換えるのがよい（Cascade/EVICT 型。lossless で学習不要）。
- **Determinism**: datacenter GPU では batch invariance の実コストが 24–74% と報告されている。llama.cpp と
  mlx-lm は、Qwen 系で spec 出力と greedy 出力が一致しないことを「仕様」または未解決として扱っている。
  L1 を保証するエンジンは依然として無い。ただし macOS 27 + M3 世代で、`MathMode.safe` の Metal compiler が
  誤った行を返す報告がある（mlx #4603）。qwisp も `.safe` を使っているので確認が必要。

## Question

Which 2025–2026 developments in MoE prefill, linear-attention and long-context attention kernels,
speculative decoding, deterministic inference, and Apple-Silicon engines should change what Qwisp
does next, and which of them are compatible with the strict L1 contract versus only the non-strict
fast-prefill or Bolt tiers?

This survey follows [notes/19](../../notes/19-prefill-survey-2026-07.md) (July 2026). It does not
repeat that note. It records what is new, what notes/19 missed, and where newer primary evidence
confirms or refutes its claims (see [Corrections to notes/19](#corrections-to-notes19)).

External numbers are author-reported and were not reproduced in Qwisp. Throughput numbers are not
comparable across chips, quantization formats, MLX versions, or fidelity contracts. Many of the
strongest 2026 results are M5-only (Neural Accelerators, "NAX"). The target machine is an M3 Pro and
does not have them.

## Executive result

1. **The largest prefill lever may be a dispatch threshold, not a new kernel.** On the M3 Pro,
   Qwisp's strict MoE tile shape appears to fall *below* MLX's matrix-unit threshold, so the
   "steel" MoE path probably runs as a GEMV. This is a code-reading result and must be confirmed by
   one profile before anything else (see §1.1). If it is confirmed, it also means the strict
   canonical prefill stream depends on the chip, which matters for L1.
2. **Upstream converged on Qwisp's MoE design.** MLX 0.32.3 (#4572) and llama.cpp (#28301, #30047)
   moved to expert-aligned, few-row tiles. Qwisp vendors MLX 0.31.1, so the mlx-lm comparison point
   probably uses newer kernels than Qwisp's fast-prefill probe.
3. **GatedDeltaNet has exactly one strict-compatible speedup in the literature:** a packed
   sequential recurrence with a source-level reduction contract (mlx-lm #1559). Every chunked/WY
   kernel changes greedy tokens (llama.cpp #29353 measures 98.8% top-1 agreement).
4. **notes/20's attention NO-GO is corroborated.** MLX keeps pre-M5 head_dim-256 prefill on the
   unfused path. The only 2x wins are on M5 tensor units.
5. **MTP is not a free win on this model family.** On an M4 Pro, MTP gives 1.04x on the 35B-A3B
   MoE despite 85% acceptance, while the dense 27B gets 1.57x (mlx-lm #990). MoE-specific work
   (Cascade, EVICT, EcoSpec) gets its gains from *verify-cost-aware gating*, which fits SuffixSpec
   without training.
6. **L1 is still unoccupied.** No surveyed engine claims that speculative or batched greedy output is
   bit-identical to plain greedy output, and llama.cpp's maintainer calls the divergence on Qwen3.6
   MTP "expected". A new macOS 27 Metal-compiler bug report is a live risk to Qwisp's strict refs.

## Qwisp baseline

Facts read from this checkout (not new measurements):

- **Vendored MLX.** `mlx-swift` 0.31.4 (`dc43e62d`), whose embedded MLX core reports version
  **0.31.1** (`Source/Cmlx/mlx/mlx/version.h`). FastPrefill and the strict hybrid path both use these
  kernels. None of the 2026-08/09 MLX MoE, GDN, or SDPA changes below are present.
- **Strict MoE prefill.** [`steelMoEGather`](../../swift/Sources/QwispCore/SeedlessFusedVerify.swift)
  works as follows:
  1. It reads the routed indices on the CPU (the route must be flushed).
  2. It builds fixed R=16 tiles per expert on the CPU.
  3. It copies the post-norm buffer into an `MLXArray` through a host `Array`.
  4. It calls `gatherQuantizedMatmul` on `x = [Gt, 16, H]` (so the kernel sees M=16 per batch
     element) and `eval`s the result.
  5. It copies the result back into the Seedless buffer.

  The FastPrefill header records "4-5 CPU↔GPU syncs per layer" for the strict hybrid path. The R=16
  path is in `HEAD`. The working tree also has an uncommitted `QWISP_MOE_SORTED=1` experiment
  (`steelMoEGatherSorted`) that uses mlx-lm's `x=[M·k,1,H]` layout. That layout has M=1, so it
  reaches `gather_qmm_rhs` once B/E ≥ 4.
- **Non-strict FastPrefill** ([`FastPrefill.swift`](../../swift/Sources/QwispCore/FastPrefill.swift),
  uncommitted) evaluates each 2048-token chunk as one MLX graph, using mlx-lm's sorted
  `x=[M·k, 1, H]` layout, then exports KV/GDN state to Seedless.
- **Measured on the M3 Pro** (from the brief): decode about 55 tok/s; strict prefill 110–200 tok/s
  against mlx-lm about 600 tok/s; MoE `gather_qmm` is 60–85% of prefill time. On a different
  machine, notes/19 §8 measured GDN at 37% of a chunk at depth 0 and attention at 76% at depth 47K.
- **Speculation.** SuffixSpec with a SCORE gate that verifies only if `SCORE > r·D`, with a
  *constant* per-row cost `r` (`QWISP_SPEC_GATE_ROWCOST`; see `Tell.swift`), plus SpecDec++-style
  truncation and adaptive K. Prior internal verdicts, which this survey does not override without
  new evidence:
  - DFlash was parked as structurally NO-GO at M=1 greedy (commit `b2d9b8b`).
  - Strict spec-in-batch is "a wash" (handoff 2026-07-22).
  - MTP-D1 acceptance was 0.51–0.83 (notes/15).

## 1. MoE prefill: grouped expert GEMM on Apple GPUs

### 1.1 Local finding: the strict R=16 tile is below MLX's matrix-unit threshold on M3 Pro

`GatherQMM::eval_gpu` in the vendored MLX 0.31.1 (`mlx/backend/metal/quantized.cpp`) dispatches as
follows:

1. If `M == 1 && B >= 16 && right_sorted && B/E >= 4`, it uses `gather_qmm_rhs`, the mlx-lm sorted
   path. Its non-NAX tile is `bm=16, bn=32, bk=32`.
2. Else, if `M >= vector_limit`, it uses `gather_qmm`, the steel `simdgroup_matrix` path with
   `bm=32, bn=32`.
3. Else, for transposed weights, it uses `gather_qmv`, the GEMV path.

`vector_limit = get_qmv_batch_limit(K, N, device)` depends on the GPU architecture string:

- arch gen 13/14 (M1/M2), non-'d' size class: **14** for `K, N ≤ 2048`.
- Every other non-'d' chip (including gen 15 = M3): **18** for `K, N ≤ 2048`.
- 'd' chips: 32.

On the target machine, `MTLDevice.architecture.name` returns **`applegpu_g15s`** (read on
2026-10-15; no GPU work dispatched). Both MoE projection shapes (gate/up `K=2048, N=512`; down
`K=512, N=2048`) are inside `≤ 2048`, so the limit is 18. `steelMoEGather` passes `x = [Gt, 16, H]`,
which is M=16 per tile. By this reading **the strict MoE prefill on the M3 Pro runs `gather_qmv`**.
That kernel re-reads and dequantizes the expert weights for each of the 16 rows, instead of using
the `simdgroup_matrix` path that the in-code comment ("gather_qmm_t matrix units, 2.24x @T=1024 vs
per-pair qmv") assumes.

The same arithmetic implies a second consequence: the **strict canonical prefill stream depends on
the chip**.

- For the same tile, M1/M2 non-'d' parts (limit 14) take the `gather_qmm` matrix-unit kernel.
  M3-and-later non-'d' parts (limit 18) and every 'd' part (limit 32) take `gather_qmv`. The two
  kernels round differently, so the "same" strict prefill can produce different bits on different
  chips.
- `steelQmm`'s "pad to M≥14" rule has the same gen-13/14 assumption, though the GDN projection shapes
  it serves have N > 2048 and limits of 10–12, so it appears safe there.

**Status: code reading only, not measured.** The confirmation is a single profile with the
existing `QWISP_PREFILL_PROF=1` marks (`moe.tileCPU`, `moe.xin`, `moe.mlx`, `moe.out`) on the M3 Pro, compared with the same chunk
forced to `R=32`. One run also needs a Metal capture or kernel-name log to show which kernel
executed. Do not act on this until that run exists.

### 1.2 Upstream MoE GEMM evolution (2025–2026)

| Change | What it does | Evidence (as published) | Chip |
|---|---|---|---|
| [MLX #2078](https://github.com/ml-explore/mlx/pull/2078) (2025-04) | Introduces `gather_qmm_rhs`, a sorted-rows batched kernel | Mixtral 8x7B, about 500 tokens: 189→590 tok/s | not stated |
| [MLX #4572](https://github.com/ml-explore/mlx/pull/4572) (merged 2026-09-28, v0.32.3) | Expert-aligned tile scheduler: each threadgroup holds rows of one expert. Changes both `quantized.h` (non-NAX) and `quantized_nax.h` | qwen3.5-397b-a17b affine: N=512 8.08→6.24 ms (1.30x), N=2048 13.55→11.42 ms (1.19x) | not stated |
| [MLX #4023](https://github.com/ml-explore/mlx/pull/4023) (2026-08) | Pick NAX BM from rows/expert | Base M5: at 16 rows/expert, 20.8%→29.5% of fp16 peak; Qwen3.6-35B-A3B 4-bit, 512-token chunks: 523.6→557.9 tok/s (author: from an earlier build) | M5 |
| [MLX #3918](https://github.com/ml-explore/mlx/pull/3918) (closed, unmerged) | 32x64 `gather_qmm_rhs` tile at ≥32 rows/expert | Qwen3.6-35B-A3B shapes, 4-bit, **gs=128**: +12.5–13.6% at 32 rows/expert, +19–22% at 64–128; 0.6–0.8x at ≤24. Author claims "elementwise bit-exact at every config" | M3 Max |
| [llama.cpp #28301](https://github.com/ggml-org/llama.cpp/pull/28301) (2026-09-11) | Split the 32-row `mul_mm_id` token tile into 16-row halves and skip the empty one | MUL_MAT_ID −40.8% time at 4 rows/expert, −22.6% at 16; pp512 +5.1% on a 256-expert top-8 35B-A3B at `-ub 256` | M5, simdgroup path |
| [llama.cpp #30047](https://github.com/ggml-org/llama.cpp/pull/30047) (open) | Few-row MMA `MUL_MAT_ID` | Qwen3.6-35B-A3B Q4_K expert shapes: gate/up 2.40x at 32 tokens, 1.00x at 256; regresses on M5 Max / M2 Ultra at 32–96 tokens | M3 Ultra |

Reading for Qwisp:

- **The arithmetic that matters is rows per expert.** A 2048-token chunk averages
  2048·8/256 = 64 rows/expert. A 1024-token chunk averages 32. A 512-token chunk averages 16, which
  is exactly where #4023 and #28301 show padded 32/64-row tiles wasting most of the matrix work.
  Qwisp's fixed R=16 tile is the right *size*. The open question is the kernel class it lands in
  (§1.1).
- **Bigger tiles at high rows/expert are a measured, plausibly bit-exact lever on an M3-class chip.**
  #3918 is the closest evidence to the target hardware, but it uses group size 128 and was closed
  without maintainer review of correctness. Qwisp would have to re-prove per-element invariance at
  gs=64. A tile-size change is a kernel-class change for the canonical stream, so refs must be
  regenerated.
- **The mlx-lm comparison point is a moving target.** Qwisp's ~600 tok/s mlx-lm reference was taken
  with "mlx-lm 0.32". The MLX core used by that Python environment was not recorded. If it is
  ≥0.32.3, it includes #4572, and the FastPrefill probe on vendored 0.31.1 is not an
  apples-to-apples replica. Record `mx.__version__` with every mlx-lm baseline.

### 1.3 Token permutation and fused gate+up

- **Gather inside the loader instead of materializing permuted activations.** vLLM's Triton
  `fused_moe_kernel` indexes the source row in its loader (`offs_token // top_k`) after
  [`moe_align_block_size`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/moe_align_block_size.py)
  pads each expert to the block size. oMLX's M5 kernel uses a `row_map` for the same purpose and
  reports that it avoids a "0.4-0.5 GB per layer at 8192-token chunks" replicated copy
  (`omlx/patches/m5_gather_qmm_nax.py`, oMLX `958974c1`). Qwisp's strict path materializes
  `xg = take(x, rowSrc)` plus host copies. Removing them does not change any dot-product order, so
  it is strict-compatible.
- **Fused gate+up is low-yield on Apple GPUs.** [mlx-lm #1676](https://github.com/ml-explore/mlx-lm/pull/1676)
  (M5 Pro) was bit-exact, but the author withdrew the speed claim: the Qwen3-30B-A3B delta "flips
  sign between +4.7% and -2.1%". [mlx-lm #1319](https://github.com/ml-explore/mlx-lm/pull/1319)
  (M1 Max) added 2.4 GB of persistent memory on OLMoE and regressed large-token prefill. BaseRT's
  fused gate/up+SiLU expert GEMM ([arXiv 2607.19438](https://arxiv.org/html/2607.19438v1)) runs on
  the M5 tensor path, and the paper does not isolate its contribution.
- **Metal 4 tensor ops do not change the picture on M3/M4.**
  [Rigel, arXiv 2606.12765](https://arxiv.org/abs/2606.12765) reverse-engineers `matmul2d` on an
  M4 Max: it runs on the shader cores, sustains 14.8 TFLOP/s fp16 at 2048³, and is only
  1.05–1.21x over `simdgroup_matrix`. The large gains require M5 Neural Accelerators. Apple
  reports a 3.52x TTFT speedup for Qwen3-30B-A3B 4-bit on M5 vs M4 at a 4096-token prompt
  ([Apple ML Research](https://machinelearning.apple.com/research/exploring-llms-mlx-m5), 24 GB
  MacBook Pros).

## 2. GatedDeltaNet prefill kernels

Two kernel families exist.

- **Sequential (recurrent) form.** The per-token state update is applied in token order. Qwisp's
  `gated_delta_step` (`_recurPipeline`), mlx-lm's `gated_delta_kernel` and llama.cpp's Metal
  `kernel_gated_delta_net_impl` all use it. Bits depend only on token order, so any prefill split
  composes exactly.
- **Chunked (WY / UT-transform) form.** Each chunk of C tokens is turned into intra-chunk matmuls, a
  triangular inverse, and a per-chunk state hop. Gated DeltaNet
  ([arXiv 2412.06464](https://arxiv.org/abs/2412.06464)), FLA's `chunk_gated_delta_rule` (C=64,
  [source](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/gated_delta_rule/chunk.py))
  and Kimi Delta Attention ([arXiv 2510.26692](https://arxiv.org/abs/2510.26692)) use it. The output
  depends on chunk boundaries.

| Work | Form | Evidence | Exactness |
|---|---|---|---|
| [mlx-lm #1559](https://github.com/ml-explore/mlx-lm/pull/1559) (merged 2026-08-27) | Packed sequential: 8 value rows/simdgroup, 4 lanes/row, 32 state elements/lane | **Qwen3.6-35B-A3B shapes** (Hk16 Hv32 Dk=Dv=128, T=2048), M5 Max: 2.54–2.56 → 1.34–1.38 ms per GDN layer; e2e prefill +6.4% (median of 6 A/B pairs). Reviewer repro, Qwen3-Next shapes: 1.93–2.04x at T=2048–8192 | Bitwise-pinned to an explicit shuffle_xor 1→2→4→8→16 tree comparator; `MathMode::Safe`; a canary test detects toolchains where `simd_sum` lowers differently |
| [MLX #4020](https://github.com/ml-explore/mlx/pull/4020) (merged 2026-09-15, v0.32.3) | `mx.fast.gated_delta_update`: sequential, simdgroup chunk-8, NAX chunk-16. Pre-M5 uses chunk-8 for T>8 | Speed only in images. A reviewer's plot reading (about 1.3–1.45x simdgroup, 1.6–2.2x NAX) is **unverified**. PPL identical (7.45) across kernels | Uses `metal::fast` exp/log; the NAX test needs `atol=1e-1`; a reviewer asked for a token-identity check |
| [mlx-lm #1241](https://github.com/ml-explore/mlx-lm/pull/1241) (closed) | Chunked port of FlashQLA | Closed because "the existing gated_delta_kernel … outperforms this approach for T ≥ 1024" | n/a |
| [llama.cpp #29353](https://github.com/ggml-org/llama.cpp/pull/29353) (merged 2026-10-14) | Chunked GDN, CUDA/HIP | pp +16–21% on Qwen3.5-35B-A3B (RTX 5090 / PRO 6000); +3–4% on GB10 with Qwen3.6-35B-A3B | **Top-token agreement 98.824%** vs recurrent; mean KLD 0.000463 |
| [llama.cpp Metal GDN](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/kernels/gated_delta_net.metal) | Sequential. A simplified variant is disabled with "no performance improvement" | — | Order-stable; [#28109](https://github.com/ggml-org/llama.cpp/pull/28109) (open) adds a test that a batched call is `memcmp`-equal to n single-token calls |
| oMLX `gated_delta_pipelined` | Sequential, 8 lanes/value row, 16-row TG, 12-token unrolled block | "3.0 ms vs 4.9 ms per layer call" at T=8191 on M5 Ultra (Qwen3.8 16/48 heads). FLA chunked WY was "accuracy-validated but slower than the stock kernel E2E" (`omlx/patches/qwen35_gdn_chunked.py`, oMLX `958974c1`) | Described as "the exact sequential recurrence" |

Reading for Qwisp:

- **The packed-lane layout of #1559 is the only strict-compatible GDN speedup found.** The test
  pattern also fits Qwisp's doctrine: write the reduction tree in source, compare with
  `array_equal`, and use a canary for compiler lowering. GDN was 1.33 s of a 3.6 s chunk at depth 0
  in notes/19 §8 (measured on the earlier resident machine), so a 1.8x GDN kernel would be worth
  more to Qwisp than the +6.4% it gave mlx-lm. That is an estimate. It requires a new canonical
  stream unless the packed kernel reproduces Qwisp's current reduction order bit-for-bit.
- **Chunked GDN is not strict and has weak evidence of winning at batch 1 on Apple GPUs.** Three
  independent Apple-side attempts (mlx-lm #1241, oMLX's FLA port, llama.cpp's Metal maintainers)
  found the sequential form competitive or faster. MLX #4020 is the exception, and its numbers are
  unverifiable in text. For FastPrefill (non-strict), adopting #4020 comes free with an MLX bump. It
  should be measured, not assumed.
- **Speculative rollback interacts with GDN.** oMLX snapshots `(conv_state, ssm_state)` after the
  confirmed prefix (`patches/mlx_lm_mtp/cache_rollback.py`). vLLM's replay-based rollback
  ([#61309](https://github.com/vllm-project/vllm/pull/61309), open) gives +8.6% at K=5 /
  concurrency 32 on an L40S, but 27 of 30 greedy outputs diverge at bf16 ties. Replay is
  L1-compatible only if the replay is the identical sequential kernel, which Qwisp's restore path
  already guarantees.

## 3. Long-context prefill attention on Metal

notes/20 measured a matrix-unit flash-attention prefill kernel at 0.20–0.46x of `sdpa_rows` on
M1-class hardware and closed WS-A. Upstream evidence since then is consistent with that verdict.

- **MLX keeps pre-M5 head_dim-256 prefill on the unfused path.** In vendored 0.31.1 the fused
  `sdpa_full` supports only D ∈ {64, 80, 128}. D=256 with qL > 8 falls back to unfused
  matmul→softmax→matmul (local source, `scaled_dot_product_attention.cpp`). On current MLX `main`,
  [PR #4505](https://github.com/ml-explore/mlx/pull/4505) (merged 2026-09-25) adds a non-NAX D256
  path that splits D across two simdgroups. In `use_fallback` on MLX `main` (`36969c09`,
  2026-10-14) that path is gated to non-NAX, non-fp32, causal, D=256, `qL ≥ 2048` **and
  `qL == kL`**. Every other qL > 8 D256 call stays unfused, under the source comment "Unfused path
  is faster for following shapes". The PR's numbers are mainly memory: M4 Pro 64 GB, Qwen3.8-27B, 16K prompt:
  115.8→125.5 tok/s prompt, peak 46.92→35.46 GB. Chunked prefill against an existing cache
  (qL < kL), which is Qwisp's deep-context case, still takes the unfused path there.
- **The 2x wins are M5-only.** [llama.cpp #29570](https://github.com/ggml-org/llama.cpp/pull/29570)
  (merged 2026-10-02) uses `mpp::tensor_ops::matmul2d` on an M5 Pro (20-core GPU): hs=256,
  kv=4096 went 5.21→10.52 TFLOPS, and Qwen3.8-27B Q8_0 pp512 at depth 32768 went
  174.3→252.9 tok/s. [MLX #3842](https://github.com/ml-explore/mlx/pull/3842) found the M5 D=256
  kernel limited by per-simdgroup accumulator bytes (22 vs 48 TF at bd=128) and fixed it by
  splitting D across simdgroups.
- **The one credible M1–M4 design is D-blocking with deliberate spilling.**
  [metal-flash-attention](https://github.com/philipturner/metal-flash-attention) states that at
  head dim 256 "none of the matrix blocks can fit into registers. Not even the accumulator can".
  It reports 86% (M1 architecture) and 82% (M3) forward-pass ALU utilization at D=256, using a third
  blocking dimension along D. The metric is instructions, not GEMM FLOPS, and it covers
  single-head, non-causal attention. Qwisp's NO-GO kernel blocked on Q/K tiles, not on D, so the
  design space is not exhausted. Re-entry is still P2: two upstream teams with more kernel budget
  chose not to ship it pre-M5.
- **oMLX** ships a head_dim-256 steel flash-attention prefill for Qwen3.6's 16/2 layout. Its stated
  reason is that the alternative tiled path is "~2x slower than even the stock unfused fallback at
  long context" (`omlx/patches/qwen35_fa256_attention.py`). No M3 numbers are published.

## 4. Speculative decoding for a hybrid MoE

### 4.1 Verify cost grows with rows on a sparse MoE

Under uniform routing, the expected number of distinct experts touched by t verify rows is
`N(t) = E·(1 − ((E−k)/E)^t)` (MoESD, [arXiv 2505.19645](https://arxiv.org/html/2505.19645v4)). For
Qwen3.6 (E=256, k=8) that is 8.0, 15.8, 23.3 and 30.5 experts for t = 1, 2, 3, 4. This is our own
calculation and an upper bound: real routing is correlated across adjacent tokens. At batch 1 the
routed-expert bytes per layer therefore come close to doubling with the first draft token. A dense
model reads the same weights for every row. Measured consequences:

| Source | Setting | Finding |
|---|---|---|
| [Cascade, arXiv 2506.20675](https://arxiv.org/html/2506.20675v1) | RTX 6000 Ada; Mixtral, Phi-3.5-MoE, OLMoE, DeepSeek-MoE, Qwen1.5-MoE; mostly n-gram drafting | At K=7, verify overhead is 2.3–3x on MoE vs ≤3% on dense LLaMA-3-8B. Speculation slowed 18 of 35 model-task pairs, by up to 1.5x. A utility gate (disable when token gain / verify cost < 1, then search K) caps slowdown at about 5% and gives +7–14% over static K |
| [EVICT, arXiv 2605.00342](https://arxiv.org/abs/2605.00342) | A100-80GB, batch 1, SGLang, EAGLE-3 trees | Truncates the draft tree using an offline-profiled verify-cost model. Up to 2.35x over AR; 1.21x average over EAGLE-3 on Qwen3-30B-A3B. Training-free and lossless |
| [EcoSpec, arXiv 2607.12696](https://arxiv.org/html/2607.12696) | 8×H200, batch 1 | Unique experts/layer: AR 8.0 vs EAGLE-3 23.7 (Qwen3-235B-A22B), vs MTP 31.4 (DeepSeek-V3.1). DeepSeek-V3.1 MTP gives only 1.10x over AR (EcoSpec 1.15x). EcoSpec needs a trained expert predictor |
| [mlx-lm #990](https://github.com/ml-explore/mlx-lm/pull/990) (open) | **M4 Pro, 4-bit, native Qwen MTP depth 1, temp 0** | Qwen3.5-35B-A3B: 85.3→87.9 tok/s (**1.04x**) at 85.2% acceptance. Dense Qwen3.6-27B: 15.7→24.6 tok/s (1.57x). Author: "On M4 Pro MoE speedup is marginal regardless of acceptance rate." Greedy MTP output matched plain generation |
| [MoE-Spec, arXiv 2602.16052](https://arxiv.org/html/2602.16052v1) | A100, batch 1 | Caps experts per layer during verify for +10–30% over EAGLE-3. **Lossy** (drops experts); not usable in strict mode |

Reading for Qwisp:

- **The SCORE gate's constant row cost was tuned on a dense model.** Its comment says
  `r=478‰ (dense)`. On this MoE the marginal cost of row t+1 falls as the expert union saturates,
  and it varies with residency and chunk state. Replacing the constant `r·D` with a per-device
  measured `cost(D)` table, profiled at startup in line with notes/02's "measure cost on device"
  policy, is the Cascade/EVICT idea applied to SuffixSpec. It changes only *whether and how much*
  to verify, so it is strict-compatible and needs no training. The expected gain is modest. Cascade
  reports +7–14% over static K, and Qwisp already has adaptive K and truncation.
- **MTP should stay parked on the serving path unless the M3 Pro ratio is different.** mlx-lm #990
  is the closest external measurement (same model family, Apple GPU) and shows 1.04x. Qwisp's own
  history agrees: strict spec-in-batch was "a wash" and DFlash was NO-GO at M=1. A different result
  would need a verify kernel whose cost grows sub-linearly in rows (§4.2). Third-party MTP
  acceptance for Qwen3-Next is an average acceptance length of 3.33 at draft length 7 (4.32 on
  coding) on SPEED-Bench ([Nemotron 3 Super report, arXiv 2604.12374](https://arxiv.org/html/2604.12374)).
  The official [Qwen3.6-35B-A3B model card](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
  publishes no acceptance figure.
- **An off-the-shelf DFlash drafter now exists for this exact model**
  ([z-lab/Qwen3.6-35B-A3B-DFlash](https://huggingface.co/z-lab/Qwen3.6-35B-A3B-DFlash): 3.61x on
  math500, B200, concurrency 1, block 16). The upstream repo warns that "MLX's current quantized
  matmul kernel becomes less efficient at larger verify widths", and no Apple numbers are published.
  This does not reopen #98 without a new measurement.

### 4.2 Few-row verify kernels

[llama.cpp #29869](https://github.com/ggml-org/llama.cpp/pull/29869) (merged 2026-10-05) adds
2–16-row `simdgroup_matrix` mat-mul kernels for M1–M4 in which "each weight is dequantized once for
all rows". Its thresholds were tuned on an M3 Ultra. Measured on Qwen3.8-27B Q4_0:

- Q4_0 MUL_MAT time ratio: 0.58 at n=2, 0.22 at n=8.
- Batched decode at batch 8: 38.6→119.7 t/s.
- DFlash2 decode: 30.2→110.0 t/s on code.

The PR does not claim bit-identity with the mat-vec kernels. The MoE counterpart
([#30047](https://github.com/ggml-org/llama.cpp/pull/30047)) is open and regresses on M5 Max and M2
Ultra, which shows that few-row MoE gains are chip-specific.

For Qwisp this is the mechanism that could change the spec-verify cost curve. Under strict, though,
a verify row must be bit-identical to the M=1 decode row. oMLX's `row_exact_qmv` keeps that property and still shares
dequantization. Each weight block is decoded once and applied to all rows, while every (row,
column) accumulator keeps the one-row `qmv` K order and the same final `simd_sum`
(`omlx/patches/row_exact_qmv.py`). That is the same idea as Qwisp's own b2/b3 B-exact qmv variants.
Those won 8–10% in isolation but lost about 1 ms in-chain at B=3 (handoff 2026-07-22), so the oMLX
design is confirmation, not a new lever. A true MMA few-row kernel would need a joint
re-canonicalization of decode and verify. Treat it as P2 research, not a quick port.

### 4.3 Suffix decoding and draft-length control

- [SuffixDecoding, arXiv 2411.04975v3](https://arxiv.org/html/2411.04975v3) (NeurIPS 2025, H100):
  - AgenticSQL: 5.3x mean speedup.
  - SWE-Bench: 2.5x speedup.
  - Spec-Bench chat: suffix alone gives about 1.66x; a hybrid that falls back to EAGLE-3 gives 2.5x.
  - Speculation length is `α·p`, where p is the match length.

  vLLM exposes it as `method: "suffix"` (defaults: max tree depth 24, `min_token_prob` 0.1;
  [docs](https://docs.vllm.ai/en/latest/features/speculative_decoding/)). Qwisp's SuffixSpec is
  already this design, so nothing new needs adopting.
- The draft-length controllers (SpecDec++, DISCO, SVIP, AdaEAGLE) do not model MoE expert-union
  cost. Only Cascade and EVICT (§4.1) do, so they are the relevant prior art for the next gate.
- vLLM's "Dynamic Speculative Decoding" chooses K from *concurrency*, not acceptance
  ([docs](https://docs.vllm.ai/en/latest/features/speculative_decoding/dynamic_speculative_decoding/)).
  It does not apply to single-stream Qwisp.

## 5. Deterministic and batch-invariant inference

### 5.1 Measured cost of batch invariance (datacenter GPUs)

| Source | Setting | Cost |
|---|---|---|
| [Thinking Machines](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/) | Qwen3-8B, 1000 sequences of 90–110 tokens, one GPU (model unnamed) | vLLM default 26 s; deterministic, unoptimized 55 s; deterministic with improved attention 42 s (+61.5%). The "~20% vs cuBLAS" GEMM figure has no stated shape or GPU |
| [SGLang blog](https://lmsys.org/blog/2025-09-22-sglang-deterministic/) | Qwen3-8B, H200, 256 requests | Overhead by attention backend: FlashInfer +24.4–46.0%, FA3 +27.2–35.7%, Triton +44.6–55.1% |
| [LLM-42, arXiv 2601.17768](https://arxiv.org/html/2601.17768) | H100-PCIe, Llama-3.1-8B | SGLang deterministic mode loses 24–36% offline throughput. The batch-invariant GEMM reaches 194 vs cuBLAS 527 TFLOPS |
| [TBIK, arXiv 2511.17826](https://arxiv.org/html/2511.17826) | 4×H20, Qwen3-8B, TP4 | Batch-invariant ops cost 10–33% end to end; the tree matmul reaches about 63% of cuBLAS |
| [CoRun, arXiv 2608.14376](https://arxiv.org/html/2608.14376) | vLLM 0.25.0 batch invariance (GPU not stated in our read) | Up to 74% throughput loss; a 32-request decode step goes 19.4→88.4 ms |
| [vLLM RL blog](https://vllm.ai/blog/2025-11-10-bitwise-consistent-train-inference) | Qwen3-1.7B, hardware not stated | Bitwise-consistent on-policy RL is "2.4x slower" |

The published range is roughly 10–74%, and the independent papers sit at the high end. None of these
costs map onto Qwisp directly. Qwisp is single-device with fixed chunk shapes, and its measured
obstacle is latency and occupancy rather than arithmetic order (handoff 2026-07-22). The table
does, however, explain why no competitor offers L1.

### 5.2 State of engine guarantees (as of 2026-10-15)

- **vLLM** batch invariance is "currently in beta" (`VLLM_BATCH_INVARIANT=1`) and now covers NVIDIA
  CC≥8.0, AMD Instinct and Intel XPU ([docs](https://docs.vllm.ai/en/latest/features/batch_invariance.html)).
  The open PR [#52522](https://github.com/vllm-project/vllm/pull/52522) makes EAGLE3/DFlash
  speculation batch-invariant. It does not claim that spec output equals non-spec output, and MTP is
  untested because GDN layers are not yet batch-invariant.
- **SGLang**'s deterministic roadmap ([#10278](https://github.com/sgl-project/sglang/issues/10278),
  closed 2026-02) still lists speculative decoding as unchecked.
- **llama.cpp**: the deterministic PR [#16016](https://github.com/ggml-org/llama.cpp/pull/16016) is
  still an open draft. On [#23335](https://github.com/ggml-org/llama.cpp/issues/23335)
  (Qwen3.6-27B MTP, M4 Pro, Metal), where `draft-mtp` changes the greedy stream, the maintainer
  replied: "This is expected - we use different kernels for different batch sizes."
- **mlx / mlx-lm** still have no official determinism statement. In
  [mlx-lm #1470](https://github.com/ml-explore/mlx-lm/issues/1470) (open), greedy speculative output
  diverges from plain decoding at exact bf16 logit ties between the T>1 GEMM and the T=1 GEMV. A
  docs PR that would have called this "expected" (#1592) was closed unmerged.
- **ExLlamaV3** removed atomics in its GDN and MoE kernels. Run-to-run identity went from 64/100 to
  100/100 on GSM8K prompts, but the maintainer still offers "no hard guarantee"
  ([#353](https://github.com/turboderp-org/exllamav3/issues/353)).
- **LLM-42** ([arXiv 2601.17768](https://arxiv.org/html/2601.17768)) is the closest prior art to
  Qwisp's design. It decodes on a fast path, has a fixed-shape verifier replay the tokens, and
  commits the matching prefix. Applied to 2–50% deterministic traffic, it stays within 1–8% of the
  non-deterministic baseline (H100-PCIe). This is "verify, don't constrain", applied to
  determinism rather than speculation.

### 5.3 A new risk: Metal compiler correctness on macOS 27

[mlx #4603](https://github.com/ml-explore/mlx/issues/4603) (open, labeled `bug`) reports a custom
kernel compiled with `MathModeSafe` that returns whole wrong output rows, deterministically, on an
**M3 Ultra under macOS 27.0 (26A428)**. The same kernel is exact under Relaxed/Fast math and passes
on an M2 Max under macOS 26.6.2. The reporter reproduces it without MLX and attributes it to the OS
runtime Metal compiler.

Qwisp compiles its Seedless kernels with `opts.mathMode = .safe` (`SeedlessMetalForward.swift`),
and the target machine is an M3 Pro on Darwin 27.0.0. This is not evidence that Qwisp is affected.
A row-wrong miscompile would normally turn RAWTESTS or the fidelity matrix red. It is, however,
the cheapest risk in this survey to retire. Record the OS build next to every ref set, and treat a
strict red cell that appears only after an OS update as a compiler suspect before treating it as a
regression.

## 6. Apple-Silicon engines

| Engine | Relevant 2026 mechanism | Published number (chip, model) | Relevance |
|---|---|---|---|
| MLX / mlx-lm | v0.30.0 NAX support (macOS ≥26.2); v0.32.3 adds #4567/#4572 gather tiles and #4020 GDN kernels ([releases](https://github.com/ml-explore/mlx/releases)); mlx-lm #1559 packed GDN | See §1–§2. Release notes themselves carry no numbers | Version baseline for every comparison; FastPrefill inherits these kernels only after a vendored-MLX bump |
| llama.cpp Metal | [#28948](https://github.com/ggml-org/llama.cpp/pull/28948) fuses top-k routing, MoE weighted reduce, RMS_NORM+SCALE and SSM_CONV+SiLU | qwen35moe 35B.A3B Q4_K_M: **M2 Ultra** pp2048 2309→2416, tg32 90.3→104.9 (+16%); **M5 Max** pp2048 3652→3925, tg128 104.7→117.8 | Mechanical decode fusions on this model family. Qwisp's raw decode already fuses much of this (notes/06–10); compare per-op, not by headline |
| BaseRT | Raw-Metal runtime; separate GEMV/GEMM kernels; dequant inside GEMM; v2 adds a Metal-4 expert GEMM with fused gate/up+SiLU | [v1, arXiv 2607.00501](https://arxiv.org/html/2607.00501v1), **M4 Pro**, Qwen3-30B-A3B Q4, BaseRT/MLX v0.31.2: pp128 738/415, pp512 838/664, pp2048 721/720. [v2, arXiv 2607.19438](https://arxiv.org/html/2607.19438v1), **M5 Pro**, Qwen3.6-35B-A3B 4-bit, BaseRT / mlx-lm 0.31.3 (MLX 0.32.0) / llama.cpp: pp2048 2788/2426/1628, tg128 110.7/101.9/63.1 | BaseRT's prefill lead over MLX vanishes at pp2048 on M4 Pro, so its advantage is short-prompt overhead, not GEMM speed. No kernel source is published |
| oMLX (local `958974c1`, pins mlx 0.32.3) | Lightning MTP (patched from mlx-lm #990); 256-token GDN boundary snapshots for prefix cache; Qwen3.6 D256 flash-attention; gate/up concat (claimed bit-identical); fused decode router; MoE offload with resident-first prefill | Router: "~51 us per layer at decode against a ~5 us fused launch … ~2 ms of a ~9 ms decode step" for Qwen3.6-35B-A3B (`patches/qwen35_moe_router.py`, chip not stated). Offload prefill: 396→636 tok/s at 50% residency from SSD, Qwen3.8-Flash-Next, M3 Ultra (`docs/experimental/MoE_Expert_Offload.md`) | Its offload document independently measured layer-ahead expert prediction as a loss ("decode was 2-15% slower"), which matches Qwisp's doctrine. Its packed prefill excludes Qwen MoE from exact packing because "Qwen MoE routers and shared-expert gates pick kernels by row count" |
| Ollama 0.19 MLX backend | Prefix-cache snapshots and cross-branch reuse | Qwen3.5-35B-A3B prefill 1,154→1,810 tok/s ([blog](https://ollama.com/blog/mlx)); **hardware not stated in the post** | Not usable as a hardware datapoint |

No primary source in this window measured Qwen3.5/3.6-35B-A3B prefill on an M3 Pro. The nearest
pre-M5 points come from different chips and engines, so they are context only, not a target:

- M2 Max llama.cpp, Qwen3.6-35B-A3B Q4_K_M: pp4096 = 671 t/s (user report,
  [discussion #4167](https://github.com/ggml-org/llama.cpp/discussions/4167#discussioncomment-18555918)).
- M4 Pro MLX, Qwen3-30B-A3B: pp512 = 664 tok/s (BaseRT v1).

## Corrections to notes/19

| notes/19 claim | Verdict on newer/primary evidence |
|---|---|
| §1 Thinking Machines: "≈20% GEMM, e2e 26s→42s" | **Confirmed but incomplete.** The unoptimized deterministic run was 55 s; 42 s is +61.5%. "~20% vs cuBLAS" is TML's own figure with no stated shape. Independent measurements are worse (LLM-42: 194 vs 527 TFLOPS on H100-PCIe; TBIK: about 63% of cuBLAS) |
| §1 llama.cpp deterministic PR #16016 is an unmerged draft | **Confirmed**, still open. New: the maintainer calls MTP-vs-greedy divergence on Qwen3.6 "expected" (#23335) |
| §1 vLLM batch invariance opt-in | **Confirmed**, still beta; now also AMD and XPU |
| §1 MLX has no official determinism statement | **Still true** on 2026-10-15; the mlx-lm docs PR #1592 was closed unmerged |
| §1 "steelMoEGather ships MLX steel gather_qmm_t (simdgroup_matrix)" | **Likely false on the M3 Pro** by dispatch-code reading: R=16 < `vector_limit` 18 on `applegpu_g15s` → `gather_qmv` (§1.1). Probably true on the earlier gen-13/14 non-'d' machine. Must be measured |
| §2 "NAX path gated to M5-class chips" | **Confirmed** (`device.h`: NAX requires arch gen ≥17, or ≥18 on phones) |
| §2 Rigel: MPP only 1.05–1.21x over simdgroup_matrix on M4 Max | **Confirmed** at [arXiv 2606.12765](https://arxiv.org/abs/2606.12765) |
| §2 "Prefill-specialized quant kernel effectively does not exist; nothing unclaimed to harvest" | **Refuted for MoE.** 2026 brought active, measured work on expert-aligned / few-row grouped quant GEMM tiles (MLX #4572, #4023, #3918; llama.cpp #28301, #30047). The lever is tile scheduling at low rows/expert, not dequant strategy |
| §5 BaseRT "up to 1.78x prefill over MLX on Qwen3-30B-A3B @ M4 Pro" | **Confirmed but narrow.** 1.78x is pp128 only (738 vs 415); at pp2048 the two are tied (721 vs 720) |
| notes/20 door "M5-class NAX may change the verdict" | **Confirmed directionally.** M5 tensor-API attention roughly doubles TFLOPS (llama.cpp #29570). MLX itself keeps pre-M5 D256 chunked prefill unfused, which corroborates the M1-class NO-GO |

## Prioritized techniques

"Strict" means the technique can live in the L1 path. It may still require a one-time ref
re-canonicalization, following the steel-hybrid playbook. Benefits marked *estimate* are this
survey's arithmetic, not measurements.

| # | Technique | What changes in Qwisp | Strict? | Expected benefit and evidence | Effort / risk | Priority |
|---|---|---|---|---|---|---|
| 1 | Confirm which kernel the strict MoE tile hits on M3 Pro | Measurement only: `QWISP_PREFILL_PROF=1` decomposition plus a kernel-name check, R=16 vs R=32 | n/a | Decides whether the gap is a dispatch-threshold artifact. Evidence: §1.1 code reading (`vector_limit` 18 on `applegpu_g15s` vs tile M=16). The in-code "2.24x vs per-pair qmv" suggests what the matrix path is worth, but it was measured on a different machine | One session, no code change / none | **P0** |
| 2 | Make strict MoE tiles land in the matrix-unit class on every chip | Choose R (or call the qmm kernel directly) so `M ≥ vector_limit` everywhere, e.g. R=32 with padding, or own the kernel choice instead of inheriting MLX's per-chip threshold | Yes, with ref regeneration | If #1 confirms: a large share of the 60–85% MoE term (*estimate*). Also makes the strict prefill stream chip-independent, which L1 needs across devices. Evidence: #3918 (M3 Max, bit-exact claimed) shows larger tiles win at ≥32 rows/expert; R=32 wastes rows at low rows/expert (#4023, #28301) | Low code / medium process (canonical change, cross-device ref check) | **P0** (gated on #1) |
| 3 | Remove host round-trips in strict MoE | Replace `bufToArr` host `Array` copies, `take`-materialized `xg`, and per-layer `eval`/`arrToBuf` with buffer-backed arrays or a raw gather loader (vLLM-style in-loader row index) | Yes (no arithmetic change) | Unknown until #1 attributes `moe.tileCPU` / `moe.xin` / `moe.out`. FastPrefill's header lists "4-5 CPU↔GPU syncs per layer" as a cause | Medium / low | **P1** (P0 if #1 shows these marks are large) |
| 4 | Packed sequential GDN kernel (mlx-lm #1559 layout) | New strict GDN prefill kernel with a source-level reduction tree and canary test | Yes, with ref regeneration unless it matches the current order | 1.8–2.0x per GDN layer at Qwen3.6 shapes on M5 Max (#1559). GDN was about 37% of a depth-0 chunk in notes/19 §8 (different machine) → at most about 15% of prefill wall at shallow depth (*estimate*, an upper bound because the 37% includes GDN projections that #1559 does not touch) | Medium / low (pattern is proven bitwise-pinned upstream) | **P1** |
| 5 | Re-baseline mlx-lm and bump vendored MLX for FastPrefill | Record `mx.__version__` with each mlx-lm baseline; evaluate mlx-swift with MLX ≥0.32.3 (#4572 tiles, #4020 GDN, #4505 D256 memory) for the non-strict path | FastPrefill only (a bump also changes kernels under the strict hybrid path → refs) | #4572: 1.19–1.32x on gather_qmm (chip not stated). Removes a version confound from the 3–5x gap | Medium / medium (bump touches strict kernels; must be isolated or followed by ref regeneration) | **P1** |
| 6 | Measured verify-cost curve for the SCORE gate | Replace the constant dense-tuned `r·D` with a startup-profiled `cost(D)` per device (expert-union aware) | Yes (decides only what to verify) | Cascade +7–14% over static K and ≤5% worst-case slowdown (RTX 6000 Ada); EVICT 1.21x over EAGLE-3 (A100). Incremental over Qwisp's existing adaptive K/trunc (*estimate*: low single-digit to ~10%) | Low–medium / low | **P1** |
| 7 | macOS 27 compiler check | Record OS build with refs; re-run RAWTESTS + strict matrix after OS updates; keep `.safe` but watch mlx #4603 | n/a | Risk retirement, not speed. Evidence: mlx #4603 (M3 Ultra, macOS 27.0, `MathModeSafe` wrong rows) | Very low / none | **P1** |
| 8 | Chunked (WY) GDN for FastPrefill | Use MLX #4020 `gated_delta_update` (chunk-8 pre-M5) inside the non-strict MLX graph | No (top-1 agreement 98.8% in llama.cpp #29353) | Mixed: #4020's numbers are image-only; mlx-lm #1241 and oMLX found chunked slower on Apple at B=1 | Low via MLX bump / low | **P2** |
| 9 | D-blocked D256 flash attention pre-M5 | New attention-prefill kernel blocking along D (metal-flash-attention design), additive and flag-off like notes/20 | Yes if KV order is fixed and chunk-invariant | metal-flash-attention claims 82% ALU on M3 at D=256 (instruction metric, non-causal, single-head). MLX and llama.cpp both declined to ship pre-M5 fused D256 for qL<kL | High / high (notes/20 already burned a phase) | **P2** |
| 10 | Native MTP / DFlash drafter on the serving path | Re-enable MTP-D1 or the z-lab DFlash drafter | Lossless (exact verify) | mlx-lm #990: 1.04x on 35B-A3B (M4 Pro) despite 85% acceptance; internal DFlash NO-GO at M=1 (#98) | High / high | **P2** (keep parked) |
| 11 | Few-row MMA verify kernels | Dequant-once multi-row kernel for verify rows | Only if row-exact vs M=1 decode (oMLX `row_exact_qmv` pattern); MMA form needs joint re-canonicalization | llama.cpp #29869: batch-8 decode 38.6→119.7 t/s (M3 Ultra, dense 27B). Qwisp's own B-exact variants lost in-chain at B=3 | High / medium | **P2** |
| 12 | Lossy prefill/verify shortcuts (SpecPrefill token dropping, A8 activations, MoE-Spec expert capping, approximate triangular inverse) | Bolt tier at most | No | oMLX SpecPrefill (experimental, off by default) keeps 20% of tokens above 8192 (lossy, `model_settings.py`); MoE-Spec retains about 95% quality on Qwen3 | — | Not promoted |

## Measured follow-ups (2026-10-15, M3 Pro)

Measured on the target machine after this survey was written (M3 Pro 14-core GPU, 36 GB,
macOS 27, Ornith-1.5-35B-A3B MLX 4-bit, fans pinned at max). These settle or revise items above.

| Survey item | Measurement | Result |
|---|---|---|
| #1/#2 strict tile below the `qmv` threshold | `QWISP_STEEL_R=32` vs 16, strict prefill | The threshold reading is correct (`get_qmv_batch_limit` = 18 here), but R=32 alone moved strict prefill only 148→164 tok/s (3.2K prompt) and 126→131 (8.4K). The tile kernel is not the main cost of strict prefill on this machine. |
| #5 MLX version | Same code on mlx-swift 0.31.4 vs 0.32.3, fast prefill tier | No gain from 0.32 (16K: 517 vs 499 tok/s; 32K: 435 vs 421). 0.32 still routes head_dim-256 prefill SDPA to the unfused path on pre-M5 GPUs. 0.32 also changes the strict stream on all four bench regimes, so the bump was not kept. |
| Memory | `/usr/bin/footprint`, `vmmap`, `vm_stat` during deep prefill | Unwired MLX weights were being compressed under ordinary desktop load (system compressor ~18 GB while qwisp ran unwired vs 4 GB wired). The MLX free-buffer pool also grew ~8 GB per long prefill. Wiring weights + 4 GB only around prefill and capping the pool fixed the instability; wiring through decode made the raw decode ~20% slower (21.9 vs 17.9 ms/token). |
| #9 attention memory | Unfused D256 SDPA at 32K builds a ~2 GB `[heads, 2048, S]` score tensor per layer | Query-blocking the causal SDPA (512 rows) removed the deep-context collapse in the fast tier (32K: erratic 18–331 → 421–435 tok/s). |

Fast prefill tier vs mlx-lm 0.32 on the same prompts (prefill tok/s): 16K 499–517 vs 538, 32K
421–435 vs 457. Strict prefill: 145 (16K), 56 (32K). Decode: unchanged by the tier, parity with
mlx-lm at shallow context (≈17.5–18 ms/token).

## Unverified or dropped claims

Do not quote these as evidence without re-checking them:

- **MLX #4020 GDN chunk-kernel speedups** (about 1.3–1.45x simdgroup, 1.6–2.2x NAX at B=1). These
  are a reviewer's reading of benchmark images; no textual number exists.
- **Chip for MLX #4572 and #2078.** The PRs state no hardware. #4572's author measured the related
  #4567 on an M5 Ultra, so treat #4572's numbers as likely M5-class, not M3.
- **mlx-lm #1241 chunked GDN** "≈340 Ktok/s vs 12 Ktok/s at T=8192". This was claimed in a PR that
  the maintainers closed as slower; the numbers contradict the closing rationale.
- **mlx-lm #952/#1032** "+8.6% fused gate+up on Qwen3-30B-A3B". It cites another PR and was not
  measured in the PR itself.
- **FP8-Flow-MoE (arXiv 2511.02302)** "1.7x forward". Seen only in a search snippet; the paper was
  not opened.
- **llama.cpp #27461** "132→370 tok/s pp4096 on M5 Pro". It comes from a fork commit, not the PR.
- **Ollama 0.19 MLX numbers.** The blog states no hardware; third-party attributions to M5 Max are
  secondary.
- **vllm-metal "82x TTFT".** It appears only in a mirror repo, so it was dropped.
- **RunAnywhere MetalRT** decode numbers appear only in a Hacker News launch post, so they were
  dropped.
- **TreeWY (arXiv 2608.20961)** and **STree (arXiv 2505.14969)**, tree verification for GDN/SSM.
  Their abstracts carry no numbers.
- **DraftExpert (arXiv 2607.24434)** and **SP-MoE (arXiv 2510.10302)**. These are abstract-only
  claims with unnamed hardware.
- **A GLM-4.5-Air-FP8 + MTP bitwise-equal report on vLLM #27433.** It is a single user comment.
- **CoRun (arXiv 2608.14376) GPU model.** It was not stated in the portion we read.
- **The arXiv 2506.09501 title.** The phrase "Give Me FP32 or Give Me Death" in the brief is
  not the arXiv title. The paper is "Understanding and Mitigating Numerical Sources of
  Nondeterminism in LLM Inference".
- **MLX #4505's non-NAX D256 gate** is verified in `main` source (§3), not in the PR text.

The following were not re-verified by this survey and are carried forward only by reference:
notes/19's M1 AMX bit-exact GEMM (arXiv 2606.25426), and the chip codes assumed for M1/M2 Max
parts in §1.1. Only `applegpu_g15s` for the M3 Pro was read directly.

## Primary sources

Local source (read on 2026-10-15):

- Qwisp: `swift/Sources/QwispCore/SeedlessFusedVerify.swift` (`steelMoEGather`, `steelQmm`,
  `forwardRowsHybrid`), `FastPrefill.swift`, `Tell.swift` (SCORE gate), `SeedlessMetalForward.swift`
  (compile options).
- Vendored MLX 0.31.1 in mlx-swift `dc43e62d`: `mlx/backend/metal/quantized.cpp`
  (`get_qmv_batch_limit`, `GatherQMM::eval_gpu`), `scaled_dot_product_attention.cpp`
  (`use_fallback`), `device.h` (NAX gate).
- oMLX `958974c1f4b0065d53f21348bbe8d29724679d41` (2026-10-15): `omlx/patches/qwen35_gdn_chunked.py`,
  `qwen35_fa256_attention.py`, `m5_gather_qmm_nax.py`, `row_exact_qmv.py`, `qwen35_moe_router.py`,
  `omlx/prefill/packed.py`, `omlx/model_settings.py`, `docs/experimental/MoE_Expert_Offload.md`.

MLX / mlx-lm (ml-explore):

- MLX PRs [#2078](https://github.com/ml-explore/mlx/pull/2078),
  [#3842](https://github.com/ml-explore/mlx/pull/3842),
  [#3918](https://github.com/ml-explore/mlx/pull/3918),
  [#4020](https://github.com/ml-explore/mlx/pull/4020),
  [#4023](https://github.com/ml-explore/mlx/pull/4023),
  [#4505](https://github.com/ml-explore/mlx/pull/4505),
  [#4567](https://github.com/ml-explore/mlx/pull/4567),
  [#4572](https://github.com/ml-explore/mlx/pull/4572).
- MLX issue [#4603](https://github.com/ml-explore/mlx/issues/4603);
  [releases](https://github.com/ml-explore/mlx/releases);
  `main` source at `36969c09`
  ([scaled_dot_product_attention.cpp](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/scaled_dot_product_attention.cpp)).
- mlx-lm PRs [#990](https://github.com/ml-explore/mlx-lm/pull/990),
  [#1241](https://github.com/ml-explore/mlx-lm/pull/1241),
  [#1319](https://github.com/ml-explore/mlx-lm/pull/1319),
  [#1559](https://github.com/ml-explore/mlx-lm/pull/1559),
  [#1676](https://github.com/ml-explore/mlx-lm/pull/1676).
- mlx-lm issue [#1470](https://github.com/ml-explore/mlx-lm/issues/1470).
- Apple ML Research,
  [Exploring LLMs with MLX and the Neural Accelerators in the M5 GPU](https://machinelearning.apple.com/research/exploring-llms-mlx-m5).

llama.cpp (ggml-org):

- PRs [#16016](https://github.com/ggml-org/llama.cpp/pull/16016),
  [#28109](https://github.com/ggml-org/llama.cpp/pull/28109),
  [#28301](https://github.com/ggml-org/llama.cpp/pull/28301),
  [#28948](https://github.com/ggml-org/llama.cpp/pull/28948),
  [#29353](https://github.com/ggml-org/llama.cpp/pull/29353),
  [#29570](https://github.com/ggml-org/llama.cpp/pull/29570),
  [#29869](https://github.com/ggml-org/llama.cpp/pull/29869),
  [#30047](https://github.com/ggml-org/llama.cpp/pull/30047).
- Issue [#23335](https://github.com/ggml-org/llama.cpp/issues/23335);
  [discussion #4167 comment](https://github.com/ggml-org/llama.cpp/discussions/4167#discussioncomment-18555918);
  [Metal GDN kernel](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/kernels/gated_delta_net.metal).

Papers:

- Yang et al., [Gated Delta Networks](https://arxiv.org/abs/2412.06464), arXiv:2412.06464.
- Kimi Team, [Kimi Linear](https://arxiv.org/abs/2510.26692), arXiv:2510.26692.
- Kumaresan, [Rigel](https://arxiv.org/abs/2606.12765), arXiv:2606.12765.
- Base Compute, [BaseRT](https://arxiv.org/html/2607.00501v1), arXiv:2607.00501, and
  [BaseRT on M5](https://arxiv.org/html/2607.19438v1), arXiv:2607.19438.
- Saxena et al., [Cascade](https://arxiv.org/html/2506.20675v1), arXiv:2506.20675.
- Huang et al., [MoESD](https://arxiv.org/html/2505.19645v4), arXiv:2505.19645.
- Pan et al., [EVICT](https://arxiv.org/abs/2605.00342), arXiv:2605.00342.
- Xie et al., [EcoSpec](https://arxiv.org/html/2607.12696), arXiv:2607.12696.
- McDanel et al., [MoE-Spec](https://arxiv.org/html/2602.16052v1), arXiv:2602.16052.
- NVIDIA, [Nemotron 3 Super](https://arxiv.org/html/2604.12374), arXiv:2604.12374.
- Oliaro et al., [SuffixDecoding](https://arxiv.org/html/2411.04975v3), arXiv:2411.04975.
- [LLM-42](https://arxiv.org/html/2601.17768), arXiv:2601.17768.
- [TBIK](https://arxiv.org/html/2511.17826), arXiv:2511.17826.
- [CoRun](https://arxiv.org/html/2608.14376), arXiv:2608.14376.
- Yuan et al.,
  [Understanding and Mitigating Numerical Sources of Nondeterminism in LLM Inference](https://arxiv.org/html/2506.09501v2),
  arXiv:2506.09501.

Engineering blogs and docs:

- Thinking Machines,
  [Defeating Nondeterminism in LLM Inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/).
- LMSYS, [Towards Deterministic Inference in SGLang](https://lmsys.org/blog/2025-09-22-sglang-deterministic/).
- vLLM [batch invariance docs](https://docs.vllm.ai/en/latest/features/batch_invariance.html),
  [speculative decoding docs](https://docs.vllm.ai/en/latest/features/speculative_decoding/), and
  PRs [#52522](https://github.com/vllm-project/vllm/pull/52522) and
  [#61309](https://github.com/vllm-project/vllm/pull/61309).
- SGLang [#10278](https://github.com/sgl-project/sglang/issues/10278).
- ExLlamaV3 [#353](https://github.com/turboderp-org/exllamav3/issues/353).
- [metal-flash-attention](https://github.com/philipturner/metal-flash-attention) README.
- Model cards: [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B),
  [z-lab/Qwen3.6-35B-A3B-DFlash](https://huggingface.co/z-lab/Qwen3.6-35B-A3B-DFlash).
