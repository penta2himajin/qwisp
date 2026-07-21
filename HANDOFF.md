# HANDOFF: Stage 1 lane-batch — core landed bit-exact (test 90 green); NEXT = mixer projection split for speed
Date: 2026-07-21 | Status: IN-PROGRESS | Branch: claude/lane-batch (pushed) | Root: /Users/penta2himajin/repos/qwisp

> STOP: Before trusting anything below, run "Verify First". Code is truth, not this doc.

## Verify First
- `git branch --show-current` → `claude/lane-batch`; `git log --oneline -1` → "feat(seedless): lane-batched greedy decode core".
- `jacquard run scripts/test_raw.sh` → RAWTESTS **90/90** (test 90 = lane_batch_bitexact, WRITE-LOCKED, total=90).
- GPU rule: use `~/bin/jacquard` (run=shared / measure=exclusive; ambient ~14% ⇒ ALWAYS `JACQUARD_GPU_IDLE_THRESHOLD=20` for measure). Never two GPU procs; brew service must be stopped.

## Context (why)
Goal: parallel sub-agent fan-out at 50-100 tok/s per stream + TTFT seconds (user target). Route A chosen = lossless batched decode on the raw engine. Measured Stage 0 (Release, real server): serialize per-stream 85-93 but TTFT ladder →27s@8 concurrent; QWISP_BATCH=8 per-stream 17. User-confirmed reinterpretation: old "spec×batch no gain" verdict was B=64-aggregate framing; at fan-out B=2-8 they are complementary (spec-in-batch = Stage 2, only if Stage 1 leaves a gap at B≥4).

## State
- [x] `SeedlessLaneBatch.swift` (NEW, additive, frozen files untouched): B lanes = per-lane SeedlessFusedForward (own caches), driver = weights + M=B scratch. Per layer: norm M=B → mixer per-lane M=1 (staged via lane_row_copy kernel) → resid+postNorm M=B → MoE M=B → resid M=B.
- [x] Locked test 90 lane_batch_bitexact: B=3 divergent-history lanes, 4 chained steps, output rows AND cache trajectories bit-identical to solo M=1. PASS first run.
- [x] Bench runner `lane-batch-bench` (catalog; QWISP_LANE_B, QWISP_LANE_CTX). Measured (resident, ctx=1024, diverse): B=1 78.3 | B=2 55.9/stream (112 agg) | B=3 37.7 (113) | B=4 32.3 (129) | B=8 16.7 (134, =MLX parity).
- [ ] NEXT: per-stream at B=2-3 below the fused M-row curve (M=2 13.1ms→77, M=3 ~16.4→61 vs measured 17.9/26.6ms). ROOT CAUSE: mixers run per-lane M=1 ⇒ GDN/attn PROJECTION weights re-read B times (GDN = 30/40 layers, big in/out projs).

## Next Action (the speed fix)
Split each mixer: dense projections at M=B on driver scratch (row-independent ⇒ M-invariant ⇒ still bit-exact) + sequence-coupled core per lane (GDN: conv shift + recurrent + norm/gate; attn: q/k norm + RoPE@lane-position + KV append + SDPA) + out-proj at M=B. Piece kernels exist as statics (encodeGdnPrepRows, encodeGdnFusionConvShift, encodeGdnNormGateRows, encodeAttnQPrepRows/KPrepRows, encodeQmmRows) — READ encodeGdnLayerRows (SeedlessFusedVerify.swift ~2851) and encodeAttnLayerRows (~2271) FIRST to map exact scratch flow, then recompose in SeedlessLaneBatch with row-copies between driver scratch rows and lane scratch row 0 (KB-scale, trivial). Keep test 90 green after EVERY step — it is the lossless contract.
Target: B=3 ≥ ~55-61/stream. Then: (a) stepArgmaxBatch (final norm+lm_head+argmax M=B — M-invariance already tested), (b) scheduler wiring (extend ContinuousScheduler BatchEngine or new LaneServe path behind serialize), (c) acceptance = #120 replay harness (fan-out trace: per-stream ≥50, TTFT seconds, 6/6 bit-identical vs serialize).

## Key facts
- Fixture pattern for engine tests: test 23 fused_forward_rows_bitexact (synthetic 2-layer, GPU no model).
- Driver/lanes share weights physically (same MLX buffers); memory cost = lanes' arenas only.
- Research assets: `_b` MoE kernels (SeedlessMetalForward.swift:3095+, unused — encodeMoEBlockRows already per-row), archived-experiments/* branches, ghost/issue6 verdicts in memory (spec-in-batch rejection = B=64 framing, reinterpreted with user 2026-07-21).
- Stage 2 (per-row spec, tinycodr DraftGate design) only if B≥4 per-stream still short after the split.

## Gotchas
- SeedlessLaneBatch reaches internal members of SeedlessFusedForward (same module) — fine, but do NOT modify the frozen class; extensions/new files only.
- kv.len += 1 / gc.swapState() bookkeeping at ENCODE time (mirrors encodePreMoE) — per lane per layer, exactly once.
- MLXRandom.seed set once in runAll — test order matters for fixtures; do not reorder tests.
- LSP "No such module MLX"/"Cannot find X" = permanent SourceKit noise; xcodebuild is truth.
- zsh eats `===` in compound commands; quote or split.

## Pointers
- swift/Sources/QwispCore/SeedlessLaneBatch.swift (core) · SeedlessVerifyTests.swift test 90 (locked contract) · PrefixCachePoC.swift laneBatchBench · docs/measurement.md
- Memory: opencode-parallel-repro-verdict, spec-gate-longctx-verdict, issue6-batching-validated, ghost-mode-resident-verdict
