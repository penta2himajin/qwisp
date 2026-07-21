# HANDOFF: PR #126 MERGED; parallel-speedup ceiling investigation CLOSED (kernel+system+roofline) — B-scaling is at the floor, next = #121
Date: 2026-07-22 | Status: GREEN | Branch: claude/lane-ceiling (probe commits, PR to open/merge) | Root: /Users/penta2himajin/repos/qwisp

> STOP: Before trusting anything below, run the checks in "Verify First".
> If any observation differs from the expected value, this handoff is stale —
> the code is the truth, not this document. Report the mismatch before proceeding.

## Verify First
- `git branch --show-current` → `claude/lane-ceiling`; `git status` → clean; PR #126 → MERGED (main has e492bbc).
- `git log --oneline -2` → `f19630c` probe MLX quantizedMM M-scaling, on top of main merge.
- `~/bin/jacquard run scripts/test_raw.sh` → RAWTESTS **91/91** (tests 90+91 = lane lossless contract, WRITE-LOCKED, total=91).
- GPU rule: `~/bin/jacquard` (run=shared / measure=exclusive; `JACQUARD_GPU_IDLE_THRESHOLD=20` for measure). Never two GPU procs; brew service stopped (`brew services info qwisp`).

## Next Action
Open/merge the small probe PR from `claude/lane-ceiling` (7b0b309 already merged via #126; f19630c = MLX reference probe). Then start #121 (prefix-cache-aware admission on lanes) — the ceiling investigation is CLOSED (see CEILING VERDICT).

## CEILING VERDICT (2026-07-22, M1 Max 64GB / 400GB/s / SLC 48MB)
Question: is further parallel (B-lane) decode speedup possible? Answer: NO at the kernel and system level; only two small orthogonal levers remain (below).
- Kernel: 3 designs measured NO-GO (Stage-1b merge +2-3 tok/s ceiling; tiled 9x slower; qmm4_rows_b register-collapse 4-6x slower). MLX's own quantizedMM measured on the same shape scales ~+19µs/row — SAME as our qmv port — and MLX's dispatch (get_qmv_batch_limit, metal/quantized.cpp) keeps per-row qmv up to M=12 for [K=2048,N=8192] on this chip, steel only beyond: the reference implementation has no better small-M kernel.
- System: QWISP_BATCH=6 (MLX batched model path) on the 6-stream replay = 16.62s vs lanes 16.67s (tie), not bit-exact, and loses at B=8 (17 vs 21.9/stream, #120). No upside.
- Roofline: active bytes/token ≈ 1.7GB weights + 0.5GB GDN-state r+w per lane → pure-BW ideal B=3/B=1 cost ratio 1.43x; actual 1.75x ⇒ lane batching realizes ~82% of the ideal RELATIVE scaling. The absolute gap to roofline (~2.2x at B=1) is the M=1 engine latency character, already established near-optimal by prior campaigns (mlx-api audit, fusion campaign) — it is not a batching problem.

## Then
1. (next workstream, #121) prefix-cache-aware admission on the lane path: admits re-pay shared system+tools prefixes; sharing the prefill across lanes is the remaining fan-out lever (TTFT with real 8-50K prompts). Design seam: LaneBatchSlots.admit re-creates the lane per request — a shared-prefix snapshot/restore (prefix cache machinery, LLMBackend.swift#generateCached idioms) could seed the lane state instead of cold prefill.
2. (small, optional) lane chainK: batch K chained steps per CB with GPU token feedback (solo chainedStepArgmax machinery + stop flag) — kills the ~1.1ms/step wall−GPU gap at B=3 ⇒ est. +5%; trades streaming granularity. Only worth doing opportunistically.
3. Stage 2 (per-row spec-in-batch, tinycodr DraftGate) remains the only substantial per-stream decode lever left — a DIFFERENT axis (acceptance economics, not batching efficiency); needs its own measurement before any build.

## Commands
- gate:  `~/bin/jacquard run scripts/test_raw.sh` → RAWTESTS 91/91; `scripts/test_bench_batch.sh` PASS; `scripts/test_completion.sh` 57/57; `scripts/test_tokenizer.sh` 5/5
- build: `cd swift && xcodebuild build -scheme qwisp -configuration Release -destination 'platform=macOS' -derivedDataPath ./.xcode-build-rel -skipPackagePluginValidation` (scheme `qwisp-poc` for the gate binary; run scripts from REPO ROOT, not swift/)
- bench: `JACQUARD_GPU_IDLE_THRESHOLD=20 ~/bin/jacquard measure env QWISP_RUN=lane-batch-bench QWISP_MODEL="$HOME/.mtplx/models/Youssofal--Qwen3.6-35B-A3B-MTPLX-Optimized-Speed-FP16" swift/.xcode-build-rel/Build/Products/Release/qwisp-poc stream` (⚠ `stream` arg REQUIRED; bench prints layers-only AND full-argmax rows + gpu ms)
- replay acceptance: see tools/opencode-repro/README.md; synthetic 6-stream trace recipe lives in this session's history — craft JSONL recs `{id,method:"POST",path:"/v1/chat/completions",tArrival:0,inflightAtArrival:6,body:"<chat JSON, stream:true>"}`, then `node tools/opencode-repro/replay_concurrent.mjs run <trace> 127.0.0.1:<port> serial|concurrent <out.json> --greedy` and `... diff ref.json cmp.json`
- lane serve: `QWISP_LANES=<B> qwisp serve` (resident ≥32GB, greedy; `QWISP_LANE_CTX` per-lane ctx, default 16384)

## State
- [x] DONE: mixer projection split fast path (96aa5c1) — M=B projections + offset-bound per-lane cores; zero staging copies.
- [x] DONE: stepArgmaxBatch + locked test 91 (759c158) — 1-CB batched greedy step; total=91.
- [x] DONE: LaneServe wiring (adbfdf4) — LaneBatchSlots on ContinuousScheduler + LaneBackend behind QWISP_LANES; step batches ACTIVE lanes only.
- [x] DONE: canonical admit (9ae9932) — hybrid prefill@1024 + Tell first-token (engine.logits qmmTiled + MLX.argMax). Replay 6/6 byte-identical, 1.63x, TTFT 20-28ms flat (serialize ladder was 3ms→23.4s).
- [x] MEASURED full greedy step (resident, ctx=1024): B=1 80.4 / B=2 59.6 / B=3 46.0 / B=4 36.2 / B=8 20.4 tok/s (agg 162.9). Layers-only: 86.9/62.4/49.3/40.3/21.9.
- [x] DONE: Stage 1b decomposition probe (7b0b309) — lane-kernel-bench; all 3 B-scaling levers NO-GO (see Rejected).
- [ ] TODO: none on this branch — PR review; next workstream = #121 (see Then).

## Rejected
- Stage 1b (merge per-lane sequence-coupled dispatches into B-lane kernels): lane-kernel-bench decomposition shows those kernels total only ~1.6ms/lane of the ~4.4ms increment; dispatch-tax share ~0.5ms → ceiling +2-3 tok/s @B=3. NO-GO. (Probe: `QWISP_RUN=lane-kernel-bench`, no model.)
- Tiled (shared-dequant) projections at decode M: 9x slower than qmv rows at every M≤8 ([N=8192,K=2048]). NO-GO.
- qmm4_rows_b (B-row qmv, weight reads shared across ≤4 rows): bit-exact by construction (byte-compare PASS) but 4-6x SLOWER at every B — x_thread[4][16] register file collapses occupancy. Kernel kept in SeedlessLaneBatch.swift as evidence, NOT wired. Do not re-propose weight-read amortization for qmv decode shapes without a zero-extra-register design.
- Tiled lm_head (QWISP_LMHEAD_QMV=0) for the batch step: 1.7-2x SLOWER at B=2-8 (B=3 argmax 21.7→36.3ms). qmv default stays. Do not re-propose.
- "Shared laneX/laneOut staging serializes lanes on GPU": buffer separation changed nothing. Real model: encoder-wide hazard barriers ⇒ lane cores never overlap; dispatch COUNT is the cost. Do not re-propose buffer separation as an overlap lever.
- Raw chunk-64 prefill + solo stepArgmax first token in admit: produces the PRE-hybrid stream → diverges from the canonical serialize stream ~100+ tokens in (f16 near-tie chains). Canonical = hybrid@1024 + qmmTiled/MLX.argMax first token (TellRuntime.swift wiring, Tell.prefill).

## Do Not Touch
- `swift/Sources/QwispCore/SeedlessVerifyTests.swift`: WRITE-LOCKED (total=91 guard; tests 90+91 = lane lossless contract).
- Frozen forward path (SeedlessEngine/SeedlessMetalForward/SeedlessFusedVerify/Tell/ExpertArena/ExpertSource): SeedlessLaneBatch/LaneServe reach internals (same module) — never MODIFY frozen files.
- `refs/*.safetensors`: raw-greedy only.

## Decisions
- FACT: the canonical greedy stream INCLUDES steel-hybrid prefill@1024 (TellRuntime: "default ON, canonical since refs re-canonicalization"). Any new decode path must mirror Tell.prefill + first-token kernels exactly or it forks the stream.
- FACT: lane batching is composition-independent — lane-concurrent ≡ lane-serial 6/6 on the real model (and tests 90/91 at synthetic dims).
- DECISION: LaneServe admits with makeFused(maxM:1024) per request (~200MB transient scratch/lane, resident tier) to get canonical chunk-1024 hybrid prefill. Alternative (persistent small-maxM lanes + raw prefill): rejected, non-canonical stream.
- DECISION: step() batches only ACTIVE lanes, rebuilt on active-set change (per-lane cost ~linear in B; idle lanes must not ride).
- FACT (was ASSUMPTION, now measured): GDN state traffic is NOT the floor — recurrence DRAM excess is only ~0.2-0.4ms/lane; the residual ~2.8ms/lane is qmv M-scaling (~+19µs/row @[8192,2048]) across projections/MoE, the kernel family floor.
- DECISION: PR #126 opened referencing #120/#121 (not closing #121 — prefix-sharing on lanes is a real remaining lever).

## Gotchas & Blockers
- `qwisp-poc` needs the `stream` positional arg (env alone silently no-ops). Run scripts from repo root (relative paths break from swift/).
- The replay harness now collects `delta.reasoning_content` too (thinking models emit content late; without it len=0 and identity is vacuous).
- gh token lacks `notifications` scope — updateSubscription mutation fails; PR author is auto-subscribed anyway.
- kv.len/gc.swapState() at ENCODE time per lane per layer exactly once; in the fast path swapState sits after that lane's recurrence encode.
- MLXRandom.seed once in runAll — do not reorder tests. SourceKit "No such module MLX" = noise. zsh eats `===`.

## Pointers
- Read first: swift/Sources/QwispCore/SeedlessLaneBatch.swift (batch core, fast/staged paths, offset wrappers); swift/Sources/QwispCore/LaneServe.swift (slots + backend, canonical admit); tools/opencode-repro/replay_concurrent.mjs (acceptance harness).
- Canonical prefill pattern: swift/Sources/QwispCore/TellRuntime.swift#prefill + the hybrid wiring block (~line 184).
- PR: https://github.com/penta2himajin/qwisp/pull/126 · Issues: #120 (verdict), #121 (parked follow-up).
- Memory: latest-handoff, opencode-parallel-repro-verdict, issue6-batching-validated.
