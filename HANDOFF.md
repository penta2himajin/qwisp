# HANDOFF: lane-batch workstream COMPLETE — PR #126 open (bit-exact fan-out serving, 6/6 replay, 1.63x, TTFT flat)
Date: 2026-07-21 | Status: GREEN (awaiting PR review) | Branch: claude/lane-batch (pushed, PR #126 → main) | Root: /Users/penta2himajin/repos/qwisp

> STOP: Before trusting anything below, run the checks in "Verify First".
> If any observation differs from the expected value, this handoff is stale —
> the code is the truth, not this document. Report the mismatch before proceeding.

## Verify First
- `git branch --show-current` → `claude/lane-batch`; `git status` → clean; `gh pr view 126 --json state -q .state` → OPEN.
- `git log --oneline -3` → `9ae9932` fix(server) canonical hybrid prefill, `adbfdf4` feat(server) lane serving, `759c158` feat(seedless) stepArgmaxBatch.
- `~/bin/jacquard run scripts/test_raw.sh` → RAWTESTS **91/91** (tests 90+91 = lane lossless contract, WRITE-LOCKED, total=91).
- GPU rule: `~/bin/jacquard` (run=shared / measure=exclusive; `JACQUARD_GPU_IDLE_THRESHOLD=20` for measure). Never two GPU procs; brew service stopped (`brew services info qwisp`).

## Next Action
Wait for / respond to review on PR #126 (github.com/penta2himajin/qwisp/pull/126). No code work pending on this branch. If the user wants more per-stream at B≥3 (currently 46.0 tok/s @B=3 vs the ~55-61 aspiration), the lever is Stage 1b — see Then.

## Then
1. (post-merge follow-up, only if per-stream @B≥3 matters) Stage 1b: B-lane kernels — batch GDN conv-shift/recurrence and attn SDPA across lanes in ONE dispatch each (needs per-lane cache buffer indexing: argument buffers or shared cache allocation). BEFORE attempting: profile per-stage to verify the ASSUMPTION that ~100 per-lane dispatches/step (GDN 2×30 + attn 4×10) are the gap, and that GDN state traffic (~480MB/lane/step) is the floor.
2. (parked, #121) prefix-cache-aware admission on the lane path: admits re-pay shared system+tools prefixes; sharing the prefill across lanes is the next TTFT lever for real OpenCode fan-out (prompts ~8-50K).
3. Stage 2 (per-row spec-in-batch, tinycodr DraftGate) only if B≥4 per-stream still short after 1b.

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
- [ ] TODO: none on this branch — PR review, then optional Stage 1b (see Then).

## Rejected
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
- ASSUMPTION: per-lane GDN state traffic ~480MB/lane/step is the per-stream floor at B≥3 — unverified; profile before Stage 1b.
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
