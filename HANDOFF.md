# HANDOFF: lane-batch mixer projection split SHIPPED (test 90 green, B=3 37.7→49.3/stream) — next = stepArgmaxBatch + scheduler wiring
Date: 2026-07-21 | Status: GREEN (rate-limit shutdown, no WIP) | Branch: claude/lane-batch (pushed) | Root: /Users/penta2himajin/repos/qwisp

> STOP: Before trusting anything below, run the checks in "Verify First".
> If any observation differs from the expected value, this handoff is stale —
> the code is the truth, not this document. Report the mismatch before proceeding.

## Verify First
- `git branch --show-current` → `claude/lane-batch`; `git status` → clean.
- `git log --oneline -2` → `96aa5c1` feat(seedless): lane-batch mixer projection split, then `9d4ac0d` docs handoff.
- `~/bin/jacquard run scripts/test_raw.sh` → RAWTESTS **90/90** (test 90 = lane_batch_bitexact, WRITE-LOCKED, total=90).
- GPU rule: `~/bin/jacquard` (run=shared / measure=exclusive; ambient ~14% ⇒ ALWAYS `JACQUARD_GPU_IDLE_THRESHOLD=20` for measure). Never two GPU procs; brew service must be stopped.

## Next Action
Implement `stepArgmaxBatch` (final norm + lm_head + argmax at M=B) as an additive method on `SeedlessLaneBatch` in swift/Sources/QwispCore/SeedlessLaneBatch.swift, mirroring the driver's solo stepArgmax op-chain (M-invariance already underwrites bit-exactness). Completion: extend locked test 90 (or add within total=90 budget per lock rules — do NOT bump total without the lock ritual) to assert batched argmax tokens ≡ per-lane solo stepArgmax; RAWTESTS stays 90/90-green-equivalent.

## Then
1. Scheduler wiring: extend ContinuousScheduler BatchEngine or a new LaneServe path behind `serialize` (see swift/Sources/qwisp/Server.swift#AsyncLock) so fan-out HTTP requests ride SeedlessLaneBatch instead of serialize.
2. Acceptance = #120 replay harness (fan-out trace): per-stream ≥50 tok/s, TTFT seconds, 6/6 bit-identical vs serialize.
3. ONLY IF per-stream still short at acceptance: Stage 1b (B-lane kernels: batch conv-shift/recurrence/SDPA across lanes via shared cache allocation or argument buffers — new kernels, higher risk) or Stage 2 (per-row spec, tinycodr DraftGate design).

## Commands
- gate:  `~/bin/jacquard run scripts/test_raw.sh` → RAWTESTS 90/90
- build: `cd swift && xcodebuild build -scheme qwisp-poc -configuration Release -destination 'platform=macOS' -derivedDataPath ./.xcode-build-rel -skipPackagePluginValidation`
- bench: `JACQUARD_GPU_IDLE_THRESHOLD=20 ~/bin/jacquard measure env QWISP_RUN=lane-batch-bench QWISP_MODEL="$HOME/.mtplx/models/Youssofal--Qwen3.6-35B-A3B-MTPLX-Optimized-Speed-FP16" swift/.xcode-build-rel/Build/Products/Release/qwisp-poc stream`
  (⚠ the `stream` positional arg is REQUIRED — without it qwisp-poc silently no-ops. Bench prints GPU ms per B now.)

## State
- [x] DONE: Stage 1 core — SeedlessLaneBatch bit-exact B-lane step (commit dbadf1b, prior session).
- [x] DONE: mixer projection split fast path (commit 96aa5c1): row-independent stages at M=B on driver (in-proj/qkv demux, GDN prep, norm/gate, sigmoid gate, out/o-proj); ONLY sequence-coupled kernels per lane (GDN conv-shift + recurrence = 2/layer; attn q-prep/k-prep/v-append/SDPA = 4/layer), bound at byte offsets into driver scratch rows — zero staging copies. Fallback (fusion flags off) = staged hybridDense path, kept in-file.
- [x] MEASURED (resident, ctx=1024, diverse prompts; wall ms | per-stream | aggregate | gpu ms):
      B=1 11.51 | 86.9 | — | 9.94 · B=2 16.02 | 62.4 | 124.8 | 14.22 · B=3 20.30 | 49.3 | 147.8 | 18.68 · B=4 24.80 | 40.3 | 161.3 | 22.98 · B=8 45.60 | 21.9 | 175.4 | 42.27
      (was: 78.3 / 55.9 / 37.7 / 32.3 / 16.7-134agg. Fused M-row reference curve: M=2 13.1ms→77, M=3 ~16.4→61.)
- [ ] TODO: stepArgmaxBatch → scheduler wiring → #120 replay acceptance (order above).

## Rejected
- "Shared laneX/laneOut staging is the GPU serializer" hypothesis: splitting to per-lane buffers changed NOTHING (B=3 gpu 21.94 vs 21.78). Real model: encoder-wide hazard barriers mean lane cores NEVER overlap in one encoder regardless of buffer separation; cost driver = per-lane dispatch COUNT. Do not re-propose buffer-separation as an overlap lever.
- Copying bP/aP rows to lanes on the staged path: hybridDense recomputes a/b per lane from x; weights [32,H] 4-bit ≈ 32KB — negligible vs qkv/out (MB-scale). Not worth a new seam.
- (carried) spec×batch "no gain" verdict was B=64-aggregate framing; at fan-out B=2-8 they are complementary — Stage 2 only if B≥4 short after acceptance (user-confirmed 2026-07-21).

## Do Not Touch
- `swift/Sources/QwispCore/SeedlessVerifyTests.swift`: WRITE-LOCKED (total=90 guard). Test 90 = the lossless contract; keep green after EVERY step.
- Frozen forward path (SeedlessEngine/SeedlessMetalForward/SeedlessFusedVerify/Tell/ExpertArena/ExpertSource): SeedlessLaneBatch reaches their internal statics/pipelines (same module) — fine; never MODIFY them, extensions/new files only.
- `refs/*.safetensors`: raw-greedy only.

## Decisions
- DECISION: per-lane cores use OFFSET-BOUND wrappers (laneConvShift/laneDeltaStep/laneQPrep/laneKPrep/laneWriteKV/laneSdpa in SeedlessLaneBatch) over the frozen pipelines — same pipeline + constants + grid, only setBuffer offsets differ ⇒ bit-exact by construction. Alternative (adding offset params to frozen encode statics): rejected, touches frozen file.
- DECISION: fast path requires fuseGDN/fuseATTN/fuseA1 + pipelines non-nil; flag-off falls back to the staged path (bisectability preserved). Fast path is what test 90 exercises (default flags).
- FACT: wall−gpu gap ≈ 1.6-2.0 ms at all B (encode+readback), so remaining per-stream gap vs the M-row curve is GPU-side per-lane dispatch latency (~100 dispatches/lane/step: GDN 2×30 + attn 4×10).
- ASSUMPTION: GDN recurrence state traffic (state [32,128,128] f32 ≈ 8MB r+w × 30 layers ≈ 480MB/lane/step) is the irreducible per-lane floor — NOT yet isolated by measurement; profile per-stage before attempting Stage 1b.
- DECISION: accept B=3 = 49.3 for now and proceed to wiring (aggregate 147.8 already ≈ 1.7x serialize single-stream); revisit per-lane kernels only if #120 acceptance falls short of per-stream ≥50.

## Gotchas & Blockers
- RATE LIMIT: session ended at 5h 100% used — resume after reset.
- `qwisp-poc` needs the `stream` positional arg; `QWISP_RUN=...` alone prints smoke and exits (bit me this session).
- kv.len += 1 / gc.swapState() at ENCODE time, per lane per layer, exactly once — in the fast path swapState happens after that lane's laneDeltaStep encode (conv encoded pre-swap in the earlier loop; verified green).
- setBuffer offsets are BYTES (row × width × 2 for f16, × 4 for g/beta f32); all current offsets are well-aligned power-of-2 multiples.
- MLXRandom.seed set once in runAll — do not reorder tests. LSP "No such module MLX" = SourceKit noise; xcodebuild is truth. zsh eats `===`; quote or split.

## Pointers
- Read first: swift/Sources/QwispCore/SeedlessLaneBatch.swift (whole file, ~430 lines — fast/staged paths + wrappers); swift/Sources/QwispCore/SeedlessFusedVerify.swift#encodeGdnLayerRows,#encodeAttnLayerRows (the op-chains being recomposed); swift/Sources/QwispCore/PrefixCachePoC.swift#laneBatchBench (bench, now with gpu ms).
- Pattern to follow for stepArgmaxBatch: driver's solo stepArgmax in SeedlessFusedVerify.swift (SeedlessFusedForward) — reuse its encode statics at M=B.
- Context (why lane-batch at all): Stage 0 measurement + Route A rationale in commit 9d4ac0d's HANDOFF version (git show 9d4ac0d:HANDOFF.md).
- Memory: opencode-parallel-repro-verdict, spec-gate-longctx-verdict, issue6-batching-validated, latest-handoff.
