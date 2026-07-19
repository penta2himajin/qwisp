# HANDOFF: #98 A1 CLOSED-NEGATIVE (fused CB shipped, fusion hypothesis falsified) — PR #115 open; next: owner decision (fused drafter kernel devloop vs park #98)
Date: 2026-07-19 | Status: DECISION-PENDING | Branch: claude/issue98-dflash-rawdrafter | Root: /Users/penta2himajin/repos/qwisp

> STOP: Before trusting anything below, run the checks in "Verify First".
> If any observation differs from the expected value, this handoff is stale —
> the code is the truth, not this document. Report the mismatch before proceeding.

## Verify First
- `git branch --show-current` → `claude/issue98-dflash-rawdrafter`; `git status` → clean.
- `git log --oneline -2` → `059699c` chore(profraw) on `a99afe3` probe(dflash) fused CB. main == `fadae32`.
- PR #115 OPEN (ready) → merge on green review; RAWTESTS **98/98** (total=98 guard now), COMPTEST 33/33, TOKTEST 5/5, BENCHBATCH PASS — all verified this session on a fresh binary.
- `brew services info qwisp | head -2` → Running: true (stop before ANY GPU run; start after).
- Model artifacts unchanged: `~/.mtplx/models/Youssofal--Qwen3.6-35B-A3B-MTPLX-Optimized-Speed-FP16`, drafter `~/.mtplx/models/z-lab--Qwen3.6-35B-A3B-DFlash/`.

## Next Action
1. Merge PR #115 on green (default-OFF, correctness-neutral, all gates green).
2. OWNER DECISION on #98: (a) fused drafter layer kernel devloop (verify-class Metal fusion; est. drafter ~10-16ms/block → code ~90-105 tok/s vs 82.7 = +10-25%; devloop-scale) or (b) PARK #98 at the seam+record and pick another workstream (default per earlier queue: batching #6; also parked: #112 stable-prefix persist = devloop-1-loop size, QAD finetune recon 5-min first step).

## A1 Verdict (measured 2026-07-19, full record = issue #98 comment + commit a99afe3)
- flag-off code 82.7 tok/s vs fused DFlash 32.2 (agentic 26.3 / longctx 44.3 / shortnl 15.0). Lossless 128/128 × 4 regimes with fusion ON.
- FALSIFIED (do not re-propose): ① CPU-round-trip/CB-context as drafter cost (fusion bought ~2 tok/s; drafter slow even first-in-CB while verify section of SAME CB fast) ② GPU clock ramp (in-CB high-occupancy ballast ran slow itself +34ms, sped up nothing) ③ buffer provenance (Metal-owned copies = noCopy aliases, 39ms both) ④ MPS granularity (~0.5-1ms/encode; ≥44 calls/block ⇒ floor 25-45ms > ≤15ms bar).
- Stage split (ctx=1): SDPA ≈2ms; lm_head ≈33-36ms and layers ≈20-24ms REGARDLESS of implementation (qmm4_tiled ≙ MPS f16; hand-rolled ≙ MPS) — dispatch-granularity-bound, not kernel-choice-bound.
- KILLER ARITHMETIC: at drafter ~110ms/block even p=7/7 every block = 71 tok/s < 82.7 flag-off. Fused drafter kernel is PREREQUISITE to any dflash win, not an optimization.

## What Shipped (branch, PR #115)
- Fused draft+verify seam: `stepArgmax(draftPrologue:draftTokensBuf:)` additive (default nil = byte-identical, resident-only); drafter prologue → blit draft ids → M-row verify in ONE CB, one readback. `DFlashDispatch.draftFused`; fused-first in runSpecLoop + raw-spec bench loop (snapshot BEFORE fused step). `QWISP_DFLASH_NOFUSE=1` = split-path A/B.
- `DFlashRawDrafter`: prepare()/encode(cb:)/finish() split; GEMMs = MPSMatrixMultiplication; lm_head = dequantized f16 (+0.6GB, resident-only); ctx path per-row M=1 GEMMs (batched MPS is NOT byte-M-invariant → would break locked 97); weights COPIED to Metal-owned buffers (noCopy obligation dropped); dflash_fmm_tiled kernel deleted (falsified).
- Locked test 98 `dflash_fused_draft_verify`: fused ≡ split bit-equal (drafts/evals/KV/GDN). total=98.

## Commands
- build: `cd swift && xcodebuild build -scheme qwisp-poc -configuration Release -destination 'platform=macOS' -derivedDataPath ./.xcode-build-rel -skipPackagePluginValidation`
- gates: `scripts/test_raw.sh` (98/98) / `scripts/test_completion.sh` 33/33 / `scripts/test_tokenizer.sh` 5/5 / `scripts/test_bench_batch.sh` PASS
- bench: `QWISP_DFLASH=1 QWISP_GEN=128 QWISP_RUN=raw-spec QWISP_MTP_REF=refs/resident/code.safetensors swift/.xcode-build-rel/Build/Products/Release/qwisp-poc stream` (+`QWISP_DFLASH_TRACE=1` → [dflash-fused-time]/[dflash-trace]; `QWISP_DFLASH_NOFUSE=1` → split + [dflash-raw-time]; `QWISP_DFLASH_MLX=1` → MLX drafter)
- drafter micro-bench: `QWISP_RUN=dflash-raw-bench` (back-to-back, synthetic V=8192 head, ~39ms steady)

## Do Not Touch
- `swift/Sources/QwispCore/SeedlessVerifyTests.swift`: WRITE-LOCKED (total=98 guard).
- Frozen forward path: additive nil-guarded seams only (stepArgmax prologue params are the precedent-conform additive seam).
- `refs/*.safetensors`: raw greedy only. `refs/resident/*` = N=128 G-D refs (exist locally).
- Never two GPU processes (incl. brew service).

## Decisions / Doctrine added this session
- Batched MPS GEMM output is NOT M-invariant at the byte level → any variable-M path feeding a byte-stable cache must be per-row M=1 (locked 97 contract).
- Generic dispatch backends (MLX graph / hand-rolled simple kernels / MPS) all land 40-190ms for a 6-layer dense M=8 drafter forward; verify-class fused kernels (~25µs/dispatch) are the only demonstrated regime meeting ≤15ms. Kernel fusion investment is the price of admission for small-model raw forwards on this engine.

## Gotchas & Blockers (carried + new)
- Workflow args serialization flake → hardcode args in script file, launch via {scriptPath} (workaround proven).
- Bench N caps at ref spec_greedy length (refs/resident/* = 128).
- Rate limits were 5h 96% / 7d 64% at PREVIOUS session end; this session ran driver-only (no devloops). Check /status before starting a kernel devloop.
- SourceKit "No such module MLX" = LSP noise; xcodebuild is truth.
- zsh: `===` in a compound command errors ("== not found") — quote or avoid.

## Pointers
- swift/Sources/QwispCore/DFlashRawDrafter.swift (MPS drafter + prepare/encode/finish), DFlashDispatch.swift#draftFused, SeedlessFusedVerify.swift#stepArgmax (fused seam), TellRuntime.swift (fused-first wiring both loops).
- Full measured trail: issue #98 comments (2026-07-19 ×2), PRs #109-#115.
- Parked: #112 stable-prefix persist (user-approved design); QAD finetune recon (check z-lab training-code publication first, ~5min).
