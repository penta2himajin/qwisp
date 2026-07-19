# HANDOFF: #98 A1 drafter SOLVED (fmm16, 13.4ms) but aggregate still loses — corrected economics: next go/no-go = prefill-ctx bootstrap (accept lever). PR #115 open
Date: 2026-07-19 | Status: DECISION-PENDING | Branch: claude/issue98-dflash-rawdrafter | Root: /Users/penta2himajin/repos/qwisp

> STOP: Before trusting anything below, run the checks in "Verify First".
> If any observation differs from the expected value, this handoff is stale —
> the code is the truth, not this document. Report the mismatch before proceeding.

## Verify First
- `git branch --show-current` → `claude/issue98-dflash-rawdrafter`; `git status` → clean.
- `git log --oneline -1` → `2d120fe` feat(dflash) dflash_fmm16. main == `fadae32`.
- PR #115 OPEN (ready, title mentions fmm16); gates verified this session: RAWTESTS **98/98**, BENCHBATCH PASS, COMPTEST 33/33, TOKTEST 5/5.
- `brew services info qwisp | head -2` → Running: true (STOP before ANY GPU run; check `ps aux | grep -E "qwisp serve|qwisp-poc"` first — user asked for a GPU-process check before every GPU run).

## Current Numbers (N=128 resident, M1 Max 64GB)
- fused DFlash: code **52.9** / agentic 50.8 / longctx 58.9 / shortnl 27.9 — all 128/128 lossless.
- flag-off: 82.7 / 76.0 / 90.5 / 81.9. DFlash loses on aggregate in every regime.
- drafter CB in-loop 13.4ms (c_draft ≈ 1.1×F, F=12.1ms); standalone 7ms; kernel probe: lm_head 622MB f16 3.25ms/191GB/s, dependent chain 0.055ms/GEMM.

## THE CORRECTED COST MODEL (memorize this; the old projection was wrong)
cost/tok = (c_draft + r_8 + P_rej·r_(p̄+1) + cpu) / T, in units of F=12.1ms.
- r_8 ≈ 3.2 (verify M=8 = 38.7ms — round-3 economics misread this as ~12ms).
- P_rej·r_(p̄+1) ≈ 19.4ms/block at code (partial-reject rebuild; OMITTED from the
  original #98 projection "95-105 tok/s" — that number was wrong).
- Aggregate code T=4.19 ⇒ even c_draft=0 loses (62ms > 50.7 parity). Measured 52.9 fits the model.
- Steady tail (ctx>60) T≈6 ⇒ parity..+8% now; ~+15% at c_draft 0.5.
⇒ THE DRAFTER IS NO LONGER THE BOTTLENECK. Binding constraints: cold-start ctx (aggregate T) and the structural r_8+rebuild cost.

## Next Action (owner picked "kernel" branch this session; now a new fork)
1. Merge PR #115 on green (still valid: default-OFF, all gates green, big kernel + record).
2. NEXT GO/NO-GO: **prefill-ctx bootstrap prototype** — feed prompt-derived ctx rows to the drafter at block 1 so aggregate T reaches the steady 5.6-6. Ceiling if it works: code ~75-90 / longctx ~85-100 vs 82.7/90.5 ⇒ win only if aggregate T ≥ ~5.5. If bootstrap fails → #98 parks as "shipped opt-in, default OFF, loses on aggregate".
3. Secondary (only if bootstrap greens): drafter polish 13.4→~7ms (fuse small ops, argmax tune) = +3-4 tok/s class; hysteresis dispatch (shortnl auto-off) before any default-ON talk.
4. Probably NO-GO (check before attempting): rebuild-avoidance via partial GDN rewind — needs per-row GDN state snapshots (same physics as spec-gdn-incompat).

## What Shipped This Session (branch, PR #115; commits a99afe3 + 2d120fe)
- Fused draft+verify seam (stepArgmax additive draftPrologue/draftTokensBuf; DFlashDispatch.draftFused; QWISP_DFLASH_NOFUSE A/B; locked test 98; total=98).
- dflash_fmm16 kernel (qmv-class f16 M-row GEMM, M-invariant by construction ⇒ ctx batching legal under locked-97; MPS deleted; ONE encoder).
- lm_head dequant f16 (+0.6GB resident-only); drafter weights copied to Metal-owned buffers.
- Probes: QWISP_RUN=dflash-gemm-bench (kernel bw), dflash-raw-bench (forward), QWISP_DFLASH_TRACE ([dflash-fused-time]/[dflash-raw-time]/[dflash-trace]).

## Falsified / Closed (do not re-propose — full trail in #98 comments 2026-07-19 ×3)
- CPU-round-trip/CB-context as drafter cost; GPU clock ramp (2nd kill); buffer provenance (noCopy vs copied); allocation granularity (slab vs separate); memory pressure (91% free); MPS granularity (~0.5-1ms/encode); fmm_rows/fmm_tiled kernel shapes (35GB/s class); qmm4_tiled barrier-tree at drafter shapes (33ms); per-row `.item()`; MLX drafter as shipping path; block=16; sliding ring-buffer v1.
- "Mystery in-loop tax": NEVER EXISTED — it was kernel bandwidth + MPS overhead + misreading r_8 as 12ms.

## Commands
- build: `cd swift && xcodebuild build -scheme qwisp-poc -configuration Release -destination 'platform=macOS' -derivedDataPath ./.xcode-build-rel -skipPackagePluginValidation`
- gates: scripts/test_raw.sh (98/98) / test_completion.sh 33/33 / test_tokenizer.sh 5/5 / test_bench_batch.sh PASS
- bench: `QWISP_DFLASH=1 QWISP_GEN=128 QWISP_RUN=raw-spec QWISP_MTP_REF=refs/resident/{code,agentic,longctx,shortnl}.safetensors .../qwisp-poc stream`

## Do Not Touch
- SeedlessVerifyTests.swift WRITE-LOCKED (total=98). Frozen forward path: additive nil-guarded seams only. refs/* raw-greedy only. Never two GPU processes; ps-check before every GPU run (user instruction this session).

## Gotchas
- Workflow args flake → hardcode in script file + {scriptPath}. Bench N caps at ref length (resident=128). SourceKit MLX noise. zsh `===` errors. Rate limits: check /status before devloops (this session ran driver-only).

## Pointers
- DFlashRawDrafter.swift (fmm16 kernel + prepare/encode/finish + gemmBench), DFlashDispatch.swift#draftFused, SeedlessFusedVerify.swift#stepArgmax seam, TellRuntime.swift fused-first wiring.
- Cost-model discussion + ceiling table: issue #98 comment (round 4). PRs #109-#115.
- Parked: #112 stable-prefix persist; QAD finetune recon (z-lab training code check, ~5min); batching #6.
