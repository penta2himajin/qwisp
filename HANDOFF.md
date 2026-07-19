# HANDOFF: #98 A1 = NO-GO vs flag-off chain (bootstrap gate resolved). PR #115 MERGED, PR #116 (bootstrap) OPEN. Decision: merge #116 + park A1 + move to batching #6
Date: 2026-07-19 | Status: DECISION-PENDING (recommend park) | Branch: claude/issue98-dflash-bootstrap | Root: /Users/penta2himajin/repos/qwisp

> STOP: Before trusting anything below, run the checks in "Verify First".
> If any observation differs from the expected value, this handoff is stale —
> the code is the truth, not this document. Report the mismatch before proceeding.

## Verify First
- `git branch --show-current` → `claude/issue98-dflash-bootstrap`; `git status` → clean.
- `git log --oneline -1` → `bc37cef` feat(dflash) prefill-ctx bootstrap. main == `1ca80a1` (PR #115 merged).
- PR #115 (fmm16 + fused CB) MERGED. PR #116 (bootstrap) OPEN — recommend merge (correctness-neutral accept lever).
- Gates verified this session: RAWTESTS **98/98**, BENCHBATCH PASS, COMPTEST 33/33, TOKTEST 5/5.
- **GPU rule (user instruction): `ps aux | grep -E "qwisp serve|qwisp-poc|mlx"` before EVERY GPU run; stop brew service first; if a proc is running, monitor don't collide.** `brew services info qwisp` → Running true at handoff.

## THE VERDICT: A1 is structurally NO-GO vs flag-off chain (M=1 greedy)
Drafter is fully solved (fmm16, 13.4ms/block) AND bootstrap landed (accept lever works), yet DFlash loses to flag-off in EVERY regime. This is NOT a tuning gap — it's the batch-1 latency physics:
- flag-off `chainedStepArgmax` ≈ 12.1ms/tok (M=1 forwards already near the batch-1 floor; chain amortizes dispatch over K greedy tokens). Very strong baseline on the d0≈90% draftless span.
- DFlash block=8: verify M=8 = r_8·F ≈ 3.2·12.1 = 39ms → 9.75ms/tok at accept=4, + drafter 13.4/T + reject-rebuild ⇒ ~16ms/tok.
- Break-even needs T>~5 AND drafter≤7ms AND no rebuild — a stack only longctx's tail approaches. block=4 doesn't rescue (r_4=2.0, fewer tok/block, same class).
- Same physics as [[seedless-vs-suffixspec-nogo]] / batch-1 wall: **speculation wins at compute-bound M>1, NOT M=1 greedy where chain is cheap.**

## Measured (paired A/B, N=128 resident, service stopped)
| regime | boot=0 | boot=1 | accept/step | flag-off |
|---|---|---|---|---|
| code | 56.8 | 62.5 | 3.19→4.00 | 82.7 |
| agentic | 54.5 | 62.7 | 3.59→4.54 | 76.0 |
| longctx | 65.5 | 62.6 | 4.78→5.33 | 90.5 |
| shortnl | 28.6 | 26.7 | 1.02→0.95 | 81.9 |
(longctx regresses under bootstrap: one-time full-prompt ctx feed > accept gain at long prompt. shortnl non-transfer.)
Finding: accept ramp is POSITION/content-dependent, not ctx-size alone (ctx=95 block 1 still gives p=2,1,0,0 then climbs).

## Recommended Next Action (owner call — this is a workstream-park decision)
1. **Merge PR #116** — bootstrap is a genuine correctness-neutral accept improvement; bank it for any future DFlash-favorable regime.
2. **Park #98 A1**: shipped opt-in (`QWISP_DFLASH=1` + `QWISP_DFLASH_BOOTSTRAP=1`), default OFF. Infrastructure (fmm16 kernel, fused draft+verify CB, tap, bootstrap) is solid and reusable; the limiter is the M=1 chain baseline being too strong.
3. **Move to batching #6** — the M>1 continuous-batching regime is where block-drafting economics actually favor speculation (verify M amortized across requests), and where this machinery pays off. See memory [[issue6-batching-validated]].
4. Unexplored DFlash-favorable angle if #98 is ever revived: slower device tier (8GB streaming, single forwards IO-bound → flips the economics). v1 is resident-scoped; would be a separate measurement.

## What Shipped (all on branch history; PRs #109-#116)
- #109-#114: tap, MLX drafter, parity, dispatch, c_draft probe (merged).
- #115 (merged): fused draft+verify CB seam (stepArgmax additive draftPrologue/draftTokensBuf), dflash_fmm16 kernel (qmv-class f16 M-row GEMM, M-invariant by construction), lm_head dequant f16, locked test 98 (total=98).
- #116 (open): prefill-ctx bootstrap (forwardRowsHybrid encodeTap seam + Tell.prefill onChunk + QWISP_DFLASH_BOOTSTRAP wiring).

## Falsified / Closed (do not re-propose — full trail: #98 comments 2026-07-19 ×4)
CPU-round-trip/CB-context; GPU clock ramp (×2); buffer provenance; allocation granularity; memory pressure; MPS granularity; fmm_rows/fmm_tiled (35GB/s); qmm4_tiled barrier-tree (33ms); per-row .item(); MLX drafter as ship path; block=16; sliding ring v1. "Mystery in-loop tax" NEVER existed (= kernel bw + MPS + misreading r_8 as 12ms). NEW: bootstrap accept lever REAL but insufficient; A1 vs flag-off = structural NO-GO (batch-1 physics).

## Commands
- build: `cd swift && xcodebuild build -scheme qwisp-poc -configuration Release -destination 'platform=macOS' -derivedDataPath ./.xcode-build-rel -skipPackagePluginValidation`
- gates: scripts/test_raw.sh (98/98) / test_completion.sh 33/33 / test_tokenizer.sh 5/5 / test_bench_batch.sh PASS
- A/B bench: `env QWISP_DFLASH=1 [QWISP_DFLASH_BOOTSTRAP=1] QWISP_GEN=128 QWISP_RUN=raw-spec QWISP_MTP_REF=refs/resident/{code,agentic,longctx,shortnl}.safetensors .../qwisp-poc stream` (flag-off = drop QWISP_DFLASH). Probes: dflash-gemm-bench, dflash-raw-bench, QWISP_DFLASH_TRACE.

## Do Not Touch
- SeedlessVerifyTests.swift WRITE-LOCKED (total=98). Frozen forward path: additive nil-guarded seams only. refs/* raw-greedy only. Never two GPU processes; ps-check before every GPU run.

## Pointers
- DFlashRawDrafter.swift (fmm16 + prepare/encode/finish), DFlashDispatch.swift#draftFused, SeedlessFusedVerify.swift#stepArgmax seam + forwardRowsHybrid tap, TellRuntime.swift prefill onChunk + both loop wirings.
- Verdict + cost model: issue #98 comments (rounds 3-4 + bootstrap gate). PRs #109-#116.
- Parked also: #112 stable-prefix persist (user-approved design); QAD finetune recon (z-lab training-code check ~5min).
