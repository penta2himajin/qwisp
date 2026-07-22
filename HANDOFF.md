# HANDOFF: #121 SHIPPED on branch — prefix-cache-aware lane admission (aligned-boundary restore); PR to open/merge, then L3 relaxation discussion
Date: 2026-07-22 | Status: GREEN | Branch: claude/lane-prefix-admission (PR to open → merge) | Root: /Users/penta2himajin/repos/qwisp

> STOP: Before trusting anything below, run the checks in "Verify First".
> If any observation differs from the expected value, this handoff is stale —
> the code is the truth, not this document. Report the mismatch before proceeding.

## Verify First
- `git branch --show-current` → `claude/lane-prefix-admission`; `git status` → clean; PR #126/#127 MERGED (main has 9317732).
- `git log --oneline -2` → `0bd0747` feat(seedless): prefix-cache-aware lane admission (+ a docs commit on top).
- `~/bin/jacquard run scripts/test_raw.sh` → RAWTESTS **92/92** (test 92 = lane_admit_restore_bitexact, WRITE-LOCKED, total=92).
- `scripts/test_completion.sh` → COMPTEST **67/67** (laneadmit_* 7 + prefixram maxlcp_* 3 new).
- GPU rule: `~/bin/jacquard` (run=shared / measure=exclusive). Never two GPU procs; brew service stopped.

## Next Action
1. Open/merge the PR from `claude/lane-prefix-admission` (Closes #121).
2. Then the user wants to DISCUSS L3 relaxation (see L3 section below) — margin-accept MTP in lanes is the one high-EV experiment; do not build before that discussion.

## What shipped this session (#121)
`LaneBatchSlots.admit` consults two `PrefixRAMStore` tiers before prefilling and restores the
longest cached prefix into the fresh lane (`restorePersistentState`), prefilling only the delta:
- **sharedStore** — recurrence-detected harness prefixes (fan-out sharing; capture boundary =
  floor(LCP with any stored key / chunk) when ≥ stableMinTokens(1024)).
- **convStore** — per-conversation last-aligned-boundary states (multi-turn extension).
- Two stores because save()'s supersede semantics would drop a shared-harness boundary entry
  whenever a longer full-prompt key lands (alternate requests would re-pay the shared prefill).
- **ALL boundaries are ABSOLUTE chunk-aligned (1024 hybrid / 64 raw)** ⇒ delta prefill reproduces
  the cold path's chunk boundaries exactly ⇒ bit-exact BY CONSTRUCTION (no chunk-composition
  invariance assumption on the steel-hybrid kernels; serialize's arbitrary-boundary restores lean
  on PREFIXE2E instead).
- Default ON; `QWISP_LANE_PREFIX=0` or `QWISP_LANE_PREFIX_MB=0` opts out (budget default 3072MB,
  1/3 shared / 2/3 conv). `LaneBatchSlots.restoreHits` = observability counter.
- Pure plan logic `LaneBatchSlots.admitPlan` (COMPTEST `laneadmit_*`); `PrefixRAMStore.maxCommonPrefix` added.

## Measured (synthetic 6-stream fan-out, 12.4K-char shared system prefix, max_tokens 256, QWISP_LANES=6)
- Identity vs strict serial reference: **6/6 byte-identical** (prefix ON and OFF).
- Admission lever isolated: lanes prefix OFF 56.2s → ON **32.8s = 1.71x** (cold admit ~3.9s/req → restore+delta ~0.15s).
- vs serialize concurrent 38.1s (1.16x), vs serial reference 42.6s (1.30x). Streams are short
  (256 tok) and decode-dominated — the admission win scales with prompt length (real traces 8-50K).
- Trace generator + driver: scratchpad `gen_fanout_prefix.py`, `driver.sh`, `fanout6-prefix.jsonl`
  (session scratchpad; regenerate as needed — recipe: 6 recs, shared ~12.4K-char system, distinct
  short user tasks, stream:true, temperature 0).
- MEASURED (server logs, probes mt64/B=1): the earlier "server ~2x/step" read was a MISATTRIBUTION.
  Decomposition: (1) DOMINANT = admits stall the single decode thread (ContinuousScheduler.loop):
  warm admit ~2.4s (delta prefill ~880 tok @ ~350-450 tok/s), cold ~6.2s; 6 admits = 18.3s of the
  32.8s wall (56%) — prefill-overlap admission is the next lane lever. (2) True per-token server
  tax is only 14% @B=1 (69.0 vs 80.4) / ~25% @B=6 (19.3 vs ~26 interp) — cause located:
  Server.swift:196 + ChatCompletion.swift:210 re-decode the FULL output per token (O(n^2)
  tokenizer.decode(outIds)) + splitOutput/<tool_call> full scans + JSON/SSE per token per stream.
  Incremental detokenize kills it, server-layer only. (3) serialize's apparent 40 tok/s was
  TTFT+lock in wall; real serialize decode 74-79 tok/s (accept 2-22% on this synthetic thinking
  content — spec adds little there).

## Then
1. L3 relaxation discussion with the user (owner asked explicitly). Assets + estimates in the
   previous handoff's L3 section, summarized: margin-accept MTP-D1 in lanes = verify rows vanish;
   est. B=3 46→~64 tok/s/stream (+40%); all parts exist (raw MTP-D1 port green default OFF,
   measured D1 accepts .506-.829, lastMargin instrumentation, tinycodr DraftGate design); quality
   must be MEASURED (token-match + task eval; bolt fidelity 88.7-98.2 = reference dial). Stacked
   realistic ceiling B=3 ~75-85 tok/s (~1.6-1.8x). Greedy-equal-only re-canonicalization: single-digit %, not worth alone.
2. (small, optional) lane chainK — batch K chained steps per CB (~+5% @B=3, trades streaming granularity).
3. Server per-token overhead decomposition (see OBSERVATION above) — possibly a bigger real-world
   lever than any remaining kernel work.

## Commands
- gate:  `~/bin/jacquard run scripts/test_raw.sh` → RAWTESTS 92/92; `scripts/test_bench_batch.sh` PASS; `scripts/test_completion.sh` 67/67; `scripts/test_tokenizer.sh` 5/5
- build: `cd swift && xcodebuild build -scheme qwisp -configuration Release -destination 'platform=macOS' -derivedDataPath ./.xcode-build-rel -skipPackagePluginValidation` (scheme `qwisp-poc` for the gate binary; run scripts from REPO ROOT)
- replay gate: driver pattern = start `QWISP_MODEL=... QWISP_PORT=<p> [QWISP_LANES=6] qwisp serve`,
  wait for /v1/models 200, then `node tools/opencode-repro/replay_concurrent.mjs run <trace> 127.0.0.1:<p> serial|concurrent <out.json> --greedy`, then `... diff ref.json cmp.json`. One server at a time (GPU exclusive).
- lane serve: `QWISP_LANES=<B> qwisp serve` (resident ≥32GB, greedy; `QWISP_LANE_CTX` per-lane ctx default 16384; `QWISP_LANE_PREFIX[_MB]` see above)

## State
- [x] DONE #121: prefix-cache-aware lane admission (0bd0747) — see "What shipped".
- [x] Locked test 92 lane_admit_restore_bitexact (total=92): blob restore + delta ≡ cold bit-exact
      (delta rows, cache trajectories, 3 chained stepArgmaxBatch vs solo), truncated blob rejected.
- [x] Replay gate GREEN (identity 6/6; admission 1.71x; beats serialize concurrent).
- [ ] TODO: open PR (Closes #121); after merge → L3 discussion (do NOT start building Stage 2 before it).

## Rejected / Doctrine (carried)
- Release-time state save (prompt+generated as key): REJECTED — decode M=1 raw steps produce
  different KV bits than hybrid prefill of the same tokens; restoring such state forks the
  canonical stream. Only PREFILL-computed boundary states are saveable.
- One-store design: REJECTED (supersede kills shared boundary entries — see above).
- Ceiling verdict (PR #126/#127, Lean-certified): B-lane kernel/system speedup CLOSED; empirical
  aggregate ceiling 233 tok/s; b2/b3 B-exact variants opt-in `QWISP_LANE_BEXACT=1` default OFF
  (isolated-dispatch win ≠ in-chain win). Do not re-propose weight-read amortization for qmv decode
  shapes without a zero-extra-register design.
- Canonical stream INCLUDES steel-hybrid prefill@1024 + first token via engine.logits(qmmTiled)+MLX.argMax.
  Any new admit path must mirror it exactly (LaneServe.makeLane + admit loop is the reference).

## Do Not Touch
- `swift/Sources/QwispCore/SeedlessVerifyTests.swift`: WRITE-LOCKED (total=92 guard; tests 90-92 = lane contracts).
- Frozen forward path (SeedlessEngine/SeedlessMetalForward/SeedlessFusedVerify/Tell/ExpertArena/ExpertSource).
- `refs/*.safetensors`: raw-greedy only.

## Gotchas & Blockers
- Lane replay TTFT ≈ SSE connection ack (~300ms), NOT first token — the lane server yields the
  role chunk before admit; do not read lane TTFT as prefill time (serialize TTFT ≈ real first token).
- `qwisp-poc` needs the `stream` positional arg. Run scripts from repo root.
- kv.len/gc.swapState() at ENCODE time per lane per layer exactly once (fast path: after that lane's recurrence encode).
- MLXRandom.seed once in runAll — do not reorder tests. SourceKit "No such module MLX" = noise. zsh eats `===`.
- gh token lacks `notifications` scope — updateSubscription mutation fails; PR author is auto-subscribed anyway.

## Pointers
- Read first: swift/Sources/QwispCore/LaneServe.swift (admitPlan + admit + makeLane); swift/Sources/QwispCore/PrefixPersist.swift (PrefixRAMStore + maxCommonPrefix); SeedlessVerifyTests.swift test 92.
- Issues: #121 (this, closes on merge) · #120 (repro harness) · #117/#89/#112 (prefix cache tiers this builds on).
- Memory: latest-handoff, opencode-parallel-repro-verdict, prefix-cache-progress.
