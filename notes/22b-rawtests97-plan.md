# WS-B Stage B round 2 plan — RAWTESTS 97 `lane_admit_restore_cross_arena_bitexact`

Round 1 (B1 core: sizing policy + seqBudget threading + canAdmit memory gate)
is DONE and pushed (`6d36aa7` on `claude/lane-ctx-adaptive`). This file is the
plan for round 2, the last correctness item before the GPU bench.

## Why this test exists

Stage B's central claim is notes/22 "Verified load-bearing fact 2": a
`persistentStateData()` blob captured from a lane whose arena was allocated at
size S1 restores **bit-correctly** into a lane whose arena is size S2 > S1.
Before Stage B every lane shared one arena size, so this cross-size path was
never exercised; after Stage B (per-request arena sizing) it is the NORMAL
case for #121 prefix-cache fan-out — a small-prompt lane's blob routinely
lands in a bigger lane. The wire format already supports it (used-slice save,
destination-stride write, `len <= kv.maxLen` validation, verified in recon),
but nothing LOCKS it. This test locks it.

## Contract

Extend `swift/Sources/QwispCore/SeedlessVerifyTests.swift`, bumping the
WRITE-LOCKED counter `let total = 96` → **97** (line ~52). ADD a new test;
never modify tests 90-96.

Model the new test on **test 92 `lane_admit_restore_bitexact`** (same file,
~line 5646) — reuse its whole synthetic-weight fixture verbatim (the `q4`/`q8`/
`q4e`/`mkMoE` helpers, the 2-layer GDN+attn `layers` array, `eW/lW/fnW`,
`freshCaches()`). The ONLY structural change is that `mkFwd()` becomes
parameterised on arena size:

```swift
func mkFwd(_ seqLen: Int) -> SeedlessFusedVerify.SeedlessFusedForward? {
    SeedlessFusedVerify.SeedlessFusedForward(layers: layers, caches: freshCaches(),
                                             maxM: 4, H: H, maxSeqLen: seqLen)
}
```
(test 92 hardcodes `maxSeqLen: 64`.)

### Assertions (all must hold)

Pick `S1 = 32`, `S2 = 128` (S2 > S1, both comfortably above the 5-token
history), and `S3 = 4` (deliberately smaller than the 3-token saved `len`).

1. **Cross-size restore is bit-exact.** Donor at `S1` computes prefix `W3`
   (M=3) → `persistentStateData()`. A cold lane at `S2` computes `W3` itself.
   A restored lane at `S2` restores the S1 blob. Both then run the SAME delta
   `Dx` (M=2). Assert `bitEqual` on the delta output AND on every layer cache
   via `readLayerCache(li)` for li in 0...1 (conv/rec on the GDN layer,
   k/v on the attn layer) — exactly test 92's comparison set.
2. **Reverse direction also holds.** Donor at `S2`, restore into `S1`
   (still valid: saved `len` 3 ≤ 32). Same bit-exact assertions. This is the
   direction that would break if restore ever wrote at the SOURCE's stride
   instead of the destination's.
3. **Oversized restore is refused, not silently truncated.** A blob whose
   saved `len` exceeds the destination's `maxLen` must make
   `restorePersistentState` return `false` (destination at `S3 = 4`, blob
   holding 3 tokens is fine — so instead build the donor with a LONGER history:
   e.g. donor at `S1` runs `W3` then another chunk so `len` > `S3`). The
   caller then discards that lane (cold fallback). Assert `false` is returned.
4. **Batch composition across mixed arena sizes.** Reuse test 92's tail: build
   `SeedlessLaneBatch(driver:lanes:)` over `[restoredAtS2, coldAtS1]` — i.e.
   lanes with DIFFERENT `maxSeqLen` riding one batch — and step 3 tokens via
   `stepArgmaxBatch`, asserting both lanes agree with each other and with a
   solo `stepArgmax` reference. This is the direct locked proof of notes/22
   "fact 1" (heterogeneous arena sizes are kernel-safe), which today rests
   only on code reading.

## Prohibitions

- Do not modify tests 90-96 or any `tokenbudget_*`/`lanesize_*`/`lanemem_*`
  COMPTEST case. The counter goes 96 → 97 by ADDITION only.
- Do not touch the frozen forward path, `persistentStateData`/
  `restorePersistentState` themselves (this round only LOCKS their existing
  behavior), chunk sizes, or `ContinuousBatchEngine`/`BatchBackend`.
- Do not touch round 1's B1 code — if a genuine bug in it surfaces, stop and
  report rather than silently patching, since round 1 is already committed.

## Gates

- Build: `cd swift && xcodebuild build -scheme qwisp-poc -configuration Release
  -destination 'platform=macOS' -derivedDataPath ./.xcode-build-rel
  -skipPackagePluginValidation` (RAWTESTS lives in the **qwisp-poc** scheme —
  rebuild it, not just `qwisp`, or the gate runs a stale binary).
- `~/bin/jacquard run scripts/test_raw.sh` → RAWTESTS **97/97**.
- COMPTEST stays 90/90, BENCHBATCHTEST PASS (neither should be affected).

## After this round

The correctness half of Stage B is complete. Remaining: the GPU bench from
notes/22 "Bench verification" (driver/owner-run, GPU-exclusive):
1. 35K-token prompt admits and streams correctly (today: silent empty) — the
   motivating E2E. Reuse `tools/lane_budget_probe.mjs`'s filler generator.
2. Paired no-regression A/B at B=1..4 on short prompts (sizing overhead within
   noise, ±5%).
3. Footprint gate with **≥3** simultaneous large admits (not just 2 — the
   round-1 review's `reservedBytes` TOCTOU fix specifically needs ≥3 free
   slots contending in one round to be exercised; 2 passed even with the bug).
   Use `footprint <pid>`, never `ps rss`.
Then open the PR (branch `claude/lane-ctx-adaptive`, issue #141).
