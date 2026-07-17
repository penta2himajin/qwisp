# #47 Part A — loop-diagnosis scripts (preserved)

Analysis tooling from the #47 Part A investigation (bolt C=64 deterministic repetition loop).
Part A closed as a **capacity wall** (C=64 residency < per-layer footprint ~103); 11 probes,
8+ interventions, all NO-GO on decode-time levers. Full record: memory
`bolt-loopy-47-parta-guard`. Swift instrumentation lives (env-gated, default off) on branch
`claude/47-guard-rewind`.

## VERDICT (2026-07-16, probe 12 — matched clean control): the trigger question is CLOSED, negative

Probe 11's "margin×miss co-occurrence burst separates 3/3" did NOT survive a matched clean
control (it was within-trace binning; no clean long run was ever checked against). With the
coldW (softmax cold-gate share) column added and all five traces regenerated bit-identically
(`rho-*.tsv`):

* The rectangle fires **19 FP burst bins on the clean 1500-tok QS run** (`rho-qs`, canonical
  thresholds). Every variant tried (coldW rectangle, hazard ratio ρ = coldW/margin raw and
  margin-saturated, EWMA accumulators, consecutive-bin persistence) keeps ≥8 clean FPs or
  loses the loop detections.
* **Killer counterexample: `rho-hrqs` (loops) and `rho-qs` (clean) share an identical prefix
  trajectory** — hrqs is the same QS decode with recalib refresh disabled, and its full bin
  vector `00000000125226` is exactly the first 14 bins of clean QS. At the hazard window the
  two runs are observably IDENTICAL; only the later refresh rescue differs. No function of
  per-step (margin, miss, coldW) can tell them apart at trigger time.
* Clean QS also contains hazard episodes (bins 44554-class) STRONGER than the fatal ramps of
  STORY/TCP — which loop under default refresh.

This is the empirical face of `LoopTrigger.lean`'s necessity-only caveat: hazard steps are
where divergence CAN begin (theorem), but whether it cascades depends on dynamics (refresh
timing, basin) invisible to per-step observables. **Prediction is impossible on these
observables; only verification (computing the true token = spending the IO) can
discriminate.** Adaptive demand-swap therefore cannot be hazard-triggered — remaining options
are the known product ones (RAM/C=128, strict fallback, documentation).

## Probe 14 (flip anatomy, TF strict replay): EVERYTHING burns — the "didn't burn" side does not exist

Instrumentation: `QWISP_TOK_DUMP` (emitted token ids), miss-trace cols 9-10 (`entropy`,
`top8` id:logit), and `QWISP_TF_REPLAY`/`QWISP_TF_OUT` (teacher-force a bolt token stream
through strict `--lossless`, recording per-position strict argmax + margin = realized-flip
ground truth). Evidence: `p14/` (`p-*.toks`, `p-*.tsv` 10-col, `p-*.tf.tsv`); analyzer
`p14an.py`.

1. **Ground-truth correction: all 5 runs loop** (token-level detlag2): story@156, tcp@151,
   hrqs@146, **sky@87** (period-1 `scscsc…` in the ANSWER stream — stdout was discarded, the
   reasoning-stream rollstab missed it), **qs@1244** (`Wait, this!` — the probe-12 "clean
   control" just burned late). On C=64 chain-off, LOOPY is not a tail risk; it is the
   asymptotic fate, with survival time 87–1244 tok.
2. **Flip anatomy**: bolt's trajectory diverges from strict at position 4–37 in EVERY run
   and keeps a 6–16 % background realized-flip rate pre-loop. In-loop flips collapse to
   0.5–1.9 % — the loop is the shared greedy of bolt AND strict on the polluted prefix
   (the rewind-failure mechanism, now quantified). Ignition is usually preceded by a dense
   flip cluster (story 147-155→156), but identical clusters occur mid-run and recover —
   still no prediction.
3. **Lever B (hazard-gated exact-step replay) is dead**: flips routinely occur at
   bolt-margin 8–16 (confident-wrong — the substitution+state drift moves the whole logit
   landscape, not just near-ties), so no bolt-side gate covers them; correcting all flips =
   verifying every step = strict. Flips at strict-margin ≥3 (17 in qs) directly expose the
   accumulated KV/state-drift channel.
4. **Survival-time reframe**: LOOPY = hitting time of repetition attractors in the effective
   (capacity-damaged) model's landscape. Decode-time interventions re-roll the walk
   (probe 13's lottery); the attractor density itself is set by capacity. Decode-time
   non-LOOPY-fication is closed with evidence at every layer: prediction (12), timing
   (13), correction (14).

## Probe 17 (oracle, `p17/mixprec.py`): mixed-precision residency — **GREEN, the first live lever**

The capacity fix within the same RAM: keep the top-K4 experts (routing frequency) at 4-bit
and requantize the tail through a 2-bit round trip (gs=64; the 4-bit gs=64 affine grid
represents 2-bit gs=64 points exactly, so the model computes with true 2-bit tails). Python
oracle, full coverage (no buddy) — isolates the PRECISION axis; the capacity axis is already
measured in Swift (C=128 ⇒ 4/4 healthy).

Verdict (K4=40, 216 experts at 2-bit, greedy 4 prompts, token-level): **no loops (4/4), sky
terminates naturally at EOS, 8-gram repetition metrics ≈ full-4-bit baseline, text fully
coherent** (correct quicksort code, nuanced story revision). Direct contrast with buddy
(right-size WRONG function → 5/5 burn): a coarser version of the RIGHT function carries no
attractors — `SubstituteCorrect`'s δ-comparison validated in the strongest form.

8GB RAM math (64 4-bit-slot units/layer): K4 + M/2 ≤ 64 ⇒ coverage 96 (32+64) … 128
(0+128) experts resident vs today's 64 — crossing or nearing the measured C=128 healthy
line, with buddy only in the deep tail. Remaining engineering: 2-bit gqmm kernel variant,
mixed-slot arena, offline tail requant; remaining design question: the K4/M split (oracle
can sweep K4=0 to bound the all-2-bit corner).

### Design-point sweep (2026-07-17): K4 ∈ {40, 20, 0} × {loops, 8-gram min, TF-match %}

`mixprec.py` extended: incremental demotion (40→20→0, each step 2-bit-round-trips only the
newly-tailed original 4-bit rows, one model load), plus a TF-fidelity grade — teacher-force
the o0 (full-4-bit) token stream through the patched model, % argmax match. K4=40 free-run
reuses the probe-17 `o1-*` outputs (bit-identical patch; `k40-*` are symlinks).

| K4 | coverage @8GB (K4+M) | loops (detlag2) | 8-gram min (rollstab) | TF-match story/tcp/qs/sky |
|----|----------------------|-----------------|----------------------|---------------------------|
| 40 | 88 (40+48)  | 0/4 | 0.99–1.00 | 86.9 / 90.5 / 92.8 / 91.0 % |
| 20 | 108 (20+88) | 0/4 | 0.99–1.00 | 83.4 / 89.3 / 91.7 / 89.5 % |
| 0  | 128 (0+128) | 1/4 transient (tcp period-22 @236–306, self-escapes, tail clean) | 0.94–1.00 | 79.8 / 83.8 / 89.3 / 89.7 % |

Reading: precision damage is monotone but gentle from 40→20 (TF −1..−3.5 pt, repetition
metrics unchanged, 0/4 loops) and cracks at the all-2-bit corner — k0-tcp shows the first
attractor sighting on the precision axis (a period-22 episode that ESCAPES, unlike buddy
loops which are terminal — coarser-RIGHT-function attractors are weak). **Chosen design
point: K4=20 / M=88 ⇒ coverage 108** — the largest coverage with clean loop/repetition
metrics; conservative fallback K4=32 / M=64 (coverage 96) if Swift disagrees. K4=0 excluded.

## Probe 15 (`QWISP_BUDDY_DITHER=k`): buddy-table dithering — NO-GO, landscape beats bias

Signal-processing framing: the loop is a limit cycle of a coarsely-quantized feedback system,
and dithering is the textbook remedy — a FIXED buddy table makes the substitution error a
fixed function of context (systematic drift bias toward the same cliffs), so rotate each cold
expert's substitute among its top-k coactivation candidates by token position (deterministic,
zero IO, sync-refresh only). Also covers the "swap residents every few tokens" family: the
ADAPTIVE version (recalib R=16/32) was probe 8 (lottery, 2/4 at best — the observation window
self-pollutes), and post-establishment rotation is provably useless (strict itself cannot
escape, closure ③).

Verdict (k=3, 5 prompts, ASYNC=0, token-level onsets, `p15/`): **another lottery re-roll,
not a fix.** tcp +69, sky +1237 — but story regressed from surviving (>1017) to burning at
248, qs/hrqs −20. 3 up 2 down, within-arm variance 87→1324. Refutes "systematic drift bias
is the dominant cause": decorrelating the error direction does not reliably delay ignition —
the attractor density of the capacity-damaged landscape dominates, and every table policy
(fixed, adaptive, rotating) samples from the same family. Pre-onset text quality is intact
under dithering.

**Probe 13 addendum (`QWISP_HAZARD_REFRESH`): burst-timed forced refresh is also NO-GO.**
Tested whether the refresh rescue that saved clean QS could be made deliberate (fire a sync
refresh on the both-bad burst). Verdict: refresh timing is a pure trajectory lottery —
sync-vs-async alone flips outcomes both ways, and burst-timed firing flipped a clean control
INTO a loop while saving nothing. The one theoretically-sound lever left is hazard-gated
exact-step replay (correction, not perturbation; `traj_stable_of_margin` gives trajectory ≡
strict modulo classifier soundness; Neo roofline ~12-13 tok/s) — parked until the Neo
slow-NAND tier has real users.

## History: the probe-11 lead (superseded by the verdict above)

`conj.py` — **margin × miss co-occurrence burst**: `both_bad = margin < 3 AND miss/routed >
0.20`; windows of >=4/10 both-bad rows cluster in the pre-loop ramp *within looping traces*
(STORY/TCP/QS-highR thirds rise toward the loop). Margin alone (probe 9) and miss alone
(probe 5) fail even within-trace; the conjunction is the flip-hazard form (qwisp-lean
`argmax_stable_of_margin`: a flip needs small margin *and* large substitution error). The
within-trace structure is real and theory-backed — it is *prediction* that fails, per the
verdict above. Thresholds via `MARGIN_THR/MISS_THR/BIN/BURST`; ρ mode via `RHO_THR` (needs
the 8-col coldW trace).

## Instrumentation → analyzer map

The Swift side writes trace files when these env vars point at a path (branch
`claude/47-guard-rewind`):

| env knob                     | emits                                    | analyzer      |
|------------------------------|------------------------------------------|---------------|
| `QWISP_MISS_TRACE=path`      | per-tok `tok miss routed M coldGate totGate margin` (7-col tsv) | `missan.py`, `conj.py` |
| `QWISP_MARGIN_TRACE=path`    | per-step LM-head top1−top2 margin        | (feeds the margin column above) |
| `QWISP_MISS_HIST=path`       | pre-cliff (layer,expert) cold histogram  | (inspected inline) |
| `QWISP_BOLT_STABILITY_GUARD=1` | LoopGuard rollback+detector (opt-in)   | `detlag.py`, `detlag2.py` |

Reproduce with chain ON (`QWISP_CHAIN_K=0` masks some loops — see memory). QS needs
`QWISP_BOLT_RECALIB_R=100000` to loop (recalib perturbation otherwise rescues it).

## Scripts

- `conj.py trace.tsv…` — margin×miss burst discriminator (positive lead). See above.
- `missan.py trace.tsv onset_tok` — miss-rate in clean/ramp/loop windows (probe 5: miss alone
  is **non-causal** — clean episodes survive 40%+ miss-rate; kept as the negative-result tool).
- `detlag.py toks…` / `detlag2.py toks…` — loop-period + detection-lag → rollback buffer depth
  `W` sizing for the guard. detlag2 is the conservative variant (fires only on sustained
  period-p runs, span >= max(3p, 24)); it drove the guard's `W=64`.
- `rollstab.py gen.txt…` — rolling word-8gram distinct ratio; locates loop onset (`<0.5`).
  This is the ground-truth LOOPY detector the other analyses key off.
- `sweep.sh "gpu_gb…" repeats` — LOOPY-rate vs GPU-pressure (`simulate` ballast) sweep harness.

## Data

`mm-{qs,story,tcp,sky}.tsv` — the probe-11 backing traces (7-col). `conj.py mm-*.tsv`
reproduces the within-trace finding: both-bad rate rises toward the loop (STORY 2→6→17% by
third, TCP 0→12→42%). NOTE: mm-qs is a CLEAN run (default refresh; no loop, no onset) — its
19 burst bins are the false positives that motivated probe 12.

`rho-{qs,sky,story,tcp,hrqs}.tsv` — probe-12 traces (8-col, adds coldW = per-layer-softmax
cold gate share summed over layers). Bit-identical trajectories to the mm runs (verified:
rho-qs.gen ≡ mm-qs.gen). `rho-hrqs` = QS with `QWISP_BOLT_RECALIB_R=100000` (loops);
`rho-qs`/`rho-hrqs` are the identical-prefix counterexample pair. `sweep_det.py` /
`persist.py` reproduce the probe-12 verdict tables. Ground truth: story loops at word≈100
("not not"), tcp soft-loops, hrqs loops at word≈75 ("O-average:"), qs/sky clean.

Repro commands (GPU-exclusive; each run pays a one-time strict-speed calibration):

```sh
BIN=swift/.xcode-build-rel/Build/Products/Release/qwisp   # branch claude/47-guard-rewind
QWISP_DEVICE_RAM=8 QWISP_MARGIN_TRACE=/dev/null QWISP_MISS_TRACE=out.tsv \
  "$BIN" chat --max-tokens 1000 "$PROMPT" 2>out.err >/dev/null
# prompts: STORY="Write a short story about a lighthouse keeper who discovers a message in a bottle."
#          TCP="Explain how TCP congestion control works in detail, covering slow start, congestion avoidance, fast retransmit, and fast recovery."
#          QS="Write a detailed step-by-step explanation of how quicksort works, with a Python implementation."  (--max-tokens 1500)
#          SKY="Explain why the sky appears blue in plain English, in about three paragraphs."                    (--max-tokens 1500)
# hrqs: QS + QWISP_BOLT_RECALIB_R=100000
```
