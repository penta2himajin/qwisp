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
