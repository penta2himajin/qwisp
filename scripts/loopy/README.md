# #47 Part A — loop-diagnosis scripts (preserved)

Analysis tooling from the #47 Part A investigation (bolt C=64 deterministic repetition loop).
Part A closed as a **capacity wall** (C=64 residency < per-layer footprint ~103); 11 probes,
8+ interventions, all NO-GO on decode-time levers. Full record: memory
`bolt-loopy-47-parta-guard`. Preserved here so a future **adaptive demand-swap / margin-gated
escalation** campaign can reproduce the trace analysis without re-deriving it. Swift
instrumentation lives (env-gated, default off) on branch `claude/47-guard-rewind`.

## The one positive lead

`conj.py` — **margin × miss co-occurrence burst**, the only trigger that generalised (3/3
testable prompts). `both_bad = margin < 3 AND miss/routed > 0.20`; a window of >=4/10 both-bad
decode steps clusters in the pre-loop ramp while clean is sparse. Margin alone (probe 9) and
miss alone (probe 5) each fail to separate — only the product does (qwisp-lean
`argmax_stable_of_margin`: an argmax flip needs small margin *and* large substitution error).
Caveats: hand-tuned thresholds (exposed as `MARGIN_THR/MISS_THR/BIN/BURST` env vars), ~10-20 tok
lead, measured on the chain-off / high-R path. Not yet wired to escalation or GPU-margin
(chain-on) instrumentation — that is the next step if the campaign resumes.

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
reproduces the finding: both-bad rate rises toward the loop (STORY 2→6→17% by third, TCP
0→12→42%, QS clusters bursts at onset≈570). Regenerating needs the model + GPU, so kept.
