#!/usr/bin/env python3
"""margin x miss co-occurrence burst analysis (#47 Part A, probe 11 — the ONLY positive
pre-loop trigger lead among 11 probes). Reconstructs the "both-bad burst" discriminator.

Input: a QWISP_MISS_TRACE tsv with 7 cols (tok miss routed M coldGate totGate margin),
produced by BoltServe.swift when QWISP_MISS_TRACE=path and QWISP_MARGIN_TRACE is also set
(the margin column is Tell.lastMargin aligned into the same forward). Decode rows are M==1.

Discriminator (hand-tuned, per bolt-loopy-47-parta-guard memory):
  both_bad(row) = margin < MARGIN_THR  AND  miss/routed > MISS_THR
  burst bin     = >= BURST both-bad rows in a window of BIN decode steps
Neither margin alone (probe 9) nor miss alone (probe 5) separates pre-loop from clean —
their conjunction does (argmax_stable_of_margin: flip needs small margin AND large error).
Which term spikes is prompt-dependent (STORY=margin-driven, QS=miss-driven), the product
catches all. Clean thirds should read ~0 burst bins; the pre-loop/ramp third clusters.

ρ mode (hazard ratio, needs the 8-col trace with coldW = per-layer-softmax cold gate share
summed over layers): RHO_THR set ⇒ a row is "bad" when coldW/max(margin,eps) ≥ RHO_THR —
the single-statistic form derived in qwisp-lean LoopTrigger.lean (2·E/margin with the
Lipschitz scale absorbed into the threshold). Same windowed-burst framing for comparability.

Usage: conj.py trace.tsv [trace.tsv ...]
Env overrides: MARGIN_THR=3 MISS_THR=0.20 BIN=10 BURST=4 RHO_THR= (empty=rectangle mode)
"""
import os, sys

MARGIN_THR = float(os.environ.get("MARGIN_THR", 3))
MISS_THR   = float(os.environ.get("MISS_THR", 0.20))
BIN        = int(os.environ.get("BIN", 10))
BURST      = int(os.environ.get("BURST", 4))
RHO_THR    = os.environ.get("RHO_THR")  # set ⇒ ρ mode


def analyze(path):
    rows = [l.split("\t") for l in open(path).read().splitlines()[1:] if l.strip()]
    dec = []  # decode rows only (M==1): (tok, bad, rho)
    for r in rows:
        tok, miss, routed, M = int(r[0]), int(r[1]), int(r[2]), int(r[3])
        margin = float(r[6])
        if M != 1 or routed == 0:
            continue
        rho = None
        if len(r) >= 8:
            rho = float(r[7]) / max(margin, 1e-3)
        if RHO_THR is not None:
            if rho is None:
                print(f"{path}: no coldW column — ρ mode needs the 8-col trace")
                return
            bad = rho >= float(RHO_THR)
        else:
            bad = margin < MARGIN_THR and miss / routed > MISS_THR
        dec.append((tok, bad, rho))
    if not dec:
        print(f"{path}: no M==1 decode rows")
        return
    n = len(dec)
    # per-third bad rate
    thirds = [dec[: n // 3], dec[n // 3 : 2 * n // 3], dec[2 * n // 3 :]]
    tr = [sum(x[1] for x in t) / len(t) * 100 if t else 0 for t in thirds]
    # burst bins
    burst_bins = []
    for i in range(0, n - BIN + 1, BIN):
        c = sum(x[1] for x in dec[i : i + BIN])
        if c >= BURST:
            burst_bins.append((dec[i][0], c))
    mode = f"rho>={RHO_THR}" if RHO_THR is not None else f"m<{MARGIN_THR}&mr>{MISS_THR}"
    print(f"{path}: decode_steps={n} [{mode}]  bad thirds "
          f"[{tr[0]:.0f}% {tr[1]:.0f}% {tr[2]:.0f}%]  "
          f"burst_bins(>={BURST}/{BIN})={len(burst_bins)}")
    for tok, c in burst_bins:
        print(f"    burst @ tok~{tok}: {c}/{BIN} bad")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for p in sys.argv[1:]:
        analyze(p)
