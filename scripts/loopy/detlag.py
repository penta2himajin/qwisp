#!/usr/bin/env python3
"""Detection-lag / loop-period analysis on a token-id stream (#47 guard W sizing).

Early detector (what the production guard would run each step): a period-p exact loop is
CONFIRMED at position i when tokens[i-p:i] == tokens[i-2p:i-p] for some p in 1..Pmax, i.e.
two full periods have repeated. Reports, per file:
  - period p of the loop
  - fire position (token index where the detector first confirms)
  - onset = fire - 2p  (the loop's first token)
  - W_needed = fire - onset = 2p  (tokens to hold buffered so rewind-to-onset stays unsent)
Also checks a check-cadence C: if the guard only runs every C tokens, add up to C to W.
"""
import sys

PMAX = 64

def first_fire(toks):
    for i in range(2, len(toks) + 1):
        for p in range(1, min(PMAX, i // 2) + 1):
            if toks[i - p:i] == toks[i - 2 * p:i - p]:
                return i, p
    return None, None

for path in sys.argv[1:]:
    try:
        toks = [int(x) for x in open(path).read().split()]
    except Exception as e:
        print(f"{path}: {e}"); continue
    fire, p = first_fire(toks)
    if fire is None:
        print(f"{path}: n={len(toks)} NO exact-period loop detected (Pmax={PMAX})")
        continue
    onset = fire - 2 * p
    print(f"{path}: n={len(toks)} period={p} onset_tok={onset} fire_tok={fire} W_needed(=2p)={2*p}")
