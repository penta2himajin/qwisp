#!/usr/bin/env python3
"""Conservative loop detector for W sizing (#47). Fires only when the tail is exactly
periodic with period p over a span of at least max(REPS*p, MIN_SPAN) tokens — so it does
NOT trip on legitimate short repeats ('aa', '10 10', 'ha ha ha') but catches sustained
degeneration. Reports lag = fire - onset = the buffer depth W needed for a clean rewind."""
import sys

PMAX = 64
REPS = 3          # require >= this many exact repetitions
MIN_SPAN = 24     # ...and the periodic run must span >= this many tokens

def periodic_tail(toks, i, p):
    """Longest run ending at i that is exactly period-p. Returns span length."""
    span = p
    while i - span - p >= 0 and toks[i - span - p:i - span] == toks[i - p:i][:p] and \
          toks[i - span:i] == toks[i - span - p:i - p]:  # extend one more period back
        span += p
    # recompute span directly: count how many trailing tokens are period-p consistent
    span = 0
    while i - span - p >= 0 and toks[i - span - p:i - span] == toks[i - span:i][-p:] if span else True:
        break
    # simpler exact count:
    s = p
    while i - s - p >= 0 and toks[i - s - p:i - s] == toks[i - p:i]:
        s += p
    return s

def first_fire(toks):
    for i in range(2, len(toks) + 1):
        for p in range(1, min(PMAX, i // 2) + 1):
            # need REPS repeats => span >= REPS*p, and span >= MIN_SPAN
            need = max(REPS * p, MIN_SPAN)
            if i < need:
                continue
            block = toks[i - p:i]
            reps = 1
            j = i - p
            while j - p >= 0 and toks[j - p:j] == block:
                reps += 1; j -= p
            span = reps * p
            if reps >= REPS and span >= MIN_SPAN:
                onset = i - span
                return i, p, span, onset
    return None, None, None, None

for path in sys.argv[1:]:
    toks = [int(x) for x in open(path).read().split()]
    fire, p, span, onset = first_fire(toks)
    if fire is None:
        print(f"{path}: n={len(toks)} no sustained loop (REPS={REPS},MIN_SPAN={MIN_SPAN})")
        continue
    print(f"{path}: n={len(toks)} period={p} onset={onset} fire={fire} lag=W_needed={fire-onset}")
