#!/usr/bin/env python3
"""Persistence analysis: bin-count vectors + EWMA accumulator, clean vs loop traces."""
import sys
SP = sys.argv[1]
TRACES = [("story", "LOOP"), ("tcp", "LOOP"), ("hrqs", "LOOP"), ("qs", "CLEAN"), ("sky", "CLEAN")]

def load(name):
    rows = [l.split("\t") for l in open(f"{SP}/rho-{name}.tsv").read().splitlines()[1:]]
    return [(int(r[0]), int(r[1])/int(r[2]), float(r[6]), float(r[7]))
            for r in rows if r[3] == "1" and int(r[2]) > 0]

def badf(x):  # rectangle baseline
    return x[2] < 3 and x[1] > 0.20

for n, gt in TRACES:
    dec = load(n)
    counts = []
    for i in range(0, len(dec) - 9, 10):
        counts.append(sum(badf(x) for x in dec[i:i+10]))
    print(f"{n:6}({gt}) tokrange 0..{dec[-1][0]}  bins:", "".join(str(min(c,9)) for c in counts))
    # EWMA of rho_sat3, halflife 15 rows
    a = 0.955  # decay
    ew, mx, trail = 0.0, 0.0, []
    for x in dec:
        rho = x[3]/max(x[2], 3)
        ew = a*ew + (1-a)*rho
        trail.append(ew); mx = max(mx, ew)
    peak_i = trail.index(mx)
    print(f"       EWMA(rho_sat3): max={mx:.2f} at row {peak_i}/{len(dec)} (tok {dec[peak_i][0]}), "
          f"q50={sorted(trail)[len(trail)//2]:.2f}")
