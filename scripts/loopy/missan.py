#!/usr/bin/env python3
"""Analyze miss-trace vs loop onset (#47). Args: trace.tsv onset_tok
Reports miss-rate (miss/routed) in windows: clean (before onset-margin), ramp (around onset),
loop (after). Shows whether divergence spikes at the cliff (swappable) or is broad/flat."""
import sys
rows = [l.split('\t') for l in open(sys.argv[1]).read().splitlines()[1:] if l.strip()]
data = [(int(r[0]), int(r[1]), int(r[2])) for r in rows]  # tok, miss, routed
onset = int(sys.argv[2]) if len(sys.argv) > 2 else None

def rate(seg):
    m = sum(x[1] for x in seg); r = sum(x[2] for x in seg)
    return (m / r * 100 if r else 0, m / len(seg) if seg else 0, len(seg))

print(f"total steps={len(data)}  routed/step={data[0][2] if data else 0} (40L x 8 = 320 expected)")
print(f"overall miss-rate: {rate(data)[0]:.1f}%  avg miss/step: {rate(data)[1]:.1f}")
if onset:
    clean = [x for x in data if x[0] < onset - 40]
    ramp  = [x for x in data if onset - 40 <= x[0] < onset + 10]
    loop  = [x for x in data if x[0] >= onset + 10]
    for name, seg in [("clean (<onset-40)", clean), ("ramp (onset±)", ramp), ("loop (>onset+10)", loop)]:
        if seg:
            mr, mps, n = rate(seg)
            print(f"  {name:22} n={n:4}  miss-rate={mr:5.1f}%  avg-miss/step={mps:5.1f}")
# trajectory sample every ~50 steps
print("trajectory (tok: miss/routed):")
step = max(1, len(data) // 30)
for i in range(0, len(data), step):
    t, m, r = data[i]
    bar = "#" * int(m / max(1, r) * 40)
    print(f"  {t:5}: {m:3}/{r:3} {bar}")
