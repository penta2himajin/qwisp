#!/usr/bin/env python3
"""Detector comparison: rectangle vs coldW-rectangle vs rho variants, burst-windowed."""
import sys
SP = sys.argv[1]
TRACES = [("story", "LOOP"), ("tcp", "LOOP"), ("hrqs", "LOOP"), ("qs", "CLEAN"), ("sky", "CLEAN")]
BIN, BURST = 10, 4

def load(name):
    rows = [l.split("\t") for l in open(f"{SP}/rho-{name}.tsv").read().splitlines()[1:]]
    return [(int(r[0]), int(r[1])/int(r[2]), float(r[6]), float(r[7]))
            for r in rows if r[3] == "1" and int(r[2]) > 0]  # (tok, missrate, margin, coldW)

def bursts(dec, bad):
    out = []
    for i in range(0, len(dec) - BIN + 1, BIN):
        c = sum(bad(x) for x in dec[i:i+BIN])
        if c >= BURST: out.append((dec[i][0], c))
    return out

DETS = {
    "A rect(m<3,miss>.2)":   lambda x: x[2] < 3 and x[1] > 0.20,
    "B rectW(m<3,cW>8)":     lambda x: x[2] < 3 and x[3] > 8,
    "B rectW(m<3,cW>10)":    lambda x: x[2] < 3 and x[3] > 10,
    "B rectW(m<3,cW>12)":    lambda x: x[2] < 3 and x[3] > 12,
    "C rho_raw>=3":          lambda x: x[3]/max(x[2], 1e-3) >= 3,
    "C rho_raw>=8":          lambda x: x[3]/max(x[2], 1e-3) >= 8,
    "D rho_sat3>=2":         lambda x: x[3]/max(x[2], 3) >= 2,
    "D rho_sat3>=3":         lambda x: x[3]/max(x[2], 3) >= 3,
    "D rho_sat3>=4":         lambda x: x[3]/max(x[2], 3) >= 4,
}
data = {n: load(n) for n, _ in TRACES}
hdr = f"{'detector':24}" + "".join(f"{n+'('+g[0]+')':>14}" for n, g in TRACES)
print(hdr); print("-" * len(hdr))
for dn, f in DETS.items():
    cells = []
    for n, gt in TRACES:
        b = bursts(data[n], f)
        cells.append(f"{len(b):>14}")
    print(f"{dn:24}" + "".join(cells))
print("\nburst positions (tok) per detector on LOOP traces:")
for dn, f in DETS.items():
    pos = {n: [t for t, _ in bursts(data[n], f)] for n, gt in TRACES if gt == "LOOP"}
    print(f"  {dn:24} story={pos['story']} tcp={pos['tcp']} hrqs={pos['hrqs']}")
