#!/usr/bin/env python3
"""Rolling word-8-gram distinct ratio over a generation (LOOPY onset locator).
Usage: rollstab.py file [file...]  — prints per-file: min ratio, onset word index (first window <0.5), total words."""
import sys

def ratios(words, n=8, window=256, step=32):
    out = []
    for start in range(0, max(1, len(words) - window + 1), step):
        tail = words[start:start + window]
        if len(tail) < n * 2:
            continue
        grams = {tuple(tail[i:i + n]) for i in range(len(tail) - n + 1)}
        out.append((start, len(grams) / (len(tail) - n + 1)))
    return out

for path in sys.argv[1:]:
    words = open(path).read().split()
    rs = ratios(words)
    if not rs:
        print(f"{path}: too short ({len(words)} words)")
        continue
    mn = min(r for _, r in rs)
    onset = next((s for s, r in rs if r < 0.5), None)
    tail_r = rs[-1][1]
    print(f"{path}: words={len(words)} min={mn:.2f} tail={tail_r:.2f} onset_word={onset}")
