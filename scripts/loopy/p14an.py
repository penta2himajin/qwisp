#!/usr/bin/env python3
"""#47 probe 14 — "why didn't it burn" analysis.

Joins, per trace:
  p-X.tsv     bolt miss trace (M=1 rows; 10-col: ...margin coldW entropy top8)
  p-X.toks    bolt emitted token ids (line i = generated position i)
  p-X.tf.tsv  strict TF replay: pos boltTok strictPred match margin(strict, -1 at pos 0)

Reports, per trace:
  1. realized flips (match==0): count, positions, first flip
  2. theorem check: strict margin at flip positions (should be small — divergence_has_hazard)
  3. mechanism split at hazard-ish strict positions (margin<3): flipped vs not
  4. repetition affinity of flips: does boltTok extend a recent repeat that strictPred breaks
     (bolt flips INTO a cycle) or vice versa
Usage: p14an.py dir name [name...]
"""
import sys

def load(d, n):
    tf = [l.split("\t") for l in open(f"{d}/{n}.tf.tsv").read().splitlines()[1:]]
    tf = [(int(r[0]), int(r[1]), int(r[2]), int(r[3]), float(r[4])) for r in tf]
    toks = [int(l) for l in open(f"{d}/{n}.toks").read().split()]
    miss = {}
    for l in open(f"{d}/{n}.tsv").read().splitlines()[1:]:
        r = l.split("\t")
        if r[3] == "1" and int(r[2]) > 0:
            miss[int(r[0])] = (int(r[1]) / int(r[2]), float(r[6]), float(r[8]))  # mr, margin_bolt, entropy
    return tf, toks, miss

def rep_affinity(toks, pos, tok, maxp=16):
    """Would emitting `tok` at `pos` extend an exact period-p repeat of the recent tail?"""
    for p in range(1, min(maxp, pos) + 1):
        if toks[pos - p] == tok and pos >= 2 * p and toks[pos - 2 * p:pos - p] == toks[pos - p:pos]:
            return True
    return False

def analyze(d, n):
    tf, toks, miss = load(d, n)
    flips = [(pos, bt, sp, mg) for pos, bt, sp, m, mg in tf if m == 0 and pos > 0]
    haz = [(pos, bt, sp, m, mg) for pos, bt, sp, m, mg in tf if 0 <= mg < 3 and pos > 0]
    hf = sum(1 for _, _, _, m, _ in haz if m == 0)
    print(f"== {n}: positions={len(tf)}  flips={len(flips)}"
          f"  first_flip={flips[0][0] if flips else None}")
    print(f"   strict-margin<3 positions={len(haz)}  of which flipped={hf}"
          f"  ({hf/len(haz)*100 if haz else 0:.0f}%)  [mechanism 1 = the rest]")
    if flips:
        mgs = sorted(mg for _, _, _, mg in flips if mg >= 0)
        big = [f for f in flips if f[3] >= 3]
        print(f"   flip strict-margin p50={mgs[len(mgs)//2]:.2f} max={mgs[-1]:.2f}"
              f"  flips with margin>=3 (theorem-violating-ish, state-drift): {len(big)}")
        into = sum(1 for pos, bt, _, _ in flips if rep_affinity(toks, pos, bt))
        outof = sum(1 for pos, _, sp, _ in flips if rep_affinity(toks, pos, sp))
        print(f"   repetition affinity: bolt-flips-INTO-repeat={into}  strict-side-extends-repeat={outof}")
        for pos, bt, sp, mg in flips[:12]:
            mr = miss.get(pos)
            print(f"     flip@{pos}: bolt={bt} strict={sp} strictMargin={mg:.2f}"
                  f"  boltRow={'mr=%.2f bm=%.2f H=%.2f' % mr if mr else 'spec-span (no M=1 row)'}"
                  f"  {'INTO-repeat' if rep_affinity(toks, pos, bt) else ''}")
        if len(flips) > 12:
            print(f"     ... +{len(flips)-12} more flips")

if __name__ == "__main__":
    d = sys.argv[1]
    for n in sys.argv[2:]:
        analyze(d, n)
