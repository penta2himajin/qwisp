"""T0: strict fidelity via token comparison (no model load).

For the STRICT method, teacher-forced fidelity vs the canonical ref is logically equivalent to
comparing the free-run OUT_TOKENS (already dumped by the speed run) against spec_greedy — and
strictly stronger: it bit-verifies the actual free-run trajectory. This replaces the second
model load (mlx-fidelity) per strict cell. NOT valid for bolt: bolt free-run diverges
autoregressively (greedy chaos), its fidelity axis requires the teacher-forced pass.

Usage: bench_tokcmp.py <ref.safetensors> <dump_file>
  dump_file: text containing an OUT_TOKENS:<csv> line (bench.sh speed-run dump).
Prints: "X/Y=Z%" (bench.sh fidelity column format) to stdout; details to stderr on mismatch.
"""
from __future__ import annotations
import sys

import mlx.core as mx


def main():
    ref_path, dump_path = sys.argv[1], sys.argv[2]
    g = [int(t) for t in mx.load(ref_path)["spec_greedy"].tolist()]
    out = None
    for line in open(dump_path).read().splitlines():
        if line.startswith("OUT_TOKENS:"):
            out = [int(t) for t in line[len("OUT_TOKENS:"):].split(",") if t]
    if out is None:
        print("NA(no OUT_TOKENS)")
        return
    n = min(len(g), len(out))
    mism = [(i, out[i], g[i]) for i in range(n) if out[i] != g[i]]
    print(f"{n - len(mism)}/{n}={100.0 * (n - len(mism)) / n:.1f}%")
    if mism:
        print(f"[tokcmp] STRICT DIVERGENCE ref={ref_path} first5={mism[:5]}", file=sys.stderr)


if __name__ == "__main__":
    main()
