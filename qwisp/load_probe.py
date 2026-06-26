"""miss 処理パイプラインの分解 — 「IO税」が SSD読み か np→mx変換 か concat か eval かを特定.

io_probe で SSD 読みは 0.146ms/expert・帯域律速と判明。だが 2tok verify の cold は ≤16/層＝
最大でも 2.3ms/forward の SSD のはず。full_profile の「IO税 52ms」はそれでは説明できない。
→ load_expert_slices(pread+frombuffer+mx.array) と concat と eval の各段を実測し真因を出す。

実行: PY -m qwisp.load_probe [--experts 16 --layers 40]
"""
from __future__ import annotations
import argparse
import os
import time

import numpy as np
import mlx.core as mx

from .expert_source import ExpertSource, PROJS, PARTS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.expanduser("~/.mtplx/models/qwisp-experts-2bit"))
    ap.add_argument("--experts", type=int, default=16, help="1層あたり miss expert 数（2tok verify想定）")
    ap.add_argument("--layers", type=int, default=40)
    ap.add_argument("--reps", type=int, default=20)
    args = ap.parse_args()
    src = ExpertSource(args.dir)
    K = args.experts
    keys = [f"{p}.{q}" for p in PROJS for q in PARTS]
    rng = np.random.default_rng(0)

    def timeit(fn, reps=args.reps, warm=3):
        for _ in range(warm):
            fn()
        t = time.perf_counter()
        for _ in range(reps):
            fn()
        return (time.perf_counter() - t) / reps * 1e3

    # (1) raw pread だけ（mx 変換なし）= SSD
    fd = os.open(os.path.join(args.dir, src.wm[src._key(0, "gate_proj", "weight")]), os.O_RDONLY)
    hdr, ds = src._header(src.wm[src._key(0, "gate_proj", "weight")])
    strides = []
    for p in PROJS:
        for q in PARTS:
            t = hdr[src._key(0, p, q)]; b, e = t["data_offsets"]; n = t["shape"][0]
            strides.append((ds + b, (e - b) // n))

    def raw_pread():
        exps = rng.integers(0, 256, K)
        for e in exps:
            for base, st in strides:
                os.pread(fd, st, base + int(e) * st)
    t_raw = timeit(raw_pread)

    # (2) src.slice = pread + np.frombuffer + mx.array（1スライス）→ K expert×9
    def slice_full():
        exps = rng.integers(0, 256, K)
        return [src.slice(0, p, q, int(e)) for e in exps for p in PROJS for q in PARTS]
    t_slice = timeit(lambda: slice_full())

    # (3) + mx.eval（GPU materialize 強制）
    def slice_eval():
        arrs = slice_full(); mx.eval(arrs); return arrs
    t_slice_eval = timeit(slice_eval)

    # (4) load_expert_slices（並列 pread 版, 実 cache 経路）
    from concurrent.futures import ThreadPoolExecutor
    pool = ThreadPoolExecutor(max_workers=8)

    def load_par():
        exps = rng.integers(0, 256, K).tolist()
        return src.load_expert_slices(0, exps, pool)
    t_load = timeit(load_par)

    def load_par_eval():
        d = load_par(); mx.eval([v for dd in d.values() for v in dd.values()]); return d

    t_load_eval = timeit(load_par_eval)

    # (5) + concat（ExpertCache.gather の stacking）
    sample = load_par()
    per = [sample[e] for e in sample]

    def concat():
        return {k: mx.concatenate([s[k] for s in per], axis=0) for k in keys}
    t_concat = timeit(lambda: mx.eval(list(concat().values())))

    print(f"\n[load] K={K} miss expert/層, {len(strides)}スライス/expert, reps={args.reps}")
    print("-" * 56)
    print(f"  (1) raw pread のみ (SSD)       : {t_raw:7.3f} ms/層")
    print(f"  (2) +frombuffer+mx.array(lazy) : {t_slice:7.3f} ms/層")
    print(f"  (3) (2)+mx.eval (GPU materialize): {t_slice_eval:7.3f} ms/層")
    print(f"  (4) load_expert_slices 並列(lazy): {t_load:7.3f} ms/層")
    print(f"  (5) (4)+mx.eval                : {t_load_eval:7.3f} ms/層")
    print(f"  (6) concat+eval                : {t_concat:7.3f} ms/層")
    print("-" * 56)
    print(f"  ×{args.layers}層/forward 換算:")
    print(f"    SSD(raw)        : {t_raw*args.layers:6.1f} ms")
    print(f"    np→mx変換+eval  : {(t_slice_eval-t_raw)*args.layers:6.1f} ms")
    print(f"    concat          : {t_concat*args.layers:6.1f} ms")
    print(f"  → miss税の真因: "
          f"{'SSD' if t_raw > (t_slice_eval-t_raw) else 'np→mx変換/eval'} が支配")


if __name__ == "__main__":
    main()
