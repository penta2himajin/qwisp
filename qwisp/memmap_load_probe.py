"""memmap fancy-index 経路 vs 現行 pread+concat 経路の速度比較.

mlx に選択読み API は無い(agent 調査)が、np.memmap の fancy-index は選択読み可(active+2.1MB 実測)。
memmap[idx] は [k,...] contiguous を numpy 1 op で作る＝per-expert pread ループ + mx.concatenate を
同時に置換しうる。streaming 税の concat(25ms)+np→mx(7ms) を縮められるか実測する。

実行: PY -m qwisp.memmap_load_probe [--k 16 --reps 30]
"""
from __future__ import annotations
import argparse
import json
import os
import struct
import time

import numpy as np
import mlx.core as mx

from .expert_source import ExpertSource, PROJS, PARTS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.expanduser("~/.mtplx/models/qwisp-experts-2bit"))
    ap.add_argument("--k", type=int, default=16, help="選択 expert 数/層")
    ap.add_argument("--layers", type=int, default=40)
    ap.add_argument("--reps", type=int, default=30)
    args = ap.parse_args()
    k = args.k
    keys = [f"{p}.{q}" for p in PROJS for q in PARTS]
    src = ExpertSource(args.dir)
    rng = np.random.default_rng(0)

    # safetensors header → 各 (proj,part) の [256,...] memmap を作る
    path = os.path.join(args.dir, "experts_2bit.safetensors")
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n)); ds = 8 + n
    _DTNP = {"U32": np.uint32, "F16": np.float16, "BF16": np.uint16, "F32": np.float32}
    maps = {}   # (layer,proj,part) -> np.memmap [256,...]
    for L in range(args.layers):
        for p in PROJS:
            for q in PARTS:
                key = f"language_model.model.layers.{L}.mlp.switch_mlp.{p}.{q}"
                t = hdr[key]; b, e = t["data_offsets"]
                maps[(L, p, q)] = np.memmap(path, dtype=_DTNP[t["dtype"]], mode="r",
                                            offset=ds + b, shape=tuple(t["shape"]))

    from concurrent.futures import ThreadPoolExecutor
    pool = ThreadPoolExecutor(max_workers=8)

    def current_path(L, idx):
        """現行: load_expert_slices(並列 pread) + concat → {proj.part: [k,...]}"""
        d = src.load_expert_slices(L, idx, pool)
        per = [d[e] for e in idx]
        return {kk: mx.concatenate([s[kk] for s in per], axis=0) for kk in keys}

    def memmap_path(L, idx):
        """memmap: 各 tensor を fancy-index → contiguous → mx.array（concat 不要）"""
        out = {}
        for p in PROJS:
            for q in PARTS:
                sub = np.ascontiguousarray(maps[(L, p, q)][idx])
                out[f"{p}.{q}"] = mx.array(sub)
        return out

    def bench(fn, warm=5):
        for _ in range(warm):
            idx = rng.integers(0, 256, k).tolist()
            mx.eval(list(fn(0, idx).values()))
        t = time.perf_counter()
        for _ in range(args.reps):
            idx = rng.integers(0, 256, k).tolist()
            r = fn(rng.integers(0, args.layers), idx)
            mx.eval(list(r.values()))
        return (time.perf_counter() - t) / args.reps * 1e3

    # 正しさ: 両経路が同じ bytes を返すか（同 idx, layer0）
    idx = [0, 5, 9, 13, 20, 40, 100, 200][:k] if k <= 8 else list(range(k))
    a = current_path(0, idx); b = memmap_path(0, idx)
    mx.eval(list(a.values()) + list(b.values()))
    ok = all(bool(mx.all(a[kk] == b[kk]).item()) for kk in keys)
    print(f"[mm] 正しさ（current vs memmap, 同 idx）: {'一致' if ok else '不一致!'}")

    t_cur = bench(current_path)
    t_mm = bench(memmap_path)
    print(f"\n[mm] k={k} expert/層, reps={args.reps}")
    print(f"  現行 pread+concat : {t_cur:7.3f} ms/層  → ×{args.layers}層 = {t_cur*args.layers:.1f} ms/fwd")
    print(f"  memmap fancy-index: {t_mm:7.3f} ms/層  → ×{args.layers}層 = {t_mm*args.layers:.1f} ms/fwd")
    print(f"  → memmap は {t_cur/t_mm:.2f}x {'速い' if t_mm < t_cur else '遅い'}"
          f"（差 {(t_cur-t_mm)*args.layers:+.1f} ms/fwd）")


if __name__ == "__main__":
    main()
