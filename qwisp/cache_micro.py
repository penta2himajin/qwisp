"""streaming decode 税の正体切り分け — concat(mx) か cache python bookkeeping か.

concat_micro で「mx.concatenate 自体は decode の U≤16 では ~0.5ms/層」と判明。docs/09 の
D-G=+90ms は concat だけでは説明できない。ExpertCache.gather の python 部
（tolist 相当の np.unique・lock×2・dict/LRU・list 内包）が 40 層×serial で効いている疑い。

ここでは実 ExpertCache を all-hit（store 事前充填, IO ゼロ）で回し:
  gather() 全体  vs  pure concat(mx)  vs  np.unique+remap(python)
を分け、decode 税が「python serial overhead」か「mx」かを確定する。

実行: PY -m qwisp.cache_micro [--U 16 --layers 40 --steps 200]
"""
from __future__ import annotations
import argparse
import time

import numpy as np
import mlx.core as mx

from .cache import ExpertCache
from .concat_micro import make_experts, PROJS, PARTS


class _DummySource:
    """all-hit 前提＝load は呼ばれない。呼ばれたら明示エラー。"""
    def load_expert_slices(self, layer, experts, pool):
        raise RuntimeError("miss が発生（all-hit のはず）")


def bench(fn, steps, warm=20):
    for _ in range(warm):
        r = fn(); mx.eval(r) if isinstance(r, dict) is False else mx.eval(list(r.values()))
    t = time.perf_counter()
    for _ in range(steps):
        r = fn()
        mx.eval(list(r.values()) if isinstance(r, dict) else r)
    return (time.perf_counter() - t) / steps * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--U", type=int, default=16)
    ap.add_argument("--layers", type=int, default=40)
    ap.add_argument("--steps", type=int, default=200)
    args = ap.parse_args()
    U = args.U

    exp, packed = make_experts()
    keys = [f"{p}.{q}" for p in PROJS for q in PARTS]

    # 実 ExpertCache を 1 層・all-hit で（store 事前充填, budget=256 で evict なし）
    c = ExpertCache(_DummySource(), budget_per_layer=256, io_workers=0)
    for e in range(256):
        c._store[(0, e)] = exp[e]
    Ulist = list(range(U))
    per = [exp[e] for e in Ulist]

    rng = np.random.default_rng(0)
    inds_np = rng.integers(0, U, size=(2, 8)).astype(np.int32)
    inds = mx.array(inds_np)

    # (a) gather() 全体（python bookkeeping + concat）
    def gather_full():
        return c.gather(0, Ulist)
    t_gather = bench(gather_full, args.steps)

    # (b) pure concat（mx だけ）
    def concat_only():
        return {k: mx.concatenate([s[k] for s in per], axis=0) for k in keys}
    t_concat = bench(concat_only, args.steps)

    # (c) routing python: tolist + np.unique + remap（StreamingSwitchGLU の per-layer python）
    def routing_py():
        a = np.asarray(inds.tolist())
        Ua, inv = np.unique(a, return_inverse=True)
        return mx.array(inv.reshape(a.shape).astype(np.int32))
    t_route = bench(routing_py, args.steps)

    # 1 層あたりの python bookkeeping = gather - concat
    book = t_gather - t_concat
    print(f"\n[cache] U={U}  all-hit  steps={args.steps}")
    print("-" * 50)
    print(f"  ExpertCache.gather() 全体    : {t_gather:7.3f} ms/層")
    print(f"  pure concat (mx)             : {t_concat:7.3f} ms/層")
    print(f"  gather の python bookkeeping  : {book:7.3f} ms/層  (= gather - concat)")
    print(f"  routing python (tolist+unique): {t_route:7.3f} ms/層  (StreamingSwitchGLU 側)")
    print("-" * 50)
    per_fwd_py = (book + t_route) * args.layers
    per_fwd_concat = t_concat * args.layers
    print(f"  ×{args.layers}層/forward:")
    print(f"    python serial (book+route) : {per_fwd_py:7.1f} ms/fwd")
    print(f"    concat (mx)                : {per_fwd_concat:7.1f} ms/fwd")
    print(f"  → decode 税の支配は "
          f"{'python serial' if per_fwd_py > per_fwd_concat else 'mx concat'}")


if __name__ == "__main__":
    main()
