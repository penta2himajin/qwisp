"""verify forward の内訳プロファイル — どこを削れば速くなるか特定する.

MTP 投機の 97% を占める verify(main 2-token forward) を分解:
  - per-layer の inds.tolist() 同期（GPU 直列化）= churn コスト
  - cache.gather の IO（cold miss の pread）
  - lm_head（vocab 248320）の割合
  - masked-combine の 2× gather コスト

実行: PY -m qwisp.verify_profile <model> <2bit_dir> [--hot 64 --cold-B 37 --ctx 512]
"""
from __future__ import annotations
import argparse
import sys
import time

import numpy as np
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import KVCache

from .mtp_decode import attach_mixed, _fwd
from . import mixed_engine


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("dir2")
    ap.add_argument("--hot", type=int, default=64)
    ap.add_argument("--cold-B", type=int, default=37)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--fast-hot", action="store_true", help="持続 hot バッファ + GPU remap")
    args = ap.parse_args()

    model, tok = load(args.model)
    lm = model.language_model
    caches = attach_mixed(model, lm, tok, args.model, args.dir2, args.hot, args.cold_B,
                          fast_hot=args.fast_hot)
    c4, c2 = caches

    # MixedSwitchGLU を計測ラップ（tolist 同期時間 / gather IO 時間 / miss 数）
    P = {"tolist": 0.0, "gather": 0.0, "n": 0, "miss0": 0}
    SG = mixed_engine.MixedSwitchGLU
    orig_side = SG._side
    orig_call = SG.__call__

    def timed_side(self, x, inds_np, sel, cache, bits):
        t = time.perf_counter()
        # cache.gather を測るため _grp 内 gather をフック…ここでは _side 全体を gather 近似計上
        r = orig_side(self, x, inds_np, sel, cache, bits)
        P["gather"] += time.perf_counter() - t
        return r

    def timed_call(self, x, inds):
        t = time.perf_counter()
        _ = np.asarray(inds.tolist())            # 同期コスト（GPU flush 含む）
        P["tolist"] += time.perf_counter() - t
        P["n"] += 1
        return orig_call(self, x, inds)

    SG._side = timed_side
    SG.__call__ = timed_call

    # プロンプト & warm
    base = "def quicksort(a):\n    if len(a)<=1: return a\n    p=a[len(a)//2]\n"
    ids = tok.encode(base)
    while len(ids) < args.ctx:
        ids = ids + tok.encode(base)
    prompt = ids[:args.ctx]
    main_kv = lm.make_cache()
    H, lg = _fwd(lm, mx.array(prompt)[None], main_kv); mx.eval(lg)
    u = int(mx.argmax(lg[0, -1]).item())
    d = int(mx.argmax(lg[0, -1]).item())

    # 計測: verify forward(2-token) × steps、lm_head 有/無、CPU内訳
    for c in (c4, c2):
        c.reset_stats()
    P.update({"tolist": 0.0, "gather": 0.0, "n": 0})
    t_full = t_model = 0.0
    for i in range(args.steps):
        toks = mx.array([[u, d]])
        t = time.perf_counter()
        h = lm.model(toks, cache=main_kv); mx.eval(h)
        t_model += time.perf_counter() - t
        t = time.perf_counter()
        lg = lm.lm_head(h); mx.eval(lg)
        t_full += time.perf_counter() - t
        u = int(mx.argmax(lg[0, -1]).item())
        d = u
    SG._side = orig_side; SG.__call__ = orig_call

    n_fwd = args.steps
    hit = c4.hits + c2.hits; miss = c4.misses + c2.misses
    print(f"\n[vprof] {n_fwd} verify forwards (2-token), ctx={args.ctx}, "
          f"hot={args.hot}/cold-B={args.cold_B}")
    print(f"  model fwd (40 layers) : {t_model/n_fwd*1e3:7.1f} ms/fwd")
    print(f"  lm_head (vocab 248320): {t_full/n_fwd*1e3:7.1f} ms/fwd")
    print(f"  -- within model fwd (CPU側, inline) --")
    print(f"  per-layer tolist sync : {P['tolist']/n_fwd*1e3:7.1f} ms/fwd "
          f"({P['tolist']/(t_model)*100:.0f}% of model, {P['n']/n_fwd:.0f} layers/fwd)")
    print(f"  cache.gather (IO+setup): {P['gather']/n_fwd*1e3:7.1f} ms/fwd")
    print(f"  cache hit rate        : {hit/(hit+miss):.3f}  (miss/fwd={miss/n_fwd:.0f})")
    print(f"  total verify          : {(t_model+t_full)/n_fwd*1e3:7.1f} ms/fwd "
          f"= {2*n_fwd/(t_model+t_full):.1f} tok/s(2/fwd)")


if __name__ == "__main__":
    main()
