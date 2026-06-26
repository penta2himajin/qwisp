"""end-to-end prefill 検証 — GPU-routed mixed-resident(16GB帯)の prefill partition 経路を実機で測る.

micro(twogather_micro)で per-layer 1.34-2.24x・bit一致を確認済。ここでは実 12GB モデルで
(1)partition ON/OFF の last-token logits が一致するか（実重み・実 shape での正しさ）
(2)prefill 時間が何倍速くなるか
を、同一ロードのまま GPURoutedMixedSwitchGLU.PREFILL_T をトグルして測る。

実行: PY -m qwisp.prefill_bench <4bit_model> <2bit_dir> [--ctx 1024 --hot 64 --reps 3]
"""
from __future__ import annotations
import argparse
import sys
import time

import mlx.core as mx

from .gpu_routed import load_gpu_routed, GPURoutedMixedSwitchGLU
from .mtp_decode import _fwd


def time_prefill(lm, ids, reps):
    best = float("inf")
    last_logits = None
    for _ in range(reps):
        kv = lm.make_cache()
        mx.eval(lm.parameters())
        t = time.perf_counter()
        h, lg = _fwd(lm, ids, kv)
        mx.eval(lg)
        dt = time.perf_counter() - t
        best = min(best, dt)
        last_logits = lg
    return best, last_logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("dir2")
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--hot", type=int, default=64)
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()

    print(f"[prefill] load_gpu_routed hot={args.hot} ...", file=sys.stderr)
    model, tok = load_gpu_routed(args.model, args.dir2, hot_b=args.hot)
    lm = model.language_model if hasattr(model, "language_model") else model

    base = "def quicksort(a):\n    if len(a)<=1: return a\n    p=a[len(a)//2]\n    return quicksort([x for x in a if x<p])+[p]+quicksort([x for x in a if x>p])\n"
    enc = tok.encode(base)
    ids = []
    while len(ids) < args.ctx:
        ids += enc
    ids = mx.array(ids[:args.ctx])[None]
    print(f"[prefill] ctx={args.ctx} tokens", file=sys.stderr)

    # partition OFF（旧 full two-gather: PREFILL_T を超大に）
    GPURoutedMixedSwitchGLU.PREFILL_T = 10**9
    t_off, lg_off = time_prefill(lm, ids, args.reps)

    # partition ON（既定 8）
    GPURoutedMixedSwitchGLU.PREFILL_T = 8
    t_on, lg_on = time_prefill(lm, ids, args.reps)

    # 正しさ: last-token logits 一致
    a = lg_on[0, -1]; b = lg_off[0, -1]
    diff = float(mx.max(mx.abs(a - b)).item())
    rel = diff / (float(mx.max(mx.abs(b)).item()) + 1e-9)
    am_on = int(mx.argmax(a).item()); am_off = int(mx.argmax(b).item())

    print(f"\n[prefill] ctx={args.ctx} hot={args.hot} reps={args.reps}")
    print(f"  full two-gather (旧) : {t_off*1e3:8.1f} ms")
    print(f"  partition       (新) : {t_on*1e3:8.1f} ms   ({t_off/t_on:.2f}x 速い)")
    print(f"  正しさ last-logits  : max|Δ|={diff:.2e} rel={rel:.2e} argmax {am_on}=={am_off} "
          f"{'OK' if (rel < 1e-3 and am_on == am_off) else 'MISMATCH'}")


if __name__ == "__main__":
    main()
