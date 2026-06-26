"""mix-streaming の実 cold-miss 率を測る — IO 税が「高 miss×SSD」か「低 miss＝別要因」かを確定.

io_probe: cold SSD=0.146ms/expert(帯域律速)。load_probe: pread はコピー律速。
残る鍵は「1 forward で実際に何個 cold をストリームするか」。cache 統計で hit/miss/forward を出し
実 IO 量 = misses × 0.146ms を full_profile の IO 税 52ms と突き合わせる。

実行: PY -m qwisp.stream_hitrate <4bit_model> <2bit_dir> [--hot 48 --cold-B 64 --gen 96]
"""
from __future__ import annotations
import argparse
import sys
import time

import mlx.core as mx
from mlx_lm.sample_utils import make_sampler
from mlx_lm.generate import stream_generate

from .mixed_engine import build_mixed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("dir2")
    ap.add_argument("--hot", type=int, default=48)
    ap.add_argument("--cold-B", type=int, default=64)
    ap.add_argument("--gen", type=int, default=96)
    ap.add_argument("--ctx", type=int, default=128)
    args = ap.parse_args()

    print(f"[hr] build_mixed hot={args.hot} cold-B={args.cold_B} ...", file=sys.stderr)
    # build_mixed は hot_b 引数のみ。cold-B は c2 予算 → 直接構築する。
    model, tok, c4, c2 = build_mixed(args.model, args.dir2, args.hot)
    c2.B = args.cold_B                                     # cold キャッシュ予算
    c4.B = args.hot

    base = "def quicksort(a):\n    if len(a)<=1: return a\n    p=a[len(a)//2]\n"
    ids = tok.encode(base)
    while len(ids) < args.ctx:
        ids = ids + tok.encode(base)
    prompt = ids[:args.ctx]

    s = make_sampler(temp=0.0)
    # warmup（prefill + 数トークン）でキャッシュを定常化してから統計リセット
    n = 0
    for r in stream_generate(model, tok, prompt=prompt, max_tokens=8, sampler=s):
        n += 1
    c2.reset_stats(); c4.reset_stats()
    t0 = time.perf_counter()
    steps = 0
    for r in stream_generate(model, tok, prompt=prompt, max_tokens=args.gen, sampler=s):
        steps += 1
    dt = time.perf_counter() - t0

    fwd = steps                                           # decode は 1 token/forward
    print(f"\n[hr] decode {steps} tok in {dt:.2f}s = {steps/dt:.1f} tok/s")
    print(f"  cold(c2): hit={c2.hits} miss={c2.misses} hit_rate={c2.hit_rate:.3f}  "
          f"resident={c2.resident_experts()}")
    print(f"  hot(c4):  hit={c4.hits} miss={c4.misses} hit_rate={c4.hit_rate:.3f}")
    miss_per_fwd = c2.misses / max(fwd, 1)
    print(f"  cold miss/forward = {miss_per_fwd:.1f}  → SSD IO ≈ {miss_per_fwd*0.146:.1f} ms/forward "
          f"(0.146ms/expert)")
    print(f"  全 expert アクセス/forward(層×top8 概算) = {40*8}（うち cold miss {miss_per_fwd:.1f}）")


if __name__ == "__main__":
    main()
