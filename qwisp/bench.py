"""bench — streaming engine の実 tok/s・peak RSS・cache hit を予算 B 別に計測.

シミュ予測（Step2: B=48-64 で ~17-21 tok/s）を実機で裏取りする。

実行:
  PY="$HOME/Library/Application Support/MTPLX/runtime-venv/bin/python"
  "$PY" -m qwisp.bench "$MODEL" --budgets 32,64,128 --ctx 256 --gen 64
"""

from __future__ import annotations

import argparse
import resource
import sys
import time

import mlx.core as mx
from mlx_lm.generate import stream_generate
from mlx_lm.sample_utils import make_sampler

from .cache import ExpertCache
from .loader import load_streaming


def rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9


def make_prompt(tok, n):
    base = "def process(items):\n    out = []\n    for it in items:\n        out.append(it*2)\n    return out\n\n"
    ids = tok.encode(base)
    return (tok.encode(base * max(1, n // max(1, len(ids)))))[:n]


def bench_budget(model_dir, B, ctx, gen):
    src_holder = {}

    # 各 B で fresh にロード（cache 状態を分離）
    cache = None
    model, tok, src = load_streaming(model_dir)  # 一旦 source を得る
    cache = ExpertCache(src, budget_per_layer=B)
    # streaming_moe に cache を後付け
    from .streaming_moe import StreamingSwitchGLU
    for _, mod in model.named_modules():
        if isinstance(mod, StreamingSwitchGLU):
            mod._cache = cache

    sampler = make_sampler(temp=0.0)
    prompt = make_prompt(tok, ctx)
    cache.reset_stats()
    n = 0
    last = None
    t0 = time.perf_counter()
    for resp in stream_generate(model, tok, prompt=prompt, max_tokens=gen, sampler=sampler):
        n += 1
        last = resp
    wall = time.perf_counter() - t0
    dec_tps = getattr(last, "generation_tps", float("nan")) if last else float("nan")
    return {
        "B": B, "ctx": len(prompt), "gen": n,
        "decode_tps": dec_tps, "wall_s": wall,
        "hit_rate": cache.hit_rate, "resident_experts": cache.resident_experts(),
        "peak_rss": rss_gb(), "mx_peak": (mx.get_peak_memory() / 1e9),
        "dram_cache_gb": cache.resident_experts() * src.per_expert_bytes() / 1e9,
    }


def main():
    ap = argparse.ArgumentParser(description="Qwisp streaming bench")
    ap.add_argument("model")
    ap.add_argument("--budgets", default="32,64,128")
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--gen", type=int, default=64)
    args = ap.parse_args()

    hdr = (f"{'B/層':>5} {'ctx':>6} {'gen':>4} {'decode_tps':>11} {'hit':>6} "
           f"{'residExp':>9} {'cacheGB':>8} {'peakRSS':>8}")
    print(hdr)
    print("-" * len(hdr))
    for b in [int(x) for x in args.budgets.split(",")]:
        r = bench_budget(args.model, b, args.ctx, args.gen)
        print(f"{r['B']:>5} {r['ctx']:>6} {r['gen']:>4} {r['decode_tps']:>11.1f} "
              f"{r['hit_rate']:>6.3f} {r['resident_experts']:>9} "
              f"{r['dram_cache_gb']:>7.2f}G {r['peak_rss']:>7.2f}G", flush=True)


if __name__ == "__main__":
    main()
