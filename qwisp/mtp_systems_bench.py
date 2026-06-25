"""Stage B — MTP×mixed の systems 実測（実エンジン、no-prefetch=serial ブラケット）.

問い: 制約 RAM の mixed streaming で、verify 窓 W=2(D1) の実 per-forward レイテンシは
W=1(AR) の何倍か。sim は union-miss の flash コストでこれを予測した（不確実部）。実機で測る。

net_tps（受理率は MTPLX 実測 graft、このモデルの mtplx_runtime.json: accept(D1)=0.886）:
  D0 = 1 / T(W=1)              （MTP 無し）
  D1 = accepted(D1) / T(W=2)   accepted = 1 + 0.886 = 1.886
実 forward は同期 pread＝prefetch 無し ⇒ sim の **serial** ブラケットに対応。

手順: 実トークン列 S を greedy 生成 → 各 (config, W) で cache を cold start し prompt prefill で
温め、S を W 窓で食わせ各 forward を mx.eval+計測。steady 部の平均を取る。

実行:
  PY -m qwisp.mtp_systems_bench <4bit_model> <2bit_dir> \
     --all4-B 56 --hot 64 --cold-B 96 --ctx 512 --measure 192
"""
from __future__ import annotations
import argparse
import re
import sys
import time

import numpy as np
import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.generate import stream_generate
from mlx_lm.sample_utils import make_sampler
from mlx_lm.models.qwen3_next import Qwen3NextSparseMoeBlock

from .expert_source import ExpertSource
from .cache import ExpertCache
from .loader import load_streaming
from .streaming_moe import StreamingSwitchGLU
from .mixed_engine import MixedSwitchGLU, _calibrate

_LAYER_RE = re.compile(r"\.layers\.(\d+)\.")
ACCEPT_D1 = 0.886            # mtplx_runtime.json acceptance_by_depth[0]
B4, B2 = 1769472, 983040


def build_all4(model_dir, B):
    model, tok, src = load_streaming(model_dir)
    cache = ExpertCache(src, budget_per_layer=B)
    for _, m in model.named_modules():
        if isinstance(m, StreamingSwitchGLU):
            m._cache = cache
    return model, tok, [cache], B * 40 * B4 / 1e9


def build_mixed(model_dir, dir2, hot_b, cold_b):
    model, tok, src4 = load_streaming(model_dir)
    src2 = ExpertSource(dir2)
    c4 = ExpertCache(src4, budget_per_layer=hot_b)    # hot 全常駐
    c2 = ExpertCache(src2, budget_per_layer=cold_b)   # cold 制約 → 実 miss
    counts = _calibrate(model, tok)
    for name, blk in model.named_modules():
        if isinstance(blk, Qwen3NextSparseMoeBlock):
            m = _LAYER_RE.search(name)
            layer = int(m.group(1)) if m else 0
            c = counts.get(id(blk), np.zeros(256, np.int64))
            hot = set(np.argsort(c)[-hot_b:].tolist())
            blk.switch_mlp = MixedSwitchGLU(layer, hot, c4, c2)
    gb = (hot_b * B4 + cold_b * B2) * 40 / 1e9
    return model, tok, [c4, c2], gb


def measure(model, caches, prompt, S, W, warmup=16):
    """cold start → prefill → S を W 窓で食わせ各 forward を計測。平均 forward 秒と per-token。"""
    for c in caches:
        c.clear()
    kv = make_prompt_cache(model)
    mx.eval(model(mx.array(prompt)[None], cache=kv))     # prefill 温め
    for c in caches:
        c.reset_stats()
    times = []
    for i in range(0, len(S) - W + 1, W):
        chunk = mx.array(S[i:i + W])[None]
        t0 = time.perf_counter()
        logits = model(chunk, cache=kv)
        mx.eval(logits)
        times.append(time.perf_counter() - t0)
    times = times[warmup:]                                # steady 部
    mean_fwd = float(np.mean(times))
    hits = sum(c.hits for c in caches); miss = sum(c.misses for c in caches)
    hr = hits / (hits + miss) if (hits + miss) else 0.0
    return mean_fwd, hr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("dir2")
    ap.add_argument("--all4-B", type=int, default=56)
    ap.add_argument("--hot", type=int, default=64)
    ap.add_argument("--cold-B", type=int, default=96)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--measure", type=int, default=192, help="計測トークン長 S")
    args = ap.parse_args()

    acc1 = 1.0 + ACCEPT_D1

    # 実トークン列 S を all4 で greedy 生成（in-distribution な routing 用）
    print("[sysb] generating reference sequence ...", file=sys.stderr)
    model, tok, caches, gb = build_all4(args.model, 256)
    base = ("def merge_sort(arr):\n    \"\"\"Sort a list with merge sort and return it.\"\"\"\n"
            "    if len(arr) <= 1:\n        return arr\n")
    prompt = tok.encode(base)
    while len(prompt) < args.ctx:
        prompt = prompt + tok.encode(base)
    prompt = prompt[:args.ctx]
    s = make_sampler(temp=0.0)
    S = []
    for r in stream_generate(model, tok, prompt=prompt, max_tokens=args.measure + 32, sampler=s):
        S.append(r.token)
    del model, caches
    if hasattr(mx, "clear_cache"):
        mx.clear_cache()

    rows = []
    configs = [
        ("all4", lambda: build_all4(args.model, args.all4_B)),
        ("mixed", lambda: build_mixed(args.model, args.dir2, args.hot, args.cold_B)),
    ]
    for name, builder in configs:
        print(f"[sysb] building {name} ...", file=sys.stderr)
        model, tok, caches, gb = builder()
        t1, hr1 = measure(model, caches, prompt, S, 1)
        t2, hr2 = measure(model, caches, prompt, S, 2)
        d0 = 1.0 / t1
        d1 = acc1 / t2
        rows.append((name, gb, t1, t2, hr1, hr2, d0, d1))
        del model, caches
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()

    print(f"\n[sysb] accepted(D1)={acc1:.3f}  ctx={args.ctx} S={len(S)}  (serial/no-prefetch bracket)")
    hdr = (f"{'cfg':6} {'GB':>5} {'T(W1)':>8} {'T(W2)':>8} {'W2/W1':>6} "
           f"{'hit1':>5} {'hit2':>5} {'D0 tps':>7} {'D1 tps':>7} {'D1/D0':>6}")
    print(hdr); print("-" * len(hdr))
    for name, gb, t1, t2, hr1, hr2, d0, d1 in rows:
        print(f"{name:6} {gb:>4.1f}G {t1*1e3:>7.1f}m {t2*1e3:>7.1f}m {t2/t1:>6.2f} "
              f"{hr1:>5.2f} {hr2:>5.2f} {d0:>7.1f} {d1:>7.1f} {d1/d0:>6.2f}")
    # クロス比較: mixed+MTP(D1) vs all4-noMTP(D0)
    a = next(r for r in rows if r[0] == "all4")
    m = next(r for r in rows if r[0] == "mixed")
    print(f"\n[sysb] mixed+MTP(D1)={m[7]:.1f} vs all4-noMTP(D0)={a[6]:.1f} tps "
          f"= {m[7]/a[6]:.2f}x （同程度 RAM: all4 {a[1]:.1f}G / mixed {m[1]:.1f}G）")


if __name__ == "__main__":
    main()
