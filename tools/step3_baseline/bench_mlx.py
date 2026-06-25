#!/usr/bin/env python3
"""Qwisp Step 3② — 素の MLX ベースライン（streaming エンジンが超えるべき基準値）。

フロア候補機で素の mlx_lm（hook 無し・AR）を回し、context 長を振って
prefill/decode tok/s と peak memory を記録する。streaming 版の比較基準になる。

mlx_lm を持つ mtplx runtime-venv の python で実行:
    PY="$HOME/Library/Application Support/MTPLX/runtime-venv/bin/python"
    "$PY" bench_mlx.py --model "$HOME/.mtplx/models/Youssofal--...-FP16" \
        --ctx 128,2048,8192 --gen 64
"""

import argparse
import sys
import time

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import stream_generate
from mlx_lm.sample_utils import make_sampler


def peak_gb():
    for fn in ("get_peak_memory",):
        f = getattr(mx, fn, None)
        if f:
            return f() / 1e9
    metal = getattr(mx, "metal", None)
    if metal and getattr(metal, "get_peak_memory", None):
        return metal.get_peak_memory() / 1e9
    return float("nan")


def reset_peak():
    for obj in (mx, getattr(mx, "metal", None)):
        f = getattr(obj, "reset_peak_memory", None) if obj else None
        if f:
            f()
            return


def make_prompt(tokenizer, n_tokens):
    """おおよそ n_tokens のプロンプトを作る（繰り返しコード片で長さ調整）。"""
    base = ("# utility\ndef process(items):\n    out = []\n    for it in items:\n"
            "        out.append(it * 2)\n    return out\n\n")
    ids = tokenizer.encode(base)
    reps = max(1, n_tokens // max(1, len(ids)))
    text = base * reps
    ids = tokenizer.encode(text)[:n_tokens]
    return ids


def bench_one(model, tokenizer, ctx, gen):
    prompt_ids = make_prompt(tokenizer, ctx)
    sampler = make_sampler(temp=0.0)
    reset_peak()
    prompt_tps = gen_tps = float("nan")
    n = 0
    t_start = time.perf_counter()
    last = None
    for resp in stream_generate(model, tokenizer, prompt=prompt_ids,
                                max_tokens=gen, sampler=sampler):
        n += 1
        last = resp
    # mlx_lm の GenerationResponse は prompt_tps / generation_tps を持つ（版依存）
    if last is not None:
        prompt_tps = getattr(last, "prompt_tps", float("nan"))
        gen_tps = getattr(last, "generation_tps", float("nan"))
    wall = time.perf_counter() - t_start
    return {
        "ctx": len(prompt_ids), "gen": n,
        "prefill_tps": prompt_tps, "decode_tps": gen_tps,
        "peak_gb": peak_gb(), "wall_s": wall,
    }


def main():
    ap = argparse.ArgumentParser(description="Qwisp Step3 MLX baseline")
    ap.add_argument("--model", required=True)
    ap.add_argument("--ctx", default="128,2048,8192", help="context 長（カンマ区切り）")
    ap.add_argument("--gen", type=int, default=64)
    args = ap.parse_args()

    print(f"[mlx] loading {args.model} ...", file=sys.stderr)
    model, tokenizer = load(args.model)
    ctxs = [int(x) for x in args.ctx.split(",")]

    hdr = f"{'ctx':>7} {'gen':>5} {'prefill_tps':>12} {'decode_tps':>11} {'peak_GB':>8} {'wall_s':>8}"
    print(hdr)
    print("-" * len(hdr))
    for ctx in ctxs:
        r = bench_one(model, tokenizer, ctx, args.gen)
        print(f"{r['ctx']:>7} {r['gen']:>5} {r['prefill_tps']:>12.1f} "
              f"{r['decode_tps']:>11.1f} {r['peak_gb']:>8.2f} {r['wall_s']:>8.2f}", flush=True)

    print("\n[mlx] これが streaming 版が超えるべき素の AR 基準値（decode_tps）。", file=sys.stderr)
    print("[mlx] peak_GB が『フロア機に載るか』の実測。", file=sys.stderr)


if __name__ == "__main__":
    main()
