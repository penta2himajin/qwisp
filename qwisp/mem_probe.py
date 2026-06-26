"""GPU-routed mixed の常駐メモリ実測 — 16GB Mac に収まるか（task 3, docs/09）.

lean ローダで全 18GB を経由せず構築し、peak RSS と文脈長別の KV を測る。
KV 量子化（kv_bits）で 64K 文脈を 16GB に収められるかを確認する。

実行: PY -m qwisp.mem_probe <4bit_model> <2bit_dir> [--ctx 8192 --kv-bits 8]
"""
from __future__ import annotations
import argparse
import resource
import sys
import time

import mlx.core as mx
from mlx_lm.generate import stream_generate
from mlx_lm.sample_utils import make_sampler

from .gpu_routed import load_gpu_routed


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9  # macOS: bytes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("dir2")
    ap.add_argument("--hot", type=int, default=64)
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--gen", type=int, default=64)
    ap.add_argument("--kv-bits", type=int, default=0, help="0=f16 / 8 / 4")
    args = ap.parse_args()

    print(f"[mem] before load: RSS={rss_gb():.2f} GB", file=sys.stderr)
    t0 = time.perf_counter()
    model, tok = load_gpu_routed(args.model, args.dir2, args.hot)
    # forward を1回流して常駐を確定
    base = "def f(x):\n    return x*2\n"
    ids = tok.encode(base)
    while len(ids) < args.ctx:
        ids = ids + tok.encode(base)
    prompt = ids[:args.ctx]
    if hasattr(mx, "clear_cache"):
        mx.clear_cache()
    model_set = mx.get_active_memory() / 1e9              # forward 前＝モデル常駐のみ
    print(f"[mem] model resident only: {model_set:.2f} GB  ({time.perf_counter()-t0:.0f}s)",
          file=sys.stderr)

    # stream_generate に chunked prefill+decode させ peak を測る（kv_bits で KV 量子化）
    if hasattr(mx, "reset_peak_memory"):
        mx.reset_peak_memory()
    kw = dict(prefill_step_size=512)                      # chunked prefill で transient を抑制
    if args.kv_bits:
        kw.update(kv_bits=args.kv_bits, quantized_kv_start=0)
    s = make_sampler(temp=0.0)
    last = None
    for r in stream_generate(model, tok, prompt=prompt, max_tokens=args.gen, sampler=s, **kw):
        last = r
    build_rss = model_set
    tps = getattr(last, "generation_tps", float("nan"))
    peak = rss_gb()
    print(f"\n[mem] === GPU-routed mixed, ctx={args.ctx}, kv_bits={args.kv_bits or 'f16'} ===")
    print(f"  resident (build)  : {build_rss:.2f} GB")
    print(f"  peak (with KV+gen): {peak:.2f} GB  (mx_peak={mx.get_peak_memory()/1e9:.2f})")
    print(f"  decode tps        : {tps:.1f} tok/s")
    fits = peak < 15.0
    print(f"  16GB 判定         : {'FIT' if fits else 'NG'}（OS 余裕に ~1GB 残すなら <15GB 目安）")


if __name__ == "__main__":
    main()
