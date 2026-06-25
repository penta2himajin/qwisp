"""verify — streaming engine の出力が full モデルと一致するか（安全網）.

greedy 生成で full vs streaming のトークン列を比較。PoC 4.1 が層単位 bit 一致を示した上での
end-to-end 確認。full と streaming を順に（同時でなく）ロードしメモリ peak を抑える。

実行:
  PY="$HOME/Library/Application Support/MTPLX/runtime-venv/bin/python"
  "$PY" -m qwisp.verify "$MODEL" --gen 24
"""

from __future__ import annotations

import argparse
import sys

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import stream_generate
from mlx_lm.sample_utils import make_sampler

from .cache import ExpertCache
from .loader import load_streaming
from .streaming_moe import StreamingSwitchGLU


def greedy_tokens(model, tok, prompt_ids, gen):
    sampler = make_sampler(temp=0.0)
    toks = []
    for resp in stream_generate(model, tok, prompt=prompt_ids, max_tokens=gen, sampler=sampler):
        toks.append(resp.token)
    return toks


def main():
    ap = argparse.ArgumentParser(description="Qwisp streaming verify")
    ap.add_argument("model")
    ap.add_argument("--prompt", default="def fibonacci(n):\n    ")
    ap.add_argument("--gen", type=int, default=24)
    ap.add_argument("--budget", type=int, default=64)
    args = ap.parse_args()

    # full（先に）
    print("[verify] loading full model ...", file=sys.stderr)
    full, tok = load(args.model)
    ids = tok.encode(args.prompt)
    full_toks = greedy_tokens(full, tok, ids, args.gen)
    del full
    if hasattr(mx, "clear_cache"):
        mx.clear_cache()

    # streaming（cache 付き）
    print("[verify] loading streaming model ...", file=sys.stderr)
    model, tok2, src = load_streaming(args.model)
    cache = ExpertCache(src, budget_per_layer=args.budget)
    for _, mod in model.named_modules():
        if isinstance(mod, StreamingSwitchGLU):
            mod._cache = cache
    stream_toks = greedy_tokens(model, tok2, ids, args.gen)

    match = full_toks == stream_toks
    n_match = sum(1 for a, b in zip(full_toks, stream_toks) if a == b)
    print(f"\n[verify] full   : {full_toks}", file=sys.stderr)
    print(f"[verify] stream : {stream_toks}", file=sys.stderr)
    print(f"[verify] matched {n_match}/{len(full_toks)} tokens", file=sys.stderr)
    print(f"[verify] VERDICT: {'PASS — full と完全一致' if match else 'MISMATCH'}", file=sys.stderr)
    sys.exit(0 if match else 1)


if __name__ == "__main__":
    main()
