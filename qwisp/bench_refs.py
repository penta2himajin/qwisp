"""Generate the Qwisp measurement ref set (code/agentic/longctx/shortnl) in ONE model load.

Reproducible: prompts live in qwisp/bench_prompts.py (committed). Generated refs are written to
<repo>/refs/<regime>.safetensors (gitignored — regenerate with this script, do not commit).

Each ref stores {spec_prompt, spec_greedy} (spec_greedy = Python-4bit greedy; the Swift bench
recomputes its own f32-full greedy for the lossless check). This is what the Swift runners
(suffix-spec / bolt) consume via QWISP_MTP_REF.

Run (MTPLX runtime venv has mlx_lm):
  PY="$HOME/Library/Application Support/MTPLX/runtime-venv/bin/python3"
  PYTHONPATH=<repo> "$PY" -m qwisp.bench_refs <model_dir> [--nspec 128]
"""
from __future__ import annotations
import argparse
import os

import mlx.core as mx
from mlx_lm import load
import qwisp.mtp_decode as MD
from qwisp.bench_prompts import PROMPTS

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT = os.path.join(REPO_ROOT, "refs")


def build_ids(tok, text: str, ctx):
    ids = tok.encode(text)
    if ctx is None:
        return ids
    if len(ids) < ctx:                      # short regime: pad by repetition
        base = list(ids)
        while len(ids) < ctx:
            ids = ids + base
    return ids[:ctx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--nspec", type=int, default=128, help="greedy tokens per ref (measurement horizon)")
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    model, tok = load(args.model)
    lm = model.language_model

    for name, spec in PROMPTS.items():
        ids = build_ids(tok, spec["text"], spec["ctx"])
        g_out, _ = MD.greedy(lm, ids, args.nspec)
        path = os.path.join(args.out, f"{name}.safetensors")
        mx.save_safetensors(path, {
            "spec_prompt": mx.array(ids, mx.int32),
            "spec_greedy": mx.array(g_out, mx.int32),
        })
        print(f"[bench-ref] {name:8s} ctx={len(ids):5d} nspec={args.nspec} "
              f"src={spec['source']} -> {path}")
    print(f"[bench-ref] done: {len(PROMPTS)} refs in {args.out}")


if __name__ == "__main__":
    main()
