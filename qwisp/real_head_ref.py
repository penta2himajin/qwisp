"""embed_tokens(QuantizedEmbedding 4bit) + final norm + lm_head(QuantizedLinear 4bit) を
REAL 重みで検証 (M2b-3). full forward の入口と出口。

実行: PY -m qwisp.real_head_ref [--out /tmp/qwisp_head_ref.safetensors --T 5]
"""
from __future__ import annotations
import argparse
import os

import numpy as np
import mlx.core as mx
from mlx_lm import load


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.expanduser(
        "~/.mtplx/models/Youssofal--Qwen3.6-35B-A3B-MTPLX-Optimized-Speed-FP16"))
    ap.add_argument("--out", default="/tmp/qwisp_head_ref.safetensors")
    ap.add_argument("--T", type=int, default=5)
    args = ap.parse_args()

    model, _ = load(args.model, lazy=True)
    lm = model.language_model
    et = lm.model.embed_tokens
    norm = lm.model.norm
    head = lm.lm_head

    H = 2048
    rng = np.random.default_rng(29)
    ids = mx.array(rng.integers(0, 248320, size=(1, args.T)).astype(np.int32))
    embed_out = et(ids)                          # [1,T,H]
    h = mx.array((rng.standard_normal((1, args.T, H)) * 1.0).astype(np.float32))
    normed = norm(h)
    logits = head(normed)                        # [1,T,vocab]
    mx.eval(ids, embed_out, h, normed, logits)
    print(f"[head] ids={ids.shape} embed={embed_out.shape} logits={logits.shape} "
          f"argmax0={int(mx.argmax(logits[0,0]).item())}")

    out = {
        "ids": ids, "embed_out": embed_out, "h": h, "logits": logits,
        "norm_weight": norm.weight,
        "embed.weight": et.weight, "embed.scales": et.scales, "embed.biases": et.biases,
        "lm_head.weight": head.weight, "lm_head.scales": head.scales, "lm_head.biases": head.biases,
    }
    mx.eval(list(out.values()))
    mx.save_safetensors(args.out, out)
    print(f"[head] saved → {args.out}  (embed.weight={out['embed.weight'].shape})")


if __name__ == "__main__":
    main()
