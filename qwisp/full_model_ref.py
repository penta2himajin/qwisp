"""実モデルの FULL forward(40層, cache=None)の logits を dump して Swift と一致検証 (M2b-3).

実行: PY -m qwisp.full_model_ref [--out /tmp/qwisp_full_ref.safetensors --T 6]
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
    ap.add_argument("--out", default="/tmp/qwisp_full_ref.safetensors")
    ap.add_argument("--T", type=int, default=6)
    args = ap.parse_args()

    model, _ = load(args.model)  # full materialize（Python 側は普通にロード）
    rng = np.random.default_rng(31)
    ids = mx.array(rng.integers(0, 248320, size=(1, args.T)).astype(np.int32))

    # 手動で層を回して中間 hidden を捕捉（cache=None, mask は model と同じ）
    from mlx_lm.models.base import create_attention_mask, create_ssm_mask
    lm = model.language_model.model
    h = lm.embed_tokens(ids)
    dumps = {"ids": ids, "h_embed": h.astype(mx.float32)}
    fa_mask = create_attention_mask(h, None)
    ssm_mask = create_ssm_mask(h, None)
    for i, layer in enumerate(lm.layers):
        mask = ssm_mask if layer.is_linear else fa_mask
        h = layer(h, mask=mask, cache=None)
        if i in (0, 1, 3, 19, 39):
            dumps[f"h_after_{i}"] = h.astype(mx.float32)
    h = lm.norm(h)
    dumps["h_normed"] = h.astype(mx.float32)
    logits = model.language_model.lm_head(h)
    mx.eval(logits)
    dumps["logits"] = logits
    am = [int(mx.argmax(logits[0, t]).item()) for t in range(args.T)]
    print(f"[full] ids={ids.shape} logits={logits.shape} argmax={am} dtype={h.dtype}")

    # float32 パス（activations を f32 に: quantized_matmul は x dtype に従い f32 出力）
    h32 = lm.embed_tokens(ids).astype(mx.float32)
    for i, layer in enumerate(lm.layers):
        mask = ssm_mask if layer.is_linear else fa_mask
        h32 = layer(h32, mask=mask, cache=None)
    h32 = lm.norm(h32)
    logits32 = model.language_model.lm_head(h32)
    mx.eval(logits32)
    dumps["logits_f32"] = logits32
    am32 = [int(mx.argmax(logits32[0, t]).item()) for t in range(args.T)]
    print(f"[full] f32 argmax={am32} dtype={h32.dtype}")

    mx.save_safetensors(args.out, dumps)
    print(f"[full] saved → {args.out}")


if __name__ == "__main__":
    main()
