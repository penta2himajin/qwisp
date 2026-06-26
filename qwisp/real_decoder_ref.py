"""実モデルの完全な DecoderLayer を REAL 量子化重みで検証 (M2b-3).

input_layernorm → (linear_attn | self_attn) → residual → post_attention_layernorm
→ mlp(MoE) → residual。linear 層(0)と full-attn 層(3)の両方を出せる。

実行: PY -m qwisp.real_decoder_ref --layer 0 --out /tmp/qwisp_dec0_ref.safetensors
      PY -m qwisp.real_decoder_ref --layer 3 --out /tmp/qwisp_dec3_ref.safetensors
"""
from __future__ import annotations
import argparse
import os

import numpy as np
import mlx.core as mx
from mlx_lm import load


def qlin(mod, prefix, out):
    out[f"{prefix}.weight"] = mod.weight
    out[f"{prefix}.scales"] = mod.scales
    out[f"{prefix}.biases"] = mod.biases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.expanduser(
        "~/.mtplx/models/Youssofal--Qwen3.6-35B-A3B-MTPLX-Optimized-Speed-FP16"))
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--S", type=int, default=4)
    args = ap.parse_args()
    out_path = args.out or f"/tmp/qwisp_dec{args.layer}_ref.safetensors"

    model, _ = load(args.model, lazy=True)
    layer = model.language_model.model.layers[args.layer]

    B, S, H = 1, args.S, 2048
    rng = np.random.default_rng(23 + args.layer)
    x = mx.array((rng.standard_normal((B, S, H)) * 1.0).astype(np.float32))
    mx.eval(x)

    # linear 層は ssm_mask(None), full 層は "causal"
    mask = None if layer.is_linear else "causal"
    y = layer(x, mask=mask, cache=None)
    mx.eval(y)
    print(f"[dec{args.layer}] is_linear={layer.is_linear} x={x.shape} y={y.shape} "
          f"y.sum={float(mx.sum(y).item()):.6f}")

    out = {"x": x, "y": y,
           "input_layernorm_weight": layer.input_layernorm.weight,
           "post_attention_layernorm_weight": layer.post_attention_layernorm.weight}

    if layer.is_linear:
        la = layer.linear_attn
        out["conv1d"] = la.conv1d.weight
        out["la_norm_weight"] = la.norm.weight
        out["A_log"] = la.A_log
        out["dt_bias"] = la.dt_bias
        for n in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"):
            qlin(getattr(la, n), n, out)
    else:
        a = layer.self_attn
        out["q_norm_weight"] = a.q_norm.weight
        out["k_norm_weight"] = a.k_norm.weight
        for n in ("q_proj", "k_proj", "v_proj", "o_proj"):
            qlin(getattr(a, n), n, out)

    mlp = layer.mlp
    qlin(mlp.gate, "gate", out)
    qlin(mlp.shared_expert_gate, "shared_expert_gate", out)
    for p in ("gate_proj", "up_proj", "down_proj"):
        qlin(getattr(mlp.switch_mlp, p), f"switch_mlp.{p}", out)
        qlin(getattr(mlp.shared_expert, p), f"shared_expert.{p}", out)

    mx.eval(list(out.values()))
    mx.save_safetensors(out_path, out)
    print(f"[dec{args.layer}] saved → {out_path}")


if __name__ == "__main__":
    main()
