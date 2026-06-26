"""Qwen3NextAttention (full-attention 層) の Swift 移植検証用参照 (M2b-2).

partial RoPE(factor=0.25 → 64dim) + gated output(o_proj(out*sigmoid(gate))) +
GQA(16 q-heads / 2 kv-heads, head_dim=256) + q/k RMSNorm。
cache=None, mask="causal" の単一チャンク前向き。

実行: PY -m qwisp.attn_ref [--out /tmp/qwisp_attn_ref.safetensors --L 6]
"""
from __future__ import annotations
import argparse

import numpy as np
import mlx.core as mx
from mlx_lm.models.qwen3_next import Qwen3NextAttention
from qwisp.gdn_layer_ref import build_args


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/qwisp_attn_ref.safetensors")
    ap.add_argument("--L", type=int, default=6)
    args = ap.parse_args()
    B, L, H = 1, args.L, 2048
    rng = np.random.default_rng(13)

    attn = Qwen3NextAttention(build_args())

    def rnd(shape, scale=0.02):
        return mx.array((rng.standard_normal(shape) * scale).astype(np.float32))

    attn.q_proj.weight = rnd(attn.q_proj.weight.shape)
    attn.k_proj.weight = rnd(attn.k_proj.weight.shape)
    attn.v_proj.weight = rnd(attn.v_proj.weight.shape)
    attn.o_proj.weight = rnd(attn.o_proj.weight.shape)
    attn.q_norm.weight = mx.array(rng.uniform(0.8, 1.2, attn.q_norm.weight.shape).astype(np.float32))
    attn.k_norm.weight = mx.array(rng.uniform(0.8, 1.2, attn.k_norm.weight.shape).astype(np.float32))

    x = rnd((B, L, H), scale=1.0)
    mx.eval(x, attn.parameters())

    out = attn(x, mask="causal", cache=None)
    mx.eval(out)
    print(f"[attn] x={x.shape} out={out.shape} out.sum={float(mx.sum(out).item()):.6f} "
          f"rope_dim={int(256*0.25)}")

    mx.save_safetensors(args.out, {
        "x": x,
        "q_proj": attn.q_proj.weight,
        "k_proj": attn.k_proj.weight,
        "v_proj": attn.v_proj.weight,
        "o_proj": attn.o_proj.weight,
        "q_norm": attn.q_norm.weight,
        "k_norm": attn.k_norm.weight,
        "out": out,
    })
    print(f"[attn] saved → {args.out}  (B={B} L={L} H={H})")


if __name__ == "__main__":
    main()
