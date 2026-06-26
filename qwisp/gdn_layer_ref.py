"""GatedDeltaNet *層* 全体(qwen3_5.GatedDeltaNet.__call__)の Swift 移植検証用参照 (M2b-1).

★ 実モデルは qwen3_5_moe(=qwen3_5.py)で、qwen3_next.py とは GatedDeltaNet の構造が違う:
   in_proj を qkv / z / a / b の 4 本に分離（fix_query_key_value_ordering は無い）。
   実 checkpoint も in_proj_qkv/in_proj_z/in_proj_a/in_proj_b。

recurrent 核(gated_delta_update)は gdn_ref.py で検証済。こちらは核を包む
in_proj_qkv/z/b/a + grouped causal conv1d(+silu) + q/k rms_norm スケール +
RMSNormGated(z) + out_proj を丸ごと検証する。cache=None, mask=None の単一チャンク前向き。

use_kernel=False 経路固定のため layer.train()。Swift 側は GatedDelta.update。

実行: PY -m qwisp.gdn_layer_ref [--out /tmp/qwisp_gdn_layer_ref.safetensors --S 4]
"""
from __future__ import annotations
import argparse

import numpy as np
import mlx.core as mx
from mlx_lm.models.qwen3_5 import GatedDeltaNet, TextModelArgs


def build_text_args() -> TextModelArgs:
    return TextModelArgs.from_dict({
        "model_type": "qwen3_5_moe_text",
        "hidden_size": 2048,
        "num_hidden_layers": 40,
        "num_attention_heads": 16,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "linear_num_value_heads": 32,
        "linear_num_key_heads": 16,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "rms_norm_eps": 1e-6,
        "vocab_size": 248320,
        "num_experts": 256,
        "num_experts_per_tok": 8,
        "moe_intermediate_size": 512,
        "shared_expert_intermediate_size": 512,
        "norm_topk_prob": True,
        "full_attention_interval": 4,
        "rope_parameters": {"rope_type": "default", "rope_theta": 10000000,
                            "partial_rotary_factor": 0.25},
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/qwisp_gdn_layer_ref.safetensors")
    ap.add_argument("--S", type=int, default=4)
    args = ap.parse_args()
    B, S, H = 1, args.S, 2048
    rng = np.random.default_rng(11)

    layer = GatedDeltaNet(build_text_args())
    layer.train()  # use_kernel=False 経路に固定

    def rnd(shape, scale=0.02):
        return mx.array((rng.standard_normal(shape) * scale).astype(np.float32))

    layer.in_proj_qkv.weight = rnd(layer.in_proj_qkv.weight.shape)
    layer.in_proj_z.weight = rnd(layer.in_proj_z.weight.shape)
    layer.in_proj_b.weight = rnd(layer.in_proj_b.weight.shape)
    layer.in_proj_a.weight = rnd(layer.in_proj_a.weight.shape)
    layer.conv1d.weight = rnd(layer.conv1d.weight.shape, scale=0.1)
    layer.out_proj.weight = rnd(layer.out_proj.weight.shape)
    layer.norm.weight = mx.array((rng.uniform(0.8, 1.2, layer.norm.weight.shape)).astype(np.float32))
    A = rng.uniform(0.1, 16, size=layer.A_log.shape[0]).astype(np.float32)
    layer.A_log = mx.array(np.log(A))
    layer.dt_bias = mx.array(np.ones(layer.dt_bias.shape[0], np.float32))

    x = rnd((B, S, H), scale=1.0)
    mx.eval(x, layer.parameters())

    out = layer(x, mask=None, cache=None)
    mx.eval(out)
    print(f"[gdn-layer] x={x.shape} out={out.shape} out.sum={float(mx.sum(out).item()):.6f}")

    dump = {
        "x": x,
        "in_proj_qkv": layer.in_proj_qkv.weight,
        "in_proj_z": layer.in_proj_z.weight,
        "in_proj_b": layer.in_proj_b.weight,
        "in_proj_a": layer.in_proj_a.weight,
        "conv1d": layer.conv1d.weight,        # [conv_dim, K, 1]
        "out_proj": layer.out_proj.weight,
        "norm_weight": layer.norm.weight,     # [head_v_dim=128]
        "A_log": layer.A_log,
        "dt_bias": layer.dt_bias,
        "out": out,
    }
    mx.save_safetensors(args.out, dump)
    print(f"[gdn-layer] saved → {args.out}  (B={B} S={S} H={H} conv_dim={layer.conv_dim})")


if __name__ == "__main__":
    main()
