"""GatedDeltaNet *層* 全体(Qwen3NextGatedDeltaNet.__call__)の Swift 移植検証用参照.

M2b-1 の層 wrapping。recurrent 核(gated_delta_update)は gdn_ref.py で検証済。
こちらは核を包む in_proj_qkvz/in_proj_ba + fix_query_key_value_ordering +
grouped causal conv1d(+silu) + q/k rms_norm スケール + RMSNormGated(z) + out_proj を
丸ごと検証する。cache=None, mask=None の単一チャンク前向き。

use_kernel=False 経路に固定するため layer.train() で training=True にする
(__call__ 内 use_kernel=not self.training)。Swift 側は GatedDelta.update(=use_kernel False相当)。

実行: PY -m qwisp.gdn_layer_ref [--out /tmp/qwisp_gdn_layer_ref.safetensors --S 4]
"""
from __future__ import annotations
import argparse

import numpy as np
import mlx.core as mx
from mlx_lm.models.qwen3_next import Qwen3NextGatedDeltaNet, ModelArgs


def build_args() -> ModelArgs:
    return ModelArgs(
        model_type="qwen3_5_moe_text",
        hidden_size=2048,
        num_hidden_layers=40,
        intermediate_size=512,
        num_attention_heads=16,
        linear_num_value_heads=32,
        linear_num_key_heads=16,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        num_experts=256,
        num_experts_per_tok=8,
        decoder_sparse_step=1,
        shared_expert_intermediate_size=512,
        mlp_only_layers=[],
        moe_intermediate_size=512,
        rms_norm_eps=1e-6,
        vocab_size=248320,
        num_key_value_heads=2,
        rope_theta=10000000.0,
        partial_rotary_factor=0.25,
        max_position_embeddings=262144,
        head_dim=256,
        full_attention_interval=4,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/qwisp_gdn_layer_ref.safetensors")
    ap.add_argument("--S", type=int, default=4)
    args = ap.parse_args()
    B, S, H = 1, args.S, 2048
    rng = np.random.default_rng(11)

    layer = Qwen3NextGatedDeltaNet(build_args())
    layer.train()  # use_kernel=False 経路に固定

    # 既定の重みを決定論的に上書き(乱数依存を排除しビット比較を安定化)
    def rnd(shape, scale=0.02):
        return mx.array((rng.standard_normal(shape) * scale).astype(np.float32))

    layer.in_proj_qkvz.weight = rnd(layer.in_proj_qkvz.weight.shape)
    layer.in_proj_ba.weight = rnd(layer.in_proj_ba.weight.shape)
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
        "in_proj_qkvz": layer.in_proj_qkvz.weight,
        "in_proj_ba": layer.in_proj_ba.weight,
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
