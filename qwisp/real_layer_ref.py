"""実モデル layer-0 の linear_attn サブブロックを REAL 量子化重みで検証する参照 (M2b-3 橋渡し).

lazy load で 20GB を materialize せず layer-0 だけ触る。input_layernorm + linear_attn
(GatedDeltaNet, 4bit量子化 in_proj/out_proj) を実重みで前向きし、Swift と bit 比較。
これで Swift の「量子化 Linear 経路 + GatedDeltaNetLayer」を実重みで一括検証する。

dump する量子化 Linear は weight(packed uint32)/scales/biases。Swift は quantizedMatmul
(group_size=64, bits=4, affine) で再計算。

実行: PY -m qwisp.real_layer_ref [--model <dir> --out /tmp/qwisp_real_layer_ref.safetensors --S 4]
"""
from __future__ import annotations
import argparse
import os

import numpy as np
import mlx.core as mx
from mlx_lm import load


def qlin_dump(mod, prefix, out):
    """QuantizedLinear の weight/scales/biases を dump（bits/group_size は config 既知 4/64）。"""
    out[f"{prefix}.weight"] = mod.weight
    out[f"{prefix}.scales"] = mod.scales
    out[f"{prefix}.biases"] = mod.biases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.expanduser(
        "~/.mtplx/models/Youssofal--Qwen3.6-35B-A3B-MTPLX-Optimized-Speed-FP16"))
    ap.add_argument("--out", default="/tmp/qwisp_real_layer_ref.safetensors")
    ap.add_argument("--S", type=int, default=4)
    args = ap.parse_args()

    model, _ = load(args.model, lazy=True)  # lazy: 触った tensor だけ materialize
    lm = model.language_model.model
    layer = lm.layers[0]
    assert layer.is_linear, "layer 0 は linear_attn のはず"
    la = layer.linear_attn

    B, S, H = 1, args.S, 2048
    rng = np.random.default_rng(17)
    x = mx.array((rng.standard_normal((B, S, H)) * 1.0).astype(np.float32))
    mx.eval(x)

    # r = linear_attn(input_layernorm(x))  （decoder layer の前半）
    xn = layer.input_layernorm(x)
    r = la(xn, mask=None, cache=None)
    mx.eval(xn, r)
    print(f"[real-layer] x={x.shape} r={r.shape} r.sum={float(mx.sum(r).item()):.6f}")

    out = {"x": x, "input_layernorm_weight": layer.input_layernorm.weight,
           "xn": xn, "r": r,
           "conv1d": la.conv1d.weight, "norm_weight": la.norm.weight,
           "A_log": la.A_log, "dt_bias": la.dt_bias}
    for name in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"):
        qlin_dump(getattr(la, name), name, out)

    mx.eval(list(out.values()))
    mx.save_safetensors(args.out, out)
    print(f"[real-layer] saved → {args.out}  (in_proj_qkv.weight={out['in_proj_qkv.weight'].shape} "
          f"dtype={out['in_proj_qkv.weight'].dtype})")


if __name__ == "__main__":
    main()
