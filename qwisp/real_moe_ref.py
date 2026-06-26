"""実モデル layer-0 の MoE ブロック(Qwen3NextSparseMoeBlock)を REAL 量子化重みで検証 (M2b-3).

gate(8bit)→softmax precise→argpartition top8→normalize→switch_mlp(4bit gather_qmm)→combine
+ shared_expert(4bit dense)+shared_expert_gate(8bit→sigmoid)。lazy load で layer-0 のみ materialize。

実行: PY -m qwisp.real_moe_ref [--out /tmp/qwisp_real_moe_ref.safetensors --T 6]
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
    ap.add_argument("--out", default="/tmp/qwisp_real_moe_ref.safetensors")
    ap.add_argument("--T", type=int, default=6)
    args = ap.parse_args()

    model, _ = load(args.model, lazy=True)
    mlp = model.language_model.model.layers[0].mlp
    assert type(mlp).__name__ == "Qwen3NextSparseMoeBlock"

    T, H = args.T, 2048
    rng = np.random.default_rng(19)
    x = mx.array((rng.standard_normal((T, H)) * 1.0).astype(np.float32))
    mx.eval(x)

    y = mlp(x)          # 完全な MoE ブロック出力
    mx.eval(y)
    print(f"[real-moe] x={x.shape} y={y.shape} y.sum={float(mx.sum(y).item()):.6f} "
          f"topk={mlp.top_k} E={mlp.num_experts}")

    out = {"x": x, "y": y}
    qlin(mlp.gate, "gate", out)                       # 8bit
    qlin(mlp.shared_expert_gate, "shared_expert_gate", out)  # 8bit
    for p in ("gate_proj", "up_proj", "down_proj"):   # switch_mlp 4bit [E,...]
        qlin(getattr(mlp.switch_mlp, p), f"switch_mlp.{p}", out)
        qlin(getattr(mlp.shared_expert, p), f"shared_expert.{p}", out)  # 4bit dense

    mx.eval(list(out.values()))
    mx.save_safetensors(args.out, out)
    print(f"[real-moe] saved → {args.out}  (switch gate_proj={out['switch_mlp.gate_proj.weight'].shape})")


if __name__ == "__main__":
    main()
