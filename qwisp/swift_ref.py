"""Swift PoC のビット一致検証用の参照入出力を safetensors で吐く.

M1: gate_proj 単体の gather_qmm（x, inds, w/scales/biases, expected）。
M2a: switch_mlp 全体（gate/up/down の gather_qmm + swiglu）の expected_moe。
Swift は同じ重み・x・inds で再計算し、expected / expected_moe と一致するか検証する。

実行: PY -m qwisp.swift_ref [--out /tmp/qwisp_ref.safetensors]
"""
from __future__ import annotations
import argparse
import os

import numpy as np
import mlx.core as mx
from mlx_lm.models.activations import swiglu


def qmm(xe, w, s, b, inds):
    return mx.gather_qmm(xe, w, s, b, rhs_indices=inds, transpose=True,
                         group_size=64, bits=2, mode="affine", sorted_indices=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.expanduser("~/.mtplx/models/qwisp-experts-2bit"))
    ap.add_argument("--out", default="/tmp/qwisp_ref.safetensors")
    args = ap.parse_args()

    W = mx.load(os.path.join(args.dir, "experts_2bit.safetensors"))
    P = "language_model.model.layers.0.mlp.switch_mlp"
    g = {p: {q: W[f"{P}.{p}.{q}"] for q in ("weight", "scales", "biases")}
         for p in ("gate_proj", "up_proj", "down_proj")}
    mx.eval([v for d in g.values() for v in d.values()])
    E, OUT, _ = g["gate_proj"]["weight"].shape
    IN = 2048
    print(f"[ref] gate={g['gate_proj']['weight'].shape} up={g['up_proj']['weight'].shape} "
          f"down={g['down_proj']['weight'].shape}")

    T, K = 2, 8
    rng = np.random.default_rng(42)
    x = mx.array((rng.standard_normal((T, IN)) * 0.1).astype(np.float32))
    inds = mx.array(rng.integers(0, E, size=(T, K)).astype(np.int32))
    xe = mx.expand_dims(x, (-2, -3))

    # M1: gate 単体
    y_gate = qmm(xe, g["gate_proj"]["weight"], g["gate_proj"]["scales"], g["gate_proj"]["biases"], inds)
    y_gate = y_gate.reshape(T, K, OUT)

    # M2a: switch_mlp 全体 = down(swiglu(gate(x), up(x)))
    gate = qmm(xe, g["gate_proj"]["weight"], g["gate_proj"]["scales"], g["gate_proj"]["biases"], inds)
    up = qmm(xe, g["up_proj"]["weight"], g["up_proj"]["scales"], g["up_proj"]["biases"], inds)
    h = swiglu(gate, up)
    down = qmm(h, g["down_proj"]["weight"], g["down_proj"]["scales"], g["down_proj"]["biases"], inds)
    y_moe = down.squeeze(-2)                       # [T,K,H]
    mx.eval(y_gate, y_moe)
    print(f"[ref] gate out={y_gate.shape} sum={float(mx.sum(y_gate).item()):.6f}")
    print(f"[ref] moe  out={y_moe.shape} sum={float(mx.sum(y_moe).item()):.6f}")

    out = {
        "x": x, "inds": inds.astype(mx.int32),
        # M1 互換（gate を w/scales/biases として）
        "w": g["gate_proj"]["weight"], "scales": g["gate_proj"]["scales"],
        "biases": g["gate_proj"]["biases"], "expected": y_gate,
        # M2a: 全 proj + 期待 MoE 出力
        "expected_moe": y_moe,
    }
    for p in ("gate_proj", "up_proj", "down_proj"):
        for q in ("weight", "scales", "biases"):
            out[f"{p}.{q}"] = g[p][q]
    mx.save_safetensors(args.out, out)
    print(f"[ref] saved → {args.out}  (T={T} K={K} IN={IN} OUT={OUT} bits=2 gs=64)")


if __name__ == "__main__":
    main()
