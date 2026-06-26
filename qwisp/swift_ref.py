"""Swift PoC のビット一致検証用に、gather_qmm の参照入出力を safetensors で吐く.

Swift 側は同じ 2bit store の layer0 gate_proj を load し、ここで保存した x/inds で gather_qmm を
再計算 → expected と一致するか検証する。これで「mlx-swift が Python mlx と同じ量子化 matmul を
ビット一致で出せる」ことを M1 で確認する。

実行: PY -m qwisp.swift_ref [--out /tmp/qwisp_ref.safetensors]
"""
from __future__ import annotations
import argparse
import os

import numpy as np
import mlx.core as mx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.expanduser("~/.mtplx/models/qwisp-experts-2bit"))
    ap.add_argument("--out", default="/tmp/qwisp_ref.safetensors")
    args = ap.parse_args()

    store = os.path.join(args.dir, "experts_2bit.safetensors")
    W = mx.load(store)
    pre = "language_model.model.layers.0.mlp.switch_mlp.gate_proj"
    w = W[f"{pre}.weight"]; s = W[f"{pre}.scales"]; b = W[f"{pre}.biases"]
    mx.eval(w, s, b)
    E, OUT, in_packed = w.shape           # [256, 512, 128]（2bit, in=2048）
    IN = 2048
    print(f"[ref] gate_proj.weight={w.shape} scales={s.shape} biases={b.shape}")

    T, K = 2, 8
    rng = np.random.default_rng(42)
    x = mx.array((rng.standard_normal((T, IN)) * 0.1).astype(np.float32))
    inds = mx.array(rng.integers(0, E, size=(T, K)).astype(np.int32))
    xe = mx.expand_dims(x, (-2, -3))                    # [T,1,1,IN]
    y = mx.gather_qmm(xe, w, s, b, rhs_indices=inds, transpose=True,
                      group_size=64, bits=2, mode="affine", sorted_indices=False)
    y = y.reshape(T, K, OUT)
    mx.eval(y)
    print(f"[ref] gather_qmm out={y.shape}  sum={float(mx.sum(y).item()):.6f}  "
          f"max={float(mx.max(mx.abs(y)).item()):.6f}")

    # Swift が再現するための入力 + 期待出力を保存（safetensors）
    mx.save_safetensors(args.out, {
        "x": x, "inds": inds.astype(mx.int32), "expected": y,
        # 参照用に weight も同梱（Swift は store から直接 load しても良いが自己完結のため）
        "w": w, "scales": s, "biases": b,
    })
    print(f"[ref] saved → {args.out}")
    print(f"[ref] meta: T={T} K={K} IN={IN} OUT={OUT} bits=2 gs=64 mode=affine transpose=True")


if __name__ == "__main__":
    main()
