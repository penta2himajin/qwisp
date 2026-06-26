"""GatedDeltaNet recurrent 核(gated_delta_update, use_kernel=False)の Swift 移植検証用参照.

M2b-1 の crux。Swift が同じ q/k/v/a/b/A_log/dt_bias で gated_delta_ops を再計算し、
out/state とビット一致するか検証する。実形状: Hk=16 Hv=32 Dk=Dv=128（config 由来）。

実行: PY -m qwisp.gdn_ref [--out /tmp/qwisp_gdn_ref.safetensors --T 4]
"""
from __future__ import annotations
import argparse

import numpy as np
import mlx.core as mx
from mlx_lm.models.gated_delta import gated_delta_update


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/qwisp_gdn_ref.safetensors")
    ap.add_argument("--T", type=int, default=4)
    args = ap.parse_args()
    B, T, Hk, Hv, Dk, Dv = 1, args.T, 16, 32, 128, 128
    rng = np.random.default_rng(7)

    def rnd(*shape):
        return mx.array((rng.standard_normal(shape) * 0.3).astype(np.float32))

    q = rnd(B, T, Hk, Dk)
    k = rnd(B, T, Hk, Dk)
    v = rnd(B, T, Hv, Dv)
    a = rnd(B, T, Hv)
    b = rnd(B, T, Hv)
    A_log = mx.array(np.log(rng.uniform(0.1, 16, size=Hv)).astype(np.float32))
    dt_bias = mx.array(np.ones(Hv, np.float32))
    mx.eval(q, k, v, a, b, A_log, dt_bias)

    out, state = gated_delta_update(q, k, v, a, b, A_log, dt_bias, None, None, use_kernel=False)
    mx.eval(out, state)
    print(f"[gdn] out={out.shape} state={state.shape}  out.sum={float(mx.sum(out).item()):.6f} "
          f"state.sum={float(mx.sum(state).item()):.6f}")

    mx.save_safetensors(args.out, {
        "q": q, "k": k, "v": v, "a": a, "b": b, "A_log": A_log, "dt_bias": dt_bias,
        "out": out, "state": state,
    })
    print(f"[gdn] saved → {args.out}  (B={B} T={T} Hk={Hk} Hv={Hv} Dk={Dk} Dv={Dv})")


if __name__ == "__main__":
    main()
