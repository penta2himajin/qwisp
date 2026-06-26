"""Swift ArenaBench と同条件の Python 版 — concat 排除(arena)の効果を直接比較する.

40層 MoE forward を 2 変種で測る（T=2 K=8 B=64 2bit, Swift と同一）:
  A arena(concat無): 事前 stack 済 [B,...] を gather_qmm で引く＝GPU-routed 床（Python は resident のみ可）
  B concat(streaming): 毎層 選択 expert を concat → gather_qmm（Python streaming が払う税）
Swift arena は streaming しながら A の床に届く（M3 で in-place 更新が viable なため）。

実行: PY -m qwisp.arena_bench_py [--layers 40 --B 64 --reps 50]
"""
from __future__ import annotations
import argparse
import time

import numpy as np
import mlx.core as mx
from mlx_lm.models.activations import swiglu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=40)
    ap.add_argument("--T", type=int, default=2)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--B", type=int, default=64)
    ap.add_argument("--reps", type=int, default=50)
    args = ap.parse_args()
    L, T, K, B = args.layers, args.T, args.K, args.B
    IN, I = 2048, 512
    gs, bits = 64, 2

    def qproj(outd, ind):
        fp = mx.random.normal((B, outd, ind)) * 0.02
        w, s, b = mx.quantize(fp, group_size=gs, bits=bits)
        return w, s, b

    layers = []
    for _ in range(L):
        layers.append({p: qproj(*sh) for p, sh in
                       (("gate", (I, IN)), ("up", (I, IN)), ("down", (IN, I)))})
    mx.eval([t for ly in layers for pr in ly.values() for t in pr])

    x0 = mx.random.normal((T, IN)) * 0.1
    inds = mx.array(np.random.default_rng(0).integers(0, B, size=(T, K)).astype(np.int32))
    mx.eval(x0, inds)

    def qmm(xe, pr):
        w, s, b = pr
        return mx.gather_qmm(xe, w, s, b, rhs_indices=inds, transpose=True,
                             group_size=gs, bits=bits, mode="affine", sorted_indices=False)

    def moe(x, ly):
        xe = mx.expand_dims(x, (-2, -3))
        g = qmm(xe, ly["gate"]); u = qmm(xe, ly["up"])
        h = swiglu(g, u)
        d = qmm(h, ly["down"]).squeeze(-2)      # [T,K,H]
        return d.sum(axis=1)                    # K 合算 → [T,IN]

    # A: arena（concat 無, 事前 stack 済を gather）
    def fwd_arena():
        x = x0
        for ly in layers:
            x = moe(x, ly)
        return x

    # B: concat（毎層 選択 expert を concat してから gather）— streaming 税を模擬
    def moe_concat(x, ly):
        inds_np = np.asarray(inds.tolist())
        U, inv = np.unique(inds_np, return_inverse=True)
        Umx = mx.array(U.astype(np.int32))
        remap = mx.array(inv.reshape(inds_np.shape).astype(np.int32))
        sub = {}
        for p in ("gate", "up", "down"):
            w, s, b = ly[p]
            sub[p] = (mx.contiguous(w[Umx]), mx.contiguous(s[Umx]), mx.contiguous(b[Umx]))
        xe = mx.expand_dims(x, (-2, -3))
        g = mx.gather_qmm(xe, *sub["gate"], rhs_indices=remap, transpose=True, group_size=gs, bits=bits, mode="affine")
        u = mx.gather_qmm(xe, *sub["up"], rhs_indices=remap, transpose=True, group_size=gs, bits=bits, mode="affine")
        h = swiglu(g, u)
        d = mx.gather_qmm(h, *sub["down"], rhs_indices=remap, transpose=True, group_size=gs, bits=bits, mode="affine").squeeze(-2)
        return d.sum(axis=1)

    def fwd_concat():
        x = x0
        for ly in layers:
            x = moe_concat(x, ly)
        return x

    def bench(fn, warm=10):
        for _ in range(warm):
            mx.eval(fn())
        t = time.perf_counter()
        for _ in range(args.reps):
            mx.eval(fn())
        return (time.perf_counter() - t) / args.reps * 1e3

    ms_arena = bench(fwd_arena)
    ms_concat = bench(fwd_concat)
    print(f"\n[py-bench] {L}層 MoE forward (T={T} K={K} B={B}, 2bit), reps={args.reps}")
    print(f"  A arena (concat無, GPU-routed 床)  : {ms_arena:6.2f} ms/forward  ({1000/ms_arena:.0f} tok/s 相当)")
    print(f"  B concat (streaming 税)            : {ms_concat:6.2f} ms/forward  ({1000/ms_concat:.0f} tok/s 相当)")
    print(f"  → concat 税 = {ms_concat-ms_arena:+.2f} ms/forward（Swift arena はこれを払わない）")


if __name__ == "__main__":
    main()
