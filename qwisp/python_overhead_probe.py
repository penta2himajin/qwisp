"""Python 固有オーバーヘッドの分離測定 — コンパイル言語(mlx-swift/C++)で削れる分はどれか.

切り分け:
  sync 27ms の内訳 = 純 GPU drain（言語非依存） + tolist の Python list 構築/転送（言語依存）
  forward の per-op dispatch = Python→pybind の呼び出しコスト（言語依存）
GPU drain・concat・gather_qmm 計算・IO は Metal/mlx-core の仕事＝言語を変えても不変。

実行: PY -m qwisp.python_overhead_probe [--layers 40 --reps 30]
"""
from __future__ import annotations
import argparse
import time

import numpy as np
import mlx.core as mx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=40)
    ap.add_argument("--reps", type=int, default=30)
    args = ap.parse_args()
    L = args.layers

    # 各「層」が realistic な GPU 仕事（小 matmul）→ inds を出す。drain 対象の pending graph を作る。
    H = 2048
    W = mx.random.normal((H, H)) * 0.02
    x0 = mx.random.normal((1, H))
    mx.eval(W, x0)

    def layer(x):
        h = x @ W                                   # GPU 仕事
        inds = mx.argpartition(h, kth=-8, axis=-1)[..., -8:]
        return h, inds

    def bench(fn, warm=5):
        for _ in range(warm):
            fn()
        t = time.perf_counter()
        for _ in range(args.reps):
            fn()
        return (time.perf_counter() - t) / args.reps * 1e3

    # A: 同期なし（pipeline）— 末尾だけ eval。GPU-routed 相当の床。
    def A_no_sync():
        x = x0
        for _ in range(L):
            x, inds = layer(x)
        mx.eval(x)

    # B: 各層 mx.eval(inds)（GPU drain のみ、Python list 構築なし）
    def B_eval():
        x = x0
        for _ in range(L):
            x, inds = layer(x)
            mx.eval(inds)
        mx.eval(x)

    # C: 各層 inds.tolist()（drain + Python list 構築 + 転送）
    def C_tolist():
        x = x0
        for _ in range(L):
            x, inds = layer(x)
            _ = inds.tolist()
        mx.eval(x)

    # D: 各層 np.asarray(inds.tolist())（現行 engine と同じ numpy 化）
    def D_nparray():
        x = x0
        for _ in range(L):
            x, inds = layer(x)
            _ = np.asarray(inds.tolist())
        mx.eval(x)

    tA = bench(A_no_sync); tB = bench(B_eval); tC = bench(C_tolist); tD = bench(D_nparray)
    print(f"\n[pyov] {L} 層, reps={args.reps}  (小matmul/層で realistic drain)")
    print("-" * 56)
    print(f"  A 同期なし(pipeline床)        : {tA:7.3f} ms")
    print(f"  B mx.eval(inds)/層(GPU drain) : {tB:7.3f} ms   (B-A = 純 drain {tB-tA:+.3f})")
    print(f"  C inds.tolist()/層           : {tC:7.3f} ms   (C-B = Python list+転送 {tC-tB:+.3f})")
    print(f"  D np.asarray(tolist())/層     : {tD:7.3f} ms   (D-C = numpy 化 {tD-tC:+.3f})")
    print(f"  → sync の言語非依存分(drain) = {tB-tA:.2f}ms / Python 依存分(list+np) = {tD-tB:.2f}ms")

    # ② per-op dispatch: グラフ構築（eval なし）の Python コスト
    NOPS = 2000
    def build_graph():
        x = x0
        for _ in range(NOPS):
            x = x + 1.0                              # 1 op/回, eval しない＝純 dispatch
        return x
    for _ in range(5):
        build_graph()
    t = time.perf_counter()
    for _ in range(args.reps):
        build_graph()
    per_op = (time.perf_counter() - t) / args.reps / NOPS * 1e6
    print(f"\n[pyov] mx op の Python dispatch: {per_op:.2f} µs/op")
    # forward の概算 op 数: 40層 × (~15 MoE ops + ~10 attn ops) ≈ 1000
    est_ops = L * 25
    print(f"  forward ~{est_ops} op 概算 → dispatch 計 {per_op*est_ops/1e3:.2f} ms（コンパイル言語で削減対象）")


if __name__ == "__main__":
    main()
