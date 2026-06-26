"""mmap 遅延ページングの make-or-break テスト — gather_qmm は触れた expert だけ resident にするか.

仮説: 2bit store を mx.load(mmap) し [256,...] 量子化バッファを GPU-routed のまま gather_qmm すれば
OS が触れた expert ページだけ resident にし cold を evict＝sync/concat ゼロで 8GB streaming が成立。
検証: 全 9.4GB を mmap → 数 expert だけ gather_qmm → RSS が小さいまま(=遅延ページング成功)か
9.4GB に膨らむ(=mlx が全materialize＝不成立)か を RSS と mlx active memory で測る。

実行: PY -m qwisp.mmap_probe [--dir ~/.mtplx/models/qwisp-experts-2bit]
"""
from __future__ import annotations
import argparse
import os
import resource
import time

import numpy as np
import mlx.core as mx


def rss_gb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1e9 if r > 1e7 else r / 1e6   # mac=bytes, linux=KB 近似


def active_gb():
    try:
        return mx.get_active_memory() / 1e9
    except Exception:
        return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.expanduser("~/.mtplx/models/qwisp-experts-2bit"))
    args = ap.parse_args()
    path = os.path.join(args.dir, "experts_2bit.safetensors")
    sz = os.path.getsize(path) / 1e9
    print(f"[mmap] store = {sz:.1f} GB")
    print(f"[mmap] RSS before load = {rss_gb():.2f} GB  active = {active_gb():.2f} GB")

    t = time.perf_counter()
    W = mx.load(path)                                    # safetensors → mmap (lazy のはず)
    print(f"[mmap] mx.load done in {(time.perf_counter()-t)*1e3:.0f}ms  "
          f"keys={len(W)}  RSS={rss_gb():.2f}GB active={active_gb():.2f}GB")

    # layer0 の量子化バッファ [256, ...]
    pre = "language_model.model.layers.0.mlp.switch_mlp"
    w = W[f"{pre}.gate_proj.weight"]; s = W[f"{pre}.gate_proj.scales"]; b = W[f"{pre}.gate_proj.biases"]
    print(f"[mmap] gate_proj.weight shape={w.shape} dtype={w.dtype}  "
          f"(全256 expert, {w.nbytes/1e6:.0f}MB)")

    # gather_qmm: 8 expert だけ触る（remap）。x=[2,H]。
    H = 2048
    x = mx.random.normal((2, H))
    inds = mx.array(np.array([[0, 5, 9, 13, 20, 40, 100, 200],
                              [1, 5, 9, 13, 20, 40, 100, 200]], np.int32))
    xe = mx.expand_dims(x, (-2, -3))
    print(f"[mmap] RSS just before gather_qmm = {rss_gb():.2f}GB active={active_gb():.2f}GB")
    t = time.perf_counter()
    y = mx.gather_qmm(xe, w, s, b, rhs_indices=inds, transpose=True,
                      group_size=64, bits=2, mode="affine", sorted_indices=False)
    mx.eval(y)
    dt = (time.perf_counter() - t) * 1e3
    print(f"[mmap] gather_qmm(8 expert) eval {dt:.1f}ms  "
          f"RSS={rss_gb():.2f}GB active={active_gb():.2f}GB  out={y.shape}")

    # 判定: 8 expert(数MB)だけ触れたのに active/RSS が層全体(数百MB)〜全store(9.4GB)に膨れたか
    print(f"\n[mmap] 判定:")
    print(f"  もし active が ~数MB増 → 遅延 gather 成功（触れた expert だけ）")
    print(f"  もし active が ~{w.nbytes/1e6:.0f}MB増(層全体) → 層単位 materialize（forward毎に全256 page-in）")
    print(f"  もし RSS が {sz:.0f}GB へ → 全 store materialize（8GB 不成立）")

    # 追試: 全 40 層 gather を回し、RSS が頭打ち(OS evict)か単調増(全保持)か
    print(f"\n[mmap] 全40層×{{gate,up}} gather を1巡 → RSS 推移（頭打ち=OS evict成功 / 単調増=全保持）:")
    for L in range(40):
        p = f"language_model.model.layers.{L}.mlp.switch_mlp"
        for proj in ("gate_proj", "up_proj"):            # 共に in=2048（x=xe そのまま）
            ww = W[f"{p}.{proj}.weight"]; ss = W[f"{p}.{proj}.scales"]; bb = W[f"{p}.{proj}.biases"]
            mx.eval(mx.gather_qmm(xe, ww, ss, bb, rhs_indices=inds, transpose=True,
                                  group_size=64, bits=2, mode="affine", sorted_indices=False))
        if L % 10 == 9 or L == 0:
            print(f"  after layer {L:2}: RSS={rss_gb():.2f}GB active={active_gb():.2f}GB")


if __name__ == "__main__":
    main()
