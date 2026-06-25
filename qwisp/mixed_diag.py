"""切り分け診断 — MixedSwitchGLU(hot-b=256) の prefill と decode を ref と突き合わせる.

ref = StreamingSwitchGLU(全4bit). mixed = MixedSwitchGLU(全 hot=4bit).
同一の (x, inds) に対し、switch_mlp 単体出力の最大絶対差を layer ごとに測る。
prefill(T>1) と decode(T=1) を別々に検査して、どちらの経路が壊れているか特定する。
"""
from __future__ import annotations
import sys
import numpy as np
import mlx.core as mx

from .loader import load_streaming
from .expert_source import ExpertSource
from .cache import ExpertCache
from .mixed_engine import MixedSwitchGLU


def main():
    model_dir, dir2 = sys.argv[1], sys.argv[2]
    src4 = ExpertSource(model_dir)
    c4 = ExpertCache(src4, budget_per_layer=256)

    ref = __import__("qwisp.streaming_moe", fromlist=["StreamingSwitchGLU"]).StreamingSwitchGLU
    refmod = ref(src4, 0, group_size=64, bits=4, cache=c4)
    mix = MixedSwitchGLU(0, set(range(256)), c4, c4)  # 全 hot, cold=未使用

    dim = 2048
    rng = np.random.default_rng(0)

    for T in (5, 1):
        x = mx.array(rng.standard_normal((1, T, dim)).astype(np.float32) * 0.1)
        inds = mx.array(rng.integers(0, 256, size=(1, T, 8)).astype(np.int32))
        yr = refmod(x, inds)
        ym = mix(x, inds)
        d = float(mx.max(mx.abs(yr - ym)).item())
        print(f"T={T}: ref{tuple(yr.shape)} mix{tuple(ym.shape)} maxabs_diff={d:.3e}")


if __name__ == "__main__":
    main()
