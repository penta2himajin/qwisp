"""GPU-routed mixed switch_mlp — per-layer tolist 同期を完全排除（feasibility 実験, docs/09）.

full_profile の発見: IO ゼロの B(全常駐)でも tolist 同期で A の 6× 遅い＝同期がパイプライン破壊。
→ hot/cold 全 expert を**持続 GPU バッファ**に置き、**GPU の inds で直接 gather_qmm**（CPU 往復ゼロ、
tolist 無し）。これで mixed 全常駐(12GB)が A(94tok/s) にどこまで近づくかを測る。

制約 RAM streaming には使えない（全 expert 常駐前提）。狙いは **16GB で mixed-resident を高速化**:
4bit 全 18GB は載らないが mixed 12GB は載る → GPU-route で同期税を消せれば 13→40+tok/s。

fast_hot が回帰したのは cold 側が cache+tolist のままだったため。ここは両側 GPU-route・mx.contiguous。
"""
from __future__ import annotations
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.activations import swiglu

from .expert_source import PROJS, PARTS


class GPURoutedMixedSwitchGLU(nn.Module):
    """hot(4bit)/cold(2bit) を持続 GPU バッファに常駐させ GPU inds で gather（tolist 無し）。"""

    def __init__(self, layer, hot_set, src4, src2):
        super().__init__()
        self._layer = layer
        hot_sorted = sorted(hot_set)
        cold_sorted = [e for e in range(256) if e not in hot_set]
        # 全 expert を 2 バッファに pre-stack（contiguous）
        self._hotbuf = self._stack(src4, layer, hot_sorted)
        self._coldbuf = self._stack(src2, layer, cold_sorted)
        lut_hot = np.zeros(256, np.int32)
        lut_cold = np.zeros(256, np.int32)
        ishot = np.zeros(256, bool)
        for s, e in enumerate(hot_sorted):
            lut_hot[e] = s; ishot[e] = True
        for s, e in enumerate(cold_sorted):
            lut_cold[e] = s
        self._lut_hot = mx.array(lut_hot)
        self._lut_cold = mx.array(lut_cold)
        self._is_hot = mx.array(ishot)

    @staticmethod
    def _stack(src, layer, experts):
        buf = {}
        for p in PROJS:
            for q in PARTS:
                arr = mx.concatenate([src.slice(layer, p, q, e) for e in experts], axis=0)
                buf[f"{p}.{q}"] = mx.contiguous(arr)
        mx.eval(list(buf.values()))
        return buf

    def _qmm(self, x, buf, remap, bits):
        xe = mx.expand_dims(x, (-2, -3))

        def q(xx, proj):
            return mx.gather_qmm(xx, buf[f"{proj}.weight"], buf[f"{proj}.scales"],
                                 buf[f"{proj}.biases"], rhs_indices=remap, transpose=True,
                                 group_size=64, bits=bits, mode="affine", sorted_indices=False)
        h = swiglu(q(xe, "gate_proj"), q(xe, "up_proj"))
        return q(h, "down_proj").squeeze(-2)

    def __call__(self, x, inds):
        # tolist 無し: 全部 GPU 演算。remap も is_hot も GPU LUT 引き。
        remap_hot = self._lut_hot[inds]
        remap_cold = self._lut_cold[inds]
        y_hot = self._qmm(x, self._hotbuf, remap_hot, 4)
        y_cold = self._qmm(x, self._coldbuf, remap_cold, 2)
        return mx.where(self._is_hot[inds][..., None], y_hot, y_cold)
