"""GPU-routed mixed switch_mlp — per-layer tolist 同期を完全排除（feasibility 実験, docs/09）.

full_profile の発見: IO ゼロの B(全常駐)でも tolist 同期で A の 6× 遅い＝同期がパイプライン破壊。
→ hot/cold 全 expert を**持続 GPU バッファ**に置き、**GPU の inds で直接 gather_qmm**（CPU 往復ゼロ、
tolist 無し）。これで mixed 全常駐(12GB)が A(94tok/s) にどこまで近づくかを測る。

制約 RAM streaming には使えない（全 expert 常駐前提）。狙いは **16GB で mixed-resident を高速化**:
4bit 全 18GB は載らないが mixed 12GB は載る → GPU-route で同期税を消せれば 13→40+tok/s。

fast_hot が回帰したのは cold 側が cache+tolist のままだったため。ここは両側 GPU-route・mx.contiguous。
"""
from __future__ import annotations
import re

import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.activations import swiglu
from mlx_lm.models.qwen3_next import Qwen3NextSparseMoeBlock

from .expert_source import ExpertSource, PROJS, PARTS

_LAYER_RE = re.compile(r"\.layers\.(\d+)\.")


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
        # prefill partition 用の numpy 版（CPU で position を分割する）
        self._lut_hot_np = lut_hot
        self._lut_cold_np = lut_cold
        self._is_hot_np = ishot

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

    _probe_sync = False    # 診断: True で per-layer tolist 同期だけ足す（drain コスト分離）
    # この token 数を超えたら prefill partition（redundant 2x matmul を消す, docs/09）。
    # 以下は decode/verify（grid 小, sync ゼロ優先で full two-gather）。
    PREFILL_T = 8

    def __call__(self, x, inds):
        if self._probe_sync:
            _ = inds.tolist()         # GPU drain を強制（streaming の tolist を模擬）
        tok = 1
        for d in inds.shape[:-1]:
            tok *= d
        if tok > self.PREFILL_T:
            return self._prefill_partition(x, inds)
        # decode/verify（grid 小）: tolist 無し full two-gather。remap も is_hot も GPU LUT 引き。
        remap_hot = self._lut_hot[inds]
        remap_cold = self._lut_cold[inds]
        y_hot = self._qmm(x, self._hotbuf, remap_hot, 4)
        y_cold = self._qmm(x, self._coldbuf, remap_cold, 2)
        return mx.where(self._is_hot[inds][..., None], y_hot, y_cold)

    def _prefill_partition(self, x, inds):
        """prefill: 全 (token,expert) position を hot/cold に分割し、各側を subset の matmul
        だけ実行→scatter で戻す。full two-gather の 2N matmul を N に半減。
        large-T なので tolist 同期1回は完全に償却される。出力は decode と同形 [*lead,K,H]。"""
        inds_np = np.asarray(inds.tolist())                 # [*lead, K]
        lead = inds_np.shape[:-1]
        K = inds_np.shape[-1]
        Ttok = int(np.prod(lead)) if lead else 1
        H = x.shape[-1]
        xf = x.reshape(Ttok, H)                              # [Ttok, H]
        flat_e = inds_np.reshape(-1)                         # [N]  expert id / position
        flat_hot = self._is_hot_np[flat_e]                  # [N]  bool
        tok_of_pos = np.repeat(np.arange(Ttok, dtype=np.int32), K)  # [N]
        N = flat_e.shape[0]
        out = mx.zeros((N, H), dtype=x.dtype)
        for sel, buf, lut, bits in ((flat_hot, self._hotbuf, self._lut_hot_np, 4),
                                    (~flat_hot, self._coldbuf, self._lut_cold_np, 2)):
            pos = np.nonzero(sel)[0]
            if pos.size == 0:
                continue
            x_sub = xf[mx.array(tok_of_pos[pos])]           # [n, H]  token 行を gather（重複可）
            remap = mx.array(lut[flat_e[pos]].reshape(-1, 1))   # [n,1]  expert→slot
            y = self._qmm(x_sub, buf, remap, bits)          # [n,1,H]
            out[mx.array(pos)] = y.reshape(pos.size, H)
        return out.reshape(*lead, K, H)


def load_gpu_routed(model_dir, dir2, hot_b=64):
    """lean ローダ: full 18GB を経由せず lazy→streaming で calibrate→GPU バッファ構築.

    peak RSS ≈ 非expert 1.8GB + mixed 全常駐 12GB（4bit 全常駐 18GB を materialize しない）。
    16GB Mac に収めるための経路。返り値 (model, tok)。
    """
    from .loader import load_streaming
    from .mixed_engine import _calibrate
    model, tok, src4 = load_streaming(model_dir)          # lazy, streaming, 低 RSS
    src2 = ExpertSource(dir2)
    counts = _calibrate(model, tok)                       # streaming 上で hot 集計（低 RSS）
    for name, blk in model.language_model.named_modules() if hasattr(model, "language_model") \
            else model.named_modules():
        if isinstance(blk, Qwen3NextSparseMoeBlock):
            layer = int(_LAYER_RE.search(name).group(1))
            c = counts.get(id(blk), np.zeros(256, np.int64))
            hot = set(np.argsort(c)[-hot_b:].tolist())
            blk.switch_mlp = GPURoutedMixedSwitchGLU(layer, hot, src4, src2)
    return model, tok
