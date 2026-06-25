"""StreamingSwitchGLU — switch_mlp の差し替え（常駐 expert を持たず、cache/source から取得）.

mlx_lm の QuantizedSwitchLinear/SwitchGLU を忠実再現（PoC 4.1 で bit 一致確認済）:
  gather_qmm(x, w, scales, biases, rhs_indices, transpose=True, group_size, bits, mode="affine")
  up,gate → swiglu(gate, up) → down → squeeze(-2)。sorted_indices=False で任意 indices に正しい。

v0.1: キャッシュ無しの on-demand（毎回 source から必要 experts を読む。正しいが遅い）。
v0.2 でこの取得を ExpertCache（LRU→MTP 予測器 prefetch）に差し替える。
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models.activations import swiglu


class StreamingSwitchGLU(nn.Module):
    def __init__(self, source, layer_idx: int, group_size: int = 64, bits: int = 4,
                 cache=None):
        super().__init__()
        # mlx Module は array/Module 属性だけを param/child 扱い。下記は通常属性。
        self._src = source
        self._layer = layer_idx
        self._gs = group_size
        self._bits = bits
        self._cache = cache  # v0.2: ExpertCache。None なら on-demand。
        # A-1 probe: 前トークンの (sub, remap) を再利用し per-layer 同期/CPU 作業を消す。
        # 出力は不正（前トークンの experts を使う）。同期除去の速度天井を測る用。
        self.probe_no_sync = False
        self._prev = None

    def _experts(self, U):
        if self._cache is not None:
            return self._cache.gather(self._layer, U)
        return self._src.load_experts(self._layer, U)

    def _qmm(self, x, sub, proj, remap):
        return mx.gather_qmm(
            x, sub[f"{proj}.weight"], sub[f"{proj}.scales"], sub[f"{proj}.biases"],
            rhs_indices=remap, transpose=True, group_size=self._gs, bits=self._bits,
            mode="affine", sorted_indices=False)

    def __call__(self, x, inds):
        if self.probe_no_sync and self._prev is not None and self._prev[1].shape == inds.shape:
            # 同期除去の天井測定: 前トークンの (sub, remap) を再利用（.tolist/cache をスキップ）
            sub, remap = self._prev
        else:
            # unique(sorted)＋remap を np.unique 一発で（旧 set/sorted/dict/np.vectorize を置換）。
            inds_np = np.asarray(inds.tolist())
            U_arr, inv = np.unique(inds_np, return_inverse=True)
            U = U_arr.tolist()
            remap = mx.array(inv.reshape(inds_np.shape).astype(np.int32))
            sub = self._experts(U)
            if self.probe_no_sync:
                self._prev = (sub, remap)

        xe = mx.expand_dims(x, (-2, -3))
        x_up = self._qmm(xe, sub, "up_proj", remap)
        x_gate = self._qmm(xe, sub, "gate_proj", remap)
        h = swiglu(x_gate, x_up)
        x_down = self._qmm(h, sub, "down_proj", remap)
        return x_down.squeeze(-2)
