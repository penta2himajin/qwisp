"""ExpertSource — safetensors から per-expert スライスを on-demand 取得（多シャード対応）.

格納形式（実モデル確認済）: layers.{L}.mlp.switch_mlp.{gate,up,down}_proj.{weight(U32 4bit),
scales(F16), biases(F16)}、先頭次元に 256 experts スタック → expert e は連続バイトスライス。
オフセットを safetensors ヘッダから計算し、その expert のバイトだけ pread する。
"""

from __future__ import annotations

import json
import os
import struct

import mlx.core as mx
import numpy as np

_DT = {"U32": np.uint32, "F16": np.float16, "BF16": np.uint16, "F32": np.float32}
PROJS = ("gate_proj", "up_proj", "down_proj")
PARTS = ("weight", "scales", "biases")
_PREFIX = "language_model.model.layers"


class ExpertSource:
    def __init__(self, model_dir: str):
        self.dir = model_dir
        with open(os.path.join(model_dir, "model.safetensors.index.json")) as f:
            self.wm = json.load(f)["weight_map"]
        self._hdr: dict[str, tuple[dict, int]] = {}
        self._fd: dict[str, int] = {}

    def _key(self, layer: int, proj: str, part: str) -> str:
        return f"{_PREFIX}.{layer}.mlp.switch_mlp.{proj}.{part}"

    def _header(self, shard: str):
        if shard not in self._hdr:
            with open(os.path.join(self.dir, shard), "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                hdr = json.loads(f.read(n))
            self._hdr[shard] = (hdr, 8 + n)
        return self._hdr[shard]

    def _fdesc(self, shard: str) -> int:
        if shard not in self._fd:
            # v0.1: F_NOCACHE は付けない（OS ページキャッシュに hot pages を持たせる）。
            self._fd[shard] = os.open(os.path.join(self.dir, shard), os.O_RDONLY)
        return self._fd[shard]

    def slice(self, layer: int, proj: str, part: str, e: int) -> mx.array:
        key = self._key(layer, proj, part)
        shard = self.wm[key]
        hdr, data_start = self._header(shard)
        t = hdr[key]
        b, end = t["data_offsets"]
        n_exp = t["shape"][0]
        stride = (end - b) // n_exp
        buf = os.pread(self._fdesc(shard), stride, data_start + b + e * stride)
        arr = np.frombuffer(buf, _DT[t["dtype"]]).reshape([1] + t["shape"][1:])
        return mx.array(arr)

    def load_experts(self, layer: int, experts: list[int]) -> dict[str, mx.array]:
        """experts を stack（[len, ...]）で返す。proj.part キー。"""
        sub = {}
        for proj in PROJS:
            for part in PARTS:
                sub[f"{proj}.{part}"] = mx.concatenate(
                    [self.slice(layer, proj, part, e) for e in experts], axis=0)
        return sub

    def per_expert_bytes(self, layer: int = 0) -> int:
        total = 0
        for proj in PROJS:
            for part in PARTS:
                key = self._key(layer, proj, part)
                hdr, _ = self._header(self.wm[key])
                t = hdr[key]
                b, end = t["data_offsets"]
                total += (end - b) // t["shape"][0]
        return total

    def close(self):
        for fd in self._fd.values():
            os.close(fd)
        self._fd.clear()
