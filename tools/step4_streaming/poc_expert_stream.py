#!/usr/bin/env python3
"""Qwisp Step 4 PoC — on-demand expert load from mmap'd safetensors（E-2 リスク検証）.

最重要・最安の未検証点（docs/01 E-2: MLX に独立 VRAM 無し、offload 先が NAND）を潰す:
  「expert e をディスクから on-demand で読み、常駐モデルの該当 expert と bit 一致し、
   読むバイトは ~1.73MB/expert に限定される」

格納形式（実モデル確認済）:
  switch_mlp.{gate,up,down}_proj.{weight(U32 4bit packed), scales(F16), biases(F16)}
  各テンソルは先頭次元に 256 experts スタック → expert e は連続バイトスライス。
  per-expert = 3 proj × (weight 512KB + scales 32KB + biases 32KB) ≈ 1.73MB。

これが OK なら IO/メモリ機構は成立（compute は MLX 標準 quantized_matmul なので risk 外）。

実行（mlx を持つ runtime-venv python）:
  PY="$HOME/Library/Application Support/MTPLX/runtime-venv/bin/python"
  "$PY" poc_expert_stream.py --model "$HOME/.mtplx/models/Youssofal--...-FP16"
"""

import argparse
import fcntl
import json
import os
import struct
import sys
import time

import mlx.core as mx
import numpy as np

F_NOCACHE = getattr(fcntl, "F_NOCACHE", 48)
DT = {"U32": (np.uint32, 4), "F16": (np.float16, 2), "I32": (np.int32, 4),
      "U16": (np.uint16, 2), "F32": (np.float32, 4)}

PROJS = ["gate_proj", "up_proj", "down_proj"]
PARTS = ["weight", "scales", "biases"]


def read_header(shard):
    with open(shard, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    return hdr, 8 + n  # header dict, data 開始オフセット


class ExpertLoader:
    """1 expert 分のバイトだけ pread して mx.array を復元する（常駐は持たない）。"""

    def __init__(self, shard, hdr, data_start, nocache=True):
        self.fd = os.open(shard, os.O_RDONLY)
        if nocache:
            fcntl.fcntl(self.fd, F_NOCACHE, 1)
        self.hdr = hdr
        self.data_start = data_start

    def close(self):
        os.close(self.fd)

    def load_tensor_expert(self, key, e):
        t = self.hdr[key]
        np_dt, _ = DT[t["dtype"]]
        b, end = t["data_offsets"]
        n_exp = t["shape"][0]
        stride = (end - b) // n_exp
        off = self.data_start + b + e * stride
        buf = os.pread(self.fd, stride, off)
        arr = np.frombuffer(buf, dtype=np_dt).reshape([1] + t["shape"][1:])
        return mx.array(arr), stride

    def load_expert(self, layer, e):
        """1 expert の 9 テンソルをまとめて読む。返り値 dict と総バイト。"""
        out, total = {}, 0
        base = f"language_model.model.layers.{layer}.mlp.switch_mlp"
        for proj in PROJS:
            for part in PARTS:
                a, nbytes = self.load_tensor_expert(f"{base}.{proj}.{part}", e)
                out[f"{proj}.{part}"] = a
                total += nbytes
        return out, total


def main():
    ap = argparse.ArgumentParser(description="Qwisp Step4 expert-stream PoC")
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--experts", default="0,5,100,255")
    ap.add_argument("--shard", default="model-00001-of-00004.safetensors")
    args = ap.parse_args()

    shard = os.path.join(args.model, args.shard)
    hdr, data_start = read_header(shard)
    experts = [int(x) for x in args.experts.split(",")]
    base = f"language_model.model.layers.{args.layer}.mlp.switch_mlp"

    # --- ground truth: MLX 正規ローダで shard をロード（検証用・一度きり）---
    print(f"[poc] loading ground truth via mx.load({args.shard}) ...", file=sys.stderr)
    t0 = time.perf_counter()
    gt = mx.load(shard)
    mx.eval(gt[f"{base}.gate_proj.weight"])
    print(f"[poc] mx.load done in {time.perf_counter()-t0:.1f}s", file=sys.stderr)

    loader = ExpertLoader(shard, hdr, data_start, nocache=True)

    print(f"\n[poc] === bit-exact 検証（layer {args.layer}）===")
    all_ok = True
    per_expert_bytes = None
    for e in experts:
        streamed, total = loader.load_expert(args.layer, e)
        per_expert_bytes = total
        ok_e = True
        for proj in PROJS:
            for part in PARTS:
                key = f"{base}.{proj}.{part}"
                gt_slice = gt[key][e:e + 1]
                mine = streamed[f"{proj}.{part}"]
                same = bool(mx.array_equal(gt_slice, mine).item())
                ok_e = ok_e and same
        print(f"  expert {e:>3}: {'OK (全9テンソル bit一致)' if ok_e else 'MISMATCH'}")
        all_ok = all_ok and ok_e

    print(f"\n[poc] per-expert bytes = {per_expert_bytes/1024:.0f} KB "
          f"= {per_expert_bytes/1e6:.3f} MB", file=sys.stderr)

    # --- cold ロード時間（純 IO、ground truth 解放後）---
    del gt
    mx.clear_cache() if hasattr(mx, "clear_cache") else None
    print("[poc] === cold ロード時間（F_NOCACHE）===", file=sys.stderr)
    import random
    rng = random.Random(0)
    samples = [rng.randrange(0, 256) for _ in range(64)]
    t0 = time.perf_counter()
    tot = 0
    for e in samples:
        _, b = loader.load_expert(args.layer, e)
        tot += b
    dt = time.perf_counter() - t0
    per = dt / len(samples)
    print(f"[poc] {len(samples)} experts: {dt*1000:.0f}ms total, "
          f"{per*1000:.3f}ms/expert, {tot/dt/1e9:.2f} GB/s", file=sys.stderr)
    print(f"[poc] → 320 expert-miss/token なら最悪 {per*320*1000:.0f}ms/token "
          f"（キャッシュで miss 率↓ぶん短縮）", file=sys.stderr)

    loader.close()
    print(f"\n[poc] VERDICT: {'PASS — on-demand expert load 成立（E-2 機構 OK）' if all_ok else 'FAIL'}",
          file=sys.stderr)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
