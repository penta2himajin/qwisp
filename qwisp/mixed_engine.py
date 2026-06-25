"""Design B — mixed-precision two-gather streaming（hot 4bit / cold 2bit）.

decode（tokens=1）で、選択 expert を hot（静的 top-B、4bit cache）と cold（2bit source）に
分割し、各精度で gather_qmm → per-slot 出力を組み立てる。prefill は 4bit fallback。

両精度は disk 保管済（build_2bit_experts.py）＝ロード時 requant 不要（DynaExq/HOBBIT 流）。

実行: PY -m qwisp.mixed_engine <4bit_model> <2bit_dir> [--hot-b 128]
"""
from __future__ import annotations
import argparse
import sys
import time

import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from mlx_lm.generate import stream_generate
from mlx_lm.sample_utils import make_sampler
from mlx_lm.models.activations import swiglu
from mlx_lm.models.qwen3_next import Qwen3NextSparseMoeBlock

from .expert_source import ExpertSource
from .cache import ExpertCache
from .loader import load_streaming
from .streaming_moe import StreamingSwitchGLU


class MixedSwitchGLU(nn.Module):
    """hot=4bit / cold=2bit の two-gather。hot_set は静的 top-B。"""

    def __init__(self, layer, hot_set, c4, c2):
        super().__init__()
        self._layer = layer
        self._hot = hot_set        # set[int]（4bit 維持）
        self._c4 = c4              # 4bit ExpertCache
        self._c2 = c2              # 2bit ExpertCache

    def _grp(self, x, U, remap, cache, bits):
        sub = cache.gather(self._layer, U)
        xe = mx.expand_dims(x, (-2, -3))

        def qmm(xx, proj):
            return mx.gather_qmm(xx, sub[f"{proj}.weight"], sub[f"{proj}.scales"],
                                 sub[f"{proj}.biases"], rhs_indices=remap, transpose=True,
                                 group_size=64, bits=bits, mode="affine", sorted_indices=False)
        h = swiglu(qmm(xe, "gate_proj"), qmm(xe, "up_proj"))
        return qmm(h, "down_proj").squeeze(-2)  # [...,ngrp,dim]

    def __call__(self, x, inds):
        # prefill or many-token: 4bit 単一 gather（簡易）
        if x.shape[1] != 1:
            inds_np = np.asarray(inds.tolist())
            U, inv = np.unique(inds_np, return_inverse=True)
            remap = mx.array(inv.reshape(inds_np.shape).astype(np.int32))
            return self._grp(x, U.tolist(), remap, self._c4, 4)

        # decode: hot/cold split → two-gather
        ids = np.asarray(inds.tolist()).reshape(-1)        # [8]
        hot_pos = [i for i, e in enumerate(ids) if int(e) in self._hot]
        cold_pos = [i for i in range(len(ids)) if i not in hot_pos]
        dim = x.shape[-1]
        out = mx.zeros((1, 1, len(ids), dim), dtype=x.dtype)

        def fill(pos, cache, bits):
            if not pos:
                return
            sub_ids = [int(ids[i]) for i in pos]
            U, inv = np.unique(np.asarray(sub_ids), return_inverse=True)
            remap = mx.array(inv.reshape(1, 1, len(pos)).astype(np.int32))
            y = self._grp(x, U.tolist(), remap, cache, bits)   # [1,1,len(pos),dim]
            for j, i in enumerate(pos):
                out[0, 0, i] = y[0, 0, j]
        fill(hot_pos, self._c4, 4)
        fill(cold_pos, self._c2, 2)
        return out


def build_mixed(model_dir, dir2, hot_b):
    model, tok, src4 = load_streaming(model_dir)
    src2 = ExpertSource(dir2)
    c4 = ExpertCache(src4, budget_per_layer=256)
    c2 = ExpertCache(src2, budget_per_layer=256)
    # 静的 hot set: calibration で頻度上位（簡易に layer 共通の便宜セット=全部 hot 上位）
    counts = _calibrate(model, tok)
    n = 0
    for _, blk in model.named_modules():
        if isinstance(blk, Qwen3NextSparseMoeBlock):
            c = counts.get(id(blk), np.zeros(256, np.int64))
            hot = set(np.argsort(c)[-hot_b:].tolist()) if hot_b < 256 else set(range(256))
            blk.switch_mlp = MixedSwitchGLU(n, hot, c4, c2)
            n += 1
    return model, tok, c4, c2


def _calibrate(model, tok):
    counts = {}
    orig = Qwen3NextSparseMoeBlock.__call__

    def patched(self, x):
        g = mx.softmax(self.gate(x), axis=-1, precise=True)
        inds = mx.argpartition(g, kth=-8, axis=-1)[..., -8:]
        mx.eval(inds)
        c = counts.setdefault(id(self), np.zeros(256, np.int64))
        np.add.at(c, np.array(inds).reshape(-1), 1)
        return orig(self, x)

    Qwen3NextSparseMoeBlock.__call__ = patched
    s = make_sampler(temp=0.0)
    for p in ["def f(x):\n    return x*2\n", "import os\nclass A:\n    def run(self):\n"]:
        for _ in stream_generate(model, tok, prompt=tok.encode(p * 8)[:256], max_tokens=1, sampler=s):
            break
    Qwen3NextSparseMoeBlock.__call__ = orig
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("dir2")
    ap.add_argument("--hot-b", type=int, default=128)
    ap.add_argument("--gen", type=int, default=48)
    args = ap.parse_args()

    # 参照（v0.2 全4bit）
    print("[mix] reference (all-4bit) ...", file=sys.stderr)
    model, tok, src = load_streaming(args.model)
    cache = ExpertCache(src, 256)
    for _, m in model.named_modules():
        if isinstance(m, StreamingSwitchGLU):
            m._cache = cache
    prompt = tok.encode("def process(items):\n    out=[]\n    for it in items:\n        ")[:32]
    s = make_sampler(temp=0.0)
    t0 = time.perf_counter(); ref = []; last = None
    for r in stream_generate(model, tok, prompt=prompt, max_tokens=args.gen, sampler=s):
        ref.append(r.token); last = r
    ref_tps = getattr(last, "generation_tps", float("nan"))
    del model, cache
    if hasattr(mx, "clear_cache"):
        mx.clear_cache()

    # mixed（hot4/cold2 two-gather）
    print("[mix] mixed (hot4/cold2 two-gather) ...", file=sys.stderr)
    model, tok, c4, c2 = build_mixed(args.model, args.dir2, args.hot_b)
    last = None; deg = []
    for r in stream_generate(model, tok, prompt=prompt, max_tokens=args.gen, sampler=s):
        deg.append(r.token); last = r
    mix_tps = getattr(last, "generation_tps", float("nan"))

    match = sum(1 for a, b in zip(ref, deg) if a == b)
    print(f"\n[mix] hot={args.hot_b}@4 / cold={256-args.hot_b}@2bit")
    print(f"  ref(all-4bit) decode_tps = {ref_tps:.1f}")
    print(f"  mixed         decode_tps = {mix_tps:.1f}  ({mix_tps/ref_tps:.2f}x)")
    print(f"  quality token match = {match}/{len(ref)}")


if __name__ == "__main__":
    main()
