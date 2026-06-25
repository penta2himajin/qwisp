"""A-2a — prev-token prefetch ＋ GPU slot_of remap（per-layer 同期を消す近似版）.

設計（docs/07 A-2）:
- 各層 GPU `slot_of`[256] int32（expert→slot, miss=-1）。forward は remap=slot_of[inds]
  （GPU、.tolist() 不要）。miss は slot0 にクランプ（出力近似）。
- スタックは永続 [B,...]。トークン間の after_token() で「直前トークンの experts」を
  prefetch（load+slot 更新）→ 次トークンが常駐ヒット。slot 更新は per-layer critical
  path の外（トークン間）に出すので forward は同期なし。
- 初回（cold）層だけ exact（sync）で populate。

これで「速度（同期除去の現実効果）」と「prev-token prefetch だけで verify 一致率
（=投機の成立度＝局所性）」を同時に測る。一致率が高ければ A-2b（miss 検出+再計算）で
厳密化する価値あり。

実行:
  PY="$HOME/Library/Application Support/MTPLX/runtime-venv/bin/python"
  "$PY" -m qwisp.prefetch "$MODEL" --budget 128 --gen 64
"""

from __future__ import annotations

import argparse
import sys
import time

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.generate import stream_generate
from mlx_lm.sample_utils import make_sampler

from .cache import ExpertCache
from .expert_source import PARTS, PROJS
from .loader import load_streaming
from .streaming_moe import StreamingSwitchGLU

NEXP = 256


class PrefetchCache:
    def __init__(self, source, budget_per_layer: int):
        self.src = source
        self.B = budget_per_layer
        self._stacks: dict[int, dict[str, mx.array]] = {}
        self._slot_gpu: dict[int, mx.array] = {}      # [256] int32, expert→slot or -1
        self._slot_cpu: dict[int, dict[int, int]] = {}
        self._lru: dict[int, "object"] = {}
        self._next_free: dict[int, int] = {}
        self._warm: set[int] = set()
        self._captured: list = []
        self.misses = 0
        self.hits = 0

    def _load(self, layer, e):
        return {f"{p}.{q}": self.src.slice(layer, p, q, e) for p in PROJS for q in PARTS}

    def _init_layer(self, layer, e0):
        from collections import OrderedDict
        row = self._load(layer, e0)
        self._stacks[layer] = {
            pp: mx.zeros((self.B,) + tuple(a.shape[1:]), dtype=a.dtype) for pp, a in row.items()}
        self._slot_gpu[layer] = mx.zeros((NEXP,), dtype=mx.int32) - 1
        self._slot_cpu[layer] = {}
        self._lru[layer] = OrderedDict()
        self._next_free[layer] = 0

    def _ensure(self, layer, experts):
        if layer not in self._stacks:
            self._init_layer(layer, experts[0])
        st = self._stacks[layer]
        sof = self._slot_cpu[layer]
        lru = self._lru[layer]
        sg = self._slot_gpu[layer]
        for e in experts:
            if e in sof:
                self.hits += 1
                lru.move_to_end(e)
                continue
            self.misses += 1
            if self._next_free[layer] < self.B:
                s = self._next_free[layer]
                self._next_free[layer] += 1
            else:
                ev_e, _ = lru.popitem(last=False)  # LRU expert を evict
                s = sof.pop(ev_e)
                sg[ev_e] = -1
            row = self._load(layer, e)
            for pp, a in row.items():
                st[pp][s] = a[0]   # slot 行をインプレース差し替え（トークン間＝critical path 外）
            sof[e] = s
            lru[e] = None
            sg[e] = s

    def gather(self, layer, inds):
        """forward 中（同期なし）: remap=slot_of[inds]、永続スタックを返す。inds を記録。"""
        if layer not in self._warm:
            # 初回のみ exact（sync）で populate
            u = np.unique(np.asarray(inds.tolist())).tolist()
            self._ensure(layer, u)
            self._warm.add(layer)
        sg = self._slot_gpu[layer]
        remap = mx.maximum(sg[inds], 0)   # GPU gather、miss は slot0（近似）
        self._captured.append((layer, inds))
        return self._stacks[layer], remap

    def after_token(self):
        """トークン間: 直前トークンの experts を prefetch（次トークンの常駐に）。"""
        for layer, inds in self._captured:
            u = np.unique(np.asarray(inds.tolist())).tolist()
            self._ensure(layer, u)
        self._captured.clear()


def greedy(model, tok, prompt, gen, after=None):
    sampler = make_sampler(temp=0.0)
    toks = []
    last = None
    t0 = time.perf_counter()
    for r in stream_generate(model, tok, prompt=prompt, max_tokens=gen, sampler=sampler):
        if after is not None:
            after()
        toks.append(r.token)
        last = r
    return toks, getattr(last, "generation_tps", float("nan")), time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser(description="Qwisp A-2a prefetch probe")
    ap.add_argument("model")
    ap.add_argument("--budget", type=int, default=128)
    ap.add_argument("--gen", type=int, default=64)
    args = ap.parse_args()

    # 参照（exact, LRU）
    print("[a2a] exact reference ...", file=sys.stderr)
    model, tok, src = load_streaming(args.model)
    cache = ExpertCache(src, budget_per_layer=args.budget)
    blocks = [m for _, m in model.named_modules() if isinstance(m, StreamingSwitchGLU)]
    for m in blocks:
        m._cache = cache
    prompt = tok.encode("def process(items):\n    out = []\n    for it in items:\n        ")[:32]
    ref_toks, ref_tps, _ = greedy(model, tok, prompt, args.gen)
    del model, cache
    if hasattr(mx, "clear_cache"):
        mx.clear_cache()

    # prefetch（A-2a, 近似）
    print("[a2a] prefetch (approx) ...", file=sys.stderr)
    model, tok, src = load_streaming(args.model)
    ctx = PrefetchCache(src, budget_per_layer=args.budget)
    blocks = [m for _, m in model.named_modules() if isinstance(m, StreamingSwitchGLU)]
    for m in blocks:
        m._prefetch = ctx
    pf_toks, pf_tps, _ = greedy(model, tok, prompt, args.gen, after=ctx.after_token)

    match = sum(1 for a, b in zip(ref_toks, pf_toks) if a == b)
    # 先頭から何トークン一致が続くか（発散点）
    run = 0
    for a, b in zip(ref_toks, pf_toks):
        if a == b:
            run += 1
        else:
            break
    print(f"\n[a2a] exact decode_tps  = {ref_tps:.1f}")
    print(f"[a2a] prefetch decode_tps = {pf_tps:.1f}  ({pf_tps/ref_tps:.2f}x)")
    print(f"[a2a] token match = {match}/{len(ref_toks)}  (先頭連続一致 {run})")
    print(f"[a2a] prefetch miss率 = {ctx.misses/(ctx.hits+ctx.misses+1e-9):.3f}", file=sys.stderr)


if __name__ == "__main__":
    main()
