"""ExpertCache — per-layer LRU の常駐 expert キャッシュ（Step2 sim の予算 B に対応）.

各層が独立に B 個の expert を常駐（hit なら再 pread しない）。gather(layer, U) は U を
stack で返し、使った expert を LRU に入れて B を超えたら oldest を evict。

v0.1 は LRU。v0.2 で MTP 予測器 prefetch を足す（gather 前に next-token 予測の experts を
先読み）。SwiftLM の「app-LRU < OS-cache」は zero-copy mmap 文脈の話で、copy する我々の
経路では明示キャッシュが必要（docs/07）。
"""

from __future__ import annotations

from collections import OrderedDict

import mlx.core as mx

from .expert_source import PARTS, PROJS


class ExpertCache:
    def __init__(self, source, budget_per_layer: int):
        self.src = source
        self.B = budget_per_layer
        self._store: dict[tuple[int, int], dict[str, mx.array]] = {}
        self._lru: dict[int, "OrderedDict[int, None]"] = {}
        self.hits = 0
        self.misses = 0

    def _ensure(self, layer: int, e: int) -> dict[str, mx.array]:
        key = (layer, e)
        lru = self._lru.setdefault(layer, OrderedDict())
        cached = self._store.get(key)
        if cached is not None:
            self.hits += 1
            lru.move_to_end(e)
            return cached
        self.misses += 1
        slices = {f"{p}.{q}": self.src.slice(layer, p, q, e) for p in PROJS for q in PARTS}
        self._store[key] = slices
        lru[e] = None
        return slices

    def gather(self, layer: int, U: list[int]) -> dict[str, mx.array]:
        per = [self._ensure(layer, e) for e in U]  # 現 U は先に全 load（evict 前）
        sub = {}
        for p in PROJS:
            for q in PARTS:
                pp = f"{p}.{q}"
                sub[pp] = mx.concatenate([s[pp] for s in per], axis=0)
        # B を超えた分を evict（most-recent＝現 U は残る）
        lru = self._lru[layer]
        while len(lru) > self.B:
            old, _ = lru.popitem(last=False)
            self._store.pop((layer, old), None)
        return sub

    def reset_stats(self):
        self.hits = self.misses = 0

    def clear(self):
        """常駐 expert を全破棄（W パス間で cold start を揃える用）。"""
        self._store.clear()
        self._lru.clear()
        self.hits = self.misses = 0

    @property
    def hit_rate(self) -> float:
        t = self.hits + self.misses
        return self.hits / t if t else 0.0

    def resident_experts(self) -> int:
        return len(self._store)
