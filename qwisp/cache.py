"""ExpertCache — per-layer LRU の常駐 expert キャッシュ（Step2 sim の予算 B に対応）.

各層が独立に B 個の expert を常駐（hit なら再 pread しない）。gather(layer, U) は U を
stack で返し、使った expert を LRU に入れて B を超えたら oldest を evict。

v0.1 は LRU。v0.2 で MTP 予測器 prefetch を足す（gather 前に next-token 予測の experts を
先読み）。SwiftLM の「app-LRU < OS-cache」は zero-copy mmap 文脈の話で、copy する我々の
経路では明示キャッシュが必要（docs/07）。
"""

from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import mlx.core as mx

from .expert_source import PARTS, PROJS


class ExpertCache:
    def __init__(self, source, budget_per_layer: int, io_workers: int = 8):
        self.src = source
        self.B = budget_per_layer
        self._store: dict[tuple[int, int], dict[str, mx.array]] = {}
        self._lru: dict[int, "OrderedDict[int, None]"] = {}
        self.hits = 0
        self.misses = 0
        # miss は os.pread×9/expert＝syscall latency 律速 → スレッドプールで並列化（GIL 解放）。
        self._pool = ThreadPoolExecutor(max_workers=io_workers) if io_workers > 0 else None

    def gather(self, layer: int, U: list[int]) -> dict[str, mx.array]:
        lru = self._lru.setdefault(layer, OrderedDict())
        miss = [e for e in U if (layer, e) not in self._store]
        if miss:
            loaded = self.src.load_expert_slices(layer, miss, self._pool)  # 並列 pread
            for e in miss:
                self._store[(layer, e)] = loaded[e]
        self.hits += len(U) - len(miss)
        self.misses += len(miss)
        for e in U:                                # LRU 更新（現 U は most-recent）
            lru[e] = None
            lru.move_to_end(e)
        per = [self._store[(layer, e)] for e in U]
        sub = {}
        for p in PROJS:
            for q in PARTS:
                pp = f"{p}.{q}"
                sub[pp] = mx.concatenate([s[pp] for s in per], axis=0)
        while len(lru) > self.B:                   # B 超過を evict（現 U は残る）
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
