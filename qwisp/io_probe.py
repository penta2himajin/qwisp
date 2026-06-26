"""8GB streaming の IO 律速要素を特定 — SSD帯域 か syscallレイテンシ か メモリコピー か.

各 cold expert は 9 スライス(proj×part)を散在オフセットから pread＝小読み×9。これが
(a)SSD 帯域律速（bytes/peakBW）なら 2bit 已に最小でコピー overlap しか手段が無い
(b)レイテンシ律速（syscall/seek 多発）なら 9→1 読みへの再配置/結合で大幅短縮可能
を実測で切り分ける。F_NOCACHE で page-cache を外し真の SSD 読みを測る。

実行: PY -m qwisp.io_probe [--experts 64]
"""
from __future__ import annotations
import argparse
import fcntl
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .expert_source import ExpertSource, PROJS, PARTS

F_NOCACHE = 48  # macOS fcntl: uncached I/O


def open_nocache(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        fcntl.fcntl(fd, F_NOCACHE, 1)
    except OSError:
        pass
    return fd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.expanduser("~/.mtplx/models/qwisp-experts-2bit"))
    ap.add_argument("--experts", type=int, default=64)
    args = ap.parse_args()

    src = ExpertSource(args.dir)
    shard = src.wm[src._key(0, "gate_proj", "weight")]
    path = os.path.join(args.dir, shard)
    hdr, data_start = src._header(shard)

    # 各 (proj,part) の expert あたり stride（バイト）を集計
    slices = []   # (proj, part, stride, base_offset_of_expert0)
    bytes_per_expert = 0
    for p in PROJS:
        for q in PARTS:
            t = hdr[src._key(0, p, q)]
            b, end = t["data_offsets"]
            n_exp = t["shape"][0]
            stride = (end - b) // n_exp
            slices.append((p, q, stride, data_start + b))
            bytes_per_expert += stride
    print(f"[io] bytes/expert = {bytes_per_expert/1024:.1f} KB  ({len(slices)} スライス/expert)")
    print(f"[io] スライスサイズ: {[s[2]//1024 for s in slices]} KB")

    rng = np.random.default_rng(0)

    # --- (1) ピーク逐次 SSD 帯域: 200MB 連続を F_NOCACHE で読む ---
    fd = open_nocache(path)
    CH = 200 * 1024 * 1024
    t = time.perf_counter()
    os.pread(fd, CH, data_start)
    dt = time.perf_counter() - t
    peak_bw = CH / dt / 1e9
    os.close(fd)
    print(f"\n[io] ピーク逐次BW (200MB連続, NOCACHE): {peak_bw:.2f} GB/s  ({dt*1e3:.1f}ms)")

    # --- (2) cold ランダム expert 読み（NOCACHE）: 9 散在 pread を逐次 ---
    def read_expert_seq(fd, layer, e):
        for p, q, stride, base in slices:
            os.pread(fd, stride, base + e * stride)   # layer 0 固定（base は layer0）

    fd = open_nocache(path)
    exps = rng.integers(0, 256, size=args.experts).tolist()
    t = time.perf_counter()
    for e in exps:
        read_expert_seq(fd, 0, e)
    dt = time.perf_counter() - t
    os.close(fd)
    per_exp_seq = dt / args.experts * 1e3
    bw_seq = bytes_per_expert * args.experts / dt / 1e9
    print(f"[io] cold逐次: {per_exp_seq:.3f} ms/expert  実効BW {bw_seq:.2f} GB/s  "
          f"({args.experts}exp, {len(slices)*args.experts} syscall)")

    # --- (3) cold ランダム expert 読み（NOCACHE）: ThreadPool 並列 ---
    fd = open_nocache(path)
    pool = ThreadPoolExecutor(max_workers=8)
    jobs = [(e, p, q, stride, base) for e in exps for (p, q, stride, base) in slices]
    t = time.perf_counter()
    list(pool.map(lambda j: os.pread(fd, j[3], j[4] + j[0] * j[3]), jobs))
    dt = time.perf_counter() - t
    os.close(fd)
    per_exp_par = dt / args.experts * 1e3
    bw_par = bytes_per_expert * args.experts / dt / 1e9
    print(f"[io] cold並列(8w): {per_exp_par:.3f} ms/expert  実効BW {bw_par:.2f} GB/s")

    # --- (4) 理論: bytes/peakBW（帯域律速ならこれに近い） ---
    bw_floor = bytes_per_expert / (peak_bw * 1e9) * 1e3
    print(f"\n[io] 帯域律速の理論floor: {bw_floor:.3f} ms/expert (= {bytes_per_expert/1024:.0f}KB / {peak_bw:.1f}GB/s)")
    print(f"[io] 律速判定:")
    print(f"     cold並列 {per_exp_par:.3f}ms vs 帯域floor {bw_floor:.3f}ms "
          f"→ {'帯域律速(これ以上は overlap のみ)' if per_exp_par < bw_floor*2 else 'レイテンシ律速(9→1結合で短縮可)'}")
    print(f"     並列/逐次 = {per_exp_seq/per_exp_par:.2f}x （>2 ならレイテンシ律速の証左）")


if __name__ == "__main__":
    main()
