"""memmap を並列化して parallel pread と公平比較する.

前回 memmap が 3.5x 遅かったのは逐次だったため。memmap/mmap API の並列化手段:
  (B) threaded: ThreadPool で memmap fancy-index を分割（page fault 中 GIL 解放なら効く）
  (D) madvise(WILLNEED): 必要 expert ページを非同期 prefetch してから読む（OS readahead 並列）
を、cold ページ(MADV_DONTNEED で evict)で parallel pread と比較する。load→numpy のみ計測。

実行: PY -m qwisp.io_parallel_probe [--k 16 --reps 20]
"""
from __future__ import annotations
import argparse
import json
import mmap as mmaplib
import os
import struct
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

PROJS = ("gate_proj", "up_proj", "down_proj")
PARTS = ("weight", "scales", "biases")
_DT = {"U32": np.uint32, "F16": np.float16, "BF16": np.uint16, "F32": np.float32}
PAGE = 16384  # Apple ARM64 page


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.expanduser("~/.mtplx/models/qwisp-experts-2bit"))
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    path = os.path.join(args.dir, "experts_2bit.safetensors")
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n)); ds = 8 + n

    NL = 40
    # 各 (layer,proj,part): file 絶対 offset(b0), stride(bytes/expert), dtype, row shape
    meta = {}
    for L in range(NL):
        for p in PROJS:
            for q in PARTS:
                key = f"language_model.model.layers.{L}.mlp.switch_mlp.{p}.{q}"
                t = hdr[key]; b, e = t["data_offsets"]
                n_exp = t["shape"][0]; stride = (e - b) // n_exp
                meta[(L, p, q)] = (ds + b, stride, _DT[t["dtype"]], tuple(t["shape"][1:]))

    fd = os.open(path, os.O_RDONLY)
    # 全ファイル mmap（madvise を file-絶対 offset で叩く）
    full = mmaplib.mmap(fd, 0, prot=mmaplib.PROT_READ)
    # per-tensor np.memmap（fancy-index 用）
    nps = {k: np.memmap(path, dtype=v[2], mode="r", offset=v[0], shape=(256,) + v[3])
           for k, v in meta.items()}
    pool = ThreadPoolExecutor(max_workers=args.workers)
    rng = np.random.default_rng(0)

    def evict(L):
        for p in PROJS:
            for q in PARTS:
                b0, stride, _, _ = meta[(L, p, q)]
                pa = (b0 // PAGE) * PAGE
                length = ((b0 + 256 * stride - pa + PAGE - 1) // PAGE) * PAGE
                try: full.madvise(mmaplib.MADV_DONTNEED, pa, length)
                except Exception: pass

    def A_parallel_pread(L, idx):
        jobs = [(meta[(L, p, q)], e) for e in idx for p in PROJS for q in PARTS]
        def rd(j):
            (b0, stride, dt, sh), e = j
            return np.frombuffer(os.pread(fd, stride, b0 + e * stride), dt)
        return list(pool.map(rd, jobs))

    def B_serial_memmap(L, idx):
        return [np.ascontiguousarray(nps[(L, p, q)][idx]) for p in PROJS for q in PARTS]

    def C_threaded_memmap(L, idx):
        keys = [(L, p, q) for p in PROJS for q in PARTS]
        return list(pool.map(lambda k: np.ascontiguousarray(nps[k][idx]), keys))

    def D_madvise_memmap(L, idx):
        # 必要 expert ページを WILLNEED で非同期 prefetch → 読む
        for p in PROJS:
            for q in PARTS:
                b0, stride, _, _ = meta[(L, p, q)]
                for e in idx:
                    off = b0 + e * stride
                    pa = (off // PAGE) * PAGE
                    length = ((off + stride - pa + PAGE - 1) // PAGE) * PAGE
                    try: full.madvise(mmaplib.MADV_WILLNEED, pa, length)
                    except Exception: pass
        return [np.ascontiguousarray(nps[(L, p, q)][idx]) for p in PROJS for q in PARTS]

    def D2_madvise_threaded(L, idx):
        for p in PROJS:
            for q in PARTS:
                b0, stride, _, _ = meta[(L, p, q)]
                for e in idx:
                    off = b0 + e * stride; pa = (off // PAGE) * PAGE
                    length = ((off + stride - pa + PAGE - 1) // PAGE) * PAGE
                    try: full.madvise(mmaplib.MADV_WILLNEED, pa, length)
                    except Exception: pass
        keys = [(L, p, q) for p in PROJS for q in PARTS]
        return list(pool.map(lambda k: np.ascontiguousarray(nps[k][idx]), keys))

    methods = [("A parallel-pread(現行)", A_parallel_pread), ("B serial-memmap", B_serial_memmap),
               ("C threaded-memmap", C_threaded_memmap), ("D madvise-memmap", D_madvise_memmap),
               ("D2 madvise+threaded", D2_madvise_threaded)]

    print(f"\n[iop] k={args.k} expert, cold(MADV_DONTNEED で evict), reps={args.reps}, workers={args.workers}")
    print("-" * 56)
    for name, fn in methods:
        # warm 1 回（JIT/alloc）→ 各 rep は別 layer を cold 化して計測
        fn(0, list(range(args.k)))
        ts = []
        for r in range(args.reps):
            L = int(rng.integers(0, NL))
            idx = rng.integers(0, 256, args.k).tolist()
            evict(L)
            t = time.perf_counter(); fn(L, idx); ts.append(time.perf_counter() - t)
        print(f"  {name:24}: {np.median(ts)*1e3:7.3f} ms (median, k={args.k})")
    print("\n  ※ A が基準。memmap 系(C/D/D2)が A 以下なら memmap も並列化で対等以上")


if __name__ == "__main__":
    main()
