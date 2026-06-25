#!/usr/bin/env python3
"""Qwisp Step 3① — 実効 flash 帯域の実測（go/no-go ゲート再較正用）。

go/no-go ゲートの最重要・未検証仮定は flash 帯域（docs/05 は保守値 1GB/s を仮置き）。
ここを実数化する。**1 expert 相当（~1.6MB）のランダム読みの持続スループット**を、
**page cache をバイパス**して測る（macOS は fcntl F_NOCACHE）。

測定対象は実モデルの safetensors シャード（experts が実在する NAND 上の同じ経路）。
チャンクサイズを振り、LLM-in-a-flash の「大きな連続読みほど速い（bundling）」も観測する。

出力の `~1.6MB / nocache` の MB/s を Step 2 に差し戻す:
    python ../step2_cache_sim/simulate.py --trace ... --flash-bw <実測bytes/s>

使い方:
    python bench_flash.py --model "$HOME/.mtplx/models/Youssofal--...-FP16"
    python bench_flash.py --file /path/to/shard.safetensors --reads 400
"""

import argparse
import fcntl
import glob
import os
import random
import sys
import time

F_NOCACHE = getattr(fcntl, "F_NOCACHE", 48)  # macOS: 48
MB = 1024 * 1024


def pick_file(args):
    if args.file:
        return args.file
    shards = sorted(glob.glob(os.path.join(args.model, "*.safetensors")),
                    key=os.path.getsize, reverse=True)
    if not shards:
        raise SystemExit("model dir に .safetensors が無い。--file 指定を。")
    return shards[0]  # 最大シャード


def bench(path, chunk, reads, nocache, seed):
    size = os.path.getsize(path)
    if size <= chunk:
        raise SystemExit(f"file ({size}B) が chunk ({chunk}B) より小さい")
    fd = os.open(path, os.O_RDONLY)
    try:
        if nocache:
            fcntl.fcntl(fd, F_NOCACHE, 1)
        rng = random.Random(seed)
        max_off = size - chunk
        # 計測（4K 整列したランダムオフセット）
        t0 = time.perf_counter()
        total = 0
        for _ in range(reads):
            off = rng.randrange(0, max_off) & ~0xFFF
            total += len(os.pread(fd, chunk, off))
        dt = time.perf_counter() - t0
    finally:
        os.close(fd)
    return total / dt, dt / reads  # bytes/s, 平均秒/read


def main():
    ap = argparse.ArgumentParser(description="Qwisp Step3 flash bandwidth bench")
    ap.add_argument("--model", help="モデルディレクトリ（最大シャードを使う）")
    ap.add_argument("--file", help="直接ファイル指定")
    ap.add_argument("--reads", type=int, default=400)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--chunks-mb", default="0.5,1.0,1.6,2.0,4.0,8.0")
    args = ap.parse_args()

    path = pick_file(args)
    size = os.path.getsize(path)
    print(f"[flash] target: {path} ({size/1e9:.1f} GB)", file=sys.stderr)
    print(f"[flash] reads={args.reads}/config  (F_NOCACHE={F_NOCACHE})", file=sys.stderr)
    chunks = [float(x) for x in args.chunks_mb.split(",")]

    hdr = f"{'chunkMB':>8} {'mode':>9} {'MB/s':>9} {'ms/read':>9}  {'tok-budget@15tps':>16}"
    print(hdr)
    print("-" * len(hdr))
    results = {}
    for ch in chunks:
        chunk = int(ch * MB)
        for nocache in (True, False):
            bps, spr = bench(path, chunk, args.reads, nocache, args.seed)
            mode = "nocache" if nocache else "cached"
            # 参考: 320 expert読み/token を 1/15s=66.7ms に収める観点での感触
            print(f"{ch:>8.1f} {mode:>9} {bps/MB:>9.1f} {spr*1000:>9.3f}", flush=True)
            if nocache:
                results[ch] = bps

    key = min(results, key=lambda c: abs(c - 1.6))  # ~1.6MB に最も近い
    bps = results[key]
    print(f"\n[flash] 実効ランダム読み（nocache, {key}MB）= {bps/MB:.1f} MB/s = {bps:.3e} B/s",
          file=sys.stderr)
    print(f"[flash] → Step2 再較正: --flash-bw {int(bps)}", file=sys.stderr)
    # 1GB/s 仮定との比
    print(f"[flash] 保守仮定 1GB/s に対して {bps/1e9:.2f}×", file=sys.stderr)


if __name__ == "__main__":
    main()
