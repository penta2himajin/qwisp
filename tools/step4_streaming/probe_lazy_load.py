#!/usr/bin/env python3
"""Step4 決定実験：mlx_lm.load が expert を遅延 mmap に保つか（アーキ (A)/(B) 選択）.

RSS（resident set size）を load 前/後/生成後で測る。mmap'd ファイルページは fault 時のみ RSS に乗る。
- load 後 RSS ~20GB → eager 実体化 → (A) 明示キャッシュ必須（12GB 機ではロードすら不可）。
- load 後 RSS 小・生成後も < ~12GB → (B) 遅延 mmap＋OS キャッシュ委任が viable。
- gather_qmm がスタック expert を「選択行だけ fault」するか「全体 fault」するかが分岐点。

macOS の ru_maxrss は bytes（Linux は KB）。
"""
import resource
import sys
import time

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import stream_generate
from mlx_lm.sample_utils import make_sampler


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9  # macOS: bytes


def mxpeak():
    f = getattr(mx, "get_peak_memory", None)
    return (f() / 1e9) if f else float("nan")


def main():
    path = sys.argv[1]
    gen_n = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    print(f"[probe] before load: RSS={rss_gb():.2f} GB", flush=True)
    t = time.perf_counter()
    model, tok = load(path)
    print(f"[probe] after load:  RSS={rss_gb():.2f} GB  (load {time.perf_counter()-t:.1f}s)", flush=True)

    sampler = make_sampler(temp=0.0)
    ids = tok.encode("def fib(n):\n    ")[:8]
    n = 0
    for _ in stream_generate(model, tok, prompt=ids, max_tokens=gen_n, sampler=sampler):
        n += 1
    print(f"[probe] after gen {n}:  RSS={rss_gb():.2f} GB  mx_peak={mxpeak():.2f} GB", flush=True)

    rss = rss_gb()
    print(f"\n[probe] VERDICT: "
          + ("(B) viable — 遅延 mmap で resident 抑制" if rss < 14
             else "(A) 必須 — eager 実体化（~全 expert 常駐）"), flush=True)


if __name__ == "__main__":
    main()
