"""hybrid cache の reject rollback が正しく復元するかを単独で検証.

手順: prompt を流す → [a,b] を verify(基準 logits) → snapshot → 再度 [a,b] を流して
rollback → もう一度同じ位置で 1 トークン流し、基準と一致するか。
線形注意(ArraysCache)の recurrent/conv state が shallow snapshot で本当に戻るかを暴く。

実行: PY -m qwisp.rollback_test <4bit_model> [--ctx 64]
"""
from __future__ import annotations
import argparse
import sys

import mlx.core as mx
from mlx_lm import load

from .mtp_decode import _fwd, _snap_light, _rollback_light, _snap, _restore


def argmax_after(lm, kv, toks):
    h, lg = _fwd(lm, toks, kv)
    return lg, h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("--ctx", type=int, default=64)
    args = ap.parse_args()
    print("[rb] loading ...", file=sys.stderr)
    model, tok = load(args.model)
    lm = model.language_model
    base = "def quicksort(a):\n    if len(a)<=1: return a\n    p=a[len(a)//2]\n"
    ids = tok.encode(base)
    while len(ids) < args.ctx:
        ids = ids + tok.encode(base)
    prompt = mx.array(ids[:args.ctx])[None]

    a, b = ids[1], ids[2]                                  # 適当な2トークン
    cont = ids[3]                                          # rollback 後に流す1トークン

    for label, snap_fn, rb_fn in (("light", _snap_light, _rollback_light),
                                  ("heavy", _snap, _restore)):
        kv = lm.make_cache()
        _fwd(lm, prompt, kv)                               # prefill

        # 基準: rollback が完全なら「[a,b] を一度も流さず cont を流した」のと同じになるはず。
        kv_ref = lm.make_cache()
        _fwd(lm, prompt, kv_ref)
        ref_lg, _ = _fwd(lm, mx.array([[cont]]), kv_ref)
        ref_am = int(mx.argmax(ref_lg[0, -1]).item())

        # 検証: snapshot → [a,b] → rollback(2) → cont を流す
        snap = snap_fn(kv)
        _fwd(lm, mx.array([[a, b]]), kv)
        if rb_fn is _rollback_light:
            rb_fn(kv, snap, 2)
        else:
            rb_fn(kv, snap)
        got_lg, _ = _fwd(lm, mx.array([[cont]]), kv)
        got_am = int(mx.argmax(got_lg[0, -1]).item())

        diff = float(mx.max(mx.abs(got_lg[0, -1] - ref_lg[0, -1])).item())
        rel = diff / (float(mx.max(mx.abs(ref_lg[0, -1])).item()) + 1e-9)
        print(f"[rb] {label}: argmax ref={ref_am} got={got_am} "
              f"max|Δ|={diff:.3e} rel={rel:.3e}  "
              f"{'OK' if (got_am == ref_am and rel < 1e-3) else 'BROKEN'}")


if __name__ == "__main__":
    main()
