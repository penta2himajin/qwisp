"""MTP spec の lossless 違反(match 24/48)を1ロードで切り分ける.

greedy と speculative(light) を同一 4bit モデルで走らせ、
(1)token 列の最初の分岐位置、(2)各 step の (u,d,v,w,accept) トレースを出す。
off-by-one / rollback 破損 / head 品質 のどれかを特定する。

実行: PY -m qwisp.spec_debug <4bit_model> [--ctx 128 --gen 32]
"""
from __future__ import annotations
import argparse
import sys

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import KVCache

from .mtp_head import build_head
from .mtp_decode import _fwd, greedy, _snap_light, _rollback_light


def spec_trace(lm, head, prompt, max_tokens):
    main_kv = lm.make_cache()
    mtp_kv = KVCache()
    p = mx.array(prompt)[None]
    H, lg = _fwd(lm, p, main_kv)
    u_arr = mx.argmax(lg[:, -1:], axis=-1)
    u = int(u_arr.item())
    last_h = H[:, -1:]
    head(H[:, :-1], p[:, 1:], cache=mtp_kv)
    mx.eval(mtp_kv.state)

    out, trace, steps = [], [], 0
    while len(out) < max_tokens:
        steps += 1
        dl = head(last_h, u_arr, cache=mtp_kv)
        d_arr = mx.argmax(dl[:, -1:], axis=-1)
        ud = mx.concatenate([u_arr, d_arr], axis=1)
        snap = _snap_light(main_kv)
        H2, lg2 = _fwd(lm, ud, main_kv)
        vw = mx.argmax(lg2[0, :2], axis=-1)
        d, v, w = (int(x) for x in mx.concatenate([d_arr[0], vw]).tolist())
        out.append(u)
        acc = (v == d)
        trace.append((u, d, v, w, acc))
        if acc:
            out.append(d)
            head(H2[:, 0:1], d_arr, cache=mtp_kv)
            u, u_arr, last_h = w, vw[1:2].reshape(1, 1), H2[:, 1:2]
        else:
            _rollback_light(main_kv, snap, 2)
            uv = mx.concatenate([u_arr, vw[0:1].reshape(1, 1)], axis=1)   # [u,v] 再投入（u は確定）
            H1, _ = _fwd(lm, uv, main_kv)
            u, u_arr, last_h = v, vw[0:1].reshape(1, 1), H1[:, 0:1]
    return out[:max_tokens], trace, steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("--ctx", type=int, default=128)
    ap.add_argument("--gen", type=int, default=32)
    args = ap.parse_args()
    print("[dbg] loading ...", file=sys.stderr)
    model, tok = load(args.model)
    lm = model.language_model
    head = build_head(args.model, lm)

    base = "def quicksort(a):\n    if len(a)<=1: return a\n    p=a[len(a)//2]\n"
    ids = tok.encode(base)
    while len(ids) < args.ctx:
        ids = ids + tok.encode(base)
    prompt = ids[:args.ctx]

    g, _ = greedy(lm, prompt, args.gen)
    s, trace, steps = spec_trace(lm, head, prompt, args.gen)

    # 最初の分岐
    div = next((i for i, (a, b) in enumerate(zip(g, s)) if a != b), None)
    match = sum(1 for a, b in zip(g, s) if a == b)
    acc = sum(1 for t in trace if t[4])
    print(f"\n[dbg] match={match}/{len(g)}  steps={steps}  accept={acc}/{steps}={acc/steps:.3f}")
    print(f"[dbg] 最初の分岐 index = {div}")
    print(f"[dbg] greedy[:16] = {g[:16]}")
    print(f"[dbg] spec  [:16] = {s[:16]}")
    print(f"\n[dbg] step trace (u,d,v,w,accept)  ※ v は greedy の次トークンと一致すべき:")
    gi = 0
    for k, (u, d, v, w, a) in enumerate(trace[:12]):
        gmark = f"g[{gi}]={g[gi] if gi < len(g) else '-'}"
        print(f"  step{k:2}: u={u:6} d={d:6} v={v:6} w={w:6} accept={a}   ({gmark})")
        gi += 2 if a else 1


if __name__ == "__main__":
    main()
