"""Stage A milestone-2 — 実 MTP 投機デコード（draft+verify）の end-to-end 実測.

D=1: main の hidden h_i と次トークン u を MTP ヘッドに渡し次々トークン d をドラフト →
main で [u,d] を1パス verify → v=argmax(main後u) が d と一致なら 2トークン受理。

このモデルは hybrid（30/40 が GatedDeltaNet 線形注意）で **KV cache が trim 不能**。
標準投機の reject ロールバックが使えないので、main cache を **state snapshot/restore** で巻き戻す
（KVCache/ArraysCache の .state は round-trip 可、検証済）。MTP cache は draft 位置が常に有効
なのでロールバック不要（accept 時のみ catch-up で1位置追加）。

正しさ: 投機出力は greedy と完全一致するはず（D1 verify が main 等価を保証）。

実行: PY -m qwisp.mtp_decode <model_dir> [--mixed <2bit_dir> --hot 64 --cold-B 88 --gen 96]
"""
from __future__ import annotations
import argparse
import re
import sys
import time

import numpy as np
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import KVCache
from mlx_lm.models.qwen3_next import Qwen3NextSparseMoeBlock

from .expert_source import ExpertSource
from .cache import ExpertCache
from .loader import load_streaming
from .streaming_moe import StreamingSwitchGLU
from .mixed_engine import MixedSwitchGLU, _calibrate
from .mtp_head import build_head, load_mtp_weights, MTPHead

_LAYER_RE = re.compile(r"\.layers\.(\d+)\.")


def _snap(kv):
    out = []
    for c in kv:
        s = c.state
        if isinstance(s, (list, tuple)):
            out.append([mx.array(a) if a is not None else None for a in s])
        else:
            out.append(mx.array(s) if s is not None else None)
    return out


def _restore(kv, snap):
    for c, s in zip(kv, snap):
        c.state = s


def _fwd(lm, toks, kv):
    """main を回し (hidden[1,L,H], logits[1,L,V])。"""
    h = lm.model(toks, cache=kv)
    return h, lm.lm_head(h)


def speculative(lm, head, prompt, max_tokens):
    """MTP D1 投機デコード。返り値: (tokens, n_steps, n_accept_draft)。"""
    main_kv = lm.make_cache()
    mtp_kv = KVCache()
    p = mx.array(prompt)[None]
    H, lg = _fwd(lm, p, main_kv)                       # prefill
    u = int(mx.argmax(lg[0, -1]).item())
    last_h = H[:, -1:]                                  # h_{P-1} [1,1,H]
    # MTP prefill: 位置 0..P-2（x_i=fc(emb(t_{i+1}),h_i)）
    head(H[:, :-1], p[:, 1:], cache=mtp_kv)
    mx.eval(mtp_kv.state)

    out = []
    steps = 0
    acc_draft = 0
    while len(out) < max_tokens:
        steps += 1
        # draft d from (last_h, u)
        dl = head(last_h, mx.array([[u]]), cache=mtp_kv)
        d = int(mx.argmax(dl[0, -1]).item())
        # verify: main on [u,d]（snapshot で reject 巻き戻し可能に）
        snap = _snap(main_kv)
        H2, lg2 = _fwd(lm, mx.array([[u, d]]), main_kv)
        v = int(mx.argmax(lg2[0, 0]).item())
        out.append(u)
        if v == d:                                     # accept draft
            acc_draft += 1
            out.append(d)
            w = int(mx.argmax(lg2[0, 1]).item())
            head(H2[:, 0:1], mx.array([[d]]), cache=mtp_kv)   # catch-up mtp 位置
            u, last_h = w, H2[:, 1:2]
        else:                                          # reject: main を u のみ commit に巻き戻し
            _restore(main_kv, snap)
            H1, _ = _fwd(lm, mx.array([[u]]), main_kv)
            u, last_h = v, H1[:, 0:1]
        mx.eval([c.state for c in main_kv] + [mtp_kv.state])
    return out[:max_tokens], steps, acc_draft


def greedy(lm, prompt, max_tokens):
    kv = lm.make_cache()
    p = mx.array(prompt)[None]
    _, lg = _fwd(lm, p, kv)
    t = int(mx.argmax(lg[0, -1]).item())
    out = [t]
    while len(out) < max_tokens:
        _, lg = _fwd(lm, mx.array([[t]]), kv)
        t = int(mx.argmax(lg[0, -1]).item())
        out.append(t)
    return out[:max_tokens]


def attach_mixed(model, lm, tok, model_dir, dir2, hot_b, cold_b):
    src4 = ExpertSource(model_dir)
    src2 = ExpertSource(dir2)
    c4 = ExpertCache(src4, budget_per_layer=hot_b)
    c2 = ExpertCache(src2, budget_per_layer=cold_b)
    counts = _calibrate(model, tok)
    for name, blk in lm.named_modules():
        if isinstance(blk, Qwen3NextSparseMoeBlock):
            m = _LAYER_RE.search(name)
            layer = int(m.group(1)) if m else 0
            c = counts.get(id(blk), np.zeros(256, np.int64))
            hot = set(np.argsort(c)[-hot_b:].tolist())
            blk.switch_mlp = MixedSwitchGLU(layer, hot, c4, c2)
    return [c4, c2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--mixed", default=None, help="2bit dir（指定で mixed streaming）")
    ap.add_argument("--hot", type=int, default=64)
    ap.add_argument("--cold-B", type=int, default=88)
    ap.add_argument("--gen", type=int, default=96)
    ap.add_argument("--ctx", type=int, default=128)
    args = ap.parse_args()

    print("[dec] loading ...", file=sys.stderr)
    model, tok = load(args.model)
    lm = model.language_model
    head = build_head(args.model, lm)

    base = "def quicksort(a):\n    if len(a)<=1: return a\n    p=a[len(a)//2]\n"
    ids = tok.encode(base)
    while len(ids) < args.ctx:
        ids = ids + tok.encode(base)
    prompt = ids[:args.ctx]

    # 正しさ: full-residency で greedy と投機が一致するか
    g = greedy(lm, prompt, args.gen)
    sp, steps, accd = speculative(lm, head, prompt, args.gen)
    match = sum(1 for a, b in zip(g, sp) if a == b)
    print(f"[dec] correctness: spec vs greedy {match}/{len(g)} "
          f"(steps={steps}, draft_accept={accd}/{steps}={accd/steps:.3f})")

    # 速度: greedy(AR) vs spec、必要なら mixed streaming で
    caches = None
    if args.mixed:
        caches = attach_mixed(model, lm, tok, args.model, args.mixed, args.hot, args.cold_B)
        print(f"[dec] mixed streaming attached (hot={args.hot}/cold-B={args.cold_B})",
              file=sys.stderr)

    t0 = time.perf_counter(); g = greedy(lm, prompt, args.gen); t_ar = time.perf_counter() - t0
    t0 = time.perf_counter(); sp, steps, accd = speculative(lm, head, prompt, args.gen)
    t_sp = time.perf_counter() - t0
    match = sum(1 for a, b in zip(g, sp) if a == b)
    eng = "mixed-stream" if args.mixed else "full-resident"
    print(f"\n[dec] engine={eng}  gen={args.gen}")
    print(f"  AR  greedy : {args.gen/t_ar:6.1f} tok/s")
    print(f"  MTP D1 spec: {args.gen/t_sp:6.1f} tok/s  ({t_ar/t_sp:.2f}x)  "
          f"draft_accept={accd/steps:.3f} steps={steps} match={match}/{len(g)}")


if __name__ == "__main__":
    main()
