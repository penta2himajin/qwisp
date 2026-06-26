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
from .mixed_engine import MixedSwitchGLU, _calibrate, Prefetcher
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


def _snap_light(kv):
    """非trimmable cache(線形注意 ArraysCache, 固定小サイズ)だけ shallow snapshot。
    KVCache(full-attn) は trim で巻き戻すので snapshot 不要。array コピー無し（参照のみ）。"""
    return [(i, list(c.state)) for i, c in enumerate(kv) if not c.is_trimmable()]


def _rollback_light(kv, snap, n):
    """reject 時の巻き戻し: trimmable は trim(n)、非trimmable は restore（→ pre-[u,d]）。"""
    for c in kv:
        if c.is_trimmable():
            c.trim(n)
    for i, s in snap:
        kv[i].state = s


def speculative(lm, head, prompt, max_tokens, light=True, profile=False, pf=None):
    """MTP D1 投機デコード。返り値: (tokens, n_steps, n_accept_draft)。
    light=True: hybrid 用の軽量巻き戻し（KVCache=trim / ArraysCache=shallow snapshot）。"""
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
    prof = {"draft": 0.0, "verify": 0.0, "catchup": 0.0, "reject": 0.0}
    while len(out) < max_tokens:
        steps += 1
        t0 = time.perf_counter()
        # draft d from (last_h, u)
        dl = head(last_h, mx.array([[u]]), cache=mtp_kv)
        d = int(mx.argmax(dl[0, -1]).item())
        prof["draft"] += time.perf_counter() - t0; t0 = time.perf_counter()
        # verify: main on [u,d]（reject 巻き戻し用に snapshot）
        snap = _snap_light(main_kv) if light else _snap(main_kv)
        if pf is not None:
            pf.kick()                              # 前 forward の cold を背景ロード（この verify と overlap）
        H2, lg2 = _fwd(lm, mx.array([[u, d]]), main_kv)
        v = int(mx.argmax(lg2[0, 0]).item())
        if pf is not None:
            pf.snapshot()
        prof["verify"] += time.perf_counter() - t0; t0 = time.perf_counter()
        out.append(u)
        if v == d:                                     # accept draft
            acc_draft += 1
            out.append(d)
            w = int(mx.argmax(lg2[0, 1]).item())
            head(H2[:, 0:1], mx.array([[d]]), cache=mtp_kv)   # catch-up mtp 位置
            u, last_h = w, H2[:, 1:2]
            mx.eval([c.state for c in main_kv] + [mtp_kv.state])
            prof["catchup"] += time.perf_counter() - t0
        else:                                          # reject: pre-[u,d] に戻し u のみ再処理
            if light:
                _rollback_light(main_kv, snap, 2)
            else:
                _restore(main_kv, snap)
            H1, _ = _fwd(lm, mx.array([[u]]), main_kv)
            u, last_h = v, H1[:, 0:1]
            mx.eval([c.state for c in main_kv] + [mtp_kv.state])
            prof["reject"] += time.perf_counter() - t0
    if profile:
        tot = sum(prof.values())
        print("[prof] " + "  ".join(f"{k}={v/steps*1e3:.1f}ms({v/tot*100:.0f}%)"
                                    for k, v in prof.items()) + f"  steps={steps}",
              file=sys.stderr)
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


def attach_mixed(model, lm, tok, model_dir, dir2, hot_b, cold_b, fast_hot=False,
                 io_workers=8, prefetch=False):
    src4 = ExpertSource(model_dir)
    src2 = ExpertSource(dir2)
    c4 = ExpertCache(src4, budget_per_layer=hot_b, io_workers=io_workers)
    c2 = ExpertCache(src2, budget_per_layer=cold_b, io_workers=io_workers)
    counts = _calibrate(model, tok)
    recorder = {} if prefetch else None
    for name, blk in lm.named_modules():
        if isinstance(blk, Qwen3NextSparseMoeBlock):
            m = _LAYER_RE.search(name)
            layer = int(m.group(1)) if m else 0
            c = counts.get(id(blk), np.zeros(256, np.int64))
            hot = set(np.argsort(c)[-hot_b:].tolist())
            sg = MixedSwitchGLU(layer, hot, c4, c2, fast_hot=fast_hot)
            sg._recorder = recorder
            blk.switch_mlp = sg
    pf = Prefetcher(c2, recorder) if prefetch else None
    return [c4, c2], pf


def attach_gpu_routed(model, lm, tok, model_dir, dir2, hot_b):
    """GPU-routed mixed（全 expert を持続 GPU バッファに常駐, tolist 無し）。docs/09 §4.7。"""
    from .gpu_routed import GPURoutedMixedSwitchGLU
    src4 = ExpertSource(model_dir)
    src2 = ExpertSource(dir2)
    counts = _calibrate(model, tok)
    for name, blk in lm.named_modules():
        if isinstance(blk, Qwen3NextSparseMoeBlock):
            layer = int(_LAYER_RE.search(name).group(1))
            c = counts.get(id(blk), np.zeros(256, np.int64))
            hot = set(np.argsort(c)[-hot_b:].tolist())
            blk.switch_mlp = GPURoutedMixedSwitchGLU(layer, hot, src4, src2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--mixed", default=None, help="2bit dir（指定で mixed streaming）")
    ap.add_argument("--gpu-routed", default=None, help="2bit dir（指定で GPU-routed mixed 常駐）")
    ap.add_argument("--hot", type=int, default=64)
    ap.add_argument("--cold-B", type=int, default=88)
    ap.add_argument("--gen", type=int, default=96)
    ap.add_argument("--ctx", type=int, default=128)
    ap.add_argument("--profile", action="store_true", help="ステップ内訳を計測表示")
    ap.add_argument("--fast-hot", action="store_true", help="持続 hot バッファ + GPU remap")
    ap.add_argument("--prefetch", action="store_true", help="async cold prefetch を有効化")
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

    # 正しさ: light/heavy 両方が greedy と一致するか（full-residency 参照）
    g_ref = greedy(lm, prompt, args.gen)
    for lab, lt in (("light", True), ("heavy", False)):
        sp, steps, accd = speculative(lm, head, prompt, args.gen, light=lt)
        match = sum(1 for a, b in zip(g_ref, sp) if a == b)
        print(f"[dec] correctness({lab}): {match}/{len(g_ref)} "
              f"(steps={steps}, draft_accept={accd/steps:.3f})")

    # 速度: greedy(AR) vs spec(light) ± prefetch、エンジンを選択
    pf = None
    caches = None
    if args.gpu_routed:
        attach_gpu_routed(model, lm, tok, args.model, args.gpu_routed, args.hot)
        print(f"[dec] GPU-routed mixed resident attached (hot={args.hot})", file=sys.stderr)
    elif args.mixed:
        caches, pf = attach_mixed(model, lm, tok, args.model, args.mixed, args.hot, args.cold_B,
                                  fast_hot=args.fast_hot, prefetch=args.prefetch)
        print(f"[dec] mixed streaming attached (hot={args.hot}/cold-B={args.cold_B}"
              f"{' +prefetch' if args.prefetch else ''})", file=sys.stderr)

    def timed(fn):
        t0 = time.perf_counter(); r = fn(); return r, time.perf_counter() - t0

    (g, _), t_ar = timed(lambda: (greedy(lm, prompt, args.gen), 0))
    eng = ("gpu-routed-mixed" if args.gpu_routed else
           ("mixed-stream" + ("+fasthot" if args.fast_hot else "")) if args.mixed else "full-resident")
    print(f"\n[dec] engine={eng}  gen={args.gen}  ctx={args.ctx}")
    print(f"  AR greedy           : {args.gen/t_ar:6.1f} tok/s")
    # warmup 影響を排すため 2 回ずつ交互計測
    if pf is not None:
        (sp_p, _, _), t_p = timed(lambda: speculative(lm, head, prompt, args.gen, light=True, pf=pf))
        ph = sum(c.prefetch_hits for c in caches)
    (sp_l, steps_l, accd), t_l = timed(lambda: speculative(lm, head, prompt, args.gen, light=True, profile=args.profile))
    m_l = sum(1 for a, b in zip(g_ref, sp_l) if a == b)
    print(f"  MTP D1 spec light   : {args.gen/t_l:6.1f} tok/s  ({t_ar/t_l:.2f}x)  "
          f"draft_accept={accd/steps_l:.3f}  match={m_l}/{len(g_ref)}")
    if pf is not None:
        m_p = sum(1 for a, b in zip(g_ref, sp_p) if a == b)
        print(f"  MTP D1 spec +prefetch: {args.gen/t_p:6.1f} tok/s  ({t_ar/t_p:.2f}x)  "
              f"match={m_p}/{len(g_ref)}  prefetch_hits={ph}  (順序: pf を先に計測)")


if __name__ == "__main__":
    main()
