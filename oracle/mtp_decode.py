"""Stage A milestone-2 — 実 MTP 投機デコード（draft+verify）の end-to-end 実測.

D=1: main の hidden h_i と次トークン u を MTP ヘッドに渡し次々トークン d をドラフト →
main で [u,d] を1パス verify → v=argmax(main後u) が d と一致なら 2トークン受理。

このモデルは hybrid（30/40 が GatedDeltaNet 線形注意）で **KV cache が trim 不能**。
標準投機の reject ロールバックが使えないので、main cache を **state snapshot/restore** で巻き戻す
（KVCache/ArraysCache の .state は round-trip 可、検証済）。MTP cache は draft 位置が常に有効
なのでロールバック不要（accept 時のみ catch-up で1位置追加）。

正しさ: 投機出力は greedy と完全一致するはず（D1 verify が main 等価を保証）。

実行: PY -m oracle.mtp_decode <model_dir> [--mixed <2bit_dir> --hot 64 --cold-B 88 --gen 96]
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
    u_arr = mx.argmax(lg[:, -1:], axis=-1)             # [1,1] GPU 配列（materialize しない）
    u = int(u_arr.item())
    last_h = H[:, -1:]                                  # h_{P-1} [1,1,H]
    head(H[:, :-1], p[:, 1:], cache=mtp_kv)            # MTP prefill
    mx.eval(mtp_kv.state)

    out = []
    steps = 0
    acc_draft = 0
    prof = {"draft": 0.0, "verify": 0.0, "rest": 0.0}
    dec_t0 = time.perf_counter()                        # prefill 除外の decode 計時
    while len(out) < max_tokens:
        steps += 1
        t0 = time.perf_counter()
        # draft: d を materialize せず GPU 配列のまま [u,d] を作る（同期削減の肝）
        dl = head(last_h, u_arr, cache=mtp_kv)
        d_arr = mx.argmax(dl[:, -1:], axis=-1)         # [1,1]
        ud = mx.concatenate([u_arr, d_arr], axis=1)    # [1,2] 配列直接 feed
        prof["draft"] += time.perf_counter() - t0; t0 = time.perf_counter()
        snap = _snap_light(main_kv) if light else _snap(main_kv)
        if pf is not None:
            pf.kick()
        H2, lg2 = _fwd(lm, ud, main_kv)
        vw = mx.argmax(lg2[0, :2], axis=-1)            # v=pos0, w=pos1
        # d,v,w を 1 回の tolist で（per-step 同期 1 回）
        d, v, w = (int(x) for x in mx.concatenate([d_arr[0], vw]).tolist())
        if pf is not None:
            pf.snapshot()
        prof["verify"] += time.perf_counter() - t0; t0 = time.perf_counter()
        out.append(u)
        if v == d:                                     # accept → 2トークン
            acc_draft += 1
            out.append(d)
            head(H2[:, 0:1], d_arr, cache=mtp_kv)      # catch-up mtp
            u, u_arr, last_h = w, vw[1:2].reshape(1, 1), H2[:, 1:2]
        else:                                          # reject: pre-[u,d] に戻し [u,v] を再投入。
            if light:                                  # u は確定トークン → cache に戻さないと
                _rollback_light(main_kv, snap, 2)      # 以降の文脈から u が欠落し lossless 違反になる。
            else:
                _restore(main_kv, snap)
            uv = mx.concatenate([u_arr, vw[0:1].reshape(1, 1)], axis=1)   # [1,2] = [u, v]
            H1, _ = _fwd(lm, uv, main_kv)
            # 次の draft は head(hidden_u, v): u の hidden(=position0) が次トークンの文脈。
            u, u_arr, last_h = v, vw[0:1].reshape(1, 1), H1[:, 0:1]
        prof["rest"] += time.perf_counter() - t0
    mx.eval(last_h)                                    # 末尾の lazy 残を flush して decode 計時を正す
    dec_secs = time.perf_counter() - dec_t0
    if profile:
        tot = sum(prof.values())
        print("[prof] " + "  ".join(f"{k}={v/steps*1e3:.1f}ms({v/tot*100:.0f}%)"
                                    for k, v in prof.items()) + f"  steps={steps}",
              file=sys.stderr)
    return out[:max_tokens], steps, acc_draft, dec_secs


def greedy(lm, prompt, max_tokens):
    """naive AR（decode-only 秒も返す）。配列直接 feed + async で per-token 同期を緩和。"""
    kv = lm.make_cache()
    p = mx.array(prompt)[None]
    _, lg = _fwd(lm, p, kv)
    y = mx.argmax(lg[:, -1:], axis=-1)                  # [1,1] 配列
    mx.async_eval(y)
    arrs = [y]
    dec_t0 = time.perf_counter()
    for _ in range(max_tokens - 1):
        _, lg = _fwd(lm, y, kv)                          # 配列直接 feed
        y = mx.argmax(lg[:, -1:], axis=-1)
        mx.async_eval(y)
        arrs.append(y)
    mx.eval(y)
    dec_secs = time.perf_counter() - dec_t0
    out = [int(a.item()) for a in arrs][:max_tokens]
    return out, dec_secs


def attach_mixed(model, lm, tok, model_dir, dir2, hot_b, cold_b, fast_hot=False,
                 io_workers=8, prefetch=False, memmap=False):
    src4 = ExpertSource(model_dir, memmap=memmap)
    src2 = ExpertSource(dir2, memmap=memmap)
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


def attach_gpu_routed(model, lm, tok, model_dir, dir2, hot_b, adaptive_T=None):
    """GPU-routed mixed（全 expert を持続 GPU バッファに常駐, tolist 無し）。docs/09 §4.7。
    adaptive_T 指定で層別 hot サイズ（calibration-aware）。"""
    from .gpu_routed import GPURoutedMixedSwitchGLU, hot_set_for
    src4 = ExpertSource(model_dir)
    src2 = ExpertSource(dir2)
    counts = _calibrate(model, tok)
    for name, blk in lm.named_modules():
        if isinstance(blk, Qwen3NextSparseMoeBlock):
            layer = int(_LAYER_RE.search(name).group(1))
            c = counts.get(id(blk), np.zeros(256, np.int64))
            hot = hot_set_for(c, hot_b, adaptive_T)
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
    ap.add_argument("--adaptive-T", type=float, default=None,
                    help="GPU-routed の層別 hot サイズ（頻度累積マス閾, 例 0.8）")
    ap.add_argument("--memmap", action="store_true", help="mixed の miss ロードを np.memmap 経由に")
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
    g_ref, _ = greedy(lm, prompt, args.gen)
    for lab, lt in (("light", True), ("heavy", False)):
        sp, steps, accd, _ = speculative(lm, head, prompt, args.gen, light=lt)
        match = sum(1 for a, b in zip(g_ref, sp) if a == b)
        print(f"[dec] correctness({lab}): {match}/{len(g_ref)} "
              f"(steps={steps}, draft_accept={accd/steps:.3f})")

    # 速度: greedy(AR) vs spec(light)、エンジンを選択。tok/s は **decode-only**（prefill 除外）。
    pf = None
    caches = None
    if args.gpu_routed:
        attach_gpu_routed(model, lm, tok, args.model, args.gpu_routed, args.hot, args.adaptive_T)
        print(f"[dec] GPU-routed mixed resident attached (hot={args.hot}"
              f"{f' adaptive_T={args.adaptive_T}' if args.adaptive_T else ''})", file=sys.stderr)
    elif args.mixed:
        caches, pf = attach_mixed(model, lm, tok, args.model, args.mixed, args.hot, args.cold_B,
                                  fast_hot=args.fast_hot, prefetch=args.prefetch, memmap=args.memmap)
        print(f"[dec] mixed streaming attached (hot={args.hot}/cold-B={args.cold_B}"
              f"{' +prefetch' if args.prefetch else ''})", file=sys.stderr)

    g, ar_secs = greedy(lm, prompt, args.gen)                 # decode-only 秒
    sp_l, steps_l, accd, sp_secs = speculative(lm, head, prompt, args.gen, light=True, profile=args.profile)
    eng = ("gpu-routed-mixed" if args.gpu_routed else
           ("mixed-stream" + ("+fasthot" if args.fast_hot else "")) if args.mixed else "full-resident")
    m_l = sum(1 for a, b in zip(g_ref, sp_l) if a == b)
    ar_tps = (args.gen - 1) / ar_secs
    sp_tps = args.gen / sp_secs
    print(f"\n[dec] engine={eng}  gen={args.gen}  ctx={args.ctx}  (decode-only tok/s, prefill 除外)")
    print(f"  AR greedy           : {ar_tps:6.1f} tok/s")
    print(f"  MTP D1 spec light   : {sp_tps:6.1f} tok/s  ({sp_tps/ar_tps:.2f}x)  "
          f"draft_accept={accd/steps_l:.3f}  match={m_l}/{len(g_ref)}")


if __name__ == "__main__":
    main()
