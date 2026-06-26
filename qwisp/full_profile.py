"""全体ボトルネック分解 — engine 構成を段階的に変えて verify forward を測る.

同じ 2-token forward を以下で測り、各層の差から税を切り分ける:
  A resident   : mlx 純正 SwitchGLU（全 expert RAM 常駐, GPU gather, tolist 無し）= 床
  B stream@256 : StreamingSwitchGLU 全常駐（IO 無し・但し per-layer tolist あり）
  C stream@cb  : StreamingSwitchGLU 制約（miss=実 IO）
  D mixed@256  : MixedSwitchGLU hot/cold 全常駐（two-gather, IO 無し）
  E mixed@cb   : MixedSwitchGLU 制約（現行）

税の分解:
  B-A = GPU-routing で消せる machinery 税（tolist 同期 + concat + cache ロジック）
  C-B = IO 税（all4）          D-B = mixed two-gather 税          E-D = mixed IO 税

実行: PY -m qwisp.full_profile <4bit_model> <2bit_dir> [--cold-B 37 --ctx 512 --steps 30]
"""
from __future__ import annotations
import argparse
import sys
import time

import numpy as np
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.qwen3_next import Qwen3NextSparseMoeBlock

from .expert_source import ExpertSource
from .cache import ExpertCache
from .streaming_moe import StreamingSwitchGLU
from .mixed_engine import MixedSwitchGLU, _calibrate
from .mtp_decode import _fwd, _LAYER_RE


def time_fwd(lm, prompt, steps, warm=5):
    """2-token verify forward を steps 回、平均 ms とlm_head 込み/無しを返す。"""
    kv = lm.make_cache()
    H, lg = _fwd(lm, mx.array(prompt)[None], kv); mx.eval(lg)
    u = int(mx.argmax(lg[0, -1]).item()); d = u
    for _ in range(warm):
        H, lg = _fwd(lm, mx.array([[u, d]]), kv); mx.eval(lg); u = int(mx.argmax(lg[0, -1]).item()); d = u
    t_model = t_head = 0.0
    for _ in range(steps):
        t = time.perf_counter(); h = lm.model(mx.array([[u, d]]), cache=kv); mx.eval(h); t_model += time.perf_counter() - t
        t = time.perf_counter(); lg = lm.lm_head(h); mx.eval(lg); t_head += time.perf_counter() - t
        u = int(mx.argmax(lg[0, -1]).item()); d = u
    return t_model / steps * 1e3, t_head / steps * 1e3


def patch_timers():
    """Streaming/Mixed の per-layer tolist 同期と cache.gather を計時。"""
    P = {"tolist": 0.0, "n": 0}
    import qwisp.streaming_moe as sm
    import qwisp.mixed_engine as me

    def wrap(cls):
        orig = cls.__call__
        def timed(self, x, inds):
            t = time.perf_counter(); _ = inds.tolist(); P["tolist"] += time.perf_counter() - t; P["n"] += 1
            return orig(self, x, inds)
        cls.__call__ = timed
        return orig
    o1 = wrap(sm.StreamingSwitchGLU); o2 = wrap(me.MixedSwitchGLU)
    return P, (sm.StreamingSwitchGLU, o1), (me.MixedSwitchGLU, o2)


def attach_streaming(lm, src4, budget):
    c = ExpertCache(src4, budget_per_layer=budget)
    for name, blk in lm.named_modules():
        if isinstance(blk, Qwen3NextSparseMoeBlock):
            L = int(_LAYER_RE.search(name).group(1))
            blk.switch_mlp = StreamingSwitchGLU(src4, L, cache=c)
    return c


def attach_mixed_eng(lm, src4, src2, hot_b, cold_b, counts):
    c4 = ExpertCache(src4, budget_per_layer=hot_b); c2 = ExpertCache(src2, budget_per_layer=cold_b)
    for name, blk in lm.named_modules():
        if isinstance(blk, Qwen3NextSparseMoeBlock):
            L = int(_LAYER_RE.search(name).group(1))
            cc = counts.get(id(blk), np.zeros(256, np.int64))
            hot = set(np.argsort(cc)[-hot_b:].tolist())
            blk.switch_mlp = MixedSwitchGLU(L, hot, c4, c2)
    return c4, c2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("dir2")
    ap.add_argument("--cold-B", type=int, default=37); ap.add_argument("--hot", type=int, default=64)
    ap.add_argument("--ctx", type=int, default=512); ap.add_argument("--steps", type=int, default=30)
    args = ap.parse_args()

    print("[full] load (resident) ...", file=sys.stderr)
    model, tok = load(args.model)
    lm = model.language_model
    base = "def quicksort(a):\n    if len(a)<=1: return a\n    p=a[len(a)//2]\n"
    ids = tok.encode(base)
    while len(ids) < args.ctx:
        ids = ids + tok.encode(base)
    prompt = ids[:args.ctx]

    rows = []
    # A: resident original（tolist 無し）
    m, h = time_fwd(lm, prompt, args.steps)
    rows.append(("A resident", m, h, None))

    # streaming/mixed は tolist を計時
    P, (SSG, oss), (MSG, oms) = patch_timers()
    src4 = ExpertSource(args.model); src2 = ExpertSource(args.dir2)
    counts = _calibrate(model, tok)

    def run(label, attach_fn):
        P["tolist"] = 0.0; P["n"] = 0
        attach_fn()
        m, h = time_fwd(lm, prompt, args.steps)
        rows.append((label, m, h, P["tolist"] / args.steps * 1e3))

    run("B stream@256", lambda: attach_streaming(lm, src4, 256))
    run(f"C stream@{args.cold_B}", lambda: attach_streaming(lm, src4, args.cold_B))
    run("D mixed@256", lambda: attach_mixed_eng(lm, src4, src2, args.hot, 256, counts))
    run(f"E mixed@{args.cold_B}", lambda: attach_mixed_eng(lm, src4, src2, args.hot, args.cold_B, counts))
    SSG.__call__ = oss; MSG.__call__ = oms

    # F: GPU-routed mixed（全常駐, tolist 無し）— 同期税を消した時の上限
    from .gpu_routed import GPURoutedMixedSwitchGLU
    def attach_gpu_routed():
        for name, blk in lm.named_modules():
            if isinstance(blk, Qwen3NextSparseMoeBlock):
                L = int(_LAYER_RE.search(name).group(1))
                cc = counts.get(id(blk), np.zeros(256, np.int64))
                hot = set(np.argsort(cc)[-args.hot:].tolist())
                blk.switch_mlp = GPURoutedMixedSwitchGLU(L, hot, src4, src2)
    attach_gpu_routed()
    m, h = time_fwd(lm, prompt, args.steps)
    rows.append(("F gpu-routed", m, h, 0.0))

    # G: F + ダミー tolist（concat 無し・同期だけ）→ 純粋な同期 drain コストを分離
    GPURoutedMixedSwitchGLU._probe_sync = True
    m, h = time_fwd(lm, prompt, args.steps)
    GPURoutedMixedSwitchGLU._probe_sync = False
    rows.append(("G gpu-routed+sync", m, h, None))

    print(f"\n[full] 2-token verify forward, ctx={args.ctx}, steps={args.steps}, hot={args.hot}")
    print(f"{'config':14} {'model ms':>9} {'lm_head ms':>11} {'tolist ms':>10} {'total ms':>9} {'tok/s(2/fwd)':>12}")
    print("-" * 70)
    a_model = rows[0][1]
    for label, m, h, tl in rows:
        tot = m + h
        tls = f"{tl:.1f}" if tl is not None else "-"
        print(f"{label:14} {m:>9.1f} {h:>11.1f} {tls:>10} {tot:>9.1f} {2/(tot/1e3):>12.1f}")
    # 税の分解
    def get(p): return next(r[1] for r in rows if r[0].startswith(p))
    print("\n[full] 税の分解（model ms）:")
    print(f"  A resident(床)            : {get('A'):.1f}")
    print(f"  B-A = machinery税(tolist+concat, GPU-routing 上振れ余地): {get('B')-get('A'):+.1f}")
    print(f"  C-B = IO税(all4 stream)   : {get('C')-get('B'):+.1f}")
    print(f"  D-B = mixed two-gather税  : {get('D')-get('B'):+.1f}")
    print(f"  E-D = mixed IO税          : {get('E')-get('D'):+.1f}")
    print(f"  E total(現行) vs A床      : {get('E'):.1f} vs {get('A'):.1f}  (streaming税 {get('E')-get('A'):+.1f})")
    print(f"  F gpu-routed mixed(全常駐, tolist 無し): {get('F'):.1f}  "
          f"(D mixed@256={get('D'):.1f} から同期税 {get('D')-get('F'):+.1f} を回収)")
    print(f"  G F+ダミーtolist(concat無・同期のみ): {get('G'):.1f}  "
          f"(G-F={get('G')-get('F'):+.1f}=純 sync drain / D-G={get('D')-get('G'):+.1f}=concat+cache)")


if __name__ == "__main__":
    main()
