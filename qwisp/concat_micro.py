"""streaming concat 税の分解 — モデル無し・実形状の合成 expert slice で測る.

ExpertCache.gather は毎 forward、常駐 expert slice を 9 テンソル(proj×part)に concat し直す
（hit でも再 stack）。docs/09 で D-G=+90ms の支配コスト。ここでは 12GB ロード無しに:
  (1) concat 自体のコスト（|U| と launch-overhead の効き）
  (2) arena+LUT gather_qmm の天井（GPU-routed=concat 無しの F 相当）
  (3) concat+gather_qmm の合計に占める concat 割合
を測り、Metal 融合カーネルの回収上限と「9 concat 集約」の効果を見る。

実形状: gate/up out512 in2048, down out2048 in512, 4bit gs64, top_k=8。
実行: PY -m qwisp.concat_micro [--U 12 --steps 300]
"""
from __future__ import annotations
import argparse
import time

import numpy as np
import mlx.core as mx
from mlx_lm.models.activations import swiglu

from .expert_source import PROJS, PARTS

E = 256
GS = 64
SHP = {"gate_proj": (512, 2048), "up_proj": (512, 2048), "down_proj": (2048, 512)}


def make_experts():
    """256 expert を {e: {proj.part: [1,...]}} で（量子化済テンソル, 実 dtype）。"""
    exp = {}
    packed = {}   # arena 構築用に proj.part -> [E,...] も作る
    for proj, (o, i) in SHP.items():
        fp = mx.random.normal((E, o, i)) * 0.02
        w, s, b = mx.quantize(fp, group_size=GS, bits=4)
        packed[f"{proj}.weight"] = mx.contiguous(w)
        packed[f"{proj}.scales"] = mx.contiguous(s)
        packed[f"{proj}.biases"] = mx.contiguous(b)
    mx.eval(list(packed.values()))
    for e in range(E):
        d = {}
        for k, arr in packed.items():
            d[k] = arr[e:e + 1]                 # [1,...] slice（streaming store と同形）
        exp[e] = d
    mx.eval([v for d in exp.values() for v in d.values()])
    return exp, packed


def bench(fn, steps, warm=20):
    for _ in range(warm):
        mx.eval(fn())
    t = time.perf_counter()
    for _ in range(steps):
        mx.eval(fn())
    return (time.perf_counter() - t) / steps * 1e3


def qmm_from(buf, x, remap, bits=4):
    xe = mx.expand_dims(x, (-2, -3))

    def q(xx, proj):
        return mx.gather_qmm(xx, buf[f"{proj}.weight"], buf[f"{proj}.scales"],
                             buf[f"{proj}.biases"], rhs_indices=remap, transpose=True,
                             group_size=GS, bits=4, mode="affine", sorted_indices=False)
    h = swiglu(q(xe, "gate_proj"), q(xe, "up_proj"))
    return q(h, "down_proj").squeeze(-2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--U", type=int, default=12, help="この forward で必要な unique expert 数")
    ap.add_argument("--steps", type=int, default=300)
    args = ap.parse_args()
    U = args.U

    exp, packed = make_experts()
    per = [exp[e] for e in range(U)]            # 常駐 slice（hit 状態）
    keys = [f"{p}.{q}" for p in PROJS for q in PARTS]

    # (1) 現行 concat: 9 テンソルを |U| slice から stack
    def concat_only():
        return {k: mx.concatenate([s[k] for s in per], axis=0) for k in keys}
    t_concat = bench(concat_only, args.steps)

    # gather_qmm 本体（concat 済 buffer で）。x=[U_tokens? ] 簡略に T=2,W=8
    T, W = 2, 8
    rng = np.random.default_rng(0)
    inds = mx.array(rng.integers(0, U, size=(T, W)).astype(np.int32))
    x = mx.random.normal((T, 2048))
    mx.eval(x, inds)

    def concat_then_qmm():
        sub = {k: mx.concatenate([s[k] for s in per], axis=0) for k in keys}
        return qmm_from(sub, x, inds)
    t_full = bench(concat_then_qmm, args.steps)

    # (2) arena 天井: 全 256 を pre-stack 済（packed）→ remap で gather_qmm（concat 無し=GPU-routed F）
    def arena_qmm():
        return qmm_from(packed, x, inds)
    t_arena = bench(arena_qmm, args.steps)

    # (3) 9 concat の launch-overhead を見る: 1 テンソルだけ concat
    def concat_one():
        return mx.concatenate([s["gate_proj.weight"] for s in per], axis=0)
    t_one = bench(concat_one, args.steps)

    L = "{:34} {:8.3f} ms"
    print(f"\n[concat] U={U} (常駐hit)  T={T} W={W}  steps={args.steps}")
    print("-" * 48)
    print(L.format("concat only (9 tensors)", t_concat))
    print(L.format("  concat 1 tensor (launch 1本)", t_one))
    print(L.format("concat + gather_qmm", t_full))
    print(L.format("arena + gather_qmm (天井=F相当)", t_arena))
    print("-" * 48)
    print(f"  concat 税 (full - arena)       : {t_full - t_arena:+.3f} ms")
    print(f"  concat が full に占める割合     : {t_concat / t_full * 100:.0f}%")
    print(f"  9 concat の launch 非線形性     : 9x1本={t_one*9:.3f} vs 実9本={t_concat:.3f}")
    print(f"\n  per-layer 税 ×40層              : concat {(t_full-t_arena)*40:.0f} ms/fwd")


if __name__ == "__main__":
    main()
