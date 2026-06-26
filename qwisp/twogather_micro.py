"""two-gather +5ms の内訳分解 — モデル無し・実形状の合成バッファで切り分ける.

GPURoutedMixedSwitchGLU の two-gather は各 (token,expert) slot を hot(4bit)/cold(2bit)
両経路で計算し where で半分捨てる＝gather_qmm launch が native の 2 倍。
docs/09 の F-A=+5ms がこの構造のどこ（冗長 compute / 余分な LUT gather / where / launch）に
あるかを、12GB ロード無しで測る。これで (ii-a) 融合カーネルの回収上限と必要形が決まる。

実形状: hidden=2048, moe_inter=512, top_k=8, E=256(hot64/cold192), group_size=64。
2-token verify forward を模擬（T=2, W=8）。

実行: PY -m qwisp.twogather_micro [--T 2 --steps 200]
"""
from __future__ import annotations
import argparse
import time

import numpy as np
import mlx.core as mx
from mlx_lm.models.activations import swiglu

H = 2048      # hidden
I = 512       # moe intermediate
E = 256
HOT = 64
GS = 64


def make_buf(experts, bits):
    """experts 個分の gate/up[I,H], down[H,I] を bits で量子化した contiguous バッファ。"""
    buf = {}
    shapes = {"gate_proj": (experts, I, H), "up_proj": (experts, I, H), "down_proj": (experts, H, I)}
    for proj, shp in shapes.items():
        fp = mx.random.normal(shp) * 0.02
        w, s, b = mx.quantize(fp, group_size=GS, bits=bits)
        buf[f"{proj}.weight"] = mx.contiguous(w)
        buf[f"{proj}.scales"] = mx.contiguous(s)
        buf[f"{proj}.biases"] = mx.contiguous(b)
    mx.eval(list(buf.values()))
    return buf


def qmm(x, buf, remap, bits):
    xe = mx.expand_dims(x, (-2, -3))

    def q(xx, proj):
        return mx.gather_qmm(xx, buf[f"{proj}.weight"], buf[f"{proj}.scales"],
                             buf[f"{proj}.biases"], rhs_indices=remap, transpose=True,
                             group_size=GS, bits=bits, mode="affine", sorted_indices=False)
    h = swiglu(q(xe, "gate_proj"), q(xe, "up_proj"))
    return q(h, "down_proj").squeeze(-2)


def bench(fn, steps, warm=20):
    for _ in range(warm):
        mx.eval(fn())
    t = time.perf_counter()
    for _ in range(steps):
        mx.eval(fn())
    return (time.perf_counter() - t) / steps * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=2)
    ap.add_argument("--steps", type=int, default=300)
    args = ap.parse_args()
    T, W = args.T, 8

    hotbuf = make_buf(HOT, 4)
    coldbuf = make_buf(E - HOT, 2)
    allbuf4 = make_buf(E, 4)        # native 参照: 全 256 を 4bit 単一バッファ

    # LUT（hot=先頭64, cold=残り192）
    lut_hot = np.zeros(E, np.int32); lut_cold = np.zeros(E, np.int32); ishot = np.zeros(E, bool)
    for s, e in enumerate(range(HOT)):
        lut_hot[e] = s; ishot[e] = True
    for s, e in enumerate(range(HOT, E)):
        lut_cold[e] = s
    lut_hot = mx.array(lut_hot); lut_cold = mx.array(lut_cold); is_hot = mx.array(ishot)

    rng = np.random.default_rng(0)
    inds = mx.array(rng.integers(0, E, size=(T, W)).astype(np.int32))
    x = mx.random.normal((T, H))      # SwitchGLU: x=[T,hidden]; expand+index broadcast → per-expert
    mx.eval(x, inds)

    L = "{:30} {:8.3f} ms"
    print(f"\n[micro] two-gather 分解  T={T} W={W} E={E} hot={HOT} gs={GS} steps={args.steps}")
    print("-" * 46)

    # 1) native 単一 gather（全 256 を 4bit, remap=inds そのまま）= A 床の MoE 部分
    t_native = bench(lambda: qmm(x, allbuf4, inds, 4), args.steps)
    print(L.format("A native single-gather(4bit)", t_native))

    # 2) 現行 two-gather + where（GPURouted と同一）
    def two_gather_where():
        rh = lut_hot[inds]; rc = lut_cold[inds]
        yh = qmm(x, hotbuf, rh, 4); yc = qmm(x, coldbuf, rc, 2)
        return mx.where(is_hot[inds][..., None], yh, yc)
    t_full = bench(two_gather_where, args.steps)
    print(L.format("F two-gather + where(現行)", t_full))

    # 3) two-gather のみ（where 無し, yh+yc を足すだけ → where コスト分離）
    def two_gather_add():
        rh = lut_hot[inds]; rc = lut_cold[inds]
        return qmm(x, hotbuf, rh, 4) + qmm(x, coldbuf, rc, 2)
    t_add = bench(two_gather_add, args.steps)
    print(L.format("  two-gather + add(no where)", t_add))

    # 4) hot 経路だけ（片側 gather のコスト＝launch+compute の 1 本分）
    def hot_only():
        return qmm(x, hotbuf, lut_hot[inds], 4)
    t_hot = bench(hot_only, args.steps)
    print(L.format("  hot path only(1 gather)", t_hot))

    # 5) LUT gather 2 本 + where のみ（matmul 抜き＝ルーティング諸経費）
    def routing_only():
        rh = lut_hot[inds]; rc = lut_cold[inds]
        m = is_hot[inds][..., None]
        return rh.sum() + rc.sum() + m.sum()
    t_route = bench(routing_only, args.steps)
    print(L.format("  routing LUT+mask only", t_route))

    print("-" * 46)
    print(f"  two-gather 税 (F - A native)      : {t_full - t_native:+.3f} ms")
    print(f"  where のコスト (F - add)          : {t_full - t_add:+.3f} ms")
    print(f"  冗長 2nd gather (add - hot_only)  : {t_add - t_hot:+.3f} ms")
    print(f"  routing 諸経費 (LUT+mask)         : {t_route:+.3f} ms")
    print("\n  解釈: 融合カーネルが回収しうるのは [where + 冗長gather + launch]。")
    print("        routing 諸経費が支配的なら融合の旨味は薄い。")

    # --- prefill partition 経路の正しさ + 速度（実クラス GPURoutedMixedSwitchGLU）---
    from .gpu_routed import GPURoutedMixedSwitchGLU as G
    obj = G.__new__(G)
    obj._layer = 0
    obj._probe_sync = False
    obj._hotbuf = hotbuf; obj._coldbuf = coldbuf
    obj._lut_hot = lut_hot; obj._lut_cold = lut_cold; obj._is_hot = is_hot
    ih_np = ishot; lh_np = np.asarray(lut_hot); lc_np = np.asarray(lut_cold)
    obj._is_hot_np = ih_np; obj._lut_hot_np = lh_np; obj._lut_cold_np = lc_np

    # 正しさ: partition 出力 == full two-gather+where（同じ matmul, 片側だけ計算して戻すだけ）
    y_part = obj._prefill_partition(x, inds); mx.eval(y_part)
    y_ref = two_gather_where(); mx.eval(y_ref)
    diff = float(mx.max(mx.abs(y_part - y_ref)).item())
    rel = diff / (float(mx.max(mx.abs(y_ref)).item()) + 1e-9)
    print(f"\n[micro] prefill partition 検証  out shape={tuple(y_part.shape)}")
    print(f"  max|partition - two_gather_where| = {diff:.2e}  (rel {rel:.2e})  "
          f"{'OK' if rel < 1e-4 else 'MISMATCH'}")

    # 速度: partition vs 現行 full two-gather（同 T）
    t_part = bench(lambda: obj._prefill_partition(x, inds), max(args.steps // 4, 20))
    print(f"  partition forward = {t_part:.3f} ms  /  full two-gather+where = {t_full:.3f} ms"
          f"  ({t_full / t_part:.2f}x 速い)" if t_part < t_full else
          f"  partition forward = {t_part:.3f} ms  /  full two-gather+where = {t_full:.3f} ms")


if __name__ == "__main__":
    main()
