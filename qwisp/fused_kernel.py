"""融合 gather-matmul Metal カーネル go/no-go プロトタイプ.

streaming の concat は「選んだ expert を小バッファに pre-gather して gather_qmm に渡す」ため。
gather_qmm は source 全体を materialize するので選択読みできない（mmap_probe で確認）。
→ カスタム MSL カーネルで「[E,...] バッファから rhs_indices の expert だけ直読みして
量子化 matmul」＝concat も全materialize も無し。go/no-go: (1)gather_qmm と bit一致 (2)concat 比速度。

2bit affine 確定式（mlx 実測）: dequant(o,i)=q*scale[o,i//gs]+bias[o,i//gs],
q=(W[o,i//16]>>((i%16)*2))&3, LSB-first。transpose=True で y[o]=Σ_i deq(W[e,o,i])*x[i]。

実行: PY -m qwisp.fused_kernel [--bits 2 --T 2 --K 8]
"""
from __future__ import annotations
import argparse
import time

import numpy as np
import mlx.core as mx
from mlx_lm.models.activations import swiglu

_SRC = r"""
    uint gid = thread_position_in_grid.x;
    int T = params[0], K = params[1], in_ = params[2], out = params[3];
    int in_packed = params[4], gs = params[5], gs_count = params[6], bits = params[7];
    int total = T * K * out;
    if ((int)gid >= total) return;
    int o  = gid % out;
    int tk = gid / out;
    int t  = tk / K;
    int e  = inds[tk];                       // expert id（inds は [T*K] flat）
    uint mask = (1u << bits) - 1u;
    int vpw = 32 / bits;                      // values per uint32
    float acc = 0.0f;
    int wbase = (e * out + o) * in_packed;
    int sbase = (e * out + o) * gs_count;
    for (int i = 0; i < in_; i++) {
        uint word = w[wbase + i / vpw];
        uint q = (word >> ((i % vpw) * bits)) & mask;
        int g = i / gs;
        float sc = (float)scales[sbase + g];
        float bi = (float)biases[sbase + g];
        float wv = (float)q * sc + bi;
        acc += wv * x[t * in_ + i];           // x は [T, in]（k 跨ぎで共有）
    }
    y[gid] = acc;
"""

_KERNEL = mx.fast.metal_kernel(
    name="fused_gather_qmm",
    input_names=["x", "inds", "w", "scales", "biases", "params"],
    output_names=["y"],
    source=_SRC,
)

# 最適化版: 1 threadgroup(=1 SIMD, 32 lane) で出力1要素を協調計算。
# lane が連続 word を読む(coalesced) → simd_sum で縮約。naive の非coalesced を解消。
_SRC_OPT = r"""
    uint tg  = threadgroup_position_in_grid.x;    // 出力要素 (tk,o)
    uint lane = thread_position_in_threadgroup.x; // 0..31
    int T = params[0], K = params[1], in_ = params[2], out = params[3];
    int in_packed = params[4], gs = params[5], gs_count = params[6], bits = params[7];
    int total = T * K * out;
    if ((int)tg >= total) return;
    int o  = tg % out;
    int tk = tg / out;
    int t  = tk / K;
    int e  = inds[tk];
    uint mask = (1u << bits) - 1u;
    int vpw = 32 / bits;
    int wbase = (e * out + o) * in_packed;
    int sbase = (e * out + o) * gs_count;
    float acc = 0.0f;
    for (int wi = (int)lane; wi < in_packed; wi += 32) {
        uint word = w[wbase + wi];               // 連続 lane→連続 word = coalesced
        int i0 = wi * vpw;
        int g = i0 / gs;                          // gs>=64>vpw でword内一定→hoist
        float sc = (float)scales[sbase + g];
        float bi = (float)biases[sbase + g];
        for (int bb = 0; bb < vpw; bb++) {
            uint q = (word >> (bb * bits)) & mask;
            acc += ((float)q * sc + bi) * x[t * in_ + i0 + bb];
        }
    }
    acc = simd_sum(acc);                          // 32 lane 縮約
    if (lane == 0) y[tg] = acc;
"""

_KERNEL_OPT = mx.fast.metal_kernel(
    name="fused_gather_qmm_opt",
    input_names=["x", "inds", "w", "scales", "biases", "params"],
    output_names=["y"],
    source=_SRC_OPT,
)


def fused_gather_qmm(x, w, scales, biases, inds, bits, group_size=64):
    """x:[T,in] f32, w:[E,out,in_packed]u32, scales/biases:[E,out,gs_count]f16, inds:[T,K]i32.
    返り [T,K,out] f32。gather_qmm(transpose=True, affine) と等価。"""
    T, in_ = x.shape
    E, out, in_packed = w.shape
    K = inds.shape[1]
    gs_count = scales.shape[2]
    params = mx.array([T, K, in_, out, in_packed, group_size, gs_count, bits], dtype=mx.int32)
    total = T * K * out
    (y,) = _KERNEL(
        inputs=[x.astype(mx.float32), inds.reshape(-1).astype(mx.int32),
                w, scales.astype(mx.float16), biases.astype(mx.float16), params],
        output_shapes=[(total,)],
        output_dtypes=[mx.float32],
        grid=(total, 1, 1),
        threadgroup=(min(256, total), 1, 1),
    )
    return y.reshape(T, K, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, default=2)
    ap.add_argument("--T", type=int, default=2)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--E", type=int, default=64, help="バッファ内 expert 数（cold 常駐想定）")
    ap.add_argument("--steps", type=int, default=200)
    args = ap.parse_args()
    T, K, E, bits, gs = args.T, args.K, args.E, args.bits, 64
    IN, OUT = 2048, 512

    fp = mx.random.normal((E, OUT, IN)) * 0.05
    w, s, b = mx.quantize(fp, group_size=gs, bits=bits)
    x = mx.random.normal((T, IN))
    rng = np.random.default_rng(0)
    inds = mx.array(rng.integers(0, E, size=(T, K)).astype(np.int32))
    mx.eval(w, s, b, x, inds)

    # 参照: gather_qmm（transpose=True）
    xe = mx.expand_dims(x, (-2, -3))                    # [T,1,1,IN]
    ref = mx.gather_qmm(xe, w, s, b, rhs_indices=inds, transpose=True,
                        group_size=gs, bits=bits, mode="affine", sorted_indices=False)
    ref = ref.reshape(T, K, OUT)
    got = fused_gather_qmm(x, w, s, b, inds, bits, gs)
    mx.eval(ref, got)
    diff = float(mx.max(mx.abs(ref - got)).item())
    rel = diff / (float(mx.max(mx.abs(ref)).item()) + 1e-9)
    print(f"\n[fused] bits={bits} T={T} K={K} E={E}  IN={IN} OUT={OUT}")
    print(f"  bit一致: max|Δ|={diff:.3e} rel={rel:.3e}  {'OK' if rel < 1e-2 else 'MISMATCH'}")

    def bench(fn, steps=args.steps, warm=20):
        for _ in range(warm):
            mx.eval(fn())
        t = time.perf_counter()
        for _ in range(steps):
            mx.eval(fn())
        return (time.perf_counter() - t) / steps * 1e3

    # bare kernel（変換を loop 外に出す＝公平比較）
    xf = x.astype(mx.float32); indf = inds.reshape(-1).astype(mx.int32)
    sf = s.astype(mx.float16); bf = b.astype(mx.float16)
    total = T * K * OUT

    params = mx.array([T, K, IN, OUT, w.shape[2], gs, s.shape[2], bits], dtype=mx.int32)

    def bare(tg):
        (y,) = _KERNEL(inputs=[xf, indf, w, sf, bf, params], output_shapes=[(total,)],
                       output_dtypes=[mx.float32], grid=(total, 1, 1), threadgroup=(tg, 1, 1))
        return y

    def opt():
        (y,) = _KERNEL_OPT(inputs=[xf, indf, w, sf, bf, params], output_shapes=[(total,)],
                           output_dtypes=[mx.float32], grid=(total * 32, 1, 1), threadgroup=(32, 1, 1))
        return y

    # opt の bit 一致も確認
    yo = opt().reshape(T, K, OUT); mx.eval(yo)
    rel_o = float(mx.max(mx.abs(yo - ref)).item()) / (float(mx.max(mx.abs(ref)).item()) + 1e-9)
    print(f"  opt bit一致: rel={rel_o:.3e}  {'OK' if rel_o < 1e-2 else 'MISMATCH'}")

    t_gqmm = bench(lambda: mx.gather_qmm(xe, w, s, b, rhs_indices=inds, transpose=True,
                                         group_size=gs, bits=bits, mode="affine", sorted_indices=False))
    t_naive = bench(lambda: bare(64))
    t_opt = bench(opt)
    print(f"\n  速度: gather_qmm(全E常駐) = {t_gqmm:.3f}ms")
    print(f"        fused naive          = {t_naive:.3f}ms  ({t_naive/t_gqmm:.1f}x)")
    print(f"        fused opt(SIMD)      = {t_opt:.3f}ms  ({t_opt/t_gqmm:.1f}x)")
    print(f"  ※ streaming 比較対象 concat(~0.4)+gather_qmm(~0.45)=~0.85ms。fused opt<0.85 なら勝ち")


if __name__ == "__main__":
    main()
