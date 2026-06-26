"""精度設計の探索 — 全常駐(GPU-routing)を 12GB Mac に収める最小サイズの GREEN config を探す.

制約: GPU-routing は全 256 expert 常駐前提＝expert 数は減らせない。**平均 bit を下げる**しかない。
config (hot_b, hot_bits, cold_bits, cold_gs) を振り、品質（greedy 一致）と expert サイズ（解析式）を測る。

サイズ式（per-expert, bits=b, group_size=g; gate/up=512x2048, down=2048x512）:
  weight = 393216*b バイト,  scales+biases = 12582912/g バイト
  例: 4bit/g64=1.77MB, 3bit/g64=1.34MB, 2bit/g64=0.98MB, 2bit/g128=0.88MB, 2bit/g256=0.84MB

品質は mixed_probe の roundtrip（量子化ノイズを subset に注入し 4bit ストレージに戻す）で測る。
モデルは1回ロードし switch_mlp 重みを snapshot → config 毎に restore して評価。

実行: PY -m qwisp.precision_search <4bit_model>
"""
from __future__ import annotations
import sys

import numpy as np
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.qwen3_next import Qwen3NextSparseMoeBlock

from .mixed_probe import greedy, calibrate

NONEXPERT_GB = 1.39
NLAYER = 256  # experts per block
N_BLOCKS = 40


def expert_bytes(bits, gs):
    return 393216 * bits + 12582912 // gs


def total_gb(hot_b, hot_bits, hot_gs, cold_bits, cold_gs):
    per_layer = hot_b * expert_bytes(hot_bits, hot_gs) + (256 - hot_b) * expert_bytes(cold_bits, cold_gs)
    return per_layer * N_BLOCKS / 1e9 + NONEXPERT_GB


def degrade(qsl, idx, tbits, tgs):
    """idx の expert に (tbits,tgs) の量子化ノイズを注入し、元の 4bit/gs ストレージに戻す。"""
    gs, bits, mode = qsl.group_size, qsl.bits, getattr(qsl, "mode", "affine")
    fp = mx.dequantize(qsl.weight, qsl.scales, qsl.biases, group_size=gs, bits=bits, mode=mode)
    w2, s2, b2 = mx.quantize(fp[idx], group_size=tgs, bits=tbits, mode=mode)
    fp[idx] = mx.dequantize(w2, s2, b2, group_size=tgs, bits=tbits, mode=mode)
    nw, ns, nb = mx.quantize(fp, group_size=gs, bits=bits, mode=mode)
    qsl.weight, qsl.scales, qsl.biases = nw, ns, nb


def main():
    model_dir = sys.argv[1]
    print(f"[psearch] loading {model_dir} ...", file=sys.stderr)
    model, tok = load(model_dir)
    base = greedy(model, tok)

    counts = calibrate(model, tok)
    # 各 block の (qsls, hot_order)
    blocks = []
    for _, blk in model.named_modules():
        if isinstance(blk, Qwen3NextSparseMoeBlock):
            c = counts.get(id(blk), np.zeros(256, np.int64))
            order = np.argsort(c)[::-1]               # 頻度降順（先頭が hot）
            qsls = [blk.switch_mlp.gate_proj, blk.switch_mlp.up_proj, blk.switch_mlp.down_proj]
            blocks.append((qsls, order))

    # snapshot（元 4bit 重み）を numpy(CPU) に退避＝GPU は 12GB に保つ
    snap = [[(np.array(q.weight), np.array(q.scales), np.array(q.biases)) for q in qsls]
            for qsls, _ in blocks]

    def restore():
        for (qsls, _), s in zip(blocks, snap):
            for q, (w, sc, b) in zip(qsls, s):
                q.weight, q.scales, q.biases = mx.array(w), mx.array(sc), mx.array(b)
        mx.eval(model.parameters())

    # 探索する config: (hot_b, hot_bits, hot_gs, cold_bits, cold_gs)
    configs = [
        (64, 4, 64, 2, 64),     # baseline GREEN（13.47GB）
        (64, 3, 64, 2, 64),     # hot 3bit（12.37GB）
        (56, 4, 64, 2, 64),
        (48, 4, 64, 2, 64),     # hot48（12.97GB）
        (40, 4, 64, 2, 64),
        (64, 3, 32, 2, 64),     # hot 3bit/gs32（品質寄り, ほぼ同サイズ）
        (96, 3, 64, 2, 64),     # hot 多め@3bit
        (128, 3, 64, 2, 64),    # hot128@3（品質寄り・cold 少）
    ]

    print(f"\n{'hot_b':>5} {'hotbit':>6} {'coldbit':>7} {'cold_gs':>7} "
          f"{'GB(experts+ne)':>14} {'match':>7} {'verdict':>7}")
    print("-" * 60)
    for hb, hbit, hgs, cbit, cgs in configs:
        restore()
        for (qsls, order) in blocks:
            hot_idx = mx.array(order[:hb].astype(np.int32))
            cold_idx = mx.array(order[hb:].astype(np.int32))
            for q in qsls:
                if hbit < 4:
                    degrade(q, hot_idx, hbit, hgs)
                degrade(q, cold_idx, cbit, cgs)
        mx.eval(model.parameters())
        deg = greedy(model, tok)
        match = sum(1 for a, b in zip(base, deg) if a == b)
        gb = total_gb(hb, hbit, hgs, cbit, cgs)
        verdict = "GREEN" if match >= len(base) - 1 else ("ok" if match >= len(base) * 0.9 else "NO")
        print(f"{hb:>5} {hbit:>6} {cbit:>7} {cgs:>7} {gb:>13.2f}G {match:>4}/{len(base)} {verdict:>7}")


if __name__ == "__main__":
    main()
