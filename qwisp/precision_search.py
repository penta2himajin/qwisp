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

from .mixed_probe import greedy, calibrate, PROMPT

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
    lm = model.language_model
    base = greedy(model, tok)                       # baseline greedy（参照列）
    # 頑健メトリック用: teacher-forced で参照列 (prompt+base) を1パス、各位置の baseline 分布を保存
    prompt_ids = tok.encode(PROMPT)
    ref_ids = mx.array(prompt_ids + list(base))[None]      # [1, L]
    gen_lo = len(prompt_ids) - 1                           # 生成領域の予測開始位置
    base_logits = lm(ref_ids)[0, gen_lo:-1]                # [G, V]
    base_argmax = mx.argmax(base_logits, axis=-1)
    base_logp = base_logits - mx.logsumexp(base_logits, axis=-1, keepdims=True)
    mx.eval(base_argmax, base_logp)
    tgt = ref_ids[0, gen_lo + 1:]

    counts = calibrate(model, tok)
    # 各 block の (qsls, hot_order)
    blocks = []
    for _, blk in model.named_modules():
        if isinstance(blk, Qwen3NextSparseMoeBlock):
            c = counts.get(id(blk), np.zeros(256, np.int64)).astype(np.float64)
            order = np.argsort(c)[::-1]               # 頻度降順（先頭が hot）
            csort = c[order]                          # 降順カウント
            qsls = [blk.switch_mlp.gate_proj, blk.switch_mlp.up_proj, blk.switch_mlp.down_proj]
            blocks.append((qsls, order, csort))

    # snapshot（元 4bit 重み）を numpy(CPU) に退避＝GPU は 12GB に保つ
    snap = [[(np.array(q.weight), np.array(q.scales), np.array(q.biases)) for q in qsls]
            for qsls, _, _ in blocks]

    def restore():
        for (qsls, _, _), s in zip(blocks, snap):
            for q, (w, sc, b) in zip(qsls, s):
                q.weight, q.scales, q.biases = mx.array(w), mx.array(sc), mx.array(b)
        mx.eval(model.parameters())

    def adaptive_hb(csort, T, lo=16, hi=128):
        """頻度累積マスが T を超える最小 K（=その層の hot 数）。[lo,hi] にクランプ。"""
        cum = np.cumsum(csort) / max(csort.sum(), 1)
        k = int(np.searchsorted(cum, T) + 1)
        return max(lo, min(hi, k))

    def run(label, per_block_hb, hbit, hgs, cbit, cgs):
        restore()
        total_hot = 0
        for (qsls, order, csort), hb in zip(blocks, per_block_hb):
            total_hot += hb
            hot_idx = mx.array(order[:hb].astype(np.int32))
            cold_idx = mx.array(order[hb:].astype(np.int32))
            for q in qsls:
                if hbit < 4:
                    degrade(q, hot_idx, hbit, hgs)
                degrade(q, cold_idx, cbit, cgs)
        mx.eval(model.parameters())
        # 頑健メトリック: teacher-forced（非複利）。top-1 一致率 + 平均 KL(base||quant)
        q_logits = lm(ref_ids)[0, gen_lo:-1]
        q_argmax = mx.argmax(q_logits, axis=-1)
        q_logp = q_logits - mx.logsumexp(q_logits, axis=-1, keepdims=True)
        top1 = float(mx.mean((q_argmax == base_argmax).astype(mx.float32)).item())   # baseline と同じ予測か
        kl = float(mx.mean(mx.sum(mx.exp(base_logp) * (base_logp - q_logp), axis=-1)).item())
        # 参考: 旧 greedy 完全一致（複利・脆い）
        eb = total_hot * expert_bytes(hbit, hgs) + (256 * N_BLOCKS - total_hot) * expert_bytes(cbit, cgs)
        gb = eb / 1e9 + NONEXPERT_GB
        avg = total_hot / N_BLOCKS
        v = "GREEN" if (top1 >= 0.98 and kl < 0.02) else ("ok" if top1 >= 0.95 else "NO")
        print(f"{label:18} avg_hot={avg:5.1f} {gb:>7.2f}G  top1={top1:.3f} KL={kl:.4f}  {v:>6}")

    nb = len(blocks)
    print(f"\n--- 頑健メトリック（teacher-forced top1+KL）で縮小候補を再評価 ---")
    print(f"{'config':18} {'avg_hot':>8} {'GB':>8}  {'metric':>22}  verdict")
    run("baseline h64/c2", [64] * nb, 4, 64, 2, 64)     # 13.47G
    run("h48/c2", [48] * nb, 4, 64, 2, 64)              # 12.97G（旧 greedy NO）
    run("h32/c2", [32] * nb, 4, 64, 2, 64)              # 12.46G
    run("h64/c2 gs128", [64] * nb, 4, 64, 2, 128)       # 12.71G（旧 NO, KL で再判定）
    run("h64@3/c2", [64] * nb, 3, 64, 2, 64)           # 12.37G（hot 3bit）
    run("h48/c2 gs128", [48] * nb, 4, 64, 2, 128)       # 12.2G（複合）
    for T in (0.70, 0.80):
        hbs = [adaptive_hb(cs, T) for (_, _, cs) in blocks]
        run(f"adaptive T={T}", hbs, 4, 64, 2, 64)


if __name__ == "__main__":
    main()
