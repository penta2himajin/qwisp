"""mixed-precision のミニマル品質検証（two-gather 不要）.

真の混合精度の主リスク＝品質。同一モデル・同一量子化方式で experts を低bitに
roundtrip（dequant4→quant(b)→dequant→requant4）し、格納は4bitのまま精度だけ落として
既存 gather で動かし、greedy 出力が4bit baseline とどれだけ一致するかを測る。

- cold_idx=None: 全 experts を低bit化（最悪ケース）。これが保てば mixed は当然OK。
- cold_idx=頻度下位: hot 高bit/cold 低bit の真の mixed。

実行: PY -m qwisp.mixed_probe "$MODEL" --cold-bits 2
"""
from __future__ import annotations
import argparse
import sys

import numpy as np
import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import stream_generate
from mlx_lm.sample_utils import make_sampler
from mlx_lm.models.switch_layers import QuantizedSwitchLinear
from mlx_lm.models.qwen3_next import Qwen3NextSparseMoeBlock

CALIB = [
    "def quicksort(arr):\n    if len(arr)<=1: return arr\n    p=arr[len(arr)//2]\n",
    "The quick brown fox jumps over the lazy dog. In computing, ",
    "import numpy as np\ndef relu(x): return np.maximum(0,x)\n",
]

PROMPT = "def fibonacci(n):\n    \"\"\"Return the nth Fibonacci number.\"\"\"\n"
GEN = 40


def greedy(model, tok):
    s = make_sampler(temp=0.0)
    return [r.token for r in stream_generate(model, tok, prompt=tok.encode(PROMPT),
                                             max_tokens=GEN, sampler=s)]


def roundtrip(qsl, cold_bits, cold_idx=None):
    gs, bits, mode = qsl.group_size, qsl.bits, getattr(qsl, "mode", "affine")
    fp = mx.dequantize(qsl.weight, qsl.scales, qsl.biases, group_size=gs, bits=bits, mode=mode)
    src = fp if cold_idx is None else fp[cold_idx]
    w2, s2, b2 = mx.quantize(src, gs, cold_bits, mode=mode)
    deg = mx.dequantize(w2, s2, b2, group_size=gs, bits=cold_bits, mode=mode)
    if cold_idx is None:
        fp = deg
    else:
        fp[cold_idx] = deg
    nw, ns, nb = mx.quantize(fp, gs, bits, mode=mode)
    qsl.weight, qsl.scales, qsl.biases = nw, ns, nb


def calibrate(model, tok):
    """各 MoE block の expert 使用頻度を集計（hot 判定用）。id(block)->counts[256]。"""
    counts = {}
    orig = Qwen3NextSparseMoeBlock.__call__

    def patched(self, x):
        g = mx.softmax(self.gate(x), axis=-1, precise=True)
        inds = mx.argpartition(g, kth=8, axis=-1)[..., -8:]
        mx.eval(inds)
        c = counts.setdefault(id(self), np.zeros(256, np.int64))
        np.add.at(c, np.array(inds).reshape(-1), 1)
        return orig(self, x)

    Qwen3NextSparseMoeBlock.__call__ = patched
    s = make_sampler(temp=0.0)
    for p in CALIB:
        for _ in stream_generate(model, tok, prompt=tok.encode(p * 10)[:512], max_tokens=1, sampler=s):
            break
    Qwen3NextSparseMoeBlock.__call__ = orig
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--cold-bits", type=int, default=2)
    ap.add_argument("--hot-b", type=int, default=0, help="0=全 expert 低bit(最悪) / >0=hot を4bit維持し残りを低bit(mixed)")
    args = ap.parse_args()
    print(f"[mix] loading {args.model} ...", file=sys.stderr)
    model, tok = load(args.model)
    base = greedy(model, tok)

    if args.hot_b <= 0:
        qsls = [m for _, m in model.named_modules() if isinstance(m, QuantizedSwitchLinear)]
        print(f"[mix] roundtrip {len(qsls)} switch-linears to {args.cold_bits}bit (ALL experts) ...",
              file=sys.stderr)
        for q in qsls:
            roundtrip(q, args.cold_bits, cold_idx=None)
        label = f"all-experts {args.cold_bits}bit"
    else:
        counts = calibrate(model, tok)
        nblk = 0
        for _, blk in model.named_modules():
            if not isinstance(blk, Qwen3NextSparseMoeBlock):
                continue
            c = counts.get(id(blk), np.zeros(256, np.int64))
            cold = np.argsort(c)[:max(0, 256 - args.hot_b)]  # 頻度下位＝cold
            cold_idx = mx.array(cold.astype(np.int32))
            for proj in (blk.switch_mlp.gate_proj, blk.switch_mlp.up_proj, blk.switch_mlp.down_proj):
                roundtrip(proj, args.cold_bits, cold_idx=cold_idx)
            nblk += 1
        print(f"[mix] mixed: {nblk} blocks, hot={args.hot_b}@4bit, cold={256-args.hot_b}@{args.cold_bits}bit",
              file=sys.stderr)
        label = f"mixed hot{args.hot_b}@4 / cold@{args.cold_bits}"

    mx.eval(model.parameters())
    deg = greedy(model, tok)
    match = sum(1 for a, b in zip(base, deg) if a == b)
    run = 0
    for a, b in zip(base, deg):
        if a == b:
            run += 1
        else:
            break
    print(f"\n[mix] {label} vs 4bit baseline:")
    print(f"  token match = {match}/{GEN}  (先頭連続一致 {run})")
    print(f"  base[:10]={base[:10]}")
    print(f"  deg [:10]={deg[:10]}")


if __name__ == "__main__":
    main()
