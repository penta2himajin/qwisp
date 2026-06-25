"""事前処理 — 全 expert の 2bit 版を disk に作成（mixed-precision の「両精度保管」）.

4bit の switch_mlp 重みを dequant→quant(2bit)→requant し、別 safetensors に保存。
streaming 時に cold expert をこの 2bit ファイルから読む（IO 半減）。ストレージは安い
（オフライン・on-time でない）ので全 expert を 2bit でも持つ＝動的精度（DynaExq/HOBBIT）。

出力: <out_dir>/experts_2bit.safetensors ＋ model.safetensors.index.json（ExpertSource 互換）。

実行: PY -m qwisp.build_2bit_experts <model_dir> <out_dir>
"""
from __future__ import annotations
import glob
import json
import os
import sys

import mlx.core as mx

GS, SRC_BITS, DST_BITS = 64, 4, 2


def main():
    model_dir, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)

    weights = {}
    for wf in glob.glob(os.path.join(model_dir, "model*.safetensors")):
        weights.update(mx.load(wf))  # lazy mmap

    out = {}
    bases = sorted({k[:-7] for k in weights if ".switch_mlp." in k and k.endswith(".weight")})
    print(f"[2bit] {len(bases)} switch-linear tensors to requant ...", file=sys.stderr)
    for i, base in enumerate(bases):
        w, s, b = weights[base + ".weight"], weights[base + ".scales"], weights[base + ".biases"]
        fp = mx.dequantize(w, s, b, group_size=GS, bits=SRC_BITS, mode="affine")
        w2, s2, b2 = mx.quantize(fp, group_size=GS, bits=DST_BITS, mode="affine")
        out[base + ".weight"], out[base + ".scales"], out[base + ".biases"] = w2, s2, b2
        mx.eval(w2, s2, b2)
        del fp
        if (i + 1) % 30 == 0:
            print(f"[2bit]   {i+1}/{len(bases)}", file=sys.stderr, flush=True)

    fname = "experts_2bit.safetensors"
    mx.save_safetensors(os.path.join(out_dir, fname), out)
    # ExpertSource 互換の index.json（全 key → 単一ファイル）
    with open(os.path.join(out_dir, "model.safetensors.index.json"), "w") as f:
        json.dump({"weight_map": {k: fname for k in out}}, f)
    sz = os.path.getsize(os.path.join(out_dir, fname)) / 1e9
    print(f"[2bit] wrote {len(out)} tensors -> {out_dir}/{fname} ({sz:.1f} GB)", file=sys.stderr)


if __name__ == "__main__":
    main()
