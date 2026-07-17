#!/usr/bin/env python3
"""Regenerate the 2-bit expert-tail artifact (notes/18 workstream ③).

Reads the 4-bit MTPLX checkpoint's switch_mlp tensors, requantizes ALL
40 layers x 256 experts x {gate,up,down}_proj to affine 2-bit gs=64, and
writes a single experts_2bit.safetensors + model.safetensors.index.json
(same tensor names/shapes/dtypes as the shipped naive artifact, so
ExpertSource serves it unmodified).

Modes:
  naive — mx.quantize(mx.dequantize(4bit), bits=2): regenerates the original
          ~/.mtplx/models/qwisp-experts-2bit artifact (provenance-verified
          bit-exact against it, notes/18).
  cal   — MSE-optimal affine fit (scripts/loopy/p17/mixprec.py#cal2bit, the
          sweep-validated pipeline: 0/4 loops at K4=8 and K4=0). The fitted
          (q, s, b) are packed DIRECTLY — re-encoding the dequantized grid
          through mx.quantize would min/max-refit and distort any group not
          spanning all 4 codes.

Run (GPU-exclusive — `brew services stop qwisp` first):
  ~/.venvs/mlx/bin/python3 oracle/requant_experts_2bit.py ~/.mtplx/models/qwisp-experts-2bit-cal cal
"""
import json
import os
import sys

import mlx.core as mx
import numpy as np

GS = 64
MDL = os.path.expanduser("~/.mtplx/models/Youssofal--Qwen3.6-35B-A3B-MTPLX-Optimized-Speed-FP16")
OUT = os.path.expanduser(sys.argv[1])
MODE = sys.argv[2] if len(sys.argv) > 2 else "cal"
assert MODE in ("naive", "cal"), MODE
PROJS = ("gate_proj", "up_proj", "down_proj")


def cal2bit_codes(g):
    """mixprec.py#cal2bit, returning the fitted (q, s, b) instead of the
    dequantized grid. g: [G, 64] f32 groups."""
    mn = g.min(axis=1, keepdims=True)
    mxv = g.max(axis=1, keepdims=True)
    s = (mxv - mn) / 3
    s = mx.where(mx.abs(s) < 1e-8, mx.ones_like(s), s)
    b = mn
    for _ in range(15):
        q = mx.clip(mx.round((g - b) / s), 0, 3)
        qm = q.mean(axis=1, keepdims=True)
        gm = g.mean(axis=1, keepdims=True)
        var = ((q - qm) ** 2).mean(axis=1, keepdims=True)
        cov = ((q - qm) * (g - gm)).mean(axis=1, keepdims=True)
        s = mx.where(var > 1e-12, cov / mx.maximum(var, mx.array(1e-12)), s)
        s = mx.where(mx.abs(s) < 1e-8, mx.ones_like(s), s)
        b = gm - s * qm
    q = mx.clip(mx.round((g - b) / s), 0, 3)
    return q, s, b


def pack2bit(q_np):
    """[..., N] uint codes -> [..., N/16] u32, element i in bits 2i..2i+1
    (MLX quantized.h layout; verified by the self-test below)."""
    r = q_np.astype(np.uint32).reshape(*q_np.shape[:-1], -1, 16)
    return (r << (2 * np.arange(16, dtype=np.uint32))).sum(axis=-1, dtype=np.uint32)


# layout self-test: pack known codes, dequantize with s=1 b=0 -> codes back
_qt = np.random.default_rng(0).integers(0, 4, size=(2, GS))
_w = mx.array(pack2bit(_qt))
_deq = mx.dequantize(_w, mx.ones((2, 1), dtype=mx.float16),
                     mx.zeros((2, 1), dtype=mx.float16), group_size=GS, bits=2)
assert np.array_equal(np.array(_deq, dtype=np.int64), _qt), "pack2bit layout mismatch vs MLX"

idx = json.load(open(f"{MDL}/model.safetensors.index.json"))["weight_map"]
shards = {f: mx.load(f"{MDL}/{f}") for f in sorted(set(idx.values()))}
moe_keys = sorted({k.rsplit(".", 1)[0] for k in idx if ".switch_mlp." in k})
layers = sorted({int(k.split(".layers.")[1].split(".")[0]) for k in moe_keys})
print(f"[requant] {len(layers)} MoE layers, mode={MODE}", flush=True)

tensors = {}
for li in layers:
    for pname in PROJS:
        base = f"language_model.model.layers.{li}.mlp.switch_mlp.{pname}"
        w4, s4, b4 = (shards[idx[f"{base}.{t}"]][f"{base}.{t}"]
                      for t in ("weight", "scales", "biases"))
        deq = mx.dequantize(w4, s4, b4, group_size=GS, bits=4)  # [256, R, IN] f16
        shp = deq.shape
        if MODE == "naive":
            w2, s2, b2 = mx.quantize(deq.reshape(-1, shp[-1]), group_size=GS, bits=2)
            w2 = w2.reshape(shp[0], shp[1], -1)
            s2 = s2.reshape(shp[0], shp[1], -1).astype(mx.float16)
            b2 = b2.reshape(shp[0], shp[1], -1).astype(mx.float16)
            mx.eval(w2, s2, b2)
        else:
            g = deq.reshape(-1, GS).astype(mx.float32)
            q, s, b = cal2bit_codes(g)
            mx.eval(q, s, b)
            w2 = mx.array(pack2bit(np.array(q, dtype=np.uint8).reshape(shp[0], shp[1], -1)))
            s2 = s.reshape(shp[0], shp[1], -1).astype(mx.float16)
            b2 = b.reshape(shp[0], shp[1], -1).astype(mx.float16)
            mx.eval(w2, s2, b2)
            # per-tensor sanity: stored dequant reproduces the fitted grid
            chk = mx.dequantize(w2[0], s2[0], b2[0], group_size=GS, bits=2).astype(mx.float32)
            ref = (s * q + b).reshape(shp)[0].astype(mx.float32)
            err = mx.abs(chk - ref).max().item()
            tol = 2e-3 * mx.abs(s).max().item()  # f16 rounding of s,b only
            assert err <= tol, f"{base}: dequant mismatch {err} > {tol}"
        tensors[f"{base}.weight"] = w2
        tensors[f"{base}.scales"] = s2
        tensors[f"{base}.biases"] = b2
    print(f"[requant] layer {li} done", flush=True)

os.makedirs(OUT, exist_ok=True)
mx.save_safetensors(f"{OUT}/experts_2bit.safetensors", tensors)
json.dump({"weight_map": {k: "experts_2bit.safetensors" for k in tensors}},
          open(f"{OUT}/model.safetensors.index.json", "w"))
print(f"[requant] wrote {len(tensors)} tensors -> {OUT}", flush=True)
