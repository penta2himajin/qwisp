#!/usr/bin/env python3
"""Qwisp Step 4 PoC 4.1 — streaming MoE forward の出力一致（compute 配線）.

PoC 4.0 は IO/メモリ層（expert を on-demand 読みして bit 一致）を実証した。
本 PoC は compute 層：**「全256 experts 常駐」でなく「必要な experts のサブセットだけ」**を
ディスクから読み、`gather_qmm` で計算した routed-expert 出力が、full の switch_mlp と一致するか。

これが通れば「正しく動く streaming MoE 層」の中核が成立する。

mlx_lm の実装に忠実に再現:
  QuantizedSwitchLinear: mx.gather_qmm(x, w, scales, biases, rhs_indices, transpose=True,
                                       group_size=64, bits=4, mode="affine")
  SwitchGLU: xe=expand_dims(x,(-2,-3)); up,gate→swiglu(gate,up)→down; squeeze(-2)
  token 数 < 64 なら sort 無し（do_sort=indices.size>=64）→ 4 token に絞って sort 回避。

実行:
  PY="$HOME/Library/Application Support/MTPLX/runtime-venv/bin/python"
  "$PY" poc_moe_forward.py --model "$HOME/.mtplx/models/Youssofal--...-FP16"
"""

import argparse
import os
import sys

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.generate import stream_generate
from mlx_lm.models.activations import swiglu
from mlx_lm.models.qwen3_next import Qwen3NextSparseMoeBlock

from poc_expert_stream import PARTS, PROJS, ExpertLoader, read_header

_CAP = {}


def _capture_call(orig):
    def patched(self, x):
        if "block" not in _CAP:
            _CAP["block"] = self
            _CAP["x"] = x
        return orig(self, x)
    return patched


def build_subset(loader, layer, experts):
    """必要 experts のサブセットを stack（[len(U), ...]）で組む。"""
    base = f"language_model.model.layers.{layer}.mlp.switch_mlp"
    sub = {}
    for proj in PROJS:
        for part in PARTS:
            arrs = [loader.load_tensor_expert(f"{base}.{proj}.{part}", e)[0] for e in experts]
            sub[f"{proj}.{part}"] = mx.concatenate(arrs, axis=0)
    return sub


def qmm(x, sub, proj, remap):
    return mx.gather_qmm(
        x, sub[f"{proj}.weight"], sub[f"{proj}.scales"], sub[f"{proj}.biases"],
        rhs_indices=remap, transpose=True, group_size=64, bits=4, mode="affine",
        sorted_indices=False,
    )


def streaming_switch_glu(x, sub, remap):
    """SwitchGLU を忠実再現（サブセット重み＋remap した indices）。"""
    xe = mx.expand_dims(x, (-2, -3))
    x_up = qmm(xe, sub, "up_proj", remap)
    x_gate = qmm(xe, sub, "gate_proj", remap)
    h = swiglu(x_gate, x_up)          # activation(x_up, x_gate) = swiglu(gate, up)
    x_down = qmm(h, sub, "down_proj", remap)
    return x_down.squeeze(-2)


def main():
    ap = argparse.ArgumentParser(description="Qwisp Step4 streaming MoE forward PoC")
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--tokens", type=int, default=4, help="<8 で indices.size<64 → sort 回避")
    ap.add_argument("--shard", default="model-00001-of-00004.safetensors")
    args = ap.parse_args()

    # --- 実 hidden state を捕捉するためモデルを 1 forward 流す ---
    orig = Qwen3NextSparseMoeBlock.__call__
    Qwen3NextSparseMoeBlock.__call__ = _capture_call(orig)
    print(f"[poc] loading model ...", file=sys.stderr)
    model, tok = load(args.model)
    ids = tok.encode("def add(a, b):\n    return a + b\n")[:16]
    for _ in stream_generate(model, tok, prompt=ids, max_tokens=1):
        break
    Qwen3NextSparseMoeBlock.__call__ = orig

    block = _CAP["block"]
    x_full = _CAP["x"]                 # [1, seq, hidden]
    # 4 token に絞って sort 回避（full/stream 両方 no-sort で厳密一致）
    x = x_full[:, :args.tokens, :]
    k = block.top_k

    gates = mx.softmax(block.gate(x), axis=-1, precise=True)
    inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
    mx.eval(x, inds)

    # --- full（全256 experts 常駐）---
    y_full = block.switch_mlp(x, inds)
    mx.eval(y_full)

    # --- streaming（必要サブセットのみディスクから）---
    inds_np = np.array(inds.tolist())
    U = sorted(set(int(v) for v in inds_np.flatten()))
    pos = {e: i for i, e in enumerate(U)}
    remap = mx.array(np.vectorize(pos.get)(inds_np).astype(np.int32))

    shard = os.path.join(args.model, args.shard)
    hdr, data_start = read_header(shard)
    loader = ExpertLoader(shard, hdr, data_start, nocache=False)
    sub = build_subset(loader, args.layer, U)
    loader.close()

    y_stream = streaming_switch_glu(x, sub, remap)
    mx.eval(y_stream)

    # --- 比較 ---
    diff = mx.abs(y_full - y_stream)
    max_abs = float(mx.max(diff).item())
    mean_abs = float(mx.mean(diff).item())
    scale = float(mx.mean(mx.abs(y_full)).item())
    ok = bool(mx.allclose(y_full, y_stream, atol=1e-3, rtol=1e-3).item())

    print(f"\n[poc] tokens={args.tokens} top_k={k} unique_experts={len(U)}/256")
    print(f"[poc] y shape={tuple(y_full.shape)}  |y_full|~{scale:.4f}")
    print(f"[poc] max|diff|={max_abs:.2e}  mean|diff|={mean_abs:.2e}")
    print(f"\n[poc] VERDICT: {'PASS — streaming MoE forward が full と一致' if ok else 'FAIL'}",
          file=sys.stderr)
    print(f"[poc] → 必要な {len(U)} experts だけ常駐で routed 出力を正しく計算（全256 不要）",
          file=sys.stderr)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
