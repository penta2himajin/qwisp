"""Stage A — 実 MTP ヘッド（mtp.safetensors sidecar）の実装と実受理率の測定.

mlx_lm は qwen3_5/qwen3_next ロード時に mtp.* を破棄するので、sidecar を自前でロードする。
MTP = EAGLE/DeepSeek 流の 1 段ドラフト: main の hidden h_i と次トークン emb を融合(fc) →
Qwen3.6 デコーダ層 1 個(self_attn full + 256-expert MoE) → mtp.norm → main lm_head 流用 →
次々トークンを予測。受理率は mtplx_runtime.json 実測 0.886（D1）。

量子化: experts のみ 4bit **gs=32**（実形状由来）、attention/gate/shared_expert/fc/norm は F16。
norm 規約: MTP 同梱 ckpt は norm 重みが「-1 シフト」格納（sanitize の should_shift）→ ロードで +1。

マイルストーン1（このファイル）: teacher-forced 並列で実受理率を測り、ヘッドの正しさと
経験的未知（concat 順 / hidden pre|post-final-norm）を受理率最大化で確定する。KV ロールバック不要。

実行: PY -m qwisp.mtp_head <model_dir> [--ctx 256 --gen 128]
"""
from __future__ import annotations
import argparse
import json
import struct
import sys

import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from mlx_lm.models.activations import swiglu
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.qwen3_next import Qwen3NextAttention

MTP_GS, MTP_BITS = 32, 4
NORM_SUFFIXES = ("norm",)  # 全 *norm* 重みに +1（Gemma 流シフト）


def _read_safetensors(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        data_start = 8 + n
        out = {}
        for k, t in hdr.items():
            if k == "__metadata__":
                continue
            b, e = t["data_offsets"]
            f.seek(data_start + b)
            raw = f.read(e - b)
            dt = {"F16": np.float16, "U32": np.uint32, "F32": np.float32,
                  "BF16": np.uint16}[t["dtype"]]
            arr = np.frombuffer(raw, dt).reshape(t["shape"])
            out[k] = mx.array(arr) if t["dtype"] != "BF16" else mx.array(arr).view(mx.bfloat16)
    return out


def load_mtp_weights(model_dir, shift_extra=False):
    """mtp.safetensors を読み、norm に +1、experts を [256,...] に stack して返す。

    shift_extra: pre_fc_norm_* と mtp.norm にも +1 するか（sanitize の suffix に無いので既定 False）。
    in/post_layernorm・q/k_norm は documented suffix にマッチ → 常に +1。
    """
    import os
    W = _read_safetensors(os.path.join(model_dir, "mtp.safetensors"))
    P = "mtp.layers.0."
    g = {}

    def norm(k):
        return W[k] + mx.array(1.0, W[k].dtype)  # -1 シフト復元（dtype 保持）

    def maybe(k):
        return norm(k) if shift_extra else W[k]

    g["fc"] = W["mtp.fc.weight"]
    g["pre_emb"] = maybe("mtp.pre_fc_norm_embedding.weight")
    g["pre_hid"] = maybe("mtp.pre_fc_norm_hidden.weight")
    g["final_norm"] = maybe("mtp.norm.weight")
    g["in_ln"] = norm(P + "input_layernorm.weight")
    g["post_ln"] = norm(P + "post_attention_layernorm.weight")
    # attention
    for nm in ("q_proj", "k_proj", "v_proj", "o_proj"):
        g[nm] = W[P + f"self_attn.{nm}.weight"]
    g["q_norm"] = norm(P + "self_attn.q_norm.weight")
    g["k_norm"] = norm(P + "self_attn.k_norm.weight")
    # MoE F16 parts
    g["gate"] = W[P + "mlp.gate.weight"]
    g["se_gate"] = W[P + "mlp.shared_expert_gate.weight"]
    for nm in ("gate_proj", "up_proj", "down_proj"):
        g["se_" + nm] = W[P + f"mlp.shared_expert.{nm}.weight"]
    # experts: stack per-expert quantized arrays → [256, out, ...]
    for proj in ("gate_proj", "up_proj", "down_proj"):
        for part in ("weight", "scales", "biases"):
            stack = [W[P + f"mlp.experts.{e}.{proj}.{part}"] for e in range(256)]
            g[f"E.{proj}.{part}"] = mx.stack(stack, axis=0)
    return g


class MTPHead(nn.Module):
    def __init__(self, args, g, embed, lm_head):
        super().__init__()
        self.args = args
        self._embed = embed          # main embed_tokens（共有）
        self._lm_head = lm_head      # main lm_head（共有）
        eps = args.rms_norm_eps
        H = args.hidden_size
        self.fc = nn.Linear(2 * H, H, bias=False)
        self.pre_fc_norm_embedding = nn.RMSNorm(H, eps)
        self.pre_fc_norm_hidden = nn.RMSNorm(H, eps)
        self.self_attn = Qwen3NextAttention(args)
        self.input_layernorm = nn.RMSNorm(H, eps)
        self.post_attention_layernorm = nn.RMSNorm(H, eps)
        self.gate = nn.Linear(H, args.num_experts, bias=False)
        self.shared_expert_gate = nn.Linear(H, 1, bias=False)
        self.se_gate = nn.Linear(H, args.shared_expert_intermediate_size, bias=False)
        self.se_up = nn.Linear(H, args.shared_expert_intermediate_size, bias=False)
        self.se_down = nn.Linear(args.shared_expert_intermediate_size, H, bias=False)
        self.norm = nn.RMSNorm(H, eps)
        self._E = {p: {q: g[f"E.{p}.{q}"] for q in ("weight", "scales", "biases")}
                   for p in ("gate_proj", "up_proj", "down_proj")}
        # 重みロード
        self.fc.weight = g["fc"]
        self.pre_fc_norm_embedding.weight = g["pre_emb"]
        self.pre_fc_norm_hidden.weight = g["pre_hid"]
        self.norm.weight = g["final_norm"]
        self.input_layernorm.weight = g["in_ln"]
        self.post_attention_layernorm.weight = g["post_ln"]
        self.self_attn.q_proj.weight = g["q_proj"]
        self.self_attn.k_proj.weight = g["k_proj"]
        self.self_attn.v_proj.weight = g["v_proj"]
        self.self_attn.o_proj.weight = g["o_proj"]
        self.self_attn.q_norm.weight = g["q_norm"]
        self.self_attn.k_norm.weight = g["k_norm"]
        self.gate.weight = g["gate"]
        self.shared_expert_gate.weight = g["se_gate"]
        self.se_gate.weight = g["se_gate_proj"]
        self.se_up.weight = g["se_up_proj"]
        self.se_down.weight = g["se_down_proj"]

    def _moe(self, x):
        gates = mx.softmax(self.gate(x), axis=-1, precise=True)
        k = self.args.num_experts_per_tok
        inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
        scores = mx.take_along_axis(gates, inds, axis=-1)
        if self.args.norm_topk_prob:
            scores = scores / scores.sum(axis=-1, keepdims=True)
        xe = mx.expand_dims(x, (-2, -3))

        def qmm(xx, proj):
            E = self._E[proj]
            return mx.gather_qmm(xx, E["weight"], E["scales"], E["biases"],
                                 rhs_indices=inds, transpose=True, group_size=MTP_GS,
                                 bits=MTP_BITS, mode="affine", sorted_indices=False)
        h = swiglu(qmm(xe, "gate_proj"), qmm(xe, "up_proj"))
        y = qmm(h, "down_proj").squeeze(-2)            # [B,L,k,H]
        y = (y * scores[..., None]).sum(axis=-2)
        sh = self.se_down(swiglu(self.se_gate(x), self.se_up(x)))
        sh = mx.sigmoid(self.shared_expert_gate(x)) * sh
        return y + sh

    def __call__(self, h_prev, next_tok, concat_order="emb_hid", mask_mode="causal",
                 cache=None, return_hidden=False):
        """h_prev:[B,L,H] main hidden, next_tok:[B,L] 条件トークン → 次々トークンの logits。
        cache=KVCache を渡すと self_attn が KV 蓄積（投機ループ用、offset で RoPE）。
        return_hidden=True で (logits, x) を返す。x は final-norm 前の head 内部 hidden で、
        D2+ の自己連鎖ドラフト（EAGLE 流）に h_prev として食わせる。"""
        emb = self._embed(next_tok)
        e = self.pre_fc_norm_embedding(emb)
        hh = self.pre_fc_norm_hidden(h_prev)
        cat = mx.concatenate([e, hh] if concat_order == "emb_hid" else [hh, e], axis=-1)
        x = self.fc(cat)
        if cache is not None:
            mask = create_attention_mask(x, cache) if x.shape[1] > 1 else None
        elif mask_mode == "diag":
            Lq = x.shape[1]
            mask = mx.where(mx.eye(Lq, dtype=mx.bool_), 0.0, -1e9).astype(x.dtype)
        else:
            mask = create_attention_mask(x, None)
        r = self.self_attn(self.input_layernorm(x), mask, cache)
        x = x + r
        x = x + self._moe(self.post_attention_layernorm(x))
        logits = self._lm_head(self.norm(x))
        return (logits, x) if return_hidden else logits


def build_head(model_dir, lm):
    """loaded language_model から MTPHead を構築（正準構成）。"""
    g = load_mtp_weights(model_dir, shift_extra=False)
    head = MTPHead(lm.args, g, lm.model.embed_tokens, lm.lm_head)
    mx.eval(head.parameters())
    return head


def _main_forward(lm, inputs):
    """main LM を手動で回し (pre_norm_hidden[B,L,H], logits[B,L,V]) を返す。"""
    inner = lm.model
    h = inner.embed_tokens(inputs)
    mask = create_attention_mask(h, None)
    from mlx_lm.models.base import create_ssm_mask
    ssm = create_ssm_mask(h, None)
    cache = [None] * len(inner.layers)
    for layer, c in zip(inner.layers, cache):
        m = ssm if layer.is_linear else mask
        h = layer(h, mask=m, cache=c)
    pre = h
    post = inner.norm(h)
    logits = lm.lm_head(post)
    return pre, post, logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--ctx", type=int, default=128)
    ap.add_argument("--gen", type=int, default=128)
    ap.add_argument("--sweep", action="store_true", help="norm-shift×hidden を総当り")
    args = ap.parse_args()

    print("[mtp] loading main model ...", file=sys.stderr)
    model, tok = load(args.model)
    lm = model.language_model
    margs = lm.args
    # greedy 自己生成で teacher-forced 列 S を作る
    base = "def binary_search(arr, target):\n    lo, hi = 0, len(arr) - 1\n    while lo <= hi:\n"
    ids = tok.encode(base)
    while len(ids) < args.ctx:
        ids = ids + tok.encode(base)
    seq = ids[:args.ctx]
    print(f"[mtp] greedy self-gen {args.gen} tokens ...", file=sys.stderr)
    cur = mx.array(seq)[None]
    for _ in range(args.gen):
        _, _, lg = _main_forward(lm, cur)
        nt = int(mx.argmax(lg[0, -1]).item())
        cur = mx.concatenate([cur, mx.array([[nt]])], axis=1)
    full = cur                                            # [1, ctx+gen]

    # teacher-forced: 全位置の main hidden/greedy を 1 パスで
    pre, post, logits = _main_forward(lm, full)
    greedy = mx.argmax(logits, axis=-1)[0]                # [L] 各位置の次トークン予測
    L = full.shape[1]

    # 評価窓: ctx..L-2（次々トークンが存在する範囲）
    lo, hi = args.ctx, L - 2
    next_tok = full[:, 1:]                                 # token_{i+1}
    tgt = full[0, lo + 2:hi + 2]

    if args.sweep:
        combos = [(s, hm) for s in (False, True) for hm in ("pre", "post")]
    else:
        combos = [(False, "post")]                         # 正準構成
    for shift_extra, hid_mode in combos:
        g = load_mtp_weights(args.model, shift_extra=shift_extra)
        head = MTPHead(margs, g, lm.model.embed_tokens, lm.lm_head)
        mx.eval(head.parameters())
        hsrc = pre if hid_mode == "pre" else post
        for mm in ("causal", "diag"):
            draft = mx.argmax(head(hsrc[:, :-1], next_tok, "emb_hid", mm), axis=-1)[0]
            acc = float(mx.mean((draft[lo:hi] == tgt).astype(mx.float32)).item())
            print(f"  [canonical] shift_extra={int(shift_extra)} hidden={hid_mode:4} "
                  f"mask={mm:6} acceptance={acc:.3f}  (n={hi-lo}, doc=0.886)")


if __name__ == "__main__":
    main()
