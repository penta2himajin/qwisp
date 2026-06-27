"""MTP head の Swift 移植検証用参照 (M2c).

main の post-final-norm hidden h_prev と次トークンを MTP head に食わせて次々トークンを
draft する。Swift は mtp.safetensors を自前ロードして同じ hidden/token で再計算し、
draft argmax が一致するか＋acceptance を検証する。

dump: hidden[W,H](h_prev), tok[W](next_tok), draft[W](期待 argmax), target[W](greedy 正解)。
実行: PY -m qwisp.mtp_ref <model_dir> [--ctx 96 --gen 64 --win 16 --out /tmp/qwisp_mtp_ref.safetensors]
"""
from __future__ import annotations
import argparse

import mlx.core as mx
from mlx_lm import load
from qwisp.mtp_head import build_head, _main_forward
import qwisp.mtp_decode as MD


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--ctx", type=int, default=96)
    ap.add_argument("--gen", type=int, default=64)
    ap.add_argument("--win", type=int, default=16)
    ap.add_argument("--out", default="/tmp/qwisp_mtp_ref.safetensors")
    args = ap.parse_args()

    model, tok = load(args.model)
    lm = model.language_model
    head = build_head(args.model, lm)

    import os
    base = os.environ.get("QWISP_REF_PROMPT",
                          "def quicksort(a):\n    if len(a)<=1: return a\n    p=a[len(a)//2]\n")
    ids = tok.encode(base)
    while len(ids) < args.ctx:
        ids = ids + tok.encode(base)
    cur = mx.array(ids[:args.ctx])[None]
    for _ in range(args.gen):
        _, _, lg = _main_forward(lm, cur)
        cur = mx.concatenate([cur, mx.array([[int(mx.argmax(lg[0, -1]).item())]])], axis=1)
    full = cur

    pre, post, logits = _main_forward(lm, full)
    greedy = mx.argmax(logits, axis=-1)[0]
    L = full.shape[1]
    lo, hi = args.ctx, L - 2
    next_tok = full[:, 1:]
    draft = mx.argmax(head(post[:, :-1], next_tok, "emb_hid", "causal"), axis=-1)[0]
    tgt = full[0, lo + 2:hi + 2]
    acc = float(mx.mean((draft[lo:hi] == tgt).astype(mx.float32)).item())
    print(f"[mtp-ref] acceptance={acc:.3f} (n={hi - lo}, doc=0.886)")

    # 実プロンプト(ctx)で greedy/spec を回し token 列を dump（Swift 投機の正しさ検証用）
    spec_prompt = ids[:args.ctx]
    nspec = int(os.environ.get("QWISP_REF_NSPEC", "48"))
    g_out, _ = MD.greedy(lm, spec_prompt, nspec)
    sp_out, st, accd, _ = MD.speculative(lm, head, spec_prompt, nspec, light=True)
    spm = sum(1 for a, b in zip(g_out, sp_out) if a == b)
    print(f"[mtp-ref] spec: greedy一致 {spm}/48 accept={accd / st:.3f} (real prompt)")

    # 全系列を dump（causal attention 文脈を Swift で一致させるため）
    mx.save_safetensors(args.out, {
        "hidden": post[0, :-1],                       # h_prev [L-1, H]
        "tok": next_tok[0].astype(mx.int32),          # next_tok [L-1]
        "draft": draft.astype(mx.int32),              # 期待 argmax [L-1]
        "lo_hi": mx.array([lo, hi], mx.int32),
        "target": full[0, lo + 2:hi + 2].astype(mx.int32),
        "spec_prompt": mx.array(spec_prompt, mx.int32),
        "spec_greedy": mx.array(g_out, mx.int32),
        "spec_spec": mx.array(sp_out, mx.int32),
    })
    print(f"[mtp-ref] saved → {args.out} (L-1={full.shape[1] - 1}, eval [{lo},{hi}))")


if __name__ == "__main__":
    main()
