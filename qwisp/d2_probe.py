"""D2(2段ドラフト)の受理率天井を teacher-forced で測る — 8GB streaming の IO 償却を深める前の GO/NO-GO.

D1 streaming は MTP reject 修正後 1.53x。IO 律速の forward では「1 forward で受理されるトークン数」を
増やすほど速い。D2 は head を自己連鎖（EAGLE 流: head 自身の hidden を h_prev に）させ、
[u,d1,d2] を 1 forward で verify。期待受理長 1+a1+a1·a2 が D1 の 1+a1 を上回れば streaming で効く。

ただし head は D1 学習なので自己連鎖の受理率は未知。ここで測る:
  - batch D1 受理率 a1 を実測 0.94 と突き合わせ（方法の校正）
  - batch D2 条件付き受理率 a2 = P(d2 正解 | d1 正解)
校正で a1 が実測に一致すれば a2 は信頼できる。

実行: PY -m qwisp.d2_probe <model_dir> [--ctx 128 --gen 96]
"""
from __future__ import annotations
import argparse
import sys

import mlx.core as mx

from mlx_lm import load
from .mtp_head import build_head, _main_forward


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("--ctx", type=int, default=128)
    ap.add_argument("--gen", type=int, default=96)
    args = ap.parse_args()
    print("[d2] loading ...", file=sys.stderr)
    model, tok = load(args.model)
    lm = model.language_model
    head = build_head(args.model, lm)

    base = "def binary_search(arr, target):\n    lo, hi = 0, len(arr) - 1\n    while lo <= hi:\n"
    ids = tok.encode(base)
    while len(ids) < args.ctx:
        ids = ids + tok.encode(base)
    cur = mx.array(ids[:args.ctx])[None]
    print(f"[d2] greedy self-gen {args.gen} ...", file=sys.stderr)
    for _ in range(args.gen):
        _, _, lg = _main_forward(lm, cur)
        cur = mx.concatenate([cur, mx.array([[int(mx.argmax(lg[0, -1]).item())]])], axis=1)
    full = cur
    pre, post, logits = _main_forward(lm, full)
    greedy = mx.argmax(logits, axis=-1)[0]                 # token_{i+1} の予測（= teacher 次トークン）
    L = full.shape[1]
    lo, hi = args.ctx, L - 3                                # 3-ahead が要る

    # D1: head(post_i, tok_{i+1}) → token_{i+2} を予測、hidden X1 も取る
    next_tok = full[:, 1:]
    lg1, X1 = head(post[:, :-1], next_tok, "emb_hid", "causal", return_hidden=True)
    d1 = mx.argmax(lg1, axis=-1)[0]                         # d1[i] = 予測 token_{i+2}
    tgt2 = full[0, 2:]                                      # 真の token_{i+2}
    ok1 = (d1[lo:hi] == tgt2[lo:hi])

    # D2: 自己連鎖 head(X1_i, d1_i) → token_{i+3} を予測
    d1_tok = d1[None]                                       # [1, L-1]  draft を次の条件トークンに
    lg2 = head(X1, d1_tok, "emb_hid", "causal")
    d2 = mx.argmax(lg2, axis=-1)[0]                         # d2[i] = 予測 token_{i+3}
    tgt3 = full[0, 3:]
    # 位置合わせ: X1[i] は token_{i+2} 予測の hidden。D2 はそこから token_{i+3}。
    ok2 = (d2[lo:hi] == tgt3[lo:hi])

    a1 = float(mx.mean(ok1.astype(mx.float32)).item())
    a2_uncond = float(mx.mean(ok2.astype(mx.float32)).item())
    both = float(mx.mean((ok1 & ok2).astype(mx.float32)).item())
    a2_cond = both / a1 if a1 > 0 else 0.0

    # 期待受理長（1 forward あたり）: u は常に確定 → 1 + a1 + a1·a2_cond
    elen_d1 = 1 + a1
    elen_d2 = 1 + a1 + a1 * a2_cond
    print(f"\n[d2] ctx={args.ctx} gen={args.gen}  window n={hi-lo}")
    print(f"  D1 受理率 a1            : {a1:.3f}   (実測 spec draft_accept≈0.94 と校正)")
    print(f"  D2 条件付 a2=P(d2|d1)   : {a2_cond:.3f}   (uncond {a2_uncond:.3f})")
    print(f"  期待受理長/forward  D1  : {elen_d1:.3f}")
    print(f"                      D2  : {elen_d2:.3f}   ({elen_d2/elen_d1:.2f}x)")
    print(f"  → IO律速streaming では tok/s も概ね {elen_d2/elen_d1:.2f}x 見込み"
          f" (verify が 2→3 token に増える分の IO 微増は別途)")
    if a1 < 0.85 or a1 > 0.99:
        print(f"  ⚠ a1={a1:.3f} が実測0.94と乖離 → batch近似が不正確、D2実装で要再測")


if __name__ == "__main__":
    main()
