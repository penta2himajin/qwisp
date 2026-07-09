"""Maximal generation-time context: full-skeleton + isolated-header + brother-annotation.

Owner's #2: does the RICHEST blind prompt — every expand section sees the entire outline
(full skeleton) + its own assigned point + per-sibling scope annotations — stop losing to
single-pass? This stacks all the generation-time signals we have. Blind-concat, no merge.

Compares: brother-annot (prior best) vs full-context vs single-pass.

Run:
  PY="$HOME/Library/Application Support/MTPLX/runtime-venv/bin/python3"
  PYTHONPATH=<repo> "$PY" -m qwisp.ghost_fullctx "$HOME/.mtplx/models/unsloth--Qwen3.6-35B-A3B-UD-MLX-3bit"
"""
from __future__ import annotations
import argparse
import time

from mlx_lm import load, generate, batch_generate
from qwisp.ghost_plato import TASK, ids, repetition
from qwisp.ghost_gate import judge
from qwisp.ghost_brother import annotated_skeleton, brother_prompt


def full_prompt(tok, pairs, i):
    """full skeleton + isolated header + brother annotations."""
    title, scope = pairs[i]
    skel = "\n".join(f"{j+1}. {t}" for j, (t, _) in enumerate(pairs))
    bros = "\n".join(f"- {t}: {s}" for j, (t, s) in enumerate(pairs) if j != i)
    return ids(tok,
        f"Task: {TASK}\n\nThe full outline of the blog post is:\n{skel}\n\n"
        f"YOUR section: {i+1}. {title}\nYour scope: {scope}\n\n"
        f"Your co-authors write the other sections simultaneously; stay strictly in your lane, "
        f"do NOT cover their topics or reuse their imagery:\n{bros}\n\n"
        f"Write ONLY your section — 2 short paragraphs, no title.")


def duel(model, tok, a, b):
    return (1 if judge(model, tok, TASK, a, b) == "A" else 0) + \
           (1 if judge(model, tok, TASK, b, a) == "B" else 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--sections", type=int, default=8)
    ap.add_argument("--sec-tokens", type=int, default=120)
    args = ap.parse_args()
    model, tok = load(args.model)
    B = args.sections
    pairs = annotated_skeleton(model, tok, B)
    heads = [t for t, _ in pairs]

    t = time.perf_counter()
    bro = batch_generate(model, tok, [brother_prompt(tok, pairs, i) for i in range(B)],
                         max_tokens=args.sec_tokens, verbose=False).texts
    t_bro = time.perf_counter() - t

    t = time.perf_counter()
    full = batch_generate(model, tok, [full_prompt(tok, pairs, i) for i in range(B)],
                          max_tokens=args.sec_tokens, verbose=False).texts
    t_full = time.perf_counter() - t

    single = generate(model, tok, ids(tok, TASK), max_tokens=args.sec_tokens * B)

    bro_txt = "\n\n".join(f"## {h}\n{s}" for h, s in zip(heads, bro))
    full_txt = "\n\n".join(f"## {h}\n{s}" for h, s in zip(heads, full))

    print("=" * 70)
    print(f"FULL-CONTEXT (skeleton + isolated + brother) · B={B} · sec={args.sec_tokens}t")
    print("=" * 70)
    print(f"cross-rep (trigram):  brother {repetition(bro):.1%}  ->  full-context {repetition(full):.1%}")
    print(f"wall:  brother {t_bro:.1f}s   full-context {t_full:.1f}s")
    print("-" * 70)
    print("judge (2-order duel; 2=win 1=tie 0=loss):")
    print(f"  brother-annot vs single-pass : {duel(model, tok, bro_txt, single)}/2")
    print(f"  full-context  vs single-pass : {duel(model, tok, full_txt, single)}/2  <-- gating result")
    print(f"  full-context  vs brother     : {duel(model, tok, full_txt, bro_txt)}/2")
    print("=" * 70)
    print("\n### FULL-CONTEXT OUTPUT\n" + full_txt)


if __name__ == "__main__":
    main()
