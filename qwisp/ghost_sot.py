"""Standard-SoT prompting fix — did ghost under-specify the expand prompt?

Bug caught by owner: ghost's expand prompt passed only the ASSIGNED header ("write the
'Local Cuisine' section"), NOT the full skeleton. Standard Skeleton-of-Thought (Ning 2023)
gives each parallel expansion the QUESTION + the WHOLE skeleton (all points) + "expand ONLY
point i" — so every section sees the division of labor and stays in its lane. Passing only
the isolated point is blinder than SoT and likely drives the section overlap (e.g. two
sections both writing Na Pali because neither knows the other exists).

The skeleton is a SHARED prefix across all sections -> cheap (ctxpad proxy: long prompts
~-12%), and KV-cacheable once. This tests whether the standard-SoT prompt closes the quality
gap that the mis-implemented "isolated header" version left open.

Compares: isolated-header (old ghost) vs full-skeleton (standard SoT) vs single-pass.

Run:
  PY="$HOME/Library/Application Support/MTPLX/runtime-venv/bin/python3"
  PYTHONPATH=<repo> "$PY" -m qwisp.ghost_sot "$HOME/.mtplx/models/unsloth--Qwen3.6-35B-A3B-UD-MLX-3bit"
"""
from __future__ import annotations
import argparse
import time

from mlx_lm import load, generate, batch_generate
from qwisp.ghost_plato import TASK, HEADERS, ids, repetition
from qwisp.ghost_gate import judge


def iso_prompt(tok, header):
    """OLD ghost: isolated header only (blinder than SoT)."""
    return ids(tok, f"You are writing a travel blog post about a recent trip to Hawaii. "
                    f"Write ONLY the section titled '{header}' — 2 short paragraphs, no title.")


def sot_prompt(tok, headers, i):
    """STANDARD SoT: task + full skeleton + expand ONLY point i (sees division of labor)."""
    skel = "\n".join(f"{j+1}. {h}" for j, h in enumerate(headers))
    return ids(tok,
        f"Task: {TASK}\n\nThe full outline of the blog post is:\n{skel}\n\n"
        f"Write ONLY section {i+1} ('{headers[i]}'). The other sections are written separately by "
        f"co-authors — stay strictly in this section's lane and do NOT cover the other sections' "
        f"topics or reuse their likely imagery. 2 short paragraphs, no title.")


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
    heads = HEADERS[:args.sections]
    B = len(heads)

    t = time.perf_counter()
    iso = batch_generate(model, tok, [iso_prompt(tok, h) for h in heads],
                         max_tokens=args.sec_tokens, verbose=False).texts
    t_iso = time.perf_counter() - t

    t = time.perf_counter()
    sot = batch_generate(model, tok, [sot_prompt(tok, heads, i) for i in range(B)],
                         max_tokens=args.sec_tokens, verbose=False).texts
    t_sot = time.perf_counter() - t

    single = generate(model, tok, ids(tok, TASK), max_tokens=args.sec_tokens * B)

    iso_txt = "\n\n".join(f"## {h}\n{s}" for h, s in zip(heads, iso))
    sot_txt = "\n\n".join(f"## {h}\n{s}" for h, s in zip(heads, sot))

    print("=" * 70)
    print(f"STANDARD-SoT prompt fix · B={B} · sec={args.sec_tokens}t")
    print("=" * 70)
    print(f"cross-rep (trigram):  isolated-header {repetition(iso):.1%}  ->  full-skeleton {repetition(sot):.1%}")
    print(f"wall:  isolated {t_iso:.1f}s   full-skeleton {t_sot:.1f}s  (skeleton = shared prefix)")
    print("-" * 70)
    print("judge (2-order duel; 2=win 1=tie 0=loss):")
    print(f"  isolated-header ghost vs single-pass : {duel(model, tok, iso_txt, single)}/2")
    print(f"  full-skeleton  ghost vs single-pass  : {duel(model, tok, sot_txt, single)}/2  <-- gating result")
    print(f"  full-skeleton  ghost vs isolated     : {duel(model, tok, sot_txt, iso_txt)}/2")
    print("=" * 70)
    print("\n### FULL-SKELETON (standard SoT) OUTPUT\n" + sot_txt)


if __name__ == "__main__":
    main()
