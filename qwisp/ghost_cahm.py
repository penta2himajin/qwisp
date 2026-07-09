"""CAHM post-generation merge for ghost (arXiv 2502.00977, Ou & Lapata 2025).

CAHM = Context-Aware Hierarchical Merging (for long-doc summarization): pieces generated
independently, then reconciled bottom-up by an LLM merge that is GROUNDED in context. All my
prior fixes were GENERATION-TIME (Hogwild full sibling visibility, SoT full skeleton) and all
failed. CAHM's untested lever is a POST-GENERATION merge: editing existing drafts to remove
redundancy + restore arc is easier than avoiding redundancy during blind generation.

Port: swap CAHM's grounding from source-doc→hallucination to siblings+outline→redundancy/arc.
Use Support/Refine (keep prose, rewrite against context), NOT Replace (discards prose). At B=8
the drafts are small, so pass full drafts to a single N-ary merge (the paper's Extract step is
a scaling optimization for large sources — skipped, noted).

Compares: blind-concat (base ghost) vs CAHM-merged vs single-pass, on quality (judge) + wall.

Run:
  PY="$HOME/Library/Application Support/MTPLX/runtime-venv/bin/python3"
  PYTHONPATH=<repo> "$PY" -m qwisp.ghost_cahm "$HOME/.mtplx/models/unsloth--Qwen3.6-35B-A3B-UD-MLX-3bit"
"""
from __future__ import annotations
import argparse
import time

from mlx_lm import load, generate, batch_generate
from qwisp.ghost_plato import TASK, HEADERS, ids, repetition
from qwisp.ghost_gate import judge


def blind_prompt(tok, header):
    return ids(tok, f"You are writing a travel blog post about a recent trip to Hawaii. "
                    f"Write ONLY the section titled '{header}' — 2 short paragraphs, no title.")


def merge_prompt(tok, heads, drafts, budget):
    skel = "\n".join(f"{i+1}. {h}" for i, h in enumerate(heads))
    body = "\n\n".join(f"[Section {i+1}: {h}]\n{d}" for i, (h, d) in enumerate(zip(heads, drafts)))
    return ids(tok,
        f"You are the editor assembling a travel blog post about a recent trip to Hawaii.\n\n"
        f"Intended outline:\n{skel}\n\n"
        f"Below are {len(heads)} draft sections written INDEPENDENTLY, so they overlap and repeat "
        f"each other — the same landmarks, images, foods, and phrases recur across sections:\n\n{body}\n\n"
        f"Merge them into ONE coherent blog post (~{budget} words of budget):\n"
        f"- Remove cross-section repetition: each landmark, image, food, and phrase appears ONCE, "
        f"in the section where it fits best.\n"
        f"- Keep every DISTINCT piece of content; cut only redundancy.\n"
        f"- Add a short intro and smooth transitions so it reads as one authored narrative with an "
        f"arc, not stitched fragments.\nOutput only the finished post.")


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
    budget = args.sec_tokens * B

    # base: parallel blind generation
    t = time.perf_counter()
    drafts = batch_generate(model, tok, [blind_prompt(tok, h) for h in heads],
                            max_tokens=args.sec_tokens, verbose=False).texts
    t_gen = time.perf_counter() - t

    # CAHM post-generation merge (Support/Refine, grounded in siblings+outline)
    t = time.perf_counter()
    merged = generate(model, tok, merge_prompt(tok, heads, drafts, budget), max_tokens=budget)
    t_merge = time.perf_counter() - t

    # single-pass reference
    t = time.perf_counter()
    single = generate(model, tok, ids(tok, TASK), max_tokens=budget)
    t_single = time.perf_counter() - t

    blind = "\n\n".join(f"## {h}\n{s}" for h, s in zip(heads, drafts))

    print("=" * 70)
    print(f"CAHM post-merge · B={B} · sec={args.sec_tokens}t")
    print("=" * 70)
    print(f"cross-rep (trigram):  blind-concat {repetition(drafts):.1%}  ->  merged {repetition([merged]):.1%}*")
    print("   (*merged is one text; blind is {} sections — rep not directly comparable, see judge)".format(B))
    print(f"wall:  gen {t_gen:.1f}s + merge {t_merge:.1f}s = {t_gen+t_merge:.1f}s   single-pass {t_single:.1f}s")
    print(f"       -> CAHM-ghost is {(t_gen+t_merge)/t_single:.2f}x the single-pass wall")
    print("-" * 70)
    print("judge (2-order duel; 2=win 1=tie 0=loss):")
    print(f"  blind-concat  vs single-pass : {duel(model, tok, blind, single)}/2")
    print(f"  CAHM-merged   vs single-pass : {duel(model, tok, merged, single)}/2  <-- gating result")
    print(f"  CAHM-merged   vs blind-concat: {duel(model, tok, merged, blind)}/2")
    print("=" * 70)
    print("\n### CAHM-MERGED OUTPUT\n" + merged.strip())


if __name__ == "__main__":
    main()
