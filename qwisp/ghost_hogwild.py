"""Hogwild visibility — QUALITY upper-bound probe (Python, no engine change).

Real Hogwild (arXiv 2504.06261) gives sections token-level visibility into each other via
shared KV. Before paying the engine cost (cross-slot attention + custom RoPE re-rotation),
test the GATING question cheaply: does mutual visibility fix ghost's repetition well enough
to beat single-pass on the judge?

Upper bound = 2-pass: pass 1 blind (fast drafts), pass 2 regenerate each section while it
SEES all siblings' full pass-1 drafts (more info than Hogwild's in-progress view). If this
upper bound can't beat single-pass, Hogwild is closed regardless of speed. If it wins, the
ctxpad proxy already showed the single-pass engine version runs at ~2.3x, so build it.

Run:
  PY="$HOME/Library/Application Support/MTPLX/runtime-venv/bin/python3"
  PYTHONPATH=<repo> "$PY" -m qwisp.ghost_hogwild "$HOME/.mtplx/models/unsloth--Qwen3.6-35B-A3B-UD-MLX-3bit"
"""
from __future__ import annotations
import argparse
import time

from mlx_lm import load, generate, batch_generate
from qwisp.ghost_plato import TASK, HEADERS, ids, section_prompt, repetition
from qwisp.ghost_gate import judge


def duel(model, tok, task, a, b):
    """2-order judge to blunt position bias. Returns a's score: 2=clear win, 1=tie/split, 0=loss."""
    w1 = judge(model, tok, task, a, b)   # a is A
    w2 = judge(model, tok, task, b, a)   # a is B
    return (1 if w1 == "A" else 0) + (1 if w2 == "B" else 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--sections", type=int, default=8)
    ap.add_argument("--sec-tokens", type=int, default=120)
    args = ap.parse_args()
    model, tok = load(args.model)
    heads = HEADERS[:args.sections]
    B = len(heads)

    # pass 1: blind (base ghost)
    t = time.perf_counter()
    p1 = batch_generate(model, tok, [section_prompt(tok, h, []) for h in heads],
                        max_tokens=args.sec_tokens, verbose=False).texts
    t1 = time.perf_counter() - t

    # pass 2: each section regenerated seeing ALL siblings' pass-1 drafts (visibility upper bound)
    t = time.perf_counter()
    p2_prompts = [section_prompt(tok, heads[i], [p1[j] for j in range(B) if j != i]) for i in range(B)]
    p2 = batch_generate(model, tok, p2_prompts, max_tokens=args.sec_tokens, verbose=False).texts
    t2 = time.perf_counter() - t

    # single-pass reference (coherent baseline)
    t = time.perf_counter()
    single = generate(model, tok, ids(tok, TASK), max_tokens=args.sec_tokens * B)
    ts = time.perf_counter() - t

    blind = "\n\n".join(f"## {h}\n{s}" for h, s in zip(heads, p1))
    seen = "\n\n".join(f"## {h}\n{s}" for h, s in zip(heads, p2))

    print("=" * 70)
    print(f"HOGWILD visibility upper-bound · B={B} · sec={args.sec_tokens}t")
    print("=" * 70)
    print(f"cross-rep (trigram):  blind {repetition(p1):.1%}  ->  seen-siblings {repetition(p2):.1%}")
    print(f"wall: pass1(blind) {t1:.1f}s  pass2(visible) {t2:.1f}s  single {ts:.1f}s")
    print("-" * 70)
    print("judge (2-order duel; 2=win 1=tie 0=loss):")
    print(f"  blind ghost   vs single-pass : {duel(model, tok, TASK, blind, single)}/2")
    print(f"  visible ghost vs single-pass : {duel(model, tok, TASK, seen, single)}/2  <-- gating result")
    print(f"  visible ghost vs blind ghost : {duel(model, tok, TASK, seen, blind)}/2")
    print("=" * 70)
    print("\n### VISIBLE-GHOST OUTPUT (pass 2)\n" + seen)


if __name__ == "__main__":
    main()
