"""4-regime applicability test: does ghost apply / win on the REAL bench prompts?

Uses the actual qwisp bench prompts (code/agentic/longctx/shortnl). Ghost needs a LONG,
decomposable output to amortize the skeleton + parallelize. This measures, per regime:
  1. decomposability gate verdict (AoT gate),
  2. single-pass output length (short output => nothing to parallelize => ghost N/A),
  3. for the genuinely item-parallel regime (agentic: 3 independent tool calls), an actual
     parallel-ghost vs single-pass comparison.

shortnl == the Hawaii blog already tested exhaustively (ghost loses 0/2 on arc), so the new
signal is code/agentic/longctx.

Run:
  PY="$HOME/Library/Application Support/MTPLX/runtime-venv/bin/python3"
  PYTHONPATH=<repo> "$PY" -m qwisp.ghost_regimes "$HOME/.mtplx/models/unsloth--Qwen3.6-35B-A3B-UD-MLX-3bit"
"""
from __future__ import annotations
import argparse
import time

from mlx_lm import load, generate, batch_generate
from qwisp.bench_prompts import PROMPTS
from qwisp.ghost_gate import decomposable, ids


def ntok(tok, s):
    return len(tok.encode(s))


def agentic_parallel(model, tok, task):
    """Item-level ghost: generate each of the 3 tool calls in parallel, then concat.
    The one regime whose output is genuinely independent items."""
    tools = ["get_weather", "book_flight", "convert_currency"]
    prompts = [ids(tok, f"{task}\n\nEmit ONLY the single JSON tool call for '{t}' needed to "
                        f"answer the user (no other calls, no prose).") for t in tools]
    t0 = time.perf_counter()
    parts = batch_generate(model, tok, prompts, max_tokens=80, verbose=False).texts
    dt = time.perf_counter() - t0
    return parts, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    args = ap.parse_args()
    model, tok = load(args.model)

    print("=" * 72)
    print("4-REGIME ghost applicability (real bench prompts)")
    print("=" * 72)
    for name, spec in PROMPTS.items():
        task = spec["text"]
        ok, reason = decomposable(model, tok, task)
        single = generate(model, tok, ids(tok, task), max_tokens=256)
        n = ntok(tok, single)
        snip = " ".join(single.split())[:110]
        print(f"\n[{name}]  gate={'GHOST' if ok else 'WHOLE'}  single-pass output={n} tok")
        print(f"   gate reason: {reason[:100]}")
        print(f"   single-pass: {snip}")
        # applicability read
        if n < 60:
            print(f"   -> output too short ({n} tok): nothing to parallelize, ghost N/A")

    # agentic: the one genuinely item-parallel regime — actually run parallel ghost
    print("\n" + "=" * 72)
    print("AGENTIC item-parallel ghost (3 independent tool calls) vs single-pass")
    print("=" * 72)
    task = PROMPTS["agentic"]["text"]
    parts, dt = agentic_parallel(model, tok, task)
    single = generate(model, tok, ids(tok, task), max_tokens=200)
    print(f"parallel wall {dt:.1f}s for 3 calls (~{sum(ntok(tok,p) for p in parts)} tok)")
    for t_, p in zip(["get_weather", "book_flight", "convert_currency"], parts):
        print(f"\n  [{t_}] {' '.join(p.split())[:160]}")
    print(f"\n  [single-pass] {' '.join(single.split())[:300]}")


if __name__ == "__main__":
    main()
