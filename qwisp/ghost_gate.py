"""AoT-style gates for ghost-mode (arXiv 2502.12018).

Ghost only helps on DECOMPOSABLE requests (independent parallel sections); on
reasoning/code it breaks (Plato measured math -67% / coding -57%). Two gates:

1. decomposable() — cheap per-request classifier: can this be split into independent
   sections written in parallel without needing each other? YES -> ghost, NO -> run whole.
   (AoT's "complexity-reduction / atoms with no incoming edges are independent" signal,
    reduced to one cheap model call.)
2. judge() — AoT's quality-aware termination: LLM-as-judge picks the better of
   {ghost output, single-pass output}. Use OFFLINE to calibrate which task types are
   safe for ghost; the cheap runtime guard is gate #1.

Run (demo: gate routes 3 tasks, judge compares on the decomposable one):
  PY="$HOME/Library/Application Support/MTPLX/runtime-venv/bin/python3"
  PYTHONPATH=<repo> "$PY" -m qwisp.ghost_gate "$HOME/.mtplx/models/unsloth--Qwen3.6-35B-A3B-UD-MLX-3bit"
"""
from __future__ import annotations
import argparse
import re

from mlx_lm import load, generate

TASKS = {
    "blog (decomposable)":  "Compose an engaging travel blog post about a recent trip to Hawaii, "
                            "highlighting cultural experiences and must-see attractions.",
    "math (sequential)":    "A train leaves Boston at 60 mph. Two hours later a second train leaves "
                            "the same station at 80 mph on the same track. How far from Boston do they meet? "
                            "Show your reasoning.",
    "code (sequential)":    "Write a Python function that merges two sorted linked lists into one sorted "
                            "linked list, then explain its time complexity.",
}


def ids(tok, user):
    m = [{"role": "user", "content": user}]
    try:
        return tok.apply_chat_template(m, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(m, add_generation_prompt=True)


def decomposable(model, tok, task):
    """Runtime gate: True if ghost-mode should apply. One cheap classifier call."""
    q = (f"Task: \"{task}\"\n\n"
         "Can this be split into 3+ sections that can be written FULLY IN PARALLEL, where no "
         "section needs the content of another to be written correctly? A story arc, a proof, or "
         "code where later parts depend on earlier ones is NOT parallelizable. "
         "Answer with exactly one word on the first line: YES or NO. Then one short reason.")
    out = generate(model, tok, ids(tok, q), max_tokens=60).strip()
    first = out.splitlines()[0].upper() if out else ""
    verdict = "YES" in first and "NO" not in first
    reason = " ".join(out.splitlines()[1:])[:120] if len(out.splitlines()) > 1 else out[:120]
    return verdict, reason


def judge(model, tok, task, a_text, b_text):
    """AoT quality gate: which output better answers the task? Returns 'A'/'B'/'tie'.
    ponytail: single order -> mild position bias; fine as an offline calibration signal."""
    q = (f"Task: {task}\n\n[Response A]\n{a_text}\n\n[Response B]\n{b_text}\n\n"
         "Which response better answers the task, considering coherence, NON-repetition across "
         "parts, and completeness? Answer exactly 'A', 'B', or 'tie' on the first line.")
    out = generate(model, tok, ids(tok, q), max_tokens=20).strip().upper()
    tok0 = re.sub(r"[^AB]", "", out.splitlines()[0]) if out else ""
    return "A" if tok0.startswith("A") else "B" if tok0.startswith("B") else "tie"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    args = ap.parse_args()
    model, tok = load(args.model)

    print("=== GATE #1: decomposability routing ===")
    for name, task in TASKS.items():
        ok, reason = decomposable(model, tok, task)
        print(f"  {name:22s} -> {'GHOST' if ok else 'WHOLE':5s}   ({reason})")

    # GATE #2 demo: judge ghost-style (independent sections concatenated) vs single-pass,
    # on the decomposable blog task.
    print("\n=== GATE #2: judge ghost vs single-pass (blog) ===")
    blog = TASKS["blog (decomposable)"]
    single = generate(model, tok, ids(tok, blog), max_tokens=480)
    heads = ["Arrival", "Culture", "Food", "Farewell"]
    secs = [generate(model, tok, ids(tok,
            f"Write ONLY the '{h}' section (2 paragraphs) of a Hawaii travel blog, no title."),
            max_tokens=120) for h in heads]
    ghost = "\n\n".join(f"## {h}\n{s.strip()}" for h, s in zip(heads, secs))
    winner = judge(model, tok, blog, ghost, single)  # A=ghost, B=single
    print(f"  winner: {'ghost' if winner == 'A' else 'single-pass' if winner == 'B' else 'tie'}")
    print("  (offline signal — calibrate ghost's safe task types with this, gate at runtime via #1)")


if __name__ == "__main__":
    main()
