"""isolated-header + brother-annotation, generation-time (blind-concat, NO post-hoc merge).

Owner's intent (distinct from the CAHM post-merge I built earlier): keep parallel blind
generation + flat concat, but enrich each section's prompt with CAHM-style *brother
annotations* — a short scope note for every SIBLING section describing what it covers and
what to leave to others. Each section thus demarcates its own lane precisely instead of
straying into a sibling's territory (the overlap that drove ghost's repetition).

Difference from the earlier SoT test: that passed only bare sibling HEADERS. This passes a
per-sibling SCOPE annotation ("covers X; leave Y to section N") — explicit boundaries.

Compares: isolated-header (base) vs isolated+brother-annotation vs single-pass.

Run:
  PY="$HOME/Library/Application Support/MTPLX/runtime-venv/bin/python3"
  PYTHONPATH=<repo> "$PY" -m qwisp.ghost_brother "$HOME/.mtplx/models/unsloth--Qwen3.6-35B-A3B-UD-MLX-3bit"
"""
from __future__ import annotations
import argparse
import re
import time

from mlx_lm import load, generate, batch_generate
from qwisp.ghost_plato import TASK, HEADERS, ids, repetition
from qwisp.ghost_gate import judge


def annotated_skeleton(model, tok, B):
    """One call -> B (title, scope) pairs. Scope = what it covers + what to leave to others."""
    q = (f"Task: {TASK}\n\nCreate an outline of exactly {B} sections. For EACH section give the "
         f"title and a one-sentence SCOPE note: what it covers, and which specific topics/landmarks "
         f"it should LEAVE to other sections to avoid overlap.\n"
         f"Format strictly one per line: N. Title | scope note")
    out = generate(model, tok, ids(tok, q), max_tokens=60 + 40 * B)
    pairs = []
    for line in out.splitlines():
        m = re.match(r"^\s*\d+[.)]\s*(.+?)\s*\|\s*(.+)$", line.strip())
        if m:
            pairs.append((m.group(1).strip(" *#"), m.group(2).strip()))
    if len(pairs) < B:  # fallback: template headers, generic scopes
        pairs = [(h, f"covers {h.lower()}; leave the other sections' topics to them") for h in HEADERS[:B]]
    return pairs[:B]


def iso_prompt(tok, title):
    return ids(tok, f"You are writing a travel blog post about a recent trip to Hawaii. "
                    f"Write ONLY the section titled '{title}' — 2 short paragraphs, no title.")


def brother_prompt(tok, pairs, i):
    title, scope = pairs[i]
    bros = "\n".join(f"- {t}: {s}" for j, (t, s) in enumerate(pairs) if j != i)
    return ids(tok,
        f"You are writing section {i+1} of a travel blog post about a recent trip to Hawaii.\n\n"
        f"YOUR section: {title}\nYour scope: {scope}\n\n"
        f"Your co-authors are simultaneously writing these OTHER sections. Stay strictly in your "
        f"lane: do NOT cover their topics, landmarks, or reuse their imagery:\n{bros}\n\n"
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
    iso = batch_generate(model, tok, [iso_prompt(tok, t_) for t_, _ in pairs],
                         max_tokens=args.sec_tokens, verbose=False).texts
    t_iso = time.perf_counter() - t

    t = time.perf_counter()
    bro = batch_generate(model, tok, [brother_prompt(tok, pairs, i) for i in range(B)],
                         max_tokens=args.sec_tokens, verbose=False).texts
    t_bro = time.perf_counter() - t

    single = generate(model, tok, ids(tok, TASK), max_tokens=args.sec_tokens * B)

    iso_txt = "\n\n".join(f"## {h}\n{s}" for h, s in zip(heads, iso))
    bro_txt = "\n\n".join(f"## {h}\n{s}" for h, s in zip(heads, bro))

    print("=" * 70)
    print(f"BROTHER-ANNOTATION (generation-time, blind-concat) · B={B} · sec={args.sec_tokens}t")
    print("=" * 70)
    print("annotated skeleton:")
    for i, (t_, s) in enumerate(pairs):
        print(f"  {i+1}. {t_} | {s[:70]}")
    print(f"\ncross-rep (trigram):  isolated {repetition(iso):.1%}  ->  brother-annot {repetition(bro):.1%}")
    print(f"wall:  isolated {t_iso:.1f}s   brother-annot {t_bro:.1f}s  (both blind-concat, no merge)")
    print("-" * 70)
    print("judge (2-order duel; 2=win 1=tie 0=loss):")
    print(f"  isolated       vs single-pass : {duel(model, tok, iso_txt, single)}/2")
    print(f"  brother-annot  vs single-pass : {duel(model, tok, bro_txt, single)}/2  <-- gating result")
    print(f"  brother-annot  vs isolated    : {duel(model, tok, bro_txt, iso_txt)}/2")
    print("=" * 70)
    print("\n### BROTHER-ANNOTATION OUTPUT\n" + bro_txt)


if __name__ == "__main__":
    main()
