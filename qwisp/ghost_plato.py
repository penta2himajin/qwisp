"""ghost-mode + Plato/SGD-style shared-context waves (mechanism PoC, MLX).

Base ghost generates all B sections BLIND to each other -> motif repetition + lost arc
(non-lossless #1). Plato (arXiv 2402.12280) fix: partition sections into K topological
WAVES; each wave sees the already-generated text of prior waves via a growing shared
context, so later sections stop repeating earlier ones. Cost: parallelism drops ~K x
(K waves of B/K instead of one wave of B), which is exactly the speed<->coherence
tradeoff we want to quantify.

This validates the mechanism in Python (fast iterate) before any resident-engine port:
sweep K, measure cross-section repetition vs effective tok/s.

Run:
  PY="$HOME/Library/Application Support/MTPLX/runtime-venv/bin/python3"
  PYTHONPATH=<repo> "$PY" -m qwisp.ghost_plato \
    "$HOME/.mtplx/models/unsloth--Qwen3.6-35B-A3B-UD-MLX-3bit" --sections 8 --waves 1,2,4,8
"""
from __future__ import annotations
import argparse
import re
import time

from mlx_lm import load, generate, batch_generate

TASK = ("Compose an engaging travel blog post about a recent trip to Hawaii, "
        "highlighting cultural experiences and must-see attractions.")
HEADERS = ["Arrival and First Impressions", "Iconic Landscapes and Attractions",
           "Local Cuisine and Flavors", "Cultural Traditions and People",
           "Sacred and Historical Sites", "Adventure and the Outdoors",
           "Hidden Gems Off the Beaten Path", "Reflections and Farewell",
           "Beaches and Snorkeling", "Volcanoes and Craters", "Hula and Music",
           "Where to Stay and Getting Around"]


def ids(tok, user):
    m = [{"role": "user", "content": user}]
    try:
        return tok.apply_chat_template(m, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(m, add_generation_prompt=True)


def ntok(tok, s):
    return len(tok.encode(s))


def section_prompt(tok, header, context):
    u = (f"You are writing a travel blog post about a recent trip to Hawaii. "
         f"Write ONLY the section titled '{header}' — 2 short paragraphs, no title, no other sections.")
    if context:  # Plato growing shared-context prefix + explicit anti-repeat
        joined = "\n---\n".join(context)
        u += ("\n\nSections already written by your co-authors (do NOT reuse their imagery, "
              f"phrases, sensory details, or examples — stay complementary):\n{joined}")
    return ids(tok, u)


def waves(headers, k):
    """Partition headers into waves. k>0: k roughly-equal contiguous waves.
    k==0 (SMART/DAG): one maximal-parallel wave of independents + a final tiny wave
    for the integrative section (conclusion) that genuinely depends on the rest."""
    if k == 0:
        return [headers[:-1], headers[-1:]] if len(headers) > 1 else [headers]
    n = len(headers)
    size = (n + k - 1) // k
    return [headers[i:i + size] for i in range(0, n, size)]


def trigrams(text):
    w = re.findall(r"[a-z]+", text.lower())
    return [tuple(w[i:i + 3]) for i in range(len(w) - 2)]


def repetition(sections):
    """cross-section repetition: fraction of distinct trigrams that appear in >=2 sections.
    Higher = more sections echo each other = worse coherence."""
    from collections import Counter
    seen_in = Counter()
    for s in sections:
        for tg in set(trigrams(s)):
            seen_in[tg] += 1
    if not seen_in:
        return 0.0
    shared = sum(1 for c in seen_in.values() if c >= 2)
    return shared / len(seen_in)


def run_waves(model, tok, headers, k, sec_tokens):
    context, sections, wall, gtok = [], [], 0.0, 0
    for wave in waves(headers, k):
        prompts = [section_prompt(tok, h, context) for h in wave]
        t = time.perf_counter()
        br = batch_generate(model, tok, prompts, max_tokens=sec_tokens, verbose=False)
        wall += time.perf_counter() - t
        sections += br.texts
        context += br.texts
        gtok += sum(ntok(tok, s) for s in br.texts)
    return sections, wall, gtok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--sections", type=int, default=8)
    ap.add_argument("--sec-tokens", type=int, default=120)
    ap.add_argument("--waves", default="1,2,4,8")
    args = ap.parse_args()
    Ks = [int(x) for x in args.waves.split(",")]
    headers = HEADERS[:args.sections]

    model, tok = load(args.model)
    t = time.perf_counter()
    base = generate(model, tok, ids(tok, TASK), max_tokens=200)
    ss = ntok(tok, base) / (time.perf_counter() - t)
    print(f"[baseline] single-stream {ss:.1f} tok/s\n")

    rows, dumps = [], {}
    for k in Ks:
        secs, wall, gtok = run_waves(model, tok, headers, k, args.sec_tokens)
        rows.append((k, len(waves(headers, k)), wall, gtok / wall, repetition(secs)))
        dumps[k] = secs

    print("=" * 72)
    print(f"PLATO-WAVE ghost · B={args.sections} sections · sec={args.sec_tokens}t · ss={ss:.0f} tok/s")
    print("=" * 72)
    print(f"{'K':>2} {'waves':>5} {'wall_s':>7} {'eff tok/s':>10} {'eff×':>6} {'cross-rep':>10}")
    for k, nw, wall, eff, rep in rows:
        print(f"{k:>2} {nw:>5} {wall:>7.2f} {eff:>10.1f} {eff/ss:>5.2f}x {rep:>9.1%}")
    print("=" * 72)
    print("  cross-rep = fraction of trigrams shared across >=2 sections (lower = less repetition)")

    for k in (Ks[0], Ks[-1]):
        print(f"\n########## K={k} sections ##########")
        for h, s in zip(headers, dumps[k]):
            print(f"\n## {h}\n{s.strip()}")


if __name__ == "__main__":
    main()
