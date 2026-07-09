"""ghost-mode PoC — resident-only latency trick (NOT streaming/bolt; opposite mechanism).

Idea (Skeleton-of-Thought, Ning 2023, adapted): single-stream decode is batch=1
latency-bound. Decompose one request into a short skeleton (section headers), then
batch-expand every section IN PARALLEL through the resident batch engine
(mlx_lm.batch_generate = B independent sequences). Throughput-as-latency.

Explicitly NON-lossless: sections are conditioned on the skeleton, not on each other,
so cross-section coherence (arc, non-repetition) is traded for parallelism. Scoped to
shortnl-style decomposable outputs; never code/agentic.

This run does (a) a B-sweep to find the diverse-batching expand ceiling, and
(b) a cheap-skeleton variant (tight skeleton budget + a zero-cost template floor) to
measure how much of the Amdahl tax the skeleton phase costs.

NOTE ON ENGINE: the batched expand runs on MLX batched execution by nature. The raw
engine has no independent-B primitive (only shared-prefix M-row verify), so ghost's
dominant phase is inherently MLX; raw would only accelerate the skeleton (the tax).

Run:
  PY="$HOME/Library/Application Support/MTPLX/runtime-venv/bin/python3"
  PYTHONPATH=<repo> "$PY" -m qwisp.ghost \
    "$HOME/.mtplx/models/unsloth--Qwen3.6-35B-A3B-UD-MLX-3bit"
"""
from __future__ import annotations
import argparse
import re
import time

from mlx_lm import load, generate, batch_generate

TASK = ("Compose an engaging travel blog post about a recent trip to Hawaii, "
        "highlighting cultural experiences and must-see attractions.")

# zero-cost template skeleton (generic blog sections) — measures the cheap-skeleton floor
TEMPLATE = ["Arrival and First Impressions", "Iconic Landscapes and Attractions",
            "Local Cuisine and Flavors", "Cultural Traditions and People",
            "Sacred and Historical Sites", "Reflections and Practical Tips",
            "Hidden Gems Off the Beaten Path", "Adventure and the Outdoors",
            "Nightlife and Local Vibe", "Where to Stay", "Getting Around", "Farewell"]


def ids(tok, user: str):
    msgs = [{"role": "user", "content": user}]
    try:  # Qwen3 thinking models: skip <think> trace for clean, comparable prose
        return tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, add_generation_prompt=True)


def strip_think(text: str) -> str:
    return text.split("</think>")[-1].strip()


def ntok(tok, text: str) -> int:
    return len(tok.encode(text))


def parse_headers(text: str, n: int) -> list[str]:
    heads = []
    for line in strip_think(text).splitlines():
        line = line.strip()
        m = re.match(r"^(?:\d+[.)]|[-*•])\s*(.+)$", line)
        cand = (m.group(1) if m else line).strip(" *#:").strip()
        if 2 <= len(cand) <= 80 and not cand.lower().startswith(("here", "sure", "outline")):
            heads.append(cand)
    return heads[:n]


def gen_skeleton(model, tok, task, n, skel_tokens):
    """Return (headers, wall_s, tokens_generated). skel_tokens<=0 => zero-cost template."""
    if skel_tokens <= 0:
        return TEMPLATE[:n], 0.0, 0
    p = (f"{task}\n\nGive ONLY a numbered list of exactly {n} short section titles. "
         f"No prose, just the {n} titles.")
    t = time.perf_counter()
    txt = generate(model, tok, ids(tok, p), max_tokens=skel_tokens)
    dt = time.perf_counter() - t
    heads = parse_headers(txt, n)
    return heads, dt, ntok(tok, txt)


def expand(model, tok, heads, sec_tokens):
    prompts = [ids(tok, (
        f"You are writing a travel blog post about a recent trip to Hawaii. "
        f"Write ONLY the section titled '{h}' — 2 short paragraphs, no title, "
        f"no other sections.")) for h in heads]
    t = time.perf_counter()
    br = batch_generate(model, tok, prompts, max_tokens=sec_tokens, verbose=False)
    dt = time.perf_counter() - t
    toks = sum(ntok(tok, s) for s in br.texts)
    return br.texts, dt, toks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--sweep", default="4,6,8,12", help="section counts B to sweep")
    ap.add_argument("--sec-tokens", type=int, default=120)
    ap.add_argument("--skel-tokens", type=int, default=120, help="skeleton budget; sweep includes cheap variants")
    ap.add_argument("--task", default=TASK)
    args = ap.parse_args()
    Bs = [int(x) for x in args.sweep.split(",")]

    model, tok = load(args.model)

    # ── single-stream baseline decode rate (measure once) ─────────────────────
    t = time.perf_counter()
    base = generate(model, tok, ids(tok, args.task), max_tokens=240)
    t_base = time.perf_counter() - t
    ss_rate = ntok(tok, base) / t_base
    print(f"[baseline] single-stream {ss_rate:.1f} tok/s ({ntok(tok, base)} tok / {t_base:.2f}s)\n")

    rows = []
    best = None
    # (a) B-sweep with a normal (model-generated) skeleton
    for B in Bs:
        heads, t_sk, sk_tok = gen_skeleton(model, tok, args.task, B, args.skel_tokens)
        if len(heads) < 2:
            print(f"[skip B={B}] skeleton parse failed"); continue
        secs, t_exp, exp_tok = expand(model, tok, heads, args.sec_tokens)
        wall = t_sk + t_exp; total = sk_tok + exp_tok
        exp_rate, eff = exp_tok / t_exp, total / wall
        rows.append(("normal", len(heads), t_sk, t_exp, exp_rate, eff))
        if best is None or eff > best[0]:
            best = (eff, heads, secs)
    # (b) cheap-skeleton variants at the best B: tight budget + zero-cost template
    Bbest = max(Bs)
    for tag, skt in [("cheap-skel(40t)", 40), ("template(0t)", 0)]:
        heads, t_sk, sk_tok = gen_skeleton(model, tok, args.task, Bbest, skt)
        if len(heads) < 2:
            print(f"[skip {tag}] skeleton parse failed"); continue
        secs, t_exp, exp_tok = expand(model, tok, heads, args.sec_tokens)
        wall = t_sk + t_exp; total = sk_tok + exp_tok
        rows.append((tag, len(heads), t_sk, t_exp, exp_tok / t_exp, total / wall))

    print("=" * 78)
    print(f"GHOST-MODE sweep · {args.model.split('/')[-1]} · sec={args.sec_tokens}t · ss={ss_rate:.0f} tok/s")
    print("=" * 78)
    print(f"{'variant':16s} {'B':>3} {'skel_s':>7} {'exp_s':>6} {'exp tok/s':>10} {'exp×':>6} {'eff tok/s':>10} {'eff×':>6}")
    for tag, B, tsk, texp, er, eff in rows:
        print(f"{tag:16s} {B:>3} {tsk:>7.2f} {texp:>6.2f} {er:>10.1f} {er/ss_rate:>5.2f}x {eff:>10.1f} {eff/ss_rate:>5.2f}x")
    print("=" * 78)

    if best:
        print("\n### BEST GHOST OUTPUT (read for coherence — repetition/arc = non-lossless cost)")
        for h, s in zip(best[1], best[2]):
            print(f"\n## {h}\n{s.strip()}")


if __name__ == "__main__":
    main()
