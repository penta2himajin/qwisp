"""ghost-mode on qwisp's OWN engine (resident continuous batching), not the mlx_lm PoC.

Python side is tokenizer-only (Swift owns the resident model). It:
  1. builds a template skeleton -> B chunk prompts (chat template, thinking off) -> token ids,
  2. shells out to `QWISP_RUN=ghost` (QwispModel.runGhost: independent-B forwardContinuous),
  3. detokenizes the dumped tokens for a coherence read; the SPEED is measured in Swift.

Run:
  PY="$HOME/Library/Application Support/MTPLX/runtime-venv/bin/python3"
  MODEL="$HOME/.mtplx/models/Youssofal--Qwen3.6-35B-A3B-MTPLX-Optimized-Speed-FP16"
  BIN=swift/.xcode-build-rel/Build/Products/Release/qwisp-poc
  PYTHONPATH=<repo> "$PY" -m qwisp.ghost_raw "$MODEL" "$BIN" --sections 8 --sec-tokens 120
"""
from __future__ import annotations
import argparse
import os
import re
import subprocess
import tempfile

from transformers import AutoTokenizer

TEMPLATE = ["Arrival and First Impressions", "Iconic Landscapes and Attractions",
            "Local Cuisine and Flavors", "Cultural Traditions and People",
            "Sacred and Historical Sites", "Reflections and Practical Tips",
            "Hidden Gems Off the Beaten Path", "Adventure and the Outdoors",
            "Nightlife and Local Vibe", "Where to Stay", "Getting Around", "Farewell",
            "Beaches and Snorkeling Spots", "Volcanoes and Craters", "Waterfalls and Rainforests",
            "Surfing and Water Sports", "Hula and Traditional Music", "Luau and Festivals",
            "Coffee Farms and Plantations", "Wildlife and Marine Life", "Shopping and Local Markets",
            "History of the Monarchy", "Island Hopping Guide", "Sunrise and Sunset Views",
            "Budget Travel Tips", "Family-Friendly Activities", "Romantic Getaway Ideas",
            "Hiking Trails and Lookouts", "Local Legends and Myths", "Weather and Best Season",
            "Photography Hotspots", "Souvenirs to Bring Home"]


def chunk_ids(tok, header):
    msg = [{"role": "user", "content": (
        f"You are writing a travel blog post about a recent trip to Hawaii. "
        f"Write ONLY the section titled '{header}' — 2 short paragraphs, no title, "
        f"no other sections.")}]
    try:  # render to string (robust across transformers/BatchEncoding quirks), then encode
        s = tok.apply_chat_template(msg, add_generation_prompt=True, enable_thinking=False, tokenize=False)
    except TypeError:
        s = tok.apply_chat_template(msg, add_generation_prompt=True, tokenize=False)
    return tok.encode(s, add_special_tokens=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("bin")
    ap.add_argument("--sections", type=int, default=8)
    ap.add_argument("--sec-tokens", type=int, default=120)
    ap.add_argument("--smart", action="store_true",
                    help="Plato dependency-aware 2-wave: last section is an integrative conclusion")
    ap.add_argument("--ctxpad", type=int, default=0,
                    help="pad each slot's prompt to ~N tokens (Hogwild proxy: mimics shared-attention KV length)")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    heads = TEMPLATE[:args.sections]
    if args.smart:  # make the LAST section the integrative one that depends on the rest
        heads = TEMPLATE[:args.sections - 1] + ["Reflections and Farewell"]
    prompts = [chunk_ids(tok, h) for h in heads]
    if args.ctxpad:  # front-pad with filler so each slot attends to ~ctxpad keys (speed-only probe)
        filler = tok.encode("In Hawaii the ocean meets the mountains under a bright sky. ",
                            add_special_tokens=False)
        pad = (filler * (args.ctxpad // len(filler) + 1))[:args.ctxpad]
        prompts = [pad + p for p in prompts]

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        for ids in prompts:
            f.write(",".join(str(i) for i in ids) + "\n")
        pf = f.name

    env = dict(os.environ, QWISP_RUN="ghost-smart" if args.smart else "ghost", QWISP_MODEL=args.model,
               QWISP_GHOST_PROMPTS=pf, QWISP_GHOST_GEN=str(args.sec_tokens),
               QWISP_GHOST_DUMP="1")
    r = subprocess.run([args.bin, "stream"], env=env, capture_output=True, text=True)
    os.unlink(pf)
    out = r.stdout + r.stderr

    # speed lines
    for line in out.splitlines():
        if line.strip().startswith("[ghost]") or "tok/s" in line or "→" in line:
            print(line)

    # detokenize dumped sections
    secs = {}
    for m in re.finditer(r"GHOST_TOKENS:(\d+):([\d,]+)", out):
        b = int(m.group(1))
        ids = [int(x) for x in m.group(2).split(",") if x]
        secs[b] = tok.decode(ids, skip_special_tokens=True)
    if secs:
        print("\n### GHOST OUTPUT (qwisp engine, read for coherence)")
        for b in sorted(secs):
            print(f"\n## {heads[b]}\n{secs[b].strip()}")
    else:
        print("\n[no GHOST_TOKENS dumped]\n--- raw output ---\n" + out[:2000])


if __name__ == "__main__":
    main()
