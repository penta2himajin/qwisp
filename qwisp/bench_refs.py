"""Generate the Qwisp measurement ref set (code/agentic/longctx/shortnl) in ONE model load.

Reproducible: prompts live in qwisp/bench_prompts.py (committed). Generated refs are written to
<repo>/refs/<regime>.safetensors (gitignored — regenerate with this script, do not commit).

Each ref stores {spec_prompt, spec_greedy}. This is what the Swift runners (suffix-spec / bolt)
consume via QWISP_MTP_REF.

spec_greedy provenance: the Python-4bit greedy written by the generate step is a BOOTSTRAP value
only — Python runs f16 SDPA/conv while the Swift engine (and mlx-fidelity's teacher forcing) is
f32-full, so near-tie argmax flips make strict fidelity read <100% against it (observed: longctx
120/128, shortnl 126/128, all benign rank<=2 near-ties). The canonical reference is the Swift
strict f32-full greedy; after generating, replace spec_greedy via --ingest-swift with a dump from
either runner (both are the exact same computation, lossless-proven):
  QWISP_RUN=suffix-spec QWISP_DUMP_TOKENS=1 ... qwisp-poc stream | grep OUT_TOKENS  > /tmp/code.toks
  (or QWISP_RUN=bolt QWISP_SWIFT_REF=1 QWISP_DUMP_TOKENS=1 ... | grep STRICT_TOKENS)
  PYTHONPATH=<repo> "$PY" -m qwisp.bench_refs --ingest-swift code /tmp/code.toks

Run (MTPLX runtime venv has mlx_lm):
  PY="$HOME/Library/Application Support/MTPLX/runtime-venv/bin/python3"
  PYTHONPATH=<repo> "$PY" -m qwisp.bench_refs <model_dir> [--nspec 128]
"""
from __future__ import annotations
import argparse
import os

import mlx.core as mx
from mlx_lm import load
import qwisp.mtp_decode as MD
from qwisp.bench_prompts import PROMPTS

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT = os.path.join(REPO_ROOT, "refs")


def build_ids(tok, text: str, ctx):
    ids = tok.encode(text)
    if ctx is None:
        return ids
    if len(ids) < ctx:                      # short regime: pad by repetition
        base = list(ids)
        while len(ids) < ctx:
            ids = ids + base
    return ids[:ctx]


def ingest_swift(pairs, out_dir: str, nspec: int):
    """Replace spec_greedy in refs/<regime>.safetensors with a Swift strict-greedy token dump.

    Each CSV file is the runner's stdout (or a slice of it); the token line may carry an
    OUT_TOKENS:/STRICT_TOKENS: prefix or be a bare comma-separated int list. No model load.
    """
    for regime, csv_path in pairs:
        path = os.path.join(out_dir, f"{regime}.safetensors")
        cur = dict(mx.load(path))
        mx.eval(*cur.values())  # materialize: mx.load is mmap-lazy; saving over the same
        # path before eval zeroes the untouched tensors (spec_prompt corruption bug)
        toks = None
        for line in open(csv_path).read().strip().splitlines():
            line = line.strip()
            for pfx in ("STRICT_TOKENS:", "OUT_TOKENS:"):
                if line.startswith(pfx):
                    line = line[len(pfx):]
                    break
            else:
                if not (line and all(c.isdigit() or c == "," for c in line)):
                    continue
            toks = [int(t) for t in line.split(",") if t]
        if not toks:
            raise SystemExit(f"[bench-ref] ingest {regime}: no token csv found in {csv_path}")
        if len(toks) < nspec:
            raise SystemExit(f"[bench-ref] ingest {regime}: {len(toks)} tokens < nspec {nspec}")
        cur["spec_greedy"] = mx.array(toks[:nspec], mx.int32)
        tmp = path + ".tmp.safetensors"
        mx.save_safetensors(tmp, cur)
        os.replace(tmp, path)
        print(f"[bench-ref] ingest {regime:8s}: spec_greedy <- Swift strict ({len(toks)} toks, kept {nspec}) -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?")
    ap.add_argument("--nspec", type=int, default=128, help="greedy tokens per ref (measurement horizon)")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--ingest-swift", nargs=2, action="append", metavar=("REGIME", "CSV_FILE"), default=None,
                    help="replace spec_greedy in refs/<REGIME>.safetensors with Swift strict tokens (repeatable; no model load)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.ingest_swift:
        ingest_swift(args.ingest_swift, args.out, args.nspec)
        return
    if not args.model:
        raise SystemExit("model dir required (or use --ingest-swift)")

    model, tok = load(args.model)
    lm = model.language_model

    for name, spec in PROMPTS.items():
        ids = build_ids(tok, spec["text"], spec["ctx"])
        g_out, _ = MD.greedy(lm, ids, args.nspec)
        path = os.path.join(args.out, f"{name}.safetensors")
        mx.save_safetensors(path, {
            "spec_prompt": mx.array(ids, mx.int32),
            "spec_greedy": mx.array(g_out, mx.int32),
        })
        print(f"[bench-ref] {name:8s} ctx={len(ids):5d} nspec={args.nspec} "
              f"src={spec['source']} -> {path}")
    print(f"[bench-ref] done: {len(PROMPTS)} refs in {args.out}")


if __name__ == "__main__":
    main()
