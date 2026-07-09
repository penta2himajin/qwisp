"""Best-use-case PoC: parallel multi-FILE code generation (NOT ghost's split-one-output).

Distinct from ghost: we don't split one document into fragments (that loses arc). We batch N
INDEPENDENT whole files, each its own full single-pass -> per-file lossless + batching speed.
The only cross-file coupling is the INTERFACE (file B calls file A's methods), which we ground
with a shared contract (real API signatures — a followable constraint, unlike ghost's made-up
scope notes). Measures:
  1. speed: sequential 4 files vs batched 4 files,
  2. lossless: is file[0] batched == file[0] solo (token match)?
  3. consistency: do cross-file calls match defined methods, WITH vs WITHOUT the shared contract?

App: a small CLI task manager (models/storage/cli/test sharing a TaskStore interface).

Run:
  PY="$HOME/Library/Application Support/MTPLX/runtime-venv/bin/python3"
  PYTHONPATH=<repo> "$PY" -m qwisp.ghost_multifile "$HOME/.mtplx/models/unsloth--Qwen3.6-35B-A3B-UD-MLX-3bit"
"""
from __future__ import annotations
import argparse
import re
import time

from mlx_lm import load, generate, batch_generate

APP = "a small command-line task manager"
FILES = {
    "models.py":     "the Task dataclass and an in-memory TaskStore class implementing the contract methods",
    "storage.py":    "save_store(store, path) and load_store(path) serializing TaskStore to/from JSON",
    "cli.py":        "an argparse CLI (subcommands add/list/done) that builds a TaskStore, calls its methods, and uses storage",
    "test_store.py": "pytest tests exercising TaskStore via the contract API",
}
IFACE = ["add", "list", "complete", "save_store", "load_store"]  # ground-truth interface names


def ids(tok, user):
    s = tok.apply_chat_template([{"role": "user", "content": user}],
                                add_generation_prompt=True, enable_thinking=False, tokenize=False)
    return tok.encode(s, add_special_tokens=False)


# ownership: who DEFINES what; everyone else IMPORTS it (the fix for redefinition drift)
OWNS = {
    "models.py":     ("Task, TaskStore", "nothing (this is the base module)"),
    "storage.py":    ("save_store, load_store", "Task and TaskStore via `from models import Task, TaskStore`"),
    "cli.py":        ("the argparse CLI / main()", "TaskStore via `from models import TaskStore`, "
                                                  "save_store/load_store via `from storage import save_store, load_store`"),
    "test_store.py": ("the pytest tests", "TaskStore via `from models import TaskStore`"),
}


def file_prompt(name, desc, contract, owns=False):
    c = (f"\n\nShared interface contract — ALL files MUST use these EXACT names/signatures:\n{contract}\n"
         if contract else "")
    o = ""
    if owns:
        d, i = OWNS[name]
        o = (f"\n\nOwnership (STRICT — this is a multi-file package): this file DEFINES ONLY: {d}. "
             f"IMPORT everything else, do NOT redefine it: {i}.")
    return ids(tok_g, f"You are writing ONE file of {APP}: `{name}` — {desc}.{c}{o}\n"
                      f"Output only the complete Python code for `{name}`.")


def composition(files):
    """does the fileset assemble? want exactly ONE definition of each shared class, and importers."""
    store_defs = sum(len(re.findall(r"class\s+TaskStore\b", c)) for c in files.values())
    task_defs = sum(len(re.findall(r"class\s+Task\b", c)) for c in files.values())
    importers = sum(1 for n, c in files.items()
                    if n != "models.py" and re.search(r"from\s+models\s+import", c))
    return store_defs, task_defs, importers


def defined_names(code):
    return set(re.findall(r"def\s+(\w+)\s*\(", code))


def called_names(code):
    return set(re.findall(r"\.(\w+)\s*\(", code)) | set(re.findall(r"\b(\w+)\s*\(", code))


def consistency(files):
    """fraction of interface methods USED that are actually DEFINED somewhere in the fileset."""
    defined = set().union(*(defined_names(c) for c in files.values()))
    used_iface = set()
    for c in files.values():
        used_iface |= (called_names(c) & set(IFACE))
    if not used_iface:
        return 1.0, defined & set(IFACE), set()
    ok = used_iface & defined
    return len(ok) / len(used_iface), defined & set(IFACE), used_iface - defined


def toklist(tok, s):
    return tok.encode(s)


def main():
    global tok_g
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--file-tokens", type=int, default=320)
    args = ap.parse_args()
    model, tok = load(args.model); tok_g = tok
    names = list(FILES)

    contract = generate(model, tok, ids(tok, f"Design the shared Python interface for {APP}. "
        f"Output ONLY: the Task dataclass fields, and exact method signatures for TaskStore "
        f"(add, list, complete) and storage functions save_store/load_store. Terse, no bodies."),
        max_tokens=200)

    # sequential (baseline): 4 files one at a time, WITH contract
    t = time.perf_counter()
    seq = {n: generate(model, tok, file_prompt(n, FILES[n], contract), max_tokens=args.file_tokens)
           for n in names}
    t_seq = time.perf_counter() - t

    # parallel batch WITH signature contract
    t = time.perf_counter()
    par = dict(zip(names, batch_generate(model, tok, [file_prompt(n, FILES[n], contract) for n in names],
                                         max_tokens=args.file_tokens, verbose=False).texts))
    t_par = time.perf_counter() - t

    # parallel batch WITHOUT contract (drift test)
    nocon = dict(zip(names, batch_generate(model, tok, [file_prompt(n, FILES[n], None) for n in names],
                                           max_tokens=args.file_tokens, verbose=False).texts))

    # parallel batch WITH contract + OWNERSHIP (the composition fix)
    owned = dict(zip(names, batch_generate(model, tok, [file_prompt(n, FILES[n], contract, owns=True) for n in names],
                                           max_tokens=args.file_tokens, verbose=False).texts))

    # lossless: file[0] solo vs in-batch (token match)
    solo0 = generate(model, tok, file_prompt(names[0], FILES[names[0]], contract), max_tokens=args.file_tokens)
    a, b = toklist(tok, solo0), toklist(tok, par[names[0]])
    m = min(len(a), len(b)); match = sum(1 for i in range(m) if a[i] == b[i])
    lossless_rate = match / max(1, m)

    con_par, def_par, miss_par = consistency(par)
    con_noc, def_noc, miss_noc = consistency(nocon)
    ntok = sum(len(toklist(tok, v)) for v in par.values())

    print("=" * 72)
    print(f"MULTI-FILE parallel code-gen (best use case) · {len(names)} files")
    print("=" * 72)
    print(f"speed:   sequential {t_seq:.1f}s   parallel-batch {t_par:.1f}s   -> {t_seq/t_par:.2f}x")
    print(f"lossless: file[0] solo-vs-batched token match = {lossless_rate:.1%} ({match}/{m})")
    print("composition (want: 1 TaskStore def, 1 Task def, 3 importers-of-models):")
    for tag, fs in [("no-contract", nocon), ("signature-contract", par), ("ownership-contract", owned)]:
        sd, td, imp = composition(fs)
        ok = "✓ assembles" if (sd == 1 and imp >= 2) else "✗ redefines"
        print(f"   {tag:20s}: TaskStore defs={sd}  Task defs={td}  models-importers={imp}  {ok}")
    print("=" * 72)
    print("\n### CONTRACT\n" + contract.strip()[:400])
    for n in names:
        print(f"\n### {n} (ownership-contract)\n" + owned[n].strip()[:600])


if __name__ == "__main__":
    main()
