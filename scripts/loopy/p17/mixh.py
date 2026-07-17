#!/usr/bin/env python3
"""Core-hit share h for the mixed-precision speed roofline (#47 lever Q, notes/18).

h(K4) = fraction of routed (layer, slot) pairs that land in the per-layer top-K4-by-frequency
core set, measured by teacher-forcing the o0 (full-4-bit greedy) streams through the UNPATCHED
model with a gate-counting hook. Two bases:
  h_calib — core picked from the short calib-text forward (cold-start pessimistic)
  h_self  — core picked from the full trace itself (rolling-recalib optimistic)
Weight-byte factor vs all-4-bit: 0.556 + 0.444*h (slot4=1728KiB, slot2=960KiB).

Run: mlx-python mixh.py <p17dir>  (GPU exclusive)
"""
import os, sys
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

MDL = os.path.expanduser("~/.mtplx/models/Youssofal--Qwen3.6-35B-A3B-MTPLX-Optimized-Speed-FP16")
PROMPTS = {
    "story": "Write a short story about a lighthouse keeper who discovers a message in a bottle.",
    "tcp": "Explain how TCP congestion control works in detail, covering slow start, congestion avoidance, fast retransmit, and fast recovery.",
    "qs": "Write a detailed step-by-step explanation of how quicksort works, with a Python implementation.",
    "sky": "Explain why the sky appears blue in plain English, in about three paragraphs.",
}
K4S = [8, 20, 40]

out = sys.argv[1]
model, tok = load(MDL)
root = getattr(model, "language_model", model)
layers = root.model.layers if hasattr(root, "model") else root.layers
moe = [l.mlp for l in layers if hasattr(l.mlp, "switch_mlp")]
nL = len(moe)
print(f"[mixh] {nL} MoE layers", flush=True)

cur = {"counts": None}   # hook target, swappable
for li, m in enumerate(moe):
    orig = m.gate
    def make(li, orig):
        def call(x):
            y = orig(x)
            k = mx.argpartition(y, -8, axis=-1)[..., -8:]
            c = cur["counts"][li]
            for e in k.reshape(-1).tolist():
                c[e] += 1
            return y
        return call
    m.gate = make(li, orig)

def fresh(): return [[0] * 256 for _ in range(nL)]

# calib basis (same text mixprec uses)
calibC = fresh(); cur["counts"] = calibC
ids = tok.encode("\n".join(PROMPTS.values()))
_ = root.model(mx.array([ids])) if hasattr(root, "model") else root(mx.array([ids]))
mx.eval(_)
print(f"[mixh] calib counted (layer0 total={sum(calibC[0])})", flush=True)

# full-trace basis: TF the o0 streams (decode-phase routing only — count from the prompt end)
traceC = fresh()
for name in PROMPTS:
    o0 = [int(x) for x in open(f"{out}/o0-{name}.toks").read().split()]
    pids = list(tok.apply_chat_template([{"role": "user", "content": PROMPTS[name]}],
                                        add_generation_prompt=True))
    # prompt prefill outside the trace counts (separate bucket, discarded)
    junk = fresh(); cur["counts"] = junk
    cache = make_prompt_cache(model)
    _ = model(mx.array([pids]), cache=cache); mx.eval(_)
    cur["counts"] = traceC
    for s in range(0, len(o0), 512):
        _ = model(mx.array([o0[s:s + 512]]), cache=cache); mx.eval(_)
    print(f"[mixh] traced {name} ({len(o0)} tok)", flush=True)

def h(basis, K4):
    """routed-slot share of top-K4(basis) sets, evaluated on traceC."""
    hit = tot = 0
    for li in range(nL):
        core = set(sorted(range(256), key=lambda e: (-basis[li][e], e))[:K4])
        hit += sum(traceC[li][e] for e in core)
        tot += sum(traceC[li])
    return hit / tot

print(f"\n{'K4':>4} {'h_calib':>8} {'h_self':>8} {'bytes_calib':>12} {'bytes_self':>11}")
for k4 in K4S:
    hc, hs = h(calibC, k4), h(traceC, k4)
    print(f"{k4:>4} {hc:8.3f} {hs:8.3f} {0.556+0.444*hc:12.3f} {0.556+0.444*hs:11.3f}", flush=True)
print("MIXH DONE", flush=True)
