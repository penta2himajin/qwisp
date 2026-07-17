#!/usr/bin/env python3
"""#47 lever Q design-point sweep (probe 17 extension, oracle): K4 ∈ {40, 20, 0} experts
kept at 4-bit per layer (by routing frequency on the experiment prompts), ALL other experts'
switch_mlp weights requantized through a 2-bit round trip (gs=64 affine; the 4-bit gs=64
grid represents 2-bit gs=64 points exactly, so the model computes with true 2-bit tails).
K4=0 is the all-2-bit corner.

Per design point: (a) greedy free-run battery (loops / 8-gram via detlag2.py + rollstab.py),
(b) TF-fidelity — teacher-force the o0 (full-4-bit) token stream through the patched model
and count argmax match %.

Patching is INCREMENTAL (K4 40 → 20 → 0): each step demotes only the newly-tailed rows,
which are still original 4-bit — so one model load serves all points, and the K4=40 weights
are bit-identical to the original probe-17 run (its o1-* free-run outputs are reused as
k40-* when present).

Run: mlx-python mixprec.py <outdir>  (GPU exclusive)
"""
import os, sys, subprocess
import mlx.core as mx
import numpy as np
from mlx_lm import load, stream_generate
from mlx_lm.models.cache import make_prompt_cache

MDL = os.path.expanduser("~/.mtplx/models/Youssofal--Qwen3.6-35B-A3B-MTPLX-Optimized-Speed-FP16")
K4S = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "40,20,0").split(",")]  # descending — incremental demotion
CAL = len(sys.argv) > 3 and sys.argv[3] == "cal"   # calibrated-affine 2-bit (MSE-optimal) instead of min/max
PFX = "ck" if CAL else "k"                          # output tag prefix (ck8-* vs k8-*)


def cal2bit(deq):
    """MSE-optimal affine 2-bit per gs=64 group: alternating code-assignment / least-squares
    (s,b) fit, init = min/max affine (the naive baseline — so result is never worse in MSE).
    Returns dequantized values on the fitted grid. Levels stay equally spaced (b + s*q),
    so the exact 4-bit gs=64 re-encode property is preserved (spacing s -> s4 = span/15
    always divides the level offsets)."""
    shp = deq.shape
    g = deq.reshape(-1, 64).astype(mx.float32)
    mn = g.min(axis=1, keepdims=True)
    mxv = g.max(axis=1, keepdims=True)
    s = (mxv - mn) / 3
    s = mx.where(mx.abs(s) < 1e-8, mx.ones_like(s), s)
    b = mn
    for _ in range(15):
        q = mx.clip(mx.round((g - b) / s), 0, 3)
        qm = q.mean(axis=1, keepdims=True)
        gm = g.mean(axis=1, keepdims=True)
        var = ((q - qm) ** 2).mean(axis=1, keepdims=True)
        cov = ((q - qm) * (g - gm)).mean(axis=1, keepdims=True)
        s = mx.where(var > 1e-12, cov / mx.maximum(var, mx.array(1e-12)), s)
        s = mx.where(mx.abs(s) < 1e-8, mx.ones_like(s), s)
        b = gm - s * qm
    q = mx.clip(mx.round((g - b) / s), 0, 3)
    d2 = (s * q + b).astype(deq.dtype)
    return d2.reshape(shp)
GS = 64
PROMPTS = {
    "story": "Write a short story about a lighthouse keeper who discovers a message in a bottle.",
    "tcp": "Explain how TCP congestion control works in detail, covering slow start, congestion avoidance, fast retransmit, and fast recovery.",
    "qs": "Write a detailed step-by-step explanation of how quicksort works, with a Python implementation.",
    "sky": "Explain why the sky appears blue in plain English, in about three paragraphs.",
}
MAXTOK = {"story": 1000, "tcp": 1000, "qs": 1500, "sky": 1500}

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
model, tok = load(MDL)

# ── locate MoE layers ──
root = getattr(model, "language_model", model)
layers = root.model.layers if hasattr(root, "model") else root.layers
moe = [l.mlp for l in layers if hasattr(l.mlp, "switch_mlp")]
print(f"[mixprec] {len(moe)} MoE layers", flush=True)

# ── routing frequency: replace the gate ATTRIBUTE (the MoE forward looks up self.gate;
# instance __call__ patching would NOT intercept — Python calls type(obj).__call__) ──
counts = [[0] * 256 for _ in moe]
hooked = []
for li, m in enumerate(moe):
    orig = m.gate
    def make(li, orig):
        def call(x):
            y = orig(x)
            k = mx.argpartition(y, -8, axis=-1)[..., -8:]
            for e in k.reshape(-1).tolist():
                counts[li][e] += 1
            return y
        return call
    hooked.append((m, orig))
    m.gate = make(li, orig)

calib = "\n".join(PROMPTS.values())
ids = tok.encode(calib)
_ = root.model(mx.array([ids])) if hasattr(root, "model") else root(mx.array([ids]))
mx.eval(_)
for m, orig in hooked:
    m.gate = orig
tot = sum(counts[0])
assert tot > 0, "gate hook did not fire — routing counts empty"
print(f"[mixprec] calib routing counted (layer0 total={tot})", flush=True)

orders = [sorted(range(256), key=lambda e: (-c[e], e)) for c in counts]


def gen(tag):
    for name, p in PROMPTS.items():
        if os.path.exists(f"{out}/{tag}-{name}.toks"):
            print(f"[mixprec] {tag}-{name}: exists, skip", flush=True)
            continue
        msgs = [{"role": "user", "content": p}]
        prompt = tok.apply_chat_template(msgs, add_generation_prompt=True)
        toks = []
        for r in stream_generate(model, tok, prompt=prompt, max_tokens=MAXTOK[name]):
            toks.append(r.token)
        with open(f"{out}/{tag}-{name}.toks", "w") as f:
            f.write("\n".join(map(str, toks)))
        with open(f"{out}/{tag}-{name}.txt", "w") as f:
            f.write(tok.decode(toks))
        print(f"[mixprec] {tag}-{name}: {len(toks)} toks", flush=True)


def demote(rows_per_layer):
    """2-bit round trip on the given (still original 4-bit) expert rows, per layer."""
    for li, m in enumerate(moe):
        rows = rows_per_layer[li]
        if not rows:
            continue
        tsel = mx.array(rows)
        sw = m.switch_mlp
        for pname in ("gate_proj", "up_proj", "down_proj"):
            lin = getattr(sw, pname)
            w, s, b = lin.weight, lin.scales, lin.biases
            deq = mx.dequantize(mx.take(w, tsel, axis=0), mx.take(s, tsel, axis=0),
                                mx.take(b, tsel, axis=0), group_size=GS, bits=4)
            shp = deq.shape
            if CAL:
                d2 = cal2bit(deq.reshape(-1, shp[-1]))
                if li == 0 and pname == "gate_proj":   # one-time sanity: MSE naive vs calibrated (f32 — f16 squares underflow)
                    q2n = mx.quantize(deq.reshape(-1, shp[-1]), group_size=GS, bits=2)
                    d2n = mx.dequantize(*q2n, group_size=GS, bits=2)
                    df = deq.reshape(-1, shp[-1]).astype(mx.float32)
                    en = ((df - d2n.astype(mx.float32)) ** 2).mean().item()
                    ec = ((df - d2.astype(mx.float32)) ** 2).mean().item()
                    print(f"[mixprec] cal sanity L0 gate: MSE naive={en:.3e} cal={ec:.3e} ({100*(1-ec/max(en,1e-30)):.1f}% lower)", flush=True)
            else:
                q2 = mx.quantize(deq.reshape(-1, shp[-1]), group_size=GS, bits=2)
                d2 = mx.dequantize(*q2, group_size=GS, bits=2)
            rq = mx.quantize(d2, group_size=GS, bits=4)
            nw, ns, nb = (x.reshape(len(rows), shp[1], -1) for x in rq)
            # numpy round trip for row assignment (mx fancy setitem is version-dependent)
            wn, sn, bn = np.array(w), np.array(s), np.array(b)
            wn[rows], sn[rows], bn[rows] = np.array(nw.astype(w.dtype)), np.array(ns.astype(s.dtype)), np.array(nb.astype(b.dtype))
            lin.weight, lin.scales, lin.biases = mx.array(wn), mx.array(sn), mx.array(bn)
        mx.eval(sw.gate_proj.weight, sw.up_proj.weight, sw.down_proj.weight)
        if li % 10 == 0:
            print(f"[mixprec] patched layer {li} (+{len(rows)} rows)", flush=True)


def tf_match(name):
    """Teacher-force o0's token stream through the current model; % argmax match."""
    o0 = [int(x) for x in open(f"{out}/o0-{name}.toks").read().split()]
    msgs = [{"role": "user", "content": PROMPTS[name]}]
    pids = list(tok.apply_chat_template(msgs, add_generation_prompt=True))
    seq = pids + o0
    cache = make_prompt_cache(model)
    preds = []
    for s in range(0, len(seq), 512):
        logits = model(mx.array([seq[s:s + 512]]), cache=cache)
        preds.extend(mx.argmax(logits[0], axis=-1).tolist())
    hit = sum(1 for i in range(len(pids) - 1, len(seq) - 1) if preds[i] == seq[i + 1])
    return hit, len(o0)


gen("o0")  # baseline, unpatched 4-bit (reused from probe 17 when present)

# reuse probe-17 K4=40 free-run outputs (bit-identical patch) as k40-*
for name in PROMPTS:
    for ext in ("toks", "txt"):
        src, dst = f"{out}/o1-{name}.{ext}", f"{out}/k40-{name}.{ext}"
        if os.path.exists(src) and not os.path.exists(dst):
            os.symlink(f"o1-{name}.{ext}", dst)

tf = {}
prev = 256  # demoted so far: ranks [K4, prev) are newly tailed at each step
for k4 in K4S:
    demote([sorted(o[k4:prev]) for o in orders])
    prev = k4
    print(f"[mixprec] === design point K4={k4} (tail {256 - k4} experts at 2-bit, cal={CAL}) ===", flush=True)
    gen(f"{PFX}{k4}")
    tf[k4] = {}
    for name in PROMPTS:
        hit, n = tf_match(name)
        tf[k4][name] = (hit, n)
        print(f"[mixprec] TF {PFX}{k4}-{name}: {hit}/{n} = {100*hit/n:.2f}%", flush=True)

# ── summary: loops (detlag2), 8-gram min (rollstab), TF-match ──
here = os.path.dirname(os.path.abspath(__file__))
for k4 in K4S:
    toks = [f"{out}/{PFX}{k4}-{n}.toks" for n in PROMPTS]
    txts = [f"{out}/{PFX}{k4}-{n}.txt" for n in PROMPTS if os.path.exists(f"{out}/{PFX}{k4}-{n}.txt")]
    print(f"\n=== K4={k4} ===", flush=True)
    subprocess.run([sys.executable, f"{here}/../detlag2.py"] + toks)
    if txts:
        subprocess.run([sys.executable, f"{here}/../rollstab.py"] + txts)
    for name in PROMPTS:
        hit, n = tf[k4][name]
        print(f"TF {name}: {100*hit/n:.2f}%", flush=True)
print("MIXPREC SWEEP DONE", flush=True)
