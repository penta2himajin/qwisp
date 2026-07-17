#!/usr/bin/env python3
"""#47 lever Q plausibility check (probe 17, oracle): does a 2-bit TAIL reintroduce loops
or wreck quality? Keeps top-K4 experts (by routing frequency on the experiment prompts)
at 4-bit and requantizes ALL other experts' switch_mlp weights through a 2-bit round trip
(gs=64 affine; 4-bit gs=64 grid represents 2-bit gs=64 points exactly, so the re-encode
is lossless and the model computes with true 2-bit-valued tails).

Isolates the PRECISION axis: full coverage (no buddy substitution). The capacity axis
(which subset is resident) is already measured in Swift (C=128 => 4/4 healthy).

Run: mlx-python mixprec.py <outdir>  (GPU exclusive)
"""
import os, sys, json
import mlx.core as mx
from mlx_lm import load, stream_generate

MDL = os.path.expanduser("~/.mtplx/models/Youssofal--Qwen3.6-35B-A3B-MTPLX-Optimized-Speed-FP16")
K4 = 40          # experts kept at 4-bit per layer
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

# ── routing frequency: no hook needed — run each layer's gate directly is fragile; instead
# replace the ATTRIBUTE (the MoE forward looks up self.gate, so an attribute-level wrapper
# intercepts; instance __call__ patching would NOT — Python calls type(obj).__call__).
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


def gen(tag):
    for name, p in PROMPTS.items():
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


gen("o0")  # baseline, unpatched 4-bit

# ── patch: 2-bit round trip on tail experts ──
import numpy as np
for li, m in enumerate(moe):
    c = counts[li]
    order = sorted(range(256), key=lambda e: (-c[e], e))
    tail = sorted(order[K4:])
    tsel = mx.array(tail)
    sw = m.switch_mlp
    for pname in ("gate_proj", "up_proj", "down_proj"):
        lin = getattr(sw, pname)
        w, s, b = lin.weight, lin.scales, lin.biases
        deq = mx.dequantize(mx.take(w, tsel, axis=0), mx.take(s, tsel, axis=0),
                            mx.take(b, tsel, axis=0), group_size=GS, bits=4)
        shp = deq.shape
        q2 = mx.quantize(deq.reshape(-1, shp[-1]), group_size=GS, bits=2)
        d2 = mx.dequantize(*q2, group_size=GS, bits=2)
        rq = mx.quantize(d2, group_size=GS, bits=4)
        nw, ns, nb = (x.reshape(len(tail), shp[1], -1) for x in rq)
        # numpy round trip for row assignment (mx fancy setitem is version-dependent)
        wn, sn, bn = np.array(w), np.array(s), np.array(b)
        wn[tail], sn[tail], bn[tail] = np.array(nw.astype(w.dtype)), np.array(ns.astype(s.dtype)), np.array(nb.astype(b.dtype))
        lin.weight, lin.scales, lin.biases = mx.array(wn), mx.array(sn), mx.array(bn)
    mx.eval(sw.gate_proj.weight, sw.up_proj.weight, sw.down_proj.weight)
    if li % 10 == 0:
        print(f"[mixprec] patched layer {li}", flush=True)
print(f"[mixprec] tail (256-{K4}) experts -> 2-bit in all layers", flush=True)

gen("o1")  # mixed precision
print("MIXPREC DONE", flush=True)
