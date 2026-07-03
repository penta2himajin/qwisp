#!/usr/bin/env python3
"""
diag_lmhead2.py — Quantify Cauchy-Schwarz looseness for 2-bit lm_head margin-cert.

Hypothesis: rowE[v]·||h||₂  is ~√K≈45× looser than actual |⟨e_v,h⟩|,
so the cert condition (notes/05 §3.2) can never clear.

Outputs:
 - Distribution stats for margin4, rowE2/3, ||h||, bound2/3, actualErr2/3, looseness2/3
 - certWouldFire rate with C-S bound (exact over top-50 challengers)
 - argmaxMatch rate for 2-bit and 3-bit (near-lossless "Track B" viability)
 - Hypothetical cert rate if bound = k × actualErr  (k∈{2,5,10})
 - Same analysis repeated for 3-bit

Usage:
  PY="$HOME/Library/Application Support/MTPLX/runtime-venv/bin/python3"
  PYTHONPATH=<repo> "$PY" qwisp/diag_lmhead2.py
"""

from __future__ import annotations
import os, sys, random, time
import numpy as np

import mlx.core as mx
from mlx_lm import load

# ── paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
from qwisp.bench_prompts import PROMPTS   # noqa: E402

MODEL_DIR = os.path.expanduser(
    "~/.mtplx/models/Youssofal--Qwen3.6-35B-A3B-MTPLX-Optimized-Speed-FP16"
)
SHARD4 = os.path.join(MODEL_DIR, "model-00004-of-00004.safetensors")

# ── tunables ──────────────────────────────────────────────────────────────────
NGEN            = 16        # decode tokens per prompt (4 prompts → ~68 captures total)
SAMPLE_N        = 20_000    # rows for rowE distribution stats
CHUNK           = 4_096     # rows per build chunk  (< 16 MB per matrix)
EPSILON         = 0.05      # safety slack per spec §3.2
TOP_K_CHALL     = 50        # challenger candidates for cert (top-K by logitsN)
K_HYPO          = [2, 5, 10]  # tightening factors for hypothetical cert

random.seed(42)
np.random.seed(42)

START_TIME = time.time()

# ── helpers ───────────────────────────────────────────────────────────────────
def _pct(arr, p): return float(np.percentile(arr, p))

def _dist(arr, label):
    return (f"  {label:<28s}  "
            f"mean={np.mean(arr):8.4f}  median={_pct(arr,50):8.4f}  "
            f"p95={_pct(arr,95):8.4f}  min={np.min(arr):8.4f}  max={np.max(arr):8.4f}")


# ── Step 1 — load raw 4-bit lm_head tensors ───────────────────────────────────
print("=" * 80)
print("Step 1: loading lm_head tensors from shard 4")
_t = time.time()
raw = mx.load(SHARD4)
w4 = raw["language_model.lm_head.weight"]    # (V, H//8) uint32
s4 = raw["language_model.lm_head.scales"]    # (V, 32)   f16
b4 = raw["language_model.lm_head.biases"]    # (V, 32)   f16
mx.eval(w4, s4, b4)
V, Ww = w4.shape
H = Ww * 8   # bits=4 → 8 values per uint32
print(f"  lm_head: V={V}  H={H}  w4={w4.shape}/{w4.dtype}  "
      f"s4={s4.shape}/{s4.dtype}  ({time.time()-_t:.1f}s)")


# ── Step 2 — build 2-bit and 3-bit re-quantizations + rowE sample ─────────────
def build_requantized(w4_, s4_, b4_, bits, label):
    """
    Re-quantize 4-bit lm_head weights to `bits` bits by:
      1. dequant 4-bit → f32
      2. quantize f32 → bits
    Returns (w_alt, s_alt, b_alt, rowE_sample_np, sample_rows_list).
    rowE_sample: ||dqAlt[v] - dq4[v]||₂  for SAMPLE_N random rows.
    """
    sample_rows = sorted(random.sample(range(V), SAMPLE_N))
    sample_set  = set(sample_rows)
    rowE_sample = np.zeros(SAMPLE_N, dtype=np.float32)
    ptr         = {r: i for i, r in enumerate(sample_rows)}

    w_chunks, s_chunks, b_chunks = [], [], []
    n_chunks = (V + CHUNK - 1) // CHUNK
    _t = time.time()
    print(f"\nStep 2: building {label} ({bits}-bit, {n_chunks} chunks)...", flush=True)

    for ci, start in enumerate(range(0, V, CHUNK)):
        end = min(start + CHUNK, V)

        # dequantize 4-bit chunk → f32
        dq4c = mx.dequantize(
            w4_[start:end], s4_[start:end], b4_[start:end], group_size=64, bits=4
        ).astype(mx.float32)
        mx.eval(dq4c)

        # re-quantize to target bits
        wc, sc, bc = mx.quantize(dq4c, group_size=64, bits=bits)

        # dequantize back → row-wise L2 error
        dqc   = mx.dequantize(wc, sc, bc, group_size=64, bits=bits).astype(mx.float32)
        err   = dqc - dq4c
        rowEc = mx.sqrt(mx.sum(err ** 2, axis=-1))
        mx.eval(wc, sc, bc, rowEc)

        w_chunks.append(wc)
        s_chunks.append(sc)
        b_chunks.append(bc)

        rowEc_np = np.array(rowEc)
        for r in range(start, end):
            if r in sample_set:
                rowE_sample[ptr[r]] = rowEc_np[r - start]

        if (ci + 1) % 10 == 0 or ci == n_chunks - 1:
            print(f"  chunk {ci+1}/{n_chunks}  ({time.time()-_t:.1f}s)", flush=True)

    w_all = mx.concatenate(w_chunks, axis=0)
    s_all = mx.concatenate(s_chunks, axis=0)
    b_all = mx.concatenate(b_chunks, axis=0)
    mx.eval(w_all, s_all, b_all)
    print(f"  {label} done in {time.time()-_t:.1f}s  "
          f"w={w_all.shape}/{w_all.dtype}  s={s_all.shape}/{s_all.dtype}")
    return w_all, s_all, b_all, rowE_sample, sample_rows


w2, s2, b2, rowE2_sample, _srows2 = build_requantized(w4, s4, b4, 2, "2-bit")
w3, s3, b3, rowE3_sample, _srows3 = build_requantized(w4, s4, b4, 3, "3-bit")

# pre-convert scales/biases to f16 for quantized_matmul
s4_f16 = s4                         # already f16
b4_f16 = b4
s2_f16 = s2.astype(mx.float16)
b2_f16 = b2.astype(mx.float16)
s3_f16 = s3.astype(mx.float16)
b3_f16 = b3.astype(mx.float16)
mx.eval(s2_f16, b2_f16, s3_f16, b3_f16)


# ── Step 3 — load model and capture hidden vectors ─────────────────────────────
print("\n" + "=" * 80)
print("Step 3: loading model and capturing lm_head inputs...")
_t = time.time()
model, tok = load(MODEL_DIR)
lm = model.language_model
print(f"  model loaded in {time.time()-_t:.1f}s")


def greedy_capture(lm_, prompt_ids, max_tokens):
    """
    Greedy decode with hidden-state capture.
    Returns (token_list, hiddens_list).
    hiddens_list[i] = np.ndarray([H], float32) — final-norm output fed to lm_head.
      index 0  = last position of prefill
      index 1..max_tokens-1 = each decode step
    """
    kv       = lm_.make_cache()
    hiddens  = []

    # prefill
    p  = mx.array(prompt_ids)[None]
    h  = lm_.model(p, cache=kv)         # [1, L, H]
    hl = h[0, -1].astype(mx.float32)    # [H]
    mx.eval(hl)
    hiddens.append(np.array(hl, copy=True))

    lg = lm_.lm_head(h)
    y  = mx.argmax(lg[:, -1:], axis=-1)
    mx.eval(y)
    out = [int(y.item())]

    # decode
    for _ in range(max_tokens - 1):
        h  = lm_.model(y, cache=kv)         # [1, 1, H]
        hv = h[0, 0].astype(mx.float32)     # [H]
        mx.eval(hv)
        hiddens.append(np.array(hv, copy=True))

        lg = lm_.lm_head(h)
        y  = mx.argmax(lg[:, -1:], axis=-1)
        mx.eval(y)
        out.append(int(y.item()))

    return out, hiddens


all_hiddens: list[np.ndarray] = []
print(f"  Prompts: {list(PROMPTS.keys())},  NGEN={NGEN}")
for name, spec in PROMPTS.items():
    ids = tok.encode(spec["text"])
    if spec["ctx"]:
        ids = ids[:spec["ctx"]]
    _t = time.time()
    _, h_list = greedy_capture(lm, ids, NGEN)
    all_hiddens.extend(h_list)
    print(f"  [{name}] ctx={len(ids)}  captured={len(h_list)}  "
          f"total={len(all_hiddens)}  ({time.time()-_t:.1f}s)")

N_TOK = len(all_hiddens)
print(f"  Total hidden vectors: {N_TOK}")


# ── Step 4 — per-token analysis ───────────────────────────────────────────────
print("\n" + "=" * 80)
print(f"Step 4: per-token analysis ({N_TOK} tokens)...")

margin4_arr     = np.zeros(N_TOK, np.float32)
h_norm_arr      = np.zeros(N_TOK, np.float32)

rowE2_vs_arr    = np.zeros(N_TOK, np.float32)
bound2_arr      = np.zeros(N_TOK, np.float32)
actual_err2_arr = np.zeros(N_TOK, np.float32)
looseness2_arr  = np.zeros(N_TOK, np.float32)
cert2_fire_arr  = np.zeros(N_TOK, bool)
match2_arr      = np.zeros(N_TOK, bool)
hypo2_arr       = {k: np.zeros(N_TOK, bool) for k in K_HYPO}

rowE3_vs_arr    = np.zeros(N_TOK, np.float32)
bound3_arr      = np.zeros(N_TOK, np.float32)
actual_err3_arr = np.zeros(N_TOK, np.float32)
looseness3_arr  = np.zeros(N_TOK, np.float32)
cert3_fire_arr  = np.zeros(N_TOK, bool)
match3_arr      = np.zeros(N_TOK, bool)
hypo3_arr       = {k: np.zeros(N_TOK, bool) for k in K_HYPO}

# 20k row subsample for max-actualErr robustness stat (reuse full logits)
rob_rows  = np.random.choice(V, size=20_000, replace=False)
max_act2_20k = np.zeros(N_TOK, np.float32)
max_act3_20k = np.zeros(N_TOK, np.float32)

_loop_t = time.time()

for ti, h_np in enumerate(all_hiddens):
    h_norm = float(np.sqrt(np.sum(h_np.astype(np.float64) ** 2)))
    h_norm_arr[ti] = h_norm
    h_f16 = mx.array(h_np, dtype=mx.float16)[None]   # [1, H]

    # ── Full logits via quantized_matmul (f16 ops) ────────────────────────────
    l4_mx = mx.quantized_matmul(h_f16, w4, scales=s4_f16, biases=b4_f16,
                                 bits=4, group_size=64)[0]   # [V] f16
    l2_mx = mx.quantized_matmul(h_f16, w2, scales=s2_f16, biases=b2_f16,
                                 bits=2, group_size=64)[0]
    l3_mx = mx.quantized_matmul(h_f16, w3, scales=s3_f16, biases=b3_f16,
                                 bits=3, group_size=64)[0]
    mx.eval(l4_mx, l2_mx, l3_mx)

    l4 = np.array(l4_mx.astype(mx.float32))
    l2 = np.array(l2_mx.astype(mx.float32))
    l3 = np.array(l3_mx.astype(mx.float32))

    # ── 4-bit reference ───────────────────────────────────────────────────────
    vs_4    = int(np.argmax(l4))
    top2_4  = np.partition(l4, -2)[-2:]; top2_4.sort()
    margin4 = float(top2_4[-1] - top2_4[-2])
    margin4_arr[ti] = margin4

    # ── Robustness: max |l2-l4| and |l3-l4| over 20k subsample ──────────────
    # (reuse already-computed full logits — just index into them)
    max_act2_20k[ti] = float(np.max(np.abs(l2[rob_rows] - l4[rob_rows])))
    max_act3_20k[ti] = float(np.max(np.abs(l3[rob_rows] - l4[rob_rows])))

    # ─────────────────────────────────────────────────────────────────────────
    # 2-bit analysis
    # ─────────────────────────────────────────────────────────────────────────
    vs_2           = int(np.argmax(l2))
    match2_arr[ti] = (vs_2 == vs_4)
    actual_err2    = abs(float(l2[vs_2]) - float(l4[vs_2]))
    actual_err2_arr[ti] = actual_err2

    # Exact rowE for v* row
    dq4_v2 = mx.dequantize(w4[vs_2:vs_2+1], s4[vs_2:vs_2+1], b4[vs_2:vs_2+1],
                            group_size=64, bits=4).astype(mx.float32)
    dq2_v  = mx.dequantize(w2[vs_2:vs_2+1], s2[vs_2:vs_2+1], b2[vs_2:vs_2+1],
                            group_size=64, bits=2).astype(mx.float32)
    rE2_vs = float(mx.sqrt(mx.sum((dq2_v - dq4_v2) ** 2)).item())
    mx.eval()

    bound2 = rE2_vs * h_norm + EPSILON
    rowE2_vs_arr[ti]  = rE2_vs
    bound2_arr[ti]    = bound2
    looseness2_arr[ti] = bound2 / (actual_err2 + 1e-9)

    # Cert check: top-TOP_K_CHALL by l2, compute exact rowE for each candidate
    top_k2   = np.argpartition(l2, -TOP_K_CHALL)[-TOP_K_CHALL:]
    tk2_mx   = mx.array(top_k2.tolist())
    dq4_tk2  = mx.dequantize(w4[tk2_mx], s4[tk2_mx], b4[tk2_mx],
                              group_size=64, bits=4).astype(mx.float32)
    dq2_tk2  = mx.dequantize(w2[tk2_mx], s2[tk2_mx], b2[tk2_mx],
                              group_size=64, bits=2).astype(mx.float32)
    rE2_tk2  = np.array(mx.sqrt(mx.sum((dq2_tk2 - dq4_tk2) ** 2, axis=-1)))
    mx.eval()

    ub2   = l2[top_k2] + rE2_tk2 * h_norm + EPSILON
    mask2 = (top_k2 != vs_2)
    max_ub2 = float(np.max(ub2[mask2])) if mask2.any() else -1e30
    lb2     = float(l2[vs_2]) - rE2_vs * h_norm - EPSILON
    cert2_fire_arr[ti] = (lb2 > max_ub2)

    # Hypothetical cert: bound = k × actualErr
    l2_excl         = l2.copy(); l2_excl[vs_2] = -np.inf
    vs_2nd          = int(np.argmax(l2_excl))
    actual_err2_2nd = abs(float(l2[vs_2nd]) - float(l4[vs_2nd]))
    for k in K_HYPO:
        hypo2_arr[k][ti] = (
            float(l2[vs_2]) - k * actual_err2    - EPSILON >
            float(l2[vs_2nd]) + k * actual_err2_2nd + EPSILON
        )

    # ─────────────────────────────────────────────────────────────────────────
    # 3-bit analysis
    # ─────────────────────────────────────────────────────────────────────────
    vs_3           = int(np.argmax(l3))
    match3_arr[ti] = (vs_3 == vs_4)
    actual_err3    = abs(float(l3[vs_3]) - float(l4[vs_3]))
    actual_err3_arr[ti] = actual_err3

    dq4_v3 = mx.dequantize(w4[vs_3:vs_3+1], s4[vs_3:vs_3+1], b4[vs_3:vs_3+1],
                            group_size=64, bits=4).astype(mx.float32)
    dq3_v  = mx.dequantize(w3[vs_3:vs_3+1], s3[vs_3:vs_3+1], b3[vs_3:vs_3+1],
                            group_size=64, bits=3).astype(mx.float32)
    rE3_vs = float(mx.sqrt(mx.sum((dq3_v - dq4_v3) ** 2)).item())
    mx.eval()

    bound3 = rE3_vs * h_norm + EPSILON
    rowE3_vs_arr[ti]   = rE3_vs
    bound3_arr[ti]     = bound3
    looseness3_arr[ti] = bound3 / (actual_err3 + 1e-9)

    top_k3   = np.argpartition(l3, -TOP_K_CHALL)[-TOP_K_CHALL:]
    tk3_mx   = mx.array(top_k3.tolist())
    dq4_tk3  = mx.dequantize(w4[tk3_mx], s4[tk3_mx], b4[tk3_mx],
                              group_size=64, bits=4).astype(mx.float32)
    dq3_tk3  = mx.dequantize(w3[tk3_mx], s3[tk3_mx], b3[tk3_mx],
                              group_size=64, bits=3).astype(mx.float32)
    rE3_tk3  = np.array(mx.sqrt(mx.sum((dq3_tk3 - dq4_tk3) ** 2, axis=-1)))
    mx.eval()

    ub3   = l3[top_k3] + rE3_tk3 * h_norm + EPSILON
    mask3 = (top_k3 != vs_3)
    max_ub3 = float(np.max(ub3[mask3])) if mask3.any() else -1e30
    lb3     = float(l3[vs_3]) - rE3_vs * h_norm - EPSILON
    cert3_fire_arr[ti] = (lb3 > max_ub3)

    l3_excl         = l3.copy(); l3_excl[vs_3] = -np.inf
    vs_3nd          = int(np.argmax(l3_excl))
    actual_err3_3nd = abs(float(l3[vs_3nd]) - float(l4[vs_3nd]))
    for k in K_HYPO:
        hypo3_arr[k][ti] = (
            float(l3[vs_3]) - k * actual_err3    - EPSILON >
            float(l3[vs_3nd]) + k * actual_err3_3nd + EPSILON
        )

    if (ti + 1) % 10 == 0 or ti == N_TOK - 1:
        print(f"  token {ti+1}/{N_TOK}  ({time.time()-_loop_t:.1f}s)  "
              f"match2={match2_arr[:ti+1].mean()*100:.1f}%  "
              f"match3={match3_arr[:ti+1].mean()*100:.1f}%  "
              f"loose2={looseness2_arr[:ti+1].mean():.1f}x", flush=True)


# ── Step 5 — summary table ───────────────────────────────────────────────────
print("\n" + "=" * 80)
print("SUMMARY TABLE")
print("=" * 80)

print(f"\n── Input statistics ({N_TOK} tokens, H={H}, V={V}) ──")
print(_dist(h_norm_arr,  "||h||₂"))
print(_dist(margin4_arr, "margin4 (l4 top1−top2)"))

print(f"\n── 2-bit Cauchy-Schwarz bound analysis ──")
print(_dist(rowE2_sample,    "rowE2  (20k-row sample)"))
print(_dist(rowE2_vs_arr,    "rowE2[v*]  (exact, per tok)"))
print(_dist(bound2_arr,      "bound2 = rowE2[v*]·||h||+ε"))
print(_dist(actual_err2_arr, "actualErr2 = |l2[v*]−l4[v*]|"))
print(_dist(max_act2_20k,    "max|l2−l4| over 20k rows"))
print(_dist(looseness2_arr,  "looseness2 = bound2/actualErr2"))

print(f"\n── 2-bit cert and argmax ──")
cert2_rate  = float(cert2_fire_arr.mean())
match2_rate = float(match2_arr.mean())
print(f"  certWouldFire2 (C-S, top-{TOP_K_CHALL} challengers) : "
      f"{cert2_fire_arr.sum()}/{N_TOK} = {cert2_rate*100:.1f}%")
print(f"  argmaxMatch2   (plain 2-bit, near-lossless)         : "
      f"{match2_arr.sum()}/{N_TOK} = {match2_rate*100:.1f}%")

print(f"\n── 2-bit hypothetical cert rates (tighter bound = k × actualErr) ──")
for k in K_HYPO:
    hr = float(hypo2_arr[k].mean())
    print(f"  k={k:2d}  (bound = {k}×actualErr)  :  "
          f"{hypo2_arr[k].sum()}/{N_TOK} = {hr*100:.1f}%")

print(f"\n── 3-bit Cauchy-Schwarz bound analysis ──")
print(_dist(rowE3_sample,    "rowE3  (20k-row sample)"))
print(_dist(rowE3_vs_arr,    "rowE3[v*]  (exact, per tok)"))
print(_dist(bound3_arr,      "bound3 = rowE3[v*]·||h||+ε"))
print(_dist(actual_err3_arr, "actualErr3 = |l3[v*]−l4[v*]|"))
print(_dist(max_act3_20k,    "max|l3−l4| over 20k rows"))
print(_dist(looseness3_arr,  "looseness3 = bound3/actualErr3"))

print(f"\n── 3-bit cert and argmax ──")
cert3_rate  = float(cert3_fire_arr.mean())
match3_rate = float(match3_arr.mean())
print(f"  certWouldFire3 (C-S, top-{TOP_K_CHALL} challengers) : "
      f"{cert3_fire_arr.sum()}/{N_TOK} = {cert3_rate*100:.1f}%")
print(f"  argmaxMatch3   (plain 3-bit, near-lossless)         : "
      f"{match3_arr.sum()}/{N_TOK} = {match3_rate*100:.1f}%")

print(f"\n── 3-bit hypothetical cert rates (tighter bound = k × actualErr) ──")
for k in K_HYPO:
    hr = float(hypo3_arr[k].mean())
    print(f"  k={k:2d}  (bound = {k}×actualErr)  :  "
          f"{hypo3_arr[k].sum()}/{N_TOK} = {hr*100:.1f}%")

print(f"\n── Key ratio summary ──")
cs_slack_theory = float(np.sqrt(H))
print(f"  H={H}   √H (theoretical iid C-S slack) = {cs_slack_theory:.1f}")
print(f"  mean ||h||₂                            = {h_norm_arr.mean():.3f}")
print(f"  mean ||h||₂ / √H                       = {h_norm_arr.mean()/np.sqrt(H):.3f}  "
      f"(≈1 if h is ~unit-norm per element)")
print(f"")
print(f"  2-bit:  mean rowE2[v*]·||h||           = "
      f"{(rowE2_vs_arr * h_norm_arr).mean():.4f}  (C-S bound without ε)")
print(f"  2-bit:  mean actualErr2[v*]             = {actual_err2_arr.mean():.4f}")
print(f"  2-bit:  mean looseness2                 = {looseness2_arr.mean():.1f}×")
print(f"  2-bit:  min  looseness2                 = {looseness2_arr.min():.1f}×")
print(f"")
print(f"  3-bit:  mean rowE3[v*]·||h||           = "
      f"{(rowE3_vs_arr * h_norm_arr).mean():.4f}  (C-S bound without ε)")
print(f"  3-bit:  mean actualErr3[v*]             = {actual_err3_arr.mean():.4f}")
print(f"  3-bit:  mean looseness3                 = {looseness3_arr.mean():.1f}×")
print(f"  3-bit:  min  looseness3                 = {looseness3_arr.min():.1f}×")

print("\n" + "=" * 80)
print(f"Total wall time: {time.time() - START_TIME:.0f}s")
print("=" * 80)
