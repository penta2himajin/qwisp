"""Lever-② offline cross-projection residual measurement (span corrector go/no-go).

For each layer, model the bolt C=64 residency (top-64 experts by prompt-region gate
mass, frozen for the gen region) and measure, for every expert that MISSES during the
128-token gen region, how well its output is reconstructed from the resident span:

    f_e(x)  ≈  Σ_b c_{e,b}·f_b(x) + β_e·x + v_e      (gate_fold corrector form)

Corners reported: novice (v only) / affine novice (β,v) / best single buddy /
span top-j (j=2,4,8,16,64). Fit = ridge on prompt-region activations (rolling-recalib
analog), eval = held-out gen-region miss tokens, weighted by miss gate mass.

Key outputs per regime × layer-group:
  - missMass/obs + top1Share by MASS (checks the "13% by count but ~30% by mass" risk)
  - mass-weighted relative residual ||f_e - pred||²/||f_e||² per corrector corner

Run (MTPLX runtime venv):
  PY="$HOME/Library/Application Support/MTPLX/runtime-venv/bin/python3"
  PYTHONPATH=<repo> "$PY" -m qwisp.span_residual <model_dir> [--regimes code,agentic,...]
      [--c 64] [--ctx-keep 512] [--ridge 1e-3] [--json out.json]
"""
from __future__ import annotations
import argparse
import json
import os
import sys

import mlx.core as mx
import numpy as np
from mlx_lm import load

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REFS_DIR = os.path.join(REPO_ROOT, "refs")
REGIMES = ("code", "agentic", "longctx", "shortnl")
SPAN_JS = (2, 4, 8, 16, 64)


def find_moe_layers(model):
    """[(layer_idx, moe_block)] for every decoder layer with a switch_mlp."""
    layers = model.language_model.model.layers if hasattr(model, "language_model") \
        else model.model.layers
    out = []
    for i, lyr in enumerate(layers):
        if hasattr(lyr.mlp, "switch_mlp"):
            out.append((i, lyr.mlp))
    return out


def capture_activations(model, moe, ids, ctx_keep):
    """One teacher-forced forward; returns per-layer MoE input x [Tkeep, d] (f16).

    Tkeep = min(len(ids), ctx_keep + 128): last ctx_keep prompt positions + gen region.
    """
    store = {}
    by_id = {id(blk): li for li, blk in moe}
    cls = type(moe[0][1])
    orig = cls.__call__  # instance __call__ assignment is ignored by type lookup

    def hooked(self, x):
        li = by_id.get(id(self))
        if li is not None:
            store[li] = x[0].astype(mx.float16)  # [T, d]
        return orig(self, x)

    cls.__call__ = hooked
    try:
        logits = model(ids[None])
        mx.eval(logits, *store.values())
    finally:
        cls.__call__ = orig
    T = ids.shape[0]
    keep = min(T, ctx_keep + 128)
    return {li: x[T - keep:] for li, x in store.items()}


def route(blk, x):
    """Replicate Qwen3NextSparseMoeBlock routing. x [T,d] f32 → inds [T,8], scores [T,8]."""
    gates = mx.softmax(blk.gate(x), axis=-1, precise=True)
    k = blk.top_k
    inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
    scores = mx.take_along_axis(gates, inds, axis=-1)
    if blk.norm_topk_prob:
        scores = scores / scores.sum(axis=-1, keepdims=True)
    return np.array(inds), np.array(scores.astype(mx.float32))


def expert_outputs(blk, x, expert_ids):
    """f_e(x) for each e in expert_ids. x [T,d] → [T, E, d] f32 (numpy)."""
    inds = mx.broadcast_to(mx.array(expert_ids, dtype=mx.int32)[None, :],
                           (x.shape[0], len(expert_ids)))
    y = blk.switch_mlp(x.astype(mx.float16), inds)
    mx.eval(y)
    return np.array(y.astype(mx.float32))


def fit_layer(blk, x_f16, gen_len, C, ridge, fit_window="prompt"):
    """Measure one layer. Returns dict of stats or None if no misses.

    fit_window: "prompt" = fit on prompt region (prefill-frozen analog);
                "genhalf" = fit on first half of gen region (rolling-recalib analog,
                eval restricted to misses in the second half).
    """
    x = x_f16.astype(mx.float32)
    T = x.shape[0]
    n_prompt = T - gen_len
    inds, scores = route(blk, x)

    # residency: top-C by prompt-region gate mass, frozen for gen region
    nE = blk.num_experts
    mass = np.zeros(nE)
    np.add.at(mass, inds[:n_prompt].ravel(), scores[:n_prompt].ravel())
    resident = np.sort(np.argsort(mass)[-C:])
    res_set = set(resident.tolist())

    # miss events in gen region: (t, e, w, rank) — rank by score within the token's top-8
    g_inds, g_scores = inds[n_prompt:], scores[n_prompt:]
    order = np.argsort(-g_scores, axis=1)          # rank 0 = largest weight
    miss = []                                       # (t_global, e, w, rank)
    for t in range(gen_len):
        for r in range(g_inds.shape[1]):
            j = order[t, r]
            e = int(g_inds[t, j])
            if e not in res_set:
                miss.append((n_prompt + t, e, float(g_scores[t, j]), r))
    n_obs = gen_len
    stats = {
        "missMass_obs": sum(w for _, _, w, _ in miss) / n_obs,
        "top1Share_count": (sum(1 for m in miss if m[3] == 0) / len(miss)) if miss else 0.0,
        "top1Share_mass": (sum(w for _, _, w, r in miss if r == 0)
                           / max(1e-9, sum(w for _, _, w, _ in miss))) if miss else 0.0,
        "nMiss": len(miss),
    }
    if not miss:
        return stats

    if fit_window == "genhalf":
        half = n_prompt + gen_len // 2
        miss = [m for m in miss if m[0] >= half]
        fit_pos = np.arange(n_prompt, half)
        if not miss:
            stats["nMiss"] = 0
            return stats
    else:
        # fit on prompt region (subsample to <=256 positions)
        fit_pos = np.linspace(0, n_prompt - 1, min(256, n_prompt)).astype(int)
    miss_experts = sorted({e for _, e, _, _ in miss})
    all_pos = np.concatenate([fit_pos, np.arange(n_prompt, T)])
    x_np = np.array(x)[all_pos]                     # [P, d]
    nF = len(fit_pos)
    x_sel = x[mx.array(all_pos.astype(np.int32))]

    F = expert_outputs(blk, x_sel, resident.tolist())      # [P, C, d] resident features
    Y = expert_outputs(blk, x_sel, list(miss_experts))     # [P, Em, d] targets
    e2col = {e: i for i, e in enumerate(miss_experts)}

    # centered ridge over 65 scalars (C buddies + β); per-dim intercept via demeaning
    Phi = np.concatenate([F[:nF], x_np[:nF, None, :]], axis=1)   # [nF, C+1, d]
    mu = Phi.mean(axis=0)                                        # [C+1, d]
    Phic = (Phi - mu).reshape(nF, C + 1, -1)
    G = np.einsum('tif,tjf->ij', Phic, Phic) / nF                # [C+1, C+1]
    Ymu = Y[:nF].mean(axis=0)                                    # [Em, d]
    B = np.einsum('tif,tef->ie', Phic, Y[:nF] - Ymu[None]) / nF  # [C+1, Em]
    lam = ridge * np.trace(G) / (C + 1)
    Greg = G + lam * np.eye(C + 1)

    def solve(idx):
        """ridge solve restricted to feature subset idx → coef [len(idx), Em]"""
        return np.linalg.solve(Greg[np.ix_(idx, idx)], B[idx])

    full_idx = np.arange(C + 1)
    coef_full = solve(full_idx)                                  # [C+1, Em]

    # in-sample residual of the full-span fit (fit-quality check vs distribution shift)
    pred_ins = np.einsum('tif,ie->tef', Phic, coef_full) + Ymu[None]
    ins = ((Y[:nF] - pred_ins) ** 2).sum((0, 2)) / ((Y[:nF] ** 2).sum((0, 2)) + 1e-12)
    stats["span64_insample"] = float(ins.mean())

    # predictions on eval miss tokens
    def predict(idx, coef, e, t_local):
        f = np.concatenate([F[t_local], x_np[t_local, None]], axis=0)  # [C+1, d]
        c = coef[:, e2col[e]]
        return (c[:, None] * (f[idx] - mu[idx])).sum(0) + Ymu[e2col[e]]

    # per-corner residuals, mass weighted over miss events
    corners = {f"span{j}": 0.0 for j in SPAN_JS}
    corners.update({"novice": 0.0, "affine": 0.0, "buddy1": 0.0})
    wsum = 0.0
    # affine = β,v only (feature = x alone); novice = v only (mean)
    beta_idx = np.array([C])
    coef_beta = solve(beta_idx)
    # per-expert sparse refits (top-j buddies by |c|·feature scale)
    fscale = np.sqrt(np.diag(G)[:C])
    for e in miss_experts:
        col = e2col[e]
        cb = np.abs(coef_full[:C, col]) * fscale
        ev = [(t, w) for t, ee, w, _ in miss if ee == e]
        t_loc = np.array([nF + (t - n_prompt) for t, _ in ev])
        ws = np.array([w for _, w in ev])
        y_true = Y[t_loc, col]                                   # [n, d]
        ynorm = (y_true ** 2).sum(1) + 1e-12
        wsum += ws.sum()

        def acc(name, pred):
            corners[name] += (ws * ((y_true - pred) ** 2).sum(1) / ynorm).sum()

        acc("novice", Ymu[col][None])
        acc("affine", np.stack([predict(beta_idx, coef_beta, e, t) for t in t_loc]))
        # best single buddy (generous proxy for today's buddy table)
        b_best = int(np.argmax(cb))
        cb1 = solve(np.array([b_best]))
        acc("buddy1", np.stack([predict(np.array([b_best]), cb1, e, t) for t in t_loc]))
        for j in SPAN_JS:
            if j >= C:
                idx = full_idx
                cj = coef_full
            else:
                idx = np.sort(np.argpartition(cb, -j)[-j:])
                idx = np.concatenate([idx, [C]])                 # keep β
                cj = solve(idx)
            acc(f"span{j}", np.stack([predict(idx, cj, e, t) for t in t_loc]))
    for k in corners:
        corners[k] /= max(1e-9, wsum)
    stats["residual"] = corners
    return stats


def layer_group(li):
    return "early(0-9)" if li < 10 else ("mid(10-29)" if li < 30 else "late(30-39)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--regimes", default=",".join(REGIMES))
    ap.add_argument("--c", type=int, default=64)
    ap.add_argument("--ctx-keep", type=int, default=512)
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument("--fit-window", default="prompt", choices=("prompt", "genhalf"))
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    print("[span] loading model ...", file=sys.stderr)
    model, tok = load(args.model)
    moe = find_moe_layers(model)
    print(f"[span] {len(moe)} MoE layers", file=sys.stderr)

    results = {}
    for regime in args.regimes.split(","):
        ref = mx.load(os.path.join(REFS_DIR, f"{regime}.safetensors"))
        ids = mx.concatenate([ref["spec_prompt"].astype(mx.int32),
                              ref["spec_greedy"].astype(mx.int32)])
        gen_len = ref["spec_greedy"].shape[0]
        print(f"[span] {regime}: T={ids.shape[0]} gen={gen_len}", file=sys.stderr)
        acts = capture_activations(model, moe, ids, args.ctx_keep)

        per_layer = {}
        for li, blk in moe:
            st = fit_layer(blk, acts[li], gen_len, args.c, args.ridge, args.fit_window)
            per_layer[li] = st
            print(f"[span] {regime} L{li:02d} missMass={st['missMass_obs']:.3f} "
                  f"nMiss={st['nMiss']} top1mass={st['top1Share_mass']:.2f} "
                  + (f"span8={st['residual']['span8']:.3f} buddy1={st['residual']['buddy1']:.3f} "
                     f"ins={st.get('span64_insample', -1):.3f}"
                     if "residual" in st else "(no miss)"), file=sys.stderr)
        results[regime] = per_layer

        # aggregate per layer-group, weighted by per-layer miss mass
        print(f"\n== {regime} (C={args.c}) ==")
        groups = {}
        for li, st in per_layer.items():
            groups.setdefault(layer_group(li), []).append(st)
        hdr = ["group", "missM", "top1cnt", "top1mass", "novice", "affine",
               "buddy1"] + [f"span{j}" for j in SPAN_JS]
        print(" ".join(f"{h:>9}" for h in hdr))
        for g in ("early(0-9)", "mid(10-29)", "late(30-39)"):
            sts = groups.get(g, [])
            w = np.array([s["missMass_obs"] for s in sts])
            if w.sum() == 0:
                continue
            row = [g,
                   f"{np.mean(w):.3f}",
                   f"{np.average([s['top1Share_count'] for s in sts], weights=w):.2f}",
                   f"{np.average([s['top1Share_mass'] for s in sts], weights=w):.2f}"]
            for k in ["novice", "affine", "buddy1"] + [f"span{j}" for j in SPAN_JS]:
                vals = [s["residual"][k] for s in sts if "residual" in s]
                wv = [s["missMass_obs"] for s in sts if "residual" in s]
                row.append(f"{np.average(vals, weights=wv):.3f}" if vals else "-")
            print(" ".join(f"{c:>9}" for c in row))

    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=1)
        print(f"[span] json → {args.json}", file=sys.stderr)


if __name__ == "__main__":
    main()
