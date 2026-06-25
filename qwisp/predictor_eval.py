"""(う) de-risk — cross-layer expert 予測器を学習し coverage を測る.

仮説: 層 L-1 の gate入力 hidden から 層 L の experts を tiny MLP で予測できる
（cross-layer、ProMoE/Fate）。zero-shot（次層 gate を現 hidden に当てる）= 77%。
学習器がそれを超え 96% 級に届けば、prefetch で miss-IO を隠す路線が成立。

手順:
1. prefill を数プロンプト走らせ、各層の (gate入力 x, top-8 experts) を捕捉。
2. 層 L について X=層(L-1)の x, Y=層 L の experts(multi-hot) で MLP 学習。
3. test で top-8 coverage を測り、zero-shot と比較。

実行:
  PY="$HOME/Library/Application Support/MTPLX/runtime-venv/bin/python"
  "$PY" -m qwisp.predictor_eval "$MODEL"
"""

from __future__ import annotations

import sys
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx_lm import load
from mlx_lm.generate import stream_generate
from mlx_lm.sample_utils import make_sampler
from mlx_lm.models.qwen3_next import Qwen3NextSparseMoeBlock

NEXP = 256
K = 8
LAYERS = 40

PROMPTS = [
    "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr)//2]\n",
    "import numpy as np\n\ndef softmax(x):\n    e = np.exp(x - np.max(x))\n    return e / e.sum()\n\nclass Layer:\n",
    "The history of computing began with mechanical calculators. In the 19th century, Charles Babbage designed\n",
    "You are an agent with tools read_file, edit_file, run_tests. The user wants to fix a failing test in\n",
    "fn merge_sort<T: Ord + Clone>(v: &[T]) -> Vec<T> {\n    if v.len() <= 1 { return v.to_vec(); }\n",
    "SELECT customers.name, SUM(orders.amount) AS total FROM customers JOIN orders ON\n",
]


def capture(model, tok, max_len=1024):
    cap = []
    orig = Qwen3NextSparseMoeBlock.__call__

    def patched(self, x):
        g = mx.softmax(self.gate(x), axis=-1, precise=True)
        inds = mx.argpartition(g, kth=-K, axis=-1)[..., -K:]
        mx.eval(x, inds)
        cap.append((np.array(x).astype(np.float16).reshape(-1, x.shape[-1]),
                    np.array(inds).reshape(-1, K)))
        return orig(self, x)

    Qwen3NextSparseMoeBlock.__call__ = patched
    sampler = make_sampler(temp=0.0)
    for p in PROMPTS:
        ids = tok.encode(p * 30)[:max_len]  # 反復で長さ確保（6 種 × max_len 例/層）
        for _ in stream_generate(model, tok, prompt=ids, max_tokens=1, sampler=sampler):
            break
    Qwen3NextSparseMoeBlock.__call__ = orig
    return cap


def group_by_layer(cap):
    nprompt = len(cap) // LAYERS
    perX = [[] for _ in range(LAYERS)]
    perE = [[] for _ in range(LAYERS)]
    for i, (x, e) in enumerate(cap[:nprompt * LAYERS]):
        L = i % LAYERS
        perX[L].append(x)
        perE[L].append(e)
    return ([np.concatenate(perX[L]) for L in range(LAYERS)],
            [np.concatenate(perE[L]) for L in range(LAYERS)])


def multihot(E):
    Y = np.zeros((E.shape[0], NEXP), np.float32)
    Y[np.arange(E.shape[0])[:, None], E] = 1.0
    return Y


class MLP(nn.Module):
    def __init__(self, h=256):
        super().__init__()
        self.l1 = nn.Linear(2048, h)
        self.l2 = nn.Linear(h, NEXP)

    def __call__(self, x):
        return self.l2(nn.relu(self.l1(x)))


def topk_cov(logits, E_actual):
    pred = np.argpartition(np.asarray(logits), -K, axis=-1)[:, -K:]
    cov = 0.0
    for i in range(E_actual.shape[0]):
        cov += len(set(pred[i]) & set(E_actual[i])) / K
    return cov / E_actual.shape[0]


def train_layer(Xtr, Ytr_mh, Xte, Ete, epochs=12, bs=256):
    model = MLP()
    opt = optim.Adam(learning_rate=1e-3)

    def loss_fn(m, x, y):
        return mx.mean(nn.losses.binary_cross_entropy(m(x), y, with_logits=True))

    lvg = nn.value_and_grad(model, loss_fn)
    Xtr_m, Ytr_m = mx.array(Xtr.astype(np.float32)), mx.array(Ytr_mh)
    N = Xtr.shape[0]
    for _ in range(epochs):
        perm = np.random.permutation(N)
        for s in range(0, N, bs):
            idx = perm[s:s + bs].tolist()
            _, g = lvg(model, Xtr_m[idx], Ytr_m[idx])
            opt.update(model, g)
            mx.eval(model.parameters(), opt.state)
    logits = model(mx.array(Xte.astype(np.float32)))
    mx.eval(logits)
    return topk_cov(logits, Ete)


def main():
    M = sys.argv[1]
    print(f"[pred] loading {M} ...", file=sys.stderr)
    model, tok = load(M)
    t0 = time.perf_counter()
    cap = capture(model, tok)
    print(f"[pred] captured {len(cap)} calls in {time.perf_counter()-t0:.0f}s", file=sys.stderr)
    Xs, Es = group_by_layer(cap)
    del model
    if hasattr(mx, "clear_cache"):
        mx.clear_cache()

    sample = list(range(3, LAYERS, 6))
    print(f"{'layer':>5} {'N':>6} {'trained_cov':>11}")
    t_all = []
    for L in range(1, LAYERS):
        n = min(len(Xs[L - 1]), len(Es[L]))
        X_prev, E_cur = Xs[L - 1][:n], Es[L][:n]
        ntr = int(n * 0.8)
        cov = train_layer(X_prev[:ntr], multihot(E_cur[:ntr]), X_prev[ntr:], E_cur[ntr:])
        t_all.append(cov)
        if L in sample:
            print(f"{L:>5} {n:>6} {cov:>11.3f}", flush=True)
    print(f"\n[pred] trained coverage 平均={np.mean(t_all):.3f} "
          f"(zero-shot 0.771, 文献 0.96)", file=sys.stderr)


if __name__ == "__main__":
    main()
