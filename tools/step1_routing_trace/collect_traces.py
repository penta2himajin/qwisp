#!/usr/bin/env python3
"""Qwisp Step 1 — routing trace collector (MLX / mlx_lm).

Qwen3.6-35B-A3B (MoE) の router を計装し、レイヤ別・トークン別の top-k expert ID 列を
JSONL にログする。Step 2（キャッシュシミュレーション）の入力。

なぜ MLX か（②の最終決定）:
- transformers を選んだ唯一の動機は `output_router_logits` の hook 容易性だけで、
  routing 自体はフレームワーク非依存（gate の argmax）。
- 手元の実用モデルは MLX 量子化形式（mtplx 配布、4bit / 256 experts / top-8 / 40層）で、
  24GB に載って今すぐ動く唯一の道が MLX。量子化 routing = 出荷忠実で go/no-go に最も効く。
  詳細: ../../docs/02-roadmap.md, memory: qwisp-open-decisions。

hook 点（mlx_lm のソースを実読して確定）:
- モデル実装は mlx_lm.models.qwen3_5 / qwen3_next。MoE ブロックは
  `qwen3_next.Qwen3NextSparseMoeBlock`。その __call__ は
      gates = softmax(self.gate(x));  inds = argpartition(gates)[..., -top_k:]
  で `inds` が top-k expert ID。これを再計算して捕捉する（gate は小行列で安価）。
- layer_idx は「初回呼び出し順＝デコーダ層順」で id マッピング（属性パス非依存）。
- 素の mlx_lm.load は sanitize で mtp.* を落とす → MTP 抜きの純 AR モデル。
  よって本スクリプトは **AR モード**（投機デコード無し）で routing を取る＝
  draft の routing を混ぜない（受理トークン列の routing のみ）。

実行（mlx_lm を持つ mtplx runtime-venv の python を使う）:
  PY="$HOME/Library/Application Support/MTPLX/runtime-venv/bin/python"
  "$PY" collect_traces.py \
      --model "$HOME/.mtplx/models/Youssofal--Qwen3.6-35B-A3B-MTPLX-Optimized-Speed-FP16" \
      --prompts prompts.sample.jsonl \
      --out traces.jsonl \
      --max-new-tokens 128

まず配線確認:
  "$PY" collect_traces.py --model <dir> --prompts prompts.sample.jsonl --out /tmp/smoke.jsonl --smoke

出力 (1行=1トークン×1レイヤ):
  {"prompt_id","category","phase":"prefill"|"decode","token_idx","layer_idx","expert_ids":[...]}
"""

import argparse
import json
import sys

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import stream_generate
from mlx_lm.sample_utils import make_sampler
from mlx_lm.models.qwen3_next import Qwen3NextSparseMoeBlock


class RoutingRecorder:
    """MoE gate の top-k inds を行ごと（=トークンごと）に記録。

    layer_idx は instance の初回出現順で 0..(num_moe_layers-1) を割り当てる
    （mlx_lm の forward は層を順に呼ぶので層インデックスと一致）。
    """

    def __init__(self):
        self.records = []
        self._layer_of = {}            # id(module) -> layer_idx
        self._next_layer = 0
        self.layer_token_counter = {}  # layer_idx -> running token count
        self.prompt_id = None
        self.category = None
        self.enabled = False

    def begin_prompt(self, prompt_id, category):
        self.prompt_id = prompt_id
        self.category = category
        self.layer_token_counter = {}
        self.records = []

    def layer_index_for(self, module):
        key = id(module)
        idx = self._layer_of.get(key)
        if idx is None:
            idx = self._next_layer
            self._layer_of[key] = idx
            self._next_layer += 1
        return idx

    def record(self, module, inds_rows):
        if not self.enabled:
            return
        layer_idx = self.layer_index_for(module)
        start = self.layer_token_counter.get(layer_idx, 0)
        for row, expert_ids in enumerate(inds_rows):
            self.records.append({
                "prompt_id": self.prompt_id,
                "category": self.category,
                "token_idx": start + row,
                "layer_idx": layer_idx,
                "expert_ids": expert_ids,
            })
        self.layer_token_counter[layer_idx] = start + len(inds_rows)


RECORDER = RoutingRecorder()


def install_hook():
    """Qwen3NextSparseMoeBlock.__call__ を包んで inds を捕捉する。"""
    orig_call = Qwen3NextSparseMoeBlock.__call__

    def patched_call(self, x):
        if RECORDER.enabled:
            # __call__ 内と同じ式で top-k expert を再計算（gate は dim×num_experts の小行列）。
            gates = self.gate(x)
            gates = mx.softmax(gates, axis=-1, precise=True)
            k = self.top_k
            inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
            mx.eval(inds)
            rows = inds.reshape(-1, k).tolist()
            RECORDER.record(self, rows)
        return orig_call(self, x)

    Qwen3NextSparseMoeBlock.__call__ = patched_call


def load_prompts(path):
    prompts = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line))
    return prompts


def encode_prompt(tokenizer, text):
    """chat template があれば適用（coding/agentic は chat 形式が実用途に近い）。"""
    try:
        if getattr(tokenizer, "chat_template", None):
            ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                add_generation_prompt=True,
            )
            return list(ids)
    except Exception as e:  # noqa: BLE001 - テンプレ非対応時は素のエンコードに退避
        print(f"[qwisp] chat_template fallback: {e}", file=sys.stderr)
    return tokenizer.encode(text)


def run_prompt(model, tokenizer, prompt_obj, max_new_tokens, out_f):
    text = prompt_obj["text"]
    pid = prompt_obj["prompt_id"]
    cat = prompt_obj.get("category", "unknown")

    prompt_ids = encode_prompt(tokenizer, text)
    prompt_len = len(prompt_ids)

    RECORDER.begin_prompt(pid, cat)
    RECORDER.enabled = True
    sampler = make_sampler(temp=0.0)  # greedy: routing を決定的にする
    n = 0
    for _resp in stream_generate(
        model, tokenizer, prompt=prompt_ids,
        max_tokens=max_new_tokens, sampler=sampler,
    ):
        n += 1
    RECORDER.enabled = False

    # token_idx < prompt_len は prefill、以降は decode。
    written = 0
    for rec in RECORDER.records:
        rec["phase"] = "prefill" if rec["token_idx"] < prompt_len else "decode"
        out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        written += 1
    print(f"[qwisp] {pid}: prompt_len={prompt_len} gen={n} rows={written}",
          file=sys.stderr)
    return prompt_len, written


def smoke_test(model, tokenizer):
    """1 短プロンプトで配線を検証：層数・top_k 形状・id レンジを assert。"""
    RECORDER.begin_prompt("smoke", "smoke")
    RECORDER.enabled = True
    sampler = make_sampler(temp=0.0)
    for _ in stream_generate(model, tokenizer, prompt=tokenizer.encode("def add(a, b):"),
                             max_tokens=4, sampler=sampler):
        pass
    RECORDER.enabled = False

    recs = RECORDER.records
    assert recs, "no routing captured — hook が発火していない"
    layers = sorted({r["layer_idx"] for r in recs})
    ks = {len(r["expert_ids"]) for r in recs}
    max_id = max(max(r["expert_ids"]) for r in recs)
    print(f"[qwisp][smoke] layers={len(layers)} (0..{layers[-1]}) "
          f"top_k set={ks} max_expert_id={max_id} rows={len(recs)}", file=sys.stderr)
    assert len(ks) == 1, f"top_k が一定でない: {ks}"
    print("[qwisp][smoke] OK", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description="Qwisp Step 1 routing trace (MLX)")
    ap.add_argument("--model", required=True, help="MLX model dir (mtplx 配布の Qwen3.6-35B-A3B)")
    ap.add_argument("--prompts", help="prompts JSONL")
    ap.add_argument("--out", help="output traces JSONL")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--smoke", action="store_true", help="配線検証のみ（1短プロンプト）")
    args = ap.parse_args()

    install_hook()
    print(f"[qwisp] loading {args.model} ...", file=sys.stderr)
    model, tokenizer = load(args.model)

    if args.smoke:
        smoke_test(model, tokenizer)
        return

    if not (args.prompts and args.out):
        raise SystemExit("--prompts と --out が必要（または --smoke）")

    prompts = load_prompts(args.prompts)
    print(f"[qwisp] {len(prompts)} prompts", file=sys.stderr)
    with open(args.out, "w", encoding="utf-8") as out_f:
        for p in prompts:
            run_prompt(model, tokenizer, p, args.max_new_tokens, out_f)
    print(f"[qwisp] traces -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
