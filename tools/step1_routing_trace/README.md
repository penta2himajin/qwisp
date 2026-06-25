# Step 1 — routing trace collector（MLX）

Qwen3.6-35B-A3B の MoE router を計装し、**レイヤ別・トークン別の top-k expert ID 列**を JSONL にログする。Step 2（キャッシュシミュレーション）の入力になる。

## なぜこれが最初の一手か

- routing は「モデル × プロンプト」の性質で**ハード非依存**。go/no-go ゲート（Step 2）は trace の集計ヒット率で決まる → ここの品質がプロジェクト全体の生殺与奪。

## フレームワーク：MLX（mlx_lm）に確定

transformers を選んだ唯一の動機は `output_router_logits` の hook 容易性だけで、routing 自体はフレームワーク非依存（gate の argmax）。だが手元の実用モデルが **MLX 量子化形式**（mtplx 配布、4bit / 256 experts / top-8 / 40層）で、**24GB に載って今すぐ動く唯一の道が MLX**。かつ量子化 routing = 出荷忠実で go/no-go に最も効く。経緯は `../../docs/02-roadmap.md` / `../../docs/03-conversation-log.md`。

## hook 点（mlx_lm ソース実読で確定）

- モデル実装は `mlx_lm.models.qwen3_5` / `qwen3_next`。MoE は `qwen3_next.Qwen3NextSparseMoeBlock`。
  `__call__` 内で `gates = softmax(self.gate(x))` → `inds = argpartition(gates)[..., -top_k:]` が top-k expert ID。
- `collect_traces.py` はこの式を**再計算**して `inds` を捕捉（gate は dim×num_experts の小行列で安価）。
- `layer_idx` は instance 初回出現順＝デコーダ層順（属性パス非依存で堅牢）。
- 素の `mlx_lm.load` は `sanitize` で `mtp.*` を落とす → **MTP 抜きの純 AR モデル**。よって本収集は
  **AR モード**で routing を取る＝投機デコードの draft の routing を混ぜない（受理トークン列のみ）。

## プロンプト比率（確定）

| カテゴリ | 比率 | 意図 |
| --- | --- | --- |
| `coding` | 45% | 主用途 |
| `agentic` | 45% | 主用途。tool-calling ループ |
| `long_context` | 10% | 用途でなく**キャッシュ膨張のストレス検査枠**（長い prefill が瞬間ワーキングセットを膨らませ no-go を出しうる唯一のケース） |

**プロンプトファイル**:
- `prompts.sample.jsonl` — 最小雛形（7行）。
- `prompts.jsonl` — 確定比率の作業セット（coding9 / agentic9 / long_context2 = 45/45/10）。本番は実リポジトリ・実ログで各カテゴリを増やす。

**プロンプト行スキーマ**:
- `{"prompt_id", "category", "text"}` — インライン本文（coding/agentic）。
- `{"prompt_id", "category", "instruction", "text_file"}` — 大きな実素材をファイル参照（long_context）。`text_file` は prompts JSONL 相対で解決し、`instruction` を先頭に付ける。素材は `corpora/`（`.gitignore` 済み）に 8K–16K tokens の本物を置く（→ `corpora/README.md`）。未配置の long_context 行は警告スキップされ、coding/agentic は素材不要で即実行できる。

## 使い方

mlx_lm を持つ **mtplx runtime-venv の python** を使う（システム python3 には mlx_lm が無い）:

```bash
PY="$HOME/Library/Application Support/MTPLX/runtime-venv/bin/python"
M="$HOME/.mtplx/models/Youssofal--Qwen3.6-35B-A3B-MTPLX-Optimized-Speed-FP16"

# 配線検証（1短プロンプト、層数・top_k・id レンジを assert）
"$PY" collect_traces.py --model "$M" --smoke

# 本収集
"$PY" collect_traces.py --model "$M" \
    --prompts prompts.sample.jsonl --out traces.jsonl --max-new-tokens 128
```

検証済み（実モデル smoke）: `layers=40 (0..39) top_k=8 max_expert_id=255` ＝ config（num_hidden_layers=40 / num_experts_per_tok=8 / num_experts=256）と一致。

出力 (`traces.jsonl`)、1行=1トークン×1レイヤ:

```json
{"prompt_id":"code-01","category":"coding","phase":"decode","token_idx":98,"layer_idx":0,"expert_ids":[236,33,157,218,43,66,212,106]}
```

## 次

`traces.jsonl` が貯まったら Step 2（`tools/step2_cache_sim/`、未作成）で DRAM 予算を振りながら LRU/LFU/Belady のヒット率を出す。判定ゲートは `../../docs/02-roadmap.md` Step 2。
