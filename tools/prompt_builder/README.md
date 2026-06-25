# prompt_builder — ベンチ由来プロンプト生成（決定論的）

選定ベンチ（`../../docs/04-benchmark-selection.md`）のサブセットを取得し、Step 1 collector の
スキーマ（`{prompt_id, category, text}`）に正規化して `prompts.jsonl` を生成する。

## 特徴

- **stdlib のみ**（`datasets` 不要 → loading-script / trust_remote_code の非決定性を回避、
  MTPLX runtime-venv を汚さない）。どの python3 でも動く。
- フェッチ = HF 生ファイル（`/resolve/main/`）＋ datasets-server `/rows`。**公開データで token 不要**。
  `HF_TOKEN` を env に置けばレート制限緩和に使う（任意・チャットに貼らない）。
- **決定論**: `seed` 固定 → 同じ manifest なら同じ prompts。選択 id を lock に記録。

## 対応ベンチ（全6・実フェッチ検証済）

| adapter | source | カテゴリ | プロンプト源 |
| --- | --- | --- | --- |
| `swe_verified` | `princeton-nlp/SWE-bench_Verified`（/rows） | coding | `problem_statement`（repo/commit 文脈付） |
| `livecodebench` | `livecodebench/code_generation_lite`（生 jsonl） | coding | `question_content`（+`starter_code`） |
| `bfcl` | `gorilla-llm/Berkeley-Function-Calling-Leaderboard`（生 json×N） | agentic | `question`＋`function` を tool-use 整形 |
| `terminal_bench` | `zai-org/terminal-bench-2-verified`（tree→各 `instruction.md`） | agentic | `instruction.md` |
| `ruler` | `rbiswasfc/ruler`（/rows、`*_8k`） | long_context | `input`（長文脈そのまま） |
| `repoqa` | evalplus GitHub release gz | long_context | `content` 連結（char budget）＋needle 説明 |

## 使い方

```bash
# 本番（manifest の n で 45/45/10）
python3 build_prompts.py --manifest benchmarks.manifest.json \
    --out ../step1_routing_trace/prompts.bench.jsonl \
    --lock prompts.lock.json

# 検証（各ベンチ n を上書き / 単一ベンチのみ）
python3 build_prompts.py --manifest benchmarks.manifest.json \
    --out /tmp/p.jsonl --lock /tmp/p.lock.json --limit-per-bench 2
python3 build_prompts.py ... --only ruler
```

その後 Step 1 へ:
```bash
PY="$HOME/Library/Application Support/MTPLX/runtime-venv/bin/python"
"$PY" ../step1_routing_trace/collect_traces.py --model <dir> \
    --prompts ../step1_routing_trace/prompts.bench.jsonl --out traces.jsonl --max-new-tokens 128
```

## 追跡方針

- **追跡**: `build_prompts.py` / `benchmarks.manifest.json` / `prompts.lock.json`（id のみ）。
- **非追跡（gitignore）**: 生成された `prompts.bench.jsonl`（再現可能＋データ再配布回避）。
- 手書きの `../step1_routing_trace/prompts.sample.jsonl`（オフライン/ネット不要のフォールバック）は追跡。

## 比率

manifest の `n` 合計が 45/45/10 を担保（既定 coding 18 / agentic 18 / long_context 4）。
`--limit-per-bench` を使うと各ベンチ一律になり比率は崩れる（検証専用）。
