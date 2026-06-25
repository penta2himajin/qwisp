# 04 — Step 1 プロンプトのベンチマーク化（選定と調査）

調査時点 2026-06。Step 1 routing trace のプロンプトを手書きから**既存ベンチのサブセット**に置換する方針と、その選定・実スキーマ調査の記録。

## 方針

- routing trace は**入力プロンプトさえあればよい**（モデルで生成して routing を取るだけ。正解・採点ハーネス不要）。→ 各ベンチから「入力部分」だけ抽出すればよい。
- データ本体はコミットせず、**HF から取得 → seed 固定 sample → `prompts.jsonl` 生成**。再現用に **manifest（どのベンチ・config・split・n・seed）と lock（解決済み index/id）だけ追跡**。
- 比率は確定の **coding 45 / agentic 45 / long_context 10**（[[prompt-mix-decision]]）を各ベンチの `n` で担保。
- long_context は「用途」でなく**キャッシュ膨張ストレス枠**なので、長さ可変の合成（RULER）を中心に。

## 選定（ユーザー確定）

| 枠 | ベンチ | 取得元（確定） | フィールド | 状態 |
| --- | --- | --- | --- | --- |
| coding 45% | **SWE-bench Verified** | `princeton-nlp/SWE-bench_Verified`（datasets-server /rows） | `problem_statement` | ✅実フェッチ済 |
| coding 45% | **LiveCodeBench** | `livecodebench/code_generation_lite` 生 `test.jsonl`（Range 先頭のみ） | `question_content`(+`starter_code`) | ✅実フェッチ済 |
| agentic 45% | **BFCL v4** | `gorilla-llm/Berkeley-Function-Calling-Leaderboard` 生 `BFCL_v3_*.json` | `question`+`function` を tool-use 整形 | ✅実フェッチ済 |
| agentic 45% | **Terminal-Bench 2.0** | `zai-org/terminal-bench-2-verified` tree→各 `<task>/instruction.md` | `instruction.md` | ✅実フェッチ済 |
| long_context 10% | **RULER** | `rbiswasfc/ruler`（/rows、config `*_8k`、split `validation`） | `input` | ✅実フェッチ済 |
| long_context 10% | **RepoQA** | evalplus GitHub release gz（`2024-06-23`） | `content` 連結＋`needles` 説明 | ✅実フェッチ済 |

## 実スキーマ調査（HF datasets-server `/first-rows` 等）

### ✅ SWE-bench Verified（確認済）
- `princeton-nlp/SWE-bench_Verified`, config `default`, split `test`, 500行。
- フィールド: `repo, instance_id, base_commit, problem_statement, patch, test_patch, hints_text, created_at, version, FAIL_TO_PASS, PASS_TO_PASS, environment_setup_commit, difficulty`。
- **プロンプト源 = `problem_statement`**（issue 本文）。任意で `repo`/`base_commit` を文脈付与。
- 長文脈転用: `princeton-nlp/SWE-bench_bm25_27K` は検索済みコード文脈つき（長文脈枠の代替候補）。

### ⚠ LiveCodeBench（要確認）
- `livecodebench/code_generation_lite` の `/first-rows?config=default&split=test` は **404**。
  config が version タグ（例 `release_v1`〜）で、`trust_remote_code` 必要の可能性。
- 想定フィールド: `question_title, question_content, starter_code`（`question_content` がプロンプト源）。**実ロードで確認すること。**

### ⚠ Terminal-Bench 2.0（要確認）
- `zai-org/terminal-bench-2-verified` の `/first-rows?config=default&split=train` は **404**。config/split 不一致。
- 候補リポ: `zai-org/terminal-bench-2-verified`（verified 修正版）, `harborframework/terminal-bench-2.0`, `penfever/terminal-bench-2`。
- 89 hard タスク。タスクは env＋instruction で、**行データでなくファイル同梱の可能性**あり。instruction フィールド名は実ロードで確認。

### ⚠ BFCL v4（非標準フォーマット）
- `gorilla-llm/Berkeley-Function-Calling-Leaderboard`。**標準 HF 行ではなく、カテゴリ別の複数 JSON(L) ファイル**（simple / multiple / parallel / multi_turn / java / javascript ...）。
- 各行 = `{question:[turns], function:[tool schemas], ...}`。
- 取得は `huggingface_hub` でファイル DL → jsonl パース。**マッピング: `question` + `function` → tool-use chat プロンプト**（chat template に tools を渡す形）。
- 1,800+ タスク。multi-turn/multi-step あり。

### ⚠ RULER（生成器）
- 単一の正準静的 HF dataset は無い（合成生成器）。選択肢: (a) コミュニティ prebuilt（長さ別）、(b) RULER 生成器を 8K/16K で自前実行。
- 13 タスク（8 NIAH + 集約/QA/multi-hop）。**長さ可変＝ストレス枠に最適**。取得方法は未確定。

### ⚠ RepoQA（要確認）
- evalplus の `repoqa` パッケージ配布。HF dataset id 要確認。長コード＋needle 関数の検索。オンドメイン長文脈。

## 構築プラン（実装予定 `tools/prompt_builder/`）

- **別 venv**（`datasets` + `huggingface_hub` のみ、MLX 不要）。MTPLX runtime-venv は汚さない（datasets 未導入を確認済）。
- **per-benchmark アダプタ関数**（フィールドマッピングをコードに持つ。BFCL のような非標準にも対応）。
- `benchmarks.manifest.json`（追跡）: `{bench: {hf_id, config, split, category, n, seed, adapter}}`。
- `*.lock.json`（追跡）: 解決済み index/instance_id（完全再現）。
- 出力 `prompts.jsonl`（既存スキーマ: `text` or `instruction`+`text_file`）。

## 実装結果（解決済み）

保留5件は実フェッチで全解決。`datasets` ライブラリは使わず（loading-script の非決定性回避）、
**HF 生ファイル `/resolve/main/` ＋ datasets-server `/rows` を stdlib で直取り**する方針で
`tools/prompt_builder/build_prompts.py` を実装。全6アダプタが実データで取得・正規化を確認。

- datasets-server は LiveCodeBench/Terminal-Bench/BFCL を rows 提供できず（501/500＝loading-script・
  非標準）→ 生ファイル直取りに切替で解決。
- BFCL は `multi_turn_*` が `function` キーを持たない別スキーマ → 単一ターン系
  （simple/multiple/parallel/live_*）に限定＋欠落 skip で解決。
- RULER は合成生成器だが prebuilt `rbiswasfc/ruler` の `*_8k` を採用。`input` がそのまま長文脈。
- RepoQA は evalplus GitHub release の gz（version `2024-06-23`）。

**性能**（並行化済み）: ベンチ間＋ベンチ内フェッチを ThreadPool 並行、LiveCodeBench は
Range で先頭のみ、巨大DL はオンディスクキャッシュ。コールド ~104s（I/O 律速）/ ウォーム ~2.5s。
seed 固定＋順序保存で**決定論**（再実行で lock 一致を確認）。

生成物: `tools/step1_routing_trace/prompts.bench.jsonl`（gitignore、再現は manifest+lock）。

> 関連: builder `tools/prompt_builder/`、比率 [[prompt-mix-decision]]、収集器 `tools/step1_routing_trace/`、決定状況 [[qwisp-open-decisions]]。
