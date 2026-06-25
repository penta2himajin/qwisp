# 04 — Step 1 プロンプトのベンチマーク化（選定と調査）

調査時点 2026-06。Step 1 routing trace のプロンプトを手書きから**既存ベンチのサブセット**に置換する方針と、その選定・実スキーマ調査の記録。

## 方針

- routing trace は**入力プロンプトさえあればよい**（モデルで生成して routing を取るだけ。正解・採点ハーネス不要）。→ 各ベンチから「入力部分」だけ抽出すればよい。
- データ本体はコミットせず、**HF から取得 → seed 固定 sample → `prompts.jsonl` 生成**。再現用に **manifest（どのベンチ・config・split・n・seed）と lock（解決済み index/id）だけ追跡**。
- 比率は確定の **coding 45 / agentic 45 / long_context 10**（[[prompt-mix-decision]]）を各ベンチの `n` で担保。
- long_context は「用途」でなく**キャッシュ膨張ストレス枠**なので、長さ可変の合成（RULER）を中心に。

## 選定（ユーザー確定）

| 枠 | ベンチ | HF dataset | 状態 |
| --- | --- | --- | --- |
| coding 45% | **SWE-bench Verified** | `princeton-nlp/SWE-bench_Verified` | ✅スキーマ確認済 |
| coding 45% | **LiveCodeBench** | `livecodebench/code_generation_lite` | ⚠ config/split 要確認 |
| agentic 45% | **BFCL v4** | `gorilla-llm/Berkeley-Function-Calling-Leaderboard` | ⚠ 非標準（複数 JSON ファイル） |
| agentic 45% | **Terminal-Bench 2.0** | `zai-org/terminal-bench-2-verified`（候補: `harborframework/...`, `penfever/...`） | ⚠ config/split 要確認 |
| long_context 10% | **RULER** | 正準静的 dataset 無し（生成器）。prebuilt or 自前生成 | ⚠ 取得方法 未確定 |
| long_context 10% | **RepoQA** | `repoqa` パッケージ（evalplus）配布。HF id 要確認 | ⚠ 要確認 |

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

## 残 TODO（実装前）

1. LiveCodeBench の config/split/フィールドを実ロードで確定（version タグ）。
2. Terminal-Bench 2.0 の正リポと instruction フィールド（行 or 同梱ファイル）。
3. BFCL のファイル取得＋tool-use 整形アダプタ。
4. RULER の取得方法（prebuilt vs 自前生成、長さ 8K/16K）。
5. RepoQA の HF id とフィールド。

> 関連: 比率 [[prompt-mix-decision]]、収集器 `tools/step1_routing_trace/`、決定状況 [[qwisp-open-decisions]]。
