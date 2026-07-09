# Qwisp

Apple Silicon (MLX) 上で **Qwen3.6 系 MoE モデル**を「制約デバイスでも実用速度で」動かすことを狙う、単一モデル特化ローカル推論エンジンの**調査・設計**リポジトリ。

antirez の [DwarfStar4 (DS4)](https://antirez.com/news/165) に着想を得つつ、方向性は逆 ―― DS4 が「巨大モデルを高メモリ機にギリ載せる」のに対し、本プロジェクトは **MoE のスパース性を活かして、より非力な Apple Silicon に reach を広げる**ことを目指す。中核アイデアは Apple の **AFM 3 Core Advanced** 戦略（大きなスパースモデルをフラッシュに常駐させ、必要なスライスだけ DRAM にロード）の自前 OSS 版。

実行時（ランタイム）は **Tell**、そのデコードエンジン（生 Metal 実装、旧 MLX）は **Seedless**。

## Quickstart（server / CLI）

```bash
# ビルド（Metal Toolchain 必須）
cd swift && xcodebuild build -scheme qwisp -configuration Release \
  -destination 'platform=macOS' -derivedDataPath ./.xcode-build-rel -skipPackagePluginValidation
BIN=swift/.xcode-build-rel/Build/Products/Release/qwisp
export QWISP_MODEL=$HOME/.mtplx/models/Youssofal--Qwen3.6-35B-A3B-MTPLX-Optimized-Speed-FP16

# OpenAI 互換サーバ（QWISP_PORT 既定 8080）
"$BIN" serve
#   GET  /v1/models
#   POST /v1/chat/completions   （stream:true で SSE、既定は非ストリーミング JSON）

# CLI（in-process、stdout にストリーミング）
"$BIN" chat "日本語で自己紹介して"
```

**サンプリングについて（重要）**: エンジンは *lossless greedy*。OpenAI リクエストの
`temperature` / `top_p` / `n` は**受理するが無視**する（決定的・greedy 出力）。指定されると
レスポンスに `x-qwisp-warning: sampling params ignored (greedy/lossless engine)` ヘッダが付き、
`serve` 起動時にもその旨を表示する。tools / logprobs / n>1 は未対応。

## ターゲット

- **プラットフォーム**：Apple Silicon のみ、MLX ベース（llama.cpp ではなく MLX を出発点にする）
- **モデル**：主 = Qwen3.6-35B-A3B (MoE, 35B total / 3B active)、従 = Qwen3.6-27B (dense)
- **名前の由来**：**Qwen** + **wisp**（鬼火・小さな霊）。Qwen に紐づけつつ、軽量フットプリントで小さく速く働く助っ人。

## ドキュメント

| ファイル | 内容 |
| --- | --- |
| [`docs/00-overview.md`](docs/00-overview.md) | 基本方針・全体像・過拘束の認識 |
| [`docs/01-research-notes.md`](docs/01-research-notes.md) | AFM 3 Core Advanced と関連研究（フラッシュ常駐／MoE offloading） |
| [`docs/02-roadmap.md`](docs/02-roadmap.md) | 着手手順（Step 1–4）と go/no-go ゲート |
| [`docs/03-conversation-log.md`](docs/03-conversation-log.md) | 設計に至る議論の記録 |
| [`docs/04-benchmark-selection.md`](docs/04-benchmark-selection.md) | Step1 プロンプトのベンチ由来化（SWE-bench/BFCL/RULER 等） |
| [`docs/05-go-no-go-first-read.md`](docs/05-go-no-go-first-read.md) | go/no-go 初回実測（GREEN、実 flash 帯域較正） |
| [`docs/06-step4-poc.md`](docs/06-step4-poc.md) | Step4 PoC（expert streaming bit一致）・MTP×streaming・アーキ決定 |
| [`docs/07-positioning.md`](docs/07-positioning.md) | ポジショニング（SwiftLM 比較）・哲学・4つ組・v0.1 計画 |

## ステータス / ライセンス

- Visibility: **Private**（公開したくなったら GitHub 側でフラグ切替）
- License: 未定
