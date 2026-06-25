# 03 — 設計に至る議論の記録

quick-wichtel の方針が固まるまでの対話を時系列で要約。一次ソースは `01-research-notes.md` 参照。

## 1. 発端：DS4 の Qwen 版を作れないか

- アイデア：antirez の **DwarfStar4 (DS4)** の Qwen 版。ターゲットは Qwen3.6-27B / 35B-A3B。デバイス制約を可能な限り広げたい。
- DS4 の実体：DeepSeek V4 Flash (284B) **専用**の llama.cpp 派生エンジン。非対称 2/8bit 量子化、disk KV キャッシュ一級市民化、steering 組み込み。狙いは「巨大モデルを 96–128GB 機にギリ載せる」。
- 最初の批判：DS4 の動機（載らないものを載せる）は状況依存。Qwen3.6 は **27B≈Q4 17GB / 35B-A3B≈Q4 21GB** で既に commodity HW で動き、llama.cpp/MLX/Ollama でサポート済み。→ DS4 の主動機が消える。さらに「デバイス制約を広げる＝より非力な機」は DS4 の高メモリ向け設計と逆向き。
- ただし Qwen3.6 固有で専用化の価値がある軸：(1) Gated DeltaNet のまともな実装（llama.cpp に遅延ギャップ、MLX は無し）、(2) MoE 非対称量子化 + expert streaming、(3) 検証済みエージェント層。
- 目的の二股を指摘：**(A) reach 最大化** vs **(B) DS4 的・1 モデル高品質エージェント**。設計が逆を向く。

## 2. プラットフォーム確定：Apple Silicon / MLX のみ

- ユーザー判断：MLX 的な環境特化最適化が良い。ターゲットは Apple Silicon の macOS のみ。**最弱 = M1〜M 系無印**を想定。「非力な SoC 上で十分高性能な LLM を高速に」。基本 A 寄せ + 最適化。
- 評価：scoping として筋が良い。MLX が出発点として正しい（Apple 上で llama.cpp 比 23–41% 速い、DeltaNet 遅延も無い）。
- ただし**過拘束を指摘**：「最弱 M1(8GB) × 27B/35B × 十分高性能」は三立しない。
  - base 帯域：M1 68 / M4 120 / M5 153 GB/s。base RAM は M4 以降でようやく 16GB 下限。8GB 機は 8B すら常時スワップ（実効 0.4 tok/s 級）。
  - 35B-A3B(21GB) は 8/16GB に載らず。27B も 16GB は IQ3 で辛うじて、低帯域で数 tok/s。
  - → 現実的フロアは **24GB 以上 + M4/M5 級帯域**。
- 和解点：**MoE はボトルネック反転**（帯域安い・容量高い）。帯域は物理上限、容量は量子化 + expert streaming で攻める。→ 27B より **35B-A3B 推し**が defensible。
- リスク：expert streaming の局所性次第で stall / NAND 摩耗。「十分高性能」未定義のドリフト。
- 次の一手として「フロアはどこにコミットできるか」を確認。

## 3. AFM 3 Core Advanced の調査要請

- ユーザー：Apple の **AFM3 Core Advanced** 的戦略が取れないか、関連研究込みで知りたい。
- 判明：それはまさにこの方向の「製品版」。20B スパース・1–4B アクティブ・フラッシュ常駐 + プロンプト依存ロード。機構は **IFPruning**（プロンプトで FFN 行/列を動的選択）。**最も高性能な Apple silicon 限定**。
- ★ 核心の分析：**IFP はプロンプト単位でマスク固定（churn 小）→ フラッシュ常駐が成立**。一方 **Qwen MoE はトークン単位ルーティング（churn 大）** → 素朴コピーは難問の方を引き受ける。
- 関連研究を整理：LLM in a flash（土台）、FlashMoE / ProMoE / OD-MoE / HOBBIT / EdgeMoE / Mixtral-Offloading / MoE-Infinity（MoE offloading）、IFPruning（学習路線）。
- 重要な決着材料：**Apple ですら弱チップに載せない** → フロア引き上げを支持。
- 固有の留保：フラッシュ 1GB/s の壁、offloading 研究は CUDA/PCIe 前提で **MLX/unified memory へは方策の翻訳**（移植不可）= 新規性ニッチ。

## 4. 着手順の合意

- 安くて go/no-go を出せる順。エンジンは最後。
- **Step 1**：routing trace 収集（ハード非依存・今日から可）
- **Step 2**：キャッシュシミュレーション（LRU/LFU/Belady、定量ゲート）
- **Step 3**：素の MLX ベースライン + フロア確定
- 詳細は `02-roadmap.md`。

## 5. 確定事項 / 未決事項

**確定：**
- プラットフォーム Apple Silicon / MLX のみ
- 主モデル Qwen3.6-35B-A3B（MoE）、27B は従
- 中核：MoE expert streaming（AFM 戦略の自前 OSS 版）
- フロアは最弱 M1/8GB ではなく引き上げる方向

**未決（実測で詰める）：**
- 正確なフロア機種 + RAM
- 「十分高性能」の 4 つ組定義
- Step 1 のフック（mlx_lm / transformers）
- キャッシュ方策の選定（Step 2 の結果次第）
