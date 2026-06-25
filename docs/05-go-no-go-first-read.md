# 05 — go/no-go 初回実測（Step 1→2 一次読み）

実施 2026-06。Qwisp 初の go/no-go 定量判定。**結論：GREEN（routing/キャッシュ側は GO、streaming エンジンを作る賭けが正当化された）。** ただし systems 工学（MLX mmap-from-NAND）は未検証で「動いた」ではなく「作ってよい」。

## セットアップ

- モデル：`Qwen3.6-35B-A3B-MTPLX-Optimized-Speed-FP16`（4bit MLX、256 experts / top-8 / 40層）。
- プロンプト：ベンチ由来 40 件（SWE-bench Verified / LiveCodeBench / BFCL / Terminal-Bench 2.0 / RULER / RepoQA、比率 45/45/10。`tools/prompt_builder/`）。
- trace：AR モード × `max_new_tokens=128`、**2,294,400 行**（長文脈 prefill 込み、prompt_len 最大 10.6K）。
- キャッシュモデル：層ごと独立、予算 B=常駐 expert 数/層。decode hit を主指標。
- ゲート仮定（**要 Step 3 実測**）：expert 1.6MB（4bit）/ flash **1GB/s**（保守値）/ target 15 tok/s（→ 66.7ms/token 予算）。

## 結果

### LRU（実現可能方策の下界）

| B/層 | expert DRAM | decode hit | +lat/tok | gate |
|---:|---:|---:|---:|:--|
| 16 | 1.0G | 0.437 | 288ms | no |
| 32 | 2.0G | 0.593 | 209ms | no |
| 48 | 3.1G | 0.688 | 160ms | no |
| 64 | 4.1G | 0.759 | 124ms | no |
| 96 | 6.1G | 0.857 | 73ms | no |
| **128** | **8.2G** | **0.916** | **43ms** | **GO** |
| 256(全載) | 16.4G | 0.987 | 6.6ms | GO |

### Belady（oracle 上界）

| B/層 | expert DRAM | decode hit | +lat/tok | gate | LRU比 |
|---:|---:|---:|---:|:--|---:|
| 64 | 4.1G | **0.883** | 60ms | **GO** | +0.124 |
| 128 | 8.2G | 0.965 | 18ms | GO | +0.049 |

## 主要な発見

1. **routing に明確な局所性がある**。256 均等分散でない（全載りで compulsory miss わずか 1.3%）。キャッシュが効く前提を *測って* 確認＝GO 方向の最大の収穫。
2. **素の LRU でも保守的 flash 1GB/s で B=128/層（8.2GB）で 15 tok/s GO**。
3. **賢いキャッシュは GO 予算を半減**：Belady の GO 閾値は **B=64/層（4.1GB）** ＝ LRU の半分。B=64 は LRU で NO-GO(123ms) → Belady で GO(60ms)。LRU→Belady の +0.124 がちょうど GO ラインをまたぐ。FlashMoE 等の ML キャッシュ（LRU/LFU 比 +51%）は Belady に肉薄 → **予測器への投資価値が定量化**（[[step1-mlx-hook]] の C節研究が効く証拠）。

## reach の実数（hot 常駐＋cold ストリーム）

非 expert 必須常駐 ≒ 全20GB − 全expert16.4GB ≒ **3.6GB**。＋KV/context は別途。

| 構成 | 常駐（expert＋非expert） | ストリーム | 収まる機 |
|---|---:|---:|---|
| 全載り（streaming無し） | 20GB | 0 | 24GB |
| LRU streaming GO | 8.2＋3.6 ≒ 11.8GB | 8.2GB | **16GB 級** |
| smart-cache GO（Belady級） | 4.1＋3.6 ≒ 7.7GB | 12.3GB | **12GB 級も射程** |

→ 「24GB native」→「16GB は LRU、12GB は賢いキャッシュ」。混合精度 expert・低量子化でさらに下。

## 留保（希望の質を保つ）

- **flash 1GB/s は保守値**。実効が速ければ全方策で GO 予算が下がる。**最重要の未検証仮定 → Step 3 で実測**。
- **MLX mmap-from-NAND streaming は未検証**＝最難関かつ新規性の核（docs `01` E-2）。シミュ GO は工学の成功を保証しない。
- oracle は上界。実 ML キャッシュは gap の大半を取るが全部ではない。
- 5,120 decode トークン（40×128）の一次読み。decode 長・プロンプト数を増やす余地。

## 実測較正（Step 3 結果、2026-06）

`tools/step3_baseline/` で実測。仮定を実数に置換した。

**実効 flash 帯域**（`bench_flash.py`、F_NOCACHE、purge 後の cold）: 1.6MB ランダム読み = **約 4.0 GB/s**（保守仮定 1GB/s の **4.18×**）。チャンク大で更に速い（2MB で 5.6GB/s＝bundling 効果）。※測定前に `sudo purge` 必須（さもないと RAM キャッシュ served で 15GB/s と過大評価される）。

**素の AR ベースライン**（`bench_mlx.py`、4bit）: decode **50-54 tok/s**、peak **19.8-21.9GB**（@8K）。→ 全載りフロア ~22GB＝24GB 機。target 15 tok/s は保守的。

**gate の物理修正**: 旧 gate は追加レイテンシを全予算と比較し compute 時間を二重計上していた。`--baseline-tok-s` を追加し `net_tps = 1/(1/baseline + miss_lat)` を直接出すよう修正（simulate.py）。

**再較正後 go/no-go**（flash 4.18GB/s 実測、baseline 54、target 15）:

| policy | GO 閾値 | expert DRAM | 旧（1GB/s, 二重計上）|
|---|---|---:|---|
| LRU | **B=48** | 3.1GB | B=128 / 8.2GB |
| Belady | **≤B=32** | ≤2.0GB | B=64 / 4.1GB |

net_tps（実 tok/s）例: LRU B=48→17.6, B=64→20.8, B=128→34.7 / Belady B=48→25.8, B=64→30.4。

**実測 reach マップ**（非expert＋KV@8K ≒ 5.5GB を加算）:

| 構成 | 予算 | 常駐 | 機 |
|---|---|---:|---|
| 全載り | 256 | 21.9GB | 24GB |
| LRU 15tps | B=48 | 8.6GB | **12GB** |
| Belady 15tps | ≤B=32 | ≤7.5GB | 12GB 余裕 |
| LRU 30tps | B≈110 | ~12.5GB | 16GB |
| Belady 30tps | B=64 | 9.6GB | 12GB タイト |

→ **実測帯域で素の LRU streaming が 35B-A3B を 15 tok/s で 12GB 機に収める。** 当初 24GB から大幅拡大。

**量子化の知見（重要）**: unsloth UD-MLX-3bit はメモリ -3GB だが decode -35%（54→35 tok/s。3bit dequant 重・4bit 側が MTPLX 速度最適化済）。**一律低bit は「軽く速く」にならず速度予算を食う**。真のサイズ落としレバーは**混合精度 expert（hot 高bit／cold 低bit）**＝streaming エンジン内で実装（docs `01` C節 HOBBIT/EdgeMoE）。

## 次（Step 4 へ）

routing/キャッシュ側＋実帯域で GREEN 確度が上がった。残る最難関は **MLX mmap-from-NAND streaming の工学**（docs `01` E-2、未検証＝「作ってよい」段階）。
1. expert/非expert 分離ロード（非expert 常駐）。
2. MLX + mmap-from-NAND のキャッシュ方策実装（Step 2 で勝った方策＝予測器寄りを翻訳）。
3. 混合精度 expert で「サイズ落とし」を速度を殺さず実現。
4. 4つ組（[[qwisp-open-decisions]]）を実測値で確定：例 `{35B-A3B 4bit, ≥15 tok/s, 8K ctx, 12GB(M?)}`。

> 再現: `tools/step1_routing_trace/collect_traces.py`（trace）→ `tools/step2_cache_sim/simulate.py`（sim）。trace/生成プロンプトは gitignore、再現は manifest+lock。
