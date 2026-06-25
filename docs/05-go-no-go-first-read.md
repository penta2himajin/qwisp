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

## 次（Step 3）

1. **実効 flash 帯域の実測**（1.6MB ランダム読み、cache bypass）→ ゲート再較正（最大の仮定を実数化）。
2. **素の MLX ベースライン**（フロア機での fit / peak mem / prefill・decode tok/s × context 長）＝「超えるべき基準値」。
3. 上記でゲートを引き直し → 4つ組（[[qwisp-open-decisions]]）を確定へ。

> 再現: `tools/step1_routing_trace/collect_traces.py`（trace）→ `tools/step2_cache_sim/simulate.py`（sim）。trace/生成プロンプトは gitignore、再現は manifest+lock。
