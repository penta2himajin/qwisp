# lossless 高速化の探索: 到達点と batch=1 床の証明

作成: 2026-06-28 / 対象: Swift PoC（streaming decode, Qwen3.6-35B-A3B）
前提ノート: `00-strict-vs-near-lossless.md`（lossless 基準＝原典 MLX 準拠、ベースライン系譜）

このノートは「lossless な単流（batch=1）decode を speculation 以外でどこまで速くできるか」を
多角的に実測し、**到達点（verify の刷新で大幅高速化）と原理床（novel text は batch=1 で頭打ち）**を
正典化する。結論を先に:

- **★positive: verify を batched f32-full に刷新 → code/agentic で旧 SpecK の 3.3-4.9x**（8GB 88 / 16GB 132 tok/s, lossless）。
- **★negative: nl(novel/high-entropy) は ~20 tok/s の batch=1 床で原理的頭打ち**。compile/量子化/カーネル融合/tree draft は全て実証的に否定。唯一の道は batching(=speculation, accept 要)で、それは反復content でしか効かない。
- **SuffixSpec が Pareto 最適**（全領域で SpecK 以上）。task 別 dispatch は不要。

---

## 1. verify の刷新（positive, 最大の成果）

### 1-1. matmul は壁でない（investigate C, commit 99c7c23）
spec verify が逐次 greedy と drift する真因を切り分け。micro-test で **quantized matmul / RoPE / rmsNorm /
softmax / GDN updateKernel は全て order-stable(rel=0)**。drift 源は **attention fused SDPA の
`.causal`(L>1) vs `.none`(L=1) 経路差のみ**（matmul の L 依存でなく、verify が draft key も mask 越しに
見る key 数差に由来）。

### 1-2. per-query .none → batched f32-full（commit 77b7b71 → 677ef28）
- **per-query .none**(commit 77b7b71): 射影 batched(order-stable)＋SDPA だけ per-query で exact prefix .none。
  micro-test で量子化射影 rel=0.000＝strict bit-exact。但し seqMT と同等速度で勝たず。
- **★batched f32-full**(commit 677ef28, 既定): verify の divergent op は **attention SDPA と GDN conv1d の 2 つだけ**。
  両者を f32 化すれば verify forward 全体が逐次 decode と bit-exact（micro-test attn=1.08e-6）。
  逐次化(seqMT/perQueryNone)不要の単一 batched forward が **provably lossless かつ最速**。
  `QWISP_F32_ATTN/CONV` 既定1。f16 batched は ~7e-4 drift だが SuffixSpec の reject 自己訂正で実用 lossless(保証なし)。

### 1-3. maxK 上限を C 比例で解放（commit 677ef28）
旧 maxK=4 上限は f16 運頼み回避の保護。f32-full は bit-exact ゆえ撤廃。**真の上限は精度でなく
per-layer cache 容量**: D+1 トークン verify で 1 層が同時に要するユニーク expert 数が C 超で
wrong-slot=silent garbage。実測安全境界 **C=64→maxK24 / C=128→maxK48 = maxK ≤ C×3/8**。
`maxK=min(QWISP_DRAFT_K, C×3/8)` でクランプ(超過時ログ)。

### 1-4. 実測（vs Swift-greedy 100% lossless, commit e79bef7）
| | 8GB C=64 | 16GB C=128 |
|---|---|---|
| mix(反復code) @maxK24/48 | **88** | **132** |
| mix @maxK16(既定) | 76 | 114 |
| nl(高entropy) | ~18-24 | ~29 |

旧 SpecK ~27 比で mix **3.3-4.9x**。

---

## 2. nl(novel) 床の証明（negative, 全 lever を実証否定）

nl で accept が伸びない（suffix 0.23）。その先を speculation 以外で攻められるか網羅検証した。

### 2-1. 壁は streaming でなく forward 自体（commit 5c7caaa, b41abfd）
`QWISP_RUN=forward-cost`（hot-pin で IO 排除, teacher-forced で L 計時）:
- **cost(L) ≈ 50ms + 8.4ms·L**。50ms が forward 1回ごとの固定費。
- misses/forward=0.0（IO でない）、C=64↔128 で不変（streaming でない）。
- `forwardHiddenSkip` 有効層数 sweep: **~1.2ms/層 × 40層 = 50ms の launch/latency chain**。
- C=64 実 decode の追加 IO 税は ~9ms（C=128 ~4ms, PROF2）＝二次的。compute/launch 床は C 非依存
  （8GB の高速化が 16GB に直結）。

### 2-2. 全 lever の実証否定
| lever | commit | 結果 |
|---|---|---|
| **MTP depth-D tree** | e9a6028, d04221a | pre-flight: nl top-3 coverage 83%(top-1 68.8%)・confidence 単調＝tree 前提は成立。但し gating 実測で nl 天井 ~20 tok/s 不変。cost(L)=50+8.4L で accept↑が throughput に変換されず。 |
| **confidence gating** | d04221a | depth-D の無駄回収に有用な安全弁(nl depth4 9.1→19.2)。但し floor は破れず。`QWISP_MTP_GATE`。 |
| **mx.compile** | 2559448 | elementwise-only 3.07x だが matmul 融合不可。実層は matmul 支配ゆえ mixed/quant 0.85x で net マイナス。 |
| **MoE 3-bit 量子化** | e1ed5df | gather qmm で 8bit≈4bit＝この expert スケール(512×2048)は **bandwidth-bound でなく dispatch-bound**。bytes 削減(3-bit)で速くならず。T 大では 3-bit が MLX 汎用 kernel に落ち 0.6x。 |
| **A3 カーネル融合** | ad1ca78, c6adb80 | DispatchBench: 独立 op は MLX が既に overlap、逐次 elementwise のみ ~5µs/op で積算。**既存 GDN 4→1 融合(fuseGDN)は batch=1 で ~2%**。天井低い。 |
| chunkwise/pipeline | (既存 runPipelineDecode) | resident では overlap 対象なし(IO tier 無)。M0 超えず。 |

### 2-3. 床の正体（最終結論）
**50ms = 40 層 × mx.fast-optimal な matmul/gather/recurrent kernel の batch=1 latency-bound 実行**。
各 matmul は weight 全体を読むのに 1 行しか計算せず GPU を underutilize するが、kernel 自体は既に最適。
これは融合・compile・量子化で動かない。**唯一 latency-bound matmul を効率化する道 = batching(=speculation)**で、
それは accept を要し反復 content でしか効かない。∴ **nl(novel)は batch=1 床で原理的に ~20 tok/s 頭打ち**
（MLX/Apple Silicon の物理であり実装不足でない）。

---

## 3. 手法比較: SuffixSpec は Pareto 最適（task 別 dispatch 不要）

同一 ref, C=64, vs Swift-greedy 100% lossless:
| 手法 | nl(novel) | mix(反復) | draft コスト |
|---|---|---|---|
| SpecK(runSpecVerify) | 16.6 | 21.4 | K=4 回 no-sync 自己 forward(高) |
| **SuffixSpec(既定)** | **18.8** | **76.2** | suffix lookup(0) |
| MTP-spec gate0.7 | 18.4 | — | MTP head(中) |

**SuffixSpec が nl/mix とも SpecK 以上**＝回帰なし。draft が無料ゆえ反復では長 draft(76)、novel では
greedy フォールバック(18.8)に自己適応。**task 別に run を振り分ける仕組みは不要**（SuffixSpec が全領域で
支配的で、dispatch は複雑性のみ増やす）。旧「SpecK 27 tok/s」は反復プロンプトの数字で、nl の ~18 は
全手法共通の床。

---

## 4. 残された軸（lossless 単流高速化は出尽くし）

- code/agentic(workload の ~90%)は SuffixSpec で 3.3-4.9x 達成済。nl(~10%)は原理床。
- 次は製品軸: code/agentic 堅牢化（多様プロンプトでの安定性・edge case）、full-resident C=256(24GB アンカー)、
  品質検証拡充、GitHub issue #1(MTP, depth-D 化で実装済)・#2(verify, f32-full で解決済)のクローズ整理。

## 再現
全ベンチは `QWISP_GDN_TTEST=1 qwisp-poc`（CompileBench/ExpertBitBench/DispatchBench, モデル不要）または
`QWISP_RUN=<name> qwisp-poc stream`（forward-cost / mtp-draft-calib / suffix-spec / mtp-spec-verify /
spec-verify）。ref 生成は `qwisp/mtp_ref.py`（`QWISP_REF_PROMPT` で nl/mix/code を切替、`QWISP_MTP_REF` で指定）。
評価は `QWISP_SWIFT_REF=1`（vs Swift-exact-greedy。長 horizon の vs-Python は f16 自己回帰発散で無意味）。
