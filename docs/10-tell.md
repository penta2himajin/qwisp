# 10 — Tell：Qwisp コアランタイム

> **⚠️ 2026-06-28 更新（最新の正典は `notes/01-speedup-investigation.md`）**：
> - **標準手法 = `suffix-spec`(SuffixSpec, Tell.swift)**。SuffixDecoding draft + **batched f32-full exact verify**(lossless)。
>   既定実行(QWISP_RUN 無指定)はこれ。全領域で SpecK/Fast 以上の **Pareto 最適**（nl 18.8/mix 76+ vs SpecK 16.6/21）。
>   既定 maxK=C×3/8。★但し C×3/8 は真の安全境界でない(2026-07-01): diverse routing で per-layer
>   expert union は ~2×C まで膨張し silent garbage=**lossless-by-luck** だった。**union-overflow guard**(実 union を
>   ensure で観測、overflow prefix を re-verify)で strict-lossless 化。honest 値: 8GB C=64 code 34/mix 43/nl 21,
>   16GB C=128 code 148/mix 62/nl 20, C=192+ ほぼ full, C=256 mix 275/code 217。旧「mix 88/132 @100%」は運の数字。
> - 旧 `spec-verify`(SpecK)・`buddy-no-sync`(Fast) は **TellExperiments.swift へ移管**(QWISP_RUN で利用可、比較基準)。
> - **lossless 単流高速化の探索は出尽くし**: compile/MoE 3bit/A3 融合/tree draft は全て実証否定、nl は batch=1 床(~20 tok/s)。詳細 notes/01。
> - 以下 2026-06-27 の記述(SpecK 基準)は履歴として残置。

> **⚠️ 2026-06-27 後半 大幅更新**：strict-vs-near の決着・命名統一・MLX 基準化により本書の一部結論を訂正。
> - **8GB lossless ベースライン = `spec-verify`(SpecK)**。**「M2 strict / M0 near が天井」は誤り**で、M0/M2 は long horizon で発散＝near-lossless（真 lossless は SpecK のみ）。
> - runner は記述名へ統一（`runHotCold*`/`M0-M6` 廃止）、dispatch は **`QWISP_RUN=<name>`** に集約。
> - lossless 基準は**原典 MLX 準拠**（teacher-forced で確認済: mlx-swift は per-token で MLX と 98.4-100% 一致、不一致は f16 near-tie のみ）。
> - **strict-vs-near と 8GB/16GB の現行結論の正典は `notes/00-strict-vs-near-lossless.md`**。本書 §2/§3/§7 は新名・新結論へ更新済、§4-6（構造的天井・negative・測定較正）は引き続き有効。

> **位置づけ**：Tell は Qwisp の**コアランタイム**（独自スケジューラ＋最適化済みカーネルを内包）。各 `runXxx` は「実験」ではなく**ランタイムの実行戦略（経路）**であり、同時にベンチ入口を兼ねる。
> **命名**：William Tell（一射必中でリンゴを射抜く名手）由来。Apple Silicon ＋ Tell の主題である exact/lossless（正確に当てる）の二重掛け。
> **出典と鮮度**：本書は commit `34c1abb`（2026-06-27 08:27）までのコミットログから再構成。関数名・env・数値は `swift/Sources/QwispCore/Tell.swift` 実体と照合して維持すること。**near-lossless 領域（buddy 等）は現在進行形で、評価が短時間で反転している**（buddy は 07:41 に「starved 救出」→ 08:27 に撤回）。スナップショットとして読むこと。
> **数値の前提**：すべて **dev機（pentasmbp = MacBook Pro）で 8GB 予算を課した測定**。実機 8GB（M1 Air / MacBook Neo）の持続値ではない（§8）。

---

## 1. 設計の核

「**streaming の I/O を GPU 計算の裏に隠す**」ことを主目的にした MLX 上の独自スケジューラ。

- MLX の batched eval を回避し、**chunk 単位 asyncEval** しながら **次 chunk の expert を background prefetch** → I/O を計算に隠す。
- **cross-layer 予測 prefetch**（"Fate one-pass" 相当）を MLX 上で実現。
- 束ねる部品：`StreamingModel`（`runChunk(lo,hi)`、`predictLayerInds`）、`ExpertArena`/`LayerExpertCache`（`captureGateInput`/`captureInds`/`pin`、persistent arena、parallel pread via `concurrentPerform`、GPU slot table）、`GatedDeltaNetLayer`、fused kernel、`MTPHead`、`Speculative`。

---

## 2. 実行戦略（mode）一覧

速度は dev機・本モデル（Qwen3.6-35B-A3B 4bit, 256 experts/top-8/40層）の概算 tok/s（8GB C=64、特記あれば併記）。lossless は **vs Swift-exact**（同一エンジン exact）・long(128tok) 評価。

| `QWISP_RUN=` (旧名) | 機構 | 速度 | lossless(long) | 状態 |
|---|---|---:|---|---|
| **spec-verify** (SpecK) | buddy draft + exact verify（投機） | 27 / C=128 34-36 | **strict** | ★8GB lossless ベースライン |
| **buddy-no-sync** (Fast) | 純 no-sync + buddy 差替（verify 無） | 56-58 | near(@C=64 のみ) | 最速 near（C>64 発散） |
| no-sync-gate-escalate (Hybrid) | no-sync draft + per-token gate escalate | membership 15-17 / margin 36-49 | membership=strict(遅)/margin=near | margin は long 発散 |
| predict-prefetch (M0) | 2-pass 自己予測 prefetch | 28-30 | **near**（long 11%） | 旧「lossless」は誤り |
| cross-layer-predict (M2) | cross-layer 予測 1-pass（Fate） | 26-28 | **near**（long 12%） | 旧「lossless」は誤り |
| mtp-spec-verify (M4) | MTP head 投機 × exact verify | 25 | **strict** | seqMT verify で lossless |
| ss-moe-draft-verify (Spec) | SS-MoE no-sync draft + verify | 20–24 | strict | negative（速度出ず・§5） |
| pipeline-decode (M5) | layer pipeline | 27.4 | — | negative（§5） |
| predict-fixpoint (M6) | routing 不動点 multipass | 7 | strict | negative（§5） |
| cross-layer-cheap (M1) | 軽量 1-pass 予測 | — | — | negative（予測 hidden ズレ） |
| mmap-gather | mmap 全 expert resident gather | 45 / RSS 24GB | — | 24GB 専用（8GB locality 無） |

補助 calib/診断（`QWISP_RUN=`）：`coverage`（top-64 が code 86%/nl 94%）、`miss-coverage`（exact 比 coverage：code static 63.8/online 76.5%、worst layer code 0%/nl 12%）、`predictor-recall`（pre-attention 予測器 82-84%・§4-2）、`skippability`（code 3/40・nl 37/40）、`mlx-fidelity`（teacher-forced で MLX 準拠を確認: hard 100%/long 98.4%）。
- パラメータ env（dispatch でなく挙動制御・据置）：`QWISP_CACHE_C` / `QWISP_GEN` / `QWISP_DRAFT_K` / `QWISP_SKIPMODE=3`(buddy) / `QWISP_MARGIN` / `QWISP_PIN` / `QWISP_CALIB` / `QWISP_SWIFT_REF=1`(vs Swift-exact) / `QWISP_MTP_REF`。

---

## 3. 確定見解と流動部の分離

### 確か（全訂正を生き残った構造的天井）

- **保証付き lossless（8GB）＝ `spec-verify`(SpecK) ≈27 tok/s**（buddy draft + exact verify、long(128tok) でも vs Swift-exact 100%）。**M0/M2 は long horizon で発散＝near-lossless**（旧記の「M2 strict/M0 near が 8GB lossless 天井」は **easy-short-ref artifact による誤り** → notes/00 で訂正済）。16GB は同 SpecK を C=128 へ（34-36, strict のまま）。forward-compute 律速の構造的天井（§4）自体は不変。
- 参考：full-resident（24GB・streaming なし）の素 AR baseline ≈50–54 tok/s、bare forward ≈15ms（≈65 tok/s 上限）。streaming はこの差分を **routing tax** として払う。16–24GB の GPU-routed resident 路線は `docs/09`（別トラック）。

### 流動（near-lossless・直近で評価反転）

- **buddy no-sync ≈58／buddy hybrid 51.6**：自然動作点 **C=64 でのみ 98% near-lossless**。
- **★single-buddy には天井**：残る 2% は「良い single-hot 代替が無い cold expert」由来で、co-activation でも output-similarity でも回収できない。
- **★撤回された主張**：「buddy が starved C（16–24）の no-sync 崩壊を 98% に救出」は **誤り**。Swift-greedy ＋ hard-long-ref で測ると starved C で buddy は **11–12% に崩壊**する。先の 98% は **easy-short-ref vs Python の二重 artifact**。buddy は starved 救済策ではなく、C=64 級でのみ効く near 高速経路。
- 予測可能 prompt（nl）限定：hot-pin no-sync 47（code drift、一般 unsafe）。

---

## 4. なぜ M0/M2 が天井か（構造的理由）

1. **ボトルネックは IO ではない**。M0 overlap で prefetch-wait=0。真因は (a) **per-layer routing sync** と (b) **forward compute 自体**（per token ≈25ms：GDN ≈11ms ＞ MoE-gather ≈8.8ms ＞ MoE-shared ≈3.2ms ＞ attn ≈1.8ms）。inds readback 0.2ms で無視可能。
2. **GatedDeltaNet が cross-layer 予測深度を ≈2層に制限**。M2 は chunk=2 で 20 chunks/token の sync を払う。pre-attention 予測器は GDN ゆえ 82–84% 止まり（標準 attention の論文値 94.69% に届かない）。
3. **mmap GPU-gather は ≈24GB 要る**（expert に locality なし）→ arena+sync が 8GB の正解。
4. **forward-compute fusion 無効**（GDN in_proj / MoE gate+up）。mlx quantizedMatmul は FLOP/memory-bound。
5. **prompt-dependence wall**：予測可能（nl）はどの手法でも速く lossless、敏感（code）は exact 経路に縛られる。coverage でなく sensitivity が drift を決める。
6. M2 の ensure は ≈98% pread IO（55–72 miss/token, ≈97MB/tok @≈17GB/s）。既に persistent arena＋parallel pread＋LRU＋GPU slot で最適化済、mmap でも改善しない。

---

## 5. 潰した negative results（後続の時間節約用）

- **output-similarity buddy（A）**：cold→出力 cosine 最類似 hot へ remap。co-act buddy と同値（C=64 で 98%）、starved では noisy。**buddy 選択は 2% の bottleneck でない** → 次は (B) 再構成/残差補正。
- **M5 pipeline**：27.4＝M2 超えるが M0 に勝てず。
- **M6 multipass**：100% 到達も maxP≈6（avg 5.4 pass）で 7 tok/s。M2 の 1-pass 最適性を補強。
- **SS-MoE / SpecK**：accept 0.94–0.97 lossless だが速度出ず（IO は既に無料、bottleneck は sync で SS-MoE 前提＝IO-bound と不一致）。
- **layer-skip draft**：draft は安いが accept 崩壊（6.0→1.1）。cheap かつ accurate な draft は本設定に無い。
- **partial-resume escalate**：first-miss 層=0（layer 0 が毎 token hot 外）＝共有 prefix なしで無利得。
- **prefetch-verify / async cold prefetch**：residual-miss ~31/tok、warm 空振り（直近 cold=まだ resident、LRU が拾う）。temporal locality は LRU で枯れている。
- **GDN in_proj fusion / no-sync skip(mode1)・renorm(mode2) / M0 top-K prefetch**：いずれも無効 or 有害。

---

## 6. 測定の較正ルール（最重要・本プロジェクトの最大の落とし穴）

参照・テスト条件のズレで結論が**三度**ひっくり返っている：

1. batched verify が f16 で greedy と非一致（~7e-4 累積で argmax 反転）→ `seqMultiToken`（1トークンずつ verify）で bit 一致。原因は full-attention 層のみ（GDN/MoE は bit-exact）。
2. Python-f16-ref が長 horizon で Swift-f16-greedy と発散 → 長 decode 参照に使うな。
3. starved buddy の「98% 救出」は **easy-short-ref vs Python の二重 artifact**。

→ **鉄則：真値は Swift-greedy（同一エンジンの exact, `QWISP_SWIFT_REF=1`）。測定は hard ＋ long ref（128tok 級）。easy-short-ref の高数字は疑う。** 加えて flash 帯域測定前に `sudo purge`、実機は数分連続の定常 tok/s を測る（バースト不可・§8）。

---

## 7. env フラグ早見

**dispatch（どの runner を回すか）**: `QWISP_RUN=<name>`（**無指定＝既定で標準手法 suffix-spec を実行**。旧 spec-verify/buddy-no-sync は明示名で利用可）。`<name>` 一覧は §2、または不正値で起動すると候補が表示される。`main.swift` の `runners` テーブルが実体。

**パラメータ（挙動制御, dispatch とは独立）**: `QWISP_CACHE_C`(cache slot 数) / `QWISP_GEN`(生成長) / `QWISP_DRAFT_K`(投機 draft 長) / `QWISP_CALIB`(calib token 数) / `QWISP_SKIPMODE`（1=skip, 2=skip+renorm, 3=**buddy**）/ `QWISP_MARGIN`（gate しきい, 0=membership）/ `QWISP_PIN`(pin 数) / `QWISP_MTP_REF`(ref パス) / `QWISP_SWIFT_REF=1`（**vs Swift-exact** 評価; long の vs-Python は f16 で無意味）/ `QWISP_M0_SELK`・`QWISP_M0_TAU`（predict-prefetch 選択的 margin）/ `QWISP_M0_TOPK`(neg) / `QWISP_VERIFY_SEQ`・`QWISP_VERIFY_NOSYNC`（verify 制御）/ `QWISP_PARTIAL`・`QWISP_PREFETCH`（gate-escalate の variant）/ `QWISP_MULTI`(cross-layer multi-source) / `QWISP_*_PROF`（profiling）/ `QWISP_BUDDY_OUTSIM`・`QWISP_FUSE_GDN`(neg)。

env 読み出しは `Tell.envInt/envFloat/envStr/envFlag` ヘルパに集約済（Tell.swift）。

---

## 8. 未解決 / 次

- **(B) 再構成・残差補正**：single-buddy の天井（残り 2%）を超える現行の live 方向。single-hot 代替が無い cold expert を、複数 hot の線形結合や残差で近似する路線。
- **A18 Pro（MacBook Neo）で custom Metal kernel（GatedDeltaNet recurrent）が compile/動作するか未検証**。MLX 一般が Neo で動くのは確認済みだが自前 kernel は別物。実機でのみ判定可能。
- **実機 8GB の持続 tok/s 未測定**。本書は dev機の 8GB 予算測定。M1 Air（68GB/s・7–8 GPU）/ Neo（60GB/s・5 GPU・ファンレス throttle）では目減りする見込み。
- 残る guaranteed-lossless レバーは「学習した深い予測器」か「16GB」。8GB の M0/M2 天井自体は forward-compute 律速で動かない見込み。

> 鉄則（再掲）：**参照は Swift-greedy／測定は hard-long-ref／負け筋は negative として残す／数値は機種を明記。**
