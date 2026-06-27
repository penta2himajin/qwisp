# 10 — Tell：Qwisp コアランタイム

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

速度は dev機・本モデル（Qwen3.6-35B-A3B 4bit, 256 experts/top-8/40層）の概算 tok/s。

| mode | 入口(env) | 速度 | lossless | 適用域 | 状態 |
|---|---|---:|---|---|---|
| **M2** | （1-pass 経路） | ~27 | **strict (100%)** | 8GB 汎用 | ★採用（保証 lossless） |
| **M0** | 既定 / `QWISP_ONLY_M0` | ~30 | **near (98%)** | 8GB 汎用 | ★採用（near） |
| buddy no-sync | `QWISP_SKIPMODE=3` | ~58 | near (98% **@C=64 のみ**) | 自然動作点 | near 候補（starved C は不可・§3） |
| buddy hybrid | `SKIPMODE=3`+`QWISP_MARGIN≥2` | 51.6 | near (+5% exact escalate 保険) | 8GB | 候補 |
| hot-pin no-sync | `QWISP_HOTCOLD_FAST` | 47 | nl 100% / code 22% | 予測可能 prompt 限定 | 一般には unsafe |
| hybrid (membership/margin) | `QWISP_HOTCOLD_HYBRID`/`ONLY_HYBRID` | ~53 | margin=near / membership=strict | 8GB | buddy に発展 |
| output-sim buddy | `QWISP_BUDDY_OUTSIM=1` | — | — | — | **negative**（co-act 比で利得なし） |
| SS-MoE SpecK | `QWISP_HOTCOLD_SPEC` | 20–24 | strict | — | negative（速度出ず・§5） |
| M5 pipeline | — | 27.4 | strict | — | negative（§5） |
| M6 multipass | `QWISP_M6` | 7 | strict | — | negative（§5） |
| mmap gather | `QWISP_MMAP_GATHER` | 45 / RSS 24GB | — | 24GB 専用 | 8GB に locality なし |
| prefetch-verify | `QWISP_PREFETCH` | 23.7 | near | — | frontier 拡張せず |

補助 calib：`runHotColdCalib`（top-64 が code 86%/nl 94%）、`runHotColdDiag`（exact 比 coverage：code static 63.8/online 76.5%、nl 86.6/89.7%、worst layer code 0%/nl 12%）、`runPredictorCalib`（pre-attention 予測器、§4-2）、`runSwiftCalib`（skippability：code 3/40・nl 37/40）。

---

## 3. 確定見解と流動部の分離

### 確か（全訂正を生き残った構造的天井）

- **保証付き lossless（8GB）：M2 ≈27 tok/s（strict 100%）／M0 ≈30 tok/s（near 98%）**。これが forward-compute 律速で決まる構造的天井（§4）。
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

`QWISP_ONLY_M0` / `QWISP_M0_SELK`・`QWISP_M0_TAU`（選択的 margin prefetch）/ `QWISP_M0_TOPK`（neg）/ `QWISP_M6` / `QWISP_SKIPMODE`（1=skip, 2=skip+renorm, 3=**buddy**）/ `QWISP_BUDDY_OUTSIM`（neg）/ `QWISP_MARGIN` / `QWISP_ONLY_HYBRID` / `QWISP_PARTIAL` / `QWISP_PREFETCH` / `QWISP_MMAP_GATHER` / `QWISP_HOTCOLD_{CALIB,FAST,DIAG,ONLINE,ADAPT,SPEC}` / `QWISP_SWIFT_REF`・`QWISP_SWIFT_CALIB` / `QWISP_VERIFY_SEQ`・`QWISP_VERIFY_NOSYNC`・`QWISP_BATCHED_VERIFY` / `QWISP_M2_PROF`・`QWISP_M2_PROF2`（profiling）/ `QWISP_FUSE_GDN`（neg）。

---

## 8. 未解決 / 次

- **(B) 再構成・残差補正**：single-buddy の天井（残り 2%）を超える現行の live 方向。single-hot 代替が無い cold expert を、複数 hot の線形結合や残差で近似する路線。
- **A18 Pro（MacBook Neo）で custom Metal kernel（GatedDeltaNet recurrent）が compile/動作するか未検証**。MLX 一般が Neo で動くのは確認済みだが自前 kernel は別物。実機でのみ判定可能。
- **実機 8GB の持続 tok/s 未測定**。本書は dev機の 8GB 予算測定。M1 Air（68GB/s・7–8 GPU）/ Neo（60GB/s・5 GPU・ファンレス throttle）では目減りする見込み。
- 残る guaranteed-lossless レバーは「学習した深い予測器」か「16GB」。8GB の M0/M2 天井自体は forward-compute 律速で動かない見込み。

> 鉄則（再掲）：**参照は Swift-greedy／測定は hard-long-ref／負け筋は negative として残す／数値は機種を明記。**
