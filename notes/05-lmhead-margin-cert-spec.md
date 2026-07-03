# lm_head margin-certified 2-bit — 形式仕様 / 検証可能ゴール定義

**Author (goal owner): Fable/Opus.** Branch `feat/raw-verify`. This document is the *contract*.
実装(GLM-5.2 via `glm-code`)・テスト著者(Sonnet)・敵対的レビュー(Sonnet)・build/test 実行と最終確認
(Fable/Opus 直接 Bash)は全て「§6 受け入れゲート」を唯一の合否基準とする。曖昧点は Fable に差し戻す。

---

## 1. ゴール(一文)

resident raw fused decode/verify の lm_head(4-bit, ~273MB/step read)を、**margin-certified 2-bit
先読み + 未確定時のみ 4-bit fallback** に置換し(opt-in `QWISP_LMHEAD2=1`、既定 off)、**greedy argmax
を 4-bit 経路と bit-exact に保ったまま**(strict lossless)lm_head read bytes を確定時 ~157MB へ削減する
(全体 1.23GB の ~9% 削減上限、cert 率依存)。

## 2. 背景(実測済み事実)

- lm_head = `[V=248320, H=2048]` 4-bit affine gs=64、**weight 242.5MB + scales/biases 30.3MB = 272.8MB
  resident**、embed と非共有(RawEngine.swift ~140)。
- fused 経路(RawFusedVerify.swift `encodeFinalOps` ~1776)は常に full logits `[M,V]` f16 を materialize
  → `argmax_rows`(~319)で token id へ。greedy に不要な全 logits を毎回計算。
- lm_head は M=1 step の最大単一 op(qmv 化前 40%)。近似・shortlist 系コードは現状ゼロ。
- Ω(N·K) 定理により**無条件の lossless 削減は不可能** → 条件付き(cert)削減が唯一の strict 経路。

## 3. 機構(margin certification)

### 3.1 追加重み(RawEngine build 時に一度だけ構築)
- `W4` = dequant(lmW, lmS, lmB)(f32, 計算のみ・保持しない)。
- `(lmW2, lmS2, lmB2)` = `MLX.quantized(W4, groupSize: 64, bits: 2, mode: .affine)` → resident
  (~127MB + 30.3MB)。**2-bit は「4-bit dequant 結果」の量子化**(原重みでなく)— 誤差ベクトルが
  4-bit 経路との差そのものになるため。
- `rowE[v]` = ‖dequant2(v) − dequant4(v)‖₂(f32→f16 保存, `[V]`)。per-row 誤差ノルム。
- 追加 resident 合計 ~158MB(24GB dev 機で問題なし。8GB streaming tier では負担 → 本機能は
  **resident-tier 専用**。streaming との併用は将来判断)。

### 3.2 per-step certification(kernel 連鎖、1-CB 内・CPU sync 無し)
h = final normed hidden(行 m)。数学:

  |logit2(v) − logit4(v)| = |⟨dq2(v) − dq4(v), h⟩| ≤ rowE[v] · ‖h‖₂   (Cauchy–Schwarz)

kernel 連鎖(既存 lm_head+argmax の置換、`QWISP_LMHEAD2=1` 時のみ):
1. `hnorm_rows`: ‖h_m‖₂ を f32 で(`[M]`)。
2. `qmm2_rows`: 2-bit qmv で logits2 `[M,V]`(f32 accumulate、f16 store — 既存 qmm4 と同 idiom)。
3. `argmax_cert_rows`: 行 m ごとに 1 pass で
   - `a_v = logits2[v]` の top-1(値と idx = v*)
   - `b_v = logits2[v] + rowE[v]·‖h‖₂ + ε` の top-2(idx 付き)
   を reduction で求め、challenger = (b_top1.idx ≠ v*) ? b_top1 : b_top2 とし、
   **cert 条件: `a_{v*} − rowE[v*]·‖h‖₂ − ε > challenger`** なら `tokensOut[m] = v*`, `certFlag[m]=1`;
   さもなくば `certFlag[m]=0`。
   - **ε = 0.05(f32/f16 丸め・累積順差の安全 slack、定数)**。数学 bound は実数演算前提のため、
     kernel 間 fp 差を ε が支配することを G1 乱数テストで実証する(違反 0 件)。
   - cert 成立時 strict inequality ゆえ 4-bit 側の first-index tie-break と衝突しない。
4. `qmm4_rows_fallback`: 既存 4-bit qmv と同一計算だが threadgroup 冒頭で `certFlag[m]==1` なら即 return
   (weight を読まない)。uncert 行のみ logits4 を logits バッファへ上書き。
5. `argmax_rows_fallback`: 既存 argmax_rows + certFlag early-out。uncert 行の `tokensOut[m]` を
   4-bit logits の argmax で上書き(= **fallback 行の token は文字通り既存 4-bit 経路の産物**)。
6. telemetry: `certCount` device カウンタ(atomic add)を step 毎に累積、run 終了時に
   `[RawSpec] lmhead2 cert-rate: X/Y=Z%` を print(RawSpecRunner)。

### 3.3 正しさ論拠
- **fallback 行**: 出力 = 既存 4-bit kernel の argmax そのもの → 自明に bit-exact。
- **cert 行**: bound + ε が logit 差を支配する限り argmax2 = argmax4 が数学的に成立。fp 安全性は
  ε slack + G1 大規模乱数検証(cert=1 の全行で 4-bit argmax と一致、violation 0)+ G2 実重み
  identity oracle で三重に固定。
- 既定 off(`QWISP_LMHEAD2` 未設定)は全 byte 不変。`engine.logits()`(prefill 初token/self-check)は
  4-bit のまま触らない(両モードで共通 → identity に影響しない)。

## 4. 実装制約

- 変更ファイル: `RawFusedVerify.swift`(kernel 3 本 + encodeFinalOps 分岐 + HeadBufs 拡張)、
  `RawEngine.swift`(2-bit copy + rowE 構築)、`RawSpecRunner.swift`(telemetry print のみ)。
- **既存 kernel・既定経路は byte 不変**。RAWTESTS 既存 31 本 PASS 維持。
- 禁止: テスト改変・閾値緩和・ε の増減による「テスト合わせ」(ε は 0.05 固定。変えたくなったら
  Fable に差し戻し=bound の欠陥を意味する)。
- noCopy 寿命罠(asType 一時 array retention)厳守。2-bit 構築は mx.eval 後に MTLBuffer 化。

## 5. 作業ループ(今回の役割分担 — Haiku→GLM-5.2 に組み替え)

1. **Fable/Opus**: 本仕様(完了)。
2. **Sonnet(テスト著者, Agent)**: §6-G1 のテストを `RawVerifyTests.swift` に追加(31→34)+
   `qwisp/test_lmhead2.sh`(G2 実行スクリプト)。完成後 lock dir にスナップショット(書換不可)。
3. **GLM-5.2(実装, `glm-code` — shell 不可・read/edit のみ)**: §3 を実装。
   **build/テスト実行は Fable が直接 Bash で行い、エラーを GLM に差し戻す**(GPU 排他も Fable 管理)。
4. **Sonnet(敵対的レビュー, Agent)**: テスト完全性(lock 照合)・§3 忠実性(特に cert 不等式の向き、
   ε 定数、challenger の top-2 処理、fallback early-out が weight を読まないこと)・G1/G2 実測。
   必要修正は GLM へ差し戻し(軽微なら Sonnet が直接修正可)。PASS まで 3↔4 ループ。
5. **Fable/Opus(最終 gate)**: G1–G5 を直接 Bash で再実行し bit で確認 → owner へ commit 可否。

## 6. 受け入れゲート(唯一の合否基準)

### G1 — RAWTESTS(書換不可・Sonnet 著、31→34)
- **lmhead2_cert_soundness**: 乱数 W(f16)→ 4-bit/2-bit 量子化 + rowE。乱数 h 多数(≥10⁴ 行相当、
  near-tie を含む adversarial 分布も混ぜる)で cert kernel を実行し、**certFlag=1 の全行で
  tokensOut == 4-bit full argmax(bit)**。violation は 1 件でも FAIL。cert 率も print(参考)。
- **lmhead2_fallback_exact**: 人工的に margin を潰した(rowE を巨大化 or near-tie h)入力で
  certFlag=0 経路を強制し、tokensOut == 既存 argmax_rows(4-bit logits)と bit 一致。
- **lmhead2_step_identity**: 実 kernel 連鎖 stepArgmax 相当で `QWISP_LMHEAD2` on/off の tokensOut が
  M∈{1,8,17} で byte 一致(乱数重み)。
- 期待: `RAWTESTS 34/34`。

### G2 — 実重み identity oracle(最重要・偽造不能)
4 regime × GEN=128、resident fused(`QWISP_RAW_FUSED=1`):
`QWISP_LMHEAD2=1` の OUT_TOKENS == `QWISP_LMHEAD2=0` の OUT_TOKENS(**128/128 byte-identical**)かつ
self-check 128/128 LOSSLESS。cert-rate telemetry を全 regime で記録(cert 率自体は gate でないが必須報告)。

### G3 — 非退行
`QWISP_LMHEAD2` 未設定で出力・速度とも従来と不変(RAWTESTS 34/34 の既存 31 本含む)。

### G5 — 速度(★ハードゲート・go/no-go、A3 の教訓により最初から)
clean interleaved warm(単独プロセス)で:
- **go**: {code, agentic, shortnl} の平均 tok/s が baseline 比 **≥ +3%** かつ 全 regime で −2% 未満の
  退行なし(longctx は verify M 大で lm_head 償却済みのため参考値)。
- **no-go**: 未達なら opt-in 温存 or 撤去を owner 判断へ(finding として cert 率と bytes 削減実測を記録)。
- 期待メカニズム: cert 時 lm_head read 273→157MB ≈ 全体 bytes −9%。M=1 memory-bound ゆえ
  cert 率 ~100% なら +5〜9% が理論上限。cert 率が低ければ 2-bit 読みが純増で負ける — それを G5 が検出。

## 7. 環境(確定情報)

- BIN/build/refs/RAWTESTS/model: notes/04 §8 と同一。baseline(fused resident, 2026-07-04 clean):
  code 68.9 / agentic ~57.4 / longctx ~117.5 / shortnl ~31?(要再測、gate 時に interleaved で取り直す)。
- GLM 呼び出し: `GLM_DIR=/Users/penta2himajin/repos/qwisp zsh -ic '~/bin/glm-code "<指示>"'`
  (shell 不可設定済み。build/テストは Fable 側)。
- multi-session: push 前 `git pull --ff-only origin feat/raw-verify`。auto-commit 禁止(owner gate)。

---

## 8. 最終判定(Fable gate, 2026-07-04)— 手法 A/B とも NO-GO(数学的閉路)

**プロセス**: Fable 仕様 → Sonnet 書換不可テスト(31→35, cert 正例 lmhead2_cert_fires 含む)→
**GLM-5.2(glm-code, shell 無し)が Round A/B を実装**・Fable が build/test 駆動(build error 2件を
差し戻し修正)→ RAWTESTS 35/35(本番 GPU kernel 連鎖を lmhead2CertStep 経由で直接 gate)。
実装は完全に正しい: code regime で identity IDENTICAL・128/128 LOSSLESS。

**しかし cert-rate = 0/128 = 0.0%、速度 66.9→44.5 tok/s(−33%)。**

**定量診断(qwisp/diag_lmhead2.py, 実重み+実 h 64 token)**:
- ‖h‖₂≈112、margin4 mean 5.44。rowE2[v*] mean 0.394 → **C-S bound mean 44.2 vs 実誤差 mean 1.70 =
  緩さ mean 102×(min 8.3×)**。誤差ベクトルと h は高次元でほぼ直交=√H≈45 の iid slack が実測で確認。
- per-group C-S ~22×・L1/L∞ 更に悪化 → **いかなる標準 norm bound でも cert は発火しない**(実誤差の
  ~1.5× の sound bound が必要=射影を知ることと等価で原理矛盾)。仮想 tight bound(2×実誤差)でも 37.5%。
- 3-bit: bound 23.3 vs 誤差 0.67 → cert 0%。plain argmax 一致 2-bit 85.9% / 3-bit 92.2% =
  bolt TF fidelity(87.7-96.5%)未満で near-lossless トラックとしても不採用。
- **★bit-plane 2+2(C 設計)への波及: plane 残差 cert も構造同一の壁(最低 8-13×)→ strict 変種は
  不成立。C は near-lossless(L3)変種のみ生存**。

**総括**: 「lm_head 18% 削減」の strict 経路は本診断で**閉路**。Ω(N·K) weight-read floor
([[lean-lossless-limits-proven]])の具体的発現: dense lm_head のデータ依存 skip は sound bound の
緩さ(≥8×)が margin(O(5))を常に上回り不可能。残る lm_head 削減は (i) 語彙 shortlist 等の
near-lossless 設計 (ii) 出力層の低bit 蒸留/再学習(モデル変更)のみ=どちらも strict 圏外。

**disposition(owner 判断待ち)**: 実装(kernel 5本+配線, 全35テスト green)は正しいが on で有害
(−33%)・off で死重(hottest file に~600行)。repo 前例(B1 −13% 差戻し / A5 パリティ差戻し)に従い
**「commit(記録+kernel 資産温存)→ 直後に revert commit」を推奨**(git 履歴に検証済み 2-bit qmv /
flagged early-out / top-2 cert reduction kernel を保全しつつ、working tree は clean に)。
notes/05 と diag_lmhead2.py は残す。
