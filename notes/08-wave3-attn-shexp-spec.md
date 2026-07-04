# fusion wave 3: attn 層 + MoE shared expert — 形式仕様

**Author (goal owner): Fable.** Branch `feat/raw-verify` @a8f86d9. Contract = 本書 §4。
前提 = notes/06(PoC/physics)+ notes/07(GDN wave1/2、§6-8 の教訓: unit green≠配線・歪んだ oracle・
「テストと本番の矛盾は STOP して報告」)。ループ: grounding 検証 → Sonnet 書換不可テスト →
Sonnet 実装 → Sonnet 敵対レビュー → Fable gate(独立実測)。

## 1. ゴール
- **attn 層(10 層 × core 13 dispatch)を ~6 に**: `QWISP_FUSE_ATTN=1` opt-in。
- **MoE shared expert(40 層 × 6 dispatch)を 3 に**: `QWISP_FUSE_SHEXP=1` opt-in。
- 削減 ~−190 本/forward(理論 ~+420µs)。全段 bit-exact、既定 off で byte 不変。

## 2. 現状(grounding、workflow 第1段で要再検証 — 相違があれば STOP して報告)
attn(encodeAttnLayerRows ~951-982、F5 で resid+post は融合済み): ②qmm q(w.qW→sc.qOut、gate 込み
[M, numHeads*2*headDim])③qmm k→sc.kOut ④qmm v→sc.vOut ⑤extract_q(qOut→qX)⑥rmsnorm q(qX→qN,
w.qNorm)⑦rmsnorm k(kOut→kN, w.kNorm)⑧rope q(qN→qRot)⑨rope k(kN→kRot)⑩write_kv k ⑪write_kv v
⑫sdpa ⑬sigmoid_mul(attnOut⊙sig(qOut gate)→gated)⑭qmm o。
shared(encodeMoESharedRows ~642-650): ⑨qmm shG ⑩qmm shU ⑪swiglu ⑫qmm shD ⑬qmm8 sgl
⑭final_combine(y+sigmoid(sgl[:,0])·sharedY)。

## 3. 融合原子
### attn(`QWISP_FUSE_ATTN=1`)
- **A1 qkv demux in-proj(②③④→1)**: GDN F1 demux の再利用パターン。q/k/v 重みを N 軸連結し
  1 demux qmm で sc.qOut/kOut/vOut へ書き分け(境界 8 整列を検証)。−2。
- **A2 q-prep(⑤⑥⑧→1)**: per-(m,head) 1 threadgroup: qOut から extract → rmsnorm(既存 tree 厳守、
  w.qNorm)→ rope(既存 rope_rows の式・角度計算を厳密再現、startOffset 対応)→ qRot。−2。
- **A3 k-prep(⑦⑨⑩→1)**: rmsnorm(k)→rope(k)→write_kv(cache scatter を既存 write_kv_rows と同一に)。−2。
  (⑪write_kv v は単独維持。)
- **A4 sdpa+sigmoid_mul(⑫⑬→1)【stretch】**: sdpa の出力書き込みで sigmoid(qOut gate) を乗算。
  sdpa kernel は複雑ゆえ**bit 一致が数回で取れなければ A4 は撤退可**(gate は A4 無しでも成立するよう設定)。−1。
- 到達: 13 → 7(A4 込み 6)。
### shared(`QWISP_FUSE_SHEXP=1`)
- **S1 shG+shU+swiglu(⑨⑩⑪→1)**: fuseGU の plain-qmm 版(gather 無し)。**register 圧迫の前例に従い
  最初から M==1 分岐**(fuseSHEXPActive(M)= flag && M==1)。−2。
- **S2 sgl+final_combine(⑬⑭→1)**: final_combine kernel 内で sgl の 8 dot(K=H)を自前計算(qmm8 の
  8bit dequant 式を厳密再現)→ sigmoid → combine。−1。
- 到達: 6 → 3。

## 4. 受け入れゲート
- **G1**: 書換不可テスト(lock dir `$CLAUDE_JOB_DIR/tmp/locked5`、参照=本番 kernel のみ・MLX 演算での
  参照代用禁止 — wave2 の 1 ULP 教訓)。原子ごと bit-exact + M 不変。RAWTESTS 41 → 41+N 全 PASS。
- **G2**: 各 flag 単独 + 全 flag 同時(FUSE_GU+GDN+ATTN+SHEXP)vs 全 off で OUT_TOKENS byte-identical +
  128/128 LOSSLESS(code + longctx)。
- **G3**: flag off 全 byte 不変。
- **G5(ハード)**: paired A/B prof に fuseATTN/fuseSHEXP lane を追加し(pairedAB 呼び出し 2 行)、M=1 median:
  **fuseATTN ≥ +80µs かつ fuseSHEXP ≥ +120µs かつ 合計 ≥ +280µs**(A4 撤退時は attn 理論 −60 本=130µs
  の 60% で ≥ +80µs のまま)。M=8 lane で median ≤ −200µs の退行が出た原子は M==1 分岐で救済(S1 は最初から)。
- **配線の真実は G5**(unit green で満足しない)。実装は「1 原子 = 1 dispatch」。テストと本番 kernel の
  矛盾を見つけたら実装を曲げず STOP して報告。
## 5. 環境
notes/04 §8 と同一。BIN=swift/.xcode-build-rel/Build/Products/Release/qwisp-poc。baseline(2026-07-04):
code 全off 71.5 / 全on(GU+GDN) 78.2 tok/s。RAWTESTS 41/41 @a8f86d9+wave2 コミット済。
commit は Fable のみ(実装 agent の commit 禁止)。GPU 排他=同時に 1 build/model job。

---

## 6. 最終判定(Fable gate, 2026-07-04)— 採用(I期の尾、+~0.2ms)+ 重要 findings 2 件

**正しさ**: G1 46/46(lock 照合済・参照=本番 kernel)/ G2 全 flag(GU+GDN+ATTN+SHEXP)vs 全 off
IDENTICAL + 128/128 LOSSLESS(longctx も implementer smoke で確認)。A4(sdpa epilogue)は規定どおり skip。

**速度(paired A/B, M=1 median)**: fuseSHEXP **+181µs**(S2 修正後)/ fuseATTN +11〜+151(≈+50-100µs 級, σ大)。
wave 3 計 ≈ **+0.2ms ≈ +1.6%**。§4 gate(+280µs)には未達だが全原子が正または中立・opt-in default-off の
ため採用。**dispatch 融合鉱脈は逓減域 = I期はここで完了**。

### ★finding 1: S2 の直列化バグ(fused kernel の grid 設計原則)
初期実装は grid (M,1)×64 threads で combine(H=2048)を 64 thread 直列化 → **−384µs 退行**。
combine が読むのは sgl[row 0] の 1 dot のみと判明 → grid (M, H/256)×256 に再設計、各 tg が row-0 dot を
既存演算順で冗長計算(2048 MAC=無視可)→ combine 全並列 → **+181µs に正転**。
**原則: 融合 kernel の grid は「吸収した op のうち最大並列度のもの」に合わせる(reduction 側に
合わせると widest op が直列化する)。冗長計算は並列度確保より遥かに安い。**

### ★finding 2: 小 grid・少数層の融合は逓減(bisect 実測)
A1 +57µs(理論 44 ✓)/ A2+A3 ≈ +10µs(理論 88 — 依存チェーンでも tiny grid(10層×少 head)では
kernel 効率損と相殺)/ S1 +40µs(理論 176 — gather 流用 kernel の register 圧が食う)。
**大勝ちは「多層×多 op の長い依存チェーン」(GDN 30層×8op = +1.1-1.4ms)に集中。** 原子別
kill-switch(QWISP_FUSE_S1/A1)は bisect 計測法として温存。

**プロセス**: GLM ハーネス障害(OpenCode done検知ループ→Pi 移行→長考 idle 誤爆)で実装は Sonnet 完遂。
S2 の perf バグは Fable の G5 独立実測が検出し(lane 別 bisect → kernel 構造読み)、Fable が外科修正。
locked テスト(bit-exact)が修正の安全網として機能。

**I期(dispatch 削減)完了宣言**: 1354 → ~690 本/forward(M=1)。paired A/B 累計 ≈ +1.5ms/step。
e2e code 全 flag +9.4%(clean interleaved 実測、GDN 支配)。**次 = II期 megakernel(I/J: 全40層 1 kernel
+ h 常駐、帯域床 ~270 tok/s への本命)+ default 昇格判断(owner)**。
