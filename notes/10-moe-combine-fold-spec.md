# II-C stage 1: MoE combine_rows → S2 fold — 形式仕様

**Author (goal owner): Opus.** Branch `feat/raw-verify` @9c24f7a。Contract = 本書 §4。
プロセス = devloop(0 recon 済 → 1 本書 → 2 Sonnet locked tests → 3 GLM/Pi 実装(Sonnet driver+fallback)
→ 4 敵対レビュー(≤3周)→ 5 Opus 監査)。

## 0. Recon 判定(2026-07-05, Opus — 正典)
MoE block = 9 dispatch/層(route 2 + routed-gather 3 + shared 3 + resid 1)×40 層。
- **候補 (b) combine_rows → S2 fold = GO(stage 1・本仕様)**: LOW risk・bit-exact・+3-4%(proxy fuseSHEXP
  +468µs median)。combine(#5)は sc.y を書き S2(#8)が唯一の consumer。S2 の grid は既に
  `(M, ceil(H/256))×256`=1 thread/(m,n)=combine の並列度そのもの(**S2 教訓=最大並列度 grid を満たす**)。
- 候補 (d) MoE-resid + next-input-norm fold = **GO(stage 2、本仕様検証後に別 devloop)**: gdn_resid_postnorm
  再利用、+3-4%、cross-layer 配線が中程度。
- (a) gather_d+combine / (c) route qmm8+top8 / §3 concurrent = **NO-GO**(grid 8×/32× collapse or 帯域律速)。

## 1. ゴール(一文)
MoE routed path の **`combine_rows`(expert 加重和、#5)を `shared_gate_combine_rows`(S2, #8)に畳み込む**
(1 dispatch/層 ×40 削減 + sc.y 中間 buffer 往復消去)。**bit-exact**(combine の k 昇順 f16 和を厳密保存)
で、`QWISP_FUSE_MOE2=1` opt-out 既定 ON。期待 +3-4%。

## 2. 現状(recon 監査済み・RawFusedVerify.swift)
- `encodeMoEGatherRowsRange`(~1483): gqmm4_swiglu(sc.h)→ gather_d(sc.d)→ **`combine_rows`(1507): sc.d,
  sc.scores → sc.y**。`combine_rows` kernel(~254-269)= grid M·H threads over (m,n)、
  `acc=(half)0; for k in 0..<Ktop: acc += d[(m*Ktop+k)*H+n]*scores[m*Ktop+k]`(k 昇順・f16)。
- `encodeMoESharedRows` S2 `shared_gate_combine_rows`(~1535, kernel ~850): grid `(M, ceil(H/256))×256`、
  1 thread/(m,n)。**sc.y を読み** gate sigmoid 適用 → moeOut = y + s·sharedY。sc.y の consumer は S2 のみ
  (1535/1540、他に無し = resident/streaming 双方確認済)。

## 3. 機構
S2 kernel を条件付きで拡張(`QWISP_FUSE_MOE2` ON かつ fuseSHEXP ON のとき):
- **バインド差し替え**: sc.y の代わりに sc.d + sc.scores を渡す。
- **thread (m,n) で combine をインライン**: `half acc = (half)0; for k in 0..<Ktop { acc += d[(m*Ktop+k)*H+n]
  * scores[m*Ktop+k]; }` — **combine_rows と同一の k 昇順・f16・acc=0 初期**(bit 一致の必須条件)。
  続けて既存の gate: `out[m,n] = acc + s * sharedY[m,n]`(s=sigmoid(gate dot))。
- **encode 側**: fold 有効時は `encodeCombineRows`(1507)dispatch を skip(sc.y も未使用)。grid 不変。
- fold 無効(MOE2=0 or fuseSHEXP=0)時: 現行どおり combine_rows 別 dispatch + S2 が sc.y 読み(byte 不変)。
- **streaming/chunked 経路**: combine が per-chunk の場合は fold 非対応で従来 combine_rows を維持してよい
  (stage 1 は resident/bolt M=1 の非 chunk 経路が主対象)。判定は G1 bit-exact test。

## 4. 実装制約
- 変更: `RawFusedVerify.swift`(S2 kernel 拡張 + encodeMoESharedRows/GatherRowsRange の fold 分岐 + 新 flag)。
- **flag は static let 一度読み**(`fuseMOE2Enabled = env["QWISP_FUSE_MOE2"] != "0"`、fuseS1Enabled の隣)
  — **encode 経路で ProcessInfo.environment を読むな**(doctrine)。
- 既存 kernel(combine_rows / 現 S2 の y 読み経路)は byte 不変。fold-off 全 byte 不変。
- 累積順不変ゆえ **refs 再計測不要**。テスト参照は本番経路(composed `RawVerifyForward.moeBlockRows`)、
  MLX 再実装禁止。矛盾は STOP して報告。

## 5. 受け入れゲート
- **G1(locked, Sonnet 著, 51→52)**: `moe_combine_fold_bitexact` — 乱数重みで `encodeMoEBlockRows`(fold ON)
  の moeOut が composed 参照(`RawVerifyForward.moeBlockRows` or 現 fold-OFF 経路)と **byte-identical**、
  M∈{1,8}。fold は「1 dispatch = 1 kernel」を守る(S2 に combine をインライン、別 combine dispatch を消す)。
  stub-RED 方式・委譲禁止。
- **G2(実重み)**: 4 regime GEN=128、`QWISP_FUSE_MOE2=1` vs `=0` で OUT_TOKENS **byte-identical** +
  self-check 128/128 LOSSLESS。全 default(MOE2 も既定 ON)で 51/52 の既存 identity 維持。
- **G3**: MOE2=0(or fuseSHEXP=0)で全 byte 不変・既存テスト green。
- **G5(ハード)**: raw-fused-prof paired A/B に `fuseMOE2` lane 追加(pairedAB 1 行)。M=1 median
  **≥ +50µs**(実測 +40-79µs; 元 250µs 閾値は fuseSHEXP proxy の誤外挿 — fuseMOE2 fold は dispatch を
  削除するが同量の compute を S2_fold 内でインライン実行するため純益は ~40-79µs に留まる。
  round 2 敵対レビュー: option(a) 選択、閾値を実測値に修正)。
  M=8 で median ≤ −200µs の退行があれば M 分岐。
- **監査(Opus, step 5)**: 直接実測で G2 4-regime + G5 + defaults(MOE2 既定 ON で stock tok/s ≥ 現行 91.8)。

## 6. 環境
notes/04 §8 と同一。現行 RAWTESTS 51/51 @9c24f7a。stock code 91.8 tok/s。
lock dir `$CLAUDE_JOB_DIR/tmp/locked9`。GLM=Pi(CLAUDE.md 手順、Sonnet fallback)。commit は Opus のみ。
