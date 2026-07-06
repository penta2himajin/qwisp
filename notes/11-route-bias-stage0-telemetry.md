# notes/11 — 案B Stage 0: bolt routing telemetry + headroom 実測(miniloop 契約)

**目的**: gate-score residency bias(Cache-Prior 式)の着手判断材料。bolt decode の per-layer 実 routing を
CPU から可視化し、「cold-selection 率 × near-tie margin 分布」= ε で動かせる headroom を実測する。
**測定のみ・挙動変更ゼロ**(diag off = 現行 bolt と byte-identical)。headroom が小さければ Stage 1 は no-go、
telemetry は資産として残る。branch feat/raw-verify。背景 memory: [[expert-reuse-drafting-devloop]] 案B 節。

## 背景(確認済み事実)
- bolt(`RawSpecRunner.swift runBoltMode`)は per-layer top-C を freeze し、cold expert は gather 時に
  buddy slot へ remap(`encodeMoEGatherRowsRange`, slotTable)。**選択は gate 純 argmax**(`route_top8_rows`)
  で residency 非考慮。combine の gate score は cold のまま=fidelity 低下(shortnl 81-84%)の源泉。
- bolt 1-CB 中、per-layer routing inds(`MoEScratch.inds`)と gate logits(`MoEScratch.gl`)は**単一 scratch を
  40 層で使い回す**ため CB 完了後は最終層しか残らない。→ 層別 side-buffer への copy が必要。
- 常駐判定: `provider.cache.slotOf.keys`(pinned top-C)。cold 判定: `buddyExpertCPU[e] != e`。

## 実装(3点、全て additive・diag-off で無効)
1. **層別 side-buffer copy(GPU)**: `RawFusedForward` に
   `public var diagRouteBufs: (inds: MTLBuffer, gl: MTLBuffer)? = nil` を追加。
   非 nil かつ `streamMode == .bolt` かつ **M == 1** のとき、`encodeLayerBolt` の route の後に
   小 kernel `diag_copy_route` を 1 dispatch 追加: inds(Ktop 個 int32)と gl(E 個 half)を
   層 offset(inds: li*Ktop, gl: li*E)へコピー。route/gather/combine 本体は**無改変**。
   nil のとき dispatch 追加ゼロ=既存経路 byte-identical。
2. **CPU 集計(純関数)**:
   ```swift
   static func computeRouteDiag(inds: [Int32], gl: [Float16], resident: Set<Int>,
                                buddyExpert: [Int32], Ktop: Int)
       -> (coldSelected: [Int], margins: [Float])
   ```
   routed の各 expert e について cold(buddyExpert[e] != e)を判定。cold e の margin =
   `gl[e] − max{gl[r] : r ∈ resident, r ∉ routed}`(=常駐 expert が +ε で e を置換するのに必要な ε)。
   resident 未選択が空なら margin = +inf(除外)。
3. **diag runner 配線**: `runBoltMode` に env `QWISP_BOLT_DIAG=1` 分岐: side-buffer 確保 →
   `fwd2.diagRouteBufs` 設定 → **chain 無効化(diag 中 chainK=0、1 CB=1 step にして side-buffer が
   step 毎に読めるように)** → 各 stepArgmax 後(D==0 greedy 経路のみで可)に全40層分を読み
   `computeRouteDiag` で累積 → decode 終了後に集計出力:
   ```
   [BoltDiag] group=early(0-12)|mid(13-26)|late(27-39)
     coldRate=<cold-selected 数/routed 数> tokensWithCold=<%>
     margin p10/p50/p90=<logit 単位>  flip@eps={0.5:<%>,1:<%>,2:<%>,4:<%>}
   ```
   flip@eps = cold selection のうち margin < ε の割合(=ε でその選択が常駐に flip する)。

## GATE(miniloop pass/fail)
- **G-A unit(model-free, RawVerifyTests に追加, locked)**:
  1. `diag_copy_route` copy 正しさ: 合成 inds/gl → side-buffer の層 offset に bit-exact コピー。
  2. `computeRouteDiag` 正しさ: 合成ケース(cold/resident/margin 手計算値)と一致、resident 未選択空で +inf 除外。
- **G-B identity(model 要, audit)**: `QWISP_BOLT_DIAG=1` と未設定で bolt の **OUT_TOKENS byte-identical**
  (測定が出力を変えない)。既存 RAWTESTS 全 PASS。
- **G-C 構造(audit が encode 読解)**: diag-off で dispatch 追加ゼロ。route/gather kernel 無改変。
  strict/resident 経路無改変。
- **G-D 測定報告(pass/fail にしない、driver が headroom 判定)**: shortnl + code、C=64、
  `QWISP_RUN=bolt`(または raw-spec bolt)QWISP_BOLT_DIAG=1 で [BoltDiag] 表を報告。

## Doctrine
- テスト不可侵(lock)・commit 禁止(driver が gate)・GPU/build は排他1ジョブ
  (**着手前に `pgrep -f qwisp-poc` と xcodebuild の不在を確認**)。
- flag-off byte-identical が絶対。diag は測定専用で数値に触れない。
- Stage 1(ε bias kernel)は本 miniloop の範囲外(headroom 判定後に別途)。
