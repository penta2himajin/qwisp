# notes/12 — 案B Stage 1: gate-score residency bias(route_top8 +ε)miniloop 契約

**目的**: bolt の router 選択を常駐 expert 側に ε だけ nudge し、buddy 置換(cold の score で buddy 重み)を
「resident 自身の score+重み」に置き換えて TF fidelity を上げる(Cache-Prior 式 routing-steer)。
Stage 0(commit 77e5994)実測で headroom 確定済み: coldRate 28-42%・margin p50 0.2-0.5・flip@ε=0.5 で 51-72%。
**ε=0(既定)は既存 bolt と byte-identical が絶対**。branch feat/raw-verify。背景 [[expert-reuse-drafting-devloop]]。

## ground truth(driver 照合済み、実装はこれに従う)
- `route_top8_rows` kernel(`RawFusedVerify.swift:115-149`): **選択は raw logit**(`work[tid] = lg`)の
  決定的 K 回 argmax reduction。**score は softmax gate**(`gates[tid] = (float)(half)(e/Z)`)で、選択後に
  top-8 内 renorm(:148)。encode 版=`encodeRouteTop8Rows`(:1305)、呼出=`encodeMoERouteRows`(:1526)。
- **bias は選択のみに掛ける**: `work[tid] = lg + (resident[tid] != 0 ? eps : 0.0f)`。gates/softmax/renorm は
  無改変=選ばれた expert は自分の真の gate score で combine される(これが fidelity 改善の機序)。
- **residency 判定に slotTable は使えない**(cold も buddy の valid slot を持つ)。resident ⟺
  `provider.cache.buddyExpertCPU[e] == e`。per-layer の int32 mask(resident=1/cold=0)を CPU で作る。
- ε の単位 = raw gate logit(BoltDiag の margin と同一単位)。

## 実装(additive)
1. **新 kernel `route_top8_rows_bias`**(既存 kernel 無改変=非bolt 経路は構造的に byte-identical):
   route_top8_rows のコピー + buffer 2本追加(`device const int* resident`, `constant float& eps`)+
   `work[tid] = lg + ((tid < N && resident[tid] != 0) ? eps : 0.0f)` の1点差分。別 pipeline に compile。
   encode 版 `encodeRouteTop8RowsBias(...)` を encodeRouteTop8Rows と同 idiom で。
2. **RawFusedForward に bias 状態**: `routeBiasMasks: [MTLBuffer]?` + `routeBiasEps: Float = 0` +
   `public func setRouteBias(masks: [[Int32]], eps: Float)`(per-layer mask を MTLBuffer 化して保持)。
3. **encodeLayerBolt → encodeMoEBlockRows → encodeMoERouteRows に bias を thread**:
   `bias: (mask: MTLBuffer, eps: Float)? = nil` optional 引数(diag と同 pattern)。
   encodeMoERouteRows は bias 非 nil なら bias kernel、nil なら既存 kernel。
   **★bias は全 M に適用**(greedy M=1 も verify M>1 も)— bolt の spec≡greedy 自己整合には
   routing 規則が M に依らず同一であることが必須(diag の M==1 限定とは違う。混同するな)。
4. **runBoltMode 配線**: env `QWISP_ROUTE_BIAS_EPS`(Float, 既定 0)。>0 のとき phase 5 後に
   providers から mask(bexp[e]==e)を構築し `fwd2.setRouteBias(masks, eps)`。
   **★QWISP_RAW_TF の teacher-forced backend(fwdTF, `RawSpecRunner` ~841-846 の
   fwdTF.setBoltTables 直後)にも同じ setRouteBias を掛ける**(TF が biased engine を測るように。
   これを忘れると TF が unbiased routing を測り G-D が無意味になる)。
5. self-test hook `routeTop8RowsBias(logits: MLXArray, residentMask: [Int32], eps: Float, M: Int,
   N: Int = 256, K: Int = 8) -> (MLXArray, MLXArray)?`(routeTop8Rows :109 と同 idiom の
   単発 dispatch wrapper、G-A テストが呼ぶ)。

## GATE
- **G-A unit(model-free, RawVerifyTests に追加 62→64, locked)**:
  1. `route_bias_eps0_identity`: 合成 logits(M∈{1,2}, N=256, K=8)で bias kernel(eps=0, mask 任意)と
     既存 routeTop8Rows の inds+scores が **bit-exact 一致**。さらに all-resident mask + eps>0
     (全 logit 等シフト=選択不変)でも inds 一致・scores bit-exact を確認(adversarial)。
  2. `route_bias_neartie_flip`: 手作り near-tie(cold expert の lg が resident を margin だけ上回る合成、
     他は十分低い): margin < eps → 選択順が flip し resident が上位、scores は「選ばれた expert 自身の
     gate 値」(bias が score に漏れていない事の検証)。margin > eps → 選択不変。手計算値で照合。
- **G-B(audit, model 要)**: ①env 未設定 baseline vs `QWISP_ROUTE_BIAS_EPS=0` で OUT_TOKENS
  byte-identical ②`QWISP_ROUTE_BIAS_EPS=0.5` を2回走らせ OUT_TOKENS 同一(決定性)
  ③eps=0.5 + `QWISP_RAWSPEC_CHECK=1` で bolt self-check(spec vs greedy, 同 eps)が self-consistent。
- **G-C(audit がコード読解)**: 既存 route_top8_rows kernel/encode が無改変。bias は bolt 経路のみ
  (strict/resident 無改変)。fwdTF への setRouteBias 配線があるか(★G-D の妥当性の要)。
  全 M 適用になっているか(M==1 gate で verify から漏れていないか)。
- **G-D(数値報告のみ、headroom/効果判定は driver)**: C=64, GEN=128, `QWISP_RAW_TF=1` で
  **ε ∈ {0, 0.1, 0.25, 0.5, 1.0} を小さい側から** shortnl と code の 2 regime を sweep し、
  各 cell の「bolt TF fidelity vs strict-canonical」%(と tok/s)を表で報告。
  実行例: `QWISP_RUN=raw-spec QWISP_RAW_C=64 QWISP_RAW_BOLT=1 QWISP_RAW_TF=1
  QWISP_ROUTE_BIAS_EPS=<ε> QWISP_MTP_REF=refs/shortnl.safetensors QWISP_GEN=128 <BIN> stream`。
  baseline(ε=0)の TF も必ず同一プロセスモードで取り直す(過去値と比べない)。

## Doctrine
- ε=0/env 未設定 = byte-identical が絶対。既存 kernel は触らない(別 kernel を足す)。
- unit green≠配線: audit は encodeMoERouteRows/encodeLayerBolt/fwdTF を読んで配線を確認。
- テスト参照は本番 kernel(routeTop8Rows)基準。CPU 再実装 oracle 禁止。矛盾は STOP して報告。
- GPU/build 排他: 着手前に `pgrep -f qwisp-poc`・xcodebuild 不在確認。
- commit 禁止(driver が gate)。locked tests 不可侵(lockDir=.locks/route-bias-stage1)。
- G-D は報告のみ。ε の採否・default 化の判断は driver/owner。
