# notes/13 — bolt recalib+bias 採用・default 昇格(miniloop 契約)

**owner 判定(2026-07-06)**: rolling re-calib + residency bias を bolt の default に昇格。
実験実証済み(commit 0b2924f, @512tok): R=128+ε0.25 で longctx +2.7 / shortnl +2.5 / code +2.6 /
agentic −0.4(R=128 の窓 thrash、owner 承知で採用)。bias が refresh pread を 25-35% 削減。
背景 [[expert-reuse-drafting-devloop]] / notes/11-12。branch feat/raw-verify。

## 何を作るか
現状 recalib は **TF loop 限定の実験配線**。採用 = ①free-run(bolt decode loop)への本配線
②default flip(`QWISP_BOLT_RECALIB_R` 既定 128 / `QWISP_ROUTE_BIAS_EPS` 既定 0.25、bolt のみ、
0 指定で opt-out=旧 bolt と byte-identical)③観測 coverage の完全化(下記)④速度実測。

## ground truth(driver 設計済み — この通りに作る)
### 観測 coverage 問題と解(kernel 無改変)
- free-run bolt は chain(K=8 を 1 CB)+ verify(M>1)を使う。現 diag copy は「M==1 のみ・slot 0 のみ」
  → chain span は最終 step しか残らず 1/8 サンプリング、verify は観測ゼロ。採用には全 step 観測が要る。
- **解**: `diag_copy_route` kernel は flat copy ゆえ**パラメータだけで一般化**できる:
  - M>1: `kE.x = M*Ktop` を渡すだけ(kernel 無改変で M 行 copy)。
  - chain: encode は CPU 側で per-step に行われる(chainedStepArgmax が K step をループ encode)ので、
    **encode 時に chain 位置 k を instance var に置き、bind offset を slot 次元でずらす**。
- **RawFusedForward に追加**: `diagObsMaxM: Int = 1` / `diagChainSlot: Int = 0`(instance var)。
  encodeLayerBolt の inds copy 条件を `diagRouteBufs != nil && M <= diagObsMaxM` に、offset を
  `((diagChainSlot * nLayers + li) * diagObsMaxM) * Ktop * 4`、copy 長 `kE.x = M*Ktop` に。
  gl copy は従来どおり(slot 0 相当・M==1 時のみ・offset li*E*2 — Stage-0 diag 用で recalib は使わない)。
  **defaults(1, 0)で旧 layout `li*Ktop*4` に退化 = Stage-0 diag / locked test 61 は無傷**。
  `chainedStepArgmax` は bolt+diagRouteBufs 時に per-step で `diagChainSlot = k` を set(encode 後 0 に戻す)。
- side buffer 確保(free-run recalib 時): `chainKMax * nLayers * diagObsMaxM * Ktop * 4`
  (chainKMax = 使用 chainK、非 chain step は slot 0)。~数百 KB。

### free-run recalib 配線(runBoltMode phase 6)
- env: `QWISP_BOLT_RECALIB_R` **既定 128**(`Tell.envInt`)。0 で off=旧挙動。
  `QWISP_ROUTE_BIAS_EPS` **既定 0.25**。0 で off。**bolt(runBoltMode)のみ**。他 tier 不変。
- 観測: 各 loop iteration の後、今回 dispatch した形に応じて side buffer を読む —
  ①D==0 per-step: slot 0, M=1 ②D==0 chain span(emitted 数 E_c): slot 0..E_c-1 を各 M=1
  ③verify: slot 0, M=D+1(+pending, A3 無し bolt は D+1) ④rebuild forward: 観測せず(rollback 後の再構築)。
  読んだ inds から distinct→winCounts/winCoact 累積(TF 実験配線と同じ数式)。
- refresh: `out.count` が R 境界を跨いだ iteration 末尾で: per-layer 窓 top-C ensure(pread)→
  buildBuddyTable(winCoact)→fwd2.setBoltTables→(eps>0 なら)mask 再構築+setRouteBias→窓リセット。
  refresh 回数と pread misses を [BoltRecalib] 行で出力。
- **TF loop の既存 recalib 配線は温存**(default が変わるだけで両 loop 一貫)。
- decay(`QWISP_ROUTE_BIAS_DECAY_H`)は既定 0 のまま(recalib が根本治療、decay は knob 温存)。

## GATE
- **G-A unit(locked, 64→66)**:
  1. `diag_copy_slot_m_layout`: 新 offset 式の検証 — 合成 inds を (slot, li, M) の組合せ
     (slot∈{0,3}, li∈{0,2}, M∈{1,4})で copy し、side buffer の期待 offset に bit-exact、
     かつ **defaults(slot=0, maxMobs=1, M=1) が旧 layout li*Ktop*4 と一致**(test 61 互換の証明)。
     新 self-test hook(既存 diagCopyRouteSelfTest は不変のまま別 hook を追加)。
  2. `recalib_obs_accumulate`: 観測累積の純関数(inds[M*Ktop]→distinct→counts/coact pairwise)を
     合成ケース手計算で照合(重複 expert 行、M>1 の行独立性)。実装が inline なら純関数に抽出して test。
- **G-B(audit, model 要)**:
  1. **opt-out 恒等**: `QWISP_BOLT_RECALIB_R=0 QWISP_ROUTE_BIAS_EPS=0` の OUT_TOKENS が
     旧 bolt(直前 commit の挙動)と byte-identical。
  2. **default 決定性**: env 無し(新 default)で 2 回実行し OUT_TOKENS 同一。
  3. 既存 RAWTESTS 全 PASS(66/66)。
- **G-C(audit コード読解)**: chain 全 step が slot 別に観測されるか(chainedStepArgmax の
  diagChainSlot set/reset)/ verify M>1 観測 / 非 bolt 経路(strict/resident)無改変 /
  Stage-0 diag(QWISP_BOLT_DIAG)が旧 layout のまま動くか。
- **G-D(数値報告のみ)**:
  1. TF@512(scratchpad refs512, `QWISP_RAW_TF=1`, default env): 4 regime の TF% が実験値
     (87.5/75.0/93.4/92.4 近傍)を再現するか。
  2. **free-run 速度**: default vs opt-out の bolt tok/s(C=64, GEN=512, fast-SSD と
     `QWISP_SSD_THROTTLE_GBS=1.5`)。refresh burst の実測コスト。表で報告(採否済みだが
     async 化の要否判断材料 = driver/owner)。

## Doctrine
- opt-out(R=0 かつ eps=0)= 旧 bolt と byte-identical が絶対。非 bolt tier は 1 バイトも挙動を変えない。
- kernel(diag_copy_route / route_top8_rows / route_top8_rows_bias)は無改変。パラメータ/offset のみ。
- unit green≠配線: audit は chainedStepArgmax / encodeLayerBolt / runBoltMode loop を読む。
- 既存 locked tests(61-64)不可侵。新 lock dir = `.locks/recalib-adoption`。
- GPU/build 排他(pgrep 確認)。commit 禁止(driver が gate)。G-D は報告のみ。
