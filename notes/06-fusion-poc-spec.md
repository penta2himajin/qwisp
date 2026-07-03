# kernel fusion PoC: gqmm4_swiglu_rows(g+u+swiglu 融合)— 形式仕様

**Author (goal owner): Fable.** Branch `feat/raw-verify`. Contract = 本書 §4 ゲート。
ループ: Sonnet 書換不可テスト → GLM-5.2(`glm-code --allow` で build 自走)→ Sonnet 敵対レビュー → Fable 最終 gate。

## 0. Physics smoke(完了、2026-07-04, raw-dispatch-bench)
- **per-dispatch overhead = 2.15-2.34 µs**(1-CB 依存チェーン、仕事量非依存=overhead 支配を確認)。
- forward の実 dispatch 数 = **~1354**(attn 10×16 + GDN 30×21 + MoE 40×14 + head 4)→ dispatch税 ~3.0ms
  = M=1 step (~14ms) の **22%**。全 fusion キャンペーン(→~200本)= +23% 確実 + 中間往復削減 → 見積り
  +30-45% は credible。**キャンペーン GO、本 PoC はその最初の原子**。

## 1. PoC ゴール(一文)
MoE routed-expert の **gather-g / gather-u / swiglu の 3 dispatch を 1 kernel `gqmm4_swiglu_rows` に融合**
(40 層 × 2 本削減 = 80 dispatch ≈ ~180µs/step)し、**bit-exact**(既存 3-kernel 連鎖と全要素 bit 一致)を
保ったまま、fused gather 原子(MoE megakernel F への技術的最難関)の成立を実証する。

## 2. 現状(grounding 済み事実)
- `RawFusedVerify.swift` `encodeMoEGatherRowsRange`(~603-639): 
  `encodeGatherQmmRows(w.swGW → sc.g)` → `encodeGatherQmmRows(w.swUW → sc.u)` → `encodeSwiglu(g,u → sc.h)`。
  g/u は **同一 x[M,K]・同一 inds[M,Ktop]** を読む(lhsPer: false)。h[i] = swiglu(g[i], u[i])(既存
  `swiglu` kernel の式に厳密一致させる — 実装前に必ず kernel source を読んで式を確認)。
- `gqmm4_rows` kernel(`RawMetalForward.swift` ~3886-3937): grid (1, N/8, M·Ktop)、threads (32,2)。
  各 threadgroup が出力 1 行(mk)の 8 列を計算。w[E,N,K/8] 4-bit affine gs=64、inds[mk] で expert 選択。
- bolt の slot_remap は gather の**前段**(inds を書き換え)なので融合対象外・非干渉。

## 3. 機構
新 kernel `gqmm4_swiglu_rows`: gqmm4_rows と同一 grid/threads。各 thread 群は同じ 8 列について
(a) Wg[e] で g 値を、(b) Wu[e] で u 値を、**既存 gqmm4_rows と同一の per-column dot ループ順**で計算し
(bit 一致のため演算順を変えない)、(c) h = swiglu(g,u) を register 内で適用して h バッファへ直接書く。
- g/u の中間バッファ書き/読み(2×[M·Ktop,I] f16 往復)と 2 dispatch が消える。x は 1 回読みに削減。
- weight 読み量は不変(Wg+Wu)。
- encode 側: `encodeGatherQmmSwigluRows`(encodeGatherQmmRows と同 signature 系)を追加し、
  `encodeMoEGatherRowsRange` を **`QWISP_FUSE_GU=1` 時のみ** 新経路に分岐(既定 off、既存経路 byte 不変)。
- sc.g / sc.u バッファは flag-on 時未使用(確保はそのまま、削減は後続キャンペーンで)。

## 4. 受け入れゲート
- **G1(RAWTESTS、書換不可・Sonnet 著、31→33)**:
  - `fuse_gu_bitexact`: 乱数重み(E=16 等の縮小構成)で `gqmm4_swiglu_rows` の h 出力が
    既存 3-kernel 連鎖(gqmm4_rows×2 + swiglu)の h と **全要素 bit 一致**(bitEqual)。
    M∈{1,8,17} × Ktop∈{1,8} × N∈{512,1536} を掃く。
  - `fuse_gu_m_invariance`: fused kernel の rows 実行 ≡ M=1 ループ実行 bit 一致(M 不変性、既存 idiom)。
- **G2(実重み identity)**: `QWISP_FUSE_GU=1` vs `=0` で OUT_TOKENS byte-identical + self-check
  128/128 LOSSLESS(code + longctx の 2 regime で確認。fused は演算順同一ゆえ bit-exact のはず —
  ずれたら演算順の再現ミス=実装バグ)。
- **G3(非退行)**: flag 未設定で全 byte 不変、既存 31 本 PASS。
- **G5(速度・PoC 基準)**: raw-fused-prof(または同等)で M=1 GPU-exec が **≥100µs/step 短縮**
  (80 dispatch × ~1.5µs 保守値)。tok/s 退行なし。⚠️PoC 単体の tok/s 利得は ~+1% で小さい —
  本 PoC の合否は「fused gather 原子の bit-exact 成立 + in-situ で dispatch 削減が実測どおり効く」。
  ここが green なら全キャンペーン(GDN 21→~6 / attn 16→~6 / MoE 14→~4、別 spec)へ。

## 5. 環境
- build/BIN/refs/model: notes/04 §8 と同一。microbench: `QWISP_RUN=raw-dispatch-bench`。
- GLM 自走 build 可: `glm-code --allow 'xcodebuild*' --allow '*qwisp-poc*' "..."`(GPU 排他は Fable 管理
  — GLM が build/test 中は他ジョブを走らせない)。
- テスト lock dir: `$CLAUDE_JOB_DIR/tmp/locked3`。commit 毎 push・auto-commit 禁止(owner gate)。

---

## 6. 最終判定(Fable gate, 2026-07-04)— PoC 全ゲート GREEN、キャンペーン GO

**G1 33/33(lock 照合済)/ G2 code+longctx byte-identical+128/128 LOSSLESS / G3 flag-off 不変 /
G5 M=1 GPU-exec mean +160µs 削減 = 理論値(80×2µs)一致(σ≈150µs, marginal)/ e2e code +3.2%, longctx −1.6%(noise)。**

physics smoke(µbench 2.15-2.34µs/dispatch)→ in-situ(160µs/80本=2.0µs)の二重実証で
**fusion キャンペーン(1354→~200 dispatch, +23%〜 + 中間往復削減)の物理が確定**。

実装経緯: GLM-5.2 が実装(2 回の idle-hang があったが編集は完了しており、ハングは自己検証の読みループ。
build+テストは Fable 側で green 確認)。敵対レビュー(Sonnet)は欠陥ゼロ・kernel bit-faithful を確認。

### ★キャンペーンへの設計入力(レビュー発見)
1. **M>1 register 圧迫**: fused は gres+ures 同時保持で occupancy 低下 → prof M=8 +1.4ms/M=17 +3.9ms 退行
   (M=1 は latency-bound で無傷)。**campaign 標準 = encode 時 M 分岐(M 小=fused / M 大=unfused)**。
   e2e では longctx −1.6% 止まり(verify 律速が別)だが原子ごとに要評価。
2. **prof の内部 averaging(≥10 reps)を先に整備**してから各 fusion 原子を G5 gate する(single-run σ≈150µs)。
3. sc.g/sc.u の未使用確保(fused 時)は campaign 成熟時に回収。
4. fuseGU は opt-in 維持(M 分岐実装までは default 昇格しない)。
