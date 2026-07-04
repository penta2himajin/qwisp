# Phase II-a: GPU token feedback + double-buffered pipelining — 形式仕様

**Author (goal owner): Fable.** Branch `feat/raw-verify` @12842a9。Contract = 本書 §4。
プロセス = qwisp-fusion-loop workflow(0 recon 済 → 1 本書 → 2 Sonnet locked tests → 3 GLM/Pi 実装
(Sonnet driver+fallback)→ 4 敵対レビュー(≤3周)→ 5 Fable 監査)。

## 0. Recon 判定(2026-07-04, Opus — 正典はこの節)
- **megakernel(d1-handoff vision I/J)= NO-GO(物理)**: Metal に cross-TG barrier 無し・forward-progress
  非保証(spin 同期は deadlock 級)・単一 TG は ~12.5GB/s で 7× 退行。**vision J は誤診**: h[1,2048]=4KB の
  常駐化は ~0.007ms にしかならない。床までの ~8ms の正体 = **M=1 GEMV の帯域利用率 ~26%(occupancy)**。
  ICB は repo 実測(icb-bench)で CPU-encode −11% のみ = null(GPU-bound ゆえ)。Apple が cross-TG barrier を
  出荷するまで再挑戦禁止。
- **M=1 step 会計**: wall 14.4ms = GPU 13.2(≈ dispatch 間隙 1.5 + weight床 3.6 + **GEMV 低稼働 ~8**)+ CPU 1.2。
- 生き残り: **II-a 本仕様(~+1.2ms)** / II-b concurrent routed∥shared(投機的, +0.5-1ms, 別判断)/
  II-c MoE 連鎖融合(+0.4ms, 逓減, 保留)/ **e′ M=1 GEMV split-K = 最大の獲物(~8ms 級, 別キャンペーン要 recon)**。

## 1. ゴール
decode の **K step を 1 command buffer に連結**(GPU 側 token feedback、CPU 非復帰)+ **double-buffered
CB 投入**(CPU が step k+1 群を encode する間に GPU が step k 群を実行)で、CPU 露出 1.2ms/step を隠蔽。
**期待 +~9%(bolt/greedy・D==0 span 限定)**。opt-in `QWISP_CHAIN_K=<k>`(既定 off=現行 per-step)。

## 2. 機構(recon 監査済みの事実に基づく)
- `hd.tokensIn` は device buffer(現状 CPU が `.update(from:)` で充填、`encodeEmbed` index 3)。
  `argmax_rows` は `hd.tokensOut`(device)へ書く。**CPU 往復が唯一の切れ目**。
- **indirect-embed**: embed kernel の変種(または binding 差し替え)で step k+1 の embed が
  `hd.tokensOut`(step k の結果)を直接読む。offset/ring で K step 分の token 列を GPU 上に残す
  (最終 readback で K token 一括回収 → OUT_TOKENS 連続性維持)。
- **per-step 既知定数**: kv len は step 毎に +1 されるだけ → sdpa の baseLenPlus1 / write_kv pos を
  step 毎の定数として encode(K 本分)。GDN ping-pong / convHist は A→B→A… の交互 binding を encode。
  cert/telemetry readback は chain 中は無効(chain は greedy 専用)。
- **double-buffer**: 2 本の CB を交互に(CB_i commit 後、wait せず CB_{i+1} を encode; 前々回のを wait)。
  encode(~0.83ms/K-step 群)と GPU 実行を重畳。
- **spec loop 統合**: suffix draft には CPU 上の token が要る → **chain は D==0 の greedy span のみ**。
  runSpecLoop で drafts.isEmpty が続く限り chain(K 上限)、draft が出たら per-step へ復帰。
  accept=0 の regime(code/shortnl)は全 step が対象。snapshot/rollback は chain 境界でのみ取得
  (chain 中に verify は無い=rollback 規約と非干渉)。

## 3. 実装制約
- 変更: `RawFusedVerify.swift`(chainedStepArgmax(K) 追加: K-step encode + indirect embed + 交互 binding
  + 一括 readback。既存 stepArgmax 不変)、`RawSpecRunner.swift`(greedy span で chain 使用、
  `QWISP_CHAIN_K` 配線)。double-buffer は第2段(まず 1-CB K-step で G1/G2 を green に、その後 pipeline 化)。
- 既存 kernel 不変(embed の indirect 変種は追加。既存 embed_rows_q4 は byte 不変)。flag-off 全 byte 不変。
- テスト改変禁止・commit 禁止(driver=Fable のみ)・1原子1dispatch 原則は本 wave では「1 step 群=既存
  dispatch 列の K 連結」(新たな融合はしない — dispatch 数は不変、CPU 境界だけを消す)。

## 4. 受け入れゲート
- **G1(locked、参照=本番経路)**: `chained_greedy_bitexact` — 乱数重み esModel で
  `chainedStepArgmax(K)` の K token 列+終了時 cache 状態(KV len/内容・GDN state)が
  `stepArgmax` K 回逐次と **bit 一致**(K∈{2,3,8})。`chained_boundary` — chain→per-step→chain の
  切替(spec 復帰想定)で同 bit 一致。stub-RED 方式。**stub は forwardRows/stepArgmax への委譲禁止**
  (GREEN-by-delegation は不合格 — 事故の前例あり。stub は nil を返して RED を証明すること)。
- **G2(実重み)**: code regime GEN=128、`QWISP_CHAIN_K=8` vs off で OUT_TOKENS **byte-identical** +
  self-check 128/128 LOSSLESS。全 fusion flag 併用でも identity。
- **G3**: flag off 全 byte 不変・既存 46 本 PASS 維持。
- **G5(ハード・wall 基準)**: 純 greedy 実測(code、accept=0)で **K=8 chain の wall/step が per-step 比
  ≥5% 短縮**(interleaved、複数回 median)。GPU-exec でなく **wall**(利得は CPU 側)を測ること。
  paired A/B prof は GPU-exec 基準ゆえ本 wave の主計器にしない(補助のみ)。
- 満たさない場合: double-buffer 化(第2段)まで実装して再測 → それでも <5% なら no-go 記録で撤収。

## 5. 環境
notes/04 §8 と同一。RAWTESTS 現在 46/46 @12842a9。lock dir `$CLAUDE_JOB_DIR/tmp/locked6`。
GLM は Pi ハーネス(CLAUDE.md 手順: タスクファイル化・短プロンプト・idle 1200s・-c 1回・Sonnet fallback)。

---

## 6. 最終監査(Fable, 2026-07-04)— II-a 採用(chain 単独 +13%)+ 合成不全を発見

- G1 48/48(lock 照合・RED 証明済 stub)/ G2 chain-off/on/全flag+chain 3構成 byte-identical + 128/128
  LOSSLESS / G5 **chain 単独 wall +13-14%(67.7-68.9 → 76.2-77.9 tok/s、Fable interleaved ×2 再現)** =
  仕様 gate(≥5%)大幅クリア。**code regime の wall 最高値を更新**。
- **★既知問題(次ラウンド)**: fusion flag との合成で chain 利得が消失(GDN/ATTN/SHEXP+chain=66-68)、
  **GU+chain は 2.3 tok/s の壊滅的退行**。correctness は全構成 green。示唆: fused encode 分岐に
  per-encode の隠れ CPU コスト(ProcessInfo env 読み/バッファ割当/ensure 再試行の疑い)。同根仮説:
  **fusion の GPU-exec 利得(GDN +1.4ms)が wall に現れない謎**も per-step encode の CPU 増で相殺されて
  いる可能性 — 診断すれば fusion I期の価値が wall でも回収できるかもしれない(次ラウンドの本丸)。
- 推奨構成(現時点): **QWISP_CHAIN_K=8 単独**(flags なし)= 76-78 tok/s。
