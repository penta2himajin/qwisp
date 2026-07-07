# notes/14 — bolt async refresh(recalib pread の decode 重畳)設計書

**問題**: notes/13 採用の rolling re-calib は frRefresh(`RawSpecRunner.swift#runBoltMode`)が
**同期 pread burst**: R=128 境界で全 40 層 × 窓 top-C ensure を decode 停止して実行。
slow-NAND(1.5GB/s)で **shortnl −51%(107.9→52.7)/ code −56%(141.6→62.3)** — bolt の
ターゲット HW で価値半減。fast-SSD は −24%…+8%。目標 = slow-NAND tok/s の回復。

## recon 確定事実(2026-07-06, @8237cf5 実読)

1. **bolt decode は io=0**: streamMode==.bolt は frozen slot table(MTLBuffer)で gather、
   decode 中に `ensure()` を呼ぶ経路は**皆無**(forwardRows/stepArgmax/chainedStepArgmax/
   rebuild forward 全て fused bolt 経路、strict の per-layer ensure は通らない)。
   → **decode 中 SSD は完全 idle = 重畳帯域はタダ**。
2. **全 bolt 呼出は blocking**: `cb.commit(); cb.waitUntilCompleted()`(RawFusedVerify.swift:3333、
   chain も 1 CB で同様)。→ **loop 先頭 = GPU idle = 安全な mutation point**(CPU turn)。
   現行 sync frRefresh が race-free なのはこのため。
3. **GPU race 分析(本 recon の核心)**:
   - 生きた arena slot への直接 pread は CB in-flight 中**無条件で unsafe**: buddy remap により
     cold expert は任意の hot slot へ写像される = **どの常駐 slot もどの step でも読まれ得る**。
     「読まれない slot だけ書く」安全部分集合は存在しない。
   - **staging buffer(GPU が一切読まない別領域)への background pread は常時安全**。
   - arena slot バイト・slotTables(MTLBuffer)・routeBiasMasks の書換えは **CPU turn 限定なら安全**
     (in-flight CB ゼロ、unified memory + .storageModeShared で追加 sync 不要)。
   - stale slot-table 罠: slot バイト書換えと setBoltTables/setRouteBias/buddy 表再構築を
     **同一 CPU turn で原子的に**行えば slot-consistent(7-13pt TF 崩壊の前科の回避条件)。
4. **決定性の罠(新規発見)**: 自由走行の「IO 完了したら swap」は swap 位置が IO タイミング依存
   → routing が変わり OUT_TOKENS が run 毎に変わる = **決定性 gate(self-check 代替)が死ぬ**。
   swap は **token-index 固定スケジュール**(未完了なら短時間 block して待つ)にする必要がある。
5. コスト整合: 実測 −51% は「refresh 毎 diff ~数百 expert × ~1.8MB を 1.5GB/s で読む
   = refresh 毎 ~0.5-1s × 4 回/512tok」と整合。IO 総量は不変なので**重畳のみが回復手段**。

## 設計判定: (a)+(b) hybrid = chunked staging + 固定 token スケジュール swap

- **(b) 単独(B3 式 per-step 同期 budget)は棄却**: CPU turn の同期 pread は GPU idle 中に走る
  = wall-clock に全額直課金。burst を均すだけで**総 IO は critical path から消えない**
  → −51% を回復できない(gate 未達が構造的に確定)。
- **(a) 素のフル staging は棄却**: worst diff C=64 × 40 層 ≈ 4.5GB staging は 8GB 機で不成立。
- **採用**: (a) の staging を (b) の budget で chunk 化。staging は B expert 分(~15-30MB)のみ。

### 機構(runBoltMode phase 6 の frRefresh 再構成)
1. **R 境界(loop 先頭)= plan 作成のみ**(mutation 無し):
   窓 counts/coact を snapshot → per-layer 窓 top-C → diff(newTop ∖ 常駐)→ victim slot 割当
   (LRU tick 古い順、newTop の slot は除外、決定的 tie-break=既存 sorted 流儀)→
   job 列 [(layer, expert, victimSlot)] を chunk(B 個)に分割。窓リセット。
2. **background pread**: 直列 background queue が chunk j を staging arena(ExpertArena N=B を
   1 本流用、9 tensor 並列 pread は loadMany 相当)へ読む。完了 flag。single staging・逐次
   (chunk j swap 時に j+1 の pread を kick)で足りる — chunk IO(B=8 で ~9ms)≪ S token decode。
3. **swap = 固定 token スケジュール**: chunk j は `out.count ≥ 境界 + (j+1)·S` の loop 先頭で
   swap。IO 未完なら **block して待つ**(決定性維持。通常 wait≈0)。swap 内容(原子的 1 CPU turn):
   staging→victim slot へ memcpy(sliceBytes、~0.2ms/expert)→ slotOf/expertAt/tick 更新
   (ensure() は通さない=same-call 不変量に触れない)→ buildBuddyTable(snapshot coact)→
   setBoltTables → (eps>0)mask 再構築+setRouteBias → 全て同 turn 完結。
4. **plan 未消化で次 R 境界**: 残 chunk を境界で block 消化(burst 尾)してから次 plan。
   sane B/S では非発生(G-D で確認)。
5. env: `QWISP_BOLT_REFRESH_ASYNC`(既定 1、**0 = 現行 sync frRefresh と byte-identical**)、
   `QWISP_BOLT_REFRESH_B`(既定 8)、`QWISP_BOLT_REFRESH_S`(既定 1 token)。
   B×⌊R/S⌋ ≥ 予想 diff を満たす既定(8×128=1024 ≥ 数百)。
6. **TF 側は schedule emulation**: async の fidelity 影響 = swap の token 位置分割だけ
   (IO threading は routing に無関係)。TF loop は同じ token-index chunk swap を**同期で**再現
   すれば async 機構なしで fidelity を測れる。TF 配線はこの emulation を実装。

## GATE(miniloop 契約の骨子)
- **G-A unit(locked, 66→68 目安)**:
  1. `refresh_plan_deterministic`: 合成 counts/coact/slot 状態 → plan(diff/victim/chunk 分割)が
     手計算と一致、同入力 2 回で同一(決定的 tie-break)。victim が newTop slot と pinned を侵さない。
  2. `chunk_swap_atomic`: 合成 staging → swap 後の slotOf/expertAt/buddyTableCPU/slotTables 内容が
     「sync ensure で同じ最終集合に到達した状態」と一致(中間 chunk 状態でも表が slot-consistent)。
- **G-B(audit, model 要)**:
  1. **opt-out 恒等**: `QWISP_BOLT_REFRESH_ASYNC=0` の OUT_TOKENS が 8237cf5 と byte-identical。
  2. **決定性 ×2**: default env(async)で 2 回実行し OUT_TOKENS 同一(固定スケジュールの証明)。
  3. RAWTESTS 全 PASS(68/68)。
- **G-C(audit コード読解)**: swap が loop 先頭(CPU turn)以外で slot/table を触らないか /
  staging 以外への background 書込みゼロか / block 待ちの決定性 / 非 bolt 経路無改変 /
  chain・verify・rebuild 経路すべてで swap 位置が out.count 基準か。
- **G-D(数値報告のみ, driver が測る)**:
  1. **slow-NAND 回復(本丸)**: `QWISP_SSD_THROTTLE_GBS=1.5` C=64 GEN=512 で sync vs async の
     bolt tok/s。目標 = shortnl 52.7 → 95+ / code 62.3 → 120+(sync 比 +80% 級)。
  2. TF@512(schedule emulation): 4 regime が sync recalib 値(88.3/74.0/93.0/92.6)± noise。
  3. fast-SSD 非退行。[BoltRecalib] preadMisses / block-wait 累積 ms を報告(B/S tuning 材料)。

## VERDICT(2026-07-07 実装完了・実測)
devloop(tests 67-68 locked)+driver 修正で採用。設計からの delta 4点:
1. **bg pread は read-ahead pipeline**(ping-pong staging ×2 + bounded-buffer semaphore)。
   「swap 後に次 chunk kick」の逐次案は chain span で loop head が少ない regime(code accept 7+)
   だと boundary burst に縮退(+13% 止まり)。pipeline 化で +48% に到達。
2. **B 既定 32 / S 既定 0=auto(半窓 R/2 均等割り)**。全幅は table 鮮度遅れで shortnl TF が
   旧 bolt 割れ(71.3<72.5)、半窓で回復(73.2)。S 詰めすぎ(S=2)は swap block=GPU idle で両軸悪化。
   B は regime split(shortnl 16 / code 64 が最適)→ TODO-2 R per-workload と同梱で knob 温存。
3. **swap の table 更新は affected layers のみ**(per-layer setBoltTable/updateRouteBiasMask
   in-place — 層独立なので full rebuild と内容同一)。全層 rebuild+40 mask 再確保は swap 毎数 ms
   ×多 swap で IO-bound 上限を食っていた。
4. **run 末尾に未 swap chunk drain(計時外)必須**: 残すと semaphore 収支 < 初期値で
   `_dispatch_semaphore_dispose` trap(SIGTRAP、pipe 出力全喪失で無言 crash)。
- GATE 実績: G-A 68/68 / G-B.1 opt-out=8237cf5 binary と byte-identical / G-B.2 決定性×2 ✓ /
  G-D slow-NAND(1.5GB/s) sync→async: shortnl 58.0→81.8(+41%) code 62.4→92.5(+48%)
  agentic 80.6 longctx 89.7。fast-SSD 非退行(±3%、agentic +7%)。
  TF@512(emulation): shortnl 73.2 / code 93.0 / agentic 92.4 / longctx 87.9 —
  全 regime 旧 bolt 以上(sync recalib 比 −0.8/±0/−0.2/−0.4 の鮮度税、採用条項クリア)。
- 残 sliver: code は IO-bound 理論上限 ~120(B=64 で 111 実測)。B/S/R の per-workload 同時 tuning
  は TODO-2。bench_batch 在プロセス連続 cell では bg thread park が累積し得る(単発 CLI は無害)。

## Doctrine
- swap・table 書換えは **CPU turn 限定・原子的**。background thread は staging 以外に触れない。
- 決定性 gate が正: swap は token-index 固定、IO 完了待ちは block(自由走行 swap 禁止)。
- opt-out(ASYNC=0)= 現行 sync と byte-identical が絶対。非 bolt tier / strict は 1 バイト不変。
- ensure() を swap に流用しない(same-call 不変量・LRU tick 汚染回避)。slot 直接代入。
- 既存 locked tests(57-66)不可侵。新 lock dir = `.locks/async-refresh`。
- GPU/build 排他(pgrep 確認)。commit 禁止(driver が gate)。G-D は driver 実測。
