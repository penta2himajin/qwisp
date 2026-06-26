# 08 — mixed-precision（hot 4bit / cold 2bit）go/no-go

実施 2026-06。両精度ストレージ案（hot を 4bit 維持・cold を 2bit でキャッシュ）の **GO 判定**。
**結論：GREEN（制約 RAM regime で強い reach レバー）。動作点は hot-b=64。**

mixed-precision は **resident-speed レバーではなく RAM-reach レバー**。全 expert が常駐する
regime では two-gather の compute overhead だけが乗り遅くなる（実エンジン実測 0.71x）。価値は
**budget < 全載り の制約 regime**で出る：cold を 2bit でキャッシュ常駐 → 同 RAM に ~1.8倍の
expert を載せ miss 率↓、miss 時の IO も cold は半減。これが two-gather の overhead を上回る。

## 1. 機構の正しさ（実エンジン）

`qwisp/mixed_engine.py`（`MixedSwitchGLU`、decode で hot4/cold2 の two-gather、prefill は 4bit）。

- バグ修正：`named_modules()` は MoE 層を**逆順(39→0)**で返す。`build_mixed` の連番カウンタが
  全層の expert を取り違えていた → 実 layer 番号（module 名）を使用（commit `d87bc47`）。
- `qwisp/mixed_diag.py`：単層 forward が参照 `StreamingSwitchGLU` と **bit-exact**（prefill/decode 共 0.0）。
- hot-b=256（全 4bit 経由）→ **24/24 一致**＝機構 OK。

## 2. 品質（フルモデル in-place roundtrip, `mixed_probe.py`）

| hot-b | cold | token match | 判定 |
|------:|-----:|:-----------:|:----:|
| 128 | 128@2bit | 48/48 | GREEN |
| **64** | **192@2bit** | **40/40** | **GREEN（最良動作点）** |
| 32 | 224@2bit | 13/40 | NO（token13 で発散）|
| 16 | 240@2bit | 8/40 | NO |
| 0（全2bit）| — | 発散 | NO |

**hot64 が sweet spot**：品質 GREEN を保ちつつ hot を最小化（=cold reach を最大化）。
hot32 以下は品質の崖の向こう。

## 3. 制約 regime の net_tps（`tools/step2_cache_sim/simulate_mixed.py`）

byte 予算 LRU シミュレーション。trace=2.29M 行（40層/top-8）、flash 4.0GB/s、baseline 54 tok/s、
target 15。per-expert 実測：4bit=1.77MB / 2bit=0.98MB（weight 半分・scales/biases 同）。
**式は IO のみ**モデル化（two-gather compute は含まず）＝miss 支配の制約 regime で妥当、
高 RAM 端では mixed を過大評価。総GB = expert DRAM（非expert 常駐 ~1.8GB + KV は別途）。

| 総GB | all4 net_tps | **mixed hot64** | 倍率 |
|-----:|:------------:|:---------------:|:----:|
| 2.0 | 12.7（no）| **18.3（GO）** | 1.44x ※no→GO |
| 3.0 | 15.1 | **22.5** | 1.49x |
| 4.0 | 17.7 | **27.1** | 1.53x |
| 6.0 | 23.4 | **36.2** | 1.55x |
| 8.0 | 29.7 | **44.2** | 1.49x |
| 12.0 | 41.4 | 50.9 | 1.23x（両者 baseline 54 に収束）|

- **固定 RAM で ~1.5x の net_tps**（hot64, 制約 regime）。
- **等速度なら ~半分の RAM**：all4 は 8GB で 29.7 tok/s、mixed hot64 は 4GB で 27.1。
- 2GB では all4 が NO-GO(12.7) なのに mixed が GO(18.3)＝**到達不能域を実用化**。
- hot を絞るほど速い（h32>h64>h128）が品質と背反。**品質 GREEN 境界の hot64 が最適**。

## 4. 判定と適用

**GO。** mixed-precision（hot64@4bit / cold@2bit、両精度ディスク保管）は qwisp の哲学
（8–16GB 制約マシンで実用性能を極限化）の中核レバー。reach を ~1.8倍にし固定 RAM で ~1.5x、
品質は 4bit 同等（40/40）。

**留意**：高 RAM（miss≈0）regime では two-gather overhead で逆に遅くなる（実測 0.71x）。
→ **制約 regime 専用**に切替える / もしくは MLX 融合カーネルで two-gather を解消するのが次段。
動的精度（DynaExq/HOBBIT/MxMoE 系）の「ストレージは安いので両精度持つ」前提と一致。

## 5. mixed-precision が MTP を蘇生させる（Step4 Stage B）

`tools/step4_streaming/sim_mtp_mixed.py`。Stage A（docs/06）の結論は「**MTP は max-reach
(flash-bound) regime では損、~16GB の速度ティアでしか乗らない**」。理由は verify 窓 D+1 の
union-miss が accepted より速く増え、**accepted あたり flash 仕事が D で増加**、flash 帯域が
硬い上限になるから。mixed-precision はこの致命点を直接攻める（cold miss 半減＋reach↑で miss 数↓）。

byte 予算で all4 vs mixed(hot64) を同 RAM 比較、各 RAM で最良 depth を判定。
**mixed の two-gather compute overhead も係数 1.41（resident 実測 0.71x）で保守的に計上**。
prefetch off=serial / ideal overlap=max(Tcomp,Tflash)（HOBBIT 上限）。

**最良 depth（penalty=1.41, prefetch overlap, target 比較）**:

| 総GB | all4 最良 | mixed 最良 | 機構 |
|-----:|:---------:|:----------:|:-----|
| 4 | **D0** 26.9 | **D1** 52.5 | all4 は flash-bound で MTP 損／mixed は overlap 域で MTP 勝ち |
| 6 | **D0** 42.5 | **D1** 56.1 | 同上、mixed+MTP=1.32x |
| 8 | D1 64.4 | D1 56.1 | all4 もこの辺で compute-bound 化（Stage A の ~8-16GB 域）|

- **MTP が最良になる閾値が all4 ~8GB → mixed ~4GB に下がる**。mixed では D1 が serial/overlap
  双方で最良（4-8GB 全域）。
- 固定 RAM の到達速度：mixed+MTP(D1) は **4GB で 52.5 / 6GB で 56.1 tok/s**（overlap）。
  all4-noMTP の同 RAM（26.9 / 42.5）に対し **1.3–2.0x**。serial でも 6GB 36.8 vs 23.8＝1.55x。
- D1 が常に最適（D2/D3 は streaming 下で悪化＝Stage A と一致）。
- 融合カーネル（penalty=1.0）なら mixed 6GB D1 overlap は **79.1=全載り上限に到達**（4GB でも 52.5）。

**含意（Stage A の上書き）**：MTP は「RAM 余裕時の速度ティア」専用ではない。**mixed-precision と
組むと制約 regime（4–8GB）でも MTP D1 が主役**になり、qwisp の目玉動作点に乗る。
留保は Stage A と同じ（acceptance/mult は MTPLX 全載り graft、union 独立ルーティング仮定、
overlap は理想 prefetch 上限）＋ two-gather penalty は verify バッチで実際は <1.41 の可能性。

## 6. 実エンジン実測（Step4 Stage B-systems, `qwisp/mtp_systems_bench.py`）

sim の不確実部（windowed verify の実 flash/compute）を**実エンジンで実測**。制約キャッシュ
（hot=64 を c4 常駐 / cold を c2 制約＝実 disk miss）で W=1(AR) と W=2(D1 verify) の実
per-forward レイテンシを測り、**このモデルの実測受理率 0.886**（mtplx_runtime.json）と合成:
D0=1/T(W1), D1=1.886/T(W2)。同期 pread＝**serial(no-prefetch) ブラケット**。MTP ヘッドは
未使用（受理率は graft、Stage A で実ドラフト化）。MTP 重みは `mtp.safetensors`（sidecar）に健在。

| 総GB | cfg | T(W1) | T(W2) | **W2/W1** | D0 tps | **D1 tps** | D1/D0 |
|----:|:----|:-----:|:-----:|:---------:|:------:|:----------:|:-----:|
| 6 | all4 | 70.8m | 116.7m | 1.65 | 14.1 | 16.2 | 1.14 |
| 6 | **mixed** | 73.5m | 85.1m | **1.16** | 13.6 | **22.2** | **1.63** |
| 8 | all4 | 54.9m | 76.0m | 1.39 | 18.2 | 24.8 | 1.36 |
| 8 | **mixed** | 54.6m | 67.0m | **1.23** | 18.3 | **28.2** | **1.54** |

- **核心**: mixed の **W2/W1=1.16–1.23**（verify 窓が accepted あたりほぼタダ）。all4 は 1.39–1.65
  （union-miss が重い）。cold を 2bit にすると verify の追加 miss IO が半減＋reach で miss 数↓。
- **MTP D1 は mixed で 1.54–1.63x**（all4 は 1.14–1.36x のみ）。
- **mixed+MTP(D1) vs all4-noMTP(D0) = 1.57x@6GB / 1.54x@8GB**（同 RAM）。
- sim(serial,6GB,hot64) 予測 D1/D0: all4 1.12（実 1.14✓）/ mixed 1.29（実 **1.63**＝実機が更に良い。
  per-forward 固定オーバヘッドが窓で償却＝sim 未モデル化分）。**sim の定性予測を実機が裏付け（GREEN）**。

留保: serial ブラケット（prefetch 未実装。overlap はこれ以上）/ mixed の T(W1) は masked-combine
の 2× MoE matmul overhead 込み（融合カーネルで更に改善余地、8GB では既に all4 と同等）/ 受理率は
実測 graft（Stage A で実ヘッド化）。**Stage B GREEN → 次は Stage A: `mtp.safetensors` を実装し実ドラフトで end-to-end**。

## 7. 実 MTP ヘッド + 投機デコード（Stage A, `qwisp/mtp_head.py` `qwisp/mtp_decode.py`）

mlx_lm は mtp.* を破棄するため sidecar `mtp.safetensors` を自前ロード。MTPHead = EAGLE/DeepSeek
1段ドラフト（fc 融合 + Qwen3.6 層1個[full-attn F16 + 256-expert MoE 4bit gs32] + mtp.norm +
lm_head 流用）。経験的未知を受理率最大化で確定: **concat=emb_hid / norm+1 は構造4 norm のみ
(pre_fc_*・mtp.norm は除く) / hidden は pre=post**。

**M1 受理率（teacher-forced）**: **0.929**（128tok, doc 0.886）。履歴依存大（causal 0.915 vs diag 0.52）
→ MTP KV 同期必須。

**M2 投機ループ（実機 end-to-end）**: D1 = draft(MTP) → main で [u,d] 1パス verify → v==d で
2トークン受理。**障害**: 本モデルは hybrid（30/40 が GatedDeltaNet 線形注意）で **KV cache が
trim 不能**（`can_trim=False`）→ 標準 reject ロールバック不可 → **main cache を state
snapshot/restore で巻き戻し**（KVCache/ArraysCache の `.state` round-trip 検証済）。MTP cache は
draft 位置が常に有効でロールバック不要。

| engine | 予算 | AR greedy | MTP D1 spec | 倍率 | 正しさ | live受理 |
|:-------|:----:|:---------:|:-----------:|:----:|:------:|:--------:|
| full-resident | 全載り | 31.5 | 31.8 | 1.01x | **48/48** | 0.92 |
| mixed-stream | ~8GB | 5.5 | 7.3 | **1.34x** | **96/96** | 0.94 |
| mixed-stream | ~6GB | 5.3 | 7.5 | **1.42x** | **96/96** | 0.94 |

- **正しさ: 投機出力が greedy と完全一致（48/48・96/96）** ＝ D1 verify が main 等価を保証。
- live 受理率 0.92–1.0 ＝ teacher-forced/doc と一致。**ドラフトは本物**（graft でない）。
- full-resident は 1.01x（main forward が安価で snapshot/MTP overhead が利得を食う＝Stage B と整合）。
  **mixed streaming 制約 RAM で 1.34–1.42x**（制約が強いほど main forward 高コスト→利得増）。
- Stage B systems 推定（1.54–1.63x）よりやや低い差＝snapshot（非trimmable hybrid 対策）・per-step
  sync・MTP forward overhead。async eval / 融合カーネル / 軽量 snapshot で縮む余地。
- 絶対 tok/s が低いのは naive Python ループ（per-token mx.eval、未最適化）ゆえ。倍率は同一土俵で公正。

**Stage A GREEN**: 実 MTP ヘッドで実ドラフトの end-to-end 投機デコードが**正しく動作し、制約
RAM の mixed streaming で実速度向上**を確認。mixed-precision×MTP の sim→実機まで一貫して GREEN。
