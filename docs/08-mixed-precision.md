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
