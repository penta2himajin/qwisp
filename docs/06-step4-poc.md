# 06 — Step 4 PoC：expert streaming 機構の de-risk

実施 2026-06。go/no-go GREEN（`05`）の後、最難関の systems リスク（docs `01` E-2：MLX に独立 VRAM が無く offload 先が NAND）を**一次 de-risk**した。

## PoC 4.0：on-demand expert load（✅PASS）

**問い**：MLX で「expert e をディスクから必要時に読み、常駐は hot 集合だけ、出力は正規ロードと一致、読むバイトは expert 1個分に限定」できるか。

**方法**（`tools/step4_streaming/poc_expert_stream.py`）：safetensors の `switch_mlp.{gate,up,down}_proj.{weight,scales,biases}`（先頭次元に 256 experts スタック）から、expert e の連続バイトスライスだけを pread → mx.array 復元 → `mx.load`（MLX 正規ローダ）の同 expert と比較。

**結果**：
- **bit 一致**：expert 0/5/100/255 × 全9テンソルが完全一致。
- **per-expert = 1.769 MB**（gate の 1.6MB 仮定と整合）。
- **VERDICT PASS** ＝ IO/メモリ機構は成立。compute は MLX 標準 quantized_matmul（bits=4/group_size=64）で risk 外。

→ **E-2 の核心リスク（前例の薄い MLX+NAND streaming）は「動く」ことが実機で確認**。プロジェクトで一番不確実だった部分が GREEN に。

## 重要発見：MLX は safetensors をネイティブに mmap

`mx.load`(5.4GB シャード)が **0.1 秒**で返った＝**遅延/mmap ロード**。MLX は mmap-from-NAND の半分を内蔵している。

→ 設計の分岐が見えた：

| 方式 | trade-off |
|---|---|
| (a) 手動 pread＋自前キャッシュ | ポリシー制御可（Step2 の oracle gap を取りに行ける）／実装多 |
| (b) mx.load 遅延 mmap に委任 | 実装激減／eviction は OS-LRU でポリシー不可 |

おそらく **(a) を主**にしつつ（予測器キャッシュで Belady gap を取りに行く価値が Step2 で定量化済）、(b) を初期の足場に使うのが筋。

## PoC 4.1：streaming MoE forward の出力一致（✅PASS、compute 層）

**問い**：全256 experts 常駐でなく「必要 experts のサブセットだけ」読んで計算した routed 出力が full と一致するか。

**方法**（`tools/step4_streaming/poc_moe_forward.py`）：実 hidden state を捕捉し、top-k で選ばれた experts のサブセット（unique）だけを ExpertLoader で読み、`gather_qmm`（mlx_lm の QuantizedSwitchLinear/SwitchGLU に忠実）で計算 → full の `switch_mlp(x, inds)` と比較。token<8 で sort 回避し厳密一致を取る。

**結果**：4 token / top_k=8 → unique **29/256 experts**。**max|diff| = mean|diff| = 0.00e+00（bit 完全一致）**。VERDICT PASS。

→ **必要な 29 experts だけ常駐で routed 出力を full と完全同一に計算できる**。streaming MoE 層の compute が正しい。

## engine コアの両層が GREEN

| PoC | 層 | 結果 |
|---|---|---|
| 4.0 | IO/メモリ | expert on-demand load が bit 一致、1.77MB/expert |
| 4.1 | compute | subset-only forward が full と bit 一致（diff=0） |

「正しく動く streaming MoE 層」が実モデルで端から端まで成立。AFM3-CA の中核機構が Qwen3.6-35B-A3B で動くことが実証された。

## 留保

- PoC 4.0 の cold 計測（8GB/s）は直前 `mx.load` の mmap でページが温まり過大評価。**gate の帯域は bench_flash の clean 4 GB/s を維持**。
- PoC は**1層・少 token・no-sort** で実証。実 engine では sort 経路・全40層・実 tok/s（実機 streaming のオーバーヘッド）が残課題。

## Stage A：MTP × streaming は両立するか（trace-sim、`sim_mtp.py`）

**問い**：MTP 投機デコード（depth D）は expert streaming と同時に使えて実際に速くなるか。

**モデル**：verify は窓 K=D+1 トークンを1パス → 各層で expert 和集合に触れる。OUR 実測 AR 54tok/s を基準に MTPLX の depth 倍率（D1 1.465×/D2 1.436×/D3 1.140×）＋受理率を graft し T_compute(D)。streaming ペナルティ T_flash(D,B)=verify 窓の union-miss×expert_bytes÷flash_bw（prefill で温め後 decode を窓処理、OUR 4.18GB/s）。prefetch 2ブラケット: serial=T_compute+T_flash / overlap=max(T_compute,T_flash)。

**結果**（net_tps の best depth）:

| B/層 | DRAM | serial(prefetch off) | overlap(prefetch on) |
|---:|---:|---|---|
| 32 | 2.3G | D1 14.0 | **D0 18.1**（MTP 負け）|
| 64 | 4.5G | D1 21.1 | **D0 30.6** |
| 96 | 6.8G | D1 30.2 | **D0 51.7** |
| 128 | 9.1G | D1 40.5 | **D1 79.1**（逆転）|
| 256 | 18.1G | D1 69.0 | D1 79.1 |

**核心**：`miss/verify` は ~(D+1)倍にスケールするが accepted は sub-linear（1.0/1.89/2.43/2.40）→ **miss/accepted-token が D とともに増える**（B=64: 77→82→94→129）。

**結論：両立するが効くのは regime 次第。**
- compute-bound（B≥128, ~16GB）: MTP 効く。D1 で 1.4-1.5×。
- flash-bound（B≤64, ~12GB＝max reach 域）: MTP 効かない/損。**prefetch でも救えない**（帯域が硬い上限＝HOBBIT 留保の定量確認）。
- **D1 が常に最適、D2/D3 は streaming 下で悪化。**

**設計含意（reach↔速度の2動作点）**: Max reach=12GB/素 AR streaming/17-21tok/s vs Speed=16GB/MTP D1/40-79tok/s。**MTP は 12GB reach の目玉には乗らず、RAM 余裕時の速度ティア。**

留保: draft 窓を AR 連続トークンで近似、倍率/受理は MTPLX マシン graft、union は独立ルーティング仮定（実測 miss が ~線形＝窓内相関低）。

## アーキ決定実験（`probe_lazy_load.py`）→ (A) 明示キャッシュ必須

`mlx_lm.load` が expert を遅延 mmap に保つか（→OS 委任の (B) が使えるか）を RSS で実測：
**load 前 0.10GB → load 後 20.14GB → 生成後 20.14GB（mx_peak 19.6GB）**。
＝**全 20GB を eager 実体化**。(B) 遅延 mmap 委任は**不可**（12GB 機では load 破綻）。

→ **(A)：expert を materialize しないカスタムロード経路＋自前キャッシュ**で確定。
ポリシー制御が効く＝Step2 の Belady gap（予測器）を取りに行ける＝Qwisp の差別化点。

## 前例（戦略上の注意）

mlx-swift で同空間の実装が既に存在：
- **SwiftLM**（github SharpAI/SwiftLM）: ネイティブ MLX-Swift、100B+ MoE の SSD streaming、~10GB resident で 122B、OpenAI 互換、macOS/iOS アプリ。
- **TurboQuant-MLX**: Qwen 122B を 16GB Mac mini で MoE expert streaming。

→ full-Swift は viable（将来の製品化の道）。docs `01` E-2 の「未踏ニッチ」前提は弱まり、差別化を**測定駆動のキャッシュ方策／Qwen3.6 特化／2 動作点設計**に絞る必要。**ゼロから作る前に SwiftLM/TurboQuant を再利用/土台候補として精査**する価値が高い。

## 次（実 engine v0.1）

検証は Python/MLX（モデル＋MTP が既存、差別化は*方策*）。将来製品化は Swift。
1. **カスタムロード**：非expert（~3.6GB）常駐、switch_mlp.* は未ロード。
2. **StreamingSwitchGLU**（PoC 4.1 製品化）＋ ExpertCache（LRU→予測器）＋ ExpertLoader。
3. 全40層で実生成 → `verify.py` で full と一致＋**実 tok/s・実 peak**（Max reach 動作点で裏取り）。
4. 以降: async prefetch / 予測器キャッシュ / 混合精度 / MTP（Speed 動作点）。

> 関連: go/no-go [[go-no-go-first-read]]、実装 `tools/step4_streaming/`、決定状況 [[qwisp-open-decisions]]。
