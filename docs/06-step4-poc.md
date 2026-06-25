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

## 次（実 engine へ）

1. **switch_mlp を streaming 版に差し替え**（LRU キャッシュ＋ExpertLoader）→ 全40層で実モデル生成し、出力一致＆**実 tok/s・実 peak memory** を測る（シミュ gate の実機検証）。
2. 非expert 分離常駐。
3. キャッシュ方策（予測器寄り、Step2 の oracle gap を取りに行く）。
4. 混合精度 expert（hot高bit/cold低bit）。
5. MTP prefetch 統合。

> 関連: go/no-go [[go-no-go-first-read]]、実装 `tools/step4_streaming/`、決定状況 [[qwisp-open-decisions]]。
