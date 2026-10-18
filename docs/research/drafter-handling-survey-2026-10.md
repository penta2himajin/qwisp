# Drafter handling in speculative decoding: survey for qwisp's fast mode (2026-10)

日付: 2026-10-16。対象機: MacBook Pro M3 Pro（14-core GPU、36 GB）、Ornith-1.5-35B-A3B / Qwen3.6-35B-A3B
MLX 4-bit。この調査では GPU を一切使っていない。手段は source の読解（ローカル tree と shallow clone）、
論文、web ページである。

前提として次の 2 本を参照し、その内容は繰り返さない。

- [dflash-drafter-feasibility-2026-10.md](dflash-drafter-feasibility-2026-10.md)（以下 **FEAS**）:
  drafter の構成、#98 の経緯、expert-union によるバイトモデル（§5.1）、ステップコストモデル（§5.3）、
  GDN tape replay の案（§5.4）、DeviceCalibration の案（§7）、oMLX 0.7.0 実測（§8）。
- [inference-optimization-survey-2026-10.md](inference-optimization-survey-2026-10.md)（以下 **OPT**）:
  §4.1（MoE の verify コスト: MoESD / Cascade / EVICT / EcoSpec / MoE-Spec / mlx-lm #990）、
  §4.2（少行 verify kernel）、§4.3（SuffixDecoding と draft 長制御）。

表記: 「推測」は筆者の推論、「推定」は数値モデルによる見積もりを示す。外部の数値はすべて
出典側の自己申告で、qwisp では再現していない。F は M3 Pro での 1 トークンあたり decode 時間
（約 16.7 ms）、r_M は M 行 verify の 1 行 decode に対するコスト比（M3 Pro 実測: r_2 1.23、
r_4 1.86、r_9 3.92、r_17 6.94）を表す。

## 要約

- **FEAS §8 の oMLX 0.7.0 実測を動かしていたのは、dflash-mlx の固定閾値 policy だった。** 同梱
  dflash-mlx 0.1.10+omlx.9 の `_AdaptiveBlockPolicy` は、直近 4 cycle の受理 < 0.75 または
  tokens/cycle < 3.5 で block 4 に落とし、64 cycle ごとに full block を試す。r_M（verify コスト）は入力に
  無い。受理 < 0.75 は tpc < 4.0 と同値なので、block 6/8 設定の測定は大半が 4 行 verify だったはずで
  ある（推測）。ただし block 8 設定が block 4 固定に負けた理由はこれだけでは説明できず、未解決。oMLX main の
  `_DepthController` はこれとは別物で、深さごとの実測時間と位置別受理から
  `score(d) = (1+p1+p1p2+…)/t(d)` を最大化する。
- **深さ制御の決定則は「受理推定 × 実測コスト表の argmax」に収束している。** EVICT、vLLM adaptive
  verification、SmartSpec/TurboSpec、oMLX `_DepthController` がこの形である。MoE の verify コストを
  明示的に扱うのは Cascade と EVICT だけである。batch size 依存の gate（vLLM、SGLang、TRT-LLM）は
  batch 1 の qwisp には移せない。
- **M3 Pro の r_M に当てはめると、最適な verify 幅は 4〜5 行である（推定）。** 損益分岐 tokens/cycle は
  4 行で 2.19、8 行で 3.84。FEAS §8 の tokens/cycle では、4 行で code 1.34×、shortnl 1.00×、8 行の code は
  0.68× と予測される。帯域律速機での MoE target の公開実測（MTP/DFlash）も 1.1〜1.25×、最良深さ 2〜3 に
  集まっている。
- **4-bit drafter で受理はほぼ落ちない**（0〜2.4%: arXiv 2607.04244、llama.cpp #27342、qwisp 実測）。
  語彙の刈り込み（FR-Spec 32k で受理 93%）は効果と損失が拮抗するので、測ってから決める。
- **GDN の rollback は、serving エンジンも位置ごと snapshot から入力 tape + replay に移りつつある**
  （vLLM RecoverSSM、SGLang ReplaySSM-spec、ReplaySSM 論文、dflash-mlx）。qwisp の形状では、位置ごと
  snapshot は block 8 で毎 verify 約 480 MiB（約 0.27F）かかるのに対し、tape replay は約 0.07F で済む
  （推定）。逐次 greedy との完全一致を保証する tape 実装は存在しない。
- **suffix draft とモデル draft の切り替えは「suffix の期待受理 ≥ モデル drafter の期待受理なら suffix」に
  収束している**（SuffixDecoding 論文、Arctic Inference）。suffix tree と DFlash2 を併用した実装は
  見つからなかった。
- **提案の P0**: 未知量（位置別受理、M=2..9 の r_M、実ルートの expert union、drafter 時間）を測る。
  そのうえで、block 8 で draft して verify 幅をコストモデルの argmax で選ぶ（d=0 が drafter off で、
  shortnl もここで扱う）。drafter は incoai DFlash2 の 4-bit g64、GDN rollback は tape replay にする。

## 1. 受理率から draft 深さ / block 長を決める制御

### 1.1 論文: 制御の信号で 4 系統に分かれる

| 系統 | 例（実在を arXiv で確認） | 決定則 | 信号 |
|---|---|---|---|
| draft 確信度での打ち切り（学習不要） | AdaEDL [2410.18351]、SVIP [2411.18462]、EAGLE-2 [2406.16858] | AdaEDL: `1 − sqrt(γ·H_draft) < λ` で停止（γ=0.2）。λ は受理率 EMA（β1=0.5）を目標 0.9 と比べて増減。SVIP: `sqrt(H_draft) > h`（h=0.3）で停止。EAGLE-2: 経路上の draft 確信度の積を node 価値とし、上位 m node を verify（深さは固定 6、形だけ動的） | draft の entropy / 確率 |
| 学習した受理予測器 | SpecDec++ [2405.19715]、DISCO [2405.04304]、AdaEAGLE [2412.18910]、HeteroSpec [2505.13254] | SpecDec++: 予測受理確率の積から `P(どこかで reject) > h` で停止（h=0.7 推奨、閾値型が最適と証明）。DISCO: 2 層 FFN の確信度 < τ で停止 | 追加ヘッドの出力 |
| コストモデル / goodput 最大化 | SmartSpec→TurboSpec [2406.14066]、EVICT [2605.00342]、Cascade [2506.20675]、TETRIS [2502.15197] | SmartSpec: `k = argmax (期待受理+1)/(モデル化した batch 時間)`、期待受理は移動平均 α から `(1−α^{k+1})/(1−α)`、k=0 で投機停止。EVICT: `k* = argmax Ê[A(T_k)] / C(k)`、C(k) はオフラインで測った step 遅延表（k=1..32）。Cascade: 効用 `U = ETR_spec / (t_spec/t_base)`、試験相（4 反復 × 最大 4 試行で K を山登り）と設定相（16 反復）を交互に回し、U<1 なら K=0 として設定相の長さを倍々に延ばす | 測った遅延 + 受理率 or draft 確率 |
| bandit | BanditSpec [2505.15141]、Nightjar [2512.22420] | BanditSpec: 腕 = 投機長 γ∈{0..4}（0 は投機なし）または draft 手法。UCBSpec で選ぶ。Nightjar: batch size ごとの文脈つき bandit、γ=0 からの切替ペナルティつき | 実測 tok/s |

数値（いずれも出典の自己申告）: SpecDec++ は 2×A100 で固定 K 比 +7〜11%。AdaEDL は A100 で静的長比
+10〜57%。Cascade は RTX 6000 Ada 上の MoE 5 種で、最悪の減速を約 5%（静的 K=1/2/3 では 26/38/54%）
に抑え、静的 K 比 +7〜14%。EVICT は A100 の SGLang 上で Qwen3-30B-A3B ほかに対し EAGLE-3 比 1.21〜1.25×。
平均受理長は 3.65 から 3.09 に下がるが、活性 expert は 32.5% 減り、verify 遅延は 26.6% 減る。
PEARL [2408.11850] は draft と verify を並行させる方式なので、GPU を 1 つしか持たない Apple Silicon の
単一ストリームには当てはまらない（推測）。

MoE の verify コストを明示的にモデルに入れているのは Cascade と EVICT だけである（OPT §4.3 と同じ結論）。
batch 1 では SmartSpec / TETRIS / Nightjar の batch 依存部分は定数になるので、移せるのは
「受理の推定 × 実測コスト表の argmax」という骨格だけである。

### 1.2 エンジンの実装（ソースで確認した決定則）

- **llama.cpp**（`common/speculative.cpp` @ `f12aaa3`, 2026-10-16）: draft の top-1 確率が `p_min` を
  下回ったら draft を打ち切る（`cur_p->data[0].p < params.p_min`）。DFlash2 では lattice を貪欲にたどり、
  argmax の softmax 値 `1/Σexp(s_k−s_max)` が `p_min` を下回ったら止める。draft 長が `n_min` 未満なら
  その step は投機しない。既定は `n_max=3, n_min=0, p_min=0.0`（`common/common.h`）。受理率 EMA や
  コストによる on/off は無い。
- **vLLM**（@ `074c83a` / `e365792`, 2026-10-15〜16）:
  - Dynamic SD（`num_speculative_tokens_per_batch_size`）は batch size → K の静的表で、K=0 が off。
  - **adaptive verification**（`vllm/v1/worker/gpu/spec_decode/adaptive_verification.py`, PR #47808,
    既定 off）。draft 位置ごとの受理推定値の cumprod を「生存確率」とし、全要求で降順に並べ、
    `draft_budget = argmax((num_sampling_reqs + cumsum(scores)) / (draft_cost_ms + verify_cost_ms[n : n+budget+1]))`
    で verify する draft 数を決める（筆者がソースを読んで確認）。コスト表は起動時に CUDA graph 単位で
    プロファイルする。受理推定は DSpark の確信度ヘッドか、オンラインで再較正する小さなロジスティック
    モデル（`acceptance_estimator.py`）から得る。draft はすべて生成し、削るのは verify だけである。
  - 未マージの #52559 は DFlash の K を {0,1,3,7,15} から、測った drafter/target コストと受理生存の EMA で
    選ぶ。
- **SGLang**（@ `0693457`）`--speculative-adaptive`（PR #21599）: 受理 draft 数の EMA（α=0.2、warmup
  10 batch、5 batch ごとに再判定）にヒステリシス（±0.5）をつけて段を上下させる。batch 1〜7 の既定段は
  [1,3,5,7]、batch 64 以上は [0]（off）。段 0 から次の判定では最小の正の段を試す。EAGLE/EAGLE3
  topk=1 だけが対象で、DFlash は固定 block である。
- **TensorRT-LLM**（@ `0c1882b`）: `max_concurrency` と `draft_len_schedule`（batch size → draft 長）に
  加え、`acceptance_rate_window_size` / `acceptance_rate_threshold` を持つ。移動平均が閾値を下回ると
  投機は**永久に**無効化され、戻らない。
- **HF transformers**: `assistant_confidence_threshold` を、(draft 確率, 一致) の組から ROC で
  `argmin(fpr + 3·fnr)` としてオンラインで再推定する（DISCO の教師なし版）。
- **mlx-lm**: `num_draft_tokens` は固定（既定 2）。適応は無い。

### 1.3 Apple Silicon の実装: dflash-mlx と oMLX

**FEAS §8 の oMLX 0.7.0 実測で働いていたのは dflash-mlx の `_AdaptiveBlockPolicy` である。FEAS §4.1 が
説明する oMLX main の `_DepthController` ではない。** 根拠は次のとおり。

- `/Applications/oMLX.app` 同梱の oMLX は 0.7.0（2026-10-01 tag）。Qwen3.5 系で batched drafter を
  使う条件は `engine_type == "vlm"`（`omlx/utils/model_loading.py:1237`）だが、Ornith の config は
  `qwen3_5_moe` で `vision_config` を持たない。
- FEAS §8 が読んだログ行 "DFlash generation complete" は、同梱 tree では `omlx/engine/dflash.py`
  （`DFlashEngine`）にしか無い。この engine は同梱の `dflash_mlx 0.1.10+omlx.9` を包んでいる。
- oMLX main で dflash-mlx engine を外して batched drafter に一本化したのは commit `7a4ea399`
  （2026-10-14, #4508）で、0.7.0 より後である。

（以上はコードパスからの推測だが、根拠は強い。）

同梱 `dflash_mlx/engine/spec_epoch.py` の `_AdaptiveBlockPolicy`（`verify_mode="adaptive"` が既定）:

- full block ≤ 4 なら policy は無効になる。つまり block 4 の測定は固定 4 行だった。
- full block（6 または 8）で直近 4 サイクルの `acceptance = (commits − cycles)/commits < 0.75` または
  `tokens/cycle < 3.5` なら reduced（block 4）へ落とす。
- reduced で 64 サイクル経つと full を 8 サイクル試す（早期 probe は 16 サイクル以上かつ直近 8 回の
  全受理率 ≥ 0.75 のとき 4 サイクル）。probe の受理 ≥ 0.80、tokens/cycle ≥ 3.5、かつ probe の tok/s ≥
  reduced の tok/s なら full に戻る。
- prompt が 32,768 token 以上なら最初から reduced で始める。
- `copyspec_mode` の既定は `conservative`（§7.3）。

この定義では `acceptance = 1 − 1/tpc` なので、「受理 < 0.75」は「tpc < 4.0」と同じ条件になる。
FEAS §8 の block 6/8 設定は、どの cell も要求平均で tpc ≤ 4.0 だった。したがって大半の cycle は reduced
（4 行）で走り、受理の高い区間と probe の間だけ 6/8 行で verify していたはずである（推測）。
ただしこれでは、block 8 設定の code が block 4 固定より tokens/cycle で低く（2.61 対 2.93）、tok/s で
18% 遅い理由までは説明できない。drafter が reduced 時に短い block で draft する分布の違いや、probe の
コストが候補として考えられるが、未解決である。次に oMLX 0.7.0 で測るときは、events に出る
`adaptive_block_reductions` / `adaptive_block_cycles` / `adaptive_block_min` を記録すれば決着がつく。
いずれにせよこの閾値（0.75 / 3.5 / 0.80）は固定定数で、verify コスト（r_M）を入力に持たない。

oMLX main（`/Users/nara/repos/omlx` @ `958974c1`）の `_DepthController`
（`omlx/patches/mlx_lm_mtp/batch_generator.py`）は、コストを実測する設計である。

- `score(d) = (1 + p1 + p1·p2 + … ) / t_est(d)` を最大にする d を選ぶ。p_j は位置 j の条件付き受理の
  EMA（`ALPHA=0.08`）、t[d] は深さ d のサイクル実時間 EMA（時間窓 `TAU_MS=400`、外れ値は 2 倍超で
  重み 0.25）。
- 未測定の深さは、測定済みの両端から求めた傾き（1 行あたりの限界コスト）で外挿する。
- 起動時に max..1 の各深さと深さ 0（通常 decode）×3 を 1 回ずつ測る warmup sweep を行う。
- 切り替えは 3% 超の改善のときだけ（`HYSTERESIS=1.03`）。最良の対抗深さを probe する頻度は
  サイクルの 15% 以内に抑える（`PROBE_DUTY`）。
- 投機の最良 score が深さ 0 の `EXIT_MARGIN`（既定 1.15、実測した「ループ税」で置き換え）倍に
  届かない判定が 16 回続くと（`EXIT_STREAK=16`）、その要求を通常 decoder に戻す。
- docstring には、固定の浅め bias・純粋な tok/s bandit・per-cycle の学習補正などを M3 Ultra で A/B
  して退けた経緯が書かれている（「pure realized tok/s bandit (exploration tax)」）。

mlx-vlm（@ `4623cf9`）の DFlash2 は `_dflash_next_block_size` で直近 8 ラウンドの受理率から block を
決める。受理率 < 0.30 または平均受理 < 2.0 なら縮め、< 0.50 なら −2、≥ 0.85 かつ全受理率 ≥ 0.75 なら
+2。初期値 3 行、上限は `min(5, block_size)`。これが README の「three to five rows」の実体である。

## 2. verify コストのモデル化と MoE ターゲット

### 2.1 qwisp の既知事項（再掲しない）

expert-union のバイトモデルと M1 Max 実測 r_M の一致は FEAS §5.1 にある。MoESD / Cascade / EcoSpec /
MoE-Spec / mlx-lm #990 は OPT §4.1 にある。ここでは新しい一次資料だけを足す。

### 2.2 実ルーティングでの expert union（新しい測定）

- **"The Limits of Speculation: Bounding Speculative Decoding in Mixture-of-Experts"**
  [arXiv 2609.22156, 2026-08-26]。1×A100-80GB、bf16、batch 1、vLLM fork（full CUDA graph）、
  Qwen3-Coder-30B-A3B-Instruct（128 expert, top-8）+ SpecForge EAGLE-3。Table A1 の中央値:

  | K（draft 数） | 検証位置 m | verify 時間 T_T | 層あたり unique expert |
  |---:|---:|---:|---:|
  | 1 | 2 | 8.98 ms | 12.1 |
  | 5 | 6 | 11.51 ms | 22.6 |
  | 10 | 11 | 13.67 ms | 30.2 |

  一様独立ルーティングなら 128 expert / top-8 で m=2 は 15.5、m=6 は 41.1、m=11 は 65.1 になる
  （筆者計算）。実測はその 55〜78% にとどまり、同一系列の連続トークンは expert を強く共有している。
  著者の oracle では受理長は L_max=6 を超えると約 2.1 で飽和し、速度上限は約 2.34×（オフライン反実仮想）。
- **XShare** [arXiv 2602.07265]: 同じ要求の連続した投機トークンどうしの expert 重なりは、独立トークン
  どうしより「2–3× larger」と報告している（batched serving の文脈）。
- **qwisp との緊張関係（未解決）**: FEAS §5.1 では、M1 Max の r_M が**一様**モデルのバイト比と M=9 まで
  0〜7% で一致した。実ルーティングの union が一様モデルより小さいなら、bytes 以外の項（per-row の
  attention/GDN/dequant 計算や、kernel が union を活かせていない部分）が同程度あることになる。
  どちらなのかは、qwisp の verify 行の実ルートから U(M) を数えれば決まる（提案 P0-2）。

### 2.3 MoE で投機の利得が小さい理由と、文献にある緩和策

- MoESD [2505.19645] の主張は「中くらいの batch では MoE の方が dense より SD の恩恵が大きい」で、
  batch 1 の利得は主張していない。目標効率 `T(B,1)/T(B,γ)` が受理率より速度をよく説明すると述べている。
- EAGLE-1 [2401.15077] は Mixtral 8x7B の MT-bench で 1.50×（τ=3.25）にとどまり、複数トークンの verify で
  より多くの expert を読むことを理由に挙げている。
- 緩和策は 3 種類ある。
  1. **draft を短くする**: L4 上の llama.cpp MTP sweep（§6）で最良は n_max=2。
  2. **verify する draft を受理推定 × 実測コストで削る**: EVICT、vLLM adaptive verification。
     lossless のまま使える。
  3. **verify で使う expert 数に上限を置く**: MoE-Spec [2602.16052]。出力分布を変える（lossy）。
     qwisp の fast mode は「数値の非 bit-exact」は許すが、ルーティングを変えるのは別次元の近似なので、
     品質評価なしには採らない（推測にもとづく方針）。

### 2.4 エンジンのコスト較正

| 実装 | 較正の仕方 |
|---|---|
| vLLM adaptive verification | 起動時に draft/verify の step 時間を CUDA graph 単位で測って表にする（合成 8192 ctx、`VLLM_ADAPTIVE_VERIFICATION_PROFILE_CONTEXT_LEN`） |
| EVICT | オフラインで k=1..32 の step 遅延を各 32 回測った表 |
| oMLX `_DepthController` | オンライン。warmup sweep で各深さを実測し、限界コストは測定済み深さ間の傾きとする。通常 decode との差（ループ税）を model インスタンスに保存して次の要求に引き継ぎ、10 分で既定値に減衰させる |
| dflash-mlx | `runtime/chip_detect.py` で `applegpu_g{gen}{suffix}` から世代 / tier / NAX 有無を判定する。verify 専用 int4 qmm（M=16）を「MoE target と 40 層以上の dense」で自動有効化する |
| z-lab/dflash MLX、mlx-vlm | 実測ではなく固定上限。「quantized target/draft では `block_size <= 5`」（z-lab README）。mlx-vlm README: 「eight rows exactly fill the verifier threadgroup, while nine or ten rows pad to sixteen」 |

推測: verify 行のコストが MLX の kernel 形状（8 行、16 行の境界）で段差を持つことは、qwisp の自前
kernel には当てはまらない。qwisp は自分の r_M 曲線を測ればよい。M3 Pro 実測（r_2 1.23、r_4 1.86、
r_9 3.92）は FEAS §5.1 の一様バイトモデル（1.31 / 1.89 / 3.19）と M≤4 でほぼ一致し、M=9 では 23% 上回る。
M3 Pro では M≥5 あたりから計算の寄与が見え始めている（推定）。

## 3. drafter の量子化と軽量化

### 3.1 重みの量子化: 4-bit drafter で受理はほとんど落ちない

| 出典 | 設定 | 結果 |
|---|---|---|
| FEAS §8（qwisp 実測、M3 Pro、oMLX 0.7.0） | incoai DFlash2、Ornith 4-bit target | 4-bit g64 が最速。tokens/cycle は 8-bit と同等。BF16 はさらに遅い |
| Kim et al., "Quantize the Target, Quantize the Drafter" [arXiv 2607.04244] Table 5 | Qwen3.5-4B W4A16 target、DFlash drafter（5 層、537M）、block 16 | 平均受理長は BF16 5.03 に対し INT4 RTN 4.91、AWQ 4.92、GPTQ 4.98。GPTQ drafter で速度 3.32× → 3.55×（遅延は RTX 5000 Ada で測定） |
| llama.cpp PR #27342（2026-08-27 merge） | M5 Pro 64 GB、Qwen3.8-27B Q4_K_M、DFlash2 block 8 | 受理 BF16 4.92 / Q8_0 5.08 / Q4_K_M 5.03、decode 19.31 / 18.43 / 18.89 tok/s |
| mlx-lm discussion #890 | M3 Ultra、Llama-3.1-8B 4-bit、EAGLE-3 head | 4-bit head でも受理 0.34 は不変、0.94× → 1.05× |
| incoai/splash `DEVELOPMENT.md` | Qwen3.6-35B-A3B | drafter の各 projection を「af4g64 planes」に量子化し、target の block kernel で回す |
| dflash-mlx README | — | 「Validated large DFlash drafts default to `w4` in memory」 |

量子化の向きはどの資料でも同じで、受理の低下は 0〜2.4% である。dense 27B が target の llama.cpp では
4-bit drafter による速度の差がほぼ見えない（target が支配的なため、推測）。M3 Pro の qwisp は drafter の
バイトが step の数割を占めるので、4-bit の効果は大きい（FEAS §5.2、推定 0.74F → 0.27F/block）。

### 3.2 語彙の刈り込み（lm_head）

- **FR-Spec** [arXiv 2502.14856, ACL 2025]: Llama-3-8B の drafting 時間のうち lm_head が 49%（softmax を
  含めると 62%）。頻度上位の部分語彙に絞ったときの受理長は、全語彙 3.89 に対して 64k で 3.80（97.7%）、
  32k で 3.63（93.3%）、16k で 3.40、8k で 3.13。A800 上で EAGLE-2 比平均 1.12×。target 側は全語彙で
  verify するので lossless である。
- **VocabTrim** [arXiv 2506.22694]: 128k → 約 32k で受理（block efficiency）は 1〜5.6% 低下。
- 学習時に語彙を刈る方式 [arXiv 2603.05210]: 13,264 語彙（−90%）で τ は MT-Bench 2.89 → 2.65。
- EAGLE-3 の公開 checkpoint は `draft_vocab_size 32000` と d2t/t2d 写像を持つ（例:
  `lmsys/SGLang-EAGLE3-Qwen3-Coder-30B-A3B-Instruct-SpecForge`、vocab 151,936）。

qwisp への当てはめ（推定）: Qwen3.6 の語彙は 248,320、target の 4-bit lm_head は 286 MB で、FEAS §5.2 の
4-bit drafter 1 block あたり約 0.50 GB の過半を占める。32k に刈れば lm_head は約 37 MB になり、
block あたり約 0.25 GB、つまり約 2.3 ms（約 0.14F）減る。ただし DFlash2 の selector は各位置の上位 16
候補を logits から取るので、刈り込みは candidate 集合も変える。FR-Spec の 32k で −6.7% 程度の受理低下が
起こりうる。数値は qwisp の regime で測るまで仮置きである。

### 3.3 層の削減と block 長

- 層を削る drafter（LayerSkip [2404.16710]、Kangaroo [2404.18911]）はどれも再学習を前提にしている。
  公開 DFlash/DFlash2 drafter は 6 層で固定なので、qwisp で層を減らすには再学習が要る。対象外とする。
- DFlash 論文 [2602.06036] Table 8（Math500）では、block 16 で学習した drafter を block 8 で使っても
  3.87× / τ 5.09 で、block 8 で学習したもの（3.97× / 5.21）にほぼ等しい。逆に block 8 学習を block 16 で
  使うと 3.78× / 5.02 で伸びない。論文は「大きい block で学習したものは小さい推論 block に汎化するが、
  逆は成り立たない」と述べ、計算律速の環境では block を縮める方が速いこともあると書く（要約経由の引用）。
  したがって、trained block 8 の drafter を 3〜5 行で verify する運用は学習条件と矛盾しない。
- 4-bit target で tree verify を使うと verify コストが利得を食う。HierSpec [2505.22179] はこれを理由に
  tree draft を sequence draft に変換している。DFlash2 も tree ではなく候補経路を 1 本選ぶ chain verify で
  ある（§5）。

## 4. 再帰状態（GDN / SSM）のロールバック

FEAS §5.4 は qwisp 側の現状を記述している。ping-pong snapshot は直前 1 step しか戻せないので、
partial reject のたびに rebuild forward が走る。FEAS §5.4 はそれを置き換える tape replay を提案した。
ここでは他の実装を 4 方式に分けて比べる。

### 4.1 4 つの方式

| 方式 | 中身 | 採用例（ソースで確認） |
|---|---|---|
| A. draft 位置ごとの全状態 snapshot | verify 中に各位置の状態を別 slot に書き、受理長で slot を選ぶ | vLLM 既定（`gdn_attn.py`: `spec_state_indices_tensor` は `[batch, num_spec+1]`、kernel は `num_accepted_tokens − 1` の slot から読み始める）、SGLang 既定（`intermediate_ssm_state_cache` は層 × draft token ごとに全状態）、llama.cpp（`n_rs_seq` snapshot plane、`seq_rm` で plane を差し替えるのでコピーは無い）、mlx-lm PR #1821（`RecurrentCache`） |
| B. checkpoint + 再計算 | verify 前の状態に戻し、受理分を通常 forward でやり直す | llama.cpp server の speculative checkpointing（PR #19493）、Mamba-in-the-Llama [2408.15237]、mlx-lm PR #990（confirmed prefix と draft を 2 chunk に分け、間で snapshot）、**qwisp の現行 rebuild forward** |
| C. 入力（innovation）tape + replay | read-only の checkpoint と、トークンごとの小さな記録（delta または補正、k、g、conv 入力）を残し、受理分だけ畳み込む | dflash-mlx（`recurrent_rollback_cache.py`）、vLLM RecoverSSM（GDN 版は PR #58863 / #61309、open）、SGLang `--enable-linear-replayssm-spec`、ReplaySSM [2610.13778]、SpecLA [2607.16673] |
| D. 出力専用 verify + 代数的再構成 | verify 時は状態を書かず、WY/UT 変換で各位置の出力を出し、commit 時に受理状態を再構成する | TreeWY [2608.20961]、SGLang ReplaySSM-spec（UT 変換で output-only）、STree [2505.14969]（SSM の tree） |

2026 年の serving エンジンは A から C/D に移りつつある。理由はメモリ帯域と容量である。
vLLM PR #61309（Qwen3.5-4B + MTP、1×L40S）は c=32、K=5 で +8.6% だが、c=1 では利得が無く、K=1 では −3%。
PR #58863 も c=1、K=2 で 1.9〜4% 遅いと書く。TreeWY も、メモリが逼迫したときだけ最大 1.49× を出し、
余裕があるときは 0.97〜0.99× だった。

### 4.2 qwisp の形状でのコスト（推定）

Ornith の GDN は 30 層、value head 32、key head 16、head dim 128、conv kernel 4（`config.json`）。
状態は 1 層 32×128×128。f32 なら 2 MiB/層、30 層で 60 MiB である。

| 方式 | verify 1 回あたりの追加トラフィック（block 8） | M3 Pro での時間（~110 GB/s） |
|---|---|---|
| A. 位置ごと snapshot | 8 × 60 MiB ≈ 480 MiB の書き込み | 約 4.6 ms ≈ 0.27F（毎 step。full accept でも払う） |
| B. rebuild forward（現行） | r_(p+1)·F の forward 1 回 | block 8 で約 1.5〜2.0F。partial reject の step だけ |
| C. tape + replay | tape 書き込み ≈ 40 KiB × 30 層 × 8 行 ≈ 10 MB。replay は状態の読み書き 120 MiB | 約 0.1 ms + 約 1.1 ms ≈ 0.07F。partial reject のときだけ |

tape の 40 KiB/行・層は dflash-mlx の構成から求めた（delta `[Hv,Dv]` f32、k `[Hk,Dk]`、g `[Hv]`、
conv 前の qkv 8192 要素 bf16）。帯域律速の M3 Pro では A は割に合わない。C は FEAS §5.4 の見積もりと
整合する。

### 4.3 実装の細部

- **dflash-mlx**（同梱 0.1.10+omlx.9 と upstream `6080323` で同じ構造）:
  - `arm_rollback()` で conv/ssm を checkpoint する。
  - verify kernel `gated_delta_kernel_with_tape` は `innovation_tape = (v − kv_mem)·β`（`state *= g` の後）を
    f32 で出力する。`record_tape(tape, k, g, qkv)` で保存する。
  - `rollback(n)` は snapshot に戻してから、`tape_replay_kernel` で `state = state·g + k ⊗ delta` を
    受理 n+1 step だけ回す。q も出力も計算しない rank-1 更新である。
  - conv は `concat(snapshot_conv, tape_qkv)[accepted : accepted+K−1]` の切り出しで作り直す。
  - 注意（同梱 `kernels.py` で筆者が確認）: replay kernel は step ごとに状態を
    入力 dtype に丸める。verify kernel は T step を f32 のまま保持して最後に丸める。両者の丸めは
    一致しない。
- **vLLM の conv state**: conv buffer を `conv_kernel − 1 + num_spec` に延ばし、`num_accepted_tokens` で
  読み出し位置をずらす（`causal_conv1d.py` のコメント）。conv については snapshot もコピーも要らない。
- **SGLang ReplaySSM-spec**（`gdn_replayssm_spec_decode.py`）: 直近 L 個の確定 step の `(d, k, g)` を
  要求ごとの循環 buffer に、凍結 checkpoint `h0` と並べて置く。verify の出力は UT 変換で output-only に
  再構成し、reject は「pointer move」で済ませる。bf16 記録は high/low の 2 つに分けて誤差を補償する。
- **SpecLA** [2607.16673]: 状態の更新を次の verify まで遅らせ、verify kernel に融合する。token replay に
  比べて commit 遅延は 2.74〜4.28× 小さい（H100、GDN-1.3B）。
- **fidelity**:
  - vLLM #58863 のテストは、f32 状態なら commit 結果が逐次 per-token 状態と「bit-identical」、
    bf16 状態なら 1 ulp 以内、と主張する。
  - #61309 の greedy 比較では 30 中 27 の出力がどこかで分岐した。分岐点はすべて、bf16 の top-1/top-2 が
    ちょうど同値か 1 ulp 差の位置だった。
  - 逐次 greedy と完全に一致すると主張する tape 実装は見つからなかった。
  - qwisp の fast mode では問題にならない。strict に持ち込む場合は、FEAS §5.4 のとおり M=1 と同じ
    kernel・同じ丸め順で replay することが前提になる。
- **drafter 側の状態**: z-lab/dflash と mlx-vlm は context の K/V だけを cache し、block 自身の K/V は
  その step で捨てる（「only context KV is cached」）。受理行の hidden だけを次の draft に渡すので、
  drafter には rollback が要らない（§5.1）。

## 5. block-diffusion drafter（DFlash / DFlash2）の実装

FEAS §4 は oMLX、mlx-vlm、dflash-mlx、Splash の概要と Apple 上の数値をまとめている。ここではそれ以外と、
qwisp の設計に効く細部だけを足す。

### 5.1 drafter の context の扱い（どの実装も同じ形）

- **z-lab/dflash**（@ `07ebd93`, 2026-08-18, `dflash/model_mlx.py`）:
  - `_LayerHook` が `target_layer_ids` の各層の出力を捕まえる。それを連結し
    `h_ctx = hidden_norm(fc(target_hidden))` を作る。fc は `8×2048 → 2048`。
  - 各 draft 層は `h_ctx` から context の K/V を作って cache に積む。block の K/V（`[last_token] +
    [mask]×(bs−1)`）はその step で連結するだけで cache しない。
  - 全層が sliding のときは、prefill でも最後の `sliding_window − 1` 行しか保持しない。
  - verify 後は `hidden[:, :accepted+1]` だけを次に渡す。
- **incoai/splash**（@ `be3b083`, 2026-10-14）:
  - 「every draft layer's keys and values of the last SPLASH_DRAFT_SLIDING_WINDOW positions」を
    ring として持つ。K は `[head][ring_position][dim]`、V は `[head][dim][ring_position]` の配置。
  - 幾何は固定である（`ExecutionGeometry.h`）: draft query 8 行、提案 7、target verify 8 行、
    window 2048、候補 16、selector rank 256。適応 block、tree verify、n-gram draft は持たない
    （委託した調査で `runtime/` と `server/` を grep した結果。筆者は未確認）。
  - draft attention の split 数は 4 で固定。「the combine order, and with it the rounding, is the same
    on every machine」のため。
- **qwisp に効く点**（推測）: incoai DFlash2 は 6 層すべてが sliding window 2048、`is_causal: false`。
  context は 6 層 × 2048 位置 × K/V（8 KV head × 128）× bf16 ≈ 50 MB の固定 ring に収まり、persistent
  Metal buffer に向いている。target の tap は 8 層 × 2048 × bf16 = 32 KiB/token と小さい。chain-ahead を
  含むすべての forward で tap を書いておけば、drafter が off の間に溜まった分も、再開時に fc を一括で
  かけるだけで追いつける。

### 5.2 verify の形と block の適応

| 実装 | verify | block 長 | drafter 量子化 |
|---|---|---|---|
| z-lab/dflash MLX | chain、greedy は最初の不一致まで | 固定。「quantized target/draft では `block_size <= 5`」 | `--draft-bits 4` の例あり |
| dflash-mlx upstream `6080323` | `dflash` / `adaptive`（§1.3）/ `ddtree`（best-first heap、`_DDTREE_TOP_WIDTH=2`、Ringel & Romano [2604.12989]）/ `off` | DFlash2 は `default_block_tokens=5`、`max_block_tokens=min(5, block_size)`、`supports_ddtree=False`、`supports_copyspec=False` | 既定 w4 |
| dflash-mlx 同梱 0.1.10+omlx.9（FEAS §8 の実測） | DFlash2 は ddtree 不可（`spec_epoch.py:1982`） | **5 行上限なし**。block 6/8 は設定どおり verify されていた（筆者確認） | oMLX の `dflash_draft_quant_*` |
| mlx-vlm `4623cf9` | chain | 3〜5 行を受理で増減（§1.3） | 4-bit 推奨 |
| oMLX main `958974c1` | Lightning MTP verify | `_DepthController`（§1.3）、上限 9 | load 時 affine |
| splash | chain（候補経路 1 本） | 8 行固定 | af4g64 |
| vLLM | DFlash2 は PR #52816（2026-08-21 merge）。path walk は「one Triton program per request」 | 固定。adaptive K の #52559 は open | — |
| SGLang | PR #35371（2026-08-19 merge）。selector decode を draft CUDA graph に入れる | 固定（`--speculative-num-draft-tokens`） | quantized target lm_head のテストあり |
| llama.cpp | DFlash v1 #22105、DFlash2 #27342（2026-08-27 merge） | `n_max` を trained block で頭打ち、`p_min` で打ち切り | Q4_K_M で受理不変（§3.1） |
| TensorRT-LLM | DFlash2 #18155（2026-09-14 merge） | — | — |

DFlash2 の selector と conv のコスト: vLLM #52816 では conv + selector が 20.70 ms の step のうち
0.174 ms（0.84%）。DFlash2 blog（inco.ai, 2026-08-18、要約経由）では +1.3% のサイクル遅延。
recall@16 は 87.8〜99.5%、recall@1 は 72.9〜85.4%。oracle が top-16 から最良経路を選べたとすると
τ 6.79（実際は 4.27）。

### 5.3 block 長についての一次資料

- z-lab/Qwen3.6-35B-A3B-DFlash（v1）カード（B200、SGLang、concurrency 1）の受理長:

  | Task | Block 4 | Block 8 | Block 16 |
  |---|---:|---:|---:|
  | GSM8K | 3.455 | 5.404 | 6.954 |
  | MATH-500 | 3.553 | 5.743 | 7.629 |
  | HumanEval | 3.459 | 5.352 | 6.797 |
  | MBPP | 3.358 | 4.973 | 6.052 |
  | MT-Bench | 3.075 | 4.326 | 5.118 |

  カードは「Block size 8 is the recommended default for higher-concurrency serving」と書く。
  MT-Bench、concurrency 1 では block 8（2.32×）が block 16（2.24×）を上回る。
- DFlash 論文: block 8 学習モデルが block 全体を受理する割合は 35.7%。適応的な block 長は future work と
  している（要約経由）。
- DBloom [arXiv 2608.30427, 2026-08-31]: block が丸ごと受理されることが多いと指摘する（"stranded
  speed-up"）。推論時に block を広げると、前の方の位置でも draft の分布がずれる。block 16 → 24 への
  追加学習で τ は中央値 +0.8（要約経由）。
- **qwisp の regime との差**: FEAS §8 の M3 Pro 実測は tokens/cycle 2.2〜4.0 で、カードの 4〜7 を大きく
  下回る。原因の候補は 3 つある。target が 4-bit であること（FEAS §5.3 の「feature shift」）、
  greedy で thinking 無しであること、384 token と短いこと。どれが効いているかは分かっていない。

### 5.4 Apple Silicon の数値（FEAS §4.3 に無いもの）

- llama.cpp #27342: M5 Pro 64 GB、Qwen3.8-27B（dense）Q4_K_M、DFlash2 block 8、T=1、GSM8K 先頭 8 問。
  10.42 → 19.31 tok/s（1.85×、受理 4.92）。
- M3 Pro、または M3 世代で 35B-A3B に DFlash/DFlash2 を使った数値は、FEAS §8 の qwisp 自身の測定以外に
  見つからなかった。

## 6. ローカル / エッジ環境での MoE ターゲットの実測利得

OPT §4.1 の mlx-lm #990（M4 Pro、Qwen3.5-35B-A3B の MTP で 1.04×）に加えて見つかった測定を挙げる。
GitHub の issue / PR やブログの測定は、測った本人の記録として一次扱いにした。

| 出典 | ハードウェア | target / drafter | 結果 |
|---|---|---|---|
| [stared/benching-local-llms-on-apple-silicon](https://github.com/stared/benching-local-llms-on-apple-silicon) | M5 Max 128 GB | Qwen3.6-35B-A3B Q8_0、llama.cpp MTP `--spec-draft-n-max 3` | decode@128: 93 → 105 tok/s（+12%）。同じページで dense 27B は 17 → 32（+75%） |
| [ikyle.me 2026 blog](https://ikyle.me/blog/2026/how-to-setup-a-local-coding-agent-on-macos) | M1 Max 64 GB | Gemma 4 26B-A4B（MoE）+ Q8 MTP draft | 58.2 → 72.2 tok/s（1.24×）。n_max 1..6 で 68.4 / 72.0 / 72.2 / 70.7 / 63.7 / 61.2（1 プロンプト） |
| [hanxiao/Qwen3.6-35B-A3B-MTP-L4](https://github.com/hanxiao/Qwen3.6-35B-A3B-MTP-L4) | NVIDIA L4（~300 GB/s） | Qwen3.6-35B-A3B UD-Q4_K_XL、llama.cpp MTP | baseline 65.5。n_max 1/2/3/4/6 で 75/81/79/77/70 tok/s、MTP 受理 0.85/0.74/0.63/0.57/0.44。最良は n_max 2 で約 1.24× |
| [jarvislabs blog](https://jarvislabs.ai/blog/qwen36-mtp-llamacpp-rtxpro6000) | RTX PRO 6000 Blackwell | Qwen3.6-35B-A3B Q8_0、MTP n_max 3 | 193.36 → 225.48 tok/s（1.17×、受理 55%）。dense 27B は 1.73× |
| [unsloth HF discussion #14](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF/discussions/14) | RTX 3090 | Qwen3.6-35B-A3B UD-Q4_K_XL | baseline 約 136〜140 tok/s に対し、llama.cpp の投機はどのモードも負けた。DFlash（#22105）は max 4/8/16 で 74.9/77.0/65.8 |
| [oMLX v0.5.0 release notes](https://newreleases.io/project/github/jundot/omlx/release/v0.5.0) | M3 Ultra | Qwen3.6-35B-A3B、Lightning MTP（適応深さ + verify 形状 kernel） | 89.6 → 140.4 tok/s（1.57×）、greedy 400 token。量子化と受理は記載なし（vendor 報告） |
| [z-lab/Qwen3.6-35B-A3B-DFlash](https://huggingface.co/z-lab/Qwen3.6-35B-A3B-DFlash) | B200 | SGLang、block 8/16 | concurrency 1 で 2.24〜3.61×（データセンター、計算に余裕のある regime） |
| FEAS §8（qwisp） | **M3 Pro** | oMLX 0.7.0 + incoai DFlash2 4-bit | 最良は code +15%（block 4）、agentic +25%（block 6）、longctx +13%（block 4）。shortnl は全 block で −10〜13% |

読み取れること（推測）:

1. 帯域律速の機械では、MoE target の MTP/DFlash 利得は 1.1〜1.25× に集まる。同じ機械で dense 27B は
   1.7× 前後出る。最適な draft 深さは 2〜3 である（L4、M1 Max の sweep）。
2. 1.5× を超える Apple 上の MoE の報告は oMLX Lightning MTP（M3 Ultra、1.57×）だけである。本人は
   適応深さと verify 形状 kernel の効果だとしている。M3 Ultra は M=1 が帯域を使い切っていない
   （FEAS §7 の u₁ が低い）側なので、M3 Pro にそのまま移るとは限らない。
3. FEAS §8 の M3 Pro 実測（+13〜25%、block 4〜6）は 1. の帯と合っている。qwisp が目指せる幅の目安になる。
4. Qwen3-30B-A3B + 小型 Qwen3 draft の Mac での数値は見つからなかった。

## 7. suffix（n-gram）draft とモデル draft の併用

### 7.1 SuffixDecoding 論文とその実装

- **論文** [arXiv 2411.04975v3、要約経由]: 「always speculate using SuffixDecoding first」。suffix tree の
  SCORE（経路確率の和で、期待受理数の推定）が τ を超えたら suffix draft を使い、超えなければ EAGLE-3
  に落とす。τ は「fallback speculator の平均受理数の近く」に置く。agentic では τ=0、Spec-Bench では
  τ=5〜7（EAGLE-3 の平均約 4.65）が最良だった。投機長は `MAX_SPEC = α·p`（p は一致長、α∈[1,4]）。
  H100 で AgenticSQL 4.1×、Spec-Bench 2.5×（EAGLE-3 単独は 2.4×）。
- **Arctic Inference**（snowflakedb @ `aca5d9a`, `arctic_inference/vllm/model_runner.py:1059-1072`）:
  - `min_score = 0 if method == "suffix" else self.drafter.model.n_predict`。suffix の score が
    `min_score` 以上なら suffix を採り、そうでなければ model draft を使う（`s_tokens if s_tokens else a_tokens`）。
  - model drafter は毎 step 走らせ、suffix 探索は CPU 側で重ねる。
  - score は `prob *= count/node->count; score += prob`（`suffix_tree.cc`）で、確率 0.1 未満の token は
    捨てる。
- **vLLM**: `method: "suffix"` は単独手法（#25784）。ngram/suffix と EAGLE を混ぜる #24344 は open。
  RFC #18633 の規則は「n-gram 一致があればその系列の EAGLE を省き、無ければ EAGLE」。
- **SGLang**: 混成は未マージ（#37237 は open）。batch 単位で、retrieval が使える要求の割合が閾値以上の
  ときだけ retrieval を選ぶ。

### 7.2 qwisp の現行 SCORE gate との関係

qwisp の `Tell.suffixDraft` + SCORE gate（`Tell.swift`: `score × 1000 > drafted × ROWCOST‰` なら verify、
既定 478‰）は論文と同じ系統の規則である。違うのは閾値の基準で、qwisp は「draft 行数 × 1 行コスト」を
使い、論文と Arctic は「fallback の期待受理数」を使う。モデル drafter を足すと、比べる相手は
「chain」ではなく「DFlash の期待 tokens/cycle をそのコストで割ったもの」になる。

### 7.3 dflash-mlx CopySpec（DFlash 内蔵の prompt lookup）

- `engine/copyspec.py`: prompt と確定 token の 6-gram（FNV ハッシュ）を索引にする。直近 5 token と
  staged token で引き、block_len−1 個を丸ごと取れる位置があればその cycle の draft を置き換える。
  そのときも drafter の context は `advance_context` で更新する。
- `conservative`（既定）は「disables copyspec after one missed copy block」。
- `auto` は `CopySpecAutoGate` の A/B latch である。24 cycle を off で測り、on で copy block 6 個
  （最大 48 cycle）測る。on の tok/s ≥ off なら 200 cycle engage、負ければ 200·2^min(連敗,4) cycle
  休眠する。help には「a clear win on strong-copyspec targets like large MoE」とある。
- upstream `6080323` では DFlash2 drafter で `supports_copyspec=False`。同梱 0.1.10+omlx.9 ではその
  制限が無く、FEAS §8 の実測中も conservative の copyspec が働きえた（推測。発火回数はログに無い）。

### 7.4 共通する形

切替の規則はどれも同じ形に収束している。n-gram/suffix の期待受理 score が、モデル drafter の平均受理
（または draft 長）以上なら suffix を使い、それ以外はモデル drafter を使う。モデル drafter の context は
suffix を使った step でも更新し続ける。suffix tree と DFlash2 を組み合わせた実装は見つからなかった。

## qwisp への提案（優先度つき）

前提: 対象は `--fast` の非 bit-exact tier に限る。strict path と WRITE-LOCKED tests は変えない。
forward path は凍結されているので、追加は flag-off で byte-identical になる seam に限る（FEAS §6 と同じ）。
以下の数値で「推定」とあるものは、M3 Pro の実測 r_M（線形補間で r_3 1.54、r_5 2.27、r_6 2.68、r_8 3.51）、
drafter コスト c = 0.27F（FEAS §5.2 の 4-bit 推定）、固定オーバーヘッド o = 1 ms ≈ 0.06F から計算した。

**基準の計算（推定）。** d 個の draft を verify する（M = d+1 行）1 サイクルの損益分岐 tokens/cycle は
`T*(d) = c + r_(d+1) + o`（F 単位）になる。T*(1)=1.56、T*(2)=1.87、T*(3)=2.19、T*(5)=3.01、T*(7)=3.84。
FEAS §8 の tokens/cycle（同じ行数で測ったもの）を当てはめると、4 行での予測は code 1.34×、
agentic 1.46×、longctx 1.31×、shortnl 1.00×。8 行では code 0.68×、agentic 1.04×。oMLX の実測
（4 行で +13〜18%）はこの予測より小さい。oMLX のサイクル固定費が qwisp の o より大きいためと読めるが、
oMLX の o は測っていないので推測にとどまる。

### P0（実装の前提、または最大の損失を消すもの）

**P0-1. 計測だけで決まる未知量を先に測る**（engine 変更なし。GPU は単独で使う）。

- (a) incoai DFlash2 の位置別条件付き受理 p_j を、qwisp 自身の greedy 出力に対して 4 regime で測る。
  FEAS §6 Phase 1 の Python oracle を使い、drafter は 4-bit g64 にする。
- (b) r_M を M = 2..9 のすべてで、浅い context と 16K の両方で測る（今は 2, 4, 9, 17 しか無い）。
- (c) verify 行の実ルートから層あたり unique expert U(M) を数え、一様モデル（FEAS §5.1）と比べる。
  §2.2 の緊張関係をここで決着させる。
- (d) 4-bit drafter 1 block の時間を測る。

理由: §1 のどの制御器も、入力は p_j と t(d) の 2 つだけである。どちらも qwisp の regime と kernel で
測らない限り決まらない。p_j の幾何近似を FEAS §8 に当てはめると code α≈0.8、shortnl α≈0.6 になる
（推定）。最適行数は前者で 4〜5、後者では d≥1 に利得がほぼ無い。
測り方: (a) は accept-length ヒストグラムと p_j 曲線、(b) は既存の seqmt-m probe、(c) は routing telemetry
（notes/11 系）の流用。

**P0-2. verify 深さの制御規則: 「block 8 で draft し、verify する prefix をコストモデルの argmax で選ぶ」。**

- 毎サイクル、trained block の 8 位置（7 draft）を 1 回の drafter forward で作る。drafter のバイトは
  行数 ≤ 16 では変わらない（FEAS §5.2）。
- verify する draft 数 d ∈ {0..7} は `score(d) = (1 + Σ_{j≤d} Π_{i≤j} p_i) / t(d)` を最大にする値にする。
  これは oMLX `_DepthController`、vLLM adaptive verification、EVICT と同じ形である。
  - p_i は位置別受理の EMA（α≈0.08、oMLX と同じ）で、初期値は P0-1(a) の regime 平均にする。
  - t(d) の初期値は DeviceCalibration の `c + r_(d+1)·F + o`、d=0 は chain の実測 ms/token。その後は
    d ごとの実時間 EMA（時間窓 約 400 ms、2 倍超の外れ値は減衰）で置き換える。
- 切り替えは 3% 超の改善があるときだけにする。対抗深さの probe はサイクルの 10〜15% 以内に収める。
- d=0 は chain（`QWISP_CHAIN_AHEAD` を含む）に渡し、drafter forward も省く。ただし tap の記録
  （§5.1、32 KiB/token）は続ける。
- 再開は Cascade の back-off を使う。d=0 が続いた期間を倍々に延ばしてから、最良の d>0 を 4 サイクル
  試す。

dflash-mlx の固定閾値（0.75 / 3.5）を採らない理由: その閾値は r_M を入力に持たず、M3 Pro の実測では
block 8 設定が block 4 固定に負けた（§1.3）。
期待効果（推定）: code・longctx 1.2〜1.3×、agentic 1.3〜1.45×、shortnl 約 1.0×（−3% 以内）。
oMLX の実測 +13〜25% を下限の目安とする。
測り方: 4 regime × 2 run の tok/s を chain と比べ、さらに固定 d=1..7 の sweep の最良値（後知恵の最適）と
比べる。選ばれた d の分布と probe 比率をログに出す。

**P0-3. drafter を切る条件（shortnl 対策）は P0-2 の d=0 に含め、別の gate を作らない。**

- shortnl の 4 行 tokens/cycle は 2.19 で、T*(3) = 2.19 とちょうど等しい（推定）。利得は無く、損失だけが
  probe と drafter の固定費から出る。
- 要求単位では次のように扱う。
  - 最初の cycle は P0-1(a) の regime 事前分布で score を評価する。
  - `best score(d>0) < score(0) × 1.15` が 16 cycle 続いたら park する（oMLX の EXIT_MARGIN /
    EXIT_STREAK と同じ形。1.15 は実測で置き換える）。
  - 短い応答（数十 token）は park の判定が終わる前に終わる。
- 損失の上限は、そうした要求の最初の 16 cycle が d≈3（ほぼ損益分岐）で走る分に限られる（推定 −0〜3%）。
- 測り方: shortnl の tok/s を chain 比で測り、目標は ≥ −3%。park までの cycle 数をログに出す。

**P0-4. drafter の形式: incoai/Qwen3.6-35B-A3B-DFlash2 を 4-bit affine g64 で事前量子化して配る。**

- lm_head と embed は target の 4-bit のものを共有する（dequant 済み F16 の copy は作らない）。
- 根拠:
  - M3 Pro 実測で 4-bit g64 が最速で、受理は 8-bit と同等だった（FEAS §8）。
  - 4-bit で受理が下がる幅は 0〜2.4%（§3.1）。
  - block 8 学習の drafter を少ない行数で verify しても学習条件と矛盾しない（§3.3）。
  - Ornith 専用の jzinno（block 16）は M3 Pro で incoai より遅かった（FEAS §8）。
- BF16 drafter は配布しない。block 16 の drafter は対象外にする。
- 期待効果（推定）: drafter 0.74F → 0.27F/block（FEAS §5.2）。
- 測り方: P0-1(d) と P0-1(a) の 4-bit / BF16 比較。

**P0-5. GDN の rollback は tape replay（§4 の方式 C）にし、位置ごとの snapshot（方式 A）は採らない。**

- verify 中に、行・層ごとに delta = (v − kv_mem)·β（f32）、k、g と conv 前の qkv を tape に書く
  （block 8 で約 10 MB）。
- partial reject のときは次の順で戻す。
  1. ping-pong snapshot に戻す。
  2. 受理 p+1 行だけを rank-1 更新で replay する。
  3. conv は `concat(snapshot_conv, tape_qkv)` を切り出して作る。
  4. attention KV は長さを巻き戻す。
- replay の dispatch は次サイクルの command buffer の先頭に入れる（SpecLA の「次の verify に融合」と
  同じ発想）。こうすると host の往復が増えない。
- 方式 A は block 8 で毎 verify 約 480 MiB を書き、約 0.27F かかる（§4.2）。帯域律速の M3 Pro では
  割に合わない。vLLM の c=1 で C が A に勝たないのは、データセンター GPU では状態の書き込みが step
  時間に比べて小さいからと読める（推測）。M3 Pro ではこの比率が逆になる。
- 期待効果（推定）: rebuild forward（block 8 で 1.5〜2.0F × reject 率）が約 0.07F になる。FEAS §5.3 の
  S1 → S2 で約 +0.3×。
- 測り方: replay 後の状態と rebuild forward 後の状態の差（fast mode なので許容誤差つき、f32 で
  max-abs）を出す。加えて、reject 率の高い shortnl での tok/s。
- fast mode でも、replay は verify と同じ丸め（f32 保持）にそろえる。dflash-mlx のように step ごとに
  丸めると、verify と commit の状態がずれる（§4.3）。

### P1（効果はあるが P0 の後で測って決める）

**P1-1. 既存 suffix draft との併用: 「期待 tokens/time が大きい方」を毎 step 選ぶ。**

- 規則（推測にもとづく設計。SuffixDecoding / Arctic の「suffix score ≥ モデル drafter の期待受理なら
  suffix」の形を、qwisp のコスト表に置き換えたもの）:
  - suffix の draft が非空なら `(1 + SCORE)/t(D_suffix)` と P0-2 の `max_d score(d)` を比べ、大きい方を
    使う。t(·) は同じ r_M 表から取る。
  - どちらも chain に負けるなら chain にする。
  - suffix を使った step でも drafter の tap と context は更新する。
- 既存の SCORE gate の定数 478‰ は、この比較の中で r_M 表に置き換わる（OPT §4.1 の提案と同じ方向）。
- 1-token MTP head は、drafter が無いときの fallback としてだけ残す。DFlash の 2 行 verify は MTP-1 と
  同じ行数で、受理はそれ以上と見込める（推測）。
- 期待効果（推定）: agentic / longctx の suffix 命中 span では長い suffix draft を使い、
  draftless span は DFlash で埋める。agentic の上積みは +5〜15%（幅は大きい）。
- 測り方: draft 源（suffix / DFlash / chain）ごとの受理と時間の内訳をログに出し、4 regime で
  P0-2 単独と比べる。

**P1-2. 確信度による打ち切りを P0-2 に足す。**

- DFlash2 は各位置の候補スコアを selector の副産物として出す。llama.cpp の DFlash2 `p_min` や
  SpecDec++ の閾値形と同じように、block ごとの確信度を p_i の補正に使う。
- 先に qwisp の regime で較正（ECE）を測る。EVICT は draft 確率の ECE を 0.01〜0.02 と報告しているが、
  4-bit target・4-bit drafter で同じとは限らない。
- 期待効果（推定）: 受理の分散が大きい code / longctx で数 %。
- 測り方: P0-2 と A/B する。

**P1-3. DeviceCalibration と推奨 / 警告の UX。**

FEAS §7 の `specEconomics` を、今回の実測に合わせて次のように具体化する。

- 起動時に、実ルートの prompt で r_M（M = 1..8）と drafter 1 block を測る（1 秒未満）。結果は
  (GPU 名とコア数, model hash, drafter hash, qwisp version) をキーに cache する。
- 判定量は予測 code 利得 `Ĝ = T_code(4 行) / T*(3)` とする。T_code(4 行) は drafter と model の組で
  決まる定数で、当面は M3 Pro の測定値 2.93 tokens/cycle（FEAS §8）を使い、P0-1(a) の測定で置き換える。
  - Ĝ ≥ 1.2 なら「推奨」（drafter があれば有効化し、無ければ `hf download` の 1 行案内と予測範囲を出す）。
  - 1.05 ≤ Ĝ < 1.2 なら「任意」（有効化は可、既定は off）。
  - Ĝ < 1.05 なら「非推奨」（`--fast` でも既定 off、明示指定には警告）。
  - M3 Pro の推定は Ĝ ≈ 1.34 で「推奨」に入る。
- streaming / partial tier（resident でない）では常に off にする。verify の union が SSD 読み出しに
  なるためである。
- 「帯域律速なら推奨、計算律速なら任意」という二分は使わない。判別する量は FEAS §7 の κ = r_M/r_bytes
  （計算の残差）、u₁（M=1 の帯域利用率）、o/F（固定費）である。
  - κ ≫ 1 の機械（計算が弱い、少行 kernel が非効率）では DFlash は負ける。RTX 3090 の llama.cpp で
    全モード負けたのはこの側の例と読める（推測）。
  - u₁ が低い Max/Ultra では余剰行が固定費に隠れて、利得が大きくなる（oMLX M3 Ultra 1.57×）。
- 表示例: `drafter: recommended (r4=1.86, T*=2.19, predicted code ×1.2–1.35 / chat ×1.0)`
- 測り方: M1 Max と M3 Pro の 2 台で、予測 Ĝ と実測 G-D の差を ±10% 以内にする。

### P2（今は採らない。条件が変われば再検討）

- **drafter lm_head の語彙刈り込み（32k）**: 推定 −0.14F/block だが、受理が最大 7% 程度下がりうる
  （FR-Spec の 32k は 93.3%）。DFlash2 selector の候補も変わる。P0-1(a) の受理曲線を見てから判断する。
- **tree / DDTree verify**: 4-bit target では tree の verify コストが利得を食う（HierSpec）。dflash-mlx
  も DFlash2 では ddtree を無効にしている。MoE の union も行数とともに増える。
- **verify 時の expert 上限（MoE-Spec 型）**: ルーティングを変える近似なので、fast mode でも品質評価が
  先に要る。
- **block > 8 の drafter / DBloom 型の追加学習**: M3 Pro では r_9 = 3.92 なので、受理が伸びても 8 行を
  超える verify は割に合わない（推定）。
- **少行 MMA verify kernel**（OPT §4.2）: M ≤ 4 では実測 r_M がバイト比とほぼ一致するので、kernel の
  改良では下げられない（推定）。M ≥ 5 の計算残差を削る用途に限られる。

## 検証できなかった主張

一次資料で確認できなかったもの、または要約ツール経由でしか読んでいないもの:

- DFlash 論文 Table 8、DFlash2 blog、SuffixDecoding 論文、Arctic blog、DBloom の引用文は、WebFetch の要約を
  経由している（数値は要約が示したもの）。
- 「oMLX 0.7.0 の実測が dflash-mlx 経路だった」はコードパスとログ文字列からの推測で、実行時のログでは
  確かめていない。copyspec が実測中に発火したかどうかも分からない。
- oMLX v0.5.0 の 1.57×（M3 Ultra）の量子化・受理・深さ。Splash の、投機だけによる利得・受理・block。
- Qwen3-30B-A3B + 小型 draft の Mac での数値。EAGLE-3 の Qwen3-30B-A3B / 235B での数値（SpecBundle は
  図だけ）。
- llama.cpp issue #23752 が引く「35B-A3B MTP で 5–14× の退行」（gist にアクセスできなかった）。
- arXiv 2609.22471（"Expert Coactivation"）。存在を確認できなかった。
- Snakes and Ladders の手法の詳細（PMLR の抄録と二次資料のみ）、TreeWY / SpecLA のコード。
- vLLM / SGLang の DFlash worker が draft KV をどう配置・破棄しているか（PR 本文のみで、source は
  読んでいない）。llama.cpp #22105 の本文。
- llama.cpp で `n_rs_seq` を最初に入れたのがどの PR か。
- SP-MoE、MoE-SpAc、SpecMoE、MoE-SpeQ、QSpec、ML-SpecQD のハードウェアと詳細（抄録のみ）。
- 調査中に「SpecServe」「DSD」という名前の独立した論文は見つからなかった。AdaSpec [2503.05096] が
  SpecServe に当たる可能性がある（推測）。

## Sources

Local source（2026-10-16 に読んだもの）:

- oMLX.app 0.7.0（`/Applications/oMLX.app/Contents/Resources/omlx`）: `engine/dflash.py`,
  `engine_pool.py`, `utils/model_loading.py`, `model_settings.py`,
  `patches/mlx_lm_mtp/batch_generator.py`。同梱 `dflash_mlx 0.1.10+omlx.9`（`framework-mlx-base/.../site-packages/dflash_mlx`）:
  `engine/spec_epoch.py`（`_AdaptiveBlockPolicy`）, `engine/copyspec.py`, `engine/config.py`,
  `runtime/config.py`, `runtime/chip_detect.py`, `recurrent_rollback_cache.py`, `kernels.py`, `engine/target_qwen_gdn.py`。
- oMLX repo `/Users/nara/repos/omlx` @ `958974c1`（2026-10-15）: `docs/experimental/dflash_mlx_integration.md`,
  `omlx/patches/mlx_lm_mtp/batch_generator.py`（`_DepthController`）, commit `7a4ea399`（2026-10-14, #4508）。
- Drafter checkpoints: `~/models/Qwen3.6-35B-A3B-DFlash2-incoai`（config.json, README）,
  `~/models/Ornith-1.5-35B-A3B-DFlash2-jzinno`（config.json, README, evaluation.json）,
  `~/models/Ornith-1.5-35B-A3B-DFlash`, `~/models/Ornith-1.5-35B-A3B-MLX-4bit/config.json`。
- qwisp: `swift/Sources/QwispCore/Tell.swift`（SCORE gate）, `TellRuntime.swift`, `notes/15-mtp-hybrid-draft-spec.md`。
- Shallow clones（scratchpad, 2026-10-15〜16）: vLLM `e365792` / `074c83a`
  （`vllm/v1/worker/gpu/spec_decode/adaptive_verification.py`, `acceptance_estimator.py`, `v1/attention/backends/gdn_attn.py`,
  `model_executor/layers/mamba/ops/causal_conv1d.py`）, SGLang `0693457`（`speculative/adaptive_spec_params.py`,
  `mem_cache/memory_pool.py`, `speculative/dflash_worker_v2.py`, `kernels/ops/attention/fla/gdn_replayssm_spec_decode.py`）,
  llama.cpp `f12aaa3`（`common/speculative.cpp`, `common/common.h`, `src/llama-memory-recurrent.cpp`, `src/models/delta-net-base.cpp`）,
  TensorRT-LLM `0c1882b`（`tensorrt_llm/_torch/speculative/drafter.py`, `llmapi/llm_args.py`）, mlx-lm `93e6d4a`,
  z-lab/dflash `07ebd93`（`dflash/model_mlx.py`）, incoai/splash `be3b083`（`runtime/metal/abi/ExecutionGeometry.h`,
  `DFlashDraft.hpp`, `DraftSelector.hpp`, `docs/performance.md`, `DEVELOPMENT.md`）, bstnxbt/dflash-mlx `6080323`,
  Blaizzy/mlx-vlm `4623cf9`（`speculative/dflash.py`, `drafters/dflash2/`）, snowflakedb/ArcticInference `aca5d9a`
  （`arctic_inference/vllm/model_runner.py`, `csrc/suffix_decoding/suffix_tree.cc`）, HF transformers `683b721`
  （`generation/candidate_generator.py`）。

PRs / issues:

- vLLM [#47808](https://github.com/vllm-project/vllm/pull/47808), [#32374](https://github.com/vllm-project/vllm/pull/32374),
  [#52559](https://github.com/vllm-project/vllm/pull/52559), [#52816](https://github.com/vllm-project/vllm/pull/52816),
  [#58863](https://github.com/vllm-project/vllm/pull/58863), [#61309](https://github.com/vllm-project/vllm/pull/61309),
  [#25784](https://github.com/vllm-project/vllm/pull/25784), [#24344](https://github.com/vllm-project/vllm/pull/24344)。
- SGLang [#21599](https://github.com/sgl-project/sglang/pull/21599), [#35371](https://github.com/sgl-project/sglang/pull/35371),
  [#37237](https://github.com/sgl-project/sglang/pull/37237), RFC [#28511](https://github.com/sgl-project/sglang/issues/28511)。
- llama.cpp [#22105](https://github.com/ggml-org/llama.cpp/pull/22105), [#22400](https://github.com/ggml-org/llama.cpp/pull/22400),
  [#22673](https://github.com/ggml-org/llama.cpp/pull/22673), [#19493](https://github.com/ggml-org/llama.cpp/pull/19493),
  [#27342](https://github.com/ggml-org/llama.cpp/pull/27342), issues [#23011](https://github.com/ggml-org/llama.cpp/issues/23011),
  [#23752](https://github.com/ggml-org/llama.cpp/issues/23752)。
- TensorRT-LLM [#8194](https://github.com/NVIDIA/TensorRT-LLM/pull/8194), [#18155](https://github.com/NVIDIA/TensorRT-LLM/pull/18155)。
- mlx-lm [#990](https://github.com/ml-explore/mlx-lm/pull/990), [#1821](https://github.com/ml-explore/mlx-lm/pull/1821),
  discussion [#890](https://github.com/ml-explore/mlx-lm/discussions/890)。

Papers（arXiv）:

- 深さ制御: SpecDec++ [2405.19715](https://arxiv.org/abs/2405.19715), EAGLE-2 [2406.16858](https://arxiv.org/abs/2406.16858),
  BanditSpec [2505.15141](https://arxiv.org/abs/2505.15141), TETRIS [2502.15197](https://arxiv.org/abs/2502.15197),
  AdaEDL [2410.18351](https://arxiv.org/abs/2410.18351), SmartSpec/TurboSpec [2406.14066](https://arxiv.org/abs/2406.14066),
  DISCO [2405.04304](https://arxiv.org/abs/2405.04304), SVIP [2411.18462](https://arxiv.org/abs/2411.18462),
  AdaEAGLE [2412.18910](https://arxiv.org/abs/2412.18910), PEARL [2408.11850](https://arxiv.org/abs/2408.11850),
  HeteroSpec [2505.13254](https://arxiv.org/abs/2505.13254), Nightjar [2512.22420](https://arxiv.org/abs/2512.22420),
  AdaSpec [2503.05096](https://arxiv.org/abs/2503.05096), Cascade [2506.20675](https://arxiv.org/abs/2506.20675),
  EVICT [2605.00342](https://arxiv.org/abs/2605.00342)。
- MoE: MoESD [2505.19645](https://arxiv.org/abs/2505.19645), EAGLE [2401.15077](https://arxiv.org/abs/2401.15077),
  The Limits of Speculation [2609.22156](https://arxiv.org/abs/2609.22156), XShare [2602.07265](https://arxiv.org/abs/2602.07265),
  MoE-Spec [2602.16052](https://arxiv.org/abs/2602.16052)。
- drafter 軽量化: Quantize the Target, Quantize the Drafter [2607.04244](https://arxiv.org/abs/2607.04244),
  FR-Spec [2502.14856](https://arxiv.org/abs/2502.14856), VocabTrim [2506.22694](https://arxiv.org/abs/2506.22694),
  [2603.05210](https://arxiv.org/abs/2603.05210), HierSpec [2505.22179](https://arxiv.org/abs/2505.22179),
  LayerSkip [2404.16710](https://arxiv.org/abs/2404.16710), Kangaroo [2404.18911](https://arxiv.org/abs/2404.18911)。
- 再帰状態: STree [2505.14969](https://arxiv.org/abs/2505.14969), Mamba in the Llama [2408.15237](https://arxiv.org/abs/2408.15237),
  TreeWY [2608.20961](https://arxiv.org/abs/2608.20961), SpecLA [2607.16673](https://arxiv.org/abs/2607.16673),
  ReplaySSM [2610.13778](https://arxiv.org/abs/2610.13778), Snakes and Ladders（PMLR 262:292-304）。
- DFlash 系: DFlash [2602.06036](https://arxiv.org/abs/2602.06036), DBloom [2608.30427](https://arxiv.org/abs/2608.30427),
  DDTree [2604.12989](https://arxiv.org/abs/2604.12989)。
- suffix: SuffixDecoding [2411.04975](https://arxiv.org/abs/2411.04975)。

Model cards / blogs:

- [incoai/Qwen3.6-35B-A3B-DFlash2](https://huggingface.co/incoai/Qwen3.6-35B-A3B-DFlash2),
  [z-lab/Qwen3.6-35B-A3B-DFlash](https://huggingface.co/z-lab/Qwen3.6-35B-A3B-DFlash),
  [jzinno/Ornith-1.5-35B-A3B-DFlash2](https://huggingface.co/jzinno/Ornith-1.5-35B-A3B-DFlash2)。
- [DFlash 2 blog](https://inco.ai/blog/dflash2/), [Fastest Speculative Decoding in vLLM with Arctic Inference](https://www.snowflake.com/en/blog/engineering/fast-speculative-decoding-vllm-arctic/), [SuffixDecoding at Production Scale](https://www.snowflake.com/en/engineering-blog/suffixdecoding-arctic-inference-vllm/)（どちらも検索結果の要約経由）、
  ローカル実測のページは §6 の表に URL を併記した。
