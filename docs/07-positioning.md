# 07 — ポジショニングと v0.1 実装計画

competitive 精査（2026-06）を経た Qwisp の確定ポジショニングと実装計画。

## 哲学：極限まで高められた "実用" 性能

**制約機（Apple Silicon・限られた RAM）で、実用に直結する {能力・文脈・速度} を物理限界まで高める。**

「実用」が核：synthetic な速度でなく、実タスク（コーディング／エージェント／長文脈）で使えること。
- **能力**：Qwen3.6-35B-A3B（SWE-bench 73.4＝active の 10倍級に勝つ）。
- **文脈**：64K 以上（ネイティブ 262K。DeltaNet ハイブリッドで長文脈が安い）。
- **速度**：制約機で ≥30 tok/s（インタラクティブに使える実用速度）。

## DS4・SwiftLM の鏡像

| | 目的 | モデル(active) | 速度 |
| --- | --- | --- | --- |
| DwarfStar4 | 巨大を高メモリ機にギリ載せる | 284B | 載れば可 |
| SwiftLM | 巨大 MoE を**汎用に**小機へ | 122-397B (10-22B) | 遅い (5.2 tok/s) を許容 |
| **Qwisp** | **十分賢いモデルを制約機で極限速度** | **35B-A3B (3B)** | **速さが第一目的** |

**モデル選択が thesis**：active 3B＝ストリーム負荷が SwiftLM の 1/3。「能力を保つ最小 active の sweet spot を選び、速さに全振り」＝汎用 SwiftLM には構造的に取れない位置。

## 競合（SharpAI: SwiftLM / TurboQuant-MLX）

- **SwiftLM**：ネイティブ MLX-Swift。expert SSD streaming、**OS ページキャッシュで eviction**（app-LRU は OS-cache に負けると測定し棄却）、**前トークン routing で先読み prefetch（~70%hit）**、**別 draft（Qwen3.5-9B, ~10.6GB 常駐）**で投機、汎用 MoE、runtime **top-k 削減**で高速化（品質劣化）。122B-A10B/M1 Ultra で 5.2 tok/s。
- **TurboQuant-MLX**：3-bit 重み量子化＋KV 量子化。「sparse MoE のメモリ壁＝ディスク帯域の壁」（＝我々の flash-bound 結論）。
- **手元 mtplx**：MTP 投機＋TurboQuant（KV/重み量子化）。**expert streaming は無い**＝補完的。

## 差別化（MTP ヘッドに集約）

1. **MTP-head を prefetch オラクル＋(高B時)投機に**：SwiftLM の ~10.6GB 常駐は**別 draft 9B**が占める。Qwisp はモデル内蔵 **MTP ヘッド（~0.5GB, `mtp.safetensors`）**を使う → 常駐を大幅圧縮、より小さい機へ。
2. **先読みの質**：SwiftLM は前トークン流用（reactive ~70%）。Qwisp は MTP で**実次トークンを予測**→その experts を prefetch（受理 ~88%）→ **hit を 80-90% へ**。
3. **混合精度 expert**：SwiftLM は top-k 削減で高速化（expert を捨て品質劣化）。Qwisp は **top-k 維持で cold expert を低bit化**＝品質保持で速度。
4. **単一モデル特化**（DS4 哲学）：Qwen3.6 の実測ルーティングに全チューニング。

**やらないこと**：汎用化、top-k 削減での品質劣化、reach のためだけの低速化。

**注意（cache policy）**：SwiftLM の「app-LRU < OS-cache」は彼らの**zero-copy mmap** 文脈の話。Python/MLX で expert を copy する我々の経路では OS-cache の無料再利用が効かず**明示キャッシュが必要**＝eviction 方策（LRU→MTP 予測器）の余地は我々の文脈では残る。v0.1 で zero-copy mmap→gather_qmm が Python で可能かを確認し、cache 戦略を決める。

## 4つ組（決定④、速さ第一に reframe）

> **`{Qwen3.6-35B-A3B 4bit（cold→混合精度）, 最低 ≥30 tok/s（狙い 50）, 64K ctx（KV量子化）, フロア 12GB(KV量子化)/16GB(余裕) Apple Silicon}`**

「RAM 予算の制約下で tok/s を最大化」が主、reach は従。

## v0.1 実装計画（Python/MLX、製品化フェーズで Swift）

**目標**：expert を materialize しないカスタムロードで、出力一致を保ちつつ、bounded resident で実生成 → 実 tok/s・実 peak を Max-reach 動作点で測り、シミュ gate を実機検証。prefetch 無しの baseline。

`qwisp/` パッケージ:
- `loader.py` — モデルロード surgery（mlx_lm 骨格＋非expert のみ常駐、switch_mlp.* は未ロード）。ExpertSource（mmap/pread で per-expert スライス）。
- `cache.py` — ExpertCache（明示・bounded、v0.1 は LRU、予測器を差せる IF）。
- `streaming_moe.py` — StreamingSwitchGLU（PoC 4.1 を製品化、cache 経由 gather_qmm）。
- `engine.py` — streaming モデル組立＋生成ループ。
- `verify.py` — full との出力一致（安全網）。
- `bench.py` — tok/s＋peak RSS 計測。

マイルストーン:
1. **ロード surgery**：非expert 常駐、switch_mlp 未ロード → load 後 RSS ≪ 20GB を確認（要点）。
2. **正しさ**：生成して full と出力一致（許容誤差）。
3. **計測**：tok/s＋peak RSS を B・context（64K 含む）で → sim 予測と突き合わせ（gate の実機検証）。
4. **方式決定**：zero-copy mmap（OS-cache）が Python/MLX で可能か → cache 戦略確定。

v0.2 以降：MTP-head prefetch（差別化）／混合精度 expert／KV 量子化（64K）／Speed 動作点（高B で MTP 投機）。

## v0.1 実測結果（M1-M3 完了、2026-06）

`qwisp/` パッケージで実装・実機計測（24GB Apple Silicon、ctx 232、gen 64）。

**M1 ロード surgery**：`load_streaming`（mlx_lm.load(lazy=True)→switch_mlp 差し替え→非expert のみ eval）で **RSS 1.79GB**（full eager 20.14GB）。expert 非常駐ロード成立。非expert 常駐は ~1.8GB（推定より小）。

**M2 正しさ（`verify.py`）**：streaming（LRU cache）の greedy 出力が full と **24/24 トークン完全一致**。PoC 4.1 の層単位 bit 一致に続く end-to-end 確認。

**M3 実 tok/s（`bench.py`、AC 電源・確定値）**：

| B/層 | decode tok/s | cache | peak RSS | 機 |
|---:|---:|---:|---:|---|
| 32 | 11.9 | 2.26G | 4.38G | 8GB 余裕 |
| 64 | 15.2 | 4.53G | 6.75G | 8GB |
| 128 | 23.1 | 9.05G | 11.3G | 12GB |

→ **35B-A3B が 6.75GB で 15tok/s / 11.3GB で 23tok/s ＝制約機で実用速度を実機達成。**

給電依存：バッテリー駆動だと ~2-12% 低下（低B ほど顕著＝IO/Python 負荷が CPU スロットルに敏感）。計測は AC で確定。

**シミュ突き合わせ**：実測はシミュ net_tps 予測より ~27% 低い（B=64: 15.2 vs 20.8）。差は給電でなく **Python オーケストレーション＋毎回の subset concat**（シミュ未モデル）。overlap(prefetch) 予測は B=64 で 30tok/s ＝ **MTP-head prefetch＋Python オーバーヘッド削減で 15→25+ tok/s の余地**が定量化。

**v0.2 の作業対象（gap を埋める）**：(1) cache に stacked subset を保持し concat を省く、(2) async prefetch（pread は GIL 解放）で flash と compute を overlap、(3) MTP-head 予測で prefetch hit を上げる。

### v0.2 最適化（実測）

プロファイルで decode の self-time 最大は `StreamingSwitchGLU.__call__`（Python オーケストレーション）と判明。

- **np.unique 化（採用、+12%）**：`set/sorted/dict/np.vectorize` → `np.unique(return_inverse=True)` 一発。verify 24/24 維持。decode B=64 15.2→17.0、B=128 23.1→25.9 tok/s。
- **slotted cache（棄却・ネガティブ結果）**：永続 [B,...] スタック＋slot 単位インプレース更新（`a[slot]=row`、GPU slot lookup も可）を試作。だが **-8〜15% 悪化**。mlx の行インプレース更新はサブセット concat より速くない（関数的 scatter で安くない）。低 RSS は Metal バッファが ru_maxrss に出ない見かけ。**revert**。
- **残るボトルネック（真因）**：per-layer の `inds.tolist()` **GPU-eval 同期**。on-demand では層ごとに「router 計算→CPU で expert id→ロード/gather」の往復が必要で、forward パイプラインが層ごとに直列化。B=128（hit 高・IO 少）でも overhead ~20ms/token がほぼこの同期。slotted では触れられない。

**同期を消すには**（=本命だが重い）：MTP-head 予測で forward 前に必要 expert を常駐させ、**GPU slot lookup（`slot_of[inds]`、`.tolist()` 不要）＋miss 時のみ再計算**する投機的キャッシュ。これは差別化（MTP-head prefetch）の本体でもある＝v0.3 の研究課題。

### 64K 文脈の実機検証（哲学「文脈」軸、B=64）

合成プロンプト（コード片反復で長さ確保）で系の性質（メモリ・速度＝長さ依存・内容非依存）を測定：

| ctx | prefill_tps | decode_tps | mxpeak |
|---:|---:|---:|---:|
| 4,096 | 271 | 15.5 | 7.76GB |
| 16,384 | 268 | 15.1 | 8.85GB |
| 65,536 | 206 | **15.3** | **13.26GB** |

- **decode が長さに依らずフラット（15.5→15.3）** ＝ Gated DeltaNet ハイブリッド（40層中ほぼ線形注意=O(1)状態）の payoff。長文脈でも decode が重くならない。
- **64K が ~13.3GB に載る → 16GB 機**（KV量子化＋B縮小で 12GB 射程）。KV 増は 16× 長で +5.5GB のみ＝長文脈 KV が安いことを実測。
- 留保: `hit` は反復プロンプトで楽観（routing 多様性は実素材=RULER/RepoQA で要再測）、64K prefill は ~5分（thrash 由来、要改善）。**メモリ・decode 速度の結論は有効。**

→ 哲学3本柱のうち**「文脈」軸を実機で確認**（能力=35B-A3B/SWE73.4、速度=8GBで17・12GBで26tok/s、文脈=64K/16GB・decodeフラット）。

### A-1: 同期除去の天井測定（probe）

`.tolist()` 同期が真のボトルネックか確かめるため、decode 中に前トークンの (sub, remap) を再利用して per-layer の同期/CPU 作業をゼロにする probe（`StreamingSwitchGLU.probe_no_sync`、出力は不正、速度天井測定用）：

| | decode tok/s |
|---|---:|
| probe OFF（正しい baseline） | 17.6 |
| probe ON（同期除去の天井） | **79.9** |

→ **per-layer 同期除去で 4.5×** ＝同期が圧倒的ボトルネックと確証、A（投機的 prefetch エンジン）を強く正当化。
ただし 79.9 は前トークンの**同一 experts 再利用**で mlx が定数最適化する理想値（full-AR 54 すら超える非現実）。**正しい no-sync エンジンの現実的天井は ~40-54 tok/s 級（現状 26 の約2×）**と見る。

### A-2 計画（正しい投機的キャッシュ）

1. 各層 GPU `slot_of`[256] int32（slot or -1）。forward は `remap = slot_of[inds]`（GPU、`.tolist()` 不要）。
2. トークン間に prev-token（or MTP-head 予測）の experts を prefetch → 常駐＋slot_of 更新。記録は1トークン1回の batch sync。
3. miss（`slot_of[inds] < 0`）を GPU フラグで検出 → そのトークンのみ同期して再計算（投機的）。
4. slotted の in-place 更新が遅い問題は、prefetch をトークン間（per-layer critical path 外）に出すことで回避を狙う。

### A-2a 実測（近似版、`qwisp/prefetch.py`）→ 構造的な壁

prev-token prefetch＋GPU slot_of remap（miss は slot0 クランプ＝近似）で速度と一致率を測定（B=128）：

| | decode tok/s | 一致 |
|---|---:|---|
| exact | 18.0 | 64/64 |
| prefetch(近似) | **33.0 (1.84×)** | 37/64（38 で発散）|

- **同期除去の現実効果 = 1.84×**（probe の 4.5× は同一 experts 再利用の理想値）。
- **miss率 17.8%**＝320 access/token のうち ~57 miss ＝**ほぼ全トークンに miss**。
- **∴ exactness と高速は直接相反**：厳密化（miss 検出→再計算）はほぼ毎トークン再計算になり 1.84× が消える。per-layer miss 検出＝同期＝1.84× の源と相反。
- 根本原因は**トークン単位 routing churn**（連続トークンで experts が 17.8% 入替、Step1 の churn と整合）。prev-token prefetch は構造的に限界。

**結論**：同期除去の高速化を exact に成立させるには miss を near-zero にする必要があり、それは churn ゆえ困難。MTP-head prefetch（実次トークン予測）は prev-token より miss を下げうるが near-zero 保証なし＝**不確実な v0.3 研究賭け、quick win ではない**。

→ **当面の shippable エンジンは v0.2（np.unique, exact, 8GBで17/12GBで26 tok/s, 64K/16GB）。** 速度の次段（同期除去の exact 化）は MTP-head 予測精度に懸かる研究課題として分離。

### prefetch 予測の軸を間違えていた（関連研究調査）

オフライン解析（実 trace 204,800 decode 行）で **時間方向（cross-token, 同一層）予測**を評価：

| 予測器 | coverage(hit) | miss |
|---|---:|---:|
| window=1（prev-token） | 0.337 | 0.663 |
| window=16 | 0.725 | 0.275 |
| freq top-128 | 0.850 | 0.150 |

→ **prev-token は 66% miss**＝time 方向は churn で不能（A-2a の合成 17.8% は反復ゆえ楽観）。

**文献の答え＝予測すべきは「層方向（cross-layer, 同一トークンの次層）」**：
- Cross-Layer Gate（arXiv 2502.12224）: 次層 **96%**、2-3層先 ~90%。
- ProMoE（2410.22134）: 予測器＋proactive prefetch、LRU比 **2.06×**、**精度無劣化（厳密）**。
- OD-MoE shadow: **>99%**。MoE-SpeQ（2511.14102）: **2.34×**。Pre-gated MoE/Fate: **1.8×**。
- 機構: 層 L 計算中に L+k の gate を先回し→experts 予測→IO prefetch を計算に overlap。miss は on-demand＝**correctness 厳密**。

留保: (1) 我々は既に cache 済み（B=128 hit 85%, IO ~13%）→実効ゲインは無キャッシュ比の文献値より小さい可能性。(2) SSD offload は DRAM比 5-12× エネルギー/token（バッテリー実用の留保）。

→ measure-first で **cross-layer 予測精度を自モデルで実測**してから本実装を判断。

### cross-layer 予測の実測（自モデル、zero-shot）

層 L の gate入力 hidden に層 L+k の gate を当て、実 L+k experts への coverage：

| 予測 | coverage | miss |
|---|---:|---:|
| k=1（次層） | 0.771 | 0.229 |
| k=2 | 0.710 | 0.290 |
| k=3 | 0.669 | 0.331 |

- **層方向 77% ≫ 時間方向 34%**＝文献の「cross-layer が正解」を自モデルで確認。
- **だが zero-shot 77% < 既存 LRU cache hit 85%**＝「次層 gate を現層 hidden に当てる」だけでは既存キャッシュに勝てない（L+1 の attention/DeltaNet が hidden を変える）。文献 96-99% は**学習予測器 or pre-gating＋finetune**前提。
- 我々は既に cache 済み（IO ~13%）→ **完璧な予測器でもゲイン上限 ~13%**（miss の IO を1層先回しで隠すのみ）。同期-wait（1.84× の源）は cross-layer でも残る。

### 速度最適化の結論（空間を地図化）

| 手法 | 結果 |
|---|---|
| np.unique | +12%（採用・exact）|
| slotted cache | 棄却（-15%）|
| 同期除去（A-2a） | 1.84× だが churn で発散（exact 不成立）|
| 時間方向 prefetch | 66% miss（churn の壁）|
| cross-layer prefetch | 77% zero-shot（軸は正、既存 cache 85% 未満、文献 96% は要学習、ゲイン IO ~13% 上限）|

→ **合理的労力での速度天井は v0.2（exact, 8GB17/12GB26 tok/s）**。さらなる速度は「cross-layer 予測器を学習（~13% IO-overlap）」or「投機的近似計算＋発散検出（不確実）」で、労力大・ゲイン限定＝v0.3 研究として分離。**speed は実用域で確定、reach/文脈/品質で締めるのが筋。**

### 次の打ち手の評価（調査）と判断

| 軸 | 手間 | 見返り | 判断 |
|---|---|---|---|
| (う) cross-layer 予測器学習（ProMoE 2層MLP、本環境で MLX 学習可、1-2日） | 中 | 速度 ~13%（既存 cache 85% ゆえ IO 分に bounded） | **見送り**（ROI 低、(い) の結果後に再検討）|
| (い) mixed-precision expert（HOBBIT: miss を低bitロード→load 4×・精度 max-1%） | 中〜大 | 速度 ~6-13%＋**reach/品質保持**（top-k drop と違い品質を捨てない＝差別化） | 採用（KV量子化の後）|
| (い) KV 量子化（4bit KV→1/4・RULER64K <1%劣化、手元 TurboQuant 流用） | 低〜中 | **64K を 13.3GB→~9GB（12GB 余裕・より小機）** | **最優先で着手** |

速度は churn/IO の物理天井に到達済み。哲学「実用性能」は速度だけでなく **reach・文脈・品質**を含むので、伸びしろは (い)＝KV量子化（長文脈 reach）→ mixed-precision（品質保持のサイズ/速度・差別化）にある。(う) は (い) の結果後に再検討。

### KV 量子化の実測 → 低価値（hybrid が既に KV を解いている）

mlx_lm ネイティブ KV 量子化（`kv_bits=4`）を 64K で実測：

| ctx | kv_bits | decode_tps | mxpeak |
|---:|---:|---:|---:|
| 16K | FP16 | 16.8 | 8.85GB |
| 16K | 4 | 16.0 | 8.60GB |
| 64K | FP16 | 16.1 | 13.26GB |
| 64K | 4 | 12.9 | 12.22GB（-1.04, -8%）|

- 期待（~4GB 削減）に反し **64K で 1GB のみ＋速度やや低下**。理由：**40層中 full-attention は ~10層**（残り DeltaNet=O(1)）→ **KV が元々小さく量子化対象が少ない**。「文脈が安い」哲学の追加裏付け。
- **64K の支配項は KV でなく expert cache（B=64 で 4.5GB）＋prefill 活性**。
- → **KV 量子化は本 model では見送り**。reach の本命レバーは **mixed-precision expert**（cold を低bit→expert cache 半減/2×B、品質保持＝HOBBIT 流の差別化）。

### mixed-precision の de-risk（品質）→ 大工事・不確実

低bit expert の品質コストを 4bit↔3bit(unsloth) のトークン一致で測定：**9/40、6 で発散**。
- 一律低bit は品質を明確に落とす。ただし unsloth 3bit は別量子化方式で **conflated**（bit幅のみの効果でない）。
- 本命の mixed（hot 4bit/cold 2bit）はこれより劣化小のはず（HOBBIT ~1%）だが、確認には**本実装が必要**：2bit源作成（4bit→dequant→requant）＋混合bit two-gather（`gather_qmm` 単一bitゆえ hot/cold 2回 gather 合成）＋verify ＝**数時間級・品質リスク・quick de-risk 不可**。

→ **diminishing returns の境界**。mixed-precision は唯一の残レバーだが「大工事＋不確実品質＋quick de-risk 不可」。proxy は非好意的。**速度（churn/IO 天井）・reach（KV は小, expert cache が支配）・各レバー ROI をすべて実測で地図化済み。v0.2 を shippable な到達点として締めるのが合理。**

### (う) cross-layer 予測器の de-risk（学習）→ 割に合わない

`qwisp/predictor_eval.py`：prefill から `(層L-1 gate入力, 層L experts)` を ~4900例/層 集め、tiny MLP（2048→256→256）を MLX 学習し top-8 coverage を測定：

| | coverage |
|---|---|
| trained MLP（平均） | **0.535** |
| zero-shot（次層 gate を現 hidden に） | 0.771 |
| 既存 LRU cache hit | 0.85 |
| 文献（ProMoE/Fate） | 0.96 |

- **trained 0.54 < zero-shot 0.77 < cache 0.85**＝予測器を入れても既存 cache に勝てない。zero-shot は本物の次層 gate ゆえ強く、小 MLP は限データで再学習しきれず下回る。
- 文献 96% には大予測器＋多様大量データ＋Fate 等の正確手法が要り、hybrid model への transfer 不確実。届いても見返り ~13%（cache 85% ゆえ IO 分 bounded）。
- → **(う) も「大工事＋不確実＋見返り限定」。measure-first が安価に却下。**

## 最終結論：最適化空間を完全に地図化、v0.2 で締める

| レバー | 実測 de-risk | 判定 |
|---|---|---|
| np.unique | +12% | **採用（v0.2）** |
| slotted cache | -15% | 棄却 |
| 同期除去（A-2a） | 1.84× だが churn 発散 | exact 不成立 |
| cross-layer 予測器（う） | trained 0.54 < cache 0.85 | 割に合わず |
| mixed-precision（A） | proxy 発散・大工事・quick de-risk 不可 | 高リスク低 ROI |
| KV 量子化 | 64K -1GB のみ（hybrid が KV を解決済） | 見送り |

**v0.2 = shippable 到達点**：35B-A3B が exact・8GB17/12GB26 tok/s・64K/16GB・full と bit 一致。「極限まで高められた実用性能」を実機達成。

### mixed-precision のミニマル品質検証 → GREEN（先の悲観を覆す）

同一モデル・同一方式で expert を低bit roundtrip（精度だけ落とし格納は4bit、既存 gather で動かす）し品質測定（`qwisp/mixed_probe.py`）：

| 条件 | token 一致 |
|---|---|
| 全 experts 2bit（最悪） | 8/40（token4 で発散）|
| **mixed hot128@4 / cold128@2bit** | **40/40 完全一致** |
| **mixed hot128@4 / cold128@3bit** | **40/40 完全一致** |

- **hot を 4bit 維持・cold のみ低bit化で出力は full と完全一致**＝HOBBIT 仮説が我々のモデルで成立。cold は選択頻度が低く低精度でも argmax を変えない。
- 先の悲観（mixed は不確実）は **conflated proxy（unsloth 3bit-all 別方式）＋最悪ケース（全2bit）由来の誤り**。ミニマル検証で**主リスク（品質）retire**。
- 留保：40tok/1prompt の一次証拠。本番は多/長/多様 prompt で再確認。

→ **評価が変わる：mixed-precision が唯一の生きたレバー**。予測器（う）死亡・KV量子化低価値の中、mixed は**品質 GREEN・reach・差別化**。価値=cold 半分を 2bit→miss-IO 半減/格納半減（reach）/品質保持（top-k drop と違う差別化）。コスト=混合bit two-gather 実装（中〜大）＋MLX 2bit dequant 速度（要検証）。

> 関連: go/no-go [[go-no-go-first-read]]、Step4 PoC [[step4-poc]]、MTP×streaming [[mtp-streaming]]。
