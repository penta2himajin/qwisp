# 8GB lossless の strict vs near 整理（再現可能）

作成: 2026-06-27 / 対象: Swift PoC `runNoSyncGateEscalate`（buddy hybrid, C=64）

## 背景（認識のズレ）

「8GB で **buddy hybrid(C=64, margin≥2) + M2 escalate** により **strict lossless を ~52 tok/s** で達成した」
という認識があったが、これは **2 つの別の動作点を融合した誤認**だった。実測で決着させたのがこのノート。

結論を先に:
- **strict lossless を保証する buddy 機構は実装されている ＝ `runSpecVerify`（buddy draft + exact verify の投機デコード）。** long128tok でも vs Swift-greedy **100%**。ご記憶の「buddy + M2(exact) フォールバックで lossless」はこれ。
- **SpecK 速度: C=64 ~27-28（≈M0）/ C=128 ~34-36（M0 の ~1.3x）。** strict のまま C を増やすと速くなる → **16GB strict lossless の勝ち筋**。
- **margin≥2 は strict ではない（near-lossless）。** long で vs Swift-greedy 12% に発散。「**52 tok/s**」は near の easy(48tok) 数字。
- **membership ゲートも strict だが二重 forward で C=64 ~15-17（M0 以下）。** strict 目的なら **SpecK か素の M0/M2(~27)**。

## ゲートの意味論（コードから確定）

`swift/Sources/QwispCore/Tell.swift` `runNoSyncGateEscalate` の per-token accept 判定:

```swift
if useMargin {                          // margin≥thresh ゲート（QWISP_MARGIN>0）
    let marginArr = sv[n-1] - sv[n-2]   // no-sync(buddy) 出力の top1-top2 margin
    accept = marginArr.item >= marginThresh   // ← 確信度ヒューリスティック（exact と未照合）
} else {                                // membership ゲート（QWISP_MARGIN=0）
    accept = missArr.item == 0          // ← 全 routed が cache 内 = no-sync gather が exact と bit 一致
}
if accept { cur = nosyncTok }           // 採用時は no-sync(buddy) draft をそのまま採用
else { /* restore → exact 1-forward に escalate */ }
```

| ゲート | accept 条件 | 採用トークンの保証 | 性質 |
|---|---|---|---|
| **margin≥2** | top1-top2 margin が大 | **検証なし**。buddy draft をそのまま信用 | **near-lossless** |
| **membership** (margin=0) | miss==0（全 routed が cache 内） | **bit 一致**（exact と同一計算）。残りは exact escalate | **strict lossless** |

ポイント: margin ゲートは「自信ありそうな buddy 出力」を **exact と照合せず**採用する。高 margin でも buddy が誤れば誤りを採用する → **原理的に strict ではあり得ない**。
membership は採用＝bit 一致・非採用＝exact escalate なので、出力は M0 とトークン単位で完全一致 → strict lossless。

### lossless 機構の本命 ＝ SpecK（投機デコード）

`runSpecVerify`（Tell.swift:398-422）が「buddy + exact フォールバックで lossless」の正体:

```swift
let seq = MLX.concatenated([uArr, draftArr], axis: 1)   // [u, d1..dK]（buddy no-sync で draft）
let (_, vlg) = try model.forwardHidden(seq, caches: mc) // verify = exact gather（skipMode=0）
let evals = MLX.argMax(vlg[0, 0..<(K+1)], ...)          // exact の argmax
var p = 0
while p < K && drafts[p] == evals[p] { p += 1 }         // draft が exact と一致した分だけ受理
// reject 時: accepted prefix を exact 再走で commit し、訂正トークン evals[p] を採用
```

採用トークンは「draft==exact の一致分」か「exact の訂正トークン」のみ → **出力は exact greedy とビット一致＝strict lossless**。
buddy は draft の accept 率を上げる役（外れても verify が捕まえるので lossless は壊れない）。
**margin との違い**: margin は draft を exact と**照合せず**採る（near）。SpecK は**必ず exact verify と照合**する（strict）。

## 再現コマンド

ビルド（metallib のため `swift build` 不可、xcodebuild Release 必須）:

```bash
cd swift
xcodebuild build -scheme qwisp-poc -configuration Release \
  -destination 'platform=macOS' -derivedDataPath ./.xcode-build-rel -skipPackagePluginValidation
```

確認実験（C=64 buddy, pin=48 = 決定(i) の構成。`QWISP_SWIFT_REF=1` で **vs Swift-exact-greedy** を測る）:

```bash
BIN=./.xcode-build-rel/Build/Products/Release/qwisp-poc
run() { QWISP_RUN=no-sync-gate-escalate QWISP_CACHE_C=64 QWISP_PIN=48 QWISP_SKIPMODE=3 \
  QWISP_CALIB=48 QWISP_GEN=128 QWISP_SWIFT_REF=1 QWISP_MARGIN=$1 \
  QWISP_MTP_REF=/tmp/qwisp_$2_ref.safetensors $BIN stream 2>&1 | grep NoSyncGateEscalate; }
run 0 hard   # membership / hard(48tok)
run 0 long   # membership / long(128tok)
run 2 hard   # margin>=2  / hard(48tok)
run 2 long   # margin>=2  / long(128tok)
```

- ref: `/tmp/qwisp_hard_ref.safetensors`(greedy 48tok) / `/tmp/qwisp_long_ref.safetensors`(greedy 128tok, high-entropy)。
  生成は `qwisp/mtp_ref.py`（`QWISP_REF_PROMPT` / `QWISP_REF_NSPEC`）。
- **評価は必ず vs Swift-greedy**。128tok で vs-Python を見ると mlx-swift(stream) と mlx_lm(resident) の f16
  丸め差が自己回帰で蓄積し、M0(exact) すら 11% に見える（無意味）。

## 結果（2026-06-27 実測）

| gate | ref | tok/s | escalate | vs Python | **vs Swift-greedy** |
|---|---|---|---|---|---|
| membership (strict) | hard(48) | 16.7 | 100% | 100% | **100%** |
| membership (strict) | long(128) | 15.6 | 100% | 11% | **98%** ※ |
| margin≥2 (決定i, near) | hard(48) | 49.1 | 6% | 100% | **100%** |
| margin≥2 (決定i, near) | long(128) | 35.8 | 23% | 13% | **12%** |

### SpecK（lossless 機構）実測

```bash
run() { QWISP_RUN=spec-verify QWISP_CACHE_C=$1 QWISP_SKIPMODE=3 QWISP_DRAFT_K=4 \
  QWISP_CALIB=48 QWISP_GEN=128 QWISP_SWIFT_REF=1 QWISP_MTP_REF=/tmp/qwisp_$2_ref.safetensors \
  $BIN stream 2>&1 | grep SpecVerify; }
run 64 hard; run 64 long; run 128 hard; run 128 long
```

| SpecK buddy (K=4) | ref | tok/s | accept/step | **vs Swift-greedy** | RSS |
|---|---|---|---|---|---|
| **C=64** | hard(48) | 26.6 | 3.64/4 | **100%** | 6.9GB |
| **C=64** | long(128) | 27.8 | 3.74/4 | **100%** | 6.9GB |
| **C=128** | hard(48) | 36.2 | 4.00/4 | **100%** | ~11GB |
| **C=128** | long(128) | 34.2 | 3.74/4 | **100%** | ~11GB |

- **SpecK は真の strict lossless**: adversarial な long128tok でも vs Swift-greedy **100%**（margin が 12% に発散したのと対照的）。
  membership が long で 98%（restore バグ）なのと違い **SpecK の restore は正しく 100%** → strict 経路としても SpecK が優れる。
- **C を増やすと strict のまま速くなる**: C=64 ~27（≈M0）→ **C=128 ~34-36（M0 の ~1.3x）**。accept/step が 3.74→4.00 に上昇。
- **C=128 buddy の no-sync 崩壊（下記(c)）は SpecK では無害**: verify が正しさを保証するので buddy は accept 率だけに効く。

### ★ M0/M2 も near-lossless だった（head-to-head, メモリ訂正）

「strict lossless = M0/M2 ~27」というメモリ記述を検証するため、M0/M2/SpecK を C=64・vs Swift-greedy で head-to-head 実測:

```bash
m0(){ QWISP_RUN=predict-prefetch QWISP_CACHE_C=64 QWISP_GEN=128 QWISP_SWIFT_REF=1 QWISP_MTP_REF=/tmp/qwisp_$1_ref.safetensors $BIN stream 2>&1|grep "PredictPrefetch"; }
m2(){ QWISP_RUN=cross-layer-predict ... ; }   # 同様
```

| 手法 | hard(48) tok/s / vs Swift | long(128) tok/s / **vs Swift** | lossless? |
|---|---|---|---|
| **M0** (予測 prefetch + no-sync) | 24.3 / 100% | 27.8 / **11%** | ❌ near（long で発散） |
| **M2** (temporal 予測 one-pass) | 25.8 / 98% | 25.4 / **12%** | ❌ near |
| **SpecK** (buddy draft + exact verify) | 26.8 / 100% | 25.6 / **100%** | ✅ strict |

- **M0/M2 は long horizon で崩壊（vs Swift-exact 11-12%）。** 両者は**予測ベース**（M0=pass-1 の no-sync 出力から routing 予測→prefetch→pass-2 no-sync gather、M2=前 token の temporal 予測）。予測元が corrupted hidden ゆえ recall<100%、自己回帰で drift→発散。**hard(48tok) で 100% に見えたのは short-ref artifact**（buddy 98% が artifact だったのと同型）。
- **M0 の prefetch margin (`QWISP_M0_TOPK`) でも救えない**: 8→16→32→64 と上げても vs Swift-exact は 11%（TOPK=32 が「vs Python 100%」を出すが Swift-exact は 11% のまま、TOPK=64 は Python12/Swift80 とカオス → 発散軌道が偶然 Python に一致しただけの artifact）。
- **SpecK だけが long でも Swift-exact 100%** = 毎トークン exact verify で照合するため構成的に保証。**8GB で真に lossless なのは SpecK のみ**。
- **∴ メモリの「strict lossless = M0/M2 ~27」は誤り。正しくは「strict lossless = SpecK ~26-28、M0/M2 は near-lossless」。**

### ★ lossless ベースライン: SpecK → SuffixSpec に更新（2026-06-28, commit 658ae3d）

**現在の lossless ベースラインは SuffixSpec（SuffixDecoding draft + clean exact verify, 訓練なし, `QWISP_RUN=suffix-spec`）**:
- prompt+生成履歴の suffix lookup で draft を**無料生成(0.00ms)** → exact verify で照合。draft が何でも verify が保証ゆえ lossless。
- 経緯: issue#1(depth-1 MTP draft=20tok, 律速は draft でなく verify)・issue#2 A1(batched-lossless f32 verify=順序安定 f32 でも 30%止まり)を反証し、issue#2 軸B(SuffixDecoding)で到達。GitHub issue #1/#2 参照。

**SpecK（旧ベースライン, `QWISP_RUN=spec-verify`, ~27 / C=128 34-36）は比較基準として有効**。

### ★★ verify = batched f32-full に更新＋maxK 上限解放（2026-06-28, commit 677ef28, investigate C）

**旧 maxK=4 上限の真因を解明し撤廃した。** 旧理解「verify forward 全体の bit-exact 化が要る」は半分正しく、**実際は divergent op は 2 つだけ**:
1. **attention SDPA**: `.causal`(L>1) vs `.none`(L=1) の fused kernel 経路差で f16 ~7e-4 drift。**真因は matmul の L 依存でなく key 数差(verify は draft key も mask 越しに見る)**。micro-test で matmul/RoPE/rmsNorm/softmax/quantized matmul/GDN updateKernel は全て order-stable(rel=0)と確認。
2. **GDN conv1d**: batched≠逐次の f16 drift。

→ **この 2 op を f32 化(f32-attn + f32-conv = f32-full)するだけで verify forward 全体が逐次 decode と bit-exact**（micro-test: perQueryNone-quant rel=0.000 / ATTN-TTEST f32 1.08e-6）。逐次化(seqMT=層丸ごと per-token / perQueryNone=SDPA のみ per-query)は不要で、**単一 batched forward が provably lossless かつ最速**。既定 ON(`QWISP_F32_ATTN/CONV` 既定1)。

**f16 batched は ~7e-4 drift だが SuffixSpec は reject 自己訂正ゆえ実用上 lossless**（誤受理は draft[i] と真 argmax の precise near-tie のみ・保証なし）。f32-full は near-tie でも bit-exact ゆえ**保証付き**。

**maxK 上限の真因は精度でなく per-layer cache 容量**: D+1 トークン verify で 1 層が同時に要するユニーク expert 数が C 超で wrong-slot=silent garbage(クラッシュせず誤受理→品質崩壊)。~~実測安全境界 C=64→maxK24 / C=128→maxK48 = maxK ≤ C×3/8~~。★**訂正(2026-07-01): C×3/8 は安全でない**。diverse routing で union は ~2×C まで膨張し C×3/8 でも overflow=silent garbage、argmax 偶然生存 run のみ 100%=**lossless-by-luck**(C=64 code 3/4 rep 非lossless 実測)。真の strict-lossless は **union-overflow guard**: `LayerExpertCache.ensure` に渡る実 routing の distinct 数で per-layer union を観測(GPU sync 不要)、union>C の step は「union≤C に収まる最長 prefix」へ縮小して re-verify(比例縮小で 1-2 回収束、prefix<1 のみ safe single-token)。honest 値: C=64 code 21/mix 43/nl 20, C=128 code 110/mix 64, C=192+ ほぼ full, C=256 従来通り(mix 282)。`maxK = min(QWISP_DRAFT_K, C×3/8)` は初期上限だが実効上限は guard が動的決定。env QWISP_OVERFLOW_MARGIN(既定80)/QWISP_OVERFLOW_DBG。

実測 vs Swift-greedy **全て 100% lossless**（mix=反復code / nl=自然文）:

| | 8GB C=64 | 16GB C=128 |
|---|---|---|
| mix @maxK16(既定) | 76 tok/s | 114 tok/s |
| mix @maxK24/48(C 上限) | 88 | 132 |
| nl(高entropy) | 24 | 29 |

旧 SpecK ~27 比で **mix 最大 3.3-4.9x**。nl は draft が当たらず greedy 近傍(~24-29)＝SuffixDecoding の特性(反復で伸びる)。

### 読み取り

1. **margin≥2 は near-lossless（strict でない）。** hard(48tok) では 100% に見えるが、long(128tok) で
   **vs Swift-greedy 12%** に発散。原因: margin 採用（long で 77%）は無検証ゆえ、わずかな per-token 誤差が
   **自己回帰で複利的に蓄積** → ある時点で文脈が壊れて以降全滅。**短 horizon で lossless に見えても長 horizon で崩れる**。
   決定(i) の「52tok/s・escalate 5%・near-lossless」は **easy/短(48tok) ref の数字**だった。

2. **membership は ~strict だが遅い（しかも long で 98%）。** escalate 100%（C=64 は footprint 103 > 64 で毎 token どこか miss）。
   no-sync draft を捨てる**二重 forward** ゆえ **15-17 tok/s**。long で 98%（restore バグ, 下記(a)）。
   → strict かつ速い、を狙うなら membership より **SpecK**。

3. **「strict lossless 52」は存在しない。真に lossless なのは SpecK のみ:**
   - strict（Swift-exact 再現）→ **SpecK ~27-28**（M0/M2 は long で発散＝near、membership は ~15-17 で long 98%）
   - 速度が要る → margin≥2 **~36-49** だが **near-lossless で長 horizon 非保証**

   **C を増やせば SpecK は strict のまま速くなる**:
   - **C=128(16GB): SpecK ~34-36 = strict lossless かつ 8GB SpecK の ~1.3x** ← 16GB の本命。
   - 52 は依然 near の数字。strict かつ速い、は SpecK を C で伸ばすのが筋。

## lossless 基準ポリシー（MLX 準拠）

方針: **lossless の ground truth は原典 MLX（mlx / mlx_lm）のセマンティクスに置く**（mlx-swift 固有の挙動ではなく、MLX 仕様に準拠させる）。言語・ハードで変わる実装詳細ではなく、MLX が規定する数値セマンティクスを正とする。

ただし f16 自己回帰の性質上、**2つのレベルを必ず分けて測る**必要がある:

1. **エンジン忠実度（mlx-swift が MLX に準拠しているか）**
   - **free-running の長 horizon トークン一致では測れない**: mlx-swift(f16) と mlx_lm(f16) は 128tok で ~89% 乖離するが、これは**手法の良し悪しでなく f16 の丸め順序差が argmax 反転を経て複利化したカオス**。faithful な再実装でも長 horizon では乖離する。
   - 正しい忠実度メトリクス = **teacher-forced（同一 prefix を強制し per-token で argmax/logits を比較）**。層単位では既に f32 rel ~1e-6 で一致（[[real-model-arch]]）。**per-token teacher-forced で MLX と一致すれば「エンジンは MLX 準拠」と判定**。これは別途の検証 TODO。

2. **手法 losslessness（手法がエンジンの exact を再現するか）**
   - 手法選択の基準は「エンジン自身の exact greedy（=MLX 準拠なら MLX の出力）を再現するか」。
   - SpecK=✅（verify で構成的に保証）、M0/M2=❌（予測で発散）。これは vs Swift-exact(=エンジン exact) で正しく測れる。

**運用**: 手法比較は引き続き **vs Swift-exact**（手法 losslessness を正しく分離）。並行して **teacher-forced の MLX 忠実度チェック**を入れ、「エンジンが MLX 準拠」を担保する。両者が満たされて初めて「MLX 基準で lossless」と言える。

### ★ (e) 実測: エンジンは MLX 準拠（teacher-forced 確認, 2026-06-27）

`measureMLXFidelity`（`QWISP_RUN=mlx-fidelity`）: reference gR を強制入力し、mlx-swift exact の per-token argmax を mlx_lm(gR) と比較。

```bash
QWISP_RUN=mlx-fidelity QWISP_CACHE_C=64 QWISP_GEN=128 QWISP_MTP_REF=/tmp/qwisp_long_ref.safetensors $BIN stream
```

| teacher-forced | per-token 一致 | mismatch |
|---|---|---|
| hard(48) | **100%** | 0 |
| long(128) | **98.4%** | 2、**両方 near-tie**（gR が rank1=僅差2位, logit gap **0.062**） |

- **mlx-swift エンジンは MLX 準拠**: per-token で mlx_lm と 98.4-100% 一致。不一致は **f16 near-tie のみ**（2トークンが ~0.06 logit 差でほぼ同点、40層+lm_head の f16 累積丸めがどちらを上にするか決める）。**ポートのバグでなく f16 の物理（言語・ハード依存）**。
- **free-running 11% の謎を完全説明**: p14 で最初の near-tie flip → 以降カオス発散。teacher-forced 2 flip が free-running 全崩壊の原因。
- **∴ 8GB ベースラインの土台が確定**: (1) エンジン MLX 準拠 ✅ + (2) SpecK がエンジン exact 再現 ✅ → **SpecK は MLX 基準で per-token lossless**（near-tie 除く）。
- **lossless の定義**: 「free-running トークン列の完全一致」は f16 では原理的に不可（near-tie で MLX 自身が非決定的）。正しい定義は **「per-token で MLX 準拠（near-tie 除く）」** で、SpecK はこれを満たす。

## 未解決（このノートで発見、別途調査）

- **(a) membership long が 98%（2tok ズレ）**: 構成上 100% のはず。escalate 前の no-sync draft が GDN 再帰状態を
  進め、`restore(trim:1)` が取りこぼしている疑い。strict 保証の検証に関わるので要追跡。
- **(b) margin≥2 の long 崩壊は「決定(i) 全否定」か**: long_ref は high-entropy の adversarial。実ワークロード
  (code/agentic/長文脈) が hard 寄りか long 寄りかで near-lossless の実用性が決まる。**「長 horizon・実プロンプトで
  vs Swift-greedy が何 % 維持できるか」が near-lossless 採否の本当の判定軸**。
- **(c) C>64 buddy 崩壊（no-sync 経路のみ）**: C=128 buddy は pure no-sync で 12%（C=64 は 98%）。
  **ただし SpecK 経路では無害**（verify が正しさを保証、buddy は accept 率のみ）。near-lossless no-sync を 16GB で使う場合のみ要修正。
- **(d) 16GB strict lossless の伸びしろ**: SpecK C=128 は K=4/accept~3.74。**K を上げる / C を 128→160（16GB 上限）** で
  accept・throughput がどこまで伸びるか未測。24GB は full-resident(C=256) no-sync=exact で ~59 が上限アンカー。
- **(e) ✅解決: エンジンは MLX 準拠**（teacher-forced hard 100% / long 98.4%、不一致は f16 near-tie のみ gap~0.06）。
  `measureMLXFidelity`/`QWISP_RUN=mlx-fidelity`。8GB SpecK ベースラインの土台が確定。詳細は上の lossless 基準ポリシー節。

## メモリとの差分（要更新）

- `nosync-approx-improve` の決定(i): 「near-lossless」は正しいが「**長 horizon で発散・easy ref 限定**」を明記すべき。
- `footprint-vs-budget`: 「C=128→miss0→strict lossless 59」は**偽**（C=128 静的 pin は 15% miss、miss=0 領域なし）。
- `status-8gb-done-16gb-next`: 「near-lossless fast=52」は **48tok easy 限定の数字**である注記が必要。
- **★最大の訂正**: 多数のメモリが「strict lossless = M0/M2 ~27」と記すが**誤り**（M0/M2 は long で発散＝near-lossless）。
  正しくは **strict lossless = SpecK（8GB C=64 ~27-28 / 16GB C=128 ~34-36）。SpecK を 8GB RAM ベースラインとする。**
- lossless 基準は **原典 MLX に準拠**（teacher-forced 忠実度 + 手法 losslessness の2層で判定）。
