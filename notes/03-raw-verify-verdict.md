# D1 raw(seedless)fused engine — A/B 判定書(U4)

作成: 2026-07-03 / branch `feat/raw-verify` @0be9b60 / M1 Max, resident(C=256 相当)

## 結論(判定)

- **正しさ: 完全達成。** raw fused engine は全 regime で self-consistent LOSSLESS
  (spec-vs-greedy 128/128)。code/agentic/mtp_ref では **MLX 正準 refs と token-exact 100%**。
  guard/cert/VSEQ/margin 認証は一切不要 — batched≡sequential が **構造保証**。
- **速度: 比較可能セルで MLX strict の ~0.8x → 「速度≥現行」の採用条件は未達。**
  **default 切替はしない**(このブランチのまま、backend オプションとして保持)。
- 位置づけ: **Tell ランタイムの差し替え可能 backend** として完成
  (MLX backend=margin 認証 lossless / raw backend=構造保証 lossless)。

## 実測マトリクス(N=128, maxK=96, 同一 prompt・同一 refs)

| regime | raw fused(1-CB step) | MLX strict(campaign 後) | raw/MLX | raw 品質 vs 正準 refs | raw self-check |
|---|---|---|---|---|---|
| code    | 44.9 tok/s | 56.3 | 0.80 | **128/128=100%** | LOSSLESS |
| agentic | 38.4 | 46.2 | 0.83 | **128/128=100%** | LOSSLESS |
| longctx | 62.7 | 44.1 | (1.42)* | 2/128(別軌道) | LOSSLESS |
| shortnl | 46.9 | 56.0 | (0.84)* | 11/128(別軌道) | LOSSLESS |
| mtp_ref | 107.8 | 138.8 | 0.78 | **128/128=100%** | LOSSLESS |

\* longctx/shortnl は raw engine が MLX 正準と別の(それ自体決定的な)greedy 軌道を辿る
(hidden rel ~9e-2 の近接 tie 反転、既知)。別テキストの生成なので速度は直接比較不可。
raw を採用する場合は **refs を raw 正準で再生成(U3)** すればこの 2 regime も fidelity 軸 100% になる
(self-check は既に全 regime 100%)。

## アーキテクチャ(何を作ったか)

`RawFusedForward`(RawFusedVerify.swift): **decode/verify step 全体が 1 command buffer**。
- `stepArgmax(tokens)`: embed_rows_q4 → 40 層(norm→mixer(attn|GDN)→resid→postNorm→MoE→resid)
  → final norm → lm_head(qmm4_tiled)→ argmax_rows。**ループ内 MLX op ゼロ、readback は int32 [M] のみ**。
- cache 全常駐: KV [KV,maxLen,D]、GDN conv hist / rec state は **ping-pong**。
- spec の partial reject rollback = KV len 巻き戻し + ping-pong swap 戻し(**コピー無し**、直前 1 step 限定)。
- composed(per-op)経路と **同一 kernel・同一丸め列**(elementwise も raw kernel に統一済)
  → test_raw.sh 24 本が fused≡composed を bit レベルでゲート。

## 学び(再発防止・引き継ぎ)

1. **asMTLBuffer(noCopy:true) の寿命規約**: `asType()` 一時 MLXArray の zero-copy buffer を
   長寿命 struct に包むと allocator 再利用で重みが実行中に破壊される。合成テストは
   alloc が少なく**偶然通過**し、実モデルで全トークン 0 のゴミが出た。
   → 全 prepare* が裏 MLXArray を `retained` で保持。
2. **self-check 単独は不十分**: 壊れた engine の self-check は「ゴミ≡ゴミ LOSSLESS」。
   **composed との実重み OUT_TOKENS diff** が決定的ゲートだった。
3. fused≡composed の数値乖離は combine 段の FMA 丸め差 1 点のみ(段階バイセクト
   `fusedMoEBlockRowsDump` で特定)→ composed 側を同一 kernel に統一して解消。
   統一後も既存正準 refs と code/agentic/mtp で 100% 一致(出力トークン不変)。

## 残 upside(採用条件を満たすには)

現状 M=1 純 decode 22.3ms/tok(44.9 tok/s)vs MLX 17.8ms/tok(56.3)。ギャップ ~4.5ms/tok。
- **profile 先**(過去の教訓: single-thread kernel が 63% だった前例)。候補: sdpa/conv の
  M=1 dispatch 形状、route_top8_rows の 1-threadgroup 律速、lm_head qmm4_tiled の M=1 効率
  (qmv 形へ切替可)、CB commit/wait レイテンシ。
- MLX strict 側は speedup campaign(B2/A2/A1/A4/A3/B3)適用済み・raw は初回最適化のみ、
  という非対称も考慮(raw にも伸び代がある)。
- bolt との相性: buddy=slot 差し替えは raw の明示 buffer と好相性(将来の streaming tier)。

## 再現手順

```sh
# build
cd swift && xcodebuild build -scheme qwisp-poc -configuration Release \
  -destination 'platform=macOS' -derivedDataPath ./.xcode-build-rel -skipPackagePluginValidation
# 回帰(モデル不要, 24 テスト)
qwisp/test_raw.sh
# raw fused spec(実モデル)
QWISP_RUN=raw-spec QWISP_RAW_FUSED=1 QWISP_GEN=128 QWISP_RAWSPEC_CHECK=1 \
  QWISP_MTP_REF=refs/code.safetensors \
  swift/.xcode-build-rel/Build/Products/Release/qwisp-poc stream
```

---
## ★追記(2026-07-03 続き, commit 5d2ca73): qmv lm_head で U4 判定が覆る

profile(raw-fused-prof)で M=1 decode の 40% が lm_head と判明 → lm_head を qmm4_tiled から
**qmm4(per-row qmv, 高 occupancy)** に変更(既定 ON, QWISP_LMHEAD_QMV=0 で戻せる)。tiled の 8KB
threadgroup 低 occupancy が weight 再利用の利を上回り、**全 M で qmv が速い無条件改善**(M=1 step
21.2→14.0ms, M=17 139→96ms)。qmv も M 不変(既存テスト)ゆえ decode≡verify=self-consistent 維持。

### 更新後 A/B(raw fused+qmv vs campaign 済み MLX strict, GEN=128)
| regime | raw fused+qmv | MLX strict | 倍率 | 品質 |
|--|--|--|--|--|
| code | **67.6** | 56.3 | **1.20x** | refs 128/128 token-exact + self LOSSLESS |
| agentic | **59.5** | 46.2 | **1.29x** | refs 128/128 token-exact + self LOSSLESS |
| longctx | **112.5** | 44.1 | **2.55x** | raw 固有正準(near-tie 別軌道)・self LOSSLESS |
| shortnl | **70.2** | 56.0 | **1.25x** | raw 固有正準・self LOSSLESS |

### 更新後の判定
**raw fused engine(qmv lm_head)は全 regime で MLX strict を上回り(1.2-2.55x)、かつ構造保証 lossless
(cert/guard/VSEQ 不要)。code/agentic は MLX 正準と token-exact。→ default 昇格候補。**
残: ①longctx/shortnl の fidelity 軸 100% 化 = U3(raw 正準で refs 再生成)②全 RAM tier(streaming)検証
③本番配線(Tell default backend 切替の是非は owner 判断, auto-commit 禁止)。RAWTESTS 24/24 green。

---
## ★追記(2026-07-03 続き2, commit b3f06fb/392dc86): streaming tier(C<256)対応と判定

RawFusedForward を C<256 per-layer LRU arena(LayerExpertCache)で動くようにした。

### 機構
- **RawFusedExpertProvider**: 層別 C-slot arena の 9 gather buffer(gate/up/down×w/s/b)+ 同期 ensure(pread)。
  gqmm4_rows は inds で重み先頭次元を index するだけなので、**kernel 変更ゼロ**で arena buffer に差し替え可能。
- **strict streaming**: MoE を route/gather/shared の 3 フェーズに分割し、層ごとに CB を切って
  route inds を readback → `ensure`(miss pread)→ 層別 slotTable 書込 → 新 CB で slot_remap_rows+gather。
  **union>C は貪欲行チャンク分割で厳密対応**(chunk 間で CB を wait してから次 ensure)→ guard/cert 不要の
  lossless-by-construction を streaming でも維持。draft 長のハード上限が消える(chunk が増えるだけ)。
- **bolt**: TellBolt 準拠 calib(counts+coact、strict readback に相乗りする indsCaptureHook=無料)→
  exact prefill → top-C ensure + buildBuddyTable → **層別 frozen table を GPU buffer に凍結 → 1-CB 維持で io=0**。
  deviation: B3 in-decode fetch / A3 pending-prefix は未実装(v1)。
- runner: `QWISP_RAW_C=<C>`(streaming, maxK default C·3/8)、`QWISP_RAW_BOLT=1`、
  `QWISP_RAWSTREAM_CHECK=1`(resident fused と OUT_TOKENS diff する gate)。

### 正しさ gate(全通過, M1 Max)
- RAWTESTS 24→**28/28**(strict≡resident bit テスト C=8<E=16 合成: chunk 発生・LRU eviction 連鎖・bolt exact-table)
- 実重み: strict C=64 code GEN=128 → ref 128/128・**stream-vs-resident 128/128 IDENTICAL**・self LOSSLESS。
  C=128 agentic GEN=64 → 64/64 IDENTICAL + LOSSLESS。resident 回帰 69.6 tok/s 128/128(経路不変)。

### 速度(GEN=128, timed=decode, throttle=preadInto leaky-bucket 1.5GB/s, MLX は同条件 bench-batch 実測)
strict(両者 lossless):
| cell | raw fused | MLX strict | 比 |
|--|--|--|--|
| C=64 fast code/agentic | 24.5 / 18.3 | 21.7 / 20.2 | 1.13x / 0.91x |
| C=128 fast code/agentic | 31.3 / 26.4 | 28.6 / 28.0 | 1.09x / 0.94x |
| C=64 slow code/agentic | 6.4 / 3.5 | 5.9 / 4.0 | 1.08x / 0.88x |
| C=128 slow code/agentic | 11.6 / 7.9 | 11.0 / 8.5 | 1.05x / 0.93x |

bolt(L3 near-lossless):
| cell | raw bolt | MLX bolt | 比 |
|--|--|--|--|
| C=64 fast code/agentic | 105.5 / 56.7 | 31.1 / 33.3 | **3.4x / 1.7x** |
| C=64 slow code/agentic | 100.5 / 54.1 | 29.1 / 31.6 | **3.5x / 1.7x** |

### 判定
- **strict streaming: MLX strict とパリティ**(code +5-13% / agentic −6-12%)。両者とも per-layer sync 律速で、
  raw の resident での優位(dispatch スラック)は sync 床に食われる。lossless 保証は raw の方が強い
  (構造保証、guard/OVERFLOW_MARGIN/prefill-chunk 定数などの運用注意が不要)。
- **bolt streaming: raw が 1.7-3.5x で圧勝、throttle 完全非感応(io=0)**。slow-NAND 8GB 相当で 100 tok/s 級(code)。
  1-CB no-sync 構造が raw の edge と噛み合う本命セル。
- 残課題: ① raw bolt の品質軸(teacher-forced fidelity)未計測(free-run 3-4% は greedy-chaos 指標で無意味、
  MLX bolt の TF ~88-97% に相当する測定が要る)② B3 in-decode fetch / A3 pending-prefix の移植(bolt 更なる上積み)
  ③ runner の THROTTLE_DEFER 対応(slow セルの wall 時間短縮のみ、数値不変)。
