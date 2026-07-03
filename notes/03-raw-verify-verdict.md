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
