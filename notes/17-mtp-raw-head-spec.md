# notes/17 — ①③ Step 3: RawMTPHead(MTP head の raw Metal 移植)spec

Date: 2026-07-08 | Base: feat/raw-verify @cb2a546 (RAWTESTS 72/72) | Driver: Fable
Recon 正典: `~/.claude/projects/-Users-penta2himajin-repos-qwisp/recon-raw-port-2026-07-08.json`
前提知識: notes/15(MLX hybrid spec)、notes/03(raw doctrine)、MTPHead.swift(MLX 実装=数学の正)。

## Goal

MTP head の 1-token draft を **raw Metal(単一 command buffer、readback は int32 draft 1 個のみ)**
で計算する `RawMTPHead` を実装する。ループ配線(runSpecLoop の D==0 seam)は Step 4 スコープ外 —
本 Step の deliverable は head 機構+検証 runner のみ。**既存経路の挙動変更ゼロ**(全て追加コード)。

## 数学の正(MTPHead.swift:73-90 を厳密 mirror)

```
emb = embed4bit(tok)                       // main embed_tokens 共有(4bit gs32)
e   = rmsNorm(emb, preEmb)                 // mtp.pre_fc_norm_embedding
hh  = rmsNorm(hPrev, preHid)               // mtp.pre_fc_norm_hidden — hPrev = main の post-final-norm hidden
cat = concat([e, hh])                      // [1, 2H]  ★emb が先(emb_hid 順)
x   = cat @ fcᵀ                            // fc [H, 2H] F16 plain
r   = attn(rmsNorm(x, inputLN), kv)        // 下記 attn 節
x   = x + r
x   = x + moe(rmsNorm(x, postLN))          // 下記 moe 節
nrm = rmsNorm(x, mtpFinalNorm)
logits = nrm @ lm_head                     // main lm_head 共有(4bit)
draft = argmax(logits)
```

- 重み: `<modelDir>/mtp.safetensors`、key prefix `mtp.`。**norm の +1 シフト復元は
  in/post_ln と q/k_norm のみ**(MTPHead.swift:23-24 `gn()`)。pre_fc_norm_* / norm(final) は据置。
- dtype: **attn q/k/v/o・fc・moe router gate・shared expert 一式・norm = F16 plain**。
  **experts のみ 4bit gs=32**(main と同形式、[256,...] へ stack: MTPHead.swift:43-45)。
- attn 形状: numHeads 16 / numKVHeads 2 / headDim 256 / ropeDim 64 / ropeBase 1e7 —
  **main full-attn 層と同一**(attnLayerRows のデフォルトと一致、要 threading 無し)。
  q_proj は [16*2*256, H](q+gate の qd2 形式)、sigmoid gating 込み(main attn と同じ)。

## KV 設計(★MLX hybrid からの意図的差分 — 必ずこの通りに)

- 旧 m2c loop(accept .506-.829 の実測根拠)は draft を `cache: mtpKV` で呼ぶ
  (Speculative.swift:44)= **履歴 attention あり**。現行 Tell hybrid(Tell.swift:420)は
  feed との二重書き回避で `cache: nil`(履歴なし)に落ちている。
- raw は KV-read draft を採る: **draft step = 現 position の k/v を計算し、kCache[0..<len]+自分
  に対して sdpa、ただし cache へは commit しない**(write_kv 後 len を戻す、あるいは no-write)。
  buffer 内容の残骸は無害(len が唯一の真実)。
- KV への書き込みは **feed API のみ**が行う(Step 5 で mtpFeedPlan の row-map に接続):
  `func feedPairs(hRows: <[n,H] 供給元>, toks: [Int32])` — n pair を position len.. に ingest
  (batch 可、prefill ingest と feed-after-commit の共通経路)。
- position/RoPE: pair index がそのまま position(prefill で P-1 pair → position 0..P-2)。
  len カウンタが管理。maxSeqLen は main と同じ上限を確保。
- rollback 不要(head-sync 規律: committed pair のみ ingest、notes/15 §head-sync)。

## 実装構成(全て追加、既存 encode 経路に触らない)

1. **新 kernel `fmm_rows`**(唯一の新規 Metal kernel): plain F16 matmul rows
   out[M,N] = x[M,K] @ Wᵀ、W は [N,K] F16 行 major(safetensors のまま)。qmm4(per-row GEMV,
   RawMetalForward.swift:256 idiom)に倣う。encoder `encodeFmmRows(enc, w, x, out, M, K, N)`。
   fc / q/k/v/o / router gate / shared expert(gate,up,down) / shared_expert_gate 全てに共用。
2. **`RawMTPWeights`**: mtp.safetensors → MTLBuffers loader(+1 norm shift を load 時に適用、
   experts stack は MLX で組んでから mtlBuf 化で可)。embed/lm_head/finalNorm 系は既存 HeadBufs
   相当を共有(RawFusedVerify.swift:3456 付近 attachHead の材料と同じ store key)。
3. **`RawMTPHead`(RawFusedVerify 内 or 新 file RawMTPHead.swift)**:
   - `func draftArgmax(hPrevBuf: MTLBuffer, hPrevRow: Int, tok: Int32) -> Int?` —
     単一 CB で上記数学を encode。hPrev は normedBuffer(Step 2 accessor)の row を直 bind
     (CPU readback 無し)。readback は draft int32 のみ。
   - `func feedPairs(hBuf: MTLBuffer, rowRange: Range<Int>, toks: [Int32])` — batch ingest
     (M-row で attn を causal に流し k/v を write_kv、logits 不要なので lm_head skip)。
   - 内部 encoder は既存を最大再利用: encodeRmsNormRows / rope rows / sdpa / write_kv /
     extract_q / sigmoid_mul / resid_add / encodeMoERouteRows / gather(gqmm4_swiglu_rows) /
     combine / swiglu。attn と shared だけ F16 なので encodeFmmRows 差し替えの mirror encoder
     (`encodeMTPAttn` / `encodeMTPShared`)を新設(既存 encodeLayer は変更禁止)。
4. **検証 runner `QWISP_RUN=mtp-raw-validate`**(driver が実モデルで回す、locked 外):
   実 mtp 重み+実 hidden で raw draftArgmax vs MLX `MTPHead()` argmax を L 位置比較
   (MTPHeadValidation.swift:93 のハーネス流用、prefill ingest → 逐次 draft)。
   出力: `[mtp-raw] argmax L/L OK` 形式。

## API 契約(★2026-07-08 改訂 — driver が production 側に stub を固定済、変更禁止)

**incident 記録**: 初回 wave で test author が `RawMTPHead()`(重み注入なし)の API を lock し、
実装側が「test の RNG seed を再現して重みを再生成し MLX ops で参照計算を実行する」
`_draftArgmaxTestMode` で green を偽装した(feedPairs はカウンタ加算のみ)。正直な実装が
構造的に不可能な API が根本原因。対策=重みは **MTLBuffer/MLXArray で注入**(seed トリック死)、
KV は**内容が draft 出力に影響する assert**(カウンタ偽装死)。

固定 API(RawFusedVerify.swift `RawMTPHead`、driver が stub 済):
- `struct WeightsSpec` — geometry + 全重み MLXArray(F16 plain 群 / q4 triple 群 /
  **RECOVERED norm**=+1 shift は loader 責務でありspec には復元済み値を渡す)+
  `expertGroupSize`(synthetic=64=suite 統一、production mtp experts=32 → threading 必須)。
- `init?(spec: WeightsSpec)` — MTLBuffers+pipelines+KV buffers 構築。
- `draftArgmax(hPrevBuf:hPrevRow:tok:) -> Int?` — KV[0..<len]+自分に attention(READ-ONLY)。
- `feedPairs(hBuf:rowRange:toks:) -> Bool` — 唯一の KV writer、len を n 進める。
- `var len: Int`。

## Gates(locked tests、synthetic・モデル不要、RAWTESTS 73→75)

- **T-fmm(test 73, 済・lock 済扱いで変更禁止)**: fmm_rows vs MLX matmul、rel ≤ 1e-3 +
  M-invariance bit 一致。
- **T-head(test 74 再設計)**: 実形状 synthetic(H=2048, V=256, E=16, Ktop=8, I=512,
  nH=16, nKV=2, hD=256, rD=64, base 1e7 — suite の test-23 と同じ「実形状 synthetic」idiom、
  gather/sdpa kernel の形状制約を満たす)。重みを WeightsSpec で注入。
  **参照 = production MLX クラスの合成**(MLX 素 op 手組みでなく): `AttentionLayer`(.plain 射影,
  同 geometry)+ `MoEBlock`(expertBits 4, expertGroupSize 64, 同重み)+ `ModelHead.embed` +
  `Proj.quantized` を MTPHead.callWithHidden:73-90 の合成順で呼ぶ — rope/GQA/KV 規約ズレを
  構造的に排除。assert:
  (a) len=0 draft: raw argmax == 参照 argmax。
  (b) **feedPairs で 2 pair ingest 後の draft**: 参照側は同 pair を `cache: KVCache()` で
      ingest してから draft(履歴 3 key の attention)— raw argmax == 参照 argmax。
      ★これが KV 内容の実効性 assert(len カウンタだけでは絶対に通らない)。
  (c) draft 直後の再 draft が同値 + len 不変(READ-ONLY 証明)。
- **T-kv(test 75 再設計)**: 同一重みで head A=batch feed(rowRange 0..<3) vs
  head B=逐次 feed(1 行×3 回) → 両者の draftArgmax 一致 + len==3 一致
  (rope position の実書き込みを強制。MLX 参照不要、raw-vs-raw)。
- 既存 73 tests green 維持(挙動変更ゼロの証明)。

## Non-goals(このループでやるな)

- runSpecLoop / Tell への配線(Step 4)。mtpFeedPlan 接続(Step 5)。
- 速度最適化(M>1 CPU-overhead 削り等)。fuse flag 群との相互作用。
- MLX hybrid(QWISP_MTP_DRAFT)側の cache:nil 変更 — MLX 側は触らない。
- width>1 / D2。streaming tier(C<nE)対応。

## 参照 file:line

- MTPHead.swift:18-59(重み key/+1 shift/形状)、:73-90(forward 数学)、:93-125(validation)。
- Speculative.swift:35-70(KV-read draft の意味論=旧 m2c)。
- RawFusedVerify.swift:3491(stepArgmax=CB 構成の手本)、:2173-2208(fused attn encode)、
  :3587-3601(Step 2 の hiddenRows/normedBuffer)、:1699-1703(shared expert encode 例=qmm 版)。
- RawMetalForward.swift:256(qmm4 kernel idiom)、:1064(write_kv)、:1479(sdpa)、:1628(rope)。
- RawVerifyTests.swift:22(bitEqual)、:1131/1285(synthetic fused 構築例)。
