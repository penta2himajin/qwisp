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

## Gates(locked tests、synthetic・モデル不要、RAWTESTS 72→+n)

- **T-fmm**: fmm_rows vs MLX `matmul(x, Wᵀ)`(f16) — M∈{1,3}、K/N 非整列(例 K=96,N=40)含む。
  bit-exact が理想だが f16 累積順で不可なら rel ≤ 1e-3 + **M-invariance(M=1 row と M=3 の
  該当 row が bit 一致)** を必須とする(order-stable 規律)。
- **T-head**: synthetic 小型 MTP 重み(H 縮小可)で raw draftArgmax vs 「MLX ops で
  MTPHead.callWithHidden の数学を組んだ参照」— argmax 一致 + hidden rel ≤ 1e-2。
  参照は production kernel の再実装でなく MLX 素 op 合成(既存 suite の参照 idiom)。
- **T-kv**: feedPairs で n pair ingest → draft 1 回 → len 不変(draft が commit しない)を
  カウンタで assert。position 進行(rope 位置)は feed 後の draft argmax が「同 pair を
  1 CB batch で ingest した場合」と一致することで検証。
- 既存 72 tests green 維持(挙動変更ゼロの証明)。

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
