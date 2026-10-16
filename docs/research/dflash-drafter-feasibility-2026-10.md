# DFlash drafter for Ornith-1.5-35B-A3B on qwisp: feasibility (2026-10)

Date: 2026-10-16. Target machine: MacBook Pro M3 Pro (14-core GPU, 36 GB), Ornith-1.5-35B-A3B MLX
4-bit, resident tier. No GPU work was run for this note (another process held the GPU). Every
speed figure below that is not quoted from a source is an **estimate** and is labelled as one.

## 要約（日本語）

- 公式 drafter `ornith-ai/Ornith-1.5-35B-A3B-DFlash` を `~/models/Ornith-1.5-35B-A3B-DFlash` に取得した
  （model.safetensors 736 MiB、合計 738 MB、BF16、0.386B params）。config.json は z-lab の
  Qwen3.6-35B-A3B-DFlash とバイト単位で同一で、重みだけが異なる。Ornith のカードには受理長・速度の数値が無い。
- Ornith/Qwen3.6-35B-A3B 向けの DFlash2 は HF 上に存在する（公式は `incoai/Qwen3.6-35B-A3B-DFlash2`、
  Ornith 向けはコミュニティ製の `jzinno/...-DFlash2` と `DaoCloud/...-DFlash2-2.6B-A0.3B`）。ornith-ai と
  z-lab の名前空間には 35B-A3B 用の DFlash2 は無い。
- #98 は 2026-07-19 に「M=1 greedy では chain に勝てない」として park され、2026-08-01 に NOT_PLANNED で
  close された。当初の blocker（#90 seqMT-free verify）は #90 の Step 0 で「消すべき税が存在しない」と
  測定され、既に解消済み。close の本当の理由は、drafter のコスト（M1 Max で 13.4 ms/block）と、
  partial reject のたびに走る rebuild forward。
- M3 Pro での推定（推定値）: bandwidth-bound な verify のコストは expert union のバイト比
  （M=8 で 2.95 倍）でほぼ決まる。#98 の設計をそのまま移植すると code で 0.8〜1.0 倍となり負ける。
  drafter の 4-bit 化と GDN tape-replay による rebuild の撤廃を両方入れると、block 8 で code 1.2〜1.45 倍、
  agentic 約 1.55 倍、shortnl は gate で chain に戻すことになる。block 16 は損益分岐付近、block 4 は
  block 8 とほぼ同等。
- 「bandwidth-bound 機なら推奨」という判定は半分だけ正しい。bandwidth-bound な機械どうしでは利得は
  ほぼ同じで、端末間の差を生むのは (1) M=8〜16 の verify が compute-bound に転じるかどうか、
  (2) step あたりの固定オーバーヘッド、(3) RAM tier（streaming では不可）の 3 つ。判定には、実測の
  r_B から求める損益分岐受理長 T* を DeviceCalibration に持たせるのがよい。
- 実測（§8、oMLX 0.7.0、M3 Pro）: 3 種の drafter のうち速くなったのは agentic だけだった
  （incoai の DFlash2、block 8、4-bit で 56.8 → 68.8 tok/s、+21%）。code と shortnl は 5〜13% 遅くなり、
  1 サイクルあたりの出力トークン数は 2.3〜2.9 にとどまった（agentic だけ 3.8〜4.0）。block 16 は block 8
  より長く受理されなかった。block を 4 に縮めると code +15%、agentic +18%、longctx +13% になり、
  shortnl だけが 10〜13% 遅いまま残った。drafter は 4-bit（group 64）が最速で、8-bit と受理は変わらない。
  したがって drafter の有効化と深さは、端末単位ではなく、要求ごとの受理の実測で切り替える必要がある。

## 1. Download and availability

### 1.1 Downloaded: `ornith-ai/Ornith-1.5-35B-A3B-DFlash`

`hf download ornith-ai/Ornith-1.5-35B-A3B-DFlash --local-dir ~/models/Ornith-1.5-35B-A3B-DFlash`
succeeded (repo sha `942d20fee84d04c9ecbf18e9c6fc4144914a1d5d`).

| File | Size |
|---|---|
| `model.safetensors` | 736.1 MiB (771,819,680 B), sha256 `b9d55100f43e1a2bd56e287ac11309086ccb827ae5d6f546dc9197c2c75b34f6` (matches the HF LFS hash) |
| `config.json` | 1.2 KB |
| `README.md` | 6.0 KB |
| `assets/` | logo + `ornith_35b_eval.png` (target-model benchmarks only) |
| total | 738 MB on disk |

Tensor inventory (safetensors header): 385,906,176 parameters, all BF16. `fc.weight [2048, 16384]`,
`hidden_norm`, `norm`, and six `layers.N` (q/k/v/o, q_norm/k_norm, gate/up/down, two RMSNorms). There
is **no `embed_tokens` and no `lm_head`**: the drafter borrows both from the target.

`config.json` is byte-identical to `z-lab/Qwen3.6-35B-A3B-DFlash`'s config (`diff` empty, both
1,222 B), while the weights differ (z-lab sha256 `1fb90ef5…`, Ornith `b9d55100…`). So this is the
z-lab architecture retrained (or fine-tuned) against the Ornith target.

### 1.2 Other drafters for this architecture (not downloaded)

HF API search (`/api/models?search=DFlash2`, `search=DFlash`, 2026-10-16). Downloads are the API's
rolling count.

| Repo | Kind | Downloads | Likes | Created | Notes |
|---|---|---:|---:|---|---|
| `ornith-ai/Ornith-1.5-35B-A3B-DFlash` | DFlash v1, official | 288,130 | 29 | 2026-09-20 | downloaded |
| `z-lab/Qwen3.6-35B-A3B-DFlash` | DFlash v1, official (Qwen3.6 target) | 26,473 | 296 | 2026-04-17 | used by #98 |
| `incoai/Qwen3.6-35B-A3B-DFlash2` | DFlash2, official (Inco AI) | 3,541 | 12 | 2026-09-17 | Qwen3.6 base target, block 8 |
| `mlx-community/Qwen3.6-35B-A3B-DFlash2-4bit` | DFlash2, MLX 4-bit re-pack of Inco's Splash draft | 2,849 | 10 | 2026-09-23 | "not published or endorsed by Inco AI" |
| `jzinno/Ornith-1.5-35B-A3B-DFlash2` | DFlash2, community, Ornith target | 1,110 | 7 | 2026-08-23 | warm-started from `z-lab/Qwen3.5-35B-A3B-DFlash` |
| `DaoCloud/Ornith-1.5-35B-A3B-DFlash2-2.6B-A0.3B` | DFlash2, MoE drafter (2.6B total / 0.3B active) | 228 | 2 | 2026-09-07 | taps `[3,12,21,30,39]`, 3 layers |
| `DaoCloud/Ornith-1.5-35B-A3B-DFlash2-2.6B-A0.3B-NVFP4` | same, NVFP4 | 352 | 1 | 2026-09-10 | |
| `Cobra91310/Ornith-1.5-35B-A3B-DFlash2-draft-GGUF`, `3ntr0py-t4m3r/Ornith-1.5-DFlash2-Draft-GGUF`, `aminya/Qwen3.6-35B-A3B-DFlash2-GGUF`, `risk-magician/…-AMX_Q8R.amxq` | GGUF/other re-packs | 343–1,408 | 0–4 | 2026-09/10 | |
| `z-lab/Qwen3.5-35B-A3B-DFlash` | DFlash v1 (Qwen3.5) | 3,416 | 47 | 2026-03-02 | |

`ornith-ai/Ornith-1.5-35B-A3B-DFlash2` and `z-lab/Qwen3.6-35B-A3B-DFlash2` do not exist (the API
returns the not-found/unauthorised error). z-lab's DFlash2 collection currently lists only
Muse-Glimmer-30B and Qwen3.8-27B ([z-lab/dflash README](https://github.com/z-lab/dflash)).

## 2. The drafter: architecture and published numbers

### 2.1 Architecture (from `config.json`, the tensor header, and the #98 recon of z-lab's `model.py`)

| Item | Value |
|---|---|
| Class | `DFlashDraftModel` (`auto_map: dflash.DFlashDraftModel`, `trust_remote_code`), `model_type: qwen3` |
| Layers | 6: `layer_types` = 5 × `sliding_attention` + 1 × `full_attention`, `sliding_window: 4096` |
| Hidden | 2048; MLP `intermediate_size` 6144 (SiLU, gate/up/down) |
| Attention | 32 query heads, 8 KV heads, `head_dim` 128, q_norm/k_norm, RoPE θ = 1e7, no bias |
| Block | `dflash_config.block_size: 16` (trained). Serving examples on the card use 8 |
| Target taps | `target_layer_ids: [1, 6, 11, 16, 22, 27, 32, 37]` of `num_target_layers: 40` → 8 × 2048 = 16384 features per token → `fc [2048, 16384]` → `hidden_norm` |
| Mask token | `mask_token_id: 248077` (the first padded row: the Ornith tokenizer defines ids 0–248076, the embedding has 248,320 rows) |
| Vocab | 248,320; `tie_word_embeddings: false`; embedding and LM head come from the **target** |
| dtype | `bfloat16` |
| EOS | 248044 (`<|endoftext|>`) |

How a block is drafted, per the #98 recon of `z-lab/dflash` `dflash_generate` and the oMLX
implementation (§4): one drafter forward over `[anchor, mask × (B−1)]`. The anchor is the last
committed token. Embeddings come from the target's `embed_tokens`. Each drafter layer computes K/V
over `[fc(ctx) ‖ noise rows]`: the projected target features of committed tokens enter every
layer's KV, not just a prefix. Attention is non-causal inside the block. The drafter KV for the
context rows is cached across blocks. Logits come from the target's `lm_head`, and each position
takes its argmax. There is a single denoising step and no iteration.

### 2.2 Numbers on the Ornith drafter card

The Ornith card publishes **no acceptance length and no speedup** for the drafter. What it says:

- Serving: vLLM `"num_speculative_tokens":8`; SGLang `--speculative-dflash-block-size 8`; "The
  examples above use a speculative block size / token count of `8`."
- Sampling: "For general tasks: `temperature=0.6`, `top_p=0.95`, `top_k=20`"; "To reproduce the
  reported benchmarks: `temperature=1.0`". The only reported benchmarks are target-model quality
  scores in `assets/ornith_35b_eval.png` (e.g. SWE-bench Verified 79, TB2.1 68.5), not drafter
  speed.
- License MIT (z-lab's Qwen3.6 drafter is Apache-2.0). Runtimes: "Transformers ≥ 5.8.1", "vLLM ≥
  0.20.2 (tested with 0.28.0)", "SGLang = 0.5.18".

### 2.3 Numbers from the sibling cards (same architecture)

**`z-lab/Qwen3.6-35B-A3B-DFlash`** (same config, Qwen3.6 target). Setup, verbatim: "Runtime:
SGLang on 1x NVIDIA B200 GPU, tensor parallel size 1, `bfloat16`", "Decoding: greedy, thinking
enabled, max output length 4096 tokens", "Accept length: `completion_tokens / spec_verify_ct` per
generation turn". Mean accept length at concurrency 1:

| Workload | MTP steps=3 | DFlash block=4 | MTP steps=7 | DFlash block=8 | MTP steps=15 | DFlash block=16 |
| --- | --- | --- | --- | --- | --- | --- |
| gsm8k | 3.474 | 3.455 | 5.288 | 5.404 | 6.453 | 6.954 |
| math500 | 3.559 | 3.553 | 5.522 | 5.743 | 6.840 | 7.629 |
| humaneval | 3.369 | 3.459 | 4.952 | 5.352 | 5.878 | 6.797 |
| mbpp | 3.280 | 3.358 | 4.613 | 4.973 | 5.292 | 6.052 |
| mt-bench | 3.135 | 3.075 | 4.365 | 4.326 | 5.061 | 5.118 |

Throughput, concurrency 1 (B200): DFlash block=8 gave 2.32x (mt-bench) to 3.27x (math500), and
block=16 gave 2.24x to 3.61x. "DFlash reaches up to `3.61x` speedup at concurrency 1 and `2.89x` at
concurrency 32." At concurrency 32, block=8 is fastest on 4/5 workloads (2.03x–2.86x). "Block size
`8` is the recommended default for higher-concurrency serving. Block size `16` gives longer accept
lengths and strong concurrency-1 throughput in most workloads."

**`incoai/Qwen3.6-35B-A3B-DFlash2`** (DFlash 2: per-position top-k candidates plus a path
selector, two-tap dynamic convolutions; config `block_size 8`, all 6 layers sliding with window
2048, `selector_top_k 16`, `selector_rank 256`). Setup: "SGLang on one NVIDIA GB300", "Speculation
block size: 8 (7 draft tokens per verification step)", "temperature 1.0, top-p 0.95, top-k 20 …
with thinking enabled". Acceptance length (MTP vs DFlash 2): GSM8K 4.92 vs **5.21**, MATH-500 5.02
vs **5.34**, HumanEval 4.44 vs **4.90**, MBPP 4.36 vs **4.76**, MT-Bench 4.26 vs **4.41**.
Concurrency-1 speedup: 1.53x (MT-Bench) to 2.09x (MATH-500).

**`jzinno/Ornith-1.5-35B-A3B-DFlash2`** (community, Ornith target). Setup: "40 prompts … 10 each
from chat, code, math, and STEM", "greedy sampling, and up to 512 output tokens", "The target is the
official Ornith NVFP4 checkpoint", "Both external drafts propose up to 10 tokens". Results at
concurrency 1: DGX Spark (GB10), AR 79.3 tok/s, Ornith DFlash2 114.2 tok/s with accepted length 4.02,
"44.0% above target-only serving". H200: AR 253.6 tok/s, DFlash2 472.9 tok/s with accepted length
3.98. The unadapted Qwen3.5 DFlash warm start reached 3.48 (Spark) and 3.51 (H200). Ornith's NEXTN
head reached 1.92–1.93.

**`DaoCloud/Ornith-1.5-35B-A3B-DFlash2-2.6B-A0.3B`** (MoE drafter). Setup: vLLM, H200, concurrency 1,
BF16 target, "`temperature=0.6`, `top_p=0.95`, `top_k=20`", 7 draft tokens. Selected accept lengths
(DFlash2 / native MTP): GSM8K 4.880 / 3.844, HumanEval 4.905 / 4.032, MBPP 4.201 / 3.786,
LiveCodeBench 3.777 / 3.554, MT-Bench 3.155 / 3.133, SpeedBench/writing 2.572 / 2.730,
SpeedBench/roleplay 2.250 / 2.355. Its per-position acceptance (e.g. GSM8K p0 0.872 … p6 0.299)
gives an empirical decay profile used qualitatively in §5.

### 2.4 qwisp's own measured acceptance (#98, M1 Max, z-lab Qwen3.6 drafter, quantized target, block 8)

From the #98 bootstrap-gate comment (PR #116): accept/step without/with prefill-ctx bootstrap:
code 3.19 → 4.00, agentic 3.59 → 4.54, longctx 4.78 → 5.33, shortnl 1.02 → 0.95. As accept
lengths T = accept/step + 1, this gives code 4.19/5.00, agentic 4.59/5.54, longctx 5.78/6.33, and
shortnl 2.02/1.95. These are the most qwisp-relevant T values: they come from the quantized target
and qwisp's own regimes. They were measured with the Qwen3.6 drafter on a Qwen3.6 target. The Ornith
pair has not been measured.

## 3. Why #98 was closed, and whether its blocker still applies

Sources: `gh issue view 98 --comments`, `gh issue view 90 --comments`, archived branch
`archived-experiments/dflash`, `docs/research/inference-optimization-survey-2026-10.md` §4.1.

**Timeline.**

1. **Opened 2026-07-18** as "hard-blocked on seqMT-free verify only" (#90). The premise was that with
   today's verify "each extra row sequentializes → K=8 verify costs ~O(K) forwards". The sketch assumed
   r_M ≈ 1.0–1.1 and projected "~3.5x on draftless spans".
2. **#90 Step 0 (PR #107, M1 Max) removed the blocker by measurement.** r_2 = 1.26–1.29, r_4 =
   1.85–1.97, r_9 = 3.41, r_17 = 5.88–5.98, and the "cpu-gap is flat (~1.5-5ms, noise-level) across
   M". The GDN state path costs "≤0.1 forward-units for ALL rows". Verdict: "there is no
   sequentialization tax to remove". **#90 is CLOSED/COMPLETED (2026-07-18)** as a measured NO-GO on
   the planned surgery. The r_M curve became "the canonical M>1 verify-cost input". So the
   blocker named in #98's title no longer applies: it never existed as a removable cost.
3. **Recon and re-measure (PR #108).** On stepArgmax including lm_head: r_2 1.27, r_4 2.01, r_9 3.51,
   r_17 6.14, so lm_head amortises like the rest. The re-projection was "~1.35–1.54x on draftless
   spans" at block 8.
4. **Phase A1 (PRs #115, #116, M1 Max).** A raw drafter (`dflash_fmm16`, 110 → 13.4 ms/block), a fused
   draft+verify command buffer, and a prefill-ctx bootstrap were built, with lossless 128/128 × 4
   regimes. Measured tok/s with bootstrap vs flag-off chain: code 62.5 vs 82.7, agentic 62.7 vs 76.0,
   longctx 62.6 vs 90.5, shortnl 26.7 vs 81.9.
5. **Parked 2026-07-19, closed NOT_PLANNED 2026-08-01, code removed from main** and archived verbatim
   on `archived-experiments/dflash` (DFlashDrafter / DFlashRawDrafter / DFlashDispatch /
   DFlashParityProbe, the tap-buffer seam, `stepArgmax(draftPrologue:)`, locked tests 97/98, and
   `oracle/dflash_model_mlx.py`).

**The actual reason for closing** is economic, not a blocker. In the issue's own corrected model,
`cost/tok = (c_draft + r_8 + P_rej·r_(p̄+1) + cpu) / T`, with:

- verify M=8 ≈ 3.2 × 12.1 ms ≈ 39 ms, against an M1 Max chain of 12.1 ms/token;
- drafter 13.4 ms/block ≈ 1.1 forwards;
- a **rebuild forward on every partial reject**: `TellRuntime.runSpecLoop` rolls back one step and
  re-forwards `[u] + accepted drafts`, because the GDN ping-pong snapshot can only undo the whole
  last step (notes/03).

The conclusion was "break-even … T > ~5 AND drafter ≤ 7ms AND no rebuild — a stack of speculative
wins". Rebuild avoidance was judged "likely NO-GO" without being measured. Revival conditions were
batching (#6) or "a slower device tier where single forwards are IO-bound".

**Does it still apply on the M3 Pro?**

- The verify-cost physics carries over. §5 shows that the M1 Max r_M curve is reproduced by the
  expert-union byte ratio, which is a model property. Expect a similar r_8 ≈ 3.1–3.2 on the M3 Pro
  (estimate).
- Two of the three loss terms were never physics. (a) Drafter bytes: the archived drafter read about
  1.39 GB per block (BF16 layers plus a dequantized F16 lm_head of 622 MB). A 4-bit drafter that
  reuses the target's 4-bit head reads about 0.50 GB. (b) The rebuild forward can be replaced by GDN
  tape replay. dflash-mlx ships "Tape-replay rollback … records an innovation tape during verify and
  replays only the accepted steps", and oMLX ships "GDN state replay for Qwen3.5". Neither is
  bit-exact by claim, so qwisp would have to prove byte identity under its own kernels (§6).
- The baseline also changed. The M3 Pro chain is about 17 ms/token at about 87% of the measured copy
  ceiling (1.8 GB / 17 ms ≈ 106 GB/s vs 121–123 GB/s, commit d5ff219). That is a cleanly
  bandwidth-bound baseline, which is the case the park comment's revival condition (2) points at.
  On the M1 Max, 12.1 ms moved about 149 GB/s of a 400 GB/s part.

## 4. How others implement DFlash on Apple Silicon

### 4.1 oMLX (local tree `/Users/nara/repos/omlx` @ `958974c1`, 2026-10-15)

Files: `omlx/speculative/dflash/{__init__,drafter,variants,selection,kernels}.py`,
`omlx/patches/mlx_lm_mtp/` (verify, acceptance, rollback, depth controller),
`omlx/patches/qwen35_verify_qmm.py`, `omlx/scheduler.py`, and the design doc
`docs/experimental/dflash_mlx_integration.md`.

- **What the drafter consumes.** Hidden states of the target layers named by the drafter config
  (`capture_layer_ids` = `target_layer_ids`), captured during chunked prefill
  (`_dflash_prefill_capture`, `_dflash_seed_prefill`) and returned by every verify forward for the
  verified positions. New captures are projected (`fc` + `hidden_norm`) and written into a
  per-request **context ring** of `(n_kv, C, D)` K/V per draft layer, "with RoPE already applied, so
  slot order does not matter for the non-causal context attention". Only the newest `W − 1` positions
  are attended (`dflash_draft_window_size`, default the draft's `sliding_window`).
- **Drafting.** "drafts a block from `[anchor, mask, ..., mask]`" in one forward. Rows of a batch
  share one forward with per-row RoPE offsets. The drafter's projections run under the same
  verify-shape small-M qmm kernels (`set_verify_qmm_armed(True)`). `predraft` encodes the *next*
  block on the GPU before the host has read how many rows committed. `adopt_predraft` keeps it if
  the host commits the same count; otherwise it is discarded. Drafts can be affine-quantized at load
  (`dflash_draft_quant_*`). For DFlash2, `selection.py`/`kernels.py` implement the candidate-path
  selector as Metal kernels.
- **Verification.** One target forward through the Lightning-MTP verify path. "Greedy rows accept by
  prefix match; sampled rows use the draft distribution for Leviathan acceptance." The block is
  "clamped to the trained block and to the Lightning MTP verify limit (9)".
- **Rollback.** "Rejected positions are trimmed from the target caches (KV trim with an undo log …,
  GDN state replay for Qwen3.5)". In `patches/mlx_lm_mtp/qwen35_model.py`, the verify processes "the
  confirmed prefix and draft suffix as two chunks, snapshotting the SSM/conv state in between", stores
  `(conv_state, ssm_state)` as `rollback_state`, and on rejection `mtp_partial_rollback` restores it
  and replays the kept prefix.
- **Adaptive verify depth** (`_DepthController`, default `dflash_verify_mode=adaptive`). It picks the
  verified prefix length d maximising `score(d) = (1 + p1 + p1·p2 + …) / t_est(d)`, with t measured
  per depth and the marginal row cost taken as the measured slope, "so a fine-grained MoE on a
  bandwidth-limited chip learns its true (large) marginal within the first few cycles". Depth 0 means
  plain decode: "speculation is only profitable while the multi-token verify forward is cheap
  relative to a plain decode step … (… MoE expert loads scale with verify tokens) … measured 0.67x on
  gemma4 26B story/16k with the best depth choice."
- **Apple measurements in the tree.** None for DFlash throughput. The verify-qmm header notes "measured
  on Qwen3.6-27B/M3 Ultra the lm_head scales linearly (1.2ms at M=1 → 4.3ms at M=4)". Its kernels are
  sg8 for 4–8 rows and a simdgroup-MMA `mma` tile for 7–24 rows that "dequantizes 16-column weight
  tiles into threadgroup memory once and multiplies them against every row", with "fp32 accumulation
  … differs from stock qmm at bf16 tail-ULP level".

### 4.2 mlx-lm and mlx-vlm

- **mlx-lm 0.32.0** (scratch venv): `grep -ri dflash mlx_lm` finds nothing. mlx-lm has no DFlash.
- **mlx-vlm** (not installed locally, so this is from the README only): "DFlash, DFlash2, and DSpark"
  via `--draft-model`, described as "typically 2–3× faster". For DFlash2: "For the fastest quantized
  setup, convert the drafter to 4-bit; the verifier adapts between three and five rows from recent
  acceptance". oMLX loads draft checkpoints "through mlx-vlm's drafter classes (DFlash v1, DFlash2,
  Laguna, Muse Glimmer assistant)".
- **z-lab/dflash MLX backend** (README): "For quantized targets or drafts, use `block_size <= 5`: MLX's
  current quantized matmul kernel becomes less efficient at larger verify widths."

### 4.3 dflash-mlx and Splash (the two Apple-specific engines with numbers)

- **dflash-mlx** (`bstnxbt/dflash-mlx`): "Block-diffusion draft generates 16 tokens in one pass. Target
  verifies in one pass." Features: tape-replay rollback, verify-specialised int4 qmm with an M=16
  path ("M5-class … NAX M=16 path …; older Apple GPUs keep the steel simdgroup-MMA fallbacks"), and an
  adaptive verify policy. On "Lossless", it says: "Output can still differ from pure AR because of MLX
  dispatch divergence". Benchmarks: "Apple M5 Max, 64 GB …, MLX 0.31.1 … stock
  `mlx_lm.stream_generate` baseline vs DFlash … 3 repeats, median", one math prompt.
  Qwen3.6-35B-A3B-4bit: 1024 tok 138.26 → 300.33 tok/s (**2.20x**, acceptance 91.02%), 2048 tok
  1.82x, 4096 tok 1.56x, 8192 tok **1.33x**. The speedup decays with length on the MoE but not on the
  dense 27B (2.78–3.06x). That fits an attention/verify cost that grows with context on the MoE
  target.
- **Splash** (`incoai/splash`, Inco AI's own Metal engine with DFlash2, "Apple M3 or newer"):
  Qwen3.6-35B-A3B decode 175 tok/s on an M5 Pro (20-core) and **209 tok/s on an M3 Max (40-core)**,
  against llama.cpp at 69 and 66 ("SPEED-Bench coding prompts … greedy"). The M5 Pro launch table
  gives "Decode · short prompt 210 tok/s (1.7×)" against the next-fastest engine. Splash is **not
  bit-exact**: "Same token ranked first: Splash and llama.cpp … 97.83–98.14%" on the 35B.

## 5. Feasibility for qwisp on the M3 Pro (estimates)

Everything in this section is an **estimate** from a bandwidth model calibrated on the M1 Max r_M
measurements. Script: scratchpad `est/est.py`. Its arithmetic is reproduced below.

### 5.1 Bytes per verify row: the expert-union model

From the Ornith 4-bit safetensors headers: routed experts 18,119 MB (256 experts × 40 layers →
**1.769 MB per expert per layer, 70.8 MB per expert-slot across 40 layers**), GDN 570 MB, lm_head
286 MB, attention 153 MB, shared expert 71 MB, gates and norms 23 MB. The GDN f32 recurrent state is
30 layers × 32 × 128 × 128 × 4 B = 63 MB, read and written, so about 126 MB.

- Fixed per forward: 1103 + 126 ≈ **1229 MB**. Routed at M=1: 8 × 70.8 = 566 MB. Total
  **≈ 1795 MB/token**, which matches the ≈ 1.8 GB/token observed.
- With M rows and independent uniform top-8-of-256 routing, the expected distinct experts per layer
  are U(M) = 256·(1 − (1 − 8/256)^M): U(2) = 15.8, U(4) = 30.5, U(8) = 57.4, U(9) = 63.6,
  U(16) = 102.0, U(17) = 106.8.
- r_bytes(M) = (1229 + U(M)·70.8) / 1795.

| M | bytes (MB) | r_bytes | M1 Max measured r_M (#90/#108) | residual (measured / bytes) |
|---|---:|---:|---:|---:|
| 2 | 2344 | 1.31 | 1.26–1.29 | 0.98 |
| 4 | 3390 | 1.89 | 1.85–2.01 | 1.01 |
| 8 | 5293 | 2.95 | (3.2 interpolated) | — |
| 9 | 5732 | 3.19 | 3.41 (3.51 stepArgmax) | 1.07 |
| 16 | 8446 | 4.70 | — | — |
| 17 | 8786 | 4.89 | 5.88–5.98 (6.14) | 1.21 |

**Finding.** The measured verify cost is the expert-union byte ratio to within 0–7% up to M=9. The
excess at M=17 (+21%) is per-row compute: attention, GDN, and dequant arithmetic. Verify is
therefore bandwidth-bound, and its cost ratio is a property of the *model's routing*, not of the
device. Arithmetic intensity: about 5.8 GFLOP per row (≈ 2.9 B active params incl. lm_head) gives
about 3.3 FLOP/B at M=1, about 8.8 FLOP/B at M=8 (46 GFLOP / 5.29 GB), and about 11 FLOP/B at M=16.
The M3 Pro's nominal ridge is about 5 TFLOPS / 122 GB/s ≈ 40 FLOP/B. The effective ridge for
4-bit dequant small-M kernels is lower, so M=16 is where compute starts to show.

For the M3 Pro, the residual is applied at 1.0x and 1.5x of the M1 Max excess. The M3 Pro has less
compute per byte of achieved bandwidth than the M1 Max did. Result: **r_4 ≈ 1.94–1.97, r_8 ≈
3.13–3.22, r_16 ≈ 5.6–6.1** (estimates). Real routing is temporally correlated, which would lower U.
The M1 Max fit suggests the uniform model is not far off.

**Long-context caveat.** Ornith's attention layers have 2 KV heads × 256 dim. At 16K context, one
row-pass over KV is 2 × 256 × 16384 × 2 (K,V) × 2 B × 10 layers ≈ 335 MB. If verify rows do not
share KV loads, M=8 adds about 2.3 GB (+≈ 21 ms) per verify at 16K. The split `sdpa_rows_part`
shares K/V across GQA heads, and whether it shares across rows needs checking. This matches
dflash-mlx's MoE speedup decaying from 2.20x at 1K to 1.33x at 8K.

### 5.2 Drafter cost per block

Weights read once per block (rows ≤ 16 do not change the bytes):

- Layers: 6 × (q 4096×2048 + k, v 1024×2048 each + o 2048×4096 + 3 × 6144×2048) = 6 × 58.7 M params
  → **704 MB BF16**, or **198 MB 4-bit** (g64 affine, 0.5625 B/param).
- `fc` 2048×16384: 67 MB BF16 or 19 MB 4-bit. It applies only to newly committed ctx rows, but the
  weight is still read once per block.
- LM head: the target's 4-bit `lm_head`, 286 MB. The archived build used a dequantized F16 copy of
  622 MB.
- Drafter ctx attention: ≤ 4096 rows × 8 KV × 128 × 2 × 2 B × 6 layers ≈ 50 MB, which is small.

| Variant | Bytes/block | ms at ~106–110 GB/s | in forwards (F = 17 ms) |
|---|---:|---:|---:|
| archived (BF16 layers + F16 head) | ~1.39 GB | ~12.6–13 | **~0.74** |
| BF16 layers + target 4-bit head | ~1.06 GB | ~9.6–10 | ~0.57 |
| 4-bit layers + 4-bit fc + target 4-bit head | ~0.50 GB | ~4.6–5 | **~0.27** |

The archived build achieved about 104 GB/s on the M1 Max for 1.39 GB in 13.4 ms, so these rates are
not optimistic. Drafter quantization can reduce acceptance. oMLX, mlx-vlm and z-lab all recommend
or allow 4-bit drafts; the effect on the Ornith drafter is unmeasured.

### 5.3 Step cost model and the estimates

Per verify step with block B (M = B rows: anchor + B−1 drafts):

```
t_step = c_draft + r_B·F + Σ_{p<B−1} P(p)·ρ(p) + o          tokens/step = T (accept length incl. bonus)
speedup vs chain = T·F / t_step                               (F = 17 ms ≈ chain ms/token on M3 Pro)
ρ(p) = r_(p+1)·F with the rebuild forward (today), or ≈ 0.06F with GDN tape replay (estimate)
```

P(p) uses a constant per-position acceptance α solved from T, with T = (1 − α^B)/(1 − α). That gives
P_rej = 1 − α^(B−1), which is 0.65 at T = 5.0 with B = 8. o = 0.03F (one fused CB, int32 readback).

**Calibration check against the M1 Max measurement.** Code with bootstrap, T = 5.0, c_draft = 1.1F,
rebuild on: the model predicts 0.93x, while #98 measured 62.5/82.7 = **0.76x**. The model is about
20% optimistic. About 14 ms/step on the M1 Max is not explained by drafter + verify + rebuild:
CB/readback latency, non-geometric (bimodal) acceptance, and chain disabled while DFlash is active.
The table below therefore also gives a pessimistic overhead of o = 0.5F.

Speedups (estimates; r_M residual 1.0x and 1.5x give the low–high ranges, o = 0.03F unless stated):

| Scenario | Block | T source | T | Speedup (o=0.03F) | Speedup (o=0.5F) |
|---|---|---|---:|---:|---:|
| S0: archived design ported (drafter 0.74F, rebuild fwd) | 8 | qwisp code+boot | 5.00 | 0.97–0.99x | ~0.90x |
| S0 | 8 | card math500 | 5.74 | 1.17–1.19x | ~1.07x |
| S0 | 16 | card math500 | 7.63 | 0.85–0.90x | ~0.85x |
| S1: 4-bit drafter (0.27F), rebuild fwd | 8 | qwisp code+boot | 5.00 | 1.07–1.09x | ~0.98x |
| S2: 4-bit drafter + GDN tape replay | 4 | card mt-bench / math500 | 3.08 / 3.55 | 1.34–1.35x / 1.55–1.57x | 1.13x / 1.30x |
| S2 | 8 | qwisp shortnl | 1.95 | 0.54–0.56x (gate must fall back to chain) | — |
| S2 | 8 | qwisp code, no boot | 4.19 | 1.17–1.20x | ~1.06x |
| S2 | 8 | card mt-bench | 4.33 | 1.21–1.24x | ~1.10x |
| S2 | 8 | qwisp code+boot | 5.00 | **1.40–1.44x** (~83–85 tok/s) | ~1.27x |
| S2 | 8 | qwisp agentic+boot | 5.54 | 1.56–1.60x | ~1.41x |
| S2 | 8 | qwisp longctx+boot | 6.33 | 1.79–1.83x (before the §5.1 long-ctx caveat) | ~1.61x |
| S2 | 16 | card mt-bench / math500 | 5.12 / 7.63 | 0.80–0.86x / 1.19–1.28x | — |

Reading:

1. **The archived design loses on the M3 Pro too** (S0 ≈ 0.8–1.0x on qwisp's regimes). The #98
   verdict carries over unless both the drafter bytes and the rebuild are removed.
2. **Block 8 with a 4-bit drafter and GDN tape replay is the only configuration with a clear
   predicted win**: about 1.2–1.6x on code and agentic, about 1.06–1.4x with pessimistic overheads.
   Rebuild removal is worth more than drafter quantization (S1 → S2 adds about 0.3x at T = 5).
3. **Block 16 is break-even at best.** r_16 ≈ 5.6–6.1 eats the extra acceptance, and P_rej ≈ 0.83–0.96
   makes every residual cost bite. Block 4 is about as good as block 8 because r grows quickly.
   This argues for oMLX-style adaptive verify depth (draft 8, verify the best prefix) rather than a
   fixed block.
4. **shortnl and chat-like spans need the gate.** At T ≈ 2, DFlash runs at about half speed.
   qwisp's #119/SCORE gate (or an oMLX-style depth-0 park) has to hand those spans back to the
   chain.
5. **The deciding unknowns are T for the Ornith pair on qwisp's regimes and the M3 Pro r_8.** Both
   can be measured cheaply (§6, Phase 1). T carries the most risk: the drafter was trained on a BF16
   target, and qwisp runs a 4-bit one ("feature shift", #98 caveats).

### 5.4 GDN state and rollback

- **Today (notes/03, `TellRuntime.runSpecLoop`).** Verify runs as one `stepArgmax` over M rows. The
  GDN conv history and recurrent state are ping-pong buffers, and attention KV is a length.
  `backend.rollback(snap)` undoes exactly one step (KV length rewind plus ping-pong swap, "コピー無し、
  直前 1 step 限定"). On a partial reject the loop then calls
  `backend.forward([u] + drafts.prefix(p))` to rebuild the state at row p. That rebuild is a full
  M = p+1 forward (r_(p+1)·F): about 1.5–2.0F at block 8, paid on 55–80% of steps.
- **Proposed: GDN tape replay.** During the verify, record per row and per GDN layer the
  post-projection recurrence inputs (k, v, decay g, β) and the conv input row. That is about 32 KB
  per row-layer, ≈ 8 MB for 8 rows × 30 layers. On a partial reject: restore the ping-pong snapshot,
  truncate attention KV to base + p + 1, and run **the same M=1 recurrence kernel** over rows 0..p
  from the tape. The #90 measurement puts the whole GDN recurrence at 0.2–1.1 ms at any M. The
  target taps for rows 0..p come from the verify itself (causal), so no rebuild forward is needed.
- **Lossless argument (has to be proven, not assumed).** qwisp's strict contract already makes each
  verify row byte-identical to an M=1 decode row. The tape therefore holds the exact inputs the M=1
  path would have produced. Replaying them through the M=1 recurrence and conv kernels in the same
  order gives a byte-identical state by construction, if the replay uses the identical kernel and
  rounding sequence. This is the "per-row snapshot" idea that #98 marked "likely NO-GO" without a
  measurement. dflash-mlx and oMLX ship it, in non-bit-exact form.

## 6. Implementation plan for qwisp

Constraints from CLAUDE.md: Python first for physical plausibility; the forward path is frozen
(additive, flag-off byte-identical seams only, which is what #98's seams were); RAWTESTS must stay
green; locked tests are only extended.

**Phase 1: plausibility, all measurement, no engine change** (run standalone; one GPU process at a
time).

1. *Accept transfer (Python/MLX oracle, the main go/no-go input).* Revive
   `oracle/dflash_model_mlx.py` and `oracle/dflash_parity.py` from `archived-experiments/dflash`.
   Pair the Ornith 4-bit MLX target with `~/models/Ornith-1.5-35B-A3B-DFlash`, run greedy on qwisp's
   four bench regimes at block 4/8/16, with and without prefill-ctx bootstrap, and with BF16 and
   4-bit drafter weights. Record T and the per-position acceptance curve. The oracle is off the
   serving path, and T is a statistical quantity, so the MLX target is fine for this.
   Go threshold (estimate-derived): code or agentic T(block 8) ≥ 4.5 with the 4-bit drafter.
2. *r_M on the M3 Pro.* Re-run the #90 probe (`QWISP_RUN=seqmt-m` from PR #107, revive it if it is
   no longer on main) for M ∈ {1, 2, 4, 5, 8, 9, 16} at shallow and 16K context, with
   `QWISP_RESIDENCY_SET=1`. Compare with r_bytes(M) above, and check whether verify rows share KV
   loads at 16K.
3. *GDN tape-replay algebra (Python, numpy f32).* Check that replaying per-row (k, v, g, β, conv
   rows) reproduces the sequential state and that the tape budget is about 32 KB per row-layer.
   This proves the bookkeeping. Byte identity is proven later by a locked Metal test.
4. *Drafter cost.* Time a 4-bit drafter forward at M = 8 in MLX on the M3 Pro (expect about 5 ms).
   Also time the archived `dflash_fmm16` path, both with 4-bit weights and with the target's qmm4
   lm_head.

Phase 2 proceeds only if the §5.3 model fed with the measured T and r_8 predicts at least 1.2x on
code or agentic.

**Phase 2: engine (revive and extend; flag `QWISP_DFLASH=1`, default off).**

1. Cherry-pick from `archived-experiments/dflash`:
   `swift/Sources/QwispCore/{DFlashDrafter,DFlashRawDrafter,DFlashDispatch}.swift`, the tap-buffer
   seam and `stepArgmax(draftPrologue:draftTokensBuf:)` in `SeedlessFusedVerify.swift`, the bootstrap
   hook and the DFlash branch in `TellRuntime.runSpecLoop`, and locked tests 97/98, renumbered onto
   today's total, with `total = N` bumped. Default the drafter dir to
   `~/models/Ornith-1.5-35B-A3B-DFlash` (the archived default was the z-lab path under `~/.mtplx`).
2. 4-bit drafter: affine g64 quantization at load, or a pre-quantized checkpoint, through qwisp's qmm4
   rows kernels. Use the **target's 4-bit lm_head** (qmm4 rows) instead of the dequantized F16 copy.
   This saves about 0.6 GB RAM and about 3 ms per block (estimate).
3. GDN tape replay in place of the rebuild forward (§5.4). The verify CB writes the tape under the
   flag only (nil ⇒ zero encodes ⇒ byte-identical flag-off). Add a replay kernel entry that calls the
   existing M=1 recurrence/conv kernels. New WRITE-LOCKED tests: `gdn_tape_replay_equiv`
   (replay-state ≡ rebuild-forward state, byte-equal, every p ∈ [0, B−2], synthetic weights) and
   flag-off identity.
4. Adaptive verify depth: draft a block of 8, then verify the prefix k ∈ {0 (chain), 2, 4, 6, 8} that
   maximises `(1 + p1 + p1·p2 + …)/t(k)`, using measured t(k) (oMLX `_DepthController` shape). k = 0
   hands the span back to the chain, with `QWISP_CHAIN_AHEAD` intact. Keep SuffixSpec first on
   suffix-hit spans (notes/15 hybrid policy).
5. Product wiring: `qwisp pull` of the drafter (`swift/Sources/qwisp/Pull.swift`), a CLI/config flag
   (`swift/Sources/qwisp/Config.swift`), and resident tier only (a streaming verify multiplies SSD
   reads by the union).
6. Gates: RAWTESTS (new locked tests), COMPTEST/TOKTEST, lossless 128/128 × 4 regimes flag-on,
   flag-off byte-identical, then a speed G-D vs the chain per regime on the M3 Pro.

## 7. Device auto-detection: which metric discriminates

The proposed rule "bandwidth-bound machine → recommend the drafter, compute-bound → optional" needs
refinement for this MoE.

- **On every bandwidth-bound device the predicted gain is roughly the same.** If verify is
  bandwidth-bound, r_B ≈ r_bytes(B), which depends on the model's routing (§5.1) and not on GB/s.
  Ridge points on Apple GPUs sit in a narrow band (nominal ≈ 25–45 FLOP/B), above the verify
  intensity of about 9–11 FLOP/B. A bandwidth-vs-FLOP classifier would therefore label nearly every
  Mac the same way.
- **What actually varies between devices:**
  1. **κ_B = measured r_B / r_bytes(B)**, the compute residual. κ ≈ 1 means verify is
     bandwidth-bound and DFlash pays per §5.3. κ ≫ 1 means the M=8–16 verify became compute-bound
     (weak GPU, inefficient small-M dequant kernels), and then DFlash *loses*: it is not "optional"
     there. M5-class GPUs with neural accelerators push κ toward 1, which is consistent with
     dflash-mlx's 2.2x on an M5 Max.
  2. **u₁ = bytes(1) / (F · BW_copy)**, M=1 bandwidth utilisation. On the M3 Pro, u₁ ≈ 0.87 (bandwidth-bound).
     A device with low u₁ (latency- or dispatch-bound M=1, as on large Max/Ultra parts) hides extra
     rows partly behind fixed costs, which makes r_B < r_bytes(B) and DFlash *more* attractive.
  3. **Fixed per-step overhead o/F** (CB submit and readback latency relative to F). The residency-set
     work cut submit latency per 48 chain CBs from 163–232 ms to 19–24 ms (2c199cb), and
     overhead like this decides the o = 0.03F vs 0.5F columns above.
  4. **RAM tier.** Streaming and partial tiers turn the union growth into SSD reads, so DFlash is off
     there.
- **Proposed decision variable: the break-even accept length**
  `T*(B) = (c_draft + r_B·F + ρ̄ + o) / F` (in forwards), computed from measurements. Recommend DFlash
  when T*(best B) is well below the regime T (code/agentic ≈ 5 per #98).

**Hosting it in `DeviceCalibration`.** The file already states the policy ("RAM で mode は静的 gate、
cost model は実測"), and notes/02 lists "speculation 損益分岐" as a derived variable that must be
measured on the device.

1. Add `DeviceCalibration.specEconomics(engine:drafter:) -> SpecEconomics` with fields `F`, `r[M]`
   for M ∈ {1, 4, 8}, `cDraft`, `kappa8`, `u1`, `bwCopy`, `tStar[B]`. Measurement: 3 reps of
   stepArgmax at each M on a fixed real-text prompt (real routing, not synthetic tokens), one drafter
   forward, and the existing moe-roofline blit-copy ceiling. Total under 1 s, run once after model
   load. Cache it keyed by (GPU name/cores, model hash, drafter hash, qwisp version), with
   `QWISP_SPEC_ECON` to override or skip.
2. Decision at startup:
   - not resident → DFlash off;
   - resident and T*(best B) ≤ 4.0 → "recommended". If the drafter is missing, log a one-line hint
     with the `hf download` command and the predicted range; if it is present, enable it;
   - 4.0 < T* ≤ 5.0 → "optional" (code/agentic only, online gate decides);
   - T* > 5.0 (κ ≫ 1 or overhead-dominated) → "not recommended".
3. At runtime, the adaptive depth controller (Phase 2 step 4) remains the final arbiter per span.
   Calibration only sets the prior r(k) table and the startup recommendation.
4. `DeviceCalibration.describeAll()` / `qwisp` device inspector prints `u1`, `κ8`, and `T*`
   alongside the RAM tier.

## 8. Measured: oMLX 0.7.0 with three drafters on the M3 Pro (2026-10-16)

Setup: oMLX 0.7.0 (`omlx-cli serve --no-cache --memory-guard off`; the default guard aborted
the agentic request at ~22 GB), target Ornith-1.5-35B-A3B-uncensored MLX 4-bit, greedy,
`max_tokens` 384, the four qwisp bench regimes, two runs each, auto fans. Decode tok/s is
`usage.generation_tokens_per_second`; acceptance and cycles come from oMLX's
"DFlash generation complete" log line (acceptance = share of output tokens that came from
the drafter; tokens/cycle = output tokens / cycles). `dflash_verify_mode` was left at the
default ("adaptive"). Outputs are not bit-exact with the no-drafter run.

| decode tok/s | code | agentic | longctx | shortnl |
|---|---|---|---|---|
| no drafter | 59.7 / 60.0 | 56.9 / 56.7 | 53.2 / 52.4 | 59.3 / 59.4 |
| DFlash v1, ornith-ai, b16, BF16 | 59.8 / 59.3 | 43.0 / 42.0 | 45.5 / 45.5 | 43.9 / 43.9 |
| DFlash2, jzinno (Ornith), b16, 4-bit | 56.2 / 55.7 | 53.8 / 54.1 | 43.2 / 43.0 | 43.4 / 43.0 |
| DFlash2, incoai (Qwen3.6), b8, 4-bit | 56.6 / 57.0 | **69.1 / 68.5** | 55.7 / 56.1 | 52.1 / 51.6 |

| tokens/cycle | code | agentic | longctx | shortnl |
|---|---|---|---|---|
| DFlash v1, b16, BF16 | 2.84 | 2.88 | 2.80 | 2.26 |
| DFlash2 jzinno, b16, 4-bit | 2.80 | 3.84 | 2.63 | 2.37 |
| DFlash2 incoai, b8, 4-bit | 2.61 | 4.00 | 2.93 | 2.34 |

Other runs: the same three drafters in BF16 were all at or below the no-drafter row
(incoai BF16: 53.8/50.2, 55.3/64.3, 50.4/50.8, 45.2/47.7). DFlash v1 with oMLX's 4-bit draft
quantization was erratic (code 17-20 tok/s, draft phase 1.0-1.6 s per request); it was not
investigated further.

Block size and drafter quantization (incoai DFlash2, same setup, `dflash_block_size` set
explicitly; decode tok/s, mean of two runs, tokens/cycle in parentheses):

| | code | agentic | longctx | shortnl |
|---|---|---|---|---|
| no drafter | 59.9 | 56.8 | 52.8 | 59.4 |
| 4-bit g64, block 4 | **69.0** (2.93) | 67.2 (3.20) | **59.6** (2.87) | 51.4 (2.19) |
| 4-bit g64, block 6 | 64.0 (2.87) | **71.2** (3.92) | 55.4 (2.87) | 53.2 (2.36) |
| 4-bit g64, block 8 (default) | 56.8 (2.61) | 68.8 (4.00) | 55.9 (2.93) | 51.9 (2.34) |
| 8-bit g64, block 4 | 66.8 (2.93) | 66.4 (3.23) | 58.4 (2.87) | 51.2 (2.22) |
| 4-bit g32, block 4 | 68.1 (2.91) | 62.8 (3.17) | 59.0 (2.84) | 50.9 (2.19) |
| BF16, block 8 | 52.0 | 59.8 | 50.6 | 46.5 |

4-bit g64 is the fastest drafter format and does not lower tokens/cycle relative to 8-bit.
A rerun of DFlash v1 with 4-bit draft quantization (block 16) gave 60.7 / 44.3 / 43.4 / 47.0,
so the 17-20 tok/s in the first run was transient, not a property of the format.

What this shows on this machine:

1. At oMLX's default block 8 only agentic gains (+21 %). With block 4, code (+15 %), agentic
   (+18 %) and longctx (+13 %) all gain; block 6 is best for agentic (+25 %). Shortnl loses
   10-13 % at every block size (2.2-2.4 tokens/cycle). The drafter, and its depth, have to be
   gated by measured acceptance per request (or per stream window), not switched on per device.
2. Block 16 does not buy longer accepted runs than block 8 here (2.6-2.9 tokens/cycle on
   code for all three), so block 16 only adds verify and draft cost. The Qwen3.6-trained
   incoai drafter is at least as good on Ornith as the Ornith-trained jzinno one.
3. The tokens-per-cycle figures are below the τ ≈ 4-6 the drafter cards report on GPUs. With
   qwisp's measured r_M on this machine (M=4: 1.86, M=9: 3.92), a verify of the full block
   cannot pay for 2.6 tokens/cycle; verifying 3-4 rows (r ≈ 1.6-1.9) is the only shape that
   can win on code-like text, and then only narrowly. The §5.3 estimates for code (1.2-1.45×)
   are optimistic for the M3 Pro; agentic (≈ 1.55×) is consistent with the measured +21 %
   given oMLX's higher per-cycle overhead.

## Sources

- Hugging Face: [ornith-ai/Ornith-1.5-35B-A3B-DFlash](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-DFlash)
  (card, config, safetensors header; rev `942d20f`),
  [z-lab/Qwen3.6-35B-A3B-DFlash](https://huggingface.co/z-lab/Qwen3.6-35B-A3B-DFlash) (rev `f181eec`),
  [incoai/Qwen3.6-35B-A3B-DFlash2](https://huggingface.co/incoai/Qwen3.6-35B-A3B-DFlash2),
  [mlx-community/Qwen3.6-35B-A3B-DFlash2-4bit](https://huggingface.co/mlx-community/Qwen3.6-35B-A3B-DFlash2-4bit),
  [jzinno/Ornith-1.5-35B-A3B-DFlash2](https://huggingface.co/jzinno/Ornith-1.5-35B-A3B-DFlash2),
  [DaoCloud/Ornith-1.5-35B-A3B-DFlash2-2.6B-A0.3B](https://huggingface.co/DaoCloud/Ornith-1.5-35B-A3B-DFlash2-2.6B-A0.3B);
  HF API `https://huggingface.co/api/models?search=DFlash2` (2026-10-16).
- Papers and blogs: DFlash, [arXiv:2602.06036](https://arxiv.org/abs/2602.06036); [DFlash 2 blog](https://inco.ai/blog/dflash2/).
- Code and READMEs: [z-lab/dflash](https://github.com/z-lab/dflash),
  [bstnxbt/dflash-mlx](https://github.com/bstnxbt/dflash-mlx),
  [incoai/splash](https://github.com/incoai/splash) (README and `docs/performance.md`),
  [Blaizzy/mlx-vlm](https://github.com/Blaizzy/mlx-vlm) README; oMLX local tree @ `958974c1`
  (`docs/experimental/dflash_mlx_integration.md`, `omlx/speculative/dflash/*`,
  `omlx/patches/mlx_lm_mtp/{batch_generator,qwen35_model}.py`, `omlx/patches/qwen35_verify_qmm.py`);
  mlx-lm 0.32.0 site-packages (grep, no DFlash).
- qwisp: issues #98 and #90 (with comments), PRs #107/#108/#115/#116, branch `archived-experiments/dflash`,
  notes/02, notes/03, notes/04, `docs/research/inference-optimization-survey-2026-10.md`,
  `swift/Sources/QwispCore/{TellRuntime,Tell,DeviceCalibration}.swift`, commits `d5ff219`, `2c199cb`, `19fbe92`.
