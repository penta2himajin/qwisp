# notes/11 — gqmm3: bit-exact 3-bit affine MoE-gather kernel (calibrated lower-RAM tier)

Driver spec (step 1). Grounds the devloop for **product#1: calibrated 3-bit lower-RAM tier** (task #18).
Recon `wuct7rbup` = **GO**, verified against ground truth by driver (quantized.h math, config.json, safetensors dtypes) — no #16-style recon error.

## Motivation
The UD model `unsloth--Qwen3.6-35B-A3B-UD-MLX-3bit` is mixed-precision: experts **3-bit** / shared **4-bit** / attn+gate **8-bit** / embed **6-bit**. Recon #16 measured TF-fidelity **93.3%** — overturns the naive-uniform-3-bit precision-floor and makes a genuine lower-RAM near-lossless tier viable. The **only missing kernel** is a 3-bit gather-qmv (`gqmm3`); every other precision already has a bit-exact raw kernel (qmm8, qmm4/gqmm4, embed=dequantized(6)).

## Scope of THIS devloop = Stage 1 only (go/no-go for the kernel)
Stage 1 is self-contained and is "the whole go/no-go" (recon). Stages 2–4 (bits-plumbing → full-model mixed-precision forward → TF-fidelity audit) are the larger *product* cost and follow **after** Stage 1 proves the kernel bit-exact. Do not begin them in this loop.

## Ground truth (driver-verified)
- `get_pack_factor<3>()=8`, `get_bytes_per_pack<3>()=3`: **8 values → exactly 3 bytes = 24 bits**, byte-oriented, no uint32-boundary spanning. `quantized.h:17-27`.
- Real tensors (layer 0): `switch_mlp.gate_proj.weight U32 [256,512,192]` (E=256,N=512,K=2048 → 2048·3/32=192 ✓); `down_proj.weight U32 [256,2048,48]` (K=512 → 48 ✓).
- **scales/biases dtype = BF16** `[E,N,K/64]`, group_size=64, affine, `w=scale·q+bias`. VERIFIED on real safetensors. **This is the #1 correctness lever** — gqmm3 must template the scale dtype (`half | bfloat`); current gqmm4/qmm8 hardcode `half` and downcast BF16→f16 at bind (`RawMetalForward.swift:1199`).
- `block_size` for bits=3 = `values_per_thread(16)·SIMD(32)=512` — **IDENTICAL to gqmm4**. Grid `(1,N/8,Ktop)`, group `(32,2,1)`, `simd_sum` reduction, expert-offset math, fast-condition (K%512==0 & N%8==0 & gs==64) all UNCHANGED. K=2048 (gate/up), K=512 (down), N∈{512,2048} all satisfy.

## Kernel design (port of gqmm4 :1120-1191; 4 deltas)
`gqmm3` = `gqmm4` with the unpack swapped. Constants: `packs_per_thread=2, pack_factor=8, bytes_per_pack=3, values_per_thread=16, block_size=512, scale_step_per_thread=4`, `in_vec_size_w = K·3/8` (bytes), `in_vec_size_g = K/64`.

1. **Per-thread weight offset**: `simd_lid * packs_per_thread * bytes_per_pack = simd_lid*6` (gqmm4: `*8`).
2. **Per-block advance**: `ws += block_size*3/8 = 192` bytes (gqmm4: `256`).
3. **`ld16_b3`** (port `load_vector<bits=3>` quantized.h:47-61) — pre-divide x over 16 values in two `i+=8` blocks. Per 8: `xt[0]=x0; xt[1]=x1/8; xt[2]=x2/64; xt[3]=x3/2; xt[4]=x4/16; xt[5]=x5/128; xt[6]=x6/4; xt[7]=x7/32`. `sum=Σx`.
4. **`qd3`** (port `qdot<3>` quantized.h:213-232) — uint8 reads `w0,w1,w2`, exact add order incl. the two straddle `*256.0f` terms:
   ```
   accum += (w0&0x07)*xt0;  accum += (w0&0x38)*xt1;  accum += (w0&0xc0)*xt2;  accum += (w1&0x01)*(xt2*256.0f);
   accum += (w1&0x0e)*xt3;  accum += (w1&0x70)*xt4;  accum += (w1&0x80)*xt5;  accum += (w2&0x03)*(xt5*256.0f);
   accum += (w2&0x1c)*xt6;  accum += (w2&0xe0)*xt7;
   return scale*accum + sum*bias;
   ```
   Loop `values_per_thread/8 = 2` packs with MLX cumulative-advance (`x_thread += 8*i; w += 3*i`).

The masked-unshifted-weight × pre-scaled-x trick + straddle `*256.0f` + `simd_sum` tree are copied verbatim from MLX → bit-exact, exactly as gqmm4 reached rel 0.000e0. Only cross-byte values are v2 (b0/b1) and v5 (b1/b2).

**`gqmm3Rows`**: mirror `gatherQmmRows` (:3926) M=1-per-row loop (`lhsPerExpert`). No gemm/qmm reordering — reduction order per row = M=1 order, so no M>1 reconciliation risk (contrast the L verify NO-GO).

## Locked test — `gqmm3_rows_bitexact` (RawVerifyTests, write-locked)
Model on `gqmm4_rows_bitexact` (:159) + `bitEqual` (max|Δ|==0 over f32, :21). MLX bits:3 API proven in-repo (`ExpertBitBench`).
1. `E=64, K=2048, N=512, Ktop=4`. `wf=randn([E,N,K])`; `(wq,sc,bi)=MLX.quantized(wf, groupSize:64, bits:3, mode:.affine)`.
2. **Oracle = MLX** (NOT self-loop): `MLX.quantizedMatmul(xm, wq[e], scales, biases, transpose:true, groupSize:64, bits:3)` per selected expert row → pins gqmm3 to MLX.
3. Compare `gqmm3` (M=1) and `gqmm3Rows` (M∈{1,2,9,17,25}, per-row inds pool) → `bitEqual == 0`.
4. **TWO dtype cases**: f16 scales AND **bf16 scales** (`wf=bfloat16`). The bf16 case forces the kernel to template scale-dtype = real-model fidelity. **Mandatory** — without it the kernel can pass f16 yet diverge on the BF16 model.

## Acceptance (driver audit, step 5)
- RAWTESTS: existing 54/54 unchanged + `gqmm3_rows_bitexact` PASS (both dtypes, all M).
- `gqmm3` byte-identical to MLX bits:3 oracle at max|Δ|==0.
- No regression to gqmm4/qmm8 paths (they are untouched; gqmm3 is additive).
- **Out of scope for this loop** (Stage 2-4, separate continuation): config.json per-tensor bits → WeightStore map; route `switch_mlp.*`→gqmm3; switch attn/gate raw paths bits:4→8; embed 6-bit; end-to-end UD-model logits; reproduce 93.3% TF fidelity.

## Risk register
- BF16 scales (handled: bf16 locked-test case + templated dtype). — the real correctness lever.
- Stage 3 (mixed-precision end-to-end wiring) is larger than the kernel and where residual risk lives — deferred.
- No M>1 tiling risk (per-row M=1 loop).
