# notes/18 — Mixed-precision residency (4-bit core + 2-bit tail): the #47 Part A capacity fix

Project scope. Grounds the implementation of **probe 17's lever Q** — the only live lever after
16 NO-GO decode-side probes (#47 Part A record: memory `bolt-loopy-47-parta-guard`;
`scripts/loopy/README.md` probes 12–17).

## Problem & evidence

8GB bolt (C=64) LOOPY is a **capacity wall**: per-layer routed footprint ~103 experts > 64
resident slots ⇒ buddy substitution accumulates distribution-level damage ⇒ repetition
attractors (asymptotic fate, survival 87–1244 tok; probe 14). All decode-time interventions
are closed with evidence: prediction (p12), timing (p13), correction (p14), dithering (p15),
sampling (p16).

**Probe 17 + design-point sweep (oracle, `scripts/loopy/p17/mixprec.py`)**: keeping a top-K4
core at 4-bit and requantizing the tail through a 2-bit round trip (affine gs=64) at FULL
coverage is clean — a coarser version of the RIGHT function carries no attractors
(SubstituteCorrect δ-comparison), unlike buddy (full-precision WRONG function, 5/5 burn).

| K4 | loops | 8-gram min | TF-match story/tcp/qs/sky |
|----|-------|------------|---------------------------|
| 40 | 0/4 | 0.99–1.00 | 86.9 / 90.5 / 92.8 / 91.0 % |
| 20 | 0/4 | 0.99–1.00 | 83.4 / 89.3 / 91.7 / 89.5 % |
| 8  | 0/4 | 1.00      | 82.0 / 86.7 / 90.4 / 89.1 % |
| 0  | 1/4 transient (self-escaping) | 0.94–1.00 | 79.8 / 83.8 / 89.3 / 89.7 % |

**Scope boundary: bolt tier only.** Strict L1 lossless is defined on the 4-bit quantised
greedy stream and its path (weights, kernels, refs) is untouched. Mixed residency changes the
*effective model bolt computes with* — bolt is the near-lossless tier, and its TF-fidelity
band today is ~88.7% (8GB); the sweep's TF numbers sit in the same band.

## Corrected RAM math (supersedes the README's "K4 + M/2 ≤ 64 slot units")

The slot-unit approximation counted weights only. **Scales/biases do not shrink at 2-bit**
(gs=64 both ⇒ same group count): 192 KiB/expert at either precision.

Per expert (I=512, H=2048, gs=64): weight 4-bit = 1536 KiB, 2-bit = 768 KiB;
scales+biases = 192 KiB. ⇒ **slot4 = 1728 KiB, slot2 = 960 KiB** (ratio 0.56, not 0.5).

Equal-RAM coverage on the C=64 byte budget (110,592 KiB/layer):
`M = 115.2 − 1.8·K4` ⇒ K4=40→cov 83, K4=32→89, **K4=20→99, K4=8→108**, K4=0→115.

Consequences:
- The C=128 healthy line is NOT reachable within the same bytes (max 115 at K4=0, which the
  sweep excludes on quality). Reachable band: **99–108** (K4 20→8), between the measured
  Swift capacity points C=96 (3/4 healthy) and C=128 (4/4).
- The K4/M split must be derived at runtime from the **byte** budget (DeviceCalibration
  fitC-style), not hard-coded slot units.
- Tail miss IO: weight bytes halve (scales/biases don't) ⇒ per-miss pread 1728→960 KiB
  (−44%) — also helps slow-NAND (Neo).

## Assets already in place (verified 2026-07-17)

1. **2-bit artifact exists**: `~/.mtplx/models/qwisp-experts-2bit/experts_2bit.safetensors`
   (10.0 GB, 2026-06-26; all 40 layers × 256 experts × switch_mlp {gate,up,down}×{weight U32,
   scales F16, biases F16}, affine gs=64, same tensor names/index format as the model).
   **Provenance verified bit-exact**: slice(L7,E123) == `mx.quantize(mx.dequantize(4bit), bits=2)`
   — i.e. exactly what the oracle sweep computed with. Workstream ③ (offline tail requant) is
   effectively DONE; remaining: archive the regeneration script in `oracle/`, and a load-time
   shape/gs sanity check.
2. **Kernel precedent**: `gqmm3`/`gqmm3Rows` (notes/11, in `SeedlessMetalForward.swift`) —
   the additive sub-4-bit port pattern (MLX `qdot<bits>` verbatim, bit-exact locked test
   against the MLX oracle, gqmm4 untouched) is proven. bits=2 is *simpler* than bits=3:
   pack_factor 16/u32, no byte-straddle terms.
3. **Seam**: `SeedlessFusedExpertProvider` (C / gatherBuffers / ensure) +
   `ArenaExpertProvider` → `prepareMoEBlockBufs(expertOverride:)` → `encodeMoEBlockRows`.
   Buddy/recalib/async-refresh machinery in `BoltServe` operates on slots, not bytes.
4. **ExpertSource is directory-generic**: a second instance pointed at the 2-bit dir serves
   tail preads with zero new IO code (same key prefix, same safetensors index format —
   shapes verified: gate weight [256,512,128] U32, scales [256,512,32] F16).

## Workstreams

### W1 — `gqmm2` kernel family (go/no-go gate, notes/11 Stage-1 pattern)
- `gqmm2Rows` port of `gqmm4_rows`: bits=2 ⇒ per-u32 16 values; `ld16_b2` pre-divides x by
  4^k within each byte group; `qd2` masks `(w >> 2k) & 0x3`-equivalents with the MLX
  masked-unshifted-weight × pre-scaled-x trick, exact add order from `quantized.h qdot<2>`
  (ground-truth read of quantized.h REQUIRED first, as notes/11 did — do not trust this
  paragraph's constants).
- Locked test `gqmm2_rows_bitexact` modeled on `gqmm3_rows_bitexact`: oracle = MLX
  `quantizedMatmul(bits:2)`, `bitEqual == 0`, M ∈ {1,2,9,17,25}. Scales dtype: artifact is
  F16 ⇒ f16 case mandatory (bf16 case optional symmetry).
- **Mixed dispatch design**: single new kernel family `gqmm_mix{,_swiglu,_rows,_swiglu_rows}`
  taking the 9 arena buffers + `K4` constant; remap slot `s < K4` ⇒ 4-bit offsets/qd4 path,
  else 2-bit offsets/qd2 path. The branch is uniform per tid.z (expert slot is uniform per
  threadgroup) ⇒ no simd divergence. Existing gqmm4* kernels untouched (ADDITIVE).
  Alternative if register pressure/code bloat measures badly: two partitioned dispatches
  (core rows / tail rows) with row masking — decide on measurement.

### W2 — mixed-slot arena + cache policy
- Arena layout per proj: **weight buffer = [K4 slots × slice4][M slots × slice2] contiguous**
  (one buffer, offset math in kernel); scales/biases buffers stay uniform `[K4+M, …]`
  (identical slice size both pools) ⇒ provider still returns 9 buffers + binds K4.
- `LayerExpertCache` extension (or `MixedLayerExpertCache`): slots `0..<K4` = core (pinned,
  freeze-basis top-K4 by routing frequency — same basis buildBuddyTable already uses);
  slots `K4..<K4+M` = tail LRU. Tail misses pread from the 2-bit ExpertSource.
- Core refresh: recalib freeze re-picks top-K4 ⇒ core swap rides the existing staged
  async-refresh path (slot classes fixed; only occupants change).
- Buddy table: unchanged mechanics, now only for deep-tail misses beyond coverage
  (expected rare: footprint ~103 vs coverage 99–108).

### W3 — wiring + budget
- DeviceCalibration: mixed tier for 8GB bolt — derive (K4, M) from byte budget with K4 from
  the chosen design point; env overrides `QWISP_MIX_K4` / existing `QWISP_CACHE_C` semantics
  documented. Strict tier untouched.
- BoltServe: calib/recalib freeze picks core set; miss/telemetry counters distinguish
  tail-hit / tail-miss(pread) / deep-tail(buddy).

### W4 — measurement (acceptance)
- RAWTESTS: existing 79 unchanged + `gqmm2_rows_bitexact` (+ mixed-kernel bitexact vs
  composed reference) — gate stays green every commit.
- `scripts/test_bench_batch.sh`, tokenizer/completion tests unchanged.
- **LOOPY battery**: 4 prompts × 1500 tok, 8GB sim, token-level detlag2 — target 4/4 clean
  (today's C=64 bolt: 0/4). This is the project's reason to exist.
- TF-fidelity (QWISP_TF_REPLAY): within the sweep band (≈88–92% at the chosen K4) — no worse
  than today's 8GB bolt fid 88.7.
- Speed: floor = parity with today's C=64 bolt; roofline estimate is a GAIN (below).

## Speed roofline (h measured 2026-07-17, `p17/mixh.py`)

M=1/M-row decode is weight-byte-bound on the routed gather (GEMV reads each weight once;
rows kernels read per (row, ki) pair, so verify forwards make routed-gather the dominant
byte term). Mixed bytes factor vs all-4-bit = `0.556 + 0.444·h`, where h = routed-slot share
landing in the per-layer top-K4 core. Measured on the o0 traces (4 prompts, 5000 tok):

| K4 | h calib-basis | h self-basis (≈rolling recalib) | bytes factor |
|----|---------------|---------------------------------|--------------|
| 8  | 0.079 | 0.150 | **0.59–0.62** |
| 20 | 0.187 | 0.298 | 0.64–0.69 |
| 40 | 0.343 | 0.477 | 0.71–0.77 |

Routing mass is flat (top-8 experts carry only 15% of routed slots — consistent with the
probe-7 diffuse cold set), so at K4=8 nearly all routed reads are 2-bit: **routed weight
traffic −38..41%**. With routed gather at fraction g of bolt step time, speedup ≈
1/(1−0.39g): g=0.5 → +24%, g=0.7 → +38%. Byte accounting of the verify forward suggests g
is high (M rows × 8 experts × 1728 KiB × 40 layers dominates); expect **+20–40% bolt tok/s
on fast-SSD 8GB** (today ~166), floor parity. Second-order: tail miss/refresh/B3-fetch IO
−44%/expert (main Neo slow-NAND win), fewer buddy events at coverage 108. Strict tier
unaffected. All numbers to be confirmed by W4 measurement (doctrine).

## Design point

**Primary: K4=8 / M=100 ⇒ coverage 108** — the K4=8 oracle point measured clean (0/4 loops,
8-gram min 1.00, TF mean −1.4 pt vs K4=20), and 108 is the only clean coverage that crosses
the measured per-layer footprint (~103), collapsing average-case buddy substitution to the
deep tail. Fallbacks in order: K4=20 / M=79 (coverage 99), K4=32 / M=57 (coverage 89) if the
Swift LOOPY battery disagrees. K4 stays a tunable end-to-end. Final table:
`scripts/loopy/README.md` probe-17 section.

## Risks

- **Coverage 99–108 sits between measured Swift capacity points** (96⇒3/4, 128⇒4/4) and the
  oracle measured full coverage — the Swift LOOPY battery (W4) is the real verdict, and K4
  stays tunable end-to-end.
- Core selection basis differs from oracle (rolling freeze basis vs one-shot calib
  frequency) — absorb with the battery; if unstable, pin core from a wider window.
- Mixed kernel register pressure / instruction bloat in the fused swiglu variants — measure
  early (gatherBench pattern); partitioned-dispatch fallback documented in W1.
- Naive affine 2-bit is the weakest 2-bit format (ds4 ships calibrated IQ2_XXS/Q2_K for
  routed experts) ⇒ sweep quality is a **conservative lower bound**; calibrated tail requant
  is a follow-up upside, not a dependency.
- Async refresh staged-swap must copy per-slot-class byte counts — audit
  `BoltAsyncRefresh`/staging arenas for hard-coded slice sizes.
