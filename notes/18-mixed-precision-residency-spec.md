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
   effectively DONE; regeneration script archived: `oracle/requant_experts_2bit.py`
   (naive + cal modes). Remaining: a load-time shape/gs sanity check.
   **Calibrated artifact** (2026-07-18): `~/.mtplx/models/qwisp-experts-2bit-cal/` — same
   360 tensor names/shapes/dtypes (header-identical), MSE-optimal affine fit
   (mixprec.py#cal2bit, sweep-validated 0/4 loops at K4∈{8,0}); the fitted (q,s,b) are
   packed directly (a `mx.quantize` re-encode would min/max-refit and distort groups not
   spanning all 4 codes). Verified: L7/E123 gate MSE −31.0% vs naive.
2. **Kernel precedent**: `gqmm3`/`gqmm3Rows` (in `SeedlessMetalForward.swift`; the notes/11
   3-bit UD-tier product spec is RETIRED 2026-07-18 — mixed residency supersedes that
   lower-RAM direction, owner decision; the gqmm3 kernel + its locked tests remain as
   shipped reference, spec text lives in git history) —
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
- Arena layout per proj: **weight buffers SPLIT per class** — w4 `[K4, …]` + w2 `[M, …]`
  (as shipped in the W1b kernels: separate buffer args, w2 indexed by slot−K4); scales/biases
  stay ONE uniform buffer `[K4+M, …]` (identical slice size both pools) ⇒ 12 gather buffers
  + K4 constant. (Earlier contiguous-single-buffer idea superseded by the kernel contract.)
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

**Sim-measurement caveat (2026-07-17)**: an MLX `gather_qmm` bits=2-vs-4 microbench at
production shapes reads r≈0.92–1.0 — INVALID as a proxy: MLX gather at these shapes runs at
~3% of the byte roofline (M=1 gate 364 µs for ~4 MiB), i.e. launch/overhead-bound — the very
reason the raw engine exists. The meaningful pre-wiring speed sim is the **gqmm2_rows kernel
prototype benched against gqmm4_rows** (persistent buffers, single-CB GPU timestamps, the
gatherBench pattern — note: the `QWISP_RUN=raw-gather-bench` dispatch was pruned from
qwisp-poc during productization; re-wire it or bench via a RAWTESTS-style entry). That is
W1's front half — the speed sim and the first implementation step are the same work. Final
tok/s only from the 8GB device sim (QWISP_DEVICE_RAM=8) after W1+W2.

**Kernel sim measured (2026-07-17, `QWISP_RUN=gqmm2-bench`, W1 landed)**: on the M1 Max
(400 GB/s) dev box, gqmm2_rows vs gqmm4_rows reads **r = 0.91–1.0** at all production
shapes/M. Mechanism: at M≥16 the 4-bit kernel sits AT the DRAM roof (measured 376–378 GB/s
effective on weight bytes) while the 2-bit kernel moves half the bytes in the same time —
i.e. qd2/qd4 have near-identical instruction streams (same load count, same FMA-per-value),
so the 2-bit kernel is **value/issue-bound**, and on a Max-class part both walls coincide.
Consequence for the product: on the 8GB TARGET tier (base chips, ~100 GB/s) the 4-bit
kernel's 350+ GB/s DRAM demand is 3.5× oversubscribed ⇒ deeply byte-bound ⇒ the 2-bit
issue-limit time stays well below the DRAM time and **r→~0.56 applies there** — the +20–40%
estimate survives, but ONLY on low-BW devices. On Pro/Max-class machines mixed residency
buys ~no kernel speed (it still buys the LOOPY fix + RAM). Exactly the right shape: the
tier that needs the fix is the tier that gets the speedup. Confirm on-device in W4.

### W5 (deferred consideration, owner-requested 2026-07-17) — bolt-draft + batched strict verify = bit-exact "strict-turbo" on partial tiers

Not part of this project's acceptance; recorded for after W1–W4.

**Construction**: draft = the mixed-residency bolt (2.4 GB arena, io≈0, resident-class speed);
verifier = strict streaming (C=128); verify pipeline = the SHIPPED SuffixSpec batched verify
(f32 canonical, union-overflow guard + exact safe-prefix, seqMultiToken GDN exactness) with
the draft SOURCE swapped/augmented: suffix hit → suffix draft (free), else bolt free-run
draft. Committed tokens are always the verifier's argmax ⇒ **output is bit-exact strict L1**
— this is a strict-tier speedup, not a bolt variant (naming: strict-turbo, not bolt).

**Why the arithmetic works only on partial tiers**: draft-model spec is dead on 32GB+
resident (latency-bound ⇒ t_draft ≈ 0.9·t_target ⇒ ≤1.1×; SuffixSpec/MTP-D1 already own the
cheap-draft slots) and irrelevant where strict is resident. On 16–24GB the draft runs
resident-class (~6 ms/tok) while the target pays per-token miss-IO + per-layer sync; batched
verify amortizes both by the accept-run length. Measured acceptance a = TF-match 87–92%
(sweep) / p14 background flip 6–16% ⇒ E[run] ≈ 8–12 tok/verify ⇒ ~8 ms/tok ≈ 120 tok/s-class
bit-exact strict on prose where SuffixSpec is weak (est. 2–3× the nl streaming cell); hybrid
draft keeps code/agentic monotone.

**Soundness / LOOPY**: lever B died on GATING unsoundness (p14: flips at bolt-margin 8–16 —
no gate covers them); here every token is verified, soundness by construction. LOOPY is
structurally eliminated in this tier: context only ever contains strict tokens, so the
polluted-prefix mechanism (p14's shared-greedy loop) never forms — bolt drift is cut every
8–12 tokens.

**RAM**: 8GB can't hold both arenas (solo mixed bolt remains the 8GB answer); 16GB ≈ 15 GB
borderline (shrink strict C to fit — measure); 24GB comfortable; 32GB+ pointless.

**De-risk order**: (1) oracle acceptance at coverage-108 + deep-tail buddy (the sweep was
full-coverage; real draft a is somewhat lower — this sets the true run length); (2) after
W1–W3, a small W5 spike: swap the draft source only, verifier untouched (Prohibition 3
safe); (3) decision bar: 16/24GB nl cell must significantly beat today's strict streaming;
code cell non-regression via hybrid draft.

## W4 results (2026-07-18, dev box M1 Max, QWISP_DEVICE_RAM=8 sim, evidence: scripts/loopy/w4/)

LOOPY battery (4 canonical prompts, 1000–1500 tok, token-level detlag2, same build both arms):

| prompt | generic bolt C=64 | mixed K4=8/M2=100 (coverage 108) |
|---|---|---|
| story | loop @185 (p16) | **clean** (coherent prose) |
| tcp | loop @173 (p11) | loop @194 (p25) — residual |
| qs | loop @570 (p8) | **clean** (correct code) |
| sky | loop @87 (p1) | **clean, natural EOS** |

**0/4 → 3/4.** The tcp residual is capacity-consistent: the C-spectrum measured tcp as the
heaviest prompt (first healthy at C=128) and coverage 108 < 128 is unreachable in the same
bytes (corrected RAM math above). Generic reproduces the p14 ground truth exactly (sky@87).

TF-fidelity (QWISP_TF_REPLAY through strict, full-length maxSeqLen): mixed clean streams
**story 78.1% / qs 86.7%** ≈ oracle full-coverage (82.0/90.4) minus ~4pt of deep-tail buddy
— exactly the expected product-vs-oracle gap. Loop-containing streams read inflated
(mixed-tcp 98.2%, generic 94.7–96.1%) because in-loop positions match at 98%+ (p14).

Speed: mixed ≈ generic wall-time parity on the M1 Max (value-bound, as predicted by
gqmm2-bench); the +20–40% low-BW claim remains unverifiable on this box — needs a base-chip
device or a BW-throttle sim.

Bugs found (neither mixed-specific): (a) flaky SIGSEGV/SIGTRAP at `qwisp chat` process exit
in BOTH arms — MLX scheduler teardown race (`get_default_stream` on destructed singleton)
against the detached decode thread's tail; file separately. (b) QWISP_TOK_DUMP not written
on early-natural-EOS runs (mixed-sky verdict taken from text). (c) TF-replay pitfall: the
replay chat invocation must pass `--max-tokens ≥ stream length` or the KV allocation
truncates and predictions collapse to ~10% match (cost one debugging round — documented).

**Verdict: GO as opt-in.** QWISP_MIXED=1 turns 8GB bolt LOOPY from an asymptotic fate
(4/4, survival 87–1244 tok) into a residual risk on the heaviest prompts (1/4), at
oracle-predicted fidelity. Default-ON decision deferred (owner): candidates for closing the
tcp residual = rolling-refresh dynamics tuning, calibrated 2-bit tail (quality upside),
or accepting + documenting with the LoopGuard opt-in as belt-and-braces.

## Lever A (2026-07-18): gs=128 calibrated tail ⇒ coverage 128 — DEFAULT ON (with LoopGuard)

Phase A(a) results superseding the residual above (evidence `scripts/loopy/w4c/`):

- **cal gs=64 artifact** (`qwisp-experts-2bit-cal`, MSE −31% vs naive): TF up but loops
  MOVED, not fixed — cov 108 story@893/sky@1234 (2/4), cov 115 tcp@240/story-p83 (2/4).
  Confirms the loop driver is COVERAGE (deep-tail buddy), not tail precision.
- **cal gs=128 artifact** (`qwisp-experts-2bit-cal128`): halved scales/biases ⇒ slot
  960→864 KiB ⇒ **M2 = 110592/864 = exactly 128 at K4=0** — the measured C=128 healthy
  line in the same C=64 byte budget. Calibration pays for the coarser grouping
  (cal-gs128 MSE still −29.6% vs naive-gs64; oracle 0/4, TF parity with cal-gs64).
- **Product battery cov 128 (canonical 4): 4/4 CLEAN** — detlag2 0/4, long-period tail
  scan clean to p≤300, no internal transient loop (span ≥24). TF strict-replay
  84.6/82.1/86.3/90.1.
- **Soak (10 diverse runs incl. 2×3000 tok): 7/10 clean, 3 loops** — gc p2@~894,
  bash p18@~1400, story3000 **p262**@~949 (long-period semantic cycle). Verdict
  correction: cov 128 makes loops RARE AND LATE (vs generic bolt 4/4 @87–1244), not
  impossible — LOOPY stays the asymptotic fate of approximate decode; coverage moves
  the horizon. The C=128 "healthy line" itself was only ever measured on the canonical
  4 prompts.
- **Shipping posture**: mixed cov128 default-ON **plus LoopGuard default-ON**
  (belt-and-braces): guard heals period ≤64 invisibly (64-token hold-back, 0 FP on
  2300 clean tokens); documented residual = long-period (>64) semantic loops, which
  would need a ~800-token stream delay to buffer — not viable for interactive
  streaming. Both env-disable-able (QWISP_MIXED=0 / QWISP_BOLT_STABILITY_GUARD=0).
- Kernel/product wiring: gqmm2_rows takes gsz; gqmm_mix kernels take gs2 (tail branch
  only; gs=128 requires K4=0 — the shared-uniform s/b layout can't mix group sizes).
  Locked test `gqmm2_gs128_mix_bitexact` (RAWTESTS 89). Detector lesson: detlag2
  PMAX=64 missed a period-83 loop (found by eye) — long-period scan is now part of
  the verdict procedure.
- **DEFAULT ON**: QWISP_MIXED defaults to 1; artifact auto-detected
  (cal128 → cal → legacy naive, or QWISP_EXPERTS_2BIT); QWISP_MIXED=0 disables;
  no artifact → generic bolt (warn only on explicit opt-in).
  QWISP_BOLT_STABILITY_GUARD defaults to 1 (see soak verdict above).

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
