import Qwispmath.NearLossless

/-!
# Beyond buddy-substitute on slow-NAND (Neo): can we go faster than 14.8 tok/s?

Buddy-substitute drives `io → 0`, so it sits at the compute-bound ceiling `1/cf` (≈14.8 tok/s,
`NearLossless.tokps_le_ceiling`). To beat it we must attack `cf` itself, or produce more than
one token per forward. Candidates (all *estimates* / 試算 — assumptions labeled):

* **A. buddy + speculation (SuffixSpec / MTP).** The big lever. On Neo, speculation was
  previously neutral because the expert-union grew → more SSD-IO → cancelled the accept gain
  (`bb39p2ye4`). **Buddy removes that IO**, so speculation finally pays: a batched verify of
  `K` draft tokens reads the resident experts once (amortized) and emits `accept` tokens.
  `tokps = accept / (cf · (amort + (1-amort)·K))` with io=0.
* **B. buddy + kernel-fusion.** LOSSLESS: fuse the per-layer dispatch (single-thread kernels
  were 63% of GPU-exec on M-series, `nl-encode-bound-fusion`) → `cf ·= fuse` (≈0.8). No quality
  cost — stacks on everything.
* **C. buddy + low-bit experts.** `cf` bandwidth-part scales `bits/4` (`QualitySpeed`).
* **D. stack A+B+C.**

Key unknown was `amort`: the fraction of a forward amortized across batched verify positions.
**MEASURED** (2026-07-01, `QWISP_RUN=forward-cost`, hot-pinned = io excluded, 4-bit 35B-A3B,
this dev machine, 2 runs): marginal compute `b ≈ 5.2 ms/token` (rock-stable: L=24 per-token
= 5.19 & 5.17 across runs), fixed/dispatch cost `a ≈ 28–66 ms` (dominates; load-sensitive;
50 ms historically on M1 Max). ⇒ **`amort = a/(a+b) ≈ 0.85`** (dispatch-bound forward,
batching amortizes ≥84%). Neo estimate `≈ 0.75` (weaker A18 GPU ⇒ larger marginal `b`).

**Also measured (reproducible): a small-batch penalty.** Per-token barely improves at K=2
(~40 ms) / K=4 (~27 ms); amortization only engages at **K ≥ 8** (per-token → 9.6 → 5.2). So
speculation must use draft depth K ≥ 8 to pay off — small-K spec ≈ greedy. This is exactly why
low-accept **nl (can't fill K≥8) gets ~nothing** from spec, while high-accept code/agentic does.
-/

namespace Qwisp.BeyondBuddy

/-! ## Proven bound: speculation headroom is `K/(cf·amort)`. -/
section Bound
variable {F : Type*} [Field F] [LinearOrder F] [IsStrictOrderedRing F]

/-- Speculative throughput with io=0 (buddy): `accept` tokens per batched verify of `K` drafts,
whose cost is `cf·(amort + (1-amort)·K)` (amortized part fixed, compute part scales with K). -/
def specTokps (cf accept K amort : F) : F := accept / (cf * (amort + (1 - amort) * K))

/-- **Speculation ceiling.** With `accept ≤ K`, `amort ∈ (0,1]`, throughput is at most
`K/(cf·amort)`. As `amort → 1` (compute fully amortized) this is `K·(1/cf)` = `K×` the buddy
speed; the gap to it is exactly the un-amortized per-draft compute. -/
theorem specTokps_le (cf accept K amort : F)
    (hcf : 0 < cf) (ham : 0 < amort) (hamle : amort ≤ 1) (hK : 0 < K) (hacc : accept ≤ K) :
    specTokps cf accept K amort ≤ K / (cf * amort) := by
  have hcomp : (0 : F) ≤ (1 - amort) * K := mul_nonneg (by linarith) (le_of_lt hK)
  have hden : 0 < cf * (amort + (1 - amort) * K) := by
    apply mul_pos hcf; nlinarith
  have hden2 : 0 < cf * amort := mul_pos hcf ham
  unfold specTokps
  rw [div_le_div_iff₀ hden hden2]
  nlinarith [mul_le_mul_of_nonneg_right hacc (le_of_lt hden2),
             mul_nonneg (le_of_lt hK) (mul_nonneg (le_of_lt hcf) hcomp)]

end Bound

/-! ## Neo numeric estimates (Float). -/

def cf0 : Float := 67.6         -- ms, routing+compute floor (4-bit), from measured decomposition
def strict : Float := 6.6       -- tok/s strict-lossless
def buddy : Float := 14.8       -- tok/s buddy-substitute (io→0, ≈98% quality)
def amort : Float := 0.75       -- Neo estimate (dev MEASURED ≈0.85; weaker A18 GPU ⇒ ~0.75)
def cfBwFrac : Float := 0.5     -- share of cf that is bandwidth-bound (low-bit lever)
def fuse : Float := 0.8         -- kernel-fusion cf multiplier (lossless)

/-- cf after low-bit (bits) and optional fusion. -/
def cfMs (bits : Float) (withFuse : Bool) : Float :=
  cf0 * ((1.0 - cfBwFrac) + cfBwFrac * (bits / 4.0)) * (if withFuse then fuse else 1.0)

/-- Buddy+speculation throughput (io=0). -/
def specT (bits : Float) (withFuse : Bool) (accept K : Float) : Float :=
  accept * 1000.0 / (cfMs bits withFuse * (amort + (1.0 - amort) * K))

def r1 (x : Float) : Float := (x * 10.0).round / 10.0

structure Method where
  name : String
  qualityPct : Float
  tokps : Float
deriving Repr, Inhabited

/-- K≥8 required (measured small-batch penalty). code/agentic sustain K=8 (accept≈6);
nl (accept≈1.8) cannot fill K≥8 ⇒ sits in the small-batch zone ⇒ spec ≈ buddy (no gain). -/
def methods : List Method :=
  [ { name := "strict L1 (baseline)",                  qualityPct := 100.0, tokps := strict },
    { name := "buddy-substitute (io->0)",              qualityPct := 98.0,  tokps := buddy },
    { name := "B: buddy + fusion (LOSSLESS cf cut)",   qualityPct := 98.0,  tokps := 1000.0 / cfMs 4.0 true },
    { name := "A: buddy + spec [code, acc6/K8]",       qualityPct := 98.0,  tokps := specT 4.0 false 6.0 8.0 },
    { name := "A: buddy + spec [nl, acc1.8/K8 -> penalty]", qualityPct := 98.0, tokps := specT 4.0 false 1.8 8.0 },
    { name := "A+B: buddy + spec + fusion [code]",     qualityPct := 98.0,  tokps := specT 4.0 true 6.0 8.0 },
    { name := "A+B+C: +3-bit [code]",                  qualityPct := 95.0,  tokps := specT 3.0 true 6.0 8.0 },
    { name := "aggressive: +2-bit spec [agentic acc7/K8]", qualityPct := 83.0, tokps := specT 2.0 true 7.0 8.0 } ]

-- (name, quality%, tok/s, × over strict, × over buddy)
#eval methods.map (fun m =>
  (m.name, m.qualityPct, r1 m.tokps, r1 (m.tokps / strict), r1 (m.tokps / buddy)))

-- Speculation headroom ceiling K/(cf·amort) [tok/s] for K = 4, 8 (amort=0.6 vs ideal 1.0):
#eval let ceil (K a : Float) := K * 1000.0 / (cf0 * a)
  ( ("K=4 amort0.6", r1 (ceil 4.0 0.6)), ("K=4 amort1.0", r1 (ceil 4.0 1.0)),
    ("K=8 amort0.6", r1 (ceil 8.0 0.6)), ("K=8 amort1.0", r1 (ceil 8.0 1.0)) )

end Qwisp.BeyondBuddy
