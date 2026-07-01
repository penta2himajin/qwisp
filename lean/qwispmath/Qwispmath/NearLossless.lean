import Mathlib

/-!
# Near-lossless (L3) throughput ceiling — roofline model + per-environment numbers

Companion to the strict-lossless limit proofs (`Qwispmath.lean`). Here we ask the *other*
question: if Qwisp relaxes to **near-lossless** (L3 — skip / substitute / predict experts,
or drop bits), what is the **theoretical upper bound** on decode tok/s, per RAM/SSD tier?

## Model (roofline, per token)

For batch=1 MoE decode the per-token wall time splits into two parts:

* `cf` — the **routing+compute floor**: attention, shared/gate matmuls, kernel dispatch, and
  crucially the MoE routing that must finish *before* the correct experts can be fetched.
  This is IO-*irreducible* (you cannot prefetch an expert you have not yet routed to), so it
  is exactly the part near-lossless expert-IO tricks cannot remove.
* `io` — the **expert streaming time**: (missed-expert bytes)/bandwidth. This is what
  near-lossless *reduces*, by a factor `f ∈ [0,1]` (f=1 strict, f→0 = every expert served
  from cache/substitute with no SSD read).

`period cf io f = cf + f·io`, and `tokps = 1/period`.

## What Lean proves here

* `tokps_le_ceiling` — **however aggressive (any f ≥ 0), throughput never exceeds `1/cf`,
  the IO-free compute-bound speed.** The near-lossless ceiling *is* the compute floor.
* `speedup_le` — the speedup over strict is **Amdahl-bounded by `1 + io/cf`** (the IO
  fraction). No expert-skipping scheme can do better.
* `period_antitone` — more aggressive (smaller f) never slower.

The `#eval`s at the bottom instantiate this with measured numbers and print, per tier,
[current strict tok/s → near-lossless ceiling → max speedup].
-/

namespace Qwisp.NearLossless

/-! ## Proven roofline bounds (over any `LinearOrderedField`, hence ℝ and ℚ). -/
section Model
variable {K : Type*} [Field K] [LinearOrder K] [IsStrictOrderedRing K]

/-- Per-token wall time: irreducible floor `cf` plus expert-IO `io` kept at fraction `f`. -/
def period (cf io f : K) : K := cf + f * io

/-- Decode throughput (tokens per unit time). -/
def tokps (cf io f : K) : K := 1 / period cf io f

/-- **Near-lossless ceiling.** For any aggressiveness `f ≥ 0`, throughput is at most the
IO-free compute-bound speed `1/cf`. Expert-IO reduction cannot beat the routing+compute
floor — the ceiling of near-lossless *is* the resident/compute-bound speed. -/
theorem tokps_le_ceiling (cf io f : K) (hcf : 0 < cf) (hio : 0 ≤ io) (hf : 0 ≤ f) :
    tokps cf io f ≤ 1 / cf := by
  unfold tokps period
  exact one_div_le_one_div_of_le hcf (by nlinarith [mul_nonneg hf hio])

/-- Near-lossless speedup over strict = the time ratio `period_strict / period_nl`
(equivalently the throughput ratio `tokps_nl / tokps_strict`). -/
def speedup (cf io f : K) : K := period cf io 1 / period cf io f

/-- **Amdahl speedup bound.** Speedup over strict (`f = 1`) is at most `1 + io/cf`, i.e. it is
bounded by the current IO fraction. Attained only in the limit `f → 0`. -/
theorem speedup_le (cf io f : K) (hcf : 0 < cf) (hio : 0 ≤ io) (hf : 0 ≤ f) :
    speedup cf io f ≤ (cf + io) / cf := by
  have hfio : (0 : K) ≤ f * io := mul_nonneg hf hio
  unfold speedup period
  rw [one_mul]
  gcongr
  linarith

/-- The Amdahl bound in transparent form: `(cf + io)/cf = 1 + io/cf`. -/
theorem maxSpeedup_eq (cf io : K) (hcf : cf ≠ 0) : (cf + io) / cf = 1 + io / cf := by
  rw [add_div, div_self hcf]

/-- More aggressive near-lossless (smaller `f`) is never slower. -/
theorem period_antitone (cf io f₁ f₂ : K) (hio : 0 ≤ io) (h : f₁ ≤ f₂) :
    period cf io f₁ ≤ period cf io f₂ := by
  unfold period; gcongr

end Model

/-! ## Numeric instantiation (Float, for readable decimals).

Inputs, all per-token, mix regime. `ioMs` = expert-streaming time NOT hidden by overlap.
Sources: Neo = measured under the slow-NAND throttle emulation (125.79 MB/tok ÷ 1.5 GB/s
≈ 83.9 ms; strict 6.6 tok/s). Fast-SSD / resident tiers: IO is prefetch-hidden / cached
(`ioMs ≈ 0`) per the greedy-ceiling profiling — those tiers are dispatch/speculation-bound.
16GB C=128 = measured 132 tok/s. 24/32GB are the same resident regime (io≈0). -/

structure Env where
  name : String
  strictTokps : Float   -- current strict-lossless (measured)
  ioMs : Float          -- per-token non-hidden expert-IO (ms); 0 when IO is free
deriving Repr, Inhabited

/-- Irreducible routing+compute floor, ms (serial decomposition: total − io). -/
def Env.cfMs (e : Env) : Float := 1000.0 / e.strictTokps - e.ioMs
/-- Near-lossless ceiling: expert-IO fully eliminated (f→0). -/
def Env.nlCeilTokps (e : Env) : Float := 1000.0 / e.cfMs
/-- Max speedup = 1 + io/cf (Amdahl). -/
def Env.maxSpeedup (e : Env) : Float := e.nlCeilTokps / e.strictTokps

def envs : List Env :=
  [ { name := "Neo 8GB slow-NAND (1.5 GB/s)", strictTokps := 6.6,  ioMs := 83.9 },
    { name := "8GB fast-SSD (IO prefetch-hidden)", strictTokps := 88.0, ioMs := 0.0 },
    { name := "16GB C=128 (resident-ish)",   strictTokps := 132.0, ioMs := 0.0 },
    { name := "24GB C=192 (resident)",       strictTokps := 138.0, ioMs := 0.0 },
    { name := "32GB C=256 no-sync (resident)", strictTokps := 145.0, ioMs := 0.0 } ]

/-- Optimistic add-on lever: low-bit experts (4→`bits`) scale the *bandwidth-bound* part of
the compute floor. Upper-upper bound (assumes compute fully BW-bound; real is lower because
kernel dispatch does not shrink with bits). Applied on top of the IO-free ceiling. -/
def Env.lowbitCeilTokps (e : Env) (bits : Float) : Float :=
  e.nlCeilTokps * (4.0 / bits)

-- Per-tier: (name, strict tok/s, near-lossless IO-ceiling, max speedup ×)
#eval envs.map (fun e =>
  (e.name, e.strictTokps, e.nlCeilTokps.toUInt32.toNat, e.maxSpeedup))

-- Neo, with low-bit stacked on IO elimination (optimistic): 4-bit vs 3-bit vs 2-bit
#eval let neo := envs.head!
  (neo.nlCeilTokps, neo.lowbitCeilTokps 3.0, neo.lowbitCeilTokps 2.0)

end Qwisp.NearLossless
