import Qwispmath.NearLossless

/-!
# Quality ↔ speed trade-off for near-lossless decode (Neo 8GB slow-NAND focus)

Extends the roofline ceiling (`NearLossless.lean`) with the *quality cost* of each
near-lossless lever, to answer: **how much tok/s (and speedup) do we buy per % of accuracy
lost?**

## What is rigorous vs anchored

* **tok/s / speedup** — computed from the proven roofline `period = cf + f·io`
  (`NearLossless.tokps_le_ceiling`, `speedup_le`). Rigorous.
* **qualityPct** — token-match vs the strict-L1 (4-bit greedy) reference. The ONE measured
  anchor is buddy-substitution ≈ 98 % (M-series, `nosync-approx-improve`). Expert-drop and
  low-bit figures are literature-anchored ESTIMATES (marked). To get exact numbers you would
  measure token-match on the eval set — the framework here computes the curve from them.

## Levers (Neo, roofline params from the measured throttle test)

`cf₀ = 67.6 ms` routing+compute floor (4-bit), `io₀ = 83.9 ms` expert-IO (f=1), strict 6.6 tok/s.
* **expert-drop (top-m of 8)**: `io` scales `m/8`; `cf` held fixed (conservative). Quality falls
  with discarded gate mass.
* **buddy-substitute**: serve SSD-missed experts from a cached similar expert ⇒ `io → 0` (no
  SSD read) while keeping all 8 contributions ⇒ quality ≈ 98 %. Dominates top-m skipping.
* **low-bit (b bits)**: scales `io` and the bandwidth-bound fraction of `cf` by `b/4`.
  `cfBwFrac` = share of `cf` that is memory-bandwidth-bound (assumed 0.5; `= 1.0` reproduces
  the optimistic ceiling from `NearLossless`).
-/

namespace Qwisp.QualitySpeed

/-! ## Neo roofline instantiation (Float, for the numeric curve). -/

def cf0 : Float := 67.6        -- ms, routing+compute floor at 4-bit
def io0 : Float := 83.9        -- ms, expert-IO at f = 1
def strict : Float := 6.6      -- tok/s, current strict-lossless
def cfBwFrac : Float := 0.5    -- share of cf that is bandwidth-bound (scales with bits)

/-- A near-lossless operating point. `ioFrac` = expert-IO kept (1 strict, 0 none);
`bits` = expert precision (4 strict); `qualityPct` = token-match vs strict L1. -/
structure Op where
  name : String
  qualityPct : Float
  ioFrac : Float
  bits : Float
deriving Repr, Inhabited

def Op.cfMs (o : Op) : Float := cf0 * ((1.0 - cfBwFrac) + cfBwFrac * (o.bits / 4.0))
def Op.ioMs (o : Op) : Float := io0 * o.ioFrac * (o.bits / 4.0)
def Op.tokps (o : Op) : Float := 1000.0 / (o.cfMs + o.ioMs)
def Op.speedup (o : Op) : Float := o.tokps / strict
def Op.qualityDrop (o : Op) : Float := 100.0 - o.qualityPct
/-- Efficiency: tok/s gained per 1 % of quality lost (higher = better deal). -/
def Op.tokpsPerPct (o : Op) : Float :=
  if o.qualityDrop < 0.01 then 0.0 else (o.tokps - strict) / o.qualityDrop

/-- round to 1 decimal for readable output. -/
def r1 (x : Float) : Float := (x * 10.0).round / 10.0

/-- The Neo near-lossless trade-off ladder (increasing aggressiveness). -/
def neoLadder : List Op :=
  [ { name := "strict L1 (4-bit, all 8 experts)", qualityPct := 100.0, ioFrac := 1.0,   bits := 4.0 },
    { name := "drop smallest expert (top-7)",     qualityPct := 99.5,  ioFrac := 0.875, bits := 4.0 },  -- est
    { name := "top-6 experts",                    qualityPct := 98.0,  ioFrac := 0.75,  bits := 4.0 },  -- est
    { name := "top-4 experts",                    qualityPct := 93.0,  ioFrac := 0.5,   bits := 4.0 },  -- est
    { name := "top-2 experts",                    qualityPct := 80.0,  ioFrac := 0.25,  bits := 4.0 },  -- est
    { name := "buddy-substitute misses (io->0)",  qualityPct := 98.0,  ioFrac := 0.0,   bits := 4.0 },  -- MEASURED anchor
    { name := "buddy + 3-bit experts",            qualityPct := 95.0,  ioFrac := 0.0,   bits := 3.0 },  -- est
    { name := "buddy + 2-bit experts",            qualityPct := 83.0,  ioFrac := 0.0,   bits := 2.0 } ] -- est

-- (name, quality%, tok/s, speedup×, tok/s per %quality-lost)
#eval neoLadder.map (fun o =>
  (o.name, o.qualityPct, r1 o.tokps, r1 o.speedup, r1 o.tokpsPerPct))

-- Optimistic variant: if cf were FULLY bandwidth-bound (cfBwFrac = 1), low-bit gains more.
-- buddy 4/3/2-bit tok/s under that assumption (matches NearLossless ceiling numbers):
#eval let cf (b : Float) := cf0 * (b / 4.0)
  (r1 (1000.0 / cf 4.0), r1 (1000.0 / cf 3.0), r1 (1000.0 / cf 2.0))

end Qwisp.QualitySpeed
