# qwisp-lean — formal limits of lossless MoE-decode acceleration

Lean 4 / Mathlib formalization answering: *is there a mathematically-grounded way to do
Qwisp's quantized-MoE decode with fewer operations or less weight IO at the **same strict
(L1, bit-exact) losslessness**?* (error-correction / coded computing, information geometry,
quantized-integer exact algebra were the candidate levers.)

## Verdict

**No lossless-speedup identity exists to discover for batch=1 quantized-MoE decode**, and the
key negative result is now a *theorem*, not an empirical observation. Concretely:

| # | theorem (`Qwispmath.lean`) | what it certifies | reduces |
|---|---|---|---|
| T1 | `zero_point_factor` | asymmetric-quant zero-point term factors out; the shared activation sum `∑xᵢ` is computed once across all rows & top-k experts. Exact over any `CommRing` (⇒ ℤ). | compute (marginal, real) |
| T2 | `moe_combine_linear` | routing-weighted top-k combine `∑ wₑ•(Eₑx)` equals one merged operator `(∑ wₑ•Eₑ)x`. Algebraically lossless. | nothing (merged op still needs every expert) |
| — | `int_add_assoc` / `float_add_not_assoc` | integer accumulation is associative, but IEEE-754 `Float` is **not** (`(1e20+-1e20)+1 = 1` vs `1e20+(-1e20+1) = 0`). This is *why* the exact transforms are L1 only on the integer path and merely L2 (distribution-level) on FP. | — |
| **T4** | **`read_lower_bound`** | **CAPSTONE.** No reconstructor reading only a proper subset `S ⊊ support` of the weights can compute the exact GEMV `∑ᵢ wᵢxᵢ` for all inputs — an adversary flips an unread weight. ⇒ **no lossless scheme (coding / CSE / factoring / prediction) reduces weight reads below full support.** The streaming/IO floor `Ω(N·K)` is a theorem. | nothing — proves impossibility |

**Interpretation for Qwisp.** The one exact *compute* win (T1) is marginal and already the
standard trick; the *communication* floor that dominates the slow-NAND (Neo) tier is
**provably irreducible** at strict L1. So there is nothing left to find on the exact-math
axis — batching (amortize `W` across B verify positions, already in SuffixSpec) remains the
only lossless lever, and near-lossless (L3) is a separate, quality-bounded trade. This
matches the empirically-exhausted single-stream speedup work.

The load-bearing theorems depend only on the standard trusted axioms
(`propext`, `Quot.sound`, `Classical.choice`) — no `sorry`. Only the Float *demonstration*
uses `native_decide` (necessary: `Float` is an opaque `@[extern]` op the kernel cannot reduce).

## Near-lossless (L3) throughput ceiling — `Qwispmath/NearLossless.lean`

The strict-L1 axis is closed (above). The *other* question: if Qwisp relaxes to
**near-lossless** (skip/substitute/predict experts to cut expert-IO), what is the
theoretical tok/s ceiling, per RAM/SSD tier? We formalize a roofline
`period = cf + f·io` (`cf` = the IO-irreducible routing+compute floor — you can't
prefetch an expert you haven't routed to; `io` = expert-streaming time; `f∈[0,1]` = IO kept)
and prove:

* `tokps_le_ceiling` — for any aggressiveness, throughput ≤ `1/cf`: **the near-lossless
  ceiling is exactly the IO-free compute-bound speed.**
* `speedup_le` — speedup over strict is **Amdahl-bounded by `1 + io/cf`** (the IO fraction).

Instantiated with measured numbers (`#eval`), the ceiling per tier is:

| tier | strict tok/s (now) | near-lossless ceiling | max speedup | binding constraint |
|---|---|---|---|---|
| **Neo 8GB slow-NAND (1.5 GB/s)** | 6.6 | **≈14.8** | **2.24×** | IO-bound (io≈84ms, cf≈68ms) |
| 8GB fast-SSD | 88 | 88 | 1.00× | IO already prefetch-hidden |
| 16GB C=128 | 132 | 132 | 1.00× | dispatch/speculation-bound |
| 24GB C=192 | 138 | 138 | 1.00× | dispatch-bound |
| 32GB C=256 no-sync | 145 | 145 | 1.00× | dispatch-bound |

**Finding:** near-lossless expert-IO reduction helps **only the slow-NAND Neo tier**, and even
there the ceiling is ≈**2.2×** (not 8–40×) — capped by Neo's own ~68 ms/token routing+compute
floor, which no expert-skipping can touch. Every faster-IO / resident tier is already
compute/dispatch-bound, so expert-skipping buys ≈**1.0×**. A second, quality-costlier lever —
low-bit experts (4→3→2 bit) — could stack on Neo to a (optimistic, compute-fully-BW-bound)
≈19.7 / 29.6 tok/s, but that shrinks the compute floor, not the IO, and trades more quality.

## Build / reproduce

Toolchain is managed by [mise](https://mise.jdx.dev) (`mise.toml` pins `elan`).

```sh
cd qwispmath
# elan installs lake+leantar under ~/.elan/toolchains/<ver>/bin
export PATH="$HOME/.elan/toolchains/$(sed 's#leanprover/lean4:#leanprover--lean4---#' lean-toolchain)/bin:$PATH"
lake exe cache get     # downloads prebuilt Mathlib oleans (~8600 files)
lake build Qwispmath   # compiles only Qwispmath.lean (~5s)
```

### Gotcha (cost us an hour, don't repeat)

`lake exe cache get` must find **`leantar`** in the Lean sysroot. Running it via
`mise exec -- elan run <ver> lake …` breaks sysroot detection → `leantar not found` → the
cache silently fails → `import Mathlib` falls back to **compiling all of Mathlib from source**
(1–2 h at 100 %+ CPU). Fix: put the toolchain's own `bin` (which contains both `lake` and
`leantar`) directly on `PATH`, as above. Verify with `find …/build/lib/lean/Mathlib -name '*.olean' | wc -l` → should be ~8222, not ~2700.
