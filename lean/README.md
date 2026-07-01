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
