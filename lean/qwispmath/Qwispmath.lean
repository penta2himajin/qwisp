import Mathlib
import Qwispmath.NearLossless
import Qwispmath.QualitySpeed

/-!
# Qwisp — formal certification of lossless math-acceleration limits

Research (coded-computing / info-geometry / quantized-integer algebra) converged:
there is **no lossless-speedup identity to discover** for batch=1 quantized-MoE decode —
every exact transform either reads all K weights (O(N·K) floor) or saves only marginal
compute, and the rest is lossy. Lean's role is **certification + a proven lower bound**:

* `zero_point_factor`   — the one real (marginal) exact COMPUTE win, over any CommRing (ℤ).
* `moe_combine_linear`  — expert top-k combine = merged operator, algebraically lossless.
* `int_add_assoc` / `float_add_not_assoc` — WHY the integer path is where exact transforms
  live (FP reassociation is not bit-exact) — the L1-vs-L2 boundary, formalized.
* `read_lower_bound`    — CAPSTONE: no reconstructor reading a proper subset of weights can
  compute the exact GEMV for all inputs ⇒ the streaming-IO floor is a THEOREM, not a gap.
-/

namespace Qwisp
open Finset

variable {R : Type*} {ι : Type*}

/-- **T1 — asymmetric-quant zero-point extraction (real, marginal compute win).**
For a quant group, `∑ (qᵢ - z)·xᵢ = (∑ qᵢ·xᵢ) - z·(∑ xᵢ)`. The activation sum `∑ xᵢ`
is independent of the output row and of the expert, so it is computed **once** and reused
across all rows and all 8 top-k experts. Exact over any commutative ring (hence over ℤ,
the quantized-integer path). -/
theorem zero_point_factor [CommRing R] (s : Finset ι) (q x : ι → R) (z : R) :
    ∑ i ∈ s, (q i - z) * x i = (∑ i ∈ s, q i * x i) - z * ∑ i ∈ s, x i := by
  rw [Finset.mul_sum, ← Finset.sum_sub_distrib]
  exact Finset.sum_congr rfl (fun i _ => by ring)

/-- **T2 — MoE top-k combine is a merged linear operator (algebraically lossless).**
`(∑ₑ wₑ • Eₑ) x = ∑ₑ wₑ • (Eₑ x)`: routing-weighted expert combination equals applying a
single merged operator. Lossless *algebraically*; it does not reduce reads, since the merged
operator still depends on every expert — see `read_lower_bound`. -/
theorem moe_combine_linear [CommSemiring R] {M N : Type*}
    [AddCommMonoid M] [Module R M] [AddCommMonoid N] [Module R N]
    {κ : Type*} (S : Finset κ) (w : κ → R) (E : κ → M →ₗ[R] N) (x : M) :
    (∑ e ∈ S, w e • E e) x = ∑ e ∈ S, w e • (E e x) := by
  simp [LinearMap.sum_apply, LinearMap.smul_apply]

/-- Integer accumulation is associative — so `zero_point_factor` etc. are **bit-exact**
on the quantized-integer path. -/
theorem int_add_assoc (a b c : ℤ) : (a + b) + c = a + (b + c) := add_assoc a b c

set_option linter.style.nativeDecide false in
/-- …but IEEE-754 `Float` addition is **not** associative: `(1e20 + -1e20) + 1` evaluates to
`1`, while `1e20 + (-1e20 + 1)` evaluates to `0` (the `+1` is lost to rounding). Stated via
`Bool` equality of the actual IEEE ops (propositional `=` on `Float` is not decidable). This
is the formal reason the exact transforms are L1 only on the integer path — the same
reassociation is merely L2 (distribution-level) on floating point.

`native_decide` is deliberate here: `Float` arithmetic is an opaque `@[extern]` op that the
kernel cannot reduce, so witnessing the *actual* IEEE-754 behaviour requires running the
compiler's real float arithmetic. (The two values are `1.0` and `0.0` — see `#eval` below.) -/
theorem float_add_not_assoc :
    (((1e20 : Float) + -1e20) + 1 == 1e20 + (-1e20 + 1)) = false := by
  native_decide

/-- info: (1.000000, 0.000000) -/
#guard_msgs in
#eval (((1e20 : Float) + -1e20) + 1, 1e20 + (-1e20 + 1))

/-- **T4 — CAPSTONE: the O(N·K) weight-read floor is fundamental.**
Model any "reconstructor" `g` that reads the weights only on a proper subset `S ⊊ support`
(it takes `w` restricted to `S`, plus the full input `x`). Then `g` cannot equal the exact
GEMV `∑ᵢ wᵢ·xᵢ` for all `w, x`: an adversary flips the weight at an unread index `k ∉ S`,
producing a different true output while `g`'s inputs are unchanged. Hence **no lossless
scheme reduces weight reads below full support** — the streaming/IO bottleneck is a theorem,
not an engineering gap (coding, CSE, factoring, prediction all cannot beat it). -/
theorem read_lower_bound [Ring R] [Nontrivial R] {n : ℕ}
    (S : Finset (Fin n)) (k : Fin n) (hk : k ∉ S)
    (g : (∀ i, i ∈ S → R) → (Fin n → R) → R) :
    ¬ ∀ (w x : Fin n → R), g (fun i _ => w i) x = ∑ i, w i * x i := by
  classical
  intro h
  -- probe input: indicator at the unread index k, so `∑ wᵢ·xᵢ = w k`.
  set x : Fin n → R := fun i => if i = k then (1 : R) else 0 with hxdef
  have hsum : ∀ w : Fin n → R, (∑ i, w i * x i) = w k := by
    intro w
    have hstep : ∀ i ∈ (Finset.univ : Finset (Fin n)),
        w i * x i = if i = k then w k else 0 := by
      intro i _; by_cases hik : i = k <;> simp [hxdef, hik]
    rw [Finset.sum_congr rfl hstep, Finset.sum_ite_eq' Finset.univ k (fun _ => w k)]
    simp
  -- two weight tensors that agree on S (both 0 there, since k ∉ S) but differ at k.
  have hrestr : (fun (i : Fin n) (_ : i ∈ S) => (0 : R))
              = (fun (i : Fin n) (_ : i ∈ S) => if i = k then (1 : R) else 0) := by
    funext i hi
    have hik : i ≠ k := fun e => hk (e ▸ hi)
    simp [hik]
  have e0 := h (fun _ => (0 : R)) x
  have e1 := h (fun i => if i = k then (1 : R) else 0) x
  rw [hsum] at e0 e1
  -- g sees identical inputs (restrictions equal) ⇒ equal outputs ⇒ 0 = 1.
  rw [hrestr] at e0
  rw [e1] at e0
  simp at e0

end Qwisp
