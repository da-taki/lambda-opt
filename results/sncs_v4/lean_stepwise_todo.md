# Step-varying product bound -- Lean theorem (statement + proof sketch)

This file records the **exact** Lean 4 statement of the step-varying product
bound so it can be mechanized cleanly, independent of whether the Mathlib
toolchain was provisioned on the experiment host. The candidate implementation
lives at `lean/LambdaOpt/StepwiseProduct.lean`; whether `lake build` verified it
on this host is recorded in `SNCS_V4_REVIEWER_PROOFING_SUMMARY.md`.

## Concept

Theorem 1 uses a single Lipschitz constant `L`: `D(t_R+k) ≤ Lᵏ·δ`. When each
post-resume step has its own constant `L_j` (e.g. a changed schedule, or
contraction that varies along the trajectory), the tight bound is the **product**

    D(t_R + k) ≤ δ · ∏_{j=0}^{k-1} L_j .

This is the formal backing for experiment 3 (the step-varying bound that makes
deep-net bounds non-vacuous).

## Exact Lean statement

```lean
import Mathlib.Topology.MetricSpace.Lipschitz
import Mathlib.Algebra.BigOperators.Order
import LambdaOpt.Defs

open NNReal BigOperators

namespace LambdaOpt

variable {X : Type*} [MetricSpace X]

/-- Compose a sequence of (possibly different) step functions:
    `composeSteps f x k = f_{k-1} (... (f_1 (f_0 x)))`. -/
def composeSteps (f : ℕ → X → X) (x : X) : ℕ → X
  | 0     => x
  | k + 1 => f k (composeSteps f x k)

/-- **Step-varying product bound.** If step `j` is `L_j`-Lipschitz, then after
    `k` composed steps the divergence between two trajectories is at most
    `(∏_{j<k} L_j) · dist x y`. Specialises to Theorem 1 when `L_j` is constant. -/
theorem stepwise_product_bound
    (f : ℕ → X → X) (L : ℕ → ℝ≥0)
    (hf : ∀ j, LipschitzWith (L j) (f j))
    (x y : X) (k : ℕ) :
    dist (composeSteps f x k) (composeSteps f y k)
      ≤ (∏ j ∈ Finset.range k, (L j : ℝ)) * dist x y := by
  induction k with
  | zero => simp [composeSteps]
  | succ k ih =>
    simp only [composeSteps, Finset.prod_range_succ]
    calc dist (f k (composeSteps f x k)) (f k (composeSteps f y k))
        ≤ (L k : ℝ) * dist (composeSteps f x k) (composeSteps f y k) :=
          (hf k).dist_le_mul _ _
      _ ≤ (L k : ℝ) * ((∏ j ∈ Finset.range k, (L j : ℝ)) * dist x y) :=
          mul_le_mul_of_nonneg_left ih (NNReal.coe_nonneg _)
      _ = (∏ j ∈ Finset.range k, (L j : ℝ)) * (L k : ℝ) * dist x y := by ring

/-- Corollary: a uniform Lipschitz bound recovers the geometric `Lᵏ` factor,
    i.e. this generalises `trajectory_lipschitz` / Theorem 1. -/
theorem stepwise_product_bound_const
    (f : ℕ → X → X) (L : ℝ≥0)
    (hf : ∀ j, LipschitzWith L (f j))
    (x y : X) (k : ℕ) :
    dist (composeSteps f x k) (composeSteps f y k) ≤ (L : ℝ) ^ k * dist x y := by
  have h := stepwise_product_bound f (fun _ => L) hf x y k
  simpa [Finset.prod_const, Finset.card_range] using h

end LambdaOpt
```

## Integration notes

* `composeSteps f` with `f j := Step_O` (a fixed optimizer step) and `L j := L`
  reduces to the existing `trajectory_lipschitz`; the corollary above states that
  reduction, so the new theorem strictly generalises Theorem 1.
* To connect to `rewrittenTrajectory`, instantiate `x := R (τ(t_R))`,
  `y := τ(t_R)` and `f j :=` the step at time `t_R + j` (whose schedule may vary).
  Then `composeSteps f x k = τ^R(t_R+k)` and `composeSteps f y k = τ(t_R+k)`, and
  `dist x y ≤ δ` gives `D(t_R+k) ≤ (∏_{j<k} L_j)·δ`.
* The proof uses only `LipschitzWith.dist_le_mul`, `Finset.prod_range_succ`,
  `mul_le_mul_of_nonneg_left`, and `NNReal.coe_nonneg` -- all stable Mathlib API
  at the pinned revision -- so it is expected to compile without `sorry`.
