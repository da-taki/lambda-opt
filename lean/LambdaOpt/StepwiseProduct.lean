/-
  Step-varying product bound (supplementary addition).

  Generalises Theorem 1 (single Lipschitz constant L, giving D ≤ L^k·δ) to the
  case where each step j has its own Lipschitz constant L_j:

      dist(composeSteps f x k, composeSteps f y k) ≤ (∏_{j<k} L_j) · dist x y

  This is the formal backing for the practical step-varying bound used in the
  stepwise-bound experiment. It uses only stable Mathlib API.
-/
import Mathlib.Topology.MetricSpace.Lipschitz
import Mathlib.Tactic
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
    `(∏_{j<k} L_j) · dist x y`. -/
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

/-- A uniform Lipschitz bound recovers the geometric `L^k` factor, so this
    theorem generalises Theorem 1 / `trajectory_lipschitz`. -/
theorem stepwise_product_bound_const
    (f : ℕ → X → X) (L : ℝ≥0)
    (hf : ∀ j, LipschitzWith L (f j))
    (x y : X) (k : ℕ) :
    dist (composeSteps f x k) (composeSteps f y k) ≤ (L : ℝ) ^ k * dist x y := by
  have h := stepwise_product_bound f (fun _ => L) hf x y k
  simpa [Finset.prod_const, Finset.card_range] using h

end LambdaOpt
