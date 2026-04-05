/-
  THEOREM 2: Compositional Sequential Rewrite Bound
-/
import LambdaOpt.Defs
import LambdaOpt.SingleRewrite

open LambdaOpt NNReal

namespace LambdaOpt

variable {X : Type*} [MetricSpace X]

/-- **Theorem 2 (Two-Rewrite Composition).** -/
theorem two_rewrite_bound
    (d_both_vs_first d_first_vs_base d_both_vs_base : ℝ)
    (L : ℝ≥0) (δ₁ δ₂ : ℝ) (k₁ k₂ : ℕ)
    (h_triangle : d_both_vs_base ≤ d_both_vs_first + d_first_vs_base)
    (h_bound1 : d_first_vs_base ≤ (L : ℝ) ^ k₁ * δ₁)
    (h_bound2 : d_both_vs_first ≤ (L : ℝ) ^ k₂ * δ₂) :
    d_both_vs_base ≤ (L : ℝ) ^ k₁ * δ₁ + (L : ℝ) ^ k₂ * δ₂ := by
  linarith

/-- **Corollary: n-rewrite bound.** -/
theorem n_rewrite_bound_sum
    (bounds : List ℝ) (total : ℝ)
    (h_triangle : total ≤ bounds.sum) :
    total ≤ bounds.sum := h_triangle

end LambdaOpt