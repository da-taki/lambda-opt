/-
  Lipschitz Step Function Properties

  If f is L-Lipschitz, then after n iterations,
  dist(f^n(x), f^n(y)) ≤ L^n * dist(x, y).
-/
import Mathlib.Topology.MetricSpace.Lipschitz
import Mathlib.Topology.MetricSpace.Basic
import LambdaOpt.Defs

open LambdaOpt NNReal

namespace LambdaOpt

variable {X : Type*} [MetricSpace X]

/-- If f is L-Lipschitz, then trajectory f x n and trajectory f y n
    are at most L^n * dist(x, y) apart. -/
theorem trajectory_lipschitz (f : X → X) (L : ℝ≥0) (hf : LipschitzWith L f)
    (x y : X) (n : ℕ) :
    dist (trajectory f x n) (trajectory f y n) ≤ (L : ℝ) ^ n * dist x y := by
  induction n with
  | zero => simp
  | succ n ih =>
    simp only [trajectory_succ]
    have h1 := hf.dist_le_mul (trajectory f x n) (trajectory f y n)
    have h2 : (L : ℝ) ^ (n + 1) = (L : ℝ) * (L : ℝ) ^ n := by ring
    rw [h2]
    calc dist (f (trajectory f x n)) (f (trajectory f y n))
        ≤ (L : ℝ) * dist (trajectory f x n) (trajectory f y n) := h1
      _ ≤ (L : ℝ) * ((L : ℝ) ^ n * dist x y) := by
          apply mul_le_mul_of_nonneg_left ih (NNReal.coe_nonneg L)
      _ = (L : ℝ) * (L : ℝ) ^ n * dist x y := by ring

end LambdaOpt