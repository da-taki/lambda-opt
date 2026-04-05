/-
  THEOREM 1: Single Rewrite Bound

  If Step is L-Lipschitz and ‖R(S) − S‖ ≤ δ, then for all k ≥ 0:
    dist(τᴿ(t_R + k), τ(t_R + k)) ≤ L^k · δ
-/
import LambdaOpt.Defs
import LambdaOpt.LipschitzStep

open LambdaOpt NNReal

namespace LambdaOpt

variable {X : Type*} [MetricSpace X]

/-- Auxiliary: trajectory f x (t_R + k) = trajectory f (trajectory f x t_R) k -/
theorem trajectory_shift (f : X → X) (x : X) (t_R k : ℕ) :
    trajectory f x (t_R + k) = trajectory f (trajectory f x t_R) k := by
  induction k with
  | zero => simp
  | succ k ih =>
    have : t_R + (k + 1) = (t_R + k) + 1 := by omega
    rw [this, trajectory_succ, trajectory_succ, ih]

/-- **Theorem 1 (Single Rewrite Bound).** -/
theorem single_rewrite_bound
    (f : X → X) (R : X → X) (x : X)
    (L : ℝ≥0) (δ : ℝ) (t_R k : ℕ)
    (hf : LipschitzWith L f)
    (hR : DeltaBounded R (trajectory f x t_R) δ)
    (hδ : 0 ≤ δ) :
    dist (rewrittenTrajectory f R x t_R (t_R + k))
         (trajectory f x (t_R + k)) ≤ (L : ℝ) ^ k * δ := by
  rw [rewrittenTrajectory_post, trajectory_shift]
  calc dist (trajectory f (R (trajectory f x t_R)) k)
            (trajectory f (trajectory f x t_R) k)
      ≤ (L : ℝ) ^ k * dist (R (trajectory f x t_R)) (trajectory f x t_R) :=
        trajectory_lipschitz f L hf _ _ k
    _ ≤ (L : ℝ) ^ k * δ := by
        apply mul_le_mul_of_nonneg_left hR
        positivity

end LambdaOpt