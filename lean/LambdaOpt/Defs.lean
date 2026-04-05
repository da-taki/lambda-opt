/-
  λOpt Core Definitions
-/
import Mathlib.Topology.MetricSpace.Basic
import Mathlib.Topology.MetricSpace.Lipschitz

set_option linter.unusedSectionVars false

open Metric

namespace LambdaOpt

variable {X : Type*} [MetricSpace X]

/-- A trajectory of length n starting from x under step function f. -/
def trajectory (f : X → X) (x : X) : ℕ → X
  | 0     => x
  | n + 1 => f (trajectory f x n)

/-- A rewritten trajectory: apply rewrite R at time t_R, then continue with f. -/
def rewrittenTrajectory (f : X → X) (R : X → X) (x : X) (t_R : ℕ) : ℕ → X
  | n => if n < t_R then trajectory f x n
         else trajectory f (R (trajectory f x t_R)) (n - t_R)

/-- The divergence at time t between baseline and rewritten trajectories. -/
def divergence (f : X → X) (R : X → X) (x : X) (t_R : ℕ) (t : ℕ) : ℝ :=
  dist (rewrittenTrajectory f R x t_R t) (trajectory f x t)

/-- δ-bounded rewrite: the displacement is at most δ. -/
def DeltaBounded (R : X → X) (x : X) (δ : ℝ) : Prop :=
  dist (R x) x ≤ δ

@[simp]
theorem trajectory_succ (f : X → X) (x : X) (n : ℕ) :
    trajectory f x (n + 1) = f (trajectory f x n) := rfl

@[simp]
theorem trajectory_zero (f : X → X) (x : X) :
    trajectory f x 0 = x := rfl

/-- After the rewrite point, the rewritten trajectory is just a fresh
    trajectory starting from R(S_{t_R}). -/
theorem rewrittenTrajectory_post (f : X → X) (R : X → X) (x : X)
    (t_R k : ℕ) :
    rewrittenTrajectory f R x t_R (t_R + k) =
      trajectory f (R (trajectory f x t_R)) k := by
  simp [rewrittenTrajectory]

end LambdaOpt