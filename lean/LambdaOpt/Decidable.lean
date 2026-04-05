/-
  THEOREM 3: Decidability of ε-Trajectory-Equivalence
-/
import Mathlib.Data.Real.Basic
import Mathlib.Tactic

set_option linter.unusedVariables false

namespace LambdaOpt

/-- ε-trajectory-equivalence: the bound is at most ε. -/
def epsilonEquivalent (bound ε : ℝ) : Prop := bound ≤ ε

/-- The bound Σᵢ δᵢ · L^(T − tᵢ) is computable from finite data. -/
def computeBound (L : ℝ) (deltas : List ℝ) (exponents : List ℕ) : ℝ :=
  (deltas.zip exponents).map (fun p => p.1 * L ^ p.2) |>.sum

/-- **Theorem 3.** ε-equivalence reduces to checking computeBound ≤ ε. -/
theorem epsilon_equivalence_iff_bound
    (L : ℝ) (deltas : List ℝ) (exponents : List ℕ) (ε : ℝ) :
    epsilonEquivalent (computeBound L deltas exponents) ε ↔
    computeBound L deltas exponents ≤ ε := by
  rfl

/-- For rational inputs, the comparison is decidable. -/
def rationalBoundDecidable
    (L_q : ℚ) (deltas_q : List ℚ) (exps : List ℕ) (ε_q : ℚ) :
    Bool :=
  let bound := (deltas_q.zip exps).map (fun p => p.1 * L_q ^ p.2) |>.sum
  decide (bound ≤ ε_q)

end LambdaOpt