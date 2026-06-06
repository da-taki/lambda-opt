# Experiment 10 -- Lean theorem mapping

This maps each paper-level result to its Lean 4 mechanization (or records that it
is empirical-only). Source inspected: `lean/LambdaOpt/*.lean` at the repo branch
`sncs-v4-reviewer-experiments`.

The Lean development pins Lean 4.29.0 and Mathlib `8850ed9` (see
`lean/lake-manifest.json`). The build status recorded here is the result of
running `lake build` in `lean/` on the experiment host; see
`SNCS_V4_REVIEWER_PROOFING_SUMMARY.md` for the captured console output.

## Mapping table

| paper_result | Lean_file | Lean_theorem_name | mechanized_yes_no | depends_on | notes |
| --- | --- | --- | --- | --- | --- |
| Core state/trajectory model (S=(θ,M,h,t), τ, τ^R, divergence, δ-bounded) | `LambdaOpt/Defs.lean` | `trajectory`, `rewrittenTrajectory`, `divergence`, `DeltaBounded`, `rewrittenTrajectory_post` | yes (definitions + 1 lemma) | Mathlib `MetricSpace` | Defines the objects every theorem is stated over. `rewrittenTrajectory_post` is the key rewrite-splitting lemma. |
| Iterated-step contraction `dist(fⁿ x, fⁿ y) ≤ Lⁿ·dist(x,y)` | `LambdaOpt/LipschitzStep.lean` | `trajectory_lipschitz` | yes | `LipschitzWith`, `Defs.trajectory` | The engine behind Theorem 1; induction on n. |
| **Theorem 1** Single-rewrite bound `D(t_R+k) ≤ Lᵏ·δ` | `LambdaOpt/SingleRewrite.lean` | `single_rewrite_bound` | **yes** | `trajectory_lipschitz`, `rewrittenTrajectory_post`, `trajectory_shift` | Fully mechanized, no `sorry`. |
| Trajectory shift lemma `τ(t_R+k)=τ_{τ(t_R)}(k)` | `LambdaOpt/SingleRewrite.lean` | `trajectory_shift` | yes | `Defs.trajectory` | Auxiliary for Theorem 1. |
| **Theorem 2** Sequential composition (additive bound) | `LambdaOpt/Composition.lean` | `two_rewrite_bound`, `n_rewrite_bound_sum` | **yes** | triangle inequality, `single_rewrite_bound` shape | Two-rewrite case proved by `linarith`; n-rewrite is the summed form. |
| **Theorem 3** Decidability of ε-trajectory-equivalence | `LambdaOpt/Decidable.lean` | `epsilon_equivalence_iff_bound`, `rationalBoundDecidable` | **yes** | `Mathlib.Data.Real.Basic`, `decide` on ℚ | Reduces ε-equivalence to a finite rational comparison. |
| **Theorem 4** Stochastic noise floor `σ/(1−L)` | (none) | -- | **no** | would need Mathlib martingale/probability | README states this is empirically validated; Lean mechanization is future work. Validated empirically in `experiments/results/adam_noise_floor/`. |
| Practical a-priori Lipschitz estimator (sampled L) | (none -- Python only) | -- | **no** | -- | `experiments/src/lipschitz_apriori.py`. Empirical estimator; experiment 9 quantifies the sampled-vs-worst-case gap. Not a theorem. |
| Repaired displacement metric (θ-commensurate δ) | (none -- Python only) | -- | **no** | -- | `scripts/sncs_v4_common.py: metric_delta/full_repaired`. A modelling choice validated empirically (experiment 7), not a Lean object. |
| **Step-varying product bound** `D(t_R+k) ≤ δ·∏_{j<k} L_j` | `LambdaOpt/StepwiseProduct.lean` (added in this branch) | `stepwise_product_bound`, `stepwise_product_bound_const`, `composeSteps` | **yes** | `LipschitzWith`, `Finset.prod` | Generalises `trajectory_lipschitz` to per-step constants. **Mechanized and built**: `lake build` succeeds with 3292 jobs, 0 errors, 0 sorry (see `lean_build_status.md`). `stepwise_product_bound_const` recovers Theorem 1's `Lᵏ` factor. |

## Notes

* Theorems 1-3 and their supporting lemmas are mechanized with no `sorry`
  (verified by reading the source; the build log in the summary records whether
  `lake build` reproduced this on the host).
* Theorem 4 and the two engineering constructions (sampled-L estimator, repaired
  metric) are deliberately **not** claimed as mechanized. They are empirical /
  modelling contributions; conflating them with the mechanized core would
  overstate the formal guarantee.
* The step-varying product bound is the natural Lean generalisation of the
  existing `trajectory_lipschitz` lemma and is the recommended next mechanization
  target because it is what makes the deep-net bounds non-vacuous (experiment 3).
