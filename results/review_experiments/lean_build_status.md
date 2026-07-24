# Lean build status

**Status: A. Lean build PASSED.**

## Exact command

```
# from repo root
cd lean
lake exe cache get        # fetch prebuilt Mathlib oleans (see provisioning note)
lake build                # default target = LambdaOpt (+ new StepwiseProduct)
```

## Result / output summary

```
warning: LambdaOpt/SingleRewrite.lean:31:5: unused variable `hδ`   (benign linter warning)
✔ [3291/3292] Built LambdaOpt (50s)
Build completed successfully (3292 jobs).
```

* **3292 build jobs, 0 errors, 0 `sorry`.** (The pre-existing project is 3291
  jobs; the +1 is the new `LambdaOpt/StepwiseProduct.lean` added on this branch.)
* The only diagnostic is a benign `unusedVariables` linter warning on the
  pre-existing `SingleRewrite.lean` (the unused hypothesis `hδ`); it is a warning,
  not an error, and does not affect soundness.
* Mechanized theorems confirmed by this build:
  * `LambdaOpt.single_rewrite_bound` (Theorem 1)
  * `LambdaOpt.trajectory_lipschitz` (iterated contraction)
  * `LambdaOpt.two_rewrite_bound`, `n_rewrite_bound_sum` (Theorem 2)
  * `LambdaOpt.epsilon_equivalence_iff_bound`, `rationalBoundDecidable` (Theorem 3)
  * **`LambdaOpt.stepwise_product_bound` and `stepwise_product_bound_const`
    (NEW: step-varying product bound, this branch).**

## What had to be fixed to build (environment, not proofs)

The clone did **not** build out of the box; two environment issues were fixed:

1. **Empty `lean-toolchain`.** The committed `lean/lean-toolchain` was a 0-byte
   file, so `lake` resolved to the *latest* Lean (v4.30.0) instead of the pinned
   v4.29.0, which would mismatch the Mathlib revision. Fixed by writing
   `leanprover/lean4:v4.29.0` into `lean/lean-toolchain` (committed).

2. **Stale `.lake` dependency package URLs.** The committed
   `lean/.lake/packages/*` carried dependency checkouts whose remote URLs no
   longer matched the manifest; `lake exe cache get` reported
   `"<pkg>: URL has changed; you might need to delete ... manually"` followed by
   `error: external command 'git' exited with code 128`. Fixed by removing
   `lean/.lake/packages/*` and re-running `lake exe cache get`, which re-cloned
   every dependency at its manifest revision and downloaded the prebuilt Mathlib
   olean cache (8248 files, 100%). These `.lake` changes are build/cache
   artifacts and are **not** committed.

## New theorem

`lean/LambdaOpt/StepwiseProduct.lean` adds the step-varying product bound that
backs experiment 3:

```
theorem stepwise_product_bound (f : ℕ → X → X) (L : ℕ → ℝ≥0)
    (hf : ∀ j, LipschitzWith (L j) (f j)) (x y : X) (k : ℕ) :
    dist (composeSteps f x k) (composeSteps f y k)
      ≤ (∏ j ∈ Finset.range k, (L j : ℝ)) * dist x y
```

plus `stepwise_product_bound_const`, which recovers the geometric `Lᵏ` factor of
Theorem 1 (so the new theorem strictly generalises Theorem 1). The statement is
also recorded in `lean_stepwise_todo.md`.

## Dependency of Python experiments on Lean

**None.** The experiment code (`scripts/*.py`, `experiments/`) uses only Python +
PyTorch + NumPy/SciPy. The Lean development is the separate formal-proofs side.
The experiment CSV/JSON/Markdown outputs were produced and validated
independently of the Lean build; the commit and push were never blocked on Lean.
