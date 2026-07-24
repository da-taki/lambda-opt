# manuscript-ready tables

Auto-generated from `results/review_experiments/*.json`. > **Honesty note.** All `L` values are estimated by finite sampling of perturbation directions (a priori numerical estimator). They are empirical estimates, **not** proof certificates. Decisions labelled `certified_safe` are *analyzer* decisions resting on sampled `L`; they are not formal certificates. See `lipschitz_sampling_stress.md` for the quantified gap between sampled `L` and structured worst-case directions.

## Table 1. Baseline comparison

| method | cases | dangerous caught | false neg | false pos | abstained |
| --- | --- | --- | --- | --- | --- |
| A. parameter-norm delta | 72 | 6/46 | 40 | 0 | 0 |
| B. optimizer-buffer delta | 72 | 19/46 | 27 | 8 | 0 |
| C. raw product metric | 72 | 46/46 | 0 | 26 | 0 |
| D. block-normalized metric | 72 | 46/46 | 0 | 26 | 0 |
| E. loss-probe heuristic | 72 | 26/46 | 20 | 0 | 0 |
| F. repaired lambda-Opt | 72 | 39/46 | 0 | 8 | 7 |

## Table 2. Certification coverage

| model | cases | certified | flagged | abstained | coverage | FP | FN |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gpt_mini | 12 | 3 | 9 | 0 | 1 | 3 | 0 |
| mnist_mlp | 36 | 9 | 27 | 0 | 1 | 8 | 0 |
| quadratic | 36 | 9 | 20 | 7 | 0.8056 | 0 | 0 |
| resnet18 | 12 | 3 | 9 | 0 | 1 | 0 | 0 |
| ALL | 96 | 24 | 65 | 7 | 0.9271 | 11 | 0 |

## Table 3. Step-varying bound comparison

| bound type | cases | holds | vacuous | median looseness | certifies safe | abstains |
| --- | --- | --- | --- | --- | --- | --- |
| A. fixed clean L | 84 | 56 | 0 | 3.277 | 0 | 28 |
| B. fixed rewritten L | 84 | 53 | 0 | 17.1 | 0 | 29 |
| C. fixed max L | 84 | 56 | 0 | 26.14 | 0 | 28 |
| D. stepwise product | 84 | 56 | 0 | 2.955 | 0 | 28 |
| E. empirical envelope (diag) | 84 | 77 | 0 | 1.035 | 0 | 0 |

## Table 4. Multiseed validation

| model | seeds | coverage | FN | FP | bound fails | delta-R^2 | loss/div rho |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gpt_mini | 1 | 1+/-0 | 0+/-0 | 3+/-0 | 1+/-0 | 0.2944+/-0 | 0.9645+/-0 |
| mnist_mlp | 3 | 1+/-0 | 0+/-0 | 2.667+/-0.4714 | 1+/-0 | 0.1069+/-0.001709 | 0.9102+/-0.02611 |
| quadratic | 5 | 0.8667+/-0.08498 | 0+/-0 | 0+/-0 | 7.8+/-0.4 | 0.793+/-0.1291 | 0.9801+/-0.006949 |
| resnet18 | 1 | 1+/-0 | 0+/-0 | 0+/-0 | 1+/-0 | 0.4944+/-0 | 0.7305+/-0 |

## Table 5. Real CIFAR-10 checkpoint matrix

| metric | value |
| --- | --- |
| total | 36 |
| actual dangerous | 27 |
| actual benign | 9 |
| true positives | 27 |
| true negatives | 9 |
| false positives | 0 |
| false negatives | 0 |
| abstentions | 0 |
| sensitivity | 1 |
| specificity | 1 |
| coverage | 1 |

## Table 6. Runtime overhead

| model | params | 1x1s | 3x1s | 5x5s | 10x10s | 20x15s |
| --- | --- | --- | --- | --- | --- | --- |
| mnist_mlp | 52650 | 0.004 | 0.014 | 0.098 | 0.404 | 1.197 |
| resnet18 | 11181642 | 1.807 | 4.888 | 35.403 | 140.968 | 412.934 |
| gpt_mini | 85312 | 0.033 | 0.068 | 0.614 | 2.350 | 6.180 |

## Table 7. Metric repair ablation

| finding | value |
| --- | --- |
| raw (theta,M) delta for scheduler_mismatch is exactly 0 | True |
| repaired mean delta for scheduler_mismatch | 0.02471 |
| raw mean delta for reset_v (buffer units) | 1.578 |
| denom-aware mean delta for reset_v (theta units) | 1.557 |
| block-norm classification changes vs raw | 0/30 |

## Table 8. Threshold sensitivity

| quantity | value |
| --- | --- |
| grid points total | 42 |
| grid points with zero false negatives | 29 |
| grid points with nonzero FN | 13 |
| best-balance epsilon | 0.001 |
| best-balance margin | 1.05 |
| best-balance sensitivity | 1 |
| best-balance specificity | 0.7241 |
| best-balance coverage | 0.869 |

## Table 9. Sampled-L stress

| direction | mean L (probes>=10) |
| --- | --- |
| random | 5.562 |
| gradient_aligned | 5.516 |
| trajectory_aligned | 5.563 |
| top_hessian | 1.286 |
| adam_denom_sensitive | 5.6 |
| schedule_time | 5.591 |

## Table 10. Lean theorem mapping

| paper result | Lean file | theorem | mechanized |
| --- | --- | --- | --- |
| Theorem 1 (single rewrite) | SingleRewrite.lean | single_rewrite_bound | yes |
| Iterated contraction L^n | LipschitzStep.lean | trajectory_lipschitz | yes |
| Theorem 2 (composition) | Composition.lean | two_rewrite_bound | yes |
| Theorem 3 (decidability) | Decidable.lean | epsilon_equivalence_iff_bound | yes |
| Theorem 4 (noise floor) | (none) | - | no (empirical) |
| Step-varying product bound | StepwiseProduct.lean | stepwise_product_bound | yes (3292 jobs) |
