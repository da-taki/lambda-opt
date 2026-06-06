# Experiment 8 -- Threshold sensitivity and ROC-style sweep

> **Honesty note.** All `L` values are estimated by finite sampling of perturbation directions (a priori numerical estimator). They are empirical estimates, **not** proof certificates. Decisions labelled `certified_safe` are *analyzer* decisions resting on sampled `L`; they are not formal certificates. See `lipschitz_sampling_stress.md` for the quantified gap between sampled `L` and structured worst-case directions.

## Setup

Per-scenario displacement and raw L are computed once; the epsilon x margin grid only re-evaluates the decision rule. Sensitivity/specificity are computed over **decided** cases (abstentions reported separately as coverage/abstention).

Exact command:

```
python scripts/run_sncs_v4_threshold_sensitivity.py --models quadratic mnist_mlp resnet18
```

## ALL-models sensitivity (rows = epsilon, cols = margin)

| epsilon \ margin | 1.0 | 1.05 | 1.1 | 1.25 | 1.5 | 2.0 |
| --- | --- | --- | --- | --- | --- | --- |
| 0.001 | 0.93 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 0.005 | 0.79 | 0.98 | 1.00 | 1.00 | 1.00 | 1.00 |
| 0.01 | 0.71 | 0.90 | 1.00 | 1.00 | 1.00 | 1.00 |
| 0.05 | 0.67 | 0.85 | 1.00 | 1.00 | 1.00 | 1.00 |
| 0.1 | 0.67 | 0.83 | 1.00 | 1.00 | 1.00 | 1.00 |
| 0.5 | 0.67 | 0.83 | 1.00 | 1.00 | 1.00 | 1.00 |
| 1.0 | 0.67 | 0.83 | 1.00 | 1.00 | 1.00 | 1.00 |

## ALL-models specificity

| epsilon \ margin | 1.0 | 1.05 | 1.1 | 1.25 | 1.5 | 2.0 |
| --- | --- | --- | --- | --- | --- | --- |
| 0.001 | 0.72 | 0.72 | 0.72 | 0.72 | 0.72 | 0.72 |
| 0.005 | 0.72 | 0.72 | 0.72 | 0.72 | 0.72 | 0.72 |
| 0.01 | 0.72 | 0.72 | 0.72 | 0.72 | 0.72 | 0.72 |
| 0.05 | 0.72 | 0.72 | 0.72 | 0.72 | 0.72 | 0.72 |
| 0.1 | 0.72 | 0.72 | 0.72 | 0.72 | 0.72 | 0.72 |
| 0.5 | 0.72 | 0.72 | 0.72 | 0.72 | 0.72 | 0.72 |
| 1.0 | 0.72 | 0.72 | 0.72 | 0.72 | 0.72 | 0.72 |

## ALL-models coverage rate

| epsilon \ margin | 1.0 | 1.05 | 1.1 | 1.25 | 1.5 | 2.0 |
| --- | --- | --- | --- | --- | --- | --- |
| 0.001 | 0.82 | 0.87 | 0.98 | 1.00 | 1.00 | 0.96 |
| 0.005 | 0.90 | 0.88 | 0.98 | 1.00 | 1.00 | 0.96 |
| 0.01 | 0.96 | 0.93 | 0.98 | 1.00 | 1.00 | 0.96 |
| 0.05 | 1.00 | 0.96 | 0.98 | 1.00 | 1.00 | 0.96 |
| 0.1 | 1.00 | 0.98 | 0.98 | 1.00 | 1.00 | 0.96 |
| 0.5 | 1.00 | 0.98 | 0.98 | 1.00 | 1.00 | 0.96 |
| 1.0 | 1.00 | 0.98 | 0.98 | 1.00 | 1.00 | 0.96 |

## ALL-models false-negative rate

| epsilon \ margin | 1.0 | 1.05 | 1.1 | 1.25 | 1.5 | 2.0 |
| --- | --- | --- | --- | --- | --- | --- |
| 0.001 | 0.07 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 0.005 | 0.21 | 0.02 | 0.00 | 0.00 | 0.00 | 0.00 |
| 0.01 | 0.29 | 0.10 | 0.00 | 0.00 | 0.00 | 0.00 |
| 0.05 | 0.33 | 0.15 | 0.00 | 0.00 | 0.00 | 0.00 |
| 0.1 | 0.33 | 0.17 | 0.00 | 0.00 | 0.00 | 0.00 |
| 0.5 | 0.33 | 0.17 | 0.00 | 0.00 | 0.00 | 0.00 |
| 1.0 | 0.33 | 0.17 | 0.00 | 0.00 | 0.00 | 0.00 |

## Answers to the required questions

**1. Is zero false negatives only caused by an extreme threshold?** 29/42 of the grid points have zero false negatives. Zero-FN holds across a broad region of the grid, not just an extreme corner -- the no-false-negative property is robust to threshold choice.

**2. What threshold gives the best balance?** epsilon=0.001, margin=1.05 (sensitivity=1.00, specificity=0.72, coverage=0.87) by Youden's J among points with >=50% coverage.

**3. How sensitive are claims to epsilon?** Reading across each row (fixed margin), increasing epsilon converts abstentions/flags into `certified_safe`, raising coverage but eventually admitting false negatives. See the coverage and FNR grids above.

**4. What operating point should the paper report?** Report epsilon=0.001, margin=1.05 as the balanced operating point, and explicitly state that zero-FN holds over 29/42 grid points. Pair any 'zero false negative' claim with this operating point rather than presenting it as threshold-free.

## Limitations

* Sensitivity/specificity are over decided cases; a conservative operating point trades coverage for safety via abstention.
* `L` is sampled; the grid inherits that empirical caveat.
* Ground-truth danger thresholds (divergence/loss-gap) are fixed here; they are separate from the analyzer's epsilon/margin.