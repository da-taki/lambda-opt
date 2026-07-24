# Experiment 2 -- Certification coverage

> **Honesty note.** All `L` values are estimated by finite sampling of perturbation directions (a priori numerical estimator). They are empirical estimates, **not** proof certificates. Decisions labelled `certified_safe` are *analyzer* decisions resting on sampled `L`; they are not formal certificates. See `lipschitz_sampling_stress.md` for the quantified gap between sampled `L` and structured worst-case directions.

## Setup

The canonical repaired analyzer (repaired displacement + `L=max(L_clean,L_rewritten)` + abstain) is run on all 12 scenarios for each diagnostic. Coverage = fraction of cases with a non-abstaining decision. Ground-truth danger is threshold-defined (see per-row `div_threshold`/`loss_gap_threshold`).

Exact command:

```
python scripts/run_decision_coverage.py --models quadratic mnist_mlp resnet18 gpt_mini
```

## Overall

| metric | value |
| --- | --- |
| total cases | 96 |
| certified safe | 24 |
| flagged dangerous | 65 |
| abstained | 7 |
| coverage rate | 0.9271 |
| abstention rate | 0.07292 |
| false positives | 11 |
| false negatives | 0 |

Main reason for abstention: **expansive regime but bound below danger threshold (cannot certify)** (7 cases).

## By model

| model | cases | certified | flagged | abstained | coverage | abstention | FP | FN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gpt_mini | 12 | 3 | 9 | 0 | 1 | 0 | 3 | 0 |
| mnist_mlp | 36 | 9 | 27 | 0 | 1 | 0 | 8 | 0 |
| quadratic | 36 | 9 | 20 | 7 | 0.8056 | 0.1944 | 0 | 0 |
| resnet18 | 12 | 3 | 9 | 0 | 1 | 0 | 0 | 0 |

## Reading

* Coverage rate overall is **92.7%** (89/96 cases decided).
* Abstention rate is **7.3%**; the dominant cause is `expansive regime but bound below danger threshold (cannot certify)`.
* False negatives: **0** (a dangerous rewrite certified safe). False positives: **11** (a benign rewrite flagged dangerous -- the acceptable conservative cost).

## Limitations

* `L` is sampled, so `certified_safe` is an empirical analyzer verdict.
* ResNet-18/GPT-mini rows use the small CPU-budget configs; coverage on those is indicative, not production-scale.
* Abstention is the designed, honest output when sampled `L` sits on the contraction boundary or the bound falls between epsilon and the danger threshold.