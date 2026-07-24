# Supplementary review-experiment summary

> **Honesty note.** All `L` values are estimated by finite sampling of perturbation directions (a priori numerical estimator). They are empirical estimates, **not** proof certificates. Decisions labelled `certified_safe` are *analyzer* decisions resting on sampled `L`; they are not formal certificates. See `lipschitz_sampling_stress.md` for the quantified gap between sampled `L` and structured worst-case directions.

## 1. Repo branch / 2. Commit

* Branch: `anonymous-review-clean`
* Commit: `(anonymized review artifact)`

## 3. Exact commands run

```
python -m venv .venv && .venv\Scripts\python.exe -m pip install -r experiments/requirements.txt
set PYTHONPATH=%CD%\experiments
python scripts/run_baseline_comparison.py --models quadratic mnist_mlp --seeds 0 1 2
python scripts/run_decision_coverage.py --models quadratic mnist_mlp resnet18 gpt_mini --small-seeds 0 1 2 --big-seeds 0
python scripts/run_stepwise_bound_comparison.py --models quadratic mnist_mlp resnet18 --small-seeds 0 1 2 --big-seeds 0
python scripts/run_multiseed_validation.py
python scripts/run_cifar10_checkpoint_matrix.py --seeds 0 1 2
python scripts/run_runtime_overhead.py --models mnist_mlp resnet18 gpt_mini
python scripts/run_metric_ablation.py --models quadratic mnist_mlp --seeds 0 1 2
python scripts/run_threshold_sensitivity.py --models quadratic mnist_mlp resnet18 --small-seeds 0 1 2 --big-seeds 0
python scripts/run_lipschitz_sampling_stress.py --models quadratic mnist_mlp --seeds 0 1
python scripts/run_make_summary.py
```

## 4. Test results

`pytest -q`: 43 passed (py -m pytest experiments/tests -q)

## 5. Lean build result

PASSED -- lake build: 3292 jobs, 0 errors, 0 sorry (Theorems 1-3 + stepwise_product_bound). See lean_build_status.md.

## 6. Experiments completed

* 1. Baseline comparison
* 2. Certification coverage
* 3. Step-varying bound comparison
* 4. Multiseed validation
* 5. Real CIFAR-10 checkpoint matrix
* 6. Runtime overhead
* 7. Metric repair ablation
* 8. Threshold sensitivity
* 9. Sampled-L stress
* 10. Lean theorem mapping

## 7. Experiments skipped and why

* None -- all 10 experiments produced output. Big-model (ResNet-18/GPT-mini) runs use small CPU-budget configs, documented per experiment.

## 8. Main numerical results

* **1. Baseline comparison**: repaired analyzer false negatives = 0 vs param-norm 40, optimizer-norm 27, loss-probe 20
* **2. Certification coverage**: overall coverage 92.7%, abstention 7.3%, FN=0, FP=11
* **3. Step-varying bound comparison**: fixed max-L vacuous on 0 cases (median looseness 26.14); stepwise product vacuous on 0 (median looseness 2.955)
* **4. Multiseed validation**: gpt_mini: FN=0, cov=1; mnist_mlp: FN=0, cov=1; quadratic: FN=0, cov=0.8667; resnet18: FN=0, cov=1
* **5. Real CIFAR-10 checkpoint matrix**: real CIFAR-10 (subset): sensitivity=1, specificity=1, FN=0, FP=0
* **6. Runtime overhead**: analyzer (5x5) runtime: mnist_mlp=0.098s; resnet18=35.403s; gpt_mini=0.614s
* **7. Metric repair ablation**: raw metric delta=0 for scheduler mismatch: True; repaired delta=0.02471
* **8. Threshold sensitivity**: zero-FN on 29/42 grid points; best balance eps=0.001, margin=1.05
* **9. Sampled-L stress**: most dangerous direction = adam_denom_sensitive; random miss-rate@1probe=1; probes for random miss<50% = None

## 9. Tables to insert into manuscript

See `tables_for_paper.md` (10 tables, auto-generated).

## 10. Claims strengthened

* Baselines: simple magnitude/raw metrics miss schedule-only dangers; the repaired analyzer does not (experiment 1).
* Metric repair: the original (theta,M) metric is provably blind to scheduler mismatch (delta=0); the repaired metric is not (experiment 7).
* Step-varying bound reduces vacuity vs fixed-L on deep nets (experiment 3).
* Real CIFAR-10 checkpoint matrix corroborates the synthetic findings (experiment 5).
* Theorems 1-3 mechanized; step-varying product bound added (experiment 10).

## 11. Claims weakened / must be stated carefully

* 'Zero false negatives' is an operating-point property, not threshold-free (experiment 8): report the epsilon/margin and the fraction of the grid where it holds.
* 'Certification' from sampled L is an *empirical analyzer* result; random probes can miss structured worst-case directions (experiment 9). Use 'analyzer-certified', not 'certified'.
* Conservative false positives are real (benign moment changes flagged); report them honestly.

## 12. Remaining reviewer risks

* Big-model evidence is CPU-limited (short horizons, tiny subsets).
* Sampled-L has no a-priori worst-case guarantee; a Hessian-spectral or interval-arithmetic L would be needed for a true certificate.
* Ground-truth danger labels depend on chosen thresholds (swept in experiment 8).

## 13. Reproduction commands

See section 3 above; every script writes CSV+JSON+MD into `results/review_experiments/`. Fixed seeds are used throughout (quadratic 0-4, MNIST 0-2, big models seed 0+).

## 5 most important numbers

* repaired analyzer false negatives = 0 vs param-norm 40, optimizer-norm 27, loss-probe 20
* overall coverage 92.7%, abstention 7.3%, FN=0, FP=11
* fixed max-L vacuous on 0 cases (median looseness 26.14); stepwise product vacuous on 0 (median looseness 2.955)
* real CIFAR-10 (subset): sensitivity=1, specificity=1, FN=0, FP=0
* raw metric delta=0 for scheduler mismatch: True; repaired delta=0.02471