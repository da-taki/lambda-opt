# Experiment 4 -- Multiple-seed validation

> **Honesty note.** All `L` values are estimated by finite sampling of perturbation directions (a priori numerical estimator). They are empirical estimates, **not** proof certificates. Decisions labelled `certified_safe` are *analyzer* decisions resting on sampled `L`; they are not formal certificates. See `lipschitz_sampling_stress.md` for the quantified gap between sampled `L` and structured worst-case directions.

## Setup

The canonical analyzer is run across multiple seeds per diagnostic; we report mean +/- standard deviation of every numeric metric. `delta_linearity_r2` is the R^2 of repaired-delta vs max divergence across the 12 scenarios; `spearman_loss_divergence` is Spearman rho between max loss-gap and max divergence.

Exact command:

```
python scripts/run_sncs_v4_multiseed_validation.py
```

Seeds used: quadratic=[0, 1, 2, 3, 4]; mnist_mlp=[0, 1, 2]; resnet18=[0]; gpt_mini=[0].

## Mean +/- std by model

| model | seeds | coverage | FN | FP | abstained | bound failures | delta-R^2 | loss/div rho |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gpt_mini | 1 | 1.00+/-0.00 | 0.00+/-0.00 | 3.00+/-0.00 | 0.00+/-0.00 | 1.00+/-0.00 | 0.29+/-0.00 | 0.96+/-0.00 |
| mnist_mlp | 3 | 1.00+/-0.00 | 0.00+/-0.00 | 2.67+/-0.47 | 0.00+/-0.00 | 1.00+/-0.00 | 0.11+/-0.00 | 0.91+/-0.03 |
| quadratic | 5 | 0.87+/-0.08 | 0.00+/-0.00 | 0.00+/-0.00 | 1.60+/-1.02 | 7.80+/-0.40 | 0.79+/-0.13 | 0.98+/-0.01 |
| resnet18 | 1 | 1.00+/-0.00 | 0.00+/-0.00 | 0.00+/-0.00 | 0.00+/-0.00 | 1.00+/-0.00 | 0.49+/-0.00 | 0.73+/-0.00 |

## Notes & limitations

* Quadratic uses 5 seeds, MNIST MLP 3; ResNet-18 and GPT-mini default to 1 seed on CPU (each big-model run re-pretrains + analyzes 12 scenarios with two sampled-L estimates apiece, which is the dominant cost). Raise `--seeds-resnet18`/`--seeds-gpt_mini` to add seeds on a GPU host.
* `false_negatives` is the headline robustness metric: a dangerous rewrite certified safe. Across seeds it should stay at/near zero.
* `L` is sampled; all decisions are empirical analyzer decisions.