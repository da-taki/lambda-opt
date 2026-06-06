# Experiment 7 -- Metric repair ablation

> **Honesty note.** All `L` values are estimated by finite sampling of perturbation directions (a priori numerical estimator). They are empirical estimates, **not** proof certificates. Decisions labelled `certified_safe` are *analyzer* decisions resting on sampled `L`; they are not formal certificates. See `lipschitz_sampling_stress.md` for the quantified gap between sampled `L` and structured worst-case directions.

## Setup

Five metrics, identical fixed `L = max(L_clean, L_rewritten)`, on the scenarios where the original metric was weak. Only the displacement changes, so differences in bound/decision are attributable to the metric. Deltas below are means over models/seeds.

Exact command:

```
python scripts/run_sncs_v4_metric_ablation.py --models quadratic mnist_mlp
```

## Mean displacement (delta) by metric

| scenario | A raw(theta,M) | B +sched/time | C block-norm | D denom-aware v | E repaired |
| --- | --- | --- | --- | --- | --- |
| scheduler_mismatch | 0 | 0.02471 | 0 | 0 | 0.02471 |
| reset_second_moment | 1.578 | 1.578 | 3.781 | 1.557 | 1.557 |
| ema_loaded_as_parameters | 2.172 | 2.172 | 2.172 | 2.172 | 2.172 |
| stale_adam_moments | 1.66 | 1.66 | 3.901 | 1.661 | 0.06884 |
| mild_lr_decrease | 0 | 0.02471 | 0 | 0 | 0.02471 |

## Answers to the required questions

**1. Does the old metric give zero/misleading delta for scheduler mismatch?** YES -- the raw (theta,M) metric is **exactly 0** for `scheduler_mismatch` (a pure schedule change moves neither theta nor M), so its bound `delta*L^k = 0` is vacuous/misleading. The repaired metric assigns a theta-commensurate mean delta of **0.02471**.

**2. Does denominator-aware v geometry better reflect reset-v severity?** The raw metric scores `reset_second_moment` purely by buffer norm (mean **1.578**, in second-moment units, not comparable to divergence). The denominator-aware metric maps the v change into its theta-space step effect (mean **1.557**), which is directly comparable to the measured divergence and reflects that a v reset makes the next Adam step much larger.

**3. Does block normalization change classification?** Block normalization changes the analyzer decision in **0/30** cells relative to the raw metric -- it rescales moment vs parameter contributions, which moves borderline cases across the abstain/flag line.

**4. Which metric is used in the repaired analyzer?** **E (full_repaired)** = direct theta move + injected one-step update change (which subsumes the schedule, time and denominator-aware-v effects). It is the only metric that is both non-blind to schedule changes and in theta-commensurate units.

## Limitations

* Deltas across metrics are not on a single common scale by construction (that is the point); compare each metric to the measured divergence in the CSV.
* `L` is sampled and shared across metrics here to isolate the metric.
* See the `bound_holds` column for where each metric's bound is actually valid.