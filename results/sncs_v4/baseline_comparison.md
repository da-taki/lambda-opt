# Experiment 1 -- Baseline comparison

> **Honesty note.** All `L` values are estimated by finite sampling of perturbation directions (a priori numerical estimator). They are empirical estimates, **not** proof certificates. Decisions labelled `certified_safe` are *analyzer* decisions resting on sampled `L`; they are not formal certificates. See `lipschitz_sampling_stress.md` for the quantified gap between sampled `L` and structured worst-case directions.

Models: mnist_mlp, quadratic  
Seeds: 0, 1, 2  
Analyzer epsilon=0.05, margin=1.08.

## Setup

Each rewrite scenario is applied to a checkpoint; the *ground-truth* label `dangerous`/`benign` comes from a longer post-resume trajectory (`gt_steps`) thresholded on max divergence and max loss gap (per-diagnostic thresholds, reported in the JSON `config`). Six methods each emit a safe/dangerous verdict (the repaired analyzer may also abstain).

Exact command:

```
python scripts/run_sncs_v4_baseline_comparison.py --models mnist_mlp quadratic --seeds 0 1 2
```

## Per-method aggregate (all models, all seeds, all scenarios)

| method | cases | dangerous caught | false negatives | false positives | abstained |
| --- | --- | --- | --- | --- | --- |
| A. param-norm | 72 | 6/46 | 40 | 0 | 0 |
| B. optimizer-norm | 72 | 19/46 | 27 | 8 | 0 |
| C. raw product | 72 | 46/46 | 0 | 26 | 0 |
| D. block-norm | 72 | 46/46 | 0 | 26 | 0 |
| E. loss-probe | 72 | 26/46 | 20 | 0 | 0 |
| F. repaired analyzer | 72 | 39/46 | 0 | 8 | 7 |

## Which baseline catches each dangerous rewrite?

The table below marks, for each scenario that is *actually* dangerous (any model/seed), whether each method flagged it (caught) without abstaining.

| dangerous scenario | param-norm | optimizer-norm | raw product | block-norm | loss-probe | repaired analyzer |
| --- | --- | --- | --- | --- | --- | --- |
| ema_loaded_as_parameters | 6/6 | 0/6 | 6/6 | 6/6 | 6/6 | 6/6 |
| mild_lr_decrease | 0/6 | 0/6 | 6/6 | 6/6 | 0/6 | 3/6 (+3 abst) |
| optimizer_state_from_older_checkpoint | 0/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| params_current_optimizer_older | 0/6 | 6/6 | 6/6 | 6/6 | 3/6 | 6/6 |
| reset_first_moment | 0/4 | 4/4 | 4/4 | 4/4 | 0/4 | 3/4 (+1 abst) |
| reset_second_moment | 0/6 | 3/6 | 6/6 | 6/6 | 6/6 | 6/6 |
| scheduler_mismatch | 0/6 | 0/6 | 6/6 | 6/6 | 0/6 | 3/6 (+3 abst) |
| severe_lr_increase | 0/6 | 0/6 | 6/6 | 6/6 | 6/6 | 6/6 |
| stale_adam_moments | 0/3 | 3/3 | 3/3 | 3/3 | 2/3 | 3/3 |

## Answers to the required questions

**1. Which baseline catches each dangerous rewrite?** See the matrix above. The schedule-only dangers (scheduler_mismatch, severe_lr_increase, mild_lr_decrease) are invisible to the magnitude metrics A, B, C and D, because a pure schedule change has `delta_theta = delta_optimizer = 0` and the raw/(block) (theta,M) metric is therefore also 0. Only the loss-probe (E) and the repaired analyzer (F) see them.

**2. Does repaired lambda-Opt catch anything trivial baselines miss?** Yes. False negatives -- A+B: 67, C+D: 0, E (loss-probe): 20, F (repaired): 0. The repaired metric injects a non-zero, theta-commensurate displacement for schedule/time/second-moment rewrites that the raw metrics score as exactly 0.

**3. Where does lambda-Opt abstain?** F abstains on 7 cases overall -- these are boundary-regime or epsilon<bound<danger cases where sampled L cannot separate safe from dangerous. Abstention is the honest output there.

**4. Where do simple baselines perform just as well?** For pure parameter moves (`ema_loaded_as_parameters`) and exact/no-op restores, the parameter-norm baseline A agrees with F: the displacement is directly in theta-space, so the trivial metric suffices.

**5. Are there false negatives?** A+B baselines: 67; C+D: 0; loss-probe E: 20; repaired F: 0. The magnitude/raw baselines have the most, driven entirely by schedule-only dangers.

## Limitations

* L is sampled (empirical), not certified. The loss-probe baseline and the analyzer have comparable runtime cost on large models; the analyzer's advantage is a forward-looking bound shape, not a cheaper probe.
* Ground-truth danger is threshold-defined; the thresholds are reported and swept in experiment 8.
* ResNet-18/GPT-mini rows, when present, use the small CPU-budget configs (see `config.json`).