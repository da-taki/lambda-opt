# Experiment 3 -- Step-varying bound comparison

> **Honesty note.** All `L` values are estimated by finite sampling of perturbation directions (a priori numerical estimator). They are empirical estimates, **not** proof certificates. Decisions labelled `certified_safe` are *analyzer* decisions resting on sampled `L`; they are not formal certificates. See `lipschitz_sampling_stress.md` for the quantified gap between sampled `L` and structured worst-case directions.

## Setup

Bound `D(t_R+k) <= delta * prod_{j<k} L_j`. Per-step `L_j` for the step-varying bound (D) are estimated **a-priori** along the known baseline continuation (the baseline is rewrite-independent). (E) uses the realised per-step ratios and is a diagnostic lower reference only.

Exact command:

```
python scripts/run_stepwise_bound_comparison.py --models quadratic mnist_mlp resnet18
```

## Aggregate over all scenarios/models/seeds

| bound type | cases | holds | certifies safe | abstains | vacuous (final) | median looseness (final) |
| --- | --- | --- | --- | --- | --- | --- |
| A. fixed clean L | 84 | 56 | 0 | 28 | 0 | 3.277 |
| B. fixed rewritten L | 84 | 53 | 0 | 29 | 0 | 17.1 |
| C. fixed max L | 84 | 56 | 0 | 28 | 0 | 26.14 |
| D. stepwise product | 84 | 56 | 0 | 28 | 0 | 2.955 |
| E. empirical envelope (diag) | 84 | 77 | 0 | 0 | 0 | 1.035 |

## Answers to the required questions

**1. Does the stepwise bound reduce vacuity compared with fixed L?** Fixed max-L (C) is vacuous on 0 cases with median looseness 26.138210263889924; the stepwise product (D) is vacuous on 0 with median looseness 2.9551333561393482. Lower looseness = tighter, less vacuous.

**2. How many abstentions become useful decisions (C -> D)?** 0 cases that abstain under fixed max-L become decisive under the stepwise product bound.

**3. Does it ever fail?** Bound-holds counts are in the table. The step-varying bound (D) is a valid upper bound only when each `L_j` actually bounds that step's contraction between the two trajectories; with sampled `L_j` this can fail when a probe misses a worst-case direction (see experiment 9). Any `holds < cases` entry is reported honestly rather than hidden.

**4. Does it make ResNet/GPT-mini bounds less absurd?** Compare the median looseness of C vs D restricted to large models in the CSV; the stepwise product collapses the `L^k` blow-up that makes fixed-L bounds on deep nets astronomically loose.

## Limitations

* Per-step `L_j` are sampled; (D) inherits the empirical caveat.
* (E) is diagnostic (uses realised ratios) and never used for decisions.
* For large models `L_j` is sampled every 2 steps to fit the CPU budget.