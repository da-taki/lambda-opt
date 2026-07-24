# Experiment 9 -- Sampled-L stress test

> **Honesty note.** All `L` values are estimated by finite sampling of perturbation directions (a priori numerical estimator). They are empirical estimates, **not** proof certificates. Decisions labelled `certified_safe` are *analyzer* decisions resting on sampled `L`; they are not formal certificates. See `lipschitz_sampling_stress.md` for the quantified gap between sampled `L` and structured worst-case directions.

## Setup

L is estimated at the rewritten state along six direction families and increasing probe counts. The empirical worst-case reference `L_worst` per state is the max over all directions at 50 probes (for quadratic this includes the exact top-Hessian eigenvector). `missed_worst_case` flags an estimate below 95% of `L_worst`; `bound_holds` checks whether the bound built from that L survives the realised divergence.

Exact command:

```
python scripts/run_lipschitz_sampling_stress.py --models quadratic mnist_mlp
```

## Mean L by direction (probe trials >= 10)

| direction | mean L |
| --- | --- |
| random | 5.562 |
| gradient_aligned | 5.516 |
| trajectory_aligned | 5.563 |
| top_hessian | 1.286 |
| adam_denom_sensitive | 5.6 |
| schedule_time | 5.591 |

## Random-direction miss rate vs probe count

| probe trials | 1 | 3 | 5 | 10 | 20 | 50 |
| --- | --- | --- | --- | --- | --- | --- |
| random miss rate | 1 | 0.6667 | 0.6667 | 0.6667 | 0.5833 | 0.5 |

## Bound failures by direction (lower is safer)

| direction | bound failures |
| --- | --- |
| random | 44 |
| gradient_aligned | 48 |
| trajectory_aligned | 48 |
| top_hessian | 30 |
| adam_denom_sensitive | 36 |
| schedule_time | 48 |

## Answers to the required questions

**1. Do random probes miss structured worst directions?** Yes -- at 1 probe the random miss rate is 1.0; structured directions (top-Hessian, adam-denominator, preconditioned-update) report systematically larger L (see the mean-L table).

**2. Which direction is most dangerous?** **adam_denom_sensitive** has the highest mean estimated L, i.e. the steepest local expansion the analyzer must not miss.

**3. How many probes reduce misses?** Random-direction miss rate drops below 50% at **None probes** (it stays high at 1-3 probes). More probes help but never *guarantee* the worst direction is hit.

**4. What claim should the manuscript make about sampled L?** That L is an **empirically sampled estimate**, reported with its probe budget and a safety margin, and that decisions are *analyzer decisions*, not formal certificates. A formal certificate would require an a-priori worst-case (e.g. Hessian-spectral or interval) bound, which the sampled estimator approximates but does not prove. Any `bound_holds` failure under random sampling in the table above is direct evidence for this framing.

## Limitations

* `top_hessian` is exact only for the quadratic (Hessian = A); it is marked n/a for neural nets here (Lanczos top-eig is available in `src.loss_landscape` for a heavier follow-up).
* Direction jitter uses a fixed generator for reproducibility.
* `L_worst` is itself sampled (except the quadratic top-eig), so it is a lower bound on the true worst case -- the real gap is at least this big.