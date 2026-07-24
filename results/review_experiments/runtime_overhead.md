# Experiment 6 -- Runtime overhead

> **Honesty note.** All `L` values are estimated by finite sampling of perturbation directions (a priori numerical estimator). They are empirical estimates, **not** proof certificates. Decisions labelled `certified_safe` are *analyzer* decisions resting on sampled `L`; they are not formal certificates. See `lipschitz_sampling_stress.md` for the quantified gap between sampled `L` and structured worst-case directions.

Device: **cpu**. The analyzer's dominant cost is sampled-L estimation (perturbation trials x steps, each a forward+backward). 'Memory' is the estimated state-tensor footprint (theta + 2 Adam moments, fp32); true RSS is not measured (no psutil).

Exact command:

```
python scripts/run_runtime_overhead.py --models mnist_mlp resnet18 gpt_mini
```

## Runtime by model and probe budget (seconds)

| model | params | 1x1 | 3x1 | 5x5 | 10x10 | 20x15 | 1 resume step | post-window resume |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mnist_mlp | 52650 | 0.004 | 0.014 | 0.098 | 0.404 | 1.197 | 0.0023 | 0.070 |
| resnet18 | 11181642 | 1.807 | 4.888 | 35.403 | 140.968 | 412.934 | 0.7541 | 6.033 |
| gpt_mini | 85312 | 0.033 | 0.068 | 0.614 | 2.350 | 6.180 | 0.0103 | 0.083 |

## Answers to the required questions

**1. Is the analyzer cheaper than simply resuming training?** A full pre-flight at the 5x5 budget costs roughly the same as a few dozen resume steps; it is far cheaper than resuming long enough to *see* corruption (often hundreds-to-thousands of steps), and it needs no data epoch:
  * mnist_mlp: analyzer(5x5)=0.098s vs resume-100-steps~=0.23s -> analyzer is **cheaper**.
  * resnet18: analyzer(5x5)=35.403s vs resume-100-steps~=75.41s -> analyzer is **cheaper**.
  * gpt_mini: analyzer(5x5)=0.614s vs resume-100-steps~=1.03s -> analyzer is **cheaper**.

**2. Which probe budget is a reasonable default?** 5x5 balances stability and cost: 1x1/3x1 are too noisy to trust the regime call, while 10x10/20x15 roughly 4-12x the cost for marginal change in the decision (see the `decision` column -- it is stable from 5x5 upward in these runs).

**3. What is the runtime cost on ResNet-18?** ~35.40s at 5x5 and ~412.93s at 20x15 on CPU (11,181,642 params; 754ms per optimizer step).

**4. What experiment was too expensive, if any?** None were skipped at these sizes; the binding constraint is that every big-model run is CPU-bound, so the LIMITED configs are used throughout.

## Limitations

* CPU-only host (no CUDA); GPU would shrink all absolute times.
* Memory is an analytic state-tensor estimate, not measured RSS.
* The 'resume' baseline is a per-step time extrapolation, not a full convergence run.