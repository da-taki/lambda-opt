# Experiment 5 -- Real CIFAR-10 checkpoint matrix

> **Honesty note.** All `L` values are estimated by finite sampling of perturbation directions (a priori numerical estimator). They are empirical estimates, **not** proof certificates. Decisions labelled `certified_safe` are *analyzer* decisions resting on sampled `L`; they are not formal certificates. See `lipschitz_sampling_stress.md` for the quantified gap between sampled `L` and structured worst-case directions.

## Setup

Model: ResNet-18 (~11.2M params). Data: **real CIFAR-10 train split (first 256 images)** (real torchvision CIFAR-10). 
**LIMITED run** (CPU budget): subset=256 images, batch=16, pretrain_steps=6, post_steps=8, gt_steps=12, seeds=[0, 1, 2]. This is the smallest honest real-CIFAR-10 version; it is **not** full training. Scale with the CLI flags on a GPU host.

Exact command:

```
python scripts/run_cifar10_checkpoint_matrix.py --seeds 0 1 2 --subset 256 --pretrain-steps 6 --post-steps 8
```

## Exact danger definition and thresholds

```
actual_label = dangerous  iff
      max_post_resume_divergence >= divergence_threshold
   or max_post_resume_loss_gap   >= loss_gap_threshold
```
divergence_threshold = 0.1 * ||theta_resume|| (per seed; see CSV column), loss_gap_threshold = 0.1.

## Confusion summary (all seeds)

| metric | value |
| --- | --- |
| total cases | 36 |
| actual dangerous | 27 |
| actual benign | 9 |
| true positives | 27 |
| true negatives | 9 |
| false positives | 0 |
| false negatives | 0 |
| abstentions | 0 |
| sensitivity | 1 |
| specificity | 1 |
| coverage rate | 1 |
| abstention rate | 0 |

## Reading

* Sensitivity (recall on dangerous) = **1.0**, specificity = **1.0**.
* False negatives = **0** (dangerous rewrite predicted safe); these are the unacceptable errors.
* False positives = **0** (benign rewrite flagged dangerous) -- the acceptable conservative cost.
* Whether this used full CIFAR-10 or a subset: **a real CIFAR-10 subset** (256 images), not full training. Labelled LIMITED.

## Limitations

* Short horizon: a few pre-train / post-resume steps. Divergence trends are real but small-scale.
* `L` sampled with a tiny budget (2x2) on 11M params; estimates are noisy -- this is why some moment-only rewrites draw conservative false positives.
* Thresholds are reported and swept in experiment 8.