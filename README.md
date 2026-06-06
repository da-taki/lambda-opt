# λOpt — A Mechanized Calculus for Semantic Safety of Optimizer-State Rewrites

> *Can you know, before resuming training, whether a checkpoint restore will corrupt convergence? λOpt answers that question from the checkpoint alone.*

---

## What This Is

Modern deep learning training pipelines routinely modify optimizer state between steps: restoring checkpoints, reloading Adam moment buffers, inserting EMA shadows, reconfiguring schedulers, freezing LoRA adapters. These operations are **assumed to be safe**. That assumption is unexamined and frequently wrong.

λOpt is a minimal formal calculus in which optimizer state is modeled as explicit semantic state. Rewrites are **perturbation operators** with computable displacement bounds. Theorems bound how much a rewrite can shift the training trajectory — before you run it.

This repository contains:
- The full Lean 4 mechanization (6 files, 3,291 build jobs, zero errors, zero sorry axioms)
- The complete experimental codebase (quadratic losses, MLP, ResNet-18, GPT-mini transformer)
- A checkpoint safety case study on ResNet-18/CIFAR-10
- All results, plots, and hyperparameter configurations

---

## The Central Result

Let a training configuration be S = (θ, M, h, t): model parameters, optimizer auxiliary state (moments, EMA), hyperparameter schedule, discrete time. A rewrite operator R: S → S modifies state between steps. Its displacement is δ = ‖R(S) − S‖.

**Theorem 1 (Single Rewrite Safety Bound).** If the optimizer step function is L-Lipschitz and a rewrite R satisfies ‖R(S) − S‖ ≤ δ, then for all k ≥ 0:

```
D(t_R + k) = ‖τ^R(t_R + k) − τ(t_R + k)‖ ≤ L^k · δ
```

When L < 1 (contractive regime, typical for well-tuned Adam), perturbations decay exponentially. When L > 1, the bound correctly predicts growth. Both are computable before resume.

**Theorem 2** (Sequential Composition): For n sequential rewrites, the total bound is the sum of individual terms — additive, not multiplicative.

**Theorem 3** (Decidability): ε-trajectory equivalence is decidable in finite arithmetic given computable L, δ, and tolerance ε.

**Theorem 4** (Stochastic Extension): In the contractive regime with bounded gradient noise σ, expected post-rewrite divergence converges to a noise floor σ/(1 − L).

Theorems 1–3 are **fully mechanized in Lean 4** against Mathlib. Theorem 4 is empirically validated; Lean mechanization is future work pending Mathlib martingale support.

---

## Key Empirical Results

| Metric | Result |
|--------|--------|
| δ-linearity (R²) | **1.0000** |
| Loss-divergence Spearman ρ | **0.942** |
| Bound tightness range | [0.02, 0.96] |
| Lean 4 build | 3,291 jobs, **0 errors, 0 sorry** |
| Unit tests | **33/33 pass** |
| ResNet-18 bound holds | D(t) ≤ B(t) confirmed |
| Checkpoint safety false negatives | **0 / 3 scenarios** |
| GPT-mini transformer bound holds | D(t) ≤ B(t) confirmed |
| Stochastic bound (σ ≤ 0.005) | Confirmed, 50 trials each |

The same rewrite applied to Adam vs. SGD on identical loss landscapes produces qualitatively different outcomes — λOpt correctly predicts both. Adam (L_pred = 1.039): divergence decays to near-zero over 250 steps. SGD (L_pred = 1.188, exact from Hessian): divergence grows by 10 orders of magnitude over 150 steps.

---

## Concrete Rewrites Instantiated

**R_EMA** (EMA insertion): R(θ, M, h, t) = (αθ_ema + (1 − α)θ, M, h, t), δ = α‖θ_ema − θ‖

**R_ckpt** (Checkpoint restore): R(θ, M, h, t) = (θ, M′, h, t′) where M′ is a stale moment snapshot, δ = ‖M′ − M‖

**R_LoRA** (LoRA freeze): R(θ, M, h, t) = (θ, Π_F M, h, t) where Π_F zeros moment rows for frozen indices F, δ = ‖(I − Π_F)M‖

---

## Project Structure

```
lambda-opt/
├── lean/
│   ├── LambdaOpt/              # Core definitions and mechanized theorems
│   ├── LambdaOpt.lean
│   ├── lakefile.lean           # Lean 4 build config, Mathlib pinned
│   └── lake-manifest.json
├── experiments/
│   ├── src/                    # Training state, step rules, rewrites, bounds, metrics
│   ├── scripts/                # Reproducible experiment entry points
│   ├── configs/                # Quadratic, MLP, ResNet-18, GPT-mini configs
│   ├── tests/                  # Lightweight Python tests
│   ├── results/                # JSON/CSV experiment outputs
│   └── figures/                # Generated plots
└── README.md
```

---

## Quickstart

### Run the Lean 4 proofs

```bash
cd lean/
lake build
# Expected: 3,291 build jobs, 0 errors, 0 sorry axioms
```

Requires Lean 4.29.0 and Mathlib at the commit pinned in `lakefile.toml`. Install Lean via [elan](https://github.com/leanprover/elan).

### Run experiments

```bash
cd experiments/
pip install -r requirements.txt

# Quadratic loss validation (Theorems 1–3)
python scripts/run_single_rewrite.py
python scripts/run_composition.py
python scripts/run_delta_scaling.py
python scripts/run_separation.py
python scripts/run_apriori_vs_actual.py

# Neural network experiments
python scripts/run_neural_net.py --config configs/mlp_mnist.yaml
python scripts/run_neural_net.py --config configs/resnet18_cifar10.yaml
python scripts/run_transformer.py --config configs/gpt_mini_wikitext2.yaml

# Checkpoint safety case study
python scripts/run_checkpoint_safety.py

# Stochastic extension (Theorem 4)
python scripts/run_stochastic_validation.py
```

### Additional diagnostic experiments

The following scripts extend the empirical section with targeted diagnostics. They save CSV/JSON outputs under `experiments/results/<experiment>/` and figures under `experiments/figures/<experiment>/`.

```bash
cd experiments/

python scripts/run_directional_delta.py
python scripts/run_checkpoint_matrix.py
python scripts/run_optimizer_comparison.py
python scripts/run_bound_tightening_ablation.py
python scripts/run_adam_noise_floor.py
python scripts/run_checkpoint_lipschitz_fix.py
python scripts/run_lipschitz_budget_ablation.py
```

The checkpoint matrix default uses ResNet-18 on synthetic CIFAR-shaped data for a CPU-friendly smoke benchmark; use `--full` for the full CIFAR-10 severity matrix. The checkpoint Lipschitz-fix script shows that severe `v` resets require estimating `L` at the rewritten state or using `max(L_clean, L_rewritten)`. The empirical-envelope bound in `run_bound_tightening_ablation.py` is diagnostic only, because it uses observed post-rewrite ratios. The Adam noise-floor script uses an unsafetied default `L` so that the stochastic floor `σ/(1-L)` remains defined.

All experiments use fixed seeds by default. Hyperparameters are logged per-experiment. Full neural-network runs can be substantially slower than the quadratic defaults.

### Run tests

```bash
cd experiments/
python -m pytest tests -q
```

---

## The A Priori Lipschitz Estimator

The core practical contribution is the ability to compute L **before** running the rewritten trajectory. The estimator:

1. At the rewrite point S_{t_R}, applies 20 small perturbations of magnitude ε = 10⁻⁴
2. Runs each perturbed trajectory for 15 steps
3. Computes the geometric mean per-step contraction rate
4. Applies a safety margin of 1.03–1.08×

This captures Adam's adaptive self-correction, which linearized spectral analysis systematically mischaracterizes. The estimator places Adam in the contractive regime (L < 1) even on nominally expansive loss landscapes (κ = 100, lr = 0.05), matching empirical L_actual across all 21 test cases.

---

## Why This Matters

The standard engineering approach to checkpoint safety is to resume training and watch the loss curve. If something went wrong, you find out after wasting compute. λOpt replaces this with a **pre-flight check**: given a checkpoint and a proposed rewrite, compute δ and L_pred, check whether δ · L^T ≤ ε, and decide before running.

The framework has zero false negatives across all tested failure scenarios. Conservative false positives (an unnecessary re-examination) are the acceptable cost. Silent training corruption is not.

---

## Paper

**λOpt: A Mechanized Calculus for Semantic Safety of Optimizer-State Rewrites**  
Taknoor Singh — Independent Research, Patiala, Punjab, India  
Under review at *Springer Machine Learning*

```
@misc{singh2025lambdaopt,
  title     = {λOpt: A Mechanized Calculus for Semantic Safety of Optimizer-State Rewrites},
  author    = {Singh, Taknoor},
  year      = {2025},
  note      = {Under review, Springer Machine Learning}
}
```

---

## Reproducibility

All experiments are fully reproducible from this repository. Fixed seeds, logged hyperparameters, and the Lean 4 build environment (version 4.29.0, Mathlib commit in `lake-manifest.json`) together guarantee bit-for-bit reproducibility of all reported results.

If anything doesn't reproduce, open an issue.

---

## SNCS v4 Reviewer-Proofing Experiments

The `scripts/` directory (top level) and `results/sncs_v4/` hold a suite of
reviewer-proofing experiments added on the `sncs-v4-reviewer-experiments` branch.
They stress-test the analyzer against simpler baselines, quantify coverage and
runtime, ablate the displacement metric, sweep thresholds, and probe the honesty
limits of the sampled-L estimator. Each experiment writes **CSV + JSON +
Markdown** into `results/sncs_v4/`, and every Markdown file documents the setup,
exact command, metrics, failures, and limitations.

> **Honesty note.** `L` is estimated by finite sampling of perturbation
> directions. It is an *empirical* estimate, not a proof certificate; decisions
> labelled `certified_safe` are *analyzer* decisions. See
> `results/sncs_v4/lipschitz_sampling_stress.md` for the quantified gap.

Shared infrastructure lives in `scripts/sncs_v4_common.py` (the four diagnostics
— quadratic, MNIST MLP, ResNet-18/CIFAR-10, GPT-mini — the 12 canonical
checkpoint-rewrite scenarios, the five metric variants, and the repaired
analyzer). The repaired analyzer uses a theta-commensurate displacement and
`L = max(L_clean, L_rewritten)`.

```bash
# from the repo root, with the project venv active and PYTHONPATH=experiments
python scripts/run_sncs_v4_baseline_comparison.py          # Exp 1: baselines
python scripts/run_sncs_v4_certification_coverage.py        # Exp 2: coverage
python scripts/run_sncs_v4_stepwise_bound_comparison.py     # Exp 3: step-varying bound
python scripts/run_sncs_v4_multiseed_validation.py          # Exp 4: multiple seeds
python scripts/run_sncs_v4_cifar10_checkpoint_matrix.py     # Exp 5: real CIFAR-10 matrix
python scripts/run_sncs_v4_runtime_overhead.py              # Exp 6: runtime overhead
python scripts/run_sncs_v4_metric_ablation.py               # Exp 7: metric repair ablation
python scripts/run_sncs_v4_threshold_sensitivity.py         # Exp 8: threshold/ROC sweep
python scripts/run_sncs_v4_lipschitz_sampling_stress.py     # Exp 9: sampled-L stress
python scripts/run_sncs_v4_make_summary.py                  # Exp 11: tables + summary
```

Experiment 10 (Lean theorem mapping) is documented in
`results/sncs_v4/lean_theorem_mapping.md`, with a candidate step-varying product
bound in `lean/LambdaOpt/StepwiseProduct.lean`
(statement also in `results/sncs_v4/lean_stepwise_todo.md`). The headline summary
and manuscript-ready tables are in
`results/sncs_v4/SNCS_V4_REVIEWER_PROOFING_SUMMARY.md` and
`results/sncs_v4/sncs_v4_tables_for_paper.md`.

ResNet-18 and GPT-mini runs default to small CPU-budget configurations (real data,
short horizons), clearly labelled "LIMITED" in their outputs; scale them up with
the documented CLI flags on a GPU host.
