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
│   ├── LambdaOpt/
│   │   ├── Basic.lean          # Core definitions: TrainingConfig, RewriteOp, Trajectory
│   │   ├── Theorem1.lean       # Single rewrite bound + proof
│   │   ├── Theorem2.lean       # Sequential composition bound + proof
│   │   ├── Theorem3.lean       # Decidability of ε-equivalence
│   │   ├── LipschitzLemmas.lean # Auxiliary trajectory Lipschitz lemmas
│   │   └── Instances.lean      # Decidable instances for rational arithmetic
│   ├── lakefile.toml           # Lean 4 build config, Mathlib pinned
│   └── lake-manifest.json
├── experiments/
│   ├── quadratic/              # 3-regime quadratic loss experiments (contractive/boundary/expansive)
│   ├── mlp_mnist/              # 52,650-param MLP on MNIST
│   ├── resnet18_cifar10/       # 11.18M-param ResNet-18 on CIFAR-10
│   ├── gptmini_wikitext2/      # 881K-param GPT-mini transformer on WikiText-2
│   └── stochastic/             # Theorem 4 validation, 4 noise levels × 50 trials
├── safety_case_study/
│   └── checkpoint_safety.py    # Pre-resume workflow: stale moments, scheduler mismatch, EMA
├── lipschitz_estimator/
│   └── estimator.py            # Numerical a priori L_pred estimation (20 trials × 15 steps)
├── results/
│   ├── plots/                  # All 12 result plots (Figures 1–10, 9A/9B/9C)
│   ├── summary.csv             # Full results table across all experiments
│   └── stochastic/             # Per-noise-level divergence and bound curves
├── tests/
│   └── test_suite.py           # 33 unit tests (all pass)
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
pip install torch torchvision numpy scipy matplotlib tqdm

# Quadratic loss validation (Theorems 1–3)
python experiments/quadratic/run_all.py

# Neural network experiments
python experiments/mlp_mnist/train.py
python experiments/resnet18_cifar10/train.py
python experiments/gptmini_wikitext2/train.py

# Checkpoint safety case study
python safety_case_study/checkpoint_safety.py

# Stochastic extension (Theorem 4)
python experiments/stochastic/run_stochastic.py
```

All experiments use fixed seeds (`torch.manual_seed(42)`, `numpy.random.seed(42)`). Hyperparameters are logged per-experiment. Full runtime on a single GPU: approximately 2–4 hours for all neural network experiments combined.

### Run tests

```bash
python -m pytest tests/test_suite.py -v
# Expected: 33/33 pass
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