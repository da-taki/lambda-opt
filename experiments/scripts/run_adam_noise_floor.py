"""Adam-adjusted stochastic noise-floor experiment.

Compares vanilla sigma/(1-L) against Adam effective-learning-rate floors.
Noise is injected per coordinate into the gradient; divergence is measured
as an L2 norm, so sqrt(dim)-adjusted floors are also reported.

Usage:
    cd experiments
    py scripts/run_adam_noise_floor.py
"""

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import (
    AdamStep,
    TrainingState,
    apriori_lipschitz_numerical,
    make_constant_schedule,
    make_quadratic_hessian,
    make_quadratic_loss,
    run_trajectory,
)
from src.bounds import classify_regime
from src.rewrites import Rewrite


SAFETY = 1.0
DEFAULT_SIGMAS = [0.001, 0.005, 0.01, 0.025, 0.05, 0.1]


class ParameterPerturbationRewrite(Rewrite):
    name = "ParameterPerturbation"

    def __init__(self, perturbation: torch.Tensor):
        self.perturbation = perturbation

    def apply(self, state: TrainingState) -> TrainingState:
        return TrainingState(
            theta=state.theta + self.perturbation,
            moments=state.moments.clone(),
            schedule_fn=state.schedule_fn,
            t=state.t,
        )

    def delta(self, state: TrainingState) -> float:
        return self.perturbation.norm(2).item()


def parse_sigmas(value):
    if isinstance(value, list):
        return value
    return [float(v.strip()) for v in value.split(",") if v.strip()]


def make_initial_state(args, schedule):
    torch.manual_seed(args.seed)
    return TrainingState(
        theta=torch.randn(args.dimension) * 0.1,
        moments=torch.zeros(args.dimension, 2),
        schedule_fn=schedule,
        t=0,
    )


def make_rewrite(args, state_at_tR):
    torch.manual_seed(args.seed + 1000)
    direction = torch.randn(args.dimension)
    direction = direction / direction.norm(2)
    return ParameterPerturbationRewrite(direction * args.delta)


def eta_eff_stats(state, args):
    hp = state.schedule_fn(state.t)
    b2 = hp["beta2"]
    t_bc = max(state.t, 1)
    v_hat = state.moments[:, 1] / max(1 - b2 ** t_bc, 1e-30)
    eta_eff = hp["lr"] / (v_hat.clamp(min=0.0).sqrt() + hp["eps"])
    finite = eta_eff[torch.isfinite(eta_eff)]
    if finite.numel() == 0:
        return {"mean": float("inf"), "median": float("inf")}
    return {
        "mean": finite.mean().item(),
        "median": finite.median().item(),
    }


def run_noisy_trial(args, s_at_tR, rewrite, step_fn, loss_fn, sigma, seed):
    torch.manual_seed(seed)
    s_base = s_at_tR.clone()
    s_rw = rewrite.apply(s_at_tR)
    divergence = []

    for _ in range(args.total_steps - args.rewrite_step):
        divergence.append((s_rw.theta - s_base.theta).norm(2).item())
        _, grad_base = loss_fn(s_base.theta)
        _, grad_rw = loss_fn(s_rw.theta)
        noise_base = torch.randn_like(grad_base) * sigma
        noise_rw = torch.randn_like(grad_rw) * sigma
        s_base, _ = step_fn(s_base, grad_base + noise_base)
        s_rw, _ = step_fn(s_rw, grad_rw + noise_rw)

    divergence.append((s_rw.theta - s_base.theta).norm(2).item())
    return divergence


def relative_error(predicted, empirical):
    if not math.isfinite(predicted):
        return float("inf")
    return abs(predicted - empirical) / max(abs(empirical), 1e-30)


def prediction_floors(sigma, L_pred, eta_stats, args):
    if L_pred >= 1.0:
        denom = max(1.0 - min(L_pred, 0.999999), 1e-12)
    else:
        denom = 1.0 - L_pred
    sqrt_dim = math.sqrt(args.dimension)
    vanilla = sigma / denom
    adj_mean = eta_stats["mean"] * sigma / denom
    adj_median = eta_stats["median"] * sigma / denom
    return {
        "vanilla_floor": vanilla,
        "adam_adjusted_floor_mean_eta": adj_mean,
        "adam_adjusted_floor_median_eta": adj_median,
        "dimension_adjusted_floor_mean_eta": sqrt_dim * adj_mean,
        "dimension_adjusted_floor_median_eta": sqrt_dim * adj_median,
    }


def best_model(row):
    candidates = {
        "vanilla": row["relative_error_vanilla"],
        "adam_adjusted_mean_eta": row["relative_error_adjusted_mean_eta"],
        "adam_adjusted_median_eta": row["relative_error_adjusted_median_eta"],
        "dimension_adjusted_mean_eta": row["relative_error_dimension_adjusted_mean_eta"],
        "dimension_adjusted_median_eta": row["relative_error_dimension_adjusted_median_eta"],
    }
    return min(candidates, key=candidates.get)


def write_csv(path, rows, fields):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{k: row[k] for k in fields} for row in rows])


def plot_outputs(metrics, curves, figure_dir):
    os.makedirs(figure_dir, exist_ok=True)
    sigmas = [row["sigma"] for row in metrics]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sigmas, [row["empirical_floor_mean"] for row in metrics], "ko-", label="empirical mean")
    ax.plot(sigmas, [row["vanilla_floor"] for row in metrics], "r--o", label="vanilla")
    ax.plot(sigmas, [row["adam_adjusted_floor_mean_eta"] for row in metrics], "b--o", label="Adam adj mean eta")
    ax.plot(sigmas, [row["dimension_adjusted_floor_mean_eta"] for row in metrics], "g--o", label="sqrt(dim) Adam adj")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("sigma")
    ax.set_ylabel("noise floor")
    ax.set_title("Empirical vs predicted noise floor")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(Path(figure_dir) / "empirical_vs_predicted_floor.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sigmas, [row["relative_error_vanilla"] for row in metrics], "r-o", label="vanilla")
    ax.plot(sigmas, [row["relative_error_adjusted_mean_eta"] for row in metrics], "b-o", label="Adam adj mean eta")
    ax.plot(sigmas, [row["relative_error_adjusted_median_eta"] for row in metrics], "c-o", label="Adam adj median eta")
    ax.plot(sigmas, [row["relative_error_dimension_adjusted_mean_eta"] for row in metrics], "g-o", label="sqrt(dim) mean eta")
    ax.plot(sigmas, [row["relative_error_dimension_adjusted_median_eta"] for row in metrics], "m-o", label="sqrt(dim) median eta")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("sigma")
    ax.set_ylabel("relative error")
    ax.set_title("Relative error by sigma")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(Path(figure_dir) / "relative_error_by_sigma.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    grouped = {}
    for row in curves:
        grouped.setdefault(row["sigma"], []).append(row)
    for sigma, rows in grouped.items():
        rows = sorted(rows, key=lambda r: r["step"])
        ax.plot(
            [r["step"] for r in rows],
            [r["mean_divergence"] for r in rows],
            lw=1.5,
            label=f"sigma={sigma:g}",
        )
    ax.set_yscale("log")
    ax.set_xlabel("post-rewrite step")
    ax.set_ylabel("mean D(t)")
    ax.set_title("Divergence curves by sigma")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(Path(figure_dir) / "divergence_curves_by_sigma.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(sigmas, [row["eta_eff_mean"] for row in metrics], "bo-", label="mean eta_eff")
    ax.plot(sigmas, [row["eta_eff_median"] for row in metrics], "go-", label="median eta_eff")
    ax.set_xscale("log")
    ax.set_xlabel("sigma")
    ax.set_ylabel("eta_eff")
    ax.set_title("Adam effective learning rate")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(Path(figure_dir) / "eta_eff_by_sigma.png", dpi=180)
    plt.close(fig)


def run(args):
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.figure_dir, exist_ok=True)

    sigmas = parse_sigmas(args.sigmas)
    schedule = make_constant_schedule(lr=args.lr, beta1=args.beta1, beta2=args.beta2, eps=args.eps)
    A = make_quadratic_hessian(args.dimension, args.condition_number, args.seed)
    loss_fn = make_quadratic_loss(A)
    step_fn = AdamStep()

    s0 = make_initial_state(args, schedule)
    baseline = run_trajectory(s0.clone(), step_fn, loss_fn, args.total_steps)
    s_at_tR = baseline.states[args.rewrite_step]
    rewrite = make_rewrite(args, s_at_tR)
    delta = rewrite.delta(s_at_tR)
    eta_stats = eta_eff_stats(s_at_tR, args)

    L_raw = apriori_lipschitz_numerical(
        s_at_tR,
        step_fn,
        loss_fn,
        n_perturbations=args.lipschitz_perturbations,
        n_steps=args.lipschitz_steps,
        eps=args.lipschitz_eps,
    )
    L_pred = L_raw * args.safety_margin
    regime = classify_regime(L_pred)
    print(f"L_pred={L_pred:.6f} [{regime}], delta={delta:.6f}")
    print(f"eta_eff mean={eta_stats['mean']:.6g}, median={eta_stats['median']:.6g}")

    metrics = []
    curve_rows = []
    empirical_means = []

    for sigma in sigmas:
        all_div = []
        for trial in range(args.trials):
            div = run_noisy_trial(
                args,
                s_at_tR,
                rewrite,
                step_fn,
                loss_fn,
                sigma,
                seed=args.seed + trial * 1009 + int(sigma * 1_000_000),
            )
            all_div.append(div)

        arr = np.array(all_div)
        mean_curve = arr.mean(axis=0)
        std_curve = arr.std(axis=0)
        floor_window = arr[:, -min(args.floor_window, arr.shape[1]):]
        per_trial_floor = floor_window.mean(axis=1)
        emp_mean = float(per_trial_floor.mean())
        emp_std = float(per_trial_floor.std())
        emp_median = float(np.median(per_trial_floor))
        empirical_means.append(emp_mean)

        floors = prediction_floors(sigma, L_pred, eta_stats, args)
        row = {
            "sigma": sigma,
            "trials": args.trials,
            "L_pred": L_pred,
            "L_raw": L_raw,
            "regime": regime,
            "delta": delta,
            "eta_eff_mean": eta_stats["mean"],
            "eta_eff_median": eta_stats["median"],
            "empirical_floor_mean": emp_mean,
            "empirical_floor_std": emp_std,
            "empirical_floor_median": emp_median,
            **floors,
        }
        row.update({
            "relative_error_vanilla": relative_error(row["vanilla_floor"], emp_mean),
            "relative_error_adjusted_mean_eta": relative_error(row["adam_adjusted_floor_mean_eta"], emp_mean),
            "relative_error_adjusted_median_eta": relative_error(row["adam_adjusted_floor_median_eta"], emp_mean),
            "relative_error_dimension_adjusted_mean_eta": relative_error(row["dimension_adjusted_floor_mean_eta"], emp_mean),
            "relative_error_dimension_adjusted_median_eta": relative_error(row["dimension_adjusted_floor_median_eta"], emp_mean),
        })
        row["best_model_by_relative_error"] = best_model(row)
        metrics.append(row)

        for step in range(mean_curve.shape[0]):
            curve_rows.append({
                "sigma": sigma,
                "step": step,
                "mean_divergence": float(mean_curve[step]),
                "std_divergence": float(std_curve[step]),
            })

        print(
            f"sigma={sigma:g} empirical={emp_mean:.6g} "
            f"vanilla_err={row['relative_error_vanilla']:.3g} "
            f"dim_adj_err={row['relative_error_dimension_adjusted_mean_eta']:.3g} "
            f"best={row['best_model_by_relative_error']}"
        )

    monotonic = all(b >= a for a, b in zip(empirical_means, empirical_means[1:]))
    for row in metrics:
        row["monotonic_empirical_floor"] = monotonic

    metric_fields = [
        "sigma", "trials", "L_pred", "delta",
        "empirical_floor_mean", "empirical_floor_std", "empirical_floor_median",
        "vanilla_floor", "adam_adjusted_floor_mean_eta",
        "adam_adjusted_floor_median_eta", "dimension_adjusted_floor_mean_eta",
        "dimension_adjusted_floor_median_eta", "relative_error_vanilla",
        "relative_error_adjusted_mean_eta", "relative_error_adjusted_median_eta",
        "relative_error_dimension_adjusted_mean_eta",
        "relative_error_dimension_adjusted_median_eta",
        "best_model_by_relative_error", "monotonic_empirical_floor",
        "eta_eff_mean", "eta_eff_median",
    ]
    curve_fields = ["sigma", "step", "mean_divergence", "std_divergence"]
    out = Path(args.output_dir)
    write_csv(out / "noise_floor_metrics.csv", metrics, metric_fields)
    write_csv(out / "noise_curves.csv", curve_rows, curve_fields)
    with open(out / "noise_floor_metrics.json", "w") as f:
        json.dump({"metrics": metrics, "curves": curve_rows}, f, indent=2)
    summary = {
        "sigmas": sigmas,
        "trials": args.trials,
        "L_pred": L_pred,
        "regime": regime,
        "delta": delta,
        "eta_eff_mean": eta_stats["mean"],
        "eta_eff_median": eta_stats["median"],
        "monotonic_empirical_floor": monotonic,
        "best_model_counts": {
            name: sum(1 for row in metrics if row["best_model_by_relative_error"] == name)
            for name in sorted(set(row["best_model_by_relative_error"] for row in metrics))
        },
    }
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    config = vars(args).copy()
    config.update({
        "sigmas": sigmas,
        "noise_scale_note": "sigma is per-coordinate gradient noise; divergence is L2 norm.",
        "dimension_adjusted_note": "sqrt(dim) variants account for L2 aggregation of independent per-coordinate noise.",
    })
    with open(out / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    plot_outputs(metrics, curve_rows, args.figure_dir)
    print(f"\nSaved metrics -> {out}")
    print(f"Saved figures -> {Path(args.figure_dir)}")
    return {"metrics": metrics, "summary": summary}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--trials", type=int, default=20)
    p.add_argument("--sigmas", default=",".join(str(v) for v in DEFAULT_SIGMAS))
    p.add_argument("--delta", type=float, default=0.01)
    p.add_argument("--rewrite-step", type=int, default=250)
    p.add_argument("--total-steps", type=int, default=500)
    p.add_argument("--dimension", type=int, default=10)
    p.add_argument("--condition-number", type=float, default=10.0)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--safety-margin", type=float, default=SAFETY)
    p.add_argument("--lipschitz-perturbations", type=int, default=10)
    p.add_argument("--lipschitz-steps", type=int, default=8)
    p.add_argument("--lipschitz-eps", type=float, default=1e-4)
    p.add_argument("--floor-window", type=int, default=50)
    p.add_argument("--output-dir", default="results/adam_noise_floor")
    p.add_argument("--figure-dir", default="figures/adam_noise_floor")
    args = p.parse_args()
    if args.trials <= 0:
        raise ValueError("--trials must be positive")
    if args.rewrite_step < 0 or args.rewrite_step > args.total_steps:
        raise ValueError("--rewrite-step must be between 0 and --total-steps")
    if args.delta <= 0:
        raise ValueError("--delta must be positive")
    parse_sigmas(args.sigmas)
    return args


if __name__ == "__main__":
    run(parse_args())
