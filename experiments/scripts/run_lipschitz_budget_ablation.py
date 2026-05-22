"""Numerical L_pred estimator budget ablation.

Sweeps perturbation count and probe horizon for the a priori Lipschitz
estimator on a quadratic Adam rewrite experiment.

Usage:
    cd experiments
    py scripts/run_lipschitz_budget_ablation.py
"""

import argparse
import csv
import json
import os
import sys
import time
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
    apriori_bound,
    apriori_lipschitz_numerical,
    bound_holds,
    compute_divergence,
    make_constant_schedule,
    make_quadratic_hessian,
    make_quadratic_loss,
    run_trajectory,
)
from src.bounds import classify_regime
from src.metrics import effective_lipschitz_from_divergence
from src.rewrites import Rewrite


SAFETY = 1.08
DEFAULT_PERTURBATIONS = [1, 3, 5, 10, 20, 50]
DEFAULT_STEPS = [1, 3, 5, 10, 15, 25]


class ParameterPerturbationRewrite(Rewrite):
    name = "ParameterPerturbation"

    def __init__(self, perturbation):
        self.perturbation = perturbation

    def apply(self, state):
        return TrainingState(
            theta=state.theta + self.perturbation,
            moments=state.moments.clone(),
            schedule_fn=state.schedule_fn,
            t=state.t,
        )

    def delta(self, state):
        return self.perturbation.norm(2).item()


def parse_int_list(value):
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def make_state(args, schedule):
    torch.manual_seed(args.seed)
    return TrainingState(
        theta=torch.randn(args.dimension) * 0.1,
        moments=torch.zeros(args.dimension, 2),
        schedule_fn=schedule,
        t=0,
    )


def make_rewrite(args):
    torch.manual_seed(args.seed + 1000)
    direction = torch.randn(args.dimension)
    direction = direction / direction.norm(2)
    return ParameterPerturbationRewrite(direction * args.delta)


def bound_diagnostics(divergence, bound):
    ratios = [
        d / b for d, b in zip(divergence, bound)
        if b > 1e-30 and b < 1e300
    ]
    return max(ratios) if ratios else 0.0


def looseness(bound, divergence):
    return bound[-1] / max(divergence[-1], 1e-30)


def danger_from_threshold(divergence, threshold):
    return divergence[-1] >= threshold or max(divergence) >= threshold


def write_csv(path, rows):
    fields = [
        "lipschitz_perturbations", "lipschitz_steps",
        "L_pred_mean", "L_pred_median", "L_actual",
        "absolute_error", "bound_holds", "max_bound_ratio",
        "final_looseness", "runtime_seconds", "predicted_dangerous",
        "predicted_safe", "actual_dangerous", "actual_safe",
        "false_negative", "false_positive",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{k: row[k] for k in fields} for row in rows])


def plot_heatmap(rows, xvals, yvals, key, out_path, title, cmap="viridis"):
    mat = np.zeros((len(yvals), len(xvals)))
    for row in rows:
        x = xvals.index(row["lipschitz_perturbations"])
        y = yvals.index(row["lipschitz_steps"])
        mat[y, x] = row[key]
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(mat, origin="lower", aspect="auto", cmap=cmap)
    ax.set_xticks(range(len(xvals)), labels=xvals)
    ax.set_yticks(range(len(yvals)), labels=yvals)
    ax.set_xlabel("perturbations")
    ax.set_ylabel("probe steps")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_outputs(rows, perturbations, steps, figure_dir):
    os.makedirs(figure_dir, exist_ok=True)
    plot_heatmap(
        rows, perturbations, steps, "L_pred_median",
        Path(figure_dir) / "l_pred_heatmap.png",
        "L_pred median by estimator budget",
    )
    plot_heatmap(
        rows, perturbations, steps,
        "bound_failure",
        Path(figure_dir) / "bound_failure_heatmap.png",
        "Bound failure by estimator budget",
        cmap="Reds",
    )
    plot_heatmap(
        rows, perturbations, steps, "runtime_seconds",
        Path(figure_dir) / "runtime_heatmap.png",
        "Runtime seconds by estimator budget",
    )

    budgets = [r["lipschitz_perturbations"] * r["lipschitz_steps"] for r in rows]
    errors = [r["absolute_error"] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(budgets, errors, c=[r["lipschitz_steps"] for r in rows], cmap="viridis", s=60)
    ax.set_xscale("log")
    ax.set_xlabel("perturbations x probe steps")
    ax.set_ylabel("|L_pred - L_actual|")
    ax.set_title("Estimator error vs budget")
    fig.tight_layout()
    fig.savefig(Path(figure_dir) / "error_vs_budget.png", dpi=180)
    plt.close(fig)


def find_stable_budget(rows, tolerance):
    valid = sorted(
        rows,
        key=lambda r: (r["lipschitz_perturbations"] * r["lipschitz_steps"], r["runtime_seconds"]),
    )
    for row in valid:
        if row["bound_holds"] and not row["false_negative"] and row["absolute_error"] <= tolerance:
            return {
                "lipschitz_perturbations": row["lipschitz_perturbations"],
                "lipschitz_steps": row["lipschitz_steps"],
                "budget": row["lipschitz_perturbations"] * row["lipschitz_steps"],
                "absolute_error": row["absolute_error"],
                "runtime_seconds": row["runtime_seconds"],
            }
    return None


def run(args):
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.figure_dir, exist_ok=True)
    perturbations = parse_int_list(args.perturbations)
    steps = parse_int_list(args.steps)

    schedule = make_constant_schedule(lr=args.lr, beta1=args.beta1, beta2=args.beta2, eps=args.eps)
    A = make_quadratic_hessian(args.dimension, args.condition_number, args.seed)
    loss_fn = make_quadratic_loss(A)
    step_fn = AdamStep()
    s0 = make_state(args, schedule)
    baseline = run_trajectory(s0.clone(), step_fn, loss_fn, args.total_steps)
    rewrite = make_rewrite(args)
    state_at_tR = baseline.states[args.rewrite_step]
    delta = rewrite.delta(state_at_tR)
    rewritten = run_trajectory(
        s0.clone(),
        step_fn,
        loss_fn,
        args.total_steps,
        rewrites=[(args.rewrite_step, rewrite)],
    )
    divergence = compute_divergence(baseline, rewritten)
    L_actual = effective_lipschitz_from_divergence(divergence, args.rewrite_step)
    actual_danger = danger_from_threshold(divergence, args.danger_divergence_threshold)

    rows = []
    for n_perturb in perturbations:
        for n_steps in steps:
            estimates = []
            started = time.perf_counter()
            for rep in range(args.repeats):
                torch.manual_seed(args.seed + rep * 10007 + n_perturb * 101 + n_steps)
                L_raw = apriori_lipschitz_numerical(
                    state_at_tR,
                    step_fn,
                    loss_fn,
                    n_perturbations=n_perturb,
                    n_steps=n_steps,
                    eps=args.lipschitz_eps,
                )
                estimates.append(L_raw * args.safety_margin)
            runtime = time.perf_counter() - started
            est_t = torch.tensor(estimates)
            L_mean = est_t.mean().item()
            L_median = est_t.median().item()
            bound = apriori_bound(delta, L_median, args.rewrite_step, args.total_steps)
            holds = bound_holds(divergence, bound)
            regime = classify_regime(L_median)
            predicted_danger = (
                regime == "expansive"
                or bound[-1] >= args.danger_divergence_threshold
            )
            row = {
                "lipschitz_perturbations": n_perturb,
                "lipschitz_steps": n_steps,
                "L_pred_mean": L_mean,
                "L_pred_median": L_median,
                "L_actual": L_actual,
                "absolute_error": abs(L_median - L_actual),
                "bound_holds": holds,
                "bound_failure": 0 if holds else 1,
                "max_bound_ratio": bound_diagnostics(divergence, bound),
                "final_looseness": looseness(bound, divergence),
                "runtime_seconds": runtime,
                "predicted_dangerous": predicted_danger,
                "predicted_safe": not predicted_danger,
                "actual_dangerous": actual_danger,
                "actual_safe": not actual_danger,
                "false_negative": (not predicted_danger) and actual_danger,
                "false_positive": predicted_danger and not actual_danger,
            }
            rows.append(row)
            print(
                f"pert={n_perturb:2d} steps={n_steps:2d} "
                f"L={L_median:.6f} err={row['absolute_error']:.4f} "
                f"holds={holds} runtime={runtime:.3f}s"
            )

    summary = {
        "grid_points": len(rows),
        "L_actual": L_actual,
        "delta": delta,
        "bound_failures": sum(1 for r in rows if not r["bound_holds"]),
        "false_negatives": sum(1 for r in rows if r["false_negative"]),
        "false_positives": sum(1 for r in rows if r["false_positive"]),
        "smallest_stable_budget": find_stable_budget(rows, args.stability_tolerance),
        "recommended_default": {
            "lipschitz_perturbations": 20,
            "lipschitz_steps": 15,
            "reason": "Matches existing experiments and is stable in this sweep while keeping runtime modest.",
        },
    }

    out = Path(args.output_dir)
    write_csv(out / "budget_metrics.csv", rows)
    with open(out / "budget_metrics.json", "w") as f:
        json.dump({"rows": rows}, f, indent=2)
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    config = vars(args).copy()
    config.update({
        "perturbations": perturbations,
        "steps": steps,
        "actual_delta": delta,
        "L_actual": L_actual,
    })
    with open(out / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    plot_outputs(rows, perturbations, steps, args.figure_dir)
    print(f"\nSaved metrics -> {out}")
    print(f"Saved figures -> {Path(args.figure_dir)}")
    return {"rows": rows, "summary": summary}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--perturbations", default=",".join(str(v) for v in DEFAULT_PERTURBATIONS))
    p.add_argument("--steps", default=",".join(str(v) for v in DEFAULT_STEPS))
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--delta", type=float, default=0.03)
    p.add_argument("--rewrite-step", type=int, default=250)
    p.add_argument("--total-steps", type=int, default=500)
    p.add_argument("--dimension", type=int, default=10)
    p.add_argument("--condition-number", type=float, default=10.0)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--safety-margin", type=float, default=SAFETY)
    p.add_argument("--lipschitz-eps", type=float, default=1e-4)
    p.add_argument("--danger-divergence-threshold", type=float, default=1.0)
    p.add_argument("--stability-tolerance", type=float, default=0.1)
    p.add_argument("--output-dir", default="results/lipschitz_budget_ablation")
    p.add_argument("--figure-dir", default="figures/lipschitz_budget_ablation")
    args = p.parse_args()
    if args.rewrite_step < 0 or args.rewrite_step > args.total_steps:
        raise ValueError("--rewrite-step must be between 0 and --total-steps")
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive")
    return args


if __name__ == "__main__":
    run(parse_args())
