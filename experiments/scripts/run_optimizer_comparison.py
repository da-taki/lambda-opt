"""Optimizer comparison for lambdaOpt.

Same quadratic system, same parameter rewrite, different optimizers.
This tests whether lambdaOpt's predictions are optimizer-specific.

Usage:
    cd experiments
    py scripts/run_optimizer_comparison.py
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import (
    AdamStep,
    AdamWStep,
    RMSPropStep,
    SGDStep,
    TrainingState,
    apriori_bound,
    apriori_lipschitz_numerical,
    apriori_lipschitz_sgd,
    bound_holds,
    compute_divergence,
    hessian_eigenvalues_exact,
    loss_gap,
    make_constant_schedule,
    make_quadratic_hessian,
    make_quadratic_loss,
    run_trajectory,
    tightness_ratio,
)
from src.bounds import classify_regime
from src.metrics import effective_lipschitz_from_divergence
from src.rewrites import Rewrite


SAFETY = 1.08


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


def normalize(vec):
    n = vec.norm(2)
    if not torch.isfinite(n) or n.item() <= 1e-12:
        raise ValueError("Cannot normalize zero or non-finite direction")
    return vec / n


def make_direction(args, A, loss_fn, state_at_tR):
    if args.direction == "random":
        torch.manual_seed(args.seed + 1000)
        return normalize(torch.randn(args.dimension))
    if args.direction == "gradient":
        _, grad = loss_fn(state_at_tR.theta)
        return normalize(grad)
    eigvals, eigvecs = torch.linalg.eigh(A)
    if args.direction == "top_hessian":
        return normalize(eigvecs[:, -1])
    if args.direction == "bottom_hessian":
        return normalize(eigvecs[:, 0])
    raise ValueError(f"Unknown direction: {args.direction}")


def make_initial_state(args, schedule, moment_cols=2):
    torch.manual_seed(args.seed)
    return TrainingState(
        theta=torch.randn(args.dimension) * 0.1,
        moments=torch.zeros(args.dimension, moment_cols),
        schedule_fn=schedule,
        t=0,
    )


def optimizer_specs(args):
    return [
        {
            "name": "Adam",
            "step_fn": AdamStep(),
            "estimator": "numerical",
        },
        {
            "name": "AdamW",
            "step_fn": AdamWStep(weight_decay=args.weight_decay),
            "estimator": "numerical",
        },
        {
            "name": "SGD",
            "step_fn": SGDStep(momentum=0.0),
            "estimator": "exact_spectral_sgd",
            "momentum": 0.0,
        },
        {
            "name": "SGD_momentum",
            "step_fn": SGDStep(momentum=args.sgd_momentum),
            "estimator": "numerical",
        },
        {
            "name": "RMSProp",
            "step_fn": RMSPropStep(alpha=args.rmsprop_alpha),
            "estimator": "numerical",
        },
    ]


def bound_diagnostics(divergence, bound):
    ratios = []
    for d, b in zip(divergence, bound):
        if b > 1e-30 and b < 1e300:
            ratios.append(d / b)
    return max(ratios) if ratios else 0.0


def median_post_tightness(tightness, rewrite_step):
    vals = [v for i, v in enumerate(tightness) if i >= rewrite_step and 0.0 < v < 1.0]
    return float(torch.tensor(vals).median().item()) if vals else 0.0


def actual_dangerous(divergence, gaps, args):
    return (
        divergence[-1] >= args.danger_divergence_threshold
        or (max(gaps) if gaps else 0.0) >= args.danger_loss_gap_threshold
    )


def predicted_dangerous(delta, L_pred, regime, args):
    final_bound = delta * (L_pred ** max(0, args.total_steps - args.rewrite_step))
    return (
        regime == "expansive"
        or final_bound >= args.danger_divergence_threshold
    )


def write_metrics_csv(path, rows):
    fields = [
        "optimizer", "target_delta", "actual_delta", "L_pred", "L_actual",
        "L_pred_estimator", "regime", "final_divergence", "max_divergence",
        "final_loss_gap", "max_loss_gap", "bound_holds", "max_bound_ratio",
        "median_tightness", "final_tightness", "predicted_safe",
        "predicted_dangerous", "actual_dangerous",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fields})


def write_curves_csv(path, curve_rows):
    fields = ["step", "optimizer", "divergence", "bound", "loss_gap", "tightness"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(curve_rows)


def plot_outputs(results, figure_dir):
    os.makedirs(figure_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, r in results.items():
        ax.semilogy(r["steps"], [max(v, 1e-15) for v in r["divergence"]], label=name, lw=1.5)
    ax.set_xlabel("step")
    ax.set_ylabel("D(t)")
    ax.set_title("Divergence by optimizer")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(Path(figure_dir) / "divergence_by_optimizer.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, r in results.items():
        ax.semilogy(r["steps"][:-1], [max(v, 1e-15) for v in r["loss_gap"]], label=name, lw=1.5)
    ax.set_xlabel("step")
    ax.set_ylabel("|loss gap|")
    ax.set_title("Loss gap by optimizer")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(Path(figure_dir) / "loss_gap_by_optimizer.png", dpi=180)
    plt.close(fig)

    labels = list(results.keys())
    final_divs = [results[k]["final_divergence"] for k in labels]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(labels, final_divs, color="steelblue")
    ax.set_ylabel("D(T)")
    ax.set_title("Final divergence by optimizer")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(Path(figure_dir) / "final_divergence_by_optimizer.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    for name, r in results.items():
        ax.scatter(r["L_pred"], r["L_actual"], s=70, label=name)
    vals = [v for r in results.values() for v in (r["L_pred"], r["L_actual"])]
    lo, hi = min(vals) * 0.95, max(vals) * 1.05
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.35)
    ax.axhline(1, color="gray", ls=":", alpha=0.4)
    ax.axvline(1, color="gray", ls=":", alpha=0.4)
    ax.set_xlabel("L_pred")
    ax.set_ylabel("L_actual")
    ax.set_title("L_pred vs L_actual")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(Path(figure_dir) / "l_pred_vs_l_actual_by_optimizer.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, r in results.items():
        ax.plot(r["steps"], r["tightness"], label=name, lw=1.5)
    ax.set_xlabel("step")
    ax.set_ylabel("D(t) / B(t)")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Bound tightness by optimizer")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(Path(figure_dir) / "bound_tightness_by_optimizer.png", dpi=180)
    plt.close(fig)


def run(args):
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.figure_dir, exist_ok=True)

    schedule = make_constant_schedule(
        lr=args.lr, beta1=args.beta1, beta2=args.beta2, eps=args.eps
    )
    A = make_quadratic_hessian(args.dimension, args.condition_number, args.seed)
    eigs = hessian_eigenvalues_exact(A)
    loss_fn = make_quadratic_loss(A)

    rows = []
    curve_rows = []
    results = {}

    for spec in optimizer_specs(args):
        name = spec["name"]
        step_fn = spec["step_fn"]
        s0 = make_initial_state(args, schedule)
        baseline = run_trajectory(s0.clone(), step_fn, loss_fn, args.total_steps)
        state_at_tR = baseline.states[args.rewrite_step]
        direction = make_direction(args, A, loss_fn, state_at_tR)
        rewrite = ParameterPerturbationRewrite(direction * args.delta)
        actual_delta = rewrite.delta(state_at_tR)

        if spec["estimator"] == "exact_spectral_sgd":
            L_raw = apriori_lipschitz_sgd(eigs, lr=args.lr, momentum=0.0)
            L_pred = L_raw * args.safety_margin
        else:
            L_raw = apriori_lipschitz_numerical(
                state_at_tR,
                step_fn,
                loss_fn,
                n_perturbations=args.lipschitz_perturbations,
                n_steps=args.lipschitz_steps,
                eps=args.lipschitz_eps,
            )
            L_pred = L_raw * args.safety_margin

        regime = classify_regime(L_pred)
        rewritten = run_trajectory(
            s0.clone(),
            step_fn,
            loss_fn,
            args.total_steps,
            rewrites=[(args.rewrite_step, rewrite)],
        )
        divergence = compute_divergence(baseline, rewritten)
        gaps = loss_gap(baseline.losses, rewritten.losses)
        bound = apriori_bound(actual_delta, L_pred, args.rewrite_step, args.total_steps)
        tightness = tightness_ratio(divergence, bound)
        holds = bound_holds(divergence, bound)
        L_actual = effective_lipschitz_from_divergence(divergence, args.rewrite_step)
        pred_danger = predicted_dangerous(actual_delta, L_pred, regime, args)
        actual_danger = actual_dangerous(divergence, gaps, args)

        row = {
            "optimizer": name,
            "target_delta": args.delta,
            "actual_delta": actual_delta,
            "L_pred": L_pred,
            "L_raw": L_raw,
            "L_actual": L_actual,
            "L_pred_estimator": spec["estimator"],
            "regime": regime,
            "final_divergence": divergence[-1],
            "max_divergence": max(divergence),
            "final_loss_gap": gaps[-1] if gaps else 0.0,
            "max_loss_gap": max(gaps) if gaps else 0.0,
            "bound_holds": holds,
            "max_bound_ratio": bound_diagnostics(divergence, bound),
            "median_tightness": median_post_tightness(tightness, args.rewrite_step),
            "final_tightness": tightness[-1],
            "predicted_safe": not pred_danger,
            "predicted_dangerous": pred_danger,
            "actual_dangerous": actual_danger,
        }
        rows.append(row)

        steps = list(range(len(divergence)))
        results[name] = {
            "steps": steps,
            "divergence": divergence,
            "bound": bound,
            "loss_gap": gaps,
            "tightness": tightness,
            **row,
        }
        for step in steps:
            curve_rows.append({
                "step": step,
                "optimizer": name,
                "divergence": divergence[step],
                "bound": bound[step],
                "loss_gap": gaps[step] if step < len(gaps) else "",
                "tightness": tightness[step],
            })

        print(
            f"{name:12s} L_pred={L_pred:.4f} L_actual={L_actual:.4f} "
            f"D(T)={divergence[-1]:.6e} max_D={max(divergence):.6e} "
            f"bound={'PASS' if holds else 'FAIL'}"
        )

    summary = {
        "total_optimizers": len(rows),
        "bound_failures": sum(1 for r in rows if not r["bound_holds"]),
        "predicted_dangerous": sum(1 for r in rows if r["predicted_dangerous"]),
        "predicted_safe": sum(1 for r in rows if r["predicted_safe"]),
        "actual_dangerous": sum(1 for r in rows if r["actual_dangerous"]),
        "actual_safe": sum(1 for r in rows if not r["actual_dangerous"]),
    }
    config = {
        "seed": args.seed,
        "dimension": args.dimension,
        "condition_number": args.condition_number,
        "lr": args.lr,
        "target_delta": args.delta,
        "direction": args.direction,
        "rewrite_step": args.rewrite_step,
        "total_steps": args.total_steps,
        "safety_margin": args.safety_margin,
        "danger_divergence_threshold": args.danger_divergence_threshold,
        "danger_loss_gap_threshold": args.danger_loss_gap_threshold,
        "optimizers": [s["name"] for s in optimizer_specs(args)],
    }

    out = Path(args.output_dir)
    write_metrics_csv(out / "optimizer_metrics.csv", rows)
    write_curves_csv(out / "divergence_curves.csv", curve_rows)
    with open(out / "optimizer_metrics.json", "w") as f:
        json.dump({"metrics": rows, "curves": results}, f, indent=2)
    with open(out / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    plot_outputs(results, args.figure_dir)
    print(f"\nSaved metrics -> {out}")
    print(f"Saved figures -> {Path(args.figure_dir)}")
    return {"metrics": rows, "summary": summary}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--delta", type=float, default=0.01)
    p.add_argument(
        "--direction",
        choices=["random", "gradient", "top_hessian", "bottom_hessian"],
        default="random",
    )
    p.add_argument("--rewrite-step", type=int, default=100)
    p.add_argument("--total-steps", type=int, default=200)
    p.add_argument("--dimension", type=int, default=10)
    p.add_argument("--condition-number", type=float, default=100.0)
    p.add_argument("--lr", type=float, default=0.021)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--safety-margin", type=float, default=SAFETY)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--sgd-momentum", type=float, default=0.9)
    p.add_argument("--rmsprop-alpha", type=float, default=0.99)
    p.add_argument("--lipschitz-perturbations", type=int, default=10)
    p.add_argument("--lipschitz-steps", type=int, default=8)
    p.add_argument("--lipschitz-eps", type=float, default=1e-4)
    p.add_argument("--danger-loss-gap-threshold", type=float, default=0.01)
    p.add_argument("--danger-divergence-threshold", type=float, default=1.0)
    p.add_argument("--output-dir", default="results/optimizer_comparison")
    p.add_argument("--figure-dir", default="figures/optimizer_comparison")
    args = p.parse_args()
    if args.rewrite_step < 0 or args.rewrite_step > args.total_steps:
        raise ValueError("--rewrite-step must be between 0 and --total-steps")
    if args.delta <= 0:
        raise ValueError("--delta must be positive")
    return args


if __name__ == "__main__":
    run(parse_args())
