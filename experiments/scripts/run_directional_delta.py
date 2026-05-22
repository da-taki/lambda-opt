"""Directional delta experiment.

Same ||delta theta||, different perturbation directions on a quadratic loss.
This isolates directional sensitivity that is invisible to scalar delta alone.

Usage:
    cd experiments
    py scripts/run_directional_delta.py
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
    TrainingState,
    apriori_bound,
    apriori_lipschitz_numerical,
    bound_holds,
    compute_divergence,
    loss_gap,
    make_constant_schedule,
    make_quadratic_hessian,
    make_quadratic_loss,
    run_trajectory,
    tightness_ratio,
)
from src.bounds import classify_regime
from src.rewrites import Rewrite


SAFETY = 1.08


class ParameterPerturbationRewrite(Rewrite):
    """Apply theta <- theta + perturbation, preserving optimizer state."""

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


def _normalize(vec: torch.Tensor, fallback: torch.Tensor) -> torch.Tensor:
    norm = vec.norm(2)
    if torch.isfinite(norm) and norm.item() > 1e-12:
        return vec / norm
    fb_norm = fallback.norm(2)
    if fb_norm.item() <= 1e-12:
        raise ValueError("Cannot normalize zero vector and zero fallback")
    return fallback / fb_norm


def build_directions(A, loss_fn, baseline, rewrite_step, trajectory_lag, seed):
    state_at_tR = baseline.states[rewrite_step]
    n = state_at_tR.theta.numel()

    torch.manual_seed(seed + 1000)
    fallback = torch.randn(n)
    fallback = fallback / fallback.norm(2)

    lag_step = max(0, rewrite_step - trajectory_lag)
    trajectory_vec = state_at_tR.theta - baseline.states[lag_step].theta

    _, grad = loss_fn(state_at_tR.theta)
    eigvals, eigvecs = torch.linalg.eigh(A)

    torch.manual_seed(seed + 2000)
    random_vec = torch.randn(n)

    return {
        "trajectory_aligned": _normalize(trajectory_vec, fallback),
        "gradient_aligned": _normalize(grad, fallback),
        "random_gaussian": _normalize(random_vec, fallback),
        "top_hessian": _normalize(eigvecs[:, -1], fallback),
        "bottom_hessian": _normalize(eigvecs[:, 0], fallback),
    }, {
        "trajectory_lag": trajectory_lag,
        "lag_step": lag_step,
        "top_hessian_eigenvalue": eigvals[-1].item(),
        "bottom_hessian_eigenvalue": eigvals[0].item(),
    }


def write_metrics_csv(path, rows):
    fields = [
        "direction",
        "target_delta",
        "actual_delta",
        "L_pred",
        "regime",
        "final_divergence",
        "max_divergence",
        "final_loss_gap",
        "max_loss_gap",
        "bound_holds",
        "median_tightness",
        "final_tightness",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fields})


def write_curves_csv(path, curve_rows):
    fields = ["step", "direction", "divergence", "bound", "loss_gap"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(curve_rows)


def plot_curves(results, figure_dir):
    os.makedirs(figure_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    for direction, r in results.items():
        ax.semilogy(
            r["steps"],
            [max(v, 1e-15) for v in r["divergence"]],
            lw=1.6,
            label=direction,
        )
    ax.set_xlabel("step")
    ax.set_ylabel("D(t)")
    ax.set_title("Directional delta: divergence")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(Path(figure_dir) / "divergence_by_direction.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for direction, r in results.items():
        ax.semilogy(
            r["steps"][:-1],
            [max(v, 1e-15) for v in r["loss_gap"]],
            lw=1.6,
            label=direction,
        )
    ax.set_xlabel("step")
    ax.set_ylabel("|loss gap|")
    ax.set_title("Directional delta: loss gap")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(Path(figure_dir) / "loss_gap_by_direction.png", dpi=180)
    plt.close(fig)

    labels = list(results.keys())
    final_divs = [results[k]["final_divergence"] for k in labels]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(labels, final_divs, color="steelblue")
    ax.set_ylabel("D(T)")
    ax.set_title("Final divergence by direction")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(Path(figure_dir) / "final_divergence_bar.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for direction, r in results.items():
        ax.plot(r["steps"], r["tightness"], lw=1.6, label=direction)
    ax.set_xlabel("step")
    ax.set_ylabel("D(t) / bound(t)")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Directional delta: tightness")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(Path(figure_dir) / "tightness_by_direction.png", dpi=180)
    plt.close(fig)


def run(args):
    if args.rewrite_step < 0 or args.rewrite_step > args.total_steps:
        raise ValueError("--rewrite-step must be between 0 and --total-steps")
    if args.delta <= 0:
        raise ValueError("--delta must be positive")
    if args.dimension <= 0:
        raise ValueError("--dimension must be positive")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.figure_dir, exist_ok=True)

    schedule = make_constant_schedule(lr=args.lr, beta1=args.beta1, beta2=args.beta2, eps=args.eps)
    A = make_quadratic_hessian(args.dimension, args.condition_number, args.seed)
    loss_fn = make_quadratic_loss(A)
    step_fn = AdamStep()

    torch.manual_seed(args.seed)
    s0 = TrainingState(
        theta=torch.randn(args.dimension) * 0.1,
        moments=torch.zeros(args.dimension, 2),
        schedule_fn=schedule,
        t=0,
    )

    baseline = run_trajectory(s0.clone(), step_fn, loss_fn, args.total_steps)
    state_at_tR = baseline.states[args.rewrite_step]

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

    directions, direction_meta = build_directions(
        A, loss_fn, baseline, args.rewrite_step, args.trajectory_lag, args.seed
    )

    metrics_rows = []
    curve_rows = []
    results = {}

    for name, direction in directions.items():
        perturbation = direction * args.delta
        rewrite = ParameterPerturbationRewrite(perturbation)
        actual_delta = rewrite.delta(state_at_tR)

        torch.manual_seed(args.seed)
        rewritten = run_trajectory(
            s0.clone(),
            step_fn,
            loss_fn,
            args.total_steps,
            rewrites=[(args.rewrite_step, rewrite)],
        )

        divergence = compute_divergence(baseline, rewritten)
        bound = apriori_bound(actual_delta, L_pred, args.rewrite_step, args.total_steps)
        gaps = loss_gap(baseline.losses, rewritten.losses)
        tightness = tightness_ratio(divergence, bound)
        holds = bound_holds(divergence, bound)

        post_tightness = [
            v for step, v in enumerate(tightness)
            if step >= args.rewrite_step and 0.0 < v < 1.0
        ]
        median_tightness = (
            float(torch.tensor(post_tightness).median().item()) if post_tightness else 0.0
        )

        row = {
            "direction": name,
            "target_delta": args.delta,
            "actual_delta": actual_delta,
            "L_pred": L_pred,
            "L_raw": L_raw,
            "regime": regime,
            "final_divergence": divergence[-1],
            "max_divergence": max(divergence),
            "final_loss_gap": gaps[-1] if gaps else 0.0,
            "max_loss_gap": max(gaps) if gaps else 0.0,
            "bound_holds": holds,
            "median_tightness": median_tightness,
            "final_tightness": tightness[-1],
        }
        metrics_rows.append(row)

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
                "direction": name,
                "divergence": divergence[step],
                "bound": bound[step],
                "loss_gap": gaps[step] if step < len(gaps) else "",
            })

        print(
            f"{name:20s} delta={actual_delta:.6f} "
            f"D(T)={divergence[-1]:.6e} max_D={max(divergence):.6e} "
            f"bound={'PASS' if holds else 'FAIL'}"
        )

    config = {
        "seed": args.seed,
        "dimension": args.dimension,
        "target_delta": args.delta,
        "rewrite_step": args.rewrite_step,
        "total_steps": args.total_steps,
        "condition_number": args.condition_number,
        "lr": args.lr,
        "beta1": args.beta1,
        "beta2": args.beta2,
        "eps": args.eps,
        "safety_margin": args.safety_margin,
        "L_raw": L_raw,
        "L_pred": L_pred,
        "regime": regime,
        **direction_meta,
    }

    output_dir = Path(args.output_dir)
    write_metrics_csv(output_dir / "metrics.csv", metrics_rows)
    write_curves_csv(output_dir / "divergence_curves.csv", curve_rows)
    with open(output_dir / "metrics.json", "w") as f:
        json.dump({"metrics": metrics_rows, "curves": results}, f, indent=2)
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    plot_curves(results, args.figure_dir)

    print(f"\nSaved metrics -> {output_dir}")
    print(f"Saved figures -> {Path(args.figure_dir)}")
    return {"metrics": metrics_rows, "config": config}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dimension", type=int, default=10)
    p.add_argument("--delta", type=float, default=0.03)
    p.add_argument("--rewrite-step", type=int, default=250)
    p.add_argument("--total-steps", type=int, default=500)
    p.add_argument("--condition-number", type=float, default=10.0)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--trajectory-lag", type=int, default=10)
    p.add_argument("--lipschitz-perturbations", type=int, default=20)
    p.add_argument("--lipschitz-steps", type=int, default=15)
    p.add_argument("--lipschitz-eps", type=float, default=1e-4)
    p.add_argument("--safety-margin", type=float, default=SAFETY)
    p.add_argument("--output-dir", default="results/directional_delta")
    p.add_argument("--figure-dir", default="figures/directional_delta")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
