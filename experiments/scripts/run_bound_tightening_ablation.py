"""Bound-tightening ablation for lambdaOpt.

Compares fixed clean-state L against rewritten-state, max(clean, rewritten),
rolling, and diagnostic empirical-envelope bounds on a quadratic system.

Usage:
    cd experiments
    py scripts/run_bound_tightening_ablation.py
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
    if args.direction == "random_gaussian":
        torch.manual_seed(args.seed + 1000)
        return normalize(torch.randn(args.dimension))
    if args.direction == "gradient":
        _, grad = loss_fn(state_at_tR.theta)
        return normalize(grad)
    _, eigvecs = torch.linalg.eigh(A)
    if args.direction == "top_hessian":
        return normalize(eigvecs[:, -1])
    if args.direction == "bottom_hessian":
        return normalize(eigvecs[:, 0])
    raise ValueError(f"Unknown direction: {args.direction}")


def safe_power(base, exponent):
    if exponent == 0:
        return 1.0
    if base <= 0:
        return 0.0
    log_val = exponent * math.log(base)
    if log_val > math.log(1e300):
        return 1e300
    if log_val < -700:
        return 0.0
    return math.exp(log_val)


def rolling_bound(delta, L_by_step, t_R, T):
    bound = [0.0] * (T + 1)
    bound[t_R] = delta
    cumul = delta
    for t in range(t_R + 1, T + 1):
        cumul *= L_by_step.get(t - 1, L_by_step[max(k for k in L_by_step if k <= t - 1)])
        bound[t] = cumul
    return bound


def empirical_envelope_bound(delta, divergence, t_R, T, safety):
    ratios = {}
    last = 1.0
    for t in range(t_R, T):
        if divergence[t] > 1e-15:
            last = max(divergence[t + 1] / divergence[t], 1e-12) * safety
        ratios[t] = last
    return rolling_bound(delta, ratios, t_R, T), list(ratios.values())


def estimate_L(state, step_fn, loss_fn, args):
    return apriori_lipschitz_numerical(
        state,
        step_fn,
        loss_fn,
        n_perturbations=args.lipschitz_perturbations,
        n_steps=args.lipschitz_steps,
        eps=args.lipschitz_eps,
    ) * args.safety_margin


def rolling_L_values(baseline, rewritten, step_fn, loss_fn, args):
    L_by_step = {}
    last = None
    for t in range(args.rewrite_step, args.total_steps):
        if last is None or (t - args.rewrite_step) % args.rolling_every == 0:
            if args.rolling_source == "rewritten":
                state = rewritten.states[t]
            elif args.rolling_source == "max_pair":
                L_clean = estimate_L(baseline.states[t], step_fn, loss_fn, args)
                L_rw = estimate_L(rewritten.states[t], step_fn, loss_fn, args)
                last = max(L_clean, L_rw)
                L_by_step[t] = last
                continue
            else:
                state = baseline.states[t]
            last = estimate_L(state, step_fn, loss_fn, args)
        L_by_step[t] = last
    return L_by_step


def bound_diagnostics(divergence, bound):
    first_violation = None
    ratios = []
    for i, (d, b) in enumerate(zip(divergence, bound)):
        if b > 1e-30 and b < 1e300:
            r = d / b
            ratios.append(r)
            if r > 1.0 and first_violation is None:
                first_violation = i
    final_ratio = 0.0
    if bound[-1] > 1e-30 and bound[-1] < 1e300:
        final_ratio = divergence[-1] / bound[-1]
    return {
        "first_violation_step": first_violation,
        "max_bound_ratio": max(ratios) if ratios else 0.0,
        "final_bound_ratio": final_ratio,
    }


def looseness(bound, divergence, t_R):
    final = bound[-1] / max(divergence[-1], 1e-30)
    vals = [
        b / max(d, 1e-30)
        for b, d in zip(bound[t_R:], divergence[t_R:])
        if b > 0 and d > 1e-30 and b < 1e300
    ]
    return final, (float(torch.tensor(vals).median().item()) if vals else 0.0)


def summarize_L(values):
    if len(values) == 1:
        return f"{values[0]:.6g}"
    t = torch.tensor(values)
    return (
        f"min={t.min().item():.6g};"
        f"median={t.median().item():.6g};"
        f"max={t.max().item():.6g}"
    )


def evaluate_method(method, L_values, bound, divergence, t_R):
    tight = tightness_ratio(divergence, bound)
    diag = bound_diagnostics(divergence, bound)
    loose_final, loose_median = looseness(bound, divergence, t_R)
    post_tight = [v for i, v in enumerate(tight) if i >= t_R and 0.0 < v < 1.0]
    return {
        "method": method,
        "L_values": L_values,
        "L_summary": summarize_L(L_values),
        "final_bound": bound[-1],
        "max_bound": max(bound),
        "bound_holds": bound_holds(divergence, bound),
        "median_tightness": float(torch.tensor(post_tight).median().item()) if post_tight else 0.0,
        "final_tightness": tight[-1],
        "looseness_factor_final": loose_final,
        "looseness_factor_median": loose_median,
        **diag,
    }, tight


def write_csv(path, rows, fields):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{k: row[k] for k in fields} for row in rows])


def plot_outputs(divergence, bounds, metrics, figure_dir, t_R):
    os.makedirs(figure_dir, exist_ok=True)
    steps = list(range(len(divergence)))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.semilogy(steps, [max(v, 1e-15) for v in divergence], "k-", lw=2, label="actual D(t)")
    for method, bound in bounds.items():
        ax.semilogy(steps, [max(v, 1e-15) for v in bound], lw=1.3, label=method)
    ax.axvline(t_R, color="gray", ls=":", alpha=0.5)
    ax.set_xlabel("step")
    ax.set_ylabel("D(t), B(t)")
    ax.set_title("Bounds vs actual divergence")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(Path(figure_dir) / "bounds_vs_actual.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for row in metrics:
        method = row["method"]
        tight = tightness_ratio(divergence, bounds[method])
        ax.plot(steps, tight, lw=1.3, label=method)
    ax.set_xlabel("step")
    ax.set_ylabel("D(t) / B(t)")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Bound tightness by method")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(Path(figure_dir) / "bound_tightness_by_method.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = [row["method"] for row in metrics]
    values = [min(row["looseness_factor_final"], 1e12) for row in metrics]
    ax.bar(labels, values, color="steelblue")
    ax.set_yscale("log")
    ax.set_ylabel("B(T) / D(T)")
    ax.set_title("Final looseness by method")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(Path(figure_dir) / "final_looseness_by_method.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for row in metrics:
        vals = row["L_values"]
        ax.scatter([row["method"]] * len(vals), vals, s=24)
    ax.axhline(1.0, color="gray", ls=":", alpha=0.4)
    ax.set_ylabel("L")
    ax.set_title("L values by method")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(Path(figure_dir) / "l_values_by_method.png", dpi=180)
    plt.close(fig)


def run(args):
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
    clean_state = baseline.states[args.rewrite_step]
    direction = make_direction(args, A, loss_fn, clean_state)
    rewrite = ParameterPerturbationRewrite(direction * args.delta)
    rewritten_state = rewrite.apply(clean_state)
    actual_delta = rewrite.delta(clean_state)
    rewritten = run_trajectory(
        s0.clone(), step_fn, loss_fn, args.total_steps, rewrites=[(args.rewrite_step, rewrite)]
    )
    divergence = compute_divergence(baseline, rewritten)
    gaps = loss_gap(baseline.losses, rewritten.losses)

    L_clean = estimate_L(clean_state, step_fn, loss_fn, args)
    L_rewritten = estimate_L(rewritten_state, step_fn, loss_fn, args)
    L_max = max(L_clean, L_rewritten)

    bounds = {
        "fixed_clean_L": apriori_bound(actual_delta, L_clean, args.rewrite_step, args.total_steps),
        "fixed_rewritten_L": apriori_bound(actual_delta, L_rewritten, args.rewrite_step, args.total_steps),
        "max_clean_rewritten_L": apriori_bound(actual_delta, L_max, args.rewrite_step, args.total_steps),
    }
    L_map = {
        "fixed_clean_L": [L_clean],
        "fixed_rewritten_L": [L_rewritten],
        "max_clean_rewritten_L": [L_max],
    }

    rolling_values = rolling_L_values(baseline, rewritten, step_fn, loss_fn, args)
    bounds["rolling_L"] = rolling_bound(actual_delta, rolling_values, args.rewrite_step, args.total_steps)
    L_map["rolling_L"] = list(rolling_values.values())

    bounds["empirical_envelope_L"], empirical_L = empirical_envelope_bound(
        actual_delta, divergence, args.rewrite_step, args.total_steps, args.safety_margin
    )
    L_map["empirical_envelope_L"] = empirical_L

    metrics = []
    curves = []
    for method, bound in bounds.items():
        row, tight = evaluate_method(method, L_map[method], bound, divergence, args.rewrite_step)
        row.update({
            "final_divergence": divergence[-1],
            "max_divergence": max(divergence),
            "final_loss_gap": gaps[-1] if gaps else 0.0,
            "max_loss_gap": max(gaps) if gaps else 0.0,
        })
        metrics.append(row)
        for step, (d, b, t) in enumerate(zip(divergence, bound, tight)):
            curves.append({
                "step": step,
                "method": method,
                "divergence": d,
                "bound": b,
                "tightness": t,
                "loss_gap": gaps[step] if step < len(gaps) else "",
            })
        print(
            f"{method:24s} holds={row['bound_holds']} "
            f"final_looseness={row['looseness_factor_final']:.3e} "
            f"max_D/B={row['max_bound_ratio']:.3e}"
        )

    metric_fields = [
        "method", "L_summary", "final_bound", "max_bound",
        "final_divergence", "max_divergence", "bound_holds",
        "first_violation_step", "max_bound_ratio", "final_bound_ratio",
        "median_tightness", "final_tightness", "looseness_factor_final",
        "looseness_factor_median", "final_loss_gap", "max_loss_gap",
    ]
    curve_fields = ["step", "method", "divergence", "bound", "tightness", "loss_gap"]
    out = Path(args.output_dir)
    write_csv(out / "bound_metrics.csv", metrics, metric_fields)
    write_csv(out / "bound_curves.csv", curves, curve_fields)
    with open(out / "bound_metrics.json", "w") as f:
        json.dump({"metrics": metrics, "curves": curves}, f, indent=2)
    summary = {
        "methods": len(metrics),
        "valid_methods": [row["method"] for row in metrics if row["bound_holds"]],
        "tightest_valid_by_final_looseness": min(
            (row for row in metrics if row["bound_holds"]),
            key=lambda r: r["looseness_factor_final"],
            default=None,
        )["method"],
        "fixed_clean_L": L_clean,
        "fixed_rewritten_L": L_rewritten,
        "actual_delta": actual_delta,
    }
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    config = vars(args).copy()
    config.update({
        "actual_delta": actual_delta,
        "L_clean": L_clean,
        "L_rewritten": L_rewritten,
        "L_max_clean_rewritten": L_max,
        "empirical_envelope_note": "Diagnostic only: uses observed post-rewrite divergence ratios.",
        "rolling_L_note": f"Re-estimated every {args.rolling_every} steps from {args.rolling_source}.",
    })
    with open(out / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    plot_outputs(divergence, bounds, metrics, args.figure_dir, args.rewrite_step)
    print(f"\nSaved metrics -> {out}")
    print(f"Saved figures -> {Path(args.figure_dir)}")
    return {"metrics": metrics, "summary": summary}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--delta", type=float, default=0.03)
    p.add_argument(
        "--direction",
        choices=["random_gaussian", "gradient", "top_hessian", "bottom_hessian"],
        default="random_gaussian",
    )
    p.add_argument("--rewrite-step", type=int, default=250)
    p.add_argument("--total-steps", type=int, default=500)
    p.add_argument("--dimension", type=int, default=10)
    p.add_argument("--condition-number", type=float, default=10.0)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--safety-margin", type=float, default=SAFETY)
    p.add_argument("--rolling-every", type=int, default=25)
    p.add_argument("--rolling-source", choices=["baseline", "rewritten", "max_pair"], default="max_pair")
    p.add_argument("--lipschitz-perturbations", type=int, default=10)
    p.add_argument("--lipschitz-steps", type=int, default=8)
    p.add_argument("--lipschitz-eps", type=float, default=1e-4)
    p.add_argument("--output-dir", default="results/bound_tightening_ablation")
    p.add_argument("--figure-dir", default="figures/bound_tightening_ablation")
    args = p.parse_args()
    if args.rewrite_step < 0 or args.rewrite_step > args.total_steps:
        raise ValueError("--rewrite-step must be between 0 and --total-steps")
    if args.delta <= 0:
        raise ValueError("--delta must be positive")
    if args.rolling_every <= 0:
        raise ValueError("--rolling-every must be positive")
    return args


if __name__ == "__main__":
    run(parse_args())
