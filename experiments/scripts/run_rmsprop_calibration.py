"""RMSProp Lipschitz-estimator calibration.

Diagnoses the RMSProp false negative from optimizer comparison by sweeping
safety margin, perturbation count, probe horizon, and clean/rewritten/max
state estimators on the same quadratic setup.
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
    RMSPropStep,
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
)
from src.bounds import classify_regime
from src.metrics import effective_lipschitz_from_divergence
from src.rewrites import Rewrite


DEFAULT_MARGINS = [1.0, 1.03, 1.08, 1.15, 1.25, 1.5]
DEFAULT_PERTURBATIONS = [3, 5, 10, 20, 50]
DEFAULT_STEPS = [1, 3, 5, 10, 15, 25, 50]


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


def parse_float_list(value):
    return [float(v.strip()) for v in value.split(",") if v.strip()]


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


def actual_dangerous(divergence, gaps, args):
    return (
        divergence[-1] >= args.danger_divergence_threshold
        or (max(gaps) if gaps else 0.0) >= args.danger_loss_gap_threshold
    )


def predicted_dangerous(delta, L_pred, args):
    regime = classify_regime(L_pred)
    final_bound = delta * (L_pred ** max(0, args.total_steps - args.rewrite_step))
    return regime == "expansive" or final_bound >= args.danger_divergence_threshold


def bound_ratios(divergence, bound):
    ratios = [d / b for d, b in zip(divergence, bound) if b > 1e-30 and b < 1e300]
    final = divergence[-1] / bound[-1] if bound[-1] > 1e-30 and bound[-1] < 1e300 else 0.0
    return (max(ratios) if ratios else 0.0), final


def estimate_raw_L(cache, key, state, step_fn, loss_fn, args, n_perturb, n_steps):
    if key in cache:
        return cache[key]
    start = time.perf_counter()
    torch.manual_seed(args.seed + n_perturb * 1009 + n_steps * 9173 + hash(key[0]) % 1000)
    value = apriori_lipschitz_numerical(
        state,
        step_fn,
        loss_fn,
        n_perturbations=n_perturb,
        n_steps=n_steps,
        eps=args.lipschitz_eps,
    )
    elapsed = time.perf_counter() - start
    cache[key] = (value, elapsed)
    return cache[key]


def build_rows(args):
    schedule = make_constant_schedule(lr=args.lr, eps=args.eps)
    A = make_quadratic_hessian(args.dimension, args.condition_number, args.seed)
    loss_fn = make_quadratic_loss(A)
    step_fn = RMSPropStep(alpha=args.rmsprop_alpha)
    s0 = make_state(args, schedule)
    baseline = run_trajectory(s0.clone(), step_fn, loss_fn, args.total_steps)
    state_clean = baseline.states[args.rewrite_step]
    rewrite = make_rewrite(args)
    delta = rewrite.delta(state_clean)
    state_rewritten = rewrite.apply(state_clean)
    rewritten = run_trajectory(
        s0.clone(), step_fn, loss_fn, args.total_steps, rewrites=[(args.rewrite_step, rewrite)]
    )
    divergence = compute_divergence(baseline, rewritten)
    gaps = loss_gap(baseline.losses, rewritten.losses)
    L_actual = effective_lipschitz_from_divergence(divergence, args.rewrite_step)
    actual_danger = actual_dangerous(divergence, gaps, args)

    margins = parse_float_list(args.safety_margins)
    perturbations = parse_int_list(args.perturbations)
    probe_steps = parse_int_list(args.probe_steps)
    methods = ["clean_state_L", "rewritten_state_L", "max_clean_rewritten_L", "longer_horizon_L"]
    cache = {}
    rows = []

    for n_perturb in perturbations:
        for n_steps in probe_steps:
            clean_raw, clean_rt = estimate_raw_L(
                cache, ("clean", n_perturb, n_steps), state_clean, step_fn, loss_fn, args, n_perturb, n_steps
            )
            rewritten_raw, rewritten_rt = estimate_raw_L(
                cache, ("rewritten", n_perturb, n_steps), state_rewritten, step_fn, loss_fn, args, n_perturb, n_steps
            )
            long_steps = max(n_steps + args.longer_horizon_extra, n_steps * 2)
            long_raw, long_rt = estimate_raw_L(
                cache, ("longer_horizon", n_perturb, long_steps),
                state_clean, step_fn, loss_fn, args, n_perturb, long_steps,
            )
            raw_by_method = {
                "clean_state_L": (clean_raw, clean_rt),
                "rewritten_state_L": (rewritten_raw, rewritten_rt),
                "max_clean_rewritten_L": (max(clean_raw, rewritten_raw), clean_rt + rewritten_rt),
                "longer_horizon_L": (long_raw, long_rt),
            }
            for margin in margins:
                for method in methods:
                    raw_L, runtime = raw_by_method[method]
                    L_pred = raw_L * margin
                    bound = apriori_bound(delta, L_pred, args.rewrite_step, args.total_steps)
                    holds = bound_holds(divergence, bound)
                    max_ratio, final_ratio = bound_ratios(divergence, bound)
                    pred_danger = predicted_dangerous(delta, L_pred, args)
                    row = {
                        "safety_margin": margin,
                        "perturbations": n_perturb,
                        "probe_steps": n_steps,
                        "estimator_method": method,
                        "L_pred": L_pred,
                        "L_raw": raw_L,
                        "L_actual": L_actual,
                        "bound_holds": holds,
                        "max_bound_ratio": max_ratio,
                        "final_bound_ratio": final_ratio,
                        "predicted_dangerous": pred_danger,
                        "actual_dangerous": actual_danger,
                        "false_negative": (not pred_danger) and actual_danger,
                        "false_positive": pred_danger and not actual_danger,
                        "runtime_seconds": runtime,
                    }
                    rows.append(row)

    context = {
        "delta": delta,
        "L_actual": L_actual,
        "actual_dangerous": actual_danger,
        "final_divergence": divergence[-1],
        "max_divergence": max(divergence),
        "final_loss_gap": gaps[-1] if gaps else 0.0,
        "max_loss_gap": max(gaps) if gaps else 0.0,
        "margins": margins,
        "perturbations": perturbations,
        "probe_steps": probe_steps,
    }
    return rows, context


def write_csv(path, rows):
    fields = [
        "safety_margin", "perturbations", "probe_steps", "estimator_method",
        "L_pred", "L_actual", "bound_holds", "max_bound_ratio",
        "final_bound_ratio", "predicted_dangerous", "actual_dangerous",
        "false_negative", "false_positive", "runtime_seconds",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{k: row[k] for k in fields} for row in rows])


def heatmap(rows, margins, perturbations, probe_steps, method, value_fn):
    mat = np.zeros((len(probe_steps), len(perturbations)))
    subset = [r for r in rows if r["estimator_method"] == method]
    # use the smallest margin for L heatmap, and all margins aggregated for boolean heatmaps by max
    for i, steps in enumerate(probe_steps):
        for j, perturb in enumerate(perturbations):
            cell = [r for r in subset if r["probe_steps"] == steps and r["perturbations"] == perturb]
            mat[i, j] = value_fn(cell)
    return mat


def plot_outputs(rows, context, figure_dir):
    os.makedirs(figure_dir, exist_ok=True)
    perturbations = context["perturbations"]
    probe_steps = context["probe_steps"]
    margins = context["margins"]

    base_method = "clean_state_L"
    l_mat = heatmap(
        rows, margins, perturbations, probe_steps, base_method,
        lambda cell: next(r["L_pred"] for r in cell if r["safety_margin"] == margins[0])
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(l_mat, origin="lower", aspect="auto")
    ax.set_xticks(range(len(perturbations)), labels=perturbations)
    ax.set_yticks(range(len(probe_steps)), labels=probe_steps)
    ax.set_xlabel("perturbations")
    ax.set_ylabel("probe steps")
    ax.set_title("RMSProp clean-state L_pred by budget")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(Path(figure_dir) / "l_pred_by_budget.png", dpi=180)
    plt.close(fig)

    fn_mat = heatmap(
        rows, margins, perturbations, probe_steps, base_method,
        lambda cell: sum(1 for r in cell if r["false_negative"])
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(fn_mat, origin="lower", aspect="auto", cmap="Reds")
    ax.set_xticks(range(len(perturbations)), labels=perturbations)
    ax.set_yticks(range(len(probe_steps)), labels=probe_steps)
    ax.set_xlabel("perturbations")
    ax.set_ylabel("probe steps")
    ax.set_title("False negatives across margins (clean-state L)")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(Path(figure_dir) / "false_negative_heatmap.png", dpi=180)
    plt.close(fig)

    bf_mat = heatmap(
        rows, margins, perturbations, probe_steps, base_method,
        lambda cell: sum(1 for r in cell if not r["bound_holds"])
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(bf_mat, origin="lower", aspect="auto", cmap="Reds")
    ax.set_xticks(range(len(perturbations)), labels=perturbations)
    ax.set_yticks(range(len(probe_steps)), labels=probe_steps)
    ax.set_xlabel("perturbations")
    ax.set_ylabel("probe steps")
    ax.set_title("Bound failures across margins (clean-state L)")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(Path(figure_dir) / "bound_failure_heatmap.png", dpi=180)
    plt.close(fig)

    methods = sorted(set(r["estimator_method"] for r in rows))
    data = [[r["max_bound_ratio"] for r in rows if r["estimator_method"] == m] for m in methods]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(data, tick_labels=methods, showfliers=True)
    ax.set_yscale("log")
    ax.set_ylabel("max D/B")
    ax.set_title("Max bound ratio by estimator method")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(Path(figure_dir) / "max_bound_ratio_by_method.png", dpi=180)
    plt.close(fig)


def summarize(rows, context):
    fixes = [r for r in rows if not r["false_negative"] and r["bound_holds"]]
    cheapest = min(
        fixes,
        key=lambda r: (r["runtime_seconds"], r["perturbations"] * r["probe_steps"], r["safety_margin"]),
        default=None,
    )
    by_method = {}
    for method in sorted(set(r["estimator_method"] for r in rows)):
        subset = [r for r in rows if r["estimator_method"] == method]
        by_method[method] = {
            "rows": len(subset),
            "false_negatives": sum(1 for r in subset if r["false_negative"]),
            "bound_failures": sum(1 for r in subset if not r["bound_holds"]),
            "min_margin_fixing_false_negative": min(
                (r["safety_margin"] for r in subset if not r["false_negative"]),
                default=None,
            ),
        }
    return {
        "total_rows": len(rows),
        "context": context,
        "false_negatives": sum(1 for r in rows if r["false_negative"]),
        "bound_failures": sum(1 for r in rows if not r["bound_holds"]),
        "cheapest_configuration_fixing_false_negative_and_bound": cheapest,
        "by_method": by_method,
    }


def run(args):
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.figure_dir, exist_ok=True)
    rows, context = build_rows(args)
    out = Path(args.output_dir)
    write_csv(out / "rmsprop_calibration_metrics.csv", rows)
    with open(out / "rmsprop_calibration_metrics.json", "w") as f:
        json.dump({"rows": rows}, f, indent=2)
    summary = summarize(rows, context)
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(out / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    plot_outputs(rows, context, args.figure_dir)

    print(f"Rows: {len(rows)}")
    print(f"False negatives: {summary['false_negatives']}")
    print(f"Bound failures: {summary['bound_failures']}")
    if summary["cheapest_configuration_fixing_false_negative_and_bound"]:
        c = summary["cheapest_configuration_fixing_false_negative_and_bound"]
        print(
            "Cheapest fix: "
            f"method={c['estimator_method']} margin={c['safety_margin']} "
            f"pert={c['perturbations']} steps={c['probe_steps']} "
            f"L={c['L_pred']:.6f} runtime={c['runtime_seconds']:.4f}s"
        )
    print(f"Saved metrics -> {out}")
    print(f"Saved figures -> {Path(args.figure_dir)}")
    return {"rows": rows, "summary": summary}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--delta", type=float, default=0.01)
    p.add_argument("--rewrite-step", type=int, default=100)
    p.add_argument("--total-steps", type=int, default=200)
    p.add_argument("--dimension", type=int, default=10)
    p.add_argument("--condition-number", type=float, default=100.0)
    p.add_argument("--lr", type=float, default=0.021)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--rmsprop-alpha", type=float, default=0.99)
    p.add_argument("--safety-margins", default=",".join(str(v) for v in DEFAULT_MARGINS))
    p.add_argument("--perturbations", default=",".join(str(v) for v in DEFAULT_PERTURBATIONS))
    p.add_argument("--probe-steps", default=",".join(str(v) for v in DEFAULT_STEPS))
    p.add_argument("--longer-horizon-extra", type=int, default=10)
    p.add_argument("--lipschitz-eps", type=float, default=1e-4)
    p.add_argument("--danger-loss-gap-threshold", type=float, default=0.01)
    p.add_argument("--danger-divergence-threshold", type=float, default=1.0)
    p.add_argument("--output-dir", default="results/rmsprop_calibration")
    p.add_argument("--figure-dir", default="figures/rmsprop_calibration")
    p.add_argument("--small", action="store_true")
    args = p.parse_args()
    if args.small:
        args.safety_margins = "1.0,1.08,1.25"
        args.perturbations = "3,10"
        args.probe_steps = "1,5,15"
        args.output_dir = "results/rmsprop_calibration_small"
        args.figure_dir = "figures/rmsprop_calibration_small"
    return args


if __name__ == "__main__":
    run(parse_args())
