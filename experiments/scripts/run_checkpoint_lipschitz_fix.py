"""Rewritten-state Lipschitz diagnostic for checkpoint corruptions.

Compares bounds based on L_clean, L_rewritten, and max(L_clean, L_rewritten)
for the checkpoint corruption matrix, with special focus on reset_v_only.

Usage:
    cd experiments
    py scripts/run_checkpoint_lipschitz_fix.py
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import AdamStep, apriori_bound, apriori_lipschitz_numerical, bound_holds, make_constant_schedule
from run_checkpoint_matrix import (
    actual_dangerous,
    apply_scenario,
    bound_diagnostics,
    build_scenarios,
    load_model_and_batches,
    run_post_trajectories,
    scenario_values,
    train_to_resume,
)


SAFETY = 1.08


def estimate_L(state, step_fn, loss_fn, args):
    return apriori_lipschitz_numerical(
        state,
        step_fn,
        loss_fn,
        n_perturbations=args.lipschitz_perturbations,
        n_steps=args.lipschitz_steps,
        eps=args.lipschitz_eps,
    ) * args.safety_margin


def make_loss_fn_from_matrix_module(model, criterion, batches, counter):
    from run_checkpoint_matrix import make_loss_fn
    return make_loss_fn(model, criterion, batches, counter)


def prediction_from_bound(final_bound, regime, args):
    return regime == "expansive" or final_bound >= args.danger_divergence_threshold


def evaluate_bound(method, delta, L_value, divergence, args):
    bound = apriori_bound(delta, L_value, 0, args.post_steps)
    diag = bound_diagnostics(divergence, bound)
    return {
        f"bound_holds_{method}": bound_holds(divergence, bound) if delta > 0 else True,
        f"max_bound_ratio_{method}": diag["max_bound_ratio"],
        f"first_violation_step_{method}": diag["first_violation_step"],
        f"final_bound_{method}": bound[-1],
    }, bound


def write_csv(path, rows):
    fields = [
        "scenario_family", "scenario_name", "severity", "delta",
        "L_clean", "L_rewritten", "L_max",
        "bound_holds_clean", "bound_holds_rewritten", "bound_holds_max",
        "max_bound_ratio_clean", "max_bound_ratio_rewritten", "max_bound_ratio_max",
        "first_violation_step_clean", "first_violation_step_rewritten", "first_violation_step_max",
        "actual_final_divergence", "actual_max_divergence",
        "final_loss_gap", "max_loss_gap", "actual_dangerous",
        "false_negative_clean", "false_negative_rewritten", "false_negative_max",
        "notes",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fields})


def write_curves(path, curves):
    fields = ["scenario_family", "scenario_name", "step", "divergence", "loss_gap", "bound_clean", "bound_rewritten", "bound_max"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(curves)


def summarize(rows):
    methods = ["clean", "rewritten", "max"]
    return {
        "total_scenarios": len(rows),
        "bound_failures": {
            m: sum(1 for r in rows if not r[f"bound_holds_{m}"])
            for m in methods
        },
        "false_negatives": {
            m: sum(1 for r in rows if r[f"false_negative_{m}"])
            for m in methods
        },
        "reset_v_only": next((r for r in rows if r["scenario_name"] == "reset_v_only"), None),
    }


def plot_outputs(rows, curve_rows, figure_dir):
    os.makedirs(figure_dir, exist_ok=True)
    methods = ["clean", "rewritten", "max"]

    failures = [sum(1 for r in rows if not r[f"bound_holds_{m}"]) for m in methods]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(methods, failures, color=["#9ca3af", "#2563eb", "#16a34a"])
    ax.set_ylabel("bound failures")
    ax.set_title("Bound failures by L-estimation method")
    fig.tight_layout()
    fig.savefig(Path(figure_dir) / "bound_failures_by_method.png", dpi=180)
    plt.close(fig)

    ratios = [[r[f"max_bound_ratio_{m}"] for r in rows] for m in methods]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.boxplot(ratios, tick_labels=methods, showfliers=True)
    ax.set_yscale("log")
    ax.set_ylabel("max D/B")
    ax.set_title("Max bound ratio by method")
    fig.tight_layout()
    fig.savefig(Path(figure_dir) / "max_bound_ratio_by_method.png", dpi=180)
    plt.close(fig)

    reset_rows = [
        r for r in curve_rows
        if r["scenario_family"] == "moment_reset" and r["scenario_name"] == "reset_v_only"
    ]
    if reset_rows:
        reset_rows = sorted(reset_rows, key=lambda r: r["step"])
        fig, ax = plt.subplots(figsize=(8, 5))
        steps = [r["step"] for r in reset_rows]
        ax.semilogy(steps, [max(r["divergence"], 1e-15) for r in reset_rows], "k-", lw=2, label="D(t)")
        ax.semilogy(steps, [max(r["bound_clean"], 1e-15) for r in reset_rows], "--", label="clean L")
        ax.semilogy(steps, [max(r["bound_rewritten"], 1e-15) for r in reset_rows], "--", label="rewritten L")
        ax.semilogy(steps, [max(r["bound_max"], 1e-15) for r in reset_rows], "--", label="max L")
        ax.set_xlabel("post-resume step")
        ax.set_ylabel("D(t), B(t)")
        ax.set_title("reset_v_only bounds")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(Path(figure_dir) / "reset_v_only_bounds.png", dpi=180)
        plt.close(fig)


def run(args):
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.figure_dir, exist_ok=True)

    values = scenario_values(args)
    args.ema_decays = values["ema_decays"]

    model, criterion, batches, data_note = load_model_and_batches(args)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: ResNet-18, params={n_params:,}, data={data_note}")

    base_schedule = make_constant_schedule(
        lr=args.lr, beta1=args.beta1, beta2=args.beta2, eps=args.eps
    )
    step_fn = AdamStep()
    resume_state, checkpoints, ema, _ = train_to_resume(
        args, model, criterion, batches, base_schedule, step_fn, n_params
    )
    scenarios = build_scenarios(args, resume_state, checkpoints, ema, base_schedule, n_params)
    print(f"Scenarios: {len(scenarios)}")

    clean_counter = [args.pretrain_steps]
    clean_loss = make_loss_fn_from_matrix_module(model, criterion, batches, clean_counter)
    L_clean = estimate_L(resume_state, step_fn, clean_loss, args)
    print(f"L_clean={L_clean:.6f}")

    rows = []
    curve_rows = []
    for scenario in scenarios:
        scenario_state, delta = apply_scenario(scenario, resume_state, base_schedule)

        rewritten_counter = [args.pretrain_steps]
        rewritten_loss = make_loss_fn_from_matrix_module(model, criterion, batches, rewritten_counter)
        L_rewritten = estimate_L(scenario_state, step_fn, rewritten_loss, args)
        L_max = max(L_clean, L_rewritten)

        divergence, loss_gaps = run_post_trajectories(
            args, model, criterion, batches, step_fn, resume_state, scenario_state
        )
        actual_danger = actual_dangerous(divergence, loss_gaps, args)

        eval_clean, bound_clean = evaluate_bound("clean", delta, L_clean, divergence, args)
        eval_rewritten, bound_rewritten = evaluate_bound("rewritten", delta, L_rewritten, divergence, args)
        eval_max, bound_max = evaluate_bound("max", delta, L_max, divergence, args)

        pred_clean = prediction_from_bound(bound_clean[-1], "expansive" if L_clean > 1.001 else "contractive", args)
        pred_rewritten = prediction_from_bound(bound_rewritten[-1], "expansive" if L_rewritten > 1.001 else "contractive", args)
        pred_max = prediction_from_bound(bound_max[-1], "expansive" if L_max > 1.001 else "contractive", args)

        row = {
            "scenario_family": scenario.family,
            "scenario_name": scenario.name,
            "severity": scenario.severity,
            "delta": delta,
            "L_clean": L_clean,
            "L_rewritten": L_rewritten,
            "L_max": L_max,
            **eval_clean,
            **eval_rewritten,
            **eval_max,
            "actual_final_divergence": divergence[-1],
            "actual_max_divergence": max(divergence),
            "final_loss_gap": loss_gaps[-1] if loss_gaps else 0.0,
            "max_loss_gap": max(loss_gaps) if loss_gaps else 0.0,
            "actual_dangerous": actual_danger,
            "false_negative_clean": (not pred_clean) and actual_danger,
            "false_negative_rewritten": (not pred_rewritten) and actual_danger,
            "false_negative_max": (not pred_max) and actual_danger,
            "notes": scenario.notes,
        }
        rows.append(row)

        for step, div in enumerate(divergence):
            curve_rows.append({
                "scenario_family": scenario.family,
                "scenario_name": scenario.name,
                "step": step,
                "divergence": div,
                "loss_gap": loss_gaps[step] if step < len(loss_gaps) else "",
                "bound_clean": bound_clean[step],
                "bound_rewritten": bound_rewritten[step],
                "bound_max": bound_max[step],
            })

        print(
            f"{scenario.family:30s} {scenario.name:18s} "
            f"L_rw={L_rewritten:.3f} "
            f"holds clean/rw/max={row['bound_holds_clean']}/{row['bound_holds_rewritten']}/{row['bound_holds_max']}"
        )

    summary = summarize(rows)
    output_dir = Path(args.output_dir)
    write_csv(output_dir / "metrics.csv", rows)
    write_curves(output_dir / "curves.csv", curve_rows)
    with open(output_dir / "metrics.json", "w") as f:
        json.dump({"rows": rows, "curves": curve_rows}, f, indent=2)
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    config = vars(args).copy()
    config.update({
        "dataset": data_note,
        "scenario_values": values,
        "method_note": "clean uses L at clean resume state; rewritten uses L at corrupted state; max uses max(clean, rewritten).",
    })
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    plot_outputs(rows, curve_rows, args.figure_dir)

    print("\nSummary:")
    print(json.dumps(summary["bound_failures"], indent=2))
    print("False negatives:")
    print(json.dumps(summary["false_negatives"], indent=2))
    print(f"\nSaved metrics -> {output_dir}")
    print(f"Saved figures -> {Path(args.figure_dir)}")
    return {"rows": rows, "summary": summary}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pretrain-steps", type=int, default=4)
    p.add_argument("--post-steps", type=int, default=2)
    p.add_argument("--resume-step", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--danger-loss-gap-threshold", type=float, default=0.05)
    p.add_argument("--danger-divergence-threshold", type=float, default=1.0)
    p.add_argument("--lipschitz-perturbations", type=int, default=2)
    p.add_argument("--lipschitz-steps", type=int, default=2)
    p.add_argument("--lipschitz-eps", type=float, default=1e-4)
    p.add_argument("--safety-margin", type=float, default=SAFETY)
    p.add_argument("--output-dir", default="results/checkpoint_lipschitz_fix")
    p.add_argument("--figure-dir", default="figures/checkpoint_lipschitz_fix")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--small", action="store_true", default=True)
    mode.add_argument("--full", action="store_true")
    args = p.parse_args()
    if args.full:
        args.small = False
        if args.pretrain_steps == 4:
            args.pretrain_steps = 220
        if args.post_steps == 2:
            args.post_steps = 80
        if args.batch_size == 2:
            args.batch_size = 32
        if args.lipschitz_perturbations == 2:
            args.lipschitz_perturbations = 5
        if args.lipschitz_steps == 2:
            args.lipschitz_steps = 5
    if args.resume_step is not None and args.resume_step != args.pretrain_steps:
        raise ValueError("This script uses resume_step == pretrain_steps; set --pretrain-steps instead.")
    return args


if __name__ == "__main__":
    run(parse_args())
