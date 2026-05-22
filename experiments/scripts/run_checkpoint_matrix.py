"""Checkpoint corruption matrix.

Broad practical checkpoint/resume corruption benchmark for lambdaOpt.

Default mode is intentionally small: ResNet-18 on CIFAR-shaped synthetic data
with reduced severities. Use --full for CIFAR-10 and the full severity matrix.

Usage:
    cd experiments
    py scripts/run_checkpoint_matrix.py --small
    py scripts/run_checkpoint_matrix.py --full
"""

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import (
    AdamStep,
    CheckpointRewrite,
    EMARewrite,
    LoRAFreezeRewrite,
    TrainingState,
    apriori_bound,
    apriori_lipschitz_numerical,
    bound_holds,
    make_constant_schedule,
)
from src.bounds import classify_regime


SAFETY = 1.08


@dataclass
class Scenario:
    family: str
    name: str
    severity: str
    notes: str
    kind: str
    expected_benign: bool = False
    moments: torch.Tensor | None = None
    theta: torch.Tensor | None = None
    schedule: object | None = None


def flatten_params(model):
    return torch.cat([p.data.reshape(-1).detach().cpu() for p in model.parameters()])


def unflatten_params(model, flat):
    flat = flat.detach().cpu()
    idx = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(flat[idx:idx + n].reshape(p.shape).to(p.device))
        idx += n


def load_model_and_batches(args):
    from torchvision import datasets, models, transforms

    torch.manual_seed(args.seed)
    model = models.resnet18(num_classes=10)
    criterion = nn.CrossEntropyLoss()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    needed = args.pretrain_steps + args.post_steps + args.lipschitz_steps + 5
    if args.full:
        dataset = datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform
        )
        data_note = "CIFAR-10 train split"
    else:
        dataset = datasets.FakeData(
            size=max(needed * args.batch_size, args.batch_size),
            image_size=(3, 32, 32),
            num_classes=10,
            transform=transform,
            random_offset=args.seed,
        )
        data_note = "synthetic CIFAR-shaped FakeData for small/default mode"

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    batches = []
    for x, y in loader:
        batches.append((x, y))
        if len(batches) >= needed:
            break
    if len(batches) < args.pretrain_steps + args.post_steps:
        raise RuntimeError("Not enough cached batches for requested steps")

    return model, criterion, batches, data_note


def make_loss_fn(model, criterion, batches, counter):
    def loss_fn(theta):
        unflatten_params(model, theta)
        idx = counter[0] % len(batches)
        counter[0] += 1
        x, y = batches[idx]
        model.zero_grad(set_to_none=True)
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        grad = torch.cat([p.grad.reshape(-1).detach().cpu() for p in model.parameters()])
        return loss.item(), grad
    return loss_fn


def train_to_resume(args, model, criterion, batches, schedule, step_fn, n_params):
    theta0 = flatten_params(model)
    state = TrainingState(
        theta=theta0.clone(),
        moments=torch.zeros(n_params, 2),
        schedule_fn=schedule,
        t=0,
    )
    ema_decays = sorted(set(args.ema_decays))
    ema = {decay: theta0.clone() for decay in ema_decays}
    checkpoints = {0: {"theta": state.theta.clone(), "moments": state.moments.clone(), "t": state.t}}
    losses = []

    counter = [0]
    loss_fn = make_loss_fn(model, criterion, batches, counter)
    for step in range(args.pretrain_steps):
        checkpoints[step] = {
            "theta": state.theta.clone(),
            "moments": state.moments.clone(),
            "t": state.t,
        }
        loss, grad = loss_fn(state.theta)
        losses.append(loss)
        state, _ = step_fn(state, grad)
        for decay in ema_decays:
            ema[decay] = decay * ema[decay] + (1.0 - decay) * state.theta
        if step % max(1, args.pretrain_steps // 5) == 0:
            print(f"  pretrain step {step}/{args.pretrain_steps} loss={loss:.4f}", end="\r")

    checkpoints[args.pretrain_steps] = {
        "theta": state.theta.clone(),
        "moments": state.moments.clone(),
        "t": state.t,
    }
    print(f"  pretrain step {args.pretrain_steps}/{args.pretrain_steps} done")
    return state.clone(), checkpoints, ema, losses


def scenario_values(args):
    if args.full:
        return {
            "stale_by": [10, 25, 50, 100, 200],
            "lr_multipliers": [2.0, 5.0, 10.0],
            "ema_decays": [0.9, 0.99, 0.999],
            "freeze_fracs": [0.1, 0.25, 0.5, 1.0],
        }
    return {
        "stale_by": [10, 25],
        "lr_multipliers": [2.0, 10.0],
        "ema_decays": [0.9, 0.999],
        "freeze_fracs": [0.25, 1.0],
    }


def build_scenarios(args, resume_state, checkpoints, ema, base_schedule, n_params):
    values = scenario_values(args)
    scenarios = []

    scenarios.append(Scenario(
        family="benign",
        name="identity_noop_restore",
        severity="none",
        notes="No-op checkpoint restore; state and schedule unchanged",
        kind="identity",
        expected_benign=True,
    ))
    scenarios.append(Scenario(
        family="benign",
        name="exact_checkpoint_restore",
        severity="matching_state",
        notes="Restore checkpoint with matching optimizer state at resume",
        kind="moments",
        moments=resume_state.moments.clone(),
        expected_benign=True,
    ))
    scenarios.append(Scenario(
        family="benign",
        name="scheduler_resume_no_mismatch",
        severity="1x",
        notes="Resume with identical learning-rate schedule",
        kind="identity",
        expected_benign=True,
    ))
    scenarios.append(Scenario(
        family="benign",
        name="ema_shadow_present_not_loaded",
        severity="not_loaded",
        notes="Synthetic EMA shadow exists but parameters are unchanged",
        kind="identity",
        expected_benign=True,
    ))
    mild_lr_schedule = make_constant_schedule(
        lr=args.lr * 0.5, beta1=args.beta1, beta2=args.beta2, eps=args.eps
    )
    scenarios.append(Scenario(
        family="benign",
        name="mild_lr_decrease_0.5x",
        severity="0.5x",
        notes="Resume with a mild LR decrease; included as a specificity stress row",
        kind="schedule",
        schedule=mild_lr_schedule,
        expected_benign=True,
    ))

    for stale_by in values["stale_by"]:
        stale_step = max(0, args.pretrain_steps - stale_by)
        stale = checkpoints[stale_step]["moments"].clone()
        scenarios.append(Scenario(
            family="stale_adam_moments",
            name=f"stale_by_{stale_by}",
            severity=str(stale_by),
            notes=f"Adam moments from step {stale_step}; requested stale_by={stale_by}",
            kind="moments",
            moments=stale,
        ))

    m_reset = resume_state.moments.clone()
    m_reset[:, 0] = 0.0
    scenarios.append(Scenario(
        family="moment_reset", name="reset_m_only", severity="m",
        notes="Reset first Adam moment only", kind="moments", moments=m_reset,
    ))

    v_reset = resume_state.moments.clone()
    v_reset[:, 1] = 0.0
    scenarios.append(Scenario(
        family="moment_reset", name="reset_v_only", severity="v",
        notes="Reset second Adam moment only", kind="moments", moments=v_reset,
    ))

    both_reset = torch.zeros_like(resume_state.moments)
    scenarios.append(Scenario(
        family="moment_reset", name="reset_m_and_v", severity="m+v",
        notes="Reset both Adam moments", kind="moments", moments=both_reset,
    ))

    for mult in values["lr_multipliers"]:
        wrong_schedule = make_constant_schedule(
            lr=args.lr * mult, beta1=args.beta1, beta2=args.beta2, eps=args.eps
        )
        scenarios.append(Scenario(
            family="scheduler_mismatch",
            name=f"lr_{mult:g}x",
            severity=f"{mult:g}x",
            notes="State unchanged; resume schedule uses multiplied learning rate",
            kind="schedule",
            schedule=wrong_schedule,
        ))

    for decay in values["ema_decays"]:
        theta_ema = ema[decay].clone()
        scenarios.append(Scenario(
            family="ema_as_parameters",
            name=f"ema_decay_{decay}",
            severity=str(decay),
            notes="Synthetic EMA shadow loaded as model parameters",
            kind="theta",
            theta=theta_ema,
        ))

    for frac in values["freeze_fracs"]:
        k = max(1, int(n_params * frac))
        frozen_indices = torch.arange(k)
        rewrite = LoRAFreezeRewrite(frozen_indices=frozen_indices)
        s_lora = rewrite.apply(resume_state)
        scenarios.append(Scenario(
            family="synthetic_lora_moment_freeze",
            name=f"freeze_frac_{frac:g}",
            severity=f"{frac:g}",
            notes="Synthetic optimizer-row freezing, not real LoRA adapter training",
            kind="moments",
            moments=s_lora.moments,
        ))

    return scenarios


def apply_scenario(scenario, resume_state, base_schedule):
    if scenario.kind == "identity":
        return resume_state.clone(), 0.0
    if scenario.kind == "moments":
        rewrite = CheckpointRewrite(stale_moments=scenario.moments, stale_t=resume_state.t)
        new_state = rewrite.apply(resume_state)
        new_state.t = resume_state.t
        return new_state, rewrite.delta(resume_state)
    if scenario.kind == "theta":
        rewrite = EMARewrite(theta_ema=scenario.theta, alpha=1.0)
        return rewrite.apply(resume_state), rewrite.delta(resume_state)
    if scenario.kind == "schedule":
        new_state = TrainingState(
            theta=resume_state.theta.clone(),
            moments=resume_state.moments.clone(),
            schedule_fn=scenario.schedule,
            t=resume_state.t,
        )
        return new_state, 0.0
    raise ValueError(f"Unknown scenario kind: {scenario.kind}")


def run_post_trajectories(args, model, criterion, batches, step_fn, base_state, scenario_state):
    base_counter = [args.pretrain_steps]
    scenario_counter = [args.pretrain_steps]
    loss_base = make_loss_fn(model, criterion, batches, base_counter)
    loss_scenario = make_loss_fn(model, criterion, batches, scenario_counter)

    s_base = base_state.clone()
    s_scenario = scenario_state.clone()
    divergence = []
    base_losses = []
    scenario_losses = []

    for _ in range(args.post_steps):
        divergence.append((s_scenario.theta - s_base.theta).norm(2).item())
        loss_b, grad_b = loss_base(s_base.theta)
        loss_s, grad_s = loss_scenario(s_scenario.theta)
        base_losses.append(loss_b)
        scenario_losses.append(loss_s)
        s_base, _ = step_fn(s_base, grad_b)
        s_scenario, _ = step_fn(s_scenario, grad_s)

    divergence.append((s_scenario.theta - s_base.theta).norm(2).item())
    loss_gaps = [abs(a - b) for a, b in zip(base_losses, scenario_losses)]
    return divergence, loss_gaps


def safe_power(base, exp):
    if exp == 0:
        return 1.0
    if base <= 0:
        return 0.0
    log_val = exp * math.log(base)
    if log_val > math.log(1e300):
        return 1e300
    if log_val < -700:
        return 0.0
    return math.exp(log_val)


def predict_dangerous(delta, L_pred, regime, args):
    predicted_final_bound = delta * safe_power(L_pred, args.post_steps)
    return (
        regime == "expansive"
        or predicted_final_bound >= args.danger_divergence_threshold
    ), predicted_final_bound


def actual_dangerous(divergence, loss_gaps, args):
    final_div = divergence[-1]
    max_loss_gap = max(loss_gaps) if loss_gaps else 0.0
    return (
        final_div >= args.danger_divergence_threshold
        or max_loss_gap >= args.danger_loss_gap_threshold
    )


def bound_diagnostics(divergence, bound):
    ratios = []
    first_violation_step = None
    max_ratio = 0.0
    max_violation_ratio = 0.0

    for step, (d, b) in enumerate(zip(divergence, bound)):
        if b > 1e-30 and b < 1e300:
            ratio = d / b
            ratios.append(ratio)
            max_ratio = max(max_ratio, ratio)
            if ratio > 1.0:
                max_violation_ratio = max(max_violation_ratio, ratio)
                if first_violation_step is None:
                    first_violation_step = step

    final_bound_ratio = 0.0
    if bound and bound[-1] > 1e-30 and bound[-1] < 1e300:
        final_bound_ratio = divergence[-1] / bound[-1]

    return {
        "max_bound_violation_ratio": max_violation_ratio,
        "first_violation_step": first_violation_step,
        "final_bound_ratio": final_bound_ratio,
        "max_bound_ratio": max_ratio,
    }


def write_csv(path, rows):
    fields = [
        "scenario_family", "scenario_name", "severity", "delta", "L_pred",
        "regime", "predicted_safe", "predicted_dangerous",
        "actual_final_divergence", "actual_max_divergence",
        "final_loss_gap", "max_loss_gap", "actual_dangerous",
        "expected_benign",
        "false_positive", "false_negative", "bound_holds",
        "max_bound_violation_ratio", "first_violation_step",
        "final_bound_ratio", "max_bound_ratio", "notes",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fields})


def plot_outputs(rows, curves, figure_dir):
    os.makedirs(figure_dir, exist_ok=True)
    families = sorted(set(row["scenario_family"] for row in rows))

    fig, axes = plt.subplots(len(families), 1, figsize=(9, max(3, 2.6 * len(families))), sharex=True)
    if len(families) == 1:
        axes = [axes]
    for ax, family in zip(axes, families):
        for key, data in curves.items():
            if data["family"] != family:
                continue
            ax.semilogy([max(v, 1e-15) for v in data["divergence"]], lw=1.2, label=data["name"])
        ax.set_title(family)
        ax.set_ylabel("D(t)")
        ax.legend(fontsize=7, ncol=2)
    axes[-1].set_xlabel("post-resume step")
    fig.tight_layout()
    fig.savefig(Path(figure_dir) / "divergence_by_scenario_family.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(len(families), 1, figsize=(9, max(3, 2.6 * len(families))), sharex=True)
    if len(families) == 1:
        axes = [axes]
    for ax, family in zip(axes, families):
        for key, data in curves.items():
            if data["family"] != family:
                continue
            ax.semilogy([max(v, 1e-15) for v in data["loss_gap"]], lw=1.2, label=data["name"])
        ax.set_title(family)
        ax.set_ylabel("|loss gap|")
        ax.legend(fontsize=7, ncol=2)
    axes[-1].set_xlabel("post-resume step")
    fig.tight_layout()
    fig.savefig(Path(figure_dir) / "loss_gap_by_scenario_family.png", dpi=180)
    plt.close(fig)

    fp = sum(1 for row in rows if row["false_positive"])
    fn = sum(1 for row in rows if row["false_negative"])
    tp = sum(1 for row in rows if row["predicted_dangerous"] and row["actual_dangerous"])
    tn = sum(1 for row in rows if row["predicted_safe"] and not row["actual_dangerous"])
    fig, ax = plt.subplots(figsize=(5, 4))
    matrix = torch.tensor([[tn, fp], [fn, tp]], dtype=torch.float32)
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], labels=["pred safe", "pred dangerous"])
    ax.set_yticks([0, 1], labels=["actual safe", "actual dangerous"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(matrix[i, j].item()), ha="center", va="center", color="black")
    ax.set_title("Checkpoint matrix confusion summary")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(Path(figure_dir) / "confusion_matrix.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {family: f"C{i}" for i, family in enumerate(families)}
    for row in rows:
        ax.scatter(
            row["delta"], row["actual_final_divergence"],
            c=colors[row["scenario_family"]],
            label=row["scenario_family"],
            alpha=0.8,
        )
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), fontsize=8)
    ax.set_xlabel("delta")
    ax.set_ylabel("final divergence")
    ax.set_title("Delta vs final divergence")
    fig.tight_layout()
    fig.savefig(Path(figure_dir) / "delta_vs_final_divergence.png", dpi=180)
    plt.close(fig)


def summarize(rows):
    actual_dangerous = sum(1 for row in rows if row["actual_dangerous"])
    actual_safe = sum(1 for row in rows if not row["actual_dangerous"])
    true_positives = sum(1 for row in rows if row["predicted_dangerous"] and row["actual_dangerous"])
    true_negatives = sum(1 for row in rows if row["predicted_safe"] and not row["actual_dangerous"])
    false_positives = sum(1 for row in rows if row["false_positive"])
    false_negatives = sum(1 for row in rows if row["false_negative"])
    benign_rows = [row for row in rows if row["expected_benign"]]
    benign_actual_safe = sum(1 for row in benign_rows if not row["actual_dangerous"])
    benign_predicted_safe = sum(1 for row in benign_rows if row["predicted_safe"])
    return {
        "total_scenarios": len(rows),
        "predicted_dangerous": sum(1 for row in rows if row["predicted_dangerous"]),
        "predicted_safe": sum(1 for row in rows if row["predicted_safe"]),
        "actual_dangerous": actual_dangerous,
        "actual_safe": actual_safe,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "true_positives": true_positives,
        "true_negatives": true_negatives,
        "specificity": true_negatives / actual_safe if actual_safe else None,
        "sensitivity": true_positives / actual_dangerous if actual_dangerous else None,
        "total_benign_rows": len(benign_rows),
        "benign_predicted_safe": benign_predicted_safe,
        "benign_predicted_dangerous": len(benign_rows) - benign_predicted_safe,
        "benign_actual_safe": benign_actual_safe,
        "benign_actual_dangerous": len(benign_rows) - benign_actual_safe,
        "bound_failures": sum(1 for row in rows if not row["bound_holds"]),
    }


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

    lip_counter = [args.pretrain_steps]
    lip_loss = make_loss_fn(model, criterion, batches, lip_counter)
    L_raw = apriori_lipschitz_numerical(
        resume_state,
        step_fn,
        lip_loss,
        n_perturbations=args.lipschitz_perturbations,
        n_steps=args.lipschitz_steps,
        eps=args.lipschitz_eps,
    )
    L_pred_base = L_raw * args.safety_margin
    print(f"Base L_pred={L_pred_base:.6f}")

    rows = []
    curves = {}

    for scenario in scenarios:
        scenario_state, delta = apply_scenario(scenario, resume_state, base_schedule)

        if scenario.kind == "schedule":
            lip_counter = [args.pretrain_steps]
            lip_loss = make_loss_fn(model, criterion, batches, lip_counter)
            L_raw_s = apriori_lipschitz_numerical(
                scenario_state,
                step_fn,
                lip_loss,
                n_perturbations=args.lipschitz_perturbations,
                n_steps=args.lipschitz_steps,
                eps=args.lipschitz_eps,
            )
            L_pred = L_raw_s * args.safety_margin
        else:
            L_pred = L_pred_base

        regime = classify_regime(L_pred)
        if scenario.kind == "identity" or (delta <= 1e-12 and scenario.kind != "schedule"):
            pred_danger, pred_final_bound = False, 0.0
        else:
            pred_danger, pred_final_bound = predict_dangerous(delta, L_pred, regime, args)

        divergence, loss_gaps = run_post_trajectories(
            args, model, criterion, batches, step_fn, resume_state, scenario_state
        )
        bnd = apriori_bound(delta, L_pred, 0, args.post_steps)
        holds = bound_holds(divergence, bnd) if delta > 0 else True
        bound_diag = bound_diagnostics(divergence, bnd)
        actual_danger = actual_dangerous(divergence, loss_gaps, args)

        row = {
            "scenario_family": scenario.family,
            "scenario_name": scenario.name,
            "severity": scenario.severity,
            "delta": delta,
            "L_pred": L_pred,
            "regime": regime,
            "predicted_safe": not pred_danger,
            "predicted_dangerous": pred_danger,
            "actual_final_divergence": divergence[-1],
            "actual_max_divergence": max(divergence),
            "final_loss_gap": loss_gaps[-1] if loss_gaps else 0.0,
            "max_loss_gap": max(loss_gaps) if loss_gaps else 0.0,
            "actual_dangerous": actual_danger,
            "expected_benign": scenario.expected_benign,
            "false_positive": pred_danger and not actual_danger,
            "false_negative": (not pred_danger) and actual_danger,
            "bound_holds": holds,
            **bound_diag,
            "notes": (
                f"{scenario.notes}; predicted_final_bound={pred_final_bound:.6g}; "
                f"actual_dangerous iff final_divergence >= {args.danger_divergence_threshold} "
                f"or max_loss_gap >= {args.danger_loss_gap_threshold}"
            ),
        }
        rows.append(row)

        key = f"{scenario.family}/{scenario.name}"
        curves[key] = {
            "family": scenario.family,
            "name": scenario.name,
            "divergence": divergence,
            "loss_gap": loss_gaps,
            "bound": bnd,
        }

        print(
            f"{scenario.family:30s} {scenario.name:18s} "
            f"delta={delta:.4f} L={L_pred:.3f} "
            f"pred={'danger' if pred_danger else 'safe'} "
            f"actual={'danger' if actual_danger else 'safe'} "
            f"FN={row['false_negative']}"
        )

    summary = summarize(rows)
    config = {
        "mode": "full" if args.full else "small",
        "model": "resnet18",
        "dataset": data_note,
        "seed": args.seed,
        "pretrain_steps": args.pretrain_steps,
        "post_steps": args.post_steps,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "beta1": args.beta1,
        "beta2": args.beta2,
        "eps": args.eps,
        "safety_margin": args.safety_margin,
        "L_pred_base": L_pred_base,
        "danger_definition": {
            "actual_dangerous": (
                "final_divergence >= danger_divergence_threshold "
                "or max_loss_gap >= danger_loss_gap_threshold"
            ),
            "danger_divergence_threshold": args.danger_divergence_threshold,
            "danger_loss_gap_threshold": args.danger_loss_gap_threshold,
            "predicted_dangerous": (
                "regime == expansive or delta * L_pred^post_steps >= danger_divergence_threshold"
            ),
        },
        "scenario_values": values,
        "benign_rows": [
            "identity_noop_restore",
            "exact_checkpoint_restore",
            "scheduler_resume_no_mismatch",
            "ema_shadow_present_not_loaded",
            "mild_lr_decrease_0.5x",
        ],
    }

    output_dir = Path(args.output_dir)
    write_csv(output_dir / "matrix_metrics.csv", rows)
    with open(output_dir / "matrix_metrics.json", "w") as f:
        json.dump({"rows": rows, "curves": curves}, f, indent=2)
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    plot_outputs(rows, curves, args.figure_dir)

    print("\nSummary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print(f"\nSaved metrics -> {output_dir}")
    print(f"Saved figures -> {Path(args.figure_dir)}")
    return {"rows": rows, "summary": summary}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pretrain-steps", type=int, default=4)
    p.add_argument("--post-steps", type=int, default=2)
    p.add_argument("--resume-step", type=int, default=None,
                   help="Alias reserved for compatibility; defaults to pretrain-steps.")
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
    p.add_argument("--output-dir", default="results/checkpoint_matrix")
    p.add_argument("--figure-dir", default="figures/checkpoint_matrix")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--small", action="store_true", default=True,
                      help="Reduced matrix on synthetic CIFAR-shaped data; default.")
    mode.add_argument("--full", action="store_true",
                      help="Full severity matrix on CIFAR-10.")
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
