"""A priori L vs actual L → Plot 8.
Runs all 3 quadratic configs, computes L_pred from Hessian at multiple
timepoints, measures actual contraction rate from divergence.
Validates: is the spectral prediction accurate?
"""
import argparse, json, os, yaml, torch, glob
from pathlib import Path
from src import (
    TrainingState, make_constant_schedule, AdamStep,
    EMARewrite,
    run_trajectory, compute_divergence,
    apriori_lipschitz_quadratic,
    hessian_eigenvalues_exact, make_quadratic_hessian, make_quadratic_loss,
)
from src.metrics import effective_lipschitz_from_divergence


def analyze_config(cfg_path):
    cfg = yaml.safe_load(open(cfg_path))
    n, T, seed = cfg["model"]["n"], cfg["training"]["T"], cfg["training"]["seed"]
    opt = cfg["optimizer"]
    schedule = make_constant_schedule(lr=opt["lr"], beta1=opt["beta1"], beta2=opt["beta2"])

    A = make_quadratic_hessian(n, cfg["model"]["condition_number"], seed)
    loss_fn = make_quadratic_loss(A)
    step_fn = AdamStep()
    eigs = hessian_eigenvalues_exact(A)

    torch.manual_seed(seed)
    s0 = TrainingState(theta=torch.randn(n)*0.1, moments=torch.zeros(n,2),
                       schedule_fn=schedule, t=0)
    baseline = run_trajectory(s0.clone(), step_fn, loss_fn, T)

    points = []
    for frac in [0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9]:
        t_R = int(T * frac)
        if t_R >= T - 10:
            continue
        state_at_tR = baseline.states[t_R]

        L_pred = apriori_lipschitz_quadratic(
            eigs, lr=opt["lr"], beta1=opt["beta1"], beta2=opt["beta2"],
            eps=opt["eps"], t=state_at_tR.t, state=state_at_tR)

        # measure actual: apply small EMA rewrite, observe contraction
        R = EMARewrite(
            theta_ema=state_at_tR.theta + torch.randn(n) * 0.01,
            alpha=0.999)

        torch.manual_seed(seed)
        rw_log = run_trajectory(s0.clone(), step_fn, loss_fn, T, rewrites=[(t_R, R)])
        div = compute_divergence(baseline, rw_log)
        L_actual = effective_lipschitz_from_divergence(div, t_R)

        points.append({
            "config": str(cfg_path), "frac": frac, "t_R": t_R,
            "L_pred": L_pred, "L_actual": L_actual,
            "lr": opt["lr"], "condition_number": cfg["model"]["condition_number"],
        })
        print(f"  {Path(cfg_path).stem} @ {frac:.0%}: L_pred={L_pred:.4f}, L_actual={L_actual:.4f}")

    return points


def run(config_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    configs = sorted(glob.glob(str(Path(config_dir) / "quadratic_*.yaml")))
    if not configs:
        print(f"No quadratic configs found in {config_dir}")
        return

    all_points = []
    for cfg_path in configs:
        print(f"\n--- {Path(cfg_path).stem} ---")
        all_points.extend(analyze_config(cfg_path))

    results = {"points": all_points}
    out_path = Path(out_dir) / "apriori_vs_actual_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--configs", default="configs")
    p.add_argument("--out", default="results")
    args = p.parse_args()
    run(args.configs, args.out)