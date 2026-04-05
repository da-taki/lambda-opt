"""A priori L vs actual L → Plot 8. Uses numerical estimator."""
import argparse, json, os, yaml, torch, glob
from pathlib import Path
from src import (
    TrainingState, make_constant_schedule, AdamStep,
    EMARewrite,
    run_trajectory, compute_divergence,
    apriori_lipschitz_numerical,
    make_quadratic_hessian, make_quadratic_loss,
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

        # numerical a priori L
        L_pred = apriori_lipschitz_numerical(
            state_at_tR, step_fn, loss_fn,
            n_perturbations=20, n_steps=15)

        # actual L from divergence
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


def run(configs, out):
    os.makedirs(out, exist_ok=True)
    cfgs = sorted(glob.glob(str(Path(configs) / "quadratic_*.yaml")))
    if not cfgs:
        print(f"No quadratic configs in {configs}")
        return
    all_points = []
    for c in cfgs:
        print(f"\n--- {Path(c).stem} ---")
        all_points.extend(analyze_config(c))
    results = {"points": all_points}
    out_path = Path(out) / "apriori_vs_actual_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--configs", default="configs")
    p.add_argument("--out", default="results")
    run(**vars(p.parse_args()))