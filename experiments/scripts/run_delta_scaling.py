"""δ-scaling → Plot 5: D(T) vs δ linearity test."""
import argparse, json, os, yaml, torch
from pathlib import Path
from src import (
    TrainingState, make_constant_schedule, AdamStep,
    EMARewrite, run_trajectory, compute_divergence,
    delta_linearity_r2,
    make_quadratic_hessian, make_quadratic_loss,
)


def run(cfg_path, out_dir):
    cfg = yaml.safe_load(open(cfg_path))
    os.makedirs(out_dir, exist_ok=True)
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

    t_R = int(T * 0.5)
    torch.manual_seed(123)
    direction = torch.randn(n)
    direction = direction / direction.norm(2)

    scales = cfg["delta_scaling"]["scales"]
    deltas, div_at_T = [], []

    for scale in scales:
        theta_target = baseline.states[t_R].theta + direction * scale
        R = EMARewrite(theta_ema=theta_target, alpha=1.0)
        delta = R.delta(baseline.states[t_R])

        torch.manual_seed(seed)
        rw_log = run_trajectory(s0.clone(), step_fn, loss_fn, T, rewrites=[(t_R, R)])
        div = compute_divergence(baseline, rw_log)

        deltas.append(delta)
        div_at_T.append(div[-1])
        print(f"  scale={scale:.4f}  delta={delta:.8f}  D(T)={div[-1]:.8f}")

    r2 = delta_linearity_r2(deltas, div_at_T)
    print(f"\nR2 = {r2:.4f} ({'PASS' if r2 >= 0.95 else 'FAIL'})")

    results = {
        "t_R": t_R, "scales": scales,
        "deltas": deltas, "divergence_at_T": div_at_T, "R2": r2,
    }
    out_path = Path(out_dir) / "delta_scaling_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/quadratic_contractive.yaml")
    p.add_argument("--out", default="results")
    run(**vars(p.parse_args()))