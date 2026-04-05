"""Separation experiment → Plot 7.
Adam (contractive) vs SGD (mildly expansive).
"""
import argparse, json, os, yaml, torch
from pathlib import Path
from src import (
    TrainingState, make_constant_schedule, AdamStep, SGDStep,
    EMARewrite,
    run_trajectory, compute_divergence,
    apriori_lipschitz_numerical, apriori_lipschitz_sgd,
    bound_holds,
    make_quadratic_hessian, make_quadratic_loss,
    hessian_eigenvalues_exact,
)
from src.bounds import apriori_bound, classify_regime

SAFETY = 1.08


def run_contractive(cfg_path):
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

    t_R = int(T * 0.5)
    L_raw = apriori_lipschitz_numerical(
        baseline.states[t_R], step_fn, loss_fn, n_perturbations=30, n_steps=20)
    L_pred = L_raw * SAFETY

    R = EMARewrite(
        theta_ema=baseline.states[t_R].theta + torch.randn(n) * 0.01,
        alpha=1.0)
    delta = R.delta(baseline.states[t_R])

    torch.manual_seed(seed)
    rw_log = run_trajectory(s0.clone(), step_fn, loss_fn, T, rewrites=[(t_R, R)])
    div = compute_divergence(baseline, rw_log)
    bnd = apriori_bound(delta, L_pred, t_R, T)
    holds = bound_holds(div, bnd)

    post = [d for d in div[t_R:] if d > 1e-15 and d < 1e30]
    grew = max(post) > post[0] * 1.5 if len(post) >= 2 else False

    print(f"[contractive/Adam] L_pred={L_pred:.4f}, delta={delta:.6f}")
    if len(post) >= 2:
        print(f"  {'GREW' if grew else 'DECAYED'}: {post[0]:.6f} -> {post[-1]:.6f}")
    print(f"  bound holds: {'PASS' if holds else 'FAIL'}")

    return {
        "label": "contractive", "optimizer": "Adam",
        "regime": classify_regime(L_pred), "L_pred": L_pred,
        "delta": delta, "t_R": t_R, "T": T,
        "divergence": div, "apriori_bound": bnd,
        "bound_holds": holds, "divergence_grew": grew,
    }


def run_expansive(cfg_path):
    cfg = yaml.safe_load(open(cfg_path))
    n = cfg["model"]["n"]
    seed = cfg["training"]["seed"]
    kappa = cfg["model"]["condition_number"]

    lr = 0.021
    T_sgd = 300
    schedule = make_constant_schedule(lr=lr)

    A = make_quadratic_hessian(n, kappa, seed)
    loss_fn = make_quadratic_loss(A)
    eigs = hessian_eigenvalues_exact(A)
    step_fn = SGDStep(momentum=0.0)

    L_raw = apriori_lipschitz_sgd(eigs, lr=lr, momentum=0.0)
    L_pred = L_raw * SAFETY  # safety margin on SGD too
    print(f"\n[expansive/SGD] L_raw={L_raw:.4f}, L_pred={L_pred:.4f}, lr={lr}")

    torch.manual_seed(seed)
    s0 = TrainingState(theta=torch.randn(n)*0.1, moments=torch.zeros(n,2),
                       schedule_fn=schedule, t=0)
    baseline = run_trajectory(s0.clone(), step_fn, loss_fn, T_sgd)

    t_R = T_sgd // 2
    R = EMARewrite(
        theta_ema=baseline.states[t_R].theta + torch.randn(n) * 0.001,
        alpha=1.0)
    delta = R.delta(baseline.states[t_R])
    print(f"  delta={delta:.6f}")

    torch.manual_seed(seed)
    rw_log = run_trajectory(s0.clone(), step_fn, loss_fn, T_sgd, rewrites=[(t_R, R)])
    div = compute_divergence(baseline, rw_log)
    bnd = apriori_bound(delta, L_pred, t_R, T_sgd)
    holds = bound_holds(div, bnd)

    post = [d for d in div[t_R:] if d > 1e-15 and d < 1e30]
    if len(post) >= 2:
        grew = max(post) > post[0] * 1.5
        print(f"  {'GREW' if grew else 'DECAYED'}: {post[0]:.6f} -> {post[-1]:.6f}")
    else:
        grew = False
        print(f"  No valid divergence data")
    print(f"  bound holds: {'PASS' if holds else 'FAIL'}")

    return {
        "label": "expansive", "optimizer": "SGD",
        "regime": classify_regime(L_pred), "L_pred": L_pred,
        "delta": delta, "t_R": t_R, "T": T_sgd,
        "divergence": div, "apriori_bound": bnd,
        "bound_holds": holds, "divergence_grew": grew,
    }


def run(contractive, expansive, out):
    os.makedirs(out, exist_ok=True)
    r_safe = run_contractive(contractive)
    r_danger = run_expansive(expansive)
    results = {"contractive": r_safe, "expansive": r_danger}
    out_path = Path(out) / "separation_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, default=lambda x: x if isinstance(x, (int, float, bool, str))
                  else list(x) if hasattr(x, '__iter__') else str(x))
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--contractive", default="configs/quadratic_contractive.yaml")
    p.add_argument("--expansive", default="configs/quadratic_expansive.yaml")
    p.add_argument("--out", default="results")
    run(**vars(p.parse_args()))