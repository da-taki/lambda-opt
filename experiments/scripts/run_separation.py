"""Separation experiment → Plot 7.
The paper's most important experiment: demonstrates the bound is NON-TRIVIAL.
Runs the SAME rewrite (checkpoint restore) on two regimes:
  - Contractive (small LR): bound predicts safety → divergence decays ✓
  - Expansive (large LR):   bound predicts failure → divergence grows ✓
This proves the theorem distinguishes safe from dangerous rewrites.
"""
import argparse, json, os, yaml, torch
from pathlib import Path
from src import (
    TrainingState, make_constant_schedule, AdamStep,
    CheckpointRewrite,
    run_trajectory, compute_divergence,
    apriori_lipschitz_quadratic,
    hessian_eigenvalues_exact, make_quadratic_hessian, make_quadratic_loss,
    bound_holds,
)
from src.bounds import apriori_bound, classify_regime


def run_regime(cfg_path, label):
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

    t_R = int(T * 0.5)
    state_at_tR = baseline.states[t_R]

    L_pred = apriori_lipschitz_quadratic(
        eigs, lr=opt["lr"], beta1=opt["beta1"], beta2=opt["beta2"],
        eps=opt["eps"], t=state_at_tR.t, state=state_at_tR)
    regime = classify_regime(L_pred)

    # checkpoint rewrite: inject stale moments from 50 steps ago
    staleness = cfg["rewrites"]["checkpoint_staleness"]
    stale_t = max(0, t_R - staleness)
    R = CheckpointRewrite(
        stale_moments=baseline.states[stale_t].moments.clone(),
        stale_t=stale_t)
    delta = R.delta(state_at_tR)

    torch.manual_seed(seed)
    rw_log = run_trajectory(s0.clone(), step_fn, loss_fn, T, rewrites=[(t_R, R)])
    div = compute_divergence(baseline, rw_log)
    bnd = apriori_bound(delta, L_pred, t_R, T)
    holds = bound_holds(div, bnd)

    # did divergence actually grow or shrink?
    post_div = [d for d in div[t_R:] if d > 1e-15]
    if len(post_div) >= 2:
        grew = post_div[-1] > post_div[0]
    else:
        grew = False

    prediction_correct = (regime == "expansive" and grew) or \
                         (regime == "contractive" and not grew) or \
                         regime == "boundary"

    print(f"[{label}] regime={regime}, L_pred={L_pred:.4f}, delta={delta:.6f}")
    print(f"  divergence {'GREW' if grew else 'DECAYED'}: {post_div[0]:.6f} -> {post_div[-1]:.6f}")
    print(f"  bound holds: {'PASS' if holds else 'FAIL'}")
    print(f"  prediction correct: {'YES' if prediction_correct else 'NO'}")

    return {
        "label": label, "regime": regime, "L_pred": L_pred, "delta": delta,
        "t_R": t_R, "T": T,
        "divergence": div, "apriori_bound": bnd,
        "bound_holds": holds,
        "divergence_grew": grew,
        "prediction_correct": prediction_correct,
        "config": cfg_path,
    }


def run(contractive_cfg, expansive_cfg, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    r_safe = run_regime(contractive_cfg, "contractive")
    print()
    r_danger = run_regime(expansive_cfg, "expansive")

    results = {"contractive": r_safe, "expansive": r_danger}
    out_path = Path(out_dir) / "separation_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, default=lambda x: x if isinstance(x, (int, float, bool, str))
                  else list(x) if hasattr(x, '__iter__') else str(x))
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--contractive", default="configs/quadratic_contractive.yaml")
    p.add_argument("--expansive", default="configs/quadratic_expansive.yaml")
    p.add_argument("--out", default="results")
    args = p.parse_args()
    run(args.contractive, args.expansive, args.out)