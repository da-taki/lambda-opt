"""2-rewrite composition → Plot 3.
Compositional bound: Σ δ_j · L_pred^(t − t_Rj)
"""
import argparse, json, os, yaml, torch
from pathlib import Path
from src import (
    TrainingState, make_constant_schedule, AdamStep,
    EMARewrite, LoRAFreezeRewrite,
    run_trajectory, compute_divergence,
    apriori_lipschitz_quadratic,
    hessian_eigenvalues_exact, make_quadratic_hessian, make_quadratic_loss,
    bound_holds,
)
from src.bounds import compositional_bound, classify_regime


def run(cfg_path, out_dir):
    cfg = yaml.safe_load(open(cfg_path))
    os.makedirs(out_dir, exist_ok=True)
    n, T, seed = cfg["model"]["n"], cfg["training"]["T"], cfg["training"]["seed"]
    opt = cfg["optimizer"]
    schedule = make_constant_schedule(lr=opt["lr"], beta1=opt["beta1"], beta2=opt["beta2"])

    A = make_quadratic_hessian(n, cfg["model"]["condition_number"], seed)
    loss_fn = make_quadratic_loss(A)
    step_fn = AdamStep()
    eigs = hessian_eigenvalues_exact(A)

    s0 = TrainingState(theta=(torch.manual_seed(seed), torch.randn(n) * 0.1)[1],
                       moments=torch.zeros(n, 2), schedule_fn=schedule, t=0)
    torch.manual_seed(seed)
    s0 = TrainingState(theta=torch.randn(n)*0.1, moments=torch.zeros(n,2),
                       schedule_fn=schedule, t=0)
    baseline = run_trajectory(s0.clone(), step_fn, loss_fn, T)

    t1, t2 = int(T * 0.25), int(T * 0.5)

    # a priori L at t1 (use t1 state)
    L_pred = apriori_lipschitz_quadratic(
        eigs, lr=opt["lr"], beta1=opt["beta1"], beta2=opt["beta2"],
        eps=opt["eps"], t=baseline.states[t1].t, state=baseline.states[t1])

    R1 = EMARewrite(
        theta_ema=baseline.states[t1].theta + torch.randn(n) * 0.01,
        alpha=cfg["rewrites"]["ema_alpha"])
    delta1 = R1.delta(baseline.states[t1])

    k = int(n * cfg["rewrites"]["lora_freeze_frac"])
    R2 = LoRAFreezeRewrite(frozen_indices=torch.arange(k))
    delta2 = R2.delta(baseline.states[t2])

    torch.manual_seed(seed)
    rw_log = run_trajectory(s0.clone(), step_fn, loss_fn, T,
                            rewrites=[(t1, R1), (t2, R2)])
    div = compute_divergence(baseline, rw_log)

    comp_bnd = compositional_bound([delta1, delta2], L_pred, [t1, t2], T)
    holds = bound_holds(div, comp_bnd)
    regime = classify_regime(L_pred)

    print(f"Compositional bound: {'PASS' if holds else 'FAIL'} [{regime}]")
    print(f"  L_pred={L_pred:.6f}, delta1={delta1:.6f}@{t1}, delta2={delta2:.6f}@{t2}")
    print(f"  max_div={max(div):.6f}, max_bound={max(comp_bnd):.6f}")

    results = {
        "L_pred": L_pred, "regime": regime, "t1": t1, "t2": t2,
        "delta1": delta1, "delta2": delta2,
        "divergence": div, "compositional_bound": comp_bnd,
        "bound_holds": holds,
    }
    out_path = Path(out_dir) / "composition_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, default=lambda x: list(x) if hasattr(x, '__iter__')
                  and not isinstance(x, str) else x)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/quadratic_contractive.yaml")
    p.add_argument("--out", default="results")
    run(**vars(p.parse_args()))