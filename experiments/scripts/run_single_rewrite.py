"""Single-rewrite experiments → Plots 1, 2.
Uses a priori L_pred from Hessian spectrum — the bound is PREDICTIVE.
"""
import argparse, json, os, yaml, torch
from pathlib import Path
from src import (
    TrainingState, make_constant_schedule, AdamStep,
    EMARewrite, CheckpointRewrite, LoRAFreezeRewrite,
    run_trajectory, compute_divergence,
    estimate_lipschitz_trajectory, median_lipschitz,
    apriori_bound, apriori_lipschitz_quadratic,
    hessian_eigenvalues_exact, make_quadratic_hessian, make_quadratic_loss,
    tightness_ratio, bound_holds, loss_gap,
)
from src.bounds import classify_regime


def make_state(n, schedule, seed):
    torch.manual_seed(seed)
    return TrainingState(
        theta=torch.randn(n) * 0.1, moments=torch.zeros(n, 2),
        schedule_fn=schedule, t=0)


def build_rewrite(kind, state_at_tR, cfg, n):
    if kind == "EMA":
        return EMARewrite(
            theta_ema=state_at_tR.theta + torch.randn(n) * 0.01,
            alpha=cfg["rewrites"]["ema_alpha"])
    elif kind == "Checkpoint":
        return CheckpointRewrite(
            stale_moments=state_at_tR.moments + torch.randn_like(state_at_tR.moments) * 0.01,
            stale_t=max(0, state_at_tR.t - cfg["rewrites"]["checkpoint_staleness"]))
    elif kind == "LoRA":
        k = int(n * cfg["rewrites"]["lora_freeze_frac"])
        return LoRAFreezeRewrite(frozen_indices=torch.arange(k))


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

    # baseline
    s0 = make_state(n, schedule, seed)
    baseline = run_trajectory(s0, step_fn, loss_fn, T)

    # local L for comparison
    L_local_est = estimate_lipschitz_trajectory(
        baseline, step_fn, loss_fn,
        sample_every=cfg["lipschitz"]["sample_every"],
        n_samples=cfg["lipschitz"]["n_samples"],
        eps=cfg["lipschitz"]["eps"])
    L_local = median_lipschitz(L_local_est)

    results = {"L_local": L_local, "L_local_estimates": L_local_est, "runs": []}

    for rw_kind in ["EMA", "Checkpoint", "LoRA"]:
        for frac in cfg["rewrites"]["rewrite_times"]:
            t_R = int(T * frac)
            state_at_tR = baseline.states[t_R]

            # a priori L from Hessian + optimizer state at rewrite time
            L_pred = apriori_lipschitz_quadratic(
                eigs, lr=opt["lr"], beta1=opt["beta1"], beta2=opt["beta2"],
                eps=opt["eps"], t=state_at_tR.t, state=state_at_tR)
            regime = classify_regime(L_pred)

            R = build_rewrite(rw_kind, state_at_tR, cfg, n)
            delta = R.delta(state_at_tR)

            s0_rw = make_state(n, schedule, seed)
            rw_log = run_trajectory(s0_rw, step_fn, loss_fn, T, rewrites=[(t_R, R)])

            div = compute_divergence(baseline, rw_log)
            bnd = apriori_bound(delta, L_pred, t_R, T)
            tight = tightness_ratio(div, bnd)
            holds = bound_holds(div, bnd)
            lgap = loss_gap(baseline.losses, rw_log.losses)

            run_data = {
                "rewrite": rw_kind, "t_R": t_R, "frac": frac,
                "delta": delta, "L_pred": L_pred, "regime": regime,
                "divergence": div, "apriori_bound": bnd,
                "tightness": tight, "bound_holds": holds,
                "loss_gap": lgap, "max_divergence": max(div),
            }
            results["runs"].append(run_data)
            print(f"{'PASS' if holds else 'FAIL'} {rw_kind} @ {frac:.0%}: "
                  f"L_pred={L_pred:.4f} [{regime}] delta={delta:.6f} max_div={max(div):.6f}")

    out_path = Path(out_dir) / "single_rewrite_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, default=lambda x: x if isinstance(x, (int, float, bool, str))
                  else list(x) if hasattr(x, '__iter__') else str(x))
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/quadratic_contractive.yaml")
    p.add_argument("--out", default="results")
    run(**vars(p.parse_args()))