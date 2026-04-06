"""
Stochastic bound validation → Theorem 4 empirical evidence.
Injects i.i.d. Gaussian noise into SGD gradients on the quadratic
contractive system. Runs 50 trials per noise level, checks that
E[D(t)] <= stochastic bound at every timestep.

Usage:
    cd experiments/
    python scripts/run_stochastic_validation.py --config configs/quadratic_contractive.yaml --out results/
"""

import argparse, json, os, yaml, torch
import numpy as np
from pathlib import Path
from src import (
    TrainingState, make_constant_schedule, AdamStep,
    EMARewrite, run_trajectory, compute_divergence,
    apriori_lipschitz_numerical,
    make_quadratic_hessian, make_quadratic_loss,
)
from src.bounds import classify_regime

SAFETY = 1.03
N_TRIALS = 50
SIGMA_LEVELS = [0.001, 0.005, 0.01, 0.05]


def stochastic_bound(delta, L, sigma, k_steps):
    """L^k * delta + sigma/(1-L) * (1 - L^k) for contractive L < 1."""
    out = []
    for k in range(k_steps + 1):
        rewrite_term = (L ** k) * delta
        if abs(L - 1.0) < 1e-10:
            noise_term = sigma * k
        else:
            noise_term = (sigma / (1 - L)) * (1 - L ** k)
        out.append(rewrite_term + noise_term)
    return out


def run_noisy_trial(s0, step_fn, loss_fn, T, t_R, rewrite, sigma, seed):
    """Single trial: run baseline and rewritten trajectory with injected noise."""
    n = s0.theta.shape[0]
    torch.manual_seed(seed)

    # baseline with noise
    s_bl = s0.clone()
    for step in range(t_R):
        loss, grad = loss_fn(s_bl.theta)
        noise = torch.randn(n) * sigma
        s_bl, _ = step_fn(s_bl, grad + noise)

    s_at_tR = s_bl.clone()
    s_rw = rewrite.apply(s_at_tR)

    divergence = []
    s_bl2 = s_at_tR.clone()
    for step in range(T - t_R):
        div = (s_rw.theta - s_bl2.theta).norm(2).item()
        divergence.append(div)
        loss_bl, grad_bl = loss_fn(s_bl2.theta)
        loss_rw, grad_rw = loss_fn(s_rw.theta)
        noise_bl = torch.randn(n) * sigma
        noise_rw = torch.randn(n) * sigma
        s_bl2, _ = step_fn(s_bl2, grad_bl + noise_bl)
        s_rw, _ = step_fn(s_rw, grad_rw + noise_rw)

    divergence.append((s_rw.theta - s_bl2.theta).norm(2).item())
    return divergence


def run(config, out):
    cfg = yaml.safe_load(open(config))
    os.makedirs(out, exist_ok=True)
    n, T, seed = cfg["model"]["n"], cfg["training"]["T"], cfg["training"]["seed"]
    opt = cfg["optimizer"]
    schedule = make_constant_schedule(lr=opt["lr"], beta1=opt["beta1"], beta2=opt["beta2"])

    A = make_quadratic_hessian(n, cfg["model"]["condition_number"], seed)
    loss_fn = make_quadratic_loss(A)
    step_fn = AdamStep()

    torch.manual_seed(seed)
    s0 = TrainingState(theta=torch.randn(n) * 0.1, moments=torch.zeros(n, 2),
                       schedule_fn=schedule, t=0)

    # get L_pred from clean baseline
    from src import run_trajectory
    baseline = run_trajectory(s0.clone(), step_fn, loss_fn, T)
    t_R = int(T * 0.5)
    L_raw = apriori_lipschitz_numerical(
        baseline.states[t_R], step_fn, loss_fn,
        n_perturbations=20, n_steps=15)
    L_pred = L_raw * SAFETY
    regime = classify_regime(L_pred)
    print(f"L_pred={L_pred:.6f} [{regime}]")

    # fixed rewrite
    R = EMARewrite(
        theta_ema=baseline.states[t_R].theta + torch.randn(n) * 0.01,
        alpha=0.999)
    delta = R.delta(baseline.states[t_R])
    print(f"delta={delta:.6f}, t_R={t_R}, T={T}")

    results = {"L_pred": L_pred, "regime": regime, "delta": delta,
               "t_R": t_R, "T": T, "sigma_levels": SIGMA_LEVELS,
               "n_trials": N_TRIALS, "sigma_results": {}}

    for sigma in SIGMA_LEVELS:
        print(f"\n--- sigma={sigma} ({N_TRIALS} trials) ---")
        all_divs = []
        for trial in range(N_TRIALS):
            div = run_noisy_trial(s0, step_fn, loss_fn, T, t_R, R, sigma, seed=trial*100)
            all_divs.append(div)
            if trial % 10 == 0:
                print(f"  trial {trial}/{N_TRIALS}", end="\r")

        all_divs = np.array(all_divs)  # (N_TRIALS, T - t_R + 1)
        mean_div = all_divs.mean(axis=0).tolist()
        std_div = all_divs.std(axis=0).tolist()

        bnd = stochastic_bound(delta, L_pred, sigma, T - t_R)

        # check bound holds for mean divergence at every timestep
        holds = all(m <= b + 1e-6 for m, b in zip(mean_div, bnd))

        # empirical noise floor: mean of last 20 steps
        empirical_floor = float(np.mean(all_divs[:, -20:]))
        theoretical_floor = sigma / (1 - L_pred) if L_pred < 1 else float("inf")
        floor_rel_error = abs(empirical_floor - theoretical_floor) / max(theoretical_floor, 1e-10)

        print(f"  bound holds (mean): {'PASS' if holds else 'FAIL'}")
        print(f"  empirical floor={empirical_floor:.6f}, theoretical={theoretical_floor:.6f}, "
              f"rel_err={floor_rel_error:.4f}")

        results["sigma_results"][str(sigma)] = {
            "sigma": sigma,
            "mean_divergence": mean_div,
            "std_divergence": std_div,
            "stochastic_bound": bnd,
            "bound_holds": holds,
            "empirical_noise_floor": empirical_floor,
            "theoretical_noise_floor": theoretical_floor,
            "floor_relative_error": floor_rel_error,
        }

    # summary
    print("\n" + "=" * 50)
    print("STOCHASTIC BOUND VALIDATION SUMMARY")
    print("=" * 50)
    all_pass = True
    for sigma, res in results["sigma_results"].items():
        r = res
        mark = "PASS" if r["bound_holds"] else "FAIL"
        if not r["bound_holds"]:
            all_pass = False
        print(f"  sigma={sigma}: {mark}  floor_rel_err={r['floor_relative_error']:.4f}")
    print(f"\nAll bounds hold: {all_pass}")

    out_path = Path(out) / "stochastic_validation_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/quadratic_contractive.yaml")
    p.add_argument("--out", default="results")
    run(**vars(p.parse_args()))