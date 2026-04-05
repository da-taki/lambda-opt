"""Real-world checkpoint safety analysis → Plot 10.
Tests 3 practical failure modes on ResNet-18/CIFAR-10:
  1. Stale moments (load step-20 moments at step-100)
  2. Scheduler mismatch (wrong LR at resume)
  3. EMA deletion (remove EMA shadow before resume)

For each: predict regime a priori, then measure actual behavior.
"""
import argparse, json, os, yaml, torch
import torch.nn as nn
from pathlib import Path
from torchvision import datasets, transforms, models
from src import (
    TrainingState, make_constant_schedule, make_cosine_schedule, AdamStep,
    EMARewrite, CheckpointRewrite, LoRAFreezeRewrite,
    apriori_lipschitz_numerical,
    bound_holds,
)
from src.bounds import apriori_bound, classify_regime
from src.metrics import effective_lipschitz_from_divergence

SAFETY = 1.08


def flatten_params(model):
    return torch.cat([p.data.reshape(-1) for p in model.parameters()])


def unflatten_params(model, flat):
    idx = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(flat[idx:idx+n].reshape(p.shape))
        idx += n


def run(config, out):
    cfg = yaml.safe_load(open(config))
    os.makedirs(out, exist_ok=True)
    T, seed = cfg["training"]["T"], cfg["training"]["seed"]
    opt_cfg = cfg["optimizer"]
    bs = cfg["training"]["batch_size"]

    torch.manual_seed(seed)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
    dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=False)

    model = models.resnet18(num_classes=10)
    criterion = nn.CrossEntropyLoss()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: resnet18, params: {n_params}, T: {T}")

    schedule = make_constant_schedule(
        lr=opt_cfg["lr"], beta1=opt_cfg["beta1"], beta2=opt_cfg["beta2"])
    step_fn = AdamStep()

    # cache batches
    batch_cache = []
    for i, (x, y) in enumerate(loader):
        batch_cache.append((x, y))
        if len(batch_cache) >= T:
            break

    call_ctr = [0]
    def make_loss_fn():
        def loss_fn(theta):
            unflatten_params(model, theta)
            idx = call_ctr[0] % len(batch_cache)
            call_ctr[0] += 1
            x, y = batch_cache[idx]
            model.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            grad = torch.cat([p.grad.reshape(-1) for p in model.parameters()])
            return loss.item(), grad.detach()
        return loss_fn

    t_R = T // 2
    stale_t = T // 10  # step 20 for T=200

    # === Phase 1: Train baseline, save checkpoints ===
    print("Phase 1: Training baseline + saving checkpoints...")
    theta0 = flatten_params(model)
    s = TrainingState(theta=theta0.clone(), moments=torch.zeros(n_params, 2),
                      schedule_fn=schedule, t=0)

    # EMA shadow
    ema_shadow = theta0.clone()
    ema_alpha = 0.999

    checkpoints = {}  # step -> (theta, moments, ema)
    call_ctr[0] = 0
    loss_fn = make_loss_fn()
    baseline_losses = []

    for step in range(T):
        # save checkpoints at key points
        if step in [stale_t, t_R]:
            checkpoints[step] = {
                "theta": s.theta.clone(),
                "moments": s.moments.clone(),
                "ema": ema_shadow.clone(),
                "t": s.t,
            }

        loss, grad = loss_fn(s.theta)
        baseline_losses.append(loss)
        s, _ = step_fn(s, grad)

        # update EMA
        ema_shadow = ema_alpha * ema_shadow + (1 - ema_alpha) * s.theta

        if step % 50 == 0:
            print(f"  step {step}/{T} loss={loss:.4f}", end="\r")

    print(f"\n  Baseline done. Final loss={baseline_losses[-1]:.4f}")
    s_baseline_final = s.clone()

    # === Phase 2: Define 3 rewrite scenarios ===
    s_at_tR = TrainingState(
        theta=checkpoints[t_R]["theta"],
        moments=checkpoints[t_R]["moments"],
        schedule_fn=schedule,
        t=checkpoints[t_R]["t"])

    scenarios = {}

    # Scenario 1: Stale moments (load step-20 moments at step-100)
    stale_mom = checkpoints[stale_t]["moments"]
    R_stale = CheckpointRewrite(stale_moments=stale_mom, stale_t=stale_t)
    scenarios["stale_moments"] = {
        "rewrite": R_stale,
        "description": f"Load moments from step {stale_t} at step {t_R}",
    }

    # Scenario 2: Scheduler mismatch (suddenly use 10x LR)
    wrong_schedule = make_constant_schedule(
        lr=opt_cfg["lr"] * 10, beta1=opt_cfg["beta1"], beta2=opt_cfg["beta2"])
    # Model this as: state unchanged, but schedule changes
    # δ = 0 for state, but effective L changes
    scenarios["scheduler_mismatch"] = {
        "description": f"Resume with 10x LR ({opt_cfg['lr']*10})",
        "wrong_schedule": wrong_schedule,
    }

    # Scenario 3: EMA deletion (would have used EMA params, now use raw)
    ema_at_tR = checkpoints[t_R]["ema"]
    R_ema_del = EMARewrite(theta_ema=checkpoints[t_R]["theta"], alpha=1.0)
    # The "deletion" means we DON'T apply EMA — so the rewrite is identity
    # But the interesting case: what if someone accidentally loads EMA as params?
    R_ema_load = EMARewrite(theta_ema=ema_at_tR, alpha=1.0)
    scenarios["ema_as_params"] = {
        "rewrite": R_ema_load,
        "description": f"Load EMA shadow as parameters at step {t_R}",
    }

    # === Phase 3: Predict + Measure each scenario ===
    results = {"scenarios": {}, "t_R": t_R, "T": T, "n_params": n_params}

    for name, scenario in scenarios.items():
        print(f"\n--- Scenario: {name} ---")
        print(f"  {scenario['description']}")

        if name == "scheduler_mismatch":
            # special case: no state rewrite, but different schedule
            s_rw = TrainingState(
                theta=s_at_tR.theta.clone(),
                moments=s_at_tR.moments.clone(),
                schedule_fn=scenario["wrong_schedule"],
                t=s_at_tR.t)
            delta = 0.0  # state unchanged

            # L_pred with wrong schedule
            call_ctr[0] = t_R
            loss_fn_lip = make_loss_fn()
            L_raw = apriori_lipschitz_numerical(
                s_rw, step_fn, loss_fn_lip,
                n_perturbations=5, n_steps=5)
            L_pred = L_raw * SAFETY
        else:
            R = scenario["rewrite"]
            s_rw = R.apply(s_at_tR)
            delta = R.delta(s_at_tR)

            call_ctr[0] = t_R
            loss_fn_lip = make_loss_fn()
            L_raw = apriori_lipschitz_numerical(
                s_at_tR, step_fn, loss_fn_lip,
                n_perturbations=5, n_steps=5)
            L_pred = L_raw * SAFETY

        regime = classify_regime(L_pred)
        print(f"  PREDICTION: L_pred={L_pred:.4f} [{regime}], delta={delta:.4f}")

        if regime == "contractive":
            prediction = "stable recovery"
        elif regime == "boundary":
            prediction = "mild drift"
        else:
            prediction = "divergence growth"
        print(f"  PREDICTED OUTCOME: {prediction}")

        # Run rewritten trajectory
        s_run = s_rw.clone()
        s_base = s_at_tR.clone()
        call_ctr_rw = [t_R]
        call_ctr_bl = [t_R]

        def make_lf(ctr):
            def lf(theta):
                unflatten_params(model, theta)
                idx = ctr[0] % len(batch_cache)
                ctr[0] += 1
                x, y = batch_cache[idx]
                model.zero_grad()
                loss = criterion(model(x), y)
                loss.backward()
                grad = torch.cat([p.grad.reshape(-1) for p in model.parameters()])
                return loss.item(), grad.detach()
            return lf

        lf_rw = make_lf(call_ctr_rw)
        lf_bl = make_lf(call_ctr_bl)

        divergence = []
        rw_losses = []
        bl_losses = []
        remaining = T - t_R

        for step in range(remaining):
            div = (s_run.theta - s_base.theta).norm(2).item()
            divergence.append(div)

            loss_rw, grad_rw = lf_rw(s_run.theta)
            loss_bl, grad_bl = lf_bl(s_base.theta)
            rw_losses.append(loss_rw)
            bl_losses.append(loss_bl)

            s_run, _ = step_fn(s_run, grad_rw)
            s_base, _ = step_fn(s_base, grad_bl)

        divergence.append((s_run.theta - s_base.theta).norm(2).item())

        # analyze
        bnd = apriori_bound(delta, L_pred, 0, remaining)
        holds = bound_holds(divergence, bnd)
        L_actual = effective_lipschitz_from_divergence(
            [0.0]*t_R + divergence, t_R) if max(divergence) > 1e-10 else 1.0
        loss_gaps = [abs(a - b) for a, b in zip(rw_losses, bl_losses)]

        # recovery: steps until divergence drops below 50% of peak
        peak_div = max(divergence)
        recovery_step = None
        if peak_div > 1e-10:
            for i, d in enumerate(divergence):
                if i > 0 and d < peak_div * 0.5:
                    recovery_step = i
                    break

        # actual outcome
        if len(divergence) >= 2:
            final_vs_initial = divergence[-1] / max(divergence[0], 1e-15)
            if final_vs_initial < 0.5:
                actual_outcome = "stable recovery"
            elif final_vs_initial < 2.0:
                actual_outcome = "mild drift"
            else:
                actual_outcome = "divergence growth"
        else:
            actual_outcome = "unknown"

        correct = prediction == actual_outcome

        print(f"  ACTUAL OUTCOME: {actual_outcome}")
        print(f"  Prediction {'CORRECT' if correct else 'WRONG'}")
        print(f"  div: {divergence[0]:.4f} -> {divergence[-1]:.4f}")
        print(f"  bound holds: {'PASS' if holds else 'FAIL'}")
        print(f"  L_actual={L_actual:.4f}, recovery_step={recovery_step}")

        results["scenarios"][name] = {
            "description": scenario["description"],
            "delta": delta, "L_pred": L_pred, "L_actual": L_actual,
            "regime": regime, "prediction": prediction,
            "actual_outcome": actual_outcome, "correct": correct,
            "divergence": divergence, "bound": bnd,
            "bound_holds": holds,
            "loss_gap": loss_gaps,
            "recovery_step": recovery_step,
            "peak_divergence": peak_div,
        }

    # summary
    print("\n" + "="*60)
    print("CHECKPOINT SAFETY ANALYSIS SUMMARY")
    print("="*60)
    n_correct = sum(1 for s in results["scenarios"].values() if s["correct"])
    n_total = len(results["scenarios"])
    for name, s in results["scenarios"].items():
        mark = "OK" if s["correct"] else "XX"
        print(f"  [{mark}] {name}: predicted={s['prediction']}, actual={s['actual_outcome']}")
    print(f"\nAccuracy: {n_correct}/{n_total}")

    out_path = Path(out) / "checkpoint_safety_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, default=lambda x: x if isinstance(x, (int, float, bool, str))
                  else list(x) if hasattr(x, '__iter__') else str(x))
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/resnet18_cifar10.yaml")
    p.add_argument("--out", default="results")
    run(**vars(p.parse_args()))