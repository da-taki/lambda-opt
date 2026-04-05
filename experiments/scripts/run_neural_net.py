"""Neural network experiments → Plot 9. Memory-efficient for large models."""
import argparse, json, os, yaml, torch
import torch.nn as nn
from pathlib import Path
from src import (
    TrainingState, make_constant_schedule, AdamStep,
    EMARewrite,
    apriori_lipschitz_numerical,
    bound_holds, loss_gap,
)
from src.bounds import apriori_bound, classify_regime
from src.metrics import effective_lipschitz_from_divergence

SAFETY = 1.03


class SimpleMLP(nn.Module):
    def __init__(self, input_dim=784, hidden_sizes=[64, 32], num_classes=10):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_sizes:
            layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.view(x.size(0), -1))


def get_model_and_data(cfg):
    model_type = cfg["model"]["type"]
    if model_type == "mlp":
        from torchvision import datasets, transforms
        transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.1307,), (0.3081,))])
        dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
        loader = torch.utils.data.DataLoader(dataset, batch_size=cfg["training"]["batch_size"],
                                              shuffle=False)
        return SimpleMLP(hidden_sizes=cfg["model"]["hidden_sizes"],
                         num_classes=cfg["model"]["num_classes"]), loader
    elif model_type == "resnet18":
        from torchvision import datasets, transforms, models
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
        dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
        loader = torch.utils.data.DataLoader(dataset, batch_size=cfg["training"]["batch_size"],
                                              shuffle=False)
        return models.resnet18(num_classes=cfg["model"]["num_classes"]), loader
    raise ValueError(f"Unknown model: {model_type}")


def flatten_params(model):
    return torch.cat([p.data.reshape(-1) for p in model.parameters()])


def unflatten_params(model, flat):
    idx = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(flat[idx:idx+n].reshape(p.shape))
        idx += n


def run_lightweight(s0, step_fn, loss_fn, T, rewrite_at=None, rewrite_op=None):
    """Memory-efficient trajectory: only stores theta divergence per step,
    not the full state history. Returns (final_state, losses, theta_at_tR)."""
    s = s0.clone()
    losses = []
    theta_at_tR = None
    for step in range(T):
        if rewrite_at is not None and step == rewrite_at:
            theta_at_tR = s.theta.clone()
            if rewrite_op is not None:
                s = rewrite_op.apply(s)
        loss, grad = loss_fn(s.theta)
        losses.append(loss)
        s, _ = step_fn(s, grad)
        if step % 50 == 0:
            print(f"  step {step}/{T}", end="\r")
    print(f"  step {T}/{T} done")
    return s, losses, theta_at_tR


def run(config, out):
    cfg = yaml.safe_load(open(config))
    os.makedirs(out, exist_ok=True)
    T, seed = cfg["training"]["T"], cfg["training"]["seed"]
    opt_cfg = cfg["optimizer"]

    torch.manual_seed(seed)
    model, loader = get_model_and_data(cfg)
    criterion = nn.CrossEntropyLoss()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {cfg['model']['type']}, params: {n_params}, T: {T}")

    schedule = make_constant_schedule(
        lr=opt_cfg["lr"], beta1=opt_cfg["beta1"], beta2=opt_cfg["beta2"])
    theta0 = flatten_params(model)
    s0 = TrainingState(theta=theta0, moments=torch.zeros(n_params, 2),
                       schedule_fn=schedule, t=0)

    batch_cache = []
    for i, (x, y) in enumerate(loader):
        batch_cache.append((x, y))
        if len(batch_cache) >= T:
            break

    call_counter = [0]
    def make_loss_fn():
        def loss_fn(theta_flat):
            unflatten_params(model, theta_flat)
            idx = call_counter[0] % len(batch_cache)
            call_counter[0] += 1
            x, y = batch_cache[idx]
            model.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            grad = torch.cat([p.grad.reshape(-1) for p in model.parameters()])
            return loss.item(), grad.detach()
        return loss_fn

    step_fn = AdamStep()
    t_R = int(T * 0.5)

    # baseline run — lightweight, no state storage
    print("Running baseline...")
    call_counter[0] = 0
    loss_fn_base = make_loss_fn()
    s_base_final, base_losses, _ = run_lightweight(
        s0.clone(), step_fn, loss_fn_base, T, rewrite_at=t_R)

    # get state at t_R by running up to t_R
    print("Getting state at rewrite point...")
    call_counter[0] = 0
    loss_fn_mid = make_loss_fn()
    s_mid = s0.clone()
    for step in range(t_R):
        loss, grad = loss_fn_mid(s_mid.theta)
        s_mid, _ = step_fn(s_mid, grad)

    # numerical a priori L at t_R
    print(f"Computing numerical a priori L at t={t_R}...")
    call_counter[0] = t_R
    loss_fn_lip = make_loss_fn()
    L_raw = apriori_lipschitz_numerical(
        s_mid, step_fn, loss_fn_lip,
        n_perturbations=5, n_steps=5)
    L_pred = L_raw * SAFETY
    regime = classify_regime(L_pred)
    print(f"  L_pred={L_pred:.4f} [{regime}]")

    # EMA rewrite
    R = EMARewrite(
        theta_ema=s_mid.theta + torch.randn(n_params) * 0.001,
        alpha=0.999)
    delta = R.delta(s_mid)

    # rewritten run — compute divergence step by step
    print("Running rewritten trajectory...")
    call_counter[0] = 0
    loss_fn_rw = make_loss_fn()
    s_rw = s0.clone()
    s_bl = s0.clone()
    call_counter_bl = [0]
    def make_loss_fn_bl():
        def loss_fn(theta_flat):
            unflatten_params(model, theta_flat)
            idx = call_counter_bl[0] % len(batch_cache)
            call_counter_bl[0] += 1
            x, y = batch_cache[idx]
            model.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            grad = torch.cat([p.grad.reshape(-1) for p in model.parameters()])
            return loss.item(), grad.detach()
        return loss_fn
    loss_fn_bl2 = make_loss_fn_bl()

    divergence = []
    rw_losses = []
    for step in range(T):
        if step == t_R:
            s_rw = R.apply(s_rw)

        # compute divergence (theta only, no storage)
        div = (s_rw.theta - s_bl.theta).norm(2).item()
        divergence.append(div)

        loss_rw, grad_rw = loss_fn_rw(s_rw.theta)
        loss_bl, grad_bl = loss_fn_bl2(s_bl.theta)
        rw_losses.append(loss_rw)

        s_rw, _ = step_fn(s_rw, grad_rw)
        s_bl, _ = step_fn(s_bl, grad_bl)

        if step % 50 == 0:
            print(f"  step {step}/{T} div={div:.6f}", end="\r")

    # final divergence
    divergence.append((s_rw.theta - s_bl.theta).norm(2).item())
    print(f"\n  step {T}/{T} done, final_div={divergence[-1]:.6f}")

    bnd = apriori_bound(delta, L_pred, t_R, T)
    holds = bound_holds(divergence, bnd)
    L_actual = effective_lipschitz_from_divergence(divergence, t_R)
    lgap = loss_gap(base_losses, rw_losses)

    print(f"\nEMA rewrite @ t={t_R}: delta={delta:.6f}")
    print(f"  bound holds: {'PASS' if holds else 'FAIL'}")
    print(f"  L_pred={L_pred:.4f}, L_actual={L_actual:.4f}")
    print(f"  max_div={max(divergence):.6f}")

    results = {
        "model": cfg["model"]["type"], "n_params": n_params,
        "L_pred": L_pred, "L_actual": L_actual, "regime": regime,
        "delta": delta, "t_R": t_R,
        "divergence": divergence, "apriori_bound": bnd,
        "bound_holds": holds, "loss_gap": lgap,
    }
    out_path = Path(out) / f"neural_net_{cfg['model']['type']}_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, default=lambda x: x if isinstance(x, (int, float, bool, str))
                  else list(x) if hasattr(x, '__iter__') else str(x))
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/mlp_mnist.yaml")
    p.add_argument("--out", default="results")
    run(**vars(p.parse_args()))