"""Neural network experiments → Plot 9.
Runs on MLP/MNIST or ResNet-18/CIFAR-10.
Uses Lanczos Hessian eigenvalues for a priori L_pred.
"""
import argparse, json, os, yaml, torch
import torch.nn as nn
from pathlib import Path
from src import (
    TrainingState, make_constant_schedule, AdamStep,
    EMARewrite, CheckpointRewrite,
    run_trajectory, compute_divergence,
    apriori_lipschitz_quadratic,
    bound_holds, tightness_ratio, loss_gap,
)
from src.bounds import apriori_bound, classify_regime
from src.loss_landscape import hessian_eigenvalues_lanczos
from src.metrics import effective_lipschitz_from_divergence


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
        model = SimpleMLP(
            hidden_sizes=cfg["model"]["hidden_sizes"],
            num_classes=cfg["model"]["num_classes"])
        return model, loader
    elif model_type == "resnet18":
        from torchvision import datasets, transforms, models
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
        dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
        loader = torch.utils.data.DataLoader(dataset, batch_size=cfg["training"]["batch_size"],
                                              shuffle=False)
        model = models.resnet18(num_classes=cfg["model"]["num_classes"])
        return model, loader
    else:
        raise ValueError(f"Unknown model: {model_type}")


def flatten_params(model):
    return torch.cat([p.data.reshape(-1) for p in model.parameters()])


def unflatten_params(model, flat):
    idx = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(flat[idx:idx+n].reshape(p.shape))
        idx += n


def run(cfg_path, out_dir):
    cfg = yaml.safe_load(open(cfg_path))
    os.makedirs(out_dir, exist_ok=True)
    T, seed = cfg["training"]["T"], cfg["training"]["seed"]
    opt_cfg = cfg["optimizer"]

    torch.manual_seed(seed)
    model, loader = get_model_and_data(cfg)
    criterion = nn.CrossEntropyLoss()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {cfg['model']['type']}, params: {n_params}")

    # wrap model as our state-based system
    schedule = make_constant_schedule(
        lr=opt_cfg["lr"], beta1=opt_cfg["beta1"], beta2=opt_cfg["beta2"])

    theta0 = flatten_params(model)
    s0 = TrainingState(theta=theta0, moments=torch.zeros(n_params, 2),
                       schedule_fn=schedule, t=0)

    # create loss_fn that uses the model
    data_iter = iter(loader)
    batch_cache = []
    for i, (x, y) in enumerate(data_iter):
        batch_cache.append((x, y))
        if len(batch_cache) >= T:
            break

    def loss_fn(theta_flat):
        unflatten_params(model, theta_flat)
        step_idx = min(len(batch_cache) - 1, loss_fn._step_counter)
        loss_fn._step_counter += 1
        x, y = batch_cache[step_idx % len(batch_cache)]
        model.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        grad = torch.cat([p.grad.reshape(-1) for p in model.parameters()])
        return loss.item(), grad.detach()
    loss_fn._step_counter = 0

    step_fn = AdamStep()

    # baseline
    loss_fn._step_counter = 0
    baseline = run_trajectory(s0.clone(), step_fn, loss_fn, T)

    # Hessian via Lanczos at midpoint
    t_R = int(T * 0.5)
    state_mid = baseline.states[t_R]
    unflatten_params(model, state_mid.theta)
    mid_batch = batch_cache[t_R % len(batch_cache)]
    hessian_k = cfg["model"].get("hessian_k", 20)
    hessian_iters = cfg["model"].get("hessian_lanczos_iters", 100)
    print(f"Computing top-{hessian_k} Hessian eigenvalues via Lanczos...")
    eigs = hessian_eigenvalues_lanczos(
        model, mid_batch, criterion, k=hessian_k, num_iterations=hessian_iters)
    print(f"  eigenvalue range: [{eigs.min():.4f}, {eigs.max():.4f}]")

    L_pred = apriori_lipschitz_quadratic(
        eigs, lr=opt_cfg["lr"], beta1=opt_cfg["beta1"], beta2=opt_cfg["beta2"],
        eps=opt_cfg["eps"], t=state_mid.t, state=state_mid)
    regime = classify_regime(L_pred)
    print(f"  L_pred={L_pred:.4f} [{regime}]")

    # EMA rewrite
    R = EMARewrite(
        theta_ema=state_mid.theta + torch.randn(n_params) * 0.001,
        alpha=0.999)
    delta = R.delta(state_mid)

    loss_fn._step_counter = 0
    rw_log = run_trajectory(s0.clone(), step_fn, loss_fn, T, rewrites=[(t_R, R)])
    div = compute_divergence(baseline, rw_log)

    bnd = apriori_bound(delta, L_pred, t_R, T)
    holds = bound_holds(div, bnd)
    L_actual = effective_lipschitz_from_divergence(div, t_R)
    lgap = loss_gap(baseline.losses, rw_log.losses)

    print(f"\nEMA rewrite @ t={t_R}: delta={delta:.6f}")
    print(f"  bound holds: {'PASS' if holds else 'FAIL'}")
    print(f"  L_pred={L_pred:.4f}, L_actual={L_actual:.4f}")
    print(f"  max_div={max(div):.6f}")

    results = {
        "model": cfg["model"]["type"], "n_params": n_params,
        "L_pred": L_pred, "L_actual": L_actual, "regime": regime,
        "delta": delta, "t_R": t_R,
        "divergence": div, "apriori_bound": bnd,
        "bound_holds": holds, "loss_gap": lgap,
        "hessian_eigenvalues": eigs.tolist(),
    }
    out_path = Path(out_dir) / f"neural_net_{cfg['model']['type']}_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, default=lambda x: x if isinstance(x, (int, float, bool, str))
                  else list(x) if hasattr(x, '__iter__') else str(x))
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/mlp_mnist.yaml")
    p.add_argument("--out", default="results")
    run(**vars(p.parse_args()))