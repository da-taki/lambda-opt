"""A priori Lipschitz constant estimation.

Two methods:
1. Numerical: Run Step_O from perturbed states at rewrite point, measure contraction.
   A priori (before rewritten trajectory), captures Adam's nonlinear v̂ dynamics.
   PRIMARY method for Adam.
2. Spectral: max_i |1 - η·λ_i| — exact for SGD, approximate baseline for Adam.
"""
import math
import torch
from typing import Optional
from .state import TrainingState
from .step import StepFn


def apriori_lipschitz_numerical(
    state: TrainingState, step_fn: StepFn, loss_fn,
    n_perturbations: int = 20, n_steps: int = 10, eps: float = 1e-4,
) -> float:
    """Numerical a priori L at a specific state.

    Runs n_steps from state and perturbed copies, measures per-step
    contraction rate. A priori: uses only local info at t_R.
    """
    ratios = []
    for _ in range(n_perturbations):
        perturb = torch.randn_like(state.theta) * eps
        s_base = state.clone()
        s_pert = state.clone()
        s_pert.theta = s_pert.theta + perturb
        d_prev = perturb.norm(2).item()

        for step in range(n_steps):
            _, g_base = loss_fn(s_base.theta)
            _, g_pert = loss_fn(s_pert.theta)
            s_base, _ = step_fn(s_base, g_base)
            s_pert, _ = step_fn(s_pert, g_pert)
            d_after = (s_pert.theta - s_base.theta).norm(2).item()
            if d_prev > 1e-15 and d_after > 1e-30:
                ratios.append(d_after / d_prev)
            d_prev = d_after

    if not ratios:
        return 1.0
    log_r = [math.log(max(r, 1e-30)) for r in ratios]
    return math.exp(sum(log_r) / len(log_r))


def apriori_lipschitz_sgd(
    eigenvalues: torch.Tensor, lr: float, momentum: float = 0.0,
) -> float:
    """Exact L for SGD: max_i |1 - η·λ_i|."""
    if momentum == 0:
        return (1 - lr * eigenvalues).abs().max().item()
    L_max = 0.0
    for lam in eigenvalues:
        tr = momentum + 1 - lr * lam.item()
        det = momentum
        disc = tr * tr - 4 * det
        r = 0.5 * (abs(tr) + math.sqrt(disc)) if disc >= 0 else math.sqrt(det)
        L_max = max(L_max, r)
    return L_max


def apriori_lipschitz_quadratic(
    eigenvalues: torch.Tensor, lr: float,
    beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8,
    t: int = 100, state: Optional[TrainingState] = None,
) -> float:
    """Spectral L_pred for Adam (approximate). Kept for SGD/comparison."""
    eigenvalues = eigenvalues.float()
    if state is not None and state.moments.shape[1] >= 2:
        v = state.moments[:, 1]
        bc = 1 - beta2 ** max(t, 1)
        v_hat = (v / bc).clamp(min=1e-16)
        if len(eigenvalues) == len(v_hat):
            eig_s = eigenvalues.sort().values
            v_s = v_hat.sort().values
            rho = (1 - lr * eig_s / (v_s.sqrt() + eps)).abs()
        else:
            return max((1 - lr * l / (v_hat.sqrt().median() + eps)).abs().item() for l in eigenvalues)
    else:
        rho = (1 - lr * eigenvalues / (eigenvalues.abs().sqrt() * 0.01 + eps)).abs()
    return rho.max().item()


def apriori_lipschitz_adam(eigenvalues, state):
    hp = state.schedule_fn(state.t)
    return apriori_lipschitz_quadratic(
        eigenvalues, lr=hp["lr"], beta1=hp["beta1"],
        beta2=hp["beta2"], eps=hp["eps"], t=state.t, state=state)


def apriori_lipschitz_from_hessian(hessian_eigenvalues, state, optimizer_type="adam", momentum=0.9):
    hp = state.schedule_fn(state.t)
    if optimizer_type == "adam":
        return apriori_lipschitz_quadratic(
            hessian_eigenvalues, lr=hp["lr"], beta1=hp["beta1"],
            beta2=hp["beta2"], eps=hp["eps"], t=state.t, state=state)
    elif optimizer_type == "sgd":
        return apriori_lipschitz_sgd(hessian_eigenvalues, lr=hp["lr"], momentum=momentum)
    raise ValueError(f"Unknown optimizer: {optimizer_type}")