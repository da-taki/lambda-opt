"""A priori Lipschitz constant from loss landscape + optimizer hyperparameters.

Core insight: for Adam on loss L(θ) with Hessian H at a point θ*, the
per-step contraction rate of the θ-component of the update map is governed
by the spectral radius of the Jacobian of Step_O w.r.t. θ.

For Adam at quasi-steady-state (bias-corrected moments stabilized):
    θ_{t+1} = θ_t - η · m̂_t / (√v̂_t + ε)

Near a fixed point where ∇L ≈ Hθ, the linearized update's Jacobian has
per-coordinate contraction factor:

    ρ_i = |1 - η · λ_i / (√(v̂_i) + ε)|

where λ_i is the i-th Hessian eigenvalue and v̂_i is the bias-corrected
second moment estimate for coordinate i.

The overall Lipschitz constant is:
    L_pred = max_i ρ_i

This is computed BEFORE running any rewritten trajectory, making the
bound in Theorem 1 a genuine a priori prediction.

For SGD with momentum μ and learning rate η:
    L_pred = max_i |1 - η·λ_i|  (simplified, ignoring momentum coupling)
    With momentum: spectral radius of 2×2 block [[μ, 1], [-η·λ_i·μ, 1-η·λ_i]]

For neural networks: use top-k Hessian eigenvalues from Lanczos as λ_i,
and per-parameter v̂_i from the current Adam state.
"""
import math
import torch
from typing import Optional
from .state import TrainingState


def apriori_lipschitz_quadratic(
    eigenvalues: torch.Tensor,
    lr: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    t: int = 100,
    state: Optional[TrainingState] = None,
) -> float:
    """A priori L_pred for Adam on a quadratic loss.

    If state is provided, uses actual v̂ from the optimizer state.
    Otherwise, estimates v̂ from the eigenvalues at quasi-steady-state:
        v̂_i ≈ λ_i² · ‖θ‖² / n  (approximate for quadratic at steady state)

    Returns L_pred = max_i |1 - η·λ_i / (√v̂_i + ε)|
    """
    eigenvalues = eigenvalues.float()

    if state is not None and state.moments.shape[1] >= 2:
        # use actual second moment from optimizer state
        v = state.moments[:, 1]
        bc = 1 - beta2 ** max(t, 1)
        v_hat = v / bc
        v_hat = v_hat.clamp(min=1e-16)

        # map eigenvalues to parameters: if dimensions match, direct;
        # otherwise use eigenvalue statistics
        if len(eigenvalues) == len(v_hat):
            # sort both to align (eigenvalue i ↔ coordinate i after rotation)
            eig_sorted = eigenvalues.sort().values
            v_sorted = v_hat.sort().values
            rho = (1 - lr * eig_sorted / (v_sorted.sqrt() + eps)).abs()
        else:
            # use eigenvalue distribution against v_hat distribution
            rho_list = []
            for lam in eigenvalues:
                r = (1 - lr * lam / (v_hat.sqrt().median() + eps)).abs()
                rho_list.append(r.item())
            return max(rho_list)
    else:
        # quasi-steady-state estimate: v̂_i ≈ (λ_i · σ_θ)² where σ_θ = ‖θ‖/√n
        # at convergence, gradients are small, so v̂ is dominated by history
        # simplification: assume v̂_i is proportional to λ_i²
        # then ρ_i = |1 - η·λ_i / (|λ_i|·σ + ε)| ≈ |1 - η·sign(λ_i)/σ|
        # better: just use the raw eigenvalue formula with eps regularization
        rho = (1 - lr * eigenvalues / (eigenvalues.abs().sqrt() * 0.01 + eps)).abs()

    return rho.max().item()


def apriori_lipschitz_adam(
    eigenvalues: torch.Tensor,
    state: TrainingState,
) -> float:
    """Convenience: extract HPs from state and compute L_pred."""
    hp = state.schedule_fn(state.t)
    return apriori_lipschitz_quadratic(
        eigenvalues, lr=hp["lr"], beta1=hp["beta1"],
        beta2=hp["beta2"], eps=hp["eps"], t=state.t, state=state)


def apriori_lipschitz_sgd(
    eigenvalues: torch.Tensor,
    lr: float,
    momentum: float = 0.0,
) -> float:
    """A priori L for SGD.

    Without momentum: L = max_i |1 - η·λ_i|
    With momentum μ: spectral radius of [[μ, 1], [-η·λ_i·μ, 1-η·λ_i]]
    """
    if momentum == 0:
        rho = (1 - lr * eigenvalues).abs()
        return rho.max().item()

    # with momentum: ρ = max eigenvalue of 2x2 block per coordinate
    L_max = 0.0
    for lam in eigenvalues:
        a = momentum
        b = 1 - lr * lam.item()
        # characteristic equation: ρ² - (a+b)ρ + (ab + μ·η·λ) = 0
        # but the matrix is [[μ, 1], [-μηλ, 1-ηλ]]
        # trace = μ + 1 - ηλ, det = μ(1-ηλ) + μηλ = μ
        tr = momentum + 1 - lr * lam.item()
        det = momentum
        disc = tr * tr - 4 * det
        if disc >= 0:
            r = 0.5 * (abs(tr) + math.sqrt(disc))
        else:
            r = math.sqrt(det)  # complex eigenvalues: |ρ| = √det
        L_max = max(L_max, r)
    return L_max


def apriori_lipschitz_from_hessian(
    hessian_eigenvalues: torch.Tensor,
    state: TrainingState,
    optimizer_type: str = "adam",
    momentum: float = 0.9,
) -> float:
    """Dispatch to the right a priori estimator."""
    hp = state.schedule_fn(state.t)
    if optimizer_type == "adam":
        return apriori_lipschitz_quadratic(
            hessian_eigenvalues, lr=hp["lr"], beta1=hp["beta1"],
            beta2=hp["beta2"], eps=hp["eps"], t=state.t, state=state)
    elif optimizer_type == "sgd":
        return apriori_lipschitz_sgd(
            hessian_eigenvalues, lr=hp["lr"], momentum=momentum)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")