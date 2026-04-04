"""Local Lipschitz estimation via finite differences.

This gives worst-case local sensitivity L_local — typically L >> 1.
Used in Plot 4 as a comparison baseline against L_pred.
"""
import torch
from typing import List
from .state import TrainingState
from .step import StepFn


def estimate_lipschitz_at_step(
    state: TrainingState, step_fn: StepFn, loss_fn,
    n_samples: int = 10, eps: float = 1e-4,
) -> float:
    """L ≈ max_i ‖Step(S+εᵢ) − Step(S)‖ / ‖εᵢ‖"""
    _, grad = loss_fn(state.theta)
    s_new, _ = step_fn(state, grad)
    L_max = 0.0
    for _ in range(n_samples):
        s_pert = TrainingState(
            theta=state.theta + torch.randn_like(state.theta) * eps,
            moments=state.moments + torch.randn_like(state.moments) * eps,
            schedule_fn=state.schedule_fn, t=state.t)
        _, grad_p = loss_fn(s_pert.theta)
        s_new_p, _ = step_fn(s_pert, grad_p)
        num = TrainingState.distance(s_new, s_new_p)
        den = TrainingState.distance(state, s_pert)
        L_max = max(L_max, num / max(den, 1e-12))
    return L_max


def estimate_lipschitz_trajectory(
    trajectory_log, step_fn: StepFn, loss_fn,
    sample_every: int = 10, **kwargs,
) -> List[float]:
    estimates, L = [], 1.0
    for i, s in enumerate(trajectory_log.states[:-1]):
        if i % sample_every == 0:
            L = estimate_lipschitz_at_step(s, step_fn, loss_fn, **kwargs)
        estimates.append(L)
    return estimates