"""λOpt Step Rules: Step_O(S) → (S', L_empirical)"""
import torch
from typing import Tuple
from .state import TrainingState


class StepFn:
    def __call__(self, state: TrainingState, grad: torch.Tensor) -> Tuple[TrainingState, float]:
        raise NotImplementedError


class AdamStep(StepFn):
    def __call__(self, state: TrainingState, grad: torch.Tensor) -> Tuple[TrainingState, float]:
        hp = state.schedule_fn(state.t)
        lr, b1, b2, eps = hp["lr"], hp["beta1"], hp["beta2"], hp["eps"]
        m, v = state.moments[:, 0], state.moments[:, 1]
        m_new = b1 * m + (1 - b1) * grad
        v_new = b2 * v + (1 - b2) * grad.pow(2)
        t_bc = state.t + 1
        m_hat = m_new / (1 - b1 ** t_bc)
        v_hat = v_new / (1 - b2 ** t_bc)
        theta_new = state.theta - lr * m_hat / (v_hat.sqrt() + eps)
        new_state = TrainingState(
            theta=theta_new,
            moments=torch.stack([m_new, v_new], dim=1),
            schedule_fn=state.schedule_fn, t=t_bc)
        emp_L = TrainingState.distance(state, new_state) / max(grad.norm(2).item(), 1e-12)
        return new_state, emp_L


class SGDStep(StepFn):
    def __init__(self, momentum: float = 0.9):
        self.mu = momentum

    def __call__(self, state: TrainingState, grad: torch.Tensor) -> Tuple[TrainingState, float]:
        lr = state.schedule_fn(state.t)["lr"]
        vel = state.moments[:, 0]
        vel_new = self.mu * vel + grad
        theta_new = state.theta - lr * vel_new
        new_mom = state.moments.clone()
        new_mom[:, 0] = vel_new
        new_state = TrainingState(
            theta=theta_new, moments=new_mom,
            schedule_fn=state.schedule_fn, t=state.t + 1)
        emp_L = TrainingState.distance(state, new_state) / max(grad.norm(2).item(), 1e-12)
        return new_state, emp_L