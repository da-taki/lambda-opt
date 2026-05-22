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


class AdamWStep(StepFn):
    def __init__(self, weight_decay: float = 0.01):
        self.weight_decay = weight_decay

    def __call__(self, state: TrainingState, grad: torch.Tensor) -> Tuple[TrainingState, float]:
        hp = state.schedule_fn(state.t)
        lr, b1, b2, eps = hp["lr"], hp["beta1"], hp["beta2"], hp["eps"]
        m, v = state.moments[:, 0], state.moments[:, 1]
        m_new = b1 * m + (1 - b1) * grad
        v_new = b2 * v + (1 - b2) * grad.pow(2)
        t_bc = state.t + 1
        m_hat = m_new / (1 - b1 ** t_bc)
        v_hat = v_new / (1 - b2 ** t_bc)
        theta_new = (
            state.theta * (1 - lr * self.weight_decay)
            - lr * m_hat / (v_hat.sqrt() + eps)
        )
        new_state = TrainingState(
            theta=theta_new,
            moments=torch.stack([m_new, v_new], dim=1),
            schedule_fn=state.schedule_fn, t=t_bc)
        emp_L = TrainingState.distance(state, new_state) / max(grad.norm(2).item(), 1e-12)
        return new_state, emp_L


class RMSPropStep(StepFn):
    def __init__(self, alpha: float = 0.99, momentum: float = 0.0):
        self.alpha = alpha
        self.momentum = momentum

    def __call__(self, state: TrainingState, grad: torch.Tensor) -> Tuple[TrainingState, float]:
        hp = state.schedule_fn(state.t)
        lr, eps = hp["lr"], hp["eps"]
        square_avg = state.moments[:, 1]
        square_avg_new = self.alpha * square_avg + (1 - self.alpha) * grad.pow(2)
        if self.momentum > 0:
            buf = state.moments[:, 0]
            buf_new = self.momentum * buf + grad / (square_avg_new.sqrt() + eps)
            theta_new = state.theta - lr * buf_new
            m0 = buf_new
        else:
            theta_new = state.theta - lr * grad / (square_avg_new.sqrt() + eps)
            m0 = state.moments[:, 0]
        new_state = TrainingState(
            theta=theta_new,
            moments=torch.stack([m0, square_avg_new], dim=1),
            schedule_fn=state.schedule_fn, t=state.t + 1)
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
