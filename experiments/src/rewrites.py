"""λOpt Rewrite Operators: R: S → S with computable δ-bounds."""
import torch
from .state import TrainingState


class Rewrite:
    name: str = "base"
    def apply(self, state: TrainingState) -> TrainingState:
        raise NotImplementedError
    def delta(self, state: TrainingState) -> float:
        raise NotImplementedError


class EMARewrite(Rewrite):
    """θ_new = α·θ_ema + (1−α)·θ → displacement = α·(θ_ema − θ)"""
    name = "EMA"
    def __init__(self, theta_ema: torch.Tensor, alpha: float = 0.999):
        self.theta_ema = theta_ema
        self.alpha = alpha

    def apply(self, state: TrainingState) -> TrainingState:
        return TrainingState(
            theta=self.alpha * self.theta_ema + (1 - self.alpha) * state.theta,
            moments=state.moments.clone(),
            schedule_fn=state.schedule_fn, t=state.t)

    def delta(self, state: TrainingState) -> float:
        return (self.alpha * (self.theta_ema - state.theta).norm(2)).item()


class CheckpointRewrite(Rewrite):
    """Replace moments with stale snapshot. δ = ‖M' − M‖"""
    name = "Checkpoint"
    def __init__(self, stale_moments: torch.Tensor, stale_t: int):
        self.stale_moments = stale_moments
        self.stale_t = stale_t

    def apply(self, state: TrainingState) -> TrainingState:
        sm = self.stale_moments.clone()
        sm[:, 1] = sm[:, 1].clamp(min=0.0)  # v (2nd moment) must be >= 0
        return TrainingState(
            theta=state.theta.clone(), moments=sm,
            schedule_fn=state.schedule_fn, t=self.stale_t)

    def delta(self, state: TrainingState) -> float:
        return (self.stale_moments - state.moments).norm(2).item()


class LoRAFreezeRewrite(Rewrite):
    """Zero moment rows for frozen params. δ = ‖zeroed moments‖"""
    name = "LoRA"
    def __init__(self, frozen_indices: torch.Tensor):
        self.frozen_indices = frozen_indices

    def apply(self, state: TrainingState) -> TrainingState:
        m = state.moments.clone()
        m[self.frozen_indices] = 0.0
        return TrainingState(
            theta=state.theta.clone(), moments=m,
            schedule_fn=state.schedule_fn, t=state.t)

    def delta(self, state: TrainingState) -> float:
        return state.moments[self.frozen_indices].norm(2).item()