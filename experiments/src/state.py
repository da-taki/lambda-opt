"""λOpt Training State: S = (θ, M, h, t) ∈ ℝⁿ × ℝⁿˣᵏ × (ℕ→ℝᵈ) × ℕ"""
import torch
from dataclasses import dataclass
from typing import Callable


@dataclass
class TrainingState:
    theta: torch.Tensor
    moments: torch.Tensor
    schedule_fn: Callable[[int], dict]
    t: int = 0

    def clone(self) -> "TrainingState":
        return TrainingState(
            theta=self.theta.clone(), moments=self.moments.clone(),
            schedule_fn=self.schedule_fn, t=self.t)

    def norm(self) -> float:
        return torch.sqrt(self.theta.pow(2).sum() + self.moments.pow(2).sum()).item()

    @staticmethod
    def distance(a: "TrainingState", b: "TrainingState") -> float:
        return torch.sqrt(
            (a.theta - b.theta).pow(2).sum() +
            (a.moments - b.moments).pow(2).sum()).item()

    @staticmethod
    def theta_distance(a: "TrainingState", b: "TrainingState") -> float:
        return (a.theta - b.theta).norm(2).item()


def make_constant_schedule(lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8) -> Callable:
    hp = {"lr": lr, "beta1": beta1, "beta2": beta2, "eps": eps}
    return lambda t: hp


def make_cosine_schedule(lr_max=1e-3, lr_min=1e-5, T=10000,
                         beta1=0.9, beta2=0.999, eps=1e-8) -> Callable:
    import math
    def _s(t):
        lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * t / T))
        return {"lr": lr, "beta1": beta1, "beta2": beta2, "eps": eps}
    return _s