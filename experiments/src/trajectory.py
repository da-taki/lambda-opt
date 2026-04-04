"""λOpt Trajectory Runner.
τ(S₀,T) = [S₀, Step(S₀), …, Stepᵀ(S₀)]
"""
import torch
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from .state import TrainingState
from .step import StepFn
from .rewrites import Rewrite


@dataclass
class TrajectoryLog:
    thetas: List[torch.Tensor] = field(default_factory=list)
    losses: List[float] = field(default_factory=list)
    lipschitz: List[float] = field(default_factory=list)
    states: List[TrainingState] = field(default_factory=list)


def run_trajectory(
    state: TrainingState, step_fn: StepFn, loss_fn, T: int,
    rewrites: Optional[List[Tuple[int, Rewrite]]] = None,
) -> TrajectoryLog:
    rewrites = sorted(rewrites or [], key=lambda x: x[0])
    rw_idx = 0
    log = TrajectoryLog()
    s = state.clone()
    for step in range(T):
        while rw_idx < len(rewrites) and rewrites[rw_idx][0] == step:
            s = rewrites[rw_idx][1].apply(s)
            rw_idx += 1
        log.thetas.append(s.theta.clone())
        log.states.append(s.clone())
        loss, grad = loss_fn(s.theta)
        log.losses.append(loss)
        s, emp_L = step_fn(s, grad)
        log.lipschitz.append(emp_L)
    log.thetas.append(s.theta.clone())
    log.states.append(s.clone())
    return log


def compute_divergence(baseline: TrajectoryLog, rewritten: TrajectoryLog) -> List[float]:
    return [(r - b).norm(2).item() for b, r in zip(baseline.thetas, rewritten.thetas)]