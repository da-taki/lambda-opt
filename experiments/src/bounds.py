
import math
from typing import List


def _safe_power(base: float, exponent: int, cap: float = 1e300) -> float:
    """Compute base^exponent safely via log-space."""
    if exponent == 0:
        return 1.0
    if base <= 0:
        return 0.0
    log_val = exponent * math.log(base)
    if log_val > math.log(cap):
        return cap
    if log_val < -700:
        return 0.0
    return math.exp(log_val)


def worst_case_bound(delta: float, L: float, t_R: int, T: int) -> List[float]:
    return [0.0 if t < t_R else delta * _safe_power(L, t - t_R)
            for t in range(T + 1)]


def apriori_bound(delta: float, L_pred: float, t_R: int, T: int) -> List[float]:
    return [0.0 if t < t_R else delta * _safe_power(L_pred, t - t_R)
            for t in range(T + 1)]


def product_bound(delta: float, per_step_ratios: List[float],
                  t_R: int, T: int) -> List[float]:
    bound = [0.0] * t_R
    cumul = delta
    bound.append(cumul)
    for i, r in enumerate(per_step_ratios):
        if t_R + 1 + i >= T:
            break
        cumul *= r
        bound.append(cumul)
    while len(bound) < T + 1:
        bound.append(bound[-1] if bound else 0.0)
    return bound[:T + 1]


def compositional_bound(
    deltas: List[float], L_pred: float,
    rewrite_times: List[int], T: int,
) -> List[float]:
    bound = []
    for t in range(T + 1):
        b = 0.0
        for delta_j, t_Rj in zip(deltas, rewrite_times):
            if t >= t_Rj:
                b += delta_j * _safe_power(L_pred, t - t_Rj)
        bound.append(b)
    return bound


def classify_regime(L_pred: float) -> str:
    if L_pred < 0.999:
        return "contractive"
    elif L_pred > 1.001:
        return "expansive"
    else:
        return "boundary"


def is_vacuous(bound: List[float], threshold: float = 1e30) -> bool:
    """A bound is vacuous if it ever exceeds `threshold` (overflow regime)."""
    return any(b > threshold for b in bound)