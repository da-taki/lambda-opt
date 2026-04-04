"""λOpt Bound Computations.

Four bounds, from loosest to tightest:
  1. worst_case_bound:    δ · L_local^(t−t_R)     — vacuous when L>1
  2. apriori_bound:       δ · L_pred^(t−t_R)      — predictive, from Hessian
  3. product_bound:       δ · ∏ L_i               — tightest, uses per-step data
  4. compositional_bound: Σ_j δ_j · L^(t−t_Rj)   — Thm 2 for multiple rewrites

All use safe log-space arithmetic to avoid overflow.
"""
import math
from typing import List, Tuple


def _safe_power(base: float, exponent: int, cap: float = 1e300) -> float:
    """Compute base^exponent safely via log-space."""
    if base <= 0 or exponent <= 0:
        return base if exponent == 0 else 0.0
    log_val = exponent * math.log(base)
    if log_val > math.log(cap):
        return cap
    if log_val < -700:
        return 0.0
    return math.exp(log_val)


def worst_case_bound(delta: float, L: float, t_R: int, T: int) -> List[float]:
    """B(t) = δ · L^(t−t_R). Worst-case, typically vacuous for L >> 1."""
    return [0.0 if t < t_R else delta * _safe_power(L, t - t_R)
            for t in range(T + 1)]


def apriori_bound(delta: float, L_pred: float, t_R: int, T: int) -> List[float]:
    """B(t) = δ · L_pred^(t−t_R). A priori from Hessian spectrum.
    When L_pred < 1, this is tight and exponentially decaying.
    When L_pred > 1, this correctly predicts divergence growth."""
    return [0.0 if t < t_R else delta * _safe_power(L_pred, t - t_R)
            for t in range(T + 1)]


def product_bound(delta: float, per_step_ratios: List[float],
                  t_R: int, T: int) -> List[float]:
    """B(t) = δ · ∏_{i=0}^{t−t_R} L_i. Tightest possible.
    Uses actual per-step contraction ratios from divergence data.
    NOTE: This is descriptive, not predictive — used for comparison only."""
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
    deltas: List[float],
    L_pred: float,
    rewrite_times: List[int],
    T: int,
) -> List[float]:
    """Thm 2: B(t) = Σ_j δ_j · L_pred^(t − t_Rj) for t ≥ t_Rj.
    Generalizes to n sequential rewrites via summation."""
    bound = []
    for t in range(T + 1):
        b = 0.0
        for delta_j, t_Rj in zip(deltas, rewrite_times):
            if t >= t_Rj:
                b += delta_j * _safe_power(L_pred, t - t_Rj)
        bound.append(b)
    return bound


def classify_regime(L_pred: float) -> str:
    """Classify the rewrite regime for reporting.
    contractive: L < 0.999  → perturbation decays, rewrite is safe
    boundary:    0.999 ≤ L ≤ 1.001 → perturbation persists
    expansive:   L > 1.001  → perturbation grows, rewrite is dangerous
    """
    if L_pred < 0.999:
        return "contractive"
    elif L_pred > 1.001:
        return "expansive"
    else:
        return "boundary"