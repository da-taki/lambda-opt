"""λOpt Metrics for theorem-to-empirical validation."""
import math
import numpy as np
from scipy import stats
from typing import List


def tightness_ratio(actual: List[float], bound: List[float]) -> List[float]:
    """D(t)/B(t) ∈ [0,1]. 0 where B=0 or B=inf."""
    out = []
    for a, b in zip(actual, bound):
        if b > 1e-12 and b < 1e200:
            out.append(min(a / b, 1.0))
        else:
            out.append(0.0)
    return out


def bound_holds(actual: List[float], bound: List[float], tol: float = 1e-6) -> bool:
    """Thm 1/2 pass: D(t) ≤ B(t) ∀t."""
    return all(a <= b + tol for a, b in zip(actual, bound))


def loss_gap(losses_a: List[float], losses_b: List[float]) -> List[float]:
    return [abs(a - b) for a, b in zip(losses_a, losses_b)]


def spearman_correlation(x: List[float], y: List[float]) -> float:
    r, _ = stats.spearmanr(x, y)
    return float(r) if not math.isnan(r) else 0.0


def delta_linearity_r2(deltas: List[float], divs: List[float]) -> float:
    if len(set(deltas)) < 2:
        return 0.0
    _, _, r, _, _ = stats.linregress(deltas, divs)
    return r ** 2


def median_lipschitz(estimates: List[float], warmup_frac: float = 0.1) -> float:
    start = int(len(estimates) * warmup_frac)
    return float(np.median(estimates[start:])) if estimates[start:] else 1.0


def effective_lipschitz_from_divergence(
    divergence: List[float], t_R: int,
) -> float:
    """Geometric mean of D(t+1)/D(t) post-rewrite. Descriptive only."""
    post = divergence[t_R:]
    log_ratios = []
    for i in range(len(post) - 1):
        if post[i] > 1e-15:
            log_ratios.append(math.log(max(post[i+1], 1e-30) / post[i]))
    if not log_ratios:
        return 1.0
    return math.exp(sum(log_ratios) / len(log_ratios))