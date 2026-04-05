"""Tests: separation property — L_pred correctly predicts divergence behavior."""
import torch, pytest
from src import (
    TrainingState, make_constant_schedule, SGDStep,
    EMARewrite, run_trajectory, compute_divergence,
)
from src.lipschitz_apriori import apriori_lipschitz_sgd
from src.bounds import apriori_bound, classify_regime
from src.metrics import bound_holds


def _run_rewrite_test(lr, eigs, T=200, safety=1.05):
    n = len(eigs)
    A = torch.diag(eigs.float())
    loss_fn = lambda theta: (0.5 * (theta @ A @ theta).item(), A @ theta)
    step_fn = SGDStep(momentum=0.0)
    schedule = make_constant_schedule(lr=lr)

    L_pred = apriori_lipschitz_sgd(eigs, lr=lr, momentum=0.0)

    torch.manual_seed(0)
    s0 = TrainingState(theta=torch.randn(n)*0.1, moments=torch.zeros(n,2),
                       schedule_fn=schedule, t=0)
    baseline = run_trajectory(s0.clone(), step_fn, loss_fn, T)

    t_R = T // 2
    R = EMARewrite(
        theta_ema=baseline.states[t_R].theta + torch.randn(n) * 0.01,
        alpha=1.0)
    delta = R.delta(baseline.states[t_R])

    torch.manual_seed(0)
    rw_log = run_trajectory(s0.clone(), step_fn, loss_fn, T, rewrites=[(t_R, R)])
    div = compute_divergence(baseline, rw_log)

    L_bound = L_pred * safety
    bnd = apriori_bound(delta, L_bound, t_R, T)
    holds = bound_holds(div, bnd)

    # filter out inf/nan
    post = [d for d in div[t_R:] if d > 1e-15 and d < 1e30]
    if len(post) >= 2:
        grew = max(post) > post[0] * 1.5
    else:
        grew = False

    return L_pred, grew, holds


def test_contractive_decays():
    eigs = torch.tensor([1.0, 3.0, 5.0, 8.0, 10.0])
    L_pred, grew, holds = _run_rewrite_test(lr=0.01, eigs=eigs)
    assert L_pred < 1.0
    assert not grew
    assert holds


def test_expansive_grows():
    """Mildly expansive: L_pred ≈ 1.15 so divergence grows without overflow."""
    # |1 - 0.03 * 38| = |1 - 1.14| = 0.14... wait
    # We want max_i |1 - lr*λ_i| slightly > 1
    # eigs = [1, 2, 5, 10], lr = 0.11 → |1-1.1|=0.1, |1-0.22|=0.78, |1-0.55|=0.45, |1-1.1|=0.1
    # Need: lr * max_eig slightly > 2 → |1 - lr*max_eig| > 1
    # eigs = [1, 5, 10], lr = 0.22 → |1-2.2| = 1.2 ✓
    eigs = torch.tensor([1.0, 3.0, 5.0, 10.0])
    lr = 0.22
    L_pred, grew, holds = _run_rewrite_test(lr=lr, eigs=eigs, T=40)
    assert L_pred > 1.0, f"L_pred={L_pred}"
    assert grew, f"Divergence should grow (L_pred={L_pred:.3f})"


def test_bound_always_holds():
    configs = [
        (torch.tensor([1.0, 5.0, 10.0]), 0.01),
        (torch.tensor([1.0, 5.0, 10.0]), 0.15),
        (torch.tensor([1.0, 2.0, 3.0]), 0.05),
    ]
    for eigs, lr in configs:
        _, _, holds = _run_rewrite_test(lr=lr, eigs=eigs, T=100, safety=1.1)
        assert holds


def test_regime_classification_matches_behavior():
    eigs = torch.tensor([1.0, 2.0, 4.0])
    L, grew, _ = _run_rewrite_test(lr=0.01, eigs=eigs)
    assert classify_regime(L) == "contractive"
    assert not grew

    eigs2 = torch.tensor([1.0, 3.0, 5.0, 10.0])
    L2, grew2, _ = _run_rewrite_test(lr=0.22, eigs=eigs2, T=40)
    assert classify_regime(L2) == "expansive"
    assert grew2