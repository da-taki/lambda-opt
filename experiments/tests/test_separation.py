"""Tests: separation property — L_pred correctly predicts divergence behavior.

This is the theorem's non-triviality test:
  L_pred < 1 → divergence MUST decay after rewrite
  L_pred > 1 → divergence CAN grow (and does on our expansive config)

We verify on small quadratics with known spectrum.
"""
import torch, pytest
from src import (
    TrainingState, make_constant_schedule, AdamStep, SGDStep,
    EMARewrite, run_trajectory, compute_divergence,
)
from src.lipschitz_apriori import apriori_lipschitz_sgd
from src.bounds import apriori_bound, classify_regime
from src.metrics import bound_holds


def _run_rewrite_test(lr, eigs, T=200):
    """Run SGD (no momentum) on quadratic with given eigenvalues.
    Returns (L_pred, divergence_grew, bound_held).
    """
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

    bnd = apriori_bound(delta, L_pred, t_R, T)
    holds = bound_holds(div, bnd)

    post = [d for d in div[t_R:] if d > 1e-15]
    grew = post[-1] > post[0] * 1.1 if len(post) >= 2 else False

    return L_pred, grew, holds


def test_contractive_decays():
    """Small LR → L_pred < 1 → divergence decays."""
    eigs = torch.tensor([1.0, 3.0, 5.0, 8.0, 10.0])
    L_pred, grew, holds = _run_rewrite_test(lr=0.01, eigs=eigs)
    assert L_pred < 1.0, f"L_pred={L_pred}"
    assert not grew, "Divergence should decay in contractive regime"
    assert holds, "Bound must hold in contractive regime"


def test_expansive_grows():
    """Large LR → L_pred > 1 → divergence grows."""
    eigs = torch.tensor([1.0, 3.0, 5.0, 8.0, 30.0])
    L_pred, grew, holds = _run_rewrite_test(lr=0.1, eigs=eigs, T=100)
    assert L_pred > 1.0, f"L_pred={L_pred}"
    assert grew, "Divergence should grow in expansive regime"
    # bound still holds (it's an upper bound, just a large one)
    assert holds, "Bound must still hold even in expansive regime"


def test_bound_always_holds():
    """Regardless of regime, the a priori bound must never be violated."""
    configs = [
        (torch.tensor([1.0, 5.0, 10.0]), 0.01),   # contractive
        (torch.tensor([1.0, 5.0, 10.0]), 0.15),    # expansive
        (torch.tensor([1.0, 2.0, 3.0]), 0.05),     # near boundary
    ]
    for eigs, lr in configs:
        L_pred, _, holds = _run_rewrite_test(lr=lr, eigs=eigs, T=100)
        regime = classify_regime(L_pred)
        assert holds, f"Bound violated! lr={lr}, L_pred={L_pred:.4f} [{regime}]"


def test_regime_classification_matches_behavior():
    """classify_regime should agree with actual divergence behavior."""
    # contractive
    eigs = torch.tensor([1.0, 2.0, 4.0])
    L, grew, _ = _run_rewrite_test(lr=0.01, eigs=eigs)
    assert classify_regime(L) == "contractive"
    assert not grew

    # expansive
    eigs2 = torch.tensor([1.0, 2.0, 20.0])
    L2, grew2, _ = _run_rewrite_test(lr=0.15, eigs=eigs2, T=80)
    assert classify_regime(L2) == "expansive"
    assert grew2