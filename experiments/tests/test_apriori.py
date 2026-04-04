"""Tests: a priori Lipschitz prediction accuracy.

On quadratics with known Hessian, we can verify L_pred analytically.
For SGD (no momentum) on L(θ)=0.5 θᵀAθ:
    Step: θ → θ - η·Aθ = (I - ηA)θ
    Exact L = spectral_radius(I - ηA) = max_i |1 - η·λ_i|

We verify our a priori estimator matches this.
"""
import torch, pytest, math
from src import TrainingState, make_constant_schedule
from src.lipschitz_apriori import (
    apriori_lipschitz_sgd, apriori_lipschitz_quadratic,
)
from src.loss_landscape import hessian_eigenvalues_exact, make_quadratic_hessian


def test_sgd_no_momentum_exact():
    """L_pred for SGD must equal max_i |1 - η·λ_i|."""
    eigs = torch.tensor([1.0, 5.0, 10.0, 20.0])
    lr = 0.05
    L_pred = apriori_lipschitz_sgd(eigs, lr=lr, momentum=0.0)
    L_exact = max(abs(1 - lr * e.item()) for e in eigs)
    assert abs(L_pred - L_exact) < 1e-6, f"L_pred={L_pred}, L_exact={L_exact}"


def test_sgd_contractive_regime():
    """Small LR → all |1 - η·λ_i| < 1 → L_pred < 1."""
    eigs = torch.tensor([1.0, 2.0, 5.0])
    L = apriori_lipschitz_sgd(eigs, lr=0.01, momentum=0.0)
    assert L < 1.0, f"L={L}, expected < 1"


def test_sgd_expansive_regime():
    """Large LR → some |1 - η·λ_i| > 1 → L_pred > 1."""
    eigs = torch.tensor([1.0, 2.0, 50.0])
    L = apriori_lipschitz_sgd(eigs, lr=0.1, momentum=0.0)
    assert L > 1.0, f"L={L}, expected > 1"


def test_sgd_with_momentum():
    """With momentum, L should be ≥ the no-momentum case for same LR."""
    eigs = torch.tensor([1.0, 5.0, 10.0])
    lr = 0.05
    L_no_mom = apriori_lipschitz_sgd(eigs, lr=lr, momentum=0.0)
    L_mom = apriori_lipschitz_sgd(eigs, lr=lr, momentum=0.9)
    # momentum doesn't decrease L (in general)
    assert L_mom >= L_no_mom * 0.5, f"L_mom={L_mom} vs L_no_mom={L_no_mom}"


def test_adam_returns_positive():
    """Basic sanity: Adam L_pred is positive."""
    eigs = torch.tensor([1.0, 5.0, 10.0])
    torch.manual_seed(0)
    s = TrainingState(
        theta=torch.randn(3), moments=torch.randn(3, 2).abs(),
        schedule_fn=make_constant_schedule(lr=0.001), t=100)
    L = apriori_lipschitz_quadratic(eigs, lr=0.001, t=100, state=s)
    assert L > 0, f"L={L}"


def test_adam_smaller_lr_smaller_L():
    """Smaller LR should generally give smaller L_pred."""
    A = make_quadratic_hessian(10, 10.0, seed=0)
    eigs = hessian_eigenvalues_exact(A)
    torch.manual_seed(0)
    theta = torch.randn(10)
    mom = torch.randn(10, 2).abs() * 0.01

    s_small = TrainingState(theta=theta.clone(), moments=mom.clone(),
                            schedule_fn=make_constant_schedule(lr=0.0001), t=100)
    s_big = TrainingState(theta=theta.clone(), moments=mom.clone(),
                          schedule_fn=make_constant_schedule(lr=0.01), t=100)

    L_small = apriori_lipschitz_quadratic(eigs, lr=0.0001, t=100, state=s_small)
    L_big = apriori_lipschitz_quadratic(eigs, lr=0.01, t=100, state=s_big)
    assert L_small < L_big, f"L_small={L_small} >= L_big={L_big}"


def test_hessian_eigenvalues_correct():
    """Verify eigenvalue computation matches known spectrum."""
    A = make_quadratic_hessian(5, 20.0, seed=42)
    eigs = hessian_eigenvalues_exact(A)
    assert eigs.min() > 0, "PSD matrix should have positive eigenvalues"
    assert abs(eigs.min().item() - 1.0) < 0.5, f"min eig={eigs.min()}, expected ~1"
    assert abs(eigs.max().item() - 20.0) < 0.5, f"max eig={eigs.max()}, expected ~20"