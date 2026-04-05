"""Tests: a priori Lipschitz prediction accuracy."""
import torch, pytest
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
    assert abs(L_pred - L_exact) < 1e-6


def test_sgd_contractive_regime():
    eigs = torch.tensor([1.0, 2.0, 5.0])
    assert apriori_lipschitz_sgd(eigs, lr=0.01, momentum=0.0) < 1.0


def test_sgd_expansive_regime():
    eigs = torch.tensor([1.0, 2.0, 50.0])
    assert apriori_lipschitz_sgd(eigs, lr=0.1, momentum=0.0) > 1.0


def test_sgd_with_momentum():
    eigs = torch.tensor([1.0, 5.0, 10.0])
    L_no = apriori_lipschitz_sgd(eigs, lr=0.05, momentum=0.0)
    L_mom = apriori_lipschitz_sgd(eigs, lr=0.05, momentum=0.9)
    assert L_mom >= L_no * 0.5


def test_adam_returns_positive():
    eigs = torch.tensor([1.0, 5.0, 10.0])
    torch.manual_seed(0)
    s = TrainingState(
        theta=torch.randn(3), moments=torch.randn(3, 2).abs(),
        schedule_fn=make_constant_schedule(lr=0.001), t=100)
    L = apriori_lipschitz_quadratic(eigs, lr=0.001, t=100, state=s)
    assert L > 0


def test_adam_zero_lr_gives_L_near_one():
    """With LR→0, update vanishes, so L_pred → 1 (identity map)."""
    A = make_quadratic_hessian(10, 10.0, seed=0)
    eigs = hessian_eigenvalues_exact(A)
    torch.manual_seed(0)
    s = TrainingState(
        theta=torch.randn(10), moments=torch.randn(10, 2).abs() * 0.01,
        schedule_fn=make_constant_schedule(lr=1e-10), t=100)
    L = apriori_lipschitz_quadratic(eigs, lr=1e-10, t=100, state=s)
    assert abs(L - 1.0) < 0.01, f"L={L}, expected ~1.0"


def test_hessian_eigenvalues_correct():
    A = make_quadratic_hessian(5, 20.0, seed=42)
    eigs = hessian_eigenvalues_exact(A)
    assert eigs.min() > 0
    assert abs(eigs.min().item() - 1.0) < 0.5
    assert abs(eigs.max().item() - 20.0) < 0.5