"""Tests: local Lipschitz estimation sanity."""
import torch, pytest
from src import TrainingState, make_constant_schedule, AdamStep, SGDStep
from src.lipschitz_local import estimate_lipschitz_at_step


def _identity_loss(n=10):
    A = torch.eye(n)
    return lambda theta: (0.5 * (theta @ A @ theta).item(), A @ theta)


@pytest.fixture
def state():
    torch.manual_seed(0)
    return TrainingState(
        theta=torch.randn(10), moments=torch.zeros(10, 2),
        schedule_fn=make_constant_schedule(lr=0.01), t=0)


def test_positive(state):
    assert estimate_lipschitz_at_step(state, AdamStep(), _identity_loss(), n_samples=5) > 0


def test_finite(state):
    assert estimate_lipschitz_at_step(state, AdamStep(), _identity_loss(), n_samples=5) < 1e6


def test_sgd_reasonable():
    """SGD on L=0.5‖θ‖²: theoretical L=|1−lr|=0.9. Check ballpark."""
    torch.manual_seed(42)
    s = TrainingState(
        theta=torch.randn(10), moments=torch.zeros(10, 2),
        schedule_fn=make_constant_schedule(lr=0.1), t=0)
    L = estimate_lipschitz_at_step(s, SGDStep(momentum=0.0), _identity_loss(), n_samples=50, eps=1e-3)
    assert 0.5 < L < 2.0, f"L={L}"