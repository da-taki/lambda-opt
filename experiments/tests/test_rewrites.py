"""Tests: rewrite δ-bound correctness and apply semantics."""
import torch, pytest
from src import TrainingState, make_constant_schedule
from src import EMARewrite, CheckpointRewrite, LoRAFreezeRewrite


@pytest.fixture
def state():
    torch.manual_seed(0)
    return TrainingState(
        theta=torch.randn(10), moments=torch.randn(10, 2),
        schedule_fn=make_constant_schedule(), t=50)


def test_ema_delta_matches_distance(state):
    theta_ema = state.theta + torch.randn(10) * 0.1
    R = EMARewrite(theta_ema=theta_ema, alpha=0.999)
    new = R.apply(state)
    assert abs(TrainingState.theta_distance(state, new) - R.delta(state)) < 1e-5


def test_ema_moments_preserved(state):
    R = EMARewrite(theta_ema=torch.randn(10), alpha=0.999)
    new = R.apply(state)
    assert torch.allclose(new.moments, state.moments)
    assert new.t == state.t


def test_checkpoint_delta_is_moment_distance(state):
    stale = state.moments + torch.randn(10, 2) * 0.05
    R = CheckpointRewrite(stale_moments=stale, stale_t=10)
    assert abs(R.delta(state) - (stale - state.moments).norm(2).item()) < 1e-5


def test_checkpoint_preserves_theta(state):
    R = CheckpointRewrite(stale_moments=torch.zeros(10, 2), stale_t=0)
    assert torch.allclose(R.apply(state).theta, state.theta)


def test_lora_zeros_frozen_moments(state):
    R = LoRAFreezeRewrite(frozen_indices=torch.arange(5))
    new = R.apply(state)
    assert (new.moments[:5] == 0).all()
    assert torch.allclose(new.moments[5:], state.moments[5:])


def test_lora_delta_is_frozen_norm(state):
    R = LoRAFreezeRewrite(frozen_indices=torch.arange(5))
    assert abs(R.delta(state) - state.moments[:5].norm(2).item()) < 1e-5


def test_delta_bounds_component_distance(state):
    """Core property: δ bounds the actual displacement per component."""
    R_ema = EMARewrite(theta_ema=torch.randn(10), alpha=0.99)
    assert TrainingState.theta_distance(state, R_ema.apply(state)) <= R_ema.delta(state) + 1e-5

    R_ckpt = CheckpointRewrite(stale_moments=torch.randn(10, 2), stale_t=0)
    new_ckpt = R_ckpt.apply(state)
    assert (new_ckpt.moments - state.moments).norm(2).item() <= R_ckpt.delta(state) + 1e-5

    R_lora = LoRAFreezeRewrite(frozen_indices=torch.arange(3))
    new_lora = R_lora.apply(state)
    assert (new_lora.moments - state.moments).norm(2).item() <= R_lora.delta(state) + 1e-5