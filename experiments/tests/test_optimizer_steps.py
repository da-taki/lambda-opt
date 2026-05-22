import torch

from src import AdamWStep, RMSPropStep, TrainingState, make_constant_schedule


def test_adamw_zero_gradient_applies_weight_decay():
    state = TrainingState(
        theta=torch.ones(4),
        moments=torch.zeros(4, 2),
        schedule_fn=make_constant_schedule(lr=0.1),
        t=0,
    )
    new_state, _ = AdamWStep(weight_decay=0.01)(state, torch.zeros(4))
    assert torch.allclose(new_state.theta, torch.ones(4) * 0.999)


def test_rmsprop_step_is_finite():
    state = TrainingState(
        theta=torch.ones(4),
        moments=torch.zeros(4, 2),
        schedule_fn=make_constant_schedule(lr=0.01),
        t=0,
    )
    new_state, emp_l = RMSPropStep()(state, torch.arange(1.0, 5.0))
    assert torch.isfinite(new_state.theta).all()
    assert torch.isfinite(new_state.moments).all()
    assert emp_l > 0
