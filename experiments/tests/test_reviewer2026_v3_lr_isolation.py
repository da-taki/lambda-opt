from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import run_reviewer2026_v3_lr_isolation as V3


def test_empty_state_reset_preserves_checkpoint_param_groups():
    model = torch.nn.Linear(3, 2)
    opt = torch.optim.AdamW(model.parameters(), lr=0.0005, betas=(0.8, 0.95), eps=1e-7, weight_decay=0.02, amsgrad=True)
    x = torch.randn(4, 3)
    loss = model(x).pow(2).sum()
    loss.backward()
    opt.step()
    checkpoint_state = opt.state_dict()
    fresh = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    V3.load_optimizer_param_groups_empty_state(fresh, checkpoint_state)
    assert V3.public_groups(fresh.state_dict()) == V3.public_groups(checkpoint_state)
    assert fresh.param_groups[0]["lr"] == checkpoint_state["param_groups"][0]["lr"]
    assert fresh.param_groups[0]["betas"] == checkpoint_state["param_groups"][0]["betas"]
    assert fresh.param_groups[0]["eps"] == checkpoint_state["param_groups"][0]["eps"]
    assert fresh.param_groups[0]["weight_decay"] == checkpoint_state["param_groups"][0]["weight_decay"]
    assert fresh.param_groups[0]["amsgrad"] is True
    assert V3.opt_state_entries(fresh) == 0
    assert V3.opt_state_keys(fresh) == []