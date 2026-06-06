"""Tests for the SNCS v4 shared infrastructure (scripts/sncs_v4_common.py).

These validate the parts the reviewer experiments depend on:
  * the 12-scenario taxonomy is complete,
  * the repaired metric assigns NON-zero displacement to a pure scheduler
    change while the old (theta,M) metric assigns ZERO -- the central repair,
  * the denominator-aware metric reacts to a second-moment reset,
  * the analyzer returns only valid decisions and the bound math is finite.
"""
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import sncs_v4_common as C  # noqa: E402


@pytest.fixture(scope="module")
def ctx():
    return C.build_quadratic(seed=42)


def test_scenarios_complete(ctx):
    scs = C.build_scenarios(ctx)
    assert len(scs) == 12
    assert [s.name for s in scs] == C.SCENARIOS


def test_repaired_metric_fixes_scheduler_mismatch(ctx):
    scs = {s.name: s for s in C.build_scenarios(ctx)}
    sm = scs["scheduler_mismatch"]
    raw = C.metric_delta(ctx.resume_state, sm.state_after, "raw_theta_M", sm.comps)
    rep = C.metric_delta(ctx.resume_state, sm.state_after, "full_repaired", sm.comps)
    # Old metric is blind to a pure schedule change ...
    assert raw == pytest.approx(0.0, abs=1e-9)
    # ... the repaired metric is not.
    assert rep > 1e-6


def test_denom_aware_reacts_to_v_reset(ctx):
    scs = {s.name: s for s in C.build_scenarios(ctx)}
    vr = scs["reset_second_moment"]
    denom_aware = C.metric_delta(ctx.resume_state, vr.state_after, "denom_aware_v", vr.comps)
    assert denom_aware > 0.0
    assert math.isfinite(denom_aware)


def test_all_metric_variants_finite_nonneg(ctx):
    for s in C.build_scenarios(ctx):
        for mt in C.METRIC_TYPES:
            d = C.metric_delta(ctx.resume_state, s.state_after, mt, s.comps)
            assert math.isfinite(d) and d >= 0.0


def test_proxy_update_finite_on_v_reset(ctx):
    import torch
    m = ctx.resume_state.moments[:, 0]
    v_reset = torch.zeros_like(ctx.resume_state.moments[:, 1])
    upd = C.proxy_step_update(ctx.resume_state.theta, m, v_reset, ctx.base_hp, ctx.resume_state.t)
    assert torch.isfinite(upd).all()


def test_analyzer_decisions_valid(ctx):
    L_raw, L_pred = C.estimate_L(ctx.resume_state, ctx.step_fn,
                                 ctx.make_loss_fn(ctx.pretrain_steps),
                                 n_perturbations=ctx.lip_perturbations,
                                 n_steps=ctx.lip_steps, eps=ctx.lip_eps)
    valid = {"certified_safe", "flagged_dangerous", "abstained"}
    for s in C.build_scenarios(ctx):
        rd = C.repaired_delta(ctx.resume_state, s.state_after)
        dec = C.analyzer_decision(rd, L_pred, ctx.post_steps, epsilon=0.05,
                                  margin=1.1, danger_threshold=ctx.div_threshold)
        assert dec.decision in valid
        assert math.isfinite(dec.bound_final)


def test_ground_truth_labels(ctx):
    s = C.build_scenarios(ctx)[0]  # exact_restore -> benign
    gt = C.run_ground_truth(ctx, s, steps=ctx.post_steps)
    assert gt.actual_label in {"benign", "dangerous"}
    assert gt.actual_label == "benign"
    assert len(gt.divergence) == ctx.post_steps + 1


def test_stepwise_bound_matches_product(ctx):
    curve = C.stepwise_bound_curve(2.0, [0.9, 0.9, 0.9])
    assert curve[0] == pytest.approx(2.0)
    assert curve[-1] == pytest.approx(2.0 * 0.9 ** 3)
