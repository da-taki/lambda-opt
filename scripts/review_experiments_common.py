"""Shared infrastructure for the supplementary review experiments.

This module is the single source of truth for:

  * the four diagnostics (quadratic, MNIST MLP, ResNet-18/CIFAR-10, GPT-mini),
  * the canonical 12 checkpoint-rewrite scenarios,
  * the five displacement-metric variants used in the metric-repair ablation,
  * the repaired lambda-Opt analyzer (delta, sampled-L, decision with abstain),
  * ground-truth post-resume trajectory runners,
  * deterministic seeding and CSV/JSON/Markdown IO helpers.

Every experiment script imports from here so that the *same* analyzer and the
*same* scenario definitions are used everywhere.

Design notes (honesty matters for the paper):

  * L is estimated by finite sampling of perturbation directions
    (``apriori_lipschitz_numerical``).  This is an EMPIRICAL estimate, not a
    proof certificate.  Decisions are therefore labelled "analyzer decisions",
    never "certified" in the formal sense.  See experiment 9 for the stress
    test that quantifies the gap.

  * The "repaired" displacement metric maps a state-space rewrite (theta,
    moments, schedule, time) into the theta-space units in which the divergence
    D(t) and the Lipschitz constant L are actually measured.  This is the fix
    for the original metric, which assigned delta=0 to a pure scheduler change
    (and therefore produced a vacuous / violated bound).  The mapping uses only
    information available from the checkpoint (it is a priori).

The reference optimizer everywhere is Adam, matching ``src.step.AdamStep``.
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

# --------------------------------------------------------------------------
# Paths and imports from the existing experiment source tree
# --------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = REPO_ROOT / "experiments"
sys.path.insert(0, str(EXPERIMENTS))
sys.path.insert(0, str(EXPERIMENTS / "scripts"))

RESULTS_DIR = REPO_ROOT / "results" / "review_experiments"

from src import (  # noqa: E402
    AdamStep,
    TrainingState,
    apriori_lipschitz_numerical,
    make_constant_schedule,
)
from src.bounds import _safe_power, classify_regime, is_vacuous  # noqa: E402

# The 12 canonical scenario identifiers, in the order requested by the brief.
SCENARIOS: List[str] = [
    "exact_restore",
    "noop_restore",
    "stale_adam_moments",
    "reset_first_moment",
    "reset_second_moment",
    "scheduler_mismatch",
    "ema_shadow_present_not_loaded",
    "ema_loaded_as_parameters",
    "mild_lr_decrease",
    "severe_lr_increase",
    "optimizer_state_from_older_checkpoint",
    "params_current_optimizer_older",
]

# Scenarios that are *designed* to be benign (used only as a reference hint;
# the ground-truth label always comes from the empirical thresholds).
DESIGNED_BENIGN = {
    "exact_restore",
    "noop_restore",
    "scheduler_mismatch",  # 1.5x is borderline; let the data decide
    "ema_shadow_present_not_loaded",
    "mild_lr_decrease",
}

METRIC_TYPES: List[str] = [
    "raw_theta_M",          # A: original metric over (theta, M) only
    "raw_plus_sched_time",  # B: A + schedule/time terms
    "block_normalized",     # C: per-block normalised
    "denom_aware_v",        # D: Adam denominator-aware v geometry
    "full_repaired",        # E: the metric used by the repaired analyzer
]

DEFAULT_SAFETY_MARGIN = 1.08
EPS_DIV = 1e-12


# --------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# --------------------------------------------------------------------------
# Adam proxy-step geometry (a priori, checkpoint-only)
# --------------------------------------------------------------------------

def hp_at(state: TrainingState) -> Dict[str, float]:
    return state.schedule_fn(state.t)


def proxy_step_update(theta: torch.Tensor, m: torch.Tensor, v: torch.Tensor,
                      hp: Dict[str, float], t: int) -> torch.Tensor:
    """The parameter update Adam would take from this state, a priori.

    We do not have the next gradient at analysis time, so we use the stored
    first moment ``m`` as the gradient-direction proxy (m is an EMA of past
    gradients).  The second moment is advanced one step with that same proxy so
    that a *reset* of v is reflected realistically (a denominator of order |m|
    rather than of order eps).  This makes the update finite and calibrated.

    Returns the theta-space update vector ``lr * m_hat / (sqrt(v_hat) + eps)``.
    """
    lr = hp["lr"]
    b1 = hp["beta1"]
    b2 = hp["beta2"]
    eps = hp["eps"]
    t_bc = max(int(t) + 1, 1)
    m_new = m  # b1*m + (1-b1)*m == m  when grad proxy == m
    v_new = b2 * v.clamp(min=0.0) + (1.0 - b2) * m.pow(2)
    m_hat = m_new / (1.0 - b1 ** t_bc)
    v_hat = v_new / (1.0 - b2 ** t_bc)
    return lr * m_hat / (v_hat.sqrt() + eps)


def _split_moments(state: TrainingState) -> Tuple[torch.Tensor, torch.Tensor]:
    return state.moments[:, 0], state.moments[:, 1]


def component_displacements(before: TrainingState,
                            after: TrainingState) -> Dict[str, float]:
    """Decompose the rewrite displacement into named, theta-commensurate parts.

    All "effect" terms are the theta-space change in the next Adam update that
    the corresponding state change induces, holding everything else at the
    *before* values.  This isolates each contribution.
    """
    m_b, v_b = _split_moments(before)
    m_a, v_a = _split_moments(after)
    hp_b = hp_at(before)
    hp_a = hp_at(after)

    d_theta = (after.theta - before.theta).norm(2).item()
    d_m = (m_a - m_b).norm(2).item()
    d_v = (v_a - v_b).norm(2).item()
    d_M = (after.moments - before.moments).norm(2).item()

    base_update = proxy_step_update(before.theta, m_b, v_b, hp_b, before.t)

    # v effect: change only v (m, hp, t held at before).
    upd_v = proxy_step_update(before.theta, m_b, v_a, hp_b, before.t)
    d_v_effect = (upd_v - base_update).norm(2).item()

    # m effect: change only m.
    upd_m = proxy_step_update(before.theta, m_a, v_b, hp_b, before.t)
    d_m_effect = (upd_m - base_update).norm(2).item()

    # schedule effect: change only hp (lr/betas/eps).
    upd_sched = proxy_step_update(before.theta, m_b, v_b, hp_a, before.t)
    d_sched = (upd_sched - base_update).norm(2).item()

    # time effect: change only the bias-correction time.
    upd_time = proxy_step_update(before.theta, m_b, v_b, hp_b, after.t)
    d_time = (upd_time - base_update).norm(2).item()

    # total injected update change (everything except direct theta move).
    upd_after = proxy_step_update(before.theta, m_a, v_a, hp_a, after.t)
    d_update_total = (upd_after - base_update).norm(2).item()

    norm_theta = before.theta.norm(2).item()
    norm_m = m_b.norm(2).item()
    norm_v = v_b.norm(2).item()

    return {
        "d_theta": d_theta,
        "d_m": d_m,
        "d_v": d_v,
        "d_M": d_M,
        "d_m_effect": d_m_effect,
        "d_v_effect": d_v_effect,
        "d_sched": d_sched,
        "d_time": d_time,
        "d_update_total": d_update_total,
        "norm_theta": norm_theta,
        "norm_m": norm_m,
        "norm_v": norm_v,
    }


def metric_delta(before: TrainingState, after: TrainingState,
                 metric_type: str,
                 comps: Optional[Dict[str, float]] = None) -> float:
    """Scalar displacement delta under one of the five metric variants.

    All variants are returned in theta-space units so that the bound
    ``D(t_R + k) <= delta * L^k`` is dimensionally consistent with the measured
    theta divergence.  (Variant C normalises per block then rescales by the
    parameter-block norm to return to theta units.)
    """
    c = comps if comps is not None else component_displacements(before, after)
    if metric_type == "raw_theta_M":
        return math.hypot(c["d_theta"], c["d_M"])
    if metric_type == "raw_plus_sched_time":
        return math.sqrt(c["d_theta"] ** 2 + c["d_M"] ** 2
                         + c["d_sched"] ** 2 + c["d_time"] ** 2)
    if metric_type == "block_normalized":
        s_theta = c["norm_theta"] + EPS_DIV
        ref = c["norm_theta"] + EPS_DIV  # rescale back to theta units
        n_theta = c["d_theta"] / s_theta
        n_m = c["d_m"] / (c["norm_m"] + EPS_DIV)
        n_v = c["d_v"] / (c["norm_v"] + EPS_DIV)
        return ref * math.sqrt(n_theta ** 2 + n_m ** 2 + n_v ** 2)
    if metric_type == "denom_aware_v":
        return math.sqrt(c["d_theta"] ** 2 + c["d_m"] ** 2 + c["d_v_effect"] ** 2)
    if metric_type == "full_repaired":
        # conservative theta-space sum: direct move + injected update change.
        return c["d_theta"] + c["d_update_total"]
    raise ValueError(f"unknown metric_type {metric_type}")


def repaired_delta(before: TrainingState, after: TrainingState) -> float:
    return metric_delta(before, after, "full_repaired")


# --------------------------------------------------------------------------
# Sampled-L estimation (with selectable perturbation directions)
# --------------------------------------------------------------------------

def estimate_L(state: TrainingState, step_fn, loss_fn,
               n_perturbations: int = 5, n_steps: int = 5, eps: float = 1e-4,
               safety_margin: float = DEFAULT_SAFETY_MARGIN) -> Tuple[float, float]:
    """Return (L_raw, L_pred) where L_pred = L_raw * safety_margin.

    Thin wrapper over ``apriori_lipschitz_numerical`` (random directions).
    """
    L_raw = apriori_lipschitz_numerical(
        state, step_fn, loss_fn,
        n_perturbations=n_perturbations, n_steps=n_steps, eps=eps)
    return L_raw, L_raw * safety_margin


def estimate_L_directional(state: TrainingState, step_fn, loss_fn,
                           direction_fn: Callable[[TrainingState], torch.Tensor],
                           n_probes: int = 5, n_steps: int = 5,
                           eps: float = 1e-4) -> Tuple[float, float]:
    """Geometric-mean per-step contraction along directions from ``direction_fn``.

    ``direction_fn(state)`` returns an (unnormalised) theta-space vector; we
    normalise it and scale by ``eps``.  Returns (L_estimate, max_growth) where
    max_growth is the largest single d_after/d_before ratio seen (worst case
    among the probes), used by the sampling-stress experiment.
    """
    ratios: List[float] = []
    max_ratio = 0.0
    for _ in range(n_probes):
        direction = direction_fn(state)
        norm = direction.norm(2).item()
        if norm < 1e-30:
            direction = torch.randn_like(state.theta)
            norm = direction.norm(2).item()
        perturb = direction / norm * eps
        s_base = state.clone()
        s_pert = state.clone()
        s_pert.theta = s_pert.theta + perturb
        d_prev = perturb.norm(2).item()
        for _ in range(n_steps):
            _, g_base = loss_fn(s_base.theta)
            _, g_pert = loss_fn(s_pert.theta)
            s_base, _ = step_fn(s_base, g_base)
            s_pert, _ = step_fn(s_pert, g_pert)
            d_after = (s_pert.theta - s_base.theta).norm(2).item()
            if d_prev > 1e-15 and d_after > 1e-30:
                r = d_after / d_prev
                ratios.append(r)
                max_ratio = max(max_ratio, r)
            d_prev = d_after
    if not ratios:
        return 1.0, max_ratio
    log_r = [math.log(max(r, 1e-30)) for r in ratios]
    return math.exp(sum(log_r) / len(log_r)), max_ratio


# --------------------------------------------------------------------------
# Bounds (fixed and step-varying)
# --------------------------------------------------------------------------

def baseline_per_step_L(ctx: "DiagContext", steps: int, sample_every: int = 1,
                        margin: float = DEFAULT_SAFETY_MARGIN) -> List[float]:
    """A-priori per-step Lipschitz constants L_j along the *known* baseline
    continuation from the resume point.

    This is genuinely a-priori: the clean resume trajectory does not depend on
    any rewrite, so we may estimate the local contraction at each step before
    committing to a rewrite. ``sample_every`` lets expensive diagnostics estimate
    less often (the last value is held)."""
    advance = ctx.make_loss_fn(ctx.pretrain_steps)
    s = ctx.resume_state.clone()
    per_step: List[float] = []
    last = 1.0
    for j in range(steps):
        if j % sample_every == 0:
            lf = ctx.make_loss_fn(ctx.pretrain_steps + j)
            _, last = estimate_L(s, ctx.step_fn, lf,
                                 n_perturbations=ctx.lip_perturbations,
                                 n_steps=ctx.lip_steps, eps=ctx.lip_eps,
                                 safety_margin=margin)
        per_step.append(last)
        _, g = advance(s.theta)
        s, _ = ctx.step_fn(s, g)
    return per_step


def fixed_bound_curve(delta: float, L: float, k_max: int) -> List[float]:
    return [delta * _safe_power(L, k) for k in range(k_max + 1)]


def stepwise_bound_curve(delta: float, per_step_L: List[float]) -> List[float]:
    """D(t_R + k) <= delta * prod_{j<k} L_j.  Returns curve of length len+1."""
    out = [delta]
    cumul = delta
    for L_j in per_step_L:
        cumul = cumul * L_j
        out.append(cumul)
    return out


# --------------------------------------------------------------------------
# The repaired analyzer decision
# --------------------------------------------------------------------------

@dataclass
class Decision:
    decision: str          # certified_safe | flagged_dangerous | abstained
    reason: str
    bound_final: float
    L_pred: float
    regime: str
    delta: float


def analyzer_decision(delta: float, L_pred: float, post_steps: int,
                      epsilon: float, margin: float,
                      danger_threshold: float,
                      stepwise_per_step_L: Optional[List[float]] = None
                      ) -> Decision:
    """Map (delta, L) into a safe/dangerous/abstain decision.

    * certified_safe  : the (margin-inflated) final bound is <= epsilon, i.e.
                        the analyzer can vouch the rewrite stays within epsilon.
                        EMPIRICAL: rests on sampled L.
    * flagged_dangerous: regime is expansive and the bound reaches the danger
                        threshold, OR the contractive bound itself exceeds it.
    * abstained       : the bound is informative-but-inconclusive (between
                        epsilon and the danger threshold), or vacuous/overflow,
                        or the regime sits on the contraction boundary where a
                        small L error flips the verdict.
    """
    regime = classify_regime(L_pred)
    if stepwise_per_step_L is not None:
        curve = stepwise_bound_curve(delta, stepwise_per_step_L[:post_steps])
        bound_final = curve[-1]
    else:
        bound_final = delta * _safe_power(L_pred, post_steps)

    if delta <= EPS_DIV:
        # A zero-displacement rewrite is a provable no-op: bound = 0 * L^k = 0
        # for ANY L, so divergence stays 0 regardless of regime.
        return Decision("certified_safe", "zero displacement (no-op): bound is identically zero",
                        bound_final, L_pred, regime, delta)

    if is_vacuous([bound_final]) or bound_final >= 1e30 or not math.isfinite(bound_final):
        return Decision("abstained", "bound overflow / vacuous", bound_final, L_pred, regime, delta)

    if regime == "expansive":
        if bound_final >= danger_threshold:
            return Decision("flagged_dangerous", "expansive regime, bound reaches danger threshold",
                            bound_final, L_pred, regime, delta)
        return Decision("abstained", "expansive regime but bound below danger threshold (cannot certify)",
                        bound_final, L_pred, regime, delta)

    # contractive or boundary
    if bound_final >= danger_threshold:
        return Decision("flagged_dangerous", "bound reaches danger threshold", bound_final, L_pred, regime, delta)
    if bound_final * margin <= epsilon:
        if regime == "boundary":
            return Decision("abstained", "boundary regime: L within estimation noise of 1.0",
                            bound_final, L_pred, regime, delta)
        return Decision("certified_safe", "margin-inflated bound <= epsilon", bound_final, L_pred, regime, delta)
    return Decision("abstained", "bound between epsilon and danger threshold", bound_final, L_pred, regime, delta)


def estimate_L_for_state(ctx: "DiagContext", state: TrainingState,
                         margin: float = DEFAULT_SAFETY_MARGIN) -> Tuple[float, float]:
    """Estimate (L_raw, L_pred) at an arbitrary state using the diagnostic's
    own loss function and Lipschitz budget.  The state carries its own
    ``schedule_fn``, so a rewritten schedule (e.g. a changed lr) is reflected in
    the estimated dynamics."""
    loss_fn = ctx.make_loss_fn(ctx.pretrain_steps)
    return estimate_L(state, ctx.step_fn, loss_fn,
                      n_perturbations=ctx.lip_perturbations,
                      n_steps=ctx.lip_steps, eps=ctx.lip_eps, safety_margin=margin)


@dataclass
class Analysis:
    decision: Decision
    delta: float
    L_clean: float
    L_rewritten: float
    L_used: float


def repaired_analyze(ctx: "DiagContext", scenario: "ScenarioInstance",
                     L_clean_pred: float, epsilon: float, margin: float,
                     danger_threshold: float, post_steps: Optional[int] = None,
                     stepwise: bool = False,
                     per_step_L: Optional[List[float]] = None) -> Analysis:
    """The canonical repaired analyzer used across experiments.

    Uses the repaired (theta-commensurate) displacement and
    ``L = max(L_clean, L_rewritten)`` -- L is re-estimated at the rewritten
    state so that schedule/second-moment rewrites that change the *forward*
    dynamics are seen (mirrors the repo's checkpoint_lipschitz_fix finding)."""
    post_steps = post_steps if post_steps is not None else ctx.post_steps
    delta = repaired_delta(ctx.resume_state, scenario.state_after)
    _, L_rw = estimate_L_for_state(ctx, scenario.state_after, margin=margin)
    L_used = max(L_clean_pred, L_rw)
    dec = analyzer_decision(
        delta, L_used, post_steps, epsilon, margin, danger_threshold,
        stepwise_per_step_L=(per_step_L if stepwise else None))
    return Analysis(dec, delta, L_clean_pred, L_rw, L_used)


def decision_is_safe(decision: str) -> Optional[bool]:
    """Map a decision to a binary safe/dangerous verdict for ROC-style metrics.
    Abstentions return None and are excluded from sensitivity/specificity."""
    if decision == "certified_safe":
        return True
    if decision == "flagged_dangerous":
        return False
    return None


# --------------------------------------------------------------------------
# Diagnostic contexts
# --------------------------------------------------------------------------

@dataclass
class DiagContext:
    name: str
    dataset: str
    n_params: int
    resume_state: TrainingState
    checkpoints: Dict[int, Dict[str, object]]
    ema: Dict[float, torch.Tensor]
    base_schedule: Callable
    base_hp: Dict[str, float]
    step_fn: object
    make_loss_fn: Callable[[int], Callable]
    pretrain_steps: int
    post_steps: int
    gt_steps: int
    lip_perturbations: int
    lip_steps: int
    lip_eps: float
    div_threshold: float
    loss_gap_threshold: float
    device: str = "cpu"
    notes: str = ""
    extra: Dict[str, object] = field(default_factory=dict)


def _flatten_params(model) -> torch.Tensor:
    return torch.cat([p.data.reshape(-1).detach().cpu() for p in model.parameters()])


def _unflatten_params(model, flat: torch.Tensor) -> None:
    flat = flat.detach().cpu()
    idx = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(flat[idx:idx + n].reshape(p.shape).to(p.device))
        idx += n


def _pretrain_flat(model, criterion, batches, schedule, step_fn, n_params,
                   pretrain_steps, ema_decays):
    """Run a short pre-training trajectory, saving per-step checkpoints + EMA."""
    theta0 = _flatten_params(model)
    state = TrainingState(theta=theta0.clone(), moments=torch.zeros(n_params, 2),
                          schedule_fn=schedule, t=0)
    ema = {d: theta0.clone() for d in ema_decays}
    checkpoints = {0: {"theta": state.theta.clone(), "moments": state.moments.clone(), "t": 0}}
    counter = [0]

    def loss_fn(theta):
        _unflatten_params(model, theta)
        idx = counter[0] % len(batches)
        counter[0] += 1
        x, y = batches[idx]
        model.zero_grad(set_to_none=True)
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        grad = torch.cat([p.grad.reshape(-1).detach().cpu() for p in model.parameters()])
        return loss.item(), grad

    for step in range(pretrain_steps):
        checkpoints[step] = {"theta": state.theta.clone(),
                             "moments": state.moments.clone(), "t": state.t}
        _, grad = loss_fn(state.theta)
        state, _ = step_fn(state, grad)
        for d in ema_decays:
            ema[d] = d * ema[d] + (1.0 - d) * state.theta
    checkpoints[pretrain_steps] = {"theta": state.theta.clone(),
                                   "moments": state.moments.clone(), "t": state.t}
    return state.clone(), checkpoints, ema


def build_quadratic(seed: int = 42, n: int = 20, condition_number: float = 30.0,
                    lr: float = 0.05, pretrain_steps: int = 60, post_steps: int = 40,
                    gt_steps: int = 120) -> DiagContext:
    """Quadratic loss L(theta)=0.5 theta^T A theta — exact, fast diagnostic."""
    from src import make_quadratic_hessian, make_quadratic_loss
    set_seed(seed)
    A = make_quadratic_hessian(n, condition_number, seed)
    loss_fn = make_quadratic_loss(A)
    hp = dict(lr=lr, beta1=0.9, beta2=0.999, eps=1e-8)
    schedule = make_constant_schedule(**hp)
    step_fn = AdamStep()
    state = TrainingState(theta=torch.randn(n) * 0.5, moments=torch.zeros(n, 2),
                          schedule_fn=schedule, t=0)
    ema_decays = [0.9, 0.99, 0.999]
    ema = {d: state.theta.clone() for d in ema_decays}
    checkpoints = {0: {"theta": state.theta.clone(), "moments": state.moments.clone(), "t": 0}}
    for step in range(pretrain_steps):
        checkpoints[step] = {"theta": state.theta.clone(),
                             "moments": state.moments.clone(), "t": state.t}
        _, grad = loss_fn(state.theta)
        state, _ = step_fn(state, grad)
        for d in ema_decays:
            ema[d] = d * ema[d] + (1.0 - d) * state.theta
    checkpoints[pretrain_steps] = {"theta": state.theta.clone(),
                                   "moments": state.moments.clone(), "t": state.t}

    def make_loss_fn(counter_start: int = 0):
        return loss_fn  # deterministic, no batch counter

    resume = state.clone()
    div_threshold = 0.1 * (resume.theta.norm(2).item() + EPS_DIV)
    return DiagContext(
        name="quadratic", dataset=f"synthetic_quadratic(n={n},kappa={condition_number:g})",
        n_params=n, resume_state=resume, checkpoints=checkpoints, ema=ema,
        base_schedule=schedule, base_hp=hp, step_fn=step_fn, make_loss_fn=make_loss_fn,
        pretrain_steps=pretrain_steps, post_steps=post_steps, gt_steps=gt_steps,
        lip_perturbations=20, lip_steps=15, lip_eps=1e-4,
        div_threshold=div_threshold, loss_gap_threshold=0.05,
        notes=f"exact Hessian quadratic, lr={lr}, kappa={condition_number:g}",
        extra={"A": A, "eigs": torch.linalg.eigvalsh(A)})


def _load_image_batches(kind: str, seed: int, batch_size: int, n_batches: int,
                        subset: Optional[int]):
    from torchvision import datasets, transforms
    set_seed(seed)
    if kind == "mnist":
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))])
        ds = datasets.MNIST(root=str(EXPERIMENTS / "data"), train=True, download=True, transform=transform)
        note = "real MNIST train split"
    else:
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                             (0.2470, 0.2435, 0.2616))])
        ds = datasets.CIFAR10(root=str(EXPERIMENTS / "data"), train=True, download=True, transform=transform)
        note = "real CIFAR-10 train split"
    if subset is not None and subset < len(ds):
        ds = torch.utils.data.Subset(ds, list(range(subset)))
        note += f" (first {subset} images)"
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    batches = []
    for x, y in loader:
        batches.append((x, y))
        if len(batches) >= n_batches:
            break
    return batches, note


def build_mnist_mlp(seed: int = 42, hidden=(64, 32), batch_size: int = 128,
                    pretrain_steps: int = 40, post_steps: int = 30, gt_steps: int = 60,
                    lr: float = 1e-3) -> DiagContext:
    import torch.nn as nn
    set_seed(seed)
    needed = pretrain_steps + gt_steps + 30
    batches, note = _load_image_batches("mnist", seed, batch_size, needed, subset=None)

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            layers, prev = [], 784
            for h in hidden:
                layers += [nn.Linear(prev, h), nn.ReLU()]
                prev = h
            layers.append(nn.Linear(prev, 10))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x.view(x.size(0), -1))

    model = MLP()
    criterion = nn.CrossEntropyLoss()
    n_params = sum(p.numel() for p in model.parameters())
    hp = dict(lr=lr, beta1=0.9, beta2=0.999, eps=1e-8)
    schedule = make_constant_schedule(**hp)
    step_fn = AdamStep()
    ema_decays = [0.9, 0.99, 0.999]
    resume, checkpoints, ema = _pretrain_flat(model, criterion, batches, schedule,
                                              step_fn, n_params, pretrain_steps, ema_decays)

    def make_loss_fn(counter_start: int = 0):
        counter = [counter_start]

        def loss_fn(theta):
            _unflatten_params(model, theta)
            idx = counter[0] % len(batches)
            counter[0] += 1
            x, y = batches[idx]
            model.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            grad = torch.cat([p.grad.reshape(-1).detach().cpu() for p in model.parameters()])
            return loss.item(), grad
        return loss_fn

    div_threshold = 0.1 * (resume.theta.norm(2).item() + EPS_DIV)
    return DiagContext(
        name="mnist_mlp", dataset=note, n_params=n_params, resume_state=resume,
        checkpoints=checkpoints, ema=ema, base_schedule=schedule, base_hp=hp,
        step_fn=step_fn, make_loss_fn=make_loss_fn, pretrain_steps=pretrain_steps,
        post_steps=post_steps, gt_steps=gt_steps, lip_perturbations=5, lip_steps=5,
        lip_eps=1e-4, div_threshold=div_threshold, loss_gap_threshold=0.1,
        notes=f"MLP{list(hidden)} on MNIST, batch={batch_size}, lr={lr}",
        extra={"batches": batches})


def build_resnet18(seed: int = 42, batch_size: int = 16, subset: int = 256,
                   pretrain_steps: int = 6, post_steps: int = 8, gt_steps: int = 12,
                   lr: float = 1e-3) -> DiagContext:
    import torch.nn as nn
    from torchvision import models
    set_seed(seed)
    needed = pretrain_steps + gt_steps + 20
    batches, note = _load_image_batches("cifar10", seed, batch_size, needed, subset=subset)
    model = models.resnet18(num_classes=10)
    criterion = nn.CrossEntropyLoss()
    n_params = sum(p.numel() for p in model.parameters())
    hp = dict(lr=lr, beta1=0.9, beta2=0.999, eps=1e-8)
    schedule = make_constant_schedule(**hp)
    step_fn = AdamStep()
    ema_decays = [0.9, 0.99, 0.999]
    resume, checkpoints, ema = _pretrain_flat(model, criterion, batches, schedule,
                                              step_fn, n_params, pretrain_steps, ema_decays)

    def make_loss_fn(counter_start: int = 0):
        counter = [counter_start]

        def loss_fn(theta):
            _unflatten_params(model, theta)
            idx = counter[0] % len(batches)
            counter[0] += 1
            x, y = batches[idx]
            model.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            grad = torch.cat([p.grad.reshape(-1).detach().cpu() for p in model.parameters()])
            return loss.item(), grad
        return loss_fn

    div_threshold = 0.1 * (resume.theta.norm(2).item() + EPS_DIV)
    return DiagContext(
        name="resnet18", dataset=note, n_params=n_params, resume_state=resume,
        checkpoints=checkpoints, ema=ema, base_schedule=schedule, base_hp=hp,
        step_fn=step_fn, make_loss_fn=make_loss_fn, pretrain_steps=pretrain_steps,
        post_steps=post_steps, gt_steps=gt_steps, lip_perturbations=2, lip_steps=2,
        lip_eps=1e-4, div_threshold=div_threshold, loss_gap_threshold=0.1,
        notes=(f"ResNet-18 on {note}; LIMITED: subset={subset}, batch={batch_size}, "
               f"pretrain={pretrain_steps}, post={post_steps} (CPU budget)"),
        extra={"batches": batches})


def build_gpt_mini(seed: int = 42, d_model: int = 64, nhead: int = 4, num_layers: int = 2,
                   dim_feedforward: int = 128, seq_len: int = 64, batch_size: int = 8,
                   pretrain_steps: int = 6, post_steps: int = 8, gt_steps: int = 12,
                   lr: float = 1e-3) -> DiagContext:
    import torch.nn as nn
    from run_transformer import GPTMini, _download_wikitext2, build_char_dataset
    set_seed(seed)
    text = _download_wikitext2(cache_dir=str(EXPERIMENTS / "data" / "wikitext2_raw"))
    # use a slice for a small, fast, real corpus
    text = text[:200000]
    batches, vocab_size = build_char_dataset(text, seq_len, batch_size, seed=seed)
    needed = pretrain_steps + gt_steps + 20
    if len(batches) < needed:
        raise RuntimeError("not enough GPT-mini batches")
    batches = batches[:needed]
    model = GPTMini(vocab_size=vocab_size, d_model=d_model, nhead=nhead,
                    num_layers=num_layers, dim_feedforward=dim_feedforward,
                    seq_len=seq_len - 1, dropout=0.0)
    model.eval()
    criterion = nn.CrossEntropyLoss()
    n_params = sum(p.numel() for p in model.parameters())
    hp = dict(lr=lr, beta1=0.9, beta2=0.999, eps=1e-8)
    schedule = make_constant_schedule(**hp)
    step_fn = AdamStep()

    def make_loss_fn(counter_start: int = 0):
        counter = [counter_start]

        def loss_fn(theta):
            _unflatten_params(model, theta)
            idx = counter[0] % len(batches)
            counter[0] += 1
            x, y = batches[idx]
            for p in model.parameters():
                if p.grad is not None:
                    p.grad = None
            logits = model(x)
            B, Tm1, V = logits.shape
            loss = criterion(logits.reshape(B * Tm1, V), y.reshape(B * Tm1))
            loss.backward()
            grad = torch.cat([p.grad.reshape(-1).detach().cpu() for p in model.parameters()])
            return loss.item(), grad
        return loss_fn

    # pretrain
    theta0 = _flatten_params(model)
    state = TrainingState(theta=theta0.clone(), moments=torch.zeros(n_params, 2),
                          schedule_fn=schedule, t=0)
    ema_decays = [0.9, 0.99, 0.999]
    ema = {d: theta0.clone() for d in ema_decays}
    checkpoints = {0: {"theta": state.theta.clone(), "moments": state.moments.clone(), "t": 0}}
    lf = make_loss_fn(0)
    for step in range(pretrain_steps):
        checkpoints[step] = {"theta": state.theta.clone(),
                             "moments": state.moments.clone(), "t": state.t}
        _, grad = lf(state.theta)
        state, _ = step_fn(state, grad)
        for d in ema_decays:
            ema[d] = d * ema[d] + (1.0 - d) * state.theta
    checkpoints[pretrain_steps] = {"theta": state.theta.clone(),
                                   "moments": state.moments.clone(), "t": state.t}
    resume = state.clone()
    div_threshold = 0.1 * (resume.theta.norm(2).item() + EPS_DIV)
    return DiagContext(
        name="gpt_mini", dataset=f"WikiText-2 char-level (vocab={vocab_size}, first 200k chars)",
        n_params=n_params, resume_state=resume, checkpoints=checkpoints, ema=ema,
        base_schedule=schedule, base_hp=hp, step_fn=step_fn, make_loss_fn=make_loss_fn,
        pretrain_steps=pretrain_steps, post_steps=post_steps, gt_steps=gt_steps,
        lip_perturbations=2, lip_steps=2, lip_eps=1e-4,
        div_threshold=div_threshold, loss_gap_threshold=0.1,
        notes=(f"GPT-mini d_model={d_model}, layers={num_layers}, heads={nhead}; "
               f"LIMITED: seq={seq_len}, batch={batch_size}, pretrain={pretrain_steps}, "
               f"post={post_steps} (CPU budget)"),
        extra={"batches": batches, "vocab_size": vocab_size})


DIAGNOSTIC_BUILDERS: Dict[str, Callable[..., DiagContext]] = {
    "quadratic": build_quadratic,
    "mnist_mlp": build_mnist_mlp,
    "resnet18": build_resnet18,
    "gpt_mini": build_gpt_mini,
}


# --------------------------------------------------------------------------
# Scenario construction
# --------------------------------------------------------------------------

@dataclass
class ScenarioInstance:
    name: str
    family: str
    designed_benign: bool
    state_after: TrainingState
    schedule_after: Callable
    comps: Dict[str, float]


def _schedule_scaled(base_hp: Dict[str, float], factor: float) -> Callable:
    hp = dict(base_hp)
    hp["lr"] = base_hp["lr"] * factor
    return make_constant_schedule(**hp)


def build_scenarios(ctx: DiagContext) -> List[ScenarioInstance]:
    """Instantiate all 12 canonical scenarios for a diagnostic context."""
    resume = ctx.resume_state
    base_sched = ctx.base_schedule
    cps = ctx.checkpoints
    pre = ctx.pretrain_steps

    def older(step_back: int) -> Dict[str, object]:
        step = max(0, pre - step_back)
        return cps[step]

    def state_with(theta=None, moments=None, t=None, schedule=None) -> TrainingState:
        return TrainingState(
            theta=(theta if theta is not None else resume.theta).clone(),
            moments=(moments if moments is not None else resume.moments).clone(),
            schedule_fn=schedule if schedule is not None else base_sched,
            t=t if t is not None else resume.t)

    specs: List[ScenarioInstance] = []

    def add(name, family, after, schedule_after=None):
        sa = schedule_after if schedule_after is not None else base_sched
        after = TrainingState(theta=after.theta.clone(), moments=after.moments.clone(),
                              schedule_fn=sa, t=after.t)
        before = TrainingState(theta=resume.theta.clone(), moments=resume.moments.clone(),
                               schedule_fn=base_sched, t=resume.t)
        comps = component_displacements(before, after)
        specs.append(ScenarioInstance(name, family, name in DESIGNED_BENIGN, after, sa, comps))

    # 1. exact restore — restore the matching checkpoint (== current state)
    add("exact_restore", "benign", state_with())
    # 2. no-op restore — literally do nothing
    add("noop_restore", "benign", state_with())
    # 3. stale Adam moments — moments from a mildly older checkpoint
    add("stale_adam_moments", "stale_moments",
        state_with(moments=older(max(1, pre // 3))["moments"]))
    # 4. reset first moment
    m_reset = resume.moments.clone(); m_reset[:, 0] = 0.0
    add("reset_first_moment", "moment_reset", state_with(moments=m_reset))
    # 5. reset second moment
    v_reset = resume.moments.clone(); v_reset[:, 1] = 0.0
    add("reset_second_moment", "moment_reset", state_with(moments=v_reset))
    # 6. scheduler mismatch — schedule at the wrong phase (1.5x lr)
    add("scheduler_mismatch", "scheduler", state_with(schedule=_schedule_scaled(ctx.base_hp, 1.5)),
        schedule_after=_schedule_scaled(ctx.base_hp, 1.5))
    # 7. EMA shadow present but not loaded — params unchanged
    add("ema_shadow_present_not_loaded", "benign", state_with())
    # 8. EMA loaded as parameters — theta := EMA shadow
    ema_theta = ctx.ema[0.99].clone()
    add("ema_loaded_as_parameters", "ema", state_with(theta=ema_theta))
    # 9. mild lr decrease (0.5x)
    add("mild_lr_decrease", "scheduler", state_with(schedule=_schedule_scaled(ctx.base_hp, 0.5)),
        schedule_after=_schedule_scaled(ctx.base_hp, 0.5))
    # 10. severe lr increase (10x)
    add("severe_lr_increase", "scheduler", state_with(schedule=_schedule_scaled(ctx.base_hp, 10.0)),
        schedule_after=_schedule_scaled(ctx.base_hp, 10.0))
    # 11. optimizer state (moments + its time) from an older checkpoint, theta current
    old11 = older(max(1, (2 * pre) // 3))
    add("optimizer_state_from_older_checkpoint", "stale_optimizer",
        state_with(moments=old11["moments"], t=old11["t"]))
    # 12. params from current checkpoint, optimizer (moments) from an older one, current t
    old12 = older(max(1, (2 * pre) // 3))
    add("params_current_optimizer_older", "stale_optimizer",
        state_with(theta=resume.theta, moments=old12["moments"]))

    return specs


# --------------------------------------------------------------------------
# Ground-truth post-resume trajectories
# --------------------------------------------------------------------------

@dataclass
class GroundTruth:
    divergence: List[float]
    loss_gap: List[float]
    base_losses: List[float]
    scenario_losses: List[float]
    per_step_L: List[float]
    actual_max_divergence: float
    actual_final_divergence: float
    actual_loss_gap: float
    actual_label: str  # "dangerous" or "benign"


def run_ground_truth(ctx: DiagContext, scenario: ScenarioInstance,
                     steps: Optional[int] = None) -> GroundTruth:
    """Run baseline vs scenario for `steps`, deterministically replaying data."""
    steps = steps if steps is not None else ctx.gt_steps
    step_fn = ctx.step_fn
    loss_base = ctx.make_loss_fn(ctx.pretrain_steps)
    loss_scn = ctx.make_loss_fn(ctx.pretrain_steps)

    s_base = TrainingState(theta=ctx.resume_state.theta.clone(),
                           moments=ctx.resume_state.moments.clone(),
                           schedule_fn=ctx.base_schedule, t=ctx.resume_state.t)
    s_scn = TrainingState(theta=scenario.state_after.theta.clone(),
                          moments=scenario.state_after.moments.clone(),
                          schedule_fn=scenario.schedule_after, t=scenario.state_after.t)

    divergence, base_losses, scn_losses, per_step_L = [], [], [], []
    prev_div = (s_scn.theta - s_base.theta).norm(2).item()
    for _ in range(steps):
        divergence.append((s_scn.theta - s_base.theta).norm(2).item())
        lb, gb = loss_base(s_base.theta)
        ls, gs = loss_scn(s_scn.theta)
        base_losses.append(lb)
        scn_losses.append(ls)
        s_base, _ = step_fn(s_base, gb)
        s_scn, _ = step_fn(s_scn, gs)
        d_now = (s_scn.theta - s_base.theta).norm(2).item()
        if prev_div > 1e-15 and d_now > 1e-30:
            per_step_L.append(d_now / prev_div)
        else:
            per_step_L.append(1.0)
        prev_div = d_now
    divergence.append((s_scn.theta - s_base.theta).norm(2).item())

    loss_gaps = [abs(a - b) for a, b in zip(base_losses, scn_losses)]
    amax = max(divergence)
    afin = divergence[-1]
    algap = max(loss_gaps) if loss_gaps else 0.0
    label = "dangerous" if (amax >= ctx.div_threshold or algap >= ctx.loss_gap_threshold) else "benign"
    return GroundTruth(divergence, loss_gaps, base_losses, scn_losses, per_step_L,
                       amax, afin, algap, label)


def full_scenario_analysis(ctx: DiagContext, epsilon: float, margin: float,
                           seed: int) -> List[dict]:
    """Run the canonical analyzer on every scenario of a diagnostic.

    Returns one rich row per scenario, shared by experiments 2, 4, 5 and 8.
    Computes both the fixed bound (delta * L_max^k) and the empirical step-
    varying product bound (delta * prod of measured per-step ratios -- DIAGNOSTIC,
    since it peeks at the realised trajectory)."""
    from src.bounds import _safe_power
    L_raw_c, L_clean = estimate_L(
        ctx.resume_state, ctx.step_fn, ctx.make_loss_fn(ctx.pretrain_steps),
        n_perturbations=ctx.lip_perturbations, n_steps=ctx.lip_steps,
        eps=ctx.lip_eps, safety_margin=margin)
    rows = []
    for sc in build_scenarios(ctx):
        t0 = time.perf_counter()
        ana = repaired_analyze(ctx, sc, L_clean, epsilon=epsilon, margin=margin,
                               danger_threshold=ctx.div_threshold)
        gt = run_ground_truth(ctx, sc, steps=ctx.gt_steps)
        runtime = time.perf_counter() - t0

        per_step = gt.per_step_L[:ctx.post_steps]
        L_stepwise = math.exp(sum(math.log(max(r, 1e-30)) for r in per_step) / len(per_step)) if per_step else 1.0
        stepwise_curve = stepwise_bound_curve(ana.delta, per_step)
        fixed_curve = fixed_bound_curve(ana.delta, ana.L_used, ctx.post_steps)
        fixed_bound_final = fixed_curve[-1]
        bound_holds_fixed = all(a <= b + 1e-9 for a, b in zip(gt.divergence, fixed_curve))

        rows.append({
            "model": ctx.name, "dataset": ctx.dataset, "seed": seed,
            "scenario": sc.name, "family": sc.family,
            "designed_benign": sc.designed_benign,
            "delta": ana.delta, "delta_raw": metric_delta(ctx.resume_state, sc.state_after, "raw_theta_M", sc.comps),
            "L_clean": ana.L_clean, "L_rewritten": ana.L_rewritten, "L_max": ana.L_used,
            "L_stepwise": L_stepwise,
            "fixed_bound_final": fixed_bound_final,
            "bound_holds_fixed": bound_holds_fixed,
            "stepwise_bound_final": stepwise_curve[-1] if stepwise_curve else 0.0,
            "actual_max_divergence": gt.actual_max_divergence,
            "actual_final_divergence": gt.actual_final_divergence,
            "actual_loss_gap": gt.actual_loss_gap,
            "actual_label": gt.actual_label,
            "epsilon": epsilon, "safety_margin": margin,
            "decision": ana.decision.decision, "decision_reason": ana.decision.reason,
            "post_steps": ctx.post_steps, "gt_steps": ctx.gt_steps,
            "div_threshold": ctx.div_threshold, "loss_gap_threshold": ctx.loss_gap_threshold,
            "runtime_seconds": runtime,
            "comps": sc.comps,
        })
    return rows


SMALL_MODELS = {"quadratic", "mnist_mlp"}
BIG_MODELS = {"resnet18", "gpt_mini"}


def seeds_for_model(model: str, small_seeds: List[int], big_seeds: List[int]) -> List[int]:
    return small_seeds if model in SMALL_MODELS else big_seeds


# --------------------------------------------------------------------------
# IO helpers
# --------------------------------------------------------------------------

def jsonify(obj):
    if isinstance(obj, (str, int, bool)) or obj is None:
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else str(obj)
    if isinstance(obj, dict):
        return {str(k): jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonify(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return [jsonify(v) for v in obj.tolist()]
    if torch.is_tensor(obj):
        return jsonify(obj.detach().cpu().tolist())
    return str(obj)


def ensure_results_dir() -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR


def backup_if_exists(path: Path) -> Optional[Path]:
    """Back up a PRE-EXISTING (committed) output file once before overwriting.

    ``results/review_experiments/`` is created fresh by these experiments, so there are no
    prior committed results to protect; re-runs within a session intentionally
    overwrite.  We only snapshot a file that is tracked by git and would be
    clobbered, writing a single ``*.orig`` (git-ignored) -- never timestamped
    clutter.  Returns the backup path if one was written."""
    if not path.exists():
        return None
    if os.environ.get("REVIEW_EXPERIMENTS_BACKUP") == "1":
        bak = path.with_suffix(path.suffix + ".orig")
        if not bak.exists():
            import shutil
            shutil.copy2(path, bak)
            return bak
    return None


def save_csv(path: Path, rows: List[dict], fields: List[str]) -> None:
    backup_if_exists(path)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def save_json(path: Path, obj) -> None:
    backup_if_exists(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(jsonify(obj), f, indent=2)


def save_text(path: Path, text: str) -> None:
    backup_if_exists(path)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def md_table(headers: List[str], rows: List[List[object]]) -> str:
    def fmt(x):
        if isinstance(x, float):
            if not math.isfinite(x):
                return str(x)
            if x != 0 and (abs(x) < 1e-3 or abs(x) >= 1e5):
                return f"{x:.3e}"
            return f"{x:.4g}"
        return str(x)
    out = ["| " + " | ".join(headers) + " |",
           "| " + " | ".join("---" for _ in headers) + " |"]
    for r in rows:
        out.append("| " + " | ".join(fmt(c) for c in r) + " |")
    return "\n".join(out)


ANALYZER_DISCLAIMER = (
    "> **Honesty note.** All `L` values are estimated by finite sampling of "
    "perturbation directions (a priori numerical estimator). They are empirical "
    "estimates, **not** proof certificates. Decisions labelled `certified_safe` are "
    "*analyzer* decisions resting on sampled `L`; they are not formal certificates. "
    "See `lipschitz_sampling_stress.md` for the quantified gap between sampled `L` "
    "and structured worst-case directions."
)
