"""Tests: bound computation correctness."""
import math, pytest
from src.bounds import (
    worst_case_bound, apriori_bound, product_bound,
    compositional_bound, classify_regime, _safe_power,
)


def test_safe_power_normal():
    assert abs(_safe_power(2.0, 10) - 1024.0) < 1e-6


def test_safe_power_zero_exponent():
    """L^0 = 1 for any base."""
    assert _safe_power(30.0, 0) == 1.0
    assert _safe_power(0.5, 0) == 1.0


def test_safe_power_overflow():
    assert _safe_power(30.0, 500) == 1e300


def test_safe_power_underflow():
    assert _safe_power(0.001, 1000) < 1e-10


def test_safe_power_zero_base():
    assert _safe_power(0.0, 100) == 0.0


def test_apriori_bound_zero_before_tR():
    bnd = apriori_bound(delta=1.0, L_pred=0.9, t_R=50, T=100)
    assert all(b == 0.0 for b in bnd[:50])
    assert bnd[50] == 1.0  # delta * L^0 = 1.0


def test_apriori_bound_decays_when_contractive():
    bnd = apriori_bound(delta=1.0, L_pred=0.95, t_R=0, T=100)
    for i in range(1, len(bnd)):
        assert bnd[i] <= bnd[i-1] + 1e-10


def test_apriori_bound_grows_when_expansive():
    bnd = apriori_bound(delta=1.0, L_pred=1.05, t_R=0, T=50)
    for i in range(1, len(bnd)):
        assert bnd[i] >= bnd[i-1] - 1e-10


def test_compositional_additivity():
    delta1, delta2 = 0.5, 0.3
    t1, t2, T = 10, 20, 50
    L = 0.95
    comp = compositional_bound([delta1, delta2], L, [t1, t2], T)
    ind1 = apriori_bound(delta1, L, t1, T)
    ind2 = apriori_bound(delta2, L, t2, T)
    for t in range(T + 1):
        assert abs(comp[t] - (ind1[t] + ind2[t])) < 1e-10


def test_product_bound_length():
    bnd = product_bound(delta=1.0, per_step_ratios=[0.98]*50, t_R=10, T=60)
    assert len(bnd) == 61


def test_product_bound_tracks_ratios():
    ratio = 0.95
    bnd = product_bound(delta=1.0, per_step_ratios=[ratio]*100, t_R=0, T=100)
    assert abs(bnd[50] - ratio**50) < 1e-6


def test_classify_regime():
    assert classify_regime(0.5) == "contractive"
    assert classify_regime(0.999) == "boundary"
    assert classify_regime(1.0) == "boundary"
    assert classify_regime(1.001) == "boundary"
    assert classify_regime(1.5) == "expansive"