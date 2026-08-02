"""Ground-truth recovery tests for the phase-7 Arrhenius classifier (needs numpy)."""

from __future__ import annotations

import math
import os
import sys

import pytest

np = pytest.importorskip("numpy")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase7_arrhenius import (  # noqa: E402
    aggregate,
    diagnose,
    fit_arrhenius,
    fit_vft,
    jeffreys_rate,
    per_token_hazard,
)

TEMPS = [0.5, 0.6, 0.7, 0.85, 1.0, 1.15, 1.3]
MEAN_LEN = 400.0
N = 400  # samples per temperature


def _points_from_hazard(hazard_of_t, temps=TEMPS, n=N, length=MEAN_LEN, rng=None):
    """Build sweep points: convert a per-token hazard law to match counts."""
    pts = []
    for t in temps:
        h = hazard_of_t(t)
        p = 1.0 - math.exp(-h * length)  # survival model, inverse of the analysis
        m = int(round(p * n)) if rng is None else int(rng.binomial(n, min(max(p, 0), 1)))
        pts.append({"temperature": t, "matched": m, "samples": n, "mean_len": length})
    return pts


def test_hazard_and_jeffreys_roundtrip():
    # p -> hazard -> p is consistent through the survival model.
    for p in (0.01, 0.1, 0.5, 0.9):
        h = per_token_hazard(p, 100.0)
        assert abs((1 - math.exp(-h * 100.0)) - p) < 1e-6
    assert 0.0 < jeffreys_rate(0, 50) < 0.02  # never exactly zero
    assert 0.98 < jeffreys_rate(50, 50) < 1.0  # never exactly one


def test_arrhenius_law_is_classified_arrhenius():
    # Pure Arrhenius: h = A exp(-E/T).
    A, E = 5e-3, 2.0
    pts = _points_from_hazard(lambda t: A * math.exp(-E / t))
    r = diagnose(pts, n_boot=300, seed=1)
    assert r["verdict"] == "ARRHENIUS", r
    # Activation energy recovered from the linear fit.
    assert abs(r["E_arrhenius"] - E) < 0.3
    # Curvature CI straddles zero.
    assert r["c2_ci"][0] <= 0 <= r["c2_ci"][1]


def test_vft_law_is_classified_super_arrhenius_and_recovers_T0():
    # Vogel-Fulcher-Tammann: h = A exp(-B/(T - T0)), finite critical T0.
    # Amplitude/length chosen so the rate stays sub-saturation across the range
    # (~0.2%..76%) — the informative regime for a curvature fit.
    A, B, T0 = 0.05, 1.2, 0.35
    pts = _points_from_hazard(lambda t: A * math.exp(-B / (t - T0)), length=100.0)
    r = diagnose(pts, n_boot=300, seed=2)
    assert r["verdict"] == "SUPER-ARRHENIUS", r
    assert r["c2"] < 0 and r["c2_ci"][1] < 0  # concave, CI excludes 0
    assert r["delta_aic_vft_minus_arr"] > 2  # VFT preferred
    # T0 is only a coarse point estimate under discrete counts + Jeffreys floor
    # (the diagnosis is robust, the divergence temperature approximate): recover
    # a positive T0 in the right ballpark, safely below the coldest sample.
    assert 0 < r["T0"] < min(TEMPS)
    assert abs(r["T0"] - T0) < 0.2
    assert r["fragility_ratio"] > 1.5  # genuinely fragile


def test_fit_vft_matches_known_parameters_noise_free():
    A, B, T0 = 2.0, 0.8, 0.3
    ln_h = [math.log(A) - B / (t - T0) for t in TEMPS]
    t0, b, ln_a, sse = fit_vft(TEMPS, ln_h)
    assert abs(t0 - T0) < 0.02 and abs(b - B) < 0.05 and abs(ln_a - math.log(A)) < 0.05
    assert sse < 1e-4


def test_fit_arrhenius_recovers_slope():
    E, a = 3.5, -1.0
    inv_t = [1.0 / t for t in TEMPS]
    ln_h = [a - E * x for x in inv_t]
    e, intercept, sse = fit_arrhenius(inv_t, ln_h)
    assert abs(e - E) < 1e-6 and abs(intercept - a) < 1e-6 and sse < 1e-12


def test_noise_dominated_curve_is_inconclusive():
    # The real language-switching run: a non-monotonic zig-zag from sampling
    # noise (rate swings ~22%->3%->14%->11%->8%->17%), no fittable shape.
    rates = [0.22, 0.03, 0.14, 0.11, 0.08, 0.17]
    pts = [{"temperature": t, "matched": int(round(p * 144)), "samples": 144, "mean_len": 400.0}
           for t, p in zip(TEMPS[:6], rates)]
    r = diagnose(pts, n_boot=100)
    assert "INCONCLUSIVE" in r["verdict"] and "noise" in r["verdict"].lower()
    assert r["direction_changes"] >= 2


def test_saturated_rate_is_inconclusive_and_flagged():
    # Rate pinned at the ceiling everywhere but the coldest point: not enough
    # informative (sub-saturation) temperatures to fit a shape.
    pts = [{"temperature": t, "matched": (5 if t == 0.5 else N), "samples": N, "mean_len": MEAN_LEN}
           for t in TEMPS]
    r = diagnose(pts, n_boot=50)
    assert "INCONCLUSIVE" in r["verdict"] and "saturat" in r["verdict"]
    assert len(r["saturated_temps"]) == len(TEMPS) - 1


def test_too_few_temperatures_is_inconclusive():
    pts = _points_from_hazard(lambda t: 1e-2 * math.exp(-1 / t), temps=[0.7, 1.0, 1.3])
    r = diagnose(pts, n_boot=50)
    assert "INCONCLUSIVE" in r["verdict"]


def test_aggregate_pools_counts_and_lengths():
    rows = [
        {"behavior_id": "b", "temperature": 1.0, "n_matched": 3, "n_samples": 10, "mean_completion_tokens": 100},
        {"behavior_id": "b", "temperature": 1.0, "n_matched": 5, "n_samples": 10, "mean_completion_tokens": 200},
        {"behavior_id": "b", "temperature": 0.5, "n_matched": 0, "n_samples": 8, "mean_completion_tokens": 50},
    ]
    g = aggregate(rows, "behavior_id")["b"]
    hot = next(c for c in g if c["temperature"] == 1.0)
    assert hot["matched"] == 8 and hot["samples"] == 20
    assert abs(hot["mean_len"] - 150.0) < 1e-6  # sample-weighted
