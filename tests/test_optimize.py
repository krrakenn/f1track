"""Tests: optimization – QP bounds, monotonic improvement.

Covers spec tests 7 (QP bounds), 8 (monotonic improvement).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _make_test_track(name: str = "test_oval", n: int = 200, R_a: float = 300.0, R_b: float = 150.0):
    """Create a simple oval test track."""
    from f1line.geometry.track import build_track

    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    cx = R_a * np.cos(theta)
    cy = R_b * np.sin(theta)

    raw_df = pd.DataFrame({
        "x_m": cx,
        "y_m": cy,
        "w_tr_right_m": np.full(n, 6.0),
        "w_tr_left_m": np.full(n, 6.0),
    })
    return build_track(name, raw_df, ds_m=5.0)


# ─────────────────────────────────────────────────────────────────────────────
# Test 7: QP bounds
# ─────────────────────────────────────────────────────────────────────────────


def test_qp_bounds(default_params):
    """Test 7: min-curvature line never violates track boundaries beyond tolerance."""
    pytest.importorskip("cvxpy")

    from f1line.optimize.min_curvature import solve_min_curvature
    from f1line.optimize.utils import check_bounds

    track = _make_test_track()
    alpha = solve_min_curvature(track, lambda_reg=1e-4)

    ok, max_viol = check_bounds(alpha, track, tol=0.05)
    assert ok, f"QP boundary violated: max_violation={max_viol:.4f} m"


# ─────────────────────────────────────────────────────────────────────────────
# Test 8: Monotonic improvement
# ─────────────────────────────────────────────────────────────────────────────


def test_monotonic_improvement(default_params):
    """Test 8: min-curvature lap time < centerline lap time on a test track."""
    pytest.importorskip("cvxpy")

    from f1line.optimize.min_curvature import solve_min_curvature
    from f1line.vehicle.qss import simulate_lap_on_line

    track = _make_test_track()

    # Centerline: alpha = 0
    result_cl = simulate_lap_on_line(track.x, track.y, default_params)
    lt_centerline = result_cl.lap_time_s

    # Min-curvature line
    alpha_mc = solve_min_curvature(track, lambda_reg=1e-4)
    px, py = track.point_at(alpha_mc)
    result_mc = simulate_lap_on_line(px, py, default_params)
    lt_mincurv = result_mc.lap_time_s

    assert lt_mincurv <= lt_centerline, (
        f"Min-curvature ({lt_mincurv:.3f} s) is NOT faster than "
        f"centerline ({lt_centerline:.3f} s)"
    )


def _wiggly_track():
    """A closed track with fast and slow corners (min-time differs from min-curvature)."""
    from f1line.geometry.track import build_track

    t = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    r = 500 + 120 * np.sin(3 * t) + 40 * np.cos(5 * t)
    raw = pd.DataFrame({"x_m": r * np.cos(t), "y_m": 0.6 * r * np.sin(t),
                        "w_tr_right_m": 6.0, "w_tr_left_m": 6.0})
    return build_track("wiggly", raw, ds_m=8.0, smoothing_factor=0.0, half_width_margin_m=1.0)


def test_min_time_monotonic_and_consistent(default_params):
    """Tests 8 + 9: min-time <= min-curvature <= centerline; NLP lap time ~ QSS lap time."""
    pytest.importorskip("casadi")
    from f1line.optimize.min_curvature import solve_min_curvature
    from f1line.optimize.min_time import solve_min_time
    from f1line.optimize.utils import check_bounds
    from f1line.vehicle.qss import simulate_lap_on_line

    track = _wiggly_track()
    cfg = {"optimize": {"coarse_ds_m": 0.0, "w_reg": 1e-4,
                        "ipopt": {"max_iter": 2000, "tol": 1e-6, "print_level": 0}}}
    lt_center = simulate_lap_on_line(track.x, track.y, default_params).lap_time_s
    a_mc = solve_min_curvature(track, lambda_reg=1e-4)
    lt_mc = simulate_lap_on_line(*track.point_at(a_mc), default_params).lap_time_s
    history: list = []
    a_mt, q_mt, info = solve_min_time(track, default_params, a_mc, cfg, history=history)

    assert info["method"] == "ipopt", info
    assert q_mt.lap_time_s <= lt_mc + 1e-6 <= lt_center + 1e-6
    assert info["nlp_qss_diff_pct"] < 1.0, info
    ok, viol = check_bounds(a_mt, track, tol=0.05)
    assert ok, viol
    assert len(history) > 0 and all(len(h["alpha"]) == track.n for h in history)


def test_resample_alpha_round_trip():
    """Coarse -> fine offset resampling is smooth and preserves a smooth profile."""
    from f1line.optimize.utils import resample_alpha_to_track

    fine = _wiggly_track()
    coarse = fine.resampled(20.0)
    a_coarse = 4.0 * np.sin(2 * np.pi * 3 * coarse.s / coarse.total_length)
    a_fine = resample_alpha_to_track(a_coarse, coarse, fine)
    expected = 4.0 * np.sin(2 * np.pi * 3 * fine.s / fine.total_length)
    assert np.abs(a_fine - expected).max() < 0.05
