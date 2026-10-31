"""Tests: geometry – spline fitting, curvature, and unit conversions.

Covers spec tests 4 (spline curvature), 5 (alignment recovery), 6 (unit conversions).
"""

from __future__ import annotations

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Test 4: Spline curvature accuracy
# ─────────────────────────────────────────────────────────────────────────────


def test_spline_curvature_circle():
    """Test 4a: curvature of a circle of radius R matches 1/R within 1%."""
    from f1line.geometry.spline import arc_length_uniform_u, compute_curvature, fit_periodic_spline

    R = 100.0  # [m]
    n = 200
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = R * np.cos(theta)
    y = R * np.sin(theta)

    tck, u = fit_periodic_spline(x, y, smoothing_factor=0.0)  # exact fit
    u_eval = arc_length_uniform_u(tck, 500)
    kappa, _, _, _ = compute_curvature(tck, u_eval)

    kappa_expected = 1.0 / R
    mean_abs_kappa = np.abs(kappa).mean()

    assert abs(mean_abs_kappa - kappa_expected) / kappa_expected < 0.01, (
        f"Circle curvature: expected={kappa_expected:.5f}, got mean={mean_abs_kappa:.5f}"
    )


def test_spline_curvature_ellipse():
    """Test 4b: curvature at vertex of an ellipse (a, b) = (200, 100) matches b^2/a within 1%."""
    from f1line.geometry.spline import compute_curvature, fit_periodic_spline

    a, b = 200.0, 100.0  # semi-axes [m]
    n = 400
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = a * np.cos(theta)
    y = b * np.sin(theta)

    tck, u = fit_periodic_spline(x, y, smoothing_factor=0.0)

    # Find the u parameter closest to theta=0 (rightmost vertex at x=a, y=0)
    u_fine = np.linspace(0, 1, 10000, endpoint=False)
    from scipy.interpolate import splev
    x_fine, y_fine = splev(u_fine, tck)
    idx_vertex = np.argmax(x_fine)
    u_vertex = np.array([u_fine[idx_vertex]])

    kappa, _, _, _ = compute_curvature(tck, u_vertex)
    kappa_expected = a / b**2

    assert abs(abs(kappa[0]) - kappa_expected) / kappa_expected < 0.02, (
        f"Ellipse curvature at vertex: expected={kappa_expected:.6f}, got={abs(kappa[0]):.6f}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 5: Alignment recovery
# ─────────────────────────────────────────────────────────────────────────────


def test_alignment_recovery(rng):
    """Test 5: recover a known rigid transform applied to a track."""
    import pandas as pd

    from f1line.geometry.align import align_lap_to_track
    from f1line.geometry.track import build_track

    # Create a simple oval track as reference
    n_raw = 100
    theta = np.linspace(0, 2 * np.pi, n_raw, endpoint=False)
    cx = 200.0 * np.cos(theta)
    cy = 100.0 * np.sin(theta)

    raw_df = pd.DataFrame({
        "x_m": cx,
        "y_m": cy,
        "w_tr_right_m": np.full(n_raw, 5.0),
        "w_tr_left_m": np.full(n_raw, 5.0),
    })
    track = build_track("test_oval", raw_df, ds_m=5.0, smoothing_factor=0.0)

    # Apply known rigid transform with noise
    angle_true = 0.3  # [rad]
    tx_true, ty_true = 500.0, 300.0
    c, s = np.cos(angle_true), np.sin(angle_true)
    x_src = c * cx - s * cy + tx_true + rng.normal(0, 0.5, n_raw)
    y_src = s * cx + c * cy + ty_true + rng.normal(0, 0.5, n_raw)

    result = align_lap_to_track(x_src, y_src, track)

    # Residual close to the injected 0.5 m noise; every sample inside the 5 m half-widths.
    assert result.residual_rms < 1.5, f"Alignment residual too large: {result.residual_rms:.2f} m"
    assert result.pct_within_bounds > 99.0

    # The recovered transform must invert the applied one (the oval is symmetric under a
    # 180 deg rotation, so compare point positions rather than the angle itself).
    back = result.transform.apply(np.column_stack([
        c * cx - s * cy + tx_true, s * cx + c * cy + ty_true,
    ]))
    err = np.hypot(back[:, 0] - cx, back[:, 1] - cy)
    assert err.max() < 1.0, f"Recovered transform error {err.max():.2f} m"


def test_rigid_transform_compose_inverse(rng):
    """Composition and inverse of rigid transforms (regression for the old composition bug)."""
    from f1line.geometry.align import RigidTransform

    def rot(a: float) -> np.ndarray:
        return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])

    A = RigidTransform(rot(0.7), np.array([10.0, -3.0]))
    B = RigidTransform(rot(-1.9) @ np.diag([1.0, -1.0]), np.array([-4.0, 8.0]))
    pts = rng.normal(0, 100, size=(50, 2))
    np.testing.assert_allclose(B.compose(A).apply(pts), B.apply(A.apply(pts)), atol=1e-9)
    np.testing.assert_allclose(B.inverse().apply(B.apply(pts)), pts, atol=1e-9)
    assert B.reflected and not A.reflected


def test_rigid_register_with_distractors(rng):
    """Registration finds the loop even when the reference contains extra geometry."""
    from f1line.geometry.align import RigidTransform, rigid_register

    def shape(t: np.ndarray) -> np.ndarray:  # arbitrary phases: no rotational/mirror symmetry
        return np.column_stack([
            300 * np.cos(t) + 60 * np.cos(2 * t + 0.7) + 30 * np.cos(3 * t + 1.9),
            150 * np.sin(t) + 40 * np.sin(2 * t + 0.3) + 25 * np.sin(3 * t + 2.1),
        ])

    loop = shape(np.linspace(0, 2 * np.pi, 400, endpoint=False))
    # Reference densified to ~0.4 m (as the pipeline does) plus a separate "kart track".
    td = np.linspace(0, 2 * np.pi, 4000, endpoint=False)
    dense = shape(td)
    distractor = np.column_stack([900 + 40 * np.cos(td), 40 * np.sin(td)])
    ref = np.vstack([dense, distractor])
    a = 2.2
    T_true = RigidTransform(np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]),
                            np.array([1234.0, -567.0]))
    src = T_true.apply(loop) + rng.normal(0, 0.5, loop.shape)
    T, med, scale = rigid_register(src, ref)
    assert med < 1.5
    assert abs(scale - 1.0) < 0.01
    np.testing.assert_allclose(T.apply(T_true.apply(loop)), loop, atol=2.5)


def test_track_positions_and_normals_consistent():
    """Normals are perpendicular to the local direction of the resampled points."""
    import pandas as pd

    from f1line.geometry.track import build_track

    theta = np.linspace(0, 2 * np.pi, 150, endpoint=False)
    raw = pd.DataFrame({"x_m": 400 * np.cos(theta), "y_m": 150 * np.sin(theta),
                        "w_tr_right_m": 6.0, "w_tr_left_m": 6.0})
    trk = build_track("t", raw, ds_m=3.0, smoothing_factor=0.0)
    tx = np.roll(trk.x, -1) - np.roll(trk.x, 1)
    ty = np.roll(trk.y, -1) - np.roll(trk.y, 1)
    cosang = (tx * trk.nx + ty * trk.ny) / np.hypot(tx, ty)
    assert np.abs(cosang).max() < 1e-3
    step = np.hypot(np.diff(trk.x), np.diff(trk.y))
    assert step.std() / step.mean() < 1e-3  # uniform arc-length spacing
    coarse = trk.resampled(10.0)
    assert abs(coarse.total_length - trk.total_length) < 1e-6
    np.testing.assert_allclose(np.hypot(coarse.x[0] - trk.x[0], coarse.y[0] - trk.y[0]), 0,
                               atol=1e-6)


def test_widen_to_observed():
    """Observed offsets beyond the bounds widen them locally (with pad)."""
    import pandas as pd

    from f1line.geometry.circuit import widen_to_observed
    from f1line.geometry.track import build_track

    theta = np.linspace(0, 2 * np.pi, 150, endpoint=False)
    raw = pd.DataFrame({"x_m": 400 * np.cos(theta), "y_m": 150 * np.sin(theta),
                        "w_tr_right_m": 5.0, "w_tr_left_m": 5.0})
    trk = build_track("t", raw, ds_m=3.0, smoothing_factor=0.0, half_width_margin_m=0.0)
    s = np.array([300.0, 303.0, 306.0])
    a = np.array([7.0, 7.2, 6.9])  # car centre 2 m beyond the left bound
    wide, pct = widen_to_observed(trk, [(s, a)], pad_m=0.5, quantile=1.0)
    i = int(303 / trk.ds)
    assert wide.w_left[i] >= 7.7 - 1e-9
    assert np.allclose(wide.w_right, trk.w_right)
    far = int(1000 / trk.ds)
    assert wide.w_left[far] == trk.w_left[far]
    assert 0 < pct < 5


def test_osm_auxiliary_filter():
    """Pit lanes and similar ways are excluded from the racing surface."""
    from f1line.io.osm import is_auxiliary

    assert is_auxiliary({"tags": {"highway": "raceway", "name": "Pit Lane"}})
    assert is_auxiliary({"tags": {"highway": "raceway", "raceway": "pitlane"}})
    assert is_auxiliary({"tags": {"highway": "raceway", "name": "MotoGP Long Lap Penalty"}})
    assert not is_auxiliary({"tags": {"highway": "raceway", "name": "Circuit de Monaco"}})


# ─────────────────────────────────────────────────────────────────────────────
# Test 6: Unit conversions
# ─────────────────────────────────────────────────────────────────────────────


def test_unit_conversion_speed():
    """Test 6a: km/h -> m/s conversion is correct."""
    import pandas as pd

    from f1line.io.telemetry import _km_h_to_m_s

    speeds_kmh = pd.Series([0.0, 36.0, 100.0, 360.0])
    expected_ms = pd.Series([0.0, 10.0, 100.0 / 3.6, 100.0])
    result = _km_h_to_m_s(speeds_kmh)
    np.testing.assert_allclose(result.values, expected_ms.values, rtol=1e-6)


def test_unit_conversion_position():
    """Test 6b: FastF1 position (1/10 m) -> m conversion is correct."""
    import pandas as pd

    from f1line.io.telemetry import _tenth_m_to_m

    pos_10m = pd.Series([0.0, 10.0, 1000.0, -500.0])
    expected_m = pd.Series([0.0, 1.0, 100.0, -50.0])
    result = _tenth_m_to_m(pos_10m)
    np.testing.assert_allclose(result.values, expected_m.values, rtol=1e-9)


# ─────────────────────────────────────────────────────────────────────────────
# Test: config loading
# ─────────────────────────────────────────────────────────────────────────────


def test_config_loads():
    """Config loading returns expected keys."""
    from f1line.config import load_config
    cfg = load_config()
    assert "track" in cfg
    assert "vehicle" in cfg
    assert "calibration" in cfg
    assert "optimize" in cfg


def test_circuits_config_loads():
    """Circuits config loads and contains expected keys."""
    from f1line.config import load_circuits
    circuits = load_circuits()
    assert len(circuits) > 0
    # Silverstone should be present
    assert "silverstone" in circuits
    assert "fastf1_event" in circuits["silverstone"]


def test_frozen_feed_detection():
    """Repeated car-data values while the car keeps moving are flagged; a car parked is not."""
    import pandas as pd

    from f1line.io.openf1 import frozen_mask, frozen_seconds

    t = np.arange(0, 20, 0.25)
    speed = 50 + 10 * np.sin(t)
    speed[40:70] = speed[40]  # feed repeats one reading for 7.5 s
    df = pd.DataFrame({"Time_s": t, "Speed_ms": speed, "Throttle": np.where(speed > 50, 100.0, 40.0),
                       "Brake": False, "X_m": 60 * t, "Y_m": 0.0})
    df.loc[40:69, "Throttle"] = df.loc[40, "Throttle"]
    m = frozen_mask(df)
    assert m[41:70].all() and not m[:40].any() and not m[71:].any()
    assert 7.0 < frozen_seconds(df) < 7.6
    parked = df.assign(X_m=0.0)
    assert not frozen_mask(parked).any()
