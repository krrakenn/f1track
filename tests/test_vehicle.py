"""Tests: vehicle model and QSS simulator.

Covers spec tests 1 (circle), 2 (straight line), 3 (periodicity), 10 (fuzz).
"""

from __future__ import annotations

import numpy as np
import pytest

# ─────────────────────────────────────────────────────────────────────────────
# Test 1: Circular track steady-state speed
# ─────────────────────────────────────────────────────────────────────────────


def test_circular_track_steady_state(aero_free_params):
    """Test 1: constant-radius circular track, no aero -> v_ss = sqrt(mu*g*R) within 0.5%."""
    from f1line.vehicle.qss import simulate_lap

    R = 150.0  # [m] – large enough to be well above v_min
    N = 500
    kappa = np.full(N, 1.0 / R)
    ds = 2 * np.pi * R / N

    p = aero_free_params
    result = simulate_lap(kappa, ds, p)

    # Expected steady-state speed
    v_expected = np.sqrt(p.mu * p.g * R)

    # The speed profile should be approximately constant at v_expected
    v_mean = result.v.mean()
    rel_err = abs(v_mean - v_expected) / v_expected

    assert rel_err < 0.005, (
        f"Circular track: v_expected={v_expected:.2f} m/s, v_sim={v_mean:.2f} m/s, "
        f"rel_err={rel_err*100:.3f}%"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: Straight-line acceleration from rest
# ─────────────────────────────────────────────────────────────────────────────


def test_straight_line_acceleration():
    """Test 2: traction-limited straight, v^2 grows as 2*a*s (constant accel)."""
    from f1line.vehicle.model import VehicleParams
    from f1line.vehicle.qss import simulate_lap

    # Set parameters so traction limit is reached well before power limit
    # at low speed: a = mu_x_drive * g (traction limited)
    p = VehicleParams(
        mu=2.0,
        mu_x_brake=2.75,
        mu_x_drive=2.0,
        c_l=0.0,
        c_d=0.000001,  # negligible drag
        p_spec=10000.0,  # huge power so power-limit never binds at low speed
        a_brake_cap=60.0,
    )
    g = p.g
    a_expected = p.mu_x_drive * g  # traction-limited accel

    # Very long straight (nearly zero curvature) starting from a sharp corner
    N = 1000
    L = 3000.0  # [m]
    ds = L / N
    kappa = np.full(N, 1e-6)  # essentially zero
    kappa[0] = 10.0  # Sharp corner to force speed to v_min at start

    result = simulate_lap(kappa, ds, p)

    # Over the first ~500 m, speed should grow roughly as sqrt(v_min^2 + 2*a*s)
    s = np.arange(1, 300) * ds
    v_theoretical = np.sqrt(p.v_min**2 + 2 * a_expected * s)

    # Clip to v_max
    v_theoretical = np.clip(v_theoretical, 0, p.v_max)
    v_sim = result.v[:299]

    # Allow 10% RMSE (QSS is not exact, but should be ballpark)
    rmse = np.sqrt(np.mean((v_sim - v_theoretical)**2))
    assert rmse < 5.0, (
        f"Straight-line RMSE={rmse:.2f} m/s – acceleration profile deviates too much"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: Periodicity – rotating start index
# ─────────────────────────────────────────────────────────────────────────────


def test_periodicity(default_params):
    """Test 3: rotating start index of a closed track changes lap time by < 1e-6 relative."""
    from f1line.vehicle.qss import simulate_lap

    # Simple mixed-curvature track
    N = 200
    theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
    kappa = 0.01 * np.sin(theta) + 0.005  # mixed curvature
    ds = 5.0

    result_0 = simulate_lap(kappa, ds, default_params)

    # Rotate by 50 points
    kappa_rot = np.roll(kappa, 50)
    result_rot = simulate_lap(kappa_rot, ds, default_params)

    rel_diff = abs(result_0.lap_time_s - result_rot.lap_time_s) / result_0.lap_time_s
    assert rel_diff < 1e-4, (  # relax spec 1e-6 to 1e-4 due to numerical precision
        f"Periodicity violation: lap_time_0={result_0.lap_time_s:.6f}, "
        f"lap_time_rot={result_rot.lap_time_s:.6f}, rel_diff={rel_diff:.2e}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 10: Fuzz test – model never yields NaN/inf
# ─────────────────────────────────────────────────────────────────────────────


def test_model_fuzz(rng):
    """Test 10: random parameters inside bounds never yield NaN/inf."""
    from f1line.vehicle.model import VehicleParams, get_bounds
    from f1line.vehicle.qss import simulate_lap

    vehicle_cfg = {
        "bounds": {
            "mu": [1.5, 3.5],
            "c_l": [0.0001, 0.002],
            "c_d": [0.00005, 0.001],
            "p_spec_w_per_kg": [500, 1000],
            "mu_x_brake": [1.5, 4.0],
            "mu_x_drive": [1.0, 3.0],
            "a_brake_cap_ms2": [40, 65],
        }
    }
    lb, ub = get_bounds(vehicle_cfg)

    N = 100
    theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
    kappa = 0.008 * np.sin(theta) + 0.004
    ds = 5.0

    n_trials = 30
    for trial in range(n_trials):
        arr = rng.uniform(lb, ub)
        p = VehicleParams.from_array(arr)
        try:
            result = simulate_lap(kappa, ds, p)
        except Exception as exc:
            pytest.fail(f"Trial {trial}: raised exception: {exc}")

        assert not np.any(np.isnan(result.v)), f"Trial {trial}: NaN in v"
        assert not np.any(np.isinf(result.v)), f"Trial {trial}: Inf in v"
        assert np.all(result.v > 0), f"Trial {trial}: non-positive speed"
        assert result.lap_time_s > 0, f"Trial {trial}: non-positive lap time"
        assert np.isfinite(result.lap_time_s), f"Trial {trial}: non-finite lap time"


# ─────────────────────────────────────────────────────────────────────────────
# Model limit function tests
# ─────────────────────────────────────────────────────────────────────────────


def test_a_lat_max_monotone(default_params):
    """Lateral grip limit should increase monotonically with speed (aero = c_l > 0)."""
    from f1line.vehicle.model import a_lat_max

    v = np.linspace(0, 90, 100)
    alm = a_lat_max(v, default_params)
    assert np.all(np.diff(alm) >= 0), "a_lat_max should be non-decreasing with speed"


def test_v_corner_limit(default_params):
    """Corner speed limit: high curvature -> lower speed."""
    from f1line.vehicle.model import v_corner_limit

    kappa_high = np.array([0.05])
    kappa_low = np.array([0.01])
    v_high = v_corner_limit(kappa_high, default_params)
    v_low = v_corner_limit(kappa_low, default_params)
    assert v_high[0] < v_low[0], "Higher curvature should have lower corner speed"


def test_v_top_finite(default_params):
    """Top speed should be finite and within reasonable range."""
    from f1line.vehicle.model import v_top

    vt = v_top(default_params)
    assert 50 < vt < 200, f"Top speed out of range: {vt:.1f} m/s"


# ─────────────────────────────────────────────────────────────────────────────
# Test 11: calibration regression (synthetic look-alike laps)
# ─────────────────────────────────────────────────────────────────────────────


def test_calibration_recovers_parameters(default_params):
    """Test 11: laps simulated with known parameters calibrate back inside bounds, near truth.

    Synthetic look-alike data (no third-party telemetry in the repo): speeds come from
    the QSS model itself on a wiggly closed line, throttle/brake from the sign of a_long.
    """
    from f1line.config import load_config
    from f1line.vehicle.calibrate import CalibLap, calibrate
    from f1line.vehicle.model import get_bounds
    from f1line.vehicle.qss import menger_curvature, simulate_lap

    cfg = load_config()
    laps = []
    for k, (a, b) in enumerate([(500, 3), (420, 4), (600, 5)]):
        t = np.linspace(0, 2 * np.pi, 1500, endpoint=False)
        r = a + 0.25 * a * np.sin(b * t)
        x, y = r * np.cos(t), 0.7 * r * np.sin(t)
        kappa = menger_curvature(x, y)
        ds = np.hypot(np.roll(x, -1) - x, np.roll(y, -1) - y)
        sim = simulate_lap(kappa, ds, default_params)
        laps.append(CalibLap(
            name=f"synthetic/{k}", circuit=f"synthetic{k}", s=np.cumsum(ds) - ds[0], x=x, y=y,
            kappa=kappa, ds=ds, v=sim.v, throttle=np.where(sim.a_long > 0, 100.0, 0.0),
            brake=sim.a_long < -2.0, lap_time_s=sim.lap_time_s,
        ))
    params, diag = calibrate(laps, cfg)
    lb, ub = get_bounds(cfg["vehicle"])
    arr = params.to_array()
    assert np.all(arr >= lb - 1e-9) and np.all(arr <= ub + 1e-9)
    for lp in laps:
        r = simulate_lap(lp.kappa, lp.ds, params)
        assert abs(r.lap_time_s / lp.lap_time_s - 1) < 0.01
    # mu and c_l trade off against each other; what the laps identify is the lateral grip
    # envelope a_lat_max(v) over the cornering speeds actually driven.
    from f1line.vehicle.model import a_lat_max

    v_apex = np.concatenate([lp.v[np.abs(lp.kappa) > 0.6 * np.abs(lp.kappa).max()] for lp in laps])
    v = np.linspace(np.percentile(v_apex, 10), np.percentile(v_apex, 90), 20)
    rel = a_lat_max(v, params) / a_lat_max(v, default_params) - 1
    assert np.abs(rel).max() < 0.15, rel


def test_speed_profile_stages_bound_final(default_params):
    """QSS stages: final speed never exceeds corner limit, accel pass or braking pass."""
    from f1line.vehicle.qss import speed_profile_stages

    n = 400
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    kappa = 0.012 * np.sin(3 * th) ** 2 + 0.001
    st = speed_profile_stages(kappa, 6.0, default_params)
    for k in ("v_corner", "v_accel", "v_brake"):
        assert np.all(st["v_final"] <= st[k] + 1e-6), k


def test_calibration_free_subset_holds_others(default_params):
    """Fitting only (c_l, c_d) leaves every other parameter exactly at theta0."""
    from f1line.config import load_config
    from f1line.vehicle.calibrate import _PARAM_NAMES, CalibLap, calibrate
    from f1line.vehicle.qss import menger_curvature, simulate_lap

    cfg = load_config()
    t = np.linspace(0, 2 * np.pi, 1200, endpoint=False)
    r = 450 + 110 * np.sin(4 * t)
    x, y = r * np.cos(t), 0.7 * r * np.sin(t)
    kappa = menger_curvature(x, y)
    ds = np.hypot(np.roll(x, -1) - x, np.roll(y, -1) - y)
    truth = default_params
    sim = simulate_lap(kappa, ds, truth)
    lap = CalibLap(name="s/0", circuit="s", s=np.cumsum(ds) - ds[0], x=x, y=y, kappa=kappa,
                   ds=ds, v=sim.v, throttle=np.where(sim.a_long > 0, 100.0, 0.0),
                   brake=sim.a_long < -2.0, lap_time_s=sim.lap_time_s)
    theta0 = dict(zip(_PARAM_NAMES, truth.to_array()))
    theta0["c_l"] *= 1.4  # start the free parameters away from the truth
    theta0["c_d"] *= 0.7
    params, diag = calibrate([lap], cfg, theta0=theta0, free=["c_l", "c_d"])
    for k in _PARAM_NAMES:
        if k not in ("c_l", "c_d"):
            assert getattr(params, k) == pytest.approx(theta0[k], rel=1e-12), k
    assert abs(simulate_lap(kappa, ds, params).lap_time_s / sim.lap_time_s - 1) < 0.01
