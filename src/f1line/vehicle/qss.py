"""Quasi-Steady-State (QSS) lap time simulator.

Implements the forward/backward speed profile method from Section 5.3
of the project spec. Fully vectorized NumPy; numba acceleration is
optional and applied automatically if installed.

Algorithm:
1. Compute cornering speed limit v_lim at each point.
2. Forward pass: accelerate from each point subject to friction ellipse.
3. Backward pass: brake from each point subject to friction ellipse.
4. Combine: v = min(forward, backward, v_lim).
5. Iterate until convergence (handles the periodic lap start/end).
6. Compute lap time = sum(2*ds / (v_i + v_{i+1})).

Internal units: m, s, m/s, m/s^2.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from f1line.vehicle.model import (
    VehicleParams,
    a_brake_max,
    a_drive_max,
    a_lat_max,
    v_corner_limit,
)

logger = logging.getLogger(__name__)

_EPS = 1e-9


@dataclass
class QSSResult:
    """Result of a QSS lap simulation.

    Attributes
    ----------
    v:
        Speed profile [m/s]. Shape (N,).
    a_long:
        Longitudinal acceleration [m/s^2]. Shape (N,).
    a_lat:
        Lateral acceleration magnitude [m/s^2]. Shape (N,).
    lap_time_s:
        Total lap time [s].
    """

    v: np.ndarray
    a_long: np.ndarray
    a_lat: np.ndarray
    lap_time_s: float


def simulate_lap(
    kappa: np.ndarray,
    ds: float | np.ndarray,
    params: VehicleParams,
    n_laps_warmup: int = 3,
    tol: float = 1e-6,
) -> QSSResult:
    """Run a QSS lap simulation on a closed circuit.

    Parameters
    ----------
    kappa:
        Signed curvature at each point [1/m]. Shape (N,).
    ds:
        Arc-length spacing [m]. Scalar or array of shape (N,).
    params:
        Vehicle model parameters.
    n_laps_warmup:
        Number of warmup laps to establish periodic steady state.
        The result is taken from the final lap.
    tol:
        Convergence tolerance on lap time [s].

    Returns
    -------
    QSSResult
    """
    kappa = np.asarray(kappa, dtype=float)
    n = len(kappa)

    if np.isscalar(ds):
        ds_arr = np.full(n, float(ds))
    else:
        ds_arr = np.asarray(ds, dtype=float)

    # Step 1: Corner speed limit
    v_lim = v_corner_limit(kappa, params)
    v_lim = np.clip(v_lim, params.v_min, params.v_max)

    # Setup periodic arrays
    v_rep = np.tile(v_lim, 3)
    kappa_rep = np.tile(kappa, 3)
    ds_rep = np.tile(ds_arr, 3)
    v_lim_rep = np.tile(v_lim, 3)

    # Call optimized loop
    v_out = _run_qss_loop(
        v_rep, kappa_rep, ds_rep, v_lim_rep,
        n_laps_warmup, params.mu, params.mu_x_brake, params.mu_x_drive,
        params.c_l, params.c_d, params.p_spec, params.a_brake_cap,
        params.g, params.v_min, params.v_max
    )

    # Take the middle lap (index n to 2n)
    v = np.clip(v_out[n : 2 * n], params.v_min, params.v_max)

    # Compute derived quantities
    v_next = np.roll(v, -1)
    a_long_arr = (v_next**2 - v**2) / (2.0 * ds_arr + _EPS)
    a_lat_arr = v**2 * np.abs(kappa)
    lap_time = float(np.sum(2.0 * ds_arr / (v + v_next + _EPS)))

    logger.debug("QSS lap time: %.3f s (%.3f min)", lap_time, lap_time / 60)

    return QSSResult(
        v=v,
        a_long=a_long_arr,
        a_lat=a_lat_arr,
        lap_time_s=lap_time,
    )


def simulate_lap_on_line(
    x: np.ndarray,
    y: np.ndarray,
    params: VehicleParams,
    n_laps_warmup: int = 3,
) -> QSSResult:
    """QSS simulation on an arbitrary (x, y) line using Menger curvature.

    Parameters
    ----------
    x, y:
        Coordinates of the racing line [m]. Shape (N,).
    params:
        Vehicle model parameters.
    n_laps_warmup:
        Warmup laps for periodic convergence.

    Returns
    -------
    QSSResult
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Arc-length spacing between consecutive points
    dx = np.roll(x, -1) - x
    dy = np.roll(y, -1) - y
    ds_arr = np.sqrt(dx**2 + dy**2)
    ds_arr = np.maximum(ds_arr, _EPS)

    kappa = menger_curvature(x, y)
    return simulate_lap(kappa, ds_arr, params, n_laps_warmup=n_laps_warmup)


def speed_profile_stages(
    kappa: np.ndarray, ds: float | np.ndarray, params: VehicleParams
) -> dict[str, np.ndarray]:
    """Intermediate speed profiles of the QSS method, for explanation/visualization.

    Returns
    -------
    dict
        ``v_corner``: cornering limit from curvature alone [m/s];
        ``v_accel``: forward (acceleration) pass from the corner limits [m/s];
        ``v_brake``: backward (braking) pass from the corner limits [m/s];
        ``v_final``: the converged QSS profile [m/s] (<= all of the above).
    """
    kappa = np.asarray(kappa, dtype=float)
    n = len(kappa)
    ds_arr = np.full(n, float(ds)) if np.isscalar(ds) else np.asarray(ds, dtype=float)
    p = params
    v_lim = np.clip(v_corner_limit(kappa, p), p.v_min, p.v_max)

    def ellipse(v: float, k: float) -> float:
        frac = (v * v * abs(k) / (a_lat_max(v, p) + _EPS)) ** 2
        return np.sqrt(max(0.0, 1.0 - frac))

    def passes(direction: int) -> np.ndarray:
        # Two laps so the periodic wrap-around is settled; keep the second.
        idx = np.arange(2 * n) % n
        if direction < 0:
            idx = idx[::-1]
        v = v_lim.copy()
        for a, b in zip(idx[:-1], idx[1:]):
            if direction > 0:
                acc = max(float(a_drive_max(v[a], p)), 0.0) * ellipse(v[a], kappa[a])
                v[b] = min(v[b], np.sqrt(v[a] ** 2 + 2 * ds_arr[a] * acc))
            else:
                dec = float(a_brake_max(v[a], p)) * ellipse(v[a], kappa[a])
                v[b] = min(v[b], np.sqrt(v[a] ** 2 + 2 * ds_arr[b] * dec))
        return v

    return {
        "v_corner": v_lim,
        "v_accel": passes(+1),
        "v_brake": passes(-1),
        "v_final": simulate_lap(kappa, ds_arr, params).v,
    }


def menger_curvature(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Signed three-point (Menger) curvature of a closed polyline [1/m] (positive = left)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x_prev = np.roll(x, 1)
    y_prev = np.roll(y, 1)
    x_next = np.roll(x, -1)
    y_next = np.roll(y, -1)

    a_vec_x = x - x_prev
    a_vec_y = y - y_prev
    b_vec_x = x_next - x
    b_vec_y = y_next - y

    cross = a_vec_x * b_vec_y - a_vec_y * b_vec_x
    norm_a = np.sqrt(a_vec_x**2 + a_vec_y**2)
    norm_b = np.sqrt(b_vec_x**2 + b_vec_y**2)
    norm_c = np.sqrt((x_next - x_prev) ** 2 + (y_next - y_prev) ** 2)

    denom = norm_a * norm_b * norm_c + _EPS
    return 2.0 * cross / denom


try:
    from numba import njit
except ImportError:
    def njit(f): return f


@njit
def _run_qss_loop(
    v_rep: np.ndarray,
    kappa_rep: np.ndarray,
    ds_rep: np.ndarray,
    v_lim_rep: np.ndarray,
    n_laps_warmup: int,
    mu: float,
    mu_x_brake: float,
    mu_x_drive: float,
    c_l: float,
    c_d: float,
    p_spec: float,
    a_brake_cap: float,
    g: float,
    v_min: float,
    v_max: float
) -> np.ndarray:
    eps = 1e-9
    v_fwd = v_rep.copy()
    v_bwd = v_rep.copy()

    for _lap in range(n_laps_warmup + 1):
        # Forward pass
        v_fwd[:] = v_rep
        for i in range(len(v_rep) - 1):
            vi = v_fwd[i]
            vi2 = vi * vi

            # alat
            alat = vi2 * abs(kappa_rep[i])
            alat_max = mu * (g + c_l * vi2)

            ellipse_frac = (alat / (alat_max + eps)) ** 2
            avail_frac = 0.0
            if ellipse_frac < 1.0:
                avail_frac = (1.0 - ellipse_frac) ** 0.5

            # drive
            ad1 = p_spec / max(vi, v_min)
            ad2 = mu_x_drive * (g + c_l * vi2)
            a_drive_raw = min(ad1, ad2) - c_d * vi2
            a_drive = max(a_drive_raw * avail_frac, 0.0)

            v_next_sq = vi2 + 2.0 * ds_rep[i] * a_drive
            v_next = min(v_next_sq**0.5 if v_next_sq > 0 else 0.0, v_lim_rep[i + 1], v_max)
            v_fwd[i + 1] = min(v_fwd[i + 1], v_next)

        # Backward pass
        v_bwd[:] = v_fwd
        for i in range(len(v_rep) - 1, 0, -1):
            vi = v_bwd[i]
            vi2 = vi * vi

            alat = vi2 * abs(kappa_rep[i])
            alat_max = mu * (g + c_l * vi2)

            ellipse_frac = (alat / (alat_max + eps)) ** 2
            avail_frac = 0.0
            if ellipse_frac < 1.0:
                avail_frac = (1.0 - ellipse_frac) ** 0.5

            ab_raw = mu_x_brake * (g + c_l * vi2) + c_d * vi2
            a_brake = min(ab_raw, a_brake_cap) * avail_frac

            v_prev_sq = vi2 + 2.0 * ds_rep[i - 1] * a_brake
            v_prev = min(v_prev_sq**0.5 if v_prev_sq > 0 else 0.0, v_lim_rep[i - 1], v_max)
            v_bwd[i - 1] = min(v_bwd[i - 1], v_prev)

        v_rep[:] = v_bwd

    return v_rep
