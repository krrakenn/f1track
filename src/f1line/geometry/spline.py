"""Periodic smoothing spline, arc-length resampling, and analytic curvature.

All paths are treated as closed (periodic) curves parameterized by arc length.
Internal units: meters, radians, m^-1 for curvature.

Design decisions (logged in docs/DECISIONS.md):
- Use scipy.interpolate.splprep with per=True for periodic splines.
- Smoothing factor s chosen so that residual RMS ≈ 1 m (configurable).
- Curvature computed analytically from spline derivatives – never from raw samples.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from scipy.interpolate import splev, splprep

logger = logging.getLogger(__name__)

# Small epsilon to guard divisions
_EPS = 1e-12


def fit_periodic_spline(
    x: np.ndarray,
    y: np.ndarray,
    smoothing_factor: Optional[float] = None,
    smoothing_rule: str = "rms_1.0m",
    n_eval: int = 10_000,
) -> tuple:
    """Fit a periodic smoothing spline to (x, y) data.

    The curve is assumed to be closed: the last point should NOT duplicate the
    first. The spline is evaluated on a fine uniform-parameter grid for
    curvature diagnostics.

    Parameters
    ----------
    x, y:
        Coordinate arrays [m]. Shape (N,). Must form a closed loop (the
        spline automatically joins the end back to the start).
    smoothing_factor:
        Explicit scipy smoothing factor *s*. If None, estimated from
        ``smoothing_rule``.
    smoothing_rule:
        Rule for auto-selecting *s*. Currently supports ``"rms_<value>m"``
        (e.g. ``"rms_1.0m"``), which targets a residual RMS of *value* metres.
    n_eval:
        Number of points for the initial fine evaluation (for diagnostics).

    Returns
    -------
    (tck, u)
        ``tck``: spline representation as returned by :func:`splprep`.
        ``u``: parameter values at the input knots.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)

    if smoothing_factor is None:
        # Parse rule like "rms_1.0m"
        target_rms_m = 1.0
        if smoothing_rule.startswith("rms_"):
            try:
                target_rms_m = float(smoothing_rule[4:].rstrip("m"))
            except ValueError:
                logger.warning("Cannot parse smoothing_rule=%r; using rms=1.0m", smoothing_rule)
        # s ≈ n * (target_rms)^2  (per scipy convention)
        smoothing_factor = n * (target_rms_m**2)

    logger.debug("Fitting periodic spline: N=%d, s=%.3g", n, smoothing_factor)
    # With per=True, splprep ignores the last data point (it is assumed to repeat the
    # first), so close the loop explicitly and drop the duplicate's parameter afterwards.
    tck, u = splprep(
        [np.append(x, x[0]), np.append(y, y[0])], s=smoothing_factor, per=True, quiet=True
    )
    u = u[:-1]

    # Residual check
    x_fit, y_fit = splev(u, tck)
    rms = float(np.sqrt(np.mean((x - x_fit) ** 2 + (y - y_fit) ** 2)))
    logger.debug("Spline residual RMS: %.4f m", rms)

    return tck, u


def smooth_closed_path(
    x: np.ndarray, y: np.ndarray, rms_m: float, ds_m: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit a periodic smoothing spline to a noisy closed path and resample it.

    Used for telemetry-derived paths: raw samples are never differentiated, curvature
    comes from the spline. Near-duplicate samples (< 0.1 m apart) are dropped first
    because ``splprep`` rejects zero-length steps.

    Parameters
    ----------
    x, y:
        Path samples [m], in driving order, first point not repeated at the end.
    rms_m:
        Target residual RMS of the smoothing spline [m].
    ds_m:
        Output spacing [m].

    Returns
    -------
    (s, x, y, kappa, u)
        Arc length [m], coordinates [m], signed curvature [1/m], spline parameter.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    step = np.hypot(np.diff(x, append=x[0]), np.diff(y, append=y[0]))
    keep = step > 0.1
    tck, _ = fit_periodic_spline(x[keep], y[keep], smoothing_rule=f"rms_{rms_m}m")
    s, xs, ys, u = resample_uniform(tck, ds_m=ds_m)
    kappa, _, _, _ = compute_curvature(tck, u)
    return s, xs, ys, kappa, u


def resample_uniform(
    tck: tuple,
    ds_m: float = 3.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resample a periodic spline at uniformly-spaced arc-length intervals.

    Parameters
    ----------
    tck:
        Spline representation from :func:`fit_periodic_spline`.
    ds_m:
        Target arc-length spacing [m].

    Returns
    -------
    (s, x, y, u)
        ``s``: arc length of each point [m], shape (N,), ``s[0] = 0``.
        ``x``, ``y``: coordinates [m], shape (N,), equally spaced in arc length.
        ``u``: spline parameter of each point, shape (N,). Evaluate derivatives
        (curvature, normals) at exactly these values so they match ``x, y``.
    """
    total_len = spline_length(tck)
    n_points = max(3, int(round(total_len / ds_m)))
    u_out = arc_length_uniform_u(tck, n_points)
    x_out, y_out = splev(u_out, tck)
    s_out = np.arange(n_points) * (total_len / n_points)
    return s_out, np.asarray(x_out), np.asarray(y_out), u_out


def spline_length(tck: tuple, n_eval: int = 100_000) -> float:
    """Total length of a closed spline [m], from a fine polyline approximation."""
    u_fine = np.linspace(0.0, 1.0, n_eval, endpoint=False)
    x_fine, y_fine = splev(u_fine, tck)
    dx = np.diff(np.append(x_fine, x_fine[0]))
    dy = np.diff(np.append(y_fine, y_fine[0]))
    return float(np.sqrt(dx**2 + dy**2).sum())


def compute_curvature(
    tck: tuple,
    u: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute signed analytic curvature and heading at parameter values *u*.

    Positive curvature = left turn (standard mathematical convention).
    Curvature formula:  κ = (x' y'' - y' x'') / (x'^2 + y'^2)^(3/2)

    Parameters
    ----------
    tck:
        Spline representation.
    u:
        Parameter values in [0, 1).

    Returns
    -------
    (kappa, heading, nx, ny)
        ``kappa``: signed curvature [1/m], shape (N,).
        ``heading``: tangent angle [rad], shape (N,).
        ``nx``, ``ny``: left-pointing unit normal, shape (N,).
    """
    # First derivatives
    dx, dy = splev(u, tck, der=1)
    # Second derivatives
    ddx, ddy = splev(u, tck, der=2)

    speed_sq = dx**2 + dy**2
    speed = np.sqrt(speed_sq)

    kappa = (dx * ddy - dy * ddx) / (speed_sq * speed + _EPS)
    heading = np.arctan2(dy, dx)

    # Left-pointing normals: n = (-y', x') / |T|
    nx = -dy / (speed + _EPS)
    ny = dx / (speed + _EPS)

    return kappa, heading, nx, ny


def compute_curvature_at_points(
    tck: tuple,
    u_uniform: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convenience wrapper: curvature at uniformly spaced parameter values.

    See :func:`compute_curvature` for return values.
    """
    return compute_curvature(tck, u_uniform)


def arc_length_uniform_u(tck: tuple, n_points: int) -> np.ndarray:
    """Return parameter values *u* for *n_points* uniformly spaced in arc length.

    Parameters
    ----------
    tck:
        Spline representation.
    n_points:
        Number of desired output points.

    Returns
    -------
    np.ndarray
        Parameter values in [0, 1), shape (n_points,).
    """
    u_fine = np.linspace(0.0, 1.0, 100_000, endpoint=False)
    x_fine, y_fine = splev(u_fine, tck)

    dx = np.diff(np.append(x_fine, x_fine[0]))
    dy = np.diff(np.append(y_fine, y_fine[0]))
    seg = np.sqrt(dx**2 + dy**2)
    cumlen = np.concatenate([[0.0], np.cumsum(seg)])
    total = cumlen[-1]

    target = np.linspace(0.0, total, n_points, endpoint=False)
    u_out = np.interp(target, cumlen, np.append(u_fine, 1.0))
    return u_out
