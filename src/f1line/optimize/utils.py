"""Optimization utilities: resampling, warm starts, validity checks."""

from __future__ import annotations

import logging

import numpy as np
from scipy.interpolate import CubicSpline

from f1line.geometry.track import Track

logger = logging.getLogger(__name__)


def resample_alpha_to_track(
    alpha_src: np.ndarray,
    track_src: Track,
    track_dst: Track,
) -> np.ndarray:
    """Interpolate lateral offsets from one track resolution to another.

    Parameters
    ----------
    alpha_src:
        Lateral offsets on the source track. Shape (N_src,).
    track_src:
        Source track (e.g., coarse).
    track_dst:
        Destination track (e.g., fine).

    Returns
    -------
    np.ndarray
        Interpolated offsets on the destination track. Shape (N_dst,).
    """
    # Periodic cubic interpolation: linear interpolation from a coarse grid leaves a kink
    # at every coarse node, which reads as spurious curvature on the fine grid.
    s_src = np.append(track_src.s, track_src.total_length) / track_src.total_length
    s_dst = track_dst.s / track_dst.total_length
    spline = CubicSpline(s_src, np.append(alpha_src, alpha_src[0]), bc_type="periodic")
    return spline(s_dst)


def check_bounds(
    alpha: np.ndarray,
    track: Track,
    tol: float = 0.01,
) -> tuple[bool, float]:
    """Check that alpha values respect track boundaries within tolerance.

    Parameters
    ----------
    alpha:
        Lateral offsets [m].
    track:
        Track object.
    tol:
        Tolerance for boundary violation [m].

    Returns
    -------
    (ok, max_violation)
        ``ok``: True if no violation exceeds ``tol``.
        ``max_violation``: maximum violation [m].
    """
    viol_left = np.maximum(alpha - track.w_left, 0.0)
    viol_right = np.maximum(-alpha - track.w_right, 0.0)
    max_viol = float(np.maximum(viol_left, viol_right).max())
    return max_viol <= tol, max_viol
