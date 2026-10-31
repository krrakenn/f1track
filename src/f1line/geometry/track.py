"""Track dataclass: arc-length parameterized centerline with widths and normals.

Provides the canonical representation used by all downstream modules.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import Optional

import numpy as np
import pandas as pd
from scipy.interpolate import splev
from scipy.ndimage import gaussian_filter1d

from f1line.geometry.spline import (
    compute_curvature,
    fit_periodic_spline,
    resample_uniform,
    spline_length,
)

logger = logging.getLogger(__name__)


@dataclass
class Track:
    """Arc-length parameterized representation of a racing circuit.

    All coordinates and distances are in SI units (meters, radians, 1/m).

    Attributes
    ----------
    name:
        Human-readable circuit name.
    s:
        Arc length along the center line [m]. Shape (N,), ``s[0] = 0``.
    x:
        Center-line x coordinates [m]. Shape (N,).
    y:
        Center-line y coordinates [m]. Shape (N,).
    heading:
        Tangent angle [rad]. Shape (N,).
    kappa:
        Signed curvature [1/m] (positive = left turn). Shape (N,).
    nx:
        Left-pointing unit normal x component. Shape (N,).
    ny:
        Left-pointing unit normal y component. Shape (N,).
    w_left:
        Admissible car-centre offset to the left of travel direction [m]. Shape (N,).
    w_right:
        Admissible car-centre offset to the right of travel direction [m]. Shape (N,).
    ds:
        Arc-length spacing [m] (uniform).
    total_length:
        Total circuit length [m].
    tck:
        Underlying spline representation (from :func:`fit_periodic_spline`).
    u_uniform:
        Spline parameter values at the resampled points. Shape (N,).
    """

    name: str
    s: np.ndarray
    x: np.ndarray
    y: np.ndarray
    heading: np.ndarray
    kappa: np.ndarray
    nx: np.ndarray
    ny: np.ndarray
    w_left: np.ndarray
    w_right: np.ndarray
    ds: float
    total_length: float
    tck: tuple = field(repr=False)
    u_uniform: np.ndarray = field(repr=False)

    @property
    def n(self) -> int:
        """Number of discretization points."""
        return len(self.s)

    @property
    def alpha_left_max(self) -> np.ndarray:
        """Maximum left offset (positive) per point [m]."""
        return self.w_left

    @property
    def alpha_right_max(self) -> np.ndarray:
        """Maximum right offset magnitude per point [m] (the bound is ``-w_right``)."""
        return self.w_right

    def point_at(self, alpha: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute (x, y) for lateral offset *alpha* at each center-line point.

        Parameters
        ----------
        alpha:
            Signed lateral offsets [m] (positive = left). Shape (N,).

        Returns
        -------
        (px, py):
            Coordinates of the offset points [m]. Shape (N,).
        """
        px = self.x + alpha * self.nx
        py = self.y + alpha * self.ny
        return px, py

    def boundaries(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Left and right car-centre limits as (lx, ly, rx, ry) [m]."""
        lx, ly = self.point_at(self.w_left)
        rx, ry = self.point_at(-self.w_right)
        return lx, ly, rx, ry

    def rolled(self, k: int) -> "Track":
        """Return a copy whose first point is the current point ``k`` (start/finish shift)."""
        roll = lambda a: np.roll(a, -k)  # noqa: E731
        return replace(
            self,
            x=roll(self.x), y=roll(self.y), heading=roll(self.heading), kappa=roll(self.kappa),
            nx=roll(self.nx), ny=roll(self.ny), w_left=roll(self.w_left),
            w_right=roll(self.w_right), u_uniform=roll(self.u_uniform),
        )

    def with_widths(self, w_left: np.ndarray, w_right: np.ndarray) -> "Track":
        """Return a copy with new admissible offsets."""
        return replace(self, w_left=np.asarray(w_left, float), w_right=np.asarray(w_right, float))

    def resampled(self, ds_m: float) -> "Track":
        """Same track (same start point and spline) at a different uniform spacing ``ds_m``."""
        n_new = max(3, int(round(self.total_length / ds_m)))
        s_new = np.arange(n_new) * (self.total_length / n_new)
        # s -> u is monotone modulo 1 (the start may sit anywhere on the spline): unwrap it.
        u_unw = np.unwrap(self.u_uniform * 2 * np.pi) / (2 * np.pi)
        s_ext = np.append(self.s, self.total_length)
        u_ext = np.append(u_unw, u_unw[0] + 1.0)
        u_new = np.mod(np.interp(s_new, s_ext, u_ext), 1.0)
        x, y = splev(u_new, self.tck)
        kappa, heading, nx, ny = compute_curvature(self.tck, u_new)
        L = self.total_length
        return replace(
            self, s=s_new, x=np.asarray(x), y=np.asarray(y), heading=heading, kappa=kappa,
            nx=nx, ny=ny,
            w_left=np.interp(s_new, self.s, self.w_left, period=L),
            w_right=np.interp(s_new, self.s, self.w_right, period=L),
            ds=L / n_new, u_uniform=u_new,
        )


def build_track(
    name: str,
    raw_df: pd.DataFrame,
    ds_m: float = 3.0,
    smoothing_rule: str = "rms_1.0m",
    half_width_margin_m: float = 1.0,
    smoothing_factor: Optional[float] = None,
    lowpass_sigma_m: float = 0.0,
) -> Track:
    """Build a :class:`Track` from a raw TUM-style DataFrame.

    Parameters
    ----------
    name:
        Circuit name.
    raw_df:
        DataFrame with columns ``x_m, y_m, w_tr_right_m, w_tr_left_m`` (widths to the
        white lines, in driving direction).
    ds_m:
        Target arc-length spacing [m].
    smoothing_rule:
        Smoothing rule passed to :func:`fit_periodic_spline`.
    half_width_margin_m:
        Margin subtracted from each width to get the admissible car-centre offset [m].
        Negative values allow the car centre beyond the white line (track-limits rule:
        part of the car must stay on the white line).
    smoothing_factor:
        Explicit scipy smoothing factor (overrides ``smoothing_rule``).
    lowpass_sigma_m:
        If > 0, the *reference line* is the fitted center line low-pass filtered with a
        periodic Gaussian of this standard deviation [m] (sampled every 1 m, re-fitted),
        and the widths are recomputed as offsets from that reference line to the original
        track edges, so the admissible region is unchanged. This removes the polygon-
        vertex kinks of OSM-derived center lines: lines expressed as reference + offset
        (driven laps, optimizer variables) otherwise inherit them, and at 75 m/s a kink
        reads as >10 g.

    Returns
    -------
    Track
        Fully built track object.
    """
    x_raw = raw_df["x_m"].to_numpy(dtype=float)
    y_raw = raw_df["y_m"].to_numpy(dtype=float)
    w_right_raw = raw_df["w_tr_right_m"].to_numpy(dtype=float)
    w_left_raw = raw_df["w_tr_left_m"].to_numpy(dtype=float)

    logger.info("Building track %r: %d raw points", name, len(x_raw))

    tck, u_raw = fit_periodic_spline(
        x_raw, y_raw, smoothing_factor=smoothing_factor, smoothing_rule=smoothing_rule
    )
    edges = None
    if lowpass_sigma_m > 0:
        _, xd, yd, ud = resample_uniform(tck, ds_m=1.0)
        _, _, nxd, nyd = compute_curvature(tck, ud)
        wl_d = np.interp(ud, u_raw, w_left_raw, period=1.0)
        wr_d = np.interp(ud, u_raw, w_right_raw, period=1.0)
        edges = (
            np.column_stack([xd + wl_d * nxd, yd + wl_d * nyd]),
            np.column_stack([xd - wr_d * nxd, yd - wr_d * nyd]),
        )
        step = spline_length(tck) / len(xd)
        xf = gaussian_filter1d(xd, lowpass_sigma_m / step, mode="wrap")
        yf = gaussian_filter1d(yd, lowpass_sigma_m / step, mode="wrap")
        tck, u_raw = fit_periodic_spline(xf, yf, smoothing_rule="rms_0.05m")

    # Positions, curvature and normals all evaluated at the same arc-length-uniform u.
    s, x_out, y_out, u_uniform = resample_uniform(tck, ds_m=ds_m)
    n = len(s)
    kappa, heading, nx, ny = compute_curvature(tck, u_uniform)

    if edges is None:
        w_right_interp = np.interp(u_uniform, u_raw, w_right_raw, period=1.0)
        w_left_interp = np.interp(u_uniform, u_raw, w_left_raw, period=1.0)
    else:
        w_left_interp = _edge_offsets(edges[0], x_out, y_out, nx, ny, s, +1)
        w_right_interp = _edge_offsets(edges[1], x_out, y_out, nx, ny, s, -1)
    w_right_safe = np.clip(w_right_interp - half_width_margin_m, 0.1, None)
    w_left_safe = np.clip(w_left_interp - half_width_margin_m, 0.1, None)

    total_length = float(s[1] * n) if n > 1 else 0.0

    track = Track(
        name=name,
        s=s,
        x=x_out,
        y=y_out,
        heading=heading,
        kappa=kappa,
        nx=nx,
        ny=ny,
        w_left=w_left_safe,
        w_right=w_right_safe,
        ds=total_length / n,
        total_length=total_length,
        tck=tck,
        u_uniform=u_uniform,
    )

    logger.info(
        "Track %r built: N=%d, ds=%.2f m, total_length=%.1f m", name, n, track.ds, total_length
    )
    return track


def _edge_offsets(
    edge: np.ndarray, x: np.ndarray, y: np.ndarray, nx: np.ndarray, ny: np.ndarray,
    s: np.ndarray, side: int,
) -> np.ndarray:
    """Offset [m] from reference points (x, y) to an edge polyline, along the normals.

    Every edge point is assigned to its nearest reference point and its signed normal
    offset is computed; per reference point the edge offset is the median of its assigned
    points (interpolated over points that received none). ``side`` = +1 for the left
    edge (positive offsets), -1 for the right edge (returned as positive widths).
    """
    from scipy.spatial import cKDTree

    n = len(x)
    _, idx = cKDTree(np.column_stack([x, y])).query(edge, k=1)
    off = side * ((edge[:, 0] - x[idx]) * nx[idx] + (edge[:, 1] - y[idx]) * ny[idx])
    out = np.full(n, np.nan)
    order = np.argsort(idx)
    idx_s, off_s = idx[order], off[order]
    bounds = np.searchsorted(idx_s, np.arange(n + 1))
    for i in range(n):
        if bounds[i + 1] > bounds[i]:
            out[i] = np.median(off_s[bounds[i] : bounds[i + 1]])
    good = ~np.isnan(out)
    L = s[-1] + (s[1] - s[0])
    return np.interp(s, s[good], out[good], period=L)


def track_to_dataframe(track: Track) -> pd.DataFrame:
    """Serialize a :class:`Track` to a DataFrame (for inspection/export).

    Parameters
    ----------
    track:
        Track object.

    Returns
    -------
    pd.DataFrame
        Columns: s, x, y, heading, kappa, nx, ny, w_left, w_right.
    """
    return pd.DataFrame(
        {
            "s": track.s,
            "x": track.x,
            "y": track.y,
            "heading": track.heading,
            "kappa": track.kappa,
            "nx": track.nx,
            "ny": track.ny,
            "w_left": track.w_left,
            "w_right": track.w_right,
        }
    )
