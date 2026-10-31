"""Rigid registration of telemetry paths to reference geometry, and projection onto a track.

FastF1/OpenF1 positions (after /10 to get meters) live in the F1 timing frame; TUM tracks
and OpenStreetMap ways live in their own metric frames. This module estimates the 2-D
rigid transform (rotation, translation, optional reflection) between them with a trimmed
ICP started from many initial rotations, so neither the start point nor the orientation
of the inputs needs to be known. Scale is estimated as a diagnostic only (expected ~1).

Decision log: see docs/DECISIONS.md (multi-start trimmed ICP, per-side bounds check).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from f1line.geometry.track import Track

logger = logging.getLogger(__name__)

_EPS = 1e-12


@dataclass
class RigidTransform:
    """2-D rigid transform ``p_out = R @ p_in + t`` (R may contain a reflection).

    Attributes
    ----------
    R:
        2x2 orthogonal matrix.
    t:
        Translation [m], shape (2,).
    """

    R: np.ndarray
    t: np.ndarray

    @property
    def reflected(self) -> bool:
        """True if the transform mirrors the plane (det R < 0)."""
        return bool(np.linalg.det(self.R) < 0)

    @property
    def angle_rad(self) -> float:
        """Rotation angle of the proper-rotation part [rad]."""
        R = self.R @ np.diag([1.0, -1.0]) if self.reflected else self.R
        return float(np.arctan2(R[1, 0], R[0, 0]))

    def apply(self, xy: np.ndarray) -> np.ndarray:
        """Transform points, shape (M, 2)."""
        return xy @ self.R.T + self.t

    def inverse(self) -> "RigidTransform":
        """Inverse transform."""
        Rt = self.R.T
        return RigidTransform(Rt, -Rt @ self.t)

    def compose(self, first: "RigidTransform") -> "RigidTransform":
        """Return ``self ∘ first`` (apply ``first``, then ``self``)."""
        return RigidTransform(self.R @ first.R, self.R @ first.t + self.t)


@dataclass
class AlignResult:
    """Result of aligning a telemetry lap to a track.

    Attributes
    ----------
    transform:
        Transform mapping telemetry coordinates into the track frame.
    x_aligned, y_aligned:
        Telemetry positions after the transform [m].
    s_proj:
        Arc-length position of each sample on the track [m].
    alpha_proj:
        Lateral offset of each sample (positive = left) [m].
    residual_rms:
        RMS lateral offset of the samples from the center line [m].
    median_abs_offset:
        Median |alpha| [m].
    pct_within_bounds:
        Percentage of samples inside the per-side admissible offsets [%].
    direction:
        +1 if the lap runs in the track's direction, -1 if reversed.
    scale:
        Least-squares scale between matched point sets (diagnostic, expected ~1).
    """

    transform: RigidTransform
    x_aligned: np.ndarray
    y_aligned: np.ndarray
    s_proj: np.ndarray
    alpha_proj: np.ndarray
    residual_rms: float
    median_abs_offset: float
    pct_within_bounds: float
    direction: int
    scale: float

    # Backwards-compatible accessors
    @property
    def reflected(self) -> bool:
        return self.transform.reflected

    @property
    def theta_rad(self) -> float:
        return self.transform.angle_rad


def _kabsch(src: np.ndarray, dst: np.ndarray) -> RigidTransform:
    """Best proper rotation + translation mapping ``src`` onto ``dst`` (least squares)."""
    cs, cd = src.mean(axis=0), dst.mean(axis=0)
    H = (src - cs).T @ (dst - cd)
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1.0, np.sign(np.linalg.det(Vt.T @ U.T)) or 1.0])
    R = Vt.T @ D @ U.T
    return RigidTransform(R, cd - R @ cs)


def _scale_estimate(src: np.ndarray, dst: np.ndarray) -> float:
    """Umeyama-style scale between matched point sets (diagnostic only)."""
    a = src - src.mean(axis=0)
    b = dst - dst.mean(axis=0)
    return float(np.sqrt((b**2).sum() / ((a**2).sum() + _EPS)))


def rigid_register(
    src: np.ndarray,
    ref: np.ndarray,
    allow_reflection: bool = True,
    n_angles: int = 24,
    max_iter: int = 200,
    trim: float = 0.85,
    tol_m: float = 1e-3,
) -> tuple[RigidTransform, float, float]:
    """Register point set ``src`` onto ``ref`` with multi-start trimmed ICP.

    Parameters
    ----------
    src:
        Points to move, shape (M, 2) [m] (e.g. a driven lap).
    ref:
        Reference points, shape (K, 2) [m] (e.g. a densified center line or OSM ways).
        ``ref`` may contain extra geometry (pit lanes, other layouts); trimming the worst
        ``1 - trim`` fraction of matches keeps ICP robust to it.
    allow_reflection:
        Also test the mirrored ``src``.
    n_angles:
        Number of initial rotations, evenly spaced over 360 deg.
    max_iter:
        ICP iterations per start.
    trim:
        Fraction of closest matches used in each Kabsch step.
    tol_m:
        Stop when the update moves points by less than this [m].

    Returns
    -------
    (transform, median_residual_m, scale)
        ``transform`` maps ``src`` into the frame of ``ref``.
    """
    src = np.asarray(src, dtype=float)
    ref = np.asarray(ref, dtype=float)
    tree = cKDTree(ref)
    c_src, c_ref = src.mean(axis=0), ref.mean(axis=0)
    n_keep = max(3, int(trim * len(src)))

    best: tuple[RigidTransform, float] | None = None
    for flip in ([False, True] if allow_reflection else [False]):
        F = np.diag([1.0, -1.0]) if flip else np.eye(2)
        for ang in np.linspace(0.0, 2 * np.pi, n_angles, endpoint=False):
            c, s = np.cos(ang), np.sin(ang)
            Rot = np.array([[c, -s], [s, c]]) @ F
            T = RigidTransform(Rot, c_ref - Rot @ c_src)
            for _ in range(max_iter):
                moved = T.apply(src)
                d, idx = tree.query(moved, k=1)
                sel = np.argsort(d)[:n_keep]
                step = _kabsch(moved[sel], ref[idx[sel]])
                T = step.compose(T)
                if np.abs(step.apply(moved[sel]) - moved[sel]).max() < tol_m:
                    break
            d, _ = tree.query(T.apply(src), k=1)
            med = float(np.median(d))
            if best is None or med < best[1]:
                best = (T, med)

    T, med = best
    d, idx = tree.query(T.apply(src), k=1)
    sel = np.argsort(d)[:n_keep]
    scale = _scale_estimate(src[sel], ref[idx[sel]])
    logger.debug(
        "rigid_register: median residual %.2f m, angle %.1f deg, reflected=%s, scale=%.4f",
        med, np.degrees(T.angle_rad), T.reflected, scale,
    )
    return T, med, scale


def project_to_track(
    x_pts: np.ndarray, y_pts: np.ndarray, track: Track
) -> tuple[np.ndarray, np.ndarray]:
    """Project points onto the center line; return arc length and signed lateral offset.

    Each point is projected onto the two center-line segments adjacent to its nearest
    vertex and the closer projection is kept.

    Returns
    -------
    (s_proj, alpha_proj):
        Arc-length position [m] in ``[0, total_length)`` and lateral offset [m]
        (positive = left of travel direction).
    """
    pts = np.column_stack([x_pts, y_pts])
    cl = np.column_stack([track.x, track.y])
    _, idx = cKDTree(cl).query(pts, k=1)

    best_d = np.full(len(pts), np.inf)
    s_proj = np.zeros(len(pts))
    alpha = np.zeros(len(pts))
    for i0 in ((idx - 1) % track.n, idx):
        i1 = (i0 + 1) % track.n
        a, b = cl[i0], cl[i1]
        seg = b - a
        seg_len2 = (seg**2).sum(axis=1) + _EPS
        t = np.clip(((pts - a) * seg).sum(axis=1) / seg_len2, 0.0, 1.0)
        foot = a + t[:, None] * seg
        d = np.hypot(*(pts - foot).T)
        better = d < best_d
        best_d[better] = d[better]
        s_proj[better] = (track.s[i0] + t * np.sqrt(seg_len2))[better]
        # Interpolate the normal between the two vertices for a smooth sign/offset.
        nxx = (1 - t) * track.nx[i0] + t * track.nx[i1]
        nyy = (1 - t) * track.ny[i0] + t * track.ny[i1]
        al = (pts[:, 0] - foot[:, 0]) * nxx + (pts[:, 1] - foot[:, 1]) * nyy
        alpha[better] = al[better]
    return np.mod(s_proj, track.total_length), alpha


def within_bounds(s: np.ndarray, alpha: np.ndarray, track: Track) -> np.ndarray:
    """Boolean mask: is each (s, alpha) inside the per-side admissible offsets?"""
    wl = np.interp(s, track.s, track.w_left, period=track.total_length)
    wr = np.interp(s, track.s, track.w_right, period=track.total_length)
    return (alpha <= wl) & (alpha >= -wr)


def travel_direction(s: np.ndarray, total_length: float) -> int:
    """+1 if successive samples mostly advance in s, -1 if they mostly go backwards."""
    ds = np.diff(s)
    ds = (ds + total_length / 2) % total_length - total_length / 2  # unwrap the lap seam
    return 1 if np.median(ds) >= 0 else -1


def align_lap_to_track(
    x_tel: np.ndarray,
    y_tel: np.ndarray,
    track: Track,
    test_reflection: bool = True,
) -> AlignResult:
    """Align a driven lap (telemetry positions [m]) to a track and project it.

    Parameters
    ----------
    x_tel, y_tel:
        Telemetry positions in meters (after /10 conversion). Shape (M,).
    track:
        Target track.
    test_reflection:
        If True, also test a mirrored version of the lap.

    Returns
    -------
    AlignResult
    """
    src = np.column_stack([np.asarray(x_tel, float), np.asarray(y_tel, float)])
    # Densify the reference to ~1 m so point-to-point ICP approximates point-to-line.
    k = max(1, int(np.ceil(track.ds)))
    t = np.linspace(0, 1, k, endpoint=False)
    xn, yn = np.roll(track.x, -1), np.roll(track.y, -1)
    ref = np.column_stack([
        (track.x[:, None] * (1 - t) + xn[:, None] * t).ravel(),
        (track.y[:, None] * (1 - t) + yn[:, None] * t).ravel(),
    ])
    T, _, scale = rigid_register(src, ref, allow_reflection=test_reflection)
    aligned = T.apply(src)
    s_proj, alpha = project_to_track(aligned[:, 0], aligned[:, 1], track)
    inside = within_bounds(s_proj, alpha, track)
    result = AlignResult(
        transform=T,
        x_aligned=aligned[:, 0],
        y_aligned=aligned[:, 1],
        s_proj=s_proj,
        alpha_proj=alpha,
        residual_rms=float(np.sqrt(np.mean(alpha**2))),
        median_abs_offset=float(np.median(np.abs(alpha))),
        pct_within_bounds=float(inside.mean() * 100.0),
        direction=travel_direction(s_proj, track.total_length),
        scale=scale,
    )
    logger.info(
        "Aligned to %r: rms=%.2f m, median|a|=%.2f m, within=%.1f%%, reflected=%s, dir=%+d, scale=%.4f",
        track.name, result.residual_rms, result.median_abs_offset, result.pct_within_bounds,
        T.reflected, result.direction, scale,
    )
    return result
