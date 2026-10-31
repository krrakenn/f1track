"""Build a circuit's :class:`Track` in the F1 telemetry frame from TUM or OpenStreetMap data.

Canonical frame: the F1 timing frame (telemetry X/Y divided by 10, i.e. meters). Using it
means real laps need no alignment, MultiViewer corner positions apply directly, and maps
can use the official display rotation.

Geometry sources, in order of preference:

1. **TUM racetrack-database** (measured center line and widths). The TUM layout is
   registered onto the reference driven lap; it is accepted only if the driven laps fit
   it (median |offset| and % inside bounds, see ``track.tum_match`` in the config). This
   rejects outdated layouts (e.g. Albert Park pre-2022, Yas Marina pre-2021).
2. **OpenStreetMap** ``highway=raceway`` ways. The driven lap is registered onto all ways
   in the area; every driven point is then snapped to its nearest OSM point, which yields
   the center line of the loop actually raced (pit lanes and unused layouts are skipped
   automatically). Widths are the configured nominal width.

In both cases the admissible offsets are finally widened to cover every observed car
position plus a pad, because real (legal) laps define the minimum usable width.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter1d, median_filter
from scipy.spatial import cKDTree

from f1line.geometry.align import (
    RigidTransform,
    project_to_track,
    rigid_register,
    travel_direction,
    within_bounds,
)
from f1line.geometry.spline import smooth_closed_path
from f1line.geometry.track import Track, build_track

logger = logging.getLogger(__name__)


@dataclass
class CircuitGeometry:
    """A built track plus provenance and quality diagnostics."""

    track: Track
    source: str  # "tum" | "osm"
    diagnostics: dict[str, Any] = field(default_factory=dict)


def _densify(polylines: list[np.ndarray], step_m: float) -> np.ndarray:
    """Resample polylines to points every ``step_m`` [m]; returns (K, 2)."""
    pts = []
    for pl in polylines:
        seg = np.diff(pl, axis=0)
        L = np.hypot(seg[:, 0], seg[:, 1])
        for a, d, seg_len in zip(pl[:-1], seg, L):
            k = max(1, int(np.ceil(seg_len / step_m)))
            t = np.arange(k)[:, None] / k
            pts.append(a + t * d)
        pts.append(pl[-1:])
    return np.vstack(pts)


def _reference_path(laps: list[pd.DataFrame]) -> np.ndarray:
    """Positions (M, 2) [m] of the fastest lap, in driving order."""
    df = laps[0]
    return df[["X_m", "Y_m"]].to_numpy(dtype=float)


def _tum_in_f1_frame(raw_df: pd.DataFrame, T_tum_to_f1: RigidTransform) -> pd.DataFrame:
    """Map TUM center-line points into the F1 frame (swap widths if the map mirrors)."""
    xy = T_tum_to_f1.apply(raw_df[["x_m", "y_m"]].to_numpy(dtype=float))
    out = raw_df.copy()
    out["x_m"], out["y_m"] = xy[:, 0], xy[:, 1]
    if T_tum_to_f1.reflected:
        out["w_tr_left_m"], out["w_tr_right_m"] = raw_df["w_tr_right_m"], raw_df["w_tr_left_m"]
    return out


def _orient(raw_df: pd.DataFrame, ref_xy: np.ndarray, build: Any) -> pd.DataFrame:
    """Reverse the raw point order (and swap sides) if it runs against the driven lap."""
    trk = build(raw_df)
    s, _ = project_to_track(ref_xy[:, 0], ref_xy[:, 1], trk)
    if travel_direction(s, trk.total_length) > 0:
        return raw_df
    rev = raw_df.iloc[::-1].reset_index(drop=True).copy()
    rev["w_tr_left_m"], rev["w_tr_right_m"] = (
        rev["w_tr_right_m"].to_numpy(), rev["w_tr_left_m"].to_numpy()
    )
    return rev


def _lap_offsets(track: Track, laps: list[pd.DataFrame]) -> list[tuple[np.ndarray, np.ndarray]]:
    return [
        project_to_track(df["X_m"].to_numpy(float), df["Y_m"].to_numpy(float), track)
        for df in laps
    ]


def _fit_stats(
    track: Track, offsets: list[tuple[np.ndarray, np.ndarray]], tol_m: float = 0.0
) -> dict[str, float]:
    s = np.concatenate([o[0] for o in offsets])
    a = np.concatenate([o[1] for o in offsets])
    tol_track = track.with_widths(track.w_left + tol_m, track.w_right + tol_m)
    return {
        "median_abs_offset_m": float(np.median(np.abs(a))),
        "pct_within_bounds": float(within_bounds(s, a, tol_track).mean() * 100.0),
        "max_abs_offset_m": float(np.abs(a).max()),
    }


def widen_to_observed(
    track: Track,
    offsets: list[tuple[np.ndarray, np.ndarray]],
    pad_m: float,
    window_m: float = 15.0,
    quantile: float = 1.0,
) -> tuple[Track, float]:
    """Widen admissible offsets so observed car centres (+ pad) are inside.

    Observed offsets are pooled in windows of ``window_m`` along the lap (telemetry is
    sparse: one sample every ~20 m at speed) and the ``quantile`` of each side is used,
    so a single noisy sample does not widen the track (quantile 1.0 = every sample).

    Returns
    -------
    (track, pct_points_widened)
    """
    n = track.n
    s_all = np.concatenate([o[0] for o in offsets])
    a_all = np.concatenate([o[1] for o in offsets])
    half = window_m / 2
    order = np.argsort(s_all)
    s_all, a_all = s_all[order], a_all[order]
    L = track.total_length
    s_ext = np.concatenate([s_all - L, s_all, s_all + L])
    a_ext = np.tile(a_all, 3)
    lo = np.searchsorted(s_ext, track.s - half)
    hi = np.searchsorted(s_ext, track.s + half)
    left = np.zeros(n)
    right = np.zeros(n)
    for i in range(n):
        w = a_ext[lo[i] : hi[i]]
        if len(w):
            left[i] = max(0.0, np.quantile(w, quantile))
            right[i] = max(0.0, np.quantile(-w, quantile))
    k = max(1, int(round(window_m / track.ds)) | 1)
    left = maximum_filter1d(left, size=k, mode="wrap") + pad_m
    right = maximum_filter1d(right, size=k, mode="wrap") + pad_m
    wl = np.maximum(track.w_left, left)
    wr = np.maximum(track.w_right, right)
    widened = float(((wl > track.w_left) | (wr > track.w_right)).mean() * 100.0)
    return track.with_widths(wl, wr), widened


def osm_centerline(
    polylines: list[np.ndarray],
    ref_xy: np.ndarray,
    rms_m: float,
    max_snap_m: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Center line (P, 2) [m, F1 frame] of the raced loop, from OSM ways.

    Parameters
    ----------
    polylines:
        OSM raceway ways in local metres (auxiliary ways already removed).
    ref_xy:
        Reference driven lap in the F1 frame [m], driving order.
    rms_m:
        Smoothing target of the driven-lap spline [m].
    max_snap_m:
        Driven points whose nearest OSM point is farther than this are treated as gaps.
    """
    osm_pts = _densify(polylines, step_m=1.0)
    # Register driven lap -> OSM, then express OSM in the F1 frame.
    T, med, scale = rigid_register(ref_xy, osm_pts, allow_reflection=True)
    osm_f1 = T.inverse().apply(osm_pts)

    # Smooth driven path at 2 m spacing, snap each point to the nearest OSM point.
    _, xs, ys, _, _ = smooth_closed_path(ref_xy[:, 0], ref_xy[:, 1], rms_m=1.0, ds_m=2.0)
    drv = np.column_stack([xs, ys])
    d, idx = cKDTree(osm_f1).query(drv, k=1)
    snap = osm_f1[idx] - drv
    # Median-filter the snap vectors to suppress jumps onto parallel ways at junctions.
    snap = np.column_stack([median_filter(snap[:, j], size=15, mode="wrap") for j in range(2)])
    ok = np.hypot(snap[:, 0], snap[:, 1]) <= max_snap_m
    center = drv[ok] + snap[ok]
    diag = {
        "osm_register_median_m": med,
        "osm_scale": scale,
        "osm_reflected": T.reflected,
        "pct_snapped": float(ok.mean() * 100.0),
    }
    if ok.mean() < 0.9:
        logger.warning("OSM: only %.0f%% of the driven lap snapped to OSM ways", ok.mean() * 100)
    return center, diag


def telemetry_reference(
    name: str, laps: list[pd.DataFrame], width_m: float, make: Any
) -> Track:
    """Last-resort geometry: reference line = mean of the driven laps, nominal width.

    The fastest lap is smoothed into a provisional reference, every lap is projected onto
    it, and the reference is shifted by the (smoothed) mean lateral offset. Because racing
    lines hug apexes, this "center" is biased towards the inside of corners; the observed-
    position widening afterwards keeps every real lap inside the bounds. Flagged as
    approximate in the diagnostics.
    """
    ref = _reference_path(laps)
    _, xs, ys, _, _ = smooth_closed_path(ref[:, 0], ref[:, 1], rms_m=1.0, ds_m=2.0)
    prov = make(pd.DataFrame({"x_m": xs, "y_m": ys, "w_tr_right_m": width_m / 2,
                              "w_tr_left_m": width_m / 2}), "rms_0.5m")
    offsets = _lap_offsets(prov, laps)
    s_all = np.concatenate([o[0] for o in offsets])
    a_all = np.concatenate([o[1] for o in offsets])
    order = np.argsort(s_all)
    mean_a = np.interp(prov.s, s_all[order], a_all[order], period=prov.total_length)
    mean_a = median_filter(mean_a, size=max(3, int(30 / prov.ds) | 1), mode="wrap")
    cx, cy = prov.point_at(mean_a)
    return make(pd.DataFrame({"x_m": cx, "y_m": cy, "w_tr_right_m": width_m / 2,
                              "w_tr_left_m": width_m / 2}), "rms_0.5m")


def build_circuit_geometry(
    name: str,
    laps: list[pd.DataFrame],
    cfg: dict[str, Any],
    circuit_cfg: dict[str, Any],
    tum_df: Optional[pd.DataFrame] = None,
    osm_candidates: Optional[list[tuple[str, list[np.ndarray]]]] = None,
) -> CircuitGeometry:
    """Build the circuit track in the F1 frame (see module docstring).

    Parameters
    ----------
    name:
        Circuit slug.
    laps:
        Driven laps (fastest first) with ``X_m, Y_m`` in the F1 frame [m].
    cfg:
        Full config.
    circuit_cfg:
        The circuit's entry from ``circuits.yaml``.
    tum_df:
        Raw TUM track (``x_m, y_m, w_tr_right_m, w_tr_left_m``) or None.
    osm_candidates:
        Ordered (label, polylines) candidates in local metres, e.g. OSM circuit relations
        then all non-auxiliary raceway ways. The first one the driven lap snaps to
        (>= ``osm_min_snapped_pct``) is used.
    """
    tcfg = cfg.get("track", {})
    ds = float(tcfg.get("ds_m", 3.0))
    margin = float(tcfg.get("half_width_margin_m", -1.0))
    pad = float(tcfg.get("envelope_pad_m", 0.5))
    match = tcfg.get("tum_match", {})
    ref_xy = _reference_path(laps)
    diag: dict[str, Any] = {}

    sigma = float(tcfg.get("centerline_lowpass_sigma_m", 8.0))

    def make(raw: pd.DataFrame, rule: str) -> Track:
        return build_track(
            name, raw, ds_m=ds, smoothing_rule=rule, half_width_margin_m=margin,
            lowpass_sigma_m=sigma,
        )

    geom: Optional[CircuitGeometry] = None
    if tum_df is not None:
        tum_track = make(tum_df, str(tcfg.get("spline_smoothing_rule", "rms_1.0m")))
        dense = np.column_stack([tum_track.x, tum_track.y])
        T, med, scale = rigid_register(ref_xy, dense, allow_reflection=True)
        raw_f1 = _tum_in_f1_frame(tum_df, T.inverse())
        raw_f1 = _orient(raw_f1, ref_xy, lambda r: make(r, "rms_1.0m"))
        track = make(raw_f1, str(tcfg.get("spline_smoothing_rule", "rms_1.0m")))
        stats = _fit_stats(
            track, _lap_offsets(track, laps), tol_m=float(match.get("position_tolerance_m", 1.0))
        )
        half_w = float(np.median((track.w_left + track.w_right) / 2))
        ok = (
            stats["median_abs_offset_m"]
            < float(match.get("max_median_offset_frac", 0.5)) * half_w
            and stats["pct_within_bounds"] >= float(match.get("min_pct_within", 95.0))
        )
        diag["tum"] = {**stats, "register_median_m": med, "scale": scale, "accepted": ok}
        logger.info("%s: TUM fit %s -> %s", name, stats, "accepted" if ok else "REJECTED")
        if ok:
            geom = CircuitGeometry(track, "tum")

    if geom is None:
        center, odiag = None, {}
        min_snap = float(tcfg.get("osm_min_snapped_pct", 95.0))
        for label, polylines in osm_candidates or []:
            c, d = osm_centerline(
                polylines, ref_xy,
                rms_m=float(tcfg.get("centerline_rms_m", 0.5)),
                max_snap_m=float(tcfg.get("osm_max_snap_m", 25.0)),
            )
            logger.info("%s: OSM candidate %r -> %.1f%% snapped", name, label, d["pct_snapped"])
            if center is None or d["pct_snapped"] > odiag["pct_snapped"]:
                center, odiag = c, {**d, "candidate": label}
            if d["pct_snapped"] >= min_snap:
                break
        width = float(circuit_cfg.get("nominal_width_m", tcfg.get("nominal_width_default_m", 12.0)))
        if center is not None and odiag["pct_snapped"] >= min_snap:
            raw = pd.DataFrame({
                "x_m": center[:, 0], "y_m": center[:, 1],
                "w_tr_right_m": width / 2, "w_tr_left_m": width / 2,
            })
            track = make(raw, f"rms_{tcfg.get('centerline_rms_m', 0.5)}m")
            stats = _fit_stats(track, _lap_offsets(track, laps))
            diag["osm"] = {**odiag, **stats, "nominal_width_m": width}
            logger.info("%s: OSM center line %s", name, diag["osm"])
            geom = CircuitGeometry(track, "osm")
        else:
            if odiag:
                diag["osm_rejected"] = odiag
            logger.warning(
                "%s: no usable TUM/OSM geometry (best OSM covers %.0f%%); falling back to the "
                "mean driven line with nominal width %.1f m (approximate)",
                name, odiag.get("pct_snapped", 0.0), width,
            )
            track = telemetry_reference(name, laps, width, make)
            diag["telemetry"] = {**_fit_stats(track, _lap_offsets(track, laps)),
                                 "nominal_width_m": width, "approximate": True}
            geom = CircuitGeometry(track, "telemetry")

    # Start/finish: put s = 0 where the fastest lap starts.
    track = geom.track
    s0, _ = project_to_track(ref_xy[:1, 0], ref_xy[:1, 1], track)
    track = track.rolled(int(round(s0[0] / track.ds)) % track.n)

    if tcfg.get("widen_to_observed", True):
        track, pct_widened = widen_to_observed(
            track, _lap_offsets(track, laps), pad_m=pad,
            quantile=float(tcfg.get("envelope_quantile", 1.0)),
        )
        # The reference (pole) lap must lie fully inside the track, whatever the quantile.
        track, _ = widen_to_observed(track, _lap_offsets(track, laps[:1]), pad_m=pad,
                                     quantile=1.0)
        diag["pct_points_widened"] = pct_widened
    diag["length_m"] = track.total_length
    diag["final_fit"] = _fit_stats(track, _lap_offsets(track, laps))
    geom.track = track
    geom.diagnostics = diag
    return geom
