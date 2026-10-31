"""Vehicle model calibration from F1 telemetry data.

Two-step process (Section 5.6):
1. Envelope estimation: fit mu, c_l, mu_x_brake, c_d, P_spec, mu_x_drive, a_brake_cap
   from high-quantile acceleration bins.
2. Full-lap refinement: scipy.optimize.least_squares on speed trace + lap time.

Every lap is first turned into a :class:`CalibLap`: the driven X/Y path is smoothed with a
periodic spline (raw positions are never differentiated: at ~3.7 Hz they are up to 20 m
apart and raw three-point curvature implies >15 g) and speed, throttle and brake are
placed on the same grid via the circuit's track.

Outputs theta (VehicleParams) with fit diagnostics saved to models/.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.optimize import least_squares

from f1line.geometry.align import project_to_track
from f1line.geometry.spline import smooth_closed_path
from f1line.geometry.track import Track
from f1line.io.openf1 import frozen_mask
from f1line.vehicle.model import VehicleParams, get_bounds
from f1line.vehicle.qss import menger_curvature, simulate_lap

logger = logging.getLogger(__name__)

_PARAM_NAMES = ["mu", "mu_x_brake", "mu_x_drive", "c_l", "c_d", "p_spec", "a_brake_cap"]


@dataclass
class CalibLap:
    """One driven lap on a uniform arc-length grid along its own smoothed line.

    Attributes
    ----------
    name:
        Label, e.g. ``"silverstone/ANT"``.
    circuit:
        Circuit slug (for leave-one-circuit-out splits).
    s:
        Center-line arc length of each grid point [m], shape (N,).
    x, y:
        Smoothed driven line [m], shape (N,).
    kappa:
        Signed curvature of the driven line [1/m], shape (N,).
    ds:
        Driven-line segment lengths [m], shape (N,).
    v:
        Real speed [m/s] on the grid, shape (N,).
    throttle:
        Throttle [%] on the grid.
    brake:
        Brake flag on the grid.
    lap_time_s:
        Official lap time [s].
    """

    name: str
    circuit: str
    s: np.ndarray
    x: np.ndarray
    y: np.ndarray
    kappa: np.ndarray
    ds: np.ndarray
    v: np.ndarray
    throttle: np.ndarray
    brake: np.ndarray
    lap_time_s: float


def _periodic_smoothing_spline(
    s: np.ndarray, a: np.ndarray, period: float, noise_m: float, s_eval: np.ndarray
) -> np.ndarray:
    """Cubic smoothing spline a(s) with residual RMS ~ ``noise_m``, periodic in s.

    The smoothing spline is the smoothest curve (min integral of a''^2) consistent with
    the noise level, i.e. it adds the least curvature to the driven line. Periodicity is
    obtained by fitting three tiled copies and evaluating the middle one.
    """
    order = np.argsort(s)
    s, a = s[order], a[order]
    keep = np.diff(s, prepend=-np.inf) > 1e-3
    s, a = s[keep], a[keep]
    s3 = np.concatenate([s - period, s, s + period])
    a3 = np.tile(a, 3)
    spl = UnivariateSpline(s3, a3, k=3, s=len(s3) * noise_m**2)
    return spl(s_eval)


def lap_on_track(
    df: pd.DataFrame, track: Track, noise_m: float = 1.0
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Express a telemetry lap on the track grid: smoothed lateral offset + channels.

    Samples are projected onto the center line (arc length s, offset alpha). Samples
    whose projected s disagrees by > 100 m with the integrated ``Distance`` (projection
    onto the wrong branch, e.g. Suzuka's crossover) are dropped. alpha(s) is fitted with a
    periodic cubic smoothing spline (residual ~ ``noise_m``, the position accuracy) and
    the driven line is ``track.point_at(alpha)``. At ~3.7 Hz an apex is defined by 2-3
    samples; linear interpolation would turn it into a V and add spurious curvature.
    Using the center-line frame also gives driven and optimized lines the same
    parameterization.

    Returns
    -------
    (alpha, channels)
        ``alpha`` [m] on ``track.s``; ``channels``: Speed_ms, Throttle, Brake, Time_s.
    """
    x = df["X_m"].to_numpy(float)
    y = df["Y_m"].to_numpy(float)
    L = track.total_length
    s_proj, a_proj = project_to_track(x, y, track)

    # Reject samples projected onto the wrong branch (Suzuka's crossover): compare each
    # sample's along-lap position with the running median of its neighbours. (An earlier
    # prior from the speed-integrated Distance channel drifted by >300 m at Suzuka and
    # discarded 64 % of good samples.)
    step = np.diff(s_proj, prepend=s_proj[0])
    s_unw = s_proj + L * np.cumsum(np.where(step < -L / 2, 1, 0) - np.where(step > L / 2, 1, 0))
    s_med = median_filter(s_unw, size=9, mode="nearest")
    ok = np.abs(s_unw - s_med) < 60.0
    if ok.mean() < 0.9:
        logger.warning("Only %.0f%% of samples project consistently", ok.mean() * 100)
    order = np.argsort(s_proj[ok])
    s_o = s_proj[ok][order]

    def on_grid(vals: np.ndarray) -> np.ndarray:
        return np.interp(track.s, s_o, np.asarray(vals, float)[ok][order], period=L)

    alpha = _periodic_smoothing_spline(s_proj[ok], a_proj[ok], L, noise_m, track.s)

    # Frozen car-data stretches: speed from the (still moving) position stream instead,
    # pedals unknown (NaN on the grid points the stretch covers).
    speed = df["Speed_ms"].to_numpy(float).copy()
    frozen = frozen_mask(df)
    pedal_gap = np.zeros(track.n, dtype=bool)
    if frozen.any():
        t = df["Time_s"].to_numpy(float)
        # local straight-line fit of distance vs time over +-1.5 s (robust to the ~4 Hz
        # position noise, and one-sided at the lap ends where a spline misbehaves)
        # capped at 3 % above the lap's highest genuine reading: position noise must not
        # invent top speed (Suzuka: 360 km/h vs a 328 km/h official speed trap)
        v_cap = 1.03 * float(np.max(speed[~frozen]))
        for i in np.where(frozen)[0]:
            w = ok & (np.abs(t - t[i]) <= 1.5)
            if w.sum() >= 3:
                speed[i] = np.clip(np.polyfit(t[w], s_unw[w], 1)[0], 5.0, v_cap)
        # mark the grid between consecutive frozen samples (walking forward, across the
        # start line if needed)
        n = track.n
        idx = np.round(s_proj[frozen] / L * n).astype(int) % n
        for a, b in zip(idx[:-1], idx[1:]):
            span = (b - a) % n
            if span < n // 4:
                pedal_gap[(a + np.arange(span + 1)) % n] = True
        logger.info("Frozen car data for %.1f s: speed from positions, pedals unknown",
                    float(np.sum(np.diff(t, append=t[-1])[frozen])))
    throttle = on_grid(df["Throttle"])
    brake = on_grid(df["Brake"].astype(float))
    throttle[pedal_gap] = np.nan
    brake[pedal_gap] = np.nan
    channels = {
        "Speed_ms": on_grid(speed),
        "Throttle": throttle,
        "Brake": np.where(np.isnan(brake), np.nan, brake > 0.5),
        "Time_s": on_grid(df["Time_s"]),
        "Z_m": on_grid(df["Z_m"]) if "Z_m" in df.columns else np.zeros(track.n),
    }
    return alpha, channels


def prepare_lap(
    df: pd.DataFrame, name: str, circuit: str, track: Track, rms_m: float = 1.0
) -> CalibLap:
    """Smooth a telemetry lap for calibration and put its channels on its own line's grid.

    The driven path is a periodic smoothing spline through the raw positions (residual
    ``rms_m``); curvature is analytic. This reproduces the lateral accelerations of
    local circle fits to the raw samples (Silverstone 2026: peaks 5-6.5 g), whereas
    reconstructing the path as center line + smoothed offset inherits reference-line
    bends between the sparse samples and reads 9-15 g in fast corners (DECISIONS.md).
    ``track`` is used only to place the channels: each grid point of the smoothed line is
    projected onto the track and the channels (already on the track grid via
    :func:`lap_on_track`) are interpolated there.
    """
    x = df["X_m"].to_numpy(float)
    y = df["Y_m"].to_numpy(float)
    _, xs, ys, kappa, _ = smooth_closed_path(x, y, rms_m=rms_m, ds_m=track.ds)
    _, ch = lap_on_track(df, track)
    s_line, _ = project_to_track(xs, ys, track)
    L = track.total_length

    def at_line(vals: np.ndarray) -> np.ndarray:
        return np.interp(s_line, track.s, np.asarray(vals, float), period=L)

    ds = np.hypot(np.roll(xs, -1) - xs, np.roll(ys, -1) - ys)
    return CalibLap(
        name=name, circuit=circuit, s=s_line, x=xs, y=ys, kappa=kappa, ds=ds,
        v=at_line(ch["Speed_ms"]), throttle=at_line(ch["Throttle"]),
        brake=at_line(ch["Brake"]) > 0.5,
        lap_time_s=float(df["Official_LapTime_s"].iloc[0]),
    )


def prepare_lap_on_reference(
    df: pd.DataFrame, name: str, circuit: str, track: Track, alpha_ref: np.ndarray
) -> CalibLap:
    """Calibration lap whose *geometry* is a noise-free reference line on ``track``.

    The real speed/throttle/brake traces are kept, but curvature comes from
    ``track.point_at(alpha_ref)`` (e.g. the min-curvature line) instead of the
    reconstructed driven path. Rationale (DECISIONS.md): at ~3.7 Hz the driven path's
    curvature carries noise that creates spurious slow spots in the replay; the fit then
    raises grip to recover the lap time, and noise-free optimized lines become ~10 %
    faster than pole. Fitting speeds on a smooth line removes that bias.
    """
    _, ch = lap_on_track(df, track)
    px, py = track.point_at(alpha_ref)
    ds = np.hypot(np.roll(px, -1) - px, np.roll(py, -1) - py)
    return CalibLap(
        name=name, circuit=circuit, s=track.s, x=px, y=py, kappa=menger_curvature(px, py),
        ds=ds, v=ch["Speed_ms"], throttle=ch["Throttle"], brake=ch["Brake"],
        lap_time_s=float(df["Official_LapTime_s"].iloc[0]),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Envelope estimation
# ─────────────────────────────────────────────────────────────────────────────


def _binned_quantile(v: np.ndarray, a: np.ndarray, bin_ms: float, q: float, min_n: int):
    edges = np.arange(0.0, v.max() + bin_ms, bin_ms)
    vb, ab = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (v >= lo) & (v < hi)
        if m.sum() >= min_n:
            vb.append(0.5 * (lo + hi))
            ab.append(np.quantile(a[m], q))
    return np.array(vb), np.array(ab)


def envelope_points(laps: list[CalibLap], cfg: dict[str, Any]) -> dict[str, np.ndarray]:
    """Observed accelerations and their binned high-quantile envelopes (for fits and plots)."""
    calib = cfg.get("calibration", {})
    q = float(calib.get("envelope_quantile", 0.98))
    bin_ms = float(calib.get("speed_bin_kmh", 10.0)) / 3.6
    v = np.concatenate([lp.v for lp in laps])
    a_lat = np.concatenate([lp.v**2 * np.abs(lp.kappa) for lp in laps])
    a_long = np.concatenate([
        np.gradient(gaussian_filter1d(lp.v**2 / 2, 2.0, mode="wrap"), np.cumsum(lp.ds))
        for lp in laps
    ])
    thr = np.concatenate([lp.throttle for lp in laps])
    brk = np.concatenate([lp.brake for lp in laps])

    out = {"v": v, "a_lat": a_lat, "a_long": a_long, "throttle": thr, "brake": brk}
    out["lat_v"], out["lat_a"] = _binned_quantile(v, a_lat, bin_ms, q, 5)
    bm = brk & (thr < 10.0)
    out["brk_v"], out["brk_a"] = _binned_quantile(v[bm], -a_long[bm], bin_ms, q, 5)
    dm = (thr >= 98.0) & (a_lat < 5.0)
    out["drv_v"], out["drv_a"] = _binned_quantile(v[dm], a_long[dm], bin_ms, q, 3)
    return out


def estimate_envelope(laps: list[CalibLap], cfg: dict[str, Any]) -> dict[str, float]:
    """Step 1: initial parameter estimates from acceleration envelopes.

    - Lateral: ``a_lat_max = mu*g + mu*c_l*v^2`` is linear in v^2 -> robust line fit.
    - Braking (brake on, throttle < 10 %): ``mu_x_brake*g + (mu_x_brake*c_l + c_d)*v^2``,
      ``a_brake_cap`` = highest binned deceleration.
    - Drive (throttle >= 98 %, |a_lat| < 5): power regime ``P/v - c_d v^2`` at high speed,
      traction regime at low speed.

    Returns
    -------
    dict
        Estimates keyed like :class:`VehicleParams`, clipped to the config bounds.
    """
    g = float(cfg.get("vehicle", {}).get("g", 9.81))
    lb, ub = get_bounds(cfg.get("vehicle", {}))
    env = envelope_points(laps, cfg)
    est = dict(zip(_PARAM_NAMES, (lb + ub) / 2))

    if len(env["lat_v"]) >= 3:
        X = np.column_stack([np.ones_like(env["lat_v"]), env["lat_v"] ** 2])
        c, *_ = np.linalg.lstsq(X, env["lat_a"], rcond=None)
        est["mu"] = max(c[0], 0.1) / g
        est["c_l"] = max(c[1], 1e-7) / est["mu"]

    if len(env["brk_v"]) >= 3:
        X = np.column_stack([np.ones_like(env["brk_v"]), env["brk_v"] ** 2])
        c, *_ = np.linalg.lstsq(X, env["brk_a"], rcond=None)
        est["mu_x_brake"] = max(c[0], 0.1) / g
        est["c_d"] = max(c[1] - est["mu_x_brake"] * est["c_l"], 1e-6)
        est["a_brake_cap"] = float(env["brk_a"].max())

    if len(env["drv_v"]) >= 3:
        gross = env["drv_a"] + est["c_d"] * env["drv_v"] ** 2
        hi = env["drv_v"] > 0.6 * env["drv_v"].max()
        lo = env["drv_v"] < 0.4 * env["drv_v"].max()
        if hi.sum() >= 2:
            est["p_spec"] = float(np.median(gross[hi] * env["drv_v"][hi]))
        if lo.sum() >= 2:
            est["mu_x_drive"] = float(
                np.median(gross[lo] / (g + est["c_l"] * env["drv_v"][lo] ** 2))
            )

    raw = dict(est)
    arr = np.clip([est[k] for k in _PARAM_NAMES], lb, ub)
    est = {k: float(v) for k, v in zip(_PARAM_NAMES, arr)}
    logger.info("Envelope estimates (raw): %s", {k: round(v, 5) for k, v in raw.items()})
    logger.info("Envelope estimates (clipped to bounds): %s", est)
    return est


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Full-lap refinement
# ─────────────────────────────────────────────────────────────────────────────


def _residuals(
    theta_arr: np.ndarray,
    laps: list[CalibLap],
    speed_weight: float,
    time_weight: float,
    reg_weight: float,
    theta0: np.ndarray,
    scale: np.ndarray,
    stride: int,
    fixed: dict[str, float],
) -> np.ndarray:
    """Residuals: speed trace [m/s], lap time [s], and regularization toward Step 1."""
    p = VehicleParams.from_array(theta_arr, **fixed)
    res = []
    for lp in laps:
        sim = simulate_lap(lp.kappa, lp.ds, p)
        res.append(speed_weight * (sim.v[::stride] - lp.v[::stride]))
        res.append([time_weight * (sim.lap_time_s - lp.lap_time_s)])
    res.append(reg_weight * (theta_arr - theta0) / scale)
    return np.concatenate(res)


def _residuals_subset(x: np.ndarray, mask: np.ndarray, base: np.ndarray, *args: Any) -> np.ndarray:
    """:func:`_residuals` with only the ``mask`` parameters free (others held at ``base``)."""
    full = base.copy()
    full[mask] = x
    return _residuals(full, *args)


def calibrate(
    laps: list[CalibLap],
    cfg: dict[str, Any],
    theta0: Optional[dict[str, float]] = None,
    free: Optional[list[str]] = None,
) -> tuple[VehicleParams, dict[str, Any]]:
    """Calibrate vehicle parameters on a set of prepared laps.

    Parameters
    ----------
    laps:
        Prepared calibration laps (see :func:`prepare_lap`).
    cfg:
        Full config dict (uses 'vehicle' and 'calibration' sub-dicts).
    theta0:
        Optional Step 1 estimates (if None, runs Step 1 first). With ``free``, these are
        also the values of the parameters held fixed and the regularization target.
    free:
        Names of the parameters to fit (default: all). Used to refit a weekend's car
        setup (downforce, drag) around the season car.

    Returns
    -------
    (VehicleParams, diagnostics_dict)
    """
    vehicle_cfg = cfg.get("vehicle", {})
    calib_cfg = cfg.get("calibration", {})
    fixed = {
        "g": float(vehicle_cfg.get("g", 9.81)),
        "v_min": float(vehicle_cfg.get("v_min_ms", 5.0)),
        "v_max": float(vehicle_cfg.get("v_max_ms", 100.0)),
    }
    lb, ub = get_bounds(vehicle_cfg)

    if theta0 is None:
        logger.info("Step 1: envelope estimation from %d laps", len(laps))
        theta0 = estimate_envelope(laps, cfg)
    theta0_arr = np.clip(np.array([theta0[k] for k in _PARAM_NAMES]), lb, ub)
    scale = ub - lb

    w = calib_cfg.get("weights", {})
    args = (
        laps,
        float(w.get("speed", 1.0)),
        float(w.get("lap_time", 3.0)),
        float(w.get("regularization", 2.0)),
        theta0_arr,
        scale,
        int(calib_cfg.get("speed_stride", 3)),
        fixed,
    )
    mask = np.array([free is None or k in free for k in _PARAM_NAMES])
    logger.info("Step 2: full-lap refinement on %d laps (free: %s)", len(laps),
                "all" if free is None else free)
    result = least_squares(
        _residuals_subset,
        theta0_arr[mask],
        bounds=(lb[mask], ub[mask]),
        method="trf",
        loss=calib_cfg.get("loss", "soft_l1"),
        f_scale=float(calib_cfg.get("f_scale_ms", 3.0)),
        x_scale=scale[mask],
        max_nfev=int(calib_cfg.get("max_nfev", 400)),
        args=(mask, theta0_arr, *args),
    )
    x_full = theta0_arr.copy()
    x_full[mask] = result.x
    params = VehicleParams.from_array(x_full, **fixed)

    # Jacobian-based standard errors (Gauss-Newton approximation at the optimum).
    try:
        J = result.jac
        dof = max(1, len(result.fun) - len(result.x))
        s2 = 2 * result.cost / dof
        cov = np.linalg.pinv(J.T @ J) * s2
        stderr_free = np.sqrt(np.clip(np.diag(cov), 0, None))
    except (np.linalg.LinAlgError, ValueError):
        stderr_free = np.full(int(mask.sum()), np.nan)
    stderr = np.full(len(_PARAM_NAMES), np.nan)
    stderr[mask] = stderr_free

    at_bound = [
        k for k, v, lo, hi, m in zip(_PARAM_NAMES, x_full, lb, ub, mask)
        if m and (v <= lo + 1e-3 * (hi - lo) or v >= hi - 1e-3 * (hi - lo))
    ]
    if at_bound:
        logger.warning("Parameters at their bounds (model/bound mismatch?): %s", at_bound)

    diagnostics = {
        "n_laps": len(laps),
        "laps": [lp.name for lp in laps],
        "step1_estimates": theta0,
        "final_params": {k: float(v) for k, v in zip(_PARAM_NAMES, x_full)},
        "stderr": {k: float(v) for k, v in zip(_PARAM_NAMES, stderr)},
        "params_at_bound": at_bound,
        "optimizer_status": int(result.status),
        "optimizer_message": str(result.message),
        "nfev": int(result.nfev),
        "cost": float(result.cost),
    }
    logger.info("Calibration done: %s", diagnostics["final_params"])
    return params, diagnostics


def save_theta(params: VehicleParams, diagnostics: dict, path: Path) -> None:
    """Save calibrated parameters and diagnostics to a JSON file.

    Parameters
    ----------
    params:
        Calibrated vehicle parameters.
    diagnostics:
        Diagnostics dict from :func:`calibrate`.
    path:
        Output JSON file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "params": {
            "mu": params.mu,
            "mu_x_brake": params.mu_x_brake,
            "mu_x_drive": params.mu_x_drive,
            "c_l": params.c_l,
            "c_d": params.c_d,
            "p_spec": params.p_spec,
            "a_brake_cap": params.a_brake_cap,
            "g": params.g,
            "v_min": params.v_min,
            "v_max": params.v_max,
        },
        "diagnostics": diagnostics,
    }
    with path.open("w") as fh:
        json.dump(data, fh, indent=2, default=float)
    logger.info("Saved theta to %s", path)


def load_theta(path: Path) -> VehicleParams:
    """Load calibrated parameters from a JSON file.

    Parameters
    ----------
    path:
        JSON file path written by :func:`save_theta`.

    Returns
    -------
    VehicleParams
    """
    with Path(path).open() as fh:
        data = json.load(fh)
    p = data["params"]
    return VehicleParams(
        mu=p["mu"],
        mu_x_brake=p["mu_x_brake"],
        mu_x_drive=p["mu_x_drive"],
        c_l=p["c_l"],
        c_d=p["c_d"],
        p_spec=p["p_spec"],
        a_brake_cap=p["a_brake_cap"],
        g=p.get("g", 9.81),
        v_min=p.get("v_min", 5.0),
        v_max=p.get("v_max", 100.0),
    )
