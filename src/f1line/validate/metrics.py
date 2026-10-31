"""Validation metrics for lap time, speed trace, and lateral deviation."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_EPS = 1e-9


def lap_time_error_pct(predicted_s: float, actual_s: float) -> float:
    """Compute percentage lap time error.

    Parameters
    ----------
    predicted_s:
        Predicted lap time [s].
    actual_s:
        Actual (real) lap time [s].

    Returns
    -------
    float
        Signed percentage error: (predicted - actual) / actual * 100.
    """
    return (predicted_s - actual_s) / (actual_s + _EPS) * 100.0


def speed_rmse_kmh(v_pred_ms: np.ndarray, v_real_ms: np.ndarray) -> float:
    """Root-mean-square error of speed profiles in km/h.

    Parameters
    ----------
    v_pred_ms:
        Predicted speed [m/s]. Resampled to same length as v_real_ms.
    v_real_ms:
        Real speed [m/s].

    Returns
    -------
    float
        RMSE in km/h.
    """
    if len(v_pred_ms) != len(v_real_ms):
        t = np.linspace(0, 1, len(v_real_ms))
        t_pred = np.linspace(0, 1, len(v_pred_ms))
        v_pred_ms = np.interp(t, t_pred, v_pred_ms)
    diff_kmh = (v_pred_ms - v_real_ms) * 3.6
    return float(np.sqrt(np.mean(diff_kmh**2)))


def lateral_deviation(
    x_opt: np.ndarray,
    y_opt: np.ndarray,
    x_driven: np.ndarray,
    y_driven: np.ndarray,
) -> dict[str, float]:
    """Compute lateral deviation statistics between two racing lines.

    Parameters
    ----------
    x_opt, y_opt:
        Optimized line coordinates [m].
    x_driven, y_driven:
        Driven line coordinates [m].

    Returns
    -------
    dict with keys: mean_m, max_m, std_m.
    """
    from scipy.spatial import cKDTree

    tree = cKDTree(np.column_stack([x_driven, y_driven]))
    dists, _ = tree.query(np.column_stack([x_opt, y_opt]), k=1)
    return {
        "mean_m": float(np.mean(dists)),
        "max_m": float(np.max(dists)),
        "std_m": float(np.std(dists)),
    }


def delta_time(
    v_a: np.ndarray,
    v_b: np.ndarray,
    ds: float | np.ndarray,
) -> np.ndarray:
    """Cumulative time delta between two speed profiles along the lap.

    Positive = a is slower than b.

    Parameters
    ----------
    v_a, v_b:
        Speed profiles [m/s]. Shape (N,).
    ds:
        Arc-length spacing [m].

    Returns
    -------
    np.ndarray
        Cumulative time delta [s]. Shape (N,).
    """
    if np.isscalar(ds):
        ds_arr = np.full(len(v_a), float(ds))
    else:
        ds_arr = np.asarray(ds, dtype=float)

    dt_a = ds_arr / (v_a + _EPS)
    dt_b = ds_arr / (v_b + _EPS)
    return np.cumsum(dt_a - dt_b)


def build_metrics_table(
    results: list[dict[str, Any]],
) -> pd.DataFrame:
    """Build a summary metrics table from a list of per-circuit result dicts.

    Parameters
    ----------
    results:
        List of dicts with keys: circuit, lap_time_pred_s, lap_time_real_s,
        speed_rmse_kmh, lat_dev_mean_m, lat_dev_max_m.

    Returns
    -------
    pd.DataFrame
    """
    rows = []
    for r in results:
        lt_pred = r.get("lap_time_pred_s", float("nan"))
        lt_real = r.get("lap_time_real_s", float("nan"))
        lt_err = lap_time_error_pct(lt_pred, lt_real) if not np.isnan(lt_pred + lt_real) else float("nan")
        rows.append({
            "Circuit": r.get("circuit", "?"),
            "Pred LT [s]": f"{lt_pred:.3f}",
            "Real LT [s]": f"{lt_real:.3f}",
            "LT Err [%]": f"{lt_err:+.2f}",
            "Speed RMSE [km/h]": f"{r.get('speed_rmse_kmh', float('nan')):.2f}",
            "Lat Dev Mean [m]": f"{r.get('lat_dev_mean_m', float('nan')):.2f}",
            "Lat Dev Max [m]": f"{r.get('lat_dev_max_m', float('nan')):.2f}",
        })
    return pd.DataFrame(rows)
