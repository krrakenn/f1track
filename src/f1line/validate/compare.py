"""Compare racing lines: optimized vs driven vs TUM raceline vs centerline."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from f1line.validate.metrics import delta_time, lateral_deviation

logger = logging.getLogger(__name__)


def compare_lines(
    centerline_x: np.ndarray,
    centerline_y: np.ndarray,
    opt_x: np.ndarray,
    opt_y: np.ndarray,
    driven_x: Optional[np.ndarray],
    driven_y: Optional[np.ndarray],
    tum_x: Optional[np.ndarray],
    tum_y: Optional[np.ndarray],
    v_opt: Optional[np.ndarray] = None,
    v_driven: Optional[np.ndarray] = None,
    ds: float = 3.0,
) -> dict:
    """Compare multiple racing lines and return statistics.

    Parameters
    ----------
    centerline_x, centerline_y:
        Track center line [m].
    opt_x, opt_y:
        Optimized (min-time) racing line [m].
    driven_x, driven_y:
        Real driven line [m] (optional).
    tum_x, tum_y:
        TUM precomputed race line [m] (optional).
    v_opt, v_driven:
        Speed profiles for the optimized and driven lines [m/s] (optional).
    ds:
        Arc-length spacing for delta-time [m].

    Returns
    -------
    dict
        Keys: opt_vs_driven, opt_vs_tum, opt_vs_center, delta_time (if speeds given).
    """
    result = {}

    if driven_x is not None:
        result["opt_vs_driven"] = lateral_deviation(opt_x, opt_y, driven_x, driven_y)

    if tum_x is not None:
        result["opt_vs_tum"] = lateral_deviation(opt_x, opt_y, tum_x, tum_y)

    result["opt_vs_center"] = lateral_deviation(opt_x, opt_y, centerline_x, centerline_y)

    if v_opt is not None and v_driven is not None:
        n = min(len(v_opt), len(v_driven))
        result["delta_time_s"] = delta_time(
            np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(v_driven)), v_driven),
            np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(v_opt)), v_opt),
            ds,
        ).tolist()

    return result
