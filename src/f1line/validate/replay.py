"""Replay validation: simulate the driver's own line with the calibrated model."""

from __future__ import annotations

import logging

import numpy as np

from f1line.vehicle.model import VehicleParams
from f1line.vehicle.qss import QSSResult, simulate_lap_on_line

logger = logging.getLogger(__name__)


def replay_lap(
    x_driven: np.ndarray,
    y_driven: np.ndarray,
    params: VehicleParams,
) -> QSSResult:
    """Simulate the driver's own line through the calibrated vehicle model.

    Parameters
    ----------
    x_driven, y_driven:
        Driven line coordinates [m] (smoothed).
    params:
        Calibrated vehicle parameters.

    Returns
    -------
    QSSResult
        Simulated speed profile and lap time.
    """
    return simulate_lap_on_line(x_driven, y_driven, params)
