"""Pytest configuration and shared fixtures for f1line tests."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded random number generator for reproducibility."""
    return np.random.default_rng(42)


@pytest.fixture
def default_params():
    """Default vehicle parameters for testing."""
    from f1line.vehicle.model import VehicleParams
    return VehicleParams(
        mu=2.5,
        mu_x_brake=2.75,
        mu_x_drive=2.0,
        c_l=0.001,
        c_d=0.0005,
        p_spec=750.0,
        a_brake_cap=52.0,
    )


@pytest.fixture
def aero_free_params():
    """Vehicle parameters with no aero (c_l=0, c_d=0) for analytic tests."""
    from f1line.vehicle.model import VehicleParams
    return VehicleParams(
        mu=2.5,
        mu_x_brake=2.75,
        mu_x_drive=2.0,
        c_l=0.0,
        c_d=0.000001,  # tiny drag to avoid zero denominator in top speed
        p_spec=750.0,
        a_brake_cap=52.0,
    )
