"""Vehicle model parameters and limit functions.

Point-mass model with aero (downforce + drag) and a friction ellipse.
All quantities per unit mass unless noted. Internal units: SI.

Parameters
----------
mu : float
    Lateral grip coefficient [-]. Includes tyre + aero lateral coupling.
mu_x_brake : float
    Longitudinal grip coefficient under braking [-].
mu_x_drive : float
    Longitudinal grip coefficient under drive [-].
c_l : float
    Downforce coefficient per unit mass [1/m]. c_l = rho*CLA/(2m).
    a_downforce = c_l * v^2.
c_d : float
    Drag coefficient per unit mass [1/m]. c_d = rho*CdA/(2m).
    a_drag = c_d * v^2.
p_spec : float
    Specific power [W/kg]. Effective power available at the wheels per unit mass.
a_brake_cap : float
    Hard cap on braking deceleration [m/s^2] (brake thermal/hydraulic limit).
g : float
    Gravitational acceleration [m/s^2] (default 9.81).
v_min : float
    Minimum speed for numerical stability [m/s] (default 5.0).
v_max : float
    Hard cap on speed [m/s] (default 100.0).

Limit equations (Section 5.2):
    a_lat_max(v)   = mu * (g + c_l * v^2)
    a_brake_max(v) = min(mu_x_brake * (g + c_l * v^2) + c_d * v^2, a_brake_cap)
    a_drive_max(v) = min(P_spec / max(v, v_min), mu_x_drive * (g + c_l * v^2)) - c_d * v^2
    v_corner(k)    = sqrt(mu * g / (k - mu * c_l))  if k > mu * c_l  else v_max
    v_top          = (P_spec / c_d)^(1/3)  approximately (at drag = drive limit)

See docs/MODEL.md for full derivations and bound justification.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Physical constants
_G_DEFAULT = 9.81  # [m/s^2]
_V_MIN_DEFAULT = 5.0  # [m/s]
_V_MAX_DEFAULT = 100.0  # [m/s]


@dataclass
class VehicleParams:
    """Vehicle model parameters (all per unit mass, SI units).

    Attributes
    ----------
    mu : float
        Lateral grip coefficient [-].
    mu_x_brake : float
        Longitudinal grip under braking [-].
    mu_x_drive : float
        Longitudinal grip under drive [-].
    c_l : float
        Downforce coefficient [1/m].
    c_d : float
        Drag coefficient [1/m].
    p_spec : float
        Specific power [W/kg].
    a_brake_cap : float
        Hard braking deceleration cap [m/s^2].
    g : float
        Gravity [m/s^2].
    v_min : float
        Minimum speed for numerical stability [m/s].
    v_max : float
        Maximum speed cap [m/s].
    """

    mu: float
    mu_x_brake: float
    mu_x_drive: float
    c_l: float
    c_d: float
    p_spec: float
    a_brake_cap: float
    g: float = _G_DEFAULT
    v_min: float = _V_MIN_DEFAULT
    v_max: float = _V_MAX_DEFAULT

    def __post_init__(self) -> None:
        """Validate that parameters are physically plausible."""
        assert self.mu > 0, f"mu must be positive, got {self.mu}"
        assert self.c_l >= 0, f"c_l must be non-negative, got {self.c_l}"
        assert self.c_d > 0, f"c_d must be positive, got {self.c_d}"
        assert self.p_spec > 0, f"p_spec must be positive, got {self.p_spec}"
        assert self.a_brake_cap > 0, f"a_brake_cap must be positive, got {self.a_brake_cap}"

    def to_array(self) -> np.ndarray:
        """Serialize parameters to a 1-D array for optimization.

        Returns
        -------
        np.ndarray
            Shape (7,): [mu, mu_x_brake, mu_x_drive, c_l, c_d, p_spec, a_brake_cap].
        """
        return np.array(
            [self.mu, self.mu_x_brake, self.mu_x_drive, self.c_l,
             self.c_d, self.p_spec, self.a_brake_cap],
            dtype=float,
        )

    @classmethod
    def from_array(
        cls,
        arr: np.ndarray,
        g: float = _G_DEFAULT,
        v_min: float = _V_MIN_DEFAULT,
        v_max: float = _V_MAX_DEFAULT,
    ) -> "VehicleParams":
        """Deserialize from a 1-D array (inverse of :meth:`to_array`).

        Parameters
        ----------
        arr:
            Shape (7,): [mu, mu_x_brake, mu_x_drive, c_l, c_d, p_spec, a_brake_cap].
        g, v_min, v_max:
            Fixed parameters not part of the optimization vector.

        Returns
        -------
        VehicleParams
        """
        return cls(
            mu=float(arr[0]),
            mu_x_brake=float(arr[1]),
            mu_x_drive=float(arr[2]),
            c_l=float(arr[3]),
            c_d=float(arr[4]),
            p_spec=float(arr[5]),
            a_brake_cap=float(arr[6]),
            g=g,
            v_min=v_min,
            v_max=v_max,
        )

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "VehicleParams":
        """Build from the ``vehicle`` section of the config dict.

        Uses the mid-point of each bound as a default starting value.

        Parameters
        ----------
        cfg:
            The ``vehicle`` sub-dict from ``config/default.yaml``.

        Returns
        -------
        VehicleParams
        """
        bounds = cfg.get("bounds", {})

        def _mid(key: str, default: float) -> float:
            b = bounds.get(key)
            if b and len(b) == 2:
                return float(np.mean(b))
            return default

        return cls(
            mu=_mid("mu", 2.5),
            mu_x_brake=_mid("mu_x_brake", 2.75),
            mu_x_drive=_mid("mu_x_drive", 2.0),
            c_l=_mid("c_l", 0.001),
            c_d=_mid("c_d", 0.0005),
            p_spec=_mid("p_spec_w_per_kg", 750.0),
            a_brake_cap=_mid("a_brake_cap_ms2", 52.5),
            g=float(cfg.get("g", _G_DEFAULT)),
            v_min=float(cfg.get("v_min_ms", _V_MIN_DEFAULT)),
            v_max=float(cfg.get("v_max_ms", _V_MAX_DEFAULT)),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Vectorized limit functions
# ─────────────────────────────────────────────────────────────────────────────


def a_lat_max(v: np.ndarray, p: VehicleParams) -> np.ndarray:
    """Maximum lateral acceleration envelope [m/s^2].

    a_lat_max(v) = mu * (g + c_l * v^2)

    Parameters
    ----------
    v:
        Speed [m/s].
    p:
        Vehicle parameters.

    Returns
    -------
    np.ndarray
        Maximum lateral acceleration [m/s^2]. Always positive.
    """
    v = np.asarray(v, dtype=float)
    return p.mu * (p.g + p.c_l * v**2)


def a_brake_max(v: np.ndarray, p: VehicleParams) -> np.ndarray:
    """Maximum braking deceleration magnitude [m/s^2].

    a_brake_max(v) = min(mu_x_brake * (g + c_l * v^2) + c_d * v^2, a_brake_cap)

    Drag assists braking. Capped by hardware limit.

    Parameters
    ----------
    v:
        Speed [m/s].
    p:
        Vehicle parameters.

    Returns
    -------
    np.ndarray
        Maximum deceleration magnitude [m/s^2]. Always positive.
    """
    v = np.asarray(v, dtype=float)
    aero_grip = p.mu_x_brake * (p.g + p.c_l * v**2)
    drag = p.c_d * v**2
    return np.minimum(aero_grip + drag, p.a_brake_cap)


def a_drive_max(v: np.ndarray, p: VehicleParams) -> np.ndarray:
    """Maximum drive acceleration (net, after drag) [m/s^2].

    a_drive_max(v) = min(P_spec / max(v, v_min), mu_x_drive * (g + c_l * v^2)) - c_d * v^2

    Can be negative at very high speeds where drag exceeds power.

    Parameters
    ----------
    v:
        Speed [m/s].
    p:
        Vehicle parameters.

    Returns
    -------
    np.ndarray
        Net drive acceleration [m/s^2].
    """
    v = np.asarray(v, dtype=float)
    v_safe = np.maximum(v, p.v_min)
    power_limit = p.p_spec / v_safe
    traction_limit = p.mu_x_drive * (p.g + p.c_l * v**2)
    gross = np.minimum(power_limit, traction_limit)
    net = gross - p.c_d * v**2
    return net


def v_corner_limit(kappa: np.ndarray, p: VehicleParams) -> np.ndarray:
    """Steady-state cornering speed limit [m/s].

    Solves: v^2 * |kappa| = mu * (g + c_l * v^2)
    -> v^2 = mu * g / (|kappa| - mu * c_l)  if |kappa| > mu * c_l
    else unbounded (cap at v_max).

    Parameters
    ----------
    kappa:
        Curvature [1/m].
    p:
        Vehicle parameters.

    Returns
    -------
    np.ndarray
        Speed limit per point [m/s].
    """
    kappa = np.asarray(kappa, dtype=float)
    k_abs = np.abs(kappa)

    denom = k_abs - p.mu * p.c_l
    v_lim = np.where(
        denom > 1e-9,
        np.sqrt(np.clip(p.mu * p.g / denom, 0.0, p.v_max**2)),
        p.v_max,
    )
    return np.clip(v_lim, 0.0, p.v_max)


def v_top(p: VehicleParams) -> float:
    """Approximate top speed where drag equals drive power [m/s].

    v_top ≈ (P_spec / c_d)^(1/3)

    Parameters
    ----------
    p:
        Vehicle parameters.

    Returns
    -------
    float
        Approximate top speed [m/s].
    """
    return float((p.p_spec / p.c_d) ** (1.0 / 3.0))


def get_bounds(cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Return lower and upper parameter bounds as arrays matching :meth:`VehicleParams.to_array`.

    Parameters
    ----------
    cfg:
        The ``vehicle`` sub-dict from the config.

    Returns
    -------
    (lb, ub):
        Lower and upper bounds, shape (7,).
    """
    b = cfg.get("bounds", {})

    def _get(key: str, default_lo: float, default_hi: float) -> tuple[float, float]:
        val = b.get(key)
        if val and len(val) == 2:
            return float(val[0]), float(val[1])
        return default_lo, default_hi

    mu_lo, mu_hi = _get("mu", 1.5, 3.5)
    mxb_lo, mxb_hi = _get("mu_x_brake", 1.5, 4.0)
    mxd_lo, mxd_hi = _get("mu_x_drive", 1.0, 3.0)
    cl_lo, cl_hi = _get("c_l", 0.0001, 0.002)
    cd_lo, cd_hi = _get("c_d", 0.00005, 0.001)
    ps_lo, ps_hi = _get("p_spec_w_per_kg", 500.0, 1000.0)
    ab_lo, ab_hi = _get("a_brake_cap_ms2", 40.0, 65.0)

    lb = np.array([mu_lo, mxb_lo, mxd_lo, cl_lo, cd_lo, ps_lo, ab_lo])
    ub = np.array([mu_hi, mxb_hi, mxd_hi, cl_hi, cd_hi, ps_hi, ab_hi])
    return lb, ub
