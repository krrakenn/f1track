"""Static matplotlib figures for the report.

Saves figures to reports/figures/ under subdirectories:
- tracks/: circuit maps
- calibration/: a_lat vs v, a_long vs v envelopes
- validation/: speed traces, delta-time, lateral deviation
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def plot_circuit(
    name: str,
    cx: np.ndarray,
    cy: np.ndarray,
    bl_x: np.ndarray,
    bl_y: np.ndarray,
    br_x: np.ndarray,
    br_y: np.ndarray,
    kappa: np.ndarray,
    out_dir: Path,
    racing_lines: Optional[dict[str, tuple[np.ndarray, np.ndarray]]] = None,
) -> Path:
    """Plot circuit map with curvature color coding and optional racing lines.

    Parameters
    ----------
    name:
        Circuit name.
    cx, cy:
        Center-line coordinates [m].
    bl_x, bl_y:
        Left boundary coordinates [m].
    br_x, br_y:
        Right boundary coordinates [m].
    kappa:
        Signed curvature along center line [1/m].
    out_dir:
        Output directory.
    racing_lines:
        Optional dict {label: (x, y)} of additional lines to overlay.

    Returns
    -------
    Path
        Path to the saved figure.
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(f"{name.title()} – Track Map & Curvature", fontsize=14)

    # ── Left: track map ────────────────────────────────────────────────────
    ax = axes[0]
    ax.set_aspect("equal")
    ax.set_title("Track map (curvature-colored)")

    # Curvature-colored center line
    points = np.column_stack([cx, cy]).reshape(-1, 1, 2)
    segs = np.concatenate([points[:-1], points[1:]], axis=1)
    kappa_norm = np.abs(kappa[:-1])
    vmax = np.percentile(kappa_norm, 95) + 1e-9
    lc = LineCollection(segs, cmap="plasma", norm=plt.Normalize(0, vmax), linewidth=2)
    lc.set_array(kappa_norm)
    ax.add_collection(lc)
    plt.colorbar(lc, ax=ax, label="|κ| [1/m]", shrink=0.7)

    # Boundaries
    ax.plot(bl_x, bl_y, "k-", linewidth=0.8, alpha=0.5, label="Boundary")
    ax.plot(br_x, br_y, "k-", linewidth=0.8, alpha=0.5)

    # Racing lines overlay
    if racing_lines:
        colors = ["#e74c3c", "#2ecc71", "#3498db", "#f39c12"]
        for (label, (rx, ry)), col in zip(racing_lines.items(), colors):
            ax.plot(rx, ry, "-", color=col, linewidth=1.5, label=label, alpha=0.8)

    ax.legend(fontsize=8, loc="best")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    # ── Right: curvature profile ───────────────────────────────────────────
    ax2 = axes[1]
    n = len(kappa)
    s = np.linspace(0, 1, n) * (n * 3)  # approximate arc length
    ax2.plot(s, kappa * 1000, color="#2c3e50", linewidth=1)
    ax2.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax2.set_title("Signed curvature along arc length")
    ax2.set_xlabel("Arc length [m]")
    ax2.set_ylabel("κ [1/km]")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_dir = _ensure_dir(out_dir)
    fpath = out_dir / f"{name}_circuit.png"
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved circuit figure: %s", fpath)
    return fpath


def plot_speed_comparison(
    name: str,
    s_real: np.ndarray,
    v_real_ms: np.ndarray,
    s_pred: np.ndarray,
    v_pred_ms: np.ndarray,
    out_dir: Path,
) -> Path:
    """Plot real vs predicted speed traces.

    Parameters
    ----------
    name:
        Circuit + driver name for title.
    s_real, v_real_ms:
        Real speed trace.
    s_pred, v_pred_ms:
        Predicted speed trace.
    out_dir:
        Output directory.

    Returns
    -------
    Path
        Path to saved figure.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(s_real, v_real_ms * 3.6, label="Real", color="#2c3e50", linewidth=1.5)
    ax.plot(s_pred, v_pred_ms * 3.6, label="Predicted", color="#e74c3c",
            linewidth=1.5, linestyle="--")
    ax.set_xlabel("Arc length [m]")
    ax.set_ylabel("Speed [km/h]")
    ax.set_title(f"{name} – Speed trace comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out_dir = _ensure_dir(out_dir)
    fpath = out_dir / f"{name}_speed_comparison.png"
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fpath


def plot_calibration_envelope(
    name: str,
    v_ms: np.ndarray,
    a_lat_obs: np.ndarray,
    a_lat_fit: np.ndarray,
    out_dir: Path,
) -> Path:
    """Plot lateral acceleration envelope fit.

    Parameters
    ----------
    name:
        Label for title.
    v_ms:
        Speed bins [m/s].
    a_lat_obs:
        Observed high-quantile a_lat [m/s^2].
    a_lat_fit:
        Model-fitted a_lat_max [m/s^2].
    out_dir:
        Output directory.

    Returns
    -------
    Path
        Path to saved figure.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(v_ms * 3.6, a_lat_obs / 9.81, s=20, label="Observed 98th pctile", alpha=0.7)
    ax.plot(v_ms * 3.6, a_lat_fit / 9.81, "r-", label="Fitted envelope", linewidth=2)
    ax.set_xlabel("Speed [km/h]")
    ax.set_ylabel("a_lat [g]")
    ax.set_title(f"{name} – Lateral acceleration envelope")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out_dir = _ensure_dir(out_dir)
    fpath = out_dir / f"{name}_lat_envelope.png"
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fpath
