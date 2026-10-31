"""Artifact loading and payload building for the Streamlit app (no Streamlit UI here)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from f1line.config import load_config
from f1line.season import season_circuits
from f1line.vehicle.model import VehicleParams
from f1line.vehicle.qss import menger_curvature, simulate_lap

REPO = Path(__file__).resolve().parent.parent


def season_root() -> Path:
    cfg = load_config()
    return REPO / cfg["paths"]["season_dir"]


def available_circuits() -> list[dict[str, Any]]:
    """Calendar entries that have built artifacts, in round order (cancelled last)."""
    root = season_root()
    out = []
    for slug, c in season_circuits(load_config()).items():
        if (root / slug / "meta.json").exists():
            out.append({"slug": slug, **c})
    return sorted(out, key=lambda c: (c.get("round") is None, c.get("round") or 0))


def load_artifacts(slug: str) -> dict[str, Any]:
    """Track, lines, history and meta of one circuit."""
    d = season_root() / slug
    hist = np.load(d / "history.npz")
    return {
        "meta": json.loads((d / "meta.json").read_text(encoding="utf-8")),
        "track": pd.read_parquet(d / "track.parquet"),
        "lines": pd.read_parquet(d / "lines.parquet"),
        "history": {k: hist[k] for k in hist.files},
    }


def season_table() -> Optional[pd.DataFrame]:
    cfg = load_config()
    p = REPO / "reports" / f"season_{cfg['season']['year']}.csv"
    return pd.read_csv(p) if p.exists() else None


def _line_xy(track: pd.DataFrame, alpha: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return (track["x"].to_numpy() + alpha * track["nx"].to_numpy(),
            track["y"].to_numpy() + alpha * track["ny"].to_numpy())


def _seg(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.hypot(np.roll(x, -1) - x, np.roll(y, -1) - y)


def _time_along(v: np.ndarray, ds: np.ndarray, total: Optional[float] = None) -> np.ndarray:
    """Elapsed time at each grid point [s]; optionally scaled to a known lap time."""
    dt = 2 * ds / (v + np.roll(v, -1))
    t = np.concatenate([[0.0], np.cumsum(dt)[:-1]])
    if total is not None:
        t *= total / dt.sum()
    return t


def resimulate(art: dict[str, Any], params: VehicleParams) -> dict[str, Any]:
    """QSS on the stored optimal line with user parameters."""
    x, y = _line_xy(art["track"], art["lines"]["alpha_mintime"].to_numpy())
    ds = _seg(x, y)
    r = simulate_lap(menger_curvature(x, y), ds, params)
    return {"v": r.v, "t": _time_along(r.v, ds), "lap": r.lap_time_s}


def _r(a: Any, nd: int = 2) -> list:
    return np.round(np.asarray(a, dtype=float), nd).tolist()


def simulator_payload(
    art: dict[str, Any], show: dict[str, bool], live: Optional[dict[str, Any]] = None
) -> dict[str, Any]:
    """Everything the simulator component draws, as plain JSON-able lists."""
    meta, tr, ln, h = art["meta"], art["track"], art["lines"], art["history"]
    L = float(meta["length_m"])

    a_opt = ln["alpha_mintime"].to_numpy()
    xo, yo = _line_xy(tr, a_opt)
    ds_o = _seg(xo, yo)
    v_opt = ln["v_mintime"].to_numpy()
    lap_opt = float(meta["lap_times_s"]["min_time"])
    t_opt = _time_along(v_opt, ds_o)

    a_real = ln["alpha_real"].to_numpy()
    xr, yr = _line_xy(tr, a_real)
    ds_r = _seg(xr, yr)
    v_real = ln["v_real"].to_numpy()
    t_real = _time_along(v_real, ds_r, total=float(meta["pole_time_s"]))
    along_real = np.gradient(v_real**2 / 2, np.cumsum(ds_r))

    payload = {
        "circuit": meta["circuit"],
        "track": {
            "s": _r(tr["s"], 1), "x": _r(tr["x"]), "y": _r(tr["y"]),
            "nx": _r(tr["nx"], 4), "ny": _r(tr["ny"], 4),
            "wl": _r(tr["w_left"]), "wr": _r(tr["w_right"]), "kappa": _r(tr["kappa"], 5),
            "L": L, "rotation": float(meta.get("rotation_deg") or 0.0),
        },
        "lines": {
            "optimal": {
                "alpha": _r(a_opt), "v": _r(v_opt), "t": _r(t_opt, 3), "lap": lap_opt,
                "along": _r(ln["a_long_mintime"]), "alat": _r(ln["a_lat_mintime"]),
            },
            "mincurv": {"alpha": _r(ln["alpha_mincurv"]), "v": _r(ln["v_mincurv"])},
            "real": {
                "alpha": _r(a_real), "v": _r(v_real), "t": _r(t_real, 3),
                "throttle": _r(ln["throttle_real"], 0), "brake": _r(ln["brake_real"], 0),
                "along": _r(along_real),
                "label": f"{meta['pole_driver']} {meta['data_year']}",
            },
            "gap": _r(t_real - t_opt, 3),
            "live": (
                {"v": _r(live["v"]), "t": _r(live["t"], 3), "lap": live["lap"]} if live else None
            ),
        },
        "stages": {k: _r(ln[k]) for k in ("v_corner", "v_accel", "v_brake")}
        | {"v_final": _r(v_opt)},
        "history": {
            "alpha": [_r(a) for a in h["alpha"]],
            "v": [_r(v) for v in h["v"]],
            "lap": _r(h["lap_time_s"], 3),
            "stage": [str(s) for s in h["stage"]],
            "iter": [int(i) for i in h["iteration"]],
        },
        "corners": meta.get("corners", []),
        "show": show,
    }
    return payload
