"""Export season artifacts as compact static JSON for the web app (``f1line export-web``).

Writes ``<out>/data/index.json`` (all races: event, pole sitter, team colour, times, a small
track outline) and ``<out>/data/<circuit>.json`` per race (track, lines, speeds, optimizer
history), so the page loads one small index and fetches race data on demand.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from f1line.season import openf1_client, season_circuits
from f1line.vehicle.qss import menger_curvature


def _q(a: Any, nd: int) -> list:
    """Round for JSON; integers when nd == 0 (smaller payload)."""
    arr = np.round(np.asarray(a, dtype=float), nd)
    return arr.astype(int).tolist() if nd == 0 else arr.tolist()


def _line_xy(tr: pd.DataFrame, a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return tr["x"].to_numpy() + a * tr["nx"].to_numpy(), tr["y"].to_numpy() + a * tr["ny"].to_numpy()


def _real_time_on_grid(df: pd.DataFrame, tr: pd.DataFrame, L: float, lap_time: float) -> np.ndarray:
    """Pole lap elapsed time [s] at each track grid point, from the timestamped positions.

    Samples are projected onto the reference line (nearest grid point plus the along-track
    component); samples snapped to the wrong branch (Suzuka's crossover) are rejected against
    the running median. Using the feed's own timestamps places sector boundaries, the ghost
    car and the gap exactly; integrating the speed trace instead drifted up to 3 s.
    """
    from scipy.ndimage import median_filter
    from scipy.spatial import cKDTree

    c = tr[["x", "y"]].to_numpy()
    s_grid = tr["s"].to_numpy()
    pts = df[["X_m", "Y_m"]].to_numpy(float)
    times = df["Time_s"].to_numpy(float)
    _, i = cKDTree(c).query(pts)
    tx, ty = tr["ny"].to_numpy()[i], -tr["nx"].to_numpy()[i]
    s = (s_grid[i] + (pts[:, 0] - c[i, 0]) * tx + (pts[:, 1] - c[i, 1]) * ty) % L
    step = np.diff(s, prepend=s[0])
    su = s + L * np.cumsum(np.where(step < -L / 2, 1, 0) - np.where(step > L / 2, 1, 0))
    if su[0] > L / 2:
        su -= L
    ok = np.abs(su - median_filter(su, size=9, mode="nearest")) < 60.0
    ok &= (su > 0) & (su < L)
    su, times = su[ok], times[ok]
    order = np.argsort(su)
    su, times = su[order], np.maximum.accumulate(times[order])
    su = np.concatenate([[0.0], su, [L]])
    times = np.concatenate([[0.0], times, [lap_time]])
    return np.interp(s_grid, su, times)


def _elapsed(v: np.ndarray, x: np.ndarray, y: np.ndarray, total: float | None = None) -> np.ndarray:
    ds = np.hypot(np.roll(x, -1) - x, np.roll(y, -1) - y)
    dt = 2 * ds / (v + np.roll(v, -1))
    t = np.concatenate([[0.0], np.cumsum(dt)[:-1]])
    return t * (total / dt.sum()) if total else t


def _outline(tr: pd.DataFrame, rotation_deg: float, n: int = 140) -> list:
    """Rotated, normalized (0..1, y down) center line for the timing-tower silhouettes."""
    idx = np.linspace(0, len(tr) - 1, n).astype(int)
    r = np.radians(rotation_deg)
    x, y = tr["x"].to_numpy()[idx], tr["y"].to_numpy()[idx]
    xr, yr = np.cos(r) * x - np.sin(r) * y, np.sin(r) * x + np.cos(r) * y
    span = max(np.ptp(xr), np.ptp(yr))
    xn = (xr - xr.min()) / span + (1 - np.ptp(xr) / span) / 2
    yn = 1 - ((yr - yr.min()) / span + (1 - np.ptp(yr) / span) / 2)
    return np.round(np.column_stack([xn, yn]), 3).tolist()


def _pole_driver(client: Any, session_key: int, acronym: str) -> dict[str, str]:
    d = client.drivers(session_key)
    row = d[d["name_acronym"] == acronym]
    if row.empty:
        return {"name": acronym, "team": "", "colour": "#9aa4b5"}
    r = row.iloc[0]
    first = str(r.get("first_name") or "").strip()
    last = str(r.get("last_name") or "").strip().title()
    return {
        "name": f"{first} {last}".strip() or acronym,
        "team": str(r.get("team_name") or ""),
        "colour": f"#{r.get('team_colour') or '9aa4b5'}",
    }


def _jsonable(o: Any) -> Any:
    """JSON fallback for numpy scalars."""
    return o.item() if hasattr(o, "item") else str(o)


def _session_extras(client: Any, session_key: int, acronym: str, pole_time: float) -> dict[str, Any]:
    """Official extras for the pole lap: sectors, speed traps, tyre, weather, top five."""
    out: dict[str, Any] = {}
    drv = client.drivers(session_key)
    num = dict(zip(drv["name_acronym"], drv["driver_number"]))
    info = {int(r["driver_number"]): r for _, r in drv.iterrows()}
    laps = client.laps(session_key)
    lap = laps[(laps["driver_number"] == num.get(acronym))
               & ((laps["lap_duration"] - pole_time).abs() < 0.0015)]
    if not lap.empty:
        r = lap.iloc[0]
        out["sectors"] = [r.get("duration_sector_1"), r.get("duration_sector_2"),
                          r.get("duration_sector_3")]
        out["speeds"] = {"i1": r.get("i1_speed"), "i2": r.get("i2_speed"), "st": r.get("st_speed")}
        stints = pd.DataFrame(client.get("stints", [("session_key", session_key),
                                                    ("driver_number", num.get(acronym))]))
        if not stints.empty:
            n = int(r["lap_number"])
            s = stints[(stints["lap_start"] <= n) & (stints["lap_end"] >= n)]
            if not s.empty:
                s = s.iloc[0]
                out["tyre"] = {"compound": str(s["compound"]).title(),
                               "age_laps": int(s["tyre_age_at_start"]) + n - int(s["lap_start"])}
        weather = pd.DataFrame(client.get("weather", [("session_key", session_key)]))
        if not weather.empty and r.get("date_start"):
            wt = pd.to_datetime(weather["date"], utc=True, format="ISO8601")
            w = weather.iloc[int(np.argmin(np.abs(wt - pd.to_datetime(r["date_start"], utc=True))))]
            out["weather"] = {"air_c": w["air_temperature"], "track_c": w["track_temperature"],
                              "humidity_pct": w["humidity"], "wind_ms": w["wind_speed"],
                              "rain": bool(w["rainfall"])}
    res = client.session_result(session_key)
    if not res.empty:
        res = res.assign(pos=pd.to_numeric(res["position"], errors="coerce")).dropna(subset=["pos"])
        top = []
        for _, r in res.sort_values("pos").head(5).iterrows():
            d = r["duration"]
            times = [x for x in (d if isinstance(d, list) else [d]) if x]
            di = info.get(int(r["driver_number"]))
            top.append({
                "pos": int(r["pos"]),
                "acronym": str(di["name_acronym"]) if di is not None else str(r["driver_number"]),
                "colour": f"#{di['team_colour']}" if di is not None and di["team_colour"] else "#9aa4b5",
                "time": float(min(times)) if times else None,
            })
        out["top5"] = top
    return out


def export_web(cfg: dict[str, Any], out: Path) -> Path:
    """Write the web app's data files; returns the data directory."""
    year = int(cfg["season"]["year"])
    season_dir = Path(cfg["paths"]["season_dir"])
    data_dir = Path(out) / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    client = openf1_client(cfg)
    meetings = {m["circuit_key"]: m for m in client.get("meetings", [("year", year)])
                if "Testing" not in m.get("meeting_name", "")}
    theta = json.loads((Path(cfg["paths"]["models_dir"]) / f"theta_{year}.json").read_text())
    frames = int(cfg.get("web", {}).get("history_frames", 36))

    races = []
    for slug, cc in season_circuits(cfg).items():
        d = season_dir / slug
        if not (d / "meta.json").exists():
            continue
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        tr = pd.read_parquet(d / "track.parquet")
        ln = pd.read_parquet(d / "lines.parquet")
        hist = np.load(d / "history.npz")
        laps = sorted((Path(cfg["paths"]["data_interim"]) / slug).glob(f"{year}_*_Q_*.parquet"))
        session_key = int(pd.read_parquet(laps[0], columns=["SessionKey"])["SessionKey"].iloc[0])
        pole = _pole_driver(client, session_key, meta["pole_driver"])
        mt = meetings.get(int(cc["openf1_key"]), {})
        lt = meta["lap_times_s"]
        rot = float(meta.get("rotation_deg") or 0.0)

        a_opt, a_real = ln["alpha_mintime"].to_numpy(), ln["alpha_real"].to_numpy()
        v_opt, v_real = ln["v_mintime"].to_numpy(), ln["v_real"].to_numpy()
        t_opt = _elapsed(v_opt, *_line_xy(tr, a_opt))
        pole_df = pd.read_parquet(laps_sorted[0]) if (laps_sorted := sorted(
            laps, key=lambda f: float(pd.read_parquet(f, columns=["Official_LapTime_s"])
                                      ["Official_LapTime_s"].iloc[0]))) else None
        t_real = _real_time_on_grid(pole_df, tr, float(meta["length_m"]), lt["real_pole"])

        # Optimizer history as "best line found so far": keep the steps that set a new best
        # QSS lap time (IPOPT passes through infeasible, slower points on its way), so the
        # replay only ever improves; every kept frame is a real computed line.
        stages = [str(s) for s in hist["stage"]]
        laps_h = hist["lap_time_s"]
        keep, best = [0], laps_h[0]
        for i in range(1, len(stages) - 1):
            if laps_h[i] < best - 0.01:
                keep.append(i)
                best = laps_h[i]
        keep.append(len(stages) - 1)
        if len(keep) > frames:
            idx = np.round(np.linspace(0, len(keep) - 1, frames)).astype(int)
            keep = [keep[i] for i in sorted(set(idx))]
        extras = _session_extras(client, session_key, meta["pole_driver"], lt["real_pole"])
        s_grid = tr["s"].to_numpy()
        sectors = None
        if extras.get("sectors") and all(extras["sectors"]):
            b1 = float(np.interp(extras["sectors"][0], t_real, s_grid))
            b2 = float(np.interp(extras["sectors"][0] + extras["sectors"][1], t_real, s_grid))
            m1, m2 = float(np.interp(b1, s_grid, t_opt)), float(np.interp(b2, s_grid, t_opt))
            sectors = {"bounds_m": [round(b1, 1), round(b2, 1)],
                       "pole": [round(float(x), 3) for x in extras["sectors"]],
                       "model": [round(m1, 3), round(m2 - m1, 3), round(lt["min_time"] - m2, 3)]}
        xo, yo = _line_xy(tr, a_opt)
        k_opt = menger_curvature(xo, yo)
        z = ln["z_real"].to_numpy() if "z_real" in ln.columns else np.zeros(len(ln))
        race = {
            "slug": slug, "rotation": rot, "L": round(meta["length_m"], 1),
            "params": {k: meta["params"][k] for k in ("mu", "mu_x_brake", "mu_x_drive", "c_l",
                                                       "c_d", "p_spec", "a_brake_cap", "g")},
            "sectors": sectors,
            "z": _q(z - z.min(), 1),
            "track": {
                "x": _q(tr["x"], 1), "y": _q(tr["y"], 1), "nx": _q(tr["nx"], 3),
                "ny": _q(tr["ny"], 3), "wl": _q(tr["w_left"], 1), "wr": _q(tr["w_right"], 1),
                "k": _q(tr["kappa"], 4),
            },
            "opt": {"a": _q(a_opt, 2), "v": _q(v_opt, 1), "t": _q(t_opt, 3),
                    "ax": _q(ln["a_long_mintime"], 1),
                    "ay": _q(ln["a_lat_mintime"] * np.sign(k_opt), 1)},
            "real": {"a": _q(a_real, 2), "v": _q(v_real, 1), "t": _q(t_real, 3),
                     # -1 marks grid points where the car-data feed froze (pedals unknown)
                     "thr": _q(ln["throttle_real"].fillna(-1), 0),
                     "brk": _q(ln["brake_real"].fillna(-1), 0)},
            "mc": {"a": _q(ln["alpha_mincurv"], 2), "v": _q(ln["v_mincurv"], 1)},
            "stages": {"corner": _q(ln["v_corner"], 1), "accel": _q(ln["v_accel"], 1),
                       "brake": _q(ln["v_brake"], 1)},
            "hist": {
                "a": [_q(hist["alpha"][i], 2) for i in keep],
                "v": [_q(hist["v"][i], 1) for i in keep],
                "lap": _q(hist["lap_time_s"][keep], 3),
                "stage": [stages[i].split(" (")[0] for i in keep],
            },
            "corners": [{"label": c["label"], "s": round(c["s"], 1)} for c in meta["corners"]],
        }
        (data_dir / f"{slug}.json").write_text(json.dumps(race, separators=(",", ":"), default=_jsonable),
                                               encoding="utf-8")
        races.append({
            "slug": slug, "round": cc.get("round"), "code": cc.get("code", slug[:3].upper()),
            "event": mt.get("meeting_name", cc.get("fastf1_event", "")),
            "circuit": cc.get("name", slug), "country": mt.get("country_name", ""),
            "date": str(mt.get("date_start", ""))[:10],
            "length_m": round(meta["length_m"]), "geometry": meta["geometry"]["source"],
            "pole": {**pole, "acronym": meta["pole_driver"], "time": lt["real_pole"]},
            "model": {"time": lt["min_time"], "physics_only": lt.get("min_time_loco"),
                      "min_curvature": lt["min_curvature"], "centerline": lt["centerline"],
                      "correlation": theta["diagnostics"].get("correlation", {}).get(slug)},
            "extras": {k: v for k, v in extras.items() if k != "sectors"},
            "top_speed_kmh": round(float(v_opt.max()) * 3.6),
            "outline": _outline(tr, rot),
        })

    races.sort(key=lambda r: r["round"] or 99)
    gaps = np.array([r["model"]["time"] / r["pole"]["time"] - 1 for r in races]) * 100
    phys = np.array([(r["model"]["physics_only"] or r["model"]["time"]) / r["pole"]["time"] - 1
                     for r in races]) * 100
    index = {
        "season": year,
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "grip_utilization": round(theta["diagnostics"]["grip_utilization"], 3),
        "grip_by_group": {k: round(v, 3) for k, v in
                          (theta["diagnostics"].get("grip_utilization_by_group") or {}).items()},
        "per_circuit_setup": bool(theta["diagnostics"].get("setup")),
        "laps_used": theta["diagnostics"].get("n_laps"),
        "stats": {"races": len(races), "mean_gap_pct": round(float(gaps.mean()), 2),
                  "mean_abs_gap_pct": round(float(np.abs(gaps).mean()), 2),
                  "median_abs_gap_pct": round(float(np.median(np.abs(gaps))), 2),
                  "physics_mean_abs_gap_pct": round(float(np.abs(phys).mean()), 2),
                  "physics_median_abs_gap_pct": round(float(np.median(np.abs(phys))), 2)},
        "races": races,
    }
    (data_dir / "index.json").write_text(json.dumps(index, separators=(",", ":"), default=_jsonable),
                                         encoding="utf-8")
    return data_dir
