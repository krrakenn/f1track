"""End-to-end season pipeline: geometry -> calibration -> optimization -> artifacts for the app.

Per circuit, ``data/season/<circuit>/`` receives:

- ``track.parquet``   reference line, normals, curvature, admissible offsets (fine grid)
- ``lines.parquet``   centerline / min-curvature / min-time offsets and their QSS speeds,
                      plus the QSS stages (corner limit, acceleration, braking) of the
                      min-time line and the real pole lap (offset, speed, throttle, brake)
- ``history.npz``     optimizer iterates (line + lap time + speed) for the replay
- ``meta.json``       provenance, diagnostics, lap times, corners, solver info

Season-level: ``models/theta_<year>.json`` and ``reports/season_<year>.csv``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from f1line.geometry.align import project_to_track
from f1line.geometry.circuit import build_circuit_geometry
from f1line.geometry.track import Track
from f1line.io.multiviewer import circuit_info
from f1line.io.openf1 import frozen_seconds
from f1line.io.tum_tracks import _TRACK_COLS, _parse_tum_csv
from f1line.optimize.min_curvature import solve_min_curvature, sweep_lambda_and_select
from f1line.optimize.min_time import solve_min_time
from f1line.season import load_laps, osm_candidates
from f1line.validate.metrics import lap_time_error_pct, lateral_deviation, speed_rmse_kmh
from f1line.vehicle.calibrate import (
    calibrate,
    lap_on_track,
    prepare_lap,
    save_theta,
)
from f1line.vehicle.model import VehicleParams
from f1line.vehicle.qss import (
    menger_curvature,
    simulate_lap,
    simulate_lap_on_line,
    speed_profile_stages,
)

logger = logging.getLogger(__name__)


def season_dir(cfg: dict[str, Any], circuit: str) -> Path:
    """Artifact directory of one circuit."""
    return Path(cfg["paths"]["season_dir"]) / circuit


def _json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=float), encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# Geometry
# ─────────────────────────────────────────────────────────────────────────────


def build_geometry(circuit: str, ccfg: dict[str, Any], cfg: dict[str, Any]) -> tuple[Track, dict]:
    """Build (or load cached) track geometry of a circuit on the fine optimization grid."""
    out = season_dir(cfg, circuit)
    laps = [df for _, df in load_laps(cfg, circuit)]
    if not laps:
        raise RuntimeError(f"{circuit}: no laps; run fetch-season first")

    tum_df = None
    if ccfg.get("tum_file"):
        p = Path(cfg["paths"]["tum_repo"]) / "tracks" / ccfg["tum_file"]
        if p.exists():
            tum_df = _parse_tum_csv(p.read_text(encoding="utf-8"), _TRACK_COLS)
    lap_len = float(laps[0]["Distance"].iloc[-1])
    try:
        cands = osm_candidates(circuit, ccfg, cfg, lap_len)
    except RuntimeError as exc:
        logger.warning("%s: OSM unavailable (%s)", circuit, exc)
        cands = []
    geom = build_circuit_geometry(circuit, laps, cfg, ccfg, tum_df=tum_df, osm_candidates=cands)
    track = geom.track.resampled(float(cfg["optimize"].get("fine_ds_m", 4.0)))
    diag = {"source": geom.source, **geom.diagnostics}
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "s": track.s, "x": track.x, "y": track.y, "nx": track.nx, "ny": track.ny,
        "kappa": track.kappa, "w_left": track.w_left, "w_right": track.w_right,
    }).to_parquet(out / "track.parquet", index=False)
    _json(out / "geometry.json", diag)
    logger.info("%s: geometry from %s, %.0f m", circuit, geom.source, track.total_length)
    return track, diag


# ─────────────────────────────────────────────────────────────────────────────
# Calibration
# ─────────────────────────────────────────────────────────────────────────────


def scale_grip(p: VehicleParams, gamma: float) -> VehicleParams:
    """Apply a grip-utilization factor to the tyre friction coefficients."""
    return replace(p, mu=p.mu * gamma, mu_x_brake=p.mu_x_brake * gamma,
                   mu_x_drive=p.mu_x_drive * gamma)


def _gamma_for(table: pd.DataFrame, circuits: list[str]) -> float:
    """gamma at which mean log(T_opt / T_pole) over ``circuits`` is zero.

    Each circuit's log ratio is interpolated over its converged (IPOPT) grid points onto a
    fine gamma grid; points where the solver did not converge are left out.
    """
    sub = table[table["circuit"].isin(circuits) & (table["method"] == "ipopt")]
    fine = np.linspace(table["gamma"].min(), table["gamma"].max(), 401)
    curves = []
    for _, grp in sub.groupby("circuit"):
        grp = grp.sort_values("gamma")
        if len(grp) >= 2:
            curves.append(np.interp(fine, grp["gamma"], np.log(grp["t_opt_s"] / grp["t_pole_s"])))
    mean_log = np.mean(curves, axis=0)
    # mean_log decreases with gamma; interpolate the zero crossing.
    if mean_log[0] < 0 or mean_log[-1] > 0:
        logger.warning("gamma grid does not bracket the pole times: %s", mean_log[[0, -1]])
    return float(np.interp(0.0, mean_log[::-1], fine[::-1]))


_SETUP_PARAMS = ["c_l", "c_d"]
_PARAM_KEYS = ("mu", "mu_x_brake", "mu_x_drive", "c_l", "c_d", "p_spec", "a_brake_cap")


def circuit_types(cfg: dict[str, Any]) -> dict[int, str]:
    """Official circuit type per OpenF1 circuit key (Permanent / Temporary - Street / - Road)."""
    from f1line.season import openf1_client

    meetings = openf1_client(cfg).get("meetings", [("year", int(cfg["season"]["year"]))])
    return {int(m["circuit_key"]): str(m.get("circuit_type") or "Permanent") for m in meetings}


def grip_group(circuit_type: str) -> str:
    """Grip group from the official circuit type: street circuits vs everything else."""
    return "street" if "Street" in circuit_type else "permanent"


def _params_dict(p: VehicleParams) -> dict[str, float]:
    return {k: float(getattr(p, k)) for k in _PARAM_KEYS}


def calibrate_season(
    circuits: dict[str, dict[str, Any]], tracks: dict[str, Track], cfg: dict[str, Any]
) -> tuple[dict[str, VehicleParams], dict[str, Any]]:
    """Calibrate the season car per circuit: telemetry fit, weekend setup, grip by circuit type.

    1. Fit the season car theta on the qualifying laps of every circuit raced this season.
    2. Weekend setup: refit downforce c_l and drag c_d on the laps of that weekend's last
       session *before* qualifying (practice or sprint qualifying), the other parameters held
       at theta. Teams trim wing level per circuit; practice is known before qualifying.
    3. Grip utilization gamma per official circuit type group (street vs other): for gamma on
       a grid, solve the min-time NLP (coarse grid) and pick gamma so the group's mean
       log(T_opt / T_pole) is zero. Leave-one-circuit-out gammas (within the group) give the
       held-out pole predictions.

    Returns
    -------
    (params_by_circuit, diagnostics)
    """
    from f1line.season import load_practice_laps, practice_kind

    year = int(cfg["season"]["year"])
    ccal = cfg.get("calibration", {})
    calib_laps, poles = [], {}
    for name, track in tracks.items():
        laps = load_laps(cfg, name)
        if not laps or int(laps[0][1]["Year"].iloc[0]) != year:
            continue
        poles[name] = float(laps[0][1]["Official_LapTime_s"].iloc[0])
        for drv, df in laps:
            if frozen_seconds(df) > 2.0:
                logger.info("%s/%s: car data frozen %.1f s, excluded from calibration",
                            name, drv, frozen_seconds(df))
                continue
            calib_laps.append(prepare_lap(df, f"{name}/{drv}", name, track))
    logger.info("Calibrating on %d laps from %d circuits", len(calib_laps), len(poles))
    base, diag = calibrate(calib_laps, cfg)
    base_dict = diag["final_params"]

    # 2. Weekend setup from pre-qualifying laps
    setup: dict[str, VehicleParams] = {}
    setup_info: dict[str, Any] = {}
    for name in poles:
        practice = load_practice_laps(cfg, name) if ccal.get("per_circuit_setup", True) else []
        if not practice:
            setup[name] = base
            setup_info[name] = {"source": "season car"}
            continue
        laps_p = [prepare_lap(df, f"{name}/{drv}/practice", name, tracks[name])
                  for drv, df in practice if frozen_seconds(df) <= 2.0]
        p_c, d_c = calibrate(laps_p, cfg, theta0=base_dict, free=_SETUP_PARAMS)
        setup[name] = p_c
        setup_info[name] = {"source": "practice", "n_laps": len(laps_p),
                            **{k: d_c["final_params"][k] for k in _SETUP_PARAMS}}
        logger.info("%s setup from practice: c_l %.5f (season %.5f), c_d %.5f (season %.5f)",
                    name, p_c.c_l, base.c_l, p_c.c_d, base.c_d)

    # 3. Grip utilization per official circuit-type group
    types = circuit_types(cfg)
    group = {c: (grip_group(types.get(int(circuits[c]["openf1_key"]), "Permanent"))
                 if ccal.get("grip_by_circuit_type", True) else "all") for c in poles}
    rows = []
    grid = [float(g) for g in cfg["vehicle"].get("grip_utilization_grid", [0.7, 0.85, 1.0])]
    opt_cfg = {**cfg, "optimize": {**cfg["optimize"], "coarse_ds_m": 0.0}}
    for name in poles:
        coarse = tracks[name].resampled(float(cfg["optimize"].get("coarse_ds_m", 10.0)))
        a0 = solve_min_curvature(coarse, lambda_reg=1e-4)
        for gamma in grid:
            a_opt, q, info = solve_min_time(coarse, scale_grip(setup[name], gamma), a0, opt_cfg)
            if info["method"] == "ipopt":
                a0 = a_opt  # warm start the next gamma from this solution
            rows.append({"circuit": name, "gamma": gamma, "t_opt_s": q.lap_time_s,
                         "t_pole_s": poles[name], "method": info["method"]})
            logger.info("%s gamma=%.2f: optimal %.3f s (pole %.3f s, %s)",
                        name, gamma, q.lap_time_s, poles[name], info["method"])
    table = pd.DataFrame(rows)
    members = {g: [c for c in poles if group[c] == g] for g in set(group.values())}
    gamma_by_group = {g: _gamma_for(table, cs) for g, cs in members.items()}
    loco = {c: _gamma_for(table, [o for o in members[group[c]] if o != c]) for c in poles}
    params = {c: scale_grip(setup[c], gamma_by_group[group[c]]) for c in poles}
    logger.info("Grip utilization by group: %s", gamma_by_group)

    # 4. Weekend correlation (as teams correlate their simulator to practice): the expected
    #    pole is the weekend's best pre-qualifying lap times the typical practice-to-
    #    qualifying ratio of the *other* races with the same session type (FP3 or sprint
    #    qualifying); grip is tuned per weekend so the optimal lap matches it.
    correlation: dict[str, Any] = {}
    if ccal.get("practice_correlation", True):
        best_pr, kind = {}, {}
        for c in poles:
            pr = load_practice_laps(cfg, c)
            if pr:
                best_pr[c] = float(pr[0][1]["Official_LapTime_s"].iloc[0])
                kind[c] = practice_kind(cfg, c)
        ratio = {c: poles[c] / best_pr[c] for c in best_pr}
        for c in best_pr:
            peers = [ratio[o] for o in best_pr if o != c and kind[o] == kind[c]]
            peers = peers or [ratio[o] for o in best_pr if o != c]
            target = best_pr[c] * float(np.median(peers))
            sub = table[(table["circuit"] == c) & (table["method"] == "ipopt")].sort_values("gamma")
            g_c = float(np.interp(np.log(target), np.log(sub["t_opt_s"].to_numpy())[::-1],
                                  sub["gamma"].to_numpy()[::-1]))
            params[c] = scale_grip(setup[c], g_c)
            correlation[c] = {"session": kind[c], "best_practice_s": best_pr[c],
                              "ratio_used": float(np.median(peers)), "target_s": target,
                              "gamma": g_c}
            logger.info("%s: practice %.3f s x %.4f -> target %.3f s (pole %.3f), gamma %.3f",
                        c, best_pr[c], np.median(peers), target, poles[c], g_c)

    diag.update(
        base_params=diag.pop("final_params"),
        grip_utilization=float(np.mean([gamma_by_group[group[c]] for c in poles])),
        grip_utilization_by_group=gamma_by_group,
        grip_group=group,
        circuit_type={c: types.get(int(circuits[c]["openf1_key"]), "?") for c in poles},
        grip_utilization_loco=loco,
        grip_table=rows,
        poles=poles,
        setup=setup_info,
        correlation=correlation,
        setup_params={c: _params_dict(p) for c, p in setup.items()},
        circuit_params={c: _params_dict(p) for c, p in params.items()},
        final_params=_params_dict(scale_grip(base, float(np.mean(list(gamma_by_group.values()))))),
    )
    save_theta(scale_grip(base, diag["grip_utilization"]), diag,
               Path(cfg["paths"]["models_dir"]) / f"theta_{year}.json")
    return params, diag


def params_from_diag(diag: dict[str, Any], fixed: VehicleParams) -> dict[str, VehicleParams]:
    """Per-circuit parameters stored by :func:`calibrate_season`."""
    extra = {"g": fixed.g, "v_min": fixed.v_min, "v_max": fixed.v_max}
    return {c: VehicleParams(**p, **extra) for c, p in diag.get("circuit_params", {}).items()}


# ─────────────────────────────────────────────────────────────────────────────
# Optimization + artifacts
# ─────────────────────────────────────────────────────────────────────────────


def _subsample_history(history: list[dict], k: int) -> list[dict]:
    if len(history) <= k:
        return history
    idx = np.unique(np.round(np.linspace(0, len(history) - 1, k)).astype(int))
    return [history[i] for i in idx]


def optimize_circuit(
    circuit: str,
    ccfg: dict[str, Any],
    track: Track,
    params: VehicleParams,
    cfg: dict[str, Any],
    calib_diag: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Run centerline / Stage A / Stage B on one circuit and write its artifacts."""
    out = season_dir(cfg, circuit)
    year = int(cfg["season"]["year"])

    zero = np.zeros(track.n)
    q_center = simulate_lap_on_line(track.x, track.y, params)
    history: list[dict] = [{"stage": "centerline", "iteration": 0, "alpha": zero,
                            "nlp_lap_time_s": q_center.lap_time_s}]

    a_mc, lam, _, sweep = sweep_lambda_and_select(
        track, params, [float(v) for v in cfg["optimize"]["lambda_grid"]]
    )
    for i, r in enumerate(sorted(sweep, key=lambda r: -r["lap_time_s"])):
        history.append({"stage": f"min-curvature (lambda={r['lambda']:g})", "iteration": i,
                        "alpha": r["alpha"], "nlp_lap_time_s": r["lap_time_s"]})
    q_mc = simulate_lap_on_line(*track.point_at(a_mc), params)

    nlp_hist: list[dict] = []
    a_mt, q_mt, info = solve_min_time(track, params, a_mc, cfg, history=nlp_hist)
    history += _subsample_history(nlp_hist, int(cfg["optimize"].get("history_frames", 60)))
    history.append({"stage": "optimal", "iteration": 0, "alpha": a_mt,
                    "nlp_lap_time_s": q_mt.lap_time_s})

    # QSS speed for every history frame (cheap), so the replay can show speed too.
    h_alpha = np.array([h["alpha"] for h in history])
    h_v, h_t = [], []
    for a in h_alpha:
        q = simulate_lap_on_line(*track.point_at(a), params)
        h_v.append(q.v)
        h_t.append(q.lap_time_s)
    np.savez_compressed(
        out / "history.npz",
        alpha=h_alpha.astype(np.float32), v=np.array(h_v, dtype=np.float32),
        lap_time_s=np.array(h_t), stage=np.array([h["stage"] for h in history]),
        iteration=np.array([h["iteration"] for h in history]),
    )

    px, py = track.point_at(a_mt)
    ds_line = np.hypot(np.roll(px, -1) - px, np.roll(py, -1) - py)
    stages = speed_profile_stages(menger_curvature(px, py), ds_line, params)
    t_mt = np.concatenate([[0.0], np.cumsum(2 * ds_line / (q_mt.v + np.roll(q_mt.v, -1)))[:-1]])

    # Real pole lap on the same grid.
    laps = load_laps(cfg, circuit)
    pole_drv, pole_df = laps[0]
    alpha_real, ch = lap_on_track(pole_df, track)
    pole_year = int(pole_df["Year"].iloc[0])
    pole_time = float(pole_df["Official_LapTime_s"].iloc[0])
    calib_real = prepare_lap(pole_df, f"{circuit}/{pole_drv}", circuit, track)
    replay = simulate_lap(calib_real.kappa, calib_real.ds, params)
    rx, ry = track.point_at(alpha_real)

    lines = pd.DataFrame({
        "s": track.s,
        "alpha_centerline": zero, "alpha_mincurv": a_mc, "alpha_mintime": a_mt,
        "v_centerline": q_center.v, "v_mincurv": q_mc.v, "v_mintime": q_mt.v,
        "a_long_mintime": q_mt.a_long, "a_lat_mintime": q_mt.a_lat,
        "t_mintime": t_mt,
        "v_corner": stages["v_corner"], "v_accel": stages["v_accel"],
        "v_brake": stages["v_brake"],
        "alpha_real": alpha_real, "v_real": ch["Speed_ms"], "throttle_real": ch["Throttle"],
        "brake_real": ch["Brake"].astype(float), "t_real": ch["Time_s"], "z_real": ch["Z_m"],
    })
    lines.to_parquet(out / "lines.parquet", index=False)

    # Corners (MultiViewer), projected onto the track.
    mv = circuit_info(int(ccfg["openf1_key"]), [year, pole_year, year - 1],
                      Path(cfg["paths"]["cache_dir"]) / "multiviewer")
    corners, rotation = [], 0.0
    if mv:
        rotation = mv["rotation_deg"]
        cx = np.array([c["x_m"] for c in mv["corners"]])
        cy = np.array([c["y_m"] for c in mv["corners"]])
        if len(cx):
            s_c, a_c = project_to_track(cx, cy, track)
            for c, sc, ac in zip(mv["corners"], s_c, a_c):
                if abs(ac) < 60:
                    corners.append({"label": f"T{c['number']}{c['letter']}", "s": float(sc),
                                    "x": c["x_m"], "y": c["y_m"]})

    dev = lateral_deviation(px, py, rx, ry)
    gamma_loco = (calib_diag or {}).get("grip_utilization_loco", {}).get(circuit)
    loco_time = None
    if gamma_loco is not None and calib_diag is not None:
        setup_p = calib_diag.get("setup_params", {}).get(circuit, calib_diag["base_params"])
        base = VehicleParams(**{**setup_p, "g": params.g, "v_min": params.v_min,
                                "v_max": params.v_max})
        coarse = track.resampled(float(cfg["optimize"].get("coarse_ds_m", 10.0)))
        a0 = np.interp(coarse.s / coarse.total_length, track.s / track.total_length, a_mt,
                       period=1.0)
        _, q_loco, _ = solve_min_time(coarse, scale_grip(base, gamma_loco), a0,
                                      {**cfg, "optimize": {**cfg["optimize"], "coarse_ds_m": 0}})
        loco_time = q_loco.lap_time_s

    meta = {
        "circuit": circuit,
        "name": ccfg.get("name", circuit),
        "round": ccfg.get("round"),
        "status": ccfg.get("status"),
        "season": year,
        "data_year": pole_year,
        "pole_driver": pole_drv,
        "pole_time_s": pole_time,
        "length_m": track.total_length,
        "ds_m": track.ds,
        "rotation_deg": rotation,
        "corners": corners,
        "geometry": json.loads((out / "geometry.json").read_text()),
        "lap_times_s": {
            "centerline": q_center.lap_time_s,
            "min_curvature": q_mc.lap_time_s,
            "min_time": q_mt.lap_time_s,
            "real_pole": pole_time,
            "replay_pole_line": replay.lap_time_s,
            "min_time_loco": loco_time,
        },
        "lambda_selected": lam,
        "solver": info,
        "metrics": {
            "optimal_vs_pole_pct": lap_time_error_pct(q_mt.lap_time_s, pole_time),
            "optimal_loco_vs_pole_pct": (
                lap_time_error_pct(loco_time, pole_time) if loco_time else None
            ),
            "replay_vs_pole_pct": lap_time_error_pct(replay.lap_time_s, pole_time),
            "replay_speed_rmse_kmh": speed_rmse_kmh(replay.v, calib_real.v),
            "optimal_vs_real_speed_rmse_kmh": speed_rmse_kmh(q_mt.v, ch["Speed_ms"]),
            "line_dev_mean_m": dev["mean_m"],
            "line_dev_max_m": dev["max_m"],
        },
        "params": {k: getattr(params, k) for k in
                   ("mu", "mu_x_brake", "mu_x_drive", "c_l", "c_d", "p_spec", "a_brake_cap",
                    "g", "v_min", "v_max")},
    }
    _json(out / "meta.json", meta)
    logger.info(
        "%s: centerline %.2f | min-curv %.2f | min-time %.2f | pole %.2f (%s %d) [%s]",
        circuit, q_center.lap_time_s, q_mc.lap_time_s, q_mt.lap_time_s, pole_time, pole_drv,
        pole_year, info["method"],
    )
    return meta


def season_report(metas: list[dict[str, Any]], cfg: dict[str, Any]) -> pd.DataFrame:
    """One row per circuit; written to reports/season_<year>.csv."""
    rows = []
    for m in metas:
        lt, mt = m["lap_times_s"], m["metrics"]
        rows.append({
            "round": m["round"], "circuit": m["circuit"], "status": m["status"],
            "geometry": m["geometry"]["source"], "data_year": m["data_year"],
            "length_m": round(m["length_m"]),
            "centerline_s": lt["centerline"], "min_curv_s": lt["min_curvature"],
            "min_time_s": lt["min_time"], "pole_s": lt["real_pole"],
            "pole_driver": m["pole_driver"],
            "gap_pct": mt["optimal_vs_pole_pct"], "gap_loco_pct": mt["optimal_loco_vs_pole_pct"],
            "replay_err_pct": mt["replay_vs_pole_pct"],
            "replay_rmse_kmh": mt["replay_speed_rmse_kmh"],
            "line_dev_mean_m": mt["line_dev_mean_m"], "line_dev_max_m": mt["line_dev_max_m"],
            "solver": m["solver"]["method"], "nlp_qss_diff_pct": m["solver"].get("nlp_qss_diff_pct"),
        })
    df = pd.DataFrame(rows).sort_values("round", na_position="last")
    out = Path("reports") / f"season_{cfg['season']['year']}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, float_format="%.3f")
    return df
