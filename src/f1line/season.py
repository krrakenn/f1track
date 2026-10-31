"""Season-level orchestration: data fetching and per-circuit artifact paths for the 2026 calendar."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from f1line.io.openf1 import OpenF1Client, find_session, load_session_laps_openf1
from f1line.io.osm import (
    fetch_osm_circuit,
    is_auxiliary,
    polyline_length,
    relation_polylines,
    ways_to_local_xy,
)

logger = logging.getLogger(__name__)


def season_circuits(cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Circuits of the calendar to process; only already-raced ones if ``completed_only``."""
    from f1line.config import load_circuits

    circuits = load_circuits()
    if cfg.get("season", {}).get("completed_only", False):
        circuits = {k: v for k, v in circuits.items() if v.get("status") == "raced"}
    return circuits


def openf1_client(cfg: dict[str, Any]) -> OpenF1Client:
    """Build an :class:`OpenF1Client` from the config."""
    o = cfg.get("openf1", {})
    return OpenF1Client(
        cache_dir=Path(cfg["paths"]["openf1_cache"]),
        base_url=o.get("base_url", "https://api.openf1.org/v1"),
        min_interval_s=float(o.get("min_interval_s", 0.5)),
        max_retries=int(o.get("max_retries", 6)),
    )


def resolve_session(
    client: OpenF1Client, ccfg: dict[str, Any], cfg: dict[str, Any]
) -> Optional[dict[str, Any]]:
    """Pick the qualifying session to use: the modelled season if held, else the fallback year."""
    s = cfg.get("season", {})
    name = s.get("session_name", "Qualifying")
    for year in (int(s.get("year", 2026)), int(s.get("fallback_year", 2025))):
        sess = find_session(client, year, int(ccfg["openf1_key"]), name)
        if sess is not None:
            return sess
    return None


def resolve_practice_session(
    client: OpenF1Client, ccfg: dict[str, Any], cfg: dict[str, Any]
) -> Optional[dict[str, Any]]:
    """Last practice or sprint-qualifying session before qualifying (same weekend).

    Its laps are recorded before qualifying, so per-weekend car setup fitted on them can be
    used to *predict* qualifying without looking at it.
    """
    quali = resolve_session(client, ccfg, cfg)
    if quali is None:
        return None
    sessions = client.sessions(int(quali["year"]))
    q_start = pd.to_datetime(quali["date_start"], utc=True)
    cand = sessions[
        (sessions["meeting_key"] == quali["meeting_key"])
        & (sessions["session_name"].str.startswith("Practice")
           | (sessions["session_name"] == "Sprint Qualifying"))
    ].copy()
    cand["start"] = pd.to_datetime(cand["date_start"], utc=True)
    cand = cand[(cand["start"] < q_start) & ~cand["is_cancelled"].fillna(False).astype(bool)]
    if cand.empty:
        return None
    return cand.sort_values("start").iloc[-1].to_dict()


def practice_dir(cfg: dict[str, Any], circuit: str) -> Path:
    """Directory holding the pre-qualifying (practice) laps of a circuit."""
    return interim_dir(cfg, circuit) / "practice"


def fetch_circuit_practice(
    circuit: str, ccfg: dict[str, Any], cfg: dict[str, Any], client: OpenF1Client
) -> dict[str, Any]:
    """Download the fastest laps of the last pre-qualifying session of a circuit."""
    info: dict[str, Any] = {"circuit": circuit}
    sess = resolve_practice_session(client, ccfg, cfg)
    if sess is None:
        info["error"] = "no practice session found"
        return info
    info.update(session=sess["session_name"], session_key=int(sess["session_key"]))
    out = practice_dir(cfg, circuit)
    if not list(out.glob("*.parquet")):
        laps = load_session_laps_openf1(
            client, sess, top_n_drivers=int(cfg.get("openf1", {}).get("top_n_drivers", 10))
        )
        out.mkdir(parents=True, exist_ok=True)
        tag = str(sess["session_name"]).replace(" ", "")
        for drv, df in laps:
            df["Year"] = int(sess["year"])
            df["SessionKey"] = int(sess["session_key"])
            df.to_parquet(out / f"{sess['year']}_{circuit}_{tag}_{drv}.parquet", index=False)
    info["n_laps"] = len(list(out.glob("*.parquet")))
    return info


def practice_kind(cfg: dict[str, Any], circuit: str) -> str:
    """"SQ" if the stored pre-qualifying session is a sprint qualifying, else "FP"."""
    files = list(practice_dir(cfg, circuit).glob("*.parquet"))
    return "SQ" if files and "SprintQualifying" in files[0].name else "FP"


def load_practice_laps(cfg: dict[str, Any], circuit: str) -> list[tuple[str, pd.DataFrame]]:
    """Stored pre-qualifying laps of a circuit, fastest first."""
    laps = [(f.stem.rsplit("_", 1)[-1], pd.read_parquet(f))
            for f in sorted(practice_dir(cfg, circuit).glob("*.parquet"))]
    laps.sort(key=lambda t: float(t[1]["Official_LapTime_s"].iloc[0]))
    return laps


def interim_dir(cfg: dict[str, Any], circuit: str) -> Path:
    """Directory holding the per-lap parquet files of a circuit."""
    return Path(cfg["paths"]["data_interim"]) / circuit


def fetch_circuit_laps(
    circuit: str, ccfg: dict[str, Any], cfg: dict[str, Any], client: OpenF1Client
) -> dict[str, Any]:
    """Download qualifying laps from OpenF1 for one circuit.

    Laps are written to ``data/interim/<circuit>/<year>_<circuit>_Q_<DRV>.parquet``.

    Returns
    -------
    dict
        Summary: session year/key and number of laps.
    """
    info: dict[str, Any] = {"circuit": circuit}
    sess = resolve_session(client, ccfg, cfg)
    if sess is None:
        info["error"] = "no qualifying session found"
        return info
    year = int(sess["year"])
    info.update(year=year, session_key=int(sess["session_key"]))

    out_dir = interim_dir(cfg, circuit)
    existing = sorted(out_dir.glob(f"{year}_{circuit}_Q_*.parquet"))
    if existing:
        info["n_laps"] = len(existing)
    else:
        top_n = int(cfg.get("openf1", {}).get("top_n_drivers", 10))
        laps = load_session_laps_openf1(client, sess, top_n_drivers=top_n)
        out_dir.mkdir(parents=True, exist_ok=True)
        for drv, df in laps:
            df["Year"] = year
            df["SessionKey"] = int(sess["session_key"])
            df.to_parquet(out_dir / f"{year}_{circuit}_Q_{drv}.parquet", index=False)
        info["n_laps"] = len(laps)
    return info


def fetch_circuit_osm(circuit: str, ccfg: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    """Download (or load cached) OSM raceway ways for one circuit."""
    info: dict[str, Any] = {"circuit": circuit}
    try:
        data = _osm(circuit, ccfg, cfg)
        info["n_osm_ways"] = len(data["ways"])
        info["n_osm_relations"] = len(data["relations"])
    except RuntimeError as exc:
        logger.warning("%s: %s", circuit, exc)
        info["n_osm_ways"] = 0
    return info


def _osm(circuit: str, ccfg: dict[str, Any], cfg: dict[str, Any]) -> dict[str, list]:
    osm_cfg = cfg.get("osm", {})
    return fetch_osm_circuit(
        circuit,
        float(ccfg["lat"]),
        float(ccfg["lon"]),
        Path(cfg["paths"]["osm_cache"]),
        endpoints=list(osm_cfg.get("endpoints", [])),
        bbox_half_deg=float(osm_cfg.get("bbox_half_deg", 0.03)),
    )


def osm_candidates(
    circuit: str, ccfg: dict[str, Any], cfg: dict[str, Any], lap_length_m: float
) -> list[tuple[str, list[np.ndarray]]]:
    """Ordered OSM geometry candidates for a circuit.

    Circuit relations whose total length is within 25 % of the driven lap come first
    (closest length first), then all non-auxiliary raceway ways in the area.
    """
    data = _osm(circuit, ccfg, cfg)
    lat0, lon0 = float(ccfg["lat"]), float(ccfg["lon"])
    rels = []
    for r in data["relations"]:
        polys = relation_polylines(r, lat0, lon0)
        if not polys:
            continue
        L = polyline_length(polys)
        if abs(L / lap_length_m - 1.0) < 0.25:
            label = f"relation {r['id']} {r.get('tags', {}).get('name', '')!s} ({L:.0f} m)"
            rels.append((abs(L - lap_length_m), label, polys))
    rels.sort(key=lambda t: t[0])
    cands = [(label, polys) for _, label, polys in rels]
    ways = [w for w in data["ways"] if not is_auxiliary(w)]
    if ways:
        cands.append(("all raceway ways", ways_to_local_xy(ways, lat0, lon0)))
    return cands


def load_laps(cfg: dict[str, Any], circuit: str) -> list[tuple[str, pd.DataFrame]]:
    """Load stored laps of a circuit (newest year only), fastest first."""
    files = sorted(interim_dir(cfg, circuit).glob(f"*_{circuit}_Q_*.parquet"))
    if not files:
        return []
    year = max(int(f.name.split("_")[0]) for f in files)
    laps = []
    for f in files:
        if int(f.name.split("_")[0]) != year:
            continue
        df = pd.read_parquet(f)
        if "Official_LapTime_s" not in df.columns:
            df["Official_LapTime_s"] = float(df["Time_s"].max() - df["Time_s"].min())
        laps.append((f.stem.rsplit("_", 1)[-1], df))
    laps.sort(key=lambda t: float(t[1]["Official_LapTime_s"].iloc[0]))
    return laps
