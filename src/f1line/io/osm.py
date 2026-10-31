"""OpenStreetMap circuit geometry via the Overpass API.

Used for 2026 circuits that are missing from (or outdated in) the TUM racetrack-database.
Fetches, in a bounding box around the circuit:

- every ``highway=raceway`` way (with geometry), and
- every ``type=circuit`` relation (with member geometry). Circuit relations group the ways
  of one layout (e.g. "Grand Prix" vs "National"), which removes karting tracks, pit
  lanes and alternative layouts from consideration.

The caller (``geometry.circuit``) picks the relation whose length matches the driven lap,
falling back to all non-auxiliary raceway ways, and then matches the driven lap onto it.

Data (c) OpenStreetMap contributors, ODbL.
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import requests

logger = logging.getLogger(__name__)

_EARTH_RADIUS_M = 6_371_000.0
_USER_AGENT = "f1line-research/0.1 (educational)"
# Ways whose name/tags match this are not part of the racing surface.
_AUX_PATTERN = re.compile(r"pit|long lap|paddock|service|access|karting|kart|drag", re.I)
_CACHE_VERSION = "v3"


def fetch_osm_circuit(
    name: str,
    lat: float,
    lon: float,
    cache_dir: Path,
    endpoints: list[str],
    bbox_half_deg: float = 0.03,
    max_rounds: int = 8,
    timeout_s: float = 60.0,
) -> dict[str, list[dict[str, Any]]]:
    """Fetch raceway ways and circuit relations (with geometry) around (lat, lon).

    Parameters
    ----------
    name:
        Circuit slug, used for the cache file name.
    lat, lon:
        Approximate circuit centre [deg].
    cache_dir:
        Directory for the cached Overpass JSON.
    endpoints:
        Overpass interpreter URLs, tried in order (public servers are often at capacity).
    bbox_half_deg:
        Half-size of the query box [deg].
    max_rounds:
        Passes over all endpoints before giving up.
    timeout_s:
        HTTP timeout per request [s].

    Returns
    -------
    dict
        ``{"ways": [...], "relations": [...]}`` – Overpass elements. Way geometry is a
        list of {lat, lon}; relation members of type ``way`` carry their own geometry.
    """
    cache_path = Path(cache_dir) / f"{name}_{_CACHE_VERSION}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    d = bbox_half_deg
    bbox = f"{lat - d},{lon - d},{lat + d},{lon + d}"
    query = (
        f"[out:json][timeout:45];"
        # "out geom" (not "out tags geom"): relations need their member lists.
        f"(way[highway=raceway]({bbox});relation[type=circuit]({bbox}););out geom;"
    )
    for rnd in range(max_rounds):
        for url in endpoints:
            try:
                resp = requests.post(
                    url, data={"data": query}, headers={"User-Agent": _USER_AGENT},
                    timeout=timeout_s,
                )
            except requests.RequestException as exc:
                logger.warning("Overpass %s failed: %s", url, exc)
                continue
            if resp.status_code == 200:
                els = resp.json().get("elements", [])
                data = {
                    "ways": [e for e in els if e["type"] == "way"],
                    "relations": [e for e in els if e["type"] == "relation"],
                }
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(json.dumps(data), encoding="utf-8")
                logger.info(
                    "OSM %s: %d raceway ways, %d circuit relations (%s)",
                    name, len(data["ways"]), len(data["relations"]), url,
                )
                return data
            logger.warning("Overpass %s answered %s", url, resp.status_code)
            time.sleep(3.0)
        time.sleep(10.0 * (rnd + 1))
    raise RuntimeError(f"Overpass unavailable for {name} after {max_rounds} rounds")


def is_auxiliary(way: dict[str, Any]) -> bool:
    """True if a raceway way is a pit lane, long-lap loop, or similar non-racing surface."""
    tags = way.get("tags", {})
    if tags.get("raceway") in {"pitlane", "pit_lane", "pits"} or tags.get("service"):
        return True
    text = " ".join(str(tags.get(k, "")) for k in ("name", "raceway", "description", "note"))
    return bool(_AUX_PATTERN.search(text))


def _geom_to_xy(geom: list[dict[str, float]], lat0: float, lon0: float) -> np.ndarray:
    k_lat = np.pi / 180.0 * _EARTH_RADIUS_M
    k_lon = k_lat * np.cos(np.radians(lat0))
    la = np.array([g["lat"] for g in geom])
    lo = np.array([g["lon"] for g in geom])
    return np.column_stack([(lo - lon0) * k_lon, (la - lat0) * k_lat])


def ways_to_local_xy(
    ways: list[dict[str, Any]], lat0: float, lon0: float
) -> list[np.ndarray]:
    """Project way geometries to a local tangent plane around (lat0, lon0).

    Returns
    -------
    list of np.ndarray
        One (K, 2) array of [east_m, north_m] per way.
    """
    return [
        _geom_to_xy(w["geometry"], lat0, lon0)
        for w in ways
        if len(w.get("geometry") or []) >= 2
    ]


def relation_polylines(
    relation: dict[str, Any], lat0: float, lon0: float
) -> list[np.ndarray]:
    """Member-way polylines of a circuit relation (non-way and auxiliary roles skipped)."""
    out = []
    for m in relation.get("members", []):
        if m.get("type") != "way" or len(m.get("geometry") or []) < 2:
            continue
        if _AUX_PATTERN.search(str(m.get("role", ""))):
            continue
        out.append(_geom_to_xy([g for g in m["geometry"] if g], lat0, lon0))
    return out


def polyline_length(polylines: list[np.ndarray]) -> float:
    """Total length of a set of polylines [m]."""
    return float(sum(np.hypot(*np.diff(p, axis=0).T).sum() for p in polylines))
