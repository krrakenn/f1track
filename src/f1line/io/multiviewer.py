"""Corner numbers and map rotation from the MultiViewer circuit API (the source FastF1 uses).

``GET https://api.multiviewer.app/api/v1/circuits/<circuit_key>/<year>`` returns corner
positions in the F1 timing frame (1/10 m) and the rotation [deg] used by official maps.
Responses are cached as JSON.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import requests

logger = logging.getLogger(__name__)

_URL = "https://api.multiviewer.app/api/v1/circuits/{key}/{year}"


def circuit_info(key: int, years: list[int], cache_dir: Path) -> Optional[dict[str, Any]]:
    """Return circuit info for the first year that exists, or None.

    Returns
    -------
    dict or None
        ``{"year", "rotation_deg", "corners": [{"number", "letter", "x_m", "y_m"}]}``.
    """
    cache_dir = Path(cache_dir)
    for year in years:
        path = cache_dir / f"{key}_{year}.json"
        if path.exists():
            raw = json.loads(path.read_text(encoding="utf-8"))
        else:
            try:
                resp = requests.get(_URL.format(key=key, year=year), timeout=30,
                                    headers={"User-Agent": "f1line-research/0.1"})
            except requests.RequestException as exc:
                logger.warning("MultiViewer %s/%s failed: %s", key, year, exc)
                continue
            raw = resp.json() if resp.status_code == 200 else {}
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(raw), encoding="utf-8")
        if not raw or "corners" not in raw:
            continue
        return {
            "year": year,
            "rotation_deg": float(raw.get("rotation", 0.0)),
            "corners": [
                {
                    "number": int(c["number"]),
                    "letter": str(c.get("letter") or ""),
                    "x_m": float(c["trackPosition"]["x"]) / 10.0,
                    "y_m": float(c["trackPosition"]["y"]) / 10.0,
                }
                for c in raw["corners"]
            ],
        }
    return None
