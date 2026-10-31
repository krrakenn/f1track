"""OpenF1 telemetry loader (https://openf1.org), behind the same interface as the FastF1 loader.

FastF1's upstream (livetiming.formula1.com/static) answers HTTP 401 since 2026, so OpenF1
is the primary source for new sessions. OpenF1 serves the same F1 feed:

- ``location``: x, y, z in 1/10 m at ~3.7 Hz (same frame as FastF1 X/Y/Z).
- ``car_data``: speed [km/h], throttle [%], brake [0/100], n_gear, rpm, drs at ~3.7 Hz.
- ``laps``: lap start timestamps and durations [s].

Every HTTP response is cached as JSON under ``cache_dir`` so nothing is downloaded twice.
Output laps use the SI column layout of :mod:`f1line.io.telemetry`.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import urllib.parse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import requests

from f1line.io.telemetry import _km_h_to_m_s, _tenth_m_to_m

logger = logging.getLogger(__name__)

_USER_AGENT = "f1line-research/0.1 (educational; github.com/TUMFTM/racetrack-database user)"


@dataclass
class OpenF1Client:
    """Minimal cached, throttled OpenF1 REST client.

    Attributes
    ----------
    cache_dir:
        Directory for cached JSON responses.
    base_url:
        API root, e.g. ``https://api.openf1.org/v1``.
    min_interval_s:
        Minimum time between two network requests [s].
    max_retries:
        Attempts per request (exponential back-off on 429/5xx/network errors).
    """

    cache_dir: Path
    base_url: str = "https://api.openf1.org/v1"
    min_interval_s: float = 0.5
    max_retries: int = 6
    _last_request: float = 0.0

    def get(self, endpoint: str, params: list[tuple[str, Any]]) -> list[dict[str, Any]]:
        """GET ``endpoint`` with ``params`` (list of (key, value); keys may carry ``>=``/``<=``).

        Returns the decoded JSON list, from cache when available.
        """
        query = "&".join(f"{k}={urllib.parse.quote(str(v), safe=':+-.')}" for k, v in params)
        url = f"{self.base_url}/{endpoint}?{query}"
        cache_path = Path(self.cache_dir) / endpoint / (
            hashlib.sha1(url.encode()).hexdigest()[:20] + ".json"
        )
        if cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))

        for attempt in range(self.max_retries):
            wait = self.min_interval_s - (time.time() - self._last_request)
            if wait > 0:
                time.sleep(wait)
            self._last_request = time.time()
            try:
                resp = requests.get(url, headers={"User-Agent": _USER_AGENT}, timeout=120)
            except requests.RequestException as exc:
                status, detail = None, str(exc)
            else:
                if resp.status_code == 200:
                    data = resp.json()
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    cache_path.write_text(json.dumps(data), encoding="utf-8")
                    return data
                if resp.status_code == 404:
                    # OpenF1 answers 404 for "no rows match"; cache the empty result.
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    cache_path.write_text("[]", encoding="utf-8")
                    return []
                status, detail = resp.status_code, resp.text[:200]
            backoff = 2.0 * (2**attempt)
            logger.warning(
                "OpenF1 %s failed (status=%s: %s); retry %d/%d in %.0f s",
                endpoint, status, detail, attempt + 1, self.max_retries, backoff,
            )
            time.sleep(backoff)
        raise RuntimeError(f"OpenF1 request failed after {self.max_retries} attempts: {url}")

    # ── Convenience wrappers ────────────────────────────────────────────────

    def sessions(self, year: int) -> pd.DataFrame:
        """All sessions of a season (one row per session)."""
        return pd.DataFrame(self.get("sessions", [("year", year)]))

    def laps(self, session_key: int) -> pd.DataFrame:
        """All laps of a session."""
        return pd.DataFrame(self.get("laps", [("session_key", session_key)]))

    def drivers(self, session_key: int) -> pd.DataFrame:
        """Driver list (number -> acronym, team) of a session."""
        return pd.DataFrame(self.get("drivers", [("session_key", session_key)]))

    def session_result(self, session_key: int) -> pd.DataFrame:
        """Classified results of a session (may be empty for older seasons)."""
        return pd.DataFrame(self.get("session_result", [("session_key", session_key)]))

    def _window(
        self, endpoint: str, session_key: int, driver: int, t0: datetime, t1: datetime
    ) -> pd.DataFrame:
        fmt = "%Y-%m-%dT%H:%M:%S.%f"
        params = [
            ("session_key", session_key),
            ("driver_number", driver),
            ("date>", t0.astimezone(timezone.utc).strftime(fmt)),
            ("date<", t1.astimezone(timezone.utc).strftime(fmt)),
        ]
        df = pd.DataFrame(self.get(endpoint, params))
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"], utc=True, format="ISO8601")
            df = df.sort_values("date").reset_index(drop=True)
        return df

    def location(self, session_key: int, driver: int, t0: datetime, t1: datetime) -> pd.DataFrame:
        """Position samples (x, y, z in 1/10 m) of one driver in a time window."""
        return self._window("location", session_key, driver, t0, t1)

    def car_data(self, session_key: int, driver: int, t0: datetime, t1: datetime) -> pd.DataFrame:
        """Car channel samples of one driver in a time window."""
        return self._window("car_data", session_key, driver, t0, t1)


def find_session(
    client: OpenF1Client, year: int, circuit_key: int, session_name: str = "Qualifying"
) -> Optional[dict[str, Any]]:
    """Return the session dict for ``circuit_key`` in ``year``, or None if absent/cancelled/future."""
    sessions = client.sessions(year)
    if sessions.empty:
        return None
    rows = sessions[
        (sessions["circuit_key"] == circuit_key) & (sessions["session_name"] == session_name)
    ]
    now = datetime.now(timezone.utc)
    for _, row in rows.iterrows():
        if row.get("is_cancelled"):
            continue
        if pd.to_datetime(row["date_end"], utc=True) > now:
            continue
        return row.to_dict()
    return None


def _merge_lap(loc: pd.DataFrame, car: pd.DataFrame, t0: datetime) -> pd.DataFrame:
    """Merge position and car channels onto the car-data timestamps; convert to SI."""
    t_car = (car["date"] - pd.Timestamp(t0)).dt.total_seconds().to_numpy()
    t_loc = (loc["date"] - pd.Timestamp(t0)).dt.total_seconds().to_numpy()
    out = pd.DataFrame()
    out["Time_s"] = t_car
    out["Speed_ms"] = _km_h_to_m_s(car["speed"].astype(float)).to_numpy()
    # Distance [m]: integrate speed over time (trapezoid), as FastF1's add_distance does.
    dt = np.diff(t_car, prepend=t_car[0])
    v = out["Speed_ms"].to_numpy()
    out["Distance"] = np.cumsum(0.5 * (v + np.roll(v, 1)) * dt)
    out["RPM"] = car["rpm"].astype(float).to_numpy()
    out["nGear"] = car["n_gear"].astype(float).to_numpy()
    out["Throttle"] = car["throttle"].astype(float).clip(0, 100).to_numpy()
    out["Brake"] = (car["brake"].astype(float) > 0).to_numpy()
    drs = car["drs"] if "drs" in car.columns else pd.Series(np.nan, index=car.index)
    out["DRS"] = pd.to_numeric(drs, errors="coerce").to_numpy()
    for col, src in (("X_m", "x"), ("Y_m", "y"), ("Z_m", "z")):
        out[col] = _tenth_m_to_m(pd.Series(np.interp(t_car, t_loc, loc[src].astype(float))))
    out["Source"] = "openf1"
    return out


def frozen_mask(df: pd.DataFrame, min_s: float = 1.5) -> np.ndarray:
    """Samples inside stretches where the car-data feed froze (repeated values).

    The feed occasionally repeats the last speed/throttle/brake reading for many seconds
    (Suzuka 2026 pole lap: 24.5 s) while the position stream keeps moving. A stretch counts
    as frozen when speed, throttle and brake are all unchanged for longer than ``min_s``
    while the car moves more than 20 m.
    """
    v = df["Speed_ms"].to_numpy(float)
    th = df["Throttle"].to_numpy(float)
    br = df["Brake"].to_numpy(float)
    t = df["Time_s"].to_numpy(float)
    xy = df[["X_m", "Y_m"]].to_numpy(float)
    same = (np.diff(v) == 0) & (np.diff(th) == 0) & (np.diff(br) == 0)
    mask = np.zeros(len(v), dtype=bool)
    i = 0
    while i < len(same):
        if not same[i]:
            i += 1
            continue
        j = i
        while j < len(same) and same[j]:
            j += 1
        moved = np.hypot(*(xy[j] - xy[i]))
        if t[j] - t[i] > min_s and moved > 20.0:
            mask[i + 1 : j + 1] = True  # the first sample of the run is a real reading
        i = j
    return mask


def frozen_seconds(df: pd.DataFrame) -> float:
    """Total duration [s] of frozen car-data stretches in a lap (see :func:`frozen_mask`)."""
    m = frozen_mask(df)
    dt = np.diff(df["Time_s"].to_numpy(float), append=df["Time_s"].iloc[-1])
    return float(dt[m].sum())


def load_session_laps_openf1(
    client: OpenF1Client,
    session: dict[str, Any],
    top_n_drivers: int = 10,
) -> list[tuple[str, pd.DataFrame]]:
    """Fastest valid lap of the ``top_n_drivers`` quickest drivers of a session.

    Laps are dropped when they are pit-out laps, have no duration, or are faster than the
    driver's classified best time (i.e. the lap was deleted for track limits).

    Returns
    -------
    list of (driver_code, DataFrame)
        DataFrames use the column layout of :func:`f1line.io.telemetry.load_session_laps`
        plus ``Official_LapTime_s``, sorted fastest first.
    """
    sk = int(session["session_key"])
    laps = client.laps(sk)
    if laps.empty:
        logger.warning("Session %s has no laps", sk)
        return []
    drivers = client.drivers(sk)
    acronym = (
        dict(zip(drivers["driver_number"], drivers["name_acronym"])) if not drivers.empty else {}
    )

    # Best classified time per driver (catches deleted laps).
    best_official: dict[int, float] = {}
    res = client.session_result(sk)
    if not res.empty and "duration" in res.columns:
        for _, r in res.iterrows():
            d = r["duration"]
            times = [t for t in (d if isinstance(d, list) else [d]) if t]
            if times:
                best_official[int(r["driver_number"])] = float(min(times))

    n0 = len(laps)
    laps = laps[laps["lap_duration"].notna() & ~laps["is_pit_out_lap"].fillna(False).astype(bool)]
    laps = laps[laps["date_start"].notna()]
    dropped_deleted = 0
    keep = []
    for _, lap in laps.iterrows():
        best = best_official.get(int(lap["driver_number"]))
        if best is not None and lap["lap_duration"] < best - 0.02:
            dropped_deleted += 1
            continue
        keep.append(lap)
    laps = pd.DataFrame(keep)
    if laps.empty:
        return []
    fastest = laps.sort_values("lap_duration").groupby("driver_number").head(1)
    fastest = fastest.sort_values("lap_duration").head(top_n_drivers)
    logger.info(
        "Session %s: %d laps -> %d candidate (deleted=%d) -> top %d drivers",
        sk, n0, len(laps), dropped_deleted, len(fastest),
    )

    results: list[tuple[str, pd.DataFrame]] = []
    for _, lap in fastest.iterrows():
        drv = int(lap["driver_number"])
        t0 = pd.to_datetime(lap["date_start"], utc=True).to_pydatetime()
        t1 = t0 + timedelta(seconds=float(lap["lap_duration"]))
        pad = timedelta(seconds=1.0)
        loc = client.location(sk, drv, t0 - pad, t1 + pad)
        car = client.car_data(sk, drv, t0 - pad, t1 + pad)
        if len(loc) < 50 or len(car) < 50:
            logger.warning("Driver %s: too few samples (loc=%d car=%d)", drv, len(loc), len(car))
            continue
        df = _merge_lap(loc, car, t0)
        df = df[(df["Time_s"] >= 0.0) & (df["Time_s"] <= float(lap["lap_duration"]))]
        df = df.reset_index(drop=True)
        df["Distance"] -= df["Distance"].iloc[0]
        df["Official_LapTime_s"] = float(lap["lap_duration"])
        results.append((str(acronym.get(drv, drv)), df))
    return results
