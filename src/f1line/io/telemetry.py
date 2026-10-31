"""FastF1 telemetry loader with filtering, unit conversion, and parquet caching.

Unit conventions (internal – SI):
- Speed: km/h in FastF1 -> converted to m/s on load.
- Position X, Y, Z: FastF1 stores in 1/10 m -> converted to m on load.
- Distance: FastF1 adds_distance() returns m already.
- Time: timedelta -> seconds.

See docs/DATA.md for data quality notes per circuit.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# FastF1 telemetry channels we want (subset of available channels)
_TELEM_CHANNELS = ["Speed", "RPM", "nGear", "Throttle", "Brake", "DRS", "X", "Y", "Z", "Source"]

# Columns kept in the merged telemetry DataFrame (after unit conversion)
_OUTPUT_COLS = [
    "Distance",   # [m] cumulative distance along lap
    "Speed_ms",   # [m/s] car speed
    "RPM",        # [rpm]
    "nGear",      # [-] gear number
    "Throttle",   # [%] throttle pedal 0-100
    "Brake",      # [bool]
    "DRS",        # [0/8/10/14] DRS status
    "X_m",        # [m] position X
    "Y_m",        # [m] position Y
    "Z_m",        # [m] position Z
    "Time_s",     # [s] time since lap start
    "Source",     # interpolation source flag
]


def _km_h_to_m_s(speed_kmh: "pd.Series") -> "pd.Series":
    """Convert speed from km/h to m/s.

    Parameters
    ----------
    speed_kmh:
        Speed in km/h.

    Returns
    -------
    pd.Series
        Speed in m/s.
    """
    return speed_kmh / 3.6


def _tenth_m_to_m(pos_10m: "pd.Series") -> "pd.Series":
    """Convert FastF1 position (1/10 m) to meters.

    Parameters
    ----------
    pos_10m:
        Position in 1/10 m as provided by FastF1.

    Returns
    -------
    pd.Series
        Position in meters.
    """
    return pos_10m / 10.0


def load_session_laps(
    season: int,
    event: str,
    session_type: str,
    cache_dir: Path,
    top_n_per_driver: int = 1,
    drop_offtrack: bool = True,
    require_accurate: bool = True,
) -> list[tuple[str, pd.DataFrame]]:
    """Load and filter telemetry laps from a FastF1 session.

    Parameters
    ----------
    season:
        Championship year (e.g. 2024).
    event:
        Event name as used by FastF1 (e.g. "British Grand Prix").
    session_type:
        Session identifier: ``"Q"``, ``"R"``, ``"FP1"``, etc.
    cache_dir:
        Directory for FastF1 cache.
    top_n_per_driver:
        Keep the N fastest laps per driver (default 1).
    drop_offtrack:
        Drop laps that contain any ``OffTrack`` position status.
    require_accurate:
        Drop laps where ``IsAccurate == False``.

    Returns
    -------
    list of (driver_code, DataFrame)
        Each DataFrame has columns in ``_OUTPUT_COLS``.
    """
    try:
        import fastf1
    except ImportError as exc:
        raise ImportError("fastf1 is required: pip install fastf1") from exc

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_dir))

    logger.info(
        "Loading FastF1 session: season=%d event=%r session=%r", season, event, session_type
    )

    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            session = fastf1.get_session(season, event, session_type)
            session.load(telemetry=True, weather=False, messages=False)
            break
        except Exception as exc:
            wait = 5.0 * (2**attempt)
            logger.warning(
                "Session load attempt %d/%d failed: %s. Retrying in %.0f s…",
                attempt + 1,
                max_attempts,
                exc,
                wait,
            )
            if attempt < max_attempts - 1:
                time.sleep(wait)
            else:
                raise

    laps_df = session.laps

    # ── Filter deleted and in/out laps ─────────────────────────────────────
    dropped_reasons: dict[str, int] = {}
    n_initial = len(laps_df)

    if require_accurate and "IsAccurate" in laps_df.columns:
        mask_inaccurate = ~laps_df["IsAccurate"].fillna(False)
        dropped_reasons["IsAccurate==False"] = int(mask_inaccurate.sum())
        laps_df = laps_df[~mask_inaccurate]

    if "Deleted" in laps_df.columns:
        mask_deleted = laps_df["Deleted"].fillna(False)
        dropped_reasons["Deleted"] = int(mask_deleted.sum())
        laps_df = laps_df[~mask_deleted]

    if "PitOutTime" in laps_df.columns:
        mask_out = laps_df["PitOutTime"].notna()
        dropped_reasons["OutLap"] = int(mask_out.sum())
        laps_df = laps_df[~mask_out]

    if "PitInTime" in laps_df.columns:
        mask_in = laps_df["PitInTime"].notna()
        dropped_reasons["InLap"] = int(mask_in.sum())
        laps_df = laps_df[~mask_in]

    # ── Remove laps without valid lap time ──────────────────────────────────
    mask_no_time = laps_df["LapTime"].isna()
    dropped_reasons["NoLapTime"] = int(mask_no_time.sum())
    laps_df = laps_df[~mask_no_time]

    logger.info(
        "Lap filtering: %d -> %d kept (dropped: %s)",
        n_initial,
        len(laps_df),
        dropped_reasons,
    )

    # ── Top-N per driver ────────────────────────────────────────────────────
    laps_df = (
        laps_df.sort_values("LapTime")
        .groupby("Driver")
        .head(top_n_per_driver)
        .reset_index(drop=True)
    )
    logger.info("After top-%d per driver: %d laps", top_n_per_driver, len(laps_df))

    results: list[tuple[str, pd.DataFrame]] = []
    n_offtrack_dropped = 0

    for _, lap_row in laps_df.iterrows():
        driver = str(lap_row["Driver"])
        try:
            tel = lap_row.get_telemetry().add_distance()
        except Exception as exc:
            logger.warning("Driver %s: telemetry unavailable: %s", driver, exc)
            continue

        # ── OffTrack filter ────────────────────────────────────────────────
        if drop_offtrack and "Status" in tel.columns:
            if (tel["Status"] == "OffTrack").any():
                logger.info("Dropped driver %s: OffTrack segments detected", driver)
                n_offtrack_dropped += 1
                continue

        # ── Unit conversions ───────────────────────────────────────────────
        out = pd.DataFrame()
        out["Distance"] = tel["Distance"].astype(float)      # already m
        out["Speed_ms"] = _km_h_to_m_s(tel["Speed"].astype(float))
        out["RPM"] = tel["RPM"].astype(float) if "RPM" in tel.columns else float("nan")
        out["nGear"] = tel["nGear"].astype(float) if "nGear" in tel.columns else float("nan")
        out["Throttle"] = (
            tel["Throttle"].astype(float) if "Throttle" in tel.columns else float("nan")
        )
        out["Brake"] = tel["Brake"].astype(bool) if "Brake" in tel.columns else False
        out["DRS"] = tel["DRS"].astype(float) if "DRS" in tel.columns else float("nan")
        out["X_m"] = _tenth_m_to_m(tel["X"].astype(float))
        out["Y_m"] = _tenth_m_to_m(tel["Y"].astype(float))
        out["Z_m"] = _tenth_m_to_m(tel["Z"].astype(float)) if "Z" in tel.columns else 0.0
        out["Time_s"] = tel["Time"].dt.total_seconds().astype(float)
        out["Source"] = tel["Source"].astype(str) if "Source" in tel.columns else "unknown"
        if "LapTime" in lap_row and pd.notna(lap_row["LapTime"]):
            out["Official_LapTime_s"] = lap_row["LapTime"].total_seconds()
        else:
            out["Official_LapTime_s"] = float("nan")

        out.dropna(subset=["Distance", "Speed_ms", "X_m", "Y_m"], inplace=True)
        out.reset_index(drop=True, inplace=True)

        results.append((driver, out))

    if n_offtrack_dropped:
        logger.info("Dropped %d laps for OffTrack segments", n_offtrack_dropped)

    logger.info("Returning %d clean laps", len(results))
    return results


def save_laps_to_parquet(
    laps: list[tuple[str, pd.DataFrame]],
    output_dir: Path,
    season: int,
    event_slug: str,
    session_type: str,
) -> list[Path]:
    """Save per-driver lap DataFrames to parquet files.

    Parameters
    ----------
    laps:
        List of (driver_code, DataFrame) from :func:`load_session_laps`.
    output_dir:
        Directory to write parquet files.
    season:
        Championship year.
    event_slug:
        Filesystem-safe event name (e.g. ``"british_gp"``).
    session_type:
        Session type string.

    Returns
    -------
    list[Path]
        Paths of written files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for driver, df in laps:
        fname = f"{season}_{event_slug}_{session_type}_{driver}.parquet"
        fpath = output_dir / fname
        df.to_parquet(fpath, index=False)
        logger.info("Saved %s (%d rows)", fpath, len(df))
        paths.append(fpath)
    return paths


def load_lap_from_parquet(path: Path) -> pd.DataFrame:
    """Load a single lap from a parquet file.

    Parameters
    ----------
    path:
        Path to the parquet file.

    Returns
    -------
    pd.DataFrame
        Lap telemetry with SI units.
    """
    return pd.read_parquet(path)
