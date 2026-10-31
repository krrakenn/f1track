"""Download and parse TUM racetrack-database tracks and race lines.

Source: https://github.com/TUMFTM/racetrack-database (LGPL-3.0)
Data is downloaded at runtime; never committed to the repository.

Coordinate system: TUM files use meters (x_m, y_m).
"""

from __future__ import annotations

import io
import logging
import time
import urllib.request
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Base raw-content URL for the TUM racetrack-database GitHub repo
_TUM_BASE_URL = (
    "https://raw.githubusercontent.com/TUMFTM/racetrack-database/master/"
)
_TUM_TRACKS_SUBDIR = "tracks/"
_TUM_RACELINES_SUBDIR = "racelines/"

# Columns in TUM track files (after the comment header)
_TRACK_COLS = ["x_m", "y_m", "w_tr_right_m", "w_tr_left_m"]
_RACELINE_COLS = ["x_m", "y_m"]


def _download_text(url: str, retries: int = 3, backoff_s: float = 2.0) -> str:
    """Download text from *url* with exponential back-off on failure.

    Parameters
    ----------
    url:
        URL to fetch.
    retries:
        Number of attempts.
    backoff_s:
        Initial wait between retries (doubles each time).

    Returns
    -------
    str
        Decoded text content.
    """
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                return resp.read().decode("utf-8")
        except Exception as exc:
            wait = backoff_s * (2**attempt)
            logger.warning(
                "Download attempt %d/%d failed for %s: %s. Retrying in %.1f s…",
                attempt + 1,
                retries,
                url,
                exc,
                wait,
            )
            if attempt < retries - 1:
                time.sleep(wait)
            else:
                raise


def _parse_tum_csv(text: str, columns: list[str]) -> pd.DataFrame:
    """Parse a TUM CSV with a comment header line.

    The first line starting with '#' is treated as a comment and skipped.
    Remaining lines are standard comma-separated values.

    Parameters
    ----------
    text:
        Raw file content as a string.
    columns:
        Expected column names (in order).

    Returns
    -------
    pd.DataFrame
        Parsed DataFrame.
    """
    lines = [ln for ln in text.splitlines() if not ln.strip().startswith("#")]
    clean = "\n".join(lines)
    df = pd.read_csv(
        io.StringIO(clean),
        header=None,
        names=columns,
        skipinitialspace=True,
    )
    # Drop rows that are entirely NaN (can appear at end of file)
    df.dropna(how="all", inplace=True)
    df = df.astype(float)
    return df


def list_available_circuits(tum_repo_dir: Path) -> list[str]:
    """List circuit names available in the local TUM directory.

    Parameters
    ----------
    tum_repo_dir:
        Path to the locally cloned/cached TUM repo root.

    Returns
    -------
    list[str]
        Sorted list of circuit stems (e.g. ``["barcelona", "monza", ...]``).
    """
    tracks_dir = tum_repo_dir / "tracks"
    if not tracks_dir.exists():
        logger.warning("TUM tracks directory not found: %s", tracks_dir)
        return []
    stems = sorted(p.stem for p in tracks_dir.glob("*.csv"))
    return stems


def load_track(
    circuit_name: str,
    cache_dir: Path,
    tum_repo_dir: Optional[Path] = None,
    force_download: bool = False,
) -> pd.DataFrame:
    """Load a TUM track file as a DataFrame, caching to parquet.

    Coordinate convention (TUM):
    - x_m, y_m: center-line coordinates in meters.
    - w_tr_right_m: track width to the RIGHT of travel direction [m].
    - w_tr_left_m: track width to the LEFT of travel direction [m].

    Parameters
    ----------
    circuit_name:
        Name of the circuit (stem of the TUM CSV file, e.g. ``"monza"``).
    cache_dir:
        Directory to cache parquet files.
    tum_repo_dir:
        If provided, reads from the local directory instead of downloading.
    force_download:
        Re-download even if cache exists.

    Returns
    -------
    pd.DataFrame
        Columns: ``x_m, y_m, w_tr_right_m, w_tr_left_m``.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = cache_dir / f"{circuit_name}_track.parquet"

    if parquet_path.exists() and not force_download:
        logger.info("Loading track from cache: %s", parquet_path)
        return pd.read_parquet(parquet_path)

    if tum_repo_dir is not None:
        local_path = Path(tum_repo_dir) / "tracks" / f"{circuit_name}.csv"
        if local_path.exists():
            text = local_path.read_text(encoding="utf-8")
            logger.info("Loaded track from local repo: %s", local_path)
        else:
            raise FileNotFoundError(f"Track file not found: {local_path}")
    else:
        url = _TUM_BASE_URL + _TUM_TRACKS_SUBDIR + f"{circuit_name}.csv"
        logger.info("Downloading track: %s", url)
        text = _download_text(url)

    df = _parse_tum_csv(text, _TRACK_COLS)
    df.to_parquet(parquet_path, index=False)
    logger.info("Cached track to %s (%d rows)", parquet_path, len(df))
    return df


def load_raceline(
    circuit_name: str,
    cache_dir: Path,
    tum_repo_dir: Optional[Path] = None,
    force_download: bool = False,
) -> Optional[pd.DataFrame]:
    """Load the TUM precomputed race line for *circuit_name*, if available.

    Parameters
    ----------
    circuit_name:
        Circuit name (stem of TUM file).
    cache_dir:
        Directory to cache parquet files.
    tum_repo_dir:
        Local TUM repo directory.
    force_download:
        Re-download even if cache exists.

    Returns
    -------
    pd.DataFrame or None
        Columns: ``x_m, y_m``, or ``None`` if no race line exists.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = cache_dir / f"{circuit_name}_raceline.parquet"

    if parquet_path.exists() and not force_download:
        logger.info("Loading raceline from cache: %s", parquet_path)
        return pd.read_parquet(parquet_path)

    try:
        if tum_repo_dir is not None:
            local_path = Path(tum_repo_dir) / "racelines" / f"{circuit_name}.csv"
            if not local_path.exists():
                logger.info("No TUM raceline found for %s", circuit_name)
                return None
            text = local_path.read_text(encoding="utf-8")
        else:
            url = _TUM_BASE_URL + _TUM_RACELINES_SUBDIR + f"{circuit_name}.csv"
            logger.info("Downloading raceline: %s", url)
            text = _download_text(url, retries=2)

        df = _parse_tum_csv(text, _RACELINE_COLS)
        df.to_parquet(parquet_path, index=False)
        logger.info("Cached raceline to %s (%d rows)", parquet_path, len(df))
        return df
    except Exception as exc:
        logger.info("No TUM raceline for %s: %s", circuit_name, exc)
        return None


def clone_tum_repo(dest_dir: Path, force: bool = False) -> Path:
    """Clone the TUM racetrack-database repository using git.

    Falls back to downloading individual files if git is unavailable.

    Parameters
    ----------
    dest_dir:
        Directory to clone into (will create ``racetrack-database/`` inside).
    force:
        Re-clone even if already present.

    Returns
    -------
    Path
        Path to the cloned repository root.
    """
    import subprocess

    dest_dir = Path(dest_dir)
    repo_dir = dest_dir / "racetrack-database"

    if repo_dir.exists() and not force:
        logger.info("TUM repo already present at %s", repo_dir)
        return repo_dir

    dest_dir.mkdir(parents=True, exist_ok=True)
    repo_url = "https://github.com/TUMFTM/racetrack-database.git"

    logger.info("Cloning TUM racetrack-database to %s …", repo_dir)
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, str(repo_dir)],
            check=True,
            capture_output=True,
            text=True,
        )
        logger.info("Clone complete: %s", repo_dir)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        logger.warning("git clone failed (%s). Will download on demand.", exc)

    return repo_dir
