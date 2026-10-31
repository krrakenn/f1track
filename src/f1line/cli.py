"""Typer CLI for the f1line pipeline.

Commands:
    fetch-tracks      Download TUM racetrack-database and cache to parquet.
    fetch-telemetry   Load FastF1 telemetry (cache only: FastF1's upstream is blocked).
    fetch-season      Download OpenF1 qualifying laps + OSM geometry for the calendar.
    build-season      Geometry -> calibration -> optimization -> artifacts for the app.
    align             Align telemetry to track coordinate frame (TUM, legacy data).
    report            Generate docs/RESULTS.md from the season artifacts.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(
    name="f1line",
    help="F1 Optimal Racing Line Predictor – CLI",
    add_completion=False,
)

# ── Logging setup ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def _load_cfg(config: Optional[Path]) -> dict:
    from f1line.config import load_config
    return load_config(config)


# ─────────────────────────────────────────────────────────────────────────────
# fetch-tracks
# ─────────────────────────────────────────────────────────────────────────────


@app.command("fetch-tracks")
def fetch_tracks(
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Override config YAML"),
    force: bool = typer.Option(False, "--force", help="Re-download even if cached"),
) -> None:
    """Download TUM racetrack-database and cache all circuits to parquet."""
    cfg = _load_cfg(config)
    paths_cfg = cfg.get("paths", {})
    raw_dir = Path(paths_cfg.get("data_raw", "data/raw"))
    cache_dir = raw_dir / "track_cache"

    from f1line.io.tum_tracks import (
        clone_tum_repo,
        list_available_circuits,
        load_raceline,
        load_track,
    )

    repo_dir = clone_tum_repo(raw_dir, force=force)
    circuits = list_available_circuits(repo_dir)
    typer.echo(f"Found {len(circuits)} circuits in TUM repo: {circuits}")

    for name in circuits:
        try:
            df = load_track(name, cache_dir, tum_repo_dir=repo_dir, force_download=force)
            typer.echo(f"  [OK] {name}: {len(df)} points")
            load_raceline(name, cache_dir, tum_repo_dir=repo_dir, force_download=force)
        except Exception as exc:
            typer.echo(f"  [FAIL] {name}: {exc}", err=True)

    typer.echo("fetch-tracks complete.")


# ─────────────────────────────────────────────────────────────────────────────
# fetch-telemetry
# ─────────────────────────────────────────────────────────────────────────────


@app.command("fetch-telemetry")
def fetch_telemetry(
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
    circuits_filter: Optional[str] = typer.Option(
        None, "--circuits", help="Comma-separated list of circuits"
    ),
    force: bool = typer.Option(False, "--force"),
) -> None:
    """Fetch FastF1 telemetry for all configured circuits and save to parquet."""
    cfg = _load_cfg(config)
    from f1line.config import load_circuits
    from f1line.io.telemetry import load_session_laps, save_laps_to_parquet

    tel_cfg = cfg.get("telemetry", {})
    season = int(tel_cfg.get("season", 2024))
    session_type = str(tel_cfg.get("session", "Q"))
    top_n = int(tel_cfg.get("top_n_laps_per_driver", 1))
    drop_offtrack = bool(tel_cfg.get("drop_offtrack", True))
    require_accurate = bool(tel_cfg.get("require_accurate", True))

    paths_cfg = cfg.get("paths", {})
    cache_dir = Path(paths_cfg.get("cache_dir", "cache"))
    interim_dir = Path(paths_cfg.get("data_interim", "data/interim"))

    circuits = load_circuits()
    if circuits_filter:
        allowed = {c.strip().lower() for c in circuits_filter.split(",")}
        circuits = {k: v for k, v in circuits.items() if k.lower() in allowed}

    for name, circuit_cfg in circuits.items():
        event = circuit_cfg.get("fastf1_event", name)
        cir_season = int(circuit_cfg.get("season", season))
        event_slug = name.replace(" ", "_").lower()
        out_dir = interim_dir / event_slug

        # Check if already done
        existing = list(out_dir.glob(f"{cir_season}_*.parquet"))
        if existing and not force:
            typer.echo(f"  [OK] {name}: {len(existing)} laps already cached (use --force to redo)")
            continue

        typer.echo(f"  Fetching {name} ({event}, {cir_season} {session_type})…")
        try:
            laps = load_session_laps(
                cir_season, event, session_type,
                cache_dir=cache_dir,
                top_n_per_driver=top_n,
                drop_offtrack=drop_offtrack,
                require_accurate=require_accurate,
            )
            paths = save_laps_to_parquet(laps, out_dir, cir_season, event_slug, session_type)
            typer.echo(f"    -> {len(paths)} laps saved")
        except Exception as exc:
            typer.echo(f"    [FAIL] {name}: {exc}", err=True)

    typer.echo("fetch-telemetry complete.")


# ─────────────────────────────────────────────────────────────────────────────
# fetch-season (OpenF1 + OSM)
# ─────────────────────────────────────────────────────────────────────────────


@app.command("fetch-season")
def fetch_season(
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
    circuits_filter: Optional[str] = typer.Option(
        None, "--circuits", help="Comma-separated circuit slugs (default: whole calendar)"
    ),
) -> None:
    """Download qualifying laps (OpenF1) and raceway geometry (OSM) for the season calendar."""
    cfg = _load_cfg(config)
    from f1line.season import (
        fetch_circuit_laps,
        fetch_circuit_osm,
        openf1_client,
        season_circuits,
    )

    circuits = season_circuits(cfg)
    if circuits_filter:
        allowed = {c.strip().lower() for c in circuits_filter.split(",")}
        circuits = {k: v for k, v in circuits.items() if k in allowed}
    client = openf1_client(cfg)
    # OpenF1 first (fast, reliable), then OSM (public Overpass servers are often busy).
    for fetch in (lambda n, c: fetch_circuit_laps(n, c, cfg, client),
                  lambda n, c: fetch_circuit_osm(n, c, cfg)):
        for name, ccfg in circuits.items():
            try:
                typer.echo(f"  [OK] {fetch(name, ccfg)}")
            except Exception as exc:
                typer.echo(f"  [FAIL] {name}: {exc}", err=True)
    typer.echo("fetch-season complete.")


# ─────────────────────────────────────────────────────────────────────────────
# build-season (geometry -> calibration -> optimization -> artifacts)
# ─────────────────────────────────────────────────────────────────────────────


@app.command("build-season")
def build_season(
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
    circuits_filter: Optional[str] = typer.Option(
        None, "--circuits", help="Comma-separated circuit slugs to optimize (default: all)"
    ),
    reuse_theta: bool = typer.Option(
        False, "--reuse-theta", help="Load models/theta_<year>.json instead of recalibrating"
    ),
) -> None:
    """Build geometry, calibrate the season car and optimize every circuit for the app."""
    cfg = _load_cfg(config)
    import json as _json

    from f1line.pipeline import (
        build_geometry,
        calibrate_season,
        optimize_circuit,
        params_from_diag,
        season_dir,
        season_report,
    )
    from f1line.season import season_circuits
    from f1line.vehicle.calibrate import load_theta

    circuits = season_circuits(cfg)
    tracks, failed = {}, {}
    for name, ccfg in circuits.items():
        try:
            tracks[name], diag = build_geometry(name, ccfg, cfg)
            typer.echo(f"  [GEOM] {name}: {diag['source']}, {tracks[name].total_length:.0f} m")
        except Exception as exc:
            failed[name] = f"geometry: {exc}"
            typer.echo(f"  [FAIL] {name}: geometry: {exc}", err=True)

    year = int(cfg["season"]["year"])
    theta_path = Path(cfg["paths"]["models_dir"]) / f"theta_{year}.json"
    if reuse_theta and theta_path.exists():
        calib_diag = _json.loads(theta_path.read_text())["diagnostics"]
        params = params_from_diag(calib_diag, load_theta(theta_path))
    else:
        params, calib_diag = calibrate_season(circuits, tracks, cfg)
    typer.echo(f"  [THETA] grip by group={calib_diag.get('grip_utilization_by_group')} "
               f"season car={calib_diag['base_params']}")

    selected = tracks
    if circuits_filter:
        allowed = {c.strip() for c in circuits_filter.split(",")}
        selected = {k: v for k, v in tracks.items() if k in allowed}
    for name, track in selected.items():
        try:
            if name not in params:
                raise RuntimeError("no calibrated parameters (no qualifying data this season)")
            m = optimize_circuit(name, circuits[name], track, params[name], cfg, calib_diag)
            typer.echo(
                f"  [OPT] {name}: optimal {m['lap_times_s']['min_time']:.3f} s vs pole "
                f"{m['pole_time_s']:.3f} s ({m['metrics']['optimal_vs_pole_pct']:+.2f} %)"
            )
        except Exception as exc:
            failed[name] = f"optimize: {exc}"
            typer.echo(f"  [FAIL] {name}: optimize: {exc}", err=True)

    metas = []
    for name in tracks:
        p = season_dir(cfg, name) / "meta.json"
        if p.exists():
            metas.append(_json.loads(p.read_text()))
    if metas:
        typer.echo("\n" + season_report(metas, cfg).to_string(index=False))
    if failed:
        typer.echo(f"\nFailed: {failed}", err=True)
    typer.echo("build-season complete.")


# ─────────────────────────────────────────────────────────────────────────────
# align
# ─────────────────────────────────────────────────────────────────────────────


@app.command("align")
def align(
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
    circuits_filter: Optional[str] = typer.Option(None, "--circuits"),
) -> None:
    """Align telemetry X/Y to TUM track coordinates for all circuits."""
    cfg = _load_cfg(config)
    from f1line.config import load_circuits
    from f1line.geometry.align import align_lap_to_track
    from f1line.geometry.track import build_track
    from f1line.io.telemetry import load_lap_from_parquet
    from f1line.io.tum_tracks import load_track

    paths_cfg = cfg.get("paths", {})
    raw_dir = Path(paths_cfg.get("data_raw", "data/raw"))
    cache_dir = raw_dir / "track_cache"
    interim_dir = Path(paths_cfg.get("data_interim", "data/interim"))
    track_cfg = cfg.get("track", {})

    circuits = load_circuits()
    if circuits_filter:
        allowed = {c.strip() for c in circuits_filter.split(",")}
        circuits = {k: v for k, v in circuits.items() if k in allowed}

    for name, circuit_cfg in circuits.items():
        tum_file = circuit_cfg.get("tum_file", f"{name}.csv")
        circuit_stem = Path(tum_file).stem
        typer.echo(f"  Aligning {name}…")
        try:
            repo_dir = raw_dir / "racetrack-database"
            raw_df = load_track(circuit_stem, cache_dir, tum_repo_dir=repo_dir)
            track = build_track(
                name, raw_df,
                ds_m=float(track_cfg.get("ds_m", 3.0)),
                smoothing_rule=str(track_cfg.get("spline_smoothing_rule", "rms_1.0m")),
                half_width_margin_m=float(track_cfg.get("half_width_margin_m", 1.0)),
            )

            event_slug = name.replace(" ", "_").lower()
            lap_dir = interim_dir / event_slug
            lap_files = sorted(lap_dir.glob("*.parquet"))
            if not lap_files:
                typer.echo(f"    No lap files for {name}, skipping")
                continue

            n_ok = 0
            for lp in lap_files:
                df = load_lap_from_parquet(lp)
                if "X_m" not in df.columns:
                    continue
                result = align_lap_to_track(
                    df["X_m"].to_numpy(), df["Y_m"].to_numpy(), track
                )
                if result.pct_within_bounds >= 95.0:
                    n_ok += 1
                else:
                    typer.echo(
                        f"    Warning: {lp.name} only {result.pct_within_bounds:.0f}% in bounds (RMS: {result.residual_rms:.1f} m)"
                    )

                # Append aligned outputs to DataFrame and save to processed dir
                df["s_m"] = result.s_proj
                df["alpha_m"] = result.alpha_proj
                df["x_aligned_m"] = result.x_aligned
                df["y_aligned_m"] = result.y_aligned
                df["direction"] = result.direction

                processed_dir = Path("data/processed") / event_slug
                processed_dir.mkdir(parents=True, exist_ok=True)
                out_path = processed_dir / lp.name
                df.to_parquet(out_path, index=False)

            typer.echo(f"    {n_ok}/{len(lap_files)} laps aligned successfully (>=95% within bounds)")
        except Exception as exc:
            typer.echo(f"    [FAIL] {name}: {exc}", err=True)

    typer.echo("align complete.")


# ─────────────────────────────────────────────────────────────────────────────
# report
# ─────────────────────────────────────────────────────────────────────────────


@app.command("report")
def report(
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
    output: Path = typer.Option(Path("docs/RESULTS.md"), "--output"),
) -> None:
    """Generate docs/RESULTS.md from the season artifacts (run build-season first)."""
    cfg = _load_cfg(config)
    from f1line.report import write_results

    path = write_results(cfg, output)
    typer.echo(f"  Report generated at {path}")


@app.command("export-web")
def export_web(
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
    out: Path = typer.Option(Path("web"), "--out", help="Web app folder"),
) -> None:
    """Write the static web app's data (web/data/*.json) from the season artifacts."""
    cfg = _load_cfg(config)
    from f1line.export_web import export_web as run

    data_dir = run(cfg, out)
    files = sorted(data_dir.glob("*.json"))
    total = sum(f.stat().st_size for f in files) / 1e6
    typer.echo(f"  Wrote {len(files)} files ({total:.1f} MB) to {data_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app()
