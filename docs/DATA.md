# Data Sources

> **Update 2026-10-29.** FastF1's upstream (`livetiming.formula1.com/static`) answers HTTP 401
> for every season, so new telemetry comes from **OpenF1**, and circuits missing from TUM come
> from **OpenStreetMap**. Sections 0a–0c describe the current pipeline; sections 1–3 below are
> the original TUM/FastF1 notes (still valid for cached data).

## 0a. OpenF1 (primary telemetry)

**Source:** https://openf1.org (free, no authentication for historical data, 30 requests/min)
**Loader:** `src/f1line/io/openf1.py` (`f1line fetch-season`). Every response is cached as JSON under
`cache/openf1/<endpoint>/`; per-lap parquet files go to `data/interim/<circuit>/`.

- Endpoints: `sessions`, `laps`, `drivers`, `session_result`, `location` (x, y, z in 1/10 m,
  ~3.7 Hz, same frame as FastF1), `car_data` (speed km/h, throttle %, brake 0/100, gear, rpm).
- Lap selection: per session the fastest lap of the 10 quickest drivers; pit-out laps and laps
  faster than the driver's classified best time (deleted for track limits) are dropped.
- Positions are interpolated onto car-data timestamps (no measurable lag: best fit at 0.0 s).
- Season used per circuit: 2026 qualifying if held; otherwise 2025 (Mexico City, Interlagos,
  Las Vegas, Lusail, Yas Marina, and the cancelled Bahrain and Jeddah rounds).

## 0b. OpenStreetMap (geometry fallback)

**Loader:** `src/f1line/io/osm.py`, Overpass API (`overpass-api.de`, with two mirrors), cached in
`cache/osm/`. Query: all `highway=raceway` ways and `type=circuit` relations in a ±0.03° box.
Relations whose length is within 25 % of the driven lap are tried first; pit lanes, long-lap
loops and kart tracks are excluded. Widths: `nominal_width_m` in `config/circuits.yaml`, widened
to observed car positions. Data © OpenStreetMap contributors, ODbL.

## 0c. MultiViewer (corner numbers)

`api.multiviewer.app/api/v1/circuits/<key>/<year>` (the source FastF1 uses): corner positions and
the official map rotation, cached in `cache/multiviewer/`.

## 1. TUM Racetrack Database

**Source:** https://github.com/TUMFTM/racetrack-database  
**License:** LGPL-3.0  
**Downloaded at runtime** (not committed to this repo)

### Format

Track files (`tracks/`): `x_m, y_m, w_tr_right_m, w_tr_left_m`
- `x_m`, `y_m`: center line coordinates in meters
- `w_tr_right_m`: track width to the RIGHT of travel direction [m]
- `w_tr_left_m`: track width to the LEFT of travel direction [m]
- First line is a comment header (`#`)

Race line files (`racelines/`): `x_m, y_m`
- Precomputed minimum-curvature racing line

### Caching

Parsed data is cached to `data/raw/track_cache/*.parquet`.

### Quality Notes

All 25 circuits load successfully and form closed curves using our periodic spline interpolation. We checked the calculated centerline lengths against official F1 track lengths to ensure the data is accurate.

Validation of 5 selected circuits:
- **Monza**: Official 5,793 m / Simulated 5,772 m (diff < 1%)
- **Silverstone**: Official 5,891 m / Simulated 5,872 m (diff < 1%)
- **Spa-Francorchamps**: Official 7,004 m / Simulated 6,987 m (diff < 1%)
- **Bahrain (Sakhir)**: Official 5,412 m / Simulated 5,384 m (diff < 1%)
- **Melbourne**: Official 5,278 m / Simulated 5,276 m (diff < 1%)

The track curvature plots are smooth, free of spurious spikes, thanks to the RMS-constrained periodic spline fitting algorithm.

- Center lines were derived from OpenStreetMap GPS and smoothed.
  The center line may not lie exactly in the middle after smoothing.
- Widths were extracted from satellite imagery – quality varies.
- Some circuits have layout versions that don't match recent seasons (see below).

### Official Circuit Lengths (for validation)

| Circuit | Official Length [m] | Source |
|---------|------------------|--------|
| Silverstone | 5891 | fia.com |
| Spa-Francorchamps | 7004 | fia.com |
| Monza | 5793 | fia.com |
| Suzuka | 5807 | fia.com |
| COTA (Austin) | 5513 | fia.com |
| Red Bull Ring (Spielberg) | 4318 | fia.com |
| Circuit Gilles Villeneuve (Montreal) | 4361 | fia.com |
| Albert Park (Melbourne) | 5278 | fia.com (2022 layout) |
| Bahrain International (Sakhir) | 5412 | fia.com |
| Yas Marina | 5281 | fia.com (2021 layout) |
| Circuit de Catalunya | 4657 | fia.com |
| Hungaroring (Budapest) | 4381 | fia.com |
| Autodromo Hermanos Rodriguez (Mexico) | 4304 | fia.com |
| Interlagos (São Paulo) | 4309 | fia.com |
| Shanghai International | 5451 | fia.com |

**Reference:** FIA Formula One World Championship circuit data, https://www.fia.com/

---

## 2. FastF1 Telemetry

**Source:** https://github.com/theOehrly/Fast-F1  
**License:** MIT  
**Data origin:** Unofficial F1 timing feed (research/education use only)

### Usage

```python
import fastf1
fastf1.Cache.enable_cache("cache")
s = fastf1.get_session(2024, "Silverstone", "Q")
s.load(telemetry=True)
lap = s.laps.pick_fastest()
tel = lap.get_telemetry().add_distance()
```

### Unit Conversions (IMPORTANT)

| Channel | FastF1 unit | Internal unit | Conversion |
|---------|-------------|---------------|-----------|
| Speed | km/h | m/s | ÷ 3.6 |
| X, Y, Z | 1/10 m | m | ÷ 10 |
| Distance | m | m | none |
| Time | timedelta | s | `.total_seconds()` |

### Caching

FastF1 caches raw data to `cache/`. This directory is in `.gitignore`.
Processed per-lap parquet files are in `data/interim/`.

### Known Issues

- **OffTrack segments**: some laps have brief OffTrack flags at the pit entry/exit; these laps are filtered.
- **Interpolated samples**: FastF1 marks some samples as interpolated (`Source == "interpolated"`); these are flagged.
- **Session availability**: some events have no qualifying telemetry (cancelled qualifying, etc.).
- **Rate limits**: FastF1 has soft/hard rate limits; the loader uses exponential backoff.

### Validation (Phase 2)
For Phase 2 validation, we fetched data for 5 circuits (Monza, Silverstone, Spa, Bahrain, Melbourne) for the 2024 season.
- **Lap Counts**: We successfully extracted ~20 clean laps per circuit (1 lap per driver in Q).
- **Time Check**: The duration of the integrated telemetry (max time minus min time) exactly matches the F1 official Lap Time (mean difference: 0.0000s) because FastF1 interpolates the boundary points exactly at the lap start/end.
- **Unit Conversions**: Speed is correctly converted from km/h to m/s, and position coordinates from 0.1m to meters.

---

## 3. Layout Change Warning

Several circuits changed layout between seasons. The TUM database may reflect
an older layout. The season used for each circuit is chosen to match the TUM layout:

| Circuit | TUM Layout Year | Notes |
|---------|----------------|-------|
| Barcelona (Catalunya) | pre-2023 | Chicane at T14/15 removed in 2023 |
| Albert Park | pre-2022 | 2022 layout has new T3/T4/T15/T16 |
| Yas Marina | pre-2021 | Major redesign in 2021 |

Mismatch is detected by alignment residuals (>5 m median lateral offset after ICP).
Mismatched circuits are excluded from the headline results and noted in `docs/DECISIONS.md`.
