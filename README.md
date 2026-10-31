# F1 Optimal Racing Line Predictor – 2026 season

Predicts the time-optimal racing line and speed profile on every circuit of the **2026 Formula 1
calendar**, using a vehicle model calibrated from real 2026 qualifying telemetry, and compares it
with the real pole lap. An interactive app shows the line on the track with the telemetry charts
beside it and animates how the line and the speed profile are found.

---

## What it does

1. **Calendar and laps** – every 2026 round that has been run so far (18, Australia to the
   United States GP; `season.completed_only` in the config). For each, the fastest lap of the 10
   quickest qualifiers. Re-run the pipeline after each new qualifying session to add it.
2. **Track geometry** in the F1 telemetry frame: TUM racetrack-database where its layout still
   matches the driven laps, otherwise OpenStreetMap circuit relations, otherwise the drivers' mean
   line. Widths are widened to where real cars were observed.
3. **Calibration** of a point-mass car with aero and a friction ellipse on 180 laps from 18
   circuits, plus a grip-utilization factor fitted on pole times and checked leave-one-circuit-out.
4. **Optimization**: min-curvature QP (cvxpy/OSQP, sparse), then minimum-time NLP (CasADi/IPOPT,
   coarse-to-fine). Every optimizer iterate is recorded for the app.
5. **App** (Streamlit): pick a circuit; map with the optimal line coloured by speed, real pole lap
   and corner numbers; synchronized charts (speed, throttle/brake, lateral g, gap to the real lap);
   three animations: *Lap simulation* (optimal car vs real pole ghost), *How the line is found*
   (centerline → min-curvature → IPOPT iterations), *How speed is decided* (corner limits →
   acceleration pass → braking pass). Sliders re-simulate the line with your own grip, downforce
   and power.

## Web app (portfolio)

`web/` is a standalone static site: plain HTML, CSS and JavaScript, no framework and no server.
The page loads a 43 KB race index first and fetches each race (~250–450 KB, ~100 KB gzipped)
only when it is opened; animation frames cost about 1–3 ms because static layers are cached.

- **Race the pole lap:** a purple car drives the model's optimal lap against a ghost of the real
  pole lap in the pole sitter's team colour, with live clocks and the gap at every point.
- **How the line is found:** the best line found so far at each optimizer step, morphing from the
  center line to the optimum while the lap time falls.
- **How speed is decided:** corner limits, acceleration pass, braking pass, final speed.
- Timing tower of all completed 2026 rounds with pole sitter, pole time and model time; a
  season chart of model-vs-pole gaps.

```bash
make web      # f1line export-web  -> web/data/*.json
make serve    # http://localhost:8600
```

**Publish free on GitHub Pages:** push the repo to GitHub, then in the repository settings
choose *Pages → Build and deployment → Source: GitHub Actions*. The workflow in
`.github/workflows/pages.yml` publishes `web/` on every push to `main`. The site is fully
static, so Netlify, Cloudflare Pages or Vercel work too (publish directory `web`).

## Quick start

```bash
make setup     # venv + pip install -e ".[dev]"
make data      # TUM tracks, OpenF1 laps, OSM geometry (~20 min, rate-limited, cached)
make season    # geometry, calibration, optimization for all circuits
make report    # docs/RESULTS.md
make web       # static web app data
make serve     # web app at http://localhost:8600
make app       # Streamlit analysis app
```

Equivalent CLI: `f1line fetch-tracks`, `f1line fetch-season`, `f1line build-season`,
`f1line report` (every command has `--help`).

## Results

See [docs/RESULTS.md](docs/RESULTS.md) (generated) and `reports/season_2026.csv`: per-circuit
centerline / min-curvature / min-time / real pole lap times, held-out gap, replay error, line
deviation, solver status and a ±5 % sensitivity study.

## Project structure

```
config/default.yaml        all tunables          config/circuits.yaml   2026 calendar registry
src/f1line/io/             openf1.py, osm.py, multiviewer.py, tum_tracks.py, telemetry.py (FastF1)
src/f1line/geometry/       spline.py, track.py, align.py (ICP), circuit.py (TUM/OSM/telemetry)
src/f1line/vehicle/        model.py, qss.py, calibrate.py
src/f1line/optimize/       min_curvature.py (QP), min_time.py (NLP + history), utils.py
src/f1line/pipeline.py     build-season orchestration;  season.py data fetching;  report.py
app/                       streamlit_app.py, simulator.py (custom component), data.py
docs/                      DATA.md, MODEL.md, DECISIONS.md, RESULTS.md
```

## Limitations

Point mass (no yaw/roll transients, absorbed by the grip-utilization factor); no 2026 active aero
or energy clipping (effective drag/power); no elevation, banking, tyre state, wind or traffic.
Circuits without 2026 qualifying compare against a 2025 (previous-generation) lap. Madring
geometry is approximate (incomplete OSM data). See docs/RESULTS.md.

## Credits and data

- **[OpenF1](https://openf1.org)** – lap timing, car and position data. F1 timing data comes from an
  **unofficial feed** and is used for **research and education only**.
- **[TUM racetrack-database](https://github.com/TUMFTM/racetrack-database)** (LGPL-3.0) – center
  lines and widths.
- **[OpenStreetMap](https://www.openstreetmap.org/copyright)** contributors (ODbL) – circuit
  geometry via the Overpass API.
- **MultiViewer** circuit API – corner numbers and map rotation.
- **[FastF1](https://github.com/theOehrly/Fast-F1)** – earlier telemetry (its upstream is no longer
  reachable).

Raw third-party data is downloaded on demand and is not committed (`data/`, `cache/` are ignored).
