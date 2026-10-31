# Study guide: how the 2026 F1 optimal-lap project works

This guide is written so you can explain every part of the project in an interview, from
the one-line pitch down to the equations. Read it top to bottom once, then use the
"Interview questions" section to test yourself.

---

## 1. The pitch

**One line:** "I built a physics model of a 2026 Formula 1 car from public timing data, used
numerical optimization to find the fastest possible line around each circuit, and compared it
with the real pole lap of every 2026 race so far, in an interactive web app."

**Thirty seconds:** "For each of the 18 races of 2026 so far, I pull the real qualifying
telemetry from the OpenF1 API: car positions, speed, throttle and brake. I build the track
edges from map data, fit a point-mass car model with downforce, drag and tyre-grip limits to
the real laps, then solve a minimum-lap-time optimization problem with CasADi and IPOPT that
chooses the line and the speed together. Before qualifying, the forecast lands within 0.4% of
the real pole on average. Most of that accuracy comes from correlating the model to that
weekend's practice pace. On physics alone it is about 2.7% off (median). The web app races the
optimal lap against the real pole lap and animates how the optimizer finds the line."

Learn the honest version of the accuracy claim (last two sentences). Section 8 explains why.

---

## 2. Resume bullets (pick 2–3)

- Built an end-to-end Python pipeline that turns public F1 timing data (OpenF1) into
  minimum-lap-time racing lines for all 18 completed 2026 Grands Prix, using a calibrated
  point-mass vehicle model and nonlinear optimization (CasADi/IPOPT).
- Calibrated a 7-parameter car model (grip, downforce, drag, power, braking) on 166 qualifying
  laps with robust least squares, plus per-weekend setup and grip factors validated
  leave-one-circuit-out.
- Forecast pole times before qualifying with 0.41% mean error (0.36% median) by correlating the
  physics model to practice pace, versus 2.65% median for physics alone.
- Shipped a dependency-free static web app (vanilla JS + Canvas, ~70 KB of code, race data
  lazy-loaded) that replays the optimal lap against the real pole lap with synchronized
  telemetry charts and optimizer animations.

Tech keywords: Python, NumPy, SciPy, pandas, CasADi, IPOPT, cvxpy/OSQP, Numba, REST APIs,
JavaScript, HTML Canvas, pytest.

**About how it was built:** it was built with an AI coding assistant. If asked, say so plainly:
"I designed the project and directed the build with an AI coding assistant; here is how each
part works and why." Interviewers increasingly accept that. What they will not accept is
someone who cannot explain their own project, and that is what the rest of this guide is for.

---

## 3. The big picture

```
 OpenF1 API ──► laps: position x,y,z (~3.7 Hz), speed, throttle, brake, lap times
 TUM / OSM  ──► track edges (center line + left/right width)
 MultiViewer ─► corner numbers, map rotation
        │
        ▼
 1. DATA        fetch + cache, pick the 10 fastest drivers' best laps, repair frozen feed
 2. GEOMETRY    build the track in the telemetry frame, align (ICP), smooth (splines), widen
 3. CAR MODEL   point mass: grip μ, downforce c_L, drag c_D, power P, braking cap
 4. SPEED SIM   QSS: given a line, the fastest speed profile the car can drive on it
 5. CALIBRATE   fit car parameters so simulated laps match real laps
 6. OPTIMIZE    min-curvature QP (first guess) → min-time NLP (the answer)
 7. VALIDATE    compare to the real pole lap, held-out errors, sensitivity
 8. EXPORT      one JSON per race + an index → static web app
```

Commands (from the `Makefile` / `f1line` CLI): `fetch-season` → `build-season` → `report` →
`export-web`.

Code map:

| Stage | Files |
|---|---|
| Data | `src/f1line/io/openf1.py`, `season.py`, `io/osm.py`, `io/tum_tracks.py`, `io/multiviewer.py` |
| Geometry | `geometry/spline.py`, `geometry/track.py`, `geometry/align.py`, `geometry/circuit.py` |
| Car model + speed | `vehicle/model.py`, `vehicle/qss.py` |
| Calibration | `vehicle/calibrate.py`, `pipeline.py` (`calibrate_season`) |
| Optimization | `optimize/min_curvature.py`, `optimize/min_time.py` |
| Validation, report | `validate/`, `report.py` |
| Web | `export_web.py` → `web/index.html`, `web/app.js`, `web/style.css`, `web/data/*.json` |
| Settings | `config/default.yaml`, `config/circuits.yaml` |

About 6,900 lines of Python, 800 lines of JavaScript, 27 unit tests.

---

## 4. Stage by stage

### 4.1 Data (OpenF1)

**What:** OpenF1 is a free public REST API that re-serves the F1 live-timing feed. For each
session it gives lap times, car positions (x, y, z in tenths of a metre, about 3.7 samples per
second) and car data (speed, throttle %, brake on/off, gear, rpm).

**How it is used:**
- For each 2026 qualifying session, take the fastest lap of the 10 quickest drivers. Drop
  pit-out laps, and laps quicker than the driver's classified best (those were deleted for
  track limits).
- Also fetch the last session before qualifying (Practice 3, or Sprint Qualifying on sprint
  weekends). It is needed for the forecast in section 4.5.
- Everything is cached to disk (`cache/openf1/`). The free tier allows 30 requests per minute,
  so requests are throttled to one every 2.1 s.

**Data repair:** sometimes the car-data feed "freezes": speed and throttle repeat the same
values for seconds while the position keeps moving. `frozen_mask` detects these stretches. They
are then repaired by recomputing speed from the positions, and the throttle is shown grey
("feed froze"). Laps with more than about 2 s frozen are left out of calibration (that is why
it is 166 laps, not 180).

**Why not FastF1?** It is the usual Python library, but its upstream source started answering
HTTP 401 (unauthorized), so it could no longer download new sessions.

### 4.2 Track geometry

A track is stored as a **center line** (x, y points every few metres) plus, at every point,
the **width to the left and to the right**. Any line on the track is then just one number per
point: the **lateral offset α** from the center line, between −w_right and +w_left. This
"center line + offsets" idea is the key to the whole optimization.

Where the edges come from, in order of preference:
1. **TUM racetrack-database** (a university dataset of circuit center lines and widths), used
   only if its layout still matches the 2026 laps. Several circuits were rebuilt since that data
   was made (Melbourne, Barcelona, Austria, Zandvoort).
2. **OpenStreetMap** (via the Overpass API): circuits are mapped as "raceway" ways. Width is
   nominal, so it is widened to where real cars were seen.
3. **The drivers' own mean line**, for Madring (Madrid), which is barely mapped.

Techniques:
- **ICP (Iterative Closest Point)** aligns the TUM shape onto the telemetry: repeatedly match
  each point to its nearest neighbour and solve for the best rotation + translation. It uses
  several starting angles and also tries the mirror image, because data sources can be
  flipped. The match is accepted only if ≥95% of real car positions fall inside the track
  (±1 m).
- **Periodic smoothing splines** turn noisy points into a smooth closed curve whose residual is
  about 1 m. Curvature is a second derivative, so noise would explode into fake 15 g corners
  without smoothing.
- **Width widening:** in every 15 m window, take the 95th percentile of how far real cars went
  from the center, plus 0.5 m. Real legal laps prove the track is at least that wide. Using the
  95th percentile instead of the maximum ignores single noisy samples.
- **Car width:** the car's center may be up to 1 m beyond the white line, because the rules
  only require one wheel inside the track.

### 4.3 The car model (point mass)

The car is treated as a single point with mass. Everything is "per unit mass", so the
parameters are accelerations. There are 7 parameters (`vehicle/model.py`):

| Symbol | Meaning |
|---|---|
| μ | lateral tyre grip coefficient |
| μ_x,brake, μ_x,drive | longitudinal grip when braking / accelerating |
| c_L | downforce per unit mass (ρ·C_L·A / 2m) |
| c_D | drag per unit mass (ρ·C_D·A / 2m) |
| P | engine power per kg |
| a_brake,cap | hard limit on braking |

The physics, in four lines:

```
Downforce pushes the car down:   normal load = g + c_L·v²
Max cornering:                   a_lat,max   = μ·(g + c_L·v²)
Max braking:                     a_brake,max = min(μ_x,brake·(g + c_L·v²) + c_D·v², cap)
Max acceleration:                a_drive,max = min(P/v, μ_x,drive·(g + c_L·v²)) − c_D·v²
```

Key insights to be able to say:
- **Downforce makes grip grow with speed.** An F1 car can corner harder at 250 km/h than at
  100 km/h, which is why fast corners are taken nearly flat.
- **Acceleration is traction-limited at low speed** (the tyres spin) **and power-limited at
  high speed** (P/v falls as speed rises). Drag always fights it.
- **Friction ellipse:** the tyre has one grip budget shared between cornering and
  braking/accelerating: (a_lat / a_lat,max)² + (a_long / a_long,max)² ≤ 1. You cannot brake at
  full force and corner at full force at the same time. The g-force circle in the app shows
  this budget live.
- **Corner speed:** from v²·κ = μ(g + c_L v²), the fastest speed through curvature κ is
  v = √(μg / (κ − μ·c_L)). If κ < μ·c_L the corner is "flat out" (no limit from grip).

What a point mass does **not** have: yaw (rotation), weight transfer, suspension, tyre
temperature and wear, slipstream, the 2026 active aero and battery energy limits. These are
absorbed into effective values (for example, drag and power are "effective").

### 4.4 Speed on a given line: QSS (`vehicle/qss.py`)

QSS means "quasi-steady-state". Given a fixed line (so the curvature κ at each point is known),
it computes the fastest speed profile in three passes:

1. **Corner limit:** at every point, the maximum speed the curvature allows (formula above).
2. **Forward pass (acceleration):** walk forward around the lap; from each point the car can
   only gain as much speed as its acceleration limit allows over the next few metres:
   v²_next ≤ v² + 2·a_drive·ds (remaining grip after cornering, via the friction ellipse).
3. **Backward pass (braking):** walk backward, so the car arrives at every corner slow enough;
   the same idea with the braking limit.

The final speed is the minimum of the three. The lap is a loop, so it is run over three copies
and the middle one is kept. Lap time = Σ 2·ds / (v_i + v_{i+1}). It is accelerated with Numba
and takes milliseconds. The **"How speed is decided"** animation shows exactly these stages.

### 4.5 Calibration: making the model car behave like a real 2026 car

**Step 1: season car θ.** For each real qualifying lap, build the driver's line (spline
through positions) and its curvature. Run QSS with candidate parameters and compare to the
real lap: residuals are the speed trace difference and the lap time difference. Minimize with
`scipy.optimize.least_squares` using a **soft-L1 (robust) loss**, so occasional bad samples
do not dominate. It is fitted on all 166 laps at once, which gives one "2026 car". Every
parameter is checked against physical bounds (μ ≈ 3, downforce about 4 g at 290 km/h, power
about 880 W/kg).

**Step 2: weekend setup.** Teams run more wing at Monaco and less at Monza. So, per weekend,
only c_L and c_D are refitted on the last practice session. That data comes before qualifying,
so it is still a fair forecast.

**Step 3: grip factor γ.** The problem: a car calibrated on noisy reconstructed lines, then
driven on a perfectly smooth optimal line, comes out about 10% too fast. One factor γ scales the
tyre grip (μ terms) to fix that. It also absorbs what a point mass cannot do. γ is fitted per
**official circuit type** (street ≈ 0.67, permanent ≈ 0.79), because street circuits are bumpy
with walls. It is checked **leave-one-circuit-out (LOCO)**: γ for Monaco is fitted on the other
street circuits only, so Monaco's own pole time never informs its own prediction. That gives the
**physics-only** result: median error 2.65%.

**Step 4: practice correlation (the forecast).** This is how real teams use simulators: they
correlate them to Friday/Saturday running. For each weekend, the forecast pole is the best
practice lap × the typical practice-to-pole improvement. That improvement is the median ratio
over the *other* races with the same session type. Then γ is tuned so the optimal lap equals
that target. No qualifying data from that weekend is used. Result: **mean 0.41%, median
0.36%, worst 1.27%.**

**Be honest about step 4:** practice pace times a ratio, with no physics, scores about the
same. So the lap-time accuracy comes from practice data. The physics provides the **line, the
speed profile, and where on the lap time is won or lost**. The project already reports it this
way (the app shows both numbers).

### 4.6 Optimization: finding the fastest line

**Stage A: minimum-curvature QP** (`optimize/min_curvature.py`)

- Idea: a line with less bending allows more speed. Minimize the sum of squared curvature,
  approximated by second differences: Σ‖p_{i−1} − 2p_i + p_{i+1}‖², with p_i = c_i + α_i·n_i
  (center point + offset along the normal).
- Since p is linear in α, this is a **convex quadratic program** (QP): least squares with box
  constraints −w_right ≤ α ≤ w_left. Solved with **cvxpy + OSQP** using sparse matrices.
  Convex means there is one global minimum, found reliably in a fraction of a second.
- Limitation: it does not know about speed. It straightens slow and fast corners equally,
  whereas real time is lost mostly in fast corners. So it is only the first guess.

**Stage B: minimum-time NLP** (`optimize/min_time.py`)

- Decision variables at every grid point i (every 6 m, so about 900 points and 3,600 variables
  for a 5.4 km lap):
  - α_i: lateral offset (the line)
  - z_i = v_i²: speed squared (makes the dynamics linear)
  - a⁺_i ≥ 0: drive acceleration, a⁻_i ≥ 0: braking
- **Objective:** lap time Σ 2·ds_i / (v_i + v_{i+1}), plus a tiny smoothness penalty on α.
- **Constraints:**
  - dynamics: z_{i+1} = z_i + 2·ds_i·(a⁺_i − a⁻_i)
  - friction ellipse for drive and for braking (with the speed-dependent limits)
  - track edges: −w_right ≤ α_i ≤ w_left
  - curvature κ_i computed from three neighbouring points (Menger curvature), so the line and
    the speed are optimized **together**: the optimizer can take a wider line if that lets it
    carry more speed.
- It is **nonlinear and non-convex**, solved with **IPOPT** (interior-point method) through
  **CasADi** (builds the problem symbolically and gives IPOPT exact derivatives via automatic
  differentiation).
- **Coarse-to-fine:** solve on a 10 m grid first (fast), then on 6 m, warm-started from the
  coarse solution. The 4 m grid made several circuits fail to converge. 6 m converges and the
  result changes by about 0.03 s.
- **Fallbacks:** if IPOPT does not converge (it happened at 2 of 18 circuits: Barcelona and
  Monza), several candidate lines are scored with the independent QSS simulator and the fastest
  is kept.
- **Recording:** a callback stores every IPOPT iterate. The **"How the line is found"**
  animation replays the best line found so far, morphing from center line to optimum.
- **Sanity check:** the NLP lap time and an independent QSS run on the same line agree within
  0.7%, and on every circuit min-time ≤ min-curvature ≤ center line.

### 4.7 Validation

- **Gap vs pole:** model lap time vs real pole time (forecast and physics-only).
- **Pole path accuracy:** the drawn pole lap is within 0.4–0.8 m (median) of the raw position
  samples.
- **Sensitivity:** +5% grip → about −1.8% lap time, +5% downforce → about −0.6%, +5% power →
  about −0.3%. So **tyre grip dominates**, which is why grip is the parameter that gets
  calibrated per weekend.
- **Line deviation:** the model's line is 3–5 m from the pole driver's line on average.
- 27 unit tests (geometry, vehicle model, optimizer) with pytest.

### 4.8 The web app

- **Static site:** plain HTML/CSS/JavaScript, no framework, no backend. It can be hosted free on
  GitHub Pages.
- **Data:** `export_web.py` writes a small `index.json` (all races, times, outlines) and one
  JSON per race (lines, speeds, telemetry, optimizer history), 250–450 KB each (about 100 KB
  gzipped). Race files load only when a race is clicked (**lazy loading**).
- **Rendering:** HTML `<canvas>` with `requestAnimationFrame`. The track, edges and corner
  labels are drawn once into an off-screen canvas and reused every frame (**layer caching**),
  so a frame costs 1–3 ms.
- **Track width is exaggerated** for display (the legend says "track width drawn ×N"). A real
  track is about 12–15 m wide, which would be a hair-thin line at map scale and the line would
  be invisible inside it. Offsets are multiplied by the same factor, so the line stays inside
  the drawn track. This is display only; the numbers use real geometry.
- **Animations:** "Race the pole lap" (model car vs ghost of the real pole lap, both moving by
  their own real time), "How the line is found" (optimizer iterations), "How speed is decided"
  (QSS stages).
- **Polish:** tabular (equal-width) digits so live numbers do not jitter, cache-busting
  version strings, collapsible race list, responsive layout.

---

## 5. Numbers to memorize

| What | Value |
|---|---|
| Races | 18 completed 2026 rounds (Melbourne → Austin) |
| Calibration laps | 166 qualifying laps (top 10 per race, frozen-feed laps removed) |
| Forecast error vs pole | mean 0.41%, median 0.36%, worst 1.27% |
| Physics-only error | median 2.65%, mean 3.11% |
| Grip factor γ | street 0.67, permanent 0.79 |
| Season car | μ ≈ 3.06, power ≈ 880 W/kg |
| Optimization grid | 10 m coarse → 6 m fine, ~3,600 variables per circuit |
| Solver | IPOPT converged on 16/18, best-candidate fallback on 2 |
| Sampling rate of data | ~3.7 Hz (one position every ~20 m at 300 km/h) |
| Sensitivity (+5%) | grip −1.8%, downforce −0.6%, power −0.3% lap time |

---

## 6. Limitations (say these before the interviewer does)

1. **Point mass:** no yaw, weight transfer or tyre temperature. γ absorbs these on average, but
   not corner by corner.
2. **The forecast's accuracy comes from practice pace.** The physics alone is about 2.7% off.
3. **Low-rate data:** about 3.7 Hz positions means curvature has to be reconstructed with
   heavy smoothing.
4. **Not modelled:** 2026 active aero and battery energy limits (only effective drag/power),
   slipstream, elevation in the dynamics, banking, kerbs, wind.
5. **Geometry quality varies:** OSM widths are nominal, and Madring uses the drivers' mean line.
6. **Unofficial data:** OpenF1 data is for research and education, which is why the project
   credits it and is not commercial.

Good answer to "how would you improve it?": a bicycle (single-track) model with yaw and tyre
slip; an energy-management constraint for 2026 power units; higher-rate telemetry; Bayesian
calibration to get uncertainty bands on the predicted time.

---

## 7. Interview questions and short answers

1. **Why a point mass and not a full vehicle model?** It captures the main limits (grip,
   downforce, drag, power), has only 7 parameters that can be identified from public data, and
   keeps the optimization fast and robust. Public data cannot identify a full model
   (suspension, tyre curves).
2. **Why optimize speed squared instead of speed?** Because then the motion equation
   z_{i+1} = z_i + 2·a·ds is linear, which makes the problem better behaved for the solver.
3. **Why not just use the min-curvature line?** It ignores speed. It is several percent slower
   than the min-time line, because time is lost mostly in fast corners, and it treats all
   corners alike.
4. **What does IPOPT do?** It solves constrained nonlinear problems with an interior-point
   method: it keeps the iterate inside the feasible region using barrier terms and takes Newton
   steps on the optimality conditions. CasADi provides exact first and second derivatives.
5. **Is the min-time solution the global optimum?** Not guaranteed: the problem is non-convex.
   It is started from the min-curvature line (a good guess), and checked with an independent
   simulator, and it is always faster than both the min-curvature and the center line.
6. **How do you know the model is not overfitted?** Leave-one-circuit-out: each race's
   physics-only prediction uses a grip factor fitted without that race. The forecast uses only
   pre-qualifying data plus ratios from other races.
7. **Why is your model sometimes faster than pole?** It is a perfect driver on the
   mathematically best line with no mistakes, traffic or tyre warm-up issues. The forecast
   errors average close to zero (−0.25%), so there is no big bias either way.
8. **Why did you need γ at all?** Calibrating on noisy reconstructed lines inflates grip; a
   smooth optimal line then exploits it and comes out about 10% too fast. γ is one interpretable
   correction, fitted per official circuit type, and validated held-out.
9. **What was the hardest data problem?** Examples: the feed freezing (detected and repaired
   from positions); a distance channel that drifted 316 m at Suzuka and wrongly dropped 64% of
   good samples (replaced by a running-median position check); TUM layouts that no longer match
   (automatic check, OSM fallback).
10. **How does the app stay fast?** Static files, lazy per-race loading, cached canvas layers,
    and no framework.
11. **What is a friction ellipse?** A tyre's grip budget shared between cornering and
    braking/accelerating. Using all of it for braking leaves none for turning.
12. **Why is the 2026 season special?** New regulations: about 50/50 combustion/electric power
    and active aero. The car is calibrated only on 2026 laps, to avoid mixing two different
    generations of cars.

---

## 8. Why the honest framing matters

Saying "my physics model predicts pole within 0.4%" would be misleading, and an F1 or
engineering interviewer would likely spot it (that accuracy is beyond what a point mass can
do). The correct claim is stronger, because it shows you understand validation:

> "Physics alone gets within about 2.7% (median, held-out). Correlated to practice, the way
> teams correlate their simulators, the forecast is within 0.4%. I report both, because most of
> the forecast accuracy comes from practice pace, while the physics gives the line and where the
> time is won."

---

## 9. How to really own it (suggested order, about a week)

1. **Run it.** Start the web app (`.venv\Scripts\python -m http.server 8700 --directory web`)
   and explain every chart aloud.
2. **Read `vehicle/model.py` and `vehicle/qss.py`.** These are short. Change μ in a notebook and
   see the lap time change.
3. **Read `optimize/min_curvature.py`, then `optimize/min_time.py`** (`_solve_nlp` is the heart:
   variables, constraints, objective in about 80 lines).
4. **Read `pipeline.py` → `calibrate_season`** to see steps 1–4 of the calibration.
5. **Learn the concepts behind them:**
   - least squares and robust losses
   - convex vs non-convex optimization, QP vs NLP
   - the interior-point method (just the idea)
   - splines and curvature
   - ICP
   - cross-validation (leave-one-out)

   Good free sources: the CasADi "race car" example, the TUM paper by Heilmeier et al. 2019
   ("Minimum curvature trajectory planning and control for an autonomous race car"), and the
   OpenF1 docs.
6. **Practice the pitch** (section 1) and questions (section 7) out loud.

---

## 10. Glossary

- **Pole lap:** the fastest lap in qualifying; that driver starts first.
- **FP3 / Sprint Qualifying:** the last session before qualifying; used for the forecast.
- **Curvature κ:** 1 / corner radius; a hairpin has high κ, a straight has zero.
- **Lateral offset α:** how far left/right of the center line the car is.
- **QSS:** quasi-steady-state lap simulation (forward/backward speed passes).
- **QP / NLP:** quadratic program (convex, easy) / nonlinear program (general, harder).
- **IPOPT / CasADi / OSQP / cvxpy:** an NLP solver / a modelling tool with automatic
  differentiation / a QP solver / a convex modelling tool.
- **ICP:** Iterative Closest Point, aligning two point clouds.
- **LOCO:** leave-one-circuit-out validation.
- **γ (gamma):** grip-utilization factor scaling tyre grip.
- **c_L, c_D:** downforce and drag coefficients per unit mass.
