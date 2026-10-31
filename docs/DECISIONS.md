# Decision Log

All non-obvious engineering choices are logged here with date and justification.

---

## 2026-09-28: Periodic spline smoothing factor

**Decision:** Use `s = N × σ²` where σ = 1.0 m (configurable via `spline_smoothing_rule: "rms_1.0m"`).

**Justification:** TUM center-line data has some GPS noise (estimated 0.5–1.5 m). A smoothing
factor targeting 1.0 m RMS residual removes noise without over-smoothing the sharp corners.
This value is verified by inspecting curvature plots for spurious spikes.

---

## 2026-09-28: TUM Track Bounds Convention

**Decision:** The track boundaries (`w_tr_right_m` / `w_tr_left_m`) represent the physical white lines. We subtract a 1.0 m margin when checking the vehicle center.

**Justification:** We observed that the provided TUM raceline violates the parsed track boundaries by up to 1.5 - 2.0 meters on most tracks. This strongly implies the boundaries represent the *white lines* and the TUM racelines simulate the vehicle center cutting the curbs. We subtract a 1.0 m half-width margin during boundary enforcement to represent the F1 car width (2.0 m).

**Alternative considered:** Cross-validation GCV (scipy's `s=None`). Rejected because GCV
sometimes under-smooths at sharp corners, leading to curvature spikes that destabilize the QSS.

---

## 2026-09-28: Telemetry Alignment (Phase 3)

**Decision:** The F1 telemetry for Albert Park (Melbourne) from the 2024 season is excluded from calibration and evaluation.

**Justification:** The TUM track database for Melbourne relies on the pre-2022 circuit layout. In 2022, Albert Park saw significant changes to its layout (particularly the removal of the chicane in Sector 2). As a result, the ICP alignment for the 2024 FastF1 telemetry on this track fails spectacularly (median RMS lateral error >8.0 meters, and only ~50% of the lap stays within track bounds).

**Decision:** Alignment bounds are checked with a `half_width_margin_m = -1.0` configuration.

**Justification:** F1 sporting regulations dictate that a driver only needs one wheel inside the white lines to be considered within track bounds. Since the car is 2.0m wide (1.0m half-width), the vehicle's center point can legally be up to 1.0m outside the defined white line. We enforce this physically accurate bound instead of strictly confining the car center inside the white line.

---

## 2026-09-28: Calibration lap selection – qualifying only

**Decision:** Use qualifying (Q) laps as the primary calibration set.

**Justification:** Qualifying laps are the closest to the car's physical limit:
low fuel load (~30 kg), fresh tyres (push lap), and maximum attack. This maximizes
the fraction of the grip envelope observed, which is critical for envelope calibration.
Race laps have higher fuel, tyre deg, and traffic – harder to model.

---

## 2026-09-28: Season 2022–2025 (ground-effect era)

**Decision:** Do not mix pre-2022 seasons with 2022+ in a single calibration.

**Justification:** The 2022 regulation change to ground-effect aerodynamics fundamentally
changed the aero coefficients (CLA, CdA). Mixing eras would give a blended θ that
is accurate for neither era. The project targets 2022–2025 as the default era.

---

## 2026-09-28: TUM width convention verification needed at runtime

**Decision:** Do not assume which side is "left" or "right" in TUM files without verification.

**Justification:** The spec explicitly states that the TUM convention should be verified against
the real data (by checking that the TUM race line lies within the derived boundaries).
This check is performed in Phase 1 and logged per circuit.

---

## 2026-09-28: v_min = 5 m/s for numerical stability

**Decision:** Clip speed to v_min = 5 m/s inside P/v terms.

**Justification:** Prevents division by zero and numerical instability in the QSS.
5 m/s ≈ 18 km/h – well below any observed F1 speed (minimum ~50 km/h in slow hairpins).

---

## 2026-09-28: Stage B NLP – use SX (not MX) symbolic variables

**Decision:** Use CasADi SX for the NLP symbolic graph.

**Justification:** SX is faster than MX for graphs that fit in memory (< ~10k variables).
At N=500 points × 4 variables = 2000 decision variables, SX is appropriate. For N > 2000
(unlikely at 4 m spacing on typical circuits), switch to MX.

---

## 2026-09-28: Circuits excluded from headline metrics

*(To be filled as circuits are processed and alignment residuals evaluated.)*

Circuits with median lateral misfit > track half-width, or < 95% of samples within bounds,
are excluded from the headline validation table. See Phase 3 output.

---

## 2026-09-28: Asymmetric calibration loss

**Decision:** Use `soft_l1` (Huber-like) loss in `scipy.optimize.least_squares`.

**Justification:** Drivers are usually slightly below the grip limit (fuel load, safety margin,
risk management). A symmetric L2 loss would be biased by the occasional sub-limit lap.
`soft_l1` reduces the weight of large residuals from over-limit artifacts or outlier laps.
The `grip_utilization` factor approach (scaling θ post-calibration) was considered but rejected
as it requires an extra hyperparameter and can mask systematic model error.

---

*New decisions are appended below as they arise during the pipeline execution.*

---

## 2026-10-29: Data source switched to OpenF1 (FastF1 upstream blocked)

**Decision:** Qualifying laps come from OpenF1 (`io/openf1.py`); FastF1 is kept for cached data only.

**Justification:** `livetiming.formula1.com/static/*` answers HTTP 401 for every season (checked 2026-10-28), so FastF1 cannot download new sessions. OpenF1 serves the same feed (position in 1/10 m, ~3.7 Hz; car data ~3.7 Hz), free for historical data, rate-limited to 30 requests/min (throttle 2.1 s). Deleted laps are removed by comparing with the classified best time (`session_result`).

## 2026-10-29: 2026 calendar and data year per circuit

**Decision:** Model the 2026 calendar as run (OpenF1 `/sessions?year=2026`): 23 rounds incl. Madring (Madrid) and a Kuala Lumpur round; Bahrain and Saudi Arabia (cancelled) kept as extra circuits. Each circuit uses its 2026 qualifying if held, else 2025.

**Justification:** The car is calibrated only on 2026 laps (one regulation era). Circuits not yet raced (Mexico City, Interlagos, Las Vegas, Lusail, Yas Marina) or cancelled get geometry and the real-lap comparison from 2025 – flagged in the app and report.

## 2026-10-29: Canonical frame = F1 timing frame; geometry from TUM, else OSM, else telemetry

**Decision:** Tracks are built in the telemetry frame. TUM is registered onto the fastest lap (multi-start trimmed ICP, reflection tested) and accepted only if the 10 laps satisfy the Phase 3 criteria (median |offset| < half width, ≥ 95 % inside bounds) with a 1.0 m position tolerance; otherwise OSM `type=circuit` relations (closest length to the lap first), then all non-pit raceway ways; driven points are snapped to OSM to get the raced loop. Madring (OSM only 25 % mapped) falls back to the mean driven line (approximate).

**Justification:** The tolerance covers telemetry position error and kerb use (Silverstone: 0.7 % of samples > 2 m outside). Rejected TUM layouts: Melbourne (pre-2022), Catalunya, Spielberg, Zandvoort, Yas Marina (pre-2021).

## 2026-10-29: Admissible width widened to observed car positions

**Decision:** Per 15 m window, the 95th percentile of observed car-centre offsets + 0.5 m pad widens the bounds where real laps go beyond them.

**Justification:** OSM tracks have only a nominal width; real legal laps define the minimum usable width. The quantile prevents single noisy samples from widening the track (widening all of them changed Silverstone's optimum by only 0.4 s).

## 2026-10-29: Reference line low-pass (8 m) with exact edges

**Decision:** The reference (center) line is Gaussian-filtered (σ = 8 m) and the widths are recomputed against the original edges.

**Justification:** OSM-derived center lines have polygon kinks; offsets measured from a kinked reference inherit them. Recomputing widths keeps the admissible region unchanged.

## 2026-10-29: Calibration line and grip utilization factor

**Decision:** θ is fitted on the drivers' own lines (periodic spline through raw positions, rms 1 m), then one grip-utilization factor γ scaling μ, μ_x,brake, μ_x,drive is fitted so the min-time laps match the 2026 poles on average (mean log ratio = 0); γ is also refitted leaving each circuit out (LOCO) for held-out predictions.

**Justification:** Local circle fits to raw positions give realistic peaks (5–6.5 g) but the driven-line curvature is still noisy enough to bias grip up; optimized lines were ~10 % faster than pole with γ = 1. Reconstructing the driven line as reference + smoothed offset was tried and is worse (9–15 g spikes); calibrating on the min-curvature line pinned μ at its bound. γ is the bounded `grip_utilization` option named in the spec (§5.6) and also stands for transient dynamics a point mass ignores.

## 2026-10-29: Bounds widened

**Decision:** μ ≤ 4.0, c_D ≤ 0.002, μ_x,brake ≥ 1.0, μ_x,drive ≥ 0.8, P_spec ∈ [400, 1400], a_brake_cap ≥ 30.

**Justification:** Fits reached the old bounds (μ, c_D, μ_x,brake, P_spec). c_D and P_spec are effective values that also absorb 2026 straight-line energy clipping and active aero.

## 2026-10-29: ICP iterations 60 → 200

**Decision:** `rigid_register(max_iter=200)`.

**Justification:** With 60 iterations a trimmed multi-start ICP returned a wrong local match on a synthetic test (607 m error) because unconverged starts were ranked by median residual; 200 converges (0.08 m).

## 2026-10-29: Fine optimization grid 4 m → 6 m, consistent warm start

**Decision:** `fine_ds_m = 6.0`; the fine NLP is warm-started with the coarse solution's speed and accelerations (not a fresh QSS), capped at 800 iterations, and retried from the QSS state if that fails.

**Justification:** At 4 m, 5 of 25 circuits (Catalunya, Hungaroring, Kuala Lumpur, Interlagos, Jeddah) hit 3000 iterations (up to 50 min) and fell back to inferior lines (+4–20 % vs pole). The optimum is grid-insensitive (Hungaroring: 80.34 s at 10 m, 80.31 s at 6 m); at 6 m all five converge in < 6 s.

## 2026-10-29: Weekend setup from practice + grip by official circuit type

**Decision:** Per circuit, refit downforce c_L and drag c_D on that weekend's last session before qualifying (Practice 3, or Sprint Qualifying on sprint weekends), other parameters held at the season car; fit the grip factor γ separately for OpenF1's official circuit types (Temporary - Street vs Permanent/Road), with leave-one-circuit-out inside each group.

**Justification:** Wing level is set per circuit, and practice runs before qualifying, so the forecast stays honest. Measured on held-out predictions over 18 races: one γ 4.15 % mean |gap| (median 3.63 %, worst 11.5 %); setup only 4.50 %; type groups only 3.01 %; both 3.18 % (median 2.82 %, worst 8.6 %). Both together give the best median and worst case; only official classifications are used (a hand-picked "tight street" group scored better but would be cherry-picking).

## 2026-10-30: Practice correlation of grip per weekend; Suzuka sample fix

**Decision:** Per weekend, grip γ is set so the optimal lap equals the best pre-qualifying lap (FP3 or sprint qualifying) × the median pole/practice ratio of the other races with the same session type. The physics-only prediction (γ by circuit type, leave-one-out) is still reported next to it.

**Justification:** Mirrors how teams correlate simulators to practice before qualifying; uses no qualifying data. Held-out results over 18 races: forecast mean |gap| 0.38 % (median 0.27 %, worst 1.16 %) vs physics only 3.12 % (median 2.72 %, worst 8.56 %). Practice ratio alone (without physics) scores 0.38 % too, so the lap-time accuracy comes mainly from practice pace; the physics supplies the line, speed profile and where time is gained. Reported as such.

**Also:** `lap_on_track` rejected samples via the speed-integrated Distance channel, which drifted 316 m at Suzuka and discarded 64 % of good samples. It now compares each sample with the running median of its neighbours' lap positions; Suzuka's drawn pole path went from 1.9 m to 0.63 m median error vs the raw samples (others 0.4–0.8 m).
