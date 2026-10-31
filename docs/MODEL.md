# Vehicle Model Documentation

## Overview

The vehicle model is a **point-mass model with aerodynamic forces and a friction ellipse**.
It captures the essential physics limiting F1 lap performance without requiring a full
multi-body simulation. All quantities are **per unit mass** (SI units: m, s, m/s, m/s²).

---

## Parameters

| Symbol | Variable | Units | Default bounds | Physical meaning |
|--------|----------|-------|---------------|-----------------|
| μ | `mu` | – | [1.5, 3.5] | Lateral grip coefficient |
| μ_x,brake | `mu_x_brake` | – | [1.5, 4.0] | Longitudinal grip under braking |
| μ_x,drive | `mu_x_drive` | – | [1.0, 3.0] | Longitudinal grip under drive |
| c_L | `c_l` | 1/m | [0.0001, 0.002] | Downforce coeff per unit mass = ρCLA/(2m) |
| c_D | `c_d` | 1/m | [0.00005, 0.001] | Drag coeff per unit mass = ρCdA/(2m) |
| P_spec | `p_spec` | W/kg | [500, 1000] | Effective power per unit mass |
| a_brake,cap | `a_brake_cap` | m/s² | [40, 65] | Hard braking cap (thermal/hydraulic) |
| g | `g` | m/s² | 9.81 (fixed) | Gravitational acceleration |

---

## Equations

### Aerodynamic forces (per unit mass)

```
a_downforce(v) = c_L · v²    [m/s²]
a_drag(v)      = c_D · v²    [m/s²]
```

The **aerodynamic load factor** effectively increases the normal force, raising
both the lateral and longitudinal grip limits with speed.

### Lateral grip limit

```
a_lat,max(v) = μ · (g + c_L · v²)
```

At 80 m/s (288 km/h) with μ=2.5, c_L=0.001:
`a_lat,max ≈ 2.5 × (9.81 + 0.001 × 6400) ≈ 40.5 m/s² ≈ 4.1 g`

This is consistent with observed F1 lateral G-forces of 4–5 g in fast corners.

### Braking deceleration limit

```
a_brake,max(v) = min( μ_x,brake · (g + c_L · v²) + c_D · v², a_brake,cap )
```

Drag **assists** braking (adds to deceleration). The hard cap models brake
thermal and hydraulic limitations.

At 80 m/s with μ_x,brake=2.75, c_L=0.001, c_D=0.0005:
`a_brake ≈ 2.75 × 16.21 + 0.0005 × 6400 ≈ 44.6 + 3.2 = 47.8 m/s² ≈ 4.9 g`

Peak F1 braking is around 5–6 g, consistent.

### Drive acceleration limit (net)

```
a_drive,max(v) = min( P_spec / max(v, v_min), μ_x,drive · (g + c_L · v²) ) - c_D · v²
```

At low speed: traction-limited (μ_x,drive term dominates).
At high speed: power-limited (P_spec/v term dominates).
Drag always opposes.

### Friction ellipse

```
(a_lat / a_lat,max)² + (a_long / a_long,max)² ≤ 1
```

Applied separately for drive and braking. Provides the combined grip budget
between lateral and longitudinal acceleration.

### Cornering speed limit

Solving `v² · |κ| = μ · (g + c_L · v²)` for v:
```
v_corner(κ) = sqrt( μ·g / (|κ| - μ·c_L) )    if |κ| > μ·c_L
            = v_max                              otherwise
```

### Top speed (terminal velocity)

Where drive power equals drag:
```
v_top ≈ (P_spec / c_D)^(1/3)
```

For P_spec=750, c_D=0.0005: v_top ≈ (750/0.0005)^(1/3) ≈ 114 m/s ≈ 410 km/h
(Upper bound; actual top speed also limited by track length).

---

## Parameter Bound Justification

### μ ∈ [1.5, 3.5]
- At 0 m/s, `a_lat,max = μ·g`. Observed peak lateral G in slow corners ≈ 3–4 g.
- μ=2.5 gives 24.5 m/s² at 0 speed, consistent with tyre friction limits.
- Upper bound 3.5 leaves room for high-downforce circuits (Monaco, Hungaroring).

### c_L ∈ [0.0001, 0.002]
- c_L = ρ·CLA / (2m). With ρ≈1.2 kg/m³, CLA≈3.5 m², m≈800 kg:
  c_L = 1.2·3.5/(2·800) ≈ 0.0026. Upper bound 0.002 is a reasonable maximum.
- High-downforce cars (Monaco) run near the upper end.

### c_D ∈ [0.00005, 0.001]
- c_D = ρ·CdA / (2m). CdA ≈ 0.7–1.2 m² for F1:
  c_D ≈ 1.2·0.9/(2·800) ≈ 0.00068. Bounds cover low-drag (Monza) to high-drag setups.

### P_spec ∈ [500, 1000] W/kg
- 2022+ PU rated at ~1000 bhp peak ≈ 745 kW. With m≈800 kg: P/m ≈ 930 W/kg.
- Effective drivetrain losses, energy deployment limits → P_spec ≈ 600–900 W/kg.

### μ_x,brake ∈ [1.5, 4.0]
- Peak braking 5–6 g observed. At v=0: a_brake = μ_x,brake · g + cap.
- μ_x,brake=3.0 gives 29.4 m/s² at v=0; aero adds at high speed.

### μ_x,drive ∈ [1.0, 3.0]
- Drive starts traction-limited. At launch μ_x,drive·g ≈ 19.6 m/s² (2 g) – reasonable.

### a_brake_cap ∈ [40, 65] m/s²
- 40 m/s² ≈ 4.1 g, 65 m/s² ≈ 6.6 g. Hardware limit observed at about 5–6 g.

---

## Assumptions and Limitations

1. **Point mass**: no weight transfer between axles; no suspension dynamics.
2. **Constant μ**: no tyre degradation or temperature effects.
3. **No DRS**: drag reduction not modeled (causes ~3–5 km/h underestimate on straights).
4. **No elevation**: Z-channel not used in v1.
5. **No banking**: relevant at some corners (e.g., Bahrain T1, Maggots/Becketts).
6. **No wind**: atmospheric effects ignored.
7. **Symmetric tyres**: front/rear grip not distinguished.
8. **No energy deployment strategy**: MGU-K/MGU-H deployment is implicit in P_spec.

---

## Grip utilization factor γ (added 2026-10-29)

After the telemetry fit θ, one factor γ scales the tyre friction coefficients
(μ, μ_x,brake, μ_x,drive; aero c_L, c_D and power unchanged):

```
a_lat_max(v) = γ μ (g + c_L v²)      (same γ on the longitudinal grip terms)
```

γ is chosen so that the mean over the 2026 circuits of log(T_min-time / T_pole) is zero. The
min-time NLP is solved on a 10 m grid for γ on a grid and interpolated per circuit. Rationale and
alternatives: `docs/DECISIONS.md` (2026-10-29). Leave-one-circuit-out γ values give the held-out
pole predictions reported in `docs/RESULTS.md`.

## QSS stages shown in the app

`vehicle.qss.speed_profile_stages` returns the intermediate profiles of the QSS method:
the cornering limit from curvature alone, the forward (acceleration) pass, the backward
(braking) pass, and the converged profile, which is never above any of them.
