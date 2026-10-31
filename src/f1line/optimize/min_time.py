"""Stage B: minimum-time NLP optimizer using CasADi + IPOPT.

Discretizes the lap into N points and solves a nonlinear program minimizing
total lap time subject to vehicle dynamics constraints (Section 5.5).

Variables per point:
- alpha_i: lateral offset from center line [m]
- z_i = v_i^2: squared speed [m^2/s^2]
- ap_i >= 0: drive acceleration [m/s^2]
- an_i >= 0: brake deceleration [m/s^2]

Dynamics: z_{i+1} = z_i + 2 * ds_i * (ap_i - an_i)
Friction: (a_lat/a_lat_max)^2 + (ap/a_drive_max)^2 <= 1 (drive)
          (a_lat/a_lat_max)^2 + (an/a_brake_max)^2 <= 1 (brake)
Objective: sum 2*ds_i / (sqrt(z_i+eps) + sqrt(z_{i+1}+eps)) + w_reg * sum curvature^2

Coarse-to-fine: solved first at ``coarse_ds_m``, then at the track's own spacing,
warm-started from the coarse solution. Every IPOPT iterate is recorded so the app can
replay how the line evolves.

Fallback: if IPOPT fails, iterate the min-curvature QP with per-point weights
proportional to 1/v_i^2 from the QSS speed profile (slow corners count more).
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

import numpy as np

from f1line.geometry.track import Track
from f1line.optimize.min_curvature import solve_min_curvature
from f1line.optimize.utils import resample_alpha_to_track
from f1line.vehicle.model import VehicleParams
from f1line.vehicle.qss import QSSResult, simulate_lap_on_line

logger = logging.getLogger(__name__)

_EPS = 1e-6
_EPS_SQRT = 1e-4  # inside sqrt(z + eps_sqrt)
_EPS_KAPPA = 1e-10  # smooth |kappa| = sqrt(kappa^2 + eps)


def _make_iteration_recorder(ca: Any, nx: int, ng: int) -> Any:
    """CasADi callback storing the primal iterate x after every IPOPT iteration."""

    class _Recorder(ca.Callback):
        def __init__(self) -> None:
            ca.Callback.__init__(self)
            self.xs: list[np.ndarray] = []
            self.construct("iteration_recorder", {})

        def get_n_in(self) -> int:
            return ca.nlpsol_n_out()

        def get_n_out(self) -> int:
            return 1

        def get_name_in(self, i: int) -> str:
            return ca.nlpsol_out(i)

        def get_name_out(self, i: int) -> str:
            return "ret"

        def get_sparsity_in(self, i: int) -> Any:
            name = ca.nlpsol_out(i)
            if name == "f":
                return ca.Sparsity.scalar()
            if name in ("x", "lam_x"):
                return ca.Sparsity.dense(nx)
            if name in ("g", "lam_g"):
                return ca.Sparsity.dense(ng)
            return ca.Sparsity(0, 0)

        def eval(self, arg: list) -> list:
            self.xs.append(np.array(arg[0]).ravel().copy())
            return [0]

    return _Recorder()


def _solve_nlp(
    track: Track,
    params: VehicleParams,
    alpha_init: np.ndarray,
    opt_cfg: dict[str, Any],
    state_init: Optional[dict[str, np.ndarray]] = None,
    max_iter: Optional[int] = None,
) -> dict[str, Any]:
    """One IPOPT solve on ``track``'s grid. Returns solution, status and recorded iterates.

    ``state_init`` (keys ``z``, ``ap``, ``an`` on this grid) warm-starts the speed and
    acceleration variables; otherwise they come from a QSS run on ``alpha_init``.
    """
    import casadi as ca

    ipopt_cfg = opt_cfg.get("ipopt", {})
    w_reg = float(opt_cfg.get("w_reg", 1e-4))
    n = track.n
    wl, wr = track.w_left, track.w_right
    p = params

    alpha = ca.SX.sym("alpha", n)
    z = ca.SX.sym("z", n)
    ap = ca.SX.sym("ap", n)
    an = ca.SX.sym("an", n)
    X = ca.vertcat(alpha, z, ap, an)

    px = track.x + alpha * track.nx
    py = track.y + alpha * track.ny
    ip1 = np.roll(np.arange(n), -1)
    im1 = np.roll(np.arange(n), 1)

    t_lap = 0.0
    reg = 0.0
    g_list, lbg, ubg = [], [], []
    for i in range(n):
        j, k = int(ip1[i]), int(im1[i])
        bx, by = px[j] - px[i], py[j] - py[i]
        ax, ay = px[i] - px[k], py[i] - py[k]
        cx, cy = px[j] - px[k], py[j] - py[k]
        ds_i = ca.sqrt(bx**2 + by**2 + _EPS)
        na = ca.sqrt(ax**2 + ay**2 + _EPS)
        nc = ca.sqrt(cx**2 + cy**2 + _EPS)
        kappa_i = 2.0 * (ax * by - ay * bx) / (na * ds_i * nc + _EPS)

        a_lat = z[i] * ca.sqrt(kappa_i**2 + _EPS_KAPPA)
        v_i = ca.sqrt(z[i] + _EPS_SQRT)
        a_lat_max = p.mu * (p.g + p.c_l * z[i])
        a_brake_max = ca.fmin(p.mu_x_brake * (p.g + p.c_l * z[i]) + p.c_d * z[i], p.a_brake_cap)
        a_drive_max = (
            ca.fmin(p.p_spec / ca.fmax(v_i, p.v_min), p.mu_x_drive * (p.g + p.c_l * z[i]))
            - p.c_d * z[i]
        )
        lat_frac2 = (a_lat / a_lat_max) ** 2
        g_list += [
            lat_frac2 + (ap[i] / ca.fmax(a_drive_max, 0.1)) ** 2 - 1.0,
            lat_frac2 + (an[i] / a_brake_max) ** 2 - 1.0,
            z[j] - (z[i] + 2.0 * ds_i * (ap[i] - an[i])),
        ]
        lbg += [-np.inf, -np.inf, 0.0]
        ubg += [0.0, 0.0, 0.0]

        t_lap += 2.0 * ds_i / (v_i + ca.sqrt(z[j] + _EPS_SQRT))
        reg += (alpha[k] - 2 * alpha[i] + alpha[j]) ** 2

    g_vec = ca.vertcat(*g_list)
    obj = t_lap + w_reg * reg

    lbx = np.concatenate([-wr, np.full(n, p.v_min**2), np.zeros(n), np.zeros(n)])
    ubx = np.concatenate([wl, np.full(n, p.v_max**2), np.full(n, p.a_brake_cap),
                          np.full(n, p.a_brake_cap)])

    alpha_ig = np.clip(alpha_init, -wr, wl)
    qss0 = simulate_lap_on_line(*track.point_at(alpha_ig), p)
    if state_init is None:
        z_ig = np.clip(qss0.v**2, p.v_min**2, p.v_max**2)
        ap_ig = np.clip(qss0.a_long, 0.0, p.a_brake_cap)
        an_ig = np.clip(-qss0.a_long, 0.0, p.a_brake_cap)
    else:
        # Warm start from a converged solution (e.g. the coarse grid): keep its speed and
        # accelerations instead of re-deriving them, so the start point is consistent.
        z_ig = np.clip(state_init["z"], p.v_min**2, p.v_max**2)
        ap_ig = np.clip(state_init["ap"], 0.0, p.a_brake_cap)
        an_ig = np.clip(state_init["an"], 0.0, p.a_brake_cap)
    x0 = np.concatenate([alpha_ig, z_ig, ap_ig, an_ig])

    recorder = _make_iteration_recorder(ca, X.shape[0], g_vec.shape[0])
    opts = {
        "ipopt.max_iter": int(max_iter or ipopt_cfg.get("max_iter", 1500)),
        "ipopt.tol": float(ipopt_cfg.get("tol", 1e-6)),
        "ipopt.print_level": int(ipopt_cfg.get("print_level", 0)),
        "ipopt.linear_solver": "mumps",
        "ipopt.sb": "yes",
        "print_time": False,
        "iteration_callback": recorder,
    }
    solver = ca.nlpsol("min_time", "ipopt", {"x": X, "f": obj, "g": g_vec}, opts)
    lap_time_fn = ca.Function("lap_time", [X], [t_lap])

    t0 = time.time()
    try:
        sol = solver(x0=x0, lbx=lbx, ubx=ubx, lbg=np.array(lbg), ubg=np.array(ubg))
        stats = solver.stats()
        status = stats["return_status"]
        success = bool(stats["success"]) or status == "Solved_To_Acceptable_Level"
        x_opt = np.array(sol["x"]).ravel()
    except RuntimeError as exc:
        status, success, x_opt = f"exception: {exc}", False, x0
        stats = {}
    elapsed = time.time() - t0

    iterates = recorder.xs
    return {
        "x": x_opt,
        "alpha": np.clip(x_opt[:n], -wr, wl),
        "z": x_opt[n : 2 * n],
        "ap": x_opt[2 * n : 3 * n],
        "an": x_opt[3 * n : 4 * n],
        "lap_time_s": float(lap_time_fn(x_opt)),
        "status": status,
        "success": success,
        "iterations": int(stats.get("iter_count", len(iterates))),
        "elapsed_s": elapsed,
        "iterate_alphas": [it[:n].copy() for it in iterates],
        "iterate_lap_times": [float(lap_time_fn(it)) for it in iterates],
        "qss_init_lap_time_s": qss0.lap_time_s,
    }


def _fallback_weighted_qp(
    track: Track, params: VehicleParams, alpha_init: np.ndarray, max_iter: int = 8
) -> tuple[np.ndarray, list[np.ndarray], list[float]]:
    """Iterate the min-curvature QP with 1/v^2 weights until the QSS lap time stops improving."""
    alpha = alpha_init.copy()
    best_lt = simulate_lap_on_line(*track.point_at(alpha), params).lap_time_s
    alphas, lts = [], []
    for it in range(max_iter):
        v = simulate_lap_on_line(*track.point_at(alpha), params).v
        w = (np.median(v) / np.maximum(v, params.v_min)) ** 2
        cand = solve_min_curvature(track, lambda_reg=1e-4, weights=w)
        lt = simulate_lap_on_line(*track.point_at(cand), params).lap_time_s
        logger.info("Fallback QP iteration %d: lap time %.3f s", it + 1, lt)
        if lt >= best_lt - 1e-3:
            break
        alpha, best_lt = cand, lt
        alphas.append(alpha.copy())
        lts.append(lt)
    return alpha, alphas, lts


def solve_min_time(
    track: Track,
    params: VehicleParams,
    alpha_init: np.ndarray,
    cfg: dict[str, Any],
    history: Optional[list[dict[str, Any]]] = None,
) -> tuple[np.ndarray, QSSResult, dict[str, Any]]:
    """Solve the minimum-time NLP (coarse-to-fine) using CasADi + IPOPT.

    Parameters
    ----------
    track:
        Track object at the fine spacing.
    params:
        Calibrated vehicle parameters.
    alpha_init:
        Initial guess for lateral offsets on ``track`` (from Stage A). Shape (N,).
    cfg:
        Full config dict (uses the 'optimize' sub-dict).
    history:
        If given, receives one dict per recorded iterate with keys ``stage``,
        ``iteration``, ``alpha`` (on ``track``'s grid) and ``nlp_lap_time_s``.

    Returns
    -------
    (alpha_opt, qss_result, info_dict)
        ``alpha_opt``: optimal lateral offsets [m]. Shape (N,).
        ``qss_result``: independent QSS simulation on the resulting line.
        ``info_dict``: solver diagnostics, including the method that produced the result.
    """
    opt_cfg = cfg.get("optimize", {})
    coarse_ds = float(opt_cfg.get("coarse_ds_m", 10.0))
    info: dict[str, Any] = {"n_points": track.n}

    alpha_start = alpha_init
    coarse_alpha: Optional[np.ndarray] = None
    if coarse_ds > track.ds * 1.5:
        coarse = track.resampled(coarse_ds)
        a0 = resample_alpha_to_track(alpha_init, track, coarse)
        logger.info("Min-time coarse solve: N=%d (ds=%.1f m)", coarse.n, coarse.ds)
        res_c = _solve_nlp(coarse, params, a0, opt_cfg)
        logger.info("Coarse: %s in %.1f s, %d iterations, lap %.3f s",
                    res_c["status"], res_c["elapsed_s"], res_c["iterations"], res_c["lap_time_s"])
        info["coarse"] = {k: res_c[k] for k in ("status", "success", "iterations", "elapsed_s",
                                                  "lap_time_s")}
        if history is not None:
            for i, (a, lt) in enumerate(zip(res_c["iterate_alphas"], res_c["iterate_lap_times"])):
                history.append({"stage": "min-time (coarse)", "iteration": i,
                                "alpha": resample_alpha_to_track(a, coarse, track),
                                "nlp_lap_time_s": lt})
        state_init = None
        if res_c["success"]:
            alpha_start = resample_alpha_to_track(res_c["alpha"], coarse, track)
            coarse_alpha = alpha_start
            state_init = {k: resample_alpha_to_track(res_c[k], coarse, track)
                          for k in ("z", "ap", "an")}
    else:
        state_init = None

    logger.info("Min-time fine solve: N=%d (ds=%.1f m)", track.n, track.ds)
    res = _solve_nlp(track, params, alpha_start, opt_cfg, state_init=state_init,
                     max_iter=opt_cfg.get("fine_max_iter"))
    if not res["success"] and state_init is not None:
        logger.warning("Fine solve from coarse state failed (%s); retrying from QSS state",
                       res["status"])
        res = _solve_nlp(track, params, alpha_start, opt_cfg,
                         max_iter=opt_cfg.get("fine_max_iter"))
    logger.info("Fine: %s in %.1f s, %d iterations, lap %.3f s",
                res["status"], res["elapsed_s"], res["iterations"], res["lap_time_s"])
    if history is not None:
        for i, (a, lt) in enumerate(zip(res["iterate_alphas"], res["iterate_lap_times"])):
            history.append({"stage": "min-time", "iteration": i, "alpha": a,
                            "nlp_lap_time_s": lt})

    info.update(
        solver_status=res["status"], ipopt_success=res["success"],
        iterations=res["iterations"], elapsed_s=res["elapsed_s"],
        nlp_lap_time_s=res["lap_time_s"],
    )
    if res["success"]:
        alpha_opt = res["alpha"]
        info["method"] = "ipopt"
    else:
        logger.warning("IPOPT did not converge (%s); running weighted-QP fallback", res["status"])
        alpha_opt, f_alphas, f_lts = _fallback_weighted_qp(track, params, alpha_init)
        info["method"] = "weighted-qp-fallback"
        if history is not None:
            for i, (a, lt) in enumerate(zip(f_alphas, f_lts)):
                history.append({"stage": "weighted-QP fallback", "iteration": i, "alpha": a,
                                "nlp_lap_time_s": lt})
        # Judge every candidate line with the independent QSS simulator and keep the fastest:
        # the fallback, the converged coarse solution (interpolated to this grid) and the last
        # (unconverged, within bounds) IPOPT iterate, which is often within tolerance of done.
        candidates = {"weighted-qp-fallback": alpha_opt, "ipopt-unconverged": res["alpha"]}
        if coarse_alpha is not None:
            candidates["ipopt-coarse"] = np.clip(coarse_alpha, -track.w_right, track.w_left)
        lap = {k: simulate_lap_on_line(*track.point_at(a), params).lap_time_s
               for k, a in candidates.items()}
        info["method"] = min(lap, key=lap.get)
        info["candidate_lap_times_s"] = lap
        alpha_opt = candidates[info["method"]]

    qss_result = simulate_lap_on_line(*track.point_at(alpha_opt), params)
    info["qss_lap_time_s"] = qss_result.lap_time_s
    if info["method"] == "ipopt":
        pct = abs(res["lap_time_s"] - qss_result.lap_time_s) / qss_result.lap_time_s * 100
        info["nlp_qss_diff_pct"] = pct
        logger.info("NLP/QSS cross-check: NLP=%.3f s, QSS=%.3f s, diff=%.2f %%",
                    res["lap_time_s"], qss_result.lap_time_s, pct)
        if pct > 1.0:
            logger.warning("NLP/QSS discrepancy > 1%%: investigate!")
    return alpha_opt, qss_result, info
