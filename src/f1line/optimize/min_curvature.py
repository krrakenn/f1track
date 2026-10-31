"""Stage A: minimum-curvature QP optimizer.

Minimizes the sum of squared curvature-proxy terms via a convex QP in
the lateral offset variables alpha. Uses cvxpy + OSQP with sparse matrices.

Problem (Section 5.4):
    Minimize  sum |p_{i-1} - 2*p_i + p_{i+1}|^2 + lambda * sum |p_{i+1} - p_i|^2
    subject to: -w_right_i <= alpha_i <= w_left_i
    where  p_i = c_i + alpha_i * n_i

With the normals fixed to the reference line, p is affine in alpha
(p_x = c_x + diag(n_x) alpha), so with the periodic difference operators D2, D1:
    D2 p_x = D2 c_x + D2 diag(n_x) alpha  =: b_x + A_x alpha
and the objective is ||A alpha + b||^2 – a sparse least-squares problem with bounds.

Optional re-linearization: after solving, rebuild the reference line through the
result, recompute normals, and re-solve (offsets are re-expressed w.r.t. the track).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import scipy.sparse as sp

from f1line.geometry.track import Track
from f1line.vehicle.model import VehicleParams
from f1line.vehicle.qss import simulate_lap_on_line

logger = logging.getLogger(__name__)


def _periodic_diff(n: int, order: int) -> sp.csr_matrix:
    """Sparse periodic finite-difference operator (order 1: p_{i+1}-p_i; 2: p_{i-1}-2p_i+p_{i+1})."""
    if order == 1:
        return sp.diags([-1.0, 1.0, 1.0], [0, 1, -(n - 1)], shape=(n, n), format="csr")
    return sp.diags(
        [-2.0, 1.0, 1.0, 1.0, 1.0], [0, 1, -1, n - 1, -(n - 1)], shape=(n, n), format="csr"
    )


def _least_squares_terms(
    cx: np.ndarray, cy: np.ndarray, nx: np.ndarray, ny: np.ndarray, lambda_reg: float,
    weights: Optional[np.ndarray] = None,
) -> tuple[sp.csr_matrix, np.ndarray]:
    """Stack the curvature and length residuals as ``A alpha + b`` (sparse A)."""
    n = len(cx)
    D2 = _periodic_diff(n, 2)
    D1 = _periodic_diff(n, 1)
    W = sp.diags(np.sqrt(weights)) if weights is not None else sp.identity(n)
    Nx, Ny = sp.diags(nx), sp.diags(ny)
    rl = np.sqrt(lambda_reg)
    A = sp.vstack([W @ D2 @ Nx, W @ D2 @ Ny, rl * D1 @ Nx, rl * D1 @ Ny]).tocsr()
    b = np.concatenate([W @ (D2 @ cx), W @ (D2 @ cy), rl * (D1 @ cx), rl * (D1 @ cy)])
    return A, b


def _solve_qp(A: sp.csr_matrix, b: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> Optional[np.ndarray]:
    """min ||A x + b||^2 s.t. lb <= x <= ub with cvxpy/OSQP; None on failure."""
    import cvxpy as cp

    x = cp.Variable(A.shape[1])
    prob = cp.Problem(cp.Minimize(cp.sum_squares(A @ x + b)), [x >= lb, x <= ub])
    try:
        prob.solve(solver=cp.OSQP, eps_abs=1e-7, eps_rel=1e-7, max_iter=200_000, verbose=False)
    except cp.SolverError as exc:
        logger.warning("OSQP failed: %s", exc)
        return None
    if prob.status not in ("optimal", "optimal_inaccurate") or x.value is None:
        logger.warning("CVXPY status: %s", prob.status)
        return None
    return np.clip(np.asarray(x.value), lb, ub)


def solve_min_curvature(
    track: Track,
    lambda_reg: float = 1e-4,
    n_relinearize: int = 0,
    weights: Optional[np.ndarray] = None,
    history: Optional[list[np.ndarray]] = None,
) -> np.ndarray:
    """Solve the min-curvature QP and return lateral offsets alpha.

    Parameters
    ----------
    track:
        Track object.
    lambda_reg:
        Regularization weight (lambda in the spec).
    n_relinearize:
        Number of re-linearization iterations (0 = solve once).
    weights:
        Optional per-point weights on the curvature term (used by the min-time fallback).
    history:
        If given, the offsets after every (re-)solve are appended to it.

    Returns
    -------
    np.ndarray
        Optimal lateral offsets [m] w.r.t. the track center line. Shape (N,).
    """
    lb, ub = -track.w_right, track.w_left
    A, b = _least_squares_terms(track.x, track.y, track.nx, track.ny, lambda_reg, weights)
    alpha = _solve_qp(A, b, lb, ub)
    if alpha is None:
        logger.warning("Min-curvature QP failed; returning center line")
        return np.zeros(track.n)
    if history is not None:
        history.append(alpha.copy())

    for it in range(n_relinearize):
        # Reference line through the current solution; normals from its finite differences.
        px, py = track.point_at(alpha)
        tx = np.roll(px, -1) - np.roll(px, 1)
        ty = np.roll(py, -1) - np.roll(py, 1)
        tn = np.hypot(tx, ty) + 1e-12
        rnx, rny = -ty / tn, tx / tn
        # Offset beta along the new normals; convert bounds approximately via the
        # projection of the new normal on the track normal.
        cosang = np.clip(rnx * track.nx + rny * track.ny, 0.3, 1.0)
        lb_b = (lb - alpha) / cosang
        ub_b = (ub - alpha) / cosang
        A, b = _least_squares_terms(px, py, rnx, rny, lambda_reg, weights)
        beta = _solve_qp(A, b, lb_b, ub_b)
        if beta is None:
            break
        new_alpha = np.clip(alpha + beta * cosang, lb, ub)
        change = float(np.abs(new_alpha - alpha).max())
        alpha = new_alpha
        if history is not None:
            history.append(alpha.copy())
        logger.debug("Re-linearization %d: max change %.3f m", it + 1, change)
        if change < 0.05:
            break

    logger.info(
        "Min-curvature QP: lambda=%.1e, |alpha|_max=%.2f m", lambda_reg, np.abs(alpha).max()
    )
    return alpha


def sweep_lambda_and_select(
    track: Track,
    params: VehicleParams,
    lambda_grid: list[float],
    n_relinearize: int = 0,
) -> tuple[np.ndarray, float, float, list[dict]]:
    """Sweep regularization weights and select the one minimizing QSS lap time.

    Parameters
    ----------
    track:
        Track object.
    params:
        Vehicle model parameters.
    lambda_grid:
        List of lambda values to try.
    n_relinearize:
        Re-linearization iterations per solve.

    Returns
    -------
    (alpha_best, lambda_best, lap_time_best, sweep)
        ``sweep``: one dict per lambda with keys ``lambda``, ``alpha``, ``lap_time_s``.
    """
    sweep = []
    for lam in lambda_grid:
        alpha = solve_min_curvature(track, lambda_reg=float(lam), n_relinearize=n_relinearize)
        px, py = track.point_at(alpha)
        lt = simulate_lap_on_line(px, py, params).lap_time_s
        logger.info("lambda=%.1e -> lap_time=%.3f s", lam, lt)
        sweep.append({"lambda": float(lam), "alpha": alpha, "lap_time_s": lt})

    best = min(sweep, key=lambda r: r["lap_time_s"])
    logger.info("Best lambda=%.1e, lap_time=%.3f s", best["lambda"], best["lap_time_s"])
    return best["alpha"], best["lambda"], best["lap_time_s"], sweep
