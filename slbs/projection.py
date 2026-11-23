from __future__ import annotations

from itertools import product
from typing import Iterable, Tuple

import numpy as np


def project_weighted_capped_simplex(
    z: Iterable[float],
    C: Iterable[float],
    q: Iterable[float],
    umax: Iterable[float],
    b: float,
    tol: float = 1e-13,
) -> Tuple[np.ndarray, float]:
    """Weighted Euclidean projection onto box bounds and ``q @ u <= b``."""

    z = np.asarray(tuple(z), dtype=float)
    C = np.asarray(tuple(C), dtype=float)
    q = np.asarray(tuple(q), dtype=float)
    umax = np.asarray(tuple(umax), dtype=float)
    if z.shape != (3,) or C.shape != (3,) or q.shape != (3,) or umax.shape != (3,):
        raise ValueError("z, C, q, and umax must have length three")
    if np.any(C <= 0.0) or np.any(q <= 0.0) or np.any(umax < 0.0):
        raise ValueError("C and q must be positive and umax nonnegative")
    if not np.isfinite(b) or b < 0.0:
        raise ValueError("b must be finite and nonnegative")
    clipped = np.clip(z, 0.0, umax)
    if float(q @ clipped) <= b:
        return clipped, 0.0
    lo = 0.0
    hi = float(np.max(C * z / q, where=umax > 0.0, initial=0.0))
    feasible = np.zeros(3)
    if b == 0.0:
        return feasible, hi
    for _ in range(120):
        mid = 0.5 * (lo + hi)
        u = np.clip(z - mid * q / C, 0.0, umax)
        if float(q @ u) > b:
            lo = mid
        else:
            hi, feasible = mid, u
        if hi - lo <= tol * max(1.0, hi):
            break
    return feasible, hi


def projection_kkt_residual(
    u: np.ndarray,
    z: np.ndarray,
    C: np.ndarray,
    q: np.ndarray,
    umax: np.ndarray,
    b: float,
    nu: float,
    tol: float = 2e-6,
) -> float:
    stationarity = C * (u - z) + nu * q
    residuals = [
        max(0.0, float(q @ u - b)),
        max(0.0, float(-nu)),
        float(max(0.0, np.max(-u), np.max(u - umax))),
        abs(nu * float(q @ u - b)),
    ]
    for i in np.flatnonzero(umax > 0.0):  # Fixed coordinates have no stationarity sign restriction.
        bound_tol = min(tol, umax[i] / 4)  # Keep the two bounds distinct even for a tiny cap.
        if u[i] <= bound_tol:
            residuals.append(max(0.0, -float(stationarity[i])))
        elif u[i] >= umax[i] - bound_tol:
            residuals.append(max(0.0, float(stationarity[i])))
        else:
            residuals.append(abs(float(stationarity[i])))
    return float(max(residuals))


def enumerate_projection(
    z: Iterable[float], C: Iterable[float], q: Iterable[float], umax: Iterable[float], b: float
) -> np.ndarray:
    """Project onto each face; select the feasible candidate nearest to z."""

    z = np.asarray(tuple(z), dtype=float)
    C = np.asarray(tuple(C), dtype=float)
    q = np.asarray(tuple(q), dtype=float)
    umax = np.asarray(tuple(umax), dtype=float)
    candidates = []
    for states in product((0, 1, 2), repeat=3):
        states = np.asarray(states)
        free = states == 1
        u = np.where(free, z, np.where(states == 2, umax, 0.0))
        excess = q @ u - b
        if excess > 0.0:
            if not np.any(free):
                continue
            direction = q[free] / C[free]
            u[free] -= direction * (excess / (q[free] @ direction))
        if np.all(u >= 0.0) and np.all(u <= umax):
            candidates.append(u)
    return min(candidates, key=lambda u: np.dot(C, (u - z) ** 2))
