from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np

from .model import ModelParams, adjoint_rhs, marginal_benefits, state_rhs, terminal_gradient, trajectory_metrics
from .projection import project_weighted_capped_simplex, projection_kkt_residual

def validate_layout(p: ModelParams, K: int) -> int:
    if K <= 0 or p.n_steps <= 0 or p.n_steps % K != 0:
        raise ValueError("K must be positive and divide n_steps")
    return p.n_steps // K


def expand_blocks(u_blocks: np.ndarray, p: ModelParams) -> np.ndarray:
    u_blocks = np.asarray(u_blocks, dtype=float)
    if u_blocks.ndim != 2 or u_blocks.shape[1] != 3:
        raise ValueError("u_blocks must have shape (K, 3)")
    steps = validate_layout(p, len(u_blocks))
    return np.repeat(u_blocks, steps, axis=0)


def rk4_step(x: np.ndarray, u: np.ndarray, p: ModelParams, h: float) -> np.ndarray:
    k1 = state_rhs(x, u, p)
    k2 = state_rhs(x + 0.5 * h * k1, u, p)
    k3 = state_rhs(x + 0.5 * h * k2, u, p)
    k4 = state_rhs(x + h * k3, u, p)
    return x + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def solve_state(
    p: ModelParams,
    u_blocks: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Integrate the SLBS state under the held schedule."""

    u_steps = expand_blocks(u_blocks, p)
    t = np.linspace(0.0, p.T, p.n_steps + 1)
    x = np.empty((p.n_steps + 1, 3), dtype=float)
    x[0] = np.asarray(p.x0, dtype=float)
    for n in range(p.n_steps):
        x[n + 1] = rk4_step(x[n], u_steps[n], p, p.h)
        if not np.all(np.isfinite(x[n + 1])) or np.min(x[n + 1]) < -1e-7:
            raise RuntimeError(f"state integration failed at step {n + 1}")
    return t, x


def solve_adjoint(p: ModelParams, x: np.ndarray, u_blocks: np.ndarray) -> np.ndarray:
    """Backward RK4 solve with ``lambda(T)=gradient Phi(x(T))``."""

    x = np.asarray(x, dtype=float)
    if x.shape != (p.n_steps + 1, 3):
        raise ValueError("x has incompatible shape")
    u_steps = expand_blocks(u_blocks, p)
    lam = np.zeros_like(x)
    lam[-1] = terminal_gradient(p)
    dt = -p.h
    for n in range(p.n_steps - 1, -1, -1):
        u = u_steps[n]
        x_hi = x[n + 1]
        x_mid = 0.5 * (x[n] + x[n + 1])
        x_lo = x[n]
        y = lam[n + 1]
        k1 = adjoint_rhs(y, x_hi, u, p)
        k2 = adjoint_rhs(y + 0.5 * dt * k1, x_mid, u, p)
        k3 = adjoint_rhs(y + 0.5 * dt * k2, x_mid, u, p)
        k4 = adjoint_rhs(y + dt * k3, x_lo, u, p)
        lam[n] = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return lam


def simulate_feedback(
    p: ModelParams,
    K: int,
    policy: Callable[[float, np.ndarray], np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate a deployment-time state-feedback law with zero-order hold."""

    steps = validate_layout(p, K)
    t = np.linspace(0.0, p.T, p.n_steps + 1)
    x = np.empty((p.n_steps + 1, 3), dtype=float)
    x[0] = np.asarray(p.x0, dtype=float)
    u_blocks = np.zeros((K, 3), dtype=float)
    for k in range(K):
        start = k * steps
        u_blocks[k] = np.asarray(policy(t[start], x[start].copy()), dtype=float)
        for n in range(start, start + steps):
            x[n + 1] = rk4_step(x[n], u_blocks[k], p, p.h)
    return t, x, u_blocks


@dataclass
class PMPResult:
    t: np.ndarray
    x: np.ndarray
    lam: np.ndarray
    u_blocks: np.ndarray
    block_marginals: np.ndarray
    nu_blocks: np.ndarray
    converged: bool
    fixed_point_residual: float
    kkt_residual: float
    objective: float


def block_average_marginals(
    p: ModelParams,
    K: int,
    x: np.ndarray,
    lam: np.ndarray,
) -> np.ndarray:
    steps = validate_layout(p, K)
    m = np.asarray([marginal_benefits(xn, ln, p) for xn, ln in zip(x, lam)])
    out = np.zeros((K, 3), dtype=float)
    for k in range(K):
        start = k * steps
        stop = (k + 1) * steps
        out[k] = np.trapezoid(m[start : stop + 1], dx=p.h, axis=0) / (steps * p.h)
    return out


def project_marginals(
    marginal_blocks: np.ndarray,
    p: ModelParams,
    b: float,
) -> tuple[np.ndarray, np.ndarray]:
    C = np.asarray(p.C, dtype=float)
    d = np.asarray(p.d, dtype=float)
    q = np.asarray(p.q, dtype=float)
    umax = np.asarray(p.umax, dtype=float)
    targets = np.zeros_like(marginal_blocks, dtype=float)
    nu = np.zeros(len(targets), dtype=float)
    for k, marginal in enumerate(marginal_blocks):
        targets[k], nu[k] = project_weighted_capped_simplex(
            (marginal - d) / C, C, q, umax, b
        )
    return targets, nu


def _initial_control(p: ModelParams, K: int, b: float, kind: str, seed: int) -> np.ndarray:
    q = np.asarray(p.q, dtype=float)
    umax = np.asarray(p.umax, dtype=float)
    if kind == "zero":
        return np.zeros((K, 3))
    if kind == "constant":
        raw = np.repeat((0.55 * umax)[None, :], K, axis=0)
    elif kind == "profile":
        s = np.linspace(0.0, 1.0, K)
        raw = np.column_stack(
            [
                umax[0] * (0.12 + 0.70 * np.exp(-3.2 * s)),
                umax[1] * (0.10 + 0.68 * np.exp(-((s - 0.25) / 0.35) ** 2)),
                umax[2] * (0.10 + 0.62 * np.exp(-((s - 0.40) / 0.38) ** 2)),
            ]
        )
    elif kind == "random":
        raw = np.random.default_rng(seed).random((K, 3)) * umax
    else:
        raise ValueError(f"unknown initial control: {kind}")
    scale = np.minimum(1.0, b / np.maximum(raw @ q, 1e-30))
    return raw * scale[:, None]


def solve_sampled_pmp(
    p: ModelParams,
    b: float,
    K: int = 20,
    initial: str = "profile",
    initial_u: Optional[np.ndarray] = None,
    max_sweeps: int = 700,
    tol: float = 1e-6,
    alpha_fbs: float = 0.05,
    alpha_min: float = 0.003,
    seed: int = 20260712,
) -> PMPResult:
    """Solve the sampled-data PMP system by relaxed forward--backward iteration."""

    validate_layout(p, K)
    if b < 0.0 or b > p.bmax + 1e-12:
        raise ValueError("b is outside the feasible capacity range")
    if initial_u is None:
        u = _initial_control(p, K, b, initial, seed)
    else:
        u = np.asarray(initial_u, dtype=float).copy()
        if u.shape != (K, 3):
            raise ValueError("initial_u must have shape (K, 3)")
    q = np.asarray(p.q, dtype=float)
    umax = np.asarray(p.umax, dtype=float)
    if np.max(u @ q - b) > 1e-10 or np.min(u) < -1e-10 or np.max(u - umax) > 1e-10:
        raise ValueError("initial_u is infeasible")
    previous_residual = np.inf
    growth = 0
    if max_sweeps < 0:
        raise ValueError("max_sweeps must be nonnegative")
    for sweep in range(max_sweeps + 1):
        t, x = solve_state(p, u)
        lam = solve_adjoint(p, x, u)
        marginal = block_average_marginals(p, K, x, lam)
        target, nu = project_marginals(marginal, p, b)
        residual = float(np.max(np.abs(target - u)))
        if residual <= tol or sweep == max_sweeps:
            break
        if residual > 1.015 * previous_residual:
            growth += 1
        else:
            growth = max(0, growth - 1)
        if growth >= 3 and alpha_fbs > alpha_min:
            alpha_fbs = max(alpha_min, 0.5 * alpha_fbs)
            growth = 0
        u = (1.0 - alpha_fbs) * u + alpha_fbs * target
        u = np.clip(u, 0.0, umax)
        previous_residual = residual

    kkt = 0.0
    C = np.asarray(p.C, dtype=float)
    d = np.asarray(p.d, dtype=float)
    for k in range(K):
        z = (marginal[k] - d) / C
        kkt = max(kkt, projection_kkt_residual(u[k], z, C, q, umax, b, nu[k]))
    resource = u @ q
    budget_violation = float(max(0.0, np.max(resource - b)))
    objective = trajectory_metrics(t, x, u, p, b)["J"]
    converged = bool(
        sweep < max_sweeps and residual <= tol
        and budget_violation <= 1e-10
        and kkt <= 2e-4
    )
    return PMPResult(
        t=t,
        x=x,
        lam=lam,
        u_blocks=u,
        block_marginals=marginal,
        nu_blocks=nu,
        converged=converged,
        fixed_point_residual=residual,
        kkt_residual=kkt,
        objective=objective,
    )
