from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Iterable, Tuple

import numpy as np


REGIMES: Dict[str, Tuple[float, float]] = {
    "latent_dominant": (1.80, 1.20),
    "bursting_dominant": (1.20, 1.80),
    "balanced_supercritical": (1.50, 1.50),
}


@dataclass(frozen=True)
class ModelParams:
    """Direct-entry SLBS parameters; ``n_steps`` sets the RK4 resolution."""

    T: float = 60.0
    n_steps: int = 400
    mu: float = 20.0
    delta: float = 0.02
    eta: float = 0.08
    gamma: float = 0.12
    kappa: float = 0.25
    beta1: float = 1.5e-4
    beta2: float = 2.1e-4
    theta1: float = 0.002
    theta2: float = 0.002
    A_L: float = 1.0
    A_B: float = 4.0
    omega_L: float = 1.0
    omega_B: float = 4.0
    C: Tuple[float, float, float] = (40.0, 55.0, 70.0)
    d: Tuple[float, float, float] = (0.2, 0.3, 0.4)
    q: Tuple[float, float, float] = (1.0, 1.5, 2.0)
    umax: Tuple[float, float, float] = (0.9, 0.9, 0.9)
    x0: Tuple[float, float, float] = (970.0, 20.0, 10.0)

    @property
    def h(self) -> float:
        return self.T / self.n_steps

    @property
    def bmax(self) -> float:
        return float(np.dot(self.q, self.umax))

    @property
    def Nstar(self) -> float:
        return self.mu / self.delta

    @property
    def R10(self) -> float:
        return self.beta1 * self.mu / (self.delta * (self.eta + self.delta))

    @property
    def R20(self) -> float:
        return self.beta2 * self.mu / (self.delta * (self.gamma + self.delta))

    def with_updates(self, **kwargs) -> "ModelParams":
        return replace(self, **kwargs)


def params_for_reproduction_number(
    R10: float,
    R20: float,
    **overrides,
) -> ModelParams:
    base = ModelParams().with_updates(**overrides)
    beta1 = R10 * base.delta * (base.eta + base.delta) / base.mu
    beta2 = R20 * base.delta * (base.gamma + base.delta) / base.mu
    return base.with_updates(beta1=beta1, beta2=beta2)


def params_for_regime(name: str, **overrides) -> ModelParams:
    return params_for_reproduction_number(*REGIMES[name], **overrides)


def incidences(x: np.ndarray, p: ModelParams) -> Tuple[float, float]:
    S, L, B = np.asarray(x, dtype=float)
    f1 = p.beta1 * S * L / (1.0 + p.theta1 * L)
    f2 = p.beta2 * S * B / (1.0 + p.theta2 * B)
    return float(f1), float(f2)


def incidence_partials(x: np.ndarray, p: ModelParams) -> Tuple[float, float, float, float]:
    S, L, B = np.asarray(x, dtype=float)
    f1_s = p.beta1 * L / (1.0 + p.theta1 * L)
    f1_l = p.beta1 * S / (1.0 + p.theta1 * L) ** 2
    f2_s = p.beta2 * B / (1.0 + p.theta2 * B)
    f2_b = p.beta2 * S / (1.0 + p.theta2 * B) ** 2
    return float(f1_s), float(f1_l), float(f2_s), float(f2_b)


def state_rhs(x: np.ndarray, u: np.ndarray, p: ModelParams) -> np.ndarray:
    S, L, B = np.asarray(x, dtype=float)
    u1, u2, u3 = np.asarray(u, dtype=float)
    f1, f2 = incidences(x, p)
    repair = (p.gamma + p.kappa * u3) * B
    return np.array(
        [
            p.mu - (1.0 - u1) * f1 - (1.0 - u2) * f2 + repair - p.delta * S,
            (1.0 - u1) * f1 - (p.eta + p.delta) * L,
            (1.0 - u2) * f2 + p.eta * L - (p.gamma + p.kappa * u3 + p.delta) * B,
        ],
        dtype=float,
    )


def state_jacobian(x: np.ndarray, u: np.ndarray, p: ModelParams) -> np.ndarray:
    u1, u2, u3 = np.asarray(u, dtype=float)
    f1_s, f1_l, f2_s, f2_b = incidence_partials(x, p)
    a = (1.0 - u1) * f1_s
    b = (1.0 - u2) * f2_s
    c = (1.0 - u1) * f1_l
    e = (1.0 - u2) * f2_b
    g = p.gamma + p.kappa * u3
    return np.array(
        [
            [-a - b - p.delta, -c, g - e],
            [a, c - p.eta - p.delta, 0.0],
            [b, p.eta, e - g - p.delta],
        ],
        dtype=float,
    )


def adjoint_rhs(lam: np.ndarray, x: np.ndarray, u: np.ndarray, p: ModelParams) -> np.ndarray:
    """Classical adjoint equation ``dot(lambda) = -H_x``."""

    running_gradient = np.array([0.0, p.A_L, p.A_B], dtype=float)
    return -(running_gradient + state_jacobian(x, u, p).T @ np.asarray(lam, dtype=float))


def terminal_gradient(p: ModelParams) -> np.ndarray:
    """Gradient of ``omega_L L(T) + omega_B B(T)``."""

    return np.array([0.0, p.omega_L, p.omega_B], dtype=float)


def marginal_benefits(x: np.ndarray, lam: np.ndarray, p: ModelParams) -> np.ndarray:
    _S, _L, B = np.asarray(x, dtype=float)
    f1, f2 = incidences(x, p)
    ls, ll, lb = np.asarray(lam, dtype=float)
    return np.array(
        [(ll - ls) * f1, (lb - ls) * f2, p.kappa * B * (lb - ls)],
        dtype=float,
    )


def running_cost(x: np.ndarray, u: np.ndarray, p: ModelParams) -> float:
    u = np.asarray(u, dtype=float)
    return float(
        p.A_L * x[1]
        + p.A_B * x[2]
        + np.dot(p.d, u)
        + 0.5 * np.dot(p.C, u * u)
    )


def controlled_thresholds(u: Iterable[float], p: ModelParams) -> Tuple[float, float, float]:
    u1, u2, u3 = np.asarray(tuple(u), dtype=float)
    r1 = (1.0 - u1) * p.R10
    r2 = (1.0 - u2) * p.beta2 * p.mu / (
        p.delta * (p.gamma + p.kappa * u3 + p.delta)
    )
    return float(r1), float(r2), float(max(r1, r2))


def critical_certificate(p: ModelParams, zeta: float = 0.0) -> Dict[str, object]:
    """Least standing capacity for pathway factors at most ``1-zeta``."""

    target = 1.0 - float(zeta)
    if not 0.0 < target <= 1.0:
        raise ValueError("zeta must satisfy 0 <= zeta < 1")
    u1_req = max(0.0, 1.0 - target / p.R10) if p.R10 > 0.0 else 0.0
    umax = np.asarray(p.umax, dtype=float)
    q = np.asarray(p.q, dtype=float)
    if u1_req > umax[0] + 1e-12:
        return {
            "feasible": False,
            "reason": "latent_box_bound",
            "bcrit": np.inf,
            "zeta": float(zeta),
        }
    G_B = p.beta2 * p.mu / p.delta
    D0 = p.gamma + p.delta

    def u2_req(r: float) -> float:
        if G_B == 0.0:
            return 0.0
        return max(0.0, 1.0 - target * (D0 + p.kappa * r) / G_B)

    candidates = [0.0, umax[2]]
    if p.kappa > 0.0:
        candidates.extend(
            [
                (G_B / target - D0) / p.kappa,
                (G_B * (1.0 - umax[1]) / target - D0) / p.kappa,
            ]
        )
    feasible = []
    for raw_r in candidates:
        r = float(np.clip(raw_r, 0.0, umax[2]))
        u2 = u2_req(r)
        if u2 <= umax[1] + 1e-12:
            u = np.array([u1_req, min(u2, umax[1]), r], dtype=float)
            feasible.append((float(q @ u), tuple(u)))
    if not feasible:
        return {
            "feasible": False,
            "reason": "bursting_box_bounds",
            "bcrit": np.inf,
            "zeta": float(zeta),
        }
    bcrit, u_tuple = min(feasible, key=lambda item: (item[0], item[1]))
    u = np.asarray(u_tuple, dtype=float)
    R1, R2, Rc = controlled_thresholds(u, p)
    return {
        "feasible": True,
        "u": u,
        "bcrit": bcrit,
        "R1": R1,
        "R2": R2,
        "Rc": Rc,
        "zeta": float(zeta),
    }


def trajectory_metrics(
    t: np.ndarray,
    x: np.ndarray,
    u_blocks: np.ndarray,
    p: ModelParams,
    b: float,
) -> Dict[str, float]:
    intervals = len(t) - 1
    if intervals % len(u_blocks) != 0:
        raise ValueError("control blocks do not align with the supplied time grid")
    u_steps = np.repeat(u_blocks, intervals // len(u_blocks), axis=0)
    D_L = float(np.trapezoid(x[:, 1], t))
    D_B = float(np.trapezoid(x[:, 2], t))
    running_infection_cost = p.A_L * D_L + p.A_B * D_B
    terminal_cost = p.omega_L * float(x[-1, 1]) + p.omega_B * float(x[-1, 2])
    infection_cost = running_infection_cost + terminal_cost
    operating_rates = u_steps @ np.asarray(p.d) + 0.5 * np.sum(
        np.asarray(p.C)[None, :] * u_steps * u_steps, axis=1
    )
    operating_cost = float(np.sum(operating_rates * np.diff(t)))
    q = np.asarray(p.q, dtype=float)
    umax = np.asarray(p.umax, dtype=float)
    resource = u_blocks @ q
    return {
        "J": infection_cost + operating_cost,
        "D_L": D_L,
        "D_B": D_B,
        "final_infected": float(x[-1, 1] + x[-1, 2]),
        "running_infection_cost": running_infection_cost,
        "terminal_cost": terminal_cost,
        "infection_cost": infection_cost,
        "operating_cost": operating_cost,
        "mean_capacity_use": float(np.mean(resource)),
        "max_budget_violation": float(max(0.0, np.max(resource - b))),
        "max_box_violation": float(
            max(0.0, np.max(-u_blocks), np.max(u_blocks - umax[None, :]))
        ),
    }
