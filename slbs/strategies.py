"""Constant, receding-horizon and practical response strategies."""
from __future__ import annotations

import numpy as np
from scipy.optimize import BFGS, Bounds, LinearConstraint, minimize, minimize_scalar

from .model import critical_certificate, incidences, trajectory_metrics
from .projection import project_weighted_capped_simplex
from .solver import (
    block_average_marginals, rk4_step, solve_adjoint, solve_sampled_pmp,
    solve_state, validate_layout,
)


def project_schedule(raw, p, capacity):
    return np.asarray([project_weighted_capped_simplex(
        row, p.C, p.q, p.umax, capacity)[0] for row in raw])


def equal_setting(p, capacity):
    level = min(float(np.min(p.umax)), capacity / float(np.sum(p.q)))
    return np.full(3, level)


def allocate_resources(weights, p, capacity):
    """Divide resource demand and redistribute shares exceeding deployment caps."""
    weights = np.asarray(weights, float)
    bounds = np.asarray(p.q) * np.asarray(p.umax)
    shares = np.zeros(3)
    free = np.ones(3, dtype=bool)
    remaining = capacity
    for _ in range(3):
        demand = weights * free
        if demand.sum() == 0:
            break
        proposed = remaining * demand / demand.sum()
        capped = (proposed > bounds - shares + 1e-13) & free
        if not capped.any():
            shares += proposed
            break
        remaining -= np.sum(bounds[capped] - shares[capped])
        shares[capped] = bounds[capped]
        free[capped] = False
    return shares / np.asarray(p.q)


def practical_action(method, state, p, capacity, threshold=.01):
    """Equal-resource, threshold-isolation and alert-weighted responses."""
    if method == 'Equal resource':
        return allocate_resources([1, 1, 1], p, capacity)
    if method == 'Threshold isolation':
        if state[2] / np.sum(state) < threshold:
            return allocate_resources([1, 1, 1], p, capacity)
        action = np.zeros(3)
        action[1] = min(p.umax[1], capacity / p.q[1])
        action += allocate_resources([1, 0, 1], p, max(0., capacity-p.q[1]*action[1]))
        return action
    if method == 'Alert remediation':
        return allocate_resources([state[1], state[2], 2*state[2]], p, capacity)
    raise ValueError(f'Unknown response strategy: {method}')


def myopic_action(state, p, capacity, interval):
    """Project immediate propagation and remediation priorities."""
    latent_flow, bursting_flow = incidences(state, p)
    benefit = .5*interval*np.array([
        p.A_L*latent_flow, p.A_B*bursting_flow, p.A_B*p.kappa*state[2]
    ]) - np.asarray(p.d)
    return project_weighted_capped_simplex(
        benefit/np.asarray(p.C), p.C, p.q, p.umax, capacity)[0]


class ScheduleObjective:
    """Finite-horizon cost and continuous-adjoint numerical gradient."""
    def __init__(self, p, capacity, intervals):
        self.p, self.capacity, self.intervals = p, capacity, intervals
        self.key = None

    def evaluate(self, vector):
        vector = np.asarray(vector, float)
        key = vector.tobytes()
        if key == self.key:
            return float(self.value), np.asarray(self.gradient)
        control = vector.reshape(self.intervals, 3)
        time, state = solve_state(self.p, control)
        adjoint = solve_adjoint(self.p, state, control)
        marginals = block_average_marginals(self.p, self.intervals, state, adjoint)
        self.value = float(trajectory_metrics(time, state, control, self.p, self.capacity)['J'])
        self.gradient = (self.p.T/self.intervals * (
            np.asarray(self.p.d)[None, :]
            + np.asarray(self.p.C)[None, :]*control-marginals)).ravel()
        self.key = key
        return self.value, self.gradient

    def fun(self, vector):
        return self.evaluate(vector)[0]

    def jac(self, vector):
        return self.evaluate(vector)[1]


def shooting_schedule(p, capacity, intervals, initial, maxiter=180, ftol=2e-8):
    """Refine a held schedule with constrained direct shooting."""
    validate_layout(p, intervals)
    q = np.asarray(p.q)
    initial = project_schedule(initial, p, capacity)
    objective = ScheduleObjective(p, capacity, intervals)
    constraint_gradient = np.zeros((intervals, 3*intervals))
    for k in range(intervals):
        constraint_gradient[k, 3*k:3*k+3] = -q
    result = minimize(
        objective.fun, initial.ravel(),
        jac=objective.jac, method='SLSQP',
        bounds=[(0., p.umax[i]) for _ in range(intervals) for i in range(3)],
        constraints=[dict(type='ineq', fun=lambda v: capacity-v.reshape(intervals, 3)@q,
                          jac=lambda v: constraint_gradient)],
        options=dict(maxiter=maxiter, ftol=ftol, disp=False))
    control = project_schedule(result.x.reshape(intervals, 3), p, capacity)
    time, state = solve_state(p, control)
    metrics = trajectory_metrics(time, state, control, p, capacity)
    feasible = max(metrics['max_budget_violation'], metrics['max_box_violation']) <= 1e-9
    return control, bool(result.success and feasible), metrics['J']


def constant_starts(p, capacity, count, seed):
    """Initial constant settings for the common multistart comparison."""
    bounds = np.asarray(p.umax)
    values = [np.zeros(3), equal_setting(p, capacity)]
    certificate = critical_certificate(p)
    if certificate['feasible'] and certificate['bcrit'] <= capacity+1e-12:
        values.append(np.asarray(certificate['u']))
    profiles = np.asarray([
        [1, .15, .1], [.15, 1, .1], [.15, .1, 1], [.55, .35, 0], [.75, .2, .1]
    ]) * bounds
    values.extend(project_schedule(profiles, p, capacity))
    rng = np.random.default_rng(seed)
    while len(values) < count:
        values.append(project_schedule([rng.uniform(0., 1., 3)*bounds], p, capacity)[0])
    return values[:count]


def constant_schedule(p, capacity, intervals, starts=20, seed=20260712):
    """Select the lowest-cost stationary constant allocation found by multistart."""
    q, bounds = np.asarray(p.q), np.asarray(p.umax)
    objective = ScheduleObjective(p, capacity, intervals)

    def evaluate(action):
        blocks = np.repeat(np.asarray(action)[None, :], intervals, axis=0)
        value, gradient = objective.evaluate(blocks.ravel())
        return value, gradient.reshape(intervals, 3).sum(axis=0)

    def project(action):
        return project_weighted_capped_simplex(action, p.C, q, bounds, capacity)[0]

    candidates = []
    for initial in constant_starts(p, capacity, starts, seed):
        result = minimize(
            lambda u: evaluate(u)[0], initial, jac=lambda u: evaluate(u)[1],
            method='SLSQP', bounds=list(zip(np.zeros(3), bounds)),
            constraints=[dict(type='ineq', fun=lambda u: capacity-q@u, jac=lambda u: -q)],
            options=dict(maxiter=100, ftol=2e-9, disp=False))
        candidates.append(project(result.x))
    initial = min(candidates, key=lambda u: evaluate(u)[0])
    polished = minimize(
        lambda u: evaluate(u)[0], initial, jac=lambda u: evaluate(u)[1], hess=BFGS(),
        method='trust-constr', bounds=Bounds(np.zeros(3), bounds),
        constraints=[LinearConstraint(q[None, :], -np.inf, [capacity])],
        options=dict(maxiter=1000, gtol=1e-9, xtol=1e-12, barrier_tol=1e-12, verbose=0))
    candidates = [project(polished.x)]
    for fixed in range(3):
        j, k = [i for i in range(3) if i != fixed]
        for value in [0., bounds[fixed]]:
            remaining = capacity-q[fixed]*value
            if remaining < -1e-12:
                continue
            lower = max(0., (remaining-q[k]*bounds[k])/q[j])
            upper = min(bounds[j], remaining/q[j])
            if lower > upper+1e-12:
                continue

            def face(coordinate):
                action = np.zeros(3)
                action[fixed], action[j] = value, coordinate
                action[k] = (remaining-q[j]*coordinate)/q[k]
                return action

            levels = [lower]
            if upper-lower > 1e-13:
                result = minimize_scalar(lambda z: evaluate(face(z))[0],
                    bounds=(lower, upper), method='bounded',
                    options=dict(xatol=1e-12, maxiter=500))
                levels = [lower, float(result.x), upper]
            for coordinate in levels:
                action = face(coordinate)
                if np.min(action) >= -2e-10 and np.max(action-bounds) <= 2e-10 and q@action <= capacity+2e-10:
                    candidates.append(np.clip(action, 0., bounds))

    def stationary(action):
        gradient = evaluate(action)[1]
        scale = max(1., np.max(np.abs(gradient)))
        projected = project_weighted_capped_simplex(
            action-gradient/scale, np.ones(3), q, bounds, capacity)[0]
        return np.max(np.abs(projected-action)) <= 2e-5

    candidates = [action for action in candidates if stationary(action)]
    if not candidates:
        raise RuntimeError('Constant-allocation refinement did not satisfy stationarity.')
    action = min(candidates, key=lambda u: evaluate(u)[0])
    return np.repeat(action[None, :], intervals, axis=0)


def single_defense_schedule(p, capacity, intervals):
    """Compare optimized constant filtering, isolation and remediation alone."""
    candidates = []
    for i in range(3):
        upper = min(p.umax[i], capacity/p.q[i])

        def schedule(level):
            action = np.zeros(3)
            action[i] = level
            return np.repeat(action[None, :], intervals, axis=0)

        def cost(level):
            blocks = schedule(level)
            time, state = solve_state(p, blocks)
            return trajectory_metrics(time, state, blocks, p, capacity)['J']

        result = minimize_scalar(cost, bounds=(0., upper), method='bounded',
                                 options=dict(xatol=1e-10, maxiter=300))
        levels = [0., upper]+([float(result.x)] if result.success else [])
        selected = min(levels, key=cost)
        candidates.append((cost(selected), schedule(selected)))
    return min(candidates, key=lambda item: item[0])[1]


def mpc_plan(local, capacity, holds, previous):
    """Plan from the observed state; shift the last feasible plan for a warm start."""
    initial = (np.tile(equal_setting(local, capacity), (holds, 1)) if previous is None
               else np.vstack((previous[1:], previous[-1:]))[:holds])
    plan, success, _ = shooting_schedule(local, capacity, holds, initial, maxiter=80, ftol=2e-7)
    if success:
        return plan
    fbs = solve_sampled_pmp(local, capacity, K=holds, initial_u=plan, max_sweeps=700)
    polished, success, cost = shooting_schedule(local, capacity, holds, fbs.u_blocks,
                                               maxiter=180, ftol=1e-7)
    candidates = [(fbs.objective, fbs.u_blocks)] if fbs.converged else []
    if success:
        candidates.append((cost, polished))
    return min(candidates, key=lambda item: item[0])[1] if candidates else project_schedule(initial, local, capacity)


def mpc_schedule(p, capacity, intervals, prediction_intervals=4):
    """Recompute a short-horizon plan at each control update."""
    steps = validate_layout(p, intervals)
    interval = p.T/intervals
    state = np.asarray(p.x0, float)
    controls = np.zeros((intervals, 3))
    previous_plan = None
    for k in range(intervals):
        holds = min(prediction_intervals, intervals-k)
        local = p.with_updates(T=holds*interval, n_steps=holds*steps, x0=tuple(state))
        previous_plan = mpc_plan(local, capacity, holds, previous_plan)
        controls[k] = previous_plan[0]
        for _ in range(steps):
            state = rk4_step(state, controls[k], p, p.h)
    return controls
