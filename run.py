"""Reproduce the main strategy grid and outbreak-conditioned comparisons."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from slbs.model import ModelParams, critical_certificate, params_for_reproduction_number, trajectory_metrics
from slbs.solver import solve_sampled_pmp, solve_state, simulate_feedback
from slbs.strategies import (
    equal_setting, single_defense_schedule, constant_schedule,
    practical_action, myopic_action, mpc_schedule,
)

ROOT = Path(__file__).resolve().parent


def outbreak_parameters(configuration, scenario):
    """Convert fitted host counts to the per-1000 unit without changing dynamics."""
    fitted = configuration['fits'][scenario['fit']]
    rates = fitted['rates']
    policy = configuration['policy']
    population = fitted['population']
    return ModelParams(
        T=configuration['horizon'], n_steps=configuration['rk4_steps'],
        mu=1000 * rates['delta'], delta=rates['delta'], eta=rates['eta'],
        gamma=rates['gamma'], kappa=scenario['kappa'],
        beta1=rates['a_L']/1000, beta2=rates['a_B']/1000,
        theta1=rates['s_L']/1000, theta2=rates['s_B']/1000,
        x0=tuple(np.asarray(fitted['state_at_control_start']) * 1000 / population),
        A_L=policy['A_L'], A_B=policy['A_B'],
        omega_L=policy['omega_L'], omega_B=policy['omega_B'],
        q=tuple(policy['q']), umax=tuple(policy['umax']),
        d=tuple(policy['d']), C=tuple(policy['C']),
    )


def solve_responses(p, capacity, intervals, configuration, study, practical=True):
    """Compare policies using the same model, capacity, calendar and objective."""
    schedules = {}
    starts = ['profile'] if study == 'main' else ['profile', 'constant', 'zero']
    candidates = [solve_sampled_pmp(
        p, capacity, K=intervals, initial=initial,
        max_sweeps=configuration['pmp_max_sweeps'],
        tol=configuration['pmp_tolerance'], seed=configuration['seed'],
    ) for initial in starts]
    converged = [result for result in candidates if result.converged]
    if not converged:
        raise RuntimeError('No PMP start satisfies the stopping conditions.')
    pmp = min(converged, key=lambda result: result.objective)
    schedules['PMP'] = pmp.u_blocks
    seed = configuration['seed']
    if study == 'main':
        seed += int(round(100*p.R10 + 10*p.R20))
    constant = constant_schedule(
        p, capacity, intervals, starts=configuration['constant_starts'], seed=seed,
    )
    schedules['Constant'] = constant
    schedules['No additional intervention'] = np.zeros((intervals, 3))
    schedules['Matched-mean constant'] = np.repeat(
        np.mean(pmp.u_blocks, axis=0)[None, :], intervals, axis=0)

    if study == 'main':
        schedules['Equal setting'] = np.repeat(
            equal_setting(p, capacity)[None, :], intervals, axis=0)
        certificate = critical_certificate(p)
        if certificate['feasible'] and certificate['bcrit'] <= capacity + 1e-12:
            schedules['Threshold constant'] = np.repeat(
                np.asarray(certificate['u'])[None, :], intervals, axis=0)
        schedules['Single defense'] = single_defense_schedule(p, capacity, intervals)

    if study == 'outbreak' or practical:
        action = practical_action('Equal resource', np.asarray(p.x0), p, capacity)
        schedules['Equal resource'] = np.repeat(action[None, :], intervals, axis=0)
    if practical:
        for name in ['Threshold isolation', 'Alert remediation']:
            _, _, schedules[name] = simulate_feedback(
                p, intervals,
                lambda t, x, name=name: practical_action(name, x, p, capacity))
        _, _, schedules['Myopic'] = simulate_feedback(
            p, intervals, lambda t, x: myopic_action(x, p, capacity, p.T/intervals))
        schedules['MPC'] = mpc_schedule(p, capacity, intervals,
            prediction_intervals=configuration['mpc_prediction_intervals'])
    return schedules


def evaluate_case(name, p, capacity, intervals, schedules, output):
    """Save final policy metrics and held controls for one comparison."""
    rows = []
    controls = []
    states = []
    for method, blocks in schedules.items():
        t, x = solve_state(p, blocks)
        metrics = trajectory_metrics(t, x, blocks, p, capacity)
        rows.append(dict(
            scenario=name, method=method, capacity=capacity,
            cost=metrics['J'], final_compromised=metrics['final_infected'],
            latent_host_time=metrics['D_L'], bursting_host_time=metrics['D_B'],
            operating_cost=metrics['operating_cost'],
            mean_capacity_use=metrics['mean_capacity_use'],
        ))
        for k, action in enumerate(blocks):
            controls.append(dict(scenario=name, method=method, interval=k+1,
                time=k*p.T/intervals, u1=action[0], u2=action[1], u3=action[2]))
        for time, state in zip(t, x):
            states.append(dict(scenario=name, method=method, time=time,
                S=state[0], L=state[1], B=state[2]))
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(controls).to_csv(output/'controls.csv', index=False)
    pd.DataFrame(states).to_csv(output/'trajectories.csv', index=False)
    return rows


def run_main(configuration, selection, output):
    """Run the common-setting comparison on one or all transmission pairs."""
    values = configuration['transmission_factors']
    pairs = [(1.5, 1.5)] if selection == 'central' else [(a,b) for a in values for b in values]
    rows = []
    for latent_factor, bursting_factor in pairs:
        name = f'latent{latent_factor:.1f}_bursting{bursting_factor:.1f}'
        p = params_for_reproduction_number(
            latent_factor, bursting_factor, T=configuration['horizon'],
            n_steps=configuration['rk4_steps'])
        capacity = min(p.bmax, configuration['capacity_factor'] * critical_certificate(p)['bcrit'])
        schedules = solve_responses(
            p, capacity, configuration['control_intervals'], configuration, 'main')
        rows.extend(evaluate_case(name, p, capacity, configuration['control_intervals'],
                                  schedules, output/name))
    return rows


def run_outbreak(configuration, selection, output):
    """Run policy comparisons under the supplied conditional propagation fits."""
    cases = configuration['scenarios']
    if selection == 'central':
        cases = [case for case in cases if case['name'] == 'history80_b50']
    rows = []
    nominal = None
    evaluated = []
    for case in cases:
        p = outbreak_parameters(configuration, case)
        capacity = case['fraction'] * p.bmax
        schedules = solve_responses(
            p, capacity, configuration['control_intervals'], configuration,
            'outbreak', practical=case['role'] == 'primary')
        rows.extend(evaluate_case(case['name'], p, capacity, configuration['control_intervals'],
                                  schedules, output/case['name']))
        evaluated.append((case, p, capacity))
        if case['name'] == 'history80_b50':
            nominal = {name: schedules[name] for name in ['PMP', 'Constant', 'Equal resource']}
    if nominal is not None:
        replay = []
        for case, p, capacity in evaluated:
            if case['fraction'] == .5:
                replay.extend(evaluate_case(case['name'], p, capacity,
                    configuration['control_intervals'], nominal, output/'frozen'/case['name']))
        pd.DataFrame(replay).to_csv(output/'frozen_comparison.csv', index=False)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('study', choices=['main', 'outbreak'])
    parser.add_argument('--all', action='store_true', help='Run all configured scenarios.')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs')
    args = parser.parse_args()
    configuration = json.loads((ROOT/'settings.json').read_text())[args.study]
    output = args.output/args.study
    runner = run_main if args.study == 'main' else run_outbreak
    rows = runner(configuration, 'all' if args.all else 'central', output)
    pd.DataFrame(rows).to_csv(output/'comparison.csv', index=False)


if __name__ == '__main__':
    main()
