"""Sensitivity, finite-population and fitted-update studies."""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from slbs.learning import (load_assets, projected_policy, scenario_params,
                           scenarios, feature_vector, train_models)
from slbs.model import (params_for_regime, params_for_reproduction_number,
                        critical_certificate, trajectory_metrics)
from slbs.solver import solve_sampled_pmp, solve_state, simulate_feedback, rk4_step
from slbs.stochastic import simulate_finite_node_slbs, simulate_ssa_sampled, bootstrap_mean_ci
from slbs.strategies import (constant_schedule, equal_setting, mpc_plan, mpc_schedule,
                             myopic_action, practical_action)

SEED, K = 20260712, 20
REGIMES = ['latent_dominant', 'balanced_supercritical', 'bursting_dominant']
ROOT = Path(__file__).resolve().parent


def pmp(p, b, intervals=K, sweeps=1400):
    result = solve_sampled_pmp(p, b, K=intervals, max_sweeps=sweeps,
                             tol=1e-6, alpha_fbs=.05, alpha_min=.003, seed=SEED)
    if not result.converged:
        raise RuntimeError(f'FBS did not converge: residual={result.fixed_point_residual:g}')
    return result


def capacity(p):
    return min(p.bmax, 1.1*critical_certificate(p)['bcrit'])


def record(p, b, method, trajectory, **fields):
    time, state, control = trajectory
    metrics = trajectory_metrics(time, state, control, p, b)
    if max(metrics.pop('max_budget_violation'), metrics.pop('max_box_violation')) > 1e-9:
        raise RuntimeError('An issued control violates its deployment limits.')
    used = np.sum(control*np.asarray(p.q), axis=0)
    share = used/used.sum() if used.sum() else np.zeros(3)
    return dict(metrics, method=method, b=b, K=len(control), R10=p.R10, R20=p.R20,
        omega_L=p.omega_L, omega_B=p.omega_B, final_S=state[-1, 0],
        final_L=state[-1, 1], final_B=state[-1, 2],
        share_filtering=share[0], share_isolation=share[1], share_remediation=share[2], **fields)


def scheduled(p, b, method, control, **fields):
    time, state = solve_state(p, control)
    return record(p, b, method, (time, state, control), **fields)


def pmp_row(p, b, intervals=K, sweeps=1400, **fields):
    result = pmp(p, b, intervals, sweeps)
    return record(p, b, 'PMP', (result.t, result.x, result.u_blocks),
        integrated_capacity_multiplier=p.T/intervals*sum(result.nu_blocks), **fields)


def standard_rows(p, b, seed, intervals=K, include_mpc=True, all_baselines=False,
                  sweeps=1400, **fields):
    rows = [pmp_row(p, b, intervals, sweeps, **fields)]
    constant = constant_schedule(p, b, intervals, starts=20, seed=seed)
    rows.append(scheduled(p, b, 'Constant', constant, **fields))
    if include_mpc:
        controls = mpc_schedule(p, b, intervals, prediction_intervals=round(12/(p.T/intervals)))
        rows.append(scheduled(p, b, 'MPC', controls, **fields))
    if all_baselines:
        schedules = {'No additional intervention': np.zeros((intervals, 3)),
            'Equal setting': np.tile(equal_setting(p, b), (intervals, 1))}
        cert = critical_certificate(p)
        if cert['feasible'] and cert['bcrit'] <= b+1e-12:
            schedules['Threshold constant'] = np.tile(cert['u'], (intervals, 1))
        else:
            rows.append(dict(method='Threshold constant', status='infeasible', b=b, **fields))
        rows.extend(scheduled(p, b, name, u, **fields) for name, u in schedules.items())
        policy = lambda t, x: myopic_action(x, p, b, p.T/intervals)
        rows.append(record(p, b, 'Myopic', simulate_feedback(p, intervals, policy), **fields))
    return rows


def run_sensitivity(study, all_settings):
    rows = []
    regimes = REGIMES if all_settings else ['balanced_supercritical']
    if study == 'capacity_curve':
        p = params_for_regime('balanced_supercritical', n_steps=400)
        previous = None
        for b in [.4, .6, .8, 1., 1.2, 1.4]:
            result = solve_sampled_pmp(p, b, K=K, initial_u=previous, max_sweeps=1400)
            if not result.converged:
                raise RuntimeError('Capacity-curve FBS did not converge.')
            previous = result.u_blocks.copy()
            rows.append(record(p, b, 'PMP', (result.t, result.x, result.u_blocks),
                integrated_capacity_multiplier=p.T/K*sum(result.nu_blocks)))
        return rows
    if study == 'mpc_horizon':
        values = [1.2, 1.5, 1.8] if all_settings else [1.5]
        for r1 in values:
            for r2 in values:
                p = params_for_reproduction_number(r1, r2, n_steps=200)
                b = capacity(p)
                for holds in [3, 4, 6]:
                    control = mpc_schedule(p, b, K, prediction_intervals=holds)
                    rows.append(scheduled(p, b, f'MPC-H{holds}', control,
                        horizon_holds=holds, prediction_horizon_time=holds*p.T/K))
        return rows
    if study == 'parameters':
        base = params_for_regime('balanced_supercritical', n_steps=400)
        b = critical_certificate(base)['bcrit']
        for parameter, values in [('R10', [1.2, 1.35, 1.5, 1.65, 1.8]),
                                  ('R20', [1.2, 1.35, 1.5, 1.65, 1.8]),
                                  ('theta', [.0005, .001, .002, .003, .004])]:
            for value in values if all_settings else [values[0]]:
                p = params_for_reproduction_number(value if parameter == 'R10' else 1.5,
                    value if parameter == 'R20' else 1.5, n_steps=400,
                    theta1=value if parameter == 'theta' else .002,
                    theta2=value if parameter == 'theta' else .002)
                rows.append(pmp_row(p, b, parameter=parameter, value=value))
        return rows
    for regime in regimes:
        index = REGIMES.index(regime)
        base = params_for_regime(regime, n_steps=400 if study in ['calendar', 'capacity_load'] else 200)
        b = capacity(base)
        if study == 'terminal':
            for left, right in [(0., 0.), (1., 4.), (5., 20.)]:
                p = base.with_updates(omega_L=left, omega_B=right)
                compared = standard_rows(p, b, SEED+4000+index, include_mpc=False,
                                         sweeps=1200, regime=regime)
                for row in compared:
                    tail = p.with_updates(T=20., n_steps=200, omega_L=0., omega_B=0.,
                        x0=tuple(row[key] for key in ['final_S', 'final_L', 'final_B']))
                    time, state = solve_state(tail, np.zeros((10, 3)))
                    row.update(tail_horizon=20.,
                        tail_malware_host_time=np.trapezoid(state[:, 1]+state[:, 2], time),
                        tail_final_infected=sum(state[-1, 1:]))
                rows.extend(compared)
        elif study == 'initial':
            for fraction in [.01, .03, .08] if all_settings else [.03]:
                for share in [.2, .5, .8] if all_settings else [.5]:
                    infected = 1000*fraction
                    p = base.with_updates(x0=(1000-infected, infected*share, infected*(1-share)))
                    seed = SEED+5000+100*index+int(1000*fraction+10*share)
                    rows.extend(standard_rows(p, b, seed, regime=regime,
                        initial_fraction=fraction, latent_share=share))
        elif study == 'capacity':
            critical = critical_certificate(base)['bcrit']
            for ratio in [.6, .8, 1., 1.2, 1.5] if all_settings else [1.]:
                b = min(base.bmax, ratio*critical)
                rows.extend(standard_rows(base, b, SEED+int(round(100*base.R10+10*base.R20)),
                    all_baselines=True, sweeps=1200, regime=regime, b_ratio=b/critical, bcrit=critical))
        elif study == 'calendar':
            for intervals in [10, 20, 40]:
                rows.append(pmp_row(base, b, intervals, regime=regime))
                control = mpc_schedule(base, b, intervals,
                    prediction_intervals={10: 2, 20: 4, 40: 8}[intervals])
                rows.append(scheduled(base, b, 'MPC', control, regime=regime,
                    prediction_horizon_time=12.))
        elif study == 'weights':
            conditions = [('base', base)]
            conditions += [(f'A_B_over_A_L_{ratio:g}', base.with_updates(
                A_L=1., A_B=ratio, omega_L=1., omega_B=ratio)) for ratio in [1., 2.]]
            for field in ['q', 'C', 'd']:
                for component in range(3):
                    for factor in [.8, 1.2]:
                        values = np.array(getattr(base, field), float)
                        values[component] *= factor
                        conditions.append((f'{field}{component+1}_x{factor:.1f}',
                                           base.with_updates(**{field: tuple(values)})))
            for ci, (condition, p) in enumerate(conditions if all_settings else conditions[:3]):
                rows.extend(standard_rows(p, capacity(p), SEED+8000+1000*index+ci,
                    include_mpc=False, regime=regime, condition=condition))
        elif study == 'capacity_load':
            conditions = [('base', None, 1.)]+[(f'q{i+1}_x{factor:.1f}', i, factor)
                for i in range(3) for factor in [.8, 1.2]]
            for name, component, factor in conditions:
                loads = np.array(base.q)
                if component is not None:
                    loads[component] *= factor
                p = base.with_updates(q=tuple(loads))
                rows.append(pmp_row(p, b, sweeps=1200, regime=regime, condition=name,
                    fixed_b=b, perturbed_bcrit=critical_certificate(p)['bcrit']))
    return rows


def perturb_observation(state, level, rng):
    total = float(np.sum(state))
    L, B = np.maximum(np.asarray(state)[1:]*(1.+rng.uniform(-level, level, 2)), 0.)
    if L+B > total:
        scale = total/(L+B)
        L, B = L*scale, B*scale
    return np.array([total-L-B, L, B])


def observed_feedback(p, controller, b, method, model, noise=0., delay=0, seed=SEED):
    """Update the policy from delayed or noisy observations."""
    time = np.linspace(0., p.T, p.n_steps+1)
    state = np.empty((p.n_steps+1, 3))
    state[0] = p.x0
    controls = np.zeros((K, 3))
    previous = None
    steps, interval = p.n_steps//K, p.T/K
    rng = np.random.default_rng(seed)
    predictor = projected_policy(model, controller, b)
    for k in range(K):
        first = k*steps
        observed = state[max(0, k-delay)*steps].copy()
        if noise:
            observed = perturb_observation(observed, noise, rng)
        if method == 'Myopic':
            controls[k] = myopic_action(observed, controller, b, interval)
        elif method == 'Projected surrogate':
            controls[k] = predictor(time[first], observed)
        elif method == 'MPC':
            holds = min(round(12/interval), K-k)
            local = controller.with_updates(T=holds*interval, n_steps=holds*steps,
                                            x0=tuple(observed))
            previous = mpc_plan(local, b, holds, previous)
            controls[k] = previous[0]
        else:
            raise ValueError(f'Unknown feedback policy: {method}')
        for n in range(first, first+steps):
            state[n+1] = rk4_step(state[n], controls[k], p, p.h)
    return time, state, controls


def run_robustness(all_settings, model):
    rows = []
    conditions = [('nominal', 0., 0., 0, 0)]
    for replicate in range(5 if all_settings else 1):
        conditions += [('state_noise_5pct', .05, 0., 0, replicate),
                       ('state_noise_10pct', .1, 0., 0, replicate)]
    conditions += [(f'parameter_bias_{label}', 0., bias, 0, 0) for label, bias in
                   [('minus20pct', -.2), ('minus10pct', -.1), ('plus10pct', .1), ('plus20pct', .2)]]
    conditions += [('delay_1_hold', 0., 0., 1, 0), ('delay_2_holds', 0., 0., 2, 0)]
    for regime in REGIMES if all_settings else ['balanced_supercritical']:
        index = REGIMES.index(regime)
        p = params_for_regime(regime, n_steps=200)
        b = capacity(p)
        for ci, (condition, noise, bias, delay, replicate) in enumerate(conditions):
            controller = p.with_updates(beta1=p.beta1*(1+bias), beta2=p.beta2*(1+bias),
                                        eta=p.eta*(1+bias), gamma=p.gamma*(1+bias))
            seed = SEED+9000+100*index+replicate if noise else SEED+9500+100*index+ci
            fields = dict(regime=regime, condition=condition, observation_error=noise,
                          parameter_bias=bias, delay_holds=delay, replicate=replicate)
            for method in ['Myopic', 'Projected surrogate', 'MPC']:
                rows.append(record(p, b, method, observed_feedback(
                    p, controller, b, method, model, noise, delay, seed), **fields))
            if delay == 0:
                design_state = perturb_observation(np.array(p.x0), noise,
                    np.random.default_rng(seed)) if noise else np.array(p.x0)
                result = pmp(controller.with_updates(x0=tuple(design_state)), b)
                rows.append(scheduled(p, b, 'PMP', result.u_blocks, **fields))
    frame = pd.DataFrame(rows)
    nominal = frame[frame.condition == 'nominal'].set_index(['regime', 'method']).J
    frame['degradation_vs_nominal_pct'] = [100*(r.J/nominal[r.regime, r.method]-1)
                                          for r in frame.itertuples()]
    return frame.to_dict('records')


def run_detection(all_settings):
    conditions = [('Nominal', 0, 0, 0, 1, 1), ('Detection delay 3', 1, 0, 0, 1, 1),
        ('Detection delay 6', 2, 0, 0, 1, 1), ('False alerts 0.5%', 0, .005, 0, 1, 1),
        ('False alerts 1%', 0, .01, 0, 1, 1), ('Missed infections 20%', 0, 0, .2, 1, 1),
        ('Missed infections 40%', 0, 0, .4, 1, 1),
        ('False alerts 1% + missed 40%', 0, .01, .4, 1, 1),
        ('Propagation +20%', 0, 0, 0, 1.2, 1), ('Contacts +20% at t=30', 0, 0, 0, 1, 1.2)]
    rows, cache = [], {}
    for regime in REGIMES if all_settings else ['balanced_supercritical']:
        p = params_for_regime(regime, n_steps=200)
        b, steps = capacity(p), p.n_steps//K
        for name, delay, false, missed, factor, jump in conditions:
            for method in ['PMP', 'Equal resource', 'Threshold isolation', 'Alert remediation', 'Myopic']:
                time = np.linspace(0, p.T, p.n_steps+1)
                state = np.empty((p.n_steps+1, 3)); state[0] = p.x0
                control = np.zeros((K, 3))
                for k in range(K):
                    first = k*steps
                    L, B = (1-missed)*state[first, 1:]+false*state[first, 0]/2
                    observed = np.array([sum(state[first])-L-B, L, B])
                    if k >= delay:
                        if method == 'PMP' and k == delay:
                            key = (regime, delay, tuple(np.round(observed, 10)))
                            if key not in cache:
                                design = p.with_updates(T=p.T-time[first], n_steps=p.n_steps-first,
                                                        x0=tuple(observed))
                                cache[key] = pmp(design, b, K-delay, 1600).u_blocks
                            control[delay:] = cache[key]
                        elif method == 'Myopic':
                            control[k] = myopic_action(observed, p, b, p.T/K)
                        elif method != 'PMP':
                            control[k] = practical_action(method, observed, p, b)
                    for n in range(first, first+steps):
                        scale = factor*(jump if time[n] >= 30 else 1)
                        actual = p.with_updates(beta1=p.beta1*scale, beta2=p.beta2*scale)
                        state[n+1] = rk4_step(state[n], control[k], actual, p.h)
                rows.append(record(p, b, method, (time, state, control), regime=regime, scenario=name))
    return rows


def run_surrogate(assets, all_settings):
    rows = []
    names = {'MLP_marginal': 'Marginal MLP', 'MLP_action': 'Direct-action MLP',
             'ExtraTrees_marginal': 'Extra-Trees marginals'}
    selected = [s for s in assets['scenarios'] if s['split'] != 'train']
    if not all_settings:
        selected = [s for s in selected if s['scenario_id'].endswith('_00')]
    for scenario in selected:
        p, b = scenario_params(scenario), scenario['b']
        expert = assets['experts'][scenario['scenario_id']]
        reference = scheduled(p, b, 'PMP', expert['u_blocks'])
        policies = {names[name]: projected_policy(model, p, b,
            'action' if name == 'MLP_action' else 'marginal')
            for name, model in assets['models'].items()}
        policies['Single-interval flow rule'] = lambda t, x: myopic_action(x, p, b, p.T/K)
        for name, policy in policies.items():
            row = record(p, b, name, simulate_feedback(p, K, policy),
                scenario_id=scenario['scenario_id'], split=scenario['split'])
            row['objective_gap_pct'] = 100*(row['J']-reference['J'])/reference['J']
            rows.append(row)
        reference.update(scenario_id=scenario['scenario_id'], split=scenario['split'], objective_gap_pct=0.)
        rows.append(reference)
    return rows


def run_training(assets, regenerate, output):
    dataset = assets['training']
    experts = dict(assets['experts'])
    if regenerate:
        features, marginals, actions = [], [], []
        for scenario in scenarios():
            if scenario['split'] != 'train':
                continue
            p, b = scenario_params(scenario), scenario['b']
            result = pmp(p, b, sweeps=1600)
            experts[scenario['scenario_id']] = dict(
                t=result.t, x=result.x, lam=result.lam, u_blocks=result.u_blocks,
                marginal_blocks=result.block_marginals, nu_blocks=result.nu_blocks,
                objective=result.objective)
            for k in range(K):
                n = k*(p.n_steps//K)
                features.append(feature_vector(result.t[n], result.x[n], p, b))
            marginals.extend(result.block_marginals)
            actions.extend(result.u_blocks)
        dataset = dict(X=np.array(features), marginal_targets=np.array(marginals),
                       action_targets=np.array(actions))
    models = train_models(dataset)
    joblib.dump(dict(assets, models=models, training=dataset, experts=experts),
                output/'learning_assets.joblib', compress=3)


def run_finite(assets, all_settings, seeds_count, output):
    rows, ssa_rows = [], []
    settings = {'balanced': params_for_reproduction_number(1.5, 1.5, n_steps=200),
        'high_transmission': params_for_reproduction_number(2.15, 2.05,
            n_steps=200, x0=(940., 36., 24.))}
    for setting, p in settings.items():
        b = capacity(p)
        result = pmp(p, b)
        constant = constant_schedule(p, b, K, starts=20, seed=SEED+10000)[0]
        policies = {
            'PMP': lambda t, x: result.u_blocks[min(K-1, int(t/(p.T/K)))],
            'Constant': lambda t, x: constant,
            'Equal setting': lambda t, x: equal_setting(p, b),
            'Threshold constant': lambda t, x: np.asarray(critical_certificate(p)['u']),
            'Myopic': lambda t, x: myopic_action(x, p, b, p.T/K),
            'Projected surrogate': projected_policy(assets['models']['MLP_marginal'], p, b)}
        for dt in [.05, .025, .0125] if all_settings else [.0125]:
            for seed in range(SEED+10000, SEED+10000+seeds_count):
                for method, policy in policies.items():
                    trajectory = simulate_finite_node_slbs(p, K, policy, seed, dt)
                    rows.append(record(p, b, method, trajectory, setting=setting, seed=seed, dt=dt))
        for seed in range(SEED+10000, SEED+10000+min(30, seeds_count)):
            for method in ['PMP', 'Myopic']:
                trajectory = simulate_ssa_sampled(p, K, policies[method], seed)
                ssa_rows.append(record(p, b, method, trajectory, setting=setting,
                                       seed=seed, output_dt=.0125))
    frame = pd.DataFrame(rows)
    frame.to_csv(output/'finite_population.csv', index=False)
    pd.DataFrame(ssa_rows).to_csv(output/'finite_population_ssa.csv', index=False)
    rng, summaries = np.random.default_rng(SEED+10100), []
    for (setting, dt, method), group in frame.groupby(['setting', 'dt', 'method']):
        reference = frame[(frame.setting == setting)&(frame.dt == dt)&(frame.method == 'PMP')]
        joined = group[['seed', 'J']].merge(reference[['seed', 'J']], on='seed', suffixes=('', '_pmp'))
        difference = (joined.J-joined.J_pmp).to_numpy()
        low, high = bootstrap_mean_ci(difference, rng)
        summaries.append(dict(setting=setting, dt=dt, method=method, n=len(group),
            J_mean=group.J.mean(), J_median=group.J.median(), J_SD=group.J.std(ddof=1),
            paired_difference_mean=difference.mean(), paired_mean_CI95_low=low,
            paired_mean_CI95_high=high))
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('study', choices=['terminal', 'initial', 'capacity', 'calendar',
        'capacity_curve', 'capacity_load', 'mpc_horizon', 'parameters', 'weights',
        'robustness', 'detection', 'finite', 'surrogate', 'train'])
    parser.add_argument('--all', action='store_true', help='Run all settings for this study.')
    parser.add_argument('--seeds', type=int, default=None, help='Finite-population seeds (100 with --all, otherwise 3).')
    parser.add_argument('--regenerate-experts', action='store_true', help='Recompute all 30 training PMP schedules.')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs')
    parser.add_argument('--assets', type=Path, help='Alternative trusted learning_assets.joblib.')
    args = parser.parse_args()
    output = args.output.resolve()/args.study
    if (ROOT/'results').resolve() in [output, *output.parents]:
        parser.error('Choose an output directory outside results/.')
    if args.seeds is not None and args.seeds < 1:
        parser.error('--seeds must be positive')
    output.mkdir(parents=True, exist_ok=True)
    if args.study in ['surrogate', 'train', 'finite', 'robustness']:
        assets = load_assets(args.assets) if args.assets else load_assets()
    if args.study == 'surrogate':
        rows = run_surrogate(assets, args.all)
    elif args.study == 'train':
        run_training(assets, args.regenerate_experts, output)
        return
    elif args.study == 'finite':
        rows = run_finite(assets, args.all, args.seeds or (100 if args.all else 3), output)
    elif args.study == 'robustness':
        rows = run_robustness(args.all, assets['models']['MLP_marginal'])
    elif args.study == 'detection':
        rows = run_detection(args.all)
    else:
        rows = run_sensitivity(args.study, args.all)
    destination = output/f'{args.study}.csv'
    pd.DataFrame(rows).to_csv(destination, index=False)


if __name__ == '__main__':
    main()
