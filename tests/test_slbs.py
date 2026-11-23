import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fitting import decode, encode, fit_exponential, load_inputs, parameter_map, predict, split_size
from run import outbreak_parameters
from slbs.model import (
    adjoint_rhs,
    controlled_thresholds,
    critical_certificate,
    params_for_reproduction_number,
    params_for_regime,
    running_cost,
    state_rhs,
    terminal_gradient,
)
from slbs.projection import (
    enumerate_projection,
    project_weighted_capped_simplex,
    projection_kkt_residual,
)
from slbs.solver import solve_adjoint, solve_sampled_pmp, solve_state
from slbs.strategies import ScheduleObjective, shooting_schedule, myopic_action, practical_action

ROOT = Path(__file__).resolve().parents[1]


def test_learning_split_and_archived_training_features():
    from slbs.learning import load_assets, scenarios, scenario_params, feature_vector
    assets = load_assets()
    assert scenarios() == assets['scenarios']
    assert assets['training']['X'].shape == (600, 9)
    rows = []
    for scenario in assets['scenarios']:
        if scenario['split'] != 'train':
            continue
        p = scenario_params(scenario)
        expert = assets['experts'][scenario['scenario_id']]
        for k in range(20):
            rows.append(feature_vector(expert['t'][k*10], expert['x'][k*10], p, scenario['b']))
    np.testing.assert_array_equal(rows, assets['training']['X'])


def test_archived_predictors_issue_feasible_updates():
    from slbs.learning import load_assets, scenario_params, projected_policy
    assets = load_assets()
    for scenario in [assets['scenarios'][30], assets['scenarios'][40]]:
        p = scenario_params(scenario)
        for name, model in assets['models'].items():
            policy = projected_policy(model, p, scenario['b'],
                'action' if name == 'MLP_action' else 'marginal')
            action = policy(0., p.x0)
            assert np.min(action) >= 0 and np.max(action-np.array(p.umax)) <= 1e-12
            assert action@np.array(p.q) <= scenario['b']+1e-12


@pytest.mark.parametrize('simulator_name', ['simulate_finite_node_slbs', 'simulate_ssa_sampled'])
def test_finite_population_preservation_and_seed_reproducibility(simulator_name):
    from slbs import stochastic
    p = params_for_regime('balanced_supercritical', T=.3, n_steps=12)
    simulator = getattr(stochastic, simulator_name)
    policy = lambda t, x: np.array([.1, .2, .05])
    first = simulator(p, 2, policy, 20270712)
    second = simulator(p, 2, policy, 20270712)
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)
    assert np.min(first[1]) >= 0
    np.testing.assert_array_equal(first[1].sum(axis=1), np.full(len(first[0]), 1000))


def test_observation_noise_preserves_nonnegative_population():
    from studies import perturb_observation
    rng = np.random.default_rng(20260712)
    for state in [np.array([970., 20., 10.]), np.array([0., 500., 500.])]:
        observed = perturb_observation(state, .1, rng)
        assert np.min(observed) >= -1e-12
        assert abs(observed.sum()-state.sum()) < 1e-12


@pytest.fixture(scope='module')
def solution():
    p = params_for_regime('balanced_supercritical', n_steps=100)
    capacity = critical_certificate(p)['bcrit']
    return p, capacity, solve_sampled_pmp(p, capacity, K=5, max_sweeps=800)


def test_sampled_data_stationarity_and_feasibility(solution):
    p, capacity, result = solution
    assert result.converged
    assert result.fixed_point_residual < 1.1e-6
    assert result.kkt_residual < 8e-5
    assert np.max(result.u_blocks @ np.asarray(p.q)-capacity) <= 1e-12


def test_direct_shooting_agrees_with_pmp(solution):
    p, capacity, result = solution
    # Express the same objective and absolute stopping tolerance in thousands of cost units.
    scaled = p.with_updates(A_L=p.A_L/1000, A_B=p.A_B/1000,
        omega_L=p.omega_L/1000, omega_B=p.omega_B/1000,
        d=tuple(x/1000 for x in p.d), C=tuple(x/1000 for x in p.C))
    _, success, cost = shooting_schedule(scaled, capacity, 5, np.full((5, 3), .1),
                                         maxiter=100, ftol=2e-8/1000)
    assert success
    assert abs(result.objective/1000-cost)/cost < 2e-6


def test_continuous_adjoint_gradient_converges_under_mesh_refinement():
    control = np.full(18, .1)
    errors = []
    for steps in [120, 240]:
        p = params_for_regime('balanced_supercritical', T=6., n_steps=steps)
        objective = ScheduleObjective(p, .8, 6)
        _, gradient = objective.evaluate(control)
        differences = np.empty(18)
        for index in range(18):
            step = np.zeros(18)
            step[index] = 1e-5
            differences[index] = (objective.fun(control+step)-objective.fun(control-step))/(2e-5)
        errors.append(np.max(np.abs(differences-gradient)/np.maximum(1., np.abs(differences))))
    # Interpolated states and trapezoidal quadrature give a continuous-adjoint
    # approximation, whose error decreases with the state integration mesh.
    assert errors[1] < 1.5e-5
    assert errors[1]/errors[0] < .3


def test_terminal_penalty_sets_adjoint_endpoint():
    p = params_for_regime('balanced_supercritical', n_steps=100, omega_L=2., omega_B=7.)
    control = np.zeros((5, 3))
    _, states = solve_state(p, control)
    adjoint = solve_adjoint(p, states, control)
    np.testing.assert_array_equal(adjoint[-1], [0., 2., 7.])


def test_practical_and_myopic_actions_respect_shared_capacity():
    p = params_for_regime('balanced_supercritical')
    for capacity in [0., .5, 1., p.bmax]:
        for state in [np.array([970., 20., 10.]), np.array([600., 100., 300.])]:
            actions = [practical_action(name, state, p, capacity) for name in
                       ['Equal resource', 'Threshold isolation', 'Alert remediation']]
            actions.append(myopic_action(state, p, capacity, 3.))
            for action in actions:
                assert np.min(action) >= 0.
                assert np.max(action-np.asarray(p.umax)) <= 1e-12
                assert np.dot(p.q, action) <= capacity+1e-12


def test_total_population_balance_cancels_every_control_term():
    p = params_for_regime("balanced_supercritical")
    rng = np.random.default_rng(23)
    for _ in range(100):
        x = rng.uniform([100.0, 0.0, 0.0], [1200.0, 300.0, 300.0])
        u = rng.uniform(0.0, p.umax)
        observed = float(np.sum(state_rhs(x, u, p)))
        expected = p.mu - p.delta * float(np.sum(x))
        assert abs(observed - expected) < 2e-13


def test_classical_adjoint_matches_negative_hamiltonian_gradient():
    p = params_for_regime("latent_dominant")
    x = np.array([950.0, 30.0, 20.0])
    u = np.array([0.31, 0.18, 0.09])
    lam = np.array([2.0, 5.0, 8.0])
    eps = 1e-5

    def H(state):
        return running_cost(state, u, p) + lam @ state_rhs(state, u, p)

    numerical = np.array(
        [
            (H(x + eps * np.eye(3)[i]) - H(x - eps * np.eye(3)[i])) / (2.0 * eps)
            for i in range(3)
        ]
    )
    assert np.allclose(adjoint_rhs(lam, x, u, p), -numerical, rtol=3e-7, atol=3e-7)


def test_terminal_gradient_matches_terminal_cost():
    p = params_for_regime("balanced_supercritical", omega_L=1.7, omega_B=6.2)
    x = np.array([920.0, 55.0, 25.0])
    eps = 1e-6

    def phi(state):
        return p.omega_L * state[1] + p.omega_B * state[2]

    numerical = np.array(
        [
            (phi(x + eps * np.eye(3)[i]) - phi(x - eps * np.eye(3)[i]))
            / (2.0 * eps)
            for i in range(3)
        ]
    )
    assert np.allclose(terminal_gradient(p), numerical, rtol=1e-8, atol=1e-8)


def test_critical_capacity_certificate_matches_dense_endpoint_search():
    for regime in ("latent_dominant", "bursting_dominant", "balanced_supercritical"):
        p = params_for_regime(regime)
        certificate = critical_certificate(p)
        assert certificate["feasible"]
        _r1, _r2, rc = controlled_thresholds(certificate["u"], p)
        assert rc <= 1.0 + 2e-12
        u1 = max(0.0, 1.0 - 1.0 / p.R10)
        r = np.linspace(0.0, p.umax[2], 20001)
        G_B = p.beta2 * p.mu / p.delta
        u2 = np.maximum(0.0, 1.0 - (p.gamma + p.delta + p.kappa * r) / G_B)
        feasible = u2 <= p.umax[1]
        capacities = p.q[0] * u1 + p.q[1] * u2[feasible] + p.q[2] * r[feasible]
        assert abs(float(np.min(capacities)) - certificate["bcrit"]) < 8e-5


def test_margin_capacity_certificate_hits_requested_pathway_bound():
    p = params_for_regime("balanced_supercritical")
    certificate = critical_certificate(p, zeta=0.1)
    assert certificate["feasible"]
    r1, r2, rc = controlled_thresholds(certificate["u"], p)
    assert rc <= 0.9 + 2e-12
    assert abs(max(r1, r2) - 0.9) < 2e-12
    assert certificate["bcrit"] >= critical_certificate(p)["bcrit"]


def test_critical_capacity_boundary_cases():
    feasible_no_repair = params_for_reproduction_number(1.4, 1.6, kappa=0.0)
    certificate = critical_certificate(feasible_no_repair)
    assert certificate["feasible"]
    assert certificate["u"][2] == 0.0

    infeasible_bursting = params_for_reproduction_number(
        1.2, 3.0, kappa=0.0, umax=(0.9, 0.1, 0.9)
    )
    bursting_failure = critical_certificate(infeasible_bursting)
    assert bursting_failure["reason"] == "bursting_box_bounds"
    assert np.isinf(bursting_failure["bcrit"])

    infeasible_latent = params_for_reproduction_number(
        20.0, 1.1, umax=(0.1, 0.9, 0.9)
    )
    latent_failure = critical_certificate(infeasible_latent)
    assert latent_failure["reason"] == "latent_box_bound"
    assert np.isinf(latent_failure["bcrit"])

    already_controlled = params_for_reproduction_number(0.8, 0.9)
    zero = critical_certificate(already_controlled)
    assert zero["feasible"]
    assert zero["bcrit"] == 0.0
    assert np.array_equal(zero["u"], np.zeros(3))

    clipped_breakpoint = params_for_reproduction_number(1.5, 3.0)
    clipped = critical_certificate(clipped_breakpoint)
    assert clipped["feasible"]
    assert clipped["u"][2] <= clipped_breakpoint.umax[2]


def test_zero_slope_critical_capacity_has_nonunique_allocation_value():
    base = params_for_reproduction_number(1.5, 1.5)
    G_B = base.beta2 * base.mu / base.delta
    q3 = base.q[1] * base.kappa / G_B
    p = base.with_updates(q=(base.q[0], base.q[1], q3))
    certificate = critical_certificate(p)
    r0 = (G_B - p.gamma - p.delta) / p.kappa

    def capacity(r):
        u1 = max(0.0, 1.0 - 1.0 / p.R10)
        u2 = max(0.0, 1.0 - (p.gamma + p.delta + p.kappa * r) / G_B)
        return p.q[0] * u1 + p.q[1] * u2 + p.q[2] * r

    assert certificate["feasible"]
    assert abs(capacity(0.0) - capacity(r0)) < 2e-12
    assert abs(certificate["bcrit"] - capacity(0.0)) < 2e-12


def test_direct_entry_ngm_eigenvalues_match_pathway_maximum():
    p = params_for_regime("bursting_dominant")
    u = np.array([0.23, 0.31, 0.18])
    Sf = p.mu / p.delta
    F = np.diag([(1.0 - u[0]) * p.beta1 * Sf, (1.0 - u[1]) * p.beta2 * Sf])
    V = np.array(
        [
            [p.eta + p.delta, 0.0],
            [-p.eta, p.gamma + p.kappa * u[2] + p.delta],
        ]
    )
    eigenvalues = np.linalg.eigvals(F @ np.linalg.inv(V))
    r1, r2, rc = controlled_thresholds(u, p)
    assert np.allclose(np.sort(eigenvalues.real), np.sort([r1, r2]))
    assert abs(max(eigenvalues.real) - rc) < 2e-12


def test_requested_transmission_factors_respect_rate_overrides():
    p = params_for_reproduction_number(1.3, 1.7, eta=.2, gamma=.4, delta=.03, mu=60)
    np.testing.assert_allclose([p.R10, p.R20], [1.3, 1.7])


def test_zero_transmission_requires_no_capacity():
    p = params_for_regime('balanced_supercritical').with_updates(beta1=0., beta2=0.)
    assert critical_certificate(p)['bcrit'] == 0.


def test_projection_matches_independent_active_set_enumeration():
    p = params_for_regime("balanced_supercritical")
    rng = np.random.default_rng(20260712)
    C = np.asarray(p.C)
    q = np.asarray(p.q)
    umax = np.asarray(p.umax)
    for _ in range(100):
        z = rng.normal(0.35, 0.7, 3)
        b = float(rng.uniform(0.0, p.bmax))
        u, nu = project_weighted_capped_simplex(z, C, q, umax, b)
        reference = enumerate_projection(z, C, q, umax, b)
        assert np.max(np.abs(u - reference)) < 2e-10
        assert np.min(u) >= -1e-13
        assert np.max(u - umax) <= 1e-13
        assert q @ u <= b + 2e-12
        assert projection_kkt_residual(u, z, C, q, umax, b, nu) < 2e-8


def test_projection_with_unavailable_defense_and_zero_capacity():
    C = q = np.ones(3)
    caps = np.array([0., 1., 1.])
    z = np.array([2., 0., 0.])
    u, multiplier = project_weighted_capped_simplex(z, C, q, caps, 1.)
    np.testing.assert_array_equal(u, np.zeros(3))
    assert projection_kkt_residual(u, z, C, q, caps, 1., multiplier) == 0.
    broken = np.array([0.1, 0., 0.])
    assert projection_kkt_residual(broken, z, C, q, caps, 1., multiplier) >= 0.1

    z = np.array([2., 1e-14, 0.])
    u, multiplier = project_weighted_capped_simplex(z, C, q, caps, 0.)
    np.testing.assert_array_equal(u, np.zeros(3))
    assert multiplier == 1e-14
    assert projection_kkt_residual(u, z, C, q, caps, 0., multiplier) == 0.

    u, multiplier = project_weighted_capped_simplex(z, C, q, caps, 1e-15)
    assert q @ u <= 1e-15

    caps[0] = 1e-8
    u, multiplier = project_weighted_capped_simplex(z, C, q, caps, 1.)
    assert u[0] == caps[0]
    assert projection_kkt_residual(u, z, C, q, caps, 1., multiplier) == 0.


def test_projection_nonexpansiveness_and_fixed_quadratic_loss_bound():
    p = params_for_regime("balanced_supercritical")
    rng = np.random.default_rng(20260713)
    C = np.asarray(p.C)
    q = np.asarray(p.q)
    umax = np.asarray(p.umax)
    Delta = 3.0
    for _ in range(250):
        a = rng.normal(25.0, 90.0, 3)
        ahat = a + rng.normal(0.0, 20.0, 3)
        b = float(rng.uniform(0.0, p.bmax))
        v, _ = project_weighted_capped_simplex(a / C, C, q, umax, b)
        vhat, _ = project_weighted_capped_simplex(ahat / C, C, q, umax, b)
        command_error_sq = float(np.dot(C, (vhat - v) ** 2))
        marginal_error_sq = float(np.dot(1.0 / C, (ahat - a) ** 2))
        assert command_error_sq <= marginal_error_sq + 2e-10

        def psi(w):
            return Delta * (0.5 * np.dot(C, w * w) - np.dot(a, w))

        loss = psi(vhat) - psi(v)
        assert loss >= -2e-9
        assert loss <= 0.5 * Delta * marginal_error_sq + 2e-8


def test_bounded_parameter_coordinates():
    settings = load_inputs()
    for case in settings['cases']:
        values, free = parameter_map(settings, case)
        z = encode(values, free, settings)
        recovered = decode(z, values, free, settings)
        assert np.all((z >= 0) & (z <= 1))
        np.testing.assert_allclose([recovered[name] for name in free], [values[name] for name in free])


def test_calibration_windows():
    t = np.arange(391) / 60
    sizes = {60: 235, 70: 274, 80: 313, 100: 391}
    for case in load_inputs()['cases']:
        assert split_size(t, case) == sizes[case['window']]


def test_bursting_only_logistic_limit():
    settings = load_inputs()
    N, B0, rate = settings['population'], 1648., .7
    values = dict(a_L=0., a_B=rate, eta=0., gamma=0., delta=0.,
                  s_L=0., s_B=0., L0=0., B0=B0)
    t = np.linspace(0, 4, 41)
    expected = N / (1 + (N / B0 - 1) * np.exp(-rate * t))
    np.testing.assert_allclose(predict(values, t, settings), expected, rtol=2e-8, atol=1e-5)


def test_exponential_fit_uses_calibration_values_only():
    settings = load_inputs()
    case = next(c for c in settings['cases'] if c['window'] == 80)
    t = np.linspace(0, 6.5, 41)
    y = 20 * np.exp(.4 * t)
    changed = y.copy()
    changed[split_size(t, case):] *= 100
    a, b = (fit_exponential(t, data, settings, case) for data in [y, changed])
    assert a['amplitude'] == b['amplitude']
    assert a['growth_per_hour'] == b['growth_per_hour']
    assert a['continuation_rmse'] < b['continuation_rmse']


def test_outbreak_host_count_scaling_preserves_equations_and_cost():
    settings = json.loads((ROOT/'settings.json').read_text())['outbreak']
    scenario = settings['scenarios'][1]
    p = outbreak_parameters(settings, scenario)
    scale = settings['fits'][scenario['fit']]['population']/1000
    original = p.with_updates(mu=p.mu*scale, beta1=p.beta1/scale, beta2=p.beta2/scale,
        theta1=p.theta1/scale, theta2=p.theta2/scale, x0=tuple(np.asarray(p.x0)*scale),
        d=tuple(np.asarray(p.d)*scale), C=tuple(np.asarray(p.C)*scale))
    action = np.array([.3, .2, .5])
    np.testing.assert_allclose(state_rhs(np.asarray(p.x0), action, p),
        state_rhs(np.asarray(original.x0), action, original)/scale, rtol=1e-12)
    np.testing.assert_allclose(running_cost(p.x0, action, p),
        running_cost(original.x0, action, original)/scale, rtol=1e-12)


def test_calendars_and_initial_states_are_feasible():
    for name in ['main', 'outbreak']:
        settings = json.loads((ROOT/'settings.json').read_text())[name]
        assert settings['rk4_steps'] % settings['control_intervals'] == 0
    outbreak = json.loads((ROOT/'settings.json').read_text())['outbreak']
    for scenario in outbreak['scenarios']:
        p = outbreak_parameters(outbreak, scenario)
        assert min(p.x0) >= 0.
        np.testing.assert_allclose(sum(p.x0), 1000, rtol=1e-12)


def test_final_tables_cover_the_main_comparisons():
    main = pd.read_csv(ROOT/'results/main_grid.csv')
    outbreak = pd.read_csv(ROOT/'results/outbreak_control.csv')
    assert len(main) == 72
    assert len(main.groupby(['latent_factor', 'bursting_factor'])) == 9
    assert len(outbreak) == 67
    assert outbreak.scenario.nunique() == 11
    assert main.cost.gt(0).all() and outbreak.cost.gt(0).all()
