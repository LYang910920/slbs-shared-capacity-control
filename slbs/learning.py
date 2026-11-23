"""Nine-feature regression models for projected control updates."""
from pathlib import Path

import joblib
import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .model import params_for_reproduction_number
from .projection import project_weighted_capped_simplex

SEED = 20260712
ASSETS = Path(__file__).resolve().parents[1] / 'learning_assets.joblib'


def load_assets(path=ASSETS):
    """Load the fitted models and training trajectories from a trusted joblib file."""
    return joblib.load(path)


def scenarios(seed=SEED):
    rng = np.random.default_rng(seed)
    result = []
    for split, count, r_range, infected_range, capacity_range in [
        ('train', 30, (1.05, 1.95), (15., 55.), (.48, 1.02)),
        ('validation', 10, (1.08, 1.92), (17., 52.), (.50, 1.00)),
        ('extrapolation', 8, (1.96, 2.25), (55., 80.), (.42, 1.08)),
    ]:
        for index in range(count):
            r1, r2 = rng.uniform(*r_range, size=2)
            infected = rng.uniform(*infected_range)
            share = rng.uniform(.15, .85)
            capacity = rng.uniform(*capacity_range)
            result.append(dict(scenario_id=f'{split}_{index:02d}', split=split,
                R10=float(r1), R20=float(r2), b=float(capacity),
                x0=(1000.-infected, infected*share, infected-infected*share)))
    return result


def scenario_params(scenario):
    return params_for_reproduction_number(scenario['R10'], scenario['R20'],
        x0=tuple(scenario['x0']), n_steps=200)


def feature_vector(t, state, p, capacity):
    state = np.asarray(state, float)
    return np.array([t/p.T, *list(state/p.Nstar), np.log1p(state[1]),
                     np.log1p(state[2]), p.R10, p.R20, capacity/p.bmax])


def projected_policy(model, p, capacity, output_type='marginal'):
    if output_type not in {'marginal', 'action'}:
        raise ValueError('output_type must be marginal or action')

    def policy(t, state):
        prediction = model.predict(feature_vector(t, state, p, capacity)[None, :])[0]
        proposal = ((prediction-np.asarray(p.d))/np.asarray(p.C)
                    if output_type == 'marginal' else prediction)
        return project_weighted_capped_simplex(
            proposal, p.C, p.q, p.umax, capacity)[0]
    return policy


def build_models(seed=SEED):
    def mlp():
        network = MLPRegressor(hidden_layer_sizes=(12, 12), activation='tanh',
            solver='lbfgs', alpha=.15, max_iter=4000, random_state=seed,
            tol=1e-9, max_fun=40000)
        return TransformedTargetRegressor(regressor=Pipeline([
            ('scale', StandardScaler()), ('mlp', network)]), transformer=StandardScaler())
    forest = ExtraTreesRegressor(n_estimators=200, max_depth=10,
        min_samples_leaf=2, max_features=1., random_state=seed, n_jobs=1)
    return {'MLP_marginal': mlp(), 'MLP_action': mlp(),
            'ExtraTrees_marginal': TransformedTargetRegressor(
                regressor=forest, transformer=StandardScaler())}


def train_models(dataset):
    models = build_models()
    for name, model in models.items():
        target = 'action_targets' if name == 'MLP_action' else 'marginal_targets'
        model.fit(dataset['X'], dataset[target])
    return models
