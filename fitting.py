"""Fit the SLBS propagation model to a local scanner-presence series."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

from slbs.model import ModelParams, incidences, state_rhs


INPUTS = Path(__file__).with_name('settings.json')


def load_inputs():
    """Read parameter bounds, calibration windows, and initial guesses."""
    return json.loads(INPUTS.read_text())['fitting']


def observations(path):
    """Read a local observation CSV or CAIDA start/end summary directory."""
    path = Path(path)
    if path.is_dir():
        origin = datetime(2001, 7, 19, 10, tzinfo=timezone.utc).timestamp()
        stamp = np.arange(origin, origin + 6.5 * 3600 + 1, 60.)
        counts = []
        for suffix in ['start', 'end']:
            cdf = np.loadtxt(path / f'codered-july.{suffix}.cdf.txt.gz')
            index = np.searchsorted(cdf[:, 0], stamp, side='right') - 1
            counts.append(np.where(index >= 0, cdf[np.maximum(index, 0), 1], 0.))
        return (stamp - origin) / 3600., counts[0] - counts[1]
    frame = pd.read_csv(path)
    t = frame['time'].to_numpy(float)
    y = frame['observed'].to_numpy(float)
    if len(t) < 10 or not np.isfinite(np.r_[t, y]).all():
        raise ValueError('Expected at least ten finite time,observed rows.')
    if np.any(np.diff(t) <= 0) or np.min(y) < 0:
        raise ValueError('Times must increase and observations must be nonnegative.')
    return t - t[0], y


def parameter_map(settings, case):
    """Choose the free parameters of the full or restricted SLBS model."""
    values = dict(zip(settings['parameters'], settings['initial']))
    values.update(delta=settings['delta'], B0=settings['B0'])
    fixed = {'unsaturated': {'s_L': 0., 's_B': 0.},
             'bursting': {'a_L': 0., 'eta': 0., 's_L': 0., 'L0': 0.}}
    values.update(fixed.get(case['model'], {}))
    free = [name for name in settings['parameters'] if name not in fixed.get(case['model'], {})]
    return values, free


def encode(values, free, settings):
    """Map bounded parameters to unit-interval optimization coordinates."""
    bounds = dict(zip(settings['parameters'], settings['bounds']))
    z = []
    for name in free:
        lo, hi = bounds[name]
        value = values[name]
        z.append(np.log(value / lo) / np.log(hi / lo) if name in settings['log_parameters']
                 else (value - lo) / (hi - lo))
    return np.asarray(z, float)


def decode(z, values, free, settings):
    """Convert optimization coordinates to transmission and activation rates."""
    bounds = dict(zip(settings['parameters'], settings['bounds']))
    result = dict(values)
    for value, name in zip(z, free):
        lo, hi = bounds[name]
        result[name] = float(np.exp(np.log(lo) + value * np.log(hi / lo))) \
            if name in settings['log_parameters'] else float(lo + value * (hi - lo))
    return result


def predict(values, t, settings):
    """Integrate the uncontrolled SLBS equations and return the observed B proxy."""
    N = settings['population']
    p = ModelParams(T=float(t[-1]), mu=values['delta'] * N, delta=values['delta'],
                    eta=values['eta'], gamma=values['gamma'],
                    beta1=values['a_L'] / N, beta2=values['a_B'] / N,
                    theta1=values['s_L'] / N, theta2=values['s_B'] / N,
                    x0=(N - values['L0'] - values['B0'], values['L0'], values['B0']))

    def rhs(_t, state):
        return np.r_[state_rhs(state[:3], np.zeros(3), p), sum(incidences(state[:3], p))]

    result = solve_ivp(rhs, (0., float(t[-1])), np.r_[p.x0, 0.], t_eval=t,
                       method='DOP853', rtol=1e-8, atol=1e-9)
    if not result.success or np.min(result.y[:3]) < -1e-6:
        raise RuntimeError('SLBS state integration failed.')
    return settings['coverage'] * result.y[2]


def split_size(t, case):
    """Return the number of observations in the calibration window."""
    return len(t) if case['window'] == 100 else int(np.searchsorted(t, case['cutoff'] + 1e-12, side='right'))


def errors(predicted, observed, n):
    """Compute calibration and continuation RMSE on their respective windows."""
    residual = predicted - observed
    return {'calibration_rmse': float(np.sqrt(np.mean(residual[:n] ** 2))),
            'continuation_rmse': float(np.sqrt(np.mean(residual[n:] ** 2))) if n < len(observed) else None}


def fit_model(t, y, settings, case):
    """Select a bounded multistart least-squares fit by calibration error."""
    values, free = parameter_map(settings, case)
    n = split_size(t, case)
    scale = max(float(np.sqrt(np.mean(y[:n] ** 2))), 1.)

    def residual(z):
        try:
            fitted = predict(decode(z, values, free, settings), t[:n], settings)
            return (fitted - y[:n]) / scale
        except (RuntimeError, ValueError, FloatingPointError):
            return np.full(n, 1e12)

    rng = np.random.default_rng(settings['seed'])
    starts = [encode(values, free, settings)]
    starts.extend(rng.uniform(.01, .99, size=(case['starts'] - 1, len(free))))
    for guess in case['initial_guesses']:
        candidate = dict(zip(settings['parameters'], guess))
        starts.append(np.clip(encode(candidate, free, settings), 0., 1.))
    best, best_score = None, np.inf
    for start in starts:
        result = least_squares(residual, start, bounds=(np.zeros(len(free)), np.ones(len(free))),
                               method='trf', xtol=1e-9, ftol=1e-9, gtol=1e-9,
                               max_nfev=case['max_evaluations'])
        score = float(result.fun @ result.fun)
        if score < best_score:
            best, best_score = result, score
    if best is None or best_score >= 1e20:
        raise RuntimeError('No finite SLBS fit was obtained.')
    fitted = decode(best.x, values, free, settings)
    prediction = predict(fitted, t, settings)
    return {'window': case['window'], 'model': case['model'], **errors(prediction, y, n),
            **fitted, 'converged': bool(best.success)}


def fit_exponential(t, y, settings, case):
    """Fit the two-parameter exponential comparator on the same observations."""
    n = split_size(t, case)
    scale = max(float(np.sqrt(np.mean(y[:n] ** 2))), 1.)

    def prediction(z):
        return np.exp(z[0] + z[1] * t)

    result = least_squares(lambda z: (prediction(z)[:n] - y[:n]) / scale,
                           [np.log(max(y[0], 1.)), .3],
                           bounds=([np.log(1e-6), -5.], [np.log(settings['population']), 5.]),
                           max_nfev=600)
    return {'window': case['window'], 'model': 'exponential',
            **errors(prediction(result.x), y, n),
            'amplitude': float(np.exp(result.x[0])), 'growth_per_hour': float(result.x[1]),
            'converged': bool(result.success)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True, help='Local time,observed CSV or CAIDA summary directory.')
    parser.add_argument('--window', type=int, choices=[60, 70, 80, 100], default=80)
    parser.add_argument('--model', choices=['slbs', 'unsaturated', 'bursting', 'exponential'], default='slbs')
    parser.add_argument('--all', action='store_true', help='Fit all four models at all four windows.')
    parser.add_argument('--output', type=Path, default=Path('outputs/fitting.csv'))
    args = parser.parse_args()
    settings = load_inputs()
    t, y = observations(args.data)
    cases = settings['cases'] if args.all else [case for case in settings['cases']
            if case['window'] == args.window and case['model'] == ('slbs' if args.model == 'exponential' else args.model)]
    rows = []
    for case in cases:
        row = fit_exponential(t, y, settings, case) if args.model == 'exponential' and not args.all \
            else fit_model(t, y, settings, case)
        rows.append(row)
        if args.all and case['model'] == 'slbs':
            rows.append(fit_exponential(t, y, settings, case))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)


if __name__ == '__main__':
    main()
