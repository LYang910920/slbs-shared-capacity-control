"""Fixed-population competing-risk tau leap and sampled-control SSA."""
import numpy as np


def _multinomial_competing(count, rates, dt, rng):
    rates = np.maximum(np.asarray(rates, float), 0.)
    total = float(np.sum(rates))
    if count <= 0 or total <= 0.:
        return np.zeros(len(rates), dtype=int)
    probability = 1.-np.exp(-total*dt)
    return rng.multinomial(count, np.append(probability*rates/total, 1.-probability))[:-1]


def _setup(p, intervals, seed, dt):
    steps = int(round(p.T/dt))
    if abs(steps*dt-p.T) > 1e-12 or steps % intervals:
        raise ValueError('The output step must divide the horizon and every hold.')
    counts = np.rint(p.x0).astype(int)
    if np.sum(counts) != int(round(p.Nstar)):
        raise ValueError('Fixed-population simulations require N(0)=mu/delta.')
    time = np.linspace(0., p.T, steps+1)
    state = np.empty((steps+1, 3), float)
    state[0] = counts
    return np.random.default_rng(seed), counts, time, state, np.zeros((intervals, 3))


def simulate_finite_node_slbs(p, K, policy, seed, dt=.05):
    rng, counts, time, state, control = _setup(p, K, seed, dt)
    steps_per_hold = (len(time)-1)//K
    for n in range(len(time)-1):
        if n % steps_per_hold == 0:
            control[n//steps_per_hold] = policy(time[n], counts.astype(float))
        u1, u2, u3 = control[n//steps_per_hold]
        S, L, B = map(int, counts)
        s_events = _multinomial_competing(S, [
            (1.-u1)*p.beta1*L/(1.+p.theta1*L),
            (1.-u2)*p.beta2*B/(1.+p.theta2*B)], dt, rng)
        l_events = _multinomial_competing(L, [p.eta, p.delta], dt, rng)
        b_to_s = rng.binomial(B, 1.-np.exp(-(p.gamma+p.kappa*u3+p.delta)*dt))
        counts = np.array([S-sum(s_events)+l_events[1]+b_to_s,
            L+s_events[0]-sum(l_events), B+s_events[1]+l_events[0]-b_to_s])
        if np.min(counts) < 0 or np.sum(counts) != round(p.Nstar):
            raise RuntimeError('Invalid population transition')
        state[n+1] = counts
    return time, state, control


def simulate_ssa_sampled(p, K, policy, seed, output_dt=.0125):
    rng, counts, time, state, control = _setup(p, K, seed, output_dt)
    steps_per_hold = (len(time)-1)//K
    current = 0.
    changes = np.array([[-1, 1, 0], [-1, 0, 1], [0, -1, 1],
                        [1, -1, 0], [1, 0, -1]])
    for n in range(len(time)-1):
        if n % steps_per_hold == 0:
            control[n//steps_per_hold] = policy(time[n], counts.astype(float))
        u1, u2, u3 = control[n//steps_per_hold]
        end = time[n+1]
        while current < end-1e-15:
            S, L, B = map(int, counts)
            rates = np.array([(1.-u1)*p.beta1*S*L/(1.+p.theta1*L),
                (1.-u2)*p.beta2*S*B/(1.+p.theta2*B), p.eta*L, p.delta*L,
                (p.gamma+p.kappa*u3+p.delta)*B])
            total = float(np.sum(rates))
            if total <= 0.:
                current = end
                break
            waiting = float(rng.exponential(1./total))
            if current+waiting >= end:
                current = end
                break
            current += waiting
            counts += changes[int(rng.choice(5, p=rates/total))]
            if np.min(counts) < 0 or np.sum(counts) != round(p.Nstar):
                raise RuntimeError('Invalid population transition')
        state[n+1] = counts
    return time, state, control


def bootstrap_mean_ci(values, rng, draws=3000):
    values = np.asarray(values, float)
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    return tuple(np.quantile(values[indices].mean(axis=1), [.025, .975]))
