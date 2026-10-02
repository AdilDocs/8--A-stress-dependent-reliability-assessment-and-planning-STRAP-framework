"""
optimizers.py — population-based minimisers compared in the planning study.
All share the interface  run(f, lo, hi, NP, G, seed) -> (best_x, best_f, history)
where f maps a (NP x D) population to NP objective values, and history[g] is
the best objective after generation g.  Every algorithm gets the same
population size and number of function evaluations (NP x (G + 1)).

  de   : standard DE — rand/1/bin, F = 0.5, CR = 0.9 (Storn & Price, 1997)
  jde  : self-adaptive DE (Brest et al., IEEE TEVC 2006)
  pso  : constriction-factor PSO (Clerc & Kennedy 2002; w = 0.7298, c = 1.49618)
  ga   : real-coded GA — binary tournament, SBX (eta 15, pc 0.9),
         polynomial mutation (eta 20, pm 1/D), elitism (Deb & Agrawal 1995)
"""
import numpy as np

CR = 0.9


def _init(rng, lo, hi, NP):
    return lo + rng.random((NP, len(lo))) * (hi - lo)


def _mutants(rng, NP):
    idx = np.empty((NP, 3), dtype=int)
    for i in range(NP):
        idx[i] = rng.choice(np.delete(np.arange(NP), i), 3, replace=False)
    return idx


def _crossover(rng, pop, v, cr):
    NP, D = pop.shape
    mask = rng.random((NP, D)) < np.broadcast_to(np.atleast_1d(cr)[:, None] if np.ndim(cr) else cr, (NP, D))
    mask[np.arange(NP), rng.integers(0, D, NP)] = True
    return np.where(mask, v, pop)


def de(f, lo, hi, NP, G, seed, F=0.5):
    rng = np.random.default_rng(seed)
    pop = _init(rng, lo, hi, NP)
    fit = f(pop)
    hist = [fit.min()]
    for g in range(G):
        idx = _mutants(rng, NP)
        v = np.clip(pop[idx[:, 0]] + F * (pop[idx[:, 1]] - pop[idx[:, 2]]), lo, hi)
        trial = _crossover(rng, pop, v, CR)
        tf = f(trial)
        better = tf <= fit
        pop[better], fit[better] = trial[better], tf[better]
        hist.append(fit.min())
    b = np.argmin(fit)
    return pop[b], fit[b], np.array(hist)


def jde(f, lo, hi, NP, G, seed, t1=0.1, t2=0.1):
    rng = np.random.default_rng(seed)
    pop = _init(rng, lo, hi, NP)
    fit = f(pop)
    Fi, CRi = np.full(NP, 0.5), np.full(NP, 0.9)
    hist = [fit.min()]
    for g in range(G):
        Fn = np.where(rng.random(NP) < t1, 0.1 + 0.9 * rng.random(NP), Fi)
        CRn = np.where(rng.random(NP) < t2, rng.random(NP), CRi)
        idx = _mutants(rng, NP)
        v = np.clip(pop[idx[:, 0]] + Fn[:, None] * (pop[idx[:, 1]] - pop[idx[:, 2]]), lo, hi)
        trial = _crossover(rng, pop, v, CRn)
        tf = f(trial)
        better = tf <= fit
        pop[better], fit[better] = trial[better], tf[better]
        Fi[better], CRi[better] = Fn[better], CRn[better]
        hist.append(fit.min())
    b = np.argmin(fit)
    return pop[b], fit[b], np.array(hist)


def pso(f, lo, hi, NP, G, seed, w=0.7298, c1=1.49618, c2=1.49618):
    rng = np.random.default_rng(seed)
    x = _init(rng, lo, hi, NP)
    vmax = 0.2 * (hi - lo)
    v = (rng.random(x.shape) * 2 - 1) * vmax
    fx = f(x)
    pb, pf = x.copy(), fx.copy()
    gb = pb[np.argmin(pf)].copy()
    hist = [pf.min()]
    for g in range(G):
        r1, r2 = rng.random(x.shape), rng.random(x.shape)
        v = np.clip(w * v + c1 * r1 * (pb - x) + c2 * r2 * (gb - x), -vmax, vmax)
        x = np.clip(x + v, lo, hi)
        fx = f(x)
        imp = fx < pf
        pb[imp], pf[imp] = x[imp], fx[imp]
        gb = pb[np.argmin(pf)].copy()
        hist.append(pf.min())
    b = np.argmin(pf)
    return pb[b], pf[b], np.array(hist)


def ga(f, lo, hi, NP, G, seed, pc=0.9, eta_c=15.0, eta_m=20.0):
    rng = np.random.default_rng(seed)
    D = len(lo)
    pm = 1.0 / D
    pop = _init(rng, lo, hi, NP)
    fit = f(pop)
    hist = [fit.min()]
    span = hi - lo
    for g in range(G):
        a, b = rng.integers(0, NP, (2, NP))
        par = np.where((fit[a] <= fit[b])[:, None], pop[a], pop[b])
        child = par.copy()
        for k in range(0, NP - 1, 2):                       # SBX
            if rng.random() < pc:
                u = rng.random(D)
                beta = np.where(u <= 0.5, (2 * u) ** (1 / (eta_c + 1)), (1 / (2 * (1 - u))) ** (1 / (eta_c + 1)))
                p1, p2 = par[k], par[k + 1]
                child[k] = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2)
                child[k + 1] = 0.5 * ((1 - beta) * p1 + (1 + beta) * p2)
        m = rng.random((NP, D)) < pm                        # polynomial mutation
        u = rng.random((NP, D))
        dl = np.where(u < 0.5, (2 * u) ** (1 / (eta_m + 1)) - 1, 1 - (2 * (1 - u)) ** (1 / (eta_m + 1)))
        child = np.clip(np.where(m, child + dl * span, child), lo, hi)
        cf = f(child)
        e = np.argmin(fit)                                  # elitism
        worst = np.argmax(cf)
        child[worst], cf[worst] = pop[e], fit[e]
        pop, fit = child, cf
        hist.append(min(hist[-1], fit.min()))
    b = np.argmin(fit)
    return pop[b], fit[b], np.array(hist)


ALGORITHMS = {"jDE": jde, "Standard DE": de, "PSO": pso, "GA": ga}
