"""
methods.py — competing adequacy-assessment methods for the benchmark.

Every method is FITTED to the same observed outage history (failure hours and
repair durations of the generating units) and then predicts adequacy indices:

  COPT          analytical capacity outage probability table, constant FOR
                (equivalent to the two-state Markov / sequential MCS model)
  Seasonal      monthly failure rates (pooled monthly multipliers), analytical
                COPT month by month
  Two-level     peak / off-peak failure rates (peak = top 20 % load hours),
                chronological simulation
  Proposed      stress-dependent failure hazard (kappa by maximum likelihood)
                and stress-dependent repair (gamma by maximum likelihood),
                chronological simulation
"""
import numpy as np

import config as C
import data_sources as D
import model as M

KAPPA_GRID = np.round(np.arange(0.0, 3.01, 0.1), 2)
GAMMA_GRID = np.round(np.arange(0.0, 1.001, 0.05), 3)
PEAK_Q = 0.8
B_GRID = np.round(np.arange(0.0, 8.01, 0.1), 2)


def month_index(L):
    day = np.arange(L) // 24
    return np.minimum(11, day * 12 // (L // 24))


def unit_shape(kappa, profile):
    """Normalised (mean 1) hazard shape of a unit with class-nominal stress."""
    mid = {q: np.mean(v) for q, v in C.STRESS_RANGES["unit"].items()}
    e_nom = mid["eta_L"] * mid["T"] / C.T_REF * mid["eta_A"] * mid["eta_V"]
    lam = profile ** kappa
    a, b = M.calibrate_beta(1.0, e_nom, lam)
    g = 1.0 / M.ttf_law(e_nom * lam, a, b)
    return g / g.mean()


_SHAPES = {}


def shapes(profile):
    key = len(profile)
    if key not in _SHAPES:
        _SHAPES[key] = np.array([unit_shape(k, profile) for k in KAPPA_GRID])
    return _SHAPES[key]


# ---------------------------------------------------------------------------
# COPT with hour-specific forced outage rates (by month)
# ---------------------------------------------------------------------------
def _copt(caps, fors):
    tot = int(round(sum(caps)))
    P = np.zeros(tot + 1); P[0] = 1.0
    for c, q in zip(caps, fors):
        c = int(round(c))
        new = P * (1 - q)
        new[c:] += P[:tot + 1 - c] * q
        P = new
    return P, tot


def copt_indices(caps, fors_by_group, groups, load):
    """LOLE (h) and EENS (MWh) with FOR depending on the hour's group."""
    lole = eens = 0.0
    for gk in np.unique(groups):
        P, tot = _copt(caps, fors_by_group[gk])
        avail = tot - np.arange(tot + 1)
        order = np.argsort(avail)
        av, Ps = avail[order], P[order]
        cP, cPA = np.cumsum(Ps), np.cumsum(Ps * av)
        Lg = load[groups == gk]
        k = np.searchsorted(av, Lg, side="left")
        pl = np.where(k > 0, cP[np.maximum(k - 1, 0)], 0.0)
        pa = np.where(k > 0, cPA[np.maximum(k - 1, 0)], 0.0)
        lole += pl.sum()
        eens += (Lg * pl - pa).sum()
    return float(lole), float(eens)


# ---------------------------------------------------------------------------
# fitting
# ---------------------------------------------------------------------------
def fit_all(S_hist, events, profile, n_hist):
    """S_hist: (nu x L*n_hist) up states; events: (unit, hour, duration)."""
    nu = S_hist.shape[0]
    L = len(profile)
    U = S_hist.reshape(nu, n_hist, L).sum(1).astype(float)       # up counts by hour-of-year
    eu, et, ed = events
    th = et % L
    n = np.bincount(eu, minlength=nu).astype(float)
    mttr = np.array([ed[eu == i].mean() if n[i] > 0 else np.nan for i in range(nu)])
    mttr = np.where(np.isnan(mttr), np.nanmean(mttr), mttr)
    up = U.sum(1)
    out = {}

    # conventional
    lam = np.maximum(n, 0.5) / up                                  # 0.5 failure floor
    out["COPT"] = dict(lam=lam, mttr=mttr)

    # seasonal (monthly multipliers pooled over units)
    mi = month_index(L)
    Mind = np.eye(12)[mi]                                          # L x 12
    up_m = U @ Mind                                                # nu x 12
    n_m = np.bincount(mi[th], minlength=12).astype(float)
    r_m = (n_m / up_m.sum(0)) / (n.sum() / up.sum())
    lam_s = np.maximum(n, 0.5) / (up_m @ r_m)
    out["Seasonal"] = dict(lam=lam_s, r=r_m, mttr=mttr)

    # two-level (peak / off-peak)
    pk = profile >= np.quantile(profile, PEAK_Q)
    up_p, up_o = U[:, pk].sum(1), U[:, ~pk].sum(1)
    n_p = float(pk[th].sum()); n_o = float(len(th) - n_p)
    rho = (n_p / up_p.sum()) / max(n_o / up_o.sum(), 1e-12)
    lam_o = np.maximum(n, 0.5) / (up_o + rho * up_p)
    out["Two-level"] = dict(lam=lam_o, rho=rho, peak=pk, mttr=mttr)

    # proposed: profile likelihood over kappa, then gamma
    G = shapes(profile)                                            # K x L
    E = U @ G.T                                                    # nu x K exposures
    logG = np.log(G)
    ll = np.zeros(len(KAPPA_GRID))
    for k in range(len(KAPPA_GRID)):
        lf = np.bincount(eu, weights=logG[k, th], minlength=nu)
        ll[k] = (lf - n * np.log(E[:, k])).sum()
    kb = int(np.argmax(ll))
    kappa = float(KAPPA_GRID[kb])
    g = G[kb]
    lam_p = np.maximum(n, 0.5) / E[:, kb]
    sig = profile ** kappa
    ef = (g * sig).sum() / g.sum()
    s_j = sig[th] / ef
    best = (-np.inf, None, None)
    for gm in GAMMA_GRID:
        w = gm + (1 - gm) * s_j
        m = np.array([(ed[eu == i] / w[eu == i]).mean() if n[i] > 0 else np.nan for i in range(nu)])
        m = np.where(np.isnan(m), np.nanmean(m), m)
        mu = m[eu] * w
        lg = float((-np.log(mu) - ed / mu).sum())
        if lg > best[0]:
            best = (lg, gm, m)
    _, gamma, m_p = best
    out["Proposed"] = dict(lam=lam_p, g=g, kappa=kappa, gamma=float(gamma), ef=ef, mttr=m_p)

    # ---------------- additional benchmark methods ----------------
    # hour-of-day rates (24 pooled multipliers)
    hod = np.arange(L) % 24
    Hind = np.eye(24)[hod]
    up_h = U @ Hind
    n_h = np.bincount(hod[th], minlength=24).astype(float)
    r_h = (n_h / np.maximum(up_h.sum(0), 1e-12)) / (n.sum() / up.sum())
    r_h = np.maximum(r_h, 1e-3)
    out["Hour-of-day"] = dict(lam=np.maximum(n, 0.5) / (up_h @ r_h), r=r_h, mttr=mttr)

    # two-state weather model: adverse days = days with highest daily peak (top 20 %)
    dp = profile[: (L // 24) * 24].reshape(-1, 24).max(1)
    adv = np.repeat(dp >= np.quantile(dp, PEAK_Q), 24)
    adv = np.concatenate([adv, np.zeros(L - len(adv), bool)])
    up_a, up_n = U[:, adv].sum(1), U[:, ~adv].sum(1)
    n_a = float(adv[th].sum()); n_n = float(len(th) - n_a)
    rho_w = (n_a / max(up_a.sum(), 1e-12)) / max(n_n / up_n.sum(), 1e-12)
    out["Weather"] = dict(lam=np.maximum(n, 0.5) / (up_n + rho_w * up_a), rho=rho_w, adv=adv, mttr=mttr)

    # continuous load-conditioned failure rate h = lam_i exp(b l(t)), repair independent of load
    best_b = (-np.inf, 0.0)
    for b in B_GRID:
        w = np.exp(b * profile)
        Eb = U @ w
        llb = float((b * profile[th]).sum() - (n * np.log(Eb)).sum())
        if llb > best_b[0]:
            best_b = (llb, b)
    b = best_b[1]
    out["Load-dependent"] = dict(lam=np.maximum(n, 0.5) / (U @ np.exp(b * profile)), b=float(b), mttr=mttr)

    # Weibull (aging-type) time to failure, constant over the year
    spells = []
    for i in range(nu):
        x = S_hist[i].astype(np.int8)
        d = np.diff(np.concatenate([[0], x, [0]]))
        st, en = np.where(d == 1)[0], np.where(d == -1)[0]
        ln = (en - st)[1:-1] if len(st) > 2 else np.array([])
        if len(ln) > 0:
            spells.append(ln / ln.mean())
    z = np.concatenate(spells) if spells else np.array([1.0])
    from scipy.stats import weibull_min
    from scipy.special import gamma as G_
    kshape = float(np.clip(weibull_min.fit(z, floc=0)[0], 0.3, 5.0))
    mean_up = up / np.maximum(n, 0.5)
    out["Weibull"] = dict(shape=kshape, scale=mean_up / G_(1 + 1 / kshape), mttr=mttr,
                          lam=np.maximum(n, 0.5) / up)
    return out


# ---------------------------------------------------------------------------
# prediction models (hazard matrix, repair multiplier, mttr) for simulation
# ---------------------------------------------------------------------------
def hazard_model(name, fit, profile):
    L = len(profile)
    f = fit[name]
    nu = len(f["lam"])
    if name == "COPT":
        return np.repeat(f["lam"][:, None], L, 1), None, f["mttr"]
    if name == "Seasonal":
        return f["lam"][:, None] * f["r"][month_index(L)][None, :], None, f["mttr"]
    if name == "Two-level":
        mult = np.where(f["peak"], f["rho"], 1.0)
        return f["lam"][:, None] * mult[None, :], None, f["mttr"]
    if name == "Proposed":
        rep = f["gamma"] + (1 - f["gamma"]) * (profile ** f["kappa"]) / f["ef"]
        return f["lam"][:, None] * f["g"][None, :], np.repeat(rep[None, :], nu, 0), f["mttr"]
    if name == "Hour-of-day":
        return f["lam"][:, None] * f["r"][np.arange(L) % 24][None, :], None, f["mttr"]
    if name == "Weather":
        mult = np.where(f["adv"], f["rho"], 1.0)
        return f["lam"][:, None] * mult[None, :], None, f["mttr"]
    if name == "Load-dependent":
        return f["lam"][:, None] * np.exp(f["b"] * profile)[None, :], None, f["mttr"]
    raise ValueError(name)


def simulate_weibull(shape, scale, mttr, L, n_years, seed, warmup=C.WARMUP_YEARS):
    """Alternating renewal process: Weibull up-times, exponential repairs."""
    N = len(scale)
    Ltot = L * (warmup + n_years); off = L * warmup
    rng = np.random.default_rng(seed)
    S = np.ones((N, Ltot - off), dtype=np.uint8)
    for i in range(N):
        t = 0.0
        while True:
            t += scale[i] * rng.weibull(shape)
            tf = int(t)
            if tf >= Ltot:
                break
            dur = max(1, int(round(mttr[i] * rng.exponential(1.0))))
            lo, hi = max(tf, off), min(tf + dur, Ltot)
            if hi > lo:
                S[i, lo - off:hi - off] = 0
            t = tf + dur
    return S


def predict_hl1(name, fit, caps, load, profile, years, seed):
    """HL-I LOLE and EENS predicted by a fitted method."""
    f = fit[name]
    if name == "COPT":
        fr = f["mttr"] * f["lam"] / (1 + f["mttr"] * f["lam"])
        return copt_indices(caps, {0: fr}, np.zeros(len(load), int), load)
    if name == "Seasonal":
        mi = month_index(len(load))
        frs = {m: f["mttr"] * f["lam"] * f["r"][m] / (1 + f["mttr"] * f["lam"] * f["r"][m]) for m in range(12)}
        return copt_indices(caps, frs, mi, load)
    if name == "Weibull":
        lole = eens = 0.0
        for k in range(0, years, 50):
            ny = min(50, years - k)
            S = simulate_weibull(f["shape"], f["scale"], f["mttr"], len(load), ny, seed=[seed, k])
            d = np.maximum(0.0, np.tile(load, ny) - caps @ S)
            lole += float((d > C.LOL_TOL_MW).sum()); eens += float(d.sum())
        return lole / years, eens / years
    H, rep, mttr = hazard_model(name, fit, profile)
    return simulate_hl1(H, rep, mttr, caps, load, years, seed)


def simulate_hl1(H, rep, mttr, caps, load, years, seed, chunk=50):
    lole = eens = 0.0
    done = 0
    k = 0
    while done < years:
        ny = min(chunk, years - done)
        S, _ = M.simulate_custom(H, rep, mttr, ny, seed=[seed, k])
        cap = caps @ S
        d = np.maximum(0.0, np.tile(load, ny) - cap)
        lole += float((d > C.LOL_TOL_MW).sum()); eens += float(d.sum())
        done += ny; k += 1
    return lole / years, eens / years
