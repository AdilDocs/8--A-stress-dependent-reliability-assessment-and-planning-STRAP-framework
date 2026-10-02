"""
model.py — stress-dependent reliability model (v3) and its baselines.

Sections of the paper -> functions
  Test system, data, calibration ............ build_system()
  Operating stress (Eq. 1) ................... stress_matrix()
  Failure / repair process (Eqs. 2-4) ........ simulate()
  HL-I / HL-II adequacy evaluation ........... hl1_deficit(), indices(), Network

Stress:   sigma_i(t) = e_i * l(t)^kappa,  l(t) = P_load(t) / P_peak,
          e_i = eta_L,i * (T_i / T_ref) * eta_A,i * eta_V,i
Failure:  h_i(t) = 1 / TTF_i(t),  TTF_i(t) = alpha_i + beta_i / (sigma_i(t) + eps);
          a component restored at t_r fails at the first hour at which the
          accumulated hazard since t_r exceeds an Exp(1) threshold
          (proportional-hazards / cumulative-damage process).
Repair:   exponential with mean
          MTTR_i * [gamma + (1 - gamma) * sigma_i(t_f) / E[sigma_i | failure]],
          so repairs of failures that occur under high stress take longer,
          while the mean repair time over all failures stays MTTR_i.
All variants share one engine and the same random numbers (common random
numbers), and every variant has the same mean failure rate and mean repair
time per component.
"""
import numpy as np
import scipy.sparse as sp
from scipy.optimize import brentq

import config as C
import data_sources as D

CLASSES = ("unit", "transformer", "line")

VARIANTS = {
    "conventional":   ("const", "exp"),     # two-state Markov model, sequential MCS
    "stress_failure": ("stress", "exp"),    # ablation: stress-dependent failure only
    "stress_repair":  ("const", "stress"),  # ablation: stress-dependent repair only
    "proposed":       ("stress", "stress"), # this paper: both (Eqs. 2-4)
}


def ttf_law(sigma, a, b):
    return a + b / (sigma + C.TTF_EPS)


def calibrate_beta(mttf, e_nom, lam):
    """alpha, beta such that the nominal component's mean hazard over the
    chronological loading lam(t) equals 1/MTTF."""
    a = C.ALPHA_FRACTION * mttf
    f = lambda lb: np.mean(1.0 / ttf_law(e_nom * lam, a, np.exp(lb))) - 1.0 / mttf
    b = np.exp(brentq(f, np.log(1e-4 * mttf), np.log(1e4 * mttf)))
    return a, b


def _rts79_raw():
    """IEEE RTS-79 generating system as a single-bus (HL-I) case."""
    bus = np.array([[1, 3, 1.0] + [0] * 10])
    gen = np.array([[1, 0, 0, 0, 0, 1, 100, 1, c, 0] for c in D.RTS79_SYSTEM], float)
    return dict(baseMVA=100.0, bus=bus, gen=gen, branch=np.zeros((0, 13)))


def build_system(name, kappa=C.COUPLING_EXP, lole_target=C.LOLE_TARGET_H,
                 alpha_fraction=None, eps=None):
    """alpha_fraction / eps override config (sensitivity studies)."""
    if alpha_fraction is not None:
        C.ALPHA_FRACTION = alpha_fraction
    if eps is not None:
        C.TTF_EPS = eps
    raw = _rts79_raw() if name == "RTS79" else D.load_pglib(name)
    bus, gen, br = raw["bus"], raw["gen"], raw["branch"]
    bidx = {int(b): k for k, b in enumerate(bus[:, 0])}
    nb = len(bus)
    ref = int(np.where(bus[:, 1] == 3)[0][0])

    units = []
    for g in gen:
        if g[7] <= 0 or g[8] <= 0:
            continue
        n = int(np.ceil(g[8] / C.MAX_UNIT_MW - 1e-9))
        units += [(bidx[int(g[0])], g[8] / n)] * n
    u_bus = np.array([u[0] for u in units])
    u_cap = np.array([u[1] for u in units])

    br = br[br[:, 10] > 0] if len(br) else br
    f_bus = np.array([bidx[int(v)] for v in br[:, 0]], dtype=int)
    t_bus = np.array([bidx[int(v)] for v in br[:, 1]], dtype=int)
    tap = np.where(br[:, 8] == 0, 1.0, br[:, 8]) if len(br) else np.zeros(0)
    b_mw = raw["baseMVA"] / (br[:, 3] * tap) if len(br) else np.zeros(0)
    rating = np.where(br[:, 5] > 0, br[:, 5], np.inf).astype(float) if len(br) else np.zeros(0)
    rating[rating >= C.UNLIMITED_RATING_MW] = np.inf
    is_trafo = br[:, 8] != 0 if len(br) else np.zeros(0, bool)

    pd_ = np.clip(bus[:, 2], 0, None) if name != "RTS79" else np.array([1.0])
    load_bus = np.where(pd_ > 0)[0]
    share = pd_[load_bus] / pd_[load_bus].sum()

    cls = ["unit"] * len(u_cap) + ["transformer" if t else "line" for t in is_trafo]
    ids = [f"U{k}@{u_bus[k] + 1}" for k in range(len(u_cap))] + \
          [("T" if t else "L") + f"{k}:{f_bus[k] + 1}-{t_bus[k] + 1}" for k, t in enumerate(is_trafo)]
    N, nu = len(cls), len(u_cap)
    mttf = np.empty(N); mttr = np.empty(N)
    for k in range(nu):
        mttf[k], mttr[k] = D.unit_reliability(u_cap[k])
    for k in range(len(is_trafo)):
        dd = D.TRAFO_DATA if is_trafo[k] else D.LINE_DATA
        mttf[nu + k] = 8760.0 / dd["lam_per_yr"]
        mttr[nu + k] = dd["mttr_h"]

    rng = np.random.default_rng([C.BASE_SEED, C.SYSTEMS.index(name) if name in C.SYSTEMS else 99])
    fac = {q: np.empty(N) for q in ("eta_L", "T", "eta_A", "eta_V")}
    for k, c in enumerate(cls):
        for q in fac:
            fac[q][k] = rng.uniform(*C.STRESS_RANGES[c][q])
            if name == "RTS79":          # published unit data used as is
                fac[q][k] = np.mean(C.STRESS_RANGES[c][q])
    e = fac["eta_L"] * fac["T"] / C.T_REF * fac["eta_A"] * fac["eta_V"]

    profile = D.rts_load_profile()
    lam = profile ** kappa
    a = np.empty(N); b = np.empty(N)
    cache = {}
    for k, c in enumerate(cls):
        mid = {q: np.mean(v) for q, v in C.STRESS_RANGES[c].items()}
        e_nom = mid["eta_L"] * mid["T"] / C.T_REF * mid["eta_A"] * mid["eta_V"]
        key = (c, mttf[k])
        if key not in cache:
            cache[key] = calibrate_beta(mttf[k], e_nom, lam)
        a[k], b[k] = cache[key]
    sigma = e[:, None] * lam[None, :]                          # N x L, one year
    H = 1.0 / ttf_law(sigma, a[:, None], b[:, None])           # hazard per hour
    hbar = H.mean(axis=1)                                      # common failure rate
    sig_w = (H * sigma).sum(1) / H.sum(1)                      # E[sigma | failure], stress hazard
    fors = mttr / (1.0 / hbar + mttr)

    fu = fors[:nu]
    f = lambda pk: copt_lole(u_cap, fu, pk * profile)[0] - lole_target
    peak = brentq(f, 0.3 * u_cap.sum(), u_cap.sum())
    lole1, eens1 = copt_lole(u_cap, fu, peak * profile)
    rating_scale = max(1.0, peak / pd_.sum())
    rating = rating * rating_scale

    # cumulative hazard over warm-up + study year (proposed and conventional)
    reps = C.WARMUP_YEARS + 1
    Hc_stress = np.cumsum(np.tile(H, reps), axis=1)
    return dict(name=name, nb=nb, ref=ref, f_bus=f_bus, t_bus=t_bus, b_mw=b_mw, rating=rating,
                rating_scale=rating_scale, u_bus=u_bus, u_cap=u_cap, nu=nu, load_bus=load_bus,
                share=share, profile=profile, peak=peak, L=len(profile), cls=np.array(cls), ids=ids,
                N=N, mttf=mttf, mttr=mttr, fac=fac, e=e, a=a, b=b, kappa=kappa, H=H, hbar=hbar,
                sigma=sigma, sig_w=sig_w, sig_mean=sigma.mean(1), Hc_stress=Hc_stress, fors=fors,
                analytic_hl1=dict(LOLE=lole1, EENS=eens1), reserve_margin=u_cap.sum() / peak - 1.0)


def simulate(s, variant, year, want_counts=False):
    """Component states for the study year (N x L, 1 = up)."""
    hmode, rmode = VARIANTS[variant]
    N, L = s["N"], s["L"]
    Ltot = L * (C.WARMUP_YEARS + 1)
    off = Ltot - L
    sysid = C.SYSTEMS.index(s["name"]) if s["name"] in C.SYSTEMS else 99
    rng = np.random.default_rng([C.BASE_SEED, 7, sysid, year])
    K = 400
    Z = rng.exponential(1.0, size=(N, K))      # failure thresholds (common random numbers)
    E = rng.exponential(1.0, size=(N, K))      # repair draws (common random numbers)
    S = np.ones((N, L), dtype=np.uint8)
    fails = np.zeros(N, dtype=int)
    for i in range(N):
        Hc = s["Hc_stress"][i] if hmode == "stress" else None
        t, n = 0, 0
        while t < Ltot and n < K:
            if Hc is None:
                t_f = max(t, t + int(np.ceil(Z[i, n] / s["hbar"][i])) - 1)
            else:
                base = Hc[t - 1] if t > 0 else 0.0
                t_f = int(np.searchsorted(Hc, base + Z[i, n], side="left"))
            if t_f >= Ltot:
                break
            ttr = s["mttr"][i] * E[i, n]
            if rmode == "stress":
                g = C.GAMMA_FRACTION
                norm = s["sig_w"][i] if hmode == "stress" else s["sig_mean"][i]
                ttr *= g + (1.0 - g) * s["sigma"][i, t_f % L] / norm
            t_up = t_f + max(1, int(round(ttr)))
            lo, hi = max(t_f, off), min(t_up, Ltot)
            if hi > lo:
                S[i, lo - off:hi - off] = 0
            if t_f >= off:
                fails[i] += 1
            t, n = t_up, n + 1
    return (S, fails) if want_counts else S


def simulate_custom(H, rep, mttr, n_years, seed, warmup=C.WARMUP_YEARS):
    """Generic chronological simulation used by the method benchmark.
    H: (N x L) hourly failure hazard over one year (repeated every year);
    rep: (N x L) repair-mean multiplier by failure hour, or None;
    mttr: (N,) mean repair times.  Returns states (N x L*n_years) and the
    failure events inside the window as arrays (unit, hour, duration)."""
    N, L = H.shape
    Ltot = L * (warmup + n_years)
    off = L * warmup
    rng = np.random.default_rng(seed)
    Hc_all = np.cumsum(np.tile(H, warmup + n_years), axis=1)
    S = np.ones((N, Ltot - off), dtype=np.uint8)
    ev_u, ev_t, ev_d = [], [], []
    for i in range(N):
        Hc = Hc_all[i]
        t = 0
        while t < Ltot:
            base = Hc[t - 1] if t > 0 else 0.0
            t_f = int(np.searchsorted(Hc, base + rng.exponential(1.0), side="left"))
            if t_f >= Ltot:
                break
            ttr = mttr[i] * rng.exponential(1.0)
            if rep is not None:
                ttr *= rep[i, t_f % L]
            dur = max(1, int(round(ttr)))
            t_up = t_f + dur
            lo, hi = max(t_f, off), min(t_up, Ltot)
            if hi > lo:
                S[i, lo - off:hi - off] = 0
            if t_f >= off:
                ev_u.append(i); ev_t.append(t_f - off); ev_d.append(dur)
            t = t_up
    return S, (np.array(ev_u, int), np.array(ev_t, int), np.array(ev_d, float))



def copt_lole(caps, fors, load):
    """Analytical HL-I LOLE (h) and EENS (MWh) by capacity outage probability table."""
    tot = int(round(sum(caps)))
    P = np.zeros(tot + 1)
    P[0] = 1.0
    for c, q in zip(caps, fors):
        c = int(round(c))
        new = P * (1 - q)
        new[c:] += P[:tot + 1 - c] * q
        P = new
    avail = tot - np.arange(tot + 1)
    # cumulative: prob(avail < L)
    order = np.argsort(avail)
    av_s, P_s = avail[order], P[order]
    cumP = np.cumsum(P_s)
    cumPA = np.cumsum(P_s * av_s)
    k = np.searchsorted(av_s, load, side="left")       # avail < L  -> indices < k
    pl = np.where(k > 0, cumP[np.maximum(k - 1, 0)], 0.0)
    pa = np.where(k > 0, cumPA[np.maximum(k - 1, 0)], 0.0)
    return float(pl.sum()), float((load * pl - pa).sum())



def hl1_deficit(s, S):
    cap = (s["u_cap"][:, None] * S[:s["nu"]]).sum(0)
    return np.maximum(0.0, s["peak"] * s["profile"] - cap)


def indices(deficit, load):
    lol = deficit > C.LOL_TOL_MW
    starts = np.diff(np.r_[0, lol.astype(int)]) == 1
    return dict(EENS=float(deficit.sum() * C.DT), LOLE=float(lol.sum() * C.DT),
                LOLP=float(lol.mean()), LOLF=float(starts.sum()),
                R_sys=1.0 - float(deficit.sum()) / float(load.sum()))



class Network:
    """DC network adequacy with a cache of the maximum servable load level
    alpha*(outage state).  Hours with alpha_t <= alpha* need no LP; only the
    remaining hours solve the minimum load-shedding LP."""

    def __init__(self, s, inj_buses=()):
        import highspy
        self.s = s
        nb, nl, nd = s["nb"], len(s["b_mw"]), len(s["load_bus"])
        self.nb, self.nl, self.nd = nb, nl, nd
        self.inj = np.asarray(inj_buses, dtype=int)     # extra injections (storage)
        ni = len(self.inj)
        self.Dpk = np.zeros(nb)
        self.Dpk[s["load_bus"]] = s["share"] * s["peak"]
        # columns: theta | Pg(bus) | f | slack | shed(load) | alpha
        o_pg, o_f = nb, 2 * nb
        o_s, o_sh = o_f + nl, o_f + 2 * nl
        o_a = o_sh + nd
        o_in = o_a + 1
        ncol = o_in + ni
        self.o = dict(pg=o_pg, f=o_f, s=o_s, sh=o_sh, a=o_a, inj=o_in)
        R, Cc, V = [], [], []
        for l in range(nl):
            i, j, bb = s["f_bus"][l], s["t_bus"][l], s["b_mw"][l]
            R += [l] * 4; Cc += [o_f + l, i, j, o_s + l]; V += [1.0, -bb, bb, -1.0]
        for bb in range(nb):
            R.append(nl + bb); Cc.append(o_pg + bb); V.append(1.0)
        for l in range(nl):
            R += [nl + s["f_bus"][l], nl + s["t_bus"][l]]; Cc += [o_f + l] * 2; V += [-1.0, 1.0]
        for k, bb in enumerate(s["load_bus"]):
            R.append(nl + bb); Cc.append(o_sh + k); V.append(1.0)
            R.append(nl + bb); Cc.append(o_a); V.append(-s["share"][k] * s["peak"])
        for k, bb in enumerate(self.inj):
            R.append(nl + bb); Cc.append(o_in + k); V.append(1.0)
        A = sp.csc_matrix((V, (R, Cc)), shape=(nl + nb, ncol))
        inf = highspy.kHighsInf
        self.inf = inf
        lo = np.full(ncol, -inf); up = np.full(ncol, inf)
        lo[s["ref"]] = up[s["ref"]] = 0.0
        lo[o_pg:o_f] = 0.0
        lo[o_sh:o_a] = 0.0; up[o_sh:o_a] = 0.0
        lo[o_a] = 0.0; up[o_a] = 2.0
        lo[o_in:] = 0.0; up[o_in:] = 0.0

        def make(cost):
            lp = highspy.HighsLp()
            lp.num_col_, lp.num_row_ = ncol, nl + nb
            lp.col_cost_, lp.col_lower_, lp.col_upper_ = cost, lo.copy(), up.copy()
            lp.row_lower_ = np.zeros(nl + nb); lp.row_upper_ = np.zeros(nl + nb)
            lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
            lp.a_matrix_.start_ = A.indptr; lp.a_matrix_.index_ = A.indices; lp.a_matrix_.value_ = A.data
            h = highspy.Highs(); h.setOptionValue("output_flag", False); h.passModel(lp)
            return h
        c_max = np.zeros(ncol); c_max[o_a] = -1.0
        c_shed = np.zeros(ncol); c_shed[o_sh:o_a] = 1.0 + 1e-4 * np.arange(nd) / nd
        c_shed[o_in:] = 1e-5                     # storage used only when it reduces curtailment
        self.h_max, self.h_shed = make(c_max), make(c_shed)
        self.var = np.arange(o_pg, ncol, dtype=np.int32)
        self.cache = {}
        self.bus_of_unit = s["u_bus"]
        self.rating = np.where(np.isinf(s["rating"]), inf, s["rating"])
        self.n_lp = 0

    def _bounds(self, cap_bus, a_br, alpha=None, shed=False, inj_max=None):
        nd = self.nd
        im = np.zeros(len(self.inj)) if inj_max is None else inj_max
        f_up = np.where(a_br == 1, self.rating, 0.0)
        sl = np.where(a_br == 1, 0.0, self.inf)
        if shed:
            dmax = alpha * self.s["share"] * self.s["peak"]
            lo = np.r_[np.zeros(self.nb), -f_up, -sl, np.zeros(nd), alpha, np.zeros(len(im))]
            up = np.r_[cap_bus, f_up, sl, dmax, alpha, im]
        else:
            lo = np.r_[np.zeros(self.nb), -f_up, -sl, np.zeros(nd), 0.0, np.zeros(len(im))]
            up = np.r_[cap_bus, f_up, sl, np.zeros(nd), 2.0, np.zeros(len(im))]
        return lo, up

    def alpha_star(self, key, cap_bus, a_br):
        v = self.cache.get(key)
        if v is None:
            lo, up = self._bounds(cap_bus, a_br)
            self.h_max.changeColsBounds(len(self.var), self.var, lo, up)
            self.h_max.run()
            v = float(self.h_max.getSolution().col_value[self.o["a"]])
            self.cache[key] = v
            self.n_lp += 1
        return v

    def shed(self, cap_bus, a_br, alpha, inj_max=None, return_inj=False):
        lo, up = self._bounds(cap_bus, a_br, alpha, shed=True, inj_max=inj_max)
        self.h_shed.changeColsBounds(len(self.var), self.var, lo, up)
        self.h_shed.run()
        self.n_lp += 1
        x = np.asarray(self.h_shed.getSolution().col_value)
        sh = np.clip(x[self.o["sh"]:self.o["a"]], 0.0, None)
        if return_inj:
            return sh, np.clip(x[self.o["inj"]:], 0.0, None)
        return sh

    def evaluate(self, S, return_events=False):
        """HL-II per-load-bus shedding (nd x L) for component states S."""
        s = self.s
        nu = s["nu"]
        Su, Sb = S[:nu], S[nu:]
        alpha_t = s["profile"]
        L = s["L"]
        shed = np.zeros((self.nd, L), dtype=np.float32)
        # hours where the state changes
        chg = np.r_[True, np.any(S[:, 1:] != S[:, :-1], axis=0)]
        seg_start = np.where(chg)[0]
        seg_end = np.r_[seg_start[1:], L]
        events = []
        for a0, a1 in zip(seg_start, seg_end):
            su, sb = Su[:, a0], Sb[:, a0]
            key = su.tobytes() + sb.tobytes()
            if alpha_t[a0:a1].max() <= 0.0:
                continue
            cap_bus = np.bincount(self.bus_of_unit, weights=s["u_cap"] * su, minlength=self.nb)
            ast = self.alpha_star(key, cap_bus, sb)
            hrs = np.arange(a0, a1)[alpha_t[a0:a1] > ast + 1e-9]
            for t in hrs:
                shed[:, t] = self.shed(cap_bus, sb, alpha_t[t])
            if return_events and len(hrs):
                events.append((hrs, su.copy(), sb.copy()))
        return (shed, events) if return_events else shed



def customer_indices(shed):
    inter = shed > C.LOL_TOL_MW
    starts = np.diff(np.c_[np.zeros((inter.shape[0], 1), bool), inter].astype(int), axis=1) == 1
    n = inter.shape[0]
    return dict(SAIFI=float(starts.sum() / n), SAIDI=float(inter.sum() * C.DT / n))
