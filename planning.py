"""
planning.py — reliability-oriented storage siting and sizing (Section III-E, v2).

Decision:  P_b >= 0, storage power at every load bus b that curtails load in
the training scenarios (energy E_b = STORAGE_HOURS * P_b).
Objective ($/yr):
    J(P) = CRF * CAPEX * sum_b P_b                       (annualised investment)
         + VOLL * EENS(P)                                 (expected outage cost)
         + LOLE_PENALTY * max(0, LOLE(P) - LOLE_TARGET)   (reliability target)
EENS(P), LOLE(P) are sample averages over chronological scenario years of the
HL-II network model.  Each bus storage serves the curtailed load of its own bus
(discharge = min(curtailment, P_b, eta_d * SOC)); between curtailment hours it
recharges at P_b.  LOLE counts hours with residual curtailment at ANY bus, so
the objective is non-separable and piecewise constant in P (non-convex) —
which is why a population-based optimiser is used.

The final plans are re-evaluated out-of-sample with the full DC network LP in
which storage can also serve other buses (validate_plan()).
"""
import numpy as np
import numba as nb

import config as C
import model as M


# ---------------------------------------------------------------------------
# scenario data: sparse bus-level curtailment
# ---------------------------------------------------------------------------
def collect(name, variant, years, s=None, net=None):
    """Returns dict with hours (per record), year id, bus index, MW; plus the
    number of years.  Only curtailment hours are stored."""
    s = s or M.build_system(name)
    net = net or M.Network(s)
    recs = []
    for y in years:
        S = M.simulate(s, variant, y)
        shed = net.evaluate(S)
        b, t = np.nonzero(shed > C.LOL_TOL_MW)
        for bb, tt in zip(b, t):
            recs.append((y, tt, bb, float(shed[bb, tt])))
    arr = np.array(recs, dtype=float).reshape(-1, 4)
    return dict(year=arr[:, 0].astype(np.int64), hour=arr[:, 1].astype(np.int64),
                bus=arr[:, 2].astype(np.int64), mw=arr[:, 3], n_years=len(years))


class Scenarios:
    """Dense (hour-record x candidate-bus) curtailment matrix for fast evaluation."""

    def __init__(self, data, candidates=None):
        buses = np.unique(data["bus"]) if candidates is None else np.asarray(candidates)
        self.candidates = buses
        col = {b: k for k, b in enumerate(buses)}
        key = data["year"] * 100_000 + data["hour"]
        uk, inv = np.unique(key, return_inverse=True)
        self.year = (uk // 100_000).astype(np.int64)
        self.hour = (uk % 100_000).astype(np.int64)
        self.shed = np.zeros((len(uk), len(buses)))
        self.other = np.zeros(len(uk))      # curtailment at non-candidate buses
        for r, b, mw in zip(inv, data["bus"], data["mw"]):
            if b in col:
                self.shed[r, col[b]] += mw
            else:
                self.other[r] += mw
        self.n_years = data["n_years"]
        self.pmax = np.maximum(self.shed.max(axis=0) if len(uk) else np.zeros(len(buses)), 1.0)

    def evaluate(self, P):
        P = np.atleast_2d(P)
        eens, lole = _kernel(P, self.shed, self.other, self.year, self.hour,
                             C.STORAGE_HOURS, C.ETA_CH, C.ETA_DIS, C.LOL_TOL_MW)
        return eens / self.n_years, lole / self.n_years

    def objective(self, P):
        P = np.atleast_2d(P)
        eens, lole = self.evaluate(P)
        return (C.CRF * C.STORAGE_CAPEX * P.sum(axis=1) + C.VOLL * eens
                + C.LOLE_PENALTY * np.maximum(0.0, lole - C.LOLE_TARGET_PLAN))


@nb.njit(cache=True)
def _kernel(P, shed, other, year, hour, hrs, eta_c, eta_d, tol):
    NP, K = P.shape
    H = shed.shape[0]
    eens = np.zeros(NP)
    lole = np.zeros(NP)
    for p in range(NP):
        soc = np.empty(K)
        prev_shed = np.zeros(K, dtype=np.bool_)
        prev_y, prev_t = -1, 0
        for k in range(H):
            if year[k] != prev_y:                      # new year: storage full
                for b in range(K):
                    soc[b] = hrs * P[p, b]
                    prev_shed[b] = False
                prev_y = year[k]
            else:
                gap = hour[k] - prev_t - 1
                for b in range(K):
                    add = gap + (0 if prev_shed[b] else 1)
                    soc[b] = min(hrs * P[p, b], soc[b] + eta_c * P[p, b] * add)
            lol = other[k] > tol
            e = other[k]
            for b in range(K):
                sd = shed[k, b]
                if sd > 0.0:
                    d = min(sd, P[p, b], soc[b] * eta_d)
                    soc[b] -= d / eta_d
                    res = sd - d
                    e += res
                    if res > tol:
                        lol = True
                    prev_shed[b] = True
                else:
                    prev_shed[b] = False
            eens[p] += e
            if lol:
                lole[p] += 1.0
            prev_t = hour[k]
    return eens, lole


# ---------------------------------------------------------------------------
# full-network validation: chronological, storage can serve any bus
# ---------------------------------------------------------------------------
def validate_plan(s, variant, years, buses, P):
    """Out-of-sample EENS / LOLE of a plan with the full DC network."""
    P = np.asarray(P, float)
    # `buses` are load-point indices (rows of the curtailment matrix)
    net = M.Network(s, inj_buses=s["load_bus"][np.asarray(buses, dtype=int)])
    E = C.STORAGE_HOURS * P
    eens, lole = [], []
    for y in years:
        S = M.simulate(s, variant, y)
        base, events = net.evaluate(S, return_events=True)
        soc = E.copy()
        last = -10 ** 9
        e_y, l_y = 0.0, 0
        hrs = sorted(set(int(t) for h, _, _ in events for t in h))
        state = {}
        for h, su, sb in events:
            for t in h:
                state[int(t)] = (su, sb)
        for t in hrs:
            soc = np.minimum(E, soc + C.ETA_CH * P * max(0, t - last - 1))
            su, sb = state[t]
            cap_bus = np.bincount(s["u_bus"], weights=s["u_cap"] * su, minlength=s["nb"])
            dis_max = np.minimum(P, soc * C.ETA_DIS)
            sh, use = net.shed(cap_bus, sb, s["profile"][t], inj_max=dis_max, return_inj=True)
            soc = soc - use / C.ETA_DIS
            e_y += float(sh.sum())
            l_y += int(sh.sum() > C.LOL_TOL_MW)
            last = t
        eens.append(e_y)
        lole.append(l_y)
    return float(np.mean(eens)), float(np.mean(lole))
