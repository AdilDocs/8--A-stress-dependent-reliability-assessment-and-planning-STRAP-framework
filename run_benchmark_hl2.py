"""
run_benchmark_hl2.py — (1) composite (HL-II) comparison of fitted methods and
(2) computational performance of the network evaluation.

(1) For each system and true process (T0 constant rates, T1 stress-dependent),
    N_HISTORIES independent 20-year outage histories of the generating units is generated; every
    method is fitted to it and predicts HL-II indices over PRED_YEARS years.
    Branch outages follow the true process and are identical for all methods
    (common random numbers), so the comparison isolates the treatment of
    generating units.  True HL-II indices: 1000-year main runs.
    Methods: non-sequential state sampling (constant FOR), sequential
    two-state MCS, seasonal-rate MCS, two-level MCS, proposed.
(2) HL-II evaluation time per simulated year: minimum load-shedding LP at every
    hour (naive) vs cached maximum-servable-load approach (this work).

Output: results/benchmark_hl2.csv, results/benchmark_cpu.csv
"""
import time
import numpy as np
import pandas as pd
from multiprocessing import Pool

import config as C
import model as M
import methods as MT

SYSTEMS = ["IEEE39", "IEEE57", "IEEE118"]
TRUTHS = {"T0": "conventional", "T1": "proposed"}
N_HIST = 20
PRED_YEARS = 1000
NS_YEARS = 100
N_HISTORIES = 5
METHODS = ["Non-sequential MCS", "Sequential two-state MCS", "Seasonal-rate MCS", "Two-level MCS", "Proposed"]
FITKEY = {"Sequential two-state MCS": "COPT", "Seasonal-rate MCS": "Seasonal", "Two-level MCS": "Two-level",
          "Proposed": "Proposed", "Non-sequential MCS": "COPT"}
_C = {}


def ctx(name):
    if name not in _C:
        s = M.build_system(name)
        _C[name] = (s, M.Network(s))
    return _C[name]


def true_models(s, tk):
    nu, L, g = s["nu"], s["L"], C.GAMMA_FRACTION
    if tk == "T0":
        H = np.repeat(s["hbar"][:, None], L, 1)
        rep = None
    else:
        H = s["H"]
        rep = g + (1 - g) * s["sigma"] / s["sig_w"][:, None]
    return H, rep


def job(args):
    name, tk, method, hist = args
    s, net = ctx(name)
    nu, L = s["nu"], s["L"]
    sid = SYSTEMS.index(name)
    H, rep = true_models(s, tk)
    Hu, ru = H[:nu], (rep[:nu] if rep is not None else None)
    Hb, rb = H[nu:], (rep[nu:] if rep is not None else None)
    S_h, ev = M.simulate_custom(Hu, ru, s["mttr"][:nu], N_HIST, seed=[C.BASE_SEED, 700, sid, int(tk[1]), hist])
    fit = MT.fit_all(S_h, ev, s["profile"], N_HIST)
    load = s["peak"] * s["profile"]
    rows = []
    t0 = time.perf_counter()
    years = NS_YEARS if method == "Non-sequential MCS" else PRED_YEARS
    if method != "Non-sequential MCS":
        Hm, rm, mm = MT.hazard_model(FITKEY[method], fit, s["profile"])
    else:
        f = fit["COPT"]
        for_u = f["mttr"] * f["lam"] / (1 + f["mttr"] * f["lam"])
        for_b = s["fors"][nu:]
    for y in range(years):
        Sb, _ = M.simulate_custom(Hb, rb, s["mttr"][nu:], 1, seed=[C.BASE_SEED, 800, sid, int(tk[1]), y, hist])
        if method == "Non-sequential MCS":
            rng = np.random.default_rng([C.BASE_SEED, 810, sid, int(tk[1]), y, hist])
            Su = (rng.random((nu, L)) >= for_u[:, None]).astype(np.uint8)
            Sb = (rng.random((len(for_b), L)) >= for_b[:, None]).astype(np.uint8)
        else:
            Su, _ = M.simulate_custom(Hm, rm, mm, 1, seed=[C.BASE_SEED, 820, sid, int(tk[1]), y, METHODS.index(method), hist])
        shed = net.evaluate(np.vstack([Su, Sb]))
        d = shed.sum(0).astype(float)
        r = dict(system=name, truth=tk, method=method, year=y)
        r.update(M.indices(d, load))
        r.update(M.customer_indices(shed))
        rows.append(r)
    cpu = (time.perf_counter() - t0) / years
    df = pd.DataFrame(rows)
    out = df.drop(columns=["year"]).groupby(["system", "truth", "method"]).mean().reset_index()
    sd = df.groupby(["system", "truth", "method"])[["EENS", "LOLE"]].std().reset_index()
    out["EENS_CI95"] = 1.96 * sd["EENS"].values / np.sqrt(years)
    out["LOLE_CI95"] = 1.96 * sd["LOLE"].values / np.sqrt(years)
    out["years"] = years
    out["history"] = hist
    out["cpu_s_per_year"] = cpu
    out["kappa_hat"] = fit["Proposed"]["kappa"]
    out["gamma_hat"] = fit["Proposed"]["gamma"]
    if method == "Non-sequential MCS":
        out[["LOLF", "SAIFI"]] = np.nan
    return out


def cpu_job(name):
    s, _ = ctx(name)
    rows = []
    for y in range(3):
        S = M.simulate(s, "proposed", 5000 + y)
        net = M.Network(s)                          # cold cache: conservative
        t0 = time.perf_counter(); shed = net.evaluate(S); t_c = time.perf_counter() - t0
        n_c = net.n_lp
        net2 = M.Network(s)
        t0 = time.perf_counter()
        nu = s["nu"]
        shed2 = np.zeros_like(shed)
        for t in range(s["L"]):
            cap_bus = np.bincount(s["u_bus"], weights=s["u_cap"] * S[:nu, t], minlength=s["nb"])
            shed2[:, t] = net2.shed(cap_bus, S[nu:, t], s["profile"][t])
        t_n = time.perf_counter() - t0
        rows.append(dict(system=name, year=y, naive_s=t_n, cached_s=t_c, naive_lps=net2.n_lp, cached_lps=n_c,
                         max_abs_diff_MW=float(np.abs(shed.sum(0) - shed2.sum(0)).max())))
    return rows


def main():
    t0 = time.time()
    jobs = [(n, tk, m, h) for n in SYSTEMS for tk in TRUTHS for m in METHODS for h in range(N_HISTORIES)]
    with Pool(2) as p:
        res = p.map(job, jobs, chunksize=1)
    df = pd.concat(res, ignore_index=True)
    truth = []
    for n in SYSTEMS:
        mm = pd.read_csv(f"results/main_{n}.csv")
        for tk, v in TRUTHS.items():
            g = mm[mm.variant == v]
            truth.append(dict(system=n, truth=tk, EENS_true=g.HL2_EENS.mean(), LOLE_true=g.HL2_LOLE.mean(),
                              LOLF_true=g.HL2_LOLF.mean(), SAIFI_true=g.SAIFI.mean(), SAIDI_true=g.SAIDI.mean()))
    df = df.merge(pd.DataFrame(truth), on=["system", "truth"])
    for k in ("EENS", "LOLE", "SAIDI"):
        df[f"{k}_err_pct"] = 100 * (df[k] - df[f"{k}_true"]) / df[f"{k}_true"]
    df.to_csv("results/benchmark_hl2_by_history.csv", index=False)
    num = [c for c in df.columns if c not in ("system", "truth", "method", "history")]
    agg = df.groupby(["system", "truth", "method"])[num].mean().reset_index()
    for k in ("EENS", "LOLE", "SAIDI"):
        agg[f"{k}_abs_err_pct"] = df.groupby(["system", "truth", "method"])[f"{k}_err_pct"].apply(
            lambda x: np.abs(x).mean()).values
        agg[f"{k}_err_sd"] = df.groupby(["system", "truth", "method"])[f"{k}_err_pct"].std().values
    agg.to_csv("results/benchmark_hl2.csv", index=False)
    print(f"HL-II benchmark done {time.time() - t0:.0f}s", flush=True)
    import os
    if not os.path.exists("results/benchmark_cpu.csv"):
        with Pool(2) as p:
            cr = p.map(cpu_job, SYSTEMS)
        pd.DataFrame([x for r in cr for x in r]).to_csv("results/benchmark_cpu.csv", index=False)
        print(f"CPU benchmark done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
