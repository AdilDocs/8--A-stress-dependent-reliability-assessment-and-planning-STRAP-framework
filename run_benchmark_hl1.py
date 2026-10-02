"""
run_benchmark_hl1.py — known-truth comparison of adequacy methods (HL-I).

For each system and each "true" outage process (T0-T3), REPS independent
outage histories of N_HIST years are generated.  Every method is fitted to the
same history and predicts LOLE and EENS; predictions are compared with the
true indices (TRUTH_YEARS simulated years of the true process).

  T0  constant failure and repair rates (conventional model is exact)
  T1  stress-dependent failure and repair, kappa = 1
  T2  stress-dependent failure and repair, kappa = 2
  T3  mis-specified for every method: hazard ~ exp(3 (l(t) - 1)), constant repair

Output: results/benchmark_hl1.csv, results/benchmark_truth_hl1.csv
"""
import sys
import time
import numpy as np
import pandas as pd
from multiprocessing import Pool

import config as C
import model as M
import methods as MT

SYSTEMS = ["RTS79", "IEEE39", "IEEE118"]
TRUTHS = ["T0", "T1", "T2", "T3"]
TRUTH_LAB = {"T0": "Constant rates", "T1": "Stress-dependent, kappa=1",
             "T2": "Stress-dependent, kappa=2", "T3": "Exponential-in-load (mis-specified)"}
REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 20
N_HIST = 10
PRED_YEARS = 1000
TRUTH_YEARS = 4000
METHODS = ["COPT", "Seasonal", "Two-level", "Proposed"]
_T = {}


def truth(name, tk):
    key = (name, tk)
    if key in _T:
        return _T[key]
    s = M.build_system(name)
    nu, L = s["nu"], s["L"]
    p = s["profile"]
    hbar, mttr = s["hbar"][:nu], s["mttr"][:nu]
    g = C.GAMMA_FRACTION
    if tk == "T0":
        H, rep = np.repeat(hbar[:, None], L, 1), None
    elif tk in ("T1", "T2"):
        ss = s if tk == "T1" else M.build_system(name, kappa=2.0)
        H = ss["H"][:nu] * (hbar / ss["H"][:nu].mean(1))[:, None]
        sig = ss["sigma"][:nu]
        sw = (H * sig).sum(1) / H.sum(1)
        rep = g + (1 - g) * sig / sw[:, None]
    else:
        shape = np.exp(3.0 * (p - 1.0))
        H, rep = hbar[:, None] * (shape / shape.mean())[None, :], None
    d = dict(s=s, H=H, rep=rep, mttr=mttr, caps=s["u_cap"], load=s["peak"] * p, profile=p)
    _T[key] = d
    return d


def true_indices(args):
    name, tk = args
    t = truth(name, tk)
    lole, eens = MT.simulate_hl1(t["H"], t["rep"], t["mttr"], t["caps"], t["load"], TRUTH_YEARS,
                                 seed=[C.BASE_SEED, 900, SYSTEMS.index(name), TRUTHS.index(tk)])
    return dict(system=name, truth=tk, LOLE_true=lole, EENS_true=eens)


def job(args):
    name, tk, r = args
    t = truth(name, tk)
    S, ev = M.simulate_custom(t["H"], t["rep"], t["mttr"], N_HIST,
                              seed=[C.BASE_SEED, 500, SYSTEMS.index(name), TRUTHS.index(tk), r])
    t0 = time.perf_counter()
    fit = MT.fit_all(S, ev, t["profile"], N_HIST)
    t_fit = time.perf_counter() - t0
    rows = []
    for m in METHODS:
        t1 = time.perf_counter()
        lole, eens = MT.predict_hl1(m, fit, t["caps"], t["load"], t["profile"], PRED_YEARS,
                                    seed=[C.BASE_SEED, 600, SYSTEMS.index(name), TRUTHS.index(tk), r, METHODS.index(m)])
        rows.append(dict(system=name, truth=tk, rep=r, method=m, LOLE=lole, EENS=eens,
                         cpu_s=time.perf_counter() - t1 + (t_fit if m == "Proposed" else 0.0),
                         kappa_hat=fit["Proposed"]["kappa"] if m == "Proposed" else np.nan,
                         gamma_hat=fit["Proposed"]["gamma"] if m == "Proposed" else np.nan,
                         n_failures=len(ev[0])))
    return rows


def main():
    t0 = time.time()
    with Pool(2) as p:
        tr = p.map(true_indices, [(n, tk) for n in SYSTEMS for tk in TRUTHS])
    pd.DataFrame(tr).to_csv("results/benchmark_truth_hl1.csv", index=False)
    print(f"truth done {time.time() - t0:.0f}s", flush=True)
    with Pool(2) as p:
        res = p.map(job, [(n, tk, r) for n in SYSTEMS for tk in TRUTHS for r in range(REPS)], chunksize=2)
    df = pd.DataFrame([x for rr in res for x in rr]).merge(pd.DataFrame(tr), on=["system", "truth"])
    df["LOLE_err_pct"] = 100 * (df.LOLE - df.LOLE_true) / df.LOLE_true
    df["EENS_err_pct"] = 100 * (df.EENS - df.EENS_true) / df.EENS_true
    df.to_csv("results/benchmark_hl1.csv", index=False)
    print(f"done {time.time() - t0:.0f}s", flush=True)
    print(df.groupby(["system", "truth", "method"])[["LOLE_err_pct", "EENS_err_pct"]]
          .agg(lambda x: np.mean(np.abs(x))).round(1).unstack().to_string())


if __name__ == "__main__":
    main()
