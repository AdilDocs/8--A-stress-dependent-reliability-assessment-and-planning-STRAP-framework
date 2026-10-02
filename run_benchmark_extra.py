"""
run_benchmark_extra.py — adds four further methods to the known-truth benchmark,
using exactly the same outage histories, true processes and true indices as
run_benchmark_hl1.py / run_benchmark_hl2.py:
  Hour-of-day     24 pooled hour-of-day failure-rate multipliers
  Weather         two-state weather model (adverse days = top-20 % daily peaks)
  Load-dependent  continuous load-conditioned rate lam*exp(b*l(t)), b by ML
  Weibull         Weibull (aging-type) up-times, exponential repair
Output: results/benchmark_hl1_extra.csv, results/benchmark_hl2_extra.csv
"""
import time
import numpy as np
import pandas as pd
from multiprocessing import Pool

import config as C
import model as M
import methods as MT
import run_benchmark_hl1 as B1
import run_benchmark_hl2 as B2

NEW = ["Hour-of-day", "Weather", "Load-dependent", "Weibull"]


def job1(args):
    name, tk, r = args
    t = B1.truth(name, tk)
    S, ev = M.simulate_custom(t["H"], t["rep"], t["mttr"], B1.N_HIST,
                              seed=[C.BASE_SEED, 500, B1.SYSTEMS.index(name), B1.TRUTHS.index(tk), r])
    fit = MT.fit_all(S, ev, t["profile"], B1.N_HIST)
    rows = []
    for j, m in enumerate(NEW):
        lole, eens = MT.predict_hl1(m, fit, t["caps"], t["load"], t["profile"], B1.PRED_YEARS,
                                    seed=[C.BASE_SEED, 600, B1.SYSTEMS.index(name), B1.TRUTHS.index(tk), r, 10 + j])
        rows.append(dict(system=name, truth=tk, rep=r, method=m, LOLE=lole, EENS=eens))
    return rows


def job2(args):
    name, tk, method, hist = args
    s, net = B2.ctx(name)
    nu, L = s["nu"], s["L"]
    sid = B2.SYSTEMS.index(name)
    H, rep = B2.true_models(s, tk)
    Hu, ru = H[:nu], (rep[:nu] if rep is not None else None)
    Hb, rb = H[nu:], (rep[nu:] if rep is not None else None)
    S_h, ev = M.simulate_custom(Hu, ru, s["mttr"][:nu], B2.N_HIST, seed=[C.BASE_SEED, 700, sid, int(tk[1]), hist])
    fit = MT.fit_all(S_h, ev, s["profile"], B2.N_HIST)
    load = s["peak"] * s["profile"]
    rows = []
    t0 = time.perf_counter()
    if method != "Weibull":
        Hm, rm, mm = MT.hazard_model(method, fit, s["profile"])
    for y in range(B2.PRED_YEARS):
        Sb, _ = M.simulate_custom(Hb, rb, s["mttr"][nu:], 1, seed=[C.BASE_SEED, 800, sid, int(tk[1]), y, hist])
        sd = [C.BASE_SEED, 820, sid, int(tk[1]), y, 10 + NEW.index(method), hist]
        if method == "Weibull":
            f = fit["Weibull"]
            Su = MT.simulate_weibull(f["shape"], f["scale"], f["mttr"], L, 1, seed=sd)
        else:
            Su, _ = M.simulate_custom(Hm, rm, mm, 1, seed=sd)
        shed = net.evaluate(np.vstack([Su, Sb]))
        d = shed.sum(0).astype(float)
        r = dict(system=name, truth=tk, method=method, year=y, history=hist)
        r.update(M.indices(d, load)); r.update(M.customer_indices(shed))
        rows.append(r)
    cpu = (time.perf_counter() - t0) / B2.PRED_YEARS
    df = pd.DataFrame(rows)
    out = df.drop(columns=["year"]).groupby(["system", "truth", "method", "history"]).mean().reset_index()
    out["cpu_s_per_year"] = cpu
    return out


if __name__ == "__main__":
    t0 = time.time()
    tr = pd.read_csv("results/benchmark_truth_hl1.csv")
    with Pool(2) as p:
        res = p.map(job1, [(n, tk, r) for n in B1.SYSTEMS for tk in B1.TRUTHS for r in range(20)], chunksize=2)
    df = pd.DataFrame([x for rr in res for x in rr]).merge(tr, on=["system", "truth"])
    df["LOLE_err_pct"] = 100 * (df.LOLE - df.LOLE_true) / df.LOLE_true
    df["EENS_err_pct"] = 100 * (df.EENS - df.EENS_true) / df.EENS_true
    df.to_csv("results/benchmark_hl1_extra.csv", index=False)
    print(f"HL-I extra done {time.time() - t0:.0f}s", flush=True)
    print(df.groupby(["system", "truth", "method"]).LOLE_err_pct.apply(lambda x: np.abs(x).mean()).round(1).unstack().to_string(), flush=True)

    jobs = [(n, tk, m, h) for n in B2.SYSTEMS for tk in B2.TRUTHS for m in NEW for h in range(B2.N_HISTORIES)]
    with Pool(2) as p:
        res = p.map(job2, jobs, chunksize=1)
    d2 = pd.concat(res, ignore_index=True)
    truth = pd.read_csv("results/benchmark_hl2_by_history.csv")[["system", "truth", "EENS_true", "LOLE_true", "SAIDI_true"]].drop_duplicates()
    d2 = d2.merge(truth, on=["system", "truth"])
    for k in ("EENS", "LOLE", "SAIDI"):
        d2[f"{k}_err_pct"] = 100 * (d2[k] - d2[f"{k}_true"]) / d2[f"{k}_true"]
    d2.to_csv("results/benchmark_hl2_extra.csv", index=False)
    print(f"HL-II extra done {time.time() - t0:.0f}s", flush=True)
    print(d2.groupby(["system", "truth", "method"]).EENS_err_pct.apply(lambda x: np.abs(x).mean()).round(1).unstack().to_string(), flush=True)
