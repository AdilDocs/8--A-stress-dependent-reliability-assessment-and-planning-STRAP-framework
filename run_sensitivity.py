"""
run_sensitivity.py — HL-I sensitivity of the proposed vs conventional model.
Every point recalibrates the model (peak load, beta) so that the
conventional model meets the LOLE target and both models keep equal MTTF/MTTR.

Sweeps: LOLE target (reserve level), coupling exponent kappa, repair share gamma,
alpha fraction, epsilon.  Systems: RTS79, IEEE39, IEEE118 (generation only).
Output: results/sensitivity.csv (one row per system x sweep x value x year x variant)
"""
import sys
import time
import numpy as np
import pandas as pd
from multiprocessing import Pool

import config as C
import model as M

YEARS = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
SYSTEMS = ["RTS79", "IEEE39", "IEEE118"]
VARS = ["conventional", "stress_failure", "proposed"]
SWEEPS = {
    "lole_target": [1.0, 2.4, 5.0, 9.39418, 20.0, 50.0],
    "kappa": [0.0, 0.5, 1.0, 1.5, 2.0],
    "gamma_fraction": [0.25, 0.5, 0.75, 1.0],
    "alpha_fraction": [0.05, 0.1, 0.2, 0.4],
    "eps": [0.01, 0.05, 0.2],
}
DEFAULT = dict(lole_target=C.LOLE_TARGET_H, kappa=C.COUPLING_EXP, gamma_fraction=C.GAMMA_FRACTION,
               alpha_fraction=C.ALPHA_FRACTION, eps=C.TTF_EPS)


def units_view(s):
    u = dict(s)
    n = s["nu"]
    for k in ("cls", "mttf", "mttr", "e", "a", "b", "hbar", "fors", "H", "sigma", "sig_w", "sig_mean", "Hc_stress"):
        u[k] = s[k][:n]
    u["fac"] = {q: v[:n] for q, v in s["fac"].items()}
    u["N"] = n
    return u


_S = {}


def job(args):
    key, year = args
    s = _S[key]
    load = s["peak"] * s["profile"]
    out = []
    for v in VARS:
        S = M.simulate(s, v, year)
        idx = M.indices(M.hl1_deficit(s, S), load)
        out.append(dict(variant=v, year=year, LOLE=idx["LOLE"], EENS=idx["EENS"], LOLF=idx["LOLF"]))
    return out


def main():
    rows = []
    for name in SYSTEMS:
        for sweep, values in SWEEPS.items():
            for val in values:
                p = dict(DEFAULT)
                p[sweep] = val
                t0 = time.time()
                C.GAMMA_FRACTION = p["gamma_fraction"]
                s = M.build_system(name, kappa=p["kappa"], lole_target=p["lole_target"],
                                   alpha_fraction=p["alpha_fraction"], eps=p["eps"])
                key = (name, sweep, val)
                _S.clear()
                _S[key] = units_view(s)
                with Pool(2) as pool:
                    res = pool.map(job, [(key, y) for y in range(YEARS)], chunksize=25)
                for rr in res:
                    for r in rr:
                        rows.append(dict(system=name, sweep=sweep, value=val, peak=s["peak"],
                                         reserve_margin=s["reserve_margin"], **r))
                C.ALPHA_FRACTION, C.TTF_EPS = DEFAULT["alpha_fraction"], DEFAULT["eps"]
                C.GAMMA_FRACTION = DEFAULT["gamma_fraction"]
                df = pd.DataFrame(rows[-YEARS * len(VARS):])
                m = df.groupby("variant")["LOLE"].mean()
                print(f"{name} {sweep}={val}: LOLE conv {m['conventional']:.2f} fail {m['stress_failure']:.2f} "
                      f"prop {m['proposed']:.2f}  ({time.time() - t0:.0f}s)", flush=True)
        pd.DataFrame(rows).to_csv("results/sensitivity.csv", index=False)


if __name__ == "__main__":
    main()
