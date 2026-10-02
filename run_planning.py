"""
run_planning.py — storage siting/sizing experiments.

1. Scenario collection (HL-II bus-level curtailment):
     train_prop  proposed model,     years 10000-10099
     train_conv  conventional model, years 30000-30099
     test_prop   proposed model,     years 20000-20299  (out of sample)
     test_conv   conventional model, years 40000-40299  (out of sample)
2. Optimiser comparison on train_prop: 5 algorithms x SEEDS seeds, same budget.
Converged plans for each model, out-of-sample and full-network evaluation:
see run_plans_final.py.
Outputs in results/planning_<SYS>_*.{csv,npz}
"""
import os
import sys
import time
import json
import numpy as np
import pandas as pd
from multiprocessing import Pool

import config as C
import model as M
import planning as PL
import optimizers as O

NP, G = 30, 200
SEEDS = 25
SETS = {"train_prop": ("proposed", range(10000, 10100)),
        "train_conv": ("conventional", range(30000, 30100)),
        "test_prop": ("proposed", range(20000, 20300)),
        "test_conv": ("conventional", range(40000, 40300))}
VALID_YEARS = 100
OUT = "results"
_S = {}


def _collect(args):
    name, variant, years = args
    if name not in _S:
        s = M.build_system(name)
        _S[name] = (s, M.Network(s))
    s, net = _S[name]
    return PL.collect(name, variant, years, s, net)


def collect_all(name):
    path = f"{OUT}/planning_{name}_scenarios.npz"
    if os.path.exists(path):
        z = np.load(path)
        return {k: {q: z[f"{k}__{q}"] for q in ("year", "hour", "bus", "mw")} | {"n_years": int(z[f"{k}__n"])}
                for k in SETS}
    data, flat = {}, {}
    for k, (variant, years) in SETS.items():
        yl = list(years)
        chunks = [(name, variant, yl[i::8]) for i in range(8)]
        with Pool(2) as p:
            parts = p.map(_collect, chunks)
        d = {q: np.concatenate([pt[q] for pt in parts]) for q in ("year", "hour", "bus", "mw")}
        d["n_years"] = len(yl)
        data[k] = d
        for q in ("year", "hour", "bus", "mw"):
            flat[f"{k}__{q}"] = d[q]
        flat[f"{k}__n"] = len(yl)
    np.savez(path, **flat)
    return data


def _opt(args):
    name, alg, seed, setname = args
    sc = _S["sc_" + name + setname]
    lo, hi = np.zeros(len(sc.candidates)), sc.pmax * 2.0
    t0 = time.time()
    x, fbest, hist = O.ALGORITHMS[alg](sc.objective, lo, hi, NP, G, seed)
    return dict(system=name, algorithm=alg, seed=seed, train=setname, J=float(fbest),
                cpu_s=time.time() - t0, x=x.tolist(), hist=hist.tolist())


def main(name):
    t0 = time.time()
    data = collect_all(name)
    cand = np.unique(np.concatenate([data["train_prop"]["bus"], data["train_conv"]["bus"]]))
    sc = {k: PL.Scenarios(data[k], cand) for k in SETS}
    print(f"{name}: scenarios ready ({time.time() - t0:.0f}s), {len(cand)} candidate buses, "
          f"curtailment records/yr: " + ", ".join(f"{k} {len(sc[k].year) / sc[k].n_years:.1f}" for k in sc), flush=True)
    for k in ("train_prop", "train_conv"):
        _S["sc_" + name + k] = sc[k]
    # bounds common to both training sets
    pm = np.maximum(sc["train_prop"].pmax, sc["train_conv"].pmax)
    sc["train_prop"].pmax = sc["train_conv"].pmax = pm
    sc["train_prop"].objective(np.zeros((2, len(cand))))       # numba compile

    # ---- 2. optimiser comparison (proposed-model training set)
    jobs = [(name, a, sd, "train_prop") for a in O.ALGORITHMS for sd in range(SEEDS)]
    with Pool(2) as p:
        res = p.map(_opt, jobs)
    pd.DataFrame([{k: v for k, v in r.items() if k not in ("x", "hist")} for r in res]).to_csv(
        f"{OUT}/planning_{name}_runs.csv", index=False)
    np.savez(f"{OUT}/planning_{name}_hist.npz",
             **{f"{r['algorithm']}|{r['train']}|{r['seed']}": np.array(r["hist"]) for r in res})
    print(f"{name}: optimisation done ({time.time() - t0:.0f}s)", flush=True)

    print(f"{name}: done ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    for n in (sys.argv[1].split(",") if len(sys.argv) > 1 else C.SYSTEMS):
        main(n)
