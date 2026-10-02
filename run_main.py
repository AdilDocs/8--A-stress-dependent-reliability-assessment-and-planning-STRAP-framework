"""
run_main.py — main comparison: every model variant, every system, YEARS_MAIN
simulated years, HL-I and HL-II indices.  Variants of one year share the same
random thresholds and repair draws (common random numbers).

Output: results/main_<SYS>.csv  (one row per year x variant)
        results/components_<SYS>.csv (per component, proposed & conventional, summed over years)
"""
import os
import sys
import time
import numpy as np
import pandas as pd
from multiprocessing import Pool

import config as C
import model as M

OUT = "results"
os.makedirs(OUT, exist_ok=True)
_SYS, _NET = {}, {}


def get(name):
    if name not in _SYS:
        _SYS[name] = M.build_system(name)
        _NET[name] = M.Network(_SYS[name])
    return _SYS[name], _NET[name]


def job(args):
    name, year = args
    s, net = get(name)
    load = s["peak"] * s["profile"]
    top = load >= np.quantile(load, 0.95)                 # top 5 % load hours
    rows, comp = [], {}
    for v in M.VARIANTS:
        S, fails = M.simulate(s, v, year, want_counts=True)
        d1 = M.hl1_deficit(s, S)
        shed = net.evaluate(S)
        d2 = shed.sum(0).astype(float)
        row = dict(system=name, year=year, variant=v)
        row.update({f"HL1_{k}": x for k, x in M.indices(d1, load).items()})
        row.update({f"HL2_{k}": x for k, x in M.indices(d2, load).items()})
        row.update(M.customer_indices(shed))
        down_cap = s["u_cap"][:, None] * (1 - S[:s["nu"]])
        row["cap_out_mean_MW"] = float(down_cap.sum(0).mean())
        row["cap_out_peak5_MW"] = float(down_cap.sum(0)[top].mean())
        row["unit_failures"] = int(fails[:s["nu"]].sum())
        row["branch_failures"] = int(fails[s["nu"]:].sum())
        row["P95_hour_deficit"] = float(d2.max())
        rows.append(row)
        if v in ("proposed", "conventional"):
            down = 1.0 - S.astype(float)
            nd = np.maximum(1.0, down.sum(0))
            comp[v] = dict(failures=fails, down_h=down.sum(1),
                           eens_share=(down * (d2 / nd)[None, :]).sum(1),
                           lol_share=(down * ((d2 > C.LOL_TOL_MW) / nd)[None, :]).sum(1))
    return rows, comp


def main(years=C.YEARS_MAIN, procs=2, systems=C.SYSTEMS):
    for name in systems:
        t0 = time.time()
        with Pool(procs) as p:
            res = p.map(job, [(name, y) for y in range(years)], chunksize=4)
        pd.DataFrame([r for rr, _ in res for r in rr]).to_csv(f"{OUT}/main_{name}.csv", index=False)
        s, _ = get(name)
        rows = []
        for v in ("proposed", "conventional"):
            agg = {k: sum(c[v][k] for _, c in res) / years for k in res[0][1][v]}
            for i in range(s["N"]):
                rows.append(dict(variant=v, id=s["ids"][i], cls=s["cls"][i],
                                 cap=float(s["u_cap"][i]) if i < s["nu"] else np.nan,
                                 mttf=s["mttf"][i], mttr=s["mttr"][i], e=s["e"][i],
                                 **{k: float(agg[k][i]) for k in agg}))
        pd.DataFrame(rows).to_csv(f"{OUT}/components_{name}.csv", index=False)
        print(f"{name}: {years} years in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    y = int(sys.argv[1]) if len(sys.argv) > 1 else C.YEARS_MAIN
    sy = sys.argv[2].split(",") if len(sys.argv) > 2 else C.SYSTEMS
    main(y, systems=sy)
