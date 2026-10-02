"""
make_comparison_table.py — rebuilds Table "Comparison with existing methods"
(tab:compare) from the benchmark result files.
  HL-I : mean absolute LOLE error (%) over RTS-79, IEEE 39, IEEE 118 for T0-T3
  HL-II: mean absolute EENS error (%); T0 averaged over the three systems, T1 per system
  CPU  : s per simulated year, HL-II, IEEE 118, T1
Inputs : results/benchmark_hl1.csv, benchmark_hl1_extra.csv,
         results/benchmark_hl2.csv, benchmark_hl2_extra.csv
Output : results/tables/tab_compare.csv
"""
import numpy as np
import pandas as pd

R = "results"
h1 = pd.concat([pd.read_csv(f"{R}/benchmark_hl1.csv"), pd.read_csv(f"{R}/benchmark_hl1_extra.csv")])
mape1 = h1.groupby(["method", "system", "truth"]).LOLE_err_pct.apply(lambda x: np.abs(x).mean()).reset_index()
hl1 = mape1.groupby(["method", "truth"]).LOLE_err_pct.mean().unstack()

b2 = pd.read_csv(f"{R}/benchmark_hl2.csv")[["system", "truth", "method", "EENS_abs_err_pct", "cpu_s_per_year"]]
e2 = pd.read_csv(f"{R}/benchmark_hl2_extra.csv")
e2 = e2.groupby(["system", "truth", "method"]).agg(
    EENS_abs_err_pct=("EENS_err_pct", lambda x: np.abs(x).mean()),
    cpu_s_per_year=("cpu_s_per_year", "mean")).reset_index()
h2 = pd.concat([b2, e2])
name2 = {"Sequential two-state MCS": "COPT", "Seasonal-rate MCS": "Seasonal", "Two-level MCS": "Two-level"}
h2["method"] = h2.method.replace(name2)

rows = []
order = ["COPT", "Non-sequential MCS", "Weibull", "Hour-of-day", "Seasonal", "Weather", "Two-level",
         "Load-dependent", "Proposed"]
for m in order:
    r = {"method": m}
    if m in hl1.index:
        r.update({f"HL1_{t}": round(hl1.loc[m, t], 1) for t in ["T0", "T1", "T2", "T3"]})
    g = h2[h2.method == m]
    if len(g):
        r["HL2_T0_mean"] = round(g[g.truth == "T0"].EENS_abs_err_pct.mean(), 1)
        for s in ["IEEE39", "IEEE57", "IEEE118"]:
            r[f"HL2_T1_{s}"] = round(g[(g.truth == "T1") & (g.system == s)].EENS_abs_err_pct.mean(), 1)
        r["CPU_118_T1"] = round(g[(g.truth == "T1") & (g.system == "IEEE118")].cpu_s_per_year.mean(), 3)
    rows.append(r)
out = pd.DataFrame(rows)
out.to_csv(f"{R}/tables/tab_compare.csv", index=False)
print(out.to_string(index=False))
print("Note: 'COPT' at HL-II is the sequential two-state MCS (same constant-rate model).")
