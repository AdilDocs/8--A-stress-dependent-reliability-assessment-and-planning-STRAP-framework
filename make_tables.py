"""
make_tables.py — every table of the revised paper from results/ (CSV + LaTeX).
  table1_systems      test systems after calibration
  table2_validation   RTS-79 engine validation
  table3_main         HL-I / HL-II indices, all model variants, mean +- 95 % CI
  table4_effects      paired change vs conventional (%), 95 % CI
  table5_optimizers   optimiser comparison, Wilcoxon signed-rank vs jDE
  table6_planning     out-of-sample planning results
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, mannwhitneyu

import config as C
import model as M

R = "results"
T = f"{R}/tables"
os.makedirs(T, exist_ok=True)
LAB = {"IEEE39": "IEEE 39", "IEEE57": "IEEE 57", "IEEE118": "IEEE 118"}
VLAB = {"conventional": "Conventional", "stress_failure": "Stress-dep. failure only",
        "stress_repair": "Stress-dep. repair only", "proposed": "Proposed"}


def ci(x):
    x = np.asarray(x, float)
    return x.mean(), 1.96 * x.std(ddof=1) / np.sqrt(len(x))


def pm(m, h, d=2):
    return f"{m:.{d}f} $\\pm$ {h:.{d}f}"


WIDE = {"table1_systems", "table3_main", "table4_effects", "table5_optimizers", "table6_planning"}


def tex(name, caption, label, header, rows, align):
    with open(f"{T}/{name}.tex", "w") as f:
        f.write("\\begin{" + ("table*" if name in WIDE else "table") + "}[t]\n\\centering\\scriptsize\n")
        f.write(f"\\caption{{{caption}}}\n\\label{{{label}}}\n")
        f.write(f"\\resizebox{{\\linewidth}}{{!}}{{%\n\\begin{{tabular}}{{{align}}}\n\\hline\n{header} \\\\ \\hline\n")
        f.write("\n".join(r + " \\\\" for r in rows))
        f.write("\n\\hline\n\\end{tabular}}\n\\end{" + ("table*" if name in WIDE else "table") + "}\n")


# ---------------------------------------------------------------- Table 1
rows, trs = [], []
for n in C.SYSTEMS:
    s = M.build_system(n)
    r = dict(System=n, Buses=s["nb"], Units=s["nu"], Lines=int((s["cls"] == "line").sum()),
             Transformers=int((s["cls"] == "transformer").sum()), Capacity_MW=s["u_cap"].sum(),
             Peak_MW=s["peak"], Reserve_margin_pct=100 * s["reserve_margin"], Rating_scale=s["rating_scale"],
             Analytic_HL1_LOLE=s["analytic_hl1"]["LOLE"], Analytic_HL1_EENS=s["analytic_hl1"]["EENS"])
    rows.append(r)
    trs.append(f"{LAB[n]} & {r['Buses']} & {r['Units']} & {r['Lines']} & {r['Transformers']} & "
               f"{r['Capacity_MW']:.0f} & {r['Peak_MW']:.0f} & {r['Reserve_margin_pct']:.1f} & {r['Rating_scale']:.2f}")
pd.DataFrame(rows).to_csv(f"{T}/table1_systems.csv", index=False)
tex("table1_systems", "Test systems (PGLib-OPF v23.07) after calibration to the IEEE RTS-79 HL-I LOLE of 9.394 h/yr",
    "tab:systems", "System & Buses & Units & Lines & Transf. & Cap. (MW) & Peak (MW) & RM (\\%) & Rating scale",
    trs, "|l|c|c|c|c|c|c|c|c|")

# ---------------------------------------------------------------- Table 2
if os.path.exists(f"{R}/validation.csv"):
    v = pd.read_csv(f"{R}/validation.csv")
    trs = []
    for _, r in v.iterrows():
        l = f"{r.LOLE:.3f}" + (f" $\\pm$ {r.LOLE_CI95:.3f}" if r.LOLE_CI95 == r.LOLE_CI95 else "")
        e = f"{r.EENS:.1f}" + (f" $\\pm$ {r.EENS_CI95:.1f}" if r.EENS_CI95 == r.EENS_CI95 else "")
        trs.append(f"{r.method} & {l} & {e}")
    v.to_csv(f"{T}/table2_validation.csv", index=False)
    tex("table2_validation", "Engine validation on the IEEE RTS-79 generating system (hourly load model)",
        "tab:validation", "Method & LOLE (h/yr) & EENS (MWh/yr)", trs, "|l|c|c|")

# ---------------------------------------------------------------- Tables 3 / 4
main = {n: pd.read_csv(f"{R}/main_{n}.csv") for n in C.SYSTEMS if os.path.exists(f"{R}/main_{n}.csv")}
rows3, trs3, rows4, trs4 = [], [], [], []
mets = [("HL1_LOLE", 2), ("HL1_EENS", 0), ("HL2_LOLE", 2), ("HL2_EENS", 0), ("SAIFI", 3), ("SAIDI", 2),
        ("cap_out_mean_MW", 1), ("cap_out_peak5_MW", 1)]
for n, d in main.items():
    Y = d.year.nunique()
    for v in VLAB:
        g = d[d.variant == v]
        r = dict(System=n, Variant=v, Years=Y)
        cells = []
        for m, dec in mets:
            mu, h = ci(g[m])
            r[m], r[m + "_CI95"] = mu, h
            if m in ("HL1_LOLE", "HL1_EENS", "HL2_LOLE", "HL2_EENS", "SAIFI", "SAIDI"):
                cells.append(pm(mu, h, dec))
        r["HL2_EENS_P95"] = float(np.percentile(g.HL2_EENS, 95))
        rows3.append(r)
        trs3.append(f"{LAB[n]} & {VLAB[v]} & " + " & ".join(cells))
        if v != "conventional":
            r4 = dict(System=n, Variant=v)
            c4 = []
            for m in ("HL1_LOLE", "HL2_LOLE", "HL2_EENS", "cap_out_peak5_MW"):
                pp = d.pivot(index="year", columns="variant", values=m)
                diff = pp[v] - pp["conventional"]
                base = pp["conventional"].mean()
                mu, h = 100 * diff.mean() / base, 100 * 1.96 * diff.std(ddof=1) / np.sqrt(len(diff)) / base
                try:
                    pval = wilcoxon(pp[v], pp["conventional"]).pvalue
                except ValueError:
                    pval = 1.0
                r4[m + "_pct"], r4[m + "_CI95"], r4[m + "_p"] = mu, h, pval
                star = "$^{*}$" if pval < 0.01 else ""
                c4.append(f"{mu:+.1f} $\\pm$ {h:.1f}{star}")
            rows4.append(r4)
            trs4.append(f"{LAB[n]} & {VLAB[v]} & " + " & ".join(c4))
pd.DataFrame(rows3).to_csv(f"{T}/table3_main.csv", index=False)
pd.DataFrame(rows4).to_csv(f"{T}/table4_effects.csv", index=False)
tex("table3_main", f"Adequacy indices by model variant (mean $\\pm$ 95\\% CI, {Y} years per variant); "
    "all variants have identical component MTTF and MTTR", "tab:main",
    "System & Model & HL-I LOLE (h) & HL-I EENS (MWh) & HL-II LOLE (h) & HL-II EENS (MWh) & SAIFI & SAIDI (h)",
    trs3, "|l|l|c|c|c|c|c|c|")
tex("table4_effects", "Change relative to the conventional model (\\%, paired years, 95\\% CI). "
    "$^{*}$: Wilcoxon signed-rank $p<0.01$", "tab:effects",
    "System & Model & HL-I LOLE & HL-II LOLE & HL-II EENS & Cap. out at peak", trs4, "|l|l|c|c|c|c|")

# ---------------------------------------------------------------- Table 5
rows5, trs5 = [], []
for n in C.SYSTEMS:
    f = f"{R}/planning_{n}_runs.csv"
    if not os.path.exists(f):
        continue
    d = pd.read_csv(f)
    tr = d[d.train == "train_prop"]
    ref = tr[tr.algorithm == "jDE"].sort_values("seed").J.values
    jbest = tr.J.min()
    for alg, g in tr.groupby("algorithm", sort=False):
        J = g.sort_values("seed").J.values
        if alg == "jDE":
            p, sign = np.nan, ""
        else:
            p = wilcoxon(ref, J).pvalue if not np.allclose(ref, J) else 1.0
            sign = "+" if np.median(J - ref) > 0 else "-"
        r = dict(System=n, Algorithm=alg, Seeds=len(J), J_mean_M=J.mean() / 1e6, J_std_M=J.std(ddof=1) / 1e6,
                 J_best_M=J.min() / 1e6, J_worst_M=J.max() / 1e6, gap_mean_pct=100 * (J.mean() - jbest) / jbest,
                 hit_rate_0p1pct=float(np.mean(J <= jbest * 1.001)), cpu_s=g.cpu_s.mean(), p_vs_jDE=p,
                 jDE_better=(sign == "+"))
        rows5.append(r)
        ptxt = "--" if np.isnan(p) else (f"{p:.1e}" + (" (jDE better)" if sign == "+" else " (jDE worse)"))
        trs5.append(f"{LAB[n]} & {alg} & {r['J_mean_M']:.3f} $\\pm$ {r['J_std_M']:.3f} & {r['J_best_M']:.3f} & "
                    f"{r['gap_mean_pct']:.3f} & {100 * r['hit_rate_0p1pct']:.0f} & "
                    f"{'--' if r['cpu_s'] != r['cpu_s'] else format(r['cpu_s'], '.1f')} & {ptxt}")
if rows5:
    pd.DataFrame(rows5).to_csv(f"{T}/table5_optimizers.csv", index=False)
    tex("table5_optimizers", "Storage-planning optimiser comparison (25 seeds, 6030 evaluations each; "
        "$J$ in M\\$/yr; hit rate: runs within 0.1\\% of the best plan found)", "tab:optimizers",
        "System & Algorithm & $J$ mean $\\pm$ sd & $J$ best & Mean gap (\\%) & Hit (\\%) & CPU (s) & Wilcoxon $p$ vs jDE",
        trs5, "|l|l|c|c|c|c|c|l|")

# ---------------------------------------------------------------- Table 6
rows6, trs6 = [], []
for n in C.SYSTEMS:
    f = f"{R}/planning_{n}_plans.csv"
    if not os.path.exists(f):
        continue
    d = pd.read_csv(f)
    for pl in ("No storage", "Heuristic (EENS-proportional, same MW)", "Conventional-model plan",
               "Stress-aware plan (proposed model)"):
        g = d[d.plan == pl].set_index("set")
        tp, tc = g.loc["test_prop"], g.loc["test_conv"]
        r = dict(System=n, Plan=pl, MW=tp.MW, EENS_test_prop=tp.EENS, LOLE_test_prop=tp.LOLE, J_test_prop_M=tp.J / 1e6,
                 EENS_test_conv=tc.EENS, LOLE_test_conv=tc.LOLE, J_test_conv_M=tc.J / 1e6)
        vn = [k for k in g.index if k.startswith("validation_network")]
        if vn:
            r["EENS_network_validation"] = g.loc[vn[0]].EENS
            r["LOLE_network_validation"] = g.loc[vn[0]].LOLE
        rows6.append(r)
        trs6.append(f"{LAB[n]} & {pl} & {tp.MW:.0f} & {tp.EENS:.0f} & {tp.LOLE:.2f} & {tp.J / 1e6:.2f} & "
                    f"{tc.EENS:.0f} & {tc.LOLE:.2f} & {tc.J / 1e6:.2f}")
if rows6:
    pd.DataFrame(rows6).to_csv(f"{T}/table6_planning.csv", index=False)
    tex("table6_planning", "Out-of-sample performance of storage plans (300 test years per model; $J$ in M\\$/yr)",
        "tab:planning", "System & Plan & MW & \\multicolumn{3}{c|}{Proposed-model test} & "
        "\\multicolumn{3}{c|}{Conventional-model test} \\\\ & & & EENS & LOLE & $J$ & EENS & LOLE & $J$",
        trs6, "|l|l|c|c|c|c|c|c|c|")

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
for fn in sorted(os.listdir(T)):
    if fn.endswith(".csv"):
        print(f"\n== {fn}\n", pd.read_csv(f"{T}/{fn}").round(3).to_string(index=False))
