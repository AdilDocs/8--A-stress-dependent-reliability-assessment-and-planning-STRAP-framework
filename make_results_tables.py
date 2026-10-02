"""
make_results_tables.py — additional tables for Section V (Results and
Discussion).  Run after make_tables.py.  Writes results/tables/T*.tex and .csv.

  T01_parameters       all model / study parameters
  T07_mechanism        failure frequency and capacity on outage by load level
  T08_seasonal         HL-II EENS by season, proposed vs conventional
  T09_components       most critical components, proposed vs conventional
  T10_loadpoints       load-point indices of the five most affected buses
  T11_method_features  capability matrix of the compared methods
  T12_bench_hl1        known-truth accuracy of fitted methods (HL-I)
  T13_bench_hl2        fitted methods at HL-II vs true indices
  T14_sensitivity      LOLE ratio proposed / conventional over all sweeps
  T15_computation      network-evaluation speed and Monte Carlo convergence
  T16_opt_settings     optimiser settings
  T17_siting           storage siting of the final plans
"""
import os
import json
import numpy as np
import pandas as pd

import config as C
import data_sources as D
import model as M

R = "results"
T = f"{R}/tables"
os.makedirs(T, exist_ok=True)
SL = {"IEEE39": "IEEE 39", "IEEE57": "IEEE 57", "IEEE118": "IEEE 118", "RTS79": "RTS-79"}
VL = {"conventional": "Conventional", "stress_failure": "Stress-dep. failure", "stress_repair": "Stress-dep. repair",
      "proposed": "Proposed"}
TL = {"T0": "Constant rates", "T1": "Stress-dep., $\\kappa=1$", "T2": "Stress-dep., $\\kappa=2$",
      "T3": "Exp.-in-load (mis-spec.)"}


def tex(name, caption, label, header, rows, align, star=False, note=None):
    env = "table*" if star else "table"
    with open(f"{T}/{name}.tex", "w") as f:
        f.write(f"\\begin{{{env}}}[t]\n\\centering\\scriptsize\n\\renewcommand{{\\arraystretch}}{{1.15}}\n")
        f.write(f"\\caption{{{caption}}}\n\\label{{{label}}}\n")
        f.write(f"\\resizebox{{\\linewidth}}{{!}}{{%\n\\begin{{tabular}}{{{align}}}\n\\hline\n{header} \\\\ \\hline\n")
        for r in rows:
            f.write(r if r.endswith("\\hline") else r + " \\\\")
            f.write("\n")
        f.write("\\hline\n\\end{tabular}}\n")
        if note:
            f.write(f"\\\\[2pt]\\parbox{{0.95\\linewidth}}{{\\footnotesize {note}}}\n")
        f.write(f"\\end{{{env}}}\n")


def ci(x):
    x = np.asarray(x, float)
    return x.mean(), 1.96 * x.std(ddof=1) / np.sqrt(len(x))


def bold(s, on):
    return f"\\textbf{{{s}}}" if on else s


# ------------------------------------------------------------------ T01 parameters
rows = [
    ("Chronological load", "IEEE RTS-79 hourly model, 8736 h, load factor 0.614"),
    ("Generating units", "PGLib generators split into units $\\le$ 400 MW; RTS-79 MTTF/MTTR of nearest class"),
    ("Unit MTTF / MTTR (h)", "; ".join(f"{k} MW: {v[0]}/{v[1]}" for k, v in D.RTS_UNITS.items())),
    ("Lines", f"$\\lambda$ = {D.LINE_DATA['lam_per_yr']} /yr, MTTR = {D.LINE_DATA['mttr_h']:.0f} h"),
    ("Transformers", f"$\\lambda$ = {D.TRAFO_DATA['lam_per_yr']} /yr, MTTR = {D.TRAFO_DATA['mttr_h']:.0f} h"),
    ("Peak calibration", f"conventional HL-I LOLE = {C.LOLE_TARGET_H:.3f} h/yr (RTS-79)"),
]
for c, rg in C.STRESS_RANGES.items():
    qn = {"eta_L": "$\\eta_L$", "T": "$T$ ($^\\circ$C)", "eta_A": "$\\eta_A$", "eta_V": "$\\eta_V$"}
    rows.append((f"Stress factors, {c}", ", ".join(f"{qn[q]}: [{a}, {b}]" for q, (a, b) in rg.items())))
rows += [
    ("$T_{\\mathrm{ref}}$, $\\kappa$, $\\varepsilon$", f"{C.T_REF:.0f} $^\\circ$C, {C.COUPLING_EXP}, {C.TTF_EPS}"),
    ("$\\alpha$ / MTTF, $\\gamma$", f"{C.ALPHA_FRACTION}, {C.GAMMA_FRACTION}"),
    ("Simulated years", f"{C.YEARS_MAIN} per model per system (+{C.WARMUP_YEARS} warm-up year)"),
    ("Storage", f"{C.STORAGE_HOURS:.0f} h, $\\eta_c=\\eta_d$ = {C.ETA_CH}, capex {C.STORAGE_CAPEX / 1e6:.1f} M\\$/MW, CRF {C.CRF}"),
    ("VOLL, LOLE target, penalty", f"{C.VOLL / 1e3:.0f} k\\$/MWh, {C.LOLE_TARGET_PLAN} h/yr, {C.LOLE_PENALTY / 1e6:.0f} M\\$/h"),
]
pd.DataFrame(rows, columns=["Parameter", "Value"]).to_csv(f"{T}/T01_parameters.csv", index=False)
tex("T01_parameters", "Data and parameters of the study", "tab:params", "Parameter & Value",
    [f"{a} & {b}" for a, b in rows], "|l|p{0.62\\linewidth}|", star=True)

# ------------------------------------------------------------------ T07 mechanism
if os.path.exists(f"{R}/mechanism.csv"):
    me = pd.read_csv(f"{R}/mechanism.csv")
    rows_csv, trs = [], []
    for sname in ("RTS79", "IEEE39", "IEEE118"):
        for v in ("conventional", "stress_failure", "stress_repair", "proposed"):
            g = me[(me.system == sname) & (me.variant == v)].groupby("decile").mean(numeric_only=True)
            f1, f10 = g.fail_per_1000_up_h.loc[1], g.fail_per_1000_up_h.loc[10]
            c1, c10 = g.cap_out.loc[1], g.cap_out.loc[10]
            rows_csv.append(dict(system=sname, variant=v, fail_d1=f1, fail_d10=f10, fail_ratio=f10 / f1,
                                 capout_d1=c1, capout_d10=c10))
            trs.append(f"{SL[sname]} & {VL[v]} & {f1:.3f} & {f10:.3f} & {f10 / f1:.2f} & {c1:.0f} & {c10:.0f}")
        trs[-1] += " \\\\ \\hline"
    pd.DataFrame(rows_csv).to_csv(f"{T}/T07_mechanism.csv", index=False)
    tex("T07_mechanism", "Outage timing: unit failures per 1000 unit-hours in service and mean capacity on "
        "outage in the lightest (D1) and peak (D10) load deciles (500 years)", "tab:mechanism",
        "System & Model & Fail. D1 & Fail. D10 & D10/D1 & Cap. out D1 (MW) & Cap. out D10 (MW)",
        trs, "|l|l|c|c|c|c|c|")

# ------------------------------------------------------------------ T08 seasonal, T10 load points
def season_of(h):
    w = h // 168 + 1
    return np.where((w <= 8) | (w >= 44), "Winter", np.where((w >= 18) & (w <= 30), "Summer", "Spring/Fall"))


rows8, trs8, rows10, trs10 = [], [], [], []
for sname in C.SYSTEMS:
    f = f"{R}/planning_{sname}_scenarios.npz"
    if not os.path.exists(f):
        continue
    z = np.load(f)
    s = M.build_system(sname)
    res = {}
    for k in ("test_prop", "test_conv"):
        yr, hr, bus, mw = z[f"{k}__year"], z[f"{k}__hour"], z[f"{k}__bus"], z[f"{k}__mw"]
        n = int(z[f"{k}__n"])
        res[k] = (yr, hr, bus, mw, n)
    tot = {}
    for k, (yr, hr, bus, mw, n) in res.items():
        se = season_of(hr)
        tot[k] = {q: mw[se == q].sum() / n for q in ("Winter", "Spring/Fall", "Summer")}
    for q in ("Winter", "Spring/Fall", "Summer"):
        p_, c_ = tot["test_prop"][q], tot["test_conv"][q]
        rows8.append(dict(system=sname, season=q, EENS_proposed=p_, EENS_conventional=c_,
                          share_proposed=p_ / sum(tot["test_prop"].values()),
                          share_conventional=c_ / sum(tot["test_conv"].values())))
        trs8.append(f"{SL[sname]} & {q} & {c_:.0f} & {100 * rows8[-1]['share_conventional']:.1f} & "
                    f"{p_:.0f} & {100 * rows8[-1]['share_proposed']:.1f} & {100 * (p_ - c_) / c_:+.1f}")
    trs8[-1] += " \\\\ \\hline"
    # load points
    stats = {}
    for k, (yr, hr, bus, mw, n) in res.items():
        df = pd.DataFrame(dict(y=yr, h=hr, b=bus, mw=mw)).sort_values(["b", "y", "h"])
        e = df.groupby("b").mw.sum() / n
        hrs = df.groupby("b").size() / n
        df["new"] = (df.groupby(["b", "y"]).h.diff().fillna(99) > 1)
        fr = df.groupby("b").new.sum() / n
        stats[k] = pd.DataFrame(dict(E=e, H=hrs, F=fr))
    top = stats["test_prop"].sort_values("E", ascending=False).head(5).index
    for b in top:
        pr = stats["test_prop"].loc[b]
        cv = stats["test_conv"].loc[b] if b in stats["test_conv"].index else pd.Series(dict(E=0, H=0, F=0))
        busno = int(s["load_bus"][int(b)]) + 1
        rows10.append(dict(system=sname, bus=busno, EENS_conv=cv.E, EENS_prop=pr.E, freq_conv=cv.F, freq_prop=pr.F,
                           dur_conv=cv.H, dur_prop=pr.H))
        trs10.append(f"{SL[sname]} & {busno} & {cv.E:.0f} & {pr.E:.0f} & {cv.F:.3f} & {pr.F:.3f} & {cv.H:.2f} & {pr.H:.2f}")
    trs10[-1] += " \\\\ \\hline"
if rows8:
    pd.DataFrame(rows8).to_csv(f"{T}/T08_seasonal.csv", index=False)
    tex("T08_seasonal", "HL-II EENS by season (MWh/yr, 300 independent years per model)", "tab:seasonal",
        "System & Season & Conv. EENS & Conv. share (\\%) & Prop. EENS & Prop. share (\\%) & Change (\\%)",
        trs8, "|l|l|c|c|c|c|c|")
    pd.DataFrame(rows10).to_csv(f"{T}/T10_loadpoints.csv", index=False)
    tex("T10_loadpoints", "Load-point indices of the five buses with the highest EENS under the proposed model "
        "(per year; 300 independent years per model)", "tab:loadpoints",
        "System & Bus & EENS conv. & EENS prop. & Interr. conv. & Interr. prop. & Hours conv. & Hours prop.",
        trs10, "|l|c|c|c|c|c|c|c|")

# ------------------------------------------------------------------ T09 components
rows9, trs9 = [], []
for sname in C.SYSTEMS:
    f = f"{R}/components_{sname}.csv"
    if not os.path.exists(f):
        continue
    d = pd.read_csv(f)
    p = d[d.variant == "proposed"].set_index("id")
    c = d[d.variant == "conventional"].set_index("id")
    for cls in ("unit", "transformer", "line"):
        top = p[p.cls == cls].sort_values("eens_share", ascending=False).head(3 if cls == "unit" else 2)
        for cid, r in top.iterrows():
            cc = c.loc[cid]
            rows9.append(dict(system=sname, component=cid, cls=cls, cap=r.cap, failures_conv=cc.failures,
                              failures_prop=r.failures, U_conv=cc.down_h / 8736, U_prop=r.down_h / 8736,
                              EENS_conv=cc.eens_share, EENS_prop=r.eens_share))
            lab = cid.replace("_", "\\_")
            trs9.append(f"{SL[sname]} & {lab} & {cls} & {cc.failures:.2f} & {r.failures:.2f} & "
                        f"{cc.down_h / 8736:.4f} & {r.down_h / 8736:.4f} & {cc.eens_share:.0f} & {r.eens_share:.0f}")
    trs9[-1] += " \\\\ \\hline"
if rows9:
    pd.DataFrame(rows9).to_csv(f"{T}/T09_components.csv", index=False)
    tex("T09_components", "Most critical components: failures/yr, unavailability and outage-coincident HL-II EENS "
        "(MWh/yr) under the conventional and proposed models (1000 years). Unit IDs: U\\#@bus; "
        "branch IDs: L/T\\#:from-to", "tab:components",
        "System & Component & Class & Fail. conv. & Fail. prop. & $U$ conv. & $U$ prop. & EENS conv. & EENS prop.",
        trs9, "|l|l|l|c|c|c|c|c|c|", star=True)

# ------------------------------------------------------------------ T11 method features
feat = [
    ("Analytical COPT", "no", "no", "no", "no", "no", "yes", "very low"),
    ("Non-sequential MCS", "no", "no", "no", "yes", "no", "yes", "medium"),
    ("Sequential two-state MCS", "no", "no", "yes", "yes", "no", "yes", "low"),
    ("Seasonal-rate MCS/COPT", "monthly", "no", "yes", "yes", "no", "yes", "low"),
    ("Two-level (peak/off-peak)", "2 levels", "no", "yes", "yes", "no", "yes", "low"),
    ("Proposed", "hourly, continuous", "yes", "yes", "yes", "yes", "yes", "low"),
]
pd.DataFrame(feat, columns=["Method", "Time-varying failure", "Stress-dep. repair", "Freq./duration indices",
                            "Network (HL-II)", "Mean-preserving calibration", "Fitted from outage data",
                            "CPU"]).to_csv(f"{T}/T11_method_features.csv", index=False)
tex("T11_method_features", "Capabilities of the compared adequacy-assessment methods", "tab:features",
    "Method & Time-varying failure & Stress-dep. repair & Freq./dur. indices & HL-II & Mean-preserving & "
    "Fitted from data & CPU", [" & ".join(r) for r in feat], "|l|c|c|c|c|c|c|c|", star=True)

# ------------------------------------------------------------------ T12 known-truth HL-I
if os.path.exists(f"{R}/benchmark_hl1.csv"):
    b = pd.read_csv(f"{R}/benchmark_hl1.csv")
    meths = ["COPT", "Seasonal", "Two-level", "Proposed"]
    rows12, trs = [], []
    for sname in ("RTS79", "IEEE39", "IEEE118"):
        for tk in ("T0", "T1", "T2", "T3"):
            g = b[(b.system == sname) & (b.truth == tk)]
            cells, mae = [], {}
            for m in meths:
                x = g[g.method == m]
                mae[m] = (np.abs(x.LOLE_err_pct).mean(), np.abs(x.EENS_err_pct).mean())
            bestL = min(mae, key=lambda m: mae[m][0]); bestE = min(mae, key=lambda m: mae[m][1])
            for m in meths:
                x = g[g.method == m]
                bl, _ = ci(x.LOLE_err_pct); be, _ = ci(x.EENS_err_pct)
                rows12.append(dict(system=sname, truth=tk, method=m, LOLE_true=x.LOLE_true.iloc[0],
                                   EENS_true=x.EENS_true.iloc[0], LOLE_bias_pct=bl, EENS_bias_pct=be,
                                   LOLE_MAPE=mae[m][0], EENS_MAPE=mae[m][1],
                                   kappa_hat_mean=x.kappa_hat.mean(), kappa_hat_sd=x.kappa_hat.std(),
                                   gamma_hat_mean=x.gamma_hat.mean(), cpu_s=x.cpu_s.mean()))
                cells.append(bold(f"{mae[m][0]:.1f}", m == bestL) + " / " + bold(f"{mae[m][1]:.1f}", m == bestE))
            kh = g[g.method == "Proposed"]
            trs.append(f"{SL[sname]} & {TL[tk]} & {g.LOLE_true.iloc[0]:.2f} & " + " & ".join(cells) +
                       f" & {kh.kappa_hat.mean():.2f} $\\pm$ {kh.kappa_hat.std():.2f}")
        trs[-1] += " \\\\ \\hline"
    pd.DataFrame(rows12).to_csv(f"{T}/T12_bench_hl1.csv", index=False)
    nrep = b.rep.nunique()
    tex("T12_bench_hl1", f"Known-truth benchmark: mean absolute error (\\%) of LOLE / EENS predicted by methods "
        f"fitted to a 10-year outage history ({nrep} independent histories per case; bold = best)",
        "tab:bench_hl1",
        "System & True process & LOLE$_{\\mathrm{true}}$ (h) & COPT & Seasonal & Two-level & Proposed & "
        "$\\hat\\kappa$ (proposed)", trs, "|l|l|c|c|c|c|c|c|", star=True)

# ------------------------------------------------------------------ T13 HL-II benchmark
if os.path.exists(f"{R}/benchmark_hl2.csv"):
    b2 = pd.read_csv(f"{R}/benchmark_hl2.csv")
    nh = int(pd.read_csv(f"{R}/benchmark_hl2_by_history.csv").history.nunique())
    order = ["Non-sequential MCS", "Sequential two-state MCS", "Seasonal-rate MCS", "Two-level MCS", "Proposed"]
    trs = []
    for sname in C.SYSTEMS:
        for tk in ("T0", "T1"):
            g = b2[(b2.system == sname) & (b2.truth == tk)].set_index("method")
            bestE = g.EENS_abs_err_pct.idxmin(); bestL = g.LOLE_abs_err_pct.idxmin(); bestS = g.SAIDI_abs_err_pct.idxmin()
            t = g.iloc[0]
            trs.append(f"{SL[sname]} & {TL[tk]} & True (1000 yr) & {t.EENS_true:.0f} & {t.LOLE_true:.1f} & "
                       f"{t.LOLF_true:.2f} & {t.SAIDI_true:.2f} & -- & -- & -- & --")
            for m in order:
                r = g.loc[m]
                lf = "--" if np.isnan(r.LOLF) else f"{r.LOLF:.2f}"
                trs.append(f" & & {m} & {r.EENS:.0f} & {r.LOLE:.1f} & {lf} & {r.SAIDI:.2f} & "
                           + bold(f"{r.EENS_abs_err_pct:.1f}", m == bestE) + " & "
                           + bold(f"{r.LOLE_abs_err_pct:.1f}", m == bestL) + " & "
                           + bold(f"{r.SAIDI_abs_err_pct:.1f}", m == bestS) + f" & {r.cpu_s_per_year:.3f}")
            trs[-1] += " \\\\ \\hline"
    b2.to_csv(f"{T}/T13_bench_hl2.csv", index=False)
    tex("T13_bench_hl2", f"HL-II benchmark: indices predicted by methods fitted to 20-year outage histories "
        f"(mean over {nh} independent histories) and mean absolute error (\\%) against the true indices; "
        "CPU in s per simulated year; -- = not available; bold = most accurate",
        "tab:bench_hl2",
        "System & True process & Method & EENS (MWh) & LOLE (h) & LOLF (occ.) & SAIDI (h) & "
        "EENS err. & LOLE err. & SAIDI err. & CPU", trs, "|l|l|l|c|c|c|c|c|c|c|c|", star=True)

# ------------------------------------------------------------------ T14 sensitivity
if os.path.exists(f"{R}/sensitivity.csv"):
    sd = pd.read_csv(f"{R}/sensitivity.csv")
    lab = {"lole_target": "LOLE target (h/yr)", "kappa": "Coupling exponent $\\kappa$",
           "gamma_fraction": "Fixed repair share $\\gamma$", "alpha_fraction": "$\\alpha$/MTTF", "eps": "$\\varepsilon$"}
    trs, rows14 = [], []
    for sw in lab:
        for v in sorted(sd[sd.sweep == sw].value.unique()):
            cells = []
            for sname in ("RTS79", "IEEE39", "IEEE118"):
                g = sd[(sd.system == sname) & (sd.sweep == sw) & (sd.value == v)]
                p = g.pivot(index="year", columns="variant", values="LOLE")
                rr = p.proposed.mean() / p.conventional.mean()
                rf = p.stress_failure.mean() / p.conventional.mean()
                rows14.append(dict(sweep=sw, value=v, system=sname, conv=p.conventional.mean(),
                                   prop=p.proposed.mean(), ratio_prop=rr, ratio_fail=rf))
                cells.append(f"{p.conventional.mean():.2f} / {p.proposed.mean():.2f} ({rr:.2f})")
            trs.append(f"{lab[sw]} & {round(v, 2):g} & " + " & ".join(cells))
        trs[-1] += " \\\\ \\hline"
    pd.DataFrame(rows14).to_csv(f"{T}/T14_sensitivity.csv", index=False)
    tex("T14_sensitivity", "Sensitivity of HL-I LOLE (h/yr): conventional / proposed (ratio), 1500 paired years per point",
        "tab:sensitivity", "Parameter & Value & RTS-79 & IEEE 39 & IEEE 118", trs, "|l|c|c|c|c|", star=True)

# ------------------------------------------------------------------ T15 computation
trs, rows15 = [], []
if os.path.exists(f"{R}/benchmark_cpu.csv"):
    cp = pd.read_csv(f"{R}/benchmark_cpu.csv").groupby("system").mean(numeric_only=True)
else:
    cp = None
for sname in C.SYSTEMS:
    f = f"{R}/main_{sname}.csv"
    if not os.path.exists(f):
        continue
    d = pd.read_csv(f)
    cells = []
    for v in ("conventional", "proposed"):
        x = d[d.variant == v].HL2_EENS.values
        beta = x.std(ddof=1) / (x.mean() * np.sqrt(len(x)))
        n5 = int(np.ceil((x.std(ddof=1) / (0.05 * x.mean())) ** 2))
        cells += [f"{100 * beta:.2f}", f"{n5}"]
        rows15.append(dict(system=sname, variant=v, beta_pct_1000y=100 * beta, years_for_5pct=n5))
    if cp is not None and sname in cp.index:
        r = cp.loc[sname]
        cells = [f"{r.naive_s:.2f}", f"{r.cached_s:.3f}", f"{r.naive_s / r.cached_s:.0f}$\\times$",
                 f"{r.naive_lps:.0f}", f"{r.cached_lps:.0f}", f"{r.max_abs_diff_MW:.1e}"] + cells
        rows15[-1].update(dict(naive_s=r.naive_s, cached_s=r.cached_s, speedup=r.naive_s / r.cached_s,
                               naive_lps=r.naive_lps, cached_lps=r.cached_lps, max_diff=r.max_abs_diff_MW))
    trs.append(f"{SL[sname]} & " + " & ".join(cells))
pd.DataFrame(rows15).to_csv(f"{T}/T15_computation.csv", index=False)
tex("T15_computation", "Computational performance: HL-II evaluation per simulated year (LP at every hour vs cached "
    "maximum servable load; identical results) and Monte Carlo convergence of HL-II EENS "
    "($\\beta$ = coefficient of variation of the estimate after 1000 years; years needed for $\\beta$ = 5\\%)",
    "tab:computation",
    "System & Naive (s) & Cached (s) & Speed-up & LPs naive & LPs cached & Max diff. (MW) & "
    "$\\beta$ conv. (\\%) & Yrs conv. & $\\beta$ prop. (\\%) & Yrs prop.", trs, "|l|c|c|c|c|c|c|c|c|c|c|", star=True)

# ------------------------------------------------------------------ T16 optimiser settings
opt = [("jDE", "rand/1/bin; $F_i\\in[0.1,1]$, $CR_i\\in[0,1]$ self-adapted, $\\tau_1=\\tau_2=0.1$ [R7]"),
       ("Standard DE", "rand/1/bin; $F=0.5$, $CR=0.9$ [R8]"),
       ("PSO", "constriction; $w=0.7298$, $c_1=c_2=1.49618$, $v_{\\max}=0.2$ range"),
       ("GA", "real-coded; binary tournament, SBX ($\\eta_c=15$, $p_c=0.9$), polynomial mutation ($\\eta_m=20$, $p_m=1/D$), elitism"),
       ("Common", "population 30, 200 generations (6030 evaluations), 25 seeds, bounds [0, 2 max curtailment]"),
       ("Final plans", "jDE, population 40, 1000 generations, 6 seeds")]
tex("T16_opt_settings", "Optimiser settings (storage planning)", "tab:optset", "Algorithm & Settings",
    [f"{a} & {b}" for a, b in opt], "|l|p{0.7\\linewidth}|")

# ------------------------------------------------------------------ T17 siting
trs, rows17 = [], []
for sname in C.SYSTEMS:
    f = f"{R}/planning_{sname}_plans.json"
    if not os.path.exists(f):
        continue
    js = json.load(open(f))["plans"]
    sa, cv = js["Stress-aware plan (proposed model)"], js["Conventional-model plan"]
    buses = sorted(set(sa) | set(cv), key=lambda b: -max(sa.get(b, 0), cv.get(b, 0)))
    top = [b for b in buses if max(sa.get(b, 0), cv.get(b, 0)) > 0.5][:6]
    for b in top:
        rows17.append(dict(system=sname, bus=int(b), MW_stress_aware=sa.get(b, 0), MW_conventional=cv.get(b, 0)))
        trs.append(f"{SL[sname]} & {b} & {cv.get(b, 0):.1f} & {sa.get(b, 0):.1f}")
    tot_sa, tot_cv = sum(sa.values()), sum(cv.values())
    trs.append(f"{SL[sname]} & total & {tot_cv:.1f} & {tot_sa:.1f} \\\\ \\hline")
pd.DataFrame(rows17).to_csv(f"{T}/T17_siting.csv", index=False)
tex("T17_siting", "Storage siting of the final plans (MW at 4 h; largest six sites)", "tab:siting",
    "System & Bus & Conventional-model plan & Stress-aware plan", trs, "|l|c|c|c|")

print("tables:", sorted(f for f in os.listdir(T) if f.endswith(".tex")))
