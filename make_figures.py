"""
make_figures.py — every figure of the revised paper, from results/ only.
Saved to results/figures/*.png (300 dpi) and *.pdf (vector, for the manuscript).
Palette: fixed categorical order; one y-axis per panel; 95 % CIs everywhere.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C

R = "results"
F = f"{R}/figures"
os.makedirs(F, exist_ok=True)

BLUE, ORANGE, AQUA, YELLOW, MAGENTA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"
GREY, INK, INK2, GRID = "#8f8e89", "#0b0b0b", "#52514e", "#e6e5e0"
SYS_COL = {"IEEE39": BLUE, "IEEE57": ORANGE, "IEEE118": AQUA, "RTS79": VIOLET}
SYS_LAB = {"IEEE39": "IEEE 39-bus", "IEEE57": "IEEE 57-bus", "IEEE118": "IEEE 118-bus", "RTS79": "IEEE RTS-79"}
VAR_COL = {"conventional": GREY, "stress_failure": AQUA, "stress_repair": ORANGE, "proposed": BLUE}
VAR_LAB = {"conventional": "Conventional (constant rates)", "stress_failure": "Stress-dependent failure only",
           "stress_repair": "Stress-dependent repair only", "proposed": "Proposed (stress-dependent failure and repair)"}
ALG_COL = {"jDE": BLUE, "Standard DE": ORANGE, "PSO": YELLOW, "GA": MAGENTA}

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.6, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.grid": True, "axes.axisbelow": True, "lines.linewidth": 1.4,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "savefig.dpi": 300, "savefig.bbox": "tight", "font.family": "DejaVu Sans"})
W1, W2 = 3.5, 7.16      # IEEE column / page width (in)


def save(fig, name):
    fig.savefig(f"{F}/{name}.png")
    fig.savefig(f"{F}/{name}.pdf")
    plt.close(fig)
    print("saved", name)


def paired_pct(d, metric, v, ref="conventional"):
    p = d.pivot(index="year", columns="variant", values=metric)
    diff = p[v] - p[ref]
    c = p[ref].mean()
    return 100 * diff.mean() / c, 100 * 1.96 * diff.std(ddof=1) / np.sqrt(len(diff)) / c


MAIN = {n: pd.read_csv(f"{R}/main_{n}.csv") for n in C.SYSTEMS if os.path.exists(f"{R}/main_{n}.csv")}

# -------------------------------------------------------------- Fig. mechanism
if os.path.exists(f"{R}/mechanism.csv"):
    mech = pd.read_csv(f"{R}/mechanism.csv")
    systems = [s for s in ("RTS79", "IEEE39", "IEEE118") if s in mech.system.unique()]
    fig, axs = plt.subplots(2, len(systems), figsize=(W2, 3.9), sharex=True)
    for j, sname in enumerate(systems):
        for v in ("conventional", "stress_failure", "stress_repair", "proposed"):
            g = mech[(mech.system == sname) & (mech.variant == v)].groupby("decile")
            for i, (col, lab) in enumerate((("fail_per_1000_up_h", "Unit failures per\n1000 unit-hours up"),
                                            ("cap_out", "Capacity on outage (MW)"))):
                m = g[col].mean(); h = 1.96 * g[col].std() / np.sqrt(g.size())
                ax = axs[i, j]
                ls = "--" if v in ("stress_failure", "stress_repair") else "-"
                ax.plot(m.index, m.values, color=VAR_COL[v], ls=ls, marker="o", ms=3, lw=1.4,
                        label=VAR_LAB[v] if (i == 0 and j == 0) else None)
                ax.fill_between(m.index, m - h, m + h, color=VAR_COL[v], alpha=0.12, lw=0)
                if j == 0:
                    ax.set_ylabel(lab)
        axs[0, j].set_title(SYS_LAB[sname], color=INK)
        axs[1, j].set_xticks(range(1, 11))
    fig.supxlabel("Hourly load decile (1 = lightest hours, 10 = peak hours)", fontsize=8, color=INK, y=0.01)
    fig.legend(loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    save(fig, "fig_mechanism")

# -------------------------------------------------------------- Fig. main effect sizes
if MAIN:
    metrics = [("HL1_LOLE", "HL-I LOLE"), ("HL2_LOLE", "HL-II LOLE"), ("HL2_EENS", "HL-II EENS")]
    vars_ = ["stress_failure", "stress_repair", "proposed"]
    fig, axs = plt.subplots(1, 3, figsize=(W2, 2.5), sharey=True)
    for a, (m, lab) in zip(axs, metrics):
        for k, (sname, d) in enumerate(MAIN.items()):
            for i, v in enumerate(vars_):
                mu, h = paired_pct(d, m, v)
                y = i + (k - 1) * 0.22
                a.errorbar(mu, y, xerr=h, fmt="o", ms=4, color=SYS_COL[sname], ecolor=SYS_COL[sname],
                           elinewidth=1.2, capsize=0, mec="white", mew=0.8,
                           label=SYS_LAB[sname] if (i == 0 and a is axs[0]) else None)
        a.axvline(0, color=INK2, lw=0.8)
        a.set_title(lab, color=INK)
        a.grid(axis="y", visible=False)
    axs[0].set_yticks(range(len(vars_)))
    axs[0].set_yticklabels([VAR_LAB[v].replace(" (", "\n(").replace(", ", ",\n") for v in vars_])
    axs[0].invert_yaxis()
    fig.supxlabel("Change vs conventional model (%), paired years, 95 % CI; all models have equal mean failure and repair rates",
                  fontsize=7.5, color=INK, y=0.02)
    fig.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.58, 1.07))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, "fig_effect_sizes")

    # ---------------------------------------------------------- Fig. tail risk (CCDF)
    fig, axs = plt.subplots(1, len(MAIN), figsize=(W2, 2.3))
    for a, (sname, d) in zip(axs, MAIN.items()):
        for v in ("conventional", "proposed"):
            x = np.sort(d[d.variant == v]["HL2_EENS"].values)
            ccdf = 1.0 - np.arange(1, len(x) + 1) / len(x)
            a.step(x / 1e3, np.maximum(ccdf, 1e-3), where="post", color=VAR_COL[v], lw=1.6,
                   label=VAR_LAB[v].split(" (")[0])
            q = np.percentile(x, 95) / 1e3
            a.axvline(q, color=VAR_COL[v], lw=0.8, ls=":")
            if sname == list(MAIN)[0]:
                pass
        a.set_yscale("log")
        a.set_ylim(1e-3, 1.05)
        a.set_title(SYS_LAB[sname], color=INK)
        a.set_xlabel("Annual HL-II EENS (GWh)")
    axs[0].set_ylabel("P(annual EENS > x)")
    axs[0].legend(loc="lower left")
    fig.tight_layout()
    save(fig, "fig_eens_ccdf")

# -------------------------------------------------------------- Fig. sensitivity
if os.path.exists(f"{R}/sensitivity.csv"):
    sd = pd.read_csv(f"{R}/sensitivity.csv")

    def ratio_ci(g, v="proposed"):
        p = g.pivot(index="year", columns="variant", values="LOLE")
        rng = np.random.default_rng(0)
        n = len(p)
        rs = []
        a, b = p[v].values, p["conventional"].values
        for _ in range(400):
            i = rng.integers(0, n, n)
            rs.append(a[i].mean() / max(b[i].mean(), 1e-9))
        return a.mean() / b.mean(), np.percentile(rs, 2.5), np.percentile(rs, 97.5)

    panels = [("lole_target", "(a) Reliability level", "LOLE target (h/yr)", True),
              ("kappa", "(b) Load coupling", "Exponent $\\kappa$", False),
              ("gamma_fraction", "(c) Repair stress", "Fixed repair share $\\gamma$", False),
              ("alpha_fraction", "(d) Minimum life", "$\\alpha$ / MTTF", False)]
    fig, axs = plt.subplots(1, 4, figsize=(W2, 2.4), sharey=True)
    for a, (sw, ttl, lab, logx) in zip(axs, panels):
        a.set_title(ttl, color=INK, loc="left")
        for sname in [s for s in ("RTS79", "IEEE39", "IEEE118") if s in sd.system.unique()]:
            g = sd[(sd.system == sname) & (sd.sweep == sw)]
            vals = sorted(g.value.unique())
            st = np.array([ratio_ci(g[g.value == v]) for v in vals])
            a.plot(vals, st[:, 0], marker="o", ms=3.5, color=SYS_COL[sname], label=SYS_LAB[sname])
            a.fill_between(vals, st[:, 1], st[:, 2], color=SYS_COL[sname], alpha=0.12, lw=0)
        a.axhline(1.0, color=INK2, lw=0.8)
        if logx:
            a.set_xscale("log")
        a.set_xlabel(lab)
    axs[0].set_ylabel("HL-I LOLE ratio\nproposed / conventional")
    axs[0].legend(loc="upper right")
    fig.tight_layout()
    save(fig, "fig_sensitivity")


# -------------------------------------------------------------- Fig. optimiser convergence + final J
runs_files = [n for n in C.SYSTEMS if os.path.exists(f"{R}/planning_{n}_runs.csv")]
if runs_files:
    fig, axs = plt.subplots(2, len(runs_files), figsize=(W2, 4.2), squeeze=False,
                            gridspec_kw=dict(height_ratios=[1.3, 1]))
    for j, sname in enumerate(runs_files):
        hz = np.load(f"{R}/planning_{sname}_hist.npz")
        runs = pd.read_csv(f"{R}/planning_{sname}_runs.csv")
        tr = runs[runs.train == "train_prop"]
        jbest = tr.J.min()
        a = axs[0, j]
        for alg, col in ALG_COL.items():
            H = np.array([hz[k] for k in hz.files if k.startswith(alg + "|train_prop|")])
            gap = 100 * (H - jbest) / jbest
            evals = 30 * np.arange(1, H.shape[1] + 1)
            m = np.median(gap, axis=0)
            q1, q3 = np.percentile(gap, 25, axis=0), np.percentile(gap, 75, axis=0)
            a.plot(evals, np.maximum(m, 1e-4), color=col, lw=1.5, label=alg)
            a.fill_between(evals, np.maximum(q1, 1e-4), np.maximum(q3, 1e-4), color=col, alpha=0.12, lw=0)
        a.set_yscale("log")
        a.set_title(SYS_LAB[sname], color=INK)
        a.set_xlabel("Objective evaluations")
        if j == 0:
            a.set_ylabel("Gap to best plan found (%)\nmedian, IQR over seeds")
        b = axs[1, j]
        algs = list(ALG_COL)
        data = [100 * (tr[tr.algorithm == al].J.values - jbest) / jbest for al in algs]
        bp = b.boxplot(data, vert=False, widths=0.5, patch_artist=True, showfliers=True,
                       flierprops=dict(marker="o", ms=2.5, mec=INK2, mfc="none"),
                       medianprops=dict(color=INK, lw=1.2), whiskerprops=dict(color=INK2, lw=0.8),
                       capprops=dict(color=INK2, lw=0.8), boxprops=dict(lw=0))
        for patch, al in zip(bp["boxes"], algs):
            patch.set_facecolor(ALG_COL[al]); patch.set_alpha(0.85)
        b.set_yticks(range(1, len(algs) + 1))
        b.set_yticklabels(algs if j == 0 else [""] * len(algs))
        b.invert_yaxis()
        b.set_xlabel("Final gap to best plan (%)")
        b.grid(axis="y", visible=False)
    h_, l_ = axs[0, 0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, "fig_optimizers")

# -------------------------------------------------------------- Fig. planning outcome
plan_files = [n for n in C.SYSTEMS if os.path.exists(f"{R}/planning_{n}_plans.csv")]
if plan_files:
    PLAN_COL = {"No storage": GREY, "Heuristic (EENS-proportional, same MW)": YELLOW,
                "Conventional-model plan": ORANGE, "Stress-aware plan (proposed model)": BLUE}
    fig, axs = plt.subplots(1, 3, figsize=(W2, 2.5))
    width = 0.2
    for k, (pname, col) in enumerate(PLAN_COL.items()):
        for q, (metric, lab) in enumerate((("MW", "Storage installed (MW)"), ("EENS", "Out-of-sample EENS (MWh/yr)"),
                                           ("LOLE", "Out-of-sample LOLE (h/yr)"))):
            vals = []
            for sname in plan_files:
                d = pd.read_csv(f"{R}/planning_{sname}_plans.csv")
                r = d[(d.plan == pname) & (d.set == "test_prop")]
                vals.append(float(r[metric].iloc[0]))
            x = np.arange(len(plan_files)) + (k - 1.5) * width
            axs[q].bar(x, vals, width=width * 0.9, color=col, label=pname if q == 0 else None)
            axs[q].set_title(lab, color=INK)
            axs[q].set_xticks(np.arange(len(plan_files)))
            axs[q].set_xticklabels([SYS_LAB[s].replace("IEEE ", "").replace("-bus", "-bus") for s in plan_files])
            axs[q].grid(axis="x", visible=False)
    axs[2].axhline(C.LOLE_TARGET_PLAN, color=INK2, lw=0.8, ls="--")
    axs[2].text(axs[2].get_xlim()[1], C.LOLE_TARGET_PLAN, " target", va="center", ha="left", color=INK2, fontsize=7)
    fig.legend(loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout()
    save(fig, "fig_planning")

# -------------------------------------------------------------- Fig. method benchmark (known truth, HL-I)
if os.path.exists(f"{R}/benchmark_hl1.csv"):
    b = pd.read_csv(f"{R}/benchmark_hl1.csv")
    MCOL = {"COPT": GREY, "Seasonal": YELLOW, "Two-level": ORANGE, "Proposed": BLUE}
    MLAB = {"COPT": "Conventional (COPT / two-state)", "Seasonal": "Seasonal rates", "Two-level": "Two-level (peak/off-peak)",
            "Proposed": "Proposed"}
    TLAB = {"T0": "Constant", "T1": "$\\kappa$=1", "T2": "$\\kappa$=2", "T3": "Mis-spec."}
    systems = [s for s in ("RTS79", "IEEE39", "IEEE118") if s in b.system.unique()]
    fig, axs = plt.subplots(1, len(systems), figsize=(W2, 2.6), sharey=True)
    for a, sname in zip(axs, systems):
        for k, m in enumerate(MCOL):
            for i, tk in enumerate(("T0", "T1", "T2", "T3")):
                x = np.abs(b[(b.system == sname) & (b.truth == tk) & (b.method == m)].EENS_err_pct.values)
                mu, h = x.mean(), 1.96 * x.std(ddof=1) / np.sqrt(len(x))
                a.errorbar(i + (k - 1.5) * 0.18, mu, yerr=h, fmt="o", ms=4, color=MCOL[m], elinewidth=1.2,
                           capsize=0, mec="white", mew=0.8, label=MLAB[m] if (i == 0 and a is axs[0]) else None)
        a.set_xticks(range(4)); a.set_xticklabels([TLAB[t] for t in ("T0", "T1", "T2", "T3")], fontsize=7)
        a.set_title(SYS_LAB[sname], color=INK)
        a.grid(axis="x", visible=False)
    axs[0].set_ylabel("Mean absolute error of\npredicted EENS (%)")
    fig.supxlabel("True outage process: constant rates, stress-dependent ($\\kappa$ = 1, 2), exponential-in-load (mis-specified)",
                  fontsize=7.5, color=INK, y=0.0)
    fig.legend(loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save(fig, "fig_benchmark")
