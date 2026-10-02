"""Draws the STRAP framework figure (method_strap.png / .pdf)."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans"})
W, H = 17.8, 6.4
fig, ax = plt.subplots(figsize=(W, H))
ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")

INK, MUTED, LINE = "#1f2937", "#4b5563", "#9ca3af"
COLS = ["#2563eb", "#7c3aed", "#0d9488", "#ea580c", "#be123c"]
TINT = ["#eff6ff", "#f5f3ff", "#f0fdfa", "#fff7ed", "#fff1f2"]

stages = [
    ("Stress index",
     [r"$\sigma_k(t)=e_k\,\ell(t)^{\kappa}$",
      r"$e_k=\eta_{L}\,(\vartheta_k/\vartheta_{\mathrm{ref}})\,\eta_{A}\,\eta_{V}$"],
     ["hourly loading $\\ell(t)$", "component stress factors", "load coupling $\\kappa$"]),
    ("Stress-dependent\nfailure and repair",
     [r"$h_k(t)=1/\left[\alpha_k+\beta_k/(\sigma_k(t)+\varepsilon)\right]$",
      r"fail when $\int h_k\,dt\geq Z,\ Z\sim\mathrm{Exp}(1)$",
      r"$E[\mathrm{TTR}\,|\,t_f]=\mathrm{MTTR}\,[\gamma+(1-\gamma)\,\sigma(t_f)/\bar{\sigma}^{f}]$"],
     ["peak hours: more failures", "and longer repairs"]),
    ("Mean-preserving\ncalibration",
     [r"$\alpha_k=a\cdot\mathrm{MTTF}_k$",
      r"$\beta_k$: mean hazard $=1/\mathrm{MTTF}_k$",
      r"repair normalised by $\bar{\sigma}^{f}$"],
     ["same MTTF, MTTR and FOR", "as the conventional model", "$\\Rightarrow$ only timing differs"]),
    ("Sequential MC\nevaluation",
     [r"HL-I: capacity vs. load",
      r"HL-II: DC load-shedding LP",
      r"cache of servable load $\rho^{*}$"],
     ["LOLE, EENS, LOLF, SAIDI", "component and load-point", "criticality"]),
    ("Storage siting\nand sizing (jDE)",
     [r"$\min\ J=\mathrm{CRF}\cdot C^{\mathrm{cap}}\sum_b P_b^{s}$",
      r"$+\ \mathrm{VOLL}\cdot\mathrm{EENS}(P^{s})$",
      r"$+\ \psi\,[\mathrm{LOLE}(P^{s})-\mathrm{LOLE}^{*}]^{+}$"],
     ["MW per bus", "out-of-sample cost", "and reliability"]),
]

bw, gap, x0 = 3.2, 0.33, 0.25
yb, bh = 1.05, 4.15
centers = []
for i, (title, eqs, notes) in enumerate(stages):
    x = x0 + i * (bw + gap)
    centers.append(x + bw / 2)
    ax.add_patch(FancyBboxPatch((x, yb), bw, bh, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=TINT[i], ec=COLS[i], lw=1.6))
    ax.add_patch(FancyBboxPatch((x, yb + bh - 1.0), bw, 1.0, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=COLS[i], ec=COLS[i], lw=1.6))
    ax.add_patch(plt.Rectangle((x - 0.02, yb + bh - 1.0), bw + 0.04, 0.25, fc=COLS[i], ec="none"))
    ax.text(x + 0.18, yb + bh - 0.5, f"{i + 1}", color="white", fontsize=20, fontweight="bold", va="center")
    ax.text(x + 0.62, yb + bh - 0.5, title, color="white", fontsize=12.2, fontweight="bold", va="center",
            linespacing=1.1)
    y = yb + bh - 1.45
    for e in eqs:
        ax.text(x + bw / 2, y, e, ha="center", va="center", fontsize=9.6 if i == 1 else 10.4, color=INK)
        y -= 0.52
    ax.plot([x + 0.25, x + bw - 0.25], [yb + 1.38, yb + 1.38], color=COLS[i], lw=0.8, alpha=0.5)
    y = yb + 1.08
    for n in notes:
        ax.text(x + bw / 2, y, n, ha="center", va="center", fontsize=9.6, color=MUTED, style="italic")
        y -= 0.33
    if i < len(stages) - 1:
        ax.add_patch(FancyArrowPatch((x + bw + 0.03, yb + bh / 2), (x + bw + gap - 0.03, yb + bh / 2),
                                     arrowstyle="-|>", mutation_scale=18, lw=1.8, color=INK))

# data inputs (top) and parameter estimation feedback
ax.text(centers[0] - bw / 2, H - 0.42, "Inputs: ", fontsize=10.5, fontweight="bold", color=INK, va="center")
ax.text(centers[0] - bw / 2 + 0.85, H - 0.42,
        "network data (PGLib-OPF)  ·  hourly load profile (IEEE RTS-79)  ·  component MTTF / MTTR  ·  outage records",
        fontsize=10.5, color=MUTED, va="center")
ax.annotate("", xy=(centers[0], yb + bh + 0.03), xytext=(centers[0], H - 0.62),
            arrowprops=dict(arrowstyle="-|>", color=LINE, lw=1.3))

# MLE feedback from outage records to stages 1-2
ax.annotate("", xy=(centers[1], yb + bh + 0.03), xytext=(centers[1], H - 0.62),
            arrowprops=dict(arrowstyle="-|>", color=LINE, lw=1.3, ls="--"))
ax.text(centers[1] + 0.12, H - 0.85, r"$\hat\kappa,\ \hat\gamma$ by maximum likelihood", fontsize=9.6,
        color=MUTED, va="center")

# bottom brace: reduces to conventional model
ax.add_patch(FancyArrowPatch((centers[0] - bw / 2 + 0.1, 0.55), (centers[2] + bw / 2 - 0.1, 0.55),
                             arrowstyle="-", lw=1.2, color=LINE))
ax.text((centers[0] + centers[2]) / 2, 0.28,
        r"Outage model: reduces exactly to the conventional two-state model when $\kappa=0$",
        ha="center", va="center", fontsize=10, color=MUTED)
ax.add_patch(FancyArrowPatch((centers[3] - bw / 2 + 0.1, 0.55), (centers[4] + bw / 2 - 0.1, 0.55),
                             arrowstyle="-", lw=1.2, color=LINE))
ax.text((centers[3] + centers[4]) / 2, 0.28, "Assessment and planning", ha="center", va="center",
        fontsize=10, color=MUTED)

fig.savefig("results/figures/method_strap.png", dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig("results/figures/method_strap.pdf", bbox_inches="tight", facecolor="white")
print("ok")
