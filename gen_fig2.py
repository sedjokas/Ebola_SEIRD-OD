"""
gen_fig2.py
===========
Generates Figure 2 for the SEIHRF-OD manuscript:
  Panel A: R₀ as a function of initial skepticism φ₀ for three β_FR/β_I ratios
  Panel B: Critical vaccination coverage p_c = 1 - 1/R₀ under same scenarios

Uses updated posterior-median parameters (MCMC on 127 cases, SitReps 001-012).
Vertical dotted line at φ̂₀ = 0.392; shaded band = 95% CrI [0.294, 0.490].
"""

from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUTDIR = "/Users/selainkaserekakabunga/Documents/Lancet_Paper/imgs"

# ── Updated posterior-median parameters ───────────────────────────────────────
BETA_I   = 0.826
BETA_H   = 0.06
KAPPA    = 1.0 / 9
THETA_B  = 0.28
THETA_N  = 0.040
DELTA_I  = 0.18
DELTA_H  = 0.12
GAMMA_I  = 0.09
GAMMA_H  = 0.10
OMEGA_FR = 0.80
PSI_I    = 0.45
PSI_H    = 0.15

# Posterior φ₀ (updated from MCMC on 127 cases)
PHI0_MED = 0.392
PHI0_LO  = 0.294   # 95% CrI lower
PHI0_HI  = 0.490   # 95% CrI upper


# ── Analytical R₀ formula ─────────────────────────────────────────────────────
def compute_R0(phi0: float, ratio: float) -> float:
    """ratio = beta_FR / beta_I"""
    beta_FR = ratio * BETA_I
    gh_sum  = DELTA_H + GAMMA_H
    denom_B = THETA_B  + DELTA_I + GAMMA_I
    denom_N = THETA_N  + DELTA_I + GAMMA_I

    R0B = (BETA_I + BETA_H * THETA_B / gh_sum) / denom_B
    R0N = (BETA_I + BETA_H * THETA_N / gh_sum
           + beta_FR / OMEGA_FR * (
               PSI_I * DELTA_I
               + PSI_H * THETA_N * DELTA_H / gh_sum
           )) / denom_N

    trM  = (1 - phi0) * R0B + phi0 * R0N
    burial = (PSI_I * DELTA_I + PSI_H * THETA_N * DELTA_H / gh_sum) / denom_N
    detM = phi0 * (1 - phi0) * R0B * (beta_FR / OMEGA_FR) * burial
    disc = max(0.0, trM**2 - 4 * detM)
    return (trM + np.sqrt(disc)) / 2


# ── Compute curves ─────────────────────────────────────────────────────────────
phi_range = np.linspace(0.0, 0.75, 400)

ratios  = [1, 3, 5]
labels  = [r"$\beta_{FR}/\beta_I = 1$  (homogeneous model)",
           r"$\beta_{FR}/\beta_I = 3$",
           r"$\beta_{FR}/\beta_I = 5$"]
colors  = ["#9E9E9E", "#3A6EA5", "#E05A47"]
lstyles = ["--", "-", "-"]
lws     = [1.8, 2.0, 2.0]

R0_curves = []
pc_curves = []
for r in ratios:
    rv = np.array([compute_R0(p, r) for p in phi_range])
    R0_curves.append(rv)
    pc_curves.append(1 - 1 / rv)

# Values at posterior median φ̂₀
R0_at_med  = [compute_R0(PHI0_MED, r) for r in ratios]
pc_at_med  = [1 - 1/v for v in R0_at_med]

# Bias annotation at φ̂₀: p_c(ratio=3) minus p_c(ratio=1) in pp
pc_bias_pp = (pc_at_med[1] - pc_at_med[0]) * 100   # percentage points
R0_bias_pct = (R0_at_med[1] - R0_at_med[0]) / R0_at_med[0] * 100

print(f"At φ̂₀ = {PHI0_MED}:")
for r, R0, pc in zip(ratios, R0_at_med, pc_at_med):
    print(f"  β_FR/β_I = {r}: R₀ = {R0:.3f},  p_c = {pc:.3f}")
print(f"Bias (ratio=3 vs ratio=1): R₀ +{R0_bias_pct:.1f}%,  p_c +{pc_bias_pp:.1f} pp")


# ── Plot ──────────────────────────────────────────────────────────────────────
STYLE = {
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.direction": "out",
    "ytick.direction": "out",
}
plt.rcParams.update(STYLE)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2))
fig.subplots_adjust(wspace=0.35)

for ax in (ax1, ax2):
    # Shaded CrI for φ₀
    ax.axvspan(PHI0_LO, PHI0_HI, alpha=0.12, color="#3A6EA5",
               label="95% CrI for $\\hat{\\phi}_0$" if ax is ax1 else "")
    # Vertical line at posterior median φ̂₀
    ax.axvline(PHI0_MED, color="#3A6EA5", lw=1.2, ls=":",
               label=f"$\\hat{{\\phi}}_0 = {PHI0_MED}$" if ax is ax1 else "")
    ax.set_xlabel("Initial skepticism $\\phi_0$", fontsize=10)
    ax.set_xlim(0, 0.75)

# ── Panel A: R₀ vs φ₀ ────────────────────────────────────────────────────────
ax1.axhline(1.0, color="black", lw=0.9, ls=":", alpha=0.5,
            label="Epidemic threshold $\\mathcal{R}_0 = 1$")

for rv, lbl, col, ls, lw in zip(R0_curves, labels, colors, lstyles, lws):
    ax1.plot(phi_range, rv, color=col, lw=lw, ls=ls, label=lbl)

ax1.set_ylabel("$\\mathcal{R}_0$", fontsize=11)
ax1.set_title("(A)  $\\mathcal{R}_0$ vs initial skepticism",
              fontsize=10, pad=7)
ax1.set_ylim(0.6, max(R0_curves[-1]) * 1.12)
ax1.legend(fontsize=8.5, loc="upper left")

# Annotate calibrated point on ratio=3 curve
ax1.plot(PHI0_MED, R0_at_med[1], "o", color=colors[1], ms=6, zorder=5)
ax1.annotate(f"$\\mathcal{{R}}_0 = {R0_at_med[1]:.2f}$\nat $\\hat{{\\phi}}_0$",
             xy=(PHI0_MED, R0_at_med[1]),
             xytext=(PHI0_MED + 0.06, R0_at_med[1] - 0.15),
             fontsize=8, color=colors[1],
             arrowprops=dict(arrowstyle="->", color=colors[1], lw=0.8))

# ── Panel B: p_c vs φ₀ ────────────────────────────────────────────────────────
for pc, lbl, col, ls, lw in zip(pc_curves, labels, colors, lstyles, lws):
    ax2.plot(phi_range, pc * 100, color=col, lw=lw, ls=ls, label=lbl)

ax2.set_ylabel("Critical vaccination coverage $p_c$  (%)", fontsize=10)
ax2.set_title("(B)  Vaccination coverage threshold vs $\\phi_0$",
              fontsize=10, pad=7)
ax2.set_ylim(20, max(pc_curves[-1]) * 102)

# Bias bracket at φ̂₀
y_hom  = pc_at_med[0] * 100
y_het  = pc_at_med[1] * 100
x_ann  = PHI0_MED + 0.035
ax2.annotate("", xy=(x_ann, y_het), xytext=(x_ann, y_hom),
             arrowprops=dict(arrowstyle="<->", color="black", lw=1.0))
ax2.text(x_ann + 0.012, (y_hom + y_het) / 2,
         f"+{pc_bias_pp:.1f} pp\nbias",
         fontsize=8, va="center", color="black")

# Dots at posterior median
ax2.plot(PHI0_MED, y_hom, "o", color=colors[0], ms=5, zorder=5)
ax2.plot(PHI0_MED, y_het, "o", color=colors[1], ms=5, zorder=5)

ax2.legend(fontsize=8.5, loc="upper left")

plt.tight_layout()

for ext in ["pdf", "png"]:
    path = os.path.join(OUTDIR, f"fig2_R0_analysis.{ext}")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    print(f"Saved {path}")

print("Done.")
