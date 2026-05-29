"""
gen_fig4.py
===========
Generates Figure 4 for the SEIHRF-OD manuscript:
  Panel A: Tornado plot of first-order Sobol sensitivity indices for D(90)
  Panel B: Prior vs posterior density for β_FR, φ₀, γ_comm, α

Uses updated posterior-median parameters (MCMC on 127 cases, SitReps 001-012).
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.integrate import solve_ivp
from scipy.stats import norm, gaussian_kde
from SALib.sample import saltelli
from SALib.analyze import sobol as sobol_analyze

OUTDIR = "/Users/selainkaserekakabunga/Documents/Lancet_Paper/imgs"
ROOT   = "/Users/selainkaserekakabunga/Documents/Lancet_Paper"

# ── Posterior-median parameters (updated) ─────────────────────────────────────
P_MED = dict(
    N=120_000, beta_I=0.826, beta_H=0.06, beta_FR=1.610, beta_FS=0.002,
    kappa=1/9, theta_B=0.28, theta_N=0.040, delta_I=0.18, delta_H=0.12,
    gamma_I=0.09, gamma_H=0.10, omega_FR=0.80, omega_FS=3.00,
    psi_I=0.45, psi_H=0.15, alpha=0.037, gamma_comm=0.022, delta_C=0.045,
    beta_D=8.00, phi0=0.392,
)

T_MAX = 90
DAYS  = np.linspace(0, T_MAX, T_MAX * 5 + 1)


def C_func(t):
    m = t + 21.0
    if m < 17:   return 0.30
    elif m < 24: return 0.55
    elif m < 27: return 0.65
    elif m <= 29: return 1.00
    else:         return 0.60


def odes(t, y, p):
    SB, EB, IB, HB, RB, SN, EN, IN, HN, RN, FR, FS, Dcum = y
    N = SB+EB+IB+HB+RB+SN+EN+IN+HN+RN
    if N <= 0: return [0.]*13
    Ct  = C_func(t)
    lB  = (p[0]*(IB+IN) + p[1]*(HB+HN) + p[2]*FS) / N   # beta_I, beta_H, beta_FS
    lN  = (p[0]*(IB+IN) + p[1]*(HB+HN) + p[3]*FR) / N   # beta_FR
    Dv  = (p[4]*(IB+IN) + p[5]*(HB+HN)) / N              # delta_I, delta_H
    phi = SN/(SB+SN) if (SB+SN)>0 else p[6]              # phi0
    mu_BN = p[7]*phi + p[8]*Ct                            # alpha, delta_C
    mu_NB = p[9] + p[10]*Dv                               # gamma_comm, beta_D

    dSB=-lB*SB - mu_BN*SB + mu_NB*SN
    dEB= lB*SB - p[11]*EB                                 # kappa
    dIB= p[11]*EB - (p[12]+p[4]+p[13])*IB                # theta_B, gamma_I
    dHB= p[12]*IB - (p[5]+p[14])*HB                      # gamma_H
    dRB= p[13]*IB + p[14]*HB
    dSN=-lN*SN + mu_BN*SB - mu_NB*SN
    dEN= lN*SN - p[11]*EN
    dIN= p[11]*EN - (p[15]+p[4]+p[13])*IN                # theta_N
    dHN= p[15]*IN - (p[5]+p[14])*HN
    dRN= p[13]*IN + p[14]*HN
    dFR= p[16]*p[4]*IN + p[17]*p[5]*HN - p[18]*FR        # psi_I, psi_H, omega_FR
    dFS= (p[4]*IB + p[5]*HB + (1-p[16])*p[4]*IN + (1-p[17])*p[5]*HN - p[19]*FS)  # omega_FS
    dDcum= p[4]*(IB+IN) + p[5]*(HB+HN)
    return [dSB,dEB,dIB,dHB,dRB,dSN,dEN,dIN,dHN,dRN,dFR,dFS,dDcum]

# param vector order:
# [0]beta_I [1]beta_H [2]beta_FS [3]beta_FR [4]delta_I [5]delta_H [6]phi0
# [7]alpha [8]delta_C [9]gamma_comm [10]beta_D [11]kappa [12]theta_B
# [13]gamma_I [14]gamma_H [15]theta_N [16]psi_I [17]psi_H [18]omega_FR [19]omega_FS

def p_vec(pm):
    return [pm["beta_I"], pm["beta_H"], pm["beta_FS"], pm["beta_FR"],
            pm["delta_I"], pm["delta_H"], pm["phi0"], pm["alpha"], pm["delta_C"],
            pm["gamma_comm"], pm["beta_D"], pm["kappa"], pm["theta_B"],
            pm["gamma_I"], pm["gamma_H"], pm["theta_N"], pm["psi_I"], pm["psi_H"],
            pm["omega_FR"], pm["omega_FS"]]

def run_D90(pm):
    N, phi0 = pm["N"], pm["phi0"]
    frac = 0.0002
    y0 = [(1-phi0)*N*(1-frac),0,(1-phi0)*N*frac,0,0,
          phi0*N*(1-frac),0,phi0*N*frac,0,0,0,0,0]
    pv = p_vec(pm)
    sol = solve_ivp(odes,(0,T_MAX),y0,args=(pv,),t_eval=DAYS,
                    method="RK45",rtol=1e-6,atol=1e-8)
    return sol.y[12,-1]


# ── Panel A: Sobol sensitivity analysis ───────────────────────────────────────
# Vary 7 parameters across plausible ranges; hold others at median
print("Computing Sobol indices (N=512 × 2k+2 = 4608 model runs)...")

problem = {
    "num_vars": 7,
    "names": [r"$\beta_{FR}$", r"$\phi_0$", r"$\beta_I$",
              r"$\gamma_{\rm comm}$", r"$\delta_C$", r"$\alpha$", r"$\theta_N$"],
    "bounds": [
        [1.121, 2.093],    # beta_FR: 95% CrI
        [0.294, 0.490],    # phi0: 95% CrI
        [0.686, 0.964],    # beta_I: 95% CrI
        [0.006, 0.067],    # gamma_comm: 95% CrI
        [0.012, 0.141],    # delta_C: 95% CrI
        [0.011, 0.114],    # alpha: 95% CrI
        [0.024, 0.055],    # theta_N: 95% CrI
    ],
}

X = saltelli.sample(problem, 512, calc_second_order=False)
Y = np.zeros(len(X))

for i, x in enumerate(X):
    if i % 500 == 0:
        print(f"  {i}/{len(X)}")
    pm = dict(P_MED)
    pm["beta_FR"]    = x[0]
    pm["phi0"]       = x[1]
    pm["beta_I"]     = x[2]
    pm["gamma_comm"] = x[3]
    pm["delta_C"]    = x[4]
    pm["alpha"]      = x[5]
    pm["theta_N"]    = x[6]
    Y[i] = run_D90(pm)

Si = sobol_analyze.analyze(problem, Y, calc_second_order=False,
                           print_to_console=False)

print("\nFirst-order Sobol indices:")
for name, s1, conf in zip(problem["names"], Si["S1"], Si["S1_conf"]):
    print(f"  {name:25s}  S1={s1:.3f} ± {conf:.3f}")


# ── Color palette ──────────────────────────────────────────────────────────────
C_CORAL  = "#E05A47"
C_BLUE   = "#3A6EA5"
C_GRAY   = "#9E9E9E"


# ── Figure 4 layout ────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5),
                                gridspec_kw={"width_ratios": [1.1, 1]})
fig.subplots_adjust(wspace=0.38)

# ── Panel A: Tornado plot ──────────────────────────────────────────────────────
order = np.argsort(Si["S1"])[::-1]
names_sorted = [problem["names"][i] for i in order]
s1_sorted    = Si["S1"][order]
ci_sorted    = Si["S1_conf"][order]

colors = []
for s in s1_sorted:
    if s > 0.25:   colors.append(C_CORAL)
    elif s > 0.15: colors.append(C_BLUE)
    else:          colors.append(C_GRAY)

y_pos = np.arange(len(names_sorted))
ax1.barh(y_pos, s1_sorted, color=colors, height=0.6, edgecolor="white", lw=0.4)
ax1.errorbar(s1_sorted, y_pos, xerr=ci_sorted,
             fmt="none", color="black", capsize=3, lw=0.9)
ax1.set_yticks(y_pos)
ax1.set_yticklabels(names_sorted, fontsize=11)
ax1.set_xlabel("First-order Sobol index  $S_i$", fontsize=10)
ax1.set_title("(A)  Sensitivity of $D$(90) to parameter uncertainty",
              fontsize=10, pad=8)
ax1.axvline(0.25, color=C_CORAL, lw=0.8, ls="--", alpha=0.5)
ax1.axvline(0.15, color=C_BLUE,  lw=0.8, ls="--", alpha=0.5)
ax1.set_xlim(-0.02, max(s1_sorted)*1.25)
ax1.invert_yaxis()

legend_patches = [
    mpatches.Patch(color=C_CORAL, label=r"Primary  ($S_i > 0.25$)"),
    mpatches.Patch(color=C_BLUE,  label=r"Secondary  ($S_i \in (0.15, 0.25]$)"),
    mpatches.Patch(color=C_GRAY,  label=r"Minor  ($S_i \leq 0.15$)"),
]
ax1.legend(handles=legend_patches, fontsize=8, loc="lower right")

# ── Panel B: Prior vs Posterior ────────────────────────────────────────────────
draws_df = pd.read_csv(os.path.join(ROOT, "posterior_draws.csv"))

from scipy.stats import norm as sp_norm
from scipy.stats import gamma as sp_gamma_dist

# Show: β_I (data-informed), β_FR (non-identifiable), φ₀ (slight update), γ_comm (prior-dom)
panel_params = [
    (r"$\beta_I$",           "beta_I",     "normal",  0.75, 0.08, [0.40, 1.15],  C_CORAL),
    (r"$\beta_{FR}$",        "beta_FR",    "normal",  1.60, 0.25, [0.80, 2.60],  C_BLUE),
    (r"$\phi_0$",            "phi0",       "normal",  0.38, 0.05, [0.20, 0.58],  "#6BAF8A"),
    (r"$\gamma_{\rm comm}$", "gamma_comm", "gamma",   2.0,  1/80.,[0.0,  0.10],  "#9966CC"),
]

# We map each param to [0,1] x-space for ridge plot
y_offset = [0.0, 0.25, 0.50, 0.75]

for k, (label, col_name, dist, p1, p2, xlim, col) in enumerate(panel_params):
    post = draws_df[col_name].values
    xmin, xmax = xlim
    xs = np.linspace(xmin, xmax, 300)
    xs_norm = (xs - xmin) / (xmax - xmin)   # map to [0,1]

    # Posterior KDE
    kde = gaussian_kde(post, bw_method="scott")
    ys = kde(xs); ys_norm = ys / ys.max()

    # Prior density
    if dist == "normal":
        prior_y = sp_norm.pdf(xs, p1, p2)
    else:
        prior_y = sp_gamma_dist.pdf(xs, p1, scale=p2)
    prior_y_norm = prior_y / prior_y.max()

    offset = y_offset[k]
    ax2.fill_between(xs_norm, offset, offset + ys_norm * 0.22,
                     color=col, alpha=0.45)
    ax2.plot(xs_norm, offset + ys_norm * 0.22, color=col, lw=1.6)
    ax2.plot(xs_norm, offset + prior_y_norm * 0.22, color=col, lw=1.1, ls="--", alpha=0.7)

    # x-axis tick marks at min and max
    ax2.text(0.0, offset - 0.025, f"{xmin:.2g}", fontsize=7,
             ha="center", va="top", color="0.5")
    ax2.text(1.0, offset - 0.025, f"{xmax:.2g}", fontsize=7,
             ha="center", va="top", color="0.5")
    ax2.text(1.03, offset + 0.02, label,
             fontsize=10, ha="left", va="bottom", color=col)

ax2.set_xlim(-0.04, 1.20)
ax2.set_ylim(-0.08, 1.05)
ax2.axis("off")
ax2.set_title("(B)  Prior (dashed) vs posterior (filled) for key parameters",
              fontsize=10, pad=8)

# Shared legend for panel B
post_patch  = mpatches.Patch(facecolor="0.6", alpha=0.5, label="Posterior")
prior_patch = mpatches.Patch(facecolor="white", edgecolor="0.4",
                              linestyle="--", label="Prior")
# Use a line proxy
from matplotlib.lines import Line2D
post_line  = Line2D([0],[0], color="0.4", lw=2,   label="Posterior (filled)")
prior_line = Line2D([0],[0], color="0.4", lw=1.5, ls="--", label="Prior")
ax2.legend(handles=[post_line, prior_line], fontsize=8, loc="lower right",
           bbox_to_anchor=(1.0, 0.0))

# ── Save ──────────────────────────────────────────────────────────────────────
for ext in ["pdf", "png"]:
    path = os.path.join(OUTDIR, f"fig4_sensitivity.{ext}")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    print(f"Saved {path}")

print("Done.")
