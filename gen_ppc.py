"""
gen_ppc.py
==========
Posterior predictive check (PPC) for the SEIHRF-OD model.

Uses existing posterior_draws.csv (8 000 draws, fitted on all 13 days).
For each draw, integrates the ODE and samples from NegBin2 to produce
the posterior predictive distribution. Compares with observed daily cases.

Output: imgs/figS_ppc.pdf / .png
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
from scipy.stats import nbinom

# ── Paths ──────────────────────────────────────────────────────────────────────
LANCET = "/Users/selainkaserekakabunga/Documents/Lancet_Paper"
DRAWS  = os.path.join(LANCET, "posterior_draws.csv")
DATA   = os.path.join(LANCET, "data")
OUT_PDF = os.path.join(LANCET, "imgs", "figS_ppc.pdf")
OUT_PNG = os.path.join(LANCET, "imgs", "figS_ppc.png")

# ── Fixed parameters (Stan transformed parameters) ────────────────────────────
FIXED = dict(
    beta_H   = 0.06,
    beta_FS  = 0.002,
    kappa    = 1.0 / 9.0,
    theta_B  = 0.28,
    delta_I  = 0.18,
    delta_H  = 0.12,
    gamma_I  = 0.09,
    gamma_H  = 0.10,
    psi_I    = 0.45,
    psi_H    = 0.15,
    omega_FR = 0.80,
    omega_FS = 3.00,
    beta_D   = 8.00,
    N_pop    = 120_000.0,
    seed_frac= 2e-4,
)

# Conflict anchors: [start_day, level] × 5
X_R = [0.0, 0.30, 17.0, 0.55, 24.0, 0.65, 27.0, 1.00, 30.0, 0.60]

# ── Conflict intensity C(t) ────────────────────────────────────────────────────
def conflict_C(t: float) -> float:
    c = 0.0
    for k in range(5):
        if t >= X_R[2 * k]:
            c = X_R[2 * k + 1]
    return c

# ── ODE right-hand side (identical to seihrf_od.stan) ─────────────────────────
def rhs(t: float, y: np.ndarray, p: dict) -> list:
    SB, EB, IB, HB, RB, SN, EN, IN, HN, RN, FR, FS = y
    N = SB + EB + IB + HB + RB + SN + EN + IN + HN + RN
    if N <= 0:
        return [0.0] * 12
    Ct    = conflict_C(t)
    S_sum = SB + SN
    phi   = SN / S_sum if S_sum > 0 else p["phi0"]
    Dvis  = (p["delta_I"] * (IB + IN) + p["delta_H"] * (HB + HN)) / N
    mu_BN = p["alpha"] * phi + p["delta_C"] * Ct
    mu_NB = p["gamma_comm"] + p["beta_D"] * Dvis
    lam_B = (p["beta_I"] * (IB + IN) + p["beta_H"] * (HB + HN) + p["beta_FS"] * FS) / N
    lam_N = (p["beta_I"] * (IB + IN) + p["beta_H"] * (HB + HN) + p["beta_FR"] * FR) / N

    dSB = -lam_B * SB - mu_BN * SB + mu_NB * SN
    dEB =  lam_B * SB - p["kappa"] * EB
    dIB =  p["kappa"] * EB - (p["theta_B"] + p["delta_I"] + p["gamma_I"]) * IB
    dHB =  p["theta_B"] * IB - (p["delta_H"] + p["gamma_H"]) * HB
    dRB =  p["gamma_I"] * IB + p["gamma_H"] * HB
    dSN = -lam_N * SN + mu_BN * SB - mu_NB * SN
    dEN =  lam_N * SN - p["kappa"] * EN
    dIN =  p["kappa"] * EN - (p["theta_N"] + p["delta_I"] + p["gamma_I"]) * IN
    dHN =  p["theta_N"] * IN - (p["delta_H"] + p["gamma_H"]) * HN
    dRN =  p["gamma_I"] * IN + p["gamma_H"] * HN
    dFR =  p["psi_I"] * p["delta_I"] * IN + p["psi_H"] * p["delta_H"] * HN - p["omega_FR"] * FR
    dFS =  (p["delta_I"] * IB + p["delta_H"] * HB
            + (1 - p["psi_I"]) * p["delta_I"] * IN
            + (1 - p["psi_H"]) * p["delta_H"] * HN
            - p["omega_FS"] * FS)
    return [dSB, dEB, dIB, dHB, dRB, dSN, dEN, dIN, dHN, dRN, dFR, dFS]

# ── Initial conditions ─────────────────────────────────────────────────────────
def make_y0(phi0: float) -> list:
    N    = FIXED["N_pop"]
    sf   = FIXED["seed_frac"]
    NB   = (1 - phi0) * N
    NN   = phi0 * N
    return [NB*(1-sf), 0, NB*sf, 0, 0,
            NN*(1-sf), 0, NN*sf, 0, 0,
            0, 0]

# ── Run ODE for one parameter draw ────────────────────────────────────────────
def run_ode(row: dict, T: int = 13) -> np.ndarray:
    """Return mu[1..T] = kappa*(E_B + E_N) at integer time points."""
    p = {**FIXED, **row}
    y0 = make_y0(row["phi0"])
    sol = solve_ivp(
        fun=lambda t, y: rhs(t, y, p),
        t_span=[0.0, float(T)],
        y0=y0,
        t_eval=np.arange(1, T + 1, dtype=float),
        method="RK45",
        rtol=1e-6,
        atol=1e-8,
        max_step=0.5,
    )
    kappa = p["kappa"]
    mu = kappa * (sol.y[1] + sol.y[6])   # kappa*(E_B + E_N)
    return np.maximum(mu, 1e-9)

# ── NegBin2 sample: mean=mu, var=mu + mu²/phi ─────────────────────────────────
def sample_negbin2(mu: float, phi: float, rng: np.random.Generator) -> int:
    p_nb = phi / (mu + phi)
    return int(rng.negative_binomial(phi, p_nb))

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading case data …")
cases_raw = pd.read_csv(
    os.path.join(DATA, "insp_sitrep__new_confirmed_cases__daily.csv"),
    dtype={"nom": str, "new_confirmed_cases": str},
)
cases_raw["date"] = pd.to_datetime(cases_raw["date"], dayfirst=True)
cases_raw["new_confirmed_cases"] = pd.to_numeric(
    cases_raw["new_confirmed_cases"], errors="coerce"
).fillna(0)
cases_daily = (
    cases_raw.groupby("date")["new_confirmed_cases"]
    .sum().sort_index().reset_index()
    .rename(columns={"new_confirmed_cases": "n"})
)
all_dates = pd.date_range(cases_daily["date"].min(),
                          cases_daily["date"].max(), freq="D")
cases_full = (
    pd.DataFrame({"date": all_dates})
    .merge(cases_daily, on="date", how="left")
    .fillna({"n": 0})
)
cases_full["n"] = cases_full["n"].astype(int)
y_obs   = cases_full["n"].values        # shape (13,)
dates   = cases_full["date"].values
T       = len(y_obs)
print(f"T={T}  dates: {dates[0].astype('datetime64[D]')} → {dates[-1].astype('datetime64[D]')}")
print(f"y_obs = {y_obs.tolist()}")

# ── Load posterior draws ───────────────────────────────────────────────────────
print("\nLoading posterior draws …")
draws_df = pd.read_csv(DRAWS)
N_DRAWS  = len(draws_df)
print(f"  {N_DRAWS} draws loaded.")

# Subsample for speed (500 draws sufficient for smooth quantiles)
N_SAMPLE = min(500, N_DRAWS)
rng      = np.random.default_rng(seed=42)
idx      = rng.choice(N_DRAWS, size=N_SAMPLE, replace=False)
draws    = draws_df.iloc[idx].reset_index(drop=True)

# ── Run PPC ───────────────────────────────────────────────────────────────────
PARAM_COLS = ["beta_I", "beta_FR", "phi0", "theta_N",
              "alpha", "gamma_comm", "delta_C", "phi_obs"]

print(f"\nRunning ODE for {N_SAMPLE} draws …")
mu_matrix  = np.zeros((N_SAMPLE, T))
rep_matrix = np.zeros((N_SAMPLE, T), dtype=int)

for i, (_, row) in enumerate(draws[PARAM_COLS].iterrows()):
    if (i + 1) % 100 == 0:
        print(f"  draw {i+1}/{N_SAMPLE}")
    mu_i = run_ode(dict(row), T=T)
    mu_matrix[i] = mu_i
    for t in range(T):
        rep_matrix[i, t] = sample_negbin2(mu_i[t], row["phi_obs"], rng)

# Quantiles of posterior predictive distribution
ppc_lo95  = np.percentile(rep_matrix, 2.5,  axis=0)
ppc_lo50  = np.percentile(rep_matrix, 25.0, axis=0)
ppc_med   = np.percentile(rep_matrix, 50.0, axis=0)
ppc_hi50  = np.percentile(rep_matrix, 75.0, axis=0)
ppc_hi95  = np.percentile(rep_matrix, 97.5, axis=0)

# Quantiles of mu (deterministic component)
mu_med   = np.median(mu_matrix, axis=0)
mu_lo95  = np.percentile(mu_matrix, 2.5,  axis=0)
mu_hi95  = np.percentile(mu_matrix, 97.5, axis=0)

# Coverage statistics
in50 = np.sum((y_obs >= ppc_lo50) & (y_obs <= ppc_hi50))
in95 = np.sum((y_obs >= ppc_lo95) & (y_obs <= ppc_hi95))
print(f"\nPPC coverage:")
print(f"  50% CrI: {in50}/{T} points ({100*in50/T:.0f}%)  [expected ~50%]")
print(f"  95% CrI: {in95}/{T} points ({100*in95/T:.0f}%)  [expected ~95%]")

# ── Plot ───────────────────────────────────────────────────────────────────────
LANCET_BLUE   = "#004E7D"
CORAL         = "#E8735A"
AMBER         = "#D4820A"
RED_SHADE     = "#D32F2F"

x = np.arange(T)
date_labels = [pd.Timestamp(d).strftime("%d %b") for d in dates]

fig, ax = plt.subplots(figsize=(9, 4.5))

# Posterior predictive bands
ax.fill_between(x, ppc_lo95, ppc_hi95,
                color=LANCET_BLUE, alpha=0.15, label="95% posterior predictive")
ax.fill_between(x, ppc_lo50, ppc_hi50,
                color=LANCET_BLUE, alpha=0.30, label="50% posterior predictive")

# Posterior median of mu (smooth expected trajectory)
ax.plot(x, mu_med, color=LANCET_BLUE, lw=1.8, ls="--",
        label="Posterior median (µ)")

# Observed data
ax.scatter(x, y_obs, color=CORAL, s=55, zorder=5,
           edgecolors="white", linewidths=0.8,
           label="Observed (INSP SitReps)")

# Conflict events — thin vertical shading for t > 16 (anchor 2 onwards)
# These are outside the calibration window but shown for context
for start, end, label_txt in [(17, 23.5, "Conflict\nevents")]:
    if start <= T:
        ax.axvspan(min(start, T-0.5), min(end, T-0.5),
                   alpha=0.08, color=RED_SHADE, zorder=0)

# Coverage annotation
cov_text = (f"PPC coverage\n"
            f"50% CrI: {in50}/{T} ({100*in50/T:.0f}%)\n"
            f"95% CrI: {in95}/{T} ({100*in95/T:.0f}%)")
ax.text(0.02, 0.97, cov_text, transform=ax.transAxes,
        fontsize=8, va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.85))

# Formatting
ax.set_xticks(x)
ax.set_xticklabels(date_labels, rotation=45, ha="right", fontsize=8)
ax.set_xlabel("Date (INSP SitRep)", fontsize=10)
ax.set_ylabel("Daily confirmed cases", fontsize=10)
ax.set_title(
    "Posterior predictive check — SEIHRF-OD (INRB-UMIE/Ebola_DRC_2026, build 13d78cb)\n"
    f"Calibration: all {T} days (14–26 May 2026, {N_DRAWS} posterior draws)",
    fontsize=9,
)
ax.legend(fontsize=8, loc="upper right")
ax.spines[["top", "right"]].set_visible(False)
ax.set_ylim(bottom=0)
ax.set_xlim(-0.5, T - 0.5)

fig.tight_layout()
fig.savefig(OUT_PDF, bbox_inches="tight")
fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nFigure saved:\n  {OUT_PDF}\n  {OUT_PNG}")
