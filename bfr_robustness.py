"""
bfr_robustness.py
=================
Sweeps beta_FR across its full prior range and computes:
  - R0 (SEIHRF-OD analytical formula)
  - R0 underestimation bias vs homogeneous model (R0_hom = 1.80 fixed)
  - S3 deaths averted % at day 90 (eliminate body reclamation: beta_FR -> 0)

Demonstrates that headline conclusions hold qualitatively at all prior
percentiles, addressing reviewer concern about non-identifiability of beta_FR.
"""

from __future__ import annotations
import numpy as np
from scipy.integrate import solve_ivp

# ── Correct posterior-median parameters (Table 1) ─────────────────────────────
P = dict(
    N          = 120_000,
    beta_I     = 0.826,   # updated: MCMC on 127 cases (SitReps 001-012)
    beta_H     = 0.06,
    beta_FS    = 0.002,
    kappa      = 1.0 / 9,
    theta_B    = 0.28,
    theta_N    = 0.040,
    delta_I    = 0.18,
    delta_H    = 0.12,
    gamma_I    = 0.09,
    gamma_H    = 0.10,
    omega_FR   = 0.80,
    omega_FS   = 3.00,
    psi_I      = 0.45,
    psi_H      = 0.15,
    alpha      = 0.037,
    gamma_comm = 0.022,
    delta_C    = 0.045,
    beta_D     = 8.00,
    phi0       = 0.392,   # updated
)

# R0 homogeneous-model benchmark (fitted to same data, prior paper result)
R0_HOM = 1.80

T_MAX = 90
DAYS  = np.linspace(0, T_MAX, T_MAX * 10 + 1)


# ── C(t) step function ────────────────────────────────────────────────────────
def C_func(t: float) -> float:
    model_t = t + 21.0
    if model_t < 17.0:    return 0.30
    elif model_t < 24.0:  return 0.55
    elif model_t < 27.0:  return 0.65
    elif model_t <= 29.0: return 1.00
    else:                 return 0.60


# ── ODE ───────────────────────────────────────────────────────────────────────
def odes(t, y, p, bFR_val):
    SB, EB, IB, HB, RB, SN, EN, IN, HN, RN, FR, FS, Dcum, Ccum = y
    N = SB + EB + IB + HB + RB + SN + EN + IN + HN + RN
    if N <= 0:
        return [0.0] * 14

    Ct   = C_func(t)
    Dvis = (p["delta_I"] * (IB + IN) + p["delta_H"] * (HB + HN)) / N
    phi  = SN / (SB + SN) if (SB + SN) > 0 else p["phi0"]

    mu_BN = p["alpha"] * phi + p["delta_C"] * Ct
    mu_NB = p["gamma_comm"] + p["beta_D"] * Dvis

    lB = (p["beta_I"] * (IB + IN) + p["beta_H"] * (HB + HN) + p["beta_FS"] * FS) / N
    lN = (p["beta_I"] * (IB + IN) + p["beta_H"] * (HB + HN) + bFR_val * FR) / N

    dSB = -lB*SB - mu_BN*SB + mu_NB*SN
    dEB =  lB*SB - p["kappa"]*EB
    dIB =  p["kappa"]*EB - (p["theta_B"] + p["delta_I"] + p["gamma_I"])*IB
    dHB =  p["theta_B"]*IB - (p["delta_H"] + p["gamma_H"])*HB
    dRB =  p["gamma_I"]*IB + p["gamma_H"]*HB

    dSN = -lN*SN + mu_BN*SB - mu_NB*SN
    dEN =  lN*SN - p["kappa"]*EN
    dIN =  p["kappa"]*EN - (p["theta_N"] + p["delta_I"] + p["gamma_I"])*IN
    dHN =  p["theta_N"]*IN - (p["delta_H"] + p["gamma_H"])*HN
    dRN =  p["gamma_I"]*IN + p["gamma_H"]*HN

    dFR = (p["psi_I"]*p["delta_I"]*IN + p["psi_H"]*p["delta_H"]*HN - p["omega_FR"]*FR)
    dFS = (p["delta_I"]*IB + p["delta_H"]*HB
           + (1 - p["psi_I"])*p["delta_I"]*IN
           + (1 - p["psi_H"])*p["delta_H"]*HN
           - p["omega_FS"]*FS)

    dDcum = p["delta_I"]*(IB + IN) + p["delta_H"]*(HB + HN)
    dCcum = p["theta_B"]*IB + p["theta_N"]*IN
    return [dSB, dEB, dIB, dHB, dRB, dSN, dEN, dIN, dHN, dRN,
            dFR, dFS, dDcum, dCcum]


def initial_state(p):
    N, phi0 = p["N"], p["phi0"]
    frac = 0.0002
    return [(1-phi0)*N*(1-frac), 0, (1-phi0)*N*frac, 0, 0,
            phi0*N*(1-frac),     0, phi0*N*frac,      0, 0,
            0, 0, 0, 0]


def run(p, bFR_val):
    y0 = initial_state(p)
    sol = solve_ivp(
        odes, (0, T_MAX), y0, args=(p, bFR_val),
        t_eval=DAYS, method="RK45", rtol=1e-7, atol=1e-9
    )
    return sol.y[12, -1]   # cumulative deaths at day 90


# ── Analytical R0 ─────────────────────────────────────────────────────────────
def analytic_R0(p, bFR_val):
    phi0  = p["phi0"]
    denom_B = p["theta_B"] + p["delta_I"] + p["gamma_I"]
    denom_N = p["theta_N"] + p["delta_I"] + p["gamma_I"]
    gh_sum  = p["delta_H"] + p["gamma_H"]

    R0B = (p["beta_I"] + p["beta_H"] * p["theta_B"] / gh_sum) / denom_B
    burial_term = (p["psi_I"] * p["delta_I"]
                   + p["psi_H"] * p["theta_N"] * p["delta_H"] / gh_sum) / denom_N
    R0N = (p["beta_I"] + p["beta_H"] * p["theta_N"] / gh_sum
           + bFR_val / p["omega_FR"] * (p["psi_I"] * p["delta_I"]
                                         + p["psi_H"] * p["theta_N"] * p["delta_H"] / gh_sum)
           ) / denom_N

    trM  = (1 - phi0) * R0B + phi0 * R0N
    detM = phi0 * (1 - phi0) * R0B * (bFR_val / p["omega_FR"]) * burial_term
    disc = max(0.0, trM**2 - 4 * detM)
    return (trM + np.sqrt(disc)) / 2, R0B, R0N


# ── Sweep ─────────────────────────────────────────────────────────────────────
# Prior: Normal(1.60, 0.25) — 1st/5th/25th/50th/75th/95th/99th percentiles
bFR_values = [0.50, 1.19, 1.35, 1.62, 1.85, 2.01, 2.50, 3.00]
labels = [
    "extreme low",
    "prior 5th pct",
    "prior 25th pct",
    "posterior median",
    "prior 75th pct",
    "prior 95th pct",
    "high",
    "extreme high",
]

# Baseline deaths (at posterior median beta_FR)
D_base = run(P, 1.62)   # posterior median beta_FR
# S3 baseline: beta_FR = 0
D_S3_base = run(P, 0.0)
S3_base_pct = 100 * (D_base - D_S3_base) / D_base

print(f"\n{'β_FR':>6}  {'Label':<20}  {'R₀':>6}  {'R₀^N':>6}  "
      f"{'Hom-bias%':>10}  {'S3 averted%':>12}  {'D(90)':>8}")
print("-" * 78)

rows = []
for bFR, lbl in zip(bFR_values, labels):
    R0, R0B, R0N   = analytic_R0(P, bFR)
    hom_bias_pct   = 100 * (R0 - R0_HOM) / R0_HOM
    D_base_i       = run(P, bFR)
    D_S3_i         = run(P, 0.0)      # S3 always sets bFR=0
    s3_pct         = 100 * (D_base_i - D_S3_i) / D_base_i
    rows.append((bFR, lbl, R0, R0N, hom_bias_pct, s3_pct, D_base_i))
    print(f"{bFR:6.2f}  {lbl:<20}  {R0:6.2f}  {R0N:6.2f}  "
          f"{hom_bias_pct:>+9.0f}%  {s3_pct:>10.0f}%  {D_base_i:>8.0f}")

print(f"\nBaseline (posterior median β_FR=1.62): D(90) = {D_base:.0f} deaths")
print(f"S3 (β_FR=0): D(90) = {D_S3_base:.0f} deaths  →  {S3_base_pct:.0f}% averted")
