"""
compute_scenarios.py
====================
Computes S1/S2/S3/S1+S3 scenario death-averted percentages and R0 values
using the updated MCMC posterior medians (127-case run, SitReps 001-012).

Also computes MCMC-based scenario percentages from posterior_draws.csv.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

# ── Updated posterior-median parameters (MCMC on 127 cases) ──────────────────
P = dict(
    N          = 120_000,
    beta_I     = 0.826,
    beta_H     = 0.06,
    beta_FR    = 1.610,
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
    phi0       = 0.392,
)

T_MAX = 90
DAYS  = np.linspace(0, T_MAX, T_MAX * 10 + 1)


def C_func(t: float, scale: float = 1.0) -> float:
    model_t = t + 21.0
    if model_t < 17.0:    c = 0.30
    elif model_t < 24.0:  c = 0.55
    elif model_t < 27.0:  c = 0.65
    elif model_t <= 29.0: c = 1.00
    else:                  c = 0.60
    return c * scale


def odes(t, y, p, gc_scale=1.0, bFR_override=None, C_scale=1.0, gc_start=None):
    SB, EB, IB, HB, RB, SN, EN, IN, HN, RN, FR, FS, Dcum, Ccum = y
    N = SB + EB + IB + HB + RB + SN + EN + IN + HN + RN
    if N <= 0:
        return [0.0] * 14

    gc_mult = gc_scale if (gc_start is None or t >= gc_start) else 1.0
    gc  = p["gamma_comm"] * gc_mult
    bFR = bFR_override if bFR_override is not None else p["beta_FR"]
    Ct  = C_func(t, scale=C_scale)

    lB = (p["beta_I"]*(IB+IN) + p["beta_H"]*(HB+HN) + p["beta_FS"]*FS) / N
    lN = (p["beta_I"]*(IB+IN) + p["beta_H"]*(HB+HN) + bFR*FR) / N

    Dvis  = (p["delta_I"]*(IB+IN) + p["delta_H"]*(HB+HN)) / N
    phi   = SN / (SB + SN) if (SB + SN) > 0 else p["phi0"]
    mu_BN = p["alpha"]*phi + p["delta_C"]*Ct
    mu_NB = gc + p["beta_D"]*Dvis

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
           + (1-p["psi_I"])*p["delta_I"]*IN
           + (1-p["psi_H"])*p["delta_H"]*HN
           - p["omega_FS"]*FS)

    dDcum = p["delta_I"]*(IB+IN) + p["delta_H"]*(HB+HN)
    dCcum = p["theta_B"]*IB + p["theta_N"]*IN

    return [dSB, dEB, dIB, dHB, dRB,
            dSN, dEN, dIN, dHN, dRN,
            dFR, dFS, dDcum, dCcum]


def initial_state(p):
    N, phi0 = p["N"], p["phi0"]
    frac = 0.0002
    return [(1-phi0)*N*(1-frac), 0, (1-phi0)*N*frac, 0, 0,
            phi0*N*(1-frac),     0, phi0*N*frac,      0, 0,
            0, 0, 0, 0]


def run_scenario(p, gc_scale=1.0, bFR_override=None, C_scale=1.0, gc_start=None):
    sol = solve_ivp(
        odes, (0, T_MAX), initial_state(p),
        args=(p, gc_scale, bFR_override, C_scale, gc_start),
        t_eval=DAYS, method="RK45", rtol=1e-7, atol=1e-9
    )
    return sol.y[12, -1]   # cumulative deaths at day 90


# ── Analytical R0 ─────────────────────────────────────────────────────────────
def analytic_R0(p):
    phi0  = p["phi0"]
    denom_B = p["theta_B"] + p["delta_I"] + p["gamma_I"]
    denom_N = p["theta_N"] + p["delta_I"] + p["gamma_I"]
    gh_sum  = p["delta_H"] + p["gamma_H"]

    R0B = (p["beta_I"] + p["beta_H"] * p["theta_B"] / gh_sum) / denom_B
    R0N = (p["beta_I"] + p["beta_H"] * p["theta_N"] / gh_sum
           + p["beta_FR"] / p["omega_FR"] * (
               p["psi_I"] * p["delta_I"]
               + p["psi_H"] * p["theta_N"] * p["delta_H"] / gh_sum
           )) / denom_N

    trM  = (1 - phi0) * R0B + phi0 * R0N
    burial_term = (p["psi_I"] * p["delta_I"]
                   + p["psi_H"] * p["theta_N"] * p["delta_H"] / gh_sum) / denom_N
    detM = phi0 * (1 - phi0) * R0B * (p["beta_FR"] / p["omega_FR"]) * burial_term
    disc = max(0.0, trM**2 - 4 * detM)
    R0   = (trM + np.sqrt(disc)) / 2
    return R0, R0B, R0N


# ── 1. Plug-in deterministic scenario percentages ────────────────────────────
print("=== Plug-in (posterior-median parameters) ===")
R0, R0B, R0N = analytic_R0(P)
print(f"R0^B = {R0B:.3f},  R0^N = {R0N:.3f}")
print(f"Plugin R0 = {R0:.3f}")

D_base = run_scenario(P)
D_S1   = run_scenario(P, gc_scale=2.0, gc_start=14.0)
D_S2   = run_scenario(P, C_scale=0.5)
D_S3   = run_scenario(P, bFR_override=0.0)
D_S13  = run_scenario(P, gc_scale=2.0, bFR_override=0.0, gc_start=14.0)

s1  = 100 * (D_base - D_S1)  / D_base
s2  = 100 * (D_base - D_S2)  / D_base
s3  = 100 * (D_base - D_S3)  / D_base
s13 = 100 * (D_base - D_S13) / D_base

print(f"\nBase D(90) = {D_base:.0f}")
print(f"S1 (double comm):    D={D_S1:.0f}  → {s1:.0f}% averted")
print(f"S2 (halve conflict): D={D_S2:.0f}  → {s2:.0f}% averted")
print(f"S3 (no reclaim):     D={D_S3:.0f}  → {s3:.0f}% averted")
print(f"S1+S3 combined:      D={D_S13:.0f}  → {s13:.0f}% averted")


# ── 2. MCMC-based scenario percentages from posterior_draws.csv ───────────────
print("\n=== MCMC posterior scenario percentages (from posterior_draws.csv) ===")

DRAWS_PATH = "/Users/selainkaserekakabunga/Documents/Lancet_Paper/posterior_draws.csv"
draws_df = pd.read_csv(DRAWS_PATH)
print(f"Loaded {len(draws_df)} posterior draws")

# Use every 8th draw for speed (1000 draws)
sample = draws_df.iloc[::8].reset_index(drop=True)
print(f"Using {len(sample)} draws for scenario integration")

s1_vals, s2_vals, s3_vals, s13_vals, r0_vals = [], [], [], [], []

for i, row in sample.iterrows():
    if i % 200 == 0:
        print(f"  draw {i}/{len(sample)}...")
    pi = dict(P)
    pi["beta_I"]     = row["beta_I"]
    pi["beta_FR"]    = row["beta_FR"]
    pi["phi0"]       = row["phi0"]
    pi["alpha"]      = row["alpha"]
    pi["gamma_comm"] = row["gamma_comm"]
    pi["delta_C"]    = row["delta_C"]
    pi["theta_N"]    = row["theta_N"]

    r0_i, _, _ = analytic_R0(pi)
    r0_vals.append(r0_i)

    db_i  = run_scenario(pi)
    ds1_i = run_scenario(pi, gc_scale=2.0, gc_start=14.0)
    ds2_i = run_scenario(pi, C_scale=0.5)
    ds3_i = run_scenario(pi, bFR_override=0.0)
    ds13_i = run_scenario(pi, gc_scale=2.0, bFR_override=0.0, gc_start=14.0)

    if db_i > 0:
        s1_vals.append(100*(db_i - ds1_i)/db_i)
        s2_vals.append(100*(db_i - ds2_i)/db_i)
        s3_vals.append(100*(db_i - ds3_i)/db_i)
        s13_vals.append(100*(db_i - ds13_i)/db_i)

r0_arr = np.array(r0_vals)
print(f"\nMCMC R0: median={np.median(r0_arr):.3f}  "
      f"95%CrI [{np.percentile(r0_arr,2.5):.3f}, {np.percentile(r0_arr,97.5):.3f}]")

for label, vals in [("S1", s1_vals), ("S2", s2_vals), ("S3", s3_vals), ("S1+S3", s13_vals)]:
    arr = np.array(vals)
    print(f"{label}: median={np.median(arr):.0f}%  "
          f"95%CrI [{np.percentile(arr,2.5):.0f}%, {np.percentile(arr,97.5):.0f}%]")
