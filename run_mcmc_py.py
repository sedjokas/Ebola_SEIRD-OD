"""
run_mcmc_py.py
==============
Python equivalent of run_mcmc.R using CmdStanPy.
Calibrates the SEIHRF-OD model on the latest INRB-UMIE data
(SitReps 001-012, data freeze 26 May 2026, build 13d78cb).

Requirements (already installed):
    cmdstanpy == 1.3.0
    numpy, scipy, pandas
"""

from __future__ import annotations
import os
import json
import numpy as np
import pandas as pd
from cmdstanpy import CmdStanModel

# ── Paths ──────────────────────────────────────────────────────────────────────
LANCET = "/Users/selainkaserekakabunga/Documents/Lancet_Paper"
DATA   = os.path.join(LANCET, "data")
STAN   = os.path.join(LANCET, "seihrf_od.stan")

# ── 1. Load and prepare case data ─────────────────────────────────────────────
print("Loading data …")

cases_raw = pd.read_csv(
    os.path.join(DATA, "insp_sitrep__new_confirmed_cases__daily.csv"),
    dtype={"nom": str, "new_confirmed_cases": str}
)
cases_raw["date"] = pd.to_datetime(cases_raw["date"], dayfirst=True)
cases_raw["new_confirmed_cases"] = pd.to_numeric(
    cases_raw["new_confirmed_cases"], errors="coerce"
).fillna(0)

cases_daily = (
    cases_raw.groupby("date")["new_confirmed_cases"]
    .sum()
    .sort_index()
    .reset_index()
    .rename(columns={"new_confirmed_cases": "n"})
)

# Fill contiguous date gaps with 0
all_dates = pd.date_range(cases_daily["date"].min(), cases_daily["date"].max(), freq="D")
cases_full = (
    pd.DataFrame({"date": all_dates})
    .merge(cases_daily, on="date", how="left")
    .fillna({"n": 0})
)
cases_full["n"] = cases_full["n"].astype(int)

y_cases = cases_full["n"].tolist()
T = len(y_cases)

print(f"T = {T} days | Total cases = {sum(y_cases)}")
print(f"Date range: {cases_full['date'].iloc[0].date()} → {cases_full['date'].iloc[-1].date()}")
print(f"Daily series: {y_cases}")

# ── 2. Compute phi0 from contact-tracing proxy ────────────────────────────────
# phi0_obs = 1 - mean(r_c) over first 5 reporting days
# r_c = cumulative_contacts_isolated / cumulative_contacts_listed
# Use fixed value from manuscript (first 5 SitReps proxy unchanged)
phi0_obs    = 0.38
phi0_obs_sd = 0.05

# ── 3. Build Stan data dictionary ─────────────────────────────────────────────
stan_data = {
    "T":           T,
    "y_cases":     y_cases,
    "N_pop":       120_000.0,
    "phi0_obs":    phi0_obs,
    "phi0_obs_sd": phi0_obs_sd,
    # Five conflict anchors: [start_day, level] × 5
    "x_r_conflict": [
         0.0, 0.30,   # anchor 1: pre-epidemic baseline
        17.0, 0.55,   # anchor 2: Nyankunde exposure, 11 May
        24.0, 0.65,   # anchor 3: CDC announcement, 18 May
        27.0, 1.00,   # anchor 4: Rwampara/Mongbwalu peak, 21-23 May
        30.0, 0.60,   # anchor 5: persistent insecurity
    ],
    "rel_tol":   1e-6,
    "abs_tol":   1e-8,
    "max_steps": 10_000,
}

# Save for reproducibility
with open(os.path.join(LANCET, "stan_data.json"), "w") as f:
    json.dump(stan_data, f, indent=2)
print("Stan data saved → stan_data.json")

# ── 4. Compile Stan model ─────────────────────────────────────────────────────
print("\nCompiling Stan model …")
model = CmdStanModel(stan_file=STAN)
print("Compilation complete.")

# ── 5. Run MCMC ───────────────────────────────────────────────────────────────
print("\nRunning MCMC (4 chains × 2000 warmup + 2000 sampling) …")
print("This will take 30–90 minutes depending on CPU speed.\n")

fit = model.sample(
    data            = stan_data,
    chains          = 4,
    parallel_chains = 4,
    iter_warmup     = 2000,
    iter_sampling   = 2000,
    adapt_delta     = 0.95,
    max_treedepth   = 12,
    seed            = 42,
    show_progress   = True,
    output_dir      = LANCET,
)

# ── 6. Convergence diagnostics ────────────────────────────────────────────────
print("\n=== Convergence summary ===")
params = ["beta_I", "beta_FR", "phi0", "theta_N", "alpha",
          "gamma_comm", "delta_C", "phi_obs", "R0"]
summary = fit.summary(sig_figs=4)

# Filter to key parameters
key_rows = summary[summary.index.isin(params)]
print(key_rows[["Mean", "StdDev", "5%", "50%", "95%", "R_hat", "N_Eff"]].to_string())

rhat_vals = summary["R_hat"].dropna()
ess_vals  = summary["N_Eff"].dropna()
print(f"\nMax R-hat: {rhat_vals.max():.4f}  (should be < 1.02)")
print(f"Min ESS:   {ess_vals.min():.0f}  (should be > 400)")

# ── 7. Export posterior draws ─────────────────────────────────────────────────
draws_df = fit.draws_pd(vars=params)
out_csv  = os.path.join(LANCET, "posterior_draws.csv")
draws_df.to_csv(out_csv, index=False)
print(f"\nPosterior draws saved → {out_csv}")
print(f"Shape: {draws_df.shape[0]} draws × {draws_df.shape[1]} columns")

# ── 8. Quick posterior summaries for manuscript ───────────────────────────────
print("\n=== Posterior medians and 95% CrI ===")
for p in params:
    if p in draws_df.columns:
        vals = draws_df[p]
        print(f"  {p:<15} median={vals.median():.3f}  "
              f"95%CrI [{vals.quantile(0.025):.3f}, {vals.quantile(0.975):.3f}]")

print("\nDone. Share the output above so figures can be regenerated.")
