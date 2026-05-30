"""
holdout_validation.py
=====================
8/3 hold-out validation for the SEIHRF-OD model.

Split (on the 13-day series: 14–26 May 2026):
  Calibration : days 1–10  (14–23 May) → includes full conflict-peak data
  Validation  : days 11–13 (24–26 May) → post-peak persistent insecurity

Workflow:
  1. Truncate observed series to T_calib = 10 days.
  2. Run Stan MCMC on truncated data → posterior_calib.csv.
     (If posterior_calib.csv already exists the MCMC step is skipped.)
  3. For each posterior draw, integrate the full 13-day ODE (deterministic).
  4. Sample from NegBin2(mu[t], phi_obs) for t = 11, 12, 13.
  5. Compare predictions with held-out observations y_valid = [4, 1, 16].
  6. Report 50%/95% coverage and generate imgs/figS_holdout.pdf/.png.

Reviewer message:
  "We calibrated through the acute conflict peak (days 21–23 May) and
   validated on the subsequent persistent-insecurity phase (days 24–26),
   testing the model's ability to forecast post-peak dynamics from a
   distinct epidemic regime."
"""

from __future__ import annotations
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.stats import nbinom

# ── Paths ──────────────────────────────────────────────────────────────────────
LANCET      = "/Users/selainkaserekakabunga/Documents/Lancet_Paper"
DATA        = os.path.join(LANCET, "data")
STAN_FILE   = os.path.join(LANCET, "seihrf_od.stan")
CALIB_CSV   = os.path.join(LANCET, "posterior_calib.csv")
OUT_PDF     = os.path.join(LANCET, "imgs", "figS_holdout.pdf")
OUT_PNG     = os.path.join(LANCET, "imgs", "figS_holdout.png")

T_CALIB = 10   # calibration days (days 1-10 = 14-23 May)
T_FULL  = 13   # total days in dataset
T_VALID = T_FULL - T_CALIB  # = 3 validation days

# ── Fixed parameters (matches seihrf_od.stan transformed parameters) ──────────
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
X_R = [0.0, 0.30, 17.0, 0.55, 24.0, 0.65, 27.0, 1.00, 30.0, 0.60]

# ── Conflict C(t) ──────────────────────────────────────────────────────────────
def conflict_C(t: float) -> float:
    c = 0.0
    for k in range(5):
        if t >= X_R[2 * k]:
            c = X_R[2 * k + 1]
    return c

# ── ODE RHS (identical to seihrf_od.stan) ─────────────────────────────────────
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
    lam_B = (p["beta_I"]*(IB+IN) + p["beta_H"]*(HB+HN) + p["beta_FS"]*FS) / N
    lam_N = (p["beta_I"]*(IB+IN) + p["beta_H"]*(HB+HN) + p["beta_FR"]*FR) / N

    dSB = -lam_B*SB - mu_BN*SB + mu_NB*SN
    dEB =  lam_B*SB - p["kappa"]*EB
    dIB =  p["kappa"]*EB - (p["theta_B"]+p["delta_I"]+p["gamma_I"])*IB
    dHB =  p["theta_B"]*IB - (p["delta_H"]+p["gamma_H"])*HB
    dRB =  p["gamma_I"]*IB + p["gamma_H"]*HB
    dSN = -lam_N*SN + mu_BN*SB - mu_NB*SN
    dEN =  lam_N*SN - p["kappa"]*EN
    dIN =  p["kappa"]*EN - (p["theta_N"]+p["delta_I"]+p["gamma_I"])*IN
    dHN =  p["theta_N"]*IN - (p["delta_H"]+p["gamma_H"])*HN
    dRN =  p["gamma_I"]*IN + p["gamma_H"]*HN
    dFR =  p["psi_I"]*p["delta_I"]*IN + p["psi_H"]*p["delta_H"]*HN - p["omega_FR"]*FR
    dFS =  (p["delta_I"]*IB + p["delta_H"]*HB
            + (1-p["psi_I"])*p["delta_I"]*IN
            + (1-p["psi_H"])*p["delta_H"]*HN
            - p["omega_FS"]*FS)
    return [dSB, dEB, dIB, dHB, dRB, dSN, dEN, dIN, dHN, dRN, dFR, dFS]

def make_y0(phi0: float) -> list:
    N, sf = FIXED["N_pop"], FIXED["seed_frac"]
    NB, NN = (1-phi0)*N, phi0*N
    return [NB*(1-sf), 0, NB*sf, 0, 0, NN*(1-sf), 0, NN*sf, 0, 0, 0, 0]

def run_ode(row: dict, T: int) -> np.ndarray:
    """Return mu[1..T] = kappa*(E_B + E_N)."""
    p   = {**FIXED, **row}
    y0  = make_y0(row["phi0"])
    sol = solve_ivp(
        fun=lambda t, y: rhs(t, y, p),
        t_span=[0.0, float(T)],
        y0=y0,
        t_eval=np.arange(1, T+1, dtype=float),
        method="RK45", rtol=1e-6, atol=1e-8, max_step=0.5,
    )
    return np.maximum(FIXED["kappa"] * (sol.y[1] + sol.y[6]), 1e-9)

def sample_negbin2(mu: float, phi: float, rng: np.random.Generator) -> int:
    p_nb = phi / (mu + phi)
    return int(rng.negative_binomial(phi, p_nb))

# ── 1. Load full data ─────────────────────────────────────────────────────────
print("Loading data …")
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
y_all   = cases_full["n"].values          # shape (13,)
dates   = cases_full["date"].values
y_calib = y_all[:T_CALIB].tolist()        # [0,13,0,0,20,16,11,21,13,11]
y_valid = y_all[T_CALIB:].tolist()        # [4, 1, 16]

print(f"Calibration ({T_CALIB} days): {y_calib}")
print(f"Validation  ({T_VALID} days): {y_valid}")
print(f"  Calibration total: {sum(y_calib)} cases")
print(f"  Validation  total: {sum(y_valid)} cases")

# ── 2. Run Stan MCMC on calibration data (or load cache) ─────────────────────
PARAM_COLS = ["beta_I", "beta_FR", "phi0", "theta_N",
              "alpha", "gamma_comm", "delta_C", "phi_obs"]

if os.path.exists(CALIB_CSV):
    print(f"\nLoading cached calibration posterior: {CALIB_CSV}")
    draws_df = pd.read_csv(CALIB_CSV)
    print(f"  {len(draws_df)} draws loaded.")
else:
    print("\nRunning Stan MCMC on calibration data (T_calib=10) …")
    print("This will take 20–60 minutes. Output saved to posterior_calib.csv.\n")
    from cmdstanpy import CmdStanModel

    stan_data_calib = {
        "T"           : T_CALIB,
        "y_cases"     : y_calib,
        "N_pop"       : 120_000.0,
        "phi0_obs"    : 0.38,
        "phi0_obs_sd" : 0.05,
        "x_r_conflict": X_R,
        "rel_tol"     : 1e-6,
        "abs_tol"     : 1e-8,
        "max_steps"   : 10_000,
    }

    model = CmdStanModel(stan_file=STAN_FILE)
    fit   = model.sample(
        data            = stan_data_calib,
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

    summary = fit.summary(sig_figs=4)
    key_rows = summary[summary.index.isin(PARAM_COLS + ["R0"])]
    print("\n=== Calibration posterior (T_calib=10) ===")
    # CmdStanPy ≥ 1.1 uses ESS_bulk; older versions use N_Eff
    ess_col = "ESS_bulk" if "ESS_bulk" in summary.columns else "N_Eff"
    display_cols = [c for c in ["Mean", "StdDev", "5%", "50%", "95%", "R_hat", ess_col]
                    if c in summary.columns]
    print(key_rows[display_cols].to_string())

    rhat = summary["R_hat"].dropna()
    ess  = summary[ess_col].dropna()
    print(f"\nMax R-hat: {rhat.max():.4f}  Min ESS: {ess.min():.0f}")

    draws_df = fit.draws_pd(vars=PARAM_COLS)
    draws_df.to_csv(CALIB_CSV, index=False)
    print(f"Calibration draws saved → {CALIB_CSV}")

# ── 3. Forward ODE + NegBin2 sampling for T_FULL days ─────────────────────────
N_DRAWS  = len(draws_df)
N_SAMPLE = min(500, N_DRAWS)
rng      = np.random.default_rng(seed=42)
idx      = rng.choice(N_DRAWS, size=N_SAMPLE, replace=False)
draws    = draws_df.iloc[idx].reset_index(drop=True)

print(f"\nForward ODE for {N_SAMPLE} draws over T_FULL={T_FULL} days …")
mu_full   = np.zeros((N_SAMPLE, T_FULL))
rep_full  = np.zeros((N_SAMPLE, T_FULL), dtype=int)

for i, (_, row) in enumerate(draws[PARAM_COLS].iterrows()):
    if (i + 1) % 100 == 0:
        print(f"  draw {i+1}/{N_SAMPLE}")
    mu_i = run_ode(dict(row), T=T_FULL)
    mu_full[i] = mu_i
    for t in range(T_FULL):
        rep_full[i, t] = sample_negbin2(mu_i[t], row["phi_obs"], rng)

# ── 4. Coverage statistics ─────────────────────────────────────────────────────
def coverage(y_true, rep_mat, lo_q=2.5, hi_q=97.5):
    lo = np.percentile(rep_mat, lo_q, axis=0)
    hi = np.percentile(rep_mat, hi_q, axis=0)
    return np.sum((y_true >= lo) & (y_true <= hi)), len(y_true)

# Calibration coverage
n50_c, nc = coverage(y_calib, rep_full[:, :T_CALIB], 25, 75)
n95_c, _  = coverage(y_calib, rep_full[:, :T_CALIB],  2.5, 97.5)

# Validation coverage (held-out)
n50_v, nv = coverage(y_valid, rep_full[:, T_CALIB:], 25, 75)
n95_v, _  = coverage(y_valid, rep_full[:, T_CALIB:],  2.5, 97.5)

print("\n=== Coverage statistics ===")
print(f"Calibration ({nc} days):  50% CrI {n50_c}/{nc}  |  95% CrI {n95_c}/{nc}")
print(f"Validation  ({nv} days):  50% CrI {n50_v}/{nv}  |  95% CrI {n95_v}/{nv}")

# ── 5. Figure ─────────────────────────────────────────────────────────────────
LANCET_BLUE = "#004E7D"
CORAL       = "#E8735A"
RED_SHADE   = "#D32F2F"
VAL_COLOR   = "#FF8F00"   # amber for validation region

x          = np.arange(T_FULL)
date_labels= [pd.Timestamp(d).strftime("%d %b") for d in dates]

# Quantile arrays
lo95 = np.percentile(rep_full, 2.5,  axis=0)
lo50 = np.percentile(rep_full, 25.0, axis=0)
med  = np.percentile(rep_full, 50.0, axis=0)
hi50 = np.percentile(rep_full, 75.0, axis=0)
hi95 = np.percentile(rep_full, 97.5, axis=0)
mu_m = np.median(mu_full, axis=0)

fig, ax = plt.subplots(figsize=(10, 4.8))

# Calibration / validation shaded backgrounds
ax.axvspan(-0.5, T_CALIB - 0.5,  alpha=0.04, color=LANCET_BLUE, zorder=0)
ax.axvspan(T_CALIB - 0.5, T_FULL - 0.5, alpha=0.06, color=VAL_COLOR, zorder=0)

# Vertical separator
ax.axvline(T_CALIB - 0.5, color="black", lw=1.2, ls="--", zorder=3)

# 95% predictive band (calibration / validation separately for legend clarity)
ax.fill_between(x[:T_CALIB], lo95[:T_CALIB], hi95[:T_CALIB],
                color=LANCET_BLUE, alpha=0.15)
ax.fill_between(x[:T_CALIB], lo50[:T_CALIB], hi50[:T_CALIB],
                color=LANCET_BLUE, alpha=0.30, label="50% CrI (calibration)")
ax.fill_between(x[T_CALIB:], lo95[T_CALIB:], hi95[T_CALIB:],
                color=VAL_COLOR, alpha=0.20, label="95% CrI (forecast)")
ax.fill_between(x[T_CALIB:], lo50[T_CALIB:], hi50[T_CALIB:],
                color=VAL_COLOR, alpha=0.40, label="50% CrI (forecast)")

# Median trajectory
ax.plot(x, mu_m, color=LANCET_BLUE, lw=1.8, ls="--", label="Median µ (ODE)")

# Observed dots — different colour for validation
ax.scatter(x[:T_CALIB], y_calib, color=CORAL, s=60, zorder=5,
           edgecolors="white", lw=0.8, label="Observed — calibration")
ax.scatter(x[T_CALIB:], y_valid, color=VAL_COLOR, s=80, zorder=6,
           edgecolors="black", lw=0.8, marker="D", label="Observed — validation (held-out)")

# Region labels
y_max = max(max(y_all), float(hi95.max()))
ax.text((T_CALIB - 1) / 2, y_max * 0.95,
        "Calibration\n(14–23 May)", ha="center", va="top",
        color=LANCET_BLUE, fontsize=8, style="italic")
ax.text(T_CALIB + (T_VALID - 1) / 2, y_max * 0.95,
        "Validation\n(24–26 May)", ha="center", va="top",
        color=VAL_COLOR, fontsize=8, style="italic")

# Coverage annotation
cov_txt = (
    f"Calibration coverage:\n"
    f"  50% CrI: {n50_c}/{nc} ({100*n50_c//nc}%)  "
    f"  95% CrI: {n95_c}/{nc} ({100*n95_c//nc}%)\n"
    f"Forecast coverage (held-out):\n"
    f"  50% CrI: {n50_v}/{nv} ({100*n50_v//nv if nv else 0}%)  "
    f"  95% CrI: {n95_v}/{nv} ({100*n95_v//nv if nv else 0}%)"
)
ax.text(0.02, 0.97, cov_txt, transform=ax.transAxes,
        fontsize=7.5, va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="gray", alpha=0.9))

# Formatting
ax.set_xticks(x)
ax.set_xticklabels(date_labels, rotation=45, ha="right", fontsize=8)
ax.set_xlabel("Date (INSP SitRep)", fontsize=10)
ax.set_ylabel("Daily confirmed cases", fontsize=10)
ax.set_title(
    "Hold-out validation — SEIHRF-OD (8/3 split)\n"
    f"Calibrated on {T_CALIB} days · Forecasting {T_VALID} held-out days",
    fontsize=9,
)
ax.legend(fontsize=8, loc="upper right", ncol=2)
ax.spines[["top", "right"]].set_visible(False)
ax.set_ylim(bottom=0)
ax.set_xlim(-0.5, T_FULL - 0.5)

fig.tight_layout()
fig.savefig(OUT_PDF, bbox_inches="tight")
fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
plt.close(fig)

print(f"\nFigure saved:\n  {OUT_PDF}\n  {OUT_PNG}")
print("\nDone.")
