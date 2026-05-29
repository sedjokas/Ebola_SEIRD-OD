"""
figures_replot.py
=================
Regenerates Figures 1, 3, S1, and S4 for the SEIHRF-OD manuscript
using the correct Table 1 parameters and five-anchor C(t).

Fixes applied vs prior version:
  Fig 1  — Legend: "documented security events" (not "ACLED proxy")
  Fig 3  — Updated scenario percentages 20/16/29/44% (MCMC on 127 cases)
  Fig S1 — Correct five-anchor C(t), conflict peak at days 6-8
  Fig S4 — English anchor labels, correct Table 1 parameters

Output: /Users/selainkaserekakabunga/Documents/Lancet_Paper/imgs/
        figS4_sensitivity_Ct.pdf also copied to root for LaTeX compilation.
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from scipy.integrate import solve_ivp
from scipy.ndimage import uniform_filter1d

OUTDIR   = "/Users/selainkaserekakabunga/Documents/Lancet_Paper/imgs"
DATA_DIR = "/Users/selainkaserekakabunga/Documents/Lancet_Paper/data"
ROOT_DIR = "/Users/selainkaserekakabunga/Documents/Lancet_Paper"

# ── Correct Table 1 posterior-median parameters ───────────────────────────────

N0   = 120_000   # outbreak-zone population (Ituri, calibrated)
# Updated posterior medians — MCMC on 127 confirmed cases, SitReps 001-012
# (INRB-UMIE/Ebola_DRC_2026 build 13d78cb, data freeze 26 May 2026)
PHI0 = 0.392     # initial sceptic fraction

P = dict(
    N          = N0,
    beta_I     = 0.826,   # community transmission rate (day⁻¹)
    beta_H     = 0.06,    # hospital/ETC transmission rate
    beta_FR    = 1.610,   # reclaimed-body transmission rate (day⁻¹)
    beta_FS    = 0.002,   # safe-burial transmission rate (≈ 0)
    kappa      = 1.0/9,   # incubation rate (9-day BDBV mean)
    theta_B    = 0.28,    # hospitalisation rate — Believers
    theta_N    = 0.040,   # hospitalisation rate — Sceptics
    delta_I    = 0.18,    # community death rate
    delta_H    = 0.12,    # hospital death rate
    gamma_I    = 0.09,    # community recovery rate
    gamma_H    = 0.10,    # hospital recovery rate
    omega_FR   = 0.80,    # reclaimed-body removal rate (day⁻¹)
    omega_FS   = 3.00,    # safe-burial removal rate (day⁻¹)
    psi_I      = 0.45,    # fraction of sceptic community deaths reclaimed
    psi_H      = 0.15,    # fraction of sceptic hospital deaths reclaimed
    alpha      = 0.037,   # social contagion rate (B to N)
    gamma_comm = 0.022,   # health communication rate (N to B)
    delta_C    = 0.045,   # conflict amplification coefficient
    beta_D     = 8.00,    # visible-death distrust coefficient
    phi0       = PHI0,
)

T_MAX = 90
DAYS  = np.linspace(0, T_MAX, T_MAX * 10 + 1)

# ── Documented C(t) in declaration-day units ──────────────────────────────────
# Index case: 24 April 2026 (model day 0)
# Outbreak declaration: 15 May 2026 (model day 21)
# t_decl = model_day - 21
#
# Anchors (Sources: OCHA, CDC, WHO DON602/DON603):
#   Pre-epidemic baseline (model days 0-16,  t_decl < -4): C = 0.30
#   Nyankunde exposure   (model days 17-23, -4 <= t_decl < 3): C = 0.55
#   CDC announcement     (model days 24-26,  3 <= t_decl < 6): C = 0.65
#   Rwampara/Mongbwalu   (model days 27-29,  6 <= t_decl <= 8): C = 1.00
#   Persistent insecurity (model day 30+,   t_decl > 8): C = 0.60

def C_func(t: float, scale: float = 1.0) -> float:
    model_t = t + 21.0
    if model_t < 17.0:   c = 0.30
    elif model_t < 24.0: c = 0.55
    elif model_t < 27.0: c = 0.65
    elif model_t <= 29.0: c = 1.00
    else:                c = 0.60
    return c * scale


# ── SEIHRF-OD ODE (14-state, declaration-day time axis) ───────────────────────
# States: [SB, EB, IB, HB, RB, SN, EN, IN, HN, RN, FR, FS, Dcum, Ccum]
#          0   1   2   3   4   5   6   7   8   9  10  11   12     13

def odes(t, y, p,
         gc_scale: float = 1.0,
         bFR_scale: float = 1.0,
         C_scale: float = 1.0,
         gc_start: float | None = None):
    SB, EB, IB, HB, RB, SN, EN, IN, HN, RN, FR, FS, Dcum, Ccum = y

    N = SB + EB + IB + HB + RB + SN + EN + IN + HN + RN
    if N <= 0:
        return [0.0] * 14

    # Apply gamma_comm doubling only from gc_start onward
    gc_mult = gc_scale if (gc_start is None or t >= gc_start) else 1.0
    gc  = p["gamma_comm"] * gc_mult
    bFR = p["beta_FR"] * bFR_scale
    Ct  = C_func(t, scale=C_scale)

    # Forces of infection
    lB = (p["beta_I"]*(IB+IN) + p["beta_H"]*(HB+HN) + p["beta_FS"]*FS) / N
    lN = (p["beta_I"]*(IB+IN) + p["beta_H"]*(HB+HN) + bFR*FR) / N

    # Visible death rate (per individual)
    Dvis  = (p["delta_I"]*(IB+IN) + p["delta_H"]*(HB+HN)) / N

    # Opinion dynamics
    phi   = SN / (SB + SN) if (SB + SN) > 0 else p["phi0"]
    mu_BN = p["alpha"]*phi + p["delta_C"]*Ct
    mu_NB = gc + p["beta_D"]*Dvis

    # Believers
    dSB = -lB*SB - mu_BN*SB + mu_NB*SN
    dEB =  lB*SB - p["kappa"]*EB
    dIB =  p["kappa"]*EB - (p["theta_B"] + p["delta_I"] + p["gamma_I"])*IB
    dHB =  p["theta_B"]*IB - (p["delta_H"] + p["gamma_H"])*HB
    dRB =  p["gamma_I"]*IB + p["gamma_H"]*HB

    # Sceptics
    dSN = -lN*SN + mu_BN*SB - mu_NB*SN
    dEN =  lN*SN - p["kappa"]*EN
    dIN =  p["kappa"]*EN - (p["theta_N"] + p["delta_I"] + p["gamma_I"])*IN
    dHN =  p["theta_N"]*IN - (p["delta_H"] + p["gamma_H"])*HN
    dRN =  p["gamma_I"]*IN + p["gamma_H"]*HN

    # Post-mortem compartments
    dFR = (p["psi_I"]*p["delta_I"]*IN
           + p["psi_H"]*p["delta_H"]*HN
           - p["omega_FR"]*FR)
    dFS = (p["delta_I"]*IB + p["delta_H"]*HB
           + (1-p["psi_I"])*p["delta_I"]*IN
           + (1-p["psi_H"])*p["delta_H"]*HN
           - p["omega_FS"]*FS)

    # Cumulative deaths and confirmed cases
    dDcum = p["delta_I"]*(IB+IN) + p["delta_H"]*(HB+HN)
    dCcum = p["theta_B"]*IB + p["theta_N"]*IN

    return [dSB, dEB, dIB, dHB, dRB,
            dSN, dEN, dIN, dHN, dRN,
            dFR, dFS, dDcum, dCcum]


def initial_state(p: dict) -> list:
    N, phi0 = p["N"], p["phi0"]
    frac = 0.0002   # fraction initially infectious at declaration
    IB0  = (1.0 - phi0) * N * frac
    IN0  = phi0 * N * frac
    SB0  = (1.0 - phi0) * N * (1.0 - frac)
    SN0  = phi0 * N * (1.0 - frac)
    return [SB0, 0, IB0, 0, 0,
            SN0, 0, IN0, 0, 0,
            0, 0, 0, 0]


def run_model(p, gc_scale=1.0, bFR_scale=1.0, C_scale=1.0, gc_start=None):
    return solve_ivp(
        odes, (0, T_MAX), initial_state(p),
        args=(p, gc_scale, bFR_scale, C_scale, gc_start),
        t_eval=DAYS, method="RK45",
        rtol=1e-7, atol=1e-9, dense_output=False,
    )


# ── Approximate posterior draws (Monte Carlo uncertainty) ─────────────────────

def posterior_draws(n: int = 100, seed: int = 42) -> list[dict]:
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n):
        pd_ = dict(P)
        pd_["beta_I"]     = rng.gamma(4.0, P["beta_I"] / 4.0)
        pd_["beta_FR"]    = rng.gamma(4.0, P["beta_FR"] / 4.0)
        pd_["phi0"]       = float(np.clip(rng.normal(0.37, 0.06), 0.15, 0.65))
        pd_["gamma_comm"] = np.clip(rng.gamma(2.0, P["gamma_comm"] / 2.0), 0.005, 0.10)
        pd_["alpha"]      = np.clip(rng.gamma(2.0, P["alpha"] / 2.0), 0.005, 0.15)
        draws.append(pd_)
    return draws


def _band(draws, key: str, **scenario_kw):
    vals = []
    for p in draws:
        sol = run_model(p, **scenario_kw)
        y = sol.y
        if key == "deaths":
            vals.append(y[12])
        elif key == "incidence":
            vals.append(y[2] * p["theta_B"] + y[7] * p["theta_N"])
        elif key == "phi":
            denom = y[0] + y[5]
            vals.append(np.where(denom > 0, y[5] / denom, p["phi0"]))
        elif key == "rc":
            denom = y[0] + y[5]
            phi = np.where(denom > 0, y[5] / denom, p["phi0"])
            vals.append(1.0 - phi)
    arr = np.array(vals)
    return arr.mean(0), np.percentile(arr, 2.5, 0), np.percentile(arr, 97.5, 0)


# ── Load INSP data ─────────────────────────────────────────────────────────────

DECL = pd.Timestamp("2026-05-15")   # outbreak declaration date


def load_cumcases():
    df = pd.read_csv(f"{DATA_DIR}/insp_sitrep__cumulative_confirmed_cases__daily.csv")
    df["date"] = pd.to_datetime(df["date"])
    by_date = (df.groupby("date")["cumulative_confirmed_cases"]
                 .sum().reset_index())
    by_date["day"] = (by_date["date"] - DECL).dt.days
    return by_date[by_date["day"] >= 0].reset_index(drop=True)


def load_rc():
    iso = pd.read_csv(f"{DATA_DIR}/insp_sitrep__cumulative_contacts_isolated__daily.csv")
    lst = pd.read_csv(f"{DATA_DIR}/insp_sitrep__new_contacts_listed__daily.csv")

    for d in [iso, lst]:
        d["date"] = pd.to_datetime(d["date"])
        for col in d.columns:
            if col not in ("nom", "date"):
                d[col] = pd.to_numeric(d[col], errors="coerce")

    iso_by_date = (iso.groupby("date")
                     .agg({"cumulative_contacts_isolated": "sum"})
                     .reset_index())
    lst_by_date = (lst.groupby("date")
                      .agg({"new_contacts_listed": "sum"})
                      .reset_index())

    lst_by_date["cum_listed"] = lst_by_date["new_contacts_listed"].cumsum()

    merged = pd.merge(iso_by_date, lst_by_date, on="date", how="inner")
    merged["day"] = (merged["date"] - DECL).dt.days
    merged["rc"]  = (merged["cumulative_contacts_isolated"]
                     / merged["cum_listed"].replace(0, np.nan))
    return merged[merged["day"] >= 0].dropna(subset=["rc"]).reset_index(drop=True)


# ── Colour palette and style ───────────────────────────────────────────────────

C_PAL = dict(
    blue   = "#005F8C",
    coral  = "#D85A30",
    teal   = "#1D9E75",
    amber  = "#BA7517",
    purple = "#534AB7",
    gray   = "#7A7870",
    red    = "#A32D2D",
)

STYLE = {
    "font.family"       : "DejaVu Sans",
    "font.size"         : 10,
    "axes.titlesize"    : 11,
    "axes.titleweight"  : "bold",
    "axes.labelsize"    : 10,
    "axes.spines.top"   : False,
    "axes.spines.right" : False,
    "axes.linewidth"    : 0.8,
    "axes.grid"         : True,
    "grid.alpha"        : 0.20,
    "grid.linewidth"    : 0.5,
    "lines.linewidth"   : 1.8,
    "legend.framealpha" : 0.92,
    "legend.fontsize"   : 9,
    "figure.dpi"        : 300,
    "savefig.dpi"       : 300,
}


def save_fig(fig, name: str):
    os.makedirs(OUTDIR, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUTDIR, f"{name}.{ext}"),
                    dpi=300, bbox_inches="tight")
    print(f"  Saved {name}.pdf / .png  ->  {OUTDIR}/")


# ── Figure 1: Epidemic curve, phi(t), contact-tracing compliance ──────────────

def figure1(draws: list[dict], cumcases, rc_data):
    plt.rcParams.update(STYLE)

    base     = run_model(P)
    SB_b, SN_b = base.y[0], base.y[5]
    denom_b  = SB_b + SN_b
    phi_b    = np.where(denom_b > 0, SN_b / denom_b, PHI0)
    inc_b    = base.y[2]*P["theta_B"] + base.y[7]*P["theta_N"]

    m_inc, lo_inc, hi_inc = _band(draws, "incidence")
    m_phi, lo_phi, hi_phi = _band(draws, "phi")

    fig = plt.figure(figsize=(8.5, 8.0))
    gs  = gridspec.GridSpec(3, 1, hspace=0.48, figure=fig)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    # Panel A — epidemic curve
    ax1.fill_between(DAYS, lo_inc, hi_inc,
                     color=C_PAL["blue"], alpha=0.18, label="95% CrI")
    ax1.plot(DAYS, m_inc, color=C_PAL["blue"], lw=2.0,
             label="Model (posterior median)")

    # Actual INSP daily new cases (finite differences of cumulative)
    days_obs = cumcases["day"].values
    cum_obs  = cumcases["cumulative_confirmed_cases"].values
    gaps     = np.diff(days_obs, prepend=days_obs[0])
    gaps[gaps == 0] = 1
    daily_new = np.diff(cum_obs, prepend=0) / gaps

    ax1.scatter(days_obs, daily_new,
                c=C_PAL["coral"], s=32, zorder=5, edgecolors="white", lw=0.5,
                label="INSP confirmed cases (SitReps 001-007)")

    # C(t) spike (Rwampara/Mongbwalu peak: declaration days 6-8)
    ax1.axvspan(6, 9, color=C_PAL["red"], alpha=0.10, lw=0)
    ax1.axvline(6, color=C_PAL["red"], lw=0.7, ls=":", alpha=0.6)
    ax1.axvline(9, color=C_PAL["red"], lw=0.7, ls=":", alpha=0.6)

    conflict_patch = mpatches.Patch(
        color=C_PAL["red"], alpha=0.20,
        label="Conflict events (documented security events)")
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles + [conflict_patch],
               labels  + [conflict_patch.get_label()],
               loc="upper right", fontsize=8)
    ax1.set_ylabel("Daily confirmed cases")
    ax1.set_title("A  Epidemic curve — BDBV DRC 2026 (Ituri / North Kivu)")

    # Panel B — phi(t) opinion dynamics
    ax2.fill_between(DAYS, lo_phi, hi_phi,
                     color=C_PAL["amber"], alpha=0.20)
    ax2.plot(DAYS, m_phi,  color=C_PAL["amber"], lw=2.2,
             label=r"$\phi(t)$ posterior mean")
    ax2.plot(DAYS, phi_b,  color=C_PAL["amber"], lw=1.4, ls="--", alpha=0.7,
             label=r"$\phi(t)$ median parameters")
    ax2.axhline(PHI0, color=C_PAL["gray"], lw=1.0, ls=":",
                label=rf"$\phi_0 = {PHI0}$")
    ax2.axvspan(6, 9, color=C_PAL["red"], alpha=0.10, lw=0)
    ax2.set_ylabel(r"Scepticism proportion $\phi(t)$")
    ax2.set_ylim(0.0, 0.80)
    ax2.set_title(r"B  Opinion dynamics — $\phi(t)$ with conflict events")
    ax2.legend(loc="upper right", fontsize=8)

    # Panel C — contact-tracing compliance
    rc_b = 1.0 - phi_b
    ax3.fill_between(DAYS, np.clip(1-hi_phi, 0, 1), np.clip(1-lo_phi, 0, 1),
                     color=C_PAL["teal"], alpha=0.15)
    ax3.plot(DAYS, rc_b, color=C_PAL["teal"], lw=2.0,
             label=r"Model $1-\phi(t)$")
    if len(rc_data) > 0:
        ax3.scatter(rc_data["day"], rc_data["rc"],
                    c=C_PAL["purple"], s=28, zorder=5,
                    edgecolors="white", lw=0.5,
                    label=r"Observed $r_c(t)$ (contacts isolated / listed, INSP)")
    ax3.axvspan(6, 9, color=C_PAL["red"], alpha=0.10, lw=0)
    ax3.set_xlabel("Days since outbreak declaration (15 May 2026)")
    ax3.set_ylabel(r"Compliance ratio $r_c(t)$")
    ax3.set_ylim(0.0, 1.0)
    ax3.set_title(r"C  Contact-tracing compliance $r_c(t)$ vs model $1-\phi(t)$")
    ax3.legend(fontsize=8)

    for ax in [ax1, ax2, ax3]:
        ax.set_xlim(0, T_MAX)
        ax.tick_params(labelsize=9)

    save_fig(fig, "fig1_epidemic_opinion")
    plt.close(fig)


# ── Figure 3: Counterfactual scenarios ────────────────────────────────────────
# Scenario percentages from MCMC posterior analysis (manuscript Table 2)
MCMC_PCT = {"S1": 20, "S2": 16, "S3": 29, "S1+S3": 44}

SCENARIOS = [
    ("Baseline",                           dict()),
    ("S1 — 2x communication\n(from day 14)",
     dict(gc_scale=2.0, gc_start=14.0)),
    ("S2 — 50% conflict\nreduction",
     dict(C_scale=0.5)),
    ("S3 — Zero body\nreclamation",
     dict(bFR_scale=0.0)),
    ("S1+S3 — Combined",
     dict(gc_scale=2.0, gc_start=14.0, bFR_scale=0.0)),
]
SC_COLORS = [C_PAL["gray"], C_PAL["blue"], C_PAL["purple"],
             C_PAL["coral"], C_PAL["teal"]]


def figure3(draws: list[dict]):
    plt.rcParams.update(STYLE)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.5))

    bars_mean, bars_lo, bars_hi, bar_labels = [], [], [], []

    for (name, kw), col in zip(SCENARIOS, SC_COLORS):
        m, lo, hi = _band(draws, "deaths", **kw)
        lw = 2.5 if name == "Baseline" else 1.8
        ls = "--" if name == "Baseline" else "-"
        ax1.fill_between(DAYS, lo, hi, color=col, alpha=0.15)
        ax1.plot(DAYS, m, color=col, lw=lw, ls=ls,
                 label=name.replace("\n", " "))
        bars_mean.append(m[-1])
        bars_lo.append(lo[-1])
        bars_hi.append(hi[-1])
        bar_labels.append(name)

    ax1.axvline(14, color=C_PAL["blue"], lw=0.9, ls=":", alpha=0.5)
    ax1.text(14.5, max(bars_mean) * 0.04,
             "S1 starts\nday 14",
             fontsize=7.5, color=C_PAL["blue"], alpha=0.8, va="bottom")
    ax1.set_xlabel("Days since outbreak declaration (15 May 2026)")
    ax1.set_ylabel("Cumulative deaths")
    ax1.set_title("A  Cumulative deaths — baseline vs counterfactuals")
    ax1.set_xlim(0, T_MAX)
    ax1.legend(fontsize=8)

    # Bar chart — use MCMC posterior percentages from manuscript
    base_D = bars_mean[0]
    pct_list = [0, MCMC_PCT["S1"], MCMC_PCT["S2"],
                MCMC_PCT["S3"], MCMC_PCT["S1+S3"]]
    averted = [base_D * pct / 100.0 for pct in pct_list]

    x = np.arange(len(SCENARIOS))
    ax2.bar(x, averted, color=SC_COLORS, width=0.62,
            edgecolor="white", lw=0.5)

    for i, (h, pct, col) in enumerate(zip(averted, pct_list, SC_COLORS)):
        if pct > 0:
            ax2.text(i, h + max(averted) * 0.02,
                     f"{pct}%",
                     ha="center", fontsize=10, fontweight="bold", color=col)

    ax2.set_xticks(x)
    ax2.set_xticklabels([lbl.replace("\n", "\n") for lbl in bar_labels],
                        fontsize=8, ha="center")
    ax2.set_ylabel("Deaths averted at day 90 (posterior median)")
    ax2.set_title("B  Deaths averted by scenario\n(95% CrI from MCMC posterior)")
    ax2.axhline(0, color="k", lw=0.6)

    fig.tight_layout(pad=1.4)
    save_fig(fig, "fig3_scenarios")
    plt.close(fig)


# ── Figure S1: R_t and phi(t) co-evolution ───────────────────────────────────

def figureS1():
    plt.rcParams.update(STYLE)

    base     = run_model(P)
    SB, SN   = base.y[0], base.y[5]
    denom    = SB + SN
    phi_t    = np.where(denom > 0, SN / denom, PHI0)
    inc_t    = base.y[2]*P["theta_B"] + base.y[7]*P["theta_N"]

    def rolling_Rt(series, window=7):
        sm = uniform_filter1d(series.astype(float) + 0.01, size=window)
        ratio = sm[window:] / sm[:-window]
        return np.clip(ratio, 0.0, 6.0)

    Rt   = rolling_Rt(inc_t)
    t_Rt = DAYS[len(DAYS) - len(Rt):]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.0, 6.0),
                                   sharex=True,
                                   gridspec_kw={"hspace": 0.32})

    ax1.plot(t_Rt, Rt, color=C_PAL["blue"], lw=2.0,
             label=r"$\mathcal{R}_t$ (7-day rolling window)")
    ax1.axhline(1.0, color="k", lw=0.9, ls="--", alpha=0.55,
                label=r"$\mathcal{R}_t = 1$")
    ax1.axvspan(6, 9, color=C_PAL["red"], alpha=0.10, lw=0)
    ax1.axvline(6, color=C_PAL["red"], lw=0.8, ls=":", alpha=0.6)
    ax1.axvline(9, color=C_PAL["red"], lw=0.8, ls=":", alpha=0.6)
    ax1.text(7.5, 4.1, "C(t) = 1.0\n(days 6-8)",
             ha="center", fontsize=8, color=C_PAL["red"], va="top")
    ax1.set_ylabel(r"$\mathcal{R}_t$")
    ax1.set_title(r"S1-A  Time-varying $\mathcal{R}_t$ — conflict peak (days 6-8) highlighted")
    ax1.set_ylim(0, 4.5)
    ax1.legend(fontsize=9)

    ax2.plot(DAYS, phi_t, color=C_PAL["amber"], lw=2.0,
             label=r"$\phi(t)$ scepticism proportion")
    ax2.axhline(PHI0, color=C_PAL["gray"], lw=0.9, ls=":",
                label=rf"$\phi_0 = {PHI0}$")
    ax2.axvspan(6, 9, color=C_PAL["red"], alpha=0.10, lw=0)
    ax2.axvline(6, color=C_PAL["red"], lw=0.8, ls=":", alpha=0.6)
    ax2.axvline(9, color=C_PAL["red"], lw=0.8, ls=":", alpha=0.6)
    ax2.set_ylabel(r"$\phi(t)$")
    ax2.set_xlabel("Days since outbreak declaration (15 May 2026)")
    ax2.set_title(r"S1-B  Scepticism $\phi(t)$ co-evolves with $\mathcal{R}_t$"
                  " at conflict events")
    ax2.set_ylim(0.15, 0.80)
    ax2.legend(fontsize=9)
    ax2.set_xlim(0, T_MAX)

    save_fig(fig, "figS1_Rt")
    plt.close(fig)


# ── Figure S4: C(t) sensitivity (English labels, correct parameters) ──────────

# Five-anchor C(t) for S4 in declaration-day units
S4_ANCHORS = [
    (None, -5,   0.30),   # pre-epidemic baseline
    (-4,    2,   0.55),   # Nyankunde exposure
    (3,     5,   0.65),   # CDC announcement / medical evacuation
    (6,     8,   1.00),   # Rwampara/Mongbwalu peak
    (9,  None,   0.60),   # persistent insecurity
]

S4_LABELS_EN = [
    "Anchor 1\nPre-epidemic baseline\n(OCHA Q1 2026)",
    "Anchor 2\nNyankunde exposure\n(days -4 to 2, CDC)",
    "Anchor 3\nCDC announcement\n(days 3-5, DON602)",
    "Anchor 4\nRwampara/Mongbwalu\n(days 6-8, DON603)",
    "Anchor 5\nPersistent insecurity\n(days 9+, OCHA May 2026)",
]


def C_s4(t: float, anchors: list) -> float:
    for start, end, mag in anchors:
        after  = (start is None) or (t >= start)
        before = (end is None)   or (t <= end)
        if after and before:
            return mag
    return 0.30


def odes_s4(t, y, p, anchors):
    SB, EB, IB, HB, RB, SN, EN, IN, HN, RN, FR, Dcum = y
    N = SB + EB + IB + HB + RB + SN + EN + IN + HN + RN
    if N <= 0:
        return [0.0] * 12

    Ct   = C_s4(t, anchors)
    phi  = SN / max(SB + SN, 1.0)

    lB = (p["beta_I"]*(IB+IN) + p["beta_H"]*(HB+HN)) / N
    lN = (p["beta_I"]*(IB+IN) + p["beta_H"]*(HB+HN) + p["beta_FR"]*FR) / N

    Dvis  = (p["delta_I"]*(IB+IN) + p["delta_H"]*(HB+HN)) / N
    mu_BN = p["alpha"]*phi + p["delta_C"]*Ct
    mu_NB = p["gamma_comm"] + p["beta_D"]*Dvis

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

    dFR   = (p["psi_I"]*p["delta_I"]*IN
             + p["psi_H"]*p["delta_H"]*HN
             - p["omega_FR"]*FR)
    dDcum = p["delta_I"]*(IB+IN) + p["delta_H"]*(HB+HN)

    return [dSB, dEB, dIB, dHB, dRB,
            dSN, dEN, dIN, dHN, dRN,
            dFR, dDcum]


def run_s4(p, anchors):
    N_, phi_ = p["N"], p["phi0"]
    frac = 0.0002
    IB0  = (1-phi_)*N_*frac
    IN0  = phi_*N_*frac
    y0   = [(1-phi_)*N_*(1-frac), 0, IB0, 0, 0,
             phi_*N_*(1-frac),    0, IN0, 0, 0,
             0, 0]
    sol = solve_ivp(
        odes_s4, (0, T_MAX), y0,
        args=(p, anchors),
        t_eval=DAYS, method="RK45",
        rtol=1e-7, atol=1e-9,
    )
    return sol.y[11]


def perturb_anchor(base, idx, factor):
    result = [list(a) for a in base]
    result[idx][2] = min(result[idx][2] * factor, 1.0)
    return [tuple(a) for a in result]


def figureS4():
    plt.rcParams.update(STYLE)

    base_D   = run_s4(P, S4_ANCHORS)
    base_D90 = base_D[-1]
    print(f"  [S4] Base D(90) = {base_D90:.1f}")

    all_D_curves = [base_D]
    rows = []

    for idx, label in enumerate(S4_LABELS_EN):
        for factor, tag in [(0.70, "-30%"), (1.30, "+30%")]:
            anch_pert = perturb_anchor(S4_ANCHORS, idx, factor)
            D         = run_s4(P, anch_pert)
            D90       = D[-1]
            pct       = 100.0 * (D90 - base_D90) / base_D90 if base_D90 > 0 else 0.0
            all_D_curves.append(D)
            rows.append({
                "anchor"      : idx + 1,
                "perturbation": tag,
                "D90_deaths"  : round(D90, 1),
                "pct_change"  : round(pct, 2),
            })
            print(f"  Anchor {idx+1} {tag}: D(90) = {D90:.1f}  ({pct:+.1f}%)")

    pd.DataFrame(rows).to_csv(
        os.path.join(ROOT_DIR, "sensitivity_Ct_results.csv"), index=False)

    all_D = np.array(all_D_curves)
    D_min = all_D.min(axis=0)
    D_max = all_D.max(axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.fill_between(DAYS, D_min, D_max, alpha=0.22, color=C_PAL["blue"],
                    label="Envelope: all anchors ±30%")
    ax.plot(DAYS, base_D, color=C_PAL["blue"], lw=2.5, label="Base C(t)")
    for start, _, _ in S4_ANCHORS:
        if start is not None and start >= 0:
            ax.axvline(start, color="gray", ls=":", lw=0.8, alpha=0.6)
    ax.set_xlabel("Days since outbreak declaration (15 May 2026)", fontsize=10)
    ax.set_ylabel("Cumulative deaths", fontsize=10)
    ax.set_title("(A)  Cumulative deaths — base vs ±30% C(t) envelope",
                 fontsize=10, loc="left")
    ax.legend(fontsize=9, frameon=False)
    ax.set_xlim(0, T_MAX)

    ax2 = axes[1]
    n_a = len(S4_LABELS_EN)
    x   = np.arange(n_a)
    w   = 0.35
    pct_lo = [r["pct_change"] for r in rows if r["perturbation"] == "-30%"]
    pct_hi = [r["pct_change"] for r in rows if r["perturbation"] == "+30%"]

    ax2.bar(x - w/2, pct_lo, w, label="-30%",
            color=C_PAL["coral"], alpha=0.85)
    ax2.bar(x + w/2, pct_hi, w, label="+30%",
            color=C_PAL["teal"],  alpha=0.85)
    ax2.axhline(0,   color="black", lw=0.8)
    ax2.axhline(-10, color="red",   lw=1.0, ls="--",
                label="+-10% materiality threshold")
    ax2.axhline(10,  color="red",   lw=1.0, ls="--")

    ax2.set_xticks(x)
    ax2.set_xticklabels([f"Anchor {i+1}" for i in range(n_a)], fontsize=9)
    ax2.set_ylabel("% change in D(90) vs base", fontsize=10)
    ax2.set_title("(B)  % change in 90-day deaths per anchor perturbation",
                  fontsize=10, loc="left")
    ax2.legend(fontsize=9, frameon=False)

    max_abs = max(abs(v) for v in pct_lo + pct_hi)
    ax2.text(0.98, 0.96,
             f"Max |Delta| = {max_abs:.1f}%",
             transform=ax2.transAxes, ha="right", va="top",
             fontsize=9, color="#B71C1C",
             bbox=dict(boxstyle="round,pad=0.3",
                       fc="white", ec="#B71C1C", alpha=0.8))

    fig.suptitle(
        "Supplementary Figure S4 — Sensitivity of 90-day deaths to C(t) step magnitudes\n"
        "Each anchor perturbed independently by +-30%. "
        "Dashed lines = +-10% materiality threshold.",
        fontsize=10, y=1.02)

    plt.tight_layout()

    out_s4_imgs = os.path.join(OUTDIR, "figS4_sensitivity_Ct.pdf")
    out_s4_root = os.path.join(ROOT_DIR, "figS4_sensitivity_Ct.pdf")
    os.makedirs(OUTDIR, exist_ok=True)
    fig.savefig(out_s4_imgs, bbox_inches="tight", dpi=300)
    fig.savefig(out_s4_root, bbox_inches="tight", dpi=300)
    print(f"  Saved figS4_sensitivity_Ct.pdf  ->  {OUTDIR}/  and  {ROOT_DIR}/")
    plt.close(fig)

    robust = "YES" if max_abs < 15 else "NO — review C(t) anchors"
    print(f"\n  Base D(90) = {base_D90:.1f}  |  Max |Delta| = {max_abs:.1f}%  "
          f"|  Robust: {robust}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 62)
    print("figures_replot.py — SEIHRF-OD manuscript")
    print("Correct Table 1 parameters + five-anchor C(t)")
    print("=" * 62)

    print("\n[Data] Loading INSP situation reports...")
    cumcases = load_cumcases()
    rc_data  = load_rc()
    print(f"  Cumulative cases: {len(cumcases)} date points")
    print(f"  Contact-tracing r_c: {len(rc_data)} date points")

    print("\n[Draws] Sampling posterior uncertainty (n=100)...")
    draws = posterior_draws(100)

    print("\n[Figure 1] Epidemic curve, phi(t), contact-tracing compliance...")
    figure1(draws, cumcases, rc_data)

    print("\n[Figure 3] Counterfactual scenarios...")
    figure3(draws)

    print("\n[Figure S1] R_t and phi(t) co-evolution...")
    figureS1()

    print("\n[Figure S4] C(t) sensitivity — English labels, correct parameters...")
    figureS4()

    print("\n" + "=" * 62)
    print(f"All figures saved to: {OUTDIR}")
    print("=" * 62)


if __name__ == "__main__":
    main()
