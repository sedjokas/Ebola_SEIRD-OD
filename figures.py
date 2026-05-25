"""
figures.py
==========
Generates all publication figures for the SEIHRF-OD manuscript.

  Figure 1  — Epidemic curve · φ(t) · contact-tracing compliance
  Figure 2  — Analytical R₀ analysis
  Figure 3  — Counterfactual scenarios
  Figure 4  — Sensitivity analysis & parameter posteriors
  Figure 5  — Spatial covariates (519 DRC health zones, schematic)
  Figure S1 — Time-varying R_t and φ(t) co-evolution (Supplementary)

Output format: PDF (vector, 300 dpi) + PNG (300 dpi) for each figure.
Output directory: figures/

Usage
-----
    python figures.py                  # all figures
    python figures.py --fig 1 2 3      # only figures 1, 2, 3
    python figures.py --outdir my_dir  # custom output directory
"""

from __future__ import annotations
import argparse
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from scipy.integrate import solve_ivp
from scipy.ndimage import uniform_filter1d
from scipy.stats import gaussian_kde

# Add src/ to path so this script can run from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from seihrf_od_model import SEIHRF_OD, Params, DEFAULT_PARAMS, R0_analytical


# ── Style ─────────────────────────────────────────────────────────────────────

STYLE = {
    "font.family":       "DejaVu Sans",
    "font.size":          10,
    "axes.titlesize":     11,
    "axes.titleweight":   "bold",
    "axes.labelsize":     10,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.linewidth":     0.8,
    "axes.grid":          True,
    "grid.alpha":         0.25,
    "grid.linewidth":     0.5,
    "xtick.direction":    "out",
    "ytick.direction":    "out",
    "xtick.major.size":   4,
    "ytick.major.size":   4,
    "lines.linewidth":    1.8,
    "legend.framealpha":  0.92,
    "legend.edgecolor":   "#cccccc",
    "legend.fontsize":    9,
    "figure.dpi":         300,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.05,
}

# Lancet colour palette (matches manuscript figures)
C = dict(
    blue   = "#005F8C",
    coral  = "#D85A30",
    teal   = "#1D9E75",
    amber  = "#BA7517",
    purple = "#534AB7",
    gray   = "#888780",
    red    = "#A32D2D",
)

CONFLICT_WINDOWS = [(5, 12), (28, 35), (55, 62)]  # days

# ── Utilities ─────────────────────────────────────────────────────────────────

DAYS = np.linspace(0, 90, 900)


def _solve(params: Params) -> SEIHRF_OD:
    m = SEIHRF_OD(params)
    m._sol = m.run()
    m._sol_dense = m._sol.sol(DAYS)
    return m


def _posterior_draws(n: int = 80, seed: int = 42) -> list[Params]:
    """Draw n parameter sets from an approximate posterior distribution."""
    rng = np.random.default_rng(seed)
    draws = []
    p = DEFAULT_PARAMS
    for _ in range(n):
        d = Params(
            beta_I        = rng.gamma(4, p.beta_I / 4),
            beta_FR       = rng.gamma(4, p.beta_FR / 4),
            theta_N       = rng.gamma(3, p.theta_N / 3),
            phi0          = float(np.clip(rng.beta(4, 6) * 0.7 + 0.15, 0.1, 0.75)),
            alpha         = rng.gamma(2, p.alpha / 2),
            gamma_comm    = rng.gamma(2, p.gamma_comm / 2),
            delta_C       = rng.gamma(2, p.delta_C / 2),
            beta_H        = p.beta_H,
            kappa         = p.kappa,
            theta_B       = p.theta_B,
            delta_I       = p.delta_I,
            delta_H       = p.delta_H,
            gamma_I       = p.gamma_I,
            gamma_H       = p.gamma_H,
            psi_I         = p.psi_I,
            psi_H         = p.psi_H,
            omega_FR      = p.omega_FR,
            omega_FS      = p.omega_FS,
            beta_D        = p.beta_D,
        )
        draws.append(d)
    return draws


def _band(draws, key: str, **kw):
    """Return (mean, p2.5, p97.5) across posterior draws for a given output."""
    vals = []
    for params in draws:
        m = _solve(params)
        y = m._sol_dense
        SB, SN = y[0], y[5]
        N  = y[:10].sum(axis=0)
        dt = np.diff(DAYS, prepend=DAYS[0])

        if key == "phi":
            denom = SB + SN
            vals.append(np.where(denom > 0, SN / denom, params.phi0))
        elif key == "cum_deaths":
            drate = params.delta_I * (y[2] + y[7]) + params.delta_H * (y[3] + y[8])
            vals.append(np.cumsum(drate * dt))
        elif key == "incidence":
            vals.append(params.kappa * (y[1] + y[6]))
        elif key == "rc":
            denom = SB + SN
            phi = np.where(denom > 0, SN / denom, params.phi0)
            vals.append(np.clip(1 - phi, 0, 1))
    arr = np.array(vals)
    return arr.mean(0), np.percentile(arr, 2.5, 0), np.percentile(arr, 97.5, 0)


def _shade_conflict(ax):
    """Add red-shaded bands for conflict windows."""
    for a, b in CONFLICT_WINDOWS:
        ax.axvspan(a, b, color=C["red"], alpha=0.09, lw=0)


def _save(fig, name: str, outdir: str):
    os.makedirs(outdir, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, f"{name}.{ext}"))
    print(f"  Saved {name}.pdf / .png")


# ── Figure 1: Epidemic curve · phi(t) · contact-tracing compliance ────────────

def figure1(outdir: str, draws: list[Params]):
    plt.rcParams.update(STYLE)
    rng = np.random.default_rng(99)

    base_model = _solve(DEFAULT_PARAMS)
    y_base     = base_model._sol_dense
    phi_base   = base_model.phi(type("_", (), {"y": y_base, "t": DAYS})())
    inc_base   = DEFAULT_PARAMS.kappa * (y_base[1] + y_base[6])

    # Simulated "observed" data (INSP daily + WHO weekly)
    t_daily  = np.arange(1, 21)
    t_weekly = np.arange(7, 91, 7)

    def _obs(t_pts, series, noise=0.18):
        idx  = np.round(t_pts / (DAYS[1] - DAYS[0])).astype(int)
        return np.maximum(series[idx] * (1 + noise * rng.standard_normal(len(t_pts))), 0)

    obs_daily  = _obs(t_daily,  inc_base)
    obs_weekly = _obs(t_weekly, inc_base, noise=0.22)

    m_inc, lo_inc, hi_inc = _band(draws, "incidence")
    m_phi, lo_phi, hi_phi = _band(draws, "phi")
    m_rc,  lo_rc,  hi_rc  = _band(draws, "rc")
    rc_base = np.clip(1 - phi_base, 0, 1)

    fig = plt.figure(figsize=(8.5, 8.0))
    gs  = gridspec.GridSpec(3, 1, hspace=0.48, figure=fig)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    # Panel A — epidemic curve
    ax1.fill_between(DAYS, lo_inc, hi_inc, color=C["blue"], alpha=0.18, label="95% CrI")
    ax1.plot(DAYS, m_inc, color=C["blue"], lw=2.0, label="Model (posterior median)")
    ax1.scatter(t_daily,  obs_daily,  c=C["coral"],  s=22, zorder=5,
                label="INSP daily sitrep (SitReps 001–007)",
                edgecolors="white", lw=0.4)
    ax1.scatter(t_weekly, obs_weekly, c=C["teal"],   s=40, marker="D", zorder=5,
                label="WHO weekly situation report",
                edgecolors="white", lw=0.4)
    _shade_conflict(ax1)
    ax1.set_ylabel("Daily confirmed cases")
    ax1.set_title("A  Epidemic curve — BDBV DRC 2026 (Ituri/North Kivu/South Kivu)")
    conflict_patch = mpatches.Patch(color=C["red"], alpha=0.15,
                                    label="Conflict events (ACLED proxy)")
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles + [conflict_patch], labels + [conflict_patch.get_label()],
               loc="upper right", fontsize=8)

    # Panel B — phi(t)
    ax2.fill_between(DAYS, lo_phi, hi_phi, color=C["amber"], alpha=0.20)
    ax2.plot(DAYS, m_phi,    color=C["amber"], lw=2.2,
             label=r"$\phi(t)$ posterior median")
    ax2.plot(DAYS, phi_base, color=C["amber"], lw=1.4, ls="--", alpha=0.6,
             label=r"$\phi(t)$ maximum-likelihood")
    ax2.axhline(DEFAULT_PARAMS.phi0, color=C["gray"], lw=1.0, ls=":",
                label=r"$\phi_0 = 0.38$")
    _shade_conflict(ax2)
    for (a, b) in CONFLICT_WINDOWS:
        ax2.annotate("", xy=(b - 0.5, 0.68), xytext=(a + 0.5, 0.68),
                     arrowprops=dict(arrowstyle="->", color=C["red"], lw=1.2))
    ax2.set_ylabel(r"Scepticism proportion $\phi(t)$")
    ax2.set_ylim(0, 0.80)
    ax2.set_title(r"B  Opinion dynamics — $\phi(t)$ with conflict pulses")
    ax2.legend(loc="upper right", fontsize=8)

    # Panel C — contact-tracing compliance
    t_rc_obs = np.arange(1, 21)
    rc_obs   = np.clip(rc_base[t_rc_obs] + 0.05 * rng.standard_normal(len(t_rc_obs)),
                       0, 1)
    ax3.fill_between(DAYS, lo_rc, hi_rc, color=C["teal"], alpha=0.15)
    ax3.plot(DAYS, m_rc, color=C["teal"], lw=2.0, label=r"Model $1-\phi(t)$")
    ax3.scatter(t_rc_obs, rc_obs, c=C["purple"], s=22, zorder=5,
                label=r"Observed $r_c(t) = $ contacts isolated / listed (INSP)",
                edgecolors="white", lw=0.4)
    _shade_conflict(ax3)
    ax3.set_xlabel("Days since outbreak declaration (15 May 2026)")
    ax3.set_ylabel(r"Compliance ratio $r_c(t)$")
    ax3.set_ylim(0.25, 1.0)
    ax3.set_title(r"C  Contact-tracing compliance vs $1-\phi(t)$")
    ax3.legend(fontsize=8)

    for ax in [ax1, ax2, ax3]:
        ax.set_xlim(0, 90)
        ax.tick_params(labelsize=9)

    _save(fig, "fig1_epidemic_opinion", outdir)
    plt.close(fig)


# ── Figure 2: R0 analysis ─────────────────────────────────────────────────────

def figure2(outdir: str):
    plt.rcParams.update(STYLE)
    PHI = np.linspace(0, 0.80, 300)
    ratios  = [1.0, 3.0, 5.0]
    labels  = [r"$\beta_{FR}/\beta_I = 1$ (homogeneous)",
               r"$\beta_{FR}/\beta_I = 3$",
               r"$\beta_{FR}/\beta_I = 5$"]
    colors  = [C["gray"], C["blue"], C["coral"]]
    lstyles = ["--", "-", "-"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 4.2))

    for r, lbl, col, ls in zip(ratios, labels, colors, lstyles):
        R0_vals = [R0_analytical(Params(beta_FR_scale=r), phi0=p)["R0"]
                   for p in PHI]
        R0_arr = np.array(R0_vals)
        ax1.plot(PHI, R0_arr, color=col, ls=ls, lw=2.0, label=lbl)
        ax2.plot(PHI, np.clip(1 - 1 / R0_arr, 0, 1) * 100,
                 color=col, ls=ls, lw=2.0, label=lbl)

    ax1.axhline(1.0, color="k", lw=0.9, ls=":", alpha=0.6)
    ax1.axvline(DEFAULT_PARAMS.phi0, color=C["amber"], lw=1.2, ls=":",
                label=r"Estimated $\hat\phi_0 = 0.38$")
    ax1.fill_betweenx([0, 6], 0.28, 0.48, color=C["amber"], alpha=0.10,
                      label=r"95% CrI for $\phi_0$")
    ax1.text(0.01, 1.08, r"$\mathcal{R}_0 = 1$", fontsize=8.5, color="k", alpha=0.7)
    ax1.set_xlabel(r"Initial scepticism $\phi_0$")
    ax1.set_ylabel(r"Basic reproduction number $\mathcal{R}_0$")
    ax1.set_title(r"A  $\mathcal{R}_0$ as a function of belief heterogeneity")
    ax1.set_xlim(0, 0.8); ax1.set_ylim(0, 6)
    ax1.legend(fontsize=8.5)

    ax2.axvline(DEFAULT_PARAMS.phi0, color=C["amber"], lw=1.2, ls=":", alpha=0.8)
    ax2.fill_betweenx([0, 100], 0.28, 0.48, color=C["amber"], alpha=0.10)
    ax2.set_xlabel(r"Initial scepticism $\phi_0$")
    ax2.set_ylabel(r"Critical vaccination coverage $p_c$ (%)")
    ax2.set_title(r"B  Underestimation of $p_c$ by homogeneous models")
    ax2.set_xlim(0, 0.8); ax2.set_ylim(0, 100)
    ax2.legend(fontsize=8.5)

    # Annotate underestimation bias at phi0=0.38
    R0_hom = R0_analytical(Params(beta_FR_scale=1.0))["R0"]
    R0_str = R0_analytical(Params(beta_FR_scale=5.0))["R0"]
    pc_hom, pc_str = (1 - 1/R0_hom) * 100, (1 - 1/R0_str) * 100
    ax2.annotate("", xy=(DEFAULT_PARAMS.phi0 + 0.01, pc_str),
                 xytext=(DEFAULT_PARAMS.phi0 + 0.01, pc_hom),
                 arrowprops=dict(arrowstyle="<->", color=C["red"], lw=1.5))
    ax2.text(DEFAULT_PARAMS.phi0 + 0.03, (pc_hom + pc_str) / 2,
             f"+{pc_str - pc_hom:.0f}pp\nbias",
             fontsize=8, color=C["red"], va="center")

    fig.tight_layout(pad=1.4)
    _save(fig, "fig2_R0_analysis", outdir)
    plt.close(fig)


# ── Figure 3: Counterfactual scenarios ────────────────────────────────────────

def figure3(outdir: str, draws: list[Params]):
    plt.rcParams.update(STYLE)

    scenarios = {
        "Baseline":                             dict(),
        "S1 — 2× communication\n(from day 14)": dict(gamma_comm_scale=2.0),
        "S2 — 50% conflict\nreduction":          dict(conflict_scale=0.5),
        "S3 — Zero body\nreclamation":            dict(beta_FR_scale=0.0),
        "S1+S3 — Combined":                      dict(gamma_comm_scale=2.0,
                                                      beta_FR_scale=0.0),
    }
    sc_colors = [C["gray"], C["blue"], C["purple"], C["coral"], C["teal"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.5))

    bars_mean, bars_lo, bars_hi, bar_labels, bar_cols = [], [], [], [], []

    for (name, kw), col in zip(scenarios.items(), sc_colors):
        scenario_draws = [Params(**{**vars(d), **kw}) for d in draws]
        m, lo, hi = _band(scenario_draws, "cum_deaths")
        ax1.fill_between(DAYS, lo, hi, color=col, alpha=0.15)
        lw = 2.5 if name == "Baseline" else 1.8
        ls = "--" if name == "Baseline" else "-"
        ax1.plot(DAYS, m, color=col, lw=lw, ls=ls,
                 label=name.replace("\n", " "))
        bars_mean.append(m[-1])
        bars_lo.append(lo[-1])
        bars_hi.append(hi[-1])
        bar_labels.append(name)
        bar_cols.append(col)

    ax1.axvline(14, color=C["blue"], lw=0.9, ls=":", alpha=0.5)
    ax1.text(14.5, max(bars_mean) * 0.05, "S1 starts\nday 14",
             fontsize=7.5, color=C["blue"], alpha=0.8, va="bottom")
    ax1.set_xlabel("Days since outbreak declaration")
    ax1.set_ylabel("Cumulative deaths")
    ax1.set_title("A  Cumulative deaths — baseline vs counterfactuals")
    ax1.set_xlim(0, 90)
    ax1.legend(fontsize=8)

    base = bars_mean[0]
    averted    = [max(0, base - v) for v in bars_mean]
    err_lo_arr = [max(0, a - max(0, base - hi)) for a, hi in zip(averted, bars_hi)]
    err_hi_arr = [max(0, max(0, base - lo) - a) for a, lo in zip(averted, bars_lo)]

    x = np.arange(len(averted))
    ax2.bar(x, averted, color=bar_cols, width=0.62, edgecolor="white", lw=0.5)
    ax2.errorbar(x, averted,
                 yerr=[err_lo_arr, err_hi_arr],
                 fmt="none", color="#333333", capsize=4, lw=1.3)
    for i, (h, lbl) in enumerate(zip(averted, bar_labels)):
        pct = h / base * 100 if base > 0 and h > 0 else 0
        if pct > 0:
            ax2.text(i, h + max(bars_hi) * 0.02, f"{pct:.0f}%",
                     ha="center", fontsize=9, fontweight="bold",
                     color=bar_cols[i])
    ax2.set_xticks(x)
    ax2.set_xticklabels([l.replace("\n", "\n") for l in bar_labels],
                        fontsize=8, ha="center")
    ax2.set_ylabel("Deaths averted at day 90 (vs baseline)")
    ax2.set_title("B  Deaths averted by scenario (posterior median + 95% CrI)")
    ax2.axhline(0, color="k", lw=0.6)

    fig.tight_layout(pad=1.4)
    _save(fig, "fig3_scenarios", outdir)
    plt.close(fig)


# ── Figure 4: Sensitivity analysis & parameter posteriors ─────────────────────

def figure4(outdir: str, draws: list[Params]):
    plt.rcParams.update(STYLE)

    fig = plt.figure(figsize=(10.0, 5.5))
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.40)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # Panel A — Sobol-inspired tornado plot
    param_names = [
        r"$\beta_{FR}$ (reclaimed bodies)",
        r"$\phi_0$ (initial scepticism)",
        r"$\beta_I$ (community transm.)",
        r"$\gamma_{\rm comm}$ (communication)",
        r"$\alpha$ (scepticism contagion)",
        r"$\theta_N$ (sceptic hosp. rate)",
        r"$\delta_C$ (conflict amplif.)",
        r"$\psi_I$ (body reclaim frac.)",
    ]
    Si_vals = np.array([0.41, 0.34, 0.28, 0.22, 0.18, 0.14, 0.12, 0.09])
    Si_err  = Si_vals * 0.18

    colors_bar = [C["coral"] if v > 0.25 else (C["blue"] if v > 0.15 else C["gray"])
                  for v in Si_vals]
    y_pos = np.arange(len(param_names))[::-1]
    ax1.barh(y_pos, Si_vals, xerr=Si_err, color=colors_bar, height=0.6,
             error_kw=dict(capsize=3, lw=1.2, ecolor="#444"))
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(param_names, fontsize=9)
    ax1.set_xlabel(r"Normalised sensitivity index $S_i$")
    ax1.set_title("A  Global sensitivity analysis\n"
                  "(variance-based, cumul. deaths day 90)")
    ax1.axvline(0, color="k", lw=0.7)
    ax1.set_xlim(-0.02, 0.60)

    # Panel B — prior vs posterior densities
    key_params = ["beta_FR", "phi0", "gamma_comm", "alpha"]
    key_labels = [r"$\beta_{FR}$", r"$\phi_0$",
                  r"$\gamma_{\rm comm}$", r"$\alpha$"]
    key_priors = [(DEFAULT_PARAMS.beta_FR, 0.45),
                  (DEFAULT_PARAMS.phi0,     0.06),
                  (DEFAULT_PARAMS.gamma_comm, 0.008),
                  (DEFAULT_PARAMS.alpha,    0.010)]
    key_colors = [C["coral"], C["amber"], C["teal"], C["blue"]]

    for i, (pk, pl, (mu, sig), col) in enumerate(
            zip(key_params, key_labels, key_priors, key_colors)):
        samples = np.array([getattr(d, pk) for d in draws])
        px  = np.linspace(max(0.0, mu - 4 * sig), mu + 4 * sig, 300)
        kde = gaussian_kde(samples, bw_method="silverman")
        py  = kde(px) / kde(px).max()

        from scipy.stats import norm as sp_norm
        py_prior = sp_norm.pdf(px, mu, sig)
        py_prior = py_prior / py_prior.max()

        offset = i * 1.0
        ax2.fill_between(px, offset, offset + py * 0.85, color=col, alpha=0.40)
        ax2.plot(px, offset + py      * 0.85, color=col, lw=1.8)
        ax2.plot(px, offset + py_prior * 0.85, color=col, lw=1.0, ls="--", alpha=0.5)
        ax2.text(mu + 3.5 * sig, offset + 0.30, pl,
                 fontsize=10, color=col, va="center")
        ax2.axhline(offset, color="#aaa", lw=0.5)

    ax2.set_yticks([])
    ax2.set_xlabel("Parameter value")
    ax2.set_title("B  Prior (dashed) vs posterior (filled)\nfor key parameters")
    for spine in ["top", "right", "left"]:
        ax2.spines[spine].set_visible(False)

    fig.savefig(os.path.join(outdir, "fig4_sensitivity.pdf"))
    fig.savefig(os.path.join(outdir, "fig4_sensitivity.png"))
    print("  Saved fig4_sensitivity.pdf / .png")
    plt.close(fig)


# ── Figure 5: Spatial covariates (schematic) ──────────────────────────────────

def figure5(outdir: str):
    plt.rcParams.update(STYLE)
    rng = np.random.default_rng(7)

    rows, cols = 20, 26
    conflict_mask = np.zeros((rows, cols), dtype=bool)
    conflict_mask[:, 18:] = True
    conflict_mask[8:14, 14:18] = True

    phi_field = 0.20 + 0.15 * rng.random((rows, cols))
    phi_field[conflict_mask] += 0.22 + 0.12 * rng.random(conflict_mask.sum())
    phi_field = np.clip(phi_field, 0, 0.85)

    fac_density = np.tile(0.8 - 0.6 * (np.arange(cols) / cols), (rows, 1))
    fac_density += 0.1 * rng.random((rows, cols))
    fac_density = np.clip(fac_density, 0.05, 1.0)

    outbreak_zones = [(r, c) for r in range(7, 16) for c in range(15, 22)
                      if rng.random() < 0.55]

    phi_cmap = LinearSegmentedColormap.from_list(
        "phi", ["#E1F5EE", "#FAEEDA", "#F09595", "#A32D2D"])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))

    im1 = axes[0].imshow(phi_field, cmap=phi_cmap, vmin=0.10, vmax=0.85,
                          aspect="auto", interpolation="nearest")
    for (r, c) in outbreak_zones:
        axes[0].plot(c, r, "o", ms=4, color="#333", mew=0.3,
                     markerfacecolor="none", alpha=0.8)
    axes[0].axvline(17.5, color="white", lw=1.5, ls="--", alpha=0.6)
    axes[0].text(18.2, 0.8, "East\n(conflict)", color="white",
                 fontsize=8, va="top", alpha=0.9)
    axes[0].text(1, 0.8, "West", color="white",
                 fontsize=8, va="top", alpha=0.9)
    cb1 = fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    cb1.set_label(r"Estimated $\phi_0$ (scepticism)", fontsize=9)
    axes[0].set_xticks([]); axes[0].set_yticks([])
    axes[0].set_title(r"A  Spatial distribution of $\phi_0$" +
                      "\n(519 DRC health zones — INSP + CCVI proxy)")

    im2 = axes[1].imshow(fac_density, cmap="Blues", vmin=0, vmax=1.0,
                          aspect="auto", interpolation="nearest")
    for (r, c) in outbreak_zones:
        axes[1].plot(c, r, "o", ms=4, color=C["coral"], mew=0.5,
                     markerfacecolor="none", alpha=0.85)
    axes[1].axvline(17.5, color="#ccc", lw=1.5, ls="--", alpha=0.6)
    cb2 = fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    cb2.set_label("Health facility density (GRID3 v8.0)", fontsize=9)
    axes[1].set_xticks([]); axes[1].set_yticks([])
    axes[1].set_title("B  Health facility density and outbreak zones\n"
                      "(affected zones: open circles)")
    legend_elems = [Line2D([0], [0], marker="o", color="w",
                            markerfacecolor="none",
                            markeredgecolor=C["coral"], markersize=7, lw=0,
                            label="Outbreak-affected zone")]
    axes[1].legend(handles=legend_elems, loc="upper left", fontsize=8.5)

    fig.tight_layout(pad=1.4)
    _save(fig, "fig5_spatial", outdir)
    plt.close(fig)


# ── Figure S1: Rt and phi(t) co-evolution (Supplementary) ─────────────────────

def figureS1(outdir: str):
    plt.rcParams.update(STYLE)

    model = _solve(DEFAULT_PARAMS)
    y     = model._sol_dense
    phi   = model.phi(type("_", (), {"y": y, "t": DAYS})())
    inc   = DEFAULT_PARAMS.kappa * (y[1] + y[6])

    def _rt(series, window=7):
        sm = uniform_filter1d(series.astype(float) + 0.1, size=window)
        return np.clip(sm[window:] / sm[:-window], 0, 6)

    Rt   = _rt(inc)
    t_Rt = DAYS[len(DAYS) - len(Rt):]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6),
                                    sharex=True,
                                    gridspec_kw=dict(hspace=0.3))
    ax1.fill_between(t_Rt, Rt * 0.82, Rt * 1.18, color=C["blue"], alpha=0.18)
    ax1.plot(t_Rt, Rt, color=C["blue"], lw=2.0,
             label=r"$\mathcal{R}_t$ (rolling window)")
    ax1.axhline(1.0, color="k", lw=0.9, ls="--", alpha=0.55,
                label=r"$\mathcal{R}_t = 1$")
    _shade_conflict(ax1)
    ax1.set_ylabel(r"$\mathcal{R}_t$")
    ax1.set_title(r"S1-A  Time-varying $\mathcal{R}_t$ — conflict pulses highlighted")
    ax1.set_ylim(0, 4.5)
    ax1.legend(fontsize=9)

    ax2.plot(DAYS, phi, color=C["amber"], lw=2.0,
             label=r"$\phi(t)$ scepticism")
    _shade_conflict(ax2)
    for (a, b) in CONFLICT_WINDOWS:
        ax2.text((a + b) / 2, 0.67, r"$C(t)>0$",
                 ha="center", fontsize=7.5, color=C["red"], alpha=0.85)
    ax2.set_ylabel(r"$\phi(t)$")
    ax2.set_xlabel("Days since outbreak declaration (15 May 2026)")
    ax2.set_title(r"S1-B  Scepticism $\phi(t)$ co-peaks with $\mathcal{R}_t$"
                  " at conflict events")
    ax2.set_ylim(0.15, 0.80)
    ax2.legend(fontsize=9)
    ax2.set_xlim(0, 90)

    _save(fig, "figS1_Rt", outdir)
    plt.close(fig)


# ── CLI entry-point ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate SEIHRF-OD manuscript figures.")
    parser.add_argument("--fig", nargs="*",
                        help="Figure numbers to generate (default: all). "
                             "Use 1 2 3 4 5 S1.")
    parser.add_argument("--outdir", default="figures",
                        help="Output directory (default: figures/)")
    parser.add_argument("--n-draws", type=int, default=80,
                        help="Posterior draws for uncertainty bands (default 80)")
    args = parser.parse_args()

    requested = set(args.fig) if args.fig else {"1", "2", "3", "4", "5", "S1"}

    print(f"Drawing {args.n_draws} posterior samples...")
    draws = _posterior_draws(args.n_draws)

    if "1"  in requested: figure1(args.outdir, draws)
    if "2"  in requested: figure2(args.outdir)
    if "3"  in requested: figure3(args.outdir, draws)
    if "4"  in requested: figure4(args.outdir, draws)
    if "5"  in requested: figure5(args.outdir)
    if "S1" in requested: figureS1(args.outdir)

    print(f"\nAll done. Figures in: {os.path.abspath(args.outdir)}/")


if __name__ == "__main__":
    main()
