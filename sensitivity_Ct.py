"""
Sensitivity analysis for C(t) step magnitudes — SEIHRF-OD model
Lancet Infectious Diseases submission, 2026

Each of the five C(t) anchors is perturbed independently by ±30%.
Cumulative deaths at 90 days are compared across all perturbations.

Outputs:
    - figS4_sensitivity_Ct.pdf      (Supplementary Figure S4)
    - sensitivity_Ct_results.csv

Dependencies:
    pip install numpy scipy matplotlib pandas
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.integrate import solve_ivp

# ── Posterior median parameters (Table 1) ─────────────────────────────────────

PARAMS = {
    "N"          : 500_000,   # outbreak zone population (Ituri, approximate)
    "beta_I"     : 0.74,      # community transmission rate (day⁻¹)
    "beta_FR"    : 1.62,      # reclaimed-body transmission rate (day⁻¹)
    "phi_0"      : 0.37,      # initial skeptic fraction
    "delta_C"    : 0.50,      # conflict amplification of distrust
    "alpha"      : 0.05,      # spontaneous B→N conversion (day⁻¹)
    "sigma"      : 1 / 9,     # incubation rate (BDBV ~9 days)
    "gamma_I"    : 1 / 6,     # community recovery rate (day⁻¹)
    "delta_I"    : 0.09,      # community death rate (day⁻¹)
    "theta_B"    : 0.60,      # hospitalization rate — Believers
    "theta_N"    : 0.07,      # hospitalization rate — Skeptics (low by definition)
    "gamma_H"    : 1 / 8,     # hospital recovery rate (day⁻¹)
    "delta_H"    : 0.28,      # hospital death rate (day⁻¹)
    "psi_I"      : 0.85,      # fraction of Skeptic community deaths → F_R
    "psi_H"      : 0.50,      # fraction of Skeptic hospital deaths → F_R
    "omega_FR"   : 1 / 3,     # body safe-burial rate (day⁻¹)
    "gamma_comm" : 0.02,      # communication N→B conversion (day⁻¹)
    "beta_D"     : 3e-6,      # visible-death distrust effect (day⁻¹ per death)
}

T_MAX  = 90    # projection horizon (days)
T_SPAN = (0, T_MAX)
T_EVAL = np.linspace(0, T_MAX, T_MAX * 10 + 1)

# ── Documented C(t) — five anchors (all citable) ──────────────────────────────
# Format: (start_day, end_day_inclusive, magnitude)
# Sources:
#   Anchor 0 (≤0, 1–16): OCHA Ituri Q1 2026 — cited in WHO DON602
#   Anchor 1 (17–23):    Nyankunde accidental exposure — CDC; The Guardian
#   Anchor 2 (24–26):    CDC announcement + medical evacuation — CDC
#   Anchor 3 (27–29):    Rwampara/Mongbwalu peak cluster — Le Devoir; DON603
#   Anchor 4 (30+):      Persistent insecurity — OCHA May 2026; DON603

BASE_ANCHORS = [
    (None, 16,   0.30),   # pre-epidemic + early constraints
    (17,   23,   0.55),   # Nyankunde exposure
    (24,   26,   0.65),   # CDC announcement / evacuation
    (27,   29,   1.00),   # Rwampara + Mongbwalu peak
    (30,   None, 0.60),   # persistent insecurity
]

ANCHOR_LABELS = [
    "Anchre 1 : fond pré-épidémique\n(jours ≤16, OCHA Q1 2026)",
    "Ancre 2 : exposition Nyankunde\n(jours 17–23, CDC)",
    "Ancre 3 : annonce CDC + évacuation\n(jours 24–26, CDC)",
    "Ancre 4 : cluster Rwampara/Mongbwalu\n(jours 27–29, DON603)",
    "Ancre 5 : insécurité persistante\n(jours 30+, OCHA mai 2026)",
]

PERTURB = 0.30   # ±30 %


def C_func(t: float, anchors: list) -> float:
    """Evaluate C(t) at time t given a list of (start, end, magnitude) anchors."""
    for start, end, mag in anchors:
        after = (start is None) or (t >= start)
        before = (end is None) or (t <= end)
        if after and before:
            return mag
    return 0.0


# ── SEIHRF-OD ODE system ──────────────────────────────────────────────────────

def odes(t, state, p, anchors):
    SB, EB, IB, HB, RB, SN, EN, IN, HN, RN, FR, Dcum = state

    N         = p["N"]
    C         = C_func(t, anchors)
    phi       = SN / max(SB + SN, 1.0)   # current skeptic fraction

    lam = (p["beta_I"] * (IB + IN) + p["beta_FR"] * FR) / N

    mu_BN = p["alpha"] * phi + p["delta_C"] * C
    mu_NB = p["gamma_comm"] + p["beta_D"] * Dcum

    # Believers
    dSB = -lam * SB - mu_BN * SB + mu_NB * SN
    dEB = lam * SB - p["sigma"] * EB - mu_BN * EB
    dIB = p["sigma"] * EB - (p["gamma_I"] + p["delta_I"] + p["theta_B"]) * IB
    dHB = p["theta_B"] * IB - (p["gamma_H"] + p["delta_H"]) * HB
    dRB = p["gamma_I"] * IB + p["gamma_H"] * HB

    # Skeptics
    dSN = -lam * SN + mu_BN * SB - mu_NB * SN
    dEN = lam * SN - p["sigma"] * EN + mu_BN * EB
    dIN = p["sigma"] * EN - (p["gamma_I"] + p["delta_I"] + p["theta_N"]) * IN
    dHN = p["theta_N"] * IN - (p["gamma_H"] + p["delta_H"]) * HN
    dRN = p["gamma_I"] * IN + p["gamma_H"] * HN

    # Reclaimed bodies (from Skeptic deaths only — Believers comply with safe burial)
    dFR = (p["psi_I"] * p["delta_I"] * IN
           + p["psi_H"] * p["delta_H"] * HN) - p["omega_FR"] * FR

    # Cumulative deaths (all compartments)
    dDcum = (p["delta_I"] * (IB + IN) + p["delta_H"] * (HB + HN))

    return [dSB, dEB, dIB, dHB, dRB, dSN, dEN, dIN, dHN, dRN, dFR, dDcum]


def initial_state(p: dict) -> list:
    N    = p["N"]
    phi0 = p["phi_0"]
    I0   = 1   # index case, Believer health worker
    SB0  = (1 - phi0) * (N - I0)
    SN0  = phi0 * (N - I0)
    return [SB0, 0, I0, 0, 0,   # Believers
            SN0, 0, 0,  0, 0,   # Skeptics
            0, 0]                # F_R, D_cum


def run_model(p: dict, anchors: list) -> dict:
    y0  = initial_state(p)
    sol = solve_ivp(
        odes, T_SPAN, y0,
        args=(p, anchors),
        t_eval=T_EVAL,
        method="RK45",
        rtol=1e-6, atol=1e-8,
        dense_output=False,
    )
    return {"t": sol.t, "Dcum": sol.y[11], "IB": sol.y[2], "IN": sol.y[8]}


# ── Build perturbed anchor sets ────────────────────────────────────────────────

def perturb_anchors(base: list, anchor_idx: int, factor: float) -> list:
    """Return a copy of base anchors with anchor_idx magnitude multiplied by factor."""
    result = [list(a) for a in base]
    result[anchor_idx][2] = min(result[anchor_idx][2] * factor, 1.0)
    return [tuple(a) for a in result]


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("C(t) sensitivity analysis — SEIHRF-OD model")
    print("=" * 60)

    # Base run
    print("[sensitivity] Running base model...")
    base = run_model(PARAMS, BASE_ANCHORS)
    base_D90 = base["Dcum"][-1]
    print(f"  Base cumulative deaths at 90 days: {base_D90:.1f}")

    # Perturbed runs
    rows = []
    all_D_curves = [base["Dcum"]]

    for idx, label in enumerate(ANCHOR_LABELS):
        for factor, tag in [(1 - PERTURB, "-30%"), (1 + PERTURB, "+30%")]:
            anchors_pert = perturb_anchors(BASE_ANCHORS, idx, factor)
            res = run_model(PARAMS, anchors_pert)
            D90 = res["Dcum"][-1]
            pct_change = 100 * (D90 - base_D90) / base_D90
            all_D_curves.append(res["Dcum"])
            rows.append({
                "anchor"       : idx + 1,
                "anchor_label" : label.replace("\n", " "),
                "perturbation" : tag,
                "D90_deaths"   : round(D90, 1),
                "pct_change"   : round(pct_change, 2),
            })
            print(f"  Anchor {idx+1} {tag}: D(90) = {D90:.1f}  ({pct_change:+.1f}%)")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    df.to_csv("sensitivity_Ct_results.csv", index=False)
    print("\n[sensitivity] Results saved → sensitivity_Ct_results.csv")

    # ── Plot ──────────────────────────────────────────────────────────────────
    all_D = np.array(all_D_curves)
    D_min = all_D.min(axis=0)
    D_max = all_D.max(axis=0)
    t     = base["t"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel A — epidemic trajectories envelope
    ax = axes[0]
    ax.fill_between(t, D_min, D_max, alpha=0.25, color="#004E7D",
                    label="Envelope ±30% all anchors")
    ax.plot(t, base["Dcum"], color="#004E7D", linewidth=2.5, label="Base C(t)")

    # Mark C(t) anchor transitions
    for start, _, _ in BASE_ANCHORS:
        if start is not None:
            ax.axvline(start, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)

    ax.set_xlabel("Day since index case (24 April 2026)", fontsize=10)
    ax.set_ylabel("Cumulative deaths", fontsize=10)
    ax.set_title("(A)  Cumulative deaths — base vs ±30% C(t) envelope",
                 fontsize=10, loc="left")
    ax.legend(fontsize=9, frameon=False)
    ax.set_xlim(0, T_MAX)

    # Panel B — bar chart of % change per anchor
    ax2 = axes[1]
    n_anchors = len(ANCHOR_LABELS)
    x = np.arange(n_anchors)
    width = 0.35

    pct_lo = [r["pct_change"] for r in rows if r["perturbation"] == "-30%"]
    pct_hi = [r["pct_change"] for r in rows if r["perturbation"] == "+30%"]

    bars_lo = ax2.bar(x - width / 2, pct_lo, width, label="-30%",
                      color="#D45500", alpha=0.8)
    bars_hi = ax2.bar(x + width / 2, pct_hi, width, label="+30%",
                      color="#2E7D32", alpha=0.8)

    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.axhline(-10, color="red", linestyle="--", linewidth=1,
                label="±10% materiality threshold")
    ax2.axhline(10, color="red", linestyle="--", linewidth=1)

    ax2.set_xticks(x)
    ax2.set_xticklabels([f"Ancre {i+1}" for i in range(n_anchors)],
                        fontsize=9)
    ax2.set_ylabel("% change in D(90) vs base", fontsize=10)
    ax2.set_title("(B)  % change in 90-day deaths per anchor perturbation",
                  fontsize=10, loc="left")
    ax2.legend(fontsize=9, frameon=False)

    # Annotate max absolute change
    max_abs = max(abs(v) for v in pct_lo + pct_hi)
    ax2.text(0.98, 0.96,
             f"Max |Δ| = {max_abs:.1f}%",
             transform=ax2.transAxes, ha="right", va="top",
             fontsize=9, color="#B71C1C",
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#B71C1C", alpha=0.8))

    fig.suptitle(
        "Supplementary Figure S4 — Sensitivity of 90-day deaths to C(t) step magnitudes\n"
        "Each anchor perturbed independently by ±30%. Dashed lines = ±10% materiality threshold.",
        fontsize=10, y=1.02)

    plt.tight_layout()
    plt.savefig("figS4_sensitivity_Ct.pdf", bbox_inches="tight", dpi=300)
    plt.close()
    print("[sensitivity] Figure saved → figS4_sensitivity_Ct.pdf")

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n── Summary ──────────────────────────────────────────────────")
    print(f"  Base D(90)             : {base_D90:.1f} deaths")
    print(f"  Envelope D(90) range   : {D_min[-1]:.1f} – {D_max[-1]:.1f}")
    print(f"  Max % change (any anchor, ±30%): {max_abs:.1f}%")
    robust = "YES" if max_abs < 15 else "NO — review C(t) anchors"
    print(f"  Conclusions robust?    : {robust}")
    print("─────────────────────────────────────────────────────────────")

    if max_abs < 15:
        print("\nLancet-ready statement (Supplementary Methods):")
        print(
            f"  Perturbing each C(t) step magnitude by ±30% changed cumulative\n"
            f"  90-day deaths by at most {max_abs:.0f}%, confirming that\n"
            f"  scenario comparisons are robust to uncertainty in the conflict\n"
            f"  intensity forcing function (Supplementary Figure S4)."
        )


if __name__ == "__main__":
    main()
