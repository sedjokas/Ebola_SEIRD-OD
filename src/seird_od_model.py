"""
seird_od_model.py
=================
SEIRD-OD: Coupled epidemic–opinion-dynamics model for the 2026
Bundibugyo Ebola outbreak (Ituri Province, DRC).

Mathematical reference
----------------------
Equations (1)–(12) and the opinion-dynamics layer (Eq. 13) from:

  "Political distrust, armed conflict, and body reclamation drive Ebola
   transmission dynamics in eastern Democratic Republic of Congo:
   a coupled epidemic–opinion dynamics model (SEIRD-OD)"
   [Submitted to The Lancet Infectious Diseases, 2026]

Repository data source
----------------------
INRB-UMIE/Ebola_DRC_2026  https://github.com/INRB-UMIE/Ebola_DRC_2026
Build 235a3c3 / data snapshot 493d506, accessed 22 May 2026.

Model verified symbolically with SymPy 1.13 (five checks):
  1. Population-flow balance
  2. F_R mass balance
  3. Death partitioning (F_R + F_S = total deaths from N)
  4. R0 via Next Generation Matrix
  5. phi ODE algebraic consistency with S_B / S_N equations

Usage
-----
>>> from src.seird_od_model import SEIRD_OD, DEFAULT_PARAMS
>>> model = SEIRD_OD(DEFAULT_PARAMS)
>>> sol   = model.run(t_end=90)
>>> t, y  = sol.t, sol.y
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Callable
from scipy.integrate import solve_ivp

# ── Default parameter set (posterior medians, Table 1 of the manuscript) ─────

@dataclass
class Params:
    """All SEIRD-OD parameters with units and interpretations."""

    # Transmission rates (day⁻¹)
    beta_I:  float = 0.38   # community transmission
    beta_H:  float = 0.06   # hospital/ETC transmission
    beta_FR: float = 1.60   # reclaimed-body transmission  (F_R compartment)
    beta_FS: float = 0.002  # safe-burial transmission    (≈ 0)

    # Disease progression (day⁻¹)
    kappa:    float = 1/9   # 1 / mean incubation period (9-day BDBV mean)
    theta_B:  float = 0.28  # hospitalisation rate — Believers
    theta_N:  float = 0.04  # hospitalisation rate — Sceptics (≪ theta_B)
    delta_I:  float = 0.18  # community case-fatality rate
    delta_H:  float = 0.12  # hospital  case-fatality rate
    gamma_I:  float = 0.09  # community recovery rate
    gamma_H:  float = 0.10  # hospital  recovery rate

    # Body-disposal rates (day⁻¹)
    omega_FR: float = 0.80  # unsafe (reclaimed) body disposal rate
    omega_FS: float = 3.00  # safe burial disposal rate

    # Body-reclamation fractions ∈ [0, 1]
    psi_I: float = 0.45     # fraction of sceptic community deaths reclaimed
    psi_H: float = 0.15     # fraction of sceptic hospital  deaths reclaimed

    # Opinion-dynamics parameters
    phi0:       float = 0.38   # initial scepticism proportion
    alpha:      float = 0.04   # social-contagion rate of scepticism (day⁻¹)
    beta_D:     float = 8.0    # death-driven belief-update coefficient
    gamma_comm: float = 0.025  # health-communication effectiveness (day⁻¹)
    delta_C:    float = 0.60   # conflict-amplification coefficient

    # Population
    N0: int = 120_000           # initial total population

    # Optional scaling factors for scenario analysis (default: no change)
    gamma_comm_scale: float = 1.0
    beta_FR_scale:    float = 1.0
    conflict_scale:   float = 1.0


DEFAULT_PARAMS = Params()


# ── Conflict-intensity function C(t) ─────────────────────────────────────────

def conflict_C(t: float, scale: float = 1.0) -> float:
    """
    Piecewise conflict-intensity proxy C(t), calibrated from ACLED event
    data (ACLED QA pending in INRB-UMIE repo, issue #14).

    Three pulses corresponding to reported security incidents in Ituri and
    North Kivu during the first 90 days after outbreak declaration:
      - Pulse 1:  days  5–12  (intensity 0.80)
      - Pulse 2:  days 28–35  (intensity 1.20)
      - Pulse 3:  days 55–62  (intensity 0.50)
    """
    c = 0.0
    if  5 <= t <= 12: c = 0.80
    if 28 <= t <= 35: c = 1.20
    if 55 <= t <= 62: c = 0.50
    return c * scale


# ── ODE right-hand side ───────────────────────────────────────────────────────

def _rhs(t: float, y: np.ndarray, p: Params) -> list[float]:
    """
    Right-hand side of the 12-dimensional SEIRD-OD ODE system.

    State vector
    ------------
    y = [S_B, E_B, I_B, H_B, R_B,
         S_N, E_N, I_N, H_N, R_N,
         F_R, F_S]

    Compartment indices
    -------------------
    0  S_B   Susceptible  Believers
    1  E_B   Exposed      Believers
    2  I_B   Infectious   Believers  (community)
    3  H_B   Hospitalised Believers
    4  R_B   Recovered    Believers
    5  S_N   Susceptible  Sceptics
    6  E_N   Exposed      Sceptics
    7  I_N   Infectious   Sceptics   (community)
    8  H_N   Hospitalised Sceptics
    9  R_N   Recovered    Sceptics
    10 F_R   Reclaimed bodies  (unsafe burial — HIGH transmission)
    11 F_S   Safe-burial bodies (low transmission)
    """
    SB, EB, IB, HB, RB, SN, EN, IN, HN, RN, FR, FS = y

    N = SB + EB + IB + HB + RB + SN + EN + IN + HN + RN
    if N <= 0:
        return [0.0] * 12

    # Scale parameters for scenario analysis
    bFR = p.beta_FR * p.beta_FR_scale
    gc  = p.gamma_comm * p.gamma_comm_scale
    Ct  = conflict_C(t, scale=p.conflict_scale)

    # Forces of infection (Eqs. 1–2)
    lB = (p.beta_I * (IB + IN) + p.beta_H * (HB + HN) + p.beta_FS * FS) / N
    lN = (p.beta_I * (IB + IN) + p.beta_H * (HB + HN) + bFR * FR) / N

    # Visible death rate D_vis (Eq. 14)
    Dvis = (p.delta_I * (IB + IN) + p.delta_H * (HB + HN)) / N

    # Opinion-driven conversion rates
    phi    = SN / (SB + SN) if (SB + SN) > 0 else p.phi0
    mu_BN  = p.alpha * phi                   # Believers  → Sceptics
    mu_NB  = gc + p.beta_D * Dvis            # Sceptics   → Believers

    # Believers (Eqs. 3–7)
    dSB = -lB * SB - mu_BN * SB + mu_NB * SN
    dEB =  lB * SB - p.kappa * EB
    dIB =  p.kappa * EB - (p.theta_B + p.delta_I + p.gamma_I) * IB
    dHB =  p.theta_B * IB - (p.delta_H + p.gamma_H) * HB
    dRB =  p.gamma_I * IB + p.gamma_H * HB

    # Sceptics (Eqs. 8–12)
    dSN = -lN * SN + mu_BN * SB - mu_NB * SN
    dEN =  lN * SN - p.kappa * EN
    dIN =  p.kappa * EN - (p.theta_N + p.delta_I + p.gamma_I) * IN
    dHN =  p.theta_N * IN - (p.delta_H + p.gamma_H) * HN
    dRN =  p.gamma_I * IN + p.gamma_H * HN

    # Post-mortem compartments (Eqs. 13–14)
    dFR = (p.psi_I * p.delta_I * IN
           + p.psi_H * p.delta_H * HN
           - p.omega_FR * FR)
    dFS = (p.delta_I * IB + p.delta_H * HB
           + (1 - p.psi_I) * p.delta_I * IN
           + (1 - p.psi_H) * p.delta_H * HN
           - p.omega_FS * FS)

    return [dSB, dEB, dIB, dHB, dRB,
            dSN, dEN, dIN, dHN, dRN,
            dFR, dFS]


# ── Main model class ──────────────────────────────────────────────────────────

class SEIRD_OD:
    """
    SEIRD-OD model object.

    Parameters
    ----------
    params : Params
        Model parameters (see Params dataclass).

    Examples
    --------
    >>> model = SEIRD_OD(DEFAULT_PARAMS)
    >>> sol   = model.run(t_end=90)

    # Scenario S3: zero body reclamation
    >>> p_s3 = Params(beta_FR_scale=0.0)
    >>> sol_s3 = SEIRD_OD(p_s3).run()
    """

    STATE_LABELS = [
        "S_B", "E_B", "I_B", "H_B", "R_B",
        "S_N", "E_N", "I_N", "H_N", "R_N",
        "F_R", "F_S",
    ]

    def __init__(self, params: Params = DEFAULT_PARAMS):
        self.p = params

    def _initial_conditions(self) -> list[float]:
        N  = self.p.N0
        p0 = self.p.phi0
        return [
            (1 - p0) * N * 0.9998, 0.0, (1 - p0) * N * 0.0002, 0.0, 0.0,
            p0       * N * 0.9998, 0.0, p0       * N * 0.0002, 0.0, 0.0,
            0.0, 0.0,
        ]

    def run(
        self,
        t_end: float = 90.0,
        t_eval: np.ndarray | None = None,
        rtol: float = 1e-7,
        atol: float = 1e-9,
    ):
        """
        Integrate the ODE system from t=0 to t=t_end.

        Returns a scipy OdeResult with dense_output=True.
        """
        if t_eval is None:
            t_eval = np.linspace(0, t_end, int(t_end * 10) + 1)

        return solve_ivp(
            fun=lambda t, y: _rhs(t, y, self.p),
            t_span=[0.0, t_end],
            y0=self._initial_conditions(),
            t_eval=t_eval,
            dense_output=True,
            method="RK45",
            max_step=0.5,
            rtol=rtol,
            atol=atol,
        )

    def phi(self, sol) -> np.ndarray:
        """Time series of scepticism proportion phi(t) from a solution."""
        SB = sol.y[0]
        SN = sol.y[5]
        denom = SB + SN
        return np.where(denom > 0, SN / denom, self.p.phi0)

    def cumulative_deaths(self, sol) -> np.ndarray:
        """Cumulative deaths from numerical integration of the death-rate."""
        dt   = np.diff(sol.t, prepend=sol.t[0])
        IB, IN = sol.y[2], sol.y[7]
        HB, HN = sol.y[3], sol.y[8]
        N = sol.y[:10].sum(axis=0)
        drate = self.p.delta_I * (IB + IN) + self.p.delta_H * (HB + HN)
        return np.cumsum(drate * dt)

    def daily_incidence(self, sol) -> np.ndarray:
        """Approximate daily incidence (kappa * E_B + kappa * E_N)."""
        return self.p.kappa * (sol.y[1] + sol.y[6])


# ── Analytical R0 ─────────────────────────────────────────────────────────────

def R0_analytical(params: Params, phi0: float | None = None) -> dict:
    """
    Compute R0 analytically via the Next Generation Matrix (see Supp. B).

    Returns
    -------
    dict with keys: R0_B, R0_N, R0 (population-weighted), phi0
    """
    p   = params
    phi = phi0 if phi0 is not None else p.phi0

    kB  = p.theta_B + p.delta_I + p.gamma_I
    kN  = p.theta_N + p.delta_I + p.gamma_I
    kH  = p.delta_H + p.gamma_H
    bFR = p.beta_FR * p.beta_FR_scale

    # Eq. R0_B (Manuscript Eq. 5)
    R0_B = (p.beta_I + p.beta_H * p.theta_B / kH) / kB

    # Eq. R0_N (Manuscript Eq. 6) — includes unsafe-funeral amplifier
    R0_N = (
        p.beta_I
        + p.beta_H * p.theta_N / kH
        + bFR / p.omega_FR * (
            p.psi_I * p.delta_I
            + p.psi_H * p.delta_H * p.theta_N / kH
        )
    ) / kN

    # Population-weighted R0 (Eq. 4)
    R0 = (1 - phi) * R0_B + phi * R0_N

    return {"R0_B": R0_B, "R0_N": R0_N, "R0": R0, "phi0": phi}
