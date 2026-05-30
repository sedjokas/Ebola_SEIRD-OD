# Ebola SEIHRF-OD

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.PLACEHOLDER.svg)](https://doi.org/10.5281/zenodo.PLACEHOLDER)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

This repository contains the code, data, and figures for:

> Selain K. Kasereka, Kyandoghere Kyamakya, Jean-Jacques T. Muyembe,
> "Political distrust, armed conflict, and body reclamation drive Ebola
> transmission dynamics in eastern Democratic Republic of Congo: a
> coupled epidemic and opinion dynamics model (SEIHRF-OD)"
> — under review, *The Lancet Infectious Diseases*, 2026
>
> University of Klagenfurt · INRB/INOHA Kinshasa · INSP Kinshasa

---

## What this is

When families in eastern DRC started reclaiming the bodies of relatives who
died of Ebola from treatment centres — sometimes violently — and carrying
them home for traditional burial, existing epidemic models had no way to
represent that. Those models assume everyone follows public-health guidance.
Here they clearly did not.

SEIHRF-OD splits the population into two groups: people who accept the EVD
diagnosis (Believers, *B*) and people who deny or doubt it (Sceptics, *N*).
These are analytical categories representing levels of institutional trust;
they carry no moral judgment about affected communities, whose distrust is a
structural product of conflict, prior exploitation, and systemic exclusion.

Sceptics avoid hospitals and their deceased relatives' bodies enter a separate
compartment **F_R** (reclaimed bodies), with transmission coefficient
β_FR that substantially exceeds community and hospital rates. A third layer
tracks how the sceptic proportion φ(t) changes over time, driven by social
contagion, visible community deaths, health communication, and armed-conflict
intensity C(t).

Calibration uses full Bayesian MCMC (CmdStan/CmdStanPy) on daily INSP
situation-report data (127 confirmed cases, SitReps 001–012). The basic
reproduction number R₀ is derived analytically via the Next Generation Matrix
and evaluated across all posterior draws.

---

## Repo layout

```
Ebola_SEIHRF-OD/
├── seihrf_od.stan               # Stan ODE model — 12-compartment SEIHRF-OD
├── seihrf_od_profile.stan       # Profile-likelihood wrapper (fixes one param at a time)
├── run_mcmc.R                   # CmdStanR MCMC calibration script
├── run_mcmc_py.py               # CmdStanPy MCMC calibration script (Python)
├── figures_replot.py            # Figs 1, 3, S1, S4 — correct Table-1 parameters
├── gen_fig2.py                  # Fig 2 — R₀ vs φ₀ and p_c vs φ₀
├── gen_fig4.py                  # Fig 4 — Sobol tornado + prior/posterior panel
├── gen_ppc.py                   # Posterior predictive check → figS_ppc
├── holdout_validation.py        # 8/3 hold-out validation → figS_holdout
├── bfr_robustness.py            # Supp Table S1 — β_FR prior-support sweep
├── compute_scenarios.py         # Counterfactual S1/S2/S3/S1+S3 with MCMC CrI
├── profile_likelihood.py        # Profile-likelihood identifiability → figS2
├── sensitivity_Ct.py            # C(t) sensitivity analysis → figS4
├── acled_pipeline.py            # C(t) reconstruction from documented security events
├── stan_data.json               # Prepared Stan input (127 cases, SitReps 001–012)
├── posterior_draws.csv          # 8 000 MCMC draws (4 chains × 2 000 samples)
├── profile_likelihood_results.csv
├── sensitivity_Ct_results.csv
├── requirements.txt
├── data/
│   ├── insp_sitrep__new_confirmed_cases__daily.csv
│   ├── insp_sitrep__cumulative_confirmed_cases__daily.csv
│   ├── insp_sitrep__cumulative_confirmed_deaths__daily.csv
│   ├── insp_sitrep__cumulative_contacts_isolated__daily.csv
│   └── insp_sitrep__new_contacts_listed__daily.csv
├── imgs/
│   ├── fig1_epidemic_opinion.{pdf,png}   # Epidemic curve + opinion dynamics
│   ├── fig2_R0_analysis.{pdf,png}        # R₀ vs φ₀ and p_c panels
│   ├── fig3_scenarios.{pdf,png}          # Counterfactual scenarios
│   ├── fig4_sensitivity.{pdf,png}        # Sobol tornado + prior/posterior
│   ├── fig5_spatial.{pdf,png}            # Spatial covariates (metapopulation)
│   ├── figS1_Rt.{pdf,png}               # Time-varying Rt
│   ├── figS2_profile_likelihood.pdf      # Profile-likelihood identifiability
│   ├── figS4_sensitivity_Ct.pdf          # C(t) sensitivity
│   ├── figS_ppc.{pdf,png}               # Posterior predictive check (all 13 days)
│   └── figS_holdout.{pdf,png}           # Hold-out validation (8/3 split)
└── src/
    ├── seihrf_od_model.py       # Python ODE implementation (exploratory)
    └── seird_od_model.py
```

---

## Getting started

### Requirements

```bash
pip install -r requirements.txt
```

Python dependencies: `numpy`, `scipy`, `matplotlib`, `pandas`, `cmdstanpy`,
`SALib` (Sobol sensitivity), `sympy`.

### MCMC calibration (Python — recommended)

```bash
python run_mcmc_py.py
# → posterior_draws.csv  (8 000 draws)
# → stan_data.json
```

4 chains × 2 000 warm-up + 2 000 sampling, seed=42, CmdStanPy.
Convergence: max R̂ = 1.0011, min ESS = 3 042 (excellent).

### MCMC calibration (R)

```r
install.packages(c("cmdstanr", "posterior", "bayesplot", "loo",
                   "dplyr", "readr", "lubridate"))
cmdstanr::install_cmdstan()
Rscript run_mcmc.R
```

### Generate figures

```bash
python figures_replot.py   # figs 1, 3, S1, S4
python gen_fig2.py         # fig 2
python gen_fig4.py         # fig 4
```

### Counterfactual scenarios

```bash
python compute_scenarios.py   # prints S1/S2/S3/S1+S3 deaths-averted table
```

### β_FR robustness (Supp Table S1)

```bash
python bfr_robustness.py   # sweeps β_FR across prior 5th–95th pct range
```

### Identifiability and sensitivity

```bash
python profile_likelihood.py   # → imgs/figS2_profile_likelihood.pdf
python sensitivity_Ct.py       # → imgs/figS4_sensitivity_Ct.pdf
```

### Validation

```bash
# Posterior predictive check (uses existing posterior_draws.csv — no new MCMC)
python gen_ppc.py              # → imgs/figS_ppc.pdf/.png

# 8/3 hold-out validation (runs new MCMC on 10-day calibration set, ~2 min)
# Result cached in posterior_calib.csv after first run
python holdout_validation.py   # → imgs/figS_holdout.pdf/.png
```

---

## The model

### State variables (12 compartments)

|  | Believers (B) | Sceptics (N) |
|---|---|---|
| Susceptible | S_B | S_N |
| Exposed | E_B | E_N |
| Infectious | I_B | I_N |
| Hospitalised | H_B | H_N |
| Recovered | R_B | R_N |

Plus two post-mortem compartments: **F_R** (reclaimed body — high
transmission, β_FR) and **F_S** (safe burial — negligible transmission,
β_FS ≈ 0).

### Forces of infection

```
λ_B = [β_I(I_B+I_N) + β_H(H_B+H_N) + β_FS·F_S] / N
λ_N = [β_I(I_B+I_N) + β_H(H_B+H_N) + β_FR·F_R] / N
```

The only structural difference between the two groups is the last term:
Sceptics interact with reclaimed bodies; Believers do not.

### Opinion-dynamics layer

B↔N conversion rates:

```
μ_BN(t) = α·φ(t) + δ_C·C(t)       — social contagion + conflict amplification
μ_NB(t) = γ_comm + β_D·D_vis(t)   — health communication + visible deaths
```

The sceptic proportion φ(t) satisfies:

```
dφ/dt =  α·φ(1−φ)              — scepticism spreads person-to-person
        − β_D·D_vis(t)·φ       — visible deaths erode denial
        − γ_comm·φ             — health communication shifts opinion
        + δ_C·C(t)·(1−φ)      — conflict recruits Believers into Scepticism
```

C(t) is a piecewise step function anchored to five documented security events
in Ituri and North Kivu (see `acled_pipeline.py` and the Data section below).

### Analytical reproduction number

Using the Next Generation Matrix (van den Driessche & Watmough, 2002),
R₀ is the dominant eigenvalue of a 2×2 effective NGM matrix **M**:

```
R₀ = [tr(M) + √(tr(M)² − 4·det(M))] / 2

tr(M)  = (1−φ₀)·R₀_B + φ₀·R₀_N
det(M) = φ₀(1−φ₀)·R₀_B · [β_FR/ω_FR · burial term] ≥ 0
```

The weighted average tr(M) equals R₀ only when β_FR = 0 (no body
reclamation). When β_FR > 0 the exact R₀ falls below tr(M) by
det(M)/tr(M), which is under 10% across the posterior parameter range.

Group-specific reproduction numbers at posterior medians:

```
R₀_B ≈ 1.641   (β_I=0.826, θ_B=0.28 fixed)
R₀_N ≈ 3.247   (β_FR=1.610, θ_N=0.04)
```

Posterior-median R₀ ≈ **2.17** (MCMC median across all 8 000 draws).
Plug-in estimate from parameter medians: 2.172 — negligible Jensen gap
(<0.1%) with 127 confirmed cases.

A homogeneous model fitted to the same data yields R₀ ≈ 1.80, which
**underestimates the SEIHRF-OD value by 21%**.

---

## Calibration results

Data: 127 confirmed cases, INRB-UMIE/Ebola_DRC_2026 build `13d78cb`
(SitReps 001–012, data freeze 26 May 2026).

| Parameter | Posterior median | 95% CrI | Status |
|---|---|---|---|
| β_I (community transmission, day⁻¹) | 0.826 | 0.69–0.96 | Data-informed |
| β_FR (reclaimed-body transmission, day⁻¹) | 1.610 | 1.12–2.09 | Prior-dominated |
| φ₀ (initial scepticism) | 0.392 | 0.29–0.49 | Data-informed |
| θ_N (sceptic hospitalisation rate, day⁻¹) | prior | — | Prior-dominated |
| α, γ_comm, δ_C | prior | — | Prior-dominated |
| **R₀** | **2.17** | **1.82–2.54** | Derived |

Convergence: max R̂ = 1.0011, min ESS = 3 042 (4 chains × 2 000 draws each).

β_FR, α, and δ_C yield flat profile-likelihood curves (non-identifiable
at the 95% level with the current data series). β_I and φ₀ are
one-sided identifiable. All headline conclusions hold across the full
prior support of β_FR (see `bfr_robustness.py` and Supp Table S1).

---

## Counterfactual scenarios

From the calibrated posterior (127 cases; cumulative deaths at day 90):

| | Intervention | Deaths averted (median; 95% CrI) |
|---|---|---|
| S1 | Double communication rate from day 14 (γ_comm × 2) | **20%** (3–42%) |
| S2 | Halve conflict intensity throughout (C(t) × 0.5) | **16%** (3–38%) |
| S3 | Eliminate body reclamation (β_FR = 0) | **29%** (11–56%) |
| S1+S3 | Combined | **44%** (16–67%) |

**Primary inference:** the relative ranking of intervention channels
(S3 > S1 > S2) is more robust than absolute death projections, which
depend on 100-fold extrapolation beyond the calibration window.
Sweeping β_FR across its prior 5th–95th percentile [1.19, 2.01] day⁻¹
keeps S3 deaths averted in the range 26–36% and R₀_N > R₀_B at all
values tested.

---

## Sensitivity analysis

Global first-order Sobol sensitivity indices for cumulative deaths at day 90:

| Parameter | S_i | Driver role |
|---|---|---|
| β_I (community transmission) | **0.60** | Primary |
| γ_comm (communication rate) | 0.15 | Secondary |
| δ_C (conflict amplification) | 0.12 | Secondary |
| α, β_FR, φ₀, θ_N | ≤ 0.06 each | Minor |

β_I is dominant because it updates substantially from its prior with
127 cases. β_FR remains prior-dominated; its low Sobol index reflects
the current identifiability limit, not a low physical importance.

C(t) sensitivity: perturbing each of the five conflict-intensity anchors
independently by ±30% changes cumulative 90-day deaths by at most **7.6%**
(Anchor 5, persistent insecurity, days 30+). Anchors 2–4 each produce
changes below 1.2%.

---

## Validation

### Posterior predictive check (full dataset)

Using all 8 000 posterior draws on the 13-day observed series:

| | 50% CrI | 95% CrI |
|---|---|---|
| Coverage (13 days) | 6/13 (46%) | 11/13 (85%) |

### Hold-out validation — 8/3 split

Calibration on days 1–10 (14–23 May 2026, 105 cases) with days 11–13
(24–26 May, 21 cases) held out as the validation set. The split places
the full acute conflict peak in calibration and tests forecast accuracy
on the subsequent persistent-insecurity regime.

Calibration posterior (T=10): β_I=0.832 [0.710, 0.956], R₀=2.19 [1.87, 2.52],
max R̂=1.001, min ESS=3 133 — consistent with full-data posterior.

| | 50% CrI | 95% CrI |
|---|---|---|
| Calibration coverage (10 days) | 3/10 (30%) | 9/10 (90%) |
| **Forecast coverage (3 held-out days)** | 1/3 (33%) | **3/3 (100%)** ✅ |

All three held-out observations (24–26 May) fall within the 95% credible
interval of the out-of-sample forecast.

---

## Data

Epidemiological input: INSP daily situation reports, sourced from
[INRB-UMIE/Ebola\_DRC\_2026](https://github.com/INRB-UMIE/Ebola_DRC_2026),
build `13d78cb` (data freeze 26 May 2026; accessed 28 May 2026).

```
SitReps included: 001–012 (SitRep 003 missing)
Confirmed cases at calibration: 127
Files:
  insp_sitrep__new_confirmed_cases__daily.csv
  insp_sitrep__cumulative_confirmed_cases__daily.csv
  insp_sitrep__cumulative_confirmed_deaths__daily.csv
  insp_sitrep__cumulative_contacts_isolated__daily.csv
  insp_sitrep__new_contacts_listed__daily.csv
```

Conflict-intensity function C(t): piecewise step function anchored to
five documented security events (see `acled_pipeline.py`):

| Anchor | Date (model day) | C value | Event |
|---|---|---|---|
| 1 | Pre-epidemic (< day 17) | 0.30 | OCHA Ituri Q1 2026 background |
| 2 | Day 17 (11 May 2026) | 0.55 | US health worker exposed at Nyankunde |
| 3 | Day 24 (18 May 2026) | 0.65 | CDC announcement + Berlin evacuation |
| 4 | Days 27–29 (21–23 May 2026) | 1.00 | Rwampara/Mongbwalu tent burnings |
| 5 | Day 30+ | 0.60 | Persistent insecurity; >100 000 displaced |

WHO situation reports:
[DON602](https://www.who.int/emergencies/disease-outbreak-news/item/2026-DON602) ·
[DON603](https://www.who.int/emergencies/disease-outbreak-news/item/2026-DON603)

---

## Citation

```bibtex
@article{kasereka2026seihrf,
  author  = {Kasereka, Selain K. and Kyamakya, Kyandoghere
             and Muyembe, Jean-Jacques T.},
  title   = {Political distrust, armed conflict, and body reclamation
             drive {Ebola} transmission dynamics in eastern {DRC}:
             a coupled epidemic and opinion dynamics model ({SEIHRF-OD})},
  journal = {The Lancet Infectious Diseases},
  year    = {2026},
  note    = {Under review}
}
```

Data: INRB-UMIE, *Ebola\_DRC\_2026*, build `13d78cb`, GitHub, 2026.
[https://github.com/INRB-UMIE/Ebola\_DRC\_2026](https://github.com/INRB-UMIE/Ebola_DRC_2026)

---

## Licence

Code: MIT. Data files are from INRB, INSP, and WHO; see
[INRB-UMIE/Ebola\_DRC\_2026](https://github.com/INRB-UMIE/Ebola_DRC_2026)
for their respective licences.
