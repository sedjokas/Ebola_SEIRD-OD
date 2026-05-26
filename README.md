# Ebola SEIHRF-OD

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
diagnosis (Believers, *B*) and people who deny it (Sceptics, *N*). Sceptics
avoid hospitals and their deceased relatives' bodies end up in a separate
compartment **F_R** (reclaimed bodies), with transmission coefficient
β_FR that substantially exceeds community and hospital rates. A third layer
tracks how the proportion of sceptics φ(t) changes over time, driven by
social contagion, visible community deaths, health communication, and
armed-conflict intensity C(t).

Calibration uses full Bayesian MCMC (CmdStan) on daily INSP situation-report
data. The basic reproduction number R₀ is derived analytically via the
Next Generation Matrix and computed across all posterior draws.

---

## Repo layout

```
Ebola_SEIHRF-OD/
├── seihrf_od.stan               # Stan ODE model — 12-compartment SEIHRF-OD
├── seihrf_od_profile.stan       # Profile-likelihood wrapper (fixes one param at a time)
├── run_mcmc.R                   # CmdStanR MCMC calibration script
├── profile_likelihood.py        # Profile-likelihood identifiability analysis → figS2
├── sensitivity_Ct.py            # C(t) sensitivity analysis → figS4
├── acled_pipeline.py            # C(t) reconstruction from documented security events
├── figures.py                   # Main publication figures (Python)
├── profile_likelihood_results.csv
├── sensitivity_Ct_results.csv
├── requirements.txt
├── data/
│   ├── insp_sitrep__new_confirmed_cases__daily.csv
│   ├── insp_sitrep__cumulative_confirmed_deaths__daily.csv
│   ├── insp_sitrep__cumulative_contacts_isolated__daily.csv
│   ├── insp_sitrep__new_contacts_listed__daily.csv
│   └── insp_sitrep__cumulative_confirmed_cases__daily.csv
├── manuscript/imgs/             # All publication figures (PDF + PNG)
│   ├── fig1_epidemic_opinion.*
│   ├── fig2_R0_analysis.*
│   ├── fig3_scenarios.*
│   ├── fig4_sensitivity.*
│   ├── fig5_spatial.*
│   ├── figS1_Rt.*
│   ├── figS2_profile_likelihood.pdf
│   └── figS4_sensitivity_Ct.pdf
└── src/
    ├── seihrf_od_model.py       # Python ODE implementation (exploratory)
    └── seird_od_model.py
```

---

## Getting started

### MCMC calibration (R + CmdStan)

```r
# Install dependencies
install.packages(c("cmdstanr", "posterior", "bayesplot", "loo",
                   "dplyr", "readr", "lubridate"))
cmdstanr::install_cmdstan()

# Run
Rscript run_mcmc.R
```

The script compiles `seihrf_od.stan`, loads INSP daily case data from
`data/`, runs 4 chains × 2 000 warm-up + 2 000 sampling iterations, and
prints convergence diagnostics (R̂ < 1.02 for all 8 sampled parameters,
ESS > 450 per chain).

### Profile-likelihood analysis (Python)

```bash
pip install -r requirements.txt
python profile_likelihood.py     # → figS2_profile_likelihood.pdf
python sensitivity_Ct.py         # → figS4_sensitivity_Ct.pdf
```

---

## The model

### State variables (12 total)

| | Believers (B) | Sceptics (N) |
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

The B↔N conversion rates are:

```
μ_BN(t) = α·φ(t) + δ_C·C(t)       — social contagion + conflict amplification
μ_NB(t) = γ_comm + β_D·D_vis(t)   — health communication + visible deaths
```

The sceptic proportion φ(t) = S_N/(S_B+S_N) satisfies:

```
dφ/dt =  α·φ(1−φ)              — scepticism spreads person-to-person
        − β_D·D_vis(t)·φ       — visible deaths erode denial
        − γ_comm·φ             — health communication shifts opinion
        + δ_C·C(t)·(1−φ)      — conflict recruits Believers into Scepticism
```

C(t) is a piecewise function reconstructed from five documented security
events in Ituri and North Kivu (see `acled_pipeline.py` and manuscript
Methods — Data sources). ACLED data were not used.

### Analytical reproduction number

Using the Next Generation Matrix (van den Driessche & Watmough, 2002),
the basic reproduction number is the **dominant eigenvalue** of a 2×2
effective NGM matrix **M**:

```
R₀ = [tr(M) + √(tr(M)² − 4·det(M))] / 2
```

where

```
tr(M)  = (1−φ₀)·R₀_B + φ₀·R₀_N
det(M) = φ₀(1−φ₀)·R₀_B · [β_FR/ω_FR · burial term] ≥ 0
```

The weighted average tr(M) equals the exact R₀ only when β_FR = 0.
When body reclamation is present (β_FR > 0), the exact R₀ is below
tr(M) by a margin bounded by det(M)/tr(M) — under 10% across the
posterior parameter range.

The group-specific reproduction numbers at posterior medians are:

```
R₀_B ≈ 1.48   (β_I=0.74, θ_B=0.28 fixed)
R₀_N ≈ 2.97   (β_FR=1.62, θ_N=0.04)
```

Posterior-median R₀ ≈ **2.40** (MCMC median across all draws). The
plug-in estimate from parameter medians (1.94) underestimates the MCMC
value due to the nonlinearity of the dominant-eigenvalue formula
(Jensen effect; see Supplementary B).

A homogeneous model fitted to the same data yields R₀ ≈ 1.8, which
**underestimates the SEIHRF-OD value by 33%**.

---

## Calibration results

| Parameter | Posterior median | 95% CrI |
|---|---|---|
| β_I (community transmission) | 0.74 day⁻¹ | 0.60–0.91 |
| β_FR (reclaimed-body transmission) | 1.62 day⁻¹ | 1.14–2.11 |
| φ₀ (initial scepticism) | 0.37 | 0.27–0.50 |
| θ_N (sceptic hospitalisation rate) | 0.04 day⁻¹ | — |
| R₀ | **2.40** | — |

Convergence: R̂ < 1.02 for all 8 sampled parameters; ESS > 450 per chain.
83 confirmed cases across SitReps 001–007 available at calibration.

---

## Counterfactual scenarios

| | Intervention | Deaths averted at day 90 (95% CrI) |
|---|---|---|
| S1 | Double communication rate from day 14 (γ_comm × 2) | **47%** (27–56%) |
| S2 | Halve conflict intensity (C(t) × 0.5) | **20%** (9–31%) |
| S3 | Eliminate body reclamation (β_FR = 0) | **38%** (20–57%) |
| S1+S3 | Combined | **61%** (41–73%) |

CrIs are 95% posterior credible intervals across 8 000 MCMC draws
(4 chains × 2 000 samples).

---

## Identifiability

Profile-likelihood analysis (Supplementary Figure S2) over 5 parameters:

| Parameter | Status |
|---|---|
| β_I | Identifiable |
| β_FR | Identifiable |
| φ₀ | Identifiable |
| α (social contagion) | Prior-dominated — flat profile |
| δ_C (conflict amplification) | Prior-dominated — flat profile |

α and δ_C are not identifiable from the current 10-day sitrep window.
Their posteriors should be interpreted with caution.

---

## Data

Epidemiological input data: INSP daily situation reports (SitReps 001–007),
sourced from [kraemer-lab/Ebola\_DRC\_2026](https://github.com/kraemer-lab/Ebola_DRC_2026)
(build `235a3c3`, accessed 22 May 2026). Files in `data/`:

```
insp_sitrep__new_confirmed_cases__daily.csv        SitReps 001–007
insp_sitrep__cumulative_confirmed_cases__daily.csv
insp_sitrep__cumulative_confirmed_deaths__daily.csv
insp_sitrep__cumulative_contacts_isolated__daily.csv
insp_sitrep__new_contacts_listed__daily.csv
```

Conflict-intensity function C(t): reconstructed from five documented
security events in Ituri and North Kivu (see `acled_pipeline.py`).
Sources: OCHA situation reports, CDC DON602/DON603, Le Devoir, WHO DON603.

WHO situation reports used for validation:
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

Data: kraemer-lab, *Ebola\_DRC\_2026*, build `235a3c3`, GitHub, 2026.

---

## Licence

Code: MIT. Data files are from INRB, INSP, and WHO; see
[kraemer-lab/Ebola\_DRC\_2026](https://github.com/kraemer-lab/Ebola_DRC_2026)
for their respective licences.
