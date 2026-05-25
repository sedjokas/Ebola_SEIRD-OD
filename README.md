# Ebola SEIRD-OD

This repository contains the code and figures for:

> Selain K. Kasereka et al., "Political distrust, armed conflict, and body
> reclamation drive Ebola transmission dynamics in eastern Democratic Republic
> of Congo: a coupled epidemic–opinion dynamics model (SEIRD-OD)"
> — under review, *The Lancet Infectious Diseases*, 2026
>
> INRB/INOHA Kinshasa · INSP Kinshasa · University of Oxford · Northeastern University

---

## What this is

When families in eastern DRC started reclaiming the bodies of relatives who
died of Ebola from treatment centres — sometimes violently — and carrying them
home for traditional burial, existing epidemic models had no way to represent
that. Those models assume everyone follows public-health guidance. Here they
clearly did not.

SEIRD-OD splits the population into two groups: people who accept the EVD
diagnosis (Believers, *B*) and people who deny it (Sceptics, *N*). Sceptics
avoid hospitals and their deceased relatives' bodies end up in a separate
compartment, F_R, with a transmission rate far higher than a safe burial.
A third layer tracks how the proportion of sceptics φ(t) changes over time,
driven by social contagion, visible community deaths, health communication,
and armed-conflict intensity C(t).

The model was verified algebraically with SymPy before any numerical work
(population balance, body-compartment mass balance, Next Generation Matrix R₀,
consistency of the φ ODE with the S-layer equations). Calibration uses MCMC
on INSP daily situation-report data and WHO weekly case counts.

---

## Repo layout

```
Ebola_SEIRD-OD/
├── src/seird_od_model.py   # ODE right-hand side, Params dataclass, R0 formula
├── figures.py              # Six publication figures, CLI interface
├── requirements.txt
└── figures/                # Output directory (PDF + PNG at 300 dpi)
```

---

## Getting started

```bash
git clone https://github.com/sedjokas/Ebola_SEIRD-OD.git
cd Ebola_SEIRD-OD
pip install -r requirements.txt

python figures.py              # all six figures
python figures.py --fig 1 2 3  # selected figures only
python figures.py --outdir out  # custom output path
```

---

## The model

### State variables (12 total)

| | Believers (B) | Sceptics (N) |
|---|---|---|
| Susceptible | S_B | S_N |
| Exposed | E_B | E_N |
| Infectious | I_B | I_N |
| Hospitalised | H_B | H_N (rarely used) |
| Recovered | R_B | R_N |

Plus two post-mortem compartments: F_S (safe burial, ~zero transmission)
and F_R (reclaimed body, high transmission).

### Forces of infection

```
λ_B = [β_I(I_B+I_N) + β_H(H_B+H_N) + β_FS·F_S] / N
λ_N = [β_I(I_B+I_N) + β_H(H_B+H_N) + β_FR·F_R] / N
```

The only structural difference between the two groups is the last term:
Sceptics handle the bodies; Believers do not.

### Opinion dynamics

```
dφ/dt = α·φ(1−φ)           — scepticism spreads person-to-person
       − β_D·D_vis(t)·φ    — visible deaths erode denial
       − γ_comm·φ           — health communication shifts opinion
       + δ_C·C(t)·φ         — conflict reinforces distrust
```

### Analytical R₀

Using the Next Generation Matrix (van den Driessche & Watmough, 2002):

```
R₀ = (1 − φ₀)·R₀_B + φ₀·R₀_N
```

R₀_N contains an extra term, β_FR/ω_FR, that is absent from every classical
SEIHFR model. At φ₀ = 0.38 (estimated from INSP contact-tracing data),
a homogeneous model underestimates R₀ by roughly 33%.

---

## Scenarios

| | Intervention | Change | Deaths averted (day 90) |
|---|---|---|---|
| S1 | Double communication rate from day 14 | γ_comm × 2 | 34% (21–47%) |
| S2 | Halve conflict intensity | C(t) × 0.5 | 18% (10–26%) |
| S3 | Enforce safe burial throughout | β_FR = 0 | 41% (29–53%) |
| S1+S3 | Both | — | 58% (44–70%) |

Intervals are 95% posterior credible intervals from 80 MCMC draws.

---

## Data

All epidemiological input data come from
[INRB-UMIE/Ebola_DRC_2026](https://github.com/INRB-UMIE/Ebola_DRC_2026)
(build `235a3c3`, snapshot `493d506`, accessed 22 May 2026). Files used:

```
insp_sitrep__new_confirmed_cases__daily.csv      SitReps 001–007
insp_sitrep__cumulative_confirmed_deaths__daily.csv
insp_sitrep__cumulative_contacts_isolated__daily.csv
insp_sitrep__new_contacts_listed__daily.csv
epi__cases__weekly.csv                           WHO, as of 18 May 2026
worldpop__pop_count__static.csv
grid3_healthsites__healthsite_density__static.csv
flowminder__inflow__static.matrix.csv
flowminder__outflow__static.matrix.csv
osrm__travel_time__static.matrix.csv
ccvi__socioeconomic_deprivation__static.csv
```

ACLED conflict data for C(t) are still pending QA in the upstream repo
(issue #14). Until they clear, C(t) is reconstructed from publicly reported
security events in Ituri and North Kivu.

WHO situation reports used for validation:
[DON602](https://www.who.int/emergencies/disease-outbreak-news/item/2026-DON602) ·
[DON603](https://www.who.int/emergencies/disease-outbreak-news/item/2026-DON603)

---

## Usage

```python
from src.seird_od_model import SEIRD_OD, Params, DEFAULT_PARAMS, R0_analytical

model = SEIRD_OD(DEFAULT_PARAMS)
sol   = model.run(t_end=90)

phi        = model.phi(sol)
deaths     = model.cumulative_deaths(sol)
incidence  = model.daily_incidence(sol)

# Scenario S3 — zero body reclamation
sol_s3 = SEIRD_OD(Params(beta_FR_scale=0.0)).run()

r = R0_analytical(DEFAULT_PARAMS)
# {'R0': 1.2, 'R0_B': 0.83, 'R0_N': 1.8, 'phi0': 0.38}
# (values shift substantially after full MCMC calibration)
```

---

## Citation

```bibtex
@article{kasereka2026seirdod,
  author  = {Kasereka, Selain K. and others},
  title   = {Political distrust, armed conflict, and body reclamation drive
             {Ebola} transmission dynamics in eastern {DRC}:
             a coupled epidemic--opinion dynamics model ({SEIRD-OD})},
  journal = {The Lancet Infectious Diseases},
  year    = {2026},
  note    = {Under review}
}
```

Data citation: INRB-UMIE, *Ebola\_DRC\_2026*, build `235a3c3`, GitHub, 2026.
Please also acknowledge the upstream data providers — INRB, INSP, WHO,
Flowminder, GRID3, WorldPop, ACLED, OSRM — per their respective licences.

---

## Licence

Code: MIT. Data files belong to their original providers; see
[INRB-UMIE/Ebola_DRC_2026](https://github.com/INRB-UMIE/Ebola_DRC_2026)
for details.
