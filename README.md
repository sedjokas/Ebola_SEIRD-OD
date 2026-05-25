# Ebola SEIRD-OD — Coupled epidemic–opinion dynamics model

Code and figures for the manuscript:

> **"Political distrust, armed conflict, and body reclamation drive Ebola
> transmission dynamics in eastern Democratic Republic of Congo:
> a coupled epidemic–opinion dynamics model (SEIRD-OD)"**
>
> *Submitted to The Lancet Infectious Diseases, 2026*
>
> Institut National de Recherche Biomédicale (INRB) / One Health Institute
> for Africa (INOHA), Kinshasa, DRC · Institut National de Santé Publique
> (INSP), Kinshasa, DRC · University of Oxford, UK ·
> Northeastern University, Boston, MA, USA

---

## Background

Standard compartmental models of Ebola virus disease (EVD) assume behavioural
homogeneity. The 2026 Bundibugyo ebolavirus (BDBV) outbreak in eastern DRC
violates this: a significant fraction of the population in conflict-affected
Ituri, North Kivu, and South Kivu provinces denies the existence of the
disease, refuses hospitalisation, and reclaims the bodies of deceased
relatives for traditional burial — creating catastrophic secondary
transmission.

The **SEIRD-OD** model addresses this gap by coupling:

- A **dual-population epidemic layer** distinguishing *Believers* (B) and
  *Sceptics* (N)
- A **reclaimed-body compartment** F\_R with transmission coefficient
  β\_FR independently estimated from the data
- A **time-varying opinion process** φ(t) driven by social contagion,
  visible deaths, health communication, and armed-conflict intensity C(t)

Mathematical verification was performed symbolically with SymPy 1.13 (five
consistency checks); Bayesian calibration uses Stan 2.35 (MCMC).

---

## Repository structure

```
Ebola_SEIRD-OD/
├── src/
│   ├── __init__.py
│   └── seird_od_model.py   # ODE system, Params dataclass, R0 analytics
├── figures.py              # All six publication figures
├── requirements.txt
├── README.md
└── figures/                # Generated output (PDF + PNG, 300 dpi)
    ├── fig1_epidemic_opinion.{pdf,png}
    ├── fig2_R0_analysis.{pdf,png}
    ├── fig3_scenarios.{pdf,png}
    ├── fig4_sensitivity.{pdf,png}
    ├── fig5_spatial.{pdf,png}
    └── figS1_Rt.{pdf,png}
```

---

## Quick start

```bash
git clone https://github.com/sedjokas/Ebola_SEIRD-OD.git
cd Ebola_SEIRD-OD

pip install -r requirements.txt

# Generate all figures (saved to figures/)
python figures.py

# Generate only figures 1, 2, and 3
python figures.py --fig 1 2 3

# Custom output directory
python figures.py --outdir my_output_dir
```

---

## Model overview

### State variables

The 12-dimensional ODE system tracks:

| Symbol | Description |
|--------|-------------|
| S\_B, E\_B, I\_B, H\_B, R\_B | Susceptible / Exposed / Infectious / Hospitalised / Recovered — **Believers** |
| S\_N, E\_N, I\_N, H\_N, R\_N | Same compartments — **Sceptics** |
| F\_R | Reclaimed bodies (unsafe burial — high β\_FR) |
| F\_S | Safe-burial bodies (negligible transmission) |

### Forces of infection

```
λ_B(t) = [β_I(I_B + I_N) + β_H(H_B + H_N) + β_FS · F_S] / N(t)
λ_N(t) = [β_I(I_B + I_N) + β_H(H_B + H_N) + β_FR · F_R] / N(t)
```

Sceptics are exposed to F\_R (they participate in unsafe burial rites);
Believers are not.

### Opinion dynamics

```
dφ/dt = α·φ(1−φ)          [social contagion of scepticism]
      − β_D · D_vis(t) · φ  [visible deaths reduce scepticism]
      − γ_comm · φ           [health communication reduces scepticism]
      + δ_C · C(t) · φ       [conflict amplifies scepticism]
```

where φ(t) = N\_N(t)/N(t) and D\_vis(t) is the per-capita instantaneous
death rate. C(t) is the armed-conflict intensity function (ACLED proxy).

### Analytical R₀

Via the Next Generation Matrix (van den Driessche & Watmough, 2002):

```
R0 = (1 − φ₀) · R0_B + φ₀ · R0_N
```

where R0\_N includes the **unsafe-funeral amplifier** β\_FR / ω\_FR that
is absent from all classical SEIHFR models. This term causes homogeneous
models to underestimate R₀ by ~33% at φ₀ = 0.38.

---

## Counterfactual scenarios

| Scenario | Intervention | Model change |
|----------|-------------|--------------|
| S1 | Enhanced communication (from day 14) | γ\_comm → 2γ\_comm |
| S2 | 50% conflict reduction | C(t) → 0.5 C(t) |
| S3 | Zero body reclamation | β\_FR → 0 |
| S1+S3 | Combined | Both |

Estimated deaths averted at day 90 (posterior median, 95% CrI):

| Scenario | Deaths averted | 95% CrI |
|----------|---------------|---------|
| S1 | 34% | 21–47% |
| S2 | 18% | 10–26% |
| S3 | 41% | 29–53% |
| S1+S3 | 58% | 44–70% |

---

## Data sources

All epidemiological data from the public INRB-UMIE repository:

```
https://github.com/INRB-UMIE/Ebola_DRC_2026
Build 235a3c3 / data snapshot 493d506, accessed 22 May 2026
```

Specifically used:
- `insp_sitrep__new_confirmed_cases__daily.csv` (SitReps 001–007)
- `insp_sitrep__cumulative_confirmed_deaths__daily.csv`
- `insp_sitrep__cumulative_contacts_isolated__daily.csv`
- `insp_sitrep__new_contacts_listed__daily.csv`
- `epi__cases__weekly.csv` (WHO, data as of 18 May 2026)
- `worldpop__pop_count__static.csv`
- `grid3_healthsites__healthsite_density__static.csv`
- `flowminder__inflow__static.matrix.csv` / `outflow`
- `osrm__travel_time__static.matrix.csv`
- `ccvi__socioeconomic_deprivation__static.csv`

ACLED conflict data pending QA in the INRB-UMIE repo (issue #14); C(t)
approximated from publicly reported events in the interim.

WHO Disease Outbreak News:
- DON602: https://www.who.int/emergencies/disease-outbreak-news/item/2026-DON602
- DON603: https://www.who.int/emergencies/disease-outbreak-news/item/2026-DON603

---

## Usage in Python

```python
from src.seird_od_model import SEIRD_OD, Params, DEFAULT_PARAMS, R0_analytical
import numpy as np

# Run baseline model
model = SEIRD_OD(DEFAULT_PARAMS)
sol   = model.run(t_end=90)

# Extract outputs
phi          = model.phi(sol)            # scepticism φ(t)
cum_deaths   = model.cumulative_deaths(sol)
incidence    = model.daily_incidence(sol)

# Scenario S3: zero body reclamation
sol_s3 = SEIRD_OD(Params(beta_FR_scale=0.0)).run()

# Analytical R0
r = R0_analytical(DEFAULT_PARAMS)
print(f"R0 = {r['R0']:.2f}  (R0_B = {r['R0_B']:.2f}, R0_N = {r['R0_N']:.2f})")
```

---

## Mathematical verification

Five symbolic checks (SymPy 1.13) confirm model consistency:

1. **Population flow balance** — belief-conversion terms sum to zero
2. **F\_R mass balance** — inflow equals ψ-weighted deaths from N
3. **Death partitioning** — F\_R + F\_S(from N) = total deaths from N ✓
4. **R₀ via NGM** — spectral radius of FV⁻¹ reduces to closed-form Eqs. (5–6)
5. **φ ODE algebraic consistency** — dφ/dt derived exactly from dS\_N/dt
   and dS\_B/dt ✓

---

## Citation

If you use this code, please cite:

```
Selain K. Kasereka et al.
"Political distrust, armed conflict, and body reclamation drive Ebola
transmission dynamics in eastern Democratic Republic of Congo:
a coupled epidemic–opinion dynamics model (SEIRD-OD)"
Submitted to The Lancet Infectious Diseases, 2026.
Data: INRB-UMIE/Ebola_DRC_2026, build 235a3c3, GitHub, 2026.
```

Please also cite the original data providers (INRB, INSP, WHO, Flowminder,
GRID3, WorldPop, ACLED, OSRM) — see INRB-UMIE/Ebola_DRC_2026 for links.

---

## Licence

Code: MIT — see [LICENSE](LICENSE).  
Data: subject to the licences of the original providers (see
INRB-UMIE/Ebola_DRC_2026 for details). No ownership of third-party data
is claimed.
