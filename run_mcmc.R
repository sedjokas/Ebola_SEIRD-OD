# =============================================================================
# run_mcmc.R
# Calibrate the SEIHRF-OD model using CmdStanR
# =============================================================================
library(cmdstanr)
library(posterior)
library(bayesplot)

# ── 1. Compile the Stan model ──────────────────────────────────────────────────
mod <- cmdstan_model("seihrf_od.stan")

# ── 2. Prepare data from kraemer-lab/Ebola_DRC_2026 ──────────────────────────
# Data downloaded from: https://github.com/kraemer-lab/Ebola_DRC_2026
# (data/insp_sitrep/processed/ directory)

library(dplyr)
library(readr)
library(lubridate)

# Daily confirmed cases — aggregate across health zones, fill date gaps
cases_raw <- read_csv("data/insp_sitrep__new_confirmed_cases__daily.csv",
                      col_types = cols(nom = col_character(),
                                       date = col_date(),
                                       new_confirmed_cases = col_double()))
cases_daily <- cases_raw %>%
  group_by(date) %>%
  summarise(n = sum(new_confirmed_cases, na.rm = TRUE), .groups = "drop") %>%
  arrange(date)

# Fill contiguous date gaps with 0
all_dates <- seq(min(cases_daily$date), max(cases_daily$date), by = "day")
cases_full <- tibble(date = all_dates) %>%
  left_join(cases_daily, by = "date") %>%
  mutate(n = replace_na(n, 0))

y_cases <- as.integer(cases_full$n)
cat("T =", length(y_cases), "days |",
    "Total cases =", sum(y_cases), "\n")
cat("Date range:", format(min(cases_full$date)), "→",
                   format(max(cases_full$date)), "\n")

# Initial scepticism from contact-tracing proxy
# r_c = cumulative_contacts_isolated / cumulative_contacts_listed
# Using paper estimate phi0_obs = 0.38 (1 - r_c, first 5 SitRep days)
phi0_obs    <- 0.38
phi0_obs_sd <- 0.05

stan_data <- list(
  T             = length(y_cases),
  y_cases       = y_cases,
  N_pop         = 120000,       # WorldPop GRID3 v4.4 — affected health zones
  phi0_obs      = phi0_obs,
  phi0_obs_sd   = phi0_obs_sd,
  # Conflict-intensity anchors — 5 documented security events
  # [start_day, level] × 5 = 10 values; see manuscript Methods — Data sources
  x_r_conflict  = c( 0, 0.30,  # anchor 1: OCHA Q1 2026 baseline (DON602)
                    17, 0.55,  # anchor 2: Nyankunde exposure, 11 May (CDC)
                    24, 0.65,  # anchor 3: CDC announcement + Berlin evac., 18 May
                    27, 1.00,  # anchor 4: Rwampara + Mongbwalu cluster, 21-23 May
                    30, 0.60), # anchor 5: persistent insecurity (DON603)
  rel_tol   = 1e-6,
  abs_tol   = 1e-8,
  max_steps = 10000L
)

# ── 3. Run MCMC ───────────────────────────────────────────────────────────────
fit <- mod$sample(
  data            = stan_data,
  chains          = 4,
  parallel_chains = 4,
  iter_warmup     = 2000,
  iter_sampling   = 2000,
  adapt_delta     = 0.95,
  max_treedepth   = 12,
  seed            = 42
)

# ── 4. Convergence diagnostics ────────────────────────────────────────────────
print(fit$summary(c("beta_I","beta_FR","phi0","theta_N","alpha",
                    "gamma_comm","delta_C","phi_obs","R0")))

# R-hat check (all should be < 1.01)
rhat_vals <- fit$summary()$rhat
cat("Max R-hat:", max(rhat_vals, na.rm=TRUE), "\n")
cat("Min ESS: ", min(fit$summary()$ess_bulk, na.rm=TRUE), "\n")

# ── 5. Posterior predictive check ─────────────────────────────────────────────
y_rep <- fit$draws("y_rep", format = "matrix")
ppc_dens_overlay(y = y_cases,
                 yrep = y_rep[sample(nrow(y_rep), 100), ])

# ── 6. LOO cross-validation ───────────────────────────────────────────────────
library(loo)
log_lik <- fit$draws("log_lik", format = "matrix")
loo_result <- loo(log_lik)
print(loo_result)
