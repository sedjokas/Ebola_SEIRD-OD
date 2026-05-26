# =============================================================================
# run_mcmc.R
# Calibrate the SEIHRF-OD model using CmdStanR
# =============================================================================
library(cmdstanr)
library(posterior)
library(bayesplot)

# ── 1. Compile the Stan model ──────────────────────────────────────────────────
mod <- cmdstan_model("seihrf_od.stan")

# ── 2. Prepare data from INRB-UMIE/Ebola_DRC_2026 ─────────────────────────────
# Replace with actual values from the repo CSV files

# Daily confirmed cases: insp_sitrep__new_confirmed_cases__daily.csv
y_cases <- c(2, 3, 4, 3, 5, 6, 8, 7, 9, 10)  # placeholder — replace with real data

# Initial scepticism from contact-tracing proxy
# r_c = contacts_isolated / contacts_listed (first 5 reporting days)
r_c_mean <- 0.62   # replace with actual r_c from the repo
phi0_obs  <- 1 - r_c_mean   # = 0.38
phi0_obs_sd <- 0.05

stan_data <- list(
  T             = length(y_cases),
  y_cases       = y_cases,
  N_pop         = 120000,         # WorldPop aggregate for affected zones
  phi0_obs      = phi0_obs,
  phi0_obs_sd   = phi0_obs_sd,
  # Conflict anchor parameters (5 anchors x 2 values = 10 elements)
  # Format: start_day, level  (each anchor active until next start_day)
  x_r_conflict  = c( 0, 0.30,    # anchor 1: baseline (OCHA Q1 2026)
                    17, 0.55,    # anchor 2: Nyankunde exposure (day 17)
                    24, 0.65,    # anchor 3: CDC announcement (day 24)
                    27, 1.00,    # anchor 4: peak cluster Rwampara+Mongbwalu
                    30, 0.60),   # anchor 5: persistent insecurity
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
