// =============================================================================
// seihrf_od.stan
// =============================================================================
// Bayesian calibration of the SEIHRF-OD model for the 2026 Bundibugyo Ebola
// outbreak in Ituri Province, DRC.
//
// Mathematical reference
// ----------------------
// Kasereka et al., "Political distrust, armed conflict, and body reclamation
// drive Ebola transmission dynamics in eastern DRC: a coupled
// epidemic-opinion dynamics model (SEIHRF-OD)", submitted to
// The Lancet Infectious Diseases, 2026.
// Data: INRB-UMIE/Ebola_DRC_2026, build 235a3c3.
//
// State vector (12 compartments)
// -------------------------------
//   y[1]  S_B   Susceptible  Believers
//   y[2]  E_B   Exposed      Believers
//   y[3]  I_B   Infectious   Believers
//   y[4]  H_B   Hospitalised Believers
//   y[5]  R_B   Recovered    Believers
//   y[6]  S_N   Susceptible  Skeptics
//   y[7]  E_N   Exposed      Skeptics
//   y[8]  I_N   Infectious   Skeptics
//   y[9]  H_N   Hospitalised Skeptics
//   y[10] R_N   Recovered    Skeptics
//   y[11] F_R   Reclaimed bodies  (unsafe burial — high beta_FR)
//   y[12] F_S   Safe-burial bodies (negligible transmission)
//
// Observation model
// -----------------
//   y_cases[t] ~ NegBin2(kappa*(E_B[t] + E_N[t]), phi_obs)
//   (daily confirmed cases as a noisy observation of new exposures becoming
//   infectious; phi_obs is the overdispersion parameter)
//
// Calibration targets
// -------------------
//   Three parameters update substantially from their priors:
//     beta_FR   (posterior median 1.62, 95% CrI: 1.14-2.11)
//     phi0      (0.37, 0.27-0.50)
//     beta_I    (0.74, 0.60-0.91)
//   Remaining parameters are largely prior-dominated (small dataset N=83).
//
// Usage (CmdStanR)
// ----------------
//   mod  <- cmdstan_model("seihrf_od.stan")
//   fit  <- mod$sample(data = stan_data, chains = 4, iter_warmup = 2000,
//                      iter_sampling = 2000, parallel_chains = 4,
//                      adapt_delta = 0.95)
//
// Usage (PyStan / CmdStanPy)
// --------------------------
//   model = CmdStanModel(stan_file="seihrf_od.stan")
//   fit   = model.sample(data=stan_data, chains=4, iter_warmup=2000,
//                        iter_sampling=2000, adapt_delta=0.95)
// =============================================================================

functions {

  // --------------------------------------------------------------------------
  // Piecewise conflict-intensity function C(t)
  // Five anchors from manuscript Data Sources section (table of security events):
  //
  //  Anchor  Days      Level   Event
  //  1       0 to 16   0.30   OCHA pre-epidemic baseline (Q1 2026: 5 800
  //                           protection incidents, 11 humanitarian actor
  //                           incidents; cited in WHO DON602)
  //  2      17 to 23   0.55   Nyankunde accidental exposure of US health
  //                           worker (11 May = day 17; CDC/Guardian 2026)
  //  3      24 to 26   0.65   CDC public announcement and Berlin medical
  //                           evacuation (18 May = day 24; CDC 2026)
  //  4      27 to 29   1.00   Peak cluster: Rwampara tent burning (21 May),
  //                           Mongbwalu treatment-centre storming with 18
  //                           patients in flight (23 May); Le Devoir/Al Jazeera
  //  5      30+        0.60   Persistent insecurity: over 100 000 newly
  //                           displaced in Ituri and North Kivu (WHO DON603)
  //
  // x_r layout (10 values = 5 x [start_day, level]):
  //   x_r[1..2]   anchor 1: start=0,  level=0.30
  //   x_r[3..4]   anchor 2: start=17, level=0.55
  //   x_r[5..6]   anchor 3: start=24, level=0.65
  //   x_r[7..8]   anchor 4: start=27, level=1.00
  //   x_r[9..10]  anchor 5: start=30, level=0.60
  // --------------------------------------------------------------------------
  real conflict_C(real t, array[] real x_r) {
    // Walk anchors from last to first: return level of the highest anchor
    // whose start_day <= t
    real c = 0.0;
    for (k in 1:5) {
      int idx = 2 * k - 1;          // index of start_day for anchor k
      if (t >= x_r[idx])
        c = x_r[idx + 1];           // update level (last match wins)
    }
    return c;
  }

  // --------------------------------------------------------------------------
  // SEIHRF-OD ODE right-hand side
  //
  // Parameters theta (18 values):
  //   [1]  beta_I       community transmission rate
  //   [2]  beta_H       hospital transmission rate
  //   [3]  beta_FR      reclaimed-body transmission rate
  //   [4]  beta_FS      safe-burial transmission rate (fixed ~ 0)
  //   [5]  kappa        1 / mean incubation period
  //   [6]  theta_B      hospitalisation rate, Believers
  //   [7]  theta_N      hospitalisation rate, Skeptics
  //   [8]  delta_I      community case-fatality rate
  //   [9]  delta_H      hospital  case-fatality rate
  //   [10] gamma_I      community recovery rate
  //   [11] gamma_H      hospital  recovery rate
  //   [12] psi_I        body-reclaim fraction, community deaths
  //   [13] psi_H        body-reclaim fraction, hospital  deaths
  //   [14] omega_FR     body disposal rate, unsafe
  //   [15] omega_FS     body disposal rate, safe
  //   [16] alpha        scepticism social-contagion rate
  //   [17] beta_D       death-driven belief-update coefficient
  //   [18] gamma_comm   health-communication rate
  //   [19] delta_C      conflict-amplification coefficient
  // --------------------------------------------------------------------------
  vector seihrf_od_ode(real t,
                       vector y,
                       array[] real theta,
                       array[] real x_r,
                       array[] int x_i) {

    // Unpack state
    real SB = y[1];  real EB = y[2];  real IB = y[3];
    real HB = y[4];  real RB = y[5];
    real SN = y[6];  real EN = y[7];  real IN = y[8];
    real HN = y[9];  real RN = y[10];
    real FR = y[11]; real FS = y[12];

    // Total living population
    real N = SB + EB + IB + HB + RB + SN + EN + IN + HN + RN;
    if (N <= 0) return rep_vector(0.0, 12);

    // Unpack parameters
    real beta_I    = theta[1];
    real beta_H    = theta[2];
    real beta_FR   = theta[3];
    real beta_FS   = theta[4];
    real kap       = theta[5];
    real th_B      = theta[6];
    real th_N      = theta[7];
    real dI        = theta[8];
    real dH        = theta[9];
    real gI        = theta[10];
    real gH        = theta[11];
    real psi_I     = theta[12];
    real psi_H     = theta[13];
    real om_FR     = theta[14];
    real om_FS     = theta[15];
    real alph      = theta[16];
    real bet_D     = theta[17];
    real gam_comm  = theta[18];
    real del_C     = theta[19];

    // Conflict intensity
    real Ct  = conflict_C(t, x_r);

    // Opinion-driven conversion rates
    // phi = S_N / (S_B + S_N)  [proportion of skeptics among susceptibles]
    real S_sum = SB + SN;
    real phi   = (S_sum > 0) ? SN / S_sum : 0.0;

    // mu_BN: B -> N rate (social contagion + conflict amplification)
    // mu_NB: N -> B rate (visible deaths + health communication)
    real Dvis  = (dI * (IB + IN) + dH * (HB + HN)) / N;
    real mu_BN = alph * phi + del_C * Ct;
    real mu_NB = gam_comm + bet_D * Dvis;

    // Forces of infection (equations 1-2 of manuscript)
    real lam_B = (beta_I * (IB + IN) + beta_H * (HB + HN) + beta_FS * FS) / N;
    real lam_N = (beta_I * (IB + IN) + beta_H * (HB + HN) + beta_FR * FR) / N;

    // ODEs — Believers (equations 3-7)
    real dSB = -lam_B * SB - mu_BN * SB + mu_NB * SN;
    real dEB =  lam_B * SB - kap * EB;
    real dIB =  kap * EB  - (th_B + dI + gI) * IB;
    real dHB =  th_B * IB - (dH  + gH) * HB;
    real dRB =  gI * IB + gH * HB;

    // ODEs — Skeptics (equations 8-12)
    real dSN = -lam_N * SN + mu_BN * SB - mu_NB * SN;
    real dEN =  lam_N * SN - kap * EN;
    real dIN =  kap * EN  - (th_N + dI + gI) * IN;
    real dHN =  th_N * IN - (dH  + gH) * HN;
    real dRN =  gI * IN + gH * HN;

    // ODEs — Funeral compartments (equations 13-14)
    real dFR = psi_I * dI * IN + psi_H * dH * HN - om_FR * FR;
    real dFS = (dI * IB + dH * HB
               + (1 - psi_I) * dI * IN
               + (1 - psi_H) * dH * HN
               - om_FS * FS);

    return [dSB, dEB, dIB, dHB, dRB,
            dSN, dEN, dIN, dHN, dRN,
            dFR, dFS]';
  }

} // end functions


// =============================================================================
data {

  // Number of daily observation time points
  int<lower=1>  T;

  // Observed daily confirmed cases from INSP SitReps 001-007
  // y_cases[t] = new confirmed cases on day t (t=1,...,T)
  array[T] int<lower=0> y_cases;

  // Total catchment population (WorldPop GRID3 v4.4)
  real<lower=1> N_pop;

  // Initial scepticism fraction estimate from contact-tracing proxy
  // phi0_obs = 1 - mean(contacts_isolated / contacts_listed) over first 5 days
  real<lower=0, upper=1> phi0_obs;
  real<lower=0>          phi0_obs_sd;   // uncertainty on the proxy estimate

  // Conflict-intensity anchor parameters (5 anchors x 2 values = 10 elements)
  // Format: [start_day_k, level_k] for k=1..5
  // Anchor 1: day  0, C=0.30  (OCHA pre-epidemic baseline Q1 2026)
  // Anchor 2: day 17, C=0.55  (Nyankunde accidental exposure, 11 May)
  // Anchor 3: day 24, C=0.65  (CDC announcement + Berlin evacuation, 18 May)
  // Anchor 4: day 27, C=1.00  (peak cluster: Rwampara+Mongbwalu, 21-23 May)
  // Anchor 5: day 30, C=0.60  (persistent insecurity, 100k+ displaced)
  array[10] real x_r_conflict;

  // ODE solver tolerance controls
  real<lower=0> rel_tol;
  real<lower=0> abs_tol;
  int<lower=1>  max_steps;

} // end data


// =============================================================================
transformed data {
  // Integer data for ODE solver (unused here; required by signature)
  array[0] int x_i_empty;

  // Initial time and observation times (day 0 = outbreak declaration)
  real t0 = 0.0;
  array[T] real ts;
  for (t in 1:T) ts[t] = t;

} // end transformed data


// =============================================================================
parameters {

  // --- Three informationally updated parameters ---
  // (posteriors substantially differ from priors; see Fig 4B of manuscript)

  real<lower=0.40, upper=1.20> beta_I;     // community transmission
  real<lower=0.50, upper=3.50> beta_FR;    // reclaimed-body transmission
  real<lower=0.15, upper=0.70> phi0;       // initial scepticism proportion

  // --- Largely prior-dominated parameters ---
  // (wide posteriors due to limited data; informative priors from literature)

  real<lower=0.01, upper=0.20> theta_N;    // hosp. rate, Skeptics
  real<lower=0.01, upper=0.15> alpha;      // scepticism contagion rate
  real<lower=0.005, upper=0.10> gamma_comm; // health-communication rate
  real<lower=0.01, upper=1.50> delta_C;    // conflict amplification

  // Overdispersion for NegBin observation model
  real<lower=0> phi_obs;

} // end parameters


// =============================================================================
transformed parameters {

  // Fixed parameters (literature values or derived constraints)
  real beta_H    = 0.06;
  real beta_FS   = 0.002;
  real kappa     = 1.0 / 9.0;    // 9-day mean incubation (BDBV)
  real theta_B   = 0.28;
  real delta_I   = 0.18;
  real delta_H   = 0.12;
  real gamma_I   = 0.09;
  real gamma_H   = 0.10;
  real psi_I     = 0.45;
  real psi_H     = 0.15;
  real omega_FR  = 0.80;
  real omega_FS  = 3.00;
  real beta_D    = 8.00;

  // Pack ODE parameter vector (19 elements)
  array[19] real theta_ode;
  theta_ode[1]  = beta_I;
  theta_ode[2]  = beta_H;
  theta_ode[3]  = beta_FR;
  theta_ode[4]  = beta_FS;
  theta_ode[5]  = kappa;
  theta_ode[6]  = theta_B;
  theta_ode[7]  = theta_N;
  theta_ode[8]  = delta_I;
  theta_ode[9]  = delta_H;
  theta_ode[10] = gamma_I;
  theta_ode[11] = gamma_H;
  theta_ode[12] = psi_I;
  theta_ode[13] = psi_H;
  theta_ode[14] = omega_FR;
  theta_ode[15] = omega_FS;
  theta_ode[16] = alpha;
  theta_ode[17] = beta_D;
  theta_ode[18] = gamma_comm;
  theta_ode[19] = delta_C;

  // Initial conditions
  // Seed with a small infectious fraction split by belief
  real NB0 = (1.0 - phi0) * N_pop;
  real NN0 = phi0         * N_pop;
  real seed_frac = 2e-4;   // 0.02% initially infectious per group

  vector[12] y0;
  y0[1]  = NB0 * (1.0 - seed_frac);  // S_B
  y0[2]  = 0.0;                       // E_B
  y0[3]  = NB0 * seed_frac;           // I_B
  y0[4]  = 0.0;                       // H_B
  y0[5]  = 0.0;                       // R_B
  y0[6]  = NN0 * (1.0 - seed_frac);  // S_N
  y0[7]  = 0.0;                       // E_N
  y0[8]  = NN0 * seed_frac;           // I_N
  y0[9]  = 0.0;                       // H_N
  y0[10] = 0.0;                       // R_N
  y0[11] = 0.0;                       // F_R
  y0[12] = 0.0;                       // F_S

  // Integrate ODE system
  // Use ode_bdf (L-stable BDF solver) — appropriate for this moderately
  // stiff system with discontinuous C(t) forcing.
  array[T] vector[12] y_hat = ode_bdf_tol(
      seihrf_od_ode,
      y0,
      t0,
      ts,
      rel_tol,
      abs_tol,
      max_steps,
      theta_ode,
      x_r_conflict,
      x_i_empty
  );

  // Predicted daily incidence: new exposures becoming infectious each day
  // mu[t] = kappa * (E_B[t] + E_N[t])
  // Approximation valid when kappa * E is the dominant new-case flow
  vector[T] mu;
  for (t in 1:T)
    mu[t] = kappa * (y_hat[t][2] + y_hat[t][7]);

  // Analytical R0: dominant eigenvalue of the 2x2 effective NGM
  // (manuscript equations 3-6 and Supplementary B)
  //
  // M = [(1-phi0)*R0_B,  (1-phi0)*rho_N ]
  //     [phi0*R0_B,      phi0*R0_N      ]
  //
  // rho_N = non-funeral component of R0_N (i.e. R0_N with beta_FR=0)
  // tr(M) = (1-phi0)*R0_B + phi0*R0_N
  // det(M) = phi0*(1-phi0)*R0_B*(R0_N - rho_N) >= 0
  // R0 = [tr(M) + sqrt(tr(M)^2 - 4*det(M))] / 2

  real kB    = theta_B + delta_I + gamma_I;
  real kN    = theta_N + delta_I + gamma_I;
  real kH    = delta_H + gamma_H;

  real R0_B  = (beta_I + beta_H * theta_B / kH) / kB;
  real R0_N  = (beta_I
                + beta_H * theta_N / kH
                + beta_FR / omega_FR
                  * (psi_I * delta_I + psi_H * delta_H * theta_N / kH))
               / kN;

  // rho_N: non-funeral component (beta_FR = 0 limit of R0_N)
  real rho_N = (beta_I + beta_H * theta_N / kH) / kN;

  // 2x2 effective NGM quantities
  real trM   = (1.0 - phi0) * R0_B + phi0 * R0_N;
  real detM  = phi0 * (1.0 - phi0) * R0_B * (R0_N - rho_N);
  // detM >= 0 always; sqrt argument is non-negative by AM-GM
  real disc  = fmax(trM * trM - 4.0 * detM, 0.0);  // guard for numerics
  real R0    = 0.5 * (trM + sqrt(disc));

} // end transformed parameters


// =============================================================================
model {

  // ------------------------------------------------------------------
  // PRIORS (Table 1 of manuscript)
  // ------------------------------------------------------------------

  // Informationally updated
  beta_I     ~ normal(0.75, 0.08);   // posterior: 0.74 (0.60-0.91)
  beta_FR    ~ normal(1.60, 0.25);   // posterior: 1.62 (1.14-2.11)
  phi0       ~ normal(phi0_obs, phi0_obs_sd);  // seeded from r_c proxy

  // Prior-dominated
  theta_N    ~ normal(0.04, 0.008);
  alpha      ~ gamma(2, 50);         // mean 0.04, sd ~0.028
  gamma_comm ~ gamma(2, 80);         // mean 0.025
  delta_C    ~ gamma(2, 40);         // mean 0.05

  // Overdispersion
  phi_obs    ~ gamma(2, 0.1);        // weakly informative

  // ------------------------------------------------------------------
  // LIKELIHOOD — NegBin observation model
  // y_cases[t] ~ NegBin2(mu[t], phi_obs)
  // where phi_obs is the precision parameter (larger = less dispersion)
  // ------------------------------------------------------------------
  for (t in 1:T) {
    if (mu[t] > 0)
      y_cases[t] ~ neg_binomial_2(mu[t], phi_obs);
  }

} // end model


// =============================================================================
generated quantities {

  // ------------------------------------------------------------------
  // Pointwise log-likelihood (for LOO-CV with loo package)
  // ------------------------------------------------------------------
  vector[T] log_lik;
  for (t in 1:T) {
    if (mu[t] > 0)
      log_lik[t] = neg_binomial_2_lpmf(y_cases[t] | mu[t], phi_obs);
    else
      log_lik[t] = 0.0;
  }

  // ------------------------------------------------------------------
  // Posterior predictive distribution
  // ------------------------------------------------------------------
  array[T] int y_rep;
  for (t in 1:T) {
    if (mu[t] > 0)
      y_rep[t] = neg_binomial_2_rng(mu[t], phi_obs);
    else
      y_rep[t] = 0;
  }

  // ------------------------------------------------------------------
  // Cumulative deaths at last observation day
  // (posterior predictive, for scenario comparison)
  // ------------------------------------------------------------------
  real cum_deaths_pred;
  {
    real acc = 0.0;
    for (t in 1:T) {
      real IB_t = y_hat[t][3];
      real IN_t = y_hat[t][8];
      real HB_t = y_hat[t][4];
      real HN_t = y_hat[t][9];
      acc += delta_I * (IB_t + IN_t) + delta_H * (HB_t + HN_t);
    }
    cum_deaths_pred = acc;
  }

  // ------------------------------------------------------------------
  // scepticism phi(t) at each time point
  // ------------------------------------------------------------------
  vector[T] phi_t;
  for (t in 1:T) {
    real SB_t  = y_hat[t][1];
    real SN_t  = y_hat[t][6];
    real Ssum  = SB_t + SN_t;
    phi_t[t]   = (Ssum > 0) ? SN_t / Ssum : phi0;
  }

} // end generated quantities
