// =============================================================================
// seihrf_od_profile.stan
// Profile-likelihood wrapper for the SEIHRF-OD model
//
// Identical to seihrf_od.stan EXCEPT:
//   • Three extra data fields: profile_param_idx, profile_value, profile_sigma
//   • In the model block the profiled parameter's prior is replaced by a
//     tight normal(profile_value, profile_sigma) that effectively fixes it
//     to the grid value while all other parameters remain free.
//
// profile_param_idx:
//   0 = no parameter fixed (standard MAP estimation)
//   1 = beta_I
//   2 = beta_FR
//   3 = phi0
//   4 = alpha
//   5 = delta_C
// =============================================================================

functions {

  real conflict_C(real t, array[] real x_r) {
    real c = 0.0;
    for (k in 1:5) {
      int idx = 2 * k - 1;
      if (t >= x_r[idx])
        c = x_r[idx + 1];
    }
    return c;
  }

  vector seihrf_od_ode(real t,
                       vector y,
                       array[] real theta,
                       array[] real x_r,
                       array[] int x_i) {

    real SB = y[1];  real EB = y[2];  real IB = y[3];
    real HB = y[4];  real RB = y[5];
    real SN = y[6];  real EN = y[7];  real IN = y[8];
    real HN = y[9];  real RN = y[10];
    real FR = y[11]; real FS = y[12];

    real N = SB + EB + IB + HB + RB + SN + EN + IN + HN + RN;
    if (N <= 0) return rep_vector(0.0, 12);

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

    real Ct  = conflict_C(t, x_r);
    real S_sum = SB + SN;
    real phi   = (S_sum > 0) ? SN / S_sum : 0.0;
    real Dvis  = (dI * (IB + IN) + dH * (HB + HN)) / N;
    real mu_BN = alph * phi + del_C * Ct;
    real mu_NB = gam_comm + bet_D * Dvis;

    real lam_B = (beta_I * (IB + IN) + beta_H * (HB + HN) + beta_FS * FS) / N;
    real lam_N = (beta_I * (IB + IN) + beta_H * (HB + HN) + beta_FR * FR) / N;

    real dSB = -lam_B * SB - mu_BN * SB + mu_NB * SN;
    real dEB =  lam_B * SB - kap * EB;
    real dIB =  kap * EB  - (th_B + dI + gI) * IB;
    real dHB =  th_B * IB - (dH  + gH) * HB;
    real dRB =  gI * IB + gH * HB;

    real dSN = -lam_N * SN + mu_BN * SB - mu_NB * SN;
    real dEN =  lam_N * SN - kap * EN;
    real dIN =  kap * EN  - (th_N + dI + gI) * IN;
    real dHN =  th_N * IN - (dH  + gH) * HN;
    real dRN =  gI * IN + gH * HN;

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


data {
  int<lower=1>  T;
  array[T] int<lower=0> y_cases;
  real<lower=1> N_pop;
  real<lower=0, upper=1> phi0_obs;
  real<lower=0>          phi0_obs_sd;
  array[10] real x_r_conflict;
  real<lower=0> rel_tol;
  real<lower=0> abs_tol;
  int<lower=1>  max_steps;

  // Profile likelihood control
  int<lower=0, upper=5> profile_param_idx;  // 0=none, 1-5=param to fix
  real                  profile_value;       // grid value to fix
  real<lower=0>         profile_sigma;       // tightness (e.g. 1e-4)
}


transformed data {
  array[0] int x_i_empty;
  real t0 = 0.0;
  array[T] real ts;
  for (t in 1:T) ts[t] = t;
}


parameters {
  real<lower=0.40, upper=1.20> beta_I;
  real<lower=0.50, upper=3.50> beta_FR;
  real<lower=0.15, upper=0.70> phi0;
  real<lower=0.01, upper=0.20> theta_N;
  real<lower=0.01, upper=0.15> alpha;
  real<lower=0.005, upper=0.10> gamma_comm;
  real<lower=0.01, upper=1.50> delta_C;
  real<lower=0> phi_obs;
}


transformed parameters {
  real beta_H    = 0.06;
  real beta_FS   = 0.002;
  real kappa     = 1.0 / 9.0;
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

  real NB0 = (1.0 - phi0) * N_pop;
  real NN0 = phi0         * N_pop;
  real seed_frac = 2e-4;

  vector[12] y0;
  y0[1]  = NB0 * (1.0 - seed_frac);
  y0[2]  = 0.0;
  y0[3]  = NB0 * seed_frac;
  y0[4]  = 0.0;
  y0[5]  = 0.0;
  y0[6]  = NN0 * (1.0 - seed_frac);
  y0[7]  = 0.0;
  y0[8]  = NN0 * seed_frac;
  y0[9]  = 0.0;
  y0[10] = 0.0;
  y0[11] = 0.0;
  y0[12] = 0.0;

  array[T] vector[12] y_hat = ode_bdf_tol(
      seihrf_od_ode, y0, t0, ts,
      rel_tol, abs_tol, max_steps,
      theta_ode, x_r_conflict, x_i_empty
  );

  vector[T] mu;
  for (t in 1:T)
    mu[t] = fmax(kappa * (y_hat[t][2] + y_hat[t][7]), 1e-9);

  real kB   = theta_B + delta_I + gamma_I;
  real kN   = theta_N + delta_I + gamma_I;
  real kH   = delta_H + gamma_H;
  real R0_B = (beta_I + beta_H * theta_B / kH) / kB;
  real R0_N = (beta_I + beta_H * theta_N / kH
               + beta_FR / omega_FR
                 * (psi_I * delta_I + psi_H * delta_H * theta_N / kH)) / kN;
  real rho_N = (beta_I + beta_H * theta_N / kH) / kN;
  real trM  = (1.0 - phi0) * R0_B + phi0 * R0_N;
  real detM = phi0 * (1.0 - phi0) * R0_B * (R0_N - rho_N);
  real disc = fmax(trM * trM - 4.0 * detM, 0.0);
  real R0   = 0.5 * (trM + sqrt(disc));
}


model {
  // ── Priors — standard unless parameter is being profiled ──────────────────
  // For the profiled parameter, replace the original prior with a tight
  // normal centred on profile_value. This pins it to the grid value while
  // all other parameters remain free (= true profile likelihood).

  if (profile_param_idx == 1)
    beta_I     ~ normal(profile_value, profile_sigma);
  else
    beta_I     ~ normal(0.75, 0.08);

  if (profile_param_idx == 2)
    beta_FR    ~ normal(profile_value, profile_sigma);
  else
    beta_FR    ~ normal(1.60, 0.25);

  if (profile_param_idx == 3)
    phi0       ~ normal(profile_value, profile_sigma);
  else
    phi0       ~ normal(phi0_obs, phi0_obs_sd);

  // theta_N, gamma_comm: never profiled — standard priors always
  theta_N    ~ normal(0.04, 0.008);
  gamma_comm ~ gamma(2, 80);

  if (profile_param_idx == 4)
    alpha      ~ normal(profile_value, profile_sigma);
  else
    alpha      ~ gamma(2, 50);

  if (profile_param_idx == 5)
    delta_C    ~ normal(profile_value, profile_sigma);
  else
    delta_C    ~ gamma(2, 40);

  phi_obs    ~ gamma(2, 0.1);

  // ── Likelihood ─────────────────────────────────────────────────────────────
  for (t in 1:T)
    y_cases[t] ~ neg_binomial_2(mu[t], phi_obs);
}


generated quantities {
  vector[T] log_lik;
  for (t in 1:T)
    log_lik[t] = neg_binomial_2_lpmf(y_cases[t] | mu[t], phi_obs);
}
