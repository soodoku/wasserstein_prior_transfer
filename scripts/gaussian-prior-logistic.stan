data {
  int<lower=0> N;
  int<lower=0> D;
  matrix[N, D] x;
  array[N] int<lower=0, upper=1> y;
  vector[D] mu_prior;
  cov_matrix[D] Sigma_prior;
}
parameters {
  vector[D] beta;
}
model {
  y ~ bernoulli_logit(x * beta);
  beta ~ multi_normal(mu_prior, Sigma_prior);
}