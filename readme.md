# Wasserstein Prior Transfer for Bayesian Inference

Prior Art: https://github.com/bob-carpenter/case-studies/tree/master/empirical-prior and https://statmodeling.stat.columbia.edu/2025/05/13/chaining-bayes-priors-from-posteriors/

## Overview

This code demonstrates three approaches for transferring prior information between related datasets using Wasserstein optimal transport methods. The scenario involves fitting logistic regression models on data from Ohio and New York, where the underlying parameters are the same but the covariate distributions differ.

## Methods Compared

1. **KDE Method (Zhong et al.)**: Uses kernel density estimation to create an empirical prior from Ohio posterior samples

2. **Wasserstein Moment Matching**: Fits a Gaussian prior by matching the first and second moments of the Ohio posterior (closed-form Wasserstein-optimal)

3. **Wasserstein Barycenters**: Approximates the Ohio posterior with a Gaussian mixture model, then uses this as a mixture prior for NY data

## Key Features

- Simulates logistic regression data with known true parameters
- Compares transfer learning approaches against flat priors
- Evaluates methods using MSE, posterior variance, and credible interval coverage
- Implements adaptive component selection for Gaussian mixture models (BIC and Bayesian approaches)

## Dependencies

- `cmdstanpy` - Bayesian sampling via Stan
- `scipy` - Optimization and statistical distributions  
- `sklearn` - Gaussian mixture models
- `ot` - Optimal transport library
- `numpy` - Numerical computing

## Usage

1. Ensure Stan model files (`flat-logistic.stan`, `empirical-logistic.stan`, `gaussian-prior-logistic.stan`, `mixture-prior-logistic.stan`) are in the working directory
2. Run the script to compare all three transfer methods
3. Results include posterior summaries and performance metrics for each approach

## Output

The script prints comparison metrics including:
- Mean squared error vs. true parameters
- Average posterior variance  
- 95% credible interval coverage rates
- Posterior mean estimates

This framework can be extended to other Bayesian models and transfer learning scenarios where you have informative posteriors from related datasets.