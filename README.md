# esag

ESAG regression for directional data, in Rust.

`esag` ports and re-engineers Zehao Yu's ESAG R code
([Zehaoyu217/ESAG](https://github.com/Zehaoyu217/ESAG)) as a model-fitting
library in the style of scikit-learn. ESAG is the elliptically symmetric
angular Gaussian distribution on the unit sphere S^{d−1}. The library covers:

* the ESAG(μ, γ) distribution in any dimension d ≥ 3, with Yu & Huang's
  parameterization: density, sampling, covariance and eigen-structure;
* maximum-likelihood fitting, without covariates or with them
  (`(μ_i, γ_i) = B·(1, x_i)`);
* residuals, bootstrap goodness of fit, prediction regions, and bootstrap
  hypothesis tests (RoC, D, M, LR) for isotropy and covariate effects;
* a command-line tool for CSV data.

It reproduces the R code's numbers and the papers' published real-data
results, and runs 7,000 to 69,000 times faster than the R code.

## Papers

* Z. Yu & X. Huang (2024). A new parameterization for elliptically symmetric
  angular Gaussian distributions of arbitrary dimension. *Electronic Journal
  of Statistics* 18(1):301–334. doi:10.1214/23-EJS2210 (arXiv:2212.05634).
* Z. Yu & X. Huang (2025). Regression analysis of elliptically symmetric
  directional data. *Computational Statistics & Data Analysis* 208:108167.
  doi:10.1016/j.csda.2025.108167 (arXiv:2408.03268).
* Z. Yu (2024). *Statistical Inference Based on Elliptically Symmetric
  Distributions for Directional Data*. PhD dissertation, University of South
  Carolina.
* P. J. Paine, S. P. Preston, M. Tsagris & A. T. A. Wood (2018). An
  elliptically symmetric angular Gaussian distribution. *Statistics and
  Computing* 28:689–697.

## Quick start

```rust
use esag::regression::EsagRegressor;
use esag::inference::{bootstrap_gof, hypothesis_test, prediction_region, GofMethod, Restriction};

// x: n × p covariates (no intercept column), y: n × d unit vectors
let model = EsagRegressor::new()
    .with_feature_names(vec!["dose".into()])
    .fit(&x, &y)?;                                  // L-BFGS with exact gradients

println!("{}", model.summary());                    // coefficients, SEs, z, p, AIC/BIC
let directions = model.predict(&x_new)?;            // n × d predicted mean directions
let dists = model.predict_distribution(&x_new)?;    // Vec<Esag>: full conditional laws
let mean_loglik = model.score(&x_test, &y_test)?;   // held-out mean log-likelihood
let resid = model.residuals(&x, &y)?;               // Q_i and T1_i (≈ χ²_{d−1})

let gof = bootstrap_gof(&model, &x, &y, GofMethod::TwoSampleReference, 200, 1)?;
let iso = hypothesis_test(&model, &x, &y, Restriction::Isotropy, 200, 1000, 2)?;
let pr  = prediction_region(&model, &x, &y, &[1.0], 5000, 100, &[0.95, 0.9], 3)?;

std::fs::write("model.json", model.to_json()?)?;     // save; EsagRegressorFit::from_json loads
```

To fit a single distribution to i.i.d. data, use `EsagEstimator::new().fit(&y)`.
`Esag::new(&mu, &gamma)` gives the distribution itself, with `log_pdf`,
`sample`, `covariance`, `mean_direction` and `angles`.

### Run the regression demo

```sh
cargo run --release --example regression_demo
```

Part A simulates a d = 3 regression with known coefficients. It then fits,
predicts on held-out data, checks residuals and goodness of fit, builds
prediction regions, runs tests (one false null, one true), computes bootstrap
SEs and saves the model. Part B reproduces the real-data regression of Yu &
Huang (2025, Sec. 7.1). The full output is in
[`docs/demo_output.txt`](docs/demo_output.txt), and the whole run takes about
6 s.

### Command line

```sh
cargo build --release
B=target/release/esag
$B simulate --n 800 --out sim.csv
$B fit sim.csv --y y1,y2,y3 --x x1 --model model.json
$B predict model.json sim.csv --x x1 --out pred.csv
$B gof model.json sim.csv --y y1,y2,y3 --x x1 --boot 200
$B test model.json sim.csv --y y1,y2,y3 --x x1 --null isotropy --boot 200   # also: mu | gamma | feature
$B region model.json sim.csv --y y1,y2,y3 --x x1 --at 0.5
```

## API overview

| item | role |
|---|---|
| `regression::EsagRegressor` | configuration and builder: `fit_intercept`, `max_iter`, `tol`, `init` (`Auto`, `DataDriven`, `RStyle`, `Custom`), `with_restarts`, `with_mask` (restricted fits), `with_hessian`, `with_feature_names`. `fit(x, y)` and `fit_warm(x, y, b0)`. |
| `regression::EsagRegressorFit` | fitted model: `coef`, `intercept()`, `slopes()`, `predict`, `predict_distribution`, `linear_predictor`, `score`, `log_likelihood_rows`, `residuals`, `std_errors`, `cov_params`, `summary`, `aic`, `bic`, `sample_responses`, `to_json` and `from_json`. |
| `estimator::{EsagEstimator, EsagFit}` | i.i.d. fit (no covariates), same options. |
| `estimator::{Fit, Predict, Score}` | scikit-learn-style traits, implemented by the estimators. |
| `distribution::Esag` | the distribution: `log_pdf`, `pdf`, `sample`, `covariance`, `cov_pow`, `angles`, `mean_direction`, `residuals`, `region_statistic`. |
| `preprocessing::MinMaxScaler` | rescales covariates to [1, 2], as the papers recommend. |
| `inference::bootstrap_gof` | goodness-of-fit test: `TwoSampleReference` (paper's Algorithm 1) or `OneSampleChiSquare` (the R code's variant). |
| `inference::hypothesis_test` | RoC, D, M, LR with parametric-bootstrap p-values. Nulls: `Isotropy`, `MuOfFeature(j)`, `GammaOfFeature(j)`, `Feature(j)`, `Custom(mask)`. |
| `inference::{prediction_region, prediction_regions, region_coverage}` | CSDA Algorithm 4, with bootstrap refits shared across query points. |
| `inference::bootstrap_coef` | pairs-bootstrap SEs and percentile intervals. |
| `regression::canonicalize_coef` | picks the canonical sign of the first γ group (see identifiability below). |
| `param::gamma_from_covariance` | the inverse of the parameterization: the canonical γ for any μ and V. |

Every bootstrap procedure runs its replicates in parallel with rayon. Each
replicate and each Monte Carlo stream has its own xoshiro256++ generator,
derived from your seed, a purpose tag and an index. Results are therefore
reproducible and don't depend on the thread count; a test checks this.

## Validation

All `cargo test --release` suites pass: 124 tests, plus 7 slow studies and
benchmarks marked `#[ignore]`.

**Against the original R code.** Fixtures in `tests/fixtures` were produced
by running the R code, and their README records the R version, seeds and
schema.

| quantity | agreement |
|---|---|
| `parameter` (λ, θ, φ), `rotation`, `ONB`, `V`, `V⁻¹`, `V^½` for d = 3–6, 144 cases | ≤ 1e-10 relative |
| `log_likelihood`, 48 cases for d = 3–5 | 2e-14 relative |
| joint log-likelihood at R's MLE and at the true coefficients | ≤ 1e-11 relative |
| `MLE()` regression fits, n = 200 and 1000 | Rust's maximum is equal or higher (+5e-10 and +1.7e-6); fitted μ_i and V_i agree to ≤ 1.5e-4; SEs agree to within 2% |
| `MLE()` i.i.d. fits, d = 3 and 4 | same optimum; μ and V agree to ≤ 1.7e-5 |
| `residual_T` and `ks.test(T1, "pchisq", d−1)` | ≤ 1e-10 and ≤ 1e-12 |
| `Confidence_Region` and `Prediction_Region` quantiles | within Monte Carlo error |

**Against the published real-data results.** The Llobregat hydrochemistry
data (CSDA 2025 Sec. 7.1; `data/README.md` gives provenance) are a d = 4
regression with X = Anoia tributaries or lower-Llobregat tributaries.

| quantity | paper | esag |
|---|---|---|
| μ intercept / slope | (1.99, 5.74, 7.95, 4.59) / (1.28, 2.83, 1.06, 1.20) | identical to 2 d.p. |
| eigenvalues of V, Anoia / lower Llobregat | (0.37, 0.62, 4.44) / (0.19, 0.54, 9.61) | identical |
| ‖γ‖, Anoia / lower Llobregat | 6.24 / 17.03 | 6.24 / 17.03 |
| isotropy test: RoC, D, LR | 1.518, 1.518, 275.4 | 1.518, 1.518, 275.41 |
| μ-dependence test: RoC, D, LR | 1.059, 1.062, 31.6 | 1.059, 1.062, 31.58 |
| γ-dependence test: RoC, D, M, LR | 1.090, 1.090, 9.4e-5, 57.6 | 1.090, 1.090, 9.7e-5, 57.56 |
| prediction region q at 70/80/90%, Anoia | 0.0286, 0.0364, 0.0495 | 0.0287, 0.0365, 0.0495 |

The fitted V matrices match the paper's printed matrices to its 2 decimals,
and `tests/paper_hydrochem.rs` checks all of the above.

**Calibration under the null.** These are the ignored tests, run with
`cargo test --release -- --ignored`, at the 5% level.

| test | rejection rate |
|---|---|
| goodness of fit (200 replications), two-sample / one-sample variant | 0.070 / 0.055 |
| γ free of x, a true null (1,000 replications), RoC / D / M / LR | 0.041 / 0.042 / 0.046 / 0.044 |
| the same LR with a χ²₂ reference instead of the bootstrap | 0.123 |

At 1,000 replications the standard error of a rejection rate is 0.007.

The last row shows why the papers use the bootstrap.

## Performance

Measured on the same machine (Apple M5 Pro), with the same data and settings
as the R timings stored in the fixtures. Run
`cargo run --release --example bench_vs_r`.

| task | R 4.6.1 | esag | speed-up |
|---|---|---|---|
| MLE + Hessian, regression d=3, n=200, p=2 | 7.02 s | 0.70 ms | 10,000× |
| MLE + Hessian, regression d=3, n=1000, p=2 | 36.5 s | 1.0 ms | 36,000× |
| MLE + Hessian, i.i.d. d=3, n=300 | 3.24 s | 0.45 ms | 7,200× |
| MLE + Hessian, i.i.d. d=4, n=300 | 12.4 s | 0.97 ms | 12,800× |
| `Bootstrap_GoF`, 20 refits, n=200 | 79 s | 1.2 ms | 69,000× |
| `Prediction_Region`, m=2000, 5 refits | 21 s | 0.47 ms | 45,000× |

`cargo run --release --example perf_suite` times a wider set of workloads.
The second column is the first Rust release; minimum of two runs each.

| workload | v0.1 | now | speed-up |
|---|---|---|---|
| log-density + gradient, per observation, d = 3 / 6 | 178 ns / 1.4 µs | 37 / 124 ns | 4.8× / 11.7× |
| fit with SEs, d = 3, n = 1000 | 8.7 ms | 1.4 ms | 6× |
| fit with SEs, d = 4, n = 2000 | 93 ms | 7.0 ms | 13× |
| fit with SEs, d = 5, n = 2000 | 2.77 s | 9.0 ms | 307× |
| the paper's hydrochemistry fit (d = 4) | 18.2 ms | 1.9 ms | 9× |
| bootstrap GoF, 200 replicates, n = 1000 | 139 ms | 45 ms | 3× |
| prediction region, 100 refits, n = 1000 | 92 ms | 26 ms | 3.5× |
| hypothesis test, 200 replicates, d = 3 / d = 4 | 811 / 819 ms | 382 / 221 ms | 2.1× / 3.7× |

Where the speed comes from:

* **A trig-free, allocation-free kernel.** `yᵀV⁻¹y` is computed from O(d)
  projections onto the closed-form basis, followed by plane rotations
  applied to a (d−1)-vector. Each angle enters only through its cosine and
  sine, which are ratios of norms of γ entries, so no trigonometric function
  is evaluated and nothing is allocated.
* **A hand-written reverse-mode gradient.** It costs 1.3 to 2.3 times the
  value in every dimension. R's `optim` instead spends 2·dim(B) likelihood
  calls per gradient on finite differences.
* **Fewer iterations.** The L-BFGS memory matches the parameter count. A
  moment-based start uses the tangent-plane scatter, inverted to γ. The
  default starts run in parallel and share an iteration budget, so a start
  stuck near a singular point of the parameterization cannot hold up the
  fit.
* **Cheap standard errors.** The observed information is exact at d = 3 and
  uses differences of the exact gradient for d ≥ 4. One Newton step reuses it
  to polish the optimum to ‖∇ℓ‖ ≈ 1e-10.
* **Faster Monte Carlo.** xoshiro256++ streams, a sampler with precomputed
  scaled eigenvectors, an O(d) region statistic, and linear-time quantile
  selection.
* **Parallelism.** rayon runs over observations within a fit and over
  bootstrap replicates, with refits inside bootstraps kept sequential.

## Improvements over the R code

* **Stable in the tails.** `log M_p(α)` uses a backward continued fraction for
  α < −2, accurate to about 1e-14. The R recursion returns `NaN` or `−Inf`
  there.
* **Any dimension.** The R isotropy fit, the γ-restricted start and
  `Bootstrap_gof` hard-code d = 4.
* **Correct bootstrap sampling.** R's `hypothesis_testing_*` draws
  bootstrap responses from `Bx[i,]` (the bootstrap counter) instead of
  `Bx[j,]` (the observation). The Rust version draws each observation from its
  own null parameters.
* **More robust fitting.** Fits try two starts and check convergence with a
  gradient test. Bootstrap fits retry from a second start. Restricted fits
  start from the full fit, and full refits use the best of two starts, which
  fixed an over-rejection of the LR test (8.7% down to 5.0%).
* **The paper's goodness-of-fit statistic.** Algorithm 1 (two-sample KS
  against a model-based reference) is the default; the R code's one-sample χ²
  variant is still available.
* **Results the R code doesn't return.** The observed LR with bootstrap and
  χ² p-values, bootstrap SEs, AIC and BIC, JSON persistence, prediction
  regions at any covariate value, and held-out coverage.

The full list of R-code issues found while porting is in
`tests/fixtures/README.md`.

## Identifiability, and what to report

v_j and −v_j give the same V, so several γ describe the same distribution.
Yu & Huang call them *equivalent*. Negating the first γ group is an exact
linear symmetry for every d, and `esag` canonicalises it. Intercept-only fits
go further: they report the canonical γ of their fitted V, via
`param::gamma_from_covariance`, so equivalent fits print identical γ in any
dimension. In a regression, the other equivalences for d ≥ 4 are not linear
in the covariates, so R and Rust can still return different raw γ for an
identical fitted distribution. Report and compare `predict_distribution`
quantities (μ_i, V_i, eigenvalues and ‖γ‖), and use `bootstrap_coef` rather
than Hessian SEs for γ, as the papers recommend. `docs/MATH.md` has the
details.

## Layout

```
src/scalar.rs, dual.rs          Scalar trait; forward-mode AD
src/param.rs, onb.rs,           γ → (λ, θ, φ); basis; Givens rotation; eigen-frame of V
  rotation.rs, frame.rs
src/loglik.rs                   log-density, stable log M_p, exact gradient
src/optim/                      L-BFGS (strong + approximate Wolfe), FD Hessian
src/stats/                      normal, χ², KS (exact and asymptotic, R-compatible), quantiles
src/distribution.rs             Esag
src/regression.rs, estimator.rs EsagRegressor / EsagEstimator (scikit-learn style)
src/preprocessing.rs            MinMaxScaler
src/inference/                  goodness of fit, prediction regions, hypothesis tests, bootstrap
src/bin/esag.rs                 CLI
examples/                       regression_demo, bench_vs_r
tests/                          R parity, paper reproduction, API, inference
reference/r/                    the original R sources
docs/MATH.md                    the mathematics, with references
```

## License

MIT, the same license as the original R code. `data/hydrochem_At_LLt.csv`
comes from the R package `compositions` (GPL ≥ 2); see `data/README.md`.
