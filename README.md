# esag

ESAG regression for directional data, in Rust.

`esag` ports and re-engineers Zehao Yu's ESAG R code
([Zehaoyu217/ESAG](https://github.com/Zehaoyu217/ESAG)) as a model-fitting
library with a statsmodels-style API and output, in Rust and Python. ESAG
is the elliptically symmetric
angular Gaussian distribution on the unit sphere S^{d−1}. The library covers:

* the ESAG(μ, γ) distribution in any dimension d ≥ 3, with Yu & Huang's
  parameterization: density, sampling, covariance and eigen-structure;
* maximum-likelihood fitting, without covariates or with them
  (`(μ_i, γ_i) = B·(1, x_i)`);
* residuals, bootstrap goodness of fit, prediction regions, and bootstrap
  hypothesis tests (RoC, D, M, LR, and a score statistic LM) for covariate
  effects on the mean and on an anisotropic shape, and for isotropy, with
  the second moments in `M` computed exactly instead of by Monte Carlo, and
  a score test that refits only the restricted model;
* an opt-in smooth shape parameterization (`Shape::LogTangent`) for the same
  distributions, which removes the discontinuities of `γ ↦ V` that make large
  regressions (d = 7–8, 5–6 covariates) slow and unreliable;
* the LASSO family (Project 3): LASSO, adaptive LASSO, group, adaptive group
  and sparse group LASSO on any set of coefficients, isotropy and
  eigenvalue-tie penalties, and the paper's six methods as coded in its R
  scripts, with λ paths, BIC/AIC/CV selection and relaxed refits;
* statsmodels-style results (`params`, `bse`, `pvalues`, `conf_int`,
  `summary()`), in Rust and in the Python package (`python/`);
* a command-line tool for CSV data.

It reproduces the R code's numbers and the papers' published real-data
results, and runs 4,200 to 103,000 times faster than the R code.

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
* Z. Yu & X. Huang. LASSO-based inference for elliptically symmetric angular
  Gaussian distributions. Draft for *Statistica Sinica* ("Project 3").
* P. J. Paine, S. P. Preston, M. Tsagris & A. T. A. Wood (2018). An
  elliptically symmetric angular Gaussian distribution. *Statistics and
  Computing* 28:689–697.

## Quick start

### Python

```sh
cd python && pip install maturin && maturin develop --release   # or: maturin build --release
```

```python
import esag

model = esag.ESAG(Y, X)                 # endog: n x d unit vectors, exog: n x p (const added)
res = model.fit()
print(res.summary())                    # statsmodels-style table, one block per equation
res.params, res.bse, res.pvalues, res.conf_int()

res.test("shape", "temp", n_boot=500)   # RoC, D, M, LR, LM with bootstrap p-values
res.test("constant_shape", n_boot=500)  # nulls: mu | shape | feature | constant_shape | isotropy
res.score_test("shape", "temp")         # LM and M from restricted refits only: faster

reg = model.fit_regularized("adaptive_group_lasso", groups="covariate_part")
print(reg.summary())                    # relaxed refit of the selected model; dropped terms marked
reg.path                                # alpha, lambda, df, llf, BIC, active groups per step
```

Penalties: `lasso`, `adaptive_lasso`, `group_lasso`, `adaptive_group_lasso`,
`sparse_group_lasso` (`l1_ratio`), `isotropy`, `eigen_gap`, or
`paper="RoC-LASSO"` (any of the paper's six). `target` picks the
coefficients (`slopes`, `mean`, `shape_slopes`, `shape`), `groups` the
grouping (`covariate`, `covariate_part`, `parameter`, `all`, `gamma`, or
explicit lists of `("mu[1]", "temp")` pairs), `adaptive=True|r|False` the
weights, and `select` the rule (`bic`, `aic`, `cv`). `alpha=` fixes the
penalty weight on the per-observation scale, as in statsmodels. Inputs may
be numpy or pandas, and pandas column names flow into the output.
`python/README.md` has more, and `python examples/demo.py` runs a full
session.

### Rust

```rust
use esag::regression::EsagRegressor;
use esag::inference::{bootstrap_gof, hypothesis_test, prediction_region, score_test, GofMethod, Restriction};

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
let shp = hypothesis_test(&model, &x, &y, Restriction::GammaOfFeature(0), 500, 0, 2)?;
let fast = score_test(&model, &x, &y, Restriction::ConstantShape, 500, 0, 2)?;
let pr  = prediction_region(&model, &x, &y, &[1.0], 5000, 100, &[0.95, 0.9], 3)?;

std::fs::write("model.json", model.to_json()?)?;     // save; EsagRegressorFit::from_json loads
```

For regressions at d ≥ 4 with covariates the smooth shape is
the reliable choice. Each observation's law is still an ESAG with the same
number of parameters, but with covariates "shape linear in x" is a
different regression model in each parameterization, so compare the two by
AIC/BIC (`docs/MATH.md` §7):

```rust
use esag::regression::Shape;
let model = EsagRegressor::new().with_shape(Shape::LogTangent).fit(&x, &y)?;
```

Penalized fits (Project 3) select λ along a path:

```rust
use esag::penalized::{Grouping, PenalizedEsag, Target};
// which covariates move the mean direction and which move the shape?
let path = PenalizedEsag::adaptive_group_lasso()
    .grouping(Grouping::CovariatePart)
    .fit(&x, &y)?;
println!("{}", path.summary());                     // statsmodels-style, relaxed refit with SEs
println!("{}", path.path_summary());                // λ, log-lik, df, BIC, active groups
// other members of the family
let l1 = PenalizedEsag::lasso().target(Target::MeanSlopes).cv(10, 1).fit(&x, &y)?;
let sg = PenalizedEsag::sparse_group_lasso(0.5).adaptive(1.0).fit(&x, &y)?;
// i.i.d. data: partial isotropy (tied eigenvalues), or the paper's methods
let ties = PenalizedEsag::eigen_gap().fit(&x0, &y)?;
let paper = PenalizedEsag::paper("RoC-LASSO")?.fit(&x0, &y)?;
```

The statsmodels-style front end wraps the same pieces:

```rust
use esag::model::EsagModel;
let model = EsagModel::new(&y, &x).exog_names(&["temp", "ph"]);
let res = model.fit()?;                             // res.params(), bse(), pvalues(), conf_int(0.05)
let t = res.test(Restriction::GammaOfFeature(0), 500, 1)?;
println!("{}\n{}", res.summary(), t.summary());
```

**Estimating V(x).** The covariance `V(x)` is a nonlinear function of all
the coefficients, so it gets its own estimator with bootstrap uncertainty:

```rust
use esag::inference::covariance_intervals;
let v = model.predict_covariance(&x_new)?;            // V̂(x) for each row
let est = covariance_intervals(&model, &x, &y, &x_new, 500, 0.95, 1)?;
println!("{}", est[0].summary());   // shape eigenvalues ± SE, basic intervals, axes, V̂ entries ± SE
```

In Python: `res.covariance(X_new)` and `res.covariance_ci(x0, n_boot=500)`.
`docs/MATH.md` §5b has the simulation check (consistency and interval
coverage).

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
0.6 s.

`cargo run --release --example big_fit` fits d = 7–8 regressions with 5–6
covariates in both shape parameterizations,
`cargo run --release --example hard_scenario -- fit|tests|lasso` runs the
fits, tests and penalized selection on a hard scenario (μ and V(x) strongly
x-dependent, correlated covariates), and
`cargo run --release --example project3_paper` runs the six paper methods
and the eigen-gap penalty on an experiment-1 replicate.

### Command line

```sh
cargo build --release
B=target/release/esag
$B simulate --n 800 --out sim.csv
$B fit sim.csv --y y1,y2,y3 --x x1 --model model.json
$B predict model.json sim.csv --x x1 --out pred.csv
$B gof model.json sim.csv --y y1,y2,y3 --x x1 --boot 200
$B test model.json sim.csv --y y1,y2,y3 --x x1 --null gamma --feature 0 --boot 200   # also: mu | feature | constant-shape | isotropy; --score
$B region model.json sim.csv --y y1,y2,y3 --x x1 --at 0.5
$B fit sim.csv --y y1,y2,y3 --x x1 --shape log-tangent
$B penalize sim.csv --y y1,y2,y3 --x x1 --penalty adaptive-group-lasso --groups covariate-part
$B penalize sim.csv --y y1,y2,y3 --x x1 --penalty sparse-group-lasso --l1-ratio 0.3 --select cv
$B penalize sim.csv --y y1,y2,y3 --paper RoC-LASSO                          # the paper's methods (i.i.d.)
```

## API overview

| item | role |
|---|---|
| `regression::EsagRegressor` | configuration and builder: `fit_intercept`, `max_iter`, `tol`, `init` (`Auto`, `DataDriven`, `RStyle`, `Custom`), `with_restarts`, `with_mask` (restricted fits), `with_hessian`, `with_feature_names`, `with_shape` (`Shape::Gamma` or `Shape::LogTangent`). `fit(x, y)` and `fit_warm(x, y, b0)`. |
| `model::{EsagModel, EsagResults}` | statsmodels-style front end: `EsagModel::new(endog, exog)`, `fit()`, `fit_regularized(penalty)`; results deref to the fit and add `test`, `gof`, `prediction_region`. |
| `regression::EsagRegressorFit` | fitted model: `params`, `bse`, `zvalues`, `pvalues`, `conf_int`, `llf`, `nobs`, `df_model`, `df_resid`, `coef`, `intercept()`, `slopes()`, `predict`, `predict_distribution`, `linear_predictor`, `score`, `log_likelihood_rows`, `residuals`, `std_errors`, `cov_params`, `summary`, `aic`, `bic`, `sample_responses`, `to_json` and `from_json`. |
| `estimator::{EsagEstimator, EsagFit}` | i.i.d. fit (no covariates), same options. |
| `estimator::{Fit, Predict, Score}` | scikit-learn-style traits, implemented by the estimators. |
| `distribution::Esag` | the distribution: `log_pdf`, `pdf`, `sample`, `covariance`, `cov_pow`, `angles`, `mean_direction`, `residuals`, `region_statistic`. |
| `preprocessing::MinMaxScaler` | rescales covariates to [1, 2], as the papers recommend. |
| `inference::bootstrap_gof` | goodness-of-fit test: `TwoSampleReference` (paper's Algorithm 1) or `OneSampleChiSquare` (the R code's variant). |
| `inference::hypothesis_test` | RoC, D, M, LR and the score statistic LM with parametric-bootstrap p-values. Nulls: `MuOfFeature(j)`, `GammaOfFeature(j)`, `Feature(j)`, `ConstantShape`, `Isotropy`, `Custom(mask)`. |
| `inference::score_test` | the same nulls from restricted refits only (LM and M): 1.4–1.9× faster for log-tangent, 2–4× (up to 29×) for γ. |
| `inference::{prediction_region, prediction_regions, region_coverage}` | CSDA Algorithm 4, with bootstrap refits shared across query points. |
| `inference::bootstrap_coef` | pairs-bootstrap SEs and percentile intervals. |
| `regression::canonicalize_coef` | picks the canonical sign of the first γ group (see identifiability below). |
| `param::gamma_from_covariance` | the inverse of the parameterization: the canonical γ for any μ and V. |
| `logtangent` | the smooth shape coordinates: `LtFrame`, `covariance`, `s_from_covariance`, `loglik_lt`, `loglik_lt_grad`, `loglik_lt_hess` (exact). |
| `penalized::PenalizedEsag` | the LASSO family: `Penalty::{Lasso, GroupLasso, SparseGroupLasso, EigenGap}` × `Target::{Slopes, MeanSlopes, ShapeSlopes, Shape}` × `Grouping::{Covariate, CovariatePart, Parameter, All, GammaGroups, Custom}` × `Weighting::{Plain, Adaptive(r), RoC}`; presets `lasso()`, `adaptive_lasso()`, `group_lasso()`, `adaptive_group_lasso()`, `sparse_group_lasso(α)`, `isotropy()`, `eigen_gap()`, `paper(name)`; `LambdaGrid`, `Selection::{Bic, Aic, CrossValidation}`. `fit` returns a `PenalizedPath` (per-λ fits, criterion, selected point, relaxed refit, `summary()`, `path_summary()`). |
| `distribution::Esag::second_moment` | exact `E[YYᵀ]` (one 1-D integral, ~1 µs). |
| `optim::prox` | proximal Newton for smooth + group-norm / eigen-gap penalties. |

Every bootstrap procedure runs its replicates in parallel with rayon. Each
replicate and each Monte Carlo stream has its own xoshiro256++ generator,
derived from your seed, a purpose tag and an index. Results are therefore
reproducible and don't depend on the thread count; a test checks this.

## Validation

All `cargo test --release` suites pass: 179 tests, plus 37 slow studies,
diagnostics and benchmarks marked `#[ignore]`. The Python package's 23
pytest tests (`python/tests`) pass on Python 3.9 and 3.13.

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
and `tests/paper_hydrochem.rs` checks all of the above. With exact moments
the `M` statistics are 2.17e−4 (isotropy; dissertation 2.2e−4) and 9.47e−5
(γ-dependence; 9.4e−5).

**Against the Project 3 R code.** `tests/fixtures/p3_*.json` were produced
by evaluating the user's method scripts verbatim (`docs/project3/SPEC.md`).

| check | result |
|---|---|
| coded objective of the six methods (weights, RoC floor) at R's 18 optima | equal to optim's value to 1e−9 |
| optimizing from R's optima | never lower than R |
| microbiome application (RoC-LASSO, BIC) | λ = 1 selected; μ̂ within 0.05 of the paper's; zeros at γ₁, γ₅ |
| eigen-gap penalty on data with tied eigenvalues | true tie pattern recovered |
| driver selection, d = 5 and d = 7–8 regressions | exactly the true mean/shape drivers kept |

From R's start (truth + 0.1) the optimum reached on the γ objective depends
on the optimizer's path: R itself differs by up to 28 log-likelihood units
between two machines, and esag matches or beats R on 8 of 18 fits.

**Calibration under the null.** These are the ignored tests, run with
`cargo test --release -- --ignored`, at the 5% level.

| test | rejection rate |
|---|---|
| goodness of fit (200 replications), two-sample / one-sample variant | 0.070 / 0.055 |
| γ free of x, a true null (1,000 replications), RoC / D / M / LR | 0.041 / 0.042 / 0.046 / 0.044 |
| the same LR with a χ²₂ reference instead of the bootstrap | 0.123 |

At 1,000 replications the standard error of a rejection rate is 0.007.

The last row shows why the papers use the bootstrap.

**Size and power when V is anisotropic** (`cargo run --release --example
aniso_study -- size|power`). The setting is n = 2000, 5 covariates + const,
B = 199, and data from regressions whose V(x) is far from the identity:
the median ratio of the largest to the smallest shape eigenvalue is 2–4 at
d = 3, 8–10 at d = 5 and 33–85 at d = 8. There are four nulls: μ free of
x1, the shape free of x1, no effect of x5, and the shape free of every
covariate (the same anisotropic eigenvalues at every x). Each null has 300
replications (s.e. 0.013) unless stated.

| shape, d | rejection at 5%, RoC / D / M / LR / LM, over the four nulls |
|---|---|
| γ, d = 3 | 0.037–0.060 |
| log-tangent, d = 3 | 0.027–0.063 |
| γ, d = 5 | 0.023–0.060; LR averages 0.037 (slightly conservative, `docs/MATH.md` §6) |
| log-tangent, d = 5 | 0.033–0.073 |
| log-tangent, d = 8 | 0.027–0.070 |
| γ, d = 8 (score test: M and LM; 200 replications) | 0.015–0.070; LM averages 0.050 |

At an effect a quarter of the simulated one (c = 0.25), RoC, D, LR and LM
reject 14–95% of the time at d = 3, 53–100% at d = 5 (13–75% at c = 0.1)
and 99–100% at d = 8 (log-tangent). The score statistic LM has LR's power:
the two differ by at most 0.08, and mostly by 0.02 or less. `M` stays at
3–8% at d = 3, because it averages moment residuals over all observations,
and a centred covariate's effect cancels in that average to first order.

**A hard scenario** (`cargo run --release --example hard_scenario --
fit|tests|lasso|recovery`): correlated covariates (ρ = 0.5), moderate concentration
(‖μ₀‖ = 2.5), μ and V both strongly x-dependent, sparse effects (x1 moves
μ and the shape, x2 only μ, x3 only the shape, x4 both weakly, x5
nothing). Fits with SEs on one core, 12 replications per cell:

| d | shape | n = 200 | n = 500 | n = 2000 | converged | V̂(x) error at n = 2000 |
|---|---|---|---|---|---|---|
| 3 | log-tangent | 0.6 ms | 0.9 ms | 2.5 ms | 36/36 | 0.12 |
| 3 | γ | 1.7 ms | 3.4 ms | 12.6 ms | 36/36 | 0.12 |
| 4 | log-tangent | 0.9 ms | 1.8 ms | 6.4 ms | 36/36 | 0.19 |
| 4 | γ | 100 ms | 152 ms | 122 ms | 16/36 (3 on the singular set) | 0.23 |
| 5 | log-tangent | 1.9 ms | 3.4 ms | 11.4 ms | 36/36 | 0.26 |
| 5 | γ | 60 ms | 73 ms | 0.69 s | 2/36 (26 on the singular set) | 0.34 |
| 8 | log-tangent | 12 ms | 17 ms | 53 ms | 35/36 | 0.44 |
| 8 | γ | 0.12 s | 0.17 s | 0.30 s | 0/36 (35 on the singular set) | 0.56 |

The V̂(x) error is the mean geodesic distance from the true V(x) over 200
new covariate points. γ at d ≥ 5 does not converge on this data because
its fits stop in traps, local suprema at the singular set of the γ map
(see "The γ parameterization: what porting found" below); the log-tangent
model does.

**How well μ(x) and V(x) are recovered** (`hard_scenario -- recovery`,
`TRUTH=gamma`: data from the γ model, so the log-tangent fits are
misspecified; 12 replications). V(x) is 0.5–0.9 from isotropy and moves
0.14–0.40 with x (geodesic distance; 0.25 is roughly a 13–28% error in V's
eigenvalues):

| d | estimator | V̂ error, n = 500 | n = 2000 | μ̂ direction error, n = 2000 |
|---|---|---|---|---|
| 3 | log-tangent, full model | 0.22 | 0.12 | 1.6° |
| 3 | γ, full model | 0.27 | 0.12 | 1.5° |
| 4 | log-tangent, full model | 0.43 | 0.26 | 2.1° |
| 4 | γ, full model | 0.62 | 0.24 | 2.1° |
| 5 | log-tangent, full model | 0.57 | 0.29 | 2.6° |
| 5 | log-tangent, true covariates only | 0.45 | 0.24 | 2.1° |
| 5 | log-tangent, constant shape | 0.44 | 0.37 | 2.7° |
| 5 | log-tangent, adaptive LASSO (relaxed) | 0.48 | 0.29 | 2.7° |
| 5 | γ, full model | 0.82 | 0.32 | 2.6° |

On γ's own data the log-tangent fit recovers V(x) about as well as the γ
fit (within 0.02 at n = 2000), and better at n = 500 or where γ is trapped
(d = 5). The error halves when n quadruples. At n = 2000 the full model
beats a constant shape. At n = 500 (84 coefficients at d = 5) the
estimation noise of the shape slopes is as large as V's dependence on x:
a constant shape, or the true covariates only, do better than the full
model. On log-tangent data the picture is the same (full model 0.12 /
0.19 / 0.26 at n = 2000 for d = 3 / 4 / 5).

Bootstrap tests on the same data (n = 2000, B = 199, 200 replications,
s.e. 0.015; at d = 8 score tests, 100 replications), rejection rates at 5%:

| null | log-tangent d = 3 | γ d = 3 | log-tangent d = 5 | log-tangent d = 8, LM |
|---|---|---|---|---|
| μ free of x3 (true) | 0.035–0.050 | 0.030–0.060 | 0.055–0.065 | 0.070 |
| shape free of x2 (true) | 0.035–0.065 | 0.045–0.065 | 0.045–0.060 | 0.020 |
| x5 no effect (true) | 0.030–0.060 | 0.050–0.065 | 0.055–0.075 | 0.060 |
| shape constant (false) | 1.00 | 0.87–0.92 | 1.00 | 1.00 |
| shape free of x1 (false) | 1.00 | 0.70–0.79 | 1.00 | 1.00 |
| x4 no effect, weak (false) | 0.92–0.96 | 0.75–0.84 | 0.87–0.96 | 0.93 |

Ranges are over RoC, D, M, LR and LM for the true nulls and over RoC, D,
LR and LM for the false ones. `M` rejects the false nulls only 3.5–31% of the
time (it has no first-order power against slopes, `docs/MATH.md` §6).

LASSO family on the same data (log-tangent, BIC at relaxed refits, 40
replications; 12 at d = 8), covariate parts found:

| d, n | penalty | strong parts (of 4) | weak x4 parts (of 2) | false (of 4) | time per path, one core |
|---|---|---|---|---|---|
| 3, 2000 | LASSO | 4.00 | 1.27 | 0.23 | 95 ms |
| 3, 2000 | adaptive LASSO | 4.00 | 0.97 | 0.00 | 0.10 s |
| 3, 2000 | group LASSO (mean / shape parts) | 4.00 | 0.40 | 0.00 | 82 ms |
| 3, 2000 | adaptive group LASSO | 4.00 | 0.45 | 0.00 | 85 ms |
| 3, 2000 | sparse group LASSO (α = 0.5) | 4.00 | 0.85 | 0.03 | 96 ms |
| 5, 2000 | LASSO | 3.98 | 1.57 | 1.40 | 0.47 s |
| 5, 2000 | adaptive LASSO | 4.00 | 1.30 | 0.28 | 0.55 s |
| 5, 2000 | group LASSO | 3.55 | 0.55 | 0.10 | 0.33 s |
| 5, 2000 | adaptive group LASSO | 3.88 | 0.10 | 0.00 | 0.35 s |
| 5, 2000 | sparse group LASSO | 3.77 | 0.78 | 0.20 | 0.45 s |
| 8, 2000 | LASSO | 3.17 | 1.33 | 1.00 | 2.6 s |
| 8, 2000 | adaptive LASSO | 3.75 | 1.25 | 0.50 | 3.0 s |
| 8, 2000 | group LASSO | 2.00 | 0 | 0 | 1.6 s |
| 8, 2000 | adaptive group LASSO | 2.00 | 0 | 0 | 1.8 s |
| 8, 2000 | sparse group LASSO | 2.00 | 0.58 | 0 | 2.4 s |

The strong parts are x1's mean and shape effects, x2's mean effect and x3's
shape effect; x4's are a quarter as large. Adaptive LASSO finds every
strong part at d = 3 and 5 with few false ones. At d = 8 a shape group has
27 coefficients, and BIC's `27 · log n` charge per group keeps the group
penalties from admitting any shape effect; 5-fold cross-validation (held-out
log-likelihood) goes the other way and admits nearly everything (adaptive
group LASSO: 4/4 strong, 1.75/2 weak, 1.5 false; the others 3.8–4 false).
At n = 300 every method is conservative (1.2–3.2 strong parts at d = 3 and
5). At n = 100, d = 5 (84 coefficients, where the unpenalized MLE does not
exist) the paths still select regular sparse models (§8 of `docs/MATH.md`).

**Estimating V(x)** (`cargo run --release --example v_recovery`; data from
a known regression with 5 covariates + const). The geodesic error of V̂(x)
against the true V(x) shrinks at the 1/√n rate in both shape models: at
n = 2000 it is 0.11–0.46 for d = 3–8. 95% basic-bootstrap intervals for
the shape eigenvalues of V(x₀) cover 93–98% of the time (d = 3 and 5,
n = 2000), and geodesic balls cover 93.5–97.5%. Details and the frame
caveat of the log-tangent model are in `docs/MATH.md` §5b and §7.

## Performance

Measured on the same machine (Apple M5 Pro), with the same data and settings
as the R timings stored in the fixtures. R is single-threaded, so the
one-core column is the like-for-like comparison. Run
`RAYON_NUM_THREADS=1 cargo run --release --example bench_vs_r`.

| task | R 4.6.1 | esag, 1 core | speed-up, 1 core | esag, all cores |
|---|---|---|---|---|
| MLE + Hessian, regression d=3, n=200, p=2 | 7.02 s | 1.29 ms | 5,400× | 0.77 ms |
| MLE + Hessian, regression d=3, n=1000, p=2 | 36.5 s | 3.07 ms | 11,900× | 1.33 ms |
| MLE + Hessian, i.i.d. d=3, n=300 | 3.24 s | 0.50 ms | 6,400× | 0.76 ms |
| MLE + Hessian, i.i.d. d=4, n=300 | 12.4 s | 1.50 ms | 8,300× | 1.08 ms |
| `Bootstrap_GoF`, 20 refits, n=200 | 79 s | 2.73 ms | 29,000× | 0.77 ms |
| `Prediction_Region`, m=2000, 5 refits | 21 s | 0.63 ms | 33,000× | 0.39 ms |

### Why it is fast on one core

Parallelism adds a factor of about 10 on 15 threads; everything else is
algorithmic and holds on a single core. Memory stays small: a B = 500 test
at n = 2000 peaks at 5–9 MB (31 MB for d = 8). Replicates are generated,
refitted and reduced to four numbers one at a time.

* **Exact derivatives.** The R code uses `optim` with finite-difference
  gradients (k + 1 likelihood passes per gradient for k coefficients) and a
  numerical Hessian. esag computes the gradient in about one pass (dual
  numbers and an adjoint for γ, closed-form Daleckii–Krein derivatives for
  the log-tangent shape) and the Hessian exactly.
* **A compiled per-observation kernel.** It works on the eigen-structure of
  V directly, with no d × d inverse, and uses a stable `log M_p` with fixed
  sizes on the stack: 37–160 ns per observation for γ (d = 3–8) and
  55–1,070 ns for the log-tangent shape.
* **Exact moments for M.** The R code estimates `E[y²|x_i]` by Monte Carlo
  (10⁴ draws per observation per replicate). esag uses a 1-D integral that
  is exact to 1e−12, with both tails of the quadrature summed in closed
  form: 0.3–0.6 µs per observation.
* **Bootstrap refits that start where they end.** Each refit is
  warm-started at the original fit and whitened with its inverse
  information, takes a Newton-sized first step, and stops on a statistical
  rule: coefficients within 0.0014 SE, log-likelihood within 1e−6. A refit
  takes 6–10 likelihood passes at n = 2000. Where γ has cliffs (d ≥ 5 with
  covariates) the metric is the empirical information at the refit's
  start, recomputed every 50 iterations.
* **A smooth shape model where γ is singular.** The log-tangent shape
  removes the discontinuities of γ ↦ V that make big γ regressions take
  hundreds of iterations or fail.
* **One whitened start for the smooth model.** A log-tangent fit runs the
  moment start alone, whitened with the exact information there: 9–10
  likelihood passes instead of 2 × 23–34 for the two raced starts (both
  starts reached the same maximum in 192 of 192 simulated datasets). Fits
  with SEs at n = 2000, 5 covariates: 2.8 / 12 / 50 ms at d = 3 / 5 / 8,
  2.4–2.9× faster than before.
* **Structure of the restricted models.** Under the shape-constant null
  every observation has the same shape S, so one eigen-decomposition per
  pass serves them all (restricted refits 4–9× faster at d = 5–8). The
  score statistic's information has Kronecker structure
  `(x̃x̃ᵀ) ⊗ (ggᵀ)`, so it is one matrix product over the unique products
  (LM 2.2–4.2 ms at d = 8, from 8.8–10.9).
* **No time spent where no answer exists.** A γ fit with observations
  pinned at the singular set, or a fit whose likelihood is unbounded
  (V(x) degenerating at small n), skips the restart waves that can only
  find other such points (γ d = 8, n = 2000: 1.7 → 0.3 s), and says why
  (`singular_obs`, `degenerate_obs`).
* **Small things that add up.** Matrix-product assembly of η = B X̃ᵀ and of
  the gradient, cube and square roots instead of `powf`, a Newton finish
  for standard errors, and early pruning of dominated starts.

`cargo run --release --example perf_suite` times a wider set of workloads
(medians, all threads; γ shape):

| workload | v0.1 | previous speed pass | 2026-10-24 | now |
|---|---|---|---|---|
| fit with SEs, d = 3, n = 1000 | 8.7 ms | 1.4 ms | 1.4 ms | 1.2 ms |
| fit with SEs, d = 5, n = 2000 | 2.77 s | 9.0 ms | 12 ms | 11 ms |
| the paper's hydrochemistry fit (d = 4) | 18.2 ms | 1.9 ms | 1.0 ms | 1.0 ms |
| bootstrap GoF, 200 replicates, n = 1000 | 139 ms | 45 ms | 36 ms | 23 ms |
| prediction region, 100 refits, n = 1000 | 92 ms | 26 ms | 18 ms | 15 ms |
| hypothesis test, 200 replicates, d = 3, n = 1000 | 811 ms | 382 ms | 106 ms | 72 ms |
| hypothesis test, 300 replicates, hydrochemistry | 819 ms | 221 ms | 44 ms | 13 ms |
| `regression_demo`, whole run | 6.3 s | 2.0 s | 0.6 s | 0.53 s |

The 2026-10-24 column was measured with other processes running (load
average about 5); "now" on an idle machine.

Bootstrap hypothesis tests at B = 500, n = 2000, 5 covariates + const, on
data whose V(x) is anisotropic (`cargo run --release --example test_bench
3,5,8`, with `RAYON_NUM_THREADS=1` for one core). The four nulls are the
shape free of every covariate, μ free of x1, the shape free of x1, and no
effect of x5. The full test bootstraps RoC, D, M, LR and LM; the score test
(`score_test`) bootstraps LM and M from restricted refits only, with the
same LM and M replicates:

| d | log-tangent, full | log-tangent, score | γ, full | γ, score |
|---|---|---|---|---|
| **one core** | | | | |
| 3 | 1.7–1.9 s | 1.3–1.4 s | 2.9–3.6 s | 1.1–1.6 s |
| 5 | 5.0–8.9 s; shape constant 5.0 s | 2.4–5.3 s; shape constant 2.4 s | 24–44 s; shape constant 96 s | 6.6–11 s; shape constant 2.7 s |
| 8 | 21–32 s; shape constant 21 s | 6.4–18 s; shape constant 6.4 s | 87–289 s; shape constant 285 s† | 28–100 s; shape constant 11 s† |
| **15 threads** | | | | |
| 3 | 0.17–0.18 s | 0.12–0.13 s | 0.25–0.31 s | 0.11–0.15 s |
| 5 | 0.46–0.78 s | 0.25–0.49 s | 2.3–3.9 s; shape constant 8.0 s | 0.63–1.0 s; shape constant 0.30 s |
| 8 | 2.2–3.4 s | 0.80–1.9 s | 8.5–28 s | 1.0–8.3 s |

† B = 50, scaled by 10. γ at d ≥ 5 is slower because its likelihood is
rough there, not because its kernel is: refits need 50–800 evaluations and
have local maxima, and with strong shape effects they stop in traps
(`docs/MATH.md` §2, §6). A log-tangent replicate costs about 7 evaluations
per refit, near the floor for a quasi-Newton refit from the original fit;
its time is the per-observation eigen-decomposition (1.1 µs at d = 8). The
score statistic's information is assembled from its Kronecker structure:
0.25 ms per replicate at d = 3 and 2.2–4.2 ms at d = 8. The Python session
in `python/examples/demo.py` (d = 3, γ, five nulls, five penalties) runs
each test in 1.1–1.8 s and each penalty path in 0.6–1.3 s.

Large regressions (`cargo run --release --example big_fit`, fit with SEs;
data drawn from the log-tangent model):

| d, covariates | n | γ shape | log-tangent shape |
|---|---|---|---|
| 7, 5–6 | 1,000 | 75–140 ms, did not converge | 7–9 ms, converged |
| 7, 5–6 | 20,000 | 0.32–1.5 s, one not converged | 78 ms |
| 8, 5–6 | 1,000 | 85–377 ms, did not converge | 14–17 ms, converged |
| 8, 5–6 | 20,000 | 0.33–0.34 s, did not converge | 85–104 ms |

(All threads. The γ fits that do not converge report the observations on
the singular set, `singular_obs`, and no longer run the restart waves.)

On data drawn from the γ model the γ fit takes 40 ms–1.5 s (bit-for-bit
reproducible; fits that end at a cliff of the parameterization are flagged
`converged = false`), and the log-tangent fit stays fast but is a different
regression model there; compare the two with AIC/BIC (same parameter count).

Penalized paths (31 λ values, BIC at relaxed refits): isotropy, i.i.d.
d = 6: 25–45 ms; eigen-gap, i.i.d. d = 6: 0.2 s; driver selection with
6 covariates, n = 2,000: 0.85 s (d = 7, 189 parameters) and 1.3 s (d = 8,
245 parameters).

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
  default starts race in lockstep (deterministically): a start that
  converges caps the others, starts far behind the leader or crawling
  against a cliff of the γ parameterization are stopped, and the highest
  likelihood wins.
* **Cheap standard errors.** The observed information is exact at d = 3 (and
  for the log-tangent shape in every d) and uses differences of the exact
  gradient for γ at d ≥ 4. It is assembled with one matrix product per chunk,
  and with SEs requested L-BFGS hands over to Newton steps that reuse it.
* **Exact moments and preconditioned refits.** The `M` statistic's moments
  come from a 1-D integral instead of 1000 draws per observation; bootstrap
  refits run in coordinates whitened by the original fit's observed
  information.
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
* **Fits that say why they did not converge.** A γ fit with observations on
  a singular set of the parameterization (`singular_obs`), or a fit whose
  likelihood is unbounded because V(x) degenerates at high-leverage
  observations (`degenerate_obs`, small n), is flagged with a summary note,
  and the penalized paths stay well defined there (they descend from the
  null model and never select a degenerate refit).
* **The paper's goodness-of-fit statistic.** Algorithm 1 (two-sample KS
  against a model-based reference) is the default; the R code's one-sample χ²
  variant is still available.
* **Results the R code doesn't return.** The observed LR with bootstrap and
  χ² p-values, bootstrap SEs, AIC and BIC, JSON persistence, prediction
  regions at any covariate value, and held-out coverage.

The full list of R-code issues found while porting is in
`tests/fixtures/README.md`.

## The γ parameterization: what porting found

`γ ↦ V` is discontinuous wherever a group `j ≥ 2`, or the last two entries
of such a group, vanish while lower groups do not: approaching such a point
from different directions gives `V`s that differ by O(1) however close one
gets (`param::singular_set_tests`; `reference/checks/gamma_discontinuity.R`
shows it with the original R functions). Group 1 and `γ = 0` as a whole are
continuous. Consequences:

* **Large regressions.** With `γ_i = Γx̃_i`, observations near that set
  create cliffs in the likelihood. At d = 7–8 with 5–6 covariates the
  likelihood is multimodal and optimizers stall with one observation pinned
  at the set; this, not arithmetic, is what made big fits slow.
  `Shape::LogTangent` has no such set.
* **Traps.** Near the set an observation's density depends on the angle
  of the vanishing pair, and a tiny move of B turns that angle all the way
  round. So every observation on the set at its best angle is a trap: a
  local supremum of the likelihood that is not a regular maximum, where
  optimizers stall. Each observation brings its own family of traps, and a
  fit's free shape slopes can reach them, most easily at observations with
  large covariate values, even when the true coefficients keep every observation far
  from the set; one shape covariate is enough. With strong shape effects at
  d = 5 and 8 (`examples/hard_scenario.rs`), 0–2 of 12 γ fits converge at
  every n from 200 to 2000; fits started at the true coefficients end in
  traps too (13 of 16 at d = 5); the observation with the largest gradient
  sits at its density's maximum over the angle with its trailing pair
  nearly zero (10⁻⁷–10⁻³ of its group norm); and more restarts only find
  other traps. Regular maxima exist and, where found, were mostly higher
  than the trap (4 of 5 comparisons), but gradient methods do not reach
  them reliably. This is a property of the γ-linear model, not of the
  optimizer; the log-tangent model on the same kind of data converges in
  251 of 252 fits from n = 200 on, and at d = 4 in 36 of 36, 19–110×
  faster than γ.
* **The frame's pole.** `onb(μ)` is singular where `μ_0 = μ_1 = 0`. When
  the data's mean direction is near that set (for example near `e_d`), γ
  fits slow down even at d = 3 (105 vs 28 ms); the γ model is not
  equivariant under rotations of the response.
* **Project 3.** Exact zeros of a group `j ≥ 2` or of a trailing pair sit
  on the set, and both of the paper's simulation truths lie on it (nudging
  them by 1e−9 changes `V` by 4.6 and 0.8–1.0). The penalized module
  therefore offers the paper's methods as a reproduction mode and recommends
  penalties on the smooth shape. `docs/project3/METHODOLOGY.md` has the
  analysis; `docs/project3/SPEC.md` §10 lists discrepancies between the R
  code and the draft (for example Fig. 2 was scored against another
  experiment's truth, and the application uses Alistipes and `x/‖x‖` rather
  than Bacteroides and √proportions; the published numbers reproduce with
  the code as written).

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
src/optim/                      L-BFGS (resumable; strong + approximate Wolfe), FD Hessian,
                                  proximal Newton for penalized fits
src/logtangent.rs               smooth log-tangent shape: kernel, exact gradient and Hessian
src/penalized.rs                Project 3: LASSO-type penalized fits, λ paths, selection
src/stats/                      normal, χ², KS (exact and asymptotic, R-compatible), quantiles
src/distribution.rs             Esag
src/regression.rs, estimator.rs EsagRegressor / EsagEstimator (scikit-learn style)
src/regression/fit_profile.rs   profiling and diagnostic studies of the fits (ignored tests)
src/preprocessing.rs            MinMaxScaler
src/inference/                  goodness of fit, prediction regions, hypothesis tests, bootstrap
src/bin/esag.rs                 CLI
examples/                       regression_demo, bench_vs_r, perf_suite, big_fit, hard_scenario,
                                  aniso_study, test_bench, v_recovery, project3_paper
tests/                          R parity, paper reproduction, API, inference
reference/r/                    the original R sources
reference/checks/, project3/    R scripts behind the findings above
docs/MATH.md                    the mathematics, with references
docs/project3/                  Project 3: what the R code does, review and corrected design
```

## License

MIT, the same license as the original R code. `data/hydrochem_At_LLt.csv`
comes from the R package `compositions` (GPL ≥ 2); see `data/README.md`.
