---
name: esag
description: Operate the esag package (Rust crate, CLI and Python binding for ESAG regression of directions on covariates) - fit regressions in any shape parameterization (gamma, gamma-rot, spectral, log-tangent, gamma-transported), run bootstrap and score tests, LASSO-family penalized fits, estimate V(x) with intervals, predict, simulate, reproduce the papers, run the benchmark and diagnostic studies, and build and test the crate and the Python wheel. Use whenever a task touches ESAG, directional regression on the sphere, Yu & Huang's gamma parameterization, or this repository.
---

# esag: how to operate it

The repository (`~/Developer/esag-rs`, branch `perf/speed` in the worktree
`~/Developer/esag-rs.worktrees/perf-speed`, private GitHub
`Zehaoyu217/esag-rs`) holds a Rust implementation of ESAG regression
(Elliptically Symmetric Angular Gaussian: a unit vector y on S^(d−1) with
mean direction mu(x) and covariance V(x) depending on covariates x), a CLI
(`esag`) and a Python package (`python/`, PyO3 + maturin). Read
`README.md` for the overview, `docs/MATH.md` for every formula and
measurement, `docs/project3/` for the LASSO paper, `ARCHITECTURE.md` for
the crate layout and the R-code conventions it reproduces.

## 1. Mental model

- The model: `eta_i = B x_i` with `x_i = (1, covariates)`; the first d rows
  of B are mu(x), the remaining (d−2)(d+1)/2 rows are the shape
  coordinates of V(x) in one of five charts (`Shape`). All charts have the
  same coefficient count; with covariates they are different regression
  models, so compare them by AIC/BIC (log-likelihood).
- Shapes (Rust `Shape::…`, CLI `--shape …`, Python `shape="…"`):
  - `GammaRot` / `gamma-rot`: Yu & Huang's groups made smooth (group 1 is
    Paine's gamma; group j ≥ 2 is a log eigenvalue ratio `kappa_j` plus a
    rotation vector `omega_j`). Python's default (`shape="auto"`). Always
    converges; zero groups are regular points (good for group penalties).
    Depends on the frame reference `m` and a basis rotation `R0` from a
    pilot fit, both stored with the fit.
  - `Spectral` / `spectral`: the eigen-decomposition itself (log-eigenvalues
    `h[k]`, one Cayley rotation `a[k]`). Ordering-free; same frame
    conventions as gamma-rot.
  - `LogTangent` / `log-tangent`: `V = Xi exp(S) Xi^T`, smooth and
    first-order identifiable everywhere, fastest, no eigen reading.
  - `Gamma` / `gamma`: the papers' map (R parity; Rust and CLI default).
    Has a singular set from d = 4: regressions with covariates can stop in
    traps. `fit.singular_obs` lists the observations on the set,
    `fit.non_regular` marks a certified non-regular maximum.
  - `GammaTransported` / `gamma-transported`: gamma in the transported frame
    (no pole at mu_0 = mu_1 = 0).
- Which to use: gamma-rot for an eigen-interpretable model that converges;
  spectral when the ordering of the axes must not matter; log-tangent when
  speed or regularity at ties matters most; gamma only to reproduce the
  papers or at d = 3 without covariates. Where eigenvalues of V(x) cross
  inside the data (strong shape effects, d ≥ 8), penalize the eigen charts
  (adaptive LASSO / adaptive group LASSO over the eigen groups); see
  `docs/MATH.md` §2 for the measured table.
- Report V(x), not raw coefficients: gamma-type coordinates are identified
  only up to a discrete gauge. Use `predict_covariance` /
  `covariance_intervals` (Rust), `res.covariance()` / `res.covariance_ci()`
  (Python).

## 2. Build, test, environments

```sh
cd ~/Developer/esag-rs.worktrees/perf-speed            # cwd resets after every shell command: prefix each
cargo build --release                                   # library + CLI (target/release/esag)
cargo test --release                                    # ~160 unit + integration + doc tests, ~2 min with build
cargo clippy --release --all-targets                    # must stay clean
cargo fmt
cargo test --release --lib <name> -- --ignored --nocapture   # one of the ignored studies (section 7)
cd python && .venv/bin/maturin develop --release && .venv/bin/pytest -q tests   # Python 3.13 dev venv
cd python && .venv/bin/maturin build --release          # abi3 wheel in python/target/wheels/
```

Python 3.9 check: `uv pip install --python <venv39>/bin/python --reinstall --no-deps <wheel>`
then `<venv39>/bin/python -m pytest -q tests`. One-core timings: set
`RAYON_NUM_THREADS=1`. Never commit `Research_project3.zip` or the PDF in
the main checkout (untracked, 525 MB). Commit only on `perf/speed` (never
push to master); `git push origin perf/speed` works from the worktree; for
`gh` use `GH_TOKEN=$(gh auth token --user Zehaoyu217)`.

## 3. Python (statsmodels-style)

```python
import esag
model = esag.ESAG(Y, X, shape="auto")        # Y (n, d) unit vectors; X (n, p), const added
                                              # shape: auto|gamma-rot|spectral|gamma|gamma-transported|log-tangent
                                              # lt_reference=m, rot_basis=R0 to refit a stored model's frame
res = model.fit(maxiter=None, tol=None, restarts=0, seed=0, cov=True)
res.summary(); res.params; res.bse; res.pvalues; res.conf_int(0.05); res.cov_params()
res.llf, res.aic, res.bic, res.nobs, res.df_model, res.df_resid, res.converged, res.n_iter
res.singular_obs, res.degenerate_obs, res.non_regular, res.set_aside_obs
res.lt_reference, res.rot_basis                # the frame; pass back to ESAG(...) for the same model
res.predict(X_new); res.linear_predictor(X_new); res.resid; res.loglike_obs(); res.simulate(X_new, seed)
res.covariance(X_new)                          # (m, d, d) V-hat
ci = res.covariance_ci(x0, n_boot=500, level=0.95); ci.summary(); ci.eigen_table; ci.covariance_frame; ci.radius
t = res.test("shape", "temp", n_boot=500)      # nulls: isotropy | mu | shape | feature | constant_shape
t.pvalue, t.table, t.statistic, t.summary()    # RoC, D, M, LR, LM with bootstrap p-values
res.score_test("shape", "temp")                # LM and M from restricted refits only (faster)
res.gof(n_boot=200); res.prediction_region(x0, levels=(0.95, 0.9), n_mc=5000, n_boot=100)
res.to_json(); esag.ESAGResults.from_json(model, s)
reg = model.fit_regularized("adaptive_group_lasso", target="shape_slopes", groups="eigen_parts")
reg.summary(); reg.params; reg.relaxed; reg.alpha; reg.path; reg.params_path; reg.path_summary()
esag.Distribution(mu=[...], gamma=[...]).sample(1000, seed=1)   # .logpdf, .covariance, .second_moment, .mean_direction
esag.PENALTIES, esag.TARGETS, esag.GROUPINGS, esag.NULLS
```

`fit_regularized` arguments: `penalty` in `lasso | adaptive_lasso |
group_lasso | adaptive_group_lasso | sparse_group_lasso (l1_ratio) |
isotropy | eigen_gap` (i.i.d., log-tangent); `target` in `slopes | mean |
shape_slopes | shape`; `groups` in `covariate | covariate_part | parameter |
all | gamma` (the gamma family's groups, for gamma-rot each (kappa_j,
omega_j)) `| eigen_parts` (gamma-rot: each kappa_j alone and each omega_j as
a group; spectral: eigenvalues vs rotation) or explicit lists of
`("mu[1]", "temp")` pairs; `adaptive=True|r|False`; `select` in `bic | aic
| cv` (`cv_folds`, `seed`); `alpha` fixes the per-observation weight;
`paper="LASSO|aLASSO|RoC-LASSO|GLASSO|aGLASSO|RoC-GLASSO"` reproduces the
Project 3 R scripts (i.i.d., gamma). `python/README.md` has the details and
`python examples/demo.py [d]` a full session.

## 4. Rust API

```rust
use esag::regression::{EsagRegressor, EsagRegressorFit, Shape, Init};
let fit = EsagRegressor::new()
    .with_shape(Shape::GammaRot)                 // Gamma (default), LogTangent, GammaTransported, Spectral
    .with_feature_names(vec!["dose".into()])
    .with_lt_reference(None)                     // Some(m): frame reference; .with_rot_basis(Some(R0)) for the eigen charts
    .with_restarts(0, 0).with_hessian(true).with_parallel(true).with_tol(1e-6).with_max_iter(2000)
    .with_init(Init::Auto)                       // DataDriven | RStyle | Custom(B0); .with_mask(Some(mask)) for restricted fits
    .fit(&x, &y)?;                               // x: n × p (no intercept column), y: n × d unit rows
fit.summary(); fit.params(); fit.bse(); fit.pvalues(); fit.conf_int(0.05); fit.aic(); fit.bic();
fit.predict(&x_new)?; fit.predict_distribution(&x_new)?; fit.predict_covariance(&x_new)?;
fit.linear_predictor(&x)?; fit.log_likelihood_rows(&x, &y)?; fit.score(&x, &y)?; fit.residuals(&x, &y)?;
fit.distribution_at(&eta, fit.lt_frame().as_ref())?;   // one eta in the fit's own frame
fit.converged; fit.singular_obs; fit.degenerate_obs; fit.non_regular; fit.set_aside_obs;
fit.gr_frame(); fit.sp_frame(); fit.lt_frame();          // the stored frames
fit.to_json()?; EsagRegressorFit::from_json(&s)?;
```

Inference (`esag::inference`): `hypothesis_test(&fit, &x, &y, Restriction, B, mc, seed)`
with `Restriction::{MuOfFeature(j), GammaOfFeature(j), Feature(j), ConstantShape, Isotropy, Custom(mask)}`;
`score_test(...)` (restricted refits only); `bootstrap_gof(&fit, &x, &y, GofMethod::{TwoSampleReference, OneSampleChiSquare}, B, seed)`;
`prediction_region(&fit, &x, &y, &x0, n_mc, B, &levels, seed)`, `prediction_regions`, `region_coverage`;
`covariance_intervals(&fit, &x, &y, &x_new, B, level, seed)` (basic pairs-bootstrap intervals for V(x));
`bootstrap_coef`. Penalized (`esag::penalized`): `PenalizedEsag::{lasso, adaptive_lasso, group_lasso,
adaptive_group_lasso, sparse_group_lasso(a), isotropy, eigen_gap, paper(name)}()` then
`.target(Target::{Slopes, MeanSlopes, ShapeSlopes, Shape}).grouping(Grouping::{Covariate, CovariatePart,
Parameter, All, GammaGroups, EigenParts, Custom(groups)}).adaptive(r).shape(Shape).lt_reference(m)
.rot_basis(R0).bic()|.aic()|.cv(folds, seed).fit(&x, &y)?` giving a `PenalizedPath`
(`best()`, `relaxed`, `summary()`, `path_summary()`). statsmodels-style front end:
`esag::model::EsagModel::new(&y, &x).shape(..).exog_names(..).fit()? / fit_regularized(pen)?`.
Distributions: `esag::distribution::Esag::{new(mu, gamma), from_covariance(mu, V), from_eta}` with
`log_pdf, sample, covariance, second_moment, mean_direction, angles`; i.i.d. fits with
`esag::estimator::EsagEstimator`. Charts: `esag::gamma_rot::GrFrame`, `esag::spectral::SpFrame`,
`esag::logtangent::LtFrame` (`covariance`, `coords_from_covariance`, kernels).

## 5. Command line

```sh
B=target/release/esag
$B simulate --n 800 --seed 2024 --out sim.csv
$B fit sim.csv --y y1,y2,y3 --x x1 --shape gamma-rot --restarts 0 --model model.json
$B predict model.json sim.csv --x x1 --out pred.csv
$B gof model.json sim.csv --y y1,y2,y3 --x x1 --boot 200 [--chisq]
$B test model.json sim.csv --y y1,y2,y3 --x x1 --null shape --feature 0 --boot 500 [--score] [--mc 0]
   # --null isotropy | mu | gamma | feature | constant-shape
$B region model.json sim.csv --y y1,y2,y3 --x x1 --at 0.5
$B penalize sim.csv --y y1,y2,y3 --x x1 --penalty adaptive-group-lasso --target shape-slopes \
   --groups gamma --shape gamma-rot --select bic [--r 1] [--folds 10] [--paper RoC-LASSO] [--paper-grid]
```

CLI defaults: `fit --shape gamma`, `penalize --shape log-tangent`; pass
the shape explicitly for the eigen charts.

## 6. Reading a fit and avoiding the known pitfalls

- `converged == false` with `singular_obs` non-empty (gamma only): the fit
  is in a trap; refit with `gamma-rot` or `spectral`, or read
  `non_regular` (a certified non-regular maximum is converged).
- `degenerate_obs` non-empty: the likelihood is unbounded (about as many
  coefficients as observations): use `fit_regularized`.
- Eigen charts at d ≥ 8 with strong shape effects: the full model
  overfits (likelihood gain far above K/2); use the adaptive (group)
  LASSO on the shape slopes with BIC; penalized paths take 10–30 s at
  d = 8 (the log-tangent's 1–3 s).
- Frames: refitting the same gamma-rot/spectral/log-tangent model on other
  data needs the stored `lt_reference` (and `rot_basis`); simulation truths
  defined in another frame are misspecified otherwise.
- Group 1 sign / equivalence classes: compare fitted distributions
  (`predict_distribution`, `covariance`), never raw gamma coefficients
  across fits.
- Bootstrap procedures are reproducible for a given seed regardless of the
  thread count; `RAYON_NUM_THREADS=1` gives one-core timings.

## 7. Examples, benchmarks and diagnostic studies

- `cargo run --release --example regression_demo` (end-to-end, Yu & Huang 2025 Sec. 7.1 reproduction).
- `cargo run --release --example hard_scenario -- fit|tests|lasso|recovery` with env
  `D=3,5` `SHAPE=gamma|lt|gr` `NS=200,500,2000` `R` `B`: the hard scenario (mu and V(x) strongly x-dependent).
- `cargo run --release --example test_bench` (`D=3,5`, `SHAPE=gamma|lt|gr|sp`): B = 500 bootstrap tests at n = 2000.
- `cargo run --release --example aniso_study -- truth|size|power` (`SHAPE`, `R`, `B`, `C`, `NULLS`, `FAST=1`).
- `cargo run --release --example v_recovery`, `big_fit`, `perf_suite [--quick]`, `bench_vs_r`,
  `project3_paper`, `p3_corrected`.
- Ignored studies in `src/regression/fit_profile.rs` (run with
  `cargo test --release --lib <name> -- --ignored --nocapture`, env in each doc comment):
  `gamma_rot_prod` (production comparison of gamma-rot, spectral, log-tangent, full and penalized;
  `D N R SEED0 DATA=gamma|rot|spec SCALE PEN STD PRESET TRACE`), `gamma_rot_fits`, `gamma_big_d`,
  `gamma_trap_anatomy`, `gamma_smooth_fits`, `spectral_fits`, `gamma_transported_fits`, and others.
- Debug env vars: `ESAG_P3_DEBUG=1` (penalized path timings), `ESAG_PROX_DEBUG=1`
  (proximal Newton iterations), `ESAG_DEBUG_START=1` (gamma-rot starts), `ESAG_DEBUG_STRATUM=1`,
  `ESAG_DEBUG_HOMOTOPY=1` (gamma trap resolution).

## 8. Where the theory and numbers live

`docs/MATH.md`: §1 density, §2 gamma, its singular sets and traps, the
blown-up-coordinate resolution, gamma-rot and spectral with every
measurement, §3 regression and starts, §5b V(x) estimation, §6 tests,
§7 log-tangent, §8 penalized fits. `docs/project3/METHODOLOGY.md` and
`SPEC.md`: the LASSO paper and the discrepancies found in its R code.
`reference/` has the original R code and checks; `tests/fixtures` the R
parity data.
