# esag (Python)

Python bindings for the `esag` Rust crate: ESAG regression for directional
data with a statsmodels-style API. The fits, bootstrap tests and penalized
paths all run in Rust, in parallel, with the GIL released.

## Install

```sh
pip install maturin
cd python
maturin develop --release          # into the active virtualenv
# or build a wheel (abi3, Python >= 3.9):
maturin build --release
```

Requires numpy; pandas is optional. With pandas installed, results come
back as DataFrames labelled with your column names.

## Use

```python
import esag

model = esag.ESAG(Y, X)            # Y: n x d unit vectors, X: n x p covariates
res = model.fit()
print(res.summary())
```

```
                           ESAG Regression Results
==============================================================================
Dep. Variable:        east, north, up   No. Observations:                 2000
Model:                           ESAG   Df Residuals:                     1970
Shape:                          gamma   Df Model:                           30
Method:                           MLE   Log-Likelihood:             -2157.2158
...
       mu[1]      coef    std err          z     P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
const           1.4311      0.028     51.947     0.000       1.377       1.485
temp           -0.5738      0.026    -21.869     0.000      -0.625      -0.522
...
```

| statsmodels-style | |
|---|---|
| `res.params`, `bse`, `tvalues` (`zvalues`), `pvalues` | DataFrames: rows `const` + covariates, columns `mu[k]`, then `gamma[k]` (or `s[k]`) |
| `res.conf_int(alpha)` | (equation, exog) × (lower, upper) |
| `res.cov_params()` | covariance of the estimated coefficients |
| `res.llf`, `aic`, `bic`, `nobs`, `df_model`, `df_resid`, `converged` | |
| `res.predict(exog)`, `linear_predictor`, `resid`, `loglike_obs()`, `simulate()` | |
| `res.summary(alpha)` | text summary (prints; `as_text()`; HTML in notebooks) |

### Bootstrap tests

```python
t = res.test("isotropy", n_boot=500)        # shape = 0
t = res.test("mu", "temp", n_boot=500)      # mean direction free of temp
t = res.test("shape", "temp")               # shape free of temp
t = res.test("feature", "temp")             # temp has no effect
t.pvalue        # {'RoC': ..., 'D': ..., 'M': ..., 'LR': ..., 'LR_chi2': ...}
t.table         # statistic and bootstrap p-value per statistic
print(t.summary())
```

Every test computes RoC, D, M (with exact second moments) and LR from one
set of bootstrap replicates. B = 500 with n = 2000 and 5 covariates takes
0.3–1.8 s at d = 3 and about 1 s at d = 5 (log-tangent).

### LASSO family

```python
reg = model.fit_regularized("adaptive_lasso")                     # every slope
reg = model.fit_regularized("group_lasso")                        # one group per covariate
reg = model.fit_regularized("adaptive_group_lasso", groups="covariate_part")
reg = model.fit_regularized("sparse_group_lasso", l1_ratio=0.3)
reg = model.fit_regularized("lasso", target="mean", select="cv")  # mean slopes only, 10-fold CV
reg = model.fit_regularized("group_lasso",
                            groups=[[("mu[1]", "temp"), ("mu[2]", "temp")]])
reg = model.fit_regularized("lasso", alpha=0.02)                  # fixed weight, no path
print(reg.summary())       # relaxed refit of the selected model, dropped terms marked
reg.params                 # penalized coefficients at the selected weight
reg.relaxed                # an ESAGResults (SEs, tests, ...)
reg.path                   # alpha, lambda, df, llf, criterion, n_active, selected
```

| argument | values |
|---|---|
| `penalty` | `lasso`, `adaptive_lasso`, `group_lasso`, `adaptive_group_lasso`, `sparse_group_lasso`, `isotropy`, `eigen_gap` (i.i.d., log-tangent) |
| `target` | `slopes` (default), `mean`, `shape_slopes`, `shape` |
| `groups` | `covariate` (default), `covariate_part`, `parameter`, `all`, `gamma`, or lists of `(parameter, variable)` pairs |
| `adaptive` | `True` (r = 1), a number r, or `False` |
| `select` | `bic` (default), `aic`, `cv` (`cv_folds`, `seed`) |
| `alpha` | penalty weight(s) per observation (`lambda / n`); default: automatic path |
| `paper` | `"LASSO"`, `"aLASSO"`, `"RoC-LASSO"`, `"GLASSO"`, `"aGLASSO"`, `"RoC-GLASSO"`: the Project 3 R scripts' methods |

Covariates are standardized internally for the penalty; coefficients are
reported on the original scale.

### Shapes

`shape="auto"` (default) fits the papers' γ parameterization when d ≤ 4 or
there are no covariates, and the smooth log-tangent parameterization
otherwise. The γ map has a singular set that makes big γ regressions slow
and unreliable. With covariates the two are different regression models
for the shape (the mean model is the same); compare them by AIC/BIC.

### Distributions

```python
dist = esag.Distribution(mu=[1.0, -0.5, 2.0], gamma=[0.6, -0.3])
y = dist.sample(1000, seed=1)
dist.logpdf(y), dist.covariance, dist.second_moment, dist.mean_direction
```

## Tests and demo

```sh
python -m pytest tests            # 20 tests, ~3 s
python examples/demo.py           # fit, five bootstrap tests, five penalties (d = 3)
python examples/demo.py 5         # the same at d = 5
```
