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
| `res.singular_obs`, `res.degenerate_obs` | why a fit did not converge: observations on a singular set of γ (the fit stopped in a trap; use `shape="log-tangent"`), or with V(x) degenerating (the likelihood is unbounded; use `fit_regularized`) |
| `res.predict(exog)`, `linear_predictor`, `resid`, `loglike_obs()`, `simulate()` | |
| `res.summary(alpha)` | text summary (prints; `as_text()`; HTML in notebooks) |

### Bootstrap tests

```python
t = res.test("mu", "temp", n_boot=500)      # mean direction free of temp
t = res.test("shape", "temp")               # shape free of temp
t = res.test("feature", "temp")             # temp has no effect
t = res.test("constant_shape")              # shape free of every covariate (V(x) anisotropic,
                                            # same eigenvalues at every x)
t = res.test("isotropy")                    # shape = 0 (V = I)
t = res.score_test("shape", "temp")         # LM and M only, from restricted refits
t.pvalue        # {'RoC': ..., 'D': ..., 'M': ..., 'LR': ..., 'LM': ..., 'LR_chi2': ..., 'LM_chi2': ...}
t.table         # statistic and bootstrap p-value per statistic
print(t.summary())
```

Every test computes RoC, D, M (with exact second moments), LR and the score
statistic LM from one set of bootstrap replicates; `score_test` computes LM
and M from restricted refits only, with the same LM and M replicates and
LR's power. On anisotropic data with B = 500, n = 2000 and 5 covariates, a
test takes 0.2–0.4 s at d = 3, 0.7–0.9 s at d = 5 (log-tangent) and
2.5–4.4 s at d = 5 (γ) on 15 threads; `score_test` takes 0.13–0.18 s,
0.5–0.6 s and 0.8–1.2 s.

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
reported on the original scale. With about as many coefficients as
observations, where the unpenalized MLE does not exist, the path starts
from the null model, never selects a degenerate refit, and takes adaptive
weights from the plain penalty's selection.

### Shapes

`shape="auto"` (default) is `"gamma-rot"`: Yu & Huang's γ groups made
smooth (each group a log eigenvalue ratio and a rotation vector placing
the eigen-axis; group 1 is Paine's γ, so d = 3 is the original ESAG
parameterization). It converges where γ's fits stop in traps, keeps γ's
eigen-decomposition reading, and its zero groups are regular points, so
group penalties can select them. `"spectral"` takes the eigen-decomposition
itself as the parameters (log-eigenvalues `h[k]` and one rotation of the
axes `a[k]`); every axis enters alike, so unlike the γ family its
regression model does not depend on how the axes are ordered.
`"log-tangent"` is the smooth, first-order identifiable chart without an
eigen reading, and the fastest. `"gamma"` is the papers' map (with a
singular set from d = 4: `res.singular_obs` lists observations on it, and
`res.non_regular` marks a fit certified as a non-regular maximum);
`"gamma-transported"` is that map in the transported frame (no pole at
`μ_0 = μ_1 = 0`).

With covariates the shapes are different regression models for V(x) (the
mean model is the same) with equal parameter counts: compare them by
AIC/BIC. The gamma-rot, spectral and log-tangent models depend on the
frame's reference direction `m` (`lt_reference=`, default the data's mean
direction) and gamma-rot and spectral also on a basis rotation chosen from
a pilot log-tangent fit (`rot_basis=`); both are stored with the fit
(`res.lt_reference`, `res.rot_basis`) and printed in the summary, so the
same model can be fitted to other data with
`ESAG(Y2, X2, shape=res.model.shape, lt_reference=res.lt_reference,
rot_basis=res.rot_basis)`. Where eigenvalues of V(x) cross within the data,
the eigen charts' rotation slopes are weakly identified: penalize them
(`fit_regularized`, below); the adaptive group LASSO over the eigen groups
was the most accurate model at d = 8 in the measurements of
`docs/MATH.md` §2.

### The covariance V(x)

V depends on every coefficient nonlinearly, so per-coefficient standard
errors say little about it. Estimate it directly:

```python
V = res.covariance(X_new)                  # (m, d, d): V-hat at each row
ci = res.covariance_ci(n_boot=500)         # at the mean covariates (or pass x0)
print(ci.summary())                        # shape eigenvalues, SEs, basic 95% intervals, axes, V entries ± SE
ci.eigen_table, ci.covariance_frame, ci.radius   # radius of a geodesic 95% ball around V-hat
```

The intervals are basic (bias-corrected) pairs-bootstrap intervals. In
simulations (n = 2000, 5 covariates, d = 3 and 5) they cover 93–98% at level
95%; V-hat converges at the 1/√n rate (`docs/MATH.md` §5b).

With `shape="log-tangent"`, the model depends on the frame's reference
direction m (`lt_reference=`, default: the data's mean direction). The
summary prints it, and `res.lt_reference` returns it, so the same model can
be fitted to other data with `ESAG(Y2, X2, shape="log-tangent",
lt_reference=res.lt_reference)`; see `docs/MATH.md` §7.

### Distributions

```python
dist = esag.Distribution(mu=[1.0, -0.5, 2.0], gamma=[0.6, -0.3])
y = dist.sample(1000, seed=1)
dist.logpdf(y), dist.covariance, dist.second_moment, dist.mean_direction
```

## Tests and demo

```sh
python -m pytest tests            # 25 tests, ~3 s
python examples/demo.py           # fit, five bootstrap tests, five penalties (d = 3)
python examples/demo.py 5         # the same at d = 5
```
