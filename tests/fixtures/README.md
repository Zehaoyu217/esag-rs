# ESAG reference fixtures (generated from the R implementation)

Numerical reference values produced by running the R code in
`../src/ESAG/ESAG_Project2.R` (primary) and `../src/ESAG/ESAG.R` (older variant),
so that a Rust port can be validated against it. The R sources were **not modified**;
the driver scripts in `../rwork/` source copies with the `library("optimx")` line removed
(`optimx` is neither installed nor needed; only base `optim` and `MASS::mvrnorm` are used).

* R 4.6.1 (2026-06-24), aarch64-apple-darwin25.6.0, MASS 7.3.65.
* RNG: `Mersenne-Twister` / `Inversion` / sample.kind `Rejection` (R >= 3.6 defaults).
* Every number is written with `sprintf("%.17g")`, which round-trips an IEEE double exactly
  (so e.g. `0.1` appears as `0.10000000000000001` and `19/20` as `0.94999999999999996`).
* **Matrix convention**: every matrix is a row-major nested array, `M[i][j]` (0-based) is
  R's `M[i, j]` (1-based). Vectors are flat arrays. Coefficient matrices `B` are also given
  flattened in R's column-major order (`*_colmajor`), which is exactly `optim`'s parameter
  vector (`c(B)`).
* Each JSON file has a `meta` object with the R/MASS versions, seeds and notes.

## Source-order caveat (important for the port)

`ESAG_Project2.R`'s `log_likelihood()` calls `covariance_inv_matrix(mu, lambda, R)`, which is
**not defined in ESAG_Project2.R** – only in ESAG.R. Sourcing ESAG_Project2.R alone and
calling `log_likelihood` (hence `MLE`, `Bootstrap_GoF`, ...) fails with
`could not find function "covariance_inv_matrix"`. All fixtures were produced by sourcing
`ESAG.R` first and `ESAG_Project2.R` second, so that every common name (`ONB`, `parameter`,
`rotation`, `covariance_matrix`, `ESAG_generator`, `ESAG`, `log_likelihood`, `MLE`) is the
ESAG_Project2.R version and only `covariance_inv_matrix` (plus `covariance_matrix_2`,
`Model_diagnostic_mu/Q`, `Bootstrap_gof`) comes from ESAG.R.
`covariance_inv_matrix(mu, lambda, R)` = `P diag(1/lambda, 1) P'` is numerically identical to
ESAG_Project2.R's `covariance_matrix(mu, lambda, R, p = -1)` (max abs diff 0 on the tested case;
R's `x^(-1)` equals `1/x` bitwise here).

## Files

| file | contents |
|---|---|
| `parameterization.json` | `parameter`, `rotation`, `ONB`, `covariance_matrix_gamma` for d = 3,4,5,6 (deterministic) |
| `loglik.json` | `log_likelihood(c(mu,gamma), y)` for d = 3,4,5, 16 cases each, with intermediates |
| `regression_d3_n200.json` | d=3, p=2 regression data (seed 2024), MLE + Hessian, T1 residuals, timing |
| `regression_d3_n1000.json` | same with n=1000 (seed 2025) |
| `iid_d3_n300.json` | d=3 intercept-only fit on `ESAG(300, c(1,2,1.5), c(0.7,-0.3))`, seed 7 |
| `iid_d4_n300.json` | d=4 intercept-only fit on `ESAG(300, c(0.5,1,-1,2), c(0.6,-0.2,0.3,0.9,-0.5))`, seed 11 |
| `gof_small.json` | `Bootstrap_GoF(num_boot = 20)` on the n=200 dataset, seed 99 (stochastic sanity only) |
| `prediction_region_small.json` | `Prediction_Region(index=1, num_boot_m=2000, num_boot=5)` seed 5 and `Confidence_Region(sample_m=5000)` seed 3 (stochastic sanity only) |

Driver scripts (in `../rwork/`): `common.R` (sourcing + JSON writer), `param.R`, `loglik.R`,
`reg.R <n> <seed> <out.json> <out.rds>`, `iid.R <d3|d4> <out.json>`, `gof.R`, `pr.R`,
`probe.R`/`warn.R`/`check.R` (bug probes and sanity checks). Logs are `log_*.txt`.

### Dimension bookkeeping used everywhere

* `g = length(gamma) = (d-2)(d+1)/2` (d=3: 2, d=4: 5, d=5: 9, d=6: 14).
* `B_dim = (d^2+d-2)/2 = d + g` (d=3: 5, d=4: 9, d=5: 14).
* `lambda` has length d-1, `theta` has length d-2, `phi` has length (d-2)(d-3)/2 (empty for d=3).
* A "Bx" vector is `c(mu, gamma)` of length `B_dim`; for regression `Bx_i = B %*% X[i,]` with
  `B` a `B_dim x p` matrix whose column j multiplies covariate j (column 1 = intercept).

### `parameterization.json`

```
meta.cross_check_old_ESAG_R.max_abs_diff   {"ONB_d3_generic": 0, ..., "theta_d4_random_normal": 4.4e-16, ...}
dims[k]                                   one entry per d = 3,4,5,6
  .d, .g
  .mu_cases[m]        ONB depends only on mu, so it is given once per mu
     .label           generic | mu12_zero | only_last_nonzero(_neg) | unit_norm | mu1_zero | large_norm | mu123_zero | ...
     .mu              [d]
     .mu_norm         |mu|  (several cases have |mu| != 1)
     .ONB             d x d, columns are the orthonormal basis, LAST column = mu/|mu|  (= R's ONB(mu))
     .ONB_orthonormality_error   max|U'U - I|
  .gamma_cases[c]     >= 10 per d
     .label           zeros | small | moderate_positive | large_mixed_sign (|entries| 3..6.25) | all_negative |
                      random_normal | random_uniform_pm2 | (d=3) gamma2_zero_gamma1_pos / gamma2_zero_gamma1_neg / gamma1_zero_gamma2_neg |
                      (d>=4) first_pair_zero_rest_nonzero / second_group_zero / one_group_partially_zero
     .gamma           [g]
     .lambda [d-1], .theta [d-2], .phi [(d-2)(d-3)/2]   = parameter(gamma)[[1]], [[2]], [[3]]
     .lambda_prod     prod(lambda) (should be 1)
     .rotation        (d-1) x (d-1) = rotation(gamma)
     .rotation_orthonormality_error, .rotation_det
     .covariance[m]   one per mu case above (same order), with
        .mu_label, .mu
        .V_p1   = covariance_matrix_gamma(mu, gamma, p = 1)    (d x d)
        .V_pm1  = covariance_matrix_gamma(mu, gamma, p = -1)   (the inverse)
        .V_p05  = covariance_matrix_gamma(mu, gamma, p = 0.5)  (the symmetric square root)
```
Random gamma cases: `set.seed(20240101)` once before the loop over d; per d the draws are
`rnorm(g)` (random_normal) then `runif(g, -2, 2)` (random_uniform_pm2).
Algebraic checks on the written values (done in `check.R`): max |V V^-1 - I| = 4.3e-13,
max |V^0.5 V^0.5 - V| = 1.4e-14, eigenvalues of V = {lambda, 1} to 2.1e-13, |R'R - I| <= 5.6e-16,
|det R - 1| <= 6.7e-16, `V mu = mu` to 1.3e-14.
The cross-check shows ESAG.R's `atan2`-based theta and `apply`-based ONB agree with
ESAG_Project2.R's `acos`-based theta and loop-based ONB to <= 4.4e-16 on every case.

### `loglik.json`

```
cases[k]
  .d, .label
  .mu [d], .gamma [g], .y [d] (exactly unit-normalised before the call; .y_norm_check = |y|)
  .Bx = c(mu, gamma)
  .log_likelihood          = as.numeric(log_likelihood(Bx_i = Bx, y_i = y))
  .intermediates           recomputed in the driver with the same formulas (for debugging a port):
     .V_inv  d x d          covariance_inv_matrix(mu, lambda, R)
     .yVy                   y' V^-1 y
     .a                     y'mu / sqrt(yVy)
     .Mp [d-1]              Mp[1] = a*pnorm(a)+dnorm(a); Mp[2] = (1+a^2)*pnorm(a)+a*dnorm(a); Mp[i] = a*Mp[i-1] + (i-1)*Mp[i-2]
     .M_p                   Mp[d-1]
     .log_Cd                log((2*pi)^(-(d-1)/2))
  log_likelihood = log_Cd - (d/2)*log(yVy) + 0.5*((y'mu)^2/yVy - mu'mu) + log(M_p)
```
16 cases per d: y exactly / nearly along mu, exactly / nearly antipodal, exactly orthogonal
(ONB(mu) columns 1 and 2, so `a = 0`), two random unit y, isotropic gamma = 0, large gamma
(|entries| 3..5), mu with mu[1]=mu[2]=0, large |mu| (|mu| ~ 5.4-5.9), and small |mu| (0.1 x).
`set.seed(777)` once; per d: `eps <- rnorm(d)*0.01` then one `rnorm(d)` per `rand_sphere()` call
in case order. Cases labelled `*_numerically_delicate` have `a ~ -5.5..-5.8`, where
`M_p` is a difference of nearly cancelling terms; agreement there depends on the port's
`pnorm`/`dnorm` tail accuracy (R's `pnorm` is accurate to ~1e-16 relative even far in the tail).
Note the R code returns `-Inf` (not an error) once `M_p` underflows to 0 (e.g. |mu| = 40 antipodal) and `NaN`
when cancellation makes `M_p < 0`; `optim` copes with that (see "optim details").

### `regression_d3_n200.json`, `regression_d3_n1000.json`

```
meta.seed (2024 | 2025), meta.n, meta.d = 3, meta.p = 2, meta.B_dim = 5, meta.generation, meta.optim
B_true           5 x 2  = matrix(c(1.5,0.5,1.0,0.8,-0.4, -0.6,1.2,0.3,0.5,0.7), 5, 2)
B_true_colmajor  [10]   = c(B_true)
X                n x 2  (col 1 = 1, col 2 = x1 ~ N(0,1))
Y                n x 3  unit rows
joint_loglik_at_B_true            sum_i log_likelihood(B_true %*% X[i,], Y[i,])
joint_loglik_at_B_true_plus_0.1   same at B_true + 0.1 (every entry)
B_start          5 x 2  = matrix(c(1,1,1,0,0), 5, 2);  B_start_colmajor = c(1,1,1,0,0,1,1,1,0,0)
joint_loglik_at_B_start
mle
  .par_colmajor [10]  = optim $par (column-major: intercept column first)
  .par_matrix   5 x 2 = matrix(par, 5, 2); rows (mu1, mu2, mu3, gamma1, gamma2), cols (intercept, slope)
  .value              maximised joint log-likelihood
  .counts.function / .counts.gradient   optim counts (fn calls in line searches / gradient evaluations;
                                        each gradient costs 2*10 extra fn calls that are NOT in counts.function)
  .convergence        0
  .message            null
  .hessian      10 x 10  optim(hessian=TRUE) numerical Hessian of the log-likelihood at par (negative definite)
  .se_from_hessian_colmajor  sqrt(diag(solve(-hessian)))  (derived in the driver, not in the R source)
T1_residuals  [n]   residual_T(mles = par_matrix, Y, X)
ks_test_T1_vs_chisq2  {statistic (KS D), p_value, df = 2}  = ks.test(T1, "pchisq", 2)
timing  {mle_elapsed_sec, mle_user_sec, mle_sys_sec}   system.time() around MLE(..., HESS = TRUE)
```
Generation (exactly, in RNG order): `set.seed(seed); x1 <- rnorm(n); X <- cbind(1, x1);`
then for i in 1..n: `Y[i,] <- ESAG(1, mu = (B_true %*% X[i,])[1:3], gamma = (B_true %*% X[i,])[4:5])`.
`ESAG(1, ...)` = one `MASS::mvrnorm(1, mu, V)` (eigen-decomposition based; consumes `rnorm(3)`)
normalised to unit length. Because `mvrnorm` depends on LAPACK eigenvector signs, a port
cannot reproduce Y from the seed; use the stored X and Y.

### `iid_d3_n300.json`, `iid_d4_n300.json`

Same layout with p = 1 (`X = matrix(1, n, 1)`), no `X` stored, and
`true_mu`, `true_gamma`, `B_true = c(mu, gamma)`, `joint_loglik_at_true`,
`mle.par` (length B_dim; also split as `mu_hat`, `gamma_hat`), `mle.hessian` (B_dim x B_dim),
`ks_test_T1_vs_chisq` with df = d-1.
Generation: `set.seed(seed); Y <- ESAG(n, mu, gamma)` – a single `mvrnorm(n, mu, V)` call
(`matrix(rnorm(d*n), n)` filled column-major) with rows normalised.
`B_start = c(rep(1, d), rep(0, g))`.

### `gof_small.json`

`set.seed(99)` immediately before `Bootstrap_GoF(mles = matrix(par,5,2), Y, X, num_boot = 20)`
on the n=200 dataset. Fields: `mles_matrix`, `mles_colmajor`, `test_statistics` (which is the
KS **p-value** of the observed T1 residuals, i.e. equals `ks_test_T1_vs_chisq2.p_value` in
regression_d3_n200.json), `p_value = mean(test_statistics >= stats_boot)`, `observed_ks`,
`timing.elapsed_sec`. 20 bootstraps took 79 s (so it was not reduced to 10). Each bootstrap
round refits with `MLE(..., HESS = TRUE)` (the default), so the Hessian is computed 20 times for nothing.
Stochastic: a port can only check that its own value is in a plausible range (p_value is a
multiple of 1/20).

### `prediction_region_small.json`

* `prediction_region`: `set.seed(5)` then
  `Prediction_Region(X, Y, mles = matrix(par,5,2), index = 1, num_boot_m = 2000, num_boot = 5)`
  on the n=200 dataset. `quantiles` holds the six returned values keyed `"95% PR"`, ..., `"70% PR"`
  (= `quantile(Q_total, c(.95,.90,.85,.80,.75,.70))`, R type-7 quantiles of the 12000 pooled
  quadratic forms). Also `Bx_at_index` (fitted `mu`, `gamma` for observation 1), summaries of
  `Q_total` and of the first 2000 (non-bootstrap) values, and timing (21 s).
* `confidence_region`: `set.seed(3)` then `Confidence_Region(c(1,2,1.5), c(0.7,-0.3), sample_m = 5000)`;
  `quantiles` at 0.5 / 0.9 / 0.95 (type 7), a summary, and the `V_inv_used`.
  `q_l = (Y_l - mu/|mu|)' V^-1 (Y_l - mu/|mu|)`.

## Results at a glance

| fit | maximised log-lik | ll at truth | counts (fn, gr) | elapsed |
|---|---|---|---|---|
| regression d=3 n=200 | -219.16429822687812 | -221.6516 | 113, 34 | 7.0 s |
| regression d=3 n=1000 | -1160.5557271321 | -1165.127 | 84, 22 | 36.5 s |
| iid d=3 n=300 | -238.0692324490 | -242.1236 | 57, 20 | 3.2 s |
| iid d=4 n=300 | -349.6597453020 | -350.4929 | 112, 37 | 12.4 s |

KS test of T1 vs chi-square(d-1) at the MLE: n=200 p=0.117; n=1000 p=5.4e-9; iid d3 p=0.10;
iid d4 p=0.0046. The T1 residual is only approximately chi-square (the scaling
`|mu|^2 + sum(lambda) + 1` is a high-concentration approximation), so rejection at n=1000 on
data simulated from the model is a property of the method, not of the fixture.

## Bugs / inconsistencies noticed in the R code (F) – not fixed

Verified by running the code paths (`../rwork/probe.R`, `warn.R`, `check.R`):

1. **ESAG_Project2.R is not self-contained**: `log_likelihood()` calls `covariance_inv_matrix()`,
   defined only in ESAG.R. Standalone, `log_likelihood`, `MLE*`, `Bootstrap_GoF`, `Prediction_Region`,
   `hypothesis_testing_*` all fail with `could not find function "covariance_inv_matrix"`.
2. **`hypothesis_testing_isotropy`, `hypothesis_testing_gamma`, `hypothesis_testing_mu`**: the
   inner loop that builds the bootstrap sample is
   `for(j in 1:n){ Y_b[j,] = ESAG(1, mu = Bx[i, 1:d], gamma = Bx[i, (d+1):mle_len]) }` –
   it indexes `Bx` by the **bootstrap counter `i`** instead of the observation index `j`.
   Every bootstrap observation of round i is drawn from the parameters of observation i, and
   `num_boot > n` gives `subscript out of bounds`. Confirmed in all three functions.
3. **`MLE_isotropy` hard-codes d = 4**: `B[((l-1)*d+1):((l-1)*d+4)]` takes 4 entries per
   covariate regardless of d. For d=3, p=2 and B = 1..6 it builds `B_mle = 1 2 3 4 0 0 4 5 6 NA 0 0`
   (length 12 instead of 10, with an NA). Correct only for d = 4. As a consequence
   `hypothesis_testing_isotropy` errors (`non-conformable arguments`) for d = 3.
4. **`hypothesis_testing_gamma` start vector**: `c(mle_0)[-((length(mle_0)-d):length(mle_0))]`
   removes d+1 entries, but `MLE_restricted_gamma` expects `length(mle_0) - g` entries. Equal
   only when d+1 == g, i.e. d = 4. For d = 3 `matrix(B_mle, nrow = 5)` recycles with a
   "data length [8] is not a sub-multiple" warning and silently optimises the wrong model.
5. **ESAG.R `Bootstrap_gof`** hard-codes d = 4 (`MLE[1:4]`, `MLE[5:9]`), and calls `library(optimx)`
   inside the function body.
6. **`residual_T`, `Bootstrap_GoF`, `Prediction_Region`** document `mles` as "a vector of
   (d+2)(d-1)/2", but they compute `mles %*% t(X)`, which needs the `B_dim x p` **matrix**;
   with p = 2 a vector gives `non-conformable arguments` (with p = 1 R happens to promote the
   vector to a column and it works). All fixtures pass `matrix(par, B_dim, p)`.
7. **`Bootstrap_GoF`'s `test_statistics` is a p-value**, not a statistic:
   `stats = ks.test(T1, "pchisq", d-1)$p.value`, and `p_value = mean(stats >= stats_boot)`.
8. **`MLE` differs between the two files**: ESAG_Project2.R `MLE(B_start, Y, X, HESS = TRUE)`
   takes `B_start` as a vector and uses `X` as given (caller supplies the intercept column);
   ESAG.R `MLE(B_start = c(), Y, X = c(), H = FALSE)` prepends an intercept column to `X`
   (`cbind(1, X)`), requires `B_start` to be a **matrix** (a vector errors with
   `argument is of length zero` because `ncol(vector)` is NULL), defaults `B_start` to
   1s for mu / 0s for gamma when NULL, and defaults the Hessian to FALSE.
9. **`parameter()`** in ESAG_Project2.R dropped the dimension check that ESAG.R has
   (`if(as.integer(d) != d) return('incorrect dimensions ...')`); with a wrong-length gamma it
   returns garbage silently. ESAG.R returns a character string instead of erroring.
10. `theta`: ESAG_Project2.R uses `±acos(gamma1/r)` with the sign from `gamma2 >= 0`;
    ESAG.R uses `atan2`. They agree to 4.4e-16 on all fixture cases; the only difference is at
    `gamma2 = -0.0` (+pi vs -pi, same rotation). The ESAG.R `covariance_matrix` has no `p`
    argument; `covariance_matrix_2` (p = 0.5) and `covariance_inv_matrix` (p = -1) are separate functions.
11. Cosmetic / performance: `ONB` uses `mu[1:i-1]`, which R parses as `mu[(1:i)-1]` and equals
    `mu[1:(i-1)]` (index 0 is dropped), so it is harmless; the `joint_log_likelihood` closures
    recompute the full `Bx = t(matrix(B) %*% t(x))` inside the per-observation loop;
    `ESAG_generator(n = 1)` returns a plain vector while `n > 1` returns a matrix; `B_boot` in
    `hypothesis_testing_*` is unused; the `second_mome`/`mome_stats` Monte-Carlo statistics make
    `hypothesis_testing_*` doubly stochastic.
12. `log_likelihood` returns a 1x1 matrix (because `yVy`, `t(y)%*%mu` are matrices), not a scalar;
    `log(M_p)` produces `NaN` (warning) when `a << 0` makes `M_p` slightly negative by cancellation,
    and `-Inf` when it underflows. This happened at 12 trial points of the n=1000 fit and 1 of the
    d=4 fit (and 2 in the GoF bootstrap); the final estimates are unaffected (see next section).

## optim details (G)

`MLE()` calls `optim(par, fn, control = list(fnscale = -1, maxit = 10000), method = "BFGS", hessian = HESS)`.
Everything else is the `optim` default (verified from `optim`'s body in R 4.6.1):

| setting | value |
|---|---|
| method | BFGS (R's `vmmin`, variable-metric with inverse-Hessian BFGS update, B0 = I) |
| fnscale | -1 (maximisation: minimises `-loglik`) |
| parscale | 1 for every parameter |
| maxit | 10000 (default would be 100); counts never got close (max 37 gradients) |
| reltol | `sqrt(.Machine$double.eps)` = 1.4901161193847656e-08 |
| abstol | -Inf |
| ndeps | 0.001 for every parameter |
| gradient | no analytic gradient: central differences `(f(x+h e_i) - f(x-h e_i)) / (2h)` with h = ndeps = 1e-3 on the raw parameters. A non-finite difference raises an error. |
| convergence test | stop when `|f - f_min| <= reltol * (|f_min| + reltol)` after a line search (i.e. relative change of the objective below 1.49e-8), or when the step changes no coordinate (`reltest + x == reltest + x_old`, reltest = 10) |
| line search | backtracking: start at step 1, accept if `f` is finite and `f <= f_min + acctol * step * g'd` (acctol = 1e-4), else multiply step by stepredn = 0.2; a **non-finite objective (NaN/-Inf) at a trial point just shrinks the step**, it is not an error. If the search direction is not a descent direction the inverse-Hessian approximation is reset to I. |
| counts | `function` = objective evaluations in line searches; `gradient` = gradient evaluations (each costs 2*npar additional objective calls not included in `function`) |
| hessian | `optimHess`: for each i, the numerical gradient is evaluated at `par ± ndeps[i] e_i` and differenced, `H[i,j] = fnscale * (g_j(par+h e_i) - g_j(par-h e_i)) / (2h)`, then symmetrised `(H + H')/2`. It is the Hessian of the **log-likelihood** (fnscale multiplied back), so it is negative definite at the maximum. Cost: 2*npar gradients = 4*npar^2 objective calls. |
| RNG | not used by optim |

What this implies for comparing a Rust port:

* **Deterministic functions** (`parameterization.json`, `loglik.json`, `joint_loglik_at_*`,
  `T1_residuals`, `ks` statistic, evaluating the joint log-likelihood at the stored `par`):
  expect agreement to ~1e-12 relative or better (the only operations are small matrix products,
  acos/cos/sin, pnorm/dnorm, log). Exception: the `*_numerically_delicate` log-lik cases, where
  cancellation in `M_p` limits agreement to the accuracy of the port's `pnorm`/`dnorm` tails.
* **MLE `par`**: the stopping rule is a relative change of the objective below 1.49e-8, so the
  parameters are converged only to roughly `sqrt(1.5e-8 * |f| / curvature)`; with the Hessian
  eigenvalues here (5..300 for n=200) that is about 1e-4..1e-3 absolute. The stored `par` is one
  particular BFGS path; a different optimiser (or the same algorithm with a different gradient
  step) will land within ~1e-3 absolute per coefficient and within ~1e-6 absolute on `value`.
  A better test than comparing `par` directly: (a) evaluate the Rust joint log-likelihood at R's
  `par` and reproduce `value` to ~1e-10; (b) check that the Rust optimum is >= R's `value - 1e-6`;
  (c) check the Rust gradient at R's `par` is small (|g_i| of order 1e-3, consistent with the
  reltol/ndeps above).
* **Hessian**: finite-difference with h = 1e-3 on the gradient, which is itself a finite difference,
  so expect only ~1e-4..1e-5 relative agreement with an analytic or more accurate numerical Hessian
  (truncation error O(h^2)); the standard errors `sqrt(diag(solve(-H)))` agree to about 3-4 digits.
* The Rust optimiser must tolerate NaN / -Inf objective values at trial points during the line
  search (treat as "reject step"), otherwise it will abort on the n=1000 and d=4 fits.
