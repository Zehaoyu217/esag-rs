"""Forecasting the direction of a multivariate Gaussian time series, and how
concentrated it is, with ESAG regression: u_t = x_t / |x_t| on lagged
covariates, one step ahead, out of sample.

If x_t | past ~ N(m_t, S_t), the direction u_t is exactly projected normal
PN(m_t, S_t); ESAG is the projected normal with V mu = mu and det V = 1, so a
VAR with isotropic noise has ESAG directions (mu = A x_{t-1}) and anisotropic
noise makes ESAG an approximation. Scenarios (d = 5 unless noted):

* ``iso``: stationary VAR(1), x_t = A x_{t-1} + e_t, e ~ N(0, I);
* ``aniso``: the same with anisotropic noise (eigenvalues 4, 2, 1, .5, .25);
* ``sv``: anisotropic VAR(1) with stochastic volatility, x_t = A x_{t-1} +
  s_t e_t, log s_t AR(1) (0.97, 0.2): the direction is scale-free, the
  concentration |A x_{t-1}| / s_t is not;
* ``unitroot``: an integrated series, dx_t = delta + B dx_{t-1} + e_t: the
  direction of the levels against the direction of the changes;
* ``sparse10``: d = 10, only the lags of x1..x4 matter, n = 3000: the
  unpenalized ESAG against ``fit_regularized()`` ("auto").

Models: the oracle (true predictive PN; in ``sv`` it also knows s_t), ESAG on
the lagged vector (``esag x_{t-1}``), ESAG on the lagged direction only
(``esag u_{t-1}``), ESAG on the volatility-standardized lag (``sv``), the
Gaussian VAR fitted by least squares with its predictive PN (``VAR-OLS``;
raw and standardized), i.i.d. ESAG, and persistence (u_{t-1}).

Scores (``_pn.scores``): mean cosine to the realized direction, mean angle to
the oracle's mean direction, mean log score and its gap to the oracle, and
for the concentration (mean resultant length rho) its error against the
oracle's, their correlation, and calibration (mean rho minus realized mean
cosine).

``python examples/ts_directions.py [iso aniso sv unitroot sparse10]``
env: ``R`` replicates (10; 4 for sparse10), ``N`` training length (2000).
"""
import os
import sys
import time

import numpy as np

import esag
from _pn import (Forecast, PNRegression, esag_fit, esag_forecast, noise_cov, ols_predict,
                 ols_var, pn_forecast, print_table, random_rotation, scores, summarize, unit)

R = int(os.environ.get("R", 10))
N = int(os.environ.get("N", 2000))
NT = 1000
BURN = 300
SHAPE = os.environ.get("SHAPE", "gamma-rot")


def var_matrix(d, singular, rng):
    return random_rotation(d, rng) @ np.diag(singular) @ random_rotation(d, rng).T


def split(n_total):
    """Row ranges of the training and test targets (t = 1..)."""
    return slice(1, N + 1), slice(N + 1, N + NT + 1)


def common_models(u, lag_x, lag_u, tr, te, oracle, rng, x_target=None):
    """The models every scenario fits, as forecasts on the test rows;
    ``x_target``: the vectors for VAR-OLS (default: none)."""
    out = {"oracle": oracle}
    res = esag_fit(u[tr], lag_x[tr], SHAPE)
    out["esag x_{t-1}"] = esag_forecast(res, lag_x[te], u[te], rng)
    res = esag_fit(u[tr], lag_x[tr], SHAPE, penalty="auto")
    out["esag x_{t-1}, auto"] = esag_forecast(res, lag_x[te], u[te], rng)
    res = esag_fit(u[tr], lag_u[tr], SHAPE)
    out["esag u_{t-1}"] = esag_forecast(res, lag_u[te], u[te], rng)
    out["PN-reg x_{t-1}"] = PNRegression(u[tr], lag_x[tr]).forecast(lag_x[te], u[te], rng)
    if x_target is not None:
        coef, s = ols_var(x_target[tr], lag_x[tr])
        out["VAR-OLS"] = pn_forecast(ols_predict(coef, lag_x[te]), s, u[te], rng)
    res = esag_fit(u[tr], None, SHAPE)
    out["esag i.i.d."] = esag_forecast(res, None, u[te], rng)
    out["persistence u_{t-1}"] = Forecast(lag_u[te])
    return out


def scored(out, u, oracle):
    return {k: scores(f, u, oracle) for k, f in out.items()}


def stationary(rep, aniso):
    d = 5
    truth = np.random.default_rng(7)
    a = var_matrix(d, [0.9, 0.8, 0.6, 0.4, 0.2], truth)
    s = noise_cov(d, [4, 2, 1, 0.5, 0.25], truth) if aniso else np.eye(d)
    rng = np.random.default_rng(100 + rep)
    l = np.linalg.cholesky(s)
    total = BURN + N + NT + 1
    x = np.zeros((total, d))
    for t in range(1, total):
        x[t] = a @ x[t - 1] + l @ rng.standard_normal(d)
    x = x[BURN:]
    u = unit(x)
    lag_x = np.vstack([np.full(d, np.nan), x[:-1]])
    lag_u = np.vstack([np.full(d, np.nan), u[:-1]])
    tr, te = split(len(x))
    oracle = pn_forecast(lag_x[te] @ a.T, s, u[te], rng)
    out = common_models(u, lag_x, lag_u, tr, te, oracle, rng, x_target=x)
    return scored(out, u[te], oracle)


def stochastic_vol(rep):
    d = 5
    truth = np.random.default_rng(7)
    a = var_matrix(d, [0.9, 0.8, 0.6, 0.4, 0.2], truth)
    s = noise_cov(d, [4, 2, 1, 0.5, 0.25], truth)
    rng = np.random.default_rng(200 + rep)
    l = np.linalg.cholesky(s)
    total = BURN + N + NT + 1
    x = np.zeros((total, d))
    logsig = np.zeros(total)
    for t in range(1, total):
        logsig[t] = 0.97 * logsig[t - 1] + 0.2 * rng.standard_normal()
        x[t] = a @ x[t - 1] + np.exp(logsig[t]) * (l @ rng.standard_normal(d))
    # EWMA volatility (RiskMetrics, 0.94), known at t - 1
    ew = np.empty(total)
    ew[0] = (x[1:51] ** 2).mean()
    for t in range(1, total):
        ew[t] = 0.94 * ew[t - 1] + 0.06 * (x[t] ** 2).mean()
    sig_hat = np.sqrt(ew)
    x, logsig, sig_hat = x[BURN:], logsig[BURN:], sig_hat[BURN:]
    u = unit(x)
    lag_x = np.vstack([np.full(d, np.nan), x[:-1]])
    lag_u = np.vstack([np.full(d, np.nan), u[:-1]])
    lag_std = np.vstack([np.full(d, np.nan), x[:-1] / sig_hat[:-1, None]])
    tr, te = split(len(x))
    sig = np.exp(logsig[te])
    oracle = pn_forecast(lag_x[te] @ a.T, s[None] * sig[:, None, None] ** 2, u[te], rng)
    out = common_models(u, lag_x, lag_u, tr, te, oracle, rng, x_target=x)
    res = esag_fit(u[tr], lag_std[tr], SHAPE)
    out["esag x_{t-1}/sig_{t-1}"] = esag_forecast(res, lag_std[te], u[te], rng)
    res = esag_fit(u[tr], lag_std[tr], SHAPE, penalty="auto")
    out["esag x_{t-1}/sig_{t-1}, auto"] = esag_forecast(res, lag_std[te], u[te], rng)
    out["PN-reg x_{t-1}/sig_{t-1}"] = PNRegression(u[tr], lag_std[tr]).forecast(
        lag_std[te], u[te], rng)
    out["PN-reg u_{t-1}"] = PNRegression(u[tr], lag_u[tr]).forecast(lag_u[te], u[te], rng)
    target_std = np.vstack([np.full(d, np.nan), x[1:] / sig_hat[:-1, None]])
    coef, sh = ols_var(target_std[tr], lag_std[tr])
    out["VAR-OLS standardized"] = pn_forecast(ols_predict(coef, lag_std[te]), sh, u[te], rng)
    return scored(out, u[te], oracle)


def unit_root(rep):
    """Levels and changes of dx_t = delta + B dx_{t-1} + e_t (isotropic noise:
    both directions are exactly ESAG given the right covariates)."""
    d = 5
    truth = np.random.default_rng(9)
    b = 0.5 * random_rotation(d, truth)
    delta = 0.3 * unit(truth.standard_normal(d))
    rng = np.random.default_rng(300 + rep)
    total = BURN + N + NT + 1
    dx = np.zeros((total, d))
    for t in range(1, total):
        dx[t] = delta + b @ dx[t - 1] + rng.standard_normal(d)
    x = np.cumsum(dx, axis=0)
    x, dx = x[BURN:], dx[BURN:]
    tr, te = split(len(x))
    nanrow = np.full((1, d), np.nan)
    lag_x = np.vstack([nanrow, x[:-1]])
    lag_dx = np.vstack([nanrow, dx[:-1]])
    out = {}
    # the direction of the levels
    ul = unit(x)
    lag_ul = np.vstack([nanrow, ul[:-1]])
    cov_l = np.column_stack([lag_x, lag_dx])
    oracle = pn_forecast(lag_x[te] + delta + lag_dx[te] @ b.T, np.eye(d), ul[te], rng)
    res = esag_fit(ul[tr], cov_l[tr], SHAPE)
    out["levels: esag x_{t-1}, dx_{t-1}"] = esag_forecast(res, cov_l[te], ul[te], rng)
    out["levels: persistence"] = Forecast(lag_ul[te])
    out["levels: oracle"] = oracle
    levels = scored(out, ul[te], oracle)
    # the direction of the changes
    out = {}
    ud = unit(dx)
    lag_ud = np.vstack([nanrow, ud[:-1]])
    oracle = pn_forecast(delta + lag_dx[te] @ b.T, np.eye(d), ud[te], rng)
    res = esag_fit(ud[tr], lag_dx[tr], SHAPE)
    out["changes: esag dx_{t-1}"] = esag_forecast(res, lag_dx[te], ud[te], rng)
    res = esag_fit(ud[tr], lag_dx[tr], SHAPE, penalty="auto")
    out["changes: esag dx_{t-1}, auto"] = esag_forecast(res, lag_dx[te], ud[te], rng)
    out["changes: PN-reg dx_{t-1}"] = PNRegression(ud[tr], lag_dx[tr]).forecast(
        lag_dx[te], ud[te], rng)
    res = esag_fit(ud[tr], lag_ud[tr], SHAPE)
    out["changes: esag u_{t-1}"] = esag_forecast(res, lag_ud[te], ud[te], rng)
    res = esag_fit(ud[tr], lag_x[tr], SHAPE)
    out["changes: esag x_{t-1} (levels)"] = esag_forecast(res, lag_x[te], ud[te], rng)
    coef, s = ols_var(dx[tr], lag_dx[tr])
    out["changes: VAR-OLS on dx"] = pn_forecast(ols_predict(coef, lag_dx[te]), s, ud[te], rng)
    res = esag_fit(ud[tr], None, SHAPE)
    out["changes: esag i.i.d. (drift)"] = esag_forecast(res, None, ud[te], rng)
    out["changes: persistence"] = Forecast(lag_ud[te])
    out["changes: oracle"] = oracle
    changes = scored(out, ud[te], oracle)
    return {**levels, **changes}


SPARSE_SELECTION = []


def sparse10(rep):
    d, n = 10, max(N, 3000)
    truth = np.random.default_rng(11)
    a = np.zeros((d, d))
    a[:, :4] = truth.standard_normal((d, 4))
    a *= 0.85 / np.linalg.norm(a, 2)
    rng = np.random.default_rng(400 + rep)
    total = BURN + n + NT + 1
    x = np.zeros((total, d))
    for t in range(1, total):
        x[t] = a @ x[t - 1] + rng.standard_normal(d)
    x = x[BURN:]
    u = unit(x)
    nanrow = np.full((1, d), np.nan)
    lag_x = np.vstack([nanrow, x[:-1]])
    lag_u = np.vstack([nanrow, u[:-1]])
    tr, te = slice(1, n + 1), slice(n + 1, n + NT + 1)
    oracle = pn_forecast(lag_x[te] @ a.T, np.eye(d), u[te], rng)
    out = {"oracle": oracle}
    t0 = time.perf_counter()
    res = esag_fit(u[tr], lag_x[tr], SHAPE)
    t_full = time.perf_counter() - t0
    out["esag x_{t-1}"] = esag_forecast(res, lag_x[te], u[te], rng)
    t0 = time.perf_counter()
    reg = esag.ESAG(u[tr], lag_x[tr], shape=SHAPE).fit_regularized()
    t_auto = time.perf_counter() - t0
    out["esag x_{t-1}, auto"] = esag_forecast(reg.relaxed, lag_x[te], u[te], rng)
    t0 = time.perf_counter()
    pnr = PNRegression(u[tr], lag_x[tr])
    t_pn = time.perf_counter() - t0
    out["PN-reg x_{t-1}"] = pnr.forecast(lag_x[te], u[te], rng)
    coef, s = ols_var(x[tr], lag_x[tr])
    out["VAR-OLS"] = pn_forecast(ols_predict(coef, lag_x[te]), s, u[te], rng)
    res = esag_fit(u[tr], None, SHAPE)
    out["esag i.i.d."] = esag_forecast(res, None, u[te], rng)
    out["persistence u_{t-1}"] = Forecast(lag_u[te])
    # which lags the penalty kept, for the mean and for the shape
    p = reg.params
    rows = [r for r in p.index if r != "const"]
    mean_cols = [c for c in p.columns if c.startswith("mu[")]
    shape_cols = [c for c in p.columns if not c.startswith("mu[")]
    kept_mu = [r for r in rows if (p.loc[r, mean_cols] != 0).any()]
    kept_shape = [r for r in rows if (p.loc[r, shape_cols] != 0).any()]
    SPARSE_SELECTION.append((kept_mu, kept_shape, t_full, t_auto, reg.method, t_pn))
    return scored(out, u[te], oracle)


SCENARIOS = {
    "iso": ("Stationary VAR(1), isotropic noise (ESAG is the exact model), d = 5",
            lambda r: stationary(r, False), R),
    "aniso": ("Stationary VAR(1), anisotropic noise (ESAG approximates PN), d = 5",
              lambda r: stationary(r, True), R),
    "sv": ("VAR(1) with stochastic volatility (oracle knows s_t), d = 5", stochastic_vol, R),
    "unitroot": ("Integrated series: directions of the levels and of the changes, d = 5",
                 unit_root, R),
    "sparse10": ("d = 10, only the lags of x1..x4 matter, n = 3000", sparse10,
                 int(os.environ.get("R", 4))),
}


def main():
    which = sys.argv[1:] or list(SCENARIOS)
    for name in which:
        title, fn, reps = SCENARIOS[name]
        t0 = time.perf_counter()
        rows = {}
        for rep in range(reps):
            for k, v in fn(rep).items():
                rows.setdefault(k, []).append(v)
        results = {k: summarize(v) for k, v in rows.items()}
        if name == "unitroot":
            for part in ("levels", "changes"):
                sub = {k.split(": ", 1)[1]: v for k, v in results.items() if k.startswith(part)}
                print_table(f"{title} -- {part} ({reps} replicates, n = {N})", sub)
        else:
            print_table(f"{title} ({reps} replicates, n = {N if name != 'sparse10' else max(N, 3000)})",
                        results)
        if name == "sparse10":
            for kept_mu, kept_shape, t_full, t_auto, method, t_pn in SPARSE_SELECTION:
                print(f"    {method}: mean kept {kept_mu}, shape kept {kept_shape}; "
                      f"ESAG fit {t_full:.1f} s, penalized path {t_auto:.1f} s, PN-reg {t_pn:.1f} s")
        print(f"  [{time.perf_counter() - t0:.0f} s]")


if __name__ == "__main__":
    main()
