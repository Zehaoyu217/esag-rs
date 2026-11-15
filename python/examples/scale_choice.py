"""Which part-scale choice is reliable? ``esag.ESAG(Y, X, scale=...)`` with
``None``, ``"tyler"``, ``"auto"``, and (where the raw vectors exist) the raw
vectors' standard deviations, on designs where scaling should help and on
designs where it should not, out of sample.

Designs (d = 5, 1000 test rows, 10 replicates; n training rows below):

* ``equal``: VAR(1) with isotropic noise, parts of equal scale (no scaling
  needed);
* ``unequal``: the same VAR seen through part scales 8, 3, 1.5, 0.8, 0.4;
* ``flows``: net Poisson flows of parts of sizes 50 ... 2 (the case that broke
  plain ESAG);
* ``returns``: weakly predictable returns with a common stochastic volatility
  and per-asset volatilities 3, 2, 1, 1, 0.5, on the lagged direction;
* ``aniso``: VAR(1) with anisotropic noise on random axes (diagonal scales
  are only part of the story);
* ``concentrated``: the hard regression scenario (directions concentrated
  around a mean of norm 2.5, five correlated covariates);
* ``composition``: square-root compositions of a logistic-normal VAR with a
  rare part (3%), on the ilr lag (directions in the positive orthant).

Scores: the mean test log score on the original sphere (densities of the
scaled models include the Jacobian, so every option is scored on the same
measure), the mean cosine between the predicted and the realized direction,
and how often ``"auto"`` chose the scales. The two can disagree: an
unscaled fit whose mean direction has turned away from the data can still
describe the spread well. The summary gives each option's shortfall
against the best option of each design (its regret), in log score and in
cosine: a reliable choice has a small worst-case regret.

``python examples/scale_choice.py [design ...]``; env ``R`` (10), ``NS``
(training sizes, default ``300,2000``).
"""
import os
import sys
import time

import numpy as np

import esag
from _pn import helmert, ilr, ilr_inv, noise_cov, random_rotation, unit

R = int(os.environ.get("R", 10))
NS = [int(v) for v in os.environ.get("NS", "300,2000").split(",")]
NT = 1000
BURN = 300
D = 5


def var_matrix(d, singular, rng):
    return random_rotation(d, rng) @ np.diag(singular) @ random_rotation(d, rng).T


def series(n, rep, step, seed):
    """Rows 0..n+NT of a simulated series (after burn-in)."""
    rng = np.random.default_rng(seed + rep)
    state = None
    rows = []
    for t in range(BURN + n + NT + 1):
        state = step(state, rng)
        if t >= BURN:
            rows.append(state[0] if isinstance(state, tuple) else state)
    return np.array(rows)


def lagged(v, n):
    """(train y, train x, test y, test x, raw train targets) of directions of
    v_t on lags v_{t-1}."""
    y, x = unit(v[1:]), v[:-1]
    return y[:n], x[:n], y[n:n + NT], x[n:n + NT], v[1:n + 1]


def design_equal(n, rep):
    a = var_matrix(D, [0.8, 0.7, 0.6, 0.5, 0.4], np.random.default_rng(41))

    def step(s, rng):
        s = np.zeros(D) if s is None else s
        return a @ s + rng.standard_normal(D)
    return lagged(series(n, rep, step, 1000), n)


def design_unequal(n, rep):
    v = design_equal_raw(n, rep) * np.array([8.0, 3.0, 1.5, 0.8, 0.4])
    return lagged(v, n)


def design_equal_raw(n, rep):
    a = var_matrix(D, [0.8, 0.7, 0.6, 0.5, 0.4], np.random.default_rng(41))

    def step(s, rng):
        s = np.zeros(D) if s is None else s
        return a @ s + rng.standard_normal(D)
    return series(n, rep, step, 2000)


def design_flows(n, rep):
    rng0 = np.random.default_rng(31)
    a = var_matrix(D, [0.8, 0.6, 0.5, 0.3, 0.2], rng0)
    l = np.linalg.cholesky(noise_cov(D, [2.0, 1.0, 0.6, 0.4, 0.2], rng0))
    sizes = np.array([50.0, 20.0, 10.0, 5.0, 2.0])
    rng = np.random.default_rng(3000 + rep)
    g = np.zeros(D)
    f = []
    for t in range(BURN + n + NT + 1):
        g = a @ g + 0.3 * (l @ rng.standard_normal(D))
        if t >= BURN:
            f.append(rng.poisson(sizes * np.exp(g)) - rng.poisson(sizes * np.exp(-g)))
    f = np.array(f, dtype=float)
    f[np.abs(f).sum(1) == 0] = 1e-9
    return lagged(f, n)


def design_returns(n, rep):
    rng0 = np.random.default_rng(51)
    b = var_matrix(D, [0.3, 0.2, 0.15, 0.1, 0.05], rng0)
    drift = 0.15 * unit(rng0.standard_normal(D))
    vols = np.array([3.0, 2.0, 1.0, 1.0, 0.5])
    rng = np.random.default_rng(4000 + rep)
    z = np.zeros(D)
    logsig = 0.0
    r = []
    for t in range(BURN + n + NT + 1):
        logsig = 0.97 * logsig + 0.2 * rng.standard_normal()
        z = drift + b @ z + rng.standard_normal(D)
        if t >= BURN:
            r.append(np.exp(logsig) * vols * z)
    r = np.array(r)
    u = unit(r)
    # the lagged direction (scale-free) as the covariate
    y, x = u[1:], u[:-1]
    return y[:n], x[:n], y[n:n + NT], x[n:n + NT], r[1:n + 1]


def design_aniso(n, rep):
    rng0 = np.random.default_rng(7)
    a = var_matrix(D, [0.9, 0.8, 0.6, 0.4, 0.2], rng0)
    l = np.linalg.cholesky(noise_cov(D, [4, 2, 1, 0.5, 0.25], rng0))

    def step(s, rng):
        s = np.zeros(D) if s is None else s
        return a @ s + l @ rng.standard_normal(D)
    return lagged(series(n, rep, step, 5000), n)


def design_concentrated(n, rep):
    rng0 = np.random.default_rng(61)
    g = D * (D - 1) // 2 - 1
    center = unit(np.ones(D) + 0.3 * rng0.standard_normal(D))
    mu_b = np.zeros((D, 6))
    mu_b[:, 0] = 2.5 * center
    for j, size in [(1, 0.6), (2, 0.45), (4, 0.15)]:
        mu_b[:, j] = size * unit(rng0.standard_normal(D))
    ga_b = np.zeros((g, 6))
    for j, size in [(0, 1.0), (1, 0.45), (3, 0.45), (4, 0.15)]:
        ga_b[:, j] = size * unit(rng0.standard_normal(g))
    rng = np.random.default_rng(6000 + rep)
    m = n + NT
    z0 = rng.standard_normal((m, 1))
    x = np.sqrt(0.5) * z0 + np.sqrt(0.5) * rng.standard_normal((m, 5))
    zz = np.column_stack([np.ones(m), x])
    mus, gas = zz @ mu_b.T, zz @ ga_b.T
    y = np.array([esag.Distribution(mu=mus[i], gamma=gas[i]).sample(1, seed=int(rng.integers(2**31)))[0]
                  for i in range(m)])
    return y[:n], x[:n], y[n:], x[n:], None


def design_composition(n, rep):
    h = helmert(D)
    rng0 = np.random.default_rng(21)
    a = var_matrix(D - 1, [0.9, 0.7, 0.5, 0.3], rng0)
    l = np.linalg.cholesky(noise_cov(D - 1, [0.20, 0.10, 0.05, 0.025], rng0))
    c = ilr(np.array([0.40, 0.30, 0.18, 0.09, 0.03]), h)
    rng = np.random.default_rng(7000 + rep)
    z = c.copy()
    zs = []
    for t in range(BURN + n + NT + 1):
        z = c + a @ (z - c) + l @ rng.standard_normal(D - 1)
        if t >= BURN:
            zs.append(z)
    zs = np.array(zs)
    y = np.sqrt(ilr_inv(zs, h))
    yy, xx = y[1:], zs[:-1]
    return yy[:n], xx[:n], yy[n:n + NT], xx[n:n + NT], None


DESIGNS = {
    "equal": ("equal part scales (VAR, isotropic noise)", design_equal),
    "unequal": ("part scales 8, 3, 1.5, 0.8, 0.4 (VAR)", design_unequal),
    "flows": ("net Poisson flows, part sizes 50 ... 2", design_flows),
    "returns": ("returns: common SV, asset volatilities 3 ... 0.5, lagged direction", design_returns),
    "aniso": ("VAR with anisotropic noise on random axes", design_aniso),
    "concentrated": ("concentrated regression (the hard scenario)", design_concentrated),
    "composition": ("square-root compositions with a rare part (3%)", design_composition),
}


def score(res, yt, xt):
    xn = res.model._exog_new(xt)
    ll = np.asarray(res._native.loglike_obs(xn, yt))
    cos = (np.asarray(res.predict(xt)) * yt).sum(1).mean()
    return ll.mean(), cos


def run_design(key, n):
    title, make = DESIGNS[key]
    rows = {}
    picks = 0
    for rep in range(R):
        y, x, yt, xt, raw = make(n, rep)
        options = [("none", None), ("tyler", "tyler"), ("auto", "auto")]
        if raw is not None:
            options.append(("raw sd", raw.std(0)))
        for name, scale in options:
            try:
                res = esag.ESAG(y, x, scale=scale).fit()
                ls, cos = score(res, yt, xt)
                if name == "auto" and res.scale is not None:
                    picks += 1
            except Exception as e:  # a failed fit counts as a failure
                print(f"    [{key} n={n} rep {rep} {name}: {e}]", flush=True)
                ls, cos = np.nan, np.nan
            rows.setdefault(name, []).append((ls, cos))
    return title, rows, picks


def main():
    which = sys.argv[1:] or list(DESIGNS)
    regret = {}
    for n in NS:
        for key in which:
            t0 = time.perf_counter()
            title, rows, picks = run_design(key, n)
            ls = {k: np.array([r[0] for r in v]) for k, v in rows.items()}
            cs = {k: np.array([r[1] for r in v]) for k, v in rows.items()}
            best = np.nanmax(np.vstack([ls[k] for k in ls]), axis=0)
            best_c = np.nanmax(np.vstack([cs[k] for k in cs]), axis=0)
            print(f"\n{title}, n = {n} ({R} replicates; auto chose the scales {picks}/{R})")
            print(f"  {'option':<8} {'log score':>10} {'(se)':>7} {'regret':>8} {'max regret':>11} {'cos':>7}")
            for k in ls:
                se = np.nanstd(ls[k], ddof=1) / np.sqrt(np.sum(~np.isnan(ls[k])))
                rg = best - ls[k]
                cos = np.nanmean([r[1] for r in rows[k]])
                print(f"  {k:<8} {np.nanmean(ls[k]):10.3f} {se:7.3f} {np.nanmean(rg):8.3f} "
                      f"{np.nanmax(rg):11.3f} {cos:7.3f}", flush=True)
                regret.setdefault(k, []).append((f"{key} n={n}", np.nanmean(rg), np.nanmax(rg),
                                                 np.nanmean(best_c - cs[k])))
            print(f"  [{time.perf_counter() - t0:.0f} s]", flush=True)
    for what, idx in (("log score per observation", 1), ("mean cosine", 3)):
        print(f"\nRegret in {what} against the best option of each design")
        print(f"  {'option':<8} {'mean':>7} {'worst design (mean regret)':>40}")
        for k, v in regret.items():
            worst = max(v, key=lambda t: t[idx])
            print(f"  {k:<8} {np.mean([t[idx] for t in v]):7.3f}   {worst[0]:>28} {worst[idx]:7.3f}"
                  f"   (designs scored: {len(v)})")


if __name__ == "__main__":
    main()
