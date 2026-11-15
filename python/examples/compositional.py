"""Compositional time series on the sphere: ESAG regression on square-root
compositions (Yu & Huang's map to the positive orthant), on the direction of
compositional change (signed: a part that loses share points the negative
way), and on signed compositions (net flows) over the whole sphere.

Truth: a logistic-normal VAR(1) in ilr coordinates, z_t = c + A (z_{t-1} - c)
+ e_t, p_t = ilr^-1(z_t), D = 5 parts (one of them rare, 3%).

* ``levels``: forecast p_t from the past. The logistic-normal VAR fitted by
  least squares on z (the true family) against sqrt-ESAG (y_t = sqrt(p_t) on
  the ilr lag, or on sqrt(p_{t-1})). Scores: Aitchison distance and the
  angle between sqrt(p_hat) and sqrt(p_t) (Hellinger angle) of the point
  forecasts, and the log score on the simplex (Lebesgue measure on
  p_1..p_{D-1}): the logistic normal with its ilr Jacobian, sqrt-ESAG folded
  over the 2^D orthants with the Jacobian 2^-(D-1) prod p_i^-1/2.
* ``zeros``: the same compositions observed through multinomial counts (40
  per step, so the rare part is often 0): the logistic normal needs a zero
  replacement (+0.5 counts), sqrt-ESAG takes the zeros as they are; both
  forecast the latent composition.
* ``change``: the direction of compositional change v_t = dz_t / |dz_t| on
  S^{D-2} (Aitchison geometry: in clr coordinates the parts that gain share
  are positive, those that lose it negative), exactly PN((A - I)(z_{t-1} - c),
  S) under the truth. Scores as in ``ts_directions.py``, plus the share of
  parts whose gain or loss is forecast with the right sign.
* ``flows``: signed compositions w_t = f_t / |f_t|_1 of net flows f_t (some
  parts in, some out), on the whole sphere either as the direction
  f_t / |f_t|_2 or by the signed square root sign(w) sqrt(|w|) (Yu & Huang's
  map with the sign kept: the positive orthant is their case). Flows are a
  Gaussian VAR (``flows``) or differences of Poisson counts with
  heterogeneous sizes (``counts``), also with part scales (``scale="tyler"``
  and ``"auto"``: ESAG for the standardized directions, which the count
  flows need). Scores: L1 error of the forecast signed composition and the
  share of signs right.

``python examples/compositional.py [levels zeros change flows counts]``
env: ``R`` replicates (10), ``N`` training length (2000).
"""
import itertools
import os
import sys
import time

import numpy as np

from _pn import (Forecast, PNRegression, esag_fit, esag_forecast, helmert, ilr, ilr_inv, noise_cov,
                 ols_predict, ols_var, pn_forecast, pn_logpdf, print_table, random_rotation,
                 scores, summarize, unit)

R = int(os.environ.get("R", 10))
N = int(os.environ.get("N", 2000))
NT = 1000
BURN = 300
SHAPE = os.environ.get("SHAPE", "gamma-rot")
DP = 5
H = helmert(DP)
P0 = np.array([0.40, 0.30, 0.18, 0.09, 0.03])
SIGNS = np.array(list(itertools.product([1.0, -1.0], repeat=DP)))


def var_matrix(d, singular, rng):
    return random_rotation(d, rng) @ np.diag(singular) @ random_rotation(d, rng).T


def truth():
    rng = np.random.default_rng(21)
    a = var_matrix(DP - 1, [0.9, 0.7, 0.5, 0.3], rng)
    s = noise_cov(DP - 1, [0.20, 0.10, 0.05, 0.025], rng)
    return ilr(P0, H), a, s


def simulate_z(rep, n):
    c, a, s = truth()
    rng = np.random.default_rng(500 + rep)
    l = np.linalg.cholesky(s)
    total = BURN + n + NT + 1
    z = np.tile(c, (total, 1))
    for t in range(1, total):
        z[t] = c + a @ (z[t - 1] - c) + l @ rng.standard_normal(DP - 1)
    return z[BURN:], rng


def ilr_jacobian_logdet(p):
    """log |det d ilr / d (p_1..p_{D-1})| per row."""
    n = len(p)
    m = np.zeros((n, DP, DP - 1))
    idx = np.arange(DP - 1)
    m[:, idx, idx] = 1.0 / p[:, :DP - 1]
    m[:, DP - 1, :] = -1.0 / p[:, DP - 1:DP]
    return np.linalg.slogdet(np.einsum("ij,njk->nik", H, m))[1]


def ln_logpdf(p, zhat, s):
    """Logistic-normal log density of p on the simplex."""
    z = ilr(p, H)
    r = z - zhat
    sr = np.linalg.solve(s, r.T).T
    _, logdet = np.linalg.slogdet(s)
    k = DP - 1
    return (-0.5 * k * np.log(2 * np.pi) - 0.5 * logdet - 0.5 * (r * sr).sum(1)
            + ilr_jacobian_logdet(p))


def folded_logpdf(p, mu, v):
    """sqrt-ESAG log density of p on the simplex, folded over the orthants."""
    n = len(p)
    y = np.sqrt(p)
    lp = np.empty((len(SIGNS), n))
    for i, sg in enumerate(SIGNS):
        lp[i] = pn_logpdf(y * sg, mu, v)
    top = lp.max(0)
    return (top + np.log(np.exp(lp - top).sum(0)) - (DP - 1) * np.log(2.0)
            - 0.5 * np.log(p).sum(1))


def comp_scores(phat, p, logscore=None):
    """Aitchison distance, Hellinger angle (degrees) and log score."""
    lr = np.log(phat) - np.log(phat).mean(1, keepdims=True)
    lp = np.log(p) - np.log(p).mean(1, keepdims=True)
    return {
        "aitchison": np.linalg.norm(lr - lp, axis=1).mean(),
        "hellinger°": np.degrees(np.arccos(np.clip(np.sqrt(phat * p).sum(1), -1, 1))).mean(),
        "logscore": np.nan if logscore is None else logscore.mean(),
    }


def sqrt_esag_point(res, x_new):
    eta = np.asarray(res.linear_predictor(x_new))
    m = unit(eta[:, :DP])
    return m * m, eta[:, :DP], np.asarray(res.covariance(x_new))


def levels(rep):
    c, a, s = truth()
    z, rng = simulate_z(rep, N)
    p = ilr_inv(z, H)
    tr, te = slice(1, N + 1), slice(N + 1, N + NT + 1)
    lag_z = np.vstack([np.full((1, DP - 1), np.nan), z[:-1]])
    lag_y = np.vstack([np.full((1, DP), np.nan), np.sqrt(p[:-1])])
    out = {}
    zo = c + (lag_z[te] - c) @ a.T
    out["oracle (logistic normal)"] = comp_scores(ilr_inv(zo, H), p[te], ln_logpdf(p[te], zo, s))
    coef, sh = ols_var(z[tr], lag_z[tr])
    zh = ols_predict(coef, lag_z[te])
    out["logistic-normal VAR (OLS)"] = comp_scores(ilr_inv(zh, H), p[te], ln_logpdf(p[te], zh, sh))
    y = np.sqrt(p)
    for name, lag in [("sqrt-ESAG, ilr lag", lag_z), ("sqrt-ESAG, sqrt(p) lag", lag_y)]:
        res = esag_fit(y[tr], lag[tr], SHAPE)
        ph, mu, v = sqrt_esag_point(res, lag[te])
        out[name] = comp_scores(ph, p[te], folded_logpdf(p[te], mu, v))
    out["persistence p_{t-1}"] = comp_scores(p[N:N + NT], p[te])
    return out


def zeros(rep):
    c, a, s = truth()
    z, rng = simulate_z(rep, N)
    p = ilr_inv(z, H)
    counts = np.array([rng.multinomial(40, q) for q in p])
    obs = counts / 40.0
    repl = (counts + 0.5) / (40.0 + 0.5 * DP)
    tr, te = slice(1, N + 1), slice(N + 1, N + NT + 1)
    out = {}
    lag_z = np.vstack([np.full((1, DP - 1), np.nan), z[:-1]])
    zo = c + (lag_z[te] - c) @ a.T
    out["oracle (latent z_{t-1})"] = comp_scores(ilr_inv(zo, H), p[te])
    zr = ilr(repl, H)
    lag_zr = np.vstack([np.full((1, DP - 1), np.nan), zr[:-1]])
    coef, _ = ols_var(zr[tr], lag_zr[tr])
    out["logistic-normal VAR (+0.5 counts)"] = comp_scores(
        ilr_inv(ols_predict(coef, lag_zr[te]), H), p[te])
    y = np.sqrt(obs)
    lag_y = np.vstack([np.full((1, DP), np.nan), y[:-1]])
    res = esag_fit(y[tr], lag_y[tr], SHAPE)
    out["sqrt-ESAG on the counts (zeros kept)"] = comp_scores(sqrt_esag_point(res, lag_y[te])[0],
                                                             p[te])
    res = esag_fit(y[tr], lag_zr[tr], SHAPE)
    out["sqrt-ESAG, ilr(+0.5) lag"] = comp_scores(sqrt_esag_point(res, lag_zr[te])[0], p[te])
    out["persistence (+0.5 counts)"] = comp_scores(repl[N:N + NT], p[te])
    out["zeros share in data"] = {"aitchison": np.nan, "hellinger°": np.nan,
                                  "logscore": (obs == 0).mean()}
    return out


def change(rep):
    c, a, s = truth()
    z, rng = simulate_z(rep, N + 1)
    dz = np.diff(z, axis=0, prepend=np.nan * z[:1])
    v = unit(dz)
    tr, te = slice(2, N + 2), slice(N + 2, N + NT + 2)
    nanrow = np.full((1, DP - 1), np.nan)
    lag_z = np.vstack([nanrow, z[:-1]])
    lag_v = np.vstack([nanrow, v[:-1]])
    k = DP - 1
    oracle = pn_forecast((lag_z[te] - c) @ (a - np.eye(k)).T, s, v[te], rng)
    out = {"oracle": oracle}
    res = esag_fit(v[tr], lag_z[tr], SHAPE)
    out["esag z_{t-1}"] = esag_forecast(res, lag_z[te], v[te], rng)
    res = esag_fit(v[tr], lag_z[tr], SHAPE, penalty="auto")
    out["esag z_{t-1}, auto"] = esag_forecast(res, lag_z[te], v[te], rng)
    res = esag_fit(v[tr], lag_v[tr], SHAPE)
    out["esag v_{t-1}"] = esag_forecast(res, lag_v[te], v[te], rng)
    out["PN-reg z_{t-1}"] = PNRegression(v[tr], lag_z[tr]).forecast(lag_z[te], v[te], rng)
    coef, sh = ols_var(z[tr], lag_z[tr])
    out["VAR-OLS on z"] = pn_forecast(ols_predict(coef, lag_z[te]) - lag_z[te], sh, v[te], rng)
    res = esag_fit(v[tr], None, SHAPE)
    out["esag i.i.d."] = esag_forecast(res, None, v[te], rng)
    out["persistence v_{t-1}"] = Forecast(lag_v[te])
    result = {}
    dclr = dz[te] @ H
    for name, f in out.items():
        sc = scores(f, v[te], oracle)
        sc["signs right"] = (np.sign(f.v @ H) == np.sign(dclr)).mean()
        result[name] = sc
    return result


def flows(rep, poisson):
    d = DP
    rng0 = np.random.default_rng(31)
    a = var_matrix(d, [0.8, 0.6, 0.5, 0.3, 0.2], rng0)
    s = noise_cov(d, [2.0, 1.0, 0.6, 0.4, 0.2], rng0)
    sizes = np.array([50.0, 20.0, 10.0, 5.0, 2.0])
    rng = np.random.default_rng(600 + rep)
    l = np.linalg.cholesky(s)
    total = BURN + N + NT + 1
    g = np.zeros((total, d))
    for t in range(1, total):
        g[t] = a @ g[t - 1] + l @ rng.standard_normal(d) * (0.3 if poisson else 1.0)
    if poisson:
        # net flows: inflow minus outflow counts, intensities sizes * exp(+-g)
        f = (rng.poisson(sizes * np.exp(g)) - rng.poisson(sizes * np.exp(-g))).astype(float)
        mean = sizes * (np.exp(g) - np.exp(-g))
    else:
        f = g.copy()
        mean = None
    f, g = f[BURN:], g[BURN:]
    keep = np.abs(f).sum(1) > 0
    f[~keep] = 1e-9
    w = f / np.abs(f).sum(1, keepdims=True)
    u = unit(f)
    sq = np.sign(w) * np.sqrt(np.abs(w))
    tr, te = slice(1, N + 1), slice(N + 1, N + NT + 1)
    lag_f = np.vstack([np.full((1, d), np.nan), f[:-1]])
    out = {}

    def sc(what, wh, sign_hat=None):
        sign_hat = np.sign(wh) if sign_hat is None else sign_hat
        out[what] = {"L1": np.abs(wh - w[te]).sum(1).mean(),
                     "signs right": (sign_hat == np.sign(w[te])).mean(),
                     "all signs": (sign_hat == np.sign(w[te])).all(1).mean()}

    res = esag_fit(u[tr], lag_f[tr], SHAPE)
    uh = unit(np.asarray(res.linear_predictor(lag_f[te]))[:, :d])
    sc("ESAG on f/|f|_2", uh / np.abs(uh).sum(1, keepdims=True))
    for scale in ("tyler", "auto"):
        res = esag_fit(u[tr], lag_f[tr], SHAPE, scale=scale)
        uh = np.asarray(res.predict(lag_f[te]))
        label = f"ESAG on f/|f|_2, scale={scale}"
        if scale == "auto":
            label += f" ({'scaled' if res.scale is not None else 'unscaled'})"
        sc(label, uh / np.abs(uh).sum(1, keepdims=True))
    res = esag_fit(sq[tr], lag_f[tr], SHAPE)
    sh = unit(np.asarray(res.linear_predictor(lag_f[te]))[:, :d])
    sc("ESAG on signed sqrt", np.sign(sh) * sh * sh)
    fc = PNRegression(u[tr], lag_f[tr]).forecast(lag_f[te], u[te], rng)
    sc("PN-reg on f/|f|_2", fc.v / np.abs(fc.v).sum(1, keepdims=True))
    fc = PNRegression(sq[tr], lag_f[tr]).forecast(lag_f[te], sq[te], rng)
    sc("PN-reg on signed sqrt", np.sign(fc.v) * fc.v * fc.v)
    coef, _ = ols_var(f[tr], lag_f[tr])
    fh = ols_predict(coef, lag_f[te])
    sc("VAR-OLS on f", fh / np.abs(fh).sum(1, keepdims=True))
    if poisson:
        # (infeasible: the intensities of step t itself)
        mo = mean[BURN:][te]
        sc("oracle (true intensities at t)", mo / np.abs(mo).sum(1, keepdims=True))
    else:
        mo = lag_f[te] @ a.T
        sc("oracle (conditional mean)", mo / np.abs(mo).sum(1, keepdims=True))
    sc("persistence w_{t-1}", w[N:N + NT])
    return out


def run(name, fn):
    rows = {}
    for rep in range(R):
        for k, v in fn(rep).items():
            rows.setdefault(k, []).append(v)
    return rows


def show(title, rows, keys):
    print(f"\n{title} ({R} replicates, n = {N})")
    print("  " + f"{'model':<38}" + "".join(f"{k:>13}" for k in keys))
    for name, vals in rows.items():
        cells = []
        for k in keys:
            x = np.array([v.get(k, np.nan) for v in vals], dtype=float)
            x = x[~np.isnan(x)]
            cells.append(f"{x.mean():13.3f}" if len(x) else f"{'':>13}")
        print(f"  {name:<38}" + "".join(cells))


def main():
    which = sys.argv[1:] or ["levels", "zeros", "change", "flows", "counts"]
    for name in which:
        t0 = time.perf_counter()
        if name == "levels":
            show("Composition levels: logistic normal vs sqrt-ESAG (D = 5)", run(name, levels),
                 ["aitchison", "hellinger°", "logscore"])
        if name == "zeros":
            show("Composition levels from counts of 40 (zeros): forecasts of the latent p_t",
                 run(name, zeros), ["aitchison", "hellinger°", "logscore"])
            print("  (the last row's log-score column is the share of zero parts in the data)")
        if name == "change":
            rows = run(name, change)
            results = {k: summarize(v) for k, v in rows.items()}
            print_table(f"Direction of compositional change on S^{DP - 2} ({R} replicates, n = {N})",
                        results)
            print("  share of parts whose share change is forecast with the right sign:")
            for k, v in rows.items():
                print(f"    {k:<26} {np.mean([x['signs right'] for x in v]):.3f}")
        if name in ("flows", "counts"):
            show(f"Signed compositions of net flows ({'Poisson counts' if name == 'counts' else 'Gaussian VAR'})",
                 run(name, lambda r: flows(r, name == "counts")), ["L1", "signs right", "all signs"])
        print(f"  [{time.perf_counter() - t0:.0f} s]")


if __name__ == "__main__":
    main()
