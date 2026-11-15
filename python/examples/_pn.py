"""Shared tools for the time-series and compositional experiments
(``ts_directions.py``, ``compositional.py``): the general projected normal
PN(m, S) (x ~ N(m, S), u = x / |x|) on S^{d-1}, which covers ESAG (S = V with
V mu = mu, det V = 1), the predictive distributions of Gaussian VARs and the
oracles; Monte Carlo mean directions and mean resultant lengths; forecast
scores; ilr coordinates."""
import numpy as np

import esag

# Gauss-Legendre nodes for log I_k(alpha) = log int_0^inf s^k exp(-s^2/2 + alpha s) ds
_GL_X, _GL_W = np.polynomial.legendre.leggauss(160)


def log_ik(k: int, alpha: np.ndarray) -> np.ndarray:
    """log int_0^inf s^k exp(-s^2/2 + alpha s) ds (k >= 1), by Gauss-Legendre on a
    window of +-40 widths around the mode (the integrand is log-concave; for
    alpha << 0 it is gamma-like, with a tail longer than its width)."""
    a = np.asarray(alpha, dtype=float)
    mode = 0.5 * (a + np.sqrt(a * a + 4.0 * k))
    sd = 1.0 / np.sqrt(1.0 + k / mode**2)
    lo = np.maximum(0.0, mode - 40.0 * sd)
    hi = mode + 40.0 * sd
    half = 0.5 * (hi - lo)
    s = lo[..., None] + half[..., None] * (_GL_X + 1.0)
    logf = k * np.log(s) - 0.5 * s * s + a[..., None] * s
    top = logf.max(axis=-1, keepdims=True)
    return np.log(half) + top[..., 0] + np.log((np.exp(logf - top) * _GL_W).sum(axis=-1))


def pn_logpdf(u: np.ndarray, m: np.ndarray, s: np.ndarray) -> np.ndarray:
    """log density of u under PN(m, S) w.r.t. the surface measure of S^{d-1};
    u, m: (n, d); S: (d, d) or (n, d, d)."""
    n, d = u.shape
    s = np.broadcast_to(s, (n, d, d))
    su = np.linalg.solve(s, u[..., None])[..., 0]
    sm = np.linalg.solve(s, m[..., None])[..., 0]
    a = (u * su).sum(1)
    b = (u * sm).sum(1)
    c = (m * sm).sum(1)
    _, logdet = np.linalg.slogdet(s)
    alpha = b / np.sqrt(a)
    return (-0.5 * d * np.log(2 * np.pi) - 0.5 * logdet - 0.5 * c
            - 0.5 * d * np.log(a) + log_ik(d - 1, alpha))


def pn_moments(m: np.ndarray, s: np.ndarray, rng, ndraw: int = 2000, chunk: int = 200):
    """Monte Carlo mean direction v = E u / |E u| and mean resultant length
    rho = |E u| of PN(m, S) per row: (v (n, d), rho (n,))."""
    n, d = m.shape
    s = np.broadcast_to(s, (n, d, d))
    v = np.empty((n, d))
    rho = np.empty(n)
    for i0 in range(0, n, chunk):
        i1 = min(n, i0 + chunk)
        l = np.linalg.cholesky(s[i0:i1])
        z = rng.standard_normal((ndraw, i1 - i0, d))
        x = m[i0:i1] + np.einsum("nij,knj->kni", l, z)
        eu = (x / np.linalg.norm(x, axis=2, keepdims=True)).mean(0)
        r = np.linalg.norm(eu, axis=1)
        v[i0:i1] = eu / r[:, None]
        rho[i0:i1] = r
    return v, rho


def unit(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


class Forecast:
    """A model's one-step forecasts on the test rows: point direction v, mean
    resultant length rho (None for point forecasts) and log density of the
    realized direction (None without a density)."""

    def __init__(self, v, rho=None, logpdf=None):
        self.v, self.rho, self.logpdf = v, rho, logpdf


def pn_forecast(m, s, u, rng, ndraw=2000, symmetric=False) -> Forecast:
    """Forecast from predictive PN(m_t, S_t): ESAG (``symmetric``: E u is along
    m) or a general projected normal (Monte Carlo mean direction)."""
    v, rho = pn_moments(m, s, rng, ndraw)
    if symmetric:
        v = unit(m)
    return Forecast(v, rho, pn_logpdf(u, m, s))


def esag_fit(y, x, shape="gamma-rot", penalty=None, scale=None):
    """ESAG regression of directions y on covariates x (None: i.i.d.); with
    ``penalty`` the relaxed refit of the selected model; ``scale`` as in
    ``esag.ESAG`` (part scales of the responses)."""
    model = esag.ESAG(y, x, shape=shape, scale=scale)
    if penalty is None:
        return model.fit()
    return model.fit_regularized(penalty).relaxed


def esag_forecast(res, x_new, u, rng, ndraw=2000) -> Forecast:
    """The fitted ESAG's predictive PN(mu(x), V(x)) at the test covariates."""
    if x_new is None:
        m = np.asarray(res.mean_vector())[:1].repeat(len(u), 0)
        cov = np.asarray(res.covariance())[:1].repeat(len(u), 0)
    else:
        m = np.asarray(res.mean_vector(x_new))
        cov = np.asarray(res.covariance(x_new))
    # (ESAG is symmetric about its mean vector; with part scales the
    # predictive projected normal PN(D mu, D V D) need not be)
    return pn_forecast(m, cov, u, rng, ndraw, symmetric=res.scale is None)


def scores(f: Forecast, u: np.ndarray, oracle: Forecast) -> dict:
    """cos: mean cosine between forecast and realized direction (1 = perfect);
    angle: mean angle to the oracle's mean direction (degrees); logscore: mean
    log density of the realized directions; rho_err: mean |rho - rho_oracle|;
    rho_corr: correlation of rho with the oracle's (does the model know when
    the next direction is predictable?); calib: mean rho minus the realized
    mean cosine (0 = calibrated concentration)."""
    cos = (f.v * u).sum(1)
    out = {
        "cos": cos.mean(),
        "angle": np.degrees(np.arccos(np.clip((f.v * oracle.v).sum(1), -1, 1))).mean(),
        "logscore": np.nan if f.logpdf is None else f.logpdf.mean(),
        "rho_err": np.nan,
        "rho_corr": np.nan,
        "calib": np.nan,
    }
    if f.rho is not None:
        out["rho_err"] = np.abs(f.rho - oracle.rho).mean()
        out["rho_corr"] = np.corrcoef(f.rho, oracle.rho)[0, 1]
        out["calib"] = f.rho.mean() - cos.mean()
    return out


def ols_var(target: np.ndarray, x: np.ndarray):
    """Least squares target ~ [1, x]: (coefficients (p+1, d), residual
    covariance)."""
    z = np.column_stack([np.ones(len(x)), x]) if x is not None else np.ones((len(target), 1))
    coef, *_ = np.linalg.lstsq(z, target, rcond=None)
    resid = target - z @ coef
    return coef, resid.T @ resid / (len(target) - z.shape[1])


def ols_predict(coef: np.ndarray, x: np.ndarray) -> np.ndarray:
    z = np.column_stack([np.ones(len(x)), x])
    return z @ coef


def helmert(dparts: int) -> np.ndarray:
    """Orthonormal (D-1) x D contrast basis: ilr(p) = H clr(p)."""
    h = np.zeros((dparts - 1, dparts))
    for i in range(1, dparts):
        h[i - 1, :i] = 1.0 / i
        h[i - 1, i] = -1.0
        h[i - 1] *= np.sqrt(i / (i + 1.0))
    return h


def clr(p: np.ndarray) -> np.ndarray:
    lp = np.log(p)
    return lp - lp.mean(axis=-1, keepdims=True)


def ilr(p: np.ndarray, h: np.ndarray) -> np.ndarray:
    return clr(p) @ h.T


def ilr_inv(z: np.ndarray, h: np.ndarray) -> np.ndarray:
    e = np.exp(z @ h)
    return e / e.sum(axis=-1, keepdims=True)


def random_rotation(d: int, rng) -> np.ndarray:
    q, r = np.linalg.qr(rng.standard_normal((d, d)))
    return q * np.sign(np.diag(r))


def noise_cov(d: int, eig, rng) -> np.ndarray:
    """A covariance with eigenvalues ``eig`` and random axes."""
    q = random_rotation(d, rng)
    return q @ np.diag(eig) @ q.T


def summarize(rows: list, keys=("cos", "angle", "logscore", "rho_err", "rho_corr", "calib")):
    """Mean and standard error over replicates of each score."""
    out = {}
    for k in keys:
        vals = np.array([r[k] for r in rows], dtype=float)
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            out[k] = (np.nan, np.nan)
        else:
            se = vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else np.nan
            out[k] = (vals.mean(), se)
    return out


def print_table(title: str, results: dict, order=None, baseline="oracle"):
    """One line per model: the scores' means (and the log-score gap to the
    oracle)."""
    print(f"\n{title}")
    print(f"  {'model':<26} {'cos':>7} {'angle°':>7} {'log score':>10} {'gap':>7} "
          f"{'|Δrho|':>7} {'corr rho':>8} {'calib':>7}")
    base = results.get(baseline)
    for name in order or results:
        s = results[name]
        gap = np.nan
        if base is not None and not np.isnan(s["logscore"][0]):
            gap = base["logscore"][0] - s["logscore"][0]
        cell = lambda k, f: "" if np.isnan(s[k][0]) else format(s[k][0], f)
        print(f"  {name:<26} {cell('cos', '7.3f'):>7} {cell('angle', '7.1f'):>7} "
              f"{cell('logscore', '10.3f'):>10} {'' if np.isnan(gap) else format(gap, '7.3f'):>7} "
              f"{cell('rho_err', '7.3f'):>7} {cell('rho_corr', '8.3f'):>8} {cell('calib', '7.3f'):>7}")


class PNRegression:
    """Projected normal regression with a constant covariance: u_t ~ PN(B z_t, S)
    with z_t = [1, x_t] and det S = 1 (the scale of a projected normal is not
    identified). The exact model for the directions of a Gaussian VAR, whose
    noise covariance is fixed in the ambient space while its mean direction
    roams the sphere; ESAG instead ties V's axes to the mean direction.
    Fitted by L-BFGS on analytic gradients, with the precision S^-1 = K K^T
    (K lower triangular, log-diagonal summing to 0)."""

    def __init__(self, u, x):
        from scipy.optimize import minimize

        self.d = d = u.shape[1]
        z = np.ones((len(u), 1)) if x is None else np.column_stack([np.ones(len(u)), x])
        self.q = q = z.shape[1]
        self._tril = np.tril_indices(d, -1)
        # start: least squares of u on z, scaled to a moderate concentration
        b0, *_ = np.linalg.lstsq(z, u, rcond=None)
        theta = np.concatenate([2.0 * b0.T.ravel(), np.zeros(d - 1 + len(self._tril[0]))])
        fun = lambda th: self._negloglik(th, u, z)
        opt = minimize(fun, theta, jac=True, method="L-BFGS-B",
                       options={"maxiter": 5000, "maxfun": 20000, "gtol": 1e-7})
        self.converged = bool(opt.success)
        self.llf = -opt.fun
        self.b, self.k = self._unpack(opt.x)
        self.cov = np.linalg.inv(self.k @ self.k.T)
        self.n_params = len(theta)

    def _unpack(self, th):
        d, q = self.d, self.q
        b = th[:d * q].reshape(d, q)
        kd = th[d * q:d * q + d - 1]
        k = np.zeros((d, d))
        k[self._tril] = th[d * q + d - 1:]
        diag = np.append(kd, -kd.sum())
        k[np.diag_indices(d)] = np.exp(diag)
        return b, k

    def _negloglik(self, th, u, z):
        d, q = self.d, self.q
        b, k = self._unpack(th)
        m = z @ b.T
        a = u @ k
        bb = m @ k
        na2 = (a * a).sum(1)
        na = np.sqrt(na2)
        ab = (a * bb).sum(1)
        alpha = ab / na
        lik = log_ik(d - 1, alpha)
        r = np.exp(log_ik(d, alpha) - lik)
        ll = (-0.5 * d * np.log(2 * np.pi) - 0.5 * (bb * bb).sum(1) - 0.5 * d * np.log(na2) + lik)
        ga = (-d * a / na2[:, None] + r[:, None] * (bb / na[:, None]
              - (ab / na**3)[:, None] * a))
        gb = -bb + r[:, None] * a / na[:, None]
        gk = u.T @ ga + m.T @ gb
        gm = gb @ k.T
        gbeta = gm.T @ z
        grad_kd = np.diag(gk) * np.diag(k)
        g = np.concatenate([gbeta.ravel(), grad_kd[:-1] - grad_kd[-1], gk[self._tril]])
        return -ll.sum(), -g

    def mean(self, x):
        z = np.ones((len(x), 1)) if x is None else np.column_stack([np.ones(len(x)), x])
        return z @ self.b.T

    def forecast(self, x_new, u, rng, ndraw=2000):
        m = self.mean(x_new) if x_new is not None else np.repeat(self.b.T[:1], len(u), 0)
        return pn_forecast(m, self.cov, u, rng, ndraw)
