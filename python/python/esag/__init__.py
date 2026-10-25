"""ESAG regression for directional data, with a statsmodels-style API.

The numerical core is the ``esag`` Rust crate: maximum likelihood with exact
gradients and Hessians, the LASSO family (LASSO, adaptive LASSO, group,
adaptive group and sparse group LASSO, isotropy and eigen-gap penalties),
and parametric-bootstrap tests (RoC, D, M, LR) that run in parallel.

    >>> import esag
    >>> model = esag.ESAG(Y, X)            # endog: n x d unit vectors, exog: n x p
    >>> res = model.fit()
    >>> print(res.summary())
    >>> res.test("isotropy", n_boot=500).summary()
    >>> reg = model.fit_regularized("adaptive_group_lasso")
    >>> print(reg.summary())

``endog`` and ``exog`` may be numpy arrays or pandas objects; pandas column
names are used in the output and pandas results come back when pandas is
installed. A constant (``const``) is always part of the model; a constant
column already in ``exog`` (e.g. from ``sm.add_constant``) is dropped.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Union

import warnings

import numpy as np

from . import _esag
from ._esag import Distribution, __version__

try:  # pandas is optional
    import pandas as _pd
except ImportError:  # pragma: no cover
    _pd = None

__all__ = [
    "ESAG",
    "ESAGResults",
    "RegularizedResults",
    "TestResults",
    "CovarianceEstimate",
    "ShapeWarning",
    "Summary",
    "Distribution",
    "PENALTIES",
    "TARGETS",
    "GROUPINGS",
    "NULLS",
    "__version__",
]

class ShapeWarning(UserWarning):
    """``shape="auto"`` chose the log-tangent model instead of gamma."""


#: ``fit_regularized(penalty=...)`` choices.
PENALTIES = (
    "lasso",
    "adaptive_lasso",
    "group_lasso",
    "adaptive_group_lasso",
    "sparse_group_lasso",
    "isotropy",
    "eigen_gap",
)
#: ``fit_regularized(target=...)``: which coefficients are penalized.
TARGETS = ("slopes", "mean", "shape_slopes", "shape")
#: ``fit_regularized(groups=...)``: how the target is grouped.
GROUPINGS = ("covariate", "covariate_part", "parameter", "all", "gamma")
#: ``ESAGResults.test(null=...)`` choices.
NULLS = ("isotropy", "mu", "shape", "feature")


class Summary:
    """Text summary (statsmodels ``Summary``-like: print it, or ``as_text()``)."""

    def __init__(self, text: str):
        self._text = text

    def as_text(self) -> str:
        return self._text

    def __str__(self) -> str:
        return self._text

    def __repr__(self) -> str:
        return self._text

    def _repr_html_(self) -> str:
        import html

        return "<pre>" + html.escape(self._text) + "</pre>"


# ---------------------------------------------------------------------------
# data handling


def _names(obj, n_cols: int, prefix: str) -> list:
    cols = getattr(obj, "columns", None)
    if cols is not None:
        return [str(c) for c in cols]
    name = getattr(obj, "name", None)
    if name is not None and n_cols == 1:
        return [str(name)]
    return [f"{prefix}{j + 1}" for j in range(n_cols)]


def _as_2d(a, what: str) -> np.ndarray:
    arr = np.asarray(a, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"{what} must be one- or two-dimensional")
    return np.ascontiguousarray(arr)


def _constant_columns(x: np.ndarray) -> list:
    if x.shape[0] == 0:
        return []
    return [j for j in range(x.shape[1]) if np.all(x[:, j] == x[0, j]) and x[0, j] != 0.0]


def _frame(values: np.ndarray, index, columns):
    if _pd is None:
        return values
    return _pd.DataFrame(values, index=list(index), columns=list(columns))


def _series(values, index, name=None):
    if _pd is None:
        return np.asarray(values)
    return _pd.Series(values, index=list(index), name=name)


# ---------------------------------------------------------------------------
# model


class ESAG:
    """ESAG regression of directions ``endog`` on covariates ``exog``.

    Parameters
    ----------
    endog : (n, d) array-like
        Unit vectors on the sphere S^{d-1}.
    exog : (n, p) array-like or None
        Covariates (no constant needed; one is added). ``None`` fits an
        i.i.d. ESAG.
    shape : {"auto", "gamma", "log-tangent"}
        Shape parameterization. ``"gamma"`` is the papers' (Paine et al.
        2018; Yu & Huang). ``"log-tangent"`` writes V through the matrix
        exponential of a traceless symmetric matrix in the tangent space;
        it is smooth everywhere, so big regressions (d >= 5 with covariates)
        fit quickly and reliably. ``"auto"`` uses gamma when d <= 4 or there
        are no covariates, and log-tangent otherwise, with a
        :class:`ShapeWarning` when it picks log-tangent. With covariates the
        two are different (non-nested) regression models for the shape, as
        the shape coefficients enter linearly in different coordinates; the
        mean model is the same. Compare them by AIC/BIC.
    missing : {"none", "drop"}
        ``"drop"`` removes rows with NaN in endog or exog; ``"none"`` raises.
    lt_reference : array-like of length d, optional
        Reference direction of the log-tangent frame (default: the data's
        mean direction, stored with the fit as ``results.lt_reference`` and
        printed in the summary). The log-tangent regression model depends on
        it, since s linear in x in one frame is not linear in another. Pass
        ``lt_reference=results.lt_reference`` to fit the same model to other
        data, or set it to fit a model defined in a given frame (e.g. a
        simulation truth); it has no effect for ``shape="gamma"``.
    """

    def __init__(self, endog, exog=None, shape: str = "auto", missing: str = "none",
                 lt_reference=None):
        y = _as_2d(endog, "endog")
        self.endog_names = _names(endog, y.shape[1], "y")
        if exog is None:
            x = np.zeros((y.shape[0], 0))
            exog_names = []
        else:
            x = _as_2d(exog, "exog")
            exog_names = _names(exog, x.shape[1], "x")
        if x.shape[0] != y.shape[0]:
            raise ValueError(f"exog has {x.shape[0]} rows but endog has {y.shape[0]}")
        if missing == "drop":
            keep = ~(np.isnan(y).any(axis=1) | np.isnan(x).any(axis=1))
            y, x = y[keep], x[keep]
        elif missing != "none":
            raise ValueError("missing must be 'none' or 'drop'")
        if np.isnan(y).any() or np.isnan(x).any():
            raise ValueError("data contain NaN (use missing='drop')")
        self._drop = _constant_columns(x)
        keep_cols = [j for j in range(x.shape[1]) if j not in self._drop]
        self._n_exog_in = x.shape[1]
        x = np.ascontiguousarray(x[:, keep_cols])
        self.exog_names = ["const"] + [exog_names[j] for j in keep_cols]
        d = y.shape[1]
        if shape == "auto":
            shape = "gamma" if d <= 4 or x.shape[1] == 0 else "log-tangent"
            if shape == "log-tangent":
                warnings.warn(
                    "shape='auto' chose the log-tangent model (d >= 5 with covariates), "
                    "a different shape regression from the gamma model of Yu & Huang; "
                    "pass shape='gamma' for that model, or shape='log-tangent' to "
                    "silence this warning",
                    ShapeWarning,
                    stacklevel=2,
                )
        self.shape = shape
        self.endog = y
        self.exog = x
        self.nobs = y.shape[0]
        self.k_endog = d
        self.param_names = _esag.param_rows(d, shape)
        self._native = _esag.Model(
            y,
            x,
            exog_names=self.exog_names[1:],
            endog_names=self.endog_names,
            shape=shape,
            lt_reference=None if lt_reference is None else [float(v) for v in lt_reference],
        )

    def _exog_new(self, exog) -> np.ndarray:
        x = _as_2d(exog, "exog")
        if x.shape[1] == self._n_exog_in and self._drop:
            x = x[:, [j for j in range(x.shape[1]) if j not in self._drop]]
        if x.shape[1] != len(self.exog_names) - 1:
            raise ValueError(f"exog must have {len(self.exog_names) - 1} columns")
        return np.ascontiguousarray(x)

    def fit(self, maxiter: Optional[int] = None, tol: Optional[float] = None,
            restarts: int = 0, seed: int = 0, cov: bool = True) -> "ESAGResults":
        """Maximum-likelihood fit.

        ``maxiter`` and ``tol`` default to the core's (2000, 1e-6);
        ``restarts`` adds perturbed starts (the best likelihood wins);
        ``cov=False`` skips the observed information (no standard errors).
        """
        return ESAGResults(self, self._native.fit(maxiter=maxiter, tol=tol, restarts=restarts,
                                                  seed=seed, cov=cov))

    def fit_regularized(
        self,
        penalty: str = "adaptive_lasso",
        *,
        target: Optional[str] = None,
        groups: Union[None, str, Sequence[Sequence[tuple]]] = None,
        l1_ratio: float = 0.5,
        adaptive: Union[None, bool, float] = None,
        select: str = "bic",
        alpha: Union[None, float, Iterable[float]] = None,
        cv_folds: int = 10,
        seed: int = 1,
        paper: Optional[str] = None,
        paper_grid: bool = False,
    ) -> "RegularizedResults":
        """Penalized fit along a path of penalty weights, one selected.

        Parameters
        ----------
        penalty : str
            One of :data:`PENALTIES`:

            * ``"lasso"`` / ``"adaptive_lasso"``: L1 on each coefficient;
            * ``"group_lasso"`` / ``"adaptive_group_lasso"``: L2 norm of each
              group (``groups``), sqrt(group size) weights;
            * ``"sparse_group_lasso"``: ``l1_ratio`` * LASSO +
              (1 - ``l1_ratio``) * group LASSO;
            * ``"isotropy"``: all shape coefficients as one group (shrinks
              towards the isotropic projected normal);
            * ``"eigen_gap"``: i.i.d. only; fuses equal eigenvalues of the
              shape (needs ``shape="log-tangent"``).
        target : str, optional
            Which coefficients are penalized (:data:`TARGETS`): ``"slopes"``
            (default for the LASSO family: every non-constant coefficient),
            ``"mean"`` (mean-direction slopes), ``"shape_slopes"``, or
            ``"shape"`` (shape coefficients, constant included).
        groups : str or list, optional
            Grouping for group penalties (:data:`GROUPINGS`): ``"covariate"``
            (default: all target coefficients of one covariate),
            ``"covariate_part"`` (a covariate's mean slopes and its shape
            slopes separately), ``"parameter"`` (one group per row of B),
            ``"all"``, ``"gamma"`` (the papers' gamma groups); or explicit
            groups as lists of ``(parameter, variable)`` pairs, by name
            (``("mu[1]", "x1")``) or index.
        adaptive : bool or float, optional
            Adaptive weights ||theta_mle||^-r: ``True`` (r = 1), a number r,
            or ``False`` (plain weights). Defaults to the preset's.
        select : {"bic", "aic", "cv"}
            How the penalty weight is chosen (BIC and AIC at the relaxed
            refit; ``"cv"``: ``cv_folds``-fold held-out log-likelihood).
        alpha : float or sequence, optional
            Penalty weight(s) on the per-observation scale, as in statsmodels
            (``alpha = lambda / n``); default: an automatic path from the
            smallest weight that zeroes everything down to 0.
        paper : str, optional
            Reproduce one of the six methods exactly as in the Project 3 R
            scripts ("LASSO", "aLASSO", "RoC-LASSO", "GLASSO", "aGLASSO",
            "RoC-GLASSO"; i.i.d., gamma shape).
        """
        custom = None
        if groups is not None and not isinstance(groups, str):
            custom = [[self._coef_index(e) for e in g] for g in groups]
            groups = None
        if isinstance(adaptive, bool):
            r = 1.0 if adaptive else 0.0
        else:
            r = None if adaptive is None else float(adaptive)
        lambdas = None
        if alpha is not None:
            a = np.atleast_1d(np.asarray(alpha, dtype=np.float64))
            lambdas = [float(v) * self.nobs for v in a]
        native = self._native.fit_regularized(
            penalty=penalty,
            target=target,
            groups=groups,
            l1_ratio=l1_ratio,
            r=r,
            select=select,
            folds=cv_folds,
            seed=seed,
            lambdas=lambdas,
            paper_grid=paper_grid,
            paper=paper,
            custom_groups=custom,
        )
        return RegularizedResults(self, native)

    def _coef_index(self, e) -> tuple:
        k, j = e
        if isinstance(k, str):
            k = self.param_names.index(k)
        if isinstance(j, str):
            j = self.exog_names.index(j)
        return (int(k), int(j))

    def __repr__(self) -> str:
        return (f"<esag.ESAG: nobs={self.nobs}, d={self.k_endog}, "
                f"exog={self.exog_names}, shape={self.shape!r}>")


# ---------------------------------------------------------------------------
# results


class _CoefTable:
    """Shared accessors for results whose ``params`` is a B matrix."""

    model: ESAG

    def _df(self, m: np.ndarray):
        # statsmodels (MNLogit) layout: rows = exog names, columns = equations
        return _frame(np.asarray(m).T, self.model.exog_names, self.model.param_names)


class ESAGResults(_CoefTable):
    """Results of :meth:`ESAG.fit` (statsmodels-style attributes)."""

    def __init__(self, model: ESAG, native):
        self.model = model
        self._native = native

    # statsmodels attributes ------------------------------------------------
    @property
    def params(self):
        """Coefficients: rows = exog (``const``, covariates), columns =
        equations (``mu[k]``, then ``gamma[k]`` or ``s[k]``)."""
        return self._df(self._native.params)

    @property
    def bse(self):
        return self._df(self._native.bse)

    @property
    def tvalues(self):
        """z statistics (``coef / std err``)."""
        return self._df(self._native.zvalues)

    zvalues = tvalues

    @property
    def pvalues(self):
        return self._df(self._native.pvalues)

    def conf_int(self, alpha: float = 0.05):
        """Wald intervals; with pandas, a frame indexed by (equation, exog)
        with ``lower`` / ``upper`` columns."""
        lo, hi = self._native.conf_int(alpha)
        if _pd is None:
            return lo, hi
        idx = _pd.MultiIndex.from_product([self.model.param_names, self.model.exog_names],
                                          names=["equation", "exog"])
        return _pd.DataFrame({"lower": np.asarray(lo).ravel(), "upper": np.asarray(hi).ravel()},
                             index=idx)

    def cov_params(self):
        """Covariance of the estimated coefficients (labels ``equation:exog``)."""
        c = self._native.cov_params
        if c is None:
            raise ValueError("fitted with cov=False")
        cov, free = c
        labels = [f"{self.model.param_names[k]}:{self.model.exog_names[j]}" for k, j in free]
        return _frame(np.asarray(cov), labels, labels)

    llf = property(lambda self: self._native.llf, doc="Maximised log-likelihood.")
    aic = property(lambda self: self._native.aic)
    bic = property(lambda self: self._native.bic)
    nobs = property(lambda self: self._native.nobs)
    df_model = property(lambda self: self._native.df_model,
                        doc="Number of estimated coefficients.")
    df_resid = property(lambda self: self._native.df_resid)
    converged = property(lambda self: self._native.converged)
    n_iter = property(lambda self: self._native.n_iter)
    lt_reference = property(
        lambda self: None if self._native.lt_reference is None
        else np.asarray(self._native.lt_reference),
        doc="Reference direction m of the log-tangent frame (None for gamma). "
            "ESAG(..., lt_reference=m) fits the same model to other data.")

    def summary(self, alpha: float = 0.05) -> Summary:
        return Summary(self._native.summary(alpha))

    # prediction --------------------------------------------------------------
    def predict(self, exog=None):
        """Mean directions (unit vectors) at ``exog`` (default: the training
        covariates)."""
        x = self.model.exog if exog is None else self.model._exog_new(exog)
        out = np.asarray(self._native.predict(x))
        return _frame(out, range(out.shape[0]), self.model.endog_names)

    def linear_predictor(self, exog=None):
        """``eta = B [1, x]``: the mean vector mu followed by the shape
        coordinates, per observation."""
        x = self.model.exog if exog is None else self.model._exog_new(exog)
        out = np.asarray(self._native.linear_predictor(x))
        return _frame(out, range(out.shape[0]), self.model.param_names)

    def loglike_obs(self):
        return np.asarray(self._native.loglike_obs(self.model.exog, self.model.endog))

    @property
    def resid(self):
        """Residuals ``Q`` and ``T1`` (``T1`` is approximately chi2(d - 1))."""
        q, t1 = self._native.resid()
        if _pd is None:
            return np.column_stack([q, t1])
        return _pd.DataFrame({"Q": q, "T1": t1})

    # inference ---------------------------------------------------------------
    def test(self, null: str = "isotropy", exog: Union[None, str, int] = None,
             n_boot: int = 500, seed: int = 1) -> "TestResults":
        """Parametric-bootstrap test of a restriction (RoC, D, M, LR together).

        ``null`` is one of :data:`NULLS`:

        * ``"isotropy"``: the shape is zero (isotropic projected normal);
        * ``"mu"``: the mean direction does not depend on ``exog``;
        * ``"shape"``: the shape does not depend on ``exog``;
        * ``"feature"``: ``exog`` has no effect at all.

        ``exog`` names a covariate (or its 0-based index among the
        covariates).
        """
        if isinstance(exog, (int, np.integer)):
            exog = str(int(exog))
        return TestResults(self._native.test(null=null, feature=exog, n_boot=n_boot, seed=seed))

    def gof(self, n_boot: int = 200, seed: int = 1, one_sample: bool = False) -> dict:
        """Bootstrap goodness-of-fit test of the ESAG model (Algorithm 1)."""
        return self._native.gof(n_boot=n_boot, seed=seed, one_sample=one_sample)

    def prediction_region(self, exog, levels=(0.95, 0.9), n_mc: int = 5000,
                          n_boot: int = 100, seed: int = 1) -> dict:
        """Thresholds of bootstrap-calibrated prediction regions at one
        covariate vector."""
        x = self.model._exog_new(np.asarray(exog, dtype=np.float64)[None, :])[0]
        return self._native.prediction_region(list(map(float, x)), list(levels), n_mc=n_mc,
                                              n_boot=n_boot, seed=seed)

    # the ESAG covariance V(x) -------------------------------------------------
    def covariance(self, exog=None) -> np.ndarray:
        """Estimated ESAG covariance V(x) at each row of ``exog`` (default:
        the training covariates): an ``(m, d, d)`` array with ``V mu = mu``
        and ``det V = 1``."""
        x = self.model.exog if exog is None else self.model._exog_new(exog)
        return np.asarray(self._native.covariance(x))

    def covariance_ci(self, exog=None, n_boot: int = 500, level: float = 0.95,
                      seed: int = 1):
        """V(x) with pairs-bootstrap uncertainty.

        At each row of ``exog`` (default: the mean covariate vector), returns
        a :class:`CovarianceEstimate` with V-hat, its shape eigenvalues and
        principal axes, basic (bias-corrected) bootstrap intervals and
        standard errors for the eigenvalues and for every entry, and the
        radius of a geodesic confidence ball. A single covariate vector gives
        one estimate; a matrix gives a list.

        These intervals were checked by simulation (n = 2000, 5 covariates,
        d = 3 and 5, both shapes): 93-98% coverage at level 95%.
        """
        single = exog is None or np.ndim(exog) == 1
        if exog is None:
            x = self.model.exog.mean(axis=0, keepdims=True)
        else:
            x = np.atleast_2d(np.asarray(exog, dtype=np.float64))
            x = self.model._exog_new(x)
        out = [CovarianceEstimate(e, self.model.endog_names)
               for e in self._native.covariance_ci(np.ascontiguousarray(x), n_boot=n_boot,
                                                   level=level, seed=seed)]
        return out[0] if single else out

    def simulate(self, exog=None, seed: int = 0):
        """Draw one response per row of ``exog`` (default: the training
        covariates) from the fitted model."""
        x = self.model.exog if exog is None else self.model._exog_new(exog)
        out = np.asarray(self._native.simulate(x, seed=seed))
        return _frame(out, range(out.shape[0]), self.model.endog_names)

    def to_json(self) -> str:
        """The fitted model as JSON (reload with :meth:`from_json`)."""
        return self._native.to_json()

    @classmethod
    def from_json(cls, model: "ESAG", json: str) -> "ESAGResults":
        """Results saved with :meth:`to_json`, attached to ``model`` (the
        same data) for tests and diagnostics."""
        return cls(model, _esag.results_from_json(json, model.endog, model.exog))

    def __repr__(self) -> str:
        return (f"<esag.ESAGResults: llf={self.llf:.4f}, nobs={self.nobs}, "
                f"converged={self.converged}>")


class CovarianceEstimate:
    """V-hat(x) with bootstrap uncertainty (see :meth:`ESAGResults.covariance_ci`).

    Attributes: ``covariance`` (d x d), ``mean_direction``, ``eigenvalues``
    (shape eigenvalues, ascending; the eigenvalue along the mean direction
    is 1), ``axes`` (d x (d-1)), ``eig_lower`` / ``eig_upper`` / ``eig_se``,
    ``entry_se`` / ``entry_lower`` / ``entry_upper`` (d x d), ``radius``
    (geodesic ball), ``level``, ``n_boot``.
    """

    def __init__(self, d: dict, names):
        self._d = d
        self._names = list(names)
        for k, v in d.items():
            if k != "summary":
                setattr(self, k, np.asarray(v) if isinstance(v, (list, np.ndarray)) else v)

    @property
    def covariance_frame(self):
        """V-hat as a DataFrame labelled by the endog names (if pandas)."""
        return _frame(np.asarray(self.covariance), self._names, self._names)

    @property
    def eigen_table(self):
        """Shape eigenvalues with standard errors and intervals."""
        rows = [f"lambda[{k + 1}]" for k in range(len(self.eigenvalues))]
        cols = {"estimate": self.eigenvalues, "std err": self.eig_se,
                "lower": self.eig_lower, "upper": self.eig_upper}
        if _pd is None:
            return cols
        return _pd.DataFrame(cols, index=rows)

    def summary(self) -> Summary:
        return Summary(self._d["summary"])

    def __repr__(self) -> str:
        return self._d["summary"]


class TestResults:
    """Results of :meth:`ESAGResults.test`."""

    __test__ = False  # not a pytest class

    def __init__(self, native):
        self._native = native

    null = property(lambda self: self._native.null)
    statistic = property(lambda self: self._native.statistic,
                         doc="Observed statistics: RoC, D, M, LR.")
    pvalue = property(lambda self: self._native.pvalue,
                      doc="Bootstrap p-values, plus LR_chi2 (asymptotic).")
    df = property(lambda self: self._native.df)
    n_boot = property(lambda self: self._native.n_boot)
    n_failed = property(lambda self: self._native.n_failed)
    n_unconverged = property(lambda self: self._native.n_unconverged)
    llf_restricted = property(lambda self: self._native.llf_restricted)
    llf_full = property(lambda self: self._native.llf_full)
    bootstrap = property(lambda self: self._native.bootstrap,
                         doc="Bootstrap replicates of each statistic.")

    @property
    def table(self):
        stat, p = self.statistic, self.pvalue
        names = ["RoC", "D", "M", "LR"]
        if _pd is None:
            return {k: (stat[k], p[k]) for k in names}
        return _pd.DataFrame({"statistic": [stat[k] for k in names],
                              "pvalue": [p[k] for k in names]}, index=names)

    def summary(self) -> Summary:
        return Summary(self._native.summary())

    def __repr__(self) -> str:
        return self._native.summary()


class RegularizedResults(_CoefTable):
    """Results of :meth:`ESAG.fit_regularized`.

    ``params`` are the penalized coefficients at the selected weight;
    ``relaxed`` is the unpenalized refit of the selected model (with
    standard errors), which ``summary()`` reports.
    """

    def __init__(self, model: ESAG, native):
        self.model = model
        self._native = native

    @property
    def params(self):
        return self._df(self._native.params)

    method = property(lambda self: self._native.method)

    @property
    def relaxed(self) -> Optional[ESAGResults]:
        r = self._native.relaxed
        return None if r is None else ESAGResults(self.model, r)

    @property
    def alpha(self) -> float:
        """Selected penalty weight on the per-observation scale (lambda / n)."""
        return self._native.alphas[self._native.selected]

    @property
    def path(self):
        """One row per penalty weight: alpha, lambda, df, llf, criterion,
        number of active groups, and whether it was selected."""
        n = self._native
        cols = {
            "alpha": n.alphas,
            "lambda": n.lambdas,
            "df": n.df_path,
            "llf": n.llf_path,
            "criterion": n.criterion,
            "n_active": [sum(a) for a in n.active_path],
            "selected": [i == n.selected for i in range(len(n.alphas))],
        }
        if _pd is None:
            return cols
        return _pd.DataFrame(cols)

    def params_path(self) -> list:
        return [self._df(m) for m in self._native.params_path()]

    def summary(self) -> Summary:
        return Summary(self._native.summary())

    def path_summary(self) -> Summary:
        return Summary(self._native.path_summary())

    def __repr__(self) -> str:
        return f"<esag.RegularizedResults: {self.method}, alpha={self.alpha:.4g}>"
