//! Native module `esag._esag`: thin PyO3 wrappers around the `esag` crate.
//! The user-facing, statsmodels-style classes live in `python/esag`.

use esag::distribution::Esag;
use esag::inference::{GofMethod, HypothesisTestResult, Restriction, bootstrap_gof};
use esag::model::{EsagModel, EsagResults};
use esag::penalized::{Grouping, LambdaGrid, PenalizedEsag, PenalizedPath, Target};
use esag::regression::{EsagRegressor, Shape};
use nalgebra::DMatrix;
use numpy::ndarray::{Array1, Array2, Array3};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rand::SeedableRng;

/// A covariance matrix and the `(row, column)` coefficients it covers.
type CovParams<'py> = (Bound<'py, PyArray2<f64>>, Vec<(usize, usize)>);
type Vector<'py> = Bound<'py, PyArray1<f64>>;

fn err(e: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(e.to_string())
}

fn to_dmatrix(a: &PyReadonlyArray2<'_, f64>) -> DMatrix<f64> {
    let v = a.as_array();
    DMatrix::from_fn(v.nrows(), v.ncols(), |i, j| v[[i, j]])
}

fn to_py<'py>(py: Python<'py>, m: &DMatrix<f64>) -> Bound<'py, PyArray2<f64>> {
    Array2::from_shape_fn((m.nrows(), m.ncols()), |(i, j)| m[(i, j)]).into_pyarray(py)
}

fn shape_of(s: &str) -> PyResult<Shape> {
    match s.to_ascii_lowercase().replace('_', "-").as_str() {
        "gamma" => Ok(Shape::Gamma),
        "gamma-transported" | "gt" => Ok(Shape::GammaTransported),
        "gamma-rot" | "gammarot" | "gr" => Ok(Shape::GammaRot),
        "spectral" | "spec" | "sp" => Ok(Shape::Spectral),
        "log-tangent" | "logtangent" | "lt" => Ok(Shape::LogTangent),
        other => Err(err(format!(
            "unknown shape {other:?} (gamma, gamma-rot, spectral, gamma-transported, log-tangent)"
        ))),
    }
}

/// An ESAG regression model (see `esag.ESAG`).
#[pyclass(name = "Model", module = "esag._esag")]
struct PyModel {
    inner: EsagModel,
}

#[pymethods]
impl PyModel {
    #[new]
    #[pyo3(signature = (endog, exog=None, exog_names=None, endog_names=None, shape="gamma", hasconst=true, lt_reference=None, rot_basis=None))]
    fn new(
        endog: PyReadonlyArray2<'_, f64>,
        exog: Option<PyReadonlyArray2<'_, f64>>,
        exog_names: Option<Vec<String>>,
        endog_names: Option<Vec<String>>,
        shape: &str,
        hasconst: bool,
        lt_reference: Option<Vec<f64>>,
        rot_basis: Option<Vec<Vec<f64>>>,
    ) -> PyResult<Self> {
        let y = to_dmatrix(&endog);
        let x = match &exog {
            Some(a) => to_dmatrix(a),
            None => DMatrix::zeros(y.nrows(), 0),
        };
        if x.nrows() != y.nrows() {
            return Err(err(format!(
                "exog has {} rows but endog has {}",
                x.nrows(),
                y.nrows()
            )));
        }
        let mut m = EsagModel::new(&y, &x)
            .shape(shape_of(shape)?)
            .has_constant(hasconst);
        if let Some(n) = exog_names {
            if n.len() != x.ncols() {
                return Err(err("exog_names must have one name per exog column"));
            }
            m = m.exog_names(&n);
        }
        if let Some(n) = endog_names {
            if n.len() != y.ncols() {
                return Err(err("endog_names must have one name per endog column"));
            }
            m = m.endog_names(&n);
        }
        if let Some(r) = lt_reference {
            if r.len() != y.ncols() {
                return Err(err("lt_reference must have one entry per endog column"));
            }
            m = m.lt_reference(r);
        }
        if let Some(r0) = rot_basis {
            let p = y.ncols() - 1;
            if r0.len() != p || r0.iter().any(|row| row.len() != p) {
                return Err(err(format!("rot_basis must be {p} x {p}")));
            }
            m = m.rot_basis(DMatrix::from_fn(p, p, |r, c| r0[r][c]));
        }
        Ok(Self { inner: m })
    }

    /// Maximum-likelihood fit.
    #[pyo3(signature = (maxiter=None, tol=None, restarts=0, seed=0, cov=true))]
    fn fit(
        &self,
        py: Python<'_>,
        maxiter: Option<usize>,
        tol: Option<f64>,
        restarts: usize,
        seed: u64,
        cov: bool,
    ) -> PyResult<PyResults> {
        let mut cfg = EsagRegressor::new()
            .with_restarts(restarts, seed)
            .with_hessian(cov);
        if let Some(m) = maxiter {
            cfg = cfg.with_max_iter(m);
        }
        if let Some(t) = tol {
            cfg = cfg.with_tol(t);
        }
        let model = self.inner.clone().with_config(cfg);
        let res = py.detach(|| model.fit()).map_err(err)?;
        Ok(PyResults { inner: res })
    }

    /// Penalized fit along a λ path (see `esag.ESAG.fit_regularized`).
    #[pyo3(signature = (penalty="adaptive_lasso", target=None, groups=None, l1_ratio=0.5,
                        r=None, select="bic", folds=10, seed=1, lambdas=None,
                        paper_grid=false, paper=None, custom_groups=None))]
    #[allow(clippy::too_many_arguments)]
    fn fit_regularized(
        &self,
        py: Python<'_>,
        penalty: &str,
        target: Option<&str>,
        groups: Option<&str>,
        l1_ratio: f64,
        r: Option<f64>,
        select: &str,
        folds: usize,
        seed: u64,
        lambdas: Option<Vec<f64>>,
        paper_grid: bool,
        paper: Option<&str>,
        custom_groups: Option<Vec<Vec<(usize, usize)>>>,
    ) -> PyResult<PyRegularized> {
        let mut est = match paper {
            Some(name) => PenalizedEsag::paper(name).map_err(err)?,
            None => {
                let mut e = match penalty.to_ascii_lowercase().replace('-', "_").as_str() {
                    "lasso" | "l1" => PenalizedEsag::lasso(),
                    "adaptive_lasso" | "alasso" => PenalizedEsag::adaptive_lasso(),
                    "group_lasso" | "glasso" => PenalizedEsag::group_lasso(),
                    "adaptive_group_lasso" | "aglasso" => PenalizedEsag::adaptive_group_lasso(),
                    "sparse_group_lasso" | "sglasso" => PenalizedEsag::sparse_group_lasso(l1_ratio),
                    "isotropy" => PenalizedEsag::isotropy(),
                    "eigen_gap" | "eigengap" => PenalizedEsag::eigen_gap(),
                    other => {
                        return Err(err(format!(
                            "unknown penalty {other:?} (lasso, adaptive_lasso, group_lasso, \
                             adaptive_group_lasso, sparse_group_lasso, isotropy, eigen_gap)"
                        )));
                    }
                };
                if let Some(t) = target {
                    e = e.target(match t.to_ascii_lowercase().replace('-', "_").as_str() {
                        "slopes" | "all" => Target::Slopes,
                        "mean" | "mean_slopes" | "mu" => Target::MeanSlopes,
                        "shape_slopes" | "gamma_slopes" => Target::ShapeSlopes,
                        "shape" => Target::Shape,
                        other => {
                            return Err(err(format!(
                                "unknown target {other:?} (slopes, mean, shape_slopes, shape)"
                            )));
                        }
                    });
                }
                if let Some(g) = groups {
                    e = e.grouping(match g.to_ascii_lowercase().replace('-', "_").as_str() {
                        "covariate" | "variable" => Grouping::Covariate,
                        "covariate_part" | "part" => Grouping::CovariatePart,
                        "parameter" | "equation" => Grouping::Parameter,
                        "all" | "one" => Grouping::All,
                        "gamma" | "gamma_groups" => Grouping::GammaGroups,
                        "eigen_parts" | "eigen" => Grouping::EigenParts,
                        other => {
                            return Err(err(format!(
                                "unknown groups {other:?} (covariate, covariate_part, parameter, all, gamma)"
                            )));
                        }
                    });
                }
                if let Some(cg) = custom_groups {
                    e = e.grouping(Grouping::Custom(cg));
                }
                if let Some(r) = r {
                    e = if r == 0.0 { e.plain() } else { e.adaptive(r) };
                }
                e
            }
        };
        if paper.is_some() && r.is_some() {
            return Err(err("r cannot be combined with paper="));
        }
        est = match select.to_ascii_lowercase().as_str() {
            "bic" => est.bic(),
            "aic" => est.aic(),
            "cv" => est.cv(folds, seed),
            other => return Err(err(format!("unknown select {other:?} (bic, aic, cv)"))),
        };
        if let Some(l) = lambdas {
            est = est.lambdas(l);
        } else if paper_grid {
            est = est.with_grid(LambdaGrid::Paper);
        }
        let model = self.inner.clone();
        let path = py.detach(|| model.fit_regularized(est)).map_err(err)?;
        Ok(PyRegularized { inner: path, model })
    }

    /// Number of observations.
    #[getter]
    fn nobs(&self) -> usize {
        self.inner.endog().nrows()
    }
}

/// Names of the rows of `B` (`mu[1..d]`, then `gamma[..]` or `s[..]`).
#[pyfunction]
fn param_rows(d: usize, shape: &str) -> PyResult<Vec<String>> {
    if d < 2 {
        return Err(err("d must be at least 2"));
    }
    let shape = shape_of(shape)?;
    let g = esag::dims::gamma_len(d);
    let mu = (0..d).map(|k| format!("mu[{}]", k + 1));
    if shape == Shape::Spectral {
        // the log-eigenvalues' Helmert coordinates, then the rotation generator
        let p = d - 1;
        return Ok(mu
            .chain((0..p - 1).map(|k| format!("h[{}]", k + 1)))
            .chain((0..g - (p - 1)).map(|k| format!("a[{}]", k + 1)))
            .collect());
    }
    let tag = match shape {
        Shape::Gamma | Shape::GammaTransported | Shape::GammaRot => "gamma",
        Shape::LogTangent => "s",
        Shape::Spectral => unreachable!(),
    };
    Ok(mu.chain((0..g).map(|k| format!("{tag}[{}]", k + 1))).collect())
}

/// Load a fitted model from JSON (as written by `Results.to_json`).
#[pyfunction]
fn results_from_json(
    json: &str,
    endog: PyReadonlyArray2<'_, f64>,
    exog: PyReadonlyArray2<'_, f64>,
) -> PyResult<PyResults> {
    let fit = esag::regression::EsagRegressorFit::from_json(json).map_err(err)?;
    let (y, x) = (to_dmatrix(&endog), to_dmatrix(&exog));
    let model = EsagModel::new(&y, &x)
        .shape(fit.config.shape)
        .has_constant(fit.config.fit_intercept)
        .with_config(fit.config.clone());
    Ok(PyResults {
        inner: EsagResults { fit, model },
    })
}

/// Results of a maximum-likelihood fit.
#[pyclass(name = "Results", module = "esag._esag")]
struct PyResults {
    inner: EsagResults,
}

#[pymethods]
impl PyResults {
    /// Coefficients `B` (parameters × design columns).
    #[getter]
    fn params<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        to_py(py, self.inner.params())
    }
    #[getter]
    fn bse<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        to_py(py, &self.inner.bse())
    }
    #[getter]
    fn zvalues<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        to_py(py, &self.inner.zvalues())
    }
    #[getter]
    fn pvalues<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        to_py(py, &self.inner.pvalues())
    }
    #[pyo3(signature = (alpha=0.05))]
    fn conf_int<'py>(
        &self,
        py: Python<'py>,
        alpha: f64,
    ) -> (Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>) {
        let (lo, hi) = self.inner.conf_int(alpha);
        (to_py(py, &lo), to_py(py, &hi))
    }
    /// Covariance of the free parameters and their `(row, column)` indices.
    #[getter]
    fn cov_params<'py>(&self, py: Python<'py>) -> Option<CovParams<'py>> {
        self.inner
            .cov_params
            .as_ref()
            .map(|c| (to_py(py, c), self.inner.free.clone()))
    }
    #[getter]
    fn row_names(&self) -> Vec<String> {
        self.inner.row_names()
    }
    #[getter]
    fn column_names(&self) -> Vec<String> {
        self.inner.column_names.clone()
    }
    #[getter]
    fn llf(&self) -> f64 {
        self.inner.llf()
    }
    #[getter]
    fn aic(&self) -> f64 {
        self.inner.aic()
    }
    #[getter]
    fn bic(&self) -> f64 {
        self.inner.bic()
    }
    #[getter]
    fn nobs(&self) -> usize {
        self.inner.nobs()
    }
    #[getter]
    fn df_model(&self) -> usize {
        self.inner.df_model()
    }
    #[getter]
    fn df_resid(&self) -> f64 {
        self.inner.df_resid()
    }
    #[getter]
    fn converged(&self) -> bool {
        self.inner.converged
    }
    #[getter]
    fn singular_obs(&self) -> Vec<usize> {
        self.inner.singular_obs.clone()
    }
    #[getter]
    fn degenerate_obs(&self) -> Vec<usize> {
        self.inner.degenerate_obs.clone()
    }
    #[getter]
    fn n_iter(&self) -> usize {
        self.inner.n_iter
    }
    /// Reference direction of the transported frame (`None` for gamma).
    #[getter]
    fn lt_reference(&self) -> Option<Vec<f64>> {
        self.inner.lt_frame().map(|f| f.reference().to_vec())
    }
    /// Basis rotation of the gamma-rot frame, row-major rows (`None` for
    /// the other shapes).
    #[getter]
    fn rot_basis(&self) -> Option<Vec<Vec<f64>>> {
        let basis = self
            .inner
            .gr_frame()
            .and_then(|f| f.basis())
            .or_else(|| self.inner.sp_frame().and_then(|f| f.basis()));
        basis.map(|r0| {
            (0..r0.nrows())
                .map(|r| (0..r0.ncols()).map(|c| r0[(r, c)]).collect())
                .collect()
        })
    }
    /// Whether the fit is a certified non-regular maximum of gamma's
    /// likelihood (observations listed in `singular_obs` on its singular set).
    #[getter]
    fn non_regular(&self) -> bool {
        self.inner.non_regular
    }
    /// Observations set aside by an opt-in `set_aside` fit.
    #[getter]
    fn set_aside_obs(&self) -> Vec<usize> {
        self.inner.set_aside_obs.clone()
    }
    #[pyo3(signature = (alpha=0.05))]
    fn summary(&self, alpha: f64) -> String {
        self.inner.summary_alpha(alpha)
    }
    /// Predicted mean directions for covariates `exog` (`n × d`).
    fn predict<'py>(
        &self,
        py: Python<'py>,
        exog: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let x = to_dmatrix(&exog);
        Ok(to_py(py, &self.inner.predict(&x).map_err(err)?))
    }
    /// Linear predictor `eta = B x̃` for covariates `exog` (`n × b_dim`).
    fn linear_predictor<'py>(
        &self,
        py: Python<'py>,
        exog: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let x = to_dmatrix(&exog);
        Ok(to_py(py, &self.inner.linear_predictor(&x).map_err(err)?))
    }
    /// Per-observation log-likelihood.
    fn loglike_obs<'py>(
        &self,
        py: Python<'py>,
        exog: PyReadonlyArray2<'_, f64>,
        endog: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let (x, y) = (to_dmatrix(&exog), to_dmatrix(&endog));
        let v = self.inner.log_likelihood_rows(&x, &y).map_err(err)?;
        Ok(Array1::from_iter(v.iter().copied()).into_pyarray(py))
    }
    /// Residuals `(Q_i, T1_i)` on the training data (`T1 ≈ χ²_{d−1}`).
    fn resid<'py>(&self, py: Python<'py>) -> PyResult<(Vector<'py>, Vector<'py>)> {
        let r = self
            .inner
            .residuals(self.inner.model.exog(), self.inner.model.endog())
            .map_err(err)?;
        Ok((
            Array1::from_vec(r.q).into_pyarray(py),
            Array1::from_vec(r.t1).into_pyarray(py),
        ))
    }
    /// Bootstrap hypothesis test.
    #[pyo3(signature = (null="isotropy", feature=None, n_boot=500, seed=1))]
    fn test(
        &self,
        py: Python<'_>,
        null: &str,
        feature: Option<&str>,
        n_boot: usize,
        seed: u64,
    ) -> PyResult<PyTest> {
        let restriction = self.restriction(null, feature)?;
        let res = &self.inner;
        let t = py
            .detach(|| res.test(restriction, n_boot, seed))
            .map_err(err)?;
        Ok(PyTest { inner: t })
    }
    /// Bootstrap score test (restricted refits only: LM and M).
    #[pyo3(signature = (null, feature=None, n_boot=500, seed=1))]
    fn score_test(
        &self,
        py: Python<'_>,
        null: &str,
        feature: Option<&str>,
        n_boot: usize,
        seed: u64,
    ) -> PyResult<PyTest> {
        let restriction = self.restriction(null, feature)?;
        let res = &self.inner;
        let t = py
            .detach(|| res.score_test(restriction, n_boot, seed))
            .map_err(err)?;
        Ok(PyTest { inner: t })
    }
    /// Bootstrap goodness-of-fit test (Algorithm 1): dict of results.
    #[pyo3(signature = (n_boot=200, seed=1, one_sample=false))]
    fn gof<'py>(
        &self,
        py: Python<'py>,
        n_boot: usize,
        seed: u64,
        one_sample: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        let res = &self.inner;
        let method = if one_sample {
            GofMethod::OneSampleChiSquare
        } else {
            GofMethod::TwoSampleReference
        };
        let g = py
            .detach(|| {
                bootstrap_gof(
                    &res.fit,
                    res.model.exog(),
                    res.model.endog(),
                    method,
                    n_boot,
                    seed,
                )
            })
            .map_err(err)?;
        let d = PyDict::new(py);
        d.set_item("ks_statistic", g.ks_statistic)?;
        d.set_item("ks_pvalue", g.ks_p_value)?;
        d.set_item("pvalue", g.p_value)?;
        d.set_item("n_boot", g.n_boot)?;
        d.set_item("n_failed", g.n_failed)?;
        Ok(d)
    }
    /// Prediction-region thresholds `q` at covariates `x` (one per level).
    #[pyo3(signature = (x, levels=vec![0.95, 0.9], n_mc=5000, n_boot=100, seed=1))]
    fn prediction_region<'py>(
        &self,
        py: Python<'py>,
        x: Vec<f64>,
        levels: Vec<f64>,
        n_mc: usize,
        n_boot: usize,
        seed: u64,
    ) -> PyResult<Bound<'py, PyDict>> {
        let res = &self.inner;
        let pr = py
            .detach(|| res.prediction_region(&x, &levels, n_mc, n_boot, seed))
            .map_err(err)?;
        let d = PyDict::new(py);
        d.set_item("levels", levels)?;
        d.set_item("thresholds", pr.thresholds.clone())?;
        d.set_item("mean_direction", pr.dist.mean_direction())?;
        Ok(d)
    }
    /// Responses simulated from the fitted model at covariates `exog`.
    #[pyo3(signature = (exog, seed=0))]
    fn simulate<'py>(
        &self,
        py: Python<'py>,
        exog: PyReadonlyArray2<'_, f64>,
        seed: u64,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let x = to_dmatrix(&exog);
        let mut rng = esag::FastRng::seed_from_u64(seed);
        Ok(to_py(
            py,
            &self.inner.sample_responses(&x, &mut rng).map_err(err)?,
        ))
    }
    /// The fitted model as JSON.
    fn to_json(&self) -> PyResult<String> {
        self.inner.to_json().map_err(err)
    }
    /// `V̂(x)` for each row of `exog`: an `(m, d, d)` array.
    fn covariance<'py>(
        &self,
        py: Python<'py>,
        exog: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, numpy::PyArray3<f64>>> {
        let x = to_dmatrix(&exog);
        let vs = self.inner.predict_covariance(&x).map_err(err)?;
        let d = self.inner.d;
        Ok(Array3::from_shape_fn((vs.len(), d, d), |(i, a, b)| vs[i][(a, b)]).into_pyarray(py))
    }
    /// `V̂(x)` with pairs-bootstrap intervals at each row of `exog`: a list
    /// of dicts.
    #[pyo3(signature = (exog, n_boot=500, level=0.95, seed=1))]
    fn covariance_ci<'py>(
        &self,
        py: Python<'py>,
        exog: PyReadonlyArray2<'_, f64>,
        n_boot: usize,
        level: f64,
        seed: u64,
    ) -> PyResult<Vec<Bound<'py, PyDict>>> {
        let x = to_dmatrix(&exog);
        let res = &self.inner;
        let est = py
            .detach(|| res.covariance(&x, n_boot, level, seed))
            .map_err(err)?;
        est.iter()
            .map(|e| {
                let d = PyDict::new(py);
                d.set_item("x", e.x.clone())?;
                d.set_item("covariance", to_py(py, &e.covariance))?;
                d.set_item("mean_direction", e.mean_direction.clone())?;
                d.set_item("eigenvalues", e.eigenvalues.clone())?;
                d.set_item("axes", to_py(py, &e.axes))?;
                d.set_item("eig_lower", e.eig_lower.clone())?;
                d.set_item("eig_upper", e.eig_upper.clone())?;
                d.set_item("eig_se", e.eig_se.clone())?;
                d.set_item("entry_se", to_py(py, &e.entry_se))?;
                d.set_item("entry_lower", to_py(py, &e.entry_lower))?;
                d.set_item("entry_upper", to_py(py, &e.entry_upper))?;
                d.set_item("radius", e.radius)?;
                d.set_item("level", e.level)?;
                d.set_item("n_boot", e.n_boot)?;
                d.set_item("summary", e.summary())?;
                Ok(d)
            })
            .collect()
    }
}

/// Results of a bootstrap hypothesis test.
impl PyResults {
    fn restriction(&self, null: &str, feature: Option<&str>) -> PyResult<Restriction> {
        let j = match feature {
            Some(f) => Some(match f.parse::<usize>() {
                Ok(i) => i,
                Err(_) => self.inner.feature_index(f).map_err(err)?,
            }),
            None => None,
        };
        let need = |j: Option<usize>| j.ok_or_else(|| err(format!("null {null:?} needs feature=")));
        Ok(match null.to_ascii_lowercase().replace('-', "_").as_str() {
            "isotropy" | "isotropic" => Restriction::Isotropy,
            "mu" | "mean" => Restriction::MuOfFeature(need(j)?),
            "shape" | "gamma" => Restriction::GammaOfFeature(need(j)?),
            "feature" | "covariate" | "both" => Restriction::Feature(need(j)?),
            "constant_shape" | "shape_constant" => Restriction::ConstantShape,
            other => {
                return Err(err(format!(
                    "unknown null {other:?} (isotropy, mu, shape, feature, constant_shape)"
                )));
            }
        })
    }
}

#[pyclass(name = "TestResult", module = "esag._esag")]
struct PyTest {
    inner: HypothesisTestResult,
}

#[pymethods]
impl PyTest {
    fn summary(&self) -> String {
        self.inner.summary()
    }
    #[getter]
    fn null(&self) -> String {
        self.inner.null.clone()
    }
    /// `{"RoC": …, "D": …, "M": …, "LR": …, "LM": …}`
    #[getter]
    fn statistic<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("RoC", self.inner.roc)?;
        d.set_item("D", self.inner.d_stat)?;
        d.set_item("M", self.inner.m_stat)?;
        d.set_item("LR", self.inner.lr)?;
        d.set_item("LM", self.inner.lm)?;
        Ok(d)
    }
    /// Bootstrap p-values (and `"LR_chi2"`, `"LM_chi2"`, the asymptotic
    /// ones; `NaN` where a score test has none).
    #[getter]
    fn pvalue<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("RoC", self.inner.p_roc)?;
        d.set_item("D", self.inner.p_d)?;
        d.set_item("M", self.inner.p_m)?;
        d.set_item("LR", self.inner.p_lr)?;
        d.set_item("LR_chi2", self.inner.p_lr_chisq)?;
        d.set_item("LM", self.inner.p_lm)?;
        d.set_item("LM_chi2", self.inner.p_lm_chisq)?;
        Ok(d)
    }
    #[getter]
    fn df(&self) -> usize {
        self.inner.df
    }
    #[getter]
    fn n_boot(&self) -> usize {
        self.inner.n_boot
    }
    #[getter]
    fn n_failed(&self) -> usize {
        self.inner.n_failed
    }
    #[getter]
    fn n_unconverged(&self) -> usize {
        self.inner.n_unconverged
    }
    #[getter]
    fn llf_restricted(&self) -> f64 {
        self.inner.loglik_restricted
    }
    #[getter]
    fn llf_full(&self) -> f64 {
        self.inner.loglik_full
    }
    /// Bootstrap replicates `{"RoC": [...], ...}`.
    #[getter]
    fn bootstrap<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("RoC", self.inner.boot_roc.clone())?;
        d.set_item("D", self.inner.boot_d.clone())?;
        d.set_item("M", self.inner.boot_m.clone())?;
        d.set_item("LR", self.inner.boot_lr.clone())?;
        d.set_item("LM", self.inner.boot_lm.clone())?;
        Ok(d)
    }
}

/// Results of a penalized fit along a λ path.
#[pyclass(name = "RegularizedResults", module = "esag._esag")]
struct PyRegularized {
    inner: PenalizedPath,
    model: EsagModel,
}

#[pymethods]
impl PyRegularized {
    fn summary(&self) -> String {
        self.inner.summary()
    }
    fn path_summary(&self) -> String {
        self.inner.path_summary()
    }
    /// Coefficients at the selected λ (penalized estimate).
    #[getter]
    fn params<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        to_py(py, &self.inner.best().coef)
    }
    #[getter]
    fn row_names(&self) -> Vec<String> {
        self.inner.initial.row_names()
    }
    #[getter]
    fn column_names(&self) -> Vec<String> {
        self.inner.initial.column_names.clone()
    }
    #[getter]
    fn method(&self) -> String {
        self.inner.config.method_name()
    }
    #[getter]
    fn selected(&self) -> usize {
        self.inner.selected
    }
    /// λ values (sum scale).
    #[getter]
    fn lambdas(&self) -> Vec<f64> {
        self.inner.points.iter().map(|p| p.lambda).collect()
    }
    /// `alpha = λ/n` (mean scale, as statsmodels' `fit_regularized`).
    #[getter]
    fn alphas(&self) -> Vec<f64> {
        self.inner.points.iter().map(|p| p.lambda_n).collect()
    }
    #[getter]
    fn criterion(&self) -> Vec<f64> {
        self.inner.criterion.clone()
    }
    #[getter]
    fn llf_path(&self) -> Vec<f64> {
        self.inner.points.iter().map(|p| p.log_likelihood).collect()
    }
    #[getter]
    fn df_path(&self) -> Vec<f64> {
        self.inner.points.iter().map(|p| p.df).collect()
    }
    /// Penalty atoms (groups / entries / gaps) active at each λ.
    #[getter]
    fn active_path(&self) -> Vec<Vec<bool>> {
        self.inner.points.iter().map(|p| p.active.clone()).collect()
    }
    /// The coefficients at every λ (list of arrays).
    fn params_path<'py>(&self, py: Python<'py>) -> Vec<Bound<'py, PyArray2<f64>>> {
        self.inner
            .points
            .iter()
            .map(|p| to_py(py, &p.coef))
            .collect()
    }
    /// The relaxed (unpenalized) refit on the selected model, with standard
    /// errors, as a `Results` (`None` when the selected model is not a
    /// coefficient mask).
    #[getter]
    fn relaxed(&self) -> Option<PyResults> {
        self.inner.relaxed.as_ref().map(|f| PyResults {
            inner: EsagResults {
                fit: f.clone(),
                model: self.model.clone(),
            },
        })
    }
}

/// A single ESAG distribution `ESAG(μ, γ)`.
#[pyclass(name = "Distribution", module = "esag._esag")]
struct PyDistribution {
    inner: Esag,
}

#[pymethods]
impl PyDistribution {
    #[new]
    #[pyo3(signature = (mu, gamma=None))]
    fn new(mu: Vec<f64>, gamma: Option<Vec<f64>>) -> PyResult<Self> {
        let g = gamma.unwrap_or_else(|| vec![0.0; esag::dims::gamma_len(mu.len())]);
        Ok(Self {
            inner: Esag::new(&mu, &g).map_err(err)?,
        })
    }
    /// `n` draws (`n × d`).
    #[pyo3(signature = (n, seed=0))]
    fn sample<'py>(&self, py: Python<'py>, n: usize, seed: u64) -> Bound<'py, PyArray2<f64>> {
        let mut rng = esag::FastRng::seed_from_u64(seed);
        to_py(py, &self.inner.sample(n, &mut rng))
    }
    /// Log-density at each row of `y`.
    fn logpdf<'py>(
        &self,
        py: Python<'py>,
        y: PyReadonlyArray2<'_, f64>,
    ) -> Bound<'py, PyArray1<f64>> {
        let m = to_dmatrix(&y);
        let v = self.inner.log_pdf_rows(&m);
        Array1::from_iter(v.iter().copied()).into_pyarray(py)
    }
    #[getter]
    fn covariance<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        to_py(py, &self.inner.covariance())
    }
    #[getter]
    fn mean_direction(&self) -> Vec<f64> {
        self.inner.mean_direction()
    }
    #[getter]
    fn mu(&self) -> Vec<f64> {
        self.inner.mu().to_vec()
    }
    #[getter]
    fn gamma(&self) -> Vec<f64> {
        self.inner.gamma().to_vec()
    }
    /// Exact `E[YYᵀ]`.
    #[getter]
    fn second_moment<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        to_py(py, &self.inner.second_moment())
    }
}

#[pymodule]
fn _esag(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyModel>()?;
    m.add_class::<PyResults>()?;
    m.add_class::<PyTest>()?;
    m.add_class::<PyRegularized>()?;
    m.add_class::<PyDistribution>()?;
    m.add_function(wrap_pyfunction!(param_rows, m)?)?;
    m.add_function(wrap_pyfunction!(results_from_json, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
