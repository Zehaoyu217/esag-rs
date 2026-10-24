//! ESAG regression with a scikit-learn-style API.
//!
//! Model: for observation `i` with covariate vector `x_i ∈ R^p`, the response
//! `y_i ∈ S^{d-1}` follows `ESAG(mu_i, gamma_i)` where the stacked natural
//! parameter `eta_i = (mu_i, gamma_i) = B x̃_i` is linear in
//! `x̃_i = (1, x_i)` (intercept) — exactly the model fitted by `MLE()` in the
//! original R code. `B` is a `(d + (d-2)(d+1)/2) × (p+1)` coefficient matrix.
//!
//! ```no_run
//! use esag::regression::EsagRegressor;
//! # let (x, y) = (nalgebra::DMatrix::<f64>::zeros(1, 1), nalgebra::DMatrix::<f64>::zeros(1, 3));
//! let model = EsagRegressor::new().with_max_iter(500).fit(&x, &y).unwrap();
//! let mean_directions = model.predict(&x);        // n × d unit vectors
//! let dists = model.predict_distribution(&x);     // Vec<Esag>
//! println!("{}", model.summary());
//! ```
//!
//! Fitting maximises the joint log-likelihood with L-BFGS using *exact*
//! gradients (forward-mode automatic differentiation of the per-observation
//! log-density, then the chain rule `∂LL/∂B = Σ_i g_i x̃_iᵀ`), parallelised over
//! observations with rayon. Standard errors come from the observed information
//! (finite-difference Hessian of the exact gradient).

use crate::dims;
use crate::distribution::{Esag, residual_q};
use crate::error::{EsagError, Result};
use crate::frame::Frame;
use crate::loglik;
use crate::optim::hessian::hessian_fd;
use crate::optim::lbfgs::{LbfgsConfig, Objective, OptimResult, Status, minimize};
use crate::stats::normal::norm_sf;
use nalgebra::{DMatrix, DVector};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// How the optimiser is initialised.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Init {
    /// Default: run from both [`Init::DataDriven`] and [`Init::RStyle`] and keep
    /// the higher likelihood; if neither passes the convergence check, retry
    /// from randomly perturbed starts.
    Auto,
    /// Intercept `mu` along the sample mean direction with
    /// `|mu| = sqrt(kappa)`, `kappa` a von-Mises–Fisher-style concentration
    /// estimate (for an angular Gaussian `kappa ≈ |mu|²`); small `gamma` (0.01,
    /// to step off the non-differentiable isotropic point); zero slopes.
    DataDriven,
    /// The R code's recommendation: `mu` entries 1, `gamma` entries (near) 0,
    /// for every column. (We use 1e-3 instead of exactly 0 for `gamma`.)
    RStyle,
    /// A user-supplied full coefficient matrix `B` (`eta_len(d) × p_total`).
    Custom(DMatrix<f64>),
}

/// Hyper-parameters of the ESAG regression estimator (scikit-learn style:
/// configure, then call [`EsagRegressor::fit`]).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EsagRegressor {
    /// Add an intercept column of ones to `X` (default `true`).
    pub fit_intercept: bool,
    /// Maximum L-BFGS iterations (default 2000).
    pub max_iter: usize,
    /// Gradient tolerance (infinity norm) for convergence (default 1e-6).
    pub tol: f64,
    /// Initialisation strategy.
    pub init: Init,
    /// Number of additional randomly perturbed starts (default 0); the best
    /// optimum is kept.
    pub n_restarts: usize,
    /// Seed for the random restarts.
    pub seed: u64,
    /// Compute the observed-information Hessian (standard errors) (default true).
    pub compute_hessian: bool,
    /// L-BFGS memory (default 10).
    pub lbfgs_memory: usize,
    /// Print optimisation progress to stderr.
    pub verbose: bool,
    /// Optional free-parameter mask (`eta_len(d) × p_total`); `false` entries
    /// are held at their initial value (0 unless `Init::Custom`). Used for
    /// restricted fits in hypothesis tests.
    pub mask: Option<DMatrix<bool>>,
    /// Optional feature names for `summary()`.
    pub feature_names: Option<Vec<String>>,
}

impl Default for EsagRegressor {
    fn default() -> Self {
        Self::new()
    }
}

impl EsagRegressor {
    /// Default configuration.
    pub fn new() -> Self {
        Self {
            fit_intercept: true,
            max_iter: 2000,
            tol: 1e-6,
            init: Init::Auto,
            n_restarts: 0,
            seed: 0,
            compute_hessian: true,
            lbfgs_memory: 10,
            verbose: false,
            mask: None,
            feature_names: None,
        }
    }
    /// Whether to add an intercept column.
    pub fn with_fit_intercept(mut self, v: bool) -> Self {
        self.fit_intercept = v;
        self
    }
    /// Maximum number of iterations.
    pub fn with_max_iter(mut self, v: usize) -> Self {
        self.max_iter = v;
        self
    }
    /// Gradient tolerance.
    pub fn with_tol(mut self, v: f64) -> Self {
        self.tol = v;
        self
    }
    /// Initialisation strategy.
    pub fn with_init(mut self, v: Init) -> Self {
        self.init = v;
        self
    }
    /// Random restarts.
    pub fn with_restarts(mut self, n: usize, seed: u64) -> Self {
        self.n_restarts = n;
        self.seed = seed;
        self
    }
    /// Whether to compute the Hessian / standard errors.
    pub fn with_hessian(mut self, v: bool) -> Self {
        self.compute_hessian = v;
        self
    }
    /// Verbose optimisation output.
    pub fn with_verbose(mut self, v: bool) -> Self {
        self.verbose = v;
        self
    }
    /// Free-parameter mask for restricted fits.
    pub fn with_mask(mut self, mask: Option<DMatrix<bool>>) -> Self {
        self.mask = mask;
        self
    }
    /// Feature names (length = number of columns of `X`, excluding intercept).
    pub fn with_feature_names(mut self, names: Vec<String>) -> Self {
        self.feature_names = Some(names);
        self
    }

    /// Fit the model. `x` is `n × p` (no intercept column; may have `p = 0`),
    /// `y` is `n × d` with one unit vector per row (rows are re-normalised).
    pub fn fit(&self, x: &DMatrix<f64>, y: &DMatrix<f64>) -> Result<EsagRegressorFit> {
        let data = RegressionData::new(x, y, self.fit_intercept)?;
        let mask = self.resolve_mask(&data)?;
        let starts: Vec<Init> = match &self.init {
            Init::Auto => vec![Init::DataDriven, Init::RStyle],
            other => vec![other.clone()],
        };
        let mut best: Option<EsagRegressorFit> = None;
        for init in starts {
            let mut b0 = self.clone().with_init(init.clone()).initial_coef(&data)?;
            if !matches!(init, Init::Custom(_)) {
                // fixed (masked-out) coefficients are held at zero
                for (v, &free) in b0.iter_mut().zip(mask.iter()) {
                    if !free {
                        *v = 0.0;
                    }
                }
            }
            let fit = self.fit_from(&data, b0, mask.clone())?;
            if best.as_ref().is_none_or(|b| better(&fit, b)) {
                best = Some(fit);
            }
        }
        let mut best = best.expect("at least one start");
        // Safety net: perturbed restarts if no start passed the convergence check.
        if !best.converged && matches!(self.init, Init::Auto) && self.n_restarts == 0 {
            let mut retry = self.clone();
            retry.init = Init::Custom(best.coef.clone());
            retry.n_restarts = 6;
            retry.seed = self.seed.wrapping_add(0x5EED);
            let fit = retry.fit_from(&data, best.coef.clone(), mask)?;
            if better(&fit, &best) {
                best = fit;
                best.config.init = Init::Auto;
            }
        }
        Ok(best)
    }

    /// Fit starting from an explicit coefficient matrix (warm start). Used by
    /// bootstrap procedures; `b0` must be `eta_len(d) × p_total`.
    pub fn fit_warm(
        &self,
        x: &DMatrix<f64>,
        y: &DMatrix<f64>,
        b0: &DMatrix<f64>,
    ) -> Result<EsagRegressorFit> {
        let data = RegressionData::new(x, y, self.fit_intercept)?;
        if b0.nrows() != data.b_dim || b0.ncols() != data.p_total {
            return Err(EsagError::Shape(format!(
                "warm start must be {}×{}, got {}×{}",
                data.b_dim,
                data.p_total,
                b0.nrows(),
                b0.ncols()
            )));
        }
        let mask = self.resolve_mask(&data)?;
        self.fit_from(&data, b0.clone(), mask)
    }

    fn resolve_mask(&self, data: &RegressionData) -> Result<DMatrix<bool>> {
        match &self.mask {
            None => Ok(DMatrix::from_element(data.b_dim, data.p_total, true)),
            Some(m) => {
                if m.nrows() != data.b_dim || m.ncols() != data.p_total {
                    return Err(EsagError::Shape(format!(
                        "mask must be {}×{}, got {}×{}",
                        data.b_dim,
                        data.p_total,
                        m.nrows(),
                        m.ncols()
                    )));
                }
                if !m.iter().any(|&f| f) {
                    return Err(EsagError::Shape("mask has no free parameters".into()));
                }
                Ok(m.clone())
            }
        }
    }

    fn initial_coef(&self, data: &RegressionData) -> Result<DMatrix<f64>> {
        let (d, b_dim, p_total) = (data.d, data.b_dim, data.p_total);
        let mut b = DMatrix::zeros(b_dim, p_total);
        match &self.init {
            Init::Custom(m) => {
                if m.nrows() != b_dim || m.ncols() != p_total {
                    return Err(EsagError::Shape(format!(
                        "Init::Custom must be {b_dim}×{p_total}, got {}×{}",
                        m.nrows(),
                        m.ncols()
                    )));
                }
                b.copy_from(m);
            }
            Init::RStyle => {
                for j in 0..p_total {
                    for k in 0..d {
                        b[(k, j)] = 1.0;
                    }
                    for k in d..b_dim {
                        b[(k, j)] = 1e-3;
                    }
                }
            }
            Init::DataDriven | Init::Auto => {
                // mean resultant vector
                let n = data.n as f64;
                let mut ybar = vec![0.0; d];
                for i in 0..data.n {
                    for k in 0..d {
                        ybar[k] += data.y[i * d + k];
                    }
                }
                for v in ybar.iter_mut() {
                    *v /= n;
                }
                let rbar = ybar
                    .iter()
                    .map(|v| v * v)
                    .sum::<f64>()
                    .sqrt()
                    .clamp(1e-3, 0.999);
                // vMF-style concentration (Banerjee et al. 2005) as a scale for |mu|
                let kappa =
                    (rbar * (d as f64 - rbar * rbar) / (1.0 - rbar * rbar)).clamp(0.25, 2500.0);
                let scale = kappa.sqrt();
                for k in 0..d {
                    b[(k, 0)] = ybar[k] / rbar * scale;
                }
                for k in d..b_dim {
                    b[(k, 0)] = 0.01;
                }
            }
        }
        Ok(b)
    }

    fn fit_from(
        &self,
        data: &RegressionData,
        b0: DMatrix<f64>,
        mask: DMatrix<bool>,
    ) -> Result<EsagRegressorFit> {
        let free: Vec<(usize, usize)> = (0..data.p_total)
            .flat_map(|j| (0..data.b_dim).map(move |k| (k, j)))
            .filter(|&(k, j)| mask[(k, j)])
            .collect();
        let obj = NegLogLik {
            data,
            base: b0.clone(),
            free: &free,
        };
        let cfg = LbfgsConfig {
            memory: self.lbfgs_memory,
            max_iter: self.max_iter,
            tol_grad: self.tol,
            verbose: self.verbose,
            ..LbfgsConfig::default()
        };
        let x0: Vec<f64> = free.iter().map(|&(k, j)| b0[(k, j)]).collect();
        let mut best: OptimResult = minimize(&obj, &x0, &cfg);
        if self.n_restarts > 0 {
            let mut rng = ChaCha8Rng::seed_from_u64(self.seed);
            for _ in 0..self.n_restarts {
                let xs: Vec<f64> = free
                    .iter()
                    .zip(&x0)
                    .map(|(&(k, _), &v)| {
                        let sd = if k < data.d { 0.5 } else { 0.25 };
                        let z: f64 = StandardNormal.sample(&mut rng);
                        v + sd * z
                    })
                    .collect();
                let r = minimize(&obj, &xs, &cfg);
                if r.f.is_finite() && r.f < best.f {
                    best = r;
                }
            }
        }
        if !best.f.is_finite() {
            return Err(EsagError::Convergence(
                "objective is not finite at the optimiser's result".into(),
            ));
        }
        // Pick the canonical member of the {B, B with gamma group 1 negated}
        // equivalence class (identical likelihood, see `canonicalize_coef`).
        let mut coef = obj.assemble(&best.x);
        if canonicalize_coef(&mut coef, data.d) {
            for (xi, &(k, _)) in best.x.iter_mut().zip(&free) {
                if k == data.d || k == data.d + 1 {
                    *xi = -*xi;
                }
            }
            for (gi, &(k, _)) in best.grad.iter_mut().zip(&free) {
                if k == data.d || k == data.d + 1 {
                    *gi = -*gi;
                }
            }
        }
        let converged =
            best.converged() && best.grad_norm_inf <= MAX_MEAN_GRADIENT * data.n.max(1) as f64;

        let (hessian, cov_params) = if self.compute_hessian {
            let h = hessian_fd(&obj, &best.x, None);
            let nf = free.len();
            let hm = DMatrix::from_row_slice(nf, nf, &h);
            let cov = hm.clone().try_inverse();
            (Some(hm), cov)
        } else {
            (None, None)
        };

        let feature_names = self.param_column_names(data);
        Ok(EsagRegressorFit {
            d: data.d,
            n_features: data.p,
            fit_intercept: self.fit_intercept,
            n_obs: data.n,
            coef,
            mask,
            free,
            log_likelihood: -best.f,
            n_iter: best.n_iter,
            n_fev: best.n_fev,
            status: best.status,
            converged,
            grad_norm: best.grad_norm_inf,
            hessian,
            cov_params,
            column_names: feature_names,
            config: self.clone(),
        })
    }

    fn param_column_names(&self, data: &RegressionData) -> Vec<String> {
        let mut names = Vec::with_capacity(data.p_total);
        if self.fit_intercept {
            names.push("(Intercept)".to_string());
        }
        for j in 0..data.p {
            let nm = self
                .feature_names
                .as_ref()
                .and_then(|v| v.get(j).cloned())
                .unwrap_or_else(|| format!("x{}", j + 1));
            names.push(nm);
        }
        names
    }
}

/// Prefer converged fits, then higher likelihood.
fn better(a: &EsagRegressorFit, b: &EsagRegressorFit) -> bool {
    match (a.converged, b.converged) {
        (true, false) => true,
        (false, true) => false,
        _ => a.log_likelihood > b.log_likelihood,
    }
}

/// Mean per-observation gradient (infinity norm) below which an optimiser
/// stop counts as convergence. Guards against "function converged" stops at
/// singular points of the parameterization (coinciding eigenvalues), where
/// the likelihood is not differentiable.
const MAX_MEAN_GRADIENT: f64 = 1e-3;

/// Validated, flattened training data.
pub(crate) struct RegressionData {
    pub n: usize,
    pub d: usize,
    /// number of covariates (excluding intercept)
    pub p: usize,
    /// number of design columns (incl. intercept)
    pub p_total: usize,
    pub b_dim: usize,
    /// n × p_total row-major design matrix
    pub x: Vec<f64>,
    /// n × d row-major unit vectors
    pub y: Vec<f64>,
}

impl RegressionData {
    pub(crate) fn new(x: &DMatrix<f64>, y: &DMatrix<f64>, intercept: bool) -> Result<Self> {
        let n = y.nrows();
        let d = y.ncols();
        if d < 3 {
            return Err(EsagError::Dimension(format!(
                "Y must have d >= 3 columns, got {d}"
            )));
        }
        if n == 0 {
            return Err(EsagError::Data("no observations".into()));
        }
        if x.nrows() != n {
            return Err(EsagError::Shape(format!(
                "X has {} rows but Y has {n}",
                x.nrows()
            )));
        }
        let p = x.ncols();
        let p_total = p + usize::from(intercept);
        if p_total == 0 {
            return Err(EsagError::Shape(
                "no design columns: X has 0 columns and fit_intercept is false".into(),
            ));
        }
        let mut xf = Vec::with_capacity(n * p_total);
        let mut yf = Vec::with_capacity(n * d);
        for i in 0..n {
            if intercept {
                xf.push(1.0);
            }
            for j in 0..p {
                let v = x[(i, j)];
                if !v.is_finite() {
                    return Err(EsagError::Data(format!("X[{i},{j}] is not finite")));
                }
                xf.push(v);
            }
            let mut norm = 0.0;
            for k in 0..d {
                let v = y[(i, k)];
                if !v.is_finite() {
                    return Err(EsagError::Data(format!("Y[{i},{k}] is not finite")));
                }
                norm += v * v;
            }
            let norm = norm.sqrt();
            if norm < 1e-12 {
                return Err(EsagError::Data(format!("Y row {i} is the zero vector")));
            }
            for k in 0..d {
                yf.push(y[(i, k)] / norm);
            }
        }
        Ok(Self {
            n,
            d,
            p,
            p_total,
            b_dim: dims::eta_len(d),
            x: xf,
            y: yf,
        })
    }
}

/// Negative joint log-likelihood as a function of the free coefficients.
struct NegLogLik<'a> {
    data: &'a RegressionData,
    /// full coefficient matrix providing the fixed (masked-out) entries
    base: DMatrix<f64>,
    free: &'a [(usize, usize)],
}

impl NegLogLik<'_> {
    fn assemble(&self, params: &[f64]) -> DMatrix<f64> {
        let mut b = self.base.clone();
        for (&(k, j), &v) in self.free.iter().zip(params) {
            b[(k, j)] = v;
        }
        b
    }
}

/// Minimum number of observations per rayon chunk.
const PAR_CHUNK: usize = 64;

impl Objective for NegLogLik<'_> {
    fn value_and_gradient(&self, params: &[f64], grad: &mut [f64]) -> f64 {
        let b = self.assemble(params);
        let (f, g_full) = joint_loglik_and_grad(self.data, &b);
        for (gi, &(k, j)) in grad.iter_mut().zip(self.free) {
            *gi = -g_full[(k, j)];
        }
        -f
    }
}

/// Joint log-likelihood `Σ_i log f(y_i; B x̃_i)` and its gradient with respect
/// to the full coefficient matrix `B` (`b_dim × p_total`).
pub(crate) fn joint_loglik_and_grad(
    data: &RegressionData,
    b: &DMatrix<f64>,
) -> (f64, DMatrix<f64>) {
    let (d, bd, pt) = (data.d, data.b_dim, data.p_total);
    // B as row-major slice for fast eta_i = B x_i
    let b_rm: Vec<f64> = (0..bd)
        .flat_map(|k| (0..pt).map(move |j| b[(k, j)]))
        .collect();
    let body = |rows: std::ops::Range<usize>| -> (f64, Vec<f64>) {
        let mut f = 0.0;
        let mut g = vec![0.0; bd * pt];
        let mut eta = vec![0.0; bd];
        let mut ge = vec![0.0; bd];
        for i in rows {
            let xi = &data.x[i * pt..(i + 1) * pt];
            let yi = &data.y[i * d..(i + 1) * d];
            for k in 0..bd {
                let row = &b_rm[k * pt..(k + 1) * pt];
                eta[k] = row.iter().zip(xi).map(|(a, b)| a * b).sum();
            }
            let ll = loglik::loglik_eta_grad(&eta, yi, &mut ge);
            f += ll;
            for k in 0..bd {
                let gk = ge[k];
                let grow = &mut g[k * pt..(k + 1) * pt];
                for j in 0..pt {
                    grow[j] += gk * xi[j];
                }
            }
        }
        (f, g)
    };
    let (f, g) = if data.n >= 2 * PAR_CHUNK {
        let n_chunks = data.n.div_ceil(PAR_CHUNK);
        (0..n_chunks)
            .into_par_iter()
            .map(|c| body(c * PAR_CHUNK..((c + 1) * PAR_CHUNK).min(data.n)))
            .reduce(
                || (0.0, vec![0.0; bd * pt]),
                |(fa, mut ga), (fb, gb)| {
                    for (a, b) in ga.iter_mut().zip(&gb) {
                        *a += b;
                    }
                    (fa + fb, ga)
                },
            )
    } else {
        body(0..data.n)
    };
    (f, DMatrix::from_row_slice(bd, pt, &g))
}

/// Residual diagnostics for a fitted model.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Residuals {
    /// `Q_i = r_iᵀ V_i⁻¹ r_i`, `r_i = (I − μ̂_i μ̂_iᵀ) y_i`.
    pub q: Vec<f64>,
    /// `T1_i = (|mu_i|² + Σ lambda_i + 1) Q_i`, approximately χ²_{d−1}.
    pub t1: Vec<f64>,
}

/// A fitted ESAG regression model.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EsagRegressorFit {
    /// Ambient dimension of the response.
    pub d: usize,
    /// Number of covariates (excluding intercept).
    pub n_features: usize,
    /// Whether an intercept column was added.
    pub fit_intercept: bool,
    /// Number of training observations.
    pub n_obs: usize,
    /// Full coefficient matrix `B` (`eta_len(d) × p_total`); column 0 is the
    /// intercept when `fit_intercept`. Rows: `mu_1..mu_d, gamma_1..gamma_g`.
    pub coef: DMatrix<f64>,
    /// Free-parameter mask used in the fit.
    pub mask: DMatrix<bool>,
    /// Row/column indices of the free parameters, in optimiser order.
    pub free: Vec<(usize, usize)>,
    /// Maximised joint log-likelihood.
    pub log_likelihood: f64,
    /// L-BFGS iterations.
    pub n_iter: usize,
    /// Objective evaluations.
    pub n_fev: usize,
    /// Optimiser termination status.
    pub status: Status,
    /// Whether the optimiser reported convergence.
    pub converged: bool,
    /// Infinity norm of the gradient at the solution.
    pub grad_norm: f64,
    /// Observed information (Hessian of the negative log-likelihood) over the
    /// free parameters, if computed.
    pub hessian: Option<DMatrix<f64>>,
    /// Asymptotic covariance of the free parameters (inverse Hessian).
    pub cov_params: Option<DMatrix<f64>>,
    /// Design column names.
    pub column_names: Vec<String>,
    /// The estimator configuration used.
    pub config: EsagRegressor,
}

impl EsagRegressorFit {
    /// Number of design columns (intercept included).
    pub fn p_total(&self) -> usize {
        self.coef.ncols()
    }
    /// Length of `eta = (mu, gamma)`.
    pub fn b_dim(&self) -> usize {
        self.coef.nrows()
    }
    /// Number of free parameters.
    pub fn n_params(&self) -> usize {
        self.free.len()
    }
    /// Intercept column (`eta` at `x = 0`), if the model has one.
    pub fn intercept(&self) -> Option<DVector<f64>> {
        self.fit_intercept.then(|| self.coef.column(0).into_owned())
    }
    /// Slope coefficients (`b_dim × n_features`).
    pub fn slopes(&self) -> DMatrix<f64> {
        let start = usize::from(self.fit_intercept);
        self.coef.columns(start, self.n_features).into_owned()
    }
    /// Akaike information criterion.
    pub fn aic(&self) -> f64 {
        2.0 * self.n_params() as f64 - 2.0 * self.log_likelihood
    }
    /// Bayesian information criterion.
    pub fn bic(&self) -> f64 {
        (self.n_obs as f64).ln() * self.n_params() as f64 - 2.0 * self.log_likelihood
    }

    /// Row-major design row for one covariate vector.
    fn design_row(&self, x: &DMatrix<f64>, i: usize, out: &mut Vec<f64>) {
        out.clear();
        if self.fit_intercept {
            out.push(1.0);
        }
        for j in 0..x.ncols() {
            out.push(x[(i, j)]);
        }
    }

    fn check_x(&self, x: &DMatrix<f64>) -> Result<()> {
        if x.ncols() != self.n_features {
            return Err(EsagError::Shape(format!(
                "X has {} columns but the model was fitted with {}",
                x.ncols(),
                self.n_features
            )));
        }
        Ok(())
    }

    /// Linear predictor `eta_i = B x̃_i` for every row of `x` (`n × b_dim`).
    pub fn linear_predictor(&self, x: &DMatrix<f64>) -> Result<DMatrix<f64>> {
        self.check_x(x)?;
        let n = x.nrows();
        let bd = self.b_dim();
        let mut out = DMatrix::zeros(n, bd);
        let mut xr = Vec::with_capacity(self.p_total());
        for i in 0..n {
            self.design_row(x, i, &mut xr);
            for k in 0..bd {
                let mut s = 0.0;
                for (j, xv) in xr.iter().enumerate() {
                    s += self.coef[(k, j)] * xv;
                }
                out[(i, k)] = s;
            }
        }
        Ok(out)
    }

    /// Predicted mean directions `mu_i/|mu_i|` (`n × d` unit vectors).
    pub fn predict(&self, x: &DMatrix<f64>) -> Result<DMatrix<f64>> {
        let eta = self.linear_predictor(x)?;
        let d = self.d;
        let mut out = DMatrix::zeros(x.nrows(), d);
        for i in 0..x.nrows() {
            let norm = (0..d).map(|k| eta[(i, k)].powi(2)).sum::<f64>().sqrt();
            for k in 0..d {
                out[(i, k)] = eta[(i, k)] / norm;
            }
        }
        Ok(out)
    }

    /// The fitted conditional distribution `ESAG(mu_i, gamma_i)` for every row.
    pub fn predict_distribution(&self, x: &DMatrix<f64>) -> Result<Vec<Esag>> {
        let eta = self.linear_predictor(x)?;
        (0..x.nrows())
            .map(|i| Esag::from_eta(eta.row(i).iter().copied().collect::<Vec<_>>().as_slice()))
            .collect()
    }

    /// Per-observation log-likelihood `log f(y_i; eta_i)`.
    pub fn log_likelihood_rows(&self, x: &DMatrix<f64>, y: &DMatrix<f64>) -> Result<DVector<f64>> {
        let data = RegressionData::new(x, y, self.fit_intercept)?;
        if data.d != self.d {
            return Err(EsagError::Shape(format!(
                "Y has d = {} but the model has d = {}",
                data.d, self.d
            )));
        }
        let eta = self.linear_predictor(x)?;
        let mut out = DVector::zeros(data.n);
        let mut e = vec![0.0; self.b_dim()];
        for i in 0..data.n {
            for k in 0..self.b_dim() {
                e[k] = eta[(i, k)];
            }
            out[i] = loglik::loglik_eta(&e, &data.y[i * self.d..(i + 1) * self.d]);
        }
        Ok(out)
    }

    /// Mean log-likelihood per observation (higher is better) — the
    /// scikit-learn `score` analogue.
    pub fn score(&self, x: &DMatrix<f64>, y: &DMatrix<f64>) -> Result<f64> {
        let ll = self.log_likelihood_rows(x, y)?;
        Ok(ll.sum() / ll.len() as f64)
    }

    /// Residuals `Q_i` and `T1_i` for data `(x, y)`.
    pub fn residuals(&self, x: &DMatrix<f64>, y: &DMatrix<f64>) -> Result<Residuals> {
        let data = RegressionData::new(x, y, self.fit_intercept)?;
        let eta = self.linear_predictor(x)?;
        let d = self.d;
        let mut q = Vec::with_capacity(data.n);
        let mut t1 = Vec::with_capacity(data.n);
        for i in 0..data.n {
            let e: Vec<f64> = (0..self.b_dim()).map(|k| eta[(i, k)]).collect();
            let frame = Frame::new(&e[..d], &e[d..]);
            let yi = &data.y[i * d..(i + 1) * d];
            let qi = residual_q(&frame, yi);
            let s: f64 = frame.lambda.iter().sum();
            q.push(qi);
            t1.push((frame.mu_norm * frame.mu_norm + s + 1.0) * qi);
        }
        Ok(Residuals { q, t1 })
    }

    /// Simulate responses from the fitted model at the rows of `x`
    /// (parametric bootstrap).
    pub fn sample_responses<R: Rng + ?Sized>(
        &self,
        x: &DMatrix<f64>,
        rng: &mut R,
    ) -> Result<DMatrix<f64>> {
        let dists = self.predict_distribution(x)?;
        let d = self.d;
        let mut out = DMatrix::zeros(x.nrows(), d);
        let mut buf = vec![0.0; d];
        for (i, dist) in dists.iter().enumerate() {
            let frame = dist.frame();
            dist.sample_one(&frame, rng, &mut buf);
            for k in 0..d {
                out[(i, k)] = buf[k];
            }
        }
        Ok(out)
    }

    /// Standard errors of the coefficients (`b_dim × p_total`; `NaN` for
    /// fixed entries), if the Hessian was computed and is invertible.
    pub fn std_errors(&self) -> Option<DMatrix<f64>> {
        let cov = self.cov_params.as_ref()?;
        let mut se = DMatrix::from_element(self.b_dim(), self.p_total(), f64::NAN);
        for (idx, &(k, j)) in self.free.iter().enumerate() {
            let v = cov[(idx, idx)];
            se[(k, j)] = if v > 0.0 { v.sqrt() } else { f64::NAN };
        }
        Some(se)
    }

    /// Names of the `eta` rows: `mu[1]..mu[d], gamma[1]..gamma[g]`.
    pub fn row_names(&self) -> Vec<String> {
        let d = self.d;
        (0..self.b_dim())
            .map(|k| {
                if k < d {
                    format!("mu[{}]", k + 1)
                } else {
                    format!("gamma[{}]", k - d + 1)
                }
            })
            .collect()
    }

    /// A statsmodels-style text summary: coefficient table with standard
    /// errors, z-values and two-sided p-values.
    pub fn summary(&self) -> String {
        use std::fmt::Write as _;
        let mut s = String::new();
        let _ = writeln!(
            s,
            "ESAG regression (d = {}, n = {}, p = {} covariate(s){})",
            self.d,
            self.n_obs,
            self.n_features,
            if self.fit_intercept {
                " + intercept"
            } else {
                ""
            }
        );
        let _ = writeln!(
            s,
            "log-likelihood = {:.4}   AIC = {:.3}   BIC = {:.3}   params = {}",
            self.log_likelihood,
            self.aic(),
            self.bic(),
            self.n_params()
        );
        let _ = writeln!(
            s,
            "optimiser: {:?}, {} iterations, {} evaluations, |grad|_inf = {:.2e}",
            self.status, self.n_iter, self.n_fev, self.grad_norm
        );
        let se = self.std_errors();
        let rows = self.row_names();
        let _ = writeln!(
            s,
            "{:<22}{:>12}{:>12}{:>9}{:>10}",
            "parameter", "estimate", "std.err", "z", "p>|z|"
        );
        let _ = writeln!(s, "{}", "-".repeat(65));
        for j in 0..self.p_total() {
            let _ = writeln!(s, "[{}]", self.column_names[j]);
            for k in 0..self.b_dim() {
                let est = self.coef[(k, j)];
                if !self.mask[(k, j)] {
                    let _ = writeln!(
                        s,
                        "  {:<20}{:>12.5}{:>12}{:>9}{:>10}",
                        rows[k], est, "(fixed)", "", ""
                    );
                    continue;
                }
                match se.as_ref().map(|m| m[(k, j)]) {
                    Some(sev) if sev.is_finite() => {
                        let z = est / sev;
                        let p = 2.0 * norm_sf(z.abs());
                        let _ = writeln!(
                            s,
                            "  {:<20}{:>12.5}{:>12.5}{:>9.3}{:>10.4}",
                            rows[k], est, sev, z, p
                        );
                    }
                    _ => {
                        let _ = writeln!(
                            s,
                            "  {:<20}{:>12.5}{:>12}{:>9}{:>10}",
                            rows[k], est, "n/a", "", ""
                        );
                    }
                }
            }
        }
        s
    }

    /// Serialise the fitted model to JSON.
    pub fn to_json(&self) -> Result<String> {
        Ok(serde_json::to_string_pretty(self)?)
    }
    /// Deserialise a fitted model from JSON.
    pub fn from_json(s: &str) -> Result<Self> {
        Ok(serde_json::from_str(s)?)
    }
}

/// Canonicalise the sign of the first `gamma` group.
///
/// Negating `(gamma_1, gamma_2)` shifts `theta_1` by π, which negates the
/// eigenvectors `v_1, v_2` and leaves `V` — hence the density — unchanged, for
/// every `d` (Yu & Huang 2024 call such parameters *equivalent*). In a
/// regression `gamma_i = B_γ x_i` the symmetry is linear: negating rows
/// `d` and `d+1` of `B` in every column gives an identical likelihood. This
/// picks the representative whose first non-zero entry of row `d` (scanning
/// columns, then row `d+1`) is positive. Returns whether `coef` was flipped.
pub fn canonicalize_coef(coef: &mut DMatrix<f64>, d: usize) -> bool {
    let lead = (0..coef.ncols())
        .map(|j| coef[(d, j)])
        .chain((0..coef.ncols()).map(|j| coef[(d + 1, j)]))
        .find(|v| *v != 0.0);
    if matches!(lead, Some(v) if v < 0.0) {
        for j in 0..coef.ncols() {
            coef[(d, j)] = -coef[(d, j)];
            coef[(d + 1, j)] = -coef[(d + 1, j)];
        }
        true
    } else {
        false
    }
}

/// Build a free-parameter mask of shape `eta_len(d) × p_total` with every
/// entry `true`.
pub fn full_mask(d: usize, p_total: usize) -> DMatrix<bool> {
    DMatrix::from_element(dims::eta_len(d), p_total, true)
}

/// Mask for the *isotropic* restriction: every `gamma` row fixed (at 0) in
/// every column.
pub fn isotropic_mask(d: usize, p_total: usize) -> DMatrix<bool> {
    let mut m = full_mask(d, p_total);
    for j in 0..p_total {
        for k in d..dims::eta_len(d) {
            m[(k, j)] = false;
        }
    }
    m
}

/// Mask fixing the `mu` rows of design column `col` (R: `MLE_restricted_mu`).
pub fn mu_restricted_mask(d: usize, p_total: usize, col: usize) -> DMatrix<bool> {
    let mut m = full_mask(d, p_total);
    for k in 0..d {
        m[(k, col)] = false;
    }
    m
}

/// Mask fixing the `gamma` rows of design column `col` (R: `MLE_restricted_gamma`).
pub fn gamma_restricted_mask(d: usize, p_total: usize, col: usize) -> DMatrix<bool> {
    let mut m = full_mask(d, p_total);
    for k in d..dims::eta_len(d) {
        m[(k, col)] = false;
    }
    m
}
