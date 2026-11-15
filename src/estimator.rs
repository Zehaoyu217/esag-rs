//! Fitting a single ESAG distribution to i.i.d. directional data (no
//! covariates), plus the scikit-learn-style traits shared by the estimators.

use crate::distribution::Esag;
use crate::error::Result;
use crate::regression::{EsagRegressor, EsagRegressorFit, Init, Residuals};
use nalgebra::{DMatrix, DVector};
use rand::Rng;
use serde::{Deserialize, Serialize};

/// `fit(X, Y) -> fitted model`.
pub trait Fit<X: ?Sized, Y: ?Sized> {
    /// The fitted model type.
    type Fitted;
    /// Fit the model to data.
    fn fit(&self, x: &X, y: &Y) -> Result<Self::Fitted>;
}

/// `predict(X) -> predictions`.
pub trait Predict<X: ?Sized> {
    /// The prediction type.
    type Output;
    /// Predict for new inputs.
    fn predict(&self, x: &X) -> Result<Self::Output>;
}

/// `score(X, Y) -> mean log-likelihood`.
pub trait Score<X: ?Sized, Y: ?Sized> {
    /// Mean per-observation log-likelihood (higher is better).
    fn score(&self, x: &X, y: &Y) -> Result<f64>;
}

impl Fit<DMatrix<f64>, DMatrix<f64>> for EsagRegressor {
    type Fitted = EsagRegressorFit;
    fn fit(&self, x: &DMatrix<f64>, y: &DMatrix<f64>) -> Result<EsagRegressorFit> {
        EsagRegressor::fit(self, x, y)
    }
}
impl Predict<DMatrix<f64>> for EsagRegressorFit {
    type Output = DMatrix<f64>;
    fn predict(&self, x: &DMatrix<f64>) -> Result<DMatrix<f64>> {
        EsagRegressorFit::predict(self, x)
    }
}
impl Score<DMatrix<f64>, DMatrix<f64>> for EsagRegressorFit {
    fn score(&self, x: &DMatrix<f64>, y: &DMatrix<f64>) -> Result<f64> {
        EsagRegressorFit::score(self, x, y)
    }
}

/// Maximum-likelihood estimator of a single `ESAG(mu, gamma)` from i.i.d.
/// unit vectors (R: `MLE()` with an intercept-only design).
///
/// ```no_run
/// use esag::estimator::EsagEstimator;
/// # let y = nalgebra::DMatrix::<f64>::zeros(1, 3);
/// let fit = EsagEstimator::new().fit(&y).unwrap();
/// println!("{}", fit.summary());
/// let dist = fit.distribution();
/// ```
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EsagEstimator {
    inner: EsagRegressor,
}

impl Default for EsagEstimator {
    fn default() -> Self {
        Self::new()
    }
}

impl EsagEstimator {
    /// Default configuration.
    pub fn new() -> Self {
        Self {
            inner: EsagRegressor::new(),
        }
    }
    /// Maximum number of iterations.
    pub fn with_max_iter(mut self, v: usize) -> Self {
        self.inner.max_iter = v;
        self
    }
    /// Gradient tolerance.
    pub fn with_tol(mut self, v: f64) -> Self {
        self.inner.tol = v;
        self
    }
    /// Initialisation strategy (`Init::Custom` expects an `eta_len(d) × 1` matrix).
    pub fn with_init(mut self, v: Init) -> Self {
        self.inner.init = v;
        self
    }
    /// Random restarts.
    pub fn with_restarts(mut self, n: usize, seed: u64) -> Self {
        self.inner.n_restarts = n;
        self.inner.seed = seed;
        self
    }
    /// Whether to compute the Hessian / standard errors.
    pub fn with_hessian(mut self, v: bool) -> Self {
        self.inner.compute_hessian = v;
        self
    }
    /// Verbose optimisation output.
    pub fn with_verbose(mut self, v: bool) -> Self {
        self.inner.verbose = v;
        self
    }
    /// Shape parameterization ([`crate::regression::Shape`]).
    pub fn with_shape(mut self, shape: crate::regression::Shape) -> Self {
        self.inner.shape = shape;
        self
    }
    /// Force an isotropic fit (`gamma = 0`).
    pub fn isotropic(mut self, d: usize) -> Self {
        self.inner.mask = Some(crate::regression::isotropic_mask(d, 1));
        self
    }
    /// The underlying regression configuration.
    pub fn regressor(&self) -> &EsagRegressor {
        &self.inner
    }

    /// Fit to `y` (`n × d`, one unit vector per row).
    pub fn fit(&self, y: &DMatrix<f64>) -> Result<EsagFit> {
        let x = DMatrix::<f64>::zeros(y.nrows(), 0);
        let reg = self.inner.fit(&x, y)?;
        let eta: Vec<f64> = reg.coef.column(0).iter().copied().collect();
        let dist = reg.distribution_at(&eta, reg.lt_frame().as_ref())?;
        Ok(EsagFit { dist, reg })
    }
}

/// A fitted single ESAG distribution.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EsagFit {
    dist: Esag,
    reg: EsagRegressorFit,
}

impl EsagFit {
    /// The fitted distribution.
    pub fn distribution(&self) -> &Esag {
        &self.dist
    }
    /// Estimated `mu`.
    pub fn mu(&self) -> &[f64] {
        self.dist.mu()
    }
    /// Estimated `gamma`.
    pub fn gamma(&self) -> &[f64] {
        self.dist.gamma()
    }
    /// Estimated mean direction.
    pub fn mean_direction(&self) -> Vec<f64> {
        self.dist.mean_direction()
    }
    /// Maximised log-likelihood.
    pub fn log_likelihood(&self) -> f64 {
        self.reg.log_likelihood
    }
    /// AIC.
    pub fn aic(&self) -> f64 {
        self.reg.aic()
    }
    /// BIC.
    pub fn bic(&self) -> f64 {
        self.reg.bic()
    }
    /// Whether the optimiser converged.
    pub fn converged(&self) -> bool {
        self.reg.converged
    }
    /// Standard errors of `eta = (mu, gamma)`, if available.
    pub fn std_errors(&self) -> Option<DVector<f64>> {
        self.reg.std_errors().map(|m| m.column(0).into_owned())
    }
    /// Observed information matrix, if computed.
    pub fn hessian(&self) -> Option<&DMatrix<f64>> {
        self.reg.hessian.as_ref()
    }
    /// The underlying regression fit (intercept-only).
    pub fn as_regression(&self) -> &EsagRegressorFit {
        &self.reg
    }
    /// Text summary.
    pub fn summary(&self) -> String {
        self.reg.summary()
    }
    /// Per-observation log-likelihood.
    pub fn log_likelihood_rows(&self, y: &DMatrix<f64>) -> DVector<f64> {
        self.dist.log_pdf_rows(y)
    }
    /// Mean log-likelihood on `y`.
    pub fn score(&self, y: &DMatrix<f64>) -> f64 {
        let ll = self.log_likelihood_rows(y);
        ll.sum() / ll.len() as f64
    }
    /// Residuals `Q_i`, `T1_i`.
    pub fn residuals(&self, y: &DMatrix<f64>) -> Result<Residuals> {
        let x = DMatrix::<f64>::zeros(y.nrows(), 0);
        self.reg.residuals(&x, y)
    }
    /// Simulate `n` observations from the fitted distribution.
    pub fn sample<R: Rng + ?Sized>(&self, n: usize, rng: &mut R) -> DMatrix<f64> {
        self.dist.sample(n, rng)
    }
}
