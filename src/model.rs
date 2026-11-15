//! statsmodels-style front end: `EsagModel::new(endog, exog)`, then
//! `fit()` or `fit_regularized(penalty)`; results carry the data, so tests,
//! goodness of fit and regions are methods of the results.
//!
//! ```no_run
//! use esag::model::EsagModel;
//! use esag::penalized::PenalizedEsag;
//! use esag::inference::Restriction;
//! # let (x, y) = (nalgebra::DMatrix::<f64>::zeros(1, 1), nalgebra::DMatrix::<f64>::zeros(1, 3));
//! let model = EsagModel::new(&y, &x).exog_names(&["temp"]).endog_names(&["a", "b", "c"]);
//! let res = model.fit()?;
//! println!("{}", res.summary());
//! let iso = res.test(Restriction::Isotropy, 500, 1)?;
//! println!("{}", iso.summary());
//! let reg = model.fit_regularized(PenalizedEsag::adaptive_group_lasso())?;
//! println!("{}", reg.summary());
//! # Ok::<(), esag::error::EsagError>(())
//! ```

use crate::error::{EsagError, Result};
use crate::inference::{
    CovarianceEstimate, GofMethod, GofResult, HypothesisTestResult, PredictionRegion, Restriction,
    bootstrap_gof, covariance_intervals, hypothesis_test, prediction_region, score_test,
};
use crate::penalized::{PenalizedEsag, PenalizedPath};
use crate::regression::{EsagRegressor, EsagRegressorFit, Residuals, Shape};
use nalgebra::{DMatrix, DVector};
use rand::Rng;

/// How the responses are standardized before the fit
/// ([`EsagModel::response_scale`]). The model is then ESAG for the
/// standardized directions `u = unit(y / s)`, which on the original sphere is
/// the projected normal `PN(D μ, D V D)`, `D = diag(s)`: ESAG fixes the
/// variance along the mean direction at the geometric mean of the others,
/// and the part scales free it. That matters when the coordinates have very
/// different scales, as net flows or returns of parts of different sizes,
/// where one part dominates both the mean and the noise of the direction:
/// there ESAG's best fit turns its mean direction away from the data
/// (`docs/TIME_SERIES_AND_COMPOSITIONS.md`).
#[derive(Clone, Debug, PartialEq)]
pub enum ResponseScale {
    /// Given part scales, one positive number per coordinate (only their
    /// ratios matter).
    Fixed(Vec<f64>),
    /// The diagonal Tyler estimate from the directions ([`tyler_scales`]).
    Tyler,
    /// Fit with and without the Tyler scales and keep the lower BIC on the
    /// original sphere (the scales count `d − 1` parameters).
    Auto,
}

/// The diagonal Tyler scales of directions `y` (`n × d`, rows normalized
/// here): the maximum-likelihood scales of a zero-centred angular central
/// Gaussian with a diagonal shape (Tyler 1987), the fixed point of
/// `σ_k = d · mean_i (y_ik² / Σ_l y_il² / σ_l)`, as `s = √σ` with geometric
/// mean 1. For vectors whose parts have scales `s` (independent, or a
/// common radial law) these are the part scales, whatever the sizes of the
/// vectors.
pub fn tyler_scales(y: &DMatrix<f64>) -> Result<Vec<f64>> {
    let u = unit_rows(y)?;
    let (n, d) = (u.nrows(), u.ncols());
    let mut sig = vec![1.0; d];
    for _ in 0..2000 {
        let mut acc = vec![0.0; d];
        for i in 0..n {
            let q: f64 = (0..d).map(|k| u[(i, k)] * u[(i, k)] / sig[k]).sum();
            for (k, a) in acc.iter_mut().enumerate() {
                *a += u[(i, k)] * u[(i, k)] / q;
            }
        }
        let mut next: Vec<f64> = acc.iter().map(|a| d as f64 * a / n as f64).collect();
        if next.iter().any(|v| !(v.is_finite() && *v > 0.0)) {
            return Err(EsagError::Data(
                "Tyler scales: a response coordinate is zero in every row".into(),
            ));
        }
        let gm = (next.iter().map(|v| v.ln()).sum::<f64>() / d as f64).exp();
        for v in &mut next {
            *v /= gm;
        }
        let change = next
            .iter()
            .zip(&sig)
            .map(|(a, b)| (a / b).ln().abs())
            .fold(0.0, f64::max);
        sig = next;
        if change < 1e-12 {
            break;
        }
    }
    Ok(sig.iter().map(|v| v.sqrt()).collect())
}

/// Rows normalized to unit length (an error for a zero row).
fn unit_rows(y: &DMatrix<f64>) -> Result<DMatrix<f64>> {
    let mut u = y.clone();
    for i in 0..u.nrows() {
        let r = u.row(i).norm();
        if !(r.is_finite() && r > 0.0) {
            return Err(EsagError::Data(format!("response row {i} is zero")));
        }
        for k in 0..u.ncols() {
            u[(i, k)] /= r;
        }
    }
    Ok(u)
}

/// `unit(y / s)` (`divide`) or `unit(y ⊙ s)` row by row.
fn rescale_rows(y: &DMatrix<f64>, s: &[f64], divide: bool) -> Result<DMatrix<f64>> {
    let mut v = y.clone();
    for i in 0..v.nrows() {
        for (k, sk) in s.iter().enumerate() {
            if divide {
                v[(i, k)] /= sk;
            } else {
                v[(i, k)] *= sk;
            }
        }
    }
    unit_rows(&v)
}

/// An ESAG regression model of `endog` (`n × d` unit vectors) on `exog`
/// (`n × p` covariates; a constant column `const` is added unless
/// `has_constant(false)`).
#[derive(Clone, Debug)]
pub struct EsagModel {
    /// The responses the model is fitted to: `unit(y / s)` when the
    /// responses are standardized ([`EsagModel::response_scale`]).
    endog: DMatrix<f64>,
    /// The responses as given (rows normalized).
    raw_endog: DMatrix<f64>,
    exog: DMatrix<f64>,
    config: EsagRegressor,
    /// Part scales (geometric mean 1) of a standardized model.
    scale: Option<Vec<f64>>,
    /// Whether the scales were estimated from the data (they then count
    /// `d − 1` parameters in the criteria on the original sphere).
    scale_estimated: bool,
    /// [`ResponseScale::Auto`]: the Tyler scales, chosen or not at fit.
    auto_scale: Option<Vec<f64>>,
}

impl EsagModel {
    pub fn new(endog: &DMatrix<f64>, exog: &DMatrix<f64>) -> Self {
        Self {
            endog: endog.clone(),
            raw_endog: endog.clone(),
            exog: exog.clone(),
            config: EsagRegressor::new(),
            scale: None,
            scale_estimated: false,
            auto_scale: None,
        }
    }
    /// Standardize the responses by part scales before the fit (see
    /// [`ResponseScale`]): the model is ESAG for `unit(y / s)`, and the
    /// results' predictions, covariances, densities and simulations are on
    /// the original sphere (coefficients and tests are those of the
    /// standardized model).
    pub fn response_scale(mut self, scale: ResponseScale) -> Result<Self> {
        let d = self.raw_endog.ncols();
        let unit = unit_rows(&self.raw_endog)?;
        self.scale_estimated = !matches!(scale, ResponseScale::Fixed(_));
        let s = match scale {
            ResponseScale::Fixed(s) => {
                if s.len() != d || s.iter().any(|v| !(v.is_finite() && *v > 0.0)) {
                    return Err(EsagError::Data(format!(
                        "response scales must be {d} positive numbers"
                    )));
                }
                let gm = (s.iter().map(|v| v.ln()).sum::<f64>() / d as f64).exp();
                s.iter().map(|v| v / gm).collect()
            }
            ResponseScale::Tyler => tyler_scales(&unit)?,
            ResponseScale::Auto => {
                self.auto_scale = Some(tyler_scales(&unit)?);
                self.scale = None;
                self.endog = unit;
                return Ok(self);
            }
        };
        self.auto_scale = None;
        self.endog = rescale_rows(&unit, &s, true)?;
        self.scale = Some(s);
        Ok(self)
    }
    /// The part scales of a standardized model (geometric mean 1).
    pub fn scale(&self) -> Option<&[f64]> {
        self.scale.as_deref()
    }
    /// The responses as given (the model is fitted to [`EsagModel::endog`],
    /// their standardized version when [`EsagModel::scale`] is set).
    pub fn raw_endog(&self) -> &DMatrix<f64> {
        &self.raw_endog
    }
    /// This model with the part scales `s` (or none), for
    /// [`ResponseScale::Auto`].
    fn with_scale(&self, s: Option<Vec<f64>>) -> Result<Self> {
        let mut m = self.clone();
        m.auto_scale = None;
        let unit = unit_rows(&self.raw_endog)?;
        match s {
            Some(s) => {
                m.endog = rescale_rows(&unit, &s, true)?;
                m.scale = Some(s);
                m.scale_estimated = true;
            }
            None => {
                m.endog = unit;
                m.scale = None;
                m.scale_estimated = false;
            }
        }
        Ok(m)
    }
    /// The model with [`ResponseScale::Auto`] decided: the unscaled and the
    /// Tyler-scaled models are fitted and the one with the lower BIC on the
    /// original sphere is returned (the model itself otherwise).
    pub fn resolve_scale(&self) -> Result<Self> {
        Ok(self.resolve_scale_fits()?.0)
    }
    fn resolve_scale_fits(&self) -> Result<(Self, Option<EsagResults>)> {
        let Some(s) = &self.auto_scale else {
            return Ok((self.clone(), None));
        };
        let plain = self.with_scale(None)?;
        let scaled = self.with_scale(Some(s.clone()))?;
        let fits = (plain.fit(), scaled.fit());
        Ok(match fits {
            (Ok(a), Ok(b)) => {
                if b.bic_original() < a.bic_original() {
                    (scaled, Some(b))
                } else {
                    (plain, Some(a))
                }
            }
            (Ok(a), Err(_)) => (plain, Some(a)),
            (Err(_), Ok(b)) => (scaled, Some(b)),
            (Err(e), Err(_)) => return Err(e),
        })
    }
    /// An i.i.d. model (no covariates).
    pub fn iid(endog: &DMatrix<f64>) -> Self {
        Self::new(endog, &DMatrix::zeros(endog.nrows(), 0))
    }
    /// Covariate names (one per column of `exog`).
    pub fn exog_names<S: AsRef<str>>(mut self, names: &[S]) -> Self {
        self.config.feature_names = Some(names.iter().map(|s| s.as_ref().to_string()).collect());
        self
    }
    /// Response names (one per column of `endog`).
    pub fn endog_names<S: AsRef<str>>(mut self, names: &[S]) -> Self {
        self.config.response_names = Some(names.iter().map(|s| s.as_ref().to_string()).collect());
        self
    }
    /// Shape parameterization.
    pub fn shape(mut self, shape: Shape) -> Self {
        self.config.shape = shape;
        self
    }
    /// Reference direction `m` of the log-tangent frame (default: the
    /// data's mean direction). The log-tangent regression model depends on
    /// it: `s` linear in `x` in one frame is not linear in another. Set it to
    /// compare with a model defined in a given frame (e.g. a simulation
    /// truth).
    pub fn lt_reference(mut self, m: Vec<f64>) -> Self {
        self.config.lt_reference = Some(m);
        self
    }
    /// Basis rotation `R_0` of the gamma-rot frame (default: from a pilot
    /// fit). The gamma-rot regression model depends on it; set it, with
    /// `lt_reference`, to fit the same model to other data.
    pub fn rot_basis(mut self, r0: DMatrix<f64>) -> Self {
        self.config.rot_basis = Some(r0);
        self
    }
    /// Add a constant column (default `true`).
    pub fn has_constant(mut self, v: bool) -> Self {
        self.config.fit_intercept = v;
        self
    }
    /// Replace the estimator settings (iterations, starts, …); names, shape
    /// and the constant are kept from the model.
    pub fn with_config(mut self, cfg: EsagRegressor) -> Self {
        let (f, r, s, c, m) = (
            self.config.feature_names.take(),
            self.config.response_names.take(),
            self.config.shape,
            self.config.fit_intercept,
            self.config.lt_reference.take(),
        );
        self.config = cfg;
        self.config.feature_names = f;
        self.config.response_names = r;
        self.config.shape = s;
        self.config.fit_intercept = c;
        if m.is_some() {
            self.config.lt_reference = m;
        }
        self
    }
    /// The responses the model is fitted to (standardized when
    /// [`EsagModel::scale`] is set; [`EsagModel::raw_endog`] are as given).
    pub fn endog(&self) -> &DMatrix<f64> {
        &self.endog
    }
    pub fn exog(&self) -> &DMatrix<f64> {
        &self.exog
    }

    /// Maximum-likelihood fit with standard errors.
    pub fn fit(&self) -> Result<EsagResults> {
        if self.auto_scale.is_some() {
            let (_, res) = self.resolve_scale_fits()?;
            return res.ok_or_else(|| EsagError::Data("automatic response scale: no fit".into()));
        }
        let fit = self.config.fit(&self.exog, &self.endog)?;
        Ok(EsagResults {
            fit,
            model: self.clone(),
        })
    }

    /// Penalized fit along a λ path, λ selected by the estimator's rule
    /// (`PenalizedEsag::auto()`, the recommended penalty for these data,
    /// `lasso()`, `adaptive_lasso()`, `group_lasso()`, `hierarchical()`, …).
    /// The model's shape and names override the estimator's (paper presets
    /// keep the paper's gamma shape).
    pub fn fit_regularized(&self, penalty: PenalizedEsag) -> Result<PenalizedPath> {
        if self.auto_scale.is_some() {
            return self.resolve_scale()?.fit_regularized(penalty);
        }
        let penalty = penalty.resolved(self.exog.ncols() > 0);
        // the paper's methods are γ's, and the eigen-gap penalty is the
        // log-tangent's (i.i.d., where every shape describes the same
        // distributions): they keep their shape
        let mut p = if penalty.is_paper() || penalty.penalty == crate::penalized::Penalty::EigenGap
        {
            penalty
        } else {
            penalty.shape(self.config.shape)
        };
        if !self.config.fit_intercept {
            return Err(EsagError::Data("penalized fits need the constant".into()));
        }
        if let Some(n) = &self.config.feature_names {
            p = p.feature_names(n.clone());
        }
        if let Some(n) = &self.config.response_names {
            p = p.response_names(n.clone());
        }
        if let Some(m) = &self.config.lt_reference {
            p = p.lt_reference(m.clone());
        }
        if let Some(r0) = &self.config.rot_basis {
            p = p.rot_basis(r0.clone());
        }
        p.fit(&self.exog, &self.endog)
    }
}

/// A fitted [`EsagModel`]: the [`EsagRegressorFit`] (all its methods,
/// including `params`, `bse`, `pvalues`, `conf_int` and `summary`, via
/// `Deref`) plus the data for tests and diagnostics.
#[derive(Clone, Debug)]
pub struct EsagResults {
    pub fit: EsagRegressorFit,
    pub model: EsagModel,
}

impl std::ops::Deref for EsagResults {
    type Target = EsagRegressorFit;
    fn deref(&self) -> &EsagRegressorFit {
        &self.fit
    }
}

impl EsagResults {
    /// The part scales of a standardized model (geometric mean 1; see
    /// [`EsagModel::response_scale`]).
    pub fn scale(&self) -> Option<&[f64]> {
        self.model.scale()
    }
    /// Predicted mean directions on the original sphere: for a standardized
    /// model `unit(s ⊙ μ̂_s(x))`, the image of the standardized model's mean
    /// direction.
    pub fn predict(&self, x: &DMatrix<f64>) -> Result<DMatrix<f64>> {
        let p = self.fit.predict(x)?;
        match self.scale() {
            Some(s) => rescale_rows(&p, s, false),
            None => Ok(p),
        }
    }
    /// The mean vector `m(x)` of the predictive projected normal
    /// `PN(m(x), V(x))` on the original sphere (`V` from
    /// [`EsagResults::predict_covariance`]): `μ(x)`, or `D μ_s(x)` for a
    /// standardized model.
    pub fn mean_vector(&self, x: &DMatrix<f64>) -> Result<DMatrix<f64>> {
        let eta = self.fit.linear_predictor(x)?;
        let d = self.fit.d;
        let mut m = eta.columns(0, d).into_owned();
        if let Some(s) = self.scale() {
            for i in 0..m.nrows() {
                for (k, sk) in s.iter().enumerate() {
                    m[(i, k)] *= sk;
                }
            }
        }
        Ok(m)
    }
    /// `V̂(x)` on the original sphere at each row of `x`: for a standardized
    /// model `D V_s(x) D` (determinant 1, the covariance of the predictive
    /// projected normal; the mean direction is then an eigenvector of
    /// `D⁻¹ V D⁻¹`).
    pub fn predict_covariance(&self, x: &DMatrix<f64>) -> Result<Vec<DMatrix<f64>>> {
        let v = self.fit.predict_covariance(x)?;
        match self.scale() {
            None => Ok(v),
            Some(s) => {
                let dm = DMatrix::from_diagonal(&DVector::from_column_slice(s));
                Ok(v.into_iter().map(|vi| &dm * vi * &dm).collect())
            }
        }
    }
    /// `−d ln |u_i / s|` per row of directions `u`: the log-Jacobian of
    /// `u ↦ unit(u / s)` (`det D = 1`).
    fn log_jacobian(&self, u: &DMatrix<f64>, s: &[f64]) -> Vec<f64> {
        let d = u.ncols() as f64;
        (0..u.nrows())
            .map(|i| {
                let r2: f64 = s
                    .iter()
                    .enumerate()
                    .map(|(k, sk)| (u[(i, k)] / sk).powi(2))
                    .sum();
                -0.5 * d * r2.ln()
            })
            .collect()
    }
    /// Log-densities of directions `y` on the original sphere at covariates
    /// `x`: for a standardized model the density of `unit(y / s)` times the
    /// Jacobian of the map.
    pub fn log_likelihood_rows(&self, x: &DMatrix<f64>, y: &DMatrix<f64>) -> Result<DVector<f64>> {
        match self.scale() {
            None => self.fit.log_likelihood_rows(x, y),
            Some(s) => {
                let u = unit_rows(y)?;
                let mut l = self
                    .fit
                    .log_likelihood_rows(x, &rescale_rows(&u, s, true)?)?;
                for (li, j) in l.iter_mut().zip(self.log_jacobian(&u, s)) {
                    *li += j;
                }
                Ok(l)
            }
        }
    }
    /// The log-likelihood of the training responses on the original sphere
    /// (the fit's for an unstandardized model).
    pub fn log_likelihood_original(&self) -> f64 {
        match self.scale() {
            None => self.fit.log_likelihood,
            Some(s) => match unit_rows(&self.model.raw_endog) {
                Ok(u) => self.fit.log_likelihood + self.log_jacobian(&u, s).iter().sum::<f64>(),
                Err(_) => f64::NAN,
            },
        }
    }
    /// Parameters counted on the original sphere: the fit's, plus `d − 1`
    /// for estimated scales.
    pub fn n_params_original(&self) -> usize {
        let extra = if self.model.scale.is_some() && self.model.scale_estimated {
            self.fit.d - 1
        } else {
            0
        };
        self.fit.n_params() + extra
    }
    /// AIC on the original sphere.
    pub fn aic_original(&self) -> f64 {
        2.0 * self.n_params_original() as f64 - 2.0 * self.log_likelihood_original()
    }
    /// BIC on the original sphere.
    pub fn bic_original(&self) -> f64 {
        (self.fit.n_obs as f64).ln() * self.n_params_original() as f64
            - 2.0 * self.log_likelihood_original()
    }
    /// One response per row of `x` drawn from the fitted model, on the
    /// original sphere.
    pub fn sample_responses<R: Rng + ?Sized>(
        &self,
        x: &DMatrix<f64>,
        rng: &mut R,
    ) -> Result<DMatrix<f64>> {
        let ys = self.fit.sample_responses(x, rng)?;
        match self.scale() {
            Some(s) => rescale_rows(&ys, s, false),
            None => Ok(ys),
        }
    }
    /// Residuals of directions `y` (on the original sphere; standardized for
    /// a standardized model).
    pub fn residuals(&self, x: &DMatrix<f64>, y: &DMatrix<f64>) -> Result<Residuals> {
        match self.scale() {
            None => self.fit.residuals(x, y),
            Some(s) => self
                .fit
                .residuals(x, &rescale_rows(&unit_rows(y)?, s, true)?),
        }
    }
    /// The fit's summary, with the part scales of a standardized model.
    pub fn summary_alpha(&self, alpha: f64) -> String {
        let mut out = self.fit.summary_alpha(alpha);
        if let Some(s) = self.scale() {
            let list: Vec<String> = s.iter().map(|v| format!("{v:.4}")).collect();
            out.push_str(&format!(
                "Responses standardized by part scales s = [{}] ({}): the coefficients\n    describe unit(y / s). On the original sphere: log-likelihood {:.4},\n    AIC {:.4}, BIC {:.4}.\n",
                list.join(", "),
                if self.model.scale_estimated {
                    "estimated, diagonal Tyler"
                } else {
                    "given"
                },
                self.log_likelihood_original(),
                self.aic_original(),
                self.bic_original()
            ));
        }
        out
    }
    /// [`EsagResults::summary_alpha`] at the 5% level.
    pub fn summary(&self) -> String {
        self.summary_alpha(0.05)
    }
    /// Bootstrap test of `restriction` (RoC, D, M with exact moments, LR).
    pub fn test(
        &self,
        restriction: Restriction,
        n_boot: usize,
        seed: u64,
    ) -> Result<HypothesisTestResult> {
        hypothesis_test(
            &self.fit,
            &self.model.exog,
            &self.model.endog,
            restriction,
            n_boot,
            0,
            seed,
        )
    }
    /// The same test from the restricted fit alone (score statistic `LM` and
    /// `M`; see [`crate::inference::score_test`]): 1.4–1.9× faster for the
    /// log-tangent shape, 2–4× (up to 29×) for gamma.
    pub fn score_test(
        &self,
        restriction: Restriction,
        n_boot: usize,
        seed: u64,
    ) -> Result<HypothesisTestResult> {
        score_test(
            &self.fit,
            &self.model.exog,
            &self.model.endog,
            restriction,
            n_boot,
            0,
            seed,
        )
    }
    /// Index of a covariate by name.
    pub fn feature_index(&self, name: &str) -> Result<usize> {
        let off = usize::from(self.fit.fit_intercept);
        self.fit.column_names[off..]
            .iter()
            .position(|c| c == name)
            .ok_or_else(|| EsagError::Data(format!("no covariate named {name:?}")))
    }
    /// `V̂(x)` at each row of `x_new` with pairs-bootstrap intervals (see
    /// [`covariance_intervals`]).
    pub fn covariance(
        &self,
        x_new: &DMatrix<f64>,
        n_boot: usize,
        level: f64,
        seed: u64,
    ) -> Result<Vec<CovarianceEstimate>> {
        covariance_intervals(
            &self.fit,
            &self.model.exog,
            &self.model.endog,
            x_new,
            n_boot,
            level,
            seed,
        )
    }
    /// Bootstrap goodness-of-fit test (Algorithm 1).
    pub fn gof(&self, n_boot: usize, seed: u64) -> Result<GofResult> {
        bootstrap_gof(
            &self.fit,
            &self.model.exog,
            &self.model.endog,
            GofMethod::TwoSampleReference,
            n_boot,
            seed,
        )
    }
    /// Prediction-region thresholds at covariates `x_new`.
    pub fn prediction_region(
        &self,
        x_new: &[f64],
        levels: &[f64],
        n_mc: usize,
        n_boot: usize,
        seed: u64,
    ) -> Result<PredictionRegion> {
        prediction_region(
            &self.fit,
            &self.model.exog,
            &self.model.endog,
            x_new,
            n_mc,
            n_boot,
            levels,
            seed,
        )
    }
}
