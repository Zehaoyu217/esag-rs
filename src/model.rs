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
use crate::regression::{EsagRegressor, EsagRegressorFit, Shape};
use nalgebra::DMatrix;

/// An ESAG regression model of `endog` (`n × d` unit vectors) on `exog`
/// (`n × p` covariates; a constant column `const` is added unless
/// `has_constant(false)`).
#[derive(Clone, Debug)]
pub struct EsagModel {
    endog: DMatrix<f64>,
    exog: DMatrix<f64>,
    config: EsagRegressor,
}

impl EsagModel {
    pub fn new(endog: &DMatrix<f64>, exog: &DMatrix<f64>) -> Self {
        Self {
            endog: endog.clone(),
            exog: exog.clone(),
            config: EsagRegressor::new(),
        }
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
    pub fn endog(&self) -> &DMatrix<f64> {
        &self.endog
    }
    pub fn exog(&self) -> &DMatrix<f64> {
        &self.exog
    }

    /// Maximum-likelihood fit with standard errors.
    pub fn fit(&self) -> Result<EsagResults> {
        let fit = self.config.fit(&self.exog, &self.endog)?;
        Ok(EsagResults {
            fit,
            model: self.clone(),
        })
    }

    /// Penalized fit along a λ path, λ selected by the estimator's rule
    /// (`PenalizedEsag::lasso()`, `adaptive_lasso()`, `group_lasso()`, …).
    /// The model's shape and names override the estimator's (paper presets
    /// keep the paper's gamma shape).
    pub fn fit_regularized(&self, penalty: PenalizedEsag) -> Result<PenalizedPath> {
        let mut p = if penalty.is_paper() {
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
    /// `M`; see [`crate::inference::score_test`]): 2–4× faster.
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
