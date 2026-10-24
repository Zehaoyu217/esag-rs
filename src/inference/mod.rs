//! Model diagnostics and inference for fitted ESAG regressions: parametric
//! bootstrap goodness-of-fit (`gof`), Monte-Carlo confidence / prediction
//! regions (`regions`) and bootstrap hypothesis tests for isotropy and
//! covariate dependence (`tests`).

pub mod bootstrap;
pub mod gof;
pub mod regions;
pub mod tests;

pub use bootstrap::{BootstrapResult, bootstrap_coef};
pub use gof::{GofMethod, GofResult, bootstrap_gof};
pub use regions::{
    PredictionRegion, bootstrap_refits, confidence_region_sample, prediction_region,
    prediction_regions, region_coverage,
};
pub use tests::{HypothesisTestResult, Restriction, hypothesis_test};

use crate::regression::{EsagRegressor, EsagRegressorFit};

/// A refit configuration for bootstrap replicates: same hyper-parameters as the
/// original fit, but no Hessian, no restarts and quiet.
pub(crate) fn refit_config(fit: &EsagRegressorFit) -> EsagRegressor {
    let mut cfg = fit.config.clone();
    cfg.compute_hessian = false;
    cfg.n_restarts = 0;
    cfg.verbose = false;
    cfg.mask = Some(fit.mask.clone());
    cfg
}
