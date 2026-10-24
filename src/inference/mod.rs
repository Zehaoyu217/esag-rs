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

/// Iteration cap of bootstrap refits.
const REFIT_MAX_ITER: usize = 800;

/// A refit configuration for bootstrap replicates: same hyper-parameters as the
/// original fit, but no Hessian, no restarts and quiet.
pub(crate) fn refit_config(fit: &EsagRegressorFit) -> EsagRegressor {
    let mut cfg = fit.config.clone();
    cfg.compute_hessian = false;
    cfg.n_restarts = 0;
    cfg.verbose = false;
    // one refit per thread already saturates the machine
    cfg.parallel = false;
    cfg.mask = Some(fit.mask.clone());
    // warm-started refits that converge need a few dozen iterations; ones
    // still running after this are stuck at a cliff of the gamma
    // parameterization (d >= 5 with covariates) and cannot be rescued
    cfg.max_iter = cfg.max_iter.min(REFIT_MAX_ITER);
    // refits start near the original optimum, where the observed information
    // is nearly the same: whitening with it makes L-BFGS converge in a few
    // iterations
    cfg.precondition = fit
        .cov_params
        .as_ref()
        .and_then(crate::regression::precondition_from_cov);
    cfg
}
