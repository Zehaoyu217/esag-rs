//! Goodness-of-fit test based on the `T1` residuals with a parametric
//! bootstrap.
//!
//! Two variants are provided:
//!
//! * [`GofMethod::TwoSampleReference`] — Yu & Huang (2024), Algorithm 1: the
//!   statistic `KS_p` is the p-value of a **two-sample** KS test comparing the
//!   observed `T1` residuals with residuals of a reference sample drawn from the
//!   fitted model (one draw per observation, no re-estimation). Its null
//!   distribution comes from a parametric bootstrap that re-fits the model on
//!   each simulated data set and draws a fresh reference sample from the
//!   bootstrap fit. p-value `= mean(KS_p^(b) < KS_p)`. This is the default.
//! * [`GofMethod::OneSampleChiSquare`] — the variant implemented by
//!   `Bootstrap_GoF` in the author's regression code: the statistic is the
//!   one-sample KS p-value of `T1` against χ²_{d−1}, calibrated by the same
//!   parametric bootstrap, p-value `= mean(stat_obs ≥ stat_boot)`.
//!
//! Small p-values are evidence against the ESAG model.

use super::refit_config;
use crate::error::Result;
use crate::regression::EsagRegressorFit;
use crate::stats::ks::{ks_1samp_chisq, ks_2samp};
use nalgebra::DMatrix;
use rand::Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Which goodness-of-fit statistic to use.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum GofMethod {
    /// Yu & Huang (2024) Algorithm 1 (two-sample KS against a model-based reference sample).
    #[default]
    TwoSampleReference,
    /// The author's R `Bootstrap_GoF` (one-sample KS of `T1` against χ²_{d−1}).
    OneSampleChiSquare,
}

/// Result of the bootstrap goodness-of-fit test.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GofResult {
    /// The statistic used.
    pub method: GofMethod,
    /// KS distance for the observed data.
    pub ks_statistic: f64,
    /// KS p-value for the observed data (the test statistic `KS_p`).
    pub ks_p_value: f64,
    /// Bootstrap goodness-of-fit p-value.
    pub p_value: f64,
    /// Number of successful bootstrap replicates.
    pub n_boot: usize,
    /// Replicates whose refit failed (excluded).
    pub n_failed: usize,
    /// Bootstrap KS p-values.
    pub boot_ks_p_values: Vec<f64>,
}

/// `(KS distance, KS p-value)` of the residuals of `fit` on `(x, y)`.
fn ks_stat<R: Rng + ?Sized>(
    fit: &EsagRegressorFit,
    x: &DMatrix<f64>,
    y: &DMatrix<f64>,
    method: GofMethod,
    rng: &mut R,
) -> Result<(f64, f64)> {
    let t1 = fit.residuals(x, y)?.t1;
    match method {
        GofMethod::OneSampleChiSquare => {
            let k = ks_1samp_chisq(&t1, (fit.d - 1) as f64);
            Ok((k.statistic, k.p_value))
        }
        GofMethod::TwoSampleReference => {
            let y_ref = fit.sample_responses(x, rng)?;
            let t_ref = fit.residuals(x, &y_ref)?.t1;
            let k = ks_2samp(&t1, &t_ref);
            Ok((k.statistic, k.p_value))
        }
    }
}

/// Parametric-bootstrap goodness-of-fit test.
///
/// `n_boot` replicates run in parallel; replicate `b` uses the RNG
/// its own random stream derived from `(seed, b)`, so results do not depend on the
/// number of threads.
pub fn bootstrap_gof(
    fit: &EsagRegressorFit,
    x: &DMatrix<f64>,
    y: &DMatrix<f64>,
    method: GofMethod,
    n_boot: usize,
    seed: u64,
) -> Result<GofResult> {
    let mut rng = crate::stream_rng(seed, crate::streams::GOF, 0);
    let (d_obs, p_obs) = ks_stat(fit, x, y, method, &mut rng)?;
    let cfg = refit_config(fit);

    let boot: Vec<Option<f64>> = (0..n_boot)
        .into_par_iter()
        .map(|b| {
            let mut rng = crate::stream_rng(seed, crate::streams::GOF, 1 + b as u64);
            let yb = fit.sample_responses(x, &mut rng).ok()?;
            let refit = cfg.fit_warm(x, &yb, &fit.coef).ok()?;
            ks_stat(&refit, x, &yb, method, &mut rng)
                .ok()
                .map(|(_, p)| p)
        })
        .collect();
    let pvals: Vec<f64> = boot.iter().flatten().copied().collect();
    let n_failed = n_boot - pvals.len();
    let nb = pvals.len().max(1) as f64;
    let count = match method {
        GofMethod::TwoSampleReference => pvals.iter().filter(|&&p| p < p_obs).count(),
        GofMethod::OneSampleChiSquare => pvals.iter().filter(|&&p| p_obs >= p).count(),
    };
    Ok(GofResult {
        method,
        ks_statistic: d_obs,
        ks_p_value: p_obs,
        p_value: count as f64 / nb,
        n_boot: pvals.len(),
        n_failed,
        boot_ks_p_values: pvals,
    })
}
