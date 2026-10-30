//! Nonparametric (pairs) bootstrap of the coefficient matrix.
//!
//! Yu & Huang (2024, 2025) recommend the bootstrap over Hessian-based standard
//! errors: the likelihood is non-smooth at `gamma = 0` and `gamma` is only
//! identified up to an equivalence class (eigenvector signs). Every refit is
//! canonicalised with [`crate::regression::canonicalize_coef`], which removes
//! the first-group sign ambiguity; for `d ≥ 4` further equivalent `gamma`
//! exist, so prefer bootstrap summaries of `mu` and of the fitted `V`
//! (see [`BootstrapResult::fits`]) over raw `gamma` coefficients there.

use super::refit_config;
use crate::error::Result;
use crate::regression::EsagRegressorFit;
use crate::stats::quantile::quantiles_type7;
use nalgebra::DMatrix;
use rand::RngExt;
use rayon::prelude::*;

/// Bootstrap replicates of a fitted regression.
#[derive(Clone, Debug)]
pub struct BootstrapResult {
    /// Refitted models (one per successful replicate).
    pub fits: Vec<EsagRegressorFit>,
    /// Bootstrap standard deviation of every coefficient (`b_dim × p_total`).
    pub coef_se: DMatrix<f64>,
    /// Percentile intervals `(lo, hi)` of every coefficient at the requested level.
    pub coef_ci: (DMatrix<f64>, DMatrix<f64>),
    /// Replicates whose refit failed.
    pub n_failed: usize,
}

/// Resample rows of `(x, y)` with replacement `n_boot` times and refit
/// (warm-started at the original MLE). `level` is the central coverage of the
/// percentile intervals (e.g. 0.95).
pub fn bootstrap_coef(
    fit: &EsagRegressorFit,
    x: &DMatrix<f64>,
    y: &DMatrix<f64>,
    n_boot: usize,
    level: f64,
    seed: u64,
) -> Result<BootstrapResult> {
    let n = y.nrows();
    let cfg = refit_config(fit);
    let fits: Vec<Option<EsagRegressorFit>> = (0..n_boot)
        .into_par_iter()
        .map(|b| {
            let mut rng = crate::stream_rng(seed, crate::streams::COEF_BOOT, b as u64);
            let idx: Vec<usize> = (0..n).map(|_| rng.random_range(0..n)).collect();
            let xb = DMatrix::from_fn(n, x.ncols(), |i, j| x[(idx[i], j)]);
            let yb = DMatrix::from_fn(n, y.ncols(), |i, j| y[(idx[i], j)]);
            cfg.fit_warm(&xb, &yb, &fit.coef).ok()
        })
        .collect();
    let fits: Vec<EsagRegressorFit> = fits.into_iter().flatten().collect();
    let n_failed = n_boot - fits.len();
    let (r, c) = (fit.coef.nrows(), fit.coef.ncols());
    let mut se = DMatrix::from_element(r, c, f64::NAN);
    let mut lo = se.clone();
    let mut hi = se.clone();
    let a = (1.0 - level) / 2.0;
    for k in 0..r {
        for j in 0..c {
            let v: Vec<f64> = fits.iter().map(|f| f.coef[(k, j)]).collect();
            if v.len() >= 2 {
                let m = v.iter().sum::<f64>() / v.len() as f64;
                let var = v.iter().map(|t| (t - m).powi(2)).sum::<f64>() / (v.len() - 1) as f64;
                se[(k, j)] = var.sqrt();
                let q = quantiles_type7(&v, &[a, 1.0 - a]);
                lo[(k, j)] = q[0];
                hi[(k, j)] = q[1];
            }
        }
    }
    Ok(BootstrapResult {
        fits,
        coef_se: se,
        coef_ci: (lo, hi),
        n_failed,
    })
}
