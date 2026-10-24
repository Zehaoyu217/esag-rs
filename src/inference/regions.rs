//! Monte-Carlo confidence and prediction regions on the sphere
//! (R: `Confidence_Region`, `Prediction_Region`; Yu & Huang 2025, Algorithm 4).
//!
//! For `ESAG(mu, gamma)` the region statistic of a point `y` is
//! `q(y) = (y − μ̂)ᵀ V⁻¹ (y − μ̂)`, and a level-`1−a` region is
//! `{y : q(y) ≤ q_{1−a}}`. `q_{1−a}` is the empirical quantile of `q` over
//! Monte-Carlo draws. For a *prediction* region at a covariate value, the
//! draws from the plug-in MLE are pooled with draws from nonparametric
//! (pairs) bootstrap refits, which accounts for estimation uncertainty.
//! Plug-in regions (`n_boot = 0`) under-cover in finite samples.
//!
//! The bootstrap refits depend only on the training data, so
//! [`prediction_regions`] fits them once and reuses them for every
//! requested covariate row.

use super::refit_config;
use crate::distribution::Esag;
use crate::error::{EsagError, Result};
use crate::regression::EsagRegressorFit;
use crate::stats::quantile::quantiles_type7;
use nalgebra::DMatrix;
use rand::{Rng, RngExt, SeedableRng};
use crate::FastRng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Sample `m` region statistics `q = (Y − μ̂)ᵀ V⁻¹ (Y − μ̂)` with `Y ~ dist`
/// (R: `Confidence_Region`).
pub fn confidence_region_sample<R: Rng + ?Sized>(dist: &Esag, m: usize, rng: &mut R) -> Vec<f64> {
    let sampler = dist.sampler();
    (0..m).map(|_| sampler.region_statistic_draw(rng)).collect()
}

/// A prediction region for one covariate vector.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PredictionRegion {
    /// The plug-in distribution at the covariate vector; the region is
    /// centred at its mean direction with its `V⁻¹` as shape.
    pub dist: Esag,
    /// Coverage levels (e.g. 0.95, 0.90, …).
    pub levels: Vec<f64>,
    /// Thresholds: `y` is inside the level-`levels[l]` region iff
    /// `dist.region_statistic(y) <= thresholds[l]`.
    pub thresholds: Vec<f64>,
    /// Number of pooled Monte-Carlo statistics.
    pub n_pooled: usize,
    /// Number of bootstrap refits that contributed.
    pub n_boot: usize,
}

impl PredictionRegion {
    /// Whether `y` lies inside the region at `levels[idx]`.
    pub fn contains(&self, y: &[f64], idx: usize) -> bool {
        self.dist.region_statistic(y) <= self.thresholds[idx]
    }
}

/// `n_boot` pairs-bootstrap refits of `fit`, warm-started at its estimate.
/// Replicate `b` uses seed `seed + 1 + b`. Failed refits are skipped.
pub fn bootstrap_refits(
    fit: &EsagRegressorFit,
    x: &DMatrix<f64>,
    y: &DMatrix<f64>,
    n_boot: usize,
    seed: u64,
) -> Vec<EsagRegressorFit> {
    let n = y.nrows();
    let cfg = refit_config(fit);
    (0..n_boot)
        .into_par_iter()
        .filter_map(|b| {
            let mut rng = FastRng::seed_from_u64(seed.wrapping_add(1 + b as u64));
            let idx: Vec<usize> = (0..n).map(|_| rng.random_range(0..n)).collect();
            let xb = DMatrix::from_fn(n, x.ncols(), |i, j| x[(idx[i], j)]);
            let yb = DMatrix::from_fn(n, y.ncols(), |i, j| y[(idx[i], j)]);
            cfg.fit_warm(&xb, &yb, &fit.coef).ok()
        })
        .collect()
}

/// Prediction regions (Algorithm 4) for every row of `x_new`
/// (`m × n_features`). For each row, `n_mc` draws from the plug-in
/// distribution are pooled with `n_mc` draws from each of `n_boot` bootstrap
/// refits; `n_boot = 0` gives plug-in regions.
#[allow(clippy::too_many_arguments)]
pub fn prediction_regions(
    fit: &EsagRegressorFit,
    x: &DMatrix<f64>,
    y: &DMatrix<f64>,
    x_new: &DMatrix<f64>,
    n_mc: usize,
    n_boot: usize,
    levels: &[f64],
    seed: u64,
) -> Result<Vec<PredictionRegion>> {
    if x_new.ncols() != fit.n_features {
        return Err(EsagError::Shape(format!(
            "x_new has {} columns but the model has {} features",
            x_new.ncols(),
            fit.n_features
        )));
    }
    let refits = bootstrap_refits(fit, x, y, n_boot, seed);
    let plug = fit.predict_distribution(x_new)?;
    let boot: Vec<Vec<Esag>> = refits
        .iter()
        .map(|f| f.predict_distribution(x_new))
        .collect::<Result<_>>()?;
    Ok(plug
        .into_par_iter()
        .enumerate()
        .map(|(i, dist)| {
            let mut rng =
                FastRng::seed_from_u64(seed.wrapping_mul(0x9E37_79B9).wrapping_add(i as u64));
            let mut pooled = confidence_region_sample(&dist, n_mc, &mut rng);
            for b in &boot {
                pooled.extend(confidence_region_sample(&b[i], n_mc, &mut rng));
            }
            PredictionRegion {
                thresholds: quantiles_type7(&pooled, levels),
                n_pooled: pooled.len(),
                n_boot: boot.len(),
                levels: levels.to_vec(),
                dist,
            }
        })
        .collect())
}

/// Prediction region at one covariate vector `x_new` (R: `Prediction_Region`,
/// which fixes `x_new` to a training row).
#[allow(clippy::too_many_arguments)]
pub fn prediction_region(
    fit: &EsagRegressorFit,
    x: &DMatrix<f64>,
    y: &DMatrix<f64>,
    x_new: &[f64],
    n_mc: usize,
    n_boot: usize,
    levels: &[f64],
    seed: u64,
) -> Result<PredictionRegion> {
    let xn = DMatrix::from_row_slice(1, x_new.len(), x_new);
    Ok(prediction_regions(fit, x, y, &xn, n_mc, n_boot, levels, seed)?.remove(0))
}

/// Empirical coverage on held-out data: build the prediction region at each
/// test row's covariates and check whether its observed response falls
/// inside. Returns one proportion per level. `n_boot = 0` evaluates plug-in
/// regions.
#[allow(clippy::too_many_arguments)]
pub fn region_coverage(
    fit: &EsagRegressorFit,
    x_train: &DMatrix<f64>,
    y_train: &DMatrix<f64>,
    x_test: &DMatrix<f64>,
    y_test: &DMatrix<f64>,
    n_mc: usize,
    n_boot: usize,
    levels: &[f64],
    seed: u64,
) -> Result<Vec<f64>> {
    let regions = prediction_regions(fit, x_train, y_train, x_test, n_mc, n_boot, levels, seed)?;
    let d = fit.d;
    let n = regions.len().max(1) as f64;
    Ok((0..levels.len())
        .map(|l| {
            regions
                .iter()
                .enumerate()
                .filter(|(i, r)| {
                    let yi: Vec<f64> = (0..d).map(|k| y_test[(*i, k)]).collect();
                    r.contains(&yi, l)
                })
                .count() as f64
                / n
        })
        .collect())
}
