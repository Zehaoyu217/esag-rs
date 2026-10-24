//! Monte-Carlo confidence and prediction regions on the sphere
//! (R: `Confidence_Region`, `Prediction_Region`).
//!
//! For a distribution `ESAG(mu, gamma)` the region statistic of a point `y` is
//! `q(y) = (y − μ̂)ᵀ V⁻¹ (y − μ̂)`. A level-`1−α` region is `{y : q(y) ≤ q_{1−α}}`
//! where `q_{1−α}` is the empirical `1−α` quantile of `q` over Monte-Carlo
//! draws from the distribution. For a *prediction* region at a new covariate
//! value the draws from the plug-in MLE are pooled with draws from
//! nonparametric-bootstrap refits, which accounts for estimation uncertainty.

use super::refit_config;
use crate::distribution::Esag;
use crate::error::{EsagError, Result};
use crate::regression::EsagRegressorFit;
use crate::stats::quantile::quantiles_type7;
use nalgebra::DMatrix;
use rand::{Rng, RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Sample `m` region statistics `q = (Y − μ̂)ᵀ V⁻¹ (Y − μ̂)` with `Y ~ dist`
/// (R: `Confidence_Region`).
pub fn confidence_region_sample<R: Rng + ?Sized>(dist: &Esag, m: usize, rng: &mut R) -> Vec<f64> {
    let frame = dist.frame();
    let d = dist.dim();
    let mut buf = vec![0.0; d];
    (0..m)
        .map(|_| {
            dist.sample_one(&frame, rng, &mut buf);
            crate::distribution::region_statistic(&frame, &buf)
        })
        .collect()
}

/// A prediction region for one covariate vector.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PredictionRegion {
    /// The fitted distribution at the covariate vector (plug-in MLE).
    pub dist: Esag,
    /// Coverage levels (e.g. 0.95, 0.90, …).
    pub levels: Vec<f64>,
    /// Thresholds `q_level`: `y` is inside the level-`l` region iff
    /// `dist.region_statistic(y) <= thresholds[l]`.
    pub thresholds: Vec<f64>,
    /// Number of pooled Monte-Carlo statistics.
    pub n_pooled: usize,
    /// Number of successful bootstrap refits.
    pub n_boot: usize,
}

impl PredictionRegion {
    /// Whether `y` lies inside the region at `levels[idx]`.
    pub fn contains(&self, y: &[f64], idx: usize) -> bool {
        self.dist.region_statistic(y) <= self.thresholds[idx]
    }
}

/// Prediction region at the covariate vector `x_new` (length = number of
/// features). `n_mc` Monte-Carlo draws from the plug-in distribution are pooled
/// with `n_mc` draws from each of `n_boot` nonparametric bootstrap refits
/// (rows of `(x, y)` resampled with replacement). `n_boot = 0` gives the
/// plug-in region. (R: `Prediction_Region`, which fixes `index` to a training
/// row; here any `x_new` is allowed.)
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
    if x_new.len() != fit.n_features {
        return Err(EsagError::Shape(format!(
            "x_new has {} entries but the model has {} features",
            x_new.len(),
            fit.n_features
        )));
    }
    let xn = DMatrix::from_row_slice(1, x_new.len(), x_new);
    let dist = fit.predict_distribution(&xn)?.remove(0);
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut pooled = confidence_region_sample(&dist, n_mc, &mut rng);

    let n = y.nrows();
    let cfg = refit_config(fit);
    let boot: Vec<Option<Vec<f64>>> = (0..n_boot)
        .into_par_iter()
        .map(|b| {
            let mut rng = ChaCha8Rng::seed_from_u64(seed.wrapping_add(1 + b as u64));
            let idx: Vec<usize> = (0..n).map(|_| rng.random_range(0..n)).collect();
            let xb = DMatrix::from_fn(n, x.ncols(), |i, j| x[(idx[i], j)]);
            let yb = DMatrix::from_fn(n, y.ncols(), |i, j| y[(idx[i], j)]);
            let refit = cfg.fit_warm(&xb, &yb, &fit.coef).ok()?;
            let db = refit.predict_distribution(&xn).ok()?.remove(0);
            Some(confidence_region_sample(&db, n_mc, &mut rng))
        })
        .collect();
    let mut ok = 0;
    for b in boot.into_iter().flatten() {
        pooled.extend(b);
        ok += 1;
    }
    let thresholds = quantiles_type7(&pooled, levels);
    Ok(PredictionRegion {
        dist,
        levels: levels.to_vec(),
        thresholds,
        n_pooled: pooled.len(),
        n_boot: ok,
    })
}

/// Empirical coverage of plug-in prediction regions on held-out data: for each
/// test row, build the plug-in region at its covariates (`n_mc` draws) and
/// check whether the observed response falls inside. Returns one coverage
/// proportion per level.
pub fn region_coverage(
    fit: &EsagRegressorFit,
    x_test: &DMatrix<f64>,
    y_test: &DMatrix<f64>,
    n_mc: usize,
    levels: &[f64],
    seed: u64,
) -> Result<Vec<f64>> {
    let dists = fit.predict_distribution(x_test)?;
    let d = fit.d;
    let hits: Vec<Vec<bool>> = dists
        .par_iter()
        .enumerate()
        .map(|(i, dist)| {
            let mut rng = ChaCha8Rng::seed_from_u64(seed.wrapping_add(i as u64));
            let q = confidence_region_sample(dist, n_mc, &mut rng);
            let th = quantiles_type7(&q, levels);
            let yi: Vec<f64> = (0..d).map(|k| y_test[(i, k)]).collect();
            let qi = dist.region_statistic(&yi);
            th.iter().map(|&t| qi <= t).collect()
        })
        .collect();
    let n = hits.len().max(1) as f64;
    Ok((0..levels.len())
        .map(|l| hits.iter().filter(|h| h[l]).count() as f64 / n)
        .collect())
}
