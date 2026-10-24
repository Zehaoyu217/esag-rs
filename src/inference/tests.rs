//! Bootstrap hypothesis tests (R: `hypothesis_testing_isotropy`,
//! `hypothesis_testing_mu`, `hypothesis_testing_gamma`, `stats_gamma`,
//! `mome_stats`).
//!
//! A restricted model (null) is compared with the full model using four
//! statistics: `RoC` (mean ratio of concentrations `|mu_a,i|/|mu_0,i|`), `D`
//! (`mean_i (2 − cos∠(mu_0,i, mu_a,i)) · RoC_i`), `M` (a second-moment
//! discrepancy; Yu & Huang estimate the moments with 10⁴ Monte-Carlo draws,
//! `esag` computes them exactly by default) and the likelihood
//! ratio `LR = −2(ll_0 − ll_a)`.
//! Null distributions come from a parametric bootstrap under the restricted
//! fit. Unlike the R code, the bootstrap responses are simulated observation
//! by observation from their own fitted parameters (the R code indexes the
//! parameter matrix with the bootstrap counter), and every `d` is supported.

use super::refit_config;
use crate::distribution::second_moment_diag_from_frame;
use crate::error::{EsagError, Result};
use crate::frame::Frame;
use crate::regression::{
    EsagRegressorFit, Shape, gamma_restricted_mask, isotropic_mask, mu_restricted_mask,
};
use crate::stats::chisq::chisq_sf;
use nalgebra::DMatrix;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// The null hypothesis to test against the full model.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Restriction {
    /// All `gamma` coefficients are zero (isotropic angular Gaussian errors).
    Isotropy,
    /// The `mu` coefficients of feature `j` (0-based, excluding intercept) are zero.
    MuOfFeature(usize),
    /// The `gamma` coefficients of feature `j` (0-based, excluding intercept) are zero.
    GammaOfFeature(usize),
    /// Both `mu` and `gamma` coefficients of feature `j` are zero.
    Feature(usize),
    /// A custom free-parameter mask (`eta_len(d) × p_total`).
    Custom(DMatrix<bool>),
}

/// Results of a bootstrap hypothesis test.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HypothesisTestResult {
    /// Observed statistics.
    pub roc: f64,
    pub d_stat: f64,
    pub m_stat: f64,
    pub lr: f64,
    /// Bootstrap p-values `mean(stat_obs < stat_boot)`.
    pub p_roc: f64,
    pub p_d: f64,
    pub p_m: f64,
    pub p_lr: f64,
    /// Asymptotic χ² p-value for `LR` with `df` restricted parameters
    /// (indicative only; for the isotropy null the parameter lies on a
    /// non-differentiable point of the parameterization).
    pub p_lr_chisq: f64,
    pub df: usize,
    /// Log-likelihoods of the restricted and full fits.
    pub loglik_restricted: f64,
    pub loglik_full: f64,
    /// The restricted fit's coefficient matrix.
    pub coef_restricted: DMatrix<f64>,
    pub n_boot: usize,
    pub n_failed: usize,
    pub boot_roc: Vec<f64>,
    pub boot_d: Vec<f64>,
    pub boot_m: Vec<f64>,
    pub boot_lr: Vec<f64>,
}

fn restriction_mask(fit: &EsagRegressorFit, r: &Restriction) -> Result<DMatrix<bool>> {
    let d = fit.d;
    let pt = fit.p_total();
    let off = usize::from(fit.fit_intercept);
    let col = |j: usize| -> Result<usize> {
        if j >= fit.n_features {
            Err(EsagError::Shape(format!(
                "feature index {j} out of range (model has {} features)",
                fit.n_features
            )))
        } else {
            Ok(j + off)
        }
    };
    let m = match r {
        Restriction::Isotropy => isotropic_mask(d, pt),
        Restriction::MuOfFeature(j) => mu_restricted_mask(d, pt, col(*j)?),
        Restriction::GammaOfFeature(j) => gamma_restricted_mask(d, pt, col(*j)?),
        Restriction::Feature(j) => {
            let c = col(*j)?;
            let mut m = mu_restricted_mask(d, pt, c);
            for k in d..m.nrows() {
                m[(k, c)] = false;
            }
            m
        }
        Restriction::Custom(m) => m.clone(),
    };
    // the restriction must be nested in the full fit's free set
    for k in 0..m.nrows() {
        for j in 0..m.ncols() {
            if m[(k, j)] && !fit.mask[(k, j)] {
                return Err(EsagError::Shape(
                    "restriction frees a parameter that is fixed in the full model".into(),
                ));
            }
        }
    }
    Ok(m)
}

/// `RoC` and `D` from the `mu` parts of two linear predictors (`n × b_dim`).
fn roc_and_d(eta0: &DMatrix<f64>, eta_a: &DMatrix<f64>, d: usize) -> (f64, f64) {
    let n = eta0.nrows();
    let mut roc = 0.0;
    let mut ds = 0.0;
    for i in 0..n {
        let mut n0 = 0.0;
        let mut na = 0.0;
        let mut dot = 0.0;
        for k in 0..d {
            let a = eta0[(i, k)];
            let b = eta_a[(i, k)];
            n0 += a * a;
            na += b * b;
            dot += a * b;
        }
        let (n0, na) = (n0.sqrt(), na.sqrt());
        let r = na / n0;
        roc += r;
        ds += (2.0 - dot / (n0 * na)) * r;
    }
    (roc / n as f64, ds / n as f64)
}

/// `M = ‖ mean_i ( y_i² − E[Y_i²; eta_0,i] ) ‖²` (elementwise squares).
///
/// With `n_mc = 0` the second moments are exact
/// ([`crate::distribution::second_moment_weights`], ~1 µs per observation);
/// with `n_mc > 0` they are estimated by `n_mc` Monte-Carlo draws per
/// observation as in the R code (`mome_stats`), which adds sampling noise of
/// order `n_mc^{−1/2}` and costs ~25 µs per 1000 draws. This is the squared
/// norm, as computed by R's `mome_stats` and as reported in the papers'
/// tables (CSDA 2025 eq. (4) writes the norm; bootstrap p-values are
/// identical either way).
fn moment_stat(
    fit0: &EsagRegressorFit,
    x: &DMatrix<f64>,
    y: &DMatrix<f64>,
    n_mc: usize,
    seed: u64,
    replicate: u64,
    parallel: bool,
) -> Result<f64> {
    let d = fit0.d;
    let n = y.nrows();
    let diffs: Vec<Vec<f64>> = if n_mc == 0 {
        let eta = fit0.linear_predictor(x)?;
        let bd = eta.ncols();
        let lt = fit0.lt_frame();
        let one = |i: usize| -> Vec<f64> {
            let row: Vec<f64> = (0..bd).map(|k| eta[(i, k)]).collect();
            let frame = match &lt {
                None => Frame::new(&row[..d], &row[d..]),
                Some(_) => fit0
                    .distribution_at(&row, lt.as_ref())
                    .expect("valid log-tangent parameters")
                    .frame(),
            };
            let ey2 = second_moment_diag_from_frame(&frame);
            (0..d).map(|k| y[(i, k)].powi(2) - ey2[k]).collect()
        };
        if parallel {
            (0..n).into_par_iter().with_min_len(64).map(one).collect()
        } else {
            (0..n).map(one).collect()
        }
    } else {
        let dists = fit0.predict_distribution(x)?;
        dists
            .par_iter()
            .enumerate()
            .map(|(i, dist)| {
                let mut rng = crate::stream_rng(
                    seed,
                    crate::streams::TEST_MOMENTS + (replicate << 8),
                    i as u64,
                );
                let sampler = dist.sampler();
                let mut buf = vec![0.0; d];
                let mut ey2 = vec![0.0; d];
                for _ in 0..n_mc {
                    sampler.sample_into(&mut rng, &mut buf);
                    for k in 0..d {
                        ey2[k] += buf[k] * buf[k];
                    }
                }
                (0..d)
                    .map(|k| y[(i, k)].powi(2) - ey2[k] / n_mc as f64)
                    .collect()
            })
            .collect()
    };
    let nf = n as f64;
    let mut m = 0.0;
    for k in 0..d {
        let mean_k: f64 = diffs.iter().map(|r| r[k]).sum::<f64>() / nf;
        m += mean_k * mean_k;
    }
    Ok(m)
}

/// Run a bootstrap hypothesis test of `restriction` against the full fit.
///
/// `n_boot` parametric-bootstrap replicates from the restricted fit (each
/// refitting both models, warm-started). `n_mc = 0` (recommended) computes
/// the moments in the `M` statistic exactly; `n_mc > 0` uses that many
/// Monte-Carlo draws per observation, as the R code does. Every replicate and every Monte-Carlo
/// moment has its own random stream derived from `seed`.
pub fn hypothesis_test(
    fit_full: &EsagRegressorFit,
    x: &DMatrix<f64>,
    y: &DMatrix<f64>,
    restriction: Restriction,
    n_boot: usize,
    n_mc: usize,
    seed: u64,
) -> Result<HypothesisTestResult> {
    let d = fit_full.d;
    let mask0 = restriction_mask(fit_full, &restriction)?;
    let df = fit_full
        .free
        .iter()
        .filter(|&&(k, j)| !mask0[(k, j)])
        .count();

    // restricted start: full MLE with the restricted entries zeroed
    let mut b0 = fit_full.coef.clone();
    for k in 0..b0.nrows() {
        for j in 0..b0.ncols() {
            if !mask0[(k, j)] {
                b0[(k, j)] = 0.0;
            }
        }
    }
    let mut cfg0 = refit_config(fit_full);
    cfg0.mask = Some(mask0.clone());
    cfg0.precondition = None;
    let mut cfg_a = refit_config(fit_full);
    // full refits start far from the original full optimum (restricted fit
    // plus small extras, or the original fit on null data), where its
    // observed information is a poor metric
    cfg_a.precondition = None;
    let fit0 = {
        // once, with its observed information (the preconditioner below)
        let mut once = cfg0.clone();
        once.compute_hessian = true;
        let warm = once.fit_warm(x, y, &b0)?;
        let mut auto = once.clone();
        auto.init = crate::regression::Init::Auto;
        match auto.fit(x, y) {
            Ok(a) if a.log_likelihood > warm.log_likelihood => a,
            _ => warm,
        }
    };
    cfg0.precondition = fit0
        .cov_params
        .as_ref()
        .and_then(crate::regression::precondition_from_cov);

    let eta0 = fit0.linear_predictor(x)?;
    let eta_a = fit_full.linear_predictor(x)?;
    let (roc, d_stat) = roc_and_d(&eta0, &eta_a, d);
    let m_stat = moment_stat(&fit0, x, y, n_mc, seed, 0, true)?;
    let lr = -2.0 * (fit0.log_likelihood - fit_full.log_likelihood);

    let boot: Vec<Option<(f64, f64, f64, f64)>> = (0..n_boot)
        .into_par_iter()
        .map(|b| {
            let mut rng = crate::stream_rng(seed, crate::streams::TEST_BOOT, b as u64);
            let yb = fit0.sample_responses(x, &mut rng).ok()?;
            let f0 = cfg0.fit_warm(x, &yb, &fit0.coef).ok()?;
            // full-model start: restricted MLE, with the restricted mu entries
            // set to 1 (the R code's recommendation) and gamma entries small
            let mut ba = f0.coef.clone();
            for &(k, j) in &fit_full.free {
                if !mask0[(k, j)] {
                    ba[(k, j)] = if k < d { 1.0 } else { 1e-3 };
                }
            }
            // full refit: best of two starts (from the restricted refit, and
            // from the original full MLE) so that LR^(b) is not deflated by a
            // refit stuck below the maximum
            let fa1 = cfg_a.fit_warm(x, &yb, &ba).ok();
            // the second start guards against the local maxima that the gamma
            // parameterization creates near its singular set (the cone at
            // gamma = 0 above all); the log-tangent likelihood is smooth
            // there, so it is only used if the first refit fails
            let need_second = fit_full.config.shape == Shape::Gamma
                || !fa1.as_ref().is_some_and(|f| f.converged);
            let fa2 = if need_second {
                cfg_a.fit_warm(x, &yb, &fit_full.coef).ok()
            } else {
                None
            };
            let fa = match (fa1, fa2) {
                (Some(a), Some(b)) => {
                    if a.log_likelihood >= b.log_likelihood {
                        a
                    } else {
                        b
                    }
                }
                (Some(a), None) | (None, Some(a)) => a,
                (None, None) => return None,
            };
            let e0 = f0.linear_predictor(x).ok()?;
            let ea = fa.linear_predictor(x).ok()?;
            let (r, dd) = roc_and_d(&e0, &ea, d);
            // sequential: replicates already run in parallel
            let m = moment_stat(&f0, x, &yb, n_mc, seed, 1 + b as u64, n_mc > 0).ok()?;
            let l = -2.0 * (f0.log_likelihood - fa.log_likelihood);
            Some((r, dd, m, l))
        })
        .collect();

    let mut br = vec![];
    let mut bd = vec![];
    let mut bm = vec![];
    let mut bl = vec![];
    let mut n_failed = 0;
    for r in boot {
        match r {
            Some((a, b, c, e)) => {
                br.push(a);
                bd.push(b);
                bm.push(c);
                bl.push(e);
            }
            None => n_failed += 1,
        }
    }
    let nb = br.len().max(1) as f64;
    let pv = |obs: f64, v: &[f64]| v.iter().filter(|&&s| obs < s).count() as f64 / nb;
    Ok(HypothesisTestResult {
        roc,
        d_stat,
        m_stat,
        lr,
        p_roc: pv(roc, &br),
        p_d: pv(d_stat, &bd),
        p_m: pv(m_stat, &bm),
        p_lr: pv(lr, &bl),
        p_lr_chisq: chisq_sf(lr.max(0.0), df as f64),
        df,
        loglik_restricted: fit0.log_likelihood,
        loglik_full: fit_full.log_likelihood,
        coef_restricted: fit0.coef,
        n_boot: br.len(),
        n_failed,
        boot_roc: br,
        boot_d: bd,
        boot_m: bm,
        boot_lr: bl,
    })
}

#[cfg(test)]
mod timing {
    use super::*;
    use crate::regression::EsagRegressor;
    use rand::SeedableRng;
    use std::time::Instant;

    /// `cargo test --release --lib inference::tests::timing -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn replicate_breakdown() {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(3);
        let n = 1000;
        let x = DMatrix::from_fn(n, 1, |i, _| ((i * 37 % 101) as f64 / 50.0) - 1.0);
        let b =
            DMatrix::from_row_slice(5, 2, &[1.5, -0.6, 0.5, 1.2, 1.0, 0.3, 0.8, 0.5, -0.4, 0.7]);
        let full0 = EsagRegressor::new()
            .with_init(crate::regression::Init::Custom(b))
            .with_max_iter(0)
            .fit(
                &x,
                &DMatrix::from_fn(n, 3, |_, k| if k == 2 { 1.0 } else { 0.0 }),
            )
            .unwrap();
        let y = full0.sample_responses(&x, &mut rng).unwrap();
        let fit = EsagRegressor::new().fit(&x, &y).unwrap();
        let mask0 = restriction_mask(&fit, &Restriction::Isotropy).unwrap();
        let mut cfg0 = refit_config(&fit);
        cfg0.mask = Some(mask0.clone());
        cfg0.precondition = None;
        let cfg_a = refit_config(&fit);
        let mut b0 = fit.coef.clone();
        for k in 3..5 {
            for j in 0..2 {
                b0[(k, j)] = 0.0;
            }
        }
        let mut once = cfg0.clone();
        once.compute_hessian = true;
        let fit0 = once.fit_warm(&x, &y, &b0).unwrap();
        cfg0.precondition = fit0
            .cov_params
            .as_ref()
            .and_then(crate::regression::precondition_from_cov);
        let reps = 200;
        let (mut t_s, mut t_0, mut t_a, mut t_m) = (0.0, 0.0, 0.0, 0.0);
        let (mut it0, mut ita) = (0, 0);
        for r in 0..reps {
            let mut rng = crate::stream_rng(1, 5, r);
            let t = Instant::now();
            let yb = fit0.sample_responses(&x, &mut rng).unwrap();
            t_s += t.elapsed().as_secs_f64();
            let t = Instant::now();
            let f0 = cfg0.fit_warm(&x, &yb, &fit0.coef).unwrap();
            t_0 += t.elapsed().as_secs_f64();
            it0 += f0.n_iter;
            let mut ba = f0.coef.clone();
            for k in 3..5 {
                for j in 0..2 {
                    ba[(k, j)] = 1e-3;
                }
            }
            let t = Instant::now();
            let fa = cfg_a.fit_warm(&x, &yb, &ba).unwrap();
            let fa2 = cfg_a.fit_warm(&x, &yb, &fit.coef).unwrap();
            t_a += t.elapsed().as_secs_f64();
            ita += fa.n_iter + fa2.n_iter;

            let t = Instant::now();
            let _ = moment_stat(&f0, &x, &yb, 0, 1, 1, false).unwrap();
            t_m += t.elapsed().as_secs_f64();
        }
        let r = reps as f64;
        println!(
            "per replicate (single thread for fits): sample {:.2} ms, restricted refit {:.2} ms ({} it), 2 full refits {:.2} ms ({} it), exact M {:.2} ms",
            t_s / r * 1e3,
            t_0 / r * 1e3,
            it0 / reps as usize,
            t_a / r * 1e3,
            ita / reps as usize,
            t_m / r * 1e3
        );
    }
}
