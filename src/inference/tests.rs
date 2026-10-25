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
use crate::distribution::{second_moment_diag_from_frame, second_moment_diag_into};
use crate::error::{EsagError, Result};
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
    /// The shape coefficients of every feature are zero: the shape of
    /// `V(x)` (its eigenvalues, and its axes relative to `μ(x)`) is the same
    /// at every `x`, generally anisotropic.
    ConstantShape,
    /// A custom free-parameter mask (`eta_len(d) × p_total`).
    Custom(DMatrix<bool>),
}

/// Results of a bootstrap hypothesis test.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HypothesisTestResult {
    /// The null hypothesis, in words.
    #[serde(default)]
    pub null: String,
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
    /// Score (Lagrange-multiplier) statistic at the restricted fit, in the
    /// empirical-information form, and its bootstrap p-value. Like `M` it
    /// needs no full-model fit; unlike `M` it has power against covariate
    /// effects (`docs/MATH.md` §6). `NaN` for the gamma isotropy null:
    /// gamma = 0 is a cone of the parameterization, without a score.
    #[serde(default)]
    pub lm: f64,
    #[serde(default)]
    pub p_lm: f64,
    /// Asymptotic χ² p-value for `LM` with `df` restricted parameters.
    #[serde(default)]
    pub p_lm_chisq: f64,
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
    /// Replicates whose restricted or full refit did not pass the
    /// convergence check (their statistics are still used; for the gamma
    /// shape at d >= 5 with covariates the likelihood has cliffs, see
    /// `docs/MATH.md`).
    #[serde(default)]
    pub n_unconverged: usize,
    pub boot_roc: Vec<f64>,
    pub boot_d: Vec<f64>,
    pub boot_m: Vec<f64>,
    pub boot_lr: Vec<f64>,
    #[serde(default)]
    pub boot_lm: Vec<f64>,
}

/// One bootstrap replicate: RoC, D, M, LR, LM, and whether both refits
/// converged.
type Replicate = (f64, f64, f64, f64, f64, bool);

fn restriction_label(fit: &EsagRegressorFit, r: &Restriction) -> String {
    let name = |j: usize| {
        let off = usize::from(fit.fit_intercept);
        fit.column_names
            .get(j + off)
            .cloned()
            .unwrap_or_else(|| format!("x{}", j + 1))
    };
    match r {
        Restriction::Isotropy => "isotropy (shape = 0)".into(),
        Restriction::MuOfFeature(j) => format!("mu free of {}", name(*j)),
        Restriction::GammaOfFeature(j) => format!("shape free of {}", name(*j)),
        Restriction::Feature(j) => format!("no effect of {}", name(*j)),
        Restriction::ConstantShape => "shape free of every covariate".into(),
        Restriction::Custom(_) => "custom restriction".into(),
    }
}

impl HypothesisTestResult {
    /// statsmodels-style summary: the null, the fits, and each statistic
    /// with its bootstrap p-value (and the χ² approximation for LR).
    pub fn summary(&self) -> String {
        use crate::summary::{WIDTH, header, num};
        use std::fmt::Write as _;
        let left = vec![
            ("Null hypothesis:".to_string(), self.null.clone()),
            (
                "Restricted log-lik:".to_string(),
                num(self.loglik_restricted),
            ),
            ("Full log-lik:".to_string(), num(self.loglik_full)),
        ];
        let right = vec![
            ("Bootstrap replicates:".to_string(), self.n_boot.to_string()),
            (
                "Failed / unconverged:".to_string(),
                format!("{} / {}", self.n_failed, self.n_unconverged),
            ),
            ("Restricted df:".to_string(), self.df.to_string()),
        ];
        let mut s = header("ESAG Bootstrap Hypothesis Test", &left, &right);
        let _ = writeln!(s, "{}", "=".repeat(WIDTH));
        let _ = writeln!(
            s,
            "{:<10}{:>16}{:>18}{:>22}",
            "", "statistic", "P (bootstrap)", "P (asymptotic)"
        );
        let _ = writeln!(s, "{}", "-".repeat(WIDTH));
        // a score test has no bootstrap p-values for RoC, D and LR
        let p = |v: f64| {
            if v.is_nan() {
                "—".to_string()
            } else {
                format!("{v:.3}")
            }
        };
        let _ = writeln!(s, "{:<10}{:>16.6}{:>18}", "RoC", self.roc, p(self.p_roc));
        let _ = writeln!(s, "{:<10}{:>16.6}{:>18}", "D", self.d_stat, p(self.p_d));
        let _ = writeln!(s, "{:<10}{:>16.4e}{:>18}", "M", self.m_stat, p(self.p_m));
        let _ = writeln!(
            s,
            "{:<10}{:>16.4}{:>18}{:>22}",
            "LR",
            self.lr,
            p(self.p_lr),
            format!("{:.3e} (chi2_{})", self.p_lr_chisq, self.df)
        );
        if self.lm.is_finite() {
            let _ = writeln!(
                s,
                "{:<10}{:>16.4}{:>18}{:>22}",
                "LM",
                self.lm,
                p(self.p_lm),
                format!("{:.3e} (chi2_{})", self.p_lm_chisq, self.df)
            );
        }
        let _ = writeln!(s, "{}", "=".repeat(WIDTH));
        s
    }
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
        Restriction::ConstantShape => {
            let mut m = DMatrix::from_element(fit.coef.nrows(), pt, true);
            for k in d..m.nrows() {
                for j in off..pt {
                    m[(k, j)] = false;
                }
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
        if !parallel {
            // one replicate per thread: accumulate in place, in index order
            // (the same sums as the parallel branch)
            let (mut row, mut ey2, mut w) = (vec![0.0; bd], vec![0.0; d], vec![0.0; d - 1]);
            let mut sums = vec![0.0; d];
            for i in 0..n {
                for (k, v) in row.iter_mut().enumerate() {
                    *v = eta[(i, k)];
                }
                let frame = fit0.frame_at(&row, lt.as_ref());
                second_moment_diag_into(&frame, &mut ey2, &mut w);
                for k in 0..d {
                    sums[k] += y[(i, k)].powi(2) - ey2[k];
                }
            }
            let nf = n as f64;
            return Ok(sums.iter().map(|s| (s / nf) * (s / nf)).sum());
        }
        let one = |i: usize| -> Vec<f64> {
            let row: Vec<f64> = (0..bd).map(|k| eta[(i, k)]).collect();
            let frame = fit0.frame_at(&row, lt.as_ref());
            let ey2 = second_moment_diag_from_frame(&frame);
            (0..d).map(|k| y[(i, k)].powi(2) - ey2[k]).collect()
        };
        (0..n).into_par_iter().with_min_len(64).map(one).collect()
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
    run_test(fit_full, x, y, restriction, n_boot, n_mc, seed, true)
}

/// [`hypothesis_test`] with the statistics of the restricted fit only, the
/// score statistic `LM` and `M`: each replicate refits the restricted model
/// and nothing else. The full-model refits are most of the cost of a test
/// (and, for the gamma shape at d ≥ 5 with covariates, its slowest and most
/// multimodal part), so this is 1.4–1.9× faster for the log-tangent shape
/// and 2–4× for gamma (29× for its shape-constant null at d = 5); `LM` has
/// the size and power of `LR` (`docs/MATH.md` §6). RoC, D and LR are reported at the data, without
/// bootstrap p-values (`NaN`).
pub fn score_test(
    fit_full: &EsagRegressorFit,
    x: &DMatrix<f64>,
    y: &DMatrix<f64>,
    restriction: Restriction,
    n_boot: usize,
    n_mc: usize,
    seed: u64,
) -> Result<HypothesisTestResult> {
    run_test(fit_full, x, y, restriction, n_boot, n_mc, seed, false)
}

#[allow(clippy::too_many_arguments)]
fn run_test(
    fit_full: &EsagRegressorFit,
    x: &DMatrix<f64>,
    y: &DMatrix<f64>,
    restriction: Restriction,
    n_boot: usize,
    n_mc: usize,
    seed: u64,
    full_refits: bool,
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
    // the second full refit starts at the original full MLE, where that
    // fit's own information is the metric
    let cfg_a2 = refit_config(fit_full);
    // Full refits on null data start at the restricted refit (extra
    // coefficients at 0) and end near it, not near the original full
    // optimum: precondition them with the full model's observed information
    // at the restricted fit. Not for gamma when the null zeroes every shape
    // coefficient: gamma = 0 is a cone of the parameterization, without a
    // Hessian.
    let shape_zeroed = (d..mask0.nrows()).all(|k| (0..mask0.ncols()).all(|j| !mask0[(k, j)]));
    let gamma_cone = fit_full.config.shape == Shape::Gamma && shape_zeroed;
    let fit0 = {
        // once, with its observed information (the preconditioner below)
        let mut once = cfg0.clone();
        once.compute_hessian = true;
        let warm = once.fit_warm(x, y, &b0)?;
        let mut auto = once.clone();
        auto.init = crate::regression::Init::Auto;
        auto.local_precondition = false;
        match auto.fit(x, y) {
            Ok(a) if a.log_likelihood > warm.log_likelihood => a,
            _ => warm,
        }
    };
    cfg0.precondition = fit0
        .cov_params
        .as_ref()
        .and_then(crate::regression::precondition_from_cov);
    cfg_a.precondition = if gamma_cone || !full_refits {
        None
    } else {
        cfg_a.precondition_at(x, y, &fit0.coef, &fit_full.mask)
    };
    // extra coefficients start at their null value 0 (just off it for gamma,
    // whose likelihood is not differentiable at zero groups)
    let extra_start = |k: usize| {
        if k < d || fit_full.config.shape == Shape::LogTangent {
            0.0
        } else {
            1e-3
        }
    };

    let eta0 = fit0.linear_predictor(x)?;
    let eta_a = fit_full.linear_predictor(x)?;
    let (roc, d_stat) = roc_and_d(&eta0, &eta_a, d);
    let m_stat = moment_stat(&fit0, x, y, n_mc, seed, 0, true)?;
    let lr = -2.0 * (fit0.log_likelihood - fit_full.log_likelihood);
    // the gamma cone at zero shape has no score
    let score_defined = !gamma_cone;
    let lm = if score_defined {
        fit0.score_statistic(x, y, &fit_full.free)?
    } else {
        f64::NAN
    };

    let boot: Vec<Option<Replicate>> = (0..n_boot)
        .into_par_iter()
        .map(|b| {
            let mut rng = crate::stream_rng(seed, crate::streams::TEST_BOOT, b as u64);
            let yb = fit0.sample_responses(x, &mut rng).ok()?;
            let f0 = cfg0.fit_warm(x, &yb, &fit0.coef).ok()?;
            let (r, dd, l, ok_a) = if full_refits {
                // full-model start: the restricted refit with the extra coefficients
                // at their null value (see `extra_start`)
                let mut ba = f0.coef.clone();
                for &(k, j) in &fit_full.free {
                    if !mask0[(k, j)] {
                        ba[(k, j)] = extra_start(k);
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
                    cfg_a2.fit_warm(x, &yb, &fit_full.coef).ok()
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
                let l = -2.0 * (f0.log_likelihood - fa.log_likelihood);
                (r, dd, l, fa.converged)
            } else {
                (f64::NAN, f64::NAN, f64::NAN, true)
            };
            // sequential: replicates already run in parallel
            let m = moment_stat(&f0, x, &yb, n_mc, seed, 1 + b as u64, n_mc > 0).ok()?;
            let s = if score_defined {
                f0.score_statistic(x, &yb, &fit_full.free).ok()?
            } else {
                f64::NAN
            };
            Some((r, dd, m, l, s, f0.converged && ok_a))
        })
        .collect();

    let mut br = vec![];
    let mut bd = vec![];
    let mut bm = vec![];
    let mut bl = vec![];
    let mut bs = vec![];
    let mut n_failed = 0;
    let mut n_unconverged = 0;
    for r in boot {
        match r {
            Some((a, b, c, e, g, ok)) => {
                n_unconverged += usize::from(!ok);
                br.push(a);
                bd.push(b);
                bm.push(c);
                bl.push(e);
                bs.push(g);
            }
            None => n_failed += 1,
        }
    }
    let nb = br.len().max(1) as f64;
    let pv_all = |obs: f64, v: &[f64]| v.iter().filter(|&&s| obs < s).count() as f64 / nb;
    // RoC, D and LR need the full refits, which a score test skips
    let pv = |obs: f64, v: &[f64]| {
        if full_refits {
            pv_all(obs, v)
        } else {
            f64::NAN
        }
    };
    // replicates whose score information was singular are left out
    let p_lm = {
        let ok: Vec<f64> = bs.iter().copied().filter(|v| v.is_finite()).collect();
        if lm.is_finite() && !ok.is_empty() {
            ok.iter().filter(|&&s| lm < s).count() as f64 / ok.len() as f64
        } else {
            f64::NAN
        }
    };
    let null = restriction_label(fit_full, &restriction);
    Ok(HypothesisTestResult {
        null,
        roc,
        d_stat,
        m_stat,
        lr,
        p_roc: pv(roc, &br),
        p_d: pv(d_stat, &bd),
        p_m: pv_all(m_stat, &bm),
        p_lr: pv(lr, &bl),
        lm,
        p_lm,
        p_lm_chisq: chisq_sf(lm, df as f64),
        p_lr_chisq: chisq_sf(lr.max(0.0), df as f64),
        df,
        loglik_restricted: fit0.log_likelihood,
        loglik_full: fit_full.log_likelihood,
        coef_restricted: fit0.coef,
        n_boot: br.len(),
        n_failed,
        n_unconverged,
        boot_roc: if full_refits { br } else { vec![] },
        boot_d: if full_refits { bd } else { vec![] },
        boot_m: bm,
        boot_lr: if full_refits { bl } else { vec![] },
        boot_lm: bs,
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

    /// Per-replicate costs of `run_test` on the hard scenario (both `μ` and
    /// `V` depend on `x`), one core.
    /// `RAYON_NUM_THREADS=1 cargo test --release --lib hard_replicates -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn hard_replicates() {
        use crate::regression::Shape;
        use crate::regression::fit_profile::{config, hard_truth, simulate};
        let reps = 12;
        let ds: Vec<usize> = std::env::var("D")
            .unwrap_or_else(|_| "3,5,8".into())
            .split(',')
            .map(|v| v.parse().unwrap())
            .collect();
        let shapes = match std::env::var("SHAPE").as_deref() {
            Ok("lt") => vec![Shape::LogTangent],
            Ok("gamma") => vec![Shape::Gamma],
            _ => vec![Shape::LogTangent, Shape::Gamma],
        };
        for d in ds {
            for &shape in &shapes {
                let (x, y) = simulate(&hard_truth(d), 2000, 7, shape);
                let t = Instant::now();
                let fit = config(d, shape).with_parallel(false).fit(&x, &y).unwrap();
                println!(
                    "d={d} {shape:?}: full fit {:.1} ms (converged {})",
                    t.elapsed().as_secs_f64() * 1e3,
                    fit.converged
                );
                for null in [
                    Restriction::Isotropy,
                    Restriction::ConstantShape,
                    Restriction::GammaOfFeature(0),
                    Restriction::MuOfFeature(1),
                    Restriction::Feature(4),
                ] {
                    let mask0 = restriction_mask(&fit, &null).unwrap();
                    let mut b0 = fit.coef.clone();
                    for (v, &f) in b0.iter_mut().zip(mask0.iter()) {
                        if !f {
                            *v = 0.0;
                        }
                    }
                    let mut cfg0 = refit_config(&fit);
                    cfg0.mask = Some(mask0.clone());
                    cfg0.precondition = None;
                    let t = Instant::now();
                    let fit0 = {
                        let mut once = cfg0.clone();
                        once.compute_hessian = true;
                        let warm = once.fit_warm(&x, &y, &b0).unwrap();
                        let mut auto = once.clone();
                        auto.init = crate::regression::Init::Auto;
                        auto.local_precondition = false;
                        match auto.fit(&x, &y) {
                            Ok(a) if a.log_likelihood > warm.log_likelihood => a,
                            _ => warm,
                        }
                    };
                    let t_fit0 = t.elapsed().as_secs_f64() * 1e3;
                    cfg0.precondition = fit0
                        .cov_params
                        .as_ref()
                        .and_then(crate::regression::precondition_from_cov);
                    let shape_zeroed =
                        (d..mask0.nrows()).all(|k| (0..mask0.ncols()).all(|j| !mask0[(k, j)]));
                    let gamma_cone = shape == Shape::Gamma && shape_zeroed;
                    let mut cfg_a = refit_config(&fit);
                    let t = Instant::now();
                    cfg_a.precondition = if gamma_cone {
                        None
                    } else {
                        cfg_a.precondition_at(&x, &y, &fit0.coef, &fit.mask)
                    };
                    let t_pre = t.elapsed().as_secs_f64() * 1e3;
                    let cfg_a2 = refit_config(&fit);
                    let (mut t0, mut ta1, mut ta2, mut tm, mut tl) = (0.0, 0.0, 0.0, 0.0, 0.0);
                    let (mut f0e, mut fa1e, mut fa2e) = (0, 0, 0);
                    let mut n2 = 0;
                    for r in 0..reps {
                        let mut rng = crate::stream_rng(1, 5, r as u64);
                        let yb = fit0.sample_responses(&x, &mut rng).unwrap();
                        let t = Instant::now();
                        let f0 = cfg0.fit_warm(&x, &yb, &fit0.coef).unwrap();
                        t0 += t.elapsed().as_secs_f64();
                        f0e += f0.n_fev;
                        let mut ba = f0.coef.clone();
                        for &(k, j) in &fit.free {
                            if !mask0[(k, j)] {
                                ba[(k, j)] = if k < d || shape == Shape::LogTangent {
                                    0.0
                                } else {
                                    1e-3
                                };
                            }
                        }
                        let t = Instant::now();
                        let fa1 = cfg_a.fit_warm(&x, &yb, &ba).unwrap();
                        ta1 += t.elapsed().as_secs_f64();
                        fa1e += fa1.n_fev;
                        if shape == Shape::Gamma || !fa1.converged {
                            let t = Instant::now();
                            let fa2 = cfg_a2.fit_warm(&x, &yb, &fit.coef).unwrap();
                            ta2 += t.elapsed().as_secs_f64();
                            fa2e += fa2.n_fev;
                            n2 += 1;
                        }
                        let t = Instant::now();
                        let _ = moment_stat(&f0, &x, &yb, 0, 1, 1, false).unwrap();
                        tm += t.elapsed().as_secs_f64();
                        let t = Instant::now();
                        let _ = f0.score_statistic(&x, &yb, &fit.free);
                        tl += t.elapsed().as_secs_f64();
                    }
                    let r = reps as f64;
                    println!(
                        "   {:<28} df {:>3} | fit0 {:6.1} ms, metric {:5.1} ms | per replicate: restricted {:5.1} ms ({:3.0} fev), full {:6.1} ms ({:3.0} fev){} , M {:4.1}, LM {:4.1} ms",
                        restriction_label(&fit, &null),
                        fit.free.iter().filter(|&&(k, j)| !mask0[(k, j)]).count(),
                        t_fit0,
                        t_pre,
                        t0 / r * 1e3,
                        f0e as f64 / r,
                        ta1 / r * 1e3,
                        fa1e as f64 / r,
                        if n2 > 0 {
                            format!(
                                " + 2nd {:6.1} ms ({:3.0} fev, {n2}/{reps})",
                                ta2 / n2 as f64 * 1e3,
                                fa2e as f64 / n2 as f64
                            )
                        } else {
                            String::new()
                        },
                        tm / r * 1e3,
                        tl / r * 1e3
                    );
                }
            }
        }
    }

    /// Cost of the score statistic against the rest of a replicate, n = 2000,
    /// 5 covariates. `cargo test --release --lib score_cost -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn score_cost() {
        use crate::regression::Shape;
        for d in [3, 5, 8] {
            let (x, y) = super::gamma_refits::sim(d, 5, 2000, 11);
            for shape in [Shape::Gamma, Shape::LogTangent] {
                let fit = EsagRegressor::new().with_shape(shape).fit(&x, &y).unwrap();
                let mask0 = restriction_mask(&fit, &Restriction::GammaOfFeature(0)).unwrap();
                let mut b0 = fit.coef.clone();
                for (v, &f) in b0.iter_mut().zip(mask0.iter()) {
                    if !f {
                        *v = 0.0;
                    }
                }
                let mut cfg0 = refit_config(&fit);
                cfg0.mask = Some(mask0);
                let fit0 = cfg0.fit_warm(&x, &y, &b0).unwrap();
                let reps = 10;
                let (mut t_lm, mut t_m, mut t_0) = (0.0, 0.0, 0.0);
                for r in 0..reps {
                    let mut rng = crate::stream_rng(1, 5, r);
                    let yb = fit0.sample_responses(&x, &mut rng).unwrap();
                    let t = Instant::now();
                    let f0 = cfg0.fit_warm(&x, &yb, &fit0.coef).unwrap();
                    t_0 += t.elapsed().as_secs_f64();
                    let t = Instant::now();
                    let _ = f0.score_statistic(&x, &yb, &fit.free).unwrap();
                    t_lm += t.elapsed().as_secs_f64();
                    let t = Instant::now();
                    let _ = moment_stat(&f0, &x, &yb, 0, 1, 1, false).unwrap();
                    t_m += t.elapsed().as_secs_f64();
                }
                let r = reps as f64;
                println!(
                    "d={d} {shape:?} ({} coefficients): LM {:.2} ms, M {:.2} ms, restricted refit {:.2} ms",
                    fit.free.len(),
                    t_lm / r * 1e3,
                    t_m / r * 1e3,
                    t_0 / r * 1e3
                );
            }
        }
    }
}

#[cfg(test)]
mod gamma_refits {
    //! Where the γ likelihood has cliffs (d = 5, 5 covariates), how refit
    //! metrics compare on the full refits of the `x5` test (MATH.md §6).
    use super::*;
    use crate::distribution::Esag;
    use crate::regression::{
        EsagRegressor, free_params, joint_loglik_and_grad, joint_loglik_grad_hess,
    };
    use nalgebra::DVector;
    use rand::SeedableRng;
    use rand_distr::{Distribution, StandardNormal};
    use std::time::Instant;

    /// `examples/big_fit.rs::sim` (gamma truth).
    pub(super) fn sim(d: usize, p: usize, n: usize, seed: u64) -> (DMatrix<f64>, DMatrix<f64>) {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let bd = crate::dims::eta_len(d);
        let b = DMatrix::from_fn(bd, p + 1, |k, j| {
            let s = (k * 7 + j * 3) as f64;
            match (j, k < d) {
                (0, true) => 1.6 * (1.0 + 0.8 * (s * 0.9).sin()),
                (0, false) => 0.8 * (s * 1.3).sin(),
                (_, true) => 0.35 * (s * 0.7).cos(),
                (_, false) => 0.2 * (s * 1.1).sin(),
            }
        });
        let x = DMatrix::from_fn(n, p, |_, _| StandardNormal.sample(&mut rng));
        let mut y = DMatrix::zeros(n, d);
        let mut buf = vec![0.0; d];
        for i in 0..n {
            let eta: Vec<f64> = (0..bd)
                .map(|k| b[(k, 0)] + (0..p).map(|j| b[(k, j + 1)] * x[(i, j)]).sum::<f64>())
                .collect();
            let dist = Esag::from_eta(&eta).unwrap();
            dist.sample_one(&dist.frame(), &mut rng, &mut buf);
            for k in 0..d {
                y[(i, k)] = buf[k];
            }
        }
        (x, y)
    }

    /// Damped Newton with the exact Hessian at every iterate; returns the
    /// log-likelihood and the number of Hessians.
    fn newton(
        cfg: &EsagRegressor,
        x: &DMatrix<f64>,
        y: &DMatrix<f64>,
        b0: &DMatrix<f64>,
        mask: &DMatrix<bool>,
    ) -> (f64, usize) {
        let data = cfg.data(x, y).unwrap();
        let free = free_params(mask);
        let (bd, full, nf) = (data.b_dim, data.b_dim * data.p_total, free.len());
        let mut coef = b0.clone();
        let mut f_cur = f64::NEG_INFINITY;
        for nh in 1..=40 {
            let (f, g, h) = joint_loglik_grad_hess(&data, &coef, false);
            f_cur = f;
            let gv = DVector::from_iterator(nf, free.iter().map(|&(k, j)| g[j * bd + k]));
            let info = DMatrix::from_fn(nf, nf, |a, c| {
                let ((ka, ja), (kc, jc)) = (free[a], free[c]);
                -h[(ja * bd + ka) * full + jc * bd + kc]
            });
            let scale = info.diagonal().amax().max(1.0);
            let mut accepted = false;
            let mut lam = 0.0;
            'damp: for _ in 0..12 {
                let mut m = info.clone();
                for a in 0..nf {
                    m[(a, a)] += lam * scale;
                }
                if let Some(ch) = m.cholesky() {
                    let step = ch.solve(&gv);
                    let dec = gv.dot(&step);
                    if 0.5 * dec < 1e-6 {
                        return (f, nh);
                    }
                    let mut t = 1.0;
                    for _ in 0..8 {
                        let mut cand = coef.clone();
                        for (a, &(k, j)) in free.iter().enumerate() {
                            cand[(k, j)] += t * step[a];
                        }
                        let (f2, _) = joint_loglik_and_grad(&data, &cand, false);
                        if f2.is_finite() && f2 >= f + 1e-4 * t * dec {
                            coef = cand;
                            accepted = true;
                            break 'damp;
                        }
                        t *= 0.5;
                    }
                }
                lam = if lam == 0.0 { 1e-6 } else { lam * 10.0 };
            }
            if !accepted {
                return (f_cur, nh);
            }
        }
        (f_cur, 40)
    }

    /// Do bootstrap restricted refits (one warm start, at the bootstrap
    /// truth) reach the restricted maximum? Against a second fit from
    /// scratch (`Init::Auto`), on data from the anisotropic truth.
    /// `cargo test --release --lib restricted_refit_maxima -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn restricted_refit_maxima() {
        use rayon::prelude::*;
        let (d, p, n) = (5, 5, 2000);
        let (x, y) = sim(d, p, n, 11);
        let fit = EsagRegressor::new().fit(&x, &y).unwrap();
        for r in [
            Restriction::MuOfFeature(0),
            Restriction::GammaOfFeature(0),
            Restriction::Feature(4),
        ] {
            let mask0 = restriction_mask(&fit, &r).unwrap();
            let mut b0 = fit.coef.clone();
            for (v, &f) in b0.iter_mut().zip(mask0.iter()) {
                if !f {
                    *v = 0.0;
                }
            }
            let mut cfg0 = refit_config(&fit);
            cfg0.mask = Some(mask0.clone());
            cfg0.precondition = None;
            let mut once = cfg0.clone();
            once.compute_hessian = true;
            let fit0 = once.fit_warm(&x, &y, &b0).unwrap();
            cfg0.precondition = fit0
                .cov_params
                .as_ref()
                .and_then(crate::regression::precondition_from_cov);
            let mut auto = cfg0.clone();
            auto.init = crate::regression::Init::Auto;
            auto.local_precondition = false;
            let gaps: Vec<f64> = (0..40u64)
                .into_par_iter()
                .map(|b| {
                    let mut rng = crate::stream_rng(7, crate::streams::TEST_BOOT, b);
                    let yb = fit0.sample_responses(&x, &mut rng).unwrap();
                    let warm = cfg0.fit_warm(&x, &yb, &fit0.coef).unwrap();
                    let fresh = auto.fit(&x, &yb).unwrap();
                    fresh.log_likelihood - warm.log_likelihood
                })
                .collect();
            let worse = gaps.iter().filter(|&&g| g > 0.01).count();
            let better = gaps.iter().filter(|&&g| g < -0.01).count();
            let max = gaps.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            eprintln!(
                "{r:?}: warm restricted refit below the fresh fit in {worse}/40 (max gap {max:.3}), above it in {better}/40"
            );
        }
    }

    /// Second starts for the bootstrap restricted refits: how often each
    /// procedure ends below the best restricted maximum found by any start
    /// (warm at the bootstrap truth, the observed full fit zeroed, the
    /// bootstrap full refit zeroed, and a fit from scratch), and its cost.
    /// `cargo test --release --lib restricted_second_start -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn restricted_second_start() {
        use rayon::prelude::*;
        let (d, p, n) = (5, 5, 2000);
        let (x, y) = sim(d, p, n, 11);
        let fit = EsagRegressor::new().fit(&x, &y).unwrap();
        for r in [
            Restriction::MuOfFeature(0),
            Restriction::GammaOfFeature(0),
            Restriction::Feature(4),
            Restriction::ConstantShape,
        ] {
            let mask0 = restriction_mask(&fit, &r).unwrap();
            let zero = |b: &DMatrix<f64>| {
                let mut b = b.clone();
                for (v, &f) in b.iter_mut().zip(mask0.iter()) {
                    if !f {
                        *v = 0.0;
                    }
                }
                b
            };
            let b0 = zero(&fit.coef);
            let mut cfg0 = refit_config(&fit);
            cfg0.mask = Some(mask0.clone());
            cfg0.precondition = None;
            let mut once = cfg0.clone();
            once.compute_hessian = true;
            let warm0 = once.fit_warm(&x, &y, &b0).unwrap();
            let mut auto = once.clone();
            auto.init = crate::regression::Init::Auto;
            auto.local_precondition = false;
            let fit0 = match auto.fit(&x, &y) {
                Ok(a) if a.log_likelihood > warm0.log_likelihood => a,
                _ => warm0,
            };
            cfg0.precondition = fit0
                .cov_params
                .as_ref()
                .and_then(crate::regression::precondition_from_cov);
            let cfg_a = refit_config(&fit);
            let mut fresh_cfg = cfg0.clone();
            fresh_cfg.init = crate::regression::Init::Auto;
            fresh_cfg.local_precondition = false;
            let reps = 60u64;
            // per replicate: lls of (warm, from b0, from zeroed full refit,
            // fresh) and evaluations of the first three
            let rows: Vec<([f64; 4], [usize; 4])> = (0..reps)
                .into_par_iter()
                .map(|b| {
                    let mut rng = crate::stream_rng(7, crate::streams::TEST_BOOT, b);
                    let yb = fit0.sample_responses(&x, &mut rng).unwrap();
                    let warm = cfg0.fit_warm(&x, &yb, &fit0.coef).unwrap();
                    let from_b0 = cfg0.fit_warm(&x, &yb, &b0).unwrap();
                    let fa = cfg_a.fit_warm(&x, &yb, &fit.coef).unwrap();
                    let from_fa = cfg0.fit_warm(&x, &yb, &zero(&fa.coef)).unwrap();
                    let fresh = fresh_cfg.fit(&x, &yb).unwrap();
                    (
                        [
                            warm.log_likelihood,
                            from_b0.log_likelihood,
                            from_fa.log_likelihood,
                            fresh.log_likelihood,
                        ],
                        [warm.n_fev, from_b0.n_fev, from_fa.n_fev, fresh.n_fev],
                    )
                })
                .collect();
            let report = |name: &str, pick: &dyn Fn(&[f64; 4]) -> f64| {
                let gaps: Vec<f64> = rows
                    .iter()
                    .map(|(l, _)| l.iter().copied().fold(f64::NEG_INFINITY, f64::max) - pick(l))
                    .collect();
                let below = gaps.iter().filter(|&&g| g > 0.01).count();
                let mean = gaps.iter().sum::<f64>() / gaps.len() as f64;
                let max = gaps.iter().copied().fold(0.0, f64::max);
                eprintln!(
                    "  {name:<28} below the best in {below:>2}/{reps} (mean gap {mean:.3}, max {max:.3})"
                );
            };
            let fev = |k: usize| rows.iter().map(|(_, e)| e[k]).sum::<usize>() as f64 / reps as f64;
            eprintln!(
                "null {r:?}: evaluations warm {:.0}, from b0 {:.0}, from the zeroed full refit {:.0}, fresh {:.0}",
                fev(0),
                fev(1),
                fev(2),
                fev(3)
            );
            report("warm only", &|l| l[0]);
            report("warm + observed fit zeroed", &|l| l[0].max(l[1]));
            report("warm + full refit zeroed", &|l| l[0].max(l[2]));
            report("all three", &|l| l[0].max(l[1]).max(l[2]));
            report("warm + fresh", &|l| l[0].max(l[3]));
            report("full refit zeroed + fresh", &|l| l[2].max(l[3]));
        }
    }

    /// `cargo test --release --lib gamma_refits -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn refit_metrics() {
        let (d, p, n) = (5, 5, 2000);
        let (x, y) = sim(d, p, n, 11);
        let fit = EsagRegressor::new().fit(&x, &y).unwrap();
        let mask0 = restriction_mask(&fit, &Restriction::Feature(4)).unwrap();
        let mut b0 = fit.coef.clone();
        for (v, &f) in b0.iter_mut().zip(mask0.iter()) {
            if !f {
                *v = 0.0;
            }
        }
        let mut cfg0 = refit_config(&fit);
        cfg0.mask = Some(mask0.clone());
        cfg0.precondition = None;
        let mut once = cfg0.clone();
        once.compute_hessian = true;
        let fit0 = once.fit_warm(&x, &y, &b0).unwrap();
        cfg0.precondition = fit0
            .cov_params
            .as_ref()
            .and_then(crate::regression::precondition_from_cov);
        // the full model's information at the restricted fit (not positive
        // definite here: `precondition_at` returns None)
        let mut plain = refit_config(&fit);
        plain.precondition = plain.precondition_at(&x, &y, &fit0.coef, &fit.mask);
        plain.local_precondition = false;
        let mut empirical = plain.clone();
        empirical.local_precondition = true;
        let reps = 30;
        let mut rows: Vec<(&str, f64, usize, Vec<f64>)> =
            ["no metric", "empirical", "local Hessian", "Newton"]
                .iter()
                .map(|&s| (s, 0.0, 0, vec![]))
                .collect();
        for r in 0..reps {
            let mut rng = crate::stream_rng(7, crate::streams::TEST_BOOT, r);
            let yb = fit0.sample_responses(&x, &mut rng).unwrap();
            let f0 = cfg0.fit_warm(&x, &yb, &fit0.coef).unwrap();
            let mut ba = f0.coef.clone();
            for &(k, j) in &fit.free {
                if !mask0[(k, j)] {
                    ba[(k, j)] = if k < d { 0.0 } else { 1e-3 };
                }
            }
            let mut ll = [0.0; 4];
            for (v, row) in rows.iter_mut().enumerate() {
                let t = Instant::now();
                let (l, cost) = match v {
                    0 => {
                        let a = plain.fit_warm(&x, &yb, &ba).unwrap();
                        (a.log_likelihood, a.n_fev)
                    }
                    1 => {
                        let a = empirical.fit_warm(&x, &yb, &ba).unwrap();
                        (a.log_likelihood, a.n_fev)
                    }
                    2 => {
                        let mut c = plain.clone();
                        c.precondition = c.precondition_at(&x, &yb, &ba, &fit.mask);
                        let a = c.fit_warm(&x, &yb, &ba).unwrap();
                        (a.log_likelihood, a.n_fev)
                    }
                    _ => newton(&plain, &x, &yb, &ba, &fit.mask),
                };
                row.1 += t.elapsed().as_secs_f64();
                row.2 += cost;
                ll[v] = l;
            }
            for (v, row) in rows.iter_mut().enumerate() {
                row.3.push(ll[v] - ll[0]);
            }
        }
        println!("{reps} full refits of the x5 test, d = 5, n = 2000 (one thread)");
        for (name, t, cost, dl) in &rows {
            println!(
                "{name:<14} {:6.1} ms  {:5.1} {}  ll − no metric: higher {} lower {} (range {:+.3} … {:+.3})",
                t / reps as f64 * 1e3,
                *cost as f64 / reps as f64,
                if *name == "Newton" {
                    "Hessians"
                } else {
                    "evaluations"
                },
                dl.iter().filter(|&&v| v > 1e-3).count(),
                dl.iter().filter(|&&v| v < -1e-3).count(),
                dl.iter().cloned().fold(f64::INFINITY, f64::min),
                dl.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
            );
        }
    }
}
