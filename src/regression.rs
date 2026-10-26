//! ESAG regression with a scikit-learn-style API.
//!
//! Model: for observation `i` with covariate vector `x_i ∈ R^p`, the response
//! `y_i ∈ S^{d-1}` follows `ESAG(mu_i, gamma_i)` where the stacked natural
//! parameter `eta_i = (mu_i, gamma_i) = B x̃_i` is linear in
//! `x̃_i = (1, x_i)` (intercept) — exactly the model fitted by `MLE()` in the
//! original R code. `B` is a `(d + (d-2)(d+1)/2) × (p+1)` coefficient matrix.
//!
//! ```no_run
//! use esag::regression::EsagRegressor;
//! # let (x, y) = (nalgebra::DMatrix::<f64>::zeros(1, 1), nalgebra::DMatrix::<f64>::zeros(1, 3));
//! let model = EsagRegressor::new().with_max_iter(500).fit(&x, &y).unwrap();
//! let mean_directions = model.predict(&x);        // n × d unit vectors
//! let dists = model.predict_distribution(&x);     // Vec<Esag>
//! println!("{}", model.summary());
//! ```
//!
//! Fitting maximises the joint log-likelihood with L-BFGS using *exact*
//! gradients (a hand-written reverse-mode gradient of the per-observation
//! log-density, then the chain rule `∂LL/∂B = Σ_i g_i x̃_iᵀ`), parallelised over
//! observations with rayon and bit-for-bit reproducible. Several starts are
//! raced in lockstep (see `docs/MATH.md`, "Starts"). Standard errors come from
//! the exact observed information.

use crate::dims;
use crate::distribution::{Esag, residual_q};
use crate::error::{EsagError, Result};
use crate::frame::Frame;
use crate::loglik;
use crate::logtangent::LtFrame;
use crate::optim::lbfgs::{Lbfgs, LbfgsConfig, Objective, OptimResult, Status, minimize};
use crate::stats::normal::norm_sf;
use nalgebra::{DMatrix, DVector};
use rand::Rng;
use rand_distr::{Distribution, StandardNormal};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

fn default_true() -> bool {
    true
}

/// How the shape rows of `eta` parameterize the covariance `V`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum Shape {
    /// Yu & Huang's `gamma` (the papers' model).
    #[default]
    Gamma,
    /// The smooth log-tangent coordinates `s` of [`crate::logtangent`]:
    /// `V = μ̂μ̂ᵀ + Ξ(μ̂) exp(S) Ξ(μ̂)ᵀ`, `S` traceless symmetric, with the
    /// frame `Ξ` transported from a reference direction (the data's mean
    /// direction unless [`EsagRegressor::lt_reference`] is set). Same
    /// distributions and parameter count as `gamma`, but `V` depends smoothly
    /// on `s`, so the likelihood has no cliffs (see `docs/MATH.md`).
    LogTangent,
    /// Yu & Huang's `gamma` map, written in the tangent frame transported
    /// from a reference direction (as [`Shape::LogTangent`]'s,
    /// [`crate::gamma_transported`]) instead of `onb(mu)`. The frames
    /// coincide at the reference (the data's mean direction unless
    /// [`EsagRegressor::lt_reference`] is set), so a fit without covariates
    /// is the `gamma` fit; with covariates the model differs through how
    /// the frame turns with `mu(x)`. It has no pole at `mu_0 = mu_1 = 0`:
    /// with the data's mean direction near that set, fits need about a
    /// fifth of the iterations (`docs/MATH.md` §2).
    GammaTransported,
}

/// How the optimiser is initialised.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Init {
    /// Default: run from both [`Init::DataDriven`] and [`Init::RStyle`] and keep
    /// the higher likelihood; if neither passes the convergence check, retry
    /// from randomly perturbed starts.
    Auto,
    /// Moment-based intercept: `mu` along the sample mean direction with
    /// `‖mu‖` and `gamma` from the tangent-plane scatter of the responses
    /// (near the mode `S ≈ V/‖μ‖²`, inverted with
    /// [`crate::param::gamma_from_covariance`]); zero slopes.
    DataDriven,
    /// The R code's recommendation: `mu` entries 1, `gamma` entries (near) 0,
    /// for every column. (We use 1e-3 instead of exactly 0 for `gamma`.)
    RStyle,
    /// A user-supplied full coefficient matrix `B` (`eta_len(d) × p_total`).
    Custom(DMatrix<f64>),
}

/// Hyper-parameters of the ESAG regression estimator (scikit-learn style:
/// configure, then call [`EsagRegressor::fit`]).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EsagRegressor {
    /// Add an intercept column of ones to `X` (default `true`).
    pub fit_intercept: bool,
    /// Maximum L-BFGS iterations (default 2000).
    pub max_iter: usize,
    /// Gradient tolerance (infinity norm) for convergence (default 1e-6).
    pub tol: f64,
    /// Initialisation strategy.
    pub init: Init,
    /// Number of additional randomly perturbed starts (default 0); the best
    /// optimum is kept.
    pub n_restarts: usize,
    /// Seed for the random restarts.
    pub seed: u64,
    /// Compute the observed-information Hessian (standard errors) (default true).
    pub compute_hessian: bool,
    /// L-BFGS memory. `0` (default) = automatic: the number of free
    /// parameters, clamped to `[10, 200]`. Memory below the parameter count
    /// slows convergence badly on ESAG likelihoods (2–5× more iterations).
    pub lbfgs_memory: usize,
    /// Evaluate the likelihood in parallel over observations (default
    /// `true`). Bootstrap procedures switch it off for their refits because
    /// they already run one refit per thread.
    #[serde(default = "default_true")]
    pub parallel: bool,
    /// Print optimisation progress to stderr.
    pub verbose: bool,
    /// Optional free-parameter mask (`eta_len(d) × p_total`); `false` entries
    /// are held at their initial value (0 unless `Init::Custom`). Used for
    /// restricted fits in hypothesis tests.
    pub mask: Option<DMatrix<bool>>,
    /// Optional feature names for `summary()`.
    pub feature_names: Option<Vec<String>>,
    /// Shape parameterization (default [`Shape::Gamma`]).
    #[serde(default)]
    pub shape: Shape,
    /// Optional response (column) names for `summary()`.
    #[serde(default)]
    pub response_names: Option<Vec<String>>,
    /// Reference direction of the log-tangent frame (`Shape::LogTangent`);
    /// `None` = the sample mean direction of `Y`. A fitted model records the
    /// direction it used, so refits keep the same frame.
    #[serde(default)]
    pub lt_reference: Option<Vec<f64>>,
    /// Preconditioner for warm-started refits: a lower-triangular factor `L`
    /// of an approximate inverse Hessian over the free parameters (optimiser
    /// order). L-BFGS then works in `z` with `θ = θ₀ + L z`, where the
    /// Hessian is close to the identity. The resampling procedures set it
    /// from the original fit's observed information.
    #[serde(skip)]
    pub(crate) precondition: Option<Arc<DMatrix<f64>>>,
    /// Warm-started refit: where `precondition` is a poor metric (gamma
    /// with cliffs) or missing, precondition with the empirical information
    /// at the start instead.
    #[serde(skip)]
    pub(crate) local_precondition: bool,
    /// Stopping rule of whitened refits (predicted remaining gain); `None`
    /// = [`REFIT_DECREMENT`].
    #[serde(skip)]
    pub(crate) refit_decrement: Option<f64>,
    /// Mean gradient per observation at which L-BFGS hands over when
    /// standard errors are requested; `None` = [`NEWTON_SWITCH`].
    #[serde(skip)]
    pub(crate) handover: Option<f64>,
    /// Opt-in (default `false`): a gamma fit that stays trapped returns the
    /// converged fit of the observations not on gamma's singular set, with
    /// the others listed in [`EsagRegressorFit::set_aside_obs`]. That is the
    /// maximum-likelihood fit of the remaining `n − k` observations, not of
    /// the data, and the summary says so.
    #[serde(default)]
    pub set_aside: bool,
}

impl Default for EsagRegressor {
    fn default() -> Self {
        Self::new()
    }
}

impl EsagRegressor {
    /// Default configuration.
    pub fn new() -> Self {
        Self {
            fit_intercept: true,
            max_iter: 2000,
            tol: 1e-6,
            init: Init::Auto,
            n_restarts: 0,
            seed: 0,
            compute_hessian: true,
            lbfgs_memory: 0,
            parallel: true,
            verbose: false,
            mask: None,
            feature_names: None,
            shape: Shape::Gamma,
            response_names: None,
            lt_reference: None,
            precondition: None,
            local_precondition: false,
            refit_decrement: None,
            handover: None,
            set_aside: false,
        }
    }

    /// Whether a trapped gamma fit may set observations on its singular set
    /// aside ([`EsagRegressor::set_aside`]).
    pub fn with_set_aside(mut self, v: bool) -> Self {
        self.set_aside = v;
        self
    }
    /// Whether to add an intercept column.
    pub fn with_fit_intercept(mut self, v: bool) -> Self {
        self.fit_intercept = v;
        self
    }
    /// Maximum number of iterations.
    pub fn with_max_iter(mut self, v: usize) -> Self {
        self.max_iter = v;
        self
    }
    /// Gradient tolerance.
    pub fn with_tol(mut self, v: f64) -> Self {
        self.tol = v;
        self
    }
    /// Initialisation strategy.
    pub fn with_init(mut self, v: Init) -> Self {
        self.init = v;
        self
    }
    /// Random restarts.
    pub fn with_restarts(mut self, n: usize, seed: u64) -> Self {
        self.n_restarts = n;
        self.seed = seed;
        self
    }
    /// Whether to compute the Hessian / standard errors.
    pub fn with_hessian(mut self, v: bool) -> Self {
        self.compute_hessian = v;
        self
    }
    /// Verbose optimisation output.
    pub fn with_verbose(mut self, v: bool) -> Self {
        self.verbose = v;
        self
    }
    /// Parallel likelihood evaluation over observations.
    pub fn with_parallel(mut self, v: bool) -> Self {
        self.parallel = v;
        self
    }
    /// Free-parameter mask for restricted fits.
    pub fn with_mask(mut self, mask: Option<DMatrix<bool>>) -> Self {
        self.mask = mask;
        self
    }
    /// Feature names (length = number of columns of `X`, excluding intercept).
    pub fn with_response_names(mut self, names: Vec<String>) -> Self {
        self.response_names = Some(names);
        self
    }

    pub fn with_shape(mut self, shape: Shape) -> Self {
        self.shape = shape;
        self
    }

    pub fn with_lt_reference(mut self, m: Option<Vec<f64>>) -> Self {
        self.lt_reference = m;
        self
    }

    /// Validated data with the kernel for this configuration's shape.
    pub(crate) fn data(&self, x: &DMatrix<f64>, y: &DMatrix<f64>) -> Result<RegressionData> {
        let mut data = RegressionData::new(x, y, self.fit_intercept)?;
        if self.shape != Shape::Gamma {
            let m = match &self.lt_reference {
                Some(m) => {
                    if m.len() != data.d {
                        return Err(EsagError::Shape(format!(
                            "lt_reference has length {} but d = {}",
                            m.len(),
                            data.d
                        )));
                    }
                    m.clone()
                }
                None => mean_direction(&data.y, data.n, data.d),
            };
            if self.shape == Shape::LogTangent {
                data.lt = Some(LtFrame::new(&m));
            } else {
                data.gt = Some(LtFrame::new(&m));
            }
        }
        Ok(data)
    }

    pub fn with_feature_names(mut self, names: Vec<String>) -> Self {
        self.feature_names = Some(names);
        self
    }

    /// Fit the model. `x` is `n × p` (no intercept column; may have `p = 0`),
    /// `y` is `n × d` with one unit vector per row (rows are re-normalised).
    ///
    /// The result is deterministic: it depends only on the data and the
    /// configuration (including `seed`), not on thread timing or the number
    /// of threads.
    pub fn fit(&self, x: &DMatrix<f64>, y: &DMatrix<f64>) -> Result<EsagRegressorFit> {
        let data = self.data(x, y)?;
        let mask = self.resolve_mask(&data)?;
        let inits: Vec<Init> = match &self.init {
            Init::Auto => vec![Init::DataDriven, Init::RStyle],
            other => vec![other.clone()],
        };
        let mut starts = Vec::with_capacity(inits.len() + self.n_restarts);
        for init in &inits {
            let mut b0 = self.clone().with_init(init.clone()).initial_coef(&data)?;
            if !matches!(init, Init::Custom(_)) {
                zero_fixed(&mut b0, &mask);
            }
            starts.push(b0);
        }
        // Gamma with covariates: race a start from the log-tangent fit too
        // (`lt_informed_start`). The moment start reads the shape from the
        // pooled responses and ignores the covariates; at d = 10 its race
        // ends 330–520 below the best maximum known in 3 of 12 datasets
        // (one of them converged there) and trapped in 10.
        let lt_start = (data.lt.is_none()
            && data.gt.is_none()
            && data.p > 0
            && matches!(self.init, Init::Auto))
        .then(|| self.lt_informed_start(x, y, &data, &mask))
        .flatten();
        if let Some(s) = &lt_start {
            starts.push(s.clone());
        }
        if self.n_restarts > 0 {
            let centers = starts.clone();
            starts.extend(perturbed_starts(
                &data,
                &centers,
                &mask,
                self.n_restarts,
                self.seed,
            ));
        }
        // The log-tangent likelihood is smooth, and the moment start reaches
        // the maximum the R-style start does (192 of 192 simulated datasets,
        // `docs/MATH.md`, "Starts"): race the second start only if the first
        // fails the convergence check. For gamma the starts end at different
        // maxima in up to 10 of 16 datasets, so they are always raced.
        let single = data.lt.is_some()
            && matches!(self.init, Init::Auto)
            && self.n_restarts == 0
            && starts.len() > 1;
        let mut best = if single {
            // whitened with the exact information at the moment start, L-BFGS
            // needs 9–10 evaluations instead of 23–34 (an exact Hessian costs
            // 2–4 gradients)
            let free = free_params(&mask);
            let mut first_cfg = self.clone();
            first_cfg.local_precondition = false;
            first_cfg.precondition = info_precondition(&data, &starts[0], &free, self.parallel);
            let first = if first_cfg.precondition.is_some() {
                // converging fits need ~10 iterations; one still running
                // after DIVERGENCE_CHECK whose V(x_i) degenerates will only
                // diverge further (an unbounded likelihood)
                let mut best = first_cfg.optimize(
                    &data,
                    &starts[0],
                    &free,
                    self.max_iter.min(DIVERGENCE_CHECK),
                );
                let (mut it, mut fev) = (best.n_iter, best.n_fev);
                while !best.converged
                    && it < self.max_iter
                    && best.status == Status::MaxIterations
                    && degenerate_observations(&data, &best.coef).is_empty()
                {
                    let next = first_cfg.optimize(
                        &data,
                        &best.coef,
                        &free,
                        (self.max_iter - it).min(DIVERGENCE_CHECK),
                    );
                    it += next.n_iter;
                    fev += next.n_fev;
                    if next.beats(&best) {
                        best = next;
                    } else {
                        break;
                    }
                }
                best.n_iter = it;
                best.n_fev = fev;
                best
            } else {
                self.race(&data, &starts[..1], &mask, self.max_iter)
            };
            if first.converged || !degenerate_observations(&data, &first.coef).is_empty() {
                first
            } else {
                let second = self.race(&data, &starts[1..], &mask, self.max_iter);
                if second.beats(&first) { second } else { first }
            }
        } else {
            self.race(&data, &starts, &mask, self.max_iter)
        };
        // Safety net: if no start passed the convergence check (typically an
        // observation pinned at a singular point of the gamma
        // parameterization, see `docs/MATH.md`), race waves of perturbed
        // starts around the default starts and the best point so far, until
        // one converges.
        // Where the winner sits with observations on gamma's singular set,
        // the fit is in a trap, a non-regular local supremum (`docs/MATH.md`
        // §2). Perturbed starts fall into other traps (they rescued 0 of 51
        // such fits at d ≥ 5), even starts at the true coefficients do; the
        // weight homotopy below handles trapped fits instead.
        let trapped = |b: &DMatrix<f64>| {
            data.lt.is_none() && data.p > 0 && !near_singular_observations(&data, b).is_empty()
        };
        let pinned = !best.converged && trapped(&best.coef);
        // Nor where the likelihood is unbounded (a fitted V(x_i) heading for
        // a singular matrix): more starts only diverge further.
        let unbounded = !best.converged && !degenerate_observations(&data, &best.coef).is_empty();
        if !best.converged
            && !pinned
            && !unbounded
            && matches!(self.init, Init::Auto)
            && self.n_restarts == 0
        {
            let mut centers = starts.clone();
            centers.push(best.coef.clone());
            let extra = perturbed_starts(
                &data,
                &centers,
                &mask,
                AUTO_RESTARTS,
                self.seed.wrapping_add(0x5EED),
            );
            for wave in extra.chunks(RESTART_WAVE) {
                let alt = self.race(&data, wave, &mask, self.max_iter.min(RESTART_MAX_ITER));
                if alt.beats(&best) {
                    best = alt;
                }
                if best.converged {
                    break;
                }
            }
        }
        // still trapped: follow the regular root instead of searching for it
        if !best.converged && matches!(self.init, Init::Auto) && trapped(&best.coef) {
            let alt = lt_start.as_ref();
            if !collapsed_groups(&data, &best.coef).is_empty() {
                best = self.un_collapse(&data, best, &free_params(&mask), alt);
            }
            let outcome = if best.converged {
                Homotopy::Failed
            } else {
                self.homotopy_rounds(&data, &best, &mask, alt)
            };
            match outcome {
                Homotopy::Regular(r) => best = r,
                Homotopy::SetAside(rest, sset) if self.set_aside => {
                    // the MLE of the other n − k observations, reported as such
                    let mut wd = data.clone();
                    let mut w = vec![1.0; data.n];
                    for &i in &sset {
                        w[i] = 0.0;
                    }
                    wd.w = Some(w);
                    let mut fit = self.finalize(&wd, rest, mask)?;
                    fit.n_obs = data.n - sset.len();
                    fit.set_aside_obs = sset;
                    return Ok(fit);
                }
                _ => {
                    // No regular maximum continues from the fit without the
                    // trapped observations: the likelihood's maximum sits on
                    // the singular set. Converge to it there.
                    if let Some((at, held)) = self.stratum_maximum(&data, &best, &mask)
                        && at.neg_ll <= best.neg_ll + STRATUM_SLACK
                    {
                        if held.is_empty() {
                            // every trapped observation released: regular
                            return self.finalize(&data, at, mask);
                        }
                        let mut wd = data.clone();
                        let mut w = vec![1.0; data.n];
                        for &i in &held {
                            w[i] = 0.0;
                        }
                        wd.w = Some(w);
                        let (ll, gn) = (-at.neg_ll, at.grad_norm_inf);
                        let mut fit = self.finalize_with(&wd, at, mask, false)?;
                        fit.log_likelihood = ll;
                        fit.grad_norm = gn;
                        fit.singular_obs = held;
                        fit.non_regular = true;
                        return Ok(fit);
                    }
                }
            }
        }
        self.finalize(&data, best, mask)
    }

    /// A gamma start from the log-tangent fit of the same data, whose
    /// likelihood has no traps: its mean coefficients (the mean model is the
    /// same), and for the shape the least-squares regression on the design
    /// of the gamma of every observation's fitted `V̂(x_i)`
    /// ([`crate::param::gamma_from_covariance`]). Each observation's gamma is
    /// first moved by the exact symmetries of the parameterization
    /// ([`crate::param::gamma_align`]) to the image nearest the gamma at the
    /// mean design row, then, for a second regression, to the image nearest
    /// its fitted value: `γ ↦ V` is `2^{d−2}`-to-one, and a regression of
    /// unaligned images averages unrelated coefficients. Shape rows are
    /// regressed on their free columns only. `None` if the log-tangent fit
    /// fails.
    ///
    /// Against the gamma of `V̂(0)` with zero shape slopes, its race reaches
    /// the best maximum known in 12 of 12 datasets at d = 10 (n = 2000,
    /// standard truth), against 3 of 12: the slopes matter, and so does
    /// averaging the shape over the data instead of reading it at `x = 0`
    /// (`gamma_intercept_estimators` in `fit_profile.rs`).
    fn lt_informed_start(
        &self,
        x: &DMatrix<f64>,
        y: &DMatrix<f64>,
        data: &RegressionData,
        mask: &DMatrix<bool>,
    ) -> Option<DMatrix<f64>> {
        let (n, d, bd, pt) = (data.n, data.d, data.b_dim, data.p_total);
        let g = bd - d;
        let mut cfg = self.clone();
        cfg.shape = Shape::LogTangent;
        cfg.compute_hessian = false;
        cfg.lt_reference = None;
        let lt = cfg.fit(x, y).ok()?;
        let fr = lt.lt_frame();
        let gamma_at = |xi: &[f64]| -> Option<Vec<f64>> {
            let eta: Vec<f64> = (0..bd)
                .map(|k| (0..pt).map(|c| lt.coef[(k, c)] * xi[c]).sum())
                .collect();
            let v = lt.distribution_at(&eta, fr.as_ref()).ok()?.covariance();
            let vr: Vec<f64> = (0..d * d).map(|a| v[(a / d, a % d)]).collect();
            let gi = crate::param::gamma_from_covariance(&eta[..d], &vr);
            gi.iter().all(|v| v.is_finite()).then_some(gi)
        };
        let xbar: Vec<f64> = (0..pt)
            .map(|c| (0..n).map(|i| data.x[i * pt + c]).sum::<f64>() / n as f64)
            .collect();
        let reference = gamma_at(&xbar)?;
        let mut gs: Vec<Vec<f64>> = Vec::with_capacity(n);
        for i in 0..n {
            let mut gi = gamma_at(&data.x[i * pt..(i + 1) * pt])?;
            crate::param::gamma_align(&mut gi, &reference);
            gs.push(gi);
        }
        let mut s = DMatrix::zeros(bd, pt);
        for j in 0..pt {
            for r in 0..d {
                s[(r, j)] = lt.coef[(r, j)];
            }
        }
        for round in 0..2 {
            if round > 0 {
                for (i, gi) in gs.iter_mut().enumerate() {
                    let fitted: Vec<f64> = (0..g)
                        .map(|k| (0..pt).map(|c| s[(d + k, c)] * data.x[i * pt + c]).sum())
                        .collect();
                    crate::param::gamma_align(gi, &fitted);
                }
            }
            // shape rows grouped by their free columns
            let mut done = vec![false; g];
            for k0 in 0..g {
                if done[k0] {
                    continue;
                }
                let cols: Vec<usize> = (0..pt).filter(|&c| mask[(d + k0, c)]).collect();
                let rows: Vec<usize> = (k0..g)
                    .filter(|&k| !done[k] && (0..pt).all(|c| mask[(d + k, c)] == mask[(d + k0, c)]))
                    .collect();
                for &k in &rows {
                    done[k] = true;
                }
                if cols.is_empty() {
                    continue;
                }
                let xs = DMatrix::from_fn(n, cols.len(), |i, a| data.x[i * pt + cols[a]]);
                let rhs = DMatrix::from_fn(n, rows.len(), |i, a| gs[i][rows[a]]);
                let sol = (xs.transpose() * &xs)
                    .cholesky()?
                    .solve(&(xs.transpose() * rhs));
                for (a, &k) in rows.iter().enumerate() {
                    for (b, &c) in cols.iter().enumerate() {
                        s[(d + k, c)] = sol[(b, a)];
                    }
                }
            }
        }
        zero_fixed(&mut s, mask);
        s.iter().all(|v| v.is_finite()).then_some(s)
    }

    /// The maximum of a trapped gamma fit's likelihood, in coordinates
    /// blown up around the singular sets its suspects are near
    /// ([`BlownUp`]): there the likelihood is smooth, and L-BFGS converges by
    /// the rule of a regular fit whether each suspect's maximum is on the set
    /// (a trap, `ρ = 0`, the observation's density at its best angle) or
    /// next to it (`ρ > 0`, a regular root with curvature `O(1/r²)` in the
    /// original coordinates, out of reach of L-BFGS there). Observations
    /// newly trapped at the result join, for up to [`STRATUM_ROUNDS`]
    /// rounds; a suspect left at `ρ = 0` although the likelihood rises off
    /// the set (a saddle in `ρ`) is pushed off it and the fit repeated.
    /// The observations on the set at the result are polished there with
    /// Newton steps ([`EsagRegressor::stratum_polish`]). Returns the
    /// converged fit and the observations on the set (empty: a regular
    /// root), or `None` (log-tangent or transported shapes, a singular set
    /// that is not a pair, dependent constraints, no convergence).
    fn stratum_maximum(
        &self,
        data: &RegressionData,
        best: &RawFit,
        mask: &DMatrix<bool>,
    ) -> Option<(RawFit, Vec<usize>)> {
        if data.lt.is_some() || data.gt.is_some() {
            return None;
        }
        let (n, d, bd, pt) = (data.n, data.d, data.b_dim, data.p_total);
        let free = free_params(mask);
        let nf = free.len();
        let limit = HOMOTOPY_MAX_SET_ASIDE.max(n / 100);
        let dbg = std::env::var("ESAG_DEBUG_STRATUM").is_ok();
        let mut blown = near_singular_observations(data, &best.coef);
        let mut coef = best.coef.clone();
        let mut push: Vec<usize> = vec![];
        let (mut it, mut fev) = (best.n_iter, best.n_fev);
        for _ in 0..STRATUM_ROUNDS {
            if blown.is_empty() || blown.len() > limit {
                if dbg {
                    eprintln!(
                        "stratum: gave up with {} suspects (limit {limit})",
                        blown.len()
                    );
                }
                return None;
            }
            let k = blown.len();
            let (mut cm, mut fixed) = (DMatrix::zeros(2 * k, nf), DVector::zeros(2 * k));
            let (mut eps, mut scales) = (vec![0.0; k], vec![0.0; k]);
            for (a, &i) in blown.iter().enumerate() {
                let xi = &data.x[i * pt..(i + 1) * pt];
                let eta: Vec<f64> = (0..bd)
                    .map(|r| (0..pt).map(|c| coef[(r, c)] * xi[c]).sum())
                    .collect();
                let set = nearest_singular(data, &eta);
                let (rows, scale) = set.1.rows(d, &eta)?;
                if rows.len() != 2 || !(scale > 0.0 && scale.is_finite()) {
                    if dbg {
                        eprintln!("stratum: suspect #{i} near {set:?}: not a pair");
                    }
                    return None;
                }
                (eps[a], scales[a]) = (STRATUM_RADIUS * scale, scale);
                for (q, &r) in rows.iter().enumerate() {
                    fixed[2 * a + q] = (0..pt)
                        .filter(|&c| !mask[(r, c)])
                        .map(|c| coef[(r, c)] * xi[c])
                        .sum();
                    for (b, &(kk, c)) in free.iter().enumerate() {
                        if kk == r {
                            cm[(2 * a + q, b)] = xi[c];
                        }
                    }
                }
            }
            let cplus = (&cm * cm.transpose()).cholesky()?.solve(&cm).transpose();
            let obj = NegLogLik {
                data,
                base: coef.clone(),
                free: &free,
                parallel: self.parallel,
                trap_streak: None,
                shared: false,
            };
            let bu = BlownUp {
                inner: &obj,
                nf,
                cplus,
                c: cm,
                fixed,
                eps,
            };
            let x0 = DVector::from_iterator(nf, free.iter().map(|&(r, c)| coef[(r, c)]));
            let mut y0 = bu.y(&x0);
            for (a, i) in blown.iter().enumerate() {
                if push.contains(i) {
                    y0[nf + a] = (STRATUM_PUSH * scales[a]).sqrt();
                }
            }
            let cfg = LbfgsConfig {
                tol_grad_abs: 0.0,
                ..self.lbfgs_config(data, nf + 2 * k, self.max_iter)
            };
            let res = minimize(&bu, &y0, &cfg);
            (it, fev) = (it + res.n_iter, fev + res.n_fev);
            let x = bu.x(&res.x);
            coef = obj.assemble(x.as_slice());
            let converged = res.status.is_converged()
                && res.f.is_finite()
                && res.grad_norm_inf <= MAX_MEAN_GRADIENT * n as f64;
            let radial = bu.radial(&res.x);
            // suspects at the result that are not blown up yet: trapped, or
            // (if the run stalled) near the set
            let new: Vec<usize> = if converged {
                singular_observations(data, &coef)
            } else {
                near_singular_observations(data, &coef)
            }
            .into_iter()
            .filter(|i| !blown.contains(i))
            .collect();
            // on the set, and a saddle there (the likelihood rises off it)
            let on: Vec<usize> = (0..k)
                .filter(|&a| radial[a].0 < SINGULAR_ON * scales[a])
                .collect();
            let saddle: Vec<usize> = on
                .iter()
                .filter(|&&a| radial[a].1 < 0.0)
                .map(|&a| blown[a])
                .filter(|i| !push.contains(i))
                .collect();
            if dbg {
                eprintln!(
                    "stratum: blown up {blown:?}: {:?} it {} ll {:.2} (trapped fit {:.2}) |g| {:.1e}; ρ² {:?}; on the set {:?}, saddles {saddle:?}, new {new:?}",
                    res.status,
                    res.n_iter,
                    -res.f,
                    -best.neg_ll,
                    res.grad_norm_inf,
                    radial
                        .iter()
                        .zip(&scales)
                        .map(|(r, s)| format!("{:.1e}", r.0 / s))
                        .collect::<Vec<_>>(),
                    on.iter().map(|&a| blown[a]).collect::<Vec<_>>()
                );
            }
            if converged && new.is_empty() && saddle.is_empty() {
                let held: Vec<usize> = on.iter().map(|&a| blown[a]).collect();
                let raw = RawFit {
                    coef: coef.clone(),
                    x: x.as_slice().to_vec(),
                    neg_ll: res.f,
                    grad: vec![0.0; nf],
                    grad_norm_inf: res.grad_norm_inf,
                    n_iter: it,
                    n_fev: fev,
                    status: res.status,
                    converged,
                };
                if held.is_empty() {
                    return Some((raw, held));
                }
                // tighten on the set
                return match self.stratum_polish(data, &coef, &held, &free, mask) {
                    Some(mut p) if p.converged && p.neg_ll <= raw.neg_ll + STRATUM_SLACK => {
                        p.n_iter += it;
                        p.n_fev += fev;
                        Some((p, held))
                    }
                    _ => Some((raw, held)),
                };
            }
            if !converged && new.is_empty() && saddle.is_empty() {
                // stalled with nothing to add: finish on the set with Newton
                let held: Vec<usize> = on.iter().map(|&a| blown[a]).collect();
                if let Some(mut p) = self.stratum_polish(data, &coef, &held, &free, mask)
                    && p.converged
                    && p.neg_ll <= res.f + STRATUM_SLACK
                    && singular_observations(data, &p.coef)
                        .iter()
                        .all(|i| held.contains(i))
                {
                    p.n_iter += it;
                    p.n_fev += fev;
                    return Some((p, held));
                }
            }
            push.extend(saddle);
            blown.extend(new);
        }
        None
    }

    /// One L-BFGS run of [`EsagRegressor::stratum_maximum`]: from `coef`,
    /// with the vanishing coordinates of the observations `held` placed at
    /// [`STRATUM_RADIUS`] of their scale along their best direction and held
    /// there. `None` for dependent constraints or a singular set that is a
    /// point.
    fn stratum_polish(
        &self,
        data: &RegressionData,
        coef: &DMatrix<f64>,
        held: &[usize],
        free: &[(usize, usize)],
        mask: &DMatrix<bool>,
    ) -> Option<RawFit> {
        let (n, d, bd, pt) = (data.n, data.d, data.b_dim, data.p_total);
        // (observation, row of η, target)
        let mut cons: Vec<(usize, usize, f64)> = vec![];
        for &i in held {
            let xi = &data.x[i * pt..(i + 1) * pt];
            let eta: Vec<f64> = (0..bd)
                .map(|k| (0..pt).map(|c| coef[(k, c)] * xi[c]).sum())
                .collect();
            let (rows, scale) = nearest_singular(data, &eta).1.rows(d, &eta)?;
            if !(scale > 0.0 && scale.is_finite()) {
                return None;
            }
            let r = STRATUM_RADIUS * scale;
            let cur: Vec<f64> = rows.iter().map(|&k| eta[k]).collect();
            let cn = cur.iter().map(|v| v * v).sum::<f64>().sqrt();
            let mut dir: Vec<f64> = if cn > 0.0 {
                cur.iter().map(|v| v / cn).collect()
            } else {
                (0..rows.len())
                    .map(|a| f64::from(u8::from(a == 0)))
                    .collect()
            };
            if rows.len() == 2 {
                // the angle maximising the observation's own density there
                let y = &data.y[i * d..(i + 1) * d];
                let mut e = eta.clone();
                let mut dens = |th: f64| {
                    e[rows[0]] = r * th.cos();
                    e[rows[1]] = r * th.sin();
                    loglik::loglik_eta(&e, y)
                };
                let th0 = dir[1].atan2(dir[0]);
                let h = std::f64::consts::TAU / STRATUM_ANGLES as f64;
                let (mut bt, mut bv) = (th0, dens(th0));
                for a in 1..STRATUM_ANGLES {
                    let th = th0 + a as f64 * h;
                    let v = dens(th);
                    if v > bv {
                        (bt, bv) = (th, v);
                    }
                }
                // golden section on [bt − h, bt + h]
                let gr = 0.5 * (5f64.sqrt() - 1.0);
                let (mut lo, mut hi) = (bt - h, bt + h);
                for _ in 0..40 {
                    let (a, b) = (hi - gr * (hi - lo), lo + gr * (hi - lo));
                    if dens(a) >= dens(b) {
                        hi = b;
                    } else {
                        lo = a;
                    }
                }
                let th = 0.5 * (lo + hi);
                if dens(th) > bv {
                    bt = th;
                }
                dir = vec![bt.cos(), bt.sin()];
            }
            for (a, &k) in rows.iter().enumerate() {
                cons.push((i, k, r * dir[a]));
            }
        }
        // C over the free coefficients; fixed ones move to the target
        let (nf, nc) = (free.len(), cons.len());
        let mut cm = DMatrix::zeros(nc, nf);
        let mut rhs = DVector::zeros(nc);
        for (a, &(i, k, t)) in cons.iter().enumerate() {
            let xi = &data.x[i * pt..(i + 1) * pt];
            let fixed: f64 = (0..pt)
                .filter(|&c| !mask[(k, c)])
                .map(|c| coef[(k, c)] * xi[c])
                .sum();
            rhs[a] = t - fixed;
            for (b, &(kk, c)) in free.iter().enumerate() {
                if kk == k {
                    cm[(a, b)] = xi[c];
                }
            }
        }
        let x0 = DVector::from_iterator(nf, free.iter().map(|&(k, c)| coef[(k, c)]));
        let (x1, pinv) = if nc == 0 {
            (x0, DMatrix::zeros(0, nf))
        } else {
            let chol = (&cm * cm.transpose()).cholesky()?;
            let x1 = &x0 + cm.transpose() * chol.solve(&(&rhs - &cm * &x0));
            (x1, chol.solve(&cm))
        };
        let obj = NegLogLik {
            data,
            base: coef.clone(),
            free,
            parallel: self.parallel,
            trap_streak: None,
            shared: false,
        };
        let proj = Projected {
            inner: &obj,
            ct: cm.transpose(),
            pinv,
        };
        let cfg = LbfgsConfig {
            tol_grad_abs: 0.0,
            ..self.lbfgs_config(data, nf, self.max_iter)
        };
        let res = minimize(&proj, x1.as_slice(), &cfg);
        // Projected Newton steps with the exact Hessian: next to the
        // singular set the curvature is too large and uneven for L-BFGS.
        // The held observations' Hessians lose their vanishing rows (which
        // the projection removes anyway), so the 1/r² curvature there never
        // enters the arithmetic.
        let full = bd * pt;
        let proj_m = DMatrix::<f64>::identity(nf, nf) - cm.transpose() * &proj.pinv;
        let mut wd = data.clone();
        let mut w = vec![1.0; n];
        for &i in held {
            w[i] = 0.0;
        }
        wd.w = Some(w);
        let (mut x, mut f) = (DVector::from_column_slice(&res.x), res.f);
        let mut steps = 0;
        for _ in 0..STRATUM_NEWTON {
            let b = obj.assemble(x.as_slice());
            let (_, mut g, mut h) =
                joint_loglik_grad_hess_with(&wd, &b, self.parallel, ObsHessian::Exact);
            let (mut ge, mut he) = (vec![0.0; bd], vec![0.0; bd * bd]);
            for &i in held {
                let xi = &data.x[i * pt..(i + 1) * pt];
                let eta: Vec<f64> = (0..bd)
                    .map(|k| (0..pt).map(|c| b[(k, c)] * xi[c]).sum())
                    .collect();
                obs_hessian(
                    data,
                    ObsHessian::Exact,
                    &eta,
                    &data.y[i * d..(i + 1) * d],
                    &mut ge,
                    &mut he,
                );
                let (rows, _) = nearest_singular(data, &eta).1.rows(d, &eta)?;
                for &k in &rows {
                    ge[k] = 0.0;
                    for m in 0..bd {
                        he[k * bd + m] = 0.0;
                        he[m * bd + k] = 0.0;
                    }
                }
                for j1 in 0..pt {
                    for k1 in 0..bd {
                        g[j1 * bd + k1] += ge[k1] * xi[j1];
                        for j2 in 0..pt {
                            for k2 in 0..bd {
                                h[(j1 * bd + k1) * full + j2 * bd + k2] +=
                                    he[k1 * bd + k2] * xi[j1] * xi[j2];
                            }
                        }
                    }
                }
            }
            let gf = &proj_m * DVector::from_iterator(nf, free.iter().map(|&(k, j)| g[j * bd + k]));
            if gf.amax() <= 1e-10 * f.abs().max(1.0) {
                break;
            }
            let hf = DMatrix::from_fn(nf, nf, |a, c| {
                let ((ka, ja), (kc, jc)) = (free[a], free[c]);
                -h[(ja * bd + ka) * full + jc * bd + kc]
            });
            let scale = hf.diagonal().iter().map(|v| v.abs()).sum::<f64>() / nf as f64;
            let m = &proj_m * hf * &proj_m + cm.transpose() * &cm * scale;
            // saddle-free: |eigenvalues| (floored), so a direction of
            // positive curvature, such as an observation's angle on the
            // wrong side of its best one, is climbed instead of descended
            let eig = nalgebra::SymmetricEigen::new((&m + m.transpose()) * 0.5);
            let lmax = eig.eigenvalues.amax();
            let gq = eig.eigenvectors.transpose() * &gf;
            let sq = DVector::from_iterator(
                nf,
                (0..nf).map(|a| gq[a] / eig.eigenvalues[a].abs().max(1e-10 * lmax)),
            );
            let step = proj_m.clone() * (&eig.eigenvectors * sq); // ascent for ℓ
            let mut alpha = 1.0;
            let mut moved = false;
            for _ in 0..30 {
                let xc = &x + &step * alpha;
                let fc = obj.value(xc.as_slice());
                if fc.is_finite() && fc <= f {
                    (x, f, moved) = (xc, fc, true);
                    break;
                }
                alpha *= 0.5;
            }
            if !moved {
                break;
            }
            steps += 1;
        }
        let mut grad = vec![0.0; nf];
        let f = proj.value_and_gradient(x.as_slice(), &mut grad);
        let grad_norm_inf = grad.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let converged = f.is_finite()
            && grad_norm_inf <= MAX_MEAN_GRADIENT * n as f64
            && (res.status.is_converged() || steps > 0);
        Some(RawFit {
            coef: obj.assemble(x.as_slice()),
            x: x.as_slice().to_vec(),
            neg_ll: f,
            grad,
            grad_norm_inf,
            n_iter: res.n_iter + steps,
            n_fev: res.n_fev + steps,
            status: res.status,
            converged,
        })
    }

    /// A fit `r` (on `data`) with collapsed gamma groups, reseeded from
    /// `reference` group by group in both signs and refitted; the best
    /// refit without a collapse, if any beats `r`, else `r`.
    fn un_collapse(
        &self,
        data: &RegressionData,
        mut r: RawFit,
        free: &[(usize, usize)],
        reference: Option<&DMatrix<f64>>,
    ) -> RawFit {
        let Some(reference) = reference else { return r };
        for _ in 0..data.d {
            let groups = collapsed_groups(data, &r.coef);
            let Some(&gr) = groups.first() else { break };
            let mut found: Option<RawFit> = None;
            for start in reseeded_group(&r.coef, reference, data.d, gr) {
                let alt = self.optimize(data, &start, free, self.max_iter);
                if !collapsed_groups(data, &alt.coef).contains(&gr)
                    && found.as_ref().is_none_or(|f| alt.beats(f))
                {
                    found = Some(alt);
                }
            }
            match found {
                Some(f) => r = f,
                None => break,
            }
        }
        r
    }

    /// Weight homotopy from a trapped gamma fit to the regular root of the
    /// likelihood. The trapped observations `S` are set aside (weight 0,
    /// accumulating newly trapped ones) until the rest converges, from the
    /// trapped fit and from `alt` (the log-tangent-informed start); then
    /// their weight is raised to 1 in steps, each fit warm from the last, a
    /// step accepted only where the fit converges with no observation
    /// trapped (else it is shortened). A trap is a local feature that
    /// captures large steps; along small weight increments the fit follows
    /// the regular root. Many observations trapped at once (a group
    /// collapsing for all of them) end the attempt from that start.
    fn homotopy_rounds(
        &self,
        data: &RegressionData,
        best: &RawFit,
        mask: &DMatrix<bool>,
        alt: Option<&DMatrix<f64>>,
    ) -> Homotopy {
        let free = free_params(mask);
        let limit = HOMOTOPY_MAX_SET_ASIDE.max(data.n / 100);
        let iters = self.max_iter.min(HOMOTOPY_STEP_ITER);
        let weights = |sset: &[usize], t: f64| -> Vec<f64> {
            let mut w = vec![1.0; data.n];
            for &i in sset {
                w[i] = t;
            }
            w
        };
        let s0 = near_singular_observations(data, &best.coef);
        // the rest's fit from a start: (fit, S, members of S on the set)
        let dbg = std::env::var("ESAG_DEBUG_HOMOTOPY").is_ok();
        let rest_from = |b0: &DMatrix<f64>| -> Option<(RawFit, Vec<usize>, usize)> {
            let mut wd = data.clone();
            let mut sset = s0.clone();
            let mut cur = b0.clone();
            for _ in 0..HOMOTOPY_SET_ASIDE_ROUNDS {
                sset.sort_unstable();
                sset.dedup();
                if sset.len() > limit {
                    if dbg {
                        eprintln!("homotopy: rest gave up, {} set aside", sset.len());
                    }
                    return None;
                }
                wd.w = Some(weights(&sset, 0.0));
                let r = self.optimize(&wd, &cur, &free, iters);
                let r = if collapsed_groups(&wd, &r.coef).is_empty() {
                    r
                } else {
                    self.un_collapse(&wd, r, &free, alt)
                };
                let flagged = singular_observations(&wd, &r.coef);
                // a rest fit that did not converge sets aside its suspects
                let suspects = if r.converged {
                    flagged.clone()
                } else {
                    near_singular_observations(&wd, &r.coef)
                };
                let trapped: Vec<usize> = suspects
                    .iter()
                    .copied()
                    .filter(|i| sset.binary_search(i).is_err())
                    .collect();
                if trapped.len() > limit {
                    if dbg {
                        eprintln!(
                            "homotopy: rest collapsed ({} newly trapped, status {:?}, |g| {:.1e})",
                            trapped.len(),
                            r.status,
                            r.grad_norm_inf
                        );
                    }
                    return None;
                }
                if r.converged && trapped.is_empty() {
                    let on_set = flagged.len();
                    return Some((r, sset, on_set));
                }
                cur = r.coef;
                sset.extend(trapped);
            }
            None
        };
        let mut rests: Vec<(RawFit, Vec<usize>, usize)> =
            rest_from(&best.coef).into_iter().collect();
        if let Some(a) = alt
            && let Some(r) = rest_from(a)
            // the same set aside as the first start's: the same branch, as a
            // rule (both fold at the same weight)
            && rests.iter().all(|q| q.1 != r.1)
        {
            rests.push(r);
        }
        if std::env::var("ESAG_DEBUG_HOMOTOPY").is_ok() {
            eprintln!(
                "homotopy: start trapped {}; rests {:?}",
                s0.len(),
                rests.iter().map(|r| (r.1.len(), r.2)).collect::<Vec<_>>()
            );
        }
        // fewest set-aside observations still on the set, then highest likelihood
        rests.sort_by(|a, b| a.2.cmp(&b.2).then(a.0.neg_ll.total_cmp(&b.0.neg_ll)));
        let mut fallback: Option<(RawFit, Vec<usize>)> = None;
        for (rest, sset, _) in rests {
            let mut wd = data.clone();
            let mut cur = rest.coef.clone();
            // the whole step first (the observations put straight back), then
            // small ones
            let (mut t, mut step, mut fails) = (0.0f64, 1.0, 0);
            while t < 1.0 && fails < HOMOTOPY_MAX_FAILS {
                let tn = (t + step).min(1.0);
                wd.w = Some(weights(&sset, tn));
                let r = self.optimize(&wd, &cur, &free, iters);
                if r.converged && singular_observations(&wd, &r.coef).is_empty() {
                    cur = r.coef;
                    t = tn;
                    step = (step * 1.5).min(0.3);
                } else {
                    step = if step == 1.0 {
                        HOMOTOPY_FIRST_STEP
                    } else {
                        step * 0.4
                    };
                    fails += 1;
                }
            }
            if dbg {
                eprintln!("homotopy: S of {} reached weight {t:.3}", sset.len());
            }
            if t >= 1.0 {
                let mut fin = self.optimize(data, &cur, &free, self.max_iter);
                if fin.converged && singular_observations(data, &fin.coef).is_empty() {
                    fin.n_iter += best.n_iter;
                    fin.n_fev += best.n_fev;
                    return Homotopy::Regular(fin);
                }
            }
            if fallback.is_none() {
                fallback = Some((rest, sset));
            }
        }
        match fallback {
            Some((rest, sset)) => Homotopy::SetAside(rest, sset),
            None => Homotopy::Failed,
        }
    }

    /// Fit starting from an explicit coefficient matrix (warm start). Used by
    /// bootstrap procedures; `b0` must be `eta_len(d) × p_total`.
    pub fn fit_warm(
        &self,
        x: &DMatrix<f64>,
        y: &DMatrix<f64>,
        b0: &DMatrix<f64>,
    ) -> Result<EsagRegressorFit> {
        let data = self.data(x, y)?;
        if b0.nrows() != data.b_dim || b0.ncols() != data.p_total {
            return Err(EsagError::Shape(format!(
                "warm start must be {}×{}, got {}×{}",
                data.b_dim,
                data.p_total,
                b0.nrows(),
                b0.ncols()
            )));
        }
        let mask = self.resolve_mask(&data)?;
        let free = free_params(&mask);
        let raw = self.optimize(&data, b0, &free, self.max_iter);
        // the R-style restart rescues warm starts at low dimension; for
        // gamma at d >= 5 with covariates it does not converge either
        let retry = data.d <= 4 || data.p == 0 || data.lt.is_some();
        if raw.converged || self.max_iter == 0 || !retry {
            return self.finalize(&data, raw, mask);
        }
        // Safety net (used by every bootstrap refit): a warm start that ends
        // at a singular point of the parameterization fails the gradient
        // check; retry from the R-style start and keep the better fit.
        let mut b1 = self.clone().with_init(Init::RStyle).initial_coef(&data)?;
        zero_fixed(&mut b1, &mask);
        let alt = self.optimize(&data, &b1, &free, self.max_iter);
        let best = if alt.beats(&raw) { alt } else { raw };
        self.finalize(&data, best, mask)
    }

    fn resolve_mask(&self, data: &RegressionData) -> Result<DMatrix<bool>> {
        match &self.mask {
            None => Ok(DMatrix::from_element(data.b_dim, data.p_total, true)),
            Some(m) => {
                if m.nrows() != data.b_dim || m.ncols() != data.p_total {
                    return Err(EsagError::Shape(format!(
                        "mask must be {}×{}, got {}×{}",
                        data.b_dim,
                        data.p_total,
                        m.nrows(),
                        m.ncols()
                    )));
                }
                if !m.iter().any(|&f| f) {
                    return Err(EsagError::Shape("mask has no free parameters".into()));
                }
                Ok(m.clone())
            }
        }
    }

    fn initial_coef(&self, data: &RegressionData) -> Result<DMatrix<f64>> {
        let (d, b_dim, p_total) = (data.d, data.b_dim, data.p_total);
        let mut b = DMatrix::zeros(b_dim, p_total);
        match &self.init {
            Init::Custom(m) => {
                if m.nrows() != b_dim || m.ncols() != p_total {
                    return Err(EsagError::Shape(format!(
                        "Init::Custom must be {b_dim}×{p_total}, got {}×{}",
                        m.nrows(),
                        m.ncols()
                    )));
                }
                b.copy_from(m);
            }
            Init::RStyle => {
                // gamma: just off the isotropic kink; log-tangent: smooth at 0
                let shape0 = if data.lt.is_some() { 0.0 } else { 1e-3 };
                for j in 0..p_total {
                    for k in 0..d {
                        b[(k, j)] = 1.0;
                    }
                    for k in d..b_dim {
                        b[(k, j)] = shape0;
                    }
                }
            }
            Init::DataDriven | Init::Auto => {
                let (mu0, gamma0) = moment_start(&data.y, data.n, d);
                let shape0 = match &data.lt {
                    None => gamma0,
                    Some(fr) => {
                        let v = Frame::new(&mu0, &gamma0).cov_pow(1.0);
                        let v = DMatrix::from_row_slice(d, d, &v);
                        crate::logtangent::s_from_covariance(fr, &mu0, &v)
                    }
                };
                for k in 0..d {
                    b[(k, 0)] = mu0[k];
                }
                for (k, g) in shape0.iter().enumerate() {
                    b[(d + k, 0)] = *g;
                }
            }
        }
        Ok(b)
    }

    /// Run L-BFGS from every start and return the highest likelihood found.
    ///
    /// The starts advance in lockstep, [`RACE_ROUND`] iterations per round
    /// (in parallel), so every decision below is taken at the same points
    /// whatever the thread timing, and the result is deterministic. Once a
    /// start converges after `N` iterations, the others stop at
    /// `max(2N, N + 50)` iterations, unless they already beat the best
    /// converged likelihood (a better basin; they continue to `max_iter`).
    fn race(
        &self,
        data: &RegressionData,
        starts: &[DMatrix<f64>],
        mask: &DMatrix<bool>,
        max_iter: usize,
    ) -> RawFit {
        let mut best: Option<RawFit> = None;
        for r in self.race_all(data, starts, mask, max_iter) {
            if best.as_ref().is_none_or(|b| r.beats(b)) {
                best = Some(r);
            }
        }
        best.expect("at least one start")
    }

    /// [`EsagRegressor::race`], returning the outcome of every start (in the
    /// order of `starts`).
    fn race_all(
        &self,
        data: &RegressionData,
        starts: &[DMatrix<f64>],
        mask: &DMatrix<bool>,
        max_iter: usize,
    ) -> Vec<RawFit> {
        let free = free_params(mask);
        let objs: Vec<NegLogLik> = starts
            .iter()
            .map(|b0| self.objective(data, b0, &free))
            .collect();
        let cfg = self.lbfgs_config(data, free.len(), max_iter);
        let mut runs: Vec<Lbfgs> = objs
            .par_iter()
            .zip(starts)
            .map(|(obj, b0)| {
                let x0: Vec<f64> = free.iter().map(|&(k, j)| b0[(k, j)]).collect();
                Lbfgs::new(obj, &x0, &cfg)
            })
            .collect();
        let ok = |r: &Lbfgs| {
            r.status().is_some_and(Status::is_converged)
                && r.f().is_finite()
                && r.grad_norm_inf() <= MAX_MEAN_GRADIENT * data.n.max(1) as f64
        };
        let mut cap = usize::MAX;
        let mut best_converged = f64::INFINITY;
        let nf = data.n as f64;
        // objective value at each round boundary, per start
        let mut history: Vec<Vec<f64>> = runs.iter().map(|r| vec![r.f()]).collect();
        loop {
            // A start that has trailed the leader by more than
            // DOMINATED_PER_OBS·n after PRUNE_AFTER iterations is in a much
            // worse basin; one that gained less than STALL_PER_OBS·n over the
            // last STALL_ROUNDS rounds with a gradient far above the
            // convergence threshold is crawling against a cliff of the gamma
            // parameterization. Stop both.
            let leader = runs.iter().map(|r| r.f()).fold(f64::INFINITY, f64::min);
            for (r, h) in runs.iter_mut().zip(&history) {
                if r.status().is_some() || r.n_iter() < PRUNE_AFTER {
                    continue;
                }
                let dominated = r.f() > leader + DOMINATED_PER_OBS * nf;
                let stalled = h.len() > STALL_ROUNDS
                    && h[h.len() - 1 - STALL_ROUNDS] - r.f() < STALL_PER_OBS * nf
                    && r.grad_norm_inf() > 10.0 * MAX_MEAN_GRADIENT * nf;
                if dominated || stalled {
                    r.stop(Status::Aborted);
                }
            }
            let active: Vec<bool> = runs
                .iter()
                .map(|r| r.status().is_none() && (r.n_iter() < cap || r.f() < best_converged))
                .collect();
            if !active.iter().any(|&a| a) {
                break;
            }
            runs.par_iter_mut().zip(&active).for_each(|(r, &a)| {
                if a {
                    r.advance(RACE_ROUND);
                }
            });
            for (r, h) in runs.iter().zip(history.iter_mut()) {
                h.push(r.f());
            }
            for r in &runs {
                if ok(r) {
                    let n = r.n_iter();
                    cap = cap.min((2 * n).max(n + 50));
                    best_converged = best_converged.min(r.f());
                }
            }
        }
        let raws: Vec<RawFit> = runs
            .into_iter()
            .zip(&objs)
            .map(|(r, obj)| {
                let res = r.into_result();
                let converged = res.status.is_converged()
                    && res.f.is_finite()
                    && res.grad_norm_inf <= MAX_MEAN_GRADIENT * data.n.max(1) as f64;
                RawFit {
                    coef: obj.assemble(&res.x),
                    x: res.x,
                    neg_ll: res.f,
                    grad: res.grad,
                    grad_norm_inf: res.grad_norm_inf,
                    n_iter: res.n_iter,
                    n_fev: res.n_fev,
                    status: res.status,
                    converged,
                }
            })
            .collect();
        if self.verbose {
            eprintln!(
                "race of {}: {}",
                raws.len(),
                raws.iter()
                    .map(|r| format!(
                        "[{:?} it={} conv={} ll={:.1}]",
                        r.status, r.n_iter, r.converged, -r.neg_ll
                    ))
                    .collect::<Vec<_>>()
                    .join(" ")
            );
        }
        raws
    }

    fn objective<'a>(
        &self,
        data: &'a RegressionData,
        b0: &DMatrix<f64>,
        free: &'a [(usize, usize)],
    ) -> NegLogLik<'a> {
        NegLogLik {
            data,
            base: b0.clone(),
            free,
            parallel: self.parallel,
            // groups j ≥ 2 (where observations can get pinned) exist from d = 4;
            // in practice traps need d ≥ 5 and covariates
            trap_streak: data.gamma_cliffs().then(|| AtomicUsize::new(0)),
            shared: shared_shape(data, b0, free),
        }
    }

    fn lbfgs_config(&self, data: &RegressionData, n_free: usize, max_iter: usize) -> LbfgsConfig {
        let memory = if self.lbfgs_memory == 0 {
            n_free.clamp(10, 200)
        } else {
            self.lbfgs_memory
        };
        // With standard errors requested, `finalize` computes the exact
        // Hessian anyway and polishes with Newton steps, so L-BFGS can hand
        // over once the mean gradient per observation is below
        // NEWTON_SWITCH (≈ a quarter of the iterations saved).
        let tol_grad_abs = if self.compute_hessian {
            self.handover.unwrap_or(NEWTON_SWITCH) * data.n as f64
        } else {
            0.0
        };
        LbfgsConfig {
            memory,
            max_iter,
            tol_grad: self.tol,
            tol_grad_abs,
            verbose: self.verbose,
            ..LbfgsConfig::default()
        }
    }

    /// One L-BFGS run from `b0` over the free coefficients.
    fn optimize(
        &self,
        data: &RegressionData,
        b0: &DMatrix<f64>,
        free: &[(usize, usize)],
        max_iter: usize,
    ) -> RawFit {
        let obj = self.objective(data, b0, free);
        let cfg = self.lbfgs_config(data, free.len(), max_iter);
        let x0: Vec<f64> = free.iter().map(|&(k, j)| b0[(k, j)]).collect();
        // Where gamma's likelihood has cliffs, observations near the
        // singular set (§2 of MATH.md) give it large curvature that changes
        // quickly and depends on the responses, so the information of the
        // original fit is a poor metric for a refit. The empirical
        // information at the start tracks them, for about one evaluation.
        let local =
            if self.local_precondition && (data.gamma_cliffs() || self.precondition.is_none()) {
                empirical_precondition(data, b0, free)
            } else {
                None
            };
        // whitened with an inverse information, the Hessian is near I: start
        // with the Newton step, and stop once the predicted remaining gain
        // ½‖g_z‖² is below REFIT_DECREMENT (HANDOFF_DECREMENT when `finalize`
        // polishes with Newton steps). The absolute gradient rule of `cfg`
        // is in θ units and does not apply to g_z.
        let whitened = |l: &DMatrix<f64>, start: &[f64], iters: usize| -> OptimResult {
            let w = Whitened {
                inner: &obj,
                theta0: start.to_vec(),
                l,
            };
            let wcfg = LbfgsConfig {
                unit_scale: true,
                tol_decrement: if self.compute_hessian {
                    HANDOFF_DECREMENT
                } else {
                    self.refit_decrement.unwrap_or(REFIT_DECREMENT)
                },
                tol_grad_abs: 0.0,
                max_iter: iters,
                ..cfg.clone()
            };
            let rz = minimize(&w, &vec![0.0; free.len()], &wcfg);
            // back to θ; g_z = Lᵀ g_θ gives the θ-gradient without another
            // evaluation
            let x = w.theta(&rz.x);
            let grad: Vec<f64> = l
                .tr_solve_lower_triangular(&DVector::from_column_slice(&rz.grad))
                .map(|g| g.iter().copied().collect())
                .unwrap_or_else(|| {
                    let mut g = vec![0.0; x.len()];
                    obj.value_and_gradient(&x, &mut g);
                    g
                });
            OptimResult {
                grad_norm_inf: grad.iter().fold(0.0f64, |m, v| m.max(v.abs())),
                x,
                f: rz.f,
                grad,
                n_iter: rz.n_iter,
                n_fev: rz.n_fev,
                status: rz.status,
            }
        };
        let r = match local.as_ref().or(self.precondition.as_ref()) {
            Some(l) if local.is_some() => {
                // the empirical information goes stale as the refit moves
                // away from its start: recompute it every METRIC_REFRESH
                // iterations (a third fewer evaluations for refits that
                // travel far, as the second start of a full refit does)
                let seg = METRIC_REFRESH.min(cfg.max_iter);
                let mut r = whitened(l, &x0, seg);
                let (mut it, mut fev) = (r.n_iter, r.n_fev);
                while r.status == Status::MaxIterations && it < cfg.max_iter {
                    let Some(l2) = empirical_precondition(data, &obj.assemble(&r.x), free) else {
                        break;
                    };
                    let x = r.x.clone();
                    r = whitened(&l2, &x, METRIC_REFRESH.min(cfg.max_iter - it));
                    it += r.n_iter;
                    fev += r.n_fev;
                }
                r.n_iter = it;
                r.n_fev = fev;
                r
            }
            Some(l) if l.nrows() == free.len() && l.ncols() == free.len() => {
                whitened(l, &x0, cfg.max_iter)
            }
            _ => minimize(&obj, &x0, &cfg),
        };
        let converged = r.converged()
            && r.f.is_finite()
            && r.grad_norm_inf <= MAX_MEAN_GRADIENT * data.n.max(1) as f64;
        RawFit {
            coef: obj.assemble(&r.x),
            x: r.x,
            neg_ll: r.f,
            grad: r.grad,
            grad_norm_inf: r.grad_norm_inf,
            n_iter: r.n_iter,
            n_fev: r.n_fev,
            status: r.status,
            converged,
        }
    }

    /// Canonicalise the winning optimum, polish it with Newton steps and
    /// compute the observed information (if requested), and package the fit.
    fn finalize(
        &self,
        data: &RegressionData,
        best: RawFit,
        mask: DMatrix<bool>,
    ) -> Result<EsagRegressorFit> {
        self.finalize_with(data, best, mask, true)
    }

    /// [`EsagRegressor::finalize`]; `polish = false` keeps the coefficients
    /// (no Newton steps or L-BFGS finish), for a maximum on the singular
    /// set whose information is computed on the other observations.
    fn finalize_with(
        &self,
        data: &RegressionData,
        mut best: RawFit,
        mask: DMatrix<bool>,
        polish: bool,
    ) -> Result<EsagRegressorFit> {
        if !best.neg_ll.is_finite() {
            return Err(EsagError::Convergence(
                "objective is not finite at the optimiser's result".into(),
            ));
        }
        let free = free_params(&mask);
        let obj = NegLogLik {
            data,
            base: best.coef.clone(),
            free: &free,
            parallel: self.parallel,
            trap_streak: None,
            shared: shared_shape(data, &best.coef, &free),
        };
        // Pick the canonical member of the {B, B with gamma group 1 negated}
        // equivalence class (identical likelihood, see `canonicalize_coef`).
        let mut coef = best.coef.clone();
        if data.lt.is_none()
            && data.gt.is_none()
            && data.p_total == 1
            && mask.iter().all(|&f| f)
            && data.d >= 4
        {
            // Intercept-only fit: replace gamma by the canonical member of its
            // equivalence class (same V, same likelihood), so equivalent
            // solutions report identical gamma in every dimension.
            let d = data.d;
            let mu: Vec<f64> = (0..d).map(|k| coef[(k, 0)]).collect();
            let gamma: Vec<f64> = (d..data.b_dim).map(|k| coef[(k, 0)]).collect();
            let v = crate::frame::Frame::new(&mu, &gamma).cov_pow(1.0);
            let canon = crate::param::gamma_from_covariance(&mu, &v);
            let mut cand = coef.clone();
            for (k, g) in canon.iter().enumerate() {
                cand[(d + k, 0)] = *g;
            }
            let xs: Vec<f64> = free.iter().map(|&(k, j)| cand[(k, j)]).collect();
            let mut g = vec![0.0; xs.len()];
            let f = obj.value_and_gradient(&xs, &mut g);
            if f.is_finite() && (f - best.neg_ll).abs() <= 1e-8 * best.neg_ll.abs().max(1.0) {
                coef = cand;
                best.x = xs;
                best.grad_norm_inf = g.iter().fold(0.0f64, |m, v| m.max(v.abs()));
                best.grad = g;
            }
        }
        if data.lt.is_none() && canonicalize_coef(&mut coef, data.d) {
            for (xi, &(k, _)) in best.x.iter_mut().zip(&free) {
                if k == data.d || k == data.d + 1 {
                    *xi = -*xi;
                }
            }
            for (gi, &(k, _)) in best.grad.iter_mut().zip(&free) {
                if k == data.d || k == data.d + 1 {
                    *gi = -*gi;
                }
            }
        }

        let (hessian, cov_params) = if self.compute_hessian {
            // Exact Hessian (second-order AD per observation); up to three
            // Newton steps polish the L-BFGS optimum to ‖∇ℓ‖ ≈ 1e-10, then
            // the observed information is evaluated at the polished point.
            let nf = free.len();
            let restrict = |h_full: &[f64], pt: usize| -> DMatrix<f64> {
                let full = data.b_dim * pt;
                DMatrix::from_fn(nf, nf, |a, b| {
                    let (ka, ja) = free[a];
                    let (kb, jb) = free[b];
                    -h_full[(ja * data.b_dim + ka) * full + jb * data.b_dim + kb]
                })
            };
            // One exact Hessian at the L-BFGS optimum; Newton steps reuse it
            // (simplified Newton, each step checked with a gradient-only
            // evaluation). The step is ~‖g‖/λ_min ≲ 1e-4, so the Hessian
            // changes negligibly and is reported as the observed information.
            // (central differences for fits that did not converge, see
            // `ObsHessian::Differences`)
            let kind = |converged: bool| {
                if converged {
                    ObsHessian::Exact
                } else {
                    ObsHessian::Differences
                }
            };
            let (mut f0, mut g0, mut h_full) =
                joint_loglik_grad_hess_with(data, &coef, self.parallel, kind(best.converged));
            let mut chol = restrict(&h_full, data.p_total).cholesky();
            // (not for fits diagnosed as pinned at a singular set or with an
            // unbounded likelihood: more iterations only drift there)
            let hopeless = || {
                !best.converged
                    && (!singular_observations(data, &coef).is_empty()
                        || !degenerate_observations(data, &coef).is_empty())
            };
            if chol.is_none()
                && polish
                && self.compute_hessian
                && best.status != Status::Aborted
                && !hopeless()
            {
                // Not at a strict local maximum yet: finish with L-BFGS at
                // full tolerance, then recompute the Hessian.
                let mut fine = self.clone();
                fine.compute_hessian = false;
                let raw = fine.optimize(data, &coef, &free, self.max_iter);
                if raw.neg_ll <= best.neg_ll {
                    coef = raw.coef.clone();
                    best = RawFit {
                        converged: raw.converged || best.converged,
                        ..raw
                    };
                    (f0, g0, h_full) = joint_loglik_grad_hess_with(
                        data,
                        &coef,
                        self.parallel,
                        kind(best.converged),
                    );
                    chol = restrict(&h_full, data.p_total).cholesky();
                }
            }
            let mut f_ll = f0;
            let mut g_full = g0.as_slice().to_vec();
            if let Some(ch) = chol.filter(|_| polish) {
                for _ in 0..NEWTON_STEPS {
                    let g: DVector<f64> = DVector::from_iterator(
                        nf,
                        free.iter().map(|&(k, j)| g_full[j * data.b_dim + k]),
                    );
                    if g.amax() <= 1e-10 * f_ll.abs().max(1.0) {
                        break;
                    }
                    let step = ch.solve(&g); // ascent step for ℓ: (−∇²ℓ)⁻¹ ∇ℓ
                    let mut cand = coef.clone();
                    for (a, &(k, j)) in free.iter().enumerate() {
                        cand[(k, j)] += step[a];
                    }
                    let (f2, g2) = joint_loglik_and_grad(data, &cand, self.parallel);
                    if !(f2.is_finite() && f2 >= f_ll - 1e-12 * f_ll.abs().max(1.0)) {
                        break;
                    }
                    coef = cand;
                    f_ll = f2;
                    g_full = (0..data.p_total)
                        .flat_map(|j| (0..data.b_dim).map(move |k| (k, j)))
                        .map(|(k, j)| g2[(k, j)])
                        .collect();
                }
            }
            best.neg_ll = -f_ll;
            best.grad_norm_inf = free
                .iter()
                .map(|&(k, j)| g_full[j * data.b_dim + k].abs())
                .fold(0.0, f64::max);
            let hm = restrict(&h_full, data.p_total);
            let cov = hm.clone().try_inverse();
            (Some(hm), cov)
        } else {
            (None, None)
        };

        let feature_names = self.param_column_names(data);
        let mut config = self.clone();
        if let Some(fr) = data.lt.as_ref().or(data.gt.as_ref()) {
            config.lt_reference = Some(fr.reference().to_vec());
        }
        config.precondition = None;
        config.local_precondition = false;
        let singular_obs = if best.converged || data.lt.is_some() {
            vec![]
        } else {
            singular_observations(data, &coef)
        };
        let degenerate_obs = if best.converged {
            vec![]
        } else {
            degenerate_observations(data, &coef)
        };
        Ok(EsagRegressorFit {
            d: data.d,
            n_features: data.p,
            fit_intercept: self.fit_intercept,
            n_obs: data.n,
            singular_obs,
            degenerate_obs,
            set_aside_obs: vec![],
            non_regular: false,
            coef,
            mask,
            free,
            log_likelihood: -best.neg_ll,
            n_iter: best.n_iter,
            n_fev: best.n_fev,
            status: best.status,
            converged: best.converged,
            grad_norm: best.grad_norm_inf,
            hessian,
            cov_params,
            column_names: feature_names,
            config,
        })
    }

    fn param_column_names(&self, data: &RegressionData) -> Vec<String> {
        let mut names = Vec::with_capacity(data.p_total);
        if self.fit_intercept {
            names.push("const".to_string());
        }
        for j in 0..data.p {
            let nm = self
                .feature_names
                .as_ref()
                .and_then(|v| v.get(j).cloned())
                .unwrap_or_else(|| format!("x{}", j + 1));
            names.push(nm);
        }
        names
    }
}

/// Normalised sample mean of unit vectors `y` (row-major `n × d`), or `e_d`
/// if the mean is (numerically) zero.
fn mean_direction(y: &[f64], n: usize, d: usize) -> Vec<f64> {
    let mut m = vec![0.0; d];
    for i in 0..n {
        for k in 0..d {
            m[k] += y[i * d + k];
        }
    }
    let nrm = m.iter().map(|v| v * v).sum::<f64>().sqrt();
    if nrm > 1e-8 * n as f64 {
        m.iter_mut().for_each(|v| *v /= nrm);
    } else {
        m = vec![0.0; d];
        m[d - 1] = 1.0;
    }
    m
}

/// Moment-based starting values from unit vectors `y` (row-major `n × d`):
/// the mean direction `m̂`, the tangent-plane scatter
/// `S = mean_i (I − m̂m̂ᵀ) y_i y_iᵀ (I − m̂m̂ᵀ)`, and the near-mode
/// approximation `S ≈ V/‖μ‖²`. Hence `‖μ‖ = (Π σ_j)^{−1/(2(d−1))}` from the
/// tangent eigenvalues `σ_j` of `S`, and `γ` is the parameter of the shape of
/// `S` ([`crate::param::gamma_from_covariance`]). Group norms are capped at 25.
pub(crate) fn moment_start(y: &[f64], n: usize, d: usize) -> (Vec<f64>, Vec<f64>) {
    let mut ybar = vec![0.0; d];
    for i in 0..n {
        for k in 0..d {
            ybar[k] += y[i * d + k] / n as f64;
        }
    }
    let rbar = ybar.iter().map(|v| v * v).sum::<f64>().sqrt();
    let g_len = dims::gamma_len(d);
    if !(rbar > 1e-8) {
        let mut mu = vec![0.0; d];
        mu[d - 1] = 0.5;
        return (mu, vec![0.01; g_len]);
    }
    let m: Vec<f64> = ybar.iter().map(|v| v / rbar).collect();
    // tangent scatter
    let mut s = vec![0.0; d * d];
    let mut t = vec![0.0; d];
    for i in 0..n {
        let yi = &y[i * d..(i + 1) * d];
        let c: f64 = yi.iter().zip(&m).map(|(a, b)| a * b).sum();
        for k in 0..d {
            t[k] = yi[k] - c * m[k];
        }
        for a in 0..d {
            for b in 0..d {
                s[a * d + b] += t[a] * t[b] / n as f64;
            }
        }
    }
    // tangent eigenvalues of S via U_{-d}ᵀ S U_{-d}
    let u = crate::onb::onb(&m);
    let k = d - 1;
    let uk = DMatrix::from_fn(d, k, |r, c| u[c * d + r]);
    let sm = DMatrix::from_row_slice(d, d, &s);
    let tm = uk.transpose() * &sm * &uk;
    let eig = nalgebra::SymmetricEigen::new((&tm + tm.transpose()) * 0.5);
    let smax = eig.eigenvalues.iter().fold(0.0f64, |a, &b| a.max(b));
    if !(smax > 0.0) {
        return (m.iter().map(|v| v * 3.0).collect(), vec![0.01; g_len]);
    }
    // floor tiny eigenvalues so group norms stay moderate (ratios ≤ 26)
    let floor = smax / 26f64.powi(k as i32);
    let log_gm = eig
        .eigenvalues
        .iter()
        .map(|&x| x.max(floor).ln())
        .sum::<f64>()
        / k as f64;
    let norm = (-0.5 * log_gm).exp().clamp(0.5, 50.0);
    let mut sreg = sm.clone();
    // add the floored eigen-part back in the tangent space
    for (j, &lam) in eig.eigenvalues.iter().enumerate() {
        if lam < floor {
            let vj = &uk * eig.eigenvectors.column(j);
            sreg += (&vj * vj.transpose()) * (floor - lam);
        }
    }
    let sv: Vec<f64> = (0..d)
        .flat_map(|r| (0..d).map(move |c| (r, c)))
        .map(|(r, c)| sreg[(r, c)])
        .collect();
    let mut gamma = crate::param::gamma_from_covariance(&m, &sv);
    for gr in 1..=d - 2 {
        let off = gr * (gr + 1) / 2 - 1;
        let gnorm = gamma[off..off + gr + 1]
            .iter()
            .map(|v| v * v)
            .sum::<f64>()
            .sqrt();
        if gnorm > 25.0 {
            for v in &mut gamma[off..off + gr + 1] {
                *v *= 25.0 / gnorm;
            }
        }
        if gnorm < 1e-3 {
            // step off the isotropic kink
            for v in &mut gamma[off..off + gr + 1] {
                *v += 0.01;
            }
        }
    }
    (m.iter().map(|v| v * norm).collect(), gamma)
}

/// Outcome of [`EsagRegressor::homotopy_rounds`].
enum Homotopy {
    /// A converged fit of the full likelihood with no observation trapped.
    Regular(RawFit),
    /// The full likelihood was not reached, but the fit with these
    /// observations set aside (weight 0) converged.
    SetAside(RawFit, Vec<usize>),
    Failed,
}

/// The outcome of one optimiser run, before canonicalisation and standard
/// errors.
struct RawFit {
    coef: DMatrix<f64>,
    /// free parameters in optimiser order
    x: Vec<f64>,
    neg_ll: f64,
    grad: Vec<f64>,
    grad_norm_inf: f64,
    n_iter: usize,
    n_fev: usize,
    status: Status,
    converged: bool,
}

impl RawFit {
    /// Higher likelihood wins (we are searching for the maximum); within a
    /// relative 1e-9 a converged fit beats an unconverged one.
    fn beats(&self, other: &RawFit) -> bool {
        if !other.neg_ll.is_finite() {
            return self.neg_ll.is_finite() || (self.converged && !other.converged);
        }
        let tie = 1e-9 * other.neg_ll.abs().max(1.0);
        if (self.neg_ll - other.neg_ll).abs() <= tie {
            self.converged && !other.converged
        } else {
            self.neg_ll < other.neg_ll
        }
    }
}

/// Free coefficients `(row, column)` of a mask, column by column.
pub(crate) fn free_params(mask: &DMatrix<bool>) -> Vec<(usize, usize)> {
    (0..mask.ncols())
        .flat_map(|j| (0..mask.nrows()).map(move |k| (k, j)))
        .filter(|&(k, j)| mask[(k, j)])
        .collect()
}

/// Fixed (masked-out) coefficients are held at zero.
fn zero_fixed(b: &mut DMatrix<f64>, mask: &DMatrix<bool>) {
    for (v, &free) in b.iter_mut().zip(mask.iter()) {
        if !free {
            *v = 0.0;
        }
    }
}

/// Iterations per lockstep round of [`EsagRegressor::race`].
const RACE_ROUND: usize = 25;
/// A raced start trailing the best one by more than this many log-likelihood
/// units per observation after [`PRUNE_AFTER`] iterations is abandoned.
const DOMINATED_PER_OBS: f64 = 0.05;
const PRUNE_AFTER: usize = 100;
/// A raced start gaining less than this per observation over the last
/// [`STALL_ROUNDS`] rounds, with a large gradient, has stalled.
const STALL_PER_OBS: f64 = 1e-3;
const STALL_ROUNDS: usize = 4;

/// Iterations after which a refit preconditioned with the empirical
/// information recomputes it at the current point.
const METRIC_REFRESH: usize = 50;

/// Mean gradient per observation at which L-BFGS hands over to the Newton
/// polish when standard errors are requested.
const NEWTON_SWITCH: f64 = 1e-4;
/// Maximum Newton polishing steps (each reuses the one exact Hessian).
const NEWTON_STEPS: usize = 8;

/// Every this many iterations, the single whitened start of a log-tangent
/// fit checks for an unbounded likelihood ([`degenerate_observations`]).
const DIVERGENCE_CHECK: usize = 200;

/// Perturbed safety-net starts raced by [`EsagRegressor::fit`] (in waves of
/// [`RESTART_WAVE`]) when no default start converges.
const AUTO_RESTARTS: usize = 8;
const RESTART_WAVE: usize = 4;
/// Iteration cap for those starts: converged ESAG fits need a few hundred
/// iterations; a start still running after this is stuck at a singular point.
const RESTART_MAX_ITER: usize = 600;

/// `k` random starts, cycling through `centers`. Free coefficients get
/// Gaussian noise: s.d. 0.5 (`mu` rows) or 0.25 (`gamma` rows) on the
/// intercept column and 0.3× that, divided by the column's root mean square,
/// on the other design columns, so every column perturbs `eta` by a similar
/// amount whatever the scale of the covariate. Reproducible from `seed`.
fn perturbed_starts(
    data: &RegressionData,
    centers: &[DMatrix<f64>],
    mask: &DMatrix<bool>,
    k: usize,
    seed: u64,
) -> Vec<DMatrix<f64>> {
    let pt = data.p_total;
    let rms: Vec<f64> = (0..pt)
        .map(|j| {
            let ss: f64 = (0..data.n).map(|i| data.x[i * pt + j].powi(2)).sum();
            (ss / data.n as f64).sqrt().max(1e-12)
        })
        .collect();
    // an intercept column is all ones (rms 1); treat it as the "level" column
    let level = |j: usize| j == 0 && data.p_total > data.p;
    let mut rng = crate::stream_rng(seed, crate::streams::RESTARTS, 0);
    (0..k)
        .map(|r| {
            let mut b = centers[r % centers.len()].clone();
            for j in 0..pt {
                for kk in 0..data.b_dim {
                    if !mask[(kk, j)] {
                        continue;
                    }
                    let base = if kk < data.d { 0.5 } else { 0.25 };
                    let sd = if level(j) { base } else { 0.3 * base / rms[j] };
                    let z: f64 = StandardNormal.sample(&mut rng);
                    b[(kk, j)] += sd * z;
                }
            }
            b
        })
        .collect()
}
/// An observation counts as singular when its gradient in `η` exceeds this
/// many times the median observation's ([`singular_observations`]).
const SINGULAR_GRADIENT: f64 = 20.0;
/// ... and it lies within this relative distance of a singular set.
const SINGULAR_DISTANCE: f64 = 1e-2;
/// Within this relative distance an observation is on the singular set,
/// whatever its gradient: a trapped observation at its best angle has a
/// small angular gradient (measured at d = 10: pair ratio 6e−8 at 9× the
/// median gradient, a vanishing group at 6e−10 of the lower ones).
const SINGULAR_ON: f64 = 1e-4;
/// Weight homotopy ([`EsagRegressor::homotopy_rounds`]): rounds of setting
/// newly trapped observations aside (also stopped once a round traps as
/// many new ones as the last: the traps are spreading), the most
/// observations set aside (or 0.5% of `n`), the first small weight step,
/// and the failed steps allowed. Rescues at d = 10 set aside 1–5
/// observations in 1–3 rounds; on the hard scenario at d = 8, where they
/// fail, unbounded rounds cost 20 times the fit.
const HOMOTOPY_SET_ASIDE_ROUNDS: usize = 8;
const HOMOTOPY_MAX_SET_ASIDE: usize = 25;
const HOMOTOPY_FIRST_STEP: f64 = 0.05;
const HOMOTOPY_MAX_FAILS: usize = 8;
/// [`EsagRegressor::stratum_maximum`]: the radius, relative to their
/// scale, at which the vanishing coordinates of a trapped observation are
/// held (the limit of its density is reached to that order), the angles
/// searched for its best direction, and the rounds of joining newly
/// trapped observations.
const STRATUM_RADIUS: f64 = 1e-8;
const STRATUM_ANGLES: usize = 360;
const STRATUM_ROUNDS: usize = 8;
/// A suspect left at a saddle on the set restarts at this relative radius.
const STRATUM_PUSH: f64 = 1e-3;
/// Projected Newton steps after the L-BFGS run of a stratum polish.
const STRATUM_NEWTON: usize = 12;
/// A maximum on the singular set is kept if its log-likelihood is at most
/// this much below the trapped fit's (an observation held there that was
/// not drawn to it costs likelihood).
const STRATUM_SLACK: f64 = 1e-6;

/// Iterations of each warm-started fit within the homotopy: steps that
/// succeed need far fewer, and ones that do not would otherwise run to
/// `max_iter`.
const HOMOTOPY_STEP_ITER: usize = 600;
/// A mean direction with `√(μ̂_0² + μ̂_1²)` below this (about 17° from the
/// pole set of `onb(μ)`) gets a summary note recommending
/// [`Shape::GammaTransported`]. The slow cases measured sat at 0.12–0.19
/// (gamma 3–5× slower there and converging less often), the generic ones at
/// 0.60–0.78.
const POLE_NEAR: f64 = 0.3;

/// Observations of a gamma fit at `b` that sit at a singular set of the
/// parameterization, where `V(γ)` is discontinuous: a group `j ≥ 2` whose
/// trailing pair nearly vanishes (relative to the group norm), a group
/// `j ≥ 2` that nearly vanishes (relative to the largest lower group), or a
/// mean direction near the pole `μ̂₀ = μ̂₁ = 0` of the frame `onb(μ)` (the
/// antipode of the reference for `Shape::GammaTransported`). Flagged when
/// within [`SINGULAR_ON`] of such a set, or within [`SINGULAR_DISTANCE`]
/// with a gradient over [`SINGULAR_GRADIENT`] times the median. Near
/// such a set the observation's density changes by `O(1)` as the
/// coefficients turn around it, so the observation on the set at its best
/// angle is a trap: a non-regular local supremum of the likelihood, where
/// optimisers stall (`docs/MATH.md` §2). Empty for the log-tangent shape.
pub(crate) fn singular_observations(data: &RegressionData, b: &DMatrix<f64>) -> Vec<usize> {
    let rows = singular_rows(data, b);
    if rows.is_empty() {
        return vec![];
    }
    let mut norms: Vec<f64> = rows.iter().map(|r| r.0).collect();
    let mid = norms.len() / 2;
    let (_, med, _) = norms.select_nth_unstable_by(mid, f64::total_cmp);
    let cut = SINGULAR_GRADIENT * *med;
    // on the set (at its best angle the gradient need not be large), or
    // near it with a large gradient
    rows.iter()
        .enumerate()
        .filter(|(_, r)| r.1 < SINGULAR_ON || (r.0 > cut && r.1 < SINGULAR_DISTANCE))
        .map(|(i, _)| i)
        .collect()
}

/// Observations of a gamma fit at `b` within [`SINGULAR_DISTANCE`] of a
/// singular set, whatever their gradient: the suspects of a fit that did
/// not converge (an optimiser creeping towards a trap is stopped by the
/// race before the observation reaches the set or its gradient the
/// threshold of [`singular_observations`]). A superset of those.
pub(crate) fn near_singular_observations(data: &RegressionData, b: &DMatrix<f64>) -> Vec<usize> {
    singular_rows(data, b)
        .iter()
        .enumerate()
        .filter(|(_, r)| r.1 < SINGULAR_DISTANCE)
        .map(|(i, _)| i)
        .collect()
}

/// A singular set of the gamma parameterization (see
/// [`singular_observations`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SingularSet {
    /// The pole `μ̂₀ = μ̂₁ = 0` of the frame `onb(μ)`.
    Pole,
    /// The antipode of the reference of [`Shape::GammaTransported`].
    Antipode,
    /// The trailing pair of group `j ≥ 2`.
    Pair(usize),
    /// Group `j ≥ 2`, above a nonzero lower group.
    Zero(usize),
}

impl SingularSet {
    /// The rows of an observation's `η` that vanish on the set, and the
    /// norm they are measured against (`None` for the antipode, a point).
    fn rows(self, d: usize, eta: &[f64]) -> Option<(Vec<usize>, f64)> {
        let nrm = |r: std::ops::Range<usize>| eta[r].iter().map(|v| v * v).sum::<f64>().sqrt();
        let group = |j: usize| d + j * (j + 1) / 2 - 1..d + j * (j + 1) / 2 + j;
        match self {
            Self::Pole => Some((vec![0, 1], nrm(0..d))),
            Self::Antipode => None,
            Self::Pair(j) => {
                let g = group(j);
                Some((vec![g.end - 2, g.end - 1], nrm(g)))
            }
            Self::Zero(j) => {
                let lower = (1..j).map(|l| nrm(group(l))).fold(0.0, f64::max);
                Some((group(j).collect(), lower))
            }
        }
    }
}

/// The relative distance of an observation's `η` to the nearest singular
/// set of the gamma parameterization, and that set.
fn nearest_singular(data: &RegressionData, eta: &[f64]) -> (f64, SingularSet) {
    let d = data.d;
    let mm = eta[..d].iter().map(|v| v * v).sum::<f64>();
    let mut best = match (&data.gt, mm > 0.0) {
        // transported frame: ‖μ̂ + m‖/2, zero at the antipode
        (Some(fr), true) => {
            let inv = 1.0 / mm.sqrt();
            let m = fr.reference();
            let r = (0..d)
                .map(|k| (eta[k] * inv + m[k]).powi(2))
                .sum::<f64>()
                .sqrt()
                * 0.5;
            (r, SingularSet::Antipode)
        }
        // onb(μ): its pole μ̂_0 = μ̂_1 = 0
        (None, true) => (
            ((eta[0] * eta[0] + eta[1] * eta[1]) / mm).sqrt(),
            SingularSet::Pole,
        ),
        (_, false) => (1.0, SingularSet::Pole),
    };
    // the largest norm of the groups below the current one: a group j ≥ 2
    // vanishing above a nonzero lower group is singular too
    let mut lower = (eta[d] * eta[d] + eta[d + 1] * eta[d + 1]).sqrt();
    for gr in 2..=d - 2 {
        let off = d + gr * (gr + 1) / 2 - 1;
        let grp = &eta[off..off + gr + 1];
        let all = grp.iter().map(|v| v * v).sum::<f64>().sqrt();
        if all > 0.0 {
            let pair = (grp[gr - 1].powi(2) + grp[gr].powi(2)).sqrt();
            if pair / all < best.0 {
                best = (pair / all, SingularSet::Pair(gr));
            }
        }
        if lower > 0.0 && all / lower < best.0 {
            best = (all / lower, SingularSet::Zero(gr));
        }
        lower = lower.max(all);
    }
    best
}

/// Per observation of a gamma fit at `b`: the norm of its gradient in `η`
/// and its relative distance to the nearest singular set (see
/// [`singular_observations`]). Empty for the log-tangent shape.
fn singular_rows(data: &RegressionData, b: &DMatrix<f64>) -> Vec<(f64, f64)> {
    if data.lt.is_some() {
        return vec![];
    }
    let (d, bd, pt) = (data.d, data.b_dim, data.p_total);
    let mut eta = vec![0.0; bd];
    let mut g = vec![0.0; bd];
    (0..data.n)
        .map(|i| {
            for (k, e) in eta.iter_mut().enumerate() {
                *e = (0..pt).map(|j| b[(k, j)] * data.x[i * pt + j]).sum();
            }
            kernel_grad(data, &eta, &data.y[i * d..(i + 1) * d], &mut g);
            let gn = g.iter().map(|v| v * v).sum::<f64>().sqrt();
            (gn, nearest_singular(data, &eta).0)
        })
        .collect()
}

/// Gamma groups `j ≥ 2` that have collapsed at `b`: the group (relative to
/// the largest lower group) or its trailing pair (relative to the group)
/// nearly zero for most observations (median ratio below [`SINGULAR_ON`]).
/// Such a fit sits on the singular set for every observation at once; it
/// comes from a path that reversed the group's direction through zero.
pub(crate) fn collapsed_groups(data: &RegressionData, b: &DMatrix<f64>) -> Vec<usize> {
    if data.lt.is_some() || data.d < 4 {
        return vec![];
    }
    let (d, bd, pt) = (data.d, data.b_dim, data.p_total);
    let mut eta = vec![0.0; bd];
    let mut ratios = vec![vec![0.0; data.n]; d - 3];
    for i in 0..data.n {
        for (k, e) in eta.iter_mut().enumerate() {
            *e = (0..pt).map(|j| b[(k, j)] * data.x[i * pt + j]).sum();
        }
        let mut lower = (eta[d] * eta[d] + eta[d + 1] * eta[d + 1]).sqrt();
        for gr in 2..=d - 2 {
            let off = d + gr * (gr + 1) / 2 - 1;
            let grp = &eta[off..off + gr + 1];
            let all = grp.iter().map(|v| v * v).sum::<f64>().sqrt();
            let pair = (grp[gr - 1].powi(2) + grp[gr].powi(2)).sqrt();
            let zero = if lower > 0.0 { all / lower } else { 1.0 };
            let pole = if all > 0.0 { pair / all } else { 1.0 };
            ratios[gr - 2][i] = zero.min(pole);
            lower = lower.max(all);
        }
    }
    ratios
        .iter_mut()
        .enumerate()
        .filter_map(|(k, r)| {
            let mid = r.len() / 2;
            let (_, med, _) = r.select_nth_unstable_by(mid, f64::total_cmp);
            (*med < SINGULAR_ON).then_some(k + 2)
        })
        .collect()
}

/// `b` with the coefficients of gamma group `gr` replaced by ± its
/// intercept in `reference` (zero slopes): the two ways out of a collapse.
fn reseeded_group(
    b: &DMatrix<f64>,
    reference: &DMatrix<f64>,
    d: usize,
    gr: usize,
) -> [DMatrix<f64>; 2] {
    let off = d + gr * (gr + 1) / 2 - 1;
    let mut out = [b.clone(), b.clone()];
    for (s, c) in [1.0, -1.0].into_iter().zip(out.iter_mut()) {
        for k in off..off + gr + 1 {
            for j in 0..c.ncols() {
                c[(k, j)] = if j == 0 { s * reference[(k, 0)] } else { 0.0 };
            }
        }
    }
    out
}

/// An observation whose log-density exceeds the median observation's by
/// more than this many nats is fitted by a nearly singular `V(x_i)`
/// ([`degenerate_observations`]). Converged fits stay below 9 (n = 100 to
/// 2000, d = 5 and 8); fits that diverge reach 17 to 75.
const DEGENERATE_EXCESS: f64 = 12.0;

/// Observations whose fitted density at `b` exceeds the median
/// observation's by more than [`DEGENERATE_EXCESS`] nats: the fit is
/// driving `V(x_i)` towards a singular matrix there. With almost as many
/// coefficients as observations the likelihood is unbounded this way (the
/// MLE does not exist), whatever the optimiser does.
pub(crate) fn degenerate_observations(data: &RegressionData, b: &DMatrix<f64>) -> Vec<usize> {
    let (d, bd, pt) = (data.d, data.b_dim, data.p_total);
    let mut eta = vec![0.0; bd];
    let mut g = vec![0.0; bd];
    let ll: Vec<f64> = (0..data.n)
        .map(|i| {
            for (k, e) in eta.iter_mut().enumerate() {
                *e = (0..pt).map(|j| b[(k, j)] * data.x[i * pt + j]).sum();
            }
            kernel_grad(data, &eta, &data.y[i * d..(i + 1) * d], &mut g)
        })
        .collect();
    let mut sorted = ll.clone();
    let mid = sorted.len() / 2;
    let (_, med, _) = sorted.select_nth_unstable_by(mid, f64::total_cmp);
    let cut = *med + DEGENERATE_EXCESS;
    ll.iter()
        .enumerate()
        .filter(|&(_, v)| !(*v <= cut))
        .map(|(i, _)| i)
        .collect()
}

/// Mean per-observation gradient (infinity norm) below which an optimiser
/// stop counts as convergence. Guards against "function converged" stops at
/// singular points of the parameterization (coinciding eigenvalues), where
/// the likelihood is not differentiable.
const MAX_MEAN_GRADIENT: f64 = 1e-3;

/// Validated, flattened training data.
#[derive(Clone)]
pub(crate) struct RegressionData {
    pub n: usize,
    pub d: usize,
    /// number of covariates (excluding intercept)
    pub p: usize,
    /// number of design columns (incl. intercept)
    pub p_total: usize,
    pub b_dim: usize,
    /// n × p_total row-major design matrix
    pub x: Vec<f64>,
    /// n × d row-major unit vectors
    pub y: Vec<f64>,
    /// Log-tangent frame when the shape rows are log-tangent coordinates
    /// (`None`: gamma).
    pub lt: Option<LtFrame>,
    /// The transported frame of [`Shape::GammaTransported`] (`None`: gamma
    /// in `onb(mu)`, or log-tangent).
    pub gt: Option<LtFrame>,
    /// Observation weights of the log-likelihood (`None`: all 1), for the
    /// weight homotopy of [`EsagRegressor::homotopy_rounds`].
    pub w: Option<Vec<f64>>,
}

impl RegressionData {
    /// Whether the likelihood's cliffs and traps are handled as such
    /// (restart waves skipped at trapped fits, empirical-information
    /// metric for refits): gamma shape, `d ≥ 5` and covariates. At `d = 4`
    /// group 2's trailing pair can vanish too, but traps are rarer there
    /// and the restart waves still rescue fits (one of 12 on the hard
    /// scenario at n = 2000), so `d = 4` keeps them.
    pub(crate) fn gamma_cliffs(&self) -> bool {
        self.lt.is_none() && self.d >= 5 && self.p > 0
    }

    pub(crate) fn new(x: &DMatrix<f64>, y: &DMatrix<f64>, intercept: bool) -> Result<Self> {
        let n = y.nrows();
        let d = y.ncols();
        if d < 3 {
            return Err(EsagError::Dimension(format!(
                "Y must have d >= 3 columns, got {d}"
            )));
        }
        if n == 0 {
            return Err(EsagError::Data("no observations".into()));
        }
        if x.nrows() != n {
            return Err(EsagError::Shape(format!(
                "X has {} rows but Y has {n}",
                x.nrows()
            )));
        }
        let p = x.ncols();
        let p_total = p + usize::from(intercept);
        if p_total == 0 {
            return Err(EsagError::Shape(
                "no design columns: X has 0 columns and fit_intercept is false".into(),
            ));
        }
        let mut xf = Vec::with_capacity(n * p_total);
        let mut yf = Vec::with_capacity(n * d);
        for i in 0..n {
            if intercept {
                xf.push(1.0);
            }
            for j in 0..p {
                let v = x[(i, j)];
                if !v.is_finite() {
                    return Err(EsagError::Data(format!("X[{i},{j}] is not finite")));
                }
                xf.push(v);
            }
            let mut norm = 0.0;
            for k in 0..d {
                let v = y[(i, k)];
                if !v.is_finite() {
                    return Err(EsagError::Data(format!("Y[{i},{k}] is not finite")));
                }
                norm += v * v;
            }
            let norm = norm.sqrt();
            if norm < 1e-12 {
                return Err(EsagError::Data(format!("Y row {i} is the zero vector")));
            }
            for k in 0..d {
                yf.push(y[(i, k)] / norm);
            }
        }
        Ok(Self {
            n,
            d,
            p,
            p_total,
            b_dim: dims::eta_len(d),
            x: xf,
            y: yf,
            lt: None,
            gt: None,
            w: None,
        })
    }
}

/// Negative joint log-likelihood as a function of the free coefficients.
struct NegLogLik<'a> {
    data: &'a RegressionData,
    /// full coefficient matrix providing the fixed (masked-out) entries
    base: DMatrix<f64>,
    free: &'a [(usize, usize)],
    parallel: bool,
    /// Trap monitor: number of consecutive evaluations at which one
    /// observation's gradient exceeded [`TRAP_RATIO`] times the median (an
    /// observation pinned at a discontinuity of the gamma parameterization).
    /// `None` disables the monitor.
    trap_streak: Option<AtomicUsize>,
    /// Every observation has the same log-tangent shape (see
    /// [`shared_shape`]): one eigen-decomposition per evaluation.
    shared: bool,
}

/// The objective in whitened coordinates `θ = θ₀ + L z` (`L` lower
/// triangular): gradient `Lᵀ ∇_θ f`.
struct Whitened<'a> {
    inner: &'a NegLogLik<'a>,
    theta0: Vec<f64>,
    l: &'a DMatrix<f64>,
}

impl Whitened<'_> {
    fn theta(&self, z: &[f64]) -> Vec<f64> {
        let n = z.len();
        (0..n)
            .map(|i| self.theta0[i] + (0..=i).map(|j| self.l[(i, j)] * z[j]).sum::<f64>())
            .collect()
    }
}

impl Objective for Whitened<'_> {
    fn value_and_gradient(&self, z: &[f64], grad: &mut [f64]) -> f64 {
        let theta = self.theta(z);
        let mut gt = vec![0.0; z.len()];
        let f = self.inner.value_and_gradient(&theta, &mut gt);
        let n = z.len();
        for j in 0..n {
            grad[j] = (j..n).map(|i| self.l[(i, j)] * gt[i]).sum();
        }
        f
    }

    fn should_stop(&self) -> bool {
        self.inner.should_stop()
    }
}

/// [`NegLogLik`] on the affine subspace `C x = c` of the free coefficients,
/// for [`EsagRegressor::stratum_maximum`]: the gradient projected on the
/// null space of `C`. From a feasible start L-BFGS then stays feasible (its
/// steps combine projected gradients and earlier steps).
struct Projected<'a> {
    inner: &'a NegLogLik<'a>,
    /// `Cᵀ` (free coefficients × constraints)
    ct: DMatrix<f64>,
    /// `(C Cᵀ)⁻¹ C`
    pinv: DMatrix<f64>,
}

impl Objective for Projected<'_> {
    fn value_and_gradient(&self, x: &[f64], grad: &mut [f64]) -> f64 {
        let f = self.inner.value_and_gradient(x, grad);
        let along = &self.ct * (&self.pinv * DVector::from_column_slice(grad));
        for (g, a) in grad.iter_mut().zip(along.iter()) {
            *g -= a;
        }
        f
    }
}

/// [`NegLogLik`] in blown-up coordinates around the singular sets of `k`
/// observations ([`EsagRegressor::stratum_maximum`]). Their vanishing pairs
/// `p_i = C_i x + f_i` (the trailing pair of a gamma group, or `μ₀, μ₁` at
/// the pole of `onb(μ)`) become `(ρ_i² + ε_i)(cos θ_i, sin θ_i)`, the rest
/// of `x` its part `x_⊥` in the null space of `C`:
/// `x = x_⊥ + C⁺ (p(ρ, θ) − f)`. In `(x_⊥, ρ, θ)` the likelihood is smooth
/// up to the singular set: an observation's density there depends on the
/// angle only, and `ρ = 0` is an ordinary critical point, a maximum where
/// the observation is drawn to the set (a trap), a minimum where its
/// maximum is at `ρ > 0` (a regular root next to the set). The angular
/// curvature, `O(1/r²)` in `x`, is `O(1)` in `θ`.
struct BlownUp<'a> {
    inner: &'a NegLogLik<'a>,
    nf: usize,
    /// `C⁺ = Cᵀ (C Cᵀ)⁻¹` (free coefficients × 2k)
    cplus: DMatrix<f64>,
    /// `C`
    c: DMatrix<f64>,
    /// the fixed coefficients' part `f` of each pair coordinate
    fixed: DVector<f64>,
    /// the radius `ε_i` at `ρ_i = 0`, so that the angle is defined there
    eps: Vec<f64>,
}

impl BlownUp<'_> {
    /// The pairs at `(ρ, θ)`.
    fn pairs(&self, y: &[f64]) -> DVector<f64> {
        let (nf, k) = (self.nf, self.eps.len());
        let mut p = DVector::zeros(2 * k);
        for i in 0..k {
            let (rho, th) = (y[nf + i], y[nf + k + i]);
            let r = rho * rho + self.eps[i];
            p[2 * i] = r * th.cos();
            p[2 * i + 1] = r * th.sin();
        }
        p
    }

    /// The free coefficients at `y = (x_⊥, ρ, θ)`.
    fn x(&self, y: &[f64]) -> DVector<f64> {
        DVector::from_column_slice(&y[..self.nf]) + &self.cplus * (self.pairs(y) - &self.fixed)
    }

    /// The blown-up coordinates of the free coefficients `x`.
    fn y(&self, x: &DVector<f64>) -> Vec<f64> {
        let k = self.eps.len();
        let p = &self.c * x + &self.fixed;
        let perp = x - &self.cplus * (&self.c * x);
        let mut y: Vec<f64> = perp.as_slice().to_vec();
        y.extend((0..k).map(|i| {
            let r = (p[2 * i].powi(2) + p[2 * i + 1].powi(2)).sqrt();
            (r - self.eps[i]).max(0.0).sqrt()
        }));
        y.extend((0..k).map(|i| p[2 * i + 1].atan2(p[2 * i])));
        y
    }

    /// The radius `r_i − ε_i = ρ_i²` and the radial derivative `∂f/∂r_i`
    /// of the objective at `y`, per observation.
    fn radial(&self, y: &[f64]) -> Vec<(f64, f64)> {
        let (nf, k) = (self.nf, self.eps.len());
        let x = self.x(y);
        let mut gx = vec![0.0; nf];
        self.inner.value_and_gradient(x.as_slice(), &mut gx);
        let gp = self.cplus.transpose() * DVector::from_column_slice(&gx);
        (0..k)
            .map(|i| {
                let (rho, th) = (y[nf + i], y[nf + k + i]);
                (rho * rho, gp[2 * i] * th.cos() + gp[2 * i + 1] * th.sin())
            })
            .collect()
    }
}

impl Objective for BlownUp<'_> {
    fn value_and_gradient(&self, y: &[f64], grad: &mut [f64]) -> f64 {
        let (nf, k) = (self.nf, self.eps.len());
        let x = self.x(y);
        let mut gx = vec![0.0; nf];
        let f = self.inner.value_and_gradient(x.as_slice(), &mut gx);
        let gx = DVector::from_column_slice(&gx);
        // ∂f/∂p = C⁺ᵀ ∇f; ∂f/∂x_⊥ = (I − C⁺C)ᵀ ∇f = ∇f − Cᵀ C⁺ᵀ ∇f
        let gp = self.cplus.transpose() * &gx;
        let perp = &gx - self.c.transpose() * &gp;
        grad[..nf].copy_from_slice(perp.as_slice());
        for i in 0..k {
            let (rho, th) = (y[nf + i], y[nf + k + i]);
            let r = rho * rho + self.eps[i];
            let (c, s) = (th.cos(), th.sin());
            // p_i = r (c, s): ∂p/∂ρ = 2ρ (c, s), ∂p/∂θ = r (−s, c)
            grad[nf + i] = 2.0 * rho * (gp[2 * i] * c + gp[2 * i + 1] * s);
            grad[nf + k + i] = r * (gp[2 * i + 1] * c - gp[2 * i] * s);
        }
        f
    }
}

impl EsagRegressor {
    /// The refit preconditioner at coefficients `b`: the Cholesky factor of
    /// the inverse observed information over the free entries of `mask`, if
    /// the information is positive definite there.
    pub(crate) fn precondition_at(
        &self,
        x: &DMatrix<f64>,
        y: &DMatrix<f64>,
        b: &DMatrix<f64>,
        mask: &DMatrix<bool>,
    ) -> Option<Arc<DMatrix<f64>>> {
        let data = self.data(x, y).ok()?;
        info_precondition(&data, b, &free_params(mask), true)
    }
}

/// The Cholesky factor of the inverse observed information at `b` over
/// `free`, if the information is positive definite there.
fn info_precondition(
    data: &RegressionData,
    b: &DMatrix<f64>,
    free: &[(usize, usize)],
    parallel: bool,
) -> Option<Arc<DMatrix<f64>>> {
    let (_, _, h) = joint_loglik_grad_hess(data, b, parallel);
    let (bd, full) = (data.b_dim, data.b_dim * data.p_total);
    let nf = free.len();
    let info = DMatrix::from_fn(nf, nf, |a, c| {
        let (ka, ja) = free[a];
        let (kc, jc) = free[c];
        -h[(ja * bd + ka) * full + jc * bd + kc]
    });
    let cov = info.cholesky()?.inverse();
    precondition_from_cov(&cov)
}

/// The refit preconditioner from the empirical information
/// `Σ_i s_i s_iᵀ` at `b`, with `s_i` the score of observation `i` over the
/// free coefficients (`(k, j)` entry `∂ℓ_i/∂η_k · x̃_ij`), if it is positive
/// definite.
fn empirical_precondition(
    data: &RegressionData,
    b: &DMatrix<f64>,
    free: &[(usize, usize)],
) -> Option<Arc<DMatrix<f64>>> {
    let (info, _) = score_info(data, b, free);
    precondition_from_cov(&info.cholesky()?.inverse())
}

/// Index of the pair `(a, b)`, `a ≤ b < n`, in the order
/// `(0, 0), (0, 1), …, (0, n−1), (1, 1), …`.
#[inline]
fn pair_index(a: usize, b: usize, n: usize) -> usize {
    let (lo, hi) = (a.min(b), a.max(b));
    lo * n - lo * (lo + 1) / 2 + hi
}

/// The empirical information `Σ_i s_i s_iᵀ` and the score `Σ_i s_i` over
/// the coefficients `free` at `b`, with `s_i` the score of observation `i`
/// (entry `(k, j)`: `∂ℓ_i/∂η_k · x̃_ij`). As `s_i s_iᵀ = (x̃_i x̃_iᵀ) ⊗
/// (g_i g_iᵀ)`, every entry is one of the `pt(pt+1)/2 × bd(bd+1)/2` sums
/// `Σ_i x̃_ij x̃_ij' g_ik g_ik'`, which come out of one matrix product per
/// chunk of observations (as in [`joint_loglik_grad_hess`]): at d = 8 with
/// 6 design columns, 1.7× fewer operations than the Gram matrix of the
/// `s_i`, in one large product.
fn score_info(
    data: &RegressionData,
    b: &DMatrix<f64>,
    free: &[(usize, usize)],
) -> (DMatrix<f64>, DVector<f64>) {
    const CHUNK: usize = 256;
    let (d, bd, pt) = (data.d, data.b_dim, data.p_total);
    let (nx, nh) = (pt * (pt + 1) / 2, bd * (bd + 1) / 2);
    // c[(η pair, x pair)]; per observation the η products are one
    // contiguous column of `ggt`
    let mut c = DMatrix::<f64>::zeros(nh, nx);
    let mut gsum = DMatrix::<f64>::zeros(bd, pt);
    for start in (0..data.n).step_by(CHUNK) {
        let end = (start + CHUNK).min(data.n);
        let m = end - start;
        let xt = nalgebra::DMatrixView::from_slice(&data.x[start * pt..end * pt], pt, m);
        let eta = b * xt;
        let mut ge = DMatrix::<f64>::zeros(bd, m);
        let mut ggt = DMatrix::<f64>::zeros(nh, m);
        let mut w = DMatrix::<f64>::zeros(m, nx);
        for r in 0..m {
            let i = start + r;
            let gr = &mut ge.as_mut_slice()[r * bd..(r + 1) * bd];
            kernel_grad(
                data,
                &eta.as_slice()[r * bd..(r + 1) * bd],
                &data.y[i * d..(i + 1) * d],
                gr,
            );
            let col = &mut ggt.as_mut_slice()[r * nh..(r + 1) * nh];
            let mut q = 0;
            for k1 in 0..bd {
                let g1 = gr[k1];
                for &g2 in &gr[k1..] {
                    col[q] = g1 * g2;
                    q += 1;
                }
            }
            let xi = &data.x[i * pt..(i + 1) * pt];
            let mut a = 0;
            for j1 in 0..pt {
                for j2 in j1..pt {
                    w[(r, a)] = xi[j1] * xi[j2];
                    a += 1;
                }
            }
        }
        c.gemm(1.0, &ggt, &w, 1.0);
        gsum.gemm(1.0, &ge, &xt.transpose(), 1.0);
    }
    let nf = free.len();
    let info = DMatrix::from_fn(nf, nf, |a, e| {
        let ((ka, ja), (ke, je)) = (free[a], free[e]);
        c[(pair_index(ka, ke, bd), pair_index(ja, je, pt))]
    });
    let g = DVector::from_iterator(nf, free.iter().map(|&(k, j)| gsum[(k, j)]));
    (info, g)
}

/// Lower Cholesky factor of a covariance matrix (the preconditioner for
/// refits), if it is positive definite.
pub(crate) fn precondition_from_cov(cov: &DMatrix<f64>) -> Option<Arc<DMatrix<f64>>> {
    let sym = (cov + cov.transpose()) * 0.5;
    sym.cholesky().map(|c| Arc::new(c.l()))
}

/// Per-observation gradient ratio (largest over median, in `eta`) above
/// which an evaluation counts towards a trap. Healthy fits stay below ~10²;
/// a start pinned at a singular point of the gamma parameterization reaches
/// 10⁴–10¹² and does not recover.
const TRAP_RATIO: f64 = 1e5;
/// Stopping rule of whitened (bootstrap) refits: the predicted remaining
/// log-likelihood gain `½‖g_z‖²` (whitened with the inverse information).
/// At 1e-6 every coefficient is within ~0.0014 standard errors of the
/// maximum and the log-likelihood within 1e-6, far below bootstrap noise.
const REFIT_DECREMENT: f64 = 1e-6;
/// The same rule for whitened fits whose result `finalize` polishes with
/// Newton steps on the exact Hessian (fits with standard errors).
const HANDOFF_DECREMENT: f64 = 1e-5;
/// Consecutive trapped evaluations after which a start is abandoned.
const TRAP_STREAK: usize = 25;

impl NegLogLik<'_> {
    fn assemble(&self, params: &[f64]) -> DMatrix<f64> {
        let mut b = self.base.clone();
        for (&(k, j), &v) in self.free.iter().zip(params) {
            b[(k, j)] = v;
        }
        b
    }
}

/// The negative joint log-likelihood over the free coefficients with its
/// exact Hessian, as a [`crate::optim::prox::SmoothObjective`] for penalized
/// (LASSO-type) fits.
pub(crate) struct SmoothNegLogLik<'a> {
    inner: NegLogLik<'a>,
}

impl<'a> SmoothNegLogLik<'a> {
    pub(crate) fn new(
        data: &'a RegressionData,
        base: DMatrix<f64>,
        free: &'a [(usize, usize)],
        parallel: bool,
    ) -> Self {
        Self {
            inner: NegLogLik {
                data,
                shared: shared_shape(data, &base, free),
                base,
                free,
                parallel,
                trap_streak: None,
            },
        }
    }
}

impl SmoothNegLogLik<'_> {
    /// Number of observations.
    pub(crate) fn n_obs(&self) -> usize {
        self.inner.data.n
    }
}

impl crate::optim::prox::SmoothObjective for SmoothNegLogLik<'_> {
    fn value_grad(&self, theta: &[f64], grad: &mut [f64]) -> f64 {
        self.inner.value_and_gradient(theta, grad)
    }

    fn value_grad_hess(&self, theta: &[f64], grad: &mut [f64], hess: &mut [f64]) -> f64 {
        let data = self.inner.data;
        let b = self.inner.assemble(theta);
        let (f, g, h) = joint_loglik_grad_hess(data, &b, self.inner.parallel);
        let (bd, full) = (data.b_dim, data.b_dim * data.p_total);
        let free = self.inner.free;
        let nf = free.len();
        for (a, &(ka, ja)) in free.iter().enumerate() {
            let ia = ja * bd + ka;
            grad[a] = -g[ia];
            for (c, &(kc, jc)) in free.iter().enumerate() {
                hess[a * nf + c] = -h[ia * full + jc * bd + kc];
            }
        }
        -f
    }
}

/// Minimum number of observations per rayon chunk.
const PAR_CHUNK: usize = 64;

impl Objective for NegLogLik<'_> {
    fn value_and_gradient(&self, params: &[f64], grad: &mut [f64]) -> f64 {
        let b = self.assemble(params);
        let (f, g_full, ratio) = joint_eval(
            self.data,
            &b,
            self.parallel,
            self.trap_streak.is_some(),
            self.shared,
        );
        for (gi, &(k, j)) in grad.iter_mut().zip(self.free) {
            *gi = -g_full[(k, j)];
        }
        if let Some(streak) = &self.trap_streak {
            if ratio > TRAP_RATIO {
                streak.fetch_add(1, Ordering::Relaxed);
            } else {
                streak.store(0, Ordering::Relaxed);
            }
        }
        -f
    }

    fn should_stop(&self) -> bool {
        self.trap_streak
            .as_ref()
            .is_some_and(|s| s.load(Ordering::Relaxed) >= TRAP_STREAK)
    }
}

/// Joint log-likelihood `Σ_i log f(y_i; B x̃_i)` and its gradient with respect
/// to the full coefficient matrix `B` (`b_dim × p_total`).
pub(crate) fn joint_loglik_and_grad(
    data: &RegressionData,
    b: &DMatrix<f64>,
    parallel: bool,
) -> (f64, DMatrix<f64>) {
    let (f, g, _) = joint_eval(data, b, parallel, false, false);
    (f, g)
}

/// Whether every observation of a log-tangent fit has the same shape `S`,
/// whatever the free coefficients: an intercept, and every shape
/// coefficient of the other columns fixed (not free) at zero. The gradient
/// with respect to those fixed coefficients is then not needed, and
/// [`joint_eval`] diagonalises `S` once per evaluation
/// ([`crate::logtangent::SharedShape`]).
fn shared_shape(data: &RegressionData, base: &DMatrix<f64>, free: &[(usize, usize)]) -> bool {
    let (d, pt) = (data.d, data.p_total);
    data.lt.is_some()
        && pt > data.p
        && data.p > 0
        && !free.iter().any(|&(k, j)| k >= d && j >= 1)
        && (d..data.b_dim).all(|k| (1..pt).all(|j| base[(k, j)] == 0.0))
}

/// Scratch of one chunk of [`joint_eval`] (at most [`PAR_CHUNK`]
/// observations): `η` and `∇_η ℓ` (`bd × m`, one column per observation),
/// the chunk's gradient (`bd × pt`) and the shared shape's accumulator.
struct ChunkBuf {
    eta: Vec<f64>,
    ge: Vec<f64>,
    gm: DMatrix<f64>,
    wacc: Vec<f64>,
}

impl ChunkBuf {
    fn new(bd: usize, pt: usize, acc_len: usize) -> Self {
        Self {
            eta: vec![0.0; bd * PAR_CHUNK],
            ge: vec![0.0; bd * PAR_CHUNK],
            gm: DMatrix::zeros(bd, pt),
            wacc: vec![0.0; acc_len],
        }
    }
}

/// [`joint_loglik_and_grad`], plus (if `ratio` is set) the largest
/// per-observation gradient norm `‖∇_η ℓ_i‖` over the median one (else 0).
/// With `shared` (see [`shared_shape`]) the gradient with respect to the
/// shape coefficients of columns `j ≥ 1` is not computed (left at 0).
fn joint_eval(
    data: &RegressionData,
    b: &DMatrix<f64>,
    parallel: bool,
    ratio: bool,
    shared: bool,
) -> (f64, DMatrix<f64>, f64) {
    let (d, bd, pt) = (data.d, data.b_dim, data.p_total);
    // one eigen-decomposition of the common shape (the intercept column)
    let sh = match &data.lt {
        Some(_) if shared && data.w.is_none() => {
            let s: Vec<f64> = (d..bd).map(|k| b[(k, 0)]).collect();
            Some(crate::logtangent::SharedShape::new(&s, d - 1))
        }
        _ => None,
    };
    let acc_len = sh.as_ref().map_or(0, |s| s.acc_len());
    // B column-major: column j (the coefficients of design column j) is
    // contiguous
    let bs = b.as_slice();
    // Per chunk of observations: eta = B X_cᵀ (bd × m, one contiguous
    // column per observation) and the gradient Σ_i g_i x̃_iᵀ = G X_c, both
    // as matrix products into the chunk's buffers (allocating them per
    // evaluation, for all n observations at once, cost as much as the
    // products themselves at d = 3). Returns the chunk's log-likelihood; its
    // gradient and accumulator are left in `buf`, its squared gradient norms
    // are appended to `norms` (if `ratio`).
    let body = |rows: std::ops::Range<usize>, buf: &mut ChunkBuf, norms: &mut Vec<f64>| -> f64 {
        let m = rows.len();
        // the row-major m × pt block of the design is X_cᵀ (pt × m,
        // column-major) and, with strides (pt, 1), X_c (m × pt)
        let blk = &data.x[rows.start * pt..rows.end * pt];
        let xt = nalgebra::DMatrixView::from_slice(blk, pt, m);
        let xc = nalgebra::DMatrixView::from_slice_with_strides(blk, m, pt, pt, 1);
        let ChunkBuf { eta, ge, gm, wacc } = buf;
        let (eta, ge) = (&mut eta[..bd * m], &mut ge[..bd * m]);
        // nalgebra multiplies through `matrixmultiply` only when every
        // dimension exceeds 5, and column by column otherwise (always at
        // d = 3, where bd = 5); `small_eta` and `small_grad` do the latter
        // faster, with the same operations in the same order
        let small = !(bd > 5 && pt > 5 && m > 5);
        if small {
            small_eta(eta, bs, blk, bd, pt);
        } else {
            nalgebra::DMatrixViewMut::from_slice(eta, bd, m).gemm(1.0, b, &xt, 0.0);
        }
        wacc.fill(0.0);
        if sh.is_some() {
            // the shared-shape kernel writes the mu part only
            ge.fill(0.0);
        }
        let mut f = 0.0;
        for (r, i) in rows.enumerate() {
            let yi = &data.y[i * d..(i + 1) * d];
            let gcol = &mut ge[r * bd..(r + 1) * bd];
            let ei = &eta[r * bd..(r + 1) * bd];
            let fi = match (&sh, &data.lt) {
                (Some(sh), Some(fr)) => crate::logtangent::loglik_lt_grad_shared(
                    fr,
                    sh,
                    &ei[..d],
                    yi,
                    &mut gcol[..d],
                    wacc,
                ),
                _ => kernel_grad(data, ei, yi, gcol),
            };
            f += match &data.w {
                // weights are used without the shared-shape path only
                Some(w) if w[i] == 0.0 => {
                    gcol.fill(0.0);
                    0.0
                }
                Some(w) => {
                    gcol.iter_mut().for_each(|v| *v *= w[i]);
                    w[i] * fi
                }
                None => fi,
            };
            if ratio {
                norms.push(gcol.iter().map(|v| v * v).sum::<f64>());
            }
        }
        if small {
            small_grad(gm.as_mut_slice(), ge, blk, bd, pt);
        } else {
            gm.gemm(1.0, &nalgebra::DMatrixView::from_slice(ge, bd, m), &xc, 0.0);
        }
        f
    };
    // Fixed chunks summed in chunk order (not with rayon's `reduce`, whose
    // combination tree depends on thread timing), with or without
    // `parallel`: the result is bit-for-bit reproducible and does not
    // depend on the number of threads.
    let n_chunks = data.n.div_ceil(PAR_CHUNK);
    let range = |c: usize| c * PAR_CHUNK..((c + 1) * PAR_CHUNK).min(data.n);
    let mut f = 0.0;
    let mut g = vec![0.0; bd * pt];
    let mut norms = Vec::with_capacity(if ratio { data.n } else { 0 });
    let mut wacc = vec![0.0; acc_len];
    let mut add = |fc: f64, buf: &ChunkBuf| {
        f += fc;
        for (a, b) in g.iter_mut().zip(buf.gm.as_slice()) {
            *a += b;
        }
        for (a, b) in wacc.iter_mut().zip(&buf.wacc) {
            *a += b;
        }
    };
    if parallel && data.n >= 2 * PAR_CHUNK {
        let parts: Vec<(f64, ChunkBuf, Vec<f64>)> = (0..n_chunks)
            .into_par_iter()
            .map(|c| {
                let mut buf = ChunkBuf::new(bd, pt, acc_len);
                let mut nc = Vec::with_capacity(if ratio { PAR_CHUNK } else { 0 });
                let fc = body(range(c), &mut buf, &mut nc);
                (fc, buf, nc)
            })
            .collect();
        for (fc, buf, nc) in parts {
            add(fc, &buf);
            norms.extend(nc);
        }
    } else {
        let mut buf = ChunkBuf::new(bd, pt, acc_len);
        for c in 0..n_chunks {
            let fc = body(range(c), &mut buf, &mut norms);
            add(fc, &buf);
        }
    }
    if let Some(sh) = &sh {
        // the shape rows of the intercept column (all others are fixed)
        let mut gs = vec![0.0; bd - d];
        sh.shape_gradient(&wacc, &mut gs);
        g[d..bd].copy_from_slice(&gs);
    }
    let r = if ratio && !norms.is_empty() {
        let max = norms.iter().fold(0.0f64, |m, &v| m.max(v));
        let mid = norms.len() / 2;
        let (_, med, _) = norms.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
        // norms are squared
        (max / med.max(f64::MIN_POSITIVE)).sqrt()
    } else {
        0.0
    };
    (f, DMatrix::from_column_slice(bd, pt, &g), r)
}

/// `η = B X_cᵀ` for one chunk (`eta`: `bd × m` column-major, `bs`: `B`
/// column-major, `blk`: the chunk's row-major `m × pt` design block), in the
/// order of nalgebra's column-by-column product for small shapes:
/// `η_kr = B_k0 x_r0`, then `η_kr = B_kj x_rj + η_kr` for `j = 1..pt`
/// (bit-identical to it, and 2.2× faster at d = 3 with 5 covariates).
fn small_eta(eta: &mut [f64], bs: &[f64], blk: &[f64], bd: usize, pt: usize) {
    match bd {
        5 => small_eta_fixed::<5>(eta, bs, blk, pt),
        9 => small_eta_fixed::<9>(eta, bs, blk, pt),
        14 => small_eta_fixed::<14>(eta, bs, blk, pt),
        20 => small_eta_fixed::<20>(eta, bs, blk, pt),
        27 => small_eta_fixed::<27>(eta, bs, blk, pt),
        35 => small_eta_fixed::<35>(eta, bs, blk, pt),
        _ => {
            for (out, xr) in eta.chunks_exact_mut(bd).zip(blk.chunks_exact(pt)) {
                for (o, &v) in out.iter_mut().zip(&bs[..bd]) {
                    *o = v * xr[0];
                }
                for (j, &xj) in xr.iter().enumerate().skip(1) {
                    for (o, &v) in out.iter_mut().zip(&bs[j * bd..(j + 1) * bd]) {
                        *o += v * xj;
                    }
                }
            }
        }
    }
}

#[inline(always)]
fn small_eta_fixed<const BD: usize>(eta: &mut [f64], bs: &[f64], blk: &[f64], pt: usize) {
    for (out, xr) in eta.chunks_exact_mut(BD).zip(blk.chunks_exact(pt)) {
        let mut acc: [f64; BD] = core::array::from_fn(|k| bs[k] * xr[0]);
        for (j, &xj) in xr.iter().enumerate().skip(1) {
            let col = &bs[j * BD..(j + 1) * BD];
            for k in 0..BD {
                acc[k] += col[k] * xj;
            }
        }
        out.copy_from_slice(&acc);
    }
}

/// The chunk's gradient `G = Σ_r ∇_η ℓ_r x̃_rᵀ` (`gm`: `bd × pt`
/// column-major, `ge`: `bd × m` column-major), in the order of nalgebra's
/// column-by-column product for small shapes: `G_kj = ge_k0 x_0j`, then
/// `G_kj = ge_kr x_rj + G_kj` for `r = 1..m` (bit-identical to it).
fn small_grad(gm: &mut [f64], ge: &[f64], blk: &[f64], bd: usize, pt: usize) {
    match bd {
        5 => small_grad_fixed::<5>(gm, ge, blk, pt),
        9 => small_grad_fixed::<9>(gm, ge, blk, pt),
        14 => small_grad_fixed::<14>(gm, ge, blk, pt),
        20 => small_grad_fixed::<20>(gm, ge, blk, pt),
        27 => small_grad_fixed::<27>(gm, ge, blk, pt),
        35 => small_grad_fixed::<35>(gm, ge, blk, pt),
        _ => {
            for j in 0..pt {
                for (o, &g) in gm[j * bd..(j + 1) * bd].iter_mut().zip(&ge[..bd]) {
                    *o = g * blk[j];
                }
            }
            for (gr, xr) in ge.chunks_exact(bd).zip(blk.chunks_exact(pt)).skip(1) {
                for (j, &xj) in xr.iter().enumerate() {
                    for (o, &g) in gm[j * bd..(j + 1) * bd].iter_mut().zip(gr) {
                        *o += g * xj;
                    }
                }
            }
        }
    }
}

#[inline(always)]
fn small_grad_fixed<const BD: usize>(gm: &mut [f64], ge: &[f64], blk: &[f64], pt: usize) {
    for (j, out) in gm.chunks_exact_mut(BD).enumerate().take(pt) {
        let x0 = blk[j];
        let mut acc: [f64; BD] = core::array::from_fn(|k| ge[k] * x0);
        for (gr, xr) in ge.chunks_exact(BD).zip(blk.chunks_exact(pt)).skip(1) {
            let xj = xr[j];
            for k in 0..BD {
                acc[k] += gr[k] * xj;
            }
        }
        out.copy_from_slice(&acc);
    }
}

/// Per-observation log-density and gradient in `eta` for the data's shape
/// parameterization.
#[inline]
fn kernel_grad(data: &RegressionData, eta: &[f64], y: &[f64], g: &mut [f64]) -> f64 {
    match (&data.lt, &data.gt) {
        (Some(fr), _) => crate::logtangent::loglik_lt_grad(fr, eta, y, g),
        (None, Some(fr)) => crate::gamma_transported::loglik_gt_grad(fr, eta, y, g),
        (None, None) => loglik::loglik_eta_grad(eta, y, g),
    }
}

/// How [`joint_loglik_grad_hess`] computes the per-observation Hessians of
/// γ.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum ObsHessian {
    /// Exact ([`crate::adjoint_hess`]): 0.09, 0.22, 0.42 and 2.1 µs per
    /// observation at d = 3, 4, 5 and 8.
    Exact,
    /// Central differences of the exact gradient for `d ≥ 4` (the exact
    /// Hessian at d = 3): 1.5, 2.7 and 11 µs at d = 4, 5 and 8. They are
    /// what fits used before the exact Hessian existed, and `finalize` keeps
    /// them for fits that did not converge: such fits stop next to kinks
    /// of γ, where differences over a small step (which average over the
    /// kink) give the polishing Newton steps a better model than the
    /// pointwise Hessian (measured on unconverged d = 4 and 5 fits).
    Differences,
}

/// Per-observation value, gradient and Hessian in `eta` for the data's
/// shape parameterization.
#[inline]
fn obs_hessian(
    data: &RegressionData,
    kind: ObsHessian,
    eta: &[f64],
    y: &[f64],
    g: &mut [f64],
    h: &mut [f64],
) -> f64 {
    if let Some(fr) = &data.lt {
        // exact (one eigen-decomposition; ~2.7× the gradient at d = 8)
        return crate::logtangent::loglik_lt_hess(fr, eta, y, g, h);
    }
    if let Some(fr) = &data.gt {
        return crate::gamma_transported::loglik_gt_hess_fd(fr, eta, y, g, h);
    }
    if kind == ObsHessian::Differences && data.d >= 4 {
        return crate::hdual::loglik_eta_hess_fd(eta, y, g, h);
    }
    crate::adjoint_hess::loglik_hess_adjoint(eta, y, g, h)
}

/// Joint log-likelihood, gradient and Hessian with respect to the full
/// coefficient vector, ordered column-major (`j·b_dim + k` for entry
/// `(k, j)` of `B`). Per observation, `∇²ℓ_i(B) = (x̃_i x̃_iᵀ) ⊗ ∇²_η ℓ_i`.
pub(crate) fn joint_loglik_grad_hess(
    data: &RegressionData,
    b: &DMatrix<f64>,
    parallel: bool,
) -> (f64, Vec<f64>, Vec<f64>) {
    joint_loglik_grad_hess_with(data, b, parallel, ObsHessian::Exact)
}

/// [`joint_loglik_grad_hess`] with the per-observation Hessians of `kind`.
pub(crate) fn joint_loglik_grad_hess_with(
    data: &RegressionData,
    b: &DMatrix<f64>,
    parallel: bool,
    kind: ObsHessian,
) -> (f64, Vec<f64>, Vec<f64>) {
    let (d, bd, pt) = (data.d, data.b_dim, data.p_total);
    let full = bd * pt;
    // Unique products x_{j1} x_{j2} (j1 ≤ j2) and unique entries
    // H_i[k1, k2] (k1 ≤ k2): per chunk of observations, every unique block
    // entry Σ_i x_{ij1} x_{ij2} H_i[k1, k2] comes out of one matrix product
    // Wᵀ Hs (pt(pt+1)/2 × bd(bd+1)/2), which is then scattered into the
    // symmetric full Hessian.
    let xpairs: Vec<(usize, usize)> = (0..pt)
        .flat_map(|j1| (j1..pt).map(move |j2| (j1, j2)))
        .collect();
    let hpairs: Vec<(usize, usize)> = (0..bd)
        .flat_map(|k1| (k1..bd).map(move |k2| (k1, k2)))
        .collect();
    let b_rm: Vec<f64> = (0..bd)
        .flat_map(|k| (0..pt).map(move |j| b[(k, j)]))
        .collect();
    // Per chunk, the products and entries of observation r are column r of
    // `w` and `hs` (contiguous), and the chunk contributes `hs wᵀ`
    // (bd(bd+1)/2 × pt(pt+1)/2).
    let body = |rows: std::ops::Range<usize>| -> (f64, Vec<f64>, DMatrix<f64>) {
        let m = rows.len();
        let mut f = 0.0;
        let mut g = vec![0.0; full];
        let mut hs = DMatrix::<f64>::zeros(hpairs.len(), m);
        let mut w = DMatrix::<f64>::zeros(xpairs.len(), m);
        let mut eta = vec![0.0; bd];
        let mut ge = vec![0.0; bd];
        let mut he = vec![0.0; bd * bd];
        for (r, i) in rows.enumerate() {
            let xi = &data.x[i * pt..(i + 1) * pt];
            let yi = &data.y[i * d..(i + 1) * d];
            for k in 0..bd {
                eta[k] = b_rm[k * pt..(k + 1) * pt]
                    .iter()
                    .zip(xi)
                    .map(|(a, c)| a * c)
                    .sum();
            }
            let wi = data.w.as_ref().map_or(1.0, |w| w[i]);
            if wi == 0.0 {
                // set aside (an observation on a singular set may have huge
                // derivatives; 0·∞ would poison the sums)
                ge.fill(0.0);
                he.fill(0.0);
            } else {
                f += wi * obs_hessian(data, kind, &eta, yi, &mut ge, &mut he);
            }
            for j in 0..pt {
                for k in 0..bd {
                    g[j * bd + k] += wi * ge[k] * xi[j];
                }
            }
            // the upper triangle of H_i, row by row (the order of `hpairs`)
            let col = &mut hs.as_mut_slice()[r * hpairs.len()..(r + 1) * hpairs.len()];
            let mut q = 0;
            for k1 in 0..bd {
                let len = bd - k1;
                col[q..q + len].copy_from_slice(&he[k1 * bd + k1..(k1 + 1) * bd]);
                q += len;
            }
            let col = &mut w.as_mut_slice()[r * xpairs.len()..(r + 1) * xpairs.len()];
            for (v, &(j1, j2)) in col.iter_mut().zip(&xpairs) {
                *v = wi * xi[j1] * xi[j2];
            }
        }
        (f, g, hs * w.transpose())
    };
    // Fixed chunks (at most 64), summed in order: the result does not depend
    // on `parallel` or on the number of threads, and the per-chunk scratch
    // stays small (at d = 8 and n = 2000, one chunk of all observations
    // made the Hessian 1.4× slower on one core).
    let chunk = PAR_CHUNK.max(data.n.div_ceil(64));
    let n_chunks = data.n.div_ceil(chunk);
    let part = |c: usize| body(c * chunk..((c + 1) * chunk).min(data.n));
    let parts: Vec<(f64, Vec<f64>, DMatrix<f64>)> = if parallel && data.n >= 2 * PAR_CHUNK {
        (0..n_chunks).into_par_iter().map(part).collect()
    } else {
        (0..n_chunks).map(part).collect()
    };
    let mut f = 0.0;
    let mut g = vec![0.0; full];
    let mut c = DMatrix::<f64>::zeros(hpairs.len(), xpairs.len());
    for (fc, gc, cc) in parts {
        f += fc;
        for (a, b) in g.iter_mut().zip(&gc) {
            *a += b;
        }
        c += cc;
    }
    let mut h = vec![0.0; full * full];
    for (a, &(j1, j2)) in xpairs.iter().enumerate() {
        for (q, &(k1, k2)) in hpairs.iter().enumerate() {
            let v = c[(q, a)];
            // H is symmetric and each H_i is symmetric: four mirror images
            h[(j1 * bd + k1) * full + j2 * bd + k2] = v;
            h[(j1 * bd + k2) * full + j2 * bd + k1] = v;
            h[(j2 * bd + k1) * full + j1 * bd + k2] = v;
            h[(j2 * bd + k2) * full + j1 * bd + k1] = v;
        }
    }
    (f, g, h)
}

/// The distribution of one linear predictor `eta = (mu, shape)` in the
/// parameterization `shape`. `lt` is the tangent frame of
/// [`Shape::LogTangent`] and [`Shape::GammaTransported`]; [`Shape::Gamma`]
/// (and a missing frame) uses `onb(mu)`.
pub fn shape_distribution(shape: Shape, lt: Option<&LtFrame>, eta: &[f64]) -> Result<Esag> {
    let d = dims::dim_from_eta_len(eta.len())
        .ok_or_else(|| EsagError::Shape(format!("eta of length {} fits no d", eta.len())))?;
    match (shape, lt) {
        (Shape::LogTangent, Some(fr)) => {
            let v = crate::logtangent::covariance(fr, &eta[..d], &eta[d..]);
            Esag::from_covariance(&eta[..d], &v)
        }
        (Shape::GammaTransported, Some(fr)) => {
            let f = Frame::transported(fr, &eta[..d], &eta[d..]);
            let v = DMatrix::from_row_slice(d, d, &f.cov_pow(1.0));
            Esag::from_covariance(&eta[..d], &v)
        }
        _ => Esag::from_eta(eta),
    }
}

/// Residual diagnostics for a fitted model.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Residuals {
    /// `Q_i = r_iᵀ V_i⁻¹ r_i`, `r_i = (I − μ̂_i μ̂_iᵀ) y_i`.
    pub q: Vec<f64>,
    /// `T1_i = (|mu_i|² + Σ lambda_i + 1) Q_i`, approximately χ²_{d−1}.
    pub t1: Vec<f64>,
}

/// A fitted ESAG regression model.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EsagRegressorFit {
    /// Ambient dimension of the response.
    pub d: usize,
    /// Number of covariates (excluding intercept).
    pub n_features: usize,
    /// Whether an intercept column was added.
    pub fit_intercept: bool,
    /// Number of training observations.
    pub n_obs: usize,
    /// Full coefficient matrix `B` (`eta_len(d) × p_total`); column 0 is the
    /// intercept when `fit_intercept`. Rows: `mu_1..mu_d, gamma_1..gamma_g`.
    pub coef: DMatrix<f64>,
    /// Free-parameter mask used in the fit.
    pub mask: DMatrix<bool>,
    /// Row/column indices of the free parameters, in optimiser order.
    pub free: Vec<(usize, usize)>,
    /// Maximised joint log-likelihood.
    pub log_likelihood: f64,
    /// L-BFGS iterations.
    pub n_iter: usize,
    /// Objective evaluations.
    pub n_fev: usize,
    /// Optimiser termination status.
    pub status: Status,
    /// Whether the optimiser reported convergence.
    pub converged: bool,
    /// Infinity norm of the gradient at the solution.
    pub grad_norm: f64,
    /// Observations (row indices) at a singular set of the gamma
    /// parameterization, where `V(γ)` is discontinuous, with a large
    /// gradient: the fit stopped in a trap, a non-regular local supremum,
    /// not at a maximum (see [`singular_observations`]). Empty for
    /// converged fits and for the log-tangent shape.
    #[serde(default)]
    pub singular_obs: Vec<usize>,
    /// Observations (row indices) whose fitted density exceeds the median
    /// observation's by more than 12 nats in a fit that did not converge:
    /// `V(x_i)` is driven towards a singular matrix and the likelihood is
    /// unbounded (see [`degenerate_observations`]). Empty otherwise.
    #[serde(default)]
    pub degenerate_obs: Vec<usize>,
    /// Observations (row indices) set aside: a gamma fit that stayed
    /// trapped with them returned the converged fit of the others (their
    /// number is [`EsagRegressorFit::n_obs`]); the log-likelihood, standard
    /// errors and criteria refer to those. Empty otherwise.
    #[serde(default)]
    pub set_aside_obs: Vec<usize>,
    /// The fit converged to a maximum of the likelihood that is not
    /// regular: the observations in [`EsagRegressorFit::singular_obs`] sit
    /// on a singular set of the gamma parameterization at their best
    /// direction, and no regular maximum continued from the fit without
    /// them (see [`EsagRegressor::stratum_maximum`] and `docs/MATH.md`). Its
    /// standard errors use the information of the other observations.
    #[serde(default)]
    pub non_regular: bool,
    /// Observed information (Hessian of the negative log-likelihood) over the
    /// free parameters, if computed.
    pub hessian: Option<DMatrix<f64>>,
    /// Asymptotic covariance of the free parameters (inverse Hessian).
    pub cov_params: Option<DMatrix<f64>>,
    /// Design column names.
    pub column_names: Vec<String>,
    /// The estimator configuration used.
    pub config: EsagRegressor,
}

impl EsagRegressorFit {
    /// Number of design columns (intercept included).
    pub fn p_total(&self) -> usize {
        self.coef.ncols()
    }
    /// Length of `eta = (mu, gamma)`.
    pub fn b_dim(&self) -> usize {
        self.coef.nrows()
    }
    /// Number of free parameters.
    pub fn n_params(&self) -> usize {
        self.free.len()
    }
    /// Intercept column (`eta` at `x = 0`), if the model has one.
    pub fn intercept(&self) -> Option<DVector<f64>> {
        self.fit_intercept.then(|| self.coef.column(0).into_owned())
    }
    /// Slope coefficients (`b_dim × n_features`).
    pub fn slopes(&self) -> DMatrix<f64> {
        let start = usize::from(self.fit_intercept);
        self.coef.columns(start, self.n_features).into_owned()
    }
    /// Akaike information criterion.
    pub fn aic(&self) -> f64 {
        2.0 * self.n_params() as f64 - 2.0 * self.log_likelihood
    }
    /// Bayesian information criterion.
    pub fn bic(&self) -> f64 {
        (self.n_obs as f64).ln() * self.n_params() as f64 - 2.0 * self.log_likelihood
    }

    /// Row-major design row for one covariate vector.
    fn design_row(&self, x: &DMatrix<f64>, i: usize, out: &mut Vec<f64>) {
        out.clear();
        if self.fit_intercept {
            out.push(1.0);
        }
        for j in 0..x.ncols() {
            out.push(x[(i, j)]);
        }
    }

    fn check_x(&self, x: &DMatrix<f64>) -> Result<()> {
        if x.ncols() != self.n_features {
            return Err(EsagError::Shape(format!(
                "X has {} columns but the model was fitted with {}",
                x.ncols(),
                self.n_features
            )));
        }
        Ok(())
    }

    /// Linear predictor `eta_i = B x̃_i` for every row of `x` (`n × b_dim`).
    pub fn linear_predictor(&self, x: &DMatrix<f64>) -> Result<DMatrix<f64>> {
        self.check_x(x)?;
        let n = x.nrows();
        let bd = self.b_dim();
        let mut out = DMatrix::zeros(n, bd);
        let mut xr = Vec::with_capacity(self.p_total());
        for i in 0..n {
            self.design_row(x, i, &mut xr);
            for k in 0..bd {
                let mut s = 0.0;
                for (j, xv) in xr.iter().enumerate() {
                    s += self.coef[(k, j)] * xv;
                }
                out[(i, k)] = s;
            }
        }
        Ok(out)
    }

    /// Predicted mean directions `mu_i/|mu_i|` (`n × d` unit vectors).
    pub fn predict(&self, x: &DMatrix<f64>) -> Result<DMatrix<f64>> {
        let eta = self.linear_predictor(x)?;
        let d = self.d;
        let mut out = DMatrix::zeros(x.nrows(), d);
        for i in 0..x.nrows() {
            let norm = (0..d).map(|k| eta[(i, k)].powi(2)).sum::<f64>().sqrt();
            for k in 0..d {
                out[(i, k)] = eta[(i, k)] / norm;
            }
        }
        Ok(out)
    }

    /// The fitted conditional distribution `ESAG(mu_i, gamma_i)` for every row.
    pub fn predict_distribution(&self, x: &DMatrix<f64>) -> Result<Vec<Esag>> {
        let eta = self.linear_predictor(x)?;
        let lt = self.lt_frame();
        (0..x.nrows())
            .map(|i| {
                let row: Vec<f64> = eta.row(i).iter().copied().collect();
                self.distribution_at(&row, lt.as_ref())
            })
            .collect()
    }

    /// `V̂(x)` (`d × d`) for each row of `x`: the estimated ESAG covariance
    /// at those covariates. [`crate::inference::covariance_intervals`] adds
    /// bootstrap uncertainty.
    pub fn predict_covariance(&self, x: &DMatrix<f64>) -> Result<Vec<DMatrix<f64>>> {
        Ok(self
            .predict_distribution(x)?
            .iter()
            .map(|d| d.covariance())
            .collect())
    }

    /// The eigen-frame of `V` for one linear predictor `eta` (pass
    /// `self.lt_frame()` for log-tangent fits): cheaper than building an
    /// [`Esag`] for sampling, moments and residuals.
    pub fn frame_at(&self, eta: &[f64], lt: Option<&LtFrame>) -> Frame<f64> {
        let d = self.d;
        match (self.config.shape, lt) {
            (Shape::LogTangent, Some(fr)) => {
                crate::logtangent::eigen_frame(fr, &eta[..d], &eta[d..])
            }
            (Shape::GammaTransported, Some(fr)) => Frame::transported(fr, &eta[..d], &eta[d..]),
            _ => Frame::new(&eta[..d], &eta[d..]),
        }
    }

    /// The tangent frame of a `Shape::LogTangent` or
    /// `Shape::GammaTransported` fit.
    pub fn lt_frame(&self) -> Option<LtFrame> {
        match (self.config.shape, &self.config.lt_reference) {
            (Shape::LogTangent | Shape::GammaTransported, Some(m)) => Some(LtFrame::new(m)),
            _ => None,
        }
    }

    /// Score (Lagrange-multiplier) statistic, at this restricted fit, of the
    /// coefficients in `full` that it holds at zero, in the
    /// empirical-information form: with `s_i` the score of observation `i`
    /// over `full`, `g = Σ s_i` and `I = Σ s_i s_iᵀ`,
    /// `LM = gᵀ I⁻¹ g − g_fᵀ I_ff⁻¹ g_f` (`f`: the coefficients this fit
    /// frees; the second term removes what the optimizer left of their
    /// gradient). It needs no fit of the full model. `NaN` if `I` is
    /// singular.
    pub(crate) fn score_statistic(
        &self,
        x: &DMatrix<f64>,
        y: &DMatrix<f64>,
        full: &[(usize, usize)],
    ) -> Result<f64> {
        let mut data = RegressionData::new(x, y, self.fit_intercept)?;
        data.lt = self.lt_frame();
        let nf = full.len();
        let (info, g) = score_info(&data, &self.coef, full);
        // Order the kept coefficients first: the Cholesky factor `L` of `I`
        // then holds that of `I_ff` as its leading block, and with `Lz = g`,
        // `LM = Σ_{a ≥ k_f} z_a²` (one factorization, and no difference of
        // two large quadratic forms).
        let mut order: Vec<usize> = (0..nf).filter(|&a| self.mask[full[a]]).collect();
        let kf = order.len();
        order.extend((0..nf).filter(|&a| !self.mask[full[a]]));
        let sub = DMatrix::from_fn(nf, nf, |a, c| {
            let (u, v) = (order[a].max(order[c]), order[a].min(order[c]));
            info[(u, v)]
        });
        let gs = DVector::from_iterator(nf, order.iter().map(|&a| g[a]));
        Ok(sub
            .cholesky()
            .and_then(|ch| ch.l().solve_lower_triangular(&gs))
            .map_or(f64::NAN, |z| z.rows(kf, nf - kf).norm_squared()))
    }

    /// The distribution for one linear predictor `eta` (in this model's
    /// shape parameterization; pass `self.lt_frame()` for log-tangent fits).
    pub fn distribution_at(&self, eta: &[f64], lt: Option<&LtFrame>) -> Result<Esag> {
        shape_distribution(self.config.shape, lt, eta)
    }

    /// Per-observation log-likelihood `log f(y_i; eta_i)`.
    pub fn log_likelihood_rows(&self, x: &DMatrix<f64>, y: &DMatrix<f64>) -> Result<DVector<f64>> {
        let data = RegressionData::new(x, y, self.fit_intercept)?;
        if data.d != self.d {
            return Err(EsagError::Shape(format!(
                "Y has d = {} but the model has d = {}",
                data.d, self.d
            )));
        }
        let eta = self.linear_predictor(x)?;
        let lt = self.lt_frame();
        let mut out = DVector::zeros(data.n);
        let mut e = vec![0.0; self.b_dim()];
        let mut grad = vec![0.0; self.b_dim()];
        for i in 0..data.n {
            for k in 0..self.b_dim() {
                e[k] = eta[(i, k)];
            }
            let yi = &data.y[i * self.d..(i + 1) * self.d];
            out[i] = match (self.config.shape, &lt) {
                (Shape::LogTangent, Some(fr)) => crate::logtangent::loglik_lt(fr, &e, yi),
                (Shape::GammaTransported, Some(fr)) => {
                    crate::gamma_transported::loglik_gt_grad(fr, &e, yi, &mut grad)
                }
                _ => loglik::loglik_eta(&e, yi),
            };
        }
        Ok(out)
    }

    /// Mean log-likelihood per observation (higher is better) — the
    /// scikit-learn `score` analogue.
    pub fn score(&self, x: &DMatrix<f64>, y: &DMatrix<f64>) -> Result<f64> {
        let ll = self.log_likelihood_rows(x, y)?;
        Ok(ll.sum() / ll.len() as f64)
    }

    /// Residuals `Q_i` and `T1_i` for data `(x, y)`.
    pub fn residuals(&self, x: &DMatrix<f64>, y: &DMatrix<f64>) -> Result<Residuals> {
        let data = RegressionData::new(x, y, self.fit_intercept)?;
        let eta = self.linear_predictor(x)?;
        let d = self.d;
        let lt = self.lt_frame();
        let mut q = Vec::with_capacity(data.n);
        let mut t1 = Vec::with_capacity(data.n);
        for i in 0..data.n {
            let e: Vec<f64> = (0..self.b_dim()).map(|k| eta[(i, k)]).collect();
            let frame = self.frame_at(&e, lt.as_ref());
            let yi = &data.y[i * d..(i + 1) * d];
            let qi = residual_q(&frame, yi);
            let s: f64 = frame.lambda.iter().sum();
            q.push(qi);
            t1.push((frame.mu_norm * frame.mu_norm + s + 1.0) * qi);
        }
        Ok(Residuals { q, t1 })
    }

    /// Simulate responses from the fitted model at the rows of `x`
    /// (parametric bootstrap).
    pub fn sample_responses<R: Rng + ?Sized>(
        &self,
        x: &DMatrix<f64>,
        rng: &mut R,
    ) -> Result<DMatrix<f64>> {
        Ok(sample_with(&self.response_samplers(x)?, self.d, rng))
    }

    /// The per-observation samplers of [`EsagRegressorFit::sample_responses`]
    /// at the rows of `x`, for procedures that draw many response sets from
    /// one fit: [`sample_with`] then draws exactly what `sample_responses`
    /// would, without rebuilding `n` frames per draw (0.26 ms per bootstrap
    /// replicate at d = 3, n = 2000, 12% of a score-test replicate).
    pub(crate) fn response_samplers(
        &self,
        x: &DMatrix<f64>,
    ) -> Result<Vec<crate::distribution::Sampler>> {
        let eta = self.linear_predictor(x)?;
        let lt = self.lt_frame();
        let d = self.d;
        let mut row = vec![0.0; eta.ncols()];
        Ok((0..x.nrows())
            .map(|i| {
                for (k, r) in row.iter_mut().enumerate() {
                    *r = eta[(i, k)];
                }
                let frame = self.frame_at(&row, lt.as_ref());
                crate::distribution::Sampler::from_frame(&row[..d], &frame)
            })
            .collect())
    }

    /// Standard errors of the coefficients (`b_dim × p_total`; `NaN` for
    /// fixed entries), if the Hessian was computed and is invertible.
    pub fn std_errors(&self) -> Option<DMatrix<f64>> {
        let cov = self.cov_params.as_ref()?;
        let mut se = DMatrix::from_element(self.b_dim(), self.p_total(), f64::NAN);
        for (idx, &(k, j)) in self.free.iter().enumerate() {
            let v = cov[(idx, idx)];
            se[(k, j)] = if v > 0.0 { v.sqrt() } else { f64::NAN };
        }
        Some(se)
    }

    /// Names of the `eta` rows: `mu[1]..mu[d]`, then `gamma[1]..gamma[g]`
    /// (or `s[1]..s[g]` for the log-tangent shape).
    pub fn row_names(&self) -> Vec<String> {
        let d = self.d;
        let shape = if self.config.shape == Shape::LogTangent {
            "s"
        } else {
            "gamma"
        };
        (0..self.b_dim())
            .map(|k| {
                if k < d {
                    format!("mu[{}]", k + 1)
                } else {
                    format!("{shape}[{}]", k - d + 1)
                }
            })
            .collect()
    }

    /// Coefficients `B` (`b_dim × p_total`; rows `row_names()`, columns
    /// `column_names`), as statsmodels' `params`.
    pub fn params(&self) -> &DMatrix<f64> {
        &self.coef
    }
    /// Standard errors (`NaN` where fixed or unavailable), statsmodels' `bse`.
    pub fn bse(&self) -> DMatrix<f64> {
        self.std_errors()
            .unwrap_or_else(|| DMatrix::from_element(self.b_dim(), self.p_total(), f64::NAN))
    }
    /// z statistics `coef / se` (statsmodels' `tvalues` for these models).
    pub fn zvalues(&self) -> DMatrix<f64> {
        self.coef.zip_map(&self.bse(), |c, s| c / s)
    }
    /// Two-sided normal p-values of the z statistics.
    pub fn pvalues(&self) -> DMatrix<f64> {
        self.zvalues().map(|z| {
            if z.is_finite() {
                2.0 * norm_sf(z.abs())
            } else {
                f64::NAN
            }
        })
    }
    /// `(lower, upper)` Wald confidence limits at level `1 − alpha`.
    pub fn conf_int(&self, alpha: f64) -> (DMatrix<f64>, DMatrix<f64>) {
        let q = crate::stats::normal::norm_quantile(1.0 - alpha / 2.0);
        let se = self.bse();
        (
            self.coef.zip_map(&se, |c, s| c - q * s),
            self.coef.zip_map(&se, |c, s| c + q * s),
        )
    }
    /// Maximised log-likelihood (statsmodels' `llf`).
    pub fn llf(&self) -> f64 {
        self.log_likelihood
    }
    /// Number of observations.
    pub fn nobs(&self) -> usize {
        self.n_obs
    }
    /// Number of estimated parameters.
    pub fn df_model(&self) -> usize {
        self.n_params()
    }
    /// `nobs − df_model`.
    pub fn df_resid(&self) -> f64 {
        self.n_obs as f64 - self.n_params() as f64
    }

    /// statsmodels-style summary with 95% intervals (see [`Self::summary_alpha`]).
    pub fn summary(&self) -> String {
        self.summary_alpha(0.05)
    }

    /// statsmodels-style summary: a header block (model, data, fit
    /// statistics) and one coefficient block per parameter row of `B`, each
    /// listing the design columns with coef, std err, z, P>|z| and the
    /// `1 − alpha` confidence interval.
    pub fn summary_alpha(&self, alpha: f64) -> String {
        self.summary_titled("ESAG Regression Results", "MLE", alpha, &[], "(fixed)")
    }

    /// The summary with a custom title, method label and extra header rows.
    pub(crate) fn summary_titled(
        &self,
        title: &str,
        method: &str,
        alpha: f64,
        extra: &[(String, String)],
        fixed_note: &'static str,
    ) -> String {
        use crate::summary::{CoefRow, coef_blocks, header, num, utc_date_time};
        let (date, time) = utc_date_time();
        let dep = match &self.config.response_names {
            Some(v) if v.len() == self.d => v.join(", "),
            _ => format!("y (S^{})", self.d - 1),
        };
        let dep = if dep.len() > 24 {
            format!("{}…", &dep[..23])
        } else {
            dep
        };
        let shape = match self.config.shape {
            Shape::Gamma => "gamma",
            Shape::LogTangent => "log-tangent",
            Shape::GammaTransported => "gamma (transported)",
        };
        let mut left = vec![
            ("Dep. Variable:".to_string(), dep),
            ("Model:".to_string(), "ESAG".to_string()),
            ("Shape:".to_string(), shape.to_string()),
            ("Method:".to_string(), method.to_string()),
            ("Date:".to_string(), date),
            ("Time:".to_string(), format!("{time} UTC")),
            (
                "converged:".to_string(),
                if self.converged { "True" } else { "False" }.to_string(),
            ),
        ];
        let mut right = vec![
            ("No. Observations:".to_string(), self.n_obs.to_string()),
            ("Df Residuals:".to_string(), format!("{}", self.df_resid())),
            ("Df Model:".to_string(), self.n_params().to_string()),
            ("Log-Likelihood:".to_string(), num(self.log_likelihood)),
            ("AIC:".to_string(), num(self.aic())),
            ("BIC:".to_string(), num(self.bic())),
            (
                "Covariance Type:".to_string(),
                if self.cov_params.is_some() {
                    "observed info"
                } else {
                    "none"
                }
                .to_string(),
            ),
        ];
        for (i, e) in extra.iter().enumerate() {
            if i % 2 == 0 {
                left.push(e.clone())
            } else {
                right.push(e.clone())
            }
        }
        let se = self.std_errors();
        let rows = self.row_names();
        let blocks: Vec<(String, Vec<CoefRow>)> = (0..self.b_dim())
            .map(|k| {
                let r = (0..self.p_total())
                    .map(|j| CoefRow {
                        name: self.column_names[j].clone(),
                        coef: self.coef[(k, j)],
                        se: if self.mask[(k, j)] {
                            se.as_ref().map(|m| m[(k, j)])
                        } else {
                            None
                        },
                        note: if self.mask[(k, j)] { "n/a" } else { fixed_note },
                    })
                    .collect();
                (rows[k].clone(), r)
            })
            .collect();
        let mut s = header(title, &left, &right);
        s += &coef_blocks(&blocks, alpha);
        let mut notes: Vec<String> = vec![];
        let list = |v: &[usize]| {
            let shown: Vec<String> = v.iter().take(8).map(|i| i.to_string()).collect();
            format!(
                "{}{}",
                shown.join(", "),
                if v.len() > 8 { ", ..." } else { "" }
            )
        };
        if !self.singular_obs.is_empty() {
            let k = self.singular_obs.len();
            notes.push(format!(
                "{k} observation{} ({}) {} at a singular set of the gamma\n    parameterization, where V(gamma) is discontinuous. The fit stopped in a\n    trap there: a local supremum of the likelihood that is not a regular\n    maximum, so the standard errors are not meaningful (docs/MATH.md,\n    section 2). A trap next to the maximum pins only these observations,\n    so the fitted V(x) is usually close to the regular maximum's. The\n    log-tangent shape (Shape::LogTangent; shape=\"log-tangent\" in Python)\n    has no traps but is a different regression model, much worse on\n    strongly anisotropic data from gamma: compare the two by log-likelihood\n    (AIC/BIC) before switching.",
                if k == 1 { "" } else { "s" },
                list(&self.singular_obs),
                if k == 1 { "sits" } else { "sit" }
            ));
        }
        if !self.degenerate_obs.is_empty() {
            let k = self.degenerate_obs.len();
            notes.push(format!(
                "The likelihood is unbounded: the fit drives V(x) towards a singular\n    matrix at {k} observation{} ({}), whose density then exceeds the median\n    observation's by more than 12 nats. With {} free coefficients for {}\n    observations the MLE does not exist. Use fewer shape covariates or a\n    penalized fit (PenalizedEsag; fit_regularized in Python).",
                if k == 1 { "" } else { "s" },
                list(&self.degenerate_obs),
                self.free.len(),
                self.n_obs
            ));
        }
        if let (Shape::LogTangent, Some(m)) = (self.config.shape, &self.config.lt_reference) {
            let m: Vec<String> = m.iter().map(|v| format!("{v:.4}")).collect();
            notes.push(format!(
                "Log-tangent shape in the frame with reference direction\n    m = ({}).\n    The shape coefficients, and the model, depend on m: pass it as\n    lt_reference to fit the same model to other data.",
                m.join(", ")
            ));
        }
        if self.config.shape == Shape::Gamma && self.fit_intercept {
            let mu: Vec<f64> = (0..self.d).map(|k| self.coef[(k, 0)]).collect();
            let nm = mu.iter().map(|v| v * v).sum::<f64>().sqrt();
            let near = (mu[0] * mu[0] + mu[1] * mu[1]).sqrt() / nm;
            if nm > 0.0 && near < POLE_NEAR {
                notes.push(format!(
                    "The mean direction at x = 0 is near the pole of the gamma frame\n    onb(mu) (mu_0 = mu_1 = 0; sqrt(mu_0² + mu_1²)/|mu| = {near:.3}), where gamma\n    fits are slow and converge less often. Shape::GammaTransported\n    (shape=\"gamma-transported\" in Python) uses a frame without that pole."
                ));
            }
        }
        if let (Shape::GammaTransported, Some(m)) = (self.config.shape, &self.config.lt_reference) {
            let m: Vec<String> = m.iter().map(|v| format!("{v:.4}")).collect();
            notes.push(format!(
                "Gamma shape in the tangent frame transported from the reference\n    direction m = ({}) instead of onb(mu). With covariates the\n    model depends on m: pass it as lt_reference to fit the same model\n    to other data.",
                m.join(", ")
            ));
        }
        if !notes.is_empty() {
            s += "Notes:\n";
            for (i, n) in notes.iter().enumerate() {
                s += &format!("[{}] {n}\n", i + 1);
            }
        }
        s
    }

    /// Serialise the fitted model to JSON.
    pub fn to_json(&self) -> Result<String> {
        Ok(serde_json::to_string_pretty(self)?)
    }
    /// Deserialise a fitted model from JSON.
    pub fn from_json(s: &str) -> Result<Self> {
        Ok(serde_json::from_str(s)?)
    }
}

/// One response set (one row per sampler, `d` columns) from the samplers
/// of [`EsagRegressorFit::response_samplers`].
pub(crate) fn sample_with<R: Rng + ?Sized>(
    samplers: &[crate::distribution::Sampler],
    d: usize,
    rng: &mut R,
) -> DMatrix<f64> {
    let mut out = DMatrix::zeros(samplers.len(), d);
    let mut buf = vec![0.0; d];
    for (i, s) in samplers.iter().enumerate() {
        s.sample_into(rng, &mut buf);
        for k in 0..d {
            out[(i, k)] = buf[k];
        }
    }
    out
}

/// Canonicalise the sign of the first `gamma` group.
///
/// Negating `(gamma_1, gamma_2)` shifts `theta_1` by π, which negates the
/// eigenvectors `v_1, v_2` and leaves `V` — hence the density — unchanged, for
/// every `d` (Yu & Huang 2024 call such parameters *equivalent*). In a
/// regression `gamma_i = B_γ x_i` the symmetry is linear: negating rows
/// `d` and `d+1` of `B` in every column gives an identical likelihood. This
/// picks the representative whose first non-zero entry of row `d` (scanning
/// columns, then row `d+1`) is positive. Returns whether `coef` was flipped.
pub fn canonicalize_coef(coef: &mut DMatrix<f64>, d: usize) -> bool {
    let lead = (0..coef.ncols())
        .map(|j| coef[(d, j)])
        .chain((0..coef.ncols()).map(|j| coef[(d + 1, j)]))
        .find(|v| *v != 0.0);
    if matches!(lead, Some(v) if v < 0.0) {
        for j in 0..coef.ncols() {
            coef[(d, j)] = -coef[(d, j)];
            coef[(d + 1, j)] = -coef[(d + 1, j)];
        }
        true
    } else {
        false
    }
}

/// Build a free-parameter mask of shape `eta_len(d) × p_total` with every
/// entry `true`.
pub fn full_mask(d: usize, p_total: usize) -> DMatrix<bool> {
    DMatrix::from_element(dims::eta_len(d), p_total, true)
}

/// Mask for the *isotropic* restriction: every `gamma` row fixed (at 0) in
/// every column.
pub fn isotropic_mask(d: usize, p_total: usize) -> DMatrix<bool> {
    let mut m = full_mask(d, p_total);
    for j in 0..p_total {
        for k in d..dims::eta_len(d) {
            m[(k, j)] = false;
        }
    }
    m
}

/// Mask fixing the `mu` rows of design column `col` (R: `MLE_restricted_mu`).
pub fn mu_restricted_mask(d: usize, p_total: usize, col: usize) -> DMatrix<bool> {
    let mut m = full_mask(d, p_total);
    for k in 0..d {
        m[(k, col)] = false;
    }
    m
}

/// Mask fixing the `gamma` rows of design column `col` (R: `MLE_restricted_gamma`).
pub fn gamma_restricted_mask(d: usize, p_total: usize, col: usize) -> DMatrix<bool> {
    let mut m = full_mask(d, p_total);
    for k in d..dims::eta_len(d) {
        m[(k, col)] = false;
    }
    m
}

#[cfg(test)]
mod timing {
    use super::*;
    use std::time::Instant;

    /// `cargo test --release --lib timing -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn joint_eval_timings() {
        for &(d, p) in &[(3usize, 1usize), (4, 1), (7, 6), (8, 6)] {
            for &n in &[1000usize, 20000] {
                let bd = dims::eta_len(d);
                let x = DMatrix::from_fn(n, p, |i, j| (((i * 7 + j * 13) % 17) as f64 - 8.0) / 5.0);
                let y = DMatrix::from_fn(n, d, |i, k| {
                    ((i * 3 + k * 5) as f64 * 0.37).sin() + if k == d - 1 { 2.0 } else { 0.0 }
                });
                let data = RegressionData::new(&x, &y, true).unwrap();
                let b = DMatrix::from_fn(bd, p + 1, |k, j| {
                    0.3 * ((k * 5 + j * 3) as f64).sin() + if j == 0 && k < d { 1.0 } else { 0.0 }
                });
                for par in [false, true] {
                    let reps = if n > 5000 { 20 } else { 100 };
                    let t = Instant::now();
                    for _ in 0..reps {
                        std::hint::black_box(joint_loglik_and_grad(&data, &b, par));
                    }
                    let dt = t.elapsed().as_secs_f64() / reps as f64;
                    println!(
                        "d={d} p={p} n={n} par={par}: value+grad {:.1} µs ({:.0} ns/obs)",
                        dt * 1e6,
                        dt * 1e9 / n as f64
                    );
                }
                let t = Instant::now();
                std::hint::black_box(joint_loglik_grad_hess(&data, &b, true));
                println!(
                    "d={d} p={p} n={n}: value+grad+hess {:.1} ms",
                    t.elapsed().as_secs_f64() * 1e3
                );
            }
        }
    }
}

#[cfg(test)]
mod kernel_timing {
    use std::time::Instant;

    /// `cargo test --release --lib kernel_timing -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn kernel_vs_joint() {
        for d in [3usize, 4, 7, 8] {
            let bd = crate::dims::eta_len(d);
            let etas: Vec<Vec<f64>> = (0..64)
                .map(|s| {
                    (0..bd)
                        .map(|i| 0.3 + ((i * 7 + s) as f64 * 0.77).sin())
                        .collect()
                })
                .collect();
            let ys: Vec<Vec<f64>> = (0..64)
                .map(|s| {
                    let r: Vec<f64> = (0..d).map(|i| ((i * 3 + s) as f64 * 1.9).cos()).collect();
                    let n = r.iter().map(|v| v * v).sum::<f64>().sqrt();
                    r.iter().map(|v| v / n).collect()
                })
                .collect();
            let mut g = vec![0.0; bd];
            let reps = 200_000;
            let t = Instant::now();
            let mut acc = 0.0;
            for r in 0..reps {
                acc += crate::loglik::loglik_eta_grad(
                    std::hint::black_box(&etas[r % 64]),
                    &ys[r % 64],
                    &mut g,
                );
            }
            std::hint::black_box(acc);
            println!(
                "d={d}: kernel value+grad {:.0} ns",
                t.elapsed().as_secs_f64() * 1e9 / reps as f64
            );
            let t = Instant::now();
            for r in 0..reps {
                acc += crate::loglik::loglik_eta::<f64>(
                    std::hint::black_box(&etas[r % 64]),
                    &ys[r % 64],
                );
            }
            std::hint::black_box(acc);
            println!(
                "d={d}: kernel value {:.0} ns",
                t.elapsed().as_secs_f64() * 1e9 / reps as f64
            );
        }
    }
}

/// Where a full fit spends its time when both `μ` and `V` depend on `x`.
#[cfg(test)]
pub(crate) mod fit_profile;

/// Median over `reps` runs of `f`, in µs (for the studies).
#[cfg(test)]
fn fit_profile_median_us(reps: usize, mut f: impl FnMut()) -> f64 {
    let mut t: Vec<f64> = (0..reps)
        .map(|_| {
            let s = std::time::Instant::now();
            f();
            s.elapsed().as_secs_f64() * 1e6
        })
        .collect();
    t.sort_by(f64::total_cmp);
    t[reps / 2]
}

/// Single-core cost of γ fits.
#[cfg(test)]
mod gamma_profile;
