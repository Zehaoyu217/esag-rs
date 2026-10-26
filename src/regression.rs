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
        }
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
        // Where gamma's likelihood has cliffs and the winner sits with
        // observations on the singular set, the fit is in a trap, a
        // non-regular local supremum (`docs/MATH.md` §2). Perturbed starts
        // fall into other traps (they rescued 0 of 51 such fits), even
        // starts at the true coefficients do. Skip them.
        let pinned = !best.converged
            && data.gamma_cliffs()
            && !singular_observations(&data, &best.coef).is_empty();
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
        if pinned && data.d <= EXCLUSION_MAX_D && matches!(self.init, Init::Auto) {
            best = self.exclusion_rounds(x, y, &data, best, &mask);
        }
        self.finalize(&data, best, mask)
    }

    /// Rescue of a trapped gamma fit. A trap is not a root of the score
    /// equations, and the regular root next to it is reached by fitting
    /// without the trapped observations, warm, then with them again. This
    /// is repeated, excluding every observation trapped so far, while each
    /// round improves the likelihood (at most [`EXCLUSION_ROUNDS`]). On the
    /// hard scenario (`gamma_exclusion_rounds` in `fit_profile.rs`) this
    /// reaches a regular root, always with a higher likelihood than the
    /// trap, in 3 of 12 trapped fits at `d = 5` and 2 of 12 at `d = 6`; it
    /// never did at `d = 7` or 8, hence [`EXCLUSION_MAX_D`].
    fn exclusion_rounds(
        &self,
        x: &DMatrix<f64>,
        y: &DMatrix<f64>,
        data: &RegressionData,
        mut best: RawFit,
        mask: &DMatrix<bool>,
    ) -> RawFit {
        let free = free_params(mask);
        let mut out: Vec<usize> = vec![];
        for _ in 0..EXCLUSION_ROUNDS {
            let trapped = singular_observations(data, &best.coef);
            if trapped.is_empty() {
                break;
            }
            out.extend(trapped);
            out.sort_unstable();
            out.dedup();
            let keep: Vec<usize> = (0..data.n)
                .filter(|i| out.binary_search(i).is_err())
                .collect();
            let xs = DMatrix::from_fn(keep.len(), x.ncols(), |a, j| x[(keep[a], j)]);
            let ys = DMatrix::from_fn(keep.len(), y.ncols(), |a, j| y[(keep[a], j)]);
            let Ok(sub) = self.data(&xs, &ys) else { break };
            let rest = self.optimize(&sub, &best.coef, &free, self.max_iter);
            let mut next = self.optimize(data, &rest.coef, &free, self.max_iter);
            next.n_iter += best.n_iter + rest.n_iter;
            next.n_fev += best.n_fev + rest.n_fev;
            if !next.beats(&best) {
                break;
            }
            best = next;
            if best.converged {
                break;
            }
        }
        best
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
        let mut best: Option<RawFit> = None;
        for r in raws {
            if best.as_ref().is_none_or(|b| r.beats(b)) {
                best = Some(r);
            }
        }
        best.expect("at least one start")
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
            NEWTON_SWITCH * data.n as f64
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
        mut best: RawFit,
        mask: DMatrix<bool>,
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
            let (mut f0, mut g0, mut h_full) = joint_loglik_grad_hess(data, &coef, self.parallel);
            let mut chol = restrict(&h_full, data.p_total).cholesky();
            // (not for fits diagnosed as pinned at a singular set or with an
            // unbounded likelihood: more iterations only drift there)
            let hopeless = || {
                !best.converged
                    && (!singular_observations(data, &coef).is_empty()
                        || !degenerate_observations(data, &coef).is_empty())
            };
            if chol.is_none()
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
                    (f0, g0, h_full) = joint_loglik_grad_hess(data, &coef, self.parallel);
                    chol = restrict(&h_full, data.p_total).cholesky();
                }
            }
            let mut f_ll = f0;
            let mut g_full = g0.as_slice().to_vec();
            if let Some(ch) = chol {
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
/// Most rounds of [`EsagRegressor::exclusion_rounds`]; rescues came within
/// two rounds (eight changed nothing).
const EXCLUSION_ROUNDS: usize = 3;
/// Largest `d` at which trapped gamma fits get exclusion rounds.
const EXCLUSION_MAX_D: usize = 6;
/// A mean direction with `√(μ̂_0² + μ̂_1²)` below this (about 17° from the
/// pole set of `onb(μ)`) gets a summary note recommending
/// [`Shape::GammaTransported`]. The slow cases measured sat at 0.12–0.19
/// (gamma 3–5× slower there and converging less often), the generic ones at
/// 0.60–0.78.
const POLE_NEAR: f64 = 0.3;

/// Observations of a gamma fit at `b` that sit at a singular set of the
/// parameterization, where `V(γ)` is discontinuous: a group `j ≥ 2` whose
/// trailing pair nearly vanishes (relative to the group norm), or a mean
/// direction near the pole `μ̂₀ = μ̂₁ = 0` of the frame `onb(μ)`, in either
/// case with a gradient over [`SINGULAR_GRADIENT`] times the median. Near
/// such a set the observation's density changes by `O(1)` as the
/// coefficients turn around it, so the observation on the set at its best
/// angle is a trap: a non-regular local supremum of the likelihood, where
/// optimisers stall (`docs/MATH.md` §2). Empty for the log-tangent shape.
pub(crate) fn singular_observations(data: &RegressionData, b: &DMatrix<f64>) -> Vec<usize> {
    if data.lt.is_some() {
        return vec![];
    }
    let (d, bd, pt) = (data.d, data.b_dim, data.p_total);
    let mut eta = vec![0.0; bd];
    let mut g = vec![0.0; bd];
    let rows: Vec<(f64, f64)> = (0..data.n)
        .map(|i| {
            for (k, e) in eta.iter_mut().enumerate() {
                *e = (0..pt).map(|j| b[(k, j)] * data.x[i * pt + j]).sum();
            }
            kernel_grad(data, &eta, &data.y[i * d..(i + 1) * d], &mut g);
            let gn = g.iter().map(|v| v * v).sum::<f64>().sqrt();
            // relative distance to the nearest singular set
            let mm = eta[..d].iter().map(|v| v * v).sum::<f64>();
            let mut dist = match (&data.gt, mm > 0.0) {
                // transported frame: ‖μ̂ + m‖/2, zero at the antipode
                (Some(fr), true) => {
                    let inv = 1.0 / mm.sqrt();
                    let m = fr.reference();
                    (0..d)
                        .map(|k| (eta[k] * inv + m[k]).powi(2))
                        .sum::<f64>()
                        .sqrt()
                        * 0.5
                }
                // onb(μ): its pole μ̂_0 = μ̂_1 = 0
                (None, true) => ((eta[0] * eta[0] + eta[1] * eta[1]) / mm).sqrt(),
                (_, false) => 1.0,
            };
            for gr in 2..=d - 2 {
                let off = d + gr * (gr + 1) / 2 - 1;
                let grp = &eta[off..off + gr + 1];
                let all = grp.iter().map(|v| v * v).sum::<f64>().sqrt();
                if all > 0.0 {
                    let pair = (grp[gr - 1].powi(2) + grp[gr].powi(2)).sqrt();
                    dist = dist.min(pair / all);
                }
            }
            (gn, dist)
        })
        .collect();
    let mut norms: Vec<f64> = rows.iter().map(|r| r.0).collect();
    let mid = norms.len() / 2;
    let (_, med, _) = norms.select_nth_unstable_by(mid, f64::total_cmp);
    let cut = SINGULAR_GRADIENT * *med;
    rows.iter()
        .enumerate()
        .filter(|(_, r)| r.0 > cut && r.1 < SINGULAR_DISTANCE)
        .map(|(i, _)| i)
        .collect()
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

/// One chunk of [`joint_eval`]: log-likelihood, row-major gradient,
/// squared per-observation gradient norms (if requested) and the shared
/// shape's accumulator (if any).
type ChunkEval = (f64, Vec<f64>, Vec<f64>, Vec<f64>);

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
        Some(_) if shared => {
            let s: Vec<f64> = (d..bd).map(|k| b[(k, 0)]).collect();
            Some(crate::logtangent::SharedShape::new(&s, d - 1))
        }
        _ => None,
    };
    let acc_len = sh.as_ref().map_or(0, |s| s.acc_len());
    // Per chunk of observations: eta = B X_cᵀ (bd × m, one contiguous
    // column per observation) and the gradient Σ_i g_i x̃_iᵀ = G X_c, both
    // as matrix products.
    let body = |rows: std::ops::Range<usize>| -> ChunkEval {
        let m = rows.len();
        // row-major m × pt block of the design = column-major pt × m (X_cᵀ)
        let xt = nalgebra::DMatrixView::from_slice(&data.x[rows.start * pt..rows.end * pt], pt, m);
        let eta = b * xt;
        let mut ge = DMatrix::<f64>::zeros(bd, m);
        let mut f = 0.0;
        let mut norms = Vec::with_capacity(if ratio { m } else { 0 });
        let mut wacc = vec![0.0; acc_len];
        for (r, i) in rows.enumerate() {
            let yi = &data.y[i * d..(i + 1) * d];
            let gcol = &mut ge.as_mut_slice()[r * bd..(r + 1) * bd];
            let ei = &eta.as_slice()[r * bd..(r + 1) * bd];
            f += match (&sh, &data.lt) {
                (Some(sh), Some(fr)) => crate::logtangent::loglik_lt_grad_shared(
                    fr,
                    sh,
                    &ei[..d],
                    yi,
                    &mut gcol[..d],
                    &mut wacc,
                ),
                _ => kernel_grad(data, ei, yi, gcol),
            };
            if ratio {
                norms.push(gcol.iter().map(|v| v * v).sum::<f64>());
            }
        }
        // bd × pt gradient, returned row-major (k * pt + j)
        let gm = ge * xt.transpose();
        let mut g = vec![0.0; bd * pt];
        for k in 0..bd {
            for j in 0..pt {
                g[k * pt + j] = gm[(k, j)];
            }
        }
        (f, g, norms, wacc)
    };
    let (f, mut g, mut norms, wacc) = if parallel && data.n >= 2 * PAR_CHUNK {
        let n_chunks = data.n.div_ceil(PAR_CHUNK);
        // Chunk results are summed in chunk order (not with rayon's
        // `reduce`, whose combination tree depends on thread timing), so the
        // result is bit-for-bit reproducible.
        let parts: Vec<ChunkEval> = (0..n_chunks)
            .into_par_iter()
            .map(|c| body(c * PAR_CHUNK..((c + 1) * PAR_CHUNK).min(data.n)))
            .collect();
        let mut f = 0.0;
        let mut g = vec![0.0; bd * pt];
        let mut norms = Vec::with_capacity(if ratio { data.n } else { 0 });
        let mut wacc = vec![0.0; acc_len];
        for (fc, gc, nc, wc) in parts {
            f += fc;
            for (a, b) in g.iter_mut().zip(&gc) {
                *a += b;
            }
            norms.extend(nc);
            for (a, b) in wacc.iter_mut().zip(&wc) {
                *a += b;
            }
        }
        (f, g, norms, wacc)
    } else {
        body(0..data.n)
    };
    if let Some(sh) = &sh {
        // the shape rows of the intercept column (all others are fixed)
        let mut gs = vec![0.0; bd - d];
        sh.shape_gradient(&wacc, &mut gs);
        for (k, v) in gs.iter().enumerate() {
            g[(d + k) * pt] = *v;
        }
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
    (f, DMatrix::from_row_slice(bd, pt, &g), r)
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

/// Per-observation value, gradient and Hessian in `eta`. Exact second-order
/// AD for `d = 3` (0.36 µs); for `d ≥ 4` central differences of the exact
/// reverse-mode gradient, which are faster (d = 4: 1.25 vs 1.9 µs; d = 6:
/// 4.0 vs 23 µs) and agree with the exact Hessian to ~1e-8 relative.
#[inline]
fn obs_hessian(data: &RegressionData, eta: &[f64], y: &[f64], g: &mut [f64], h: &mut [f64]) -> f64 {
    if let Some(fr) = &data.lt {
        // exact (one eigen-decomposition; ~2.7× the gradient at d = 8)
        return crate::logtangent::loglik_lt_hess(fr, eta, y, g, h);
    }
    if let Some(fr) = &data.gt {
        return crate::gamma_transported::loglik_gt_hess_fd(fr, eta, y, g, h);
    }
    if eta.len() == 5 {
        crate::hdual::loglik_eta_hess(eta, y, g, h)
    } else {
        crate::hdual::loglik_eta_hess_fd(eta, y, g, h)
    }
}

/// Joint log-likelihood, gradient and Hessian with respect to the full
/// coefficient vector, ordered column-major (`j·b_dim + k` for entry
/// `(k, j)` of `B`). Per observation, `∇²ℓ_i(B) = (x̃_i x̃_iᵀ) ⊗ ∇²_η ℓ_i`.
pub(crate) fn joint_loglik_grad_hess(
    data: &RegressionData,
    b: &DMatrix<f64>,
    parallel: bool,
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
    let body = |rows: std::ops::Range<usize>| -> (f64, Vec<f64>, DMatrix<f64>) {
        let m = rows.len();
        let mut f = 0.0;
        let mut g = vec![0.0; full];
        let mut hs = DMatrix::<f64>::zeros(m, hpairs.len());
        let mut w = DMatrix::<f64>::zeros(m, xpairs.len());
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
            f += obs_hessian(data, &eta, yi, &mut ge, &mut he);
            for j in 0..pt {
                for k in 0..bd {
                    g[j * bd + k] += ge[k] * xi[j];
                }
            }
            for (q, &(k1, k2)) in hpairs.iter().enumerate() {
                hs[(r, q)] = he[k1 * bd + k2];
            }
            for (a, &(j1, j2)) in xpairs.iter().enumerate() {
                w[(r, a)] = xi[j1] * xi[j2];
            }
        }
        (f, g, w.tr_mul(&hs))
    };
    let (f, g, c) = if parallel && data.n >= 2 * PAR_CHUNK {
        // At most 64 chunks, summed in order.
        let chunk = PAR_CHUNK.max(data.n.div_ceil(64));
        let n_chunks = data.n.div_ceil(chunk);
        let parts: Vec<(f64, Vec<f64>, DMatrix<f64>)> = (0..n_chunks)
            .into_par_iter()
            .map(|c| body(c * chunk..((c + 1) * chunk).min(data.n)))
            .collect();
        let mut f = 0.0;
        let mut g = vec![0.0; full];
        let mut c = DMatrix::<f64>::zeros(xpairs.len(), hpairs.len());
        for (fc, gc, cc) in parts {
            f += fc;
            for (a, b) in g.iter_mut().zip(&gc) {
                *a += b;
            }
            c += cc;
        }
        (f, g, c)
    } else {
        body(0..data.n)
    };
    let mut h = vec![0.0; full * full];
    for (a, &(j1, j2)) in xpairs.iter().enumerate() {
        for (q, &(k1, k2)) in hpairs.iter().enumerate() {
            let v = c[(a, q)];
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
        let eta = self.linear_predictor(x)?;
        let lt = self.lt_frame();
        let d = self.d;
        let mut out = DMatrix::zeros(x.nrows(), d);
        let mut buf = vec![0.0; d];
        let mut row = vec![0.0; eta.ncols()];
        for i in 0..x.nrows() {
            for (k, r) in row.iter_mut().enumerate() {
                *r = eta[(i, k)];
            }
            let frame = self.frame_at(&row, lt.as_ref());
            crate::distribution::Sampler::from_frame(&row[..d], &frame).sample_into(rng, &mut buf);
            for k in 0..d {
                out[(i, k)] = buf[k];
            }
        }
        Ok(out)
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
                "{k} observation{} ({}) {} at a singular set of the gamma\n    parameterization, where V(gamma) is discontinuous. The fit stopped in a\n    trap there: a local supremum of the likelihood that is not a regular\n    maximum, so this is not an MLE in the usual sense and the standard\n    errors are not meaningful (docs/MATH.md, section 2); restarts only find\n    other traps. The log-tangent shape (Shape::LogTangent;\n    shape=\"log-tangent\" in Python) describes the same distributions with a\n    smooth parameterization, which has no traps.",
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
