//! Limited-memory BFGS (L-BFGS) for smooth unconstrained minimisation.
//!
//! A self-contained quasi-Newton minimiser aimed at maximum-likelihood problems
//! whose gradients are exact (forward-mode AD), of small dimension (5–100
//! parameters) and whose objective is moderately expensive (a sum over
//! observations). Robustness of the line search is favoured over peak speed.
//!
//! * [`Objective`] — the problem interface: `f(x)` and `∇f(x)` in one call.
//!   [`FnObjective`] adapts a closure.
//! * [`LbfgsConfig`] — memory size, tolerances and line-search constants.
//! * [`minimize`] — the solver, returning an [`OptimResult`] with a [`Status`].
//!
//! # Algorithm
//!
//! * Search direction from the L-BFGS two-loop recursion (Nocedal & Wright,
//!   *Numerical Optimization*, 2nd ed., Alg. 7.4/7.5) with the initial Hessian
//!   approximation `H₀ = γ_k I`, `γ_k = sᵀy / yᵀy` taken from the newest pair.
//!   Curvature pairs with `sᵀy ≤ 1e-10 ‖s‖₂ ‖y‖₂` are skipped so that the
//!   approximation stays positive definite.
//! * Step length by a strong-Wolfe line search (N&W Alg. 3.5 bracketing +
//!   Alg. 3.6 zoom) with safeguarded cubic-Hermite / quadratic interpolation
//!   and bisection fallback. The initial trial step is 1 once curvature
//!   information is available and `min(1, 1/‖g‖₁)` on steepest-descent steps
//!   (the first iteration and after a memory reset). A non-finite trial value
//!   or gradient is treated as "step too long" and the step is shrunk. If the
//!   curvature condition cannot be met within `max_ls_iter` evaluations, the
//!   best point found that satisfies the sufficient-decrease (Armijo) condition
//!   is accepted — this is what makes mildly non-smooth objectives (an
//!   `|x|`-like kink) terminate gracefully instead of failing; the resulting
//!   curvature pair is then subject to the `sᵀy` test above.
//! * Near the minimiser the Armijo test compares function values that agree
//!   to rounding error; trial points are then also accepted under the
//!   Hager–Zhang *approximate Wolfe* conditions (a derivative bound in place
//!   of sufficient decrease, with `f` allowed to rise by at most
//!   `1e-12·|f|`). This lets the gradient reach the accuracy of the gradient
//!   itself rather than stalling at `‖g‖ ≈ sqrt(2 λ_max ulp(f))`.
//! * If the line search finds no point with sufficient decrease, the memory is
//!   discarded and the search is retried along `−∇f`; if that fails as well the
//!   solver stops with [`Status::LineSearchFailed`].
//! * The lowest point ever evaluated (including line-search trials) is tracked
//!   and returned, so the result is never worse than any point visited.
//!
//! # Stopping rules
//!
//! Checked in this order at the start of every iteration (so if several hold
//! at once, the earlier status is reported):
//!
//! 1. `‖∇f(x_k)‖_∞ ≤ tol_grad · max(1, ‖x_k‖_∞)` → [`Status::GradientConverged`].
//! 2. `tol_f > 0` and `|f_k − f_{k−1}| ≤ tol_f · max(|f_k|, |f_{k−1}|, 1)` → [`Status::FunctionConverged`].
//! 3. `tol_x > 0` and `‖x_k − x_{k−1}‖_∞ ≤ tol_x` → [`Status::StepConverged`].
//! 4. `n_iter ≥ max_iter` → [`Status::MaxIterations`].
//! 5. `n_fev ≥ max_fev` → [`Status::MaxEvaluations`] (also reported when the
//!    budget runs out inside a line search).
//!
//! # Example
//!
//! Implementing [`Objective`] for a struct (the usual case: the struct holds
//! the data):
//!
//! ```
//! use esag::optim::lbfgs::{minimize, LbfgsConfig, Objective, Status};
//!
//! /// f(x) = Σ_i (x_i − i)²
//! struct Shifted;
//!
//! impl Objective for Shifted {
//!     fn value_and_gradient(&self, x: &[f64], grad: &mut [f64]) -> f64 {
//!         let mut f = 0.0;
//!         for (i, (xi, gi)) in x.iter().zip(grad.iter_mut()).enumerate() {
//!             let r = xi - i as f64;
//!             f += r * r;
//!             *gi = 2.0 * r;
//!         }
//!         f
//!     }
//! }
//!
//! let res = minimize(&Shifted, &[0.0; 4], &LbfgsConfig::default());
//! assert!(res.converged());
//! assert_eq!(res.status, Status::GradientConverged);
//! assert!((res.x[3] - 3.0).abs() < 1e-6);
//! ```
//!
//! Using a closure and tightening the tolerances:
//!
//! ```
//! use esag::optim::lbfgs::{minimize, FnObjective, LbfgsConfig};
//!
//! // Rosenbrock's banana function.
//! let rosen = FnObjective(|x: &[f64], g: &mut [f64]| {
//!     let (a, b) = (x[0], x[1]);
//!     g[0] = -400.0 * a * (b - a * a) - 2.0 * (1.0 - a);
//!     g[1] = 200.0 * (b - a * a);
//!     100.0 * (b - a * a).powi(2) + (1.0 - a).powi(2)
//! });
//! let cfg = LbfgsConfig { tol_grad: 1e-10, tol_f: 0.0, ..LbfgsConfig::default() };
//! let res = minimize(&rosen, &[-1.2, 1.0], &cfg);
//! assert!(res.converged());
//! assert!((res.x[0] - 1.0).abs() < 1e-8 && (res.x[1] - 1.0).abs() < 1e-8);
//! ```

use std::collections::VecDeque;
use std::fmt;

/// A differentiable objective function `f: ℝⁿ → ℝ`.
///
/// Implementations must be thread-safe (`Sync`, methods take `&self`) so that
/// callers may evaluate the objective concurrently, e.g. the parallel
/// finite-difference Hessian in [`super::hessian::hessian_fd_par`].
pub trait Objective: Sync {
    /// Evaluate `f(x)` and write `∇f(x)` into `grad` (same length as `x`).
    ///
    /// Returning a non-finite value (NaN/±∞) is allowed and is interpreted by
    /// [`minimize`] as "this point is outside the valid region".
    fn value_and_gradient(&self, x: &[f64], grad: &mut [f64]) -> f64;

    /// Evaluate `f(x)` only. The default calls [`Self::value_and_gradient`]
    /// with a scratch buffer; override if the value alone is much cheaper.
    fn value(&self, x: &[f64]) -> f64 {
        let mut grad = vec![0.0; x.len()];
        self.value_and_gradient(x, &mut grad)
    }
}

/// Adapter that turns a closure `|x, grad| -> f` into an [`Objective`].
///
/// ```
/// use esag::optim::lbfgs::{FnObjective, Objective};
///
/// let sq = FnObjective(|x: &[f64], g: &mut [f64]| {
///     g.copy_from_slice(x);
///     g.iter_mut().for_each(|v| *v *= 2.0);
///     x.iter().map(|v| v * v).sum()
/// });
/// assert_eq!(sq.value(&[3.0, 4.0]), 25.0);
/// ```
pub struct FnObjective<F>(pub F);

impl<F> Objective for FnObjective<F>
where
    F: Fn(&[f64], &mut [f64]) -> f64 + Sync,
{
    fn value_and_gradient(&self, x: &[f64], grad: &mut [f64]) -> f64 {
        (self.0)(x, grad)
    }
}

/// Configuration for [`minimize`].
///
/// Construct with struct-update syntax from the defaults:
/// `LbfgsConfig { tol_grad: 1e-8, ..LbfgsConfig::default() }`.
#[derive(Clone, Debug)]
pub struct LbfgsConfig {
    /// Number of curvature pairs `(s, y)` kept. Default 10. `0` gives scaled
    /// steepest descent.
    pub memory: usize,
    /// Maximum number of accepted iterations. Default 500.
    pub max_iter: usize,
    /// Maximum number of objective evaluations. Default `20 · max_iter`.
    pub max_fev: usize,
    /// Gradient tolerance: stop when `‖∇f(x)‖_∞ ≤ tol_grad · max(1, ‖x‖_∞)`.
    /// Default 1e-6. This is scipy's `gtol` (an absolute infinity-norm test)
    /// relaxed proportionally once the iterate itself is larger than 1 in
    /// magnitude; for `‖x‖_∞ ≤ 1` it is exactly scipy's rule.
    pub tol_grad: f64,
    /// Relative function-decrease tolerance: stop when
    /// `|f_k − f_{k+1}| ≤ tol_f · max(|f_k|, |f_{k+1}|, 1)`. Default 1e-12.
    /// `0.0` disables the test.
    pub tol_f: f64,
    /// Step tolerance: stop when `‖x_{k+1} − x_k‖_∞ ≤ tol_x`. Default `0.0`,
    /// which disables the test.
    pub tol_x: f64,
    /// Sufficient-decrease (Armijo) constant of the strong Wolfe conditions.
    /// Default 1e-4. Must satisfy `0 < c1 < c2 < 1`.
    pub c1: f64,
    /// Curvature constant of the strong Wolfe conditions. Default 0.9.
    pub c2: f64,
    /// Maximum number of objective evaluations per line search. Default 40.
    pub max_ls_iter: usize,
    /// Print one line per iteration to stderr. Default `false`.
    pub verbose: bool,
    /// Optional shared, dynamically lowerable iteration limit: the solver
    /// stops with [`Status::MaxIterations`] once `n_iter` reaches
    /// `min(max_iter, budget)`. Lets concurrent runs from different starts
    /// cut each other short (see `EsagRegressor`'s `Init::Auto`).
    pub iter_budget: Option<std::sync::Arc<std::sync::atomic::AtomicUsize>>,
}

impl Default for LbfgsConfig {
    fn default() -> Self {
        Self {
            memory: 10,
            max_iter: 500,
            max_fev: 500 * 20,
            tol_grad: 1e-6,
            tol_f: 1e-12,
            tol_x: 0.0,
            c1: 1e-4,
            c2: 0.9,
            max_ls_iter: 40,
            verbose: false,
            iter_budget: None,
        }
    }
}

/// Why [`minimize`] stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Status {
    /// `‖∇f‖_∞ ≤ tol_grad · max(1, ‖x‖_∞)`.
    GradientConverged,
    /// The relative decrease of `f` between two iterations fell below `tol_f`.
    FunctionConverged,
    /// The step `‖x_{k+1} − x_k‖_∞` fell below `tol_x` (only if `tol_x > 0`).
    StepConverged,
    /// `max_iter` iterations were performed.
    MaxIterations,
    /// `max_fev` objective evaluations were used.
    MaxEvaluations,
    /// No point with sufficient decrease could be found, even along `−∇f`.
    /// Usually means the iterate is at a numerical optimum (gradient noise) or
    /// at a genuine kink; the best point found is returned.
    LineSearchFailed,
    /// The objective or its gradient was non-finite at the starting point.
    NonFinite,
}

impl Status {
    /// `true` for the three `*Converged` variants.
    pub fn is_converged(self) -> bool {
        matches!(
            self,
            Status::GradientConverged | Status::FunctionConverged | Status::StepConverged
        )
    }
}

impl fmt::Display for Status {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Status::GradientConverged => "gradient converged",
            Status::FunctionConverged => "function value converged",
            Status::StepConverged => "step size converged",
            Status::MaxIterations => "maximum iterations reached",
            Status::MaxEvaluations => "maximum function evaluations reached",
            Status::LineSearchFailed => "line search failed",
            Status::NonFinite => "non-finite objective at the starting point",
        };
        f.write_str(s)
    }
}

/// Result of [`minimize`]: the best point found and diagnostics.
#[derive(Clone, Debug)]
pub struct OptimResult {
    /// Minimiser (the lowest point evaluated).
    pub x: Vec<f64>,
    /// Objective value at `x`.
    pub f: f64,
    /// Gradient at `x`.
    pub grad: Vec<f64>,
    /// `‖grad‖_∞`.
    pub grad_norm_inf: f64,
    /// Number of accepted iterations (line searches that produced a step).
    pub n_iter: usize,
    /// Number of calls to [`Objective::value_and_gradient`].
    pub n_fev: usize,
    /// Termination reason.
    pub status: Status,
}

impl OptimResult {
    /// `true` if the status is one of the three `*Converged` variants.
    pub fn converged(&self) -> bool {
        self.status.is_converged()
    }
}

// ---------------------------------------------------------------------------
// Small dense-vector helpers.
// ---------------------------------------------------------------------------

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// `y += a · x`
fn axpy(a: f64, x: &[f64], y: &mut [f64]) {
    for (yi, xi) in y.iter_mut().zip(x) {
        *yi += a * xi;
    }
}

fn norm_inf(v: &[f64]) -> f64 {
    v.iter().fold(0.0, |m, x| m.max(x.abs()))
}

fn norm_1(v: &[f64]) -> f64 {
    v.iter().map(|x| x.abs()).sum()
}

fn norm_2(v: &[f64]) -> f64 {
    dot(v, v).sqrt()
}

fn all_finite(v: &[f64]) -> bool {
    v.iter().all(|x| x.is_finite())
}

// ---------------------------------------------------------------------------
// L-BFGS memory and direction.
// ---------------------------------------------------------------------------

/// One curvature pair `(s, y)` with cached scalars.
struct Pair {
    s: Vec<f64>,
    y: Vec<f64>,
    /// `1 / sᵀy`
    rho: f64,
    /// `sᵀy / yᵀy`, the `H₀` scaling used while this is the newest pair.
    gamma: f64,
}

/// Two-loop recursion (N&W Alg. 7.4): returns `d = −H_k ∇f`.
///
/// With an empty memory this is plain steepest descent `d = −∇f` (unscaled;
/// the caller scales the first trial step instead).
fn lbfgs_direction(mem: &VecDeque<Pair>, g: &[f64]) -> Vec<f64> {
    let mut q = g.to_vec();
    let Some(newest) = mem.back() else {
        q.iter_mut().for_each(|v| *v = -*v);
        return q;
    };
    // First loop: newest → oldest.
    let mut alphas = Vec::with_capacity(mem.len());
    for p in mem.iter().rev() {
        let a = p.rho * dot(&p.s, &q);
        axpy(-a, &p.y, &mut q);
        alphas.push(a);
    }
    q.iter_mut().for_each(|v| *v *= newest.gamma);
    // Second loop: oldest → newest (alphas were pushed newest-first).
    for (p, &a) in mem.iter().zip(alphas.iter().rev()) {
        let b = p.rho * dot(&p.y, &q);
        axpy(a - b, &p.s, &mut q);
    }
    q.iter_mut().for_each(|v| *v = -*v);
    q
}

// ---------------------------------------------------------------------------
// Best-point tracking.
// ---------------------------------------------------------------------------

struct Best {
    f: f64,
    x: Vec<f64>,
    g: Vec<f64>,
}

impl Best {
    fn new(f: f64, x: &[f64], g: &[f64]) -> Self {
        Self {
            f,
            x: x.to_vec(),
            g: g.to_vec(),
        }
    }

    fn update(&mut self, f: f64, x: &[f64], g: &[f64]) {
        if f < self.f {
            self.f = f;
            self.x.copy_from_slice(x);
            self.g.copy_from_slice(g);
        }
    }
}

// ---------------------------------------------------------------------------
// Strong-Wolfe line search (N&W Alg. 3.5 + 3.6).
// ---------------------------------------------------------------------------

/// A fully evaluated trial point `x₀ + α d`.
struct Trial {
    alpha: f64,
    x: Vec<f64>,
    f: f64,
    g: Vec<f64>,
    /// `φ'(α) = ∇f(x₀ + α d) · d`
    dphi: f64,
}

/// Scalar information about one end of a bracket (no `x`/`g` needed).
#[derive(Clone, Copy)]
struct End {
    alpha: f64,
    /// `+∞` for a non-finite trial point.
    phi: f64,
    /// `NaN` when unknown (non-finite trial point).
    dphi: f64,
}

impl Trial {
    fn end(&self) -> End {
        End {
            alpha: self.alpha,
            phi: self.f,
            dphi: self.dphi,
        }
    }
}

enum Eval {
    Point(Trial),
    NonFinite,
    Budget,
}

enum LsOutcome {
    Accepted(Trial),
    Failed,
    Budget,
}

/// Minimiser of the cubic Hermite interpolant through `(a, fa, da)` and
/// `(b, fb, db)` (N&W eq. 3.59), or `None` if it has no interior minimiser.
fn cubic_min(a: End, b: End) -> Option<f64> {
    let (da, db) = (a.dphi, b.dphi);
    let span = b.alpha - a.alpha;
    if span == 0.0 {
        return None;
    }
    let d1 = da + db - 3.0 * (a.phi - b.phi) / (a.alpha - b.alpha);
    let disc = d1 * d1 - da * db;
    if disc < 0.0 {
        return None;
    }
    let d2 = span.signum() * disc.sqrt();
    let denom = db - da + 2.0 * d2;
    if denom == 0.0 {
        return None;
    }
    let c = b.alpha - span * (db + d2 - d1) / denom;
    c.is_finite().then_some(c)
}

/// Minimiser of the quadratic through `(a, fa, da)` and `(b, fb)`, or `None`
/// if the quadratic is not strictly convex.
fn quad_min(a: End, b: End) -> Option<f64> {
    let span = b.alpha - a.alpha;
    let denom = 2.0 * (b.phi - a.phi - a.dphi * span);
    if !(denom > 0.0) {
        return None;
    }
    let c = a.alpha - a.dphi * span * span / denom;
    c.is_finite().then_some(c)
}

/// Safeguarded interpolation inside the bracket `[lo, hi]` (either order):
/// cubic Hermite if both derivatives are known, else quadratic, else bisection;
/// candidates too close to either end are replaced by bisection.
fn interpolate(lo: End, hi: End) -> f64 {
    let a = lo.alpha.min(hi.alpha);
    let b = lo.alpha.max(hi.alpha);
    let margin = 0.1 * (b - a);
    let inside = |c: f64| c > a + margin && c < b - margin;
    if hi.phi.is_finite() {
        if hi.dphi.is_finite()
            && let Some(c) = cubic_min(lo, hi)
            && inside(c)
        {
            return c;
        }
        if let Some(c) = quad_min(lo, hi)
            && inside(c)
        {
            return c;
        }
    }
    lo.alpha + 0.5 * (hi.alpha - lo.alpha)
}

struct LineSearch<'a> {
    obj: &'a dyn Objective,
    cfg: &'a LbfgsConfig,
    x0: &'a [f64],
    d: &'a [f64],
    phi0: f64,
    dphi0: f64,
    /// Below this `α`, `x₀ + α d` is indistinguishable from `x₀`.
    alpha_floor: f64,
    n_fev: &'a mut usize,
    best: &'a mut Best,
}

impl LineSearch<'_> {
    fn eval(&mut self, alpha: f64) -> Eval {
        if *self.n_fev >= self.cfg.max_fev {
            return Eval::Budget;
        }
        let x: Vec<f64> = self
            .x0
            .iter()
            .zip(self.d)
            .map(|(xi, di)| xi + alpha * di)
            .collect();
        let mut g = vec![0.0; x.len()];
        let f = self.obj.value_and_gradient(&x, &mut g);
        *self.n_fev += 1;
        if !f.is_finite() || !all_finite(&g) {
            return Eval::NonFinite;
        }
        self.best.update(f, &x, &g);
        let dphi = dot(&g, self.d);
        Eval::Point(Trial {
            alpha,
            x,
            f,
            g,
            dphi,
        })
    }

    fn armijo(&self, t: &Trial) -> bool {
        t.f <= self.phi0 + self.cfg.c1 * t.alpha * self.dphi0
    }

    fn curvature(&self, t: &Trial) -> bool {
        t.dphi.abs() <= -self.cfg.c2 * self.dphi0
    }

    /// Hager–Zhang approximate Wolfe acceptance (Hager & Zhang 2005, eq. 4.1
    /// with δ = 0.1): near a minimiser the Armijo test compares function
    /// values that agree to the last few digits, so it is replaced by the
    /// derivative bound `φ'(α) ≤ (2δ − 1) φ'(0)` that a quadratic model
    /// implies, provided `φ(α)` exceeds `φ(0)` by at most `1e-12 |φ(0)|`
    /// (rounding level). Combined with [`Self::curvature`] this lets the
    /// solver drive the gradient down to the accuracy of the *gradient*
    /// instead of stalling at `‖g‖ ≈ sqrt(2 λ_max ulp(f))`.
    fn approx_wolfe(&self, t: &Trial) -> bool {
        const DELTA: f64 = 0.1;
        let eps_f = 1e-12 * self.phi0.abs().max(1.0);
        t.f <= self.phi0 + eps_f && t.dphi <= (2.0 * DELTA - 1.0) * self.dphi0
    }

    /// Whether a trial point is acceptable as the result of the search.
    fn acceptable(&self, t: &Trial) -> bool {
        (self.armijo(t) || self.approx_wolfe(t)) && self.curvature(t)
    }

    fn origin(&self) -> End {
        End {
            alpha: 0.0,
            phi: self.phi0,
            dphi: self.dphi0,
        }
    }

    /// Bracketing phase (N&W Alg. 3.5).
    fn run(&mut self, alpha_init: f64) -> LsOutcome {
        if !(self.dphi0 < 0.0) || !(alpha_init > 0.0) {
            return LsOutcome::Failed;
        }
        // The previous trial (None ⇔ α = 0).
        let mut prev: Option<Trial> = None;
        let mut alpha = alpha_init;
        // Smallest α at which a non-finite value was seen.
        let mut alpha_cap = f64::INFINITY;
        let mut iters_left = self.cfg.max_ls_iter;
        while iters_left > 0 {
            iters_left -= 1;
            match self.eval(alpha) {
                Eval::Budget => return LsOutcome::Budget,
                Eval::NonFinite => {
                    // Treat as "too long": shrink towards the last finite point.
                    alpha_cap = alpha;
                    let lo = prev.as_ref().map_or(0.0, |t| t.alpha);
                    let next = lo + 0.5 * (alpha - lo);
                    if next <= lo || next <= self.alpha_floor {
                        break;
                    }
                    alpha = next;
                }
                Eval::Point(t) => {
                    if self.acceptable(&t) {
                        return LsOutcome::Accepted(t);
                    }
                    let prev_phi = prev.as_ref().map_or(self.phi0, |p| p.f);
                    if !self.armijo(&t) || (prev.is_some() && t.f >= prev_phi) {
                        let hi = t.end();
                        return self.zoom(prev, hi, iters_left);
                    }
                    if self.curvature(&t) {
                        return LsOutcome::Accepted(t);
                    }
                    if t.dphi >= 0.0 {
                        let hi = prev.as_ref().map_or(self.origin(), Trial::end);
                        return self.zoom(Some(t), hi, iters_left);
                    }
                    // Still descending with sufficient decrease: extrapolate.
                    let next = if alpha_cap.is_finite() {
                        0.5 * (alpha + alpha_cap)
                    } else {
                        2.0 * alpha
                    };
                    prev = Some(t);
                    alpha = next;
                }
            }
        }
        // Budget of trials exhausted while still descending: accept the last
        // point with sufficient decrease, if any.
        match prev {
            Some(t) => LsOutcome::Accepted(t),
            None => LsOutcome::Failed,
        }
    }

    /// Zoom phase (N&W Alg. 3.6). `lo` is `None` for `α = 0`.
    fn zoom(&mut self, mut lo: Option<Trial>, mut hi: End, iters_left: usize) -> LsOutcome {
        for _ in 0..iters_left {
            let lo_end = lo.as_ref().map_or(self.origin(), Trial::end);
            let width = (hi.alpha - lo_end.alpha).abs();
            let scale = lo_end.alpha.abs().max(hi.alpha.abs());
            if width <= 1e-14 * scale || scale <= self.alpha_floor {
                break;
            }
            let alpha_j = interpolate(lo_end, hi);
            match self.eval(alpha_j) {
                Eval::Budget => return LsOutcome::Budget,
                Eval::NonFinite => {
                    hi = End {
                        alpha: alpha_j,
                        phi: f64::INFINITY,
                        dphi: f64::NAN,
                    };
                }
                Eval::Point(t) => {
                    if self.acceptable(&t) {
                        return LsOutcome::Accepted(t);
                    }
                    if !self.armijo(&t) || t.f >= lo_end.phi {
                        hi = t.end();
                    } else {
                        if self.curvature(&t) {
                            return LsOutcome::Accepted(t);
                        }
                        if t.dphi * (hi.alpha - lo_end.alpha) >= 0.0 {
                            hi = lo_end;
                        }
                        lo = Some(t);
                    }
                }
            }
        }
        // Curvature condition not met within the budget: accept the best
        // sufficient-decrease point if there is one.
        match lo {
            Some(t) => LsOutcome::Accepted(t),
            None => LsOutcome::Failed,
        }
    }
}

// ---------------------------------------------------------------------------
// Driver.
// ---------------------------------------------------------------------------

/// Minimise `obj` starting from `x0` with L-BFGS and a strong-Wolfe line search.
///
/// See the [module documentation](self) for the algorithm and stopping rules.
/// The returned point is the lowest one evaluated; `status` tells why the
/// iteration stopped.
///
/// # Panics
///
/// If `cfg` violates `0 < c1 < c2 < 1`.
pub fn minimize(obj: &dyn Objective, x0: &[f64], cfg: &LbfgsConfig) -> OptimResult {
    assert!(
        0.0 < cfg.c1 && cfg.c1 < cfg.c2 && cfg.c2 < 1.0,
        "LbfgsConfig requires 0 < c1 < c2 < 1 (got c1 = {}, c2 = {})",
        cfg.c1,
        cfg.c2
    );

    let n = x0.len();
    let mut n_fev = 0usize;
    let mut x = x0.to_vec();
    let mut g = vec![0.0; n];
    let mut f = obj.value_and_gradient(&x, &mut g);
    n_fev += 1;
    if !f.is_finite() || !all_finite(&g) {
        return OptimResult {
            grad_norm_inf: norm_inf(&g),
            x,
            f,
            grad: g,
            n_iter: 0,
            n_fev,
            status: Status::NonFinite,
        };
    }

    let mut best = Best::new(f, &x, &g);
    let mut mem: VecDeque<Pair> = VecDeque::with_capacity(cfg.memory);
    let mut n_iter = 0usize;
    let mut f_prev = f;
    let mut last_df = f64::INFINITY;
    let mut last_dx = f64::INFINITY;

    let status = loop {
        // --- termination tests at the current iterate --------------------
        let gnorm = norm_inf(&g);
        if gnorm <= cfg.tol_grad * norm_inf(&x).max(1.0) {
            break Status::GradientConverged;
        }
        if cfg.tol_f > 0.0 && last_df <= cfg.tol_f * f.abs().max(f_prev.abs()).max(1.0) {
            break Status::FunctionConverged;
        }
        if cfg.tol_x > 0.0 && last_dx <= cfg.tol_x {
            break Status::StepConverged;
        }
        let limit = match &cfg.iter_budget {
            Some(b) => cfg
                .max_iter
                .min(b.load(std::sync::atomic::Ordering::Relaxed)),
            None => cfg.max_iter,
        };
        if n_iter >= limit {
            break Status::MaxIterations;
        }
        if n_fev >= cfg.max_fev {
            break Status::MaxEvaluations;
        }

        // --- search direction --------------------------------------------
        let mut d = lbfgs_direction(&mem, &g);
        let mut dphi0 = dot(&d, &g);
        if !(dphi0 < 0.0) || !all_finite(&d) {
            // Not a descent direction (rounding / bad curvature): reset.
            mem.clear();
            d = g.iter().map(|v| -v).collect();
            dphi0 = -dot(&g, &g);
        }
        let steepest_step = (1.0 / norm_1(&g)).min(1.0);
        let alpha_init = if mem.is_empty() { steepest_step } else { 1.0 };
        let alpha_floor =
            |d: &[f64]| f64::EPSILON * (1.0 + norm_inf(&x)) / norm_inf(d).max(f64::MIN_POSITIVE);

        // --- line search (with one steepest-descent retry) ----------------
        let mut outcome = LineSearch {
            obj,
            cfg,
            x0: &x,
            d: &d,
            phi0: f,
            dphi0,
            alpha_floor: alpha_floor(&d),
            n_fev: &mut n_fev,
            best: &mut best,
        }
        .run(alpha_init);

        if matches!(outcome, LsOutcome::Failed) && !mem.is_empty() {
            mem.clear();
            d = g.iter().map(|v| -v).collect();
            dphi0 = -dot(&g, &g);
            outcome = LineSearch {
                obj,
                cfg,
                x0: &x,
                d: &d,
                phi0: f,
                dphi0,
                alpha_floor: alpha_floor(&d),
                n_fev: &mut n_fev,
                best: &mut best,
            }
            .run(steepest_step);
        }

        let t = match outcome {
            LsOutcome::Accepted(t) => t,
            LsOutcome::Failed => break Status::LineSearchFailed,
            LsOutcome::Budget => break Status::MaxEvaluations,
        };

        // --- curvature pair ----------------------------------------------
        let s: Vec<f64> = t.x.iter().zip(&x).map(|(a, b)| a - b).collect();
        let y: Vec<f64> = t.g.iter().zip(&g).map(|(a, b)| a - b).collect();
        last_dx = norm_inf(&s);
        let sy = dot(&s, &y);
        let yy = dot(&y, &y);
        if cfg.memory > 0 && sy.is_finite() && sy > 1e-10 * norm_2(&s) * norm_2(&y) && yy > 0.0 {
            if mem.len() == cfg.memory {
                mem.pop_front();
            }
            mem.push_back(Pair {
                rho: 1.0 / sy,
                gamma: sy / yy,
                s,
                y,
            });
        }

        // --- accept the step ---------------------------------------------
        f_prev = f;
        last_df = (f - t.f).abs();
        x = t.x;
        f = t.f;
        g = t.g;
        n_iter += 1;

        if cfg.verbose {
            eprintln!(
                "lbfgs iter {n_iter:4}  f = {f:+.10e}  |g|_inf = {:.3e}  step = {:.3e}  nfev = {n_fev}  mem = {}",
                norm_inf(&g),
                t.alpha,
                mem.len()
            );
        }
    };

    // Never return a point worse than the best one seen.
    let (x, f, g) = if best.f < f {
        (best.x, best.f, best.g)
    } else {
        (x, f, g)
    };
    OptimResult {
        grad_norm_inf: norm_inf(&g),
        x,
        f,
        grad: g,
        n_iter,
        n_fev,
        status,
    }
}

// ---------------------------------------------------------------------------
// Tests.
// ---------------------------------------------------------------------------

/// Test objectives shared with `hessian.rs`.
#[cfg(test)]
pub(crate) mod test_objectives {
    use super::Objective;

    /// Chained Rosenbrock `Σ_{i<n-1} 100 (x_{i+1} − x_i²)² + (1 − x_i)²`.
    pub struct Rosenbrock;

    impl Objective for Rosenbrock {
        fn value_and_gradient(&self, x: &[f64], grad: &mut [f64]) -> f64 {
            let n = x.len();
            grad.iter_mut().for_each(|v| *v = 0.0);
            let mut f = 0.0;
            for i in 0..n.saturating_sub(1) {
                let r = x[i + 1] - x[i] * x[i];
                let t = 1.0 - x[i];
                f += 100.0 * r * r + t * t;
                grad[i] += -400.0 * x[i] * r - 2.0 * t;
                grad[i + 1] += 200.0 * r;
            }
            f
        }
    }

    /// `f(x) = ½ xᵀ A x + bᵀ x` with `A` symmetric, row-major `n × n`.
    pub struct Quadratic {
        pub a: Vec<f64>,
        pub b: Vec<f64>,
    }

    impl Quadratic {
        pub fn n(&self) -> usize {
            self.b.len()
        }

        /// Diagonal `A` with eigenvalues log-spaced from 1 to `cond`;
        /// `b = −A·1` so that the minimiser is the ones vector.
        pub fn ill_conditioned(n: usize, cond: f64) -> Self {
            let mut a = vec![0.0; n * n];
            for i in 0..n {
                a[i * n + i] = cond.powf(i as f64 / (n - 1) as f64);
            }
            let b: Vec<f64> = (0..n).map(|i| -a[i * n + i]).collect();
            Self { a, b }
        }

        /// Dense symmetric positive-definite `A = MᵀM + I` from a tiny
        /// deterministic LCG.
        pub fn dense(n: usize) -> Self {
            let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
            let mut next = || {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
            };
            let m: Vec<f64> = (0..n * n).map(|_| next()).collect();
            let mut a = vec![0.0; n * n];
            for i in 0..n {
                for j in 0..n {
                    let s: f64 = (0..n).map(|k| m[k * n + i] * m[k * n + j]).sum();
                    a[i * n + j] = s + if i == j { 1.0 } else { 0.0 };
                }
            }
            let b: Vec<f64> = (0..n).map(|_| next()).collect();
            Self { a, b }
        }
    }

    impl Objective for Quadratic {
        fn value_and_gradient(&self, x: &[f64], grad: &mut [f64]) -> f64 {
            let n = self.n();
            let mut f = 0.0;
            for i in 0..n {
                let ax: f64 = (0..n).map(|j| self.a[i * n + j] * x[j]).sum();
                grad[i] = ax + self.b[i];
                f += 0.5 * x[i] * ax + self.b[i] * x[i];
            }
            f
        }
    }
}

#[cfg(test)]
mod tests {
    use super::test_objectives::{Quadratic, Rosenbrock};
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn max_abs_err(x: &[f64], target: impl Fn(usize) -> f64) -> f64 {
        x.iter()
            .enumerate()
            .fold(0.0, |m, (i, v)| m.max((v - target(i)).abs()))
    }

    fn tight() -> LbfgsConfig {
        LbfgsConfig {
            tol_grad: 1e-10,
            // only the gradient test may stop these runs
            tol_f: 0.0,
            ..LbfgsConfig::default()
        }
    }

    fn mgh_start(n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| if i % 2 == 0 { -1.2 } else { 1.0 })
            .collect()
    }

    #[test]
    fn rosenbrock_2() {
        let res = minimize(&Rosenbrock, &[-1.2, 1.0], &tight());
        eprintln!(
            "rosenbrock n=2: status={:?} iters={} fev={} f={:.3e}",
            res.status, res.n_iter, res.n_fev, res.f
        );
        assert!(res.converged(), "{:?}", res.status);
        assert!(max_abs_err(&res.x, |_| 1.0) < 1e-8);
        assert!(
            res.n_iter >= 10 && res.n_iter <= 100,
            "n_iter = {}",
            res.n_iter
        );
    }

    #[test]
    fn rosenbrock_10_and_50() {
        for &n in &[10usize, 50] {
            // The chained Rosenbrock has a second, local minimum near
            // (−1, 1, …, 1) for n ≥ 4, so start from the origin rather than
            // the alternating MGH start (covered by the next test).
            let x0 = vec![0.0; n];
            let res = minimize(&Rosenbrock, &x0, &tight());
            eprintln!(
                "rosenbrock n={n}: status={:?} iters={} fev={} f={:.3e}",
                res.status, res.n_iter, res.n_fev, res.f
            );
            assert!(res.converged(), "n = {n}: {:?}", res.status);
            assert!(max_abs_err(&res.x, |_| 1.0) < 1e-8, "n = {n}");
            assert!(res.f < 1e-16);
        }
    }

    #[test]
    fn rosenbrock_mgh_start_reaches_a_minimum() {
        // From the alternating start the chained Rosenbrock may land in the
        // local minimum; either way the gradient must vanish.
        for &n in &[10usize, 50] {
            let res = minimize(&Rosenbrock, &mgh_start(n), &tight());
            eprintln!(
                "rosenbrock (MGH start) n={n}: status={:?} iters={} fev={} f={:.6e}",
                res.status, res.n_iter, res.n_fev, res.f
            );
            assert!(res.converged(), "n = {n}: {:?}", res.status);
            assert!(res.grad_norm_inf <= 1e-10);
            let global = max_abs_err(&res.x, |_| 1.0) < 1e-8;
            let local =
                (res.x[0] + 1.0).abs() < 0.05 && res.x[1..].iter().all(|v| (v - 1.0).abs() < 0.05);
            assert!(global || local, "unexpected minimiser {:?}", res.x);
        }
    }

    #[test]
    fn ill_conditioned_quadratic() {
        // Eigenvalues log-spaced in [1, 1e4] (likelihood Hessians are far
        // better conditioned than this).
        let n = 20;
        let q = Quadratic::ill_conditioned(n, 1e4);
        let cfg = LbfgsConfig {
            tol_grad: 1e-6,
            tol_f: 0.0,
            max_iter: 5000,
            ..LbfgsConfig::default()
        };
        let res = minimize(&q, &vec![0.0; n], &cfg);
        eprintln!(
            "ill-conditioned quadratic (cond 1e4) n={n}: status={:?} iters={} fev={} |g|={:.2e} max|x-1|={:.2e}",
            res.status,
            res.n_iter,
            res.n_fev,
            res.grad_norm_inf,
            max_abs_err(&res.x, |_| 1.0)
        );
        assert_eq!(res.status, Status::GradientConverged);
        assert!(max_abs_err(&res.x, |_| 1.0) < 1e-6);
    }

    #[test]
    fn very_ill_conditioned_quadratic_reaches_precision_floor() {
        // cond 1e6: L-BFGS(10) is slow here and stops when f no longer
        // changes in double precision; x must still be accurate.
        let n = 20;
        let q = Quadratic::ill_conditioned(n, 1e6);
        let cfg = LbfgsConfig {
            tol_grad: 1e-6,
            max_iter: 5000,
            max_fev: 100_000,
            ..LbfgsConfig::default()
        };
        let res = minimize(&q, &vec![0.0; n], &cfg);
        let f_star0 = -0.5 * (0..n).map(|i| q.a[i * n + i]).sum::<f64>();
        eprintln!(
            "cond 1e6: status={:?} iters={} |g|={:.2e} max|x-1|={:.2e} |f-f*|/|f*|={:.2e}",
            res.status,
            res.n_iter,
            res.grad_norm_inf,
            max_abs_err(&res.x, |_| 1.0),
            (res.f - f_star0).abs() / f_star0.abs()
        );
        assert!(res.converged(), "{:?}", res.status);
        // L-BFGS(10) crawls on this spectrum and the relative f-test stops
        // it once a full iteration gains < 1e-12·|f|; f is then accurate.
        assert!((res.f - f_star0).abs() <= 1e-8 * f_star0.abs());
    }

    #[test]
    fn dense_quadratic() {
        let q = Quadratic::dense(30);
        let res = minimize(&q, &[0.5; 30], &LbfgsConfig::default());
        eprintln!(
            "dense quadratic n=30: status={:?} iters={} fev={}",
            res.status, res.n_iter, res.n_fev
        );
        assert!(res.converged());
        assert!(res.grad_norm_inf <= 1e-6);
    }

    /// `f(x) = Σ sqrt(x_i² + ε²) + ½ Σ (x_i − c_i)²` with |c_i| < 1: the
    /// minimiser sits at the (smoothed) kink x ≈ 0.
    struct SmoothedKink {
        c: Vec<f64>,
        eps2: f64,
    }

    impl Objective for SmoothedKink {
        fn value_and_gradient(&self, x: &[f64], grad: &mut [f64]) -> f64 {
            let mut f = 0.0;
            for (i, (xi, gi)) in x.iter().zip(grad.iter_mut()).enumerate() {
                let r = (xi * xi + self.eps2).sqrt();
                let d = xi - self.c[i];
                f += r + 0.5 * d * d;
                // r == 0 only when eps2 == 0 and xi == 0: use the subgradient 0.
                *gi = if r > 0.0 { xi / r } else { 0.0 } + d;
            }
            f
        }
    }

    #[test]
    fn smoothed_kink_terminates_gracefully() {
        let n = 8;
        let kink = SmoothedKink {
            c: (0..n)
                .map(|i| 0.9 * if i % 2 == 0 { 1.0 } else { -1.0 } * (i + 1) as f64 / n as f64)
                .collect(),
            eps2: 1e-12,
        };
        let x0: Vec<f64> = (0..n).map(|i| 2.0 - i as f64 * 0.5).collect();
        let res = minimize(&kink, &x0, &LbfgsConfig::default());
        eprintln!(
            "smoothed kink n={n}: status={:?} iters={} fev={} f={:.12} |g|={:.2e} x[0]={:.3e}",
            res.status, res.n_iter, res.n_fev, res.f, res.grad_norm_inf, res.x[0]
        );
        assert!(res.f.is_finite());
        assert!(all_finite(&res.x));
        assert!(res.converged(), "{:?}", res.status);
        let f_at_zero = kink.value(&vec![0.0; n]);
        assert!(res.f <= f_at_zero + 1e-9);
        assert!(res.x.iter().all(|v| v.abs() < 1e-5));
    }

    #[test]
    fn exact_kink_does_not_panic() {
        // The genuinely non-smooth version (ε = 0): no strong-Wolfe point
        // exists at the kink. We only require a sane finite answer close to
        // the minimiser x = 0 and no panic.
        let n = 4;
        let kink = SmoothedKink {
            c: vec![0.5, -0.3, 0.7, -0.9],
            eps2: 0.0,
        };
        let res = minimize(&kink, &[1.0, -1.0, 2.0, 3.0], &LbfgsConfig::default());
        eprintln!(
            "exact kink: status={:?} iters={} fev={} f={:.12} x={:?}",
            res.status, res.n_iter, res.n_fev, res.f, res.x
        );
        assert!(res.f.is_finite() && all_finite(&res.x));
        // No strong-Wolfe point exists at an exact kink, so only require a
        // finite answer that is close to optimal and better than the start.
        assert!(res.f <= kink.value(&vec![0.0; n]) + 5e-2);
        assert!(res.f < kink.value(&[1.0, -1.0, 2.0, 3.0]));
    }

    #[test]
    fn non_finite_start() {
        let nan = FnObjective(|_x: &[f64], g: &mut [f64]| {
            g.iter_mut().for_each(|v| *v = 0.0);
            f64::NAN
        });
        let res = minimize(&nan, &[1.0, 2.0], &LbfgsConfig::default());
        assert_eq!(res.status, Status::NonFinite);
        assert!(!res.converged());
        assert_eq!(res.n_fev, 1);
        assert_eq!(res.n_iter, 0);
        assert_eq!(res.x, vec![1.0, 2.0]);
        assert!(res.f.is_nan());

        let inf_grad = FnObjective(|_x: &[f64], g: &mut [f64]| {
            g[0] = f64::INFINITY;
            1.0
        });
        let res = minimize(&inf_grad, &[1.0], &LbfgsConfig::default());
        assert_eq!(res.status, Status::NonFinite);
    }

    /// `f(x) = −ln x + x/10` on x > 0 (minimum at 10), NaN elsewhere. From
    /// x₀ = 40 the second L-BFGS step overshoots into x < 0, so the line search
    /// must shrink past a non-finite trial.
    struct LogBarrier {
        nonfinite_calls: AtomicUsize,
    }

    impl Objective for LogBarrier {
        fn value_and_gradient(&self, x: &[f64], grad: &mut [f64]) -> f64 {
            if x[0] <= 0.0 {
                self.nonfinite_calls.fetch_add(1, Ordering::Relaxed);
                grad[0] = f64::NAN;
                return f64::NAN;
            }
            grad[0] = -1.0 / x[0] + 0.1;
            -x[0].ln() + 0.1 * x[0]
        }
    }

    #[test]
    fn non_finite_trial_points_are_shrunk() {
        let obj = LogBarrier {
            nonfinite_calls: AtomicUsize::new(0),
        };
        let res = minimize(&obj, &[40.0], &tight());
        eprintln!(
            "log barrier: status={:?} iters={} fev={} nonfinite={} x={:.12}",
            res.status,
            res.n_iter,
            res.n_fev,
            obj.nonfinite_calls.load(Ordering::Relaxed),
            res.x[0]
        );
        assert!(res.converged(), "{:?}", res.status);
        assert!((res.x[0] - 10.0).abs() < 1e-8);
        assert!(obj.nonfinite_calls.load(Ordering::Relaxed) >= 1);
    }

    #[test]
    fn dyn_objective_closure_and_determinism() {
        // A closure-wrapping struct used through `&dyn Objective`.
        let shift = 3.0;
        let obj = FnObjective(move |x: &[f64], g: &mut [f64]| {
            let mut f = 0.0;
            for (xi, gi) in x.iter().zip(g.iter_mut()) {
                let r = xi - shift;
                f += r * r * r * r + r * r;
                *gi = 4.0 * r * r * r + 2.0 * r;
            }
            f
        });
        let dyn_obj: &dyn Objective = &obj;
        let cfg = LbfgsConfig::default();
        let r1 = minimize(dyn_obj, &[0.0, 10.0, -5.0], &cfg);
        assert!(r1.converged());
        assert!(max_abs_err(&r1.x, |_| shift) < 1e-5);

        // Deterministic: identical inputs give bit-identical outputs.
        let r2 = minimize(dyn_obj, &[0.0, 10.0, -5.0], &cfg);
        assert_eq!(r1.x, r2.x);
        assert_eq!(r1.f.to_bits(), r2.f.to_bits());
        assert_eq!(r1.grad, r2.grad);
        assert_eq!(
            (r1.n_iter, r1.n_fev, r1.status),
            (r2.n_iter, r2.n_fev, r2.status)
        );

        let r3 = minimize(&Rosenbrock, &mgh_start(10), &tight());
        let r4 = minimize(&Rosenbrock, &mgh_start(10), &tight());
        assert_eq!(r3.x, r4.x);
        assert_eq!((r3.n_iter, r3.n_fev), (r4.n_iter, r4.n_fev));
    }

    #[test]
    fn iteration_and_evaluation_limits() {
        let cfg = LbfgsConfig {
            max_iter: 3,
            ..tight()
        };
        let res = minimize(&Rosenbrock, &[-1.2, 1.0], &cfg);
        assert_eq!(res.status, Status::MaxIterations);
        assert_eq!(res.n_iter, 3);
        assert!(!res.converged());

        let cfg = LbfgsConfig {
            max_fev: 7,
            ..tight()
        };
        let res = minimize(&Rosenbrock, &[-1.2, 1.0], &cfg);
        assert_eq!(res.status, Status::MaxEvaluations);
        assert!(res.n_fev <= 7);
        // The best point is still better than the start.
        assert!(res.f < Rosenbrock.value(&[-1.2, 1.0]));
    }

    #[test]
    fn step_tolerance_and_function_tolerance() {
        let cfg = LbfgsConfig {
            tol_grad: 0.0,
            tol_f: 0.0,
            tol_x: 1e-3,
            ..LbfgsConfig::default()
        };
        let res = minimize(&Rosenbrock, &[-1.2, 1.0], &cfg);
        assert_eq!(res.status, Status::StepConverged);

        // With tol_grad = 0 and tol_x = 0, only tol_f can stop a smooth problem
        // (or a failed line search once rounding dominates).
        let cfg = LbfgsConfig {
            tol_grad: 0.0,
            tol_f: 1e-12,
            ..LbfgsConfig::default()
        };
        let res = minimize(&Quadratic::dense(5), &[1.0; 5], &cfg);
        assert!(
            matches!(
                res.status,
                Status::FunctionConverged | Status::LineSearchFailed
            ),
            "{:?}",
            res.status
        );
        assert!(res.grad_norm_inf < 1e-6);
    }

    #[test]
    fn scaled_objective_like_a_loglikelihood() {
        // Negative log-likelihood-sized objective: f ≈ 1e3 at the optimum.
        let base = Quadratic::dense(12);
        let obj = FnObjective(|x: &[f64], g: &mut [f64]| {
            let f = base.value_and_gradient(x, g);
            g.iter_mut().for_each(|v| *v *= 50.0);
            50.0 * f + 1000.0
        });
        let res = minimize(&obj, &[0.0; 12], &LbfgsConfig::default());
        eprintln!(
            "scaled quadratic: status={:?} iters={} fev={} f={:.6} |g|={:.2e}",
            res.status, res.n_iter, res.n_fev, res.f, res.grad_norm_inf
        );
        assert!(res.converged());
        // tol_f is relative to |f| ≈ 1e3, so the gradient is only ~1e-4 small.
        assert!(res.grad_norm_inf <= 1e-3);
    }

    #[test]
    fn zero_memory_is_scaled_steepest_descent() {
        let cfg = LbfgsConfig {
            memory: 0,
            max_iter: 5000,
            max_fev: 100_000,
            ..LbfgsConfig::default()
        };
        let res = minimize(&Quadratic::dense(5), &[1.0; 5], &cfg);
        assert!(res.converged(), "{:?}", res.status);
    }

    #[test]
    fn already_at_minimum() {
        let res = minimize(&Rosenbrock, &[1.0, 1.0], &LbfgsConfig::default());
        assert_eq!(res.status, Status::GradientConverged);
        assert_eq!(res.n_iter, 0);
        assert_eq!(res.n_fev, 1);
    }

    #[test]
    #[should_panic(expected = "0 < c1 < c2 < 1")]
    fn bad_wolfe_constants_panic() {
        let cfg = LbfgsConfig {
            c1: 0.95,
            ..LbfgsConfig::default()
        };
        let _ = minimize(&Rosenbrock, &[0.0, 0.0], &cfg);
    }

    #[test]
    fn lbfgs_direction_matches_bfgs_on_one_pair() {
        // With one pair, H = (I − ρ s yᵀ) γI (I − ρ y sᵀ) + ρ s sᵀ.
        let s = vec![1.0, 2.0, -0.5];
        let y = vec![0.5, 1.0, 1.0];
        let sy = dot(&s, &y);
        let yy = dot(&y, &y);
        let mut mem = VecDeque::new();
        mem.push_back(Pair {
            rho: 1.0 / sy,
            gamma: sy / yy,
            s: s.clone(),
            y: y.clone(),
        });
        let g = vec![0.3, -0.7, 1.1];
        let d = lbfgs_direction(&mem, &g);
        let n = 3;
        let rho = 1.0 / sy;
        let gamma = sy / yy;
        let delta = |i: usize, j: usize| if i == j { 1.0 } else { 0.0 };
        for i in 0..n {
            let mut hg = 0.0;
            for j in 0..n {
                let mut hij = 0.0;
                for k in 0..n {
                    let lik = delta(i, k) - rho * s[i] * y[k];
                    let rkj = delta(k, j) - rho * y[k] * s[j];
                    hij += lik * gamma * rkj;
                }
                hij += rho * s[i] * s[j];
                hg += hij * g[j];
            }
            assert!((d[i] + hg).abs() < 1e-12, "component {i}");
        }
    }

    #[test]
    #[ignore]
    fn experiment_ill_conditioned() {
        let n = 20;
        let q = Quadratic::ill_conditioned(n, 1e6);
        for &(m, c2) in &[
            (10usize, 0.9),
            (10, 0.1),
            (20, 0.9),
            (20, 0.1),
            (40, 0.9),
            (5, 0.9),
        ] {
            let cfg = LbfgsConfig {
                memory: m,
                c2,
                tol_grad: 1e-9,
                tol_f: 0.0,
                max_iter: 5000,
                max_fev: 100_000,
                ..LbfgsConfig::default()
            };
            let res = minimize(&q, &vec![0.0; n], &cfg);
            eprintln!(
                "m={m} c2={c2}: status={:?} iters={} fev={} |g|={:.2e} xerr={:.2e}",
                res.status,
                res.n_iter,
                res.n_fev,
                res.grad_norm_inf,
                max_abs_err(&res.x, |_| 1.0)
            );
        }
        // verbose trace of the first iterations
        let cfg = LbfgsConfig {
            tol_grad: 1e-9,
            tol_f: 0.0,
            max_iter: 30,
            verbose: true,
            ..LbfgsConfig::default()
        };
        let _ = minimize(&q, &vec![0.0; n], &cfg);
    }
}
