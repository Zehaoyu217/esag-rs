//! Proximal Newton for "smooth + weighted sparsity penalty" objectives,
//!
//! ```text
//! F(θ) = f(θ) + λ Σ_g w_g(θ) φ_g(θ),     w_g(θ) = b_g^{r(θ)},
//! ```
//!
//! the form of every LASSO-type penalty in this crate (Yu & Huang, Project
//! 3, and the corrected designs of `docs/project3/METHODOLOGY.md`). The
//! *atoms* `φ_g ≥ 0` come from a [`PenaltyAtoms`] implementation with an
//! exact proximal operator:
//!
//! * [`GroupNorms`]: `φ_g = ‖θ_{G_g}‖₂` over disjoint index groups
//!   (singletons = elementwise LASSO); prox = block soft-thresholding.
//! * [`EigenGaps`]: `φ_j = σ_(j+1) − σ_(j)`, the gaps between the sorted
//!   eigenvalues of a symmetric traceless matrix stored in log-tangent
//!   coordinates; zero gaps are tied eigenvalues (partial isotropy). Prox =
//!   eigen-decomposition + isotonic regression (PAVA), exact for any
//!   weights.
//!
//! Weights: base `b_g ≥ 0` (`∞` pins the atom at zero) raised to a fixed
//! exponent `r` (0 = plain, 1 = adaptive) or to the ratio of concentration
//! `r(θ) = max(1, ‖μ̃‖/‖μ‖)` of the unpenalized block `μ` (RoC). For RoC
//! the weights depend smoothly on `μ`; with [`RocMode::Joint`] (the paper's
//! objective) their `μ`-derivative
//! `λ Σ_g φ_g w_g ln(b_g) ∂r/∂μ`, `∂r/∂μ = −‖μ̃‖ μ/‖μ‖³` (0 on the floor),
//! enters the smooth part; with [`RocMode::FixedPoint`] (the recommended
//! default) `r` is frozen, the problem solved, `r` updated, until it
//! settles.
//!
//! # Algorithm (proximal Newton; Lee, Sun & Saunders 2014)
//!
//! At `θ_k`, with `H` the exact Hessian of `f` (shifted to be positive
//! definite if needed) and the weights frozen at `w(θ_k)`, the model
//! `∇h·(z − θ_k) + ½(z − θ_k)ᵀH(z − θ_k) + λ Σ_g w_g φ_g(z)` is minimised by
//! FISTA with restarts (only matrix–vector products: no likelihood
//! evaluations). The step `Δ = z* − θ_k` is accepted by backtracking on the
//! true `F`. Stationarity is certified by the gradient mapping
//! `‖θ − prox_{τP}(θ − τ∇h)‖/τ` for small `τ`, which for group norms is the
//! usual KKT residual.

use nalgebra::{DMatrix, DVector, SymmetricEigen};

/// A smooth objective with gradient and Hessian.
pub trait SmoothObjective: Sync {
    /// `f(θ)` and `∇f(θ)` (written into `grad`).
    fn value_grad(&self, theta: &[f64], grad: &mut [f64]) -> f64;
    /// `f(θ)`, `∇f(θ)` and the Hessian `∇²f(θ)` (row-major `n × n`).
    fn value_grad_hess(&self, theta: &[f64], grad: &mut [f64], hess: &mut [f64]) -> f64;
}

/// The atoms `φ_g(θ) ≥ 0` of a penalty, with their exact proximal map.
pub trait PenaltyAtoms: Sync + Send + std::fmt::Debug {
    /// Number of atoms.
    fn len(&self) -> usize;
    /// Whether there are no atoms.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// `φ_g(θ)` for every atom.
    fn atoms(&self, theta: &[f64], out: &mut [f64]);
    /// In place: `v ← argmin_z ½‖z − v‖² + Σ_g t_g φ_g(z)` (`t_g` may be ∞).
    fn prox(&self, v: &mut [f64], t: &[f64]);
    /// Indices of the parameters the penalty acts on.
    fn support(&self) -> Vec<usize>;
}

/// `φ_g = ‖θ_{G_g}‖₂` over disjoint groups of indices.
#[derive(Clone, Debug, PartialEq)]
pub struct GroupNorms {
    pub groups: Vec<Vec<usize>>,
}

impl PenaltyAtoms for GroupNorms {
    fn len(&self) -> usize {
        self.groups.len()
    }
    fn atoms(&self, theta: &[f64], out: &mut [f64]) {
        for (o, g) in out.iter_mut().zip(&self.groups) {
            *o = g.iter().map(|&i| theta[i] * theta[i]).sum::<f64>().sqrt();
        }
    }
    fn prox(&self, v: &mut [f64], t: &[f64]) {
        for (g, &tg) in self.groups.iter().zip(t) {
            let nrm = g.iter().map(|&i| v[i] * v[i]).sum::<f64>().sqrt();
            let scale = if !(tg < nrm) { 0.0 } else { 1.0 - tg / nrm };
            for &i in g {
                v[i] *= scale;
            }
        }
    }
    fn support(&self) -> Vec<usize> {
        self.groups.iter().flatten().copied().collect()
    }
}

/// Sparse-group atoms over disjoint groups: first one `‖θ_G‖₂` per group,
/// then one `|θ_i|` per member (in group order). With weights
/// `(1−α)w_G` and `αw_i` this is the sparse group LASSO; its prox is exact:
/// soft-threshold the entries, then block-soft-threshold each group.
#[derive(Clone, Debug, PartialEq)]
pub struct SparseGroupNorms {
    pub groups: Vec<Vec<usize>>,
}

impl SparseGroupNorms {
    /// Index of the first entry atom.
    fn n_groups(&self) -> usize {
        self.groups.len()
    }
}

impl PenaltyAtoms for SparseGroupNorms {
    fn len(&self) -> usize {
        self.groups.len() + self.groups.iter().map(Vec::len).sum::<usize>()
    }
    fn atoms(&self, theta: &[f64], out: &mut [f64]) {
        let mut e = self.n_groups();
        for (o, g) in out.iter_mut().zip(&self.groups) {
            *o = g.iter().map(|&i| theta[i] * theta[i]).sum::<f64>().sqrt();
        }
        for g in &self.groups {
            for &i in g {
                out[e] = theta[i].abs();
                e += 1;
            }
        }
    }
    fn prox(&self, v: &mut [f64], t: &[f64]) {
        let mut e = self.n_groups();
        for (gi, g) in self.groups.iter().enumerate() {
            for &i in g {
                let te = t[e];
                e += 1;
                v[i] = if !(te < v[i].abs()) {
                    0.0
                } else {
                    v[i] - te.copysign(v[i])
                };
            }
            let nrm = g.iter().map(|&i| v[i] * v[i]).sum::<f64>().sqrt();
            let tg = t[gi];
            let scale = if !(tg < nrm) { 0.0 } else { 1.0 - tg / nrm };
            for &i in g {
                v[i] *= scale;
            }
        }
    }
    fn support(&self) -> Vec<usize> {
        self.groups.iter().flatten().copied().collect()
    }
}

/// `φ_j = σ_(j+1) − σ_(j)`: gaps between the sorted eigenvalues of the
/// traceless symmetric `p × p` matrix `S` whose log-tangent coordinates
/// ([`crate::logtangent::s_to_matrix`]) are `θ[offset .. offset + s_len]`.
#[derive(Clone, Debug, PartialEq)]
pub struct EigenGaps {
    pub p: usize,
    pub offset: usize,
}

impl EigenGaps {
    fn s_len(&self) -> usize {
        (self.p - 1) * (self.p + 2) / 2
    }
    fn eigen(&self, theta: &[f64]) -> SymmetricEigen<f64, nalgebra::Dyn> {
        let s = &theta[self.offset..self.offset + self.s_len()];
        let m = crate::logtangent::s_to_matrix(s, self.p);
        SymmetricEigen::new(DMatrix::from_row_slice(self.p, self.p, &m))
    }
    /// Sorted eigenvalues (ascending) and the matching eigenvectors.
    fn sorted(&self, theta: &[f64]) -> (Vec<f64>, DMatrix<f64>) {
        let e = self.eigen(theta);
        let mut idx: Vec<usize> = (0..self.p).collect();
        idx.sort_by(|&a, &b| e.eigenvalues[a].total_cmp(&e.eigenvalues[b]));
        let vals = idx.iter().map(|&i| e.eigenvalues[i]).collect();
        let vecs = DMatrix::from_fn(self.p, self.p, |r, c| e.eigenvectors[(r, idx[c])]);
        (vals, vecs)
    }
}

impl PenaltyAtoms for EigenGaps {
    fn len(&self) -> usize {
        self.p - 1
    }
    fn atoms(&self, theta: &[f64], out: &mut [f64]) {
        let (vals, _) = self.sorted(theta);
        // gaps at rounding level (ties reconstructed through S) are ties
        let scale = vals.iter().fold(1.0f64, |a, &v| a.max(v.abs()));
        for j in 0..self.p - 1 {
            let g = vals[j + 1] - vals[j];
            out[j] = if g <= 1e-12 * scale { 0.0 } else { g };
        }
    }
    fn prox(&self, v: &mut [f64], t: &[f64]) {
        // The penalty is orthogonally invariant and, on the cone of sorted
        // eigenvalues, linear: Σ_j t_j (x_{j+1} − x_j) = cᵀx with
        // c_1 = −t_1, c_j = t_{j−1} − t_j, c_p = t_{p−1}. So the prox keeps the
        // eigenvectors and solves an isotonic regression of ζ − c.
        let (zeta, q) = self.sorted(v);
        let p = self.p;
        // Eigenvalues joined by an infinite threshold are tied exactly: merge
        // them into blocks first; block b with mean ζ̄_b and size m_b gets the
        // linear coefficient c_b = t(left boundary) − t(right boundary), so
        // the block values solve a weighted isotonic regression of
        // ζ̄_b − c_b/m_b with weights m_b.
        let mut starts = vec![0usize];
        for j in 1..p {
            if !t[j - 1].is_finite() {
                continue;
            }
            starts.push(j);
        }
        let nb = starts.len();
        let mut targets = Vec::with_capacity(nb);
        let mut weights = Vec::with_capacity(nb);
        for b in 0..nb {
            let (a, e) = (starts[b], if b + 1 < nb { starts[b + 1] } else { p });
            let m = (e - a) as f64;
            let mean = zeta[a..e].iter().sum::<f64>() / m;
            let left = if a > 0 { t[a - 1] } else { 0.0 };
            let right = if e < p { t[e - 1] } else { 0.0 };
            targets.push(mean - (left - right) / m);
            weights.push(m);
        }
        let xb = pava_weighted(&targets, &weights);
        let mut x = vec![0.0; p];
        for b in 0..nb {
            let (a, e) = (starts[b], if b + 1 < nb { starts[b + 1] } else { p });
            for xj in &mut x[a..e] {
                *xj = xb[b];
            }
        }
        let m = &q * DMatrix::from_diagonal(&DVector::from_vec(x)) * q.transpose();
        let flat: Vec<f64> = (0..p)
            .flat_map(|r| (0..p).map(move |c| (r, c)))
            .map(|(r, c)| m[(r, c)])
            .collect();
        let s = crate::logtangent::matrix_to_s(&flat, p);
        v[self.offset..self.offset + s.len()].copy_from_slice(&s);
    }
    fn support(&self) -> Vec<usize> {
        (self.offset..self.offset + self.s_len()).collect()
    }
}

/// Weighted isotonic (non-decreasing) least-squares fit (PAVA).
pub fn pava_weighted(y: &[f64], w: &[f64]) -> Vec<f64> {
    // blocks of (weighted sum, total weight, count)
    let mut sums: Vec<f64> = Vec::with_capacity(y.len());
    let mut ws: Vec<f64> = Vec::with_capacity(y.len());
    let mut counts: Vec<usize> = Vec::with_capacity(y.len());
    for (&v, &wi) in y.iter().zip(w) {
        sums.push(v * wi);
        ws.push(wi);
        counts.push(1);
        while sums.len() > 1 {
            let k = sums.len();
            if sums[k - 2] / ws[k - 2] > sums[k - 1] / ws[k - 1] {
                let (s, ww, c) = (
                    sums.pop().unwrap(),
                    ws.pop().unwrap(),
                    counts.pop().unwrap(),
                );
                sums[k - 2] += s;
                ws[k - 2] += ww;
                counts[k - 2] += c;
            } else {
                break;
            }
        }
    }
    let mut out = Vec::with_capacity(y.len());
    for ((s, ww), c) in sums.iter().zip(&ws).zip(&counts) {
        out.extend(std::iter::repeat_n(s / ww, *c));
    }
    out
}

/// Isotonic (non-decreasing) least-squares fit with unit weights (PAVA).
pub fn pava_nondecreasing(y: &[f64]) -> Vec<f64> {
    // blocks of (sum, count)
    let mut sums: Vec<f64> = Vec::with_capacity(y.len());
    let mut counts: Vec<usize> = Vec::with_capacity(y.len());
    for &v in y {
        sums.push(v);
        counts.push(1);
        while sums.len() > 1 {
            let k = sums.len();
            if sums[k - 2] / counts[k - 2] as f64 > sums[k - 1] / counts[k - 1] as f64 {
                let (s, c) = (sums.pop().unwrap(), counts.pop().unwrap());
                sums[k - 2] += s;
                counts[k - 2] += c;
            } else {
                break;
            }
        }
    }
    let mut out = Vec::with_capacity(y.len());
    for (s, c) in sums.iter().zip(&counts) {
        out.extend(std::iter::repeat_n(s / *c as f64, *c));
    }
    out
}

/// How RoC weights are handled.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum RocMode {
    /// `r(μ)` inside the objective (the paper's formulation); its
    /// `μ`-gradient pushes `‖μ̂‖` down.
    Joint,
    /// Solve with `r` frozen, update `r = max(1, ‖μ̃‖/‖μ̂‖)`, repeat
    /// (recommended: the paper's feedback without the incentive to shrink
    /// `‖μ̂‖`).
    FixedPoint,
}

/// Exponent `r` of the weights `b_g^r`.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum Exponent {
    /// A constant `r` (`0` = plain, `1` = adaptive).
    Fixed(f64),
    /// `r = max(1, ‖μ̃‖ / ‖μ‖)` with `μ = θ[mu]` and `‖μ̃‖ = mu_tilde_norm`.
    RoC {
        mu: Vec<usize>,
        mu_tilde_norm: f64,
        mode: RocMode,
    },
}

/// `λ Σ_g b_g^{r(θ)} φ_g(θ)`.
#[derive(Clone, Debug)]
pub struct Penalty<A: PenaltyAtoms> {
    pub atoms: A,
    /// Base weights `b_g ≥ 0` (`∞` pins the atom at zero, `0` leaves it free).
    pub base: Vec<f64>,
    pub exponent: Exponent,
    pub lambda: f64,
}

impl<A: PenaltyAtoms> Penalty<A> {
    /// The exponent `r(θ)`.
    pub fn r(&self, theta: &[f64]) -> f64 {
        match &self.exponent {
            Exponent::Fixed(r) => *r,
            Exponent::RoC {
                mu, mu_tilde_norm, ..
            } => {
                let nm = mu.iter().map(|&i| theta[i] * theta[i]).sum::<f64>().sqrt();
                (mu_tilde_norm / nm).max(1.0)
            }
        }
    }

    /// Weights `w_g = b_g^r` (with `0^0 = 1`, `∞^r = ∞`, `b^0 = 1`).
    pub fn weights_at_r(&self, r: f64) -> Vec<f64> {
        self.base
            .iter()
            .map(|&b| {
                if r == 0.0 {
                    if b.is_infinite() { f64::INFINITY } else { 1.0 }
                } else {
                    b.powf(r)
                }
            })
            .collect()
    }

    /// Weights at `θ`.
    pub fn weights(&self, theta: &[f64]) -> Vec<f64> {
        self.weights_at_r(self.r(theta))
    }

    /// `λ Σ_g w_g φ_g` with weights at `r` (atoms with infinite weight
    /// contribute 0 at 0, ∞ otherwise).
    pub fn value_at_r(&self, theta: &[f64], r: f64) -> f64 {
        let w = self.weights_at_r(r);
        let mut phi = vec![0.0; self.atoms.len()];
        self.atoms.atoms(theta, &mut phi);
        self.lambda
            * phi
                .iter()
                .zip(&w)
                .map(|(&p, &wg)| if p == 0.0 { 0.0 } else { wg * p })
                .sum::<f64>()
    }

    /// `λ Σ_g w_g(θ) φ_g(θ)`.
    pub fn value(&self, theta: &[f64]) -> f64 {
        self.value_at_r(theta, self.r(theta))
    }

    /// Add the `μ`-gradient of the weights (joint RoC only, off the floor).
    fn add_joint_gradient(&self, theta: &[f64], grad: &mut [f64]) {
        let Exponent::RoC {
            mu,
            mu_tilde_norm,
            mode: RocMode::Joint,
        } = &self.exponent
        else {
            return;
        };
        let nm2: f64 = mu.iter().map(|&i| theta[i] * theta[i]).sum();
        let nm = nm2.sqrt();
        if mu_tilde_norm / nm <= 1.0 {
            return; // floor: r ≡ 1 locally
        }
        let w = self.weights(theta);
        let mut phi = vec![0.0; self.atoms.len()];
        self.atoms.atoms(theta, &mut phi);
        // ∂/∂r Σ_g w_g φ_g = Σ_g φ_g w_g ln b_g ;  ∂r/∂μ = −‖μ̃‖ μ/‖μ‖³
        let s: f64 = phi
            .iter()
            .zip(&w)
            .zip(&self.base)
            .map(|((&p, &wg), &b)| {
                if p == 0.0 || b == 0.0 || !b.is_finite() {
                    0.0
                } else {
                    p * wg * b.ln()
                }
            })
            .sum();
        let coef = -self.lambda * s * mu_tilde_norm / (nm2 * nm);
        for &i in mu {
            grad[i] += coef * theta[i];
        }
    }
}

/// Settings for [`prox_newton`].
#[derive(Clone, Debug)]
pub struct ProxConfig {
    /// Outer (Newton) iterations. Default 200.
    pub max_iter: usize,
    /// Stationarity tolerance on the gradient mapping (sum scale). Default
    /// 1e-6.
    pub tol: f64,
    /// FISTA iterations per subproblem. Default 1000 (the outer iteration
    /// tolerates inexact subproblem solutions).
    pub max_inner: usize,
    /// Sufficient-decrease constant. Default 1e-4.
    pub sigma: f64,
    /// Fixed-point iterations for RoC weights in `FixedPoint` mode.
    pub max_roc_iter: usize,
}

impl Default for ProxConfig {
    fn default() -> Self {
        Self {
            max_iter: 200,
            tol: 1e-6,
            max_inner: 1000,
            sigma: 1e-4,
            max_roc_iter: 50,
        }
    }
}

/// Outcome of [`prox_newton`].
#[derive(Clone, Debug)]
pub struct ProxResult {
    pub theta: Vec<f64>,
    /// `F(θ) = f(θ) + penalty`.
    pub objective: f64,
    /// `f(θ)`.
    pub smooth: f64,
    /// Stationarity measure (gradient mapping) at `θ`.
    pub kkt: f64,
    /// Exponent `r` of the weights at the solution.
    pub r: f64,
    pub n_iter: usize,
    pub converged: bool,
    /// The last Hessian of `f` used (row-major), to warm-start the next
    /// solve along a path.
    pub hessian: Vec<f64>,
}

/// Minimise `f(θ) + λ Σ_g b_g^{r} φ_g(θ)` from `theta0`.
pub fn prox_newton<A: PenaltyAtoms>(
    f: &dyn SmoothObjective,
    pen: &Penalty<A>,
    theta0: &[f64],
    cfg: &ProxConfig,
) -> ProxResult {
    prox_newton_warm(f, pen, theta0, cfg, None)
}

/// [`prox_newton`] starting with a Hessian from a nearby solve (e.g. the
/// previous point of a λ path), which is refreshed only when the model
/// stops predicting the decrease.
pub fn prox_newton_warm<A: PenaltyAtoms>(
    f: &dyn SmoothObjective,
    pen: &Penalty<A>,
    theta0: &[f64],
    cfg: &ProxConfig,
    hess0: Option<&[f64]>,
) -> ProxResult {
    match &pen.exponent {
        Exponent::RoC {
            mode: RocMode::FixedPoint,
            ..
        } => {
            let mut r = pen.r(theta0);
            let mut res = prox_newton_r(f, pen, theta0, cfg, Some(r), hess0);
            let mut total = res.n_iter;
            for _ in 0..cfg.max_roc_iter {
                let r_new = pen.r(&res.theta);
                if (r_new - r).abs() <= 1e-10 * r.max(1.0) {
                    break;
                }
                r = r_new;
                let h = res.hessian.clone();
                res = prox_newton_r(f, pen, &res.theta.clone(), cfg, Some(r), Some(&h));
                total += res.n_iter;
            }
            res.n_iter = total;
            res
        }
        _ => prox_newton_r(f, pen, theta0, cfg, None, hess0),
    }
}

/// One proximal-Newton solve; `r_fixed` freezes the exponent.
fn prox_newton_r<A: PenaltyAtoms>(
    f: &dyn SmoothObjective,
    pen: &Penalty<A>,
    theta0: &[f64],
    cfg: &ProxConfig,
    r_fixed: Option<f64>,
    hess0: Option<&[f64]>,
) -> ProxResult {
    let n = theta0.len();
    let r_at = |th: &[f64]| r_fixed.unwrap_or_else(|| pen.r(th));
    let value = |th: &[f64]| pen.value_at_r(th, r_at(th));
    let mut theta = theta0.to_vec();
    // infinite-weight atoms start at their constraint
    {
        let w = pen.weights_at_r(r_at(&theta));
        if w.iter().any(|v| v.is_infinite()) {
            let t: Vec<f64> = w
                .iter()
                .map(|&v| if v.is_infinite() { f64::INFINITY } else { 0.0 })
                .collect();
            pen.atoms.prox(&mut theta, &t);
        }
    }
    let mut grad = vec![0.0; n];
    let mut hess = vec![0.0; n * n];
    let (mut fs, mut since_hessian) = match hess0 {
        Some(h0) if h0.len() == n * n => {
            hess.copy_from_slice(h0);
            (f.value_grad(&theta, &mut grad), 1usize)
        }
        _ => (f.value_grad_hess(&theta, &mut grad, &mut hess), 0usize),
    };
    let mut obj = fs + value(&theta);
    // positive-definite model Hessian and its largest eigenvalue, recomputed
    // only with the Hessian
    let mut model_h: Option<(DMatrix<f64>, f64)> = None;
    let mut n_iter = 0;
    let mut kkt = f64::INFINITY;
    let mut converged = false;
    // consecutive accepted steps whose decrease is at the rounding level of F
    let mut flat = 0usize;
    let smooth_grad = |th: &[f64], g: &[f64]| -> Vec<f64> {
        let mut gh = g.to_vec();
        if r_fixed.is_none() {
            pen.add_joint_gradient(th, &mut gh);
        }
        gh
    };
    while obj.is_finite() {
        let w = pen.weights_at_r(r_at(&theta));
        let gh = smooth_grad(&theta, &grad);
        kkt = gradient_mapping(&theta, &gh, pen, &w);
        if kkt <= cfg.tol {
            converged = true;
            break;
        }
        if n_iter >= cfg.max_iter {
            break;
        }
        n_iter += 1;
        if model_h.is_none() {
            let h = positive_definite(&hess, n);
            let l = largest_eigenvalue(&h);
            model_h = Some((h, l));
        }
        let (h, l) = model_h.as_ref().unwrap();
        let z = solve_model(&theta, &gh, h, *l, pen, &w, cfg.max_inner);
        let delta: Vec<f64> = z.iter().zip(&theta).map(|(a, b)| a - b).collect();
        let pen_now = pen.lambda * weighted_atoms(&pen.atoms, &z, &w)
            - pen.lambda * weighted_atoms(&pen.atoms, &theta, &w);
        let dd = gh.iter().zip(&delta).map(|(a, b)| a * b).sum::<f64>() + pen_now;
        if !(dd < 0.0) {
            converged = kkt <= cfg.tol.sqrt();
            break;
        }
        let mut t = 1.0;
        let mut accepted = false;
        let mut cand = vec![0.0; n];
        let mut gc = vec![0.0; n];
        let mut f_cand = f64::NAN;
        for _ in 0..40 {
            for i in 0..n {
                cand[i] = theta[i] + t * delta[i];
            }
            if t < 1.0 {
                // keep the zero/tie pattern of the full step's constraints
                let tinf: Vec<f64> = w
                    .iter()
                    .map(|&v| if v.is_infinite() { f64::INFINITY } else { 0.0 })
                    .collect();
                if tinf.iter().any(|v| v.is_infinite()) {
                    pen.atoms.prox(&mut cand, &tinf);
                }
            }
            let fc = f.value_grad(&cand, &mut gc);
            let oc = fc + value(&cand);
            if oc.is_finite() && oc <= obj + cfg.sigma * t * dd {
                accepted = true;
                f_cand = fc;
                break;
            }
            t *= 0.5;
        }
        if !accepted {
            if since_hessian == 0 {
                break;
            }
            // a stale Hessian may be the problem: refresh and retry
            fs = f.value_grad_hess(&theta, &mut grad, &mut hess);
            since_hessian = 0;
            model_h = None;
            continue;
        }
        let new_obj = f_cand + value(&cand);
        // Stalled at the tolerance: steps that no longer decrease F beyond
        // rounding, with the residual already below √tol (an atom flipping
        // at its threshold). Without this the loop refreshes the Hessian
        // every iteration until max_iter (5 s at d = 5, n = 2000).
        if obj - new_obj <= FLAT_DECREASE * obj.abs().max(1.0) && kkt <= cfg.tol.sqrt() {
            flat += 1;
            if flat >= FLAT_STEPS {
                theta.copy_from_slice(&cand);
                fs = f_cand;
                grad.copy_from_slice(&gc);
                obj = new_obj;
                let gh = smooth_grad(&theta, &grad);
                kkt = gradient_mapping(&theta, &gh, pen, &pen.weights_at_r(r_at(&theta)));
                converged = kkt <= cfg.tol.sqrt();
                break;
            }
        } else {
            flat = 0;
        }
        // full steps whose decrease matches the model keep the Hessian (it
        // changes little along a warm-started path); otherwise refresh it
        let good = t == 1.0 && (obj - new_obj) >= 0.5 * (-dd);
        theta.copy_from_slice(&cand);
        if good && since_hessian < HESSIAN_REUSE {
            grad.copy_from_slice(&gc);
            fs = f_cand;
            since_hessian += 1;
        } else {
            fs = f.value_grad_hess(&theta, &mut grad, &mut hess);
            since_hessian = 0;
            model_h = None;
        }
        obj = fs + value(&theta);
    }
    ProxResult {
        r: r_at(&theta),
        theta,
        objective: obj,
        smooth: fs,
        kkt,
        n_iter,
        converged,
        hessian: hess,
    }
}

/// Newton steps that may reuse one Hessian.
const HESSIAN_REUSE: usize = 5;
/// A decrease of `F` below this relative size counts as rounding.
const FLAT_DECREASE: f64 = 1e-13;
/// Consecutive such steps after which a solve near the tolerance stops.
const FLAT_STEPS: usize = 3;

fn weighted_atoms<A: PenaltyAtoms>(atoms: &A, theta: &[f64], w: &[f64]) -> f64 {
    let mut phi = vec![0.0; atoms.len()];
    atoms.atoms(theta, &mut phi);
    phi.iter()
        .zip(w)
        .map(|(&p, &wg)| {
            if p == 0.0 || !wg.is_finite() {
                0.0
            } else {
                wg * p
            }
        })
        .sum()
}

/// `‖θ − prox_{τP}(θ − τ g)‖_∞ / τ` for a small `τ`: the minimal-norm
/// element of `g + ∂P(θ)` for convex atoms (the KKT residual).
fn gradient_mapping<A: PenaltyAtoms>(theta: &[f64], g: &[f64], pen: &Penalty<A>, w: &[f64]) -> f64 {
    let tau = 1e-7 / (1.0 + g.iter().fold(0.0f64, |a, &b| a.max(b.abs())));
    let mut v: Vec<f64> = theta.iter().zip(g).map(|(a, b)| a - tau * b).collect();
    let t: Vec<f64> = w.iter().map(|&wg| tau * pen.lambda * wg).collect();
    pen.atoms.prox(&mut v, &t);
    theta
        .iter()
        .zip(&v)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max)
        / tau
}

/// Largest eigenvalue of a symmetric positive-definite matrix by power
/// iteration (relative change < 1e-6), with a 2% safety margin (an
/// over-estimate only shortens FISTA's step).
fn largest_eigenvalue(h: &DMatrix<f64>) -> f64 {
    let n = h.nrows();
    if n <= 8 {
        return SymmetricEigen::new(h.clone()).eigenvalues.max();
    }
    let mut v = DVector::from_fn(n, |i, _| 1.0 + 0.1 * ((i * 7 % 11) as f64));
    v /= v.norm();
    let mut lam = 0.0;
    for _ in 0..200 {
        let hv = h * &v;
        let new = v.dot(&hv);
        let nrm = hv.norm();
        if nrm == 0.0 {
            break;
        }
        v = hv / nrm;
        if (new - lam).abs() <= 1e-6 * new.abs() {
            lam = new;
            break;
        }
        lam = new;
    }
    1.02 * lam
}

/// Symmetrised `H`, shifted by `τ I` if needed so that it is positive
/// definite (`τ` grows geometrically from `1e-8·max|H_ii|`).
fn positive_definite(hess: &[f64], n: usize) -> DMatrix<f64> {
    let mut h = DMatrix::from_row_slice(n, n, hess);
    h = (&h + h.transpose()) * 0.5;
    if h.clone().cholesky().is_some() {
        return h;
    }
    let scale = (0..n)
        .map(|i| h[(i, i)].abs())
        .fold(0.0f64, f64::max)
        .max(1e-12);
    let mut tau = 1e-8 * scale;
    loop {
        let mut ht = h.clone();
        for i in 0..n {
            ht[(i, i)] += tau;
        }
        if ht.clone().cholesky().is_some() {
            return ht;
        }
        tau *= 10.0;
    }
}

/// Minimise `gᵀ(z−θ) + ½(z−θ)ᵀH(z−θ) + λ Σ w_g φ_g(z)` by FISTA with
/// adaptive restarts (O'Donoghue & Candès).
fn solve_model<A: PenaltyAtoms>(
    theta: &[f64],
    g: &[f64],
    h: &DMatrix<f64>,
    l: f64,
    pen: &Penalty<A>,
    w: &[f64],
    max_inner: usize,
) -> Vec<f64> {
    let step = 1.0 / l.max(1e-300);
    let t: Vec<f64> = w.iter().map(|&wg| step * pen.lambda * wg).collect();
    let th = DVector::from_column_slice(theta);
    let gv = DVector::from_column_slice(g);
    let prox = |v: &mut DVector<f64>| pen.atoms.prox(v.as_mut_slice(), &t);
    let model = |z: &DVector<f64>| -> f64 {
        let dz = z - &th;
        gv.dot(&dz)
            + 0.5 * dz.dot(&(h * &dz))
            + pen.lambda * weighted_atoms(&pen.atoms, z.as_slice(), w)
    };
    let mut z = th.clone();
    prox(&mut z);
    let mut y = z.clone();
    let mut tk = 1.0f64;
    let mut m_prev = model(&z);
    for it in 0..max_inner {
        let grad_y = &gv + h * (&y - &th);
        let mut z_new = &y - grad_y * step;
        prox(&mut z_new);
        let m_new = model(&z_new);
        if m_new > m_prev {
            tk = 1.0;
            y = z.clone();
            continue;
        }
        let change = (&z_new - &z).amax();
        let t_next = 0.5 * (1.0 + (1.0 + 4.0 * tk * tk).sqrt());
        y = &z_new + (&z_new - &z) * ((tk - 1.0) / t_next);
        tk = t_next;
        z = z_new;
        m_prev = m_new;
        if change <= 1e-10 * (1.0 + z.amax()) && it > 2 {
            break;
        }
    }
    z.as_slice().to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// f(θ) = ½‖Aθ − b‖²
    struct LeastSquares {
        a: DMatrix<f64>,
        b: DVector<f64>,
    }
    impl SmoothObjective for LeastSquares {
        fn value_grad(&self, theta: &[f64], grad: &mut [f64]) -> f64 {
            let t = DVector::from_column_slice(theta);
            let r = &self.a * t - &self.b;
            let g = self.a.transpose() * &r;
            grad.copy_from_slice(g.as_slice());
            0.5 * r.dot(&r)
        }
        fn value_grad_hess(&self, theta: &[f64], grad: &mut [f64], hess: &mut [f64]) -> f64 {
            let f = self.value_grad(theta, grad);
            let h = self.a.transpose() * &self.a;
            let n = theta.len();
            for i in 0..n {
                for j in 0..n {
                    hess[i * n + j] = h[(i, j)];
                }
            }
            f
        }
    }

    fn singletons(n: usize) -> GroupNorms {
        GroupNorms {
            groups: (0..n).map(|i| vec![i]).collect(),
        }
    }

    #[test]
    fn orthogonal_lasso_is_soft_thresholding() {
        let b = DVector::from_vec(vec![3.0, -0.5, 1.2, 0.0, -2.0]);
        let f = LeastSquares {
            a: DMatrix::identity(5, 5),
            b,
        };
        let pen = Penalty {
            atoms: singletons(5),
            base: vec![1.0; 5],
            exponent: Exponent::Fixed(0.0),
            lambda: 1.0,
        };
        let r = prox_newton(&f, &pen, &[0.0; 5], &ProxConfig::default());
        let want = [2.0, 0.0, 0.2, 0.0, -1.0];
        assert!(r.converged);
        for (a, b) in r.theta.iter().zip(&want) {
            assert!((a - b).abs() < 1e-10, "{:?}", r.theta);
        }
        assert_eq!(r.theta[1], 0.0);
        assert_eq!(r.theta[3], 0.0);
    }

    #[test]
    fn orthogonal_group_lasso_is_block_soft_thresholding() {
        let b = DVector::from_vec(vec![3.0, 4.0, 0.3, -0.4, 1.0]);
        let f = LeastSquares {
            a: DMatrix::identity(5, 5),
            b,
        };
        let pen = Penalty {
            atoms: GroupNorms {
                groups: vec![vec![0, 1], vec![2, 3]],
            },
            base: vec![1.0, 1.0],
            exponent: Exponent::Fixed(0.0),
            lambda: 1.0,
        };
        let r = prox_newton(&f, &pen, &[1.0; 5], &ProxConfig::default());
        let want = [2.4, 3.2, 0.0, 0.0, 1.0];
        for (a, b) in r.theta.iter().zip(&want) {
            assert!((a - b).abs() < 1e-10, "{:?}", r.theta);
        }
    }

    #[test]
    fn correlated_lasso_satisfies_kkt() {
        let a = DMatrix::from_fn(30, 6, |i, j| {
            (((i * 7 + j * 13) % 11) as f64 - 5.0) / 3.0 + if i % 6 == j { 1.0 } else { 0.0 }
        });
        let truth = DVector::from_vec(vec![2.0, 0.0, -1.0, 0.0, 0.0, 0.5]);
        let b = &a * &truth + DVector::from_fn(30, |i, _| 0.1 * ((i as f64) * 1.7).sin());
        let f = LeastSquares { a, b };
        let pen = Penalty {
            atoms: singletons(6),
            base: vec![1.0; 6],
            exponent: Exponent::Fixed(0.0),
            lambda: 3.0,
        };
        let r = prox_newton(&f, &pen, &[0.0; 6], &ProxConfig::default());
        assert!(r.converged, "kkt {}", r.kkt);
        assert!(r.theta.contains(&0.0), "{:?}", r.theta);
    }

    #[test]
    fn infinite_weights_pin_and_zero_weights_free() {
        let b = DVector::from_vec(vec![3.0, 4.0, 0.3, -0.4]);
        let f = LeastSquares {
            a: DMatrix::identity(4, 4),
            b,
        };
        let pen = Penalty {
            atoms: GroupNorms {
                groups: vec![vec![0, 1], vec![2, 3]],
            },
            base: vec![f64::INFINITY, 0.0],
            exponent: Exponent::Fixed(1.0),
            lambda: 1.0,
        };
        let r = prox_newton(&f, &pen, &[1.0; 4], &ProxConfig::default());
        let want = [0.0, 0.0, 0.3, -0.4];
        for (a, b) in r.theta.iter().zip(&want) {
            assert!((a - b).abs() < 1e-10, "{:?}", r.theta);
        }
    }

    #[test]
    fn roc_joint_gradient_matches_finite_differences() {
        let pen = Penalty {
            atoms: singletons(4),
            base: vec![0.0, 0.0, 1.0 / 0.8, 1.0 / 0.05],
            exponent: Exponent::RoC {
                mu: vec![0, 1],
                mu_tilde_norm: 5f64.sqrt() * 1.3,
                mode: RocMode::Joint,
            },
            lambda: 0.7,
        };
        let theta = [1.7, 1.1, 0.6, 0.2];
        let mut g = [0.0; 4];
        pen.add_joint_gradient(&theta, &mut g);
        for i in 0..2 {
            let h = 1e-6;
            let (mut tp, mut tm) = (theta, theta);
            tp[i] += h;
            tm[i] -= h;
            let fd = (pen.value(&tp) - pen.value(&tm)) / (2.0 * h);
            assert!((fd - g[i]).abs() < 1e-7, "i={i}: fd {fd} vs {}", g[i]);
        }
    }

    #[test]
    fn sparse_group_prox_matches_brute_force() {
        let sg = SparseGroupNorms {
            groups: vec![vec![0, 1, 2], vec![3, 4]],
        };
        let v0 = [1.2, -0.3, 0.05, 0.4, -0.2];
        let t = [0.3, 0.6, 0.1, 0.1, 0.1, 0.05, 0.05];
        let mut v = v0.to_vec();
        sg.prox(&mut v, &t);
        let obj = |z: &[f64]| {
            let mut a = vec![0.0; 7];
            sg.atoms(z, &mut a);
            0.5 * z.iter().zip(&v0).map(|(x, y)| (x - y).powi(2)).sum::<f64>()
                + a.iter().zip(&t).map(|(x, y)| x * y).sum::<f64>()
        };
        let best = obj(&v);
        let mut seed = 5u64;
        for _ in 0..20000 {
            let mut w = v.clone();
            for x in w.iter_mut() {
                seed = seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                *x += ((seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5) * 0.05;
            }
            assert!(obj(&w) >= best - 1e-12);
        }
        // the second group is zeroed as a whole, the third entry by the L1 part
        assert_eq!(&v[3..], &[0.0, 0.0]);
        assert_eq!(v[2], 0.0);
    }

    #[test]
    fn eigengap_prox_infinite_thresholds_tie_exactly() {
        let gaps = EigenGaps { p: 4, offset: 0 };
        let mut v = vec![0.9, -0.4, 0.3, 0.2, -0.6, 0.35, 0.1, -0.25, 0.7];
        gaps.prox(&mut v, &[f64::INFINITY, 0.0, f64::INFINITY]);
        let mut a = [0.0; 3];
        gaps.atoms(&v, &mut a);
        assert!(a[0] < 1e-14 && a[2] < 1e-14 && a[1] > 0.1, "{a:?}");
    }

    #[test]
    fn pava_and_eigengap_prox() {
        assert_eq!(
            pava_nondecreasing(&[1.0, 3.0, 2.0, 4.0]),
            vec![1.0, 2.5, 2.5, 4.0]
        );
        assert_eq!(pava_nondecreasing(&[3.0, 2.0, 1.0]), vec![2.0, 2.0, 2.0]);
        // prox of the gap penalty against brute-force search over S (p = 3)
        let p = 3;
        let gaps = EigenGaps { p, offset: 0 };
        let s0 = [0.9, -0.4, 0.3, 0.2, -0.6];
        for t in [[0.05, 0.4], [0.5, 0.01], [2.0, 2.0]] {
            let mut v = s0.to_vec();
            gaps.prox(&mut v, &t);
            let obj = |s: &[f64]| {
                let mut a = [0.0; 2];
                gaps.atoms(s, &mut a);
                0.5 * s.iter().zip(&s0).map(|(x, y)| (x - y).powi(2)).sum::<f64>()
                    + t[0] * a[0]
                    + t[1] * a[1]
            };
            let best = obj(&v);
            // no random perturbation does better
            let mut seed = 17u64;
            for _ in 0..4000 {
                let mut w = v.clone();
                for x in w.iter_mut() {
                    seed = seed
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    *x += ((seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5) * 0.02;
                }
                assert!(obj(&w) >= best - 1e-12, "t {t:?}: prox not optimal");
            }
            // the trace (sum of eigenvalues) is kept: the matrix stays traceless
            let m = crate::logtangent::s_to_matrix(&v, p);
            assert!((m[0] + m[4] + m[8]).abs() < 1e-12);
        }
    }
}
