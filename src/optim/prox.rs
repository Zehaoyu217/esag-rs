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
    /// Disjoint blocks of parameters, each with the atoms that act on it
    /// and on nothing else (`(parameters, atoms)`), or `None` if the atoms
    /// do not separate this way. With blocks, the model of each proximal
    /// Newton step is solved by block coordinate descent
    /// ([`PenaltyAtoms::prox_block`]); without, by FISTA.
    fn blocks(&self) -> Option<Vec<(Vec<usize>, Vec<usize>)>> {
        None
    }
    /// [`PenaltyAtoms::prox`] on the parameters of block `b` of
    /// [`PenaltyAtoms::blocks`] only (`t` indexed by atom, as in `prox`).
    fn prox_block(&self, _b: usize, _v: &mut [f64], _t: &[f64]) {
        unreachable!("prox_block without blocks")
    }
    /// Whether the only atom of block `b` is the Euclidean norm of its
    /// parameters (the model is then minimized over the block exactly).
    fn block_is_norm(&self, _b: usize) -> bool {
        false
    }
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
    fn blocks(&self) -> Option<Vec<(Vec<usize>, Vec<usize>)>> {
        Some(
            self.groups
                .iter()
                .enumerate()
                .map(|(b, g)| (g.clone(), vec![b]))
                .collect(),
        )
    }
    fn prox_block(&self, b: usize, v: &mut [f64], t: &[f64]) {
        let g = &self.groups[b];
        let nrm = g.iter().map(|&i| v[i] * v[i]).sum::<f64>().sqrt();
        let scale = if !(t[b] < nrm) { 0.0 } else { 1.0 - t[b] / nrm };
        for &i in g {
            v[i] *= scale;
        }
    }
    fn block_is_norm(&self, _b: usize) -> bool {
        true
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
    fn blocks(&self) -> Option<Vec<(Vec<usize>, Vec<usize>)>> {
        let mut e = self.n_groups();
        Some(
            self.groups
                .iter()
                .enumerate()
                .map(|(b, g)| {
                    let mut atoms = vec![b];
                    atoms.extend(e..e + g.len());
                    e += g.len();
                    (g.clone(), atoms)
                })
                .collect(),
        )
    }
    fn prox_block(&self, b: usize, v: &mut [f64], t: &[f64]) {
        let e0 = self.n_groups() + self.groups[..b].iter().map(Vec::len).sum::<usize>();
        let g = &self.groups[b];
        for (k, &i) in g.iter().enumerate() {
            let te = t[e0 + k];
            v[i] = if !(te < v[i].abs()) {
                0.0
            } else {
                v[i] - te.copysign(v[i])
            };
        }
        let nrm = g.iter().map(|&i| v[i] * v[i]).sum::<f64>().sqrt();
        let scale = if !(t[b] < nrm) { 0.0 } else { 1.0 - t[b] / nrm };
        for &i in g {
            v[i] *= scale;
        }
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
    // positive-definite model Hessian, its largest eigenvalue and whether the
    // curvature floor was applied; recomputed with the Hessian or when the
    // floor changes
    let mut model_h: Option<(DMatrix<f64>, f64, bool)> = None;
    // the floor of the free block's eigenvalues (indefinite Hessians only),
    // adapted to the agreement between the model and the objective
    let mut floor = CURVATURE_FLOOR;
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
            #[cfg(test)]
            let clock = std::time::Instant::now();
            let inactive = inactive_coordinates(&theta, &gh, pen, &w, &hess, n);
            let (h, floored) = positive_definite_masked(&hess, n, &inactive, floor);
            #[cfg(test)]
            let clock = crate::regression::stage::lap("prox_model", clock);
            let l = largest_eigenvalue(&h);
            #[cfg(test)]
            crate::regression::stage::lap("prox_lmax", clock);
            model_h = Some((h, l, floored));
        }
        // (a copy: the floor adaptation below may rebuild the model)
        let (h, l, floored) = model_h.clone().unwrap();
        #[cfg(test)]
        let clock = std::time::Instant::now();
        let z = solve_model(&theta, &gh, &h, l, pen, &w, cfg.max_inner);
        #[cfg(test)]
        crate::regression::stage::lap("prox_inner", clock);
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
            // a model too flat on the free block (floored curvature) may be
            // the problem: raise the floor and retry with the same Hessian
            if floored && floor < FLOOR_MAX {
                floor = (floor * 10.0).min(FLOOR_MAX);
                model_h = None;
                continue;
            }
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
        // the model's predicted decrease against the actual one: on a floored
        // model a decrease far above the prediction means the floor is too
        // high along the step (the smooth part has negative curvature there),
        // a cut step or a decrease far below it that it is too low
        if floored {
            let dv = DVector::from_column_slice(&delta);
            let pred = -dd - 0.5 * t * dv.dot(&(&h * &dv));
            let ratio = if pred > 0.0 {
                (obj - new_obj) / (t * pred)
            } else {
                1.0
            };
            if (t < 1.0 || ratio < 0.5) && floor < FLOOR_MAX {
                floor = (floor * 10.0).min(FLOOR_MAX);
                model_h = None;
            } else if t == 1.0 && ratio > 1.5 && floor > FLOOR_MIN {
                floor = (floor * 0.1).max(FLOOR_MIN);
                model_h = None;
            }
        }
        if std::env::var_os("ESAG_PROX_DEBUG").is_some() {
            let lmin = h
                .clone()
                .symmetric_eigenvalues()
                .iter()
                .cloned()
                .fold(f64::INFINITY, f64::min);
            let step = delta.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
            let active = {
                let mut phi = vec![0.0; pen.atoms.len()];
                pen.atoms.atoms(&cand, &mut phi);
                phi.iter().filter(|v| **v > 1e-12).count()
            };
            // the raw (observed) Hessian: its smallest eigenvalue and the
            // number of negative ones; the curvature along the step in the
            // model and in the raw Hessian
            let raw = {
                let m = DMatrix::from_row_slice(n, n, &hess);
                (&m + m.transpose()) * 0.5
            };
            let raw_eig = raw.clone().symmetric_eigenvalues();
            let raw_min = raw_eig.iter().cloned().fold(f64::INFINITY, f64::min);
            let raw_neg = raw_eig.iter().filter(|v| **v < 0.0).count();
            let dv = DVector::from_column_slice(&delta);
            let curv_model = dv.dot(&(&h * &dv));
            let curv_raw = dv.dot(&(&raw * &dv));
            eprintln!(
                "    it {n_iter}: kkt {kkt:.2e} t {t:.3} dd {dd:.3e} decrease {:.3e} |step| {step:.2e} model eig [{lmin:.2e}, {l:.2e}] raw min {raw_min:.2e} ({raw_neg} neg) δᵀHδ model {curv_model:.2e} raw {curv_raw:.2e} floor {floor:.0e} stale {since_hessian} active {active}",
                obj - new_obj
            );
        }
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
    let _ = floor;
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
/// Initial floor of the model Hessian's eigenvalues on the free coordinates,
/// as a fraction of the largest (an indefinite observed Hessian only); the
/// solve adapts it within [`FLOOR_MIN`], [`FLOOR_MAX`].
const CURVATURE_FLOOR: f64 = 1e-3;
const FLOOR_MIN: f64 = 1e-7;
const FLOOR_MAX: f64 = 1.0;
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
/// The coordinates the proximal step keeps at zero from `theta`: those at
/// zero whose proximal gradient step (any step size: the test does not
/// depend on it for the norm atoms) returns zero. Their atoms are inactive
/// at this iterate; the model Hessian treats them diagonally.
fn inactive_coordinates<A: PenaltyAtoms>(
    theta: &[f64],
    g: &[f64],
    pen: &Penalty<A>,
    w: &[f64],
    hess: &[f64],
    n: usize,
) -> Vec<bool> {
    let scale = (0..n)
        .map(|i| hess[i * n + i].abs())
        .fold(0.0f64, f64::max)
        .max(1e-12);
    let tau = 1.0 / scale;
    let mut z: Vec<f64> = theta.iter().zip(g).map(|(t, gi)| t - tau * gi).collect();
    let t: Vec<f64> = w.iter().map(|&wg| tau * pen.lambda * wg).collect();
    pen.atoms.prox(&mut z, &t);
    (0..n).map(|i| theta[i] == 0.0 && z[i] == 0.0).collect()
}

/// A positive-definite model Hessian. A Hessian that is already positive
/// definite is used as it is. Otherwise the coordinates in `inactive` (kept
/// at zero by the proximal step) get a diagonal model, `|H_ii|` floored, and
/// the block of the other coordinates is made positive definite by the
/// saddle-free rule (eigenvalues replaced by their absolute values, floored
/// at `1e−8` of the largest), so that the step on the free coordinates is
/// Newton's with their own curvature. The previous rule, adding `τI` to the
/// whole Hessian until the Cholesky factorization succeeds, inflated every
/// direction by the size of the most negative eigenvalue; the saddle-free
/// rule on the whole Hessian still polluted the free block through the
/// flipped eigenvectors' components there. With the eigen-chart
/// likelihoods, whose observed Hessian has dozens of negative eigenvalues
/// in the zeroed shape-slope directions at the large-`λ` points of a path,
/// either turned the Newton step into a short gradient step and the solves
/// crawled for 200 iterations (a minute per path at `d = 8`, against
/// seconds for the log-tangent, whose Hessian is positive definite).
fn positive_definite_masked(
    hess: &[f64],
    n: usize,
    inactive: &[bool],
    floor: f64,
) -> (DMatrix<f64>, bool) {
    let mut h = DMatrix::from_row_slice(n, n, hess);
    h = (&h + h.transpose()) * 0.5;
    if h.clone().cholesky().is_some() {
        return (h, false);
    }
    let scale = (0..n)
        .map(|i| h[(i, i)].abs())
        .fold(0.0f64, f64::max)
        .max(1e-12);
    let free: Vec<usize> = (0..n).filter(|&i| !inactive[i]).collect();
    let mut out = DMatrix::zeros(n, n);
    for i in 0..n {
        if inactive[i] {
            out[(i, i)] = h[(i, i)].abs().max(1e-8 * scale);
        }
    }
    let mut floored = false;
    if !free.is_empty() {
        let m = free.len();
        let sub = DMatrix::from_fn(m, m, |a, b| h[(free[a], free[b])]);
        let fixed = if sub.clone().cholesky().is_some() {
            sub
        } else {
            floored = true;
            saddle_free(&sub, floor)
        };
        for a in 0..m {
            for b in 0..m {
                out[(free[a], free[b])] = fixed[(a, b)];
            }
        }
    }
    if out.clone().cholesky().is_some() {
        return (out, floored);
    }
    // rounding left it indefinite: the diagonal shift
    let mut tau = 1e-8 * scale;
    loop {
        let mut ht = out.clone();
        for i in 0..n {
            ht[(i, i)] += tau;
        }
        if ht.clone().cholesky().is_some() {
            return (ht, floored);
        }
        tau *= 10.0;
    }
}

/// Negative and small eigenvalues raised to `floor` times the largest.
/// Along a direction of negative curvature of the smooth part the model
/// then prescribes a long step, which the line search cuts if the penalty
/// does not make the objective rise; replacing the eigenvalue by its
/// absolute value instead (the saddle-free rule) prescribes a short one and
/// the solves crawl. The solve adapts `floor` to the agreement between the
/// model's predicted decrease and the actual one.
fn saddle_free(h: &DMatrix<f64>, floor: f64) -> DMatrix<f64> {
    let eig = nalgebra::SymmetricEigen::new(h.clone());
    let lmax = eig
        .eigenvalues
        .iter()
        .map(|v| v.abs())
        .fold(0.0f64, f64::max)
        .max(1e-12);
    let floor = floor * lmax;
    let d = DMatrix::from_diagonal(&eig.eigenvalues.map(|v| v.max(floor)));
    let fixed = &eig.eigenvectors * d * eig.eigenvectors.transpose();
    (&fixed + fixed.transpose()) * 0.5
}

/// `u = argmin ½uᵀHu − cᵀu + τ‖u‖₂` for symmetric positive definite `H`
/// (eigen-decomposition `eig`), written into `z` at `coords`: `u = 0` if
/// `‖c‖ ≤ τ`, else `u = (H + sI)⁻¹c` with `s > 0` the root of
/// `s ‖(H + sI)⁻¹c‖ = τ` (increasing in `s`), by Newton's method safeguarded
/// by bisection.
fn group_block_min(
    eig: &SymmetricEigen<f64, nalgebra::Dyn>,
    c: &[f64],
    tau: f64,
    coords: &[usize],
    z: &mut [f64],
) {
    let cn = c.iter().map(|v| v * v).sum::<f64>().sqrt();
    if !(cn > tau) {
        for &i in coords {
            z[i] = 0.0;
        }
        return;
    }
    let (v, e) = (&eig.eigenvectors, &eig.eigenvalues);
    let m = coords.len();
    // c in the eigenbasis
    let ce: Vec<f64> = (0..m)
        .map(|k| (0..m).map(|a| v[(a, k)] * c[a]).sum())
        .collect();
    // ψ(s) = s ‖(H + sI)⁻¹ c‖ − τ and ψ'(s)
    let psi = |s: f64| -> (f64, f64) {
        let mut q = 0.0;
        let mut dq = 0.0;
        for k in 0..m {
            let f = s / (e[k] + s);
            q += ce[k] * ce[k] * f * f;
            dq += ce[k] * ce[k] * f * e[k] / ((e[k] + s) * (e[k] + s));
        }
        let nq = q.sqrt();
        (nq - tau, if nq > 0.0 { dq / nq } else { 0.0 })
    };
    // bracket: ψ(0) = −τ < 0, ψ(s) → ‖c‖ − τ > 0; s ≥ τ e_max/(‖c‖ − τ) suffices
    let emax = e.iter().fold(0.0f64, |a, &b| a.max(b));
    let (mut lo, mut hi) = (0.0, tau * emax.max(1e-300) / (cn - tau) + tau);
    while psi(hi).0 < 0.0 {
        hi *= 2.0;
    }
    let mut s = 0.5 * (lo + hi);
    for _ in 0..100 {
        let (f, df) = psi(s);
        if f > 0.0 {
            hi = s;
        } else {
            lo = s;
        }
        let next = if df > 0.0 { s - f / df } else { f64::NAN };
        let next = if next > lo && next < hi {
            next
        } else {
            0.5 * (lo + hi)
        };
        if (next - s).abs() <= 1e-15 * s.max(1e-300) {
            s = next;
            break;
        }
        s = next;
    }
    for (a, &i) in coords.iter().enumerate() {
        z[i] = (0..m).map(|k| v[(a, k)] * ce[k] / (e[k] + s)).sum();
    }
}

/// The model of [`solve_model`] by block coordinate descent, for atoms on
/// disjoint blocks ([`PenaltyAtoms::blocks`]). The coordinates no positive
/// finite weight penalizes (outside every block, or in blocks whose atoms
/// all have weight 0) are minimized exactly together, by the Cholesky
/// factor of their block of `H`; each penalized block `B` takes the
/// proximal step of the majorizer `L_B I ⪰ H_BB` (`L_B` the largest
/// eigenvalue of `H_BB`), so every step lowers the model; blocks pinned at
/// zero (weight ∞) stay there. After a sweep of every block, the blocks
/// away from zero are swept until they settle, then every block again,
/// until a full sweep moves no coordinate by more than `1e-12 (1 + max|z|)`
/// (glmnet's active-set cycles). The model is strictly convex, so the limit
/// is FISTA's. At d = 8 with 210 coefficients a path's model solves took
/// about 1/20 of FISTA's operations (`penalized_speed` in
/// `regression/fit_profile.rs`).
fn solve_model_blocks<A: PenaltyAtoms>(
    theta: &[f64],
    g: &[f64],
    h: &DMatrix<f64>,
    pen: &Penalty<A>,
    w: &[f64],
    blocks: &[(Vec<usize>, Vec<usize>)],
    max_sweeps: usize,
) -> Vec<f64> {
    let n = theta.len();
    let mut z = theta.to_vec();
    let mut in_block = vec![false; n];
    // per atom, the threshold of its block's proximal step
    let mut t = vec![0.0; pen.atoms.len()];
    // per penalized block: L_B, and for a norm atom on several parameters
    // the eigen-decomposition of H_BB (the exact block minimizer)
    let mut penalized: Vec<(usize, f64)> = vec![];
    let mut exact: Vec<Option<(SymmetricEigen<f64, nalgebra::Dyn>, f64)>> = vec![];
    for (b, (coords, atoms)) in blocks.iter().enumerate() {
        if atoms.iter().all(|&a| w[a] == 0.0) {
            continue;
        }
        for &i in coords {
            in_block[i] = true;
        }
        if atoms.iter().all(|&a| w[a].is_infinite()) {
            for &i in coords {
                z[i] = 0.0;
            }
            continue;
        }
        let (lb, eig) = if coords.len() == 1 {
            (h[(coords[0], coords[0])], None)
        } else {
            let sub =
                DMatrix::from_fn(coords.len(), coords.len(), |a, c| h[(coords[a], coords[c])]);
            let eig = SymmetricEigen::new(sub);
            (eig.eigenvalues.max(), Some(eig))
        };
        let lb = lb.max(1e-300);
        for &a in atoms {
            t[a] = pen.lambda * w[a] / lb;
        }
        let norm = atoms.len() == 1 && pen.atoms.block_is_norm(b) && w[atoms[0]].is_finite();
        exact.push(eig.filter(|_| norm).map(|e| (e, pen.lambda * w[atoms[0]])));
        penalized.push((b, lb));
    }
    let free: Vec<usize> = (0..n).filter(|&i| !in_block[i]).collect();
    let chol_free = (!free.is_empty())
        .then(|| DMatrix::from_fn(free.len(), free.len(), |a, c| h[(free[a], free[c])]).cholesky())
        .flatten();
    // r = g + H (z − θ), the model's smooth gradient at z
    let dz = DVector::from_iterator(n, z.iter().zip(theta).map(|(a, b)| a - b));
    let mut r: Vec<f64> = (DVector::from_column_slice(g) + h * dz)
        .iter()
        .copied()
        .collect();
    let hs = h.as_slice();
    let add_col = |r: &mut [f64], j: usize, delta: f64| {
        for (ri, hij) in r.iter_mut().zip(&hs[j * n..(j + 1) * n]) {
            *ri += hij * delta;
        }
    };
    let mut old = vec![0.0; n];
    let mut sweeps = 0;
    let mut full = true;
    let mut active: Vec<bool> = vec![true; penalized.len()];
    while sweeps < max_sweeps {
        sweeps += 1;
        #[cfg(test)]
        crate::regression::stage::INNER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let mut moved = 0.0f64;
        if let Some(ch) = &chol_free {
            let rf = DVector::from_iterator(free.len(), free.iter().map(|&i| r[i]));
            let step = ch.solve(&rf);
            for (a, &i) in free.iter().enumerate() {
                let delta = -step[a];
                z[i] += delta;
                moved = moved.max(delta.abs());
                add_col(&mut r, i, delta);
            }
        }
        for (k, &(b, lb)) in penalized.iter().enumerate() {
            if !full && !active[k] {
                continue;
            }
            let coords = &blocks[b].0;
            for &i in coords {
                old[i] = z[i];
            }
            if let Some((eig, tau)) = &exact[k] {
                // argmin_u ½uᵀH_BB u − cᵀu + τ‖u‖ with c = H_BB z_B − r_B
                let c: Vec<f64> = coords
                    .iter()
                    .map(|&i| coords.iter().map(|&j| h[(i, j)] * old[j]).sum::<f64>() - r[i])
                    .collect();
                group_block_min(eig, &c, *tau, coords, &mut z);
            } else {
                for &i in coords {
                    z[i] -= r[i] / lb;
                }
                pen.atoms.prox_block(b, &mut z, &t);
            }
            let mut nonzero = false;
            for &i in coords {
                let delta = z[i] - old[i];
                if delta != 0.0 {
                    moved = moved.max(delta.abs());
                    add_col(&mut r, i, delta);
                }
                nonzero |= z[i] != 0.0;
            }
            active[k] = nonzero;
        }
        let settled = moved <= 1e-12 * (1.0 + z.iter().fold(0.0f64, |m, v| m.max(v.abs())));
        if settled {
            if full {
                break;
            }
            full = true;
        } else if full {
            full = false;
        }
    }
    z
}

/// Minimise `gᵀ(z−θ) + ½(z−θ)ᵀH(z−θ) + λ Σ w_g φ_g(z)`: by block
/// coordinate descent where the atoms act on disjoint blocks
/// ([`solve_model_blocks`]), else by FISTA ([`solve_model_fista`]).
fn solve_model<A: PenaltyAtoms>(
    theta: &[f64],
    g: &[f64],
    h: &DMatrix<f64>,
    l: f64,
    pen: &Penalty<A>,
    w: &[f64],
    max_inner: usize,
) -> Vec<f64> {
    match pen.atoms.blocks() {
        Some(blocks) => solve_model_blocks(theta, g, h, pen, w, &blocks, max_inner),
        None => solve_model_fista(theta, g, h, l, pen, w, max_inner),
    }
}

/// [`solve_model`] by FISTA with step `1/l` and adaptive restarts
/// (O'Donoghue & Candès).
fn solve_model_fista<A: PenaltyAtoms>(
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
        #[cfg(test)]
        crate::regression::stage::INNER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
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

    /// Block coordinate descent and FISTA reach the same minimizer of the
    /// model (the same model value; the same coordinates where `H` is well
    /// conditioned), for group and sparse-group atoms, with free (weight 0),
    /// pinned (∞) and unpenalized coordinates.
    #[test]
    fn block_descent_matches_fista() {
        let n = 23;
        let a = DMatrix::from_fn(n + 7, n, |i, j| ((i * 7 + j * 13) as f64 * 0.61).sin());
        let theta: Vec<f64> = (0..n).map(|i| ((i * 5) as f64 * 0.9).cos()).collect();
        let g: Vec<f64> = (0..n).map(|i| ((i * 3) as f64 * 1.7).sin() * 4.0).collect();
        let groups: Vec<Vec<usize>> = vec![
            vec![0],
            vec![1, 2],
            vec![3, 4, 5],
            vec![7],
            vec![8, 9],
            vec![10, 11, 12, 13],
            vec![15],
            vec![16, 17],
            vec![19, 20, 21],
        ];
        #[allow(clippy::too_many_arguments)]
        fn check<A: PenaltyAtoms>(
            theta: &[f64],
            g: &[f64],
            pen: &Penalty<A>,
            w: &[f64],
            h: &DMatrix<f64>,
            coords: bool,
            what: &str,
        ) {
            let n = theta.len();
            let l = SymmetricEigen::new(h.clone()).eigenvalues.max() * 1.02;
            let blocks = pen.atoms.blocks().unwrap();
            let zb = solve_model_blocks(theta, g, h, pen, w, &blocks, 100_000);
            let zf = solve_model_fista(theta, g, h, l, pen, w, 200_000);
            let model = |z: &[f64]| {
                let dz = DVector::from_iterator(n, z.iter().zip(theta).map(|(a, b)| a - b));
                DVector::from_column_slice(g).dot(&dz)
                    + 0.5 * dz.dot(&(h * &dz))
                    + pen.lambda * weighted_atoms(&pen.atoms, z, w)
            };
            let (mb, mf) = (model(&zb), model(&zf));
            assert!(
                mb <= mf + 1e-10 * mf.abs(),
                "{what}: model {mb:e} vs FISTA {mf:e}"
            );
            if coords {
                let diff = zb
                    .iter()
                    .zip(&zf)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0, f64::max);
                // (FISTA's own stopping rule leaves ~1e-7)
                assert!(diff < 1e-6, "{what}: {diff:e}");
            }
        }
        for (ridge, coords) in [(0.05, false), (2.0, true)] {
            let h = a.transpose() * &a + DMatrix::identity(n, n) * ridge;
            for lambda in [0.3, 3.0, 30.0] {
                let gn = Penalty {
                    atoms: GroupNorms {
                        groups: groups.clone(),
                    },
                    base: vec![],
                    exponent: Exponent::Fixed(1.0),
                    lambda,
                };
                let mut w = vec![1.0; groups.len()];
                w[1] = 0.0;
                w[4] = f64::INFINITY;
                w[6] = 2.5;
                check(
                    &theta,
                    &g,
                    &gn,
                    &w,
                    &h,
                    coords,
                    &format!("groups, ridge {ridge}, λ = {lambda}"),
                );
                let sg = Penalty {
                    atoms: SparseGroupNorms {
                        groups: groups.clone(),
                    },
                    base: vec![],
                    exponent: Exponent::Fixed(1.0),
                    lambda,
                };
                let mut w = vec![0.7; sg.atoms.len()];
                w[2] = f64::INFINITY;
                w[groups.len() + 4] = 0.0;
                check(
                    &theta,
                    &g,
                    &sg,
                    &w,
                    &h,
                    coords,
                    &format!("sparse groups, ridge {ridge}, λ = {lambda}"),
                );
            }
        }
    }

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
