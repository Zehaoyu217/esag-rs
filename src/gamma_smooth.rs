//! `γ`-smooth: Yu & Huang's `γ` with smooth completions.
//!
//! # What is kept from `γ`
//!
//! The coordinates and their meaning: `γ` ∈ ℝ^{(d−2)(d+1)/2} in groups
//! `G_j` (`j = 1..d−2`, `j + 1` entries); the eigenvalue ratios
//! `K_j = 1 + ‖G_j‖`, so that `λ_1 ≤ … ≤ λ_{d−1}` with `Π λ = 1`
//! ([`crate::param`]); and the eigenvector each group places. In the kernel
//! of [`crate::loglik::quad_form_parts`], group `j` acts on the first `j + 1`
//! coordinates of the tangent frame, and its plane rotations send `e_j` to
//! the *signed reversal* of `G_j`:
//!
//! ```text
//! v_j = (1/‖G_j‖) Σ_k (−1)^k G_{j,k} e_{j−k},
//! ```
//!
//! the eigenvector of `λ_{j+1}` within the block. That part is continuous.
//!
//! # What changes
//!
//! * **The completion.** The other `j` columns of the block, which carry the
//!   lower eigenvectors, come in `γ` from plane rotations whose first angle
//!   is `atan2` of the group's trailing pair. Where that pair vanishes and
//!   `‖G_j‖ > 0`, the angle is undefined but still rotates the lower
//!   eigenvectors, so `V` jumps (`docs/MATH.md` §2). Here the block is
//!   `Q_j = R(c_j → v_j) B_j` instead, with `c_j` a fixed unit *reference*,
//!   `R(c → v) = H_{v+c} H_c` the minimal rotation taking `c` to `v`
//!   (`H_w = I − 2wwᵀ/(wᵀw)`), and `B_j = H_{e_j − c_j}` the reflection
//!   taking `e_j` to `c_j`. So `Q_j e_j = v_j` as in `γ`, and `Q_j` is smooth
//!   in `v_j` except at `v_j = −c_j`: one ray per group instead of a
//!   codimension-2 subspace. With `c_j` the data's own direction (see
//!   [`GsFrame::self_referenced`]) that ray points away from the data.
//!   Group 1 (a 2-dimensional block, whose completion is the 90° rotation of
//!   `v_1`, smooth everywhere) is exactly as in `γ`.
//! * **The tangent frame.** `onb(μ)` is singular on the great sphere
//!   `μ_0 = μ_1 = 0`; here the frame is `Ξ(μ̂) = R(m → μ̂) E` from a
//!   reference mean direction `m`, as in the log-tangent shape
//!   ([`crate::logtangent::LtFrame`]), smooth except at `μ̂ = −m`.
//!
//! What cannot change: where a whole group `j ≥ 2` vanishes above unequal
//! lower eigenvalues, `V` still depends on the direction of approach. Any
//! parameterization that writes `V` as eigenvalue gaps plus eigenvector
//! directions has such points (`docs/MATH.md` §2); they have codimension
//! `j + 1 ≥ 3`.
//!
//! Without covariates the family of distributions is the same as `γ`'s
//! (every `V` with distinct eigenvalues has a `γ`-smooth representation, by
//! [`GsFrame::gamma_from_covariance`]). At `d = 3` (group 1 only), and with
//! `μ̂ = m`, the covariance is exactly `γ`'s.
//!
//! # Status: an experiment, not used by the regression
//!
//! Prototype fits (`gamma_smooth_fits` in `src/regression/fit_profile.rs`,
//! n = 2000, 5 covariates, 8 datasets per cell) found:
//!
//! * **Convergence.** γ-smooth fits converge where `γ`'s are trapped: 8 of 8
//!   against 0 of 8 on the hard scenario at `d = 5`, and 8 of 8 against
//!   0 of 8 on γ-smooth data from the standard scenario at `d = 8`.
//! * **But they are slow.** They take 60–70 iterations on the standard
//!   scenario at `d = 5` and 150–1600 at `d = 8` and on the hard scenario
//!   (log-tangent: ~20). Fits on the hard scenario also gain more than a
//!   regular maximum should (52–73 log-likelihood units over the truth,
//!   against `K/2 = 42`). There the eigenvector directions sweep nearly
//!   the whole sphere, so observations reach the antipode `−c_j` and the
//!   zero groups (largest gradients at `1 + v·c ≈ 0.00–0.05`), the
//!   singular points no such parameterization can avoid.
//! * **A different regression model.** On data from `γ` it recovers `V(x)`
//!   no better than `γ` itself: `V̂` error 0.35 against 0.25 (standard
//!   scenario, `d = 5`), 0.90 against 0.46 (standard, `d = 8`), 0.25
//!   against 0.24 (hard, `d = 4`) and 0.32 against 0.32 (hard, `d = 5`, where
//!   `γ` converged in 2 of 8). The log-tangent shape did better than both
//!   on such data (0.285, hard scenario, `d = 5`).
//!
//! So the trailing-pair traps can be removed by changing the completion,
//! but the result is neither `γ`'s regression model nor as well behaved
//! as the log-tangent one. Gradients here are forward-mode (slow); there is
//! no reverse-mode kernel or Hessian.

use crate::dims;
use crate::dual::{Dual, seed_array};
use crate::loglik::{log_m_p, root_d_minus_1};
use crate::logtangent::LtFrame;
use crate::scalar::Scalar;
use crate::stats::normal::LN_2PI;
use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};

/// The reference directions of the `γ`-smooth parameterization: `m` for the
/// tangent frame, and one unit `c_j ∈ ℝ^{j+1}` per group `j = 2..d−2`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GsFrame {
    lt: LtFrame,
    /// `refs[j − 2] = c_j`.
    refs: Vec<Vec<f64>>,
    /// `b_j = e_j − c_j` (the reflection `B_j`), and `2/(b_j·b_j)` (0 when
    /// `c_j = e_j`, where `B_j = I`).
    bvec: Vec<Vec<f64>>,
    bscale: Vec<f64>,
}

impl GsFrame {
    /// Frame with reference mean direction `m` and group references `refs`
    /// (`refs[j − 2]` of length `j + 1`, for `j = 2..d−2`; normalised).
    ///
    /// # Panics
    /// If `m.len() < 3`, `m` is zero, or the references have the wrong
    /// number or lengths, or are zero.
    pub fn new(m: &[f64], refs: Vec<Vec<f64>>) -> Self {
        let d = m.len();
        assert!(d >= 3, "gamma-smooth needs d >= 3, got d = {d}");
        assert_eq!(refs.len(), d - 3, "d = {d} has {} groups j >= 2", d - 3);
        let mut out_refs = Vec::with_capacity(d - 3);
        let (mut bvec, mut bscale) = (Vec::with_capacity(d - 3), Vec::with_capacity(d - 3));
        for (i, c) in refs.into_iter().enumerate() {
            let j = i + 2;
            assert_eq!(
                c.len(),
                j + 1,
                "group {j} reference must have length {}",
                j + 1
            );
            let nc = c.iter().map(|v| v * v).sum::<f64>().sqrt();
            assert!(
                nc > 0.0 && nc.is_finite(),
                "group {j} reference must be non-zero"
            );
            let c: Vec<f64> = c.iter().map(|v| v / nc).collect();
            let mut b: Vec<f64> = c.iter().map(|v| -v).collect();
            b[j] += 1.0;
            let bb: f64 = b.iter().map(|v| v * v).sum();
            bscale.push(if bb > 1e-30 { 2.0 / bb } else { 0.0 });
            bvec.push(b);
            out_refs.push(c);
        }
        Self {
            lt: LtFrame::new(m),
            refs: out_refs,
            bvec,
            bscale,
        }
    }

    /// `m = e_d` and `c_j = e_j` for every group.
    pub fn standard(d: usize) -> Self {
        let mut m = vec![0.0; d];
        m[d - 1] = 1.0;
        let refs = (2..=d.saturating_sub(2))
            .map(|j| {
                let mut c = vec![0.0; j + 1];
                c[j] = 1.0;
                c
            })
            .collect();
        Self::new(&m, refs)
    }

    /// The frame in which `(mu, v)` has every block at its reference
    /// (`v_j = c_j`, so `Q_j = B_j`), and the coordinates of `(mu, v)` in it.
    /// `m = μ̂`. The sign of each `c_j` is fixed by making its largest entry
    /// positive.
    pub fn self_referenced(mu: &[f64], v: &DMatrix<f64>) -> (Self, Vec<f64>) {
        let d = mu.len();
        let nm = mu.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(nm > 0.0, "the reference mean must be non-zero");
        let m: Vec<f64> = mu.iter().map(|x| x / nm).collect();
        let lt = LtFrame::new(&m);
        let e = lt.tangent_basis();
        let p = d - 1;
        let xi = DMatrix::from_fn(d, p, |r, c| e[c * d + r]);
        let (lam, mut cur) = sorted_tangent_eigen(&xi, v);
        let mut refs = vec![vec![]; d - 3];
        let mut gamma = vec![0.0; dims::gamma_len(d)];
        for gr in (1..=d - 2).rev() {
            let n = gr + 1;
            let off = gr * (gr + 1) / 2 - 1;
            let r = (lam[gr] / lam[gr - 1] - 1.0).max(1e-12);
            let a: Vec<f64> = (0..n).map(|i| cur[(i, gr)]).collect();
            if gr == 1 {
                write_group1(&a, r, &mut gamma[off..off + 2]);
                continue;
            }
            let imax = (0..n).fold(0, |b, i| if a[i].abs() > a[b].abs() { i } else { b });
            let s = if a[imax] >= 0.0 { 1.0 } else { -1.0 };
            let c: Vec<f64> = a.iter().map(|x| s * x).collect();
            for k in 0..n {
                gamma[off + k] = sign(k) * c[gr - k] * r;
            }
            // remaining eigenvectors into the block's lower frame: Qᵀ = B_j
            let mut b: Vec<f64> = c.iter().map(|x| -x).collect();
            b[gr] += 1.0;
            let bb: f64 = b.iter().map(|x| x * x).sum();
            if bb > 1e-30 {
                for l in 0..gr {
                    let bz: f64 = (0..n).map(|i| b[i] * cur[(i, l)]).sum();
                    for i in 0..n {
                        cur[(i, l)] -= 2.0 * bz / bb * b[i];
                    }
                }
            }
            refs[gr - 2] = c;
        }
        (Self::new(&m, refs), gamma)
    }

    /// The dimension `d`.
    pub fn dim(&self) -> usize {
        self.lt.dim()
    }

    /// The tangent frame (reference mean direction `m` and basis `E`).
    pub fn lt_frame(&self) -> &LtFrame {
        &self.lt
    }

    /// The group references `c_2, …, c_{d−2}`.
    pub fn refs(&self) -> &[Vec<f64>] {
        &self.refs
    }

    /// `Ξ(μ̂)`, `d × (d−1)`.
    pub fn tangent_frame(&self, mu: &[f64]) -> DMatrix<f64> {
        let d = self.dim();
        let mut xi = DMatrix::zeros(d, d - 1);
        let (mut z, mut u) = (vec![0.0; d], vec![0.0; d]);
        let mut y = vec![0.0; d];
        for i in 0..d {
            y.iter_mut().for_each(|v| *v = 0.0);
            y[i] = 1.0;
            project(&self.lt, mu, &y, &mut z, &mut u);
            for j in 0..d - 1 {
                xi[(i, j)] = z[j];
            }
        }
        xi
    }

    /// The covariance `V` of `ESAG(mu, γ)` in this parameterization.
    pub fn covariance(&self, mu: &[f64], gamma: &[f64]) -> DMatrix<f64> {
        let d = self.dim();
        let p = d - 1;
        let mut basis = DMatrix::zeros(d, p);
        let (mut z, mut k) = (vec![0.0; d], vec![0.0; d]);
        let mut y = vec![0.0; d];
        for i in 0..d {
            y.iter_mut().for_each(|v| *v = 0.0);
            y[i] = 1.0;
            transform(self, mu, gamma, &y, &mut z, &mut k);
            for j in 0..p {
                basis[(i, j)] = z[j];
            }
        }
        let lam = eigenvalues(&k[..d - 2], d);
        let nm = mu.iter().map(|x| x * x).sum::<f64>().sqrt();
        let mh: Vec<f64> = if nm > 0.0 {
            mu.iter().map(|x| x / nm).collect()
        } else {
            self.lt.reference().to_vec()
        };
        DMatrix::from_fn(d, d, |r, c| {
            mh[r] * mh[c]
                + (0..p)
                    .map(|j| lam[j] * basis[(r, j)] * basis[(c, j)])
                    .sum::<f64>()
        })
    }

    /// Coordinates of `(mu, v)` in this parameterization (`v` scaled to
    /// determinant 1 on the tangent space). Exact ties of eigenvalues get a
    /// gap of `1e−12`.
    pub fn gamma_from_covariance(&self, mu: &[f64], v: &DMatrix<f64>) -> Vec<f64> {
        let d = self.dim();
        let xi = self.tangent_frame(mu);
        let (lam, mut cur) = sorted_tangent_eigen(&xi, v);
        let mut gamma = vec![0.0; dims::gamma_len(d)];
        for gr in (1..=d - 2).rev() {
            let n = gr + 1;
            let off = gr * (gr + 1) / 2 - 1;
            let r = (lam[gr] / lam[gr - 1] - 1.0).max(1e-12);
            let a: Vec<f64> = (0..n).map(|i| cur[(i, gr)]).collect();
            if gr == 1 {
                write_group1(&a, r, &mut gamma[off..off + 2]);
                continue;
            }
            let c = &self.refs[gr - 2];
            let s = if a.iter().zip(c).map(|(x, y)| x * y).sum::<f64>() >= 0.0 {
                1.0
            } else {
                -1.0
            };
            let vv: Vec<f64> = a.iter().map(|x| s * x).collect();
            for k in 0..n {
                gamma[off + k] = sign(k) * vv[gr - k] * r;
            }
            let g: Vec<f64> = gamma[off..off + n].to_vec();
            for l in 0..gr {
                let mut col: Vec<f64> = (0..n).map(|i| cur[(i, l)]).collect();
                complete_t(self, gr, &g, &mut col);
                for i in 0..n {
                    cur[(i, l)] = col[i];
                }
            }
        }
        gamma
    }
}

/// `(−1)^k`.
#[inline(always)]
fn sign(k: usize) -> f64 {
    if k.is_multiple_of(2) { 1.0 } else { -1.0 }
}

/// Group 1 from its top eigenvector `a ∈ ℝ²` (either sign gives the same
/// block): `v_1 = (−G_1, G_0)/‖G‖`.
fn write_group1(a: &[f64], r: f64, out: &mut [f64]) {
    let s = if a[1] >= 0.0 { 1.0 } else { -1.0 };
    out[0] = s * a[1] * r;
    out[1] = -s * a[0] * r;
}

/// Tangent eigenvalues of `v` in the frame `xi` (ascending, scaled to
/// product 1) and their eigenvectors (columns, in tangent coordinates).
fn sorted_tangent_eigen(xi: &DMatrix<f64>, v: &DMatrix<f64>) -> (Vec<f64>, DMatrix<f64>) {
    let p = xi.ncols();
    let a = xi.transpose() * v * xi;
    let eig = nalgebra::SymmetricEigen::new((&a + a.transpose()) * 0.5);
    let mut idx: Vec<usize> = (0..p).collect();
    idx.sort_by(|&i, &j| eig.eigenvalues[i].total_cmp(&eig.eigenvalues[j]));
    let log_gm = idx.iter().map(|&i| eig.eigenvalues[i].ln()).sum::<f64>() / p as f64;
    let lam: Vec<f64> = idx
        .iter()
        .map(|&i| (eig.eigenvalues[i].ln() - log_gm).exp())
        .collect();
    let cur = DMatrix::from_fn(p, p, |r, c| eig.eigenvectors[(r, idx[c])]);
    (lam, cur)
}

/// `λ_1..λ_{d−1}` from the ratios `K_1..K_{d−2}`.
fn eigenvalues(k: &[f64], d: usize) -> Vec<f64> {
    let mut log_prod = 0.0;
    for (i, &ki) in k.iter().enumerate() {
        log_prod += ki.ln() * (d - 2 - i) as f64;
    }
    let mut lam = vec![(-log_prod / (d - 1) as f64).exp()];
    for j in 0..d - 2 {
        let next = lam[j] * k[j];
        lam.push(next);
    }
    lam
}

/// `z ← Ξ(μ̂)ᵀ y` (first `d − 1` entries; `u`, length `≥ d`, is scratch);
/// returns `(t, y·μ, ‖μ‖²)` with `t = μ̂·y`. At `μ = 0` it uses `μ̂ = m`,
/// and at `μ̂ = −m` exactly `v = E_0` (the conventions of the log-tangent
/// kernel).
#[inline(always)]
fn project<S: Scalar>(lt: &LtFrame, mu: &[S], y: &[f64], z: &mut [S], u: &mut [S]) -> (S, S, S) {
    let d = mu.len();
    let m = lt.reference();
    let e = lt.tangent_basis();
    let mut mm = S::zero();
    let mut ymu = S::zero();
    for k in 0..d {
        mm += mu[k] * mu[k];
        ymu += mu[k] * y[k];
    }
    let zero_mu = mm.re() == 0.0;
    let inv = if zero_mu {
        S::one()
    } else {
        S::one() / mm.sqrt()
    };
    let mh = |k: usize| -> S {
        if zero_mu {
            S::from_f64(m[k])
        } else {
            mu[k] * inv
        }
    };
    let t = if zero_mu {
        S::from_f64((0..d).map(|k| m[k] * y[k]).sum())
    } else {
        ymu * inv
    };
    let mut vv = S::zero();
    let mut vy = S::zero();
    for k in 0..d {
        let vk = mh(k) + m[k];
        vv += vk * vk;
        vy += vk * y[k];
    }
    let antipode = vv.re() == 0.0;
    let kappa = if antipode {
        S::from_f64(2.0 * (0..d).map(|k| e[k] * y[k]).sum::<f64>())
    } else {
        vy * 2.0 / vv
    };
    for (k, uk) in u.iter_mut().enumerate().take(d) {
        let vk = if antipode {
            S::from_f64(e[k])
        } else {
            mh(k) + m[k]
        };
        *uk = S::from_f64(y[k]) - kappa * vk;
    }
    for j in 0..d - 1 {
        let col = &e[j * d..(j + 1) * d];
        let mut acc = S::zero();
        for k in 0..d {
            acc += u[k] * col[k];
        }
        z[j] = acc;
    }
    (t, ymu, mm)
}

/// `x ← Q_jᵀ x` on the first `j + 1` entries, for group `j ≥ 2` with
/// entries `g`: `Q_jᵀ = B_j H_{c_j} H_{v_j + c_j}`.
#[inline(always)]
fn complete_t<S: Scalar>(fr: &GsFrame, gr: usize, g: &[S], x: &mut [S]) {
    let n = gr + 1;
    let c = &fr.refs[gr - 2];
    let mut r2 = S::zero();
    for gk in g.iter().take(n) {
        r2 += *gk * *gk;
    }
    if r2.re() > 0.0 {
        let inv = S::one() / r2.sqrt();
        // w = v + c, v_{gr−k} = (−1)^k G_k / ‖G‖
        let w = |i: usize| -> S {
            let k = gr - i;
            g[k] * (inv * sign(k)) + c[i]
        };
        let mut ww = S::zero();
        let mut wx = S::zero();
        for (i, xi) in x.iter().enumerate().take(n) {
            let wi = w(i);
            ww += wi * wi;
            wx += wi * *xi;
        }
        if ww.re() > 0.0 {
            let f = wx * 2.0 / ww;
            for (i, xi) in x.iter_mut().enumerate().take(n) {
                *xi -= f * w(i);
            }
        }
        let mut cx = S::zero();
        for (i, xi) in x.iter().enumerate().take(n) {
            cx += *xi * c[i];
        }
        for (i, xi) in x.iter_mut().enumerate().take(n) {
            *xi -= cx * (2.0 * c[i]);
        }
    }
    let bs = fr.bscale[gr - 2];
    if bs > 0.0 {
        let b = &fr.bvec[gr - 2];
        let mut bx = S::zero();
        for (i, xi) in x.iter().enumerate().take(n) {
            bx += *xi * b[i];
        }
        for (i, xi) in x.iter_mut().enumerate().take(n) {
            *xi -= bx * (bs * b[i]);
        }
    }
}

/// The linear map `y ↦` eigen-coordinates: `z` (length `≥ d`) gets the
/// coordinates of `y` along the eigenvectors of `λ_1..λ_{d−1}`, `k` (length
/// `≥ d − 2`) the ratios `K_j`. Returns `(t, y·μ, ‖μ‖²)`.
#[inline(always)]
fn transform<S: Scalar>(
    fr: &GsFrame,
    mu: &[S],
    gamma: &[S],
    y: &[f64],
    z: &mut [S],
    k: &mut [S],
) -> (S, S, S) {
    let d = mu.len();
    // k doubles as the projection's scratch; it is filled with the K_j below
    let out = project(&fr.lt, mu, y, z, k);
    for gr in (1..=d - 2).rev() {
        let off = gr * (gr + 1) / 2 - 1;
        let g = &gamma[off..off + gr + 1];
        let mut r2 = S::zero();
        for gk in g {
            r2 += *gk * *gk;
        }
        let r = r2.sqrt();
        k[gr - 1] = r + 1.0;
        if gr == 1 {
            // as γ: (c, s) = (G_0, G_1)/‖G‖ on plane (0, 1)
            if r.re() > 0.0 {
                let (c, s) = (g[0] / r, g[1] / r);
                let (x0, x1) = (z[0], z[1]);
                z[0] = c * x0 + s * x1;
                z[1] = c * x1 - s * x0;
            }
        } else {
            complete_t(fr, gr, g, z);
        }
    }
    out
}

#[inline(always)]
fn loglik_core<S: Scalar>(
    fr: &GsFrame,
    mu: &[S],
    gamma: &[S],
    y: &[f64],
    z: &mut [S],
    k: &mut [S],
) -> S {
    let d = mu.len();
    let p = d - 1;
    let ng = d - 2;
    let (t, ymu, mm) = transform(fr, mu, gamma, y, z, k);
    let mut prod = S::one();
    for i in 1..=ng {
        let e = (d - 1 - i) as i32;
        prod *= if e == 1 { k[i - 1] } else { k[i - 1].powi(e) };
    }
    let mut inv_l = root_d_minus_1(prod, d);
    let mut q = t * t;
    for j in 0..p {
        q += z[j] * z[j] * inv_l;
        if j < ng {
            inv_l /= k[j];
        }
    }
    let alpha = ymu / q.sqrt();
    S::from_f64(-0.5 * p as f64 * LN_2PI) - q.ln() * (0.5 * d as f64)
        + (alpha * alpha - mm) * 0.5
        + log_m_p(alpha, p)
}

/// Log-density at the unit vector `y` for stacked `eta = (mu, γ)`.
///
/// # Panics
/// If `eta.len()` is not [`dims::eta_len`] of `d = y.len()`, or the frame
/// has another dimension.
pub fn loglik_gs<S: Scalar>(fr: &GsFrame, eta: &[S], y: &[f64]) -> S {
    let d = y.len();
    assert_eq!(fr.dim(), d, "frame dimension {} but d = {d}", fr.dim());
    assert_eq!(eta.len(), dims::eta_len(d), "eta has length {}", eta.len());
    let (mu, gamma) = eta.split_at(d);
    match d {
        3 => fixed::<S, 3>(fr, mu, gamma, y),
        4 => fixed::<S, 4>(fr, mu, gamma, y),
        5 => fixed::<S, 5>(fr, mu, gamma, y),
        6 => fixed::<S, 6>(fr, mu, gamma, y),
        7 => fixed::<S, 7>(fr, mu, gamma, y),
        8 => fixed::<S, 8>(fr, mu, gamma, y),
        _ => {
            let (mut z, mut k) = (vec![S::zero(); d], vec![S::zero(); d]);
            loglik_core(fr, mu, gamma, y, &mut z, &mut k)
        }
    }
}

#[inline(always)]
fn fixed<S: Scalar, const D: usize>(fr: &GsFrame, mu: &[S], gamma: &[S], y: &[f64]) -> S {
    let (mut z, mut k) = ([S::zero(); D], [S::zero(); D]);
    loglik_core(fr, mu, gamma, y, &mut z, &mut k)
}

fn grad_dual<const N: usize>(fr: &GsFrame, eta: &[f64], y: &[f64], grad: &mut [f64]) -> f64 {
    let x: [Dual<N>; N] = seed_array(eta);
    let v = loglik_gs(fr, &x, y);
    grad.copy_from_slice(&v.eps);
    v.re
}

/// Log-density and its exact gradient in `eta` (forward mode).
///
/// # Panics
/// If `d` is outside `3..=8`, or as [`loglik_gs`].
pub fn loglik_gs_grad(fr: &GsFrame, eta: &[f64], y: &[f64], grad: &mut [f64]) -> f64 {
    match eta.len() {
        5 => grad_dual::<5>(fr, eta, y, grad),
        9 => grad_dual::<9>(fr, eta, y, grad),
        14 => grad_dual::<14>(fr, eta, y, grad),
        20 => grad_dual::<20>(fr, eta, y, grad),
        27 => grad_dual::<27>(fr, eta, y, grad),
        35 => grad_dual::<35>(fr, eta, y, grad),
        n => panic!("gamma-smooth gradient implemented for d = 3..=8, eta length {n}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distribution::Esag;

    fn pseudo(n: usize, seed: usize, scale: f64) -> Vec<f64> {
        (0..n)
            .map(|i| scale * ((i * 7919 + seed * 104_729) as f64 * 0.618_033_988_7).sin())
            .collect()
    }

    fn frame(d: usize, seed: usize) -> GsFrame {
        let m = pseudo(d, seed + 1, 1.0);
        let refs = (2..=d - 2)
            .map(|j| pseudo(j + 1, seed + 10 + j, 1.0))
            .collect();
        GsFrame::new(&m, refs)
    }

    fn max_abs(a: &DMatrix<f64>) -> f64 {
        a.iter().fold(0.0f64, |m, v| m.max(v.abs()))
    }

    #[test]
    fn covariance_is_esag_and_matches_the_kernel() {
        for d in 3..=8 {
            let fr = frame(d, d);
            let mu = pseudo(d, 3, 1.4);
            let g = pseudo(dims::gamma_len(d), 5, 0.9);
            let v = fr.covariance(&mu, &g);
            let vm = &v * nalgebra::DVector::from_column_slice(&mu);
            for k in 0..d {
                assert!((vm[k] - mu[k]).abs() < 1e-12, "V mu = mu at d = {d}");
            }
            assert!(
                (v.determinant() - 1.0).abs() < 1e-10,
                "det V = 1 at d = {d}"
            );
            assert!(max_abs(&(&v - v.transpose())) < 1e-13);
            let y = {
                let y = pseudo(d, 7, 1.0);
                let n = y.iter().map(|x| x * x).sum::<f64>().sqrt();
                y.iter().map(|x| x / n).collect::<Vec<_>>()
            };
            let mut eta = mu.clone();
            eta.extend_from_slice(&g);
            let dist = Esag::from_covariance(&mu, &v).unwrap();
            let want = crate::loglik::loglik_mu_gamma(&mu, dist.gamma(), &y);
            let got: f64 = loglik_gs(&fr, &eta, &y);
            assert!(
                (got - want).abs() < 1e-9 * want.abs().max(1.0),
                "d = {d}: {got} vs {want}"
            );
        }
    }

    #[test]
    fn inverse_map_round_trips() {
        for d in 3..=8 {
            let fr = frame(d, 2 * d);
            let mu = pseudo(d, 4, 2.0);
            let g = pseudo(dims::gamma_len(d), 6, 1.1);
            let v = fr.covariance(&mu, &g);
            let g2 = fr.gamma_from_covariance(&mu, &v);
            let v2 = fr.covariance(&mu, &g2);
            assert!(
                max_abs(&(&v - &v2)) < 1e-9,
                "d = {d}: {}",
                max_abs(&(&v - &v2))
            );
            let (sf, gs) = GsFrame::self_referenced(&mu, &v);
            let v3 = sf.covariance(&mu, &gs);
            assert!(max_abs(&(&v - &v3)) < 1e-9, "self-referenced d = {d}");
        }
    }

    #[test]
    fn d3_equals_gamma_at_the_reference() {
        let mu = [0.7, -1.1, 1.9];
        let nm = mu.iter().map(|x| x * x).sum::<f64>().sqrt();
        let m: Vec<f64> = mu.iter().map(|x| x / nm).collect();
        let fr = GsFrame::new(&m, vec![]);
        for g in [[0.3, -0.8], [-1.2, 0.05], [2.0, 2.0]] {
            let a = fr.covariance(&mu, &g);
            let b = Esag::new(&mu, &g).unwrap().covariance();
            assert!(max_abs(&(&a - &b)) < 1e-12);
        }
    }

    /// Where γ's trailing pair vanishes, γ's V depends on the direction of
    /// approach; γ-smooth's does not.
    #[test]
    fn continuous_where_gamma_jumps() {
        let d = 4;
        let mu = [0.4, 0.9, -0.3, 1.5];
        let fr = frame(d, 1);
        let base = [0.6, -0.4, 1.1, 0.0, 0.0];
        let at = |g: &[f64]| fr.covariance(&mu, g);
        let gam = |g: &[f64]| Esag::new(&mu, g).unwrap().covariance();
        for eps in [1e-4, 1e-7] {
            let mut a = base;
            let mut b = base;
            a[3] = eps;
            b[4] = eps; // pair (eps, 0) vs (0, eps): angle 0 vs π/2
            assert!(
                max_abs(&(at(&a) - at(&b))) < 1e-3,
                "gamma-smooth continuous"
            );
            assert!(max_abs(&(gam(&a) - gam(&b))) > 0.05, "gamma jumps");
        }
    }

    #[test]
    fn forward_gradient_matches_differences() {
        for d in [3usize, 4, 5, 8] {
            let fr = frame(d, 3 * d);
            let mut eta = pseudo(d, 8, 1.3);
            eta.extend(pseudo(dims::gamma_len(d), 9, 0.7));
            let y = {
                let y = pseudo(d, 10, 1.0);
                let n = y.iter().map(|x| x * x).sum::<f64>().sqrt();
                y.iter().map(|x| x / n).collect::<Vec<_>>()
            };
            let mut g = vec![0.0; eta.len()];
            let f = loglik_gs_grad(&fr, &eta, &y, &mut g);
            assert!((f - loglik_gs::<f64>(&fr, &eta, &y)).abs() < 1e-14);
            for i in 0..eta.len() {
                let h = 1e-6;
                let (mut a, mut b) = (eta.clone(), eta.clone());
                a[i] += h;
                b[i] -= h;
                let fd =
                    (loglik_gs::<f64>(&fr, &a, &y) - loglik_gs::<f64>(&fr, &b, &y)) / (2.0 * h);
                assert!(
                    (fd - g[i]).abs() < 1e-6 * g[i].abs().max(1.0),
                    "d = {d} i = {i}"
                );
            }
        }
    }
}
