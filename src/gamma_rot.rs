//! `γ`-rot: Yu & Huang's `γ` with smooth groups `j ≥ 2` (a refinement of
//! the parameterization; an experiment).
//!
//! # What is kept from `γ`
//!
//! * The layout of the coordinates: `γ ∈ ℝ^{(d−2)(d+1)/2}` in groups `G_j`
//!   (`j = 1..d−2`, `j + 1` entries), group `j` placing eigen-axis `j + 1`
//!   of the tangent covariance inside the first `j + 1` axes of the frame.
//! * Group 1 exactly as in `γ`: `K_1 = λ_2/λ_1 = 1 + ‖G_1‖`, the axis of
//!   `λ_1` at angle `atan2(G_{1,2}, G_{1,1})` in the first tangent plane.
//! * The meaning of a group `j ≥ 2`: a ratio of consecutive eigenvalues and
//!   the direction of an eigen-axis. `γ = 0` is the isotropic distribution.
//!
//! # What changes
//!
//! Group `j ≥ 2` is read as `(κ_j, ω_j) ∈ ℝ × ℝ^j` instead of as a norm and a
//! direction:
//!
//! * `K_j = λ_{j+1}/λ_j = exp(κ_j)`: the same ratio, on the log scale and
//!   signed. `κ_j = 0` is still the tie `λ_{j+1} = λ_j`; `κ_j < 0` lets the
//!   eigenvalues cross as `x` varies, which `γ`'s `K_j = 1 + ‖G_j‖ ≥ 1`
//!   cannot.
//! * The axis is `v_j = Q_j e_{j+1}` with `Q_j = I + a(s²) Ω + b(s²) Ω²`,
//!   where `Ω` is the skew matrix of the block with `Ω e_{j+1} = (ω_j, 0)`
//!   and `s = ‖ω_j‖`: the minimal rotation taking the block's pole `e_{j+1}`
//!   to `v_j`. Its other columns, the lower eigen-axes, are the parallel
//!   transport of `e_1..e_j` along that great circle. Two links:
//!   [`RotLink::Stereo`], `a = 4/(4 + s²)`, `b = 2/(4 + s²)` (the Cayley
//!   transform: `v_j` is the inverse stereographic image of `ω_j/2`, the
//!   rotation angle `2 atan(s/2) < π`, and every entry of `V` is rational in
//!   `ω_j`); and [`RotLink::Exp`], `a = sin s/s`, `b = (1 − cos s)/s²` (the
//!   exponential map: rotation angle `s`, periodic in `s`).
//! * The tangent frame is `Ξ(μ̂) = R(m → μ̂) E`, transported from a reference
//!   direction `m` ([`crate::logtangent::LtFrame`]) as in
//!   `Shape::GammaTransported`, instead of `onb(μ)`.
//!
//! `V = Ξ Q diag(λ) Qᵀ Ξᵀ + μ̂μ̂ᵀ` with `Q = Q_{d−2} (Q_{d−3} ⊕ 1) ⋯ (Q_1 ⊕ I)`,
//! `log λ_1 = 0`, `log λ_2 = log K_1`, `log λ_{j+1} = log λ_j + κ_j`, then
//! centred so that `Π λ = 1`.
//!
//! # Why
//!
//! `γ ↦ V` is discontinuous where a group's trailing pair vanishes and
//! where a group `j ≥ 2` vanishes above unequal lower eigenvalues
//! (`docs/MATH.md` §2). Both come from reading the group as a norm and a
//! direction and from completing the direction's frame by plane rotations
//! whose angles divide by that norm. Here nothing divides by a norm for
//! `j ≥ 2`: `(κ, ω) ↦ V` is `C^∞` everywhere (group 1 keeps `γ`'s kink at
//! `G_1 = 0`, where `V` is continuous but not differentiable), so a
//! regression `γ(x) = Γ x̃` has a likelihood without cliffs and without the
//! non-regular suprema of `γ`. What remains is non-identifiability, not
//! discontinuity: at a tie `λ_{j+1} = λ_j` the rotation within the tied
//! plane is a flat direction, and at isotropy every `ω_j` is flat (the
//! information is singular there, as in every chart built from eigenvalues
//! and eigenvectors). The map is not injective either: the sign of an axis
//! (`v_j ↦ −v_j`), the labelling of the eigenvalues and, for `Exp`,
//! `s ↦ s + 2π` give the same `V` from other coordinates. That is a discrete
//! gauge, fixed by [`GrFrame::coords_from_covariance`] (eigenvalues
//! ascending, `v_j · e_{j+1} ≥ 0`).
//!
//! Without covariates the family of distributions is `γ`'s (every `V` has
//! coordinates). At `d = 3` the map is exactly `γ`'s in the transported
//! frame. With covariates it is a different regression model: an axis moves
//! along a great circle at a constant rate in `x` and a log-ratio is affine
//! in `x`, where `γ` takes the polar coordinates of an affine function.
//!
//! Status: an experiment with forward-mode gradients, driven by
//! `gamma_rot_fits` in `src/regression/fit_profile.rs`.

use crate::dims;
use crate::dual::{Dual, seed_array};
use crate::gamma_smooth::{project, sorted_tangent_eigen};
use crate::loglik::log_m_p;
use crate::logtangent::LtFrame;
use crate::scalar::Scalar;
use crate::stats::normal::LN_2PI;
use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};

/// How a group's rotation vector `ω_j` turns the block's pole into its axis.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RotLink {
    /// Cayley transform: rotation angle `2 atan(‖ω‖/2) < π`, rational, no wrap.
    Stereo,
    /// Exponential map: rotation angle `‖ω‖`, periodic.
    Exp,
}

/// The `γ`-rot parameterization: a reference mean direction and a link.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GrFrame {
    lt: LtFrame,
    link: RotLink,
}

impl GrFrame {
    /// Frame with reference mean direction `m` (normalised) and link `link`.
    ///
    /// # Panics
    /// If `m.len() < 3` or `m` is zero.
    pub fn new(m: &[f64], link: RotLink) -> Self {
        assert!(m.len() >= 3, "gamma-rot needs d >= 3, got d = {}", m.len());
        Self {
            lt: LtFrame::new(m),
            link,
        }
    }

    /// The dimension `d`.
    pub fn dim(&self) -> usize {
        self.lt.dim()
    }

    /// The link.
    pub fn link(&self) -> RotLink {
        self.link
    }

    /// The tangent frame (reference mean direction `m` and basis `E`).
    pub fn lt_frame(&self) -> &LtFrame {
        &self.lt
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

    /// Eigenvalues `λ_1..λ_{d−1}` of the tangent covariance (product 1), in
    /// the order of the axes: `λ_1, λ_2` from group 1, `λ_{j+1}` from group `j`.
    pub fn eigenvalues(&self, gamma: &[f64]) -> Vec<f64> {
        let d = self.dim();
        assert_eq!(
            gamma.len(),
            dims::gamma_len(d),
            "gamma has length {}",
            gamma.len()
        );
        let mut ell = vec![0.0; d];
        log_eigen(gamma, d, &mut ell);
        let p = d - 1;
        let mean = ell[..p].iter().sum::<f64>() / p as f64;
        (0..p).map(|k| (ell[k] - mean).exp()).collect()
    }

    /// The axis `v_j = Q_j e_{j+1}` of group `gr ≥ 2`, in the coordinates of
    /// its block (length `gr + 1`).
    pub fn axis(&self, gamma: &[f64], gr: usize) -> Vec<f64> {
        assert!(gr >= 2, "group {gr} has no rotation vector");
        let off = gr * (gr + 1) / 2 - 1;
        let w = &gamma[off + 1..off + gr + 1];
        let s2: f64 = w.iter().map(|x| x * x).sum();
        let (a, b) = link_coeffs(self.link, s2);
        let mut v: Vec<f64> = w.iter().map(|x| a * x).collect();
        v.push(1.0 - b * s2);
        v
    }

    /// The covariance `V` of `ESAG(mu, γ)` in this parameterization.
    pub fn covariance(&self, mu: &[f64], gamma: &[f64]) -> DMatrix<f64> {
        let d = self.dim();
        assert_eq!(mu.len(), d, "mu has length {}", mu.len());
        assert_eq!(
            gamma.len(),
            dims::gamma_len(d),
            "gamma has length {}",
            gamma.len()
        );
        let p = d - 1;
        let mut basis = DMatrix::zeros(d, p);
        let (mut z, mut ell) = (vec![0.0; d], vec![0.0; d]);
        let mut y = vec![0.0; d];
        for i in 0..d {
            y.iter_mut().for_each(|v| *v = 0.0);
            y[i] = 1.0;
            transform(self, mu, gamma, &y, &mut z, &mut ell);
            for j in 0..p {
                basis[(i, j)] = z[j];
            }
        }
        let mean = ell[..p].iter().sum::<f64>() / p as f64;
        let lam: Vec<f64> = (0..p).map(|k| (ell[k] - mean).exp()).collect();
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
    /// determinant 1 on the tangent space): the canonical member of the
    /// equivalence class, with eigenvalues ascending along the axes (so
    /// every `κ_j ≥ 0`) and every axis in the hemisphere of its pole
    /// (`v_j · e_{j+1} ≥ 0`; for `Stereo` then `‖ω_j‖ ≤ 2`, for `Exp`
    /// `‖ω_j‖ ≤ π/2`).
    pub fn coords_from_covariance(&self, mu: &[f64], v: &DMatrix<f64>) -> Vec<f64> {
        let d = self.dim();
        let xi = self.tangent_frame(mu);
        let (lam, mut cur) = sorted_tangent_eigen(&xi, v);
        let mut gamma = vec![0.0; dims::gamma_len(d)];
        for gr in (2..=d - 2).rev() {
            let off = gr * (gr + 1) / 2 - 1;
            let mut a: Vec<f64> = (0..=gr).map(|i| cur[(i, gr)]).collect();
            let na = a.iter().map(|x| x * x).sum::<f64>().sqrt();
            a.iter_mut().for_each(|x| *x /= na);
            if a[gr] < 0.0 {
                a.iter_mut().for_each(|x| *x = -*x);
            }
            let omega: Vec<f64> = match self.link {
                RotLink::Stereo => (0..gr).map(|i| 2.0 * a[i] / (1.0 + a[gr])).collect(),
                RotLink::Exp => {
                    let sn = (0..gr).map(|i| a[i] * a[i]).sum::<f64>().sqrt();
                    let th = a[gr].clamp(-1.0, 1.0).acos();
                    if sn < 1e-300 {
                        vec![0.0; gr]
                    } else {
                        (0..gr).map(|i| th * a[i] / sn).collect()
                    }
                }
            };
            gamma[off] = (lam[gr] / lam[gr - 1]).ln();
            gamma[off + 1..off + gr + 1].copy_from_slice(&omega);
            for l in 0..gr {
                let mut col: Vec<f64> = (0..=gr).map(|i| cur[(i, l)]).collect();
                rotate_t(self.link, gr, &omega, &mut col);
                for i in 0..=gr {
                    cur[(i, l)] = col[i];
                }
            }
        }
        let r = lam[1] / lam[0] - 1.0;
        let a = [cur[(0, 0)], cur[(1, 0)]];
        let na = (a[0] * a[0] + a[1] * a[1]).sqrt();
        let s = if a[0].abs() > 1e-12 {
            a[0].signum()
        } else {
            a[1].signum()
        };
        gamma[0] = r * s * a[0] / na;
        gamma[1] = r * s * a[1] / na;
        gamma
    }
}

/// `(a, b)` of `Q = I + a Ω + b Ω²` for `s² = ‖ω‖²`.
#[inline(always)]
fn link_coeffs<S: Scalar>(link: RotLink, s2: S) -> (S, S) {
    match link {
        RotLink::Stereo => {
            let den = s2 + 4.0;
            (S::from_f64(4.0) / den, S::from_f64(2.0) / den)
        }
        RotLink::Exp => {
            if s2.re() < 1e-10 {
                (
                    S::one() - s2 * (1.0 / 6.0),
                    S::from_f64(0.5) - s2 * (1.0 / 24.0),
                )
            } else {
                let s = s2.sqrt();
                (s.sin() / s, (S::one() - s.cos()) / s2)
            }
        }
    }
}

/// `x ← Q_jᵀ x` on the first `gr + 1` entries, for the rotation vector `w`
/// (length `gr`) of group `gr ≥ 2`: `Qᵀ = I − a Ω + b Ω²`.
#[inline(always)]
fn rotate_t<S: Scalar>(link: RotLink, gr: usize, w: &[S], x: &mut [S]) {
    let mut s2 = S::zero();
    for wi in w.iter().take(gr) {
        s2 += *wi * *wi;
    }
    let (a, b) = link_coeffs(link, s2);
    let xb = x[gr];
    let mut dot = S::zero();
    for i in 0..gr {
        dot += w[i] * x[i];
    }
    let f = a * xb + b * dot;
    for i in 0..gr {
        x[i] -= f * w[i];
    }
    x[gr] = xb + a * dot - b * s2 * xb;
}

/// Unnormalised log-eigenvalues `ℓ_1..ℓ_{d−1}` into `ell`.
#[inline(always)]
fn log_eigen<S: Scalar>(gamma: &[S], d: usize, ell: &mut [S]) {
    ell[0] = S::zero();
    let r1 = (gamma[0] * gamma[0] + gamma[1] * gamma[1]).sqrt();
    ell[1] = (r1 + 1.0).ln();
    for gr in 2..=d - 2 {
        let off = gr * (gr + 1) / 2 - 1;
        ell[gr] = ell[gr - 1] + gamma[off];
    }
}

/// `y ↦` eigen-coordinates `z` (length `≥ d`) and unnormalised
/// log-eigenvalues `ell` (length `≥ d`). Returns `(t, y·μ, ‖μ‖²)`.
#[inline(always)]
fn transform<S: Scalar>(
    fr: &GrFrame,
    mu: &[S],
    gamma: &[S],
    y: &[f64],
    z: &mut [S],
    ell: &mut [S],
) -> (S, S, S) {
    let d = mu.len();
    let out = project(&fr.lt, mu, y, z, ell);
    for gr in (1..=d - 2).rev() {
        let off = gr * (gr + 1) / 2 - 1;
        let g = &gamma[off..off + gr + 1];
        if gr == 1 {
            let r = (g[0] * g[0] + g[1] * g[1]).sqrt();
            if r.re() > 0.0 {
                let (c, s) = (g[0] / r, g[1] / r);
                let (x0, x1) = (z[0], z[1]);
                z[0] = c * x0 + s * x1;
                z[1] = c * x1 - s * x0;
            }
        } else {
            rotate_t(fr.link, gr, &g[1..], z);
        }
    }
    log_eigen(gamma, d, ell);
    out
}

#[inline(always)]
fn loglik_core<S: Scalar>(
    fr: &GrFrame,
    mu: &[S],
    gamma: &[S],
    y: &[f64],
    z: &mut [S],
    ell: &mut [S],
) -> S {
    let d = mu.len();
    let p = d - 1;
    let (t, ymu, mm) = transform(fr, mu, gamma, y, z, ell);
    let mut mean = S::zero();
    for e in ell.iter().take(p) {
        mean += *e;
    }
    mean = mean * (1.0 / p as f64);
    let mut q = t * t;
    for k in 0..p {
        q += z[k] * z[k] * (mean - ell[k]).exp();
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
pub fn loglik_gr<S: Scalar>(fr: &GrFrame, eta: &[S], y: &[f64]) -> S {
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
        9 => fixed::<S, 9>(fr, mu, gamma, y),
        10 => fixed::<S, 10>(fr, mu, gamma, y),
        _ => {
            let (mut z, mut ell) = (vec![S::zero(); d], vec![S::zero(); d]);
            loglik_core(fr, mu, gamma, y, &mut z, &mut ell)
        }
    }
}

#[inline(always)]
fn fixed<S: Scalar, const D: usize>(fr: &GrFrame, mu: &[S], gamma: &[S], y: &[f64]) -> S {
    let (mut z, mut ell) = ([S::zero(); D], [S::zero(); D]);
    loglik_core(fr, mu, gamma, y, &mut z, &mut ell)
}

fn grad_dual<const N: usize>(fr: &GrFrame, eta: &[f64], y: &[f64], grad: &mut [f64]) -> f64 {
    let x: [Dual<N>; N] = seed_array(eta);
    let v = loglik_gr(fr, &x, y);
    grad.copy_from_slice(&v.eps);
    v.re
}

/// Log-density and its exact gradient in `eta` (forward mode).
///
/// # Panics
/// If `d` is outside `3..=10`, or as [`loglik_gr`].
pub fn loglik_gr_grad(fr: &GrFrame, eta: &[f64], y: &[f64], grad: &mut [f64]) -> f64 {
    match eta.len() {
        5 => grad_dual::<5>(fr, eta, y, grad),
        9 => grad_dual::<9>(fr, eta, y, grad),
        14 => grad_dual::<14>(fr, eta, y, grad),
        20 => grad_dual::<20>(fr, eta, y, grad),
        27 => grad_dual::<27>(fr, eta, y, grad),
        35 => grad_dual::<35>(fr, eta, y, grad),
        44 => grad_dual::<44>(fr, eta, y, grad),
        54 => grad_dual::<54>(fr, eta, y, grad),
        n => panic!("gamma-rot gradients are implemented for d = 3..=10 (eta length {n})"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{RngExt, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    fn unit(v: &mut [f64]) {
        let n = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        v.iter_mut().for_each(|x| *x /= n);
    }

    fn reference(d: usize, rng: &mut ChaCha8Rng) -> Vec<f64> {
        let mut m: Vec<f64> = (0..d).map(|_| rng.random_range(-1.0..1.0)).collect();
        unit(&mut m);
        m
    }

    /// Random coordinates: `‖G_1‖ ≤ 1.5`, `|κ_j| ≤ 1`, `‖ω_j‖ ≤ wmax`.
    fn coords(d: usize, wmax: f64, rng: &mut ChaCha8Rng) -> Vec<f64> {
        let mut g = vec![0.0; dims::gamma_len(d)];
        g[0] = rng.random_range(-1.0..1.0);
        g[1] = rng.random_range(-1.0..1.0);
        for gr in 2..=d - 2 {
            let off = gr * (gr + 1) / 2 - 1;
            g[off] = rng.random_range(-1.0..1.0);
            let mut w: Vec<f64> = (0..gr).map(|_| rng.random_range(-1.0..1.0)).collect();
            unit(&mut w);
            let s = rng.random_range(0.0..wmax);
            for i in 0..gr {
                g[off + 1 + i] = s * w[i];
            }
        }
        g
    }

    fn mu(d: usize, rng: &mut ChaCha8Rng) -> Vec<f64> {
        let mut m = reference(d, rng);
        let k = rng.random_range(0.5..4.0);
        m.iter_mut().for_each(|x| *x *= k);
        m
    }

    fn max_abs_diff(a: &DMatrix<f64>, b: &DMatrix<f64>) -> f64 {
        (a - b).iter().map(|x| x.abs()).fold(0.0, f64::max)
    }

    /// `coords → V → coords → V` returns the same `V`, and `V` is a valid
    /// ESAG covariance (`V μ̂ = μ̂`, `det V = 1`), for `d = 3..=10` and both links.
    #[test]
    fn covariance_roundtrip() {
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        for link in [RotLink::Stereo, RotLink::Exp] {
            for d in 3..=10 {
                for _ in 0..5 {
                    let fr = GrFrame::new(&reference(d, &mut rng), link);
                    let m = mu(d, &mut rng);
                    let g = coords(d, 3.0, &mut rng);
                    let v = fr.covariance(&m, &g);
                    let mh: Vec<f64> = {
                        let n = m.iter().map(|x| x * x).sum::<f64>().sqrt();
                        m.iter().map(|x| x / n).collect()
                    };
                    for r in 0..d {
                        let vm: f64 = (0..d).map(|c| v[(r, c)] * mh[c]).sum();
                        assert!((vm - mh[r]).abs() < 1e-10, "V μ̂ ≠ μ̂ at d={d}");
                    }
                    assert!(
                        (v.determinant() - 1.0).abs() < 1e-8,
                        "det V = {}",
                        v.determinant()
                    );
                    let g2 = fr.coords_from_covariance(&m, &v);
                    let v2 = fr.covariance(&m, &g2);
                    assert!(
                        max_abs_diff(&v, &v2) < 1e-8,
                        "d={d} {link:?}: V differs by {}",
                        max_abs_diff(&v, &v2)
                    );
                    // the canonical member: κ_j ≥ 0 and the axes in the pole's hemisphere
                    for gr in 2..=d - 2 {
                        let off = gr * (gr + 1) / 2 - 1;
                        assert!(g2[off] >= -1e-12);
                        let a = fr.axis(&g2, gr);
                        assert!(a[gr] >= -1e-12);
                    }
                }
            }
        }
    }

    /// Coordinates in the canonical gauge roundtrip exactly.
    #[test]
    fn coordinates_roundtrip_in_the_canonical_gauge() {
        let mut rng = ChaCha8Rng::seed_from_u64(11);
        for link in [RotLink::Stereo, RotLink::Exp] {
            for d in 4..=8 {
                let fr = GrFrame::new(&reference(d, &mut rng), link);
                let m = mu(d, &mut rng);
                let mut g = coords(d, 1.2, &mut rng);
                g[0] = g[0].abs() + 0.1; // first entry of group 1 positive
                for gr in 2..=d - 2 {
                    let off = gr * (gr + 1) / 2 - 1;
                    g[off] = g[off].abs() + 0.05; // ascending eigenvalues
                }
                let v = fr.covariance(&m, &g);
                let g2 = fr.coords_from_covariance(&m, &v);
                for (a, b) in g.iter().zip(&g2) {
                    assert!((a - b).abs() < 1e-8, "{link:?} d={d}: {g:?} vs {g2:?}");
                }
            }
        }
    }

    /// At `d = 3` (group 1 only) the map is `γ` in the transported frame.
    #[test]
    fn d3_is_gamma_in_the_transported_frame() {
        let mut rng = ChaCha8Rng::seed_from_u64(3);
        let m = reference(3, &mut rng);
        let fr = GrFrame::new(&m, RotLink::Stereo);
        let gs = crate::gamma_smooth::GsFrame::gamma_transported(&m);
        for _ in 0..20 {
            let mut eta = mu(3, &mut rng);
            eta.extend(coords(3, 1.0, &mut rng));
            let mut y = reference(3, &mut rng);
            unit(&mut y);
            let a = loglik_gr::<f64>(&fr, &eta, &y);
            let b = crate::gamma_smooth::loglik_gs::<f64>(&gs, &eta, &y);
            assert!((a - b).abs() < 1e-12, "{a} vs {b}");
        }
    }

    fn fd_grad(fr: &GrFrame, eta: &[f64], y: &[f64]) -> Vec<f64> {
        let h = 1e-6;
        (0..eta.len())
            .map(|k| {
                let mut ep = eta.to_vec();
                let mut em = eta.to_vec();
                ep[k] += h;
                em[k] -= h;
                (loglik_gr::<f64>(fr, &ep, y) - loglik_gr::<f64>(fr, &em, y)) / (2.0 * h)
            })
            .collect()
    }

    /// The forward-mode gradient matches central differences, including at
    /// the points where `γ` is singular: a group's rotation vector zero (the
    /// pole), a zero log-ratio (a tie), and for `Exp` a rotation by exactly
    /// `π` (the antipode).
    #[test]
    fn gradient_matches_differences_at_the_special_points() {
        let mut rng = ChaCha8Rng::seed_from_u64(5);
        for link in [RotLink::Stereo, RotLink::Exp] {
            for d in [4usize, 5, 6, 8] {
                let fr = GrFrame::new(&reference(d, &mut rng), link);
                let base: Vec<f64> = {
                    let mut e = mu(d, &mut rng);
                    e.extend(coords(d, 1.5, &mut rng));
                    e
                };
                let off2 = d + 2; // group 2 starts at γ index 2
                let mut points = vec![base.clone()];
                let mut pole = base.clone();
                for i in 0..2 {
                    pole[off2 + 1 + i] = 0.0;
                }
                points.push(pole);
                let mut tie = base.clone();
                tie[off2] = 0.0;
                points.push(tie);
                if link == RotLink::Exp {
                    let mut anti = base.clone();
                    let w = &mut anti[off2 + 1..off2 + 3];
                    let s = (w[0] * w[0] + w[1] * w[1]).sqrt();
                    w[0] *= std::f64::consts::PI / s;
                    w[1] *= std::f64::consts::PI / s;
                    points.push(anti);
                }
                for eta in &points {
                    let mut y = reference(d, &mut rng);
                    unit(&mut y);
                    let mut g = vec![0.0; eta.len()];
                    let f = loglik_gr_grad(&fr, eta, &y, &mut g);
                    assert!(f.is_finite());
                    let fd = fd_grad(&fr, eta, &y);
                    for k in 0..eta.len() {
                        let tol = 1e-5 * (1.0 + g[k].abs());
                        assert!(
                            (g[k] - fd[k]).abs() < tol,
                            "{link:?} d={d} k={k}: dual {} vs fd {}",
                            g[k],
                            fd[k]
                        );
                    }
                }
            }
        }
    }

    /// The gradient is continuous through the antipode (`Exp`, `‖ω_2‖ = π`)
    /// and through the pole (`ω_2 = 0`): a kink would change it by O(1).
    #[test]
    fn gradient_is_continuous_through_pole_and_antipode() {
        let mut rng = ChaCha8Rng::seed_from_u64(9);
        let d = 5;
        for link in [RotLink::Stereo, RotLink::Exp] {
            let fr = GrFrame::new(&reference(d, &mut rng), link);
            let mut eta = mu(d, &mut rng);
            eta.extend(coords(d, 1.0, &mut rng));
            let mut y = reference(d, &mut rng);
            unit(&mut y);
            let off2 = d + 2;
            let mut radii = vec![0.0];
            if link == RotLink::Exp {
                radii.push(std::f64::consts::PI);
            }
            for r0 in radii {
                let h = 1e-4;
                let mut grads = vec![];
                for t in [-h, h] {
                    let mut e = eta.clone();
                    let r = r0 + t;
                    e[off2 + 1] = r * 0.6;
                    e[off2 + 2] = r * 0.8;
                    let mut g = vec![0.0; e.len()];
                    loglik_gr_grad(&fr, &e, &y, &mut g);
                    grads.push(g);
                }
                let jump = grads[0]
                    .iter()
                    .zip(&grads[1])
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0, f64::max);
                assert!(
                    jump < 1e-2,
                    "{link:?} radius {r0}: gradient jumps by {jump}"
                );
            }
        }
    }

    /// The Jacobian of `coords ↦ W` (the tangent covariance) has full rank at
    /// a generic point and drops rank by one at a tie `κ_2 = 0`.
    #[test]
    fn jacobian_rank() {
        let mut rng = ChaCha8Rng::seed_from_u64(13);
        let d = 5;
        let p = d - 1;
        let fr = GrFrame::new(&reference(d, &mut rng), RotLink::Stereo);
        let m = mu(d, &mut rng);
        let jac = |g: &[f64]| -> DMatrix<f64> {
            let h = 1e-6;
            let ng = g.len();
            let mut j = DMatrix::zeros(p * (p + 1) / 2, ng);
            for k in 0..ng {
                let mut gp = g.to_vec();
                let mut gm = g.to_vec();
                gp[k] += h;
                gm[k] -= h;
                let vp = fr.covariance(&m, &gp);
                let vm = fr.covariance(&m, &gm);
                let xi = fr.tangent_frame(&m);
                let wp = xi.transpose() * vp * &xi;
                let wm = xi.transpose() * vm * &xi;
                let mut r = 0;
                for a in 0..p {
                    for b in a..p {
                        j[(r, k)] = (wp[(a, b)] - wm[(a, b)]) / (2.0 * h);
                        r += 1;
                    }
                }
            }
            j
        };
        let g = coords(d, 1.0, &mut rng);
        let sv = jac(&g).singular_values();
        let (smax, smin) = (sv.max(), sv.min());
        assert!(smin > 1e-4 * smax, "generic point: singular values {sv}");
        let mut tie = g.clone();
        tie[2] = 0.0; // κ_2 = 0
        let sv = jac(&tie).singular_values();
        let mut s: Vec<f64> = sv.iter().copied().collect();
        s.sort_by(f64::total_cmp);
        assert!(s[0] < 1e-6 * s[s.len() - 1], "tie: singular values {s:?}");
        assert!(s[1] > 1e-4 * s[s.len() - 1], "tie: singular values {s:?}");
    }
}
