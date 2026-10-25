//! Log-tangent shape parameterization of ESAG: a smooth alternative to `γ`.
//!
//! # Why
//!
//! Yu & Huang's map `γ ↦ V` ([`crate::param`]) is discontinuous on a
//! codimension-2 set. For every `γ` group `j ≥ 2`, the angle
//! `θ_j = atan2(G_n, G_{n−1})` is undefined where the last two coordinates of
//! the group vanish while `‖G_(j)‖ > 0`, yet it still rotates `V`.
//! Approaching such a point from different directions gives covariances that
//! differ by `O(1)` however close the points are. In a regression
//! `(μ_i, γ_i) = B x̃_i`, observations near that set put cliffs into the
//! likelihood over `B`, and quasi-Newton methods stall on them. This module
//! parameterizes the same family of distributions smoothly. It is opt-in:
//! nothing else in the crate uses it.
//!
//! # The parameterization
//!
//! Let `d ≥ 3` and `p = d − 1`. A fixed reference direction `m ∈ S^{d−1}`
//! and a fixed orthonormal basis `E` (`d × p`) of `m⊥` make up an
//! [`LtFrame`]. For `μ̂ = μ/‖μ‖ ≠ −m`, let `R(m→μ̂)` be the minimal rotation
//! taking `m` to `μ̂`: it rotates in `span(m, μ̂)` and is the identity on its
//! orthogonal complement (parallel transport along the great circle). Then
//! `Ξ(μ̂) = R(m→μ̂) E` is an orthonormal basis of `μ̂⊥` that depends smoothly
//! on `μ̂`, and
//!
//! ```text
//! V = μ̂μ̂ᵀ + Ξ exp(S) Ξᵀ,    S = Sᵀ ∈ ℝ^{p×p},  tr S = 0.
//! ```
//!
//! So `Vμ = μ` and `det V = e^{tr S} = 1` hold by construction. The map
//! `(μ, S) ↦ (μ, V)` is a diffeomorphism onto the ESAG parameter space
//! wherever `μ̂ ≠ −m`, with inverse `S = log(Ξᵀ V Ξ)`
//! ([`s_from_covariance`]). Unlike `γ`, there are no equivalent parameters
//! and no canonicalization is needed. `S = 0` is isotropy.
//!
//! The shape coordinates `s ∈ ℝ^{(d−2)(d+1)/2}` (the same length as `γ`, so
//! `η = (μ, s)` has length [`dims::eta_len`]) are orthonormal coordinates of
//! `S` under the Frobenius inner product, so `‖s‖ = ‖S‖_F`. This is the
//! affine-invariant Riemannian distance of the tangent block `Ξᵀ V Ξ = e^S`
//! from the identity. In order ([`s_to_matrix`], [`matrix_to_s`]):
//!
//! * `s_0..s_{p−2}`: the diagonal, as coefficients on the Helmert basis
//!   `h_k = (1, …, 1, −k, 0, …, 0)/√(k(k+1))` (`k` ones), `k = 1..p−1`, an
//!   orthonormal basis of the traceless diagonal matrices;
//! * then one coordinate per off-diagonal pair `a < b`, in row-major
//!   upper-triangle order, with `S_ab = S_ba = s/√2`.
//!
//! The count is `(p−1) + p(p−1)/2 = (p−1)(p+2)/2 = (d−2)(d+1)/2`.
//!
//! # The frame and its single singular point
//!
//! The minimal rotation is a product of two Householder reflections,
//! `R(m→μ̂) = H_v H_m` with `v = μ̂ + m` and `H_x = I − 2xxᵀ/(xᵀx)`: `H_m`
//! sends `m` to `−m`, `H_v` sends `−m` to `μ̂`, and both fix
//! `span(m, μ̂)⊥`. Since `H_m E = E`,
//!
//! ```text
//! Ξ(μ̂) = H_v E,     z = Ξᵀ y = Eᵀ (y − κ v),   κ = 2 (v·y)/(v·v),
//! ```
//!
//! which costs `O(d)` plus the `p × d` product with `Eᵀ`; no `d × d` matrix
//! is formed. (Equivalently `R = I + K + K²/(1 + mᵀμ̂)`,
//! `K = μ̂mᵀ − mμ̂ᵀ`; the unit tests check the two against each other.)
//! `v·v = 2(1 + mᵀμ̂)`, so the frame is smooth everywhere except at the
//! antipode `μ̂ = −m`, where `v = 0`. Near it the frame turns quickly: its
//! derivative grows like `1/‖μ̂ + m‖`. This singular point cannot be
//! avoided in general. A smooth `Ξ` on all of `S^{d−1}` would be a global
//! frame of its tangent bundle, which exists only for `d ∈ {2, 4, 8}`
//! (Bott–Milnor, Kervaire). For `d = 4, 8` the unit quaternions and
//! octonions give one, which this module does not use. Choose `m` near the
//! bulk of the data, for example the mean direction, so that every fitted
//! `μ̂_i` stays far from `−m`.
//!
//! The kernel's guard conventions: at `μ̂ = −m` exactly it uses `v = E_0`,
//! the first column of `E`, which is the rotation by `π` in the plane
//! `(m, E_0)`. At `μ = 0` it uses `μ̂ = m`. Both branches have constant
//! frames, so the frame contributes nothing to the gradient there.
//!
//! # The density and its gradient
//!
//! The ESAG density depends on `V` only through
//!
//! ```text
//! q = yᵀV⁻¹y = (μ̂·y)² + zᵀ e^{−S} z,     alpha = (y·μ)/√q,
//! log f = −(p/2) log 2π − (d/2) log q + ½(alpha² − ‖μ‖²) + log M_p(alpha).
//! ```
//!
//! [`loglik_lt`] diagonalises `S = QΛQᵀ` by cyclic Jacobi in parallel
//! (round-robin) order, so the rotations of each round are independent and
//! overlap in the processor, on the upper triangle only. It stops once the
//! off-diagonal norm is at most `1e-15 ‖S‖_F`, after at most 16 sweeps. It
//! applies each rotation to `z` as it goes, so `w = Qᵀz` comes out without
//! forming `Q`, and `zᵀ e^{−S} z = Σ_a e^{−λ_a} w_a²`. [`loglik_lt_grad`]
//! also accumulates `Q`, then differentiates by hand in reverse mode:
//!
//! * `∂q/∂S = Q (F ∘ wwᵀ) Qᵀ`, with `F` the first divided differences of
//!   `x ↦ e^{−x}` at the eigenvalues. Near-equal eigenvalues use
//!   `F_ab = e^{−λ_a} expm1(−δ)/δ`, `δ = λ_b − λ_a ≥ 0`, which is accurate
//!   for every gap and gives `F_aa = −e^{−λ_a}` at `δ = 0`. Then
//!   `∂q/∂s` is the inner product with each basis matrix.
//! * `∂q/∂z = 2 e^{−S} z = 2Q(e^{−λ} ∘ w)`, then `ū = E z̄` and, from
//!   `u = y − κv`, `v̄ = −κū − (2(ū·v)/(v·v)) u`. With the `(μ̂·y)²` term,
//!   `μ̂̄ = v̄ + 2(μ̂·y) y`, and `μ̂ = μ/‖μ‖` gives
//!   `∂q/∂μ = (μ̂̄ − (μ̂·μ̂̄) μ̂)/‖μ‖`.
//! * As in [`crate::adjoint`], `∇ log f = q̄ ∇q + ȳmu y − μ (μ block)` with
//!   `ᾱ = alpha + (log M_p)'(alpha)`, `q̄ = −(d + ᾱ alpha)/(2q)` and
//!   `ȳmu = ᾱ/√q`.
//!
//! At `S = 0` the eigen-decomposition is skipped: `zᵀ e^{−S} z = ‖z‖²`,
//! `∂q/∂S = −zzᵀ` and `∂q/∂z = 2z`, exactly.
//!
//! # The Hessian
//!
//! [`loglik_lt_hess`] adds the exact Hessian from the same
//! eigen-decomposition. `log f = Φ(q, ymu) − ‖μ‖²/2` with
//! `Φ = −(d/2) log q + ψ(ymu/√q)`, `ψ(a) = a²/2 + log M_p(a)`, so
//!
//! ```text
//! ∇² log f = Φ_q ∇²q + Φ_qq ∇q∇qᵀ + Φ_qy (∇q eᵀ + e∇qᵀ) + Φ_yy eeᵀ − [I 0; 0 0],
//! ```
//!
//! with `e = (y, 0)` and `∇²q` in three blocks:
//!
//! * *Shape.* By the second-order Daleckii–Krein formula,
//!   `zᵀ D²e^{−S}[H, K] z = 2 Σ_abc T_cab w_a w_b H̃_ac K̃_cb` for symmetric
//!   `H, K` (`H̃ = QᵀHQ`), with `T_cab = e^{−x}[λ_a, λ_c, λ_b]` the second
//!   divided differences. For gaps of at least ½ they come from the
//!   recursion on `F`, below that from the series
//!   `e^{−λ_lo} Σ_n (−1)ⁿ h_{n−2}(δ₁, δ₂)/n!`. In the original coordinates
//!   the form is `2 Wt(u, y, x, v)` at `H = e_x e_uᵀ`, `K = e_y e_vᵀ`, where
//!   `Wt = Σ_c Q_uc Q_yc G_c[x][v]` and `G_c = R_w T_c R_wᵀ`,
//!   `R_w = Q diag(w)`. `Wt` is symmetric within its index pairs, so it is a
//!   `p(p+1)/2`-square matrix, about `p⁵/4 + 1.5 p⁴` operations in all
//!   (9k at `p = 7`). Each entry
//!   of the shape block is a sum of four of its entries, with prefix sums
//!   for the Helmert coordinates.
//! * *Cross.* `∂/∂s_i ∂q/∂z = 2Q(F ∘ B̃_i)w`, chained through
//!   `K = Qᵀ ∂z/∂μ̂` and `∂μ̂/∂μ = P/‖μ‖`, where `P = I − μ̂μ̂ᵀ`.
//! * *Mean.* `A = ∂²q/∂μ̂² = 2yyᵀ + 2Kᵀ e^{−Λ} K + Σ_k ū_k ∇²u_k`, using the
//!   Householder second derivatives, then
//!   `∂²q/∂μ² = (PAP − (g⊥μ̂ᵀ + μ̂g⊥ᵀ + (g·μ̂) P))/‖μ‖²` with `g = ∂q/∂μ̂`,
//!   `g⊥ = Pg`.
//!
//! The value returned by all three functions is bit-identical. They perform
//! the same floating-point operations up to the log-density, and the
//! gradients of [`loglik_lt_grad`] and [`loglik_lt_hess`] agree bit for bit.
//! For `d ≤ 8` they work on stack buffers only, and larger `d` fall back to
//! heap buffers. See `docs/MATH.md`, "Log-tangent shape parameterization
//! (smooth option)".

use crate::dims;
use crate::loglik::{log_m_p_derivs, log_m_p_value};
use crate::onb::onb;
use crate::stats::normal::LN_2PI;
use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};
use std::f64::consts::{FRAC_1_SQRT_2, SQRT_2};

/// Entries of the unit reference direction below this are zeroed before it
/// is passed to [`onb`] (see [`LtFrame::new`]).
const FLUSH: f64 = 1e-70;

/// Jacobi stops once the off-diagonal Frobenius norm of the rotated matrix
/// is at most this times the Frobenius norm of the input.
const JACOBI_TOL: f64 = 1e-15;

/// An off-diagonal entry with `|a_ij|` at most this times the Frobenius norm
/// of the input is set to zero instead of being rotated away. With the input
/// scaled to unit norm, this also keeps `h² + g²` in the rotation formula
/// far from underflow.
const JACOBI_SKIP: f64 = 1e-17;

/// Below this ratio `|a_xy|/|a_yy − a_xx|`, a Jacobi rotation takes
/// `t = a_xy/(a_yy − a_xx)`, `c = 1`, which is exact to rounding (the
/// neglected terms are `O(t²) < 3e-17`) and costs one division instead of
/// two square roots and a division. This covers most rotations of the last
/// sweeps.
const SMALL_ANGLE: f64 = 5e-9;

/// Sweep cap in the likelihood kernel. On random inputs with `p ≤ 7`,
/// convergence takes at most 6 sweeps (5.0 on average for `p = 7`).
const KERNEL_SWEEPS: usize = 16;

/// Sweep cap for the non-kernel functions ([`covariance`],
/// [`s_from_covariance`]), which may see any `p`.
const EXACT_SWEEPS: usize = 64;

/// From this eigenvalue gap on, the divided differences of `e^{−x}` are
/// formed directly as `(e_a − e_b)/(λ_a − λ_b)`, which then loses at most a
/// factor `1/(1 − e^{−0.5}) ≈ 2.5` to cancellation. Below it they use a
/// series.
const DD_DIRECT: f64 = 0.5;

// =========================================================================
// The frame
// =========================================================================

/// The fixed reference of the log-tangent parameterization: a unit
/// direction `m ∈ S^{d−1}` and an orthonormal basis `E` of `m⊥`.
///
/// The frame at a mean direction `μ̂` is `Ξ(μ̂) = R(m→μ̂) E`
/// ([`frame_at`](Self::frame_at)). It is smooth except at `μ̂ = −m` (see the
/// [module documentation](self)).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LtFrame {
    /// The unit reference direction `m`.
    m: Vec<f64>,
    /// `d × p` column-major: an orthonormal basis of `m⊥`.
    e: Vec<f64>,
}

impl LtFrame {
    /// Builds the frame for the reference direction `m`, which is
    /// normalised. `E` is columns `0..d−1` of [`onb(m)`](crate::onb::onb),
    /// cleaned up by two passes of Gram–Schmidt against `m` and the preceding
    /// columns. Then `Ξ(m) = E` coincides with the basis `U_{1..d−1}` that
    /// the `γ` parameterization uses at `μ̂ = m`.
    ///
    /// `onb`'s closed form is exact in exact arithmetic, and its fallbacks
    /// (`e_j` for vanishing partial sums) are orthonormal. But when entries
    /// of `m` differ in magnitude by more than about `1e80`, its partial
    /// sums of squares underflow and it can return inaccurate columns or
    /// NaN (for example at `m = (1e-155, 0, 0, 1)`). So entries of `m̂` below
    /// `1e-70` are zeroed before calling `onb`, and the clean-up then makes
    /// `E` orthonormal and orthogonal to the true `m̂` to rounding, for every
    /// `m`. It changes a well-conditioned `onb` by rounding only, and leaves
    /// the standard frame unchanged.
    ///
    /// # Panics
    /// If `m.len() < 3`, or `m` is zero or not finite.
    pub fn new(m: &[f64]) -> Self {
        let d = m.len();
        assert!(d >= 3, "the log-tangent frame needs d >= 3, got d = {d}");
        assert!(
            m.iter().all(|v| v.is_finite()),
            "the reference direction must be finite"
        );
        // scale by the largest entry first, so the norm cannot overflow
        let big = m.iter().fold(0.0f64, |a, &v| a.max(v.abs()));
        assert!(big > 0.0, "the reference direction must be non-zero");
        let mut mh: Vec<f64> = m.iter().map(|&v| v / big).collect();
        let n = dot(&mh, &mh).sqrt();
        mh.iter_mut().for_each(|v| *v /= n);
        let p = d - 1;
        // onb divides by √(s (m_i² + s)) with partial sums s = Σ_{k<i} m_k²,
        // which underflows to 0 (and gives NaN) when s < ~1e-160 while
        // m_i = 0. Entries below FLUSH (relative to the largest, which is
        // ≥ 1/√d) are therefore zeroed for onb; Gram–Schmidt against the
        // true m then removes the O(FLUSH) error.
        let flushed: Vec<f64> = mh
            .iter()
            .map(|&x| if x.abs() < FLUSH { 0.0 } else { x })
            .collect();
        let mut e = onb(&flushed)[..p * d].to_vec();
        for j in 0..p {
            let (done, rest) = e.split_at_mut(j * d);
            let col = &mut rest[..d];
            for _ in 0..2 {
                let c = dot(col, &mh);
                col.iter_mut().zip(&mh).for_each(|(x, &b)| *x -= c * b);
                for prev in done.chunks_exact(d) {
                    let c = dot(col, prev);
                    col.iter_mut().zip(prev).for_each(|(x, &b)| *x -= c * b);
                }
            }
            let nc = dot(col, col).sqrt();
            col.iter_mut().for_each(|x| *x /= nc);
        }
        Self { m: mh, e }
    }

    /// The standard frame `m = e_d`, with `E = [e_1, …, e_{d−1}]`.
    ///
    /// # Panics
    /// If `d < 3`.
    pub fn standard(d: usize) -> Self {
        assert!(d >= 3, "the log-tangent frame needs d >= 3, got d = {d}");
        let mut m = vec![0.0; d];
        m[d - 1] = 1.0;
        Self::new(&m)
    }

    /// The dimension `d`.
    #[inline]
    pub fn dim(&self) -> usize {
        self.m.len()
    }

    /// The unit reference direction `m`.
    #[inline]
    pub fn reference(&self) -> &[f64] {
        &self.m
    }

    /// The basis `E` of `m⊥`, `d × (d−1)` column-major: `out[j*d + r]` is
    /// row `r` of column `j`.
    #[inline]
    pub fn tangent_basis(&self) -> &[f64] {
        &self.e
    }

    /// The frame `Ξ(μ̂) = R(m→μ̂) E` at `μ̂ = mu/‖mu‖`, as a `d × (d−1)`
    /// matrix: an orthonormal basis of `μ̂⊥`. The guard conventions of the
    /// [module documentation](self) apply at `mu = 0` and `μ̂ = −m`.
    ///
    /// # Panics
    /// If `mu.len() != d`.
    pub fn frame_at(&self, mu: &[f64]) -> DMatrix<f64> {
        let d = self.dim();
        assert_eq!(mu.len(), d, "mu has length {} but d = {d}", mu.len());
        let mut mu_hat = vec![0.0; d];
        let mut v = vec![0.0; d];
        let rf = reflector_into(&self.m, &self.e[..d], mu, &mut mu_hat, &mut v);
        let mut xi = DMatrix::zeros(d, d - 1);
        for (j, ej) in self.e.chunks_exact(d).enumerate() {
            let kj = 2.0 * dot(&v, ej) / rf.vv;
            for r in 0..d {
                xi[(r, j)] = ej[r] - kj * v[r];
            }
        }
        xi
    }

    /// `μ̂ = mu/‖mu‖`, or `m` at `mu = 0`.
    fn mu_hat(&self, mu: &[f64]) -> Vec<f64> {
        let d = self.dim();
        let mut mu_hat = vec![0.0; d];
        let mut v = vec![0.0; d];
        reflector_into(&self.m, &self.e[..d], mu, &mut mu_hat, &mut v);
        mu_hat
    }
}

/// What [`reflector_into`] found about `mu`.
#[derive(Clone, Copy)]
struct Reflector {
    /// `‖mu‖²`.
    mm: f64,
    /// `1/‖mu‖`, 0 at `mu = 0`.
    inv_n: f64,
    /// `v·v`.
    vv: f64,
    /// `mu ≠ 0`: `μ̂` depends on `mu`.
    mu_free: bool,
    /// `mu ≠ 0` and `μ̂ ≠ −m`: the frame depends on `mu`.
    frame_free: bool,
}

/// `μ̂ = mu/‖mu‖` (or `m` at `mu = 0`) and the reflector `v = μ̂ + m` with
/// `Ξ(μ̂) = H_v E` (or `v = E_0` at `μ̂ = −m`). `e0` is the first column of
/// `E`.
#[inline(always)]
fn reflector_into(
    m: &[f64],
    e0: &[f64],
    mu: &[f64],
    mu_hat: &mut [f64],
    v: &mut [f64],
) -> Reflector {
    let d = mu.len();
    let mut mm = 0.0;
    for k in 0..d {
        mm += mu[k] * mu[k];
    }
    let mu_free = mm > 0.0;
    let inv_n = if mu_free { 1.0 / mm.sqrt() } else { 0.0 };
    if mu_free {
        for k in 0..d {
            mu_hat[k] = mu[k] * inv_n;
        }
    } else {
        mu_hat.copy_from_slice(m);
    }
    let mut vv = 0.0;
    for k in 0..d {
        let vk = mu_hat[k] + m[k];
        v[k] = vk;
        vv += vk * vk;
    }
    let frame_free = mu_free && vv > 0.0;
    if vv == 0.0 {
        v.copy_from_slice(e0);
        vv = 0.0;
        for k in 0..d {
            vv += v[k] * v[k];
        }
    }
    Reflector {
        mm,
        inv_n,
        vv,
        mu_free,
        frame_free,
    }
}

#[inline(always)]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    let mut s = 0.0;
    for (x, y) in a.iter().zip(b) {
        s += x * y;
    }
    s
}

// =========================================================================
// Shape coordinates
// =========================================================================

/// Number of shape coordinates for a `p × p` matrix: `(p−1)(p+2)/2`.
#[inline]
const fn s_len(p: usize) -> usize {
    (p - 1) * (p + 2) / 2
}

/// `1/√(k(k+1))`, the normaliser of the Helmert vector `h_k`.
#[inline(always)]
fn helmert(k: usize) -> f64 {
    1.0 / ((k * (k + 1)) as f64).sqrt()
}

/// The symmetric traceless `p × p` matrix `S` (row-major) with shape
/// coordinates `s`: the Helmert coordinates of the diagonal, then the
/// off-diagonal pairs `a < b` in row-major order with `S_ab = S_ba = s/√2`.
/// The basis is orthonormal under the Frobenius inner product, so
/// `‖S‖_F = ‖s‖`.
///
/// # Panics
/// If `p < 1` or `s.len() != (p−1)(p+2)/2`.
pub fn s_to_matrix(s: &[f64], p: usize) -> Vec<f64> {
    assert!(p >= 1, "need p >= 1");
    assert_eq!(
        s.len(),
        s_len(p),
        "s has length {} but p = {p} needs {}",
        s.len(),
        s_len(p)
    );
    let mut out = vec![0.0; p * p];
    s_to_matrix_into(s, p, &mut out);
    out
}

/// The shape coordinates of the symmetric traceless part of the `p × p`
/// row-major matrix `m`: its orthogonal projection, in the Frobenius inner
/// product, onto the span of the basis of [`s_to_matrix`]. The inverse of
/// [`s_to_matrix`] on symmetric traceless matrices.
///
/// # Panics
/// If `p < 1` or `m.len() != p²`.
pub fn matrix_to_s(m: &[f64], p: usize) -> Vec<f64> {
    assert!(p >= 1, "need p >= 1");
    assert_eq!(m.len(), p * p, "matrix has length {} but p = {p}", m.len());
    let mut s = vec![0.0; s_len(p)];
    let mut prefix = m[0];
    for k in 1..p {
        let mkk = m[k * p + k];
        s[k - 1] = (prefix - k as f64 * mkk) * helmert(k);
        prefix += mkk;
    }
    let mut idx = p - 1;
    for a in 0..p {
        for b in a + 1..p {
            s[idx] = (m[a * p + b] + m[b * p + a]) * FRAC_1_SQRT_2;
            idx += 1;
        }
    }
    s
}

/// [`s_to_matrix`] into a buffer of length `p²`; returns `‖s‖²`.
#[inline(always)]
fn s_to_matrix_into(s: &[f64], p: usize, out: &mut [f64]) -> f64 {
    let mut ss = 0.0;
    // S_jj = Σ_{k=j+1}^{p−1} s_{k−1} h_k − j s_{j−1} h_j (h_k = helmert(k))
    let mut suffix = 0.0;
    for j in (0..p).rev() {
        let mut sjj = suffix;
        if j >= 1 {
            let c = s[j - 1] * helmert(j);
            sjj -= j as f64 * c;
            suffix += c;
            ss += s[j - 1] * s[j - 1];
        }
        out[j * p + j] = sjj;
    }
    let mut idx = p - 1;
    for a in 0..p {
        for b in a + 1..p {
            let x = s[idx] * FRAC_1_SQRT_2;
            out[a * p + b] = x;
            out[b * p + a] = x;
            ss += s[idx] * s[idx];
            idx += 1;
        }
    }
    ss
}

/// `∂/∂s` of a function of `S` with symmetric gradient `g` (only the
/// diagonal and upper triangle of the row-major `p × p` buffer are read):
/// `⟨g, B_i⟩` for each basis matrix `B_i`, scaled by `scale`.
#[inline(always)]
fn s_grad_from_upper(g: &[f64], p: usize, scale: f64, out: &mut [f64]) {
    let mut prefix = g[0];
    for k in 1..p {
        let gkk = g[k * p + k];
        out[k - 1] = scale * ((prefix - k as f64 * gkk) * helmert(k));
        prefix += gkk;
    }
    let mut idx = p - 1;
    for a in 0..p {
        for b in a + 1..p {
            out[idx] = scale * (SQRT_2 * g[a * p + b]);
            idx += 1;
        }
    }
}

// =========================================================================
// Symmetric eigen-decomposition (cyclic Jacobi, round-robin order)
// =========================================================================

/// One Jacobi rotation of a round: plane `(x, y)`, `x < y`, its cosine and
/// sine, and `t·a_xy` (the change of the diagonal).
#[derive(Clone, Copy, Default)]
struct Rot {
    x: usize,
    y: usize,
    c: f64,
    s: f64,
    ta: f64,
    on: bool,
}

/// Pair `i` of round `r` of the round-robin (circle) schedule on
/// `n = p + (p mod 2)` slots: slot `n − 1` is fixed and the others rotate.
/// For odd `p`, slot `n − 1 = p` is a dummy and pair 0 is skipped. Every
/// pair `x < y` of `0..p` occurs exactly once in the `n − 1` rounds, and
/// the pairs of a round are disjoint.
#[inline(always)]
fn rr_pair(r: usize, i: usize, n: usize) -> (usize, usize) {
    let (x, y) = if i == 0 {
        (r, n - 1)
    } else {
        ((r + i) % (n - 1), (r + n - 1 - i) % (n - 1))
    };
    if x < y { (x, y) } else { (y, x) }
}

/// Diagonalises the symmetric `p × p` matrix `a` (row-major, `p = w.len()`;
/// only the diagonal and the upper triangle are read or written) in place
/// by cyclic Jacobi in round-robin order, `A ← JᵀAJ` per rotation. `a` must
/// be scaled to unit Frobenius norm (or be zero), so that no intermediate
/// can overflow or underflow. On return `a` holds the eigenvalues on its
/// diagonal, its off-diagonal Frobenius norm is at most `JACOBI_TOL` (unless
/// `max_sweeps` ran out), and `w ← Qᵀ w`. If `WANT_Q`, the rows of `qt` are
/// rotated as well (`qt ← Qᵀ qt`), so `qt = I` on entry gives the
/// eigenvectors as its rows. `rot` needs length `≥ (p + 1)/2 + 1`. Returns
/// the number of sweeps.
///
/// The rotations of one round act on disjoint planes, so they commute, and
/// none changes another's 2×2 block. All their angles are computed from the
/// matrix at the start of the round, then all are applied. This is the
/// sequential result, and the independent angle computations overlap in
/// the processor. On random `S` with `p = 7` this takes 5.0 sweeps on
/// average, against 5.3 in row order, and is 1.5× faster; updating only the
/// upper triangle (no mirrored stores) saves another 25%.
#[inline(always)]
fn jacobi<const WANT_Q: bool>(
    a: &mut [f64],
    w: &mut [f64],
    qt: &mut [f64],
    rot: &mut [Rot],
    max_sweeps: usize,
) -> usize {
    let p = w.len();
    let n = p + (p & 1);
    let first = p & 1;
    let half = n / 2;
    let tol_sq = JACOBI_TOL * JACOBI_TOL;
    let skip = JACOBI_SKIP;
    for sweep in 0..max_sweeps {
        let mut off = 0.0;
        for k in 0..p {
            for l in k + 1..p {
                off += a[k * p + l] * a[k * p + l];
            }
        }
        // also stops on NaN
        if !(off > tol_sq) {
            return sweep;
        }
        for r in 0..n - 1 {
            for i in first..half {
                let (x, y) = rr_pair(r, i, n);
                let apq = a[x * p + y];
                let h = a[y * p + y] - a[x * p + x];
                rot[i] = if !(apq.abs() > skip) {
                    Rot {
                        x,
                        y,
                        ..Rot::default()
                    }
                } else if apq.abs() < SMALL_ANGLE * h.abs() {
                    // t = g/(2h)·(1 − O((g/h)²)) = a_xy/h and c = 1 − t²/2 = 1
                    // to rounding, as |t| < 5e-9
                    let t = apq / h;
                    Rot {
                        x,
                        y,
                        c: 1.0,
                        s: t,
                        ta: t * apq,
                        on: true,
                    }
                } else {
                    // t = tan φ is the smaller root of t² + 2τt − 1 = 0,
                    // τ = h/g, h = a_yy − a_xx, g = 2a_xy, i.e.
                    // t = sgn(h) g/(|h| + r), r = √(h² + g²); then
                    // c = 1/√(1 + t²) = (|h| + r)/√(2r(|h| + r)) and
                    // s = tc = sgn(h) g/√(2r(|h| + r)). Two square roots and
                    // one division in sequence, against three divisions in
                    // the textbook form via τ.
                    let g = 2.0 * apq;
                    let r = (h * h + g * g).sqrt();
                    let rh = r + h.abs();
                    let inv = 1.0 / (2.0 * r * rh).sqrt();
                    let sg = g * 1.0f64.copysign(h);
                    Rot {
                        x,
                        y,
                        c: rh * inv,
                        s: sg * inv,
                        ta: sg * apq / rh,
                        on: true,
                    }
                };
            }
            for i in first..half {
                let Rot { x, y, c, s, ta, on } = rot[i];
                if !on {
                    a[x * p + y] = 0.0;
                    continue;
                }
                // upper triangle only: (A_kx, A_ky) for k < x, (A_xk, A_ky)
                // for x < k < y, (A_xk, A_yk) for k > y
                for k in 0..x {
                    let (g, h) = (a[k * p + x], a[k * p + y]);
                    a[k * p + x] = c * g - s * h;
                    a[k * p + y] = s * g + c * h;
                }
                for k in x + 1..y {
                    let (g, h) = (a[x * p + k], a[k * p + y]);
                    a[x * p + k] = c * g - s * h;
                    a[k * p + y] = s * g + c * h;
                }
                for k in y + 1..p {
                    let (g, h) = (a[x * p + k], a[y * p + k]);
                    a[x * p + k] = c * g - s * h;
                    a[y * p + k] = s * g + c * h;
                }
                a[x * p + x] -= ta;
                a[y * p + y] += ta;
                a[x * p + y] = 0.0;
                let (wx, wy) = (w[x], w[y]);
                w[x] = c * wx - s * wy;
                w[y] = s * wx + c * wy;
                if WANT_Q {
                    for k in 0..p {
                        let (g, h) = (qt[x * p + k], qt[y * p + k]);
                        qt[x * p + k] = c * g - s * h;
                        qt[y * p + k] = s * g + c * h;
                    }
                }
            }
        }
    }
    max_sweeps
}

/// Eigen-decomposition of the symmetric `p × p` row-major matrix `a`:
/// `(λ, Q)` with `a = Q diag(λ) Qᵀ`, the eigenvectors as the columns of `Q`.
fn sym_eigen(a: &[f64], p: usize) -> (Vec<f64>, DMatrix<f64>) {
    let mut a = a.to_vec();
    let scale = dot(&a, &a).sqrt();
    let mut qt = vec![0.0; p * p];
    for k in 0..p {
        qt[k * p + k] = 1.0;
    }
    if scale > 0.0 {
        let inv = 1.0 / scale;
        a.iter_mut().for_each(|x| *x *= inv);
        let mut w = vec![0.0; p];
        let mut rot = vec![Rot::default(); p + 2];
        jacobi::<true>(&mut a, &mut w, &mut qt, &mut rot, EXACT_SWEEPS);
    }
    let lam = (0..p).map(|k| scale * a[k * p + k]).collect();
    (lam, DMatrix::from_fn(p, p, |r, c| qt[c * p + r]))
}

/// Coefficients `(−1)^k/(k+1)!`, `k = 0..16`, of `(1 − e^{−δ})/δ`; for
/// `0 ≤ δ < ½` the truncation error is below 2e-19.
const DD1_COEF: [f64; 16] = {
    let mut c = [1.0; 16];
    let mut k = 1;
    while k < 16 {
        c[k] = c[k - 1] * (-1.0 / (k + 1) as f64);
        k += 1;
    }
    c
};

/// The first divided difference of `x ↦ e^{−x}` at `(la, lb)`, given
/// `ea = e^{−la}` and `eb = e^{−lb}`; `−e^{−la}` at `la = lb`. For gaps
/// `δ < ½` it is `−e^{−min λ} (1 − e^{−δ})/δ` with the series of the last
/// factor (Horner), which needs no branch at `δ = 0` and is accurate for
/// every gap.
#[inline(always)]
fn exp_neg_divided_difference(la: f64, lb: f64, ea: f64, eb: f64) -> f64 {
    let delta = lb - la;
    if delta.abs() >= DD_DIRECT {
        (ea - eb) / (la - lb)
    } else {
        // factor out the larger exponential, e^{−min λ}
        let (e_lo, dl) = if delta >= 0.0 {
            (ea, delta)
        } else {
            (eb, -delta)
        };
        let mut acc = DD1_COEF[15];
        for &c in DD1_COEF[..15].iter().rev() {
            acc = acc * dl + c;
        }
        -e_lo * acc
    }
}

// =========================================================================
// Covariance and its inverse
// =========================================================================

/// Checks `mu` and `s` against the frame; returns `(d, p)`.
fn check_mu_s(frame: &LtFrame, mu: &[f64], s: &[f64]) -> (usize, usize) {
    let d = frame.dim();
    assert_eq!(mu.len(), d, "mu has length {} but d = {d}", mu.len());
    assert_eq!(
        s.len(),
        dims::gamma_len(d),
        "s has length {} but d = {d} needs {}",
        s.len(),
        dims::gamma_len(d)
    );
    (d, d - 1)
}

/// The eigen-frame of `V(μ, s)` (eigenvalues ascending, then `μ̂`) for
/// sampling, residuals and moments: the tangent eigenvalues are `exp(σ_j)`
/// for the eigenvalues `σ_j` of `S` and the eigenvectors are `Ξ(μ̂) q_j`, so one
/// `(d−1) × (d−1)` eigen-decomposition suffices (no inverse map to γ).
pub fn eigen_frame(frame: &LtFrame, mu: &[f64], s: &[f64]) -> crate::frame::Frame<f64> {
    let d = frame.dim();
    let p = d - 1;
    let (eigenvalues, eigenvectors) = sym_eigen(&s_to_matrix(s, p), p);
    let mut idx: Vec<usize> = (0..p).collect();
    idx.sort_by(|&a, &b| eigenvalues[a].total_cmp(&eigenvalues[b]));
    let xi = frame.frame_at(mu);
    let mu_norm = mu.iter().map(|v| v * v).sum::<f64>().sqrt();
    let mut basis = vec![0.0; d * d];
    for (c, &j) in idx.iter().enumerate() {
        let q = eigenvectors.column(j);
        for r in 0..d {
            basis[c * d + r] = (0..p).map(|a| xi[(r, a)] * q[a]).sum();
        }
    }
    for r in 0..d {
        basis[(d - 1) * d + r] = if mu_norm > 0.0 {
            mu[r] / mu_norm
        } else {
            frame.reference()[r]
        };
    }
    crate::frame::Frame {
        d,
        mu_norm,
        lambda: idx.iter().map(|&j| eigenvalues[j].exp()).collect(),
        basis,
    }
}

/// The ESAG covariance `V = μ̂μ̂ᵀ + Ξ exp(S) Ξᵀ` (`d × d`, symmetric, with
/// `Vμ = μ` and `det V = 1`) for mean `mu` and shape coordinates `s`, with
/// `Ξ = frame.frame_at(mu)` and `S = s_to_matrix(s, d − 1)`.
///
/// # Panics
/// If `mu.len() != d` or `s.len() != (d−2)(d+1)/2`.
pub fn covariance(frame: &LtFrame, mu: &[f64], s: &[f64]) -> DMatrix<f64> {
    let (d, p) = check_mu_s(frame, mu, s);
    let xi = frame.frame_at(mu);
    let (lam, q) = sym_eigen(&s_to_matrix(s, p), p);
    // B = Ξ Q e^{Λ/2}, V = μ̂μ̂ᵀ + B Bᵀ
    let mut b = xi * q;
    for (j, l) in lam.iter().enumerate() {
        let f = (0.5 * l).exp();
        b.column_mut(j).iter_mut().for_each(|x| *x *= f);
    }
    let mu_hat = nalgebra::DVector::from_vec(frame.mu_hat(mu));
    let v = &b * b.transpose() + &mu_hat * mu_hat.transpose();
    debug_assert_eq!(v.nrows(), d);
    // exactly symmetric: floating-point addition commutes
    let vt = v.transpose();
    (v + vt) * 0.5
}

/// The shape coordinates `s` of a covariance: `S = log(Ξᵀ V Ξ)` with
/// `Ξ = frame.frame_at(mu)`, by a symmetric eigen-decomposition, then
/// [`matrix_to_s`]. The inverse of [`covariance`].
///
/// Only the restriction of `v` to `μ̂⊥` is used, and dropping the trace of
/// the logarithm rescales it to determinant 1 (as
/// [`gamma_from_covariance`](crate::param::gamma_from_covariance) does).
/// `v` should be symmetric positive definite; its symmetric part is used,
/// and eigenvalues are floored at `1e-12` times the largest, as in
/// `gamma_from_covariance`.
///
/// # Panics
/// If `mu.len() != d` or `v` is not `d × d`.
pub fn s_from_covariance(frame: &LtFrame, mu: &[f64], v: &DMatrix<f64>) -> Vec<f64> {
    let d = frame.dim();
    assert_eq!(mu.len(), d, "mu has length {} but d = {d}", mu.len());
    assert!(
        v.nrows() == d && v.ncols() == d,
        "v is {}x{} but d = {d}",
        v.nrows(),
        v.ncols()
    );
    let p = d - 1;
    let xi = frame.frame_at(mu);
    let t = xi.transpose() * v * &xi;
    let t = (&t + t.transpose()) * 0.5;
    let t_rm: Vec<f64> = (0..p)
        .flat_map(|r| (0..p).map(move |c| (r, c)))
        .map(|(r, c)| t[(r, c)])
        .collect();
    let (lam, q) = sym_eigen(&t_rm, p);
    let floor = 1e-12 * lam.iter().fold(0.0f64, |m, &x| m.max(x.abs())).max(1e-300);
    let log_l = DMatrix::from_diagonal(&nalgebra::DVector::from_iterator(
        p,
        lam.iter().map(|&l| l.max(floor).ln()),
    ));
    let s_mat = &q * log_l * q.transpose();
    let s_rm: Vec<f64> = (0..p)
        .flat_map(|r| (0..p).map(move |c| (r, c)))
        .map(|(r, c)| s_mat[(r, c)])
        .collect();
    matrix_to_s(&s_rm, p)
}

// =========================================================================
// The log-density and its gradient
// =========================================================================

/// Checks the dimensions shared by [`loglik_lt`] and [`loglik_lt_grad`].
fn check_eta(frame: &LtFrame, eta: &[f64], y: &[f64]) -> usize {
    let d = y.len();
    assert!(d >= 3, "the ESAG density needs d >= 3, got d = {d}");
    assert_eq!(
        frame.dim(),
        d,
        "the frame has dimension {} but y has length {d}",
        frame.dim()
    );
    assert_eq!(
        eta.len(),
        dims::eta_len(d),
        "eta has length {} but d = {d} needs {}",
        eta.len(),
        dims::eta_len(d)
    );
    d
}

/// ESAG log-density at the unit vector `y` for `eta = (mu, s)` in the
/// log-tangent parameterization with reference `frame`.
///
/// The value is bit-identical to that of [`loglik_lt_grad`].
///
/// # Panics
/// If `d = y.len() < 3`, `frame.dim() != d`, or `eta.len()` is not
/// `(d²+d−2)/2`.
pub fn loglik_lt(frame: &LtFrame, eta: &[f64], y: &[f64]) -> f64 {
    let d = check_eta(frame, eta, y);
    let none: &mut [f64] = &mut [];
    match d {
        3 => lt_fixed::<3, 2, false>(frame, eta, y, none),
        4 => lt_fixed::<4, 3, false>(frame, eta, y, none),
        5 => lt_fixed::<5, 4, false>(frame, eta, y, none),
        6 => lt_fixed::<6, 5, false>(frame, eta, y, none),
        7 => lt_fixed::<7, 6, false>(frame, eta, y, none),
        8 => lt_fixed::<8, 7, false>(frame, eta, y, none),
        _ => lt_heap::<false, false>(frame, eta, y, none, &mut []),
    }
}

/// ESAG log-density at the unit vector `y` for `eta = (mu, s)` (see
/// [`loglik_lt`]) and its exact gradient with respect to `eta`, written into
/// `grad`, by one forward and one hand-written reverse pass.
///
/// # Panics
/// As [`loglik_lt`], and if `grad.len() != eta.len()`.
pub fn loglik_lt_grad(frame: &LtFrame, eta: &[f64], y: &[f64], grad: &mut [f64]) -> f64 {
    let d = check_eta(frame, eta, y);
    assert_eq!(
        grad.len(),
        eta.len(),
        "grad has length {} but eta has length {}",
        grad.len(),
        eta.len()
    );
    match d {
        3 => lt_fixed::<3, 2, true>(frame, eta, y, grad),
        4 => lt_fixed::<4, 3, true>(frame, eta, y, grad),
        5 => lt_fixed::<5, 4, true>(frame, eta, y, grad),
        6 => lt_fixed::<6, 5, true>(frame, eta, y, grad),
        7 => lt_fixed::<7, 6, true>(frame, eta, y, grad),
        8 => lt_fixed::<8, 7, true>(frame, eta, y, grad),
        _ => lt_heap::<true, false>(frame, eta, y, grad, &mut []),
    }
}

/// ESAG log-density at the unit vector `y` for `eta = (mu, s)` (see
/// [`loglik_lt`]), its gradient (into `grad`, as [`loglik_lt_grad`]) and its
/// exact Hessian with respect to `eta` (into `hess`, row-major `n × n`,
/// `n = eta.len()`, exactly symmetric), from one eigen-decomposition of `S`.
///
/// The value is bit-identical to [`loglik_lt`] and the gradient to
/// [`loglik_lt_grad`]. The derivation is in the [module documentation](self).
///
/// # Panics
/// As [`loglik_lt`], and if `grad.len() != n` or `hess.len() != n²`.
pub fn loglik_lt_hess(
    frame: &LtFrame,
    eta: &[f64],
    y: &[f64],
    grad: &mut [f64],
    hess: &mut [f64],
) -> f64 {
    let d = check_eta(frame, eta, y);
    let n = eta.len();
    assert_eq!(
        grad.len(),
        n,
        "grad has length {} but eta has length {n}",
        grad.len()
    );
    assert_eq!(
        hess.len(),
        n * n,
        "hess has length {} but eta has length {n} (needs {})",
        hess.len(),
        n * n
    );
    match d {
        3 => lt_hess_fixed::<3, 2, 2, 5, 3>(frame, eta, y, grad, hess),
        4 => lt_hess_fixed::<4, 3, 5, 9, 6>(frame, eta, y, grad, hess),
        5 => lt_hess_fixed::<5, 4, 9, 14, 10>(frame, eta, y, grad, hess),
        6 => lt_hess_fixed::<6, 5, 14, 20, 15>(frame, eta, y, grad, hess),
        7 => lt_hess_fixed::<7, 6, 20, 27, 21>(frame, eta, y, grad, hess),
        8 => lt_hess_fixed::<8, 7, 27, 35, 28>(frame, eta, y, grad, hess),
        _ => lt_heap::<true, true>(frame, eta, y, grad, hess),
    }
}

/// Scratch for the kernel: vectors of length `d` and `p`, `p × p` matrices
/// (row-major), one round of Jacobi rotations, and the Hessian's buffers
/// (empty unless the Hessian is wanted).
struct Work<'a> {
    /// `μ̂`.
    mu_hat: &'a mut [f64],
    /// The reflector `v = μ̂ + m`.
    v: &'a mut [f64],
    /// `u = H_v y`.
    u: &'a mut [f64],
    /// `ū = ∂q/∂u`, then `∂q/∂μ̂`.
    ub: &'a mut [f64],
    /// `z = Eᵀu`, rotated in place to `w = Qᵀz`.
    z: &'a mut [f64],
    /// `e^{−λ}`.
    ex: &'a mut [f64],
    /// `∂q/∂z`.
    zb: &'a mut [f64],
    /// `S`, diagonalised in place to `Λ`.
    a: &'a mut [f64],
    /// `Qᵀ` (eigenvectors as rows), then `diag(w) Qᵀ`.
    qt: &'a mut [f64],
    /// `F`, then `∂q/∂S` (upper triangle).
    fm: &'a mut [f64],
    /// `F diag(w) Qᵀ`.
    r: &'a mut [f64],
    /// One round of rotations.
    rot: &'a mut [Rot],
    /// The Hessian's buffers.
    h: HessWork<'a>,
}

/// Buffers of [`loglik_lt_hess`] (`n = d + ns` parameters, `ns` shape
/// coordinates, `np = p(p+1)/2` index pairs).
struct HessWork<'a> {
    /// The output, row-major `n × n`.
    hess: &'a mut [f64],
    /// `∇q` (`n`).
    dq: &'a mut [f64],
    /// `ū = E z̄` (`d`).
    ubar: &'a mut [f64],
    /// `Q` row-major, `qm[j*p + c] = Q_jc` (`p²`).
    qm: &'a mut [f64],
    /// The first divided differences `F` (`p²`).
    fq: &'a mut [f64],
    /// `T[c][a][b] = e^{−x}[λ_a, λ_c, λ_b]` (`p³`).
    t2: &'a mut [f64],
    /// `R_w = Q diag(w)` (`p²`).
    rw: &'a mut [f64],
    /// Scratch `p²`: `R_w T[c]`, then `Φ = F R_wᵀ`.
    xg: &'a mut [f64],
    /// Scratch `p²`: Helmert sums.
    hm: &'a mut [f64],
    /// `G^{(c)}[x][v]` by pair `{x, v}`, `gp[c*np + pair]` (`p·np`).
    gp: &'a mut [f64],
    /// `Q_uc Q_yc` by pair `{u, y}`, `aq[pair*p + c]` (`np·p`).
    aq: &'a mut [f64],
    /// `Wt` by pairs, `w2[pair(u, y)*np + pair(x, v)]` (`np²`).
    w2: &'a mut [f64],
    /// `(F ∘ B̃_i) w` for every shape coordinate, `tt[i*p + α]` (`ns·p`).
    tt: &'a mut [f64],
    /// `K = Qᵀ ∂z/∂μ̂` (`p × d`).
    kk: &'a mut [f64],
    /// Scratch (`p`).
    vx: &'a mut [f64],
    /// `pid[u*p + v]` = the index of the pair `{u, v}` (`p²`).
    pid: &'a mut [usize],
    /// Scratch (`d`).
    xv: &'a mut [f64],
    /// `∂²q/∂μ̂²` (`d²`).
    am: &'a mut [f64],
}

impl HessWork<'static> {
    /// No Hessian.
    fn empty() -> Self {
        HessWork {
            hess: &mut [],
            dq: &mut [],
            ubar: &mut [],
            qm: &mut [],
            fq: &mut [],
            t2: &mut [],
            rw: &mut [],
            xg: &mut [],
            hm: &mut [],
            gp: &mut [],
            aq: &mut [],
            w2: &mut [],
            tt: &mut [],
            kk: &mut [],
            vx: &mut [],
            pid: &mut [],
            xv: &mut [],
            am: &mut [],
        }
    }
}

/// Stack buffers for dimension `D = P + 1 ≤ 8`. Kept out of line so that
/// each dimension gets its own compact function, and re-slicing to constant
/// lengths gives every loop a compile-time trip count.
#[inline(never)]
fn lt_fixed<const D: usize, const P: usize, const GRAD: bool>(
    frame: &LtFrame,
    eta: &[f64],
    y: &[f64],
    grad: &mut [f64],
) -> f64 {
    const { assert!(D >= 3 && P + 1 == D) };
    let (mu, s) = eta.split_at(D);
    let mut dv = [[0.0; D]; 4];
    let mut pv = [[0.0; P]; 3];
    let mut pm = [[[0.0; P]; P]; 4];
    let mut rot = [Rot::default(); D];
    let [mu_hat, v, u, ub] = &mut dv;
    let [z, ex, zb] = &mut pv;
    let [a, qt, fm, r] = &mut pm;
    let w = Work {
        mu_hat,
        v,
        u,
        ub,
        z,
        ex,
        zb,
        a: a.as_flattened_mut(),
        qt: qt.as_flattened_mut(),
        fm: fm.as_flattened_mut(),
        r: r.as_flattened_mut(),
        rot: &mut rot,
        h: HessWork::empty(),
    };
    let grad = if GRAD {
        &mut grad[..dims::eta_len(D)]
    } else {
        grad
    };
    lt_core::<GRAD, false>(
        &frame.m[..D],
        &frame.e[..D * P],
        &mu[..D],
        &s[..dims::gamma_len(D)],
        &y[..D],
        grad,
        w,
    )
}

/// Stack buffers for the Hessian at dimension `D = P + 1 ≤ 8`, with
/// `NS = (D−2)(D+1)/2` shape coordinates, `NE = D + NS` parameters and
/// `NP = P(P+1)/2` index pairs.
#[inline(never)]
fn lt_hess_fixed<
    const D: usize,
    const P: usize,
    const NS: usize,
    const NE: usize,
    const NP: usize,
>(
    frame: &LtFrame,
    eta: &[f64],
    y: &[f64],
    grad: &mut [f64],
    hess: &mut [f64],
) -> f64 {
    const {
        assert!(
            D >= 3
                && P + 1 == D
                && NS == (D - 2) * (D + 1) / 2
                && NE == D + NS
                && NP == P * (P + 1) / 2
        )
    };
    let (mu, s) = eta.split_at(D);
    let mut dv = [[0.0; D]; 6];
    let mut pv = [[0.0; P]; 4];
    let mut pm = [[[0.0; P]; P]; 9];
    let mut t2 = [[[0.0; P]; P]; P];
    let mut gp = [[0.0; NP]; P];
    let mut aq = [[0.0; P]; NP];
    let mut w2 = [[0.0; NP]; NP];
    let mut tt = [[0.0; P]; NS];
    let mut kk = [[0.0; D]; P];
    let mut am = [[0.0; D]; D];
    let mut dq = [0.0; NE];
    let mut pid = [[0usize; P]; P];
    let mut rot = [Rot::default(); D];
    let [mu_hat, v, u, ub, ubar, xv] = &mut dv;
    let [z, ex, zb, vx] = &mut pv;
    let [a, qt, fm, r, qm, fq, rw, xg, hm] = &mut pm;
    let w = Work {
        mu_hat,
        v,
        u,
        ub,
        z,
        ex,
        zb,
        a: a.as_flattened_mut(),
        qt: qt.as_flattened_mut(),
        fm: fm.as_flattened_mut(),
        r: r.as_flattened_mut(),
        rot: &mut rot,
        h: HessWork {
            hess: &mut hess[..NE * NE],
            dq: &mut dq,
            ubar,
            qm: qm.as_flattened_mut(),
            fq: fq.as_flattened_mut(),
            t2: t2.as_flattened_mut().as_flattened_mut(),
            rw: rw.as_flattened_mut(),
            xg: xg.as_flattened_mut(),
            hm: hm.as_flattened_mut(),
            gp: gp.as_flattened_mut(),
            aq: aq.as_flattened_mut(),
            w2: w2.as_flattened_mut(),
            tt: tt.as_flattened_mut(),
            kk: kk.as_flattened_mut(),
            vx,
            pid: pid.as_flattened_mut(),
            xv,
            am: am.as_flattened_mut(),
        },
    };
    lt_core::<true, true>(
        &frame.m[..D],
        &frame.e[..D * P],
        &mu[..D],
        &s[..NS],
        &y[..D],
        &mut grad[..NE],
        w,
    )
}

/// Heap buffers for `d > 8`; `HESS` needs `hess` of length `n²`.
fn lt_heap<const GRAD: bool, const HESS: bool>(
    frame: &LtFrame,
    eta: &[f64],
    y: &[f64],
    grad: &mut [f64],
    hess: &mut [f64],
) -> f64 {
    let d = y.len();
    let p = d - 1;
    let (mu, s) = eta.split_at(d);
    let (ns, pp) = (s.len(), p * p);
    let mut dv = vec![0.0; 4 * d];
    let mut pv = vec![0.0; 3 * p];
    let mut pm = vec![0.0; 4 * pp];
    let mut rot = vec![Rot::default(); d];
    let (mu_hat, rest) = dv.split_at_mut(d);
    let (v, rest) = rest.split_at_mut(d);
    let (u, ub) = rest.split_at_mut(d);
    let (z, rest) = pv.split_at_mut(p);
    let (ex, zb) = rest.split_at_mut(p);
    let (a, rest) = pm.split_at_mut(pp);
    let (qt, rest) = rest.split_at_mut(pp);
    let (fm, r) = rest.split_at_mut(pp);
    let np = p * (p + 1) / 2;
    let mut pid = vec![0usize; if HESS { pp } else { 0 }];
    let mut hv = if HESS {
        vec![
            0.0;
            d + ns + 2 * d + 5 * pp + pp * p + 2 * p * np + np * np + ns * p + p * d + p + d * d
        ]
    } else {
        Vec::new()
    };
    let h = if HESS {
        let (dq, rest) = hv.split_at_mut(d + ns);
        let (ubar, rest) = rest.split_at_mut(d);
        let (xv, rest) = rest.split_at_mut(d);
        let (qm, rest) = rest.split_at_mut(pp);
        let (fq, rest) = rest.split_at_mut(pp);
        let (rw, rest) = rest.split_at_mut(pp);
        let (xg, rest) = rest.split_at_mut(pp);
        let (hm, rest) = rest.split_at_mut(pp);
        let (t2, rest) = rest.split_at_mut(pp * p);
        let (gp, rest) = rest.split_at_mut(p * np);
        let (aq, rest) = rest.split_at_mut(np * p);
        let (w2, rest) = rest.split_at_mut(np * np);
        let (tt, rest) = rest.split_at_mut(ns * p);
        let (kk, rest) = rest.split_at_mut(p * d);
        let (vx, am) = rest.split_at_mut(p);
        HessWork {
            hess,
            dq,
            ubar,
            qm,
            fq,
            t2,
            rw,
            xg,
            hm,
            gp,
            aq,
            w2,
            tt,
            kk,
            vx,
            pid: &mut pid,
            xv,
            am,
        }
    } else {
        HessWork::empty()
    };
    let w = Work {
        mu_hat,
        v,
        u,
        ub,
        z,
        ex,
        zb,
        a,
        qt,
        fm,
        r,
        rot: &mut rot,
        h,
    };
    lt_core::<GRAD, HESS>(&frame.m, &frame.e, mu, s, y, grad, w)
}

/// The forward pass, for `GRAD` the reverse sweep, and for `HESS` the
/// Hessian. See the module documentation.
#[inline(always)]
fn lt_core<const GRAD: bool, const HESS: bool>(
    m: &[f64],
    e: &[f64],
    mu: &[f64],
    s: &[f64],
    y: &[f64],
    grad: &mut [f64],
    w: Work<'_>,
) -> f64 {
    let d = mu.len();
    let p = d - 1;
    let Work {
        mu_hat,
        v,
        u,
        ub,
        z,
        ex,
        zb,
        a,
        qt,
        fm,
        r,
        rot,
        h,
    } = w;
    let (mu_hat, v, u, ub) = (&mut mu_hat[..d], &mut v[..d], &mut u[..d], &mut ub[..d]);
    let (z, ex, zb) = (&mut z[..p], &mut ex[..p], &mut zb[..p]);
    let (a, qt, fm, r) = (
        &mut a[..p * p],
        &mut qt[..p * p],
        &mut fm[..p * p],
        &mut r[..p * p],
    );

    // ---- frame: μ̂, the reflector v = μ̂ + m, z = Ξᵀy = Eᵀ(y − κv)
    let rf = reflector_into(m, &e[..d], mu, mu_hat, v);
    let (mut ymu, mut b, mut vy) = (0.0, 0.0, 0.0);
    for k in 0..d {
        ymu += y[k] * mu[k];
        b += y[k] * mu_hat[k];
        vy += y[k] * v[k];
    }
    let kappa = 2.0 * vy / rf.vv;
    for k in 0..d {
        u[k] = y[k] - kappa * v[k];
    }
    for j in 0..p {
        let ej = &e[j * d..(j + 1) * d];
        let mut acc = 0.0;
        for k in 0..d {
            acc += ej[k] * u[k];
        }
        z[j] = acc;
    }

    // ---- shape: zᵀe^{−S}z = Σ_a e^{−λ_a} w_a², w = Qᵀz (z is rotated in place)
    let ss = s_to_matrix_into(s, p, a);
    let iso = ss == 0.0;
    // the Hessian takes the general path at S = 0 too (Q = I, Λ = 0); it
    // gives the same value and gradient bit for bit
    let fast_iso = iso && !HESS;
    let zt = if fast_iso {
        let mut acc = 0.0;
        for j in 0..p {
            acc += z[j] * z[j];
        }
        acc
    } else {
        if GRAD {
            qt.fill(0.0);
            for k in 0..p {
                qt[k * p + k] = 1.0;
            }
        }
        // Jacobi on S/‖S‖_F; the eigenvalues are rescaled on the diagonal
        let scale = ss.sqrt();
        if !iso {
            let inv = 1.0 / scale;
            a.iter_mut().for_each(|x| *x *= inv);
            jacobi::<GRAD>(a, z, qt, rot, KERNEL_SWEEPS);
        }
        let mut acc = 0.0;
        for j in 0..p {
            let lj = scale * a[j * p + j];
            a[j * p + j] = lj;
            let ej = (-lj).exp();
            ex[j] = ej;
            acc += ej * z[j] * z[j];
        }
        acc
    };

    // ---- log-density
    let q = b * b + zt;
    let sq = q.sqrt();
    let alpha = ymu / sq;
    let (lmp, dlmp, d2lmp) = if GRAD {
        log_m_p_derivs(alpha, p)
    } else {
        (log_m_p_value(alpha, p), 0.0, 0.0)
    };
    let ll =
        -0.5 * p as f64 * LN_2PI - q.ln() * (0.5 * d as f64) + (alpha * alpha - rf.mm) * 0.5 + lmp;
    if !GRAD {
        return ll;
    }

    // =====================================================================
    // Reverse sweep: ∇ll = q̄ ∇q + ȳmu y − mu (mu block)
    // =====================================================================
    let ab = alpha + dlmp;
    let qb = -0.5 * (d as f64 + ab * alpha) / q;
    let ymub = ab / sq;
    let (gmu, gs) = grad.split_at_mut(d);

    // ---- ∂q/∂z into zb, ∂q/∂S (upper triangle) into fm
    if fast_iso {
        for j in 0..p {
            zb[j] = 2.0 * z[j];
        }
        for k in 0..p {
            for l in k..p {
                fm[k * p + l] = -z[k] * z[l];
            }
        }
    } else {
        if HESS {
            for c in 0..p {
                for j in 0..p {
                    h.qm[j * p + c] = qt[c * p + j];
                }
            }
        }
        // zb = 2Q(e^{−λ} ∘ w); then qt ← diag(w) Qᵀ
        zb.fill(0.0);
        for j in 0..p {
            let c = 2.0 * ex[j] * z[j];
            let wj = z[j];
            let row = &mut qt[j * p..(j + 1) * p];
            for k in 0..p {
                zb[k] += c * row[k];
                row[k] *= wj;
            }
        }
        // F: divided differences of e^{−x} at the eigenvalues
        for i in 0..p {
            fm[i * p + i] = -ex[i];
            for j in i + 1..p {
                let f = exp_neg_divided_difference(a[i * p + i], a[j * p + j], ex[i], ex[j]);
                fm[i * p + j] = f;
                fm[j * p + i] = f;
            }
        }
        if HESS {
            h.fq[..p * p].copy_from_slice(fm);
        }
        // r = F diag(w)Qᵀ; ∂q/∂S = (diag(w)Qᵀ)ᵀ r = Q (F ∘ wwᵀ) Qᵀ
        for i in 0..p {
            let ri = &mut r[i * p..(i + 1) * p];
            ri.fill(0.0);
            for j in 0..p {
                let f = fm[i * p + j];
                let row = &qt[j * p..(j + 1) * p];
                for k in 0..p {
                    ri[k] += f * row[k];
                }
            }
        }
        for k in 0..p {
            for l in k..p {
                let mut g = 0.0;
                for j in 0..p {
                    g += qt[j * p + k] * r[j * p + l];
                }
                fm[k * p + l] = g;
            }
        }
    }
    s_grad_from_upper(fm, p, qb, gs);
    if HESS {
        s_grad_from_upper(fm, p, 1.0, &mut h.dq[d..d + gs.len()]);
    }

    // ---- ū = E z̄; v̄ = −κū − (2(ū·v)/(v·v)) u; μ̂̄ = v̄ + 2b y
    if rf.mu_free {
        ub.fill(0.0);
        for j in 0..p {
            let c = zb[j];
            let ej = &e[j * d..(j + 1) * d];
            for k in 0..d {
                ub[k] += c * ej[k];
            }
        }
        if HESS {
            h.ubar[..d].copy_from_slice(ub);
        }
        let (ck, cu) = if rf.frame_free {
            let mut uv = 0.0;
            for k in 0..d {
                uv += ub[k] * v[k];
            }
            (kappa, 2.0 * uv / rf.vv)
        } else {
            (0.0, 0.0)
        };
        let mut proj = 0.0;
        for k in 0..d {
            let h = 2.0 * b * y[k] - ck * ub[k] - cu * u[k];
            ub[k] = h;
            proj += h * mu_hat[k];
        }
        // μ̂ = μ/‖μ‖: ∂q/∂μ = (μ̂̄ − (μ̂·μ̂̄) μ̂)/‖μ‖
        for k in 0..d {
            let gq = (ub[k] - proj * mu_hat[k]) * rf.inv_n;
            gmu[k] = qb * gq + ymub * y[k] - mu[k];
            if HESS {
                h.dq[k] = gq;
            }
        }
    } else {
        // μ̂ = m is constant
        for k in 0..d {
            gmu[k] = ymub * y[k] - mu[k];
        }
        if HESS {
            h.dq[..d].fill(0.0);
        }
    }

    if HESS {
        // log f = Φ(q, ymu) − ‖μ‖²/2 with Φ = −(d/2) log q + ψ(alpha),
        // ψ(a) = a²/2 + log M_p(a), alpha = ymu q^{−1/2}
        let (psi1, psi2) = (ab, 1.0 + d2lmp);
        let (a_q, a_y) = (-0.5 * alpha / q, 1.0 / sq);
        let (a_qq, a_qy) = (0.75 * alpha / (q * q), -0.5 / (q * sq));
        let fwd = Fwd {
            e,
            y,
            mu_hat,
            v,
            u,
            g: ub,
            w: z,
            ex,
            a,
            kappa,
            vv: rf.vv,
            inv_n: rf.inv_n,
            mu_free: rf.mu_free,
            frame_free: rf.frame_free,
            phi_q: qb,
            phi_qq: 0.5 * d as f64 / (q * q) + psi2 * a_q * a_q + psi1 * a_qq,
            phi_qy: psi2 * a_q * a_y + psi1 * a_qy,
            phi_yy: psi2 * a_y * a_y,
        };
        lt_hessian(&fwd, h);
    }
    ll
}

/// The forward and reverse intermediates that the Hessian needs.
struct Fwd<'a> {
    e: &'a [f64],
    y: &'a [f64],
    mu_hat: &'a [f64],
    v: &'a [f64],
    u: &'a [f64],
    /// `∂q/∂μ̂` with `μ̂` taken as a free vector (valid when `mu_free`).
    g: &'a [f64],
    /// `w = Qᵀz`.
    w: &'a [f64],
    ex: &'a [f64],
    /// `Λ` on the diagonal.
    a: &'a [f64],
    kappa: f64,
    vv: f64,
    inv_n: f64,
    mu_free: bool,
    frame_free: bool,
    /// `∂Φ/∂q` and the second derivatives of `Φ(q, ymu)`.
    phi_q: f64,
    phi_qq: f64,
    phi_qy: f64,
    phi_yy: f64,
}

/// The Hessian of the log-density (see the module documentation):
///
/// ```text
/// ∇² log f = Φ_q ∇²q + Φ_qq ∇q∇qᵀ + Φ_qy (∇q eᵀ + e∇qᵀ) + Φ_yy eeᵀ − [I 0; 0 0],
/// ```
///
/// `e = (y, 0)`, with `∇²q` from the second-order Daleckii–Krein formula
/// (shape block), the mixed derivative of `∂q/∂z` (cross block) and the
/// Householder frame (mean block). Only the upper triangle is computed;
/// the output is exactly symmetric.
#[inline(always)]
fn lt_hessian(f: &Fwd<'_>, h: HessWork<'_>) {
    let (d, p) = (f.y.len(), f.w.len());
    let pp = p * p;
    let ns = s_len(p);
    let np = p * (p + 1) / 2;
    let n = d + ns;
    let HessWork {
        hess,
        dq,
        ubar,
        qm,
        fq,
        t2,
        rw,
        xg,
        hm,
        gp,
        aq,
        w2,
        tt,
        kk,
        vx,
        pid,
        xv,
        am,
    } = h;
    let (hess, dq, ubar) = (&mut hess[..n * n], &dq[..n], &ubar[..d]);
    let (qm, fq, t2) = (&qm[..pp], &fq[..pp], &mut t2[..pp * p]);
    let (rw, xg, hm) = (&mut rw[..pp], &mut xg[..pp], &mut hm[..pp]);
    let (gp, aq, w2) = (&mut gp[..p * np], &mut aq[..np * p], &mut w2[..np * np]);
    let (tt, kk, vx) = (&mut tt[..ns * p], &mut kk[..p * d], &mut vx[..p]);
    let (xv, am, pid) = (&mut xv[..d], &mut am[..d * d], &mut pid[..pp]);
    let (y, w, mu_hat) = (f.y, f.w, f.mu_hat);

    // ---- T[c][a][b] = e^{−x}[λ_a, λ_c, λ_b], symmetric in all three
    for i in 0..p {
        for j in i..p {
            for k in j..p {
                let val = exp_neg_dd2(f.a, p, f.ex, fq, i, j, k);
                for (c, a, b) in [
                    (i, j, k),
                    (i, k, j),
                    (j, i, k),
                    (j, k, i),
                    (k, i, j),
                    (k, j, i),
                ] {
                    t2[c * pp + a * p + b] = val;
                }
            }
        }
    }

    // ---- shape block. With β(H, K) = 2 Σ_abc T[c][a][b] w_a w_b H̃_ac K̃_cb
    // (H̃ = QᵀHQ) the bilinear form of ∂²q/∂S², β(e_x e_uᵀ, e_y e_vᵀ) =
    // 2 Wt(u, y, x, v) with Wt(u, y, x, v) = Σ_c Q_uc Q_yc G_c[x][v] and
    // G_c = R_w T[c] R_wᵀ, R_w = Q diag(w). Wt is symmetric within each of
    // its index pairs, so it is an np × np matrix W = A Gᵀ.
    for x in 0..p {
        for a in 0..p {
            rw[x * p + a] = qm[x * p + a] * w[a];
        }
    }
    for c in 0..p {
        let tc = &t2[c * pp..(c + 1) * pp];
        for x in 0..p {
            let xr = &mut xg[x * p..(x + 1) * p];
            xr.fill(0.0);
            for a in 0..p {
                let ra = rw[x * p + a];
                let ta = &tc[a * p..(a + 1) * p];
                for b in 0..p {
                    xr[b] += ra * ta[b];
                }
            }
        }
        let gc = &mut gp[c * np..(c + 1) * np];
        let mut idx = 0;
        for x in 0..p {
            for v in x..p {
                gc[idx] = dot(&xg[x * p..(x + 1) * p], &rw[v * p..(v + 1) * p]);
                idx += 1;
            }
        }
    }
    let mut idx = 0;
    for uu in 0..p {
        for yy in uu..p {
            for c in 0..p {
                aq[idx * p + c] = qm[uu * p + c] * qm[yy * p + c];
            }
            idx += 1;
        }
    }
    for i in 0..np {
        let wr = &mut w2[i * np..(i + 1) * np];
        wr.fill(0.0);
        for c in 0..p {
            let ac = aq[i * p + c];
            let gc = &gp[c * np..(c + 1) * np];
            for (wv, gv) in wr.iter_mut().zip(gc) {
                *wv += ac * gv;
            }
        }
    }
    for u in 0..p {
        for v in 0..p {
            pid[u * p + v] = pair_index(u, v, p);
        }
    }
    let pid = &*pid;
    let wt = |u: usize, yy: usize, x: usize, v: usize| w2[pid[u * p + yy] * np + pid[x * p + v]];
    // off-diagonal pairs (a, b) ≤ (a2, b2): the four terms of
    // β(B_ab, B_a2b2) with B_ab = (e_a e_bᵀ + e_b e_aᵀ)/√2
    let mut i = p - 1;
    for a in 0..p {
        for b in a + 1..p {
            let mut j = i;
            for a2 in a..p {
                let b0 = if a2 == a { b } else { a2 + 1 };
                for b2 in b0..p {
                    hess[(d + i) * n + d + j] =
                        wt(b, a2, a, b2) + wt(b, b2, a, a2) + wt(a, a2, b, b2) + wt(a, b2, b, a2);
                    j += 1;
                }
            }
            i += 1;
        }
    }
    // Helmert rows against the pairs: √2 Σ_x h_k[x] (Wt(x, a, x, b) + Wt(x, b, x, a))
    let mut j = p - 1;
    for a in 0..p {
        for b in a + 1..p {
            for x in 0..p {
                vx[x] = wt(x, a, x, b) + wt(x, b, x, a);
            }
            let mut prefix = vx[0];
            for k in 1..p {
                hess[(d + k - 1) * n + d + j] = SQRT_2 * ((prefix - k as f64 * vx[k]) * helmert(k));
                prefix += vx[k];
            }
            j += 1;
        }
    }
    // Helmert against Helmert: 2 h_kᵀ M h_k', M[x][y] = Wt(x, y, x, y);
    // hm[x][k'−1] = (M h_k')[x] by prefix sums over y
    for x in 0..p {
        let mut prefix = wt(x, 0, x, 0);
        for k in 1..p {
            let mk = wt(x, k, x, k);
            hm[x * p + k - 1] = (prefix - k as f64 * mk) * helmert(k);
            prefix += mk;
        }
    }
    for k2 in 1..p {
        let mut prefix = hm[k2 - 1];
        for k in 1..p {
            let hk = hm[k * p + k2 - 1];
            if k <= k2 {
                hess[(d + k - 1) * n + d + k2 - 1] = 2.0 * ((prefix - k as f64 * hk) * helmert(k));
            }
            prefix += hk;
        }
    }

    // ---- t_i = (F ∘ B̃_i) w for the cross block: with Φ = F R_wᵀ,
    // t_ab[α] = (Q_aα Φ[α][b] + Q_bα Φ[α][a])/√2 and
    // t_k[α] = Σ_x h_k[x] Q_xα Φ[α][x]
    for al in 0..p {
        for u in 0..p {
            xg[al * p + u] = dot(&fq[al * p..(al + 1) * p], &rw[u * p..(u + 1) * p]);
        }
    }
    for al in 0..p {
        let mut prefix = qm[al] * xg[al * p];
        for k in 1..p {
            let pk = qm[k * p + al] * xg[al * p + k];
            tt[(k - 1) * p + al] = (prefix - k as f64 * pk) * helmert(k);
            prefix += pk;
        }
    }
    let mut i = p - 1;
    for a in 0..p {
        for b in a + 1..p {
            for al in 0..p {
                tt[i * p + al] = (qm[a * p + al] * xg[al * p + b]
                    + qm[b * p + al] * xg[al * p + a])
                    * FRAC_1_SQRT_2;
            }
            i += 1;
        }
    }

    // ---- mean blocks
    for r in 0..d {
        hess[r * n..(r + 1) * n].fill(0.0);
    }
    if f.mu_free {
        // A = ∂²q/∂μ̂² (μ̂ free), from (μ̂·y)²: 2yyᵀ
        for r in 0..d {
            for c in r..d {
                am[r * d + c] = 2.0 * y[r] * y[c];
            }
        }
        if f.frame_free {
            // K = Qᵀ ∂z/∂μ̂ with ∂z/∂μ̂ = Eᵀ(−κI − (2/(v·v)) v uᵀ)
            for c in 0..p {
                let kc = &mut kk[c * d..(c + 1) * d];
                kc.fill(0.0);
                for j in 0..p {
                    let qjc = qm[j * p + c];
                    let ej = &f.e[j * d..(j + 1) * d];
                    for l in 0..d {
                        kc[l] += qjc * ej[l];
                    }
                }
                let cv = 2.0 * dot(kc, f.v) / f.vv;
                for l in 0..d {
                    kc[l] = -f.kappa * kc[l] - cv * f.u[l];
                }
            }
            // cross block: ∂²q/∂μ∂s_i = P(2Kᵀ t_i)/‖μ‖, P = I − μ̂μ̂ᵀ
            for i in 0..ns {
                xv.fill(0.0);
                for c in 0..p {
                    let t = 2.0 * tt[i * p + c];
                    let kc = &kk[c * d..(c + 1) * d];
                    for l in 0..d {
                        xv[l] += t * kc[l];
                    }
                }
                let pr = dot(xv, mu_hat);
                for l in 0..d {
                    hess[l * n + d + i] = (xv[l] - pr * mu_hat[l]) * f.inv_n;
                }
            }
            // A += 2 Σ_c e^{−λ_c} K_c K_cᵀ (from ∂²q/∂z² = 2e^{−S}) + Σ_k ū_k ∇²u_k,
            // Σ_k ū_k ∇²u_k = (2κ(ū·v)/(v·v)) I + (4(ū·v)/(v·v)²)(vuᵀ + uvᵀ)
            //                − (2/(v·v))(uūᵀ + ūuᵀ)
            for c in 0..p {
                let s2 = 2.0 * f.ex[c];
                let kc = &kk[c * d..(c + 1) * d];
                for r in 0..d {
                    let kr = s2 * kc[r];
                    for cc in r..d {
                        am[r * d + cc] += kr * kc[cc];
                    }
                }
            }
            let uv = dot(ubar, f.v);
            let ci = 2.0 * f.kappa * uv / f.vv;
            let cuv = 4.0 * uv / (f.vv * f.vv);
            let cuu = 2.0 / f.vv;
            let (u, v) = (f.u, f.v);
            for r in 0..d {
                for c in r..d {
                    am[r * d + c] +=
                        cuv * (v[r] * u[c] + u[r] * v[c]) - cuu * (u[r] * ubar[c] + ubar[r] * u[c]);
                }
                am[r * d + r] += ci;
            }
        }
        // ∂²q/∂μ² = (P A P − (g⊥μ̂ᵀ + μ̂g⊥ᵀ + γP))/‖μ‖², g = ∂q/∂μ̂, γ = g·μ̂,
        // g⊥ = Pg
        for r in 0..d {
            let mut acc = 0.0;
            for c in 0..d {
                let arc = if c >= r { am[r * d + c] } else { am[c * d + r] };
                acc += arc * mu_hat[c];
            }
            xv[r] = acc;
        }
        let mam = dot(mu_hat, xv);
        let gam = dot(f.g, mu_hat);
        let in2 = f.inv_n * f.inv_n;
        for r in 0..d {
            let gr = f.g[r] - gam * mu_hat[r];
            for c in r..d {
                let gc = f.g[c] - gam * mu_hat[c];
                let pap = am[r * d + c] - mu_hat[r] * xv[c] - xv[r] * mu_hat[c]
                    + mam * mu_hat[r] * mu_hat[c];
                let prc = if r == c { 1.0 } else { 0.0 } - mu_hat[r] * mu_hat[c];
                hess[r * n + c] = in2 * (pap - (gr * mu_hat[c] + mu_hat[r] * gc + gam * prc));
            }
        }
    }

    // ---- ∇² log f from ∇²q, ∇q and e = (y, 0); upper triangle, then mirror
    let (fq_, fqq, fqy, fyy) = (f.phi_q, f.phi_qq, f.phi_qy, f.phi_yy);
    for r in 0..n {
        let dr = dq[r];
        let (cqq, cqy) = (fqq * dr, fqy * dr);
        if r < d {
            let yr = y[r];
            let (cyq, cyy) = (fqy * yr, fyy * yr);
            for c in r..d {
                let hv =
                    fq_ * hess[r * n + c] + cqq * dq[c] + (cqy * y[c] + cyq * dq[c]) + cyy * y[c]
                        - if c == r { 1.0 } else { 0.0 };
                hess[r * n + c] = hv;
                hess[c * n + r] = hv;
            }
            for c in d..n {
                let hv = fq_ * hess[r * n + c] + cqq * dq[c] + cyq * dq[c];
                hess[r * n + c] = hv;
                hess[c * n + r] = hv;
            }
        } else {
            for c in r..n {
                let hv = fq_ * hess[r * n + c] + cqq * dq[c];
                hess[r * n + c] = hv;
                hess[c * n + r] = hv;
            }
        }
    }
}

/// Index of the unordered pair `{u, v}` of `0..p` in the row-major upper
/// triangle with the diagonal (`0..p(p+1)/2`).
#[inline(always)]
fn pair_index(u: usize, v: usize, p: usize) -> usize {
    let (a, b) = if u <= v { (u, v) } else { (v, u) };
    a * (2 * p - a + 1) / 2 + (b - a)
}

/// Terms `n = 2..=DD2_TERMS` of the series of the second divided
/// difference of `e^{−x}` (truncation error below 1e-18 for gaps `< ½`).
const DD2_TERMS: usize = 17;

/// `(−1)ⁿ/n!` for `n = 0..=DD2_TERMS`.
const DD2_COEF: [f64; DD2_TERMS + 1] = {
    let mut c = [1.0; DD2_TERMS + 1];
    let mut n = 1;
    while n <= DD2_TERMS {
        c[n] = c[n - 1] * (-1.0 / n as f64);
        n += 1;
    }
    c
};

/// The second divided difference `e^{−x}[λ_i, λ_j, λ_k]` (symmetric in its
/// arguments), with `λ` on the diagonal of `a`, `ex = e^{−λ}` and the first
/// divided differences `fq`. With `lo ≤ mid ≤ hi` the sorted eigenvalues:
/// for `hi − lo ≥ ½`, `(F[lo, mid] − F[mid, hi])/(lo − hi)`; below that, the
/// series `e^{−lo} Σ_{n≥2} (−1)ⁿ h_{n−2}(mid − lo, hi − lo)/n!`, with `h_m`
/// the complete homogeneous symmetric polynomials (`h_m(x, y) = xᵐ +
/// y h_{m−1}(x, y)`), which is accurate for every gap, including ties.
#[inline(always)]
fn exp_neg_dd2(a: &[f64], p: usize, ex: &[f64], fq: &[f64], i: usize, j: usize, k: usize) -> f64 {
    let lam = |x: usize| a[x * p + x];
    let (mut lo, mut mid, mut hi) = (i, j, k);
    if lam(lo) > lam(mid) {
        std::mem::swap(&mut lo, &mut mid);
    }
    if lam(mid) > lam(hi) {
        std::mem::swap(&mut mid, &mut hi);
    }
    if lam(lo) > lam(mid) {
        std::mem::swap(&mut lo, &mut mid);
    }
    let gap = lam(hi) - lam(lo);
    if gap >= DD_DIRECT {
        (fq[lo * p + mid] - fq[mid * p + hi]) / (lam(lo) - lam(hi))
    } else {
        let d1 = lam(mid) - lam(lo);
        let (mut hm, mut xm, mut sum) = (1.0, 1.0, DD2_COEF[2]);
        for coef in &DD2_COEF[3..] {
            xm *= d1;
            hm = gap * hm + xm;
            sum += coef * hm;
        }
        ex[lo] * sum
    }
}

// =========================================================================
// A shape shared by many observations
// =========================================================================

/// The eigen-decomposition of a shape `S` that every observation shares,
/// as in regressions whose shape coefficients are zero except the
/// intercept's (the shape-constant null, intercept-only fits).
/// [`loglik_lt_grad_shared`] then costs `O(dp + p²)` per observation
/// instead of a Jacobi diagonalisation, and accumulates the shape gradient
/// in the eigenbasis, `W = Σ_i q̄_i w_i w_iᵀ`; [`SharedShape::shape_gradient`]
/// maps it back to the shape coordinates once, as `Q (F ∘ W) Qᵀ`.
#[derive(Clone, Debug)]
pub struct SharedShape {
    p: usize,
    /// The eigenvectors as rows: `qt[j*p + k] = Q_kj`.
    qt: Vec<f64>,
    /// `e^{−λ_j}`.
    ex: Vec<f64>,
    /// The first divided differences of `e^{−x}` at the eigenvalues.
    fm: Vec<f64>,
}

impl SharedShape {
    /// The decomposition of the shape with coordinates `s` (`p = d − 1`).
    ///
    /// # Panics
    /// If `s.len() != (p−1)(p+2)/2`.
    pub fn new(s: &[f64], p: usize) -> Self {
        let (lam, q) = sym_eigen(&s_to_matrix(s, p), p);
        let mut qt = vec![0.0; p * p];
        for j in 0..p {
            for k in 0..p {
                qt[j * p + k] = q[(k, j)];
            }
        }
        let ex: Vec<f64> = lam.iter().map(|l| (-l).exp()).collect();
        let mut fm = vec![0.0; p * p];
        for i in 0..p {
            fm[i * p + i] = -ex[i];
            for j in i + 1..p {
                let f = exp_neg_divided_difference(lam[i], lam[j], ex[i], ex[j]);
                fm[i * p + j] = f;
                fm[j * p + i] = f;
            }
        }
        Self { p, qt, ex, fm }
    }

    /// Length of the accumulator `W` of [`loglik_lt_grad_shared`] (a
    /// row-major `p × p`; only its upper triangle is used).
    pub fn acc_len(&self) -> usize {
        self.p * self.p
    }

    /// The gradient of `Σ_i q̄_i q_i` with respect to the shape coordinates,
    /// from the accumulator `W = Σ_i q̄_i w_i w_iᵀ`, into `out`.
    pub fn shape_gradient(&self, wacc: &[f64], out: &mut [f64]) {
        let p = self.p;
        // G = F ∘ W, symmetric
        let mut g = vec![0.0; p * p];
        for i in 0..p {
            for j in i..p {
                let v = self.fm[i * p + j] * wacc[i * p + j];
                g[i * p + j] = v;
                g[j * p + i] = v;
            }
        }
        // r = G Qᵀ, then the upper triangle of Q r = Q (F ∘ W) Qᵀ
        let mut r = vec![0.0; p * p];
        for i in 0..p {
            for j in 0..p {
                let gij = g[i * p + j];
                let row = &self.qt[j * p..(j + 1) * p];
                for l in 0..p {
                    r[i * p + l] += gij * row[l];
                }
            }
        }
        let mut m = vec![0.0; p * p];
        for k in 0..p {
            for l in k..p {
                let mut acc = 0.0;
                for i in 0..p {
                    acc += self.qt[i * p + k] * r[i * p + l];
                }
                m[k * p + l] = acc;
            }
        }
        s_grad_from_upper(&m, p, 1.0, out);
    }
}

/// [`loglik_lt_grad`] for an observation whose shape is `shared`: the
/// log-density, its gradient with respect to `mu` (into `gmu`), and the
/// observation's term `q̄ wwᵀ` of the shape gradient added to the upper
/// triangle of `wacc` (see [`SharedShape`]). The value agrees with
/// [`loglik_lt`] at the same shape to rounding.
///
/// # Panics
/// If the lengths of `mu`, `y` and `gmu` differ from `frame.dim()`, or
/// `wacc.len() != shared.acc_len()`.
pub fn loglik_lt_grad_shared(
    frame: &LtFrame,
    shared: &SharedShape,
    mu: &[f64],
    y: &[f64],
    gmu: &mut [f64],
    wacc: &mut [f64],
) -> f64 {
    let d = frame.dim();
    assert!(
        mu.len() == d && y.len() == d && gmu.len() == d && shared.p + 1 == d,
        "dimensions do not match the frame (d = {d})"
    );
    assert_eq!(
        wacc.len(),
        shared.acc_len(),
        "accumulator has the wrong length"
    );
    match d {
        3 => shared_fixed::<3, 2>(frame, shared, mu, y, gmu, wacc),
        4 => shared_fixed::<4, 3>(frame, shared, mu, y, gmu, wacc),
        5 => shared_fixed::<5, 4>(frame, shared, mu, y, gmu, wacc),
        6 => shared_fixed::<6, 5>(frame, shared, mu, y, gmu, wacc),
        7 => shared_fixed::<7, 6>(frame, shared, mu, y, gmu, wacc),
        8 => shared_fixed::<8, 7>(frame, shared, mu, y, gmu, wacc),
        _ => {
            let p = d - 1;
            let mut dv = vec![0.0; 4 * d];
            let mut pv = vec![0.0; 3 * p];
            let (mu_hat, rest) = dv.split_at_mut(d);
            let (v, rest) = rest.split_at_mut(d);
            let (u, ub) = rest.split_at_mut(d);
            let (z, rest) = pv.split_at_mut(p);
            let (w, zb) = rest.split_at_mut(p);
            shared_core(
                &frame.m, &frame.e, shared, mu, y, gmu, wacc, mu_hat, v, u, ub, z, w, zb,
            )
        }
    }
}

/// Stack buffers for [`loglik_lt_grad_shared`] at `D = P + 1 ≤ 8`.
#[inline(never)]
fn shared_fixed<const D: usize, const P: usize>(
    frame: &LtFrame,
    shared: &SharedShape,
    mu: &[f64],
    y: &[f64],
    gmu: &mut [f64],
    wacc: &mut [f64],
) -> f64 {
    let mut dv = [[0.0; D]; 4];
    let mut pv = [[0.0; P]; 3];
    let [mu_hat, v, u, ub] = &mut dv;
    let [z, w, zb] = &mut pv;
    shared_core(
        &frame.m[..D],
        &frame.e[..D * P],
        shared,
        &mu[..D],
        &y[..D],
        &mut gmu[..D],
        &mut wacc[..P * P],
        mu_hat,
        v,
        u,
        ub,
        z,
        w,
        zb,
    )
}

/// The forward pass and reverse sweep of [`lt_core`] with the shape's
/// eigen-decomposition given.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn shared_core(
    m: &[f64],
    e: &[f64],
    sh: &SharedShape,
    mu: &[f64],
    y: &[f64],
    gmu: &mut [f64],
    wacc: &mut [f64],
    mu_hat: &mut [f64],
    v: &mut [f64],
    u: &mut [f64],
    ub: &mut [f64],
    z: &mut [f64],
    w: &mut [f64],
    zb: &mut [f64],
) -> f64 {
    let d = mu.len();
    let p = d - 1;
    let (qt, ex) = (&sh.qt[..p * p], &sh.ex[..p]);
    // ---- frame, as in lt_core
    let rf = reflector_into(m, &e[..d], mu, mu_hat, v);
    let (mut ymu, mut b, mut vy) = (0.0, 0.0, 0.0);
    for k in 0..d {
        ymu += y[k] * mu[k];
        b += y[k] * mu_hat[k];
        vy += y[k] * v[k];
    }
    let kappa = 2.0 * vy / rf.vv;
    for k in 0..d {
        u[k] = y[k] - kappa * v[k];
    }
    for j in 0..p {
        let ej = &e[j * d..(j + 1) * d];
        let mut acc = 0.0;
        for k in 0..d {
            acc += ej[k] * u[k];
        }
        z[j] = acc;
    }
    // ---- shape: w = Qᵀz, zᵀe^{−S}z = Σ_j e^{−λ_j} w_j²
    let mut zt = 0.0;
    for j in 0..p {
        let row = &qt[j * p..(j + 1) * p];
        let mut acc = 0.0;
        for k in 0..p {
            acc += row[k] * z[k];
        }
        w[j] = acc;
        zt += ex[j] * acc * acc;
    }
    // ---- log-density
    let q = b * b + zt;
    let sq = q.sqrt();
    let alpha = ymu / sq;
    let (lmp, dlmp, _) = log_m_p_derivs(alpha, p);
    let ll =
        -0.5 * p as f64 * LN_2PI - q.ln() * (0.5 * d as f64) + (alpha * alpha - rf.mm) * 0.5 + lmp;
    // ---- reverse sweep: z̄ = 2Q(e^{−λ} ∘ w), W += q̄ wwᵀ
    let ab = alpha + dlmp;
    let qb = -0.5 * (d as f64 + ab * alpha) / q;
    let ymub = ab / sq;
    zb.fill(0.0);
    for j in 0..p {
        let c = 2.0 * ex[j] * w[j];
        let row = &qt[j * p..(j + 1) * p];
        for k in 0..p {
            zb[k] += c * row[k];
        }
        let cw = qb * w[j];
        for l in j..p {
            wacc[j * p + l] += cw * w[l];
        }
    }
    // ---- ū = E z̄; v̄ = −κū − (2(ū·v)/(v·v)) u; μ̂̄ = v̄ + 2b y (as in lt_core)
    if rf.mu_free {
        ub.fill(0.0);
        for j in 0..p {
            let c = zb[j];
            let ej = &e[j * d..(j + 1) * d];
            for k in 0..d {
                ub[k] += c * ej[k];
            }
        }
        let (ck, cu) = if rf.frame_free {
            let mut uv = 0.0;
            for k in 0..d {
                uv += ub[k] * v[k];
            }
            (kappa, 2.0 * uv / rf.vv)
        } else {
            (0.0, 0.0)
        };
        let mut proj = 0.0;
        for k in 0..d {
            let h = 2.0 * b * y[k] - ck * ub[k] - cu * u[k];
            ub[k] = h;
            proj += h * mu_hat[k];
        }
        for k in 0..d {
            let gq = (ub[k] - proj * mu_hat[k]) * rf.inv_n;
            gmu[k] = qb * gq + ymub * y[k] - mu[k];
        }
    } else {
        for k in 0..d {
            gmu[k] = ymub * y[k] - mu[k];
        }
    }
    ll
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The isotropic log-density is concave in `μ` (Prékopa: the integrand
    /// `r^{d−1} exp(−r²/2 + r yᵀμ − ‖μ‖²/2)` is jointly log-concave in
    /// `(r, μ)`), so isotropic regressions have a concave likelihood.
    #[test]
    fn isotropic_loglik_is_concave_in_mu() {
        for d in [3usize, 5, 8] {
            let n = dims::eta_len(d);
            let frame = LtFrame::standard(d);
            for c in 0..200 {
                let mu: Vec<f64> = (0..d)
                    .map(|k| 3.0 * ((c * 13 + k * 7) as f64 * 0.37).sin())
                    .collect();
                let yr: Vec<f64> = (0..d)
                    .map(|k| ((c * 5 + k * 3) as f64 * 1.21).cos())
                    .collect();
                let nrm = yr.iter().map(|v| v * v).sum::<f64>().sqrt();
                let y: Vec<f64> = yr.iter().map(|v| v / nrm).collect();
                let mut eta = mu.clone();
                eta.extend(std::iter::repeat_n(0.0, n - d));
                let (mut g, mut h) = (vec![0.0; n], vec![0.0; n * n]);
                loglik_lt_hess(&frame, &eta, &y, &mut g, &mut h);
                let hm = DMatrix::from_fn(d, d, |a, b| h[a * n + b]);
                let top = nalgebra::SymmetricEigen::new(hm).eigenvalues.max();
                assert!(top < 1e-9, "d={d} case {c}: largest eigenvalue {top}");
            }
        }
    }

    /// The shared-shape kernel against `loglik_lt_grad`, summed over
    /// observations: value, mu-gradient and shape gradient.
    #[test]
    fn shared_shape_matches_per_observation_kernel() {
        for d in [3usize, 4, 5, 8, 10] {
            let p = d - 1;
            let ns = dims::gamma_len(d);
            let m: Vec<f64> = (0..d).map(|k| 0.4 + 0.3 * (k as f64 * 1.7).sin()).collect();
            let frame = LtFrame::new(&m);
            let s: Vec<f64> = (0..ns)
                .map(|k| 0.6 * ((k * 5 + 1) as f64 * 0.91).sin())
                .collect();
            let shared = SharedShape::new(&s, p);
            let (mut f1, mut f2) = (0.0, 0.0);
            let mut g1 = vec![0.0; d + ns];
            let mut g2mu = vec![0.0; d];
            let mut wacc = vec![0.0; shared.acc_len()];
            let mut gs2 = vec![0.0; ns];
            for i in 0..20 {
                let mu: Vec<f64> = (0..d)
                    .map(|k| 2.0 * m[k] + 0.7 * ((i * 3 + k) as f64 * 1.3).cos())
                    .collect();
                let yr: Vec<f64> = (0..d)
                    .map(|k| ((i * 7 + k * 2) as f64 * 0.77).sin() + m[k])
                    .collect();
                let nrm = yr.iter().map(|v| v * v).sum::<f64>().sqrt();
                let y: Vec<f64> = yr.iter().map(|v| v / nrm).collect();
                let mut eta = mu.clone();
                eta.extend_from_slice(&s);
                let mut g = vec![0.0; d + ns];
                f1 += loglik_lt_grad(&frame, &eta, &y, &mut g);
                for (a, b) in g1.iter_mut().zip(&g) {
                    *a += b;
                }
                let mut gm = vec![0.0; d];
                f2 += loglik_lt_grad_shared(&frame, &shared, &mu, &y, &mut gm, &mut wacc);
                for (a, b) in g2mu.iter_mut().zip(&gm) {
                    *a += b;
                }
            }
            shared.shape_gradient(&wacc, &mut gs2);
            assert!(
                (f1 - f2).abs() < 1e-12 * f1.abs().max(1.0),
                "d={d}: {f1} vs {f2}"
            );
            for k in 0..d {
                assert!(
                    (g1[k] - g2mu[k]).abs() < 1e-10,
                    "d={d} mu[{k}]: {} vs {}",
                    g1[k],
                    g2mu[k]
                );
            }
            for k in 0..ns {
                assert!(
                    (g1[d + k] - gs2[k]).abs() < 1e-10,
                    "d={d} s[{k}]: {} vs {}",
                    g1[d + k],
                    gs2[k]
                );
            }
        }
    }

    /// splitmix64 → uniform in [lo, hi).
    struct Rng(u64);

    impl Rng {
        fn unif(&mut self) -> f64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
        }
        fn range(&mut self, lo: f64, hi: f64) -> f64 {
            lo + (hi - lo) * self.unif()
        }
        fn unit(&mut self, d: usize) -> Vec<f64> {
            let v: Vec<f64> = (0..d).map(|_| self.range(-1.0, 1.0)).collect();
            let n = dot(&v, &v).sqrt();
            v.iter().map(|x| x / n).collect()
        }
    }

    fn sym_random(p: usize, rng: &mut Rng) -> Vec<f64> {
        let mut a = vec![0.0; p * p];
        for i in 0..p {
            for j in i..p {
                let x = rng.range(-2.0, 2.0);
                a[i * p + j] = x;
                a[j * p + i] = x;
            }
        }
        a
    }

    #[test]
    fn coordinate_count_matches_gamma() {
        for d in 3..=12 {
            let p = d - 1;
            assert_eq!((p - 1) + p * (p - 1) / 2, dims::gamma_len(d));
            assert_eq!(s_len(p), dims::gamma_len(d));
        }
    }

    #[test]
    fn shape_basis_is_orthonormal_and_traceless() {
        for p in 1..=9 {
            let n = s_len(p);
            let basis: Vec<Vec<f64>> = (0..n)
                .map(|i| {
                    let mut s = vec![0.0; n];
                    s[i] = 1.0;
                    s_to_matrix(&s, p)
                })
                .collect();
            for (i, bi) in basis.iter().enumerate() {
                let tr: f64 = (0..p).map(|k| bi[k * p + k]).sum();
                assert!(tr.abs() < 1e-15, "p={p} i={i}: trace {tr}");
                for k in 0..p {
                    for l in 0..p {
                        assert_eq!(bi[k * p + l], bi[l * p + k]);
                    }
                }
                for (j, bj) in basis.iter().enumerate() {
                    let ip = dot(bi, bj);
                    let target = if i == j { 1.0 } else { 0.0 };
                    assert!((ip - target).abs() < 1e-15, "p={p} <B{i},B{j}> = {ip}");
                }
            }
        }
    }

    #[test]
    fn coordinate_maps_round_trip() {
        let mut rng = Rng(3);
        for p in 1..=9 {
            let s: Vec<f64> = (0..s_len(p)).map(|_| rng.range(-2.0, 2.0)).collect();
            let m = s_to_matrix(&s, p);
            let back = matrix_to_s(&m, p);
            for (x, y) in s.iter().zip(&back) {
                assert!((x - y).abs() < 1e-14, "p={p}");
            }
            // ‖S‖_F = ‖s‖
            assert!((dot(&m, &m) - dot(&s, &s)).abs() < 1e-13 * (1.0 + dot(&s, &s)));
            // the projection drops the trace and the antisymmetric part
            let mut m2 = m.clone();
            for k in 0..p {
                m2[k * p + k] += 0.7;
                for l in k + 1..p {
                    m2[k * p + l] += 0.3;
                    m2[l * p + k] -= 0.3;
                }
            }
            let back2 = matrix_to_s(&m2, p);
            for (x, y) in s.iter().zip(&back2) {
                assert!((x - y).abs() < 1e-14, "p={p}");
            }
        }
    }

    #[test]
    fn round_robin_covers_every_pair_once() {
        for p in 2..=13 {
            let n = p + (p & 1);
            let mut seen = vec![0u32; p * p];
            for r in 0..n - 1 {
                let mut used = vec![false; p];
                for i in (p & 1)..n / 2 {
                    let (x, y) = rr_pair(r, i, n);
                    assert!(x < y && y < p, "p={p} r={r} i={i}: ({x}, {y})");
                    assert!(!used[x] && !used[y], "p={p}: pairs of round {r} overlap");
                    used[x] = true;
                    used[y] = true;
                    seen[x * p + y] += 1;
                }
            }
            for x in 0..p {
                for y in x + 1..p {
                    assert_eq!(seen[x * p + y], 1, "p={p}: pair ({x}, {y})");
                }
            }
        }
    }

    /// Jacobi reconstructs `A = QΛQᵀ` with orthogonal `Q`, for random
    /// matrices and for matrices with repeated eigenvalues.
    #[test]
    fn jacobi_is_accurate() {
        let mut rng = Rng(11);
        let mut worst_sweeps = 0;
        for p in 2..=12 {
            for case in 0..40 {
                let a0 = match case % 4 {
                    0 | 1 => sym_random(p, &mut rng),
                    _ => {
                        // Q diag(λ) Qᵀ with λ repeated (all equal for case 3)
                        let a = sym_random(p, &mut rng);
                        let (_, q) = sym_eigen(&a, p);
                        let lam: Vec<f64> = (0..p)
                            .map(|k| if case % 4 == 3 { 1.5 } else { (k / 2) as f64 })
                            .collect();
                        let mut out = vec![0.0; p * p];
                        for i in 0..p {
                            for j in 0..p {
                                out[i * p + j] =
                                    (0..p).map(|k| q[(i, k)] * lam[k] * q[(j, k)]).sum();
                            }
                        }
                        out
                    }
                };
                let scale = dot(&a0, &a0).sqrt();
                let mut a: Vec<f64> = a0.iter().map(|x| x / scale.max(1e-300)).collect();
                let mut qt = vec![0.0; p * p];
                for k in 0..p {
                    qt[k * p + k] = 1.0;
                }
                let z0: Vec<f64> = (0..p).map(|_| rng.range(-1.0, 1.0)).collect();
                let mut w = z0.clone();
                let mut rot = vec![Rot::default(); p + 2];
                let sweeps = jacobi::<true>(&mut a, &mut w, &mut qt, &mut rot, KERNEL_SWEEPS);
                for k in 0..p {
                    a[k * p + k] *= scale;
                }
                assert!(sweeps < KERNEL_SWEEPS, "p={p}: no convergence");
                if p <= 7 {
                    worst_sweeps = worst_sweeps.max(sweeps);
                }
                for i in 0..p {
                    for j in 0..p {
                        let rec: f64 = (0..p)
                            .map(|k| qt[k * p + i] * a[k * p + k] * qt[k * p + j])
                            .sum();
                        assert!(
                            (rec - a0[i * p + j]).abs() <= 1e-14 * scale,
                            "p={p} case={case}: A[{i},{j}]"
                        );
                        let qq: f64 = (0..p).map(|k| qt[i * p + k] * qt[j * p + k]).sum();
                        let target = if i == j { 1.0 } else { 0.0 };
                        assert!((qq - target).abs() < 1e-14);
                        if i < j {
                            assert!(a[i * p + j].abs() <= 1e-15);
                        }
                    }
                    // w = Qᵀ z
                    let wi: f64 = (0..p).map(|k| qt[i * p + k] * z0[k]).sum();
                    assert!((wi - w[i]).abs() < 1e-14);
                }
            }
        }
        eprintln!("jacobi: at most {worst_sweeps} sweeps for p <= 7");
    }

    #[test]
    fn divided_differences_are_continuous() {
        for &la in &[-3.0f64, -0.2, 0.0, 1.7] {
            let ea = (-la).exp();
            for &delta in &[
                0.0f64, 1e-300, 1e-12, 1e-6, 0.3, 0.4999, 0.5, 0.5001, 2.0, -0.3, -1e-9,
            ] {
                let lb = la + delta;
                let eb = (-lb).exp();
                let got = exp_neg_divided_difference(la, lb, ea, eb);
                // reference: −e^{−la} Σ_k (−δ)^k/(k+1)!, or the plain quotient
                let exact = if delta.abs() < 0.6 {
                    let (mut term, mut sum) = (1.0, 1.0);
                    for k in 1..40 {
                        term *= -delta / (k + 1) as f64;
                        sum += term;
                    }
                    -ea * sum
                } else {
                    (ea - eb) / (la - lb)
                };
                assert!(
                    (got - exact).abs() < 2e-15 * exact.abs(),
                    "la={la} delta={delta}: {got} vs {exact}"
                );
                assert_eq!(
                    got.to_bits(),
                    exp_neg_divided_difference(lb, la, eb, ea).to_bits()
                );
            }
        }
    }

    /// The second divided difference of `e^{−x}` against the integral
    /// `∫∫_{u+v≤1} e^{−(λ_0 + u(λ_1 − λ_0) + v(λ_2 − λ_0))} du dv` (Gauss–Legendre),
    /// on both branches, across the switch at a gap of ½, and at ties.
    #[test]
    fn second_divided_differences() {
        // 40-point Gauss–Legendre on [0, 1] via Newton on P_40
        let m = 40;
        let mut nodes = Vec::new();
        for i in 0..m {
            let mut x = (std::f64::consts::PI * (i as f64 + 0.75) / (m as f64 + 0.5)).cos();
            let mut dp = 0.0;
            for _ in 0..100 {
                let (mut p0, mut p1) = (1.0, x);
                for k in 2..=m {
                    let p2 = ((2 * k - 1) as f64 * x * p1 - (k - 1) as f64 * p0) / k as f64;
                    p0 = p1;
                    p1 = p2;
                }
                dp = m as f64 * (x * p1 - p0) / (x * x - 1.0);
                let dx = p1 / dp;
                x -= dx;
                if dx.abs() < 1e-16 {
                    break;
                }
            }
            nodes.push((0.5 * (x + 1.0), 1.0 / ((1.0 - x * x) * dp * dp)));
        }
        let reference = |l: [f64; 3]| {
            // u ∈ [0, 1], v ∈ [0, 1 − u]
            let mut acc = 0.0;
            for &(u, wu) in &nodes {
                for &(t, wt) in &nodes {
                    let v = (1.0 - u) * t;
                    acc += wu
                        * wt
                        * (1.0 - u)
                        * (-(l[0] + u * (l[1] - l[0]) + v * (l[2] - l[0]))).exp();
                }
            }
            acc
        };
        for lam in [
            [0.0f64, 0.0, 0.0],
            [0.3, 0.3, 0.3],
            [-1.0, -1.0, 0.2],
            [0.1, 0.35, 0.5999],
            [0.1, 0.35, 0.6001],
            [-2.0, 0.5, 1.7],
            [1.2, -0.4, 1.2 + 1e-9],
            [3.0, -3.0, 0.0],
        ] {
            let p = 3;
            let mut a = vec![0.0; 9];
            let ex: Vec<f64> = lam.iter().map(|&l: &f64| (-l).exp()).collect();
            let mut fq = vec![0.0; 9];
            for i in 0..3 {
                a[i * 3 + i] = lam[i];
                for j in 0..3 {
                    fq[i * 3 + j] = if i == j {
                        -ex[i]
                    } else {
                        exp_neg_divided_difference(lam[i], lam[j], ex[i], ex[j])
                    };
                }
            }
            let exact = reference(lam);
            for (i, j, k) in [(0, 1, 2), (2, 0, 1), (1, 2, 0)] {
                let got = exp_neg_dd2(&a, p, &ex, &fq, i, j, k);
                assert!(
                    (got - exact).abs() < 2e-14 * exact,
                    "{lam:?}: {got} vs {exact}"
                );
            }
        }
    }

    /// The O(d) Householder form of the frame agrees with the closed form
    /// `R = I + K + K²/(1 + mᵀμ̂)`, `K = μ̂mᵀ − mμ̂ᵀ`, and `R` is the minimal
    /// rotation: orthogonal, `R m = μ̂`, identity on `span(m, μ̂)⊥`.
    #[test]
    fn frame_matches_closed_form_rotation() {
        let mut rng = Rng(5);
        for d in 3..=9 {
            let frame = LtFrame::new(&rng.unit(d));
            let m = frame.reference().to_vec();
            for _ in 0..20 {
                let mu: Vec<f64> = rng.unit(d).iter().map(|x| 3.0 * x).collect();
                let mh: Vec<f64> = mu.iter().map(|x| x / dot(&mu, &mu).sqrt()).collect();
                let c = dot(&m, &mh);
                if c < -0.99 {
                    continue;
                }
                let k = DMatrix::from_fn(d, d, |i, j| mh[i] * m[j] - m[i] * mh[j]);
                let rm = DMatrix::identity(d, d) + &k + &k * &k / (1.0 + c);
                let rtr = rm.transpose() * &rm;
                assert!((rtr - DMatrix::identity(d, d)).norm() < 1e-13);
                let rmm = &rm * nalgebra::DVector::from_vec(m.clone());
                for i in 0..d {
                    assert!((rmm[i] - mh[i]).abs() < 1e-14);
                }
                // a vector orthogonal to m and μ̂ is fixed
                let mut q2: Vec<f64> = mh.iter().zip(&m).map(|(a, b)| a - c * b).collect();
                let n2 = dot(&q2, &q2).sqrt();
                q2.iter_mut().for_each(|a| *a /= n2);
                let mut x = rng.unit(d);
                for q in [&m, &q2] {
                    let cx = dot(&x, q);
                    x.iter_mut().zip(q.iter()).for_each(|(a, &b)| *a -= cx * b);
                }
                let rx = &rm * nalgebra::DVector::from_vec(x.clone());
                for i in 0..d {
                    assert!((rx[i] - x[i]).abs() < 1e-13);
                }
                let e = DMatrix::from_column_slice(d, d - 1, frame.tangent_basis());
                let xi = frame.frame_at(&mu);
                assert!((&rm * &e - &xi).norm() < 1e-13, "d={d}");
                // z = Eᵀ Rᵀ y, the kernel's O(d) form
                let y = rng.unit(d);
                let z_ref = e.transpose() * rm.transpose() * nalgebra::DVector::from_vec(y.clone());
                let z = xi.transpose() * nalgebra::DVector::from_vec(y);
                assert!((z_ref - z).norm() < 1e-13);
            }
        }
    }

    #[test]
    fn frame_is_orthonormal_for_every_reference() {
        let mut cases: Vec<Vec<f64>> = Vec::new();
        for d in 3..=9 {
            let mut rng = Rng(d as u64);
            for _ in 0..20 {
                cases.push(rng.unit(d));
            }
            for k in 0..d {
                for sgn in [1.0, -1.0] {
                    let mut e = vec![0.0; d];
                    e[k] = sgn;
                    cases.push(e);
                }
            }
            // badly scaled entries: onb's partial sums become subnormal
            for tiny in [1e-155, 1e-161, 1e-200, 1e-300, 5e-324] {
                let mut a = vec![0.0; d];
                a[0] = tiny;
                a[d - 1] = 1.0;
                cases.push(a.clone());
                let mut b: Vec<f64> = (0..d).map(|i| tiny * (1.0 + i as f64)).collect();
                b[d - 1] = -2.0;
                cases.push(b);
                let mut c = vec![tiny; d];
                c[1] = 1.0;
                cases.push(c);
            }
            cases.push(vec![1e300; d]);
            cases.push(
                (0..d)
                    .map(|i| if i % 2 == 0 { 1e-170 } else { 1.0 })
                    .collect(),
            );
        }
        for m in cases {
            let d = m.len();
            let f = LtFrame::new(&m);
            let mh = f.reference();
            assert!((dot(mh, mh) - 1.0).abs() < 1e-15, "{m:?}");
            let e = f.tangent_basis();
            for i in 0..d - 1 {
                let ci = &e[i * d..(i + 1) * d];
                assert!(dot(ci, mh).abs() < 1e-15, "{m:?}: E_{i}·m");
                for j in 0..d - 1 {
                    let target = if i == j { 1.0 } else { 0.0 };
                    let ip = dot(ci, &e[j * d..(j + 1) * d]);
                    assert!((ip - target).abs() < 1e-15, "{m:?}: E_{i}·E_{j} = {ip}");
                }
            }
        }
        // the standard frame is exactly [e_1, …, e_{d−1}]
        for d in 3..=9 {
            let f = LtFrame::standard(d);
            for j in 0..d - 1 {
                for r in 0..d {
                    assert_eq!(f.tangent_basis()[j * d + r], if r == j { 1.0 } else { 0.0 });
                }
            }
        }
    }

    /// Value, gradient and Hessian agree on every path, including the
    /// guarded `μ = 0` and `μ̂ = −m`.
    #[test]
    fn value_and_gradient_paths_agree_bitwise() {
        let mut rng = Rng(9);
        for d in 3..=11 {
            for case in 0..30 {
                let frame = if case % 2 == 0 {
                    LtFrame::standard(d)
                } else {
                    LtFrame::new(&rng.unit(d))
                };
                let mut eta: Vec<f64> = (0..dims::eta_len(d))
                    .map(|_| rng.range(-1.5, 1.5))
                    .collect();
                match case % 5 {
                    1 => eta[d..].iter_mut().for_each(|x| *x = 0.0),
                    2 => eta[..d].iter_mut().for_each(|x| *x = 0.0),
                    3 => {
                        // μ̂ = −m exactly
                        let m = frame.reference().to_vec();
                        for k in 0..d {
                            eta[k] = -2.0 * m[k];
                        }
                    }
                    _ => {}
                }
                let y = rng.unit(d);
                let n = eta.len();
                let (mut g, mut g2, mut h) = (vec![0.0; n], vec![0.0; n], vec![0.0; n * n]);
                let v1 = loglik_lt(&frame, &eta, &y);
                let v2 = loglik_lt_grad(&frame, &eta, &y, &mut g);
                let v3 = loglik_lt_hess(&frame, &eta, &y, &mut g2, &mut h);
                assert_eq!(v1.to_bits(), v2.to_bits(), "d={d} case={case}");
                assert_eq!(v1.to_bits(), v3.to_bits(), "d={d} case={case}");
                assert_eq!(g, g2, "d={d} case={case}");
                assert!(v1.is_finite() && g.iter().all(|x| x.is_finite()));
                assert!(h.iter().all(|x| x.is_finite()), "d={d} case={case}");
            }
        }
    }

    /// At `s = 0` the kernel skips the eigen-decomposition and the
    /// `s`-gradient is `q̄ ⟨−zzᵀ, B_i⟩` exactly.
    #[test]
    fn isotropic_gradient_is_exact() {
        let mut rng = Rng(21);
        for d in 3..=9 {
            let frame = LtFrame::new(&rng.unit(d));
            let mut eta: Vec<f64> = (0..dims::eta_len(d))
                .map(|_| rng.range(-2.0, 2.0))
                .collect();
            eta[d..].iter_mut().for_each(|x| *x = 0.0);
            let y = rng.unit(d);
            let mut g = vec![0.0; eta.len()];
            loglik_lt_grad(&frame, &eta, &y, &mut g);
            let mu = &eta[..d];
            let xi = frame.frame_at(mu);
            let z = xi.transpose() * nalgebra::DVector::from_vec(y.clone());
            let p = d - 1;
            let q = 1.0;
            let ymu = dot(&y, mu);
            let alpha = ymu / q;
            let (_, d1, _) = log_m_p_derivs(alpha, p);
            let qb = -0.5 * (d as f64 + (alpha + d1) * alpha) / q;
            let gz: Vec<f64> = (0..p * p).map(|i| -z[i / p] * z[i % p] * qb).collect();
            let expect = matrix_to_s(&gz, p);
            for (i, (a, b)) in g[d..].iter().zip(&expect).enumerate() {
                assert!(
                    (a - b).abs() < 1e-14 * (1.0 + b.abs()),
                    "d={d} i={i}: {a} vs {b}"
                );
            }
        }
    }

    /// Near the antipode of `m` the frame turns fast, but values and
    /// derivatives stay finite, and the gradient and the Hessian's mean
    /// columns match finite differences at a step that resolves the scale
    /// `‖μ̂ + m‖`.
    #[test]
    fn near_the_antipode() {
        let mut rng = Rng(77);
        for d in [3usize, 5, 8] {
            let frame = LtFrame::new(&rng.unit(d));
            let m = frame.reference().to_vec();
            let e0 = frame.tangent_basis()[..d].to_vec();
            for &eps in &[1e-2, 1e-4] {
                let mu: Vec<f64> = (0..d).map(|k| 2.0 * (-m[k] + eps * e0[k])).collect();
                let mut eta = mu.clone();
                eta.extend((0..dims::gamma_len(d)).map(|_| rng.range(-1.0, 1.0)));
                let y = rng.unit(d);
                let mut g = vec![0.0; eta.len()];
                let f0 = loglik_lt_grad(&frame, &eta, &y, &mut g);
                assert!(f0.is_finite());
                let h = 1e-4 * eps;
                let n = eta.len();
                let mut hess = vec![0.0; n * n];
                let mut g0 = vec![0.0; n];
                loglik_lt_hess(&frame, &eta, &y, &mut g0, &mut hess);
                for i in 0..d {
                    let mut e = eta.clone();
                    e[i] += h;
                    let mut gp = vec![0.0; n];
                    let fp = loglik_lt_grad(&frame, &e, &y, &mut gp);
                    e[i] -= 2.0 * h;
                    let mut gm = vec![0.0; n];
                    let fm = loglik_lt_grad(&frame, &e, &y, &mut gm);
                    let fd = (fp - fm) / (2.0 * h);
                    assert!(
                        (g[i] - fd).abs() < 1e-5 * fd.abs().max(1.0),
                        "d={d} eps={eps} i={i}: {} vs {fd}",
                        g[i]
                    );
                    for r in 0..n {
                        let hd = (gp[r] - gm[r]) / (2.0 * h);
                        assert!(
                            (hess[r * n + i] - hd).abs() < 1e-4 * hd.abs().max(1.0),
                            "d={d} eps={eps}: H[{r},{i}] = {} vs {hd}",
                            hess[r * n + i]
                        );
                    }
                }
            }
        }
    }
}
