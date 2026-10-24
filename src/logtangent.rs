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
//! `∂q/∂S = −zzᵀ` and `∂q/∂z = 2z`, exactly. The value returned by the two
//! functions is bit-identical: they perform the same floating-point
//! operations up to the log-density. For `d ≤ 8` they work on stack buffers
//! only, and larger `d` fall back to heap buffers. See `docs/MATH.md`,
//! "Log-tangent shape parameterization (smooth option)".

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
/// factor `1/(1 − e^{−0.5}) ≈ 2.5` to cancellation. Below it they use
/// `expm1`.
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

/// The first divided difference of `x ↦ e^{−x}` at `(la, lb)`, given
/// `ea = e^{−la}` and `eb = e^{−lb}`; `−e^{−la}` at `la = lb`.
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
        if dl == 0.0 {
            -e_lo
        } else {
            e_lo * (-dl).exp_m1() / dl
        }
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
        _ => lt_heap::<false>(frame, eta, y, none),
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
        _ => lt_heap::<true>(frame, eta, y, grad),
    }
}

/// Scratch for the kernel: vectors of length `d` and `p`, `p × p` matrices
/// (row-major), and one round of Jacobi rotations.
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
    };
    let grad = if GRAD {
        &mut grad[..dims::eta_len(D)]
    } else {
        grad
    };
    lt_core::<GRAD>(
        &frame.m[..D],
        &frame.e[..D * P],
        &mu[..D],
        &s[..dims::gamma_len(D)],
        &y[..D],
        grad,
        w,
    )
}

/// Heap buffers for `d > 8`.
fn lt_heap<const GRAD: bool>(frame: &LtFrame, eta: &[f64], y: &[f64], grad: &mut [f64]) -> f64 {
    let d = y.len();
    let p = d - 1;
    let (mu, s) = eta.split_at(d);
    let mut dv = vec![0.0; 4 * d];
    let mut pv = vec![0.0; 3 * p];
    let mut pm = vec![0.0; 4 * p * p];
    let mut rot = vec![Rot::default(); d];
    let (mu_hat, rest) = dv.split_at_mut(d);
    let (v, rest) = rest.split_at_mut(d);
    let (u, ub) = rest.split_at_mut(d);
    let (z, rest) = pv.split_at_mut(p);
    let (ex, zb) = rest.split_at_mut(p);
    let (a, rest) = pm.split_at_mut(p * p);
    let (qt, rest) = rest.split_at_mut(p * p);
    let (fm, r) = rest.split_at_mut(p * p);
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
    };
    lt_core::<GRAD>(&frame.m, &frame.e, mu, s, y, grad, w)
}

/// The forward pass, and for `GRAD` the reverse sweep. See the module
/// documentation.
#[inline(always)]
fn lt_core<const GRAD: bool>(
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
    let zt = if iso {
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
        let inv = 1.0 / scale;
        a.iter_mut().for_each(|x| *x *= inv);
        jacobi::<GRAD>(a, z, qt, rot, KERNEL_SWEEPS);
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
    let (lmp, dlmp) = if GRAD {
        let (l, d1, _) = log_m_p_derivs(alpha, p);
        (l, d1)
    } else {
        (log_m_p_value(alpha, p), 0.0)
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
    if iso {
        for j in 0..p {
            zb[j] = 2.0 * z[j];
        }
        for k in 0..p {
            for l in k..p {
                fm[k * p + l] = -z[k] * z[l];
            }
        }
    } else {
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

    // ---- ū = E z̄; v̄ = −κū − (2(ū·v)/(v·v)) u; μ̂̄ = v̄ + 2b y
    if !rf.mu_free {
        // μ̂ = m is constant
        for k in 0..d {
            gmu[k] = ymub * y[k] - mu[k];
        }
        return ll;
    }
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
    // μ̂ = μ/‖μ‖: ∂q/∂μ = (μ̂̄ − (μ̂·μ̂̄) μ̂)/‖μ‖
    for k in 0..d {
        let gq = (ub[k] - proj * mu_hat[k]) * rf.inv_n;
        gmu[k] = qb * gq + ymub * y[k] - mu[k];
    }
    ll
}

#[cfg(test)]
mod tests {
    use super::*;

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
                let mut g = vec![0.0; eta.len()];
                let v1 = loglik_lt(&frame, &eta, &y);
                let v2 = loglik_lt_grad(&frame, &eta, &y, &mut g);
                assert_eq!(v1.to_bits(), v2.to_bits(), "d={d} case={case}");
                assert!(v1.is_finite() && g.iter().all(|x| x.is_finite()));
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
    /// gradients stay finite and the gradient matches finite differences at
    /// a step that resolves the scale `‖μ̂ + m‖`.
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
                for i in 0..d {
                    let mut e = eta.clone();
                    e[i] += h;
                    let fp = loglik_lt(&frame, &e, &y);
                    e[i] -= 2.0 * h;
                    let fm = loglik_lt(&frame, &e, &y);
                    let fd = (fp - fm) / (2.0 * h);
                    assert!(
                        (g[i] - fd).abs() < 1e-5 * fd.abs().max(1.0),
                        "d={d} eps={eps} i={i}: {} vs {fd}",
                        g[i]
                    );
                }
            }
        }
    }
}
