//! Spectral shape: the eigen-decomposition of the tangent covariance as the
//! parameters.
//!
//! # The chart
//!
//! Write `W` for the `(d−1) × (d−1)` covariance on the tangent space (unit
//! determinant) in the frame `Ξ(μ̂) = R(m → μ̂) E R_0` transported from a
//! reference direction `m` ([`crate::logtangent::LtFrame`]) and turned by a
//! fixed rotation `R_0` (the identity unless [`SpFrame::with_basis`]), so
//! that `V = Ξ W Ξᵀ + μ̂μ̂ᵀ`. The chart takes the eigen-decomposition of `W`
//! as the parameters, with `p = d − 1`:
//!
//! ```text
//! W = R diag(e^{ℓ}) Rᵀ,    Σ_k ℓ_k = 0,    R = (I − A/2)⁻¹ (I + A/2),    Aᵀ = −A
//! ```
//!
//! * `ℓ ∈ ℝ^p` are the log-eigenvalues, given by `p − 1` Helmert
//!   coordinates `h`: `ℓ = H h`, where column `m` (`m = 1..p−1`) of the
//!   `p × (p−1)` matrix `H` is `(1, …, 1, −m, 0, …, 0)/√(m(m+1))` with `m`
//!   ones. The columns are an orthonormal basis of the sum-zero subspace, so
//!   the eigenvalues `e^{ℓ_k}` have product 1 for every `h`, `h = Hᵀ ℓ`, and
//!   `h = 0` is isotropy.
//! * `R ∈ SO(p)` holds the eigenvectors: column `k` is the axis of
//!   `e^{ℓ_k}`. It is the Cayley transform of the skew-symmetric `A` whose
//!   strict upper triangle, row by row, is the remaining `p(p−1)/2`
//!   coordinates `a` (`A_rc = a_k`, `A_cr = −a_k`). `A = 0` is `R = I`: the
//!   axes are the frame's own.
//!
//! The coordinates are `γ = (h_1, …, h_{p−1}, a_1, …, a_{p(p−1)/2})`, of
//! length `(d−2)(d+1)/2 = `[`dims::gamma_len`]`(d)`, the length of every
//! other shape, so `eta = (μ, γ)` has [`dims::eta_len`]`(d)` entries.
//!
//! The log-density of the unit vector `y` is, with `t = μ̂·y`, `z = Ξᵀ y`
//! and `w = Rᵀ z`,
//!
//! ```text
//! q = t² + Σ_k w_k² e^{−ℓ_k},   α = (y·μ)/√q,
//! ll = −½ p ln 2π − (d/2) ln q + ½(α² − ‖μ‖²) + log M_p(α).
//! ```
//!
//! # Why this chart
//!
//! **Ordering-free.** `γ` and `γ`-rot build `W` one axis at a time: the
//! first two axes follow Paine's map, axis `j + 1` is then placed inside the
//! first `j + 1` axes, so the labelling of the axes is part of the structure
//! of the map, and a regression `γ(x) = Γ x̃` says different things about
//! the first and the last axis. Here every axis enters alike. Relabelling
//! the frame axes by a permutation `σ` (`R_0 ↦ R_0 P_σ`) maps
//! `ℓ ↦ P_σᵀ ℓ` and `R ↦ P_σᵀ R P_σ = Cayley(P_σᵀ A P_σ)`: it permutes the
//! rows of `ℓ` (a linear map of `h`) and the rows and columns of `A` (a
//! signed permutation of the `a_k`), gives the same `W`, and keeps an affine
//! `(h(x), a(x))` affine. The regression model class is therefore the same
//! for every labelling; no axis is privileged.
//!
//! **Smooth; flat, not broken, at the degenerate points.** `(h, a) ↦ W` is
//! `C^∞` on all of `ℝ^{(d−2)(d+1)/2}`: `e^{ℓ}` and the Cayley transform
//! (rational, with `I − A/2` of singular values `≥ 1`) have no singular
//! point, so a regression has a likelihood without cliffs. What the chart
//! has instead, like every chart made of eigenvalues and eigenvectors, is
//! non-identifiability: at an exact tie `ℓ_j = ℓ_k` the rotation within the
//! `(j, k)` plane does not change `W` (one flat direction per tied pair),
//! and at exact isotropy `ℓ = 0` every `a_k` is flat (the information is
//! singular there, and the chart is a polar-like one around isotropy, where
//! `γ`'s first group is not). These are flat directions of a smooth map, not
//! discontinuities: the log-density and its gradient are finite and
//! continuous through them ([`loglik_sp_grad`] is tested there).
//!
//! **Discrete gauge.** The map is not injective: flipping the sign of an
//! axis (`R ↦ R D`, `D = diag(±1)` with `det D = 1`) and relabelling the
//! axes give the same `W` from other coordinates.
//! [`SpFrame::coords_from_covariance`] returns the canonical member:
//! eigenvalues ascending along the axes, and among the sign choices the
//! proper rotation `R` of largest trace (the one nearest to `I`).
//! [`SpFrame::coords_from_eigen_signed`] keeps the caller's labels and
//! signs instead, so that the coordinates of neighbouring `V`s vary
//! continuously along a path.
//!
//! **Cayley, not the exponential.** With `±iθ_j` the eigenvalues of `A`,
//! `R` turns the `j`-th invariant plane of `A` by `2 atan(θ_j/2) < π`: there
//! is no wrap-around (`e^A` is periodic in `A`), every entry of `W` is
//! rational in `a`, and the kernel needs one `p × p` solve instead of a
//! matrix exponential. The price is that a rotation by exactly `π` in some
//! plane is not representable (it is the limit `‖a‖ → ∞`);
//! [`SpFrame::coords_from_covariance`] returns `NaN` for the `a` part there
//! (the `h` part stays valid). In the canonical gauge this needs `p ≥ 4`
//! and a rotation whose diagonal vanishes, which a sensible `R_0` keeps far
//! away.
//!
//! **References.** The chart depends on `m` and `R_0` as the log-tangent
//! shape depends on `m`: with `R_0` the eigenframe of a pilot estimate at
//! the mean design row, `R(x̄) ≈ I`, so `a(x̄) ≈ 0` and the rotations a
//! regression describes are small near the data's centre, far from the
//! Cayley singularity at `π`.
//!
//! # Kernels
//!
//! [`loglik_sp`] is the generic kernel (dual numbers work; it does one
//! Gaussian elimination with partial pivoting on real parts per
//! observation), [`loglik_sp_grad`] the reverse-mode gradient (`O(p³)` per
//! observation, no dual numbers, value bit-identical to `loglik_sp::<f64>`),
//! [`loglik_sp_grad_fwd`] the forward-mode gradient (the reference the
//! reverse one is tested against), [`loglik_sp_hess`] the exact Hessian
//! (below) and [`loglik_sp_hess_fd`] the Hessian by central differences of
//! [`loglik_sp_grad`].
//!
//! The reverse sweep, with `T = I + A/2`, `S = I − A/2 = Tᵀ` (so
//! `Rᵀ = S T⁻¹`), `s̃ = T⁻¹ z` and `w = S s̃`, seeds `q̄ = 1`:
//! `w̄_k = 2 w_k e^{−ℓ_k}`, `ℓ̄_k = −w_k² e^{−ℓ_k}` (then `h̄ = Hᵀ ℓ̄`);
//! `s̄̃ = T w̄`, `z̄ = T⁻ᵀ s̄̃`; the two uses of `A` give
//! `Ā = −½ (w̄ + z̄) s̃ᵀ`, so `ā_k = Ā_rc − Ā_cr = ½(s̃_r (w̄_c + z̄_c) −
//! s̃_c (w̄_r + z̄_r))` — `O(p²)` after the one `LU` of `T`. Then
//! `z̄ ← R_0 z̄` and the frame's sweep of [`crate::gamma_transported`]
//! (`μ = 0` and `μ̂ = −m` have constant frames there).
//!
//! # The Hessian
//!
//! [`loglik_sp_hess`] runs the reverse kernel (its value and gradient are
//! those of [`loglik_sp_grad`], bit for bit) and adds the closed-form
//! second derivatives of the Cayley transform. Let `M = (I − A/2)⁻¹ = T⁻ᵀ`
//! (`p` solves with the one `LU` of `T`). Then `R = 2M − I`, `s̃ = Mᵀz`,
//! `w = 2s̃ − z`, and with `E_m = e_i e_jᵀ − e_j e_iᵀ` for `a_m ↔ (i < j)`,
//!
//! ```text
//! ∂R/∂a_m = M E_m M,    ∂²R/∂a_l∂a_m = ½ M (E_l M E_m + E_m M E_l) M.
//! ```
//!
//! With `C = diag(e^{−ℓ})`, `w̄ = 2Cw`, `g̃ = M w̄ = ½(w̄ + z̄)` (the reverse
//! sweep forms it) and the two-entry vectors `σ_m = E_m s̃`, `γ_m = E_m g̃`,
//! `w` has the derivatives `D_m = ∂w/∂a_m = −Mᵀσ_m`, and the shape block of
//! `∂²q` is, for `a_l ↔ (k < l′)`, `N = MCMᵀ` and `Ω_m = 2Nσ_m − ½Mγ_m`,
//!
//! ```text
//! ∂²q/∂a_m∂a_l = 2σ_lᵀNσ_m − ½(σ_lᵀMγ_m + σ_mᵀMγ_l)
//!              = s̃_l′ Ω_m[k] − s̃_k Ω_m[l′] + ½(g̃_l′ D_m[k] − g̃_k D_m[l′]),
//! ∂²q/∂ℓ_k∂a_m = −2 c_k w_k D_m[k],        ∂²q/∂ℓ_k∂ℓ_l = δ_kl c_k w_k²:
//! ```
//!
//! `O(p)` per pair `m`, then `O(1)` per entry, and `h = Hᵀℓ` by prefix
//! sums. For the cross and mean blocks, `∂q/∂z = 2RCw` has the derivatives
//! `Mγ_m + 2RCD_m` along `a_m` and `−2c_k w_k R_{:,k}` along `ℓ_k`. With
//! the frame Jacobian `J = ∂z/∂μ̂` (turned by `R_0`), `Y = MᵀJ`,
//! `U = RᵀJ = 2Y − J` (that is `∂w/∂μ̂`), `Z = MCU`, and `Y_i`, `Z_i`, `U_k`
//! their rows,
//!
//! ```text
//! Jᵀ ∂²q/∂z∂a_m = (g̃_j Y_i − g̃_i Y_j) − 2(s̃_j Z_i − s̃_i Z_j),
//! Jᵀ ∂²q/∂z∂ℓ_k = −2 c_k w_k U_k,        Jᵀ (∂²q/∂z²) J = 2UᵀCU.
//! ```
//!
//! The frame's own curvature, `t²`, the chain through `μ̂ = μ/‖μ‖` and the
//! outer layer `∇² log f` from `∇²q` are those of the log-tangent kernel
//! (`crate::frame_hess`). Nothing divides by an eigenvalue gap or by
//! `‖a‖`, so ties, isotropy and `A = 0` are ordinary points; at `μ = 0`
//! and `μ̂ = −m` the Hessian keeps the gradient's constant frames. The
//! cost is `O(p⁴)` (the Cayley block has `p(p−1)/2` rows), against the
//! `O(p⁵)` of the log-tangent kernel's Daleckii–Krein shape block. Against
//! fourth-order differences of the gradient the entries agree to the
//! differences' own error: below `5e−9` of `max(|H_ij|, 1)` at a step of
//! `1e−3`, `2e−11` at `2e−4` (the tests require `1e−6` at random and
//! special points, `d = 3..=11`).
//!
//! # Cost
//!
//! Nanoseconds per observation for the log-density and its gradient
//! (`bench::kernel_cost`, 20 000 random observations, best of 5, Apple M5
//! Pro, `--release`), against the forward-mode spectral kernel
//! ([`loglik_sp_grad_fwd`]), the log-tangent kernel
//! ([`loglik_lt_grad`](crate::logtangent::loglik_lt_grad)) and the reverse
//! `γ`-rot kernel
//! ([`loglik_gr_grad_rev`](crate::gamma_rot_adjoint::loglik_gr_grad_rev)):
//!
//! | `d` | `N` | spectral reverse | spectral forward | log-tangent | `γ`-rot reverse |
//! |---|---|---|---|---|---|
//! | 3  | 5  | 54  | 100  | 63   | 49  |
//! | 5  | 14 | 115 | 346  | 323  | 71  |
//! | 8  | 35 | 224 | 2560 | 1187 | 196 |
//! | 10 | 54 | 339 | 6862 | 2832 | 267 |
//!
//! The reverse kernel costs 1.1× to 1.6× the reverse `γ`-rot kernel (the
//! difference is the `p × p` factorisation), 1.2× to 8× less than the
//! log-tangent kernel and 1.9× to 20× less than forward mode. A second run
//! agreed with these numbers to within 5%.
//!
//! The Hessian, each with the value and the gradient (`bench::hessian_cost`,
//! the same protocol; the differences on 2 000 observations):
//!
//! | `d` | `N` | exact | differences | reverse gradient | log-tangent exact |
//! |---|---|---|---|---|---|
//! | 3  | 5  | 100  | 522   | 56  | 123   |
//! | 5  | 14 | 302  | 3137  | 117 | 672   |
//! | 8  | 35 | 1127 | 15636 | 230 | 3239  |
//! | 10 | 54 | 2280 | 36113 | 338 | 14950 |
//!
//! The exact Hessian costs 1.8 to 6.7 gradients, 5× (`d = 3`) to 16× less
//! than the `2N` gradients of the differences, and 1.2× to 6.6× less than
//! the log-tangent kernel's exact Hessian (whose `d = 10` runs on heap
//! buffers).
//!
//! No allocation happens for `d ≤ 10` (const-generic stack buffers); larger
//! `d` use heap buffers.

use crate::dims;
use crate::dual::{Dual, seed_array};
use crate::frame_hess::{self, FrameState, Outer};
use crate::gamma_smooth::{project, sorted_tangent_eigen};
use crate::loglik::{cdf_fast, log_m_p, log_m_p_derivs};
use crate::logtangent::LtFrame;
use crate::scalar::Scalar;
use crate::stats::normal::{LN_2PI, norm_pdf};
use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

/// The spectral parameterization: a reference mean direction and an
/// optional rotation of the tangent basis.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SpFrame {
    lt: LtFrame,
    /// `R_0`, `(d−1) × (d−1)` row-major (`None`: identity).
    #[serde(default)]
    r0: Option<Vec<f64>>,
}

impl SpFrame {
    /// Frame with reference mean direction `m` (normalised) and the
    /// identity basis.
    ///
    /// # Panics
    /// If `m.len() < 3` or `m` is zero.
    pub fn new(m: &[f64]) -> Self {
        assert!(
            m.len() >= 3,
            "the spectral shape needs d >= 3, got d = {}",
            m.len()
        );
        Self {
            lt: LtFrame::new(m),
            r0: None,
        }
    }

    /// Turn the tangent basis by the rotation `r0` (`(d−1) × (d−1)`,
    /// orthogonal): the axes `R = I` refers to are then the columns of
    /// `E R_0` transported to `μ̂`.
    ///
    /// # Panics
    /// If `r0` is not `(d−1) × (d−1)`.
    pub fn with_basis(mut self, r0: &DMatrix<f64>) -> Self {
        let p = self.dim() - 1;
        assert!(
            r0.nrows() == p && r0.ncols() == p,
            "the basis rotation must be {p} × {p}"
        );
        self.r0 = Some(
            (0..p)
                .flat_map(|r| (0..p).map(move |c| (r, c)))
                .map(|(r, c)| r0[(r, c)])
                .collect(),
        );
        self
    }

    /// The dimension `d`.
    pub fn dim(&self) -> usize {
        self.lt.dim()
    }

    /// The tangent frame (reference mean direction `m` and basis `E`).
    pub fn lt_frame(&self) -> &LtFrame {
        &self.lt
    }

    /// The basis rotation `R_0` (`None`: identity).
    pub fn basis(&self) -> Option<DMatrix<f64>> {
        let p = self.dim() - 1;
        self.r0.as_ref().map(|r| DMatrix::from_row_slice(p, p, r))
    }

    /// `Ξ(μ̂) R_0`, `d × (d−1)`.
    pub fn tangent_frame(&self, mu: &[f64]) -> DMatrix<f64> {
        let d = self.dim();
        assert_eq!(mu.len(), d, "mu has length {}", mu.len());
        let mut xi = DMatrix::zeros(d, d - 1);
        let (mut z, mut u) = (vec![0.0; d], vec![0.0; d]);
        let mut y = vec![0.0; d];
        for i in 0..d {
            y.iter_mut().for_each(|v| *v = 0.0);
            y[i] = 1.0;
            project(&self.lt, mu, &y, &mut z, &mut u);
            turn(self, &mut z, &mut u);
            for j in 0..d - 1 {
                xi[(i, j)] = z[j];
            }
        }
        xi
    }

    fn check_gamma(&self, gamma: &[f64]) -> usize {
        let d = self.dim();
        assert_eq!(
            gamma.len(),
            dims::gamma_len(d),
            "gamma has length {} but d = {d} needs {}",
            gamma.len(),
            dims::gamma_len(d)
        );
        d - 1
    }

    /// Log-eigenvalues `ℓ = H h` of the tangent covariance along the axes
    /// (axis `k` has `ℓ_k`), summing to 0.
    pub fn log_eigenvalues(&self, gamma: &[f64]) -> Vec<f64> {
        let p = self.check_gamma(gamma);
        let mut ell = vec![0.0; p];
        log_eigen(&gamma[..p - 1], p, &mut ell);
        ell
    }

    /// Eigenvalues `e^{ℓ_1}, …, e^{ℓ_{d−1}}` of the tangent covariance
    /// (product 1), in the order of [`SpFrame::log_eigenvalues`].
    pub fn eigenvalues(&self, gamma: &[f64]) -> Vec<f64> {
        self.log_eigenvalues(gamma)
            .iter()
            .map(|l| l.exp())
            .collect()
    }

    /// The eigenvector rotation `R = (I − A/2)⁻¹ (I + A/2)`, `p × p`; column
    /// `k` is the axis of eigenvalue `e^{ℓ_k}`.
    pub fn rotation(&self, gamma: &[f64]) -> DMatrix<f64> {
        let p = self.check_gamma(gamma);
        let a = &gamma[p - 1..];
        let mut s = DMatrix::identity(p, p);
        let mut t = DMatrix::identity(p, p);
        let mut k = 0;
        for r in 0..p {
            for c in r + 1..p {
                let half = 0.5 * a[k];
                t[(r, c)] = half;
                t[(c, r)] = -half;
                s[(r, c)] = -half;
                s[(c, r)] = half;
                k += 1;
            }
        }
        s.lu()
            .solve(&t)
            .expect("I − A/2 has singular values >= 1 and is nonsingular")
    }

    /// The tangent covariance `W = R diag(e^{ℓ}) Rᵀ` (`(d−1) × (d−1)`,
    /// determinant 1) in the frame `Ξ(μ̂) R_0`.
    pub fn tangent_covariance(&self, gamma: &[f64]) -> DMatrix<f64> {
        let r = self.rotation(gamma);
        let lam = DVector::from_vec(self.eigenvalues(gamma));
        &r * DMatrix::from_diagonal(&lam) * r.transpose()
    }

    /// The covariance `V = Ξ W Ξᵀ + μ̂μ̂ᵀ` of `ESAG(mu, γ)` in this
    /// parameterization (`μ̂ = m` at `mu = 0`).
    pub fn covariance(&self, mu: &[f64], gamma: &[f64]) -> DMatrix<f64> {
        let d = self.dim();
        assert_eq!(mu.len(), d, "mu has length {}", mu.len());
        let xi = self.tangent_frame(mu);
        let w = self.tangent_covariance(gamma);
        let nm = mu.iter().map(|x| x * x).sum::<f64>().sqrt();
        let mh: Vec<f64> = if nm > 0.0 {
            mu.iter().map(|x| x / nm).collect()
        } else {
            self.lt.reference().to_vec()
        };
        let mut v = &xi * w * xi.transpose();
        for r in 0..d {
            for c in 0..d {
                v[(r, c)] += mh[r] * mh[c];
            }
        }
        v
    }

    /// Coordinates of `(mu, v)` in this parameterization (`v` scaled to
    /// determinant 1 on the tangent space): the canonical member of the
    /// equivalence class, with the eigenvalues ascending along the axes and
    /// `R` the proper rotation of largest trace among the sign choices of
    /// the eigenvectors. The `a` part is `NaN` when that `R` turns some
    /// plane by exactly `π` (see the [module documentation](self)).
    pub fn coords_from_covariance(&self, mu: &[f64], v: &DMatrix<f64>) -> Vec<f64> {
        let xi = self.tangent_frame(mu);
        let (lam, cur) = sorted_tangent_eigen(&xi, v);
        self.coords_from_eigen(&lam, cur)
    }

    /// [`SpFrame::coords_from_covariance`] from the tangent eigenvalues
    /// `lam` (only their ratios matter) and eigenvectors `cur` (columns, in
    /// the frame's coordinates), in the given order: axis `k` gets
    /// `lam[k]`. The signs of the columns are chosen so that `R` is the
    /// proper rotation of largest trace (nearest to `I`): column `k` is
    /// flipped when `cur[(k, k)] < 0`, and if the determinant is then `−1`
    /// the column with the smallest `|cur[(k, k)]|` is flipped too. The `a`
    /// part is `NaN` when `R` has an eigenvalue `−1`.
    pub fn coords_from_eigen(&self, lam: &[f64], cur: DMatrix<f64>) -> Vec<f64> {
        self.coords_from_eigen_with(lam, cur, true)
    }

    /// [`SpFrame::coords_from_eigen`] keeping the signs of the given
    /// eigenvectors (the caller labels and signs them consistently, so that
    /// the coordinates of neighbouring `V`s vary continuously); only when
    /// the determinant is `−1` is the last column flipped. The `a` part is
    /// `NaN` when the resulting `R` has an eigenvalue `−1`, and large when
    /// it is near one.
    pub fn coords_from_eigen_signed(&self, lam: &[f64], cur: DMatrix<f64>) -> Vec<f64> {
        self.coords_from_eigen_with(lam, cur, false)
    }

    fn coords_from_eigen_with(
        &self,
        lam: &[f64],
        mut cur: DMatrix<f64>,
        canonical: bool,
    ) -> Vec<f64> {
        let d = self.dim();
        let p = d - 1;
        assert_eq!(lam.len(), p, "lam has length {} but d − 1 = {p}", lam.len());
        assert!(
            cur.nrows() == p && cur.ncols() == p,
            "the eigenvectors must be {p} × {p}"
        );
        let mut gamma = vec![0.0; dims::gamma_len(d)];
        let logs: Vec<f64> = lam.iter().map(|l| l.ln()).collect();
        helmert_encode(&logs, p, &mut gamma[..p - 1]);
        let flip = |cur: &mut DMatrix<f64>, k: usize| {
            for r in 0..p {
                cur[(r, k)] = -cur[(r, k)];
            }
        };
        if canonical {
            for k in 0..p {
                if cur[(k, k)] < 0.0 {
                    flip(&mut cur, k);
                }
            }
        }
        if cur.determinant() < 0.0 {
            let k = if canonical {
                (0..p)
                    .min_by(|&i, &j| cur[(i, i)].abs().total_cmp(&cur[(j, j)].abs()))
                    .unwrap_or(p - 1)
            } else {
                p - 1
            };
            flip(&mut cur, k);
        }
        // A = 2 (R − I)(R + I)⁻¹, antisymmetrised against rounding
        let eye = DMatrix::identity(p, p);
        let a = (&cur + &eye)
            .try_inverse()
            .map(|inv| (&cur - &eye) * inv * 2.0);
        let mut k = p - 1;
        for r in 0..p {
            for c in r + 1..p {
                gamma[k] = match &a {
                    Some(a) => 0.5 * (a[(r, c)] - a[(c, r)]),
                    None => f64::NAN,
                };
                k += 1;
            }
        }
        gamma
    }
}

// =========================================================================
// Shared pieces of the forward pass (generic, so that the reverse kernel
// with `S = f64` performs the operations of `loglik_sp::<f64>` bit for bit)
// =========================================================================

/// `z ← R_0ᵀ z` on the first `d − 1` entries (identity without a basis);
/// `tmp` (length `≥ d − 1`) is scratch.
#[inline(always)]
fn turn<S: Scalar>(fr: &SpFrame, z: &mut [S], tmp: &mut [S]) {
    if let Some(r0) = &fr.r0 {
        let p = fr.dim() - 1;
        for j in 0..p {
            let mut acc = S::zero();
            for i in 0..p {
                acc += z[i] * r0[i * p + j];
            }
            tmp[j] = acc;
        }
        z[..p].copy_from_slice(&tmp[..p]);
    }
}

/// `c_m = 1/√(m(m+1))`, the scale of Helmert column `m`.
#[inline(always)]
fn helmert_c(m: usize) -> f64 {
    1.0 / ((m * (m + 1)) as f64).sqrt()
}

/// `ℓ = H h` into `ell[..p]`: `ℓ_k = Σ_{m>k} c_m h_{m−1} − k c_k h_{k−1}`.
#[inline(always)]
fn log_eigen<S: Scalar>(h: &[S], p: usize, ell: &mut [S]) {
    let mut suffix = S::zero();
    for k in (1..p).rev() {
        let c = helmert_c(k);
        ell[k] = suffix - h[k - 1] * (k as f64 * c);
        suffix += h[k - 1] * c;
    }
    ell[0] = suffix;
}

/// `h = Hᵀ ℓ` into `h[..p−1]`: the inverse of [`log_eigen`] on sum-zero `ℓ`
/// (and the projection onto it otherwise), and its adjoint.
fn helmert_encode(ell: &[f64], p: usize, h: &mut [f64]) {
    let mut prefix = 0.0;
    for m in 1..p {
        prefix += ell[m - 1];
        h[m - 1] = helmert_c(m) * (prefix - m as f64 * ell[m]);
    }
}

/// `T = I + A/2`, row-major `p × p`, from the upper-triangle coordinates
/// `a` (row by row).
#[inline(always)]
fn build_t<S: Scalar>(a: &[S], p: usize, t: &mut [S]) {
    for x in t[..p * p].iter_mut() {
        *x = S::zero();
    }
    for i in 0..p {
        t[i * p + i] = S::one();
    }
    let mut k = 0;
    for r in 0..p {
        for c in r + 1..p {
            let half = a[k] * 0.5;
            t[r * p + c] = half;
            t[c * p + r] = -half;
            k += 1;
        }
    }
}

/// `out ← (I − A/2) x` on `p` entries.
#[inline(always)]
fn apply_s<S: Scalar>(a: &[S], p: usize, x: &[S], out: &mut [S]) {
    out[..p].copy_from_slice(&x[..p]);
    let mut k = 0;
    for r in 0..p {
        for c in r + 1..p {
            let half = a[k] * 0.5;
            out[r] -= half * x[c];
            out[c] += half * x[r];
            k += 1;
        }
    }
}

/// `out ← (I + A/2) x` on `p` entries.
#[inline(always)]
fn apply_t(a: &[f64], p: usize, x: &[f64], out: &mut [f64]) {
    out[..p].copy_from_slice(&x[..p]);
    let mut k = 0;
    for r in 0..p {
        for c in r + 1..p {
            let half = a[k] * 0.5;
            out[r] += half * x[c];
            out[c] -= half * x[r];
            k += 1;
        }
    }
}

/// In-place `LU` factorisation with partial pivoting of the row-major
/// `p × p` matrix `a` (pivots chosen on real parts): afterwards `a` holds
/// `L` (unit lower, strictly below the diagonal) and `U`, and `piv[k]` is
/// the row swapped into position `k` at step `k`.
#[inline(always)]
fn lu_factor<S: Scalar>(a: &mut [S], p: usize, piv: &mut [usize]) {
    for k in 0..p {
        let mut best = k;
        let mut bv = a[k * p + k].re().abs();
        for i in k + 1..p {
            let v = a[i * p + k].re().abs();
            if v > bv {
                bv = v;
                best = i;
            }
        }
        piv[k] = best;
        if best != k {
            for j in 0..p {
                a.swap(k * p + j, best * p + j);
            }
        }
        let inv = S::one() / a[k * p + k];
        for i in k + 1..p {
            let l = a[i * p + k] * inv;
            a[i * p + k] = l;
            for j in k + 1..p {
                let akj = a[k * p + j];
                a[i * p + j] -= l * akj;
            }
        }
    }
}

/// `x ← T⁻¹ x` from the factorisation of [`lu_factor`].
#[inline(always)]
fn lu_solve<S: Scalar>(lu: &[S], p: usize, piv: &[usize], x: &mut [S]) {
    for k in 0..p {
        let j = piv[k];
        if j != k {
            x.swap(k, j);
        }
    }
    for i in 1..p {
        let mut acc = x[i];
        for j in 0..i {
            acc -= lu[i * p + j] * x[j];
        }
        x[i] = acc;
    }
    for i in (0..p).rev() {
        let mut acc = x[i];
        for j in i + 1..p {
            acc -= lu[i * p + j] * x[j];
        }
        x[i] = acc / lu[i * p + i];
    }
}

/// `x ← T⁻ᵀ x` from the factorisation of [`lu_factor`]
/// (`Tᵀ = Uᵀ Lᵀ P`).
#[inline(always)]
fn lu_solve_t(lu: &[f64], p: usize, piv: &[usize], x: &mut [f64]) {
    for i in 0..p {
        let mut acc = x[i];
        for j in 0..i {
            acc -= lu[j * p + i] * x[j];
        }
        x[i] = acc / lu[i * p + i];
    }
    for i in (0..p).rev() {
        let mut acc = x[i];
        for j in i + 1..p {
            acc -= lu[j * p + i] * x[j];
        }
        x[i] = acc;
    }
    for k in (0..p).rev() {
        let j = piv[k];
        if j != k {
            x.swap(k, j);
        }
    }
}

// =========================================================================
// Generic kernel
// =========================================================================

/// Scratch of the generic kernel.
struct Buf<'a, S> {
    /// The projections `z`, then `w = Rᵀ z`.
    z: &'a mut [S],
    /// Scratch of `project` and of the basis turn, then `s̃ = T⁻¹ z`.
    u: &'a mut [S],
    /// The log-eigenvalues.
    ell: &'a mut [S],
    /// `T = I + A/2`, then its `LU` factors.
    mat: &'a mut [S],
    piv: &'a mut [usize],
}

#[inline(always)]
fn loglik_core<S: Scalar>(fr: &SpFrame, mu: &[S], gamma: &[S], y: &[f64], b: Buf<'_, S>) -> S {
    let d = mu.len();
    let p = d - 1;
    let Buf {
        z,
        u,
        ell,
        mat,
        piv,
    } = b;
    let (t, ymu, mm) = project(&fr.lt, mu, y, z, u);
    turn(fr, z, u);
    let (h, a) = gamma.split_at(p - 1);
    build_t(a, p, mat);
    lu_factor(mat, p, piv);
    u[..p].copy_from_slice(&z[..p]);
    lu_solve(mat, p, piv, u);
    apply_s(a, p, u, z);
    log_eigen(h, p, ell);
    let mut q = t * t;
    for k in 0..p {
        q += z[k] * z[k] * (-ell[k]).exp();
    }
    let alpha = ymu / q.sqrt();
    S::from_f64(-0.5 * p as f64 * LN_2PI) - q.ln() * (0.5 * d as f64)
        + (alpha * alpha - mm) * 0.5
        + log_m_p(alpha, p)
}

/// Log-density at the unit vector `y` for stacked `eta = (mu, γ)`.
///
/// # Panics
/// If `d = y.len() < 3`, `eta.len()` is not [`dims::eta_len`]`(d)`, or the
/// frame has another dimension.
pub fn loglik_sp<S: Scalar>(fr: &SpFrame, eta: &[S], y: &[f64]) -> S {
    let d = y.len();
    assert!(d >= 3, "the ESAG density needs d >= 3, got d = {d}");
    assert_eq!(fr.dim(), d, "frame dimension {} but d = {d}", fr.dim());
    assert_eq!(eta.len(), dims::eta_len(d), "eta has length {}", eta.len());
    let (mu, gamma) = eta.split_at(d);
    match d {
        3 => fixed::<S, 3, 4>(fr, mu, gamma, y),
        4 => fixed::<S, 4, 9>(fr, mu, gamma, y),
        5 => fixed::<S, 5, 16>(fr, mu, gamma, y),
        6 => fixed::<S, 6, 25>(fr, mu, gamma, y),
        7 => fixed::<S, 7, 36>(fr, mu, gamma, y),
        8 => fixed::<S, 8, 49>(fr, mu, gamma, y),
        9 => fixed::<S, 9, 64>(fr, mu, gamma, y),
        10 => fixed::<S, 10, 81>(fr, mu, gamma, y),
        _ => {
            let p = d - 1;
            let (mut z, mut u, mut ell) =
                (vec![S::zero(); d], vec![S::zero(); d], vec![S::zero(); p]);
            let mut mat = vec![S::zero(); p * p];
            let mut piv = vec![0usize; p];
            let b = Buf {
                z: &mut z,
                u: &mut u,
                ell: &mut ell,
                mat: &mut mat,
                piv: &mut piv,
            };
            loglik_core(fr, mu, gamma, y, b)
        }
    }
}

#[inline(always)]
fn fixed<S: Scalar, const D: usize, const PP: usize>(
    fr: &SpFrame,
    mu: &[S],
    gamma: &[S],
    y: &[f64],
) -> S {
    const { assert!(D >= 3 && PP == (D - 1) * (D - 1)) };
    let (mut z, mut u, mut ell) = ([S::zero(); D], [S::zero(); D], [S::zero(); D]);
    let mut mat = [S::zero(); PP];
    let mut piv = [0usize; D];
    let b = Buf {
        z: &mut z,
        u: &mut u,
        ell: &mut ell,
        mat: &mut mat,
        piv: &mut piv,
    };
    loglik_core(fr, mu, gamma, y, b)
}

fn grad_dual<const N: usize>(fr: &SpFrame, eta: &[f64], y: &[f64], grad: &mut [f64]) -> f64 {
    let x: [Dual<N>; N] = seed_array(eta);
    let v = loglik_sp(fr, &x, y);
    grad.copy_from_slice(&v.eps);
    v.re
}

/// Log-density and its exact gradient in `eta` by forward-mode dual
/// numbers: the reference [`loglik_sp_grad`] is tested against, about
/// `eta.len()` times slower than it.
///
/// # Panics
/// If `d` is outside `3..=10`, or as [`loglik_sp`].
pub fn loglik_sp_grad_fwd(fr: &SpFrame, eta: &[f64], y: &[f64], grad: &mut [f64]) -> f64 {
    assert_eq!(grad.len(), eta.len(), "grad has length {}", grad.len());
    match eta.len() {
        5 => grad_dual::<5>(fr, eta, y, grad),
        9 => grad_dual::<9>(fr, eta, y, grad),
        14 => grad_dual::<14>(fr, eta, y, grad),
        20 => grad_dual::<20>(fr, eta, y, grad),
        27 => grad_dual::<27>(fr, eta, y, grad),
        35 => grad_dual::<35>(fr, eta, y, grad),
        44 => grad_dual::<44>(fr, eta, y, grad),
        54 => grad_dual::<54>(fr, eta, y, grad),
        n => panic!(
            "forward-mode spectral gradients are implemented for d = 3..=10 (eta length {n})"
        ),
    }
}

// =========================================================================
// Reverse-mode kernel
// =========================================================================

/// Log-density at the unit vector `y` for stacked `eta = (μ, γ)` in the
/// spectral parameterization `fr`, and its exact gradient in `eta`, written
/// into `grad`, by one forward and one reverse pass (`O(p³)` per
/// observation, no dual numbers).
///
/// The value is bit-identical to [`loglik_sp`]`::<f64>`; the gradient
/// agrees with [`loglik_sp_grad_fwd`] to rounding, including at `μ = 0`,
/// `μ̂ = −m`, isotropy and tied eigenvalues, and is exact for every `d ≥ 3`.
///
/// # Panics
/// If `d = y.len() < 3`, the frame has another dimension, `eta.len()` is
/// not [`dims::eta_len`]`(d)`, or `grad.len() != eta.len()`.
pub fn loglik_sp_grad(fr: &SpFrame, eta: &[f64], y: &[f64], grad: &mut [f64]) -> f64 {
    let d = y.len();
    assert!(d >= 3, "the ESAG density needs d >= 3, got d = {d}");
    assert_eq!(fr.dim(), d, "frame dimension {} but d = {d}", fr.dim());
    assert_eq!(eta.len(), dims::eta_len(d), "eta has length {}", eta.len());
    assert_eq!(grad.len(), eta.len(), "grad has length {}", grad.len());
    match d {
        3 => fixed_rev::<3, 4>(fr, eta, y, grad),
        4 => fixed_rev::<4, 9>(fr, eta, y, grad),
        5 => fixed_rev::<5, 16>(fr, eta, y, grad),
        6 => fixed_rev::<6, 25>(fr, eta, y, grad),
        7 => fixed_rev::<7, 36>(fr, eta, y, grad),
        8 => fixed_rev::<8, 49>(fr, eta, y, grad),
        9 => fixed_rev::<9, 64>(fr, eta, y, grad),
        10 => fixed_rev::<10, 81>(fr, eta, y, grad),
        _ => {
            let (mu, gamma) = eta.split_at(d);
            let p = d - 1;
            let mut v = vec![0.0; NV * d];
            let mut mat = vec![0.0; p * p];
            let mut piv = vec![0usize; p];
            core::<false>(
                fr,
                mu,
                gamma,
                y,
                grad,
                Work::new(&mut v, &mut mat, &mut piv),
                HessWork::empty(),
            )
        }
    }
}

/// Log-density, exact gradient and the Hessian in `eta` by central
/// differences of [`loglik_sp_grad`] (step `1e−5·max(1, |eta_i|)`,
/// symmetrised).
///
/// # Panics
/// As [`loglik_sp_grad`], or if `hess.len() != eta.len()²`.
pub fn loglik_sp_hess_fd(
    fr: &SpFrame,
    eta: &[f64],
    y: &[f64],
    grad: &mut [f64],
    hess: &mut [f64],
) -> f64 {
    let n = eta.len();
    assert_eq!(hess.len(), n * n, "hess must have length eta.len()²");
    let f = loglik_sp_grad(fr, eta, y, grad);
    let mut e = eta.to_vec();
    let (mut gp, mut gm) = (vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        let h = 1e-5 * eta[i].abs().max(1.0);
        e[i] = eta[i] + h;
        loglik_sp_grad(fr, &e, y, &mut gp);
        e[i] = eta[i] - h;
        loglik_sp_grad(fr, &e, y, &mut gm);
        e[i] = eta[i];
        for j in 0..n {
            hess[i * n + j] = (gp[j] - gm[j]) / (2.0 * h);
        }
    }
    for i in 0..n {
        for j in 0..i {
            let s = 0.5 * (hess[i * n + j] + hess[j * n + i]);
            hess[i * n + j] = s;
            hess[j * n + i] = s;
        }
    }
    f
}

/// Log-density, its exact gradient (into `grad`, as [`loglik_sp_grad`])
/// and its exact Hessian in `eta` (into `hess`, row-major `n × n`,
/// `n = eta.len()`, exactly symmetric), from one forward pass, one reverse
/// sweep and the closed-form second derivatives of the Cayley transform
/// (see the [module documentation](self)).
///
/// The value and gradient are bit-identical to [`loglik_sp_grad`]. At
/// `μ = 0` and `μ̂ = −m`, where the frame is held constant, the Hessian is
/// that of the gradient's convention: `q` does not depend on `μ` (`μ = 0`)
/// or depends on it through `t = μ̂·y` only (`μ̂ = −m`).
///
/// # Panics
/// As [`loglik_sp_grad`], or if `hess.len() != eta.len()²`.
pub fn loglik_sp_hess(
    fr: &SpFrame,
    eta: &[f64],
    y: &[f64],
    grad: &mut [f64],
    hess: &mut [f64],
) -> f64 {
    let d = y.len();
    assert!(d >= 3, "the ESAG density needs d >= 3, got d = {d}");
    assert_eq!(fr.dim(), d, "frame dimension {} but d = {d}", fr.dim());
    assert_eq!(eta.len(), dims::eta_len(d), "eta has length {}", eta.len());
    assert_eq!(grad.len(), eta.len(), "grad has length {}", grad.len());
    let n = eta.len();
    assert_eq!(hess.len(), n * n, "hess must have length eta.len()²");
    // const parameters: D, P = D − 1, NA = P(P−1)/2 Cayley coordinates,
    // NG = gamma_len(D), NE = eta_len(D)
    match d {
        3 => fixed_hess::<3, 2, 1, 2, 5>(fr, eta, y, grad, hess),
        4 => fixed_hess::<4, 3, 3, 5, 9>(fr, eta, y, grad, hess),
        5 => fixed_hess::<5, 4, 6, 9, 14>(fr, eta, y, grad, hess),
        6 => fixed_hess::<6, 5, 10, 14, 20>(fr, eta, y, grad, hess),
        7 => fixed_hess::<7, 6, 15, 20, 27>(fr, eta, y, grad, hess),
        8 => fixed_hess::<8, 7, 21, 27, 35>(fr, eta, y, grad, hess),
        9 => fixed_hess::<9, 8, 28, 35, 44>(fr, eta, y, grad, hess),
        10 => fixed_hess::<10, 9, 36, 44, 54>(fr, eta, y, grad, hess),
        _ => {
            let (mu, gamma) = eta.split_at(d);
            let p = d - 1;
            let na = p * (p - 1) / 2;
            let mut v = vec![0.0; NV * d];
            let mut mat = vec![0.0; p * p];
            let mut piv = vec![0usize; p];
            let mut hv = vec![0.0; n + 4 * p + 3 * d + 3 * p * p + 2 * na * p + 4 * p * d + d * d];
            let (dq, rest) = hv.split_at_mut(n);
            let (wq, rest) = rest.split_at_mut(p);
            let (cq, rest) = rest.split_at_mut(p);
            let (gt, rest) = rest.split_at_mut(p);
            let (wt, rest) = rest.split_at_mut(p);
            let (ubar, rest) = rest.split_at_mut(d);
            let (xv, rest) = rest.split_at_mut(d);
            let (xw, rest) = rest.split_at_mut(d);
            let (mm, rest) = rest.split_at_mut(p * p);
            let (mt, rest) = rest.split_at_mut(p * p);
            let (nm, rest) = rest.split_at_mut(p * p);
            let (dm, rest) = rest.split_at_mut(na * p);
            let (om, rest) = rest.split_at_mut(na * p);
            let (jt, rest) = rest.split_at_mut(p * d);
            let (yy, rest) = rest.split_at_mut(p * d);
            let (uu, rest) = rest.split_at_mut(p * d);
            let (zz, am) = rest.split_at_mut(p * d);
            let hw = HessWork {
                hess,
                dq,
                wq,
                cq,
                gt,
                wt,
                ubar,
                xv,
                xw,
                mm,
                mt,
                nm,
                dm,
                om,
                jt,
                yy,
                uu,
                zz,
                am,
            };
            core::<true>(
                fr,
                mu,
                gamma,
                y,
                grad,
                Work::new(&mut v, &mut mat, &mut piv),
                hw,
            )
        }
    }
}

/// Number of length-`d` scratch vectors in [`Work`].
const NV: usize = 8;

/// Scratch for one evaluation of the reverse kernel: [`NV`] vectors of
/// length `≥ d`, a `p × p` matrix and the pivots.
struct Work<'a> {
    /// The projections, then `s̄̃`, then `z̄`.
    z: &'a mut [f64],
    /// `u = y − κ v`.
    u: &'a mut [f64],
    /// `v = μ̂ + m` (or `E_0` at the antipode).
    v: &'a mut [f64],
    /// `μ̂` (or `m` at `μ = 0`).
    mh: &'a mut [f64],
    /// Scratch for the basis turns, then `ū`, then `μ̂̄`.
    ub: &'a mut [f64],
    /// `s̃ = T⁻¹ z`.
    st: &'a mut [f64],
    /// `w = S s̃`, then `w̄`.
    w: &'a mut [f64],
    /// `ℓ`, then `e^{−ℓ}`, then `w̄ + z̄`.
    ex: &'a mut [f64],
    /// `T = I + A/2`, then its `LU` factors.
    mat: &'a mut [f64],
    piv: &'a mut [usize],
}

impl<'a> Work<'a> {
    /// Splits a heap buffer of length `NV·d` (fallback for `d > 10`).
    fn new(v: &'a mut [f64], mat: &'a mut [f64], piv: &'a mut [usize]) -> Self {
        let d = v.len() / NV;
        let mut it = v.chunks_exact_mut(d);
        let mut next = || it.next().expect("buffer of length NV·d");
        Work {
            z: next(),
            u: next(),
            v: next(),
            mh: next(),
            ub: next(),
            st: next(),
            w: next(),
            ex: next(),
            mat,
            piv,
        }
    }
}

/// Stack buffers for dimension `D` (`PP = (D−1)²`). Out of line so that
/// each dimension gets its own compact function with compile-time trip
/// counts.
#[inline(never)]
fn fixed_rev<const D: usize, const PP: usize>(
    fr: &SpFrame,
    eta: &[f64],
    y: &[f64],
    grad: &mut [f64],
) -> f64 {
    const { assert!(D >= 3 && PP == (D - 1) * (D - 1)) };
    let (mu, gamma) = eta.split_at(D);
    let mut bufs = [[0.0; D]; NV];
    let mut mat = [0.0; PP];
    let mut piv = [0usize; D];
    let [z, u, v, mh, ub, st, w, ex] = &mut bufs;
    let work = Work {
        z,
        u,
        v,
        mh,
        ub,
        st,
        w,
        ex,
        mat: &mut mat,
        piv: &mut piv,
    };
    core::<false>(
        fr,
        &mu[..D],
        &gamma[..dims::gamma_len(D)],
        &y[..D],
        &mut grad[..dims::eta_len(D)],
        work,
        HessWork::empty(),
    )
}

/// Stack buffers for the Hessian at dimension `D` (`P = D − 1`,
/// `NA = P(P−1)/2`, `NG = gamma_len(D)`, `NE = eta_len(D)`).
#[inline(never)]
fn fixed_hess<const D: usize, const P: usize, const NA: usize, const NG: usize, const NE: usize>(
    fr: &SpFrame,
    eta: &[f64],
    y: &[f64],
    grad: &mut [f64],
    hess: &mut [f64],
) -> f64 {
    const {
        assert!(
            D >= 3
                && P + 1 == D
                && NA == P * (P - 1) / 2
                && NG == (D - 2) * (D + 1) / 2
                && NE == D + NG
        )
    };
    let (mu, gamma) = eta.split_at(D);
    let mut bufs = [[0.0; D]; NV];
    let mut mat = [[0.0; P]; P];
    let mut piv = [0usize; D];
    let [z, u, v, mh, ub, st, w, ex] = &mut bufs;
    let work = Work {
        z,
        u,
        v,
        mh,
        ub,
        st,
        w,
        ex,
        mat: mat.as_flattened_mut(),
        piv: &mut piv,
    };
    let mut dq = [0.0; NE];
    let mut pv = [[0.0; P]; 4];
    let mut dv = [[0.0; D]; 3];
    let mut pm = [[[0.0; P]; P]; 3];
    let mut pa = [[[0.0; P]; NA]; 2];
    let mut pd = [[[0.0; D]; P]; 4];
    let mut am = [[0.0; D]; D];
    let [wq, cq, gt, wt] = &mut pv;
    let [ubar, xv, xw] = &mut dv;
    let [mm, mt, nm] = &mut pm;
    let [dm, om] = &mut pa;
    let [jt, yy, uu, zz] = &mut pd;
    let hw = HessWork {
        hess: &mut hess[..NE * NE],
        dq: &mut dq,
        wq,
        cq,
        gt,
        wt,
        ubar,
        xv,
        xw,
        mm: mm.as_flattened_mut(),
        mt: mt.as_flattened_mut(),
        nm: nm.as_flattened_mut(),
        dm: dm.as_flattened_mut(),
        om: om.as_flattened_mut(),
        jt: jt.as_flattened_mut(),
        yy: yy.as_flattened_mut(),
        uu: uu.as_flattened_mut(),
        zz: zz.as_flattened_mut(),
        am: am.as_flattened_mut(),
    };
    core::<true>(
        fr,
        &mu[..D],
        &gamma[..NG],
        &y[..D],
        &mut grad[..NE],
        work,
        hw,
    )
}

/// Forward pass (recording) and reverse sweep, and for `HESS` the Hessian
/// into `hw`. See the module documentation.
#[inline(always)]
fn core<const HESS: bool>(
    fr: &SpFrame,
    mu: &[f64],
    gamma: &[f64],
    y: &[f64],
    grad: &mut [f64],
    w: Work<'_>,
    hw: HessWork<'_>,
) -> f64 {
    let d = mu.len();
    let p = d - 1;
    let lt = fr.lt_frame();
    let m = lt.reference();
    let e = lt.tangent_basis();
    let r0 = fr.r0.as_deref();
    let Work {
        z,
        u,
        v,
        mh,
        ub,
        st,
        w: wv,
        ex,
        mat,
        piv,
    } = w;
    let (z, u, v, mh, ub) = (
        &mut z[..d],
        &mut u[..d],
        &mut v[..d],
        &mut mh[..d],
        &mut ub[..d],
    );
    let (st, wv, ex) = (&mut st[..p], &mut wv[..p], &mut ex[..p]);
    let (mat, piv) = (&mut mat[..p * p], &mut piv[..p]);
    let (h, a) = gamma.split_at(p - 1);
    let (grad_mu, grad_gamma) = grad.split_at_mut(d);
    let (grad_h, grad_a) = grad_gamma.split_at_mut(p - 1);

    // =====================================================================
    // Forward pass: the operations of `loglik_core` in their order, so the
    // value is that of `loglik_sp::<f64>`.
    // =====================================================================

    // ---- frame (`gamma_smooth::project`): z = Eᵀ(y − κ v), v = μ̂ + m
    let mut mm = 0.0;
    let mut ymu = 0.0;
    for k in 0..d {
        mm += mu[k] * mu[k];
        ymu += mu[k] * y[k];
    }
    let zero_mu = mm == 0.0;
    let inv = if zero_mu { 1.0 } else { 1.0 / mm.sqrt() };
    for k in 0..d {
        mh[k] = if zero_mu { m[k] } else { mu[k] * inv };
    }
    let t: f64 = if zero_mu {
        (0..d).map(|k| m[k] * y[k]).sum()
    } else {
        ymu * inv
    };
    let mut vv = 0.0;
    let mut vy = 0.0;
    for k in 0..d {
        let vk = mh[k] + m[k];
        v[k] = vk;
        vv += vk * vk;
        vy += vk * y[k];
    }
    let antipode = vv == 0.0;
    let kappa = if antipode {
        v.copy_from_slice(&e[..d]);
        vv = 1.0;
        2.0 * (0..d).map(|k| e[k] * y[k]).sum::<f64>()
    } else {
        vy * 2.0 / vv
    };
    for k in 0..d {
        u[k] = y[k] - kappa * v[k];
    }
    for j in 0..p {
        let col = &e[j * d..(j + 1) * d];
        let mut acc = 0.0;
        for k in 0..d {
            acc += u[k] * col[k];
        }
        z[j] = acc;
    }

    // ---- basis turn: z ← R_0ᵀ z
    if let Some(r0) = r0 {
        for j in 0..p {
            let mut acc = 0.0;
            for i in 0..p {
                acc += z[i] * r0[i * p + j];
            }
            ub[j] = acc;
        }
        z[..p].copy_from_slice(&ub[..p]);
    }

    // ---- Cayley: T = I + A/2 factored, s̃ = T⁻¹ z, w = S s̃ = Rᵀ z
    build_t(a, p, mat);
    lu_factor(mat, p, piv);
    st.copy_from_slice(&z[..p]);
    lu_solve(mat, p, piv, st);
    apply_s(a, p, st, wv);

    // ---- log-eigenvalues and q = t² + Σ w_k² e^{−ℓ_k}
    log_eigen(h, p, ex);
    let mut q = t * t;
    for k in 0..p {
        let ek = (-ex[k]).exp();
        ex[k] = ek;
        q += wv[k] * wv[k] * ek;
    }
    if HESS {
        hw.wq[..p].copy_from_slice(wv);
        hw.cq[..p].copy_from_slice(ex);
    }

    // ---- log-density (the reverse sweep below does not depend on it)
    let sq = q.sqrt();
    let alpha = ymu / sq;
    let (inv_q, inv_sq) = (1.0 / q, 1.0 / sq);
    // log_m_p_derivs returns the same first two outputs, bit for bit
    let (lmp, dlmp, d2lmp) = if HESS {
        log_m_p_derivs(alpha, p)
    } else {
        let (l, d1) = log_m_p_d1(alpha, p);
        (l, d1, 0.0)
    };
    let ll =
        -0.5 * p as f64 * LN_2PI - q.ln() * (0.5 * d as f64) + (alpha * alpha - mm) * 0.5 + lmp;

    // =====================================================================
    // Reverse sweep for ∇q (seed q̄ = 1).
    // =====================================================================

    // ---- ℓ̄_k = −w_k² e_k, pulled through ℓ = H h; w̄_k = 2 w_k e_k
    let mut prefix = 0.0;
    for k in 0..p {
        let (wk, ek) = (wv[k], ex[k]);
        let lb = -(wk * wk * ek);
        if k >= 1 {
            grad_h[k - 1] = helmert_c(k) * (prefix - k as f64 * lb);
        }
        prefix += lb;
        wv[k] = 2.0 * wk * ek;
    }

    // ---- w = S s̃: s̄̃ = Sᵀ w̄ = T w̄ (into z); s̃ = T⁻¹ z: z̄ = T⁻ᵀ s̄̃
    apply_t(a, p, wv, z);
    lu_solve_t(mat, p, piv, z);

    // ---- Ā = −½ (w̄ + z̄) s̃ᵀ from both uses of A; ā_k = Ā_rc − Ā_cr
    for i in 0..p {
        ex[i] = wv[i] + z[i];
    }
    let mut k = 0;
    for r in 0..p {
        for c in r + 1..p {
            grad_a[k] = 0.5 * (st[r] * ex[c] - st[c] * ex[r]);
            k += 1;
        }
    }

    // ---- basis turn, reversed: z̄ ← R_0 z̄
    if let Some(r0) = r0 {
        for i in 0..p {
            let mut acc = 0.0;
            for j in 0..p {
                acc += r0[i * p + j] * z[j];
            }
            ub[i] = acc;
        }
        z[..p].copy_from_slice(&ub[..p]);
    }

    // ---- frame, reversed (∂q/∂μ into grad_mu): ū = E z̄,
    // μ̂̄ = v̄ + 2t y with v̄ = −κ ū − (2(ū·v)/(v·v)) u (zero for the constant
    // antipodal frame); then through μ̂ = μ/‖μ‖ (constant at μ = 0)
    for i in 0..d {
        ub[i] = (0..p).map(|j| e[j * d + i] * z[j]).sum();
    }
    if HESS {
        hw.ubar[..d].copy_from_slice(ub);
    }
    let ubv: f64 = (0..d).map(|i| ub[i] * v[i]).sum();
    for i in 0..d {
        let vb = if antipode {
            0.0
        } else {
            -kappa * ub[i] - 2.0 * ubv / vv * u[i]
        };
        ub[i] = vb + 2.0 * t * y[i];
    }
    if zero_mu {
        grad_mu.iter_mut().for_each(|g| *g = 0.0);
    } else {
        let s: f64 = (0..d).map(|i| mh[i] * ub[i]).sum();
        for i in 0..d {
            grad_mu[i] = (ub[i] - s * mh[i]) * inv;
        }
    }
    if HESS {
        hw.dq[..d].copy_from_slice(grad_mu);
        hw.dq[d..d + grad_gamma.len()].copy_from_slice(grad_gamma);
    }

    // ---- combine: ∇ll = q̄ ∇q + ȳmu ∇ymu − ½ ∇mm
    let ab = alpha + dlmp;
    let qb = -0.5 * (d as f64 + ab * alpha) * inv_q;
    let ymub = ab * inv_sq;
    for i in 0..d {
        grad_mu[i] = qb * grad_mu[i] + ymub * y[i] - mu[i];
    }
    for g in grad_gamma.iter_mut() {
        *g *= qb;
    }

    if HESS {
        // ex holds w̄ + z̄ = (I + R) w̄ = 2 M w̄
        for k in 0..p {
            hw.gt[k] = 0.5 * ex[k];
        }
        let fwd = SpFwd {
            fs: FrameState {
                e,
                y,
                mu_hat: mh,
                v,
                u,
                kappa,
                vv,
                inv_n: inv,
                mu_free: !zero_mu,
                frame_free: !zero_mu && !antipode,
            },
            r0,
            lu: mat,
            piv,
            st,
            g: ub,
            outer: Outer::new(d, q, sq, alpha, ab, d2lmp, qb),
        };
        sp_hessian(&fwd, hw);
    }
    ll
}

/// Buffers of [`loglik_sp_hess`] (`n = d + ng` parameters, `na = p(p−1)/2`
/// Cayley coordinates); empty for the gradient alone.
struct HessWork<'a> {
    /// The output, row-major `n × n`; first `∂²q` (upper triangle).
    hess: &'a mut [f64],
    /// `∇q` (`n`).
    dq: &'a mut [f64],
    /// `w = Rᵀz` (`p`).
    wq: &'a mut [f64],
    /// `c = e^{−ℓ}` (`p`).
    cq: &'a mut [f64],
    /// `g̃ = M w̄` (`p`).
    gt: &'a mut [f64],
    /// Scratch (`p`).
    wt: &'a mut [f64],
    /// `ū = E z̄` (`d`).
    ubar: &'a mut [f64],
    /// Scratch (`d` each).
    xv: &'a mut [f64],
    xw: &'a mut [f64],
    /// `M = (I − A/2)⁻¹`, its transpose and `N = M C Mᵀ` (`p²` each).
    mm: &'a mut [f64],
    mt: &'a mut [f64],
    nm: &'a mut [f64],
    /// `D_m = ∂w/∂a_m` and `Ω_m = 2Nσ_m − ½Mγ_m` by pair (`na·p` each).
    dm: &'a mut [f64],
    om: &'a mut [f64],
    /// `J = ∂z/∂μ̂`, `Y = MᵀJ`, `U = RᵀJ` and `Z = MCU` (`p × d` each).
    jt: &'a mut [f64],
    yy: &'a mut [f64],
    uu: &'a mut [f64],
    zz: &'a mut [f64],
    /// `A = ∂²q/∂μ̂²` (`d²`, upper triangle).
    am: &'a mut [f64],
}

impl HessWork<'static> {
    /// No Hessian.
    fn empty() -> Self {
        HessWork {
            hess: &mut [],
            dq: &mut [],
            wq: &mut [],
            cq: &mut [],
            gt: &mut [],
            wt: &mut [],
            ubar: &mut [],
            xv: &mut [],
            xw: &mut [],
            mm: &mut [],
            mt: &mut [],
            nm: &mut [],
            dm: &mut [],
            om: &mut [],
            jt: &mut [],
            yy: &mut [],
            uu: &mut [],
            zz: &mut [],
            am: &mut [],
        }
    }
}

/// What the Hessian needs from the forward pass and the reverse sweep,
/// besides the buffers of [`HessWork`].
struct SpFwd<'a> {
    fs: FrameState<'a>,
    r0: Option<&'a [f64]>,
    /// The `LU` factors of `T = I + A/2`.
    lu: &'a [f64],
    piv: &'a [usize],
    /// `s̃ = T⁻¹ z`.
    st: &'a [f64],
    /// `∂q/∂μ̂` with `μ̂` taken as a free vector.
    g: &'a [f64],
    outer: Outer,
}

/// The Hessian (see the module documentation): the shape block from the
/// closed-form derivatives of `w = Rᵀz` in the Cayley coordinates, the
/// cross and mean blocks through `J = ∂z/∂μ̂`, then the frame, `μ̂ = μ/‖μ‖`
/// and the outer layer of [`crate::frame_hess`].
#[inline(always)]
fn sp_hessian(f: &SpFwd<'_>, h: HessWork<'_>) {
    let fs = &f.fs;
    let d = fs.y.len();
    let p = d - 1;
    let na = p * (p - 1) / 2;
    let ng = p - 1 + na;
    let n = d + ng;
    let HessWork {
        hess,
        dq,
        wq,
        cq,
        gt,
        wt,
        ubar,
        xv,
        xw,
        mm,
        mt,
        nm,
        dm,
        om,
        jt,
        yy,
        uu,
        zz,
        am,
    } = h;
    let (hess, dq, ubar) = (&mut hess[..n * n], &dq[..n], &ubar[..d]);
    let (xv, xw) = (&mut xv[..d], &mut xw[..d]);
    let (w, c, gt, wt) = (&wq[..p], &cq[..p], &gt[..p], &mut wt[..p]);
    let (mm, mt, nm) = (&mut mm[..p * p], &mut mt[..p * p], &mut nm[..p * p]);
    let (dm, om) = (&mut dm[..na * p], &mut om[..na * p]);
    let (jt, yy, uu, zz) = (
        &mut jt[..p * d],
        &mut yy[..p * d],
        &mut uu[..p * d],
        &mut zz[..p * d],
    );
    let am = &mut am[..d * d];
    let st = &f.st[..p];
    let base = d + p - 1; // η index of a_0

    // ---- M = (I − A/2)⁻¹ = T⁻ᵀ (row i is T⁻¹ e_i), Mᵀ, N = M C Mᵀ
    for i in 0..p {
        let row = &mut mm[i * p..(i + 1) * p];
        row.fill(0.0);
        row[i] = 1.0;
        lu_solve(f.lu, p, f.piv, row);
    }
    for i in 0..p {
        for k in 0..p {
            mt[k * p + i] = mm[i * p + k];
        }
    }
    for i in 0..p {
        let mi = &mm[i * p..(i + 1) * p];
        for k in 0..p {
            wt[k] = mi[k] * c[k];
        }
        for j in i..p {
            let v = frame_hess::dot(wt, &mm[j * p..(j + 1) * p]);
            nm[i * p + j] = v;
            nm[j * p + i] = v;
        }
    }

    // ---- per pair m = (i, j): D_m = ∂w/∂a_m = −Mᵀσ_m = s̃_i M_j − s̃_j M_i
    // (rows of M) and Ω_m = 2Nσ_m − ½Mγ_m, with σ_m = s̃_j e_i − s̃_i e_j and
    // γ_m = g̃_j e_i − g̃_i e_j
    let mut m = 0;
    for i in 0..p {
        for j in i + 1..p {
            let (si, sj, gi, gj) = (st[i], st[j], gt[i], gt[j]);
            let (mi, mj) = (&mm[i * p..(i + 1) * p], &mm[j * p..(j + 1) * p]);
            let (ti, tj) = (&mt[i * p..(i + 1) * p], &mt[j * p..(j + 1) * p]);
            let (ni, nj) = (&nm[i * p..(i + 1) * p], &nm[j * p..(j + 1) * p]);
            let dr = &mut dm[m * p..(m + 1) * p];
            for k in 0..p {
                dr[k] = si * mj[k] - sj * mi[k];
            }
            let orow = &mut om[m * p..(m + 1) * p];
            for k in 0..p {
                orow[k] = 2.0 * (sj * ni[k] - si * nj[k]) - 0.5 * (gj * ti[k] - gi * tj[k]);
            }
            m += 1;
        }
    }

    // ---- shape block (upper triangle). Cayley against Cayley, l = (k, l′):
    // 2σ_lᵀNσ_m − ½(σ_lᵀMγ_m + σ_mᵀMγ_l)
    //   = s̃_l′ Ω_m[k] − s̃_k Ω_m[l′] + ½(g̃_l′ D_m[k] − g̃_k D_m[l′])
    let mut m = 0;
    for i in 0..p {
        for j in i + 1..p {
            let (orow, dr) = (&om[m * p..(m + 1) * p], &dm[m * p..(m + 1) * p]);
            let row = &mut hess[(base + m) * n..(base + m + 1) * n];
            let mut l = m;
            for k in i..p {
                let (ok, dk, sk, gk) = (orow[k], 0.5 * dr[k], st[k], 0.5 * gt[k]);
                for l2 in if k == i { j } else { k + 1 }..p {
                    row[base + l] = (st[l2] * ok - sk * orow[l2]) + (gt[l2] * dk - gk * dr[l2]);
                    l += 1;
                }
            }
            m += 1;
        }
    }
    // Helmert against Cayley: ∂²q/∂ℓ_k∂a_m = −2 c_k w_k D_m[k], then h = Hᵀℓ
    for k in 0..p {
        wt[k] = -2.0 * c[k] * w[k];
    }
    for m in 0..na {
        let dr = &dm[m * p..(m + 1) * p];
        let mut prefix = wt[0] * dr[0];
        for k in 1..p {
            let xk = wt[k] * dr[k];
            hess[(d + k - 1) * n + base + m] = helmert_c(k) * (prefix - k as f64 * xk);
            prefix += xk;
        }
    }
    // Helmert against Helmert: Hᵀ diag(c w²) H
    let mut prefix = 0.0;
    for m1 in 1..p {
        let xm = c[m1] * w[m1] * w[m1];
        prefix += c[m1 - 1] * w[m1 - 1] * w[m1 - 1];
        let cm = helmert_c(m1);
        let row = &mut hess[(d + m1 - 1) * n..(d + m1) * n];
        row[d + m1 - 1] = cm * cm * (prefix + (m1 * m1) as f64 * xm);
        let o = cm * (prefix - m1 as f64 * xm);
        for m2 in m1 + 1..p {
            row[d + m2 - 1] = helmert_c(m2) * o;
        }
    }

    // ---- mean rows
    for r in 0..d {
        hess[r * n..(r + 1) * n].fill(0.0);
    }
    if fs.mu_free {
        frame_hess::init_mean(fs.y, am);
        if fs.frame_free {
            // J = ∂z/∂μ̂ (turned), Y = MᵀJ, U = RᵀJ = 2Y − J = ∂w/∂μ̂, Z = MCU
            frame_hess::frame_jacobian(fs, f.r0, jt);
            for i in 0..p {
                let yi = &mut yy[i * d..(i + 1) * d];
                yi.fill(0.0);
                let ti = &mt[i * p..(i + 1) * p];
                for k in 0..p {
                    let mki = ti[k];
                    let jk = &jt[k * d..(k + 1) * d];
                    for l in 0..d {
                        yi[l] += mki * jk[l];
                    }
                }
            }
            for x in 0..p * d {
                uu[x] = 2.0 * yy[x] - jt[x];
            }
            for i in 0..p {
                let zi = &mut zz[i * d..(i + 1) * d];
                zi.fill(0.0);
                let mi = &mm[i * p..(i + 1) * p];
                for k in 0..p {
                    let cik = mi[k] * c[k];
                    let uk = &uu[k * d..(k + 1) * d];
                    for l in 0..d {
                        zi[l] += cik * uk[l];
                    }
                }
            }
            // A += Jᵀ(2W⁻¹)J = 2UᵀCU
            for k in 0..p {
                let s2 = 2.0 * c[k];
                let uk = &uu[k * d..(k + 1) * d];
                for r in 0..d {
                    let kr = s2 * uk[r];
                    for cc in r..d {
                        am[r * d + cc] += kr * uk[cc];
                    }
                }
            }
            // cross, Cayley: Jᵀ(Mγ_m + 2RCD_m) = (g̃_j Y_i − g̃_i Y_j) − 2(s̃_j Z_i − s̃_i Z_j),
            // then through μ̂ = μ/‖μ‖
            let mut m = 0;
            for i in 0..p {
                for j in i + 1..p {
                    let (gi, gj, si2, sj2) = (gt[i], gt[j], 2.0 * st[i], 2.0 * st[j]);
                    let (yi, yj) = (&yy[i * d..(i + 1) * d], &yy[j * d..(j + 1) * d]);
                    let (zi, zj) = (&zz[i * d..(i + 1) * d], &zz[j * d..(j + 1) * d]);
                    for l in 0..d {
                        xv[l] = (gj * yi[l] - gi * yj[l]) - (sj2 * zi[l] - si2 * zj[l]);
                    }
                    frame_hess::project_cross(fs, xv, hess, n, base + m);
                    m += 1;
                }
            }
            // cross, Helmert: Jᵀ ∂(∂q/∂z)/∂ℓ_k = −2 c_k w_k U_k, then h = Hᵀℓ
            // (xw holds the prefix sums over k)
            xw.fill(0.0);
            for k1 in 1..p {
                let (w0, u0) = (wt[k1 - 1], &uu[(k1 - 1) * d..k1 * d]);
                for l in 0..d {
                    xw[l] += w0 * u0[l];
                }
                let (wk, uk) = (wt[k1] * k1 as f64, &uu[k1 * d..(k1 + 1) * d]);
                let ck = helmert_c(k1);
                for l in 0..d {
                    xv[l] = ck * (xw[l] - wk * uk[l]);
                }
                frame_hess::project_cross(fs, xv, hess, n, d + k1 - 1);
            }
            frame_hess::add_frame_curvature(fs, ubar, am);
        }
        frame_hess::project_mean(fs, f.g, am, xv, hess, n);
    }
    frame_hess::assemble(&f.outer, dq, fs.y, hess, n);
}

/// Below this `alpha`, `log M_p` uses its continued fraction (the value of
/// `loglik::CF_BELOW`).
const CF_BELOW: f64 = -2.0;

/// `(log M_p(a), d/da log M_p(a))`, as in [`crate::adjoint`]: bit-identical
/// to the first two outputs of [`log_m_p_derivs`].
#[inline(always)]
fn log_m_p_d1(a: f64, p: usize) -> (f64, f64) {
    if a >= CF_BELOW {
        let cdf = cdf_fast(a);
        let pdf = norm_pdf(a);
        let m1 = a * cdf + pdf;
        let (mut m_km1, mut m_k) = (m1, (a * a + 1.0) * cdf + a * pdf);
        for kk in 3..=p {
            let next = a * m_k + (kk - 1) as f64 * m_km1;
            m_km1 = m_k;
            m_k = next;
        }
        (m_k.ln(), p as f64 * (m_km1 / m_k))
    } else {
        let (l, d1, _) = log_m_p_derivs(a, p);
        (l, d1)
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

    /// Random coordinates: `|h_i| ≤ 1`, `|a_k| ≤ amax`.
    fn coords(d: usize, amax: f64, rng: &mut ChaCha8Rng) -> Vec<f64> {
        let p = d - 1;
        let mut g = vec![0.0; dims::gamma_len(d)];
        for x in g[..p - 1].iter_mut() {
            *x = rng.random_range(-1.0..1.0);
        }
        for x in g[p - 1..].iter_mut() {
            *x = rng.random_range(-amax..amax);
        }
        g
    }

    /// Coordinates with the given log-eigenvalues (centred) and `|a_k| ≤ amax`.
    fn coords_with_ell(ell: &[f64], amax: f64, rng: &mut ChaCha8Rng) -> Vec<f64> {
        let p = ell.len();
        let mean = ell.iter().sum::<f64>() / p as f64;
        let centred: Vec<f64> = ell.iter().map(|l| l - mean).collect();
        let mut g = coords(p + 1, amax, rng);
        helmert_encode(&centred, p, &mut g[..p - 1]);
        g
    }

    fn mu(d: usize, rng: &mut ChaCha8Rng) -> Vec<f64> {
        let mut m = reference(d, rng);
        let k = rng.random_range(0.5..4.0);
        m.iter_mut().for_each(|x| *x *= k);
        m
    }

    /// A random rotation of `ℝ^p` (QR of a Gaussian matrix, determinant +1).
    fn rotation(p: usize, rng: &mut ChaCha8Rng) -> DMatrix<f64> {
        let a = DMatrix::from_fn(p, p, |_, _| rng.random_range(-1.0..1.0));
        let mut q = a.qr().q();
        if q.determinant() < 0.0 {
            for r in 0..p {
                q[(r, 0)] = -q[(r, 0)];
            }
        }
        q
    }

    fn max_abs_diff(a: &DMatrix<f64>, b: &DMatrix<f64>) -> f64 {
        (a - b).iter().map(|x| x.abs()).fold(0.0, f64::max)
    }

    fn frame(d: usize, turned: bool, rng: &mut ChaCha8Rng) -> SpFrame {
        let fr = SpFrame::new(&reference(d, rng));
        if turned {
            fr.with_basis(&rotation(d - 1, rng))
        } else {
            fr
        }
    }

    /// Value and gradient of the reverse kernel against the forward-mode
    /// kernel at `eta`, to `1e−9` absolute plus relative; the value must be
    /// bit-identical to `loglik_sp::<f64>`. Returns the largest scaled
    /// gradient discrepancy.
    fn check(fr: &SpFrame, eta: &[f64], y: &[f64], what: &str) -> f64 {
        let n = eta.len();
        let (mut g1, mut g2) = (vec![0.0; n], vec![0.0; n]);
        let f1 = loglik_sp_grad(fr, eta, y, &mut g1);
        let f2 = loglik_sp_grad_fwd(fr, eta, y, &mut g2);
        let f3 = loglik_sp::<f64>(fr, eta, y);
        assert!(f1.is_finite(), "{what}: value {f1}");
        assert_eq!(f1.to_bits(), f3.to_bits(), "{what}: value {f1} vs {f3}");
        assert!(
            (f1 - f2).abs() <= 1e-9 + 1e-9 * f2.abs(),
            "{what}: value {f1} vs {f2}"
        );
        let mut worst = 0.0;
        for k in 0..n {
            assert!(g1[k].is_finite(), "{what} k={k}: gradient {}", g1[k]);
            let err = (g1[k] - g2[k]).abs() / (1.0 + g2[k].abs());
            worst = f64::max(worst, err);
            assert!(
                (g1[k] - g2[k]).abs() <= 1e-9 + 1e-9 * g2[k].abs(),
                "{what} k={k}: reverse {} vs forward {}",
                g1[k],
                g2[k]
            );
        }
        worst
    }

    fn fd_grad(fr: &SpFrame, eta: &[f64], y: &[f64]) -> Vec<f64> {
        let h = 1e-6;
        (0..eta.len())
            .map(|k| {
                let mut ep = eta.to_vec();
                let mut em = eta.to_vec();
                ep[k] += h;
                em[k] -= h;
                (loglik_sp::<f64>(fr, &ep, y) - loglik_sp::<f64>(fr, &em, y)) / (2.0 * h)
            })
            .collect()
    }

    /// (1) Value and gradient of the reverse kernel agree with the
    /// forward-mode dual gradient at 20 random points for every
    /// `d = 3..=10`, with and without a basis rotation.
    #[test]
    fn matches_forward_mode_at_random_points() {
        let mut rng = ChaCha8Rng::seed_from_u64(21);
        let mut worst = 0.0;
        for d in 3..=10 {
            for turned in [false, true] {
                let fr = frame(d, turned, &mut rng);
                for i in 0..20 {
                    let mut eta = mu(d, &mut rng);
                    eta.extend(coords(d, 2.0, &mut rng));
                    let mut y = reference(d, &mut rng);
                    unit(&mut y);
                    let what = format!("d={d} turned={turned} point {i}");
                    worst = f64::max(worst, check(&fr, &eta, &y, &what));
                }
            }
        }
        println!("largest scaled gradient discrepancy: {worst:.2e}");
    }

    /// (2) The kernel's log-density equals the ESAG log-density of the
    /// covariance it parameterizes, at every `d`, with a basis rotation too.
    #[test]
    fn kernel_matches_the_covariance() {
        let mut rng = ChaCha8Rng::seed_from_u64(3);
        for d in 3..=10 {
            for turned in [false, true] {
                let fr = frame(d, turned, &mut rng);
                for _ in 0..4 {
                    let m = mu(d, &mut rng);
                    let g = coords(d, 1.5, &mut rng);
                    let mut eta = m.clone();
                    eta.extend(g.iter());
                    let mut y = reference(d, &mut rng);
                    unit(&mut y);
                    let a = loglik_sp::<f64>(&fr, &eta, &y);
                    let v = fr.covariance(&m, &g);
                    let dist = crate::distribution::Esag::from_covariance(&m, &v).unwrap();
                    let b = dist.log_pdf(&y);
                    assert!((a - b).abs() < 1e-9, "d={d} turned={turned}: {a} vs {b}");
                }
            }
        }
    }

    /// (3a) `V` is a valid ESAG covariance (`V μ̂ = μ̂`, `det V = 1`), its
    /// tangent eigenvalues are `e^{ℓ}`, and `coords → V → coords → V`
    /// returns the same `V`, with the canonical coordinates in the
    /// canonical gauge (ascending eigenvalues, `R` of non-negative trace).
    #[test]
    fn covariance_roundtrip() {
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        for d in 3..=10 {
            for turned in [false, true] {
                let fr = frame(d, turned, &mut rng);
                for _ in 0..3 {
                    let m = mu(d, &mut rng);
                    let g = coords(d, 1.5, &mut rng);
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
                    let w = fr.tangent_covariance(&g);
                    let mut ev: Vec<f64> = nalgebra::SymmetricEigen::new(w.clone())
                        .eigenvalues
                        .iter()
                        .map(|l| l.ln())
                        .collect();
                    ev.sort_by(f64::total_cmp);
                    let mut ex = fr.log_eigenvalues(&g);
                    assert!(ex.iter().sum::<f64>().abs() < 1e-12);
                    ex.sort_by(f64::total_cmp);
                    for (a, b) in ev.iter().zip(&ex) {
                        assert!((a - b).abs() < 1e-9, "log-eigenvalues {ev:?} vs {ex:?}");
                    }
                    let g2 = fr.coords_from_covariance(&m, &v);
                    assert!(g2.iter().all(|x| x.is_finite()), "{g2:?}");
                    let v2 = fr.covariance(&m, &g2);
                    assert!(
                        max_abs_diff(&v, &v2) < 1e-8,
                        "d={d} turned={turned}: V differs by {}",
                        max_abs_diff(&v, &v2)
                    );
                    let l2 = fr.log_eigenvalues(&g2);
                    for k in 1..d - 1 {
                        assert!(l2[k] >= l2[k - 1] - 1e-9, "{l2:?}");
                    }
                    assert!(fr.rotation(&g2).trace() >= -1e-12);
                }
            }
        }
    }

    /// (3b) Coordinates in the canonical gauge (ascending eigenvalues with
    /// gaps, `R` with a positive diagonal) roundtrip exactly.
    #[test]
    fn coordinates_roundtrip_in_the_canonical_gauge() {
        let mut rng = ChaCha8Rng::seed_from_u64(11);
        for d in 3..=10 {
            for turned in [false, true] {
                let fr = frame(d, turned, &mut rng);
                let p = d - 1;
                let m = mu(d, &mut rng);
                let ell: Vec<f64> = (0..p)
                    .map(|k| 0.4 * k as f64 + rng.random_range(0.0..0.3))
                    .collect();
                // Σ a_k² ≤ 1 < 2: every plane turns by less than π/2, so the
                // diagonal of R is positive and the sign gauge is recovered
                let na = p * (p - 1) / 2;
                let g = coords_with_ell(&ell, (1.0 / na as f64).sqrt(), &mut rng);
                let v = fr.covariance(&m, &g);
                let g2 = fr.coords_from_covariance(&m, &v);
                for (a, b) in g.iter().zip(&g2) {
                    assert!(
                        (a - b).abs() < 1e-7,
                        "d={d} turned={turned}: {g:?} vs {g2:?}"
                    );
                }
                // the same through the eigen-decomposition kept by the caller
                let r = fr.rotation(&g);
                let g3 = fr.coords_from_eigen_signed(&fr.eigenvalues(&g), r);
                for (a, b) in g.iter().zip(&g3) {
                    assert!((a - b).abs() < 1e-7, "d={d} signed: {g:?} vs {g3:?}");
                }
            }
        }
    }

    /// (4) The reverse gradient matches central differences of the value at
    /// a generic point, at isotropy (`γ = 0`), at a tie of two eigenvalues
    /// and at a tie with `A = 0`, and agrees with forward mode there.
    #[test]
    fn gradient_matches_differences_at_isotropy_and_ties() {
        let mut rng = ChaCha8Rng::seed_from_u64(5);
        for d in [3usize, 4, 5, 6, 8, 10] {
            for turned in [false, true] {
                let fr = frame(d, turned, &mut rng);
                let p = d - 1;
                let m = mu(d, &mut rng);
                let mut points: Vec<(&str, Vec<f64>)> = vec![];
                points.push(("generic", coords(d, 1.5, &mut rng)));
                points.push(("isotropy", vec![0.0; dims::gamma_len(d)]));
                let mut ell: Vec<f64> = (0..p).map(|_| rng.random_range(-1.0..1.0)).collect();
                ell[1] = ell[0];
                points.push(("tie", coords_with_ell(&ell, 1.5, &mut rng)));
                let mut tie0 = coords_with_ell(&ell, 1.0, &mut rng);
                tie0[p - 1..].iter_mut().for_each(|a| *a = 0.0);
                points.push(("tie at A = 0", tie0));
                if p >= 3 {
                    let mut ell3: Vec<f64> = (0..p).map(|_| rng.random_range(-1.0..1.0)).collect();
                    ell3[1] = ell3[0];
                    ell3[2] = ell3[0];
                    points.push(("triple tie", coords_with_ell(&ell3, 1.5, &mut rng)));
                }
                for (name, g) in &points {
                    let mut eta = m.clone();
                    eta.extend(g.iter());
                    let mut y = reference(d, &mut rng);
                    unit(&mut y);
                    let what = format!("d={d} turned={turned} {name}");
                    check(&fr, &eta, &y, &what);
                    let mut grad = vec![0.0; eta.len()];
                    let f = loglik_sp_grad(&fr, &eta, &y, &mut grad);
                    assert!(f.is_finite());
                    let fd = fd_grad(&fr, &eta, &y);
                    for k in 0..eta.len() {
                        let tol = 1e-5 * (1.0 + grad[k].abs());
                        assert!(
                            (grad[k] - fd[k]).abs() < tol,
                            "{what} k={k}: reverse {} vs fd {}",
                            grad[k],
                            fd[k]
                        );
                    }
                }
            }
        }
    }

    /// (5) `μ = 0` and `μ̂ = −m` give finite values and gradients equal to
    /// the forward kernel's (constant frames there).
    #[test]
    fn finite_at_zero_mu_and_the_antipode() {
        let mut rng = ChaCha8Rng::seed_from_u64(33);
        for d in [3usize, 4, 5, 7, 10] {
            for turned in [false, true] {
                let fr = frame(d, turned, &mut rng);
                let mut eta = vec![0.0; d];
                eta.extend(coords(d, 1.5, &mut rng));
                let mut y = reference(d, &mut rng);
                unit(&mut y);
                check(&fr, &eta, &y, &format!("d={d} turned={turned} mu = 0"));
                // μ̂ = −m exactly: a coordinate reference, so that m/‖m‖ = m
                // and μ = −2m normalises to −m without rounding
                let mut m = vec![0.0; d];
                m[1] = 1.0;
                let mut fr = SpFrame::new(&m);
                if turned {
                    fr = fr.with_basis(&rotation(d - 1, &mut rng));
                }
                let mut eta: Vec<f64> = m.iter().map(|x| -2.0 * x).collect();
                eta.extend(coords(d, 1.5, &mut rng));
                check(&fr, &eta, &y, &format!("d={d} turned={turned} antipode"));
                // and the covariance at μ = 0 is that of μ̂ = m
                let g = coords(d, 1.0, &mut rng);
                let v0 = fr.covariance(&vec![0.0; d], &g);
                let vm = fr.covariance(&m, &g);
                assert!(max_abs_diff(&v0, &vm) < 1e-12);
            }
        }
    }

    /// `R` is a proper rotation whose column `k` is the eigenvector of `W`
    /// for `e^{ℓ_k}`; at `p = 2` it turns by `2 atan(a/2)`.
    #[test]
    fn rotation_is_the_cayley_transform() {
        let mut rng = ChaCha8Rng::seed_from_u64(2);
        for d in 3..=10 {
            let fr = SpFrame::new(&reference(d, &mut rng));
            let p = d - 1;
            for _ in 0..3 {
                let g = coords(d, 3.0, &mut rng);
                let r = fr.rotation(&g);
                let eye = DMatrix::identity(p, p);
                assert!(max_abs_diff(&(r.transpose() * &r), &eye) < 1e-12);
                assert!((r.determinant() - 1.0).abs() < 1e-10);
                let w = fr.tangent_covariance(&g);
                let lam = fr.eigenvalues(&g);
                for k in 0..p {
                    let col = r.column(k);
                    let wc = &w * col;
                    for i in 0..p {
                        assert!((wc[i] - lam[k] * col[i]).abs() < 1e-10);
                    }
                }
                assert!((w.determinant() - 1.0).abs() < 1e-10);
                if p == 2 {
                    let th = 2.0 * (0.5 * g[1]).atan();
                    assert!((r[(0, 0)] - th.cos()).abs() < 1e-12);
                    assert!((r[(1, 0)] + th.sin()).abs() < 1e-12);
                    assert!(th.abs() < std::f64::consts::PI);
                }
            }
        }
    }

    /// Relabelling the axes is a gauge: `R_0 ↦ R_0 P`, `ℓ ↦ Pᵀ ℓ`,
    /// `A ↦ Pᵀ A P` give the same covariance, for a permutation `P`.
    #[test]
    fn relabelling_the_axes_is_a_gauge() {
        let mut rng = ChaCha8Rng::seed_from_u64(44);
        for d in [4usize, 5, 7] {
            let p = d - 1;
            let r0 = rotation(p, &mut rng);
            let fr = SpFrame::new(&reference(d, &mut rng)).with_basis(&r0);
            let m = mu(d, &mut rng);
            let g = coords(d, 1.5, &mut rng);
            // a cyclic shift of the labels
            let perm = DMatrix::from_fn(p, p, |r, c| if (r + 1) % p == c { 1.0 } else { 0.0 });
            let ell = DVector::from_vec(fr.log_eigenvalues(&g));
            let ell2 = perm.transpose() * ell;
            let a = {
                let mut a = DMatrix::zeros(p, p);
                let mut k = p - 1;
                for r in 0..p {
                    for c in r + 1..p {
                        a[(r, c)] = g[k];
                        a[(c, r)] = -g[k];
                        k += 1;
                    }
                }
                a
            };
            let a2 = perm.transpose() * a * &perm;
            let mut g2 = vec![0.0; g.len()];
            helmert_encode(ell2.as_slice(), p, &mut g2[..p - 1]);
            let mut k = p - 1;
            for r in 0..p {
                for c in r + 1..p {
                    g2[k] = a2[(r, c)];
                    k += 1;
                }
            }
            let fr2 = SpFrame::new(fr.lt_frame().reference()).with_basis(&(&r0 * &perm));
            let (v1, v2) = (fr.covariance(&m, &g), fr2.covariance(&m, &g2));
            assert!(
                max_abs_diff(&v1, &v2) < 1e-12,
                "d={d}: {}",
                max_abs_diff(&v1, &v2)
            );
        }
    }

    /// A rotation by exactly `π` in some plane (eigenvalue `−1`, possible in
    /// the canonical gauge from `p = 4` on) has no Cayley coordinates: the
    /// `a` part is `NaN`, the `h` part is valid.
    #[test]
    fn inverse_cayley_at_pi_is_nan() {
        let d = 5;
        let p = 4;
        let fr = SpFrame::new(&[1.0, 0.0, 0.0, 0.0, 0.0]);
        // I − 2 Proj onto span{(1,1,0,0), (0,0,1,1)}/√2: diagonal 0, det +1
        let r = DMatrix::from_row_slice(
            p,
            p,
            &[
                0.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, -1.0, 0.0,
            ],
        );
        assert!((r.determinant() - 1.0).abs() < 1e-12);
        let lam = [0.5, 1.0, 1.0, 2.0];
        let g = fr.coords_from_eigen(&lam, r);
        assert_eq!(g.len(), dims::gamma_len(d));
        assert!(g[..p - 1].iter().all(|x| x.is_finite()));
        assert!(g[p - 1..].iter().all(|x| x.is_nan()), "{g:?}");
        let ell = fr.log_eigenvalues(&g);
        let mean = lam.iter().map(|l| l.ln()).sum::<f64>() / p as f64;
        for k in 0..p {
            assert!((ell[k] - (lam[k].ln() - mean)).abs() < 1e-12);
        }
    }

    /// The Hessian by differences is symmetric and matches differences of
    /// the gradient taken with another step.
    #[test]
    fn hessian_differences_are_consistent() {
        let mut rng = ChaCha8Rng::seed_from_u64(17);
        let d = 5;
        let fr = frame(d, true, &mut rng);
        let mut eta = mu(d, &mut rng);
        eta.extend(coords(d, 1.0, &mut rng));
        let mut y = reference(d, &mut rng);
        unit(&mut y);
        let n = eta.len();
        let (mut g, mut h) = (vec![0.0; n], vec![0.0; n * n]);
        let f = loglik_sp_hess_fd(&fr, &eta, &y, &mut g, &mut h);
        assert_eq!(f.to_bits(), loglik_sp::<f64>(&fr, &eta, &y).to_bits());
        for i in 0..n {
            for j in 0..n {
                assert!((h[i * n + j] - h[j * n + i]).abs() < 1e-12);
            }
        }
        let step = 1e-4;
        for i in 0..n {
            let mut ep = eta.clone();
            let mut em = eta.clone();
            ep[i] += step;
            em[i] -= step;
            let (mut gp, mut gm) = (vec![0.0; n], vec![0.0; n]);
            loglik_sp_grad(&fr, &ep, &y, &mut gp);
            loglik_sp_grad(&fr, &em, &y, &mut gm);
            for j in 0..n {
                let hd = (gp[j] - gm[j]) / (2.0 * step);
                assert!(
                    (hd - h[i * n + j]).abs() < 1e-4 * (1.0 + hd.abs()),
                    "H[{i},{j}] {} vs {hd}",
                    h[i * n + j]
                );
            }
        }
    }

    /// Above `d = 10` the heap fallbacks agree with finite differences and
    /// with each other.
    #[test]
    fn heap_fallback_matches_finite_differences() {
        let mut rng = ChaCha8Rng::seed_from_u64(12);
        let d = 11;
        let fr = frame(d, true, &mut rng);
        let mut eta = mu(d, &mut rng);
        eta.extend(coords(d, 1.5, &mut rng));
        let mut y = reference(d, &mut rng);
        unit(&mut y);
        let mut g = vec![0.0; eta.len()];
        let f = loglik_sp_grad(&fr, &eta, &y, &mut g);
        assert_eq!(f.to_bits(), loglik_sp::<f64>(&fr, &eta, &y).to_bits());
        let fd = fd_grad(&fr, &eta, &y);
        for k in 0..eta.len() {
            assert!(
                (g[k] - fd[k]).abs() < 1e-5 * (1.0 + g[k].abs()),
                "k={k}: reverse {} vs fd {}",
                g[k],
                fd[k]
            );
        }
        let v = fr.covariance(&eta[..d], &eta[d..]);
        let b = crate::distribution::Esag::from_covariance(&eta[..d], &v)
            .unwrap()
            .log_pdf(&y);
        assert!((f - b).abs() < 1e-9, "{f} vs {b}");
    }

    /// Central differences (Richardson: steps `±h`, `±2h`) of the reverse
    /// gradient along coordinate `i`.
    fn gradient_difference(fr: &SpFrame, eta: &[f64], y: &[f64], i: usize, h: f64) -> Vec<f64> {
        let n = eta.len();
        let g = |k: f64| {
            let mut e = eta.to_vec();
            e[i] += k * h;
            let mut g = vec![0.0; n];
            loglik_sp_grad(fr, &e, y, &mut g);
            g
        };
        let (p1, m1, p2, m2) = (g(1.0), g(-1.0), g(2.0), g(-2.0));
        (0..n)
            .map(|r| (8.0 * (p1[r] - m1[r]) - (p2[r] - m2[r])) / (12.0 * h))
            .collect()
    }

    /// The exact Hessian at `eta`: value and gradient bit-identical to
    /// `loglik_sp_grad`, every entry finite, exactly symmetric, and the
    /// columns `cols` within `1e−6` (relative to `max(|·|, 1)`) of central
    /// differences of the gradient, and (with `fd`) all of it within `1e−6`
    /// of `loglik_sp_hess_fd`. Returns the worst error against the
    /// differences of the gradient.
    fn check_hess(
        fr: &SpFrame,
        eta: &[f64],
        y: &[f64],
        cols: &[usize],
        fd: bool,
        what: &str,
    ) -> f64 {
        let n = eta.len();
        let (mut g, mut h) = (vec![0.0; n], vec![0.0; n * n]);
        let mut g2 = vec![0.0; n];
        let f = loglik_sp_hess(fr, eta, y, &mut g, &mut h);
        let f2 = loglik_sp_grad(fr, eta, y, &mut g2);
        assert_eq!(f.to_bits(), f2.to_bits(), "{what}: value {f} vs {f2}");
        for k in 0..n {
            assert_eq!(g[k].to_bits(), g2[k].to_bits(), "{what}: gradient {k}");
        }
        for r in 0..n {
            for c in 0..n {
                assert!(
                    h[r * n + c].is_finite(),
                    "{what}: H[{r},{c}] = {}",
                    h[r * n + c]
                );
                assert_eq!(
                    h[r * n + c].to_bits(),
                    h[c * n + r].to_bits(),
                    "{what}: symmetry"
                );
            }
        }
        if fd {
            let (mut g3, mut h3) = (vec![0.0; n], vec![0.0; n * n]);
            loglik_sp_hess_fd(fr, eta, y, &mut g3, &mut h3);
            for k in 0..n * n {
                let err = (h[k] - h3[k]).abs() / h3[k].abs().max(1.0);
                assert!(
                    err < 1e-6,
                    "{what}: H[{},{}] = {} vs loglik_sp_hess_fd {}",
                    k / n,
                    k % n,
                    h[k],
                    h3[k]
                );
            }
        }
        let mut worst: f64 = 0.0;
        for &i in cols {
            let col = gradient_difference(fr, eta, y, i, 1e-3 * eta[i].abs().max(1.0));
            for r in 0..n {
                let err = (h[r * n + i] - col[r]).abs() / col[r].abs().max(1.0);
                worst = worst.max(err);
                assert!(
                    err < 1e-6,
                    "{what}: H[{r},{i}] = {} vs differences {}",
                    h[r * n + i],
                    col[r]
                );
            }
        }
        worst
    }

    /// The exact Hessian matches differences of the reverse gradient and
    /// `loglik_sp_hess_fd` at random points for every `d = 3..=11` (the
    /// stack kernels and the heap fallback), with and without a basis
    /// rotation.
    #[test]
    fn hessian_matches_differences_at_random_points() {
        let mut rng = ChaCha8Rng::seed_from_u64(61);
        let mut worst: f64 = 0.0;
        for d in 3..=11 {
            for turned in [false, true] {
                let fr = frame(d, turned, &mut rng);
                for i in 0..3 {
                    let mut eta = mu(d, &mut rng);
                    eta.extend(coords(d, 2.0, &mut rng));
                    let y = reference(d, &mut rng);
                    let all: Vec<usize> = (0..eta.len()).collect();
                    let what = format!("d={d} turned={turned} point {i}");
                    worst = worst.max(check_hess(&fr, &eta, &y, &all, true, &what));
                }
            }
        }
        println!("Hessian vs differences of the gradient: worst {worst:.1e}");
    }

    /// The same at the special points of the chart: isotropy, all
    /// eigenvalues equal with `A ≠ 0` (`h = 0`), `A = 0` with distinct
    /// eigenvalues, a tie, a triple tie, a tie at `A = 0`, and `μ̂` at and
    /// next to the reference direction.
    #[test]
    fn hessian_matches_differences_at_the_special_points() {
        let mut rng = ChaCha8Rng::seed_from_u64(62);
        let mut worst: f64 = 0.0;
        for d in [3usize, 4, 5, 6, 8, 10] {
            for turned in [false, true] {
                let fr = frame(d, turned, &mut rng);
                let p = d - 1;
                let m = mu(d, &mut rng);
                let mut points: Vec<(&str, Vec<f64>, Vec<f64>)> = vec![];
                points.push(("isotropy", m.clone(), vec![0.0; dims::gamma_len(d)]));
                let mut flat = coords(d, 1.5, &mut rng);
                flat[..p - 1].iter_mut().for_each(|h| *h = 0.0);
                points.push(("h = 0", m.clone(), flat));
                let mut a0 = coords(d, 1.5, &mut rng);
                a0[p - 1..].iter_mut().for_each(|a| *a = 0.0);
                points.push(("a = 0", m.clone(), a0));
                let mut ell: Vec<f64> = (0..p).map(|_| rng.random_range(-1.0..1.0)).collect();
                ell[1] = ell[0];
                points.push(("tie", m.clone(), coords_with_ell(&ell, 1.5, &mut rng)));
                let mut tie0 = coords_with_ell(&ell, 1.0, &mut rng);
                tie0[p - 1..].iter_mut().for_each(|a| *a = 0.0);
                points.push(("tie at a = 0", m.clone(), tie0));
                if p >= 3 {
                    let mut ell3: Vec<f64> = (0..p).map(|_| rng.random_range(-1.0..1.0)).collect();
                    ell3[1] = ell3[0];
                    ell3[2] = ell3[0];
                    let g3 = coords_with_ell(&ell3, 1.5, &mut rng);
                    points.push(("triple tie", m.clone(), g3));
                }
                let r = fr.lt_frame().reference().to_vec();
                let at_m: Vec<f64> = r.iter().map(|x| 1.7 * x).collect();
                points.push(("mu at m", at_m, coords(d, 1.5, &mut rng)));
                let near_m: Vec<f64> = r
                    .iter()
                    .map(|x| 1.7 * x + 1e-3 * rng.random_range(-1.0..1.0))
                    .collect();
                points.push(("mu near m", near_m, coords(d, 1.5, &mut rng)));
                for (name, mu, g) in &points {
                    let mut eta = mu.clone();
                    eta.extend(g.iter());
                    let y = reference(d, &mut rng);
                    let all: Vec<usize> = (0..eta.len()).collect();
                    let what = format!("d={d} turned={turned} {name}");
                    worst = worst.max(check_hess(&fr, &eta, &y, &all, true, &what));
                }
            }
        }
        println!("Hessian vs differences of the gradient: worst {worst:.1e}");
    }

    /// At the guarded points `μ = 0` and `μ̂ = −m` the frame is constant;
    /// the Hessian follows that convention: finite and symmetric, and its
    /// shape columns (along which the guard holds) match differences of
    /// the gradient. At `μ = 0` the mean block is `Φ_yy yyᵀ − I` (`q` does
    /// not depend on `μ`).
    #[test]
    fn hessian_at_the_guarded_points() {
        let mut rng = ChaCha8Rng::seed_from_u64(63);
        for d in [3usize, 4, 5, 7, 10] {
            for turned in [false, true] {
                let fr = frame(d, turned, &mut rng);
                let mut eta = vec![0.0; d];
                eta.extend(coords(d, 1.5, &mut rng));
                let y = reference(d, &mut rng);
                let shape: Vec<usize> = (d..eta.len()).collect();
                check_hess(&fr, &eta, &y, &shape, false, &format!("d={d} mu = 0"));
                // q does not depend on μ: the mean block is Φ_yy yyᵀ − I
                let n = eta.len();
                let (mut g, mut h) = (vec![0.0; n], vec![0.0; n * n]);
                loglik_sp_hess(&fr, &eta, &y, &mut g, &mut h);
                let k = (0..d)
                    .max_by(|&a, &b| y[a].abs().total_cmp(&y[b].abs()))
                    .unwrap();
                let phi_yy = (h[k * n + k] + 1.0) / (y[k] * y[k]);
                for r in 0..d {
                    for c in 0..d {
                        let expect = phi_yy * y[r] * y[c] - if r == c { 1.0 } else { 0.0 };
                        assert!(
                            (h[r * n + c] - expect).abs() < 1e-12 * (1.0 + expect.abs()),
                            "d={d} mu = 0: H[{r},{c}] = {} vs {expect}",
                            h[r * n + c]
                        );
                    }
                }
                // μ̂ = −m exactly (see finite_at_zero_mu_and_the_antipode)
                let mut m = vec![0.0; d];
                m[1] = 1.0;
                let mut fr = SpFrame::new(&m);
                if turned {
                    fr = fr.with_basis(&rotation(d - 1, &mut rng));
                }
                let mut eta: Vec<f64> = m.iter().map(|x| -2.0 * x).collect();
                eta.extend(coords(d, 1.5, &mut rng));
                check_hess(&fr, &eta, &y, &shape, false, &format!("d={d} antipode"));
            }
        }
    }

    /// Next to the antipode of `m` the frame turns fast (its derivatives
    /// grow like `1/‖μ̂ + m‖`), and the Hessian still matches differences of
    /// the gradient taken at a step that resolves that scale.
    #[test]
    fn hessian_near_the_antipode() {
        let mut rng = ChaCha8Rng::seed_from_u64(64);
        for d in [3usize, 5, 8] {
            let fr = frame(d, true, &mut rng);
            let m = fr.lt_frame().reference().to_vec();
            let e0 = fr.lt_frame().tangent_basis()[..d].to_vec();
            for eps in [1e-2, 1e-4] {
                let mut eta: Vec<f64> = (0..d).map(|k| 2.0 * (-m[k] + eps * e0[k])).collect();
                eta.extend(coords(d, 1.0, &mut rng));
                let y = reference(d, &mut rng);
                let n = eta.len();
                let (mut g, mut h) = (vec![0.0; n], vec![0.0; n * n]);
                loglik_sp_hess(&fr, &eta, &y, &mut g, &mut h);
                for i in 0..n {
                    let step = if i < d { 1e-3 * eps } else { 1e-3 };
                    let col = gradient_difference(&fr, &eta, &y, i, step);
                    for r in 0..n {
                        let scale = col[r].abs().max(1.0);
                        assert!(
                            (h[r * n + i] - col[r]).abs() < 1e-6 * scale,
                            "d={d} eps={eps}: H[{r},{i}] = {} vs {}",
                            h[r * n + i],
                            col[r]
                        );
                    }
                }
            }
        }
    }

    /// The frame serialises and deserialises to itself, with and without a
    /// basis.
    #[test]
    fn serde_roundtrip() {
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        for turned in [false, true] {
            let fr = frame(6, turned, &mut rng);
            let s = serde_json::to_string(&fr).unwrap();
            let back: SpFrame = serde_json::from_str(&s).unwrap();
            assert_eq!(fr, back);
            assert_eq!(fr.basis().is_some(), turned);
        }
    }
}

#[cfg(test)]
mod bench {
    use super::*;
    use crate::gamma_rot::{GrFrame, RotLink};
    use rand::{RngExt, SeedableRng};
    use rand_chacha::ChaCha8Rng;
    use std::hint::black_box;
    use std::time::Instant;

    /// Gradient cost per observation of the reverse spectral kernel against
    /// the forward-mode one, the log-tangent kernel and the reverse `γ`-rot
    /// kernel.
    /// `cargo test --release --lib spectral::bench -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn kernel_cost() {
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let n = 20_000;
        for d in [3usize, 5, 8, 10] {
            let k = dims::eta_len(d);
            let mut m: Vec<f64> = (0..d).map(|_| rng.random_range(-1.0..1.0)).collect();
            let nm = m.iter().map(|x| x * x).sum::<f64>().sqrt();
            m.iter_mut().for_each(|x| *x /= nm);
            let sp = SpFrame::new(&m);
            let lt = LtFrame::new(&m);
            let gr = GrFrame::new(&m, RotLink::Stereo);
            let etas: Vec<Vec<f64>> = (0..n)
                .map(|_| {
                    (0..k)
                        .map(|j| {
                            let s = if j < d { 1.5 } else { 0.6 };
                            s * rng.random_range(-1.0..1.0)
                        })
                        .collect()
                })
                .collect();
            let ys: Vec<Vec<f64>> = (0..n)
                .map(|_| {
                    let mut v: Vec<f64> = (0..d).map(|_| rng.random_range(-1.0..1.0)).collect();
                    let s = v.iter().map(|x| x * x).sum::<f64>().sqrt();
                    v.iter_mut().for_each(|x| *x /= s);
                    v
                })
                .collect();
            let mut g = vec![0.0; k];
            let time = |f: &mut dyn FnMut()| {
                let mut best = f64::INFINITY;
                for _ in 0..5 {
                    let t = Instant::now();
                    f();
                    best = best.min(t.elapsed().as_secs_f64());
                }
                best / n as f64 * 1e9
            };
            let mut acc = 0.0;
            let rev = time(&mut || {
                for i in 0..n {
                    acc += loglik_sp_grad(&sp, black_box(&etas[i]), &ys[i], &mut g);
                }
                black_box(&g);
            });
            let fwd = time(&mut || {
                for i in 0..n {
                    acc += loglik_sp_grad_fwd(&sp, black_box(&etas[i]), &ys[i], &mut g);
                }
                black_box(&g);
            });
            let ltg = time(&mut || {
                for i in 0..n {
                    acc +=
                        crate::logtangent::loglik_lt_grad(&lt, black_box(&etas[i]), &ys[i], &mut g);
                }
                black_box(&g);
            });
            let grg = time(&mut || {
                for i in 0..n {
                    acc += crate::gamma_rot_adjoint::loglik_gr_grad_rev(
                        &gr,
                        black_box(&etas[i]),
                        &ys[i],
                        &mut g,
                    );
                }
                black_box(&g);
            });
            black_box(acc);
            println!(
                "d={d:2} N={k:2}: spectral reverse {rev:5.0} ns/obs, forward {fwd:5.0}, log-tangent {ltg:5.0}, gamma-rot reverse {grg:5.0}"
            );
        }
    }

    /// Cost per observation of the exact spectral Hessian against the
    /// Hessian by differences of the reverse gradient, the reverse gradient
    /// alone and the log-tangent kernel's exact Hessian (each with the
    /// value and the gradient).
    /// `cargo test --release --lib spectral::bench::hessian_cost -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn hessian_cost() {
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let n = 20_000;
        for d in [3usize, 5, 8, 10] {
            let k = dims::eta_len(d);
            let mut m: Vec<f64> = (0..d).map(|_| rng.random_range(-1.0..1.0)).collect();
            let nm = m.iter().map(|x| x * x).sum::<f64>().sqrt();
            m.iter_mut().for_each(|x| *x /= nm);
            let fr = SpFrame::new(&m);
            let lt = LtFrame::new(&m);
            let etas: Vec<Vec<f64>> = (0..n)
                .map(|_| {
                    (0..k)
                        .map(|j| {
                            let s = if j < d { 1.5 } else { 0.6 };
                            s * rng.random_range(-1.0..1.0)
                        })
                        .collect()
                })
                .collect();
            let ys: Vec<Vec<f64>> = (0..n)
                .map(|_| {
                    let mut v: Vec<f64> = (0..d).map(|_| rng.random_range(-1.0..1.0)).collect();
                    let s = v.iter().map(|x| x * x).sum::<f64>().sqrt();
                    v.iter_mut().for_each(|x| *x /= s);
                    v
                })
                .collect();
            let (mut g, mut h) = (vec![0.0; k], vec![0.0; k * k]);
            let time = |f: &mut dyn FnMut(usize), m: usize| {
                let mut best = f64::INFINITY;
                for _ in 0..5 {
                    let t = Instant::now();
                    f(m);
                    best = best.min(t.elapsed().as_secs_f64());
                }
                best / m as f64 * 1e9
            };
            let mut acc = 0.0;
            let exact = time(
                &mut |m| {
                    for i in 0..m {
                        acc += loglik_sp_hess(&fr, black_box(&etas[i]), &ys[i], &mut g, &mut h);
                    }
                    black_box(&h);
                },
                n,
            );
            let fd = time(
                &mut |m| {
                    for i in 0..m {
                        acc += loglik_sp_hess_fd(&fr, black_box(&etas[i]), &ys[i], &mut g, &mut h);
                    }
                    black_box(&h);
                },
                n / 10,
            );
            let rev = time(
                &mut |m| {
                    for i in 0..m {
                        acc += loglik_sp_grad(&fr, black_box(&etas[i]), &ys[i], &mut g);
                    }
                    black_box(&g);
                },
                n,
            );
            let lth = time(
                &mut |m| {
                    for i in 0..m {
                        acc += crate::logtangent::loglik_lt_hess(
                            &lt,
                            black_box(&etas[i]),
                            &ys[i],
                            &mut g,
                            &mut h,
                        );
                    }
                    black_box(&h);
                },
                n,
            );
            black_box(acc);
            println!(
                "d={d:2} N={k:2}: spectral exact Hessian {exact:6.0} ns/obs, differences {fd:6.0}, reverse gradient {rev:4.0}, log-tangent exact Hessian {lth:6.0}"
            );
        }
    }
}
