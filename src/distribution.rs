//! The ESAG(μ, γ) distribution on the unit sphere S^{d-1}.
//!
//! `Esag` is an immutable value type holding the natural parameters
//! `mu ∈ R^d` and `gamma ∈ R^{(d-2)(d+1)/2}` (Yu & Huang's parameterization).
//! It exposes the density, sampling, the covariance matrix and the
//! eigen-structure (eigenvalues `lambda`, rotation angles `theta`/`phi`).

use crate::dims;
use crate::error::{EsagError, Result};
use crate::frame::Frame;
use crate::loglik;
use crate::param::{Angles, gamma_to_angles};
use nalgebra::{DMatrix, DVector};
use rand::Rng;
use rand_distr::{Distribution, StandardNormal};
use serde::{Deserialize, Serialize};

/// An ESAG distribution with parameters `(mu, gamma)`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Esag {
    mu: Vec<f64>,
    gamma: Vec<f64>,
}

impl Esag {
    /// Create an ESAG(μ, γ) distribution. `gamma` must have length
    /// `(d-2)(d+1)/2` where `d = mu.len() >= 3`; pass an all-zero `gamma` for
    /// the isotropic angular Gaussian.
    pub fn new(mu: &[f64], gamma: &[f64]) -> Result<Self> {
        let d = mu.len();
        if d < 3 {
            return Err(EsagError::Dimension(format!(
                "ESAG needs d >= 3, got d = {d}"
            )));
        }
        let g = dims::gamma_len(d);
        if gamma.len() != g {
            return Err(EsagError::Dimension(format!(
                "gamma must have length (d-2)(d+1)/2 = {g} for d = {d}, got {}",
                gamma.len()
            )));
        }
        if mu.iter().chain(gamma.iter()).any(|v| !v.is_finite()) {
            return Err(EsagError::Data("non-finite parameter".into()));
        }
        Ok(Self {
            mu: mu.to_vec(),
            gamma: gamma.to_vec(),
        })
    }

    /// Isotropic angular Gaussian with mean vector `mu` (γ = 0).
    pub fn isotropic(mu: &[f64]) -> Result<Self> {
        let g = dims::gamma_len(mu.len());
        Self::new(mu, &vec![0.0; g])
    }

    /// Build from the stacked natural parameter `eta = (mu, gamma)`.
    pub fn from_eta(eta: &[f64]) -> Result<Self> {
        let d = dims::dim_from_eta_len(eta.len()).ok_or_else(|| {
            EsagError::Dimension(format!(
                "eta length {} does not correspond to any d >= 3",
                eta.len()
            ))
        })?;
        Self::new(&eta[..d], &eta[d..])
    }

    /// The ESAG distribution with mean `mu` and covariance `v` (`V mu = mu`
    /// up to rounding; `V` is rescaled to determinant 1), via the canonical
    /// `gamma` of [`crate::param::gamma_from_covariance`]. Exactly tied
    /// tangent eigenvalues are separated by a relative 1e-10 first: the
    /// `gamma` family reaches such `V` only as a limit.
    pub fn from_covariance(mu: &[f64], v: &DMatrix<f64>) -> Result<Self> {
        let d = mu.len();
        let rowmajor: Vec<f64> = (0..d)
            .flat_map(|r| (0..d).map(move |c| (r, c)))
            .map(|(r, c)| v[(r, c)])
            .collect();
        let gamma = crate::param::gamma_from_covariance(mu, &rowmajor);
        let dist = Self::new(mu, &gamma)?;
        let scale = v.determinant().powf(1.0 / d as f64);
        let err = (dist.covariance() - v / scale).abs().max();
        if err <= 1e-8 * (v / scale).abs().max().max(1.0) {
            return Ok(dist);
        }
        // separate exact ties along the tangent eigen-directions
        let eig = nalgebra::SymmetricEigen::new((v + v.transpose()) * 0.5);
        let mut vals = eig.eigenvalues.clone();
        for (j, x) in vals.iter_mut().enumerate() {
            *x *= 1.0 + 1e-10 * j as f64;
        }
        let v2 = &eig.eigenvectors * DMatrix::from_diagonal(&vals) * eig.eigenvectors.transpose();
        let rowmajor: Vec<f64> = (0..d)
            .flat_map(|r| (0..d).map(move |c| (r, c)))
            .map(|(r, c)| v2[(r, c)])
            .collect();
        Self::new(mu, &crate::param::gamma_from_covariance(mu, &rowmajor))
    }

    /// Ambient dimension `d` (the data live on S^{d-1}).
    pub fn dim(&self) -> usize {
        self.mu.len()
    }
    /// The mean vector `mu` (not normalised).
    pub fn mu(&self) -> &[f64] {
        &self.mu
    }
    /// The shape parameter `gamma`.
    pub fn gamma(&self) -> &[f64] {
        &self.gamma
    }
    /// `eta = (mu, gamma)` stacked.
    pub fn eta(&self) -> Vec<f64> {
        let mut e = self.mu.clone();
        e.extend_from_slice(&self.gamma);
        e
    }
    /// `|mu|`, the concentration-like scale.
    pub fn mu_norm(&self) -> f64 {
        self.mu.iter().map(|v| v * v).sum::<f64>().sqrt()
    }
    /// Mean direction `mu/|mu|` (the mode of the density).
    pub fn mean_direction(&self) -> Vec<f64> {
        let n = self.mu_norm();
        self.mu.iter().map(|v| v / n).collect()
    }
    /// Eigenvalues `lambda_1 ≤ … ≤ lambda_{d-1}` (product 1) and rotation angles.
    pub fn angles(&self) -> Angles<f64> {
        gamma_to_angles(&self.gamma)
    }
    /// The covariance frame (eigenvectors + eigenvalues).
    pub fn frame(&self) -> Frame<f64> {
        Frame::new(&self.mu, &self.gamma)
    }
    /// The d×d covariance matrix `V` (`V mu = mu`, `|V| = 1`).
    pub fn covariance(&self) -> DMatrix<f64> {
        self.cov_pow(1.0)
    }
    /// `V^{-1}`.
    pub fn covariance_inv(&self) -> DMatrix<f64> {
        self.cov_pow(-1.0)
    }
    /// `V^s` for any real `s` (e.g. 0.5 for the symmetric square root).
    pub fn cov_pow(&self, s: f64) -> DMatrix<f64> {
        let d = self.dim();
        let v = self.frame().cov_pow(s); // row-major
        DMatrix::from_row_slice(d, d, &v)
    }

    /// Log-density at the unit vector `y` (with respect to surface measure).
    pub fn log_pdf(&self, y: &[f64]) -> f64 {
        debug_assert_eq!(y.len(), self.dim());
        loglik::loglik_mu_gamma(&self.mu, &self.gamma, y)
    }
    /// Density at the unit vector `y`.
    pub fn pdf(&self, y: &[f64]) -> f64 {
        self.log_pdf(y).exp()
    }
    /// Log-density for every row of `y` (n×d).
    pub fn log_pdf_rows(&self, y: &DMatrix<f64>) -> DVector<f64> {
        let d = self.dim();
        let mut out = DVector::zeros(y.nrows());
        let mut row = vec![0.0; d];
        for i in 0..y.nrows() {
            for j in 0..d {
                row[j] = y[(i, j)];
            }
            out[i] = self.log_pdf(&row);
        }
        out
    }

    /// A sampler with the scaled eigenvectors precomputed (fast repeated
    /// draws).
    pub fn sampler(&self) -> Sampler {
        Sampler::new(self)
    }

    /// Exact second moment `E[Y Yᵀ]` (`d × d`), see [`second_moment_weights`].
    pub fn second_moment(&self) -> DMatrix<f64> {
        let frame = self.frame();
        let d = frame.d;
        let (w, w_mu) = second_moment_weights(&frame.lambda, frame.mu_norm);
        let mut m = DMatrix::zeros(d, d);
        for j in 0..d {
            let (v, wj) = if j + 1 < d {
                (frame.col(j), w[j])
            } else {
                (frame.mu_hat(), w_mu)
            };
            for a in 0..d {
                for b in 0..d {
                    m[(a, b)] += wj * v[a] * v[b];
                }
            }
        }
        m
    }

    /// The diagonal of [`Esag::second_moment`], `E[Y_k²]`.
    pub fn second_moment_diag(&self) -> Vec<f64> {
        let frame = self.frame();
        second_moment_diag_from_frame(&frame)
    }

    /// Draw one sample: `X ~ N_d(mu, V)`, `Y = X/|X|`.
    pub fn sample_one<R: Rng + ?Sized>(&self, frame: &Frame<f64>, rng: &mut R, out: &mut [f64]) {
        let d = self.dim();
        out.copy_from_slice(&self.mu);
        // X = mu + Σ_j sqrt(lambda_j) z_j v_j + z_d mu_hat
        for j in 0..d {
            let z: f64 = StandardNormal.sample(rng);
            let scale = if j + 1 < d {
                frame.lambda[j].sqrt()
            } else {
                1.0
            };
            let col = &frame.basis[j * d..(j + 1) * d];
            for k in 0..d {
                out[k] += scale * z * col[k];
            }
        }
        let norm = out.iter().map(|v| v * v).sum::<f64>().sqrt();
        for v in out.iter_mut() {
            *v /= norm;
        }
    }

    /// Draw `n` samples as an n×d matrix (one unit vector per row).
    pub fn sample<R: Rng + ?Sized>(&self, n: usize, rng: &mut R) -> DMatrix<f64> {
        let d = self.dim();
        let sampler = self.sampler();
        let mut out = DMatrix::zeros(n, d);
        let mut buf = vec![0.0; d];
        for i in 0..n {
            sampler.sample_into(rng, &mut buf);
            for j in 0..d {
                out[(i, j)] = buf[j];
            }
        }
        out
    }

    /// Quadratic residual `Q = rᵀ V⁻¹ r` with `r = (I − μ̂μ̂ᵀ) y`, and the scaled
    /// residual `T1 = (|mu|² + Σ lambda + 1) · Q` which is approximately
    /// χ²_{d−1} under the model.
    pub fn residuals(&self, y: &[f64]) -> (f64, f64) {
        let frame = self.frame();
        let q = residual_q(&frame, y);
        let s: f64 = frame.lambda.iter().sum();
        let t1 = (frame.mu_norm * frame.mu_norm + s + 1.0) * q;
        (q, t1)
    }

    /// Mahalanobis-type distance `(y − μ̂)ᵀ V⁻¹ (y − μ̂)` used for
    /// confidence/prediction regions.
    pub fn region_statistic(&self, y: &[f64]) -> f64 {
        let frame = self.frame();
        region_statistic(&frame, y)
    }
}

/// `Q = rᵀ V⁻¹ r`, `r = (I − μ̂μ̂ᵀ) y`, computed in the eigenbasis.
pub(crate) fn residual_q(frame: &Frame<f64>, y: &[f64]) -> f64 {
    let d = frame.d;
    // r is y minus its component along mu_hat; mu_hat is the last basis column
    // and is an eigenvector of V^{-1} with eigenvalue 1, so
    // Q = Σ_{j<d} (v_j · y)² / lambda_j  (the mu_hat component of r is 0).
    let mut q = 0.0;
    for j in 0..d - 1 {
        let col = &frame.basis[j * d..(j + 1) * d];
        let proj: f64 = col.iter().zip(y).map(|(a, b)| a * b).sum();
        q += proj * proj / frame.lambda[j];
    }
    q
}

/// `(y − μ̂)ᵀ V⁻¹ (y − μ̂)` in the eigenbasis.
pub(crate) fn region_statistic(frame: &Frame<f64>, y: &[f64]) -> f64 {
    let d = frame.d;
    let mut q = 0.0;
    for j in 0..d {
        let col = &frame.basis[j * d..(j + 1) * d];
        // (y - mu_hat)·v_j = y·v_j - δ_{j,d-1}
        let mut proj: f64 = col.iter().zip(y).map(|(a, b)| a * b).sum();
        if j == d - 1 {
            proj -= 1.0;
            q += proj * proj;
        } else {
            q += proj * proj / frame.lambda[j];
        }
    }
    q
}

/// Fast repeated sampling from one ESAG distribution: `X = μ + W z`,
/// `z ~ N(0, I_d)`, `W = [√λ_1 v_1 | … | √λ_{d−1} v_{d−1} | μ̂]`,
/// `Y = X/‖X‖`.
#[derive(Clone, Debug)]
pub struct Sampler {
    d: usize,
    mu: Vec<f64>,
    mu_norm: f64,
    /// column-major `d × d` scaled eigenvectors
    w: Vec<f64>,
    /// eigenvalues `λ_1..λ_{d−1}`
    lambda: Vec<f64>,
    /// `√λ_1..√λ_{d−1}`
    sqrt_lambda: Vec<f64>,
}

impl Sampler {
    /// Precompute the scaled eigen-frame of `dist`.
    pub fn new(dist: &Esag) -> Self {
        let frame = dist.frame();
        let d = frame.d;
        let sqrt_lambda: Vec<f64> = frame.lambda.iter().map(|l| l.sqrt()).collect();
        let mut w = frame.basis.clone();
        for j in 0..d - 1 {
            for v in &mut w[j * d..(j + 1) * d] {
                *v *= sqrt_lambda[j];
            }
        }
        Self {
            d,
            mu: dist.mu().to_vec(),
            mu_norm: frame.mu_norm,
            w,
            lambda: frame.lambda.clone(),
            sqrt_lambda,
        }
    }

    /// Dimension `d`.
    pub fn dim(&self) -> usize {
        self.d
    }

    /// Draw one unit vector into `out` (length `d`).
    #[inline]
    pub fn sample_into<R: Rng + ?Sized>(&self, rng: &mut R, out: &mut [f64]) {
        let d = self.d;
        out.copy_from_slice(&self.mu);
        for j in 0..d {
            let z: f64 = StandardNormal.sample(rng);
            let col = &self.w[j * d..(j + 1) * d];
            for (o, c) in out.iter_mut().zip(col) {
                *o += z * c;
            }
        }
        let inv = 1.0 / out.iter().map(|v| v * v).sum::<f64>().sqrt();
        for v in out.iter_mut() {
            *v *= inv;
        }
    }

    /// Draw the region statistic `(Y − μ̂)ᵀ V⁻¹ (Y − μ̂)` of one sample
    /// directly from `z`, in `O(d)` and without forming `Y`:
    /// with `a = ‖μ‖ + z_d` and `‖X‖² = Σ_{j<d} λ_j z_j² + a²`,
    /// the statistic is `(Σ_{j<d} z_j²)/‖X‖² + (a/‖X‖ − 1)²`.
    #[inline]
    pub fn region_statistic_draw<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        let d = self.d;
        let mut zz = 0.0;
        let mut x2 = 0.0;
        for j in 0..d - 1 {
            let z: f64 = StandardNormal.sample(rng);
            zz += z * z;
            x2 += self.lambda[j] * z * z;
        }
        let zd: f64 = StandardNormal.sample(rng);
        let a = self.mu_norm + zd;
        x2 += a * a;
        let xn = x2.sqrt();
        let t = a / xn - 1.0;
        zz / x2 + t * t
    }

    /// `√λ_j` (exposed for tests).
    pub fn sqrt_lambda(&self) -> &[f64] {
        &self.sqrt_lambda
    }
}

#[cfg(test)]
mod sampler_tests {
    use super::*;
    use crate::FastRng;
    use rand::SeedableRng;

    #[test]
    fn region_statistic_from_z_matches_explicit_formula() {
        // same RNG stream: z drawn by region_statistic_draw equals the z that
        // sample_into would use, so compare against region_statistic(Y)
        for (mu, gamma) in [
            (vec![1.0, 2.0, 1.5], vec![0.7, -0.3]),
            (vec![0.5, 1.0, -1.0, 2.0], vec![0.6, -0.2, 0.3, 0.9, -0.5]),
        ] {
            let dist = Esag::new(&mu, &gamma).unwrap();
            let s = dist.sampler();
            let frame = dist.frame();
            let d = dist.dim();
            let mut r1 = FastRng::seed_from_u64(1);
            let mut r2 = FastRng::seed_from_u64(1);
            let mut y = vec![0.0; d];
            for _ in 0..1000 {
                let q1 = s.region_statistic_draw(&mut r1);
                s.sample_into(&mut r2, &mut y);
                let q2 = region_statistic(&frame, &y);
                assert!((q1 - q2).abs() < 1e-12 * q2.max(1.0), "{q1} vs {q2}");
            }
        }
    }
}

/// `E[Y_k²]` from an eigen-frame of `V`.
pub(crate) fn second_moment_diag_from_frame(frame: &Frame<f64>) -> Vec<f64> {
    let d = frame.d;
    let (w, w_mu) = second_moment_weights(&frame.lambda, frame.mu_norm);
    let mut out = vec![0.0; d];
    for j in 0..d {
        let (v, wj) = if j + 1 < d {
            (frame.col(j), w[j])
        } else {
            (frame.mu_hat(), w_mu)
        };
        for k in 0..d {
            out[k] += wj * v[k] * v[k];
        }
    }
    out
}

/// Step of the trapezoid rule in `s = ln t` used by [`second_moment_weights`].
const MOMENT_STEP: f64 = 0.35;
/// Start of the grid in `s`, shifted by `−ln(1 + ‖μ‖² + Σλ)` (the part below
/// then contributes less than `e^{−32}`).
const MOMENT_S_START: f64 = -32.0;
/// End of the grid in `s` if the terms have not become negligible earlier.
const MOMENT_S_END: f64 = 60.0;

/// Eigen-weights of the exact second moment of `Y = X/‖X‖`,
/// `X ~ N(μ, V)`, `V = Σ_j λ_j v_j v_jᵀ + μ̂μ̂ᵀ`:
/// `E[YYᵀ] = Σ_j w_j v_j v_jᵀ + w_μ μ̂μ̂ᵀ`, returned as `(w, w_μ)`.
///
/// From `1/‖x‖² = ∫₀^∞ e^{−t‖x‖²} dt` and the Gaussian integral in the eigen-
/// basis of `V` (where `μ = ‖μ‖ μ̂` has a single coordinate):
///
/// ```text
/// c(t) = (1+2t)^{−1/2} Π_j (1+2tλ_j)^{−1/2} exp(−t‖μ‖²/(1+2t)),
/// w_j  = ∫₀^∞ c(t) λ_j/(1+2tλ_j) dt,
/// w_μ  = ∫₀^∞ c(t) [1/(1+2t) + ‖μ‖²/(1+2t)²] dt,     Σ w + w_μ = 1.
/// ```
///
/// After `t = e^s` the integrands are analytic and decay exponentially in
/// both directions. The factor `exp(−t‖μ‖²/(1+2t))` limits the useful strip
/// of analyticity to `|Im s| < π/2`, so the trapezoid rule converges like
/// `exp(−π²/h)` whatever the concentration: step 0.35 (0.3 for
/// `‖μ‖ > 20`) gives ≲ 1e−12, measured up to `‖μ‖ = 80`, with about 100–300
/// nodes once negligible terms are cut (≈ 1 µs). This replaces the Monte-Carlo moments of the R code, which
/// carry sampling noise of order `n_mc^{−1/2}` and cost ~25 µs per 1000 draws.
pub fn second_moment_weights(lambda: &[f64], mu_norm: f64) -> (Vec<f64>, f64) {
    let r2 = mu_norm * mu_norm;
    let mut w = vec![0.0; lambda.len()];
    let mut w_mu = 0.0;
    let mut total = 0.0;
    let mut prev = f64::INFINITY;
    // near t = 0 the integrands total t·(1 + ‖μ‖² + Σλ): start where the
    // omitted part is below e^{MOMENT_S_START}
    let scale = 1.0 + r2 + lambda.iter().sum::<f64>();
    // very concentrated distributions need a slightly finer step for 1e-12
    let h = if mu_norm > 20.0 { 0.3 } else { MOMENT_STEP };
    let mut s = MOMENT_S_START - scale.ln();
    let lam_min = lambda.iter().fold(1.0f64, |a, &b| a.min(b));
    let s_settled = (0.5 / lam_min).ln() + 1.0;
    while s <= MOMENT_S_END {
        let t = s.exp();
        s += h;
        let a = 1.0 + 2.0 * t;
        let mut prod = a;
        for &l in lambda {
            prod *= 1.0 + 2.0 * t * l;
        }
        // trapezoid weight h times the Jacobian dt = t ds
        let c = (-t * r2 / a).exp() / prod.sqrt() * t * h;
        let mut term = c * (1.0 / a + r2 / (a * a));
        w_mu += term;
        for (wj, &l) in w.iter_mut().zip(lambda) {
            let v = c * l / (1.0 + 2.0 * t * l);
            *wj += v;
            term += v;
        }
        total += term;
        // past the peak, past every eigen-scale 1/(2λ_j), and negligible:
        // the rest decays geometrically
        if term < prev && s > s_settled && term < 1e-17 * total {
            break;
        }
        prev = term;
    }
    (w, w_mu)
}

#[cfg(test)]
mod second_moment_tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn weights_sum_to_one() {
        for (lam, r) in [
            (vec![1.0, 1.0], 0.0),
            (vec![0.3, 3.3333333333333335], 2.5),
            (vec![0.05, 0.5, 40.0], 12.0),
            (vec![0.2, 0.7, 1.5, 2.0, 2.380952380952381], 0.4),
            (vec![0.05, 0.5, 40.0], 80.0),
            (vec![1.0, 1.0], 25.0),
        ] {
            let (w, wm) = second_moment_weights(&lam, r);
            let tot: f64 = w.iter().sum::<f64>() + wm;
            assert!((tot - 1.0).abs() < 1e-11, "lam {lam:?} r {r}: total {tot}");
        }
    }

    #[test]
    fn matches_monte_carlo() {
        let dist = Esag::new(&[1.2, -0.4, 0.8, 0.3], &[0.9, -0.5, 1.4, 0.2, -0.7]).unwrap();
        let exact = dist.second_moment();
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(9);
        let n = 2_000_000;
        let sampler = dist.sampler();
        let mut buf = vec![0.0; 4];
        let mut mc = DMatrix::<f64>::zeros(4, 4);
        for _ in 0..n {
            sampler.sample_into(&mut rng, &mut buf);
            for a in 0..4 {
                for b in 0..4 {
                    mc[(a, b)] += buf[a] * buf[b] / n as f64;
                }
            }
        }
        // MC standard error ≤ 0.5/sqrt(n) ≈ 3.5e-4
        assert!((&exact - &mc).abs().max() < 2e-3, "exact {exact}\nmc {mc}");
        assert!((exact.trace() - 1.0).abs() < 1e-11);
        let diag = dist.second_moment_diag();
        for k in 0..4 {
            assert!((diag[k] - exact[(k, k)]).abs() < 1e-15);
        }
    }

    #[test]
    fn isotropic_limit_is_uniform() {
        // mu → 0, V = I: Y uniform on the sphere, E[YYᵀ] = I/d
        let (w, wm) = second_moment_weights(&[1.0, 1.0, 1.0, 1.0], 1e-12);
        for v in w.iter().chain(std::iter::once(&wm)) {
            assert!((v - 0.2).abs() < 1e-12);
        }
    }
}

#[cfg(test)]
mod second_moment_timing {
    /// `cargo test --release --lib second_moment_timing -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn cost_per_call() {
        for (lam, r) in [
            (vec![0.6, 1.6666666666666667], 2.0),
            (vec![0.3, 0.8, 1.4, 1.9, 1.5, 1.2, 0.9], 4.0),
            (vec![0.05, 0.5, 40.0], 30.0),
        ] {
            let t = std::time::Instant::now();
            let mut acc = 0.0;
            for k in 0..100_000 {
                let (w, wm) = super::second_moment_weights(
                    std::hint::black_box(&lam),
                    r + (k % 7) as f64 * 1e-3,
                );
                acc += w[0] + wm;
            }
            std::hint::black_box(acc);
            println!(
                "d={} r={r}: {:.2} µs",
                lam.len() + 1,
                t.elapsed().as_secs_f64() * 1e6 / 1e5
            );
        }
    }
}
