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
        let frame = self.frame();
        let mut out = DMatrix::zeros(n, d);
        let mut buf = vec![0.0; d];
        for i in 0..n {
            self.sample_one(&frame, rng, &mut buf);
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
