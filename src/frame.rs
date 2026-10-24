//! The eigen-frame of the ESAG covariance `V` (R: `covariance_matrix_gamma`).
//!
//! `V = P diag(lambda_1, …, lambda_{d-1}, 1) Pᵀ` with `P = [U_{1..d-1} R | μ̂]`,
//! where `U = onb(mu)` and `R = rotation(theta, phi)`. All quantities needed
//! by the density, the residuals and sampling are cheap in this frame:
//! `yᵀ V^s y = Σ_j lambda_j^s (v_j·y)² + (μ̂·y)²`.

use crate::onb::onb;
use crate::param::gamma_to_angles;
use crate::rotation::{givens_sequence, right_rotate};
use crate::scalar::Scalar;

/// Eigenvectors and eigenvalues of `V`.
#[derive(Clone, Debug)]
pub struct Frame<S: Scalar> {
    /// Dimension `d`.
    pub d: usize,
    /// `|mu|`.
    pub mu_norm: S,
    /// Eigenvalues `lambda_1..lambda_{d-1}` (the eigenvalue of `μ̂` is 1).
    pub lambda: Vec<S>,
    /// `d × d` column-major: columns `v_1..v_{d-1}` (eigenvectors for
    /// `lambda`), then `μ̂`.
    pub basis: Vec<S>,
}

impl<S: Scalar> Frame<S> {
    /// Build the frame for `ESAG(mu, gamma)`.
    pub fn new(mu: &[S], gamma: &[S]) -> Self {
        let d = mu.len();
        let ang = gamma_to_angles(gamma);
        debug_assert_eq!(ang.lambda.len(), d - 1, "mu and gamma dimensions disagree");
        let mut basis = onb(mu);
        // E = U[:, :d-1] · R, applied factor by factor to the first d-1 columns
        {
            let e = &mut basis[..(d - 1) * d];
            for (a, angle) in givens_sequence(&ang.theta, &ang.phi) {
                right_rotate(e, d, a, angle);
            }
        }
        let mu_norm = mu.iter().fold(S::zero(), |a, &v| a + v * v).sqrt();
        Self {
            d,
            mu_norm,
            lambda: ang.lambda,
            basis,
        }
    }

    /// Column `j` of the basis.
    #[inline]
    pub fn col(&self, j: usize) -> &[S] {
        &self.basis[j * self.d..(j + 1) * self.d]
    }

    /// `μ̂ = mu/|mu|`.
    #[inline]
    pub fn mu_hat(&self) -> &[S] {
        self.col(self.d - 1)
    }

    /// `yᵀ V⁻¹ y` without forming `V`.
    #[inline]
    pub fn quad_form_inv(&self, y: &[f64]) -> S {
        let d = self.d;
        let mut q = S::zero();
        for j in 0..d {
            let c = self.col(j);
            let mut p = S::zero();
            for k in 0..d {
                p += c[k] * y[k];
            }
            if j + 1 < d {
                q += p * p / self.lambda[j];
            } else {
                q += p * p;
            }
        }
        q
    }

    /// `V^s` as a `d × d` row-major matrix (`s = 1`: `V`; `-1`: `V⁻¹`;
    /// `0.5`: the symmetric square root).
    pub fn cov_pow(&self, s: f64) -> Vec<S> {
        let d = self.d;
        let mut out = vec![S::zero(); d * d];
        for j in 0..d {
            let w = if j + 1 < d {
                self.lambda[j].powf(s)
            } else {
                S::one()
            };
            let c = self.col(j);
            for r in 0..d {
                let wr = w * c[r];
                for k in 0..d {
                    out[r * d + k] += wr * c[k];
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mat(a: &[f64], b: &[f64], d: usize) -> Vec<f64> {
        let mut o = vec![0.0; d * d];
        for i in 0..d {
            for j in 0..d {
                o[i * d + j] = (0..d).map(|k| a[i * d + k] * b[k * d + j]).sum();
            }
        }
        o
    }

    #[test]
    fn covariance_properties() {
        for d in 3..=6 {
            let mu: Vec<f64> = (0..d)
                .map(|i| 0.5 + i as f64 * 0.3 - if i == 1 { 1.4 } else { 0.0 })
                .collect();
            let g: Vec<f64> = (0..crate::dims::gamma_len(d))
                .map(|i| 0.4 * (i as f64 + 1.0).sin() * 2.0)
                .collect();
            let f = Frame::new(&mu, &g);
            let v = f.cov_pow(1.0);
            let vi = f.cov_pow(-1.0);
            let vh = f.cov_pow(0.5);
            // V mu = mu
            for r in 0..d {
                let s: f64 = (0..d).map(|k| v[r * d + k] * mu[k]).sum();
                assert!((s - mu[r]).abs() < 1e-12);
            }
            // V V^-1 = I, V^0.5 V^0.5 = V
            let id = mat(&v, &vi, d);
            let vv = mat(&vh, &vh, d);
            for i in 0..d {
                for j in 0..d {
                    assert!((id[i * d + j] - if i == j { 1.0 } else { 0.0 }).abs() < 1e-11);
                    assert!((vv[i * d + j] - v[i * d + j]).abs() < 1e-11);
                }
            }
            // det V = Π lambda = 1 (via nalgebra)
            let m = nalgebra::DMatrix::from_row_slice(d, d, &v);
            assert!((m.determinant() - 1.0).abs() < 1e-10);
            // quad form
            let y: Vec<f64> = (0..d).map(|i| (i as f64 * 1.3).cos()).collect();
            let q1 = f.quad_form_inv(&y);
            let q2: f64 = (0..d)
                .map(|i| (0..d).map(|k| y[i] * vi[i * d + k] * y[k]).sum::<f64>())
                .sum();
            assert!((q1 - q2).abs() < 1e-12 * q2.abs().max(1.0));
        }
    }
}
