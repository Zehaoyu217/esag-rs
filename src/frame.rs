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

impl Frame<f64> {
    /// The frame of `γ`'s map in the tangent frame transported from `lt`'s
    /// reference ([`crate::gamma_transported`]): `P = [Ξ(μ̂) R | μ̂]`, with
    /// `Ξ(μ̂) = R(m → μ̂) E` in place of `onb(μ)`'s tangent columns.
    pub fn transported(lt: &crate::logtangent::LtFrame, mu: &[f64], gamma: &[f64]) -> Frame<f64> {
        let d = mu.len();
        let ang = gamma_to_angles(gamma);
        debug_assert_eq!(ang.lambda.len(), d - 1, "mu and gamma dimensions disagree");
        let xi = lt.frame_at(mu);
        let mut basis = vec![0.0; d * d];
        for j in 0..d - 1 {
            for r in 0..d {
                basis[j * d + r] = xi[(r, j)];
            }
        }
        let mu_norm = mu.iter().map(|v| v * v).sum::<f64>().sqrt();
        for r in 0..d {
            basis[(d - 1) * d + r] = if mu_norm > 0.0 {
                mu[r] / mu_norm
            } else {
                lt.reference()[r]
            };
        }
        {
            let e = &mut basis[..(d - 1) * d];
            for (a, angle) in givens_sequence(&ang.theta, &ang.phi) {
                right_rotate(e, d, a, angle);
            }
        }
        Frame {
            d,
            mu_norm,
            lambda: ang.lambda,
            basis,
        }
    }
}

impl Frame<f64> {
    /// The frame of the `γ`-rot map ([`crate::gamma_rot`]): the eigen-axes
    /// of `W = fr.tangent_covariance(gamma)` carried into `ℝ^d` by the
    /// tangent frame `Ξ(μ̂) R_0`, and `μ̂` last.
    pub fn gamma_rot(fr: &crate::gamma_rot::GrFrame, mu: &[f64], gamma: &[f64]) -> Frame<f64> {
        let d = mu.len();
        let p = d - 1;
        let w = fr.tangent_covariance(gamma);
        let eig = nalgebra::SymmetricEigen::new(w);
        let xi = fr.tangent_frame(mu);
        let u = &xi * &eig.eigenvectors;
        let mut basis = vec![0.0; d * d];
        for j in 0..p {
            for r in 0..d {
                basis[j * d + r] = u[(r, j)];
            }
        }
        let mu_norm = mu.iter().map(|v| v * v).sum::<f64>().sqrt();
        for r in 0..d {
            basis[p * d + r] = if mu_norm > 0.0 {
                mu[r] / mu_norm
            } else {
                fr.lt_frame().reference()[r]
            };
        }
        Frame {
            d,
            mu_norm,
            lambda: eig.eigenvalues.iter().copied().collect(),
            basis,
        }
    }

    /// The frame of the spectral map ([`crate::spectral`]): the eigen-axes
    /// are the columns of `Ξ(μ̂) R_0 R` with `R` the chart's rotation, the
    /// eigenvalues `e^{ℓ}`; `μ̂` last. No eigen-decomposition is needed.
    pub fn spectral(fr: &crate::spectral::SpFrame, mu: &[f64], gamma: &[f64]) -> Frame<f64> {
        let d = mu.len();
        let p = d - 1;
        let xi = fr.tangent_frame(mu);
        let u = &xi * fr.rotation(gamma);
        let mut basis = vec![0.0; d * d];
        for j in 0..p {
            for r in 0..d {
                basis[j * d + r] = u[(r, j)];
            }
        }
        let mu_norm = mu.iter().map(|v| v * v).sum::<f64>().sqrt();
        for r in 0..d {
            basis[p * d + r] = if mu_norm > 0.0 {
                mu[r] / mu_norm
            } else {
                fr.lt_frame().reference()[r]
            };
        }
        Frame {
            d,
            mu_norm,
            lambda: fr.eigenvalues(gamma),
            basis,
        }
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
    fn transported_frame_matches_its_kernel() {
        for d in 3..=7 {
            let m: Vec<f64> = (0..d).map(|i| 0.8 + 0.3 * (i as f64 * 1.7).sin()).collect();
            let lt = crate::logtangent::LtFrame::new(&m);
            let mu: Vec<f64> = (0..d).map(|i| 1.2 * (i as f64 * 0.9 + 0.4).cos()).collect();
            let g: Vec<f64> = (0..crate::dims::gamma_len(d))
                .map(|i| 0.7 * (i as f64 * 2.3 + 0.1).sin())
                .collect();
            let f = Frame::transported(&lt, &mu, &g);
            let v = nalgebra::DMatrix::from_row_slice(d, d, &f.cov_pow(1.0));
            let want =
                crate::gamma_smooth::GsFrame::gamma_transported(lt.reference()).covariance(&mu, &g);
            assert!((&v - &want).abs().max() < 1e-12, "d = {d}");
            let y: Vec<f64> = (0..d).map(|i| (i as f64 * 1.3).cos()).collect();
            let n = y.iter().map(|x| x * x).sum::<f64>().sqrt();
            let y: Vec<f64> = y.iter().map(|x| x / n).collect();
            let mut eta = mu.clone();
            eta.extend_from_slice(&g);
            let mut grad = vec![0.0; eta.len()];
            let a = crate::gamma_transported::loglik_gt_grad(&lt, &eta, &y, &mut grad);
            let b = crate::loglik::loglik_frame(&f, &mu, &y);
            assert!(
                (a - b).abs() < 1e-12 * b.abs().max(1.0),
                "d = {d}: {a} vs {b}"
            );
        }
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
