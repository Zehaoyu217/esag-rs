//! Yu & Huang's parameterization `gamma → (lambda, theta, phi)` of the ESAG
//! shape (R: `parameter`).
//!
//! `gamma` (length `(d-2)(d+1)/2`) is split into `d-2` consecutive groups;
//! group `i = 1..d-2` has `i+1` entries. Each group `G = (G_1..G_n)` gives
//!
//! * `K_i = |G| + 1` — the ratio of consecutive eigenvalues,
//! * `theta_i = atan2(G_n, G_{n-1})` — a rotation angle in the leading plane,
//! * `phi_{i,j} = acos(G_j / |G_{j..n}|)`, `j = 1..n-2` — further angles,
//!
//! and the eigenvalues are `lambda_1 = (Π_i K_i^{d-1-i})^{-1/(d-1)}`,
//! `lambda_j = K_{j-1} lambda_{j-1}`, so `Π lambda = 1` and
//! `lambda_1 ≤ … ≤ lambda_{d-1}`. `gamma = 0` is the isotropic case.

use crate::dims;
use crate::scalar::Scalar;

/// Eigenvalues and rotation angles derived from `gamma`.
#[derive(Clone, Debug, PartialEq)]
pub struct Angles<S> {
    /// Eigenvalues `lambda_1 ≤ … ≤ lambda_{d-1}` of `V` on the tangent space (product 1).
    pub lambda: Vec<S>,
    /// `theta_1..theta_{d-2}`.
    pub theta: Vec<S>,
    /// `phi` angles, concatenated group by group (length `(d-2)(d-3)/2`).
    pub phi: Vec<S>,
}

/// Map `gamma` to `(lambda, theta, phi)` exactly as the R function `parameter`.
///
/// # Panics
/// If `gamma.len()` is not `(d-2)(d+1)/2` for some `d ≥ 3`.
pub fn gamma_to_angles<S: Scalar>(gamma: &[S]) -> Angles<S> {
    let d = dims::dim_from_gamma_len(gamma.len()).unwrap_or_else(|| {
        panic!(
            "gamma has length {}, which is not (d-2)(d+1)/2 for any d >= 3",
            gamma.len()
        )
    });
    let ng = d - 2;
    let mut k: Vec<S> = Vec::with_capacity(ng);
    let mut theta: Vec<S> = Vec::with_capacity(ng);
    let mut phi: Vec<S> = Vec::with_capacity(dims::phi_len(d));

    for i in 1..=ng {
        let start = i * (i + 1) / 2 - 1;
        let n = i + 1;
        let g = &gamma[start..start + n];
        let ss = g.iter().fold(S::zero(), |acc, &v| acc + v * v);
        k.push(ss.sqrt() + 1.0);
        // phi_{i,j}, j = 1..n-2 (none for the first group)
        for j in 0..n.saturating_sub(2) {
            let tail = g[j..].iter().fold(S::zero(), |acc, &v| acc + v * v);
            if tail.re() == 0.0 {
                phi.push(S::zero());
            } else {
                phi.push((g[j] / tail.sqrt()).acos());
            }
        }
        let (a, b) = (g[n - 2], g[n - 1]);
        if a.re() == 0.0 && b.re() == 0.0 {
            theta.push(S::zero());
        } else {
            theta.push(b.atan2(a));
        }
    }

    // log lambda_1 = -(1/(d-1)) Σ_i (d-1-i) log K_i
    let mut log_prod = S::zero();
    for (idx, &ki) in k.iter().enumerate() {
        let i = idx + 1;
        log_prod += ki.ln() * ((d - 1 - i) as f64);
    }
    let mut lambda: Vec<S> = Vec::with_capacity(d - 1);
    lambda.push((-log_prod / ((d - 1) as f64)).exp());
    for j in 1..d - 1 {
        let prev = lambda[j - 1];
        lambda.push(prev * k[j - 1]);
    }
    Angles { lambda, theta, phi }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn d3_closed_form() {
        let a = gamma_to_angles(&[0.6_f64, -0.8]);
        let k = 2.0;
        assert_relative_eq!(a.lambda[0], k.powf(-0.5), max_relative = 1e-14);
        assert_relative_eq!(a.lambda[1], k.powf(0.5), max_relative = 1e-14);
        assert_relative_eq!(a.theta[0], (-0.8f64).atan2(0.6), max_relative = 1e-14);
        assert!(a.phi.is_empty());
    }

    #[test]
    fn isotropic_is_identity() {
        for d in 3..=7 {
            let a = gamma_to_angles(&vec![0.0_f64; dims::gamma_len(d)]);
            assert!(a.lambda.iter().all(|&l| (l - 1.0).abs() < 1e-15));
            assert!(a.theta.iter().all(|&t| t == 0.0));
            assert!(a.phi.iter().all(|&t| t == 0.0));
        }
    }

    #[test]
    fn product_one_and_ordered() {
        let mut s = 12345u64;
        let mut next = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 11) as f64 / (1u64 << 53) as f64) * 6.0 - 3.0
        };
        for d in 3..=8 {
            for _ in 0..20 {
                let g: Vec<f64> = (0..dims::gamma_len(d)).map(|_| next()).collect();
                let a = gamma_to_angles(&g);
                assert_eq!(a.lambda.len(), d - 1);
                assert_eq!(a.theta.len(), d - 2);
                assert_eq!(a.phi.len(), dims::phi_len(d));
                let prod: f64 = a.lambda.iter().product();
                assert_relative_eq!(prod, 1.0, max_relative = 1e-12);
                for w in a.lambda.windows(2) {
                    assert!(w[0] <= w[1]);
                }
            }
        }
    }
}
