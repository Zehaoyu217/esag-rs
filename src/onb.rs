//! Orthonormal basis of `R^d` whose last vector is `mu/|mu|` (R: `ONB`,
//! the `ESAG_Project2.R` version, including its degenerate branches).

use crate::scalar::Scalar;

/// Returns a `d × d` matrix in **column-major** order (`out[c*d + r]` is row
/// `r` of column `c`). Column `d-1` (0-based) is `mu/|mu|` (or `e_d` if
/// `mu = 0`), column 0 is `(-mu_2, mu_1, 0, …)/|·|` (or `e_1`), and column
/// `i-2` for `i = 3..d` is `(mu_1 mu_i, …, mu_{i-1} mu_i, -Σ_{k<i} mu_k², 0, …)`
/// normalised (or `e_{i-1}` when `mu_1 = … = mu_{i-1} = 0`).
pub fn onb<S: Scalar>(mu: &[S]) -> Vec<S> {
    let d = mu.len();
    assert!(d >= 3, "onb needs d >= 3, got {d}");
    let mut u = vec![S::zero(); d * d];
    // last column: mu_hat
    let n1 = mu.iter().fold(S::zero(), |a, &v| a + v * v);
    if n1.re() == 0.0 {
        u[(d - 1) * d + (d - 1)] = S::one();
    } else {
        let inv = S::one() / n1.sqrt();
        for r in 0..d {
            u[(d - 1) * d + r] = mu[r] * inv;
        }
    }
    // first column
    let n2 = mu[0] * mu[0] + mu[1] * mu[1];
    if n2.re() == 0.0 {
        u[0] = S::one();
    } else {
        let inv = S::one() / n2.sqrt();
        u[0] = -mu[1] * inv;
        u[1] = mu[0] * inv;
    }
    // columns 1..d-2 (0-based), from i = 3..d (1-based)
    let mut partial = n2; // Σ_{k < i} mu_k² for i = 3
    for i in 3..=d {
        let col = i - 2;
        if partial.re() == 0.0 {
            u[col * d + col] = S::one();
        } else {
            let mi = mu[i - 1];
            // |u_i|² = mi² Σ_{k<i} mu_k² + (Σ_{k<i} mu_k²)² = partial (mi² + partial)
            let norm = (partial * (mi * mi + partial)).sqrt();
            let inv = S::one() / norm;
            for r in 0..i - 1 {
                u[col * d + r] = mu[r] * mi * inv;
            }
            u[col * d + (i - 1)] = -partial * inv;
        }
        partial += mu[i - 1] * mu[i - 1];
    }
    u
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check_orthonormal(u: &[f64], d: usize) {
        for a in 0..d {
            for b in 0..d {
                let dot: f64 = (0..d).map(|r| u[a * d + r] * u[b * d + r]).sum();
                let target = if a == b { 1.0 } else { 0.0 };
                assert!((dot - target).abs() < 1e-13, "col {a}·col {b} = {dot}");
            }
        }
    }

    #[test]
    fn orthonormal_generic_and_degenerate() {
        for d in 3..=7 {
            let generic: Vec<f64> = (0..d)
                .map(|i| 0.3 + 0.7 * i as f64 * if i % 2 == 0 { 1.0 } else { -1.0 })
                .collect();
            let mut deg1 = generic.clone();
            deg1[0] = 0.0;
            deg1[1] = 0.0;
            let mut last_only = vec![0.0; d];
            last_only[d - 1] = -2.5;
            for mu in [generic, deg1, last_only, vec![0.0; d]] {
                let u = onb(&mu);
                check_orthonormal(&u, d);
                let n: f64 = mu.iter().map(|v| v * v).sum::<f64>().sqrt();
                if n > 0.0 {
                    for r in 0..d {
                        assert!((u[(d - 1) * d + r] - mu[r] / n).abs() < 1e-15);
                    }
                }
            }
        }
    }
}
