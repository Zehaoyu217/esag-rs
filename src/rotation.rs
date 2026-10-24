//! The rotation `R ∈ SO(d-1)` built from `(theta, phi)` (R: `rotation`).
//!
//! `R` is a product of plane (Givens) rotations. With `Lo(θ)` the rotation in
//! coordinates (1, 2) and `P_a(φ)` the rotation in coordinates (a, a+1)
//! (1-based, block `[[cos, −sin], [sin, cos]]`), the R code computes
//! ```text
//! R = I
//! for m in 1..=d-3:
//!     La = Π_{j=1}^{d-m-2} P_{j+1}(phi[1 - j + (d-m-1)(d-m-2)/2])
//!     R  = R · Lo(theta[d-m-1]) · La
//! R = R · Lo(theta[1])
//! ```
//! [`givens_sequence`] yields that product's factors in order, so callers can
//! right-multiply any matrix by `R` without forming it.

use crate::scalar::Scalar;

/// The plane-rotation factors of `R` in multiplication order, as
/// `(a, angle)` with `a` the 0-based first coordinate of the plane `(a, a+1)`.
pub fn givens_sequence<S: Scalar>(theta: &[S], phi: &[S]) -> Vec<(usize, S)> {
    let d = theta.len() + 2;
    let mut seq = Vec::with_capacity(theta.len() + phi.len());
    if d >= 4 {
        for m in 1..=d - 3 {
            seq.push((0, theta[d - m - 2]));
            for j in 1..=d - m - 2 {
                // 1-based index 1 - j + (d-m-1)(d-m-2)/2 → 0-based
                let idx = (d - m - 1) * (d - m - 2) / 2 - j;
                seq.push((j, phi[idx]));
            }
        }
    }
    seq.push((0, theta[0]));
    seq
}

/// Right-multiply the column-major `nrows × ncols` matrix `m` by the plane
/// rotation in columns `(a, a+1)` with angle `angle`.
#[inline]
pub fn right_rotate<S: Scalar>(m: &mut [S], nrows: usize, a: usize, angle: S) {
    let (c, s) = (angle.cos(), angle.sin());
    let (lo, hi) = m.split_at_mut((a + 1) * nrows);
    let ca = &mut lo[a * nrows..];
    let cb = &mut hi[..nrows];
    for r in 0..nrows {
        let (x, y) = (ca[r], cb[r]);
        ca[r] = c * x + s * y;
        cb[r] = c * y - s * x;
    }
}

/// The full `(d-1) × (d-1)` rotation matrix in **row-major** order.
pub fn rotation_matrix<S: Scalar>(theta: &[S], phi: &[S]) -> Vec<S> {
    let k = theta.len() + 1;
    let mut m = vec![S::zero(); k * k]; // column-major
    for i in 0..k {
        m[i * k + i] = S::one();
    }
    for (a, ang) in givens_sequence(theta, phi) {
        right_rotate(&mut m, k, a, ang);
    }
    // column-major → row-major
    let mut out = vec![S::zero(); k * k];
    for c in 0..k {
        for r in 0..k {
            out[r * k + c] = m[c * k + r];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::param::gamma_to_angles;

    fn matmul(a: &[f64], b: &[f64], k: usize) -> Vec<f64> {
        let mut o = vec![0.0; k * k];
        for i in 0..k {
            for j in 0..k {
                o[i * k + j] = (0..k).map(|l| a[i * k + l] * b[l * k + j]).sum();
            }
        }
        o
    }

    fn plane(k: usize, a: usize, ang: f64) -> Vec<f64> {
        let mut m = vec![0.0; k * k];
        for i in 0..k {
            m[i * k + i] = 1.0;
        }
        m[a * k + a] = ang.cos();
        m[(a + 1) * k + a + 1] = ang.cos();
        m[a * k + a + 1] = -ang.sin();
        m[(a + 1) * k + a] = ang.sin();
        m
    }

    #[test]
    fn d3_is_a_plane_rotation() {
        let r = rotation_matrix(&[0.7_f64], &[]);
        let e = plane(2, 0, 0.7);
        for i in 0..4 {
            assert!((r[i] - e[i]).abs() < 1e-15);
        }
    }

    #[test]
    fn matches_explicit_product_and_is_orthogonal() {
        for d in 4..=7 {
            let g: Vec<f64> = (0..crate::dims::gamma_len(d))
                .map(|i| ((i * 37 % 11) as f64 - 5.0) * 0.4 + 0.1)
                .collect();
            let a = gamma_to_angles(&g);
            let r = rotation_matrix(&a.theta, &a.phi);
            let k = d - 1;
            // explicit transcription of the R loop
            let mut e = plane(k, 0, 0.0);
            for m in 1..=d - 3 {
                let lo = plane(k, 0, a.theta[d - m - 2]);
                let mut la = plane(k, 0, 0.0);
                for j in 1..=d - m - 2 {
                    let idx = 1 - j as isize + ((d - m - 1) * (d - m - 2) / 2) as isize;
                    la = matmul(&la, &plane(k, j, a.phi[(idx - 1) as usize]), k);
                }
                e = matmul(&matmul(&e, &lo, k), &la, k);
            }
            e = matmul(&e, &plane(k, 0, a.theta[0]), k);
            for i in 0..k * k {
                assert!((r[i] - e[i]).abs() < 1e-13, "d={d} entry {i}");
            }
            // RᵀR = I
            for i in 0..k {
                for j in 0..k {
                    let dot: f64 = (0..k).map(|l| r[l * k + i] * r[l * k + j]).sum();
                    assert!((dot - if i == j { 1.0 } else { 0.0 }).abs() < 1e-13);
                }
            }
        }
    }
}
