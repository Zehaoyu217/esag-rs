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

/// Inverse of the parameterization: `γ` from eigenvalues `lambda`
/// (ascending, any positive scale; they are normalised to product 1) and a
/// rotation `r ∈ SO(d−1)` (row-major `(d−1)×(d−1)`), so that
/// `gamma_to_angles(γ)` reproduces `lambda` and `rotation_matrix(θ, φ) = r`
/// (for generic inputs; on the singular set the angle conventions of the
/// forward map apply).
///
/// `R` is peeled one group at a time from the outside in: its last column is
/// `A_{d−2} e_{d−2}`, a unit vector whose spherical coordinates are the angles
/// `(φ_{d−2,·}, θ_{d−2})` of the outermost factor; then `R ← A_{d−2}ᵀ R`
/// fixes the last coordinate and the same step repeats on the leading block.
/// Group norms come from the eigenvalue ratios, `‖G_i‖ = λ_{i+1}/λ_i − 1`.
///
/// # Panics
/// If the lengths are inconsistent or `lambda` is not positive and
/// ascending.
pub fn angles_to_gamma(lambda: &[f64], r: &[f64]) -> Vec<f64> {
    let k = lambda.len();
    let d = k + 1;
    assert!(d >= 3, "need d >= 3");
    assert_eq!(r.len(), k * k, "rotation must be (d-1)x(d-1)");
    assert!(
        lambda.iter().all(|&l| l > 0.0),
        "eigenvalues must be positive"
    );
    for w in lambda.windows(2) {
        assert!(
            w[0] <= w[1] * (1.0 + 1e-12),
            "eigenvalues must be ascending"
        );
    }
    let mut m = nalgebra::DMatrix::from_row_slice(k, k, r);
    let mut gamma = vec![0.0; dims::gamma_len(d)];
    for gr in (1..=d - 2).rev() {
        let n = gr + 1; // active block size and group length
        let v: Vec<f64> = (0..n).map(|i| m[(i, n - 1)]).collect();
        // A_gr e_{n-1} = (−sθ T, cθ T, w_2, …, w_gr), w_{gr−j} = (−1)^j S_j c_{j+1},
        // T = (−1)^{gr−1} S_{gr−1}, S_j = s_1⋯s_j.
        let mut c = vec![1.0; n.saturating_sub(2)]; // cos φ_{gr,1..gr−1}
        let mut s = vec![0.0; n.saturating_sub(2)];
        for j in 0..n.saturating_sub(2) {
            // component gr − j, normalised by the norm of components 0..=gr−j
            let idx = gr - j;
            let tail: f64 = (0..=idx).map(|i| v[i] * v[i]).sum::<f64>().sqrt();
            let sign = if j % 2 == 0 { 1.0 } else { -1.0 };
            let cj = if tail > 0.0 {
                (sign * v[idx] / tail).clamp(-1.0, 1.0)
            } else {
                1.0
            };
            c[j] = cj;
            s[j] = (1.0 - cj * cj).max(0.0).sqrt();
        }
        let sigma = if (gr - 1) % 2 == 0 { 1.0 } else { -1.0 };
        let theta = if v[0] == 0.0 && v[1] == 0.0 {
            0.0
        } else {
            (-sigma * v[0]).atan2(sigma * v[1])
        };
        // group norm from the eigenvalue ratio, then spherical → Cartesian
        let radius = (lambda[gr] / lambda[gr - 1] - 1.0).max(0.0);
        let off = gr * (gr + 1) / 2 - 1;
        let mut prod_s = 1.0;
        for j in 0..n.saturating_sub(2) {
            gamma[off + j] = radius * prod_s * c[j];
            prod_s *= s[j];
        }
        gamma[off + n - 2] = radius * prod_s * theta.cos();
        gamma[off + n - 1] = radius * prod_s * theta.sin();
        // strip the factor: m ← A_grᵀ m, with A_gr = Lo(θ) Π_{j=1}^{gr−1} P_j(φ_{gr, gr−j})
        let mut a = nalgebra::DMatrix::<f64>::identity(k, k);
        let plane = |mat: &mut nalgebra::DMatrix<f64>, p: usize, cc: f64, ss: f64| {
            // right-multiply by the rotation in plane (p, p+1)
            for row in 0..k {
                let (x, y) = (mat[(row, p)], mat[(row, p + 1)]);
                mat[(row, p)] = cc * x + ss * y;
                mat[(row, p + 1)] = cc * y - ss * x;
            }
        };
        plane(&mut a, 0, theta.cos(), theta.sin());
        for j in 1..gr {
            let jj = gr - j - 1; // φ_{gr, gr−j} (1-based) = c[gr − j − 1]
            plane(&mut a, j, c[jj], s[jj]);
        }
        m = a.transpose() * m;
    }
    gamma
}

/// `γ` for a target covariance shape: eigen-decompose the tangent-space
/// part `U_{−d}ᵀ V U_{−d}` of a symmetric positive-definite `v` (row-major
/// `d×d`; only its restriction to `μ⊥` is used and it is rescaled to
/// determinant 1), with `U = onb(mu)`. Eigenvector signs are chosen
/// canonically (largest-magnitude entry positive, then the first vector is
/// flipped if needed to make the rotation proper), so equivalent inputs map
/// to the same `γ`.
pub fn gamma_from_covariance(mu: &[f64], v: &[f64]) -> Vec<f64> {
    let d = mu.len();
    let k = d - 1;
    let u = crate::onb::onb(mu); // column-major
    let vm = nalgebra::DMatrix::from_row_slice(d, d, v);
    let uk = nalgebra::DMatrix::from_fn(d, k, |r, c| u[c * d + r]);
    let t = uk.transpose() * vm * &uk;
    let t = (&t + t.transpose()) * 0.5;
    let eig = nalgebra::SymmetricEigen::new(t);
    let mut order: Vec<usize> = (0..k).collect();
    order.sort_by(|&a, &b| eig.eigenvalues[a].partial_cmp(&eig.eigenvalues[b]).unwrap());
    let floor = 1e-12
        * eig
            .eigenvalues
            .iter()
            .fold(0.0f64, |m, &x| m.max(x.abs()))
            .max(1e-300);
    let mut lam: Vec<f64> = order
        .iter()
        .map(|&i| eig.eigenvalues[i].max(floor))
        .collect();
    let log_mean = lam.iter().map(|l| l.ln()).sum::<f64>() / k as f64;
    for l in lam.iter_mut() {
        *l = (l.ln() - log_mean).exp();
    }
    let mut q = nalgebra::DMatrix::from_fn(k, k, |r, c| eig.eigenvectors[(r, order[c])]);
    for c in 0..k {
        let (mut best, mut val) = (0usize, 0.0f64);
        for r in 0..k {
            if q[(r, c)].abs() > val.abs() + 1e-14 {
                best = r;
                val = q[(r, c)];
            }
        }
        if q[(best, c)] < 0.0 {
            for r in 0..k {
                q[(r, c)] = -q[(r, c)];
            }
        }
    }
    if q.determinant() < 0.0 {
        for r in 0..k {
            q[(r, 0)] = -q[(r, 0)];
        }
    }
    let rm: Vec<f64> = (0..k)
        .flat_map(|r| (0..k).map(move |c| (r, c)))
        .map(|(r, c)| q[(r, c)])
        .collect();
    angles_to_gamma(&lam, &rm)
}

#[cfg(test)]
mod inverse_tests {
    use super::*;
    use crate::rotation::rotation_matrix;

    fn rng(seed: u64) -> impl FnMut() -> f64 {
        let mut s = seed;
        move || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 11) as f64 / (1u64 << 53) as f64) * 4.0 - 2.0
        }
    }

    #[test]
    fn gamma_roundtrip_through_angles() {
        let mut next = rng(42);
        for d in 3..=8 {
            for _ in 0..50 {
                let g: Vec<f64> = (0..dims::gamma_len(d)).map(|_| next()).collect();
                let a = gamma_to_angles(&g);
                let r = rotation_matrix(&a.theta, &a.phi);
                let back = angles_to_gamma(&a.lambda, &r);
                for (x, y) in g.iter().zip(&back) {
                    assert!(
                        (x - y).abs() < 1e-9 * (1.0 + x.abs()),
                        "d={d}: {g:?} vs {back:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn covariance_roundtrip_and_canonical_equivalence() {
        let mut next = rng(7);
        for d in 3..=7 {
            for _ in 0..30 {
                let mu: Vec<f64> = (0..d).map(|_| 2.0 * next()).collect();
                let g: Vec<f64> = (0..dims::gamma_len(d)).map(|_| next()).collect();
                let v = crate::frame::Frame::new(&mu, &g).cov_pow(1.0);
                let g2 = gamma_from_covariance(&mu, &v);
                let v2 = crate::frame::Frame::new(&mu, &g2).cov_pow(1.0);
                for (a, b) in v.iter().zip(&v2) {
                    assert!((a - b).abs() < 1e-8 * (1.0 + a.abs()), "d={d}");
                }
                // the canonical representative is idempotent
                let g3 = gamma_from_covariance(&mu, &v2);
                for (a, b) in g2.iter().zip(&g3) {
                    assert!(
                        (a - b).abs() < 1e-8 * (1.0 + a.abs()),
                        "d={d}: {g2:?} vs {g3:?}"
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod singular_set_tests {
    use crate::distribution::Esag;

    /// Largest change of `V` over four directions of approach when the
    /// entries `idx` of `gamma` shrink to `eps` times a unit-ish direction.
    fn jump(mu: &[f64], base: &[f64], idx: std::ops::Range<usize>, eps: f64) -> f64 {
        let v_at = |s: usize| {
            let mut g = base.to_vec();
            for (c, k) in idx.clone().enumerate() {
                g[k] = eps * (((s * 7 + c * 3) as f64) * 1.3).sin();
            }
            Esag::new(mu, &g).unwrap().covariance()
        };
        let v0 = v_at(0);
        (1..4)
            .map(|s| (v_at(s) - &v0).abs().max())
            .fold(0.0, f64::max)
    }

    /// The map `gamma → V` (identical to the R code's `parameter`/`rotation`)
    /// is discontinuous wherever an angle is undefined while it still moves
    /// unequal eigen-directions: where the last two entries of a group
    /// `j ≥ 2` vanish, and where a whole group `j ≥ 2` vanishes while other
    /// groups do not. The limit of `V` then depends on the direction of
    /// approach, however close. Group 1 and `gamma = 0` as a whole are
    /// continuous. (This traps optimisers in regression fits with several
    /// covariates, and puts exact zeros of group `j ≥ 2` on a jump; see
    /// `docs/MATH.md`.)
    #[test]
    fn gamma_map_discontinuities() {
        let mu = [1.0, 2.0, -1.0, 0.5, 1.5];
        let base = [0.4, -0.7, 1.2, 0.6, -0.3, 0.3, 0.2, 0.5, -0.4];
        for eps in [1e-3, 1e-6, 1e-9] {
            // group 1 → 0: continuous
            assert!(jump(&mu, &base, 0..2, eps) < 1.0 * eps);
            // whole group 2 or 3 → 0: a jump of fixed size
            assert!(jump(&mu, &base, 2..5, eps) > 0.2);
            assert!(jump(&mu, &base, 5..9, eps) > 0.5);
            // trailing pair of group 2 → 0 (group still nonzero): a jump
            assert!(jump(&mu, &base, 3..5, eps) > 0.1);
            // all of gamma → 0: continuous (V → I)
            assert!(jump(&mu, &base, 0..9, eps) < 10.0 * eps);
        }
    }
}
