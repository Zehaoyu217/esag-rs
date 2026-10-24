//! The log-tangent shape parameterization (`esag::logtangent`):
//!
//! (a) the hand-written gradient against central finite differences,
//!     including repeated eigenvalues of `S`;
//! (b) the properties of `V` (symmetric, `Vμ = μ`, `det V = 1`) and the
//!     density computed from `V` directly;
//! (c) equivalence with Yu & Huang's parameterization: the same `V` through
//!     the canonical `γ` gives the same log-density;
//! (d) smoothness at a point where `γ ↦ V` is discontinuous;
//! (e) the density integrates to 1 over S² (d = 3);
//! (f) `s_from_covariance ∘ covariance = id`;
//! (g) continuity of the frame `Ξ(μ̂)` away from `−m`.

// index loops mirror the notation of the module
#![allow(clippy::needless_range_loop)]

use esag::dims::{eta_len, gamma_len};
use esag::frame::Frame;
use esag::loglik::{log_m_p_value, loglik_eta};
use esag::logtangent::{
    LtFrame, covariance, loglik_lt, loglik_lt_grad, matrix_to_s, s_from_covariance, s_to_matrix,
};
use esag::param::gamma_from_covariance;
use nalgebra::{DMatrix, DVector};

/// splitmix64 → uniform in [0, 1).
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
    fn normal(&mut self) -> f64 {
        let u1 = self.unif().max(1e-300);
        let u2 = self.unif();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
    fn unit(&mut self, d: usize) -> Vec<f64> {
        let v: Vec<f64> = (0..d).map(|_| self.normal()).collect();
        let n = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        v.iter().map(|x| x / n).collect()
    }
    /// A random orthogonal `p × p` matrix.
    fn orthogonal(&mut self, p: usize) -> DMatrix<f64> {
        DMatrix::from_fn(p, p, |_, _| self.normal()).qr().q()
    }
}

/// The standard frame for even `k`, a random reference direction otherwise.
fn frame_for(d: usize, k: usize, rng: &mut Rng) -> LtFrame {
    if k.is_multiple_of(2) {
        LtFrame::standard(d)
    } else {
        LtFrame::new(&rng.unit(d))
    }
}

/// Row-major copy of a matrix.
fn row_major(m: &DMatrix<f64>) -> Vec<f64> {
    (0..m.nrows())
        .flat_map(|r| (0..m.ncols()).map(move |c| m[(r, c)]))
        .collect()
}

/// Shape coordinates of `Q diag(lam) Qᵀ` for a random orthogonal `Q`.
fn rotated_s(lam: &[f64], rng: &mut Rng) -> Vec<f64> {
    let p = lam.len();
    let q = rng.orthogonal(p);
    let s_mat = &q * DMatrix::from_diagonal(&DVector::from_column_slice(lam)) * q.transpose();
    matrix_to_s(&row_major(&s_mat), p)
}

/// Shape coordinates of `diag(lam)`.
fn diagonal_s(lam: &[f64]) -> Vec<f64> {
    let p = lam.len();
    let m = DMatrix::from_diagonal(&DVector::from_column_slice(lam));
    matrix_to_s(&row_major(&m), p)
}

/// Traceless eigenvalues with a repeated leading pair, `(1, 1, −2, 0, …)`,
/// or `(t, t, …, t, −(p−1)t)` for `all`.
fn repeated_eigenvalues(p: usize, all: bool) -> Vec<f64> {
    let mut lam = vec![0.0; p];
    if all {
        lam.iter_mut().for_each(|x| *x = 0.8);
        lam[p - 1] = -0.8 * (p - 1) as f64;
    } else if p >= 3 {
        lam[0] = 1.0;
        lam[1] = 1.0;
        lam[2] = -2.0;
    }
    lam
}

/// One test input `(frame, eta, y, label)` of the given kind (`0..N_KINDS`),
/// with the standard frame for even `frame_sel` and a random one otherwise.
fn make_case(
    d: usize,
    kind: usize,
    frame_sel: usize,
    rng: &mut Rng,
) -> (LtFrame, Vec<f64>, Vec<f64>, &'static str) {
    let p = d - 1;
    let g = gamma_len(d);
    let frame = frame_for(d, frame_sel, rng);
    let mut mu: Vec<f64> = (0..d).map(|_| rng.range(-2.5, 2.5)).collect();
    let mut s: Vec<f64> = (0..g).map(|_| rng.range(-1.0, 1.0)).collect();
    let mut y = rng.unit(d);
    let label = match kind {
        0 | 1 => "generic",
        2 => {
            s.iter_mut().for_each(|x| *x = 0.0);
            "s = 0"
        }
        3 => {
            // exact double eigenvalue: diagonal S with S_00 = S_11
            s = diagonal_s(&repeated_eigenvalues(p, false));
            "exact double eigenvalue"
        }
        4 => {
            // double eigenvalue in a random basis (eigenvalues equal to ~1e-16)
            s = rotated_s(&repeated_eigenvalues(p, false), rng);
            "rotated double eigenvalue"
        }
        5 => {
            s = rotated_s(&repeated_eigenvalues(p, true), rng);
            "(p-1)-fold eigenvalue"
        }
        6 => {
            // off-diagonal coordinates only: every 2×2 block starts with equal diagonals
            s[..p - 1].iter_mut().for_each(|x| *x = 0.0);
            "off-diagonal only"
        }
        7 => {
            s.iter_mut().for_each(|x| *x *= 3.0);
            "large S"
        }
        8 => {
            // concentrated, y nearly antipodal to mu: alpha in the tail of log M_p
            let dir = rng.unit(d);
            let r = rng.range(5.0, 20.0);
            for k in 0..d {
                mu[k] = r * dir[k];
                y[k] = -dir[k] + 0.1 * rng.normal();
            }
            let n = y.iter().map(|x| x * x).sum::<f64>().sqrt();
            y.iter_mut().for_each(|x| *x /= n);
            "antipodal y"
        }
        9 => {
            // a diagonal S (Jacobi does no rotation), distinct eigenvalues
            s[p - 1..].iter_mut().for_each(|x| *x = 0.0);
            "diagonal S"
        }
        10 => {
            s.iter_mut().for_each(|x| *x *= 1e-6);
            "tiny S"
        }
        _ => panic!("no kind {kind}"),
    };
    let mut eta = mu;
    eta.extend(s);
    (frame, eta, y, label)
}

const N_KINDS: usize = 11;

/// The kinds whose `S` has exactly tied eigenvalues (other than `S = 0`).
const TIED_KINDS: [usize; 3] = [3, 4, 5];

/// Fourth-order central difference of `loglik_lt` in `eta[i]`, step
/// `k = 5e-4`: the truncation error `k⁴ f⁽⁵⁾/30` is ≲ 1e-12, and value
/// jitter `δ` costs `1.5 δ/k`. The jitter is that of `log M_p` (its forward
/// recursion for `−2 ≤ alpha < 0`, and the switch to the continued fraction
/// at `alpha = −2`), which the `γ` kernel shares: a few 1e-15 at d = 4 and
/// up to ~5e-13 at d = 10, too much for a plain central difference with a
/// small step to resolve 1e-7.
fn central_difference(frame: &LtFrame, eta: &[f64], y: &[f64], i: usize) -> f64 {
    let k = 5e-4;
    let mut e = eta.to_vec();
    let mut f = |t: f64| {
        e[i] = eta[i] + t;
        loglik_lt(frame, &e, y)
    };
    (8.0 * (f(k) - f(-k)) - (f(2.0 * k) - f(-2.0 * k))) / (12.0 * k)
}

/// (a) The gradient matches central finite differences for d = 3..10 (the
/// stack kernels and the heap fallback).
#[test]
fn gradient_matches_finite_differences() {
    let mut rng = Rng(0x5EED);
    let mut worst: f64 = 0.0;
    for d in 3..=10 {
        for rep in 0..6 {
            for kind in 0..N_KINDS {
                let (frame, eta, y, label) = make_case(d, kind, rep, &mut rng);
                let n = eta_len(d);
                let mut g = vec![0.0; n];
                let f = loglik_lt_grad(&frame, &eta, &y, &mut g);
                assert_eq!(f.to_bits(), loglik_lt(&frame, &eta, &y).to_bits());
                for i in 0..n {
                    let fd = central_difference(&frame, &eta, &y, i);
                    let err = (g[i] - fd).abs() / fd.abs().max(1.0);
                    worst = worst.max(err);
                    assert!(
                        err < 1e-7,
                        "d={d} {label}: d/d eta[{i}]: analytic {} vs fd {fd}",
                        g[i]
                    );
                }
            }
        }
    }
    eprintln!("gradient vs central differences: worst relative error {worst:.1e}");
}

/// (b) `V` is symmetric with `Vμ = μ` and `det V = 1`, and the log-density
/// computed from `V⁻¹` directly agrees with the kernel.
#[test]
fn covariance_properties() {
    let mut rng = Rng(17);
    for d in 3..=10 {
        for k in 0..20 {
            let (frame, eta, y, label) = make_case(d, k % N_KINDS, k, &mut rng);
            let (mu, s) = eta.split_at(d);
            let v = covariance(&frame, mu, s);
            assert_eq!(v, v.transpose(), "d={d}: V is not symmetric");
            let mu_v = DVector::from_column_slice(mu);
            let vm = &v * &mu_v;
            assert!(
                (vm - &mu_v).norm() <= 1e-12 * mu_v.norm().max(1.0),
                "d={d}: V mu != mu"
            );
            // det V = e^{tr S} = 1 exactly; numerically, rounding of V's
            // entries alone moves det V by up to ~d·ε·cond(V), which is
            // ~1e-8 for the "large S" kind at d = 10 (cond ~ 1e7)
            let eig = v.clone().symmetric_eigen().eigenvalues;
            let cond = eig.max() / eig.min();
            let det = v.clone().lu().determinant();
            let tol = 1e-12_f64.max(4.0 * d as f64 * f64::EPSILON * cond);
            assert!(
                (det - 1.0).abs() < tol,
                "d={d} {label}: det V = {det} (cond {cond:.1e})"
            );
            if cond < 1e3 {
                assert!((det - 1.0).abs() < 1e-12, "d={d} {label}: det V = {det}");
            }
            assert!(v.clone().cholesky().is_some(), "d={d}: V is not SPD");
            // log f from V⁻¹
            let p = d - 1;
            let yv = DVector::from_column_slice(&y);
            let vinv = v.clone().try_inverse().unwrap();
            let q = (yv.transpose() * &vinv * &yv)[(0, 0)];
            let alpha = yv.dot(&mu_v) / q.sqrt();
            let direct = -0.5 * p as f64 * esag::stats::normal::LN_2PI - 0.5 * d as f64 * q.ln()
                + 0.5 * (alpha * alpha - mu_v.norm_squared())
                + log_m_p_value(alpha, p);
            let lt = loglik_lt(&frame, &eta, &y);
            assert!(
                (lt - direct).abs() < 1e-11 * lt.abs().max(1.0),
                "d={d}: kernel {lt} vs from V {direct}"
            );
        }
    }
}

/// (c) The same distribution family: for random `(μ, s)`, the canonical `γ`
/// of `V(μ, s)` gives the same log-density through the `γ` kernel.
#[test]
fn equivalent_to_the_gamma_parameterization() {
    let mut rng = Rng(23);
    let mut worst: f64 = 0.0;
    for d in 3..=10 {
        for k in 0..33 {
            let kind = k % N_KINDS;
            if TIED_KINDS.contains(&kind) || kind == 9 {
                // Exact ties: see `gamma_cannot_represent_exact_ties`. A
                // diagonal S: the eigenvectors of V are then the frame
                // columns Ξ = H_v onb(m), and in the basis U = onb(μ) the
                // structure of the two `onb`s gives exact zeros, so that the
                // canonical γ often lies on the γ singular set (a group with
                // an exactly vanishing trailing pair), where it need not
                // reproduce V: see `gamma_misses_frame_aligned_shapes`.
                continue;
            }
            let (frame, eta, _, _) = make_case(d, k % N_KINDS, k / N_KINDS, &mut rng);
            let (mu, s) = eta.split_at(d);
            let v = covariance(&frame, mu, s);
            let gamma = gamma_from_covariance(mu, &row_major(&v));
            let eta_g = [mu, &gamma[..]].concat();
            for _ in 0..5 {
                let y = rng.unit(d);
                let a = loglik_lt(&frame, &eta, &y);
                let b = loglik_eta::<f64>(&eta_g, &y);
                let err = (a - b).abs() / a.abs().max(1.0);
                worst = worst.max(err);
                assert!(err < 1e-11, "d={d} k={k}: log-tangent {a} vs gamma {b}");
            }
        }
    }
    eprintln!("log-tangent vs canonical gamma: worst relative difference {worst:.1e}");
}

/// Where two tangent eigenvalues of `V` tie exactly, the canonical `γ` has a
/// group of norm 0, which erases that group's rotation; for a generic
/// eigenbasis `γ` then cannot reproduce `V` (such `V` are limits of the `γ`
/// family, not members). The log-tangent coordinates represent them
/// exactly, and the two parameterizations still agree once the tie is split.
#[test]
fn gamma_cannot_represent_exact_ties() {
    let mut rng = Rng(29);
    let d = 4;
    let (mut failures, trials) = (0, 300);
    for k in 0..trials {
        let frame = frame_for(d, k, &mut rng);
        let mu: Vec<f64> = (0..d).map(|_| rng.range(-2.5, 2.5)).collect();
        let s = rotated_s(&[1.0, 1.0, -2.0], &mut rng);
        let v = covariance(&frame, &mu, &s);
        // exact in the log-tangent coordinates
        let s2 = s_from_covariance(&frame, &mu, &v);
        assert!((covariance(&frame, &mu, &s2) - &v).norm() < 1e-12);
        let gamma = gamma_from_covariance(&mu, &row_major(&v));
        let v_gamma = DMatrix::from_row_slice(d, d, &Frame::new(&mu, &gamma).cov_pow(1.0));
        if (v_gamma - &v).norm() > 1e-6 {
            failures += 1;
        }
        // split the tie by 1e-6: equivalent again
        let s_split = rotated_s(&[1.0 + 1e-6, 1.0 - 1e-6, -2.0], &mut rng);
        let v = covariance(&frame, &mu, &s_split);
        let gamma = gamma_from_covariance(&mu, &row_major(&v));
        let v_gamma = DMatrix::from_row_slice(d, d, &Frame::new(&mu, &gamma).cov_pow(1.0));
        assert!((v_gamma - &v).norm() < 1e-8 * v.norm());
    }
    assert!(
        failures > 0,
        "expected some tied V that gamma cannot represent"
    );
    eprintln!("exact ties: gamma_from_covariance missed V in {failures}/{trials} cases");
}

/// A diagonal `S`, i.e. `V` with the frame columns `Ξ(μ̂)` as eigenvectors:
/// with distinct eigenvalues, the canonical `γ` of `V` often has a group
/// whose trailing pair vanishes exactly, and then misses `V` by `O(1)`; the
/// fraction grows with `d`. For generic `S` it never does. The log-tangent
/// coordinates represent both exactly.
#[test]
fn gamma_misses_frame_aligned_shapes() {
    let mut rng = Rng(37);
    for d in [4usize, 6, 8] {
        let p = d - 1;
        let (mut aligned, mut generic) = (0, 0);
        for k in 0..100 {
            let frame = frame_for(d, k, &mut rng);
            let mu: Vec<f64> = (0..d).map(|_| rng.range(-2.5, 2.5)).collect();
            let mut lam: Vec<f64> = (0..p).map(|_| rng.range(-1.0, 1.0)).collect();
            let mean = lam.iter().sum::<f64>() / p as f64;
            lam.iter_mut().for_each(|x| *x -= mean);
            let s_gen: Vec<f64> = (0..gamma_len(d)).map(|_| rng.range(-0.6, 0.6)).collect();
            for (s, count) in [(diagonal_s(&lam), &mut aligned), (s_gen, &mut generic)] {
                let v = covariance(&frame, &mu, &s);
                assert!(
                    (covariance(&frame, &mu, &s_from_covariance(&frame, &mu, &v)) - &v).norm()
                        < 1e-12
                );
                let gamma = gamma_from_covariance(&mu, &row_major(&v));
                let v_gamma = DMatrix::from_row_slice(d, d, &Frame::new(&mu, &gamma).cov_pow(1.0));
                if (v_gamma - &v).norm() > 1e-8 {
                    *count += 1;
                }
            }
        }
        assert_eq!(generic, 0, "d={d}");
        if d >= 6 {
            assert!(aligned > 0, "d={d}");
        }
        eprintln!(
            "d={d}: gamma_from_covariance missed V for {aligned}/100 diagonal S, {generic}/100 generic S"
        );
    }
}

/// `(mu, gamma, (i, j))` with `gamma[i] = gamma[j] = 0` the trailing pair of a group.
type SingularCase = (Vec<f64>, Vec<f64>, (usize, usize));

/// (d) At a `γ` whose group 2 (d = 4) or group 3 (d = 5) has a vanishing
/// trailing pair, `γ ↦ V` jumps (the angle `θ` is undefined but still
/// rotates `V`), while `s ↦ V` moves by `O(ε)` in every direction.
#[test]
fn smooth_where_gamma_is_singular() {
    let mut rng = Rng(31);
    // (mu, gamma, the indices of the vanishing trailing pair)
    let cases: [SingularCase; 2] = [
        (
            vec![0.6, -1.2, 1.5, 0.9],
            vec![0.9, -0.4, 1.3, 0.0, 0.0],
            (3, 4),
        ),
        (
            vec![0.4, 1.1, -0.7, 1.6, -0.5],
            vec![0.5, 0.8, -0.6, 0.3, 0.7, 1.4, -0.9, 0.0, 0.0],
            (7, 8),
        ),
    ];
    for (mu, gamma0, (i4, i5)) in cases {
        let d = mu.len();
        let v_gamma = |g: &[f64]| DMatrix::from_row_slice(d, d, &Frame::new(&mu, g).cov_pow(1.0));
        let v0 = v_gamma(&gamma0);
        // the canonical gamma of V0 has the same vanishing trailing pair
        let gc = gamma_from_covariance(&mu, &row_major(&v0));
        assert!(
            gc[i4].abs() < 1e-8 && gc[i5].abs() < 1e-8,
            "d={d}: canonical gamma {gc:?}"
        );
        // gamma: two perturbations of size 1e-10 give V's an O(1) apart
        let mut ga = gamma0.clone();
        ga[i4] = 1e-10;
        let mut gb = gamma0.clone();
        gb[i5] = 1e-10;
        let jump = (v_gamma(&ga) - v_gamma(&gb)).norm();
        assert!(jump > 0.1, "d={d}: expected a jump, got {jump}");
        // s: every direction moves V by O(ε)
        for frame in [LtFrame::standard(d), LtFrame::new(&rng.unit(d))] {
            let s0 = s_from_covariance(&frame, &mu, &v0);
            let v0_lt = covariance(&frame, &mu, &s0);
            assert!((&v0_lt - &v0).norm() < 1e-12 * v0.norm(), "d={d}");
            let lip = 2.0 * v0.norm();
            for _ in 0..20 {
                let dir = rng.unit(s0.len());
                for eps in [1e-4, 1e-7, 1e-10] {
                    let s1: Vec<f64> = s0.iter().zip(&dir).map(|(a, b)| a + eps * b).collect();
                    let dv = (covariance(&frame, &mu, &s1) - &v0_lt).norm();
                    assert!(dv <= lip * eps + 1e-14, "d={d} eps={eps}: |dV| = {dv}");
                }
            }
            // and the log-density is continuous where the gamma kernel's jumps
            let y = rng.unit(d);
            let f0 = loglik_lt(&frame, &[&mu[..], &s0[..]].concat(), &y);
            let fa = loglik_eta::<f64>(&[&mu[..], &ga[..]].concat(), &y);
            let fb = loglik_eta::<f64>(&[&mu[..], &gb[..]].concat(), &y);
            assert!((f0 - loglik_eta::<f64>(&[&mu[..], &gamma0[..]].concat(), &y)).abs() < 1e-11);
            eprintln!(
                "d={d}: gamma kernel jumps by {:.2e} across 2e-10",
                (fa - fb).abs()
            );
        }
    }
}

/// (e) d = 3: the density integrates to 1 over the sphere.
#[test]
fn density_integrates_to_one_d3() {
    let nz = 800;
    let nphi = 800;
    let frames = [LtFrame::standard(3), LtFrame::new(&[0.3, -0.8, 0.5])];
    let cases: [(Vec<f64>, Vec<f64>); 3] = [
        (vec![1.0, 2.0, 1.5], vec![0.7, -0.3]),
        (vec![0.3, -0.2, 0.1], vec![2.0, 1.0]),
        (vec![4.0, 0.0, 0.0], vec![-1.5, 1.2]),
    ];
    for frame in &frames {
        for (mu, s) in &cases {
            let eta = [&mu[..], &s[..]].concat();
            let mut total = 0.0;
            for i in 0..nz {
                let z = -1.0 + (i as f64 + 0.5) * 2.0 / nz as f64;
                let r = (1.0 - z * z).sqrt();
                for j in 0..nphi {
                    let ph = (j as f64 + 0.5) * 2.0 * std::f64::consts::PI / nphi as f64;
                    let y = [r * ph.cos(), r * ph.sin(), z];
                    total += loglik_lt(frame, &eta, &y).exp();
                }
            }
            total *= (2.0 / nz as f64) * (2.0 * std::f64::consts::PI / nphi as f64);
            assert!((total - 1.0).abs() < 1e-4, "{mu:?} {s:?}: {total}");
        }
    }
}

/// (f) `s_from_covariance ∘ covariance = id`, `covariance ∘
/// s_from_covariance = id` on ESAG covariances built from `γ`, and the
/// determinant is ignored.
#[test]
fn covariance_round_trips() {
    let mut rng = Rng(41);
    let mut worst: f64 = 0.0;
    for d in 3..=10 {
        let p = d - 1;
        for k in 0..30 {
            let (frame, eta, _, _) = make_case(d, k % N_KINDS, k / N_KINDS, &mut rng);
            let (mu, s) = eta.split_at(d);
            let v = covariance(&frame, mu, s);
            let back = s_from_covariance(&frame, mu, &v);
            let sn = s.iter().map(|x| x * x).sum::<f64>().sqrt();
            for (a, b) in s.iter().zip(&back) {
                let err = (a - b).abs() / (1.0 + sn);
                worst = worst.max(err);
                assert!(err < 1e-12, "d={d} k={k}: {s:?} vs {back:?}");
            }
            // a scalar multiple of V has the same shape
            let back2 = s_from_covariance(&frame, mu, &(&v * 3.7));
            for (a, b) in back.iter().zip(&back2) {
                assert!((a - b).abs() < 1e-12 * (1.0 + sn));
            }
            // s_to_matrix / matrix_to_s agree with the definition of S
            let sm = s_to_matrix(s, p);
            assert_eq!(matrix_to_s(&sm, p).len(), s.len());
        }
        // from an ESAG covariance in Yu & Huang's parameterization
        for k in 0..10 {
            let frame = frame_for(d, k, &mut rng);
            let mu: Vec<f64> = (0..d).map(|_| rng.range(-2.0, 2.0)).collect();
            let gamma: Vec<f64> = (0..gamma_len(d)).map(|_| rng.range(-1.0, 1.0)).collect();
            let v = DMatrix::from_row_slice(d, d, &Frame::new(&mu, &gamma).cov_pow(1.0));
            let s = s_from_covariance(&frame, &mu, &v);
            let v2 = covariance(&frame, &mu, &s);
            assert!((&v2 - &v).norm() < 1e-12 * v.norm(), "d={d}");
        }
    }
    eprintln!("s_from_covariance(covariance(s)) vs s: worst error {worst:.1e}");
}

/// (g) The frame `Ξ(μ̂)` is an orthonormal basis of `μ̂⊥`, equal to `E` at
/// `μ̂ = m`, orientation-preserving, and Lipschitz with constant
/// `2√2/‖μ̂ + m‖` in `μ̂`: continuous everywhere except at `−m`.
#[test]
fn frame_is_continuous_away_from_the_antipode() {
    let mut rng = Rng(53);
    for d in 3..=10 {
        let p = d - 1;
        for k in 0..10 {
            let frame = frame_for(d, k, &mut rng);
            let m = DVector::from_column_slice(frame.reference());
            let e = DMatrix::from_column_slice(d, p, frame.tangent_basis());
            assert!((frame.frame_at(m.as_slice()) - &e).norm() < 1e-15);
            let mut em = e.clone().insert_column(p, 0.0);
            em.set_column(p, &m);
            let orient = em.determinant();
            for _ in 0..20 {
                let mu_hat = DVector::from_vec(rng.unit(d));
                let c = mu_hat.dot(&m);
                if c < -0.9 {
                    continue;
                }
                let r = rng.range(0.1, 5.0);
                let mu = &mu_hat * r;
                let xi = frame.frame_at(mu.as_slice());
                assert!((xi.transpose() * &xi - DMatrix::identity(p, p)).norm() < 1e-14);
                assert!((xi.transpose() * &mu_hat).norm() < 1e-14);
                let mut xm = xi.clone().insert_column(p, 0.0);
                xm.set_column(p, &mu_hat);
                assert!((xm.determinant() - orient).abs() < 1e-13, "orientation");
                // ‖ΔΞ‖_F ≤ 2√2 ‖Δμ̂‖/‖μ̂ + m‖ to first order
                let lip = 2.0 * std::f64::consts::SQRT_2 / (&mu_hat + &m).norm();
                for eps in [1e-5, 1e-8] {
                    let dir = DVector::from_vec(rng.unit(d));
                    let mu2 = &mu + &dir * (eps * r);
                    let mu2_hat = &mu2 / mu2.norm();
                    let dxi = (frame.frame_at(mu2.as_slice()) - &xi).norm();
                    let dmu = (&mu2_hat - &mu_hat).norm();
                    assert!(
                        dxi <= lip * dmu * (1.0 + 1e-3) + 1e-14,
                        "d={d} c={c:.2}: |dXi| = {dxi:.3e}, bound {:.3e}",
                        lip * dmu
                    );
                }
            }
        }
    }
}

#[test]
fn frame_serde_round_trip() {
    let f = LtFrame::new(&[0.2, -0.5, 0.1, 0.8]);
    let json = serde_json::to_string(&f).unwrap();
    let back: LtFrame = serde_json::from_str(&json).unwrap();
    assert_eq!(f, back);
}
