//! `loglik_eta_hess` against the exact gradient `loglik_eta_grad`, evaluated
//! at run time (never against stored numbers, so the log-density internals
//! may change freely).

use esag::dims::{eta_len, gamma_len};
use esag::hdual::{HDual, loglik_eta_hess, seed_hdual};
use esag::loglik::{loglik_eta, loglik_eta_grad};

/// SplitMix64 with Box–Muller normals: deterministic and dependency-free.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn uniform(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    fn normal(&mut self) -> f64 {
        let (u, v) = (self.uniform(), self.uniform());
        (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
    }
    fn normals(&mut self, n: usize, scale: f64) -> Vec<f64> {
        (0..n).map(|_| scale * self.normal()).collect()
    }
}

fn normalise(v: &mut [f64]) {
    let n = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    v.iter_mut().for_each(|x| *x /= n);
}

/// One test point; `fine` marks coordinates that need a tiny difference step
/// (members of a nearly-zero gamma group).
struct Case {
    label: String,
    eta: Vec<f64>,
    y: Vec<f64>,
    fine: Vec<bool>,
}

/// Offset of gamma group `i` (0-based; group `i` has `i + 2` entries).
fn group_range(d: usize, i: usize) -> std::ops::Range<usize> {
    let start = d + (i + 1) * (i + 2) / 2 - 1;
    start..start + i + 2
}

fn cases(d: usize, rng: &mut Rng) -> Vec<Case> {
    let n = eta_len(d);
    let mut out = Vec::new();
    // generic random points
    for k in 0..4 {
        let mut eta = rng.normals(d, 1.5);
        eta.extend(rng.normals(gamma_len(d), 0.7));
        let mut y = rng.normals(d, 1.0);
        normalise(&mut y);
        out.push(Case {
            label: format!("random #{k}"),
            eta,
            y,
            fine: vec![false; n],
        });
    }
    // y nearly antipodal to mu: alpha ≈ −|mu| (q = 1 at y = −mu_hat)
    for &m in &[3.5, 6.0, 10.0] {
        let mut dir = rng.normals(d, 1.0);
        normalise(&mut dir);
        let mut eta: Vec<f64> = dir.iter().map(|v| m * v).collect();
        eta.extend(rng.normals(gamma_len(d), 0.6));
        let mut y: Vec<f64> = dir.iter().map(|v| -v + 0.05 * rng.normal()).collect();
        normalise(&mut y);
        out.push(Case {
            label: format!("antipodal |mu| = {m}"),
            eta,
            y,
            fine: vec![false; n],
        });
    }
    // one gamma group nearly zero (|G| ≈ 1e-3), the others generic
    for g in 0..d - 2 {
        let mut eta = rng.normals(d, 1.2);
        eta.extend(rng.normals(gamma_len(d), 0.8));
        let mut fine = vec![false; n];
        let r = group_range(d, g);
        let mut grp = rng.normals(r.len(), 1.0);
        normalise(&mut grp);
        for (k, idx) in r.clone().enumerate() {
            eta[idx] = 1e-3 * grp[k];
            fine[idx] = true;
        }
        let mut y = rng.normals(d, 1.0);
        normalise(&mut y);
        out.push(Case {
            label: format!("gamma group {g} ≈ 0"),
            eta,
            y,
            fine,
        });
    }
    out
}

/// Column `i` of the Hessian by Richardson-extrapolated central differences of
/// the exact gradient.
fn fd_column(eta: &[f64], y: &[f64], i: usize, h: f64) -> Vec<f64> {
    let n = eta.len();
    let cd = |h: f64| {
        let (mut ep, mut em) = (eta.to_vec(), eta.to_vec());
        ep[i] += h;
        em[i] -= h;
        let (mut gp, mut gm) = (vec![0.0; n], vec![0.0; n]);
        loglik_eta_grad(&ep, y, &mut gp);
        loglik_eta_grad(&em, y, &mut gm);
        let step = ep[i] - em[i];
        gp.iter()
            .zip(&gm)
            .map(|(a, b)| (a - b) / step)
            .collect::<Vec<_>>()
    };
    let (d1, d2) = (cd(h), cd(0.5 * h));
    d1.iter()
        .zip(&d2)
        .map(|(a, b)| (4.0 * b - a) / 3.0)
        .collect()
}

#[test]
fn hdual_hessian_matches_exact_gradient_d3_to_d6() {
    let mut rng = Rng(0x5EED_E5A6);
    let mut worst = 0.0f64;
    for d in 3..=6 {
        let n = eta_len(d);
        for c in cases(d, &mut rng) {
            let (eta, y) = (&c.eta, &c.y);
            let mut g = vec![0.0; n];
            let mut h = vec![0.0; n * n];
            let v = loglik_eta_hess(eta, y, &mut g, &mut h);
            let mut g0 = vec![0.0; n];
            let v0 = loglik_eta_grad(eta, y, &mut g0);
            let tag = format!("d={d} {}", c.label);
            assert!(v.is_finite(), "{tag}: value {v}");
            assert!(
                (v - v0).abs() <= 1e-12 * v0.abs().max(1.0),
                "{tag}: value {v} vs {v0}"
            );
            let v1: f64 = loglik_eta(eta, y);
            assert!(
                (v - v1).abs() <= 1e-12 * v1.abs().max(1.0),
                "{tag}: f64 value"
            );
            let gscale = g0.iter().fold(1.0f64, |m, x| m.max(x.abs()));
            for i in 0..n {
                assert!(
                    (g[i] - g0[i]).abs() <= 1e-12 * g0[i].abs().max(gscale),
                    "{tag}: grad[{i}] {} vs {}",
                    g[i],
                    g0[i]
                );
            }
            // symmetry (exact)
            for i in 0..n {
                for j in 0..n {
                    assert_eq!(h[i * n + j], h[j * n + i], "{tag}: asymmetric ({i},{j})");
                }
            }
            // against differences of the exact gradient
            let mut fd = vec![0.0; n * n];
            for i in 0..n {
                let step = if c.fine[i] {
                    1e-6
                } else {
                    1e-4 * eta[i].abs().max(1.0)
                };
                for (j, v) in fd_column(eta, y, i, step).into_iter().enumerate() {
                    fd[j * n + i] = v;
                }
            }
            let hmax = fd.iter().fold(0.0f64, |m, x| m.max(x.abs()));
            for i in 0..n {
                for j in 0..n {
                    let (a, b) = (h[i * n + j], fd[i * n + j]);
                    let scale = b.abs().max(1e-3 * hmax).max(1e-8);
                    let rel = (a - b).abs() / scale;
                    worst = worst.max(rel);
                    assert!(
                        rel <= 1e-5,
                        "{tag}: H[{i}][{j}] = {a} vs fd {b} (rel {rel:.2e})"
                    );
                }
            }
        }
    }
    eprintln!("worst Hessian relative error vs Richardson FD: {worst:.2e}");
}

/// `M_p(a) = ∫_0^∞ t^p φ(t − a) dt` by composite Simpson.
fn m_p_quad(a: f64, p: usize) -> f64 {
    let hi = a.max(0.0) + 40.0;
    let n = 200_000;
    let h = hi / n as f64;
    let f = |t: f64| {
        t.powi(p as i32) * (-0.5 * (t - a) * (t - a)).exp() / (2.0 * std::f64::consts::PI).sqrt()
    };
    let mut s = f(0.0) + f(hi);
    for i in 1..n {
        s += if i % 2 == 1 { 4.0 } else { 2.0 } * f(i as f64 * h);
    }
    s * h / 3.0
}

#[test]
fn hdual_test_tail_points_are_in_the_tail() {
    // Sanity check of the test construction: at y = −mu_hat, q = 1 and
    // alpha = −|mu|, so log f = −(p/2) log 2π + log M_p(−|mu|). The antipodal
    // cases above therefore sit at alpha ≈ −3.5, −6, −10 (tail branch).
    for d in 3..=6 {
        for &m in &[3.5, 6.0, 10.0] {
            let mut eta = vec![0.0; eta_len(d)];
            eta[0] = m;
            for (k, v) in eta[d..].iter_mut().enumerate() {
                *v = 0.3 + 0.1 * k as f64;
            }
            let mut y = vec![0.0; d];
            y[0] = -1.0;
            let f: f64 = loglik_eta(&eta, &y);
            let p = d - 1;
            let expect = -0.5 * p as f64 * (2.0 * std::f64::consts::PI).ln() + m_p_quad(-m, p).ln();
            assert!(
                (f - expect).abs() < 1e-8 * expect.abs(),
                "d={d} m={m}: {f} vs {expect}"
            );
        }
    }
}

#[test]
fn hdual_hessian_symmetric_at_isotropic_point() {
    // gamma = 0 exactly (sqrt and atan2 guards active): finite and symmetric.
    for d in 3..=6 {
        let n = eta_len(d);
        let mut eta = vec![0.0; n];
        for k in 0..d {
            eta[k] = 0.5 + 0.3 * k as f64;
        }
        let mut y: Vec<f64> = (0..d).map(|k| (k as f64 * 1.3).cos()).collect();
        normalise(&mut y);
        let mut g = vec![0.0; n];
        let mut h = vec![0.0; n * n];
        loglik_eta_hess(&eta, &y, &mut g, &mut h);
        assert!(h.iter().all(|v| v.is_finite()), "d={d}");
        for i in 0..n {
            for j in 0..n {
                assert_eq!(h[i * n + j], h[j * n + i]);
            }
        }
    }
}

/// The finite-difference fallback (d = 7, eta length 27) against `HDual<27>`
/// run directly; on a large-stack thread since `HDual<27>` is 6 KB.
#[test]
fn hdual_fallback_for_other_lengths() {
    std::thread::Builder::new()
        .stack_size(64 << 20)
        .spawn(|| {
            const D: usize = 7;
            const N: usize = eta_len(D);
            let mut rng = Rng(77);
            let mut eta = rng.normals(D, 1.3);
            eta.extend(rng.normals(gamma_len(D), 0.6));
            let mut y = rng.normals(D, 1.0);
            normalise(&mut y);
            let mut g = vec![0.0; N];
            let mut h = vec![0.0; N * N];
            let v = loglik_eta_hess(&eta, &y, &mut g, &mut h);
            let x: [HDual<N>; N] = seed_hdual(&eta);
            let r = loglik_eta(&x, &y);
            let exact = r.hessian_full();
            assert!((v - r.re).abs() <= 1e-12 * r.re.abs().max(1.0));
            let mut g0 = vec![0.0; N];
            loglik_eta_grad(&eta, &y, &mut g0);
            assert_eq!(g, g0);
            let hmax = exact.iter().flatten().fold(0.0f64, |m, x| m.max(x.abs()));
            for i in 0..N {
                assert!((r.g[i] - g0[i]).abs() <= 1e-12 * g0[i].abs().max(1.0));
                for j in 0..N {
                    assert_eq!(h[i * N + j], h[j * N + i]);
                    let (a, b) = (h[i * N + j], exact[i][j]);
                    assert!(
                        (a - b).abs() <= 1e-6 * b.abs().max(1e-2 * hmax),
                        "H[{i}][{j}]: fd {a} vs exact {b}"
                    );
                }
            }
        })
        .unwrap()
        .join()
        .unwrap();
}

#[test]
#[should_panic(expected = "hess must have length")]
fn hdual_wrong_hessian_buffer_panics() {
    let eta = [0.1, 0.2, 0.3, 0.4, 0.5];
    let y = [1.0, 0.0, 0.0];
    let mut g = [0.0; 5];
    let mut h = [0.0; 24];
    loglik_eta_hess(&eta, &y, &mut g, &mut h);
}
