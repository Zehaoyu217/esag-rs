//! Agreement of the hand-written adjoint gradient
//! (`esag::adjoint::loglik_grad_adjoint`) with the forward-mode dual-number
//! gradient, on random and degenerate inputs for d = 3..=12.
//!
//! For d ≤ 8 the reference is `loglik_eta_grad` (which dispatches to
//! `Dual<N>`); for d ≥ 9 that function falls back to finite differences, so
//! the reference is `loglik_eta` evaluated directly with `Dual<N>`.

// index loops mirror the kernel's notation
#![allow(clippy::needless_range_loop)]

use esag::adjoint::loglik_grad_adjoint;
use esag::dims::{eta_len, gamma_len};
use esag::dual::{Dual, seed_array};
use esag::loglik::{loglik_eta, loglik_eta_grad_dual as loglik_eta_grad, quad_form_parts};

/// splitmix64 → uniform in [0, 1).
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn unif(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    /// uniform in [lo, hi)
    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.unif()
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
    fn normal(&mut self) -> f64 {
        let u1 = self.unif().max(1e-300);
        let u2 = self.unif();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

fn normalize(v: &mut [f64]) {
    let n = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    v.iter_mut().for_each(|x| *x /= n);
}

fn dual_ref<const N: usize>(eta: &[f64], y: &[f64], g: &mut [f64]) -> f64 {
    let x: [Dual<N>; N] = seed_array(eta);
    let v = loglik_eta(&x, y);
    g.copy_from_slice(&v.eps);
    v.re
}

/// The forward-mode reference gradient.
fn reference(eta: &[f64], y: &[f64], g: &mut [f64]) -> f64 {
    match y.len() {
        3..=8 => loglik_eta_grad(eta, y, g),
        9 => dual_ref::<44>(eta, y, g),
        10 => dual_ref::<54>(eta, y, g),
        11 => dual_ref::<65>(eta, y, g),
        12 => dual_ref::<77>(eta, y, g),
        d => panic!("no reference for d = {d}"),
    }
}

/// Group `gr` (1-based) of gamma: offset and length.
fn group(gr: usize) -> (usize, usize) {
    (gr * (gr + 1) / 2 - 1, gr + 1)
}

/// Kinds 0..N_KINDS go into the strict sweep; the nearly degenerate kinds
/// 14 and 15 (tiny but nonzero norms) are tested separately against the
/// conditioning.
const N_KINDS: usize = 14;

/// Exponent range of the scale factor `10^-e` for the nearly degenerate kinds.
#[derive(Clone, Copy)]
struct Tiny(f64, f64);

/// One test input of the given kind (see the match arms).
fn make_case(d: usize, kind: usize, rng: &mut Rng) -> (Vec<f64>, Vec<f64>, &'static str) {
    make_case_tiny(d, kind, rng, Tiny(1.0, 9.0))
}

fn make_case_tiny(
    d: usize,
    kind: usize,
    rng: &mut Rng,
    tiny: Tiny,
) -> (Vec<f64>, Vec<f64>, &'static str) {
    let g = gamma_len(d);
    let mut mu: Vec<f64> = (0..d).map(|_| rng.range(-4.0, 4.0)).collect();
    let mut gamma: Vec<f64> = (0..g).map(|_| rng.range(-2.0, 2.0)).collect();
    let mut y: Vec<f64> = (0..d).map(|_| rng.normal()).collect();
    normalize(&mut y);
    let label = match kind {
        0 => "generic",
        1 => {
            // n2 == 0: degenerate first basis column
            mu[0] = 0.0;
            mu[1] = 0.0;
            "mu0=mu1=0"
        }
        2 => {
            // a random-length zero prefix of mu: several degenerate columns
            let m = 2 + rng.below(d - 2);
            mu[..m].iter_mut().for_each(|v| *v = 0.0);
            "mu prefix 0"
        }
        3 => {
            mu.iter_mut().for_each(|v| *v = 0.0);
            "mu=0"
        }
        4 => {
            gamma.iter_mut().for_each(|v| *v = 0.0);
            "gamma=0"
        }
        5 => {
            // zero tails: in every group zero a random suffix (possibly all)
            for gr in 1..=d - 2 {
                let (off, n) = group(gr);
                let from = rng.below(n + 1);
                gamma[off + from..off + n].iter_mut().for_each(|v| *v = 0.0);
            }
            "zero group tails"
        }
        6 => {
            // scattered single zeros inside groups
            for v in gamma.iter_mut() {
                if rng.unif() < 0.4 {
                    *v = 0.0;
                }
            }
            "scattered gamma zeros"
        }
        7 => {
            // very concentrated, y near the mode
            let m = rng.range(30.0, 60.0);
            let mut dir: Vec<f64> = (0..d).map(|_| rng.normal()).collect();
            normalize(&mut dir);
            for k in 0..d {
                mu[k] = m * dir[k];
                y[k] = dir[k] + 0.05 * rng.normal();
            }
            normalize(&mut y);
            "|mu|~50"
        }
        8 => {
            gamma.iter_mut().for_each(|v| *v *= 10.0);
            "|G|~20"
        }
        9 | 10 => {
            // y nearly antipodal to mu: alpha far in the negative tail
            let m = rng.range(2.0, 45.0);
            let mut dir: Vec<f64> = (0..d).map(|_| rng.normal()).collect();
            normalize(&mut dir);
            let noise = if kind == 9 { 0.02 } else { 0.3 };
            for k in 0..d {
                mu[k] = m * dir[k];
                y[k] = -dir[k] + noise * rng.normal();
            }
            normalize(&mut y);
            if kind == 10 {
                gamma.iter_mut().for_each(|v| *v *= 0.3);
            }
            "antipodal"
        }
        11 => {
            // degenerate mu with large gamma and zero tails together
            mu[0] = 0.0;
            mu[1] = 0.0;
            if d > 3 && rng.unif() < 0.5 {
                mu[2] = 0.0;
            }
            for v in gamma.iter_mut() {
                *v *= 8.0;
                if rng.unif() < 0.3 {
                    *v = 0.0;
                }
            }
            "mixed degenerate"
        }
        12 => {
            // y along a coordinate axis, mu on another (exact zeros in y)
            y.iter_mut().for_each(|v| *v = 0.0);
            y[rng.below(d)] = if rng.unif() < 0.5 { 1.0 } else { -1.0 };
            "axis y"
        }
        13 => {
            // mu = 0 and gamma = 0: the uniform distribution
            mu.iter_mut().for_each(|v| *v = 0.0);
            gamma.iter_mut().for_each(|v| *v = 0.0);
            "uniform"
        }
        14 => {
            // nearly degenerate groups: tiny entries next to exact zeros, so
            // that some suffix norms are tiny but nonzero
            for v in gamma.iter_mut() {
                let u = rng.unif();
                if u < 0.35 {
                    *v = 0.0;
                } else if u < 0.7 {
                    *v *= 10f64.powf(-rng.range(tiny.0, tiny.1));
                }
            }
            "tiny gamma entries"
        }
        15 => {
            // nearly degenerate basis: a tiny (not zero) prefix of mu
            let m = 2 + rng.below(d - 2);
            for v in mu[..m].iter_mut() {
                *v *= 10f64.powf(-rng.range(tiny.0, tiny.1));
            }
            "tiny mu prefix"
        }
        _ => unreachable!(),
    };
    let mut eta = mu;
    eta.extend_from_slice(&gamma);
    (eta, y, label)
}

/// `1/(smallest nonzero norm)` over the gamma suffix norms and the mu prefix
/// norms `‖mu_{<i}‖`, `i ≥ 2` (at least 1). Derivatives through a rotation
/// or basis column whose norm is `r` carry a factor `1/r` that multiplies
/// rounding errors of size `ε·O(1)` in either method.
fn conditioning(eta: &[f64], d: usize) -> f64 {
    let (mu, gamma) = eta.split_at(d);
    let mut rmin = f64::INFINITY;
    for gr in 1..=d - 2 {
        let (off, n) = group(gr);
        let mut acc = 0.0;
        for j in (0..n).rev() {
            acc += gamma[off + j] * gamma[off + j];
            if acc > 0.0 {
                rmin = rmin.min(acc.sqrt());
            }
        }
    }
    let mut s = mu[0] * mu[0];
    for i in 1..d {
        s += mu[i] * mu[i];
        if s > 0.0 {
            rmin = rmin.min(s.sqrt());
        }
    }
    (1.0 / rmin).max(1.0)
}

/// `alpha = y·mu/√q` for the report (via the public value kernel).
fn alpha_of(eta: &[f64], y: &[f64]) -> f64 {
    let d = y.len();
    let (mu, gamma) = eta.split_at(d);
    let (mut z, mut k, mut rt) = (vec![0.0; d], vec![0.0; d], vec![0.0; d]);
    let (q, ymu, _) = quad_form_parts(mu, gamma, y, &mut z, &mut k, &mut rt);
    ymu / q.sqrt()
}

/// Worst deviations of the adjoint gradient from the dual reference.
#[derive(Default)]
struct Stats {
    cases: usize,
    /// max |Δ|/|g| over components with |g| ≥ 1e-2
    rel: f64,
    rel_at: String,
    /// max |Δ| over components with |g| < 1e-2
    abs_small: f64,
    /// max |Δ|/max(1, ‖g‖∞)
    scaled: f64,
    alpha_tail: (f64, f64),
    components: usize,
    /// components outside `1e-11 |g|` / `1e-13` (but within the scaled floor)
    strict_exceed: usize,
    failures: Vec<String>,
}

/// Compares adjoint and dual on `reps` cases of every kind for d = 3..=12
/// (fewer for d ≥ 11); the value must be bit-identical.
///
/// Per component the target is `|Δ| ≤ 1e-11 |g_i|` or `|Δ| ≤ 1e-13`, and the
/// pass criterion adds the floor `|Δ| ≤ 1e-14 ‖g‖∞`: a component much smaller
/// than the gradient is the result of cancellation, and there the two
/// methods differ by the rounding of the large terms. (Every component in
/// the stress set that needs the floor was checked against a 120-digit
/// evaluation of the log-density: in each the adjoint is within 3e-13 of the
/// exact value and the dual is the less accurate one, off by 1e-11..2e-11.)
fn sweep(seed: u64, reps: usize) -> Stats {
    let mut rng = Rng(seed);
    let dump = std::env::var_os("ADJOINT_DUMP").is_some();
    let mut st = Stats {
        alpha_tail: (f64::INFINITY, f64::NEG_INFINITY),
        ..Stats::default()
    };
    for d in 3..=12usize {
        let n = eta_len(d);
        let reps = if d <= 10 { reps } else { reps.div_ceil(4) };
        let mut ga = vec![0.0; n];
        let mut gd = vec![0.0; n];
        for kind in 0..N_KINDS {
            for _ in 0..reps {
                let (eta, y, label) = make_case(d, kind, &mut rng);
                let fa = loglik_grad_adjoint(&eta, &y, &mut ga);
                let fd = reference(&eta, &y, &mut gd);
                let fv: f64 = loglik_eta(&eta, &y);
                st.cases += 1;
                assert!(fv.is_finite(), "d={d} {label}: non-finite value");
                assert_eq!(
                    fa.to_bits(),
                    fv.to_bits(),
                    "d={d} {label}: value {fa} vs loglik_eta {fv}"
                );
                assert_eq!(fa.to_bits(), fd.to_bits(), "d={d} {label}: value vs dual");
                if kind == 9 || kind == 10 {
                    let a = alpha_of(&eta, &y);
                    st.alpha_tail.0 = st.alpha_tail.0.min(a);
                    st.alpha_tail.1 = st.alpha_tail.1.max(a);
                }
                let gmax = gd.iter().fold(0.0f64, |m, v| m.max(v.abs()));
                for i in 0..n {
                    let (a, b) = (ga[i], gd[i]);
                    assert!(a.is_finite(), "d={d} {label} i={i}: adjoint {a}");
                    let diff = (a - b).abs();
                    if b.abs() >= 1e-2 {
                        let rel = diff / b.abs();
                        if rel > st.rel {
                            st.rel = rel;
                            st.rel_at = format!("d={d} {label} i={i}: adjoint {a:e} dual {b:e}");
                        }
                    } else {
                        st.abs_small = st.abs_small.max(diff);
                    }
                    st.scaled = st.scaled.max(diff / gmax.max(1.0));
                    let strict = diff <= 1e-11 * b.abs() || diff <= 1e-13;
                    let ok = strict || diff <= 1e-14 * gmax;
                    st.components += 1;
                    st.strict_exceed += usize::from(!strict);
                    if !ok && st.failures.len() < 20 {
                        st.failures.push(format!(
                            "d={d} {label} i={i}: adjoint {a:e} dual {b:e} \
                             (diff {diff:.2e}, |g|∞ {gmax:.2e})"
                        ));
                    }
                    // diagnostic dump for an external high-precision check
                    if dump && !strict {
                        eprintln!(
                            "CASEJSON {{\"i\":{i},\"adj\":{a:e},\"dual\":{b:e},\"label\":\"{label}\",\
                             \"eta\":{eta:?},\"y\":{y:?}}}"
                        );
                    }
                }
            }
        }
    }
    eprintln!(
        "adjoint vs dual, {} cases, {} components: max rel {:.2e} (|g| ≥ 1e-2; {}), \
         max abs {:.2e} (|g| < 1e-2), max |Δ|/max(1,‖g‖∞) {:.2e}; \
         {} components outside 1e-11 rel / 1e-13 abs; antipodal alpha in [{:.1}, {:.1}]",
        st.cases,
        st.components,
        st.rel,
        st.rel_at,
        st.abs_small,
        st.scaled,
        st.strict_exceed,
        st.alpha_tail.0,
        st.alpha_tail.1
    );
    st
}

#[test]
fn adjoint_matches_dual_gradient() {
    let st = sweep(0x5EED_AD10, 70);
    assert!(
        st.failures.is_empty(),
        "components outside 1e-11 rel / 1e-13 abs / 1e-14 ‖g‖∞:\n{}",
        st.failures.join("\n")
    );
    // the antipodal cases reach deep into the continued-fraction tail
    assert!(st.alpha_tail.0 < -40.0 && st.alpha_tail.1 > -2.5);
}

/// Ten times as many cases: `cargo test --release --test adjoint_grad -- --ignored`.
/// With `ADJOINT_DUMP=1` and `--nocapture`, every component outside the
/// strict target is printed as a `CASEJSON` line for an external
/// high-precision check.
#[test]
#[ignore]
fn adjoint_matches_dual_gradient_stress() {
    let mut fails = Vec::new();
    for seed in 1..=10u64 {
        fails.extend(sweep(seed * 0x9E37_79B9, 70).failures);
    }
    assert!(fails.is_empty(), "{}", fails.join("\n"));
}

#[test]
fn adjoint_near_degenerate_matches_dual_to_conditioning() {
    // Entries 1e-1..1e-9 next to exact zeros make suffix (or prefix) norms
    // tiny but nonzero. Both methods then share an error ~ ε/r_min (the
    // derivative of q along a nearly isotropic rotation is an O(r) quantity
    // obtained from O(1) terms, then divided by r), so they can only agree to
    // that level. Checked against a 120-digit evaluation on 456 such
    // components: the adjoint is the more accurate one in 82% of them
    // (median error ratio 0.017; worst relative error 2.6e-8 adjoint,
    // 1.7e-6 dual).
    let mut rng = Rng(0x7E57_0001);
    let mut worst: f64 = 0.0;
    for d in 3..=10usize {
        let n = eta_len(d);
        let mut ga = vec![0.0; n];
        let mut gd = vec![0.0; n];
        for kind in [14, 15] {
            for _ in 0..150 {
                let (eta, y, label) = make_case_tiny(d, kind, &mut rng, Tiny(1.0, 9.0));
                let fa = loglik_grad_adjoint(&eta, &y, &mut ga);
                let fd = reference(&eta, &y, &mut gd);
                assert_eq!(fa.to_bits(), fd.to_bits());
                let kappa = conditioning(&eta, d);
                let gmax = gd.iter().fold(1.0f64, |m, v| m.max(v.abs()));
                for i in 0..n {
                    let diff = (ga[i] - gd[i]).abs();
                    let tol = 1e-11 * gd[i].abs() + 1e-13 + 1e-13 * kappa * gmax;
                    worst = worst.max(diff / (kappa * gmax));
                    assert!(
                        diff <= tol,
                        "d={d} {label} i={i}: adjoint {} dual {} (κ {kappa:.1e})",
                        ga[i],
                        gd[i]
                    );
                }
            }
        }
    }
    eprintln!("near-degenerate: max |Δ|/(κ max(1,‖g‖∞)) = {worst:.2e}");
}

#[test]
fn adjoint_tail_alpha_range_is_covered() {
    // The antipodal construction must actually reach the continued-fraction
    // branch of log M_p (alpha < −2) down to about −40.
    let mut rng = Rng(7);
    for d in [3usize, 5, 8, 10] {
        let n = eta_len(d);
        let mut ga = vec![0.0; n];
        let mut gd = vec![0.0; n];
        for &m in &[2.5f64, 5.0, 10.0, 20.0, 40.0] {
            let mut dir: Vec<f64> = (0..d).map(|_| rng.normal()).collect();
            normalize(&mut dir);
            let mut eta: Vec<f64> = vec![0.0; n];
            for k in 0..d {
                eta[k] = m * dir[k];
            }
            for v in eta[d..].iter_mut() {
                *v = rng.range(-0.5, 0.5);
            }
            let y: Vec<f64> = dir.iter().map(|v| -v).collect();
            loglik_grad_adjoint(&eta, &y, &mut ga);
            reference(&eta, &y, &mut gd);
            for i in 0..n {
                let diff = (ga[i] - gd[i]).abs();
                assert!(
                    diff <= 1e-11 * gd[i].abs() || diff <= 1e-13,
                    "d={d} |mu|={m} i={i}: {} vs {}",
                    ga[i],
                    gd[i]
                );
            }
        }
    }
}

#[test]
fn adjoint_gamma_zero_has_zero_gamma_gradient() {
    // sqrt'(0) := 0: at the isotropic point the gamma gradient vanishes
    // (the dual convention), while the mu gradient is the isotropic one.
    for d in 3..=10usize {
        let n = eta_len(d);
        let mut eta = vec![0.0; n];
        for k in 0..d {
            eta[k] = 0.7 - 0.3 * k as f64;
        }
        let mut y: Vec<f64> = (0..d).map(|k| (k as f64 + 1.0).sin()).collect();
        normalize(&mut y);
        let mut ga = vec![f64::NAN; n];
        let mut gd = vec![0.0; n];
        loglik_grad_adjoint(&eta, &y, &mut ga);
        reference(&eta, &y, &mut gd);
        for i in d..n {
            assert_eq!(ga[i], 0.0, "d={d} i={i}");
            assert_eq!(gd[i], 0.0, "d={d} i={i} (dual)");
        }
        for i in 0..d {
            assert!((ga[i] - gd[i]).abs() <= 1e-13 * gd[i].abs().max(1.0));
        }
    }
}

#[test]
fn adjoint_matches_finite_differences_d9_d10() {
    // independent of the dual reference: central differences
    let mut rng = Rng(99);
    for d in [9usize, 10] {
        let n = eta_len(d);
        let (eta, y, _) = make_case(d, 0, &mut rng);
        let mut ga = vec![0.0; n];
        loglik_grad_adjoint(&eta, &y, &mut ga);
        let mut e = eta.clone();
        for i in 0..n {
            let h = 1e-6;
            e[i] = eta[i] + h;
            let fp: f64 = loglik_eta(&e, &y);
            e[i] = eta[i] - h;
            let fm: f64 = loglik_eta(&e, &y);
            e[i] = eta[i];
            let fd = (fp - fm) / (2.0 * h);
            assert!(
                (ga[i] - fd).abs() < 1e-6 * fd.abs().max(1.0),
                "d={d} i={i}: adjoint {} fd {fd}",
                ga[i]
            );
        }
    }
}

#[test]
#[should_panic(expected = "eta has length")]
fn adjoint_rejects_wrong_eta_length() {
    let mut g = vec![0.0; 4];
    loglik_grad_adjoint(&[0.0; 4], &[1.0, 0.0, 0.0], &mut g);
}
