//! The structured exact Hessian `esag::adjoint_hess::loglik_hess_adjoint`
//! against the second-order dual numbers of `esag::hdual` (run directly as
//! `HDual<N>` for every `d = 3..=9`; d = 9 is the heap path), on random and
//! degenerate inputs.
//!
//! Value and gradient must be bit-identical to `loglik_eta` and the adjoint
//! gradient; the Hessian must agree to rounding, measured entrywise against
//! the scale `√(D_i D_j)` with `D_i` the largest entry of row `i` (the
//! natural scale of a Hessian entry, also where the rows differ by orders of
//! magnitude near a singular set).

#![allow(clippy::needless_range_loop)]

use esag::adjoint::loglik_grad_adjoint;
use esag::adjoint_hess::loglik_hess_adjoint;
use esag::dims::{eta_len, gamma_len};
use esag::hdual::{HDual, seed_hdual};
use esag::loglik::loglik_eta;

/// SplitMix64 with Box–Muller normals.
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

/// Range in `eta` of γ group `gr` (1-based, `gr + 1` entries).
fn group(d: usize, gr: usize) -> std::ops::Range<usize> {
    let off = d + gr * (gr + 1) / 2 - 1;
    off..off + gr + 1
}

struct Case {
    label: String,
    eta: Vec<f64>,
    y: Vec<f64>,
}

fn cases(d: usize, rng: &mut Rng) -> Vec<Case> {
    let g = gamma_len(d);
    let mut out = Vec::new();
    let mut push = |label: String, eta: Vec<f64>, mut y: Vec<f64>| {
        normalise(&mut y);
        out.push(Case { label, eta, y });
    };
    let generic = |rng: &mut Rng| {
        let mut eta = rng.normals(d, 1.5);
        eta.extend(rng.normals(g, 0.7));
        eta
    };
    for k in 0..6 {
        let eta = generic(rng);
        let y = rng.normals(d, 1.0);
        push(format!("random #{k}"), eta, y);
    }
    // concentrated, and y far from the mode (log M_p tail branch)
    for &m in &[6.0, 12.0] {
        let mut dir = rng.normals(d, 1.0);
        normalise(&mut dir);
        let mut eta: Vec<f64> = dir.iter().map(|v| m * v).collect();
        eta.extend(rng.normals(g, 0.6));
        let y: Vec<f64> = dir.iter().map(|v| -v + 0.05 * rng.normal()).collect();
        push(format!("antipodal |mu| = {m}"), eta, y);
    }
    // a whole group small, tiny, or exactly zero
    for gr in 1..=d - 2 {
        for &scale in &[1e-3, 1e-7, 0.0] {
            let mut eta = generic(rng);
            let r = group(d, gr);
            let mut grp = rng.normals(r.len(), 1.0);
            normalise(&mut grp);
            for (k, idx) in r.enumerate() {
                eta[idx] = scale * grp[k];
            }
            let y = rng.normals(d, 1.0);
            push(format!("group {gr} scaled {scale:e}"), eta, y);
        }
    }
    // the trailing pair (θ's plane) of a group j ≥ 2 small, tiny, or zero;
    // and a longer zero tail
    for gr in 2..=d - 2 {
        for &scale in &[1e-3, 1e-8, 0.0] {
            let mut eta = generic(rng);
            let r = group(d, gr);
            eta[r.end - 1] *= scale;
            eta[r.end - 2] *= scale;
            let y = rng.normals(d, 1.0);
            push(format!("group {gr} trailing pair x {scale:e}"), eta, y);
        }
        if gr >= 3 {
            let mut eta = generic(rng);
            let r = group(d, gr);
            for idx in r.start + 1..r.end {
                eta[idx] = 0.0;
            }
            let y = rng.normals(d, 1.0);
            push(format!("group {gr} all but the first zero"), eta, y);
        }
    }
    // single zero entries
    {
        let mut eta = generic(rng);
        for gr in 1..=d - 2 {
            eta[group(d, gr).end - 1] = 0.0;
        }
        let y = rng.normals(d, 1.0);
        push("last entry of every group zero".into(), eta, y);
    }
    // isotropic
    {
        let mut eta = generic(rng);
        eta[d..].iter_mut().for_each(|v| *v = 0.0);
        let y = rng.normals(d, 1.0);
        push("gamma = 0".into(), eta, y);
    }
    // mu near and at the frame pole mu_0 = mu_1 = 0, and other zero prefixes
    for &scale in &[1e-4, 1e-9, 0.0] {
        let mut eta = generic(rng);
        eta[0] *= scale;
        eta[1] *= scale;
        let y = rng.normals(d, 1.0);
        push(format!("mu_0, mu_1 x {scale:e}"), eta, y);
    }
    {
        let mut eta = generic(rng);
        for v in eta.iter_mut().take(d - 1) {
            *v = 0.0;
        }
        let y = rng.normals(d, 1.0);
        push("mu along e_d".into(), eta, y);
    }
    {
        let mut eta = generic(rng);
        eta[..d].iter_mut().for_each(|v| *v = 0.0);
        let y = rng.normals(d, 1.0);
        push("mu = 0".into(), eta, y);
    }
    out
}

fn reference<const N: usize>(eta: &[f64], y: &[f64]) -> (f64, Vec<f64>, Vec<f64>) {
    let x: [HDual<N>; N] = seed_hdual(eta);
    let v = loglik_eta(&x, y);
    let mut h = vec![0.0; N * N];
    v.write_hessian(&mut h);
    (v.re, v.g.to_vec(), h)
}

fn hdual_ref(eta: &[f64], y: &[f64]) -> (f64, Vec<f64>, Vec<f64>) {
    match eta.len() {
        5 => reference::<5>(eta, y),
        9 => reference::<9>(eta, y),
        14 => reference::<14>(eta, y),
        20 => reference::<20>(eta, y),
        27 => reference::<27>(eta, y),
        35 => reference::<35>(eta, y),
        44 => reference::<44>(eta, y),
        n => panic!("no reference for eta length {n}"),
    }
}

/// Worst entrywise error `|a − b| / √(D_i D_j)`.
fn scaled_error(a: &[f64], b: &[f64], n: usize) -> (f64, usize, usize) {
    let dmax: Vec<f64> = (0..n)
        .map(|i| (0..n).fold(0.0f64, |m, j| m.max(b[i * n + j].abs())))
        .collect();
    let mut worst = (0.0, 0, 0);
    for i in 0..n {
        for j in 0..n {
            let scale = (dmax[i] * dmax[j]).sqrt().max(1e-300);
            let e = (a[i * n + j] - b[i * n + j]).abs() / scale;
            if e > worst.0 || e.is_nan() {
                worst = (e, i, j);
            }
        }
    }
    worst
}

/// Tolerance on the scaled error for a case. Near `G_1 = 0` both
/// implementations compute the `O(‖G_1‖)` angular curvature of a nearly
/// isotropic block from `O(1)` terms (see the module documentation).
fn tolerance(label: &str) -> f64 {
    if label.starts_with("group 1 scaled 1e-3") {
        1e-9
    } else if label.starts_with("group 1 scaled 1e-7") {
        1e-6
    } else {
        1e-12
    }
}

fn run(d: usize, seed: u64) -> f64 {
    let n = eta_len(d);
    let mut rng = Rng(seed);
    let mut worst_all = 0.0f64;
    for c in cases(d, &mut rng) {
        let (eta, y) = (&c.eta, &c.y);
        let mut g = vec![0.0; n];
        let mut h = vec![0.0; n * n];
        let v = loglik_hess_adjoint(eta, y, &mut g, &mut h);
        let tag = format!("d={d} {}", c.label);
        let v0: f64 = loglik_eta(eta, y);
        assert_eq!(v.to_bits(), v0.to_bits(), "{tag}: value {v} vs {v0}");
        let mut g0 = vec![0.0; n];
        loglik_grad_adjoint(eta, y, &mut g0);
        for i in 0..n {
            assert_eq!(
                g[i].to_bits(),
                g0[i].to_bits(),
                "{tag}: grad[{i}] {} vs {}",
                g[i],
                g0[i]
            );
        }
        for i in 0..n {
            for j in 0..n {
                assert_eq!(h[i * n + j], h[j * n + i], "{tag}: asymmetric ({i},{j})");
            }
        }
        let (vr, _, hr) = hdual_ref(eta, y);
        assert!(
            (v - vr).abs() <= 1e-12 * vr.abs().max(1.0),
            "{tag}: value vs HDual"
        );
        let (e, i, j) = scaled_error(&h, &hr, n);
        assert!(
            e <= tolerance(&c.label),
            "{tag}: H[{i}][{j}] = {} vs HDual {} (scaled error {e:.2e})",
            h[i * n + j],
            hr[i * n + j]
        );
        if tolerance(&c.label) == 1e-12 {
            worst_all = worst_all.max(e);
        }
    }
    worst_all
}

#[test]
fn matches_hdual_d3_to_d9() {
    std::thread::Builder::new()
        .stack_size(512 << 20)
        .spawn(|| {
            for d in 3..=9 {
                let w = run(d, 0xA11CE + d as u64);
                eprintln!("d={d}: worst scaled error {w:.2e} (outside group 1 near zero)");
            }
        })
        .unwrap()
        .join()
        .unwrap();
}

/// Median over `reps` batches of the mean time per call (ns).
fn time_ns(reps: usize, iters: usize, mut f: impl FnMut()) -> f64 {
    for _ in 0..iters {
        f();
    }
    let mut v: Vec<f64> = (0..reps)
        .map(|_| {
            let t = std::time::Instant::now();
            for _ in 0..iters {
                f();
            }
            t.elapsed().as_nanos() as f64 / iters as f64
        })
        .collect();
    v.sort_by(f64::total_cmp);
    v[reps / 2]
}

/// Per-observation timings: the adjoint gradient, this Hessian, central
/// differences of the gradient and (d ≤ 6) the hyper-dual Hessian.
/// `cargo test --release --test adjoint_hess bench -- --ignored --nocapture`
#[test]
#[ignore]
fn bench_hessian_kernels() {
    use esag::hdual::{loglik_eta_hess, loglik_eta_hess_fd};
    use std::hint::black_box;
    println!(
        "\n{:>3} {:>3} {:>10} {:>12} {:>11} {:>11} {:>8} {:>8}",
        "d", "N", "grad (ns)", "adjoint_hess", "fd (ns)", "hdual (ns)", "×grad", "fd/new"
    );
    for d in 3..=8 {
        let n = eta_len(d);
        let mut rng = Rng(0xBE7C + d as u64);
        let ins: Vec<(Vec<f64>, Vec<f64>)> = (0..16)
            .map(|_| {
                let mut eta = rng.normals(d, 1.5);
                eta.extend(rng.normals(gamma_len(d), 0.7));
                let mut y = rng.normals(d, 1.0);
                normalise(&mut y);
                (eta, y)
            })
            .collect();
        let mut g = vec![0.0; n];
        let mut h = vec![0.0; n * n];
        let iters = 4000 / n;
        let per = |t: f64| t / ins.len() as f64;
        let tg = per(time_ns(15, iters, || {
            for (eta, y) in &ins {
                black_box(loglik_grad_adjoint(black_box(eta), black_box(y), &mut g));
            }
        }));
        let th = per(time_ns(15, iters, || {
            for (eta, y) in &ins {
                black_box(loglik_hess_adjoint(
                    black_box(eta),
                    black_box(y),
                    &mut g,
                    &mut h,
                ));
            }
        }));
        let tf = per(time_ns(9, iters / 8 + 1, || {
            for (eta, y) in &ins {
                black_box(loglik_eta_hess_fd(
                    black_box(eta),
                    black_box(y),
                    &mut g,
                    &mut h,
                ));
            }
        }));
        let thd = if d <= 6 {
            per(time_ns(9, iters / 8 + 1, || {
                for (eta, y) in &ins {
                    black_box(loglik_eta_hess(
                        black_box(eta),
                        black_box(y),
                        &mut g,
                        &mut h,
                    ));
                }
            }))
        } else {
            f64::NAN
        };
        black_box((&g, &h));
        println!(
            "{d:>3} {n:>3} {tg:>10.0} {th:>12.0} {tf:>11.0} {thd:>11.0} {:>8.1} {:>8.1}",
            th / tg,
            tf / th
        );
    }
}
