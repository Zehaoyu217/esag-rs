//! Timing of `loglik_eta_hess` against `loglik_eta_grad` (and against the
//! `2·N` gradient evaluations a central-difference Hessian would need).
//!
//! `cargo test --release --test hdual_bench -- --ignored --nocapture`

use std::hint::black_box;
use std::time::Instant;

use esag::dims::{eta_len, gamma_len};
use esag::hdual::loglik_eta_hess;
use esag::loglik::loglik_eta_grad;

/// Median over `reps` batches of the mean time per call (ns).
fn time_ns(reps: usize, iters: usize, mut f: impl FnMut()) -> f64 {
    for _ in 0..iters {
        f();
    }
    let mut v: Vec<f64> = (0..reps)
        .map(|_| {
            let t = Instant::now();
            for _ in 0..iters {
                f();
            }
            t.elapsed().as_nanos() as f64 / iters as f64
        })
        .collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[reps / 2]
}

/// Deterministic inputs: a few (eta, y) per dimension, including a tail point.
fn inputs(d: usize) -> Vec<(Vec<f64>, Vec<f64>)> {
    let n = eta_len(d);
    let mut out = Vec::new();
    for s in 0..4 {
        let mut eta: Vec<f64> = (0..n)
            .map(|i| 0.3 + ((i * 7 + s * 13) as f64 * 0.77).sin())
            .collect();
        for v in eta[d..d + gamma_len(d)].iter_mut() {
            *v *= 0.8;
        }
        let mut y: Vec<f64> = (0..d).map(|i| ((i + 3 * s) as f64 * 1.9).cos()).collect();
        if s == 3 {
            // nearly antipodal to mu: alpha well below −2
            for k in 0..d {
                y[k] = -eta[k] + 0.05 * (k as f64).sin();
            }
            for v in eta[..d].iter_mut() {
                *v *= 3.0;
            }
        }
        let nrm = y.iter().map(|v| v * v).sum::<f64>().sqrt();
        y.iter_mut().for_each(|v| *v /= nrm);
        out.push((eta, y));
    }
    out
}

#[test]
#[ignore]
fn hdual_bench_hessian_vs_gradient() {
    println!();
    println!(
        "{:>3} {:>3} {:>12} {:>12} {:>8} {:>14}",
        "d", "N", "grad (ns)", "hess (ns)", "ratio", "2N·grad (ns)"
    );
    for d in 3..=6 {
        let n = eta_len(d);
        let ins = inputs(d);
        let mut g = vec![0.0; n];
        let mut h = vec![0.0; n * n];
        let iters = 20_000 / n;
        let tg = time_ns(15, iters, || {
            for (eta, y) in &ins {
                black_box(loglik_eta_grad(black_box(eta), black_box(y), &mut g));
            }
        }) / ins.len() as f64;
        let th = time_ns(15, iters, || {
            for (eta, y) in &ins {
                black_box(loglik_eta_hess(black_box(eta), black_box(y), &mut g, &mut h));
            }
        }) / ins.len() as f64;
        black_box((&g, &h));
        println!(
            "{d:>3} {n:>3} {tg:>12.0} {th:>12.0} {:>8.2} {:>14.0}",
            th / tg,
            2.0 * n as f64 * tg
        );
    }
}
