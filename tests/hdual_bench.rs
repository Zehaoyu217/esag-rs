#![allow(clippy::needless_range_loop)]
//! Timing of `loglik_eta_hess` against `loglik_eta_grad` and the
//! finite-difference Hessian `loglik_eta_hess_fd` (`2N + 1` gradient calls),
//! plus the cost of single `HDual<N>` operations.
//!
//! `cargo test --release --test hdual_bench -- --ignored --nocapture`

use std::hint::black_box;
use std::time::Instant;

use esag::Scalar;
use esag::dims::{eta_len, gamma_len};
use esag::hdual::{HDual, loglik_eta_hess, loglik_eta_hess_fd, seed_hdual};
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
        "{:>3} {:>3} {:>11} {:>11} {:>13} {:>14}",
        "d", "N", "grad (ns)", "hess (ns)", "hess_fd (ns)", "hess / grad"
    );
    for d in 3..=6 {
        let n = eta_len(d);
        let ins = inputs(d);
        let mut g = vec![0.0; n];
        let mut h = vec![0.0; n * n];
        let iters = 20_000 / n;
        let per = |t: f64| t / ins.len() as f64;
        let tg = per(time_ns(15, iters, || {
            for (eta, y) in &ins {
                black_box(loglik_eta_grad(black_box(eta), black_box(y), &mut g));
            }
        }));
        let th = per(time_ns(15, iters, || {
            for (eta, y) in &ins {
                black_box(loglik_eta_hess(
                    black_box(eta),
                    black_box(y),
                    &mut g,
                    &mut h,
                ));
            }
        }));
        let tf = per(time_ns(15, iters / 4 + 1, || {
            for (eta, y) in &ins {
                black_box(loglik_eta_hess_fd(
                    black_box(eta),
                    black_box(y),
                    &mut g,
                    &mut h,
                ));
            }
        }));
        black_box((&g, &h));
        println!(
            "{d:>3} {n:>3} {tg:>11.0} {th:>11.0} {tf:>13.0} {:>14.1}",
            th / tg
        );
    }
}

/// Median-of-batches time of one operation on `HDual<N>` values held in an
/// array (so operands and results go through memory, as in real code).
fn per_op<const N: usize>(f: impl Fn(&[HDual<N>; 8], &mut [HDual<N>; 8], usize)) -> f64 {
    let x = seed_hdual::<N>(&[0.3; N]);
    let src: [HDual<N>; 8] = core::array::from_fn(|k| (x[k % N] * x[(k + 1) % N] + x[k % N]).exp());
    let mut dst = src;
    let ns = time_ns(9, 50_000, || {
        for k in 0..8 {
            f(black_box(&src), &mut dst, k);
        }
    }) / 8.0;
    black_box(&dst);
    ns
}

fn op_row<const N: usize>() {
    let r = [
        per_op::<N>(|s, d, k| d[k] = s[(k + 3) & 7]),
        per_op::<N>(|_, d, k| d[k] = HDual::<N>::constant(k as f64)),
        per_op::<N>(|s, d, k| d[k] = s[(k + 3) & 7] + s[(k + 5) & 7]),
        per_op::<N>(|s, d, k| d[k] += s[(k + 5) & 7]),
        per_op::<N>(|s, d, k| d[k] = s[(k + 3) & 7] * s[(k + 5) & 7]),
        per_op::<N>(|s, d, k| d[k] *= s[(k + 5) & 7]),
        per_op::<N>(|s, d, k| d[k] = s[(k + 3) & 7] / s[(k + 5) & 7]),
        per_op::<N>(|s, d, k| d[k] = s[(k + 3) & 7] * 0.5),
        per_op::<N>(|s, d, k| d[k] = s[(k + 3) & 7].exp()),
        per_op::<N>(|s, d, k| d[k] = s[(k + 3) & 7].powi(2)),
    ];
    print!("{N:>3}");
    for v in r {
        print!(" {v:>7.1}");
    }
    println!();
}

#[test]
#[ignore]
fn hdual_bench_single_ops() {
    println!();
    println!(
        "{:>3} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7}   (ns)",
        "N", "copy", "const", "a+b", "a+=b", "a*b", "a*=b", "a/b", "a*f64", "exp", "powi2"
    );
    op_row::<5>();
    op_row::<9>();
    op_row::<14>();
    op_row::<20>();
}

/// A long loop for a sampling profiler:
/// `HDUAL_PROFILE_D=6 cargo test --release --test hdual_bench hdual_profile -- --ignored`.
#[test]
#[ignore]
fn hdual_profile_loop() {
    // Only runs when asked for, so `--ignored` timings are not disturbed.
    let Some(d) = std::env::var("HDUAL_PROFILE_D")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
    else {
        return;
    };
    let n = eta_len(d);
    let ins = inputs(d);
    let mut g = vec![0.0; n];
    let mut h = vec![0.0; n * n];
    let t = Instant::now();
    while t.elapsed().as_secs_f64() < 4.0 {
        for _ in 0..100 {
            for (eta, y) in &ins {
                black_box(loglik_eta_hess(
                    black_box(eta),
                    black_box(y),
                    &mut g,
                    &mut h,
                ));
            }
        }
    }
}
