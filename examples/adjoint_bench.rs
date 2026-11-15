//! Timing of the adjoint gradient `esag::adjoint::loglik_grad_adjoint`
//! against the value `loglik_eta::<f64>` and the forward-mode gradient
//! `loglik_eta_grad`, per call, on varied inputs.
//!
//! `cargo run --release --example adjoint_bench [-- --runs 15]`
//!
//! Each figure is the minimum over `runs` batches of the mean time per call
//! (the machine may be shared, so the minimum is the stable statistic).
//! Two input sets per dimension: the 64 inputs of `perf_suite` ("suite"),
//! and 256 random draws with 1/8 of them y nearly antipodal to mu, which
//! takes `log M_p` into its continued-fraction tail ("mixed").

use esag::adjoint::loglik_grad_adjoint;
use esag::dims::eta_len;
use esag::loglik::{loglik_eta, loglik_eta_grad};
use std::hint::black_box;
use std::time::Instant;

/// Minimum over `runs` batches of the mean ns per call of `f` over `inputs`.
fn min_ns(
    runs: usize,
    calls: usize,
    inputs: &[(Vec<f64>, Vec<f64>)],
    mut f: impl FnMut(&[f64], &[f64]) -> f64,
) -> f64 {
    let m = inputs.len();
    let mut run = || {
        let t = Instant::now();
        let mut acc = 0.0;
        for r in 0..calls {
            let (e, y) = &inputs[r % m];
            acc += f(black_box(e), black_box(y));
        }
        black_box(acc);
        t.elapsed().as_nanos() as f64 / calls as f64
    };
    run(); // warm-up
    (0..runs).map(|_| run()).fold(f64::INFINITY, f64::min)
}

fn suite_inputs(d: usize) -> Vec<(Vec<f64>, Vec<f64>)> {
    let b = eta_len(d);
    (0..64)
        .map(|s| {
            let eta: Vec<f64> = (0..b)
                .map(|i| 0.3 + ((i * 7 + s) as f64 * 0.77).sin())
                .collect();
            let r: Vec<f64> = (0..d).map(|i| ((i * 3 + s) as f64 * 1.9).cos()).collect();
            let n = r.iter().map(|v| v * v).sum::<f64>().sqrt();
            (eta, r.iter().map(|v| v / n).collect())
        })
        .collect()
}

fn mixed_inputs(d: usize) -> Vec<(Vec<f64>, Vec<f64>)> {
    let mut st = 0x9E37_79B9_7F4A_7C15u64 ^ d as u64;
    let mut unif = move || {
        st = st
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (st >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    };
    let b = eta_len(d);
    (0..256)
        .map(|s| {
            let mut eta: Vec<f64> = (0..b).map(|_| 1.5 * unif()).collect();
            let mut y: Vec<f64> = (0..d).map(|_| unif()).collect();
            if s % 8 == 7 {
                for k in 0..d {
                    eta[k] *= 4.0;
                    y[k] = -eta[k] + 0.1 * unif();
                }
            }
            let n = y.iter().map(|v| v * v).sum::<f64>().sqrt();
            y.iter_mut().for_each(|v| *v /= n);
            (eta, y)
        })
        .collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let runs = args
        .iter()
        .position(|a| a == "--runs")
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(15usize);
    println!("min over {runs} runs, ns per call");
    println!(
        "{:<6}{:>3}{:>4}{:>10}{:>10}{:>11}{:>11}{:>10}",
        "set", "d", "N", "value", "dual", "adjoint", "adj/value", "dual/adj"
    );
    for set in ["suite", "mixed"] {
        for d in [3usize, 4, 5, 6, 7, 8, 10] {
            let inputs = if set == "suite" {
                suite_inputs(d)
            } else {
                mixed_inputs(d)
            };
            let n = eta_len(d);
            let calls = 40_000;
            let mut g = vec![0.0; n];
            let tv = min_ns(runs, calls, &inputs, loglik_eta::<f64>);
            let td = min_ns(runs, calls / 4, &inputs, |e, y| {
                loglik_eta_grad(e, y, &mut g)
            });
            let ta = min_ns(runs, calls, &inputs, |e, y| {
                loglik_grad_adjoint(e, y, &mut g)
            });
            println!(
                "{set:<6}{d:>3}{n:>4}{tv:>10.1}{td:>10.1}{ta:>11.1}{:>11.2}{:>10.2}",
                ta / tv,
                td / ta
            );
        }
    }
    println!("(d = 10: `loglik_eta_grad` is central finite differences, not dual numbers)");
}
