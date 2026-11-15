//! Timing of the log-tangent kernel (`esag::logtangent::loglik_lt`, value;
//! `loglik_lt_grad`, value and gradient; `loglik_lt_hess`, value, gradient
//! and Hessian) next to the `γ` kernel (`loglik_eta::<f64>` and the adjoint
//! `loglik_eta_grad`), per call.
//!
//! `cargo run --release --example logtangent_bench [-- --runs 15]`
//!
//! Each figure is the minimum over `runs` batches of the mean time per call
//! (the machine may be shared, so the minimum is the stable statistic). The
//! inputs are 256 random draws per dimension, with 1/8 of them `y` nearly
//! antipodal to `mu` (the continued-fraction tail of `log M_p`). The shape
//! block (`s`, read as `γ` by the `γ` kernel) is zero for 1/16 of the draws
//! in set `mixed`, for none in set `full S`, and for all in set `s=0`
//! (isotropy: no eigen-decomposition). The log-tangent kernel uses the
//! standard frame.

use esag::dims::{eta_len, gamma_len};
use esag::loglik::{loglik_eta, loglik_eta_grad};
use esag::logtangent::{LtFrame, loglik_lt, loglik_lt_grad, loglik_lt_hess};
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

/// `(eta, y)` pairs; `iso` selects `s = 0` (or `gamma = 0`) for every draw,
/// `Some(false)` none, `None` the mixed set.
fn inputs(d: usize, iso: Option<bool>) -> Vec<(Vec<f64>, Vec<f64>)> {
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
            let zero = match iso {
                Some(z) => z,
                None => s % 16 == 3,
            };
            if zero {
                eta[d..d + gamma_len(d)].iter_mut().for_each(|v| *v = 0.0);
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
        "{:<8}{:>3}{:>4}{:>10}{:>10}{:>10}{:>10}{:>10}{:>9}{:>9}{:>9}",
        "set",
        "d",
        "N",
        "eta val",
        "eta grad",
        "lt val",
        "lt grad",
        "lt hess",
        "lt/eta v",
        "lt/eta g",
        "hess/grd"
    );
    for (set, iso) in [
        ("mixed", None),
        ("full S", Some(false)),
        ("s=0", Some(true)),
    ] {
        for d in [3usize, 4, 5, 6, 7, 8, 10] {
            let ins = inputs(d, iso);
            let n = eta_len(d);
            let frame = LtFrame::standard(d);
            let calls = 20_000;
            let mut g = vec![0.0; n];
            let mut h = vec![0.0; n * n];
            let tv = min_ns(runs, calls, &ins, loglik_eta::<f64>);
            let tg = min_ns(runs, calls, &ins, |e, y| loglik_eta_grad(e, y, &mut g));
            let lv = min_ns(runs, calls, &ins, |e, y| loglik_lt(&frame, e, y));
            let lg = min_ns(runs, calls, &ins, |e, y| {
                loglik_lt_grad(&frame, e, y, &mut g)
            });
            let lh = min_ns(runs, calls / 4, &ins, |e, y| {
                loglik_lt_hess(&frame, e, y, &mut g, &mut h)
            });
            println!(
                "{set:<8}{d:>3}{n:>4}{tv:>10.1}{tg:>10.1}{lv:>10.1}{lg:>10.1}{lh:>10.1}{:>9.1}{:>9.1}{:>9.1}",
                lv / tv,
                lg / tg,
                lh / lg
            );
        }
    }
}
