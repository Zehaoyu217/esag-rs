//! Large regression fits: d = 7–8 responses, 5–6 covariates.
//! `cargo run --release --example big_fit [-- --quick]`

use esag::distribution::Esag;
use esag::regression::EsagRegressor;
use nalgebra::DMatrix;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal};
use std::time::Instant;

/// Simulate `n` responses on S^{d-1} with `p` standard-normal covariates.
/// Intercept: ‖μ‖ ≈ 4, moderate anisotropy; slopes of size ~0.3.
pub fn sim(d: usize, p: usize, n: usize, seed: u64) -> (DMatrix<f64>, DMatrix<f64>, DMatrix<f64>) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let bd = esag::dims::eta_len(d);
    let b = DMatrix::from_fn(bd, p + 1, |k, j| {
        let s = (k * 7 + j * 3) as f64;
        if j == 0 {
            if k < d {
                1.6 * (1.0 + 0.8 * (s * 0.9).sin())
            } else {
                0.8 * (s * 1.3).sin()
            }
        } else if k < d {
            0.35 * (s * 0.7).cos()
        } else {
            0.2 * (s * 1.1).sin()
        }
    });
    let x = DMatrix::from_fn(n, p, |_, _| StandardNormal.sample(&mut rng));
    let mut y = DMatrix::zeros(n, d);
    let mut buf = vec![0.0; d];
    let mut eta = vec![0.0; bd];
    for i in 0..n {
        for k in 0..bd {
            eta[k] = b[(k, 0)] + (0..p).map(|j| b[(k, j + 1)] * x[(i, j)]).sum::<f64>();
        }
        let dist = Esag::from_eta(&eta).unwrap();
        dist.sample_one(&dist.frame(), &mut rng, &mut buf);
        for k in 0..d {
            y[(i, k)] = buf[k];
        }
    }
    (x, y, b)
}

fn main() {
    let quick = std::env::args().any(|a| a == "--quick");
    let ns: &[usize] = if quick { &[2000] } else { &[1000, 5000, 20000] };
    println!(
        "{:<28}{:>8}{:>8}{:>7}{:>11}{:>11}{:>10}{:>6}",
        "case", "params", "iters", "fev", "fit", "fit+SE", "|grad|", "conv"
    );
    for &(d, p) in &[(7usize, 5usize), (7, 6), (8, 5), (8, 6)] {
        for &n in ns {
            let (x, y, _) = sim(d, p, n, 11);
            let t = Instant::now();
            let f0 = EsagRegressor::new()
                .with_hessian(false)
                .fit(&x, &y)
                .unwrap();
            let t_fit = t.elapsed().as_secs_f64();
            let t = Instant::now();
            let f1 = EsagRegressor::new().fit(&x, &y).unwrap();
            let t_se = t.elapsed().as_secs_f64();
            println!(
                "{:<28}{:>8}{:>8}{:>7}{:>9.1}ms{:>9.1}ms{:>10.1e}{:>6}",
                format!("d={d} p={p} n={n}"),
                f0.n_params(),
                f0.n_iter,
                f0.n_fev,
                t_fit * 1e3,
                t_se * 1e3,
                f1.grad_norm,
                f1.converged
            );
            if (f0.log_likelihood - f1.log_likelihood).abs()
                > 1e-3 * f1.log_likelihood.abs().max(1.0)
            {
                println!(
                    "   ! loglik differs: {} vs {}",
                    f0.log_likelihood, f1.log_likelihood
                );
            }
        }
    }
}
