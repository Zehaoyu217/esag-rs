//! Large regression fits: d = 7–8 responses, 5–6 covariates.
//! `cargo run --release --example big_fit [-- --quick]`

use esag::distribution::Esag;
use esag::regression::{EsagRegressor, Shape};
use nalgebra::DMatrix;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal};
use std::time::Instant;

/// Simulate `n` responses on S^{d-1} with `p` standard-normal covariates.
/// Intercept: ‖μ‖ ≈ 4, moderate anisotropy; slopes of size ~0.3.
pub fn sim(d: usize, p: usize, n: usize, seed: u64) -> (DMatrix<f64>, DMatrix<f64>, DMatrix<f64>) {
    sim_shape(d, p, n, seed, Shape::Gamma)
}

/// As [`sim`], with the shape rows read in either parameterization (for
/// `LogTangent` the frame's reference direction is `e_d`).
pub fn sim_shape(
    d: usize,
    p: usize,
    n: usize,
    seed: u64,
    shape: Shape,
) -> (DMatrix<f64>, DMatrix<f64>, DMatrix<f64>) {
    let lt = esag::logtangent::LtFrame::standard(d);
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
        let dist = match shape {
            Shape::Gamma => Esag::from_eta(&eta).unwrap(),
            Shape::LogTangent => {
                let v = esag::logtangent::covariance(&lt, &eta[..d], &eta[d..]);
                Esag::from_covariance(&eta[..d], &v).unwrap()
            }
        };
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
        "{:<22}{:>7} | {:>27} | {:>27}",
        "case", "params", "gamma: fit+SE  iters conv", "log-tangent: fit+SE iters conv"
    );
    for (truth, label) in [
        (Shape::Gamma, "data from the gamma model"),
        (Shape::LogTangent, "data from the log-tangent model"),
    ] {
        println!("-- {label}");
        for &(d, p) in &[(7usize, 5usize), (7, 6), (8, 5), (8, 6)] {
            for &n in ns {
                let (x, y, _) = sim_shape(d, p, n, 11, truth);
                let mut cells = vec![];
                let mut lls = vec![];
                for shape in [Shape::Gamma, Shape::LogTangent] {
                    let t = Instant::now();
                    let f = EsagRegressor::new().with_shape(shape).fit(&x, &y).unwrap();
                    let dt = t.elapsed().as_secs_f64();
                    cells.push(format!(
                        "{:>10.1}ms {:>5} {:>5}{:>5}",
                        dt * 1e3,
                        f.n_iter,
                        if f.converged { "yes" } else { "NO" },
                        ""
                    ));
                    lls.push(f.log_likelihood);
                }
                println!(
                    "{:<22}{:>7} | {} | {}   ll γ {:.2} / lt {:.2}",
                    format!("d={d} p={p} n={n}"),
                    esag::dims::eta_len(d) * (p + 1),
                    cells[0],
                    cells[1],
                    lls[0],
                    lls[1]
                );
            }
        }
    }
}
