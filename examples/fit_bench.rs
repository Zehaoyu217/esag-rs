//! Single-core cost of full fits and penalized paths on the hard scenario
//! (γ data: `‖μ₀‖ = 2.5`, shape intercept of norm 1, five equicorrelated
//! covariates moving both `μ` and `V`; the truth of `hard_scenario.rs`),
//! through the public API only, so that the same file times any version of
//! the crate on the same data.
//!
//! `RAYON_NUM_THREADS=1 cargo run --release --example fit_bench`
//! env: `D=3,5,8` `N=2000` `R=6` (seeds) `SEED0=1500`
//! `SHAPES=gr,sp,lt[,gamma]` `PEN=1` (also the penalized paths).
//! Prints one line per fit and the median per cell.
use esag::distribution::Esag;
use esag::penalized::{Grouping, PenalizedEsag, Target};
use esag::regression::{EsagRegressor, Shape};
use nalgebra::DMatrix;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal};
use std::time::Instant;

fn env<T: std::str::FromStr>(key: &str, default: T) -> T {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn list(key: &str, default: &str) -> Vec<String> {
    std::env::var(key)
        .unwrap_or_else(|_| default.into())
        .split(',')
        .map(|v| v.trim().to_string())
        .collect()
}

/// Deterministic unit vector of length `n` (pseudo-random from `seed`).
fn unit(n: usize, seed: usize) -> Vec<f64> {
    let v: Vec<f64> = (0..n)
        .map(|i| ((i * 7919 + seed * 104_729) as f64 * 0.618_033_988_7).sin())
        .collect();
    let nrm = v.iter().map(|a| a * a).sum::<f64>().sqrt();
    v.iter().map(|a| a / nrm).collect()
}

/// The truth's mean direction, near `(1, …, 1)/√d`.
fn center(d: usize) -> Vec<f64> {
    let u = unit(d, 1);
    let m: Vec<f64> = (0..d).map(|k| 1.0 + 0.3 * u[k]).collect();
    let nm = m.iter().map(|a| a * a).sum::<f64>().sqrt();
    m.iter().map(|a| a / nm).collect()
}

/// The hard truth (`p = 5`), read as Yu & Huang's γ.
fn hard_truth(d: usize) -> DMatrix<f64> {
    let (bd, g) = (esag::dims::eta_len(d), esag::dims::gamma_len(d));
    let mut b = DMatrix::zeros(bd, 6);
    let m0 = center(d);
    for &(j, size, seed) in &[
        (0usize, 2.5, 1usize),
        (1, 0.6, 2),
        (2, 0.45, 3),
        (4, 0.15, 4),
    ] {
        let u = if j == 0 { m0.clone() } else { unit(d, seed) };
        for k in 0..d {
            b[(k, j)] = size * u[k];
        }
    }
    for &(j, size, seed) in &[
        (0usize, 1.0, 11usize),
        (1, 0.45, 12),
        (3, 0.45, 13),
        (4, 0.15, 14),
    ] {
        let w = unit(g, seed);
        for k in 0..g {
            b[(d + k, j)] = size * w[k];
        }
    }
    b
}

/// `n` draws with equicorrelated (ρ = 0.5) standard-normal covariates.
fn simulate(b: &DMatrix<f64>, n: usize, seed: u64) -> (DMatrix<f64>, DMatrix<f64>) {
    let (bd, p) = (b.nrows(), b.ncols() - 1);
    let d = esag::dims::dim_from_eta_len(bd).unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut x = DMatrix::zeros(n, p);
    for i in 0..n {
        let z0: f64 = StandardNormal.sample(&mut rng);
        for j in 0..p {
            let z: f64 = StandardNormal.sample(&mut rng);
            x[(i, j)] = 0.5f64.sqrt() * z0 + 0.5f64.sqrt() * z;
        }
    }
    let mut y = DMatrix::zeros(n, d);
    let (mut eta, mut buf) = (vec![0.0; bd], vec![0.0; d]);
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
    (x, y)
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn main() {
    let ds: Vec<usize> = list("D", "3,5,8")
        .iter()
        .map(|s| s.parse().unwrap())
        .collect();
    let (n, reps, seed0) = (env("N", 2000usize), env("R", 6u64), env("SEED0", 1500u64));
    let pen = env("PEN", 0usize) == 1;
    let shapes: Vec<(&str, Shape)> = list("SHAPES", "gr,sp,lt")
        .iter()
        .map(|s| match s.as_str() {
            "gr" => ("gamma-rot", Shape::GammaRot),
            "sp" => ("spectral", Shape::Spectral),
            "lt" => ("log-tangent", Shape::LogTangent),
            "gamma" => ("gamma", Shape::Gamma),
            o => panic!("unknown shape {o}"),
        })
        .collect();
    println!(
        "hard scenario, γ data, n = {n}, 5 covariates, seeds {seed0}..{}",
        seed0 + reps
    );
    for &d in &ds {
        let b = hard_truth(d);
        let data: Vec<(DMatrix<f64>, DMatrix<f64>)> =
            (seed0..seed0 + reps).map(|s| simulate(&b, n, s)).collect();
        for &(name, shape) in &shapes {
            let mut ms = vec![];
            for (k, (x, y)) in data.iter().enumerate() {
                let mut cfg = EsagRegressor::new().with_shape(shape).with_parallel(false);
                if shape != Shape::Gamma {
                    cfg = cfg.with_lt_reference(Some(center(d)));
                }
                let t = Instant::now();
                let f = cfg.fit(x, y).unwrap();
                let dt = t.elapsed().as_secs_f64() * 1e3;
                println!(
                    "fit d={d} {name:<12} seed {}: {dt:8.1} ms  ll {:.3}  conv {}",
                    seed0 + k as u64,
                    f.log_likelihood,
                    f.converged
                );
                ms.push(dt);
            }
            println!("fit d={d} {name:<12} median {:.1} ms", median(ms));
        }
        if !pen {
            continue;
        }
        for &(name, shape) in &shapes {
            if shape == Shape::Gamma {
                continue;
            }
            let mut presets = vec![("aLASSO", PenalizedEsag::adaptive_lasso())];
            if shape != Shape::LogTangent {
                presets.push((
                    "agLASSO eigen parts",
                    PenalizedEsag::adaptive_group_lasso().grouping(Grouping::EigenParts),
                ));
            }
            for (pname, pe) in presets {
                let mut ms = vec![];
                for (k, (x, y)) in data.iter().enumerate().take(reps.min(2) as usize) {
                    let pe = pe
                        .clone()
                        .shape(shape)
                        .target(Target::ShapeSlopes)
                        .lt_reference(center(d))
                        .bic()
                        .with_parallel(false);
                    let t = Instant::now();
                    let path = pe.fit(x, y).unwrap();
                    let dt = t.elapsed().as_secs_f64() * 1e3;
                    let best = path.best();
                    println!(
                        "pen d={d} {name:<12} {pname:<20} seed {}: {dt:8.1} ms  active {}  bic {:.3}",
                        seed0 + k as u64,
                        best.active.iter().filter(|a| **a).count(),
                        best.bic
                    );
                    ms.push(dt);
                }
                println!(
                    "pen d={d} {name:<12} {pname:<20} median {:.1} ms",
                    median(ms)
                );
            }
        }
    }
}
