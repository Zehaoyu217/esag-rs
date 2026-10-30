//! Project 3 in the paper's setting (d = 6, n = 100): the six methods as
//! coded (γ, reproduction mode) and the corrected eigen-gap penalty, on one
//! replicate of experiment 1. `cargo run --release --example project3_paper`
use esag::distribution::Esag;
use esag::penalized::{PenalizedEsag, selection_rates};
use nalgebra::DMatrix;
use rand::SeedableRng;
use std::time::Instant;

fn main() {
    let mu = [2.0, -5.0, 3.0, -4.0, 5.0, 6.0];
    let gamma = [
        5.0, 2.0, 3.0, 0.0, 5.0, 1.0, 0.0, 1.0, 5.0, 0.0, 3.0, 2.0, 0.0, 0.0,
    ];
    let truth = Esag::new(&mu, &gamma).unwrap();
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(1);
    let n = 100;
    let y = truth.sample(n, &mut rng);
    let x = DMatrix::<f64>::zeros(n, 0);
    let v_true = truth.covariance();
    println!(
        "{:<12}{:>9}{:>8}{:>7}{:>7}{:>7}{:>10}{:>10}",
        "method", "time", "λ(BIC)", "CR", "SE", "SP", "|μ̂−μ|", "‖V̂−V‖F"
    );
    for name in [
        "LASSO",
        "aLASSO",
        "RoC-LASSO",
        "GLASSO",
        "aGLASSO",
        "RoC-GLASSO",
    ] {
        let t = Instant::now();
        let path = PenalizedEsag::paper(name).unwrap().fit(&x, &y).unwrap();
        let dt = t.elapsed().as_secs_f64();
        let b = path.best();
        let thr = b.coef_thresholded.as_ref().unwrap();
        let g: Vec<f64> = thr.column(0).iter().skip(6).copied().collect();
        let (cr, se, sp) = selection_rates(&gamma, &g);
        let m: Vec<f64> = thr.column(0).iter().take(6).copied().collect();
        let mu_err = m
            .iter()
            .zip(&mu)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        let v_err = (Esag::new(&m, &g).unwrap().covariance() - &v_true).norm();
        println!(
            "{name:<12}{:>8.0}ms{:>8}{:>7.2}{:>7.2}{:>7.2}{:>10.3}{:>10.3}",
            dt * 1e3,
            b.lambda,
            cr,
            se,
            sp,
            mu_err,
            v_err
        );
    }
    let t = Instant::now();
    let path = PenalizedEsag::eigen_gap().fit(&x, &y).unwrap();
    let b = path.best();
    let fit = b.coef.column(0).iter().copied().collect::<Vec<f64>>();
    let lt = path.initial.lt_frame().unwrap();
    let v = esag::logtangent::covariance(&lt, &fit[..6], &fit[6..]);
    let mu_err = fit[..6]
        .iter()
        .zip(&mu)
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt();
    println!(
        "{:<12}{:>8.0}ms{:>8.2}   ties {:?}{:>10.3}{:>10.3}",
        "EigenGap",
        t.elapsed().as_secs_f64() * 1e3,
        b.lambda,
        b.active
            .iter()
            .map(|a| if *a { '|' } else { '=' })
            .collect::<String>(),
        mu_err,
        (v - &v_true).norm()
    );
}
