//! Size and power of the bootstrap tests when V(x) is anisotropic.
//!
//! Data: n = 2000, 5 standard-normal covariates + const, coefficients from
//! `big::truth` with the null's coefficients set to zero (size) or scaled
//! by `C` (power). The shape intercepts are never zero, so no truth is
//! isotropic; `truth` prints how far the true V(x) is from the identity.
//!
//! Nulls: μ free of x1, shape free of x1, x5 no effect, and the shape
//! constant (free of every covariate: V(x) has the same anisotropic
//! eigenvalues at every x, its axes following μ(x)).
//! Log-tangent fits use the simulator's frame (reference `e_d`), so the
//! nulls tested are the nulls simulated.
//!
//! `cargo run --release --example aniso_study -- truth|size|power`
//! env: `D=3,5` `SHAPE=gamma|lt|gr` `R=300` `B=199` `C=0.2` `NULLS=mu,shape,x5,const`,
//! `FAST=1` for the score test (LM and M only).
#[path = "big_fit.rs"]
#[allow(dead_code)]
mod big;
use esag::distribution::Esag;
use esag::inference::{Restriction, hypothesis_test, score_test};
use esag::logtangent::LtFrame;
use esag::regression::{EsagRegressor, Shape};
use nalgebra::DMatrix;
use rayon::prelude::*;
use std::time::Instant;

const N: usize = 2000;
const P: usize = 5;

fn env<T: std::str::FromStr>(key: &str, default: T) -> T {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn label(null: &str) -> &'static str {
    match null {
        "mu" => "mu free of x1",
        "shape" => "shape free of x1",
        "x5" => "x5 no effect",
        "const" => "shape constant",
        _ => panic!("unknown null {null}"),
    }
}

/// Free-parameter mask of a null (`true` = free; column 0 is the constant).
fn null_mask(null: &str, d: usize) -> DMatrix<bool> {
    DMatrix::from_fn(esag::dims::eta_len(d), P + 1, |k, j| match null {
        "mu" => !(k < d && j == 1),
        "shape" => !(k >= d && j == 1),
        "x5" => j != 5,
        "const" => !(k >= d && j >= 1),
        _ => panic!("unknown null {null}"),
    })
}

/// True law at covariate row `x`.
fn true_dist(b: &DMatrix<f64>, x: &[f64], d: usize, shape: Shape) -> Esag {
    let eta: Vec<f64> = (0..b.nrows())
        .map(|k| b[(k, 0)] + (0..P).map(|j| b[(k, j + 1)] * x[j]).sum::<f64>())
        .collect();
    esag::regression::shape_distribution(shape, Some(&LtFrame::standard(d)), &eta).unwrap()
}

fn q(v: &mut [f64], p: f64) -> f64 {
    v.sort_by(f64::total_cmp);
    v[((v.len() - 1) as f64 * p).round() as usize]
}

/// Largest/smallest shape eigenvalue of V(x) and the geodesic distance
/// ‖log V(x)‖_F from the identity, over the sample's covariates.
fn truth_report(ds: &[usize], shapes: &[Shape]) {
    println!("true V(x) over n = {N} covariate draws: quantiles 5% / 50% / 95%");
    for &d in ds {
        for &shape in shapes {
            for (what, b) in [
                ("simulated truth", big::truth(d, P)),
                ("shape-constant null", masked(d, "const", 0.0)),
            ] {
                let (x, _) = big::sim_from(&b, N, 1, shape);
                let (mut ratio, mut dist) = (vec![], vec![]);
                for i in 0..N {
                    let row: Vec<f64> = x.row(i).iter().copied().collect();
                    let l = true_dist(&b, &row, d, shape).frame().lambda.clone();
                    let (lo, hi) = l
                        .iter()
                        .fold((f64::INFINITY, 0.0f64), |(a, c), &v| (a.min(v), c.max(v)));
                    ratio.push(hi / lo);
                    dist.push(l.iter().map(|v| v.ln().powi(2)).sum::<f64>().sqrt());
                }
                println!(
                    "d={d} {:<10} {what:<19} λmax/λmin {:6.2} / {:6.2} / {:6.2}   ‖log V‖ {:.2} / {:.2} / {:.2}",
                    format!("{shape:?}"),
                    q(&mut ratio, 0.05),
                    q(&mut ratio, 0.5),
                    q(&mut ratio, 0.95),
                    q(&mut dist, 0.05),
                    q(&mut dist, 0.5),
                    q(&mut dist, 0.95),
                );
            }
        }
    }
}

/// `big::truth` with the null's coefficients scaled by `c`.
fn masked(d: usize, null: &str, c: f64) -> DMatrix<f64> {
    let full = big::truth(d, P);
    let mask = null_mask(null, d);
    DMatrix::from_fn(full.nrows(), full.ncols(), |k, j| {
        if mask[(k, j)] {
            full[(k, j)]
        } else {
            c * full[(k, j)]
        }
    })
}

fn study(ds: &[usize], shapes: &[Shape], nulls: &[String], c: f64) {
    let (reps, nb): (usize, usize) = (env("R", 300), env("B", 199));
    let fast = std::env::var("FAST").is_ok();
    println!(
        "rejection rates at 5%, n = {N}, p = {P} + const, B = {nb}, {reps} replications, {} threads; effect scale c = {c}{}",
        rayon::current_num_threads(),
        if fast { "; score test" } else { "" }
    );
    for &d in ds {
        for &shape in shapes {
            for null in nulls {
                let b = masked(d, null, c);
                let mask = null_mask(null, d);
                let t = Instant::now();
                let out: Vec<([f64; 5], usize)> = (0..reps)
                    .into_par_iter()
                    .map(|r| {
                        let (x, y) = big::sim_from(&b, N, 10_000 + r as u64, shape);
                        let mut cfg = EsagRegressor::new().with_shape(shape);
                        if shape == Shape::LogTangent {
                            let mut m = vec![0.0; d];
                            m[d - 1] = 1.0;
                            cfg = cfg.with_lt_reference(Some(m));
                        }
                        let fit = cfg.fit(&x, &y).unwrap();
                        let test = if fast { score_test } else { hypothesis_test };
                        let res = test(
                            &fit,
                            &x,
                            &y,
                            Restriction::Custom(mask.clone()),
                            nb,
                            0,
                            r as u64,
                        )
                        .unwrap();
                        (
                            [res.p_roc, res.p_d, res.p_m, res.p_lr, res.p_lm],
                            res.n_failed,
                        )
                    })
                    .collect();
                // NaN: a statistic the score test does not bootstrap
                let rate = |k: usize| {
                    if out.iter().all(|(p, _)| p[k].is_nan()) {
                        f64::NAN
                    } else {
                        out.iter().filter(|(p, _)| p[k] < 0.05).count() as f64 / reps as f64
                    }
                };
                let failed: usize = out.iter().map(|o| o.1).sum();
                println!(
                    "d={d} {:<10} {:<17} RoC {:.3}  D {:.3}  M {:.3}  LR {:.3}  LM {:.3}   (failed refits {failed}; {:.0} s)",
                    format!("{shape:?}"),
                    label(null),
                    rate(0),
                    rate(1),
                    rate(2),
                    rate(3),
                    rate(4),
                    t.elapsed().as_secs_f64()
                );
            }
        }
    }
}

fn main() {
    let mode = std::env::args().nth(1).unwrap_or_else(|| "truth".into());
    let ds: Vec<usize> = std::env::var("D")
        .unwrap_or_else(|_| "3,5".into())
        .split(',')
        .map(|v| v.parse().unwrap())
        .collect();
    let shapes = match std::env::var("SHAPE").as_deref() {
        Ok("gamma") => vec![Shape::Gamma],
        Ok("lt") => vec![Shape::LogTangent],
        Ok("gr") | Ok("gamma-rot") => vec![Shape::GammaRot],
        _ => vec![Shape::Gamma, Shape::LogTangent],
    };
    let nulls: Vec<String> = std::env::var("NULLS")
        .unwrap_or_else(|_| "mu,shape,x5,const".into())
        .split(',')
        .map(str::to_string)
        .collect();
    match mode.as_str() {
        "truth" => truth_report(&ds, &shapes),
        "size" => study(&ds, &shapes, &nulls, 0.0),
        "power" => study(&ds, &shapes, &nulls, env("C", 0.2)),
        _ => panic!("mode: truth | size | power"),
    }
}
