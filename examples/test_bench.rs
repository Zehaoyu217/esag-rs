//! Bootstrap hypothesis tests at scale: B = 500, n = 2000, 5 covariates.
//! `cargo run --release --example test_bench [-- d1,d2,...]`
use esag::inference::{Restriction, hypothesis_test};
use esag::regression::{EsagRegressor, Shape};
use std::time::Instant;
#[path = "big_fit.rs"]
#[allow(dead_code)]
mod big;

fn main() {
    let ds: Vec<usize> = std::env::args()
        .nth(1)
        .map(|a| a.split(',').map(|v| v.parse().unwrap()).collect())
        .unwrap_or_else(|| vec![3, 5, 8]);
    let (n, p) = (2000usize, 5usize);
    let b: usize = std::env::var("B")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(500);
    for &d in &ds {
        let shapes: Vec<Shape> = match std::env::var("SHAPE").as_deref() {
            Ok("gamma") => vec![Shape::Gamma],
            Ok("lt") => vec![Shape::LogTangent],
            _ => vec![Shape::Gamma, Shape::LogTangent],
        };
        for shape in shapes {
            let (x, y, _) = big::sim_shape(d, p, n, 11, shape);
            let t = Instant::now();
            let fit = EsagRegressor::new().with_shape(shape).fit(&x, &y).unwrap();
            let t_fit = t.elapsed().as_secs_f64();
            let only = std::env::var("ONLY").ok();
            for (label, r) in [
                ("isotropy", Restriction::Isotropy),
                ("mu free of x1", Restriction::MuOfFeature(0)),
                ("shape free of x1", Restriction::GammaOfFeature(0)),
                ("x5 no effect", Restriction::Feature(4)),
            ] {
                if only.as_deref().is_some_and(|o| !label.starts_with(o)) {
                    continue;
                }
                let t = Instant::now();
                let res = hypothesis_test(&fit, &x, &y, r, b, 0, 7).unwrap();
                println!(
                    "d={d} {shape:?} {label:<17} B={b}: {:6.2} s  (fit {:.2} s; failed {}; p RoC {:.3} D {:.3} M {:.3} LR {:.3})",
                    t.elapsed().as_secs_f64(),
                    t_fit,
                    res.n_failed,
                    res.p_roc,
                    res.p_d,
                    res.p_m,
                    res.p_lr
                );
            }
        }
    }
}
