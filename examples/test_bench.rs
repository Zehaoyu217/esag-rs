//! Bootstrap hypothesis tests at scale: B = 500, n = 2000, 5 covariates.
//! The data are anisotropic (V(x) ≠ I; `examples/aniso_study.rs truth`);
//! the nulls are the shape free of every covariate, μ free of x1, the shape
//! free of x1, and no effect of x5. Each null is tested twice: the full test
//! (RoC, D, M, LR, LM) and the score test (restricted refits only: LM, M).
//! `cargo run --release --example test_bench [-- d1,d2,...]`
use esag::inference::{Restriction, hypothesis_test, score_test};
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
            Ok("gr") | Ok("gamma-rot") => vec![Shape::GammaRot],
            _ => vec![Shape::Gamma, Shape::LogTangent],
        };
        for shape in shapes {
            let (x, y, _) = big::sim_shape(d, p, n, 11, shape);
            let t = Instant::now();
            let fit = EsagRegressor::new().with_shape(shape).fit(&x, &y).unwrap();
            let t_fit = t.elapsed().as_secs_f64();
            let only = std::env::var("ONLY").ok();
            for (label, r) in [
                ("shape constant", Restriction::ConstantShape),
                ("mu free of x1", Restriction::MuOfFeature(0)),
                ("shape free of x1", Restriction::GammaOfFeature(0)),
                ("x5 no effect", Restriction::Feature(4)),
            ] {
                if only.as_deref().is_some_and(|o| !label.starts_with(o)) {
                    continue;
                }
                let t = Instant::now();
                let res = hypothesis_test(&fit, &x, &y, r.clone(), b, 0, 7).unwrap();
                let t_full = t.elapsed().as_secs_f64();
                let t = Instant::now();
                let fast = score_test(&fit, &x, &y, r, b, 0, 7).unwrap();
                let t_score = t.elapsed().as_secs_f64();
                println!(
                    "d={d} {shape:?} {label:<17} B={b}: full {t_full:6.2} s, score {t_score:6.2} s  (fit {t_fit:.2} s; failed {}; p RoC {:.3} D {:.3} M {:.3} LR {:.3} LM {:.3})",
                    res.n_failed + fast.n_failed,
                    res.p_roc,
                    res.p_d,
                    res.p_m,
                    res.p_lr,
                    res.p_lm
                );
                // the same restricted refits (NaN: a singular score information)
                let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
                assert_eq!(bits(&res.boot_lm), bits(&fast.boot_lm));
            }
        }
    }
}
