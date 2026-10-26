//! A hard regression scenario: both `μ` and `V` depend strongly on the
//! covariates, the covariates are correlated, the concentration is
//! moderate, and the effects are sparse.
//!
//! Truth (`p = 5`, equicorrelated standard-normal covariates, ρ = 0.5):
//! `‖μ₀‖ = 2.5` along a generic direction near `(1, …, 1)/√d` (far from the
//! pole of the γ frame and from the log-tangent antipode), a shape intercept
//! of norm 1; `x1` moves `μ` (0.6) and the shape (0.45), `x2` only `μ`
//! (0.45), `x3` only the shape (0.45), `x4` both weakly (0.15), `x5`
//! nothing. Log-tangent data and fits use the frame with reference
//! `center(d)`.
//!
//! `cargo run --release --example hard_scenario -- fit|tests|lasso|recovery`
//! env: `D=3,5` `SHAPE=gamma|lt|gr` `NS=200,500,2000` `R` (replications) `B`
//! (bootstrap) `FAST=1` (score tests only) `TRUTH=gamma|lt` (the shape the
//! data come from in `fit`; default: the fitted shape).
use esag::distribution::Esag;
use esag::inference::{Restriction, hypothesis_test, score_test};
use esag::logtangent::LtFrame;
use esag::penalized::{Grouping, PenalizedEsag};
use esag::regression::{EsagRegressor, EsagRegressorFit, Shape};
use nalgebra::DMatrix;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal};
use rayon::prelude::*;
use std::time::Instant;

const P: usize = 5;

fn env<T: std::str::FromStr>(key: &str, default: T) -> T {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn list(key: &str, default: &str) -> Vec<usize> {
    std::env::var(key)
        .unwrap_or_else(|_| default.into())
        .split(',')
        .map(|v| v.parse().unwrap())
        .collect()
}

/// The shape the data come from (`TRUTH=gamma|lt`); default: the fitted one.
fn truth_shape() -> Option<Shape> {
    match std::env::var("TRUTH").as_deref() {
        Ok("gamma") => Some(Shape::Gamma),
        Ok("lt") => Some(Shape::LogTangent),
        _ => None,
    }
}

fn shapes() -> Vec<Shape> {
    match std::env::var("SHAPE").as_deref() {
        Ok("gamma") => vec![Shape::Gamma],
        Ok("lt") => vec![Shape::LogTangent],
        Ok("gr") | Ok("gamma-rot") => vec![Shape::GammaRot],
        _ => vec![Shape::LogTangent, Shape::Gamma],
    }
}

/// Deterministic unit vector of length `n`.
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

fn truth(d: usize) -> DMatrix<f64> {
    let (bd, g) = (esag::dims::eta_len(d), esag::dims::gamma_len(d));
    let mut b = DMatrix::zeros(bd, P + 1);
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

fn eta_at(b: &DMatrix<f64>, x: &[f64]) -> Vec<f64> {
    (0..b.nrows())
        .map(|k| b[(k, 0)] + (0..P).map(|j| b[(k, j + 1)] * x[j]).sum::<f64>())
        .collect()
}

fn true_dist(b: &DMatrix<f64>, x: &[f64], d: usize, shape: Shape) -> Esag {
    let eta = eta_at(b, x);
    esag::regression::shape_distribution(shape, Some(&LtFrame::new(&center(d))), &eta).unwrap()
}

fn covariates(n: usize, rng: &mut ChaCha8Rng) -> DMatrix<f64> {
    let rho: f64 = 0.5;
    let mut x = DMatrix::zeros(n, P);
    for i in 0..n {
        let z0: f64 = StandardNormal.sample(rng);
        for j in 0..P {
            let z: f64 = StandardNormal.sample(rng);
            x[(i, j)] = rho.sqrt() * z0 + (1.0 - rho).sqrt() * z;
        }
    }
    x
}

fn simulate(d: usize, n: usize, seed: u64, shape: Shape) -> (DMatrix<f64>, DMatrix<f64>) {
    let b = truth(d);
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let x = covariates(n, &mut rng);
    let mut y = DMatrix::zeros(n, d);
    let mut buf = vec![0.0; d];
    for i in 0..n {
        let row: Vec<f64> = x.row(i).iter().copied().collect();
        let dist = true_dist(&b, &row, d, shape);
        dist.sample_one(&dist.frame(), &mut rng, &mut buf);
        for k in 0..d {
            y[(i, k)] = buf[k];
        }
    }
    (x, y)
}

fn config(d: usize, shape: Shape) -> EsagRegressor {
    let cfg = EsagRegressor::new().with_shape(shape).with_parallel(false);
    if shape == Shape::LogTangent {
        cfg.with_lt_reference(Some(center(d)))
    } else {
        cfg
    }
}

/// Affine-invariant distance ‖log(V^{-1/2} W V^{-1/2})‖_F.
fn geodesic(v: &DMatrix<f64>, w: &DMatrix<f64>) -> f64 {
    let li = v.clone().cholesky().unwrap().l().try_inverse().unwrap();
    let m = &li * w * li.transpose();
    let m = (&m + m.transpose()) * 0.5;
    nalgebra::SymmetricEigen::new(m)
        .eigenvalues
        .iter()
        .map(|e| e.ln().powi(2))
        .sum::<f64>()
        .sqrt()
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

/// One-core fits: convergence, time, and V(x) against the truth on 200
/// fresh covariate points.
fn fit_study() {
    let reps = env("R", 12usize);
    println!(
        "one-core fits with standard errors, {reps} replications per cell; V error = mean geodesic distance of V̂(x) from V(x) over 200 covariate points, angle = mean error of the mean direction (degrees)"
    );
    for d in list("D", "3,5,8") {
        for &shape in &shapes() {
            for n in list("NS", "200,500,2000") {
                let b = truth(d);
                let data_shape = truth_shape().unwrap_or(shape);
                let mut rng = ChaCha8Rng::seed_from_u64(99);
                let x_test = covariates(200, &mut rng);
                // how much the true V(x) moves over the test points
                let v0 = true_dist(&b, &[0.0; P], d, data_shape).covariance();
                let spread = (0..200)
                    .map(|i| {
                        let row: Vec<f64> = x_test.row(i).iter().copied().collect();
                        geodesic(&v0, &true_dist(&b, &row, d, data_shape).covariance())
                    })
                    .sum::<f64>()
                    / 200.0;
                let rows: Vec<(bool, usize, usize, f64, f64, f64)> = (0..reps)
                    .map(|r| {
                        let (x, y) = simulate(d, n, 1000 + r as u64, data_shape);
                        let t = Instant::now();
                        let fit = config(d, shape).fit(&x, &y).unwrap();
                        let ms = t.elapsed().as_secs_f64() * 1e3;
                        let vhat = fit.predict_covariance(&x_test).unwrap();
                        let mhat = fit.predict(&x_test).unwrap();
                        let (mut gv, mut ang) = (0.0, 0.0);
                        for (i, vh) in vhat.iter().enumerate() {
                            let row: Vec<f64> = x_test.row(i).iter().copied().collect();
                            let td = true_dist(&b, &row, d, data_shape);
                            gv += geodesic(&td.covariance(), vh);
                            let mh: Vec<f64> = mhat.row(i).iter().copied().collect();
                            let c: f64 =
                                mh.iter().zip(td.mean_direction()).map(|(a, b)| a * b).sum();
                            ang += c.clamp(-1.0, 1.0).acos().to_degrees();
                        }
                        (
                            fit.converged,
                            if fit.non_regular {
                                usize::MAX
                            } else {
                                fit.singular_obs.len()
                            },
                            fit.degenerate_obs.len(),
                            ms,
                            gv / 200.0,
                            ang / 200.0,
                        )
                    })
                    .collect();
                let conv = rows.iter().filter(|r| r.0).count();
                // converged at a maximum on gamma's singular set
                let non_regular = rows.iter().filter(|r| r.0 && r.1 == usize::MAX).count();
                let pinned = rows.iter().filter(|r| !r.0 && r.1 > 0).count();
                let unbounded = rows.iter().filter(|r| r.2 > 0).count();
                let mut ms: Vec<f64> = rows.iter().map(|r| r.3).collect();
                let mut gv: Vec<f64> = rows.iter().map(|r| r.4).collect();
                let mut ang: Vec<f64> = rows.iter().map(|r| r.5).collect();
                let max_ms = ms.iter().fold(0.0f64, |a, &b| a.max(b));
                println!(
                    "d={d} {:<10} n={n:>4} K={:>3}: converged {conv:>2}/{reps} (non-regular {non_regular}; unconverged: trapped {pinned}, unbounded {unbounded}); time median {:7.1} ms, max {:7.1} ms; V error {:.3} (V(x) moves {spread:.2} from V(0)), angle {:.2}°",
                    if data_shape == shape {
                        format!("{shape:?}")
                    } else {
                        format!("{shape:?} on {data_shape:?} data")
                    },
                    esag::dims::eta_len(d) * (P + 1),
                    median(&mut ms),
                    max_ms,
                    median(&mut gv),
                    median(&mut ang),
                );
            }
        }
    }
}

/// Rejection rates at 5% for three true and three false nulls, on the same
/// datasets.
fn test_study() {
    let (reps, nb, n) = (env("R", 200usize), env("B", 199usize), env("N", 2000usize));
    let fast = std::env::var("FAST").is_ok();
    let nulls: Vec<(&str, bool, Restriction)> = vec![
        ("mu free of x3", true, Restriction::MuOfFeature(2)),
        ("shape free of x2", true, Restriction::GammaOfFeature(1)),
        ("x5 no effect", true, Restriction::Feature(4)),
        ("shape constant", false, Restriction::ConstantShape),
        ("shape free of x1", false, Restriction::GammaOfFeature(0)),
        ("x4 no effect (weak)", false, Restriction::Feature(3)),
    ];
    println!(
        "rejection rates at 5%, n = {n}, B = {nb}, {reps} replications, {} threads{}",
        rayon::current_num_threads(),
        if fast { ", score tests" } else { "" }
    );
    for d in list("D", "3,5") {
        for &shape in &shapes() {
            let t = Instant::now();
            let out: Vec<Vec<([f64; 5], usize)>> = (0..reps)
                .into_par_iter()
                .map(|r| {
                    let (x, y) = simulate(d, n, 5000 + r as u64, shape);
                    let fit = config(d, shape).fit(&x, &y).unwrap();
                    nulls
                        .iter()
                        .map(|(_, _, null)| {
                            let test = if fast { score_test } else { hypothesis_test };
                            let res = test(&fit, &x, &y, null.clone(), nb, 0, r as u64).unwrap();
                            (
                                [res.p_roc, res.p_d, res.p_m, res.p_lr, res.p_lm],
                                res.n_unconverged,
                            )
                        })
                        .collect()
                })
                .collect();
            for (k, (label, is_true, _)) in nulls.iter().enumerate() {
                let rate = |s: usize| {
                    if out.iter().all(|o| o[k].0[s].is_nan()) {
                        f64::NAN
                    } else {
                        out.iter().filter(|o| o[k].0[s] < 0.05).count() as f64 / reps as f64
                    }
                };
                let unconv: usize = out.iter().map(|o| o[k].1).sum();
                println!(
                    "d={d} {:<10} {:<20} {:<5} RoC {:.3}  D {:.3}  M {:.3}  LR {:.3}  LM {:.3}  (unconverged refits {unconv})",
                    format!("{shape:?}"),
                    label,
                    if *is_true { "size" } else { "power" },
                    rate(0),
                    rate(1),
                    rate(2),
                    rate(3),
                    rate(4),
                );
            }
            println!("   ({:.0} s)", t.elapsed().as_secs_f64());
        }
    }
}

/// Support recovery of the LASSO family per covariate part (does `x_j`
/// move `μ`? the shape?), against the truth (μ: x1, x2, x4; shape: x1,
/// x3, x4).
fn lasso_study() {
    let reps = env("R", 40usize);
    let truth_mu = [true, true, false, true, false];
    let truth_sh = [true, false, true, true, false];
    println!(
        "LASSO family, {} selection, {reps} replications; found of the 4 strong and 2 weak (x4) true covariate parts, FP of the 4 null parts, exact = whole pattern right; time per path",
        if std::env::var("SELECT").as_deref() == Ok("cv") {
            "5-fold CV (held-out log-likelihood)"
        } else {
            "BIC at relaxed refits"
        }
    );
    for d in list("D", "3,5") {
        for n in list("NS", "300,2000") {
            let presets: Vec<(&str, PenalizedEsag)> = vec![
                ("lasso", PenalizedEsag::lasso()),
                ("adaptive lasso", PenalizedEsag::adaptive_lasso()),
                (
                    "group lasso",
                    PenalizedEsag::group_lasso().grouping(Grouping::CovariatePart),
                ),
                (
                    "adaptive group lasso",
                    PenalizedEsag::adaptive_group_lasso().grouping(Grouping::CovariatePart),
                ),
                (
                    "sparse group lasso",
                    PenalizedEsag::sparse_group_lasso(0.5).grouping(Grouping::CovariatePart),
                ),
            ];
            let cv = std::env::var("SELECT").as_deref() == Ok("cv");
            for (name, pe) in presets {
                let pe = pe.lt_reference(center(d)).with_parallel(false);
                let pe = if cv { pe.cv(5, 1) } else { pe };
                let out: Vec<(usize, usize, usize, bool, f64)> = (0..reps)
                    .into_par_iter()
                    .map(|r| {
                        let (x, y) = simulate(d, n, 7000 + r as u64, Shape::LogTangent);
                        let t = Instant::now();
                        let path = pe.fit(&x, &y).unwrap();
                        let ms = t.elapsed().as_secs_f64() * 1e3;
                        let b = &path.best().coef;
                        let act = |j: usize, rows: std::ops::Range<usize>| {
                            rows.into_iter().any(|k| b[(k, j + 1)] != 0.0)
                        };
                        // strong parts: x1 (both), x2 (mean), x3 (shape); weak: x4
                        let (mut strong, mut weak, mut fp, mut exact) = (0, 0, 0, true);
                        for j in 0..P {
                            let (am, ash) = (act(j, 0..d), act(j, d..b.nrows()));
                            let hits =
                                usize::from(am && truth_mu[j]) + usize::from(ash && truth_sh[j]);
                            if j == 3 {
                                weak += hits;
                            } else {
                                strong += hits;
                            }
                            fp +=
                                usize::from(am && !truth_mu[j]) + usize::from(ash && !truth_sh[j]);
                            exact &= am == truth_mu[j] && ash == truth_sh[j];
                        }
                        (strong, weak, fp, exact, ms)
                    })
                    .collect();
                let r = reps as f64;
                let strong = out.iter().map(|o| o.0).sum::<usize>() as f64 / r;
                let weak = out.iter().map(|o| o.1).sum::<usize>() as f64 / r;
                let fp = out.iter().map(|o| o.2).sum::<usize>() as f64 / r;
                let exact = out.iter().filter(|o| o.3).count() as f64 / r;
                let mut ms: Vec<f64> = out.iter().map(|o| o.4).collect();
                println!(
                    "d={d} n={n:>4} {name:<21} strong {strong:.2}/4  weak {weak:.2}/2  FP {fp:.2}/4  exact {exact:.2}  time {:7.0} ms",
                    median(&mut ms)
                );
            }
        }
    }
}

/// How well each estimator recovers `V(x)` and `μ(x)`: the full model, the
/// model with only the true covariates (an oracle), a constant shape, the
/// adaptive LASSO's relaxed refit, and the γ full model. Mean geodesic
/// error of `V̂(x)` over 200 fresh covariate points (median over
/// replications), against how anisotropic `V(x)` is and how much it moves
/// with `x`. `TRUTH=gamma|lt` picks the data's shape (default `lt`).
fn recovery_study() {
    let reps = env("R", 12usize);
    let data_shape = truth_shape().unwrap_or(Shape::LogTangent);
    println!(
        "V(x) and mu(x) recovery, {data_shape:?} data, {reps} replications; V error = mean geodesic distance of V̂(x) from V(x) over 200 covariate points, angle = mean error of the mean direction (degrees)"
    );
    for d in list("D", "3,5") {
        let b = truth(d);
        let bd = b.nrows();
        let mut rng = ChaCha8Rng::seed_from_u64(99);
        let x_test = covariates(200, &mut rng);
        let rows: Vec<Vec<f64>> = (0..200)
            .map(|i| x_test.row(i).iter().copied().collect())
            .collect();
        let truth_v: Vec<DMatrix<f64>> = rows
            .iter()
            .map(|r| true_dist(&b, r, d, data_shape).covariance())
            .collect();
        let v0 = true_dist(&b, &[0.0; P], d, data_shape).covariance();
        let eye = DMatrix::<f64>::identity(d, d);
        let aniso = truth_v.iter().map(|v| geodesic(&eye, v)).sum::<f64>() / 200.0;
        let spread = truth_v.iter().map(|v| geodesic(&v0, v)).sum::<f64>() / 200.0;
        println!(
            "d={d}: V(x) is {aniso:.2} from isotropy and moves {spread:.2} from V(0) on average"
        );
        // free coefficients: all; the true support; shape intercept only
        let oracle = DMatrix::from_fn(bd, P + 1, |k, j| {
            j == 0
                || if k < d {
                    [1, 2, 4].contains(&j)
                } else {
                    [1, 3, 4].contains(&j)
                }
        });
        let constant = DMatrix::from_fn(bd, P + 1, |k, j| k < d || j == 0);
        for n in list("NS", "500,2000") {
            let out: Vec<[(f64, f64); 5]> = (0..reps)
                .into_par_iter()
                .map(|r| {
                    let (x, y) = simulate(d, n, 3000 + r as u64, data_shape);
                    let score = |fit: &EsagRegressorFit| {
                        let vhat = fit.predict_covariance(&x_test).unwrap();
                        let mhat = fit.predict(&x_test).unwrap();
                        let (mut gv, mut ang) = (0.0, 0.0);
                        for i in 0..200 {
                            gv += geodesic(&truth_v[i], &vhat[i]);
                            let td = true_dist(&b, &rows[i], d, data_shape);
                            let c: f64 = mhat
                                .row(i)
                                .iter()
                                .zip(td.mean_direction())
                                .map(|(a, b)| a * b)
                                .sum();
                            ang += c.clamp(-1.0, 1.0).acos().to_degrees();
                        }
                        (gv / 200.0, ang / 200.0)
                    };
                    let lt = config(d, Shape::LogTangent);
                    let full = lt.clone().fit(&x, &y).unwrap();
                    let orc = lt
                        .clone()
                        .with_mask(Some(oracle.clone()))
                        .fit(&x, &y)
                        .unwrap();
                    let cst = lt
                        .clone()
                        .with_mask(Some(constant.clone()))
                        .fit(&x, &y)
                        .unwrap();
                    let path = PenalizedEsag::adaptive_lasso()
                        .lt_reference(center(d))
                        .with_parallel(false)
                        .fit(&x, &y)
                        .unwrap();
                    let gam = config(d, Shape::Gamma).fit(&x, &y).unwrap();
                    [
                        score(&full),
                        score(&orc),
                        score(&cst),
                        score(path.relaxed.as_ref().unwrap()),
                        score(&gam),
                    ]
                })
                .collect();
            let names = [
                "log-tangent, full model",
                "log-tangent, true covariates only",
                "log-tangent, constant shape",
                "log-tangent, adaptive LASSO (relaxed)",
                "gamma, full model",
            ];
            for (k, name) in names.iter().enumerate() {
                let mut gv: Vec<f64> = out.iter().map(|o| o[k].0).collect();
                let mut ang: Vec<f64> = out.iter().map(|o| o[k].1).collect();
                println!(
                    "  n={n:>4} {name:<38} V error {:.3}  angle {:.2}°",
                    median(&mut gv),
                    median(&mut ang)
                );
            }
        }
    }
}

fn main() {
    match std::env::args().nth(1).as_deref().unwrap_or("fit") {
        "fit" => fit_study(),
        "tests" => test_study(),
        "lasso" => lasso_study(),
        "recovery" => recovery_study(),
        _ => panic!("mode: fit | tests | lasso | recovery"),
    }
}
