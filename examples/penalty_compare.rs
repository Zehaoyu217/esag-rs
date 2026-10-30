//! The LASSO family of Project 3 (plain, adaptive and RoC weights; LASSO and
//! group LASSO) on every shape, against the truth.
//!
//! * `iid`: the paper's experiments (d = 6, μ = (2, −5, 3, −4, 5, 6); exp 1
//!   γ = (5,2,3,0,5,1,0,1,5,0,3,2,0,0), exp 2 γ = (5,3,0,…,0); data from
//!   Yu & Huang's γ), penalizing the shape. The paper's six methods as coded
//!   (paper mode, γ) next to this package's versions (exact zeros, BIC at the
//!   relaxed refit) on γ, γ-rot, spectral and log-tangent.
//! * `regression`: the hard scenario of `hard_scenario.rs` (d = 5, five
//!   correlated covariates, γ data: x1 moves μ and the shape, x2 only μ, x3
//!   only the shape, x4 both weakly, x5 nothing), penalizing the shape slopes.
//!
//! Scores: V̂ error (geodesic distance to the true V, averaged over 200
//! fresh covariate rows in regression), μ̂ error, the selection (regression:
//! which covariates keep a shape slope), how many λ's of the path were solved
//! to their KKT tolerance, and the time per path.
//!
//! * `covariates`: the same data with every slope penalized (the
//!   covariates' effects on μ and on V): which covariates each method keeps
//!   for the mean and for the shape, including the hierarchical penalty.
//!
//! `cargo run --release --example penalty_compare -- iid|regression|covariates`
//! env: `R` (replicates), `NS` (sample sizes, e.g. `100,400`), `SHAPES`
//! (`gamma,gr,sp,lt`), `PAPER=0` (skip the paper mode).
use esag::distribution::Esag;
use esag::optim::prox::RocMode;
use esag::penalized::{Grouping, PenalizedEsag, PenalizedPath, Target};
use esag::regression::{EsagRegressor, Shape};
use nalgebra::DMatrix;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal};
use rayon::prelude::*;
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

/// Affine-invariant geodesic distance between SPD matrices.
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

fn shapes() -> Vec<(&'static str, Shape)> {
    list("SHAPES", "gamma,gr,sp,lt")
        .iter()
        .map(|s| match s.as_str() {
            "gamma" => ("gamma", Shape::Gamma),
            "gr" => ("gamma-rot", Shape::GammaRot),
            "sp" => ("spectral", Shape::Spectral),
            "lt" => ("log-tangent", Shape::LogTangent),
            o => panic!("unknown shape {o}"),
        })
        .collect()
}

/// The six methods for a shape: (name, estimator). The group methods use
/// the shape's eigen grouping (γ groups for γ and γ-rot, eigen parts for
/// spectral); the log-tangent's group analog is the eigen-gap penalty
/// (i.i.d.) or one group per covariate (regression).
fn methods(shape: Shape, iid: bool) -> Vec<(&'static str, PenalizedEsag)> {
    let grouping = match shape {
        Shape::Gamma | Shape::GammaRot => Grouping::GammaGroups,
        Shape::Spectral => Grouping::EigenParts,
        _ => Grouping::CovariatePart,
    };
    let group = |w: u8| {
        let base = if shape == Shape::LogTangent && iid {
            PenalizedEsag::eigen_gap()
        } else {
            PenalizedEsag::group_lasso().grouping(grouping.clone())
        };
        match w {
            0 => base.plain(),
            1 => base.adaptive(1.0),
            _ => base.roc(RocMode::FixedPoint),
        }
    };
    vec![
        ("LASSO", PenalizedEsag::lasso()),
        ("aLASSO", PenalizedEsag::adaptive_lasso()),
        ("RoC-LASSO", PenalizedEsag::lasso().roc(RocMode::FixedPoint)),
        ("GLASSO", group(0)),
        ("aGLASSO", group(1)),
        ("RoC-GLASSO", group(2)),
    ]
}

/// The selected model's coefficients and its distribution builder: the
/// relaxed refit where there is one, else the penalized point (thresholded
/// in paper mode).
fn selected(path: &PenalizedPath) -> DMatrix<f64> {
    match &path.relaxed {
        Some(r) => r.coef.clone(),
        None => {
            let b = path.best();
            b.coef_thresholded.clone().unwrap_or_else(|| b.coef.clone())
        }
    }
}

/// `(V̂ error, μ̂ error, active atoms, share of λ's solved, ms)`.
type Score = (f64, f64, usize, f64, f64);

fn score_iid(path: &PenalizedPath, ms: f64, mu: &[f64], v: &DMatrix<f64>) -> Score {
    let c = selected(path);
    let d = mu.len();
    let eta: Vec<f64> = (0..c.nrows()).map(|k| c[(k, 0)]).collect();
    let fit = path.relaxed.as_ref().unwrap_or(&path.initial);
    let vh = if path.config.is_paper() {
        Esag::from_eta(&eta).unwrap().covariance()
    } else {
        fit.distribution_at(&eta, fit.lt_frame().as_ref())
            .unwrap()
            .covariance()
    };
    let mu_err = (0..d).map(|k| (eta[k] - mu[k]).powi(2)).sum::<f64>().sqrt();
    let active = path.best().active.iter().filter(|a| **a).count();
    let solved =
        path.points.iter().filter(|p| p.converged).count() as f64 / path.points.len() as f64;
    (geodesic(v, &vh), mu_err, active, solved, ms)
}

fn summarize(label: &str, rows: &[Score]) -> String {
    let m = |f: &dyn Fn(&Score) -> f64| rows.iter().map(f).sum::<f64>() / rows.len() as f64;
    let mut ms: Vec<f64> = rows.iter().map(|r| r.4).collect();
    ms.sort_by(f64::total_cmp);
    format!(
        "{label:<34} V̂ {:.3}  μ̂ {:.3}  active {:5.2}  λ's solved {:5.1}%  {:6.0} ms",
        m(&|r| r.0),
        m(&|r| r.1),
        m(&|r| r.2 as f64),
        100.0 * m(&|r| r.3),
        ms[ms.len() / 2]
    )
}

fn iid() {
    let mu = [2.0, -5.0, 3.0, -4.0, 5.0, 6.0];
    let exps: [(&str, [f64; 14]); 2] = [
        (
            "exp 1",
            [5., 2., 3., 0., 5., 1., 0., 1., 5., 0., 3., 2., 0., 0.],
        ),
        (
            "exp 2",
            [5., 3., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        ),
    ];
    let reps = env("R", 20u64);
    let ns: Vec<usize> = list("NS", "100,400")
        .iter()
        .map(|s| s.parse().unwrap())
        .collect();
    let paper = env("PAPER", 1usize) == 1;
    for (ename, gamma) in exps {
        let truth = Esag::new(&mu, &gamma).unwrap();
        let v = truth.covariance();
        for &n in &ns {
            println!("== {ename}, n = {n}, {reps} replicates");
            let data: Vec<DMatrix<f64>> = (0..reps)
                .map(|s| truth.sample(n, &mut ChaCha8Rng::seed_from_u64(9000 + s)))
                .collect();
            let x = DMatrix::<f64>::zeros(n, 0);
            if paper {
                for name in [
                    "LASSO",
                    "aLASSO",
                    "RoC-LASSO",
                    "GLASSO",
                    "aGLASSO",
                    "RoC-GLASSO",
                ] {
                    let rows: Vec<Score> = data
                        .par_iter()
                        .map(|y| {
                            let t = Instant::now();
                            let path = PenalizedEsag::paper(name)
                                .unwrap()
                                .with_parallel(false)
                                .fit(&x, y)
                                .unwrap();
                            score_iid(&path, t.elapsed().as_secs_f64() * 1e3, &mu, &v)
                        })
                        .collect();
                    println!("{}", summarize(&format!("paper  {name} (as coded)"), &rows));
                }
            }
            for (sname, shape) in shapes() {
                for (mname, est) in methods(shape, true) {
                    let rows: Vec<Score> = data
                        .par_iter()
                        .map(|y| {
                            let t = Instant::now();
                            let path = est
                                .clone()
                                .shape(shape)
                                .target(Target::Shape)
                                .with_parallel(false)
                                .fit(&x, y)
                                .unwrap();
                            score_iid(&path, t.elapsed().as_secs_f64() * 1e3, &mu, &v)
                        })
                        .collect();
                    println!("{}", summarize(&format!("{sname:<12} {mname}"), &rows));
                }
            }
        }
    }
}

/// Deterministic unit vector of length `n` (pseudo-random from `seed`).
fn unit(n: usize, seed: usize) -> Vec<f64> {
    let v: Vec<f64> = (0..n)
        .map(|i| ((i * 7919 + seed * 104_729) as f64 * 0.618_033_988_7).sin())
        .collect();
    let nrm = v.iter().map(|a| a * a).sum::<f64>().sqrt();
    v.iter().map(|a| a / nrm).collect()
}

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

fn regression() {
    let d = env("D", 5usize);
    let reps = env("R", 8u64);
    let ns: Vec<usize> = list("NS", "500,2000")
        .iter()
        .map(|s| s.parse().unwrap())
        .collect();
    let b = hard_truth(d);
    let (bd, pt) = (b.nrows(), b.ncols());
    let (xt, _) = simulate(&b, 200, 99);
    let eta = |c: &DMatrix<f64>, i: usize| -> Vec<f64> {
        (0..bd)
            .map(|k| c[(k, 0)] + (0..pt - 1).map(|j| c[(k, j + 1)] * xt[(i, j)]).sum::<f64>())
            .collect()
    };
    let vtrue: Vec<DMatrix<f64>> = (0..200)
        .map(|i| Esag::from_eta(&eta(&b, i)).unwrap().covariance())
        .collect();
    // V̂ error of a fit with coefficients c in the frame of `fit`
    let verr = |fit: &esag::regression::EsagRegressorFit, c: &DMatrix<f64>| -> f64 {
        let lt = fit.lt_frame();
        (0..200)
            .map(|i| {
                geodesic(
                    &vtrue[i],
                    &fit.distribution_at(&eta(c, i), lt.as_ref())
                        .unwrap()
                        .covariance(),
                )
            })
            .sum::<f64>()
            / 200.0
    };
    // covariates with a shape effect in the truth (1-based): x1, x3, x4
    let shape_drivers = [1usize, 3, 4];
    for &n in &ns {
        println!(
            "== regression, d = {d}, n = {n}, {reps} replicates (γ data; shape drivers x1, x3, x4; x2 moves μ only; x5 nothing)"
        );
        let data: Vec<(DMatrix<f64>, DMatrix<f64>)> =
            (0..reps).map(|s| simulate(&b, n, 7000 + s)).collect();
        for (sname, shape) in shapes() {
            let full: Vec<(f64, bool)> = data
                .par_iter()
                .map(|(x, y)| {
                    let mut cfg = EsagRegressor::new().with_shape(shape).with_parallel(false);
                    if shape != Shape::Gamma {
                        cfg = cfg.with_lt_reference(Some(center(d)));
                    }
                    let f = cfg.fit(x, y).unwrap();
                    (verr(&f, &f.coef), f.converged)
                })
                .collect();
            println!(
                "{sname:<12} full model                     V̂ {:.3}  converged {}/{}",
                full.iter().map(|r| r.0).sum::<f64>() / full.len() as f64,
                full.iter().filter(|r| r.1).count(),
                full.len()
            );
            for (mname, est) in methods(shape, false) {
                // (V̂, drivers kept of 3, x2 kept, x5 kept, λ's solved, ms)
                let rows: Vec<(f64, usize, bool, bool, f64, f64)> = data
                    .par_iter()
                    .map(|(x, y)| {
                        let t = Instant::now();
                        let path = est
                            .clone()
                            .shape(shape)
                            .target(Target::ShapeSlopes)
                            .lt_reference(center(d))
                            .with_parallel(false)
                            .fit(x, y)
                            .unwrap();
                        let ms = t.elapsed().as_secs_f64() * 1e3;
                        let c = selected(&path);
                        let fit = path.relaxed.as_ref().unwrap_or(&path.initial);
                        let kept = |j: usize| (d..bd).any(|k| c[(k, j)] != 0.0);
                        let solved = path.points.iter().filter(|p| p.converged).count() as f64
                            / path.points.len() as f64;
                        (
                            verr(fit, &c),
                            shape_drivers.iter().filter(|&&j| kept(j)).count(),
                            kept(2),
                            kept(5),
                            solved,
                            ms,
                        )
                    })
                    .collect();
                let r = rows.len() as f64;
                let mut ms: Vec<f64> = rows.iter().map(|q| q.5).collect();
                ms.sort_by(f64::total_cmp);
                println!(
                    "{sname:<12} {mname:<18} V̂ {:.3}  drivers kept {:4.2}/3  x2 kept {:4.2}  x5 kept {:4.2}  λ's solved {:5.1}%  {:6.0} ms",
                    rows.iter().map(|q| q.0).sum::<f64>() / r,
                    rows.iter().map(|q| q.1 as f64).sum::<f64>() / r,
                    rows.iter().filter(|q| q.2).count() as f64 / r,
                    rows.iter().filter(|q| q.3).count() as f64 / r,
                    100.0 * rows.iter().map(|q| q.4).sum::<f64>() / r,
                    ms[ms.len() / 2]
                );
            }
        }
    }
}

/// Every slope penalized (the covariates' effects on μ and on V): which
/// covariates each method keeps for the mean (truth: x1, x2, x4) and for
/// the shape (truth: x1, x3, x4), V̂ and μ̂ errors on 200 fresh rows, time.
fn covariates() {
    let d = env("D", 5usize);
    let reps = env("R", 8u64);
    let ns: Vec<usize> = list("NS", "500,2000")
        .iter()
        .map(|s| s.parse().unwrap())
        .collect();
    let b = hard_truth(d);
    let (bd, pt) = (b.nrows(), b.ncols());
    let (xt, _) = simulate(&b, 200, 99);
    let eta = |c: &DMatrix<f64>, i: usize| -> Vec<f64> {
        (0..bd)
            .map(|k| c[(k, 0)] + (0..pt - 1).map(|j| c[(k, j + 1)] * xt[(i, j)]).sum::<f64>())
            .collect()
    };
    let vtrue: Vec<DMatrix<f64>> = (0..200)
        .map(|i| Esag::from_eta(&eta(&b, i)).unwrap().covariance())
        .collect();
    let mtrue: Vec<Vec<f64>> = (0..200).map(|i| eta(&b, i)[..d].to_vec()).collect();
    // (V̂ error, μ̂ error) of coefficients c in the frame of `fit`
    let errs = |fit: &esag::regression::EsagRegressorFit, c: &DMatrix<f64>| -> (f64, f64) {
        let lt = fit.lt_frame();
        let mut ve = 0.0;
        let mut me = 0.0;
        for i in 0..200 {
            let e = eta(c, i);
            ve += geodesic(
                &vtrue[i],
                &fit.distribution_at(&e, lt.as_ref()).unwrap().covariance(),
            );
            me += (0..d)
                .map(|k| (e[k] - mtrue[i][k]).powi(2))
                .sum::<f64>()
                .sqrt();
        }
        (ve / 200.0, me / 200.0)
    };
    let methods: Vec<(&str, PenalizedEsag)> = vec![
        ("aLASSO", PenalizedEsag::adaptive_lasso()),
        ("RoC-LASSO", PenalizedEsag::lasso().roc(RocMode::FixedPoint)),
        (
            "aGLASSO (cov. parts)",
            PenalizedEsag::adaptive_group_lasso().grouping(Grouping::CovariatePart),
        ),
        (
            "RoC-GLASSO (cov. parts)",
            PenalizedEsag::group_lasso()
                .grouping(Grouping::CovariatePart)
                .roc(RocMode::FixedPoint),
        ),
        (
            "a-sparse-GLASSO",
            PenalizedEsag::sparse_group_lasso(0.5)
                .grouping(Grouping::CovariatePart)
                .adaptive(1.0),
        ),
        ("hierarchical a", PenalizedEsag::hierarchical()),
        (
            "hierarchical RoC",
            PenalizedEsag::hierarchical().roc(RocMode::FixedPoint),
        ),
    ];
    let (mu_drivers, shape_drivers) = ([1usize, 2, 4], [1usize, 3, 4]);
    for &n in &ns {
        println!(
            "== covariates, d = {d}, n = {n}, {reps} replicates (γ data; μ: x1, x2, x4; V: x1, x3, x4; x5 nothing)"
        );
        let data: Vec<(DMatrix<f64>, DMatrix<f64>)> =
            (0..reps).map(|s| simulate(&b, n, 7000 + s)).collect();
        for (sname, shape) in shapes() {
            let full: Vec<(f64, f64)> = data
                .par_iter()
                .map(|(x, y)| {
                    let mut cfg = EsagRegressor::new().with_shape(shape).with_parallel(false);
                    if shape != Shape::Gamma {
                        cfg = cfg.with_lt_reference(Some(center(d)));
                    }
                    let f = cfg.fit(x, y).unwrap();
                    errs(&f, &f.coef)
                })
                .collect();
            println!(
                "{sname:<12} {:<24} V̂ {:.3}  μ̂ {:.3}",
                "full model",
                full.iter().map(|r| r.0).sum::<f64>() / full.len() as f64,
                full.iter().map(|r| r.1).sum::<f64>() / full.len() as f64,
            );
            for (mname, est) in &methods {
                // (V̂, μ̂, μ TP, μ FP, V TP, V FP, solved, ms)
                let rows: Vec<[f64; 8]> = data
                    .par_iter()
                    .map(|(x, y)| {
                        let t = Instant::now();
                        let path = est
                            .clone()
                            .shape(shape)
                            .target(Target::Slopes)
                            .lt_reference(center(d))
                            .with_parallel(false)
                            .fit(x, y)
                            .unwrap();
                        let ms = t.elapsed().as_secs_f64() * 1e3;
                        let c = selected(&path);
                        let fit = path.relaxed.as_ref().unwrap_or(&path.initial);
                        let mean = |j: usize| (0..d).any(|k| c[(k, j)] != 0.0);
                        let shp = |j: usize| (d..bd).any(|k| c[(k, j)] != 0.0);
                        let (ve, me) = errs(fit, &c);
                        let count = |f: &dyn Fn(usize) -> bool, js: &[usize]| {
                            js.iter().filter(|&&j| f(j)).count() as f64
                        };
                        let not_mu: Vec<usize> =
                            (1..pt).filter(|j| !mu_drivers.contains(j)).collect();
                        let not_shape: Vec<usize> =
                            (1..pt).filter(|j| !shape_drivers.contains(j)).collect();
                        let solved = path.points.iter().filter(|p| p.converged).count() as f64
                            / path.points.len() as f64;
                        [
                            ve,
                            me,
                            count(&mean, &mu_drivers),
                            count(&mean, &not_mu),
                            count(&shp, &shape_drivers),
                            count(&shp, &not_shape),
                            solved,
                            ms,
                        ]
                    })
                    .collect();
                let m = |k: usize| rows.iter().map(|q| q[k]).sum::<f64>() / rows.len() as f64;
                let mut ms: Vec<f64> = rows.iter().map(|q| q[7]).collect();
                ms.sort_by(f64::total_cmp);
                println!(
                    "{sname:<12} {mname:<24} V̂ {:.3}  μ̂ {:.3}  μ kept {:.2}/3 wrong {:.2}/2  V kept {:.2}/3 wrong {:.2}/2  solved {:5.1}%  {:6.0} ms",
                    m(0),
                    m(1),
                    m(2),
                    m(3),
                    m(4),
                    m(5),
                    100.0 * m(6),
                    ms[ms.len() / 2]
                );
            }
        }
    }
}

fn main() {
    match std::env::args().nth(1).as_deref() {
        Some("regression") => regression(),
        Some("covariates") => covariates(),
        _ => iid(),
    }
}
