//! How well is the ESAG covariance V(x) estimated?
//!
//! Part 1 (accuracy): data from a known regression (5 covariates + const),
//! fitted with the same shape model; V̂(x) against the true V(x) on 200
//! fixed covariate points, over replications.
//! Part 2 (uncertainty): coverage of 95% pairs-bootstrap intervals for the
//! eigenvalues of V(x₀) and of a geodesic confidence ball around V̂(x₀).
//!
//! Log-tangent fits use the simulator's frame (reference e_d); the rows
//! "LT, m=data" fit with the default reference (the data's mean direction),
//! which is a different log-tangent model: their error stops shrinking at
//! the 1/√n rate.
//!
//! `cargo run --release --example v_recovery [-- part1|part2|diagnose]`
#[path = "big_fit.rs"]
#[allow(dead_code)]
mod big;
use esag::distribution::Esag;
use esag::inference::bootstrap_refits;
use esag::logtangent::LtFrame;
use esag::regression::{EsagRegressor, EsagRegressorFit, Shape};
use nalgebra::DMatrix;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal};
use std::time::Instant;

const P: usize = 5;

/// True law at covariate row `x` (same construction as `big::sim_shape`).
fn true_dist(b: &DMatrix<f64>, x: &[f64], d: usize, shape: Shape) -> Esag {
    let eta: Vec<f64> = (0..b.nrows())
        .map(|k| b[(k, 0)] + (0..P).map(|j| b[(k, j + 1)] * x[j]).sum::<f64>())
        .collect();
    esag::regression::shape_distribution(shape, Some(&LtFrame::standard(d)), &eta).unwrap()
}

/// Affine-invariant (geodesic) distance ‖log(V^{-1/2} W V^{-1/2})‖_F.
fn geodesic(v: &DMatrix<f64>, w: &DMatrix<f64>) -> f64 {
    let l = v.clone().cholesky().unwrap().l();
    let li = l.clone().try_inverse().unwrap();
    let m = &li * w * li.transpose();
    let m = (&m + m.transpose()) * 0.5;
    nalgebra::SymmetricEigen::new(m)
        .eigenvalues
        .iter()
        .map(|e| e.ln().powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Sorted shape eigenvalues (V restricted to μ̂⊥; V μ̂ = μ̂ always).
fn shape_eigs(dist: &Esag) -> Vec<f64> {
    let mut l = dist.frame().lambda.clone();
    l.sort_by(f64::total_cmp);
    l
}

fn angle_deg(a: &[f64], b: &[f64]) -> f64 {
    let c: f64 = a.iter().zip(b).map(|(u, v)| u * v).sum();
    c.clamp(-1.0, 1.0).acos().to_degrees()
}

/// Fit with the given shape; log-tangent fits use the simulator's frame
/// (reference `e_d`) unless `data_frame` (then the default: the data's mean
/// direction, a different log-tangent model).
fn fit_in(
    shape: Shape,
    x: &DMatrix<f64>,
    y: &DMatrix<f64>,
    se: bool,
    data_frame: bool,
) -> EsagRegressorFit {
    let mut cfg = EsagRegressor::new().with_shape(shape).with_hessian(se);
    if shape == Shape::LogTangent && !data_frame {
        // the simulator's frame: reference direction e_d
        let d = y.ncols();
        let mut m = vec![0.0; d];
        m[d - 1] = 1.0;
        cfg = cfg.with_lt_reference(Some(m));
    }
    cfg.fit(x, y).unwrap()
}

fn fit(shape: Shape, x: &DMatrix<f64>, y: &DMatrix<f64>, se: bool) -> EsagRegressorFit {
    fit_in(shape, x, y, se, false)
}

fn part1() {
    let mut rng = ChaCha8Rng::seed_from_u64(99);
    let x_eval = DMatrix::from_fn(200, P, |_, _| StandardNormal.sample(&mut rng));
    println!("Part 1: accuracy of V̂(x) over 200 covariate points, 20 replications each");
    println!(
        "{:<12}{:>3}{:>7} | {:>9} {:>9} {:>11} {:>9} {:>9} | {:>5} {:>8}",
        "shape", "d", "n", "geodesic", "√n·geod", "rel.Frob", "log-eig", "μ̂ angle", "conv", "fit"
    );
    for (shape, data_frame) in [
        (Shape::Gamma, false),
        (Shape::LogTangent, false),
        (Shape::LogTangent, true),
    ] {
        for d in [3usize, 5, 8] {
            let (_, _, b) = big::sim_shape(d, P, 10, 1, shape);
            let truth: Vec<Esag> = (0..200)
                .map(|i| {
                    let xi: Vec<f64> = (0..P).map(|j| x_eval[(i, j)]).collect();
                    true_dist(&b, &xi, d, shape)
                })
                .collect();
            for n in [500usize, 2000, 10000] {
                let reps = 20;
                let (mut geo, mut frob, mut leig, mut ang, mut conv, mut secs) =
                    (0.0, 0.0, 0.0, 0.0, 0, 0.0);
                for r in 0..reps {
                    let (x, y, _) = big::sim_shape(d, P, n, 1000 + r as u64, shape);
                    let t = Instant::now();
                    let f = fit_in(shape, &x, &y, false, data_frame);
                    secs += t.elapsed().as_secs_f64();
                    conv += usize::from(f.converged);
                    let est = f.predict_distribution(&x_eval).unwrap();
                    for (tr, es) in truth.iter().zip(&est) {
                        let (v, w) = (tr.covariance(), es.covariance());
                        geo += geodesic(&v, &w);
                        frob += (&w - &v).norm() / v.norm();
                        let (lt, le) = (shape_eigs(tr), shape_eigs(es));
                        leig += lt
                            .iter()
                            .zip(&le)
                            .map(|(a, b)| (a.ln() - b.ln()).abs())
                            .fold(0.0, f64::max);
                        ang += angle_deg(&tr.mean_direction(), &es.mean_direction());
                    }
                }
                let m = (reps * 200) as f64;
                println!(
                    "{:<12}{:>3}{:>7} | {:>9.4} {:>9.2} {:>11.4} {:>9.4} {:>8.3}° | {:>2}/{:<2} {:>6.3}s",
                    match (shape, data_frame) {
                        (Shape::Gamma, _) => "Gamma".to_string(),
                        (_, false) => "LT".to_string(),
                        (_, true) => "LT, m=data".to_string(),
                    },
                    d,
                    n,
                    geo / m,
                    geo / m * (n as f64).sqrt(),
                    frob / m,
                    leig / m,
                    ang / m,
                    conv,
                    reps,
                    secs / reps as f64
                );
            }
        }
    }
}

fn part2() {
    let n = 2000;
    let n_boot = 200;
    let reps = 200;
    println!(
        "\nPart 2: 95% pairs-bootstrap intervals for V(x₀), n = {n}, B = {n_boot}, {reps} replications"
    );
    println!(
        "{:<12}{:>3} {:>6} | {:>44} | {:>12} | {:>7}",
        "shape", "d", "x₀", "eigenvalue coverage: percentile/basic", "geodesic ball", "time"
    );
    for (shape, d) in [
        (Shape::Gamma, 3usize),
        (Shape::LogTangent, 3),
        (Shape::Gamma, 5),
        (Shape::LogTangent, 5),
    ] {
        let (_, _, b) = big::sim_shape(d, P, 10, 1, shape);
        let mut x1 = vec![0.0; P];
        x1[0] = 1.0;
        let points = [("0", vec![0.0; P]), ("x1=1", x1)];
        let truths: Vec<(DMatrix<f64>, Vec<f64>)> = points
            .iter()
            .map(|(_, x0)| {
                let t = true_dist(&b, x0, d, shape);
                (t.covariance(), shape_eigs(&t))
            })
            .collect();
        let x0m = DMatrix::from_fn(2, P, |i, j| points[i].1[j]);
        let mut cover = vec![vec![0usize; d - 1]; 2];
        let mut cover_basic = vec![vec![0usize; d - 1]; 2];
        let mut ball = [0usize; 2];
        let t = Instant::now();
        for r in 0..reps {
            let (x, y, _) = big::sim_shape(d, P, n, 5000 + r as u64, shape);
            let f = fit(shape, &x, &y, true);
            let est = f.predict_distribution(&x0m).unwrap();
            let refits = bootstrap_refits(&f, &x, &y, n_boot, 77 + r as u64);
            let boots: Vec<Vec<Esag>> = refits
                .iter()
                .map(|g| g.predict_distribution(&x0m).unwrap())
                .collect();
            for p in 0..2 {
                let (vt, lt) = &truths[p];
                for k in 0..d - 1 {
                    let mut v: Vec<f64> = boots.iter().map(|e| shape_eigs(&e[p])[k]).collect();
                    v.sort_by(f64::total_cmp);
                    let m = v.len();
                    let (lo, hi) = (
                        v[(0.025 * m as f64) as usize],
                        v[((0.975 * m as f64) as usize).min(m - 1)],
                    );
                    cover[p][k] += usize::from(lo <= lt[k] && lt[k] <= hi);
                    // basic (bias-corrected) interval: [2λ̂ − q_hi, 2λ̂ − q_lo]
                    let lh = shape_eigs(&est[p])[k];
                    cover_basic[p][k] +=
                        usize::from(2.0 * lh - hi <= lt[k] && lt[k] <= 2.0 * lh - lo);
                }
                // geodesic ball: radius = 95% quantile of δ(V̂*, V̂)
                let vhat = est[p].covariance();
                let mut dists: Vec<f64> = boots
                    .iter()
                    .map(|e| geodesic(&vhat, &e[p].covariance()))
                    .collect();
                dists.sort_by(f64::total_cmp);
                let q = dists[((0.95 * dists.len() as f64) as usize).min(dists.len() - 1)];
                ball[p] += usize::from(geodesic(&vhat, vt) <= q);
            }
        }
        for p in 0..2 {
            let cov: Vec<String> = cover[p]
                .iter()
                .zip(&cover_basic[p])
                .map(|(c, cb)| {
                    format!(
                        "{:.2}/{:.2}",
                        *c as f64 / reps as f64,
                        *cb as f64 / reps as f64
                    )
                })
                .collect();
            println!(
                "{:<12}{:>3} {:>6} | {:>44} | {:>12.3} | {:>6.1}s",
                format!("{shape:?}"),
                d,
                points[p].0,
                cov.join(" "),
                ball[p] as f64 / reps as f64,
                t.elapsed().as_secs_f64()
            );
        }
    }
}

/// Bias and spread of the smallest shape eigenvalue at x₀ = 0: across
/// replications versus within the bootstrap.
fn diagnose(shape: Shape, d: usize) {
    let (n, n_boot, reps) = (2000, 200, 60);
    let (_, _, b) = big::sim_shape(d, P, 10, 1, shape);
    let truth = true_dist(&b, &[0.0; P], d, shape);
    let lt = shape_eigs(&truth);
    let x0m = DMatrix::zeros(1, P);
    let (mut ests, mut boot_sd, mut conv, mut tot, mut ll_gap) = (vec![], 0.0, 0, 0, 0.0f64);
    for r in 0..reps {
        let (x, y, _) = big::sim_shape(d, P, n, 5000 + r as u64, shape);
        let f = fit(shape, &x, &y, true);
        let e = shape_eigs(&f.predict_distribution(&x0m).unwrap()[0]);
        ests.push(e[0]);
        let refits = bootstrap_refits(&f, &x, &y, n_boot, 77 + r as u64);
        let v: Vec<f64> = refits
            .iter()
            .map(|g| shape_eigs(&g.predict_distribution(&x0m).unwrap()[0])[0])
            .collect();
        conv += refits.iter().filter(|g| g.converged).count();
        tot += refits.len();
        let m = v.iter().sum::<f64>() / v.len() as f64;
        boot_sd += (v.iter().map(|t| (t - m).powi(2)).sum::<f64>() / v.len() as f64).sqrt();
        // does a fresh (cold) fit on the first bootstrap sample beat the warm refit?
        if r < 10 {
            use rand::RngExt;
            let mut rng = ChaCha8Rng::seed_from_u64(31 + r as u64);
            let idx: Vec<usize> = (0..n).map(|_| rng.random_range(0..n)).collect();
            let xb = DMatrix::from_fn(n, P, |i, j| x[(idx[i], j)]);
            let yb = DMatrix::from_fn(n, d, |i, j| y[(idx[i], j)]);
            // same model (shape and frame) as the original fit
            let cfg = f.config.clone().with_hessian(false);
            let cold = cfg.fit(&xb, &yb).unwrap();
            let warm = cfg.fit_warm(&xb, &yb, &f.coef).unwrap();
            ll_gap = ll_gap.max(cold.log_likelihood - warm.log_likelihood);
        }
    }
    let m = ests.iter().sum::<f64>() / reps as f64;
    let sd = (ests.iter().map(|t| (t - m).powi(2)).sum::<f64>() / reps as f64).sqrt();
    println!(
        "{shape:?} d={d}: true λ_min {:.4}; estimates mean {m:.4} (bias {:+.4}), sd across reps {sd:.4}; mean bootstrap sd {:.4}; refits converged {conv}/{tot}; max(cold − warm log-lik) {ll_gap:.3}",
        lt[0],
        m - lt[0],
        boot_sd / reps as f64
    );
}

fn main() {
    let arg = std::env::args().nth(1).unwrap_or_default();
    if arg == "diagnose" {
        diagnose(Shape::Gamma, 5);
        diagnose(Shape::LogTangent, 5);
        return;
    }
    if arg != "part2" {
        part1();
    }
    if arg != "part1" {
        part2();
    }
}
