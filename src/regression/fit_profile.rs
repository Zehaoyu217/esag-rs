//! Where a full fit spends its time when both `μ` and `V` depend on `x`,
//! and why fits fail: profiling and diagnostic studies (all `#[ignore]`).

use super::*;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::time::Instant;

/// Deterministic unit vector of length `n` (pseudo-random from `seed`).
fn unit(n: usize, seed: usize) -> Vec<f64> {
    let v: Vec<f64> = (0..n)
        .map(|i| ((i * 7919 + seed * 104_729) as f64 * 0.618_033_988_7).sin())
        .collect();
    let nrm = v.iter().map(|a| a * a).sum::<f64>().sqrt();
    v.iter().map(|a| a / nrm).collect()
}

/// The truth's mean direction: generic, near `(1, …, 1)/√d`, so neither
/// the γ frame's pole (`μ₀ = μ₁ = 0`) nor the log-tangent antipode is
/// close to the data. With `POLE=1`, near `e_d` instead (the γ pole).
pub(crate) fn center(d: usize) -> Vec<f64> {
    let pole = std::env::var("POLE").is_ok();
    let u = unit(d, 1);
    let mut m: Vec<f64> = (0..d)
        .map(|k| {
            if pole {
                u[k] + if k == d - 1 { 2.0 } else { 0.0 }
            } else {
                1.0 + 0.3 * u[k]
            }
        })
        .collect();
    let nm = m.iter().map(|a| a * a).sum::<f64>().sqrt();
    m.iter_mut().for_each(|a| *a /= nm);
    m
}

/// Hard truth, `p = 5`: `‖μ₀‖ = 2.5` (moderate concentration), shape
/// intercept of norm 1; `x1` moves μ and the shape strongly, `x2` only
/// μ, `x3` only the shape, `x4` both weakly, `x5` nothing.
pub(crate) fn hard_truth(d: usize) -> DMatrix<f64> {
    let (bd, g) = (dims::eta_len(d), dims::gamma_len(d));
    let mut b = DMatrix::zeros(bd, 6);
    let m0 = center(d);
    let nm = 1.0;
    let mu_cols = [
        (0usize, 2.5, 1usize),
        (1, 0.6, 2),
        (2, 0.45, 3),
        (4, 0.15, 4),
    ];
    for &(j, size, seed) in &mu_cols {
        let u = if j == 0 {
            m0.iter().map(|a| a / nm).collect()
        } else {
            unit(d, seed)
        };
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
pub(crate) fn simulate(
    b: &DMatrix<f64>,
    n: usize,
    seed: u64,
    shape: Shape,
) -> (DMatrix<f64>, DMatrix<f64>) {
    let d = dims::dim_from_eta_len(b.nrows()).unwrap();
    simulate_with(b, n, seed, shape, 0.5, &center(d))
}

/// The `examples/big_fit.rs` truth (‖μ‖ ≈ 4 along a fixed direction,
/// slopes ~0.3): the easier "standard" scenario.
pub(crate) fn standard_truth(d: usize, p: usize) -> DMatrix<f64> {
    let bd = dims::eta_len(d);
    DMatrix::from_fn(bd, p + 1, |k, j| {
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
    })
}

/// `n` draws with equicorrelated (`rho`) standard-normal covariates; the
/// log-tangent truth is read in the frame with reference `m`.
pub(crate) fn simulate_with(
    b: &DMatrix<f64>,
    n: usize,
    seed: u64,
    shape: Shape,
    rho: f64,
    m: &[f64],
) -> (DMatrix<f64>, DMatrix<f64>) {
    let p = b.ncols() - 1;
    let bd = b.nrows();
    let d = dims::dim_from_eta_len(bd).unwrap();
    let lt = LtFrame::new(m);
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut x = DMatrix::zeros(n, p);
    for i in 0..n {
        let z0: f64 = StandardNormal.sample(&mut rng);
        for j in 0..p {
            let z: f64 = StandardNormal.sample(&mut rng);
            x[(i, j)] = rho.sqrt() * z0 + (1.0 - rho).sqrt() * z;
        }
    }
    let mut y = DMatrix::zeros(n, d);
    let (mut eta, mut buf) = (vec![0.0; bd], vec![0.0; d]);
    for i in 0..n {
        for k in 0..bd {
            eta[k] = b[(k, 0)] + (0..p).map(|j| b[(k, j + 1)] * x[(i, j)]).sum::<f64>();
        }
        let dist = match shape {
            Shape::Gamma => Esag::from_eta(&eta).unwrap(),
            Shape::LogTangent => {
                let v = crate::logtangent::covariance(&lt, &eta[..d], &eta[d..]);
                Esag::from_covariance(&eta[..d], &v).unwrap()
            }
            Shape::GammaTransported => {
                let f = crate::frame::Frame::transported(&lt, &eta[..d], &eta[d..]);
                let v = DMatrix::from_row_slice(d, d, &f.cov_pow(1.0));
                Esag::from_covariance(&eta[..d], &v).unwrap()
            }
            Shape::GammaRot => {
                // the γ-rot truth in the frame with reference m and the
                // identity basis
                let fr = crate::gamma_rot::GrFrame::stereo(m);
                let v = fr.covariance(&eta[..d], &eta[d..]);
                Esag::from_covariance(&eta[..d], &v).unwrap()
            }
        };
        dist.sample_one(&dist.frame(), &mut rng, &mut buf);
        for k in 0..d {
            y[(i, k)] = buf[k];
        }
    }
    (x, y)
}

pub(crate) fn config(d: usize, shape: Shape) -> EsagRegressor {
    let mut cfg = EsagRegressor::new().with_shape(shape).with_parallel(false);
    if shape != Shape::Gamma {
        cfg = cfg.with_lt_reference(Some(center(d)));
    }
    cfg
}

fn median_ms(reps: usize, mut f: impl FnMut()) -> f64 {
    let mut t: Vec<f64> = (0..reps)
        .map(|_| {
            let s = Instant::now();
            f();
            s.elapsed().as_secs_f64() * 1e3
        })
        .collect();
    t.sort_by(f64::total_cmp);
    t[reps / 2]
}

/// Smallest trailing-pair norm over the γ groups `j ≥ 2`, relative to
/// the group norm, and the group attaining it.
fn min_pair_ratio(gamma: &[f64], d: usize) -> (f64, usize) {
    let mut best = (f64::INFINITY, 0);
    for gr in 2..=d - 2 {
        let off = gr * (gr + 1) / 2 - 1;
        let g = &gamma[off..off + gr + 1];
        let all = g.iter().map(|v| v * v).sum::<f64>().sqrt();
        let pair = (g[gr - 1].powi(2) + g[gr].powi(2)).sqrt();
        if all > 0.0 && pair / all < best.0 {
            best = (pair / all, gr);
        }
    }
    best
}

/// Is a non-converged γ fit sitting on observations near the singular
/// set? `cargo test --release --lib gamma_singular -- --ignored --nocapture`
#[test]
#[ignore]
fn gamma_singular_diagnosis() {
    for d in [5usize, 8] {
        for seed in [7u64, 8, 9] {
            let truth = hard_truth(d);
            let (x, y) = simulate(&truth, 2000, seed, Shape::Gamma);
            let cfg = config(d, Shape::Gamma);
            let fit = cfg.fit(&x, &y).unwrap();
            let data = cfg.data(&x, &y).unwrap();
            let (bd, pt) = (data.b_dim, data.p_total);
            let mut rows: Vec<(f64, f64, usize, usize)> = vec![];
            let mut ge = vec![0.0; bd];
            let mut eta = vec![0.0; bd];
            for i in 0..data.n {
                for k in 0..bd {
                    eta[k] = (0..pt).map(|j| fit.coef[(k, j)] * data.x[i * pt + j]).sum();
                }
                kernel_grad(&data, &eta, &data.y[i * d..(i + 1) * d], &mut ge);
                let gn = ge.iter().map(|v| v * v).sum::<f64>().sqrt();
                let (ratio, gr) = min_pair_ratio(&eta[d..], d);
                rows.push((gn, ratio, gr, i));
            }
            let mut by_grad = rows.clone();
            by_grad.sort_by(|a, b| b.0.total_cmp(&a.0));
            let mut norms: Vec<f64> = rows.iter().map(|r| r.0).collect();
            norms.sort_by(f64::total_cmp);
            let med = norms[norms.len() / 2];
            let mut by_ratio = rows.clone();
            by_ratio.sort_by(|a, b| a.1.total_cmp(&b.1));
            println!(
                "d={d} seed={seed}: converged {}, ll {:.2}, |grad|_inf {:.2e}, median obs |g| {:.2}",
                fit.converged, fit.log_likelihood, fit.grad_norm, med
            );
            println!(
                "   largest obs gradients (|g|, pair/group, group): {}",
                by_grad[..6]
                    .iter()
                    .map(|r| format!("({:.1e}, {:.1e}, {})", r.0, r.1, r.2))
                    .collect::<Vec<_>>()
                    .join(" ")
            );
            println!(
                "   smallest pair ratios (pair/group, |g|): {}",
                by_ratio[..6]
                    .iter()
                    .map(|r| format!("({:.1e}, {:.1e})", r.1, r.0))
                    .collect::<Vec<_>>()
                    .join(" ")
            );
            // the gradient without the observations at the singular set
            for &(cut_ratio, cut_grad) in &[(1e-2, 20.0), (1e-3, 20.0), (1e-2, 50.0)] {
                let mut greg = DMatrix::<f64>::zeros(bd, pt);
                let mut k_sing = 0;
                for (i, row) in rows.iter().enumerate() {
                    let singular = row.1 < cut_ratio && row.0 > cut_grad * med;
                    if singular {
                        k_sing += 1;
                        continue;
                    }
                    for k in 0..bd {
                        eta[k] = (0..pt).map(|j| fit.coef[(k, j)] * data.x[i * pt + j]).sum();
                    }
                    kernel_grad(&data, &eta, &data.y[i * d..(i + 1) * d], &mut ge);
                    for k in 0..bd {
                        for j in 0..pt {
                            greg[(k, j)] += ge[k] * data.x[i * pt + j];
                        }
                    }
                }
                println!(
                    "   cut (pair < {cut_ratio:.0e}, |g| > {cut_grad}x median): {k_sing} singular; regular-part |grad|_inf {:.2e} (mean per obs {:.1e})",
                    greg.amax(),
                    greg.amax() / data.n as f64
                );
            }
            // envelope: rotate the trailing pair of the worst observation
            let (_, _, gr, i) = by_grad[0];
            for k in 0..bd {
                eta[k] = (0..pt).map(|j| fit.coef[(k, j)] * data.x[i * pt + j]).sum();
            }
            let off = d + gr * (gr + 1) / 2 - 1;
            let (a0, b0) = (eta[off + gr - 1], eta[off + gr]);
            let r = (a0 * a0 + b0 * b0).sqrt();
            let psi0 = b0.atan2(a0);
            let yi = &data.y[i * d..(i + 1) * d];
            let l0 = kernel_grad(&data, &eta, yi, &mut ge);
            let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
            for t in 0..360 {
                let psi = psi0 + (t as f64).to_radians();
                let mut e = eta.clone();
                e[off + gr - 1] = r * psi.cos();
                e[off + gr] = r * psi.sin();
                let l = kernel_grad(&data, &e, yi, &mut ge);
                lo = lo.min(l);
                hi = hi.max(l);
            }
            println!(
                "   worst obs {i}: pair radius {r:.1e}; its log-density {l0:.3}, over the pair's angle {lo:.3} .. {hi:.3}"
            );
        }
    }
}

/// Hard scenario at small n: convergence, time, and how far the fit
/// is from the truth. `cargo test --release --lib small_sample -- --ignored --nocapture`
#[test]
#[ignore]
fn small_sample_convergence() {
    let reps = 12usize;
    for d in [3usize, 5, 8] {
        for shape in [Shape::LogTangent, Shape::Gamma] {
            for n in [100usize, 200, 300, 500, 1000] {
                let truth = hard_truth(d);
                let out: Vec<(bool, f64, f64, f64, Status)> = (0..reps)
                    .into_par_iter()
                    .map(|r| {
                        let (x, y) = simulate(&truth, n, 100 + r as u64, shape);
                        let cfg = config(d, shape);
                        let t = Instant::now();
                        let fit = cfg.fit(&x, &y);
                        let dt = t.elapsed().as_secs_f64() * 1e3;
                        match fit {
                            Ok(f) => {
                                let err = (&f.coef - &truth).abs().max();
                                let bmax = f.coef.abs().max();
                                (f.converged, dt, err, bmax, f.status)
                            }
                            Err(_) => (false, dt, f64::NAN, f64::NAN, Status::Aborted),
                        }
                    })
                    .collect();
                let conv = out.iter().filter(|o| o.0).count();
                let mut times: Vec<f64> = out.iter().map(|o| o.1).collect();
                times.sort_by(f64::total_cmp);
                let mut errs: Vec<f64> = out.iter().map(|o| o.2).collect();
                errs.sort_by(f64::total_cmp);
                let bmax = out.iter().map(|o| o.3).fold(0.0f64, f64::max);
                let statuses: Vec<String> = out
                    .iter()
                    .filter(|o| !o.0)
                    .map(|o| format!("{:?}", o.4))
                    .collect();
                println!(
                    "d={d} {:<10} n={n:>4} K={:>3}: converged {conv:>2}/{reps}, time med {:7.1} max {:7.1} ms, max|B-truth| med {:6.2} max {:6.2}, max|B| {:6.1} {}",
                    format!("{shape:?}"),
                    dims::eta_len(d) * 6,
                    times[reps / 2],
                    times[reps - 1],
                    errs[reps / 2],
                    errs[reps - 1],
                    bmax,
                    statuses.join(",")
                );
            }
        }
    }
}

/// Newton with the exact Hessian (step halving) from `b0`: iterations,
/// log-likelihood, gradient and the smallest `‖γ_i‖` along the way.
pub(crate) fn newton_trace(
    data: &RegressionData,
    b0: &DMatrix<f64>,
    iters: usize,
    print: bool,
) -> (DMatrix<f64>, f64, usize) {
    let (bd, pt, d) = (data.b_dim, data.p_total, data.d);
    let k = bd * pt;
    let mut b = b0.clone();
    let mut used = 0;
    let (mut f, _, _) = joint_loglik_grad_hess(data, &b, false);
    for it in 0..iters {
        let (f0, g, h) = joint_loglik_grad_hess(data, &b, false);
        f = f0;
        used = it;
        let gv = DVector::from_column_slice(&g);
        let gmax = gv.amax();
        let min_g = (0..data.n)
            .map(|i| {
                (d..bd)
                    .map(|r| {
                        (0..pt)
                            .map(|j| b[(r, j)] * data.x[i * pt + j])
                            .sum::<f64>()
                            .powi(2)
                    })
                    .sum::<f64>()
                    .sqrt()
            })
            .fold(f64::INFINITY, f64::min);
        if print {
            println!("      newton {it:2}: ll {f0:.6}  |g| {gmax:.2e}  min|γ_i| {min_g:.2e}");
        }
        if gmax < 1e-6 {
            break;
        }
        let hm = DMatrix::from_column_slice(k, k, &h);
        let neg = -hm;
        let step = match neg.clone().cholesky() {
            Some(ch) => ch.solve(&gv),
            None => {
                // shift until positive definite
                let mut tau = 1e-3 * neg.diagonal().amax();
                loop {
                    let mut sh = neg.clone();
                    for a in 0..k {
                        sh[(a, a)] += tau;
                    }
                    if let Some(ch) = sh.cholesky() {
                        break ch.solve(&gv);
                    }
                    tau *= 10.0;
                }
            }
        };
        let mut t = 1.0;
        loop {
            let mut cand = b.clone();
            for j in 0..pt {
                for r in 0..bd {
                    cand[(r, j)] += t * step[j * bd + r];
                }
            }
            let (fc, _) = joint_loglik_and_grad(data, &cand, false);
            if fc.is_finite() && fc >= f0 {
                b = cand;
                break;
            }
            t *= 0.5;
            if t < 1e-8 {
                return (b, f0, it);
            }
        }
    }
    (b, f, used)
}

/// One γ d = 3 bootstrap-like refit: L-BFGS (whitened, as in tests)
/// against exact Newton. `cargo test --release --lib gamma3_refit -- --ignored --nocapture`
#[test]
#[ignore]
fn gamma3_refit_trace() {
    let d = 3;
    let truth = hard_truth(d);
    let (x, y) = simulate(&truth, 2000, 7, Shape::Gamma);
    let cfg = config(d, Shape::Gamma);
    let fit = cfg.fit(&x, &y).unwrap();
    // replicate data from the full fit; refit from the fit
    for r in 0..3u64 {
        let mut rng = crate::stream_rng(1, 5, r);
        let yb = fit.sample_responses(&x, &mut rng).unwrap();
        let data = cfg.data(&x, &yb).unwrap();
        let mut rc = crate::inference::refit_config(&fit);
        let t = Instant::now();
        let lb = rc.fit_warm(&x, &yb, &fit.coef).unwrap();
        let tl = t.elapsed().as_secs_f64() * 1e3;
        rc.precondition = None;
        rc.local_precondition = false;
        let plain = rc.fit_warm(&x, &yb, &fit.coef).unwrap();
        println!(
            "rep {r}: whitened L-BFGS ll {:.6} ({} fev, {:.1} ms, conv {}); plain L-BFGS ll {:.6} ({} fev)",
            lb.log_likelihood, lb.n_fev, tl, lb.converged, plain.log_likelihood, plain.n_fev
        );
        let t = Instant::now();
        let (_, fnw, it) = newton_trace(&data, &fit.coef, 30, true);
        println!(
            "      newton: ll {fnw:.6} after {it} iterations, {:.1} ms",
            t.elapsed().as_secs_f64() * 1e3
        );
    }
}

/// Iteration trace of whitened log-tangent refits (as in the tests).
/// `cargo test --release --lib lt_refit_trace -- --ignored --nocapture 2>&1 | grep -E "lbfgs|rep"`
#[test]
#[ignore]
fn lt_refit_trace() {
    let d = 5;
    let truth = hard_truth(d);
    let (x, y) = simulate(&truth, 2000, 7, Shape::LogTangent);
    let fit = config(d, Shape::LogTangent).fit(&x, &y).unwrap();
    for r in 0..3u64 {
        let mut rng = crate::stream_rng(1, 5, r);
        let yb = fit.sample_responses(&x, &mut rng).unwrap();
        let mut rc = crate::inference::refit_config(&fit);
        rc.verbose = true;
        let f = rc.fit_warm(&x, &yb, &fit.coef).unwrap();
        eprintln!(
            "rep {r}: ll {:.8} gain {:.3} fev {} it {}",
            f.log_likelihood,
            f.log_likelihood - fit.log_likelihood,
            f.n_fev,
            f.n_iter
        );
    }
}

/// Small-n log-tangent fits that fail the check: what exact Newton does
/// from where L-BFGS stopped, and the information's smallest eigenvalue.
/// `cargo test --release --lib small_n_rescue -- --ignored --nocapture`
#[test]
#[ignore]
fn small_n_rescue() {
    for (d, n) in [(5usize, 100usize), (8, 100), (8, 150)] {
        let truth = hard_truth(d);
        for r in 0..12u64 {
            let (x, y) = simulate(&truth, n, 100 + r, Shape::LogTangent);
            let cfg = config(d, Shape::LogTangent);
            let t = Instant::now();
            let fit = cfg.fit(&x, &y).unwrap();
            let tf = t.elapsed().as_secs_f64() * 1e3;
            if fit.converged {
                continue;
            }
            let data = cfg.data(&x, &y).unwrap();
            let (_, _, h) = joint_loglik_grad_hess(&data, &fit.coef, false);
            let k = data.b_dim * data.p_total;
            let hm = -DMatrix::from_column_slice(k, k, &h);
            let eig = nalgebra::SymmetricEigen::new(hm.clone());
            let (lo, hi) = eig
                .eigenvalues
                .iter()
                .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &v| {
                    (a.min(v), b.max(v))
                });
            let t = Instant::now();
            let (bn, fnw, it) = newton_trace(&data, &fit.coef, 60, false);
            let tn = t.elapsed().as_secs_f64() * 1e3;
            let (_, g) = joint_loglik_and_grad(&data, &bn, false);
            println!(
                "d={d} n={n} rep {r}: L-BFGS {:?} ll {:.4} |g| {:.2e} ({tf:.0} ms); info eig [{lo:.2e}, {hi:.2e}]; Newton: ll {fnw:.4} |g| {:.2e} after {it} it ({tn:.0} ms), max|B| {:.2}",
                fit.status,
                fit.log_likelihood,
                fit.grad_norm,
                g.amax(),
                bn.abs().max()
            );
        }
    }
}

/// Does the R-style start ever beat a converged moment start? Each start
/// alone, against the race, over many datasets.
/// `cargo test --release --lib single_start_safety -- --ignored --nocapture`
#[test]
#[ignore]
fn single_start_safety() {
    let seeds: u64 = std::env::var("SEEDS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(16);
    for (label, hard) in [("hard", true), ("standard", false)] {
        for d in [3usize, 5, 8] {
            for shape in [Shape::LogTangent, Shape::Gamma] {
                if shape == Shape::Gamma && d >= 5 {
                    continue;
                }
                for n in [300usize, 2000] {
                    let out: Vec<(bool, bool, f64, f64, f64, usize, usize)> = (0..seeds)
                        .into_par_iter()
                        .map(|seed| {
                            let (truth, m) = if hard {
                                (hard_truth(d), center(d))
                            } else {
                                let mut m = vec![0.0; d];
                                m[d - 1] = 1.0;
                                (standard_truth(d, 5), m)
                            };
                            let rho = if hard { 0.5 } else { 0.0 };
                            let (x, y) = simulate_with(&truth, n, 500 + seed, shape, rho, &m);
                            let mut cfg =
                                EsagRegressor::new().with_shape(shape).with_parallel(false);
                            if shape == Shape::LogTangent {
                                cfg = cfg.with_lt_reference(Some(m.clone()));
                            }
                            let data = cfg.data(&x, &y).unwrap();
                            let mask = cfg.resolve_mask(&data).unwrap();
                            let free = free_params(&mask);
                            let s_dd = cfg
                                .clone()
                                .with_init(Init::DataDriven)
                                .initial_coef(&data)
                                .unwrap();
                            let s_rs = cfg
                                .clone()
                                .with_init(Init::RStyle)
                                .initial_coef(&data)
                                .unwrap();
                            let dd = cfg.optimize(&data, &s_dd, &free, cfg.max_iter);
                            let rs = cfg.optimize(&data, &s_rs, &free, cfg.max_iter);
                            let race = cfg.race(&data, &[s_dd, s_rs], &mask, cfg.max_iter);
                            (
                                dd.converged,
                                rs.converged,
                                -dd.neg_ll,
                                -rs.neg_ll,
                                -race.neg_ll,
                                dd.n_fev,
                                rs.n_fev,
                            )
                        })
                        .collect();
                    let dd_conv = out.iter().filter(|o| o.0).count();
                    let rs_conv = out.iter().filter(|o| o.1).count();
                    // a converged moment start beaten by the R-style start
                    let beaten: Vec<f64> = out
                        .iter()
                        .filter(|o| o.0 && o.3 > o.2 + 1e-6)
                        .map(|o| o.3 - o.2)
                        .collect();
                    let race_gain: Vec<f64> = out
                        .iter()
                        .filter(|o| o.0 && o.4 > o.2 + 1e-6)
                        .map(|o| o.4 - o.2)
                        .collect();
                    let fev_dd = out.iter().map(|o| o.5).sum::<usize>() as f64 / seeds as f64;
                    let fev_rs = out.iter().map(|o| o.6).sum::<usize>() as f64 / seeds as f64;
                    println!(
                        "{label:<8} d={d} {:<10} n={n:>4}: moment start converged {dd_conv:>2}/{seeds}, R-style {rs_conv:>2}/{seeds}; converged moment start beaten by R-style in {} (by {:?}), by the race in {}; fev {fev_dd:.0} / {fev_rs:.0}",
                        format!("{shape:?}"),
                        beaten.len(),
                        beaten.iter().map(|v| format!("{v:.3}")).collect::<Vec<_>>(),
                        race_gain.len()
                    );
                }
            }
        }
    }
}

/// From-scratch log-tangent fits: plain L-BFGS from the moment start
/// against L-BFGS whitened with the exact information, or the empirical
/// information, at the start.
/// `RAYON_NUM_THREADS=1 cargo test --release --lib whitened_from_scratch -- --ignored --nocapture`
#[test]
#[ignore]
fn whitened_from_scratch() {
    for (d, shape, init) in [
        (3usize, Shape::LogTangent, Init::DataDriven),
        (5, Shape::LogTangent, Init::DataDriven),
        (8, Shape::LogTangent, Init::DataDriven),
        (3, Shape::Gamma, Init::DataDriven),
        (3, Shape::Gamma, Init::RStyle),
    ] {
        for seed in [7u64, 8] {
            let truth = hard_truth(d);
            let (x, y) = simulate(&truth, 2000, seed, shape);
            let mut cfg = config(d, shape);
            cfg.compute_hessian = false;
            let data = cfg.data(&x, &y).unwrap();
            let mask = cfg.resolve_mask(&data).unwrap();
            let free = free_params(&mask);
            let b0 = cfg
                .clone()
                .with_init(init.clone())
                .initial_coef(&data)
                .unwrap();
            let t = Instant::now();
            let plain = cfg.optimize(&data, &b0, &free, cfg.max_iter);
            let t_plain = t.elapsed().as_secs_f64() * 1e3;
            let t = Instant::now();
            let mut c1 = cfg.clone();
            c1.precondition = c1.precondition_at(&x, &y, &b0, &mask);
            let ok1 = c1.precondition.is_some();
            let exact = c1.optimize(&data, &b0, &free, cfg.max_iter);
            let t_exact = t.elapsed().as_secs_f64() * 1e3;
            let t = Instant::now();
            let mut c2 = cfg.clone();
            c2.local_precondition = true;
            let emp = c2.optimize(&data, &b0, &free, cfg.max_iter);
            let t_emp = t.elapsed().as_secs_f64() * 1e3;
            println!(
                "d={d} {shape:?} {init:?} seed={seed}: plain {t_plain:6.1} ms ({:3} fev, ll {:.4}) | exact-info metric (pd {ok1}) {t_exact:6.1} ms ({:3} fev, ll {:.4}) | empirical metric {t_emp:6.1} ms ({:3} fev, ll {:.4})",
                plain.n_fev, -plain.neg_ll, exact.n_fev, -exact.neg_ll, emp.n_fev, -emp.neg_ll
            );
        }
    }
}

/// `RAYON_NUM_THREADS=1 cargo test --release --lib fit_profile -- --ignored --nocapture`
#[test]
#[ignore]
fn full_fit_breakdown() {
    let n: usize = std::env::var("N")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2000);
    for d in [3usize, 5, 8] {
        for shape in [Shape::Gamma, Shape::LogTangent] {
            let truth = hard_truth(d);
            let (x, y) = simulate(&truth, n, 7, shape);
            let cfg = config(d, shape);
            let data = cfg.data(&x, &y).unwrap();
            let tg = median_ms(9, || {
                std::hint::black_box(joint_loglik_and_grad(&data, &truth, false));
            });
            let th = median_ms(3, || {
                std::hint::black_box(joint_loglik_grad_hess(&data, &truth, false));
            });
            let t = Instant::now();
            let fit = cfg.fit(&x, &y).unwrap();
            let total = t.elapsed().as_secs_f64() * 1e3;
            // the race alone (both default starts)
            let mask = cfg.resolve_mask(&data).unwrap();
            let starts: Vec<DMatrix<f64>> = [Init::DataDriven, Init::RStyle]
                .iter()
                .map(|i| {
                    cfg.clone()
                        .with_init(i.clone())
                        .initial_coef(&data)
                        .unwrap()
                })
                .collect();
            let t = Instant::now();
            let raced = cfg.race(&data, &starts, &mask, cfg.max_iter);
            let race = t.elapsed().as_secs_f64() * 1e3;
            // L-BFGS from the truth
            let free = free_params(&mask);
            let t = Instant::now();
            let from_truth = cfg.optimize(&data, &truth, &free, cfg.max_iter);
            let tt = t.elapsed().as_secs_f64() * 1e3;
            println!(
                "d={d} {:<10} K={:>3}  grad {:6.2} ms  hess {:7.2} ms (×{:4.1}) | fit {:7.1} ms (race {:7.1}, it {:4}, fev {:4}, conv {}) | from truth {:6.1} ms it {:4} fev {:4} | ll fit {:.2} truth-start {:.2}",
                format!("{shape:?}"),
                free.len(),
                tg,
                th,
                th / tg,
                total,
                race,
                raced.n_iter,
                raced.n_fev,
                fit.converged,
                tt,
                from_truth.n_iter,
                from_truth.n_fev,
                fit.log_likelihood,
                -from_truth.neg_ll,
            );
        }
    }
}

/// How often the safety net (perturbed restart waves) rescues a gamma fit
/// at d ≥ 5 whose race did not converge.
/// `cargo test --release --lib safety_net_value -- --ignored --nocapture`
#[test]
#[ignore]
fn safety_net_value() {
    let seeds: u64 = std::env::var("SEEDS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(12);
    for (label, hard) in [("hard", true), ("standard", false)] {
        for d in [5usize, 8] {
            for n in [500usize, 2000] {
                let out: Vec<(bool, bool, f64, usize, f64, f64)> = (0..seeds)
                    .into_par_iter()
                    .map(|seed| {
                        let (truth, m) = if hard {
                            (hard_truth(d), center(d))
                        } else {
                            let mut m = vec![0.0; d];
                            m[d - 1] = 1.0;
                            (standard_truth(d, 5), m)
                        };
                        let rho = if hard { 0.5 } else { 0.0 };
                        let (x, y) = simulate_with(&truth, n, 900 + seed, Shape::Gamma, rho, &m);
                        let cfg = EsagRegressor::new()
                            .with_shape(Shape::Gamma)
                            .with_parallel(false)
                            .with_hessian(false);
                        let data = cfg.data(&x, &y).unwrap();
                        let mask = cfg.resolve_mask(&data).unwrap();
                        let starts: Vec<DMatrix<f64>> = [Init::DataDriven, Init::RStyle]
                            .iter()
                            .map(|i| {
                                cfg.clone()
                                    .with_init(i.clone())
                                    .initial_coef(&data)
                                    .unwrap()
                            })
                            .collect();
                        let t = Instant::now();
                        let race = cfg.race(&data, &starts, &mask, cfg.max_iter);
                        let t_race = t.elapsed().as_secs_f64() * 1e3;
                        let pinned = singular_observations(&data, &race.coef).len();
                        // the safety net, as in `fit`
                        let mut best = race;
                        let t = Instant::now();
                        if !best.converged {
                            let mut centers = starts.clone();
                            centers.push(best.coef.clone());
                            let extra = perturbed_starts(
                                &data,
                                &centers,
                                &mask,
                                AUTO_RESTARTS,
                                cfg.seed.wrapping_add(0x5EED),
                            );
                            let ll0 = -best.neg_ll;
                            for wave in extra.chunks(RESTART_WAVE) {
                                let alt = cfg.race(
                                    &data,
                                    wave,
                                    &mask,
                                    cfg.max_iter.min(RESTART_MAX_ITER),
                                );
                                if alt.beats(&best) {
                                    best = alt;
                                }
                                if best.converged {
                                    break;
                                }
                            }
                            let t_net = t.elapsed().as_secs_f64() * 1e3;
                            (
                                false,
                                best.converged,
                                -best.neg_ll - ll0,
                                pinned,
                                t_race,
                                t_net,
                            )
                        } else {
                            (true, true, 0.0, pinned, t_race, 0.0)
                        }
                    })
                    .collect();
                let race_conv = out.iter().filter(|o| o.0).count();
                let rescued = out.iter().filter(|o| !o.0 && o.1).count();
                let failed: Vec<&(bool, bool, f64, usize, f64, f64)> =
                    out.iter().filter(|o| !o.0).collect();
                let pinned = failed.iter().filter(|o| o.3 > 0).count();
                let gains: Vec<String> = failed.iter().map(|o| format!("{:.2}", o.2)).collect();
                let t_race: f64 = out.iter().map(|o| o.4).sum::<f64>() / seeds as f64;
                let t_net: f64 = out.iter().map(|o| o.5).sum::<f64>() / seeds as f64;
                println!(
                    "{label:<8} d={d} n={n:>4}: race converged {race_conv:>2}/{seeds}; of the {} others, {pinned} pinned, {rescued} rescued by the net; ll gains {:?}; mean ms race {t_race:.0}, net {t_net:.0}",
                    failed.len(),
                    gains
                );
            }
        }
    }
}

/// The trapped state of a gamma fit (d = 5, hard scenario, n = 500, seed
/// 950): the optimizer from the moment start ends with an observation on the
/// singular set. Precondition of the two tests below.
fn trapped_gamma_raw() -> (DMatrix<f64>, DMatrix<f64>, EsagRegressor, RawFit) {
    let (x, y) = simulate(&hard_truth(5), 500, 950, Shape::Gamma);
    let cfg = config(5, Shape::Gamma);
    let data = cfg.data(&x, &y).unwrap();
    let mask = cfg.resolve_mask(&data).unwrap();
    let b0 = cfg.initial_coef(&data).unwrap();
    let raw = cfg.optimize(&data, &b0, &free_params(&mask), cfg.max_iter);
    assert!(
        !raw.converged && !singular_observations(&data, &raw.coef).is_empty(),
        "precondition: the moment-start fit of this dataset is trapped"
    );
    (x, y, cfg, raw)
}

/// A trapped gamma fit is flagged and explained (whatever the rescue in
/// `fit` then achieves on the same data); the log-tangent fit of the same
/// data converges.
#[test]
fn pinned_gamma_fit_is_reported() {
    let (x, y, cfg, raw) = trapped_gamma_raw();
    let data = cfg.data(&x, &y).unwrap();
    let mask = cfg.resolve_mask(&data).unwrap();
    let fit = cfg.finalize(&data, raw, mask).unwrap();
    assert!(!fit.converged);
    assert!(
        !fit.singular_obs.is_empty(),
        "no singular observations found"
    );
    assert!(fit.summary().contains("singular set of the gamma"));
    assert!(fit.set_aside_obs.is_empty());
    let lt = config(5, Shape::LogTangent).fit(&x, &y).unwrap();
    assert!(lt.converged && lt.singular_obs.is_empty());
}

/// That trapped fit is resolved by `fit`: either the weight homotopy reaches
/// a regular root of the full likelihood (converged, no observation on the
/// singular set, above the trap) or the blown-up coordinates certify a
/// non-regular maximum (converged, `non_regular`, not below the trap). In
/// both cases the fit is converged with a log-likelihood at least the
/// trap's, and nothing is silently set aside.
#[test]
fn trapped_gamma_fit_is_rescued() {
    let (x, y, cfg, raw) = trapped_gamma_raw();
    let data = cfg.data(&x, &y).unwrap();
    let mask = cfg.resolve_mask(&data).unwrap();
    let trap = raw.neg_ll;
    if let Homotopy::Regular(r) = cfg.homotopy_rounds(&data, &raw, &mask, None) {
        assert!(r.converged && singular_observations(&data, &r.coef).is_empty());
        assert!(r.neg_ll < trap - 1.0, "{} vs trap {trap}", r.neg_ll);
    }
    let fit = cfg.fit(&x, &y).unwrap();
    assert!(fit.set_aside_obs.is_empty());
    assert!(
        fit.converged,
        "the trapped fit was neither rescued nor certified: {}",
        fit.summary()
    );
    assert!(
        fit.log_likelihood >= -trap - 1e-6,
        "fit {} below the trap {}",
        fit.log_likelihood,
        -trap
    );
    if !fit.non_regular {
        assert!(fit.singular_obs.is_empty());
    }
}

/// Without covariates `Shape::GammaTransported` is `Shape::Gamma`: the same
/// family, so the same fitted distribution.
#[test]
fn transported_gamma_without_covariates_is_gamma() {
    for d in [3usize, 5] {
        let b = DMatrix::from_column_slice(dims::eta_len(d), 1, &{
            let mut v: Vec<f64> = center(d).iter().map(|x| 2.0 * x).collect();
            v.extend(unit(dims::gamma_len(d), 3).iter().map(|x| 0.8 * x));
            v
        });
        let (_, y) = simulate(&b, 400, 31, Shape::Gamma);
        let x = DMatrix::zeros(400, 0);
        let a = EsagRegressor::new()
            .with_parallel(false)
            .fit(&x, &y)
            .unwrap();
        let t = EsagRegressor::new()
            .with_shape(Shape::GammaTransported)
            .with_parallel(false)
            .fit(&x, &y)
            .unwrap();
        assert!(a.converged && t.converged);
        assert!(
            (a.log_likelihood - t.log_likelihood).abs() < 1e-6,
            "d = {d}"
        );
        let e = |f: &EsagRegressorFit| {
            let eta: Vec<f64> = (0..f.coef.nrows()).map(|k| f.coef[(k, 0)]).collect();
            f.distribution_at(&eta, f.lt_frame().as_ref())
                .unwrap()
                .covariance()
        };
        assert!((e(&a) - e(&t)).abs().max() < 1e-5, "d = {d}");
    }
}

/// With the mean direction near the pole of `onb` (d = 4, hard truth), the
/// transported frame converges and is faster.
#[test]
fn transported_gamma_converges_near_the_pole() {
    let d = 4;
    let mut m: Vec<f64> = unit(d, 1);
    m[d - 1] += 2.0;
    let nm = m.iter().map(|v| v * v).sum::<f64>().sqrt();
    m.iter_mut().for_each(|v| *v /= nm);
    let mut b = hard_truth(d);
    for k in 0..d {
        b[(k, 0)] = 2.5 * m[k];
    }
    let (x, y) = simulate_with(&b, 1000, 841, Shape::Gamma, 0.5, &m);
    let fit = EsagRegressor::new()
        .with_shape(Shape::GammaTransported)
        .with_parallel(false)
        .fit(&x, &y)
        .unwrap();
    assert!(fit.converged && fit.singular_obs.is_empty());
    assert!(fit.summary().contains("transported"));
    let onb = EsagRegressor::new()
        .with_parallel(false)
        .fit(&x, &y)
        .unwrap();
    assert!(onb.summary().contains("near the pole"));
}

/// The most extreme fitted shape over the observations: log-tangent
/// `max_i ‖S(x_i)‖_F` (log condition number scale of V's tangent block) and
/// the largest per-observation log-density, for converged and failed fits.
/// `cargo test --release --lib degenerate_shapes -- --ignored --nocapture`
#[test]
#[ignore]
fn degenerate_shapes() {
    for (d, n) in [
        (5usize, 100usize),
        (8, 100),
        (8, 150),
        (8, 200),
        (5, 300),
        (8, 2000),
    ] {
        let truth = hard_truth(d);
        let mut lines = vec![];
        for r in 0..12u64 {
            let (x, y) = simulate(&truth, n, 100 + r, Shape::LogTangent);
            let cfg = config(d, Shape::LogTangent);
            let fit = cfg.fit(&x, &y).unwrap();
            let eta = fit.linear_predictor(&x).unwrap();
            let smax = (0..n)
                .map(|i| {
                    (d..eta.ncols())
                        .map(|k| eta[(i, k)].powi(2))
                        .sum::<f64>()
                        .sqrt()
                })
                .fold(0.0f64, f64::max);
            let rows = fit.log_likelihood_rows(&x, &y).unwrap();
            let mut v: Vec<f64> = rows.iter().copied().collect();
            v.sort_by(f64::total_cmp);
            let lmax = v[n - 1] - v[n / 2];
            lines.push(format!(
                "{}{:.1}/{:.1}",
                if fit.converged { "" } else { "*" },
                smax,
                lmax
            ));
        }
        println!(
            "d={d} n={n}: max ‖S_i‖ / max log-density excess over the median (* = not converged): {}",
            lines.join("  ")
        );
    }
}

/// The LASSO family on the hard scenario: selected covariate parts against
/// the truth (μ: x1, x2, x4; shape: x1, x3, x4), time on one core.
/// `RAYON_NUM_THREADS=1 cargo test --release --lib lasso_probe -- --ignored --nocapture`
#[test]
#[ignore]
fn lasso_probe() {
    use crate::penalized::{Grouping, PenalizedEsag};
    let truth_mu = [true, true, false, true, false];
    let truth_sh = [true, false, true, true, false];
    let ds: Vec<usize> = std::env::var("D")
        .unwrap_or_else(|_| "3,5".into())
        .split(',')
        .map(|v| v.parse().unwrap())
        .collect();
    let ns: Vec<usize> = std::env::var("NS")
        .unwrap_or_else(|_| "100,300,2000".into())
        .split(',')
        .map(|v| v.parse().unwrap())
        .collect();
    for d in ds {
        for &n in &ns {
            let (x, y) = simulate(&hard_truth(d), n, 21, Shape::LogTangent);
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
            for (name, pe) in presets {
                let pe = pe.lt_reference(center(d)).with_parallel(false);
                let t = Instant::now();
                let res = pe.fit(&x, &y);
                let dt = t.elapsed().as_secs_f64() * 1e3;
                match res {
                    Ok(path) => {
                        let b = &path.best().coef;
                        let act = |j: usize, rows: std::ops::Range<usize>| {
                            rows.into_iter().any(|k| b[(k, j + 1)] != 0.0)
                        };
                        let mut tp = 0;
                        let mut fp = 0;
                        let mut pattern = String::new();
                        for j in 0..5 {
                            let (am, ash) = (act(j, 0..d), act(j, d..b.nrows()));
                            pattern += &format!(
                                "{}{} ",
                                if am { "m" } else { "." },
                                if ash { "s" } else { "." }
                            );
                            tp += usize::from(am && truth_mu[j]) + usize::from(ash && truth_sh[j]);
                            fp +=
                                usize::from(am && !truth_mu[j]) + usize::from(ash && !truth_sh[j]);
                        }
                        let rel = path.relaxed.as_ref();
                        println!(
                            "d={d} n={n:>4} {name:<21} {dt:8.0} ms | x1..x5 {pattern}| TP {tp}/6 FP {fp}/4 | init conv {} degen {} | relaxed conv {}",
                            path.initial.converged,
                            path.initial.degenerate_obs.len(),
                            rel.map_or("-".into(), |r| format!(
                                "{} degen {}",
                                r.converged,
                                r.degenerate_obs.len()
                            )),
                        );
                    }
                    Err(e) => println!("d={d} n={n:>4} {name:<21} {dt:8.0} ms | error: {e}"),
                }
            }
        }
    }
}

/// Path detail at small n (debugging the selection guard).
#[test]
#[ignore]
fn lasso_path_detail() {
    use crate::penalized::PenalizedEsag;
    let d = 5;
    let (x, y) = simulate(&hard_truth(d), 100, 21, Shape::LogTangent);
    let path = PenalizedEsag::lasso()
        .lt_reference(center(d))
        .with_parallel(false)
        .fit(&x, &y)
        .unwrap();
    for (i, p) in path.points.iter().enumerate() {
        let data = config(d, Shape::LogTangent).data(&x, &y).unwrap();
        let degen = degenerate_observations(&data, &p.coef).len();
        println!(
            "{i:2} λ {:9.3} active {:3} df {:4.0} ll {:9.2} crit_ll {:9.2} bic {:9.2} conv {} kkt {:.1e} degen(at penalized coef) {degen}{}",
            p.lambda,
            p.active.iter().filter(|&&a| a).count(),
            p.df,
            p.log_likelihood,
            p.criterion_loglik,
            p.bic,
            p.converged,
            p.kkt,
            if i == path.selected {
                "  <- selected"
            } else {
                ""
            }
        );
    }
}

/// Where the adaptive LASSO path spends its time (d = 5, n = 2000).
/// `ESAG_P3_DEBUG=1 cargo test --release --lib adaptive_lasso_time -- --ignored --nocapture`
#[test]
#[ignore]
fn adaptive_lasso_time() {
    use crate::penalized::PenalizedEsag;
    let d: usize = std::env::var("D")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5);
    let (x, y) = simulate(&hard_truth(d), 2000, 21, Shape::LogTangent);
    let t = Instant::now();
    let path = PenalizedEsag::adaptive_lasso()
        .lt_reference(center(d))
        .with_parallel(false)
        .fit(&x, &y)
        .unwrap();
    println!("total {:.0} ms", t.elapsed().as_secs_f64() * 1e3);
    for p in &path.points {
        println!(
            "λ {:10.3} active {:3} iters {:4} conv {} kkt {:.1e}",
            p.lambda,
            p.active.iter().filter(|&&a| a).count(),
            p.n_iter,
            p.converged,
            p.kkt
        );
    }
}

/// A small-sample fit whose likelihood is unbounded is flagged and
/// explained, and the LASSO path then still selects a regular model.
#[test]
fn unbounded_likelihood_is_reported() {
    use crate::penalized::PenalizedEsag;
    let d = 5;
    let (x, y) = simulate(&hard_truth(d), 100, 21, Shape::LogTangent);
    let fit = config(d, Shape::LogTangent).fit(&x, &y).unwrap();
    assert!(!fit.converged);
    assert!(
        !fit.degenerate_obs.is_empty(),
        "no degenerate observation found"
    );
    assert!(fit.summary().contains("The likelihood is unbounded"));
    let path = PenalizedEsag::lasso()
        .lt_reference(center(d))
        .with_parallel(false)
        .fit(&x, &y)
        .unwrap();
    let relaxed = path.relaxed.as_ref().unwrap();
    assert!(relaxed.degenerate_obs.is_empty());
    assert!(path.best().bic.is_finite());
}

/// Phases of a fit whose likelihood is unbounded (d = 8, n = 100).
#[test]
#[ignore]
fn unbounded_fit_phases() {
    let d = 8;
    let (x, y) = simulate(&hard_truth(d), 100, 100, Shape::LogTangent);
    let cfg = config(d, Shape::LogTangent);
    let data = cfg.data(&x, &y).unwrap();
    let mask = cfg.resolve_mask(&data).unwrap();
    let free = free_params(&mask);
    let b0 = cfg
        .clone()
        .with_init(Init::DataDriven)
        .initial_coef(&data)
        .unwrap();
    let t = Instant::now();
    let mut c1 = cfg.clone();
    c1.precondition = info_precondition(&data, &b0, &free, false);
    let first = c1.optimize(&data, &b0, &free, 200);
    println!(
        "first 200 it: {:.0} ms, status {:?}, it {}, fev {}, degenerate {}",
        t.elapsed().as_secs_f64() * 1e3,
        first.status,
        first.n_iter,
        first.n_fev,
        degenerate_observations(&data, &first.coef).len()
    );
    let t = Instant::now();
    let fit = cfg.fit(&x, &y).unwrap();
    println!(
        "whole fit: {:.0} ms, status {:?}, it {}, fev {}, degenerate {}",
        t.elapsed().as_secs_f64() * 1e3,
        fit.status,
        fit.n_iter,
        fit.n_fev,
        fit.degenerate_obs.len()
    );
}

/// Print the summary notes of an unbounded fit and a pinned gamma fit.
#[test]
#[ignore]
fn show_notes() {
    let (x, y) = simulate(&hard_truth(5), 100, 21, Shape::LogTangent);
    let s = config(5, Shape::LogTangent).fit(&x, &y).unwrap().summary();
    println!("{}", &s[s.find("Notes:").unwrap()..]);
    let (x, y) = simulate(&hard_truth(5), 500, 900, Shape::Gamma);
    let s = EsagRegressor::new()
        .with_parallel(false)
        .fit(&x, &y)
        .unwrap()
        .summary();
    println!("{}", &s[s.find("Notes:").unwrap()..]);
}

/// Median fit time of the big_fit case d = 7, p = 5, n = 1000 (log-tangent
/// data), threads as configured.
#[test]
#[ignore]
fn big_fit_small_n_timing() {
    let (d, n) = (7usize, 1000usize);
    let b = standard_truth(d, 5);
    let mut m = vec![0.0; d];
    m[d - 1] = 1.0;
    let (x, y) = simulate_with(&b, n, 11, Shape::LogTangent, 0.0, &m);
    for parallel in [false, true] {
        let cfg = EsagRegressor::new()
            .with_shape(Shape::LogTangent)
            .with_parallel(parallel);
        let mut t: Vec<f64> = (0..15)
            .map(|_| {
                let s = Instant::now();
                std::hint::black_box(cfg.fit(&x, &y).unwrap());
                s.elapsed().as_secs_f64() * 1e3
            })
            .collect();
        t.sort_by(f64::total_cmp);
        println!(
            "parallel {parallel}: median {:.1} ms, min {:.1} ms",
            t[7], t[0]
        );
    }
}

/// The hard truth with the shape slopes kept only on `cols` (from the
/// x1, x3, x4 of [`hard_truth`]) and scaled by `scale`.
fn shape_truth(d: usize, cols: &[usize], scale: f64) -> DMatrix<f64> {
    let mut b = hard_truth(d);
    for j in 1..b.ncols() {
        for k in d..b.nrows() {
            b[(k, j)] *= if cols.contains(&j) { scale } else { 0.0 };
        }
    }
    b
}

/// What brings γ full fits to the singular set: the truth, or the fit?
/// For each group `j ≥ 2` the trailing pair, which alone sets the angle
/// θ_j while the whole group sets the eigenvalue gap, is affine in `x`, so
/// with two or more shape covariates the true `V(x)` could itself be
/// singular inside the data. Prints, per truth, the share of observations
/// whose true pair is within 5% of the singular set, and how γ (and
/// log-tangent) full fits fare. Measured: no true pair comes within 26% of
/// its group norm, yet with the shape on x1 alone 7 of 8 fits at `d = 5`
/// end in traps: the fit's free slopes reach the set, the truth does not.
/// `cargo test --release --lib gamma_trap_reach -- --ignored --nocapture`
#[test]
#[ignore]
fn gamma_trap_reach() {
    let reps: u64 = std::env::var("R")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);
    let cases: Vec<(usize, &str, Vec<usize>, f64)> = vec![
        (4, "x1", vec![1], 1.0),
        (4, "x1,x3", vec![1, 3], 1.0),
        (4, "x1,x3,x4", vec![1, 3, 4], 1.0),
        (5, "x1", vec![1], 1.0),
        (5, "x1,x3", vec![1, 3], 1.0),
        (5, "x1,x3,x4", vec![1, 3, 4], 1.0),
        (5, "x1,x3,x4", vec![1, 3, 4], 2.0 / 3.0),
        (5, "x1,x3,x4", vec![1, 3, 4], 1.0 / 3.0),
    ];
    println!(
        "d  shape slopes  scale | truth: near-singular obs  min pair | gamma: conv  pinned  ms/fit | LT: conv"
    );
    for (d, label, cols, scale) in cases {
        let b = shape_truth(d, &cols, scale);
        let (mut near, mut minr) = (0usize, f64::INFINITY);
        let (mut conv, mut pinned, mut ms, mut ltconv) = (0, 0, 0.0, 0);
        for r in 0..reps {
            let seed = 500 + r;
            let (x, y) = simulate(&b, 2000, seed, Shape::Gamma);
            for i in 0..x.nrows() {
                let g: Vec<f64> = (d..b.nrows())
                    .map(|k| b[(k, 0)] + (0..5).map(|j| b[(k, j + 1)] * x[(i, j)]).sum::<f64>())
                    .collect();
                let (ratio, _) = min_pair_ratio(&g, d);
                near += (ratio < 0.05) as usize;
                minr = minr.min(ratio);
            }
            let s = Instant::now();
            let fit = config(d, Shape::Gamma).fit(&x, &y).unwrap();
            ms += s.elapsed().as_secs_f64() * 1e3;
            conv += fit.converged as usize;
            pinned += (!fit.singular_obs.is_empty()) as usize;
            let (x, y) = simulate(&b, 2000, seed, Shape::LogTangent);
            ltconv += config(d, Shape::LogTangent).fit(&x, &y).unwrap().converged as usize;
        }
        println!(
            "{d}  {label:<12}  {scale:.2}  | {:>6.2}%  {minr:.1e} | {conv:>2}/{reps}  {pinned:>2}/{reps}  {:>7.0} | {ltconv:>2}/{reps}",
            100.0 * near as f64 / (2000 * reps) as f64,
            ms / reps as f64
        );
    }
}

/// Smallest relative distance of any observation's γ to the singular sets
/// (the frame pole, and the trailing pairs of groups `j ≥ 2`) at `b`.
fn min_singular_distance(data: &RegressionData, b: &DMatrix<f64>) -> f64 {
    let (d, bd, pt) = (data.d, data.b_dim, data.p_total);
    let mut eta = vec![0.0; bd];
    let mut best = f64::INFINITY;
    for i in 0..data.n {
        for (k, e) in eta.iter_mut().enumerate() {
            *e = (0..pt).map(|j| b[(k, j)] * data.x[i * pt + j]).sum();
        }
        let mm = eta[..d].iter().map(|v| v * v).sum::<f64>();
        best = best.min(((eta[0] * eta[0] + eta[1] * eta[1]) / mm).sqrt());
        best = best.min(min_pair_ratio(&eta[d..], d).0);
    }
    best
}

/// Does the γ full model have a regular local maximum near the truth, and
/// is it the highest point? Fits from the default start and from the
/// truth itself (`Init::Custom`), with log-likelihoods and the distance of
/// the closest observation to the singular sets.
/// `cargo test --release --lib gamma_regular_maximum -- --ignored --nocapture`
#[test]
#[ignore]
fn gamma_regular_maximum() {
    let d = 5;
    for (label, cols) in [("x1", vec![1usize]), ("x1,x3,x4", vec![1, 3, 4])] {
        let b = shape_truth(d, &cols, 1.0);
        println!("truth shape slopes {label}");
        println!(
            "seed  ll(truth) | default: ll  conv pinned mindist | from truth: ll  conv pinned mindist | gap"
        );
        for seed in 500u64..508 {
            let (x, y) = simulate(&b, 2000, seed, Shape::Gamma);
            let cfg = config(d, Shape::Gamma);
            let data = cfg.data(&x, &y).unwrap();
            let lt = joint_loglik_and_grad(&data, &b, false).0;
            let a = cfg.fit(&x, &y).unwrap();
            let c = cfg
                .clone()
                .with_init(Init::Custom(b.clone()))
                .fit(&x, &y)
                .unwrap();
            println!(
                "{seed}  {lt:.2} | {:.2} {} {:>2} {:.1e} | {:.2} {} {:>2} {:.1e} | {:+.2}",
                a.log_likelihood,
                a.converged as u8,
                a.singular_obs.len(),
                min_singular_distance(&data, &a.coef),
                c.log_likelihood,
                c.converged as u8,
                c.singular_obs.len(),
                min_singular_distance(&data, &c.coef),
                a.log_likelihood - c.log_likelihood
            );
        }
    }
}

/// Are trapped observations high-leverage? Percentile of each trapped
/// observation's covariate norm `‖x_i‖` among all `n`, over γ fits at
/// `d = 5` with the shape on x1 alone and on x1, x3, x4.
/// `cargo test --release --lib trapped_leverage -- --ignored --nocapture`
#[test]
#[ignore]
fn trapped_leverage() {
    for cols in [vec![1usize], vec![1, 3, 4]] {
        let b = shape_truth(5, &cols, 1.0);
        let mut pct = vec![];
        for seed in 500u64..516 {
            let (x, y) = simulate(&b, 2000, seed, Shape::Gamma);
            let fit = config(5, Shape::Gamma).fit(&x, &y).unwrap();
            let norms: Vec<f64> = (0..x.nrows()).map(|i| x.row(i).norm()).collect();
            for &i in &fit.singular_obs {
                pct.push(norms.iter().filter(|&&v| v < norms[i]).count() as f64 / 20.0);
            }
        }
        pct.sort_by(f64::total_cmp);
        println!(
            "shape on {cols:?}: {} trapped observations, percentile of ‖x‖: {pct:?}",
            pct.len()
        );
    }
}

/// Affine-invariant distance `‖log(V^{-1/2} W V^{-1/2})‖_F`.
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

/// Exact Newton ascent on `ℓ(B) − (λ/2)‖shape slopes‖²` with a halving
/// line search; stops at `max|g| < 1e-6` or after `iters` steps.
fn newton_ridge(
    data: &RegressionData,
    b0: &DMatrix<f64>,
    lambda: f64,
    iters: usize,
) -> DMatrix<f64> {
    let (bd, pt, d) = (data.b_dim, data.p_total, data.d);
    let k = bd * pt;
    let pen = |b: &DMatrix<f64>| {
        (1..pt)
            .flat_map(|j| (d..bd).map(move |r| (r, j)))
            .map(|(r, j)| b[(r, j)] * b[(r, j)])
            .sum::<f64>()
            * 0.5
            * lambda
    };
    let mut b = b0.clone();
    for _ in 0..iters {
        let (f, g, h) = joint_loglik_grad_hess(data, &b, false);
        let f0 = f - pen(&b);
        let mut gv = DVector::from_column_slice(&g);
        let mut neg = -DMatrix::from_column_slice(k, k, &h);
        for j in 1..pt {
            for r in d..bd {
                let a = j * bd + r;
                gv[a] -= lambda * b[(r, j)];
                neg[(a, a)] += lambda;
            }
        }
        if gv.amax() < 1e-6 {
            break;
        }
        let step = match neg.clone().cholesky() {
            Some(ch) => ch.solve(&gv),
            None => {
                let mut tau = 1e-3 * neg.diagonal().amax();
                loop {
                    let mut sh = neg.clone();
                    for a in 0..k {
                        sh[(a, a)] += tau;
                    }
                    if let Some(ch) = sh.cholesky() {
                        break ch.solve(&gv);
                    }
                    tau *= 10.0;
                }
            }
        };
        let mut t = 1.0;
        loop {
            let mut cand = b.clone();
            for j in 0..pt {
                for r in 0..bd {
                    cand[(r, j)] += t * step[j * bd + r];
                }
            }
            let fc = joint_loglik_and_grad(data, &cand, false).0 - pen(&cand);
            if fc.is_finite() && fc >= f0 {
                b = cand;
                break;
            }
            t *= 0.5;
            if t < 1e-8 {
                return b;
            }
        }
    }
    b
}

/// Can the regular root of the γ score equations be found where L-BFGS
/// gets trapped? Strategies on the hard scenario (γ data, env `D`, `N`,
/// `R`), each scored at its end point: log-likelihood, gradient, trapped
/// observations, negative-definite Hessian, `V̂(x)` error on 200 fresh
/// points, time.
/// `cargo test --release --lib gamma_regular_root -- --ignored --nocapture`
#[test]
#[ignore]
fn gamma_regular_root() {
    let env = |k: &str, v: usize| {
        std::env::var(k)
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(v)
    };
    let (d, n, reps) = (env("D", 5), env("N", 2000), env("R", 8) as u64);
    let b = hard_truth(d);
    let (bd, pt) = (b.nrows(), b.ncols());
    let (xt, _) = simulate(&b, 200, 99, Shape::Gamma);
    let eta = |c: &DMatrix<f64>, i: usize| -> Vec<f64> {
        (0..bd)
            .map(|k| c[(k, 0)] + (0..pt - 1).map(|j| c[(k, j + 1)] * xt[(i, j)]).sum::<f64>())
            .collect()
    };
    let vtrue: Vec<DMatrix<f64>> = (0..200)
        .map(|i| Esag::from_eta(&eta(&b, i)).unwrap().covariance())
        .collect();
    let verr = |c: &DMatrix<f64>| {
        (0..200)
            .map(|i| geodesic(&vtrue[i], &Esag::from_eta(&eta(c, i)).unwrap().covariance()))
            .sum::<f64>()
            / 200.0
    };
    let names = [
        "default fit",
        "exclude trapped, refit, re-add",
        "Newton from the default fit",
        "Newton from the truth",
        "ridge continuation + Newton",
    ];
    let mut table: Vec<Vec<(f64, bool, f64, f64)>> = vec![vec![]; names.len()];
    for seed in 600..600 + reps {
        let (x, y) = simulate(&b, n, seed, Shape::Gamma);
        let cfg = config(d, Shape::Gamma);
        let data = cfg.data(&x, &y).unwrap();
        let mut ends: Vec<(DMatrix<f64>, f64)> = vec![];
        let t = Instant::now();
        let f0 = cfg.fit(&x, &y).unwrap();
        ends.push((f0.coef.clone(), t.elapsed().as_secs_f64() * 1e3));
        // exclusion and re-inclusion, up to four rounds
        let t = Instant::now();
        let mut cur = f0.clone();
        let mut out: Vec<usize> = vec![];
        for _ in 0..4 {
            if cur.converged {
                break;
            }
            let s = singular_observations(&data, &cur.coef);
            if s.is_empty() {
                break;
            }
            out.extend(s);
            out.sort_unstable();
            out.dedup();
            let keep: Vec<usize> = (0..n).filter(|i| out.binary_search(i).is_err()).collect();
            let xs = DMatrix::from_fn(keep.len(), x.ncols(), |a, j| x[(keep[a], j)]);
            let ys = DMatrix::from_fn(keep.len(), d, |a, j| y[(keep[a], j)]);
            let sub = cfg.fit_warm(&xs, &ys, &cur.coef).unwrap();
            cur = cfg.fit_warm(&x, &y, &sub.coef).unwrap();
        }
        ends.push((cur.coef.clone(), t.elapsed().as_secs_f64() * 1e3));
        let t = Instant::now();
        let c = newton_ridge(&data, &f0.coef, 0.0, 40);
        ends.push((c, t.elapsed().as_secs_f64() * 1e3));
        let t = Instant::now();
        let c = newton_ridge(&data, &b, 0.0, 40);
        ends.push((c, t.elapsed().as_secs_f64() * 1e3));
        // ridge continuation from the constant-shape fit
        let t = Instant::now();
        let constant = DMatrix::from_fn(bd, pt, |k, j| k < d || j == 0);
        let mut c = cfg
            .clone()
            .with_mask(Some(constant))
            .fit(&x, &y)
            .unwrap()
            .coef;
        for lam in [1e3, 1e2, 1e1, 1.0, 0.0] {
            c = newton_ridge(&data, &c, lam, if lam == 0.0 { 40 } else { 15 });
        }
        ends.push((c, t.elapsed().as_secs_f64() * 1e3));
        for (k, (c, ms)) in ends.iter().enumerate() {
            let (ll, g, h) = joint_loglik_grad_hess(&data, c, false);
            let gmax = g.iter().fold(0.0f64, |a, v| a.max(v.abs()));
            let nd = (-DMatrix::from_column_slice(bd * pt, bd * pt, &h))
                .cholesky()
                .is_some();
            let trapped = !singular_observations(&data, c).is_empty();
            let regular = gmax < 1e-3 && nd && !trapped;
            println!(
                "seed {seed} {:<32} ll {ll:.2}  max|g| {gmax:.1e}  trapped {}  regular {}  V err {:.3}  {ms:.0} ms",
                names[k],
                trapped as u8,
                regular as u8,
                verr(c)
            );
            table[k].push((ll, regular, verr(c), *ms));
        }
    }
    println!("d = {d}, n = {n}, {reps} datasets:");
    for (k, rows) in table.iter().enumerate() {
        let reg = rows.iter().filter(|r| r.1).count();
        let mut ve: Vec<f64> = rows.iter().map(|r| r.2).collect();
        let mut ms: Vec<f64> = rows.iter().map(|r| r.3).collect();
        ve.sort_by(f64::total_cmp);
        ms.sort_by(f64::total_cmp);
        let gain: Vec<String> = rows
            .iter()
            .zip(&table[0])
            .map(|(r, r0)| format!("{:+.1}", r.0 - r0.0))
            .collect();
        println!(
            "  {:<32} regular root {reg}/{}  median V err {:.3}  median {:.0} ms  ll vs default [{}]",
            names[k],
            rows.len(),
            ve[ve.len() / 2],
            ms[ms.len() / 2],
            gain.join(" ")
        );
    }
}

/// Exclusion rounds for trapped γ fits: drop the trapped observations,
/// refit the rest warm, re-add them, repeat (cumulative, up to `ROUNDS`).
/// Success = a regular root of the full likelihood (converged, no trapped
/// observation). Env `D`, `N`, `R`, `ROUNDS`, `STD=1` (standard truth).
/// `cargo test --release --lib gamma_exclusion_rounds -- --ignored --nocapture`
#[test]
#[ignore]
fn gamma_exclusion_rounds() {
    use rayon::prelude::*;
    let env = |k: &str, v: usize| {
        std::env::var(k)
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(v)
    };
    let (d, n, reps, rounds) = (
        env("D", 5),
        env("N", 2000),
        env("R", 12) as u64,
        env("ROUNDS", 8),
    );
    let b = if std::env::var("STD").is_ok() {
        standard_truth(d, 5)
    } else {
        hard_truth(d)
    };
    let rows: Vec<(bool, bool, usize, f64, f64, f64)> = (700..700 + reps)
        .into_par_iter()
        .map(|seed| {
            let (x, y) = simulate(&b, n, seed, Shape::Gamma);
            let cfg = config(d, Shape::Gamma);
            let data = cfg.data(&x, &y).unwrap();
            let t = Instant::now();
            let f0 = cfg.fit(&x, &y).unwrap();
            let t0 = t.elapsed().as_secs_f64() * 1e3;
            let mut cur = f0.clone();
            let mut out: Vec<usize> = vec![];
            let mut used = 0;
            while !cur.converged && used < rounds {
                let s = singular_observations(&data, &cur.coef);
                if s.is_empty() {
                    break;
                }
                used += 1;
                out.extend(s);
                out.sort_unstable();
                out.dedup();
                let keep: Vec<usize> = (0..n).filter(|i| out.binary_search(i).is_err()).collect();
                let xs = DMatrix::from_fn(keep.len(), x.ncols(), |a, j| x[(keep[a], j)]);
                let ys = DMatrix::from_fn(keep.len(), d, |a, j| y[(keep[a], j)]);
                let sub = cfg.fit_warm(&xs, &ys, &cur.coef).unwrap();
                let next = cfg.fit_warm(&x, &y, &sub.coef).unwrap();
                if next.converged || next.log_likelihood > cur.log_likelihood {
                    cur = next;
                }
            }
            let t1 = t.elapsed().as_secs_f64() * 1e3;
            (
                f0.converged,
                cur.converged,
                used,
                cur.log_likelihood - f0.log_likelihood,
                t0,
                t1,
            )
        })
        .collect();
    let c0 = rows.iter().filter(|r| r.0).count();
    let c1 = rows.iter().filter(|r| r.1).count();
    let mut t0: Vec<f64> = rows.iter().map(|r| r.4).collect();
    let mut t1: Vec<f64> = rows.iter().map(|r| r.5).collect();
    t0.sort_by(f64::total_cmp);
    t1.sort_by(f64::total_cmp);
    let gains: Vec<String> = rows
        .iter()
        .filter(|r| !r.0 && r.1)
        .map(|r| format!("{:+.1}", r.3))
        .collect();
    let rounds_used: Vec<usize> = rows.iter().map(|r| r.2).collect();
    println!(
        "d={d} n={n}{}: converged {c0}/{reps} -> {c1}/{reps} with exclusion; rounds {rounds_used:?}; ll gain where rescued [{}]; median time {:.0} -> {:.0} ms",
        if std::env::var("STD").is_ok() {
            " standard"
        } else {
            " hard"
        },
        gains.join(" "),
        t0[t0.len() / 2],
        t1[t1.len() / 2]
    );
}

/// The γ-smooth frame of a truth `b`: `m` = `center(d)`, and each group's
/// reference at the intercept's own direction, so the intercept sits at
/// `Q_j = B_j`.
fn gs_truth_frame(b: &DMatrix<f64>, d: usize) -> crate::gamma_smooth::GsFrame {
    let refs = (2..=d - 2)
        .map(|gr| {
            let off = d + gr * (gr + 1) / 2 - 1;
            (0..=gr)
                .map(|i| {
                    let k = gr - i;
                    let s = if k % 2 == 0 { 1.0 } else { -1.0 };
                    s * b[(off + k, 0)]
                })
                .collect()
        })
        .collect();
    crate::gamma_smooth::GsFrame::new(&center(d), refs)
}

/// `n` draws from the γ-smooth regression with coefficients `b` in frame
/// `fr` (equicorrelated covariates, ρ = 0.5).
fn simulate_gs(
    b: &DMatrix<f64>,
    n: usize,
    seed: u64,
    fr: &crate::gamma_smooth::GsFrame,
) -> (DMatrix<f64>, DMatrix<f64>) {
    let (x, _) = simulate(b, n, seed, Shape::Gamma);
    let d = fr.dim();
    let mut rng = ChaCha8Rng::seed_from_u64(seed ^ 0x9e37);
    let mut y = DMatrix::zeros(n, d);
    let mut buf = vec![0.0; d];
    for i in 0..n {
        let eta: Vec<f64> = (0..b.nrows())
            .map(|k| {
                b[(k, 0)]
                    + (0..b.ncols() - 1)
                        .map(|j| b[(k, j + 1)] * x[(i, j)])
                        .sum::<f64>()
            })
            .collect();
        let v = fr.covariance(&eta[..d], &eta[d..]);
        let dist = Esag::from_covariance(&eta[..d], &v).unwrap();
        dist.sample_one(&dist.frame(), &mut rng, &mut buf);
        for k in 0..d {
            y[(i, k)] = buf[k];
        }
    }
    (x, y)
}

/// A plain γ-smooth regression fit: L-BFGS on the joint negative
/// log-likelihood (forward-mode gradients), from the moment intercept
/// written in frame `fr`, slopes zero. Returns (coef, converged, iters,
/// evaluations, max|g|).
fn fit_gs(
    fr: &crate::gamma_smooth::GsFrame,
    x: &DMatrix<f64>,
    y: &DMatrix<f64>,
) -> (DMatrix<f64>, bool, usize, usize, f64) {
    fit_gs_from(fr, x, y, false)
}

/// [`fit_gs`]; with `gamma_start`, the shape intercept starts at the γ
/// moment start itself (valid for `γ`'s map in `onb(μ)`, or in the frame
/// transported from the sample mean direction, where the two coincide).
fn fit_gs_from(
    fr: &crate::gamma_smooth::GsFrame,
    x: &DMatrix<f64>,
    y: &DMatrix<f64>,
    gamma_start: bool,
) -> (DMatrix<f64>, bool, usize, usize, f64) {
    use crate::gamma_smooth::loglik_gs_grad;
    use crate::optim::lbfgs::{FnObjective, LbfgsConfig, minimize};
    let (n, d, p) = (y.nrows(), y.ncols(), x.ncols());
    let (bd, pt) = (dims::eta_len(d), p + 1);
    let yv: Vec<f64> = (0..n)
        .flat_map(|i| (0..d).map(move |k| (i, k)))
        .map(|(i, k)| y[(i, k)])
        .collect();
    let (mu0, g0) = moment_start(&yv, n, d);
    let gs0 = if gamma_start {
        g0.clone()
    } else {
        let v0 = Esag::new(&mu0, &g0).unwrap().covariance();
        fr.gamma_from_covariance(&mu0, &v0)
    };
    let mut start = vec![0.0; bd * pt];
    start[..d].copy_from_slice(&mu0);
    start[d..d + gs0.len()].copy_from_slice(&gs0);
    // optional whitening by the score information at the start (WHITEN=1)
    let nk = bd * pt;
    let lmat: Option<DMatrix<f64>> = std::env::var("WHITEN").is_ok().then(|| {
        let mut info = DMatrix::<f64>::zeros(nk, nk);
        let (mut eta, mut g) = (vec![0.0; bd], vec![0.0; bd]);
        let mut s = vec![0.0; nk];
        for i in 0..n {
            let xr = |j: usize| if j == 0 { 1.0 } else { x[(i, j - 1)] };
            for (k, e) in eta.iter_mut().enumerate() {
                *e = (0..pt).map(|j| start[j * bd + k] * xr(j)).sum();
            }
            loglik_gs_grad(fr, &eta, &yv[i * d..(i + 1) * d], &mut g);
            for j in 0..pt {
                for k in 0..bd {
                    s[j * bd + k] = g[k] * xr(j);
                }
            }
            for a in 0..nk {
                for c in 0..nk {
                    info[(a, c)] += s[a] * s[c];
                }
            }
        }
        // b = start + L⁻ᵀ θ with L Lᵀ = info
        let l = info.cholesky().unwrap().l();
        l.transpose().try_inverse().unwrap()
    });
    let to_b = |th: &[f64]| -> Vec<f64> {
        match &lmat {
            None => th.to_vec(),
            Some(m) => {
                let t = nalgebra::DVector::from_column_slice(th);
                let v = m * t;
                (0..nk).map(|a| start[a] + v[a]).collect()
            }
        }
    };
    let obj = FnObjective(|th_in: &[f64], grad_out: &mut [f64]| {
        let th = to_b(th_in);
        let mut grad = vec![0.0; nk];
        let (mut eta, mut g) = (vec![0.0; bd], vec![0.0; bd]);
        let mut f = 0.0;
        for i in 0..n {
            let xr = |j: usize| if j == 0 { 1.0 } else { x[(i, j - 1)] };
            for (k, e) in eta.iter_mut().enumerate() {
                *e = (0..pt).map(|j| th[j * bd + k] * xr(j)).sum();
            }
            f -= loglik_gs_grad(fr, &eta, &yv[i * d..(i + 1) * d], &mut g);
            for j in 0..pt {
                let w = xr(j);
                for k in 0..bd {
                    grad[j * bd + k] -= g[k] * w;
                }
            }
        }
        match &lmat {
            None => grad_out.copy_from_slice(&grad),
            Some(m) => {
                let gv = m.transpose() * nalgebra::DVector::from_column_slice(&grad);
                grad_out.copy_from_slice(gv.as_slice());
            }
        }
        f
    });
    let x0 = if lmat.is_some() {
        vec![0.0; nk]
    } else {
        start.clone()
    };
    let cfg = LbfgsConfig {
        memory: (bd * pt).clamp(10, 200),
        max_iter: 2000,
        max_fev: 40_000,
        tol_grad: 1e-6,
        ..LbfgsConfig::default()
    };
    let res = minimize(&obj, &x0, &cfg);
    if std::env::var("TRACE").is_ok() {
        let mut g = vec![0.0; nk];
        let ftruth = std::env::var("TRUTH_LL")
            .ok()
            .and_then(|v| v.parse::<f64>().ok());
        let _ = &mut g;
        println!(
            "    status {:?}  iters {}  fev {}  -ll {:.3}{}",
            res.status,
            res.n_iter,
            res.n_fev,
            res.f,
            ftruth
                .map(|t| format!("  truth -ll {t:.3}"))
                .unwrap_or_default()
        );
    }
    let coef = DMatrix::from_column_slice(bd, pt, &to_b(&res.x));
    if std::env::var("TRACE").is_ok() {
        // per-observation gradient norms and smallest group norms (j >= 2)
        let mut rows: Vec<(f64, f64, f64)> = (0..n)
            .map(|i| {
                let xr = |j: usize| if j == 0 { 1.0 } else { x[(i, j - 1)] };
                let eta: Vec<f64> = (0..bd)
                    .map(|k| (0..pt).map(|j| coef[(k, j)] * xr(j)).sum())
                    .collect();
                let mut g = vec![0.0; bd];
                loglik_gs_grad(fr, &eta, &yv[i * d..(i + 1) * d], &mut g);
                let gn = g.iter().map(|v| v * v).sum::<f64>().sqrt();
                let mut gmin = f64::INFINITY;
                let mut amin = f64::INFINITY;
                for gr in 2..=d - 2 {
                    let off = d + gr * (gr + 1) / 2 - 1;
                    let gv = &eta[off..off + gr + 1];
                    let nrm = gv.iter().map(|v| v * v).sum::<f64>().sqrt();
                    gmin = gmin.min(nrm);
                    // distance of v_j to the antipode -c_j: 1 + v·c
                    let c = &fr.refs()[gr - 2];
                    let vc: f64 = (0..=gr)
                        .map(|ii| {
                            let k = gr - ii;
                            let sg = if k % 2 == 0 { 1.0 } else { -1.0 };
                            sg * gv[k] / nrm * c[ii]
                        })
                        .sum();
                    amin = amin.min(1.0 + vc);
                }
                (gn, gmin, amin)
            })
            .collect();
        let mut gns: Vec<f64> = rows.iter().map(|r| r.0).collect();
        gns.sort_by(f64::total_cmp);
        let med = gns[n / 2];
        rows.sort_by(|a, b| b.0.total_cmp(&a.0));
        let top: Vec<String> = rows
            .iter()
            .take(4)
            .map(|r| {
                format!(
                    "|g| {:.0}x med, min group {:.1e}, antipode {:.2}",
                    r.0 / med,
                    r.1,
                    r.2
                )
            })
            .collect();
        let gmin_all = rows.iter().map(|r| r.1).fold(f64::INFINITY, f64::min);
        let amin_all = rows.iter().map(|r| r.2).fold(f64::INFINITY, f64::min);
        println!(
            "    top gradients: {}; overall min group {gmin_all:.1e}, min antipode {amin_all:.2}",
            top.join(" | ")
        );
    }
    (
        coef,
        res.status.is_converged(),
        res.n_iter,
        res.n_fev,
        res.grad_norm_inf,
    )
}

/// Does γ-smooth remove the traps? Hard scenario (env `D`, `N`, `R`), data
/// from γ-smooth (in the truth's frame) or from γ (`DATA=gamma`); γ-smooth
/// fits in the truth's frame (`FRAME=truth`) or in a frame referenced at
/// the moment start (default), against γ fits. Convergence, iterations,
/// time and `V̂(x)` error on 200 fresh points.
/// `cargo test --release --lib gamma_smooth_fits -- --ignored --nocapture`
#[test]
#[ignore]
fn gamma_smooth_fits() {
    use rayon::prelude::*;
    let env = |k: &str, v: usize| {
        std::env::var(k)
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(v)
    };
    let (d, n, reps) = (env("D", 5), env("N", 2000), env("R", 8) as u64);
    let gamma_data = std::env::var("DATA").as_deref() == Ok("gamma");
    let truth_frame = std::env::var("FRAME").as_deref() == Ok("truth");
    let b = if std::env::var("STD").is_ok() {
        standard_truth(d, 5)
    } else {
        hard_truth(d)
    };
    let (bd, pt) = (b.nrows(), b.ncols());
    let tf = gs_truth_frame(&b, d);
    let (xt, _) = simulate(&b, 200, 99, Shape::Gamma);
    let eta = |c: &DMatrix<f64>, i: usize| -> Vec<f64> {
        (0..bd)
            .map(|k| c[(k, 0)] + (0..pt - 1).map(|j| c[(k, j + 1)] * xt[(i, j)]).sum::<f64>())
            .collect()
    };
    let vtrue: Vec<DMatrix<f64>> = (0..200)
        .map(|i| {
            let e = eta(&b, i);
            if gamma_data {
                Esag::from_eta(&e).unwrap().covariance()
            } else {
                tf.covariance(&e[..d], &e[d..])
            }
        })
        .collect();
    let rows: Vec<(bool, usize, f64, f64, bool, f64, f64)> = (800..800 + reps)
        .into_par_iter()
        .map(|seed| {
            let (x, y) = if gamma_data {
                simulate(&b, n, seed, Shape::Gamma)
            } else {
                simulate_gs(&b, n, seed, &tf)
            };
            let fr = if truth_frame {
                tf.clone()
            } else {
                let yv: Vec<f64> = (0..n)
                    .flat_map(|i| (0..d).map(move |k| (i, k)))
                    .map(|(i, k)| y[(i, k)])
                    .collect();
                let (mu0, g0) = moment_start(&yv, n, d);
                let v0 = Esag::new(&mu0, &g0).unwrap().covariance();
                crate::gamma_smooth::GsFrame::self_referenced(&mu0, &v0).0
            };
            if std::env::var("TRACE").is_ok() {
                let tll: f64 = (0..n)
                    .map(|i| {
                        let e: Vec<f64> = (0..bd)
                            .map(|k| {
                                b[(k, 0)]
                                    + (0..pt - 1).map(|j| b[(k, j + 1)] * x[(i, j)]).sum::<f64>()
                            })
                            .collect();
                        let yi: Vec<f64> = (0..d).map(|k| y[(i, k)]).collect();
                        if gamma_data {
                            crate::loglik::loglik_eta::<f64>(&e, &yi)
                        } else {
                            crate::gamma_smooth::loglik_gs::<f64>(&tf, &e, &yi)
                        }
                    })
                    .sum();
                println!("  seed {seed}: truth -ll {:.3}", -tll);
            }
            let t = Instant::now();
            let (c, conv, it, _fev, _gmax) = fit_gs(&fr, &x, &y);
            let ms = t.elapsed().as_secs_f64() * 1e3;
            let verr = (0..200)
                .map(|i| {
                    let e = eta(&c, i);
                    geodesic(&vtrue[i], &fr.covariance(&e[..d], &e[d..]))
                })
                .sum::<f64>()
                / 200.0;
            // the γ fit on the same data, for reference
            let t = Instant::now();
            let g = config(d, Shape::Gamma).fit(&x, &y).unwrap();
            let gms = t.elapsed().as_secs_f64() * 1e3;
            let gerr = (0..200)
                .map(|i| {
                    geodesic(
                        &vtrue[i],
                        &Esag::from_eta(&eta(&g.coef, i)).unwrap().covariance(),
                    )
                })
                .sum::<f64>()
                / 200.0;
            (conv, it, ms, verr, g.converged, gms, gerr)
        })
        .collect();
    let med = |mut v: Vec<f64>| {
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    };
    println!(
        "d={d} n={n} {} data, {} frame, {reps} datasets:",
        if gamma_data { "gamma" } else { "gamma-smooth" },
        if truth_frame {
            "truth's"
        } else {
            "moment-referenced"
        }
    );
    println!(
        "  gamma-smooth: converged {}/{reps}, iterations {:?}, median {:.0} ms (forward-mode gradient), V error {:.3}",
        rows.iter().filter(|r| r.0).count(),
        rows.iter().map(|r| r.1).collect::<Vec<_>>(),
        med(rows.iter().map(|r| r.2).collect()),
        med(rows.iter().map(|r| r.3).collect())
    );
    println!(
        "  gamma:        converged {}/{reps}, median {:.0} ms, V error {:.3}",
        rows.iter().filter(|r| r.4).count(),
        med(rows.iter().map(|r| r.5).collect()),
        med(rows.iter().map(|r| r.6).collect())
    );
}

/// How close do the hard truth's γ groups come to zero (near-equal
/// eigenvalues) over the data? Per group: smallest `‖G_j(x_i)‖`, its
/// median, and the share of observations below 0.05 and 0.1.
/// `cargo test --release --lib group_norm_reach -- --ignored --nocapture`
#[test]
#[ignore]
fn group_norm_reach() {
    for d in [4usize, 5, 8] {
        let b = hard_truth(d);
        let (x, _) = simulate(&b, 20000, 5, Shape::Gamma);
        for gr in 1..=d - 2 {
            let off = d + gr * (gr + 1) / 2 - 1;
            let mut norms: Vec<f64> = (0..x.nrows())
                .map(|i| {
                    (0..=gr)
                        .map(|k| {
                            let r = off + k;
                            (b[(r, 0)] + (0..5).map(|j| b[(r, j + 1)] * x[(i, j)]).sum::<f64>())
                                .powi(2)
                        })
                        .sum::<f64>()
                        .sqrt()
                })
                .collect();
            norms.sort_by(f64::total_cmp);
            let below =
                |t: f64| norms.iter().filter(|&&v| v < t).count() as f64 / norms.len() as f64;
            println!(
                "d={d} group {gr}: min ‖G‖ {:.3}, median {:.3}; below 0.05: {:.2}%, below 0.1: {:.2}%",
                norms[0],
                norms[norms.len() / 2],
                100.0 * below(0.05),
                100.0 * below(0.1)
            );
        }
    }
}

/// Seeds where exclusion rounds rescue a trapped γ race (d = 5, hard
/// scenario, env `N`): for writing the regression test.
/// `cargo test --release --lib rescue_seed_search -- --ignored --nocapture`
#[test]
#[ignore]
fn rescue_seed_search() {
    use rayon::prelude::*;
    let n: usize = std::env::var("N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1000);
    let b = hard_truth(5);
    let hits: Vec<(u64, f64)> = (900u64..960)
        .into_par_iter()
        .filter_map(|seed| {
            let (x, y) = simulate(&b, n, seed, Shape::Gamma);
            let cfg = config(5, Shape::Gamma);
            let data = cfg.data(&x, &y).unwrap();
            let mask = cfg.resolve_mask(&data).unwrap();
            let free = free_params(&mask);
            let b0 = cfg.initial_coef(&data).unwrap();
            let raw = cfg.optimize(&data, &b0, &free, cfg.max_iter);
            if raw.converged || singular_observations(&data, &raw.coef).is_empty() {
                return None;
            }
            let before = raw.neg_ll;
            match cfg.homotopy_rounds(&data, &raw, &mask, None) {
                Homotopy::Regular(r) if r.neg_ll < before => Some((seed, before - r.neg_ll)),
                _ => None,
            }
        })
        .collect();
    println!("n = {n}: rescued seeds {hits:?}");
}

/// γ's map in `onb(μ)` against the frame transported from the sample mean
/// direction, with the same optimizer (whitened L-BFGS, forward-mode
/// gradients), on γ data whose mean direction is generic or near the pole
/// of `onb` (`POLE=1`); `STD=1` for the standard truth. Env `D`, `N`, `R`.
/// `cargo test --release --lib gamma_frame_pole -- --ignored --nocapture`
#[test]
#[ignore]
fn gamma_frame_pole() {
    use crate::gamma_smooth::GsFrame;
    use rayon::prelude::*;
    let env = |k: &str, v: usize| {
        std::env::var(k)
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(v)
    };
    let (d, n, reps) = (env("D", 3), env("N", 2000), env("R", 8) as u64);
    let b = if std::env::var("STD").is_ok() {
        standard_truth(d, 5)
    } else {
        hard_truth(d)
    };
    let (bd, pt) = (b.nrows(), b.ncols());
    let (xt, _) = simulate(&b, 200, 99, Shape::Gamma);
    let eta = |c: &DMatrix<f64>, i: usize| -> Vec<f64> {
        (0..bd)
            .map(|k| c[(k, 0)] + (0..pt - 1).map(|j| c[(k, j + 1)] * xt[(i, j)]).sum::<f64>())
            .collect()
    };
    let vtrue: Vec<DMatrix<f64>> = (0..200)
        .map(|i| Esag::from_eta(&eta(&b, i)).unwrap().covariance())
        .collect();
    let rows: Vec<[(bool, usize, f64, f64); 3]> = (820..820 + reps)
        .into_par_iter()
        .map(|seed| {
            let (x, y) = simulate(&b, n, seed, Shape::Gamma);
            let yv: Vec<f64> = (0..n)
                .flat_map(|i| (0..d).map(move |k| (i, k)))
                .map(|(i, k)| y[(i, k)])
                .collect();
            let (mu0, _) = moment_start(&yv, n, d);
            let frames = [GsFrame::gamma(d), GsFrame::gamma_transported(&mu0)];
            let mut out = [(false, 0usize, 0.0, 0.0); 3];
            for (f, fr) in frames.iter().enumerate() {
                let t = Instant::now();
                let (c, conv, it, _, _) = fit_gs_from(fr, &x, &y, true);
                let ms = t.elapsed().as_secs_f64() * 1e3;
                let ve = (0..200)
                    .map(|i| {
                        let e = eta(&c, i);
                        geodesic(&vtrue[i], &fr.covariance(&e[..d], &e[d..]))
                    })
                    .sum::<f64>()
                    / 200.0;
                out[f] = (conv, it, ms, ve);
            }
            let t = Instant::now();
            let g = config(d, Shape::Gamma).fit(&x, &y).unwrap();
            let ms = t.elapsed().as_secs_f64() * 1e3;
            let ve = (0..200)
                .map(|i| {
                    geodesic(
                        &vtrue[i],
                        &Esag::from_eta(&eta(&g.coef, i)).unwrap().covariance(),
                    )
                })
                .sum::<f64>()
                / 200.0;
            out[2] = (g.converged, g.n_iter, ms, ve);
            out
        })
        .collect();
    println!(
        "d={d} n={n} {} truth, mean direction {}:",
        if std::env::var("STD").is_ok() {
            "standard"
        } else {
            "hard"
        },
        if std::env::var("POLE").is_ok() {
            "near the onb pole"
        } else {
            "generic"
        }
    );
    for (f, name) in [
        "gamma, onb frame (prototype)",
        "gamma, transported frame",
        "gamma, production fit",
    ]
    .iter()
    .enumerate()
    {
        let mut it: Vec<usize> = rows.iter().map(|r| r[f].1).collect();
        let mut ms: Vec<f64> = rows.iter().map(|r| r[f].2).collect();
        let mut ve: Vec<f64> = rows.iter().map(|r| r[f].3).collect();
        it.sort_unstable();
        ms.sort_by(f64::total_cmp);
        ve.sort_by(f64::total_cmp);
        println!(
            "  {name:<30} converged {}/{reps}, median iterations {}, median {:.0} ms, V error {:.3}",
            rows.iter().filter(|r| r[f].0).count(),
            it[it.len() / 2],
            ms[ms.len() / 2],
            ve[ve.len() / 2]
        );
    }
}

/// Production fits, `Shape::Gamma` (onb) against `Shape::GammaTransported`,
/// on γ data with a generic mean direction or one near the onb pole
/// (`POLE=1`); `STD=1` for the standard truth. Env `D`, `N`, `R`. One core
/// per fit.
/// `cargo test --release --lib gamma_transported_fits -- --ignored --nocapture`
#[test]
#[ignore]
fn gamma_transported_fits() {
    use rayon::prelude::*;
    let env = |k: &str, v: usize| {
        std::env::var(k)
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(v)
    };
    let (d, n, reps) = (env("D", 3), env("N", 2000), env("R", 12) as u64);
    let b = if std::env::var("STD").is_ok() {
        standard_truth(d, 5)
    } else {
        hard_truth(d)
    };
    let (bd, pt) = (b.nrows(), b.ncols());
    let (xt, _) = simulate(&b, 200, 99, Shape::Gamma);
    let eta = |c: &DMatrix<f64>, i: usize| -> Vec<f64> {
        (0..bd)
            .map(|k| c[(k, 0)] + (0..pt - 1).map(|j| c[(k, j + 1)] * xt[(i, j)]).sum::<f64>())
            .collect()
    };
    let vtrue: Vec<DMatrix<f64>> = (0..200)
        .map(|i| Esag::from_eta(&eta(&b, i)).unwrap().covariance())
        .collect();
    let rows: Vec<[(bool, f64, f64); 2]> = (840..840 + reps)
        .into_par_iter()
        .map(|seed| {
            let (x, y) = simulate(&b, n, seed, Shape::Gamma);
            let mut out = [(false, 0.0, 0.0); 2];
            for (k, shape) in [Shape::Gamma, Shape::GammaTransported]
                .into_iter()
                .enumerate()
            {
                let cfg = EsagRegressor::new().with_shape(shape).with_parallel(false);
                let t = Instant::now();
                let fit = cfg.fit(&x, &y).unwrap();
                let ms = t.elapsed().as_secs_f64() * 1e3;
                let lt = fit.lt_frame();
                let ve = (0..200)
                    .map(|i| {
                        geodesic(
                            &vtrue[i],
                            &fit.distribution_at(&eta(&fit.coef, i), lt.as_ref())
                                .unwrap()
                                .covariance(),
                        )
                    })
                    .sum::<f64>()
                    / 200.0;
                out[k] = (fit.converged, ms, ve);
            }
            out
        })
        .collect();
    println!(
        "d={d} n={n} {} truth, mean direction {}, {reps} datasets:",
        if std::env::var("STD").is_ok() {
            "standard"
        } else {
            "hard"
        },
        if std::env::var("POLE").is_ok() {
            "near the onb pole"
        } else {
            "generic"
        }
    );
    for (k, name) in ["gamma (onb)", "gamma (transported)"].iter().enumerate() {
        let mut ms: Vec<f64> = rows.iter().map(|r| r[k].1).collect();
        let mut ve: Vec<f64> = rows.iter().map(|r| r[k].2).collect();
        ms.sort_by(f64::total_cmp);
        ve.sort_by(f64::total_cmp);
        println!(
            "  {name:<20} converged {:>2}/{reps}, median {:>6.1} ms, max {:>6.1} ms, V error {:.3}",
            rows.iter().filter(|r| r[k].0).count(),
            ms[ms.len() / 2],
            ms[ms.len() - 1],
            ve[ve.len() / 2]
        );
    }
}

// ---------------------------------------------------------------------------
// Spectral shape (experiment): V's tangent block R0 e^A diag(e^ℓ) e^{−A} R0ᵀ,
// eigenvalues as their own (log) parameters and the eigenvectors as one
// rotation, in the frame transported from m. R0 holds the moment start's
// principal axes. Coordinates: p − 1 Helmert coefficients of the sum-zero
// ℓ, then the upper triangle of the skew-symmetric A, row by row.
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct SpFrame {
    lt: crate::logtangent::LtFrame,
    /// p × p row-major: column j is the reference axis j in E coordinates
    r0: Vec<f64>,
}

/// `e^{M}` for a `p × p` row-major matrix, by scaling and squaring with a
/// degree-12 Taylor polynomial (generic, for dual numbers).
fn expm_generic<S: crate::scalar::Scalar>(m: &[S], p: usize) -> Vec<S> {
    let nrm = (0..p)
        .map(|i| (0..p).map(|j| m[i * p + j].re().abs()).sum::<f64>())
        .fold(0.0f64, f64::max);
    let mut sq = 0;
    while nrm / (1u64 << sq) as f64 > 0.5 {
        sq += 1;
    }
    let scale = 1.0 / (1u64 << sq) as f64;
    let b: Vec<S> = m.iter().map(|&v| v * scale).collect();
    let mul = |x: &[S], y: &[S]| -> Vec<S> {
        let mut o = vec![S::zero(); p * p];
        for i in 0..p {
            for k in 0..p {
                let xik = x[i * p + k];
                for j in 0..p {
                    o[i * p + j] += xik * y[k * p + j];
                }
            }
        }
        o
    };
    // Horner: E = I + B(I + B/2(I + B/3(... (I + B/12))))
    let mut e: Vec<S> = vec![S::zero(); p * p];
    for i in 0..p {
        e[i * p + i] = S::one();
    }
    for k in (1..=12).rev() {
        let mut t = mul(&b, &e);
        let inv = 1.0 / k as f64;
        for v in t.iter_mut() {
            *v = *v * inv;
        }
        for i in 0..p {
            t[i * p + i] += S::one();
        }
        e = t;
    }
    for _ in 0..sq {
        e = mul(&e, &e);
    }
    e
}

/// The tangent block's ingredients from the shape coordinates: `ℓ` (sum
/// zero) and `e^{−A}`.
fn sp_parts<S: crate::scalar::Scalar>(sh: &[S], p: usize) -> (Vec<S>, Vec<S>) {
    let mut ell = vec![S::zero(); p];
    for m in 1..p {
        let h = 1.0 / ((m * (m + 1)) as f64).sqrt();
        for k in 0..m {
            ell[k] += sh[m - 1] * h;
        }
        ell[m] -= sh[m - 1] * (m as f64 * h);
    }
    let mut a = vec![S::zero(); p * p];
    let mut idx = p - 1;
    for r in 0..p {
        for c in r + 1..p {
            // −A
            a[r * p + c] = S::zero() - sh[idx];
            a[c * p + r] = sh[idx];
            idx += 1;
        }
    }
    (ell, expm_generic(&a, p))
}

fn loglik_sp<S: crate::scalar::Scalar>(fr: &SpFrame, eta: &[S], y: &[f64]) -> S {
    let d = y.len();
    let p = d - 1;
    let (mu, sh) = eta.split_at(d);
    let m = fr.lt.reference();
    let e = fr.lt.tangent_basis();
    let mut mm = S::zero();
    let mut ymu = S::zero();
    for i in 0..d {
        mm += mu[i] * mu[i];
        ymu += mu[i] * y[i];
    }
    let inv = S::one() / mm.sqrt();
    let mh: Vec<S> = mu.iter().map(|&v| v * inv).collect();
    let t = ymu * inv;
    let v: Vec<S> = (0..d).map(|i| mh[i] + m[i]).collect();
    let vv = v.iter().fold(S::zero(), |acc, &x| acc + x * x);
    let vy = (0..d).fold(S::zero(), |acc, i| acc + v[i] * y[i]);
    let kappa = vy * 2.0 / vv;
    let u: Vec<S> = (0..d).map(|i| S::from_f64(y[i]) - kappa * v[i]).collect();
    let z: Vec<S> = (0..p)
        .map(|j| (0..d).fold(S::zero(), |acc, i| acc + u[i] * e[j * d + i]))
        .collect();
    // z0 = R0ᵀ z, w = e^{−A} z0
    let z0: Vec<S> = (0..p)
        .map(|j| (0..p).fold(S::zero(), |acc, i| acc + z[i] * fr.r0[i * p + j]))
        .collect();
    let (ell, ema) = sp_parts(sh, p);
    let mut q = t * t;
    for j in 0..p {
        let wj = (0..p).fold(S::zero(), |acc, k| acc + ema[j * p + k] * z0[k]);
        q += wj * wj * (S::zero() - ell[j]).exp();
    }
    let alpha = ymu / q.sqrt();
    S::from_f64(-0.5 * p as f64 * crate::stats::normal::LN_2PI) - q.ln() * (0.5 * d as f64)
        + (alpha * alpha - mm) * 0.5
        + crate::loglik::log_m_p(alpha, p)
}

fn sp_grad(fr: &SpFrame, eta: &[f64], y: &[f64], g: &mut [f64]) -> f64 {
    use crate::dual::{Dual, seed_array};
    fn go<const N: usize>(fr: &SpFrame, eta: &[f64], y: &[f64], g: &mut [f64]) -> f64 {
        let x: [Dual<N>; N] = seed_array(eta);
        let v = loglik_sp(fr, &x, y);
        g.copy_from_slice(&v.eps);
        v.re
    }
    match eta.len() {
        5 => go::<5>(fr, eta, y, g),
        9 => go::<9>(fr, eta, y, g),
        14 => go::<14>(fr, eta, y, g),
        20 => go::<20>(fr, eta, y, g),
        35 => go::<35>(fr, eta, y, g),
        n => panic!("eta length {n}"),
    }
}

/// `V` of the spectral shape at `eta`.
fn sp_covariance(fr: &SpFrame, eta: &[f64]) -> DMatrix<f64> {
    let d = fr.lt.dim();
    let p = d - 1;
    let (ell, ema) = sp_parts(&eta[d..], p);
    let xi = fr.lt.frame_at(&eta[..d]);
    let r0 = DMatrix::from_row_slice(p, p, &fr.r0);
    let ea = DMatrix::from_row_slice(p, p, &ema).transpose(); // e^{A}
    let rot = &xi * r0 * ea;
    let lam = DMatrix::from_diagonal(&nalgebra::DVector::from_iterator(
        p,
        ell.iter().map(|l| l.exp()),
    ));
    let nm = eta[..d].iter().map(|v| v * v).sum::<f64>().sqrt();
    let mh = nalgebra::DVector::from_iterator(d, eta[..d].iter().map(|v| v / nm));
    &mh * mh.transpose() + &rot * lam * rot.transpose()
}

/// Whitened L-BFGS for a per-observation kernel `kg` from `start` (the full
/// coefficient vector, column-major). Returns (coef, converged, iterations).
/// A per-observation log-density and gradient in `eta`.
type ObsKernel<'a> = &'a (dyn Fn(&[f64], &[f64], &mut [f64]) -> f64 + Sync);

fn fit_kernel(
    kg: ObsKernel<'_>,
    start: Vec<f64>,
    x: &DMatrix<f64>,
    y: &DMatrix<f64>,
) -> (DMatrix<f64>, bool, usize) {
    use crate::optim::lbfgs::{FnObjective, LbfgsConfig, minimize};
    let (n, d, p) = (y.nrows(), y.ncols(), x.ncols());
    let (bd, pt) = (dims::eta_len(d), p + 1);
    let nk = bd * pt;
    let yv: Vec<f64> = (0..n)
        .flat_map(|i| (0..d).map(move |k| (i, k)))
        .map(|(i, k)| y[(i, k)])
        .collect();
    let xr = |i: usize, j: usize| if j == 0 { 1.0 } else { x[(i, j - 1)] };
    let mut info = DMatrix::<f64>::zeros(nk, nk);
    let (mut eta, mut g) = (vec![0.0; bd], vec![0.0; bd]);
    let mut s = vec![0.0; nk];
    for i in 0..n {
        for (k, e) in eta.iter_mut().enumerate() {
            *e = (0..pt).map(|j| start[j * bd + k] * xr(i, j)).sum();
        }
        kg(&eta, &yv[i * d..(i + 1) * d], &mut g);
        for j in 0..pt {
            for k in 0..bd {
                s[j * bd + k] = g[k] * xr(i, j);
            }
        }
        for a in 0..nk {
            for c in 0..nk {
                info[(a, c)] += s[a] * s[c];
            }
        }
    }
    let lt_inv = info
        .cholesky()
        .unwrap()
        .l()
        .transpose()
        .try_inverse()
        .unwrap();
    let to_b = |th: &[f64]| -> Vec<f64> {
        let v = &lt_inv * nalgebra::DVector::from_column_slice(th);
        (0..nk).map(|a| start[a] + v[a]).collect()
    };
    let obj = FnObjective(|th: &[f64], out: &mut [f64]| {
        let b = to_b(th);
        let mut grad = vec![0.0; nk];
        let (mut eta, mut g) = (vec![0.0; bd], vec![0.0; bd]);
        let mut f = 0.0;
        for i in 0..n {
            for (k, e) in eta.iter_mut().enumerate() {
                *e = (0..pt).map(|j| b[j * bd + k] * xr(i, j)).sum();
            }
            f -= kg(&eta, &yv[i * d..(i + 1) * d], &mut g);
            for j in 0..pt {
                let w = xr(i, j);
                for k in 0..bd {
                    grad[j * bd + k] -= g[k] * w;
                }
            }
        }
        let gv = lt_inv.transpose() * nalgebra::DVector::from_column_slice(&grad);
        out.copy_from_slice(gv.as_slice());
        f
    });
    let cfg = LbfgsConfig {
        memory: nk.clamp(10, 200),
        max_iter: 2000,
        max_fev: 40_000,
        tol_grad: 1e-6,
        ..LbfgsConfig::default()
    };
    let res = minimize(&obj, &vec![0.0; nk], &cfg);
    (
        DMatrix::from_column_slice(bd, pt, &to_b(&res.x)),
        res.status.is_converged(),
        res.n_iter,
    )
}

/// Does the spectral shape (eigenvalues + one rotation e^A) fit without
/// traps? On γ data (hard truth unless `STD=1`; env `D`, `N`, `R`):
/// convergence, iterations and `V̂(x)` error against the γ and log-tangent
/// production fits on the same data.
/// `cargo test --release --lib spectral_fits -- --ignored --nocapture`
#[test]
#[ignore]
fn spectral_fits() {
    use rayon::prelude::*;
    let env = |k: &str, v: usize| {
        std::env::var(k)
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(v)
    };
    let (d, n, reps) = (env("D", 5), env("N", 2000), env("R", 8) as u64);
    let p = d - 1;
    let b = if std::env::var("STD").is_ok() {
        standard_truth(d, 5)
    } else {
        hard_truth(d)
    };
    let (bd, pt) = (b.nrows(), b.ncols());
    let (xt, _) = simulate(&b, 200, 99, Shape::Gamma);
    let eta_at = |c: &DMatrix<f64>, i: usize| -> Vec<f64> {
        (0..bd)
            .map(|k| c[(k, 0)] + (0..pt - 1).map(|j| c[(k, j + 1)] * xt[(i, j)]).sum::<f64>())
            .collect()
    };
    let vtrue: Vec<DMatrix<f64>> = (0..200)
        .map(|i| Esag::from_eta(&eta_at(&b, i)).unwrap().covariance())
        .collect();
    let rows: Vec<[(bool, usize, f64, f64); 3]> = (860..860 + reps)
        .into_par_iter()
        .map(|seed| {
            let (x, y) = simulate(&b, n, seed, Shape::Gamma);
            let yv: Vec<f64> = (0..n)
                .flat_map(|i| (0..d).map(move |k| (i, k)))
                .map(|(i, k)| y[(i, k)])
                .collect();
            let (mu0, g0) = moment_start(&yv, n, d);
            let v0 = Esag::new(&mu0, &g0).unwrap().covariance();
            let nm = mu0.iter().map(|v| v * v).sum::<f64>().sqrt();
            let m: Vec<f64> = mu0.iter().map(|v| v / nm).collect();
            let lt = crate::logtangent::LtFrame::new(&m);
            let e = DMatrix::from_column_slice(d, p, lt.tangent_basis());
            let tb = e.transpose() * &v0 * &e;
            let eig = nalgebra::SymmetricEigen::new((&tb + tb.transpose()) * 0.5);
            let mut w = eig.eigenvectors.clone();
            if w.determinant() < 0.0 {
                for r in 0..p {
                    w[(r, 0)] = -w[(r, 0)];
                }
            }
            let r0: Vec<f64> = (0..p)
                .flat_map(|r| (0..p).map(move |c| (r, c)))
                .map(|(r, c)| w[(r, c)])
                .collect();
            let fr = SpFrame { lt, r0 };
            // start: ℓ = centred log-eigenvalues on the Helmert basis, A = 0
            let ell: Vec<f64> = eig.eigenvalues.iter().map(|l| l.ln()).collect();
            let mean = ell.iter().sum::<f64>() / p as f64;
            let ell: Vec<f64> = ell.iter().map(|l| l - mean).collect();
            let mut start = vec![0.0; bd * pt];
            start[..d].copy_from_slice(&mu0);
            for mm in 1..p {
                let h = 1.0 / ((mm * (mm + 1)) as f64).sqrt();
                start[d + mm - 1] =
                    (0..mm).map(|k| ell[k] * h).sum::<f64>() - ell[mm] * mm as f64 * h;
            }
            let kg = |eta: &[f64], yi: &[f64], g: &mut [f64]| sp_grad(&fr, eta, yi, g);
            let t = Instant::now();
            let (c, conv, it) = fit_kernel(&kg, start, &x, &y);
            let ms = t.elapsed().as_secs_f64() * 1e3;
            let ve = (0..200)
                .map(|i| geodesic(&vtrue[i], &sp_covariance(&fr, &eta_at(&c, i))))
                .sum::<f64>()
                / 200.0;
            let mut out = [
                (conv, it, ms, ve),
                (false, 0, 0.0, 0.0),
                (false, 0, 0.0, 0.0),
            ];
            for (k, shape) in [Shape::Gamma, Shape::LogTangent].into_iter().enumerate() {
                let t = Instant::now();
                let f = EsagRegressor::new()
                    .with_shape(shape)
                    .with_parallel(false)
                    .fit(&x, &y)
                    .unwrap();
                let ms = t.elapsed().as_secs_f64() * 1e3;
                let lt = f.lt_frame();
                let ve = (0..200)
                    .map(|i| {
                        geodesic(
                            &vtrue[i],
                            &f.distribution_at(&eta_at(&f.coef, i), lt.as_ref())
                                .unwrap()
                                .covariance(),
                        )
                    })
                    .sum::<f64>()
                    / 200.0;
                out[k + 1] = (f.converged, f.n_iter, ms, ve);
            }
            out
        })
        .collect();
    println!(
        "d={d} n={n} {} truth, gamma data, {reps} datasets:",
        if std::env::var("STD").is_ok() {
            "standard"
        } else {
            "hard"
        }
    );
    for (k, name) in [
        "spectral (prototype)",
        "gamma (production)",
        "log-tangent (production)",
    ]
    .iter()
    .enumerate()
    {
        let mut it: Vec<usize> = rows.iter().map(|r| r[k].1).collect();
        let mut ve: Vec<f64> = rows.iter().map(|r| r[k].3).collect();
        it.sort_unstable();
        ve.sort_by(f64::total_cmp);
        println!(
            "  {name:<26} converged {}/{reps}, median iterations {}, V error {:.3}",
            rows.iter().filter(|r| r[k].0).count(),
            it[it.len() / 2],
            ve[ve.len() / 2]
        );
    }
}

/// How anisotropic are the standard and hard γ truths, and does the
/// likelihood prefer γ over log-tangent on their data? Per truth: mean
/// geodesic distance of V(x) from I, median largest/smallest tangent
/// eigenvalue ratio, and on γ data (n = 2000) the log-likelihood of the γ
/// and log-tangent fits (equal parameter counts, so ΔAIC = −2Δll).
/// `cargo test --release --lib truth_anisotropy -- --ignored --nocapture`
#[test]
#[ignore]
fn truth_anisotropy() {
    use rayon::prelude::*;
    for d in [5usize, 8] {
        for (name, b) in [("standard", standard_truth(d, 5)), ("hard", hard_truth(d))] {
            let (x, _) = simulate(&b, 500, 1, Shape::Gamma);
            let eye = DMatrix::<f64>::identity(d, d);
            let mut ratios = vec![];
            let mut dist = 0.0;
            for i in 0..500 {
                let eta: Vec<f64> = (0..b.nrows())
                    .map(|k| b[(k, 0)] + (0..5).map(|j| b[(k, j + 1)] * x[(i, j)]).sum::<f64>())
                    .collect();
                let fr = crate::frame::Frame::new(&eta[..d], &eta[d..]);
                let lmax = fr.lambda.iter().fold(0.0f64, |a, &v| a.max(v));
                let lmin = fr.lambda.iter().fold(f64::INFINITY, |a, &v| a.min(v));
                ratios.push(lmax / lmin);
                dist += geodesic(&eye, &Esag::from_eta(&eta).unwrap().covariance());
            }
            ratios.sort_by(f64::total_cmp);
            let gaps: Vec<f64> = (900u64..904)
                .into_par_iter()
                .map(|seed| {
                    let (x, y) = simulate(&b, 2000, seed, Shape::Gamma);
                    let g = EsagRegressor::new()
                        .with_parallel(false)
                        .fit(&x, &y)
                        .unwrap();
                    let l = EsagRegressor::new()
                        .with_shape(Shape::LogTangent)
                        .with_parallel(false)
                        .fit(&x, &y)
                        .unwrap();
                    g.log_likelihood - l.log_likelihood
                })
                .collect();
            println!(
                "d={d} {name:<8}: V(x) {:.2} from isotropy, median eigenvalue ratio {:.1}; ll(gamma) - ll(log-tangent) on gamma data: {:?}",
                dist / 500.0,
                ratios[250],
                gaps.iter().map(|v| format!("{v:+.1}")).collect::<Vec<_>>()
            );
        }
    }
}

/// Smallest relative trailing-pair norm (groups `j ≥ 2`) of each
/// observation's γ at `b` (1 where there is no such group).
fn pair_ratios(data: &RegressionData, b: &DMatrix<f64>) -> Vec<f64> {
    let (d, bd, pt) = (data.d, data.b_dim, data.p_total);
    let mut eta = vec![0.0; bd];
    (0..data.n)
        .map(|i| {
            for (k, e) in eta.iter_mut().enumerate() {
                *e = (0..pt).map(|j| b[(k, j)] * data.x[i * pt + j]).sum();
            }
            let r = min_pair_ratio(&eta[d..], d).0;
            if r.is_finite() { r } else { 1.0 }
        })
        .collect()
}

/// Saddle-free trust-region Newton ascent on the γ likelihood with exact
/// Hessians: steps `(Q|Λ|Qᵀ)⁻¹ g` from the eigen-decomposition of `−H`
/// (absolute values, floored), scaled to the trust radius in that metric,
/// radius adapted by actual/predicted gain, and a guard rejecting steps
/// that take an observation's trailing-pair ratio below half its current
/// value once it is under 1e−2. Returns (coef, regular root reached,
/// iterations, max|g|).
fn newton_tr(
    data: &RegressionData,
    b0: &DMatrix<f64>,
    iters: usize,
) -> (DMatrix<f64>, bool, usize, f64) {
    let (bd, pt) = (data.b_dim, data.p_total);
    let k = bd * pt;
    let mut b = b0.clone();
    let mut radius = 10.0;
    let mut ratios = pair_ratios(data, &b);
    let mut gmax = f64::INFINITY;
    for it in 0..iters {
        let (f, g, h) = joint_loglik_grad_hess(data, &b, false);
        gmax = g.iter().fold(0.0f64, |a, v| a.max(v.abs()));
        let neg = -DMatrix::from_column_slice(k, k, &h);
        let neg = (&neg + neg.transpose()) * 0.5;
        let eig = nalgebra::SymmetricEigen::new(neg.clone());
        let lmax = eig.eigenvalues.iter().fold(0.0f64, |a, v| a.max(v.abs()));
        let nd = eig.eigenvalues.iter().all(|&v| v > 0.0);
        if gmax < 1e-6 * f.abs().max(1.0) && nd {
            return (b, true, it, gmax);
        }
        let floor = 1e-10 * lmax;
        let gq = eig.eigenvectors.transpose() * DVector::from_column_slice(&g);
        // step and its M-norm in eigen-coordinates
        let mut sq = DVector::zeros(k);
        let mut mnorm2 = 0.0;
        for a in 0..k {
            let l = eig.eigenvalues[a].abs().max(floor);
            sq[a] = gq[a] / l;
            mnorm2 += gq[a] * gq[a] / l;
        }
        let mnorm = mnorm2.sqrt();
        let mut accepted = false;
        for _ in 0..30 {
            let scale = if mnorm > radius { radius / mnorm } else { 1.0 };
            let s = &eig.eigenvectors * (&sq * scale);
            let mut cand = b.clone();
            for j in 0..pt {
                for r in 0..bd {
                    cand[(r, j)] += s[j * bd + r];
                }
            }
            // predicted gain of the modified model
            let pred = scale * mnorm2 - 0.5 * scale * scale * mnorm2;
            let fc = joint_loglik_and_grad(data, &cand, false).0;
            let new_ratios = pair_ratios(data, &cand);
            let unsafe_step = ratios
                .iter()
                .zip(&new_ratios)
                .any(|(&old, &new)| new < 1e-2 && new < 0.5 * old);
            let gain = fc - f;
            if fc.is_finite()
                && !unsafe_step
                && gain > 0.0
                && gain >= 0.1 * pred.min(gain.abs() * 10.0)
            {
                if gain > 0.75 * pred && scale < 1.0 {
                    radius *= 2.0;
                }
                b = cand;
                ratios = new_ratios;
                accepted = true;
                break;
            }
            radius = (radius.min(mnorm * scale)) * 0.25;
            if radius < 1e-12 {
                break;
            }
        }
        if !accepted {
            return (b, false, it, gmax);
        }
    }
    (b, false, iters, gmax)
}

/// Can a saddle-free trust-region Newton reach γ's regular root on hard
/// data, from the truth and from the production fit? Env `D`, `N`, `R`,
/// `STD=1`.
/// `cargo test --release --lib gamma_newton_tr -- --ignored --nocapture`
#[test]
#[ignore]
fn gamma_newton_tr() {
    use rayon::prelude::*;
    let env = |k: &str, v: usize| {
        std::env::var(k)
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(v)
    };
    let (d, n, reps) = (env("D", 5), env("N", 2000), env("R", 8) as u64);
    let b = if std::env::var("STD").is_ok() {
        standard_truth(d, 5)
    } else {
        hard_truth(d)
    };
    let (bd, pt) = (b.nrows(), b.ncols());
    let (xt, _) = simulate(&b, 200, 99, Shape::Gamma);
    let eta_at = |c: &DMatrix<f64>, i: usize| -> Vec<f64> {
        (0..bd)
            .map(|k| c[(k, 0)] + (0..pt - 1).map(|j| c[(k, j + 1)] * xt[(i, j)]).sum::<f64>())
            .collect()
    };
    let vtrue: Vec<DMatrix<f64>> = (0..200)
        .map(|i| Esag::from_eta(&eta_at(&b, i)).unwrap().covariance())
        .collect();
    let verr = |c: &DMatrix<f64>| {
        (0..200)
            .map(|i| {
                geodesic(
                    &vtrue[i],
                    &Esag::from_eta(&eta_at(c, i)).unwrap().covariance(),
                )
            })
            .sum::<f64>()
            / 200.0
    };
    let rows: Vec<String> = (880..880 + reps)
        .into_par_iter()
        .map(|seed| {
            let (x, y) = simulate(&b, n, seed, Shape::Gamma);
            let cfg = config(d, Shape::Gamma);
            let data = cfg.data(&x, &y).unwrap();
            let t = Instant::now();
            let prod = cfg.fit(&x, &y).unwrap();
            let tp = t.elapsed().as_secs_f64();
            let t = Instant::now();
            let (c1, r1, i1, _) = newton_tr(&data, &prod.coef, 100);
            let t1 = t.elapsed().as_secs_f64();
            let t = Instant::now();
            let (c2, r2, i2, _) = newton_tr(&data, &b, 100);
            let t2 = t.elapsed().as_secs_f64();
            let ll = |c: &DMatrix<f64>| joint_loglik_and_grad(&data, c, false).0;
            format!(
                "seed {seed}: production {} ll {:.2} V {:.3} ({tp:.2} s) | Newton from it: regular {} after {i1} its, ll {:+.2}, V {:.3} ({t1:.2} s) | Newton from truth: regular {} after {i2} its, ll {:+.2}, V {:.3} ({t2:.2} s)",
                if prod.converged { "conv" } else { "trap" },
                prod.log_likelihood,
                verr(&prod.coef),
                r1 as u8,
                ll(&c1) - prod.log_likelihood,
                verr(&c1),
                r2 as u8,
                ll(&c2) - prod.log_likelihood,
                verr(&c2)
            )
        })
        .collect();
    println!(
        "d={d} n={n} {} truth:",
        if std::env::var("STD").is_ok() {
            "standard"
        } else {
            "hard"
        }
    );
    for r in rows {
        println!("  {r}");
    }
}

/// γ (and log-tangent) production fits at large d on the standard truth's γ
/// data: convergence, iterations, one-core time, `V̂(x)` error. Env `DS`
/// (default 8,9,10), `N`, `R`.
/// `cargo test --release --lib gamma_big_d -- --ignored --nocapture`
#[test]
#[ignore]
fn gamma_big_d() {
    use rayon::prelude::*;
    let env = |k: &str, v: usize| {
        std::env::var(k)
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(v)
    };
    let (n, reps) = (env("N", 2000), env("R", 6) as u64);
    let ds: Vec<usize> = std::env::var("DS")
        .unwrap_or_else(|_| "8,9,10".into())
        .split(',')
        .map(|v| v.parse().unwrap())
        .collect();
    for d in ds {
        let b = if std::env::var("HARD").is_ok() {
            hard_truth(d)
        } else {
            standard_truth(d, 5)
        };
        let (bd, pt) = (b.nrows(), b.ncols());
        let (xt, _) = simulate(&b, 200, 99, Shape::Gamma);
        let eta_at = |c: &DMatrix<f64>, i: usize| -> Vec<f64> {
            (0..bd)
                .map(|k| c[(k, 0)] + (0..pt - 1).map(|j| c[(k, j + 1)] * xt[(i, j)]).sum::<f64>())
                .collect()
        };
        let vtrue: Vec<DMatrix<f64>> = (0..200)
            .map(|i| Esag::from_eta(&eta_at(&b, i)).unwrap().covariance())
            .collect();
        // per shape: converged, iterations, seconds, V error, set aside
        type Row = [(bool, usize, f64, f64, usize); 2];
        let rows: Vec<Row> = (900..900 + reps)
            .into_par_iter()
            .map(|seed| {
                let (x, y) = simulate(&b, n, seed, Shape::Gamma);
                let mut out = [(false, 0, 0.0, 0.0, 0); 2];
                for (k, shape) in [Shape::Gamma, Shape::LogTangent].into_iter().enumerate() {
                    let t = Instant::now();
                    let f = EsagRegressor::new()
                        .with_shape(shape)
                        .with_parallel(false)
                        .fit(&x, &y)
                        .unwrap();
                    let s = t.elapsed().as_secs_f64();
                    let lt = f.lt_frame();
                    let ve = (0..200)
                        .map(|i| {
                            geodesic(
                                &vtrue[i],
                                &f.distribution_at(&eta_at(&f.coef, i), lt.as_ref())
                                    .unwrap()
                                    .covariance(),
                            )
                        })
                        .sum::<f64>()
                        / 200.0;
                    out[k] = (f.converged, f.n_iter, s, ve, f.set_aside_obs.len());
                }
                out
            })
            .collect();
        for (k, name) in ["gamma", "log-tangent"].iter().enumerate() {
            let mut t: Vec<f64> = rows.iter().map(|r| r[k].2).collect();
            let mut ve: Vec<f64> = rows.iter().map(|r| r[k].3).collect();
            let mut it: Vec<usize> = rows.iter().map(|r| r[k].1).collect();
            t.sort_by(f64::total_cmp);
            ve.sort_by(f64::total_cmp);
            it.sort_unstable();
            println!(
                "d={d:>2} K={:>3} {} {name:<12} converged {}/{reps} (set-aside fits {}, sizes {:?}), median {:>4} iterations, {:.2} s, V error {:.3}",
                bd * pt,
                if std::env::var("HARD").is_ok() {
                    "hard    "
                } else {
                    "standard"
                },
                rows.iter().filter(|r| r[k].0).count(),
                rows.iter().filter(|r| r[k].4 > 0).count(),
                rows.iter()
                    .filter(|r| r[k].4 > 0)
                    .map(|r| r[k].4)
                    .collect::<Vec<_>>(),
                it[it.len() / 2],
                t[t.len() / 2],
                ve[ve.len() / 2]
            );
        }
    }
}

/// Why do γ fits at d = 10 (standard truth) stop unconverged? Status,
/// gradient, trapped/degenerate observations, and the truth's anisotropy.
/// `cargo test --release --lib gamma_d10_diagnosis -- --ignored --nocapture`
#[test]
#[ignore]
fn gamma_d10_diagnosis() {
    let d: usize = std::env::var("D")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);
    let b = standard_truth(d, 5);
    let seeds: Vec<u64> = std::env::var("SEEDS")
        .unwrap_or_else(|_| "900,901,902".into())
        .split(',')
        .map(|v| v.parse().unwrap())
        .collect();
    for seed in seeds {
        let (x, y) = simulate(&b, 2000, seed, Shape::Gamma);
        let cfg = EsagRegressor::new().with_parallel(false);
        let data = cfg.data(&x, &y).unwrap();
        let f = cfg.fit(&x, &y).unwrap();
        let pr = pair_ratios(&data, &f.coef);
        let (imin, rmin) = pr
            .iter()
            .enumerate()
            .fold((0, 1.0f64), |a, (i, &r)| if r < a.1 { (i, r) } else { a });
        println!("seed {seed}: smallest pair ratio {rmin:.1e} (#{imin})");
        let tll = joint_loglik_and_grad(&data, &b, false).0;
        let lams: Vec<f64> = {
            let eta: Vec<f64> = (0..b.nrows()).map(|k| b[(k, 0)]).collect();
            crate::frame::Frame::new(&eta[..d], &eta[d..]).lambda
        };
        let ratio = lams.iter().fold(0.0f64, |a, &v| a.max(v))
            / lams.iter().fold(f64::INFINITY, |a, &v| a.min(v));
        println!(
            "seed {seed}: status {:?}, converged {}, iterations {}, evaluations {}, |g| {:.2e}, singular {}, degenerate {}, ll {:.2} (truth {:.2}); truth's eigenvalue ratio at x = 0: {ratio:.0}",
            f.status,
            f.converged,
            f.n_iter,
            f.n_fev,
            f.grad_norm,
            f.singular_obs.len(),
            f.degenerate_obs.len(),
            f.log_likelihood,
            tll
        );
    }
}

/// From the truth, does the γ optimizer reach a regular root at large d
/// (standard truth)? Warm fit and trust-region Newton from the truth,
/// against the default fit. Env `D` (default 10), `R`.
/// `cargo test --release --lib gamma_big_d_from_truth -- --ignored --nocapture`
#[test]
#[ignore]
fn gamma_big_d_from_truth() {
    use rayon::prelude::*;
    let env = |k: &str, v: usize| {
        std::env::var(k)
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(v)
    };
    let (d, reps) = (env("D", 10), env("R", 4) as u64);
    let b = standard_truth(d, 5);
    let rows: Vec<String> = (900..900 + reps)
        .into_par_iter()
        .map(|seed| {
            let (x, y) = simulate(&b, 2000, seed, Shape::Gamma);
            let cfg = EsagRegressor::new().with_parallel(false);
            let data = cfg.data(&x, &y).unwrap();
            let tll = joint_loglik_and_grad(&data, &b, false).0;
            let t = Instant::now();
            let def = cfg.fit(&x, &y).unwrap();
            let t0 = t.elapsed().as_secs_f64();
            let t = Instant::now();
            let warm = cfg.fit_warm(&x, &y, &b).unwrap();
            let t1 = t.elapsed().as_secs_f64();
            let t = Instant::now();
            let (c, reg, its, _) = newton_tr(&data, &b, 60);
            let t2 = t.elapsed().as_secs_f64();
            let nll = joint_loglik_and_grad(&data, &c, false).0;
            format!(
                "seed {seed}: default {} ll-truth {:+.1} ({t0:.2} s) | warm from truth {} ll-truth {:+.1} ({t1:.2} s) | Newton from truth regular {} after {its} its, ll-truth {:+.1} ({t2:.2} s)",
                if def.converged { "conv" } else { "stop" },
                def.log_likelihood - tll,
                if warm.converged { "conv" } else { "stop" },
                warm.log_likelihood - tll,
                reg as u8,
                nll - tll
            )
        })
        .collect();
    println!(
        "d={d} standard truth, n = 2000 (a regular root is about K/2 = {} above the truth):",
        dims::eta_len(d) * 6 / 2
    );
    for r in rows {
        println!("  {r}");
    }
}

/// Staged start for γ: fit μ(x) with a constant shape (no traps: all
/// observations share γ), then the full model warm from there; against the
/// default fit. Log-likelihood relative to the truth (a regular root is
/// about K/2 above it), convergence and one-core time. Env `DS`, `HARD=1`,
/// `R`, `N`.
/// `cargo test --release --lib gamma_staged_start -- --ignored --nocapture`
#[test]
#[ignore]
fn gamma_staged_start() {
    use rayon::prelude::*;
    let env = |k: &str, v: usize| {
        std::env::var(k)
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(v)
    };
    let (n, reps) = (env("N", 2000), env("R", 8) as u64);
    let ds: Vec<usize> = std::env::var("DS")
        .unwrap_or_else(|_| "8,9,10".into())
        .split(',')
        .map(|v| v.parse().unwrap())
        .collect();
    let hard = std::env::var("HARD").is_ok();
    for d in ds {
        let b = if hard {
            hard_truth(d)
        } else {
            standard_truth(d, 5)
        };
        let (bd, pt) = (b.nrows(), b.ncols());
        let rows: Vec<[(bool, f64, f64); 2]> = (920..920 + reps)
            .into_par_iter()
            .map(|seed| {
                let (x, y) = simulate(&b, n, seed, Shape::Gamma);
                let cfg = EsagRegressor::new().with_parallel(false);
                let data = cfg.data(&x, &y).unwrap();
                let tll = joint_loglik_and_grad(&data, &b, false).0;
                let t = Instant::now();
                let def = cfg.fit(&x, &y).unwrap();
                let t0 = t.elapsed().as_secs_f64();
                let t = Instant::now();
                let constant = DMatrix::from_fn(bd, pt, |k, j| k < d || j == 0);
                let stage1 = cfg
                    .clone()
                    .with_mask(Some(constant))
                    .with_hessian(false)
                    .fit(&x, &y)
                    .unwrap();
                let full = cfg.fit_warm(&x, &y, &stage1.coef).unwrap();
                let t1 = t.elapsed().as_secs_f64();
                [
                    (def.converged, def.log_likelihood - tll, t0),
                    (full.converged, full.log_likelihood - tll, t1),
                ]
            })
            .collect();
        for (k, name) in ["default start", "staged start"].iter().enumerate() {
            let mut t: Vec<f64> = rows.iter().map(|r| r[k].2).collect();
            t.sort_by(f64::total_cmp);
            let lls: Vec<String> = rows.iter().map(|r| format!("{:+.0}", r[k].1)).collect();
            println!(
                "d={d:>2} {} K={:>3} {name:<14} converged {}/{reps}, median {:.2} s, ll - truth [{}] (regular root ≈ +{})",
                if hard { "hard    " } else { "standard" },
                bd * pt,
                rows.iter().filter(|r| r[k].0).count(),
                t[t.len() / 2],
                lls.join(" "),
                bd * pt / 2
            );
        }
    }
}

/// The moment start whitened by the exact information at the start (the
/// log-tangent single start), for γ: converged and log-likelihood against
/// the truth, beside the default race. Env `DS`, `HARD=1`, `R`.
/// `cargo test --release --lib gamma_whitened_start -- --ignored --nocapture`
#[test]
#[ignore]
fn gamma_whitened_start() {
    use rayon::prelude::*;
    let env = |k: &str, v: usize| {
        std::env::var(k)
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(v)
    };
    let reps = env("R", 8) as u64;
    let hard = std::env::var("HARD").is_ok();
    let ds: Vec<usize> = std::env::var("DS")
        .unwrap_or_else(|_| "10".into())
        .split(',')
        .map(|v| v.parse().unwrap())
        .collect();
    for d in ds {
        let b = if hard {
            hard_truth(d)
        } else {
            standard_truth(d, 5)
        };
        let rows: Vec<[(bool, f64, f64); 2]> = (920..920 + reps)
            .into_par_iter()
            .map(|seed| {
                let (x, y) = simulate(&b, 2000, seed, Shape::Gamma);
                let cfg = EsagRegressor::new().with_parallel(false);
                let data = cfg.data(&x, &y).unwrap();
                let tll = joint_loglik_and_grad(&data, &b, false).0;
                let t = Instant::now();
                let def = cfg.fit(&x, &y).unwrap();
                let t0 = t.elapsed().as_secs_f64();
                let t = Instant::now();
                let mask = cfg.resolve_mask(&data).unwrap();
                let free = free_params(&mask);
                let b0 = cfg
                    .clone()
                    .with_init(Init::DataDriven)
                    .initial_coef(&data)
                    .unwrap();
                let mut wc = cfg.clone();
                wc.local_precondition = false;
                wc.precondition = info_precondition(&data, &b0, &free, false);
                let raw = wc.optimize(&data, &b0, &free, cfg.max_iter);
                let t1 = t.elapsed().as_secs_f64();
                [
                    (def.converged, def.log_likelihood - tll, t0),
                    (raw.converged, -raw.neg_ll - tll, t1),
                ]
            })
            .collect();
        for (k, name) in ["default race", "whitened moment start"].iter().enumerate() {
            let lls: Vec<String> = rows.iter().map(|r| format!("{:+.0}", r[k].1)).collect();
            let mut t: Vec<f64> = rows.iter().map(|r| r[k].2).collect();
            t.sort_by(f64::total_cmp);
            println!(
                "d={d:>2} {} {name:<22} converged {}/{reps}, median {:.2} s, ll - truth [{}] (regular root ≈ +{})",
                if hard { "hard    " } else { "standard" },
                rows.iter().filter(|r| r[k].0).count(),
                t[t.len() / 2],
                lls.join(" "),
                dims::eta_len(d) * 6 / 2
            );
        }
    }
}

/// Is the γ regression's difficulty at large d the branch of the start?
/// Starts with every slope at zero: the truth's intercept (in its own
/// branch), and the canonical member of the same V (the branch the moment
/// start would use). Env `D` (default 10), `R`, `HARD=1`.
/// `cargo test --release --lib gamma_branch_start -- --ignored --nocapture`
#[test]
#[ignore]
fn gamma_branch_start() {
    use rayon::prelude::*;
    let env = |k: &str, v: usize| {
        std::env::var(k)
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(v)
    };
    let (d, reps) = (env("D", 10), env("R", 8) as u64);
    let hard = std::env::var("HARD").is_ok();
    let b = if hard {
        hard_truth(d)
    } else {
        standard_truth(d, 5)
    };
    let (bd, pt) = (b.nrows(), b.ncols());
    let mu0: Vec<f64> = (0..d).map(|k| b[(k, 0)]).collect();
    let g0: Vec<f64> = (d..bd).map(|k| b[(k, 0)]).collect();
    let v0 = crate::frame::Frame::new(&mu0, &g0).cov_pow(1.0);
    let canon = crate::param::gamma_from_covariance(&mu0, &v0);
    let diff = g0
        .iter()
        .zip(&canon)
        .map(|(a, c)| (a - c).abs())
        .fold(0.0f64, f64::max);
    println!(
        "d={d} {}: truth intercept vs its canonical member: max |difference| {diff:.3}",
        if hard { "hard" } else { "standard" }
    );
    let rows: Vec<[(bool, f64); 2]> = (920..920 + reps)
        .into_par_iter()
        .map(|seed| {
            let (x, y) = simulate(&b, 2000, seed, Shape::Gamma);
            let cfg = EsagRegressor::new().with_parallel(false);
            let data = cfg.data(&x, &y).unwrap();
            let tll = joint_loglik_and_grad(&data, &b, false).0;
            let mut out = [(false, 0.0); 2];
            for (k, g) in [&g0, &canon].into_iter().enumerate() {
                let mut s = DMatrix::zeros(bd, pt);
                for r in 0..d {
                    s[(r, 0)] = mu0[r];
                }
                for (r, &v) in g.iter().enumerate() {
                    s[(d + r, 0)] = v;
                }
                let f = cfg.fit_warm(&x, &y, &s).unwrap();
                out[k] = (f.converged, f.log_likelihood - tll);
            }
            out
        })
        .collect();
    for (k, name) in [
        "truth's branch, zero slopes",
        "canonical branch, zero slopes",
    ]
    .iter()
    .enumerate()
    {
        let lls: Vec<String> = rows.iter().map(|r| format!("{:+.0}", r[k].1)).collect();
        println!(
            "  {name:<30} converged {}/{reps}, ll - truth [{}] (regular root ≈ +{})",
            rows.iter().filter(|r| r[k].0).count(),
            lls.join(" "),
            bd * pt / 2
        );
    }
}

/// Start the γ regression at the i.i.d. MLE (no covariates; one shared γ,
/// so no traps) with zero slopes, against the default race. Env `DS`,
/// `HARD=1`, `R`.
/// `cargo test --release --lib gamma_iid_start -- --ignored --nocapture`
#[test]
#[ignore]
fn gamma_iid_start() {
    use rayon::prelude::*;
    let env = |k: &str, v: usize| {
        std::env::var(k)
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(v)
    };
    let reps = env("R", 8) as u64;
    let hard = std::env::var("HARD").is_ok();
    let ds: Vec<usize> = std::env::var("DS")
        .unwrap_or_else(|_| "8,9,10".into())
        .split(',')
        .map(|v| v.parse().unwrap())
        .collect();
    for d in ds {
        let b = if hard {
            hard_truth(d)
        } else {
            standard_truth(d, 5)
        };
        let (bd, pt) = (b.nrows(), b.ncols());
        let rows: Vec<[(bool, f64, f64); 2]> = (920..920 + reps)
            .into_par_iter()
            .map(|seed| {
                let (x, y) = simulate(&b, 2000, seed, Shape::Gamma);
                let cfg = EsagRegressor::new().with_parallel(false);
                let data = cfg.data(&x, &y).unwrap();
                let tll = joint_loglik_and_grad(&data, &b, false).0;
                let t = Instant::now();
                let def = cfg.fit(&x, &y).unwrap();
                let t0 = t.elapsed().as_secs_f64();
                let t = Instant::now();
                let iid = cfg
                    .clone()
                    .with_hessian(false)
                    .fit(&DMatrix::zeros(x.nrows(), 0), &y)
                    .unwrap();
                let mut s = DMatrix::zeros(bd, pt);
                for r in 0..bd {
                    s[(r, 0)] = iid.coef[(r, 0)];
                }
                let f = cfg.fit_warm(&x, &y, &s).unwrap();
                let t1 = t.elapsed().as_secs_f64();
                [
                    (def.converged, def.log_likelihood - tll, t0),
                    (f.converged, f.log_likelihood - tll, t1),
                ]
            })
            .collect();
        for (k, name) in ["default race", "i.i.d. MLE start"].iter().enumerate() {
            let lls: Vec<String> = rows.iter().map(|r| format!("{:+.0}", r[k].1)).collect();
            let mut t: Vec<f64> = rows.iter().map(|r| r[k].2).collect();
            t.sort_by(f64::total_cmp);
            println!(
                "d={d:>2} {} {name:<17} converged {}/{reps}, median {:.2} s, ll - truth [{}] (regular root ≈ +{})",
                if hard { "hard    " } else { "standard" },
                rows.iter().filter(|r| r[k].0).count(),
                t[t.len() / 2],
                lls.join(" "),
                bd * pt / 2
            );
        }
    }
}

/// The log-tangent-informed start for γ: the log-tangent fit's μ
/// coefficients (the same mean model), the γ of its V̂(x = 0) as the shape
/// intercept, zero shape slopes; then the γ fit warm from there. Against
/// the default race. Env `DS`, `HARD=1`, `R`.
/// `cargo test --release --lib gamma_lt_start -- --ignored --nocapture`
#[test]
#[ignore]
fn gamma_lt_start() {
    use rayon::prelude::*;
    let env = |k: &str, v: usize| {
        std::env::var(k)
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(v)
    };
    let reps = env("R", 8) as u64;
    let hard = std::env::var("HARD").is_ok();
    let ds: Vec<usize> = std::env::var("DS")
        .unwrap_or_else(|_| "8,9,10".into())
        .split(',')
        .map(|v| v.parse().unwrap())
        .collect();
    for d in ds {
        let b = if hard {
            hard_truth(d)
        } else {
            standard_truth(d, 5)
        };
        let (bd, pt) = (b.nrows(), b.ncols());
        let (xt, _) = simulate(&b, 200, 99, Shape::Gamma);
        let eta_at = |c: &DMatrix<f64>, i: usize| -> Vec<f64> {
            (0..bd)
                .map(|k| c[(k, 0)] + (0..pt - 1).map(|j| c[(k, j + 1)] * xt[(i, j)]).sum::<f64>())
                .collect()
        };
        let vtrue: Vec<DMatrix<f64>> = (0..200)
            .map(|i| Esag::from_eta(&eta_at(&b, i)).unwrap().covariance())
            .collect();
        let verr = |f: &EsagRegressorFit| {
            let lt = f.lt_frame();
            (0..200)
                .map(|i| {
                    geodesic(
                        &vtrue[i],
                        &f.distribution_at(&eta_at(&f.coef, i), lt.as_ref())
                            .unwrap()
                            .covariance(),
                    )
                })
                .sum::<f64>()
                / 200.0
        };
        let rows: Vec<[(bool, f64, f64, f64); 3]> = (920..920 + reps)
            .into_par_iter()
            .map(|seed| {
                let (x, y) = simulate(&b, 2000, seed, Shape::Gamma);
                let cfg = EsagRegressor::new().with_parallel(false);
                let data = cfg.data(&x, &y).unwrap();
                let tll = joint_loglik_and_grad(&data, &b, false).0;
                let t = Instant::now();
                let def = cfg.fit(&x, &y).unwrap();
                let t0 = t.elapsed().as_secs_f64();
                let t = Instant::now();
                let lt = cfg
                    .clone()
                    .with_shape(Shape::LogTangent)
                    .with_hessian(false)
                    .fit(&x, &y)
                    .unwrap();
                let eta0: Vec<f64> = (0..bd).map(|k| lt.coef[(k, 0)]).collect();
                let v0 = lt
                    .distribution_at(&eta0, lt.lt_frame().as_ref())
                    .unwrap()
                    .covariance();
                let vr: Vec<f64> = (0..d)
                    .flat_map(|r| (0..d).map(move |c| (r, c)))
                    .map(|(r, c)| v0[(r, c)])
                    .collect();
                let g0 = crate::param::gamma_from_covariance(&eta0[..d], &vr);
                let mut s = DMatrix::zeros(bd, pt);
                for j in 0..pt {
                    for r in 0..d {
                        s[(r, j)] = lt.coef[(r, j)];
                    }
                }
                for (r, &v) in g0.iter().enumerate() {
                    s[(d + r, 0)] = v;
                }
                let f = cfg.fit_warm(&x, &y, &s).unwrap();
                let t1 = t.elapsed().as_secs_f64();
                [
                    (def.converged, def.log_likelihood - tll, t0, verr(&def)),
                    (f.converged, f.log_likelihood - tll, t1, verr(&f)),
                    (lt.converged, lt.log_likelihood - tll, 0.0, verr(&lt)),
                ]
            })
            .collect();
        for (k, name) in ["default race", "log-tangent start", "log-tangent fit"]
            .iter()
            .enumerate()
        {
            let lls: Vec<String> = rows.iter().map(|r| format!("{:+.0}", r[k].1)).collect();
            let mut t: Vec<f64> = rows.iter().map(|r| r[k].2).collect();
            let mut ve: Vec<f64> = rows.iter().map(|r| r[k].3).collect();
            t.sort_by(f64::total_cmp);
            ve.sort_by(f64::total_cmp);
            println!(
                "d={d:>2} {} {name:<18} converged {}/{reps}, median {:.2} s, V error {:.3}, ll - truth [{}] (regular root ≈ +{})",
                if hard { "hard    " } else { "standard" },
                rows.iter().filter(|r| r[k].0).count(),
                t[t.len() / 2],
                ve[ve.len() / 2],
                lls.join(" "),
                bd * pt / 2
            );
        }
    }
}

/// The log-tangent-informed γ start at d = 10 (standard): status and
/// gradient of the warm fit, and the trust-region Newton polish from it.
/// `cargo test --release --lib gamma_lt_start_status -- --ignored --nocapture`
#[test]
#[ignore]
fn gamma_lt_start_status() {
    use rayon::prelude::*;
    let d = 10;
    let b = standard_truth(d, 5);
    let (bd, pt) = (b.nrows(), b.ncols());
    let rows: Vec<String> = (920u64..928)
        .into_par_iter()
        .map(|seed| {
            let (x, y) = simulate(&b, 2000, seed, Shape::Gamma);
            let cfg = EsagRegressor::new().with_parallel(false);
            let data = cfg.data(&x, &y).unwrap();
            let tll = joint_loglik_and_grad(&data, &b, false).0;
            let lt = cfg.clone().with_shape(Shape::LogTangent).with_hessian(false).fit(&x, &y).unwrap();
            let eta0: Vec<f64> = (0..bd).map(|k| lt.coef[(k, 0)]).collect();
            let v0 = lt.distribution_at(&eta0, lt.lt_frame().as_ref()).unwrap().covariance();
            let vr: Vec<f64> = (0..d).flat_map(|r| (0..d).map(move |c| (r, c))).map(|(r, c)| v0[(r, c)]).collect();
            let g0 = crate::param::gamma_from_covariance(&eta0[..d], &vr);
            let mut s = DMatrix::zeros(bd, pt);
            for j in 0..pt {
                for r in 0..d {
                    s[(r, j)] = lt.coef[(r, j)];
                }
            }
            for (r, &v) in g0.iter().enumerate() {
                s[(d + r, 0)] = v;
            }
            let f = cfg.fit_warm(&x, &y, &s).unwrap();
            let t = Instant::now();
            let (c, reg, its, gm) = newton_tr(&data, &f.coef, 40);
            let tn = t.elapsed().as_secs_f64();
            let nll = joint_loglik_and_grad(&data, &c, false).0;
            format!(
                "seed {seed}: warm {:?} conv {} |g| {:.1e} singular {} ll-truth {:+.1} | Newton polish: regular {} after {its} its, |g| {gm:.1e}, ll-truth {:+.1} ({tn:.2} s)",
                f.status,
                f.converged,
                f.grad_norm,
                f.singular_obs.len(),
                f.log_likelihood - tll,
                reg as u8,
                nll - tll
            )
        })
        .collect();
    for r in rows {
        println!("{r}");
    }
}

/// Per-observation gradient and Hessian cost of the γ kernels at d = 7..10
/// (d ≤ 8 have fixed-size stack paths, d ≥ 9 heap buffers).
/// `cargo test --release --lib gamma_kernel_cost_big_d -- --ignored --nocapture`
#[test]
#[ignore]
fn gamma_kernel_cost_big_d() {
    for d in [7usize, 8, 9, 10] {
        let n = 4000;
        let k = dims::eta_len(d);
        let etas: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                (0..k)
                    .map(|j| ((i * 31 + j * 7) as f64 * 0.37).sin() + if j < d { 1.5 } else { 0.0 })
                    .collect()
            })
            .collect();
        let ys: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                let v: Vec<f64> = (0..d)
                    .map(|j| ((i * 13 + j * 5) as f64 * 0.91).cos() + 1.0)
                    .collect();
                let s = v.iter().map(|x| x * x).sum::<f64>().sqrt();
                v.iter().map(|x| x / s).collect()
            })
            .collect();
        let (mut g, mut h) = (vec![0.0; k], vec![0.0; k * k]);
        let time = |f: &mut dyn FnMut()| {
            let mut best = f64::INFINITY;
            for _ in 0..5 {
                let t = Instant::now();
                f();
                best = best.min(t.elapsed().as_secs_f64());
            }
            best / n as f64 * 1e9
        };
        let mut acc = 0.0;
        let tg = time(&mut || {
            for i in 0..n {
                acc += crate::adjoint::loglik_grad_adjoint(&etas[i], &ys[i], &mut g);
            }
        });
        let th = time(&mut || {
            for i in 0..n {
                acc += crate::adjoint_hess::loglik_hess_adjoint(&etas[i], &ys[i], &mut g, &mut h);
            }
        });
        println!("d={d:>2}: gamma gradient {tg:>6.0} ns/obs, Hessian {th:>7.0} ns/obs ({acc:.1e})");
    }
}

/// Bootstrap tests for γ at large d (standard truth, n = 2000): score and
/// full tests of the constant-shape null, one core, `B` replicates (env
/// `D`, `B`), with the time of the original fit.
/// `cargo test --release --lib gamma_big_d_tests -- --ignored --nocapture`
#[test]
#[ignore]
fn gamma_big_d_tests() {
    let env = |k: &str, v: usize| {
        std::env::var(k)
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(v)
    };
    let (d, nb) = (env("D", 8), env("B", 50));
    let b = standard_truth(d, 5);
    let (x, y) = simulate(&b, 2000, 930, Shape::Gamma);
    let t = Instant::now();
    let fit = EsagRegressor::new()
        .with_parallel(false)
        .fit(&x, &y)
        .unwrap();
    let tf = t.elapsed().as_secs_f64();
    println!(
        "d={d}: fit {} in {tf:.2} s",
        if fit.converged {
            "converged"
        } else {
            "not converged"
        }
    );
    for score in [true, false] {
        let t = Instant::now();
        let r = if score {
            crate::inference::score_test(
                &fit,
                &x,
                &y,
                crate::inference::Restriction::ConstantShape,
                nb,
                0,
                1,
            )
        } else {
            crate::inference::hypothesis_test(
                &fit,
                &x,
                &y,
                crate::inference::Restriction::ConstantShape,
                nb,
                0,
                1,
            )
        }
        .unwrap();
        let s = t.elapsed().as_secs_f64();
        println!(
            "  {} test, B = {nb}: {s:.1} s on {} threads ({:.2} s per replicate); p(LM) {:.3}, p(LR) {:.3}, failed refits {}",
            if score { "score" } else { "full " },
            rayon::current_num_threads(),
            s / nb as f64,
            r.p_lm,
            r.p_lr,
            r.n_failed
        );
    }
}

/// Trimmed γ fits: when the fit is trapped, set the trapped observations
/// aside and refit the rest warm, accumulating, until the fit converges
/// (at most 6 rounds). Converged share, number set aside, `V̂(x)` error
/// and time against the production fit. Env `D`, `HARD=1`, `R`.
/// `cargo test --release --lib gamma_trimmed_fits -- --ignored --nocapture`
#[test]
#[ignore]
fn gamma_trimmed_fits() {
    use rayon::prelude::*;
    let env = |k: &str, v: usize| {
        std::env::var(k)
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(v)
    };
    let (d, reps) = (env("D", 5), env("R", 12) as u64);
    let hard = std::env::var("HARD").is_ok();
    let b = if hard {
        hard_truth(d)
    } else {
        standard_truth(d, 5)
    };
    let (bd, pt) = (b.nrows(), b.ncols());
    let (xt, _) = simulate(&b, 200, 99, Shape::Gamma);
    let eta_at = |c: &DMatrix<f64>, i: usize| -> Vec<f64> {
        (0..bd)
            .map(|k| c[(k, 0)] + (0..pt - 1).map(|j| c[(k, j + 1)] * xt[(i, j)]).sum::<f64>())
            .collect()
    };
    let vtrue: Vec<DMatrix<f64>> = (0..200)
        .map(|i| Esag::from_eta(&eta_at(&b, i)).unwrap().covariance())
        .collect();
    let verr = |c: &DMatrix<f64>| {
        (0..200)
            .map(|i| {
                geodesic(
                    &vtrue[i],
                    &Esag::from_eta(&eta_at(c, i)).unwrap().covariance(),
                )
            })
            .sum::<f64>()
            / 200.0
    };
    let rows: Vec<(bool, f64, f64, bool, usize, f64, f64)> = (940..940 + reps)
        .into_par_iter()
        .map(|seed| {
            let (x, y) = simulate(&b, 2000, seed, Shape::Gamma);
            let cfg = EsagRegressor::new().with_parallel(false);
            let t = Instant::now();
            let f = cfg.fit(&x, &y).unwrap();
            let t0 = t.elapsed().as_secs_f64();
            let t = Instant::now();
            let mut cur = f.clone();
            let mut out: Vec<usize> = vec![];
            let mut keep: Vec<usize> = (0..x.nrows()).collect();
            for _ in 0..6 {
                if cur.converged || cur.singular_obs.is_empty() {
                    break;
                }
                // singular_obs index the current subset
                let drop: Vec<usize> = cur.singular_obs.iter().map(|&i| keep[i]).collect();
                out.extend(drop);
                out.sort_unstable();
                out.dedup();
                keep = (0..x.nrows())
                    .filter(|i| out.binary_search(i).is_err())
                    .collect();
                let xs = DMatrix::from_fn(keep.len(), x.ncols(), |a, j| x[(keep[a], j)]);
                let ys = DMatrix::from_fn(keep.len(), y.ncols(), |a, j| y[(keep[a], j)]);
                cur = cfg.fit_warm(&xs, &ys, &cur.coef).unwrap();
            }
            let t1 = t.elapsed().as_secs_f64() + t0;
            (
                f.converged,
                verr(&f.coef),
                t0,
                cur.converged,
                out.len(),
                verr(&cur.coef),
                t1,
            )
        })
        .collect();
    let med = |mut v: Vec<f64>| {
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    };
    println!(
        "d={d} {} n = 2000, {reps} datasets:\n  production: converged {}/{reps}, V error {:.3}, median {:.2} s\n  trimmed:    converged {}/{reps}, observations set aside {:?}, V error {:.3}, median {:.2} s",
        if hard { "hard" } else { "standard" },
        rows.iter().filter(|r| r.0).count(),
        med(rows.iter().map(|r| r.1).collect()),
        med(rows.iter().map(|r| r.2).collect()),
        rows.iter().filter(|r| r.3).count(),
        rows.iter().map(|r| r.4).collect::<Vec<_>>(),
        med(rows.iter().map(|r| r.5).collect()),
        med(rows.iter().map(|r| r.6).collect())
    );
}

/// Maximise `L_rest + w·L_S` (observations in `S` weighted by `w`) by
/// L-BFGS whitened with the exact information at the start (of the rest
/// alone if the sum is not definite). Returns (coef, converged, max|g| of
/// the weighted objective).
fn weighted_fit(
    rest: &RegressionData,
    sset: &RegressionData,
    w: f64,
    b0: &DMatrix<f64>,
) -> (DMatrix<f64>, bool, f64) {
    use crate::optim::lbfgs::{FnObjective, LbfgsConfig, minimize};
    let (bd, pt) = (rest.b_dim, rest.p_total);
    let k = bd * pt;
    let info = |b: &DMatrix<f64>, with_s: bool| -> DMatrix<f64> {
        let (_, _, h) = joint_loglik_grad_hess(rest, b, false);
        let mut m = -DMatrix::from_column_slice(k, k, &h);
        if with_s && w > 0.0 {
            let (_, _, hs) = joint_loglik_grad_hess(sset, b, false);
            m -= DMatrix::from_column_slice(k, k, &hs) * w;
        }
        (&m + m.transpose()) * 0.5
    };
    let chol = info(b0, true)
        .cholesky()
        .or_else(|| info(b0, false).cholesky());
    let Some(ch) = chol else {
        return (b0.clone(), false, f64::INFINITY);
    };
    let linv_t = ch.l().transpose().try_inverse().unwrap();
    let start: Vec<f64> = b0.as_slice().to_vec();
    let to_b = |th: &[f64]| -> DMatrix<f64> {
        let v = &linv_t * DVector::from_column_slice(th);
        DMatrix::from_column_slice(bd, pt, &(0..k).map(|a| start[a] + v[a]).collect::<Vec<_>>())
    };
    let eval = |b: &DMatrix<f64>| -> (f64, Vec<f64>) {
        let (f1, g1) = joint_loglik_and_grad(rest, b, false);
        let (mut f, mut g) = (f1, g1.as_slice().to_vec());
        if w > 0.0 {
            let (f2, g2) = joint_loglik_and_grad(sset, b, false);
            f += w * f2;
            for (a, v) in g.iter_mut().zip(g2.as_slice()) {
                *a += w * v;
            }
        }
        (f, g)
    };
    let obj = FnObjective(|th: &[f64], out: &mut [f64]| {
        let b = to_b(th);
        // g in the column-major order of B's entries, as the start and h
        let (f, g) = eval(&b);
        let gz = linv_t.transpose() * DVector::from_column_slice(&g);
        for (o, v) in out.iter_mut().zip(gz.iter()) {
            *o = -v;
        }
        -f
    });
    let cfg = LbfgsConfig {
        memory: k.clamp(10, 200),
        max_iter: 600,
        max_fev: 6000,
        tol_grad: 1e-7,
        ..LbfgsConfig::default()
    };
    let res = minimize(&obj, &vec![0.0; k], &cfg);
    let b = to_b(&res.x);
    let (_, g) = eval(&b);
    let gmax = g.iter().fold(0.0f64, |a, v| a.max(v.abs()));
    (b, res.status.is_converged(), gmax)
}

/// Weight homotopy to γ's regular root at large d: set the trapped
/// observations S aside (accumulating until the rest converges), then
/// raise their weight from 0 to 1 in small steps, each fit warm from the
/// last. Env `D` (default 10), `R`, `HARD=1`.
/// `cargo test --release --lib gamma_weight_homotopy -- --ignored --nocapture`
#[test]
#[ignore]
fn gamma_weight_homotopy() {
    use rayon::prelude::*;
    let env = |k: &str, v: usize| {
        std::env::var(k)
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(v)
    };
    let (d, reps) = (env("D", 10), env("R", 8) as u64);
    let hard = std::env::var("HARD").is_ok();
    let b = if hard {
        hard_truth(d)
    } else {
        standard_truth(d, 5)
    };
    let (bd, pt) = (b.nrows(), b.ncols());
    let rows: Vec<String> = (940..940 + reps)
        .into_par_iter()
        .map(|seed| {
            let (x, y) = simulate(&b, 2000, seed, Shape::Gamma);
            let n = x.nrows();
            let cfg = EsagRegressor::new().with_parallel(false);
            let data = cfg.data(&x, &y).unwrap();
            let tll = joint_loglik_and_grad(&data, &b, false).0;
            let t = Instant::now();
            let prod = cfg.fit(&x, &y).unwrap();
            if prod.converged {
                return format!("seed {seed}: production converged, ll-truth {:+.1}", prod.log_likelihood - tll);
            }
            let sub = |idx: &[usize]| -> RegressionData {
                let xs = DMatrix::from_fn(idx.len(), x.ncols(), |a, j| x[(idx[a], j)]);
                let ys = DMatrix::from_fn(idx.len(), y.ncols(), |a, j| y[(idx[a], j)]);
                cfg.data(&xs, &ys).unwrap()
            };
            // set aside until the rest converges
            let mut sset: Vec<usize> = singular_observations(&data, &prod.coef);
            let mut cur = prod.coef.clone();
            let mut rest_ok = false;
            for _ in 0..8 {
                sset.sort_unstable();
                sset.dedup();
                let keep: Vec<usize> = (0..n).filter(|i| sset.binary_search(i).is_err()).collect();
                let rd = sub(&keep);
                let (c, conv, _) = weighted_fit(&rd, &rd, 0.0, &cur);
                cur = c;
                let trapped = singular_observations(&rd, &cur);
                if conv && trapped.is_empty() {
                    rest_ok = true;
                    break;
                }
                sset.extend(trapped.iter().map(|&i| keep[i]));
            }
            if !rest_ok {
                return format!("seed {seed}: rest did not converge after setting aside {}", sset.len());
            }
            let keep: Vec<usize> = (0..n).filter(|i| sset.binary_search(i).is_err()).collect();
            let (rd, sd) = (sub(&keep), sub(&sset));
            // homotopy in the weight of S
            let mut w: f64 = 0.0;
            let mut step: f64 = 0.05;
            let mut fails = 0;
            while w < 1.0 && fails < 6 {
                let wn = (w + step).min(1.0);
                let (c, conv, _) = weighted_fit(&rd, &sd, wn, &cur);
                let trapped = !singular_observations(&sd, &c).is_empty() || !singular_observations(&rd, &c).is_empty();
                if conv && !trapped {
                    cur = c;
                    w = wn;
                    step = (step * 1.5).min(0.3);
                } else {
                    step *= 0.4;
                    fails += 1;
                }
            }
            let (fl, g) = joint_loglik_and_grad(&data, &cur, false);
            let gmax = g.iter().fold(0.0f64, |a, v| a.max(v.abs()));
            let trapped = !singular_observations(&data, &cur).is_empty();
            format!(
                "seed {seed}: production stuck (ll-truth {:+.1}); set aside {}; reached w = {w:.2}; full ll-truth {:+.1}, max|g| {gmax:.1e}, trapped {} ({:.1} s)",
                prod.log_likelihood - tll,
                sset.len(),
                fl - tll,
                trapped as u8,
                t.elapsed().as_secs_f64()
            )
        })
        .collect();
    println!(
        "d={d} {} (regular root ≈ +{} above the truth):",
        if hard { "hard" } else { "standard" },
        bd * pt / 2
    );
    for r in rows {
        println!("  {r}");
    }
}

/// At γ fits that end with huge gradients but no flagged observation (d =
/// 10, standard truth): for the observations with the largest gradients,
/// the gradient over the median, the smallest trailing-pair ratio, the
/// smallest group norm (j ≥ 2) relative to the largest, and the onb pole
/// distance.
/// `cargo test --release --lib gamma_hidden_traps -- --ignored --nocapture`
#[test]
#[ignore]
fn gamma_hidden_traps() {
    let d: usize = std::env::var("D")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);
    let b = standard_truth(d, 5);
    for seed in [900u64, 901, 902, 903] {
        let (x, y) = simulate(&b, 2000, seed, Shape::Gamma);
        let cfg = EsagRegressor::new().with_parallel(false);
        let data = cfg.data(&x, &y).unwrap();
        let f = cfg.fit(&x, &y).unwrap();
        let (bd, pt) = (data.b_dim, data.p_total);
        let mut eta = vec![0.0; bd];
        let mut g = vec![0.0; bd];
        let mut rows: Vec<(f64, f64, f64, f64, usize)> = (0..data.n)
            .map(|i| {
                for (k, e) in eta.iter_mut().enumerate() {
                    *e = (0..pt).map(|j| f.coef[(k, j)] * data.x[i * pt + j]).sum();
                }
                kernel_grad(&data, &eta, &data.y[i * d..(i + 1) * d], &mut g);
                let gn = g.iter().map(|v| v * v).sum::<f64>().sqrt();
                let pr = min_pair_ratio(&eta[d..], d).0;
                let norms: Vec<f64> = (2..=d - 2)
                    .map(|gr| {
                        let off = d + gr * (gr + 1) / 2 - 1;
                        eta[off..off + gr + 1]
                            .iter()
                            .map(|v| v * v)
                            .sum::<f64>()
                            .sqrt()
                    })
                    .collect();
                let gmax = norms.iter().fold(0.0f64, |a, &v| a.max(v));
                let (gmin, gi) =
                    norms
                        .iter()
                        .enumerate()
                        .fold(
                            (f64::INFINITY, 0),
                            |(m, mi), (k, &v)| if v < m { (v, k + 2) } else { (m, mi) },
                        );
                let mm = eta[..d].iter().map(|v| v * v).sum::<f64>();
                let pole = ((eta[0] * eta[0] + eta[1] * eta[1]) / mm).sqrt();
                (gn, pr, gmin / gmax, pole, gi)
            })
            .collect();
        let mut gs: Vec<f64> = rows.iter().map(|r| r.0).collect();
        gs.sort_by(f64::total_cmp);
        let med = gs[gs.len() / 2];
        rows.sort_by(|a, b| b.0.total_cmp(&a.0));
        println!(
            "seed {seed}: converged {}, flagged {}",
            f.converged,
            f.singular_obs.len()
        );
        for r in rows.iter().take(4) {
            println!(
                "    |g| {:.1e}x median, min pair ratio {:.1e}, smallest group (j={}) / largest {:.1e}, onb pole {:.2}",
                r.0 / med,
                r.1,
                r.4,
                r.2,
                r.3
            );
        }
    }
}

/// Group norms of the standard and hard truths' shape intercepts (the
/// eigenvalue ratios K_j − 1) at d = 8..10, and their range over 2000
/// covariate draws.
/// `cargo test --release --lib truth_group_norms -- --ignored --nocapture`
#[test]
#[ignore]
fn truth_group_norms() {
    for d in [8usize, 9, 10] {
        let b = standard_truth(d, 5);
        let (x, _) = simulate(&b, 2000, 3, Shape::Gamma);
        let norms_at = |eta: &[f64]| -> Vec<f64> {
            (1..=d - 2)
                .map(|gr| {
                    let off = d + gr * (gr + 1) / 2 - 1;
                    eta[off..off + gr + 1]
                        .iter()
                        .map(|v| v * v)
                        .sum::<f64>()
                        .sqrt()
                })
                .collect()
        };
        let e0: Vec<f64> = (0..b.nrows()).map(|k| b[(k, 0)]).collect();
        let n0 = norms_at(&e0);
        let mut mins = vec![f64::INFINITY; d - 2];
        for i in 0..2000 {
            let e: Vec<f64> = (0..b.nrows())
                .map(|k| b[(k, 0)] + (0..5).map(|j| b[(k, j + 1)] * x[(i, j)]).sum::<f64>())
                .collect();
            for (m, v) in mins.iter_mut().zip(norms_at(&e)) {
                *m = m.min(v);
            }
        }
        println!(
            "d={d} standard: group norms at x = 0 {:?}; smallest over the data {:?}",
            n0.iter().map(|v| format!("{v:.2}")).collect::<Vec<_>>(),
            mins.iter().map(|v| format!("{v:.2}")).collect::<Vec<_>>()
        );
    }
}

/// The trapped γ fits at large d, one dataset at a time (standard truth):
/// the default fit; how near the truth itself puts observations to the
/// singular set (smallest trailing-pair ratio over groups `j ≥ 2`, and how
/// many observations are under 1e−2 and 3e−2); for a trapped fit, the
/// trapped observations' ratios at the truth; and whether trust-region
/// Newton from the truth reaches a regular root, with its log-likelihood
/// against the trap's. Env `D` (default 10), `R`, `S0` (first seed, 900).
/// `cargo test --release --lib gamma_trap_anatomy -- --ignored --nocapture`
#[test]
#[ignore]
fn gamma_trap_anatomy() {
    use rayon::prelude::*;
    let env = |k: &str, v: usize| {
        std::env::var(k)
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(v)
    };
    let (d, reps, s0) = (env("D", 10), env("R", 12) as u64, env("S0", 900) as u64);
    let b = standard_truth(d, 5);
    let rows: Vec<String> = (s0..s0 + reps)
        .into_par_iter()
        .map(|seed| {
            let (x, y) = simulate(&b, 2000, seed, Shape::Gamma);
            let cfg = EsagRegressor::new().with_parallel(false);
            let data = cfg.data(&x, &y).unwrap();
            let truth_r = pair_ratios(&data, &b);
            let near = |c: f64| truth_r.iter().filter(|&&r| r < c).count();
            let rmin = truth_r.iter().fold(1.0f64, |a, &v| a.min(v));
            let t = Instant::now();
            let f = cfg.fit(&x, &y).unwrap();
            let secs = t.elapsed().as_secs_f64();
            let se_med = f.std_errors().map(|m| {
                let mut v: Vec<f64> = m.iter().copied().filter(|v| v.is_finite()).collect();
                v.sort_by(f64::total_cmp);
                v[v.len() / 2]
            });
            let mut line = format!(
                "seed {seed}: {} ({secs:.1} s), ll {:.1}, |g| {:.1e}, median SE {:.4}; truth: min pair ratio {rmin:.1e}, {} obs < 1e-2, {} < 3e-2",
                match (f.converged, f.non_regular) {
                    (true, false) => "regular    ".to_string(),
                    (true, true) => format!("non-regular (held {:?})", f.singular_obs),
                    _ => "trapped    ".to_string(),
                },
                f.log_likelihood,
                f.grad_norm,
                se_med.unwrap_or(f64::NAN),
                near(1e-2),
                near(3e-2)
            );
            if !f.converged && std::env::var("TRUTH").is_ok() {
                let fit_r = pair_ratios(&data, &f.coef);
                let at: Vec<String> = f
                    .singular_obs
                    .iter()
                    .map(|&i| format!("#{i} truth {:.1e} fit {:.1e}", truth_r[i], fit_r[i]))
                    .collect();
                let (c, reg, its, _) = newton_tr(&data, &b, 80);
                let nll = joint_loglik_and_grad(&data, &c, false).0;
                let warm = cfg.fit_warm(&x, &y, &b).unwrap();
                line += &format!(
                    "\n    trapped {}: [{}]\n    Newton from truth: regular {} after {its} its, ll {:.1} ({:+.1} vs trap); warm fit from truth: {} ll {:.1}",
                    f.singular_obs.len(),
                    at.join(", "),
                    reg,
                    nll,
                    nll - f.log_likelihood,
                    if warm.converged { "regular" } else { "trapped" },
                    warm.log_likelihood
                );
            }
            line
        })
        .collect();
    println!("d={d} standard truth, n = 2000:");
    for r in rows {
        println!("  {r}");
    }
}

/// Saddle-free trust-region Newton with the singular-set guard
/// ([`newton_tr`]) as the γ optimizer at large d, from the moment start and
/// from the log-tangent-informed start, against the default fit (standard
/// truth). Env `D` (default 10), `R`, `S0`, `IT` (Newton iterations, 150).
/// `cargo test --release --lib gamma_newton_from_starts -- --ignored --nocapture`
#[test]
#[ignore]
fn gamma_newton_from_starts() {
    use rayon::prelude::*;
    let env = |k: &str, v: usize| {
        std::env::var(k)
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(v)
    };
    let (d, reps, s0, iters) = (
        env("D", 10),
        env("R", 12) as u64,
        env("S0", 900) as u64,
        env("IT", 150),
    );
    let b = if std::env::var("HARD").is_ok() {
        hard_truth(d)
    } else {
        standard_truth(d, 5)
    };
    let rows: Vec<String> = (s0..s0 + reps)
        .into_par_iter()
        .map(|seed| {
            let (x, y) = simulate(&b, 2000, seed, Shape::Gamma);
            let cfg = EsagRegressor::new().with_parallel(false);
            let data = cfg.data(&x, &y).unwrap();
            let mask = cfg.resolve_mask(&data).unwrap();
            let t = Instant::now();
            let def = cfg.fit(&x, &y).unwrap();
            let t0 = t.elapsed().as_secs_f64();
            let mut b0 = cfg.initial_coef(&data).unwrap();
            zero_fixed(&mut b0, &mask);
            let t = Instant::now();
            let (ca, ra, ia, _) = newton_tr(&data, &b0, iters);
            let t1 = t.elapsed().as_secs_f64();
            let la = joint_loglik_and_grad(&data, &ca, false).0;
            let t = Instant::now();
            let lt = cfg.lt_informed_start(&x, &y, &data, &mask).unwrap();
            let (cb, rb, ib, _) = newton_tr(&data, &lt, iters);
            let t2 = t.elapsed().as_secs_f64();
            let lb = joint_loglik_and_grad(&data, &cb, false).0;
            format!(
                "seed {seed}: default {} ll {:.1} ({t0:.1} s) | Newton from moments {} ll {la:.1} ({ia} its, {t1:.1} s) | Newton from LT {} ll {lb:.1} ({ib} its, {t2:.1} s)",
                if def.converged { "regular" } else { "trapped" },
                def.log_likelihood,
                if ra { "regular" } else { "stopped" },
                if rb { "regular" } else { "stopped" },
            )
        })
        .collect();
    println!("d={d}, n = 2000:");
    for r in rows {
        println!("  {r}");
    }
}

/// How a trapped γ fit at large d differs from the fit reached from the
/// truth, group by group (standard truth): per γ group, the norm of its
/// intercept and of its slopes in both fits and the cosine between their
/// intercepts (and between their full coefficient blocks); for μ, the angle
/// between the intercepts. Env `D` (default 10), `SEEDS` (907,908).
/// `cargo test --release --lib gamma_bad_basin -- --ignored --nocapture`
#[test]
#[ignore]
fn gamma_bad_basin() {
    let d: usize = std::env::var("D")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);
    let seeds: Vec<u64> = std::env::var("SEEDS")
        .unwrap_or_else(|_| "907,908".into())
        .split(',')
        .map(|v| v.parse().unwrap())
        .collect();
    let b = standard_truth(d, 5);
    let pt = b.ncols();
    for seed in seeds {
        let (x, y) = simulate(&b, 2000, seed, Shape::Gamma);
        let cfg = EsagRegressor::new().with_parallel(false);
        let data = cfg.data(&x, &y).unwrap();
        let def = cfg.fit(&x, &y).unwrap();
        let (good, reg, _, _) = newton_tr(&data, &b, 80);
        let gll = joint_loglik_and_grad(&data, &good, false).0;
        println!(
            "seed {seed}: default ll {:.1}, from truth ll {gll:.1} (regular {reg})",
            def.log_likelihood
        );
        let block = |c: &DMatrix<f64>,
                     rows: std::ops::Range<usize>,
                     cols: std::ops::Range<usize>|
         -> Vec<f64> {
            cols.flat_map(|j| rows.clone().map(move |k| (k, j)))
                .map(|(k, j)| c[(k, j)])
                .collect()
        };
        let nrm = |v: &[f64]| v.iter().map(|a| a * a).sum::<f64>().sqrt();
        let cos = |a: &[f64], b: &[f64]| {
            a.iter().zip(b).map(|(p, q)| p * q).sum::<f64>() / (nrm(a) * nrm(b))
        };
        let (m1, m2) = (block(&def.coef, 0..d, 0..1), block(&good, 0..d, 0..1));
        println!(
            "  mu intercept: cos {:.3}, norms {:.2} / {:.2}",
            cos(&m1, &m2),
            nrm(&m1),
            nrm(&m2)
        );
        // align the default fit to the good one modulo the exact symmetries
        // S_j (group j negated, and the last entry of every lower group)
        let mut al = def.coef.clone();
        for j in (1..=d - 2).rev() {
            let off = d + j * (j + 1) / 2 - 1;
            let rows = off..off + j + 1;
            if cos(
                &block(&al, rows.clone(), 0..pt),
                &block(&good, rows.clone(), 0..pt),
            ) < 0.0
            {
                for c in 0..pt {
                    for k in rows.clone() {
                        al[(k, c)] = -al[(k, c)];
                    }
                    for jl in 1..j {
                        let last = d + jl * (jl + 1) / 2 - 1 + jl;
                        al[(last, c)] = -al[(last, c)];
                    }
                }
            }
        }
        let all = joint_loglik_and_grad(&data, &al, false).0;
        println!(
            "  aligned default: ll {all:.1} (the symmetry keeps it: {:.1})",
            def.log_likelihood
        );
        for j in 1..=d - 2 {
            let off = d + j * (j + 1) / 2 - 1;
            let ent: Vec<String> = (off..off + j + 1)
                .map(|k| {
                    let (u, v) = (block(&al, k..k + 1, 0..pt), block(&good, k..k + 1, 0..pt));
                    format!("{:+.2}", cos(&u, &v))
                })
                .collect();
            let rows = off..off + j + 1;
            println!(
                "  aligned group {j}: block cos {:+.3}; per entry {}",
                cos(&block(&al, rows.clone(), 0..pt), &block(&good, rows, 0..pt)),
                ent.join(" ")
            );
        }
        for j in 1..=d - 2 {
            let off = d + j * (j + 1) / 2 - 1;
            let rows = off..off + j + 1;
            let (a0, b0) = (
                block(&def.coef, rows.clone(), 0..1),
                block(&good, rows.clone(), 0..1),
            );
            let (a1, b1) = (
                block(&def.coef, rows.clone(), 1..pt),
                block(&good, rows.clone(), 1..pt),
            );
            let (aa, bb) = (
                block(&def.coef, rows.clone(), 0..pt),
                block(&good, rows.clone(), 0..pt),
            );
            let t0 = block(&b, rows.clone(), 0..1);
            println!(
                "  group {j}: intercept norm {:.2} / {:.2} (truth {:.2}), cos {:+.3}; slopes norm {:.2} / {:.2}; whole block cos {:+.3}",
                nrm(&a0),
                nrm(&b0),
                nrm(&t0),
                cos(&a0, &b0),
                nrm(&a1),
                nrm(&b1),
                cos(&aa, &bb)
            );
        }
    }
}

/// The likelihood around a trap (standard truth, large d): for the first
/// trapped observation `i` of the default fit, its trailing pair (of the
/// group nearest the singular set) is placed at `r·(cos θ, sin θ)` by the
/// smallest change of the two pair rows of `B`, and the full
/// log-likelihood is printed against the trap's — around circles of radius
/// `r = 1e−2` and `1e−1` of the group norm, and along the ray at the best
/// angle. Env `D` (default 10), `SEEDS` (906,911).
/// `cargo test --release --lib gamma_trap_landscape -- --ignored --nocapture`
#[test]
#[ignore]
fn gamma_trap_landscape() {
    let d: usize = std::env::var("D")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);
    let seeds: Vec<u64> = std::env::var("SEEDS")
        .unwrap_or_else(|_| "906,911".into())
        .split(',')
        .map(|v| v.parse().unwrap())
        .collect();
    let b = standard_truth(d, 5);
    let pt = b.ncols();
    for seed in seeds {
        let (x, y) = simulate(&b, 2000, seed, Shape::Gamma);
        let cfg = EsagRegressor::new().with_parallel(false);
        let data = cfg.data(&x, &y).unwrap();
        let fit = cfg.fit(&x, &y).unwrap();
        let Some(&i) = fit.singular_obs.first() else {
            println!("seed {seed}: not trapped");
            continue;
        };
        let bt = fit.coef.clone();
        let xi: Vec<f64> = (0..pt).map(|c| data.x[i * pt + c]).collect();
        let x2: f64 = xi.iter().map(|v| v * v).sum();
        let eta: Vec<f64> = (0..data.b_dim)
            .map(|k| (0..pt).map(|c| bt[(k, c)] * xi[c]).sum())
            .collect();
        let (ratio, j) = min_pair_ratio(&eta[d..], d);
        let off = d + j * (j + 1) / 2 - 1;
        let (ka, kb) = (off + j - 1, off + j);
        let gn = eta[off..off + j + 1]
            .iter()
            .map(|v| v * v)
            .sum::<f64>()
            .sqrt();
        let p0 = (eta[ka], eta[kb]);
        let at = |r: f64, th: f64| -> f64 {
            let mut c = bt.clone();
            let (da, db) = (r * gn * th.cos() - p0.0, r * gn * th.sin() - p0.1);
            for col in 0..pt {
                c[(ka, col)] += da * xi[col] / x2;
                c[(kb, col)] += db * xi[col] / x2;
            }
            joint_loglik_and_grad(&data, &c, false).0
        };
        let l0 = joint_loglik_and_grad(&data, &bt, false).0;
        let th0 = p0.1.atan2(p0.0);
        println!(
            "seed {seed}: observation #{i}, group {j}, pair ratio {ratio:.1e} at angle {:.0}°, trap ll {l0:.2}",
            th0.to_degrees()
        );
        let mut best = (f64::NEG_INFINITY, 0.0);
        for r in [1e-2, 1e-1] {
            let row: Vec<String> = (0..24)
                .map(|s| {
                    let th = th0 + (s as f64) * std::f64::consts::PI / 12.0;
                    let l = at(r, th);
                    if r == 1e-2 && l > best.0 {
                        best = (l, th);
                    }
                    format!("{:+.1}", l - l0)
                })
                .collect();
            println!("  r = {r:.0e}, θ − θ_trap = 0°, 15°, …: {}", row.join(" "));
        }
        let ray: Vec<String> = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1]
            .iter()
            .map(|&r| format!("{r:.0e}: {:+.2}", at(r, best.1) - l0))
            .collect();
        println!(
            "  ray at the best angle ({:.0}° from the trap's): {}",
            (best.1 - th0).to_degrees(),
            ray.join(", ")
        );
        let ray0: Vec<String> = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1]
            .iter()
            .map(|&r| format!("{r:.0e}: {:+.2}", at(r, th0) - l0))
            .collect();
        println!("  ray at the trap's angle: {}", ray0.join(", "));
    }
}

/// Is a bad basin of γ at large d (standard truth) caused by the start's
/// shape intercept or by the path of the shape slopes? The default L-BFGS
/// pipeline (race) from the log-tangent-informed start with its shape
/// intercept replaced by the truth's (slopes zero), and by the truth's
/// with its shape slopes too, against the log-tangent-informed start.
/// Env `D` (default 10), `SEEDS` (906,907,908,911).
/// `cargo test --release --lib gamma_basin_cause -- --ignored --nocapture`
#[test]
#[ignore]
fn gamma_basin_cause() {
    use rayon::prelude::*;
    let d: usize = std::env::var("D")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);
    let seeds: Vec<u64> = std::env::var("SEEDS")
        .unwrap_or_else(|_| "906,907,908,911".into())
        .split(',')
        .map(|v| v.parse().unwrap())
        .collect();
    let b = standard_truth(d, 5);
    let pt = b.ncols();
    let rows: Vec<String> = seeds
        .par_iter()
        .map(|&seed| {
            let (x, y) = simulate(&b, 2000, seed, Shape::Gamma);
            let cfg = EsagRegressor::new().with_parallel(false);
            let data = cfg.data(&x, &y).unwrap();
            let mask = cfg.resolve_mask(&data).unwrap();
            let lt = cfg.lt_informed_start(&x, &y, &data, &mask).unwrap();
            let mut s1 = lt.clone();
            for k in d..data.b_dim {
                s1[(k, 0)] = b[(k, 0)];
            }
            let mut s2 = s1.clone();
            for k in d..data.b_dim {
                for c in 1..pt {
                    s2[(k, c)] = b[(k, c)];
                }
            }
            let run = |s: &DMatrix<f64>| {
                let r = cfg.race(&data, std::slice::from_ref(s), &mask, cfg.max_iter);
                format!(
                    "{} ll {:.1}",
                    if r.converged && singular_observations(&data, &r.coef).is_empty() {
                        "regular"
                    } else {
                        "trapped"
                    },
                    -r.neg_ll
                )
            };
            format!(
                "seed {seed}: LT start {} | with the truth's shape intercept {} | and its shape slopes {}",
                run(&lt),
                run(&s1),
                run(&s2)
            )
        })
        .collect();
    for r in rows {
        println!("{r}");
    }
}

/// Apply the exact symmetry `S_j` of the γ parameterization to a γ vector:
/// negate group `j` and the last entry of every lower group (the same
/// `V` for every `μ`).
fn gamma_symmetry(g: &mut [f64], j: usize) {
    let off = j * (j + 1) / 2 - 1;
    for v in &mut g[off..off + j + 1] {
        *v = -*v;
    }
    for jl in 1..j {
        g[jl * (jl + 1) / 2 - 1 + jl] *= -1.0;
    }
}

/// `g` moved by the symmetries `S_j` (top group first) so that each group
/// has a non-negative inner product with `reference`'s.
fn gamma_align(g: &mut [f64], reference: &[f64], d: usize) {
    for j in (1..=d - 2).rev() {
        let off = j * (j + 1) / 2 - 1;
        let ip: f64 = (off..off + j + 1).map(|k| g[k] * reference[k]).sum();
        if ip < 0.0 {
            gamma_symmetry(g, j);
        }
    }
}

/// Estimators of γ's shape intercept at large d (standard truth), against
/// the truth's: the log-tangent fit's `V̂(0)` (the current start), the
/// constant-shape γ fit, and the least-squares fit to the per-observation
/// γ of the log-tangent `V̂(x_i)` aligned by the symmetries (with its
/// slopes too). Per estimator, the cosine with the truth per group (after
/// alignment) and the race from it (μ of the log-tangent fit, shape slopes
/// zero, or the least-squares slopes). Env `D` (default 10), `R`, `S0`.
/// `cargo test --release --lib gamma_intercept_estimators -- --ignored --nocapture`
#[test]
#[ignore]
fn gamma_intercept_estimators() {
    use rayon::prelude::*;
    let env = |k: &str, v: usize| {
        std::env::var(k)
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(v)
    };
    let (d, reps, s0) = (env("D", 10), env("R", 12) as u64, env("S0", 900) as u64);
    let b = standard_truth(d, 5);
    let (bd, pt, g) = (b.nrows(), b.ncols(), dims::gamma_len(d));
    let truth0: Vec<f64> = (0..g).map(|k| b[(d + k, 0)]).collect();
    let rows: Vec<String> = (s0..s0 + reps)
        .into_par_iter()
        .map(|seed| {
            let (x, y) = simulate(&b, 2000, seed, Shape::Gamma);
            let cfg = EsagRegressor::new().with_parallel(false);
            let data = cfg.data(&x, &y).unwrap();
            let mask = cfg.resolve_mask(&data).unwrap();
            let lt_start = cfg.lt_informed_start(&x, &y, &data, &mask).unwrap();
            // the log-tangent fit itself, for V̂(x_i)
            let mut lcfg = cfg.clone();
            lcfg.shape = Shape::LogTangent;
            lcfg.compute_hessian = false;
            let ltf = lcfg.fit(&x, &y).unwrap();
            let fr = ltf.lt_frame();
            let gam_at = |i: usize| -> Vec<f64> {
                let eta: Vec<f64> = (0..bd)
                    .map(|k| (0..pt).map(|c| ltf.coef[(k, c)] * data.x[i * pt + c]).sum())
                    .collect();
                let v = ltf.distribution_at(&eta, fr.as_ref()).unwrap().covariance();
                let vr: Vec<f64> = (0..d * d).map(|a| v[(a / d, a % d)]).collect();
                crate::param::gamma_from_covariance(&eta[..d], &vr)
            };
            let ref0: Vec<f64> = (0..g).map(|k| lt_start[(d + k, 0)]).collect();
            // least squares of the aligned per-observation γ on x̃
            let mut gs = DMatrix::zeros(data.n, g);
            for i in 0..data.n {
                let mut gi = gam_at(i);
                gamma_align(&mut gi, &ref0, d);
                for k in 0..g {
                    gs[(i, k)] = gi[k];
                }
            }
            let xm = DMatrix::from_fn(data.n, pt, |i, c| data.x[i * pt + c]);
            let coef_ls = (xm.transpose() * &xm)
                .cholesky()
                .unwrap()
                .solve(&(xm.transpose() * &gs));
            let ls0: Vec<f64> = (0..g).map(|k| coef_ls[(0, k)]).collect();
            // constant-shape γ fit from the log-tangent start
            let cmask = DMatrix::from_fn(bd, pt, |k, c| mask[(k, c)] && (k < d || c == 0));
            let cst = cfg.race(&data, std::slice::from_ref(&lt_start), &cmask, cfg.max_iter);
            let cs0: Vec<f64> = (0..g).map(|k| cst.coef[(d + k, 0)]).collect();
            let report = |name: &str, est: &[f64], start: &DMatrix<f64>| -> String {
                let mut e = est.to_vec();
                gamma_align(&mut e, &truth0, d);
                let cs: Vec<String> = (1..=d - 2)
                    .map(|j| {
                        let off = j * (j + 1) / 2 - 1;
                        let (u, v) = (&e[off..off + j + 1], &truth0[off..off + j + 1]);
                        let ip: f64 = u.iter().zip(v).map(|(p, q)| p * q).sum();
                        let nu = u.iter().map(|a| a * a).sum::<f64>().sqrt();
                        let nv = v.iter().map(|a| a * a).sum::<f64>().sqrt();
                        format!("{:.2}", ip / (nu * nv))
                    })
                    .collect();
                let r = cfg.race(&data, std::slice::from_ref(start), &mask, cfg.max_iter);
                let reg = r.converged && singular_observations(&data, &r.coef).is_empty();
                format!(
                    "{name}: cos [{}] → {} ll {:.1}",
                    cs.join(" "),
                    if reg { "regular" } else { "trapped" },
                    -r.neg_ll
                )
            };
            let with0 = |g0: &[f64]| {
                let mut s = lt_start.clone();
                for k in 0..g {
                    s[(d + k, 0)] = g0[k];
                }
                s
            };
            let mut ls_full = lt_start.clone();
            for k in 0..g {
                for c in 0..pt {
                    ls_full[(d + k, c)] = coef_ls[(c, k)];
                }
            }
            format!(
                "seed {seed}:\n    {}\n    {}\n    {}\n    {}",
                report("LT V(0)     ", &ref0, &lt_start),
                report("const shape ", &cs0, &with0(&cs0)),
                report("LS intercept", &ls0, &with0(&ls0)),
                report("LS + slopes ", &ls0, &ls_full)
            )
        })
        .collect();
    for r in rows {
        println!("{r}");
    }
}

/// `n` draws from the γ-rot regression (`src/gamma_rot.rs`) with
/// coefficients `b` read in frame `fr` (equicorrelated covariates, ρ = 0.5).
fn simulate_rot(
    b: &DMatrix<f64>,
    n: usize,
    seed: u64,
    fr: &crate::gamma_rot::GrFrame,
) -> (DMatrix<f64>, DMatrix<f64>) {
    let (x, _) = simulate(b, n, seed, Shape::Gamma);
    let d = fr.dim();
    let mut rng = ChaCha8Rng::seed_from_u64(seed ^ 0x9e37);
    let mut y = DMatrix::zeros(n, d);
    let mut buf = vec![0.0; d];
    for i in 0..n {
        let eta: Vec<f64> = (0..b.nrows())
            .map(|k| {
                b[(k, 0)]
                    + (0..b.ncols() - 1)
                        .map(|j| b[(k, j + 1)] * x[(i, j)])
                        .sum::<f64>()
            })
            .collect();
        let v = fr.covariance(&eta[..d], &eta[d..]);
        let dist = Esag::from_covariance(&eta[..d], &v).unwrap();
        dist.sample_one(&dist.frame(), &mut rng, &mut buf);
        for k in 0..d {
            y[(i, k)] = buf[k];
        }
    }
    (x, y)
}

/// A plain γ-rot regression fit: L-BFGS on the joint negative
/// log-likelihood (forward-mode gradients) from `start`, or from the moment
/// intercept written in frame `fr` with zero slopes. Returns (coef,
/// converged, iterations, evaluations, max|g|).
fn fit_rot(
    fr: &crate::gamma_rot::GrFrame,
    x: &DMatrix<f64>,
    y: &DMatrix<f64>,
    start: Option<&DMatrix<f64>>,
) -> (DMatrix<f64>, bool, usize, usize, f64) {
    use crate::gamma_rot::loglik_gr_grad;
    use crate::optim::lbfgs::{FnObjective, LbfgsConfig, minimize};
    let (n, d, p) = (y.nrows(), y.ncols(), x.ncols());
    let (bd, pt) = (dims::eta_len(d), p + 1);
    let yv: Vec<f64> = (0..n)
        .flat_map(|i| (0..d).map(move |k| (i, k)))
        .map(|(i, k)| y[(i, k)])
        .collect();
    let x0: Vec<f64> = match start {
        Some(s) => s.as_slice().to_vec(),
        None => {
            let (mu0, g0) = moment_start(&yv, n, d);
            let v0 = Esag::new(&mu0, &g0).unwrap().covariance();
            let gs0 = fr.coords_from_covariance(&mu0, &v0);
            let mut s = vec![0.0; bd * pt];
            s[..d].copy_from_slice(&mu0);
            s[d..d + gs0.len()].copy_from_slice(&gs0);
            s
        }
    };
    let nk = bd * pt;
    let obj = FnObjective(|th: &[f64], grad_out: &mut [f64]| {
        grad_out.iter_mut().for_each(|g| *g = 0.0);
        let (mut eta, mut g) = (vec![0.0; bd], vec![0.0; bd]);
        let mut f = 0.0;
        for i in 0..n {
            let xr = |j: usize| if j == 0 { 1.0 } else { x[(i, j - 1)] };
            for (k, e) in eta.iter_mut().enumerate() {
                *e = (0..pt).map(|j| th[j * bd + k] * xr(j)).sum();
            }
            f -= loglik_gr_grad(fr, &eta, &yv[i * d..(i + 1) * d], &mut g);
            for j in 0..pt {
                let w = xr(j);
                for k in 0..bd {
                    grad_out[j * bd + k] -= g[k] * w;
                }
            }
        }
        f
    });
    let cfg = LbfgsConfig {
        memory: nk.clamp(10, 200),
        max_iter: 2000,
        max_fev: 40_000,
        tol_grad: 1e-6,
        ..LbfgsConfig::default()
    };
    let res = minimize(&obj, &x0, &cfg);
    if std::env::var("TRACE").is_ok() {
        println!(
            "    status {:?}  iters {}  fev {}  -ll {:.3}",
            res.status, res.n_iter, res.n_fev, res.f
        );
    }
    (
        DMatrix::from_column_slice(bd, pt, &res.x),
        res.status.is_converged(),
        res.n_iter,
        res.n_fev,
        res.grad_norm_inf,
    )
}

/// Does the γ-rot chart (`src/gamma_rot.rs`) remove γ's traps, and at what
/// cost? Env: `D`, `N`, `R` (datasets), `SEED0` (first seed, default 1200),
/// `STD=1` for the standard truth (hard otherwise), `DATA=rot` for data from
/// the γ-rot model with the same coefficient matrix read in its chart
/// (default: data from γ), `SCALE` multiplying the shape slopes of that
/// truth, `LINK=exp` for the exponential link (default stereographic).
/// γ-rot fits (forward-mode L-BFGS from the moment start) against the
/// production γ and log-tangent fits on the same data: convergence,
/// iterations, time and the `V̂(x)` error on 200 fresh points.
/// `cargo test --release --lib gamma_rot_fits -- --ignored --nocapture`
#[test]
#[ignore]
fn gamma_rot_fits() {
    use crate::gamma_rot::{GrFrame, RotLink};
    use rayon::prelude::*;
    let env = |k: &str, v: usize| {
        std::env::var(k)
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(v)
    };
    let (d, n, reps) = (env("D", 5), env("N", 2000), env("R", 8) as u64);
    let seed0 = env("SEED0", 1200) as u64;
    let rot_data = std::env::var("DATA").as_deref() == Ok("rot");
    let link = if std::env::var("LINK").as_deref() == Ok("exp") {
        RotLink::Exp
    } else {
        RotLink::Stereo
    };
    let scale: f64 = std::env::var("SCALE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1.0);
    let mut b = if std::env::var("STD").is_ok() {
        standard_truth(d, 5)
    } else {
        hard_truth(d)
    };
    if rot_data && scale != 1.0 {
        for k in d..b.nrows() {
            for j in 1..b.ncols() {
                b[(k, j)] *= scale;
            }
        }
    }
    let (bd, pt) = (b.nrows(), b.ncols());
    let tf = GrFrame::new(&center(d), link);
    let lt = LtFrame::new(&center(d));
    let (xt, _) = simulate(&b, 200, 99, Shape::Gamma);
    let eta = |c: &DMatrix<f64>, i: usize| -> Vec<f64> {
        (0..bd)
            .map(|k| c[(k, 0)] + (0..pt - 1).map(|j| c[(k, j + 1)] * xt[(i, j)]).sum::<f64>())
            .collect()
    };
    let vtrue: Vec<DMatrix<f64>> = (0..200)
        .map(|i| {
            let e = eta(&b, i);
            if rot_data {
                tf.covariance(&e[..d], &e[d..])
            } else {
                Esag::from_eta(&e).unwrap().covariance()
            }
        })
        .collect();
    if rot_data {
        // how far the truth's axes turn over the data, per group
        let (xs, _) = simulate(&b, 4000, 7, Shape::Gamma);
        for gr in 2..=d - 2 {
            let off = d + gr * (gr + 1) / 2 - 1;
            let mut ang: Vec<f64> = (0..4000)
                .map(|i| {
                    let e: Vec<f64> = (0..bd)
                        .map(|k| {
                            b[(k, 0)] + (0..pt - 1).map(|j| b[(k, j + 1)] * xs[(i, j)]).sum::<f64>()
                        })
                        .collect();
                    let a = tf.axis(&e[d..], gr);
                    a[gr].clamp(-1.0, 1.0).acos().to_degrees()
                })
                .collect();
            ang.sort_by(f64::total_cmp);
            let kap: Vec<f64> = (0..4000)
                .map(|i| {
                    b[(off, 0)]
                        + (0..pt - 1)
                            .map(|j| b[(off, j + 1)] * xs[(i, j)])
                            .sum::<f64>()
                })
                .collect();
            let neg = kap.iter().filter(|k| **k < 0.0).count();
            println!(
                "  truth group {gr}: axis angle from the pole 5%/50%/95% = {:.0}/{:.0}/{:.0} deg, κ<0 (crossed) in {:.1}% of x",
                ang[200],
                ang[2000],
                ang[3800],
                100.0 * neg as f64 / 4000.0
            );
        }
    }
    // START=truth (γ-rot data only): an oracle start at the true coefficients,
    // to tell local maxima from estimation variance.
    let oracle = rot_data && std::env::var("START").as_deref() == Ok("truth");
    let joint_ll = |fr: &GrFrame, c: &DMatrix<f64>, x: &DMatrix<f64>, y: &DMatrix<f64>| -> f64 {
        (0..x.nrows())
            .map(|i| {
                let e: Vec<f64> = (0..bd)
                    .map(|k| {
                        c[(k, 0)] + (0..pt - 1).map(|j| c[(k, j + 1)] * x[(i, j)]).sum::<f64>()
                    })
                    .collect();
                let yi: Vec<f64> = (0..d).map(|k| y[(i, k)]).collect();
                crate::gamma_rot::loglik_gr::<f64>(fr, &e, &yi)
            })
            .sum()
    };
    type Row = (bool, usize, f64, f64, f64, bool, f64, f64, bool, f64, f64);
    let rows: Vec<(Row, [f64; 5])> = (seed0..seed0 + reps)
        .into_par_iter()
        .map(|seed| {
            let (x, y) = if rot_data {
                simulate_rot(&b, n, seed, &tf)
            } else {
                simulate(&b, n, seed, Shape::Gamma)
            };
            let fr = GrFrame::new(&center(d), link);
            // START=lt: the log-tangent fit's μ rows and its V̂ at the mean
            // design row as the shape intercept, zero shape slopes
            let lt_start: Option<DMatrix<f64>> = (std::env::var("START").as_deref() == Ok("lt")).then(|| {
                let l = config(d, Shape::LogTangent).fit(&x, &y).unwrap();
                let xbar: Vec<f64> = (0..pt - 1).map(|j| x.column(j).sum() / n as f64).collect();
                let e: Vec<f64> = (0..bd)
                    .map(|k| l.coef[(k, 0)] + (0..pt - 1).map(|j| l.coef[(k, j + 1)] * xbar[j]).sum::<f64>())
                    .collect();
                let vbar = crate::logtangent::covariance(&lt, &e[..d], &e[d..]);
                let g0 = fr.coords_from_covariance(&e[..d], &vbar);
                let mut s = DMatrix::zeros(bd, pt);
                for k in 0..d {
                    for j in 0..pt {
                        s[(k, j)] = l.coef[(k, j)];
                    }
                }
                for (k, g) in g0.iter().enumerate() {
                    s[(d + k, 0)] = *g;
                }
                s
            });
            let t = Instant::now();
            let (c, conv, it, _fev, gmax) = fit_rot(
                &fr,
                &x,
                &y,
                if oracle { Some(&b) } else { lt_start.as_ref() },
            );
            let ms = t.elapsed().as_secs_f64() * 1e3;
            // log-likelihood gain of the fit over the truth (γ-rot data only)
            let gain = if rot_data {
                joint_ll(&fr, &c, &x, &y) - joint_ll(&fr, &b, &x, &y)
            } else {
                f64::NAN
            };
            let verr = (0..200)
                .map(|i| {
                    let e = eta(&c, i);
                    geodesic(&vtrue[i], &fr.covariance(&e[..d], &e[d..]))
                })
                .sum::<f64>()
                / 200.0;
            // degeneracy diagnostics: largest per-observation log-density
            // excess over the median, and largest |log λ| over the data
            let diag = |c: &DMatrix<f64>| -> (f64, f64) {
                let mut lls = Vec::with_capacity(n);
                let mut maxlog = 0.0f64;
                for i in 0..n {
                    let e: Vec<f64> = (0..bd)
                        .map(|k| {
                            c[(k, 0)] + (0..pt - 1).map(|j| c[(k, j + 1)] * x[(i, j)]).sum::<f64>()
                        })
                        .collect();
                    let yi: Vec<f64> = (0..d).map(|k| y[(i, k)]).collect();
                    lls.push(crate::gamma_rot::loglik_gr::<f64>(&fr, &e, &yi));
                    for l in fr.eigenvalues(&e[d..]) {
                        maxlog = maxlog.max(l.ln().abs());
                    }
                }
                lls.sort_by(f64::total_cmp);
                (lls[n - 1] - lls[n / 2], maxlog)
            };
            let (exc_fit, ml_fit) = diag(&c);
            let (exc_tru, ml_tru) = if rot_data { diag(&b) } else { (f64::NAN, f64::NAN) };
            let (gconv, gms, gerr) = if std::env::var("SKIP_GAMMA").is_ok() {
                (false, f64::NAN, f64::NAN)
            } else {
                let t = Instant::now();
                let g = config(d, Shape::Gamma).fit(&x, &y).unwrap();
                let gms = t.elapsed().as_secs_f64() * 1e3;
                let gerr = (0..200)
                    .map(|i| {
                        geodesic(
                            &vtrue[i],
                            &Esag::from_eta(&eta(&g.coef, i)).unwrap().covariance(),
                        )
                    })
                    .sum::<f64>()
                    / 200.0;
                (g.converged, gms, gerr)
            };
            let t = Instant::now();
            let l = config(d, Shape::LogTangent).fit(&x, &y).unwrap();
            let lms = t.elapsed().as_secs_f64() * 1e3;
            let lerr = (0..200)
                .map(|i| {
                    let e = eta(&l.coef, i);
                    geodesic(&vtrue[i], &crate::logtangent::covariance(&lt, &e[..d], &e[d..]))
                })
                .sum::<f64>()
                / 200.0;
            if std::env::var("TRACE").is_ok() {
                println!(
                    "  seed {seed}: rot {} it {it} |g| {gmax:.1e} V {verr:.3} ll gain {gain:.1} excess {exc_fit:.1} (truth {exc_tru:.1}) max|log λ| {ml_fit:.2} (truth {ml_tru:.2}) | gamma {} V {gerr:.3} | lt {} V {lerr:.3}",
                    conv, gconv, l.converged
                );
            }
            (
                (conv, it, ms, verr, gmax, gconv, gms, gerr, l.converged, lms, lerr),
                [gain, exc_fit, ml_fit, exc_tru, ml_tru],
            )
        })
        .collect();
    let extra: Vec<[f64; 5]> = rows.iter().map(|r| r.1).collect();
    let rows: Vec<Row> = rows.into_iter().map(|r| r.0).collect();
    let med = |mut v: Vec<f64>| {
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    };
    if rot_data {
        let k = (bd * pt) as f64;
        println!(
            "  {} start: log-likelihood of the fit minus the truth per dataset {:?} (a regular maximum gains about K/2 = {:.0})",
            if oracle { "oracle (truth)" } else { "moment" },
            extra
                .iter()
                .map(|g| (g[0] * 10.0).round() / 10.0)
                .collect::<Vec<_>>(),
            k / 2.0
        );
        println!(
            "  degeneracy (medians over datasets): largest log-density excess over the median obs {:.1} at the fit vs {:.1} at the truth; max |log λ| {:.2} vs {:.2}",
            med(extra.iter().map(|g| g[1]).collect()),
            med(extra.iter().map(|g| g[3]).collect()),
            med(extra.iter().map(|g| g[2]).collect()),
            med(extra.iter().map(|g| g[4]).collect())
        );
    } else {
        println!(
            "  degeneracy (medians over datasets): largest log-density excess over the median obs {:.1} at the γ-rot fit; max |log λ| {:.2}",
            med(extra.iter().map(|g| g[1]).collect()),
            med(extra.iter().map(|g| g[2]).collect())
        );
    }
    println!(
        "d={d} n={n} {} truth, {} data, link {:?}, seeds {seed0}..{}:",
        if std::env::var("STD").is_ok() {
            "standard"
        } else {
            "hard"
        },
        if rot_data {
            format!("gamma-rot (shape slopes x{scale})")
        } else {
            "gamma".to_string()
        },
        link,
        seed0 + reps
    );
    println!(
        "  gamma-rot:   converged {}/{reps}, iterations {:?}, median {:.0} ms (forward-mode gradient), V error {:.3}",
        rows.iter().filter(|r| r.0).count(),
        rows.iter().map(|r| r.1).collect::<Vec<_>>(),
        med(rows.iter().map(|r| r.2).collect()),
        med(rows.iter().map(|r| r.3).collect())
    );
    println!(
        "  gamma:       converged {}/{reps}, median {:.0} ms, V error {:.3}",
        rows.iter().filter(|r| r.5).count(),
        med(rows.iter().map(|r| r.6).collect()),
        med(rows.iter().map(|r| r.7).collect())
    );
    println!(
        "  log-tangent: converged {}/{reps}, median {:.0} ms, V error {:.3}",
        rows.iter().filter(|r| r.8).count(),
        med(rows.iter().map(|r| r.9).collect()),
        med(rows.iter().map(|r| r.10).collect())
    );
}

/// The production `Shape::GammaRot` pipeline against the log-tangent one,
/// unpenalized and penalized, at realistic `n`. Env: `D`, `N`, `R`,
/// `SEED0` (default 1400), `STD=1` (standard truth), `DATA=rot` (data from
/// the γ-rot model read off the same coefficient matrix; `SCALE` multiplies
/// its shape slopes), `PEN=0` to skip the penalized fits. Penalized fits
/// penalize the shape slopes, select by BIC at relaxed refits, and are
/// scored at the relaxed refit: adaptive group LASSO over the eigen parts
/// (`κ_j` alone, `ω_j` as a group, group 1 as a pair) and over the groups
/// (`(κ_j, ω_j)` together) for γ-rot; adaptive LASSO and adaptive group
/// LASSO (covariate parts) for log-tangent. `V̂` is the mean geodesic error
/// of `V̂(x)` on 200 fresh points.
/// `cargo test --release --lib gamma_rot_prod -- --ignored --nocapture`
#[test]
#[ignore]
fn gamma_rot_prod() {
    use crate::gamma_rot::{GrFrame, RotLink};
    use crate::penalized::{Grouping, PenalizedEsag, Target};
    use rayon::prelude::*;
    let env = |k: &str, v: usize| {
        std::env::var(k)
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(v)
    };
    let (d, n, reps) = (env("D", 5), env("N", 2000), env("R", 8) as u64);
    let seed0 = env("SEED0", 1400) as u64;
    let rot_data = std::env::var("DATA").as_deref() == Ok("rot");
    let pen = std::env::var("PEN").as_deref() != Ok("0");
    let scale: f64 = std::env::var("SCALE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1.0);
    let mut b = if std::env::var("STD").is_ok() {
        standard_truth(d, 5)
    } else {
        hard_truth(d)
    };
    if rot_data && scale != 1.0 {
        for k in d..b.nrows() {
            for j in 1..b.ncols() {
                b[(k, j)] *= scale;
            }
        }
    }
    let (bd, pt) = (b.nrows(), b.ncols());
    // the γ-rot truth is read in the frame with reference center(d) and the
    // identity basis
    let tf = GrFrame::new(&center(d), RotLink::Stereo);
    let (xt, _) = simulate(&b, 200, 99, Shape::Gamma);
    let eta = |c: &DMatrix<f64>, i: usize| -> Vec<f64> {
        (0..bd)
            .map(|k| c[(k, 0)] + (0..pt - 1).map(|j| c[(k, j + 1)] * xt[(i, j)]).sum::<f64>())
            .collect()
    };
    let vtrue: Vec<DMatrix<f64>> = (0..200)
        .map(|i| {
            let e = eta(&b, i);
            if rot_data {
                tf.covariance(&e[..d], &e[d..])
            } else {
                Esag::from_eta(&e).unwrap().covariance()
            }
        })
        .collect();
    let score = |f: &EsagRegressorFit| -> f64 {
        let lt = f.lt_frame();
        (0..200)
            .map(|i| {
                geodesic(
                    &vtrue[i],
                    &f.distribution_at(&eta(&f.coef, i), lt.as_ref())
                        .unwrap()
                        .covariance(),
                )
            })
            .sum::<f64>()
            / 200.0
    };
    // (name, converged, ms, V error, active groups or NaN)
    type Cell = (String, bool, f64, f64, f64);
    let rows: Vec<Vec<Cell>> = (seed0..seed0 + reps)
        .into_par_iter()
        .map(|seed| {
            let (x, y) = if rot_data {
                simulate_rot(&b, n, seed, &tf)
            } else {
                simulate(&b, n, seed, Shape::Gamma)
            };
            let mut cells: Vec<Cell> = vec![];
            // the truth's log-likelihood on this dataset (γ-rot data): the
            // fit of the true model gains about K/2 over it at a regular
            // maximum
            let ll_truth = if rot_data {
                (0..n)
                    .map(|i| {
                        let e: Vec<f64> = (0..bd)
                            .map(|k| {
                                b[(k, 0)]
                                    + (0..pt - 1).map(|j| b[(k, j + 1)] * x[(i, j)]).sum::<f64>()
                            })
                            .collect();
                        let yi: Vec<f64> = (0..d).map(|k| y[(i, k)]).collect();
                        crate::gamma_rot::loglik_gr::<f64>(&tf, &e, &yi)
                    })
                    .sum::<f64>()
            } else {
                f64::NAN
            };
            for (name, shape) in [
                ("gamma-rot", Shape::GammaRot),
                ("log-tangent", Shape::LogTangent),
            ] {
                let t = Instant::now();
                let f = config(d, shape).fit(&x, &y).unwrap();
                let ms = t.elapsed().as_secs_f64() * 1e3;
                let gain = f.log_likelihood - ll_truth;
                cells.push((
                    format!("{name}, full model"),
                    f.converged,
                    ms,
                    score(&f),
                    gain,
                ));
            }
            if std::env::var("SPEC").is_ok() && d <= 8 {
                // the spectral prototype (eigenvalues + one rotation e^A,
                // forward-mode, whitened L-BFGS): the ordering-free eigen chart
                let t = Instant::now();
                let yv: Vec<f64> = (0..n)
                    .flat_map(|i| (0..d).map(move |k| (i, k)))
                    .map(|(i, k)| y[(i, k)])
                    .collect();
                let (mu0, g0) = moment_start(&yv, n, d);
                let v0 = Esag::new(&mu0, &g0).unwrap().covariance();
                let nm = mu0.iter().map(|v| v * v).sum::<f64>().sqrt();
                let m: Vec<f64> = mu0.iter().map(|v| v / nm).collect();
                let lt = crate::logtangent::LtFrame::new(&m);
                let p = d - 1;
                let e = DMatrix::from_column_slice(d, p, lt.tangent_basis());
                let tb = e.transpose() * &v0 * &e;
                let eig = nalgebra::SymmetricEigen::new((&tb + tb.transpose()) * 0.5);
                let mut w = eig.eigenvectors.clone();
                if w.determinant() < 0.0 {
                    for r in 0..p {
                        w[(r, 0)] = -w[(r, 0)];
                    }
                }
                let r0: Vec<f64> = (0..p)
                    .flat_map(|r| (0..p).map(move |c| (r, c)))
                    .map(|(r, c)| w[(r, c)])
                    .collect();
                let sfr = SpFrame { lt, r0 };
                let ell: Vec<f64> = eig.eigenvalues.iter().map(|l| l.ln()).collect();
                let mean = ell.iter().sum::<f64>() / p as f64;
                let ell: Vec<f64> = ell.iter().map(|l| l - mean).collect();
                let mut start = vec![0.0; bd * pt];
                start[..d].copy_from_slice(&mu0);
                for mm in 1..p {
                    let h = 1.0 / ((mm * (mm + 1)) as f64).sqrt();
                    start[d + mm - 1] =
                        (0..mm).map(|k| ell[k] * h).sum::<f64>() - ell[mm] * mm as f64 * h;
                }
                let kg = |eta: &[f64], yi: &[f64], g: &mut [f64]| sp_grad(&sfr, eta, yi, g);
                let (c, conv, _it) = fit_kernel(&kg, start, &x, &y);
                let ms = t.elapsed().as_secs_f64() * 1e3;
                let err = (0..200)
                    .map(|i| geodesic(&vtrue[i], &sp_covariance(&sfr, &eta(&c, i))))
                    .sum::<f64>()
                    / 200.0;
                cells.push((
                    "spectral (prototype), full model".to_string(),
                    conv,
                    ms,
                    err,
                    f64::NAN,
                ));
            }
            if rot_data {
                // the same model class as the truth (its basis, the identity),
                // from the default starts, and from the truth itself
                let eye = DMatrix::identity(d - 1, d - 1);
                for (name, init) in [
                    ("gamma-rot, full model, truth's basis", Init::Auto),
                    (
                        "gamma-rot, full model, from the truth",
                        Init::Custom(b.clone()),
                    ),
                ] {
                    let t = Instant::now();
                    let f = config(d, Shape::GammaRot)
                        .with_init(init)
                        .with_rot_basis(Some(eye.clone()))
                        .fit(&x, &y)
                        .unwrap();
                    let ms = t.elapsed().as_secs_f64() * 1e3;
                    let gain = f.log_likelihood - ll_truth;
                    cells.push((name.to_string(), f.converged, ms, score(&f), gain));
                }
            }
            if pen {
                let presets: Vec<(&str, PenalizedEsag)> = vec![
                    (
                        "gamma-rot, adaptive group LASSO (eigen parts)",
                        PenalizedEsag::adaptive_group_lasso()
                            .grouping(Grouping::EigenParts)
                            .shape(Shape::GammaRot),
                    ),
                    (
                        "gamma-rot, adaptive group LASSO (groups)",
                        PenalizedEsag::adaptive_group_lasso()
                            .grouping(Grouping::GammaGroups)
                            .shape(Shape::GammaRot),
                    ),
                    (
                        "gamma-rot, adaptive LASSO",
                        PenalizedEsag::adaptive_lasso().shape(Shape::GammaRot),
                    ),
                    (
                        "log-tangent, adaptive LASSO",
                        PenalizedEsag::adaptive_lasso().shape(Shape::LogTangent),
                    ),
                    (
                        "log-tangent, adaptive group LASSO (cov. parts)",
                        PenalizedEsag::adaptive_group_lasso()
                            .grouping(Grouping::CovariatePart)
                            .shape(Shape::LogTangent),
                    ),
                ];
                for (name, pe) in presets {
                    if let Ok(f) = std::env::var("PRESET")
                        && !name.contains(f.as_str())
                    {
                        continue;
                    }
                    let pe = pe
                        .target(Target::ShapeSlopes)
                        .lt_reference(center(d))
                        .bic()
                        .with_parallel(false);
                    let t = Instant::now();
                    match pe.fit(&x, &y) {
                        Ok(path) => {
                            let ms = t.elapsed().as_secs_f64() * 1e3;
                            let best = path.best();
                            let active = best.active.iter().filter(|a| **a).count() as f64;
                            let (conv, err) = match path.relaxed.as_ref() {
                                Some(r) => (r.converged, score(r)),
                                None => (best.converged, f64::NAN),
                            };
                            cells.push((name.to_string(), conv, ms, err, active));
                        }
                        Err(e) => {
                            eprintln!("  seed {seed}: {name} failed: {e}");
                            cells.push((name.to_string(), false, f64::NAN, f64::NAN, f64::NAN));
                        }
                    }
                }
            }
            if std::env::var("TRACE").is_ok() {
                for c in &cells {
                    println!(
                        "  seed {seed}: {:<48} {} {:8.0} ms  V {:.3}  active {}",
                        c.0,
                        if c.1 { "ok  " } else { "FAIL" },
                        c.2,
                        c.3,
                        c.4
                    );
                }
            }
            cells
        })
        .collect();
    let med = |mut v: Vec<f64>| {
        v.retain(|x| x.is_finite());
        if v.is_empty() {
            return f64::NAN;
        }
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    };
    println!(
        "d={d} n={n} {} truth, {} data, seeds {seed0}..{}, penalties on the shape slopes, BIC at relaxed refits:",
        if std::env::var("STD").is_ok() {
            "standard"
        } else {
            "hard"
        },
        if rot_data {
            format!("gamma-rot (shape slopes x{scale})")
        } else {
            "gamma".to_string()
        },
        seed0 + reps
    );
    for k in 0..rows[0].len() {
        let name = &rows[0][k].0;
        let conv = rows.iter().filter(|r| r[k].1).count();
        let extra = {
            let a = med(rows.iter().map(|r| r[k].4).collect());
            if !a.is_finite() {
                String::new()
            } else if name.contains("full model") {
                format!(
                    ", log-lik gain over the truth {a:.0} (K/2 = {:.0})",
                    (bd * pt) as f64 / 2.0
                )
            } else {
                format!(", active groups {a:.0}")
            }
        };
        println!(
            "  {name:<48} converged {conv}/{reps}, median {:7.0} ms, V error {:.3}{extra}",
            med(rows.iter().map(|r| r[k].2).collect()),
            med(rows.iter().map(|r| r[k].3).collect()),
        );
    }
}
