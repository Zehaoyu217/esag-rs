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

/// A gamma fit with observations pinned at the singular set is flagged and
/// explained; the log-tangent fit of the same data converges.
#[test]
fn pinned_gamma_fit_is_reported() {
    let d = 5;
    let (x, y) = simulate(&hard_truth(d), 500, 900, Shape::Gamma);
    let fit = EsagRegressor::new()
        .with_parallel(false)
        .fit(&x, &y)
        .unwrap();
    assert!(!fit.converged);
    assert!(
        !fit.singular_obs.is_empty(),
        "no singular observations found"
    );
    assert!(fit.summary().contains("singular set of the gamma"));
    let lt = config(d, Shape::LogTangent).fit(&x, &y).unwrap();
    assert!(lt.converged && lt.singular_obs.is_empty());
}

/// A trapped γ fit (d = 5, hard scenario, n = 500, seed 950) is rescued by
/// exclusion rounds: they reach a regular root (converged, no trapped
/// observation) with a higher likelihood than the trap.
#[test]
fn trapped_gamma_fit_is_rescued() {
    let (x, y) = simulate(&hard_truth(5), 500, 950, Shape::Gamma);
    let cfg = config(5, Shape::Gamma);
    let data = cfg.data(&x, &y).unwrap();
    let mask = cfg.resolve_mask(&data).unwrap();
    let b0 = cfg.initial_coef(&data).unwrap();
    let raw = cfg.optimize(&data, &b0, &free_params(&mask), cfg.max_iter);
    assert!(!raw.converged && !singular_observations(&data, &raw.coef).is_empty());
    let trap = raw.neg_ll;
    let res = cfg.exclusion_rounds(&x, &y, &data, raw, &mask);
    assert!(res.converged, "not rescued");
    assert!(singular_observations(&data, &res.coef).is_empty());
    assert!(res.neg_ll < trap - 1.0, "{} vs trap {trap}", res.neg_ll);
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
            let res = cfg.exclusion_rounds(&x, &y, &data, raw, &mask);
            (res.converged && res.neg_ll < before).then_some((seed, before - res.neg_ll))
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
