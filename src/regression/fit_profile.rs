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
    if shape == Shape::LogTangent {
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
