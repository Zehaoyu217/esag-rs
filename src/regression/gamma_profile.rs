//! Single-core cost of γ fits (`Shape::Gamma`): where the time goes, and
//! how the start strategy changes evaluations and estimates (all
//! `#[ignore]`).

use super::fit_profile::{center, hard_truth, simulate_with, standard_truth};
use super::*;
use std::time::Instant;

/// Data of the hard (`hard_truth`, ρ = 0.5) or standard (`standard_truth`,
/// ρ = 0) scenario with 5 covariates, simulated from the γ model.
pub(crate) fn scenario(d: usize, hard: bool, n: usize, seed: u64) -> (DMatrix<f64>, DMatrix<f64>) {
    if hard {
        simulate_with(&hard_truth(d), n, seed, Shape::Gamma, 0.5, &center(d))
    } else {
        let mut m = vec![0.0; d];
        m[d - 1] = 1.0;
        simulate_with(&standard_truth(d, 5), n, seed, Shape::Gamma, 0.0, &m)
    }
}

fn env_or(k: &str, v: usize) -> usize {
    std::env::var(k)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(v)
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

/// Phases of one-core γ fits with standard errors: the race (both starts),
/// and `finalize` (Hessian, Newton polish), with the evaluations of each
/// start. `cargo test --release --lib gamma_fit_phases -- --ignored --nocapture`
#[test]
#[ignore]
fn gamma_fit_phases() {
    let seeds = env_or("SEEDS", 6) as u64;
    let n = env_or("N", 2000);
    for (label, hard) in [("hard", true), ("standard", false)] {
        for d in [3usize, 4, 5] {
            let mut rows = vec![];
            for seed in 0..seeds {
                let (x, y) = scenario(d, hard, n, 500 + seed);
                let cfg = EsagRegressor::new().with_parallel(false);
                let t = Instant::now();
                let fit = cfg.fit(&x, &y).unwrap();
                let total = t.elapsed().as_secs_f64() * 1e3;
                let data = cfg.data(&x, &y).unwrap();
                let mask = cfg.resolve_mask(&data).unwrap();
                let free = free_params(&mask);
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
                let dd = cfg.optimize(&data, &starts[0], &free, cfg.max_iter);
                let rs = cfg.optimize(&data, &starts[1], &free, cfg.max_iter);
                let t = Instant::now();
                std::hint::black_box(joint_loglik_grad_hess(&data, &fit.coef, false));
                let t_hess = t.elapsed().as_secs_f64() * 1e3;
                let t = Instant::now();
                std::hint::black_box(joint_loglik_and_grad(&data, &fit.coef, false));
                let t_grad = t.elapsed().as_secs_f64() * 1e3;
                rows.push((
                    total,
                    t_race,
                    t_hess,
                    t_grad,
                    dd.n_fev,
                    rs.n_fev,
                    fit.converged,
                    -dd.neg_ll,
                    -rs.neg_ll,
                    race.n_fev,
                ));
                println!(
                    "{label:<8} d={d} seed {seed}: fit {total:6.1} ms (conv {}) | race {t_race:6.1} ms | hess {t_hess:5.2} ms grad {t_grad:5.3} ms | moment {} fev ll {:.3} | R-style {} fev ll {:.3}",
                    fit.converged, dd.n_fev, -dd.neg_ll, rs.n_fev, -rs.neg_ll
                );
            }
            let mut tot: Vec<f64> = rows.iter().map(|r| r.0).collect();
            let mut rc: Vec<f64> = rows.iter().map(|r| r.1).collect();
            println!(
                "{label:<8} d={d}: median fit {:.1} ms, race {:.1} ms; converged {}/{}",
                median(&mut tot),
                median(&mut rc),
                rows.iter().filter(|r| r.6).count(),
                rows.len()
            );
        }
    }
}

/// Metric of a whitened start: `L Lᵀ = I⁻¹` at the start.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Metric {
    /// The exact observed information.
    Exact,
    /// The empirical information (outer product of the scores), once.
    Opg,
    /// The empirical information, refreshed every `METRIC_REFRESH`
    /// iterations (`local_precondition`).
    OpgRefresh,
}

/// One whitened L-BFGS run from `b0` (with standard errors requested, so
/// the hand-over rule of fits applies); `None` if the metric is not
/// positive definite.
pub(crate) fn whitened_run(
    cfg: &EsagRegressor,
    data: &RegressionData,
    b0: &DMatrix<f64>,
    free: &[(usize, usize)],
    metric: Metric,
    max_iter: usize,
) -> Option<RawFit> {
    let mut c = cfg.clone();
    match metric {
        Metric::Exact => {
            c.local_precondition = false;
            c.precondition = Some(info_precondition(data, b0, free, cfg.parallel)?);
        }
        Metric::Opg => {
            c.local_precondition = false;
            c.precondition = Some(empirical_precondition(data, b0, free)?);
        }
        Metric::OpgRefresh => {
            c.local_precondition = true;
            c.precondition = None;
        }
    }
    Some(c.optimize(data, b0, free, max_iter))
}

/// A candidate start strategy, run to the raw fit `finalize` takes, with
/// the evaluations it used (all starts).
pub(crate) fn strategy_raw(
    cfg: &EsagRegressor,
    data: &RegressionData,
    mask: &DMatrix<bool>,
    name: &str,
) -> (RawFit, usize) {
    let free = free_params(mask);
    let starts: Vec<DMatrix<f64>> = [Init::DataDriven, Init::RStyle]
        .iter()
        .map(|i| cfg.clone().with_init(i.clone()).initial_coef(data).unwrap())
        .collect();
    let metric = |s: &str| match s {
        "exact" => Metric::Exact,
        "opg" => Metric::Opg,
        _ => Metric::OpgRefresh,
    };
    let parts: Vec<&str> = name.split('+').collect();
    match parts[0] {
        // the moment start alone, whitened; the R-style start (plain) only
        // if it fails the convergence check
        "single" => {
            let first = whitened_run(cfg, data, &starts[0], &free, metric(parts[1]), cfg.max_iter)
                .unwrap_or_else(|| cfg.optimize(data, &starts[0], &free, cfg.max_iter));
            let mut fev = first.n_fev;
            if first.converged {
                return (first, fev);
            }
            let second = cfg.optimize(data, &starts[1], &free, cfg.max_iter);
            fev += second.n_fev;
            (if second.beats(&first) { second } else { first }, fev)
        }
        // both starts whitened, each with its own metric, run to the end
        "both" => {
            let a = whitened_run(cfg, data, &starts[0], &free, metric(parts[1]), cfg.max_iter)
                .unwrap_or_else(|| cfg.optimize(data, &starts[0], &free, cfg.max_iter));
            let b = whitened_run(cfg, data, &starts[1], &free, metric(parts[1]), cfg.max_iter)
                .unwrap_or_else(|| cfg.optimize(data, &starts[1], &free, cfg.max_iter));
            let fev = a.n_fev + b.n_fev;
            (if b.beats(&a) { b } else { a }, fev)
        }
        other => panic!("unknown strategy {other}"),
    }
}

/// Per dataset: production (ll, converged, ms), then per strategy (ll,
/// converged, ms, evaluations).
type StrategyRow = ((f64, bool, f64), Vec<(f64, bool, f64, usize)>);

/// Production fits against candidate start strategies on the same data:
/// final log-likelihoods (all through `finalize`), convergence and one-core
/// time. `SEEDS`, `N`, `D` (comma list), `STRATS` (comma list, e.g.
/// `single+exact,single+opg,both+opg`).
/// `cargo test --release --lib gamma_start_strategies -- --ignored --nocapture`
#[test]
#[ignore]
fn gamma_start_strategies() {
    let seeds = env_or("SEEDS", 24) as u64;
    let n = env_or("N", 2000);
    let ds: Vec<usize> = std::env::var("D")
        .ok()
        .map(|v| v.split(',').map(|x| x.parse().unwrap()).collect())
        .unwrap_or_else(|| vec![3, 4, 5]);
    let strats: Vec<String> = std::env::var("STRATS")
        .ok()
        .map(|v| v.split(',').map(String::from).collect())
        .unwrap_or_else(|| {
            ["single+exact", "single+opg", "single+opgr", "both+opg"]
                .iter()
                .map(|s| s.to_string())
                .collect()
        });
    let scen: Vec<(&str, bool)> = match std::env::var("SCEN").as_deref() {
        Ok("hard") => vec![("hard", true)],
        Ok("standard") => vec![("standard", false)],
        _ => vec![("hard", true), ("standard", false)],
    };
    for &(label, hard) in &scen {
        for &d in &ds {
            let cfg = EsagRegressor::new().with_parallel(false);
            // per seed: production (ll, conv, ms), then per strategy (ll, conv, ms, fev)
            let rows: Vec<StrategyRow> = (0..seeds)
                .into_par_iter()
                .map(|seed| {
                    let (x, y) = scenario(d, hard, n, 500 + seed);
                    let t = Instant::now();
                    let prod = cfg.fit(&x, &y).unwrap();
                    let tp = t.elapsed().as_secs_f64() * 1e3;
                    let data = cfg.data(&x, &y).unwrap();
                    let mask = cfg.resolve_mask(&data).unwrap();
                    let cand = strats
                        .iter()
                        .map(|s| {
                            let t = Instant::now();
                            let (raw, fev) = strategy_raw(&cfg, &data, &mask, s);
                            let fit = cfg.finalize(&data, raw, mask.clone()).unwrap();
                            let ms = t.elapsed().as_secs_f64() * 1e3;
                            (fit.log_likelihood, fit.converged, ms, fev)
                        })
                        .collect();
                    ((prod.log_likelihood, prod.converged, tp), cand)
                })
                .collect();
            let mut tp: Vec<f64> = rows.iter().map(|r| r.0.2).collect();
            println!(
                "{label:<8} d={d} n={n}: production converged {}/{seeds}, median {:.1} ms",
                rows.iter().filter(|r| r.0.1).count(),
                median(&mut tp)
            );
            for (k, s) in strats.iter().enumerate() {
                let diffs: Vec<f64> = rows.iter().map(|r| r.1[k].0 - r.0.0).collect();
                let worse: Vec<f64> = diffs.iter().copied().filter(|&v| v < -1e-6).collect();
                let better: Vec<f64> = diffs.iter().copied().filter(|&v| v > 1e-6).collect();
                let mut ms: Vec<f64> = rows.iter().map(|r| r.1[k].2).collect();
                let fev: usize = rows.iter().map(|r| r.1[k].3).sum();
                println!(
                    "    {s:<14} converged {:>2}/{seeds}, median {:6.1} ms, mean fev {:6.1} | worse in {:>2} (by {}), better in {:>2} (by {})",
                    rows.iter().filter(|r| r.1[k].1).count(),
                    median(&mut ms),
                    fev as f64 / seeds as f64,
                    worse.len(),
                    worse
                        .iter()
                        .map(|v| format!("{v:.3}"))
                        .collect::<Vec<_>>()
                        .join(" "),
                    better.len(),
                    better
                        .iter()
                        .map(|v| format!("{v:.3}"))
                        .collect::<Vec<_>>()
                        .join(" "),
                );
            }
        }
    }
}

/// The kernel alone over the observations of a dataset (η precomputed),
/// against the joint gradient. `cargo test --release --lib kernel_alone -- --ignored --nocapture`
#[test]
#[ignore]
fn kernel_alone() {
    for d in [3usize, 5, 8] {
        let (x, y) = scenario(d, true, 2000, 7);
        let cfg = EsagRegressor::new().with_parallel(false);
        let data = cfg.data(&x, &y).unwrap();
        let b = hard_truth(d);
        let (bd, pt) = (data.b_dim, data.p_total);
        let mut etas = vec![0.0; data.n * bd];
        for i in 0..data.n {
            let xi = &data.x[i * pt..(i + 1) * pt];
            for k in 0..bd {
                etas[i * bd + k] = (0..pt).map(|j| b[(k, j)] * xi[j]).sum::<f64>();
            }
        }
        let mut g = vec![0.0; bd];
        let tk = super::fit_profile_median_us(31, || {
            let mut f = 0.0;
            for i in 0..data.n {
                f += kernel_grad(
                    &data,
                    &etas[i * bd..(i + 1) * bd],
                    &data.y[i * d..(i + 1) * d],
                    &mut g,
                );
            }
            std::hint::black_box(f);
        });
        let tj = super::fit_profile_median_us(31, || {
            std::hint::black_box(joint_loglik_and_grad(&data, &b, false));
        });
        let alphas: Vec<f64> = (0..data.n)
            .map(|i| {
                let e = &etas[i * bd..(i + 1) * bd];
                let yi = &data.y[i * d..(i + 1) * d];
                let (q, ymu, _) = {
                    let (mu, gamma) = e.split_at(d);
                    let mut z = vec![0.0; d];
                    let mut kk = vec![0.0; d];
                    let mut rt = vec![0.0; d];
                    crate::loglik::quad_form_parts(mu, gamma, yi, &mut z, &mut kk, &mut rt)
                };
                ymu / q.sqrt()
            })
            .collect();
        let below = alphas.iter().filter(|&&a| a < -2.0).count();
        let above = alphas.iter().filter(|&&a| a > 8.5).count();
        println!(
            "d={d}: kernel over the data {tk:.1} us ({:.0} ns/obs), joint gradient {tj:.1} us; alpha < -2 in {below}, > 8.5 in {above} of {}",
            tk * 1e3 / data.n as f64,
            data.n
        );
    }
}

/// How close the observations of a γ fit at `b` come to the non-smooth sets
/// of the parameterization, relative to the median: for each group `m`
/// the norm `‖G_m,i‖` (the cone at 0) and, for `m ≥ 2`, the trailing pair.
/// Returns (smallest ratio over observations and groups, number of
/// observations below `tau` times the median).
pub(crate) fn cone_proximity(data: &RegressionData, b: &DMatrix<f64>, tau: f64) -> (f64, usize) {
    let (d, bd, pt) = (data.d, data.b_dim, data.p_total);
    let ng = d - 2;
    let mut eta = vec![0.0; bd];
    // per group: norms over observations, and trailing-pair norms
    let mut norms: Vec<Vec<f64>> = vec![Vec::with_capacity(data.n); 2 * ng];
    for i in 0..data.n {
        for (k, e) in eta.iter_mut().enumerate() {
            *e = (0..pt).map(|j| b[(k, j)] * data.x[i * pt + j]).sum();
        }
        for gr in 1..=ng {
            let off = d + gr * (gr + 1) / 2 - 1;
            let g = &eta[off..off + gr + 1];
            norms[gr - 1].push(g.iter().map(|v| v * v).sum::<f64>().sqrt());
            if gr >= 2 {
                norms[ng + gr - 1].push((g[gr - 1].powi(2) + g[gr].powi(2)).sqrt());
            }
        }
    }
    let mut worst = f64::INFINITY;
    let mut near = vec![false; data.n];
    for v in norms.iter().filter(|v| !v.is_empty()) {
        let mut s = v.clone();
        let mid = s.len() / 2;
        let (_, med, _) = s.select_nth_unstable_by(mid, f64::total_cmp);
        let med = med.max(f64::MIN_POSITIVE);
        for (i, &r) in v.iter().enumerate() {
            worst = worst.min(r / med);
            if r < tau * med {
                near[i] = true;
            }
        }
    }
    (worst, near.iter().filter(|&&f| f).count())
}

/// Smallest relative distance of a fit's mean directions to the pole of
/// the frame `onb(mu)`, `√(mu_0² + mu_1²)/‖mu‖`, over the observations.
pub(crate) fn pole_proximity(data: &RegressionData, b: &DMatrix<f64>) -> f64 {
    let (d, pt) = (data.d, data.p_total);
    (0..data.n)
        .map(|i| {
            let mu: Vec<f64> = (0..d)
                .map(|k| (0..pt).map(|j| b[(k, j)] * data.x[i * pt + j]).sum::<f64>())
                .collect();
            let mm: f64 = mu.iter().map(|v| v * v).sum();
            if mm > 0.0 {
                ((mu[0] * mu[0] + mu[1] * mu[1]) / mm).sqrt()
            } else {
                0.0
            }
        })
        .fold(f64::INFINITY, f64::min)
}

/// Summary of [`gamma_single_diagnostic`] over cells (d, scenario, n): for
/// each threshold on the cone ratio, the datasets it would leave to the
/// single whitened start, and how many of those end below production.
/// `POLE=1` centres the hard scenario at the frame pole.
/// `cargo test --release --lib gamma_single_threshold -- --ignored --nocapture`
#[test]
#[ignore]
fn gamma_single_threshold() {
    let seeds = env_or("SEEDS", 40) as u64;
    let ns: Vec<usize> = std::env::var("NS")
        .ok()
        .map(|v| v.split(',').map(|x| x.parse().unwrap()).collect())
        .unwrap_or_else(|| vec![300, 1000, 2000]);
    let ds: Vec<usize> = std::env::var("D")
        .ok()
        .map(|v| v.split(',').map(|x| x.parse().unwrap()).collect())
        .unwrap_or_else(|| vec![3, 4, 5]);
    let scen: Vec<(&str, bool)> = match std::env::var("SCEN").as_deref() {
        Ok("hard") => vec![("hard", true)],
        Ok("standard") => vec![("standard", false)],
        _ => vec![("hard", true), ("standard", false)],
    };
    let taus = [0.05, 0.1, 0.15, 0.2, 0.3];
    let pole_tau = 0.05;
    for &(label, hard) in &scen {
        for &d in &ds {
            for &n in &ns {
                let cfg = EsagRegressor::new().with_parallel(false);
                // (ll diff, min cone ratio, min pole distance, prod conv,
                // single conv, prod ms, single ms)
                let rows: Vec<(f64, f64, f64, bool, bool, f64, f64)> = (0..seeds)
                    .into_par_iter()
                    .map(|seed| {
                        let (x, y) = scenario(d, hard, n, 500 + seed);
                        let t = Instant::now();
                        let prod = cfg.fit(&x, &y).unwrap();
                        let tp = t.elapsed().as_secs_f64() * 1e3;
                        let data = cfg.data(&x, &y).unwrap();
                        let mask = cfg.resolve_mask(&data).unwrap();
                        let t = Instant::now();
                        let (raw, _) = strategy_raw(&cfg, &data, &mask, "single+exact");
                        let fit = cfg.finalize(&data, raw, mask).unwrap();
                        let ts = t.elapsed().as_secs_f64() * 1e3;
                        let (w, _) = cone_proximity(&data, &fit.coef, 0.05);
                        let pd = pole_proximity(&data, &fit.coef);
                        (
                            fit.log_likelihood - prod.log_likelihood,
                            w,
                            pd,
                            prod.converged,
                            fit.converged,
                            tp,
                            ts,
                        )
                    })
                    .collect();
                let worse = rows.iter().filter(|r| r.0 < -1e-6).count();
                let better = rows.iter().filter(|r| r.0 > 1e-6).count();
                let mut tp: Vec<f64> = rows.iter().map(|r| r.5).collect();
                let mut ts: Vec<f64> = rows.iter().map(|r| r.6).collect();
                print!(
                    "{label:<8} d={d} n={n:>4}: prod conv {:>2}/{seeds} single conv {:>2}, single worse {worse:>2} better {better:>2}; ms {:.0} vs {:.0} |",
                    rows.iter().filter(|r| r.3).count(),
                    rows.iter().filter(|r| r.4).count(),
                    median(&mut tp),
                    median(&mut ts)
                );
                for &tau in &taus {
                    let kept: Vec<&(f64, f64, f64, bool, bool, f64, f64)> = rows
                        .iter()
                        .filter(|r| r.4 && r.1 >= tau && r.2 >= pole_tau)
                        .collect();
                    let lost: Vec<f64> = kept.iter().filter(|r| r.0 < -1e-6).map(|r| r.0).collect();
                    let worst = lost.iter().fold(0.0f64, |m, &v| m.min(v));
                    print!(
                        " τ={tau}: {:>2} kept, {} lost ({worst:.3})",
                        kept.len(),
                        lost.len()
                    );
                }
                println!();
            }
        }
    }
}

/// One standard d = 3, n = 300 dataset where the moment start ends below the
/// race: both endpoints' coefficients and log-likelihoods, and each start of
/// the race alone. `SEED` (default: scan for the first loss).
/// `cargo test --release --lib gamma_small_n_case -- --ignored --nocapture`
#[test]
#[ignore]
fn gamma_small_n_case() {
    let (d, n) = (env_or("D", 3), env_or("N", 300));
    let cfg = EsagRegressor::new().with_parallel(false);
    let mut shown = 0;
    for seed in 0..40u64 {
        let (x, y) = scenario(d, false, n, 500 + seed);
        let prod = cfg.fit(&x, &y).unwrap();
        let data = cfg.data(&x, &y).unwrap();
        let mask = cfg.resolve_mask(&data).unwrap();
        let free = free_params(&mask);
        let (raw, _) = strategy_raw(&cfg, &data, &mask, "single+exact");
        let single = cfg.finalize(&data, raw, mask.clone()).unwrap();
        if single.log_likelihood > prod.log_likelihood - 1e-6 {
            continue;
        }
        let starts: Vec<DMatrix<f64>> = [Init::DataDriven, Init::RStyle]
            .iter()
            .map(|i| {
                cfg.clone()
                    .with_init(i.clone())
                    .initial_coef(&data)
                    .unwrap()
            })
            .collect();
        let dd = cfg.optimize(&data, &starts[0], &free, cfg.max_iter);
        let rs = cfg.optimize(&data, &starts[1], &free, cfg.max_iter);
        println!(
            "seed {seed}: production {:.4} (conv {}), single {:.4}; plain moment start alone {:.4} ({} it), R-style alone {:.4} ({} it)",
            prod.log_likelihood,
            prod.converged,
            single.log_likelihood,
            -dd.neg_ll,
            dd.n_iter,
            -rs.neg_ll,
            rs.n_iter
        );
        println!(
            "  truth γ rows:      {:?}",
            (d..data.b_dim)
                .map(|k| (0..data.p_total)
                    .map(|j| format!("{:+.2}", standard_truth(d, 5)[(k, j)]))
                    .collect::<Vec<_>>())
                .collect::<Vec<_>>()
        );
        println!(
            "  production γ rows: {:?}",
            (d..data.b_dim)
                .map(|k| (0..data.p_total)
                    .map(|j| format!("{:+.2}", prod.coef[(k, j)]))
                    .collect::<Vec<_>>())
                .collect::<Vec<_>>()
        );
        println!(
            "  single γ rows:     {:?}",
            (d..data.b_dim)
                .map(|k| (0..data.p_total)
                    .map(|j| format!("{:+.2}", single.coef[(k, j)]))
                    .collect::<Vec<_>>())
                .collect::<Vec<_>>()
        );
        println!(
            "  production μ rows: {:?}",
            (0..d)
                .map(|k| (0..data.p_total)
                    .map(|j| format!("{:+.2}", prod.coef[(k, j)]))
                    .collect::<Vec<_>>())
                .collect::<Vec<_>>()
        );
        println!(
            "  single μ rows:     {:?}",
            (0..d)
                .map(|k| (0..data.p_total)
                    .map(|j| format!("{:+.2}", single.coef[(k, j)]))
                    .collect::<Vec<_>>())
                .collect::<Vec<_>>()
        );
        let (wp, kp) = cone_proximity(&data, &prod.coef, 0.05);
        let (ws, ks) = cone_proximity(&data, &single.coef, 0.05);
        println!("  cone proximity: production {wp:.3} ({kp} obs < 5%), single {ws:.3} ({ks})");
        shown += 1;
        if shown == 3 {
            break;
        }
    }
}

/// Exact Newton over the free coefficients from `b0`, with step halving,
/// until the predicted remaining gain `½ gᵀ(−H)⁻¹g` is at most `tol`.
/// `None` if the Hessian is not negative definite at an iterate, a step
/// cannot be made, or `max_steps` are used up. Returns the raw fit and the
/// number of Hessians.
pub(crate) fn newton_finish(
    cfg: &EsagRegressor,
    data: &RegressionData,
    free: &[(usize, usize)],
    b0: &DMatrix<f64>,
    max_steps: usize,
    tol: f64,
) -> Option<(RawFit, usize)> {
    let (bd, full) = (data.b_dim, data.b_dim * data.p_total);
    let nf = free.len();
    let mut b = b0.clone();
    for k in 0..max_steps {
        let (f, g, h) = joint_loglik_grad_hess(data, &b, cfg.parallel);
        let gv = DVector::from_iterator(nf, free.iter().map(|&(r, j)| g[j * bd + r]));
        let hm = DMatrix::from_fn(nf, nf, |a, c| {
            let ((ra, ja), (rc, jc)) = (free[a], free[c]);
            -h[(ja * bd + ra) * full + jc * bd + rc]
        });
        let ch = hm.cholesky()?;
        let step = ch.solve(&gv);
        let dec = 0.5 * gv.dot(&step);
        if dec <= tol {
            let grad: Vec<f64> = gv.iter().map(|v| -v).collect();
            let gn = grad.iter().fold(0.0f64, |m, v| m.max(v.abs()));
            return Some((
                RawFit {
                    coef: b.clone(),
                    x: free.iter().map(|&(r, j)| b[(r, j)]).collect(),
                    neg_ll: -f,
                    grad,
                    grad_norm_inf: gn,
                    n_iter: k,
                    n_fev: k,
                    status: Status::GradientConverged,
                    converged: f.is_finite() && gn <= MAX_MEAN_GRADIENT * data.n.max(1) as f64,
                },
                k + 1,
            ));
        }
        let mut t = 1.0;
        loop {
            let mut cand = b.clone();
            for (a, &(r, j)) in free.iter().enumerate() {
                cand[(r, j)] += t * step[a];
            }
            let (f2, _) = joint_loglik_and_grad(data, &cand, cfg.parallel);
            if f2.is_finite() && f2 >= f {
                b = cand;
                break;
            }
            t *= 0.5;
            if t < 1e-4 {
                return None;
            }
        }
    }
    None
}

/// The race stopped at mean gradient `coarse` per observation (production's
/// L-BFGS paths up to there), then every start within `margin` of the best
/// one finished with exact Newton (L-BFGS at full tolerance where Newton
/// fails); the best finished start wins. Returns the raw fit and the
/// gradient-equivalents used (Hessians counted as `hess_cost` gradients).
pub(crate) fn coarse_newton_raw(
    cfg: &EsagRegressor,
    data: &RegressionData,
    mask: &DMatrix<bool>,
    coarse: f64,
    margin: f64,
) -> RawFit {
    let free = free_params(mask);
    let starts: Vec<DMatrix<f64>> = [Init::DataDriven, Init::RStyle]
        .iter()
        .map(|i| cfg.clone().with_init(i.clone()).initial_coef(data).unwrap())
        .collect();
    let mut c = cfg.clone();
    c.handover = Some(coarse);
    let runs = c.race_all(data, &starts, mask, cfg.max_iter);
    let best_f = runs.iter().map(|r| r.neg_ll).fold(f64::INFINITY, f64::min);
    let mut best: Option<RawFit> = None;
    for r in runs {
        if !(r.neg_ll <= best_f + margin) {
            continue;
        }
        // a run stopped by the race (aborted or capped) keeps its L-BFGS
        // point; only coarse-converged runs are finished
        let fin = if r.status.is_converged() {
            newton_finish(cfg, data, &free, &r.coef, 12, HANDOFF_DECREMENT)
                .map(|(f, _)| f)
                .unwrap_or_else(|| cfg.optimize(data, &r.coef, &free, cfg.max_iter))
        } else {
            r
        };
        if best.as_ref().is_none_or(|b| fin.beats(b)) {
            best = Some(fin);
        }
    }
    best.expect("at least one start")
}

/// Production against the coarse race with Newton finishing, paired:
/// final log-likelihoods (both through `finalize`), convergence, one-core
/// time. `COARSE` (default 1e-2), `MARGIN` (default 1.0).
/// `cargo test --release --lib gamma_coarse_newton -- --ignored --nocapture`
#[test]
#[ignore]
fn gamma_coarse_newton() {
    let seeds = env_or("SEEDS", 40) as u64;
    let coarse: f64 = std::env::var("COARSE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1e-2);
    let margin: f64 = std::env::var("MARGIN")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1.0);
    let ns: Vec<usize> = std::env::var("NS")
        .ok()
        .map(|v| v.split(',').map(|x| x.parse().unwrap()).collect())
        .unwrap_or_else(|| vec![300, 1000, 2000]);
    let ds: Vec<usize> = std::env::var("D")
        .ok()
        .map(|v| v.split(',').map(|x| x.parse().unwrap()).collect())
        .unwrap_or_else(|| vec![3, 4, 5]);
    let scen: Vec<(&str, bool)> = match std::env::var("SCEN").as_deref() {
        Ok("hard") => vec![("hard", true)],
        Ok("standard") => vec![("standard", false)],
        _ => vec![("hard", true), ("standard", false)],
    };
    for &(label, hard) in &scen {
        for &d in &ds {
            for &n in &ns {
                let cfg = EsagRegressor::new().with_parallel(false);
                let rows: Vec<(f64, bool, bool, f64, f64)> = (0..seeds)
                    .into_par_iter()
                    .map(|seed| {
                        let (x, y) = scenario(d, hard, n, 500 + seed);
                        let t = Instant::now();
                        let prod = cfg.fit(&x, &y).unwrap();
                        let tp = t.elapsed().as_secs_f64() * 1e3;
                        let data = cfg.data(&x, &y).unwrap();
                        let mask = cfg.resolve_mask(&data).unwrap();
                        let t = Instant::now();
                        let raw = coarse_newton_raw(&cfg, &data, &mask, coarse, margin);
                        let fit = cfg.finalize(&data, raw, mask).unwrap();
                        let tc = t.elapsed().as_secs_f64() * 1e3;
                        (
                            fit.log_likelihood - prod.log_likelihood,
                            prod.converged,
                            fit.converged,
                            tp,
                            tc,
                        )
                    })
                    .collect();
                let worse: Vec<f64> = rows.iter().filter(|r| r.0 < -1e-6).map(|r| r.0).collect();
                let better: Vec<f64> = rows.iter().filter(|r| r.0 > 1e-6).map(|r| r.0).collect();
                let mut tp: Vec<f64> = rows.iter().map(|r| r.3).collect();
                let mut tc: Vec<f64> = rows.iter().map(|r| r.4).collect();
                let sum_tp: f64 = tp.iter().sum();
                let sum_tc: f64 = tc.iter().sum();
                println!(
                    "{label:<8} d={d} n={n:>4}: conv prod {:>2} new {:>2}/{seeds} | worse {:>2} ({}) better {:>2} ({}) | median ms {:.1} -> {:.1}, total {:.0} -> {:.0}",
                    rows.iter().filter(|r| r.1).count(),
                    rows.iter().filter(|r| r.2).count(),
                    worse.len(),
                    worse
                        .iter()
                        .map(|v| format!("{v:.3}"))
                        .collect::<Vec<_>>()
                        .join(" "),
                    better.len(),
                    better
                        .iter()
                        .map(|v| format!("{v:.3}"))
                        .collect::<Vec<_>>()
                        .join(" "),
                    median(&mut tp),
                    median(&mut tc),
                    sum_tp,
                    sum_tc
                );
            }
        }
    }
}

/// Where the coarse race with Newton finishing departs from production:
/// per start, production's race outcome, the coarse race's, the Newton
/// finish and an L-BFGS finish (full tolerance) from the coarse point.
/// `cargo test --release --lib gamma_coarse_why -- --ignored --nocapture`
#[test]
#[ignore]
fn gamma_coarse_why() {
    let (d, n) = (env_or("D", 3), env_or("N", 300));
    let hard = std::env::var("HARD").is_ok();
    let seeds = env_or("SEEDS", 40) as u64;
    let cfg = EsagRegressor::new().with_parallel(false);
    for seed in 0..seeds {
        let (x, y) = scenario(d, hard, n, 500 + seed);
        let data = cfg.data(&x, &y).unwrap();
        let mask = cfg.resolve_mask(&data).unwrap();
        let free = free_params(&mask);
        let starts: Vec<DMatrix<f64>> = [Init::DataDriven, Init::RStyle]
            .iter()
            .map(|i| {
                cfg.clone()
                    .with_init(i.clone())
                    .initial_coef(&data)
                    .unwrap()
            })
            .collect();
        let prod = cfg.race_all(&data, &starts, &mask, cfg.max_iter);
        let mut c = cfg.clone();
        c.handover = Some(1e-2);
        let coarse = c.race_all(&data, &starts, &mask, cfg.max_iter);
        let pf = |r: &RawFit| format!("{:.3} {:?} it {}", -r.neg_ll, r.status, r.n_iter);
        let pbest = prod
            .iter()
            .map(|r| -r.neg_ll)
            .fold(f64::NEG_INFINITY, f64::max);
        let mut line = format!("seed {seed:>2}: prod best {pbest:.3}");
        let mut new_best = f64::NEG_INFINITY;
        for (k, (p, q)) in prod.iter().zip(&coarse).enumerate() {
            let nt = newton_finish(&cfg, &data, &free, &q.coef, 12, HANDOFF_DECREMENT)
                .map(|(f, h)| (-f.neg_ll, h));
            let lb = cfg.optimize(&data, &q.coef, &free, cfg.max_iter);
            if q.status.is_converged() {
                new_best = new_best.max(nt.map_or(-lb.neg_ll, |v| v.0));
            } else {
                new_best = new_best.max(-q.neg_ll);
            }
            line += &format!(
                " | start {k}: prod [{}] coarse [{}] newton {} lbfgs-finish {:.3}",
                pf(p),
                pf(q),
                nt.map_or("fail".to_string(), |v| format!("{:.3} ({} H)", v.0, v.1)),
                -lb.neg_ll
            );
        }
        if new_best < pbest - 1e-3 {
            println!("{line}  <-- LOSS {:.3}", new_best - pbest);
        }
    }
}

/// The small-shape products of `joint_eval` are bit-identical to
/// nalgebra's (whose column-by-column fallback they replace).
#[test]
fn small_products_match_nalgebra() {
    let mut st = 0x5EED_u64;
    let mut rnd = || {
        st = st
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((st >> 11) as f64 / (1u64 << 53) as f64) * 4.0 - 2.0
    };
    for &bd in &[5usize, 9, 14, 20, 27, 35, 44] {
        for &pt in &[1usize, 2, 4, 5, 6, 8] {
            for &m in &[1usize, 3, 5, 6, 16, 64] {
                let b = DMatrix::from_fn(bd, pt, |_, _| rnd());
                let blk: Vec<f64> = (0..m * pt).map(|_| rnd()).collect();
                let xt = nalgebra::DMatrixView::from_slice(&blk, pt, m);
                let xc = nalgebra::DMatrixView::from_slice_with_strides(&blk, m, pt, pt, 1);
                let mut eta_na = vec![0.0; bd * m];
                nalgebra::DMatrixViewMut::from_slice(&mut eta_na, bd, m).gemm(1.0, &b, &xt, 0.0);
                let mut eta = vec![0.0; bd * m];
                small_eta(&mut eta, b.as_slice(), &blk, bd, pt);
                let ge: Vec<f64> = (0..bd * m).map(|_| rnd()).collect();
                let mut gm_na = DMatrix::<f64>::zeros(bd, pt);
                gm_na.gemm(
                    1.0,
                    &nalgebra::DMatrixView::from_slice(&ge, bd, m),
                    &xc,
                    0.0,
                );
                let mut gm = vec![0.0; bd * pt];
                small_grad(&mut gm, &ge, &blk, bd, pt);
                let small = !(bd > 5 && pt > 5 && m > 5);
                let same =
                    |a: &[f64], b: &[f64]| a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits());
                if small {
                    // where nalgebra multiplies column by column: the same bits
                    assert!(same(&eta, &eta_na), "eta bd={bd} pt={pt} m={m}");
                    assert!(same(&gm, gm_na.as_slice()), "grad bd={bd} pt={pt} m={m}");
                } else {
                    // elsewhere (matrixmultiply) only to rounding
                    for (a, b) in eta
                        .iter()
                        .zip(&eta_na)
                        .chain(gm.iter().zip(gm_na.as_slice()))
                    {
                        assert!((a - b).abs() <= 1e-12 * (1.0 + b.abs()));
                    }
                }
            }
        }
    }
}
