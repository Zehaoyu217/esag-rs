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
