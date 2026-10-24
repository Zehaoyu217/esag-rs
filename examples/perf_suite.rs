//! Performance suite: per-observation kernel, single fits, bootstrap
//! procedures. Prints median wall times. `cargo run --release --example perf_suite`
//! Pass `--quick` to skip the slowest workloads.

use esag::distribution::Esag;
use esag::estimator::EsagEstimator;
use esag::inference::{GofMethod, Restriction, bootstrap_gof, hypothesis_test, prediction_region};
use esag::loglik::{loglik_eta, loglik_eta_grad};
use esag::regression::EsagRegressor;
use nalgebra::DMatrix;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal};
use std::hint::black_box;
use std::time::Instant;

fn median<F: FnMut()>(reps: usize, mut f: F) -> f64 {
    let mut t: Vec<f64> = (0..reps)
        .map(|_| {
            let s = Instant::now();
            f();
            s.elapsed().as_secs_f64()
        })
        .collect();
    t.sort_by(|a, b| a.partial_cmp(b).unwrap());
    t[reps / 2]
}

fn sim_reg(d: usize, n: usize, seed: u64) -> (DMatrix<f64>, DMatrix<f64>) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let g = esag::dims::gamma_len(d);
    let b0: Vec<f64> = (0..d + g)
        .map(|k| {
            if k < d {
                1.5 - 0.4 * k as f64
            } else {
                0.6 * ((k as f64) * 1.3).sin()
            }
        })
        .collect();
    let b1: Vec<f64> = (0..d + g).map(|k| 0.5 * ((k as f64) * 0.7).cos()).collect();
    let x = DMatrix::from_fn(n, 1, |_, _| StandardNormal.sample(&mut rng));
    let mut y = DMatrix::zeros(n, d);
    let mut buf = vec![0.0; d];
    for i in 0..n {
        let eta: Vec<f64> = (0..d + g).map(|k| b0[k] + b1[k] * x[(i, 0)]).collect();
        let dist = Esag::from_eta(&eta).unwrap();
        dist.sample_one(&dist.frame(), &mut rng, &mut buf);
        for k in 0..d {
            y[(i, k)] = buf[k];
        }
    }
    (x, y)
}

fn hydro() -> (DMatrix<f64>, DMatrix<f64>) {
    let mut rdr = csv::Reader::from_path(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/data/hydrochem_At_LLt.csv"
    ))
    .unwrap();
    let rows: Vec<csv::StringRecord> = rdr.records().map(|r| r.unwrap()).collect();
    let n = rows.len();
    (
        DMatrix::from_fn(n, 1, |i, _| rows[i][1].parse::<f64>().unwrap()),
        DMatrix::from_fn(n, 4, |i, j| rows[i][6 + j].parse::<f64>().unwrap()),
    )
}

fn main() {
    let quick = std::env::args().any(|a| a == "--quick");
    println!("{:<58}{:>14}", "workload", "median");
    let line = |name: &str, secs: f64| {
        let (v, u) = if secs < 1e-6 {
            (secs * 1e9, "ns")
        } else if secs < 1e-3 {
            (secs * 1e6, "µs")
        } else if secs < 1.0 {
            (secs * 1e3, "ms")
        } else {
            (secs, "s ")
        };
        println!("{name:<58}{v:>11.2} {u}");
    };

    // ---- kernel
    for d in [3usize, 4, 5, 6] {
        let b = esag::dims::eta_len(d);
        let etas: Vec<Vec<f64>> = (0..64)
            .map(|s| {
                (0..b)
                    .map(|i| 0.3 + ((i * 7 + s) as f64 * 0.77).sin())
                    .collect()
            })
            .collect();
        let ys: Vec<Vec<f64>> = (0..64)
            .map(|s| {
                let r: Vec<f64> = (0..d).map(|i| ((i * 3 + s) as f64 * 1.9).cos()).collect();
                let n = r.iter().map(|v| v * v).sum::<f64>().sqrt();
                r.iter().map(|v| v / n).collect()
            })
            .collect();
        let mut g = vec![0.0; b];
        let reps = 20_000;
        let t = median(5, || {
            let mut acc = 0.0;
            for r in 0..reps {
                acc += loglik_eta::<f64>(black_box(&etas[r % 64]), black_box(&ys[r % 64]));
            }
            black_box(acc);
        }) / reps as f64;
        line(&format!("kernel loglik d={d}"), t);
        let t = median(5, || {
            let mut acc = 0.0;
            for r in 0..reps {
                acc += loglik_eta_grad(black_box(&etas[r % 64]), black_box(&ys[r % 64]), &mut g);
            }
            black_box(acc);
        }) / reps as f64;
        line(&format!("kernel loglik+grad d={d}"), t);
    }

    // ---- single fits
    for (d, n) in [(3usize, 1000usize), (3, 10_000), (4, 2000), (5, 2000)] {
        let (x, y) = sim_reg(d, n, 1);
        let t = median(7, || {
            black_box(EsagRegressor::new().fit(&x, &y).unwrap());
        });
        line(&format!("fit+SE regression d={d} n={n} p=1 (default)"), t);
        let t = median(7, || {
            black_box(
                EsagRegressor::new()
                    .with_hessian(false)
                    .fit(&x, &y)
                    .unwrap(),
            );
        });
        line(&format!("fit     regression d={d} n={n} p=1 (no SE)"), t);
    }
    let (hx, hy) = hydro();
    let t = median(7, || {
        black_box(EsagRegressor::new().fit(&hx, &hy).unwrap());
    });
    line("fit+SE hydrochem d=4 n=110 (default)", t);
    let fx: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/iid_d4_n300.json"
        ))
        .unwrap(),
    )
    .unwrap();
    let rows: Vec<Vec<f64>> = fx["Y"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| {
            r.as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap())
                .collect()
        })
        .collect();
    let y4 = DMatrix::from_fn(rows.len(), 4, |i, j| rows[i][j]);
    let t = median(7, || {
        black_box(EsagEstimator::new().fit(&y4).unwrap());
    });
    line("fit+SE iid d=4 n=300 (R fixture)", t);

    // ---- bootstrap procedures
    let (x, y) = sim_reg(3, 1000, 2);
    let fit = EsagRegressor::new().fit(&x, &y).unwrap();
    let t = median(3, || {
        black_box(bootstrap_gof(&fit, &x, &y, GofMethod::TwoSampleReference, 200, 1).unwrap());
    });
    line("bootstrap GoF d=3 n=1000, 200 replicates", t);
    let t = median(3, || {
        black_box(prediction_region(&fit, &x, &y, &[0.5], 5000, 100, &[0.95, 0.9], 3).unwrap());
    });
    line("prediction region d=3 n=1000, m=5000, 100 refits", t);
    if !quick {
        let t = median(3, || {
            black_box(hypothesis_test(&fit, &x, &y, Restriction::Isotropy, 200, 1000, 4).unwrap());
        });
        line("hypothesis test isotropy d=3 n=1000, 200 reps, mc=1000", t);
        let hfit = EsagRegressor::new().fit(&hx, &hy).unwrap();
        let t = median(3, || {
            black_box(
                hypothesis_test(
                    &hfit,
                    &hx,
                    &hy,
                    Restriction::GammaOfFeature(0),
                    300,
                    2000,
                    4,
                )
                .unwrap(),
            );
        });
        line("hypothesis test hydrochem d=4, 300 reps, mc=2000", t);
    }
    println!("threads: {}", rayon::current_num_threads());
}
