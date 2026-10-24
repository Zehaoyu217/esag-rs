//! Time the Rust implementation on the exact data sets and operations whose R
//! timings are recorded in `tests/fixtures/*.json` (R 4.6.1, same machine).
//! Run: `cargo run --release --example bench_vs_r`

use esag::estimator::EsagEstimator;
use esag::inference::{GofMethod, bootstrap_gof, prediction_region};
use esag::regression::EsagRegressor;
use nalgebra::DMatrix;
use serde_json::Value;
use std::time::Instant;

fn load(name: &str) -> Value {
    let p = format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"));
    serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap()
}
fn mat(v: &Value) -> DMatrix<f64> {
    let rows: Vec<Vec<f64>> = v
        .as_array()
        .unwrap()
        .iter()
        .map(|r| {
            r.as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap())
                .collect()
        })
        .collect();
    DMatrix::from_fn(rows.len(), rows[0].len(), |i, j| rows[i][j])
}
/// median of `reps` timings in seconds
fn time<F: FnMut()>(reps: usize, mut f: F) -> f64 {
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

fn main() {
    println!(
        "{:<52}{:>12}{:>14}{:>10}",
        "task (identical data and settings)", "R 4.6.1", "esag (Rust)", "speed-up"
    );
    let row = |name: &str, r: f64, rust: f64| {
        println!(
            "{name:<52}{:>11.2}s{:>12.2}ms{:>9.0}x",
            r,
            rust * 1e3,
            r / rust
        );
    };
    for (name, file) in [
        (
            "MLE + Hessian, regression d=3, n=200, p=2",
            "regression_d3_n200.json",
        ),
        (
            "MLE + Hessian, regression d=3, n=1000, p=2",
            "regression_d3_n1000.json",
        ),
    ] {
        let fx = load(file);
        let x = mat(&fx["X"]).columns(1, 1).into_owned();
        let y = mat(&fx["Y"]);
        let r = fx["timing"]["mle_elapsed_sec"].as_f64().unwrap();
        let t = time(21, || {
            EsagRegressor::new().fit(&x, &y).unwrap();
        });
        row(name, r, t);
    }
    for (name, file) in [
        ("MLE + Hessian, iid d=3, n=300", "iid_d3_n300.json"),
        ("MLE + Hessian, iid d=4, n=300", "iid_d4_n300.json"),
    ] {
        let fx = load(file);
        let y = mat(&fx["Y"]);
        let r = fx["timing"]["mle_elapsed_sec"].as_f64().unwrap();
        let t = time(21, || {
            EsagEstimator::new().fit(&y).unwrap();
        });
        row(name, r, t);
    }
    let fx = load("regression_d3_n200.json");
    let x = mat(&fx["X"]).columns(1, 1).into_owned();
    let y = mat(&fx["Y"]);
    let fit = EsagRegressor::new().fit(&x, &y).unwrap();
    let g = load("gof_small.json");
    let t = time(5, || {
        bootstrap_gof(&fit, &x, &y, GofMethod::OneSampleChiSquare, 20, 99).unwrap();
    });
    row(
        "bootstrap GoF (R Bootstrap_GoF), 20 refits, n=200",
        g["timing"]["elapsed_sec"].as_f64().unwrap(),
        t,
    );
    let p = load("prediction_region_small.json");
    let x0 = [x[(0, 0)]];
    let levels = [0.95, 0.9, 0.85, 0.8, 0.75, 0.7];
    let t = time(5, || {
        prediction_region(&fit, &x, &y, &x0, 2000, 5, &levels, 5).unwrap();
    });
    row(
        "prediction region, m=2000, 5 refits, n=200",
        p["prediction_region"]["timing"]["elapsed_sec"]
            .as_f64()
            .unwrap(),
        t,
    );
    println!(
        "\nRust uses {} threads (rayon); the per-fit numbers are dominated by single-threaded L-BFGS.",
        rayon::current_num_threads()
    );
}
