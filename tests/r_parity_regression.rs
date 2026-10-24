//! Fit the R fixtures' datasets with the Rust estimator and compare with R's
//! `MLE()` (optim/BFGS with numerical gradients).

use esag::estimator::EsagEstimator;
use esag::regression::{EsagRegressor, EsagRegressorFit, Init, canonicalize_coef};
use nalgebra::DMatrix;
use serde_json::Value;
use std::time::Instant;

fn load(name: &str) -> Value {
    let path = format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"));
    serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap()
}
fn vecf(v: &Value) -> Vec<f64> {
    match v {
        Value::Array(a) => a.iter().map(|x| x.as_f64().unwrap()).collect(),
        Value::Number(n) => vec![n.as_f64().unwrap()],
        _ => panic!(),
    }
}
/// Max |difference| of the fitted (mu_i, V_i) over all observations.
fn max_dist_diff(a: &EsagRegressorFit, b: &EsagRegressorFit, x: &DMatrix<f64>) -> f64 {
    let da = a.predict_distribution(x).unwrap();
    let db = b.predict_distribution(x).unwrap();
    let mut m: f64 = 0.0;
    for (p, q) in da.iter().zip(&db) {
        for (u, v) in p.mu().iter().zip(q.mu()) {
            m = m.max((u - v).abs());
        }
        m = m.max((p.covariance() - q.covariance()).abs().max());
    }
    m
}

fn mat(v: &Value) -> DMatrix<f64> {
    let rows: Vec<Vec<f64>> = v.as_array().unwrap().iter().map(vecf).collect();
    let (r, c) = (rows.len(), rows[0].len());
    DMatrix::from_fn(r, c, |i, j| rows[i][j])
}

#[test]
fn regression_fits_match_r() {
    for name in ["regression_d3_n200.json", "regression_d3_n1000.json"] {
        let fx = load(name);
        let xfull = mat(&fx["X"]);
        let x = xfull.columns(1, 1).into_owned(); // drop R's intercept column
        let y = mat(&fx["Y"]);
        let mut r_par = mat(&fx["mle"]["par_matrix"]);
        let r_value = fx["mle"]["value"].as_f64().unwrap();
        let r_secs = fx["timing"]["mle_elapsed_sec"].as_f64().unwrap();
        let r_se = vecf(&fx["mle"]["se_from_hessian_colmajor"]);
        let r_fit = EsagRegressor::new()
            .with_max_iter(0)
            .with_hessian(false)
            .fit_warm(&x, &y, &r_par)
            .unwrap();
        canonicalize_coef(&mut r_par, 3);

        for init in [Init::DataDriven, Init::RStyle] {
            let t0 = Instant::now();
            let fit = EsagRegressor::new()
                .with_init(init.clone())
                .fit(&x, &y)
                .unwrap();
            let secs = t0.elapsed().as_secs_f64();
            let dmax = (&fit.coef - &r_par).abs().max();
            let ddist = max_dist_diff(&fit, &r_fit, &x);
            eprintln!(
                "{name} init={init:?}: rust ll={:.10} R ll={r_value:.10} diff={:+.2e} | max|B_rust-B_R|={dmax:.2e} max|(mu_i,V_i) diff|={ddist:.2e} | {:?} iters={} fev={} |g|={:.1e} | rust {:.3}s vs R {r_secs:.1}s ({:.0}x)",
                fit.log_likelihood,
                fit.log_likelihood - r_value,
                fit.status,
                fit.n_iter,
                fit.n_fev,
                fit.grad_norm,
                secs,
                r_secs / secs
            );
            assert!(fit.converged, "{name}: {:?}", fit.status);
            assert!(
                fit.log_likelihood >= r_value - 1e-6,
                "{name}: rust optimum below R's"
            );
            assert!(dmax < 5e-3, "{name}: coefficients differ by {dmax}");
            assert!(
                ddist < 5e-3,
                "{name}: fitted distributions differ by {ddist}"
            );
            // standard errors: R's come from a doubly-numerical Hessian (h = 1e-3)
            let se = fit.std_errors().unwrap();
            let se_cm: Vec<f64> = (0..2)
                .flat_map(|j| (0..5).map(move |k| (k, j)))
                .map(|(k, j)| se[(k, j)])
                .collect();
            for (i, (a, b)) in se_cm.iter().zip(&r_se).enumerate() {
                assert!(
                    (a - b).abs() / b < 2e-2,
                    "{name}: se[{i}] rust {a} vs R {b}"
                );
            }
        }
    }
}

#[test]
fn residuals_at_r_mle_match_r() {
    let fx = load("regression_d3_n200.json");
    let xfull = mat(&fx["X"]);
    let x = xfull.columns(1, 1).into_owned();
    let y = mat(&fx["Y"]);
    let r_par = mat(&fx["mle"]["par_matrix"]);
    // evaluate at R's estimate: a zero-iteration "fit" warm-started at R's par
    let fit = EsagRegressor::new()
        .with_max_iter(0)
        .with_hessian(false)
        .fit_warm(&x, &y, &r_par)
        .unwrap();
    assert_eq!(fit.coef, r_par);
    let res = fit.residuals(&x, &y).unwrap();
    let want = vecf(&fx["T1_residuals"]);
    for (i, (a, b)) in res.t1.iter().zip(&want).enumerate() {
        assert!(
            (a - b).abs() <= 1e-10 * b.abs().max(1.0),
            "T1[{i}]: {a} vs {b}"
        );
    }
    assert!((fit.log_likelihood - fx["mle"]["value"].as_f64().unwrap()).abs() < 1e-9);
}

#[test]
fn iid_fits_match_r() {
    for name in ["iid_d3_n300.json", "iid_d4_n300.json"] {
        let fx = load(name);
        let y = mat(&fx["Y"]);
        let mut r_par = vecf(&fx["mle"]["par"]);
        let d = y.ncols();
        if r_par[d] < 0.0 || (r_par[d] == 0.0 && r_par[d + 1] < 0.0) {
            r_par[d] = -r_par[d];
            r_par[d + 1] = -r_par[d + 1];
        }
        let r_dist = esag::distribution::Esag::from_eta(&r_par).unwrap();
        let r_value = fx["mle"]["value"].as_f64().unwrap();
        let r_secs = fx["timing"]["mle_elapsed_sec"].as_f64().unwrap();
        let t0 = Instant::now();
        let fit = EsagEstimator::new().fit(&y).unwrap();
        let secs = t0.elapsed().as_secs_f64();
        let eta: Vec<f64> = fit.mu().iter().chain(fit.gamma()).copied().collect();
        let dmax = eta
            .iter()
            .zip(&r_par)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        eprintln!(
            "{name}: rust ll={:.10} R ll={r_value:.10} diff={:+.2e} | max|eta diff|={dmax:.2e} | rust {:.3}s vs R {r_secs:.1}s ({:.0}x)",
            fit.log_likelihood(),
            fit.log_likelihood() - r_value,
            secs,
            r_secs / secs
        );
        assert!(fit.converged());
        assert!(fit.log_likelihood() >= r_value - 1e-6);
        // gamma itself is only identified up to an equivalence class for
        // d >= 4 (eigenvector sign patterns), so compare mu and V.
        let dmu = fit
            .mu()
            .iter()
            .zip(r_dist.mu())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        let dv = (fit.distribution().covariance() - r_dist.covariance())
            .abs()
            .max();
        eprintln!(
            "{name}: max|mu diff| = {dmu:.2e}, max|V diff| = {dv:.2e} (raw gamma diff {dmax:.2e})"
        );
        assert!(dmu < 5e-3, "{name}: mu differs by {dmu}");
        assert!(dv < 5e-3, "{name}: V differs by {dv}");
    }
}
