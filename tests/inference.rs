#![allow(clippy::needless_range_loop)]
//! Inference procedures: parity with the R code where deterministic, Monte
//! Carlo agreement where stochastic, and reproducibility.

use esag::distribution::Esag;
use esag::inference::{
    GofMethod, Restriction, bootstrap_gof, confidence_region_sample, hypothesis_test,
    prediction_region,
};
use esag::regression::EsagRegressor;
use esag::stats::ks::ks_1samp_chisq;
use esag::stats::quantile::quantiles_type7;
use nalgebra::DMatrix;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde_json::Value;

fn load(name: &str) -> Value {
    let path = format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"));
    serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap()
}
fn vecf(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect()
}
fn mat(v: &Value) -> DMatrix<f64> {
    let rows: Vec<Vec<f64>> = v.as_array().unwrap().iter().map(vecf).collect();
    DMatrix::from_fn(rows.len(), rows[0].len(), |i, j| rows[i][j])
}
fn regression_fixture() -> (DMatrix<f64>, DMatrix<f64>, DMatrix<f64>) {
    let fx = load("regression_d3_n200.json");
    let x = mat(&fx["X"]).columns(1, 1).into_owned();
    (x, mat(&fx["Y"]), mat(&fx["mle"]["par_matrix"]))
}

#[test]
fn one_sample_ks_of_residuals_matches_r() {
    // R: ks.test(residual_T(par, Y, X), "pchisq", 2) at R's MLE
    let (x, y, par) = regression_fixture();
    let fit = EsagRegressor::new()
        .with_max_iter(0)
        .with_hessian(false)
        .fit_warm(&x, &y, &par)
        .unwrap();
    let t1 = fit.residuals(&x, &y).unwrap().t1;
    let ks = ks_1samp_chisq(&t1, 2.0);
    let fx = load("gof_small.json");
    let want_d = fx["observed_ks"]["statistic"].as_f64().unwrap();
    let want_p = fx["observed_ks"]["p_value"].as_f64().unwrap();
    assert!(
        (ks.statistic - want_d).abs() < 1e-12,
        "{} vs {want_d}",
        ks.statistic
    );
    assert!(
        (ks.p_value - want_p).abs() < 1e-8,
        "{} vs {want_p}",
        ks.p_value
    );
}

#[test]
fn confidence_region_quantiles_match_r_monte_carlo() {
    // R: Confidence_Region(c(1,2,1.5), c(0.7,-0.3), 5000) quantiles; R's own
    // Monte-Carlo error at m = 5000 is a few percent.
    let fx = load("prediction_region_small.json");
    let cr = &fx["confidence_region"];
    let dist = Esag::new(&[1.0, 2.0, 1.5], &[0.7, -0.3]).unwrap();
    let vinv = dist.covariance_inv();
    let want_v = mat(&cr["V_inv_used"]);
    assert!((vinv - want_v).abs().max() < 1e-12);
    let mut rng = ChaCha8Rng::seed_from_u64(3);
    let q = confidence_region_sample(&dist, 400_000, &mut rng);
    let got = quantiles_type7(&q, &[0.5, 0.9, 0.95]);
    for (g, key) in got.iter().zip(["0.5", "0.9", "0.95"]) {
        let w = cr["quantiles"][key].as_f64().unwrap();
        assert!((g - w).abs() / w < 0.06, "q{key}: rust {g} vs R {w}");
    }
}

#[test]
fn prediction_region_matches_r_monte_carlo() {
    // R: Prediction_Region(index = 1, num_boot_m = 2000, num_boot = 5)
    let (x, y, par) = regression_fixture();
    let fit = EsagRegressor::new()
        .with_max_iter(0)
        .with_hessian(false)
        .fit_warm(&x, &y, &par)
        .unwrap();
    let fx = load("prediction_region_small.json");
    let levels = [0.95, 0.90, 0.85, 0.80, 0.75, 0.70];
    let keys = ["95% PR", "90% PR", "85% PR", "80% PR", "75% PR", "70% PR"];
    let x0 = [x[(0, 0)]];
    let pr = prediction_region(&fit, &x, &y, &x0, 20_000, 30, &levels, 5).unwrap();
    for (g, k) in pr.thresholds.iter().zip(keys) {
        let w = fx["prediction_region"]["quantiles"][k].as_f64().unwrap();
        assert!((g - w).abs() / w < 0.08, "{k}: rust {g} vs R {w}");
    }
}

#[test]
fn bootstrap_results_are_reproducible_and_thread_independent() {
    let (x, y, _) = regression_fixture();
    let fit = EsagRegressor::new().fit(&x, &y).unwrap();
    let a = bootstrap_gof(&fit, &x, &y, GofMethod::TwoSampleReference, 16, 9).unwrap();
    let b = bootstrap_gof(&fit, &x, &y, GofMethod::TwoSampleReference, 16, 9).unwrap();
    assert_eq!(a.boot_ks_p_values, b.boot_ks_p_values);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let c =
        pool.install(|| bootstrap_gof(&fit, &x, &y, GofMethod::TwoSampleReference, 16, 9).unwrap());
    assert_eq!(a.boot_ks_p_values, c.boot_ks_p_values);
    assert!((0.0..=1.0).contains(&a.p_value) && a.n_boot == 16);
    let r1 = hypothesis_test(&fit, &x, &y, Restriction::Isotropy, 8, 200, 4).unwrap();
    let r2 =
        pool.install(|| hypothesis_test(&fit, &x, &y, Restriction::Isotropy, 8, 200, 4).unwrap());
    assert_eq!(r1.boot_lr, r2.boot_lr);
    assert!(r1.lr > 0.0 && r1.df == 4);
}

#[test]
fn restricted_fits_are_nested() {
    let (x, y, _) = regression_fixture();
    let fit = EsagRegressor::new().fit(&x, &y).unwrap();
    for r in [
        Restriction::Isotropy,
        Restriction::MuOfFeature(0),
        Restriction::GammaOfFeature(0),
        Restriction::Feature(0),
    ] {
        let t = hypothesis_test(&fit, &x, &y, r.clone(), 2, 50, 1).unwrap();
        assert!(t.loglik_restricted <= t.loglik_full + 1e-6, "{r:?}");
        assert!(t.lr >= -1e-6, "{r:?}: LR {}", t.lr);
    }
    assert!(hypothesis_test(&fit, &x, &y, Restriction::Feature(3), 2, 50, 1).is_err());
}

/// Size of the goodness-of-fit test under the null (slow; run with
/// `cargo test --release -- --ignored gof_size`).
#[test]
#[ignore]
fn gof_size_under_the_null() {
    let truth = Esag::new(&[2.0, -1.0, 1.5], &[0.8, -0.5]).unwrap();
    let reps = 200;
    let mut rej = [0usize; 2];
    for r in 0..reps {
        let mut rng = ChaCha8Rng::seed_from_u64(1000 + r);
        let y = truth.sample(200, &mut rng);
        let x = DMatrix::zeros(200, 0);
        let fit = EsagRegressor::new()
            .with_hessian(false)
            .fit(&x, &y)
            .unwrap();
        for (k, m) in [GofMethod::TwoSampleReference, GofMethod::OneSampleChiSquare]
            .into_iter()
            .enumerate()
        {
            let g = bootstrap_gof(&fit, &x, &y, m, 100, r).unwrap();
            rej[k] += usize::from(g.p_value < 0.05);
        }
    }
    let size: Vec<f64> = rej.iter().map(|&c| c as f64 / reps as f64).collect();
    eprintln!(
        "GoF rejection rate at 5% under the null ({reps} reps): two-sample {:.3}, one-sample {:.3}",
        size[0], size[1]
    );
    assert!(size.iter().all(|&s| s < 0.12));
}

/// Size of the bootstrap tests for a true null (slow; `--ignored test_size`).
#[test]
#[ignore]
fn test_size_under_a_true_null() {
    use rand_distr::{Distribution, StandardNormal};
    let reps = 300;
    let mut rej = [0usize; 4];
    for r in 0..reps {
        let mut rng = ChaCha8Rng::seed_from_u64(5000 + r);
        let n = 200;
        let x = DMatrix::from_fn(n, 1, |_, _| StandardNormal.sample(&mut rng));
        let mut y = DMatrix::zeros(n, 3);
        let mut buf = [0.0; 3];
        for i in 0..n {
            // mu depends on x, gamma does not  ->  H0: gamma free of x is true
            let d = Esag::new(
                &[1.5 - 0.6 * x[(i, 0)], 0.5 + 1.2 * x[(i, 0)], 1.0],
                &[0.8, -0.4],
            )
            .unwrap();
            d.sample_one(&d.frame(), &mut rng, &mut buf);
            for k in 0..3 {
                y[(i, k)] = buf[k];
            }
        }
        let fit = EsagRegressor::new()
            .with_hessian(false)
            .fit(&x, &y)
            .unwrap();
        let t = hypothesis_test(&fit, &x, &y, Restriction::GammaOfFeature(0), 100, 300, r).unwrap();
        for (k, p) in [t.p_roc, t.p_d, t.p_m, t.p_lr].into_iter().enumerate() {
            rej[k] += usize::from(p < 0.05);
        }
    }
    let size: Vec<f64> = rej.iter().map(|&c| c as f64 / reps as f64).collect();
    eprintln!(
        "rejection at 5% for a true null ({reps} reps): RoC {:.2}, D {:.2}, M {:.2}, LR {:.2}",
        size[0], size[1], size[2], size[3]
    );
    assert!(size.iter().all(|&s| s < 0.10));
}
