#![allow(clippy::needless_range_loop)]
//! Behaviour of the scikit-learn-style API on simulated data.

use esag::distribution::Esag;
use esag::estimator::{EsagEstimator, Fit, Predict, Score};
use esag::preprocessing::MinMaxScaler;
use esag::regression::{EsagRegressor, EsagRegressorFit, isotropic_mask};
use nalgebra::DMatrix;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal};

/// Simulate from a d = 3 ESAG regression with one covariate.
fn simulate(n: usize, b: &DMatrix<f64>, seed: u64) -> (DMatrix<f64>, DMatrix<f64>) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let d = 3;
    let x = DMatrix::from_fn(n, 1, |_, _| {
        let z: f64 = StandardNormal.sample(&mut rng);
        z
    });
    let mut y = DMatrix::zeros(n, d);
    let mut buf = vec![0.0; d];
    for i in 0..n {
        let eta: Vec<f64> = (0..5).map(|k| b[(k, 0)] + b[(k, 1)] * x[(i, 0)]).collect();
        let dist = Esag::from_eta(&eta).unwrap();
        dist.sample_one(&dist.frame(), &mut rng, &mut buf);
        for k in 0..d {
            y[(i, k)] = buf[k];
        }
    }
    (x, y)
}

fn b_true() -> DMatrix<f64> {
    DMatrix::from_column_slice(5, 2, &[1.5, 0.5, 1.0, 0.8, -0.4, -0.6, 1.2, 0.3, 0.5, 0.7])
}

#[test]
fn recovers_true_coefficients_with_honest_standard_errors() {
    let (x, y) = simulate(5000, &b_true(), 7);
    let fit = EsagRegressor::new().fit(&x, &y).unwrap();
    assert!(fit.converged, "{:?}", fit.status);
    let se = fit.std_errors().unwrap();
    let mut max_z: f64 = 0.0;
    for k in 0..5 {
        for j in 0..2 {
            let z = (fit.coef[(k, j)] - b_true()[(k, j)]) / se[(k, j)];
            max_z = max_z.max(z.abs());
        }
    }
    // 10 coefficients; |z| < 4 for all of them is a very safe bound
    assert!(max_z < 4.0, "max |z| = {max_z}\n{}", fit.summary());
    // the canonical sign convention picks the representative with gamma_1 intercept > 0
    assert!(fit.coef[(3, 0)] > 0.0);
}

#[test]
fn sklearn_traits_predict_score_residuals() {
    let (x, y) = simulate(800, &b_true(), 11);
    let (xt, yt) = simulate(400, &b_true(), 12);
    let reg = EsagRegressor::new();
    let fit: EsagRegressorFit = Fit::fit(&reg, &x, &y).unwrap();
    let yhat = Predict::predict(&fit, &xt).unwrap();
    assert_eq!(yhat.shape(), (400, 3));
    for i in 0..400 {
        let n: f64 = (0..3).map(|k| yhat[(i, k)].powi(2)).sum::<f64>().sqrt();
        assert!((n - 1.0).abs() < 1e-12);
    }
    let s_train = Score::score(&fit, &x, &y).unwrap();
    let s_test = Score::score(&fit, &xt, &yt).unwrap();
    assert!((s_train - fit.log_likelihood / 800.0).abs() < 1e-9);
    assert!(
        s_test < s_train + 0.2 && s_test > s_train - 0.3,
        "{s_train} {s_test}"
    );
    // residuals: T1 is roughly chi^2_2 (mean 2)
    let r = fit.residuals(&x, &y).unwrap();
    let m = r.t1.iter().sum::<f64>() / r.t1.len() as f64;
    assert!(m > 1.5 && m < 2.6, "mean T1 = {m}");
    // the true model scores better on test data than a covariate-free fit
    let fit0 = EsagRegressor::new()
        .fit(&DMatrix::zeros(800, 0), &y)
        .unwrap();
    assert!(fit0.score(&DMatrix::zeros(400, 0), &yt).unwrap() < s_test);
}

#[test]
fn isotropic_mask_fixes_gamma_and_lowers_likelihood() {
    let (x, y) = simulate(600, &b_true(), 3);
    let full = EsagRegressor::new().fit(&x, &y).unwrap();
    let iso = EsagRegressor::new()
        .with_mask(Some(isotropic_mask(3, 2)))
        .fit(&x, &y)
        .unwrap();
    for j in 0..2 {
        assert_eq!(iso.coef[(3, j)], 0.0);
        assert_eq!(iso.coef[(4, j)], 0.0);
    }
    assert_eq!(iso.n_params(), 6);
    assert!(iso.log_likelihood < full.log_likelihood);
    assert!(iso.summary().contains("(fixed)"));
}

#[test]
fn iid_estimator_recovers_distribution() {
    let truth = Esag::new(&[1.0, 2.0, 1.5], &[0.7, -0.3]).unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(5);
    let y = truth.sample(20_000, &mut rng);
    let fit = EsagEstimator::new().fit(&y).unwrap();
    let dv = (fit.distribution().covariance() - truth.covariance())
        .abs()
        .max();
    let dmu = fit
        .mu()
        .iter()
        .zip(truth.mu())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    assert!(dv < 0.08 && dmu < 0.08, "dV = {dv}, dmu = {dmu}");
    assert!(fit.std_errors().is_some());
}

#[test]
fn json_roundtrip_and_errors() {
    let (x, y) = simulate(300, &b_true(), 9);
    let fit = EsagRegressor::new()
        .with_feature_names(vec!["dose".into()])
        .fit(&x, &y)
        .unwrap();
    let back = EsagRegressorFit::from_json(&fit.to_json().unwrap()).unwrap();
    assert_eq!(back.coef, fit.coef);
    assert!(back.summary().contains("[dose]"));
    // shape errors
    assert!(
        EsagRegressor::new()
            .fit(&DMatrix::zeros(10, 1), &y)
            .is_err()
    );
    assert!(
        EsagRegressor::new()
            .fit(&x, &DMatrix::zeros(300, 2))
            .is_err()
    );
    assert!(fit.predict(&DMatrix::zeros(3, 2)).is_err());
    let mut bad = y.clone();
    bad[(0, 0)] = f64::NAN;
    assert!(EsagRegressor::new().fit(&x, &bad).is_err());
}

#[test]
fn minmax_scaling_is_an_equivalent_reparameterisation() {
    // A linear model in x is also linear in the [1, 2]-scaled covariate, so
    // the maximised likelihood must be identical.
    let (x, y) = simulate(500, &b_true(), 21);
    let (_, xs) = MinMaxScaler::new().fit_transform(&x).unwrap();
    let a = EsagRegressor::new().fit(&x, &y).unwrap();
    let b = EsagRegressor::new().fit(&xs, &y).unwrap();
    assert!(
        (a.log_likelihood - b.log_likelihood).abs() < 1e-5,
        "{} vs {}",
        a.log_likelihood,
        b.log_likelihood
    );
}

#[test]
fn higher_dimensions_fit() {
    for d in [4usize, 5] {
        let g = esag::dims::gamma_len(d);
        let mu: Vec<f64> = (0..d).map(|i| [2.0, -5.0, 3.0, 5.0, 1.0][i]).collect();
        let gamma: Vec<f64> = (0..g).map(|i| 0.5 * (i as f64 + 1.0).sin()).collect();
        let truth = Esag::new(&mu, &gamma).unwrap();
        let mut rng = ChaCha8Rng::seed_from_u64(d as u64);
        let y = truth.sample(4000, &mut rng);
        let fit = EsagEstimator::new().fit(&y).unwrap();
        assert!(fit.converged(), "d={d}");
        let dv = (fit.distribution().covariance() - truth.covariance())
            .abs()
            .max();
        assert!(dv < 0.2, "d={d}: dV = {dv}");
    }
}
