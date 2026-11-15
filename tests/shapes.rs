//! The log-tangent shape (`Shape::LogTangent`) next to Yu & Huang's gamma.

use esag::distribution::Esag;
use esag::estimator::EsagEstimator;
use esag::inference::{Restriction, hypothesis_test};
use esag::regression::{EsagRegressor, EsagRegressorFit, Shape};
use nalgebra::DMatrix;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

#[test]
fn iid_fits_agree_between_shapes() {
    // same family without covariates: same maximum, same distribution
    let mut rng = ChaCha8Rng::seed_from_u64(5);
    for (mu, gamma) in [
        (vec![2.0, -1.0, 1.5, 0.5], vec![0.8, -0.3, 1.2, 0.4, -0.6]),
        (
            vec![2.0, -5.0, 3.0, -4.0, 5.0, 6.0],
            vec![
                5.0, 2.0, 3.0, 0.1, 5.0, 1.0, 0.2, 1.0, 5.0, 0.3, 3.0, 2.0, 0.4, 0.5,
            ],
        ),
    ] {
        let y = Esag::new(&mu, &gamma).unwrap().sample(1500, &mut rng);
        let fg = EsagEstimator::new().fit(&y).unwrap();
        let fl = EsagEstimator::new()
            .with_shape(Shape::LogTangent)
            .fit(&y)
            .unwrap();
        assert!(fg.converged() && fl.converged());
        assert!(
            (fg.log_likelihood() - fl.log_likelihood()).abs() < 1e-6,
            "{} vs {}",
            fg.log_likelihood(),
            fl.log_likelihood()
        );
        let dv = (fg.distribution().covariance() - fl.distribution().covariance())
            .abs()
            .max();
        assert!(dv < 1e-6, "V differs by {dv}");
        // the log-tangent likelihood is smooth: far fewer iterations
        assert!(fl.as_regression().n_iter <= fg.as_regression().n_iter);
    }
}

fn lt_regression() -> (DMatrix<f64>, DMatrix<f64>, EsagRegressorFit) {
    let mut rng = ChaCha8Rng::seed_from_u64(9);
    let n = 400;
    let x = DMatrix::from_fn(n, 1, |i, _| ((i * 37 % 101) as f64 / 50.0) - 1.0);
    let lt = esag::logtangent::LtFrame::standard(4);
    let b = DMatrix::from_row_slice(
        9,
        2,
        &[
            1.5, 0.4, -0.5, 0.8, 1.0, -0.2, 2.0, 0.3, 0.6, 0.3, -0.4, 0.2, 0.3, -0.5, 0.1, 0.2,
            0.5, 0.1,
        ],
    );
    let mut y = DMatrix::zeros(n, 4);
    let mut buf = [0.0; 4];
    for i in 0..n {
        let eta: Vec<f64> = (0..9).map(|k| b[(k, 0)] + b[(k, 1)] * x[(i, 0)]).collect();
        let v = esag::logtangent::covariance(&lt, &eta[..4], &eta[4..]);
        let dist = Esag::from_covariance(&eta[..4], &v).unwrap();
        dist.sample_one(&dist.frame(), &mut rng, &mut buf);
        for k in 0..4 {
            y[(i, k)] = buf[k];
        }
    }
    let fit = EsagRegressor::new()
        .with_shape(Shape::LogTangent)
        .fit(&x, &y)
        .unwrap();
    (x, y, fit)
}

#[test]
fn log_tangent_regression_end_to_end() {
    let (x, y, fit) = lt_regression();
    assert!(fit.converged);
    assert!(fit.std_errors().is_some());
    assert_eq!(fit.row_names()[4], "s[1]");
    // the frame used is recorded (the data's mean direction)
    let m = fit.config.lt_reference.clone().expect("reference recorded");
    assert!((m.iter().map(|v| v * v).sum::<f64>() - 1.0).abs() < 1e-12);
    // distributions reproduce the per-row likelihood
    let dists = fit.predict_distribution(&x).unwrap();
    let rows = fit.log_likelihood_rows(&x, &y).unwrap();
    for i in [0, 17, 233] {
        let yi: Vec<f64> = (0..4).map(|k| y[(i, k)]).collect();
        assert!((dists[i].log_pdf(&yi) - rows[i]).abs() < 1e-9);
    }
    assert!((rows.sum() - fit.log_likelihood).abs() < 1e-6 * fit.log_likelihood.abs());
    let res = fit.residuals(&x, &y).unwrap();
    assert!(res.t1.iter().all(|v| v.is_finite() && *v >= 0.0));
    // JSON round trip keeps the shape and the frame
    let back = EsagRegressorFit::from_json(&fit.to_json().unwrap()).unwrap();
    assert_eq!(back.config.shape, Shape::LogTangent);
    let rows2 = back.log_likelihood_rows(&x, &y).unwrap();
    assert!((rows2 - rows).abs().max() < 1e-12);
}

#[test]
fn log_tangent_isotropy_test_rejects() {
    let (x, y, fit) = lt_regression();
    let t = hypothesis_test(&fit, &x, &y, Restriction::Isotropy, 40, 0, 3).unwrap();
    assert!(t.lr > 0.0 && t.p_lr < 0.05 && t.n_failed == 0, "{t:?}");
}
