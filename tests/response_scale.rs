//! Standardized responses (`EsagModel::response_scale`): ESAG for
//! `unit(y / s)`, i.e. `PN(D μ, D V D)` on the original sphere.
use esag::model::{EsagModel, ResponseScale, tyler_scales};
use esag::regression::Shape;
use nalgebra::DMatrix;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal};

fn unit_rows(m: &DMatrix<f64>) -> DMatrix<f64> {
    let mut u = m.clone();
    for i in 0..u.nrows() {
        let r = u.row(i).norm();
        for k in 0..u.ncols() {
            u[(i, k)] /= r;
        }
    }
    u
}

fn mean_cosine(a: &DMatrix<f64>, b: &DMatrix<f64>) -> f64 {
    (0..a.nrows()).map(|i| a.row(i).dot(&b.row(i))).sum::<f64>() / a.nrows() as f64
}

/// A VAR(1) `z_t = A z_{t−1} + e_t` with isotropic noise, observed in
/// coordinates of very different scales, `f_t = D z_t`: directions
/// `unit(f_t)` on lags `f_{t−1}`.
fn unequal_scales(n: usize, seed: u64) -> (DMatrix<f64>, DMatrix<f64>, Vec<f64>) {
    let d = 5;
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let g = DMatrix::from_fn(d, d, |_, _| StandardNormal.sample(&mut rng));
    let a = g.qr().q() * 0.7;
    let scales = vec![8.0, 3.0, 1.5, 0.8, 0.4];
    let mut z = vec![0.0; d];
    let mut f = DMatrix::zeros(n + 1, d);
    for t in 0..=n + 200 {
        let e: Vec<f64> = (0..d).map(|_| StandardNormal.sample(&mut rng)).collect();
        let next: Vec<f64> = (0..d)
            .map(|k| (0..d).map(|j| a[(k, j)] * z[j]).sum::<f64>() + e[k])
            .collect();
        z = next;
        if t >= 200 {
            for k in 0..d {
                f[(t - 200, k)] = scales[k] * z[k];
            }
        }
    }
    let y = unit_rows(&f.rows(1, n).into_owned());
    let x = f.rows(0, n).into_owned();
    (y, x, scales)
}

/// The projected normal log-density `PN(m, S)` at unit vector `u`, with the
/// package's `M_{d−1}` (`log I_{d−1}(α) = ½ ln 2π + α²/2 + ln M_{d−1}(α)`).
fn pn_logpdf(u: &[f64], m: &[f64], s: &DMatrix<f64>) -> f64 {
    let d = u.len();
    let si = s.clone().try_inverse().unwrap();
    let (uv, mv) = (
        nalgebra::DVector::from_column_slice(u),
        nalgebra::DVector::from_column_slice(m),
    );
    let a = uv.dot(&(&si * &uv));
    let b = uv.dot(&(&si * &mv));
    let c = mv.dot(&(&si * &mv));
    let alpha = b / a.sqrt();
    let ln2pi = (2.0 * std::f64::consts::PI).ln();
    -0.5 * d as f64 * ln2pi - 0.5 * s.determinant().ln() - 0.5 * c - 0.5 * d as f64 * a.ln()
        + 0.5 * ln2pi
        + 0.5 * alpha * alpha
        + esag::loglik::log_m_p_value(alpha, d - 1)
}

#[test]
fn equal_scales_change_nothing() {
    // (the log-tangent fit: one smooth start, so rounding-level changes of
    // the data move it by rounding; γ-rot's pilot stops within 1e−3 of the
    // likelihood)
    let (y, x, _) = unequal_scales(800, 3);
    let plain = EsagModel::new(&y, &x)
        .shape(Shape::LogTangent)
        .fit()
        .unwrap();
    let same = EsagModel::new(&y, &x)
        .shape(Shape::LogTangent)
        .response_scale(ResponseScale::Fixed(vec![2.5; 5]))
        .unwrap()
        .fit()
        .unwrap();
    assert_eq!(same.scale().unwrap(), &[1.0; 5]);
    // (the standardized responses are renormalized: equal up to rounding,
    // and so the fits up to their tolerance)
    let (p, q) = (plain.predict(&x).unwrap(), same.predict(&x).unwrap());
    assert!((p - q).abs().max() < 1e-6);
    let (l1, l2) = (
        plain.log_likelihood_rows(&x, &y).unwrap(),
        same.log_likelihood_rows(&x, &y).unwrap(),
    );
    assert!((l1 - l2).abs().max() < 1e-6);
    assert!((plain.log_likelihood - same.log_likelihood_original()).abs() < 1e-5);
}

#[test]
fn tyler_scales_recover_part_scales() {
    let mut rng = ChaCha8Rng::seed_from_u64(11);
    let true_s = [4.0, 2.0, 1.0, 0.5, 0.25];
    let n = 20_000;
    // a common radial factor does not matter: directions only
    let y = DMatrix::from_fn(n, 5, |_, k| {
        let z: f64 = StandardNormal.sample(&mut rng);
        true_s[k] * z
    });
    let s = tyler_scales(&y).unwrap();
    let gm = true_s.iter().map(|v: &f64| v.ln()).sum::<f64>() / 5.0;
    for k in 0..5 {
        let t = true_s[k] / gm.exp();
        assert!((s[k] / t - 1.0).abs() < 0.04, "scale {k}: {} vs {t}", s[k]);
    }
}

/// On the original sphere a standardized model is the projected normal
/// `PN(D μ, D V D)`: its density (ESAG at `unit(y / s)` times the Jacobian)
/// equals that projected normal's, built from `mean_vector` and
/// `predict_covariance`, at any direction.
#[test]
fn scaled_model_is_the_projected_normal_on_the_original_sphere() {
    let (y, x, _) = unequal_scales(1500, 5);
    let res = EsagModel::new(&y, &x)
        .shape(Shape::GammaRot)
        .response_scale(ResponseScale::Tyler)
        .unwrap()
        .fit()
        .unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(7);
    let m = 50;
    let u = unit_rows(&DMatrix::from_fn(m, 5, |_, _| {
        StandardNormal.sample(&mut rng)
    }));
    let xs = DMatrix::from_fn(m, 5, |i, k| x[(17 * i, k)]);
    let l = res.log_likelihood_rows(&xs, &u).unwrap();
    let mean = res.mean_vector(&xs).unwrap();
    let cov = res.predict_covariance(&xs).unwrap();
    for i in 0..m {
        let ui: Vec<f64> = u.row(i).iter().copied().collect();
        let mi: Vec<f64> = mean.row(i).iter().copied().collect();
        let want = pn_logpdf(&ui, &mi, &cov[i]);
        assert!((l[i] - want).abs() < 1e-9, "row {i}: {} vs {want}", l[i]);
        assert!((cov[i].determinant() - 1.0).abs() < 1e-9);
    }
}

#[test]
fn part_scales_fix_unequal_scale_directions() {
    let (y, x, _) = unequal_scales(3000, 9);
    let plain = EsagModel::new(&y, &x).shape(Shape::GammaRot).fit().unwrap();
    let tyler = EsagModel::new(&y, &x)
        .shape(Shape::GammaRot)
        .response_scale(ResponseScale::Tyler)
        .unwrap()
        .fit()
        .unwrap();
    // least squares of the directions on [1, x]
    let z = DMatrix::from_fn(
        x.nrows(),
        6,
        |i, j| if j == 0 { 1.0 } else { x[(i, j - 1)] },
    );
    let coef = (z.transpose() * &z).try_inverse().unwrap() * z.transpose() * &y;
    let ls = mean_cosine(&unit_rows(&(&z * coef)), &y);
    let c_plain = mean_cosine(&plain.predict(&x).unwrap(), &y);
    let c_tyler = mean_cosine(&tyler.predict(&x).unwrap(), &y);
    assert!(tyler.converged);
    assert!(
        c_tyler > c_plain + 0.15,
        "scaled ESAG cosine {c_tyler}, unscaled {c_plain}"
    );
    assert!(
        c_tyler > ls - 0.02,
        "scaled ESAG cosine {c_tyler}, least squares {ls}"
    );
    assert!(tyler.bic_original() < plain.bic_original());
    let auto = EsagModel::new(&y, &x)
        .shape(Shape::GammaRot)
        .response_scale(ResponseScale::Auto)
        .unwrap()
        .fit()
        .unwrap();
    assert!(auto.scale().is_some(), "Auto kept the unscaled model");
    assert!((auto.bic_original() - tyler.bic_original()).abs() < 1e-6);
    assert!(auto.summary().contains("standardized by part scales"));
    // simulations are on the original sphere: their direction agrees with
    // the prediction at a row whose prediction is concentrated
    let mut rng = ChaCha8Rng::seed_from_u64(1);
    let i = (0..x.nrows())
        .max_by(|&a, &b| {
            let ma = tyler
                .mean_vector(&x.rows(a, 1).into_owned())
                .unwrap()
                .norm();
            let mb = tyler
                .mean_vector(&x.rows(b, 1).into_owned())
                .unwrap()
                .norm();
            ma.total_cmp(&mb)
        })
        .unwrap();
    let xi = DMatrix::from_fn(4000, 5, |_, k| x[(i, k)]);
    let draws = tyler.sample_responses(&xi, &mut rng).unwrap();
    let mean: Vec<f64> = (0..5).map(|k| draws.column(k).mean()).collect();
    let norm = mean.iter().map(|v| v * v).sum::<f64>().sqrt();
    let pred = tyler.predict(&x.rows(i, 1).into_owned()).unwrap();
    let cos: f64 = (0..5).map(|k| pred[(0, k)] * mean[k] / norm).sum();
    assert!(
        cos > 0.98,
        "simulated mean direction vs prediction: cosine {cos}"
    );
}
