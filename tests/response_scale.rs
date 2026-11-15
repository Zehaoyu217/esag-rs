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
    var_directions(n, seed, &[8.0, 3.0, 1.5, 0.8, 0.4])
}

fn var_directions(n: usize, seed: u64, scales: &[f64]) -> (DMatrix<f64>, DMatrix<f64>, Vec<f64>) {
    let d = 5;
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let g = DMatrix::from_fn(d, d, |_, _| StandardNormal.sample(&mut rng));
    let a = g.qr().q() * 0.7;
    let scales = scales.to_vec();
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

/// `ResponseScale::Auto` keeps plain ESAG where the scales do not help
/// (`python/examples/scale_choice.py`: equal part scales, and directions
/// concentrated around a mean, where Tyler's zero-centred scales distort):
/// the guide's claim that `Auto` is the safe default.
#[test]
fn auto_keeps_plain_esag_where_scales_do_not_help() {
    // equal part scales
    let (y, x, _) = var_directions(2000, 21, &[1.0; 5]);
    let auto = EsagModel::new(&y, &x)
        .shape(Shape::GammaRot)
        .response_scale(ResponseScale::Auto)
        .unwrap()
        .fit()
        .unwrap();
    assert!(
        auto.scale().is_none(),
        "equal scales: Auto chose {:?}",
        auto.scale()
    );
    // concentrated directions: ESAG regression with |mu_0| = 2.5
    let mut rng = ChaCha8Rng::seed_from_u64(23);
    let (n, d) = (2000usize, 5usize);
    let x = DMatrix::from_fn(n, 3, |_, _| StandardNormal.sample(&mut rng));
    let mut y = DMatrix::zeros(n, d);
    let mut buf = vec![0.0; d];
    for i in 0..n {
        let mu: Vec<f64> = (0..d)
            .map(|k| 2.5 / (d as f64).sqrt() + [0.5, -0.3, 0.2, 0.0, -0.4][k] * x[(i, 0)])
            .collect();
        let gamma: Vec<f64> = (0..9)
            .map(|k| 0.4 * ((k as f64) * 0.9).sin() + 0.25 * x[(i, 1)] * ((k as f64) * 1.7).cos())
            .collect();
        let dist = esag::distribution::Esag::new(&mu, &gamma).unwrap();
        dist.sample_one(&dist.frame(), &mut rng, &mut buf);
        for k in 0..d {
            y[(i, k)] = buf[k];
        }
    }
    let auto = EsagModel::new(&y, &x)
        .shape(Shape::GammaRot)
        .response_scale(ResponseScale::Auto)
        .unwrap()
        .fit()
        .unwrap();
    assert!(
        auto.scale().is_none(),
        "concentrated: Auto chose {:?}",
        auto.scale()
    );
}

/// The command line: `fit --scale tyler` saves the scales with the model, and
/// `predict` from the saved model returns the library's directions on the
/// original sphere.
#[test]
fn command_line_saves_and_applies_part_scales() {
    let (y, x, _) = unequal_scales(600, 31);
    let dir = std::env::temp_dir().join(format!("esag_scale_cli_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let data = dir.join("flows.csv");
    let mut text = String::from("x1,x2,x3,x4,x5,y1,y2,y3,y4,y5\n");
    for i in 0..y.nrows() {
        let row: Vec<String> = (0..5)
            .map(|k| format!("{}", x[(i, k)]))
            .chain((0..5).map(|k| format!("{}", y[(i, k)])))
            .collect();
        text.push_str(&row.join(","));
        text.push('\n');
    }
    std::fs::write(&data, text).unwrap();
    let model = dir.join("m.json");
    let bin = env!("CARGO_BIN_EXE_esag");
    let fit = std::process::Command::new(bin)
        .args(["fit", data.to_str().unwrap(), "--y", "y1,y2,y3,y4,y5"])
        .args([
            "--x",
            "x1,x2,x3,x4,x5",
            "--shape",
            "gamma-rot",
            "--scale",
            "tyler",
        ])
        .args(["--model", model.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        fit.status.success(),
        "{}",
        String::from_utf8_lossy(&fit.stderr)
    );
    assert!(String::from_utf8_lossy(&fit.stdout).contains("standardized by part scales"));
    let saved =
        esag::regression::EsagRegressorFit::from_json(&std::fs::read_to_string(&model).unwrap())
            .unwrap();
    let lib = EsagModel::new(&y, &x)
        .shape(Shape::GammaRot)
        .response_scale(ResponseScale::Tyler)
        .unwrap()
        .fit()
        .unwrap();
    assert_eq!(saved.response_scale.as_deref(), lib.scale());
    let pred = std::process::Command::new(bin)
        .args(["predict", model.to_str().unwrap(), data.to_str().unwrap()])
        .args(["--x", "x1,x2,x3,x4,x5"])
        .output()
        .unwrap();
    assert!(
        pred.status.success(),
        "{}",
        String::from_utf8_lossy(&pred.stderr)
    );
    let want = lib.predict(&x).unwrap();
    let out = String::from_utf8_lossy(&pred.stdout);
    for (i, line) in out.lines().skip(1).take(20).enumerate() {
        let v: Vec<f64> = line
            .split(',')
            .take(5)
            .map(|t| t.parse().unwrap())
            .collect();
        for k in 0..5 {
            assert!(
                (v[k] - want[(i, k)]).abs() < 1e-8,
                "row {i}: {} vs {}",
                v[k],
                want[(i, k)]
            );
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}
