//! Parity of the deterministic ESAG math with the original R code
//! (`reference/r/ESAG_Project2.R`). Fixtures were produced by running the R
//! code; see `tests/fixtures/README.md`.

use esag::frame::Frame;
use esag::loglik::{loglik_eta, loglik_mu_gamma};
use esag::onb::onb;
use esag::param::gamma_to_angles;
use esag::rotation::rotation_matrix;
use serde_json::Value;

fn load(name: &str) -> Value {
    let path = format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"));
    serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap()
}

/// A vector that R may have written as a scalar or an array.
fn vecf(v: &Value) -> Vec<f64> {
    match v {
        Value::Array(a) => a.iter().map(|x| x.as_f64().unwrap()).collect(),
        Value::Number(n) => vec![n.as_f64().unwrap()],
        Value::Null => vec![],
        _ => panic!("not a vector: {v}"),
    }
}

/// Row-major nested array → flat row-major vector.
fn matf(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().flat_map(vecf).collect()
}

fn close(a: f64, b: f64, rel: f64, abs: f64) -> bool {
    (a - b).abs() <= abs.max(rel * a.abs().max(b.abs()))
}

fn assert_all_close(got: &[f64], want: &[f64], rel: f64, abs: f64, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(close(*g, *w, rel, abs), "{what}[{i}]: rust {g} vs R {w}");
    }
}

#[test]
fn parameterization_onb_rotation_covariance() {
    let fx = load("parameterization.json");
    let mut n_checked = 0;
    for dim in fx["dims"].as_array().unwrap() {
        let d = dim["d"].as_u64().unwrap() as usize;
        // ONB
        for mc in dim["mu_cases"].as_array().unwrap() {
            let mu = vecf(&mc["mu"]);
            let u = onb(&mu); // column-major == transpose of row-major
            let want = matf(&mc["ONB"]);
            let mut got_rm = vec![0.0; d * d];
            for c in 0..d {
                for r in 0..d {
                    got_rm[r * d + c] = u[c * d + r];
                }
            }
            assert_all_close(
                &got_rm,
                &want,
                1e-12,
                1e-14,
                &format!("ONB d={d} {}", mc["label"]),
            );
            n_checked += 1;
        }
        // gamma cases
        for gc in dim["gamma_cases"].as_array().unwrap() {
            let label = gc["label"].as_str().unwrap();
            let gamma = vecf(&gc["gamma"]);
            let a = gamma_to_angles(&gamma);
            assert_all_close(
                &a.lambda,
                &vecf(&gc["lambda"]),
                1e-12,
                1e-15,
                &format!("lambda d={d} {label}"),
            );
            assert_all_close(
                &a.theta,
                &vecf(&gc["theta"]),
                1e-12,
                1e-14,
                &format!("theta d={d} {label}"),
            );
            assert_all_close(
                &a.phi,
                &vecf(&gc["phi"]),
                1e-12,
                1e-14,
                &format!("phi d={d} {label}"),
            );
            let r = rotation_matrix(&a.theta, &a.phi);
            assert_all_close(
                &r,
                &matf(&gc["rotation"]),
                1e-12,
                1e-14,
                &format!("rotation d={d} {label}"),
            );
            for cov in gc["covariance"].as_array().unwrap() {
                let mu = vecf(&cov["mu"]);
                let f = Frame::new(&mu, &gamma);
                for (key, s) in [("V_p1", 1.0), ("V_pm1", -1.0), ("V_p05", 0.5)] {
                    let want = matf(&cov[key]);
                    let scale = want.iter().fold(0.0f64, |m, v| m.max(v.abs()));
                    assert_all_close(
                        &f.cov_pow(s),
                        &want,
                        1e-10,
                        1e-12 * scale.max(1.0),
                        &format!("{key} d={d} {label} mu={}", cov["mu_label"]),
                    );
                }
                n_checked += 1;
            }
        }
    }
    assert!(n_checked > 100, "only {n_checked} cases checked");
}

#[test]
fn log_likelihood_matches_r() {
    let fx = load("loglik.json");
    let mut worst: f64 = 0.0;
    for c in fx["cases"].as_array().unwrap() {
        let label = c["label"].as_str().unwrap();
        let d = c["d"].as_u64().unwrap() as usize;
        let mu = vecf(&c["mu"]);
        let gamma = vecf(&c["gamma"]);
        let y = vecf(&c["y"]);
        let want = c["log_likelihood"].as_f64().unwrap();
        let got = loglik_mu_gamma(&mu, &gamma, &y);
        let got2: f64 = loglik_eta(&vecf(&c["Bx"]), &y);
        assert_eq!(got, got2);
        // R evaluates M_p by the forward recursion, which cancels for
        // alpha ≈ -5.5 ("numerically_delicate"); there R is the less accurate one.
        let tol = if label.contains("delicate") {
            1e-6
        } else {
            1e-10
        };
        let rel = (got - want).abs() / want.abs().max(1.0);
        worst = worst.max(if label.contains("delicate") { 0.0 } else { rel });
        assert!(
            rel <= tol,
            "d={d} {label}: rust {got} vs R {want} (rel {rel:.2e})"
        );
    }
    eprintln!("worst relative deviation from R (non-delicate cases): {worst:.2e}");
}

#[test]
fn joint_loglik_of_regression_fixture_matches_r() {
    for name in ["regression_d3_n200.json", "regression_d3_n1000.json"] {
        let fx = load(name);
        let x = matf(&fx["X"]);
        let y = matf(&fx["Y"]);
        let n = fx["meta"]["n"].as_u64().unwrap() as usize;
        let b = matf(&fx["B_true"]); // 5 x 2 row-major
        let par = matf(&fx["mle"]["par_matrix"]);
        let joint = |bm: &[f64]| -> f64 {
            (0..n)
                .map(|i| {
                    let xi = &x[i * 2..i * 2 + 2];
                    let eta: Vec<f64> = (0..5)
                        .map(|k| bm[k * 2] * xi[0] + bm[k * 2 + 1] * xi[1])
                        .collect();
                    loglik_eta(&eta, &y[i * 3..i * 3 + 3])
                })
                .sum()
        };
        let at_true = fx["joint_loglik_at_B_true"].as_f64().unwrap();
        let at_mle = fx["mle"]["value"].as_f64().unwrap();
        let bp: Vec<f64> = b.iter().map(|v| v + 0.1).collect();
        let at_plus = fx["joint_loglik_at_B_true_plus_0.1"].as_f64().unwrap();
        assert!(
            close(joint(&b), at_true, 1e-11, 0.0),
            "{name}: {} vs {at_true}",
            joint(&b)
        );
        assert!(
            close(joint(&bp), at_plus, 1e-11, 0.0),
            "{name}: {} vs {at_plus}",
            joint(&bp)
        );
        assert!(
            close(joint(&par), at_mle, 1e-11, 0.0),
            "{name}: {} vs {at_mle}",
            joint(&par)
        );
    }
}

#[test]
fn gamma_group1_sign_is_an_exact_symmetry() {
    // theta_1 -> theta_1 + π negates v_1, v_2 and leaves V unchanged.
    for d in 3..=7usize {
        let g = esag::dims::gamma_len(d);
        let mu: Vec<f64> = (0..d).map(|i| 0.4 + (i as f64 * 0.9).sin()).collect();
        let gamma: Vec<f64> = (0..g).map(|i| 0.3 + (i as f64 * 1.7).cos()).collect();
        let mut flipped = gamma.clone();
        flipped[0] = -flipped[0];
        flipped[1] = -flipped[1];
        let v1 = Frame::new(&mu, &gamma).cov_pow(1.0);
        let v2 = Frame::new(&mu, &flipped).cov_pow(1.0);
        assert_all_close(
            &v1,
            &v2,
            1e-12,
            1e-13,
            &format!("V under group-1 flip, d={d}"),
        );
        let y: Vec<f64> = {
            let raw: Vec<f64> = (0..d).map(|i| (i as f64 * 2.3).cos()).collect();
            let n = raw.iter().map(|v| v * v).sum::<f64>().sqrt();
            raw.iter().map(|v| v / n).collect()
        };
        let a = loglik_mu_gamma(&mu, &gamma, &y);
        let b = loglik_mu_gamma(&mu, &flipped, &y);
        assert!((a - b).abs() < 1e-12, "d={d}: {a} vs {b}");
    }
}
