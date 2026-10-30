//! Project 3 (LASSO-based inference for ESAG): reproduction of the user's R
//! code on its fixtures, and the corrected log-tangent methods.

use esag::penalized::{LambdaGrid, PenalizedEsag};
use nalgebra::DMatrix;
use serde_json::Value;

fn load(name: &str) -> Value {
    let path = format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"));
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn matrix(v: &Value) -> DMatrix<f64> {
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

fn column(v: &Value) -> DMatrix<f64> {
    let xs: Vec<f64> = v
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect();
    DMatrix::from_column_slice(xs.len(), 1, &xs)
}

const METHODS: [(&str, &str); 6] = [
    ("LASSO", "LASSO"),
    ("aLASSO", "aLASSO"),
    ("RoCLASSO", "RoC-LASSO"),
    ("GLASSO", "GLASSO"),
    ("aGLASSO", "aGLASSO"),
    ("RoCGLASSO", "RoC-GLASSO"),
];

#[test]
fn paper_objectives_match_r_at_r_points() {
    // the coded objective (weights from R's unpenalized fit, RoC floor) at
    // R's untruncated optimum reproduces optim's value
    let fx = load("p3_sim_exp1_n100.json");
    let y = matrix(&fx["Y"]);
    let x = DMatrix::<f64>::zeros(y.nrows(), 0);
    let unp = column(&fx["unpenalized_mle"]["par"]);
    for (key, name) in METHODS {
        let est = PenalizedEsag::paper(name)
            .unwrap()
            .with_initial_coef(Some(unp.clone()));
        for fit in fx["methods"][key]["fits"].as_array().unwrap() {
            let lambda = fit["lambda"].as_f64().unwrap();
            let par = column(&fit["par_untruncated"]);
            let want = fit["objective_sum_scale"].as_f64().unwrap();
            let got = est.penalized_loglik(&x, &y, &par, lambda).unwrap();
            assert!(
                (got - want).abs() <= 1e-9 * want.abs().max(1.0),
                "{name} λ={lambda}: {got} vs R {want}"
            );
        }
    }
}

/// Diagnostic (prints a table): from R's start (truth + 0.1) plus esag's own
/// starts, how esag's best optimum compares with R's BFGS point. The γ
/// objective is so rugged (cliffs at the discontinuities of γ ↦ V) that the
/// optimum reached depends on the optimiser's path: R itself differs by up
/// to 28 units between two machines (docs/project3/SPEC.md §2).
/// `cargo test --release --test project3 -- --ignored --nocapture`
#[test]
#[ignore]
fn paper_methods_compared_with_r() {
    let fx = load("p3_sim_exp1_n100.json");
    let y = matrix(&fx["Y"]);
    let x = DMatrix::<f64>::zeros(y.nrows(), 0);
    let unp = column(&fx["unpenalized_mle"]["par"]);
    for (key, name) in METHODS {
        for fit in fx["methods"][key]["fits"].as_array().unwrap() {
            let lambda = fit["lambda"].as_f64().unwrap();
            let start = column(&fit["optim"]["start"]);
            let r_value = fit["objective_sum_scale"].as_f64().unwrap();
            let est = PenalizedEsag::paper(name)
                .unwrap()
                .with_initial_coef(Some(unp.clone()))
                .with_start_coef(Some(start))
                .with_grid(LambdaGrid::Custom(vec![lambda]));
            let path = est.fit(&x, &y).unwrap();
            let p = &path.points[0];
            let ours = -p.objective;
            eprintln!(
                "{name:>10} λ={lambda:>5}: esag {ours:.4}  R {r_value:.4}  (+{:.4})",
                ours - r_value
            );
            if ours < r_value - 1e-6 * r_value.abs() {
                eprintln!("   BELOW R");
            }
        }
    }
}

#[test]
fn paper_methods_never_lose_from_r_optimum() {
    // started at R's untruncated optimum the optimiser only improves it
    let fx = load("p3_sim_exp1_n100.json");
    let y = matrix(&fx["Y"]);
    let x = DMatrix::<f64>::zeros(y.nrows(), 0);
    let unp = column(&fx["unpenalized_mle"]["par"]);
    for (key, name) in METHODS {
        for fit in fx["methods"][key]["fits"].as_array().unwrap() {
            let lambda = fit["lambda"].as_f64().unwrap();
            let rpar = column(&fit["par_untruncated"]);
            let r_value = fit["objective_sum_scale"].as_f64().unwrap();
            let est = PenalizedEsag::paper(name)
                .unwrap()
                .with_initial_coef(Some(unp.clone()))
                .with_start_coef(Some(rpar))
                .with_grid(LambdaGrid::Custom(vec![lambda]));
            let ours = -est.fit(&x, &y).unwrap().points[0].objective;
            assert!(
                ours >= r_value - 1e-9 * r_value.abs(),
                "{name} λ={lambda}: {ours} < R {r_value}"
            );
        }
    }
}

#[test]
fn microbiome_application_reproduces_paper() {
    // RoC-LASSO with BIC on the application data as coded (Alistipes,
    // x/‖x‖): λ = 1 is selected and μ̂ matches the paper's Section 6
    let fx = load("p3_microbiome.json");
    let y = matrix(&fx["as_coded"]["Y"]);
    let x = DMatrix::<f64>::zeros(y.nrows(), 0);
    let unp = column(&fx["as_coded"]["unpenalized_mle"]["par"]);
    let start = column(&fx["as_coded"]["fits"][0]["optim"]["start"]);
    let est = PenalizedEsag::paper("RoC-LASSO")
        .unwrap()
        .with_initial_coef(Some(unp))
        .with_start_coef(Some(start))
        .with_grid(LambdaGrid::Custom(vec![
            0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0,
        ]));
    let path = est.fit(&x, &y).unwrap();
    eprintln!("{}", path.summary());
    let best = path.best();
    assert_eq!(best.lambda, 1.0, "BIC selects λ = 1");
    let paper_mu = [2.50, 0.76, 0.42, 2.48, 1.65, 1.78];
    for (k, &m) in paper_mu.iter().enumerate() {
        assert!(
            (best.coef[(k, 0)] - m).abs() < 0.05,
            "mu[{k}] = {}",
            best.coef[(k, 0)]
        );
    }
    let thr = best.coef_thresholded.as_ref().unwrap();
    // the paper's zeros: gamma_1 and gamma_5
    assert_eq!(thr[(6, 0)], 0.0);
    assert_eq!(thr[(10, 0)], 0.0);
}

// ---- corrected methods (log-tangent shape) ----

use esag::distribution::Esag;
use esag::optim::prox::RocMode;
use esag::penalized::{Grouping, Target};
use esag::regression::Shape;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};

fn tied_iid(n: usize, seed: u64) -> DMatrix<f64> {
    // d = 6, eigenvalues of S tied as {1} {2,3,4,5}
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
    let mu = [2.0, -5.0, 3.0, -4.0, 5.0, 6.0];
    let fr = esag::logtangent::LtFrame::new(&mu);
    let p = 5;
    let q = nalgebra::linalg::QR::new(DMatrix::from_fn(p, p, |i, j| {
        ((i * 3 + j * 7) as f64).sin()
    }))
    .q();
    let s_mat = &q
        * DMatrix::from_diagonal(&nalgebra::DVector::from_vec(vec![-2.0, 0.5, 0.5, 0.5, 0.5]))
        * q.transpose();
    let flat: Vec<f64> = (0..p)
        .flat_map(|r| (0..p).map(move |c| (r, c)))
        .map(|(r, c)| s_mat[(r, c)])
        .collect();
    let s = esag::logtangent::matrix_to_s(&flat, p);
    let v = esag::logtangent::covariance(&fr, &mu, &s);
    Esag::from_covariance(&mu, &v).unwrap().sample(n, &mut rng)
}

#[test]
fn eigengap_recovers_tied_eigenvalues() {
    let y = tied_iid(150, 1);
    let x = DMatrix::<f64>::zeros(150, 0);
    let path = PenalizedEsag::eigen_gap().fit(&x, &y).unwrap();
    let b = path.best();
    assert_eq!(
        b.active,
        vec![true, false, false, false],
        "{}",
        path.summary()
    );
    assert_eq!(b.df, 11.0); // d + (K − 1) + k(k−1)/2 − Σ m(m−1)/2 = 6 + 1 + 10 − 6
    assert!(b.converged);
    // exact ties (no thresholding) along the whole path; λ = 0 is the MLE
    assert!((path.points[0].log_likelihood - path.initial.log_likelihood).abs() < 1e-6);
}

#[test]
fn isotropy_penalty_detects_isotropic_errors() {
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(4);
    let mu = [2.0, -1.0, 1.5, 0.5];
    let x = DMatrix::<f64>::zeros(300, 0);
    let iso = Esag::new(&mu, &[0.0; 5]).unwrap().sample(300, &mut rng);
    let path = PenalizedEsag::isotropy().fit(&x, &iso).unwrap();
    assert_eq!(path.best().active, vec![false]);
    let relaxed = path.relaxed.as_ref().unwrap();
    assert!(relaxed.coef.rows(4, 5).iter().all(|&v| v == 0.0));
    let aniso = Esag::new(&mu, &[1.5, -0.5, 1.0, 0.5, -0.8])
        .unwrap()
        .sample(300, &mut rng);
    let path = PenalizedEsag::isotropy().fit(&x, &aniso).unwrap();
    assert_eq!(path.best().active, vec![true]);
}

#[test]
fn driver_selection_in_regression() {
    // d = 5, four drivers: x1 shifts the mean, x2 the shape, x3 and x4 nothing
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(7);
    let (d, n, p) = (5usize, 1500usize, 4usize);
    let bd = esag::dims::eta_len(d);
    let lt = esag::logtangent::LtFrame::standard(d);
    let b = DMatrix::from_fn(bd, p + 1, |k, j| {
        let s = (k * 7 + j * 3) as f64;
        match j {
            0 => {
                if k < d {
                    0.8 * (s * 0.9).sin() + if k == d - 1 { 3.0 } else { 0.0 }
                } else {
                    0.5 * (s * 1.3).sin()
                }
            }
            1 => {
                if k < d {
                    0.5 * (s * 0.7).cos()
                } else {
                    0.0
                }
            }
            2 => {
                if k < d {
                    0.0
                } else {
                    0.35 * (s * 1.1).sin()
                }
            }
            _ => 0.0,
        }
    });
    let x = DMatrix::from_fn(n, p, |_, _| StandardNormal.sample(&mut rng));
    let mut y = DMatrix::zeros(n, d);
    let mut buf = vec![0.0; d];
    for i in 0..n {
        let eta: Vec<f64> = (0..bd)
            .map(|k| b[(k, 0)] + (0..p).map(|j| b[(k, j + 1)] * x[(i, j)]).sum::<f64>())
            .collect();
        let v = esag::logtangent::covariance(&lt, &eta[..d], &eta[d..]);
        let dist = Esag::from_covariance(&eta[..d], &v).unwrap();
        dist.sample_one(&dist.frame(), &mut rng, &mut buf);
        for k in 0..d {
            y[(i, k)] = buf[k];
        }
    }
    let path = PenalizedEsag::adaptive_group_lasso()
        .grouping(Grouping::CovariatePart)
        .fit(&x, &y)
        .unwrap();
    // groups: x1.mu x1.shape x2.mu x2.shape x3.mu x3.shape x4.mu x4.shape
    assert_eq!(
        path.best().active,
        vec![true, false, false, true, false, false, false, false],
        "{}",
        path.summary()
    );
    let relaxed = path.relaxed.as_ref().unwrap();
    assert!(relaxed.std_errors().is_some());
    // zero groups are exact zeros on the original scale too
    assert!(relaxed.coef.column(3).iter().all(|&v| v == 0.0));
}

#[test]
fn cv_folds_fit_the_full_data_frame() {
    // log-tangent CV: every fold uses the reference direction of the full
    // data (by default its mean direction), not the one of its training rows
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(9);
    let (d, n) = (4usize, 240usize);
    let x = DMatrix::from_fn(n, 1, |_, _| StandardNormal.sample(&mut rng));
    let mut y = DMatrix::zeros(n, d);
    let mut buf = vec![0.0; d];
    for i in 0..n {
        let t = x[(i, 0)];
        let dist = Esag::new(
            &[1.0 + 0.4 * t, -0.5, 0.8, 2.0],
            &[0.6 * t, -0.3, 0.4, 0.2, 0.1],
        )
        .unwrap();
        dist.sample_one(&dist.frame(), &mut rng, &mut buf);
        for k in 0..d {
            y[(i, k)] = buf[k];
        }
    }
    let mut m: Vec<f64> = (0..d).map(|k| y.column(k).sum()).collect();
    let nrm = m.iter().map(|v| v * v).sum::<f64>().sqrt();
    m.iter_mut().for_each(|v| *v /= nrm);
    let default = PenalizedEsag::lasso().cv(4, 3).fit(&x, &y).unwrap();
    let explicit = PenalizedEsag::lasso()
        .lt_reference(m)
        .cv(4, 3)
        .fit(&x, &y)
        .unwrap();
    for (a, b) in default.criterion.iter().zip(&explicit.criterion) {
        // 1e-3 relative when each fold took its own mean direction; the
        // rest is the rounding of `m` amplified by the optimiser's tolerance
        assert!((a - b).abs() <= 1e-6 * a.abs().max(1.0), "{a} vs {b}");
    }
    assert_eq!(default.selected, explicit.selected);
}

/// RoC weights in regression: the exponent is the mean of the
/// per-observation concentration ratios, `r = max(1, (1/n) Σ_i ‖μ̃_i‖/‖μ̂_i‖)`,
/// at every point of the path (the fixed point), and it leaves the floor
/// once the penalty shrinks the mean slopes.
#[test]
fn roc_weights_in_regression_are_the_mean_ratio() {
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(11);
    let (d, n, p) = (4usize, 400usize, 2usize);
    let bd = esag::dims::eta_len(d);
    let b = DMatrix::from_fn(bd, p + 1, |k, j| {
        let s = (k * 5 + j * 3) as f64;
        match (j, k < d) {
            (0, true) => 0.6 * (s * 0.9).sin() + if k == d - 1 { 2.5 } else { 0.0 },
            (0, false) => 0.4 * (s * 1.3).sin(),
            (1, true) => 0.5 * (s * 0.7).cos(),
            (2, false) => 0.3 * (s * 1.1).sin(),
            _ => 0.0,
        }
    });
    let x = DMatrix::from_fn(n, p, |_, _| StandardNormal.sample(&mut rng));
    let mut y = DMatrix::zeros(n, d);
    let mut buf = vec![0.0; d];
    for i in 0..n {
        let eta: Vec<f64> = (0..bd)
            .map(|k| b[(k, 0)] + (0..p).map(|j| b[(k, j + 1)] * x[(i, j)]).sum::<f64>())
            .collect();
        let dist = Esag::from_eta(&eta).unwrap();
        dist.sample_one(&dist.frame(), &mut rng, &mut buf);
        for k in 0..d {
            y[(i, k)] = buf[k];
        }
    }
    // ‖μ_i‖ of coefficients c on the design z
    let norms = |c: &DMatrix<f64>, z: &DMatrix<f64>| -> Vec<f64> {
        (0..n)
            .map(|i| {
                (0..d)
                    .map(|k| {
                        let m = c[(k, 0)] + (0..p).map(|j| c[(k, j + 1)] * z[(i, j)]).sum::<f64>();
                        m * m
                    })
                    .sum::<f64>()
                    .sqrt()
            })
            .collect()
    };
    // the initial fit is on the standardized covariates
    let xs = DMatrix::from_fn(n, p, |i, j| {
        let c = x.column(j);
        let m = c.sum() / n as f64;
        let sd = (c.iter().map(|v: &f64| (v - m).powi(2)).sum::<f64>() / n as f64).sqrt();
        (x[(i, j)] - m) / sd
    });
    for shape in [
        Shape::GammaRot,
        Shape::Spectral,
        Shape::LogTangent,
        Shape::Gamma,
    ] {
        let path = PenalizedEsag::group_lasso()
            .roc(RocMode::FixedPoint)
            .shape(shape)
            .target(Target::Slopes)
            .grouping(Grouping::CovariatePart)
            .fit(&x, &y)
            .unwrap();
        let tilde = norms(&path.initial.coef, &xs);
        let mut off_floor = 0;
        // (γ's solves stop unconverged at half the path on these data, with
        // adaptive weights too: `gamma_penalized_kkt`; the smooth shapes'
        // never do)
        if shape != Shape::Gamma {
            assert!(path.points.iter().all(|p| p.converged), "{shape:?}");
        }
        for pt in path.points.iter().filter(|p| p.converged) {
            let hat = norms(&pt.coef, &x);
            let mean: f64 = tilde.iter().zip(&hat).map(|(t, h)| t / h).sum::<f64>() / n as f64;
            let want = mean.max(1.0);
            assert!(
                (pt.r - want).abs() <= 1e-7 * want,
                "{shape:?} λ/n {}: r {} against the mean ratio {want}",
                pt.lambda_n,
                pt.r
            );
            off_floor += usize::from(pt.r > 1.0 + 1e-9);
        }
        assert!(off_floor > 0, "{shape:?}: r never left the floor");
        assert!(
            path.config.method_name().starts_with("RoC-"),
            "{}",
            path.config.method_name()
        );
    }
    // the paper's joint objective stays i.i.d.
    assert!(
        PenalizedEsag::group_lasso()
            .roc(RocMode::Joint)
            .fit(&x, &y)
            .is_err()
    );
}

/// RoC-LASSO and RoC-GLASSO on i.i.d. data with every shape: the paths
/// solve, and `r ≥ 1` with `r = max(1, ‖μ̃‖/‖μ̂‖)`.
#[test]
fn roc_weights_for_every_shape_iid() {
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(5);
    let y = Esag::new(&[2.0, -1.0, 1.5, 3.0], &[1.2, -0.6, 0.0, 0.8, -0.3])
        .unwrap()
        .sample(300, &mut rng);
    let x = DMatrix::<f64>::zeros(300, 0);
    for (shape, grouping) in [
        (Shape::Gamma, Grouping::GammaGroups),
        (Shape::GammaRot, Grouping::GammaGroups),
        (Shape::Spectral, Grouping::EigenParts),
        (Shape::LogTangent, Grouping::Parameter),
    ] {
        for est in [
            PenalizedEsag::lasso().roc(RocMode::FixedPoint),
            PenalizedEsag::group_lasso()
                .roc(RocMode::FixedPoint)
                .grouping(grouping.clone()),
        ] {
            let path = est.shape(shape).target(Target::Shape).fit(&x, &y).unwrap();
            let mt = (0..4)
                .map(|k| path.initial.coef[(k, 0)].powi(2))
                .sum::<f64>()
                .sqrt();
            for pt in &path.points {
                let mh = (0..4).map(|k| pt.coef[(k, 0)].powi(2)).sum::<f64>().sqrt();
                let want = (mt / mh).max(1.0);
                assert!(
                    (pt.r - want).abs() <= 1e-6 * want,
                    "{shape:?}: r {} vs {want}",
                    pt.r
                );
            }
            assert!(path.best().converged, "{shape:?}: {}", path.summary());
        }
    }
}

/// (diagnostic) Penalized γ fits of the RoC regression test's data with
/// adaptive and RoC weights: unconverged path points and their KKT.
/// `cargo test --release --test project3 gamma_penalized_kkt -- --ignored --nocapture`
#[test]
#[ignore]
fn gamma_penalized_kkt() {
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(11);
    let (d, n, p) = (4usize, 400usize, 2usize);
    let bd = esag::dims::eta_len(d);
    let b = DMatrix::from_fn(bd, p + 1, |k, j| {
        let s = (k * 5 + j * 3) as f64;
        match (j, k < d) {
            (0, true) => 0.6 * (s * 0.9).sin() + if k == d - 1 { 2.5 } else { 0.0 },
            (0, false) => 0.4 * (s * 1.3).sin(),
            (1, true) => 0.5 * (s * 0.7).cos(),
            (2, false) => 0.3 * (s * 1.1).sin(),
            _ => 0.0,
        }
    });
    let x = DMatrix::from_fn(n, p, |_, _| StandardNormal.sample(&mut rng));
    let mut y = DMatrix::zeros(n, d);
    let mut buf = vec![0.0; d];
    for i in 0..n {
        let eta: Vec<f64> = (0..bd)
            .map(|k| b[(k, 0)] + (0..p).map(|j| b[(k, j + 1)] * x[(i, j)]).sum::<f64>())
            .collect();
        let dist = Esag::from_eta(&eta).unwrap();
        dist.sample_one(&dist.frame(), &mut rng, &mut buf);
        for k in 0..d {
            y[(i, k)] = buf[k];
        }
    }
    for shape in [Shape::Gamma, Shape::GammaRot] {
        for (name, est) in [
            ("adaptive", PenalizedEsag::adaptive_group_lasso()),
            ("RoC", PenalizedEsag::group_lasso().roc(RocMode::FixedPoint)),
        ] {
            let path = est
                .shape(shape)
                .target(Target::Slopes)
                .grouping(Grouping::CovariatePart)
                .fit(&x, &y)
                .unwrap();
            let bad: Vec<String> = path
                .points
                .iter()
                .filter(|p| !p.converged)
                .map(|p| format!("λ/n {:.3} kkt {:.1e}", p.lambda_n, p.kkt))
                .collect();
            println!(
                "{shape:?} {name}: {} of {} points unconverged {:?}; selected {:?}",
                bad.len(),
                path.points.len(),
                bad,
                path.best().active
            );
        }
    }
}
