#![allow(clippy::needless_range_loop)]
//! Reproduce the real-data ESAG regression of Yu & Huang (2025, CSDA,
//! Section 7.1; dissertation Section 2.9.1): square-root compositions of
//! (K, Na, Ca, Mg) in Llobregat-basin water, Anoia tributaries (X = 0, n = 67)
//! vs lower-Llobregat tributaries (X = 1, n = 43). Data: R package
//! `compositions`, dataset `Hydrochem` (GPL ≥ 2), see `data/README.md`.

use esag::regression::{EsagRegressor, EsagRegressorFit};
use nalgebra::DMatrix;

pub fn load() -> (DMatrix<f64>, DMatrix<f64>) {
    let path = format!("{}/data/hydrochem_At_LLt.csv", env!("CARGO_MANIFEST_DIR"));
    let mut rdr = csv::Reader::from_path(path).unwrap();
    let rows: Vec<csv::StringRecord> = rdr.records().map(|r| r.unwrap()).collect();
    let n = rows.len();
    let x = DMatrix::from_fn(n, 1, |i, _| rows[i][1].parse::<f64>().unwrap());
    let y = DMatrix::from_fn(n, 4, |i, j| rows[i][6 + j].parse::<f64>().unwrap());
    (x, y)
}

fn fit() -> EsagRegressorFit {
    let (x, y) = load();
    EsagRegressor::new()
        .with_restarts(10, 3)
        .fit(&x, &y)
        .unwrap()
}

fn assert_close(got: &[f64], want: &[f64], tol: f64, what: &str) {
    for (g, w) in got.iter().zip(want) {
        assert!(
            (g - w).abs() <= tol,
            "{what}: got {got:.3?}, paper {want:?}"
        );
    }
}

#[test]
fn reproduces_published_estimates() {
    let (x, _) = load();
    assert_eq!(x.nrows(), 110);
    let f = fit();
    assert!(f.converged);
    // mu-hat_i = (1.99, 5.74, 7.95, 4.59) + (1.28, 2.83, 1.06, 1.20) X_i
    let a0: Vec<f64> = (0..4).map(|k| f.coef[(k, 0)]).collect();
    let a1: Vec<f64> = (0..4).map(|k| f.coef[(k, 1)]).collect();
    assert_close(&a0, &[1.99, 5.74, 7.95, 4.59], 0.006, "alpha_0");
    assert_close(&a1, &[1.28, 2.83, 1.06, 1.20], 0.006, "alpha_1");

    // Distribution-level quantities (invariant to the gamma equivalence class)
    let paper = [
        // (|mu|, |gamma|, lambda, V)
        (
            11.00,
            6.24,
            [0.37, 0.62, 4.44],
            [
                0.93, 1.15, -0.76, -0.09, 1.15, 2.77, -1.41, -0.27, -0.76, -1.41, 1.99, 0.38,
                -0.09, -0.27, 0.38, 0.73,
            ],
        ),
        (
            14.10,
            17.03,
            [0.19, 0.54, 9.61],
            [
                0.63, 1.50, -0.71, -0.90, 1.50, 5.36, -2.66, -3.17, -0.71, -2.66, 2.43, 2.10,
                -0.90, -3.17, 2.10, 2.91,
            ],
        ),
    ];
    for (xv, (mu_norm, g_norm, lambda, v)) in [0.0, 1.0].iter().zip(paper) {
        let dist = f
            .predict_distribution(&DMatrix::from_element(1, 1, *xv))
            .unwrap()
            .remove(0);
        let gn = dist.gamma().iter().map(|t| t * t).sum::<f64>().sqrt();
        assert!((dist.mu_norm() - mu_norm).abs() < 0.006, "|mu| X={xv}");
        assert!((gn - g_norm).abs() < 0.006, "|gamma| X={xv}: {gn}");
        assert_close(
            &dist.angles().lambda,
            &lambda,
            0.006,
            &format!("lambda X={xv}"),
        );
        let vm = dist.covariance();
        let vr: Vec<f64> = (0..4)
            .flat_map(|i| (0..4).map(move |j| (i, j)))
            .map(|(i, j)| vm[(i, j)])
            .collect();
        // the paper prints two decimals
        assert_close(&vr, &v, 0.0151, &format!("V X={xv}"));
    }
}

#[test]
fn reproduces_published_test_statistics_and_region_quantiles() {
    use esag::inference::{Restriction, hypothesis_test, prediction_region};
    let (x, y) = load();
    let f = fit();
    // dissertation Table 2.9 (RoC, D, M, LR); M is Monte-Carlo based
    for (r, roc, dd, m, lr) in [
        (Restriction::Isotropy, 1.518, 1.518, Some(2.2e-4), 275.4),
        (Restriction::MuOfFeature(0), 1.059, 1.062, None, 31.6),
        (
            Restriction::GammaOfFeature(0),
            1.090,
            1.090,
            Some(9.4e-5),
            57.6,
        ),
    ] {
        let t = hypothesis_test(&f, &x, &y, r.clone(), 20, 4000, 2).unwrap();
        assert!((t.roc - roc).abs() < 6e-4, "{r:?} RoC {}", t.roc);
        assert!((t.d_stat - dd).abs() < 6e-4, "{r:?} D {}", t.d_stat);
        assert!((t.lr - lr).abs() < 0.06, "{r:?} LR {}", t.lr);
        if let Some(m) = m {
            assert!((t.m_stat - m).abs() / m < 0.2, "{r:?} M {}", t.m_stat);
        }
        assert!(t.p_roc < 0.05 && t.p_d < 0.05 && t.p_lr < 0.05);
    }
    // dissertation Table 2.10: q at 70/80/90 %
    for (xv, want) in [
        (0.0, [0.0286, 0.0364, 0.0495]),
        (1.0, [0.0178, 0.0227, 0.0312]),
    ] {
        let pr = prediction_region(&f, &x, &y, &[xv], 20_000, 60, &[0.70, 0.80, 0.90], 8).unwrap();
        for (g, w) in pr.thresholds.iter().zip(want) {
            assert!((g - w).abs() / w < 0.04, "X={xv}: {g} vs {w}");
        }
    }
}
