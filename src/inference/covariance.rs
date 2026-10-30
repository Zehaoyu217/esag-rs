//! Estimators of the ESAG covariance `V(x)` and their bootstrap uncertainty.
//!
//! `V(x)` is a nonlinear function of every coefficient (through `μ(x)` and
//! the shape coordinates), so the observed information of `B` says little
//! about it directly. [`covariance_intervals`] gives, at each covariate
//! vector, the plug-in estimate `V̂(x)`, its shape eigenvalues and principal
//! axes, and pairs-bootstrap uncertainty:
//!
//! * *basic* (bias-corrected) intervals `[2θ̂ − q_{1−α/2}, 2θ̂ − q_{α/2}]` for
//!   each shape eigenvalue and each entry. The MLE of the smallest
//!   eigenvalue can be biased by about one standard error; percentile
//!   intervals double that bias, basic intervals remove it to first order;
//! * bootstrap standard errors of the entries;
//! * a geodesic confidence ball `{V : ‖log(V̂^{−1/2} V V̂^{−1/2})‖_F ≤ r}`,
//!   with `r` the level-quantile of the distances of the bootstrap `V̂*`
//!   from `V̂`.
//!
//! In simulations with n = 2000 and 5 covariates (`examples/v_recovery.rs`),
//! these have 93–98% coverage at level 95% for d = 3 and 5 and both shapes.

use super::regions::bootstrap_refits;
use crate::distribution::Esag;
use crate::error::{EsagError, Result};
use crate::regression::EsagRegressorFit;
use crate::stats::quantile::quantiles_type7;
use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};

/// `V̂(x)` with its bootstrap uncertainty (see the module docs).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CovarianceEstimate {
    /// The covariate vector (without the constant).
    pub x: Vec<f64>,
    /// `V̂(x)` (`d × d`; `V̂μ̂ = μ̂`, `det V̂ = 1`).
    pub covariance: DMatrix<f64>,
    /// `μ̂(x)/‖μ̂(x)‖`.
    pub mean_direction: Vec<f64>,
    /// Shape eigenvalues: those of `V̂` on `μ̂⊥` (the remaining one, along
    /// `μ̂`, is 1 by construction), ascending.
    pub eigenvalues: Vec<f64>,
    /// Principal axes (`d × (d−1)`), in the order of `eigenvalues`.
    pub axes: DMatrix<f64>,
    /// Basic bootstrap interval of each shape eigenvalue.
    pub eig_lower: Vec<f64>,
    pub eig_upper: Vec<f64>,
    /// Bootstrap standard error of each shape eigenvalue.
    pub eig_se: Vec<f64>,
    /// Bootstrap standard errors of the entries of `V̂`.
    pub entry_se: DMatrix<f64>,
    /// Basic bootstrap intervals of the entries of `V̂`.
    pub entry_lower: DMatrix<f64>,
    pub entry_upper: DMatrix<f64>,
    /// Radius of the geodesic confidence ball around `V̂`.
    pub radius: f64,
    /// Confidence level of the intervals and the ball.
    pub level: f64,
    /// Bootstrap refits used.
    pub n_boot: usize,
}

/// Shape eigenvalues (ascending) and axes of one distribution.
fn shape_eigen(dist: &Esag) -> (Vec<f64>, DMatrix<f64>) {
    let fr = dist.frame();
    let d = fr.d;
    let mut idx: Vec<usize> = (0..d - 1).collect();
    idx.sort_by(|&a, &b| fr.lambda[a].total_cmp(&fr.lambda[b]));
    let axes = DMatrix::from_fn(d, d - 1, |r, c| fr.col(idx[c])[r]);
    (idx.iter().map(|&j| fr.lambda[j]).collect(), axes)
}

/// Affine-invariant (geodesic) distance `‖log(V^{−1/2} W V^{−1/2})‖_F` of
/// two symmetric positive-definite matrices.
pub fn geodesic_distance(v: &DMatrix<f64>, w: &DMatrix<f64>) -> f64 {
    let Some(ch) = v.clone().cholesky() else {
        return f64::NAN;
    };
    let l = ch.l();
    let Some(li) = l.try_inverse() else {
        return f64::NAN;
    };
    let m = &li * w * li.transpose();
    let m = (&m + m.transpose()) * 0.5;
    nalgebra::SymmetricEigen::new(m)
        .eigenvalues
        .iter()
        .map(|e| e.ln().powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Basic interval `[2θ̂ − q_hi, 2θ̂ − q_lo]` and the bootstrap SD.
fn basic(est: f64, boot: &[f64], level: f64) -> (f64, f64, f64) {
    if boot.len() < 2 {
        return (f64::NAN, f64::NAN, f64::NAN);
    }
    let a = (1.0 - level) / 2.0;
    let q = quantiles_type7(boot, &[a, 1.0 - a]);
    let m = boot.iter().sum::<f64>() / boot.len() as f64;
    let sd = (boot.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (boot.len() - 1) as f64).sqrt();
    (2.0 * est - q[1], 2.0 * est - q[0], sd)
}

/// `V̂(x)` at each row of `x_new` (`m × n_features`) with pairs-bootstrap
/// intervals from `n_boot` refits on `(x, y)` (`n_boot = 0`: estimates only,
/// intervals `NaN`).
pub fn covariance_intervals(
    fit: &EsagRegressorFit,
    x: &DMatrix<f64>,
    y: &DMatrix<f64>,
    x_new: &DMatrix<f64>,
    n_boot: usize,
    level: f64,
    seed: u64,
) -> Result<Vec<CovarianceEstimate>> {
    if !(level > 0.0 && level < 1.0) {
        return Err(EsagError::Data(format!(
            "level must be in (0, 1), got {level}"
        )));
    }
    let est = fit.predict_distribution(x_new)?;
    let refits = if n_boot > 0 {
        bootstrap_refits(fit, x, y, n_boot, seed)
    } else {
        Vec::new()
    };
    let boots: Vec<Vec<Esag>> = refits
        .iter()
        .filter_map(|g| g.predict_distribution(x_new).ok())
        .collect();
    let d = fit.d;
    let out = est
        .iter()
        .enumerate()
        .map(|(i, dist)| {
            let v = dist.covariance();
            let (eig, axes) = shape_eigen(dist);
            let bv: Vec<DMatrix<f64>> = boots.iter().map(|b| b[i].covariance()).collect();
            let be: Vec<Vec<f64>> = boots.iter().map(|b| shape_eigen(&b[i]).0).collect();
            let (mut eig_lower, mut eig_upper, mut eig_se) = (vec![], vec![], vec![]);
            for k in 0..d - 1 {
                let col: Vec<f64> = be.iter().map(|e| e[k]).collect();
                let (lo, hi, sd) = basic(eig[k], &col, level);
                eig_lower.push(lo);
                eig_upper.push(hi);
                eig_se.push(sd);
            }
            let mut entry_se = DMatrix::from_element(d, d, f64::NAN);
            let mut entry_lower = entry_se.clone();
            let mut entry_upper = entry_se.clone();
            for a in 0..d {
                for c in 0..d {
                    let col: Vec<f64> = bv.iter().map(|m| m[(a, c)]).collect();
                    let (lo, hi, sd) = basic(v[(a, c)], &col, level);
                    entry_lower[(a, c)] = lo;
                    entry_upper[(a, c)] = hi;
                    entry_se[(a, c)] = sd;
                }
            }
            let dists: Vec<f64> = bv.iter().map(|w| geodesic_distance(&v, w)).collect();
            let radius = if dists.len() >= 2 {
                quantiles_type7(&dists, &[level])[0]
            } else {
                f64::NAN
            };
            CovarianceEstimate {
                x: x_new.row(i).iter().copied().collect(),
                covariance: v,
                mean_direction: dist.mean_direction(),
                eigenvalues: eig,
                axes,
                eig_lower,
                eig_upper,
                eig_se,
                entry_se,
                entry_lower,
                entry_upper,
                radius,
                level,
                n_boot: boots.len(),
            }
        })
        .collect();
    Ok(out)
}

impl CovarianceEstimate {
    /// statsmodels-style summary: the shape eigenvalues with bootstrap SEs
    /// and basic intervals, the principal axes, and `V̂` with entry SEs.
    pub fn summary(&self) -> String {
        use crate::summary::{WIDTH, header, num};
        use std::fmt::Write as _;
        let d = self.covariance.nrows();
        let fmt_vec = |v: &[f64]| {
            v.iter()
                .map(|x| format!("{x:.3}"))
                .collect::<Vec<_>>()
                .join(", ")
        };
        let left = vec![
            ("Covariates x:".to_string(), fmt_vec(&self.x)),
            ("Mean direction:".to_string(), fmt_vec(&self.mean_direction)),
        ];
        let right = vec![
            ("Bootstrap (pairs):".to_string(), self.n_boot.to_string()),
            (
                format!("Geodesic {:.0}% radius:", 100.0 * self.level),
                num(self.radius),
            ),
        ];
        let mut s = header("ESAG Covariance Estimate V(x)", &left, &right);
        let lo = format!("[{:.3}", (1.0 - self.level) / 2.0);
        let hi = format!("{:.3}]", 1.0 - (1.0 - self.level) / 2.0);
        let _ = writeln!(s, "{}", "=".repeat(WIDTH));
        let _ = writeln!(
            s,
            "{:<12}{:>10}{:>10}{:>10}{:>10}   principal axis",
            "eigenvalue", "estimate", "std err", lo, hi
        );
        let _ = writeln!(s, "{}", "-".repeat(WIDTH));
        for k in 0..d - 1 {
            let axis: Vec<f64> = (0..d).map(|r| self.axes[(r, k)]).collect();
            let _ = writeln!(
                s,
                "{:<12}{:>10.4}{:>10.4}{:>10.4}{:>10.4}   ({})",
                format!("lambda[{}]", k + 1),
                self.eigenvalues[k],
                self.eig_se[k],
                self.eig_lower[k],
                self.eig_upper[k],
                fmt_vec(&axis)
            );
        }
        let _ = writeln!(
            s,
            "{:<12}{:>10.4}{:>10}{:>10}{:>10}   (mean direction; Vμ = μ)",
            "along mu", 1.0, "", "", ""
        );
        let _ = writeln!(s, "{}", "-".repeat(WIDTH));
        let _ = writeln!(s, "V (bootstrap standard errors in parentheses)");
        for a in 0..d {
            let row: Vec<String> = (0..d)
                .map(|c| {
                    format!(
                        "{:>8.4} ({:.4})",
                        self.covariance[(a, c)],
                        self.entry_se[(a, c)]
                    )
                })
                .collect();
            let _ = writeln!(s, "  {}", row.join(" "));
        }
        let _ = writeln!(s, "{}", "=".repeat(WIDTH));
        s
    }
}
