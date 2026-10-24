//! LASSO-type penalized maximum likelihood for ESAG (Yu & Huang, "LASSO-based
//! inference for elliptically symmetric angular Gaussian distributions").
//!
//! For i.i.d. unit vectors the estimator minimises
//!
//! ```text
//! −ℓ(μ, γ) + λ Σ_g ‖γ̃_g‖^{−r} ‖γ_g‖₂
//! ```
//!
//! over `(μ, γ)`, where `ℓ` is the joint log-likelihood (a sum), `γ̃` the
//! unpenalized MLE, and the groups `g` are the single entries of `γ`
//! (elementwise penalties) or the groups `γ_(j)` of the parameterization
//! (group penalties). This is the paper's `n·Q_n` with `λ = n λ_n`, so the
//! paper's grid `λ ∈ {1, 2, 4, …, 256}` is used as is. The six methods are
//!
//! | method     | groups          | `r`              |
//! |------------|-----------------|------------------|
//! | LASSO      | entries         | 0                |
//! | aLASSO     | entries         | 1                |
//! | RoC-LASSO  | entries         | `‖μ̃‖/‖μ‖`        |
//! | GLASSO     | `γ_(j)`         | 0                |
//! | aGLASSO    | `γ_(j)`         | 1                |
//! | RoC-GLASSO | `γ_(j)`         | `‖μ̃‖/‖μ‖`        |
//!
//! With RoC weights the exponent depends on the `μ` being estimated; the
//! solver ([`crate::optim::prox`]) handles that exactly. Entries (groups)
//! with `γ̃ = 0` get an infinite adaptive weight and stay at zero.
//!
//! **Caveat.** Exact zeros of a whole group `j ≥ 2`, or of its last two
//! entries, lie where `γ ↦ V` is discontinuous (`docs/MATH.md`, "Singular
//! set"); there `V` is defined by the convention `atan2(0, 0) = 0` of the
//! original R code, which `esag` follows.

use crate::dims;
use crate::distribution::Esag;
use crate::error::{EsagError, Result};
use crate::estimator::{EsagEstimator, EsagFit};
use crate::optim::prox::{Exponent, GroupPenalty, ProxConfig, prox_newton};
use crate::regression::{RegressionData, SmoothNegLogLik};
use nalgebra::DMatrix;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Which parameters share a penalty term.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Grouping {
    /// One term per entry of `γ` (LASSO family).
    Elementwise,
    /// One term per group `γ_(j)`, `j = 1..d−2` (group LASSO family).
    Groups,
    /// One term for all of `γ` (eq. 4.4 of the paper; zero ⇔ isotropy).
    Whole,
}

/// The exponent `r` of the adaptive weights `‖γ̃_g‖^{−r}`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum Weighting {
    /// `r = 0`: the plain (group) LASSO.
    Plain,
    /// A fixed `r` (the adaptive (group) LASSO uses `r = 1`).
    Adaptive(f64),
    /// `r = RoC = ‖μ̃‖/‖μ‖`, the ratio of concentration.
    RoC,
}

/// A penalty: grouping × weighting.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Penalty {
    pub grouping: Grouping,
    pub weighting: Weighting,
}

impl Penalty {
    pub const LASSO: Penalty = Penalty {
        grouping: Grouping::Elementwise,
        weighting: Weighting::Plain,
    };
    pub const ALASSO: Penalty = Penalty {
        grouping: Grouping::Elementwise,
        weighting: Weighting::Adaptive(1.0),
    };
    pub const ROC_LASSO: Penalty = Penalty {
        grouping: Grouping::Elementwise,
        weighting: Weighting::RoC,
    };
    pub const GLASSO: Penalty = Penalty {
        grouping: Grouping::Groups,
        weighting: Weighting::Plain,
    };
    pub const AGLASSO: Penalty = Penalty {
        grouping: Grouping::Groups,
        weighting: Weighting::Adaptive(1.0),
    };
    pub const ROC_GLASSO: Penalty = Penalty {
        grouping: Grouping::Groups,
        weighting: Weighting::RoC,
    };

    /// The six methods of the paper, in its order.
    pub fn paper_methods() -> [(&'static str, Penalty); 6] {
        [
            ("LASSO", Self::LASSO),
            ("aLASSO", Self::ALASSO),
            ("RoC-LASSO", Self::ROC_LASSO),
            ("GLASSO", Self::GLASSO),
            ("aGLASSO", Self::AGLASSO),
            ("RoC-GLASSO", Self::ROC_GLASSO),
        ]
    }
}

/// How the penalty parameter is chosen along the path.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum Selection {
    /// Minimise `BIC = −2ℓ + log(n)·df`.
    Bic,
    /// Minimise `AIC = −2ℓ + 2·df`.
    Aic,
    /// `folds`-fold cross-validation of the prediction RMSE (eq. 5.5).
    CrossValidation { folds: usize, seed: u64 },
}

/// Penalized ESAG estimator for i.i.d. data (scikit-learn style: configure,
/// then [`PenalizedEsag::fit`]).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PenalizedEsag {
    pub penalty: Penalty,
    /// Penalty parameters on the sum-log-likelihood scale (`λ = n λ_n`),
    /// fitted in increasing order with warm starts. Default `2^0 … 2^8`.
    pub lambdas: Vec<f64>,
    pub selection: Selection,
    /// KKT-residual tolerance of the proximal Newton solver. Default 1e-6.
    pub tol: f64,
    /// Evaluate the likelihood in parallel over observations.
    pub parallel: bool,
}

impl PenalizedEsag {
    /// The paper's defaults: `λ ∈ {1, 2, 4, …, 256}`, BIC selection.
    pub fn new(penalty: Penalty) -> Self {
        Self {
            penalty,
            lambdas: (0..=8).map(|k| f64::from(1u32 << k)).collect(),
            selection: Selection::Bic,
            tol: 1e-6,
            parallel: true,
        }
    }
    pub fn with_lambdas(mut self, lambdas: Vec<f64>) -> Self {
        self.lambdas = lambdas;
        self
    }
    pub fn with_selection(mut self, s: Selection) -> Self {
        self.selection = s;
        self
    }
    pub fn with_parallel(mut self, v: bool) -> Self {
        self.parallel = v;
        self
    }

    /// Fit the whole λ path to `y` (`n × d` unit vectors) and select λ.
    pub fn fit(&self, y: &DMatrix<f64>) -> Result<PenalizedPath> {
        if self.lambdas.is_empty() || self.lambdas.iter().any(|&l| !(l >= 0.0)) {
            return Err(EsagError::Data(
                "lambdas must be non-empty and non-negative".into(),
            ));
        }
        let initial = EsagEstimator::new().with_hessian(false).fit(y)?;
        let mut points = self.path(y, &initial)?;
        let (selected, criterion) = match self.selection {
            Selection::Bic => {
                let v: Vec<f64> = points.iter().map(|p| p.bic).collect();
                (argmin(&v), v)
            }
            Selection::Aic => {
                let v: Vec<f64> = points.iter().map(|p| p.aic).collect();
                (argmin(&v), v)
            }
            Selection::CrossValidation { folds, seed } => {
                let v = self.cv_rmse(y, folds, seed)?;
                for (p, &c) in points.iter_mut().zip(&v) {
                    p.cv_rmse = Some(c);
                }
                (argmin(&v), v)
            }
        };
        Ok(PenalizedPath {
            penalty: self.penalty,
            selection: self.selection,
            initial,
            points,
            criterion,
            selected,
        })
    }

    /// Penalized fits along the λ grid (ascending, warm-started).
    fn path(&self, y: &DMatrix<f64>, initial: &EsagFit) -> Result<Vec<PathPoint>> {
        let d = y.ncols();
        let n = y.nrows();
        let x0 = DMatrix::<f64>::zeros(n, 0);
        let data = RegressionData::new(&x0, y, true)?;
        let bd = dims::eta_len(d);
        let free: Vec<(usize, usize)> = (0..bd).map(|k| (k, 0)).collect();
        let base = DMatrix::<f64>::zeros(bd, 1);
        let obj = SmoothNegLogLik::new(&data, base, &free, self.parallel);
        let mu_t = initial.mu().to_vec();
        let gamma_t = initial.gamma().to_vec();
        let groups = penalty_groups(d, self.penalty.grouping);
        let bases: Vec<f64> = groups
            .iter()
            .map(|g| {
                g.iter()
                    .map(|&i| gamma_t[i - d].powi(2))
                    .sum::<f64>()
                    .sqrt()
            })
            .collect();
        let exponent = match self.penalty.weighting {
            Weighting::Plain => Exponent::Fixed(0.0),
            Weighting::Adaptive(r) => Exponent::Fixed(r),
            Weighting::RoC => Exponent::RoC {
                mu: (0..d).collect(),
                mu_tilde_norm: mu_t.iter().map(|v| v * v).sum::<f64>().sqrt(),
            },
        };
        let mut order: Vec<usize> = (0..self.lambdas.len()).collect();
        order.sort_by(|&a, &b| self.lambdas[a].total_cmp(&self.lambdas[b]));
        let mut theta: Vec<f64> = mu_t.iter().chain(&gamma_t).copied().collect();
        let cfg = ProxConfig {
            tol: self.tol,
            ..ProxConfig::default()
        };
        let mut out: Vec<Option<PathPoint>> = vec![None; self.lambdas.len()];
        for &k in &order {
            let pen = GroupPenalty {
                groups: groups.clone(),
                base: bases.clone(),
                exponent: exponent.clone(),
                lambda: self.lambdas[k],
            };
            let res = prox_newton(&obj, &pen, &theta, &cfg);
            theta = res.theta.clone();
            let ll = -res.smooth;
            let nnz = theta[d..].iter().filter(|v| **v != 0.0).count();
            let df = (d + nnz) as f64;
            out[k] = Some(PathPoint {
                lambda: self.lambdas[k],
                mu: theta[..d].to_vec(),
                gamma: theta[d..].to_vec(),
                log_likelihood: ll,
                objective: res.objective,
                df,
                aic: -2.0 * ll + 2.0 * df,
                bic: -2.0 * ll + (n as f64).ln() * df,
                cv_rmse: None,
                kkt: res.kkt,
                converged: res.converged,
                n_iter: res.n_iter,
            });
        }
        Ok(out
            .into_iter()
            .map(|p| p.expect("every lambda fitted"))
            .collect())
    }

    /// Mean held-out RMSE (eq. 5.5) per λ over `folds` folds.
    fn cv_rmse(&self, y: &DMatrix<f64>, folds: usize, seed: u64) -> Result<Vec<f64>> {
        let n = y.nrows();
        if folds < 2 || folds > n {
            return Err(EsagError::Data(format!("need 2 ≤ folds ≤ n, got {folds}")));
        }
        // deterministic shuffled fold assignment
        let mut idx: Vec<usize> = (0..n).collect();
        let mut rng = crate::stream_rng(seed, crate::streams::CV_FOLDS, 0);
        use rand::seq::SliceRandom;
        idx.shuffle(&mut rng);
        let fold_of: Vec<usize> = {
            let mut f = vec![0; n];
            for (pos, &i) in idx.iter().enumerate() {
                f[i] = pos % folds;
            }
            f
        };
        let per_fold: Vec<Result<Vec<f64>>> = (0..folds)
            .into_par_iter()
            .map(|k| {
                let train: Vec<usize> = (0..n).filter(|&i| fold_of[i] != k).collect();
                let test: Vec<usize> = (0..n).filter(|&i| fold_of[i] == k).collect();
                let yt = DMatrix::from_fn(train.len(), y.ncols(), |i, j| y[(train[i], j)]);
                let yv = DMatrix::from_fn(test.len(), y.ncols(), |i, j| y[(test[i], j)]);
                let inner = self.clone().with_parallel(false);
                let initial = EsagEstimator::new().with_hessian(false).fit(&yt)?;
                let pts = inner.path(&yt, &initial)?;
                pts.iter()
                    .map(|p| Ok(prediction_rmse(&Esag::new(&p.mu, &p.gamma)?, &yv)))
                    .collect()
            })
            .collect();
        let mut sum = vec![0.0; self.lambdas.len()];
        for r in per_fold {
            for (s, v) in sum.iter_mut().zip(r?) {
                *s += v / folds as f64;
            }
        }
        Ok(sum)
    }
}

/// Index groups (into `η = (μ, γ)`) for a grouping at dimension `d`.
pub fn penalty_groups(d: usize, grouping: Grouping) -> Vec<Vec<usize>> {
    let g = dims::gamma_len(d);
    match grouping {
        Grouping::Elementwise => (d..d + g).map(|i| vec![i]).collect(),
        Grouping::Whole => vec![(d..d + g).collect()],
        Grouping::Groups => (1..=d - 2)
            .map(|j| {
                let start = d + j * (j + 1) / 2 - 1;
                (start..start + j + 1).collect()
            })
            .collect(),
    }
}

fn argmin(v: &[f64]) -> usize {
    let mut best = 0;
    for (i, &x) in v.iter().enumerate() {
        if x < v[best] || (v[best].is_nan() && !x.is_nan()) {
            best = i;
        }
    }
    best
}

/// One penalized fit on the λ path.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PathPoint {
    pub lambda: f64,
    pub mu: Vec<f64>,
    /// Penalized `γ` (exact zeros where the penalty set them).
    pub gamma: Vec<f64>,
    /// Joint log-likelihood at the penalized estimate.
    pub log_likelihood: f64,
    /// Penalized objective `−ℓ + penalty`.
    pub objective: f64,
    /// Degrees of freedom: `d` + number of nonzero `γ` entries.
    pub df: f64,
    pub aic: f64,
    pub bic: f64,
    /// Cross-validated prediction RMSE (when selected by CV).
    pub cv_rmse: Option<f64>,
    /// KKT residual of the penalized problem at the solution.
    pub kkt: f64,
    pub converged: bool,
    pub n_iter: usize,
}

impl PathPoint {
    /// The fitted distribution.
    pub fn distribution(&self) -> Result<Esag> {
        Esag::new(&self.mu, &self.gamma)
    }
    /// Indices of the zero entries of `γ`.
    pub fn zeros(&self) -> Vec<usize> {
        (0..self.gamma.len())
            .filter(|&i| self.gamma[i] == 0.0)
            .collect()
    }
}

/// A fitted λ path with the selected point.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PenalizedPath {
    pub penalty: Penalty,
    pub selection: Selection,
    /// The unpenalized MLE `(μ̃, γ̃)` the adaptive weights are built from.
    pub initial: EsagFit,
    /// One point per λ, in the order of `lambdas`.
    pub points: Vec<PathPoint>,
    /// The selection criterion per λ (BIC, AIC or CV RMSE).
    pub criterion: Vec<f64>,
    /// Index of the selected λ.
    pub selected: usize,
}

impl PenalizedPath {
    /// The selected fit.
    pub fn best(&self) -> &PathPoint {
        &self.points[self.selected]
    }

    /// A text table of the path.
    pub fn summary(&self) -> String {
        let mut s = format!(
            "Penalized ESAG: {:?} / {:?}, selection {:?}\n{:>8} {:>12} {:>6} {:>12} {:>12} {:>10}\n",
            self.penalty.grouping,
            self.penalty.weighting,
            self.selection,
            "lambda",
            "loglik",
            "df",
            "BIC",
            "criterion",
            "kkt"
        );
        for (i, p) in self.points.iter().enumerate() {
            s += &format!(
                "{:>8.3} {:>12.4} {:>6} {:>12.4} {:>12.6} {:>10.1e}{}\n",
                p.lambda,
                p.log_likelihood,
                p.df,
                p.bic,
                self.criterion[i],
                p.kkt,
                if i == self.selected {
                    "  <- selected"
                } else {
                    ""
                }
            );
        }
        let b = self.best();
        s += &format!(
            "selected mu    = {:?}\nselected gamma = {:?}\n",
            round(&b.mu),
            round(&b.gamma)
        );
        s
    }
}

fn round(v: &[f64]) -> Vec<f64> {
    v.iter().map(|x| (x * 1e4).round() / 1e4).collect()
}

/// Prediction RMSE of eq. (5.5): `sqrt(mean_i r_iᵀ V̂⁻¹ r_i)` with the
/// directional residual `r_i = (I − ŷŷᵀ) y_i`, `ŷ = μ̂/‖μ̂‖`.
pub fn prediction_rmse(dist: &Esag, y: &DMatrix<f64>) -> f64 {
    let vinv = dist.covariance_inv();
    let yhat = dist.mean_direction();
    let d = y.ncols();
    let mut acc = 0.0;
    for i in 0..y.nrows() {
        let yi: Vec<f64> = (0..d).map(|k| y[(i, k)]).collect();
        let c: f64 = yi.iter().zip(&yhat).map(|(a, b)| a * b).sum();
        let r: Vec<f64> = (0..d).map(|k| yi[k] - c * yhat[k]).collect();
        let mut q = 0.0;
        for a in 0..d {
            for b in 0..d {
                q += r[a] * vinv[(a, b)] * r[b];
            }
        }
        acc += q;
    }
    (acc / y.nrows() as f64).sqrt()
}

/// Selection metrics of the paper (Section 5.1) for an estimate against the
/// truth: correction rate, sensitivity (share of true zeros found) and
/// specificity (share of true nonzeros kept).
pub fn selection_rates(gamma_true: &[f64], gamma_hat: &[f64]) -> (f64, f64, f64) {
    let (mut z_ok, mut z_tot, mut nz_ok, mut nz_tot) = (0usize, 0usize, 0usize, 0usize);
    for (&t, &h) in gamma_true.iter().zip(gamma_hat) {
        if t == 0.0 {
            z_tot += 1;
            z_ok += usize::from(h == 0.0);
        } else {
            nz_tot += 1;
            nz_ok += usize::from(h != 0.0);
        }
    }
    let cr = (z_ok + nz_ok) as f64 / gamma_true.len() as f64;
    let se = if z_tot > 0 {
        z_ok as f64 / z_tot as f64
    } else {
        f64::NAN
    };
    let sp = if nz_tot > 0 {
        nz_ok as f64 / nz_tot as f64
    } else {
        f64::NAN
    };
    (cr, se, sp)
}
