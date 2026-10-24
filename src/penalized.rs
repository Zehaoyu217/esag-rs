//! LASSO-type penalized ESAG fits (Yu & Huang, *LASSO-based inference for
//! elliptically symmetric angular Gaussian distributions*, "Project 3"),
//! with the corrections of `docs/project3/METHODOLOGY.md`.
//!
//! All objectives are on the sum scale of the R code,
//! `F = −Σ_i ℓ_i + λ·P`, so `λ` is the paper's `n·λ_n`. `μ` (and the
//! intercept column in regression) is never penalized.
//!
//! # Methods
//!
//! **Recommended (smooth log-tangent shape, [`crate::regression::Shape::LogTangent`]).**
//! The paper's `γ ↦ V` map is discontinuous at exact zeros of any group
//! above group 1 and at vanishing trailing pairs (`docs/MATH.md`), so
//! sparsity is placed on the log-tangent shape `S` instead:
//!
//! * [`Method::Isotropy`]: one group with all shape coefficients, `w‖s‖`
//!   (the paper's eq. 4.4); zero ⇔ isotropic errors.
//! * [`Method::EigenGap`] (i.i.d.): `Σ_j w_j (σ_(j+1) − σ_(j))` on the sorted
//!   eigenvalues of `S`; a zero gap is a tie `ξ_j = ξ_{j+1}` anywhere in the
//!   spectrum (partial isotropy), frame-free. This is what the paper's group
//!   LASSO meant `γ_(j) = 0` to express (`‖γ_(j)‖ = e^{gap_j} − 1`).
//! * [`Method::Drivers`] (regression): one group per covariate for its
//!   `μ`-slopes and/or its shape slopes, on standardized covariates
//!   (variable selection of drivers).
//!
//! These are solved exactly by proximal Newton ([`crate::optim::prox`]):
//! exact zeros, no thresholding.
//!
//! **Paper reproduction (`γ`, i.i.d.).** [`Method::PaperLasso`] and
//! [`Method::PaperGroupLasso`] with [`Weighting::Plain`], `Adaptive` and
//! `RoC` are the six methods as coded in the R scripts: penalties on the
//! entries / groups `γ_(j)` of `γ`, weights `b^r` with `b = 1/|γ̃|`,
//! `r = max(1, ‖μ̃‖/‖μ‖)` inside the objective for RoC. Solved by smoothing
//! continuation (`|x| → √(x² + ε²)`, ε = 1e−2 … 1e−8) with exact-gradient
//! L-BFGS, then the R code's post-hoc rule `|x| < 0.01 → 0` and its
//! BIC/AIC/SBIC (df = nonzero entries including `μ`). Reproduction only:
//! those zeros can sit on discontinuities of `γ ↦ V`.
//!
//! # Selection
//!
//! BIC (default) or AIC at the **relaxed** refit (the unpenalized MLE on
//! the selected model) with the model dimension as df, or K-fold
//! cross-validation (shuffled, seeded folds) of the held-out
//! log-likelihood (or the paper's RMSE, eq. 5.5).

use crate::dims;
use crate::distribution::Esag;
use crate::error::{EsagError, Result};
use crate::optim::lbfgs::{LbfgsConfig, Objective, minimize};
use crate::optim::prox::{
    EigenGaps, Exponent, GroupNorms, Penalty, PenaltyAtoms, ProxConfig, RocMode, SmoothObjective,
};
use crate::regression::{
    EsagRegressor, EsagRegressorFit, Init, RegressionData, Shape, SmoothNegLogLik, isotropic_mask,
};
use nalgebra::DMatrix;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// The penalty structure.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Method {
    /// All shape coefficients as one group (log-tangent): zero ⇔ isotropy.
    Isotropy,
    /// Gaps between the sorted eigenvalues of `S` (log-tangent, i.i.d.).
    EigenGap,
    /// Per-covariate groups of slopes (log-tangent regression): the `μ`
    /// slopes of each covariate (`mean`) and/or its shape slopes (`shape`).
    Drivers { mean: bool, shape: bool },
    /// The paper's elementwise penalty on `γ` (reproduction, i.i.d.).
    PaperLasso,
    /// The paper's group penalty on `γ_(1..d−2)` (reproduction, i.i.d.).
    PaperGroupLasso,
}

impl Method {
    fn is_paper(self) -> bool {
        matches!(self, Method::PaperLasso | Method::PaperGroupLasso)
    }
    fn shape(self) -> Shape {
        if self.is_paper() {
            Shape::Gamma
        } else {
            Shape::LogTangent
        }
    }
}

/// Adaptive weights `b_g^r`, `b_g = 1/φ_g(θ̃)` from the unpenalized fit.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Weighting {
    /// `r = 0` (for `EigenGap`: the concave baseline `w_j = j(k−j)/k`).
    Plain,
    /// `r = 1`.
    Adaptive,
    /// `r = max(1, ‖μ̃‖/‖μ‖)`, the ratio of concentration (i.i.d. only).
    RoC(RocMode),
}

/// The λ grid (sum scale).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum LambdaGrid {
    /// The paper's grid `0, 0.5, 1, 2, 4, …, 256`.
    Paper,
    /// `count` values from `λ_max` (everything penalized is zero) down to
    /// `ratio·λ_max`, geometrically, plus `0`.
    Auto { count: usize, ratio: f64 },
    /// Explicit values.
    Custom(Vec<f64>),
}

/// Cross-validation score.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CvScore {
    /// Held-out log-likelihood (a proper scoring rule; recommended).
    LogLikelihood,
    /// The paper's RMSE (eq. 5.5); ignores `‖μ̂‖`.
    Rmse,
}

/// How λ is selected along the path.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum Selection {
    Bic,
    Aic,
    CrossValidation {
        folds: usize,
        seed: u64,
        score: CvScore,
    },
}

/// Penalized ESAG estimator (configure, then [`PenalizedEsag::fit`]).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PenalizedEsag {
    pub method: Method,
    pub weighting: Weighting,
    pub grid: LambdaGrid,
    pub selection: Selection,
    /// Compute BIC/AIC at the relaxed refit (default `true`; ignored by the
    /// paper methods, which use the R code's rule).
    pub relax: bool,
    /// Stationarity tolerance of the solver (sum scale). Default 1e-6.
    pub tol: f64,
    /// Parallel likelihood evaluation.
    pub parallel: bool,
    /// Paper methods: unpenalized estimate (`b_dim × 1`) to build the weights
    /// from, e.g. R's `unp_mle$par` (default: `esag`'s MLE).
    pub initial_coef: Option<DMatrix<f64>>,
    /// Paper methods: start every λ from this (e.g. R's `truth + 0.1`)
    /// instead of warm-starting along the path.
    pub start_coef: Option<DMatrix<f64>>,
}

impl PenalizedEsag {
    /// Defaults: adaptive weights (the paper's methods use the paper grid,
    /// the corrected ones an automatic grid), BIC at the relaxed refit.
    pub fn new(method: Method) -> Self {
        Self {
            method,
            weighting: Weighting::Adaptive,
            grid: if method.is_paper() {
                LambdaGrid::Paper
            } else {
                LambdaGrid::Auto {
                    count: 30,
                    ratio: 1e-3,
                }
            },
            selection: Selection::Bic,
            relax: true,
            tol: 1e-6,
            parallel: true,
            initial_coef: None,
            start_coef: None,
        }
    }
    /// One of the paper's six methods by name: `LASSO`, `aLASSO`,
    /// `RoC-LASSO`, `GLASSO`, `aGLASSO`, `RoC-GLASSO`.
    pub fn paper(name: &str) -> Result<Self> {
        let (m, w) = match name {
            "LASSO" => (Method::PaperLasso, Weighting::Plain),
            "aLASSO" => (Method::PaperLasso, Weighting::Adaptive),
            "RoC-LASSO" => (Method::PaperLasso, Weighting::RoC(RocMode::Joint)),
            "GLASSO" => (Method::PaperGroupLasso, Weighting::Plain),
            "aGLASSO" => (Method::PaperGroupLasso, Weighting::Adaptive),
            "RoC-GLASSO" => (Method::PaperGroupLasso, Weighting::RoC(RocMode::Joint)),
            other => return Err(EsagError::Data(format!("unknown paper method {other:?}"))),
        };
        Ok(Self::new(m).with_weighting(w))
    }
    pub fn with_weighting(mut self, w: Weighting) -> Self {
        self.weighting = w;
        self
    }
    pub fn with_grid(mut self, g: LambdaGrid) -> Self {
        self.grid = g;
        self
    }
    pub fn with_selection(mut self, s: Selection) -> Self {
        self.selection = s;
        self
    }
    pub fn with_relax(mut self, v: bool) -> Self {
        self.relax = v;
        self
    }
    pub fn with_parallel(mut self, v: bool) -> Self {
        self.parallel = v;
        self
    }
    pub fn with_initial_coef(mut self, b: Option<DMatrix<f64>>) -> Self {
        self.initial_coef = b;
        self
    }
    pub fn with_start_coef(mut self, b: Option<DMatrix<f64>>) -> Self {
        self.start_coef = b;
        self
    }

    /// Fit the λ path to `(x, y)` (`x` may have zero columns: i.i.d.) and
    /// select λ.
    pub fn fit(&self, x: &DMatrix<f64>, y: &DMatrix<f64>) -> Result<PenalizedPath> {
        let prep = Prepared::new(self, x, y)?;
        let mut points = prep.path(self)?;
        let criterion: Vec<f64> = match self.selection {
            Selection::Bic => points.iter().map(|p| p.bic).collect(),
            Selection::Aic => points.iter().map(|p| p.aic).collect(),
            Selection::CrossValidation { folds, seed, score } => {
                let v = self.cv(x, y, &prep.lambdas, folds, seed, score)?;
                for (p, &c) in points.iter_mut().zip(&v) {
                    p.cv = Some(c);
                }
                v
            }
        };
        // smallest criterion; ties (e.g. equal relaxed fits) → fewer active
        // atoms, then the smallest λ (least shrinkage; R's which.min)
        let n_active = |p: &PathPoint| p.active.iter().filter(|&&a| a).count();
        let mut selected = 0;
        for (i, &c) in criterion.iter().enumerate() {
            let best = criterion[selected];
            let tol = 1e-9 * best.abs().max(1.0);
            let tie = (c - best).abs() <= tol;
            let (pi, ps) = (&points[i], &points[selected]);
            if c < best - tol
                || (tie && n_active(pi) < n_active(ps))
                || (tie && n_active(pi) == n_active(ps) && pi.lambda < ps.lambda)
            {
                selected = i;
            }
        }
        let relaxed = if self.method.is_paper() {
            None
        } else {
            prep.relaxed_fit(&points[selected]).ok()
        };
        Ok(PenalizedPath {
            method: self.method,
            weighting: self.weighting,
            selection: self.selection,
            initial: prep.initial,
            points,
            criterion,
            selected,
            relaxed,
        })
    }

    /// The penalized log-likelihood `Σ ℓ − λ·P` at coefficients `coef`
    /// (what the R code maximises), with this configuration's weights.
    pub fn penalized_loglik(
        &self,
        x: &DMatrix<f64>,
        y: &DMatrix<f64>,
        coef: &DMatrix<f64>,
        lambda: f64,
    ) -> Result<f64> {
        let prep = Prepared::new(self, x, y)?;
        let theta = prep.to_standardized(coef);
        let obj = SmoothNegLogLik::new(
            &prep.data,
            DMatrix::zeros(prep.data.b_dim, prep.data.p_total),
            &prep.free,
            self.parallel,
        );
        let mut g = vec![0.0; theta.len()];
        let f = obj.value_grad(&theta, &mut g);
        Ok(-(f + prep.penalty(lambda).value(&theta)))
    }

    /// Mean held-out score per λ over shuffled, seeded folds (lower is
    /// better: negative mean log-likelihood, or RMSE).
    fn cv(
        &self,
        x: &DMatrix<f64>,
        y: &DMatrix<f64>,
        lambdas: &[f64],
        folds: usize,
        seed: u64,
        score: CvScore,
    ) -> Result<Vec<f64>> {
        let n = y.nrows();
        if folds < 2 || folds > n {
            return Err(EsagError::Data(format!("need 2 ≤ folds ≤ n, got {folds}")));
        }
        let mut idx: Vec<usize> = (0..n).collect();
        let mut rng = crate::stream_rng(seed, crate::streams::CV_FOLDS, 0);
        use rand::seq::SliceRandom;
        idx.shuffle(&mut rng);
        let mut fold_of = vec![0; n];
        for (pos, &i) in idx.iter().enumerate() {
            fold_of[i] = pos % folds;
        }
        let rows = |m: &DMatrix<f64>, keep: &[usize]| {
            DMatrix::from_fn(keep.len(), m.ncols(), |i, j| m[(keep[i], j)])
        };
        let per_fold: Vec<Result<Vec<f64>>> = (0..folds)
            .into_par_iter()
            .map(|k| {
                let train: Vec<usize> = (0..n).filter(|&i| fold_of[i] != k).collect();
                let test: Vec<usize> = (0..n).filter(|&i| fold_of[i] == k).collect();
                let mut inner = self.clone().with_parallel(false);
                inner.grid = LambdaGrid::Custom(lambdas.to_vec());
                let prep = Prepared::new(&inner, &rows(x, &train), &rows(y, &train))?;
                let pts = prep.path(&inner)?;
                let (xt, yt) = (rows(x, &test), rows(y, &test));
                pts.iter()
                    .map(|p| prep.score(&p.coef, &xt, &yt, score))
                    .collect()
            })
            .collect();
        let mut mean = vec![0.0; lambdas.len()];
        for r in per_fold {
            for (m, v) in mean.iter_mut().zip(r?) {
                *m += v / folds as f64;
            }
        }
        Ok(mean)
    }
}

/// One penalized fit on the λ path.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PathPoint {
    /// λ on the sum scale; the paper's `λ_n = λ/n`.
    pub lambda: f64,
    pub lambda_n: f64,
    /// Coefficients (`b_dim × p_total`, original covariate scale). Paper
    /// methods: the untruncated optimum.
    pub coef: DMatrix<f64>,
    /// Joint log-likelihood at `coef`.
    pub log_likelihood: f64,
    /// `−ℓ + λP` at `coef` (sum scale; the R code maximises its negative).
    pub objective: f64,
    /// Exponent of the adaptive weights at the solution.
    pub r: f64,
    /// Penalty atoms that are nonzero (groups / gaps / entries).
    pub active: Vec<bool>,
    /// Degrees of freedom used by BIC/AIC.
    pub df: f64,
    /// Log-likelihood the criteria use (relaxed refit, or as coded).
    pub criterion_loglik: f64,
    pub bic: f64,
    pub aic: f64,
    /// Cross-validated score (when selected by CV).
    pub cv: Option<f64>,
    /// Stationarity residual of the penalized problem.
    pub kkt: f64,
    pub converged: bool,
    pub n_iter: usize,
    /// Paper methods: the R rule `|x| < 0.01 → 0` applied to `coef`.
    pub coef_thresholded: Option<DMatrix<f64>>,
    /// Paper methods: log-likelihood at the thresholded estimate (a large
    /// drop flags a discontinuity of `γ ↦ V`).
    pub loglik_thresholded: Option<f64>,
    /// Paper methods: SBIC as coded (`log((n+2)/24)` per df).
    pub sbic: Option<f64>,
}

/// A fitted λ path.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PenalizedPath {
    pub method: Method,
    pub weighting: Weighting,
    pub selection: Selection,
    /// The unpenalized fit behind the adaptive weights.
    pub initial: EsagRegressorFit,
    pub points: Vec<PathPoint>,
    /// Selection criterion per λ (lower is better).
    pub criterion: Vec<f64>,
    pub selected: usize,
    /// Relaxed refit at the selected λ, with standard errors where the
    /// selected model is a coefficient mask (corrected methods).
    pub relaxed: Option<EsagRegressorFit>,
}

impl PenalizedPath {
    /// The selected point.
    pub fn best(&self) -> &PathPoint {
        &self.points[self.selected]
    }

    /// A text table of the path.
    pub fn summary(&self) -> String {
        let mut s = format!(
            "Penalized ESAG: {:?}, {:?} weights, selection {:?}\n{:>10} {:>10} {:>12} {:>7} {:>12} {:>10} {:>9}\n",
            self.method,
            self.weighting,
            self.selection,
            "lambda",
            "lambda_n",
            "loglik",
            "df",
            "criterion",
            "kkt",
            "active"
        );
        for (i, p) in self.points.iter().enumerate() {
            s += &format!(
                "{:>10.4} {:>10.2e} {:>12.4} {:>7} {:>12.4} {:>10.1e} {:>9}{}\n",
                p.lambda,
                p.lambda_n,
                p.log_likelihood,
                p.df,
                self.criterion[i],
                p.kkt,
                format!(
                    "{}/{}",
                    p.active.iter().filter(|&&a| a).count(),
                    p.active.len()
                ),
                if i == self.selected {
                    "  <- selected"
                } else {
                    ""
                }
            );
        }
        s
    }
}

/// Data, kernel, weights and atoms shared by a path and its CV folds.
struct Prepared {
    data: RegressionData,
    /// covariate means / standard deviations (regression standardisation)
    center: Vec<f64>,
    scale: Vec<f64>,
    initial: EsagRegressorFit,
    /// initial estimate on the standardised scale (free-parameter order)
    theta_init: Vec<f64>,
    free: Vec<(usize, usize)>,
    atoms: Atoms,
    base: Vec<f64>,
    exponent: Exponent,
    lambdas: Vec<f64>,
    config: EsagRegressor,
}

#[derive(Clone, Debug)]
enum Atoms {
    Groups(GroupNorms),
    Gaps(EigenGaps),
}

impl Atoms {
    fn len(&self) -> usize {
        match self {
            Atoms::Groups(g) => g.len(),
            Atoms::Gaps(g) => g.len(),
        }
    }
    fn atoms(&self, theta: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; self.len()];
        match self {
            Atoms::Groups(g) => g.atoms(theta, &mut out),
            Atoms::Gaps(g) => g.atoms(theta, &mut out),
        }
        out
    }
}

impl Prepared {
    fn new(cfg: &PenalizedEsag, x: &DMatrix<f64>, y: &DMatrix<f64>) -> Result<Self> {
        let d = y.ncols();
        let p = x.ncols();
        let method = cfg.method;
        if (method.is_paper() || method == Method::EigenGap) && p > 0 {
            return Err(EsagError::Data(format!(
                "{method:?} is for i.i.d. data (no covariates)"
            )));
        }
        if matches!(method, Method::Drivers { .. }) && p == 0 {
            return Err(EsagError::Data("Method::Drivers needs covariates".into()));
        }
        if matches!(cfg.weighting, Weighting::RoC(_)) && p > 0 {
            return Err(EsagError::Data(
                "RoC weights are defined for i.i.d. data; use Adaptive in regression".into(),
            ));
        }
        // standardise covariates (mean 0, sd 1)
        let n = x.nrows() as f64;
        let center: Vec<f64> = (0..p).map(|j| x.column(j).sum() / n).collect();
        let scale: Vec<f64> = (0..p)
            .map(|j| {
                let v = x
                    .column(j)
                    .iter()
                    .map(|v| (v - center[j]).powi(2))
                    .sum::<f64>()
                    / n;
                if v > 0.0 { v.sqrt() } else { 1.0 }
            })
            .collect();
        let xs = DMatrix::from_fn(x.nrows(), p, |i, j| (x[(i, j)] - center[j]) / scale[j]);
        let config = EsagRegressor::new()
            .with_shape(method.shape())
            .with_parallel(cfg.parallel)
            .with_hessian(false);
        let initial = match (&cfg.initial_coef, method.is_paper()) {
            (Some(b), true) => config
                .clone()
                .with_max_iter(0)
                .with_init(Init::Custom(b.clone()))
                .fit(&xs, y)?,
            _ => config.fit(&xs, y)?,
        };
        let config = initial.config.clone(); // records the log-tangent frame
        let data = config.data(&xs, y)?;
        let bd = dims::eta_len(d);
        let pt = p + 1;
        let free: Vec<(usize, usize)> =
            (0..pt).flat_map(|j| (0..bd).map(move |k| (k, j))).collect();
        let idx = |k: usize, j: usize| j * bd + k;
        // a user-supplied initial estimate is used verbatim (the paper's
        // weights depend on its coordinates, not only on its V)
        let theta_init: Vec<f64> = match &cfg.initial_coef {
            Some(b) if method.is_paper() => free.iter().map(|&(k, j)| b[(k, j)]).collect(),
            _ => free.iter().map(|&(k, j)| initial.coef[(k, j)]).collect(),
        };
        let g = dims::gamma_len(d);
        let atoms = match method {
            Method::Isotropy => Atoms::Groups(GroupNorms {
                groups: vec![
                    (0..pt)
                        .flat_map(|j| (d..bd).map(move |k| idx(k, j)))
                        .collect(),
                ],
            }),
            Method::EigenGap => Atoms::Gaps(EigenGaps {
                p: d - 1,
                offset: d,
            }),
            Method::Drivers { mean, shape } => {
                let mut groups = vec![];
                for j in 1..pt {
                    if mean {
                        groups.push((0..d).map(|k| idx(k, j)).collect());
                    }
                    if shape {
                        groups.push((d..bd).map(|k| idx(k, j)).collect());
                    }
                }
                if groups.is_empty() {
                    return Err(EsagError::Data("Drivers needs mean and/or shape".into()));
                }
                Atoms::Groups(GroupNorms { groups })
            }
            Method::PaperLasso => Atoms::Groups(GroupNorms {
                groups: (d..d + g).map(|i| vec![i]).collect(),
            }),
            Method::PaperGroupLasso => Atoms::Groups(GroupNorms {
                groups: (1..=d - 2)
                    .map(|jg| {
                        let s = d + jg * (jg + 1) / 2 - 1;
                        (s..s + jg + 1).collect()
                    })
                    .collect(),
            }),
        };
        let phi0 = atoms.atoms(&theta_init);
        let base: Vec<f64> = match (cfg.weighting, &atoms) {
            (Weighting::Plain, Atoms::Gaps(gp)) => {
                // concave baseline w_j = j(k−j)/k (r = 0 uses b^0 = 1, so the
                // baseline goes into the base with r fixed at 1 below)
                let k = gp.p as f64;
                (1..gp.p).map(|j| j as f64 * (k - j as f64) / k).collect()
            }
            (Weighting::Plain, _) => vec![1.0; phi0.len()],
            _ => phi0
                .iter()
                .map(|&v| if v > 0.0 { 1.0 / v } else { f64::INFINITY })
                .collect(),
        };
        let exponent = match (cfg.weighting, &atoms) {
            (Weighting::Plain, Atoms::Gaps(_)) => Exponent::Fixed(1.0),
            (Weighting::Plain, _) => Exponent::Fixed(0.0),
            (Weighting::Adaptive, _) => Exponent::Fixed(1.0),
            (Weighting::RoC(mode), _) => Exponent::RoC {
                mu: (0..d).collect(),
                mu_tilde_norm: theta_init[..d].iter().map(|v| v * v).sum::<f64>().sqrt(),
                mode,
            },
        };
        let lambdas = match &cfg.grid {
            LambdaGrid::Paper => {
                let mut v = vec![0.0, 0.5];
                v.extend((0..=8).map(|k| f64::from(1u32 << k)));
                v
            }
            LambdaGrid::Custom(v) => v.clone(),
            LambdaGrid::Auto { count, ratio } => {
                let lmax =
                    Self::lambda_max(&data, &free, &atoms, &base, &exponent, &theta_init, &config)?;
                let mut v = vec![0.0];
                let c = (*count).max(2);
                for i in 0..c {
                    v.push(lmax * ratio.powf((c - 1 - i) as f64 / (c - 1) as f64));
                }
                v
            }
        };
        if lambdas.is_empty() || lambdas.iter().any(|&l| !(l >= 0.0)) {
            return Err(EsagError::Data(
                "lambdas must be non-empty and non-negative".into(),
            ));
        }
        Ok(Self {
            data,
            center,
            scale,
            initial,
            theta_init,
            free,
            atoms,
            base,
            exponent,
            lambdas,
            config,
        })
    }

    /// Smallest λ at which every penalized atom is zero (from the gradient
    /// at the fit with the penalized atoms set to zero), times 1.05.
    fn lambda_max(
        data: &RegressionData,
        free: &[(usize, usize)],
        atoms: &Atoms,
        base: &[f64],
        exponent: &Exponent,
        theta_init: &[f64],
        config: &EsagRegressor,
    ) -> Result<f64> {
        // the null model: penalized atoms at zero, the rest refitted
        let pen = Penalty {
            atoms: atoms_as_groups_or_gaps(atoms),
            base: base.iter().map(|_| f64::INFINITY).collect(),
            exponent: Exponent::Fixed(1.0),
            lambda: 1.0,
        };
        let obj = SmoothNegLogLik::new(
            data,
            DMatrix::zeros(data.b_dim, data.p_total),
            free,
            config.parallel,
        );
        let null = solve(&obj, &pen, theta_init, 1e-6)?;
        let mut g = vec![0.0; theta_init.len()];
        obj.value_grad(&null.theta, &mut g);
        let r = match exponent {
            Exponent::Fixed(r) => *r,
            Exponent::RoC { .. } => 1.0,
        };
        let w: Vec<f64> = base
            .iter()
            .map(|&b| if r == 0.0 { 1.0 } else { b.powf(r) })
            .collect();
        let lmax = match atoms {
            Atoms::Groups(gr) => gr
                .groups
                .iter()
                .zip(&w)
                .filter(|(_, wg)| wg.is_finite() && **wg > 0.0)
                .map(|(grp, wg)| grp.iter().map(|&i| g[i] * g[i]).sum::<f64>().sqrt() / wg)
                .fold(0.0, f64::max),
            Atoms::Gaps(gp) => {
                // bound: spectral norm of the S-gradient over the smallest weight
                let s = &g[gp.offset..gp.offset + (gp.p - 1) * (gp.p + 2) / 2];
                let m = crate::logtangent::s_to_matrix(s, gp.p);
                let e = nalgebra::SymmetricEigen::new(DMatrix::from_row_slice(gp.p, gp.p, &m));
                let spread = e.eigenvalues.max() - e.eigenvalues.min();
                let wmin = w
                    .iter()
                    .copied()
                    .filter(|v| v.is_finite() && *v > 0.0)
                    .fold(f64::INFINITY, f64::min);
                spread / wmin
            }
        };
        Ok(1.05 * lmax.max(1e-8))
    }

    /// Penalized fits along the grid (ascending λ, warm starts unless a
    /// fixed start is given).
    fn path(&self, cfg: &PenalizedEsag) -> Result<Vec<PathPoint>> {
        let mut order: Vec<usize> = (0..self.lambdas.len()).collect();
        order.sort_by(|&a, &b| self.lambdas[a].total_cmp(&self.lambdas[b]));
        let start: Option<Vec<f64>> = cfg
            .start_coef
            .as_ref()
            .map(|b| self.free.iter().map(|&(k, j)| b[(k, j)]).collect());
        let obj = SmoothNegLogLik::new(
            &self.data,
            DMatrix::zeros(self.data.b_dim, self.data.p_total),
            &self.free,
            cfg.parallel,
        );
        let mut theta = start.clone().unwrap_or_else(|| self.theta_init.clone());
        let mut out: Vec<Option<PathPoint>> = vec![None; self.lambdas.len()];
        let mut relaxed_cache: Vec<(Vec<bool>, f64, f64)> = vec![];
        let mut prev_solution: Option<Vec<f64>> = None;
        let mut last_hessian: Option<Vec<f64>> = None;
        for &k in &order {
            let lambda = self.lambdas[k];
            if let Some(s) = &start {
                theta = s.clone();
            }
            let (th, objective, r, kkt, converged, n_iter) = if cfg.method.is_paper() {
                let mut starts = vec![theta.clone()];
                if theta != self.theta_init {
                    starts.push(self.theta_init.clone());
                }
                if let Some(prev) = &prev_solution
                    && !starts.contains(prev)
                {
                    starts.push(prev.clone());
                }
                self.solve_paper(&obj, lambda, &starts)?
            } else {
                let pen = self.penalty(lambda);
                let t0 = std::time::Instant::now();
                let res = solve_warm(&obj, &pen, &theta, cfg.tol, last_hessian.as_deref())?;
                last_hessian = Some(res.hessian.clone());
                if std::env::var_os("ESAG_P3_DEBUG").is_some() {
                    eprintln!(
                        "  λ={lambda:.3}: solve {:.1} ms, {} iters, kkt {:.1e}",
                        t0.elapsed().as_secs_f64() * 1e3,
                        res.n_iter,
                        res.kkt
                    );
                }
                (
                    res.theta,
                    res.objective,
                    res.r,
                    res.kkt,
                    res.converged,
                    res.n_iter,
                )
            };
            theta = th;
            prev_solution = Some(theta.clone());
            let mut g = vec![0.0; theta.len()];
            let ll = -obj.value_grad(&theta, &mut g);
            let phi = self.atoms.atoms(&theta);
            let active: Vec<bool> = phi.iter().map(|&v| v > 1e-12).collect();
            let n = self.data.n as f64;
            let coef = self.to_original(&theta);
            let mut pt = PathPoint {
                lambda,
                lambda_n: lambda / n,
                coef,
                log_likelihood: ll,
                objective,
                r,
                active: active.clone(),
                df: 0.0,
                criterion_loglik: ll,
                bic: 0.0,
                aic: 0.0,
                cv: None,
                kkt,
                converged,
                n_iter,
                coef_thresholded: None,
                loglik_thresholded: None,
                sbic: None,
            };
            if cfg.method.is_paper() {
                // the R code's rule: threshold every entry, df = nonzeros incl. mu
                let thr: Vec<f64> = theta
                    .iter()
                    .map(|&v| if v.abs() < 0.01 { 0.0 } else { v })
                    .collect();
                let llt = -obj.value_grad(&thr, &mut g);
                let df = thr.iter().filter(|&&v| v != 0.0).count() as f64;
                pt.df = df;
                pt.criterion_loglik = llt;
                pt.bic = -2.0 * llt + n.ln() * df;
                pt.aic = -2.0 * llt + 2.0 * df;
                pt.sbic = Some(-2.0 * llt + ((n + 2.0) / 24.0).ln() * df);
                pt.loglik_thresholded = Some(llt);
                pt.coef_thresholded = Some(self.to_original(&thr));
            } else {
                let df = self.model_df(&theta, &active);
                let ll_c = if cfg.relax {
                    if let Some(c) = relaxed_cache.iter().find(|c| c.0 == active) {
                        c.1
                    } else {
                        let t0 = std::time::Instant::now();
                        let rl = self.relaxed_loglik(&theta, &active)?;
                        if std::env::var_os("ESAG_P3_DEBUG").is_some() {
                            eprintln!("  relaxed refit {:.1} ms", t0.elapsed().as_secs_f64() * 1e3);
                        }
                        relaxed_cache.push((active.clone(), rl, df));
                        rl
                    }
                } else {
                    ll
                };
                pt.df = df;
                pt.criterion_loglik = ll_c;
                pt.bic = -2.0 * ll_c + n.ln() * df;
                pt.aic = -2.0 * ll_c + 2.0 * df;
            }
            out[k] = Some(pt);
        }
        Ok(out
            .into_iter()
            .map(|p| p.expect("every lambda fitted"))
            .collect())
    }

    fn penalty(&self, lambda: f64) -> Penalty<Atoms> {
        Penalty {
            atoms: self.atoms.clone(),
            base: self.base.clone(),
            exponent: self.exponent.clone(),
            lambda,
        }
    }

    /// Model dimension of the selected model (`docs/project3/METHODOLOGY.md` §5).
    fn model_df(&self, theta: &[f64], active: &[bool]) -> f64 {
        let total = self.free.len() as f64;
        match &self.atoms {
            Atoms::Groups(g) => {
                total
                    - g.groups
                        .iter()
                        .zip(active)
                        .filter(|(_, a)| !**a)
                        .map(|(grp, _)| grp.len() as f64)
                        .sum::<f64>()
            }
            Atoms::Gaps(gp) => {
                // K distinct eigenvalues with multiplicities m_i:
                // (K − 1) + k(k−1)/2 − Σ m_i(m_i−1)/2, plus μ
                let k = gp.p;
                let mut mult = vec![1usize];
                for &a in active {
                    if a {
                        mult.push(1);
                    } else {
                        *mult.last_mut().unwrap() += 1;
                    }
                }
                let _ = theta;
                let kk = mult.len();
                let shape_df = (kk - 1) as f64 + (k * (k - 1) / 2) as f64
                    - mult.iter().map(|&m| (m * (m - 1) / 2) as f64).sum::<f64>();
                self.data.d as f64 + shape_df
            }
        }
    }

    /// Unpenalized maximum over the model selected by `active` (zero groups
    /// fixed at 0 / tied gaps kept tied).
    fn relaxed_loglik(&self, theta: &[f64], active: &[bool]) -> Result<f64> {
        let obj = SmoothNegLogLik::new(
            &self.data,
            DMatrix::zeros(self.data.b_dim, self.data.p_total),
            &self.free,
            self.config.parallel,
        );
        let pen = Penalty {
            atoms: self.atoms.clone(),
            base: active
                .iter()
                .map(|&a| if a { 0.0 } else { f64::INFINITY })
                .collect(),
            exponent: Exponent::Fixed(1.0),
            lambda: 1.0,
        };
        let res = solve(&obj, &pen, theta, 1e-6)?;
        Ok(-res.smooth)
    }

    /// Relaxed refit at a selected point, as an `EsagRegressorFit` (with
    /// standard errors when the selected model is a coefficient mask).
    fn relaxed_fit(&self, pt: &PathPoint) -> Result<EsagRegressorFit> {
        let theta: Vec<f64> = self.to_standardized(&pt.coef);
        let mut mask = DMatrix::from_element(self.data.b_dim, self.data.p_total, true);
        match &self.atoms {
            Atoms::Groups(g) => {
                for (grp, &a) in g.groups.iter().zip(&pt.active) {
                    if !a {
                        for &i in grp {
                            let (k, j) = self.free[i];
                            mask[(k, j)] = false;
                        }
                    }
                }
            }
            Atoms::Gaps(_) => {
                if pt.active.iter().all(|&a| !a) {
                    mask = isotropic_mask(self.data.d, self.data.p_total);
                } else if pt.active.iter().any(|&a| !a) {
                    return Err(EsagError::Data(
                        "tie patterns are not a coefficient mask".into(),
                    ));
                }
            }
        }
        let mut b0 = DMatrix::zeros(self.data.b_dim, self.data.p_total);
        for (&(k, j), &v) in self.free.iter().zip(&theta) {
            b0[(k, j)] = if mask[(k, j)] { v } else { 0.0 };
        }
        let x = self.design();
        let y = DMatrix::from_row_slice(self.data.n, self.data.d, &self.data.y);
        let mut cfg = self.config.clone().with_hessian(true).with_mask(Some(mask));
        cfg.init = Init::Custom(b0.clone());
        let fit = cfg.fit_warm(&x, &y, &b0)?;
        Ok(self.fit_to_original(fit))
    }

    /// Held-out score of coefficients `coef` (original scale) on `(x, y)`.
    fn score(
        &self,
        coef: &DMatrix<f64>,
        x: &DMatrix<f64>,
        y: &DMatrix<f64>,
        score: CvScore,
    ) -> Result<f64> {
        let mut fit = self.initial.clone();
        // a fit object on the original scale carrying this model's frame
        fit = self.fit_to_original(fit);
        fit.coef = coef.clone();
        match score {
            CvScore::LogLikelihood => {
                let ll = fit.log_likelihood_rows(x, y)?;
                Ok(-ll.sum() / y.nrows() as f64)
            }
            CvScore::Rmse => {
                let dists = fit.predict_distribution(x)?;
                let mut acc = 0.0;
                for (i, dist) in dists.iter().enumerate() {
                    let row = DMatrix::from_fn(1, y.ncols(), |_, k| y[(i, k)]);
                    acc += prediction_rmse(dist, &row).powi(2);
                }
                Ok((acc / y.nrows() as f64).sqrt())
            }
        }
    }

    /// The standardised design without the intercept column.
    fn design(&self) -> DMatrix<f64> {
        let pt = self.data.p_total;
        DMatrix::from_fn(self.data.n, pt - 1, |i, j| self.data.x[i * pt + j + 1])
    }

    /// Standardised-scale free parameters → original-scale coefficients.
    fn to_original(&self, theta: &[f64]) -> DMatrix<f64> {
        let mut b = DMatrix::zeros(self.data.b_dim, self.data.p_total);
        for (&(k, j), &v) in self.free.iter().zip(theta) {
            b[(k, j)] = v;
        }
        self.unstandardize(&b)
    }

    fn unstandardize(&self, bs: &DMatrix<f64>) -> DMatrix<f64> {
        let mut b = bs.clone();
        for j in 1..b.ncols() {
            let (c, s) = (self.center[j - 1], self.scale[j - 1]);
            for k in 0..b.nrows() {
                b[(k, j)] = bs[(k, j)] / s;
                b[(k, 0)] -= bs[(k, j)] * c / s;
            }
        }
        b
    }

    fn to_standardized(&self, b: &DMatrix<f64>) -> Vec<f64> {
        let mut bs = b.clone();
        for j in 1..b.ncols() {
            let (c, s) = (self.center[j - 1], self.scale[j - 1]);
            for k in 0..b.nrows() {
                bs[(k, j)] = b[(k, j)] * s;
                bs[(k, 0)] += b[(k, j)] * c;
            }
        }
        self.free.iter().map(|&(k, j)| bs[(k, j)]).collect()
    }

    /// A fit on the standardised design → the same model on the original
    /// covariate scale (coefficients transformed; SEs transformed for the
    /// slopes, dropped where the transform mixes coefficients).
    fn fit_to_original(&self, mut fit: EsagRegressorFit) -> EsagRegressorFit {
        if fit.coef.ncols() > 1 {
            fit.coef = self.unstandardize(&fit.coef);
            // the covariance of an affine transform of the coefficients
            if let Some(cov) = &fit.cov_params {
                let nf = fit.free.len();
                let mut t = DMatrix::zeros(nf, nf);
                for (a, &(k, j)) in fit.free.iter().enumerate() {
                    if j == 0 {
                        t[(a, a)] = 1.0;
                        for (b, &(k2, j2)) in fit.free.iter().enumerate() {
                            if k2 == k && j2 > 0 {
                                t[(a, b)] = -self.center[j2 - 1] / self.scale[j2 - 1];
                            }
                        }
                    } else {
                        t[(a, a)] = 1.0 / self.scale[j - 1];
                    }
                }
                fit.cov_params = Some(&t * cov * t.transpose());
                fit.hessian = None;
            }
        }
        fit
    }

    /// The paper's six methods (γ): smoothing continuation with L-BFGS on
    /// `−ℓ + λ Σ_g w_g(μ) √(φ_g² + ε²)` (RoC inside the objective, floor 1),
    /// from several starts (the γ landscape has many local optima and
    /// cliffs), returning the best untruncated optimum and its unsmoothed
    /// objective.
    fn solve_paper(
        &self,
        obj: &SmoothNegLogLik,
        lambda: f64,
        starts: &[Vec<f64>],
    ) -> Result<(Vec<f64>, f64, f64, f64, bool, usize)> {
        let Atoms::Groups(groups) = &self.atoms else {
            unreachable!()
        };
        let pen = self.penalty(lambda);
        // the given starts, plus perturbations of each (deterministic)
        let mut all: Vec<Vec<f64>> = starts.to_vec();
        let mut rng = crate::stream_rng(0x0009_A9E3, crate::streams::RESTARTS, lambda.to_bits());
        for s in starts {
            for _ in 0..PAPER_PERTURBED {
                all.push(
                    s.iter()
                        .map(|&v| {
                            let z: f64 = rand_distr::Distribution::sample(
                                &rand_distr::StandardNormal,
                                &mut rng,
                            );
                            v + (0.05 + 0.1 * v.abs()) * z
                        })
                        .collect(),
                );
            }
        }
        // Smoothing continuation with exact gradients, stopped at ε = 1e-3:
        // the γ objective often has no attained maximum (its supremum is
        // approached with small nonzero entries on the favourable side of a
        // discontinuity of γ ↦ V), so entries are kept just off zero, as R's
        // BFGS with 1e-3 finite differences does.
        // Two schedules per start: the full continuation, and a local
        // polish at ε = 1e-3 only (which cannot leave a good start's basin);
        // the starts themselves are candidates too.
        let schedules: [&[f64]; 2] = [&[1e-2, 1e-3], &[1e-3]];
        let jobs: Vec<(&Vec<f64>, &[f64])> = all
            .iter()
            .flat_map(|t| schedules.iter().map(move |sc| (t, *sc)))
            .collect();
        let eval = |theta: &[f64]| {
            let mut g = vec![0.0; theta.len()];
            obj.value_grad(theta, &mut g) + pen.value(theta)
        };
        let mut runs: Vec<(Vec<f64>, f64, bool, usize)> = jobs
            .par_iter()
            .map(|&(theta0, schedule)| {
                let mut theta = theta0.clone();
                let mut iters = 0;
                let mut converged = true;
                for &eps in schedule {
                    let sm = Smoothed {
                        obj,
                        groups,
                        pen: &pen,
                        eps,
                    };
                    let cfg = LbfgsConfig {
                        memory: theta.len().clamp(10, 100),
                        max_iter: 3000,
                        tol_grad: 1e-9,
                        ..LbfgsConfig::default()
                    };
                    let r = minimize(&sm, &theta, &cfg);
                    iters += r.n_iter;
                    converged &= r.converged();
                    theta = r.x;
                }
                let f = eval(&theta);
                (theta, f, converged, iters)
            })
            .collect();
        runs.extend(starts.iter().map(|s| (s.clone(), eval(s), false, 0)));
        let total: usize = runs.iter().map(|r| r.3).sum();
        let best = runs
            .into_iter()
            .filter(|r| r.1.is_finite())
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .ok_or_else(|| EsagError::Convergence("no finite penalized objective".into()))?;
        Ok((
            best.0.clone(),
            best.1,
            pen.r(&best.0),
            f64::NAN,
            best.2,
            total,
        ))
    }
}

/// Perturbed copies of each start in the paper-reproduction mode.
const PAPER_PERTURBED: usize = 3;

fn atoms_as_groups_or_gaps(a: &Atoms) -> Atoms {
    a.clone()
}

impl PenaltyAtoms for Atoms {
    fn len(&self) -> usize {
        Atoms::len(self)
    }
    fn atoms(&self, theta: &[f64], out: &mut [f64]) {
        match self {
            Atoms::Groups(g) => g.atoms(theta, out),
            Atoms::Gaps(g) => g.atoms(theta, out),
        }
    }
    fn prox(&self, v: &mut [f64], t: &[f64]) {
        match self {
            Atoms::Groups(g) => g.prox(v, t),
            Atoms::Gaps(g) => g.prox(v, t),
        }
    }
    fn support(&self) -> Vec<usize> {
        match self {
            Atoms::Groups(g) => g.support(),
            Atoms::Gaps(g) => g.support(),
        }
    }
}

fn solve(
    obj: &SmoothNegLogLik,
    pen: &Penalty<Atoms>,
    theta0: &[f64],
    tol: f64,
) -> Result<crate::optim::prox::ProxResult> {
    solve_warm(obj, pen, theta0, tol, None)
}

fn solve_warm(
    obj: &SmoothNegLogLik,
    pen: &Penalty<Atoms>,
    theta0: &[f64],
    tol: f64,
    hess0: Option<&[f64]>,
) -> Result<crate::optim::prox::ProxResult> {
    let cfg = ProxConfig {
        tol: tol * (obj_n(obj) as f64).sqrt().max(1.0),
        ..ProxConfig::default()
    };
    let res = crate::optim::prox::prox_newton_warm(obj, pen, theta0, &cfg, hess0);
    if !res.objective.is_finite() {
        return Err(EsagError::Convergence(
            "penalized objective is not finite".into(),
        ));
    }
    Ok(res)
}

fn obj_n(obj: &SmoothNegLogLik) -> usize {
    obj.n_obs()
}

/// The smoothed objective of the paper-reproduction mode.
struct Smoothed<'a> {
    obj: &'a SmoothNegLogLik<'a>,
    groups: &'a GroupNorms,
    pen: &'a Penalty<Atoms>,
    eps: f64,
}

impl Objective for Smoothed<'_> {
    fn value_and_gradient(&self, theta: &[f64], grad: &mut [f64]) -> f64 {
        let f = self.obj.value_grad(theta, grad);
        let w = self.pen.weights(theta);
        let lam = self.pen.lambda;
        let mut pen_val = 0.0;
        let mut dr_sum = 0.0; // Σ_g √(φ²+ε²) w_g ln b_g
        for ((grp, &wg), &b) in self.groups.groups.iter().zip(&w).zip(&self.pen.base) {
            let phi2: f64 = grp.iter().map(|&i| theta[i] * theta[i]).sum();
            let sm = (phi2 + self.eps * self.eps).sqrt();
            if wg.is_finite() {
                pen_val += wg * sm;
                for &i in grp {
                    grad[i] += lam * wg * theta[i] / sm;
                }
                if b > 0.0 {
                    dr_sum += sm * wg * b.ln();
                }
            }
        }
        if let Exponent::RoC {
            mu, mu_tilde_norm, ..
        } = &self.pen.exponent
        {
            let nm2: f64 = mu.iter().map(|&i| theta[i] * theta[i]).sum();
            let nm = nm2.sqrt();
            if mu_tilde_norm / nm > 1.0 {
                let coef = -lam * dr_sum * mu_tilde_norm / (nm2 * nm);
                for &i in mu {
                    grad[i] += coef * theta[i];
                }
            }
        }
        f + lam * pen_val
    }
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

/// The paper's selection rates (Section 5.1, names as in the paper):
/// correction rate, "SE" = share of the true zeros estimated as zero, "SP"
/// = share of the true nonzeros estimated as nonzero.
pub fn selection_rates(truth: &[f64], estimate: &[f64]) -> (f64, f64, f64) {
    let (mut z_ok, mut z_tot, mut nz_ok, mut nz_tot) = (0usize, 0usize, 0usize, 0usize);
    for (&t, &h) in truth.iter().zip(estimate) {
        if t == 0.0 {
            z_tot += 1;
            z_ok += usize::from(h == 0.0);
        } else {
            nz_tot += 1;
            nz_ok += usize::from(h != 0.0);
        }
    }
    let cr = (z_ok + nz_ok) as f64 / truth.len() as f64;
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
