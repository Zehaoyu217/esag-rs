//! LASSO-type penalized ESAG fits (Yu & Huang, *LASSO-based inference for
//! elliptically symmetric angular Gaussian distributions*, "Project 3"),
//! with the corrections of `docs/project3/METHODOLOGY.md`.
//!
//! All objectives are on the sum scale of the R code,
//! `F = −Σ_i ℓ_i + λ·P`, so `λ` is the paper's `n·λ_n`. `μ` (and the
//! intercept column in regression) is never penalized.
//!
//! # Choosing a penalty
//!
//! A penalty is three choices, combined freely, plus the weights:
//!
//! * [`Penalty`]: `Lasso` (each coefficient), `GroupLasso` (groups),
//!   `SparseGroupLasso { l1_ratio }` (both), `EigenGap` (ties between the
//!   eigenvalues of the shape, i.i.d.);
//! * [`Target`]: which coefficients, `Slopes` (default), `MeanSlopes`,
//!   `ShapeSlopes`, or `Shape` (all shape coefficients, intercept included);
//! * [`Grouping`]: `Covariate` (default), `CovariatePart` (mean / shape
//!   slopes of each covariate), `Parameter` (a row of `B`), `All`,
//!   `GammaGroups`, or `Custom`;
//! * [`Weighting`]: `Plain`, `Adaptive(r)` (`‖θ̃_g‖^{−r}` from the
//!   unpenalized fit) or `RoC` (i.i.d.).
//!
//! The presets cover the usual estimators: `PenalizedEsag::lasso()`,
//! `adaptive_lasso()`, `group_lasso()`, `adaptive_group_lasso()`,
//! `sparse_group_lasso(α)`, `isotropy()` (the paper's eq. 4.4),
//! `eigen_gap()`, and `paper(name)` for the six methods exactly as coded in
//! the Project 3 R scripts. Group weights carry Yuan & Lin's `√|g|` factor.
//!
//! The default shape is the smooth log-tangent one
//! ([`crate::regression::Shape::LogTangent`]): exact zeros of the paper's
//! `γ` (a whole group `j ≥ 2`, or its trailing pair) lie on discontinuities
//! of `γ ↦ V` (`docs/MATH.md`). Slope penalties also work with
//! `Shape::Gamma` (zero slopes are not singular points). The paper mode
//! (`Solver::PaperReproduction`) uses smoothing continuation, the R code's
//! post-hoc rule `|x| < 0.01 → 0` and its BIC/AIC/SBIC; the default solver,
//! proximal Newton ([`crate::optim::prox`]), gives exact zeros.
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
    EigenGaps, Exponent, GroupNorms, Penalty as AtomPenalty, PenaltyAtoms, ProxConfig, RocMode,
    SmoothObjective, SparseGroupNorms,
};
use crate::regression::{
    EsagRegressor, EsagRegressorFit, Init, RegressionData, Shape, SmoothNegLogLik, isotropic_mask,
};
use nalgebra::DMatrix;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// The penalty family.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum Penalty {
    /// `Σ_k w_k |β_k|` over the target coefficients (LASSO).
    Lasso,
    /// `Σ_g w_g ‖β_g‖₂` over groups of target coefficients (group LASSO).
    GroupLasso,
    /// `(1−α) Σ_g w_g ‖β_g‖₂ + α Σ_k w_k |β_k|` (sparse group LASSO), with
    /// `α = l1_ratio ∈ [0, 1]`.
    SparseGroupLasso { l1_ratio: f64 },
    /// `Σ_j w_j (σ_(j+1) − σ_(j))`, gaps between the sorted eigenvalues of
    /// the log-tangent shape `S` (i.i.d.): zero gaps are tied eigenvalues of
    /// `V` (partial isotropy).
    EigenGap,
}

/// How the target coefficients are grouped (group and sparse-group LASSO).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Grouping {
    /// One group per covariate: all its target slopes (does the covariate
    /// matter at all?).
    Covariate,
    /// Per covariate, its `μ` slopes and its shape slopes as two groups
    /// (does it move the mean direction? the shape?).
    CovariatePart,
    /// One group per parameter row (`mu[k]` or a shape coordinate) across
    /// the target columns.
    Parameter,
    /// All target coefficients as one group (with `Target::Shape`: isotropy).
    All,
    /// The groups `γ_(1), …, γ_(d−2)` of Yu & Huang's γ (per column).
    GammaGroups,
    /// Explicit groups of `(row, column)` coefficients of `B` (column 0 is
    /// the intercept).
    Custom(Vec<Vec<(usize, usize)>>),
}

/// Which coefficients are penalized. The intercept column is penalized only
/// by `Shape` and `All` (shape coefficients).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Target {
    /// Every slope (all rows, covariate columns). The default.
    Slopes,
    /// The `μ` slopes.
    MeanSlopes,
    /// The shape slopes.
    ShapeSlopes,
    /// Every shape coefficient, intercept included (i.i.d. shape sparsity,
    /// isotropy).
    Shape,
}

/// Adaptive weights `b_g^r`, `b_g = 1/φ_g(θ̃)` from the unpenalized fit.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum Weighting {
    /// `r = 0`: every weight 1 (for `EigenGap`: the concave baseline
    /// `w_j = j(k−j)/k`).
    Plain,
    /// `r` given (the adaptive LASSO uses `r = 1`).
    Adaptive(f64),
    /// `r = max(1, ‖μ̃‖/‖μ‖)`, the ratio of concentration (i.i.d. only).
    RoC(RocMode),
}

/// How the penalized problem is solved.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Solver {
    /// Proximal Newton: exact zeros, KKT certificate (default).
    Exact,
    /// The R code of Project 3: smoothing continuation, then the rule
    /// `|x| < 0.01 → 0` and the R code's BIC/AIC/SBIC (γ, i.i.d.).
    PaperReproduction,
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

/// Penalized ESAG estimator. Start from a preset (`lasso()`,
/// `adaptive_lasso()`, `group_lasso()`, `adaptive_group_lasso()`,
/// `sparse_group_lasso(α)`, `isotropy()`, `eigen_gap()`, `paper(name)`),
/// adjust with the builder methods, then [`PenalizedEsag::fit`].
///
/// ```no_run
/// use esag::penalized::{Grouping, PenalizedEsag, Target};
/// # let (x, y) = (nalgebra::DMatrix::<f64>::zeros(1, 1), nalgebra::DMatrix::<f64>::zeros(1, 3));
/// // adaptive group LASSO: which covariates move the mean, which the shape?
/// let path = PenalizedEsag::adaptive_group_lasso()
///     .grouping(Grouping::CovariatePart)
///     .fit(&x, &y)?;
/// // plain LASSO on the shape slopes only, λ by 10-fold cross-validation
/// let path = PenalizedEsag::lasso().target(Target::ShapeSlopes).cv(10, 1).fit(&x, &y)?;
/// # Ok::<(), esag::error::EsagError>(())
/// ```
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PenalizedEsag {
    pub penalty: Penalty,
    pub grouping: Grouping,
    pub target: Target,
    pub weighting: Weighting,
    /// Shape parameterization of the model (default
    /// [`Shape::LogTangent`]: smooth, exact zeros are well-posed).
    pub shape: Shape,
    pub solver: Solver,
    pub grid: LambdaGrid,
    pub selection: Selection,
    /// Multiply group weights by `√(group size)` (Yuan & Lin; default
    /// `true`, `false` in paper mode as in the R code).
    pub group_size_factor: bool,
    /// Compute BIC/AIC at the relaxed refit (default `true`; ignored by the
    /// paper mode, which uses the R code's rule).
    pub relax: bool,
    /// Stationarity tolerance of the solver (sum scale). Default 1e-6.
    pub tol: f64,
    /// Parallel likelihood evaluation.
    pub parallel: bool,
    /// Paper mode: unpenalized estimate (`b_dim × 1`) to build the weights
    /// from, e.g. R's `unp_mle$par` (default: `esag`'s MLE).
    pub initial_coef: Option<DMatrix<f64>>,
    /// Paper mode: start every λ from this (e.g. R's `truth + 0.1`) instead
    /// of warm-starting along the path.
    pub start_coef: Option<DMatrix<f64>>,
    /// Covariate names for summaries.
    #[serde(default)]
    pub feature_names: Option<Vec<String>>,
    /// Response (column) names for summaries.
    #[serde(default)]
    pub response_names: Option<Vec<String>>,
    /// Reference direction of the log-tangent frame (default: the data's
    /// mean direction); see [`EsagRegressor::lt_reference`].
    #[serde(default)]
    pub lt_reference: Option<Vec<f64>>,
}

impl PenalizedEsag {
    fn base(penalty: Penalty, grouping: Grouping, target: Target, weighting: Weighting) -> Self {
        Self {
            penalty,
            grouping,
            target,
            weighting,
            shape: Shape::LogTangent,
            solver: Solver::Exact,
            grid: LambdaGrid::Auto {
                count: 30,
                ratio: 1e-3,
            },
            selection: Selection::Bic,
            group_size_factor: true,
            relax: true,
            tol: 1e-6,
            parallel: true,
            initial_coef: None,
            start_coef: None,
            feature_names: None,
            response_names: None,
            lt_reference: None,
        }
    }
    /// LASSO on every slope coefficient.
    pub fn lasso() -> Self {
        Self::base(
            Penalty::Lasso,
            Grouping::Covariate,
            Target::Slopes,
            Weighting::Plain,
        )
    }
    /// Adaptive LASSO (`r = 1`) on every slope coefficient.
    pub fn adaptive_lasso() -> Self {
        Self::lasso().adaptive(1.0)
    }
    /// Group LASSO, one group per covariate.
    pub fn group_lasso() -> Self {
        Self::base(
            Penalty::GroupLasso,
            Grouping::Covariate,
            Target::Slopes,
            Weighting::Plain,
        )
    }
    /// Adaptive group LASSO (`r = 1`), one group per covariate.
    pub fn adaptive_group_lasso() -> Self {
        Self::group_lasso().adaptive(1.0)
    }
    /// Sparse group LASSO with L1 share `l1_ratio`, groups per covariate.
    pub fn sparse_group_lasso(l1_ratio: f64) -> Self {
        Self::base(
            Penalty::SparseGroupLasso { l1_ratio },
            Grouping::Covariate,
            Target::Slopes,
            Weighting::Plain,
        )
    }
    /// Shrink the whole shape towards isotropy (one group: all shape
    /// coefficients; the paper's eq. 4.4), adaptive weights.
    pub fn isotropy() -> Self {
        Self::base(
            Penalty::GroupLasso,
            Grouping::All,
            Target::Shape,
            Weighting::Adaptive(1.0),
        )
    }
    /// Partial isotropy for i.i.d. data: gaps between the eigenvalues of the
    /// shape, adaptive weights.
    pub fn eigen_gap() -> Self {
        Self::base(
            Penalty::EigenGap,
            Grouping::All,
            Target::Shape,
            Weighting::Adaptive(1.0),
        )
    }
    /// One of the paper's six methods, reproduced as coded in its R scripts
    /// (γ shape, i.i.d.): `LASSO`, `aLASSO`, `RoC-LASSO`, `GLASSO`,
    /// `aGLASSO`, `RoC-GLASSO`.
    pub fn paper(name: &str) -> Result<Self> {
        let (penalty, grouping, w) = match name {
            "LASSO" => (Penalty::Lasso, Grouping::All, Weighting::Plain),
            "aLASSO" => (Penalty::Lasso, Grouping::All, Weighting::Adaptive(1.0)),
            "RoC-LASSO" => (
                Penalty::Lasso,
                Grouping::All,
                Weighting::RoC(RocMode::Joint),
            ),
            "GLASSO" => (Penalty::GroupLasso, Grouping::GammaGroups, Weighting::Plain),
            "aGLASSO" => (
                Penalty::GroupLasso,
                Grouping::GammaGroups,
                Weighting::Adaptive(1.0),
            ),
            "RoC-GLASSO" => (
                Penalty::GroupLasso,
                Grouping::GammaGroups,
                Weighting::RoC(RocMode::Joint),
            ),
            other => {
                return Err(EsagError::Data(format!(
                    "unknown paper method {other:?} (LASSO, aLASSO, RoC-LASSO, GLASSO, aGLASSO, RoC-GLASSO)"
                )));
            }
        };
        let mut s = Self::base(penalty, grouping, Target::Shape, w);
        s.shape = Shape::Gamma;
        s.solver = Solver::PaperReproduction;
        s.grid = LambdaGrid::Paper;
        s.group_size_factor = false;
        s.relax = false;
        Ok(s)
    }
    /// Which coefficients are penalized.
    pub fn target(mut self, t: Target) -> Self {
        self.target = t;
        self
    }
    /// How they are grouped (group and sparse-group LASSO).
    pub fn grouping(mut self, g: Grouping) -> Self {
        self.grouping = g;
        self
    }
    /// Adaptive weights `‖θ̃_g‖^{−r}`.
    pub fn adaptive(mut self, r: f64) -> Self {
        self.weighting = Weighting::Adaptive(r);
        self
    }
    /// Plain weights (all 1).
    pub fn plain(mut self) -> Self {
        self.weighting = Weighting::Plain;
        self
    }
    /// RoC weights (i.i.d.).
    pub fn roc(mut self, mode: RocMode) -> Self {
        self.weighting = Weighting::RoC(mode);
        self
    }
    /// Shape parameterization.
    pub fn shape(mut self, shape: Shape) -> Self {
        self.shape = shape;
        self
    }
    /// Explicit λ values (sum scale; the paper's `λ_n = λ/n`).
    pub fn lambdas(mut self, v: Vec<f64>) -> Self {
        self.grid = LambdaGrid::Custom(v);
        self
    }
    /// Covariate names for summaries.
    pub fn feature_names(mut self, names: Vec<String>) -> Self {
        self.feature_names = Some(names);
        self
    }
    /// Response names for summaries.
    pub fn response_names(mut self, names: Vec<String>) -> Self {
        self.response_names = Some(names);
        self
    }
    /// Reference direction of the log-tangent frame.
    pub fn lt_reference(mut self, m: Vec<f64>) -> Self {
        self.lt_reference = Some(m);
        self
    }
    /// A short name of the method ("adaptive group LASSO", …).
    pub fn method_name(&self) -> String {
        let weights = match self.weighting {
            Weighting::Plain => "",
            Weighting::Adaptive(_) => "adaptive ",
            Weighting::RoC(_) => "RoC-",
        };
        let core = match self.penalty {
            Penalty::Lasso => "LASSO".to_string(),
            Penalty::GroupLasso
                if self.grouping == Grouping::All && self.target == Target::Shape =>
            {
                "isotropy group LASSO".to_string()
            }
            Penalty::GroupLasso => "group LASSO".to_string(),
            Penalty::SparseGroupLasso { .. } => "sparse group LASSO".to_string(),
            Penalty::EigenGap => "eigen-gap LASSO".to_string(),
        };
        let paper = if self.is_paper() { " [paper mode]" } else { "" };
        format!("{weights}{core}{paper}")
    }
    /// Select λ by BIC (default).
    pub fn bic(mut self) -> Self {
        self.selection = Selection::Bic;
        self
    }
    /// Select λ by AIC.
    pub fn aic(mut self) -> Self {
        self.selection = Selection::Aic;
        self
    }
    /// Select λ by `folds`-fold cross-validated held-out log-likelihood.
    pub fn cv(mut self, folds: usize, seed: u64) -> Self {
        self.selection = Selection::CrossValidation {
            folds,
            seed,
            score: CvScore::LogLikelihood,
        };
        self
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
    /// Whether this is a paper-reproduction preset ([`Self::paper`]).
    pub fn is_paper(&self) -> bool {
        self.solver == Solver::PaperReproduction
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
                // every fold fits the model of the full data: the same
                // log-tangent frame, not one from its own training rows
                let mut same_frame = self.clone();
                same_frame.lt_reference = prep.config.lt_reference.clone();
                let v = same_frame.cross_validate(x, y, &prep.lambdas, folds, seed, score)?;
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
        let relaxed = if self.is_paper() {
            None
        } else {
            prep.relaxed_fit(&points[selected]).ok()
        };
        Ok(PenalizedPath {
            config: self.clone(),
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
    fn cross_validate(
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
    /// The estimator configuration.
    pub config: PenalizedEsag,
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

    /// statsmodels-style summary of the selected model: header (method,
    /// target, selection, λ, active groups, fit statistics at the relaxed
    /// refit) and the coefficient blocks of the relaxed refit, with standard
    /// errors; dropped coefficients are marked. [`Self::path_summary`] lists
    /// the whole λ path.
    pub fn summary(&self) -> String {
        let b = self.best();
        let c = &self.config;
        let target = match c.target {
            Target::Slopes => "slopes",
            Target::MeanSlopes => "mean slopes",
            Target::ShapeSlopes => "shape slopes",
            Target::Shape => "shape",
        };
        let grouping = match (&c.penalty, &c.grouping) {
            (Penalty::Lasso, _) => "entries".to_string(),
            (Penalty::EigenGap, _) => "eigen-gaps".to_string(),
            (_, Grouping::Covariate) => "covariate".to_string(),
            (_, Grouping::CovariatePart) => "cov. part".to_string(),
            (_, Grouping::Parameter) => "parameter".to_string(),
            (_, Grouping::All) => "one group".to_string(),
            (_, Grouping::GammaGroups) => "gamma groups".to_string(),
            (_, Grouping::Custom(g)) => format!("{} groups", g.len()),
        };
        let selection = match c.selection {
            Selection::Bic => "BIC".to_string(),
            Selection::Aic => "AIC".to_string(),
            Selection::CrossValidation { folds, score, .. } => format!(
                "{folds}-fold CV ({})",
                if score == CvScore::LogLikelihood {
                    "log-lik"
                } else {
                    "RMSE"
                }
            ),
        };
        let n_active = b.active.iter().filter(|&&a| a).count();
        let mut extra = vec![
            ("Penalized:".to_string(), format!("{target} / {grouping}")),
            ("lambda (sum):".to_string(), crate::summary::num(b.lambda)),
            ("Selection:".to_string(), selection),
            (
                "alpha = lambda/n:".to_string(),
                crate::summary::num(b.lambda_n),
            ),
            (
                "Active atoms:".to_string(),
                format!("{n_active} / {}", b.active.len()),
            ),
            (
                if self.relaxed.is_some() {
                    "Penalized log-lik:"
                } else {
                    "Log-lik:"
                }
                .to_string(),
                crate::summary::num(b.log_likelihood),
            ),
        ];
        if let Penalty::SparseGroupLasso { l1_ratio } = c.penalty {
            extra.push(("l1_ratio:".to_string(), format!("{l1_ratio}")));
        }
        match &self.relaxed {
            Some(r) => r.summary_titled(
                "ESAG Regularized Regression Results",
                &c.method_name(),
                0.05,
                &extra,
                "(dropped)",
            ),
            None => {
                // no relaxed refit (paper mode, tie patterns): penalized
                // coefficients without standard errors
                let mut fit = self.initial.clone();
                fit.coef = b.coef_thresholded.clone().unwrap_or_else(|| b.coef.clone());
                fit.cov_params = None;
                fit.log_likelihood = b.criterion_loglik;
                fit.mask = fit.coef.map(|v| v != 0.0);
                for (k, j) in
                    (0..fit.coef.nrows()).flat_map(|k| (0..fit.coef.ncols()).map(move |j| (k, j)))
                {
                    if !self.initial.mask[(k, j)] {
                        fit.mask[(k, j)] = false;
                    }
                }
                fit.summary_titled(
                    "ESAG Regularized Regression Results",
                    &c.method_name(),
                    0.05,
                    &extra,
                    "(dropped)",
                )
            }
        }
    }

    /// A text table of the λ path.
    pub fn path_summary(&self) -> String {
        let mut s = format!(
            "Penalized ESAG: {}, {:?} weights, selection {:?}\n{:>10} {:>10} {:>12} {:>7} {:>12} {:>10} {:>9}\n",
            self.config.method_name(),
            self.config.weighting,
            self.config.selection,
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
    SparseGroups(SparseGroupNorms),
    Gaps(EigenGaps),
}

impl Atoms {
    fn len(&self) -> usize {
        match self {
            Atoms::Groups(g) => g.len(),
            Atoms::SparseGroups(g) => g.len(),
            Atoms::Gaps(g) => g.len(),
        }
    }
    fn atoms(&self, theta: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; self.len()];
        PenaltyAtoms::atoms(self, theta, &mut out);
        out
    }
}

impl Prepared {
    fn new(cfg: &PenalizedEsag, x: &DMatrix<f64>, y: &DMatrix<f64>) -> Result<Self> {
        let d = y.ncols();
        let p = x.ncols();
        let paper = cfg.is_paper();
        let iid_only = paper || cfg.penalty == Penalty::EigenGap;
        if iid_only && p > 0 {
            return Err(EsagError::Data(format!(
                "{:?} is for i.i.d. data (no covariates)",
                if paper {
                    "paper mode".to_string()
                } else {
                    format!("{:?}", cfg.penalty)
                }
            )));
        }
        if matches!(
            cfg.target,
            Target::Slopes | Target::MeanSlopes | Target::ShapeSlopes
        ) && p == 0
        {
            return Err(EsagError::Data(format!(
                "Target::{:?} penalizes slopes but there are no covariates (use Target::Shape)",
                cfg.target
            )));
        }
        if matches!(cfg.weighting, Weighting::RoC(_)) && p > 0 {
            return Err(EsagError::Data(
                "RoC weights are defined for i.i.d. data; use adaptive weights in regression"
                    .into(),
            ));
        }
        if cfg.penalty == Penalty::EigenGap && cfg.shape != Shape::LogTangent {
            return Err(EsagError::Data("EigenGap needs Shape::LogTangent".into()));
        }
        if let Penalty::SparseGroupLasso { l1_ratio } = cfg.penalty
            && !(0.0..=1.0).contains(&l1_ratio)
        {
            return Err(EsagError::Data(format!(
                "l1_ratio must be in [0, 1], got {l1_ratio}"
            )));
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
        let mut config = EsagRegressor::new()
            .with_shape(cfg.shape)
            .with_lt_reference(cfg.lt_reference.clone())
            .with_parallel(cfg.parallel)
            .with_hessian(false);
        if let Some(n) = &cfg.feature_names {
            config = config.with_feature_names(n.clone());
        }
        if let Some(n) = &cfg.response_names {
            config = config.with_response_names(n.clone());
        }
        let initial = match (&cfg.initial_coef, paper) {
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
            Some(b) if paper => free.iter().map(|&(k, j)| b[(k, j)]).collect(),
            _ => free.iter().map(|&(k, j)| initial.coef[(k, j)]).collect(),
        };
        // the target coefficients (row, column)
        let in_target = |k: usize, j: usize| -> bool {
            match cfg.target {
                Target::Slopes => j >= 1,
                Target::MeanSlopes => j >= 1 && k < d,
                Target::ShapeSlopes => j >= 1 && k >= d,
                Target::Shape => k >= d,
            }
        };
        let target: Vec<(usize, usize)> = (0..pt)
            .flat_map(|j| (0..bd).map(move |k| (k, j)))
            .filter(|&(k, j)| in_target(k, j))
            .collect();
        if target.is_empty() {
            return Err(EsagError::Data("the target selects no coefficients".into()));
        }
        let groups: Vec<Vec<usize>> = match &cfg.grouping {
            Grouping::Covariate => (0..pt)
                .map(|j| {
                    target
                        .iter()
                        .filter(|c| c.1 == j)
                        .map(|&(k, j)| idx(k, j))
                        .collect()
                })
                .collect(),
            Grouping::CovariatePart => (0..pt)
                .flat_map(|j| {
                    let mean: Vec<usize> = target
                        .iter()
                        .filter(|c| c.1 == j && c.0 < d)
                        .map(|&(k, j)| idx(k, j))
                        .collect();
                    let shape: Vec<usize> = target
                        .iter()
                        .filter(|c| c.1 == j && c.0 >= d)
                        .map(|&(k, j)| idx(k, j))
                        .collect();
                    [mean, shape]
                })
                .collect(),
            Grouping::Parameter => (0..bd)
                .map(|k| {
                    target
                        .iter()
                        .filter(|c| c.0 == k)
                        .map(|&(k, j)| idx(k, j))
                        .collect()
                })
                .collect(),
            Grouping::All => vec![target.iter().map(|&(k, j)| idx(k, j)).collect()],
            Grouping::GammaGroups => {
                if cfg.shape != Shape::Gamma {
                    return Err(EsagError::Data(
                        "Grouping::GammaGroups needs Shape::Gamma".into(),
                    ));
                }
                (0..pt)
                    .flat_map(|j| {
                        let tg = &target;
                        (1..=d - 2).map(move |jg| {
                            let start = d + jg * (jg + 1) / 2 - 1;
                            (start..start + jg + 1)
                                .filter(|&k| tg.contains(&(k, j)))
                                .map(|k| idx(k, j))
                                .collect::<Vec<usize>>()
                        })
                    })
                    .collect()
            }
            Grouping::Custom(gs) => gs
                .iter()
                .map(|g| {
                    g.iter()
                        .map(|&(k, j)| {
                            if k >= bd || j >= pt {
                                Err(EsagError::Shape(format!(
                                    "custom group entry ({k}, {j}) is outside B ({bd}×{pt})"
                                )))
                            } else {
                                Ok(idx(k, j))
                            }
                        })
                        .collect::<Result<Vec<usize>>>()
                })
                .collect::<Result<Vec<_>>>()?,
        };
        let groups: Vec<Vec<usize>> = groups.into_iter().filter(|g| !g.is_empty()).collect();
        let atoms = match cfg.penalty {
            Penalty::Lasso => Atoms::Groups(GroupNorms {
                groups: target.iter().map(|&(k, j)| vec![idx(k, j)]).collect(),
            }),
            Penalty::GroupLasso => Atoms::Groups(GroupNorms { groups }),
            Penalty::SparseGroupLasso { .. } => Atoms::SparseGroups(SparseGroupNorms { groups }),
            Penalty::EigenGap => Atoms::Gaps(EigenGaps {
                p: d - 1,
                offset: d,
            }),
        };
        // base weights: adaptive 1/φ(θ̃) (∞ at zero), group-size factor √|g|
        // on group atoms, and the sparse-group mixture (1−α) / α
        let phi0 = atoms.atoms(&theta_init);
        let sizes: Vec<f64> = match &atoms {
            Atoms::Groups(g) => g.groups.iter().map(|g| g.len() as f64).collect(),
            Atoms::SparseGroups(g) => g
                .groups
                .iter()
                .map(|g| g.len() as f64)
                .chain(std::iter::repeat_n(1.0, g.support().len()))
                .collect(),
            Atoms::Gaps(g) => vec![1.0; g.p - 1],
        };
        let mut base: Vec<f64> = match (cfg.weighting, &atoms) {
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
        let r_scale = |b: f64, factor: f64, r: f64| -> f64 {
            // weight = (b · factor^{1/r})^r = b^r · factor, so the factor is
            // not raised to the adaptive exponent
            if r == 0.0 {
                b * factor
            } else {
                b * factor.powf(1.0 / r)
            }
        };
        // plain weights use r = 1 with the base set to the factor, so that
        // the factor survives (b^0 would be 1)
        let r_fixed = match cfg.weighting {
            Weighting::Plain => 1.0,
            Weighting::Adaptive(r) => r,
            Weighting::RoC(_) => 1.0,
        };
        let gfac = |i: usize| -> f64 {
            let size = if cfg.group_size_factor && cfg.penalty != Penalty::Lasso {
                sizes[i].sqrt()
            } else {
                1.0
            };
            let mix = match (&atoms, cfg.penalty) {
                (Atoms::SparseGroups(sg), Penalty::SparseGroupLasso { l1_ratio }) => {
                    if i < sg.groups.len() {
                        1.0 - l1_ratio
                    } else {
                        l1_ratio
                    }
                }
                _ => 1.0,
            };
            size * mix
        };
        if !matches!(atoms, Atoms::Gaps(_)) {
            for (i, b) in base.iter_mut().enumerate() {
                let f = gfac(i);
                *b = if f == 0.0 {
                    0.0
                } else {
                    r_scale(*b, f, r_fixed)
                };
            }
        }
        let exponent = match (cfg.weighting, &atoms) {
            (Weighting::Plain, _) => Exponent::Fixed(1.0),
            (Weighting::Adaptive(r), _) => Exponent::Fixed(r),
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
        let pen = AtomPenalty {
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
            Atoms::SparseGroups(sg) => {
                // a group is zero at λ iff ‖S(g; λ w_i)‖ ≤ λ w_G: solve by
                // bisection per group (the left side minus the right
                // decreases in λ)
                let ng = sg.groups.len();
                let mut e = ng;
                let mut lmax: f64 = 0.0;
                for (gi, grp) in sg.groups.iter().enumerate() {
                    let we: Vec<f64> = (0..grp.len()).map(|t| w[e + t]).collect();
                    e += grp.len();
                    let wg = w[gi];
                    let excess = |lam: f64| -> f64 {
                        let s2: f64 = grp
                            .iter()
                            .zip(&we)
                            .map(|(&i, &wi)| {
                                let t = if wi.is_finite() {
                                    lam * wi
                                } else {
                                    f64::INFINITY
                                };
                                (g[i].abs() - t).max(0.0).powi(2)
                            })
                            .sum();
                        s2.sqrt()
                            - if wg.is_finite() {
                                lam * wg
                            } else {
                                f64::INFINITY
                            }
                    };
                    if excess(0.0) <= 0.0 {
                        continue;
                    }
                    let mut hi = 1.0;
                    while excess(hi) > 0.0 && hi < 1e12 {
                        hi *= 2.0;
                    }
                    let mut lo = 0.0;
                    for _ in 0..80 {
                        let mid = 0.5 * (lo + hi);
                        if excess(mid) > 0.0 {
                            lo = mid
                        } else {
                            hi = mid
                        }
                    }
                    lmax = lmax.max(hi);
                }
                lmax
            }
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
            let (th, objective, r, kkt, converged, n_iter) = if cfg.is_paper() {
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
            if cfg.is_paper() {
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

    fn penalty(&self, lambda: f64) -> AtomPenalty<Atoms> {
        AtomPenalty {
            atoms: self.atoms.clone(),
            base: self.base.clone(),
            exponent: self.exponent.clone(),
            lambda,
        }
    }

    /// Model dimension of the selected model (`docs/project3/METHODOLOGY.md`
    /// §5): every coefficient minus the penalized ones that are exactly
    /// zero, or for eigen-gaps with `K` distinct eigenvalues of
    /// multiplicities `m_i`, `d + (K − 1) + k(k−1)/2 − Σ m_i(m_i−1)/2`.
    fn model_df(&self, theta: &[f64], active: &[bool]) -> f64 {
        let total = self.free.len() as f64;
        match &self.atoms {
            Atoms::Gaps(gp) => {
                let k = gp.p;
                let mut mult = vec![1usize];
                for &a in active {
                    if a {
                        mult.push(1);
                    } else {
                        *mult.last_mut().unwrap() += 1;
                    }
                }
                let kk = mult.len();
                let shape_df = (kk - 1) as f64 + (k * (k - 1) / 2) as f64
                    - mult.iter().map(|&m| (m * (m - 1) / 2) as f64).sum::<f64>();
                self.data.d as f64 + shape_df
            }
            a => {
                let zeros = a.support().iter().filter(|&&i| theta[i] == 0.0).count();
                total - zeros as f64
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
        let pen = AtomPenalty {
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
            Atoms::Gaps(_) => {
                if pt.active.iter().all(|&a| !a) {
                    mask = isotropic_mask(self.data.d, self.data.p_total);
                } else if pt.active.iter().any(|&a| !a) {
                    return Err(EsagError::Data(
                        "tie patterns are not a coefficient mask".into(),
                    ));
                }
            }
            a => {
                // penalized coefficients that are exactly zero stay zero
                let support = a.support();
                let zero_at = |i: usize| self.theta_at_zero(&pt.coef, i);
                for i in support {
                    if zero_at(i) {
                        let (k, j) = self.free[i];
                        mask[(k, j)] = false;
                    }
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

    /// Whether free parameter `i` of `coef` (original scale) is an exact
    /// zero of the penalized fit (slopes are zero on both scales; the
    /// intercept is never a zero of a slope penalty).
    fn theta_at_zero(&self, coef: &DMatrix<f64>, i: usize) -> bool {
        let (k, j) = self.free[i];
        coef[(k, j)] == 0.0
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
            Atoms::SparseGroups(g) => g.atoms(theta, out),
            Atoms::Gaps(g) => g.atoms(theta, out),
        }
    }
    fn prox(&self, v: &mut [f64], t: &[f64]) {
        match self {
            Atoms::Groups(g) => g.prox(v, t),
            Atoms::SparseGroups(g) => g.prox(v, t),
            Atoms::Gaps(g) => g.prox(v, t),
        }
    }
    fn support(&self) -> Vec<usize> {
        match self {
            Atoms::Groups(g) => g.support(),
            Atoms::SparseGroups(g) => g.support(),
            Atoms::Gaps(g) => g.support(),
        }
    }
}

fn solve(
    obj: &SmoothNegLogLik,
    pen: &AtomPenalty<Atoms>,
    theta0: &[f64],
    tol: f64,
) -> Result<crate::optim::prox::ProxResult> {
    solve_warm(obj, pen, theta0, tol, None)
}

fn solve_warm(
    obj: &SmoothNegLogLik,
    pen: &AtomPenalty<Atoms>,
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
    pen: &'a AtomPenalty<Atoms>,
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
