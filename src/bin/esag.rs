//! `esag` command-line tool: simulate, fit, predict, diagnose and test ESAG
//! regressions on CSV data.
//!
//! ```text
//! esag simulate --n 500 --out data.csv
//! esag fit data.csv --y y1,y2,y3 --x x1 --model model.json
//! esag predict model.json data.csv --x x1 --out pred.csv
//! esag gof model.json data.csv --y y1,y2,y3 --x x1 --boot 200
//! esag test model.json data.csv --y y1,y2,y3 --x x1 --null isotropy --boot 200
//! esag penalize data.csv --y y1,y2,y3 --x x1,x2 --method drivers
//! ```

use clap::{Parser, Subcommand, ValueEnum};
use esag::FastRng;
use esag::distribution::Esag;
use esag::inference::{
    GofMethod, Restriction, bootstrap_gof, hypothesis_test, prediction_region, score_test,
};
use esag::penalized::{Grouping, LambdaGrid, PenalizedEsag, Target};
use esag::regression::{EsagRegressor, EsagRegressorFit, Shape};
use nalgebra::DMatrix;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};
use std::error::Error;
use std::path::PathBuf;
use std::time::Instant;

type Res<T> = Result<T, Box<dyn Error>>;

/// `println!` that exits quietly when stdout is a closed pipe (`esag … | head`).
macro_rules! outln {
    ($($t:tt)*) => {{
        use std::io::Write as _;
        if let Err(e) = writeln!(std::io::stdout(), $($t)*) {
            if e.kind() == std::io::ErrorKind::BrokenPipe {
                std::process::exit(0);
            }
            return Err(e.into());
        }
    }};
}

fn is_broken_pipe(e: &(dyn Error + 'static)) -> bool {
    if let Some(io) = e.downcast_ref::<std::io::Error>() {
        return io.kind() == std::io::ErrorKind::BrokenPipe;
    }
    if let Some(c) = e.downcast_ref::<csv::Error>()
        && let csv::ErrorKind::Io(io) = c.kind()
    {
        return io.kind() == std::io::ErrorKind::BrokenPipe;
    }
    false
}

#[derive(Parser)]
#[command(
    name = "esag",
    version,
    about = "ESAG regression for directional data on the sphere"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Clone, Copy, ValueEnum)]
enum NullKind {
    /// gamma = 0 (isotropic errors)
    Isotropy,
    /// the mu coefficients of --feature are 0
    Mu,
    /// the gamma coefficients of --feature are 0
    Gamma,
    /// both mu and gamma coefficients of --feature are 0
    Feature,
    /// the shape coefficients of every feature are 0 (an anisotropic shape,
    /// the same at every x)
    ConstantShape,
}

#[derive(Clone, Copy, ValueEnum)]
enum ShapeKind {
    /// Yu & Huang's gamma (the papers' model)
    Gamma,
    /// the smooth log-tangent coordinates (no singular set; recommended for
    /// regressions with covariates at d >= 4)
    LogTangent,
    /// gamma in the tangent frame transported from the data's mean
    /// direction (no pole at mu_0 = mu_1 = 0)
    GammaTransported,
    /// gamma made smooth: eigenvalue ratios and rotation vectors per group, in
    /// the transported frame (no singular set; at d = 3 Paine et al.'s gamma)
    GammaRot,
}

#[derive(Clone, Copy, ValueEnum)]
enum PenaltyKind {
    /// L1 on each target coefficient
    Lasso,
    /// LASSO with adaptive weights (r = 1 unless --r)
    AdaptiveLasso,
    /// L2 norm per group (--groups)
    GroupLasso,
    /// group LASSO with adaptive weights
    AdaptiveGroupLasso,
    /// mixture of group and L1 penalties (--l1-ratio)
    SparseGroupLasso,
    /// all shape coefficients as one group (isotropy)
    Isotropy,
    /// i.i.d.: ties between the eigenvalues of the shape
    EigenGap,
}

#[derive(Clone, Copy, ValueEnum)]
enum TargetKind {
    /// every slope
    Slopes,
    /// the mean-direction slopes
    Mean,
    /// the shape slopes
    ShapeSlopes,
    /// every shape coefficient (intercept included)
    Shape,
}

#[derive(Clone, Copy, ValueEnum)]
enum GroupKind {
    /// one group per covariate
    Covariate,
    /// per covariate: its mean slopes and its shape slopes
    CovariatePart,
    /// one group per parameter row
    Parameter,
    /// all target coefficients together
    All,
    /// the paper's gamma groups (gamma shape)
    Gamma,
}

#[derive(Clone, Copy, ValueEnum)]
enum SelectKind {
    Bic,
    Aic,
    /// cross-validated held-out log-likelihood
    Cv,
}

#[derive(Subcommand)]
enum Cmd {
    /// Write a simulated d = 3 regression data set (columns x1, y1, y2, y3).
    Simulate {
        #[arg(long, default_value_t = 500)]
        n: usize,
        #[arg(long, default_value_t = 2024)]
        seed: u64,
        #[arg(long)]
        out: PathBuf,
    },
    /// Fit an ESAG regression and save the model as JSON.
    Fit {
        data: PathBuf,
        /// Comma-separated response columns (d >= 3).
        #[arg(long)]
        y: String,
        /// Comma-separated covariate columns (omit for an intercept-only fit).
        #[arg(long, default_value = "")]
        x: String,
        #[arg(long)]
        model: Option<PathBuf>,
        /// Number of random restarts.
        #[arg(long, default_value_t = 0)]
        restarts: usize,
        /// Shape parameterization.
        #[arg(long, value_enum, default_value = "gamma")]
        shape: ShapeKind,
    },
    /// Predict mean directions (and eigenvalues) for new covariates.
    Predict {
        model: PathBuf,
        data: PathBuf,
        #[arg(long, default_value = "")]
        x: String,
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// Bootstrap goodness-of-fit test (Yu & Huang 2024, Algorithm 1).
    Gof {
        model: PathBuf,
        data: PathBuf,
        #[arg(long)]
        y: String,
        #[arg(long, default_value = "")]
        x: String,
        #[arg(long, default_value_t = 200)]
        boot: usize,
        /// Use the R code's one-sample chi-square variant instead.
        #[arg(long)]
        chisq: bool,
        #[arg(long, default_value_t = 1)]
        seed: u64,
    },
    /// Bootstrap hypothesis test (RoC, D, M, LR, LM) against a restricted model.
    Test {
        model: PathBuf,
        data: PathBuf,
        #[arg(long)]
        y: String,
        #[arg(long, default_value = "")]
        x: String,
        #[arg(long, value_enum)]
        null: NullKind,
        /// 0-based feature index for --null mu|gamma|feature.
        #[arg(long, default_value_t = 0)]
        feature: usize,
        #[arg(long, default_value_t = 200)]
        boot: usize,
        /// Monte-Carlo draws per observation for the M statistic (0 = exact
        /// moments, recommended; the R code uses Monte Carlo).
        #[arg(long, default_value_t = 0)]
        mc: usize,
        #[arg(long, default_value_t = 1)]
        seed: u64,
        /// Score test: refit only the restricted model (LM and M); 1.4-1.9x
        /// faster for log-tangent, 2-4x (up to 29x) for gamma.
        #[arg(long)]
        score: bool,
    },
    /// LASSO-type penalized fit along a lambda path with lambda selection
    /// (Yu & Huang, Project 3; see docs/MATH.md section 8).
    Penalize {
        data: PathBuf,
        #[arg(long)]
        y: String,
        #[arg(long, default_value = "")]
        x: String,
        #[arg(long, value_enum, default_value = "adaptive-lasso")]
        penalty: PenaltyKind,
        /// Which coefficients to penalize (default: the preset's).
        #[arg(long, value_enum)]
        target: Option<TargetKind>,
        /// How to group them (group and sparse-group LASSO).
        #[arg(long, value_enum)]
        groups: Option<GroupKind>,
        /// L1 share of the sparse group LASSO.
        #[arg(long, default_value_t = 0.5)]
        l1_ratio: f64,
        /// Adaptive exponent r (weights ‖θ̃‖^-r); 0 = plain weights.
        #[arg(long)]
        r: Option<f64>,
        #[arg(long, value_enum, default_value = "log-tangent")]
        shape: ShapeKind,
        /// Reproduce one of the paper's six methods as coded in its R scripts
        /// (LASSO, aLASSO, RoC-LASSO, GLASSO, aGLASSO, RoC-GLASSO; i.i.d.).
        #[arg(long)]
        paper: Option<String>,
        #[arg(long, value_enum, default_value = "bic")]
        select: SelectKind,
        /// Use the paper's grid 0, 0.5, 1, 2, ..., 256 (default: automatic).
        #[arg(long)]
        paper_grid: bool,
        #[arg(long, default_value_t = 10)]
        folds: usize,
        #[arg(long, default_value_t = 1)]
        seed: u64,
    },
    /// Prediction region thresholds at a covariate vector.
    Region {
        model: PathBuf,
        data: PathBuf,
        #[arg(long)]
        y: String,
        #[arg(long, default_value = "")]
        x: String,
        /// Comma-separated covariate values.
        #[arg(long, allow_hyphen_values = true)]
        at: String,
        #[arg(long, default_value_t = 5000)]
        mc: usize,
        #[arg(long, default_value_t = 50)]
        boot: usize,
        #[arg(long, default_value_t = 1)]
        seed: u64,
    },
}

fn split_cols(s: &str) -> Vec<String> {
    s.split(',')
        .map(str::trim)
        .filter(|c| !c.is_empty())
        .map(String::from)
        .collect()
}

/// Read the requested columns of a CSV file with a header row.
fn read_cols(path: &PathBuf, cols: &[String]) -> Res<DMatrix<f64>> {
    let mut rdr = csv::Reader::from_path(path)?;
    let header = rdr.headers()?.clone();
    let idx: Vec<usize> = cols
        .iter()
        .map(|c| {
            header
                .iter()
                .position(|h| h == c)
                .ok_or_else(|| format!("column '{c}' not found in {}", path.display()))
        })
        .collect::<Result<_, _>>()?;
    let mut rows: Vec<Vec<f64>> = Vec::new();
    for rec in rdr.records() {
        let rec = rec?;
        rows.push(
            idx.iter()
                .map(|&i| rec[i].trim().parse::<f64>())
                .collect::<Result<_, _>>()?,
        );
    }
    Ok(DMatrix::from_fn(rows.len(), cols.len(), |i, j| rows[i][j]))
}

fn load_xy(data: &PathBuf, y: &str, x: &str) -> Res<(DMatrix<f64>, DMatrix<f64>, Vec<String>)> {
    let yc = split_cols(y);
    let xc = split_cols(x);
    let ym = read_cols(data, &yc)?;
    let xm = if xc.is_empty() {
        DMatrix::zeros(ym.nrows(), 0)
    } else {
        read_cols(data, &xc)?
    };
    Ok((xm, ym, xc))
}

fn load_model(path: &PathBuf) -> Res<EsagRegressorFit> {
    Ok(EsagRegressorFit::from_json(&std::fs::read_to_string(
        path,
    )?)?)
}

fn main() {
    if let Err(e) = run() {
        if is_broken_pipe(e.as_ref()) {
            std::process::exit(0);
        }
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn run() -> Res<()> {
    match Cli::parse().cmd {
        Cmd::Simulate { n, seed, out } => {
            let b = DMatrix::from_column_slice(
                5,
                2,
                &[1.5, 0.5, 1.0, 0.8, -0.4, -0.6, 1.2, 0.3, 0.5, 0.7],
            );
            let mut rng = FastRng::seed_from_u64(seed);
            let mut w = csv::Writer::from_path(&out)?;
            w.write_record(["x1", "y1", "y2", "y3"])?;
            let mut buf = [0.0; 3];
            for _ in 0..n {
                let x: f64 = StandardNormal.sample(&mut rng);
                let eta: Vec<f64> = (0..5).map(|k| b[(k, 0)] + b[(k, 1)] * x).collect();
                let dist = Esag::from_eta(&eta)?;
                dist.sample_one(&dist.frame(), &mut rng, &mut buf);
                w.write_record(
                    [x, buf[0], buf[1], buf[2]]
                        .iter()
                        .map(|v| format!("{v:.17}")),
                )?;
            }
            w.flush()?;
            outln!(
                "wrote {n} rows to {} (true B, rows mu1..mu3,gamma1,gamma2; cols intercept,x1):\n{b:.3}",
                out.display()
            );
        }
        Cmd::Fit {
            data,
            y,
            x,
            model,
            restarts,
            shape,
        } => {
            let (xm, ym, names) = load_xy(&data, &y, &x)?;
            let t0 = Instant::now();
            let fit = EsagRegressor::new()
                .with_feature_names(names)
                .with_response_names(split_cols(&y))
                .with_restarts(restarts, 0)
                .with_shape(match shape {
                    ShapeKind::Gamma => Shape::Gamma,
                    ShapeKind::LogTangent => Shape::LogTangent,
                    ShapeKind::GammaTransported => Shape::GammaTransported,
                    ShapeKind::GammaRot => Shape::GammaRot,
                })
                .fit(&xm, &ym)?;
            outln!("{}", fit.summary());
            outln!("fitted in {:.1} ms", t0.elapsed().as_secs_f64() * 1e3);
            if let Some(p) = model {
                std::fs::write(&p, fit.to_json()?)?;
                outln!("model saved to {}", p.display());
            }
        }
        Cmd::Predict {
            model,
            data,
            x,
            out,
        } => {
            let fit = load_model(&model)?;
            let xc = split_cols(&x);
            let xm = if xc.is_empty() {
                let n = csv::Reader::from_path(&data)?.records().count();
                DMatrix::zeros(n, 0)
            } else {
                read_cols(&data, &xc)?
            };
            let dir = fit.predict(&xm)?;
            let dists = fit.predict_distribution(&xm)?;
            let d = fit.d;
            let mut header: Vec<String> = (1..=d).map(|k| format!("mean_dir{k}")).collect();
            header.push("mu_norm".into());
            header.extend((1..d).map(|k| format!("lambda{k}")));
            let mut w: csv::Writer<Box<dyn std::io::Write>> = match out {
                Some(p) => csv::Writer::from_writer(Box::new(std::fs::File::create(p)?)),
                None => csv::Writer::from_writer(Box::new(std::io::stdout())),
            };
            w.write_record(&header)?;
            for (i, dist) in dists.iter().enumerate() {
                let mut rec: Vec<String> = (0..d).map(|k| format!("{:.10}", dir[(i, k)])).collect();
                rec.push(format!("{:.10}", dist.mu_norm()));
                rec.extend(dist.angles().lambda.iter().map(|l| format!("{l:.10}")));
                w.write_record(&rec)?;
            }
            w.flush()?;
        }
        Cmd::Gof {
            model,
            data,
            y,
            x,
            boot,
            chisq,
            seed,
        } => {
            let fit = load_model(&model)?;
            let (xm, ym, _) = load_xy(&data, &y, &x)?;
            let method = if chisq {
                GofMethod::OneSampleChiSquare
            } else {
                GofMethod::TwoSampleReference
            };
            let t0 = Instant::now();
            let r = bootstrap_gof(&fit, &xm, &ym, method, boot, seed)?;
            outln!(
                "goodness of fit ({method:?}): KS distance {:.4}, KS p {:.4}; bootstrap p-value {:.3} ({} replicates, {} failed, {:.1} s)",
                r.ks_statistic,
                r.ks_p_value,
                r.p_value,
                r.n_boot,
                r.n_failed,
                t0.elapsed().as_secs_f64()
            );
        }
        Cmd::Test {
            model,
            data,
            y,
            x,
            null,
            feature,
            boot,
            mc,
            seed,
            score,
        } => {
            let fit = load_model(&model)?;
            let (xm, ym, _) = load_xy(&data, &y, &x)?;
            let restriction = match null {
                NullKind::Isotropy => Restriction::Isotropy,
                NullKind::Mu => Restriction::MuOfFeature(feature),
                NullKind::Gamma => Restriction::GammaOfFeature(feature),
                NullKind::Feature => Restriction::Feature(feature),
                NullKind::ConstantShape => Restriction::ConstantShape,
            };
            let t0 = Instant::now();
            let r = if score {
                score_test(&fit, &xm, &ym, restriction, boot, mc, seed)?
            } else {
                hypothesis_test(&fit, &xm, &ym, restriction, boot, mc, seed)?
            };
            outln!("{}", r.summary());
            outln!("{:.2} s", t0.elapsed().as_secs_f64());
        }
        Cmd::Penalize {
            data,
            y,
            x,
            penalty,
            target,
            groups,
            l1_ratio,
            r,
            shape,
            paper,
            select,
            paper_grid,
            folds,
            seed,
        } => {
            let (xm, ym, names) = load_xy(&data, &y, &x)?;
            let mut est = match &paper {
                Some(name) => PenalizedEsag::paper(name)?,
                None => {
                    let mut e = match penalty {
                        PenaltyKind::Lasso => PenalizedEsag::lasso(),
                        PenaltyKind::AdaptiveLasso => PenalizedEsag::adaptive_lasso(),
                        PenaltyKind::GroupLasso => PenalizedEsag::group_lasso(),
                        PenaltyKind::AdaptiveGroupLasso => PenalizedEsag::adaptive_group_lasso(),
                        PenaltyKind::SparseGroupLasso => {
                            PenalizedEsag::sparse_group_lasso(l1_ratio)
                        }
                        PenaltyKind::Isotropy => PenalizedEsag::isotropy(),
                        PenaltyKind::EigenGap => PenalizedEsag::eigen_gap(),
                    };
                    if let Some(t) = target {
                        e = e.target(match t {
                            TargetKind::Slopes => Target::Slopes,
                            TargetKind::Mean => Target::MeanSlopes,
                            TargetKind::ShapeSlopes => Target::ShapeSlopes,
                            TargetKind::Shape => Target::Shape,
                        });
                    }
                    if let Some(g) = groups {
                        e = e.grouping(match g {
                            GroupKind::Covariate => Grouping::Covariate,
                            GroupKind::CovariatePart => Grouping::CovariatePart,
                            GroupKind::Parameter => Grouping::Parameter,
                            GroupKind::All => Grouping::All,
                            GroupKind::Gamma => Grouping::GammaGroups,
                        });
                    }
                    if let Some(r) = r {
                        e = if r == 0.0 { e.plain() } else { e.adaptive(r) };
                    }
                    e.shape(match shape {
                        ShapeKind::Gamma => Shape::Gamma,
                        ShapeKind::LogTangent => Shape::LogTangent,
                        ShapeKind::GammaTransported => Shape::GammaTransported,
                        ShapeKind::GammaRot => Shape::GammaRot,
                    })
                }
            };
            est = match select {
                SelectKind::Bic => est.bic(),
                SelectKind::Aic => est.aic(),
                SelectKind::Cv => est.cv(folds, seed),
            };
            if paper_grid {
                est = est.with_grid(LambdaGrid::Paper);
            }
            est = est.feature_names(names).response_names(split_cols(&y));
            let t0 = Instant::now();
            let path = est.fit(&xm, &ym)?;
            outln!("{}", path.path_summary());
            outln!("{}", path.summary());
            outln!(
                "path of {} lambdas in {:.2} s",
                path.points.len(),
                t0.elapsed().as_secs_f64()
            );
        }
        Cmd::Region {
            model,
            data,
            y,
            x,
            at,
            mc,
            boot,
            seed,
        } => {
            let fit = load_model(&model)?;
            let (xm, ym, _) = load_xy(&data, &y, &x)?;
            let x_new: Vec<f64> = split_cols(&at)
                .iter()
                .map(|v| v.parse::<f64>())
                .collect::<Result<_, _>>()?;
            let levels = [0.95, 0.90, 0.85, 0.80, 0.75, 0.70];
            let pr = prediction_region(&fit, &xm, &ym, &x_new, mc, boot, &levels, seed)?;
            outln!(
                "mean direction at {x_new:?}: {:?}",
                pr.dist.mean_direction()
            );
            outln!("region: {{y : (y - m)^T V^-1 (y - m) <= q}}");
            for (l, q) in levels.iter().zip(&pr.thresholds) {
                outln!("  {:>3.0}%  q = {q:.4}", l * 100.0);
            }
        }
    }
    Ok(())
}
