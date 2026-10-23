//! # esag — Elliptically Symmetric Angular Gaussian distributions in Rust
//!
//! A port and re-engineering of Zehao Yu's ESAG R code (Yu & Huang,
//! arXiv:2212.05634) for modelling directional data on the unit sphere
//! S^{d-1}: the ESAG(μ, γ) distribution, maximum-likelihood fitting,
//! ESAG regression with covariates (scikit-learn-style `fit`/`predict`/`score`
//! API), residual diagnostics, bootstrap goodness-of-fit, prediction regions,
//! and bootstrap hypothesis tests.
//!
//! See `ARCHITECTURE.md` in the repository for the design and `docs/MATH.md`
//! for the mathematics.

pub mod scalar;
pub mod dual;
pub mod dims;
pub mod param;
pub mod onb;
pub mod rotation;
pub mod frame;
pub mod loglik;
pub mod stats;
pub mod optim;
pub mod error;
pub mod distribution;
pub mod regression;
pub mod estimator;
pub mod inference;

pub use scalar::Scalar;
