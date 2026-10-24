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

#![forbid(unsafe_code)]
// Index loops mirror the mathematics (and the R code) more clearly than
// iterator chains in the numerical kernels.
#![allow(clippy::needless_range_loop)]
// `!(x > 0.0)` is used on purpose: it is also true for NaN.
#![allow(clippy::neg_cmp_op_on_partial_ord)]

pub mod dims;
pub mod distribution;
pub mod dual;
pub mod error;
pub mod estimator;
pub mod frame;
pub mod loglik;
pub mod onb;
pub mod optim;
pub mod param;
pub mod preprocessing;
pub mod regression;
pub mod rotation;
pub mod scalar;
pub mod stats;
// pub mod inference;    // enabled once the math core + optimizer + stats land

pub use scalar::Scalar;
