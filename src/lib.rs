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

pub mod adjoint;
pub mod dims;
pub mod distribution;
pub mod dual;
pub mod error;
pub mod estimator;
pub mod frame;
pub mod gamma_smooth;
pub mod gamma_transported;
pub mod hdual;
pub mod inference;
pub mod loglik;
pub mod logtangent;
pub mod model;
pub mod onb;
pub mod optim;
pub mod param;
pub mod penalized;
pub mod preprocessing;
pub mod regression;
pub mod rotation;
pub mod scalar;
pub mod stats;
pub(crate) mod summary;

pub use scalar::Scalar;

/// The random-number generator used for every internal Monte-Carlo and
/// bootstrap stream: xoshiro256++ (fast, excellent statistical quality, and
/// a fixed algorithm, so seeded results are reproducible across versions of
/// `rand`). Replicate `b` of a bootstrap uses its own generator seeded from
/// the user's seed, so results do not depend on the number of threads.
pub type FastRng = rand::rngs::Xoshiro256PlusPlus;

/// SplitMix64 finaliser (bijective 64-bit mixer).
#[inline]
const fn splitmix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Purpose tags for [`stream_rng`]; every internal random stream has its
/// own tag so no two streams share a seed.
pub(crate) mod streams {
    pub const GOF: u64 = 1;
    pub const REGION_BOOT: u64 = 2;
    pub const REGION_MC: u64 = 3;
    pub const COEF_BOOT: u64 = 4;
    pub const TEST_BOOT: u64 = 5;
    /// Monte-Carlo moments of the `M` statistic; the replicate (0 for the
    /// observed data, `b + 1` for bootstrap replicate `b`) is folded into
    /// the stream id.
    pub const TEST_MOMENTS: u64 = 6;
    pub const RESTARTS: u64 = 7;
    pub const CV_FOLDS: u64 = 8;
}

/// An independent generator for `(seed, stream, index)`: the three are
/// mixed with SplitMix64, so streams with different tags or indices never
/// coincide (seeding with `seed + index` would make, e.g., observation `i`
/// of one stream equal to replicate `i − 1` of another).
#[inline]
pub(crate) fn stream_rng(seed: u64, stream: u64, index: u64) -> FastRng {
    use rand::SeedableRng;
    let s = splitmix64(seed ^ splitmix64(stream.wrapping_mul(0xD1B5_4A32_D192_ED03)));
    FastRng::seed_from_u64(splitmix64(s ^ splitmix64(index ^ 0xA076_1D64_78BD_642F)))
}
