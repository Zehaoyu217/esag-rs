//! Statistical utilities: normal and chi-square distributions, Kolmogorov–
//! Smirnov tests, and quantiles (R-compatible conventions).

pub mod chisq;
pub mod ks;
pub mod normal;
pub mod quantile;

pub use chisq::{chisq_cdf, chisq_pdf, chisq_quantile, chisq_sf, gamma_p, gamma_q};
pub use ks::{KsResult, ks_1samp, ks_1samp_chisq, ks_2samp};
pub use normal::{norm_cdf, norm_logpdf, norm_pdf, norm_quantile, norm_sf};
pub use quantile::{
    ecdf_fraction_ge, ecdf_fraction_gt, ecdf_fraction_lt, mean, quantile_type7, quantiles_type7,
    variance,
};
