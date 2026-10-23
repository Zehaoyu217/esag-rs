//! Standard normal distribution primitives.

use core::f64::consts::PI;

/// 1/sqrt(2π)
pub const INV_SQRT_2PI: f64 = 0.398_942_280_401_432_7;
/// log(2π)
pub const LN_2PI: f64 = 1.837_877_066_409_345_5;

/// Standard normal density φ(x).
#[inline]
pub fn norm_pdf(x: f64) -> f64 {
    INV_SQRT_2PI * (-0.5 * x * x).exp()
}

/// log φ(x)
#[inline]
pub fn norm_logpdf(x: f64) -> f64 {
    -0.5 * x * x - 0.5 * LN_2PI
}

/// Standard normal CDF Φ(x), computed from `erfc` so that it keeps full
/// relative accuracy in the lower tail (down to x ≈ -37, where it underflows).
#[inline]
pub fn norm_cdf(x: f64) -> f64 {
    0.5 * libm::erfc(-x / core::f64::consts::SQRT_2)
}

/// Upper tail 1 - Φ(x), accurate for large positive x.
#[inline]
pub fn norm_sf(x: f64) -> f64 {
    0.5 * libm::erfc(x / core::f64::consts::SQRT_2)
}

#[allow(dead_code)]
const _: f64 = PI;

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn cdf_values_match_r() {
        // R: pnorm(c(-8, -3, -1, 0, 1, 3, 8))
        assert_relative_eq!(norm_cdf(-8.0), 6.22096057427178e-16, max_relative = 1e-12);
        assert_relative_eq!(norm_cdf(-3.0), 0.00134989803163009, max_relative = 1e-12);
        assert_relative_eq!(norm_cdf(-1.0), 0.158655253931457, max_relative = 1e-12);
        assert_relative_eq!(norm_cdf(0.0), 0.5, max_relative = 1e-15);
        assert_relative_eq!(norm_cdf(1.0), 0.841344746068543, max_relative = 1e-12);
        assert_relative_eq!(norm_cdf(3.0), 0.998650101968370, max_relative = 1e-12);
        assert_relative_eq!(norm_sf(8.0), 6.22096057427178e-16, max_relative = 1e-12);
    }

    #[test]
    fn pdf_values_match_r() {
        // R: dnorm(c(0, 1, -2.5))
        assert_relative_eq!(norm_pdf(0.0), 0.398942280401433, max_relative = 1e-12);
        assert_relative_eq!(norm_pdf(1.0), 0.241970724519143, max_relative = 1e-12);
        assert_relative_eq!(norm_pdf(-2.5), 0.0175283004935685, max_relative = 1e-12);
        assert_relative_eq!(norm_logpdf(-2.5), norm_pdf(-2.5).ln(), max_relative = 1e-12);
    }
}
