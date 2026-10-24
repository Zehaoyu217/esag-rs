//! Standard normal distribution primitives (R's `dnorm`, `pnorm`, `qnorm`).

/// 1/sqrt(2π)
pub const INV_SQRT_2PI: f64 = 0.398_942_280_401_432_7;
/// log(2π)
pub const LN_2PI: f64 = 1.837_877_066_409_345_5;

/// Standard normal density φ(x) (R: `dnorm(x)`).
#[inline]
pub fn norm_pdf(x: f64) -> f64 {
    INV_SQRT_2PI * (-0.5 * x * x).exp()
}

/// log φ(x) (R: `dnorm(x, log = TRUE)`).
#[inline]
pub fn norm_logpdf(x: f64) -> f64 {
    -0.5 * x * x - 0.5 * LN_2PI
}

/// Standard normal CDF Φ(x) (R: `pnorm(x)`), computed from `erfc` so that it
/// keeps full relative accuracy in the lower tail (down to x ≈ -37, where it
/// underflows).
#[inline]
pub fn norm_cdf(x: f64) -> f64 {
    0.5 * libm::erfc(-x / core::f64::consts::SQRT_2)
}

/// Upper tail 1 - Φ(x) (R: `pnorm(x, lower.tail = FALSE)`), accurate for
/// large positive x.
#[inline]
pub fn norm_sf(x: f64) -> f64 {
    0.5 * libm::erfc(x / core::f64::consts::SQRT_2)
}

/// Standard normal quantile function Φ⁻¹(p), mirroring R's `qnorm(p)`.
///
/// Wichura's algorithm AS 241 (`PPND16`), which is also what R's `qnorm`
/// uses, with the same branch points (`|p - ½| <= 0.425`, then
/// `r = √(-ln min(p, 1 - p)) <= 5`). R's extra asymptotic branch
/// (`r >= 816`) is reachable only with `log.p = TRUE`, so for every double
/// `p` this agrees with `qnorm(p)` to within 1e-15 relative (checked against
/// R 4.6.1 on 10⁴ points from 5e-324 to 1 - 1e-15). Returns `-inf` for
/// `p = 0`, `+inf` for `p = 1`, and NaN for `p` outside `[0, 1]` or NaN.
///
/// Note that `norm_quantile(norm_cdf(x))` cannot recover a large positive
/// `x`: Φ(x) rounds to a double whose distance from 1 carries little
/// information (e.g. Φ(8) = 1 - 6.2e-16 rounds to 1 - 6.7e-16, and
/// `qnorm(pnorm(8))` is 7.99157… in R as well). Use the upper tail,
/// `-norm_quantile(norm_sf(x))`, in that case.
pub fn norm_quantile(p: f64) -> f64 {
    if p.is_nan() || !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if p == 0.0 {
        return f64::NEG_INFINITY;
    }
    if p == 1.0 {
        return f64::INFINITY;
    }
    let q = p - 0.5;
    if q.abs() <= 0.425 {
        // Central region: rational approximation in r = 0.180625 - q².
        let r = 0.180_625 - q * q;
        return q * horner(&PPND16_A, r) / horner(&PPND16_B, r);
    }
    // Tails: rational approximation in sqrt(-log(min(p, 1 - p))).
    let r = if q < 0.0 { p } else { 1.0 - p };
    let r = (-r.ln()).sqrt();
    let val = if r <= 5.0 {
        let r = r - 1.6;
        horner(&PPND16_C, r) / horner(&PPND16_D, r)
    } else {
        let r = r - 5.0;
        horner(&PPND16_E, r) / horner(&PPND16_F, r)
    };
    if q < 0.0 { -val } else { val }
}

/// Evaluate `c[0] + c[1] r + … + c[7] r^7` by Horner's rule.
#[inline]
fn horner(c: &[f64; 8], r: f64) -> f64 {
    c.iter().rev().fold(0.0, |acc, &ci| acc * r + ci)
}

// AS 241 PPND16 coefficients (Wichura 1988), degree-0 term first; the
// denominators have a leading coefficient of 1.
// The literals keep the published digits (more than an f64 holds) so they can
// be checked against the paper; clippy's excessive_precision is silenced.
#[allow(clippy::excessive_precision)]
const PPND16_A: [f64; 8] = [
    3.387_132_872_796_366_608,
    1.331_416_678_917_843_774_5e2,
    1.971_590_950_306_551_442_7e3,
    1.373_169_376_550_946_112_5e4,
    4.592_195_393_154_987_145_7e4,
    6.726_577_092_700_870_085_3e4,
    3.343_057_558_358_812_810_5e4,
    2.509_080_928_730_122_672_7e3,
];
#[allow(clippy::excessive_precision)]
const PPND16_B: [f64; 8] = [
    1.0,
    4.231_333_070_160_091_125_2e1,
    6.871_870_074_920_579_083_0e2,
    5.394_196_021_424_751_107_7e3,
    2.121_379_430_158_659_586_7e4,
    3.930_789_580_009_271_061_0e4,
    2.872_908_573_572_194_267_4e4,
    5.226_495_278_852_854_561_0e3,
];
#[allow(clippy::excessive_precision)]
const PPND16_C: [f64; 8] = [
    1.423_437_110_749_683_577_34,
    4.630_337_846_156_545_295_90,
    5.769_497_221_460_691_405_50,
    3.647_848_324_763_204_605_04,
    1.270_458_252_452_368_382_58,
    2.417_807_251_774_506_117_70e-1,
    2.272_384_498_926_918_458_33e-2,
    7.745_450_142_783_414_076_40e-4,
];
#[allow(clippy::excessive_precision)]
const PPND16_D: [f64; 8] = [
    1.0,
    2.053_191_626_637_758_821_87,
    1.676_384_830_183_803_849_40,
    6.897_673_349_851_000_045_50e-1,
    1.481_039_764_274_800_745_90e-1,
    1.519_866_656_361_645_719_66e-2,
    5.475_938_084_995_344_946_00e-4,
    1.050_750_071_644_416_843_24e-9,
];
#[allow(clippy::excessive_precision)]
const PPND16_E: [f64; 8] = [
    6.657_904_643_501_103_777_20,
    5.463_784_911_164_114_369_90,
    1.784_826_539_917_291_335_80,
    2.965_605_718_285_048_912_30e-1,
    2.653_218_952_657_612_309_30e-2,
    1.242_660_947_388_078_438_60e-3,
    2.711_555_568_743_487_578_15e-5,
    2.010_334_399_292_288_132_65e-7,
];
#[allow(clippy::excessive_precision)]
const PPND16_F: [f64; 8] = [
    1.0,
    5.998_322_065_558_879_376_90e-1,
    1.369_298_809_227_358_053_10e-1,
    1.487_536_129_085_061_485_25e-2,
    7.868_691_311_456_132_591_00e-4,
    1.846_318_317_510_054_681_80e-5,
    1.421_511_758_316_445_888_70e-7,
    2.044_263_103_389_939_785_64e-15,
];

#[cfg(test)]
// Reference values are R output printed with sprintf("%.17g").
#[allow(clippy::excessive_precision)]
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
    #[test]
    fn quantile_matches_r_qnorm() {
        // R: print(qnorm(p), digits = 17) for each p below.
        let cases = [
            (1e-300, -37.047096299361201),
            (1e-10, -6.3613409024040566),
            (1e-5, -4.2648907939228247),
            (0.001, -3.0902323061678132),
            (0.025, -1.9599639845400538),
            (0.3, -0.52440051270804078),
            (0.7, 0.52440051270804067),
            (0.9, 1.2815515655446006),
            (0.975, 1.9599639845400534),
            (0.999, 3.0902323061678132),
            (1.0 - 1e-10, 6.3613408896974226),
        ];
        for (p, want) in cases {
            assert_relative_eq!(norm_quantile(p), want, max_relative = 1e-14);
        }
        assert_eq!(norm_quantile(0.5), 0.0);
        assert_eq!(norm_quantile(0.0), f64::NEG_INFINITY);
        assert_eq!(norm_quantile(1.0), f64::INFINITY);
        assert!(norm_quantile(-0.1).is_nan());
        assert!(norm_quantile(1.1).is_nan());
        assert!(norm_quantile(f64::NAN).is_nan());
    }

    #[test]
    fn quantile_inverts_cdf() {
        // Invert in the tail that carries the information: Φ(x) for x > 0
        // (and 1 - Φ(x) for x < 0) rounds towards 1, so the round trip is
        // only well posed through the smaller tail probability.
        for &x in &[
            -37.0, -20.0, -8.0, -3.0, -1.0, -0.2, 0.0, 0.2, 1.0, 3.0, 8.0, 20.0, 37.0,
        ] {
            let back = if x <= 0.0 {
                norm_quantile(norm_cdf(x))
            } else {
                -norm_quantile(norm_sf(x))
            };
            assert_relative_eq!(back, x, max_relative = 1e-13, epsilon = 1e-15);
        }
        // Through the upper half the round trip is limited by the rounding of
        // p, exactly as in R: qnorm(pnorm(8)) = 7.9915739504939038 and
        // qnorm(pnorm(3)) = 2.9999999999999978 (an ulp of Φ(3) is worth
        // ~2.5e-14 in x there).
        assert_relative_eq!(
            norm_quantile(norm_cdf(8.0)),
            7.9915739504939038,
            max_relative = 1e-14
        );
        assert_relative_eq!(norm_quantile(norm_cdf(3.0)), 3.0, max_relative = 2e-14);
    }
}
