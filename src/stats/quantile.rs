//! Sample quantiles, moments and empirical-CDF fractions with R's
//! conventions (`quantile(type = 7)`, `mean`, `var`, `mean(x >= t)`).

/// Tolerance R allows on `probs` outside `[0, 1]` before clamping
/// (`100 * .Machine$double.eps`).
const PROB_EPS: f64 = 100.0 * f64::EPSILON;

/// Sample quantile of type 7 (R's default: `quantile(x, prob, type = 7,
/// names = FALSE)`).
///
/// With the sorted sample `x₍₁₎ ≤ … ≤ x₍ₙ₎`, `h = 1 + (n - 1)·prob`,
/// `lo = ⌊h⌋`, `hi = ⌈h⌉`, the result is `x₍lo₎` if `h = lo` or
/// `x₍hi₎ = x₍lo₎`, and `(1 - (h - lo))·x₍lo₎ + (h - lo)·x₍hi₎` otherwise,
/// evaluated in exactly that form so the result is bit-for-bit R's.
///
/// `prob` within `100·ε` of `[0, 1]` is clamped as in R. Returns NaN when
/// `x` is empty, contains NaN, or `prob` is NaN or outside `[0, 1]` (cases
/// where R returns NA or stops with an error). Sorts a copy of `x`; use
/// [`quantiles_type7`] for several probabilities.
pub fn quantile_type7(x: &[f64], prob: f64) -> f64 {
    quantiles_type7(x, &[prob])[0]
}

/// Type-7 sample quantiles for several probabilities (R:
/// `quantile(x, probs, names = FALSE)`); sorts one copy of `x`.
///
/// See [`quantile_type7`] for the definition and edge cases; the output has
/// one entry per element of `probs`, in the same order.
pub fn quantiles_type7(x: &[f64], probs: &[f64]) -> Vec<f64> {
    if x.is_empty() || x.iter().any(|v| v.is_nan()) {
        return vec![f64::NAN; probs.len()];
    }
    let mut sorted = x.to_vec();
    sorted.sort_unstable_by(f64::total_cmp);
    probs.iter().map(|&p| type7_sorted(&sorted, p)).collect()
}

/// Type-7 quantile of an already sorted, non-empty, NaN-free sample.
fn type7_sorted(sorted: &[f64], prob: f64) -> f64 {
    if prob.is_nan() || !(-PROB_EPS..=1.0 + PROB_EPS).contains(&prob) {
        return f64::NAN;
    }
    let prob = prob.clamp(0.0, 1.0);
    let n = sorted.len();
    // R: index <- 1 + max(n - 1, 0) * probs (1-based).
    let index = 1.0 + (n - 1) as f64 * prob;
    let lo = index.floor();
    let hi = index.ceil();
    let xlo = sorted[lo as usize - 1];
    let xhi = sorted[hi as usize - 1];
    if index > lo && xhi != xlo {
        let h = index - lo;
        (1.0 - h) * xlo + h * xhi
    } else {
        xlo
    }
}

/// Arithmetic mean (R: `mean(x)`).
///
/// Uses R's two-pass algorithm: `m = Σx / n`, then, if `m` is finite,
/// `m + Σ(x - m) / n`, which removes most of the rounding error of the first
/// pass. (R accumulates in `long double` where available; the R build on
/// aarch64 macOS has none, so this reproduces it exactly there.) Returns NaN
/// for an empty slice.
pub fn mean(x: &[f64]) -> f64 {
    if x.is_empty() {
        return f64::NAN;
    }
    let n = x.len() as f64;
    let m = x.iter().sum::<f64>() / n;
    if m.is_finite() {
        m + x.iter().map(|&v| v - m).sum::<f64>() / n
    } else {
        m
    }
}

/// Sample variance with denominator `n - 1` (R: `var(x)`).
///
/// Two-pass: the mean is computed as in [`mean`] and the variance as
/// `Σ(x - m)² / (n - 1)`, as in R's `cov.c`. Returns NaN when fewer than two
/// observations are given (R returns NA).
pub fn variance(x: &[f64]) -> f64 {
    if x.len() < 2 {
        return f64::NAN;
    }
    let m = mean(x);
    x.iter().map(|&v| (v - m) * (v - m)).sum::<f64>() / (x.len() - 1) as f64
}

/// Fraction of the sample with `value >= threshold` (R:
/// `mean(sample >= threshold)`), i.e. `1 - F̂(threshold⁻)` for the empirical
/// CDF. NaN entries count as "not ≥" (R would propagate NA). Returns NaN for
/// an empty sample, like R's `mean(logical(0))`.
pub fn ecdf_fraction_ge(sample: &[f64], threshold: f64) -> f64 {
    fraction(sample, |v| v >= threshold)
}

/// Fraction of the sample with `value > threshold` (R:
/// `mean(sample > threshold)`), i.e. `1 - F̂(threshold)` where `F̂` is R's
/// `ecdf(sample)`. Same edge-case conventions as [`ecdf_fraction_ge`].
pub fn ecdf_fraction_gt(sample: &[f64], threshold: f64) -> f64 {
    fraction(sample, |v| v > threshold)
}

/// Fraction of the sample with `value < threshold` (R:
/// `mean(sample < threshold)`), i.e. `F̂(threshold⁻)`. Same edge-case
/// conventions as [`ecdf_fraction_ge`].
pub fn ecdf_fraction_lt(sample: &[f64], threshold: f64) -> f64 {
    fraction(sample, |v| v < threshold)
}

fn fraction(sample: &[f64], pred: impl Fn(f64) -> bool) -> f64 {
    if sample.is_empty() {
        return f64::NAN;
    }
    sample.iter().filter(|&&v| pred(v)).count() as f64 / sample.len() as f64
}

#[cfg(test)]
// Reference values are R output printed with sprintf("%.17g").
#[allow(clippy::excessive_precision)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    const PROBS: [f64; 9] = [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.5, 0.0, 1.0];

    fn s1() -> Vec<f64> {
        vec![3.2, -1.5, 0.7, 10.25, 4.4, 4.4, 2.0]
    }

    /// R: ((1:1000 * 7919) %% 1000) / 7 (exactly reproducible, with ties).
    fn s4() -> Vec<f64> {
        (1..=1000_u32)
            .map(|i| f64::from((i * 7919) % 1000) / 7.0)
            .collect()
    }

    /// R: set.seed(21); rchisq(20, 2).
    #[rustfmt::skip]
    const S5: [f64; 20] = [
        2.4359249269186534, 1.8749480115522072, 4.9941772342209907, 0.010206981338908381,
        2.996499707399229, 4.292378234026879, 1.472432736321887, 5.120015016902185,
        4.2808302333151405, 2.1491594189736132, 3.216260820747078, 0.19818057324209942,
        1.0273252622676434, 5.2282081235435038, 5.5953977352196294, 1.5260617020427485,
        4.3981986680863603, 0.044147024456893925, 4.6468591522848222, 1.0876637907635891,
    ];

    #[test]
    fn type7_matches_r_bit_for_bit() {
        // R 4.6.1: quantile(s, c(0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.5, 0, 1), names = FALSE)
        #[rustfmt::skip]
        let cases: [(Vec<f64>, [f64; 9]); 5] = [
            (s1(), [8.4949999999999957, 6.740000000000002, 4.9849999999999977, 4.4000000000000004, 4.4000000000000004, 4.4000000000000004, 3.2000000000000002, -1.5, 10.25]),
            (s4(), [135.57857142857142, 128.44285714285715, 121.30714285714286, 114.17142857142858, 107.03571428571428, 99.899999999999991, 71.357142857142861, 0.0, 142.71428571428572]),
            (S5.to_vec(), [5.2465676041273106, 5.1308343275663173, 5.01305290162317, 4.7163227686720566, 4.4603637891359753, 4.3241243642447227, 2.7162123171589414, 0.010206981338908381, 5.5953977352196294]),
            (vec![5.5], [5.5; 9]),
            (vec![1.0, 2.0], [1.95, 1.8999999999999999, 1.8500000000000001, 1.8, 1.75, 1.7, 1.5, 1.0, 2.0]),
        ];
        for (x, want) in &cases {
            assert_eq!(&quantiles_type7(x, &PROBS)[..], &want[..]);
            for (&p, &w) in PROBS.iter().zip(want.iter()) {
                assert_eq!(quantile_type7(x, p), w);
            }
        }
        // R: quantile(s1, 1/3) = 2 (h lands exactly on an order statistic).
        assert_eq!(quantile_type7(&s1(), 1.0 / 3.0), 2.0);
        // Input order does not matter.
        let mut rev = S5.to_vec();
        rev.reverse();
        assert_eq!(quantiles_type7(&rev, &PROBS), quantiles_type7(&S5, &PROBS));
    }

    #[test]
    fn type7_edge_cases() {
        assert!(quantile_type7(&[], 0.5).is_nan());
        assert!(quantile_type7(&[1.0, f64::NAN], 0.5).is_nan());
        assert!(quantile_type7(&[1.0, 2.0], f64::NAN).is_nan());
        assert!(quantile_type7(&[1.0, 2.0], 1.1).is_nan());
        assert!(quantile_type7(&[1.0, 2.0], -0.1).is_nan());
        // Within 100 eps of [0, 1] R clamps.
        assert_eq!(quantile_type7(&[1.0, 2.0], 1.0 + 1e-15), 2.0);
        assert_eq!(quantile_type7(&[1.0, 2.0], -1e-15), 1.0);
        assert!(quantiles_type7(&[], &[]).is_empty());
    }

    #[test]
    fn mean_and_variance_match_r() {
        // R 4.6.1: mean(s), var(s).
        assert_eq!(mean(&s1()), 3.3500000000000001);
        assert_relative_eq!(variance(&s1()), 13.700833333333337, max_relative = 1e-15);
        assert_eq!(mean(&s4()), 71.357142857142861);
        assert_relative_eq!(variance(&s4()), 1702.3809523809518, max_relative = 1e-15);
        assert_eq!(mean(&S5), 2.8297437676812032);
        assert_relative_eq!(variance(&S5), 3.5529221919069323, max_relative = 1e-15);
        // Large offset: R's two-pass mean, var = 0.024999904633659753.
        let s6: Vec<f64> = (1..=5).map(|i| 1e10 + f64::from(i) / 10.0).collect();
        assert_eq!(mean(&s6), 10000000000.299999);
        assert_relative_eq!(variance(&s6), 0.024999904633659753, max_relative = 1e-12);
        assert!(mean(&[]).is_nan());
        assert!(variance(&[1.0]).is_nan());
        assert_eq!(variance(&[2.0, 2.0, 2.0]), 0.0);
    }

    #[test]
    fn ecdf_fractions_match_r() {
        // R: mean(s1 >= 4.4), mean(s1 > 4.4), mean(s1 < 4.4).
        assert_eq!(ecdf_fraction_ge(&s1(), 4.4), 0.42857142857142855);
        assert_eq!(ecdf_fraction_gt(&s1(), 4.4), 0.14285714285714285);
        assert_eq!(ecdf_fraction_lt(&s1(), 4.4), 0.5714285714285714);
        // R: mean(s4 >= 100), mean(s4 > 100), mean(s4 < 100).
        assert_eq!(ecdf_fraction_ge(&s4(), 100.0), 0.29999999999999999);
        assert_eq!(ecdf_fraction_gt(&s4(), 100.0), 0.29899999999999999);
        assert_eq!(ecdf_fraction_lt(&s4(), 100.0), 0.69999999999999996);
        assert!(ecdf_fraction_ge(&[], 0.0).is_nan());
        assert!(ecdf_fraction_gt(&[], 0.0).is_nan());
        assert!(ecdf_fraction_lt(&[], 0.0).is_nan());
    }
}
