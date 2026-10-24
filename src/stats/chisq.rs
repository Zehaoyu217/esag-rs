//! Regularised incomplete gamma functions and the chi-square distribution,
//! following R's `pgamma`, `pchisq`, `dchisq` and `qchisq` conventions.
//!
//! The incomplete gamma ratio is evaluated with the classical pair of
//! expansions: the power series for `P` when `x < a + 1` and the Legendre
//! continued fraction for `Q` otherwise (terms counted with modified Lentz,
//! then summed bottom-up). For `a < 1` the switch is at `x = 1` instead, as
//! in R's `pgamma`, and below it `Q` comes from R's `pgamma_smallx`
//! complement formula rather than `1 - P`. Wherever one tail is obtained by
//! subtraction it is at least ≈0.13, so both `P` and `Q` keep full relative
//! accuracy in their own tails. The common prefactor `x^a e^{-x} / Γ(a)` is
//! formed so that it does not lose accuracy for large arguments: directly
//! from `exp`, `pow` and `tgamma` for `a < 10`, and with Loader's
//! saddle-point form (`stirlerr` and `bd0`, as in R's `dpois_raw`) for
//! `a >= 10`.
//!
//! Observed accuracy against R 4.6.1 (dense grids): ≤ 1e-14 relative for
//! `0.2 <= df <= 10` and `x < 300` in both tails; in extreme tails with a
//! large shape (results below ~1e-80) the deviation grows to ~2e-13, the
//! conditioning limit `ε·|d ln P / d ln x|` of the problem itself.

use super::normal::norm_quantile;

/// 2π
const TWO_PI: f64 = core::f64::consts::TAU;
/// Lentz's guard against division by zero.
const FPMIN: f64 = 1e-300;
/// Iteration cap for the series and the continued fraction. Both converge
/// in O(sqrt(a)) steps near the switch-over point `x = a + 1`, so this is
/// only reached for absurdly large shapes.
const MAX_ITER: usize = 100_000;

/// Error term of Stirling's formula,
/// `stirlerr(n) = ln Γ(n + 1) - (n + ½) ln n + n - ln √(2π)`, for `n >= 10`
/// (R's internal `stirlerr`). Evaluated with eight terms of the asymptotic
/// series; at `n = 10` the first omitted term is below 1e-17 relative.
fn stirlerr_large(n: f64) -> f64 {
    // B_{2k} / (2k (2k - 1)), k = 1..8.
    const S: [f64; 8] = [
        1.0 / 12.0,
        -1.0 / 360.0,
        1.0 / 1260.0,
        -1.0 / 1680.0,
        1.0 / 1188.0,
        -691.0 / 360_360.0,
        1.0 / 156.0,
        -3617.0 / 122_400.0,
    ];
    let nn = 1.0 / (n * n);
    S.iter().rev().fold(0.0, |acc, &s| acc * nn + s) / n
}

/// Deviance term `bd0(x, np) = x ln(x / np) + np - x` (R's internal `bd0`),
/// with Loader's series when `x` and `np` are close, so it keeps relative
/// accuracy where the naive formula cancels.
fn bd0(x: f64, np: f64) -> f64 {
    if (x - np).abs() < 0.1 * (x + np) {
        let mut v = (x - np) / (x + np);
        let mut s = (x - np) * v;
        if s.abs() < f64::MIN_POSITIVE {
            return s;
        }
        let mut ej = 2.0 * x * v;
        v *= v;
        for j in 1..1000 {
            ej *= v;
            let s1 = s + ej / f64::from(2 * j + 1);
            if s1 == s {
                return s1;
            }
            s = s1;
        }
    }
    let ratio = x / np;
    let log_ratio = if ratio > 0.0 && ratio.is_finite() {
        ratio.ln()
    } else {
        x.ln() - np.ln()
    };
    x * log_ratio + np - x
}

/// The incomplete-gamma prefactor `x^a e^{-x} / Γ(a)` for `a > 0`, `x > 0`
/// (finite). Equals `a · dpois_raw(a, x)` in R's notation.
fn gamma_prefactor(a: f64, x: f64) -> f64 {
    if a < 10.0 {
        if x < 1400.0 {
            // Each factor is accurate to about an ulp, so the product is
            // much better than exp() of a large, cancelling log sum. e^{-x}
            // is split in two so that it does not underflow before the
            // (possibly large) power is applied.
            let h = (-0.5 * x).exp();
            h * (x.powf(a) / libm::tgamma(a)) * h
        } else {
            (a * x.ln() - x - libm::lgamma(a)).exp()
        }
    } else {
        // a · x^a e^{-x} / Γ(a + 1), Γ(a + 1) = √(2πa) (a/e)^a e^{stirlerr(a)}.
        (a / TWO_PI).sqrt() * (-stirlerr_large(a) - bd0(a, x)).exp()
    }
}

/// Power series `Σ_{n≥0} x^n / ((a+1)(a+2)…(a+n))`, so that
/// `P(a, x) = prefactor(a, x) / a · series`.
fn gamma_series(a: f64, x: f64) -> f64 {
    let mut ap = a;
    let mut term = 1.0;
    let mut sum = 1.0;
    for _ in 0..MAX_ITER {
        ap += 1.0;
        term *= x / ap;
        sum += term;
        if term < sum * 1e-17 {
            break;
        }
    }
    sum
}

/// Legendre continued fraction for `Γ(a, x) / (x^a e^{-x})`,
///
/// `1 / (x + 1 - a - 1·(1 - a) / (x + 3 - a - 2·(2 - a) / (x + 5 - a - …)))`,
///
/// so that `Q(a, x) = prefactor(a, x) · cf`. The modified Lentz algorithm
/// finds how many terms are needed; the fraction is then re-evaluated
/// bottom-up with that many terms, which (unlike the forward Lentz product)
/// does not accumulate rounding error over the iterations.
fn gamma_cf(a: f64, x: f64) -> f64 {
    let an = |i: usize| -(i as f64) * (i as f64 - a);
    let bn = |i: usize| x + (2 * i + 1) as f64 - a;
    let mut c = 1.0 / FPMIN;
    let mut d = 1.0 / bn(0);
    let mut terms = MAX_ITER;
    for i in 1..MAX_ITER {
        d = an(i) * d + bn(i);
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = bn(i) + an(i) / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        if (d * c - 1.0).abs() <= f64::EPSILON {
            terms = i + 2;
            break;
        }
    }
    let mut tail = 0.0;
    for i in (1..=terms).rev() {
        tail = an(i) / (bn(i) + tail);
    }
    1.0 / (bn(0) + tail)
}

/// `Q(a, x)` for `a < 1`, `x < 1`, where `P` is close to 1 and `1 - P` would
/// cancel (R's `pgamma_smallx`, upper tail). With
/// `P = x^a / Γ(a + 1) · (1 + s)`, `s = Σ_{n≥1} a (-x)^n / (n! (a + n))`,
/// it evaluates `Q = -(s + e + s·e)` with `e = expm1(a ln x - ln Γ(1 + a))`.
fn gamma_q_smallx(a: f64, x: f64) -> f64 {
    let mut sum = 0.0;
    let mut c = a;
    let mut n = 0.0;
    loop {
        n += 1.0;
        c *= -x / n;
        let term = c / (a + n);
        sum += term;
        if term.abs() <= f64::EPSILON * sum.abs() {
            break;
        }
    }
    let e = (a * x.ln() - libm::lgamma(1.0 + a)).exp_m1();
    -(sum + e + sum * e)
}

/// Returns `(P(a, x), Q(a, x))`, each computed directly in its own small tail.
fn gamma_pq(a: f64, x: f64) -> (f64, f64) {
    if a.is_nan() || x.is_nan() || a < 0.0 {
        return (f64::NAN, f64::NAN);
    }
    if x <= 0.0 {
        return (0.0, 1.0);
    }
    if a == 0.0 {
        // Point mass at zero (R: pgamma(x, shape = 0)).
        return (1.0, 0.0);
    }
    if a == f64::INFINITY {
        return if x == f64::INFINITY {
            (f64::NAN, f64::NAN)
        } else {
            (0.0, 1.0)
        };
    }
    if x == f64::INFINITY {
        return (1.0, 0.0);
    }
    // Series below the classical switch-over x = a + 1. For a < 1 the
    // continued fraction already converges well from x = 1 (as in R's
    // pgamma), and using it there keeps Q accurate where P is close to 1.
    let switch = if a < 1.0 { 1.0 } else { a + 1.0 };
    if x < switch {
        let p = (gamma_prefactor(a, x) / a * gamma_series(a, x)).min(1.0);
        let q = if a < 1.0 {
            gamma_q_smallx(a, x)
        } else {
            1.0 - p
        };
        (p, q)
    } else {
        let q = (gamma_prefactor(a, x) * gamma_cf(a, x)).min(1.0);
        (1.0 - q, q)
    }
}

/// Regularised lower incomplete gamma function
/// `P(a, x) = γ(a, x) / Γ(a)` (R: `pgamma(x, shape = a)`).
///
/// Uses the power series for `x < a + 1` (`x < 1` when `a < 1`) and `1 - Q`
/// from the continued fraction otherwise; relative accuracy is about 1e-15
/// wherever `P` is the smaller tail. Returns 0 for `x <= 0`, 1 for `x = +inf`, the point mass at
/// zero (`P = 1` for `x > 0`) when `a = 0`, and NaN for NaN input or `a < 0`.
pub fn gamma_p(a: f64, x: f64) -> f64 {
    gamma_pq(a, x).0
}

/// Regularised upper incomplete gamma function
/// `Q(a, x) = Γ(a, x) / Γ(a) = 1 - P(a, x)`
/// (R: `pgamma(x, shape = a, lower.tail = FALSE)`).
///
/// Computed directly (continued fraction for `x >= a + 1`), so it keeps full
/// relative accuracy far into the upper tail, e.g. `gamma_q(1, 30) = e^{-30}`.
/// Same conventions as [`gamma_p`] for edge cases.
pub fn gamma_q(a: f64, x: f64) -> f64 {
    gamma_pq(a, x).1
}

/// Chi-square CDF `P(X <= x)` with `df` degrees of freedom
/// (R: `pchisq(x, df)`). Equal to `gamma_p(df / 2, x / 2)`; `df` need not be
/// an integer. NaN for `df < 0` or NaN input.
pub fn chisq_cdf(x: f64, df: f64) -> f64 {
    gamma_p(0.5 * df, 0.5 * x)
}

/// Chi-square upper tail `P(X > x)` (R: `pchisq(x, df, lower.tail = FALSE)`).
///
/// Accurate far into the tail, e.g. `chisq_sf(60, 2) = exp(-30)` to a few ulp.
pub fn chisq_sf(x: f64, df: f64) -> f64 {
    gamma_q(0.5 * df, 0.5 * x)
}

/// Chi-square density (R: `dchisq(x, df)`).
///
/// Returns 0 for `x < 0`; at `x = 0` the density is `+inf` for `df < 2`,
/// `0.5` for `df = 2` and 0 for `df > 2`, as in R. NaN for `df < 0` or NaN
/// input.
pub fn chisq_pdf(x: f64, df: f64) -> f64 {
    if x.is_nan() || df.is_nan() || df < 0.0 {
        return f64::NAN;
    }
    if x < 0.0 || x == f64::INFINITY {
        return 0.0;
    }
    let a = 0.5 * df;
    if a == 0.0 {
        // Point mass at zero (R: dgamma(x, shape = 0)).
        return if x == 0.0 { f64::INFINITY } else { 0.0 };
    }
    if x == 0.0 {
        return if a < 1.0 {
            f64::INFINITY
        } else if a > 1.0 {
            0.0
        } else {
            0.5
        };
    }
    if a == f64::INFINITY {
        return 0.0;
    }
    // dgamma(x; a, scale = 2) = (x/2)^a e^{-x/2} / Γ(a) / x.
    gamma_prefactor(a, 0.5 * x) / x
}

/// Chi-square quantile function: the `x` with `P(X <= x) = p`
/// (R: `qchisq(p, df)`).
///
/// Starts from R's `qchisq_appr` initial value (small-`x` power law,
/// Wilson–Hilferty, or its upper-tail correction) and refines it with a
/// bracketed Newton iteration that falls back to bisection whenever a step
/// leaves the current bracket. For `p <= 1/2` it solves `ln P(x) = ln p`
/// (Newton in `ln x`); for `p > 1/2` it solves `ln Q(x) = ln(1 - p)` (Newton
/// in `x`; `1 - p` is exact there), so the result keeps close to full
/// relative accuracy in both tails. Returns 0 for `p = 0` (and for `df = 0`),
/// `+inf` for `p = 1`, and NaN for `p` outside `[0, 1]`, `df < 0` or NaN
/// input.
pub fn chisq_quantile(p: f64, df: f64) -> f64 {
    if p.is_nan() || df.is_nan() || df < 0.0 || !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if p == 0.0 {
        return 0.0;
    }
    if p == 1.0 {
        return f64::INFINITY;
    }
    if df == 0.0 {
        return 0.0;
    }
    if df == f64::INFINITY {
        return f64::INFINITY;
    }
    let a = 0.5 * df;
    let lower = p <= 0.5;
    let target = if lower { p.ln() } else { (1.0 - p).ln() };

    let mut x = chisq_initial(p, df);
    if x == 0.0 {
        // The small-x power law underflows: the quantile is below the
        // smallest subnormal (R also returns 0 here).
        return 0.0;
    }
    let (mut lo, mut hi) = (0.0_f64, f64::INFINITY);
    for _ in 0..2000 {
        let (pl, pu) = gamma_pq(a, 0.5 * x);
        // r(x) is increasing in x and vanishes at the root.
        let r = if lower {
            pl.ln() - target
        } else {
            target - pu.ln()
        };
        if r == 0.0 {
            return x;
        }
        if r > 0.0 {
            hi = x;
        } else {
            lo = x;
        }
        let dens = chisq_pdf(x, df);
        let mut next = if lower {
            // d ln P / d ln x = x f(x) / P(x).
            x * (-r * pl / (x * dens)).exp()
        } else {
            // d (-ln Q) / dx = f(x) / Q(x).
            x - r * pu / dens
        };
        if !(next > lo && next < hi) {
            next = if hi.is_finite() {
                0.5 * (lo + hi)
            } else {
                2.0 * x
            };
        }
        if (next - x).abs() <= 2.0 * f64::EPSILON * x || next == lo || next == hi {
            return next;
        }
        x = next;
    }
    x
}

/// Initial approximation for [`chisq_quantile`] (the main branches of R's
/// `qchisq_appr`).
fn chisq_initial(p: f64, df: f64) -> f64 {
    let a = 0.5 * df;
    let ln_p = p.ln();
    let ch = if df < -1.24 * ln_p {
        // Small chi-square: P(a, x/2) ≈ (x/2)^a / Γ(a + 1).
        ((libm::lgamma(a + 1.0) + ln_p) / a + core::f64::consts::LN_2).exp()
    } else {
        let z = norm_quantile(p);
        let c = 2.0 / (9.0 * df);
        let wh = df * (z * c.sqrt() + 1.0 - c).powi(3);
        if wh > 2.2 * df + 6.0 {
            // Upper tail: Q(a, x/2) ≈ (x/2)^{a-1} e^{-x/2} / Γ(a).
            -2.0 * ((-p).ln_1p() - (a - 1.0) * (0.5 * wh).ln() + libm::lgamma(a))
        } else {
            wh
        }
    };
    if ch.is_finite() && ch >= 0.0 { ch } else { df }
}

#[cfg(test)]
// Reference values are R output printed with sprintf("%.17g").
#[allow(clippy::excessive_precision)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn rel(got: f64, want: f64) -> f64 {
        if got == want {
            0.0
        } else {
            ((got - want) / want).abs()
        }
    }

    /// R 4.6.1, printed with sprintf("%.17g"): x, df, pchisq(x, df),
    /// pchisq(x, df, lower.tail = FALSE), dchisq(x, df).
    #[rustfmt::skip]
    const PCHISQ: &[[f64; 5]] = &[
        [1e-08, 1.0, 7.9788455947305779e-05, 0.99992021154405275, 3989.4227840672129],
        [0.001, 1.0, 0.025227120630039609, 0.97477287936996038, 12.609356355490785],
        [0.1, 1.0, 0.24817036595415071, 0.75182963404584924, 1.2000389484301361],
        [0.5, 1.0, 0.52049987781304652, 0.47950012218695343, 0.43939128946772243],
        [1.0, 1.0, 0.68268949213708596, 0.31731050786291415, 0.24197072451914337],
        [2.0, 1.0, 0.842700792949716, 0.157299207050284, 0.10377687435514867],
        [3.5, 1.0, 0.93863117086059789, 0.061368829139402066, 0.037056184523748127],
        [5.0, 1.0, 0.9746526813225318, 0.025347318677468245, 0.014644982561926489],
        [10.0, 1.0, 0.9984345977419975, 0.001565402258002551, 0.00085003666025203412],
        [30.0, 1.0, 0.99999995679536946, 4.3204630578274975e-08, 2.2280873345249662e-08],
        [60.0, 1.0, 0.99999999999999056, 9.4857375710738494e-15, 4.8194777934968125e-15],
        [150.0, 1.0, 1.0, 1.7336432457178268e-34, 8.7252583201028872e-35],
        [400.0, 1.0, 1.0, 5.5072482372124689e-89, 2.7604741810798819e-89],
        [1e-08, 2.0, 4.9999999874999999e-09, 0.99999999500000003, 0.49999999750000002],
        [0.001, 2.0, 0.0004998750208307295, 0.99950012497916929, 0.49975006248958465],
        [0.1, 2.0, 0.048770575499285991, 0.95122942450071402, 0.47561471225035701],
        [0.5, 2.0, 0.22119921692859512, 0.77880078307140488, 0.38940039153570244],
        [1.0, 2.0, 0.39346934028736658, 0.60653065971263342, 0.30326532985631671],
        [2.0, 2.0, 0.63212055882855767, 0.36787944117144233, 0.18393972058572117],
        [3.5, 2.0, 0.82622605654955483, 0.17377394345044514, 0.08688697172522257],
        [5.0, 2.0, 0.91791500137610116, 0.0820849986238988, 0.0410424993119494],
        [10.0, 2.0, 0.99326205300091452, 0.006737946999085467, 0.0033689734995427335],
        [30.0, 2.0, 0.99999969409767953, 3.0590232050182579e-07, 1.5295116025091289e-07],
        [60.0, 2.0, 0.99999999999990641, 9.3576229688401748e-14, 4.6788114844200874e-14],
        [150.0, 2.0, 1.0, 2.6786369618080778e-33, 1.3393184809040389e-33],
        [400.0, 2.0, 1.0, 1.3838965267367376e-87, 6.9194826336836878e-88],
        [1e-08, 3.0, 2.6596151946973723e-13, 0.99999999999973399, 3.9894227840672132e-05],
        [0.001, 3.0, 8.4079190580461564e-06, 0.99999159208094202, 0.012609356355490785],
        [0.1, 3.0, 0.0081625762681235229, 0.99183742373187644, 0.12000389484301362],
        [0.5, 3.0, 0.081108588345324154, 0.91889141165467592, 0.21969564473386122],
        [1.0, 3.0, 0.1987480430987992, 0.80125195690120077, 0.24197072451914337],
        [2.0, 3.0, 0.42759329552912129, 0.57240670447087871, 0.20755374871029733],
        [3.5, 3.0, 0.67923787919436107, 0.32076212080563898, 0.12969664583311843],
        [5.0, 3.0, 0.8282028557032669, 0.17179714429673312, 0.073224912809632448],
        [10.0, 3.0, 0.9814338645369568, 0.018566135463043233, 0.0085003666025203414],
        [30.0, 3.0, 0.99999861994296868, 1.3800570312932547e-06, 6.6842620035748982e-07],
        [60.0, 3.0, 0.99999999999941214, 5.8782307279069133e-13, 2.8916866760980873e-13],
        [150.0, 3.0, 1.0, 2.6349139284880449e-32, 1.3087887480154331e-32],
        [400.0, 3.0, 1.0, 2.213886593101118e-86, 1.1041896724319527e-86],
        [1e-08, 4.0, 1.2499999958333336e-17, 1.0, 2.4999999875000003e-09],
        [0.001, 4.0, 1.2495834114479175e-07, 0.99999987504165888, 0.00024987503124479239],
        [0.1, 4.0, 0.0012091042742502906, 0.99879089572574964, 0.023780735612517853],
        [0.5, 4.0, 0.026499021160743916, 0.97350097883925601, 0.097350097883925624],
        [1.0, 4.0, 0.090204010431049877, 0.90979598956895014, 0.15163266492815838],
        [2.0, 4.0, 0.26424111765711528, 0.73575888234288478, 0.18393972058572117],
        [3.5, 4.0, 0.52212165551127598, 0.47787834448872407, 0.15205220051913948],
        [5.0, 4.0, 0.71270250481635422, 0.28729749518364578, 0.1026062482798735],
        [10.0, 4.0, 0.95957231800548715, 0.040427681994512805, 0.016844867497713668],
        [30.0, 4.0, 0.99999510556287197, 4.8944371280292126e-06, 2.2942674037636936e-06],
        [60.0, 4.0, 0.9999999999970991, 2.9008631203404549e-12, 1.4036434453260265e-12],
        [150.0, 4.0, 1.0, 2.0357640909741394e-31, 1.0044888606780292e-31],
        [400.0, 4.0, 1.0, 2.7816320187408423e-85, 1.3838965267367378e-85],
        [1e-08, 9.5, 4.7170104682685699e-42, 1.0, 2.2405799704792401e-33],
        [0.001, 9.5, 2.6514745352035595e-18, 1.0, 1.2593408882061494e-14],
        [0.1, 9.5, 8.0489675852610554e-09, 0.99999999195103251, 3.7900568765809428e-07],
        [0.5, 9.5, 1.4268425672835003e-05, 0.99998573157432713, 0.0001296953326098491],
        [1.0, 9.5, 0.00031294840277249283, 0.99968705159722737, 0.0013589804544285886],
        [2.0, 9.5, 0.0056193228102785738, 0.99438067718972145, 0.011089921022670776],
        [3.5, 9.5, 0.044235727036823762, 0.95576427296317623, 0.042716953743380089],
        [5.0, 9.5, 0.134930529229361, 0.865069470770639, 0.076870864101564157],
        [10.0, 9.5, 0.60478425037533057, 0.39521574962466949, 0.08489615905441826],
        [30.0, 9.5, 0.99938254780643543, 0.0006174521935645961, 0.00023721815736777374],
        [60.0, 9.5, 0.99999999777956339, 2.2204366166259833e-09, 9.7632224211915509e-10],
        [150.0, 9.5, 1.0, 1.8264647459598649e-27, 8.6819350031566656e-28],
        [400.0, 9.5, 1.0, 3.6174081495496229e-80, 1.7749619124865659e-80],
    ];

    #[test]
    fn pchisq_and_dchisq_match_r() {
        let mut worst = 0.0_f64;
        for &[x, df, lower, upper, dens] in PCHISQ {
            for (got, want) in [(chisq_cdf(x, df), lower), (chisq_sf(x, df), upper)] {
                // Relative where the tail carries information; the larger
                // tail is compared absolutely. R itself is ~1e-15 off in a
                // few such cells, e.g. pchisq(2, 1) = 0.842700792949716
                // while erf(1) = 0.84270079294971487 (we return 0.842700792949715).
                if want > 0.5 {
                    assert!(
                        (got - want).abs() <= 2e-15,
                        "x={x} df={df}: {got} vs {want}"
                    );
                } else {
                    worst = worst.max(rel(got, want));
                }
            }
            worst = worst.max(rel(chisq_pdf(x, df), dens));
        }
        assert!(worst < 1e-14, "max relative deviation from R: {worst:e}");
    }

    #[test]
    fn gamma_pq_match_r_including_large_shape() {
        // R: a, x, pgamma(x, a), pgamma(x, a, lower.tail = FALSE).
        let cases = [
            (0.5, 0.2, 0.47291074313446191, 0.52708925686553809),
            (0.5, 1.2, 0.87866474964151753, 0.12133525035848249),
            (12.5, 3.0, 3.4267686170276234e-05, 0.99996573231382968),
            (50.0, 40.0, 0.070335066659394957, 0.92966493334060507),
            (50.0, 65.0, 0.97648760219019137, 0.023512397809808679),
            (500.0, 450.0, 0.010717238091289742, 0.98928276190871023),
            (500.0, 600.0, 0.99998774405766933, 1.2255942330622908e-05),
            (10000.0, 10000.0, 0.50132980833995511, 0.49867019166004478),
        ];
        for (a, x, p, q) in cases {
            assert_relative_eq!(gamma_p(a, x), p, max_relative = 1e-14);
            assert_relative_eq!(gamma_q(a, x), q, max_relative = 1e-14);
        }
    }

    #[test]
    fn upper_tail_keeps_relative_accuracy() {
        // Q(1, x) = e^{-x} exactly.
        assert_relative_eq!(chisq_sf(60.0, 2.0), (-30.0_f64).exp(), max_relative = 1e-15);
        assert_relative_eq!(
            chisq_sf(1400.0, 2.0),
            (-700.0_f64).exp(),
            max_relative = 1e-14
        );
        assert_relative_eq!(gamma_q(1.0, 30.0), (-30.0_f64).exp(), max_relative = 1e-15);
        assert_relative_eq!(
            gamma_p(1.0, 1e-5),
            -(-1e-5_f64).exp_m1(),
            max_relative = 1e-15
        );
    }

    #[test]
    fn qchisq_matches_r() {
        let ps = [
            1e-12, 1e-6, 0.01, 0.05, 0.3, 0.5, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99, 0.999999,
        ];
        // R: qchisq(ps, df).
        #[rustfmt::skip]
        let want: [(f64, [f64; 14]); 5] = [
            (1.0, [1.5707963267948977e-24, 1.5707963267957164e-12, 0.00015708785790970206, 0.0039321400000195249, 0.14847186183254543, 0.45493642311957283, 1.0741941708575848, 1.3233036969314669, 1.6423744151498156, 2.0722508558222392, 2.7055434540954155, 3.8414588206941263, 6.6348966010212136, 23.928126976879469]),
            (2.0, [2.000000000001e-12, 2.0000010000006668e-06, 0.020100671707002884, 0.10258658877510107, 0.71334988787746478, 1.3862943611198906, 2.4079456086518718, 2.7725887222397811, 3.2188758248682015, 3.794239969771763, 4.6051701859880918, 5.9914645471079799, 9.2103403719761818, 27.631021115871036]),
            (3.0, [2.4179879427180364e-08, 0.0002418104872012428, 0.11483180189911704, 0.35184631774927144, 1.4236522430352796, 2.3659738843753373, 3.6648707831703167, 4.1083449356323172, 4.6416276760874453, 5.3170478373170971, 6.2513886311703226, 7.8147279032511792, 11.344866730144368, 30.664849706154268]),
            (4.0, [2.8284284580803877e-06, 0.0028297613229586859, 0.29710948050653185, 0.71072302139732402, 2.1946984214069833, 3.356693980033322, 4.8784329665604087, 5.385269057779392, 5.9886166940042447, 6.744883087212421, 7.7794403397348599, 9.487729036781154, 13.276704135987622, 33.37684158165888]),
            (9.5, [0.014946147309931202, 0.28034357211690542, 2.3197113186657217, 3.6300995805284595, 6.829348005851581, 8.8422967295258026, 11.219412907372782, 11.969925737444106, 12.843451163271148, 13.912710771510568, 15.337540435587087, 17.615706915475762, 22.441343187624948, 45.843654552918409]),
        ];
        let mut worst = 0.0_f64;
        for (df, row) in want {
            for (&p, &q) in ps.iter().zip(row.iter()) {
                worst = worst.max(rel(chisq_quantile(p, df), q));
            }
        }
        assert!(worst < 1e-12, "max relative deviation from R: {worst:e}");
    }

    #[test]
    fn qchisq_extreme_tails_solve_the_tail_equation() {
        // For p this close to 1, R's own qchisq(p) drifts by up to ~1e-7;
        // chisq_quantile instead solves Q(x) = 1 - p to full precision.
        for df in [1.0, 2.0, 3.0, 4.0, 9.5] {
            for p in [1.0 - 1e-13, 1.0 - 1e-15] {
                let x = chisq_quantile(p, df);
                assert_relative_eq!(chisq_sf(x, df), 1.0 - p, max_relative = 1e-13);
            }
            for p in [1e-300, 1e-100, 1e-20] {
                let x = chisq_quantile(p, df);
                assert_relative_eq!(chisq_cdf(x, df), p, max_relative = 1e-13);
            }
            for p in [1e-9, 0.2, 0.5, 0.9, 0.999] {
                let x = chisq_quantile(p, df);
                assert_relative_eq!(chisq_cdf(x, df), p, max_relative = 1e-13);
            }
        }
        // The root underflows (R: qchisq(1e-300, 1) = 0).
        assert_eq!(chisq_quantile(1e-300, 1.0), 0.0);
    }

    #[test]
    fn edge_cases_follow_r() {
        assert_eq!(chisq_cdf(0.0, 2.0), 0.0);
        assert_eq!(chisq_cdf(-1.0, 2.0), 0.0);
        assert_eq!(chisq_sf(-1.0, 2.0), 1.0);
        assert_eq!(chisq_cdf(f64::INFINITY, 3.0), 1.0);
        assert_eq!(chisq_sf(f64::INFINITY, 3.0), 0.0);
        assert_eq!(chisq_pdf(0.0, 1.0), f64::INFINITY);
        assert_eq!(chisq_pdf(0.0, 2.0), 0.5);
        assert_eq!(chisq_pdf(0.0, 3.0), 0.0);
        assert_eq!(chisq_pdf(-1.0, 3.0), 0.0);
        assert_eq!(chisq_quantile(0.0, 3.0), 0.0);
        assert_eq!(chisq_quantile(1.0, 3.0), f64::INFINITY);
        assert!(chisq_quantile(1.5, 3.0).is_nan());
        assert!(chisq_quantile(-0.1, 3.0).is_nan());
        assert!(chisq_quantile(f64::NAN, 3.0).is_nan());
        assert!(chisq_cdf(1.0, -1.0).is_nan());
        assert!(chisq_pdf(1.0, f64::NAN).is_nan());
        assert!(gamma_p(f64::NAN, 1.0).is_nan());
        assert!(gamma_q(1.0, f64::NAN).is_nan());
    }
}
