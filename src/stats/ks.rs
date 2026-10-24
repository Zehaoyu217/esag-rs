//! Kolmogorov–Smirnov goodness-of-fit tests, reproducing R's `ks.test`
//! (R >= 4.1, two-sided alternative, default `exact = NULL`).
//!
//! * One sample: exact p-value from the Marsaglia–Tsang–Wang (2003)
//!   algorithm (R's `C_pkolmogorov_two_exact`, formerly `pkolmogorov2x`)
//!   when `n < 100` and there are no ties; otherwise the limiting Kolmogorov
//!   distribution `1 - K(√n·D)` evaluated exactly as R's
//!   `C_pkolmogorov_two_limit` does (including its `tol = 1e-6` truncation).
//! * Two samples: exact p-value from R's `psmirnov` recursion (with R's
//!   handling of ties) when `m·n < 10000`; otherwise `1 - K(√(mn/(m+n))·D)`.

use core::f64::consts::{FRAC_PI_2, FRAC_PI_4};

use super::chisq::chisq_cdf;
use super::normal::INV_SQRT_2PI;

/// Tolerance R passes to `C_pkolmogorov_two_limit` from `ks.test`.
const LIMIT_TOL: f64 = 1e-6;

/// Result of a two-sided Kolmogorov–Smirnov test (the `statistic`,
/// `p.value` and `exact` components of R's `ks.test` object).
#[derive(Clone, Debug)]
pub struct KsResult {
    /// The statistic `D = sup |F̂ - F|` (one sample) or `sup |F̂_x - F̂_y|`
    /// (two samples).
    pub statistic: f64,
    /// Two-sided p-value, clamped to `[0, 1]` as in R.
    pub p_value: f64,
    /// Number of non-NaN observations used (`m + n` for two samples).
    pub n: usize,
    /// Whether the exact null distribution was used (R's `exact` component);
    /// `false` means the asymptotic Kolmogorov distribution.
    pub exact: bool,
}

impl KsResult {
    fn undefined(n: usize) -> Self {
        KsResult {
            statistic: f64::NAN,
            p_value: f64::NAN,
            n,
            exact: false,
        }
    }
}

/// One-sample two-sided Kolmogorov–Smirnov test of `x` against the
/// continuous distribution function `cdf` (R: `ks.test(x, "pfoo", ...)`).
///
/// NaN observations are dropped (R drops NA). The statistic is R's
/// `max(c(F(x₍ᵢ₎) - (i-1)/n, 1/n - (F(x₍ᵢ₎) - (i-1)/n)))`. The p-value is exact
/// when `n < 100` and `x` has no ties, asymptotic otherwise; R warns about
/// ties in this case, here they only switch the method. An empty sample
/// gives NaN statistic and p-value; a NaN from `cdf` gives a NaN statistic
/// and p-value (R propagates NA).
pub fn ks_1samp(x: &[f64], cdf: impl Fn(f64) -> f64) -> KsResult {
    let mut xs: Vec<f64> = x.iter().copied().filter(|v| !v.is_nan()).collect();
    let n = xs.len();
    if n == 0 {
        return KsResult::undefined(0);
    }
    xs.sort_unstable_by(|a, b| a.total_cmp(b));
    let ties = xs.windows(2).any(|w| w[0] == w[1]);
    let exact = n < 100 && !ties;

    let nf = n as f64;
    let inv_n = 1.0 / nf;
    let mut stat = f64::NEG_INFINITY;
    for (i, &v) in xs.iter().enumerate() {
        let t = cdf(v) - i as f64 / nf;
        if t.is_nan() {
            return KsResult {
                statistic: f64::NAN,
                p_value: f64::NAN,
                n,
                exact,
            };
        }
        stat = stat.max(t).max(inv_n - t);
    }

    // R: pkolmogorov(STATISTIC, n, two.sided = TRUE, exact, lower.tail = FALSE)
    let p = if stat <= 0.0 {
        1.0
    } else if stat > 1.0 {
        0.0
    } else if exact {
        1.0 - pkolmogorov_two_exact(n, stat)
    } else {
        pkolmogorov_two_limit_upper(nf.sqrt() * stat, LIMIT_TOL)
    };
    KsResult {
        statistic: stat,
        p_value: p.clamp(0.0, 1.0),
        n,
        exact,
    }
}

/// One-sample KS test of `x` against the χ² distribution with `df` degrees
/// of freedom (R: `ks.test(x, "pchisq", df = df)`).
pub fn ks_1samp_chisq(x: &[f64], df: f64) -> KsResult {
    ks_1samp(x, |q| chisq_cdf(q, df))
}

/// Two-sample two-sided Kolmogorov–Smirnov test (R: `ks.test(x, y)`).
///
/// NaN observations are dropped. The statistic is computed as R does it:
/// the pooled sample is ordered stably, `±1/m`, `∓1/n` steps are summed
/// cumulatively, and with ties only the positions at the end of each group of
/// equal values are used. As in R 4.x, the exact p-value (`psmirnov`, which
/// accounts for ties by checking the boundary only between distinct pooled
/// values) is used when `m·n < 10000`, with or without ties; otherwise the
/// asymptotic Kolmogorov distribution is used. `n` in the result is `m + n`.
/// Either sample empty gives NaN statistic and p-value.
pub fn ks_2samp(x: &[f64], y: &[f64]) -> KsResult {
    let xs: Vec<f64> = x.iter().copied().filter(|v| !v.is_nan()).collect();
    let ys: Vec<f64> = y.iter().copied().filter(|v| !v.is_nan()).collect();
    let (nx, ny) = (xs.len(), ys.len());
    if nx == 0 || ny == 0 {
        return KsResult::undefined(nx + ny);
    }
    let (mx, my) = (nx as f64, ny as f64);
    let exact = mx * my < 10000.0;

    // R: w <- c(x, y); z <- cumsum(ifelse(order(w) <= n.x, 1/n.x, -1/n.y)).
    // `sort_by` is stable, like R's order(); NaNs were removed above.
    let mut pooled: Vec<(f64, bool)> = xs
        .iter()
        .map(|&v| (v, true))
        .chain(ys.iter().map(|&v| (v, false)))
        .collect();
    pooled.sort_by(|a, b| a.0.partial_cmp(&b.0).expect("NaN removed"));
    let (step_x, step_y) = (1.0 / mx, -1.0 / my);
    let total = nx + ny;
    let mut ties = false;
    let mut cum = 0.0;
    let mut stat = f64::NEG_INFINITY;
    for k in 0..total {
        cum += if pooled[k].1 { step_x } else { step_y };
        // With ties, R keeps z only where diff(sort(w)) != 0, and the last.
        if k + 1 == total || pooled[k].0 != pooled[k + 1].0 {
            stat = stat.max(cum.abs());
        } else {
            ties = true;
        }
    }

    // R: psmirnov(STATISTIC, sizes, z = if (TIES) w, exact, lower.tail = FALSE)
    let p = if stat <= 0.0 {
        1.0
    } else if stat > 1.0 {
        0.0
    } else if exact {
        let breaks: Option<Vec<bool>> = if ties {
            // z <- diff(sort(w)) != 0; z <- if (any(z)) c(0L, z, 1L)
            let mut z = Vec::with_capacity(total + 1);
            z.push(false);
            z.extend(pooled.windows(2).map(|w| w[1].0 != w[0].0));
            z.push(true);
            z[1..total].iter().any(|&b| b).then_some(z)
        } else {
            None
        };
        psmirnov_exact_upper(stat, nx, ny, breaks.as_deref())
    } else {
        let n_eff = mx * my / (mx + my);
        pkolmogorov_two_limit_upper(n_eff.sqrt() * stat, LIMIT_TOL)
    };
    KsResult {
        statistic: stat,
        p_value: p.clamp(0.0, 1.0),
        n: total,
        exact,
    }
}

/// `x^n` by repeated squaring, operation for operation as R's `R_pow_di`.
fn pow_di(mut x: f64, mut n: i32) -> f64 {
    let mut pow = 1.0;
    if n != 0 {
        if n < 0 {
            n = -n;
            x = 1.0 / x;
        }
        loop {
            if n & 1 == 1 {
                pow *= x;
            }
            n >>= 1;
            if n == 0 {
                break;
            }
            x *= x;
        }
    }
    pow
}

/// Upper tail `1 - K(x)` of the limiting Kolmogorov distribution
/// `K(x) = 1 - 2 Σ_{k≥1} (-1)^{k-1} e^{-2k²x²}`, computed as R's
/// `C_pkolmogorov_two_limit(x, lower.tail = FALSE, tol)`: for `x < 1` from
/// the theta-function series `√(2π)/x Σ_{k odd} e^{-k²π²/(8x²)}` truncated at
/// `k < ⌊√(2 - ln tol)⌋` (a single term for `tol = 1e-6`), for `x >= 1` from
/// the alternating series summed until a term is `<= tol`.
fn pkolmogorov_two_limit_upper(x: f64, tol: f64) -> f64 {
    if x <= 0.0 {
        return 1.0;
    }
    let k_max = (2.0 - tol.ln()).sqrt() as i32;
    if x < 1.0 {
        let z = -(FRAC_PI_2 * FRAC_PI_4) / (x * x);
        let w = x.ln();
        let mut s = 0.0;
        let mut k = 1;
        while k < k_max {
            s += (f64::from(k * k) * z - w).exp();
            k += 2;
        }
        1.0 - s / INV_SQRT_2PI
    } else {
        let z = -2.0 * x * x;
        let mut sign = 1.0;
        let mut k = 1.0;
        let mut new = 0.0;
        loop {
            let old = new;
            new += 2.0 * sign * (z * k * k).exp();
            sign = -sign;
            k += 1.0;
            if (old - new).abs() <= tol {
                break;
            }
        }
        new
    }
}

/// Exact `P(D_n < d)` for the one-sample two-sided statistic (Marsaglia,
/// Tsang & Wang 2003, "Evaluating Kolmogorov's distribution", JSS 8(18)),
/// transcribed from R's `C_pkolmogorov_two_exact` including its `1e140`
/// exponent bookkeeping.
fn pkolmogorov_two_exact(n: usize, d: f64) -> f64 {
    let nf = n as f64;
    let k = (nf * d) as usize + 1;
    let m = 2 * k - 1;
    let h = k as f64 - nf * d;
    let mut hm = vec![0.0; m * m];
    for i in 0..m {
        for j in 0..m {
            hm[i * m + j] = if i + 1 < j { 0.0 } else { 1.0 };
        }
    }
    for i in 0..m {
        hm[i * m] -= pow_di(h, (i + 1) as i32);
        hm[(m - 1) * m + i] -= pow_di(h, (m - i) as i32);
    }
    if 2.0 * h - 1.0 > 0.0 {
        hm[(m - 1) * m] += pow_di(2.0 * h - 1.0, m as i32);
    }
    for i in 0..m {
        for j in 0..m {
            if i + 1 > j {
                for g in 1..=(i + 1 - j) {
                    hm[i * m + j] /= g as f64;
                }
            }
        }
    }
    let (q, mut eq) = m_power(&hm, 0, m, n);
    let mut s = q[(k - 1) * m + k - 1];
    for i in 1..=n {
        s = s * i as f64 / nf;
        if s < 1e-140 {
            s *= 1e140;
            eq -= 140;
        }
    }
    s * pow_di(10.0, eq)
}

/// `C = A B` for row-major `m × m` matrices (R's `m_multiply`).
fn m_multiply(a: &[f64], b: &[f64], c: &mut [f64], m: usize) {
    for i in 0..m {
        for j in 0..m {
            let mut s = 0.0;
            for k in 0..m {
                s += a[i * m + k] * b[k * m + j];
            }
            c[i * m + j] = s;
        }
    }
}

/// `A^n` with a decimal exponent carried separately (R's `m_power`):
/// returns `(V, eV)` with `A^n = V · 10^eV`.
fn m_power(a: &[f64], ea: i32, m: usize, n: usize) -> (Vec<f64>, i32) {
    if n == 1 {
        return (a.to_vec(), ea);
    }
    let (v_half, ev_half) = m_power(a, ea, m, n / 2);
    let mut b = vec![0.0; m * m];
    m_multiply(&v_half, &v_half, &mut b, m);
    let eb = 2 * ev_half;
    let (mut v, mut ev) = if n.is_multiple_of(2) {
        (b, eb)
    } else {
        let mut v = vec![0.0; m * m];
        m_multiply(a, &b, &mut v, m);
        (v, ea + eb)
    };
    if v[(m / 2) * m + (m / 2)] > 1e140 {
        for e in v.iter_mut() {
            *e *= 1e-140;
        }
        ev += 140;
    }
    (v, ev)
}

/// Exact upper tail `P(D >= d)` of the two-sample two-sided Smirnov
/// statistic (R's `C_psmirnov_exact(..., two.sided = TRUE, lower.tail =
/// FALSE)`). `u[j]` is the probability that a random lattice path from
/// `(0, 0)` to `(i, j)` has touched the boundary `|i/m - j/n| >= q`; with
/// ties (`breaks = Some(z)`) the boundary only counts at pooled positions
/// `i + j` where `z[i + j]` marks the end of a group of equal values.
fn psmirnov_exact_upper(stat: f64, m: usize, n: usize, breaks: Option<&[bool]>) -> f64 {
    let (m, n) = if m > n { (n, m) } else { (m, n) };
    let (md, nd) = (m as f64, n as f64);
    // R: q has 0.5/mn added so rounding cannot turn an equality into an
    // inequality, e.g. abs(1/2 - 4/5) > 3/10.
    let q = (0.5 + (stat * md * nd - 1e-7).floor()) / (md * nd);
    let hit = |i: usize, j: usize| {
        (i as f64 / md - j as f64 / nd).abs() >= q && breaks.is_none_or(|z| z[i + j])
    };
    let mut u = vec![0.0; n + 1];
    for j in 1..=n {
        u[j] = if hit(0, j) { 1.0 } else { u[j - 1] };
    }
    for i in 1..=m {
        if hit(i, 0) {
            u[0] = 1.0;
        }
        for j in 1..=n {
            if hit(i, j) {
                u[j] = 1.0;
            } else {
                let v = i as f64 / (i + j) as f64;
                let w = j as f64 / (i + j) as f64;
                u[j] = v * u[j] + w * u[j - 1];
            }
        }
    }
    u[n]
}

#[cfg(test)]
// Reference values are R output printed with sprintf("%.17g").
#[allow(clippy::excessive_precision)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use serde_json::Value;

    const FIXTURES: &str = include_str!("../../tests/fixtures/ks_fixtures.json");

    fn floats(v: &Value) -> Vec<f64> {
        v.as_array()
            .expect("array")
            .iter()
            .map(|x| x.as_f64().expect("number"))
            .collect()
    }

    fn num(case: &Value, key: &str) -> f64 {
        case[key].as_f64().expect(key)
    }

    /// p-value tolerance from the brief: 1e-8 absolute for exact p-values,
    /// 1e-6 relative for asymptotic ones.
    fn assert_p(got: &KsResult, want: f64, name: &str) {
        if got.exact {
            assert!(
                (got.p_value - want).abs() <= 1e-8,
                "{name}: p {} vs {want}",
                got.p_value
            );
        } else {
            assert_relative_eq!(got.p_value, want, max_relative = 1e-6);
        }
    }

    fn uniform_cdf(q: f64) -> f64 {
        q.clamp(0.0, 1.0)
    }

    #[test]
    fn one_sample_matches_r_fixtures() {
        let fx: Value = serde_json::from_str(FIXTURES).expect("valid fixture JSON");
        let cases = fx["one_sample"].as_array().expect("one_sample");
        assert!(!cases.is_empty());
        for case in cases {
            let name = case["name"].as_str().expect("name");
            let x = floats(&case["x"]);
            let r = ks_1samp_chisq(&x, num(case, "df_test"));
            assert_eq!(r.n, x.len(), "{name}");
            assert_eq!(r.exact, case["exact"].as_bool().expect("exact"), "{name}");
            assert_relative_eq!(r.statistic, num(case, "statistic"), max_relative = 1e-12);
            assert_p(&r, num(case, "p_value"), name);
            // Observed agreement is far tighter than the brief's tolerance.
            assert_relative_eq!(r.p_value, num(case, "p_value"), max_relative = 1e-12);
        }
    }

    #[test]
    fn two_sample_matches_r_fixtures() {
        let fx: Value = serde_json::from_str(FIXTURES).expect("valid fixture JSON");
        let cases = fx["two_sample"].as_array().expect("two_sample");
        assert!(!cases.is_empty());
        for case in cases {
            let name = case["name"].as_str().expect("name");
            let (x, y) = (floats(&case["x"]), floats(&case["y"]));
            let r = ks_2samp(&x, &y);
            assert_eq!(r.n, x.len() + y.len(), "{name}");
            assert_eq!(r.exact, case["exact"].as_bool().expect("exact"), "{name}");
            assert_relative_eq!(r.statistic, num(case, "statistic"), max_relative = 1e-12);
            assert_p(&r, num(case, "p_value"), name);
            assert_relative_eq!(r.p_value, num(case, "p_value"), max_relative = 1e-12);
        }
    }

    #[test]
    fn small_and_boundary_cases_match_r() {
        // R 4.6.1, suppressWarnings(ks.test(...)): (D, p.value, exact).
        let u100: Vec<f64> = (1..=100_u32)
            .map(|i| f64::from((i * 37) % 101) / 101.0)
            .collect();
        let sq: Vec<f64> = u100.iter().map(|v| v * v).collect();
        let one = [
            (vec![0.5], 0.5, 1.0, true),
            (
                vec![0.1, 0.4, 0.7],
                0.29999999999999999,
                0.88622222222222224,
                true,
            ),
            // Ties switch R to the asymptotic distribution.
            (
                vec![0.2, 0.2, 0.9],
                0.46666666666666662,
                0.53071486318854655,
                false,
            ),
            (u100.clone(), 0.0099009900990099011, 1.0, false),
            (u100[..99].to_vec(), 0.012601260126012639, 1.0, true),
            (sq, 0.25502499754925989, 4.4866569215284347e-06, false),
        ];
        for (x, d, p, exact) in one {
            let r = ks_1samp(&x, uniform_cdf);
            assert_eq!(r.exact, exact, "n = {}", x.len());
            assert_relative_eq!(r.statistic, d, max_relative = 1e-14);
            assert_relative_eq!(r.p_value, p, max_relative = 1e-12);
        }

        let ten: Vec<f64> = (1..=10).map(f64::from).collect();
        let six: Vec<f64> = (1..=5).map(|i| f64::from(i) + 0.5).collect();
        let two = [
            // Complete separation, m > n.
            (six, vec![0.1, 0.2], 1.0, 0.095238095238095233),
            (
                ten.clone(),
                vec![2.5, 3.5, 11.0, 12.0],
                0.5,
                0.4095904095904096,
            ),
            (
                vec![2.5, 3.5, 11.0, 12.0],
                ten.clone(),
                0.5,
                0.4095904095904096,
            ),
            (vec![1.0, 1.0, 1.0], vec![1.0, 1.0], 0.0, 1.0),
            (
                ten,
                vec![2.0, 2.0, 3.0, 11.0],
                0.44999999999999996,
                0.4425574425574425,
            ),
            // NaN (R: NA) observations are dropped.
            (
                vec![1.0, f64::NAN, 3.0, 5.0],
                vec![2.0, 4.0, f64::NAN],
                0.33333333333333337,
                1.0,
            ),
        ];
        for (x, y, d, p) in two {
            let r = ks_2samp(&x, &y);
            assert!(r.exact);
            assert_relative_eq!(r.statistic, d, max_relative = 1e-14);
            assert_relative_eq!(r.p_value, p, max_relative = 1e-12);
        }
    }

    #[test]
    fn kolmogorov_limit_follows_r_truncation() {
        // R: .Call(stats:::C_pkolmogorov_two_limit, x, FALSE, tol = 1e-6).
        let cases = [
            (0.3, 0.99999069419866549),
            (0.8, 0.54414250308288747),
            (0.99, 0.28090427906966486),
            (1.0, 0.26999967167737987),
            (1.5, 0.022217962616525123),
            (2.0, 0.00067092525577969533),
            (5.0, 3.8574996959278356e-22),
            (10.0, 2.7677930534734751e-87),
        ];
        for (x, want) in cases {
            assert_relative_eq!(
                pkolmogorov_two_limit_upper(x, LIMIT_TOL),
                want,
                max_relative = 1e-15
            );
        }
        assert_eq!(pkolmogorov_two_limit_upper(0.0, LIMIT_TOL), 1.0);
    }

    #[test]
    fn degenerate_inputs() {
        let r = ks_1samp(&[], uniform_cdf);
        assert!(r.statistic.is_nan() && r.p_value.is_nan() && r.n == 0);
        let r = ks_1samp(&[f64::NAN, 0.5], uniform_cdf);
        assert_eq!(r.n, 1);
        let r = ks_1samp(&[0.5, 0.7], |_| f64::NAN);
        assert!(r.statistic.is_nan() && r.p_value.is_nan());
        let r = ks_2samp(&[1.0], &[]);
        assert!(r.statistic.is_nan() && r.p_value.is_nan() && r.n == 1);
    }
}
