//! The ESAG log-density and its exact gradient (R: `log_likelihood`).
//!
//! For a unit vector `y` and `eta = (mu, gamma)`, with `p = d-1`,
//! ```text
//! q     = yᵀ V⁻¹ y,     alpha = (y·mu)/sqrt(q)
//! log f = −(p/2) log 2π − (d/2) log q + ½(alpha² − |mu|²) + log M_p(alpha)
//! M_1(a) = aΦ(a) + φ(a),  M_2(a) = (1+a²)Φ(a) + aφ(a),
//! M_k(a) = a M_{k−1}(a) + (k−1) M_{k−2}(a).
//! ```
//! `M_p(a) = ∫_0^∞ t^p φ(t − a) dt`. For negative `a` the recursion cancels
//! catastrophically (R's code returns `NaN`/`−Inf` far in the tail). For
//! `a < −2` we instead write `M_p(a) = φ(a) I_p(b)`, `b = −a`,
//! `I_k = ∫_0^∞ t^k e^{−bt − t²/2} dt`, and use the ratios `r_k = I_k/I_{k−1}`,
//! which satisfy the all-positive continued fraction
//! `r_k = k/(b + r_{k+1})`, `I_0 = 1/(b + r_1)`. Evaluated backwards from a
//! deep starting index it has no cancellation, so
//! `log M_p = log φ(a) + log I_0 + Σ_{k≤p} log r_k` is accurate to ~1e-14
//! for every `a` (checked against numerical quadrature).
//!
//! Everything is generic over [`Scalar`], so the gradient with respect to
//! `eta` comes from one forward pass with [`Dual<N>`](crate::dual::Dual).

use crate::dims;
use crate::dual::{Dual, seed_array};
use crate::frame::Frame;
use crate::scalar::Scalar;
use crate::stats::normal::{LN_2PI, norm_cdf, norm_pdf};

/// `x^{1/(d−1)}`, with square and cube roots instead of `powf` where the
/// exponent allows (d = 3, 4, 5, 7, 9). Shared by the value and gradient
/// kernels so that both compute bit-identical values.
#[inline]
pub(crate) fn root_d_minus_1<S: Scalar>(x: S, d: usize) -> S {
    match d - 1 {
        2 => x.sqrt(),
        3 => x.cbrt(),
        4 => x.sqrt().sqrt(),
        6 => x.cbrt().sqrt(),
        8 => x.sqrt().sqrt().sqrt(),
        k => x.powf(1.0 / k as f64),
    }
}

/// `Φ(a)`, exactly 1 in double precision above 8.5 (the upper tail is
/// below 1e−17 there), which saves the `erfc` call for concentrated data.
#[inline]
pub(crate) fn cdf_fast(a: f64) -> f64 {
    if a > 8.5 { 1.0 } else { norm_cdf(a) }
}

/// Below this `alpha` the continued fraction is used.
const CF_BELOW: f64 = -2.0;

/// Continued-fraction depth that reaches full double precision for
/// `b = −alpha ≥ 2` (measured: ≤ 112 terms at b = 2, p = 10; ≤ 32 at b = 8).
#[inline]
fn cf_depth(b: f64, p: usize) -> usize {
    ((170.0 / b).ceil() as usize + 2 * p + 8).min(200)
}

/// `(log M_p(a), d/da log M_p(a), d²/da² log M_p(a))` in `f64`, stable for all `a`.
///
/// Uses `M_k' = k M_{k−1}` (with `M_0 = Φ`), so
/// `(log M_p)' = p M_{p−1}/M_p` and
/// `(log M_p)'' = p(p−1) M_{p−2}/M_p − ((log M_p)')²` (for `p = 1`:
/// `φ/M_1 − (Φ/M_1)²`).
pub fn log_m_p_derivs(a: f64, p: usize) -> (f64, f64, f64) {
    assert!(p >= 1, "log_m_p needs p >= 1");
    if a >= CF_BELOW {
        let cdf = cdf_fast(a);
        let pdf = norm_pdf(a);
        let m1 = a * cdf + pdf;
        if p == 1 {
            let d1 = cdf / m1;
            return (m1.ln(), d1, pdf / m1 - d1 * d1);
        }
        // m_{k-2}, m_{k-1}, m_k with m_0 = Φ
        let (mut m_km2, mut m_km1, mut m_k) = (cdf, m1, (a * a + 1.0) * cdf + a * pdf);
        for k in 3..=p {
            let next = a * m_k + (k - 1) as f64 * m_km1;
            m_km2 = m_km1;
            m_km1 = m_k;
            m_k = next;
        }
        let r1 = m_km1 / m_k;
        let r2 = m_km2 / m_k;
        let d1 = p as f64 * r1;
        (m_k.ln(), d1, (p * (p - 1)) as f64 * r2 - d1 * d1)
    } else {
        // M_k = φ(a) I_k(b), b = −a; r_k = I_k/I_{k−1} = k/(b + r_{k+1}), I_0 = 1/(b + r_1)
        let b = -a;
        let depth = cf_depth(b, p);
        let mut r = ((depth as f64).sqrt() - 0.5 * b).max(1e-3);
        let mut log_sum = 0.0;
        let (mut r_p, mut r_pm1) = (1.0, 1.0);
        for k in (1..depth).rev() {
            r = k as f64 / (b + r);
            if k <= p {
                log_sum += r.ln();
                if k == p {
                    r_p = r;
                }
                if k + 1 == p {
                    r_pm1 = r;
                }
            }
        }
        let i0 = 1.0 / (b + r);
        let log_m = -0.5 * a * a - 0.5 * LN_2PI + i0.ln() + log_sum;
        if p == 1 {
            // M_0/M_1 = 1/r_1, φ/M_1 = 1/I_1 = 1/(I_0 r_1)
            let d1 = 1.0 / r_p;
            return (log_m, d1, 1.0 / (i0 * r_p) - d1 * d1);
        }
        let d1 = p as f64 / r_p;
        (log_m, d1, (p * (p - 1)) as f64 / (r_p * r_pm1) - d1 * d1)
    }
}

/// `log M_p(alpha)` for `p ≥ 1`, stable for all `alpha`.
///
/// Evaluated in `f64` by [`log_m_p_derivs`] and lifted to `S` with the
/// second-order Taylor expansion in `δ = alpha − alpha.re()`: exact for `f64`
/// and for first- and second-order dual numbers, and independent of the
/// number of dual directions.
#[inline]
pub fn log_m_p<S: Scalar>(alpha: S, p: usize) -> S {
    let a0 = alpha.re();
    if S::DERIV_ORDER == 0 {
        return S::from_f64(log_m_p_value(a0, p));
    }
    let (l, d1, d2) = log_m_p_derivs(a0, p);
    let delta = alpha - a0;
    if S::DERIV_ORDER == 1 {
        delta * d1 + l
    } else {
        (delta * (0.5 * d2) + d1) * delta + l
    }
}

/// `log M_p(a)` in `f64` without derivatives (see [`log_m_p_derivs`]).
#[inline]
pub fn log_m_p_value(a: f64, p: usize) -> f64 {
    assert!(p >= 1, "log_m_p needs p >= 1");
    if a >= CF_BELOW {
        let cdf = cdf_fast(a);
        let pdf = norm_pdf(a);
        let m1 = a * cdf + pdf;
        if p == 1 {
            return m1.ln();
        }
        let (mut m_km1, mut m_k) = (m1, (a * a + 1.0) * cdf + a * pdf);
        for k in 3..=p {
            let next = a * m_k + (k - 1) as f64 * m_km1;
            m_km1 = m_k;
            m_k = next;
        }
        m_k.ln()
    } else {
        log_m_p_derivs(a, p).0
    }
}

/// Log-density at the unit vector `y` for stacked `eta = (mu, gamma)`.
///
/// # Panics
/// If `eta.len()` is not `(d²+d−2)/2` with `d = y.len()`.
#[inline]
pub fn loglik_eta<S: Scalar>(eta: &[S], y: &[f64]) -> S {
    let d = y.len();
    assert_eq!(
        eta.len(),
        dims::eta_len(d),
        "eta has length {} but d = {d} needs {}",
        eta.len(),
        dims::eta_len(d)
    );
    let (mu, gamma) = eta.split_at(d);
    loglik_mu_gamma_generic(mu, gamma, y)
}

/// Log-density for separate `mu` and `gamma`, generic over [`Scalar`].
#[inline]
pub fn loglik_mu_gamma_generic<S: Scalar>(mu: &[S], gamma: &[S], y: &[f64]) -> S {
    match mu.len() {
        3 => loglik_fixed::<S, 3>(mu, gamma, y),
        4 => loglik_fixed::<S, 4>(mu, gamma, y),
        5 => loglik_fixed::<S, 5>(mu, gamma, y),
        6 => loglik_fixed::<S, 6>(mu, gamma, y),
        7 => loglik_fixed::<S, 7>(mu, gamma, y),
        8 => loglik_fixed::<S, 8>(mu, gamma, y),
        d => {
            let mut z = vec![S::zero(); d];
            let mut k = vec![S::zero(); d];
            let mut rt = vec![S::zero(); d];
            loglik_core(mu, gamma, y, &mut z, &mut k, &mut rt)
        }
    }
}

#[inline(always)]
fn loglik_fixed<S: Scalar, const D: usize>(mu: &[S], gamma: &[S], y: &[f64]) -> S {
    let mut z = [S::zero(); D];
    let mut k = [S::zero(); D];
    let mut rt = [S::zero(); D];
    loglik_core(mu, gamma, y, &mut z, &mut k, &mut rt)
}

/// `(q, y·mu, |mu|²)` with `q = yᵀ V⁻¹ y`, computed without forming `V`, the
/// basis or any trigonometric function. Scratch slices must have length
/// `≥ d`.
///
/// * `z = U_{1..d−1}ᵀ y` uses the closed form of `onb(mu)` with running sums,
///   in `O(d)`.
/// * `p = Rᵀ z` applies the transposed plane rotations of `R` to the vector
///   `z` (instead of rotating `d`-vectors), in the order of
///   [`crate::rotation::givens_sequence`]. Each angle enters only through
///   its cosine and sine, which are ratios of suffix norms of its `gamma`
///   group: `cos φ_{i,j} = G_j/‖G_{j..}‖`, `sin φ_{i,j} = ‖G_{j+1..}‖/‖G_{j..}‖`,
///   `(cos θ_i, sin θ_i) = (G_{n−1}, G_n)/‖(G_{n−1}, G_n)‖`.
/// * `q = Σ_j p_j²/λ_j + (μ̂·y)²` with `1/λ_1 = (Π_i K_i^{d−1−i})^{1/(d−1)}`,
///   `1/λ_{j+1} = (1/λ_j)/K_j`, `K_i = ‖G_i‖ + 1`.
#[inline(always)]
pub fn quad_form_parts<S: Scalar>(
    mu: &[S],
    gamma: &[S],
    y: &[f64],
    z: &mut [S],
    k: &mut [S],
    rt: &mut [S],
) -> (S, S, S) {
    let d = mu.len();
    debug_assert!(d >= 3 && y.len() == d && gamma.len() == dims::gamma_len(d));
    // ---- projections onto the ONB columns 0..d-2
    let n2 = mu[0] * mu[0] + mu[1] * mu[1];
    z[0] = if n2.re() == 0.0 {
        S::from_f64(y[0])
    } else {
        (mu[0] * y[1] - mu[1] * y[0]) / n2.sqrt()
    };
    let mut s = n2; // Σ_{k < i0} mu_k²
    let mut dot = mu[0] * y[0] + mu[1] * y[1]; // Σ_{k < i0} mu_k y_k
    for i0 in 2..d {
        let mi = mu[i0];
        z[i0 - 1] = if s.re() == 0.0 {
            S::from_f64(y[i0 - 1])
        } else {
            (mi * dot - s * y[i0]) / (s * (mi * mi + s)).sqrt()
        };
        s += mi * mi;
        dot += mi * y[i0];
    }
    let (mm, ymu) = (s, dot);
    let t = if mm.re() == 0.0 {
        S::from_f64(y[d - 1])
    } else {
        ymu / mm.sqrt()
    };

    // ---- rotations: groups d-2, d-3, …, 1 (see `givens_sequence`)
    let ng = d - 2;
    for gr in (1..=ng).rev() {
        let off = gr * (gr + 1) / 2 - 1;
        let n = gr + 1;
        let g = &gamma[off..off + n];
        // suffix norms rt[j] = ‖G_{j..n}‖
        let mut acc = S::zero();
        for j in (0..n).rev() {
            acc += g[j] * g[j];
            rt[j] = acc.sqrt();
        }
        k[gr - 1] = rt[0] + 1.0;
        // theta_gr acts on plane (0, 1)
        let (c, sn) = if rt[n - 2].re() == 0.0 {
            (S::one(), S::zero())
        } else {
            let inv = S::one() / rt[n - 2];
            (g[n - 2] * inv, g[n - 1] * inv)
        };
        rotate_t(z, 0, c, sn);
        if gr >= 2 {
            // phi_{gr, gr-j} (1-based within the group) on plane (j, j+1), j = 1..gr-1
            for j in 1..gr {
                let jj = gr - j - 1; // 0-based phi index within the group
                let (c, sn) = if rt[jj].re() == 0.0 {
                    (S::one(), S::zero())
                } else {
                    let inv = S::one() / rt[jj];
                    (g[jj] * inv, rt[jj + 1] * inv)
                };
                rotate_t(z, j, c, sn);
            }
        }
    }

    // ---- q = Σ_j p_j² / λ_j + t²
    let mut prod = S::one();
    for i in 1..=ng {
        let e = (d - 1 - i) as i32;
        prod *= if e == 1 { k[i - 1] } else { k[i - 1].powi(e) };
    }
    let mut inv_l = root_d_minus_1(prod, d);
    let mut q = t * t;
    for j in 0..d - 1 {
        q += z[j] * z[j] * inv_l;
        if j < ng {
            inv_l /= k[j];
        }
    }
    (q, ymu, mm)
}

/// Apply the transpose of the plane rotation `[[c, −s], [s, c]]` in
/// coordinates `(a, a+1)` to `z`.
#[inline(always)]
fn rotate_t<S: Scalar>(z: &mut [S], a: usize, c: S, s: S) {
    let (x, y) = (z[a], z[a + 1]);
    z[a] = c * x + s * y;
    z[a + 1] = c * y - s * x;
}

#[inline(always)]
fn loglik_core<S: Scalar>(
    mu: &[S],
    gamma: &[S],
    y: &[f64],
    z: &mut [S],
    k: &mut [S],
    rt: &mut [S],
) -> S {
    let d = mu.len();
    let p = d - 1;
    let (q, ymu, mm) = quad_form_parts(mu, gamma, y, z, k, rt);
    let alpha = ymu / q.sqrt();
    S::from_f64(-0.5 * p as f64 * LN_2PI) - q.ln() * (0.5 * d as f64)
        + (alpha * alpha - mm) * 0.5
        + log_m_p(alpha, p)
}

/// Log-density given a precomputed frame (reference implementation; the
/// likelihood itself uses the faster frame-free kernel above).
#[inline]
pub fn loglik_frame<S: Scalar>(frame: &Frame<S>, mu: &[S], y: &[f64]) -> S {
    let d = y.len();
    let p = d - 1;
    let q = frame.quad_form_inv(y);
    let mut ymu = S::zero();
    let mut mm = S::zero();
    for k in 0..d {
        ymu += mu[k] * y[k];
        mm += mu[k] * mu[k];
    }
    let alpha = ymu / q.sqrt();
    S::from_f64(-0.5 * p as f64 * LN_2PI) - q.ln() * (0.5 * d as f64)
        + (alpha * alpha - mm) * 0.5
        + log_m_p(alpha, p)
}

/// Log-density for separate `mu` and `gamma` (`f64`).
pub fn loglik_mu_gamma(mu: &[f64], gamma: &[f64], y: &[f64]) -> f64 {
    loglik_mu_gamma_generic(mu, gamma, y)
}

#[inline]
fn grad_dual<const N: usize>(eta: &[f64], y: &[f64], grad: &mut [f64]) -> f64 {
    let x: [Dual<N>; N] = seed_array(eta);
    let v = loglik_eta(&x, y);
    grad.copy_from_slice(&v.eps);
    v.re
}

/// Log-density and its exact gradient with respect to `eta`.
///
/// Uses the hand-written reverse-mode gradient
/// [`crate::adjoint::loglik_grad_adjoint`] (1.3–2.3× the cost of the value
/// for every `d`, versus ~N× for forward mode); see
/// [`loglik_eta_grad_dual`] for the forward-mode reference.
#[inline]
pub fn loglik_eta_grad(eta: &[f64], y: &[f64], grad: &mut [f64]) -> f64 {
    crate::adjoint::loglik_grad_adjoint(eta, y, grad)
}

/// Forward-mode reference for [`loglik_eta_grad`]: dual numbers for
/// `d = 3..=8` (`eta` lengths 5, 9, 14, 20, 27, 35) and central finite
/// differences otherwise.
pub fn loglik_eta_grad_dual(eta: &[f64], y: &[f64], grad: &mut [f64]) -> f64 {
    debug_assert_eq!(eta.len(), grad.len());
    match eta.len() {
        5 => grad_dual::<5>(eta, y, grad),
        9 => grad_dual::<9>(eta, y, grad),
        14 => grad_dual::<14>(eta, y, grad),
        20 => grad_dual::<20>(eta, y, grad),
        27 => grad_dual::<27>(eta, y, grad),
        35 => grad_dual::<35>(eta, y, grad),
        _ => {
            let mut e = eta.to_vec();
            for i in 0..eta.len() {
                let h = 1e-6 * eta[i].abs().max(1.0);
                e[i] = eta[i] + h;
                let fp = loglik_eta(&e, y);
                e[i] = eta[i] - h;
                let fm = loglik_eta(&e, y);
                e[i] = eta[i];
                grad[i] = (fp - fm) / (2.0 * h);
            }
            loglik_eta(eta, y)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stats::normal::norm_pdf;

    /// M_p(a) = ∫_0^∞ t^p φ(t − a) dt by composite Simpson on a wide grid.
    fn m_p_quad(a: f64, p: usize) -> f64 {
        let hi = a.max(0.0) + 40.0;
        let n = 400_000;
        let h = hi / n as f64;
        let f = |t: f64| t.powi(p as i32) * norm_pdf(t - a);
        let mut s = f(0.0) + f(hi);
        for i in 1..n {
            let t = i as f64 * h;
            s += if i % 2 == 1 { 4.0 } else { 2.0 } * f(t);
        }
        s * h / 3.0
    }

    #[test]
    fn log_m_p_matches_quadrature() {
        for &a in &[-12.0, -8.5, -7.5, -3.0, 0.0, 2.0, 6.0] {
            for p in 1..=5 {
                let exact = m_p_quad(a, p).ln();
                let got = log_m_p(a, p);
                assert!(
                    (got - exact).abs() < 1e-8 * exact.abs().max(1.0),
                    "a={a} p={p}: {got} vs {exact}"
                );
            }
        }
    }

    #[test]
    fn log_m_p_continuous_at_switch() {
        for p in 1..=7 {
            let lo = log_m_p(CF_BELOW - 1e-12, p);
            let hi = log_m_p(CF_BELOW, p);
            assert!(
                (lo - hi).abs() < 1e-12 * lo.abs().max(1.0),
                "p={p}: {lo} vs {hi}"
            );
        }
    }

    #[test]
    fn log_m_p_matches_r_integrate_at_minus_8() {
        // R: log(integrate(function(t) t^p*dnorm(t+8), 0, Inf, rel.tol=1e-13)$value)
        let r = [
            -37.1223642616926,
            -38.5519983389687,
            -39.5894945087745,
            -40.3521504313968,
            -40.9040559649905,
            -41.2856170262934,
            -41.5246174765082,
        ];
        for (i, &v) in r.iter().enumerate() {
            let got = log_m_p(-8.0, i + 1);
            assert!(
                (got - v).abs() < 1e-12 * v.abs(),
                "p={}: {got} vs {v}",
                i + 1
            );
        }
    }

    #[test]
    fn log_m_p_derivs_match_finite_differences() {
        for &a in &[
            -40.0, -9.0, -4.0, -2.0000001, -1.9999999, -0.5, 0.0, 1.5, 7.0,
        ] {
            for p in 1..=8 {
                let (l, d1, d2) = log_m_p_derivs(a, p);
                let h = 1e-4 * (1.0f64).max(a.abs());
                let (lp, d1p, _) = log_m_p_derivs(a + h, p);
                let (lm, d1m, _) = log_m_p_derivs(a - h, p);
                let fd1 = (lp - lm) / (2.0 * h);
                let fd2 = (d1p - d1m) / (2.0 * h);
                assert!(
                    (d1 - fd1).abs() < 1e-6 * d1.abs().max(1.0),
                    "a={a} p={p}: d1 {d1} vs {fd1}"
                );
                assert!(
                    (d2 - fd2).abs() < 1e-5 * d2.abs().max(1.0),
                    "a={a} p={p}: d2 {d2} vs {fd2}"
                );
                assert!(l.is_finite());
            }
        }
    }

    #[test]
    fn log_m_p_derivative_in_the_tail() {
        // d/da log M_p = p M_{p-1}/M_p
        for &a in &[-30.0, -8.0, -3.0, -2.5] {
            for p in 2..=6 {
                let x = Dual::<1>::variable(a, 0);
                let d = log_m_p(x, p).eps[0];
                let expect = (p as f64) * (log_m_p(a, p - 1) - log_m_p(a, p)).exp();
                assert!(
                    (d - expect).abs() < 1e-10 * expect.abs(),
                    "a={a} p={p}: {d} vs {expect}"
                );
            }
        }
    }

    #[test]
    fn gradient_matches_finite_differences() {
        let cases: &[(Vec<f64>, Vec<f64>)] = &[
            (vec![1.2, -0.4, 0.9, 0.5, -0.7], vec![0.6, 0.0, 0.8]),
            (vec![-0.3, 2.0, 0.4, -1.5, 0.2], vec![-0.48, 0.6, 0.64]),
            (
                vec![0.5, 1.0, -1.0, 2.0, 0.6, -0.2, 0.3, 0.9, -0.5],
                vec![0.5, 0.5, -0.5, 0.5],
            ),
            (
                vec![
                    0.2, -0.9, 1.1, 0.4, 1.3, 0.3, -0.6, 0.8, 0.1, 0.5, -0.4, 0.7, 0.2, -0.3,
                ],
                vec![
                    0.1,
                    -0.7,
                    0.5,
                    0.3,
                    (1.0f64 - 0.01 - 0.49 - 0.25 - 0.09).sqrt(),
                ],
            ),
        ];
        for (eta, y) in cases {
            let mut g = vec![0.0; eta.len()];
            let f = loglik_eta_grad(eta, y, &mut g);
            assert!((f - loglik_eta(eta, y)).abs() < 1e-14);
            let mut e = eta.clone();
            for i in 0..eta.len() {
                let h = 1e-6;
                e[i] = eta[i] + h;
                let fp: f64 = loglik_eta(&e, y);
                e[i] = eta[i] - h;
                let fm: f64 = loglik_eta(&e, y);
                e[i] = eta[i];
                let fd = (fp - fm) / (2.0 * h);
                assert!(
                    (g[i] - fd).abs() < 1e-6 * fd.abs().max(1.0),
                    "i={i}: ad {} fd {fd}",
                    g[i]
                );
            }
        }
    }

    #[test]
    fn density_integrates_to_one_d3() {
        // Gauss–Legendre would be nicer; a fine midpoint grid in (cos θ, φ) is enough.
        let nz = 800;
        let nphi = 800;
        for (mu, gamma) in [
            (vec![1.0, 2.0, 1.5], vec![0.7, -0.3]),
            (vec![0.3, -0.2, 0.1], vec![2.0, 1.0]),
            (vec![4.0, 0.0, 0.0], vec![-1.5, 3.0]),
        ] {
            let mut total = 0.0;
            for i in 0..nz {
                let z = -1.0 + (i as f64 + 0.5) * 2.0 / nz as f64;
                let r = (1.0 - z * z).sqrt();
                for j in 0..nphi {
                    let ph = (j as f64 + 0.5) * 2.0 * std::f64::consts::PI / nphi as f64;
                    let y = [r * ph.cos(), r * ph.sin(), z];
                    total += loglik_mu_gamma(&mu, &gamma, &y).exp();
                }
            }
            total *= (2.0 / nz as f64) * (2.0 * std::f64::consts::PI / nphi as f64);
            assert!((total - 1.0).abs() < 1e-4, "{mu:?} {gamma:?}: {total}");
        }
    }

    #[test]
    fn fast_kernel_matches_frame_reference() {
        let mut st = 0xC0FFEEu64;
        let mut next = || {
            st = st
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((st >> 11) as f64 / (1u64 << 53) as f64) * 4.0 - 2.0
        };
        let mut worst: f64 = 0.0;
        for d in 3..=10usize {
            let g = dims::gamma_len(d);
            for case in 0..60 {
                let mut mu: Vec<f64> = (0..d).map(|_| 2.0 * next()).collect();
                let mut gamma: Vec<f64> = (0..g).map(|_| next()).collect();
                match case % 6 {
                    1 => {
                        mu[0] = 0.0;
                        mu[1] = 0.0;
                    }
                    2 => {
                        for v in mu.iter_mut().take(d - 1) {
                            *v = 0.0;
                        }
                    }
                    3 => gamma.iter_mut().for_each(|v| *v = 0.0),
                    4 => {
                        // zero tails inside groups
                        for v in gamma.iter_mut().skip(1).step_by(2) {
                            *v = 0.0;
                        }
                    }
                    5 => gamma.iter_mut().for_each(|v| *v *= 4.0),
                    _ => {}
                }
                let raw: Vec<f64> = (0..d).map(|_| next()).collect();
                let nr = raw.iter().map(|v| v * v).sum::<f64>().sqrt();
                let y: Vec<f64> = raw.iter().map(|v| v / nr).collect();
                let fast = loglik_mu_gamma(&mu, &gamma, &y);
                let frame = Frame::new(&mu[..], &gamma[..]);
                let reference = loglik_frame(&frame, &mu, &y);
                let rel = (fast - reference).abs() / reference.abs().max(1.0);
                worst = worst.max(rel);
                assert!(
                    rel < 1e-11,
                    "d={d} case={case}: fast {fast} vs frame {reference}"
                );
            }
        }
        eprintln!("fast kernel vs frame reference: worst relative difference {worst:.2e}");
    }

    #[test]
    fn finite_far_from_the_mode() {
        // |mu| = 40, y antipodal: R returns -Inf here.
        let v = loglik_mu_gamma(&[40.0, 0.0, 0.0], &[0.3, 0.2], &[-1.0, 0.0, 0.0]);
        assert!(v.is_finite() && v < -700.0);
    }
}
