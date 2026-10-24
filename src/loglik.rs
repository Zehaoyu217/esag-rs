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
use crate::stats::normal::LN_2PI;

/// Below this `alpha` the continued fraction is used.
const CF_BELOW: f64 = -2.0;
/// Depth of the backward continued fraction (converged to rounding for b ≥ 2).
const CF_DEPTH: usize = 100;

/// `log M_p(alpha)` for `p ≥ 1`, stable for all `alpha`.
pub fn log_m_p<S: Scalar>(alpha: S, p: usize) -> S {
    assert!(p >= 1, "log_m_p needs p >= 1");
    if alpha.re() >= CF_BELOW {
        let cdf = alpha.norm_cdf();
        let pdf = alpha.norm_pdf();
        let m1 = alpha * cdf + pdf;
        if p == 1 {
            return m1.ln();
        }
        let m2 = (alpha * alpha + 1.0) * cdf + alpha * pdf;
        let (mut prev, mut cur) = (m1, m2);
        for k in 3..=p {
            let next = alpha * cur + prev * ((k - 1) as f64);
            prev = cur;
            cur = next;
        }
        cur.ln()
    } else {
        let b = -alpha;
        // r_N ≈ sqrt(N) − b/2 for large N; the guess only affects the far tail.
        let guess = ((CF_DEPTH as f64).sqrt() - 0.5 * b.re()).max(1e-3);
        let mut r = S::from_f64(guess);
        let mut log_sum = S::zero();
        for k in (1..CF_DEPTH).rev() {
            r = S::from_f64(k as f64) / (b + r);
            if k <= p {
                log_sum += r.ln();
            }
        }
        let i0 = S::one() / (b + r);
        -(alpha * alpha) * 0.5 - 0.5 * LN_2PI + i0.ln() + log_sum
    }
}

/// Log-density at the unit vector `y` for stacked `eta = (mu, gamma)`.
///
/// # Panics
/// If `eta.len()` is not `(d²+d−2)/2` with `d = y.len()`.
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
    let frame = Frame::new(mu, gamma);
    loglik_frame(&frame, mu, y)
}

/// Log-density given a precomputed frame.
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
    let frame = Frame::new(mu, gamma);
    loglik_frame(&frame, mu, y)
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
/// Uses forward-mode AD for `d = 3..=8` (`eta` lengths 5, 9, 14, 20, 27, 35)
/// and central finite differences otherwise.
pub fn loglik_eta_grad(eta: &[f64], y: &[f64], grad: &mut [f64]) -> f64 {
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
    fn finite_far_from_the_mode() {
        // |mu| = 40, y antipodal: R returns -Inf here.
        let v = loglik_mu_gamma(&[40.0, 0.0, 0.0], &[0.3, 0.2], &[-1.0, 0.0, 0.0]);
        assert!(v.is_finite() && v < -700.0);
    }
}
