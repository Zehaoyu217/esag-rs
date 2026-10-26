//! `γ` in a transported frame: Yu & Huang's map `γ ↦ V` written in the
//! tangent frame `Ξ(μ̂) = R(m → μ̂) E` of the log-tangent shape
//! ([`crate::logtangent::LtFrame`]) instead of `onb(μ)`.
//!
//! `onb(μ)` is singular on the great sphere `μ_0 = μ_1 = 0`; the
//! transported frame only at `μ̂ = −m`, with `m` a reference direction (the
//! data's mean direction). At `μ̂ = m` the two frames coincide (`E` is
//! `onb(m)`'s tangent part), so without covariates the family, the fitted
//! distribution and the `γ` coordinates at `μ̂ = m` are those of `γ`. With
//! covariates the regression model differs through how the frame turns as
//! `μ̂(x)` moves. With the mean direction near the pole of `onb`, the same
//! optimizer needs 22 instead of 115 iterations at `d = 3` and 97 instead of
//! 430 at `d = 4`, for the same `V(x)` error (`gamma_frame_pole` in
//! `src/regression/fit_profile.rs`).
//!
//! [`loglik_gt_grad`] is the reverse-mode kernel. It is the kernel of
//! [`crate::adjoint`] with the projections onto `onb(μ)` replaced by
//! `z = Eᵀ(y − κ v)`, `v = μ̂ + m`, `κ = 2(v·y)/(v·v)`, whose reverse sweep is
//! that of the log-tangent kernel: `ū = E z̄`,
//! `v̄ = −κ ū − (2(ū·v)/(v·v)) u`, `μ̂̄ = v̄ + 2(μ̂·y) y`,
//! `∂q/∂μ = (μ̂̄ − (μ̂·μ̂̄) μ̂)/‖μ‖`. Conventions as there: at `μ = 0` it uses
//! `μ̂ = m`, at `μ̂ = −m` exactly `v = E_0`, both with constant frames.

use crate::dims;
use crate::loglik::{cdf_fast, log_m_p_derivs, root_d_minus_1};
use crate::logtangent::LtFrame;
use crate::stats::normal::{LN_2PI, norm_pdf};

/// Log-density at the unit vector `y` for stacked `eta = (mu, γ)` in the
/// transported frame `frame`, and its exact gradient in `eta`, written into
/// `grad`.
///
/// # Panics
/// If `d = y.len() < 3`, the frame has another dimension, `eta.len()` is
/// not [`dims::eta_len`]`(d)`, or `grad.len() != eta.len()`.
pub fn loglik_gt_grad(frame: &LtFrame, eta: &[f64], y: &[f64], grad: &mut [f64]) -> f64 {
    let d = y.len();
    assert!(d >= 3, "the ESAG density needs d >= 3, got d = {d}");
    assert_eq!(
        frame.dim(),
        d,
        "frame dimension {} but d = {d}",
        frame.dim()
    );
    assert_eq!(eta.len(), dims::eta_len(d), "eta has length {}", eta.len());
    assert_eq!(grad.len(), eta.len(), "grad has length {}", grad.len());
    match d {
        3 => fixed::<3, 1>(frame, eta, y, grad),
        4 => fixed::<4, 3>(frame, eta, y, grad),
        5 => fixed::<5, 6>(frame, eta, y, grad),
        6 => fixed::<6, 10>(frame, eta, y, grad),
        7 => fixed::<7, 15>(frame, eta, y, grad),
        8 => fixed::<8, 21>(frame, eta, y, grad),
        _ => {
            let (mu, gamma) = eta.split_at(d);
            let mut v = vec![0.0; NV * d];
            let mut tape = vec![[0.0; 5]; (d - 2) * (d - 1) / 2];
            core(frame, mu, gamma, y, grad, &mut v, &mut tape)
        }
    }
}

/// Number of length-`d` scratch vectors.
const NV: usize = 8;

#[inline(never)]
fn fixed<const D: usize, const R: usize>(
    frame: &LtFrame,
    eta: &[f64],
    y: &[f64],
    grad: &mut [f64],
) -> f64 {
    let (mu, gamma) = eta.split_at(D);
    let mut v = [0.0; 64];
    let mut tape = [[0.0; 5]; R];
    core(
        frame,
        &mu[..D],
        gamma,
        &y[..D],
        grad,
        &mut v[..NV * D],
        &mut tape,
    )
}

#[inline(always)]
fn core(
    frame: &LtFrame,
    mu: &[f64],
    gamma: &[f64],
    y: &[f64],
    grad: &mut [f64],
    buf: &mut [f64],
    tape: &mut [[f64; 5]],
) -> f64 {
    let d = mu.len();
    let p = d - 1;
    let ng = d - 2;
    let m = frame.reference();
    let e = frame.tangent_basis();
    let mut it = buf.chunks_exact_mut(d);
    let mut next = || it.next().unwrap();
    let (z, k, rk, wq) = (next(), next(), next(), next());
    let (u, v, mh, ub) = (next(), next(), next(), next());
    let (grad_mu, grad_gamma) = grad.split_at_mut(d);

    // ---- frame: z = Eᵀ(y − κ v), v = μ̂ + m, t = μ̂·y
    let mut mm = 0.0;
    let mut ymu = 0.0;
    for i in 0..d {
        mm += mu[i] * mu[i];
        ymu += mu[i] * y[i];
    }
    let zero_mu = mm == 0.0;
    let irm = if zero_mu { 0.0 } else { 1.0 / mm.sqrt() };
    for i in 0..d {
        mh[i] = if zero_mu { m[i] } else { mu[i] * irm };
    }
    let t: f64 = if zero_mu {
        (0..d).map(|i| m[i] * y[i]).sum()
    } else {
        ymu * irm
    };
    let mut vv = 0.0;
    for i in 0..d {
        v[i] = mh[i] + m[i];
        vv += v[i] * v[i];
    }
    let antipode = vv == 0.0;
    if antipode {
        v.copy_from_slice(&e[..d]);
        vv = 1.0;
    }
    let vy: f64 = (0..d).map(|i| v[i] * y[i]).sum();
    let kappa = 2.0 * vy / vv;
    for i in 0..d {
        u[i] = y[i] - kappa * v[i];
    }
    for j in 0..p {
        let col = &e[j * d..(j + 1) * d];
        z[j] = (0..d).map(|i| col[i] * u[i]).sum();
    }

    // ---- rotations: groups d-2, …, 1 (as in `adjoint`)
    let mut r = 0;
    for gr in (1..=ng).rev() {
        let off = gr * (gr + 1) / 2 - 1;
        let g = &gamma[off..off + gr + 1];
        let mut acc = g[gr] * g[gr];
        acc += g[gr - 1] * g[gr - 1];
        let mut rt_prev = acc.sqrt();
        let (c, sn, inv) = if rt_prev == 0.0 {
            (1.0, 0.0, 0.0)
        } else {
            let inv = 1.0 / rt_prev;
            (g[gr - 1] * inv, g[gr] * inv, inv)
        };
        tape[r] = rotate_t_record(z, 0, c, sn, inv);
        r += 1;
        for j in 1..gr {
            let jj = gr - j - 1;
            acc += g[jj] * g[jj];
            let rtj = acc.sqrt();
            let (c, sn, inv) = if rtj == 0.0 {
                (1.0, 0.0, 0.0)
            } else {
                let inv = 1.0 / rtj;
                (g[jj] * inv, rt_prev * inv, inv)
            };
            tape[r] = rotate_t_record(z, j, c, sn, inv);
            r += 1;
            rt_prev = rtj;
        }
        let kg = rt_prev + 1.0;
        k[gr - 1] = kg;
        rk[gr - 1] = 1.0 / kg;
    }

    // ---- q = t² + Σ_j z_j² ℓ_j; z becomes z̄ = 2 z ℓ
    let mut prod = 1.0;
    for i in 1..=ng {
        let ex = (d - 1 - i) as i32;
        prod *= if ex == 1 { k[i - 1] } else { k[i - 1].powi(ex) };
    }
    let mut inv_l = root_d_minus_1(prod, d);
    let mut q = t * t;
    let mut wsum = 0.0;
    for j in 0..p {
        let wj = z[j] * z[j] * inv_l;
        q += wj;
        wq[j] = wj;
        wsum += wj;
        z[j] = 2.0 * z[j] * inv_l;
        if j < ng {
            inv_l /= k[j];
        }
    }
    let sq = q.sqrt();
    let alpha = ymu / sq;
    let (inv_q, inv_sq) = (1.0 / q, 1.0 / sq);
    let (lmp, dlmp) = log_m_p_d1(alpha, p);
    let ll =
        -0.5 * p as f64 * LN_2PI - q.ln() * (0.5 * d as f64) + (alpha * alpha - mm) * 0.5 + lmp;

    // ---- reverse: rotations (∂q/∂γ into grad_gamma, z̄ back to the frame)
    let inv_p = 1.0 / p as f64;
    let mut head = 0.0;
    for gr in 1..=ng {
        let mi = gr - 1;
        head += wq[mi];
        let mut carry = ((d - 2 - mi) as f64 * inv_p * wsum - (wsum - head)) * rk[mi];
        let off = gr * (gr + 1) / 2 - 1;
        let gb = &mut grad_gamma[off..off + gr + 1];
        for jj in 0..gr - 1 {
            r -= 1;
            let (om, c, sn, rb) = rotate_t_adjoint(z, gr - jj - 1, &tape[r], carry);
            gb[jj] = c * rb - sn * om;
            carry = sn * rb + c * om;
        }
        r -= 1;
        let (om, c, sn, rb) = rotate_t_adjoint(z, 0, &tape[r], carry);
        gb[gr - 1] = c * rb - sn * om;
        gb[gr] = sn * rb + c * om;
    }

    // ---- reverse: frame (∂q/∂μ into grad_mu)
    for i in 0..d {
        ub[i] = (0..p).map(|j| e[j * d + i] * z[j]).sum();
    }
    // μ̂̄ = v̄ + 2t y, v̄ = −κ ū − (2(ū·v)/(v·v)) u (zero for the constant
    // antipodal frame); then through μ̂ = μ/‖μ‖ (constant at μ = 0)
    let ubv: f64 = (0..d).map(|i| ub[i] * v[i]).sum();
    for i in 0..d {
        let vb = if antipode {
            0.0
        } else {
            -kappa * ub[i] - 2.0 * ubv / vv * u[i]
        };
        ub[i] = vb + 2.0 * t * y[i];
    }
    if zero_mu {
        grad_mu.iter_mut().for_each(|g| *g = 0.0);
    } else {
        let s: f64 = (0..d).map(|i| mh[i] * ub[i]).sum();
        for i in 0..d {
            grad_mu[i] = (ub[i] - s * mh[i]) * irm;
        }
    }

    // ---- combine: ∇ll = q̄ ∇q + ȳmu ∇ymu − ½ ∇mm
    let ab = alpha + dlmp;
    let qb = -0.5 * (d as f64 + ab * alpha) * inv_q;
    let ymub = ab * inv_sq;
    for i in 0..d {
        grad_mu[i] = qb * grad_mu[i] + ymub * y[i] - mu[i];
    }
    for g in grad_gamma.iter_mut() {
        *g *= qb;
    }
    ll
}

/// Below this `alpha`, `log M_p` uses its continued fraction.
const CF_BELOW: f64 = -2.0;

/// `(log M_p(a), d/da log M_p(a))`, as in [`crate::adjoint`].
#[inline(always)]
fn log_m_p_d1(a: f64, p: usize) -> (f64, f64) {
    if a >= CF_BELOW {
        let cdf = cdf_fast(a);
        let pdf = norm_pdf(a);
        let m1 = a * cdf + pdf;
        let (mut m_km1, mut m_k) = (m1, (a * a + 1.0) * cdf + a * pdf);
        for kk in 3..=p {
            let next = a * m_k + (kk - 1) as f64 * m_km1;
            m_km1 = m_k;
            m_k = next;
        }
        (m_k.ln(), p as f64 * (m_km1 / m_k))
    } else {
        let (l, d1, _) = log_m_p_derivs(a, p);
        (l, d1)
    }
}

#[inline(always)]
fn rotate_t_record(z: &mut [f64], a: usize, c: f64, s: f64, inv: f64) -> [f64; 5] {
    let (x, y) = (z[a], z[a + 1]);
    let (xo, yo) = (c * x + s * y, c * y - s * x);
    z[a] = xo;
    z[a + 1] = yo;
    [xo, yo, c, s, inv]
}

#[inline(always)]
fn rotate_t_adjoint(zb: &mut [f64], a: usize, rec: &[f64; 5], rbar: f64) -> (f64, f64, f64, f64) {
    let [xo, yo, c, s, inv] = *rec;
    let (ba, bb) = (zb[a], zb[a + 1]);
    zb[a] = c * ba - s * bb;
    zb[a + 1] = s * ba + c * bb;
    let om = (ba * yo - bb * xo) * inv;
    let rb = if inv == 0.0 { 0.0 } else { rbar };
    (om, c, s, rb)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gamma_smooth::{GsFrame, loglik_gs, loglik_gs_grad};

    fn pseudo(n: usize, seed: usize, scale: f64) -> Vec<f64> {
        (0..n)
            .map(|i| scale * ((i * 7919 + seed * 104_729) as f64 * 0.618_033_988_7).sin())
            .collect()
    }

    fn unit(v: Vec<f64>) -> Vec<f64> {
        let n = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        v.into_iter().map(|x| x / n).collect()
    }

    /// Value and gradient agree with the forward-mode kernel of the same
    /// map (`GsFrame::gamma_transported`), and at `μ̂ = m` the value is
    /// `γ`'s.
    #[test]
    fn matches_forward_mode_and_gamma_at_the_reference() {
        for d in 3..=8 {
            for case in 0..4 {
                let m = unit(pseudo(d, 30 + case, 1.0));
                let lt = LtFrame::new(&m);
                let gs = GsFrame::gamma_transported(&m);
                let mut eta = match case {
                    // μ̂ = m: frames coincide
                    0 => m.iter().map(|x| 1.7 * x).collect(),
                    // near the antipode
                    1 => m
                        .iter()
                        .zip(pseudo(d, 3, 0.01))
                        .map(|(a, b)| -2.0 * a + b)
                        .collect(),
                    _ => pseudo(d, 40 + case, 1.3),
                };
                eta.extend(pseudo(dims::gamma_len(d), 50 + case, 0.9));
                let y = unit(pseudo(d, 60 + case, 1.0));
                let mut g1 = vec![0.0; eta.len()];
                let mut g2 = vec![0.0; eta.len()];
                let f1 = loglik_gt_grad(&lt, &eta, &y, &mut g1);
                let f2 = loglik_gs_grad(&gs, &eta, &y, &mut g2);
                assert!(
                    (f1 - f2).abs() < 1e-12 * f2.abs().max(1.0),
                    "d={d} case {case}: {f1} vs {f2}"
                );
                for i in 0..eta.len() {
                    assert!(
                        (g1[i] - g2[i]).abs() < 1e-9 * g2[i].abs().max(1.0),
                        "d={d} case {case} i={i}: {} vs {}",
                        g1[i],
                        g2[i]
                    );
                }
                if case == 0 {
                    let f3: f64 = crate::loglik::loglik_eta(&eta, &y);
                    let f4: f64 = loglik_gs(&gs, &eta, &y);
                    assert!(
                        (f1 - f3).abs() < 1e-11 * f3.abs().max(1.0),
                        "d={d}: {f1} vs gamma {f3}"
                    );
                    assert!((f4 - f3).abs() < 1e-11 * f3.abs().max(1.0));
                }
            }
        }
    }
}

#[cfg(test)]
mod bench {
    use super::*;
    use std::time::Instant;

    /// Gradient cost per observation against `γ`'s adjoint kernel.
    /// `cargo test --release --lib gt_kernel_cost -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn gt_kernel_cost() {
        for d in [3usize, 5, 8] {
            let n = 20_000;
            let m: Vec<f64> = (0..d).map(|i| 1.0 + 0.1 * i as f64).collect();
            let lt = LtFrame::new(&m);
            let k = dims::eta_len(d);
            let etas: Vec<Vec<f64>> = (0..n)
                .map(|i| {
                    (0..k)
                        .map(|j| {
                            ((i * 31 + j * 7) as f64 * 0.37).sin() + if j < d { 1.5 } else { 0.0 }
                        })
                        .collect()
                })
                .collect();
            let ys: Vec<Vec<f64>> = (0..n)
                .map(|i| {
                    let v: Vec<f64> = (0..d)
                        .map(|j| ((i * 13 + j * 5) as f64 * 0.91).cos() + 1.0)
                        .collect();
                    let s = v.iter().map(|x| x * x).sum::<f64>().sqrt();
                    v.iter().map(|x| x / s).collect()
                })
                .collect();
            let mut g = vec![0.0; k];
            let time = |f: &mut dyn FnMut()| {
                let mut best = f64::INFINITY;
                for _ in 0..5 {
                    let t = Instant::now();
                    f();
                    best = best.min(t.elapsed().as_secs_f64());
                }
                best / n as f64 * 1e9
            };
            let mut acc = 0.0;
            let a = time(&mut || {
                for i in 0..n {
                    acc += crate::adjoint::loglik_grad_adjoint(&etas[i], &ys[i], &mut g);
                }
            });
            let b = time(&mut || {
                for i in 0..n {
                    acc += loglik_gt_grad(&lt, &etas[i], &ys[i], &mut g);
                }
            });
            println!(
                "d={d}: gamma (onb) adjoint {a:.0} ns/obs, transported {b:.0} ns/obs ({acc:.1e})"
            );
        }
    }
}
