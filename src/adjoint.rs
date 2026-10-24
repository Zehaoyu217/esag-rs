//! Hand-written reverse-mode (adjoint) gradient of the ESAG log-density.
//!
//! [`loglik_grad_adjoint`] returns the same value as
//! [`loglik_eta::<f64>`](crate::loglik::loglik_eta) (bit for bit: the forward
//! pass performs the same floating-point operations in the same order) and
//! the exact gradient with respect to `eta = (mu, gamma)`. The forward-mode
//! path [`loglik_eta_grad`](crate::loglik::loglik_eta_grad) carries `N =
//! (d²+d−2)/2` tangents through every operation, so it costs about `N` times
//! the value; here one forward pass records the few intermediates the
//! reverse sweep needs (on the stack for `d ≤ 8`), and one reverse sweep of
//! about the same length as the forward pass yields all `N` partial
//! derivatives.
//!
//! # The computation being differentiated
//!
//! This is the frame-free kernel of
//! [`quad_form_parts`](crate::loglik::quad_form_parts) followed by the
//! log-density (`p = d − 1`; see `docs/MATH.md`):
//!
//! 1. *Projections onto `onb(mu)`.* With running sums
//!    `s_i = Σ_{k<i} mu_k²`, `dot_i = Σ_{k<i} mu_k y_k`:
//!    `z_0 = (mu_0 y_1 − mu_1 y_0)/√s_2`,
//!    `z_{i−1} = (mu_i dot_i − s_i y_i)/√(s_i s_{i+1})` for `i = 2..d−1`,
//!    and `t = dot_d/√s_d`. Degenerate columns (`s_2 = 0`, resp. `s_i = 0`)
//!    are the constants `y_0`, resp. `y_{i−1}`; `t = y_{d−1}` when `mu = 0`.
//! 2. *Rotations.* For `gamma` groups `gr = d−2, …, 1` (group `gr` holds
//!    `G_0..G_gr`), with suffix norms `rt_j = ‖G_{j..}‖`: the transposed
//!    plane rotation `(z_a, z_{a+1}) ← (c z_a + s z_{a+1}, c z_{a+1} − s z_a)`
//!    on plane `(0, 1)` with `(c, s) = (G_{gr−1}, G_gr)/rt_{gr−1}`, then on
//!    planes `(j, j+1)`, `j = 1..gr−1`, with `(c, s) = (G_jj, rt_{jj+1})/rt_jj`,
//!    `jj = gr−j−1`; `(c, s) = (1, 0)` whenever the norm is zero. Each
//!    rotation thus owns one suffix norm (`rt_{gr−1}`, `rt_{gr−2}`, …, `rt_0`,
//!    in the order the running sum produces them), and `K_gr = rt_0 + 1`.
//! 3. `q = t² + Σ_j z_j² ℓ_j` with `ℓ_0 = (Π_i K_i^{d−1−i})^{1/(d−1)}` and
//!    `ℓ_{j+1} = ℓ_j / K_{j+1}`.
//! 4. `alpha = dot_d/√q`,
//!    `ll = −(p/2) log 2π − (d/2) log q + ½(alpha² − s_d) + log M_p(alpha)`.
//!
//! # The reverse sweep
//!
//! `ll` depends on `eta` only through `q`, `ymu = y·mu = dot_d` and
//! `mm = |mu|² = s_d`, so
//!
//! ```text
//! ∇ll = q̄ ∇q + ȳmu y − mu (mu block),   q̄ = −(d + ᾱ alpha)/(2q),
//! ȳmu = ᾱ/√q,   ᾱ = alpha + (log M_p)'(alpha),
//! ```
//!
//! and the reverse sweep computes `∇q` (seed `q̄ = 1`), which needs nothing
//! from `log M_p`: the processor overlaps the sweep with the evaluation of
//! `log M_p` and its derivative, and one final `N`-wide multiply-add forms
//! `∇ll`. Writing `x̄ = ∂q/∂x`:
//!
//! * `t̄ = 2t`, `z̄_j = 2 z_j ℓ_j`; with `w_j = z_j² ℓ_j` and `W = Σ_j w_j`,
//!   `K̄_m = ((d−2−m)/(d−1) · W − Σ_{j>m} w_j)/K_m` (0-based `m`), since
//!   `log ℓ_j` is linear in the `log K_m`.
//! * The rotations are undone in reverse order with the transposed Jacobian
//!   `x̄ = c ā − s b̄`, `ȳ = s ā + c b̄`. Each rotation's `(c, s) = (u, v)/r`
//!   (with `(u, v, r) = (G_jj, rt_{jj+1}, rt_jj)`, resp.
//!   `(G_{gr−1}, G_gr, rt_{gr−1})`) depends on `(u, v)` only through its angle,
//!   and `r = ‖(u, v)‖` (which is how the suffix norms recurse) only through
//!   the radius, so in polar form `ū = c r̄ − s ω`, `v̄ = s r̄ + c ω`, with
//!   `ω = (ā z'_{a+1} − b̄ z'_a)/r` from the rotated outputs `z'` and `r̄` the
//!   adjoint of `r` from its other uses (`K = rt_0 + 1`, or the next
//!   rotation's `v`). In reverse order the norms of a group come as
//!   `rt_0, rt_1, …`, so `r̄` is carried as a scalar. The polar form has no
//!   cancellation even when a norm is tiny (the Cartesian form
//!   `ū = c̄/r − c (c̄ c + s̄ s)/r` cancels terms of size `1/r`).
//! * The projections are differentiated by the quotient rule, carrying `s̄_i`
//!   and `dot̄_i` backwards through the running sums.
//!
//! # Guarded points
//!
//! The derivative conventions are those of [`Dual`](crate::dual::Dual), so
//! the result agrees with the forward-mode gradient everywhere, including
//! the non-differentiable points the kernel guards:
//!
//! * `√0` has derivative 0: a zero suffix norm `rt_j` passes no adjoint on
//!   (the isotropic point `gamma = 0`, groups with zero tails);
//! * branches that return constants (`(c, s) = (1, 0)` for a zero norm; the
//!   degenerate basis columns `z = y_c`; `t = y_{d−1}` at `mu = 0`) have zero
//!   derivative.
//!
//! Both are implemented branch-free in the reverse sweep by recording a zero
//! reciprocal norm for the degenerate case, which zeroes every term that
//! would pass through it.
//!
//! No allocation happens for `d ≤ 8` (const-generic stack buffers, as in
//! the value kernel); larger `d` use heap buffers.

use crate::dims;
use crate::loglik::log_m_p_derivs;
use crate::stats::normal::{LN_2PI, norm_cdf, norm_pdf};

/// Log-density at the unit vector `y` for stacked `eta = (mu, gamma)`, and
/// its exact gradient with respect to `eta`, written into `grad`, by one
/// forward and one reverse pass.
///
/// The value is identical to [`loglik_eta::<f64>`](crate::loglik::loglik_eta);
/// the gradient agrees with the forward-mode
/// [`loglik_eta_grad`](crate::loglik::loglik_eta_grad) to rounding, including
/// at the guarded points described in the [module documentation](self).
/// Unlike that function, it is exact (not a finite difference) for every
/// `d ≥ 3`.
///
/// # Panics
/// If `d = y.len() < 3`, if `eta.len()` is not `(d²+d−2)/2`, or if
/// `grad.len() != eta.len()`.
pub fn loglik_grad_adjoint(eta: &[f64], y: &[f64], grad: &mut [f64]) -> f64 {
    let d = y.len();
    assert!(d >= 3, "the ESAG density needs d >= 3, got d = {d}");
    assert_eq!(
        eta.len(),
        dims::eta_len(d),
        "eta has length {} but d = {d} needs {}",
        eta.len(),
        dims::eta_len(d)
    );
    assert_eq!(
        grad.len(),
        eta.len(),
        "grad has length {} but eta has length {}",
        grad.len(),
        eta.len()
    );
    // second parameter: R = (d−2)(d−1)/2 rotations
    match d {
        3 => adjoint_fixed::<3, 1>(eta, y, grad),
        4 => adjoint_fixed::<4, 3>(eta, y, grad),
        5 => adjoint_fixed::<5, 6>(eta, y, grad),
        6 => adjoint_fixed::<6, 10>(eta, y, grad),
        7 => adjoint_fixed::<7, 15>(eta, y, grad),
        8 => adjoint_fixed::<8, 21>(eta, y, grad),
        _ => {
            let (mu, gamma) = eta.split_at(d);
            let mut v = vec![0.0; NV * d];
            let mut tape = vec![[0.0; 5]; n_rotations(d)];
            adjoint_core(mu, gamma, y, grad, Work::new(&mut v, &mut tape))
        }
    }
}

/// Number of length-`d` vectors in [`Work`].
const NV: usize = 8;

/// Number of plane rotations `(d−2)(d−1)/2` (all groups together).
#[inline(always)]
const fn n_rotations(d: usize) -> usize {
    (d - 2) * (d - 1) / 2
}

/// Stack buffers for dimension `D` with `R = (D−2)(D−1)/2` rotations.
///
/// Kept out of line so that each dimension gets its own compact function,
/// and re-slicing to constant lengths gives every loop a compile-time trip
/// count.
#[inline(never)]
fn adjoint_fixed<const D: usize, const R: usize>(eta: &[f64], y: &[f64], grad: &mut [f64]) -> f64 {
    debug_assert_eq!(R, n_rotations(D));
    const { assert!(D >= 3) };
    let (mu, gamma) = eta.split_at(D);
    let (mu, gamma, y) = (&mu[..D], &gamma[..dims::gamma_len(D)], &y[..D]);
    let grad = &mut grad[..dims::eta_len(D)];
    let mut v = [[0.0; D]; NV];
    let mut tape = [[0.0; 5]; R];
    let [z, z0, sv, dv, ir, k, rk, wq] = &mut v;
    let w = Work {
        z,
        z0,
        sv,
        dv,
        ir,
        k,
        rk,
        wq,
        tape: &mut tape,
    };
    adjoint_core(mu, gamma, y, grad, w)
}

/// The forward-pass record and the reverse-pass scratch: [`NV`] vectors of
/// length `≥ d`, and one tape entry per rotation.
struct Work<'a> {
    /// The projections, rotated in place; then `z̄ = ∂q/∂z`.
    z: &'a mut [f64],
    /// The projections before any rotation.
    z0: &'a mut [f64],
    /// `sv[i] = s_i = Σ_{k<i} mu_k²` for `i = 2..d`.
    sv: &'a mut [f64],
    /// `dv[i] = dot_i = Σ_{k<i} mu_k y_k` for `i = 2..d`.
    dv: &'a mut [f64],
    /// `ir[c]`: reciprocal of the normaliser of projection `c`, 0 for a
    /// degenerate (constant) column.
    ir: &'a mut [f64],
    /// `k[m] = K_{m+1} = ‖G_{m+1}‖ + 1`.
    k: &'a mut [f64],
    /// `rk[m] = 1/k[m]`.
    rk: &'a mut [f64],
    /// `wq[j] = w_j = z_j² ℓ_j`, the terms of `q`.
    wq: &'a mut [f64],
    /// Per rotation, in forward order: its outputs `(z'_a, z'_{a+1})`,
    /// `(c, s)`, and the reciprocal of its norm (0 where the norm is 0).
    tape: &'a mut [[f64; 5]],
}

impl<'a> Work<'a> {
    /// Splits a heap buffer of length `NV·d` (fallback for `d > 8`).
    fn new(v: &'a mut [f64], tape: &'a mut [[f64; 5]]) -> Self {
        let d = v.len() / NV;
        let mut it = v.chunks_exact_mut(d);
        let mut next = || it.next().expect("buffer of length NV·d");
        Work {
            z: next(),
            z0: next(),
            sv: next(),
            dv: next(),
            ir: next(),
            k: next(),
            rk: next(),
            wq: next(),
            tape,
        }
    }
}

/// Forward pass (recording) and reverse sweep. See the module documentation.
#[inline(always)]
fn adjoint_core(mu: &[f64], gamma: &[f64], y: &[f64], grad: &mut [f64], w: Work<'_>) -> f64 {
    let d = mu.len();
    let p = d - 1;
    let ng = d - 2;
    let Work {
        z,
        z0,
        sv,
        dv,
        ir,
        k,
        rk,
        wq,
        tape,
    } = w;
    let (z, z0, sv, dv) = (&mut z[..d], &mut z0[..d], &mut sv[..d], &mut dv[..d]);
    let (ir, k, rk, wq) = (&mut ir[..d], &mut k[..d], &mut rk[..d], &mut wq[..d]);
    let tape = &mut tape[..n_rotations(d)];
    let (grad_mu, grad_gamma) = grad.split_at_mut(d);

    // =====================================================================
    // Forward pass. Every value-relevant operation mirrors
    // `loglik::quad_form_parts` / `loglik_core`, so the returned value is
    // bit-identical to `loglik_eta::<f64>`.
    // =====================================================================

    // ---- projections onto the ONB columns 0..d-2
    let n2 = mu[0] * mu[0] + mu[1] * mu[1];
    if n2 == 0.0 {
        z[0] = y[0];
        ir[0] = 0.0;
    } else {
        let r = n2.sqrt();
        z[0] = (mu[0] * y[1] - mu[1] * y[0]) / r;
        ir[0] = 1.0 / r;
    }
    let mut s = n2;
    let mut dot = mu[0] * y[0] + mu[1] * y[1];
    for i0 in 2..d {
        let mi = mu[i0];
        sv[i0] = s;
        dv[i0] = dot;
        if s == 0.0 {
            z[i0 - 1] = y[i0 - 1];
            ir[i0 - 1] = 0.0;
        } else {
            let r = (s * (mi * mi + s)).sqrt();
            z[i0 - 1] = (mi * dot - s * y[i0]) / r;
            ir[i0 - 1] = 1.0 / r;
        }
        s += mi * mi;
        dot += mi * y[i0];
    }
    let (mm, ymu) = (s, dot);
    let (t, irm) = if mm == 0.0 {
        (y[d - 1], 0.0)
    } else {
        let r = mm.sqrt();
        (ymu / r, 1.0 / r)
    };
    z0[..d - 1].copy_from_slice(&z[..d - 1]);

    // ---- rotations: groups d-2, d-3, …, 1, one tape entry per rotation
    let mut r = 0;
    for gr in (1..=ng).rev() {
        let off = gr * (gr + 1) / 2 - 1;
        let g = &gamma[off..off + gr + 1];
        // theta_gr on plane (0, 1); its norm is rt_{gr-1} = ‖(G_{gr-1}, G_gr)‖
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
        // phi on planes (j, j+1), j = 1..gr-1; norm rt_jj, jj = gr-j-1
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
        // rt_prev is now rt_0 = ‖G‖
        let kg = rt_prev + 1.0;
        k[gr - 1] = kg;
        rk[gr - 1] = 1.0 / kg;
    }

    // ---- q = Σ_j p_j² ℓ_j + t²; z becomes z̄ = 2 z ℓ
    let mut prod = 1.0;
    for i in 1..=ng {
        let e = (d - 1 - i) as i32;
        prod *= if e == 1 { k[i - 1] } else { k[i - 1].powi(e) };
    }
    let mut inv_l = if d == 3 {
        prod.sqrt()
    } else {
        prod.powf(1.0 / (d - 1) as f64)
    };
    let mut q = t * t;
    let mut wsum = 0.0;
    for j in 0..d - 1 {
        let wj = z[j] * z[j] * inv_l;
        q += wj;
        wq[j] = wj;
        wsum += wj;
        z[j] = 2.0 * z[j] * inv_l;
        if j < ng {
            inv_l /= k[j];
        }
    }

    // ---- log-density (the reverse sweep below does not depend on it)
    let sq = q.sqrt();
    let alpha = ymu / sq;
    let (inv_q, inv_sq) = (1.0 / q, 1.0 / sq);
    let (lmp, dlmp) = log_m_p_d1(alpha, p);
    let ll =
        -0.5 * p as f64 * LN_2PI - q.ln() * (0.5 * d as f64) + (alpha * alpha - mm) * 0.5 + lmp;

    // =====================================================================
    // Reverse sweep for ∇q (seed q̄ = 1); z holds z̄.
    // =====================================================================

    // ---- rotations, in reverse: groups 1, 2, …, d-2 (∂q/∂gamma into grad).
    // Each rotation's (c, s) = (u, v)/r depends on (u, v) only through its
    // angle, and its norm r = ‖(u, v)‖ only through the radius, so
    //   ū = c r̄ − s ω,  v̄ = s r̄ + c ω,  ω = (∂q/∂angle)/r,
    // with r̄ the adjoint of r from its other uses (K = rt_0 + 1, or the
    // next rotation's v = rt_{jj+1}). In reverse order the norms of a group
    // come in the order rt_0, rt_1, …, so r̄ is carried as a scalar.
    let inv_p = 1.0 / p as f64;
    let mut head = 0.0; // Σ_{j≤m} w_j
    for gr in 1..=ng {
        let m = gr - 1;
        head += wq[m];
        // r̄t_0 = K̄_m = ((d−2−m)/(d−1) W − Σ_{j>m} w_j)/K_m
        let mut carry = ((d - 2 - m) as f64 * inv_p * wsum - (wsum - head)) * rk[m];
        let off = gr * (gr + 1) / 2 - 1;
        let gb = &mut grad_gamma[off..off + gr + 1];
        // phi rotations, last first: j = gr-1, …, 1, i.e. jj = 0, …, gr-2;
        // (u, v, r) = (G_jj, rt_{jj+1}, rt_jj)
        for jj in 0..gr - 1 {
            r -= 1;
            let (om, c, sn, rb) = rotate_t_adjoint(z, gr - jj - 1, &tape[r], carry);
            gb[jj] = c * rb - sn * om;
            carry = sn * rb + c * om;
        }
        // theta: (u, v, r) = (G_{gr-1}, G_gr, rt_{gr-1})
        r -= 1;
        let (om, c, sn, rb) = rotate_t_adjoint(z, 0, &tape[r], carry);
        gb[gr - 1] = c * rb - sn * om;
        gb[gr] = sn * rb + c * om;
    }

    // ---- projections, in reverse (∂q/∂mu into grad); (sb, db) = adjoints
    // of (s_{i0+1}, dot_{i0+1}), seeded through t = ymu/√mm
    let mut sb = -t * t * irm * irm;
    let mut db = 2.0 * t * irm;
    for i0 in (2..d).rev() {
        let mi = mu[i0];
        // s_{i0+1} = s_{i0} + mi², dot_{i0+1} = dot_{i0} + mi y_{i0}
        let mut gmi = 2.0 * mi * sb + y[i0] * db;
        // z_{i0-1} = num/√P, num = mi dot − s y_{i0}, P = s (mi² + s)
        let c = i0 - 1;
        let numb = z[c] * ir[c];
        let pb = -0.5 * numb * z0[c] * ir[c];
        let (si, di) = (sv[i0], dv[i0]);
        gmi += numb * di + 2.0 * pb * mi * si;
        db += numb * mi;
        sb += pb * (mi * mi + 2.0 * si) - numb * y[i0];
        grad_mu[i0] = gmi;
    }
    // z_0 = (mu_0 y_1 − mu_1 y_0)/√n2, n2 = s_2, dot_2 = mu_0 y_0 + mu_1 y_1
    let numb = z[0] * ir[0];
    let n2b = sb - 0.5 * numb * z0[0] * ir[0];
    grad_mu[0] = numb * y[1] + 2.0 * mu[0] * n2b + y[0] * db;
    grad_mu[1] = 2.0 * mu[1] * n2b + y[1] * db - numb * y[0];

    // ---- combine: ∇ll = q̄ ∇q + ȳmu ∇ymu − ½ ∇mm
    let ab = alpha + dlmp;
    let qb = -0.5 * (d as f64 + ab * alpha) * inv_q;
    let ymub = ab * inv_sq;
    for i in 0..d {
        grad_mu[i] = qb * grad_mu[i] + ymub * y[i] - mu[i];
    }
    for v in grad_gamma.iter_mut() {
        *v *= qb;
    }

    ll
}

/// Below this `alpha`, `log M_p` uses its continued fraction (the value of
/// `loglik::CF_BELOW`).
const CF_BELOW: f64 = -2.0;

/// `(log M_p(a), d/da log M_p(a))`, bit-identical to the first two outputs
/// of [`log_m_p_derivs`] (and the first to
/// [`log_m_p_value`](crate::loglik::log_m_p_value)).
///
/// Above the continued-fraction switch this is the forward recursion of
/// `log_m_p_value` plus one division, `(log M_p)' = p M_{p−1}/M_p`; skipping
/// the second derivative makes it about 1.7× faster than `log_m_p_derivs`
/// there (3–4 ns over the value instead of 12–17 ns). The tail (`a < −2`)
/// defers to `log_m_p_derivs`.
#[inline(always)]
fn log_m_p_d1(a: f64, p: usize) -> (f64, f64) {
    debug_assert!(p >= 2);
    if a >= CF_BELOW {
        let cdf = norm_cdf(a);
        let pdf = norm_pdf(a);
        let m1 = a * cdf + pdf;
        // M_{k−1}, M_k
        let (mut m_km1, mut m_k) = (m1, (a * a + 1.0) * cdf + a * pdf);
        for k in 3..=p {
            let next = a * m_k + (k - 1) as f64 * m_km1;
            m_km1 = m_k;
            m_k = next;
        }
        (m_k.ln(), p as f64 * (m_km1 / m_k))
    } else {
        let (l, d1, _) = log_m_p_derivs(a, p);
        (l, d1)
    }
}

/// `rotate_t` of `loglik` (apply the transpose of `[[c, −s], [s, c]]` in
/// coordinates `(a, a+1)`); returns the tape entry `[z'_a, z'_{a+1}, c, s,
/// 1/r]` with the rotated coordinates.
#[inline(always)]
fn rotate_t_record(z: &mut [f64], a: usize, c: f64, s: f64, inv: f64) -> [f64; 5] {
    let (x, y) = (z[a], z[a + 1]);
    let (xo, yo) = (c * x + s * y, c * y - s * x);
    z[a] = xo;
    z[a + 1] = yo;
    [xo, yo, c, s, inv]
}

/// Reverse of [`rotate_t_record`]: maps the output adjoints `(ā, b̄)` in
/// `zb[a..a+2]` to the input adjoints `(c ā − s b̄, s ā + c b̄)` in place and
/// returns `(ω, c, s, r̄)`, where `ω = (ā z'_{a+1} − b̄ z'_a)/r` is the angular
/// derivative over the radius (`∂z'_a/∂angle = z'_{a+1}`,
/// `∂z'_{a+1}/∂angle = −z'_a`) and `r̄ = rbar`, or 0 where `r = 0`
/// (`√0` has derivative 0; the constant `(c, s) = (1, 0)` gets `ω = 0` from
/// the recorded `1/r = 0`).
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
    use crate::loglik::log_m_p_value;

    #[test]
    fn log_m_p_d1_is_bit_identical_to_loglik() {
        for p in 2..=12 {
            for i in 0..=4000 {
                let a = -45.0 + 0.0137 * i as f64;
                for a in [a, CF_BELOW, CF_BELOW - 1e-15, CF_BELOW + 1e-15] {
                    let (l, d1) = log_m_p_d1(a, p);
                    let (l2, d12, _) = log_m_p_derivs(a, p);
                    assert_eq!(l.to_bits(), l2.to_bits(), "a={a} p={p}");
                    assert_eq!(d1.to_bits(), d12.to_bits(), "a={a} p={p}");
                    assert_eq!(l.to_bits(), log_m_p_value(a, p).to_bits(), "a={a} p={p}");
                }
            }
        }
    }
}
