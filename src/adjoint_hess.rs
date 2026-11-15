//! Exact Hessian of the ESAG log-density in `eta = (mu, gamma)`, for every
//! `d ≥ 3`, built on the forward pass and reverse sweep of the adjoint
//! gradient ([`crate::adjoint`]).
//!
//! [`loglik_hess_adjoint`] returns the value of
//! [`loglik_eta::<f64>`](crate::loglik::loglik_eta) and the gradient of
//! [`loglik_grad_adjoint`](crate::adjoint::loglik_grad_adjoint) bit for bit
//! (the same floating-point operations in the same order), and the exact
//! Hessian, which agrees with the second-order dual numbers of
//! [`crate::hdual`] to rounding.
//!
//! # Method
//!
//! Notation as in [`crate::adjoint`]: `w` are the projections of `y` onto
//! the first `d − 1` columns of `onb(mu)`, `z = P w` after the `R = (d−2)(d−1)/2`
//! plane rotations, `q = t² + Σ_j ℓ_j z_j²`, and
//! `ll = F(q, y·mu, |mu|²)`.
//!
//! **Angle coordinates.** Each rotation `r` has `(c, s) = (u, v)/ρ` with
//! `u` a γ entry, `v` the next γ entry or the norm of the group's tail and
//! `ρ = ‖(u, v)‖`. In the coordinates `θ̃ = (mu, ψ, ρ)`, with one angle
//! `ψ_r = atan2(v, u)` per rotation and the norm `ρ_m = ‖G_m‖` of every
//! group (`N` coordinates, as many as `eta`), `z` is a chain of rotations
//! by one angle each and the `ℓ_j` depend on the `ρ_m` only. There the
//! Hessian of `q` is
//!
//! * `2 Z̃ᵀ diag(ℓ) Z̃`, with `Z̃ = ∂z/∂(mu, ψ)`;
//! * the curvature of `φ = z̄ᵀ z` at fixed `z̄ = ∂q/∂z`: with forward states
//!   `f_r` and adjoints `b_r` of the rotation chain (the reverse sweep
//!   records them), `∂f_r/∂ψ_r = Ĵ_r f_r` (`Ĵ` maps `(x_a, x_{a+1})` to
//!   `(x_{a+1}, −x_a)`), so `∂²φ/∂ψ_k∂ψ_r = b_rᵀ Ĵ_r ∂f_r/∂ψ_k` for `k ≤ r`
//!   and `∂²φ/∂mu_i∂ψ_r = b_rᵀ Ĵ_r ∂f_r/∂mu_i`. One forward sweep of the
//!   tangents `∂f/∂mu_i` and `∂f/∂ψ_k` (the latter injected at rotation `k`
//!   as `Ĵ_k f_k`) gives `Z̃` at its end and every curvature entry on the
//!   way, at three flops per entry. Group `m` acts on coordinates `0..=m`
//!   only, so row `j` of `Z̃` vanishes on the angles of groups below `j`.
//!   The `mu mu` block adds `Σ_c w̄_c ∇²w_c` from closed-form second
//!   derivatives of the projections, and `t²` its own;
//! * the terms of `ℓ(ρ)`: `log ℓ_j` is linear in the `log K_m`,
//!   `K_m = ρ_m + 1`, so the `ρ` rows are suffix sums of `z̄ ⊙ Z̃` and of
//!   `ℓ_j z_j²`.
//!
//! **Outer layer.** `ll` depends on `eta` through `q`, `y·mu` and `|mu|²`
//! only, so `∇²ll = F_q ∇²q + F_qq ∇q∇qᵀ + F_qy(∇q yᵀ + y ∇qᵀ) + F_yy y yᵀ − I`
//! (the `y` terms and `−I` in the `mu` block), with the derivatives of
//! `log M_p` from [`log_m_p_derivs`]. It is applied in `θ̃`, where `y·mu`
//! and `|mu|²` depend on `mu` alone.
//!
//! **Back to γ.** The reverse sweep of the adjoint undoes the polar map
//! `(u, v) ↦ (ψ, ρ)` of one rotation at a time (group by group, the
//! rotation owning the group norm first; each `v` is the next rotation's
//! `ρ`). The Hessian follows the same steps, `H ← Tᵀ H T + ψ̄ ∇²ψ + ρ̄ ∇²ρ`
//! on two slots, with `T` the 2×2 Jacobian of the step and `ψ̄, ρ̄` the
//! gradient entries of the reverse sweep. As the steps of a group only mix
//! that group's slots, its rows are transformed in place (contiguous rows,
//! `O(N)` per step), copied into its columns, and its own block is redone
//! step by step with the second-order terms.
//!
//! **Cost** (Apple M5 Pro, one core, per observation, random points;
//! `cargo test --release --test adjoint_hess bench -- --ignored`): 0.09,
//! 0.22, 0.42, 0.76, 1.35 and 2.1 µs for d = 3..=8, i.e. 1.7–14× the
//! adjoint gradient, 2.7× faster than [`crate::hdual`] at d = 3 (9× at
//! d = 4, 20× at d = 5) and 5–7× faster than central differences of the
//! adjoint gradient (`2N + 1` gradients) for every d. Most of the time goes
//! into the two `O(N²)` passes (the assembly with `2 Z̃ᵀ diag(ℓ) Z̃`, and the
//! steps back to γ).
//!
//! **Accuracy.** Against `HDual<N>` the entries agree to ≤ 2e−13 of
//! `√(D_i D_j)` (`D_i` the largest entry of row `i`) on random points, near
//! and at zero groups, zero tails, the frame pole and `mu = 0`
//! (`tests/adjoint_hess.rs`). The exception is group 1 (the angle `θ_1`)
//! near zero, where both implementations, and the adjoint gradient, compute
//! the angular curvature `O(‖G_1‖)` of a nearly isotropic block from `O(1)`
//! terms: 1e−12 at `‖G_1‖ = 1e−3`, 1e−8 at `1e−7`.
//!
//! # Guarded points
//!
//! As in [`crate::hdual`]: a rotation with a zero norm and a degenerate
//! basis column are constants, and `√0` has zero first and second
//! derivatives. A degenerate rotation's polar step has `T = 0`. Where a
//! group's tail `G_t, …, G_n` is exactly zero (`1 ≤ t ≤ n − 1`, `n + 1`
//! entries), the hyper-dual Hessian also carries the curvature of the
//! nonzero suffix norms `‖G_{j..}‖`, `j < t`, in the zero entries,
//! `Σ_j (∂ll/∂‖G_{j..}‖)/‖G_{j..}‖` on their diagonal; it is added
//! explicitly. (`V(γ)` is discontinuous there, so neither value is a
//! derivative of the likelihood; the convention only keeps the two
//! implementations interchangeable.)
//!
//! No allocation happens for `d ≤ 8` (stack buffers); larger `d` use heap
//! buffers.

use crate::dims;
use crate::loglik::log_m_p_derivs;
use crate::stats::normal::LN_2PI;

/// Log-density at the unit vector `y` for stacked `eta = (mu, gamma)`, its
/// exact gradient (into `grad`) and its exact Hessian (into `hess`, `N × N`
/// row-major, symmetric, `N = eta.len()`).
///
/// The value is identical to [`loglik_eta::<f64>`](crate::loglik::loglik_eta)
/// and the gradient to
/// [`loglik_grad_adjoint`](crate::adjoint::loglik_grad_adjoint); the Hessian
/// agrees with [`crate::hdual`] to rounding, including at the guarded points
/// (see the [module documentation](self)).
///
/// # Panics
/// If `d = y.len() < 3`, if `eta.len()` is not `(d²+d−2)/2`, if
/// `grad.len() != eta.len()` or if `hess.len() != eta.len()²`.
pub fn loglik_hess_adjoint(eta: &[f64], y: &[f64], grad: &mut [f64], hess: &mut [f64]) -> f64 {
    let d = y.len();
    assert!(d >= 3, "the ESAG density needs d >= 3, got d = {d}");
    let n = dims::eta_len(d);
    assert_eq!(
        eta.len(),
        n,
        "eta has length {} but d = {d} needs {n}",
        eta.len()
    );
    assert_eq!(grad.len(), n, "grad must have length eta.len()");
    assert_eq!(hess.len(), n * n, "hess must have length eta.len()²");
    // const parameters: D, N, R = (D−2)(D−1)/2 rotations, ZL = (D−1)·N
    match d {
        3 => hess_fixed::<3, 5, 1, 10>(eta, y, grad, hess),
        4 => hess_fixed::<4, 9, 3, 27>(eta, y, grad, hess),
        5 => hess_fixed::<5, 14, 6, 56>(eta, y, grad, hess),
        6 => hess_fixed::<6, 20, 10, 100>(eta, y, grad, hess),
        7 => hess_fixed::<7, 27, 15, 162>(eta, y, grad, hess),
        8 => hess_fixed::<8, 35, 21, 245>(eta, y, grad, hess),
        _ => {
            let (mu, gamma) = eta.split_at(d);
            let nr = (d - 2) * (d - 1) / 2;
            let mut v = vec![0.0; NV * d];
            let mut nv = vec![0.0; NN * n];
            let mut blk = vec![0.0; (d - 1) * (d - 1)];
            let mut zt = vec![0.0; (d - 1) * n];
            let mut ztt = vec![0.0; (d - 1) * n];
            let mut tape = vec![[0.0; 5]; nr];
            let mut rec = vec![[0.0; 4]; nr];
            hess_core(
                mu,
                gamma,
                y,
                grad,
                hess,
                Work::new(
                    &mut v, &mut nv, &mut blk, &mut zt, &mut ztt, &mut tape, &mut rec,
                ),
            )
        }
    }
}

/// Number of length-`d` vectors in [`Work`].
const NV: usize = 12;
/// Number of length-`N` vectors in [`Work`].
const NN: usize = 4;

/// Stack buffers for dimension `D` (`N = eta_len(D)`, `R` rotations,
/// `ZL = (D−1)·N`).
#[inline(never)]
fn hess_fixed<const D: usize, const N: usize, const R: usize, const ZL: usize>(
    eta: &[f64],
    y: &[f64],
    grad: &mut [f64],
    hess: &mut [f64],
) -> f64 {
    const { assert!(D >= 3) };
    debug_assert_eq!(R, (D - 2) * (D - 1) / 2);
    debug_assert_eq!(N, dims::eta_len(D));
    debug_assert_eq!(ZL, (D - 1) * N);
    let n = N;
    let (mu, gamma) = eta.split_at(D);
    let (mu, gamma, y) = (&mu[..D], &gamma[..dims::gamma_len(D)], &y[..D]);
    let (grad, hess) = (&mut grad[..n], &mut hess[..n * n]);
    let mut v = [[0.0; D]; NV];
    let mut nv = [[0.0; N]; NN];
    let mut zt = [0.0; ZL];
    let mut ztt = [0.0; ZL];
    let mut tape = [[0.0; 5]; R];
    let mut rec = [[0.0; 4]; R];
    let [z, z0, sv, dv, ir, k, rk, wq, ell, zbar, kbar, wbar] = &mut v;
    let [gq, s_row, t_row, gt] = &mut nv;
    let mut blk = [[0.0; D]; D];
    let w = Work {
        z,
        z0,
        sv,
        dv,
        ir,
        k,
        rk,
        wq,
        ell,
        zbar,
        kbar,
        wbar,
        gq: &mut gq[..n],
        s_row: &mut s_row[..n],
        t_row: &mut t_row[..n],
        gt: &mut gt[..n],
        blk: blk.as_flattened_mut(),
        zt: &mut zt,
        ztt: &mut ztt,
        tape: &mut tape,
        rec: &mut rec,
    };
    hess_core(mu, gamma, y, grad, hess, w)
}

/// Forward record and scratch: [`NV`] vectors of length `≥ d`, [`NN`] of
/// length `N`, the tangents and two entries per rotation.
struct Work<'a> {
    /// Projections, rotated in place; then `z̄`, pulled back by the reverse
    /// sweep.
    z: &'a mut [f64],
    /// The projections `w` before any rotation.
    z0: &'a mut [f64],
    /// `sv[i] = s_i = Σ_{k<i} mu_k²`.
    sv: &'a mut [f64],
    /// `dv[i] = dot_i = Σ_{k<i} mu_k y_k`.
    dv: &'a mut [f64],
    /// Reciprocal normaliser of projection `c` (0 for a constant column).
    ir: &'a mut [f64],
    /// `K_{m+1}`.
    k: &'a mut [f64],
    /// `1/K_{m+1}`.
    rk: &'a mut [f64],
    /// `w_j = ℓ_j z_j²`.
    wq: &'a mut [f64],
    /// `ℓ_j`.
    ell: &'a mut [f64],
    /// `z̄_j = ∂q/∂z_j = 2 ℓ_j z_j`.
    zbar: &'a mut [f64],
    /// `∂q/∂ρ_{m+1}` (the group norms).
    kbar: &'a mut [f64],
    /// `w̄ = ∂q/∂w`.
    wbar: &'a mut [f64],
    /// `∇_η q`.
    gq: &'a mut [f64],
    /// Scratch rows of length `N`.
    s_row: &'a mut [f64],
    t_row: &'a mut [f64],
    /// The gradient of `q` in `θ̃`.
    gt: &'a mut [f64],
    /// One group's diagonal block during the polar steps (`(d−1)²`).
    blk: &'a mut [f64],
    /// Tangents `Z̃` (`(d−1) × N`, row-major; columns in the slots of `η`).
    zt: &'a mut [f64],
    /// `Z̃ᵀ` (`N × (d−1)`).
    ztt: &'a mut [f64],
    /// Per rotation, forward order: outputs `(z'_a, z'_{a+1})`, `(c, s)`
    /// and `1/ρ` (0 for a zero norm).
    tape: &'a mut [[f64; 5]],
    /// Per rotation: the adjoints `(b_a, b_{a+1})` of its outputs, `ω/ρ`
    /// (`ω = ∂q/∂ψ`) and the adjoint of its norm (0 for a zero norm).
    rec: &'a mut [[f64; 4]],
}

impl<'a> Work<'a> {
    /// Splits heap buffers (fallback for `d > 8`).
    fn new(
        v: &'a mut [f64],
        nv: &'a mut [f64],
        blk: &'a mut [f64],
        zt: &'a mut [f64],
        ztt: &'a mut [f64],
        tape: &'a mut [[f64; 5]],
        rec: &'a mut [[f64; 4]],
    ) -> Self {
        let d = v.len() / NV;
        let n = nv.len() / NN;
        let mut it = v.chunks_exact_mut(d);
        let mut next = || it.next().expect("buffer of length NV·d");
        let (z, z0, sv, dv, ir, k) = (next(), next(), next(), next(), next(), next());
        let (rk, wq, ell, zbar, kbar, wbar) = (next(), next(), next(), next(), next(), next());
        let mut itn = nv.chunks_exact_mut(n);
        let mut nextn = || itn.next().expect("buffer of length NN·N");
        let (gq, s_row, t_row, gt) = (nextn(), nextn(), nextn(), nextn());
        Work {
            z,
            z0,
            sv,
            dv,
            ir,
            k,
            rk,
            wq,
            ell,
            zbar,
            kbar,
            wbar,
            gq,
            s_row,
            t_row,
            gt,
            blk,
            zt,
            ztt,
            tape,
            rec,
        }
    }
}

/// Offset in `eta` of γ group `gr` (1-based; `gr + 1` entries).
#[inline(always)]
const fn group_offset(d: usize, gr: usize) -> usize {
    d + gr * (gr + 1) / 2 - 1
}

/// Index of the first rotation of group `gr` in forward order (groups run
/// `d − 2, …, 1`, group `gr` has `gr` rotations).
#[inline(always)]
const fn rotation_base(d: usize, gr: usize) -> usize {
    (d - 2) * (d - 1) / 2 - gr * (gr + 1) / 2
}

#[inline(always)]
fn hess_core(
    mu: &[f64],
    gamma: &[f64],
    y: &[f64],
    grad: &mut [f64],
    hess: &mut [f64],
    w: Work<'_>,
) -> f64 {
    let d = mu.len();
    let p = d - 1;
    let ng = d - 2;
    let nn = d + gamma.len();
    let Work {
        z,
        z0,
        sv,
        dv,
        ir,
        k,
        rk,
        wq,
        ell,
        zbar,
        kbar,
        wbar,
        gq,
        s_row,
        t_row,
        gt,
        blk,
        zt,
        ztt,
        tape,
        rec,
    } = w;
    let nr = (d - 2) * (d - 1) / 2;
    let (z, z0, sv, dv, ir, k) = (
        &mut z[..d],
        &mut z0[..d],
        &mut sv[..d],
        &mut dv[..d],
        &mut ir[..d],
        &mut k[..d],
    );
    let (rk, wq, ell, zbar, kbar, wbar) = (
        &mut rk[..d],
        &mut wq[..d],
        &mut ell[..d],
        &mut zbar[..d],
        &mut kbar[..d],
        &mut wbar[..d],
    );
    let (gq, s_row, t_row, gt) = (
        &mut gq[..nn],
        &mut s_row[..nn],
        &mut t_row[..nn],
        &mut gt[..nn],
    );
    let zt = &mut zt[..(d - 1) * nn];
    let ztt = &mut ztt[..(d - 1) * nn];
    let (tape, rec) = (&mut tape[..nr], &mut rec[..nr]);

    // =====================================================================
    // Forward pass: the operations of `adjoint::adjoint_core`, so value and
    // gradient are bit-identical to it.
    // =====================================================================
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

    let mut prod = 1.0;
    for i in 1..=ng {
        let e = (d - 1 - i) as i32;
        prod *= if e == 1 { k[i - 1] } else { k[i - 1].powi(e) };
    }
    let mut inv_l = crate::loglik::root_d_minus_1(prod, d);
    let mut q = t * t;
    let mut wsum = 0.0;
    for j in 0..d - 1 {
        let wj = z[j] * z[j] * inv_l;
        q += wj;
        wq[j] = wj;
        wsum += wj;
        ell[j] = inv_l;
        z[j] = 2.0 * z[j] * inv_l;
        zbar[j] = z[j];
        if j < ng {
            inv_l /= k[j];
        }
    }

    let sq = q.sqrt();
    let alpha = ymu / sq;
    let (inv_q, inv_sq) = (1.0 / q, 1.0 / sq);
    let (lmp, dlmp, d2lmp) = log_m_p_derivs(alpha, p);
    let ll =
        -0.5 * p as f64 * LN_2PI - q.ln() * (0.5 * d as f64) + (alpha * alpha - mm) * 0.5 + lmp;

    // =====================================================================
    // Reverse sweep for ∇q (as in the adjoint), recording the adjoints of
    // every rotation's outputs, ω/ρ and the adjoint of its norm.
    // =====================================================================
    let (gq_mu, gq_gamma) = gq.split_at_mut(d);
    let inv_p = 1.0 / p as f64;
    let mut head = 0.0;
    for gr in 1..=ng {
        let m = gr - 1;
        head += wq[m];
        let mut carry = ((d - 2 - m) as f64 * inv_p * wsum - (wsum - head)) * rk[m];
        kbar[m] = carry;
        let off = gr * (gr + 1) / 2 - 1;
        let gb = &mut gq_gamma[off..off + gr + 1];
        for jj in 0..gr - 1 {
            r -= 1;
            let a = gr - jj - 1;
            let (ba, bb) = (z[a], z[a + 1]);
            let (om, c, sn, rb) = rotate_t_adjoint(z, a, &tape[r], carry);
            rec[r] = [ba, bb, om, rb];
            gb[jj] = c * rb - sn * om;
            carry = sn * rb + c * om;
        }
        r -= 1;
        let (ba, bb) = (z[0], z[1]);
        let (om, c, sn, rb) = rotate_t_adjoint(z, 0, &tape[r], carry);
        rec[r] = [ba, bb, om, rb];
        gb[gr - 1] = c * rb - sn * om;
        gb[gr] = sn * rb + c * om;
    }
    wbar[..d - 1].copy_from_slice(&z[..d - 1]);

    let mut sb = -t * t * irm * irm;
    let mut db = 2.0 * t * irm;
    for i0 in (2..d).rev() {
        let mi = mu[i0];
        let mut gmi = 2.0 * mi * sb + y[i0] * db;
        let c = i0 - 1;
        let numb = z[c] * ir[c];
        let pb = -0.5 * numb * z0[c] * ir[c];
        let (si, di) = (sv[i0], dv[i0]);
        gmi += numb * di + 2.0 * pb * mi * si;
        db += numb * mi;
        sb += pb * (mi * mi + 2.0 * si) - numb * y[i0];
        gq_mu[i0] = gmi;
    }
    let numb = z[0] * ir[0];
    let n2b = sb - 0.5 * numb * z0[0] * ir[0];
    gq_mu[0] = numb * y[1] + 2.0 * mu[0] * n2b + y[0] * db;
    gq_mu[1] = 2.0 * mu[1] * n2b + y[1] * db - numb * y[0];

    let ab = alpha + dlmp;
    let qb = -0.5 * (d as f64 + ab * alpha) * inv_q;
    let ymub = ab * inv_sq;
    {
        let (grad_mu, grad_gamma) = grad.split_at_mut(d);
        for i in 0..d {
            grad_mu[i] = qb * gq_mu[i] + ymub * y[i] - mu[i];
        }
        for (g, &v) in grad_gamma.iter_mut().zip(gq_gamma.iter()) {
            *g = v * qb;
        }
    }

    // =====================================================================
    // Hessian in θ̃ = (mu, ψ, ρ). Slots: mu_i at i; in group gr (offset off,
    // n = gr + 1 entries) ψ of the rotation whose u is G_i at off + i
    // (i ≤ n − 2), and ρ_gr at off + n − 1. Every contribution to the
    // Hessian of q is added once, to whichever triangle is contiguous for
    // it; the assembly pass adds the two triangles and applies the outer
    // layer.
    // =====================================================================
    hess.fill(0.0);
    zt.fill(0.0);
    projections(mu, y, sv, dv, ir, z0, wbar, zt, hess, nn, s_row, t_row);
    if irm != 0.0 {
        t_squared(mu, y, t, irm, hess, nn, s_row);
    }

    // θ̃-gradient of q: mu, ω = ∂q/∂ψ of every rotation, ∂q/∂ρ of every group
    gt[..d].copy_from_slice(gq_mu);
    let mut r = 0;
    for gr in (1..=ng).rev() {
        let off = group_offset(d, gr);
        for kk in 0..gr {
            let [xo, yo, ..] = tape[r];
            let [ba, bb, ..] = rec[r];
            gt[off + gr - 1 - kk] = ba * yo - bb * xo;
            r += 1;
        }
        gt[off + gr] = kbar[gr - 1];
    }

    // ---- forward sweep of the tangents through the rotations: at rotation
    // r, the row b_rᵀ Ĵ_r ∂f_r/∂(·) of ψ_r over the active columns
    let mut r = 0;
    for gr in (1..=ng).rev() {
        let n = gr + 1;
        let off = group_offset(d, gr);
        for kk in 0..gr {
            let [xo, yo, c, sn, _] = tape[r];
            let [ba, bb, ..] = rec[r];
            let slot = off + n - 2 - kk;
            let (lo, hi) = zt.split_at_mut((kk + 1) * nn);
            let ra = &mut lo[kk * nn..];
            let rb = &mut hi[..nn];
            rotate_rows(&mut ra[..d], &mut rb[..d], c, sn);
            rotate_rows(&mut ra[slot + 1..], &mut rb[slot + 1..], c, sn);
            ra[slot] = yo;
            rb[slot] = -xo;
            let hrow = &mut hess[slot * nn..(slot + 1) * nn];
            for ((h, &x), &yv) in hrow[..d].iter_mut().zip(&ra[..d]).zip(&rb[..d]) {
                *h += ba * yv - bb * x;
            }
            for ((h, &x), &yv) in hrow[slot..].iter_mut().zip(&ra[slot..]).zip(&rb[slot..]) {
                *h += ba * yv - bb * x;
            }
            r += 1;
        }
    }

    // ---- ρ rows: ∂²q/∂ρ_m∂x = (a_m S_x − T_{m,x})/K_m with
    // S_x = Σ_j z̄_j Z̃_jx, T_{m,x} = Σ_{j≥m} z̄_j Z̃_jx, a_m = (d−1−m)/(d−1)
    s_row.fill(0.0);
    for j in 0..d - 1 {
        let zb = zbar[j];
        for (sx, &v) in s_row.iter_mut().zip(&zt[j * nn..(j + 1) * nn]) {
            *sx += zb * v;
        }
    }
    t_row.fill(0.0);
    for m in (1..=ng).rev() {
        let zb = zbar[m];
        for (tx, &v) in t_row.iter_mut().zip(&zt[m * nn..(m + 1) * nn]) {
            *tx += zb * v;
        }
        let am = (d - 1 - m) as f64 * inv_p;
        let rkm = rk[m - 1];
        let slot = group_offset(d, m) + m;
        let hrow = &mut hess[slot * nn..(slot + 1) * nn];
        for ((h, &sx), &tx) in hrow.iter_mut().zip(s_row.iter()).zip(t_row.iter()) {
            *h += (am * sx - tx) * rkm;
        }
    }
    // ρρ block: Σ_j w_j (c_jm c_jm' − δ c_jm)/(K_m K_m'), with the suffix
    // sums U_m = Σ_{j≥m} w_j (in s_row)
    let mut u = 0.0;
    for j in (1..d - 1).rev() {
        u += wq[j];
        s_row[j] = u;
    }
    for m in 1..=ng {
        let am = (d - 1 - m) as f64 * inv_p;
        let um = s_row[m];
        let sm = group_offset(d, m) + m;
        for m2 in m..=ng {
            let am2 = (d - 1 - m2) as f64 * inv_p;
            let um2 = s_row[m2];
            let mut v = am * am2 * wsum - am * um2 - am2 * um + um2;
            if m2 == m {
                v -= am * wsum - um;
            }
            hess[sm * nn + group_offset(d, m2) + m2] += v * rk[m - 1] * rk[m2 - 1];
        }
    }

    // ---- assembly: the two triangles, 2 Z̃ᵀ diag(ℓ) Z̃ (dot products of
    // the columns of Z̃, transposed into `ztt`), and the outer layer:
    // ∇²ll = q̄ ∇²q + F_qq g gᵀ + F_qy (g e_yᵀ + e_y gᵀ) + F_yy e_y e_yᵀ − I_mu
    // with g = ∇q and e_y = ∇(y·mu) = (y, 0)
    let p1 = d - 1;
    for j in 0..p1 {
        for x in 0..nn {
            ztt[x * p1 + j] = zt[j * nn + x];
        }
    }
    let g2 = 1.0 + d2lmp;
    let fqq = (2.0 * d as f64 + g2 * alpha * alpha + 3.0 * ab * alpha) * 0.25 * inv_q * inv_q;
    let fqy = -(g2 * alpha + ab) * 0.5 * inv_q * inv_sq;
    let fyy = g2 * inv_q;
    for a in 0..nn {
        let ga = gt[a];
        let ea = if a < d { y[a] } else { 0.0 };
        let ua = fqq * ga + fqy * ea;
        let va = fqy * ga + fyy * ea;
        let za = &ztt[a * p1..(a + 1) * p1];
        for (w, (&l, &z)) in s_row[..p1].iter_mut().zip(ell.iter().zip(za)) {
            *w = 2.0 * l * z;
        }
        let wa = &s_row[..p1];
        let gn: f64 = wa.iter().zip(za).map(|(w, z)| w * z).sum();
        let diag = &mut hess[a * nn + a];
        *diag = qb * (*diag + gn) + ga * ua + ea * va - if a < d { 1.0 } else { 0.0 };
        // e_y is zero on the γ slots
        for b in a + 1..d.max(a + 1) {
            let zb = &ztt[b * p1..(b + 1) * p1];
            let gn: f64 = wa.iter().zip(zb).map(|(w, z)| w * z).sum();
            let v = qb * (hess[a * nn + b] + hess[b * nn + a] + gn) + ua * gt[b] + va * y[b];
            hess[a * nn + b] = v;
            hess[b * nn + a] = v;
        }
        for b in d.max(a + 1)..nn {
            let zb = &ztt[b * p1..(b + 1) * p1];
            let gn: f64 = wa.iter().zip(zb).map(|(w, z)| w * z).sum();
            let v = qb * (hess[a * nn + b] + hess[b * nn + a] + gn) + ua * gt[b];
            hess[a * nn + b] = v;
            hess[b * nn + a] = v;
        }
    }

    // =====================================================================
    // Back to γ: the polar steps of the reverse sweep, group by group. The
    // rows of a group are transformed (contiguous), then copied into its
    // columns; its own block is redone step by step with the second-order
    // terms (ψ̄ ∇²ψ + ρ̄ ∇²ρ, with ψ̄ = q̄ ω and ρ̄ = q̄ ∂q/∂ρ).
    // =====================================================================
    for gr in 1..=ng {
        let n = gr + 1;
        let off = group_offset(d, gr);
        let base = rotation_base(d, gr);
        let blk = &mut blk[..n * n];
        for i in 0..n {
            for j in 0..n {
                blk[i * n + j] = hess[(off + i) * nn + off + j];
            }
        }
        for i in 0..n - 1 {
            let [_, _, c, sn, inv] = tape[base + (n - 2 - i)];
            let [_, _, om, rb] = rec[base + (n - 2 - i)];
            let mask = if inv == 0.0 { 0.0 } else { 1.0 };
            // T: rows (ψ, ρ), columns (u, v)
            let t = [-sn * inv, c * inv, c * mask, sn * mask];
            let (o2, r2) = (qb * om * inv, qb * rb * inv);
            let sm = [
                o2 * 2.0 * c * sn + r2 * sn * sn,
                o2 * (sn * sn - c * c) - r2 * c * sn,
                -o2 * 2.0 * c * sn + r2 * c * c,
            ];
            polar_step(blk, n, i, n - 1, t, sm);
            // the group's rows, every column
            let (lo, hi) = hess.split_at_mut((off + n - 1) * nn);
            let ra = &mut lo[(off + i) * nn..(off + i + 1) * nn];
            let rv = &mut hi[..nn];
            let [tpu, tpv, tru, trv] = t;
            for (x, yv) in ra.iter_mut().zip(rv.iter_mut()) {
                let (hp, hr) = (*x, *yv);
                *x = tpu * hp + tru * hr;
                *yv = tpv * hp + trv * hr;
            }
        }
        for i in 0..n {
            let row = (off + i) * nn;
            for x in 0..off {
                hess[x * nn + off + i] = hess[row + x];
            }
            for x in off + n..nn {
                hess[x * nn + off + i] = hess[row + x];
            }
            for j in 0..n {
                hess[row + off + j] = blk[i * n + j];
            }
        }
    }

    // =====================================================================
    // Exact zero tails (see the module documentation).
    // =====================================================================
    for gr in 2..=ng {
        let n = gr + 1;
        let off = group_offset(d, gr);
        let g = &gamma[off - d..off - d + n];
        let mut tz = n;
        while tz > 0 && g[tz - 1] == 0.0 {
            tz -= 1;
        }
        if tz == 0 || tz + 1 >= n {
            continue;
        }
        let base = rotation_base(d, gr);
        // rotation of the group whose u is G_jj (its norm ρ = ‖G_{jj..}‖)
        let rot = |jj: usize| base + (n - 2 - jj);
        let mut corr = 0.0;
        for j in 0..tz {
            let [xo, yo, _, _, inv] = tape[rot(j)];
            let [ba, bb, ..] = rec[rot(j)];
            let mut part = -(ba * xo + bb * yo) * inv;
            if j >= 1 {
                // ‖G_{j..}‖ is v of the rotation whose u is G_{j−1}
                let [xo2, yo2, c2, s2, inv2] = tape[rot(j - 1)];
                let [ba2, bb2, ..] = rec[rot(j - 1)];
                let radial = ba2 * xo2 + bb2 * yo2;
                let angular = ba2 * yo2 - bb2 * xo2;
                part += (s2 * radial + c2 * angular) * inv2;
            } else {
                part += kbar[gr - 1];
            }
            corr += part * inv;
        }
        for i in tz..n {
            hess[(off + i) * nn + off + i] += qb * corr;
        }
    }
    ll
}

/// `(x, y) ← (c x + s y, c y − s x)` elementwise: one transposed plane
/// rotation applied to rows `a` and `a + 1` of the tangents.
#[inline(always)]
fn rotate_rows(ra: &mut [f64], rb: &mut [f64], c: f64, s: f64) {
    for (x, yv) in ra.iter_mut().zip(rb.iter_mut()) {
        let (a, b) = (*x, *yv);
        *x = c * a + s * b;
        *yv = c * b - s * a;
    }
}

/// The projections `w_c` (rows `c` of `zt`, the `mu` columns) and the
/// curvature `Σ_c w̄_c ∇²w_c` (upper triangle of the `mu` block). With
/// `w = N/√P`: `∇w = ir (∇N − ½ w ir ∇P)` and
/// `∇²w = ir ∇²N − ½ ir³ (∇N∇Pᵀ + ∇P∇Nᵀ) − ½ w ir² ∇²P + ¾ w ir⁴ ∇P∇Pᵀ`.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn projections(
    mu: &[f64],
    y: &[f64],
    sv: &[f64],
    dv: &[f64],
    ir: &[f64],
    z0: &[f64],
    wbar: &[f64],
    zt: &mut [f64],
    hess: &mut [f64],
    nn: usize,
    gn: &mut [f64],
    gp: &mut [f64],
) {
    let d = mu.len();
    if ir[0] != 0.0 {
        // N = mu_0 y_1 − mu_1 y_0, P = mu_0² + mu_1²:
        // ∇²w = −ir³(∇N μᵀ + μ∇Nᵀ) − w ir² I + 3 w ir⁴ μμᵀ on (mu_0, mu_1)
        let (irc, wc, wb) = (ir[0], z0[0], wbar[0]);
        let g0 = [y[1], -y[0]];
        zt[0] = irc * (g0[0] - wc * irc * mu[0]);
        zt[1] = irc * (g0[1] - wc * irc * mu[1]);
        let irc2 = irc * irc;
        let (c1, c2, c3) = (
            -wb * irc2 * irc,
            -wb * wc * irc2,
            3.0 * wb * wc * irc2 * irc2,
        );
        for a in 0..2 {
            for b in a..2 {
                let mut v = c1 * (g0[a] * mu[b] + mu[a] * g0[b]) + c3 * mu[a] * mu[b];
                if a == b {
                    v += c2;
                }
                hess[a * nn + b] += v;
            }
        }
    }
    for i in 2..d {
        let c = i - 1;
        if ir[c] == 0.0 {
            continue;
        }
        // N = m dot − s y_i, P = s (m² + s) over mu_0..mu_i
        let (s, dt, m) = (sv[i], dv[i], mu[i]);
        let (irc, wc, wb) = (ir[c], z0[c], wbar[c]);
        let pm = m * m + 2.0 * s;
        for kk in 0..i {
            gn[kk] = m * y[kk] - 2.0 * mu[kk] * y[i];
            gp[kk] = 2.0 * mu[kk] * pm;
        }
        gn[i] = dt;
        gp[i] = 2.0 * s * m;
        let half = 0.5 * wc * irc;
        let row = &mut zt[c * nn..c * nn + i + 1];
        for ((z, &a), &b) in row.iter_mut().zip(&gn[..=i]).zip(&gp[..=i]) {
            *z = irc * (a - half * b);
        }
        let irc2 = irc * irc;
        let ca = wb * irc;
        let cb = -0.5 * wb * irc2 * irc;
        let cc = -0.5 * wb * wc * irc2;
        let ce = 0.75 * wb * wc * irc2 * irc2;
        for a in 0..=i {
            // cb (∇N_a ∇P_b + ∇P_a ∇N_b) + ce ∇P_a ∇P_b = x1 ∇N_b + x2 ∇P_b
            let x1 = cb * gp[a];
            let x2 = cb * gn[a] + ce * gp[a];
            let hrow = &mut hess[a * nn..a * nn + i + 1];
            for ((h, &nb), &pb) in hrow[a..].iter_mut().zip(&gn[a..=i]).zip(&gp[a..=i]) {
                *h += x1 * nb + x2 * pb;
            }
            if a < i {
                // ∇²N = −2 y_i I and ∇²P = 2 pm I + 8 μμᵀ on mu_0..mu_{i−1};
                // ∇²N = y_a, ∇²P = 4 μ_a m against mu_i
                let x3 = 8.0 * cc * mu[a];
                for (h, &mb) in hrow[a..i].iter_mut().zip(&mu[a..i]) {
                    *h += x3 * mb;
                }
                hrow[a] += -2.0 * ca * y[i] + 2.0 * cc * pm;
                hrow[i] += ca * y[a] + 4.0 * cc * mu[a] * m;
            } else {
                // ∇²P = 2 s against mu_i
                hrow[i] += 2.0 * cc * s;
            }
        }
    }
}

/// `∇²(t²) = 2∇t∇tᵀ + 2t∇²t` for `t = y·mu/‖mu‖` (upper triangle of the
/// `mu` block): `∇t = irm (y − t irm mu)`,
/// `∇²t = −irm³ (y muᵀ + mu yᵀ) − t irm² I + 3 t irm⁴ mu muᵀ`.
#[inline(always)]
fn t_squared(mu: &[f64], y: &[f64], t: f64, irm: f64, hess: &mut [f64], nn: usize, gt: &mut [f64]) {
    let d = mu.len();
    let irm2 = irm * irm;
    for k in 0..d {
        gt[k] = irm * (y[k] - t * irm * mu[k]);
    }
    // 2t∇²t = c1 (y muᵀ + mu yᵀ) + c2 I + c3 mu muᵀ
    let (c1, c2, c3) = (
        -2.0 * t * irm2 * irm,
        -2.0 * t * t * irm2,
        6.0 * t * t * irm2 * irm2,
    );
    for a in 0..d {
        let (x1, x2, x3) = (2.0 * gt[a], c1 * y[a] + c3 * mu[a], c1 * mu[a]);
        let hrow = &mut hess[a * nn..a * nn + d];
        for (((h, &gb), &mb), &yb) in hrow[a..]
            .iter_mut()
            .zip(&gt[a..d])
            .zip(&mu[a..])
            .zip(&y[a..])
        {
            *h += x1 * gb + x2 * mb + x3 * yb;
        }
        hrow[a] += c2;
    }
}

/// One polar step on slots `a` (ψ → u) and `b` (ρ → v) of the symmetric
/// `n × n` matrix `h`: `h ← Tᵀ h T + S` with `T = [[tpu, tpv], [tru, trv]]`
/// (rows ψ, ρ; columns u, v) and `S = [[suu, suv], [suv, svv]]`.
#[inline(always)]
fn polar_step(h: &mut [f64], n: usize, a: usize, b: usize, t: [f64; 4], sm: [f64; 3]) {
    let [tpu, tpv, tru, trv] = t;
    let (haa, hab, hbb) = (h[a * n + a], h[a * n + b], h[b * n + b]);
    for j in 0..n {
        if j == a || j == b {
            continue;
        }
        let (x, yv) = (h[a * n + j], h[b * n + j]);
        let nu = tpu * x + tru * yv;
        let nv = tpv * x + trv * yv;
        h[a * n + j] = nu;
        h[b * n + j] = nv;
        h[j * n + a] = nu;
        h[j * n + b] = nv;
    }
    let [suu, suv, svv] = sm;
    let huu = tpu * tpu * haa + 2.0 * tpu * tru * hab + tru * tru * hbb + suu;
    let huv = tpu * tpv * haa + (tpu * trv + tru * tpv) * hab + tru * trv * hbb + suv;
    let hvv = tpv * tpv * haa + 2.0 * tpv * trv * hab + trv * trv * hbb + svv;
    h[a * n + a] = huu;
    h[a * n + b] = huv;
    h[b * n + a] = huv;
    h[b * n + b] = hvv;
}

/// `rotate_t_record` of [`crate::adjoint`].
#[inline(always)]
fn rotate_t_record(z: &mut [f64], a: usize, c: f64, s: f64, inv: f64) -> [f64; 5] {
    let (x, y) = (z[a], z[a + 1]);
    let (xo, yo) = (c * x + s * y, c * y - s * x);
    z[a] = xo;
    z[a + 1] = yo;
    [xo, yo, c, s, inv]
}

/// `rotate_t_adjoint` of [`crate::adjoint`].
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
