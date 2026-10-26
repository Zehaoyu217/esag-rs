//! Reverse-mode (adjoint) gradient of the `γ`-rot log-density of
//! [`crate::gamma_rot`].
//!
//! [`loglik_gr_grad_rev`] returns the value of
//! [`loglik_gr`](crate::gamma_rot::loglik_gr) (its forward pass performs the
//! same floating-point operations in the same order) and the exact gradient
//! in `eta = (μ, γ)` from one forward pass that records what the reverse
//! sweep needs and one reverse sweep of about the same length: `O(d²)` per
//! observation, no dual numbers. The forward-mode
//! [`loglik_gr_grad`](crate::gamma_rot::loglik_gr_grad) carries
//! `N = (d² + d − 2)/2` tangents through every operation and costs about
//! `N` times the value.
//!
//! # The sweep
//!
//! The forward pass is `gamma_rot::quad_form` followed by the density: the
//! frame `z = Eᵀ(y − κ v)`, `v = μ̂ + m`, `κ = 2(v·y)/(v·v)`, `t = μ̂·y`
//! (`gamma_smooth::project`); the basis turn `z ← R_0ᵀ z`; for the groups
//! `gr = d−2, …, 2` the transposed rotation `z ← Q_grᵀ z` of the block
//! `z_0..z_gr` — `s² = ‖ω‖²`, `(a, b)` the link coefficients,
//! `f = a z_gr + b (ω·z)`, `z_i ← z_i − f ω_i`,
//! `z_gr ← z_gr + a (ω·z) − b s² z_gr` — with `S_gr = Π_{i≥gr} e^{κ_i/(i+1)}`
//! and the term `T_gr = S_gr e^{−κ_gr} z_gr²`; Paine's block
//! `P = S_2 (√(1+‖G_1‖²)(z_0² + z_1²) − G_{1,1}(z_0² − z_1²) − 2 G_{1,2} z_0 z_1)`;
//! `q = Σ T_gr + P`, `Q = t² + q`, `α = (μ·y)/√Q` and the density. The tape
//! holds, per group, the block `z_0..z_{gr−1}` before its rotation and the
//! scalars `s², a, b, z_gr, ω·z, S_gr e^{−κ_gr}, T_gr`.
//!
//! The reverse sweep seeds `Q̄ = 1` and walks the groups upwards:
//!
//! * `∂q/∂κ_gr = (P + Σ_{g≤gr} T_g)/(gr+1) − T_gr`, since `κ_gr` scales
//!   every term of a lower or equal group through `S_g` and its own term
//!   through `e^{−κ_gr}`;
//! * Paine's block gives `z̄_0, z̄_1` and `∂q/∂G_1`; group `gr` adds
//!   `z̄_gr = 2 S_gr e^{−κ_gr} z_gr` and pulls `z̄_0..z̄_gr` back through its
//!   rotation. With `x` the recorded block, `x_b = z_gr` before the
//!   rotation, `δ = ω·x`, and `z̄_b` the adjoint of the rotated `z_gr`:
//!   `f̄ = −ω·z̄`, `x̄_b = (1 − b s²) z̄_b + a f̄`, `δ̄ = a z̄_b + b f̄`,
//!   `ā = δ z̄_b + x_b f̄`, `b̄ = −s² x_b z̄_b + δ f̄`,
//!   `s̄² = −b x_b z̄_b + ā a′(s²) + b̄ b′(s²)`,
//!   `ω̄ = −f z̄ + δ̄ x + 2 s̄² ω`, `x̄ = z̄ + δ̄ ω`. The link derivatives are
//!   `a′ = −a/(4+s²)`, `b′ = −b/(4+s²)` for `Stereo`, and
//!   `a′ = (cos s − a)/(2s²)`, `b′ = (a/2 − b)/s²` for `Exp`, with the
//!   constants `−1/6, −1/24` of its series branch below `s² = 10⁻¹⁰`;
//! * then `z̄ ← R_0 z̄` and the frame's sweep of
//!   [`crate::gamma_transported`]: `ū = E z̄`,
//!   `v̄ = −κ ū − (2(ū·v)/(v·v)) u`, `μ̂̄ = v̄ + 2t y`,
//!   `∂Q/∂μ = (μ̂̄ − (μ̂·μ̂̄) μ̂)/‖μ‖`; zero at `μ = 0`, and `v̄ = 0` at the
//!   antipode `μ̂ = −m`, where the frames are constant;
//! * finally `∇ll = Q̄ ∇Q + ȳmu y − μ`, `Q̄ = −(d + ᾱ α)/(2Q)`,
//!   `ȳmu = ᾱ/√Q`, `ᾱ = α + (log M_{d−1})′(α)`.
//!
//! The guard conventions are those of the forward kernel (the branches at
//! `μ = 0`, `μ̂ = −m` and `s² < 10⁻¹⁰` have the derivatives of the
//! expressions they select), so the gradient agrees with the forward-mode
//! one everywhere, to rounding (tested at `10⁻⁹` absolute plus relative).
//!
//! # Cost
//!
//! Nanoseconds per observation for the log-density and its gradient
//! (`bench::kernel_cost`, 20 000 random observations, best of 5, Apple M5
//! Pro, `--release`), against the forward-mode `γ`-rot kernel
//! ([`loglik_gr_grad`](crate::gamma_rot::loglik_gr_grad)), the log-tangent
//! adjoint kernel ([`loglik_lt_grad`](crate::logtangent::loglik_lt_grad))
//! and the plain `γ` adjoint kernel
//! ([`loglik_eta_grad`](crate::loglik::loglik_eta_grad)):
//!
//! | `d` | `N` | `γ`-rot reverse | `γ`-rot forward | log-tangent | `γ` |
//! |---|---|---|---|---|---|
//! | 3  | 5  | 49  | 81   | 65   | 39  |
//! | 5  | 14 | 73  | 294  | 335  | 70  |
//! | 8  | 35 | 202 | 1881 | 1218 | 160 |
//! | 10 | 54 | 272 | 3927 | 2826 | 270 |
//!
//! The reverse kernel costs about what `γ`'s adjoint does, 1.7× to 14× less
//! than forward mode.
//!
//! No allocation happens for `d ≤ 10` (const-generic stack buffers);
//! larger `d` use heap buffers.

use crate::dims;
use crate::gamma_rot::{GrFrame, RotLink};
use crate::loglik::{cdf_fast, log_m_p_derivs};
use crate::stats::normal::{LN_2PI, norm_pdf};

/// Log-density at the unit vector `y` for stacked `eta = (μ, γ)` in the
/// `γ`-rot parameterization `fr`, and its exact gradient in `eta`, written
/// into `grad`, by one forward and one reverse pass.
///
/// The value is that of [`loglik_gr`](crate::gamma_rot::loglik_gr); the
/// gradient agrees with the forward-mode
/// [`loglik_gr_grad`](crate::gamma_rot::loglik_gr_grad) to rounding,
/// including at the guarded points (see the [module documentation](self)),
/// and is exact for every `d ≥ 3`.
///
/// # Panics
/// If `d = y.len() < 3`, the frame has another dimension, `eta.len()` is
/// not [`dims::eta_len`]`(d)`, or `grad.len() != eta.len()`.
pub fn loglik_gr_grad_rev(fr: &GrFrame, eta: &[f64], y: &[f64], grad: &mut [f64]) -> f64 {
    let d = y.len();
    assert!(d >= 3, "the ESAG density needs d >= 3, got d = {d}");
    assert_eq!(fr.dim(), d, "frame dimension {} but d = {d}", fr.dim());
    assert_eq!(eta.len(), dims::eta_len(d), "eta has length {}", eta.len());
    assert_eq!(grad.len(), eta.len(), "grad has length {}", grad.len());
    match d {
        3 => fixed::<3, 1>(fr, eta, y, grad),
        4 => fixed::<4, 3>(fr, eta, y, grad),
        5 => fixed::<5, 6>(fr, eta, y, grad),
        6 => fixed::<6, 10>(fr, eta, y, grad),
        7 => fixed::<7, 15>(fr, eta, y, grad),
        8 => fixed::<8, 21>(fr, eta, y, grad),
        9 => fixed::<9, 28>(fr, eta, y, grad),
        10 => fixed::<10, 36>(fr, eta, y, grad),
        _ => {
            let (mu, gamma) = eta.split_at(d);
            let mut v = vec![0.0; NV * d];
            let mut tz = vec![0.0; tape_len(d)];
            let mut steps = vec![Step::default(); d];
            core(
                fr,
                mu,
                gamma,
                y,
                grad,
                Work::new(&mut v, &mut tz, &mut steps),
            )
        }
    }
}

/// Log-density, exact gradient and the Hessian in `eta` by central
/// differences of [`loglik_gr_grad_rev`] (step `1e−5·max(1, |eta_i|)`,
/// symmetrised), the rule of
/// [`loglik_gr_hess_fd`](crate::gamma_rot::loglik_gr_hess_fd).
///
/// # Panics
/// As [`loglik_gr_grad_rev`], or if `hess.len() != eta.len()²`.
pub fn loglik_gr_hess_fd_rev(
    fr: &GrFrame,
    eta: &[f64],
    y: &[f64],
    grad: &mut [f64],
    hess: &mut [f64],
) -> f64 {
    let n = eta.len();
    assert_eq!(hess.len(), n * n, "hess must have length eta.len()²");
    let f = loglik_gr_grad_rev(fr, eta, y, grad);
    let mut e = eta.to_vec();
    let (mut gp, mut gm) = (vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        let h = 1e-5 * eta[i].abs().max(1.0);
        e[i] = eta[i] + h;
        loglik_gr_grad_rev(fr, &e, y, &mut gp);
        e[i] = eta[i] - h;
        loglik_gr_grad_rev(fr, &e, y, &mut gm);
        e[i] = eta[i];
        for j in 0..n {
            hess[i * n + j] = (gp[j] - gm[j]) / (2.0 * h);
        }
    }
    for i in 0..n {
        for j in 0..i {
            let s = 0.5 * (hess[i * n + j] + hess[j * n + i]);
            hess[i * n + j] = s;
            hess[j * n + i] = s;
        }
    }
    f
}

/// Number of length-`d` scratch vectors in [`Work`].
const NV: usize = 5;

/// Length of the tape of pre-rotation blocks: `Σ_{gr=2}^{d−2} gr`, rounded
/// up to `(d−2)(d−1)/2` (which is `≥ 1`, so `d = 3` has a tape too).
#[inline(always)]
const fn tape_len(d: usize) -> usize {
    (d - 2) * (d - 1) / 2
}

/// Offset of group `gr`'s pre-rotation block in the tape.
#[inline(always)]
const fn tape_off(gr: usize) -> usize {
    gr * (gr - 1) / 2 - 1
}

/// What one rotation group records for its reverse pass.
#[derive(Clone, Copy, Default)]
struct Step {
    /// `s² = ‖ω‖²`.
    s2: f64,
    /// Link coefficients `(a, b)` at `s²`.
    a: f64,
    b: f64,
    /// `z_gr` before the rotation.
    xb: f64,
    /// `ω·z_{0..gr}` before the rotation.
    dot: f64,
    /// `S_gr e^{−κ_gr}`, the weight of `z_gr²` in `q`.
    coef: f64,
    /// `T_gr = coef · z_gr²`, the group's term of `q`.
    term: f64,
}

/// Scratch for one evaluation: [`NV`] vectors of length `≥ d`, the tape of
/// pre-rotation blocks and one [`Step`] per group (indexed by `gr`).
struct Work<'a> {
    /// The projections, rotated in place; then `z̄ = ∂Q/∂z`.
    z: &'a mut [f64],
    /// `u = y − κ v`.
    u: &'a mut [f64],
    /// `v = μ̂ + m` (or `E_0` at the antipode).
    v: &'a mut [f64],
    /// `μ̂` (or `m` at `μ = 0`).
    mh: &'a mut [f64],
    /// Scratch for the basis turn, then `ū`, then `μ̂̄`.
    ub: &'a mut [f64],
    /// Pre-rotation blocks, group `gr` at [`tape_off`]`(gr)`.
    tz: &'a mut [f64],
    steps: &'a mut [Step],
}

impl<'a> Work<'a> {
    /// Splits a heap buffer of length `NV·d` (fallback for `d > 10`).
    fn new(v: &'a mut [f64], tz: &'a mut [f64], steps: &'a mut [Step]) -> Self {
        let d = v.len() / NV;
        let mut it = v.chunks_exact_mut(d);
        let mut next = || it.next().expect("buffer of length NV·d");
        Work {
            z: next(),
            u: next(),
            v: next(),
            mh: next(),
            ub: next(),
            tz,
            steps,
        }
    }
}

/// Stack buffers for dimension `D` with a tape of `R = (D−2)(D−1)/2`
/// entries. Out of line so that each dimension gets its own compact
/// function with compile-time trip counts.
#[inline(never)]
fn fixed<const D: usize, const R: usize>(
    fr: &GrFrame,
    eta: &[f64],
    y: &[f64],
    grad: &mut [f64],
) -> f64 {
    const { assert!(D >= 3) };
    debug_assert_eq!(R, tape_len(D));
    let (mu, gamma) = eta.split_at(D);
    let mut v = [[0.0; D]; NV];
    let mut tz = [0.0; R];
    let mut steps = [Step::default(); D];
    let [z, u, vv, mh, ub] = &mut v;
    let w = Work {
        z,
        u,
        v: vv,
        mh,
        ub,
        tz: &mut tz,
        steps: &mut steps,
    };
    core(
        fr,
        &mu[..D],
        &gamma[..dims::gamma_len(D)],
        &y[..D],
        &mut grad[..dims::eta_len(D)],
        w,
    )
}

/// `(a, b)` of `Q = I + a Ω + b Ω²` for `s² = ‖ω‖²`, as
/// `gamma_rot::link_coeffs` computes them in `f64`.
#[inline(always)]
fn link_coeffs(link: RotLink, s2: f64) -> (f64, f64) {
    match link {
        RotLink::Stereo => {
            let den = s2 + 4.0;
            (4.0 / den, 2.0 / den)
        }
        RotLink::Exp => {
            if s2 < 1e-10 {
                (1.0 - s2 * (1.0 / 6.0), 0.5 - s2 * (1.0 / 24.0))
            } else {
                let s = s2.sqrt();
                (s.sin() / s, (1.0 - s.cos()) / s2)
            }
        }
    }
}

/// `(da/ds², db/ds²)` at `s²`, given `(a, b) = link_coeffs(link, s²)`.
#[inline(always)]
fn link_derivs(link: RotLink, s2: f64, a: f64, b: f64) -> (f64, f64) {
    match link {
        RotLink::Stereo => {
            let inv = 1.0 / (s2 + 4.0);
            (-a * inv, -b * inv)
        }
        RotLink::Exp => {
            if s2 < 1e-10 {
                (-1.0 / 6.0, -1.0 / 24.0)
            } else {
                // a = sin s/s: da/ds = (s cos s − sin s)/s², ds/ds² = 1/(2s);
                // b = (1 − cos s)/s²: db/ds² = (s sin s/2 − (1 − cos s))/s⁴
                let s = s2.sqrt();
                let inv = 1.0 / s2;
                ((s.cos() - a) * (0.5 * inv), (0.5 * a - b) * inv)
            }
        }
    }
}

/// Forward pass (recording) and reverse sweep. See the module documentation.
#[inline(always)]
fn core(fr: &GrFrame, mu: &[f64], gamma: &[f64], y: &[f64], grad: &mut [f64], w: Work<'_>) -> f64 {
    let d = mu.len();
    let p = d - 1;
    let lt = fr.lt_frame();
    let m = lt.reference();
    let e = lt.tangent_basis();
    let link = fr.link();
    let r0 = fr.r0_slice();
    let Work {
        z,
        u,
        v,
        mh,
        ub,
        tz,
        steps,
    } = w;
    let (z, u, v, mh, ub) = (
        &mut z[..d],
        &mut u[..d],
        &mut v[..d],
        &mut mh[..d],
        &mut ub[..d],
    );
    let (grad_mu, grad_gamma) = grad.split_at_mut(d);

    // =====================================================================
    // Forward pass: the operations of `gamma_rot::quad_form` / `loglik_core`
    // in their order, so the value is that of `loglik_gr::<f64>`.
    // =====================================================================

    // ---- frame (`gamma_smooth::project`): z = Eᵀ(y − κ v), v = μ̂ + m
    let mut mm = 0.0;
    let mut ymu = 0.0;
    for k in 0..d {
        mm += mu[k] * mu[k];
        ymu += mu[k] * y[k];
    }
    let zero_mu = mm == 0.0;
    let inv = if zero_mu { 1.0 } else { 1.0 / mm.sqrt() };
    for k in 0..d {
        mh[k] = if zero_mu { m[k] } else { mu[k] * inv };
    }
    let t: f64 = if zero_mu {
        (0..d).map(|k| m[k] * y[k]).sum()
    } else {
        ymu * inv
    };
    let mut vv = 0.0;
    let mut vy = 0.0;
    for k in 0..d {
        let vk = mh[k] + m[k];
        v[k] = vk;
        vv += vk * vk;
        vy += vk * y[k];
    }
    let antipode = vv == 0.0;
    let kappa = if antipode {
        v.copy_from_slice(&e[..d]);
        vv = 1.0;
        2.0 * (0..d).map(|k| e[k] * y[k]).sum::<f64>()
    } else {
        vy * 2.0 / vv
    };
    for k in 0..d {
        u[k] = y[k] - kappa * v[k];
    }
    for j in 0..p {
        let col = &e[j * d..(j + 1) * d];
        let mut acc = 0.0;
        for k in 0..d {
            acc += u[k] * col[k];
        }
        z[j] = acc;
    }

    // ---- basis turn: z ← R_0ᵀ z
    if let Some(r0) = r0 {
        for j in 0..p {
            let mut acc = 0.0;
            for i in 0..p {
                acc += z[i] * r0[i * p + j];
            }
            ub[j] = acc;
        }
        z[..p].copy_from_slice(&ub[..p]);
    }

    // ---- rotations, groups d−2, …, 2 (one tape block and one Step each)
    let mut scale = 1.0;
    let mut q = 0.0;
    for gr in (2..=d - 2).rev() {
        let off = gr * (gr + 1) / 2 - 1;
        let kg = gamma[off];
        let om = &gamma[off + 1..off + gr + 1];
        let mut s2 = 0.0;
        for wi in om {
            s2 += wi * wi;
        }
        let (a, b) = link_coeffs(link, s2);
        let xb = z[gr];
        let mut dot = 0.0;
        for i in 0..gr {
            dot += om[i] * z[i];
        }
        let to = tape_off(gr);
        tz[to..to + gr].copy_from_slice(&z[..gr]);
        let f = a * xb + b * dot;
        for i in 0..gr {
            z[i] -= f * om[i];
        }
        let zg = xb + a * dot - b * s2 * xb;
        z[gr] = zg;
        scale *= (kg * (1.0 / (gr + 1) as f64)).exp();
        let coef = scale * (-kg).exp();
        let term = coef * zg * zg;
        q += term;
        steps[gr] = Step {
            s2,
            a,
            b,
            xb,
            dot,
            coef,
            term,
        };
    }

    // ---- Paine's block on (z_0, z_1)
    let (g1, g2) = (gamma[0], gamma[1]);
    let sq = (g1 * g1 + g2 * g2 + 1.0).sqrt();
    let (x0, x1) = (z[0], z[1]);
    let q2 = sq * (x0 * x0 + x1 * x1) - g1 * (x0 * x0 - x1 * x1) - g2 * (x0 * x1 * 2.0);
    let pterm = scale * q2;
    q += pterm;

    // ---- log-density (the reverse sweep below does not depend on it)
    let qq = t * t + q;
    let sqq = qq.sqrt();
    let alpha = ymu / sqq;
    let (inv_q, inv_sq) = (1.0 / qq, 1.0 / sqq);
    let (lmp, dlmp) = log_m_p_d1(alpha, p);
    let ll =
        -0.5 * p as f64 * LN_2PI - qq.ln() * (0.5 * d as f64) + (alpha * alpha - mm) * 0.5 + lmp;

    // =====================================================================
    // Reverse sweep for ∇Q (seed Q̄ = 1); z becomes z̄.
    // =====================================================================

    // ---- Paine's block: z̄_0, z̄_1 and ∂q/∂G_1 (all carry the scale S_2)
    let (s00, s11, s01) = (x0 * x0, x1 * x1, x0 * x1);
    let sum2 = s00 + s11;
    grad_gamma[0] = scale * (g1 / sq * sum2 - (s00 - s11));
    grad_gamma[1] = scale * (g2 / sq * sum2 - 2.0 * s01);
    z[0] = scale * (2.0 * sq * x0 - 2.0 * g1 * x0 - 2.0 * g2 * x1);
    z[1] = scale * (2.0 * sq * x1 + 2.0 * g1 * x1 - 2.0 * g2 * x0);

    // ---- rotations, groups 2, …, d−2
    let mut head = pterm; // P + Σ_{g ≤ gr} T_g
    for gr in 2..=d - 2 {
        let off = gr * (gr + 1) / 2 - 1;
        let Step {
            s2,
            a,
            b,
            xb,
            dot,
            coef,
            term,
        } = steps[gr];
        head += term;
        grad_gamma[off] = head * (1.0 / (gr + 1) as f64) - term;
        // z̄_gr from T_gr = coef z_gr² (z[gr] still holds the rotated z_gr)
        let zbb = 2.0 * coef * z[gr];
        let om = &gamma[off + 1..off + gr + 1];
        let to = tape_off(gr);
        let xa = &tz[to..to + gr];
        let gb = &mut grad_gamma[off + 1..off + gr + 1];
        // z'_i = x_i − f ω_i (i < gr): f̄ = −ω·z̄, ω̄_i = −f z̄_i, x̄_i = z̄_i
        let f = a * xb + b * dot;
        let mut fb = 0.0;
        for i in 0..gr {
            fb -= om[i] * z[i];
            gb[i] = -f * z[i];
        }
        // z'_gr = x_b + a δ − b s² x_b
        let mut xbb = zbb * (1.0 - b * s2);
        let mut dotb = a * zbb;
        let mut ab = dot * zbb;
        let mut bb = -s2 * xb * zbb;
        let mut s2b = -b * xb * zbb;
        // f = a x_b + b δ
        ab += xb * fb;
        xbb += a * fb;
        bb += dot * fb;
        dotb += b * fb;
        // δ = ω·x
        for i in 0..gr {
            gb[i] += dotb * xa[i];
            z[i] += dotb * om[i];
        }
        // (a, b) = link(s²), s² = ω·ω
        let (da, db) = link_derivs(link, s2, a, b);
        s2b += ab * da + bb * db;
        for i in 0..gr {
            gb[i] += 2.0 * s2b * om[i];
        }
        z[gr] = xbb;
    }

    // ---- basis turn, reversed: z̄ ← R_0 z̄
    if let Some(r0) = r0 {
        for i in 0..p {
            let mut acc = 0.0;
            for j in 0..p {
                acc += r0[i * p + j] * z[j];
            }
            ub[i] = acc;
        }
        z[..p].copy_from_slice(&ub[..p]);
    }

    // ---- frame, reversed (∂Q/∂μ into grad_mu): ū = E z̄,
    // μ̂̄ = v̄ + 2t y with v̄ = −κ ū − (2(ū·v)/(v·v)) u (zero for the constant
    // antipodal frame); then through μ̂ = μ/‖μ‖ (constant at μ = 0)
    for i in 0..d {
        ub[i] = (0..p).map(|j| e[j * d + i] * z[j]).sum();
    }
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
            grad_mu[i] = (ub[i] - s * mh[i]) * inv;
        }
    }

    // ---- combine: ∇ll = Q̄ ∇Q + ȳmu ∇ymu − ½ ∇mm
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

/// Below this `alpha`, `log M_p` uses its continued fraction (the value of
/// `loglik::CF_BELOW`).
const CF_BELOW: f64 = -2.0;

/// `(log M_p(a), d/da log M_p(a))`, as in [`crate::adjoint`]: bit-identical
/// to the first two outputs of [`log_m_p_derivs`].
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gamma_rot::{loglik_gr, loglik_gr_grad, loglik_gr_hess_fd};
    use nalgebra::DMatrix;
    use rand::{RngExt, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    fn unit(v: &mut [f64]) {
        let n = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        v.iter_mut().for_each(|x| *x /= n);
    }

    fn reference(d: usize, rng: &mut ChaCha8Rng) -> Vec<f64> {
        let mut m: Vec<f64> = (0..d).map(|_| rng.random_range(-1.0..1.0)).collect();
        unit(&mut m);
        m
    }

    /// Random coordinates: `‖G_1‖ ≤ √2`, `|κ_j| ≤ 1`, `‖ω_j‖ ≤ wmax`.
    fn coords(d: usize, wmax: f64, rng: &mut ChaCha8Rng) -> Vec<f64> {
        let mut g = vec![0.0; dims::gamma_len(d)];
        g[0] = rng.random_range(-1.0..1.0);
        g[1] = rng.random_range(-1.0..1.0);
        for gr in 2..=d - 2 {
            let off = gr * (gr + 1) / 2 - 1;
            g[off] = rng.random_range(-1.0..1.0);
            let mut w: Vec<f64> = (0..gr).map(|_| rng.random_range(-1.0..1.0)).collect();
            unit(&mut w);
            let s = rng.random_range(0.0..wmax);
            for i in 0..gr {
                g[off + 1 + i] = s * w[i];
            }
        }
        g
    }

    fn mu(d: usize, rng: &mut ChaCha8Rng) -> Vec<f64> {
        let mut m = reference(d, rng);
        let k = rng.random_range(0.5..4.0);
        m.iter_mut().for_each(|x| *x *= k);
        m
    }

    /// A random rotation of `ℝ^p` (QR of a Gaussian matrix, determinant +1).
    fn rotation(p: usize, rng: &mut ChaCha8Rng) -> DMatrix<f64> {
        let a = DMatrix::from_fn(p, p, |_, _| rng.random_range(-1.0..1.0));
        let mut q = a.qr().q();
        if q.determinant() < 0.0 {
            for r in 0..p {
                q[(r, 0)] = -q[(r, 0)];
            }
        }
        q
    }

    /// Value and gradient of the reverse kernel against the forward-mode
    /// kernel at `eta`, to `1e−9` absolute plus relative; returns the
    /// largest gradient discrepancy (absolute + relative scaled).
    fn check(fr: &GrFrame, eta: &[f64], y: &[f64], what: &str) -> f64 {
        let n = eta.len();
        let (mut g1, mut g2) = (vec![0.0; n], vec![0.0; n]);
        let f1 = loglik_gr_grad_rev(fr, eta, y, &mut g1);
        let f2 = loglik_gr_grad(fr, eta, y, &mut g2);
        let f3 = loglik_gr::<f64>(fr, eta, y);
        assert!(f1.is_finite(), "{what}: value {f1}");
        assert_eq!(f1.to_bits(), f3.to_bits(), "{what}: value {f1} vs {f3}");
        assert!(
            (f1 - f2).abs() <= 1e-9 + 1e-9 * f2.abs(),
            "{what}: value {f1} vs {f2}"
        );
        let mut worst = 0.0;
        for k in 0..n {
            assert!(g1[k].is_finite(), "{what} k={k}: gradient {}", g1[k]);
            let err = (g1[k] - g2[k]).abs() / (1.0 + g2[k].abs());
            worst = f64::max(worst, err);
            assert!(
                (g1[k] - g2[k]).abs() <= 1e-9 + 1e-9 * g2[k].abs(),
                "{what} k={k}: reverse {} vs forward {}",
                g1[k],
                g2[k]
            );
        }
        worst
    }

    /// Value and gradient agree with the forward-mode kernel at 20 random
    /// points for every `d = 3..=10`, both links, with and without a basis
    /// rotation.
    #[test]
    fn matches_forward_mode_at_random_points() {
        let mut rng = ChaCha8Rng::seed_from_u64(21);
        let mut worst = 0.0;
        for link in [RotLink::Stereo, RotLink::Exp] {
            for d in 3..=10 {
                for turned in [false, true] {
                    let mut fr = GrFrame::new(&reference(d, &mut rng), link);
                    if turned {
                        fr = fr.with_basis(&rotation(d - 1, &mut rng));
                    }
                    for i in 0..20 {
                        let mut eta = mu(d, &mut rng);
                        eta.extend(coords(d, 2.5, &mut rng));
                        let mut y = reference(d, &mut rng);
                        unit(&mut y);
                        let what = format!("{link:?} d={d} turned={turned} point {i}");
                        worst = f64::max(worst, check(&fr, &eta, &y, &what));
                    }
                }
            }
        }
        println!("largest scaled gradient discrepancy: {worst:.2e}");
    }

    /// The same at the special points: group 1 zero, a zero rotation
    /// vector, a zero `κ`, and for `Exp` a rotation by exactly `π`; also all
    /// of them at once, and `γ = 0`.
    #[test]
    fn matches_forward_mode_at_the_special_points() {
        let mut rng = ChaCha8Rng::seed_from_u64(5);
        for link in [RotLink::Stereo, RotLink::Exp] {
            for d in [4usize, 5, 6, 8, 10] {
                for turned in [false, true] {
                    let mut fr = GrFrame::new(&reference(d, &mut rng), link);
                    if turned {
                        fr = fr.with_basis(&rotation(d - 1, &mut rng));
                    }
                    let base: Vec<f64> = {
                        let mut e = mu(d, &mut rng);
                        e.extend(coords(d, 1.5, &mut rng));
                        e
                    };
                    let off2 = d + 2; // group 2 starts at γ index 2
                    let mut points = vec![];
                    let mut g1 = base.clone();
                    g1[d] = 0.0;
                    g1[d + 1] = 0.0;
                    points.push(("group 1 zero", g1));
                    let mut pole = base.clone();
                    pole[off2 + 1] = 0.0;
                    pole[off2 + 2] = 0.0;
                    points.push(("zero rotation vector", pole));
                    let mut zero = base.clone();
                    zero[off2] = 0.0;
                    points.push(("zero kappa", zero));
                    if link == RotLink::Exp {
                        let mut anti = base.clone();
                        let w = &mut anti[off2 + 1..off2 + 3];
                        let s = (w[0] * w[0] + w[1] * w[1]).sqrt();
                        w[0] *= std::f64::consts::PI / s;
                        w[1] *= std::f64::consts::PI / s;
                        points.push(("rotation by pi", anti));
                    }
                    let mut all = base.clone();
                    all[d..].iter_mut().for_each(|g| *g = 0.0);
                    points.push(("gamma = 0", all));
                    // every group's rotation vector zero, κ's kept
                    let mut poles = base.clone();
                    for gr in 2..=d - 2 {
                        let off = d + gr * (gr + 1) / 2 - 1;
                        poles[off + 1..off + gr + 1]
                            .iter_mut()
                            .for_each(|g| *g = 0.0);
                    }
                    points.push(("all poles", poles));
                    for (name, eta) in &points {
                        let mut y = reference(d, &mut rng);
                        unit(&mut y);
                        check(
                            &fr,
                            eta,
                            &y,
                            &format!("{link:?} d={d} turned={turned} {name}"),
                        );
                    }
                }
            }
        }
    }

    /// `μ = 0` and `μ̂ = −m` give finite values and gradients equal to the
    /// forward kernel's (constant frames there).
    #[test]
    fn matches_forward_mode_at_zero_mu_and_the_antipode() {
        let mut rng = ChaCha8Rng::seed_from_u64(33);
        for link in [RotLink::Stereo, RotLink::Exp] {
            for d in [3usize, 4, 5, 7, 10] {
                for turned in [false, true] {
                    // μ = 0 with a random reference
                    let mut fr = GrFrame::new(&reference(d, &mut rng), link);
                    if turned {
                        fr = fr.with_basis(&rotation(d - 1, &mut rng));
                    }
                    let mut eta = vec![0.0; d];
                    eta.extend(coords(d, 1.5, &mut rng));
                    let mut y = reference(d, &mut rng);
                    unit(&mut y);
                    check(
                        &fr,
                        &eta,
                        &y,
                        &format!("{link:?} d={d} turned={turned} mu = 0"),
                    );
                    // μ̂ = −m exactly: a coordinate reference, so that
                    // m/‖m‖ = m and μ = −2m normalises to −m without rounding
                    let mut m = vec![0.0; d];
                    m[1] = 1.0;
                    let mut fr = GrFrame::new(&m, link);
                    if turned {
                        fr = fr.with_basis(&rotation(d - 1, &mut rng));
                    }
                    let mut eta: Vec<f64> = m.iter().map(|x| -2.0 * x).collect();
                    eta.extend(coords(d, 1.5, &mut rng));
                    check(
                        &fr,
                        &eta,
                        &y,
                        &format!("{link:?} d={d} turned={turned} antipode"),
                    );
                }
            }
        }
    }

    /// The reverse gradient matches central differences of the value at a
    /// generic point (independently of the forward-mode kernel).
    #[test]
    fn matches_finite_differences() {
        let mut rng = ChaCha8Rng::seed_from_u64(8);
        for link in [RotLink::Stereo, RotLink::Exp] {
            for d in [3usize, 5, 7] {
                let fr = GrFrame::new(&reference(d, &mut rng), link)
                    .with_basis(&rotation(d - 1, &mut rng));
                let mut eta = mu(d, &mut rng);
                eta.extend(coords(d, 1.5, &mut rng));
                let mut y = reference(d, &mut rng);
                unit(&mut y);
                let mut g = vec![0.0; eta.len()];
                loglik_gr_grad_rev(&fr, &eta, &y, &mut g);
                let h = 1e-6;
                for k in 0..eta.len() {
                    let mut ep = eta.clone();
                    let mut em = eta.clone();
                    ep[k] += h;
                    em[k] -= h;
                    let fd = (loglik_gr::<f64>(&fr, &ep, &y) - loglik_gr::<f64>(&fr, &em, &y))
                        / (2.0 * h);
                    assert!(
                        (g[k] - fd).abs() < 1e-5 * (1.0 + g[k].abs()),
                        "{link:?} d={d} k={k}: reverse {} vs fd {fd}",
                        g[k]
                    );
                }
            }
        }
    }

    /// The Hessian by differences of the reverse gradient equals that of the
    /// forward-mode gradient (same step rule) to rounding.
    #[test]
    fn hessian_differences_match() {
        let mut rng = ChaCha8Rng::seed_from_u64(17);
        for d in [4usize, 6] {
            let fr = GrFrame::new(&reference(d, &mut rng), RotLink::Exp)
                .with_basis(&rotation(d - 1, &mut rng));
            let mut eta = mu(d, &mut rng);
            eta.extend(coords(d, 1.0, &mut rng));
            let mut y = reference(d, &mut rng);
            unit(&mut y);
            let n = eta.len();
            let (mut g1, mut h1) = (vec![0.0; n], vec![0.0; n * n]);
            let (mut g2, mut h2) = (vec![0.0; n], vec![0.0; n * n]);
            let f1 = loglik_gr_hess_fd_rev(&fr, &eta, &y, &mut g1, &mut h1);
            let f2 = loglik_gr_hess_fd(&fr, &eta, &y, &mut g2, &mut h2);
            assert!((f1 - f2).abs() < 1e-12 * f2.abs().max(1.0));
            for i in 0..n {
                assert!((g1[i] - g2[i]).abs() < 1e-9 * (1.0 + g2[i].abs()));
                for j in 0..n {
                    let (a, b) = (h1[i * n + j], h2[i * n + j]);
                    assert!(
                        (a - b).abs() < 1e-6 * (1.0 + b.abs()),
                        "d={d} H[{i},{j}]: {a} vs {b}"
                    );
                    assert!((a - h1[j * n + i]).abs() < 1e-12);
                }
            }
        }
    }

    /// Above `d = 10` the heap fallback agrees with finite differences.
    #[test]
    fn heap_fallback_matches_finite_differences() {
        let mut rng = ChaCha8Rng::seed_from_u64(12);
        let d = 11;
        let fr = GrFrame::new(&reference(d, &mut rng), RotLink::Stereo)
            .with_basis(&rotation(d - 1, &mut rng));
        let mut eta = mu(d, &mut rng);
        eta.extend(coords(d, 1.5, &mut rng));
        let mut y = reference(d, &mut rng);
        unit(&mut y);
        let mut g = vec![0.0; eta.len()];
        let f = loglik_gr_grad_rev(&fr, &eta, &y, &mut g);
        assert_eq!(f.to_bits(), loglik_gr::<f64>(&fr, &eta, &y).to_bits());
        let h = 1e-6;
        for k in 0..eta.len() {
            let mut ep = eta.clone();
            let mut em = eta.clone();
            ep[k] += h;
            em[k] -= h;
            let fd = (loglik_gr::<f64>(&fr, &ep, &y) - loglik_gr::<f64>(&fr, &em, &y)) / (2.0 * h);
            assert!(
                (g[k] - fd).abs() < 1e-5 * (1.0 + g[k].abs()),
                "k={k}: reverse {} vs fd {fd}",
                g[k]
            );
        }
    }
}

#[cfg(test)]
mod bench {
    use super::*;
    use crate::logtangent::LtFrame;
    use rand::{RngExt, SeedableRng};
    use rand_chacha::ChaCha8Rng;
    use std::hint::black_box;
    use std::time::Instant;

    /// Gradient cost per observation of the reverse `γ`-rot kernel against
    /// the forward-mode one, the log-tangent kernel and plain `γ`.
    /// `cargo test --release --lib gamma_rot_adjoint::bench -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn kernel_cost() {
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let n = 20_000;
        for d in [3usize, 5, 8, 10] {
            let k = dims::eta_len(d);
            let mut m: Vec<f64> = (0..d).map(|_| rng.random_range(-1.0..1.0)).collect();
            let nm = m.iter().map(|x| x * x).sum::<f64>().sqrt();
            m.iter_mut().for_each(|x| *x /= nm);
            let gr = GrFrame::new(&m, RotLink::Stereo);
            let lt = LtFrame::new(&m);
            let etas: Vec<Vec<f64>> = (0..n)
                .map(|_| {
                    (0..k)
                        .map(|j| {
                            let s = if j < d { 1.5 } else { 0.6 };
                            s * rng.random_range(-1.0..1.0)
                        })
                        .collect()
                })
                .collect();
            let ys: Vec<Vec<f64>> = (0..n)
                .map(|_| {
                    let mut v: Vec<f64> = (0..d).map(|_| rng.random_range(-1.0..1.0)).collect();
                    let s = v.iter().map(|x| x * x).sum::<f64>().sqrt();
                    v.iter_mut().for_each(|x| *x /= s);
                    v
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
            let rev = time(&mut || {
                for i in 0..n {
                    acc += loglik_gr_grad_rev(&gr, black_box(&etas[i]), &ys[i], &mut g);
                }
                black_box(&g);
            });
            let fwd = time(&mut || {
                for i in 0..n {
                    acc +=
                        crate::gamma_rot::loglik_gr_grad(&gr, black_box(&etas[i]), &ys[i], &mut g);
                }
                black_box(&g);
            });
            let ltg = time(&mut || {
                for i in 0..n {
                    acc +=
                        crate::logtangent::loglik_lt_grad(&lt, black_box(&etas[i]), &ys[i], &mut g);
                }
                black_box(&g);
            });
            let gam = time(&mut || {
                for i in 0..n {
                    acc += crate::loglik::loglik_eta_grad(black_box(&etas[i]), &ys[i], &mut g);
                }
                black_box(&g);
            });
            black_box(acc);
            println!(
                "d={d:2} N={k:2}: gamma-rot reverse {rev:5.0} ns/obs, forward {fwd:5.0}, log-tangent {ltg:5.0}, gamma {gam:5.0}"
            );
        }
    }
}
