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
//! # The Hessian
//!
//! [`loglik_gr_hess`] runs the same forward pass and reverse sweep (its
//! value and gradient are those of [`loglik_gr_grad_rev`], bit for bit),
//! recording also `o`, the state after the last rotation, and `β_gr`, the
//! adjoint of the output of rotation `gr` (block `0..=gr`). With `G` the
//! product of the group rotations, `o = Gᵀz` depends on the `ω` alone, and
//! `q = oᵀDo` with `D` block diagonal in `G_1` and the `κ` alone: Paine's
//! `S_2 [[√(1+‖G_1‖²) − G_{1,1}, −G_{1,2}], [−G_{1,2}, √(1+‖G_1‖²) + G_{1,1}]]`
//! on `(o_0, o_1)` and `S_gr e^{−κ_gr}` on `o_gr`. With `ō = 2Do`, `u, u′`
//! among `μ̂` and the `ω`, and `c, c′` among `G_1` and the `κ`,
//!
//! ```text
//! ∂²q/∂u∂u′ = 2 (∂o/∂u)ᵀ D (∂o/∂u′) + ∂²(ōᵀo)/∂u∂u′ at fixed ō,
//! ∂²q/∂c∂u = (∂ō/∂c)ᵀ ∂o/∂u,         ∂²q/∂c∂c′ = oᵀ (∂²D/∂c∂c′) o.
//! ```
//!
//! One forward sweep carries the tangents of the state along `μ̂` (the
//! columns of the frame Jacobian `J = ∂z/∂μ̂`, turned) and along every
//! `ω_{gr,j}` (injected at rotation `gr` as `∂(Q_grᵀ x)/∂ω_j`, `x` the
//! taped block) through the rotations, and adds the curvature of `ōᵀo` on
//! the way. A rotation is linear in the state, so a tangent `t` that
//! reaches rotation `gr` contributes
//!
//! ```text
//! β_grᵀ ∂(Q_grᵀ t)/∂ω_j = 2ω_j (k₁ t_gr + k₂ ω·t) + k₃ t_j − β_j (a t_gr + b ω·t),
//! k₁ = −a′(β·ω) − β_gr (b′s² + b),   k₂ = a′β_gr − b′(β·ω),   k₃ = aβ_gr − b(β·ω)
//! ```
//!
//! (`β·ω` over `0..gr`, `a′ = da/ds²`), and the group's own block is
//! `β_grᵀ ∂²(Q_grᵀ x)/∂ω_j∂ω_k = ω_j A_k + x_j B_k + β_j C_k + 2s̄² δ_jk`,
//! `s̄²` the reverse sweep's adjoint of `s²` and `A`, `B`, `C` vectors made
//! of `ω`, `x`, `β` and `a′, b′, a″, b″`. The link's second derivatives
//! are `2a/(4+s²)²` and `2b/(4+s²)²` for `Stereo`; for `Exp`
//! `−(a + 6a′)/(4s²)` and `(a′/2 − 2b′)/s²` from `s² = 1` on and their
//! Taylor series below it, where the closed forms cancel. The first
//! derivatives are the gradient's, with its series branch below
//! `s² = 10⁻¹⁰`. `log(S_gr e^{−κ_gr}) = Σ_{i≥gr} κ_i/(i+1) − κ_gr` and
//! `log S_2` are linear in the `κ`, so `∂ō_r/∂κ_i = ō_r (1/(i+1) − δ_ri)`
//! for `r ≤ i` (0 above): prefix sums over the rows of `∂o`.
//!
//! The tangents are stored state-major, the `ω` of group `gr` after those
//! of the groups above it. The tangents alive when group `gr` rotates are
//! then a prefix of the columns, and row `r ≥ 2` of the final `∂o` vanishes
//! after the tangents of group `r`, so the loops of the sweep and of
//! `2(∂o)ᵀD(∂o)` run over contiguous prefixes. `∂²q/∂(μ̂, ω)²` is
//! accumulated in that order in the lower triangle of the output, which
//! holds nothing else until the outer layer mirrors the upper one. The
//! frame's own curvature, `t²`, the chain through `μ̂ = μ/‖μ‖` and the outer
//! layer are those of the log-tangent kernel (`crate::frame_hess`); at
//! `μ = 0` and `μ̂ = −m` the Hessian keeps the gradient's constant frames.
//! Against fourth-order differences of the gradient the entries agree to
//! the differences' own error (below `5e−9` of `max(|H_ij|, 1)` at a step
//! of `1e−3`; the tests require `1e−6` at random and special points,
//! `d = 3..=11`, both links).
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
//! The Hessian, each with the value and the gradient (`bench::hessian_cost`,
//! the same protocol; the differences, [`loglik_gr_hess_fd_rev`], on 2 000
//! observations):
//!
//! | `d` | `N` | exact | differences | reverse gradient |
//! |---|---|---|---|---|
//! | 3  | 5  | 102  | 425   | 48  |
//! | 5  | 14 | 356  | 1796  | 73  |
//! | 8  | 35 | 1715 | 12622 | 197 |
//! | 10 | 54 | 3344 | 26459 | 271 |
//!
//! The exact Hessian costs 2 to 12 gradients, 4× (`d = 3`) to 8× less than
//! the `2N` gradients of the differences.
//!
//! No allocation happens for `d ≤ 10` (const-generic stack buffers);
//! larger `d` use heap buffers.

use crate::dims;
use crate::frame_hess::{self, FrameState, Outer};
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
            core::<false>(
                fr,
                mu,
                gamma,
                y,
                grad,
                Work::new(&mut v, &mut tz, &mut steps),
                HessWork::empty(),
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

/// Log-density, its exact gradient (into `grad`, as
/// [`loglik_gr_grad_rev`]) and its exact Hessian in `eta` (into `hess`,
/// row-major `n × n`, `n = eta.len()`, exactly symmetric), by one forward
/// pass, one reverse sweep and one forward sweep of tangents through the
/// rotations (see the [module documentation](self)).
///
/// The value and gradient are bit-identical to [`loglik_gr_grad_rev`]. At
/// `μ = 0` and `μ̂ = −m`, where the frame is held constant, the Hessian is
/// that of the gradient's convention: `Q` does not depend on `μ` (`μ = 0`)
/// or depends on it through `t = μ̂·y` only (`μ̂ = −m`).
///
/// # Panics
/// As [`loglik_gr_grad_rev`], or if `hess.len() != eta.len()²`.
pub fn loglik_gr_hess(
    fr: &GrFrame,
    eta: &[f64],
    y: &[f64],
    grad: &mut [f64],
    hess: &mut [f64],
) -> f64 {
    let d = y.len();
    assert!(d >= 3, "the ESAG density needs d >= 3, got d = {d}");
    assert_eq!(fr.dim(), d, "frame dimension {} but d = {d}", fr.dim());
    assert_eq!(eta.len(), dims::eta_len(d), "eta has length {}", eta.len());
    assert_eq!(grad.len(), eta.len(), "grad has length {}", grad.len());
    let n = eta.len();
    assert_eq!(hess.len(), n * n, "hess must have length eta.len()²");
    // const parameters: D, P = D − 1, R = tape_len(D), NG = gamma_len(D),
    // NE = eta_len(D), NC = D + R − 1 tangents, NB = beta_len(D)
    match d {
        3 => fixed_hess::<3, 2, 1, 2, 5, 3, 0>(fr, eta, y, grad, hess),
        4 => fixed_hess::<4, 3, 3, 5, 9, 6, 3>(fr, eta, y, grad, hess),
        5 => fixed_hess::<5, 4, 6, 9, 14, 10, 7>(fr, eta, y, grad, hess),
        6 => fixed_hess::<6, 5, 10, 14, 20, 15, 12>(fr, eta, y, grad, hess),
        7 => fixed_hess::<7, 6, 15, 20, 27, 21, 18>(fr, eta, y, grad, hess),
        8 => fixed_hess::<8, 7, 21, 27, 35, 28, 25>(fr, eta, y, grad, hess),
        9 => fixed_hess::<9, 8, 28, 35, 44, 36, 33>(fr, eta, y, grad, hess),
        10 => fixed_hess::<10, 9, 36, 44, 54, 45, 42>(fr, eta, y, grad, hess),
        _ => {
            let (mu, gamma) = eta.split_at(d);
            let p = d - 1;
            let nc = d + tape_len(d) - 1;
            let mut v = vec![0.0; NV * d];
            let mut tz = vec![0.0; tape_len(d)];
            let mut steps = vec![Step::default(); d];
            let mut hv =
                vec![0.0; n + 2 * p + beta_len(d) + 2 * d + p * d + p * nc + 5 * nc + d * d];
            let (dq, rest) = hv.split_at_mut(n);
            let (o, rest) = rest.split_at_mut(p);
            let (bt, rest) = rest.split_at_mut(beta_len(d));
            let (ubar, rest) = rest.split_at_mut(d);
            let (xv, rest) = rest.split_at_mut(d);
            let (ov, rest) = rest.split_at_mut(p);
            let (jt, rest) = rest.split_at_mut(p * d);
            let (tan, rest) = rest.split_at_mut(p * nc);
            let (rows, am) = rest.split_at_mut(5 * nc);
            let hw = HessWork {
                hess,
                dq,
                o,
                bt,
                ubar,
                xv,
                ov,
                jt,
                tan,
                rows,
                am,
            };
            core::<true>(
                fr,
                mu,
                gamma,
                y,
                grad,
                Work::new(&mut v, &mut tz, &mut steps),
                hw,
            )
        }
    }
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

/// Length of the record of the adjoints `β_gr` (one block of `gr + 1` per
/// group): `Σ_{gr=2}^{d−2} (gr + 1)`.
#[inline(always)]
const fn beta_len(d: usize) -> usize {
    d * (d - 1) / 2 - 3
}

/// Offset of group `gr`'s block in the record of the adjoints.
#[inline(always)]
const fn beta_off(gr: usize) -> usize {
    gr * (gr + 1) / 2 - 3
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
    core::<false>(
        fr,
        &mu[..D],
        &gamma[..dims::gamma_len(D)],
        &y[..D],
        &mut grad[..dims::eta_len(D)],
        w,
        HessWork::empty(),
    )
}

/// Stack buffers for the Hessian at dimension `D` (`P = D − 1`, a tape of
/// `R` entries, `NG` shape and `NE` parameters, `NC` tangents and `NB`
/// recorded adjoints).
#[inline(never)]
fn fixed_hess<
    const D: usize,
    const P: usize,
    const R: usize,
    const NG: usize,
    const NE: usize,
    const NC: usize,
    const NB: usize,
>(
    fr: &GrFrame,
    eta: &[f64],
    y: &[f64],
    grad: &mut [f64],
    hess: &mut [f64],
) -> f64 {
    const {
        assert!(
            D >= 3
                && P + 1 == D
                && R == tape_len(D)
                && NG == (D - 2) * (D + 1) / 2
                && NE == D + NG
                && NC == D + R - 1
                && NB == beta_len(D)
        )
    };
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
    let mut dq = [0.0; NE];
    let mut o = [0.0; P];
    let mut bt = [0.0; NB];
    let mut dv = [[0.0; D]; 2];
    let mut ov = [0.0; P];
    let mut jt = [[0.0; D]; P];
    let mut tan = [[0.0; NC]; P];
    let mut rows = [[0.0; NC]; 5];
    let mut am = [[0.0; D]; D];
    let [ubar, xv] = &mut dv;
    let hw = HessWork {
        hess: &mut hess[..NE * NE],
        dq: &mut dq,
        o: &mut o,
        bt: &mut bt,
        ubar,
        xv,
        ov: &mut ov,
        jt: jt.as_flattened_mut(),
        tan: tan.as_flattened_mut(),
        rows: rows.as_flattened_mut(),
        am: am.as_flattened_mut(),
    };
    core::<true>(fr, &mu[..D], &gamma[..NG], &y[..D], &mut grad[..NE], w, hw)
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

/// Terms of the series of `(d²a/d(s²)², d²b/d(s²)²)` for the `Exp` link
/// below `s² = 1` (truncation below `1e−16` relative).
const EXP_D2_TERMS: usize = 10;

/// `(−1)ⁿ (n+2)(n+1)/(2n+5)!` and `(−1)ⁿ (n+2)(n+1)/(2n+6)!`: the series
/// `a″(x) = Σ_n c_n xⁿ` of `a = sin √x/√x` and `b″` of `b = (1 − cos √x)/x`.
const EXP_D2_COEF: [[f64; EXP_D2_TERMS]; 2] = {
    let mut c = [[0.0; EXP_D2_TERMS]; 2];
    let mut n = 0;
    while n < EXP_D2_TERMS {
        let mut fa = 1.0; // (2n+5)!
        let mut k = 2;
        while k <= 2 * n + 5 {
            fa *= k as f64;
            k += 1;
        }
        let sign = if n % 2 == 0 { 1.0 } else { -1.0 };
        let num = ((n + 2) * (n + 1)) as f64;
        c[0][n] = sign * num / fa;
        c[1][n] = sign * num / (fa * (2 * n + 6) as f64);
        n += 1;
    }
    c
};

/// `(d²a/d(s²)², d²b/d(s²)²)` at `s²`, given `(a, b)` and their first
/// derivatives `(da, db)` at `s²`. `Stereo`: `2a/(4+s²)²`, `2b/(4+s²)²`.
/// `Exp`: `−(a + 6a′)/(4s²)` and `(a′/2 − 2b′)/s²` from `s² = 1` on,
/// below that the series (the closed forms cancel catastrophically).
#[inline(always)]
fn link_second(link: RotLink, s2: f64, a: f64, b: f64, da: f64, db: f64) -> (f64, f64) {
    match link {
        RotLink::Stereo => {
            let inv = 1.0 / (s2 + 4.0);
            let i2 = 2.0 * inv * inv;
            (a * i2, b * i2)
        }
        RotLink::Exp => {
            if s2 < 1.0 {
                let (mut sa, mut sb) = (0.0, 0.0);
                for k in (0..EXP_D2_TERMS).rev() {
                    sa = sa * s2 + EXP_D2_COEF[0][k];
                    sb = sb * s2 + EXP_D2_COEF[1][k];
                }
                (sa, sb)
            } else {
                let inv = 1.0 / s2;
                (-0.25 * (a + 6.0 * da) * inv, (0.5 * da - 2.0 * db) * inv)
            }
        }
    }
}

/// Forward pass (recording) and reverse sweep, and for `HESS` the Hessian
/// into `hw`. See the module documentation.
#[inline(always)]
fn core<const HESS: bool>(
    fr: &GrFrame,
    mu: &[f64],
    gamma: &[f64],
    y: &[f64],
    grad: &mut [f64],
    w: Work<'_>,
    hw: HessWork<'_>,
) -> f64 {
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
    if HESS {
        hw.o[..p].copy_from_slice(&z[..p]);
    }

    // ---- log-density (the reverse sweep below does not depend on it)
    let qq = t * t + q;
    let sqq = qq.sqrt();
    let alpha = ymu / sqq;
    let (inv_q, inv_sq) = (1.0 / qq, 1.0 / sqq);
    // log_m_p_derivs returns the same first two outputs, bit for bit
    let (lmp, dlmp, d2lmp) = if HESS {
        log_m_p_derivs(alpha, p)
    } else {
        let (l, d1) = log_m_p_d1(alpha, p);
        (l, d1, 0.0)
    };
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
        if HESS {
            let bo = beta_off(gr);
            hw.bt[bo..bo + gr].copy_from_slice(&z[..gr]);
            hw.bt[bo + gr] = zbb;
        }
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
    if HESS {
        hw.ubar[..d].copy_from_slice(ub);
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
    if HESS {
        hw.dq[..d].copy_from_slice(grad_mu);
        hw.dq[d..d + grad_gamma.len()].copy_from_slice(grad_gamma);
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

    if HESS {
        let fwd = GrFwd {
            fs: FrameState {
                e,
                y,
                mu_hat: mh,
                v,
                u,
                kappa,
                vv,
                inv_n: inv,
                mu_free: !zero_mu,
                frame_free: !zero_mu && !antipode,
            },
            r0,
            link,
            gamma,
            tz,
            steps,
            scale,
            sq,
            pterm,
            g: ub,
            outer: Outer::new(d, qq, sqq, alpha, ab, d2lmp, qb),
        };
        gr_hessian(&fwd, hw);
    }
    ll
}

/// Buffers of [`loglik_gr_hess`] (`n` parameters, `nc = d + n_ω` tangents,
/// `n_ω = Σ_{gr=2}^{d−2} gr` rotation entries); empty for the gradient
/// alone.
struct HessWork<'a> {
    /// The output, row-major `n × n`: first `∂²Q` in the upper triangle,
    /// and `∂²q/∂(μ̂, ω)²` in tangent order in the lower one (row `c` of it,
    /// `c + 1` entries, at row `c + 1` of `hess`; `nc ≤ n − 2`).
    hess: &'a mut [f64],
    /// `∇Q` (`n`).
    dq: &'a mut [f64],
    /// `o = Gᵀz`, the forward state after the last rotation (`p`).
    o: &'a mut [f64],
    /// `β_gr`, the adjoint of the output of group `gr`'s rotation (block
    /// `0..=gr`), at [`beta_off`]`(gr)`.
    bt: &'a mut [f64],
    /// `ū = E z̄` (`d`).
    ubar: &'a mut [f64],
    /// Scratch (`d`).
    xv: &'a mut [f64],
    /// `ō = 2Do` (`p`).
    ov: &'a mut [f64],
    /// `J = ∂z/∂μ̂` (`p × d`).
    jt: &'a mut [f64],
    /// The tangents, state-major (`p × nc`): column `l < d` is `∂(state)/∂μ̂_l`,
    /// then the columns of `ω_{gr,j}` by group, `d−2` first (see
    /// `gr_hessian`); at the end `∂o/∂(μ̂, ω)`.
    tan: &'a mut [f64],
    /// Rows over the tangents (`5 nc`).
    rows: &'a mut [f64],
    /// `A = ∂²Q/∂μ̂²` (`d²`, upper triangle).
    am: &'a mut [f64],
}

impl HessWork<'static> {
    /// No Hessian.
    fn empty() -> Self {
        HessWork {
            hess: &mut [],
            dq: &mut [],
            o: &mut [],
            bt: &mut [],
            ubar: &mut [],
            xv: &mut [],
            ov: &mut [],
            jt: &mut [],
            tan: &mut [],
            rows: &mut [],
            am: &mut [],
        }
    }
}

/// What the Hessian needs from the forward pass and the reverse sweep,
/// besides the buffers of [`HessWork`].
struct GrFwd<'a> {
    fs: FrameState<'a>,
    r0: Option<&'a [f64]>,
    link: RotLink,
    gamma: &'a [f64],
    /// Pre-rotation blocks and per-group records of the forward pass.
    tz: &'a [f64],
    steps: &'a [Step],
    /// `S_2 = Π_{i≥2} e^{κ_i/(i+1)}`.
    scale: f64,
    /// `√(1 + ‖G_1‖²)`.
    sq: f64,
    /// Paine's term `P`.
    pterm: f64,
    /// `∂Q/∂μ̂` with `μ̂` taken as a free vector.
    g: &'a [f64],
    outer: Outer,
}

/// The Hessian (see the module documentation): `q = oᵀDo` with
/// `o = Gᵀz` (`G` the product of the group rotations, depending on the
/// `ω` only) and `D` block diagonal (Paine's 2 × 2 block and the
/// `S_gr e^{−κ_gr}`, depending on `G_1` and the `κ` only). One forward
/// sweep carries the tangents of the state along `μ̂` (`J`) and along every
/// `ω_{gr,j}` (injected at its rotation) and adds the curvature of
/// `φ = ōᵀo` at fixed `ō = 2Do` on the way; then `2 (∂o)ᵀD(∂o)`, the rows
/// of `κ` and `G_1`, and the frame, `μ̂ = μ/‖μ‖` and the outer layer of
/// [`crate::frame_hess`].
///
/// The tangents are stored state-major with the `ω` columns of group `gr`
/// at `colw(gr) = d + Σ_{g>gr} g`: when group `gr` rotates, the live
/// tangents are the columns `0..colw(gr)`, and row `r ≥ 2` of the final
/// `∂o` vanishes beyond `colw(r) + r`, so every loop runs over a prefix of
/// contiguous columns.
#[inline(always)]
fn gr_hessian(f: &GrFwd<'_>, h: HessWork<'_>) {
    let fs = &f.fs;
    let d = fs.y.len();
    let p = d - 1;
    let gamma = f.gamma;
    let ng = gamma.len();
    let n = d + ng;
    let nw = tape_len(d) - 1;
    let nc = d + nw;
    let HessWork {
        hess,
        dq,
        o,
        bt,
        ubar,
        xv,
        ov,
        jt,
        tan,
        rows,
        am,
    } = h;
    let (hess, dq, o, bt) = (&mut hess[..n * n], &dq[..n], &o[..p], &bt[..beta_len(d)]);
    let (ubar, xv, ov, jt) = (&ubar[..d], &mut xv[..d], &mut ov[..p], &mut jt[..p * d]);
    let (tan, rows, am) = (&mut tan[..p * nc], &mut rows[..5 * nc], &mut am[..d * d]);
    let (dl, rest) = rows.split_at_mut(nc);
    let (fl, rest) = rest.split_at_mut(nc);
    let (el, rest) = rest.split_at_mut(nc);
    let (pv, rv) = rest.split_at_mut(nc);
    let (link, steps) = (f.link, f.steps);
    // first tangent column of group gr, the η index of ω_{gr,0}, and where
    // row c of ∂²q/∂(μ̂, ω)² starts (in the lower triangle of hess)
    let colw = |gr: usize| d + nw + 1 - gr * (gr + 1) / 2;
    let eta_w = |gr: usize| d + gr * (gr + 1) / 2;
    let hrow = |c: usize| (c + 1) * n;

    // ---- tangents along μ̂: the columns of J (zero for constant frames)
    if fs.frame_free {
        frame_hess::frame_jacobian(fs, f.r0, jt);
        for r in 0..p {
            tan[r * nc..r * nc + d].copy_from_slice(&jt[r * d..(r + 1) * d]);
        }
    } else {
        for r in 0..p {
            tan[r * nc..r * nc + d].fill(0.0);
        }
    }

    // ---- forward sweep through the rotations, groups d−2, …, 2: with
    // δ = ω·t_{0..gr}, the curvature of tangent t along ω_j is
    // β_grᵀ ∂(Q_grᵀ t)/∂ω_j = 2ω_j (k1 t_gr + k2 δ) + k3 t_j − β_j (a t_gr + b δ)
    for gr in (2..=d - 2).rev() {
        let off = gr * (gr + 1) / 2 - 1;
        let om = &gamma[off + 1..off + gr + 1];
        let Step {
            s2, a, b, xb, dot, ..
        } = steps[gr];
        let (da, db) = link_derivs(link, s2, a, b);
        let (dda, ddb) = link_second(link, s2, a, b, da, db);
        let x = &f.tz[tape_off(gr)..tape_off(gr) + gr];
        let beta = &bt[beta_off(gr)..beta_off(gr) + gr + 1];
        let (bw, bg) = (frame_hess::dot(&beta[..gr], om), beta[gr]);
        let fx = a * xb + b * dot;
        let (psi1, psi2) = (db * s2 + b, ddb * s2 + 2.0 * db);
        let k1 = -da * bw - bg * psi1;
        let k2 = -db * bw + da * bg;
        let k3 = a * bg - b * bw;
        let lc = colw(gr);
        let (dl, fl, el) = (&mut dl[..lc], &mut fl[..lc], &mut el[..lc]);
        dl.fill(0.0);
        for i in 0..gr {
            let w = om[i];
            for (dc, &tc) in dl.iter_mut().zip(&tan[i * nc..i * nc + lc]) {
                *dc += w * tc;
            }
        }
        for (c, &tb) in tan[gr * nc..gr * nc + lc].iter().enumerate() {
            fl[c] = a * tb + b * dl[c];
            el[c] = 2.0 * (k1 * tb + k2 * dl[c]);
        }
        for j in 0..gr {
            let row = &mut hess[hrow(lc + j)..hrow(lc + j) + lc];
            let (wj, bj, tj) = (om[j], beta[j], &tan[j * nc..j * nc + lc]);
            for (((h, &e), &t), &fc) in row.iter_mut().zip(&*el).zip(tj).zip(&*fl) {
                *h = e * wj + k3 * t - bj * fc;
            }
        }
        // the rotation of the live tangents
        for i in 0..gr {
            let w = om[i];
            for (t, &fc) in tan[i * nc..i * nc + lc].iter_mut().zip(&*fl) {
                *t -= w * fc;
            }
        }
        for (t, &dc) in tan[gr * nc..gr * nc + lc].iter_mut().zip(&*dl) {
            *t += a * dc - b * s2 * *t;
        }
        // the group's own second derivatives (e0 is the reverse sweep's s̄²)
        let phif = da * xb + db * dot;
        let e0 = -bw * phif + bg * (da * dot - psi1 * xb);
        let e1 = -bw * (dda * xb + ddb * dot) + bg * (dda * dot - psi2 * xb);
        // 4e1 ω_jω_k + 2k2(ω_j x_k + ω_k x_j) − 2φ_f(β_k ω_j + β_j ω_k)
        // − b(β_k x_j + β_j x_k) + 2e0 δ_jk = ω_j A_k + x_j B_k + β_j C_k + 2e0 δ_jk
        for k in 0..gr {
            el[k] = 4.0 * e1 * om[k] + 2.0 * k2 * x[k] - 2.0 * phif * beta[k];
            fl[k] = 2.0 * k2 * om[k] - b * beta[k];
            dl[k] = -2.0 * phif * om[k] - b * x[k];
        }
        for j in 0..gr {
            let row = &mut hess[hrow(lc + j) + lc..hrow(lc + j) + lc + j + 1];
            let (wj, xj, bj) = (om[j], x[j], beta[j]);
            for (k, h) in row.iter_mut().enumerate() {
                *h = wj * el[k] + xj * fl[k] + bj * dl[k];
            }
            row[j] += 2.0 * e0;
        }
        // inject the group's tangents ∂(Q_grᵀ x)/∂ω_j as columns lc + j
        let cb = 2.0 * (da * dot - psi1 * xb);
        for j in 0..gr {
            let cj = 2.0 * om[j] * phif + b * x[j];
            for i in 0..gr {
                tan[i * nc + lc + j] = -cj * om[i];
            }
            tan[j * nc + lc + j] -= fx;
            tan[gr * nc + lc + j] = om[j] * cb + a * x[j];
            for i in gr + 1..p {
                tan[i * nc + lc + j] = 0.0;
            }
        }
    }

    // ---- Gauss–Newton part 2 (∂o)ᵀD(∂o), rows 0 and 1 through Paine's
    // block D_P (every tangent), row r ≥ 2 with weight S_r e^{−κ_r} (the
    // tangents of groups ≥ r); the mean-mean entries start here
    let (g1, g2) = (gamma[0], gamma[1]);
    let (sc, sq) = (f.scale, f.sq);
    let (x0, x1) = (o[0], o[1]);
    let (d00, d01, d11) = (sc * (sq - g1), -sc * g2, sc * (sq + g1));
    {
        let (t0, t1) = (&tan[..nc], &tan[nc..2 * nc]);
        for c in 0..nc {
            dl[c] = 2.0 * (d00 * t0[c] + d01 * t1[c]);
            fl[c] = 2.0 * (d01 * t0[c] + d11 * t1[c]);
        }
        for c1 in 0..nc {
            let (a0, a1) = (t0[c1], t1[c1]);
            let row = &mut hess[hrow(c1)..hrow(c1) + c1 + 1];
            let pairs = row.iter_mut().zip(dl[..=c1].iter().zip(&fl[..=c1]));
            if c1 < d {
                for (h, (&u0, &u1)) in pairs {
                    *h = a0 * u0 + a1 * u1;
                }
            } else {
                for (h, (&u0, &u1)) in pairs {
                    *h += a0 * u0 + a1 * u1;
                }
            }
        }
    }
    for r in 2..p {
        let lr = colw(r) + r;
        let tr = &tan[r * nc..r * nc + lr];
        let w2 = 2.0 * steps[r].coef;
        for (c1, &t1) in tr.iter().enumerate() {
            let wc = w2 * t1;
            let row = &mut hess[hrow(c1)..hrow(c1) + c1 + 1];
            for (h, &t2) in row.iter_mut().zip(&tr[..=c1]) {
                *h += wc * t2;
            }
        }
    }

    // ---- into the η layout: mean-mean into A, mean-ω into the cross block
    // (through μ̂ = μ/‖μ‖), ω-ω into the upper triangle
    for r in 0..d {
        hess[r * n + r..(r + 1) * n].fill(0.0);
    }
    if fs.mu_free {
        frame_hess::init_mean(fs.y, am);
        for c1 in 0..d {
            for c2 in 0..=c1 {
                am[c2 * d + c1] += hess[hrow(c1) + c2];
            }
        }
    }
    for g in 2..=d - 2 {
        for j in 0..g {
            let c1 = colw(g) + j;
            let (e1, h1) = (eta_w(g) + j, hrow(c1));
            if fs.frame_free {
                xv.copy_from_slice(&hess[h1..h1 + d]);
                frame_hess::project_cross(fs, xv, hess, n, e1);
            }
            // groups above g come first; the group's own columns up to j
            for g2 in g + 1..=d - 2 {
                let (cb, eb) = (colw(g2), eta_w(g2));
                hess.copy_within(h1 + cb..h1 + cb + g2, e1 * n + eb);
            }
            let cb = colw(g);
            for j2 in 0..=j {
                hess[(eta_w(g) + j2) * n + e1] = hess[h1 + cb + j2];
            }
        }
    }

    // ---- rows of G_1 and κ: ∂²q/∂c∂(·) = (∂ō/∂c)ᵀ ∂o/∂(·), with
    // ∂ō_{0,1}/∂G_1 = 2 S_2 (∂D_P/∂G_1) o_{0,1} and
    // ∂ō_r/∂κ_i = ō_r (1/(i+1) − δ_ri) for r ≤ i (0 above), by the prefix
    // sums π = Σ_{r≤i} ō_r t_r
    ov[0] = 2.0 * (d00 * x0 + d01 * x1);
    ov[1] = 2.0 * (d01 * x0 + d11 * x1);
    for r in 2..p {
        ov[r] = bt[beta_off(r) + r];
    }
    let put = |hess: &mut [f64], th: usize, rv: &[f64]| {
        let et = d + th;
        if fs.frame_free {
            frame_hess::project_cross(fs, &rv[..d], hess, n, et);
        }
        for g in 2..=d - 2 {
            let (cb, eb) = (colw(g), eta_w(g));
            if eb > et {
                hess[et * n + eb..et * n + eb + g].copy_from_slice(&rv[cb..cb + g]);
            } else {
                for j in 0..g {
                    hess[(eb + j) * n + et] = rv[cb + j];
                }
            }
        }
    };
    let (v10, v11) = (
        2.0 * sc * (g1 / sq - 1.0) * x0,
        2.0 * sc * (g1 / sq + 1.0) * x1,
    );
    let (v20, v21) = (
        2.0 * sc * (g2 / sq * x0 - x1),
        2.0 * sc * (g2 / sq * x1 - x0),
    );
    {
        let (t0, t1) = (&tan[..nc], &tan[nc..2 * nc]);
        for c in 0..nc {
            rv[c] = v10 * t0[c] + v11 * t1[c];
        }
        put(hess, 0, rv);
        for c in 0..nc {
            rv[c] = v20 * t0[c] + v21 * t1[c];
        }
        put(hess, 1, rv);
        for c in 0..nc {
            pv[c] = ov[0] * t0[c] + ov[1] * t1[c];
        }
    }
    for i in 2..=d - 2 {
        let (oi, inv) = (ov[i], 1.0 / (i + 1) as f64);
        let ti = &tan[i * nc..(i + 1) * nc];
        for c in 0..nc {
            let v = oi * ti[c];
            pv[c] += v;
            rv[c] = pv[c] * inv - v;
        }
        put(hess, i * (i + 1) / 2 - 1, rv);
    }

    // ---- G_1 and κ against each other
    let sum2 = x0 * x0 + x1 * x1;
    let c3 = sc * sum2 / (sq * sq * sq);
    hess[d * n + d] = c3 * (1.0 + g2 * g2);
    hess[d * n + d + 1] = -c3 * g1 * g2;
    hess[(d + 1) * n + d + 1] = c3 * (1.0 + g1 * g1);
    let mut head = f.pterm;
    for i in 2..=d - 2 {
        let ki = d + i * (i + 1) / 2 - 1;
        let inv = 1.0 / (i + 1) as f64;
        hess[d * n + ki] = dq[d] * inv;
        hess[(d + 1) * n + ki] = dq[d + 1] * inv;
        let ti = steps[i].term;
        // (P + Σ_{g<i} T_g)/(i+1)² + T_i (i/(i+1))²
        let r = i as f64 * inv;
        hess[ki * n + ki] = head * inv * inv + ti * r * r;
        head += ti;
        for j in i + 1..=d - 2 {
            let kj = d + j * (j + 1) / 2 - 1;
            hess[ki * n + kj] = dq[ki] * (1.0 / (j + 1) as f64);
        }
    }

    // ---- the frame's curvature, μ̂ = μ/‖μ‖ and the outer layer (which
    // overwrites the lower triangle)
    if fs.mu_free {
        if fs.frame_free {
            frame_hess::add_frame_curvature(fs, ubar, am);
        }
        frame_hess::project_mean(fs, f.g, am, xv, hess, n);
    }
    frame_hess::assemble(&f.outer, dq, fs.y, hess, n);
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

    /// Central differences (Richardson: steps `±h`, `±2h`) of the reverse
    /// gradient along coordinate `i`.
    fn gradient_difference(fr: &GrFrame, eta: &[f64], y: &[f64], i: usize, h: f64) -> Vec<f64> {
        let n = eta.len();
        let g = |k: f64| {
            let mut e = eta.to_vec();
            e[i] += k * h;
            let mut g = vec![0.0; n];
            loglik_gr_grad_rev(fr, &e, y, &mut g);
            g
        };
        let (p1, m1, p2, m2) = (g(1.0), g(-1.0), g(2.0), g(-2.0));
        (0..n)
            .map(|r| (8.0 * (p1[r] - m1[r]) - (p2[r] - m2[r])) / (12.0 * h))
            .collect()
    }

    /// The exact Hessian at `eta`: value and gradient bit-identical to
    /// `loglik_gr_grad_rev`, every entry finite, exactly symmetric, and the
    /// columns `cols` within `1e−6` (relative to `max(|·|, 1)`) of central
    /// differences of the gradient, and (with `fd`) all of it within `1e−6`
    /// of `loglik_gr_hess_fd_rev`. Returns the worst error against the
    /// differences of the gradient.
    fn check_hess(
        fr: &GrFrame,
        eta: &[f64],
        y: &[f64],
        cols: &[usize],
        fd: bool,
        what: &str,
    ) -> f64 {
        let n = eta.len();
        let (mut g, mut h) = (vec![0.0; n], vec![0.0; n * n]);
        let mut g2 = vec![0.0; n];
        let f = loglik_gr_hess(fr, eta, y, &mut g, &mut h);
        let f2 = loglik_gr_grad_rev(fr, eta, y, &mut g2);
        assert_eq!(f.to_bits(), f2.to_bits(), "{what}: value {f} vs {f2}");
        for k in 0..n {
            assert_eq!(g[k].to_bits(), g2[k].to_bits(), "{what}: gradient {k}");
        }
        for r in 0..n {
            for c in 0..n {
                assert!(
                    h[r * n + c].is_finite(),
                    "{what}: H[{r},{c}] = {}",
                    h[r * n + c]
                );
                assert_eq!(
                    h[r * n + c].to_bits(),
                    h[c * n + r].to_bits(),
                    "{what}: symmetry"
                );
            }
        }
        if fd {
            let (mut g3, mut h3) = (vec![0.0; n], vec![0.0; n * n]);
            loglik_gr_hess_fd_rev(fr, eta, y, &mut g3, &mut h3);
            for k in 0..n * n {
                let err = (h[k] - h3[k]).abs() / h3[k].abs().max(1.0);
                assert!(
                    err < 1e-6,
                    "{what}: H[{},{}] = {} vs loglik_gr_hess_fd_rev {}",
                    k / n,
                    k % n,
                    h[k],
                    h3[k]
                );
            }
        }
        let mut worst: f64 = 0.0;
        for &i in cols {
            let col = gradient_difference(fr, eta, y, i, 1e-3 * eta[i].abs().max(1.0));
            for r in 0..n {
                let err = (h[r * n + i] - col[r]).abs() / col[r].abs().max(1.0);
                worst = worst.max(err);
                assert!(
                    err < 1e-6,
                    "{what}: H[{r},{i}] = {} vs differences {}",
                    h[r * n + i],
                    col[r]
                );
            }
        }
        worst
    }

    fn frame(d: usize, link: RotLink, turned: bool, rng: &mut ChaCha8Rng) -> GrFrame {
        let fr = GrFrame::new(&reference(d, rng), link);
        if turned {
            fr.with_basis(&rotation(d - 1, rng))
        } else {
            fr
        }
    }

    /// The exact Hessian matches differences of the reverse gradient and
    /// `loglik_gr_hess_fd_rev` at random points for every `d = 3..=11` (the
    /// stack kernels and the heap fallback), both links, with and without a
    /// basis rotation.
    #[test]
    fn hessian_matches_differences_at_random_points() {
        let mut rng = ChaCha8Rng::seed_from_u64(71);
        let mut worst: f64 = 0.0;
        for link in [RotLink::Stereo, RotLink::Exp] {
            for d in 3..=11 {
                for turned in [false, true] {
                    let fr = frame(d, link, turned, &mut rng);
                    for i in 0..2 {
                        let mut eta = mu(d, &mut rng);
                        eta.extend(coords(d, 2.5, &mut rng));
                        let y = reference(d, &mut rng);
                        let all: Vec<usize> = (0..eta.len()).collect();
                        let what = format!("{link:?} d={d} turned={turned} point {i}");
                        worst = worst.max(check_hess(&fr, &eta, &y, &all, true, &what));
                    }
                }
            }
        }
        println!("Hessian vs differences of the gradient: worst {worst:.1e}");
    }

    /// The same at the special points: group 1 zero, a zero rotation
    /// vector, a zero `κ`, every rotation vector zero, `γ = 0`, equal
    /// eigenvalues (`κ = 0` and `G_1 = 0` with rotations), for `Exp` a
    /// rotation by `π` and rotation vectors in each regime of the link's
    /// series (`s² < 10⁻¹⁰`, `s² < 1`), and `μ̂` at and next to the
    /// reference direction.
    #[test]
    fn hessian_matches_differences_at_the_special_points() {
        let mut rng = ChaCha8Rng::seed_from_u64(72);
        let mut worst: f64 = 0.0;
        for link in [RotLink::Stereo, RotLink::Exp] {
            for d in [3usize, 4, 5, 6, 8, 10] {
                for turned in [false, true] {
                    let fr = frame(d, link, turned, &mut rng);
                    let base: Vec<f64> = {
                        let mut e = mu(d, &mut rng);
                        e.extend(coords(d, 1.5, &mut rng));
                        e
                    };
                    let group = |gr: usize| d + gr * (gr + 1) / 2 - 1;
                    let mut points = vec![];
                    let mut g1 = base.clone();
                    g1[d] = 0.0;
                    g1[d + 1] = 0.0;
                    points.push(("group 1 zero", g1));
                    let mut all = base.clone();
                    all[d..].iter_mut().for_each(|g| *g = 0.0);
                    points.push(("gamma = 0", all));
                    if d >= 4 {
                        let o2 = group(2);
                        let mut pole = base.clone();
                        pole[o2 + 1] = 0.0;
                        pole[o2 + 2] = 0.0;
                        points.push(("zero rotation vector", pole));
                        let mut zero = base.clone();
                        zero[o2] = 0.0;
                        points.push(("zero kappa", zero));
                        let mut poles = base.clone();
                        let mut ties = base.clone();
                        ties[d] = 0.0;
                        ties[d + 1] = 0.0;
                        for gr in 2..=d - 2 {
                            let off = group(gr);
                            poles[off + 1..off + gr + 1]
                                .iter_mut()
                                .for_each(|g| *g = 0.0);
                            ties[off] = 0.0;
                        }
                        points.push(("all poles", poles));
                        points.push(("equal eigenvalues", ties));
                        if link == RotLink::Exp {
                            let scaled = |to: f64, base: &[f64]| {
                                let mut e = base.to_vec();
                                let w = &mut e[o2 + 1..o2 + 3];
                                let s = (w[0] * w[0] + w[1] * w[1]).sqrt();
                                w[0] *= to / s;
                                w[1] *= to / s;
                                e
                            };
                            points.push(("rotation by pi", scaled(std::f64::consts::PI, &base)));
                            points.push(("s = 0.9", scaled(0.9, &base)));
                            points.push(("s = 1e-6", scaled(1e-6, &base)));
                        }
                    }
                    let r = fr.lt_frame().reference().to_vec();
                    let mut at_m: Vec<f64> = r.iter().map(|x| 1.7 * x).collect();
                    at_m.extend(base[d..].iter());
                    points.push(("mu at m", at_m));
                    let mut near_m: Vec<f64> = r
                        .iter()
                        .map(|x| 1.7 * x + 1e-3 * rng.random_range(-1.0..1.0))
                        .collect();
                    near_m.extend(base[d..].iter());
                    points.push(("mu near m", near_m));
                    for (name, eta) in &points {
                        let y = reference(d, &mut rng);
                        let all: Vec<usize> = (0..eta.len()).collect();
                        let what = format!("{link:?} d={d} turned={turned} {name}");
                        worst = worst.max(check_hess(&fr, eta, &y, &all, true, &what));
                    }
                }
            }
        }
        println!("Hessian vs differences of the gradient: worst {worst:.1e}");
    }

    /// At the guarded points `μ = 0` and `μ̂ = −m` the frame is constant;
    /// the Hessian follows that convention: finite and symmetric, and its
    /// shape columns (along which the guard holds) match differences of
    /// the gradient. At `μ = 0` the mean block is `Φ_yy yyᵀ − I` (`Q` does
    /// not depend on `μ`).
    #[test]
    fn hessian_at_the_guarded_points() {
        let mut rng = ChaCha8Rng::seed_from_u64(73);
        for link in [RotLink::Stereo, RotLink::Exp] {
            for d in [3usize, 4, 5, 7, 10] {
                for turned in [false, true] {
                    let fr = frame(d, link, turned, &mut rng);
                    let mut eta = vec![0.0; d];
                    eta.extend(coords(d, 1.5, &mut rng));
                    let y = reference(d, &mut rng);
                    let shape: Vec<usize> = (d..eta.len()).collect();
                    let what = format!("{link:?} d={d} turned={turned}");
                    check_hess(&fr, &eta, &y, &shape, false, &format!("{what} mu = 0"));
                    let n = eta.len();
                    let (mut g, mut h) = (vec![0.0; n], vec![0.0; n * n]);
                    loglik_gr_hess(&fr, &eta, &y, &mut g, &mut h);
                    let k = (0..d)
                        .max_by(|&a, &b| y[a].abs().total_cmp(&y[b].abs()))
                        .unwrap();
                    let phi_yy = (h[k * n + k] + 1.0) / (y[k] * y[k]);
                    for r in 0..d {
                        for c in 0..d {
                            let expect = phi_yy * y[r] * y[c] - if r == c { 1.0 } else { 0.0 };
                            assert!(
                                (h[r * n + c] - expect).abs() < 1e-12 * (1.0 + expect.abs()),
                                "{what} mu = 0: H[{r},{c}] = {} vs {expect}",
                                h[r * n + c]
                            );
                        }
                    }
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
                    check_hess(&fr, &eta, &y, &shape, false, &format!("{what} antipode"));
                }
            }
        }
    }

    /// Next to the antipode of `m` the frame turns fast (its derivatives
    /// grow like `1/‖μ̂ + m‖`), and the Hessian still matches differences of
    /// the gradient taken at a step that resolves that scale.
    #[test]
    fn hessian_near_the_antipode() {
        let mut rng = ChaCha8Rng::seed_from_u64(74);
        for d in [3usize, 5, 8] {
            let fr = frame(d, RotLink::Stereo, true, &mut rng);
            let m = fr.lt_frame().reference().to_vec();
            let e0 = fr.lt_frame().tangent_basis()[..d].to_vec();
            for eps in [1e-2, 1e-4] {
                let mut eta: Vec<f64> = (0..d).map(|k| 2.0 * (-m[k] + eps * e0[k])).collect();
                eta.extend(coords(d, 1.0, &mut rng));
                let y = reference(d, &mut rng);
                let n = eta.len();
                let (mut g, mut h) = (vec![0.0; n], vec![0.0; n * n]);
                loglik_gr_hess(&fr, &eta, &y, &mut g, &mut h);
                for i in 0..n {
                    let step = if i < d { 1e-3 * eps } else { 1e-3 };
                    let col = gradient_difference(&fr, &eta, &y, i, step);
                    for r in 0..n {
                        let scale = col[r].abs().max(1.0);
                        assert!(
                            (h[r * n + i] - col[r]).abs() < 1e-6 * scale,
                            "d={d} eps={eps}: H[{r},{i}] = {} vs {}",
                            h[r * n + i],
                            col[r]
                        );
                    }
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

    /// Cost per observation of the exact `γ`-rot Hessian against the
    /// Hessian by differences of the reverse gradient and the reverse
    /// gradient alone (each with the value and the gradient).
    /// `cargo test --release --lib gamma_rot_adjoint::bench::hessian_cost -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn hessian_cost() {
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let n = 20_000;
        for d in [3usize, 5, 8, 10] {
            let k = dims::eta_len(d);
            let mut m: Vec<f64> = (0..d).map(|_| rng.random_range(-1.0..1.0)).collect();
            let nm = m.iter().map(|x| x * x).sum::<f64>().sqrt();
            m.iter_mut().for_each(|x| *x /= nm);
            let fr = GrFrame::new(&m, RotLink::Stereo);
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
            let (mut g, mut h) = (vec![0.0; k], vec![0.0; k * k]);
            let time = |f: &mut dyn FnMut(usize), m: usize| {
                let mut best = f64::INFINITY;
                for _ in 0..5 {
                    let t = Instant::now();
                    f(m);
                    best = best.min(t.elapsed().as_secs_f64());
                }
                best / m as f64 * 1e9
            };
            let mut acc = 0.0;
            let exact = time(
                &mut |m| {
                    for i in 0..m {
                        acc += loglik_gr_hess(&fr, black_box(&etas[i]), &ys[i], &mut g, &mut h);
                    }
                    black_box(&h);
                },
                n,
            );
            let fd = time(
                &mut |m| {
                    for i in 0..m {
                        acc +=
                            loglik_gr_hess_fd_rev(&fr, black_box(&etas[i]), &ys[i], &mut g, &mut h);
                    }
                    black_box(&h);
                },
                n / 10,
            );
            let rev = time(
                &mut |m| {
                    for i in 0..m {
                        acc += loglik_gr_grad_rev(&fr, black_box(&etas[i]), &ys[i], &mut g);
                    }
                    black_box(&g);
                },
                n,
            );
            black_box(acc);
            println!(
                "d={d:2} N={k:2}: gamma-rot exact Hessian {exact:6.0} ns/obs, differences {fd:6.0}, reverse gradient {rev:4.0}"
            );
        }
    }
}
