//! The parts of the exact Hessian that the shape charts in the transported
//! frame share: [`crate::logtangent`], [`crate::spectral`] and
//! [`crate::gamma_rot`] (through [`crate::gamma_rot_adjoint`]).
//!
//! Each chart writes the log-density of the unit vector `y` as
//! `log f = Φ(q, y·μ) − ‖μ‖²/2` with `q = t² + zᵀ W⁻¹ z`, `t = μ̂·y`, `z` the
//! coordinates of `y` in the frame `Ξ(μ̂) = R(m → μ̂) E` (turned by a fixed
//! `R_0` for the eigen charts), `Φ = −(d/2) log q + ψ(y·μ/√q)` and
//! `ψ(a) = a²/2 + log M_{d−1}(a)`. So
//!
//! ```text
//! ∇² log f = Φ_q ∇²q + Φ_qq ∇q∇qᵀ + Φ_qy (∇q eᵀ + e∇qᵀ) + Φ_yy eeᵀ − [I 0; 0 0]
//! ```
//!
//! with `e = (y, 0)` ([`Outer`], [`assemble`]). The chart supplies the
//! shape block `∂²q/∂θ²`, the cross block `∂²q/∂μ̂∂θ` and the part
//! `Jᵀ(∂²q/∂z²)J` of `A = ∂²q/∂μ̂²`, all with `μ̂` taken as a free vector,
//! through the frame Jacobian ([`frame_jacobian`])
//!
//! ```text
//! J = ∂z/∂μ̂ = B (−κI − (2/(v·v)) v uᵀ),   u = y − κv,  v = μ̂ + m,  κ = 2(v·y)/(v·v)
//! ```
//!
//! (`B = Eᵀ`, `R_0ᵀEᵀ` or, for the log-tangent kernel, `QᵀEᵀ`). This module
//! adds `2yyᵀ` from `t²` and the curvature of the Householder frame
//! ([`add_frame_curvature`]),
//!
//! ```text
//! Σ_k ū_k ∇²u_k = (2κ(ū·v)/(v·v)) I + (4(ū·v)/(v·v)²)(vuᵀ + uvᵀ) − (2/(v·v))(uūᵀ + ūuᵀ),
//! ```
//!
//! `ū = E z̄` the adjoint of `u`, and chains through `μ̂ = μ/‖μ‖` with
//! `P = I − μ̂μ̂ᵀ`: `∂²q/∂μ∂θ = P(∂²q/∂μ̂∂θ)/‖μ‖` ([`project_cross`]) and
//! `∂²q/∂μ² = (PAP − (g⊥μ̂ᵀ + μ̂g⊥ᵀ + (g·μ̂)P))/‖μ‖²` with `g = ∂q/∂μ̂`,
//! `g⊥ = Pg` ([`project_mean`]).
//!
//! The guard conventions of the gradients carry over: at `μ = 0` the
//! direction `μ̂` is the constant `m` (`mu_free = false`: `q` does not
//! depend on `μ`), and at the antipode `μ̂ = −m` the frame is constant
//! (`frame_free = false`: only `t²` depends on `μ`).

/// The transported frame at one observation, as the kernels' forward
/// passes leave it.
pub(crate) struct FrameState<'a> {
    /// `E`, `d × (d−1)` column-major.
    pub e: &'a [f64],
    pub y: &'a [f64],
    pub mu_hat: &'a [f64],
    /// The reflector `v = μ̂ + m`.
    pub v: &'a [f64],
    /// `u = y − κv`.
    pub u: &'a [f64],
    pub kappa: f64,
    /// `v·v`.
    pub vv: f64,
    /// `1/‖μ‖`.
    pub inv_n: f64,
    /// `μ ≠ 0`: `μ̂` depends on `μ`.
    pub mu_free: bool,
    /// `μ ≠ 0` and `μ̂ ≠ −m`: the frame depends on `μ`.
    pub frame_free: bool,
}

#[inline(always)]
pub(crate) fn dot(a: &[f64], b: &[f64]) -> f64 {
    let mut s = 0.0;
    for (x, y) in a.iter().zip(b) {
        s += x * y;
    }
    s
}

/// `J = B(−κI − (2/(v·v)) v uᵀ)` into `jt` (`p × d` row-major, row `c` is
/// `∂z_c/∂μ̂`), with `B = bᵀEᵀ` for the row-major `p × p` matrix `b`
/// (`B = Eᵀ` for `None`).
#[inline(always)]
pub(crate) fn frame_jacobian(f: &FrameState<'_>, b: Option<&[f64]>, jt: &mut [f64]) {
    let d = f.y.len();
    let p = d - 1;
    for c in 0..p {
        let kc = &mut jt[c * d..(c + 1) * d];
        match b {
            Some(q) => {
                kc.fill(0.0);
                for j in 0..p {
                    let qjc = q[j * p + c];
                    let ej = &f.e[j * d..(j + 1) * d];
                    for l in 0..d {
                        kc[l] += qjc * ej[l];
                    }
                }
            }
            None => kc.copy_from_slice(&f.e[c * d..(c + 1) * d]),
        }
        let cv = 2.0 * dot(kc, f.v) / f.vv;
        for l in 0..d {
            kc[l] = -f.kappa * kc[l] - cv * f.u[l];
        }
    }
}

/// `A ← 2yyᵀ` (upper triangle of the row-major `d × d` matrix `am`).
#[inline(always)]
pub(crate) fn init_mean(y: &[f64], am: &mut [f64]) {
    let d = y.len();
    for r in 0..d {
        for c in r..d {
            am[r * d + c] = 2.0 * y[r] * y[c];
        }
    }
}

/// `A += Σ_k ū_k ∇²u_k`, the curvature of the Householder frame (upper
/// triangle), for `ū = E z̄`.
#[inline(always)]
pub(crate) fn add_frame_curvature(f: &FrameState<'_>, ubar: &[f64], am: &mut [f64]) {
    let d = f.y.len();
    let uv = dot(ubar, f.v);
    let ci = 2.0 * f.kappa * uv / f.vv;
    let cuv = 4.0 * uv / (f.vv * f.vv);
    let cuu = 2.0 / f.vv;
    let (u, v) = (f.u, f.v);
    for r in 0..d {
        for c in r..d {
            am[r * d + c] +=
                cuv * (v[r] * u[c] + u[r] * v[c]) - cuu * (u[r] * ubar[c] + ubar[r] * u[c]);
        }
        am[r * d + r] += ci;
    }
}

/// Column `col` of the mean rows of `hess` (`n` columns):
/// `P x/‖μ‖` for `x = ∂²q/∂μ̂∂θ_i`.
#[inline(always)]
pub(crate) fn project_cross(f: &FrameState<'_>, x: &[f64], hess: &mut [f64], n: usize, col: usize) {
    let d = f.y.len();
    let pr = dot(&x[..d], f.mu_hat);
    for l in 0..d {
        hess[l * n + col] = (x[l] - pr * f.mu_hat[l]) * f.inv_n;
    }
}

/// The mean block of `hess` (upper triangle) from `A = ∂²q/∂μ̂²` (upper
/// triangle of `am`) and `g = ∂q/∂μ̂`: `(PAP − (g⊥μ̂ᵀ + μ̂g⊥ᵀ + (g·μ̂)P))/‖μ‖²`.
/// `xv` (length `d`) is scratch.
#[inline(always)]
pub(crate) fn project_mean(
    f: &FrameState<'_>,
    g: &[f64],
    am: &[f64],
    xv: &mut [f64],
    hess: &mut [f64],
    n: usize,
) {
    let d = f.y.len();
    let mu_hat = f.mu_hat;
    for r in 0..d {
        let mut acc = 0.0;
        for c in 0..d {
            let arc = if c >= r { am[r * d + c] } else { am[c * d + r] };
            acc += arc * mu_hat[c];
        }
        xv[r] = acc;
    }
    let mam = dot(mu_hat, xv);
    let gam = dot(g, mu_hat);
    let in2 = f.inv_n * f.inv_n;
    for r in 0..d {
        let gr = g[r] - gam * mu_hat[r];
        for c in r..d {
            let gc = g[c] - gam * mu_hat[c];
            let pap =
                am[r * d + c] - mu_hat[r] * xv[c] - xv[r] * mu_hat[c] + mam * mu_hat[r] * mu_hat[c];
            let prc = if r == c { 1.0 } else { 0.0 } - mu_hat[r] * mu_hat[c];
            hess[r * n + c] = in2 * (pap - (gr * mu_hat[c] + mu_hat[r] * gc + gam * prc));
        }
    }
}

/// The derivatives of `Φ(q, y·μ)` that [`assemble`] needs.
#[derive(Clone, Copy)]
pub(crate) struct Outer {
    pub phi_q: f64,
    pub phi_qq: f64,
    pub phi_qy: f64,
    pub phi_yy: f64,
}

impl Outer {
    /// From `q`, `√q`, `alpha = y·μ/√q`, `ψ′(alpha) = alpha + (log M_p)′`,
    /// `(log M_p)″` and the kernel's own `Φ_q = −(d + ψ′ alpha)/(2q)`
    /// (the factor its gradient was scaled by).
    #[inline(always)]
    pub(crate) fn new(d: usize, q: f64, sq: f64, alpha: f64, ab: f64, d2lmp: f64, qb: f64) -> Self {
        let (psi1, psi2) = (ab, 1.0 + d2lmp);
        let (a_q, a_y) = (-0.5 * alpha / q, 1.0 / sq);
        let (a_qq, a_qy) = (0.75 * alpha / (q * q), -0.5 / (q * sq));
        Outer {
            phi_q: qb,
            phi_qq: 0.5 * d as f64 / (q * q) + psi2 * a_q * a_q + psi1 * a_qq,
            phi_qy: psi2 * a_q * a_y + psi1 * a_qy,
            phi_yy: psi2 * a_y * a_y,
        }
    }
}

/// `∇² log f` from `∇²q` (upper triangle of `hess`, `n × n` row-major,
/// overwritten), `∇q` and `e = (y, 0)`; the result is exactly symmetric.
#[inline(always)]
pub(crate) fn assemble(o: &Outer, dq: &[f64], y: &[f64], hess: &mut [f64], n: usize) {
    let d = y.len();
    let (fq_, fqq, fqy, fyy) = (o.phi_q, o.phi_qq, o.phi_qy, o.phi_yy);
    let dq = &dq[..n];
    for (r, row) in hess.chunks_exact_mut(n).enumerate() {
        let dr = dq[r];
        let (cqq, cqy) = (fqq * dr, fqy * dr);
        if r < d {
            let yr = y[r];
            let (cyq, cyy) = (fqy * yr, fyy * yr);
            for c in r..d {
                row[c] = fq_ * row[c] + cqq * dq[c] + (cqy * y[c] + cyq * dq[c]) + cyy * y[c]
                    - if c == r { 1.0 } else { 0.0 };
            }
            for (h, &qc) in row[d..].iter_mut().zip(&dq[d..]) {
                *h = fq_ * *h + cqq * qc + cyq * qc;
            }
        } else {
            for (h, &qc) in row[r..].iter_mut().zip(&dq[r..]) {
                *h = fq_ * *h + cqq * qc;
            }
        }
    }
    mirror_upper(hess, n);
}

/// Copies the upper triangle of the row-major `n × n` matrix `h` into its
/// lower triangle, by 4 × 4 tiles below the diagonal (contiguous loads
/// and stores, about twice as fast as entry by entry).
#[inline(always)]
fn mirror_upper(h: &mut [f64], n: usize) {
    const B: usize = 4;
    let nb = n / B;
    for bi in 1..nb {
        for bj in 0..bi {
            let mut t = [[0.0; B]; B];
            for a in 0..B {
                let at = (bj * B + a) * n + bi * B;
                let src = &h[at..at + B];
                for b in 0..B {
                    t[b][a] = src[b];
                }
            }
            for (b, tb) in t.iter().enumerate() {
                let at = (bi * B + b) * n + bj * B;
                h[at..at + B].copy_from_slice(tb);
            }
        }
    }
    for bi in 0..nb {
        for r in bi * B + 1..bi * B + B {
            for c in bi * B..r {
                h[r * n + c] = h[c * n + r];
            }
        }
    }
    for r in nb * B..n {
        for c in 0..r {
            h[r * n + c] = h[c * n + r];
        }
    }
}
