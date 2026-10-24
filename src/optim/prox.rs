//! Proximal Newton for "smooth + weighted group-ℓ₁" objectives,
//!
//! ```text
//! F(θ) = f(θ) + λ Σ_g w_g(θ) ‖θ_g‖₂,     w_g(θ) = c_g^{−r(θ)},
//! ```
//!
//! the form of every LASSO-type penalty in this crate: singleton groups give
//! the elementwise (adaptive) LASSO, larger groups the (adaptive) group
//! LASSO, `r = 0` the plain versions, a fixed `r > 0` the adaptive ones, and
//! `r(θ) = ‖μ̃‖/‖μ(θ)‖` (a smooth function of the unpenalized `μ` block) the
//! RoC-weighted ones of Yu & Huang (Project 3).
//!
//! # Algorithm (proximal Newton; Lee, Sun & Saunders 2014)
//!
//! The smooth part is `h(θ) = f(θ)` plus, for RoC weights, the dependence of
//! the penalty on `μ`, whose gradient
//! `∂/∂μ λ Σ_g ‖θ_g‖ c_g^{−r} = λ Σ_g ‖θ_g‖ w_g ln(c_g) · ‖μ̃‖ μ/‖μ‖³`
//! is added to `∇f`. At the iterate `θ_k`, with `H` the Hessian of `f`
//! (shifted to be positive definite if necessary) and the weights frozen at
//! `w_g(θ_k)`, the model
//!
//! ```text
//! m(z) = ∇h(θ_k)ᵀ(z − θ_k) + ½ (z − θ_k)ᵀ H (z − θ_k) + λ Σ_g w_g ‖z_g‖
//! ```
//!
//! is minimised by FISTA (block soft-thresholding gives exact zeros). The
//! step `Δ = z* − θ_k` is accepted by backtracking on the true `F` with the
//! sufficient-decrease condition `F(θ + tΔ) ≤ F(θ) + σ t D`,
//! `D = ∇h·Δ + λ Σ_g w_g (‖θ_g + Δ_g‖ − ‖θ_g‖)`.
//!
//! Convergence is declared on the KKT residual (Clarke stationarity of `F`):
//! `|∂_i h|` for unpenalized coordinates; `‖∇_g h + λ w_g θ_g/‖θ_g‖‖` for
//! nonzero groups; `max(0, ‖∇_g h‖ − λ w_g)` for zero groups. At a fixed point
//! of the iteration this is exactly zero, including for RoC weights.
//!
//! Groups whose base `c_g` is zero get infinite weight when `r > 0`: they are
//! held at zero (the adaptive LASSO convention `|θ̃| = 0 ⇒ θ = 0`).

use nalgebra::{DMatrix, DVector};

/// A smooth objective with gradient and Hessian.
pub trait SmoothObjective: Sync {
    /// `f(θ)` and `∇f(θ)` (written into `grad`).
    fn value_grad(&self, theta: &[f64], grad: &mut [f64]) -> f64;
    /// `f(θ)`, `∇f(θ)` and the Hessian `∇²f(θ)` (row-major `n × n`).
    fn value_grad_hess(&self, theta: &[f64], grad: &mut [f64], hess: &mut [f64]) -> f64;
}

/// Exponent `r` of the adaptive weights `c_g^{−r}`.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum Exponent {
    /// A constant `r` (`0` = plain LASSO, `1` = the usual adaptive LASSO).
    Fixed(f64),
    /// `r = ‖μ̃‖ / ‖μ‖` (ratio of concentration), with `μ` the parameters at
    /// indices `mu` and `‖μ̃‖ = mu_tilde_norm` from the initial fit.
    RoC { mu: Vec<usize>, mu_tilde_norm: f64 },
}

/// The penalty `λ Σ_g c_g^{−r(θ)} ‖θ_g‖₂`.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct GroupPenalty {
    /// Disjoint groups of parameter indices (singletons = elementwise).
    pub groups: Vec<Vec<usize>>,
    /// Adaptive bases `c_g ≥ 0`, usually `‖θ̃_g‖` from an initial fit.
    pub base: Vec<f64>,
    pub exponent: Exponent,
    pub lambda: f64,
}

impl GroupPenalty {
    /// The exponent `r(θ)`.
    pub fn r(&self, theta: &[f64]) -> f64 {
        match &self.exponent {
            Exponent::Fixed(r) => *r,
            Exponent::RoC { mu, mu_tilde_norm } => {
                let nm = mu.iter().map(|&i| theta[i] * theta[i]).sum::<f64>().sqrt();
                mu_tilde_norm / nm
            }
        }
    }

    /// Group weights `w_g = c_g^{−r}` at `θ` (`∞` for `c_g = 0`, `r > 0`).
    pub fn weights(&self, theta: &[f64]) -> Vec<f64> {
        let r = self.r(theta);
        self.base
            .iter()
            .map(|&c| if r == 0.0 { 1.0 } else { c.powf(-r) })
            .collect()
    }

    /// `λ Σ_g w_g ‖θ_g‖` (infinite-weight groups contribute 0 at 0).
    pub fn value(&self, theta: &[f64]) -> f64 {
        let w = self.weights(theta);
        self.lambda
            * self
                .groups
                .iter()
                .zip(&w)
                .map(|(g, &wg)| {
                    let nrm = group_norm(theta, g);
                    if nrm == 0.0 { 0.0 } else { wg * nrm }
                })
                .sum::<f64>()
    }

    /// Add `∂/∂θ` of the penalty's smooth dependence on `μ` (RoC only).
    fn add_mu_gradient(&self, theta: &[f64], grad: &mut [f64]) {
        let Exponent::RoC { mu, mu_tilde_norm } = &self.exponent else {
            return;
        };
        let w = self.weights(theta);
        let nm2: f64 = mu.iter().map(|&i| theta[i] * theta[i]).sum();
        let nm = nm2.sqrt();
        // Σ_g ‖θ_g‖ w_g ln c_g
        let s: f64 = self
            .groups
            .iter()
            .zip(&w)
            .zip(&self.base)
            .map(|((g, &wg), &c)| {
                let nrm = group_norm(theta, g);
                if nrm == 0.0 || c == 0.0 {
                    0.0
                } else {
                    nrm * wg * c.ln()
                }
            })
            .sum();
        let coef = self.lambda * s * mu_tilde_norm / (nm2 * nm);
        for &i in mu {
            grad[i] += coef * theta[i];
        }
    }
}

fn group_norm(theta: &[f64], g: &[usize]) -> f64 {
    g.iter().map(|&i| theta[i] * theta[i]).sum::<f64>().sqrt()
}

/// Settings for [`prox_newton`].
#[derive(Clone, Debug)]
pub struct ProxConfig {
    /// Outer (Newton) iterations. Default 200.
    pub max_iter: usize,
    /// KKT residual tolerance, relative to `max(1, |F|)·1e-?`: stop when the
    /// residual is below `tol`. Default 1e-6.
    pub tol: f64,
    /// FISTA iterations per subproblem. Default 5000.
    pub max_inner: usize,
    /// Sufficient-decrease constant. Default 1e-4.
    pub sigma: f64,
}

impl Default for ProxConfig {
    fn default() -> Self {
        Self {
            max_iter: 200,
            tol: 1e-6,
            max_inner: 5000,
            sigma: 1e-4,
        }
    }
}

/// Outcome of [`prox_newton`].
#[derive(Clone, Debug)]
pub struct ProxResult {
    pub theta: Vec<f64>,
    /// `F(θ) = f(θ) + penalty`.
    pub objective: f64,
    /// `f(θ)`.
    pub smooth: f64,
    /// KKT residual at `θ`.
    pub kkt: f64,
    pub n_iter: usize,
    pub converged: bool,
}

/// KKT residual of `F` at `θ`, given `∇h(θ)` (smooth gradient incl. the RoC
/// `μ` term) and the weights at `θ`.
fn kkt_residual(theta: &[f64], gh: &[f64], pen: &GroupPenalty, w: &[f64]) -> f64 {
    let mut penalized = vec![false; theta.len()];
    let mut res = 0.0f64;
    for (g, &wg) in pen.groups.iter().zip(w) {
        for &i in g {
            penalized[i] = true;
        }
        let nrm = group_norm(theta, g);
        if !wg.is_finite() {
            continue; // held at zero
        }
        let lw = pen.lambda * wg;
        if nrm > 0.0 {
            let r2: f64 = g
                .iter()
                .map(|&i| (gh[i] + lw * theta[i] / nrm).powi(2))
                .sum();
            res = res.max(r2.sqrt());
        } else {
            let gn = g.iter().map(|&i| gh[i] * gh[i]).sum::<f64>().sqrt();
            res = res.max((gn - lw).max(0.0));
        }
    }
    for (i, &p) in penalized.iter().enumerate() {
        if !p {
            res = res.max(gh[i].abs());
        }
    }
    res
}

/// Minimise `f(θ) + λ Σ_g w_g(θ) ‖θ_g‖` from `theta0` by proximal Newton.
pub fn prox_newton(
    f: &dyn SmoothObjective,
    pen: &GroupPenalty,
    theta0: &[f64],
    cfg: &ProxConfig,
) -> ProxResult {
    let n = theta0.len();
    let mut theta = theta0.to_vec();
    // infinite-weight groups start (and stay) at zero
    let w0 = pen.weights(&theta);
    for (g, &wg) in pen.groups.iter().zip(&w0) {
        if !wg.is_finite() {
            for &i in g {
                theta[i] = 0.0;
            }
        }
    }
    let mut grad = vec![0.0; n];
    let mut hess = vec![0.0; n * n];
    let mut fs = f.value_grad_hess(&theta, &mut grad, &mut hess);
    let mut obj = fs + pen.value(&theta);
    let mut n_iter = 0;
    let mut kkt = f64::INFINITY;
    let mut converged = false;
    while n_iter < cfg.max_iter && obj.is_finite() {
        let w = pen.weights(&theta);
        let mut gh = grad.clone();
        pen.add_mu_gradient(&theta, &mut gh);
        kkt = kkt_residual(&theta, &gh, pen, &w);
        if kkt <= cfg.tol {
            converged = true;
            break;
        }
        n_iter += 1;
        // positive-definite model Hessian
        let h = positive_definite(&hess, n);
        let z = solve_model(&theta, &gh, &h, pen, &w, cfg.max_inner);
        let delta: Vec<f64> = z.iter().zip(&theta).map(|(a, b)| a - b).collect();
        let pen_now: f64 = pen
            .groups
            .iter()
            .zip(&w)
            .map(|(g, &wg)| {
                if !wg.is_finite() {
                    return 0.0;
                }
                wg * (group_norm(&z, g) - group_norm(&theta, g))
            })
            .sum::<f64>()
            * pen.lambda;
        let dd = gh.iter().zip(&delta).map(|(a, b)| a * b).sum::<f64>() + pen_now;
        if !(dd < 0.0) {
            // no predicted decrease: stationary up to the model's accuracy
            converged = kkt <= cfg.tol.sqrt();
            break;
        }
        // backtracking on the true objective
        let mut t = 1.0;
        let mut accepted = false;
        let mut cand = vec![0.0; n];
        for _ in 0..40 {
            for i in 0..n {
                cand[i] = theta[i] + t * delta[i];
            }
            let mut gc = vec![0.0; n];
            let fc = f.value_grad(&cand, &mut gc);
            let oc = fc + pen.value(&cand);
            if oc.is_finite() && oc <= obj + cfg.sigma * t * dd {
                accepted = true;
                break;
            }
            t *= 0.5;
        }
        if !accepted {
            break;
        }
        theta.copy_from_slice(&cand);
        fs = f.value_grad_hess(&theta, &mut grad, &mut hess);
        obj = fs + pen.value(&theta);
    }
    if converged {
        let w = pen.weights(&theta);
        let mut gh = grad.clone();
        pen.add_mu_gradient(&theta, &mut gh);
        kkt = kkt_residual(&theta, &gh, pen, &w);
    }
    ProxResult {
        theta,
        objective: obj,
        smooth: fs,
        kkt,
        n_iter,
        converged,
    }
}

/// Symmetrised `H`, shifted by `τ I` if needed so that it is positive
/// definite (`τ` grows geometrically from `1e-8·max|H_ii|`).
fn positive_definite(hess: &[f64], n: usize) -> DMatrix<f64> {
    let mut h = DMatrix::from_row_slice(n, n, hess);
    h = (&h + h.transpose()) * 0.5;
    if h.clone().cholesky().is_some() {
        return h;
    }
    let scale = (0..n)
        .map(|i| h[(i, i)].abs())
        .fold(0.0f64, f64::max)
        .max(1e-12);
    let mut tau = 1e-8 * scale;
    loop {
        let mut ht = h.clone();
        for i in 0..n {
            ht[(i, i)] += tau;
        }
        if ht.clone().cholesky().is_some() {
            return ht;
        }
        tau *= 10.0;
    }
}

/// Minimise the model `gᵀ(z−θ) + ½(z−θ)ᵀH(z−θ) + λ Σ w_g ‖z_g‖` by FISTA
/// with restarts.
fn solve_model(
    theta: &[f64],
    g: &[f64],
    h: &DMatrix<f64>,
    pen: &GroupPenalty,
    w: &[f64],
    max_inner: usize,
) -> Vec<f64> {
    let n = theta.len();
    // step 1/L, L = largest eigenvalue of H
    let l = nalgebra::SymmetricEigen::new(h.clone())
        .eigenvalues
        .iter()
        .fold(0.0f64, |a, &b| a.max(b))
        .max(1e-300);
    let step = 1.0 / l;
    let th = DVector::from_column_slice(theta);
    let gv = DVector::from_column_slice(g);
    let prox = |v: &mut DVector<f64>| {
        for (grp, &wg) in pen.groups.iter().zip(w) {
            if !wg.is_finite() {
                for &i in grp {
                    v[i] = 0.0;
                }
                continue;
            }
            let nrm = grp.iter().map(|&i| v[i] * v[i]).sum::<f64>().sqrt();
            let thr = step * pen.lambda * wg;
            let scale = if nrm <= thr { 0.0 } else { 1.0 - thr / nrm };
            for &i in grp {
                v[i] *= scale;
            }
        }
    };
    let model = |z: &DVector<f64>| -> f64 {
        let dz = z - &th;
        let quad = gv.dot(&dz) + 0.5 * dz.dot(&(h * &dz));
        let p: f64 = pen
            .groups
            .iter()
            .zip(w)
            .map(|(grp, &wg)| {
                if wg.is_finite() {
                    wg * grp.iter().map(|&i| z[i] * z[i]).sum::<f64>().sqrt()
                } else {
                    0.0
                }
            })
            .sum::<f64>()
            * pen.lambda;
        quad + p
    };
    let mut z = th.clone();
    prox(&mut z);
    let mut y = z.clone();
    let mut tk = 1.0f64;
    let mut m_prev = model(&z);
    for it in 0..max_inner {
        // gradient of the quadratic at y
        let grad_y = &gv + h * (&y - &th);
        let mut z_new = &y - grad_y * step;
        prox(&mut z_new);
        let m_new = model(&z_new);
        let change = (&z_new - &z).amax();
        if m_new > m_prev {
            // adaptive restart (O'Donoghue & Candès)
            tk = 1.0;
            y = z.clone();
            continue;
        }
        let t_next = 0.5 * (1.0 + (1.0 + 4.0 * tk * tk).sqrt());
        y = &z_new + (&z_new - &z) * ((tk - 1.0) / t_next);
        tk = t_next;
        z = z_new;
        m_prev = m_new;
        if change <= 1e-13 * (1.0 + z.amax()) && it > 2 {
            break;
        }
    }
    let _ = n;
    z.as_slice().to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// f(θ) = ½‖Aθ − b‖²
    struct LeastSquares {
        a: DMatrix<f64>,
        b: DVector<f64>,
    }
    impl SmoothObjective for LeastSquares {
        fn value_grad(&self, theta: &[f64], grad: &mut [f64]) -> f64 {
            let t = DVector::from_column_slice(theta);
            let r = &self.a * t - &self.b;
            let g = self.a.transpose() * &r;
            grad.copy_from_slice(g.as_slice());
            0.5 * r.dot(&r)
        }
        fn value_grad_hess(&self, theta: &[f64], grad: &mut [f64], hess: &mut [f64]) -> f64 {
            let f = self.value_grad(theta, grad);
            let h = self.a.transpose() * &self.a;
            let n = theta.len();
            for i in 0..n {
                for j in 0..n {
                    hess[i * n + j] = h[(i, j)];
                }
            }
            f
        }
    }

    #[test]
    fn orthogonal_lasso_is_soft_thresholding() {
        // A = I: solution θ_i = sign(b_i) max(|b_i| − λ, 0)
        let b = DVector::from_vec(vec![3.0, -0.5, 1.2, 0.0, -2.0]);
        let f = LeastSquares {
            a: DMatrix::identity(5, 5),
            b: b.clone(),
        };
        let pen = GroupPenalty {
            groups: (0..5).map(|i| vec![i]).collect(),
            base: vec![1.0; 5],
            exponent: Exponent::Fixed(0.0),
            lambda: 1.0,
        };
        let r = prox_newton(&f, &pen, &[0.0; 5], &ProxConfig::default());
        let want = [2.0, 0.0, 0.2, 0.0, -1.0];
        assert!(r.converged);
        for (a, b) in r.theta.iter().zip(&want) {
            assert!((a - b).abs() < 1e-10, "{:?}", r.theta);
        }
        assert_eq!(r.theta[1], 0.0);
        assert_eq!(r.theta[3], 0.0);
    }

    #[test]
    fn orthogonal_group_lasso_is_block_soft_thresholding() {
        let b = DVector::from_vec(vec![3.0, 4.0, 0.3, -0.4, 1.0]);
        let f = LeastSquares {
            a: DMatrix::identity(5, 5),
            b,
        };
        let pen = GroupPenalty {
            groups: vec![vec![0, 1], vec![2, 3]],
            base: vec![1.0, 1.0],
            exponent: Exponent::Fixed(0.0),
            lambda: 1.0,
        };
        let r = prox_newton(&f, &pen, &[1.0; 5], &ProxConfig::default());
        // group 1: ‖(3,4)‖ = 5 → scale 0.8; group 2: ‖(.3,−.4)‖ = .5 < 1 → 0
        let want = [2.4, 3.2, 0.0, 0.0, 1.0];
        for (a, b) in r.theta.iter().zip(&want) {
            assert!((a - b).abs() < 1e-10, "{:?}", r.theta);
        }
    }

    #[test]
    fn correlated_lasso_satisfies_kkt() {
        let a = DMatrix::from_fn(30, 6, |i, j| {
            (((i * 7 + j * 13) % 11) as f64 - 5.0) / 3.0 + if i % 6 == j { 1.0 } else { 0.0 }
        });
        let truth = DVector::from_vec(vec![2.0, 0.0, -1.0, 0.0, 0.0, 0.5]);
        let b = &a * &truth + DVector::from_fn(30, |i, _| 0.1 * ((i as f64) * 1.7).sin());
        let f = LeastSquares { a, b };
        let pen = GroupPenalty {
            groups: (0..6).map(|i| vec![i]).collect(),
            base: vec![1.0; 6],
            exponent: Exponent::Fixed(0.0),
            lambda: 3.0,
        };
        let r = prox_newton(&f, &pen, &[0.0; 6], &ProxConfig::default());
        assert!(r.converged, "kkt {}", r.kkt);
        assert!(r.kkt < 1e-6);
        assert!(r.theta.contains(&0.0), "{:?}", r.theta);
    }

    #[test]
    fn roc_weights_kkt_and_mu_gradient() {
        // θ = (μ0, μ1 | γ0, γ1): f = ½‖θ − b‖², RoC weights on γ with μ = θ[0..2]
        let b = DVector::from_vec(vec![2.0, 1.0, 0.8, -0.05]);
        let f = LeastSquares {
            a: DMatrix::identity(4, 4),
            b,
        };
        let pen = GroupPenalty {
            groups: vec![vec![2], vec![3]],
            base: vec![0.8, 0.05],
            exponent: Exponent::RoC {
                mu: vec![0, 1],
                mu_tilde_norm: 5f64.sqrt(),
            },
            lambda: 0.02,
        };
        let r = prox_newton(&f, &pen, &[2.0, 1.0, 0.8, -0.05], &ProxConfig::default());
        assert!(r.converged, "kkt {}", r.kkt);
        assert_eq!(
            r.theta[3], 0.0,
            "small entry with large adaptive weight is zeroed"
        );
        // the μ-gradient of the penalty matches finite differences
        let theta = [1.7, 1.1, 0.6, 0.2];
        let mut g = [0.0; 4];
        pen.add_mu_gradient(&theta, &mut g);
        for i in 0..2 {
            let h = 1e-6;
            let mut tp = theta;
            let mut tm = theta;
            tp[i] += h;
            tm[i] -= h;
            let fd = (pen.value(&tp) - pen.value(&tm)) / (2.0 * h);
            assert!((fd - g[i]).abs() < 1e-8, "i={i}: fd {fd} vs {}", g[i]);
        }
    }
}
