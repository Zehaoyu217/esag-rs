//! Numerical Hessian from central differences of the analytic gradient.
//!
//! Given an [`Objective`] with an exact gradient, the Hessian is approximated
//! column by column,
//!
//! ```text
//! H[:, j] ≈ (∇f(x + h_j e_j) − ∇f(x − h_j e_j)) / (2 h_j),
//! ```
//!
//! and then symmetrised, `H ← (H + Hᵀ)/2`. Because the gradient is exact, the
//! truncation error is `O(h²)` and the rounding error `O(ε‖∇f‖/h)`, so the
//! default step `h_j = 1e-5 · max(1, |x_j|)` gives roughly 1e-10 relative
//! accuracy for well-scaled problems. Each column costs two gradient
//! evaluations: `2n` in total.
//!
//! The result is the observed information matrix when `f` is a negative
//! log-likelihood, which is what the inference code needs for standard errors.
//!
//! # Example
//!
//! ```
//! use esag::optim::hessian::{hessian_fd, to_dmatrix};
//! use esag::optim::lbfgs::FnObjective;
//!
//! // f(x) = x₀² + 3 x₀ x₁ + 5 x₁²  ⇒  H = [[2, 3], [3, 10]]
//! let f = FnObjective(|x: &[f64], g: &mut [f64]| {
//!     g[0] = 2.0 * x[0] + 3.0 * x[1];
//!     g[1] = 3.0 * x[0] + 10.0 * x[1];
//!     x[0] * x[0] + 3.0 * x[0] * x[1] + 5.0 * x[1] * x[1]
//! });
//! let h = hessian_fd(&f, &[0.7, -0.2], None);
//! assert!((h[0] - 2.0).abs() < 1e-6 && (h[1] - 3.0).abs() < 1e-6);
//! assert!((h[2] - 3.0).abs() < 1e-6 && (h[3] - 10.0).abs() < 1e-6);
//!
//! let m = to_dmatrix(&h, 2);
//! let cov = m.try_inverse().expect("positive definite");
//! assert!((cov[(0, 0)] - 10.0 / 11.0).abs() < 1e-6);
//! ```

use nalgebra::DMatrix;
use rayon::prelude::*;

use super::lbfgs::Objective;

/// Relative finite-difference step used when [`hessian_fd`] is called with
/// `step = None`: `h_j = DEFAULT_REL_STEP · max(1, |x_j|)`.
pub const DEFAULT_REL_STEP: f64 = 1e-5;

/// Step for coordinate `j`.
fn fd_step(xj: f64, step: Option<f64>) -> f64 {
    match step {
        Some(h) => {
            assert!(
                h > 0.0 && h.is_finite(),
                "hessian_fd: step must be positive and finite, got {h}"
            );
            h
        }
        None => DEFAULT_REL_STEP * xj.abs().max(1.0),
    }
}

/// Column `j` of the (not yet symmetrised) Hessian.
fn fd_column(obj: &dyn Objective, x: &[f64], j: usize, step: Option<f64>) -> Vec<f64> {
    let n = x.len();
    let h = fd_step(x[j], step);
    let mut xs = x.to_vec();
    let mut g_plus = vec![0.0; n];
    let mut g_minus = vec![0.0; n];
    xs[j] = x[j] + h;
    obj.value_and_gradient(&xs, &mut g_plus);
    xs[j] = x[j] - h;
    obj.value_and_gradient(&xs, &mut g_minus);
    // Use the realised spacing of the two perturbed points rather than 2h to
    // absorb the rounding in x_j ± h.
    let denom = (x[j] + h) - (x[j] - h);
    g_plus
        .iter()
        .zip(&g_minus)
        .map(|(p, m)| (p - m) / denom)
        .collect()
}

/// Assemble columns into a row-major matrix and symmetrise.
fn assemble(cols: Vec<Vec<f64>>, n: usize) -> Vec<f64> {
    let mut h = vec![0.0; n * n];
    for (j, col) in cols.iter().enumerate() {
        for (i, v) in col.iter().enumerate() {
            h[i * n + j] = *v;
        }
    }
    for i in 0..n {
        for j in (i + 1)..n {
            let avg = 0.5 * (h[i * n + j] + h[j * n + i]);
            h[i * n + j] = avg;
            h[j * n + i] = avg;
        }
    }
    h
}

/// Symmetric numerical Hessian of `obj` at `x` from central differences of the
/// analytic gradient, returned row-major as an `n × n` slice (`H[i*n + j]`).
///
/// `H[:, j] ≈ (∇f(x + h_j e_j) − ∇f(x − h_j e_j)) / (2 h_j)`, then
/// `H ← (H + Hᵀ)/2`.
///
/// The step is `h_j = step.unwrap_or(1e-5 · max(1, |x_j|))`; see
/// [`DEFAULT_REL_STEP`]. With `step = Some(h)` the same absolute step is used
/// for every coordinate. Uses `2n` gradient evaluations, sequentially; see
/// [`hessian_fd_par`] for the parallel version.
///
/// Non-finite gradient components propagate into the result unchanged; the
/// caller should check `h.iter().all(f64::is_finite)` if that matters.
///
/// # Panics
///
/// If `step` is `Some(h)` with `h ≤ 0` or non-finite.
pub fn hessian_fd(obj: &dyn Objective, x: &[f64], step: Option<f64>) -> Vec<f64> {
    let n = x.len();
    let cols: Vec<Vec<f64>> = (0..n).map(|j| fd_column(obj, x, j, step)).collect();
    assemble(cols, n)
}

/// Same as [`hessian_fd`], computing the columns in parallel with rayon.
///
/// Worthwhile when a gradient evaluation costs more than a few microseconds
/// (e.g. a sum over many observations). The result is bit-identical to the
/// sequential version because every column is computed independently.
pub fn hessian_fd_par(obj: &dyn Objective, x: &[f64], step: Option<f64>) -> Vec<f64> {
    let n = x.len();
    let cols: Vec<Vec<f64>> = (0..n)
        .into_par_iter()
        .map(|j| fd_column(obj, x, j, step))
        .collect();
    assemble(cols, n)
}

/// View a row-major `n × n` slice (as returned by [`hessian_fd`]) as a
/// [`nalgebra::DMatrix`].
///
/// # Panics
///
/// If `h.len() != n * n`.
pub fn to_dmatrix(h: &[f64], n: usize) -> DMatrix<f64> {
    assert_eq!(
        h.len(),
        n * n,
        "to_dmatrix: expected {} elements for n = {n}, got {}",
        n * n,
        h.len()
    );
    DMatrix::from_row_slice(n, n, h)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optim::lbfgs::FnObjective;
    use crate::optim::lbfgs::test_objectives::{Quadratic, Rosenbrock};

    fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b).fold(0.0, |m, (x, y)| m.max((x - y).abs()))
    }

    #[test]
    fn quadratic_recovers_a() {
        let n = 7;
        let q = Quadratic::dense(n);
        let x: Vec<f64> = (0..n).map(|i| 0.3 * i as f64 - 1.0).collect();
        let h = hessian_fd(&q, &x, None);
        assert_eq!(h.len(), n * n);
        let err = max_abs_diff(&h, &q.a);
        eprintln!("quadratic n={n}: max |H - A| = {err:.2e}");
        assert!(err < 1e-6, "err = {err:.3e}");

        // Ill-conditioned diagonal, large coordinates (exercises the relative step).
        let q = Quadratic::ill_conditioned(5, 1e6);
        let x = [1e3, -2e3, 50.0, 0.0, 7.5];
        let h = hessian_fd(&q, &x, None);
        for i in 0..5 {
            for j in 0..5 {
                let scale = q.a[i * 5 + i].max(q.a[j * 5 + j]);
                let err = (h[i * 5 + j] - q.a[i * 5 + j]).abs();
                assert!(err <= 1e-6 * scale, "({i},{j}) err = {err:.3e}");
            }
        }
    }

    #[test]
    fn rosenbrock_at_minimum() {
        let h = hessian_fd(&Rosenbrock, &[1.0, 1.0], None);
        let exact = [802.0, -400.0, -400.0, 200.0];
        for (got, want) in h.iter().zip(&exact) {
            let rel = (got - want).abs() / want.abs();
            assert!(rel < 1e-4, "got {got}, want {want}, rel {rel:.2e}");
        }
        // Symmetry is exact after averaging.
        assert_eq!(h[1], h[2]);

        // Away from the minimum, compare against the analytic Hessian too:
        // H = [[1200 x² − 400 y + 2, −400 x], [−400 x, 200]].
        let (x, y) = (-1.2, 1.0);
        let h = hessian_fd(&Rosenbrock, &[x, y], Some(1e-6));
        let exact = [
            1200.0 * x * x - 400.0 * y + 2.0,
            -400.0 * x,
            -400.0 * x,
            200.0,
        ];
        for (got, want) in h.iter().zip(&exact) {
            assert!(
                (got - want).abs() < 1e-4 * want.abs(),
                "got {got}, want {want}"
            );
        }
    }

    #[test]
    fn parallel_matches_sequential() {
        let n = 12;
        let q = Quadratic::dense(n);
        let x: Vec<f64> = (0..n).map(|i| (i as f64).sin()).collect();
        let hs = hessian_fd(&q, &x, None);
        let hp = hessian_fd_par(&q, &x, None);
        assert_eq!(hs, hp);

        let hs = hessian_fd(&Rosenbrock, &[-1.2, 1.0, 0.3, 2.0], Some(1e-5));
        let hp = hessian_fd_par(&Rosenbrock, &[-1.2, 1.0, 0.3, 2.0], Some(1e-5));
        assert_eq!(hs, hp);
    }

    #[test]
    fn dmatrix_conversion_and_inverse() {
        let h = [2.0, 3.0, 3.0, 10.0];
        let m = to_dmatrix(&h, 2);
        assert_eq!(m.nrows(), 2);
        assert_eq!(m[(0, 1)], 3.0);
        assert_eq!(m[(1, 0)], 3.0);
        assert_eq!(m[(1, 1)], 10.0);
        let inv = m.try_inverse().expect("invertible");
        // det = 11; inverse = [[10, -3], [-3, 2]] / 11
        assert!((inv[(0, 0)] - 10.0 / 11.0).abs() < 1e-12);
        assert!((inv[(0, 1)] + 3.0 / 11.0).abs() < 1e-12);
    }

    #[test]
    fn works_through_dyn_objective_closure() {
        let f = FnObjective(|x: &[f64], g: &mut [f64]| {
            // f = exp(x0) + x0 x1 + x1³/3
            g[0] = x[0].exp() + x[1];
            g[1] = x[0] + x[1] * x[1];
            x[0].exp() + x[0] * x[1] + x[1].powi(3) / 3.0
        });
        let obj: &dyn Objective = &f;
        let x = [0.5, -1.5];
        let h = hessian_fd(obj, &x, None);
        let exact = [x[0].exp(), 1.0, 1.0, 2.0 * x[1]];
        assert!(max_abs_diff(&h, &exact) < 1e-7);
    }

    #[test]
    fn empty_input() {
        let h = hessian_fd(&Rosenbrock, &[], None);
        assert!(h.is_empty());
        assert_eq!(to_dmatrix(&h, 0).nrows(), 0);
    }

    #[test]
    #[should_panic(expected = "step must be positive")]
    fn bad_step_panics() {
        let _ = hessian_fd(&Rosenbrock, &[1.0, 1.0], Some(0.0));
    }

    #[test]
    #[should_panic(expected = "to_dmatrix")]
    fn bad_dmatrix_length_panics() {
        let _ = to_dmatrix(&[1.0, 2.0, 3.0], 2);
    }
}
