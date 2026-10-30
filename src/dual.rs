//! Forward-mode automatic differentiation with dual numbers.
//!
//! [`Dual<N>`] carries a value (`re`) together with `N` partial derivatives
//! (`eps`), one per independent variable. Every arithmetic operation and
//! every elementary function of the [`Scalar`] trait propagates the
//! derivatives by the chain rule, so evaluating a function written
//! generically over `S: Scalar` at inputs seeded with [`Dual::variable`]
//! yields the exact gradient (to rounding) in a single forward pass.
//!
//! Derivative rules are guarded at the points where the true function is not
//! differentiable but the ESAG code needs a finite value:
//!
//! * `sqrt` at 0 → derivative 0 (subgradient; the isotropic kink γ = 0),
//! * `acos` at |x| ≥ 1 → derivative 0,
//! * `atan2(0, 0)` → derivative 0,
//! * `powf(p)` at 0 with p < 1 → derivative 0,
//! * `abs` at 0 → derivative 0.
//!
//! `ln` at 0 and division by zero propagate `inf`/`NaN` as usual.
//! Comparisons (`PartialEq`, `PartialOrd`) look at the value only.

use core::cmp::Ordering;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::scalar::Scalar;
use crate::stats::normal::{norm_cdf, norm_pdf};

/// A dual number with `N` derivative slots: `re + Σ_i eps[i]·ε_i`, `ε_i ε_j = 0`.
#[derive(Clone, Copy, Debug)]
pub struct Dual<const N: usize> {
    /// The value (real part).
    pub re: f64,
    /// Partial derivatives of the value with respect to the `N` seeded variables.
    pub eps: [f64; N],
}

impl<const N: usize> Dual<N> {
    /// A constant: value `v`, all derivatives zero.
    #[inline]
    pub const fn constant(v: f64) -> Self {
        Self {
            re: v,
            eps: [0.0; N],
        }
    }

    /// The `i`-th independent variable with value `v` (unit derivative in slot `i`).
    ///
    /// # Panics
    /// If `i >= N`.
    #[inline]
    pub fn variable(v: f64, i: usize) -> Self {
        assert!(i < N, "variable index {i} out of range for Dual<{N}>");
        let mut eps = [0.0; N];
        eps[i] = 1.0;
        Self { re: v, eps }
    }

    /// The derivative vector.
    #[inline]
    pub fn grad(&self) -> &[f64; N] {
        &self.eps
    }

    /// Chain rule helper: new value `re`, derivatives scaled by `scale`.
    #[inline]
    fn chain(self, re: f64, scale: f64) -> Self {
        let mut eps = self.eps;
        for e in eps.iter_mut() {
            *e *= scale;
        }
        Self { re, eps }
    }
}

/// Seeds `x` as the `N` independent variables: `out[i] = Dual::variable(x[i], i)`.
///
/// # Panics
/// If `x.len() != N`.
pub fn seed_variables<const N: usize>(x: &[f64]) -> Vec<Dual<N>> {
    assert_eq!(x.len(), N, "seed_variables::<{N}> needs exactly {N} values");
    x.iter()
        .enumerate()
        .map(|(i, &v)| Dual::variable(v, i))
        .collect()
}

/// Stack-allocated variant of [`seed_variables`].
///
/// # Panics
/// If `x.len() != N`.
pub fn seed_array<const N: usize>(x: &[f64]) -> [Dual<N>; N] {
    assert_eq!(x.len(), N, "seed_array::<{N}> needs exactly {N} values");
    let mut out = [Dual::<N>::constant(0.0); N];
    for (i, (o, &v)) in out.iter_mut().zip(x).enumerate() {
        *o = Dual::variable(v, i);
    }
    out
}

impl<const N: usize> Default for Dual<N> {
    fn default() -> Self {
        Self::constant(0.0)
    }
}

impl<const N: usize> From<f64> for Dual<N> {
    #[inline]
    fn from(v: f64) -> Self {
        Self::constant(v)
    }
}

// ----- comparisons: value only ---------------------------------------------

impl<const N: usize> PartialEq for Dual<N> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.re == other.re
    }
}

impl<const N: usize> PartialOrd for Dual<N> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.re.partial_cmp(&other.re)
    }
}

// ----- arithmetic: Dual ∘ Dual ----------------------------------------------

impl<const N: usize> Add for Dual<N> {
    type Output = Self;
    #[inline]
    fn add(mut self, rhs: Self) -> Self {
        self.re += rhs.re;
        for (a, b) in self.eps.iter_mut().zip(rhs.eps.iter()) {
            *a += *b;
        }
        self
    }
}

impl<const N: usize> Sub for Dual<N> {
    type Output = Self;
    #[inline]
    fn sub(mut self, rhs: Self) -> Self {
        self.re -= rhs.re;
        for (a, b) in self.eps.iter_mut().zip(rhs.eps.iter()) {
            *a -= *b;
        }
        self
    }
}

impl<const N: usize> Mul for Dual<N> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let mut eps = [0.0; N];
        for ((e, &a), &b) in eps.iter_mut().zip(self.eps.iter()).zip(rhs.eps.iter()) {
            *e = self.re * b + a * rhs.re;
        }
        Self {
            re: self.re * rhs.re,
            eps,
        }
    }
}

impl<const N: usize> Div for Dual<N> {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self {
        let re = self.re / rhs.re;
        let inv = 1.0 / rhs.re;
        let mut eps = [0.0; N];
        for ((e, &a), &b) in eps.iter_mut().zip(self.eps.iter()).zip(rhs.eps.iter()) {
            *e = (a - re * b) * inv;
        }
        Self { re, eps }
    }
}

impl<const N: usize> Neg for Dual<N> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        self.chain(-self.re, -1.0)
    }
}

// ----- arithmetic: Dual ∘ f64 -----------------------------------------------

impl<const N: usize> Add<f64> for Dual<N> {
    type Output = Self;
    #[inline]
    fn add(mut self, rhs: f64) -> Self {
        self.re += rhs;
        self
    }
}

impl<const N: usize> Sub<f64> for Dual<N> {
    type Output = Self;
    #[inline]
    fn sub(mut self, rhs: f64) -> Self {
        self.re -= rhs;
        self
    }
}

impl<const N: usize> Mul<f64> for Dual<N> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: f64) -> Self {
        self.chain(self.re * rhs, rhs)
    }
}

impl<const N: usize> Div<f64> for Dual<N> {
    type Output = Self;
    #[inline]
    fn div(self, rhs: f64) -> Self {
        let inv = 1.0 / rhs;
        self.chain(self.re * inv, inv)
    }
}

// ----- arithmetic: f64 ∘ Dual -----------------------------------------------

impl<const N: usize> Add<Dual<N>> for f64 {
    type Output = Dual<N>;
    #[inline]
    fn add(self, rhs: Dual<N>) -> Dual<N> {
        rhs + self
    }
}

impl<const N: usize> Sub<Dual<N>> for f64 {
    type Output = Dual<N>;
    #[inline]
    fn sub(self, rhs: Dual<N>) -> Dual<N> {
        rhs.chain(self - rhs.re, -1.0)
    }
}

impl<const N: usize> Mul<Dual<N>> for f64 {
    type Output = Dual<N>;
    #[inline]
    fn mul(self, rhs: Dual<N>) -> Dual<N> {
        rhs * self
    }
}

impl<const N: usize> Div<Dual<N>> for f64 {
    type Output = Dual<N>;
    #[inline]
    fn div(self, rhs: Dual<N>) -> Dual<N> {
        let re = self / rhs.re;
        rhs.chain(re, -re / rhs.re)
    }
}

// ----- assignment operators ---------------------------------------------------

impl<const N: usize> AddAssign for Dual<N> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<const N: usize> SubAssign for Dual<N> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<const N: usize> MulAssign for Dual<N> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<const N: usize> DivAssign for Dual<N> {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl<const N: usize> AddAssign<f64> for Dual<N> {
    #[inline]
    fn add_assign(&mut self, rhs: f64) {
        self.re += rhs;
    }
}

impl<const N: usize> SubAssign<f64> for Dual<N> {
    #[inline]
    fn sub_assign(&mut self, rhs: f64) {
        self.re -= rhs;
    }
}

impl<const N: usize> MulAssign<f64> for Dual<N> {
    #[inline]
    fn mul_assign(&mut self, rhs: f64) {
        *self = *self * rhs;
    }
}

impl<const N: usize> DivAssign<f64> for Dual<N> {
    #[inline]
    fn div_assign(&mut self, rhs: f64) {
        *self = *self / rhs;
    }
}

// ----- Scalar -----------------------------------------------------------------

impl<const N: usize> Scalar for Dual<N> {
    const DERIV_ORDER: u8 = 1;
    #[inline]
    fn from_f64(v: f64) -> Self {
        Self::constant(v)
    }

    #[inline]
    fn re(&self) -> f64 {
        self.re
    }

    /// `d sqrt(x) = dx / (2 sqrt(x))`; derivative 0 at `x == 0` (subgradient).
    #[inline]
    fn sqrt(self) -> Self {
        let r = self.re.sqrt();
        let scale = if self.re == 0.0 { 0.0 } else { 0.5 / r };
        self.chain(r, scale)
    }

    #[inline]
    fn exp(self) -> Self {
        let e = self.re.exp();
        self.chain(e, e)
    }

    #[inline]
    fn ln(self) -> Self {
        self.chain(self.re.ln(), 1.0 / self.re)
    }

    #[inline]
    fn sin(self) -> Self {
        let (s, c) = self.re.sin_cos();
        self.chain(s, c)
    }

    #[inline]
    fn cos(self) -> Self {
        let (s, c) = self.re.sin_cos();
        self.chain(c, -s)
    }

    /// `d acos(x) = -dx / sqrt(1 - x²)`; derivative 0 at `|x| >= 1`.
    #[inline]
    fn acos(self) -> Self {
        let scale = if self.re.abs() >= 1.0 {
            0.0
        } else {
            -1.0 / (1.0 - self.re * self.re).sqrt()
        };
        self.chain(self.re.acos(), scale)
    }

    /// `atan2(y = self, x = other)`; `d = (x dy - y dx) / (x² + y²)`, 0 at the origin.
    #[inline]
    fn atan2(self, other: Self) -> Self {
        let (y, x) = (self.re, other.re);
        let re = y.atan2(x);
        let denom = x * x + y * y;
        let mut eps = [0.0; N];
        if denom != 0.0 {
            let inv = 1.0 / denom;
            for ((e, &dy), &dx) in eps.iter_mut().zip(self.eps.iter()).zip(other.eps.iter()) {
                *e = (x * dy - y * dx) * inv;
            }
        }
        Self { re, eps }
    }

    /// `d x^p = p x^(p-1) dx`; derivative 0 at `x == 0` when `p < 1`.
    #[inline]
    fn powf(self, p: f64) -> Self {
        let scale = if self.re == 0.0 && p < 1.0 {
            0.0
        } else {
            p * self.re.powf(p - 1.0)
        };
        self.chain(self.re.powf(p), scale)
    }

    /// `d ∛x = ∛x/(3x) dx` (value from `f64::cbrt`); derivative 0 at 0.
    #[inline]
    fn cbrt(self) -> Self {
        let c = self.re.cbrt();
        let scale = if self.re == 0.0 {
            0.0
        } else {
            c / (3.0 * self.re)
        };
        self.chain(c, scale)
    }

    /// `d x^n = n x^(n-1) dx`; derivative 0 for `n == 0`.
    #[inline]
    fn powi(self, n: i32) -> Self {
        let scale = if n == 0 {
            0.0
        } else {
            f64::from(n) * self.re.powi(n - 1)
        };
        self.chain(self.re.powi(n), scale)
    }

    /// `d |x| = sign(x) dx`; derivative 0 at `x == 0`.
    #[inline]
    fn abs(self) -> Self {
        let scale = if self.re > 0.0 {
            1.0
        } else if self.re < 0.0 {
            -1.0
        } else {
            0.0
        };
        self.chain(self.re.abs(), scale)
    }

    /// `Φ(x)`, derivative `φ(x) dx`.
    #[inline]
    fn norm_cdf(self) -> Self {
        self.chain(norm_cdf(self.re), norm_pdf(self.re))
    }

    /// `φ(x)`, derivative `-x φ(x) dx`.
    #[inline]
    fn norm_pdf(self) -> Self {
        let f = norm_pdf(self.re);
        self.chain(f, -self.re * f)
    }

    #[inline]
    fn mul_add(self, a: Self, b: Self) -> Self {
        let mut eps = [0.0; N];
        for (((e, &ds), &da), &db) in eps
            .iter_mut()
            .zip(self.eps.iter())
            .zip(a.eps.iter())
            .zip(b.eps.iter())
        {
            *e = self.re * da + ds * a.re + db;
        }
        Self {
            re: self.re.mul_add(a.re, b.re),
            eps,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    type D1 = Dual<1>;
    type D2 = Dual<2>;

    const H: f64 = 1e-6;

    /// Central finite difference of a univariate f64 function.
    fn fd(f: impl Fn(f64) -> f64, x: f64) -> f64 {
        (f(x + H) - f(x - H)) / (2.0 * H)
    }

    fn check_unary(name: &str, f_dual: impl Fn(D1) -> D1, f_f64: impl Fn(f64) -> f64, xs: &[f64]) {
        for &x in xs {
            let d = f_dual(D1::variable(x, 0));
            assert_relative_eq!(d.re, f_f64(x), max_relative = 1e-14, epsilon = 1e-300);
            let num = fd(&f_f64, x);
            assert_relative_eq!(d.eps[0], num, max_relative = 1e-6, epsilon = 1e-8,);
            let _ = name;
        }
    }

    #[test]
    fn unary_rules_match_finite_differences() {
        check_unary("sqrt", |d| d.sqrt(), f64::sqrt, &[0.3, 1.0, 2.7, 10.0]);
        check_unary("exp", |d| d.exp(), f64::exp, &[-2.0, 0.0, 0.5, 3.0]);
        check_unary("ln", |d| d.ln(), f64::ln, &[0.1, 1.0, 4.5]);
        check_unary("sin", |d| d.sin(), f64::sin, &[-1.0, 0.0, 0.7, 2.5]);
        check_unary("cos", |d| d.cos(), f64::cos, &[-1.0, 0.0, 0.7, 2.5]);
        check_unary(
            "acos",
            |d| d.acos(),
            f64::acos,
            &[-0.9, -0.3, 0.0, 0.5, 0.95],
        );
        check_unary(
            "powf 0.5",
            |d| d.powf(0.5),
            |x| x.powf(0.5),
            &[0.2, 1.0, 3.0],
        );
        check_unary(
            "powf -1/3",
            |d| d.powf(-1.0 / 3.0),
            |x| x.powf(-1.0 / 3.0),
            &[0.2, 1.0, 3.0],
        );
        check_unary(
            "powf 2.5",
            |d| d.powf(2.5),
            |x| x.powf(2.5),
            &[0.2, 1.0, 3.0],
        );
        check_unary(
            "powi 3",
            |d| d.powi(3),
            |x| x.powi(3),
            &[-1.5, 0.0, 0.4, 2.0],
        );
        check_unary("powi -2", |d| d.powi(-2), |x| x.powi(-2), &[-1.5, 0.4, 2.0]);
        check_unary("abs", |d| d.abs(), f64::abs, &[-2.0, -0.1, 0.3, 5.0]);
        check_unary(
            "norm_cdf",
            |d| d.norm_cdf(),
            norm_cdf,
            &[-3.0, -0.5, 0.0, 1.2, 4.0],
        );
        check_unary(
            "norm_pdf",
            |d| d.norm_pdf(),
            norm_pdf,
            &[-3.0, -0.5, 0.0, 1.2, 4.0],
        );
        check_unary("neg", |d| -d, |x| -x, &[-1.0, 2.0]);
        check_unary("add f64", |d| d + 2.5, |x| x + 2.5, &[-1.0, 2.0]);
        check_unary("sub f64", |d| d - 2.5, |x| x - 2.5, &[-1.0, 2.0]);
        check_unary("mul f64", |d| d * 2.5, |x| x * 2.5, &[-1.0, 2.0]);
        check_unary("div f64", |d| d / 2.5, |x| x / 2.5, &[-1.0, 2.0]);
        check_unary("f64 sub", |d| 2.5 - d, |x| 2.5 - x, &[-1.0, 2.0]);
        check_unary("f64 div", |d| 2.5 / d, |x| 2.5 / x, &[-1.0, 2.0, 0.3]);
        check_unary("f64 mul", |d| 2.5 * d, |x| 2.5 * x, &[-1.0, 2.0]);
        check_unary("f64 add", |d| 2.5 + d, |x| 2.5 + x, &[-1.0, 2.0]);
    }

    #[test]
    fn binary_rules_match_finite_differences() {
        // f(x, y) for a few test functions, checking both partials.
        type F = fn(f64, f64) -> f64;
        type FD = fn(D2, D2) -> D2;
        let cases: [(FD, F); 6] = [
            (|a, b| a + b, |x, y| x + y),
            (|a, b| a - b, |x, y| x - y),
            (|a, b| a * b, |x, y| x * y),
            (|a, b| a / b, |x, y| x / y),
            (|a, b| a.atan2(b), |x, y| x.atan2(y)),
            (|a, b| a.mul_add(b, a * b), |x, y| x * y + x * y),
        ];
        let points = [
            (0.7, 1.3),
            (-2.0, 0.4),
            (1.5, -3.0),
            (0.0, 2.0),
            (-1.1, -0.9),
        ];
        for (fd_fn, f) in cases.iter() {
            for &(x, y) in &points {
                let d = fd_fn(D2::variable(x, 0), D2::variable(y, 1));
                assert_relative_eq!(d.re, f(x, y), max_relative = 1e-14, epsilon = 1e-300);
                let gx = (f(x + H, y) - f(x - H, y)) / (2.0 * H);
                let gy = (f(x, y + H) - f(x, y - H)) / (2.0 * H);
                assert_relative_eq!(d.eps[0], gx, max_relative = 1e-6, epsilon = 1e-8);
                assert_relative_eq!(d.eps[1], gy, max_relative = 1e-6, epsilon = 1e-8);
            }
        }
    }

    #[test]
    fn assign_ops_agree_with_binary_ops() {
        let a = D2::variable(1.7, 0);
        let b = D2::variable(-0.4, 1);
        let mut c = a;
        c += b;
        assert_eq!(c.eps, (a + b).eps);
        c = a;
        c -= b;
        assert_eq!(c.eps, (a - b).eps);
        c = a;
        c *= b;
        assert_eq!(c.eps, (a * b).eps);
        c = a;
        c /= b;
        assert_eq!(c.eps, (a / b).eps);
        c = a;
        c *= 3.0;
        assert_eq!(c.eps, (a * 3.0).eps);
        c = a;
        c /= 3.0;
        assert_eq!(c.eps, (a / 3.0).eps);
        c = a;
        c += 3.0;
        assert_eq!(c.re, a.re + 3.0);
        c = a;
        c -= 3.0;
        assert_eq!(c.re, a.re - 3.0);
    }

    #[test]
    fn guards_give_zero_derivative() {
        assert_eq!(D1::variable(0.0, 0).sqrt().eps[0], 0.0);
        assert_eq!(D1::variable(1.0, 0).acos().eps[0], 0.0);
        assert_eq!(D1::variable(-1.0, 0).acos().eps[0], 0.0);
        assert_eq!(D1::variable(0.0, 0).powf(0.5).eps[0], 0.0);
        assert_eq!(D1::variable(0.0, 0).abs().eps[0], 0.0);
        assert_eq!(D1::variable(2.0, 0).powi(0).eps[0], 0.0);
        let z = D2::variable(0.0, 0).atan2(D2::variable(0.0, 1));
        assert_eq!(z.re, 0.0);
        assert_eq!(z.eps, [0.0, 0.0]);
        // powf at 0 with p >= 1 is differentiable: d/dx x^1 = 1, d/dx x^2 = 0.
        assert_eq!(D1::variable(0.0, 0).powf(1.0).eps[0], 1.0);
        assert_eq!(D1::variable(0.0, 0).powf(2.0).eps[0], 0.0);
    }

    #[test]
    fn comparisons_use_value_only() {
        let a = D2::variable(1.0, 0);
        let b = D2::variable(1.0, 1);
        assert!(a == b);
        assert!(a <= b);
        assert!(D2::constant(0.5) < a);
        assert!(a > D2::constant(0.5));
        assert_eq!(a.partial_cmp(&b), Some(Ordering::Equal));
    }

    #[test]
    fn seeding_helpers() {
        let x = [1.0, 2.0, 3.0];
        let v = seed_variables::<3>(&x);
        let a = seed_array::<3>(&x);
        for i in 0..3 {
            assert_eq!(v[i].re, x[i]);
            assert_eq!(a[i].re, x[i]);
            for j in 0..3 {
                let expect = if i == j { 1.0 } else { 0.0 };
                assert_eq!(v[i].eps[j], expect);
                assert_eq!(a[i].eps[j], expect);
            }
        }
        let c: Dual<4> = 2.5.into();
        assert_eq!(c.re, 2.5);
        assert_eq!(c.eps, [0.0; 4]);
        assert_eq!(Dual::<2>::default().re, 0.0);
    }

    #[test]
    fn gradient_of_composite_function() {
        // f(x, y) = exp(x * y) / sqrt(x² + y²) + acos(x / 3) * norm_cdf(y)
        fn f<S: Scalar>(x: S, y: S) -> S {
            (x * y).exp() / (x * x + y * y).sqrt() + (x / 3.0).acos() * y.norm_cdf()
        }
        let (x, y) = (0.8, -1.3);
        let d = f(D2::variable(x, 0), D2::variable(y, 1));
        assert_relative_eq!(d.re, f(x, y), max_relative = 1e-14);
        let gx = (f(x + H, y) - f(x - H, y)) / (2.0 * H);
        let gy = (f(x, y + H) - f(x, y - H)) / (2.0 * H);
        assert_relative_eq!(d.eps[0], gx, max_relative = 1e-6);
        assert_relative_eq!(d.eps[1], gy, max_relative = 1e-6);
    }
}
