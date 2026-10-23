//! The `Scalar` abstraction: the ESAG math is written once, generically over
//! `S: Scalar`, and instantiated with `f64` (plain evaluation) or with
//! [`crate::dual::Dual<N>`] (forward-mode automatic differentiation, giving
//! exact gradients of the log-likelihood with respect to the linear predictor).
//!
//! Comparison operators on a `Scalar` compare the *value* (real part) only.

use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

pub trait Scalar:
    Copy
    + Clone
    + core::fmt::Debug
    + PartialOrd
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + Add<f64, Output = Self>
    + Sub<f64, Output = Self>
    + Mul<f64, Output = Self>
    + Div<f64, Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
    + DivAssign
    + Send
    + Sync
    + 'static
{
    /// Lift a constant (zero derivative).
    fn from_f64(v: f64) -> Self;
    /// The value (real part).
    fn re(&self) -> f64;
    fn zero() -> Self {
        Self::from_f64(0.0)
    }
    fn one() -> Self {
        Self::from_f64(1.0)
    }
    fn sqrt(self) -> Self;
    fn exp(self) -> Self;
    fn ln(self) -> Self;
    fn sin(self) -> Self;
    fn cos(self) -> Self;
    /// Inverse cosine; derivative is guarded (set to 0) at |x| >= 1.
    fn acos(self) -> Self;
    /// `self.atan2(other)` = atan2(y = self, x = other), as in `f64::atan2`.
    fn atan2(self, other: Self) -> Self;
    fn powf(self, p: f64) -> Self;
    fn powi(self, n: i32) -> Self;
    fn abs(self) -> Self;
    /// Standard normal CDF Φ(x).
    fn norm_cdf(self) -> Self;
    /// Standard normal density φ(x).
    fn norm_pdf(self) -> Self;
    /// `self * a + b` (fused when possible).
    fn mul_add(self, a: Self, b: Self) -> Self {
        self * a + b
    }
}

impl Scalar for f64 {
    #[inline]
    fn from_f64(v: f64) -> Self {
        v
    }
    #[inline]
    fn re(&self) -> f64 {
        *self
    }
    #[inline]
    fn sqrt(self) -> Self {
        f64::sqrt(self)
    }
    #[inline]
    fn exp(self) -> Self {
        f64::exp(self)
    }
    #[inline]
    fn ln(self) -> Self {
        f64::ln(self)
    }
    #[inline]
    fn sin(self) -> Self {
        f64::sin(self)
    }
    #[inline]
    fn cos(self) -> Self {
        f64::cos(self)
    }
    #[inline]
    fn acos(self) -> Self {
        f64::acos(self)
    }
    #[inline]
    fn atan2(self, other: Self) -> Self {
        f64::atan2(self, other)
    }
    #[inline]
    fn powf(self, p: f64) -> Self {
        f64::powf(self, p)
    }
    #[inline]
    fn powi(self, n: i32) -> Self {
        f64::powi(self, n)
    }
    #[inline]
    fn abs(self) -> Self {
        f64::abs(self)
    }
    #[inline]
    fn norm_cdf(self) -> Self {
        crate::stats::normal::norm_cdf(self)
    }
    #[inline]
    fn norm_pdf(self) -> Self {
        crate::stats::normal::norm_pdf(self)
    }
    #[inline]
    fn mul_add(self, a: Self, b: Self) -> Self {
        f64::mul_add(self, a, b)
    }
}
