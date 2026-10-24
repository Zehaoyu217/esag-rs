//! Second-order forward-mode automatic differentiation ("hyper-dual" numbers
//! with `N` directions).
//!
//! [`HDual<N>`] carries a value (`re`), the gradient (`g`, `N` partials) and
//! the Hessian (`h`, `N × N`) with respect to `N` seeded variables. Every
//! arithmetic operation and every elementary function of the [`Scalar`] trait
//! propagates them by the second-order chain rule, so a function written
//! generically over `S: Scalar` and evaluated at inputs seeded with
//! [`HDual::variable`] / [`seed_hdual`] returns its exact value, gradient and
//! Hessian (to rounding) in one forward pass. [`loglik_eta_hess`] does this for
//! the ESAG log-density.
//!
//! # Rules
//!
//! For a unary `f`: `g = f'(x)·x.g`, `h = f'(x)·x.h + f''(x)·x.g⊗x.g`.
//! For a binary `f(u, v)`:
//! `h = f_u u.h + f_v v.h + f_uu u.g⊗u.g + f_vv v.g⊗v.g + f_uv (u.g⊗v.g + v.g⊗u.g)`,
//! applied in one pass (product, quotient and `atan2` all use it; `mul_add`
//! has its own three-input pass).
//!
//! # Guarded points
//!
//! As for [`Dual`](crate::dual::Dual), where the function is not
//! differentiable but the ESAG code needs a finite value, **both** the first
//! and second derivatives are set to zero:
//!
//! * `sqrt` at 0,
//! * `acos` at |x| ≥ 1,
//! * `atan2(0, 0)`,
//! * `powf(p)` at 0 with p < 1 (and also the infinite second derivative of
//!   `powf(p)` at 0 for 1 < p < 2; p = 1 and p = 2 are exact),
//! * `abs` at 0 (and `abs` of NaN, as in `Dual`).
//!
//! `ln` at 0 and division by zero propagate `inf`/`NaN` as usual.
//! Comparisons (`PartialEq`, `PartialOrd`) look at the value only.
//!
//! # Storage: only the upper triangle is maintained
//!
//! To halve the work, every operation updates only the upper triangle
//! `h[i][j]`, `j ≥ i`, of the Hessian; the strictly lower part is never read
//! and is left untouched (it stays exactly zero for values built from
//! [`HDual::constant`] / [`HDual::variable`]). Read the Hessian with
//! [`HDual::hess`], [`HDual::hessian_full`] or [`HDual::write_hessian`], which
//! mirror the upper triangle.
//!
//! BENCH_PLACEHOLDER
//!
//! # Size
//!
//! `HDual<N>` is `8 (N² + N + 1)` bytes on the stack (3.4 KB for N = 20), and
//! [`seed_hdual`] returns `N` of them (67 KB for N = 20). That is well within
//! the default 2 MB thread stack for N ≤ 20.

use core::cmp::Ordering;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::loglik::{loglik_eta, loglik_eta_grad};
use crate::scalar::Scalar;
use crate::stats::normal::{norm_cdf, norm_pdf};

/// `true`: operations update only the upper triangle `j ≥ i` of `h` (faster
/// for every `N` measured, see the module docs). `false`: full `N × N` updates.
const UPPER_ONLY: bool = true;

/// First column touched in row `i` of the Hessian.
#[inline(always)]
const fn row_start(i: usize) -> usize {
    if UPPER_ONLY { i } else { 0 }
}

/// `a * b + c`, fused where the hardware has FMA (every aarch64 CPU, x86-64
/// built with `+fma`); otherwise an ordinary multiply-add (a software `fma`
/// call would be far slower).
#[inline(always)]
fn fmadd(a: f64, b: f64, c: f64) -> f64 {
    #[cfg(any(target_arch = "aarch64", target_feature = "fma"))]
    {
        a.mul_add(b, c)
    }
    #[cfg(not(any(target_arch = "aarch64", target_feature = "fma")))]
    {
        a * b + c
    }
}

/// A second-order dual number in `N` directions: value, gradient and Hessian.
///
/// Only the upper triangle (`h[i][j]` with `j ≥ i`) of `h` is maintained; see
/// the [module docs](self).
#[derive(Clone, Copy, Debug)]
pub struct HDual<const N: usize> {
    /// The value (real part).
    pub re: f64,
    /// First partial derivatives with respect to the `N` seeded variables.
    pub g: [f64; N],
    /// Second partial derivatives, `h[i][j] = ∂²/∂x_i∂x_j`. **Only the upper
    /// triangle `j ≥ i` is valid**; use [`HDual::hessian_full`] for the full
    /// symmetric matrix.
    pub h: [[f64; N]; N],
}

impl<const N: usize> HDual<N> {
    /// A constant: value `v`, zero gradient and Hessian.
    #[inline]
    pub const fn constant(v: f64) -> Self {
        Self {
            re: v,
            g: [0.0; N],
            h: [[0.0; N]; N],
        }
    }

    /// The `i`-th independent variable with value `v` (unit gradient in slot
    /// `i`, zero Hessian).
    ///
    /// # Panics
    /// If `i >= N`.
    #[inline]
    pub fn variable(v: f64, i: usize) -> Self {
        assert!(i < N, "variable index {i} out of range for HDual<{N}>");
        let mut out = Self::constant(v);
        out.g[i] = 1.0;
        out
    }

    /// The gradient.
    #[inline]
    pub fn grad(&self) -> &[f64; N] {
        &self.g
    }

    /// The Hessian entry `(i, j)` (read from the upper triangle).
    ///
    /// # Panics
    /// If `i >= N` or `j >= N`.
    #[inline]
    pub fn hess(&self, i: usize, j: usize) -> f64 {
        if j >= i { self.h[i][j] } else { self.h[j][i] }
    }

    /// The full symmetric Hessian, built by mirroring the upper triangle.
    pub fn hessian_full(&self) -> [[f64; N]; N] {
        let mut out = self.h;
        for i in 1..N {
            for j in 0..i {
                out[i][j] = out[j][i];
            }
        }
        out
    }

    /// Writes the full symmetric Hessian into `out` (`N × N`, row-major),
    /// mirroring the upper triangle.
    ///
    /// # Panics
    /// If `out.len() != N * N`.
    pub fn write_hessian(&self, out: &mut [f64]) {
        assert_eq!(out.len(), N * N, "write_hessian needs an N×N buffer");
        for (i, (orow, hrow)) in out.chunks_exact_mut(N).zip(self.h.iter()).enumerate() {
            orow[i..].copy_from_slice(&hrow[i..]);
            for (o, hcol) in orow[..i].iter_mut().zip(self.h.iter()) {
                *o = hcol[i];
            }
        }
    }

    // ----- kernels (in place) ------------------------------------------------

    /// Unary chain rule in place: value `f0`, first derivative `f1`, second
    /// derivative `f2` of the outer function at `self.re`.
    #[inline]
    fn chain_mut(&mut self, f0: f64, f1: f64, f2: f64) {
        let g = &self.g;
        for (i, (row, &gi)) in self.h.iter_mut().zip(g.iter()).enumerate() {
            let s = row_start(i);
            let c = f2 * gi;
            for (hij, &gj) in row[s..].iter_mut().zip(&g[s..]) {
                *hij = fmadd(f1, *hij, c * gj);
            }
        }
        for gi in self.g.iter_mut() {
            *gi *= f1;
        }
        self.re = f0;
    }

    /// Hessian part of a binary rule in place, `self = u`, `v = other`:
    /// `h = fu u.h + fv v.h + c_u(i) u.g_j + c_v(i) v.g_j`, where
    /// `(c_u, c_v) = coef(u.g_i, v.g_i)` must equal
    /// `(f_uu u.g_i + f_uv v.g_i, f_vv v.g_i + f_uv u.g_i)`.
    /// The gradient and value are left to the caller.
    #[inline]
    fn hess_binary_mut(
        &mut self,
        v: &Self,
        fu: f64,
        fv: f64,
        coef: impl Fn(f64, f64) -> (f64, f64),
    ) {
        let ug = &self.g;
        let vg = &v.g;
        for (i, ((row, vrow), (&ugi, &vgi))) in self
            .h
            .iter_mut()
            .zip(v.h.iter())
            .zip(ug.iter().zip(vg.iter()))
            .enumerate()
        {
            let s = row_start(i);
            let (cu, cv) = coef(ugi, vgi);
            for (((hij, &vhij), &ugj), &vgj) in row[s..]
                .iter_mut()
                .zip(&vrow[s..])
                .zip(&ug[s..])
                .zip(&vg[s..])
            {
                *hij = fmadd(fu, *hij, fmadd(fv, vhij, fmadd(cu, ugj, cv * vgj)));
            }
        }
    }

    /// `self += v` in place.
    #[inline]
    fn add_mut(&mut self, v: &Self) {
        self.re += v.re;
        for (a, &b) in self.g.iter_mut().zip(v.g.iter()) {
            *a += b;
        }
        for (i, (row, vrow)) in self.h.iter_mut().zip(v.h.iter()).enumerate() {
            let s = row_start(i);
            for (a, &b) in row[s..].iter_mut().zip(&vrow[s..]) {
                *a += b;
            }
        }
    }

    /// `self -= v` in place.
    #[inline]
    fn sub_mut(&mut self, v: &Self) {
        self.re -= v.re;
        for (a, &b) in self.g.iter_mut().zip(v.g.iter()) {
            *a -= b;
        }
        for (i, (row, vrow)) in self.h.iter_mut().zip(v.h.iter()).enumerate() {
            let s = row_start(i);
            for (a, &b) in row[s..].iter_mut().zip(&vrow[s..]) {
                *a -= b;
            }
        }
    }

    /// Scales the derivatives by `c` and sets the value to `re`.
    #[inline]
    fn scale_mut(&mut self, re: f64, c: f64) {
        self.re = re;
        for a in self.g.iter_mut() {
            *a *= c;
        }
        for (i, row) in self.h.iter_mut().enumerate() {
            let s = row_start(i);
            for a in row[s..].iter_mut() {
                *a *= c;
            }
        }
    }

    /// `self *= v` in place.
    #[inline]
    fn mul_mut(&mut self, v: &Self) {
        let (ure, vre) = (self.re, v.re);
        // f_u = v, f_v = u, f_uu = f_vv = 0, f_uv = 1
        self.hess_binary_mut(v, vre, ure, |ugi, vgi| (vgi, ugi));
        for (a, &b) in self.g.iter_mut().zip(v.g.iter()) {
            *a = ure * b + *a * vre;
        }
        self.re = ure * vre;
    }

    /// `self /= v` in place.
    #[inline]
    fn div_mut(&mut self, v: &Self) {
        let re = self.re / v.re;
        let inv = 1.0 / v.re;
        // f = u/v: f_u = 1/v, f_v = −f/v, f_uu = 0, f_uv = −1/v², f_vv = 2f/v²
        let fuv = -inv * inv;
        let fvv = 2.0 * re * inv * inv;
        self.hess_binary_mut(v, inv, -re * inv, |ugi, vgi| {
            (fuv * vgi, fmadd(fvv, vgi, fuv * ugi))
        });
        for (a, &b) in self.g.iter_mut().zip(v.g.iter()) {
            *a = (*a - re * b) * inv;
        }
        self.re = re;
    }

    /// Zero gradient and (upper-triangle) Hessian, value `re`.
    #[inline]
    fn flat(&mut self, re: f64) {
        self.re = re;
        self.g = [0.0; N];
        for (i, row) in self.h.iter_mut().enumerate() {
            row[row_start(i)..].fill(0.0);
        }
    }
}

/// Seeds `x` as the `N` independent variables: `out[i] = HDual::variable(x[i], i)`.
///
/// # Panics
/// If `x.len() != N`.
pub fn seed_hdual<const N: usize>(x: &[f64]) -> [HDual<N>; N] {
    assert_eq!(x.len(), N, "seed_hdual::<{N}> needs exactly {N} values");
    core::array::from_fn(|i| HDual::variable(x[i], i))
}

impl<const N: usize> Default for HDual<N> {
    fn default() -> Self {
        Self::constant(0.0)
    }
}

impl<const N: usize> From<f64> for HDual<N> {
    #[inline]
    fn from(v: f64) -> Self {
        Self::constant(v)
    }
}

// ----- comparisons: value only ---------------------------------------------

impl<const N: usize> PartialEq for HDual<N> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.re == other.re
    }
}

impl<const N: usize> PartialOrd for HDual<N> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.re.partial_cmp(&other.re)
    }
}

// ----- arithmetic: HDual ∘ HDual ----------------------------------------------

impl<const N: usize> Add for HDual<N> {
    type Output = Self;
    #[inline]
    fn add(mut self, rhs: Self) -> Self {
        self.add_mut(&rhs);
        self
    }
}

impl<const N: usize> Sub for HDual<N> {
    type Output = Self;
    #[inline]
    fn sub(mut self, rhs: Self) -> Self {
        self.sub_mut(&rhs);
        self
    }
}

impl<const N: usize> Mul for HDual<N> {
    type Output = Self;
    #[inline]
    fn mul(mut self, rhs: Self) -> Self {
        self.mul_mut(&rhs);
        self
    }
}

impl<const N: usize> Div for HDual<N> {
    type Output = Self;
    #[inline]
    fn div(mut self, rhs: Self) -> Self {
        self.div_mut(&rhs);
        self
    }
}

impl<const N: usize> Neg for HDual<N> {
    type Output = Self;
    #[inline]
    fn neg(mut self) -> Self {
        self.scale_mut(-self.re, -1.0);
        self
    }
}

// ----- arithmetic: HDual ∘ f64 -----------------------------------------------

impl<const N: usize> Add<f64> for HDual<N> {
    type Output = Self;
    #[inline]
    fn add(mut self, rhs: f64) -> Self {
        self.re += rhs;
        self
    }
}

impl<const N: usize> Sub<f64> for HDual<N> {
    type Output = Self;
    #[inline]
    fn sub(mut self, rhs: f64) -> Self {
        self.re -= rhs;
        self
    }
}

impl<const N: usize> Mul<f64> for HDual<N> {
    type Output = Self;
    #[inline]
    fn mul(mut self, rhs: f64) -> Self {
        self.scale_mut(self.re * rhs, rhs);
        self
    }
}

impl<const N: usize> Div<f64> for HDual<N> {
    type Output = Self;
    #[inline]
    fn div(mut self, rhs: f64) -> Self {
        let inv = 1.0 / rhs;
        self.scale_mut(self.re * inv, inv);
        self
    }
}

// ----- arithmetic: f64 ∘ HDual -----------------------------------------------

impl<const N: usize> Add<HDual<N>> for f64 {
    type Output = HDual<N>;
    #[inline]
    fn add(self, rhs: HDual<N>) -> HDual<N> {
        rhs + self
    }
}

impl<const N: usize> Sub<HDual<N>> for f64 {
    type Output = HDual<N>;
    #[inline]
    fn sub(self, mut rhs: HDual<N>) -> HDual<N> {
        rhs.scale_mut(self - rhs.re, -1.0);
        rhs
    }
}

impl<const N: usize> Mul<HDual<N>> for f64 {
    type Output = HDual<N>;
    #[inline]
    fn mul(self, rhs: HDual<N>) -> HDual<N> {
        rhs * self
    }
}

impl<const N: usize> Div<HDual<N>> for f64 {
    type Output = HDual<N>;
    /// `c / x`: first derivative `−(c/x)/x`, second `2 (c/x)/x²`.
    #[inline]
    fn div(self, mut rhs: HDual<N>) -> HDual<N> {
        let x = rhs.re;
        let re = self / x;
        let f1 = -re / x;
        rhs.chain_mut(re, f1, -2.0 * f1 / x);
        rhs
    }
}

// ----- assignment operators ---------------------------------------------------

impl<const N: usize> AddAssign for HDual<N> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.add_mut(&rhs);
    }
}

impl<const N: usize> SubAssign for HDual<N> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.sub_mut(&rhs);
    }
}

impl<const N: usize> MulAssign for HDual<N> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        self.mul_mut(&rhs);
    }
}

impl<const N: usize> DivAssign for HDual<N> {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        self.div_mut(&rhs);
    }
}

impl<const N: usize> AddAssign<f64> for HDual<N> {
    #[inline]
    fn add_assign(&mut self, rhs: f64) {
        self.re += rhs;
    }
}

impl<const N: usize> SubAssign<f64> for HDual<N> {
    #[inline]
    fn sub_assign(&mut self, rhs: f64) {
        self.re -= rhs;
    }
}

impl<const N: usize> MulAssign<f64> for HDual<N> {
    #[inline]
    fn mul_assign(&mut self, rhs: f64) {
        self.scale_mut(self.re * rhs, rhs);
    }
}

impl<const N: usize> DivAssign<f64> for HDual<N> {
    #[inline]
    fn div_assign(&mut self, rhs: f64) {
        let inv = 1.0 / rhs;
        self.scale_mut(self.re * inv, inv);
    }
}

// ----- Scalar -----------------------------------------------------------------

impl<const N: usize> Scalar for HDual<N> {
    #[inline]
    fn from_f64(v: f64) -> Self {
        Self::constant(v)
    }

    #[inline]
    fn re(&self) -> f64 {
        self.re
    }

    /// `f' = 1/(2√x)`, `f'' = −f'/(2x)`; both 0 at `x == 0`.
    #[inline]
    fn sqrt(mut self) -> Self {
        let x = self.re;
        let r = x.sqrt();
        if x == 0.0 {
            self.flat(r);
        } else {
            let f1 = 0.5 / r;
            self.chain_mut(r, f1, -0.5 * f1 / x);
        }
        self
    }

    #[inline]
    fn exp(mut self) -> Self {
        let e = self.re.exp();
        self.chain_mut(e, e, e);
        self
    }

    /// `f' = 1/x`, `f'' = −1/x²`.
    #[inline]
    fn ln(mut self) -> Self {
        let f1 = 1.0 / self.re;
        self.chain_mut(self.re.ln(), f1, -f1 * f1);
        self
    }

    #[inline]
    fn sin(mut self) -> Self {
        let (s, c) = self.re.sin_cos();
        self.chain_mut(s, c, -s);
        self
    }

    #[inline]
    fn cos(mut self) -> Self {
        let (s, c) = self.re.sin_cos();
        self.chain_mut(c, -s, -c);
        self
    }

    /// `f' = −1/√(1−x²)`, `f'' = −x/(1−x²)^{3/2}`; both 0 at `|x| >= 1`.
    #[inline]
    fn acos(mut self) -> Self {
        let x = self.re;
        if x.abs() >= 1.0 {
            self.flat(x.acos());
        } else {
            let w = 1.0 - x * x;
            let f1 = -1.0 / w.sqrt();
            self.chain_mut(x.acos(), f1, f1 * x / w);
        }
        self
    }

    /// `atan2(y = self, x = other)`. With `r² = x² + y²`:
    /// `f_y = x/r²`, `f_x = −y/r²`, `f_yy = −2xy/r⁴`, `f_xx = 2xy/r⁴`,
    /// `f_xy = (y² − x²)/r⁴`; all derivatives 0 at the origin.
    #[inline]
    fn atan2(mut self, other: Self) -> Self {
        let (y, x) = (self.re, other.re);
        let re = y.atan2(x);
        let denom = x * x + y * y;
        if denom == 0.0 {
            self.flat(re);
            return self;
        }
        let inv = 1.0 / denom;
        let inv2 = inv * inv;
        let fyy = -2.0 * x * y * inv2;
        let fxx = -fyy;
        let fxy = (y - x) * (y + x) * inv2;
        self.hess_binary_mut(&other, x * inv, -y * inv, |ygi, xgi| {
            (fmadd(fyy, ygi, fxy * xgi), fmadd(fxx, xgi, fxy * ygi))
        });
        for (dy, &dx) in self.g.iter_mut().zip(other.g.iter()) {
            *dy = (x * *dy - y * dx) * inv;
        }
        self.re = re;
        self
    }

    /// `f' = p x^{p−1}`, `f'' = (p−1) f'/x`. At `x == 0`: both 0 for `p < 1`;
    /// `f' = 1, f'' = 0` for `p = 1`; `f'' = 2` for `p = 2`; and `f'' = 0`
    /// otherwise (exact for `p > 2`, a guard for the infinite value when
    /// `1 < p < 2`).
    #[inline]
    fn powf(mut self, p: f64) -> Self {
        let x = self.re;
        let re = x.powf(p);
        if x == 0.0 {
            if p < 1.0 {
                self.flat(re);
            } else {
                let f1 = if p == 1.0 { 1.0 } else { 0.0 };
                let f2 = if p == 2.0 { 2.0 } else { 0.0 };
                self.chain_mut(re, f1, f2);
            }
        } else {
            let f1 = p * x.powf(p - 1.0);
            self.chain_mut(re, f1, (p - 1.0) * f1 / x);
        }
        self
    }

    /// `f' = n x^{n−1}`, `f'' = n(n−1) x^{n−2}` (exactly 0 for `n = 0, 1`).
    #[inline]
    fn powi(mut self, n: i32) -> Self {
        let x = self.re;
        let f1 = if n == 0 {
            0.0
        } else {
            f64::from(n) * x.powi(n - 1)
        };
        let f2 = if n == 0 || n == 1 {
            0.0
        } else {
            f64::from(n) * f64::from(n - 1) * x.powi(n.saturating_sub(2))
        };
        self.chain_mut(x.powi(n), f1, f2);
        self
    }

    /// `f' = sign(x)`, `f'' = 0`; both 0 at `x == 0` (and for NaN).
    #[inline]
    fn abs(self) -> Self {
        if self.re > 0.0 {
            self
        } else if self.re < 0.0 {
            -self
        } else {
            let mut out = self;
            out.flat(self.re.abs());
            out
        }
    }

    /// `Φ(x)`: `f' = φ(x)`, `f'' = −x φ(x)`.
    #[inline]
    fn norm_cdf(mut self) -> Self {
        let x = self.re;
        let f = norm_pdf(x);
        self.chain_mut(norm_cdf(x), f, -x * f);
        self
    }

    /// `φ(x)`: `f' = −x φ(x)`, `f'' = (x² − 1) φ(x)`.
    #[inline]
    fn norm_pdf(mut self) -> Self {
        let x = self.re;
        let f = norm_pdf(x);
        self.chain_mut(f, -x * f, (x * x - 1.0) * f);
        self
    }

    /// `self * a + b` in one pass:
    /// `h = a.re s.h + s.re a.h + b.h + s.g⊗a.g + a.g⊗s.g`.
    #[inline]
    fn mul_add(mut self, a: Self, b: Self) -> Self {
        let (sre, are) = (self.re, a.re);
        {
            let sg = &self.g;
            let ag = &a.g;
            for (i, (((row, arow), brow), (&sgi, &agi))) in self
                .h
                .iter_mut()
                .zip(a.h.iter())
                .zip(b.h.iter())
                .zip(sg.iter().zip(ag.iter()))
                .enumerate()
            {
                let s = row_start(i);
                for ((((hij, &ahij), &bhij), &sgj), &agj) in row[s..]
                    .iter_mut()
                    .zip(&arow[s..])
                    .zip(&brow[s..])
                    .zip(&sg[s..])
                    .zip(&ag[s..])
                {
                    *hij = fmadd(
                        are,
                        *hij,
                        fmadd(sre, ahij, fmadd(agi, sgj, fmadd(sgi, agj, bhij))),
                    );
                }
            }
        }
        for ((e, &da), &db) in self.g.iter_mut().zip(a.g.iter()).zip(b.g.iter()) {
            *e = sre * da + *e * are + db;
        }
        self.re = sre.mul_add(are, b.re);
        self
    }
}

// ----- the ESAG log-density with its exact Hessian ---------------------------

#[inline]
fn hess_hdual<const N: usize>(eta: &[f64], y: &[f64], grad: &mut [f64], hess: &mut [f64]) -> f64 {
    let x: [HDual<N>; N] = seed_hdual(eta);
    let v = loglik_eta(&x, y);
    grad.copy_from_slice(&v.g);
    v.write_hessian(hess);
    v.re
}

/// Central differences of the exact gradient, symmetrised.
fn hess_fd(eta: &[f64], y: &[f64], grad: &mut [f64], hess: &mut [f64]) -> f64 {
    let n = eta.len();
    let f = loglik_eta_grad(eta, y, grad);
    let mut e = eta.to_vec();
    let mut gp = vec![0.0; n];
    let mut gm = vec![0.0; n];
    for i in 0..n {
        let h = 1e-5 * eta[i].abs().max(1.0);
        let (xp, xm) = (eta[i] + h, eta[i] - h);
        e[i] = xp;
        loglik_eta_grad(&e, y, &mut gp);
        e[i] = xm;
        loglik_eta_grad(&e, y, &mut gm);
        e[i] = eta[i];
        let inv = 1.0 / (xp - xm);
        for j in 0..n {
            hess[j * n + i] = (gp[j] - gm[j]) * inv;
        }
    }
    for i in 0..n {
        for j in i + 1..n {
            let s = 0.5 * (hess[i * n + j] + hess[j * n + i]);
            hess[i * n + j] = s;
            hess[j * n + i] = s;
        }
    }
    f
}

/// Log-density, exact gradient and exact Hessian with respect to `eta`.
///
/// Returns `log f(y; eta)` (as [`loglik_eta`]), writes the gradient into
/// `grad` (length `N = eta.len()`) and the full symmetric Hessian into `hess`
/// (`N × N`, row-major). Uses [`HDual<N>`] for `d = 3..=6` (`eta` lengths 5,
/// 9, 14, 20) and central differences of [`loglik_eta_grad`] otherwise
/// (step `1e-5·max(1, |eta_i|)`, symmetrised).
///
/// # Panics
/// If `grad.len() != eta.len()`, `hess.len() != eta.len()²`, or `eta` and `y`
/// have inconsistent lengths (as [`loglik_eta`]).
pub fn loglik_eta_hess(eta: &[f64], y: &[f64], grad: &mut [f64], hess: &mut [f64]) -> f64 {
    let n = eta.len();
    assert_eq!(grad.len(), n, "grad must have length eta.len()");
    assert_eq!(hess.len(), n * n, "hess must have length eta.len()²");
    match n {
        5 => hess_hdual::<5>(eta, y, grad, hess),
        9 => hess_hdual::<9>(eta, y, grad, hess),
        14 => hess_hdual::<14>(eta, y, grad, hess),
        20 => hess_hdual::<20>(eta, y, grad, hess),
        _ => hess_fd(eta, y, grad, hess),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dual::{Dual, seed_array};

    type H2 = HDual<2>;

    /// The second variable sits at `B0` in every test point of the unary rules.
    const B0: f64 = 0.3;

    /// `u(a, b) = a + 0.2 (b − B0) a + 0.1 (b − B0)²`: equal to `a` at `b = B0`,
    /// with gradient `(1, 0.2 a)` and Hessian `[[0, 0.2], [0.2, 0.2]]` there, so
    /// `f(u)` exercises both terms of the unary rule.
    fn inner<S: Scalar>(v: &[S]) -> S {
        let t = v[1] - B0;
        v[0] + t * v[0] * 0.2 + t * t * 0.1
    }

    /// Two non-linear inner functions of `(a, b)` for the binary rules.
    fn inner2<S: Scalar>(v: &[S]) -> (S, S) {
        let (a, b) = (v[0], v[1]);
        (
            a + a * b * 0.3 - b * b * 0.1,
            b - a * a * 0.2 + a * b * 0.15,
        )
    }

    /// Gradient of `f` at `x` by forward-mode `Dual<M>`.
    fn dual_grad<const M: usize>(f: fn(&[Dual<M>]) -> Dual<M>, x: &[f64; M]) -> [f64; M] {
        f(&seed_array::<M>(x)).eps
    }

    /// Column `j` of the Hessian: Richardson-extrapolated central differences
    /// of the exact `Dual` gradient (error O(h⁴)).
    fn fd_hess_col<const M: usize>(
        f: fn(&[Dual<M>]) -> Dual<M>,
        x: &[f64; M],
        j: usize,
        h: f64,
    ) -> [f64; M] {
        let cd = |h: f64| {
            let (mut xp, mut xm) = (*x, *x);
            xp[j] += h;
            xm[j] -= h;
            let (gp, gm) = (dual_grad(f, &xp), dual_grad(f, &xm));
            let mut out = [0.0; M];
            for i in 0..M {
                out[i] = (gp[i] - gm[i]) / (xp[j] - xm[j]);
            }
            out
        };
        let (d1, d2) = (cd(h), cd(0.5 * h));
        let mut out = [0.0; M];
        for i in 0..M {
            out[i] = (4.0 * d2[i] - d1[i]) / 3.0;
        }
        out
    }

    fn close(a: f64, b: f64, rel: f64, abs: f64) -> bool {
        (a - b).abs() <= rel * b.abs().max(a.abs()) + abs
    }

    /// Value vs `f64`, gradient vs `Dual`, Hessian vs differences of the
    /// `Dual` gradient, plus the upper-triangle invariant.
    fn check<const M: usize>(
        name: &str,
        fh: fn(&[HDual<M>]) -> HDual<M>,
        fd: fn(&[Dual<M>]) -> Dual<M>,
        ff: fn(&[f64]) -> f64,
        x: &[f64; M],
    ) {
        let r = fh(&seed_hdual::<M>(x));
        let v = ff(x);
        assert!(
            close(r.re, v, 1e-14, 1e-300),
            "{name} at {x:?}: value {} vs {v}",
            r.re
        );
        let g = dual_grad(fd, x);
        for i in 0..M {
            assert!(
                close(r.g[i], g[i], 1e-13, 1e-15),
                "{name} at {x:?}: g[{i}] {} vs Dual {}",
                r.g[i],
                g[i]
            );
        }
        for j in 0..M {
            let col = fd_hess_col(fd, x, j, 1e-3 * x[j].abs().max(1.0));
            for i in 0..M {
                assert!(
                    close(r.hess(i, j), col[i], 1e-6, 1e-8),
                    "{name} at {x:?}: h[{i}][{j}] {} vs fd {}",
                    r.hess(i, j),
                    col[i]
                );
            }
        }
        for i in 0..M {
            for j in 0..i {
                assert_eq!(r.h[i][j], 0.0, "{name}: lower triangle touched");
            }
        }
    }

    const MINUS_THIRD: f64 = -1.0 / 3.0;

    // `where f64: …` lets the cases use `f64 ∘ S` operators too.
    macro_rules! cases {
        ($($name:ident = |$v:ident| $body:expr;)*) => {
            $(
                fn $name<S>($v: &[S]) -> S
                where
                    S: Scalar + AddAssign<f64> + SubAssign<f64> + MulAssign<f64> + DivAssign<f64>,
                    f64: Add<S, Output = S>
                        + Sub<S, Output = S>
                        + Mul<S, Output = S>
                        + Div<S, Output = S>,
                {
                    $body
                }
            )*
        };
    }

    cases! {
        u_sqrt = |v| inner(v).sqrt();
        u_exp = |v| inner(v).exp();
        u_ln = |v| inner(v).ln();
        u_sin = |v| inner(v).sin();
        u_cos = |v| inner(v).cos();
        u_acos = |v| inner(v).acos();
        u_powf_half = |v| inner(v).powf(0.5);
        u_powf_m13 = |v| inner(v).powf(MINUS_THIRD);
        u_powf_1 = |v| inner(v).powf(1.0);
        u_powf_15 = |v| inner(v).powf(1.5);
        u_powf_2 = |v| inner(v).powf(2.0);
        u_powf_25 = |v| inner(v).powf(2.5);
        u_powi_0 = |v| inner(v).powi(0);
        u_powi_1 = |v| inner(v).powi(1);
        u_powi_2 = |v| inner(v).powi(2);
        u_powi_3 = |v| inner(v).powi(3);
        u_powi_m2 = |v| inner(v).powi(-2);
        u_abs = |v| inner(v).abs();
        u_cdf = |v| inner(v).norm_cdf();
        u_pdf = |v| inner(v).norm_pdf();
        u_neg = |v| -inner(v);
        u_add_f = |v| inner(v) + 2.5;
        u_sub_f = |v| inner(v) - 2.5;
        u_mul_f = |v| inner(v) * 2.5;
        u_div_f = |v| inner(v) / 2.5;
        f_add_u = |v| 2.5 + inner(v);
        f_sub_u = |v| 2.5 - inner(v);
        f_mul_u = |v| 2.5 * inner(v);
        f_div_u = |v| 2.5 / inner(v);
        u_from = |v| inner(v) * S::from_f64(1.7) + S::one() - S::zero();
        b_add = |v| { let (u, w) = inner2(v); u + w };
        b_sub = |v| { let (u, w) = inner2(v); u - w };
        b_mul = |v| { let (u, w) = inner2(v); u * w };
        b_div = |v| { let (u, w) = inner2(v); u / w };
        b_atan2 = |v| { let (u, w) = inner2(v); u.atan2(w) };
        b_atan2_rev = |v| { let (u, w) = inner2(v); w.atan2(u) };
        b_mul_add = |v| { let (u, w) = inner2(v); u.mul_add(w, u * u - w) };
        b_mul_add_self = |v| { let (u, w) = inner2(v); u.mul_add(u, w.exp()) };
        b_add_assign = |v| { let (mut u, w) = inner2(v); u += w; u += 0.5; u };
        b_sub_assign = |v| { let (mut u, w) = inner2(v); u -= w; u -= 0.5; u };
        b_mul_assign = |v| { let (mut u, w) = inner2(v); u *= w; u *= 0.5; u };
        b_div_assign = |v| { let (mut u, w) = inner2(v); u /= w; u /= 0.5; u };
        b_composite = |v| {
            let (u, w) = inner2(v);
            (u * w).exp() / (u * u + w * w).sqrt() + (u / 3.0).acos() * w.norm_cdf()
                - (w * w + 0.5).ln() * u.norm_pdf()
                + u.sin() * w.cos()
                + u.atan2(w).powi(2)
        };
    }

    macro_rules! run {
        ($name:ident, $pts:expr) => {
            for &x in $pts.iter() {
                check::<2>(
                    stringify!($name),
                    $name::<H2>,
                    $name::<Dual<2>>,
                    $name::<f64>,
                    &x,
                );
            }
        };
    }

    fn at(xs: &[f64]) -> Vec<[f64; 2]> {
        xs.iter().map(|&a| [a, B0]).collect()
    }

    #[test]
    fn unary_rules_match_dual_and_finite_differences() {
        let pos = at(&[0.3, 1.0, 2.7, 10.0]);
        let any = at(&[-2.0, -0.7, 0.0, 0.5, 3.0]);
        let nonzero = at(&[-1.5, -0.4, 0.4, 2.0]);
        let unit = at(&[-0.9, -0.3, 0.0, 0.5, 0.95]);
        run!(u_sqrt, pos);
        run!(u_exp, any);
        run!(u_ln, pos);
        run!(u_sin, any);
        run!(u_cos, any);
        run!(u_acos, unit);
        run!(u_powf_half, pos);
        run!(u_powf_m13, pos);
        run!(u_powf_1, pos);
        run!(u_powf_15, pos);
        run!(u_powf_2, pos);
        run!(u_powf_25, pos);
        run!(u_powi_0, any);
        run!(u_powi_1, any);
        run!(u_powi_2, any);
        run!(u_powi_3, any);
        run!(u_powi_m2, nonzero);
        run!(u_abs, nonzero);
        run!(u_cdf, at(&[-6.0, -3.0, -0.5, 0.0, 1.2, 4.0]));
        run!(u_pdf, at(&[-6.0, -3.0, -0.5, 0.0, 1.2, 4.0]));
        run!(u_neg, any);
        run!(u_add_f, any);
        run!(u_sub_f, any);
        run!(u_mul_f, any);
        run!(u_div_f, any);
        run!(f_add_u, any);
        run!(f_sub_u, any);
        run!(f_mul_u, any);
        run!(f_div_u, nonzero);
        run!(u_from, any);
    }

    #[test]
    fn binary_rules_match_dual_and_finite_differences() {
        // (u, w) = inner2 covers all four quadrants for atan2.
        let pts = [
            [0.7, 1.3],
            [-2.0, 0.4],
            [1.5, -3.0],
            [-1.1, -0.9],
            [0.0, 2.0],
            [2.0, -0.5],
            [0.05, -1.0],
        ];
        run!(b_add, pts);
        run!(b_sub, pts);
        run!(b_mul, pts);
        run!(b_div, pts);
        run!(b_atan2, pts);
        run!(b_atan2_rev, pts);
        run!(b_mul_add, pts);
        run!(b_mul_add_self, pts);
        run!(b_add_assign, pts);
        run!(b_sub_assign, pts);
        run!(b_mul_assign, pts);
        run!(b_div_assign, pts);
        let comp = [[0.7, 1.3], [-1.1, -0.9], [0.4, -0.6], [1.2, 0.2]];
        run!(b_composite, comp);
    }

    #[test]
    fn assign_ops_agree_with_binary_ops() {
        let x = seed_hdual::<2>(&[0.7, -1.3]);
        let a = x[0].exp() * x[1];
        let b = x[1].sin() + x[0] * x[0];
        let same = |p: H2, q: H2| p.re == q.re && p.g == q.g && p.h == q.h;
        let mut c = a;
        c += b;
        assert!(same(c, a + b));
        c = a;
        c -= b;
        assert!(same(c, a - b));
        c = a;
        c *= b;
        assert!(same(c, a * b));
        c = a;
        c /= b;
        assert!(same(c, a / b));
        c = a;
        c *= 3.0;
        assert!(same(c, a * 3.0));
        c = a;
        c /= 3.0;
        assert!(same(c, a / 3.0));
        c = a;
        c += 3.0;
        assert!(same(c, a + 3.0));
        c = a;
        c -= 3.0;
        assert!(same(c, a - 3.0));
        assert!(same(3.0 * a, a * 3.0));
        assert!(same(3.0 + a, a + 3.0));
        assert!(same(a.mul_add(b, a), {
            // mul_add is fused, so compare to tolerance rather than bitwise
            let r = a * b + a;
            assert!((r.re - a.mul_add(b, a).re).abs() < 1e-15);
            a.mul_add(b, a)
        }));
    }

    /// `x` is the inner function at `(a, B0)`, so its gradient and Hessian are
    /// non-zero while its value is `a`.
    fn inner_at(a: f64) -> H2 {
        inner(&seed_hdual::<2>(&[a, B0]))
    }

    fn assert_flat(name: &str, r: H2) {
        assert_eq!(r.g, [0.0; 2], "{name}: gradient");
        assert_eq!(r.hessian_full(), [[0.0; 2]; 2], "{name}: Hessian");
    }

    #[test]
    fn guards_give_zero_first_and_second_derivatives() {
        let z = inner_at(0.0);
        assert!(z.re == 0.0 && z.g != [0.0; 2] && z.h != [[0.0; 2]; 2]);
        assert_flat("sqrt(0)", z.sqrt());
        assert_flat("acos(1)", inner_at(1.0).acos());
        assert_flat("acos(-1)", inner_at(-1.0).acos());
        assert!(inner_at(1.5).acos().re.is_nan());
        assert_flat("acos(1.5)", inner_at(1.5).acos());
        assert_flat("powf(0, 0.5)", z.powf(0.5));
        assert_flat("powf(0, -0.3)", z.powf(-0.3));
        assert_flat("powf(0, 0)", z.powf(0.0));
        assert_flat("powf(0, 1.5)", z.powf(1.5));
        assert_flat("powf(0, 3)", z.powf(3.0));
        assert_flat("abs(0)", z.abs());
        assert_flat("abs(NaN)", (z + f64::NAN).abs());
        assert_flat("powi(2, 0)", inner_at(2.0).powi(0));
        assert_flat("powi(0, 3)", z.powi(3));
        let o = z.atan2(z * 2.0 + z * z);
        assert_eq!(o.re, 0.0);
        assert_flat("atan2(0, 0)", o);
        // The same points through Dual: zero first derivatives as well.
        let zd = inner(&seed_array::<2>(&[0.0, B0]));
        assert_eq!(zd.sqrt().eps, [0.0; 2]);
        assert_eq!(zd.powf(0.5).eps, [0.0; 2]);
        assert_eq!(zd.abs().eps, [0.0; 2]);
        assert_eq!(zd.atan2(zd * 2.0).eps, [0.0; 2]);
        // Differentiable at 0: x^1, x^2, x^1 (powi), x^2 (powi).
        let id = |r: H2| r.g == z.g && r.hessian_full() == z.hessian_full();
        assert!(id(z.powf(1.0)));
        assert!(id(z.powi(1)));
        let sq = z * z;
        for r in [z.powf(2.0), z.powi(2)] {
            assert_eq!(r.g, [0.0; 2]);
            for i in 0..2 {
                for j in 0..2 {
                    assert!((r.hess(i, j) - sq.hess(i, j)).abs() < 1e-15);
                }
            }
        }
        // No NaN leaks from 0·∞ at the guarded points.
        for r in [z.powf(1.0), z.powi(1), z.powi(2), z.powf(2.0), z.powf(1.5)] {
            assert!(
                r.g.iter()
                    .chain(r.h.iter().flatten())
                    .all(|v| v.is_finite())
            );
        }
    }

    #[test]
    fn seeding_accessors_and_comparisons() {
        let x = [1.0, 2.0, 3.0];
        let s = seed_hdual::<3>(&x);
        for i in 0..3 {
            assert_eq!(s[i].re, x[i]);
            for j in 0..3 {
                assert_eq!(s[i].g[j], if i == j { 1.0 } else { 0.0 });
            }
            assert_eq!(s[i].h, [[0.0; 3]; 3]);
        }
        let c: HDual<4> = 2.5.into();
        assert_eq!((c.re, c.g, c.h), (2.5, [0.0; 4], [[0.0; 4]; 4]));
        assert_eq!(HDual::<2>::default().re, 0.0);
        assert_eq!(HDual::<2>::from_f64(1.5).re(), 1.5);
        assert_eq!(HDual::<2>::one().re, 1.0);
        assert_eq!(HDual::<2>::zero().re, 0.0);

        let f = s[0] * s[1] * s[2] + s[0] * s[0] * s[2];
        assert_eq!(f.grad(), &f.g);
        let full = f.hessian_full();
        let mut flat = [0.0; 9];
        f.write_hessian(&mut flat);
        // ∂² of xyz + x²z at (1, 2, 3)
        let expect = [[6.0, 3.0, 4.0], [3.0, 0.0, 1.0], [4.0, 1.0, 0.0]];
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(full[i][j], expect[i][j]);
                assert_eq!(flat[i * 3 + j], expect[i][j]);
                assert_eq!(f.hess(i, j), expect[i][j]);
            }
        }

        let (a, b) = (HDual::<2>::variable(1.0, 0), HDual::<2>::variable(1.0, 1));
        assert!(a == b);
        assert!(a <= b);
        assert!(HDual::<2>::constant(0.5) < a);
        assert!(a > HDual::<2>::constant(0.5));
        assert_eq!(a.partial_cmp(&b), Some(Ordering::Equal));
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn variable_index_out_of_range_panics() {
        let _ = HDual::<2>::variable(1.0, 2);
    }

    #[test]
    #[should_panic(expected = "needs exactly")]
    fn seed_length_mismatch_panics() {
        let _ = seed_hdual::<3>(&[1.0, 2.0]);
    }

    /// A five-variable function touching every Scalar method, against Dual.
    fn many<S: Scalar>(v: &[S]) -> S {
        let (a, b, c, d, e) = (v[0], v[1], v[2], v[3], v[4]);
        let r = (a * a + b * b + c * c).sqrt();
        let t = (c / r).acos() + b.atan2(a) * d.cos() - e.sin() * a;
        let m = (d - e).abs().powf(1.5) + (a * e).powi(3) / (b * b + 1.0);
        let n = (t * 0.3).norm_cdf().ln() + (m * 0.1).norm_pdf() * e.exp();
        r.mul_add(t, n) * c - m / (d * d + 2.0)
    }

    #[test]
    fn five_variable_composite() {
        for x in [
            [0.8, -1.3, 0.5, 0.4, -0.2],
            [-0.6, 0.9, -1.1, -0.7, 0.3],
            [1.4, 0.2, 0.3, 1.1, 0.6],
        ] {
            check::<5>("many", many::<HDual<5>>, many::<Dual<5>>, many::<f64>, &x);
        }
    }
}
