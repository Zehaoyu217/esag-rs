//! Smooth unconstrained minimisation: L-BFGS with a strong-Wolfe line search,
//! plus a finite-difference Hessian built from the analytic gradient.

pub mod hessian;
pub mod lbfgs;
