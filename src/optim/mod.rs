//! Smooth unconstrained minimisation: L-BFGS with a strong-Wolfe line search,
//! plus a finite-difference Hessian built from the analytic gradient, and a
//! proximal Newton method for LASSO-type penalized objectives.

pub mod hessian;
pub mod lbfgs;
pub mod prox;
