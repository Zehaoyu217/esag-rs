//! Error type for the `esag` crate.

use thiserror::Error;

/// Errors produced by model construction, fitting and inference.
#[derive(Debug, Error)]
pub enum EsagError {
    /// A dimension is inconsistent (e.g. `gamma` has a length that does not
    /// correspond to any integer `d`, or `d < 3`).
    #[error("invalid dimension: {0}")]
    Dimension(String),
    /// Shapes of inputs do not agree (e.g. `X` and `Y` have different row counts).
    #[error("shape mismatch: {0}")]
    Shape(String),
    /// Input data violate a precondition (non-unit rows, NaN, ...).
    #[error("invalid data: {0}")]
    Data(String),
    /// The optimiser did not converge.
    #[error("optimisation did not converge: {0}")]
    Convergence(String),
    /// A linear-algebra operation failed (e.g. singular Hessian).
    #[error("linear algebra failure: {0}")]
    LinAlg(String),
    /// I/O or (de)serialisation failure.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    /// JSON (de)serialisation failure.
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
    /// CSV failure.
    #[error("csv error: {0}")]
    Csv(#[from] csv::Error),
}

/// Convenience alias.
pub type Result<T> = std::result::Result<T, EsagError>;
