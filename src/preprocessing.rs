//! Covariate preprocessing.
//!
//! Yu & Huang (2025, Sec. 2.1, following Scealy & Wood 2019) recommend
//! rescaling every covariate to `[1, 2]` before fitting an ESAG regression,
//! `x ← (x − min)/(max − min) + 1`, so that covariates are on the same scale
//! as a unit-norm response. [`MinMaxScaler`] does exactly that, scikit-learn
//! style: `fit` on training data, then `transform` training and new data
//! with the training minima and ranges.

use crate::error::{EsagError, Result};
use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};

/// Min–max scaler configuration (default range `[1, 2]`).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MinMaxScaler {
    /// Target range `(lo, hi)`.
    pub feature_range: (f64, f64),
}

impl Default for MinMaxScaler {
    fn default() -> Self {
        Self {
            feature_range: (1.0, 2.0),
        }
    }
}

/// A fitted min–max scaler.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FittedMinMaxScaler {
    /// Target range.
    pub feature_range: (f64, f64),
    /// Per-column minimum of the training data.
    pub data_min: Vec<f64>,
    /// Per-column `max − min` of the training data (1 for constant columns).
    pub data_range: Vec<f64>,
}

impl MinMaxScaler {
    /// Scaler to `[1, 2]`.
    pub fn new() -> Self {
        Self::default()
    }
    /// Scaler to `[lo, hi]`.
    pub fn with_range(lo: f64, hi: f64) -> Self {
        Self {
            feature_range: (lo, hi),
        }
    }
    /// Learn per-column minima and ranges.
    pub fn fit(&self, x: &DMatrix<f64>) -> Result<FittedMinMaxScaler> {
        if x.nrows() == 0 {
            return Err(EsagError::Data("cannot fit a scaler on zero rows".into()));
        }
        let mut data_min = Vec::with_capacity(x.ncols());
        let mut data_range = Vec::with_capacity(x.ncols());
        for j in 0..x.ncols() {
            let col = x.column(j);
            let lo = col.min();
            let hi = col.max();
            if !lo.is_finite() || !hi.is_finite() {
                return Err(EsagError::Data(format!("column {j} is not finite")));
            }
            data_min.push(lo);
            data_range.push(if hi > lo { hi - lo } else { 1.0 });
        }
        Ok(FittedMinMaxScaler {
            feature_range: self.feature_range,
            data_min,
            data_range,
        })
    }
    /// `fit` then `transform` the same data.
    pub fn fit_transform(&self, x: &DMatrix<f64>) -> Result<(FittedMinMaxScaler, DMatrix<f64>)> {
        let f = self.fit(x)?;
        let t = f.transform(x)?;
        Ok((f, t))
    }
}

impl FittedMinMaxScaler {
    fn check(&self, x: &DMatrix<f64>) -> Result<()> {
        if x.ncols() != self.data_min.len() {
            return Err(EsagError::Shape(format!(
                "scaler was fitted on {} columns, got {}",
                self.data_min.len(),
                x.ncols()
            )));
        }
        Ok(())
    }
    /// Map to the target range.
    pub fn transform(&self, x: &DMatrix<f64>) -> Result<DMatrix<f64>> {
        self.check(x)?;
        let (lo, hi) = self.feature_range;
        Ok(DMatrix::from_fn(x.nrows(), x.ncols(), |i, j| {
            lo + (hi - lo) * (x[(i, j)] - self.data_min[j]) / self.data_range[j]
        }))
    }
    /// Map back to the original scale.
    pub fn inverse_transform(&self, x: &DMatrix<f64>) -> Result<DMatrix<f64>> {
        self.check(x)?;
        let (lo, hi) = self.feature_range;
        Ok(DMatrix::from_fn(x.nrows(), x.ncols(), |i, j| {
            self.data_min[j] + (x[(i, j)] - lo) / (hi - lo) * self.data_range[j]
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_to_one_two_and_back() {
        let x = DMatrix::from_row_slice(4, 2, &[0.0, 5.0, 2.0, 5.0, 4.0, 5.0, 1.0, 5.0]);
        let (f, t) = MinMaxScaler::new().fit_transform(&x).unwrap();
        assert_eq!(t[(0, 0)], 1.0);
        assert_eq!(t[(2, 0)], 2.0);
        assert_eq!(t[(3, 0)], 1.25);
        assert_eq!(t[(0, 1)], 1.0); // constant column -> lo
        let back = f.inverse_transform(&t).unwrap();
        assert!((back - x).abs().max() < 1e-15);
    }
}
