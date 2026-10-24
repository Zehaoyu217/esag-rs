//! Dimension bookkeeping shared by the whole crate.
//!
//! For data on `S^{d-1}` (unit vectors in `R^d`, `d ≥ 3`):
//!
//! | quantity | length |
//! |---|---|
//! | `gamma`  | `g = (d-2)(d+1)/2` |
//! | `eta = (mu, gamma)` | `B = d + g = (d²+d-2)/2` |
//! | `lambda` | `d-1` |
//! | `theta`  | `d-2` |
//! | `phi`    | `(d-2)(d-3)/2` |
//!
//! The inverse maps return `None` when the length does not correspond to any
//! integer `d ≥ 3`.

/// Length of the `gamma` vector for dimension `d`: `(d-2)(d+1)/2`.
#[inline]
pub const fn gamma_len(d: usize) -> usize {
    (d - 2) * (d + 1) / 2
}

/// Length of `eta = (mu, gamma)` for dimension `d`: `d + gamma_len(d) = (d²+d-2)/2`.
#[inline]
pub const fn eta_len(d: usize) -> usize {
    d + gamma_len(d)
}

/// Number of eigenvalue parameters `lambda`: `d - 1`.
#[inline]
pub const fn lambda_len(d: usize) -> usize {
    d - 1
}

/// Number of `theta` angles: `d - 2`.
#[inline]
pub const fn theta_len(d: usize) -> usize {
    d - 2
}

/// Number of `phi` angles: `(d-2)(d-3)/2` (zero for `d = 3`).
#[inline]
pub const fn phi_len(d: usize) -> usize {
    (d - 2) * (d - 3) / 2
}

/// The dimension `d ≥ 3` such that `gamma_len(d) == g`, if any.
///
/// Solves `(d-2)(d+1)/2 = g`, i.e. `d = (1 + sqrt(1 + 8(1+g)))/2`.
pub fn dim_from_gamma_len(g: usize) -> Option<usize> {
    let disc = 1 + 8 * (1 + g);
    let s = disc.isqrt();
    if s * s != disc || !(1 + s).is_multiple_of(2) {
        return None;
    }
    let d = (1 + s).div_ceil(2);
    (d >= 3 && gamma_len(d) == g).then_some(d)
}

/// The dimension `d ≥ 3` such that `eta_len(d) == b`, if any.
///
/// Solves `(d²+d-2)/2 = b`, i.e. `d = (-1 + sqrt(9 + 8b))/2`.
pub fn dim_from_eta_len(b: usize) -> Option<usize> {
    let disc = 9 + 8 * b;
    let s = disc.isqrt();
    if s * s != disc || !(s - 1).is_multiple_of(2) {
        return None;
    }
    let d = (s - 1) / 2;
    (d >= 3 && eta_len(d) == b).then_some(d)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_from_architecture() {
        assert_eq!(gamma_len(3), 2);
        assert_eq!(gamma_len(4), 5);
        assert_eq!(gamma_len(5), 9);
        assert_eq!(gamma_len(6), 14);
        assert_eq!(gamma_len(7), 20);
        assert_eq!(gamma_len(8), 27);
        assert_eq!(eta_len(3), 5);
        assert_eq!(eta_len(4), 9);
        assert_eq!(eta_len(5), 14);
        assert_eq!(eta_len(6), 20);
        assert_eq!(eta_len(7), 27);
        assert_eq!(eta_len(8), 35);
        assert_eq!(phi_len(3), 0);
        assert_eq!(phi_len(4), 1);
        assert_eq!(phi_len(5), 3);
        assert_eq!(phi_len(6), 6);
        assert_eq!(theta_len(3), 1);
        assert_eq!(lambda_len(3), 2);
    }

    #[test]
    fn round_trips_for_d_3_to_8() {
        for d in 3..=8 {
            assert_eq!(dim_from_gamma_len(gamma_len(d)), Some(d), "gamma d={d}");
            assert_eq!(dim_from_eta_len(eta_len(d)), Some(d), "eta d={d}");
            // theta + phi + lambda consistency: (d-2) + (d-2)(d-3)/2 + 1 = g ... check g = theta + phi + (d-1) - 1
            assert_eq!(theta_len(d) + phi_len(d) + lambda_len(d) - 1, gamma_len(d));
        }
    }

    #[test]
    fn invalid_lengths_are_rejected() {
        for g in 0..60 {
            let ok = (3..=12).any(|d| gamma_len(d) == g);
            assert_eq!(dim_from_gamma_len(g).is_some(), ok, "g={g}");
        }
        for b in 0..80 {
            let ok = (3..=12).any(|d| eta_len(d) == b);
            assert_eq!(dim_from_eta_len(b).is_some(), ok, "b={b}");
        }
        // d = 2 is excluded (gamma_len(2) == 0, eta_len(2) == 2).
        assert_eq!(dim_from_gamma_len(0), None);
        assert_eq!(dim_from_eta_len(2), None);
    }
}
