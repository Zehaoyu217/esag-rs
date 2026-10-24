//! statsmodels-style text output: a two-column header block and one
//! coefficient block per equation (as `MNLogit.summary()` prints them).

use crate::stats::normal::{norm_quantile, norm_sf};
use std::fmt::Write as _;

/// Width of the tables (statsmodels uses 78).
pub(crate) const WIDTH: usize = 78;

/// `(date, time)` in UTC, formatted like statsmodels (`Fri, 24 Oct 2026`,
/// `11:22:33`).
pub(crate) fn utc_date_time() -> (String, String) {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let days = secs.div_euclid(86_400);
    let rem = secs.rem_euclid(86_400);
    // civil date from days since 1970-01-01 (H. Hinnant's algorithm)
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if month <= 2 { y + 1 } else { y };
    let weekday = ["Thu", "Fri", "Sat", "Sun", "Mon", "Tue", "Wed"][days.rem_euclid(7) as usize];
    let months = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ];
    (
        format!(
            "{weekday}, {day:02} {} {year}",
            months[(month - 1) as usize]
        ),
        format!("{:02}:{:02}:{:02}", rem / 3600, (rem % 3600) / 60, rem % 60),
    )
}

/// A centred title, a `=` rule, and the header pairs in two columns.
pub(crate) fn header(title: &str, left: &[(String, String)], right: &[(String, String)]) -> String {
    let mut s = String::new();
    let _ = writeln!(s, "{:^w$}", title, w = WIDTH);
    let _ = writeln!(s, "{}", "=".repeat(WIDTH));
    let rows = left.len().max(right.len());
    let half = (WIDTH - 3) / 2;
    for i in 0..rows {
        let cell = |v: &[(String, String)], width: usize| -> String {
            match v.get(i) {
                Some((k, val)) => {
                    let pad = width.saturating_sub(k.len() + val.chars().count());
                    format!("{k}{}{val}", " ".repeat(pad.max(1)))
                }
                None => " ".repeat(width),
            }
        };
        let _ = writeln!(
            s,
            "{}   {}",
            cell(left, half),
            cell(right, WIDTH - 3 - half)
        );
    }
    s
}

/// One coefficient line of a table.
pub(crate) struct CoefRow {
    pub name: String,
    pub coef: f64,
    /// `None`: no standard error (fixed, dropped or not computed).
    pub se: Option<f64>,
    /// Shown instead of the inference columns when `se` is `None`.
    pub note: &'static str,
}

/// Coefficient blocks `(equation, rows)` with z-tests and `1 − alpha`
/// confidence intervals, statsmodels style.
pub(crate) fn coef_blocks(blocks: &[(String, Vec<CoefRow>)], alpha: f64) -> String {
    let mut s = String::new();
    let name_w = blocks
        .iter()
        .flat_map(|(e, r)| std::iter::once(e.len()).chain(r.iter().map(|c| c.name.len())))
        .max()
        .unwrap_or(0)
        .max(10)
        + 2;
    let width = name_w + 10 + 11 + 11 + 10 + 12 + 12;
    let zq = norm_quantile(1.0 - alpha / 2.0);
    let (lo_h, hi_h) = (
        format!("[{:.3}", alpha / 2.0),
        format!("{:.3}]", 1.0 - alpha / 2.0),
    );
    for (bi, (eq, rows)) in blocks.iter().enumerate() {
        let _ = writeln!(s, "{}", if bi == 0 { "=" } else { "-" }.repeat(width));
        let _ = writeln!(
            s,
            "{:>nw$}{:>10}{:>11}{:>11}{:>10}{:>12}{:>12}",
            eq,
            "coef",
            "std err",
            "z",
            "P>|z|",
            lo_h,
            hi_h,
            nw = name_w
        );
        let _ = writeln!(s, "{}", "-".repeat(width));
        for r in rows {
            match r.se {
                Some(se) if se.is_finite() && se > 0.0 => {
                    let z = r.coef / se;
                    let p = 2.0 * norm_sf(z.abs());
                    let _ = writeln!(
                        s,
                        "{:<nw$}{:>10.4}{:>11.3}{:>11.3}{:>10.3}{:>12.3}{:>12.3}",
                        r.name,
                        r.coef,
                        se,
                        z,
                        p,
                        r.coef - zq * se,
                        r.coef + zq * se,
                        nw = name_w
                    );
                }
                _ => {
                    let _ = writeln!(
                        s,
                        "{:<nw$}{:>10.4}{:>11}",
                        r.name,
                        r.coef,
                        r.note,
                        nw = name_w
                    );
                }
            }
        }
    }
    let _ = writeln!(s, "{}", "=".repeat(width));
    s
}

/// Format a number compactly for header cells.
pub(crate) fn num(v: f64) -> String {
    if !v.is_finite() {
        return format!("{v}");
    }
    let a = v.abs();
    if a != 0.0 && !(1e-3..1e6).contains(&a) {
        format!("{v:.3e}")
    } else {
        format!("{v:.4}")
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn date_is_well_formed() {
        let (d, t) = super::utc_date_time();
        assert!(d.len() >= 16 && d.contains(", "), "{d}");
        assert_eq!(t.len(), 8);
    }
}
