//! NPMI (Normalized Pointwise Mutual Information) computation for
//! `calculate_trace_filter_correlation` — a straight port of
//! `mlflow/store/analytics/trace_correlation.py`
//! (`calculate_npmi_from_counts` / `_calculate_npmi_core`).
//!
//! The store SQL that produces the four counts lives in
//! [`super::traces_analytics`]; this module is the pure-math half and is kept
//! separate (like Python's `mlflow/store/analytics` package) so it can be unit
//! tested without a database.

/// Recommended smoothing parameter for NPMI calculation — Jeffreys prior
/// (`alpha=0.5`), matching `JEFFREYS_PRIOR` in `trace_correlation.py`.
const JEFFREYS_PRIOR: f64 = 0.5;

/// The four contingency counts needed for NPMI
/// (`TraceCorrelationCounts`). `total_count` is the size of the universe
/// (all traces, or the base-filter universe when a base filter is set).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TraceCorrelationCounts {
    pub total_count: i64,
    pub filter1_count: i64,
    pub filter2_count: i64,
    pub joint_count: i64,
}

/// Result of the NPMI calculation: the unsmoothed value (with the explicit
/// `-1.0` rule for zero joint count) and the Jeffreys-prior smoothed value.
/// Either can be `NaN` when undefined — the wire layer serializes `NaN`
/// as JSON `NaN`, matching Python.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NpmiResult {
    pub npmi: f64,
    pub npmi_smoothed: f64,
}

/// `calculate_npmi_from_counts` (`trace_correlation.py:46`). Returns both the
/// unsmoothed and smoothed NPMI. `NaN` signals an undefined value (zero
/// population, zero support on either filter, or inconsistent counts).
pub fn calculate_npmi_from_counts(counts: TraceCorrelationCounts) -> NpmiResult {
    let TraceCorrelationCounts {
        joint_count,
        filter1_count,
        filter2_count,
        total_count,
    } = counts;

    // No population.
    if total_count <= 0 {
        return NpmiResult {
            npmi: f64::NAN,
            npmi_smoothed: f64::NAN,
        };
    }
    // Zero support on either filter.
    if filter1_count == 0 || filter2_count == 0 {
        return NpmiResult {
            npmi: f64::NAN,
            npmi_smoothed: f64::NAN,
        };
    }

    let n11 = joint_count; // both occur
    let n10 = filter1_count - joint_count; // only filter1
    let n01 = filter2_count - joint_count; // only filter2
    let n00 = total_count - filter1_count - filter2_count + joint_count; // neither

    if n11.min(n10).min(n01).min(n00) < 0 {
        // Inconsistent counts.
        return NpmiResult {
            npmi: f64::NAN,
            npmi_smoothed: f64::NAN,
        };
    }

    let (n11, n10, n01, n00) = (n11 as f64, n10 as f64, n01 as f64, n00 as f64);

    // Unsmoothed NPMI with the explicit -1.0 rule.
    let npmi_unsmoothed = if joint_count == 0 && filter1_count > 0 && filter2_count > 0 {
        -1.0
    } else {
        calculate_npmi_core(n11, n10, n01, n00, 0.0)
    };
    let npmi_smoothed = calculate_npmi_core(n11, n10, n01, n00, JEFFREYS_PRIOR);

    NpmiResult {
        npmi: npmi_unsmoothed,
        npmi_smoothed,
    }
}

/// `_calculate_npmi_core` (`trace_correlation.py:109`). Contingency-table NPMI
/// with optional additive smoothing, in log space for numerical stability.
fn calculate_npmi_core(n11: f64, n10: f64, n01: f64, n00: f64, smoothing: f64) -> f64 {
    let n11_s = n11 + smoothing;
    let n10_s = n10 + smoothing;
    let n01_s = n01 + smoothing;
    let n00_s = n00 + smoothing;

    let n = n11_s + n10_s + n01_s + n00_s;
    let n1 = n11_s + n10_s; // total event 1
    let n2 = n11_s + n01_s; // total event 2

    // PMI is undefined when a marginal is zero.
    if n1 <= 0.0 || n2 <= 0.0 || n11_s <= 0.0 {
        return f64::NAN;
    }

    // Perfect co-occurrence (pre-smoothing check).
    if n10 == 0.0 && n01 == 0.0 && n00 == 0.0 {
        return 1.0;
    }

    let log_n11 = n11_s.ln();
    let log_n = n.ln();
    let log_n1 = n1.ln();
    let log_n2 = n2.ln();

    let pmi = (log_n11 + log_n) - (log_n1 + log_n2);
    let denominator = -(log_n11 - log_n); // -log(n11/N)
    let npmi = pmi / denominator;

    npmi.clamp(-1.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn counts(joint: i64, f1: i64, f2: i64, total: i64) -> TraceCorrelationCounts {
        TraceCorrelationCounts {
            joint_count: joint,
            filter1_count: f1,
            filter2_count: f2,
            total_count: total,
        }
    }

    #[test]
    fn zero_population_is_nan() {
        let r = calculate_npmi_from_counts(counts(0, 0, 0, 0));
        assert!(r.npmi.is_nan() && r.npmi_smoothed.is_nan());
    }

    #[test]
    fn zero_support_is_nan() {
        let r = calculate_npmi_from_counts(counts(0, 0, 5, 100));
        assert!(r.npmi.is_nan());
    }

    #[test]
    fn no_cooccurrence_is_minus_one_unsmoothed() {
        let r = calculate_npmi_from_counts(counts(0, 20, 15, 100));
        assert_eq!(r.npmi, -1.0);
        assert!(r.npmi_smoothed.is_finite());
    }

    #[test]
    fn perfect_cooccurrence_is_one() {
        // filter1 == filter2 == joint == total: n10=n01=n00=0.
        let r = calculate_npmi_from_counts(counts(10, 10, 10, 10));
        assert_eq!(r.npmi, 1.0);
    }

    #[test]
    fn matches_python_reference_positive_association() {
        // calculate_npmi_from_counts(10, 20, 15, 100): observed joint 10 exceeds
        // the independence expectation (20*15/100 = 3) → positive NPMI.
        let r = calculate_npmi_from_counts(counts(10, 20, 15, 100));
        assert!(r.npmi > 0.0 && r.npmi <= 1.0);
    }

    #[test]
    fn inconsistent_counts_are_nan() {
        // joint > filter1 → n10 negative.
        let r = calculate_npmi_from_counts(counts(30, 20, 15, 100));
        assert!(r.npmi.is_nan());
    }
}
