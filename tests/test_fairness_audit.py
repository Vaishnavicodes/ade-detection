"""Tests for ade_detection.models.fairness_audit.

Uses synthetic data with real demographic columns (is_female, ethnicity_group)
matching the enriched ETL schema — no real MIMIC data, no plot assertions,
no performance-value assertions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ade_detection.models.fairness_audit import (
    _add_age_band,
    _subgroup_metrics,
    audit_dimension,
    run_fairness_audit,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _MockCalibratedModel:
    """Minimal calibrated model that returns seeded random probabilities."""

    def __init__(self, seed: int = 0) -> None:
        self._rng = np.random.default_rng(seed)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n = len(X)
        pos = self._rng.random(n)
        return np.column_stack([1.0 - pos, pos])


def _make_fairness_frame(n: int = 400, pos_frac: float = 0.15, seed: int = 0) -> pd.DataFrame:
    """Synthetic test frame with enriched demographic columns."""
    rng = np.random.default_rng(seed)
    n_pos = max(4, int(n * pos_frac))
    n_neg = n - n_pos
    labels = np.array([1] * n_pos + [0] * n_neg)
    rng.shuffle(labels)

    # Two gender groups of equal size
    is_female = np.array([0] * (n // 2) + [1] * (n - n // 2))

    # Four ethnicity groups
    eth_choices = ["WHITE", "BLACK", "HISPANIC", "OTHER"]
    ethnicity_group = np.array([eth_choices[i % 4] for i in range(n)])

    return pd.DataFrame(
        {
            "visit_occurrence_id": np.arange(n),
            "person_id": np.arange(n),
            "visit_start_datetime": pd.date_range("2150-01-01", periods=n, freq="D"),
            "index_datetime": pd.date_range("2150-01-02", periods=n, freq="D"),
            "age_at_admission": rng.integers(20, 90, size=n).astype(float),
            "n_drugs_first24h": rng.integers(0, 10, size=n).astype(float),
            "n_distinct_drugs_first24h": rng.integers(0, 8, size=n).astype(float),
            "prior_comorb_congestive_heart_failure": rng.integers(0, 2, size=n).astype(float),
            "n_prior_conditions": rng.integers(0, 20, size=n).astype(float),
            "n_prior_drug_exposures": rng.integers(0, 15, size=n).astype(float),
            "ade_prob": rng.random(size=n),
            "is_female": is_female,
            "ethnicity_group": ethnicity_group,
            "ade_label": labels,
        }
    )


@pytest.fixture()
def synthetic_df():
    return _make_fairness_frame(n=400, seed=42)


# ---------------------------------------------------------------------------
# _add_age_band
# ---------------------------------------------------------------------------


def test_add_age_band_creates_column(synthetic_df):
    df = _add_age_band(synthetic_df)
    assert "age_band" in df.columns


def test_add_age_band_values_are_strings(synthetic_df):
    df = _add_age_band(synthetic_df)
    assert df["age_band"].dtype == object


def test_add_age_band_known_values(synthetic_df):
    df = _add_age_band(synthetic_df)
    allowed = {"<40", "40–64", "65–79", "80+", "Unknown"}
    assert set(df["age_band"].unique()).issubset(allowed)


def test_add_age_band_does_not_mutate_input(synthetic_df):
    original_cols = set(synthetic_df.columns)
    _add_age_band(synthetic_df)
    assert set(synthetic_df.columns) == original_cols


def test_add_age_band_age_zero_maps_to_under_40():
    df = pd.DataFrame({"age_at_admission": [0.0]})
    result = _add_age_band(df)
    assert result["age_band"].iloc[0] == "<40"


def test_add_age_band_age_80_maps_to_80plus():
    df = pd.DataFrame({"age_at_admission": [80.0]})
    result = _add_age_band(df)
    assert result["age_band"].iloc[0] == "80+"


def test_add_age_band_age_65_maps_to_65_79():
    df = pd.DataFrame({"age_at_admission": [65.0]})
    result = _add_age_band(df)
    assert result["age_band"].iloc[0] == "65–79"


# ---------------------------------------------------------------------------
# _subgroup_metrics
# ---------------------------------------------------------------------------


def test_subgroup_metrics_returns_dict():
    rng = np.random.default_rng(0)
    y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    y_prob = rng.random(len(y_true))
    m = _subgroup_metrics(y_true, y_prob, threshold=0.5)
    assert isinstance(m, dict)


def test_subgroup_metrics_required_keys():
    y_true = np.array([1, 0, 1, 0, 1, 0])
    y_prob = np.array([0.8, 0.2, 0.7, 0.3, 0.9, 0.1])
    m = _subgroup_metrics(y_true, y_prob, threshold=0.5)
    assert {"n", "n_positive", "sensitivity", "specificity", "precision", "pr_auc"}.issubset(m)


def test_subgroup_metrics_empty_returns_zeros():
    m = _subgroup_metrics(np.array([]), np.array([]), threshold=0.5)
    assert m["n"] == 0
    assert m["n_positive"] == 0


def test_subgroup_metrics_sensitivity_in_unit_interval():
    y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
    y_prob = np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4, 0.5, 0.45])
    m = _subgroup_metrics(y_true, y_prob, threshold=0.5)
    if not np.isnan(m["sensitivity"]):
        assert 0.0 <= m["sensitivity"] <= 1.0


def test_subgroup_metrics_pr_auc_na_when_few_positives():
    y_true = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    y_prob = np.array([0.9, 0.1, 0.2, 0.15, 0.05, 0.1, 0.2, 0.1, 0.05, 0.1])
    m = _subgroup_metrics(y_true, y_prob, threshold=0.5)
    assert m["pr_auc"] == "N/A"


# ---------------------------------------------------------------------------
# audit_dimension — is_female (binary)
# ---------------------------------------------------------------------------


def test_audit_dimension_returns_dataframe_and_gap(synthetic_df):
    rng = np.random.default_rng(0)
    y_prob = rng.random(len(synthetic_df))
    table, gap = audit_dimension(synthetic_df, y_prob, threshold=0.15, dim_col="is_female")
    assert isinstance(table, pd.DataFrame)


def test_audit_dimension_table_has_required_columns(synthetic_df):
    rng = np.random.default_rng(0)
    y_prob = rng.random(len(synthetic_df))
    table, _ = audit_dimension(synthetic_df, y_prob, threshold=0.15, dim_col="is_female")
    for col in ("subgroup", "label", "n", "n_positive", "sensitivity", "specificity", "precision"):
        assert col in table.columns, f"Missing column: {col}"


def test_audit_dimension_is_female_two_rows(synthetic_df):
    """is_female has values 0 and 1 → exactly 2 rows in the table."""
    rng = np.random.default_rng(0)
    y_prob = rng.random(len(synthetic_df))
    table, _ = audit_dimension(synthetic_df, y_prob, threshold=0.15, dim_col="is_female")
    assert len(table) == 2


def test_audit_dimension_sensitivity_computed_for_large_groups(synthetic_df):
    """Both gender groups have n>=100, so sensitivity should be numeric."""
    rng = np.random.default_rng(1)
    y_prob = rng.random(len(synthetic_df))
    table, _ = audit_dimension(synthetic_df, y_prob, threshold=0.15, dim_col="is_female")
    large = table[table["n"] >= 100]
    sens_numeric = pd.to_numeric(large["sensitivity"], errors="coerce")
    assert sens_numeric.notna().any()


def test_audit_dimension_gap_is_float_with_two_large_groups(synthetic_df):
    """n=400 split into 2 groups of 200 each → both n>=100 → gap must be float."""
    rng = np.random.default_rng(2)
    y_prob = rng.random(len(synthetic_df))
    _, gap = audit_dimension(synthetic_df, y_prob, threshold=0.15, dim_col="is_female")
    assert isinstance(gap, float)


def test_audit_dimension_gap_none_for_single_group():
    """Single subgroup → gap cannot be computed → None."""
    df = _make_fairness_frame(n=200)
    df["is_female"] = 0  # all same
    rng = np.random.default_rng(0)
    y_prob = rng.random(len(df))
    _, gap = audit_dimension(df, y_prob, threshold=0.15, dim_col="is_female")
    assert gap is None


def test_audit_dimension_gap_nonnegative(synthetic_df):
    rng = np.random.default_rng(3)
    y_prob = rng.random(len(synthetic_df))
    _, gap = audit_dimension(synthetic_df, y_prob, threshold=0.15, dim_col="is_female")
    if gap is not None:
        assert gap >= 0.0


def test_audit_dimension_small_subgroup_flagged():
    """Create one very small gender group (n=10) — must be flagged in note."""
    df_big = _make_fairness_frame(n=200)
    df_big["is_female"] = 0
    df_small = _make_fairness_frame(n=10, seed=99)
    df_small["is_female"] = 1
    df = pd.concat([df_big, df_small], ignore_index=True)
    rng = np.random.default_rng(0)
    y_prob = rng.random(len(df))
    table, _ = audit_dimension(df, y_prob, threshold=0.15, dim_col="is_female")
    small_rows = table[table["n"] < 100]
    assert not small_rows.empty
    assert small_rows["note"].str.contains("small sample").any()


def test_audit_dimension_sorted_by_n_descending(synthetic_df):
    rng = np.random.default_rng(0)
    y_prob = rng.random(len(synthetic_df))
    table, _ = audit_dimension(synthetic_df, y_prob, threshold=0.15, dim_col="is_female")
    ns = table["n"].tolist()
    assert ns == sorted(ns, reverse=True)


# ---------------------------------------------------------------------------
# audit_dimension — ethnicity_group (multi-class string)
# ---------------------------------------------------------------------------


def test_audit_dimension_ethnicity_group_returns_rows(synthetic_df):
    rng = np.random.default_rng(0)
    y_prob = rng.random(len(synthetic_df))
    table, _ = audit_dimension(synthetic_df, y_prob, threshold=0.15, dim_col="ethnicity_group")
    assert len(table) >= 2


def test_audit_dimension_ethnicity_gap_is_float(synthetic_df):
    """All 4 ethnicity groups have n=100 each → gap must be float."""
    rng = np.random.default_rng(4)
    y_prob = rng.random(len(synthetic_df))
    _, gap = audit_dimension(synthetic_df, y_prob, threshold=0.15, dim_col="ethnicity_group")
    assert isinstance(gap, float)


# ---------------------------------------------------------------------------
# run_fairness_audit — integration
# ---------------------------------------------------------------------------


def test_run_fairness_audit_returns_dict(synthetic_df):
    model = _MockCalibratedModel(seed=0)
    results = run_fairness_audit(synthetic_df, model)
    assert isinstance(results, dict)


def test_run_fairness_audit_contains_is_female_key(synthetic_df):
    model = _MockCalibratedModel(seed=1)
    results = run_fairness_audit(synthetic_df, model)
    assert "is_female" in results


def test_run_fairness_audit_contains_ethnicity_group_key(synthetic_df):
    model = _MockCalibratedModel(seed=2)
    results = run_fairness_audit(synthetic_df, model)
    assert "ethnicity_group" in results


def test_run_fairness_audit_contains_age_band_key(synthetic_df):
    model = _MockCalibratedModel(seed=3)
    results = run_fairness_audit(synthetic_df, model)
    assert "age_band" in results


def test_run_fairness_audit_each_value_is_tuple(synthetic_df):
    model = _MockCalibratedModel(seed=4)
    results = run_fairness_audit(synthetic_df, model)
    for key, val in results.items():
        assert isinstance(val, tuple) and len(val) == 2, f"{key} value is not a 2-tuple"


def test_run_fairness_audit_gender_gap_is_float(synthetic_df):
    """is_female has 2 groups of 200 each → gap must be computable → float."""
    model = _MockCalibratedModel(seed=5)
    results = run_fairness_audit(synthetic_df, model)
    _, gap = results["is_female"]
    assert isinstance(gap, float)


def test_run_fairness_audit_ethnicity_gap_is_float(synthetic_df):
    """4 ethnicity groups of 100 each → gap must be computable → float."""
    model = _MockCalibratedModel(seed=6)
    results = run_fairness_audit(synthetic_df, model)
    _, gap = results["ethnicity_group"]
    assert isinstance(gap, float)


def test_run_fairness_audit_age_band_table_has_rows(synthetic_df):
    model = _MockCalibratedModel(seed=7)
    results = run_fairness_audit(synthetic_df, model)
    table, _ = results["age_band"]
    assert len(table) > 0


def test_run_fairness_audit_single_gender_group_gap_none():
    """When all rows are the same gender → gap is None (can't compare one group)."""
    df = _make_fairness_frame(n=400)
    df["is_female"] = 0  # all Male
    model = _MockCalibratedModel(seed=8)
    results = run_fairness_audit(df, model)
    _, gap = results["is_female"]
    assert gap is None
