"""Tests for the leakage-safe temporal feature pipeline.

Synthetic OMOP parquets are written to pytest's tmp_path fixture.  No real MIMIC
data is used and no Snorkel LabelModel is trained — we only test the feature
construction logic.

Key invariants verified:
  1. A drug given in the first 24 h is counted; one given at hour 30 is not.
  2. SIDER curated drugs are excluded from first-24h medication counts even when
     the drug is administered within the 24-h window.
  3. A condition from a PRIOR admission contributes to prior-history features.
  4. A condition from the CURRENT admission does NOT appear in prior-history features.
  5. ADE-defining ICD-9 codes (E930-E949) from prior admissions are excluded
     from prior-history features (anti-leakage hard exclusion).
  6. No label-defining code or SIDER drug string appears in any feature column name.
  7. time_aware_split: all test-set admissions are strictly later than the train set.
"""

from __future__ import annotations

import pandas as pd
import pytest

from ade_detection.features.build_features import build_features
from ade_detection.features.temporal_split import time_aware_split
from ade_detection.labeling.ade_mappings import get_patterns
from ade_detection.labeling.labeling_functions import normalize_icd9

# ---------------------------------------------------------------------------
# Shared synthetic-data fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def synthetic_processed(tmp_path):
    """
    One patient (person_id=1) with two admissions:

    Visit 100  [PRIOR]  2151-05-01 08:00 – 2151-05-10 10:00
      Conditions:
        "4280"  (CHF, dotless)  — prior, NOT ADE code → should contribute to
                                   prior_comorb_congestive_heart_failure for visit 101
        "E9354" (drug ADE e-code) — prior, IS ADE code → EXCLUDED from prior features

    Visit 101  [TARGET]  2151-06-01 09:00 – 2151-06-04 11:00
      index_datetime = 2151-06-01 09:00 + 24 h = 2151-06-02 09:00
      Conditions:
        "41071" (MI, dotless) — current admission, should NOT appear in prior features
      Drugs:
        "Furosemide 40mg Tab"    at 2151-06-01 12:00 (hour  3) → in window, not SIDER → COUNT
        "Vancomycin HCl 1g IVPB" at 2151-06-01 12:00 (hour  3) → in window, IS SIDER → EXCLUDE
        "Furosemide 40mg Tab"    at 2151-06-02 15:00 (hour 30) → outside window → EXCLUDE
    """
    person_df = pd.DataFrame(
        {
            "person_id": [1],
            "year_of_birth": [1970],
            "gender_concept_id": [8507],
            "race_concept_id": [0],
            "ethnicity_concept_id": [0],
        }
    )

    visit_df = pd.DataFrame(
        {
            "visit_occurrence_id": [100, 101],
            "person_id": [1, 1],
            "visit_start_datetime": pd.to_datetime(["2151-05-01 08:00", "2151-06-01 09:00"]),
            "visit_end_datetime": pd.to_datetime(["2151-05-10 10:00", "2151-06-04 11:00"]),
        }
    )

    condition_df = pd.DataFrame(
        {
            "condition_occurrence_id": [1, 2, 3],
            "person_id": [1, 1, 1],
            "visit_occurrence_id": [100, 100, 101],
            "condition_source_value": ["4280", "E9354", "41071"],
            "condition_start_datetime": pd.to_datetime(
                [
                    "2151-05-02 08:00",  # CHF from prior visit — should contribute
                    "2151-05-03 08:00",  # ADE e-code from prior visit — should be excluded
                    "2151-06-01 10:00",  # MI from current visit — not in prior features
                ]
            ),
            "condition_concept_id": [0, 0, 0],
            "condition_type_concept_id": [0, 0, 0],
        }
    )

    drug_df = pd.DataFrame(
        {
            "drug_exposure_id": [1, 2, 3],
            "person_id": [1, 1, 1],
            "visit_occurrence_id": [101, 101, 101],
            "drug_source_value": ["0001", "0002", "0001"],
            "drug_name_source_value": [
                "Furosemide 40mg Tab",  # hour  3 — in window, not SIDER → COUNT
                "Vancomycin HCl 1g IVPB",  # hour  3 — in window, IS SIDER → EXCLUDE
                "Furosemide 40mg Tab",  # hour 30 — outside window → EXCLUDE
            ],
            "drug_exposure_start_datetime": pd.to_datetime(
                [
                    "2151-06-01 12:00",  # +3 h from visit_start (in 24-h window)
                    "2151-06-01 12:00",  # +3 h from visit_start (SIDER drug)
                    "2151-06-02 15:00",  # +30 h from visit_start (outside window)
                ]
            ),
            "drug_concept_id": [0, 0, 0],
        }
    )

    labels_df = pd.DataFrame(
        {
            "visit_occurrence_id": [100, 101],
            "ade_prob": [0.05, 0.08],
            "ade_label": [0, 0],
        }
    )

    person_df.to_parquet(tmp_path / "person.parquet", index=False)
    visit_df.to_parquet(tmp_path / "visit_occurrence.parquet", index=False)
    condition_df.to_parquet(tmp_path / "condition_occurrence.parquet", index=False)
    drug_df.to_parquet(tmp_path / "drug_exposure.parquet", index=False)
    labels_df.to_parquet(tmp_path / "ade_labels.parquet", index=False)

    config = {
        "temporal": {"prediction_timepoint_hours": 24, "lookback_days": 365},
        "paths": {"processed_data_dir": str(tmp_path)},
    }
    return tmp_path, config


# ---------------------------------------------------------------------------
# Shared features fixture (calls build_features once; used by most tests)
# ---------------------------------------------------------------------------


@pytest.fixture()
def features(synthetic_processed):
    tmp_path, config = synthetic_processed
    return build_features(tmp_path, config)


# ---------------------------------------------------------------------------
# Invariant 1: First-24h drug window (time filter)
# ---------------------------------------------------------------------------


def test_drug_at_hour3_counted(features):
    """A drug given 3 h after admission (inside the 24-h window) is counted."""
    row101 = features[features["visit_occurrence_id"] == 101].iloc[0]
    # Only Furosemide is non-SIDER and in-window; count must be >= 1
    assert row101["n_drugs_first24h"] >= 1


def test_drug_at_hour30_not_counted(features):
    """A drug given 30 h after admission (outside the 24-h window) is NOT counted.

    The second Furosemide at hour 30 must not inflate the count beyond 1.
    """
    row101 = features[features["visit_occurrence_id"] == 101].iloc[0]
    # 3 drug rows: hour-3 Furosemide (counted), hour-3 Vancomycin (excluded-SIDER),
    # hour-30 Furosemide (excluded-time).  Only 1 should be counted.
    assert row101["n_drugs_first24h"] == 1


def test_n_distinct_drugs_first24h_counts_unique_names(features):
    """Distinct drug count equals the number of unique non-excluded names in window."""
    row101 = features[features["visit_occurrence_id"] == 101].iloc[0]
    # Only 1 distinct non-SIDER, in-window drug name (Furosemide 40mg Tab)
    assert row101["n_distinct_drugs_first24h"] == 1


# ---------------------------------------------------------------------------
# Invariant 2: SIDER ADE drug exclusion from first-24h features
# ---------------------------------------------------------------------------


def test_sider_drug_in_window_excluded_from_first24h(features):
    """Vancomycin (a SIDER drug) at hour 3 is inside the window but excluded."""
    row101 = features[features["visit_occurrence_id"] == 101].iloc[0]
    # If Vancomycin were counted, n_drugs_first24h would be 2 (Furosemide + Vancomycin)
    assert row101["n_drugs_first24h"] == 1


# ---------------------------------------------------------------------------
# Invariant 3: Prior admission contributes to prior-history features
# ---------------------------------------------------------------------------


def test_prior_charlson_comorbidity_congestive_heart_failure(features):
    """CHF code '4280' from prior visit 100 → prior_comorb_congestive_heart_failure=1."""
    row101 = features[features["visit_occurrence_id"] == 101].iloc[0]
    assert row101["prior_comorb_congestive_heart_failure"] == 1


def test_prior_visit_zero_conditions_when_no_prior(features):
    """Visit 100 has no prior admissions; all prior comorbidity flags must be 0."""
    row100 = features[features["visit_occurrence_id"] == 100].iloc[0]
    assert row100["prior_comorb_congestive_heart_failure"] == 0
    assert row100["n_prior_conditions"] == 0


def test_n_prior_conditions_counts_non_ade_codes(features):
    """n_prior_conditions for visit 101 is 1 (only CHF '4280', not the ADE e-code)."""
    row101 = features[features["visit_occurrence_id"] == 101].iloc[0]
    assert row101["n_prior_conditions"] == 1


# ---------------------------------------------------------------------------
# Invariant 4: Current-admission conditions not in prior features
# ---------------------------------------------------------------------------


def test_current_mi_condition_not_in_prior_comorbidity(features):
    """MI code '41071' belongs to visit 101 (current), so it must NOT set
    prior_comorb_myocardial_infarction=1 for visit 101 itself."""
    row101 = features[features["visit_occurrence_id"] == 101].iloc[0]
    assert row101["prior_comorb_myocardial_infarction"] == 0


# ---------------------------------------------------------------------------
# Invariant 5: ADE-defining codes excluded from prior features (anti-leakage)
# ---------------------------------------------------------------------------


def test_ade_ecode_excluded_from_prior_condition_count(features):
    """E9354 (E935.4 drug ADE e-code) from prior visit is excluded.

    Without exclusion, n_prior_conditions for visit 101 would be 2;
    with exclusion it must be exactly 1 (only CHF '4280').
    """
    row101 = features[features["visit_occurrence_id"] == 101].iloc[0]
    assert row101["n_prior_conditions"] == 1


# ---------------------------------------------------------------------------
# Invariant 6: No label-defining strings in feature column names
# ---------------------------------------------------------------------------


def test_no_excluded_codes_in_feature_columns(features):
    """No ADE-defining code or SIDER drug appears in any feature column name."""
    excluded: set[str] = set()
    for p in get_patterns():
        for drug in p["drugs"]:
            excluded.add(drug.lower())
        for prefix in p["icd9_prefixes"]:
            excluded.add(normalize_icd9(prefix).lower())
    for n in range(930, 950):
        excluded.add(f"e{n}")
    excluded.add("9952")

    meta_cols = {
        "visit_occurrence_id",
        "person_id",
        "ade_prob",
        "ade_label",
        "visit_start_datetime",
        "index_datetime",
    }
    feature_cols = [c for c in features.columns if c not in meta_cols]

    violations: list[str] = []
    for col in feature_cols:
        col_lower = col.lower()
        for exc in excluded:
            if exc and exc in col_lower:
                violations.append(f"'{exc}' in column '{col}'")

    assert not violations, "Potential leakage: " + ", ".join(violations)


# ---------------------------------------------------------------------------
# Invariant 7: time_aware_split — test set is strictly later than train
# ---------------------------------------------------------------------------


def test_time_aware_split_test_all_later_than_train(features):
    """All test-set admissions must have a later visit_start than all train admissions."""
    # With 2 admissions and test_frac=0.5: train=[visit 100], test=[visit 101]
    train_df, test_df = time_aware_split(features, test_frac=0.5)
    assert len(train_df) == 1
    assert len(test_df) == 1
    assert train_df["visit_start_datetime"].max() <= test_df["visit_start_datetime"].min()


def test_time_aware_split_no_overlap(features):
    """train and test sets are disjoint by visit_occurrence_id."""
    train_df, test_df = time_aware_split(features, test_frac=0.5)
    train_ids = set(train_df["visit_occurrence_id"])
    test_ids = set(test_df["visit_occurrence_id"])
    assert train_ids.isdisjoint(test_ids)


def test_time_aware_split_covers_all_admissions(features):
    """train + test should cover the full feature matrix."""
    train_df, test_df = time_aware_split(features, test_frac=0.5)
    assert len(train_df) + len(test_df) == len(features)


# ---------------------------------------------------------------------------
# Schema / shape sanity checks
# ---------------------------------------------------------------------------


def test_one_row_per_admission(features):
    assert features["visit_occurrence_id"].nunique() == len(features)


def test_expected_feature_columns_present(features):
    required = {
        "visit_occurrence_id",
        "person_id",
        "visit_start_datetime",
        "age_at_admission",
        "n_drugs_first24h",
        "n_distinct_drugs_first24h",
        "n_prior_conditions",
        "n_prior_drug_exposures",
        "ade_prob",
        "ade_label",
    }
    assert required.issubset(set(features.columns))


def test_prior_comorb_columns_present(features):
    comorb_cols = [c for c in features.columns if c.startswith("prior_comorb_")]
    assert len(comorb_cols) > 0, "Expected at least one prior_comorb_ column"


def test_age_at_admission_is_positive(features):
    assert (features["age_at_admission"] > 0).all()
