"""Tests for ade_detection.labeling.build_labeling_frame.

Uses tiny synthetic OMOP parquets written to pytest's tmp_path fixture.
LabelModel training is NOT tested here (too heavy for CI); see run_label_model.py.
"""

from __future__ import annotations

import pandas as pd
import pytest

from ade_detection.labeling.build_labeling_frame import build_labeling_frame

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def omop_parquets(tmp_path):
    """Three synthetic visits with partial conditions and drugs."""
    visit_df = pd.DataFrame(
        {
            "visit_occurrence_id": [100, 101, 102],
            "person_id": [1, 2, 3],
            "visit_start_datetime": pd.to_datetime(["2150-01-05", "2160-07-01", "2170-11-20"]),
            "visit_end_datetime": pd.to_datetime(["2150-01-10", "2160-07-08", "2170-11-25"]),
            "visit_source_value": ["EMERGENCY", "ELECTIVE", "EMERGENCY"],
            "admission_type_source_value": ["EMERGENCY", "ELECTIVE", "EMERGENCY"],
        }
    )
    # visit 102 deliberately has no conditions
    condition_df = pd.DataFrame(
        {
            "visit_occurrence_id": [100, 100, 101],
            "condition_source_value": ["410.0", "428.0", "250.00"],
        }
    )
    # visit 102 deliberately has no drugs
    drug_df = pd.DataFrame(
        {
            "visit_occurrence_id": [100, 101, 101],
            "drug_source_value": ["0000-0001-01", "0000-0002-02", "0000-0003-03"],
        }
    )

    visit_df.to_parquet(tmp_path / "visit_occurrence.parquet", index=False)
    condition_df.to_parquet(tmp_path / "condition_occurrence.parquet", index=False)
    drug_df.to_parquet(tmp_path / "drug_exposure.parquet", index=False)
    return tmp_path


# ---------------------------------------------------------------------------
# Schema / shape
# ---------------------------------------------------------------------------


def test_one_row_per_visit(omop_parquets):
    result = build_labeling_frame(omop_parquets)
    assert len(result) == 3
    assert result["visit_occurrence_id"].nunique() == 3


def test_expected_columns_present(omop_parquets):
    result = build_labeling_frame(omop_parquets)
    expected = {
        "visit_occurrence_id",
        "person_id",
        "icd9_codes",
        "drug_names",
        "length_of_stay_days",
        "discharge_disposition",
        "admission_type",
        "visit_start_datetime",
    }
    assert expected.issubset(set(result.columns))


# ---------------------------------------------------------------------------
# ICD-9 aggregation
# ---------------------------------------------------------------------------


def test_icd9_codes_is_list_column(omop_parquets):
    result = build_labeling_frame(omop_parquets)
    for codes in result["icd9_codes"]:
        assert isinstance(codes, list)


def test_icd9_codes_contain_strings(omop_parquets):
    result = build_labeling_frame(omop_parquets)
    for codes in result["icd9_codes"]:
        for code in codes:
            assert isinstance(code, str)


def test_icd9_codes_aggregated_per_visit(omop_parquets):
    result = build_labeling_frame(omop_parquets)
    row100 = result.loc[result["visit_occurrence_id"] == 100].iloc[0]
    assert set(row100["icd9_codes"]) == {"410.0", "428.0"}

    row101 = result.loc[result["visit_occurrence_id"] == 101].iloc[0]
    assert row101["icd9_codes"] == ["250.00"]


def test_visit_with_no_conditions_gets_empty_list(omop_parquets):
    result = build_labeling_frame(omop_parquets)
    row102 = result.loc[result["visit_occurrence_id"] == 102].iloc[0]
    assert row102["icd9_codes"] == []


# ---------------------------------------------------------------------------
# Drug name aggregation
# ---------------------------------------------------------------------------


def test_drug_names_is_list_column(omop_parquets):
    result = build_labeling_frame(omop_parquets)
    for drugs in result["drug_names"]:
        assert isinstance(drugs, list)


def test_drug_names_aggregated_per_visit(omop_parquets):
    result = build_labeling_frame(omop_parquets)
    row101 = result.loc[result["visit_occurrence_id"] == 101].iloc[0]
    assert set(row101["drug_names"]) == {"0000-0002-02", "0000-0003-03"}


def test_visit_with_no_drugs_gets_empty_list(omop_parquets):
    result = build_labeling_frame(omop_parquets)
    row102 = result.loc[result["visit_occurrence_id"] == 102].iloc[0]
    assert row102["drug_names"] == []


def test_drug_name_source_value_preferred_over_ndc(tmp_path):
    """drug_name_source_value (human-readable) takes priority over drug_source_value (NDC)."""
    visit_df = pd.DataFrame(
        {
            "visit_occurrence_id": [200],
            "person_id": [9],
            "visit_start_datetime": pd.to_datetime(["2150-03-01"]),
            "visit_end_datetime": pd.to_datetime(["2150-03-04"]),
            "visit_source_value": ["ELECTIVE"],
        }
    )
    condition_df = pd.DataFrame(
        {
            "visit_occurrence_id": pd.Series([], dtype="int64"),
            "condition_source_value": pd.Series([], dtype="str"),
        }
    )
    drug_df = pd.DataFrame(
        {
            "visit_occurrence_id": [200],
            "drug_source_value": ["0000-0099-99"],
            "drug_name_source_value": ["Vancomycin HCl 1250mg IVPB"],
        }
    )

    visit_df.to_parquet(tmp_path / "visit_occurrence.parquet", index=False)
    condition_df.to_parquet(tmp_path / "condition_occurrence.parquet", index=False)
    drug_df.to_parquet(tmp_path / "drug_exposure.parquet", index=False)

    result = build_labeling_frame(tmp_path)
    assert result.iloc[0]["drug_names"] == ["Vancomycin HCl 1250mg IVPB"]


# ---------------------------------------------------------------------------
# Length-of-stay
# ---------------------------------------------------------------------------


def test_los_computed_correctly(omop_parquets):
    result = build_labeling_frame(omop_parquets)

    row100 = result.loc[result["visit_occurrence_id"] == 100].iloc[0]
    assert abs(row100["length_of_stay_days"] - 5.0) < 1e-6  # Jan 05 -> Jan 10

    row101 = result.loc[result["visit_occurrence_id"] == 101].iloc[0]
    assert abs(row101["length_of_stay_days"] - 7.0) < 1e-6  # Jul 01 -> Jul 08


def test_los_is_non_negative(omop_parquets):
    result = build_labeling_frame(omop_parquets)
    assert (result["length_of_stay_days"] >= 0).all()


# ---------------------------------------------------------------------------
# Discharge disposition default
# ---------------------------------------------------------------------------


def test_discharge_disposition_defaults_to_unknown(omop_parquets):
    result = build_labeling_frame(omop_parquets)
    # No discharge_disposition in our synthetic visit_occurrence — should default to "unknown"
    assert (result["discharge_disposition"] == "unknown").all()


def test_admission_type_populated_from_source_column(omop_parquets):
    result = build_labeling_frame(omop_parquets)
    types = dict(zip(result["visit_occurrence_id"], result["admission_type"]))
    assert types[100] == "EMERGENCY"
    assert types[101] == "ELECTIVE"
    assert types[102] == "EMERGENCY"


def test_admission_type_falls_back_to_visit_source_value(tmp_path):
    """Parquets without admission_type_source_value fall back to visit_source_value."""
    visit_df = pd.DataFrame(
        {
            "visit_occurrence_id": [400],
            "person_id": [20],
            "visit_start_datetime": pd.to_datetime(["2150-05-01"]),
            "visit_end_datetime": pd.to_datetime(["2150-05-03"]),
            "visit_source_value": ["ELECTIVE"],
            # no admission_type_source_value column — older parquet
        }
    )
    condition_df = pd.DataFrame(
        {
            "visit_occurrence_id": pd.Series([], dtype="int64"),
            "condition_source_value": pd.Series([], dtype="str"),
        }
    )
    drug_df = pd.DataFrame(
        {
            "visit_occurrence_id": pd.Series([], dtype="int64"),
            "drug_source_value": pd.Series([], dtype="str"),
        }
    )
    visit_df.to_parquet(tmp_path / "visit_occurrence.parquet", index=False)
    condition_df.to_parquet(tmp_path / "condition_occurrence.parquet", index=False)
    drug_df.to_parquet(tmp_path / "drug_exposure.parquet", index=False)

    result = build_labeling_frame(tmp_path)
    assert result.iloc[0]["admission_type"] == "ELECTIVE"


def test_discharge_disposition_used_when_present(tmp_path):
    visit_df = pd.DataFrame(
        {
            "visit_occurrence_id": [300, 301],
            "person_id": [10, 11],
            "visit_start_datetime": pd.to_datetime(["2150-01-01", "2150-02-01"]),
            "visit_end_datetime": pd.to_datetime(["2150-01-03", "2150-02-04"]),
            "visit_source_value": ["EMERGENCY", "ELECTIVE"],
            "discharge_location_source_value": ["HOME", "DEAD/EXPIRED"],
        }
    )
    condition_df = pd.DataFrame(
        {
            "visit_occurrence_id": pd.Series([], dtype="int64"),
            "condition_source_value": pd.Series([], dtype="str"),
        }
    )
    drug_df = pd.DataFrame(
        {
            "visit_occurrence_id": pd.Series([], dtype="int64"),
            "drug_source_value": pd.Series([], dtype="str"),
        }
    )

    visit_df.to_parquet(tmp_path / "visit_occurrence.parquet", index=False)
    condition_df.to_parquet(tmp_path / "condition_occurrence.parquet", index=False)
    drug_df.to_parquet(tmp_path / "drug_exposure.parquet", index=False)

    result = build_labeling_frame(tmp_path)
    dispositions = dict(zip(result["visit_occurrence_id"], result["discharge_disposition"]))
    assert dispositions[300] == "HOME"
    assert dispositions[301] == "DEAD/EXPIRED"
