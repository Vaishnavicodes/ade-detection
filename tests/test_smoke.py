"""Smoke tests for ade-detection.

Confirms:
  - omop_mlops (platform dependency) is importable
  - ade_detection package imports correctly
  - config.yaml loads and contains required temporal keys
  - build_omop_tables small-table transforms work on synthetic data
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

# ---------------------------------------------------------------------------
# Platform dependency
# ---------------------------------------------------------------------------


def test_omop_mlops_importable():
    import omop_mlops

    assert omop_mlops.__version__ == "0.1.0"


def test_omop_mlops_features_importable():
    from omop_mlops.features import make_feature_matrix  # noqa: F401


# ---------------------------------------------------------------------------
# ade_detection package
# ---------------------------------------------------------------------------


def test_ade_detection_importable():
    import ade_detection

    assert ade_detection.__version__ == "0.1.0"


def test_build_omop_tables_importable():
    from ade_detection.data import build_omop_tables  # noqa: F401


# ---------------------------------------------------------------------------
# config.yaml
# ---------------------------------------------------------------------------

CONFIG_PATH = Path(__file__).parent.parent / "config" / "config.yaml"


def test_config_yaml_loads():
    assert CONFIG_PATH.exists(), f"config.yaml not found at {CONFIG_PATH}"
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    assert isinstance(cfg, dict)


def test_config_has_temporal_keys():
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    assert "temporal" in cfg, "config.yaml missing 'temporal' section"
    assert "prediction_timepoint_hours" in cfg["temporal"]
    assert "lookback_days" in cfg["temporal"]
    assert cfg["temporal"]["prediction_timepoint_hours"] == 24
    assert cfg["temporal"]["lookback_days"] == 365


def test_config_has_omop_tables():
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    assert "omop_tables" in cfg
    assert "person" in cfg["omop_tables"]
    assert "drug_exposure" in cfg["omop_tables"]


# ---------------------------------------------------------------------------
# build_omop_tables — synthetic data unit tests
# ---------------------------------------------------------------------------


@pytest.fixture()
def synthetic_patients():
    return pd.DataFrame(
        {
            "SUBJECT_ID": [1, 2, 3],
            "GENDER": ["M", "F", "M"],
            "DOB": ["1950-01-01", "1965-06-15", "1980-03-22"],
            "DOD": [None, None, None],
        }
    )


@pytest.fixture()
def synthetic_admissions():
    return pd.DataFrame(
        {
            "SUBJECT_ID": [1, 2, 3],
            "HADM_ID": [100, 101, 102],
            "ADMITTIME": ["2150-01-05 10:00:00", "2160-07-01 08:30:00", "2170-11-20 14:00:00"],
            "DISCHTIME": ["2150-01-10 12:00:00", "2160-07-08 09:00:00", "2170-11-25 11:00:00"],
            "ADMISSION_TYPE": ["EMERGENCY", "ELECTIVE", "EMERGENCY"],
        }
    )


def test_build_person_table_columns(synthetic_patients, tmp_path):
    from ade_detection.data.build_omop_tables import build_person_table

    synthetic_patients.to_csv(tmp_path / "PATIENTS.csv", index=False)
    result = build_person_table(tmp_path)

    assert "person_id" in result.columns
    assert "gender_source_value" in result.columns
    assert "year_of_birth" in result.columns
    assert len(result) == 3
    assert set(result["gender_source_value"]) == {"M", "F"}


def test_build_visit_occurrence_table_columns(synthetic_admissions, tmp_path):
    from ade_detection.data.build_omop_tables import build_visit_occurrence_table

    synthetic_admissions.to_csv(tmp_path / "ADMISSIONS.csv", index=False)
    result = build_visit_occurrence_table(tmp_path)

    assert "visit_occurrence_id" in result.columns
    assert "person_id" in result.columns
    assert "visit_start_datetime" in result.columns
    assert len(result) == 3
    assert list(result["visit_occurrence_id"]) == [100, 101, 102]
