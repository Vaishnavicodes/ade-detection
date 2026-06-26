"""MIMIC-III → OMOP-CDM ETL for the four small tables.

Conventions follow the OHDSI community MIMIC-III→OMOP mapping documented at:
  https://github.com/MIT-LCP/mimic-omop

Concept IDs are set to 0 for v1 with TODO comments.
TODO: map standard concept IDs via the OHDSI Athena vocabulary API before
any analysis that relies on standardised OMOP concept_ids (e.g. cross-site
comparison).  For this project's internal ADE model, source_value columns
carry the clinical codes (ICD-9, NDC) and are the join keys used downstream.

Local ETL covers four tables whose CSV files are small enough to run on a
laptop (< 100 MB each):
  PATIENTS         → person
  ADMISSIONS       → visit_occurrence
  DIAGNOSES_ICD    → condition_occurrence
  PRESCRIPTIONS    → drug_exposure

The fifth table (note) is built by build_note_table(), which reads NOTEEVENTS
(~4 GB). That function must RUN ON COLAB with the file mounted from Drive.

Usage (local):
  python -m ade_detection.data.build_omop_tables
  python -m ade_detection.data.build_omop_tables --config config/config.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_csv(raw_dir: Path, filename: str, **kwargs) -> pd.DataFrame:
    # Prefer the gzipped variant if present; pandas decompresses transparently.
    gz_path = raw_dir / (filename + ".gz")
    path = gz_path if gz_path.exists() else raw_dir / filename
    logger.info("Reading %s", path)
    df = pd.read_csv(path, compression="infer", **kwargs)
    # Normalise to UPPERCASE — MIMIC-III demo ships lowercase headers in some
    # distributions while the full release uses uppercase.
    df.columns = df.columns.str.upper()
    return df


def _write_parquet(df: pd.DataFrame, processed_dir: Path, name: str) -> None:
    processed_dir.mkdir(parents=True, exist_ok=True)
    out = processed_dir / f"{name}.parquet"
    df.to_parquet(out, index=False)
    logger.info("Wrote %s rows → %s", len(df), out)


# ---------------------------------------------------------------------------
# person  ←  PATIENTS.csv
# ---------------------------------------------------------------------------


def build_person_table(raw_dir: Path) -> pd.DataFrame:
    """Map MIMIC-III PATIENTS to OMOP person.

    OMOP columns produced
    ---------------------
    person_id               : SUBJECT_ID (int)
    gender_concept_id       : 0  TODO: 8507=Male, 8532=Female via Athena
    year_of_birth           : derived from DOB
    month_of_birth          : derived from DOB
    day_of_birth            : derived from DOB
    birth_datetime          : DOB (datetime)
    race_concept_id         : 0  TODO: MIMIC has no reliable race field
    ethnicity_concept_id    : 0
    person_source_value     : SUBJECT_ID string
    gender_source_value     : GENDER ('M'/'F')
    race_source_value       : ''
    ethnicity_source_value  : ''
    """
    df = _load_csv(raw_dir, "PATIENTS.csv", parse_dates=["DOB", "DOD"])

    dob = pd.to_datetime(df["DOB"])
    result = pd.DataFrame(
        {
            "person_id": df["SUBJECT_ID"].astype(int),
            # TODO: map via Athena — documented scope boundary for v1
            "gender_concept_id": 0,
            "year_of_birth": dob.dt.year,
            "month_of_birth": dob.dt.month,
            "day_of_birth": dob.dt.day,
            "birth_datetime": dob,
            "race_concept_id": 0,  # TODO: Athena
            "ethnicity_concept_id": 0,  # TODO: Athena
            "person_source_value": df["SUBJECT_ID"].astype(str),
            "gender_source_value": df["GENDER"].str.strip(),
            "race_source_value": "",
            "ethnicity_source_value": "",
        }
    )
    return result


# ---------------------------------------------------------------------------
# visit_occurrence  ←  ADMISSIONS.csv
# ---------------------------------------------------------------------------


def build_visit_occurrence_table(raw_dir: Path) -> pd.DataFrame:
    """Map MIMIC-III ADMISSIONS to OMOP visit_occurrence.

    OMOP columns produced
    ---------------------
    visit_occurrence_id     : HADM_ID
    person_id               : SUBJECT_ID
    visit_concept_id        : 0  TODO: 9201=Inpatient, 9202=Outpatient via Athena
    visit_start_date        : ADMITTIME date
    visit_start_datetime    : ADMITTIME
    visit_end_date          : DISCHTIME date
    visit_end_datetime      : DISCHTIME
    visit_type_concept_id   : 0  TODO: Athena
    visit_source_value      : ADMISSION_TYPE
    """
    df = _load_csv(raw_dir, "ADMISSIONS.csv")
    admit = pd.to_datetime(df["ADMITTIME"], errors="coerce")
    disch = pd.to_datetime(df["DISCHTIME"], errors="coerce")

    result = pd.DataFrame(
        {
            "visit_occurrence_id": df["HADM_ID"].astype(int),
            "person_id": df["SUBJECT_ID"].astype(int),
            "visit_concept_id": 0,  # TODO: Athena (9201 Inpatient Visit)
            "visit_start_date": admit.dt.date,
            "visit_start_datetime": admit,
            "visit_end_date": disch.dt.date,
            "visit_end_datetime": disch,
            "visit_type_concept_id": 0,  # TODO: Athena
            "visit_source_value": df["ADMISSION_TYPE"].str.strip(),
        }
    )
    return result


# ---------------------------------------------------------------------------
# condition_occurrence  ←  DIAGNOSES_ICD.csv
# ---------------------------------------------------------------------------


def build_condition_occurrence_table(raw_dir: Path) -> pd.DataFrame:
    """Map MIMIC-III DIAGNOSES_ICD to OMOP condition_occurrence.

    ICD-9-CM codes go into condition_source_value; these are the codes used
    downstream by omop_mlops.features.comorbidity_flags() and the ADE
    weak-supervision E-code labeling functions.

    OMOP columns produced
    ---------------------
    condition_occurrence_id         : synthetic row number
    person_id                       : SUBJECT_ID
    condition_concept_id            : 0  TODO: Athena ICD9→SNOMED mapping
    condition_start_date            : joined from ADMISSIONS.ADMITTIME (date only)
    condition_start_datetime        : joined from ADMISSIONS.ADMITTIME
    condition_type_concept_id       : 0  TODO: Athena
    condition_source_value          : ICD9_CODE (raw string — used for lookups)
    visit_occurrence_id             : HADM_ID
    """
    dx = _load_csv(raw_dir, "DIAGNOSES_ICD.csv")
    adm = _load_csv(raw_dir, "ADMISSIONS.csv", parse_dates=["ADMITTIME"])[["HADM_ID", "ADMITTIME"]]

    df = dx.merge(adm, on="HADM_ID", how="left")
    admit_dt = pd.to_datetime(df["ADMITTIME"])

    result = pd.DataFrame(
        {
            "condition_occurrence_id": range(1, len(df) + 1),
            "person_id": df["SUBJECT_ID"].astype(int),
            "condition_concept_id": 0,  # TODO: map via Athena — scope boundary v1
            "condition_start_date": admit_dt.dt.date,
            "condition_start_datetime": admit_dt,
            "condition_type_concept_id": 0,  # TODO: Athena
            "condition_source_value": df["ICD9_CODE"].astype(str).str.strip(),
            "visit_occurrence_id": df["HADM_ID"].astype(int),
        }
    )
    return result


# ---------------------------------------------------------------------------
# drug_exposure  ←  PRESCRIPTIONS.csv
# ---------------------------------------------------------------------------


def build_drug_exposure_table(raw_dir: Path) -> pd.DataFrame:
    """Map MIMIC-III PRESCRIPTIONS to OMOP drug_exposure.

    NDC codes go into drug_source_value; RxNorm concept_ids are left as 0
    pending Athena lookup. The drug_exposure_start_datetime is used by
    omop_mlops.features.medication_counts() for the temporal lookback window.

    OMOP columns produced
    ---------------------
    drug_exposure_id            : synthetic row number
    person_id                   : SUBJECT_ID
    drug_concept_id             : 0  TODO: NDC→RxNorm via Athena
    drug_exposure_start_date    : STARTDATE date
    drug_exposure_start_datetime: STARTDATE datetime
    drug_exposure_end_date      : ENDDATE date
    drug_exposure_end_datetime  : ENDDATE datetime
    drug_type_concept_id        : 0  TODO: Athena
    drug_source_value           : NDC code (raw — used for ADE matching)
    drug_source_concept_id      : 0
    visit_occurrence_id         : HADM_ID
    dose_unit_source_value      : DOSE_UNIT_RX
    quantity                    : DOSE_VAL_RX (float, coerced)
    """
    df = _load_csv(
        raw_dir,
        "PRESCRIPTIONS.csv",
        parse_dates=["STARTDATE", "ENDDATE"],
        low_memory=False,
    )

    start_dt = pd.to_datetime(df["STARTDATE"], errors="coerce")
    end_dt = pd.to_datetime(df["ENDDATE"], errors="coerce")

    result = pd.DataFrame(
        {
            "drug_exposure_id": range(1, len(df) + 1),
            "person_id": df["SUBJECT_ID"].astype(int),
            "drug_concept_id": 0,  # TODO: NDC→RxNorm via Athena — scope boundary v1
            "drug_exposure_start_date": start_dt.dt.date,
            "drug_exposure_start_datetime": start_dt,
            "drug_exposure_end_date": end_dt.dt.date,
            "drug_exposure_end_datetime": end_dt,
            "drug_type_concept_id": 0,  # TODO: Athena
            "drug_source_value": df["NDC"].astype(str).str.strip(),
            "drug_source_concept_id": 0,
            "visit_occurrence_id": df["HADM_ID"].astype(int),
            "dose_unit_source_value": df["DOSE_UNIT_RX"].astype(str).str.strip(),
            "quantity": pd.to_numeric(df["DOSE_VAL_RX"], errors="coerce"),
        }
    )
    return result


# ---------------------------------------------------------------------------
# note  ←  NOTEEVENTS.csv   (RUN ON COLAB — do NOT call locally)
# ---------------------------------------------------------------------------


def build_note_table(raw_dir: Path) -> pd.DataFrame:
    """Map MIMIC-III NOTEEVENTS to OMOP note.

    RUN ON COLAB ONLY — NOTEEVENTS.csv is ~4 GB and is mounted from
    Google Drive.  This function is intentionally NOT called in main().

    Set raw_dir = Path('/content/drive/MyDrive/mimic3') on Colab before calling.

    OMOP columns produced
    ---------------------
    note_id             : synthetic row number
    person_id           : SUBJECT_ID
    note_date           : CHARTDATE date
    note_datetime       : CHARTDATE datetime
    note_type_concept_id: 0  TODO: Athena (44814645 = Inpatient note)
    note_class_concept_id: 0
    note_title          : CATEGORY
    note_text           : TEXT (the raw clinical note)
    visit_occurrence_id : HADM_ID
    note_source_value   : CATEGORY
    """
    df = _load_csv(raw_dir, "NOTEEVENTS.csv", parse_dates=["CHARTDATE"])
    chart_dt = pd.to_datetime(df["CHARTDATE"], errors="coerce")

    result = pd.DataFrame(
        {
            "note_id": range(1, len(df) + 1),
            "person_id": df["SUBJECT_ID"].astype(int),
            "note_date": chart_dt.dt.date,
            "note_datetime": chart_dt,
            "note_type_concept_id": 0,  # TODO: Athena
            "note_class_concept_id": 0,
            "note_title": df["CATEGORY"].astype(str).str.strip(),
            "note_text": df["TEXT"].astype(str),
            "visit_occurrence_id": df["HADM_ID"].fillna(0).astype(int),
            "note_source_value": df["CATEGORY"].astype(str).str.strip(),
        }
    )
    return result


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_local_etl(raw_dir: Path, processed_dir: Path) -> None:
    """Run the four small-table transforms; skip build_note_table (Colab only)."""
    transforms = [
        ("person", build_person_table, (raw_dir,)),
        ("visit_occurrence", build_visit_occurrence_table, (raw_dir,)),
        ("condition_occurrence", build_condition_occurrence_table, (raw_dir,)),
        ("drug_exposure", build_drug_exposure_table, (raw_dir,)),
    ]
    for name, fn, args in transforms:
        logger.info("Building %s ...", name)
        df = fn(*args)
        _write_parquet(df, processed_dir, name)
        logger.info("  → %d rows", len(df))

    logger.info("ETL complete. note table skipped (run build_note_table on Colab).")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="MIMIC-III → OMOP ETL (4 small tables)")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    raw_dir = Path(cfg["paths"]["raw_data_dir"])
    processed_dir = Path(cfg["paths"]["processed_data_dir"])

    run_local_etl(raw_dir, processed_dir)


if __name__ == "__main__":
    main()
