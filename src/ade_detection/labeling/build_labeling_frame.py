"""Build the per-admission labeling frame from OMOP parquets.

Reads visit_occurrence, condition_occurrence, and drug_exposure from the
processed parquet directory and returns one row per visit_occurrence with
aggregated list columns (icd9_codes, drug_names) ready for Snorkel's
PandasLFApplier.

Scales to MIMIC-III full (~58 k visits, ~650 k conditions, ~4 M drug rows)
using groupby+agg(list) — no Python-level loops over rows.

Usage::

    python -m ade_detection.labeling.build_labeling_frame
    python -m ade_detection.labeling.build_labeling_frame --config config/config.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def build_labeling_frame(processed_dir: str | Path) -> pd.DataFrame:
    """Load OMOP parquets and build a per-admission DataFrame for Snorkel LFs.

    Parameters
    ----------
    processed_dir:
        Directory containing visit_occurrence.parquet, condition_occurrence.parquet,
        and drug_exposure.parquet (output of build_omop_tables.py).

    Returns
    -------
    pd.DataFrame
        One row per admission. Columns:
          visit_occurrence_id  int
          person_id            int
          icd9_codes           list[str]  – condition_source_value per visit
          drug_names           list[str]  – drug name per visit
          length_of_stay_days  float      – (visit_end - visit_start) in days
          discharge_disposition str       – ADMISSIONS.DISCHARGE_LOCATION
          admission_type       str        – ADMISSIONS.ADMISSION_TYPE
          visit_start_datetime datetime
    """
    processed_dir = Path(processed_dir)

    logger.info("Loading OMOP parquets from %s", processed_dir)
    visit_df = pd.read_parquet(processed_dir / "visit_occurrence.parquet")
    condition_df = pd.read_parquet(processed_dir / "condition_occurrence.parquet")
    drug_df = pd.read_parquet(processed_dir / "drug_exposure.parquet")

    logger.info(
        "Loaded: %d visits, %d conditions, %d drug exposures",
        len(visit_df),
        len(condition_df),
        len(drug_df),
    )

    # --- ICD-9 codes per visit ---
    condition_df = condition_df[condition_df["condition_source_value"].notna()]
    icd9_per_visit = (
        condition_df.groupby("visit_occurrence_id")["condition_source_value"]
        .agg(list)
        .rename("icd9_codes")
    )

    # --- Drug names per visit ---
    # drug_name_source_value holds PRESCRIPTIONS.DRUG (free-text name, e.g.
    # "Vancomycin HCl 1250mg IVPB"), which normalize_drug_name() can parse.
    # Fall back to drug_source_value (NDC) only for parquets from an older ETL
    # run before this column was added; lf_sider_curated will have ~0% coverage
    # in that case since NDC strings don't match generic drug names.
    if "drug_name_source_value" in drug_df.columns:
        drug_col = "drug_name_source_value"
    else:
        drug_col = "drug_source_value"

    drug_df = drug_df[drug_df[drug_col].notna()]
    drug_per_visit = drug_df.groupby("visit_occurrence_id")[drug_col].agg(list).rename("drug_names")

    # --- Length-of-stay (days) ---
    visit_df["visit_start_datetime"] = pd.to_datetime(visit_df["visit_start_datetime"])
    visit_df["visit_end_datetime"] = pd.to_datetime(visit_df["visit_end_datetime"])
    visit_df["length_of_stay_days"] = (
        (visit_df["visit_end_datetime"] - visit_df["visit_start_datetime"])
        .dt.total_seconds()
        .div(86400)
        .clip(lower=0)  # guard against reversed admission/discharge timestamps
    )

    # --- Discharge disposition ---
    # Prefer discharge_location_source_value (ADMISSIONS.DISCHARGE_LOCATION mapped
    # by build_visit_occurrence_table). Fall back to discharge_disposition for any
    # parquet written before that column was added, then to "unknown".
    if "discharge_location_source_value" in visit_df.columns:
        discharge_vals = visit_df["discharge_location_source_value"].fillna("UNKNOWN").values
    elif "discharge_disposition" in visit_df.columns:
        discharge_vals = visit_df["discharge_disposition"].values
    else:
        discharge_vals = "unknown"  # scalar — broadcast to all rows by .assign()

    # --- Admission type ---
    # admission_type_source_value is an explicit column added in ETL v2; fall back
    # to visit_source_value which holds ADMISSION_TYPE in v1 ETL parquets.
    if "admission_type_source_value" in visit_df.columns:
        admission_type_vals = visit_df["admission_type_source_value"].fillna("UNKNOWN").values
    elif "visit_source_value" in visit_df.columns:
        admission_type_vals = visit_df["visit_source_value"].fillna("UNKNOWN").values
    else:
        admission_type_vals = "UNKNOWN"

    # --- Join onto the visit spine (left join keeps all visits) ---
    frame = (
        visit_df[
            [
                "visit_occurrence_id",
                "person_id",
                "visit_start_datetime",
                "length_of_stay_days",
            ]
        ]
        .assign(discharge_disposition=discharge_vals, admission_type=admission_type_vals)
        .set_index("visit_occurrence_id")
        .join(icd9_per_visit, how="left")
        .join(drug_per_visit, how="left")
        .reset_index()
    )

    # Visits with no conditions or no drugs should have empty lists, not NaN.
    frame["icd9_codes"] = [v if isinstance(v, list) else [] for v in frame["icd9_codes"]]
    frame["drug_names"] = [v if isinstance(v, list) else [] for v in frame["drug_names"]]

    logger.info("Labeling frame: %d rows (one per admission)", len(frame))

    return frame[
        [
            "visit_occurrence_id",
            "person_id",
            "icd9_codes",
            "drug_names",
            "length_of_stay_days",
            "discharge_disposition",
            "admission_type",
            "visit_start_datetime",
        ]
    ]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Build Snorkel labeling frame from OMOP parquets")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    processed_dir = Path(cfg["paths"]["processed_data_dir"])
    frame = build_labeling_frame(processed_dir)

    out_path = processed_dir / "labeling_frame.parquet"
    frame.to_parquet(out_path, index=False)
    print(f"Wrote {len(frame):,} rows -> {out_path}")


if __name__ == "__main__":
    main()
