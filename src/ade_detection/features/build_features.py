"""ADE feature matrix construction — STUB.

ENVIRONMENT: local (small feature set) or Colab (with note embeddings).

PURPOSE
-------
Build the per-admission feature matrix used by the ADE risk classifier.
Delegates all OMOP feature extraction to omop_mlops.features so that the
temporal validity contract is enforced consistently across both app repos.

TEMPORAL VALIDITY (Hero #2)
---------------------------
All features MUST be computed using only data recorded BEFORE the prediction
timepoint.  This is enforced by passing index_dates and lookback_days to every
omop_mlops call.  The prediction timepoint per admission is:

    index_datetime = visit_start_datetime + prediction_timepoint_hours

Any record (drug exposure, condition, measurement) with a timestamp >= index_datetime
is excluded.  omop_mlops.features functions accept index_dates as a pd.Series
keyed by person_id.

INTENDED INPUTS
---------------
  person_df           : OMOP person (parquet, from build_omop_tables)
  condition_df        : OMOP condition_occurrence
  drug_df             : OMOP drug_exposure
  measurement_df      : OMOP measurement (optional for v1)
  visit_df            : OMOP visit_occurrence (for index_datetime)
  ade_labels_df       : output of extract_ade_labels (for the target column)
  config              : dict from config/config.yaml

INTENDED OUTPUT
---------------
  pd.DataFrame with:
    person_id, visit_occurrence_id,
    comorbidity flags (from omop_mlops.features.comorbidity_flags),
    drug_count         (from omop_mlops.features.medication_counts),
    [optional measurement aggs],
    ade_label          (target — 0/1)

TODO
----
  1. Load parquet tables from processed_data_dir.
  2. Compute index_datetime per admission from visit_start_datetime + config hours.
  3. Call omop_mlops.features.make_feature_matrix(index_dates=..., lookback_days=...).
  4. Merge ade_labels_df on (person_id, visit_occurrence_id).
  5. Write feature matrix to processed_data_dir/features.parquet.
"""

from __future__ import annotations

import pandas as pd

# omop_mlops.features enforces temporal validity via index_date + lookback_days.
# Always pass these — never call without them in a supervised learning context.
from omop_mlops.features import make_feature_matrix  # noqa: F401


def compute_index_dates(
    visit_df: pd.DataFrame,
    prediction_timepoint_hours: int = 24,
) -> pd.Series:
    """Return a Series of prediction timepoints indexed by person_id.

    index_datetime = visit_start_datetime + prediction_timepoint_hours.

    TODO: handle multiple admissions per person (take earliest, last, or all
    depending on study design). For v1, take the first admission per person.
    """
    raise NotImplementedError("TODO: implement compute_index_dates")


def build_feature_matrix(
    person_df: pd.DataFrame,
    condition_df: pd.DataFrame,
    drug_df: pd.DataFrame,
    visit_df: pd.DataFrame,
    ade_labels_df: pd.DataFrame,
    measurement_df: pd.DataFrame | None = None,
    prediction_timepoint_hours: int = 24,
    lookback_days: int = 365,
) -> pd.DataFrame:
    """Join OMOP tables into a temporally valid feature matrix with ADE labels.

    TODO:
      index_dates = compute_index_dates(visit_df, prediction_timepoint_hours)
      features = make_feature_matrix(
          person_df, condition_df, drug_df, measurement_df,
          index_dates=index_dates, lookback_days=lookback_days
      )
      return features.merge(ade_labels_df[["person_id", "visit_occurrence_id", "ade_label"]],
                            on="person_id", how="left")
    """
    raise NotImplementedError("TODO: implement build_feature_matrix")


def main(config_path: str = "config/config.yaml") -> None:
    """CLI entrypoint: load parquets, build features, write to processed_dir.

    TODO: implement — load config, read parquets, call build_feature_matrix,
    write output.
    """
    raise NotImplementedError("TODO: implement main")


if __name__ == "__main__":
    main()
