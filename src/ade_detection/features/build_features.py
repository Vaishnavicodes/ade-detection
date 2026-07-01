"""Leakage-safe temporal feature pipeline for ADE risk prediction.

Index date: admission_start + prediction_timepoint_hours (default 24 h).
All features use ONLY data knowable at that point:
  - Demographics: static (age, gender, race, ethnicity)
  - First-24h medications: drugs given during [admission_start, index_date]
  - Prior history: conditions/drugs from PREVIOUS admissions only,
    within lookback_days before the index date

Anti-leakage hard exclusions applied to every feature group:
  - ICD-9 codes that define the ADE label (E930-E949, 995.2x, 960-979)
  - SIDER curated ADE drug list (from ade_mappings.get_all_ade_drugs())
  - Current-admission diagnosis codes are never used as prior-history features
  - No drug administered after the 24-h index is counted in first-24h features

Why the omop_mlops high-level functions are not used for prior history:
  comorbidity_flags() and medication_counts() operate per person_id with a single
  index_date each.  MIMIC patients can have multiple admissions; each admission needs
  its own lookback window anchored at its own index_date.  We therefore implement the
  cross-join manually and call CHARLSON_ICD9_GROUPS directly from omop_mlops.constants.

Usage::

    python -m ade_detection.features.build_features
    python -m ade_detection.features.build_features --config config/config.yaml

Prerequisites: visit_occurrence.parquet, person.parquet, condition_occurrence.parquet,
drug_exposure.parquet, ade_labels.parquet must all exist in processed_data_dir.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import yaml
from omop_mlops.constants import CHARLSON_ICD9_GROUPS

from ade_detection.labeling.ade_mappings import get_all_ade_drugs, get_patterns
from ade_detection.labeling.drug_normalization import normalize_drug_name
from ade_detection.labeling.labeling_functions import (
    _is_ade_diagnosis_code,
    _is_ecode_ade,
    normalize_icd9,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level anti-leakage sets (computed once at import time)
# ---------------------------------------------------------------------------

# SIDER curated ADE drugs — excluded from all drug-derived features.
# Using any of these drugs as a feature would leak the label into the inputs
# because lf_sider_curated labels ADE when a SIDER drug co-occurs with its
# paired ICD-9 code.
_ADE_DRUGS: frozenset[str] = frozenset(get_all_ade_drugs())

# Charlson ICD-9 group prefixes normalised to MIMIC dotless format (e.g.
# "250.0" -> "2500") so startswith() comparisons work against MIMIC codes.
_CHARLSON_DOTLESS: dict[str, list[str]] = {
    name: [normalize_icd9(p) for p in prefixes] for name, prefixes in CHARLSON_ICD9_GROUPS.items()
}
_CHARLSON_GROUPS: list[str] = list(CHARLSON_ICD9_GROUPS.keys())


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _is_label_defining(code: str) -> bool:
    """True if this ICD-9 code is part of the ADE label definition.

    Covers E930–E949 (drug-ADE E-codes), 995.2x (adverse drug effect),
    and 960–979 (drug poisoning).  These are excluded to prevent label
    leakage into features.
    """
    s = str(code).strip()
    if not s or s.lower() in ("nan", "none"):
        return False
    return _is_ecode_ade(s) or _is_ade_diagnosis_code(s)


def _is_ade_drug(raw_name: str) -> bool:
    """True if this drug normalises to a SIDER ADE drug.

    Excluded to prevent label leakage into features.
    """
    normalized = normalize_drug_name(str(raw_name))
    return any(ade in normalized for ade in _ADE_DRUGS)


def _charlson_flags_for_visits(
    prior_cond: pd.DataFrame,
    all_vid_arr,
) -> pd.DataFrame:
    """Per-visit Charlson comorbidity flags from pre-filtered prior conditions.

    Parameters
    ----------
    prior_cond:
        Filtered condition rows.  Must contain 'target_vid' (visit_occurrence_id
        of the current/target admission) and 'condition_source_value'.
        Rows must already be limited to PRIOR admissions and the lookback window,
        with ADE-defining codes removed.
    all_vid_arr:
        Numpy array of all visit_occurrence_ids (for a complete output index).

    Returns
    -------
    pd.DataFrame
        Columns: visit_occurrence_id + one int (0/1) per Charlson group.
    """
    result = pd.DataFrame({"visit_occurrence_id": all_vid_arr})

    if prior_cond.empty:
        for gname in _CHARLSON_GROUPS:
            result[f"prior_comorb_{gname}"] = 0
        return result

    # Normalise all codes once to MIMIC dotless format
    norm_codes = prior_cond["condition_source_value"].apply(lambda c: normalize_icd9(str(c)))

    for gname, prefixes in _CHARLSON_DOTLESS.items():
        matches = norm_codes.apply(lambda c: any(c.startswith(p) for p in prefixes))
        matching_vids = prior_cond.loc[matches, "target_vid"].unique()
        result[f"prior_comorb_{gname}"] = (
            result["visit_occurrence_id"].isin(matching_vids).astype(int)
        )

    return result


# ---------------------------------------------------------------------------
# Main feature-building function
# ---------------------------------------------------------------------------


def build_features(processed_dir: str | Path, config: dict) -> pd.DataFrame:
    """Build a per-admission leakage-safe feature matrix.

    Parameters
    ----------
    processed_dir:
        Directory containing the OMOP parquets (visit_occurrence, person,
        condition_occurrence, drug_exposure) and ade_labels.parquet.
    config:
        Loaded config.yaml dict.  Uses:
          temporal.prediction_timepoint_hours  (default 24)
          temporal.lookback_days               (default 365)

    Returns
    -------
    pd.DataFrame
        One row per admission, keyed by visit_occurrence_id.  Columns:
          visit_occurrence_id, person_id, visit_start_datetime, index_datetime,
          age_at_admission, gender_concept_id, race_concept_id, ethnicity_concept_id,
          n_drugs_first24h, n_distinct_drugs_first24h,
          prior_comorb_{charlson_group} (one per CHARLSON_ICD9_GROUPS entry),
          n_prior_conditions, n_prior_drug_exposures,
          ade_prob, ade_label
    """
    processed_dir = Path(processed_dir)
    pred_hours: int = config["temporal"]["prediction_timepoint_hours"]
    lookback_days: int = config["temporal"]["lookback_days"]
    lookback_td = pd.Timedelta(days=lookback_days)

    # ----------------------------------------------------------------
    # Load OMOP tables
    # ----------------------------------------------------------------
    logger.info("Loading OMOP parquets from %s", processed_dir)
    visit_df = pd.read_parquet(processed_dir / "visit_occurrence.parquet")
    person_df = pd.read_parquet(processed_dir / "person.parquet")
    condition_df = pd.read_parquet(processed_dir / "condition_occurrence.parquet")
    drug_df = pd.read_parquet(processed_dir / "drug_exposure.parquet")
    labels_df = pd.read_parquet(processed_dir / "ade_labels.parquet")

    visit_df["visit_start_datetime"] = pd.to_datetime(visit_df["visit_start_datetime"])
    visit_df["visit_end_datetime"] = pd.to_datetime(visit_df["visit_end_datetime"])
    condition_df["condition_start_datetime"] = pd.to_datetime(
        condition_df["condition_start_datetime"]
    )
    drug_df["drug_exposure_start_datetime"] = pd.to_datetime(
        drug_df["drug_exposure_start_datetime"]
    )

    # Index date: 24 h (or configured hours) after admission start
    visit_df["index_datetime"] = visit_df["visit_start_datetime"] + pd.Timedelta(hours=pred_hours)

    all_vid_arr = visit_df["visit_occurrence_id"].values
    logger.info(
        "Loaded: %d admissions, %d conditions, %d drug exposures",
        len(visit_df),
        len(condition_df),
        len(drug_df),
    )

    # Prefer human-readable drug name for ADE drug matching
    drug_name_col = (
        "drug_name_source_value"
        if "drug_name_source_value" in drug_df.columns
        else "drug_source_value"
    )

    # ----------------------------------------------------------------
    # DEMOGRAPHICS (static; no temporal filter needed)
    # ----------------------------------------------------------------
    _demo_want = [
        "person_id",
        "year_of_birth",
        "gender_concept_id",
        "race_concept_id",
        "ethnicity_concept_id",
    ]
    _demo_avail = [c for c in _demo_want if c in person_df.columns]
    base = visit_df[
        ["visit_occurrence_id", "person_id", "visit_start_datetime", "index_datetime"]
    ].merge(person_df[_demo_avail], on="person_id", how="left")

    if "year_of_birth" in base.columns:
        base["age_at_admission"] = base["visit_start_datetime"].dt.year - base["year_of_birth"]
        base = base.drop(columns=["year_of_birth"])
        # MIMIC-III shifts birth years of patients aged 89+ forward by a random offset
        # to satisfy HIPAA de-identification, producing computed ages up to ~300.
        # Standard handling: clip to 90, meaning "90 or older".
        n_clipped = int((base["age_at_admission"] > 89).sum())
        if n_clipped:
            logger.info(
                "Clipped %d age_at_admission values >89 to 90 (MIMIC 89+ de-identification shift)",
                n_clipped,
            )
        base["age_at_admission"] = base["age_at_admission"].clip(upper=90)
    else:
        base["age_at_admission"] = 0

    for col in ["age_at_admission", "gender_concept_id", "race_concept_id", "ethnicity_concept_id"]:
        if col not in base.columns:
            base[col] = 0

    # ----------------------------------------------------------------
    # FIRST-24H MEDICATIONS
    # Only drugs given in [visit_start, index_date].
    # SIDER ADE drugs excluded to prevent label leakage into features.
    # ----------------------------------------------------------------
    drugs_with_window = drug_df.merge(
        visit_df[["visit_occurrence_id", "visit_start_datetime", "index_datetime"]],
        on="visit_occurrence_id",
        how="inner",
    )
    in_window = (
        drugs_with_window["drug_exposure_start_datetime"]
        >= drugs_with_window["visit_start_datetime"]
    ) & (drugs_with_window["drug_exposure_start_datetime"] <= drugs_with_window["index_datetime"])
    # excluded to prevent label leakage into features
    not_ade_drug = ~drugs_with_window[drug_name_col].apply(_is_ade_drug)
    first24h = drugs_with_window[in_window & not_ade_drug]

    if first24h.empty:
        drug_agg = pd.DataFrame(
            {
                "visit_occurrence_id": all_vid_arr,
                "n_drugs_first24h": 0,
                "n_distinct_drugs_first24h": 0,
            }
        )
    else:
        _agg = first24h.groupby("visit_occurrence_id").agg(
            n_drugs_first24h=(drug_name_col, "count"),
            n_distinct_drugs_first24h=(drug_name_col, "nunique"),
        )
        drug_agg = pd.DataFrame(
            {
                "visit_occurrence_id": all_vid_arr,
                "n_drugs_first24h": _agg["n_drugs_first24h"]
                .reindex(all_vid_arr, fill_value=0)
                .values,
                "n_distinct_drugs_first24h": _agg["n_distinct_drugs_first24h"]
                .reindex(all_vid_arr, fill_value=0)
                .values,
            }
        )

    # ----------------------------------------------------------------
    # PRIOR HISTORY
    # Conditions/drugs from PREVIOUS admissions only (visit ended before
    # current admission started), within lookback_days of the index date.
    # ADE-defining codes and SIDER drugs excluded (anti-leakage).
    #
    # Approach: cross-join each condition/drug with all target admissions
    # for the same patient, then filter to prior-only rows.  This correctly
    # handles patients with multiple admissions — each admission gets its own
    # lookback window and prior-history set.
    # ----------------------------------------------------------------

    # Attach source-visit end time to each condition row
    cond_timed = condition_df.merge(
        visit_df[["visit_occurrence_id", "visit_end_datetime"]].rename(
            columns={
                "visit_occurrence_id": "_cvid",
                "visit_end_datetime": "cond_visit_end",
            }
        ),
        left_on="visit_occurrence_id",
        right_on="_cvid",
        how="left",
    ).drop(columns=["_cvid"])

    # Reference table for target admissions (one row per admission)
    target_ref = visit_df[
        ["visit_occurrence_id", "person_id", "visit_start_datetime", "index_datetime"]
    ].rename(
        columns={
            "visit_occurrence_id": "target_vid",
            "visit_start_datetime": "target_visit_start",
            "index_datetime": "target_index",
        }
    )

    # Cross-join on person_id: each condition appears once per target admission
    # belonging to the same patient
    prior_cond = cond_timed.merge(target_ref, on="person_id")

    # Keep only rows where:
    #   (a) the condition's source admission ended BEFORE this target admission started
    #   (b) the condition started within lookback_days of this target's index date
    #   (c) the condition code is not a label-defining ADE code (anti-leakage)
    prior_cond = prior_cond[
        (prior_cond["cond_visit_end"] < prior_cond["target_visit_start"])  # prior only
        & (prior_cond["condition_start_datetime"] >= prior_cond["target_index"] - lookback_td)
        # excluded to prevent label leakage into features
        & ~prior_cond["condition_source_value"].apply(_is_label_defining)
    ].copy()

    # Count of qualifying prior conditions per target admission
    prior_cond_counts = prior_cond.groupby("target_vid").size().rename("n_prior_conditions")

    # Charlson comorbidity binary flags (visit-level, via pre-filtered cross-join)
    comorb_df = _charlson_flags_for_visits(prior_cond, all_vid_arr)

    # Prior drug exposures: same cross-join pattern
    drug_timed = drug_df.merge(
        visit_df[["visit_occurrence_id", "visit_end_datetime"]].rename(
            columns={
                "visit_occurrence_id": "_dvid",
                "visit_end_datetime": "drug_visit_end",
            }
        ),
        left_on="visit_occurrence_id",
        right_on="_dvid",
        how="left",
    ).drop(columns=["_dvid"])

    prior_drug = drug_timed.merge(target_ref, on="person_id")
    prior_drug = prior_drug[
        (prior_drug["drug_visit_end"] < prior_drug["target_visit_start"])
        & (prior_drug["drug_exposure_start_datetime"] >= prior_drug["target_index"] - lookback_td)
        # excluded to prevent label leakage into features
        & ~prior_drug[drug_name_col].apply(_is_ade_drug)
    ]

    prior_drug_counts = prior_drug.groupby("target_vid").size().rename("n_prior_drug_exposures")

    # ----------------------------------------------------------------
    # JOIN ALL FEATURE GROUPS
    # ----------------------------------------------------------------
    n_prior_cond_df = pd.DataFrame(
        {
            "visit_occurrence_id": all_vid_arr,
            "n_prior_conditions": prior_cond_counts.reindex(all_vid_arr, fill_value=0).values,
        }
    )
    n_prior_drug_df = pd.DataFrame(
        {
            "visit_occurrence_id": all_vid_arr,
            "n_prior_drug_exposures": prior_drug_counts.reindex(all_vid_arr, fill_value=0).values,
        }
    )

    features = (
        base.merge(drug_agg, on="visit_occurrence_id", how="left")
        .merge(comorb_df, on="visit_occurrence_id", how="left")
        .merge(n_prior_cond_df, on="visit_occurrence_id", how="left")
        .merge(n_prior_drug_df, on="visit_occurrence_id", how="left")
        .merge(
            labels_df[["visit_occurrence_id", "ade_prob", "ade_label"]],
            on="visit_occurrence_id",
            how="left",
        )
    )

    # Coerce numeric feature columns
    int_fill_cols = [
        "n_drugs_first24h",
        "n_distinct_drugs_first24h",
        "n_prior_conditions",
        "n_prior_drug_exposures",
    ] + [c for c in features.columns if c.startswith("prior_comorb_")]
    for col in int_fill_cols:
        if col in features.columns:
            features[col] = features[col].fillna(0).astype(int)

    for col in ["age_at_admission", "gender_concept_id", "race_concept_id", "ethnicity_concept_id"]:
        if col in features.columns:
            features[col] = features[col].fillna(0).astype(int)

    logger.info("Feature matrix: %d rows × %d columns", features.shape[0], features.shape[1])
    return features


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="Build leakage-safe ADE feature matrix from OMOP parquets"
    )
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    processed_dir = Path(cfg["paths"]["processed_data_dir"])
    features = build_features(processed_dir, cfg)

    # ----------------------------------------------------------------
    # Leakage self-check: no ADE-defining code or drug string should
    # appear in any feature column name.
    # ----------------------------------------------------------------
    excluded_strings: set[str] = set()
    for p in get_patterns():
        for drug in p["drugs"]:
            excluded_strings.add(drug.lower())
        for prefix in p["icd9_prefixes"]:
            excluded_strings.add(normalize_icd9(prefix).lower())
    for n in range(930, 950):
        excluded_strings.add(f"e{n}")
    excluded_strings.add("9952")

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
        for exc in excluded_strings:
            if exc and exc in col_lower:
                violations.append(f"  '{exc}' found in column '{col}'")

    if violations:
        raise AssertionError(
            "LEAKAGE CHECK FAILED — label-defining strings in feature columns:\n"
            + "\n".join(violations)
        )

    n_features = len(feature_cols)
    print(f"Feature matrix: {features.shape[0]:,} rows × {features.shape[1]} columns")
    print(f"Feature columns ({n_features}): {feature_cols}")
    print("LEAKAGE CHECK PASSED")

    out_path = processed_dir / "features.parquet"
    features.to_parquet(out_path, index=False)
    print(f"\nWrote {len(features):,} rows → {out_path}")


if __name__ == "__main__":
    main()
