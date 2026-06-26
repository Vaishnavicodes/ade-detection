"""Weak-supervision ADE labeling pipeline — STUB.

ENVIRONMENT: Colab (Snorkel GPU optional; CPU works for LabelModel).

DESIGN OVERVIEW
---------------
Adverse Drug Event labels are not directly available in MIMIC-III.  We use a
three-source weak-supervision approach via Snorkel's LabelModel to generate
probabilistic labels without hand-labeling thousands of admissions:

  LF-1  ICD-9 E-code heuristic
        Any admission whose DIAGNOSES_ICD contains an E-code in the ADE range
        E930–E949 (drug-induced adverse events) is labelled POSITIVE.
        Source: omop_mlops.constants.ICD9_DRUG_ADE_ECODE_RANGE.

  LF-2  Clinical note mention matching
        Scan discharge summaries (OMOP note table) for trigger phrases such as
        "adverse drug reaction", "drug reaction", "drug-induced", etc.
        Requires the NOTEEVENTS table → run on Colab with Drive mount.
        TODO: build phrase list; consider MedSpaCy or simple regex first.

  LF-3  SIDER known drug–ADE pairs
        Match prescribed drugs (drug_source_value NDC / drug name) against the
        SIDER 4.1 database of known drug side effects.
        TODO: download SIDER, build NDC→SIDER drug_id lookup table.

These three noisy labeling functions are combined by a Snorkel LabelModel
that learns their accuracies and correlations, yielding per-admission
probabilistic labels (ade_prob ∈ [0,1]).  A threshold (e.g. 0.5) converts
to hard binary labels for downstream training.

OUTPUT SCHEMA
-------------
DataFrame with one row per (person_id, visit_occurrence_id, drug_source_value):

  person_id            int       OMOP person_id
  visit_occurrence_id  int       OMOP visit (HADM_ID)
  drug_source_value    str       NDC / drug name from drug_exposure
  ade_prob             float     Snorkel LabelModel output ∈ [0, 1]
  ade_label            int       hard label: 1=ADE, 0=no ADE, -1=abstain
  index_datetime       datetime  prediction timepoint (admit + prediction_timepoint_hours)

The index_datetime is the temporal boundary: features may only use data
BEFORE this timestamp to avoid label leakage (see config.temporal).

DEPENDENCIES (install on Colab):
  pip install snorkel>=0.9.9

REFERENCES:
  Ratner et al. (2017) — Snorkel: Rapid Training Data Creation with Weak Supervision
  Tatonetti et al. (2012) — SIDER drug side-effect database
"""

from __future__ import annotations

import pandas as pd

# ---------------------------------------------------------------------------
# Labeling functions (LFs)
# ---------------------------------------------------------------------------


def lf_icd9_ecode(condition_df: pd.DataFrame) -> pd.Series:
    """LF-1: Return +1 for admissions with a drug-ADE E-code, else -1 (abstain).

    Parameters
    ----------
    condition_df:
        OMOP condition_occurrence table with 'condition_source_value' (ICD-9).

    Returns
    -------
    pd.Series indexed by visit_occurrence_id; values in {-1, 0, 1}.

    TODO: implement — check condition_source_value against E930–E949 range
          using omop_mlops.constants.ICD9_DRUG_ADE_ECODE_RANGE.
    """
    raise NotImplementedError("TODO: implement lf_icd9_ecode")


def lf_note_mention(note_df: pd.DataFrame, trigger_phrases: list[str] | None = None) -> pd.Series:
    """LF-2: Return +1 for admissions whose notes mention ADE trigger phrases.

    Parameters
    ----------
    note_df:
        OMOP note table with 'note_text' and 'visit_occurrence_id'.
    trigger_phrases:
        Strings to search for; defaults to a built-in starter list.
        TODO: expand and validate trigger phrase list.

    Returns
    -------
    pd.Series indexed by visit_occurrence_id; values in {-1, 1}.

    NOTE: Requires NOTEEVENTS — run on Colab with Drive mount.
    TODO: implement — regex search over note_text; aggregate to visit level.
    """
    raise NotImplementedError("TODO: implement lf_note_mention — Colab only")


def lf_sider_match(drug_df: pd.DataFrame, sider_lookup: pd.DataFrame) -> pd.Series:
    """LF-3: Return +1 for visits where a prescribed drug has a known ADE in SIDER.

    Parameters
    ----------
    drug_df:
        OMOP drug_exposure with 'drug_source_value' (NDC) and 'visit_occurrence_id'.
    sider_lookup:
        DataFrame mapping drug identifiers to known ADEs from SIDER 4.1.
        TODO: build sider_lookup from SIDER download (meddra_all_se.tsv).

    Returns
    -------
    pd.Series indexed by visit_occurrence_id; values in {-1, 1}.

    TODO: implement — join drug_df to sider_lookup on drug identifier;
          aggregate to visit level.
    """
    raise NotImplementedError("TODO: implement lf_sider_match")


# ---------------------------------------------------------------------------
# LabelModel integration
# ---------------------------------------------------------------------------


def build_label_matrix(
    condition_df: pd.DataFrame,
    drug_df: pd.DataFrame,
    note_df: pd.DataFrame | None = None,
    sider_lookup: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Assemble the Snorkel label matrix L (n_samples × n_labeling_functions).

    TODO: call each LF above, align on visit_occurrence_id, stack into L.
    Return DataFrame with columns [lf_icd9_ecode, lf_note_mention, lf_sider_match].
    note_df and sider_lookup may be None — abstain (-1) for missing LFs.
    """
    raise NotImplementedError("TODO: implement build_label_matrix")


def fit_label_model(label_matrix: pd.DataFrame, cardinality: int = 2) -> object:
    """Fit a Snorkel LabelModel on the label matrix.

    Requires: snorkel>=0.9.9 (install on Colab).

    TODO:
      from snorkel.labeling.model import LabelModel
      lm = LabelModel(cardinality=cardinality, verbose=True)
      lm.fit(L_train=label_matrix.values, n_epochs=500, lr=0.001, seed=42)
      return lm
    """
    raise NotImplementedError("TODO: implement fit_label_model — requires Snorkel on Colab")


def extract_ade_labels(
    condition_df: pd.DataFrame,
    drug_df: pd.DataFrame,
    visit_df: pd.DataFrame,
    note_df: pd.DataFrame | None = None,
    sider_lookup: pd.DataFrame | None = None,
    prediction_timepoint_hours: int = 24,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Full pipeline: LFs → LabelModel → probabilistic ADE labels.

    Parameters
    ----------
    condition_df:
        OMOP condition_occurrence.
    drug_df:
        OMOP drug_exposure.
    visit_df:
        OMOP visit_occurrence (needed for index_datetime = admit + hours).
    note_df:
        OMOP note table (Colab only; pass None to skip LF-2).
    sider_lookup:
        SIDER drug lookup (pass None to skip LF-3).
    prediction_timepoint_hours:
        Hours post-admission that defines the prediction window boundary.
    threshold:
        Probability cutoff for hard ade_label assignment.

    Returns
    -------
    pd.DataFrame with schema:
        person_id, visit_occurrence_id, drug_source_value,
        ade_prob, ade_label, index_datetime

    TODO: implement by calling build_label_matrix → fit_label_model →
          lm.predict_proba → threshold → attach index_datetime.
    """
    raise NotImplementedError("TODO: implement extract_ade_labels")
